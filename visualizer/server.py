#!/usr/bin/env python3
"""Visualizer for the world-encoder training data and metrics.

Serves a small website (see static/) with two views:
  - Data:    browse $RH20T/frames/<cfg>/<scene>/<cam>/<stream>/<ts>.jpg and play them back.
  - Metrics: renders every *.json in --metrics-dir (schema documented in README.md).

Stdlib only — no pip installs. Run:
    python visualizer/server.py --port 8000
    # data root resolves from --data-root, else $RH20T, else /mnt/nas/data/RH20T
"""
import argparse
import json
import math
import os
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def safe_parts(*parts):
    """True iff every path component is a plain name (no traversal, no separators)."""
    return all(SAFE_COMPONENT.match(p) and p not in (".", "..") for p in parts)


# the 28-dim SceneState.state() vector, sliced into named signal groups
STATE_GROUPS = [
    ("joints — sin", 0, ["j1", "j2", "j3", "j4", "j5", "j6"]),
    ("joints — cos", 6, ["j1", "j2", "j3", "j4", "j5", "j6"]),
    ("tcp position (symlog)", 12, ["x", "y", "z"]),
    ("tcp rotation (6D)", 15, ["r1", "r2", "r3", "r4", "r5", "r6"]),
    ("force/torque zeroed (symlog)", 21, ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]),
    ("gripper width (symlog)", 27, ["width"]),
]

_scene_state_cache = {}  # (cfg, scene) -> SceneState, capped small
_scene_state_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "wae-visualizer/1.0"

    # set by serve() on the class
    frames_root: Path = None
    metrics_dir: Path = None
    raw_root: Path = None

    def log_message(self, fmt, *args):  # quieter default log
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    # ---- helpers -------------------------------------------------------
    def _send(self, status, body, ctype, cache="no-cache"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(status, json.dumps(obj).encode(), CONTENT_TYPES[".json"])

    def _error(self, status, msg):
        self._json({"error": msg}, status=status)

    def _file(self, path: Path, cache="no-cache"):
        ctype = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self._send(HTTPStatus.OK, path.read_bytes(), ctype, cache=cache)

    # ---- routes --------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if route in ("/", "/index.html"):
                return self._file(STATIC_DIR / "index.html")
            if route.startswith("/static/"):
                return self.static_file(route[len("/static/"):])
            if route == "/api/summary":
                return self._json(self.api_summary())
            if route == "/api/scenes":
                return self._json(self.api_scenes(qs.get("cfg", "")))
            if route == "/api/scene":
                return self._json(self.api_scene(qs.get("cfg", ""), qs.get("name", "")))
            if route == "/api/frames":
                return self._json(self.api_frames(qs.get("cfg", ""), qs.get("scene", ""),
                                                  qs.get("cam", ""), qs.get("stream", "color")))
            if route == "/api/state":
                return self._json(self.api_state(qs.get("cfg", ""), qs.get("scene", ""),
                                                 qs.get("cam", ""), qs.get("stream", "color")))
            if route == "/api/metrics":
                return self._json(self.api_metrics())
            if route.startswith("/frames/"):
                return self.frame_file(route[len("/frames/"):])
            return self._error(HTTPStatus.NOT_FOUND, "no such route")
        except ValueError as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))
        except FileNotFoundError:
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def static_file(self, rel):
        parts = [p for p in rel.split("/") if p]
        if not parts or not safe_parts(*parts):
            raise ValueError("bad static path")
        path = STATIC_DIR.joinpath(*parts)
        if not path.is_file():
            raise FileNotFoundError(rel)
        return self._file(path)

    def api_summary(self):
        cfgs = []
        if self.frames_root.is_dir():
            for d in sorted(self.frames_root.iterdir()):
                if d.is_dir():
                    n = sum(1 for s in d.iterdir() if s.is_dir())
                    cfgs.append({"name": d.name, "scenes": n})
        return {
            "frames_root": str(self.frames_root),
            "metrics_dir": str(self.metrics_dir),
            "cfgs": cfgs,
        }

    def _cfg_dir(self, cfg):
        if not safe_parts(cfg):
            raise ValueError("bad cfg")
        d = self.frames_root / cfg
        if not d.is_dir():
            raise FileNotFoundError(cfg)
        return d

    def api_scenes(self, cfg):
        d = self._cfg_dir(cfg)
        scenes = sorted(s.name for s in d.iterdir() if s.is_dir())
        return {"cfg": cfg, "scenes": scenes}

    def api_scene(self, cfg, name):
        if not safe_parts(name):
            raise ValueError("bad scene")
        scene_dir = self._cfg_dir(cfg) / name
        if not scene_dir.is_dir():
            raise FileNotFoundError(name)
        cams = []
        for cam in sorted(scene_dir.iterdir()):
            if not cam.is_dir():
                continue
            streams = {}
            for stream in sorted(cam.iterdir()):
                if stream.is_dir():
                    streams[stream.name] = sum(1 for f in stream.iterdir() if f.suffix == ".jpg")
            cams.append({"name": cam.name, "streams": streams})
        return {"cfg": cfg, "scene": name, "cams": cams}

    def api_frames(self, cfg, scene, cam, stream):
        if not safe_parts(scene, cam, stream):
            raise ValueError("bad path")
        d = self._cfg_dir(cfg) / scene / cam / stream
        if not d.is_dir():
            raise FileNotFoundError(stream)
        ts = sorted(int(f.stem) for f in d.iterdir() if f.suffix == ".jpg" and f.stem.isdigit())
        return {"timestamps": ts}

    def _scene_state(self, cfg, scene):
        """SceneState for a scene, via the training code (world_tokenizer.state)."""
        key = (cfg, scene)
        with _scene_state_lock:
            if key in _scene_state_cache:
                return _scene_state_cache[key]
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from world_tokenizer.state import SceneState  # needs numpy + scipy
        # frames use "cfg3", raw uses "RH20T_cfg3" — accept either layout
        for cand in (self.raw_root / cfg / scene, self.raw_root / f"RH20T_{cfg}" / scene):
            if cand.is_dir():
                st = SceneState(str(cand))
                with _scene_state_lock:
                    if len(_scene_state_cache) >= 4:
                        _scene_state_cache.pop(next(iter(_scene_state_cache)))
                    _scene_state_cache[key] = st
                return st
        raise FileNotFoundError(f"no raw scene dir for {scene} under {self.raw_root}")

    def api_state(self, cfg, scene, cam, stream):
        """The dataloader's 28-dim state vector, evaluated at this camera's frame times."""
        ts = self.api_frames(cfg, scene, cam, stream)["timestamps"]
        if not ts:
            return {"t": [], "groups": []}
        try:
            st = self._scene_state(cfg, scene)
        except ImportError as e:
            return {"error": f"state preprocessing needs numpy+scipy on this python: {e}"}
        except FileNotFoundError as e:
            return {"error": str(e)}
        import numpy as np
        vecs = np.stack([st.state(t) for t in ts]).astype(float)  # (N, 28)
        groups = []
        for title, start, names in STATE_GROUPS:
            series = {}
            for k, name in enumerate(names):
                col = vecs[:, start + k]
                series[name] = [v if math.isfinite(v) else None for v in col.tolist()]
            groups.append({"title": title, "series": series})
        return {"serial": st.serial, "t": ts, "groups": groups}

    def frame_file(self, rel):
        # /frames/<cfg>/<scene>/<cam>/<stream>/<ts>.jpg
        parts = [p for p in rel.split("/") if p]
        if len(parts) != 5 or not safe_parts(*parts):
            raise ValueError("bad frame path")
        path = self.frames_root.joinpath(*parts)
        if not path.is_file():
            raise FileNotFoundError(rel)
        # timestamped frames never change -> cache hard
        return self._file(path, cache="public, max-age=31536000, immutable")

    def api_metrics(self):
        files = []
        if self.metrics_dir.is_dir():
            for f in sorted(self.metrics_dir.glob("*.json")):
                entry = {"file": f.name, "mtime": int(f.stat().st_mtime)}
                try:
                    entry["data"] = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError) as e:
                    entry["error"] = str(e)
                files.append(entry)
        return {"dir": str(self.metrics_dir), "files": files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=os.environ.get("RH20T", "/mnt/nas/data/RH20T"),
                    help="RH20T data root (frames live under <data-root>/frames)")
    ap.add_argument("--frames-root", default=None,
                    help="override the frames dir directly (default <data-root>/frames)")
    ap.add_argument("--raw-root", default=None,
                    help="raw scene dirs with transformed/*.npy (default <data-root>/raw)")
    ap.add_argument("--metrics-dir", default=str(Path(__file__).resolve().parent / "metrics"),
                    help="directory of metrics *.json files")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    Handler.frames_root = Path(args.frames_root or Path(args.data_root) / "frames")
    Handler.raw_root = Path(args.raw_root or Path(args.data_root) / "raw")
    Handler.metrics_dir = Path(args.metrics_dir)
    if not Handler.frames_root.is_dir():
        print(f"warning: frames root {Handler.frames_root} does not exist "
              f"(set $RH20T or --data-root); the Data tab will be empty", file=sys.stderr)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"visualizer: http://{args.host}:{args.port}  "
          f"(frames: {Handler.frames_root}, metrics: {Handler.metrics_dir})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
