"""RH20T mp4 -> timestamped jpg frames (color only; depth was not downloaded).

rh20t_api.extract has NO CLI (its __main__ is a hardcoded sample), so this wraps its
convert_dir per camera. Robust + resumable: cams without color.mp4 are skipped (some
scenes have a depth-only / dropped camera), a bad scene never aborts the batch, and
already-extracted cams are skipped on re-run. Output: <dest>/<scene>/<cam_*>/color/<ts>.jpg

Examples:
    python -m preprocessing.extract_frames --scene task_0001_user_0016_scene_0001_cfg_0003
    python -m preprocessing.extract_frames --all --num-workers 32         # all robot scenes
    python -m preprocessing.extract_frames --all --include-human          # also human demos
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool

from rh20t_api.extract import convert_dir

DEF_RAW = "/mnt/nas/data/RH20T/raw/RH20T_cfg3"
DEF_DEST = "/mnt/nas/data/RH20T/frames/cfg3"


def _is_scene(name, include_human=False):
    # scenes look like task_..._scene_..._cfg_0003 (and _human variants); skip calib/etc.
    if not (name.startswith("task_") and "scene_" in name):
        return False
    return include_human or not name.endswith("_human")


def _convert_one(raw_root, dest, scene):
    """Convert every camera in a scene that has color.mp4. Returns (scene, n_done, n_skipped)."""
    src, dst = os.path.join(raw_root, scene), os.path.join(dest, scene)
    done = skipped = 0
    for cam in sorted(os.listdir(src)):
        if not cam.startswith("cam_"):
            continue
        color = os.path.join(src, cam, "color.mp4")
        tsf = os.path.join(src, cam, "timestamps.npy")
        if not (os.path.exists(color) and os.path.exists(tsf)):
            skipped += 1
            continue
        out_color = os.path.join(dst, cam, "color")
        if os.path.isdir(out_color) and os.listdir(out_color):  # resume: already extracted
            done += 1
            continue
        try:
            convert_dir(color_file=color, timestamps_file=tsf,
                        dest_dir=os.path.join(dst, cam), depth_file=None)
            done += 1
        except Exception:
            skipped += 1
    return scene, done, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=DEF_RAW)
    ap.add_argument("--dest", default=DEF_DEST)
    ap.add_argument("--scene", default=None, help="single scene folder to convert (debug)")
    ap.add_argument("--all", action="store_true", help="convert every scene under --raw-root")
    ap.add_argument("--include-human", action="store_true", help="also extract _human demo scenes")
    ap.add_argument("--num-workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    if args.scene:
        print("converting", args.scene)
        print("done:", _convert_one(args.raw_root, args.dest, args.scene))
        return

    if not args.all:
        ap.error("pass --scene <name> for one scene, or --all for everything")

    scenes = sorted(s for s in os.listdir(args.raw_root) if _is_scene(s, args.include_human))
    print(f"{len(scenes)} scenes -> {args.dest} (workers={args.num_workers})", flush=True)
    fn = partial(_convert_one, args.raw_root, args.dest)
    failed = []
    with Pool(args.num_workers) as pool:
        for i, (scene, done, skipped) in enumerate(pool.imap_unordered(fn, scenes), 1):
            if done == 0:
                failed.append(scene)
            if i % 50 == 0 or i == len(scenes):
                print(f"[{i}/{len(scenes)}] last={scene} cams_done={done} skipped={skipped}", flush=True)
    print(f"SUMMARY: {len(scenes)} scenes, {len(failed)} with no cams extracted", flush=True)
    if failed:
        print("  no-cam scenes:", failed[:20], "..." if len(failed) > 20 else "")


if __name__ == "__main__":
    main()
