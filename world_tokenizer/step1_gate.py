"""Stage 2, Step 1 — cross-modal signal gate (cheap, no encoder yet).

Is there mutual information between video and robot_state? If frozen vision predicts the
kinematic state (pose/joints/gripper — the parts plausibly visible) above the mean baseline,
cross-modal signal exists and building the fusion encoder is justified. If ~0, the snapshot
signal is weak → go temporal. Reverse direction (state → vision) reported too. Force is
excluded (invisible in a frame, the wrong target). Scene-held-out, multi-seed, R^2.

    python -m world_tokenizer.step1_gate --seeds 5
"""
import argparse
import glob
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from world_tokenizer.model import LeJEPAVideo
from world_tokenizer.probe_curve import FRAMES, embed
from world_tokenizer.state import FT_DIMS, SceneState

RAW = "/mnt/nas/data/RH20T/raw/RH20T_cfg3"
# named slices of the 22-dim kinematic state (after removing F/T dims 21-26)
GROUPS = {"joints(sincos)": slice(0, 12), "tcp_pos": slice(12, 15),
          "quat6d": slice(15, 21), "gripper": slice(21, 22)}


def sample(frames_root, per_scene, max_scenes):
    rows = []
    for sc in sorted(d for d in os.listdir(frames_root) if d.startswith("task_"))[:max_scenes]:
        cams = sorted(glob.glob(os.path.join(frames_root, sc, "cam_*", "color")))
        if not cams:
            continue
        try:
            st = SceneState(os.path.join(RAW, sc))
        except Exception:
            continue
        fs = sorted(os.listdir(cams[0]))
        stride = max(1, len(fs) // per_scene)
        for f in fs[::stride][:per_scene]:
            ts = int(f.split(".")[0])
            try:
                kin = np.delete(st.state(ts), FT_DIMS)  # 22 dims, force excluded
            except Exception:
                continue
            if np.isfinite(kin).all():
                rows.append((os.path.join(cams[0], f), sc, kin))
    return rows


def split(scenes, seed, frac=0.3):
    u = sorted(set(scenes))
    rng = np.random.RandomState(seed)
    rng.shuffle(u)
    te = set(u[: max(1, round(len(u) * frac))])
    return ([i for i, s in enumerate(scenes) if s not in te],
            [i for i, s in enumerate(scenes) if s in te])


def _mlp():
    return MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=400)


def r2(Xin, Y, scenes, seeds, per_group=False):
    overall, groups = [], {g: [] for g in GROUPS}
    for s in range(seeds):
        tr, te = split(scenes, s)
        xs, ys = StandardScaler().fit(Xin[tr]), StandardScaler().fit(Y[tr])
        Xtr, Xte = xs.transform(Xin[tr]), xs.transform(Xin[te])
        Ytr = ys.transform(Y[tr])
        pred = ys.inverse_transform(_mlp().fit(Xtr, Ytr).predict(Xte))
        overall.append(r2_score(Y[te], pred))
        if per_group:
            for g, sl in GROUPS.items():
                groups[g].append(r2_score(Y[te][:, sl], pred[:, sl]))
    return np.array(overall), {g: np.array(v) for g, v in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-scene", type=int, default=20)
    ap.add_argument("--max-scenes", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = sample(FRAMES, args.per_scene, args.max_scenes)
    paths = [r[0] for r in rows]
    scenes = [r[1] for r in rows]
    kin = np.stack([r[2] for r in rows])
    net = LeJEPAVideo(pretrained=True).to(dev).eval()
    vision = embed(net, paths, dev)  # (N, 768)
    print(f"{len(rows)} frames | kinematic dims={kin.shape[1]} | seeds={args.seeds} | "
          f"R2>0 means predictable above the mean baseline", flush=True)

    # Direction A: vision -> kinematic state (the gate)
    ov, grp = r2(vision, kin, scenes, args.seeds, per_group=True)
    print(f"\nvision -> state   R2 = {ov.mean():.3f} ±{ov.std():.3f}")
    for g, v in grp.items():
        print(f"    {g:>16}: {v.mean():.3f} ±{v.std():.3f}")

    # Direction B: state -> vision (predict top PCA comps of vision)
    vpca = PCA(n_components=32).fit_transform(StandardScaler().fit_transform(vision))
    ov2, _ = r2(kin, vpca, scenes, args.seeds)
    print(f"\nstate -> vision(PCA32) R2 = {ov2.mean():.3f} ±{ov2.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
