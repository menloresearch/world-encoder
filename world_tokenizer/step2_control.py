"""Stage 2, Step 2 — cheap fusion control (leak-free), before building the Perceiver.

Predict force from vision, kinematics, or both — with force EXCLUDED from the input, so we test
whether force can be *inferred* cross-modally (the hidden-variable question), not copied.

  inputs (per frame):  vision = frozen e0 CLS (768) | kinematic = state minus F/T dims (22)
  targets:             contact = |force|>median (acc, chance 0.50) | |force| (Ridge R^2)
  split:               scene-held-out, multi-seed (mean±std)

Decision rule: if fused beats BOTH vision-only and kinematic-only, there's cross-modal signal
worth building the encoder for. If not, rethink before investing.
"""
import argparse
import glob
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from world_tokenizer.model import LeJEPAVideo
from world_tokenizer.probe_curve import FRAMES, embed
from world_tokenizer.state import FT_DIMS, SceneState

RAW = "/mnt/nas/data/RH20T/cfg3_raw/RH20T_cfg3"


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
                vec = st.state(ts)
                fmag = float(np.linalg.norm(st.raw_ft(ts)[:3]))
            except Exception:
                continue
            if not np.isfinite(fmag):
                continue
            kin = np.delete(vec, FT_DIMS)  # 22 dims, force excluded
            rows.append((os.path.join(cams[0], f), sc, kin, fmag))
    return rows


def split(scenes, seed, frac=0.3):
    u = sorted(set(scenes))
    rng = np.random.RandomState(seed)
    rng.shuffle(u)
    te = set(u[: max(1, round(len(u) * frac))])
    tr = [i for i, s in enumerate(scenes) if s not in te]
    tev = [i for i, s in enumerate(scenes) if s in te]
    return tr, tev


def _clf(head):
    return (MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True, max_iter=300)
            if head == "mlp" else LogisticRegression(max_iter=2000))


def _reg(head):
    return (MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True, max_iter=300)
            if head == "mlp" else Ridge(alpha=10.0))


def evalX(X, yc, fmag, scenes, seeds, head):
    cl, r2 = [], []
    for s in range(seeds):
        tr, te = split(scenes, s)
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        cl.append(_clf(head).fit(Xtr, yc[tr]).score(Xte, yc[te]))
        r2.append(r2_score(fmag[te], _reg(head).fit(Xtr, fmag[tr]).predict(Xte)))
    return np.array(cl), np.array(r2)


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
    fmag = np.array([r[3] for r in rows])
    yc = (fmag > np.median(fmag)).astype(int)
    print(f"{len(rows)} frames | contact thr={np.median(fmag):.2f}N | seeds={args.seeds} | "
          f"contact chance=0.50 | kinematic dims={kin.shape[1]}", flush=True)

    net = LeJEPAVideo(pretrained=True).to(dev).eval()
    vision = embed(net, paths, dev)  # (N, 768)
    inputs = {"vision (768)": vision, "kinematic (22)": kin,
              "fused (790)": np.concatenate([vision, kin], axis=1)}

    for head in ["linear", "mlp"]:
        print(f"\n[{head}] {'input':>14} | {'contact-acc':>14} | {'force-R2':>14}")
        for name, X in inputs.items():
            cl, r2 = evalX(X, yc, fmag, scenes, args.seeds, head)
            print(f"       {name:>14} | {cl.mean():.3f} ±{cl.std():.3f} | {r2.mean():.3f} ±{r2.std():.3f}", flush=True)


if __name__ == "__main__":
    main()
