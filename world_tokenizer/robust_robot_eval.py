"""Strengthened robot-relevant eval WITH ERROR BARS — resolves 'is the flat real or noise?'.

For each checkpoint: embed once, then over N scene splits report mean±std of
  * contact accuracy — linear + kNN (chance 0.50)
  * force-magnitude regression R^2 (Ridge)  — how well the embedding predicts |F| (continuous)

Embedding is computed once per checkpoint; the N seeds only re-fit cheap sklearn heads.

    python -m world_tokenizer.robust_robot_eval \
        --ckpts e0: e6:/mnt/nas/data/RH20T/checkpoints/phase1_lr2e5_ckpt_e6.pt \
                e10hot:/mnt/nas/data/RH20T/checkpoints/phase1_ckpt_e10.pt --seeds 5
"""
import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier

from rh20t_api.configurations import load_conf
from world_tokenizer.contact_probe import CONF, RAW, sample_with_force
from world_tokenizer.model import LeJEPAVideo
from world_tokenizer.probe_curve import FRAMES, embed


def split(scenes, seed, frac=0.3):
    uniq = sorted(set(scenes))
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    test = set(uniq[: max(1, round(len(uniq) * frac))])
    tr = [i for i, s in enumerate(scenes) if s not in test]
    te = [i for i, s in enumerate(scenes) if s in test]
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="tag:path (path ignored for tag e0)")
    ap.add_argument("--frames-root", default=FRAMES)
    ap.add_argument("--raw-root", default=RAW, help="dir of raw RH20T_cfg* scene folders")
    ap.add_argument("--conf", default=CONF, help="rh20t_api configs.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--per-scene", type=int, default=20)
    ap.add_argument("--max-scenes", type=int, default=300)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    confs = load_conf(args.conf)
    rows = sample_with_force(args.raw_root, args.frames_root, confs, args.per_scene, args.max_scenes)
    paths = [r[0] for r in rows]
    scenes = [r[1] for r in rows]
    fmag = np.array([r[2] for r in rows])
    thr = np.median(fmag)
    yc = (fmag > thr).astype(int)
    print(f"{len(rows)} frames | contact thr={thr:.2f}N | seeds={args.seeds} | "
          f"contact chance=0.50 | |F| mean={fmag.mean():.2f} std={fmag.std():.2f}", flush=True)

    net = LeJEPAVideo(pretrained=True).to(dev).eval()
    print(f"{'ckpt':>7} | {'contact-linear':>15} | {'contact-kNN':>15} | {'force-R2':>15}")
    for spec in args.ckpts:
        tag, _, path = spec.partition(":")
        if tag != "e0":
            net.load_state_dict(torch.load(path, map_location=dev)["model"])
        X = embed(net, paths, dev)
        cl, kn, r2 = [], [], []
        for s in range(args.seeds):
            tr, te = split(scenes, s)
            cl.append(LogisticRegression(max_iter=2000).fit(X[tr], yc[tr]).score(X[te], yc[te]))
            kn.append(KNeighborsClassifier(args.k).fit(X[tr], yc[tr]).score(X[te], yc[te]))
            rg = Ridge(alpha=10.0).fit(X[tr], fmag[tr])
            r2.append(r2_score(fmag[te], rg.predict(X[te])))
        m = lambda a: f"{np.mean(a):.3f}±{np.std(a):.3f}"
        print(f"{tag:>7} | {m(cl):>15} | {m(kn):>15} | {m(r2):>15}", flush=True)


if __name__ == "__main__":
    main()
