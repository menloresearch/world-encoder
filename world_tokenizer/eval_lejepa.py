"""LeJEPA-style eval over checkpoints — matches stable-pretraining's callbacks:
  * linear probe (OnlineProbe)  — labeled, scene-held-out
  * kNN probe  (OnlineKNN, k=20) — labeled, scene-held-out, parameter-free cross-check
  * RankMe                       — LABEL-FREE effective rank (dimensional-collapse detector)

RankMe formula copied from stable_pretraining/callbacks/rankme.py:
    s = svdvals(Z); p = s/sum(s) + 1e-5; RankMe = exp(-sum(p*log p))

    python -m world_tokenizer.eval_lejepa --ckpts e0 e3 e6 e10
"""
import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from world_tokenizer.model import LeJEPAVideo
from world_tokenizer.probe_curve import CKPT, FRAMES, embed, sample_frames, scene_split


def rankme(Z):
    s = torch.linalg.svdvals(torch.as_tensor(Z, dtype=torch.float32))
    p = (s / s.sum()) + 1e-5
    return float(torch.exp(-(p * torch.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default=FRAMES)
    ap.add_argument("--ckpts", nargs="+", default=["e0", "e3", "e6", "e10"])
    ap.add_argument("--ckpt-template", default=CKPT,
                    help="path template; '{}' filled with each ckpt tag (e.g. e3)")
    ap.add_argument("--per-scene", type=int, default=12)
    ap.add_argument("--max-per-class", type=int, default=300)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = sample_frames(args.frames_root, args.per_scene, args.max_per_class)
    tr, te, dropped = scene_split(rows)
    paths = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    ncl = len(set(y[tr]))
    print(f"{len(rows)} frames | {ncl} tasks (dropped {dropped}) | train={len(tr)} test={len(te)} "
          f"| chance={1 / ncl:.3f} | RankMe max={min(len(rows), 768)}", flush=True)

    net = LeJEPAVideo(pretrained=True).to(dev).eval()
    print(f"{'ckpt':>5} | {'linear':>7} | {'kNN(20)':>8} | {'RankMe':>7}")
    for c in args.ckpts:
        if c != "e0":
            net.load_state_dict(torch.load(args.ckpt_template.format(c), map_location=dev)["model"])
        X = embed(net, paths, dev)
        lin = LogisticRegression(max_iter=3000).fit(X[tr], y[tr]).score(X[te], y[te])
        knn = KNeighborsClassifier(n_neighbors=args.k).fit(X[tr], y[tr]).score(X[te], y[te])
        rm = rankme(X)
        print(f"{c:>5} | {lin:7.3f} | {knn:8.3f} | {rm:7.1f}", flush=True)


if __name__ == "__main__":
    main()
