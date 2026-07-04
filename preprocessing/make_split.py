"""Generate the frozen train/test split CSV (one-time; the CSV is committed).

Scans the RAW scene listing (not the caches, so the split is independent of which
chunks ever get precomputed), groups scenes by (cfg, task, user) — the same user
repeats the same task ~10x, so scenes within a group are near-duplicates and must
not straddle the split — and holds out --holdout-frac of groups per cfg (stratified).
Writes CSV columns: group,cfg,split. The dataloader reads this file; regenerate only
when deliberately versioning a new split (bump the filename, keep the old one).

    python preprocessing/make_split.py               # -> splits/holdout_v1.csv
"""
import argparse
import csv
import os
import re

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/mnt/nas/data/RH20T/raw")
    ap.add_argument("--cfgs", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "splits", "holdout_v1.csv"))
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    rows = []
    for n in args.cfgs:
        raw = os.path.join(args.raw_root, f"RH20T_cfg{n}")
        groups = sorted({re.sub(r"_scene_\d+", "", s) for s in os.listdir(raw)
                         if s.startswith("task_") and "_human" not in s})
        perm = rng.permutation(groups)
        n_test = max(1, round(len(groups) * args.holdout_frac))
        test = set(perm[:n_test].tolist())
        rows += [(g, n, "test" if g in test else "train") for g in groups]
        print(f"cfg{n}: {len(groups)} groups -> {n_test} test")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "cfg", "split"])
        w.writerows(sorted(rows))
    print(f"wrote {len(rows)} groups -> {args.out}")


if __name__ == "__main__":
    main()
