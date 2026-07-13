"""Gripper open/closed CLASSIFICATION from the frozen vision-only z_v — the right metric,
since gripper width is ~discrete (R² is misleading, per JQ/Nicole). Logistic probe z_v ->
open/closed (threshold = train median → balanced, chance 50%); reports accuracy, balanced
accuracy, AUROC on the held-out split, vs a raw-ViT baseline.

    python -m world_tokenizer.gripper_classify --ckpt .../phase1/all/seed0.pt --cfgs 3 4
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import ChunkDataset, load_split       # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks            # noqa: E402
from world_tokenizer.train_chunks import encode_zv                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    split = load_split()
    model = MMPerceiverChunks(d=args.d, n_queries=args.queries).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    ds = ChunkDataset(args.cache_dir, tuple(args.cfgs))
    is_test = np.array([split[ds.groups[i]] == "test" for i in ds._group_idx])
    zv, raw = encode_zv(model, ds, dev)

    grip = ds._d["motor"].reshape(len(ds), -1)[:, 21].astype(np.float64)   # row7 ch0 = symlog gripper width
    tr, te = ~is_test, is_test
    thr = np.median(grip[tr])
    y = (grip > thr).astype(int)
    print(f"\n=== gripper open/closed classification | {args.ckpt} | cfgs {args.cfgs} ===")
    print(f"n_train={tr.sum()} n_test={te.sum()} | test class balance: "
          f"{y[te].mean():.2f} open  (threshold=train median symlog width {thr:.3f})", flush=True)
    curves = []
    for name, X in [("z_v (fused)", zv), ("raw ViT", raw)]:
        sx = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sx.transform(X[tr]), y[tr])
        p = clf.predict_proba(sx.transform(X[te]))[:, 1]
        pred = (p > 0.5).astype(int)
        auc = roc_auc_score(y[te], p)
        print(f"  {name:12s}: acc {accuracy_score(y[te], pred):.3f} | "
              f"balanced-acc {balanced_accuracy_score(y[te], pred):.3f} | AUROC {auc:.3f}", flush=True)
        curves.append((name, y[te], p, auc))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "figures", "decoder", "gripper_roc.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        for name, yt, p, auc in curves:
            fpr, tpr, _ = roc_curve(yt, p)
            ax.plot(fpr, tpr, label=f"{name}  AUROC={auc:.3f}",
                    color="#2a9d8f" if "z_v" in name else "#e9c46a", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.set_title("Gripper open/closed from frozen vision-only $z_v$"); ax.legend(loc="lower right")
        fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}", flush=True)
    except Exception as e:
        print(f"  [ROC fig skipped: {e}]", flush=True)
    print("DONE gripper_classify", flush=True)


if __name__ == "__main__":
    main()
