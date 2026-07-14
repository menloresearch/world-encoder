"""Surprise / invalid-state safety detector — runs on the FROZEN Phase-1 encoder (no training).

The encoder is trained by cross-modal prediction, so its prediction error is a built-in
"surprise" signal: LOW when the robot state is consistent with what vision expects, HIGH when
it's inconsistent. We show surprise cleanly flags corrupted state (mismatched / out-of-range)
— a direct robot-safety anomaly detector, on the encoder we already have.

    python -m world_tokenizer.surprise --ckpt /mnt/nas/data/RH20T/checkpoints/phase1/all/seed0.pt \
        --cfgs 3 4 --out figures/surprise
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import ChunkDataset, load_split      # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks           # noqa: E402


def auc(valid, bad):
    """AUROC of using surprise to separate valid(0) from bad(1) — via rank-sum, no sklearn dep."""
    y = np.r_[np.zeros(len(valid)), np.ones(len(bad))]
    s = np.r_[valid, bad]
    order = s.argsort()
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--noise", type=float, default=1.0, help="σ of out-of-range motor noise")
    ap.add_argument("--out", default="figures/surprise")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    split = load_split()
    model = MMPerceiverChunks(d=args.d, n_queries=args.queries).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    ds = ChunkDataset(args.cache_dir, tuple(args.cfgs))
    is_test = np.array([split[ds.groups[i]] == "test" for i in ds._group_idx])
    te = np.flatnonzero(is_test)
    d = ds._d
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(te))                                    # for the "mismatched state" corruption

    def surprise_over(state_idx, add_noise):
        """state_idx: which test rows to take the STATE (motor/ee) from (vision always row i).
        add_noise: σ of gaussian added to valid motor channels (0 = none)."""
        out = []
        for sl in (slice(i, min(i + args.bs, len(te))) for i in range(0, len(te), args.bs)):
            vi = te[sl]                                                # vision rows (always real)
            si = te[state_idx[sl]]                                     # state rows
            rgb = torch.from_numpy(d["patch"][vi].astype(np.float32)).to(dev)
            motor = torch.from_numpy(d["motor"][si].squeeze(1).astype(np.float32)).to(dev)
            m_mask = torch.from_numpy(d["motor_mask"][si]).to(dev)
            ee = torch.from_numpy(d["ee"][si].astype(np.float32)).to(dev)
            e_mask = torch.from_numpy(d["ee_mask"][si]).to(dev)
            if add_noise:
                motor = motor + add_noise * torch.randn_like(motor) * m_mask.float()
            out.append(model.surprise(rgb, motor, m_mask, ee, e_mask).cpu().numpy())
        return np.concatenate(out)

    ident = np.arange(len(te))
    s_valid = surprise_over(ident, 0.0)                                # real, matched state
    s_mismatch = surprise_over(perm, 0.0)                              # state from a different scene
    s_noise = surprise_over(ident, args.noise)                        # matched but out-of-range

    print(f"\n=== surprise detector | {args.ckpt} | cfgs {args.cfgs} | n={len(te)} ===", flush=True)
    for name, bad in [("mismatched state", s_mismatch), ("out-of-range noise", s_noise)]:
        print(f"  valid μ={s_valid.mean():.3f}  {name} μ={bad.mean():.3f}  "
              f"→ AUROC={auc(s_valid, bad):.3f}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lo, hi = np.percentile(np.r_[s_valid, s_mismatch, s_noise], [1, 99])
        bins = np.linspace(lo, hi, 60)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(s_valid, bins, alpha=0.6, label="valid state", color="#2a9d8f", density=True)
        ax.hist(s_mismatch, bins, alpha=0.5, label="mismatched state", color="#e76f51", density=True)
        ax.hist(s_noise, bins, alpha=0.5, label="out-of-range state", color="#8e44ad", density=True)
        ax.set_xlabel("cross-modal surprise"); ax.set_ylabel("density")
        ax.set_title("Surprise flags invalid robot state (frozen encoder, no training)")
        ax.legend()
        fig.savefig(f"{args.out}/surprise_hist.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {args.out}/surprise_hist.png", flush=True)
    except Exception as e:
        print(f"  [hist skipped: {e}]", flush=True)
    print("DONE surprise", flush=True)


if __name__ == "__main__":
    main()
