"""robot_state decoder — the "superpowered linear probe" (PLAN 3.1). A small MLP decodes the
FROZEN vision-only latent z_v back to the robot's joint state, showing the latent
*reconstructs* state (not just that a linear probe can read it). Encoder frozen; only the
decoder trains (minutes). Reports MLP vs linear-probe R² and a predicted-vs-actual figure.

    python -m world_tokenizer.state_decoder --ckpt .../phase1/all/seed0.pt --cfgs 3 4 --out figures/decoder
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import ChunkDataset, load_split       # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks            # noqa: E402
from world_tokenizer.train_chunks import encode_zv, probe_r2          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default="figures/decoder")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    split = load_split()
    model = MMPerceiverChunks(d=args.d, n_queries=args.queries).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()
    ds = ChunkDataset(args.cache_dir, tuple(args.cfgs))
    is_test = np.array([split[ds.groups[i]] == "test" for i in ds._group_idx])
    zv, _ = encode_zv(model, ds, dev)

    d = ds._d
    motor = d["motor"].reshape(len(ds), -1).astype(np.float32)          # [N,24]
    mvalid = d["motor_mask"].reshape(len(ds), -1).all(0)
    y = motor[:, mvalid]
    valid_idx = np.flatnonzero(mvalid)
    tr, te = ~is_test, is_test

    sx = StandardScaler().fit(zv[tr]); sy = StandardScaler().fit(y[tr])
    Xtr = torch.tensor(sx.transform(zv[tr]), device=dev)
    Ytr = torch.tensor(sy.transform(y[tr]), device=dev)
    Xte = torch.tensor(sx.transform(zv[te]), device=dev)

    dec = nn.Sequential(nn.Linear(zv.shape[1], args.hidden), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(args.hidden, args.hidden), nn.GELU(),
                        nn.Linear(args.hidden, y.shape[1])).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Xtr)
    for ep in range(args.epochs):
        dec.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            loss = (dec(Xtr[idx]) - Ytr[idx]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    dec.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(dec(Xte).cpu().numpy())            # back to original scale

    mlp_r2 = r2_score(y[te], pred, multioutput="uniform_average")
    lin_r2 = probe_r2(zv[tr], y[tr], zv[te], y[te])
    print(f"\n=== robot_state decoder | {args.ckpt} | cfgs {args.cfgs} | n_test={te.sum()} ===")
    print(f"  MLP decoder R² = {mlp_r2:.3f}   |   linear-probe R² = {lin_r2:.3f}", flush=True)

    # predicted-vs-actual figure for interpretable dims (flattened motor: row*3 + ch)
    want = {0: "joint-0 sin(q)", 3: "joint-1 sin(q)", 6: "joint-2 sin(q)"}  # continuous joints (gripper is discrete → see gripper_classify)
    dims = [(np.where(valid_idx == k)[0][0], name) for k, name in want.items()
            if k in valid_idx]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        yte = y[te]
        fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 4))
        if len(dims) == 1:
            axes = [axes]
        rng = np.random.default_rng(0); sub = rng.choice(len(yte), min(3000, len(yte)), replace=False)
        for ax, (j, name) in zip(axes, dims):
            ax.scatter(yte[sub, j], pred[sub, j], s=4, alpha=0.3, color="#2a9d8f")
            lo, hi = yte[:, j].min(), yte[:, j].max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            r = r2_score(yte[:, j], pred[:, j])
            ax.set_title(f"{name}  (R²={r:.2f})"); ax.set_xlabel("actual"); ax.set_ylabel("decoded")
        fig.suptitle(f"robot_state decoded from frozen vision-only latent (overall R²={mlp_r2:.2f})")
        fig.savefig(f"{args.out}/state_decode.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {args.out}/state_decode.png", flush=True)
    except Exception as e:
        print(f"  [figure skipped: {e}]", flush=True)
    print("DONE state_decoder", flush=True)


if __name__ == "__main__":
    main()
