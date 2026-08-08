"""Fit a support-constrained CEM proposal from the demo action manifold (v0.5 pt. 38, rank-1 remedy).

The measured problem: CEM's proposal is `randn(n, H, A) * sigma + mu` (planning/cem.py) — independent
Gaussian per (timestep, dim). The demo action distribution is nothing like that. Measured on our data:
effective rank ~6 of 50, lag-1 autocorrelation ~0.87, and a ~34 nats/chunk log-likelihood gap against
N(0,I). The marginal is already matched by the harness's normalisation, so **the missing structure is
temporal covariance** — and as CEM refines, the returned plan walks from Mahalanobis^2 13 to 334 against
the demo covariance (demo scale ~34-50, isotropic noise ~354), i.e. it ends up indistinguishable from
noise while its predicted cost falls 4.6x.

This fits the H*A-dimensional chunk covariance of the demo actions and saves its Cholesky factor,
normalised so each element keeps unit marginal std. CEM then draws `randn @ L.T` instead of `randn`,
which injects the demo temporal correlation while leaving sigma adaptation, top-k selection and the mu
update exactly as they were. Encoder, dynamics and objective all untouched — eval-only.

  python -m world_tokenizer.pusht_action_proposal --horizon 5
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

DATA = Path("/home/menlo/dwm_data/pusht_noise")
FS = 5
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5, help="CEM horizon in MODEL steps")
    ap.add_argument("--split", default="train")
    ap.add_argument("--shrink", type=float, default=0.05,
                    help="Ledoit-Wolf style shrinkage toward the identity; keeps L invertible and "
                         "stops the proposal from collapsing onto the top demo directions")
    ap.add_argument("--out", default="/mnt/nas/data/robocasa/kepler_ckpt/pusht_action_proposal.pt")
    args = ap.parse_args()

    root = DATA / args.split
    acts = torch.load(root / "rel_actions.pth").float() / 100.0
    acts = (acts - ACTION_MEAN) / ACTION_STD            # the harness's own normalisation
    with open(root / "seq_lengths.pkl", "rb") as f:
        L = list(pickle.load(f))

    span = args.horizon * FS
    A = 2 * FS                                          # 10-D per model step
    chunks = []
    for ep, T in enumerate(L):
        for t in range(0, T - span, FS):
            a = acts[ep, t:t + span]                    # [span, 2]
            chunks.append(a.reshape(args.horizon, A))
    X = torch.stack(chunks).reshape(len(chunks), -1).double()   # [N, H*A]
    d = X.shape[1]
    print(f"{len(X)} demo chunks of dim {d} (horizon {args.horizon} x {A})", flush=True)

    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    # rescale to a CORRELATION matrix: sigma adaptation in CEM owns the scale, we only supply shape
    s = torch.sqrt(torch.diag(C)).clamp_min(1e-8)
    R = C / s[:, None] / s[None, :]
    R = (1 - args.shrink) * R + args.shrink * torch.eye(d, dtype=torch.float64)
    Lc = torch.linalg.cholesky(R)

    # diagnostics the fix is meant to move
    ev = torch.linalg.eigvalsh(R).clamp_min(0)
    eff_rank = float(ev.sum() ** 2 / (ev ** 2).sum())
    flat = Xc.reshape(len(Xc), args.horizon, A)
    if args.horizon > 1:
        a0 = flat[:, :-1].reshape(-1)
        a1 = flat[:, 1:].reshape(-1)
        lag1 = float(((a0 - a0.mean()) * (a1 - a1.mean())).mean() / (a0.std() * a1.std()))
    else:
        lag1 = float("nan")
    print(f"correlation effective rank {eff_rank:.2f} / {d}", flush=True)
    print(f"lag-1 autocorrelation across model steps {lag1:.3f}", flush=True)

    # sanity: a correlated draw must keep unit marginal std, or it silently rescales CEM's sigma
    g = torch.randn(20000, d, dtype=torch.float64) @ Lc.T
    print(f"correlated-draw marginal std: mean {g.std(0).mean():.4f} "
          f"(must be ~1.000), min {g.std(0).min():.4f}, max {g.std(0).max():.4f}", flush=True)

    # also save mean + precision of the RAW covariance, for the support PENALTY (pt. 40).
    # The proposal (correlation Cholesky) only reshapes exploration AROUND mu; because CEM sets
    # mu = topk.mean(), mu itself can still drift anywhere. Penalising Mahalanobis distance in the
    # SELECTION cost is what actually bounds the drift.
    Cs = C + 1e-4 * torch.eye(d, dtype=torch.float64)
    prec = torch.linalg.inv(Cs)
    demo_m2 = ((Xc @ prec) * Xc).sum(1)
    print(f"demo Mahalanobis^2: mean {demo_m2.mean():.1f}, median {demo_m2.median():.1f}, "
          f"p95 {demo_m2.quantile(0.95):.1f}  (dim {d} = chi2 mean)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"L": Lc.float(), "horizon": args.horizon, "action_dim": A,
                "eff_rank": eff_rank, "lag1": lag1, "shrink": args.shrink,
                "n_chunks": len(X),
                "mu_a": mu.float(), "prec": prec.float(),
                "demo_m2_mean": float(demo_m2.mean()),
                "demo_m2_p95": float(demo_m2.quantile(0.95))}, out)
    print(f"SAVED {out}", flush=True)


if __name__ == "__main__":
    main()
