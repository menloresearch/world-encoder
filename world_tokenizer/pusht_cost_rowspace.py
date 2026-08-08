"""Is the planning COST the bug, not the encoder? (v0.5 planning leg, pt. 35)

Our certificates say the representation and the dynamics are fine: 8-token dynamics are MORE
learnable (5-step rollout ratio 0.464-0.494) than the dense features of a system that plans at
0.90. Yet closed-loop planning fails at 0.08, and the failure is optimiser-induced. TRM
(arXiv 2605.22164) reports the mechanism that would explain this exactly: on their latents XY is
linearly decodable at R^2=0.998, yet the XY-probe ROWSPACE accounts for LESS THAN 1% of the
terminal-goal latent MSE. If that holds for us, then >99% of the quantity CEM minimises is
task-irrelevant variance, and the optimiser is free to buy huge cost reductions in directions
that do not move the task state. That would mean compactness does not break planning - it breaks
the naive Euclidean-MSE-to-goal cost - and it predicts the dense-vs-compact asymmetry we observe,
because a 196x384 patch grid is massively redundant and a spurious direction there has to fight
thousands of correlated features.

This measures that fraction on OUR latents, for BOTH z8 (compact) and the dense patch tokens, on
the same goal pairs the planner actually sees (t, t+25 at frameskip 5). Pure offline linear
algebra on cached tensors - no rollouts, no training, minutes not hours.

Read rule, registered before the number: if z8's task-relevant fraction is small (order 1%) AND
dense's is materially larger, the cost is indicted and the next planning arm is a probe-space (or
probe-weighted) planning cost with encoder, dynamics and planner all frozen. If the two are
comparable, the cost is exonerated and the exploitation story stays purely about landscape
smoothness (adversarial finetuning), not about what the cost measures.

  CUDA_VISIBLE_DEVICES=6 python -m world_tokenizer.pusht_cost_rowspace
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

DATA = Path("/home/menlo/dwm_data/pusht_noise")
FS = 5
GOAL_STEPS = 5           # planner's goal is 5 model-steps = 25 frames ahead
SAVE_PROBE = {}          # arm -> path; filled from --save-probe


def load(split, stem):
    root = DATA / split
    if (root / f"{stem}.pth").exists():
        lat = torch.load(root / f"{stem}.pth", mmap=True, weights_only=True)[:]
    else:
        parts = sorted(root.glob(f"{stem}_shard*of*.pth"))
        if not parts:
            return None, None, None
        lat = torch.cat([torch.load(p, weights_only=True) for p in parts], 0)
    states = torch.load(root / "states.pth").float()
    with open(root / "seq_lengths.pkl", "rb") as f:
        L = pickle.load(f)
    return lat, states, list(L)


def windows(L, span):
    return np.array([(ep, t) for ep, T in enumerate(L) for t in range(0, T - span)])


def analyse(name, lat, states, L, ridge=1e-3):
    """Fraction of goal-pair latent MSE that lives in the task-state probe's rowspace."""
    span = GOAL_STEPS * FS
    w = windows(L, span)
    if len(w) == 0:
        return None
    ep, t = w[:, 0], w[:, 1]

    # flatten tokens -> one vector per frame
    def vec(e, tt):
        z = lat[e, tt].float()
        return z.reshape(len(e), -1)

    Z0, Zg = vec(ep, t), vec(ep, t + span)
    S0 = states[ep, t].float()
    # centre and scale so the fraction is not an artefact of feature offsets
    mu = Z0.mean(0, keepdim=True)
    sd = Z0.std(0, keepdim=True) + 1e-6
    Z0n, Zgn = (Z0 - mu) / sd, (Zg - mu) / sd

    # ridge probe: standardized latent -> task state, fit on a HELD-OUT split.
    # this matters: the dense patch grid is 75k-dim against ~2k samples, so an in-sample R^2
    # is pure interpolation and tells us nothing. Report test R^2 only.
    perm = np.random.default_rng(0).permutation(len(Z0n))
    n_tr = int(0.7 * len(perm))
    tr, te = perm[:n_tr], perm[n_tr:]
    X, Y = Z0n[tr].double(), S0[tr].double()
    Xte, Yte = Z0n[te].double(), S0[te].double()
    Ym, Ys = Y.mean(0, keepdim=True), Y.std(0, keepdim=True) + 1e-8
    Yn = (Y - Ym) / Ys
    Yte_n = (Yte - Ym) / Ys
    n, d = X.shape[0], X.shape[1]
    if d <= n:
        A = X.T @ X + ridge * n * torch.eye(d, dtype=torch.float64)
        W = torch.linalg.solve(A, X.T @ Yn)      # [d, state_dim]
    else:
        # dual form: dense patch grids are ~75k-dim, so never form a d x d matrix
        K = X @ X.T + ridge * n * torch.eye(n, dtype=torch.float64)
        W = X.T @ torch.linalg.solve(K, Yn)
    r2_in = 1 - ((X @ W - Yn) ** 2).mean(0) / Yn.var(0)
    r2 = 1 - ((Xte @ W - Yte_n) ** 2).mean(0) / Yte_n.var(0)     # the honest one

    # orthonormal basis of the probe rowspace, then split the goal-difference energy
    Q, _ = torch.linalg.qr(W)                     # [d, k]
    D = (Zgn - Z0n).double()                      # what the cost is computed on
    total = (D ** 2).sum(1)
    inside = ((D @ Q) ** 2).sum(1)
    frac = (inside / total.clamp_min(1e-12))

    if SAVE_PROBE.get(name):
        # the probe is what a probe-space planning cost needs: standardisation + W + state scale
        torch.save({"mu": mu.squeeze(0), "sd": sd.squeeze(0), "W": W.float(),
                    "Ym": Ym.squeeze(0).float(), "Ys": Ys.squeeze(0).float(),
                    "r2": r2.float(), "arm": name},
                   SAVE_PROBE[name])
        print(f"SAVED probe -> {SAVE_PROBE[name]}", flush=True)

    return {
        "arm": name,
        "dim": int(d),
        "pairs": int(len(w)),
        "probe_r2_per_state_dim": [round(float(v), 4) for v in r2],
        "probe_r2_mean": round(float(r2.mean()), 4),
        "probe_r2_mean_INSAMPLE": round(float(r2_in.mean()), 4),
        "n_train": int(n),
        "task_relevant_frac_mean": round(float(frac.mean()), 6),
        "task_relevant_frac_median": round(float(frac.median()), 6),
        "rowspace_k": int(Q.shape[1]),
        "frac_if_random_subspace": round(float(Q.shape[1]) / d, 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*",
                    default=["z8_latents", "q4_latents", "q16_latents", "fsq_latents"])
    ap.add_argument("--dense", action="store_true", default=True,
                    help="also analyse the dense patch tokens (tokens.pth)")
    ap.add_argument("--out", default="results/pusht/cost_rowspace.json")
    ap.add_argument("--save-probe", nargs="*", default=[],
                    help="arms whose probe to persist for probe-space planning, e.g. z8_latents")
    args = ap.parse_args()
    for a in args.save_probe:
        SAVE_PROBE[a] = f"/mnt/nas/data/robocasa/kepler_ckpt/pusht_probe_{a}.pt"

    out = []
    for stem in args.arms:
        lat, states, L = load("val", stem)
        if lat is None:
            print(f"skip {stem} (no val cache)", flush=True)
            continue
        r = analyse(stem, lat, states, L)
        if r:
            out.append(r)
            print(json.dumps(r), flush=True)

    if args.dense:
        # the control: the dense DINOv2 patch grid their working 0.90 planner actually uses.
        # built by dino_wm/build_dense_cache.py (val/tokens.pth is NOT the patch grid).
        lat, states, L = load("val", "dense_latents")
        if lat is None:
            print("skip dense_patches (run dino_wm/build_dense_cache.py first)", flush=True)
        else:
            r = analyse("dense_patches", lat, states, L)
            if r:
                out.append(r)
                print(json.dumps(r), flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"\nSAVED {p}", flush=True)
    if len(out) >= 2:
        print("\nTASK-RELEVANT FRACTION OF GOAL-PAIR COST (TRM report <1% on theirs):")
        for r in out:
            enr = r['task_relevant_frac_mean'] / max(r['frac_if_random_subspace'], 1e-12)
            print(f"  {r['arm']:16} {r['task_relevant_frac_mean']*100:7.3f}%   "
                  f"enrichment vs chance {enr:7.2f}x   "
                  f"(test R2 {r['probe_r2_mean']:.3f}, in-sample "
                  f"{r['probe_r2_mean_INSAMPLE']:.3f}, d={r['dim']})")
        print("\nenrichment is the dimension-fair comparison; raw % is not comparable "
              "across arms of different dimensionality.")


if __name__ == "__main__":
    main()
