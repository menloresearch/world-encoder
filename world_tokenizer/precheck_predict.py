"""Pre-check A (TEMPORAL_ARCH.md §21.2): is the future LATENT predictable beyond carry-forward?

Freeze v0.1 (phase1), encode per-tick full multimodal latents over the EXISTING windows, and ask:
can a simple predictor map z_t -> z_{t+d} with LOWER error than the naive carry-forward baseline
(z_hat = z_t)? If yes at any delta -> the latent has learnable dynamics -> the §2.2 next-embedding
predictor is worth building. If even a strong (MLP) predictor cannot beat carry-forward at ANY
horizon -> red flag before building the belief-state.

All on the FROZEN v0.1 full multimodal latent z_full (nothing hidden):
    naive  : z_hat_{t+d} = z_t                 (carry-forward)
    linear : Ridge   z_t -> z_{t+d}            (matches codebase probe convention)
    mlp    : small MLP z_t -> z_{t+d}          (can a stronger predictor do better?)
Per delta: R2 (uniform avg, higher=better) + MSE ratio model/naive (<1 = beats carry-forward).
Secondary (the force differentiator): future force @ delta probed from the present latent vs the
naive "future force = present force" carry.

CAVEAT: the cache is subsampled (~1.7 s between ticks), so delta=1 is ~1.7 s ahead, delta=3 ~5 s.
This tests whether ANY learnable dynamics survive at those coarse horizons.

  .venv/bin/python -m world_tokenizer.precheck_predict \
      --v0 /mnt/nas/data/RH20T/checkpoints/phase1/kuka/seed0.pt --cfgs 6 7 \
      --out results/temporal/precheck_predict_kuka.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver2 import MMPerceiverChunks, masked_mean  # noqa: E402
from world_tokenizer.train_chunks import probe_r2, rankme                 # noqa: E402
from world_tokenizer.window_loader import make_window_loader              # noqa: E402


@torch.no_grad()
def embed_full(m, rgb, motor, mm, ee, em):
    """v0.1 full multimodal fused latent, nothing hidden. Mirrors embed_vision with hide=(). [B,d]."""
    ctx = m._context(rgb, m.motor_feats(motor, mm), ee)
    return m.fuse(ctx, m._attn_mask(mm.any(-1), em))


@torch.no_grad()
def encode(m, loader, dev, cl, deltas):
    """Per window: full latent at ctx tick cl and at cl+d; present/future force + validity."""
    m.eval()
    ticks = sorted(set([cl] + [cl + d for d in deltas]))
    Z = {k: [] for k in ticks}
    F = {k: [] for k in ticks}
    Fok = {k: [] for k in ticks}
    for b in loader:
        rgb = b["rgb"].to(dev); motor = b["motor"].to(dev); mm = b["motor_mask"].to(dev)
        ee = b["ee"].to(dev); em = b["ee_mask"].to(dev)
        for k in ticks:
            Z[k].append(embed_full(m, rgb[:, k], motor[:, k], mm[:, k], ee[:, k], em[:, k]).float().cpu().numpy())
            F[k].append(masked_mean(ee[:, k, :, :6], em[:, k]).cpu().numpy())
            Fok[k].append(em[:, k].any(-1).cpu().numpy())
    cat = lambda D: {k: np.concatenate(v) for k, v in D.items()}
    return cat(Z), cat(F), cat(Fok)


def mse(a, b):
    return float(((a - b) ** 2).mean())


def lin_predict(xtr, ytr, xte):
    sx = StandardScaler().fit(xtr)
    reg = Ridge(alpha=10.0).fit(sx.transform(xtr), ytr)
    return reg.predict(sx.transform(xte))


def mlp_predict(xtr, ytr, xte, dev, epochs=200, hid=256, lr=1e-3, bs=256, val_frac=0.1, seed=0):
    """Minibatch MLP with early stopping on a carved-out val split (avoids the full-batch overfit)."""
    sx, sy = StandardScaler().fit(xtr), StandardScaler().fit(ytr)
    Xtr, Ytr = sx.transform(xtr), sy.transform(ytr)
    n = len(Xtr); nv = max(64, int(n * val_frac))
    perm = np.random.RandomState(seed).permutation(n)
    vi, ti = perm[:nv], perm[nv:]
    Xt = torch.tensor(Xtr[ti], dtype=torch.float32, device=dev)
    Yt = torch.tensor(Ytr[ti], dtype=torch.float32, device=dev)
    Xv = torch.tensor(Xtr[vi], dtype=torch.float32, device=dev)
    Yv = torch.tensor(Ytr[vi], dtype=torch.float32, device=dev)
    Xe = torch.tensor(sx.transform(xte), dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(Xt.shape[1], hid), nn.GELU(),
                        nn.Linear(hid, Yt.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best_v, best_state, bad, nb = 1e9, None, 0, Xt.shape[0]
    for _ in range(epochs):
        net.train(); pi = torch.randperm(nb, device=dev)
        for j in range(0, nb, bs):
            idx = pi[j:j + bs]
            opt.zero_grad(); lossf(net(Xt[idx]), Yt[idx]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(Xv), Yv).item()
        if v < best_v - 1e-5:
            best_v, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 20:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(Xe).cpu().numpy()
    return sy.inverse_transform(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True, help="v0.1 single-tick checkpoint (phase1)")
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4, help="ctx ticks; input latent = tick ctx-1")
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    cl = args.ctx - 1

    v0 = MMPerceiverChunks(d=256, n_queries=8).to(dev)
    v0.load_state_dict(torch.load(args.v0, map_location=dev))
    print(f"v0.1={args.v0}  cfgs={args.cfgs}  cl(tick)={cl}  deltas={args.deltas}", flush=True)

    tr, te, _ = make_window_loader(args.cache_dir, tuple(args.cfgs), window=args.window,
                                   stride=args.stride, batch=128, num_workers=4)
    Ztr, Ftr, Foktr = encode(v0, tr, dev, cl, args.deltas)
    Zte, Fte, Fokte = encode(v0, te, dev, cl, args.deltas)
    n_tr, n_te = len(Ztr[cl]), len(Zte[cl])
    print(f"windows: train {n_tr}  test {n_te}  latent dim {Ztr[cl].shape[1]}  "
          f"RankMe(z_full test) {rankme(Zte[cl]):.1f}", flush=True)

    res = {"v0": args.v0, "cfgs": args.cfgs, "window": args.window, "ctx": args.ctx,
           "deltas": args.deltas, "stride": args.stride, "n_train": n_tr, "n_test": n_te,
           "latent_dim": int(Ztr[cl].shape[1]), "rankme_z_full": rankme(Zte[cl]), "per_delta": {}}

    x_tr, x_te = Ztr[cl], Zte[cl]
    print("\n=== LATENT PREDICTABILITY (z_t -> z_{t+d}); MSE ratio <1 beats carry-forward ===", flush=True)
    print(f"{'d':>2} | {'naiveR2':>8} {'linR2':>8} {'mlpR2':>8} | {'naiveMSE':>9} "
          f"{'lin/naive':>9} {'mlp/naive':>9}", flush=True)
    for d in args.deltas:
        y_tr, y_te = Ztr[cl + d], Zte[cl + d]
        naive_mse = mse(x_te, y_te)
        naive_r2 = float(r2_score(y_te, x_te, multioutput="uniform_average"))
        lp = lin_predict(x_tr, y_tr, x_te)
        mp = mlp_predict(x_tr, y_tr, x_te, dev)
        lin_r2 = float(r2_score(y_te, lp, multioutput="uniform_average"))
        mlp_r2 = float(r2_score(y_te, mp, multioutput="uniform_average"))
        lin_mse, mlp_mse = mse(lp, y_te), mse(mp, y_te)
        res["per_delta"][str(d)] = {
            "naive_r2": naive_r2, "lin_r2": lin_r2, "mlp_r2": mlp_r2,
            "naive_mse": naive_mse, "lin_mse": lin_mse, "mlp_mse": mlp_mse,
            "lin_over_naive_mse": lin_mse / naive_mse, "mlp_over_naive_mse": mlp_mse / naive_mse}
        print(f"{d:>2} | {naive_r2:>8.3f} {lin_r2:>8.3f} {mlp_r2:>8.3f} | {naive_mse:>9.4f} "
              f"{lin_mse / naive_mse:>9.3f} {mlp_mse / naive_mse:>9.3f}", flush=True)

    # ---- secondary: does the PRESENT full latent recover FUTURE force better than carrying force? ----
    print("\n=== FUTURE FORCE from present latent vs carry-forward force (the differentiator) ===", flush=True)
    print(f"{'d':>2} | {'latentR2':>9} {'carryR2':>9}", flush=True)
    res["future_force"] = {}
    f_now_te, f_now_ok_te = Fte[cl], Fokte[cl]
    for d in args.deltas:
        ok_tr = Foktr[cl] & Foktr[cl + d]
        ok_te = Fokte[cl] & Fokte[cl + d]
        yf_tr, yf_te = Ftr[cl + d], Fte[cl + d]
        lat_r2 = (probe_r2(Ztr[cl][ok_tr], yf_tr[ok_tr], Zte[cl][ok_te], yf_te[ok_te])
                  if ok_tr.sum() > 50 and ok_te.sum() > 50 else float("nan"))
        nok = ok_te & f_now_ok_te
        carry_r2 = (float(r2_score(yf_te[nok], f_now_te[nok], multioutput="uniform_average"))
                    if nok.sum() > 50 else float("nan"))
        res["future_force"][str(d)] = {"latent_r2": lat_r2, "carry_r2": carry_r2,
                                       "n": int(ok_te.sum())}
        print(f"{d:>2} | {lat_r2:>9.3f} {carry_r2:>9.3f}", flush=True)

    # ---- verdict (best of linear OR mlp beats carry-forward at any delta) ----
    ratios = [(d, min(v["lin_over_naive_mse"], v["mlp_over_naive_mse"]),
               "lin" if v["lin_over_naive_mse"] <= v["mlp_over_naive_mse"] else "mlp")
              for d, v in res["per_delta"].items()]
    best_d, best_ratio, best_model = min(ratios, key=lambda r: r[1])
    lin_best = min(v["lin_over_naive_mse"] for v in res["per_delta"].values())
    beats = best_ratio < 1.0
    res["verdict"] = {"any_delta_beats_carryforward": bool(beats), "best_model": best_model,
                      "best_delta": best_d, "best_over_naive_mse": best_ratio,
                      "best_linear_over_naive_mse": lin_best}
    print(f"\nVERDICT: a predictor {'BEATS' if beats else 'does NOT beat'} carry-forward in latent "
          f"space (best {best_model} @Δ{best_d}, MSE ratio {best_ratio:.3f}; best linear {lin_best:.3f}). "
          f"{'-> step 2 (next-embedding predictor) worth building.' if beats else '-> RED FLAG, revisit before belief-state.'}",
          flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
