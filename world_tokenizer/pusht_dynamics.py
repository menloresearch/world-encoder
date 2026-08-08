"""PushT rung-0 dynamics pair (claim C, 2026-08-04, protocol = PROGRESS.md pt. 18).

Trains the SAME small transformer trunk as action-conditioned latent dynamics
p(z_{t+1} | z_t, a_t) on either arm:
  z8/z1/z4/z16 : frozen state-blind Kepler token set [K,256] from
                 pusht_mc1_v03[_qK]_lp2.0 (fuse, state masked, pool=False)
  dense        : frozen ViT-B/16 patch tokens [196,768] straight from the cache
                 (DINO-WM-style strong baseline)

Registered choices (pt. 18): trunk depth 4 / d 256 / 8 heads / FFN 1024, one
action token, learned pos-emb per latent token, predicts the RESIDUAL in
per-dim standardized space; time base = 1 cache row (stride-2, 0.2 s),
a_t = both raw 10 Hz actions in the interval (4-D, a/256-1); Markov; AdamW
3e-4 / wd 1e-4, batch 128, <=60 ep, ep%10==0 held out, early-stop 10, seed 0.
Kill-rule: val MSE must beat carry-forward (pred/carry < 1) or the arm's
dynamics is unusable as built.

Also fits the planner's cost head: closed-form ridge latent -> sim state
(targets = [agent_xy, block_xy, cos, sin] — angle as cos/sin, planner reads
it back via atan2; primal for D<=8192 else dual form), lambda swept on val R².

  CUDA_VISIBLE_DEVICES=1 python -m world_tokenizer.pusht_dynamics --arm z8
  CUDA_VISIBLE_DEVICES=0 python -m world_tokenizer.pusht_dynamics --arm dense
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CKPT_ROOT = "/mnt/nas/data/robocasa/kepler_ckpt"
CACHE = "/mnt/nas/data/robocasa/kepler_cache/pusht_patches.npz"
ACT_DIR = "/mnt/nas/data/robocasa/kepler_cache/pusht_actions_lerobot/data/chunk-000"
ENC_DIR = {"z8": "pusht_mc1_v03_lp2.0", "z1": "pusht_mc1_v03_q1_lp2.0",
           "z4": "pusht_mc1_v03_q4_lp2.0", "z16": "pusht_mc1_v03_q16_lp2.0",
           "z8fsq": "pusht_mc1_v03_fsq_lp2.0"}


def enc_class(arm):
    if arm == "z8fsq":
        from world_tokenizer.pusht_fsq import MMPerceiverMCPatchFSQ
        return MMPerceiverMCPatchFSQ
    from world_tokenizer.robocasa_pretrain_v03 import MMPerceiverMCPatch
    return MMPerceiverMCPatch


class LatentDynamics(nn.Module):
    """Shared trunk; arm-specific I/O projections. Predicts standardized residual."""

    def __init__(self, n_tokens, in_dim, d=256, depth=4, heads=8, ffn=1024, act_dim=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d) if in_dim != d else nn.Identity()
        self.out_proj = nn.Linear(d, in_dim) if in_dim != d else nn.Linear(d, d)
        self.pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.act = nn.Sequential(nn.Linear(act_dim, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, ffn, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.n_tokens = n_tokens

    def forward(self, z, a):
        # z: [B, n_tokens, in_dim] standardized; a: [B, act_dim] -> residual [B, n_tokens, in_dim]
        x = self.in_proj(z) + self.pos
        x = torch.cat([x, self.act(a).unsqueeze(1)], 1)
        return self.out_proj(self.trunk(x)[:, :self.n_tokens])


def encode_zset(arm, patch, state_raw, dev, bs=512):
    enc_dir = os.path.join(CKPT_ROOT, ENC_DIR[arm])
    meta = json.load(open(os.path.join(enc_dir, "results.json")))
    mu = np.array(meta["state_mu"], np.float32)
    sd = np.array(meta["state_sd"], np.float32)
    nq = meta["args"]["queries"]
    model = enc_class(arm)(n_cams=1, d=256, state_dim=5, n_queries=nq,
                           lamb_patch=2.0, mask_ratio=0.25).to(dev).eval()
    model.load_state_dict(torch.load(os.path.join(enc_dir, "seed0.pt"),
                                     map_location=dev))
    s_std = (state_raw - mu) / sd
    zs = []
    with torch.no_grad():
        for i in range(0, len(s_std), bs):
            p = torch.from_numpy(patch[i:i + bs].astype(np.float32)).to(dev)
            s = torch.from_numpy(s_std[i:i + bs]).to(dev)
            m = model._mask(True, dev)
            z = model.fuse(model._context(p, s), m, pool=False)
            if arm == "z8fsq":
                z = model.fsq(z)
            zs.append(z.cpu())
    return torch.cat(zs).numpy(), enc_dir  # [N, nq, 256]


def ridge_probe(X_tr, y_tr, X_va, y_va, dev):
    """Closed-form ridge, lambda swept on val R^2. Primal if D small, else dual."""
    Xt = torch.from_numpy(X_tr).to(dev, torch.float32)
    yt = torch.from_numpy(y_tr).to(dev, torch.float32)
    xm, ym = Xt.mean(0), yt.mean(0)
    Xc, yc = Xt - xm, yt - ym
    Xv = torch.from_numpy(X_va).to(dev, torch.float32) - xm
    yv = torch.from_numpy(y_va).to(dev, torch.float32)
    n, D = Xc.shape
    primal = D <= 8192
    G = (Xc.T @ Xc) if primal else (Xc @ Xc.T)          # gram, computed once
    rhs = (Xc.T @ yc) if primal else yc
    eye = torch.eye(G.shape[0], device=dev)
    best = None
    for lam in [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
        sol = torch.linalg.solve(G + lam * eye, rhs)
        W = sol if primal else Xc.T @ sol
        pred = Xv @ W + ym
        r2 = 1 - ((pred - yv).square().sum(0) / (yv - yv.mean(0)).square().sum(0) + 1e-12)
        score = float(r2.mean())
        if best is None or score > best[0]:
            best = (score, lam, W.cpu(), xm.cpu(), ym.cpu(), r2.cpu())
    score, lam, W, xm, ym, r2 = best
    return {"W": W, "x_mean": xm, "y_mean": ym, "lam": lam,
            "val_r2_mean": score, "val_r2_per_dim": r2.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ENC_DIR) + ["dense"])
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-base", default=os.path.join(CKPT_ROOT, "pusht_dyn"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    d = np.load(args.cache)
    patch, state, ep, t = d["patch"], d["state"].astype(np.float32), d["ep"], d["t"]
    N = len(ep)

    # latents per arm
    enc_dir = None
    if args.arm == "dense":
        lat = patch[:, 0]                                   # [N,196,768] fp16 (cast per batch)
    else:
        lat, enc_dir = encode_zset(args.arm, patch, state, dev)  # [N,nq,256] f32
    n_tok, in_dim = lat.shape[1], lat.shape[2]
    print(f"arm={args.arm} latents [{N},{n_tok},{in_dim}]", flush=True)

    # actions: 4-D chunk (both raw 10 Hz actions inside a stride-2 row interval), a/256-1
    acts_by_ep = {}
    for e in np.unique(ep):
        df = pd.read_parquet(os.path.join(ACT_DIR, f"episode_{e:06d}.parquet"))
        acts_by_ep[int(e)] = np.stack(df["action"].to_numpy()).astype(np.float32)
    # pairs: consecutive rows, same episode, tick gap == stride
    src = np.flatnonzero((ep[1:] == ep[:-1]) & (t[1:] - t[:-1] == 2))
    A = np.stack([np.concatenate([acts_by_ep[int(ep[i])][t[i]],
                                  acts_by_ep[int(ep[i])][t[i] + 1]]) for i in src])
    A = A / 256.0 - 1.0
    is_val = (ep[src] % 10 == 0)
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"{len(src)} pairs (train {len(tr)}, val {len(va)})", flush=True)

    # per-dim standardization over train rows+tokens (pt. 18)
    tr_rows = np.unique(np.concatenate([src[tr], src[tr] + 1]))
    if args.arm == "dense":
        acc_n, acc_s, acc_q = 0, np.zeros(in_dim, np.float64), np.zeros(in_dim, np.float64)
        for i in range(0, len(tr_rows), 1024):
            x = lat[tr_rows[i:i + 1024]].astype(np.float64)
            acc_s += x.sum((0, 1)); acc_q += (x ** 2).sum((0, 1))
            acc_n += x.shape[0] * x.shape[1]
        mu_l = (acc_s / acc_n).astype(np.float32)
        sd_l = np.sqrt(acc_q / acc_n - (acc_s / acc_n) ** 2).astype(np.float32) + 1e-6
    else:
        flat = lat[tr_rows].reshape(-1, in_dim)
        mu_l, sd_l = flat.mean(0), flat.std(0) + 1e-6
    mu_t = torch.from_numpy(mu_l).to(dev)
    sd_t = torch.from_numpy(sd_l).to(dev)

    def get(rows):
        x = torch.from_numpy(lat[rows].astype(np.float32)).to(dev)
        return (x - mu_t) / sd_t

    model = LatentDynamics(n_tok, in_dim).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    A_t = torch.from_numpy(A).to(dev)
    print(f"dynamics params {n_par/1e6:.2f}M", flush=True)

    def val_mse():
        model.eval()
        se_p, se_c, m = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va), 256):
                sl = va[i:i + 256]
                z0, z1 = get(src[sl]), get(src[sl] + 1)
                r = z1 - z0
                pr = model(z0, A_t[sl])
                se_p += float((pr - r).square().sum())
                se_c += float(r.square().sum())
                m += r.numel()
        return se_p / m, se_c / m

    best, bad, best_ep = None, 0, -1
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(tr))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            sl = tr[perm[i:i + args.batch]]
            z0, z1 = get(src[sl]), get(src[sl] + 1)
            loss = (model(z0, A_t[sl]) - (z1 - z0)).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        vp, vc = val_mse()
        print(f"ep {epoch}: train {tot/nb:.5f} val {vp:.5f} carry {vc:.5f} "
              f"ratio {vp/vc:.3f}", flush=True)
        if best is None or vp < best[0]:
            best, best_ep, bad = (vp, vc), epoch, 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at ep {epoch} (best ep {best_ep})", flush=True)
                break
    model.load_state_dict(best_sd)
    model.eval()

    # open-loop rollout MSE at horizons 1/2/4/8 vs carry (val, standardized space)
    roll = {}
    with torch.no_grad():
        for H in [1, 2, 4, 8]:
            ok = np.flatnonzero(is_val[:len(src) - H + 1] if H == 1 else
                                np.array([is_val[j] and j + H - 1 < len(src)
                                          and ep[src[j]] == ep[src[j + H - 1]]
                                          and all(src[j + k] + 1 == src[j + k + 1]
                                                  for k in range(H - 1))
                                          for j in range(len(src) - H + 1)]))
            se_p, se_c, m = 0.0, 0.0, 0
            for i in range(0, len(ok), 256):
                sl = ok[i:i + 256]
                z = get(src[sl])
                for k in range(H):
                    z = z + model(z, A_t[sl + k])
                zt = get(src[sl + H - 1] + 1)
                z0 = get(src[sl])
                se_p += float((z - zt).square().sum())
                se_c += float((z0 - zt).square().sum())
                m += zt.numel()
            roll[H] = {"pred": se_p / m, "carry": se_c / m,
                       "ratio": (se_p / m) / (se_c / m), "n": int(len(ok))}
            print(f"rollout H={H}: pred {roll[H]['pred']:.5f} carry "
                  f"{roll[H]['carry']:.5f} ratio {roll[H]['ratio']:.3f}", flush=True)

    # cost-head probe: latent -> [agent_xy, block_xy, cos, sin]
    ang = state[:, 4]
    y = np.concatenate([state[:, :4], np.cos(ang)[:, None], np.sin(ang)[:, None]], 1)
    rows_tr = np.unique(src[tr]); rows_va = np.unique(src[va])
    Xtr = lat[rows_tr].reshape(len(rows_tr), -1).astype(np.float32)
    Xva = lat[rows_va].reshape(len(rows_va), -1).astype(np.float32)
    Xtr = (Xtr - np.tile(mu_l, n_tok)) / np.tile(sd_l, n_tok)
    Xva = (Xva - np.tile(mu_l, n_tok)) / np.tile(sd_l, n_tok)
    probe = ridge_probe(Xtr, y[rows_tr], Xva, y[rows_va], dev)
    print(f"probe lam {probe['lam']} val R2 {probe['val_r2_mean']:.4f} "
          f"per-dim {['%.3f' % r for r in probe['val_r2_per_dim']]}", flush=True)

    # probe on 1-step PREDICTED latents (plannability signal)
    with torch.no_grad():
        preds = []
        for i in range(0, len(va), 256):
            sl = va[i:i + 256]
            z = get(src[sl]) + model(get(src[sl]), A_t[sl])
            preds.append(z.reshape(len(sl), -1).cpu())
        Xp = torch.cat(preds) - probe["x_mean"]
        yp = Xp @ probe["W"] + probe["y_mean"]
        yv = torch.from_numpy(y[src[va] + 1]).float()
        r2p = 1 - ((yp - yv).square().sum(0) / (yv - yv.mean(0)).square().sum(0))
    print(f"probe on predicted z: R2 {['%.3f' % r for r in r2p.tolist()]}", flush=True)

    out_dir = f"{args.out_base}_{args.arm}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": best_sd, "mu": mu_l, "sd": sd_l,
                "probe": {k: (v if not torch.is_tensor(v) else v)
                          for k, v in probe.items()},
                "cfg": {"arm": args.arm, "n_tokens": n_tok, "in_dim": in_dim,
                        "enc_dir": enc_dir, "act_dim": 4, "params": n_par}},
               os.path.join(out_dir, f"seed{args.seed}.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "params": n_par, "best_ep": best_ep,
                   "val_mse": best[0], "carry_mse": best[1],
                   "ratio": best[0] / best[1], "rollout": roll,
                   "probe_lam": probe["lam"], "probe_val_r2": probe["val_r2_mean"],
                   "probe_r2_per_dim": probe["val_r2_per_dim"],
                   "probe_r2_predicted": r2p.tolist(),
                   "note": "angle probed as cos/sin (planner reads atan2); "
                           "targets raw px units"}, f, indent=1, default=float)
    print(f"SAVED {out_dir}", flush=True)


if __name__ == "__main__":
    main()
