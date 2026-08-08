"""PushT rung-0 HISTORY-2 dynamics (harness contingency knob, PROGRESS.md pt. 21).

Same trunk as pusht_dynamics.LatentDynamics, but conditions on TWO latent
frames one dyn-step (0.2 s) apart: p(z_{t+1} | z_{t-1}, z_t, a_t). Rationale:
single-frame Markov dynamics cannot observe velocity; on PushT that is
structural model error the planner exploits. Everything else pt.-18-verbatim
(residual in standardized space, 4-D interval action, AdamW 3e-4, batch 128,
<=60 ep, ep%10==0 val, early-stop 10, seed 0, carry-forward kill-rule).

Training triples: pair j is usable iff the preceding pair (row src[j]-1) is
also valid — z_{t-1} then exists at the right stride. Ckpts -> NAS
pusht_dyn2_<arm>/. Cost-head probe refit identically (current-frame latents).

  CUDA_VISIBLE_DEVICES=2 python -m world_tokenizer.pusht_dynamics_h2 --arm z8
  CUDA_VISIBLE_DEVICES=1 python -m world_tokenizer.pusht_dynamics_h2 --arm dense
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from world_tokenizer.pusht_dynamics import (ACT_DIR, CACHE, CKPT_ROOT, ENC_DIR,
                                            encode_zset, ridge_probe)


class LatentDynamicsH2(nn.Module):
    """Two-frame conditioned trunk; residual read from current-frame positions."""

    def __init__(self, n_tokens, in_dim, d=256, depth=4, heads=8, ffn=1024, act_dim=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d) if in_dim != d else nn.Identity()
        self.out_proj = nn.Linear(d, in_dim) if in_dim != d else nn.Linear(d, d)
        self.pos = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.frame = nn.Parameter(torch.randn(2, 1, d) * 0.02)
        self.act = nn.Sequential(nn.Linear(act_dim, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, ffn, batch_first=True,
                                           norm_first=True, dropout=0.0)
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.n_tokens = n_tokens

    def forward(self, zp, zc, a):
        # zp/zc: [B, n_tokens, in_dim] standardized -> residual for zc [B, n_tokens, in_dim]
        xp = self.in_proj(zp) + self.pos + self.frame[0]
        xc = self.in_proj(zc) + self.pos + self.frame[1]
        x = torch.cat([xp, xc, self.act(a).unsqueeze(1)], 1)
        h = self.trunk(x)
        return self.out_proj(h[:, self.n_tokens:2 * self.n_tokens])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ENC_DIR) + ["dense"])
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-base", default=os.path.join(CKPT_ROOT, "pusht_dyn2"))
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

    enc_dir = None
    if args.arm == "dense":
        lat = patch[:, 0]
    else:
        lat, enc_dir = encode_zset(args.arm, patch, state, dev)
    n_tok, in_dim = lat.shape[1], lat.shape[2]
    print(f"arm={args.arm} latents [{N},{n_tok},{in_dim}]", flush=True)

    acts_by_ep = {}
    for e in np.unique(ep):
        df = pd.read_parquet(os.path.join(ACT_DIR, f"episode_{e:06d}.parquet"))
        acts_by_ep[int(e)] = np.stack(df["action"].to_numpy()).astype(np.float32)
    src = np.flatnonzero((ep[1:] == ep[:-1]) & (t[1:] - t[:-1] == 2))
    A = np.stack([np.concatenate([acts_by_ep[int(ep[i])][t[i]],
                                  acts_by_ep[int(ep[i])][t[i] + 1]]) for i in src])
    A = A / 256.0 - 1.0
    has_prev = np.isin(src - 1, src)                    # z_{t-1} exists at right stride
    is_val = (ep[src] % 10 == 0)
    tr = np.flatnonzero(~is_val & has_prev)
    va = np.flatnonzero(is_val & has_prev)
    print(f"{len(src)} pairs -> {has_prev.sum()} triples (train {len(tr)}, val {len(va)})",
          flush=True)

    tr_rows = np.unique(np.concatenate([src[tr] - 1, src[tr], src[tr] + 1]))
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

    model = LatentDynamicsH2(n_tok, in_dim).to(dev)
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
                zp, zc, z1 = get(src[sl] - 1), get(src[sl]), get(src[sl] + 1)
                r = z1 - zc
                pr = model(zp, zc, A_t[sl])
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
            zp, zc, z1 = get(src[sl] - 1), get(src[sl]), get(src[sl] + 1)
            loss = (model(zp, zc, A_t[sl]) - (z1 - zc)).square().mean()
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

    # open-loop rollout MSE vs carry (val, standardized space), h2 state threaded
    roll = {}
    with torch.no_grad():
        for H in [1, 2, 4, 8]:
            ok = np.flatnonzero(np.array(
                [is_val[j] and has_prev[j] and j + H - 1 < len(src)
                 and ep[src[j]] == ep[src[j + H - 1]]
                 and all(src[j + k] + 1 == src[j + k + 1] for k in range(H - 1))
                 for j in range(len(src) - H + 1)]))
            se_p, se_c, m = 0.0, 0.0, 0
            for i in range(0, len(ok), 256):
                sl = ok[i:i + 256]
                zp, zc = get(src[sl] - 1), get(src[sl])
                for k in range(H):
                    r = model(zp, zc, A_t[sl + k])
                    zp, zc = zc, zc + r
                zt = get(src[sl + H - 1] + 1)
                z0 = get(src[sl])
                se_p += float((zc - zt).square().sum())
                se_c += float((z0 - zt).square().sum())
                m += zt.numel()
            roll[H] = {"pred": se_p / m, "carry": se_c / m,
                       "ratio": (se_p / m) / (se_c / m), "n": int(len(ok))}
            print(f"rollout H={H}: pred {roll[H]['pred']:.5f} carry "
                  f"{roll[H]['carry']:.5f} ratio {roll[H]['ratio']:.3f}", flush=True)

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

    out_dir = f"{args.out_base}_{args.arm}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": best_sd, "mu": mu_l, "sd": sd_l,
                "probe": {k: (v if not torch.is_tensor(v) else v)
                          for k, v in probe.items()},
                "cfg": {"arm": args.arm, "n_tokens": n_tok, "in_dim": in_dim,
                        "enc_dir": enc_dir, "act_dim": 4, "params": n_par,
                        "history": 2}},
               os.path.join(out_dir, f"seed{args.seed}.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "params": n_par, "best_ep": best_ep,
                   "val_mse": best[0], "carry_mse": best[1],
                   "ratio": best[0] / best[1], "rollout": roll,
                   "probe_lam": probe["lam"], "probe_val_r2": probe["val_r2_mean"],
                   "probe_r2_per_dim": probe["val_r2_per_dim"],
                   "history": 2,
                   "note": "h2: two frames one dyn-step apart; angle as cos/sin"}, f,
                  indent=1, default=float)
    print(f"SAVED {out_dir}", flush=True)


if __name__ == "__main__":
    main()
