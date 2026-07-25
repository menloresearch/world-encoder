"""v0.2 Build 2 — next-embedding predictor on the FROZEN v0.1 encoder (V0.2.md "Build 2").

Freeze v0.1, encode the per-frame FULL multimodal latent (set of M queries, or pooled), and train a
small predictor z_t -> z_{t+d} (horizon-conditioned). This is the temporal capability relocated to a
predictor (§2.2), the loss-#4 precursor, and the FLARE de-risk. Pre-check A (precheck_predict.py) already
showed a simple predictor beats carry-forward ~25-35%; this trains the real, reusable artifact.

Design notes:
- **Encoder frozen** (guardrail): training the encoder is what broke v0.2. Only the predictor learns.
- **Predict the latent SET** (guardrail), not a mean-pooled vector, so it keeps spatial/structural detail.
  `--latent pooled` runs the pre-check-validated pooled variant for comparison.
- **Plain MSE, no EMA/SIGReg by default.** LeWM needs SIGReg/EMA because it co-trains its encoder (target
  can collapse). Our target is a FIXED frozen-encoder latent -> no collapse risk -> MSE suffices.
  `--sigreg-w > 0` adds SIGReg on the predicted marginal if we want it.
- Latents are precomputed ONCE through the frozen encoder, then the predictor trains on tensors (fast).
- Eval mirrors Pre-check A: predictor MSE / carry-forward MSE per delta (<1 = beats carry), R2, RankMe.

  .venv/bin/python -m world_tokenizer.train_predictor --v0 .../phase1/kuka/seed0.pt --cfgs 6 7 \
      --latent set --tag kuka_set --out results/temporal/predictor_kuka_set.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver2 import MMPerceiverChunks  # noqa: E402
from world_tokenizer.window_loader import make_window_loader  # noqa: E402
from world_tokenizer.train_chunks import rankme               # noqa: E402


def _mlp(din, dout, hid):
    return nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, dout))


@torch.no_grad()
def embed_set(m, rgb, motor, mm, ee, em):
    """v0.1 full multimodal latent SET (nothing hidden, queries NOT pooled). [B, M, d]."""
    ctx = m._context(rgb, m.motor_feats(motor, mm), ee)
    return m.fuse(ctx, m._attn_mask(mm.any(-1), em), pool=False)


@torch.no_grad()
def encode_ticks(m, loader, dev, ticks, act_tick):
    """Full latent set at each tick -> {tick: [N,M,d]}; plus the action = symlog joint velocity
    (motor ch2, dq) at the input tick act_tick -> [N, 8] (N2 conditioning signal)."""
    m.eval()
    out = {k: [] for k in ticks}
    acts = []
    for b in loader:
        rgb = b["rgb"].to(dev); motor = b["motor"].to(dev); mm = b["motor_mask"].to(dev)
        ee = b["ee"].to(dev); em = b["ee_mask"].to(dev)
        for k in ticks:
            out[k].append(embed_set(m, rgb[:, k], motor[:, k], mm[:, k], ee[:, k], em[:, k]).float())
        acts.append((motor[:, act_tick, :, 2] * mm[:, act_tick, :, 2]).float())  # dq, zeroed where invalid
    return {k: torch.cat(v) for k, v in out.items()}, torch.cat(acts)


class Predictor(nn.Module):
    """Horizon-conditioned next-embedding predictor. latent='set' -> self-attn over the M query
    latents; latent='pooled' -> MLP on the pooled vector. AdaLN-free for now; a horizon embedding is
    added (an AdaLN action hook is the N2 / loss-#4 extension)."""
    def __init__(self, d=256, latent="set", n_horizons=3, depth=3, heads=8, hid=1024,
                 use_action=False, act_dim=8):
        super().__init__()
        self.latent = latent
        self.use_action = use_action
        self.hemb = nn.Embedding(n_horizons, d)
        if use_action:
            self.act = _mlp(act_dim, d, hid)          # N2: action (joint dq) -> conditioning vector
        if latent == "set":
            self.blocks = nn.ModuleList([nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d),
                attn=nn.MultiheadAttention(d, heads, batch_first=True),
                ln2=nn.LayerNorm(d), ffn=_mlp(d, d, hid))) for _ in range(depth)])
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        else:
            self.net = _mlp(d, d, hid)
            self.net2 = _mlp(d, d, hid)

    def forward(self, z, didx, a=None):
        cond = self.hemb(didx)                               # [B, d] (horizon)
        if self.use_action:
            cond = cond + self.act(a)                        # + action conditioning (additive AdaLN-lite)
        if self.latent == "set":
            x = z + cond.unsqueeze(1)                        # broadcast cond over M tokens
            for b in self.blocks:
                att, _ = b["attn"](b["ln1"](x), b["ln1"](x), b["ln1"](x), need_weights=False)
                x = x + att
                x = x + b["ffn"](b["ln2"](x))
            return self.head(x)                              # [B, M, d]
        x = z + cond                                         # [B, d]
        return self.net2(x + self.net(x))


def pool(z):
    return z.mean(1) if z.dim() == 3 else z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True, help="frozen v0.1 encoder (phase1)")
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--latent", default="set", choices=["set", "pooled"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/mnt/nas/data/RH20T/checkpoints/predictor")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4, help="input latent = tick ctx-1")
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--hid", type=int, default=1024)
    ap.add_argument("--sigreg-w", type=float, default=0.0, help=">0 adds SIGReg on predicted marginal")
    ap.add_argument("--patience", type=int, default=20, help="early-stop patience on val MSE")
    ap.add_argument("--use-action", action="store_true",
                    help="N2: condition the predictor on the action (joint dq at input tick)")
    ap.add_argument("--load-batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    for _ in range(15):
        if torch.cuda.is_available():
            break
        time.sleep(10)
    assert torch.cuda.is_available(), "no CUDA"
    dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cl = args.ctx - 1
    ticks = sorted(set([cl] + [cl + d for d in args.deltas]))

    enc = MMPerceiverChunks(d=256, n_queries=8).to(dev)
    enc.load_state_dict(torch.load(args.v0, map_location=dev))
    for p in enc.parameters():
        p.requires_grad = False
    print(f"[{args.tag}] frozen v0.1={args.v0} | latent={args.latent} | cfgs={args.cfgs} "
          f"| cl={cl} deltas={args.deltas}", flush=True)

    tr, te, _ = make_window_loader(args.cache_dir, tuple(args.cfgs), window=args.window,
                                   stride=args.stride, batch=args.load_batch, num_workers=args.workers)
    Ztr, Atr = encode_ticks(enc, tr, dev, ticks, cl)
    Zte, Ate = encode_ticks(enc, te, dev, ticks, cl)
    n_tr, n_te = Ztr[cl].shape[0], Zte[cl].shape[0]
    act_dim = Atr.shape[1]
    if args.use_action:                                       # standardize the action (dq) too
        amu, asd = Atr.mean(0), Atr.std(0).clamp_min(1e-6)
        Atr, Ate = (Atr - amu) / asd, (Ate - amu) / asd
    if args.latent == "pooled":
        Ztr = {k: pool(v) for k, v in Ztr.items()}
        Zte = {k: pool(v) for k, v in Zte.items()}
    print(f"[{args.tag}] encoded: train {n_tr} test {n_te} | latent shape {tuple(Ztr[cl].shape[1:])}",
          flush=True)

    # standardize per-dim (stats from train) so the regression is well-conditioned + metrics are
    # comparable to Pre-check A (which StandardScaler'd x/y). All Z below live in standardized space.
    allz = torch.cat([Ztr[k].reshape(-1, Ztr[k].shape[-1]) for k in ticks])
    mu, sd = allz.mean(0), allz.std(0).clamp_min(1e-6)
    Ztr = {k: (v - mu) / sd for k, v in Ztr.items()}
    Zte = {k: (v - mu) / sd for k, v in Zte.items()}

    # carve a val split by WINDOW index (a window's pairs stay together) for early stopping
    K = len(args.deltas)
    vperm = torch.randperm(n_tr, device=dev)
    nv = max(64, int(0.1 * n_tr))
    vidx, tidx = vperm[:nv], vperm[nv:]

    def build(idx):
        X = torch.cat([Ztr[cl][idx] for _ in args.deltas])
        Y = torch.cat([Ztr[cl + d][idx] for d in args.deltas])
        D = torch.cat([torch.full((len(idx),), i, device=dev, dtype=torch.long) for i in range(K)])
        A = torch.cat([Atr[idx] for _ in args.deltas])
        return X, Y, D, A
    Xtr, Ytr, Dtr, Atrp = build(tidx)
    Xval, Yval, Dval, Avalp = build(vidx)
    npairs = Xtr.shape[0]

    net = Predictor(d=256, latent=args.latent, n_horizons=K, depth=args.depth, hid=args.hid,
                    use_action=args.use_action, act_dim=act_dim).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"[{args.tag}] predictor params {nparams/1e6:.2f}M | {npairs} train / {Xval.shape[0]} val pairs",
          flush=True)

    sig = None
    if args.sigreg_w > 0:
        from stable_pretraining.methods.lejepa import SlicedEppsPulley
        sig = SlicedEppsPulley(num_slices=512).to(dev)

    t0 = time.time()
    best_v, best_state, bad = 1e9, None, 0
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(npairs, device=dev)
        ep_mse = ep_sg = 0.0; nb = 0
        for j in range(0, npairs, args.batch):
            idx = perm[j:j + args.batch]
            pred = net(Xtr[idx], Dtr[idx], Atrp[idx] if args.use_action else None)
            mse = (pred - Ytr[idx]).square().mean()
            loss = mse
            if sig is not None:
                sg = sig(pool(pred))
                loss = loss + args.sigreg_w * sg
                ep_sg += float(sg.detach())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ep_mse += float(mse.detach()); nb += 1
        net.eval()
        with torch.no_grad():
            vmse = float((net(Xval, Dval, Avalp if args.use_action else None) - Yval).square().mean())
        if vmse < best_v - 1e-5:
            best_v, bad = vmse, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop e{ep} (val {vmse:.4f}, best {best_v:.4f})", flush=True)
                break
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"  e{ep} train mse {ep_mse/max(nb,1):.4f} val {vmse:.4f}"
                  + (f" sig {ep_sg/max(nb,1):.3f}" if sig is not None else ""), flush=True)
    if best_state:
        net.load_state_dict(best_state)

    # ---- eval: per-delta predictor vs carry-forward (pooled space, comparable to Pre-check A) ----
    net.eval()
    res = {"v0": args.v0, "latent": args.latent, "use_action": args.use_action, "cfgs": args.cfgs,
           "ctx": args.ctx, "deltas": args.deltas, "n_train": n_tr, "n_test": n_te,
           "predictor_params_M": nparams / 1e6, "sigreg_w": args.sigreg_w, "per_delta": {}}
    x_te = Zte[cl]
    xp_te = pool(x_te).cpu().numpy()
    print(f"\n[{args.tag}] === TRAINED PREDICTOR vs carry-forward (pooled MSE ratio <1 beats) ===",
          flush=True)
    print(f"{'d':>2} | {'predR2':>8} {'naiveR2':>8} | {'pred/carry MSE':>14} {'RankMe':>7}", flush=True)
    for i, d in enumerate(args.deltas):
        y_te = Zte[cl + d]
        with torch.no_grad():
            didx = torch.full((n_te,), i, device=dev, dtype=torch.long)
            pred = net(x_te, didx, Ate if args.use_action else None)
        pp, yp = pool(pred).cpu().numpy(), pool(y_te).cpu().numpy()
        pred_mse = float(((pp - yp) ** 2).mean())
        carry_mse = float(((xp_te - yp) ** 2).mean())
        pred_r2 = float(r2_score(yp, pp, multioutput="uniform_average"))
        naive_r2 = float(r2_score(yp, xp_te, multioutput="uniform_average"))
        rm = rankme(pp)
        res["per_delta"][str(d)] = {"pred_r2": pred_r2, "naive_r2": naive_r2, "pred_mse": pred_mse,
                                    "carry_mse": carry_mse, "pred_over_carry_mse": pred_mse / carry_mse,
                                    "rankme_pred": rm}
        print(f"{d:>2} | {pred_r2:>8.3f} {naive_r2:>8.3f} | {pred_mse/carry_mse:>14.3f} {rm:>7.1f}",
              flush=True)

    best = min(v["pred_over_carry_mse"] for v in res["per_delta"].values())
    res["best_pred_over_carry_mse"] = best
    print(f"[{args.tag}] best pred/carry MSE {best:.3f} "
          f"({'beats' if best < 1 else 'does NOT beat'} carry-forward) | {(time.time()-t0)/60:.1f} min",
          flush=True)

    run_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"model": net.state_dict(), "args": vars(args)}, os.path.join(run_dir, f"seed{args.seed}.pt"))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
