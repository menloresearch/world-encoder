"""E7 (V0.3.md): resampler pre-check — queries = target sampling rate (JQ's 07-26 design).

One query per OUTPUT tick, no pooling over time. The query content is a SINGLE learnable
scalar tiled to d — identity comes entirely from embeddings (target-tick emb + latent-slot
emb) — and multiple cross-attn layers let the shared query adapt through depth. Front-end
for the predictor: context = latent SETs of the past ctx ticks (tick pos emb added),
output = predicted latent set at EVERY future tick in one forward (vs the validated
predictor's horizon-conditioned one-at-a-time MLP).

Gate (the same one the temporal encoder failed): pred/carry MSE ratio < 1 per delta;
reference = validated kuka set predictor 0.724 / 0.674 / 0.651 @ delta 1/2/3.

    python -m world_tokenizer.train_resampler \
        --v0 /mnt/nas/data/RH20T/checkpoints/phase1/kuka/seed0.pt --cfgs 6 7 \
        --tag kuka_resampler --out results/temporal/resampler_kuka.json
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

from stable_pretraining.backbone.vit import CrossAttention  # noqa: E402

from world_tokenizer.mm_perceiver import _mlp  # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks  # noqa: E402
from world_tokenizer.train_chunks import rankme  # noqa: E402
from world_tokenizer.train_predictor import encode_ticks, pool  # noqa: E402
from world_tokenizer.window_loader import make_window_loader  # noqa: E402


class TickResampler(nn.Module):
    def __init__(self, d=256, n_slots=8, ctx=4, n_out=3, depth=3, heads=8, hid=1024):
        super().__init__()
        self.n_slots, self.ctx, self.n_out = n_slots, ctx, n_out
        self.q0 = nn.Parameter(torch.zeros(1))                 # THE single learnable param
        self.tick_in = nn.Parameter(torch.randn(ctx, d) * 0.02)
        self.tick_out = nn.Parameter(torch.randn(n_out, d) * 0.02)
        self.slot = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.ca = nn.ModuleList([CrossAttention(d, d, heads) for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ffn = nn.ModuleList([_mlp(d, d, hid) for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))

    def forward(self, zctx):                                   # [B, ctx, M, d]
        B, C, M, d = zctx.shape
        ctx_tok = (zctx + self.tick_in[None, :, None, :]
                   + self.slot[None, None, :, :]).flatten(1, 2)          # [B, C*M, d]
        q = self.q0.expand(self.n_out, M, d) \
            + self.tick_out[:, None, :] + self.slot[None, :, :]          # [K, M, d]
        x = q.flatten(0, 1).unsqueeze(0).expand(B, -1, -1)               # [B, K*M, d]
        for ca, n1, ffn, n2 in zip(self.ca, self.n1, self.ffn, self.n2):
            x = x + ca(n1(x), ctx_tok)
            x = x + ffn(n2(x))
        return self.head(x).reshape(B, self.n_out, M, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=4)
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--hid", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--load-batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/temporal/resampler_kuka.json")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cl = args.ctx - 1
    ticks = sorted(set(list(range(args.ctx)) + [cl + d for d in args.deltas]))
    assert max(ticks) < args.window

    enc = MMPerceiverChunks(d=256, n_queries=8).to(dev)
    enc.load_state_dict(torch.load(args.v0, map_location=dev))
    for p in enc.parameters():
        p.requires_grad = False
    print(f"[{args.tag}] ctx ticks 0..{cl}, deltas {args.deltas}", flush=True)

    tr, te, _ = make_window_loader(args.cache_dir, tuple(args.cfgs), window=args.window,
                                   stride=args.stride, batch=args.load_batch,
                                   num_workers=args.workers)
    Ztr, _ = encode_ticks(enc, tr, dev, ticks, cl)
    Zte, _ = encode_ticks(enc, te, dev, ticks, cl)
    n_tr, n_te = Ztr[cl].shape[0], Zte[cl].shape[0]
    M = Ztr[cl].shape[1]
    print(f"[{args.tag}] encoded train {n_tr} test {n_te} M={M}", flush=True)

    allz = torch.cat([Ztr[k].reshape(-1, Ztr[k].shape[-1]) for k in ticks])
    mu, sd = allz.mean(0), allz.std(0).clamp_min(1e-6)
    Ztr = {k: (v - mu) / sd for k, v in Ztr.items()}
    Zte = {k: (v - mu) / sd for k, v in Zte.items()}

    Xtr = torch.stack([Ztr[k] for k in range(args.ctx)], dim=1)          # [N, ctx, M, d]
    Ytr = torch.stack([Ztr[cl + d] for d in args.deltas], dim=1)         # [N, K, M, d]
    Xte = torch.stack([Zte[k] for k in range(args.ctx)], dim=1)
    Yte = torch.stack([Zte[cl + d] for d in args.deltas], dim=1)

    vperm = torch.randperm(n_tr)
    nv = max(64, int(0.1 * n_tr))
    vidx, tidx = vperm[:nv], vperm[nv:]

    model = TickResampler(d=256, n_slots=M, ctx=args.ctx, n_out=len(args.deltas),
                          depth=args.depth, hid=args.hid).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] resampler params {n_par/1e6:.2f}M (q0 is 1 scalar)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = tidx[torch.randperm(len(tidx))]
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            sl = perm[i:i + args.batch]
            pred = model(Xtr[sl].to(dev))
            loss = (pred - Ytr[sl].to(dev)).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        model.eval()
        with torch.no_grad():
            val = float((model(Xtr[vidx].to(dev)) - Ytr[vidx].to(dev)).square().mean())
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 10 == 0 or bad >= args.patience:
            print(f"ep {epoch}: train {tot/nb:.4f} val {val:.4f} (best {best_val:.4f})",
                  flush=True)
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)

    # gate eval: pooled pred vs carry-forward per delta (same protocol as train_predictor)
    model.eval()
    res = {"args": vars(args), "params": n_par, "per_delta": {}}
    with torch.no_grad():
        preds = []
        for i in range(0, n_te, args.batch):
            preds.append(model(Xte[i:i + args.batch].to(dev)).cpu())
        pred = torch.cat(preds)                                          # [N, K, M, d]
    xp = pool(Zte[cl]).cpu().numpy()                                     # carry-forward
    print(f"\n[{args.tag}] === RESAMPLER vs carry-forward (pooled MSE ratio <1 beats) ===",
          flush=True)
    for j, d_ in enumerate(args.deltas):
        yp = pool(Zte[cl + d_]).cpu().numpy()
        pp = pred[:, j].mean(1).numpy()
        pred_mse = float(((pp - yp) ** 2).mean())
        carry_mse = float(((xp - yp) ** 2).mean())
        r2 = float(r2_score(yp, pp))
        rm = rankme(pp)
        res["per_delta"][d_] = {"pred_mse": pred_mse, "carry_mse": carry_mse,
                                "pred_over_carry_mse": pred_mse / carry_mse,
                                "r2": r2, "rankme": rm}
        print(f"d={d_}: pred/carry {pred_mse/carry_mse:.3f} R2 {r2:.3f} RankMe {rm:.1f}",
              flush=True)
    best = min(v["pred_over_carry_mse"] for v in res["per_delta"].values())
    res["best_pred_over_carry_mse"] = best
    res["reference_tickt_predictor"] = {"1": 0.724, "2": 0.674, "3": 0.651}
    print(f"[{args.tag}] best {best:.3f} | ref (tick-t predictor) 0.724/0.674/0.651 | "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
