"""FSQ variant of the v0.3 PushT encoder (claim C rung 0, 2026-08-04).

CompACT-style finite scalar quantization on the 8-token vision latent set:
per token, down-proj d->len(levels) dims, bound with tanh, round to the level
grid (straight-through), up-proj back to d. All vision-latent consumers (inv
prediction, patch decoder, embed path) see QUANTIZED tokens, so the dynamics
variant trains on the same discrete latents it will plan over. Levels
[8,5,5,5,5] = 5000 codes/token. Everything else = robocasa_pretrain_v03
verbatim (lp2.0, 40 ep, d 256, 8 queries, state-blind embed path).

  CUDA_VISIBLE_DEVICES=6 python -m world_tokenizer.pusht_fsq \
    --cache /mnt/nas/data/robocasa/kepler_cache/pusht_patches.npz \
    --out-base /mnt/nas/data/robocasa/kepler_ckpt/pusht_mc1_v03_fsq
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from world_tokenizer.robocasa_pretrain_v03 import MMPerceiverMCPatch

LEVELS = [8, 5, 5, 5, 5]


class FSQ(nn.Module):
    def __init__(self, d, levels=LEVELS):
        super().__init__()
        self.down = nn.Linear(d, len(levels))
        self.up = nn.Linear(len(levels), d)
        self.register_buffer("L", torch.tensor(levels, dtype=torch.float32))

    def forward(self, x):                      # x [..., d] -> quantized [..., d]
        h = torch.tanh(self.down(x)) * (self.L - 1) / 2
        hq = h + (h.round() - h).detach()      # straight-through round
        return self.up(hq / ((self.L - 1) / 2))


class MMPerceiverMCPatchFSQ(MMPerceiverMCPatch):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fsq = FSQ(self.proj_v.out_features)

    def forward(self, patch, state):
        ctx = self._context(patch, state)
        with torch.no_grad():
            tv_pp = self.tgt_v(patch)
            tv = tv_pp.mean((1, 2))
            ts = self.tgt_s(state)
        zv_set = self.fsq(self.fuse(ctx, self._mask(True, patch.device), pool=False))
        z_from_v = zv_set.mean(1)
        z_from_s = self.fuse(ctx, self._mask(False, patch.device))
        inv = (self.pred_s(z_from_v) - ts).square().mean() \
            + (self.pred_v(z_from_s) - tv).square().mean()
        B = patch.shape[0]
        n_tot = self.n_cams * self.n_patch
        n_sel = max(1, int(self.mask_ratio * n_tot))
        sel = torch.randperm(n_tot, device=patch.device)[:n_sel]
        ic, ip = sel // self.n_patch, sel % self.n_patch
        q = self.dec_pos[ic, ip].unsqueeze(0).expand(B, -1, -1)
        patch_loss = (self.dec(q, zv_set) - tv_pp[:, ic, ip]).square().mean()
        ev = self.proj_v(patch).mean((1, 2))
        es = self.proj_s(state)
        z_full = self.fuse(ctx)
        sig = self.sigreg(ev) + self.sigreg(es) + self.sigreg(z_full)
        loss = inv + self.lamb * sig + self.lamb_patch * patch_loss
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "patch": patch_loss.detach(), "z": z_full.detach()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--lamb-patch", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"

    d = np.load(args.cache)
    patch, state_raw, ep = d["patch"], d["state"].astype(np.float32), d["ep"]
    mu, sd = state_raw.mean(0), state_raw.std(0) + 1e-6
    state = (state_raw - mu) / sd
    K = patch.shape[1]
    tr_idx = np.flatnonzero(ep % 10 != 0)
    print(f"{len(ep)} samples, K={K}, levels {LEVELS}", flush=True)

    model = MMPerceiverMCPatchFSQ(
        n_cams=K, d=256, state_dim=state.shape[1], n_queries=8,
        lamb_patch=args.lamb_patch, mask_ratio=0.25).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(tr_idx)
        tot, nb = np.zeros(3), 0
        for i in range(0, len(tr_idx), args.batch):
            sl = tr_idx[i:i + args.batch]
            p = torch.from_numpy(patch[sl].astype(np.float32)).to(dev)
            s = torch.from_numpy(state[sl]).to(dev)
            out = model(p, s)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            model.update_target()
            tot += [float(out["loss"]), float(out["inv"]), float(out["patch"])]
            nb += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"ep {epoch}: loss {tot[0]/nb:.4f} inv {tot[1]/nb:.4f} "
                  f"patch {tot[2]/nb:.4f}", flush=True)

    out_dir = f"{args.out_base}_lp{args.lamb_patch}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"seed{args.seed}.pt"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"args": {**vars(args), "queries": 8, "d": 256, "fsq": LEVELS},
                   "state_mu": mu.tolist(), "state_sd": sd.tolist()},
                  f, indent=1, default=float)
    print(f"SAVED {out_dir}", flush=True)


if __name__ == "__main__":
    main()
