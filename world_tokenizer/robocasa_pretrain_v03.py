"""E3 (V0.3.md): v0.3 objective — detail pressure via masked per-patch feature decoding.

v0.2's MMPerceiverMC gives the vision latent no reason to keep object/scene detail: the
vision-side target is 16-d state and the reverse target is tgt_v(patch).mean() — a mean
over all K*196 patches, where object detail dies by construction (action-readout 0.335 vs
0.413 in the same patches; hybrid no-lift downstream). This variant adds the missing term:

  PatchDecoder — selected per-(cam,patch) position queries cross-attend to the VISION-ONLY
  latent set (the 8 queries the policy would consume, pre-pooling) and must reconstruct
  those patches' EMA-target features (masked subset each step, JEPA/MAE-style, feature
  space — we never need pixels). Self-distill variant (a): targets = own tgt_v per-patch.

Gates printed per variant (GATE line + results.json):
  state R^2 (hold >= v0.2's 0.673) | RankMe (no collapse) | patch-recon val MSE |
  action-readout ladder (proprio / raw-coarse / v0.2-z / v0.3-z, same protocol) |
  SeeSE3 probes (mutual-kNN + dSE(3), reused from seese3_probe)

    python -m world_tokenizer.robocasa_pretrain_v03 --lamb-patch 0.5 2.0 \
      --dataset /mnt/nas/data/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_pretraining.backbone.vit import CrossAttention  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from world_tokenizer.mm_perceiver import _mlp  # noqa: E402
from world_tokenizer.robocasa_pretrain import MMPerceiverMC  # noqa: E402
from world_tokenizer.train_chunks import probe_r2, rankme  # noqa: E402
from world_tokenizer.seese3_probe import (  # noqa: E402
    Z_CACHE, FRAME_IDX, coarse_pool, mutual_neighborhood, se3_readout, EE_POS)


class PatchDecoder(nn.Module):
    """Position queries -> cross-attend to the latent set -> per-patch target features."""

    def __init__(self, d, depth=2, n_heads=8):
        super().__init__()
        self.ca = nn.ModuleList([CrossAttention(d, d, n_heads) for _ in range(depth)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.ffn = nn.ModuleList([_mlp(d, d, 4 * d) for _ in range(depth)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(depth)])
        self.head = nn.Linear(d, d)

    def forward(self, q, zset):
        x = q
        for ca, n1, ffn, n2 in zip(self.ca, self.n1, self.ffn, self.n2):
            x = x + ca(n1(x), zset)
            x = x + ffn(n2(x))
        return self.head(x)


class MMPerceiverMCPatch(MMPerceiverMC):
    def __init__(self, n_cams=3, lamb_patch=1.0, mask_ratio=0.25, dec_depth=2, **kw):
        super().__init__(n_cams=n_cams, **kw)
        d = self.proj_v.out_features
        self.dec_pos = nn.Parameter(torch.randn(n_cams, self.n_patch, d) * 0.02)
        self.dec = PatchDecoder(d, depth=dec_depth)
        self.lamb_patch, self.mask_ratio = lamb_patch, mask_ratio

    def forward(self, patch, state):
        ctx = self._context(patch, state)
        with torch.no_grad():
            tv_pp = self.tgt_v(patch)                            # [B,K,196,d] per-patch
            tv = tv_pp.mean((1, 2))
            ts = self.tgt_s(state)
        zv_set = self.fuse(ctx, self._mask(True, patch.device), pool=False)  # [B,M,d]
        z_from_v = zv_set.mean(1)
        z_from_s = self.fuse(ctx, self._mask(False, patch.device))
        inv = (self.pred_s(z_from_v) - ts).square().mean() \
            + (self.pred_v(z_from_s) - tv).square().mean()

        # masked per-patch decoding from the vision-only latent SET (the detail pressure)
        B = patch.shape[0]
        n_tot = self.n_cams * self.n_patch
        n_sel = max(1, int(self.mask_ratio * n_tot))
        sel = torch.randperm(n_tot, device=patch.device)[:n_sel]
        ic, ip = sel // self.n_patch, sel % self.n_patch
        q = self.dec_pos[ic, ip].unsqueeze(0).expand(B, -1, -1)  # [B,n_sel,d]
        pred_pp = self.dec(q, zv_set)
        patch_loss = (pred_pp - tv_pp[:, ic, ip]).square().mean()

        ev = self.proj_v(patch).mean((1, 2))
        es = self.proj_s(state)
        z_full = self.fuse(ctx)
        sig = self.sigreg(ev) + self.sigreg(es) + self.sigreg(z_full)
        loss = inv + self.lamb * sig + self.lamb_patch * patch_loss
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "patch": patch_loss.detach(), "z": z_full.detach()}


def embed_all(model, patch, state, dev, bs=512):
    zs = []
    for i in range(0, len(state), bs):
        p = torch.from_numpy(patch[i:i + bs].astype(np.float32)).to(dev)
        s = torch.from_numpy(state[i:i + bs]).to(dev)
        zs.append(model.embed_vision(p, s).cpu().numpy())
    return np.concatenate(zs)


def action_readout(rep, y, ep, train_eps, test_eps):
    """RidgeCV rep -> flattened commanded action sequence; episode-held-out R^2."""
    tr, te = np.isin(ep, train_eps), np.isin(ep, test_eps)
    m = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(rep[tr], y[tr])
    return float(r2_score(y[te], m.predict(rep[te]),
                          multioutput="variance_weighted"))


def build_action_targets(dataset, ep, t, n_steps=5):
    """y[i] = concat of the commanded 12-D actions at frames t..t+n_steps-1
    (per-frame, so the stride-5 sample spacing doesn't thin the sequence)."""
    import glob
    import pandas as pd
    frames = {}
    for pq in sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "*.parquet"))):
        df = pd.read_parquet(pq, columns=["action", "episode_index"])
        frames[int(df["episode_index"].iloc[0])] = np.stack(
            df["action"].to_numpy()).astype(np.float32)
    ys, valid = [], []
    for e, tt in zip(ep, t):
        a = frames[int(e)]
        if tt + n_steps <= len(a):
            ys.append(a[tt:tt + n_steps].reshape(-1))
            valid.append(True)
        else:
            ys.append(np.zeros(a.shape[1] * n_steps, dtype=np.float32))
            valid.append(False)
    return np.stack(ys), np.array(valid)


def gates(model, d, z_cached_v02, dataset, dev):
    """Full gate battery on a trained model; references computed with the same protocol."""
    patch, state_raw, ep, t = d["patch"], d["state"].astype(np.float32), d["ep"], d["t"]
    mu, sd = state_raw.mean(0), state_raw.std(0) + 1e-6
    state = (state_raw - mu) / sd
    is_test = (ep % 10 == 0)
    tr, te = ~is_test, is_test
    train_eps = np.unique(ep[~is_test]); test_eps = np.unique(ep[is_test])

    zv = embed_all(model, patch, state, dev)
    res = {"zv_rankme": rankme(zv[te]),
           "zv_r2_state": probe_r2(zv[tr], state[tr], zv[te], state[te])}

    # action-readout ladder (same protocol for every representation)
    y, valid = build_action_targets(dataset, ep, t)
    coarse = coarse_pool(patch)
    lad = {}
    for name, rep in [("proprio", state_raw), ("raw_patch_coarse", coarse),
                      ("z_v02", z_cached_v02), ("z_this", zv)]:
        lad[name] = action_readout(rep[valid], y[valid], ep[valid],
                                   train_eps, test_eps)
    res["action_readout"] = lad

    # SeeSE3 probes (this model vs references)
    see = {}
    for name, rep in [("raw_patch_coarse", coarse), ("z_v02", z_cached_v02),
                      ("z_this", zv)]:
        ov, ch, n_ep = mutual_neighborhood(rep, state_raw[:, EE_POS], ep, k=5)
        # delta=1 (0.25 s) is the discriminative horizon: E1 found the raw-vs-fused gap
        # is 0.497 vs 0.368 there and nearly closed (0.599 vs 0.590) by 1.0 s
        see[name] = {"mutual_knn": ov, "chance": ch,
                     "dse3_d1": se3_readout(rep, state_raw, ep, t, 1,
                                            train_eps, test_eps)}
    res["seese3"] = see
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache",
                    default="/mnt/nas/data/robocasa/kepler_cache/pnp_patches.npz")
    ap.add_argument("--out-base",
                    default="/mnt/nas/data/robocasa/kepler_ckpt/pnp_mc3_v03")
    ap.add_argument("--lamb-patch", type=float, nargs="+", default=[0.5, 2.0])
    ap.add_argument("--mask-ratio", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gates", action="store_true",
                    help="skip the pnp-specific gate battery (v0.2 z ref / FRAME_IDX "
                         "are pnp-only) — for non-pnp caches, e.g. square")
    args = ap.parse_args()

    d = np.load(args.cache)
    patch, state_raw, ep = d["patch"], d["state"].astype(np.float32), d["ep"]
    mu, sd = state_raw.mean(0), state_raw.std(0) + 1e-6
    state = (state_raw - mu) / sd
    is_test = (ep % 10 == 0)
    K = patch.shape[1]
    print(f"{len(ep)} samples, K={K}, test {is_test.sum()}", flush=True)

    # v0.2 z reference straight from the policy z cache (same frames)
    if not args.no_gates:
        fc = json.load(open(FRAME_IDX))
        zc_meta = json.load(open(f"{Z_CACHE}/meta.json"))
        zmm = np.memmap(f"{Z_CACHE}/z.f32", dtype=np.float32, mode="r",
                        shape=(fc["total_frames"], zc_meta["d"]))
        rows = np.array([fc["episodes"][str(e)]["start"] for e in ep]) + d["t"]
        z_v02 = np.array(zmm[rows])

    dev = "cuda"
    tr_idx = np.flatnonzero(~is_test)
    for lp in args.lamb_patch:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = MMPerceiverMCPatch(
            n_cams=K, d=args.d, state_dim=state.shape[1], n_queries=args.queries,
            lamb_patch=lp, mask_ratio=args.mask_ratio).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"\n==== variant lamb_patch={lp} ====", flush=True)
        for epoch in range(args.epochs):
            model.train()
            np.random.shuffle(tr_idx)
            tot = np.zeros(3); nb = 0
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

        model.eval()
        res = None
        if not args.no_gates:
            res = gates(model, d, z_v02, args.dataset, dev)
            print("GATE:", json.dumps(res, default=float)[:800], flush=True)

        out_dir = f"{args.out_base}_lp{lp}"
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"seed{args.seed}.pt"))
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump({"args": vars(args), "lamb_patch": lp, "gate": res,
                       "state_mu": mu.tolist(), "state_sd": sd.tolist()},
                      f, indent=1, default=float)
        print(f"SAVED {out_dir}", flush=True)


if __name__ == "__main__":
    main()
