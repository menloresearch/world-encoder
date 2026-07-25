"""Kepler multicam JEPA pretraining on RoboCasa demo data (JQ pt-enc arm, in-domain).

Two stages:
  precompute — sample every --stride-th frame per episode from the DP frame cache
    (build_frame_cache.py memmaps), run frozen ViTv2 -> patch npz + 16-d proprio state.
  train — MMPerceiverMC (the validated 2-modality vision+state JEPA of mm_perceiver.py,
    vision extended to K cameras with a camera-slot embedding, exactly the Build-1
    trick) + probe gate: state R^2 from the vision-only latent vs raw-ViT pool, RankMe.

The saved state_dict keys (proj_v / cam_emb / mod / fuse) are what
diffusion_policy KeplerMultiCamEncoder._load_pretrained expects.

    python -m world_tokenizer.robocasa_pretrain --stage all \
      --dataset /mnt/nas/data/robocasa/datasets/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver import MMPerceiver  # noqa: E402
from world_tokenizer.train_chunks import probe_r2, rankme  # noqa: E402

NORM_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class MMPerceiverMC(MMPerceiver):
    """K-camera vision + 1 state token; cameras share proj_v, get a slot embedding."""

    def __init__(self, n_cams=3, **kw):
        super().__init__(**kw)
        self.n_cams = n_cams
        self.cam_emb = nn.Parameter(torch.randn(n_cams, self.proj_v.out_features) * 0.02)

    def _context(self, patch, state):
        # patch [B,K,196,768] -> tokens [B, K*196+1, d]
        vt = self.proj_v(patch) + self.cam_emb[None, :, None, :] + self.mod[0]
        vt = vt.flatten(1, 2)
        st = (self.proj_s(state) + self.mod[1]).unsqueeze(1)
        return torch.cat([vt, st], dim=1)

    def _mask(self, block_state, device):
        n_vis = self.n_cams * self.n_patch
        m = torch.zeros(self.n_queries, n_vis + 1, dtype=torch.bool, device=device)
        if block_state:
            m[:, n_vis:] = True
        else:
            m[:, :n_vis] = True
        return m

    def forward(self, patch, state):
        ctx = self._context(patch, state)
        with torch.no_grad():
            tv = self.tgt_v(patch).mean((1, 2))                  # pool over cams+patches
            ts = self.tgt_s(state)
        z_from_v = self.fuse(ctx, self._mask(True, patch.device))
        z_from_s = self.fuse(ctx, self._mask(False, patch.device))
        inv = (self.pred_s(z_from_v) - ts).square().mean() \
            + (self.pred_v(z_from_s) - tv).square().mean()
        ev = self.proj_v(patch).mean((1, 2))
        es = self.proj_s(state)
        z_full = self.fuse(ctx)
        sig = self.sigreg(ev) + self.sigreg(es) + self.sigreg(z_full)
        loss = inv + self.lamb * sig
        return {"loss": loss, "inv": inv.detach(), "sig": sig.detach(),
                "z": z_full.detach()}

    @torch.no_grad()
    def embed_vision(self, patch, state):
        """Vision-only latent (state hidden) — the probe/eval latent."""
        ctx = self._context(patch, state)
        return self.fuse(ctx, self._mask(True, patch.device))


def stage_precompute(args):
    fc_dir = os.path.join(os.path.dirname(args.dataset.rstrip("/")), "frame_cache")
    idx = json.load(open(os.path.join(fc_dir, "index.json")))
    n, h, w = idx["total_frames"], idx["height"], idx["width"]
    cams = sorted(idx["cams"].items())                            # stable cam order
    mms = [np.memmap(os.path.join(fc_dir, f), dtype=np.uint8, mode="r",
                     shape=(n, h, w, 3)) for _, f in cams]
    print("cams:", [c for c, _ in cams], flush=True)

    import pandas as pd
    from world_tokenizer.model import load_vitv2
    dev = "cuda"
    vit = load_vitv2(pretrained=True).to(dev).eval()

    rows = []                                                     # (ep, t, abs_idx, state)
    for pq in sorted(glob.glob(os.path.join(args.dataset, "data", "chunk-*", "*.parquet"))):
        df = pd.read_parquet(pq, columns=["observation.state", "episode_index",
                                          "frame_index"])
        ep = int(df["episode_index"].iloc[0])
        start = idx["episodes"][str(ep)]["start"]
        for t in range(0, len(df), args.stride):
            rows.append((ep, t, start + t, np.asarray(df["observation.state"].iloc[t],
                                                      dtype=np.float32)))
    print(f"{len(rows)} samples (stride {args.stride})", flush=True)

    c0, c1 = (h - 224) // 2, (w - 224) // 2
    patches = np.empty((len(rows), len(cams), 196, 768), dtype=np.float16)
    bs = args.batch
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        for k, mm in enumerate(mms):
            imgs = np.stack([mm[r[2]] for r in chunk])            # [b,256,256,3] u8
            x = torch.from_numpy(imgs[:, c0:c0 + 224, c1:c1 + 224]).to(dev)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = (x - NORM_M.to(dev)) / NORM_S.to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pt = vit(x)["patch_latent"]
            patches[i:i + len(chunk), k] = pt.float().cpu().numpy().astype(np.float16)
        if (i // bs) % 20 == 0:
            print(f"[{i}/{len(rows)}]", flush=True)

    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez(args.cache, patch=patches,
             state=np.stack([r[3] for r in rows]),
             ep=np.array([r[0] for r in rows], dtype=np.int64),
             t=np.array([r[1] for r in rows], dtype=np.int64))
    print(f"SAVED {args.cache} ({patches.nbytes / 1e9:.1f} GB)", flush=True)


def stage_train(args):
    z = np.load(args.cache)
    patch, state, ep = z["patch"], z["state"].astype(np.float32), z["ep"]
    mu, sd = state.mean(0), state.std(0) + 1e-6                   # standardize state
    state = (state - mu) / sd
    is_test = (ep % 10 == 0)                                      # episode-held-out split
    K = patch.shape[1]
    print(f"{len(ep)} samples, K={K}, test {is_test.sum()}", flush=True)

    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MMPerceiverMC(n_cams=K, d=args.d, state_dim=state.shape[1],
                          n_queries=args.queries).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_idx = np.flatnonzero(~is_test)
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(tr_idx)
        tot, nb = 0.0, 0
        for i in range(0, len(tr_idx), args.batch):
            sl = tr_idx[i:i + args.batch]
            p = torch.from_numpy(patch[sl].astype(np.float32)).to(dev)
            s = torch.from_numpy(state[sl]).to(dev)
            out = model(p, s)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            model.update_target()
            tot += float(out["loss"]); nb += 1
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"ep {epoch}: loss {tot / max(nb, 1):.4f}", flush=True)

    # probe gate: state R^2 from vision-only latent vs raw-ViT pool + RankMe
    model.eval()
    zs, raws = [], []
    for i in range(0, len(ep), 512):
        p = torch.from_numpy(patch[i:i + 512].astype(np.float32)).to(dev)
        s = torch.from_numpy(state[i:i + 512]).to(dev)
        zs.append(model.embed_vision(p, s).cpu().numpy())
        raws.append(p.mean((1, 2)).cpu().numpy())
    zv, raw = np.concatenate(zs), np.concatenate(raws)
    tr, te = ~is_test, is_test
    res = {"zv_rankme": rankme(zv[te]), "raw_rankme": rankme(raw[te]),
           "zv_r2_state": probe_r2(zv[tr], state[tr], zv[te], state[te]),
           "raw_r2_state": probe_r2(raw[tr], state[tr], raw[te], state[te]),
           "n_test": int(te.sum())}
    print("GATE:", {k: round(v, 3) if isinstance(v, float) else v
                    for k, v in res.items()}, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"seed{args.seed}.pt"))
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "gate": res,
                   "state_mu": mu.tolist(), "state_sd": sd.tolist()}, f, indent=1)
    print(f"SAVED {args.out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["precompute", "train", "all"], default="all")
    ap.add_argument("--dataset", required=True, help="lerobot dataset dir")
    ap.add_argument("--cache",
                    default="/mnt/nas/data/robocasa/kepler_cache/pnp_patches.npz")
    ap.add_argument("--out-dir",
                    default="/mnt/nas/data/robocasa/kepler_ckpt/pnp_mc3")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.stage in ("precompute", "all"):
        stage_precompute(args)
    if args.stage in ("train", "all"):
        stage_train(args)


if __name__ == "__main__":
    main()
