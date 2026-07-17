"""Diagnostic: does v0.2's vision-only latent carry PRESENT force as well as v0.1's, and is the
window mean-pool the culprit? (TEMPORAL_ARCH.md §18.8 next-step.)

Probe present force (F/T at the last tick) from vision-only reps:
  v0.1            : single-frame embed_vision(last tick)         [B,256]
  v0.2 mean-pool  : window latent, mean over N latents           [B,256]  (current eval)
  v0.2 max-pool   : window latent, max over N latents            [B,256]  (dilution test)
  v0.2 unpooled   : full [B,N,d] -> PCA-512                       (no pooling at all)
  raw ViT         : pooled patch features of the last frame       [B,768]  (floor)

Read: if v0.2 max/unpooled >> mean-pool -> POOLING dilutes (eval fix). If all v0.2 << v0.1 even
unpooled -> the OBJECTIVE isn't shaping z_v for state (bigger fix).

    python -m world_tokenizer.diag_present --v1 .../fix/kuka/seed0.pt --v0 .../phase1/kuka/seed0.pt --cfgs 6 7
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver_temporal import MMPerceiverTemporal, masked_mean  # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks                          # noqa: E402
from world_tokenizer.train_chunks import probe_r2                                    # noqa: E402
from world_tokenizer.window_loader import make_window_loader                         # noqa: E402


@torch.no_grad()
def encode(v1, v0, loader, dev):
    v1.eval(); v0.eval()
    out = {k: [] for k in ("v0", "mean", "max", "flat", "raw", "y", "ok")}
    for b in loader:
        rgb = b["rgb"].to(dev); motor = b["motor"].to(dev); mm = b["motor_mask"].to(dev)
        ee = b["ee"].to(dev); em = b["ee_mask"].to(dev); t = b["t_ms"].to(dev)
        last = rgb.shape[1] - 1
        out["v0"].append(v0.embed_vision(rgb[:, last], motor[:, last], mm[:, last],
                                         ee[:, last], em[:, last]).float().cpu().numpy())
        tok = v1.embed_vision_tokens(rgb, motor, mm, ee, em, t).float()   # [B,N,d]
        out["mean"].append(tok.mean(1).cpu().numpy())
        out["max"].append(tok.max(1).values.cpu().numpy())
        out["flat"].append(tok.reshape(tok.shape[0], -1).cpu().numpy())
        out["raw"].append(rgb[:, last].mean(1).cpu().numpy())
        out["y"].append(masked_mean(ee[:, last, :, :6], em[:, last]).cpu().numpy())
        out["ok"].append(em[:, last].any(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True); ap.add_argument("--v0", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=8); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.v1, map_location=dev); a = ck["args"]
    v1 = MMPerceiverTemporal(d=a["d"], n_latents=a["n_latents"], n_self=a["n_self"]).to(dev)
    v1.load_state_dict(ck["model"])
    v0 = MMPerceiverChunks(d=256, n_queries=8).to(dev)
    v0.load_state_dict(torch.load(args.v0, map_location=dev))

    tr, te, _ = make_window_loader(args.cache_dir, tuple(args.cfgs), window=args.window,
                                   stride=args.stride, batch=128, num_workers=4)
    Tr, Te = encode(v1, v0, tr, dev), encode(v1, v0, te, dev)
    tr_ok, te_ok = Tr["ok"], Te["ok"]
    print(f"present-force windows: train {int(tr_ok.sum())} test {int(te_ok.sum())}", flush=True)

    # PCA-512 for the unpooled flat rep (fair capacity, fast), fit on train
    pca = PCA(n_components=min(512, Tr["flat"][tr_ok].shape[0]), random_state=0).fit(Tr["flat"][tr_ok])

    def R2(rep):
        xtr = pca.transform(Tr["flat"][tr_ok]) if rep == "flat" else Tr[rep][tr_ok]
        xte = pca.transform(Te["flat"][te_ok]) if rep == "flat" else Te[rep][te_ok]
        return probe_r2(xtr, Tr["y"][tr_ok], xte, Te["y"][te_ok])

    res = {"v01": R2("v0"), "v02_mean": R2("mean"), "v02_max": R2("max"),
           "v02_unpooled": R2("flat"), "raw_floor": R2("raw"),
           "n_train": int(tr_ok.sum()), "n_test": int(te_ok.sum()),
           "v1": args.v1, "v0": args.v0, "cfgs": args.cfgs,
           "window": args.window, "stride": args.stride}
    print("\n=== PRESENT force R2 (vision-only) ===", flush=True)
    print(f"  v0.1 (single frame) : {res['v01']:.3f}", flush=True)
    print(f"  v0.2 mean-pool      : {res['v02_mean']:.3f}   <- current eval", flush=True)
    print(f"  v0.2 max-pool       : {res['v02_max']:.3f}", flush=True)
    print(f"  v0.2 unpooled (PCA) : {res['v02_unpooled']:.3f}", flush=True)
    print(f"  raw ViT (floor)     : {res['raw_floor']:.3f}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
