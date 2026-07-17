"""Diagnostic: is the additive TimeEmbed swamping the visual content of a vision token?

vt = proj_v(rgb) + mod[0] + te(t).  If ||te|| >> ||proj_v(rgb)|| the token is dominated by
time and per-patch visual detail (the force cue) is lost -> vision-only z_v can't carry force.
Reports mean L2 norm of each additive component over a batch (TEMPORAL_ARCH.md §18.10).

    python -m world_tokenizer.diag_norms --v1 .../fix2/kuka/seed0.pt --cfgs 6 7
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver_temporal import MMPerceiverTemporal          # noqa: E402
from world_tokenizer.window_loader import make_window_loader, unpack           # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.v1, map_location=dev); a = ck["args"]
    m = MMPerceiverTemporal(d=a["d"], n_latents=a["n_latents"], n_self=a["n_self"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()

    _, te_loader, _ = make_window_loader(args.cache_dir, tuple(args.cfgs), window=args.window,
                                         stride=args.stride, batch=64, num_workers=4)
    b = next(iter(te_loader))
    rgb, motor, mm, ee, em, t = unpack(b, dev)

    def norm(x):  # mean L2 over last dim
        return x.norm(dim=-1).mean().item()

    vfeat = m.proj_v(rgb)                       # [B,C,196,d]
    tenc = m.time(t)                            # [B,C,d]
    print(f"batch rgb {tuple(rgb.shape)}  t_ms range [{t.min():.0f}, {t.max():.0f}]", flush=True)
    print("\n=== additive token components (mean L2 norm) ===", flush=True)
    print(f"  proj_v(rgb)   : {norm(vfeat):.3f}   <- the visual content", flush=True)
    print(f"  mod[0] (vis)  : {m.mod[0].norm().item():.3f}", flush=True)
    print(f"  time(t)       : {norm(tenc):.3f}   <- swamps vision if >> proj_v", flush=True)
    # per-patch spread: how much do patches differ within a frame (the detail that carries force)?
    within = vfeat.std(dim=2).norm(dim=-1).mean().item()   # std across 196 patches
    print(f"  proj_v within-frame patch std (L2): {within:.3f}", flush=True)
    print(f"  ratio time/vision content         : {norm(tenc)/max(norm(vfeat),1e-6):.2f}", flush=True)
    print(f"  ratio time/within-frame detail    : {norm(tenc)/max(within,1e-6):.2f}", flush=True)


if __name__ == "__main__":
    main()
