"""Quick probe eval for a temporal (v1) checkpoint — is z_v actually useful?

Encodes the vision-only window latent z_v on held-out windows and ridge-probes it against the
window-mean state (motor dims + ee), vs the raw pooled-ViT baseline. If zv_r2 > raw_r2 the fused
temporal latent carries state vision alone can't read (the cross-modal signal), the same test as
Phase-1's train_chunks eval. Also reports RankMe. This is a sanity probe, NOT the full future-Δt gate.

    python -m world_tokenizer.eval_temporal --ckpt .../temporal/ur5/seed0.pt --cfgs 3 4
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.mm_perceiver_temporal import MMPerceiverTemporal, masked_mean  # noqa: E402
from world_tokenizer.train_chunks import probe_r2, rankme                            # noqa: E402
from world_tokenizer.window_loader import make_window_loader, unpack                 # noqa: E402


@torch.no_grad()
def encode(model, loader, dev):
    """Per window: z_v [d], raw pooled ViT [768], window-mean motor [24] + validity, ee [15] + any."""
    model.eval()
    zv, raw, ym, ye, e_any = [], [], [], [], []
    for b in loader:
        rgb, motor, m_mask, ee, e_mask, t_ms = unpack(b, dev)
        zv.append(model.embed_vision(rgb, motor, m_mask, ee, e_mask, t_ms).float().cpu().numpy())
        raw.append(rgb.mean((1, 2)).cpu().numpy())                       # pooled over ticks+patches
        ym.append((motor.mean(1).reshape(motor.shape[0], -1)).cpu().numpy())   # [B,24] window-mean
        ye.append(masked_mean(ee, e_mask).mean(1).cpu().numpy() if ee.dim() == 4
                  else masked_mean(ee, e_mask).cpu().numpy())
        e_any.append(e_mask.any((1, 2)).cpu().numpy())
    return (np.concatenate(zv), np.concatenate(raw), np.concatenate(ym),
            np.concatenate(ye), np.concatenate(e_any))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--cfgs", type=int, nargs="+", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location=dev)
    a = ck["args"]
    model = MMPerceiverTemporal(d=a["d"], n_latents=a["n_latents"], n_self=a["n_self"]).to(dev)
    model.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch')})", flush=True)

    tr, te, ds = make_window_loader(args.cache_dir, tuple(args.cfgs), window=args.window,
                                    stride=args.stride, batch=128, num_workers=4)
    zv_tr, raw_tr, ym_tr, ye_tr, ea_tr = encode(model, tr, dev)
    zv_te, raw_te, ym_te, ye_te, ea_te = encode(model, te, dev)

    mvalid = np.ones(ym_tr.shape[1], bool)                               # keep dims with variance
    mvalid = ym_tr.std(0) > 1e-6
    print(f"windows train {len(zv_tr)} test {len(zv_te)} | motor dims used {int(mvalid.sum())}", flush=True)

    res = {"rankme_zv": rankme(zv_te)}
    res["zv_r2_motor"] = probe_r2(zv_tr, ym_tr[:, mvalid], zv_te, ym_te[:, mvalid])
    res["raw_r2_motor"] = probe_r2(raw_tr, ym_tr[:, mvalid], raw_te, ym_te[:, mvalid])
    if ea_tr.sum() > 50 and ea_te.sum() > 50:
        res["zv_r2_ee"] = probe_r2(zv_tr[ea_tr], ye_tr[ea_tr], zv_te[ea_te], ye_te[ea_te])
        res["raw_r2_ee"] = probe_r2(raw_tr[ea_tr], ye_tr[ea_tr], raw_te[ea_te], ye_te[ea_te])

    print("=== probe R2 (higher=better; zv>raw means fused latent carries state) ===", flush=True)
    for k, v in res.items():
        print(f"  {k:16s} {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
