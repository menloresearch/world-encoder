"""Pre-check B (V0.2.md Build 1): does multi-camera add info before we pay for the K-cam precompute?

For a few scenes, encode each EXTERNAL camera's frame at matched timestamps with the frozen v0.1 encoder
(vision-only latent), and compare two cosine distances:
  cross-view = how different the K views' latents are at the SAME tick
  cross-tick = how much one view's latent moves ACROSS ticks (the natural scale of latent motion)
Read: cross-view / cross-tick near 0 -> views redundant (multi-cam buys little); near/above 1 -> views
carry complementary info (multi-cam worth the precompute). Also reports raw-ViT cross-view for reference.

  .venv/bin/python -m world_tokenizer.precheck_multicam --cfg 7 --scenes 6 --ticks 6 \
      --v0 /mnt/nas/data/RH20T/checkpoints/phase1/kuka/seed0.pt --out results/temporal/precheck_multicam_kuka.json
"""
import argparse
import json
import os

import numpy as np
import torch

from world_tokenizer.chunk_state import IN_HAND_OF_CFG
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks
from world_tokenizer.model import load_vitv2
from world_tokenizer.precompute_patch import patch_embed


def cam_serial(d):
    return os.path.basename(d).replace("cam_", "")


def pdist_cos(z):
    """mean pairwise cosine distance (1-cos) among rows of z [k, d]; nan if <2 rows."""
    if len(z) < 2:
        return np.nan
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    s = zn @ zn.T
    iu = np.triu_indices(len(z), k=1)
    return float((1.0 - s[iu]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default="/mnt/nas/data/RH20T/frames")
    ap.add_argument("--cfg", type=int, default=7)
    ap.add_argument("--v0", required=True)
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=6, help="ticks (shared timestamps) per scene")
    ap.add_argument("--max-cams", type=int, default=8, help="cap external cams per scene")
    ap.add_argument("--tol-ms", type=int, default=100, help="nearest-timestamp match tolerance across cams")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    wrist = IN_HAND_OF_CFG.get(args.cfg, set())

    vit = load_vitv2(pretrained=True).to(dev).eval()
    enc = MMPerceiverChunks(d=256, n_queries=8).to(dev)
    enc.load_state_dict(torch.load(args.v0, map_location=dev))
    enc.eval()

    @torch.no_grad()
    def zvis(patches):  # patches [n,196,768] -> v0.1 vision-only latent [n,256] (state hidden)
        rgb = torch.from_numpy(patches.astype(np.float32)).to(dev)
        B = rgb.shape[0]
        motor = torch.zeros(B, 8, 3, device=dev)
        mm = torch.ones(B, 8, 3, dtype=torch.bool, device=dev)
        ee = torch.zeros(B, 13, 15, device=dev)
        em = torch.ones(B, 13, dtype=torch.bool, device=dev)
        return enc.embed_vision(rgb, motor, mm, ee, em).float().cpu().numpy()

    root = os.path.join(args.frames_root, f"cfg{args.cfg}")
    scenes = sorted(d for d in os.listdir(root) if d.startswith("task_"))
    cross_view, cross_tick, raw_view = [], [], []
    n_used = 0
    for sc in scenes:
        if n_used >= args.scenes:
            break
        cams = sorted(glob_cams(os.path.join(root, sc), wrist))
        if len(cams) < 2:
            continue
        cams = cams[:args.max_cams]
        # cameras aren't filename-synced -> match by NEAREST timestamp within tol
        cam_ts = {}
        for c in cams:
            fs = sorted(int(f.split(".")[0]) for f in os.listdir(os.path.join(c, "color"))
                        if f.endswith(".jpg"))
            cam_ts[c] = np.array(fs, dtype=np.int64)
        if any(len(v) < 2 for v in cam_ts.values()):
            continue
        ref = cams[0]
        ref_ts = cam_ts[ref]
        stride = max(1, len(ref_ts) // args.ticks)
        picks_ts = ref_ts[::stride][:args.ticks]
        paths, meta = [], []
        for ti, ts0 in enumerate(picks_ts):
            matched, ok = {}, True
            for c in cams:
                arr = cam_ts[c]
                j = int(np.searchsorted(arr, ts0))
                cand = [arr[k] for k in (j - 1, j) if 0 <= k < len(arr)]
                best = min(cand, key=lambda t: abs(int(t) - int(ts0))) if cand else None
                if best is None or abs(int(best) - int(ts0)) > args.tol_ms:
                    ok = False; break
                matched[c] = int(best)
            if not ok:
                continue
            for c in cams:
                paths.append(os.path.join(c, "color", f"{matched[c]}.jpg"))
                meta.append((ti, cam_serial(c)))
        if len(paths) < 4:
            continue
        patches = patch_embed(vit, paths, dev)          # [n,196,768]
        Z = zvis(patches)                                # [n,256]
        raw = patches.reshape(len(patches), 196, -1).mean(1)  # pooled raw ViT [n,768]
        meta = np.array(meta, dtype=object)
        tick_ids = np.array([m[0] for m in meta]); cam_ids = np.array([m[1] for m in meta])
        # cross-view: within each tick, across cams
        for ti in np.unique(tick_ids):
            m = tick_ids == ti
            cross_view.append(pdist_cos(Z[m])); raw_view.append(pdist_cos(raw[m]))
        # cross-tick: within each cam, across ticks
        for c in np.unique(cam_ids):
            m = cam_ids == c
            cross_tick.append(pdist_cos(Z[m]))
        n_used += 1
        print(f"  {sc[:40]} | {len(cams)} cams x {len(np.unique(tick_ids))} matched ticks", flush=True)

    cv = float(np.nanmean(cross_view)); ct = float(np.nanmean(cross_tick)); rv = float(np.nanmean(raw_view))
    res = {"cfg": args.cfg, "scenes_used": n_used, "v0": args.v0,
           "cross_view_dist": cv, "cross_tick_dist": ct, "raw_vit_cross_view_dist": rv,
           "ratio_view_over_tick": cv / ct if ct else float("nan")}
    print("\n=== Pre-check B: does multi-cam add info? ===", flush=True)
    print(f"  cross-VIEW latent dist (same tick, diff cams): {cv:.3f}", flush=True)
    print(f"  cross-TICK latent dist (same cam, diff ticks): {ct:.3f}", flush=True)
    print(f"  raw-ViT cross-view dist (reference):           {rv:.3f}", flush=True)
    print(f"  ratio view/tick: {res['ratio_view_over_tick']:.2f}  "
          f"({'views COMPLEMENTARY -> multi-cam worth it' if res['ratio_view_over_tick'] > 0.5 else 'views REDUNDANT -> multi-cam low value'})",
          flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("saved ->", args.out, flush=True)


def glob_cams(scene_dir, wrist):
    import glob
    out = []
    for c in sorted(glob.glob(os.path.join(scene_dir, "cam_*"))):
        if not os.path.isdir(os.path.join(c, "color")):
            continue
        if cam_serial(c) in wrist:                       # external cameras only (exclude wrist)
            continue
        out.append(c)
    return out


if __name__ == "__main__":
    main()
