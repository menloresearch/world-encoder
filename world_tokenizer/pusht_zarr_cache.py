"""PushT DP zarr -> kepler pretrain inputs (claim-C rung 0, 2026-08-03).

Same contract as square_hdf5_cache.py (see its docstring): emits
  (a) --cache npz: patch [N,K,196,768] fp16 | state [N,S] f32 | ep [N] | t [N]
  (b) --dataset-out lerobot-style dir: data/chunk-000/episode_*.parquet with
      action + episode_index columns.

PushT deltas vs the Square adapter, all deliberate:
  - source = diffusion_policy ReplayBuffer zarr (data/{img,state,action},
    meta/episode_ends), not robomimic hdf5.
  - ONE camera (top-down render) -> K=1; the pretrain infers K from the cache.
  - images are 96x96 -> bilinear UPSAMPLE to 224 (Square center-crops 256->224);
    upsample-to-ViT is the standard treatment for PushT (e.g. DINO-WM).
  - state = the FULL 5-D sim state (agent_xy 2 + block_xy 2 + block_angle 1),
    NOT just the policy lowdim (agent_pos 2). v0.3-recipe encoders are
    state-BLIND, so state here only feeds probes/detectors - block pose is the
    dynamics-relevant readout target for rung 0, agent pos alone is trivial.
  - stride default 2: PushT is 10 Hz, so 2 ticks = 0.2 s/row (Square: 5 @ 20 Hz
    = 0.25 s/row) - the closest match to the established row cadence.

    CUDA_VISIBLE_DEVICES=2 python -m world_tokenizer.pusht_zarr_cache \
      --zarr /mnt/nas/data/pusht/pusht_cchi_v7_replay.zarr \
      --cache /mnt/nas/data/robocasa/kepler_cache/pusht_patches.npz \
      --dataset-out /mnt/nas/data/robocasa/kepler_cache/pusht_actions_lerobot
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import zarr

NORM_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset-out", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from world_tokenizer.model import load_vitv2
    dev = "cuda"
    vit = load_vitv2(pretrained=True).to(dev).eval()

    root = zarr.open(args.zarr, "r")
    imgs_all = root["data/img"]                       # [T,96,96,3]
    state_all = np.asarray(root["data/state"], dtype=np.float32)   # [T,5]
    acts_all = np.asarray(root["data/action"], dtype=np.float32)   # [T,2]
    ep_ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
    n_eps = len(ep_ends)
    starts = np.concatenate([[0], ep_ends[:-1]])
    print(f"{n_eps} episodes, {ep_ends[-1]} steps, img dtype {imgs_all.dtype}", flush=True)

    chunk_dir = os.path.join(args.dataset_out, "data", "chunk-000")
    os.makedirs(chunk_dir, exist_ok=True)

    eps, ts, states, patch_blocks = [], [], [], []
    for ep in range(n_eps):
        s, e = int(starts[ep]), int(ep_ends[ep])
        rows = list(range(s, e, args.stride))

        pd.DataFrame({"action": list(acts_all[s:e]),
                      "episode_index": ep}).to_parquet(
            os.path.join(chunk_dir, f"episode_{ep:06d}.parquet"))

        block = np.empty((len(rows), 1, 196, 768), np.float16)
        for i in range(0, len(rows), args.batch):
            sel = rows[i:i + args.batch]
            x = torch.from_numpy(np.asarray(imgs_all[sel])).to(dev)
            x = x.permute(0, 3, 1, 2).float()
            if x.max() > 1.5:                          # uint8-scaled storage
                x = x / 255.0
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - NORM_M.to(dev)) / NORM_S.to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pt = vit(x)["patch_latent"]
            block[i:i + len(sel), 0] = pt.float().cpu().numpy().astype(np.float16)
        patch_blocks.append(block)
        eps += [ep] * len(rows)
        ts += [r - s for r in rows]
        states.append(state_all[rows])
        if ep % 25 == 0:
            print(f"[{ep}/{n_eps}] T={e - s} rows={len(rows)}", flush=True)

    patch = np.concatenate(patch_blocks)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez(args.cache, patch=patch,
             state=np.concatenate(states).astype(np.float32),
             ep=np.array(eps, dtype=np.int64),
             t=np.array(ts, dtype=np.int64))
    print(f"SAVED {args.cache}: patch {patch.shape} ({patch.nbytes / 1e9:.1f} GB), "
          f"{len(eps)} rows, {n_eps} episode parquets in {chunk_dir}", flush=True)


if __name__ == "__main__":
    main()
