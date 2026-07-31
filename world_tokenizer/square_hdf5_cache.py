"""square_d0 robomimic hdf5 -> kepler pretrain inputs (v0.4 build, 2026-07-30).

The encoder pretrains (robocasa_pretrain{,_v03,_v04}.py) consume:
  (a) --cache  npz: patch [N,K,196,768] fp16 | state [N,S] f32 | ep [N] | t [N]
      (built for pnp by robocasa_pretrain.py stage_precompute from lerobot frame_cache)
  (b) --dataset lerobot dir: only data/chunk-*/*.parquet with columns
      action + episode_index are read (load_actions).

Square is robomimic-native (no lerobot tree), so this adapter reads the regen'd
hdf5 directly and emits both. Featurization is stage_precompute's, verbatim:
center-crop 256->224, /255, ImageNet norm, frozen ViTv2 patch_latent, bf16
autocast, fp16 store. Cams in sorted-name order (agentview, robot0_eye_in_hand)
= K axis order. State = robomimic image-config lowdim: eef_pos(3) + eef_quat(4)
+ gripper_qpos(2) = 9-D (the exact lowdim the downstream policy sees, minus the
ft window — force arms extend this later). ep = int(demo_N suffix); t = raw tick
(20 Hz), rows at --stride (default 5, = pnp cache; h snapping unchanged).

    CUDA_VISIBLE_DEVICES=4 python -m world_tokenizer.square_hdf5_cache \
      --hdf5 /mnt/nas/data/mimicgen/core/square_d0_image_ft_256.hdf5 \
      --cache /mnt/nas/data/robocasa/kepler_cache/square_patches.npz \
      --dataset-out /mnt/nas/data/robocasa/kepler_cache/square_actions_lerobot
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch

NORM_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CAMS = ["agentview_image", "robot0_eye_in_hand_image"]          # sorted, = K order
STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset-out", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    from world_tokenizer.model import load_vitv2
    dev = "cuda"
    vit = load_vitv2(pretrained=True).to(dev).eval()

    f = h5py.File(args.hdf5, "r")
    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    print(f"{len(demos)} demos, cams {CAMS}, stride {args.stride}", flush=True)

    chunk_dir = os.path.join(args.dataset_out, "data", "chunk-000")
    os.makedirs(chunk_dir, exist_ok=True)

    eps, ts, states, patch_blocks = [], [], [], []
    for di, dk in enumerate(demos):
        g = f["data"][dk]
        ep = int(dk.split("_")[1])
        T = g["actions"].shape[0]
        rows = list(range(0, T, args.stride))

        acts = g["actions"][:].astype(np.float32)                # [T,7]
        pd.DataFrame({"action": list(acts),
                      "episode_index": ep}).to_parquet(
            os.path.join(chunk_dir, f"episode_{ep:06d}.parquet"))

        st = np.concatenate([g["obs"][k][:].astype(np.float32)
                             for k in STATE_KEYS], axis=1)       # [T,9]
        block = np.empty((len(rows), len(CAMS), 196, 768), np.float16)
        for k, cam in enumerate(CAMS):
            imgs = g["obs"][cam][rows]                           # [R,256,256,3] u8
            for i in range(0, len(rows), args.batch):
                x = torch.from_numpy(imgs[i:i + args.batch, 16:240, 16:240]).to(dev)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                x = (x - NORM_M.to(dev)) / NORM_S.to(dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pt = vit(x)["patch_latent"]
                block[i:i + len(x), k] = pt.float().cpu().numpy().astype(np.float16)
        patch_blocks.append(block)
        eps += [ep] * len(rows)
        ts += rows
        states.append(st[rows])
        if di % 50 == 0:
            print(f"[{di}/{len(demos)}] {dk} T={T} rows={len(rows)}", flush=True)
    f.close()

    patch = np.concatenate(patch_blocks)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez(args.cache, patch=patch,
             state=np.concatenate(states).astype(np.float32),
             ep=np.array(eps, dtype=np.int64),
             t=np.array(ts, dtype=np.int64))
    print(f"SAVED {args.cache}: patch {patch.shape} ({patch.nbytes / 1e9:.1f} GB), "
          f"{len(eps)} rows, {len(demos)} episode parquets in {chunk_dir}", flush=True)


if __name__ == "__main__":
    main()
