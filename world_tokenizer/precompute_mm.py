"""Stage 2 data foundation: precompute frozen-e0 vision CLS + robot state per frame.

Vision backbone is frozen, so we embed once and cache. Output: an .npz with
  vision (N,768)  state (N,28)  scene (N,)   -> fast, in-memory training of the fusion encoder
(no ViT forward, no NFS reads during Stage-2 training).

    python -m world_tokenizer.precompute_mm --per-scene 40 --out /mnt/nas/data/RH20T/mm_cache.npz
"""
import argparse
import glob
import os

import numpy as np
import torch

from world_tokenizer.model import LeJEPAVideo
from world_tokenizer.probe_curve import FRAMES, embed
from world_tokenizer.state import SceneState

RAW = "/mnt/nas/data/RH20T/cfg3_raw/RH20T_cfg3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", default=FRAMES)
    ap.add_argument("--per-scene", type=int, default=40, help="strided frames per scene")
    ap.add_argument("--out", default="/mnt/nas/data/RH20T/mm_cache.npz")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    paths, states, scenes = [], [], []
    sc_names = sorted(d for d in os.listdir(args.frames_root) if d.startswith("task_"))
    for sc in sc_names:
        cams = sorted(glob.glob(os.path.join(args.frames_root, sc, "cam_*", "color")))
        if not cams:
            continue
        try:
            st = SceneState(os.path.join(RAW, sc))
        except Exception:
            continue
        fs = sorted(os.listdir(cams[0]))
        stride = max(1, len(fs) // args.per_scene)
        for f in fs[::stride][:args.per_scene]:
            ts = int(f.split(".")[0])
            try:
                v = st.state(ts)
            except Exception:
                continue
            if np.isfinite(v).all():
                paths.append(os.path.join(cams[0], f)); states.append(v); scenes.append(sc)
    print(f"{len(paths)} frames over {len(set(scenes))} scenes -> embedding vision (frozen e0)", flush=True)

    net = LeJEPAVideo(pretrained=True).to(dev).eval()
    vision = embed(net, paths, dev)  # (N, 768) CLS
    state = np.stack(states).astype(np.float32)
    scene_ids = np.array(scenes)
    np.savez(args.out, vision=vision.astype(np.float32), state=state, scene=scene_ids)
    print(f"saved {args.out} | vision {vision.shape} state {state.shape}", flush=True)


if __name__ == "__main__":
    main()
