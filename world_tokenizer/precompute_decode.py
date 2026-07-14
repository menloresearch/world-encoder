"""Build the (frame image, z_v latent) manifest for the PixNerd pixel-decoder.

For each cached chunk: z_v = frozen encoder's vision-only latent (from cached patch features,
no ViT needed), and the frame path is re-derived (scene serial via chunk_state). Writes, per
split, a `zv.npy` [N,256] + `frames.txt` (aligned paths) that PixNerd's RobotLatentDataset
reads — so PixNerd never needs the encoder at train time.

    python -m world_tokenizer.precompute_decode --cfgs 3 4 \
        --ckpt /mnt/nas/data/RH20T/checkpoints/phase1/all/seed0.pt --out /mnt/nas/data/RH20T/decode/ur5
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_tokenizer.dataloader import ChunkDataset, load_split       # noqa: E402
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks            # noqa: E402
from world_tokenizer.train_chunks import encode_zv                     # noqa: E402
from world_tokenizer.chunk_state import IN_HAND_OF_CFG, SceneChunks    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", default="/mnt/nas/data/RH20T/caches")
    ap.add_argument("--raw-root", default="/mnt/nas/data/RH20T/raw")
    ap.add_argument("--frames-root", default="/mnt/nas/data/RH20T/frames")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--latent", choices=["vision", "state"], default="vision",
                    help="vision = z_v (eval default); state = z_state (vision hidden, "
                         "motor+ee only) for the cross-modal 'reconstruct frame from "
                         "proprioception+force' decode.")
    ap.add_argument("--out", default="/mnt/nas/data/RH20T/decode/ur5")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    split = load_split()
    model = MMPerceiverChunks(d=args.d, n_queries=args.queries).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    model.eval()

    ds = ChunkDataset(args.cache_dir, tuple(args.cfgs))
    zv, _ = encode_zv(model, ds, dev, mode=args.latent)               # [N,256] vision-only or state-only latent
    d = ds._d
    scene = np.array([ds.scenes[j] for j in ds._scene_idx])           # per-sample scene name
    ts, cfg = d["ts"], d["cfg"]

    # re-derive the (external) camera serial per unique scene, then build frame paths
    serial_cache = {}
    def serial_for(n, s):
        key = (n, s)
        if key not in serial_cache:
            try:
                sc = SceneChunks(os.path.join(args.raw_root, f"RH20T_cfg{n}", s),
                                 exclude=IN_HAND_OF_CFG.get(n, ()))
                serial_cache[key] = sc.serial
            except Exception:
                serial_cache[key] = None
        return serial_cache[key]

    paths, keep = [], []
    for i in range(len(zv)):
        n, s = int(cfg[i]), str(scene[i])
        ser = serial_for(n, s)
        p = None
        if ser is not None:
            p = os.path.join(args.frames_root, f"cfg{n}", s, f"cam_{ser}", "color", f"{int(ts[i])}.jpg")
        if p and os.path.exists(p):
            paths.append(p); keep.append(i)
    keep = np.array(keep)
    zv, paths = zv[keep], np.array(paths)
    grp = np.array([split.get(g, "train") for g in (ds.groups[j] for j in ds._group_idx[keep])])

    for name in ("train", "test"):
        m = grp == name
        np.save(os.path.join(args.out, f"zv_{name}.npy"), zv[m].astype(np.float32))
        with open(os.path.join(args.out, f"frames_{name}.txt"), "w") as f:
            f.write("\n".join(paths[m].tolist()))
        print(f"{name}: {int(m.sum())} (frame,z_v) pairs", flush=True)
    print(f"manifest -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
