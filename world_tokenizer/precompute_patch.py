"""Precompute frozen-e0 vision PATCH tokens + state per frame (for the Perceiver).

The Perceiver fuses MANY tokens, so we need patch tokens (196x768), not just CLS. Stored fp16 on a
subset of frames to keep the cache small and iteration fast. Output npz:
  patch (N,196,768) fp16 | state (N,28) | scene (N)

Only cfgs whose state() is 28-dim may be combined (cfg3 + cfg4: both UR5, 6-long joint.npy).

    python -m world_tokenizer.precompute_patch --cfgs 3 4 --per-scene 15 \
        --out /dev/shm/wae_tmp/mm_patch_cfg34.npz
"""
import argparse
import glob
import os

import numpy as np
import torch

from world_tokenizer.model import load_vitv2
from world_tokenizer.state import STATE_DIM, SceneState

_NORM_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_NORM_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def patch_embed(model, paths, dev, bs=128):
    import torchvision.transforms as T
    from PIL import Image
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    out = []
    for i in range(0, len(paths), bs):
        imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
        imgs = ((imgs - _NORM_M) / _NORM_S).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pt = model(imgs)["patch_latent"]  # (B,196,768)
        out.append(pt.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas/data/RH20T", help="data root with frames/ and raw/")
    ap.add_argument("--cfgs", type=int, nargs="+", default=[3])
    ap.add_argument("--per-scene", type=int, default=15)
    ap.add_argument("--out", default="/dev/shm/wae_tmp/mm_patch.npz")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    paths, states, scenes = [], [], []
    for cfg in args.cfgs:
        frames_root = os.path.join(args.root, "frames", f"cfg{cfg}")
        raw = os.path.join(args.root, "raw", f"RH20T_cfg{cfg}")
        n0 = len(paths)
        # "_human" catches both _human and _human_2 (the latter leaked into frames/, no robot state)
        for sc in sorted(d for d in os.listdir(frames_root)
                         if d.startswith("task_") and "_human" not in d):
            cams = sorted(glob.glob(os.path.join(frames_root, sc, "cam_*", "color")))
            if not cams:
                continue
            try:
                st = SceneState(os.path.join(raw, sc))
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
                if v.shape != (STATE_DIM,):
                    raise ValueError(f"cfg{cfg} {sc}: state dim {v.shape} != {STATE_DIM}")
                if np.isfinite(v).all():
                    paths.append(os.path.join(cams[0], f)); states.append(v); scenes.append(sc)
        print(f"cfg{cfg}: {len(paths) - n0} frames", flush=True)
    print(f"{len(paths)} frames over {len(set(scenes))} scenes -> patch tokens (frozen e0)", flush=True)

    model = load_vitv2(pretrained=True).to(dev).eval()
    patch = patch_embed(model, paths, dev)
    np.savez(args.out, patch=patch, state=np.stack(states).astype(np.float32), scene=np.array(scenes))
    gb = patch.nbytes / 1e9
    print(f"saved {args.out} | patch {patch.shape} ({gb:.1f} GB fp16) state {len(states)}", flush=True)


if __name__ == "__main__":
    main()
