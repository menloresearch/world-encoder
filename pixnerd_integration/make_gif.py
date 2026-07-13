"""Make replay GIFs: decode a real episode's latent trajectory frame-by-frame.
For a chosen held-out scene, order its frames by timestamp, decode each z_v back to
pixels, and write a GIF with the REAL episode (top) beside the DECODED one (bottom).
This is reconstruction/replay of a real trajectory — NOT future prediction.

    python make_gif.py --ckpt .../last.ckpt --manifest /mnt/nas/data/RH20T/decode/ur5 \
        --n-scenes 2 --max-frames 48 --out figures/decode/gifs
"""
import argparse
import os
import re
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.models.transformer.pixnerd_t2i import PixNerDiT
from src.models.conditioner.latent import LatentConditioner
from src.models.autoencoder.pixel import PixelAE
from src.diffusion.flow_matching.sampling import EulerSampler, ode_step_fn
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.base.guidance import simple_guidance_fn
from src.data.dataset.imagenet import center_crop_fn


def scene_key(p):
    # .../frames/cfgX/<scene_dir>/cam_XXXX/color/<ts>.jpg  -> (scene_dir, cam)
    parts = p.split("/")
    return (parts[-4], parts[-3])


def ts_of(p):
    m = re.search(r"(\d+)\.jpg$", p)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="/mnt/nas/data/RH20T/decode/ur5")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-scenes", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=48)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default="figures/decode/gifs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    net = PixNerDiT(in_channels=3, patch_size=16, num_groups=16, hidden_size=512,
                    decoder_hidden_size=64, num_encoder_blocks=8, num_decoder_blocks=2,
                    num_text_blocks=2, txt_embed_dim=256, txt_max_length=1).to(dev)
    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    w = {k[len("denoiser."):]: v for k, v in sd.items() if k.startswith("denoiser.")}
    net.load_state_dict(w); net.eval()

    cond = LatentConditioner(dim=256)
    vae = PixelAE(scale=1.0)
    sampler = EulerSampler(scheduler=LinearScheduler(), w_scheduler=LinearScheduler(),
                           num_steps=args.steps, guidance=args.guidance,
                           guidance_fn=simple_guidance_fn, step_fn=ode_step_fn)

    zv = np.load(f"{args.manifest}/zv_{args.split}.npy")
    paths = open(f"{args.manifest}/frames_{args.split}.txt").read().splitlines()

    # group manifest rows by scene, order each by timestamp
    scenes = defaultdict(list)
    for i, p in enumerate(paths):
        scenes[scene_key(p)].append(i)
    for k in scenes:
        scenes[k].sort(key=lambda i: ts_of(paths[i]))
    # pick DISTINCT tasks (visually different setups), longest scene per task, for a
    # diverse demo — not the two longest scenes (which are often the same setup).
    def task_of(scene_dir):
        m = re.search(r"(task_\d+)", scene_dir)
        return m.group(1) if m else scene_dir
    best_per_task = {}
    for key, idxs in scenes.items():
        t = task_of(key[0])
        if t not in best_per_task or len(idxs) > len(best_per_task[t][1]):
            best_per_task[t] = (key, idxs)
    chosen = sorted(best_per_task.values(), key=lambda kv: -len(kv[1]))[:args.n_scenes]
    print(f"{len(best_per_task)} distinct tasks available; picking {len(chosen)}", flush=True)
    crop = partial(center_crop_fn, image_size=args.res)

    for si, (key, idxs) in enumerate(chosen):
        if len(idxs) > args.max_frames:                       # even-subsample long scenes
            sel = np.linspace(0, len(idxs) - 1, args.max_frames).round().astype(int)
            idxs = [idxs[j] for j in sel]
        print(f"scene {si}: {key[0]} ({len(idxs)} frames)", flush=True)
        reals = torch.stack([to_tensor(crop(Image.open(paths[i]).convert("RGB"))) for i in idxs])
        y = torch.from_numpy(zv[idxs]).float()

        gens = []
        with torch.no_grad():
            for b in range(0, len(idxs), args.batch):
                yb = y[b:b + args.batch]
                condition, uncondition = cond(yb)
                xT = torch.randn(len(yb), 3, args.res, args.res, device=dev)
                s = sampler(net, xT, condition, uncondition)
                s = vae.decode(s).float().clamp(-1, 1)
                gens.append(((s + 1) / 2).cpu())
        gen = torch.cat(gens)

        # compose frames: real (top) | decoded (bottom), stacked vertically w/ a 2px gap
        pad = 2
        frames = []
        for j in range(len(idxs)):
            r = (reals[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            g = (gen[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gap = np.full((pad, r.shape[1], 3), 255, np.uint8)
            frames.append(Image.fromarray(np.concatenate([r, gap, g], axis=0)))

        out = f"{args.out}/replay_{si}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0)
        print(f"  saved {out}  ({len(frames)} frames, top=real bottom=decoded)", flush=True)
    print("DONE make_gif", flush=True)


if __name__ == "__main__":
    main()
