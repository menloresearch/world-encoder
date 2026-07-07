"""Animate the encoder's attention map over a frame sequence -> GIF.

Separate pipeline from pca_viz.py (the RGB/PCA panels). Takes a temporal run of frames
from ONE holdout scene/camera, computes the encoder's vision cross-attention per frame
(what the fused bottleneck reads; see pca_viz.encoder_panels), overlays it on each frame,
and stitches the frames into a GIF. With --also-vit the frozen ViT's CLS->patch
self-attention is animated side-by-side for comparison.

Attention is normalized GLOBALLY across the whole sequence (robust 2-98 percentile) so the
heatmap intensity is comparable frame-to-frame and doesn't flicker.

Choose the scene by an EXACT id (--cfg --task --user --scene), by an explicit holdout
--group (+ --scene-idx), or randomly from the --cfg TEST split.

  # pinned exact scene (cfg3, task16, user11, scene1)
  python -m world_tokenizer.attn_gif \
      --ckpt /mnt/nas/data/RH20T/checkpoints/phase1/all/seed0.pt \
      --cfg 3 --task 16 --user 11 --scene 1 --n-frames 48 --fps 12 --also-vit \
      --out pca_viz_out/attn_scene.gif
"""
import argparse
import csv
import glob
import os
import random

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw

from world_tokenizer.model import load_vitv2
from world_tokenizer.pca_viz import (GRID, _NORM_M, _NORM_S, build_encoder, encoder_attention,
                                     scene_cam_frames, scene_dir)


def pick_sequence(split_csv, cfg, task, user, scene, group, scene_idx, cam_idx, n_frames, frames_tmpl, rng):
    """-> (name, list of n_frames evenly-strided frame paths from one scene/cam).
    Pinned mode: give cfg+task+user+scene for an EXACT scene. Else group/random by cfg."""
    if task is not None and user is not None and scene is not None:
        assert cfg is not None, "pinned scene needs --cfg too"
        name, path = scene_dir(cfg, task, user, scene, frames_tmpl)
        _, fs = scene_cam_frames(path, cam_idx)
        assert len(fs) >= 2, f"no usable frames at {path} (cam_idx={cam_idx})"
        stride = max(1, len(fs) // n_frames)
        return name, fs[::stride][:n_frames]
    rows = [r for r in csv.DictReader(open(split_csv)) if r["split"].strip() == "test"]
    if group:
        groups = [group]
    else:
        groups = [r["group"].strip() for r in rows if r["cfg"].strip() == str(cfg)]
        rng.shuffle(groups)
    for g in groups:
        prefix = g.replace(f"_cfg_{int(cfg):04d}", "")
        scenes = sorted(glob.glob(os.path.join(frames_tmpl.format(cfg=int(cfg)), prefix + "_scene_*")))
        if scene_idx >= len(scenes):
            continue
        cams = sorted(glob.glob(os.path.join(scenes[scene_idx], "cam_*", "color")))
        if cam_idx >= len(cams):
            continue
        fs = sorted(glob.glob(os.path.join(cams[cam_idx], "*.jpg")))
        if len(fs) >= 2:
            stride = max(1, len(fs) // n_frames)
            return g, fs[::stride][:n_frames]
    raise SystemExit("no usable sequence — check --cfg/--group/--scene-idx/--cam-idx")


def load_seq(paths):
    """-> (normalized [N,3,224,224], display crops [N,224,224,3] uint8)."""
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    crops = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    norm = (crops - _NORM_M) / _NORM_S
    disp = (crops.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    return norm, disp


def vit_attention(e0, norm, device, bs=32):
    """Frozen ViT last-block CLS->patch self-attention -> [N,14,14]."""
    out = []
    for i in range(0, norm.shape[0], bs):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            a = e0(norm[i:i + bs].to(device), last_self_attention=True)["last_self_attention"]
        out.append(a.float().mean(1).cpu())            # [b,heads,196] -> [b,196]
    return torch.cat(out).reshape(-1, GRID, GRID).numpy()


def patch_tokens(e0, norm, device, bs=32):
    out = []
    for i in range(0, norm.shape[0], bs):
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(e0(norm[i:i + bs].to(device))["patch_latent"].float().cpu())
    return torch.cat(out)                              # [N,196,768]


def global_norm(heat):
    """Normalize a [N,14,14] stack to 0..1 with ONE 2-98 percentile scale (no per-frame flicker)."""
    lo, hi = np.percentile(heat, 2), np.percentile(heat, 98)
    return np.clip((heat - lo) / max(hi - lo, 1e-8), 0, 1)


def overlay(crop_u8, heat01, alpha=0.55):
    """crop [224,224,3] uint8 + heat [14,14] 0..1 -> blended [224,224,3] uint8 (inferno)."""
    big = np.asarray(Image.fromarray((heat01 * 255).astype(np.uint8), "L").resize((224, 224), Image.BICUBIC))
    heat_rgb = (cm.inferno(big / 255.0)[..., :3] * 255).astype(np.float32)
    return (crop_u8 * (1 - alpha) + heat_rgb * alpha).clip(0, 255).astype(np.uint8)


def captioned(arr_u8, text):
    """Add a slim black caption bar on top of a frame."""
    im = Image.fromarray(arr_u8)
    bar = Image.new("RGB", (im.width, 18), (0, 0, 0))
    ImageDraw.Draw(bar).text((4, 4), text, fill=(255, 255, 255))
    out = Image.new("RGB", (im.width, im.height + 18))
    out.paste(bar, (0, 0)); out.paste(im, (0, 18))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", type=int, default=3, help="config id (also used as the pinned-scene cfg)")
    ap.add_argument("--task", type=int, help="pinned scene: task id (with --user --scene for an exact scene)")
    ap.add_argument("--user", type=int, help="pinned scene: user id")
    ap.add_argument("--scene", type=int, help="pinned scene: scene id")
    ap.add_argument("--group", default=None, help="[else] explicit holdout group; else a random test group of --cfg")
    ap.add_argument("--scene-idx", type=int, default=0, help="[group mode] which scene of the group")
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--also-vit", action="store_true", help="animate ViT self-attention beside encoder attention")
    ap.add_argument("--split-csv", default="splits/holdout_v1.csv")
    ap.add_argument("--frames-tmpl", default="/mnt/nas/data/RH20T/frames/cfg{cfg}")
    ap.add_argument("--out", default="pca_viz_out/attn.gif")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    name, paths = pick_sequence(args.split_csv, args.cfg, args.task, args.user, args.scene,
                                args.group, args.scene_idx, args.cam_idx, args.n_frames, args.frames_tmpl, rng)
    print(f"{name}  cam{args.cam_idx}: {len(paths)} frames", flush=True)
    norm, disp = load_seq(paths)

    e0 = load_vitv2(pretrained=True).to(args.device).eval()
    patch = patch_tokens(e0, norm, args.device)
    model, kind = build_encoder(args.ckpt, args.device)
    enc_attn = global_norm(encoder_attention(model, kind, patch, args.device))  # [N,14,14]
    vit_attn = global_norm(vit_attention(e0, norm, args.device)) if args.also_vit else None

    frames = []
    for i in range(len(paths)):
        enc = np.asarray(captioned(overlay(disp[i], enc_attn[i]), f"{kind} encoder attn"))
        if args.also_vit:
            vit = np.asarray(captioned(overlay(disp[i], vit_attn[i]), "ViT self-attn"))
            sep = np.full((enc.shape[0], 4, 3), 30, np.uint8)
            frames.append(Image.fromarray(np.concatenate([vit, sep, enc], axis=1)))
        else:
            frames.append(Image.fromarray(enc))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, disposal=2)
    print(f"wrote {args.out}  ({len(frames)} frames @ {args.fps}fps)", flush=True)


if __name__ == "__main__":
    main()
