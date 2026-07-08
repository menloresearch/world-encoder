"""Per-scene PCA + attention visualization suite -> individual PNGs (+ GIFs) + one overview.

Supersedes the single grid figures from ``pca_viz`` / ``attn_gif`` for the phase-1 story:
everything is written as standalone PNGs in a per-scene folder tree so figures can be
recomposed freely, plus one ``overview.png`` for a quick look. It drives the existing
``pca_viz`` / ``attn_gif`` helpers; those two modules still work standalone.

Model roles (fixed to the phase-1 checkpoints):
  * PCA-RGB          -- ``vision_head_all.pt``: frozen e0 ViT + trained linear ``proj_v``
                        head (768->256). The "ViT + linear head FT". A pretrained-e0 PCA is
                        painted alongside as the baseline, and the multimodal encoder's own
                        ``proj_v`` PCA as the "ours" reference.
  * FT-ViT attention -- ``vision_full_all.pt``: the full-finetuned backbone (head_only=False),
                        whose CLS->patch self-attention genuinely differs from pretrained.
  * Pretrained attn  -- e0 (``OK-AI/lejepa-vitb16-pretrain-in1k``).
  * Encoder attn     -- phase-1 Perceiver final cross-attention (the ``ALL`` encoder and the
                        matching per-robot specialist), i.e. what the fused bottleneck reads.

Three attention animations per scene: pretrained ViT, FT ViT, and our encoder (ALL + spec).

Layout::

  <out>/overview.png
  <out>/cfg3_ur5_t3_u16_s7/
      raw.png
      pca_pretrained.png   pca_headft.png   pca_encoder.png
      attn_pretrained.png  attn_pretrained.gif
      attn_ftvit.png       attn_ftvit.gif
      attn_encoder_all.png attn_encoder_all.gif
      attn_encoder_ur5.png attn_encoder_ur5.gif

  python -m world_tokenizer.viz_suite \
      --scenes-csv splits/viz_scenes_default.csv --out viz_out \
      --head-ckpt /mnt/nas/data/RH20T/checkpoints/phase1_vision/vision_head_all.pt \
      --vit-ckpt  /mnt/nas/data/RH20T/checkpoints/phase1_vision/vision_full_all.pt \
      --all-ckpt  /mnt/nas/data/RH20T/checkpoints/phase1/all/seed0.pt
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from world_tokenizer.attn_gif import captioned, global_norm, load_seq, overlay, patch_tokens, vit_attention
from world_tokenizer.model import load_vitv2
from world_tokenizer.pca_viz import (VIT_NAME, build_encoder, encoder_attention, encoder_proj,
                                     head_proj_feats, load_batch, load_ft_vit, pca_rgb,
                                     scene_cam_frames, scene_dir)

# cfg -> embodiment (matches the phase-1 specialists: flexiv=cfg1+2, ur5=cfg3+4,
# franka=cfg5, kuka=cfg6+7).
CFG_ROBOT = {1: "flexiv", 2: "flexiv", 3: "ur5", 4: "ur5", 5: "franka", 6: "kuka", 7: "kuka"}


def strided(fs, n):
    """Evenly stride a frame list down to ~n frames (keeps temporal coverage of the scene)."""
    stride = max(1, len(fs) // n)
    return fs[::stride][:n]


def save_png(arr_u8, path):
    Image.fromarray(np.asarray(arr_u8).astype(np.uint8)).save(path)


def write_gif(disp, heat01, caption, out_path, fps):
    """disp [N,224,224,3] u8 + heat [N,14,14] in 0..1 -> captioned single-panel GIF."""
    frames = [Image.fromarray(np.asarray(captioned(overlay(disp[i], heat01[i]), caption)))
              for i in range(len(disp))]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, disposal=2)


def mid_overlay(disp, heat01):
    """Representative (middle-frame) attention overlay [224,224,3] u8 for PNG + overview."""
    i = len(disp) // 2
    return overlay(disp[i], heat01[i])


# ------------------------------------------------------------------------------- per scene

def process_scene(row, models, args, device):
    """Render every PNG/GIF for one curated scene; return arrays needed by the overview."""
    cfg, task, user, scene = (int(row["cfg"]), int(row["task"]), int(row["user"]), int(row["scene"]))
    robot = CFG_ROBOT.get(cfg, f"cfg{cfg}")
    name, path = scene_dir(cfg, task, user, scene, args.frames_tmpl)
    tag = f"cfg{cfg}_{robot}_t{task}_u{user}_s{scene}"
    sdir = os.path.join(args.out, tag)
    os.makedirs(sdir, exist_ok=True)

    _, fs = scene_cam_frames(path, args.cam_idx)
    assert fs, f"no frames for {name} (cam_idx={args.cam_idx})"
    mid = fs[len(fs) // 2]

    e0, ftvit, headlin = models["e0"], models["ftvit"], models["headlin"]

    # ---- PCA (single representative mid-frame) --------------------------------------
    norm_hi, _ = load_batch([mid], args.pca_res)         # frozen-ViT panels: fine patch grid
    norm_lo, crop = load_batch([mid], 224)               # encoder panel: trained 196-token grid
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        patch_hi = e0(norm_hi.to(device))["patch_latent"].float().cpu()      # [1,hi*hi,768]
        patch_lo = e0(norm_lo.to(device))["patch_latent"].float().cpu()      # [1,196,768]
    raw_u8 = (crop[0] * 255).astype(np.uint8)
    pca_pre = pca_rgb(patch_hi, fg_mask=args.fg_mask)[0]                      # baseline e0 PCA
    pca_head = pca_rgb(head_proj_feats(args.head_ckpt, patch_hi, device), fg_mask=args.fg_mask)[0]
    enc_all = models["enc"]["all"][0]
    pca_enc = pca_rgb(encoder_proj(enc_all, patch_lo, device), fg_mask=args.fg_mask)[0]
    save_png(raw_u8, os.path.join(sdir, "raw.png"))
    save_png(pca_pre, os.path.join(sdir, "pca_pretrained.png"))
    save_png(pca_head, os.path.join(sdir, "pca_headft.png"))
    save_png(pca_enc, os.path.join(sdir, "pca_encoder.png"))

    # ---- attention over the frame sequence ------------------------------------------
    seq = strided(fs, args.n_frames)
    norm, disp = load_seq(seq)
    pre_attn = global_norm(vit_attention(e0, norm, device))                  # pretrained ViT
    ft_attn = global_norm(vit_attention(ftvit, norm, device))                # FT ViT
    patch = patch_tokens(e0, norm, device)                                   # e0 tokens -> encoders
    ov = {"attn_pretrained": mid_overlay(disp, pre_attn),
          "attn_ftvit": mid_overlay(disp, ft_attn)}
    if not args.no_gif:
        write_gif(disp, pre_attn, "pretrained ViT self-attn", os.path.join(sdir, "attn_pretrained.gif"), args.fps)
        write_gif(disp, ft_attn, "FT ViT self-attn", os.path.join(sdir, "attn_ftvit.gif"), args.fps)
    save_png(ov["attn_pretrained"], os.path.join(sdir, "attn_pretrained.png"))
    save_png(ov["attn_ftvit"], os.path.join(sdir, "attn_ftvit.png"))

    # encoder attention: ALL + the matching specialist
    enc_variants = [("all", models["enc"]["all"])]
    if robot in models["enc"]:
        enc_variants.append((robot, models["enc"][robot]))
    for etag, (emodel, ekind) in enc_variants:
        heat = global_norm(encoder_attention(emodel, ekind, patch, device))
        key = f"attn_encoder_{etag}"
        ov[key] = mid_overlay(disp, heat)
        save_png(ov[key], os.path.join(sdir, key + ".png"))
        if not args.no_gif:
            write_gif(disp, heat, f"{etag} encoder attn", os.path.join(sdir, key + ".gif"), args.fps)

    print(f"  {tag}: {len(seq)} attn frames -> {sdir}", flush=True)
    return {"tag": tag, "robot": robot, "raw": raw_u8, "pca_pretrained": pca_pre,
            "pca_headft": pca_head, "pca_encoder": pca_enc, **ov}


# --------------------------------------------------------------------------------- overview

def build_overview(results, out_path, vit_label):
    """One row per scene; PCA columns then representative-frame attention columns."""
    cols = [("image", "raw"),
            (f"PCA · {VIT_NAME}", "pca_pretrained"),
            ("PCA · head-FT (ViT+linear)", "pca_headft"),
            ("PCA · encoder proj_v", "pca_encoder"),
            ("attn · pretrained ViT", "attn_pretrained"),
            (f"attn · FT ViT ({vit_label})", "attn_ftvit"),
            ("attn · encoder ALL", "attn_encoder_all"),
            ("attn · encoder spec", "_spec")]
    n, ncol = len(results), len(cols)
    fig, axes = plt.subplots(n, ncol, figsize=(ncol * 2.4, n * 2.4), squeeze=False)
    for i, r in enumerate(results):
        for j, (_, key) in enumerate(cols):
            ax = axes[i][j]
            k = f"attn_encoder_{r['robot']}" if key == "_spec" else key
            img = r.get(k)
            if img is not None:
                ax.imshow(np.asarray(img).astype(np.uint8))
            ax.set_xticks([]); ax.set_yticks([])
        axes[i][0].set_ylabel(r["tag"], fontsize=7, rotation=0, ha="right", va="center", labelpad=30)
    for j, (t, _) in enumerate(cols):
        axes[0][j].set_title(t, fontsize=8)
    fig.suptitle("phase-1 visualization overview  (PCA: head-FT · attention: pretrained/FT-ViT/encoder)",
                 fontsize=11, y=0.999)
    fig.tight_layout(rect=(0.03, 0, 1, 0.99))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({n} scenes)", flush=True)


# -------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes-csv", default="splits/viz_scenes_default.csv",
                    help="CSV (task,user,cfg,scene) -> one scene per row")
    ap.add_argument("--out", default="viz_out")
    ck = "/mnt/nas/data/RH20T/checkpoints"
    ap.add_argument("--all-ckpt", default=f"{ck}/phase1/all/seed0.pt", help="ALL Perceiver encoder")
    ap.add_argument("--spec-tmpl", default=f"{ck}/phase1/{{robot}}/seed0.pt",
                    help="per-robot specialist encoder template ({robot})")
    ap.add_argument("--head-ckpt", default=f"{ck}/phase1_vision/vision_head_all.pt",
                    help="ViT+linear head FT (frozen e0 + proj_v) -> PCA")
    ap.add_argument("--vit-ckpt", default=f"{ck}/phase1_vision/vision_full_all.pt",
                    help="full-finetuned ViT -> FT-ViT attention")
    ap.add_argument("--pca-res", type=int, default=448,
                    help="ViT input res for the PCA panels (448->28x28 for a smooth map)")
    ap.add_argument("--fg-mask", action="store_true", help="DINOv2 foreground-masked PCA recipe")
    ap.add_argument("--n-frames", type=int, default=48, help="attention GIF frames (strided over the scene)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--cam-idx", type=int, default=0)
    ap.add_argument("--no-gif", action="store_true", help="skip GIFs, still write PNGs + overview")
    ap.add_argument("--no-overview", action="store_true")
    ap.add_argument("--frames-tmpl", default="/mnt/nas/data/RH20T/frames/cfg{cfg}")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.scenes_csv)))
    assert rows, f"no scenes in {args.scenes_csv}"
    robots_needed = {CFG_ROBOT.get(int(r["cfg"])) for r in rows}
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    print(f"{len(rows)} scenes | device {dev} | out {args.out}", flush=True)

    # Load every model once, then loop scenes.
    models = {"e0": load_vitv2(pretrained=True).to(dev).eval(),
              "ftvit": load_ft_vit(args.vit_ckpt, dev),
              "headlin": None, "enc": {}}
    models["enc"]["all"] = build_encoder(args.all_ckpt, dev)                  # (model, kind)
    for robot in sorted(x for x in robots_needed if x):
        spec = args.spec_tmpl.format(robot=robot)
        if os.path.exists(spec):
            models["enc"][robot] = build_encoder(spec, dev)
        else:
            print(f"  ! specialist missing for {robot}: {spec} (skipping spec column)", flush=True)
    print(f"models loaded: e0, ftvit({os.path.basename(args.vit_ckpt)}), "
          f"encoders={list(models['enc'])}", flush=True)

    results = [process_scene(r, models, args, dev) for r in rows]

    if not args.no_overview:
        build_overview(results, os.path.join(args.out, "overview.png"), os.path.basename(args.vit_ckpt))


if __name__ == "__main__":
    main()
