"""PCA-RGB + attention latent visualization across a robot image (LeJEPA-style).

The LeJEPA paper paints PCA of a ViT's PATCH tokens over the image ("clear semantic
relationships"). Our trained encoders pool 8 Perceiver queries into a single vector
(``PerceiverFuse`` ends in ``x.mean(1)``) — there is no patch grid at the output — so we
show four patch-aligned panels per image, arranged as baseline-vs-ours pairs:

  A  ViT PCA-RGB     — frozen e0 ``patch_latent`` -> PCA -> RGB. The paper's experiment.
                       Run at high --res for a fine patch grid (448->28x28, 672->42x42) so
                       the heatmap is smooth like LeJEPA's (their look = many tokens +
                       bicubic upsample, DINOv2 recipe; there is no viz code in their repo).
  B  encoder proj_v PCA — ``proj_v(patch) + mod[0]`` (196xd) -> same PCA as A -> RGB. The
                       encoder's learned per-patch vision projection: the last spatial layer
                       BEFORE the Perceiver fuses+pools to the non-spatial bottleneck. A|B is
                       then a like-for-like ViT-vs-ours comparison (raw ViT patches vs the
                       trained linear re-mix of them). Fixed at 14x14 (encoder uses 196 tokens).
  C  ViT attention   — e0 last-layer CLS->patch self-attention (DINO-style), heads averaged.
  D  encoder attention — vision-only fuse pass; softmax(q.k^T) of the 8 queries over the
                       196 vision patches. What our fused bottleneck READS from the image.

A|B compare PCA structure (frozen ViT vs the encoder's projection); C|D compare where
attention lands. The encoder runs at 224 (196 tokens, hardcoded in _context/_attn_mask);
only the frozen-ViT panels A and C scale with --res.
The encoder class is picked from the checkpoint keys (proj_s -> MMPerceiver,
proj_m -> MMPerceiverChunks). Motor/ee/state are dummy tensors the vision-only pass blocks,
so their values never reach the vision panels. Samples come from the holdout_v1 TEST split.

Three ways to choose frames:
  * sampling  — random TEST-split scenes across --cfgs (default), one frame each.
  * pinned    — an EXACT scene via --cfg --task --user --scene (all four), N time-strided
                frames from --cam-idx. Frame dirs are task_{t}_user_{u}_scene_{s}_cfg_{c}.
  * list      — --scenes-csv <file> (columns task,user,cfg,scene): one representative frame
                per listed scene, one row each. The curated default set of visualizations
                lives at splits/viz_scenes_default.csv (one hand-picked scene per cfg).

  # sampling mode
  python -m world_tokenizer.pca_viz \
      --ckpt /mnt/nas/data/RH20T/checkpoints/phase1/all/seed0.pt \
      --cfgs 1,3,5 --n-images 6 --out pca_viz_out/chunk_all.png

  # pinned scene (cfg3, task16, user11, scene1)
  python -m world_tokenizer.pca_viz \
      --ckpt /mnt/nas/data/RH20T/checkpoints/phase1/all/seed0.pt \
      --cfg 3 --task 16 --user 11 --scene 1 --n-images 6 --out pca_viz_out/scene.png
"""
import argparse
import csv
import glob
import os
import random
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from world_tokenizer.mm_perceiver import MMPerceiver, PerceiverFuse
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks
from world_tokenizer.model import CKPT, load_vitv2

VIT_NAME = CKPT.split("/")[-1]   # e0 backbone: lejepa-vitb16-pretrain-in1k (LeJEPA ViT-B/16, DINOv2-style)
GRID = 14  # 224 / 16 -> 14x14 patch grid (196 tokens)
_NORM_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_NORM_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- sampling

def sample_frames(split_csv, cfgs, n_images, frames_tmpl, rng):
    """Pick n_images (cfg, group, img_path), one mid-sequence frame per distinct TEST group."""
    rows = [r for r in csv.DictReader(open(split_csv)) if r["split"].strip() == "test"]
    picks = []
    for cfg, k in zip(cfgs, _split_counts(n_images, len(cfgs))):
        groups = [r["group"].strip() for r in rows if r["cfg"].strip() == str(cfg)]
        rng.shuffle(groups)
        taken = 0
        for g in groups:
            if taken >= k:
                break
            img = _repr_frame(g, cfg, frames_tmpl)
            if img:
                picks.append((cfg, g, img))
                taken += 1
    return picks


def _split_counts(n, k):
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def _repr_frame(group, cfg, frames_tmpl):
    """Middle frame of the first camera of the group's first scene (or None)."""
    prefix = group.replace(f"_cfg_{int(cfg):04d}", "")
    scenes = sorted(glob.glob(os.path.join(frames_tmpl.format(cfg=int(cfg)), prefix + "_scene_*")))
    for sc in scenes:
        for cam in sorted(glob.glob(os.path.join(sc, "cam_*", "color"))):
            imgs = sorted(glob.glob(os.path.join(cam, "*.jpg")))
            if imgs:
                return imgs[len(imgs) // 2]
    return None


def scene_dir(cfg, task, user, scene, frames_tmpl="/mnt/nas/data/RH20T/frames/cfg{cfg}"):
    """(cfg,task,user,scene) ints -> (dir_name, absolute path). Names are zero-padded to 4."""
    name = f"task_{int(task):04d}_user_{int(user):04d}_scene_{int(scene):04d}_cfg_{int(cfg):04d}"
    return name, os.path.join(frames_tmpl.format(cfg=int(cfg)), name)


def scene_cam_frames(scene_path, cam_idx=0):
    """-> (cam color dir, sorted frame paths) for the cam_idx-th camera of a scene dir."""
    cams = sorted(glob.glob(os.path.join(scene_path, "cam_*", "color")))
    if not cams:
        return None, []
    cam = cams[cam_idx if 0 <= cam_idx < len(cams) else 0]
    return cam, sorted(glob.glob(os.path.join(cam, "*.jpg")))


def pinned_frames(cfg, task, user, scene, n_images, cam_idx, frames_tmpl):
    """n_images (cfg, scene_name, path) evenly strided across ONE pinned scene/camera's timeline."""
    name, path = scene_dir(cfg, task, user, scene, frames_tmpl)
    _, fs = scene_cam_frames(path, cam_idx)
    assert fs, f"no frames at {path} (cam_idx={cam_idx}) — check --cfg/--task/--user/--scene/--cam-idx"
    stride = max(1, len(fs) // n_images)
    return [(cfg, name, p) for p in fs[::stride][:n_images]]


def scenes_from_csv(path, cam_idx, frames_tmpl):
    """One representative (mid-sequence) frame per scene listed in a CSV with columns
    task,user,cfg,scene -> [(cfg, scene_name, img_path), ...], one row per scene."""
    picks = []
    for r in csv.DictReader(open(path)):
        cfg, task, user, scene = (int(r["cfg"]), int(r["task"]), int(r["user"]), int(r["scene"]))
        name, sp = scene_dir(cfg, task, user, scene, frames_tmpl)
        _, fs = scene_cam_frames(sp, cam_idx)
        assert fs, f"no frames for {name} (cam_idx={cam_idx}) from {path}"
        picks.append((cfg, name, fs[len(fs) // 2]))
    return picks


def short_label(cfg, group):
    """Compact 2-line row label from a scene/group name: 'cfg3\\nt3 u16 s7'."""
    m = re.search(r"task_(\d+)_user_(\d+)(?:_scene_(\d+))?_cfg_(\d+)", group)
    if not m:
        return f"cfg{cfg}\n{group[:20]}"
    t, u, s, _ = m.groups()
    return f"cfg{cfg}\nt{int(t)} u{int(u)}" + (f" s{int(s)}" if s else "")


# ------------------------------------------------------------------------------- images

def load_batch(paths, res=224):
    """-> (normalized [N,3,res,res] for the model, display crops [N,res,res,3] in 0..1).
    Resize(res*8/7)+CenterCrop(res) keeps the SAME field of view at any res, so higher-res
    patch grids overlay the 224 display crop 1:1."""
    tf = T.Compose([T.Resize(round(res * 8 / 7)), T.CenterCrop(res), T.ToTensor()])
    crops = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    return (crops - _NORM_M) / _NORM_S, crops.permute(0, 2, 3, 1).numpy()


def _upsample(grid, size=224, mode=Image.BICUBIC):
    """[14,14,(3)] float 0..1 -> [224,224,(3)] float 0..1 via PIL resize."""
    is_rgb = grid.ndim == 3
    arr = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="RGB" if is_rgb else "L").resize((size, size), mode)
    return np.asarray(im).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------------- PCA

def pca_rgb(feats, fg_mask=False):
    """feats [N,P,C] -> [N,224,224,3] uint8 (grid inferred as sqrt(P)). Joint PCA across
    all patches for cross-image color comparability; robust 2-98 percentile normalization;
    bicubic upsample. DINOv2 recipe: with fg_mask, threshold PC1 to a foreground mask, map
    the NEXT 3 PCs to RGB, and paint background white — the clean 'semantic' look. Without
    it, the top-3 PCs go straight to RGB. Smoothness comes mostly from a fine grid (high
    --res), not from this function."""
    N, P, C = feats.shape
    grid = int(round(P ** 0.5))
    flat = feats.reshape(-1, C).float()
    flat = flat - flat.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(flat, full_matrices=False)        # right singular vecs = principal axes
    proj = (flat @ V[:4].T).numpy()                              # top-4: PC1 (mask) + PC2-4 (rgb)
    if fg_mask:
        pc1 = proj[:, 0]
        pc1 = (pc1 - pc1.min()) / max(pc1.max() - pc1.min(), 1e-6)
        m = pc1 > 0.5
        if m.mean() > 0.5:                                       # treat the smaller side as foreground
            m = ~m
        if not (0.05 < m.mean() < 0.95):                         # degenerate split -> skip masking
            m = np.ones(len(proj), bool)
        rgb_src = proj[:, 1:4]
    else:
        m = np.ones(len(proj), bool)
        rgb_src = proj[:, :3]
    lo, hi = np.percentile(rgb_src[m], 2, axis=0), np.percentile(rgb_src[m], 98, axis=0)
    rgb = np.clip((rgb_src - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb[~m] = 1.0                                                # background -> white
    out = np.stack([_upsample(rgb[i * P:(i + 1) * P].reshape(grid, grid, 3)) for i in range(N)])
    return (out * 255).astype(np.uint8)


# ---------------------------------------------------------------- encoder + attention

def build_encoder(ckpt_path, device):
    """Instantiate the right Perceiver class from the checkpoint keys and load weights."""
    sd = torch.load(ckpt_path, map_location="cpu")
    d, n_queries = sd["proj_v.weight"].shape[0], sd["fuse.q"].shape[0]
    if "proj_s.weight" in sd:            # Stage-2 vision+state
        model, kind = MMPerceiver(d=d, n_queries=n_queries), "mm"
    elif "proj_m.weight" in sd:          # chunk vision+motor+ee
        model, kind = MMPerceiverChunks(d=d, n_queries=n_queries), "chunk"
    else:
        raise ValueError(f"unrecognized checkpoint: {ckpt_path}")
    missing, _ = model.load_state_dict(sd, strict=False)  # sigreg buffers may differ; unused here
    need = [k for k in missing if k.startswith(("proj_v", "mod", "fuse"))]
    assert not need, f"missing weights the viz needs: {need}"
    return model.to(device).eval(), kind


def fuse_with_attn(fuse: PerceiverFuse, context, attn_mask):
    """Mirror PerceiverFuse.forward but also return the LAST layer's attention weights
    [B, heads, M, T] (recomputed from the same q/kv projections SDPA hides)."""
    B = context.shape[0]
    x = fuse.q.unsqueeze(0).expand(B, -1, -1)
    last = None
    for ca, n1, ffn, n2 in zip(fuse.ca, fuse.n1, fuse.ffn, fuse.n2):
        xq = n1(x)
        H, hd, M, Tc = ca.num_heads, ca.head_dim, xq.shape[1], context.shape[1]
        q = ca.q(xq).reshape(B, M, H, hd).transpose(1, 2)                     # [B,H,M,hd]
        k = ca.kv(context).reshape(B, Tc, 2, H, hd).permute(2, 0, 3, 1, 4)[0]  # [B,H,T,hd]
        scores = (q @ k.transpose(-2, -1)) / (hd ** 0.5)                      # [B,H,M,T]
        if attn_mask is not None:
            am = attn_mask
            while am.dim() < 4:
                am = am.unsqueeze(0 if am.dim() == 2 else 1)
            scores = scores.masked_fill(am, float("-inf"))
        last = scores.softmax(-1)
        x = x + ca(xq, context, attn_mask=attn_mask)
        x = x + ffn(n2(x))
    return x.mean(1), last


def _vision_ctx_mask(model, kind, patch, device):
    """Vision-only fuse context + attn mask (all non-vision modalities hidden), matching the
    eval latent z_v. -> (ctx [B,T,d], mask [B,M,T] or [M,T])."""
    B = patch.shape[0]
    if kind == "mm":
        ctx = model._context(patch, torch.zeros(B, 28, device=device))       # state dummy (blocked)
        mask = model._mask(block_state=True, device=device)                  # [M,197] state hidden
    else:
        mfeat = model.motor_feats(torch.zeros(B, 8, 3, device=device),
                                  torch.zeros(B, 8, 3, dtype=torch.bool, device=device))
        ctx = model._context(patch, mfeat, torch.zeros(B, 13, 15, device=device))   # [B,217,d]
        mask = model._attn_mask(torch.zeros(B, 8, dtype=torch.bool, device=device),
                                torch.zeros(B, 13, dtype=torch.bool, device=device),
                                hide=("m", "e"))                             # motor+ee hidden
    return ctx, mask


@torch.no_grad()
def encoder_attention(model, kind, patch, device):
    """Panel D: vision cross-attention heat [N,14,14] (queries -> patches, heads+queries avg)."""
    patch = patch.to(device)
    ctx, mask = _vision_ctx_mask(model, kind, patch, device)
    _, attn = fuse_with_attn(model.fuse, ctx, mask)                          # [B,H,M,T]
    heat = attn[..., :GRID * GRID].mean(1).mean(1)                           # 196 vision cols
    return heat.reshape(patch.shape[0], GRID, GRID).float().cpu().numpy()


@torch.no_grad()
def encoder_proj(model, patch, device):
    """Panel B: the encoder's per-patch vision projection proj_v(patch)+mod[0] [N,196,d].
    This is the last point in the encoder where a spatial grid still exists (BEFORE the
    Perceiver fuses+pools to the non-spatial bottleneck). A learned linear re-mix of the
    frozen ViT patches, PCA'd with the SAME method as panel A for a like-for-like ViT-vs-ours
    comparison. It is the trained analog of the PCA-256 vision control in EXPERIMENTS.md."""
    patch = patch.to(device)
    return (model.proj_v(patch) + model.mod[0]).float().cpu()


def load_ft_vit(vit_ckpt, device):
    """Load a finetuned FULL-ViT checkpoint (LeJEPAVideo state) -> the ViTv2 model used for
    panels A/C, so those show the robotics-adapted backbone instead of the pretrained e0.
    (The encoder panels B/D always keep the pretrained e0 the encoder was trained on.)"""
    from world_tokenizer.model import LeJEPAVideo
    net = LeJEPAVideo(pretrained=True)
    ck = torch.load(vit_ckpt, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    net.load_state_dict(sd, strict=False)
    return net.backbone.model.to(device).eval()   # ViTv2PretrainedModel: (x, last_self_attention=True) -> dict


@torch.no_grad()
def head_proj_feats(head_ckpt, patch, device):
    """Frozen-ViT + trained proj_v head (LeJEPAVisionHead): apply its proj_v to the (pretrained
    e0) patch tokens -> [N,196,dh]. The vision-only-trained analog of the multimodal proj_v."""
    ck = torch.load(head_ckpt, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    W, b = sd["proj_v.weight"], sd["proj_v.bias"]
    lin = torch.nn.Linear(W.shape[1], W.shape[0])
    lin.weight.data.copy_(W); lin.bias.data.copy_(b)
    return lin.to(device).eval()(patch.to(device)).float().cpu()


# -------------------------------------------------------------------------------- figure

def _norm01(h):
    return (h - h.min()) / max(h.max() - h.min(), 1e-8)


def _overlay(ax, crop, heat):
    ax.imshow(crop); ax.imshow(_upsample(_norm01(heat)), cmap="inferno", alpha=0.55)


def render(picks, crops, vit_rgb, enc_rgb, vit_attn, enc_attn, title, out_path,
           head_rgb=None, vit_label=None):
    """Columns are built dynamically; the vision-head PCA column appears only when a
    --head-ckpt was given. Each spec is (title, mode, data); mode in img/rgb/overlay."""
    n = len(picks)
    specs = [("image", "img", None),
             (f"A · {vit_label or VIT_NAME} PCA", "rgb", vit_rgb),
             ("B · encoder proj_v PCA", "rgb", enc_rgb)]
    if head_rgb is not None:
        specs.append(("E · vision-head proj_v PCA", "rgb", head_rgb))
    specs += [("C · ViT attention", "overlay", vit_attn),
              ("D · encoder attention", "overlay", enc_attn)]
    ncol = len(specs)
    fig, axes = plt.subplots(n, ncol, figsize=(ncol * 2.6, n * 2.6), squeeze=False)
    for i, (cfg, group, _) in enumerate(picks):
        for j, (_, mode, data) in enumerate(specs):
            ax = axes[i][j]
            if mode == "img":
                ax.imshow(crops[i])
            elif mode == "rgb":
                ax.imshow(data[i])
            else:
                _overlay(ax, crops[i], data[i])
            ax.set_xticks([]); ax.set_yticks([])
        axes[i][0].set_ylabel(short_label(cfg, group), fontsize=7, rotation=0, ha="right", va="center", labelpad=24)
    for j, (t, _, _) in enumerate(specs):
        axes[0][j].set_title(t, fontsize=9)
    fig.suptitle(title, fontsize=11, y=0.999)
    fig.tight_layout(rect=(0.02, 0, 1, 0.99))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({n} images)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trained Perceiver encoder .pt")
    ap.add_argument("--cfgs", default="3", help="[sampling mode] comma list of cfgs, e.g. 1,3,5")
    ap.add_argument("--scenes-csv", help="[list mode] CSV (task,user,cfg,scene) -> one row per scene, "
                    "one representative frame each. Overview across curated scenes.")
    # pinned-scene mode: give all four to visualize N time-strided frames of ONE exact scene
    ap.add_argument("--cfg", type=int, help="pinned scene: config id")
    ap.add_argument("--task", type=int, help="pinned scene: task id")
    ap.add_argument("--user", type=int, help="pinned scene: user id")
    ap.add_argument("--scene", type=int, help="pinned scene: scene id")
    ap.add_argument("--cam-idx", type=int, default=0, help="pinned scene: which camera (default 0)")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--res", type=int, default=448,
                    help="ViT input res for panels A/C: 224->14x14, 448->28x28, 672->42x42. "
                         "Higher = smoother PCA heatmap. Encoder panels B/D stay 224 (trained at 196 tokens).")
    ap.add_argument("--fg-mask", action="store_true",
                    help="DINOv2 recipe for panels A/B: PC1 foreground mask, next-3 PCs -> RGB, white background")
    ap.add_argument("--vit", help="finetuned FULL-ViT checkpoint (LeJEPAVideo) -> panels A/C use this "
                    "robotics-adapted backbone instead of pretrained e0 (encoder B/D keep e0)")
    ap.add_argument("--head-ckpt", help="LeJEPAVisionHead checkpoint -> adds a 'vision-head proj_v PCA' "
                    "column (frozen e0 + this trained proj_v head)")
    ap.add_argument("--split-csv", default="splits/holdout_v1.csv")
    ap.add_argument("--frames-tmpl", default="/mnt/nas/data/RH20T/frames/cfg{cfg}")
    ap.add_argument("--out", default="pca_viz_out/panels.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pinned = [args.cfg, args.task, args.user, args.scene]
    if args.scenes_csv:
        picks = scenes_from_csv(args.scenes_csv, args.cam_idx, args.frames_tmpl)
        src = f"{len(picks)} curated scenes from {args.scenes_csv}"
    elif any(v is not None for v in pinned):
        assert all(v is not None for v in pinned), "pinned mode needs all of --cfg --task --user --scene"
        picks = pinned_frames(args.cfg, args.task, args.user, args.scene, args.n_images,
                              args.cam_idx, args.frames_tmpl)
        src = f"scene {picks[0][1]} cam{args.cam_idx}"
    else:
        cfgs = [int(c) for c in args.cfgs.split(",") if c.strip()]
        picks = sample_frames(args.split_csv, cfgs, args.n_images, args.frames_tmpl, random.Random(args.seed))
        assert picks, "no frames sampled — check --cfgs / --frames-tmpl / --split-csv"
        src = f"cfgs {cfgs} (test split)"
    paths = [p for _, _, p in picks]
    hi = args.res // 16                                                       # ViT patch16 grid side
    print(f"{len(picks)} frames from {src} | ViT panels @ {args.res}px ({hi}x{hi})", flush=True)

    _, crops = load_batch(paths, 224)                                        # 224 display crop (same FOV as hi-res)
    norm_hi, _ = load_batch(paths, args.res)                                 # hi-res for the frozen-ViT panels
    norm_lo, _ = load_batch(paths, 224)                                      # 224 (196 tokens) for the encoder

    e0 = load_vitv2(pretrained=True).to(args.device).eval()                  # pretrained backbone (encoder + default A/C)
    vit_model = load_ft_vit(args.vit, args.device) if args.vit else e0       # panels A/C backbone
    vit_label = os.path.basename(args.vit) if args.vit else VIT_NAME
    with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
        out = vit_model(norm_hi.to(args.device), last_self_attention=True)   # panels A (PCA) + C (attention)
        patch_hi = out["patch_latent"].float().cpu()                         # [N,hi*hi,768]
        vit_attn = out["last_self_attention"].float().mean(1)                # [N,heads,hi*hi] -> [N,hi*hi]
        patch_lo = e0(norm_lo.to(args.device))["patch_latent"].float().cpu()  # [N,196,768] pretrained -> encoder + head
    vit_attn = vit_attn.reshape(len(picks), hi, hi).cpu().numpy()

    model, kind = build_encoder(args.ckpt, args.device)
    enc_attn = encoder_attention(model, kind, patch_lo, args.device)        # panel D
    proj = encoder_proj(model, patch_lo, args.device)                       # panel B: proj_v output (196 tokens)

    vit_rgb = pca_rgb(patch_hi, fg_mask=args.fg_mask)                        # panel A: smooth, hi-res
    enc_rgb = pca_rgb(proj, fg_mask=args.fg_mask)                            # panel B: same PCA method as A
    head_rgb = None
    if args.head_ckpt:                                                       # panel E: frozen e0 + trained head proj_v
        head_rgb = pca_rgb(head_proj_feats(args.head_ckpt, patch_lo, args.device), fg_mask=args.fg_mask)
    title = (f"{os.path.basename(os.path.dirname(args.ckpt))}/{os.path.basename(args.ckpt)}  "
             f"({kind}, ViT={vit_label}@{args.res})")
    render(picks, crops, vit_rgb, enc_rgb, vit_attn, enc_attn, title, args.out,
           head_rgb=head_rgb, vit_label=vit_label)


if __name__ == "__main__":
    main()
