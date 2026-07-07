"""PCA-RGB + attention latent visualization across a robot image (LeJEPA-style).

The LeJEPA paper paints PCA of a ViT's PATCH tokens over the image ("clear semantic
relationships"). Our trained encoders pool 8 Perceiver queries into a single vector
(``PerceiverFuse`` ends in ``x.mean(1)``) — there is no patch grid at the output — so we
show four patch-aligned panels per image, arranged as baseline-vs-ours pairs:

  A  ViT PCA-RGB     — frozen e0 ``patch_latent`` (196x768) -> PCA(3) -> RGB overlay.
                       The paper's exact experiment; model-independent reference.
  B  encoder PCA-RGB — ``proj_v(patch) + mod[0]`` (196xd) -> PCA(3) -> RGB overlay. PCA of
                       *our* latents (proj_v is one linear layer re-mixing the ViT patches).
  C  ViT attention   — e0 last-layer CLS->patch self-attention (DINO-style), heads averaged.
  D  encoder attention — vision-only fuse pass; softmax(q.k^T) of the 8 queries over the
                       196 vision patches. What our fused bottleneck READS from the image.

A|B compare PCA structure (baseline vs ours); C|D compare where attention lands (baseline
vs ours). The encoder class is picked from the checkpoint keys (proj_s -> MMPerceiver,
proj_m -> MMPerceiverChunks). Motor/ee/state are dummy tensors the vision-only pass blocks,
so their values never reach the vision panels. Samples come from the holdout_v1 TEST split.

  python -m world_tokenizer.pca_viz \
      --ckpt /mnt/nas/data/RH20T/checkpoints/phase1/all/seed0.pt \
      --cfgs 1,3,5 --n-images 6 --out pca_viz_out/chunk_all.png

  python -m world_tokenizer.pca_viz \
      --ckpt /mnt/nas/data/RH20T/checkpoints/exp-20260704-032544/perceiver_seed0.pt \
      --cfgs 3 --n-images 6 --out pca_viz_out/stage2_seed0.png
"""
import argparse
import csv
import glob
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from world_tokenizer.mm_perceiver import MMPerceiver, PerceiverFuse
from world_tokenizer.mm_perceiver2 import MMPerceiverChunks
from world_tokenizer.model import load_vitv2

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


# ------------------------------------------------------------------------------- images

def load_batch(paths):
    """-> (normalized [N,3,224,224] for the model, display crops [N,224,224,3] in 0..1)."""
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    crops = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    return (crops - _NORM_M) / _NORM_S, crops.permute(0, 2, 3, 1).numpy()


def _upsample(grid, size=224, mode=Image.BICUBIC):
    """[14,14,(3)] float 0..1 -> [224,224,(3)] float 0..1 via PIL resize."""
    is_rgb = grid.ndim == 3
    arr = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="RGB" if is_rgb else "L").resize((size, size), mode)
    return np.asarray(im).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------------- PCA

def pca_rgb(feats):
    """feats [N,196,C] -> [N,224,224,3] uint8. Joint PCA(3) across all patches for
    cross-image color comparability; robust per-channel 2-98 percentile normalization."""
    N, P, C = feats.shape
    flat = feats.reshape(-1, C).float()
    flat = flat - flat.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(flat, full_matrices=False)        # right singular vecs = principal axes
    proj = (flat @ V[:3].T).reshape(N, P, 3).numpy()             # project onto top-3
    lo, hi = np.percentile(proj, 2, axis=(0, 1)), np.percentile(proj, 98, axis=(0, 1))
    proj = np.clip((proj - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    out = np.stack([_upsample(proj[i].reshape(GRID, GRID, 3)) for i in range(N)])
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


@torch.no_grad()
def encoder_panels(model, kind, patch, device):
    """-> (proj_v patch tokens [N,196,d] for panel B, vision attention heat [N,14,14] for D).
    The fuse pass hides all non-vision modalities, matching the eval latent z_v."""
    patch = patch.to(device)
    B = patch.shape[0]
    proj = model.proj_v(patch) + model.mod[0]                                # [B,196,d]
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
    _, attn = fuse_with_attn(model.fuse, ctx, mask)                          # [B,H,M,T]
    heat = attn[..., :GRID * GRID].mean(1).mean(1)                           # 196 vision cols, over heads+queries
    return proj.float().cpu(), heat.reshape(B, GRID, GRID).float().cpu().numpy()


# -------------------------------------------------------------------------------- figure

def _norm01(h):
    return (h - h.min()) / max(h.max() - h.min(), 1e-8)


def render(picks, crops, vit_rgb, enc_rgb, vit_attn, enc_attn, title, out_path):
    n = len(picks)
    cols = ["A · ViT PCA", "B · encoder PCA", "C · ViT attention", "D · encoder attention"]
    fig, axes = plt.subplots(n, 5, figsize=(5 * 2.6, n * 2.6), squeeze=False)
    for i, (cfg, group, _) in enumerate(picks):
        axes[i][0].imshow(crops[i])
        axes[i][0].set_ylabel(f"cfg{cfg}\n{group[:22]}", fontsize=7, rotation=0, ha="right", va="center", labelpad=28)
        axes[i][1].imshow(vit_rgb[i])
        axes[i][2].imshow(enc_rgb[i])
        axes[i][3].imshow(crops[i]); axes[i][3].imshow(_upsample(_norm01(vit_attn[i])), cmap="inferno", alpha=0.55)
        axes[i][4].imshow(crops[i]); axes[i][4].imshow(_upsample(_norm01(enc_attn[i])), cmap="inferno", alpha=0.55)
        for j in range(5):
            axes[i][j].set_xticks([]); axes[i][j].set_yticks([])
    axes[0][0].set_title("image", fontsize=9)
    for j, c in enumerate(cols):
        axes[0][j + 1].set_title(c, fontsize=9)
    fig.suptitle(title, fontsize=11, y=0.999)
    fig.tight_layout(rect=(0.02, 0, 1, 0.99))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({n} images)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trained Perceiver encoder .pt")
    ap.add_argument("--cfgs", default="3", help="comma list of cfgs to sample from, e.g. 1,3,5")
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--split-csv", default="splits/holdout_v1.csv")
    ap.add_argument("--frames-tmpl", default="/mnt/nas/data/RH20T/frames/cfg{cfg}")
    ap.add_argument("--out", default="pca_viz_out/panels.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfgs = [int(c) for c in args.cfgs.split(",") if c.strip()]
    rng = random.Random(args.seed)
    picks = sample_frames(args.split_csv, cfgs, args.n_images, args.frames_tmpl, rng)
    assert picks, "no frames sampled — check --cfgs / --frames-tmpl / --split-csv"
    print(f"{len(picks)} frames from cfgs {cfgs} (test split)", flush=True)

    norm, crops = load_batch([p for _, _, p in picks])

    e0 = load_vitv2(pretrained=True).to(args.device).eval()
    with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == "cuda"):
        out = e0(norm.to(args.device), last_self_attention=True)
        patch = out["patch_latent"].float().cpu()                            # [N,196,768]
        vit_attn = out["last_self_attention"].float().mean(1)                # [N,heads,196] -> [N,196]
    vit_attn = vit_attn.reshape(len(picks), GRID, GRID).cpu().numpy()

    model, kind = build_encoder(args.ckpt, args.device)
    proj, enc_attn = encoder_panels(model, kind, patch, args.device)

    vit_rgb = pca_rgb(patch)
    enc_rgb = pca_rgb(proj)
    title = f"{os.path.basename(os.path.dirname(args.ckpt))}/{os.path.basename(args.ckpt)}  ({kind})"
    render(picks, crops, vit_rgb, enc_rgb, vit_attn, enc_attn, title, args.out)


if __name__ == "__main__":
    main()
