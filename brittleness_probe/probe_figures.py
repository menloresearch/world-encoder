"""Two SigLIP-only figures that go with the main probe.

  affinity_bars.png    where a patch's nearest neighbour lives (same camera vs
                       same layout vs unrelated image), read from
                       results/encoder_groups.json
  bezel_tracking.png   one token on the orange bezel in A1, matched into other
                       views — object identity survives modest view changes and
                       drifts to the robot's orange joints across the 90° re-rig

Run probe_encoders.py first (this reads its encoder_groups.json).

Usage:
  python probe_figures.py --img-dir /path/to/gridset
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import SiglipImageProcessor, SiglipVisionModel

HERE = os.path.dirname(os.path.abspath(__file__))

CAMS = list("ABCDEF")
LAYS = ["1", "2", "3"]
STEMS = [c + l for l in LAYS for c in CAMS]
MID = "google/siglip-so400m-patch14-224"

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "text.color": INK, "axes.edgecolor": INK2,
                     "font.size": 11})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-dir", default=os.environ.get(
        "GRIDSET_DIR", os.path.join(HERE, "gridset")))
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--figures", default=os.path.join(HERE, "figures"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_images(img_dir):
    files = {os.path.basename(f)[:2]: f
             for f in glob.glob(os.path.join(img_dir, "*.JPG"))}
    missing = sorted(set(STEMS) - set(files))
    if missing:
        raise FileNotFoundError(
            f"missing grid images {missing} in {img_dir} (see README.md)")
    return {s: ImageOps.exif_transpose(Image.open(files[s])).convert("RGB")
            for s in STEMS}


def siglip_tokens(imgs, device):
    proc = SiglipImageProcessor.from_pretrained(MID)
    model = SiglipVisionModel.from_pretrained(MID).to(device).eval()
    tok = {}
    with torch.no_grad():
        for s in STEMS:
            x = proc(images=imgs[s].resize((224, 224), Image.BICUBIC),
                     return_tensors="pt").pixel_values.to(device)
            tok[s] = model(pixel_values=x).last_hidden_state[0].float().cpu()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return tok


def fig_affinity(groups_json, path):
    p = json.load(open(groups_json))["siglip"]["nn_per_partner"]
    vals = [p["same_camera"] * 100, p["same_layout"] * 100, p["other"] * 100]
    labs = ["same camera,\nobjects differ", "same objects,\ncamera differs",
            "unrelated image"]
    cols = ["#eda100", "#2a78d6", "#52514e"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(labs, vals, 0.55, color=cols, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.0f}%", ha="center",
                fontsize=12, color=INK)
    ax.axhline(100 / 17, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=4)
    ax.text(2.35, 100 / 17 + 0.35, "chance (5.9%)", fontsize=9.5, color=INK2,
            ha="right")
    ax.set_ylabel("% of an image's patches whose nearest\nneighbor is in that "
                  "partner (per partner)", color=INK2, fontsize=10)
    ax.set_ylim(0, 18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
    ax.set_title("Where a patch's nearest neighbor lives (SigLIP; "
                 "14–17 / 9–10 / 2 across all four encoders)", fontsize=11,
                 pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def find_bezel_patch(img, grid):
    """Most orange patch in A1's table region. The arms' orange joints live in
    the top rows, so the search is masked to the table."""
    a1 = np.asarray(img.resize((224, 224), Image.BICUBIC), dtype=float)
    pr = a1.reshape(grid, 224 // grid, grid, 224 // grid, 3).mean((1, 3))
    orangeness = pr[..., 0] - 0.5 * (pr[..., 1] + pr[..., 2])
    mask = np.full((grid, grid), -1e9)
    mask[5:14, 1:10] = 0
    r0, c0 = np.unravel_index(np.argmax(orangeness + mask), (grid, grid))
    return int(r0), int(c0)


def fig_bezel_tracking(tok, imgs, grid, path):
    r0, c0 = find_bezel_patch(imgs["A1"], grid)
    bez = r0 * grid + c0
    q = F.normalize(tok["A1"][bez], dim=-1)

    show = ["A1", "C1", "E1", "F1", "B2", "D3"]
    W, H = 448, 336
    pw, ph = W / grid, H / grid
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.8),
                             gridspec_kw={"hspace": 0.16})
    for k, s in enumerate(show):
        ax = axes[k // 3][k % 3]
        ax.imshow(imgs[s].resize((W, H), Image.BICUBIC), extent=[0, W, H, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        sims = (F.normalize(tok[s], dim=-1) @ q).numpy()
        if s == "A1":
            sims[bez] = -2  # show where the SECOND best is on the query image
            ax.add_patch(plt.Rectangle((c0 * pw, r0 * ph), pw, ph, fill=False,
                                       edgecolor="#e34948", lw=2.6))
            lab = "query"
        else:
            lab = f"best match (sim {sims.max():.2f})"
        rb, cb = divmod(int(sims.argmax()), grid)
        ax.add_patch(plt.Circle(((cb + 0.5) * pw, (rb + 0.5) * ph), 14,
                                fill=False, edgecolor="#e34948", lw=2.6))
        ax.set_title(f"{s} — {lab}", fontsize=11, color=INK)
    fig.suptitle("Red square = one token on the bezel in A1 · red circle = the "
                 "single best-matching token in each view", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=155)
    plt.close(fig)
    print(f"query patch row {r0} col {c0}")


def main():
    args = parse_args()
    os.makedirs(args.figures, exist_ok=True)
    groups_json = os.path.join(args.results, "encoder_groups.json")
    if not os.path.exists(groups_json):
        raise FileNotFoundError(f"{groups_json} — run probe_encoders.py first")

    imgs = load_images(args.img_dir)
    tok = siglip_tokens(imgs, args.device)
    grid = int(np.sqrt(tok["A1"].shape[0]))

    fig_affinity(groups_json, os.path.join(args.figures, "affinity_bars.png"))
    fig_bezel_tracking(tok, imgs, grid,
                       os.path.join(args.figures, "bezel_tracking.png"))
    print("saved affinity_bars.png, bezel_tracking.png")


if __name__ == "__main__":
    main()
