"""Camera-vs-placement brittleness probe over four VLA vision encoders.

18 photos of a two-arm tabletop rig: 6 camera positions (A-C overhead, D-F at
table height) x 3 object layouts. Every pair of photos differs in exactly one
of three ways:

  placement  same camera, objects moved      (18 pairs)
  camera     same objects, camera moved      (45 pairs)
  both       both changed                    (90 pairs)

Metric: mean cosine distance between corresponding tokens of the patch grid at
224px (uniform squash for every encoder), i.e. the visual state a VLA's
language model consumes. A 10% brightness change on one image is carried as a
scale anchor.

Outputs (JSON to --results, PNG to --figures):
  encoder_groups.json  per-encoder group means + NN affinity
  encoder_pairs.json   all 153 pairwise distances per encoder
  grid_images.png      contact sheet of the 18 photos
  heatmap18.png        18x18 distance matrix (SigLIP)
  mds18.png            MDS map with thumbnails (SigLIP)
  kmeans18.png         k-means patch clusters painted back onto 6 views
  encoders.png         camera-vs-placement bars for all four encoders

Usage:
  python probe_encoders.py --img-dir /path/to/gridset
"""
import argparse
import glob
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from PIL import Image, ImageEnhance, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))

CAMS = list("ABCDEF")
LAYS = ["1", "2", "3"]
STEMS = [c + l for l in LAYS for c in CAMS]  # layout-major: A1..F1, A2..F2, ...

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
CAMC = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eda100", "D": "#008300",
        "E": "#4a3aa7", "F": "#e34948"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
       "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "text.color": INK, "axes.edgecolor": INK2,
                     "font.size": 11})

ENCODERS = ["siglip", "siglip2", "dinov2", "clip"]
LABEL = {"siglip": "SigLIP\nSo400m", "siglip2": "SigLIP2\nSo400m",
         "dinov2": "DINOv2\nViT-L", "clip": "CLIP\nViT-L"}

PLACE = [(c + a, c + b) for c in CAMS
         for a, b in itertools.combinations(LAYS, 2)]          # 18
CAMONLY = [(a + l, b + l) for l in LAYS
           for a, b in itertools.combinations(CAMS, 2)]        # 45
ALLP = list(itertools.combinations(STEMS, 2))                  # 153
BOTH = [p for p in ALLP if p not in PLACE and p not in CAMONLY]  # 90


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-dir", default=os.environ.get(
        "GRIDSET_DIR", os.path.join(HERE, "gridset")),
        help="directory of the 18 rig photos, named <cam><layout>*.JPG")
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--figures", default=os.path.join(HERE, "figures"))
    p.add_argument("--encoders", nargs="+", default=ENCODERS, choices=ENCODERS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_images(img_dir):
    files = {}
    for f in glob.glob(os.path.join(img_dir, "*.JPG")):
        files[os.path.basename(f)[:2]] = f
    missing = sorted(set(STEMS) - set(files))
    if missing:
        raise FileNotFoundError(
            f"missing grid images {missing} in {img_dir} (see README.md)")
    imgs = {s: ImageOps.exif_transpose(Image.open(files[s])).convert("RGB")
            for s in STEMS}
    sq224 = {s: imgs[s].resize((224, 224), Image.BICUBIC) for s in STEMS}
    bright = ImageEnhance.Brightness(sq224["A1"]).enhance(0.9)  # scale anchor
    return imgs, sq224, bright


def hf_tokens(model, proc, sq224, bright, dev):
    toks = {}
    with torch.no_grad():
        for s in STEMS + ["A1_bright"]:
            im = bright if s == "A1_bright" else sq224[s]
            x = proc(images=im, return_tensors="pt").pixel_values.to(dev)
            h = model(pixel_values=x).last_hidden_state[0].float().cpu()
            toks[s] = h
    return toks


def load_dinov2(dev):
    """Local torch.hub checkout if present, else fetch from GitHub."""
    local = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    if os.path.isdir(local):
        return torch.hub.load(local, "dinov2_vitl14", source="local",
                              pretrained=True).to(dev).eval()
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(dev).eval()


def encode(name, sq224, bright, dev):
    if name in ("siglip", "siglip2"):
        # fixed-res siglip2 ckpts ship a siglip_vision_model config, so both
        # load through SiglipVisionModel
        from transformers import SiglipImageProcessor, SiglipVisionModel
        mid = ("google/siglip-so400m-patch14-224" if name == "siglip"
               else "google/siglip2-so400m-patch14-224")
        m = SiglipVisionModel.from_pretrained(mid).to(dev).eval()
        t = hf_tokens(m, SiglipImageProcessor.from_pretrained(mid),
                      sq224, bright, dev)
    elif name == "clip":
        from transformers import CLIPImageProcessor, CLIPVisionModel
        mid = "openai/clip-vit-large-patch14"
        m = CLIPVisionModel.from_pretrained(mid).to(dev).eval()
        t = hf_tokens(m, CLIPImageProcessor.from_pretrained(mid),
                      sq224, bright, dev)
        t = {k: v[1:] for k, v in t.items()}  # drop CLS
    elif name == "dinov2":
        from torchvision import transforms
        m = load_dinov2(dev)
        tr = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        t = {}
        with torch.no_grad():
            for s in STEMS + ["A1_bright"]:
                im = bright if s == "A1_bright" else sq224[s]
                o = m.forward_features(tr(im)[None].to(dev))
                t[s] = o["x_norm_patchtokens"][0].float().cpu()
    else:
        raise ValueError(name)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return t


def aligned(toks, a, b):
    return float((1 - F.cosine_similarity(toks[a], toks[b], dim=-1)).mean())


def nn_affinity(toks):
    """For each patch, which other image holds its nearest neighbour?
    Normalized per candidate partner (2 same-camera, 5 same-layout, 10 other),
    so chance is 1/17 = 5.9% everywhere."""
    normed = {s: F.normalize(toks[s], dim=-1) for s in STEMS}
    mass_cam, mass_lay, mass_oth = [], [], []
    for s in STEMS:
        n = normed[s].shape[0]
        best_sim = torch.full((n,), -2.0)
        best_src = [""] * n
        for o in STEMS:
            if o == s:
                continue
            sim = (normed[s] @ normed[o].T).max(1).values
            upd = sim > best_sim
            for k in torch.nonzero(upd).flatten().tolist():
                best_src[k] = o
            best_sim[upd] = sim[upd]
        srcs = np.array(best_src)
        same_cam = np.mean([x[0] == s[0] for x in srcs])
        same_lay = np.mean([x[1] == s[1] for x in srcs])
        mass_cam.append(same_cam / 2)
        mass_lay.append(same_lay / 5)
        mass_oth.append((1 - same_cam - same_lay) / 10)
    return {"same_camera": float(np.mean(mass_cam)),
            "same_layout": float(np.mean(mass_lay)),
            "other": float(np.mean(mass_oth))}


def group_stats(toks):
    pairs = {f"{a}-{b}": aligned(toks, a, b) for a, b in ALLP}
    gr = {}
    for gname, plist in [("placement", PLACE), ("camera", CAMONLY),
                         ("both", BOTH)]:
        vals = [pairs[f"{a}-{b}"] for a, b in plist]
        gr[gname] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                     "max": float(np.max(vals)), "n": len(vals)}
    gr["brightness_anchor"] = aligned(toks, "A1", "A1_bright")
    gr["ratio_cam_over_place"] = gr["camera"]["mean"] / gr["placement"]["mean"]
    gr["nn_per_partner"] = nn_affinity(toks)
    return pairs, gr


def report(enc, g):
    print(f"\n== {enc} ==")
    print(f"  brightness anchor        {g['brightness_anchor']:.3f}")
    print(f"  placement (n=18)  mean {g['placement']['mean']:.3f}  "
          f"[{g['placement']['min']:.3f}..{g['placement']['max']:.3f}]")
    print(f"  camera    (n=45)  mean {g['camera']['mean']:.3f}  "
          f"[{g['camera']['min']:.3f}..{g['camera']['max']:.3f}]")
    print(f"  both      (n=90)  mean {g['both']['mean']:.3f}  "
          f"[{g['both']['min']:.3f}..{g['both']['max']:.3f}]")
    print(f"  camera/placement ratio {g['ratio_cam_over_place']:.2f}   "
          f"min-camera vs max-placement: {g['camera']['min']:.3f} vs "
          f"{g['placement']['max']:.3f}")
    p = g["nn_per_partner"]
    print(f"  NN mass per partner: same-cam {p['same_camera']*100:.1f}%  "
          f"same-layout {p['same_layout']*100:.1f}%  other "
          f"{p['other']*100:.1f}%")


def fig_contact_sheet(imgs, path):
    fig, axes = plt.subplots(3, 6, figsize=(13.2, 8.2))
    for r, l in enumerate(LAYS):
        for c, cam in enumerate(CAMS):
            ax = axes[r][c]
            ax.imshow(imgs[cam + l].resize((300, 225), Image.BICUBIC))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(CAMC[cam])
                sp.set_linewidth(2.5)
            if r == 0:
                ax.set_title(f"cam {cam}", fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(f"layout {l}", fontsize=11, color=INK)
    fig.suptitle("18 photos: 6 camera positions (columns) × 3 object layouts "
                 "(rows)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def distance_matrix(pairs):
    n = len(STEMS)
    d = np.zeros((n, n))
    for i, a in enumerate(STEMS):
        for j, b in enumerate(STEMS):
            if i != j:
                d[i, j] = pairs.get(f"{a}-{b}", pairs.get(f"{b}-{a}", 0))
    return d


def fig_heatmap(D, path):
    M = D.copy()
    np.fill_diagonal(M, np.nan)
    fig, ax = plt.subplots(figsize=(9.4, 8.2))
    im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(len(STEMS)), STEMS, fontsize=8)
    ax.set_yticks(range(len(STEMS)), STEMS, fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for k in [5.5, 11.5]:
        ax.axhline(k, color=SURFACE, lw=3)
        ax.axvline(k, color=SURFACE, lw=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.outline.set_visible(False)
    cb.set_label("aligned token-grid cosine distance (SigLIP)", color=INK2)
    ax.set_title("The three 6×6 blocks on the diagonal are SAME-LAYOUT, "
                 "camera-only pairs;\nthe off-block diagonals are same-camera "
                 "placement pairs (the light stripes)", fontsize=10.5, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_mds_thumbnails(D, imgs, path):
    n = len(STEMS)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    XY = V[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    expl = w[idx].sum() / np.maximum(w[w > 0].sum(), 1e-9)
    fig, ax = plt.subplots(figsize=(10.4, 8.6))
    for i, s in enumerate(STEMS):
        th = imgs[s].resize((84, 63), Image.BICUBIC)
        ab = AnnotationBbox(OffsetImage(np.asarray(th)), XY[i], frameon=True,
                            pad=0.1, zorder=2 + i,
                            bboxprops=dict(edgecolor=CAMC[s[0]], lw=2.5,
                                           facecolor=SURFACE))
        ax.add_artist(ab)
        ax.annotate(s, XY[i], xytext=(-34, 20), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=INK, zorder=40,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE,
                              edgecolor=CAMC[s[0]], lw=1))
    pad = 0.14 * (XY.max(0) - XY.min(0)).max()
    ax.set_xlim(XY[:, 0].min() - pad, XY[:, 0].max() + pad)
    ax.set_ylim(XY[:, 1].min() - pad, XY[:, 1].max() + pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Line2D([], [], color=CAMC[c], lw=3, label=f"camera {c}")
               for c in CAMS]
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=10,
              ncols=2, labelcolor=INK)
    ax.set_title(f"18 images placed by SigLIP latent distance (classical MDS,"
                 f" {expl:.0%} of variance) — border color = camera",
                 fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_kmeans(toks, imgs, path):
    from sklearn.cluster import KMeans
    grid = int(np.sqrt(toks["A1"].shape[0]))
    allt = torch.cat([F.normalize(toks[s], dim=-1) for s in STEMS]).numpy()
    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(allt)
    seg = km.labels_.reshape(len(STEMS), grid, grid)
    segc = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
    show = ["A1", "C1", "E1", "A2", "C3", "F1"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.8),
                             gridspec_kw={"hspace": 0.16})
    for k, s in enumerate(show):
        ax = axes[k // 3][k % 3]
        base = np.asarray(imgs[s].resize((448, 336), Image.BICUBIC).convert("L"))
        ax.imshow(base, cmap="gray", extent=[0, 448, 336, 0])
        ax.imshow(seg[STEMS.index(s)], cmap=ListedColormap(segc), vmin=-0.5,
                  vmax=5.5, alpha=0.6, interpolation="nearest",
                  extent=[0, 448, 336, 0])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(s, fontsize=12, color=INK)
    handles = [Patch(facecolor=segc[k], label=f"cluster {k+1}")
               for k in range(6)]
    fig.legend(handles=handles, loc="lower center", ncols=6, frameon=False,
               fontsize=10)
    fig.suptitle("k-means (k=6) over all 4,608 patches, painted back — same "
                 "content clusters in every view", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_encoders(results, encs, path):
    conds = [("placement", "objects moved\n(same camera)", "#2a78d6"),
             ("camera", "camera moved\n(same objects)", "#eda100"),
             ("both", "both changed", "#52514e")]
    x = np.arange(len(encs))
    wd = 0.26
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for k, (key, lab, col) in enumerate(conds):
        vals = [results[e]["groups"][key]["mean"] for e in encs]
        bars = ax.bar(x + (k - 1) * wd, vals, wd * 0.92, color=col, label=lab,
                      zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                    ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [LABEL[e] for e in encs])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("mean token-grid cosine distance", color=INK2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncols=3)
    ax.set_title("Camera > placement, in every encoder VLAs use", fontsize=12,
                 pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    imgs, sq224, bright = load_images(args.img_dir)

    results = {}
    tokens_siglip = None
    for enc in args.encoders:
        try:
            toks = encode(enc, sq224, bright, args.device)
        except Exception as e:
            print(f"!! {enc} failed: {type(e).__name__}: {e}")
            continue
        if enc == "siglip":
            tokens_siglip = toks
        pairs, gr = group_stats(toks)
        results[enc] = {"groups": gr, "pairs": pairs}
        report(enc, gr)

    with open(os.path.join(args.results, "encoder_groups.json"), "w") as f:
        json.dump({e: r["groups"] for e, r in results.items()}, f, indent=2)
    with open(os.path.join(args.results, "encoder_pairs.json"), "w") as f:
        json.dump({e: r["pairs"] for e, r in results.items()}, f, indent=2)

    fig_contact_sheet(imgs, os.path.join(args.figures, "grid_images.png"))
    if tokens_siglip is not None:
        D = distance_matrix(results["siglip"]["pairs"])
        fig_heatmap(D, os.path.join(args.figures, "heatmap18.png"))
        fig_mds_thumbnails(D, imgs, os.path.join(args.figures, "mds18.png"))
        fig_kmeans(tokens_siglip, imgs, os.path.join(args.figures, "kmeans18.png"))
    encs = [e for e in args.encoders if e in results]
    if encs:
        fig_encoders(results, encs, os.path.join(args.figures, "encoders.png"))

    print("\nfigures:", sorted(os.listdir(args.figures)))


if __name__ == "__main__":
    main()
