"""Clear MDS figure: redraw the SigLIP distance embedding without thumbnails.

The thumbnail version (mds18.png) is intuitive but the images overlap and the
first two axes only capture part of the distance matrix. Three panels:

  left    colored by camera — MDS-1 is essentially viewpoint
  middle  same coordinates colored by layout — fixed layouts stretch across the
          camera axis instead of forming islands
  right   MDS-1 vs MDS-3 — no clean hidden layout axis either

Reads cached pair distances from results/encoder_pairs.json, so it needs no GPU.

Usage:
  python probe_mds.py
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

CAMS = list("ABCDEF")
LAYS = ["1", "2", "3"]
STEMS = [c + l for l in LAYS for c in CAMS]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
CAMC = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eda100", "D": "#008300",
        "E": "#4a3aa7", "F": "#e34948"}
LAYC = {"1": "#2a78d6", "2": "#e34948", "3": "#52514e"}
MARK = {"1": "o", "2": "s", "3": "^"}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": INK2,
    "font.size": 11,
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--figures", default=os.path.join(HERE, "figures"))
    p.add_argument("--encoder", default="siglip")
    return p.parse_args()


def classical_mds(dist, dims=3):
    n = dist.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (dist ** 2) @ j
    w, v = np.linalg.eigh(b)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    pos = w[w > 0]
    coords = v[:, :dims] * np.sqrt(np.maximum(w[:dims], 0))
    explained = w[:dims] / max(pos.sum(), 1e-12)
    return coords, explained


def load_distances(pairs_json, encoder):
    pairs = json.load(open(pairs_json))[encoder]
    n = len(STEMS)
    d = np.zeros((n, n))
    for i, a in enumerate(STEMS):
        for j, b in enumerate(STEMS):
            if i == j:
                continue
            d[i, j] = pairs.get(f"{a}-{b}", pairs.get(f"{b}-{a}"))
    return d


def setup_axis(ax, x, y, xlabel, ylabel, title):
    ax.axhline(0, color=GRID, lw=0.9, zorder=0)
    ax.axvline(0, color=GRID, lw=0.9, zorder=0)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, fontsize=12, pad=10)
    pad_x = 0.12 * max(np.ptp(x), 1e-6)
    pad_y = 0.12 * max(np.ptp(y), 1e-6)
    ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
    ax.set_ylim(y.min() - pad_y, y.max() + pad_y)


def label_points(ax, xy, dx=0.007, dy=0.007):
    for i, s in enumerate(STEMS):
        ax.text(xy[i, 0] + dx, xy[i, 1] + dy, s, fontsize=8.5,
                color=INK, zorder=10)


def draw_camera_panel(ax, coords, expl, dims, title, show_family_labels=False):
    x, y = coords[:, dims[0]], coords[:, dims[1]]
    setup_axis(ax, x, y,
               f"MDS-{dims[0] + 1} ({expl[dims[0]]:.0%})",
               f"MDS-{dims[1] + 1} ({expl[dims[1]]:.0%})",
               title)
    for cam in CAMS:
        idx = [STEMS.index(cam + lay) for lay in LAYS]
        ax.plot(x[idx], y[idx], color=CAMC[cam], lw=1.7, alpha=0.65, zorder=2)
    for i, s in enumerate(STEMS):
        ax.scatter(x[i], y[i], s=80, marker=MARK[s[1]], color=CAMC[s[0]],
                   edgecolor="white", linewidth=1.0, zorder=5)
    label_points(ax, np.c_[x, y])
    if show_family_labels:
        left = np.mean(coords[[STEMS.index(c + l)
                               for c in "DEF" for l in LAYS], :], axis=0)
        right = np.mean(coords[[STEMS.index(c + l)
                                for c in "ABC" for l in LAYS], :], axis=0)
        ax.text(left[dims[0]], left[dims[1]] - 0.16, "side views D-F",
                color=INK2, ha="center", fontsize=10)
        ax.text(right[dims[0]], right[dims[1]] + 0.16, "overhead views A-C",
                color=INK2, ha="center", fontsize=10)


def draw_layout_panel(ax, coords, expl):
    x, y = coords[:, 0], coords[:, 1]
    setup_axis(ax, x, y, f"MDS-1 ({expl[0]:.0%})", f"MDS-2 ({expl[1]:.0%})",
               "same coordinates, colored by layout")
    for lay in LAYS:
        idx = [STEMS.index(cam + lay) for cam in CAMS]
        ax.plot(x[idx], y[idx], color=LAYC[lay], lw=1.7, alpha=0.45, zorder=2)
    for i, s in enumerate(STEMS):
        ax.scatter(x[i], y[i], s=80, marker=MARK[s[1]], color=LAYC[s[1]],
                   edgecolor="white", linewidth=1.0, zorder=5)
    label_points(ax, coords[:, :2])


def add_legends(fig):
    cam_handles = [Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=CAMC[c], markeredgecolor="white",
                          markersize=9, label=f"camera {c}") for c in CAMS]
    lay_handles = [Line2D([0], [0], marker=MARK[l], color="none",
                          markerfacecolor=LAYC[l], markeredgecolor="white",
                          markersize=9, label=f"layout {l}") for l in LAYS]
    fig.legend(handles=cam_handles, loc="lower left", bbox_to_anchor=(0.06, 0.02),
               ncols=6, frameon=False, fontsize=9.5)
    fig.legend(handles=lay_handles, loc="lower right", bbox_to_anchor=(0.94, 0.02),
               ncols=3, frameon=False, fontsize=9.5)


def main():
    args = parse_args()
    os.makedirs(args.figures, exist_ok=True)
    pairs_json = os.path.join(args.results, "encoder_pairs.json")
    if not os.path.exists(pairs_json):
        raise FileNotFoundError(f"{pairs_json} — run probe_encoders.py first")

    D = load_distances(pairs_json, args.encoder)
    XYZ, EXP = classical_mds(D, dims=3)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8))
    draw_camera_panel(axes[0], XYZ, EXP, (0, 1), "viewpoint dominates MDS-1",
                      show_family_labels=True)
    draw_layout_panel(axes[1], XYZ, EXP)
    draw_camera_panel(axes[2], XYZ, EXP, (0, 2),
                      "MDS-3 check: no clean hidden layout axis")

    fig.suptitle("18-image SigLIP distance embedding, redrawn without thumbnails",
                 fontsize=14, y=0.98)
    fig.text(0.5, 0.09,
             "Lines connect either the three layouts from one camera (left/right) "
             "or the six cameras for one fixed layout (middle).",
             ha="center", color=INK2, fontsize=10)
    add_legends(fig)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93), w_pad=2.0)
    path = os.path.join(args.figures, "mds18_clear.png")
    fig.savefig(path, dpi=170)
    print("saved", path)


if __name__ == "__main__":
    main()
