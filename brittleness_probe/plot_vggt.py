"""Figures for the VGGT / VGGT-Omega probe JSONs written by probe_vggt.py.

Three PNGs per model:
  <prefix>_single_image.png              single-image encoding alone
  <prefix>_single_vs_multiview.png       single-image next to all-18-view
  <prefix>_frame_global_patch_split.png  patch tokens by frame vs global stream

Numbers printed above each group are camera/object-layout distance ratios.

Usage:
  python plot_vggt.py --model vggt
  python plot_vggt.py --model omega
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
COLS = {"placement": "#2a78d6", "camera": "#eda100", "both": "#52514e"}
LEGEND = {"placement": "objects moved", "camera": "camera moved",
          "both": "both changed"}
GROUPS = ["placement", "camera", "both"]

SUMMARY_METRICS = [
    ("camera_token", "camera\ntoken"),
    ("register_tokens", "register\ntokens"),
    ("patch_tokens_aligned", "patch\ntokens"),
    ("all_tokens_aligned", "all\ntokens"),
]
SPLIT_METRICS = [
    ("frame_patch_tokens_aligned", "frame stream\npatches"),
    ("global_patch_tokens_aligned", "global stream\npatches"),
    ("patch_tokens_aligned", "concatenated\npatches"),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": INK2,
    "font.size": 11,
})

MODELS = {
    "vggt": {
        "prefix": "vggt",
        "summary": {
            "single_ylim": 0.073, "single_ratio_y": 0.064,
            "multi_ylim": 0.50, "multi_ratio_y": 0.455,
            "sharey": False, "label_pad_frac": 0.02, "ratio_fmt": "{:.1f}x",
            "single_title": "VGGT-1B single-image features: camera still moves "
                            "farther than layout",
            "suptitle": "VGGT-1B: multi-view reconstruction makes camera "
                        "identity more explicit",
            "footer": "Multi-view here means VGGT sees all 18 views jointly at "
                      "inference, not VLA training.",
        },
        "split": {
            "single_ylim": 0.12, "single_ratio_y": 0.105,
            "multi_ylim": 0.38, "multi_ratio_y": 0.34,
            "sharey": False, "label_pad_frac": 0.025, "ratio_fmt": "{:.2f}x",
            "suptitle": "VGGT-1B patch tokens split by frame vs "
                        "global-attention stream",
            "footer": "The global stream is the second half of VGGT's "
                      "concatenated token.",
        },
    },
    "omega": {
        "prefix": "vggt_omega",
        "summary": {
            "single_ylim": 0.86, "single_ratio_y": 0.77,
            "multi_ylim": 0.86, "multi_ratio_y": 0.77,
            "sharey": True, "label_pad_frac": 0.02, "ratio_fmt": "{:.1f}x",
            "single_title": "VGGT-Omega single-image features: patch tokens "
                            "remain camera-sensitive",
            "suptitle": "VGGT-Omega-1B-512: camera/register tokens are "
                        "pose-separated, patches still track viewpoint",
            "footer": "Multi-view here means VGGT-Omega sees all 18 views "
                      "jointly at inference.",
        },
        "split": {
            "single_ylim": 0.92, "single_ratio_y": 0.84,
            "multi_ylim": 0.92, "multi_ratio_y": 0.84,
            "sharey": True, "label_pad_frac": 0.017 / 0.92,
            "ratio_fmt": "{:.2f}x",
            "suptitle": "VGGT-Omega patch tokens split by frame vs "
                        "global-attention stream",
            "footer": "The global stream is the second half of Omega's "
                      "concatenated token.",
        },
    },
}

RATIO_NOTE = "Numbers above groups are camera/object-layout distance ratios. "


def plot_panel(ax, data, metrics, title, ylim, ratio_y, cfg, label_values=True):
    x = np.arange(len(metrics))
    wd = 0.24
    for j, group in enumerate(GROUPS):
        vals = [data["groups_by_metric"][m][group]["mean"] for m, _ in metrics]
        bars = ax.bar(x + (j - 1) * wd, vals, wd * 0.92, color=COLS[group],
                      label=LEGEND[group], zorder=3)
        if label_values:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + ylim * cfg["label_pad_frac"], f"{val:.3f}",
                        ha="center", fontsize=8.3, color=INK)

    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("mean cosine distance", color=INK2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylim(0, ylim)

    for i, (metric, _) in enumerate(metrics):
        ratio = data["groups_by_metric"][metric]["ratio_cam_over_place"]
        ax.text(x[i], ratio_y, cfg["ratio_fmt"].format(ratio), ha="center",
                va="bottom", fontsize=9.2, color=INK2)


def fig_pair(single, multi, metrics, cfg, titles, suptitle, footer, path,
             figsize):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=cfg["sharey"])
    plot_panel(axes[0], single, metrics, titles[0],
               cfg["single_ylim"], cfg["single_ratio_y"], cfg)
    plot_panel(axes[1], multi, metrics, titles[1],
               cfg["multi_ylim"], cfg["multi_ratio_y"], cfg)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncols=3, frameon=False, fontsize=10)
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.text(0.5, 0.035, RATIO_NOTE + footer, ha="center", color=INK2,
             fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.84), w_pad=2.4)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("saved", path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), default="vggt")
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--figures", default=os.path.join(HERE, "figures"))
    return p.parse_args()


def main():
    args = parse_args()
    model = MODELS[args.model]
    prefix = model["prefix"]
    os.makedirs(args.figures, exist_ok=True)

    def load(name):
        path = os.path.join(args.results, f"{prefix}_{name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} — run probe_vggt.py first")
        return json.load(open(path))

    single = load("single_image")
    multi = load("all18_multiview")

    summary, split = model["summary"], model["split"]

    # single-image panel on its own (the original forum upload)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    plot_panel(ax, single, SUMMARY_METRICS, summary["single_title"],
               summary["single_ylim"], summary["single_ratio_y"], summary)
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=9.5)
    fig.tight_layout()
    path = os.path.join(args.figures, f"{prefix}_single_image.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("saved", path)

    fig_pair(single, multi, SUMMARY_METRICS, summary,
             ("single-image encoding\n(monocular VLA analogue)",
              "all-18-view inference\n(reconstruction setting)"),
             summary["suptitle"], summary["footer"],
             os.path.join(args.figures, f"{prefix}_single_vs_multiview.png"),
             figsize=(14.6, 5.8))
    fig_pair(single, multi, SPLIT_METRICS, split,
             ("single-image encoding", "all-18-view inference"),
             split["suptitle"], split["footer"],
             os.path.join(args.figures,
                          f"{prefix}_frame_global_patch_split.png"),
             figsize=(14.2, 5.4))


if __name__ == "__main__":
    main()
