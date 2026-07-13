"""Blog figures from the matrix results.json (no training / no GPU):
  1. transfer-matrix heatmap (encoder x robot, vision-only z_v joint R2)
  2. fusion-vs-baselines bars (ALL encoder: z_v vs raw vs PCA per robot; joint + force)

    python -m world_tokenizer.make_figures --results-dir /mnt/nas/data/RH20T/checkpoints/phase1
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENCODERS = ["flexiv", "ur5", "franka", "kuka", "all"]
ROBOTS = ["flexiv", "ur5", "franka", "kuka"]


def load(results_dir):
    R = {}
    for enc in ENCODERS:
        p = os.path.join(results_dir, enc, "results.json")
        if os.path.exists(p):
            R[enc] = json.load(open(p))["seeds"]
    return R


def mean(seeds, robot, key):
    v = [seeds[s][robot][key] for s in seeds
         if robot in seeds[s] and key in seeds[s][robot]]
    return float(np.mean(v)) if v else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/mnt/nas/data/RH20T/checkpoints/phase1")
    ap.add_argument("--out", default="figures/all")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    R = load(args.results_dir)
    print("loaded encoders:", list(R))

    # ---- 1. transfer-matrix heatmap (z_v joint R2) ----
    M = np.full((len(ENCODERS), len(ROBOTS)), np.nan)
    for i, e in enumerate(ENCODERS):
        for j, r in enumerate(ROBOTS):
            if e in R:
                M[i, j] = mean(R[e], r, "zv_r2_motor")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(ROBOTS))); ax.set_xticklabels(ROBOTS)
    ax.set_yticks(range(len(ENCODERS)))
    ax.set_yticklabels([e.upper() if e == "all" else e for e in ENCODERS])
    ax.set_xlabel("probed on robot"); ax.set_ylabel("encoder trained on")
    ax.set_title("Transfer matrix — vision-only $z_v$ → joint R²")
    for i in range(len(ENCODERS)):
        for j in range(len(ROBOTS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmean(M) else "black", fontsize=9)
    plt.colorbar(im, label="R²")
    fig.savefig(f"{args.out}/transfer_matrix_motor.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- 2. fusion-vs-baselines bars (ALL encoder, per robot) ----
    if "all" in R:
        for metric, label in [("motor", "joint"), ("ee", "force / EE")]:
            zv = [mean(R["all"], r, f"zv_r2_{metric}") for r in ROBOTS]
            raw = [mean(R["all"], r, f"raw_r2_{metric}") for r in ROBOTS]
            pca = [mean(R["all"], r, f"pca256_r2_{metric}") for r in ROBOTS]
            x = np.arange(len(ROBOTS)); w = 0.26
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(x - w, zv, w, label="fused $z_v$ (ours)", color="#2a9d8f")
            ax.bar(x, raw, w, label="raw ViT", color="#e9c46a")
            ax.bar(x + w, pca, w, label="PCA-256", color="#e76f51")
            ax.set_xticks(x); ax.set_xticklabels(ROBOTS)
            ax.set_ylabel("R²"); ax.axhline(0, color="k", lw=0.6)
            ax.set_title(f"ALL encoder — {label} probe: fusion vs baselines")
            ax.legend()
            fig.savefig(f"{args.out}/baselines_{metric}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
    print(f"saved: transfer_matrix_motor.png, baselines_motor.png, baselines_ee.png -> {args.out}")

    # ---- 3. cross-modal gain: fused (ALL) vs vision-only-trained (all_vo) ----
    abl = os.path.join(os.path.dirname(args.results_dir), "phase1_abl")
    vo_p = os.path.join(abl, "all_vo", "results.json")
    if "all" in R and os.path.exists(vo_p):
        VO = json.load(open(vo_p))["seeds"]
        robots_m = ["flexiv", "ur5", "franka", "kuka"]
        robots_e = ["flexiv", "ur5", "kuka"]  # franka has no F/T
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, metric, robs, ttl in [(axes[0], "motor", robots_m, "joint"),
                                       (axes[1], "ee", robots_e, "force / EE")]:
            fused = [mean(R["all"], r, f"zv_r2_{metric}") for r in robs]
            vo = [mean(VO, r, f"zv_r2_{metric}") for r in robs]
            x = np.arange(len(robs)); w = 0.36
            ax.bar(x - w / 2, fused, w, label="fused (state+vision)", color="#2a9d8f")
            ax.bar(x + w / 2, vo, w, label="vision-only-trained", color="#adb5bd")
            ax.set_xticks(x); ax.set_xticklabels(robs); ax.axhline(0, color="k", lw=0.6)
            ax.set_ylabel("R²"); ax.set_title(f"{ttl} probe"); ax.legend()
        fig.suptitle("Cross-modal gain — fusion vs vision-only-trained control (same model)")
        fig.savefig(f"{args.out}/cross_modal_gain.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: cross_modal_gain.png -> {args.out}")


if __name__ == "__main__":
    main()
