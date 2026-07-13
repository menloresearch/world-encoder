"""Compile EVERY metric from the raw results.json into one complete table (results/RESULTS.md)
— the authoritative, version-controlled record of all Phase-1 evals (mean±std over seeds,
per encoder × robot). Reads the committed raw outputs in results/phase1/ and results/phase1_abl/.

    python -m world_tokenizer.compile_results
"""
import glob
import json
import os

import numpy as np

METRICS = ["rankme_zv", "zv_r2_motor", "zv_r2_ee", "raw_r2_motor", "raw_r2_ee",
           "pca256_r2_motor", "pca256_r2_ee"]
ROBOTS = ["flexiv", "ur5", "franka", "kuka"]


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    a = np.asarray(vals, float)
    return f"{a.mean():.3f} ±{a.std():.3f}"


def run_table(path):
    d = json.load(open(path))
    seeds, a = d["seeds"], d.get("args", {})
    tag = os.path.splitext(os.path.basename(path))[0]
    flags = ("".join([" vision_only" if a.get("vision_only") else "",
                      " no_joint_sigreg" if a.get("no_joint_sigreg") else ""]))
    hdr = (f"**`{tag}`** — train_cfgs={a.get('train_cfgs')} · d={a.get('d')} · "
           f"queries={a.get('queries')} · seeds={len(seeds)} · ep={a.get('epochs')}{flags}")
    lines = [hdr, "", "| robot | n_test | " + " | ".join(METRICS) + " |",
             "|" + "---|" * (len(METRICS) + 2)]
    for r in ROBOTS:
        present = [s for s in seeds if r in seeds[s]]
        if not present:
            continue
        n = seeds[present[0]][r].get("n_test", "—")
        row = [ms([seeds[s][r].get(m) for s in present]) for m in METRICS]
        lines.append(f"| {r} | {n} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = ["# Full metric tables — auto-generated (do not hand-edit)",
           "",
           "Every Phase-1 eval, **mean ± std over seeds**, per encoder × robot. Regenerate with",
           "`python -m world_tokenizer.compile_results`. Raw per-seed outputs are the committed",
           "`results/phase1/*.json` (matrix) and `results/phase1_abl/*.json` (ablations).",
           "",
           "Metrics: `rankme_zv` (effective rank, collapse check) · `*_r2_motor` / `*_r2_ee`",
           "(linear-probe R² predicting joint / force-EE state from the vision-only latent) for",
           "`zv` (fused), `raw` (frozen ViT), `pca256` (compression control). Group-held-out split.",
           "",
           "## Matrix runs (fused encoders)"]
    for p in sorted(glob.glob(f"{root}/results/phase1/*.json")):
        out += ["", run_table(p)]
    out += ["", "## Ablation runs (vision-only control · bottleneck · SIGReg)"]
    for p in sorted(glob.glob(f"{root}/results/phase1_abl/*.json")):
        out += ["", run_table(p)]
    open(f"{root}/results/RESULTS.md", "w").write("\n".join(out) + "\n")
    print(f"wrote {root}/results/RESULTS.md  ("
          f"{len(glob.glob(f'{root}/results/phase1/*.json'))} matrix + "
          f"{len(glob.glob(f'{root}/results/phase1_abl/*.json'))} ablation runs)")


if __name__ == "__main__":
    main()
