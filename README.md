# Kepler-Encoder-v0.1

A **robot-first multimodal embedding model**: it treats robot state as a modality alongside vision
and maps both into a single shared latent space, so a downstream model (VLA, world model) can reuse
one embedding that carries **force and proprioception**, not appearance alone.

Vision and robot state are two views of one body — the images a camera produces are determined by
joint configuration, gripper aperture, and contact, and force leaves essentially no trace in a single
RGB frame. Kepler-Encoder fuses a **frozen ViT**'s patch tokens with **proprioception** and
**end-effector (force/torque + TCP pose)** tokens through a learned-query cross-attention layer
(*MMPerceiver*), trained self-supervised by **masked cross-modal latent prediction** under the
LeJEPA/SIGReg objective. At evaluation only vision enters, and the vision-only latent carries
proprioception- and force-relevant structure that raw frozen-ViT features do not.

Trained and evaluated on the **RH20T** real-robot corpus (7 configs, 4 embodiments). This is a v0.1
technical report validating the **single-timestep** case; native-rate temporal fusion is the next step.

- 📄 **Paper:** [`paper/main.tex`](paper/main.tex) — full method, results, related work.
- 📊 **Results:** [`EXPERIMENTS.md`](EXPERIMENTS.md) — the single ground-truth log (transfer matrix, ablations, downstream).
- 🗂 **Data:** [`DATA.md`](DATA.md) — RH20T layout, per-config analysis, timing model.
- 🗺 **Roadmap:** [`PLAN.md`](PLAN.md).

## Repository layout

| Path | What |
|------|------|
| `world_tokenizer/` | the encoder, training (`train_chunks.py`), and evaluation / probe scripts |
| `preprocessing/` | RH20T → frames → tick-anchored chunk caches (patch features + state) |
| `metrics/` | representation-quality metrics ([`METRICS.md`](metrics/METRICS.md)) |
| `pixnerd_integration/` | latent-conditioned diffusion decoder (pixel / cross-modal decode) |
| `visualizer/` | latent / attention inspection UI |
| `paper/` | the technical report (LaTeX) |
| `results/`, `figures/` | committed experiment outputs and figures |
| `splits/` | the frozen group-held-out split (`holdout_v1.csv`) |

## Setup

Dependencies are managed with [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

One dependency is used from source (not on PyPI): **`rh20t_api`** — clone
[rh20t/rh20t_api](https://github.com/rh20t/rh20t_api) and put it on `PYTHONPATH`. All pipeline
scripts take explicit `--*-root` flags for the data location; see
[`world_tokenizer/README.md`](world_tokenizer/README.md) for the full data layout and per-step flags.

## Reproduce

The encoder is light (a frozen ViT + a ~2M-param Perceiver over precomputed features), so training is
well under a GPU-hour per run. The pipeline is driven by three scripts (set your data/output paths at
the top of each):

```bash
./run_precompute.sh    # build per-config chunk caches (patch features + state)
./run_matrix.sh        # train the 5x4 transfer matrix (4 specialists + ALL), 5 seeds
./run_ablations.sh     # cross-modal / bottleneck / joint-SIGReg ablations
```

Numbers and figures land in `results/` and `figures/`; the narrative is in
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Status & contributing

v0.1 — single-timestep, on RH20T. v0.2 work in progress: **multi-camera fusion** (per-frame) + a
**next-embedding predictor** on the frozen encoder (time lives in the predictor, not the encoder —
the original multi-frame-in-encoder design was tested and retired). The plan-vs-actual narrative is
in [`STORY.md`](STORY.md); roadmap in [`PLAN.md`](PLAN.md); live v0.2 doc = [`V0.2.md`](V0.2.md).
Issues and PRs are welcome — please keep new results reproducible and logged in `EXPERIMENTS.md`.
