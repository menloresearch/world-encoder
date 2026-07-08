# World Autoencoder

An encoder that tokenises a robot's sensor feeds (video, state, audio) with LeJEPA on RH20T, so
downstream models (VLAs, world models) can reuse one shared encoder instead of a vision-only one.

## Roadmap — Project 1: World Tokenizer

- Stage 0 — done. Benchmark existing LeJEPA checkpoints.
- Stage 1 — done. LeJEPA finetune on cfg3 video only (a no-op; keep the warm-start).
- Stage 2 — verified. Perceiver encoder on cfg3 video + robot_state; the cross-modal latent predicts
  the robot ~2× better than vision and beats a compression control on all seeds.
- Stage 3 — robot_state decoder on Stage-2 latents.
- Stage 5 — scale to video + state + audio, MJEPA training (modality × time).
- Stage 6 — state + audio decoder on Stage-5 latents.
- Stage 7 — real Microfactory data, same recipe.

Per-stage run notes and results: [`EXPERIMENTS.md`](EXPERIMENTS.md).
Setup (deps, `rh20t_api`, data paths) and run order: [`world_tokenizer/README.md`](world_tokenizer/README.md).

## Architecture

- Base: `galilai-group/lejepa`.
- Dataset: RH20T cfg3 (smallest subset; scale to others later).
- Encoder: query-transformer (Perceiver) for cross-modal compression, trained with LeJEPA.
- Decoder: PixNeRD → latent diffusion (Stage 3+).
- Latent: continuous and discrete.
- Preprocess per modality: symlog (unbounded), sin/cos (angles), 6D / canonicalize (quaternions).

Losses:

| # | loss | role |
|---|------|------|
| 1 | masked latent prediction over (modality × time) tokens | the learning signal; predict-don't-equate respects info asymmetry, avoids intersection collapse, robust to missing modalities |
| 2 | per-modality SIGReg | anti-collapse + magnitude standardiser, so modalities are commensurate before fusion |
| 3 | joint SIGReg on the fused latent | keep the fused latent high-rank |
| 4 | action-conditioned forward prediction (only if actions matter) | the causal engine; same-time alignment is only correlational |

Frameworks: [stable-pretraining](https://github.com/galilai-group/stable-pretraining) (LeJEPA +
other SSL), [le-wm](https://github.com/lucas-maes/le-wm) (full training example).

## Code

`preprocessing/` — raw RH20T → frames → WebDataset shards (`preprocess_all.sh`), plus the
data-alignment gate and per-cfg analysis.
`world_tokenizer/` — the model, training, and evals.
`metrics/` — encoder-only representation metrics (design notes in [`metrics/METRICS.md`](metrics/METRICS.md)).
Runs from the NAS: `source /mnt/nas/data/RH20T/env.sh` (nothing writes to `/`).
Run order and implementation notes in [`world_tokenizer/README.md`](world_tokenizer/README.md).
