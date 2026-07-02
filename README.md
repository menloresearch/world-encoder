# World Autoencoder

An encoder that tokenises a robot's sensor feeds (video, state, audio) with LeJEPA on RH20T, so
downstream models (VLAs, world models) can reuse one shared encoder instead of a vision-only one.

## Roadmap — Project 1: World Tokenizer

- Stage 0 — done. Benchmark existing LeJEPA checkpoints.
- Stage 1 — done. LeJEPA finetune on cfg3 video only.
- Stage 2 — core verified. Perceiver encoder, cfg3 video + robot_state.
- Stage 3 — robot_state decoder on Stage-2 latents.
- Stage 5 — scale to video + state + audio, MJEPA training.
- Stage 6 — state + audio decoder on Stage-5 latents.
- Stage 7 — real Microfactory data, same recipe.

## Stages 0-1 — done

Pipeline verified end to end: cfg3 → 2.33M frames (799 scenes, 66 tasks) → 240 WebDataset shards →
DDP continue-LeJEPA from `OK-AI/lejepa-vitb16-pretrain-in1k` → eval. Runs entirely on the NAS.

Finetuning on cfg3 video is a no-op — no help, no harm. Keep the warm-start (`e0`).
- Hot LR (2e-4) collapsed RankMe 300→158. LR 2e-5 fixes it (RankMe ~285).
- The task-id probe drop (0.91→0.74) is a saturated metric — it tracks ImageNet appearance, not damage.
- Robot-relevant eval (contact + force, 5-seed scene-held-out) is flat, and force-R²≈0 for every checkpoint.
- force-R²≈0 is the point: a single frame doesn't contain force, so vision alone can't encode it.

Eval rule from here: pair RankMe (health) with an unsaturated robot-relevant probe (contact/force),
scene-held-out and multi-seed. One saturated metric will mislead you.

## Stage 2 — video + robot_state (core verified)

Question: does fusing video + robot_state in one encoder learn genuine cross-modal structure — a
latent better than either modality alone, and is the gain *cross-modal* (not just compression)?

**What we ran** (single timestep, cfg3 video + robot_state):
- Signal gate: frozen vision → state R² = **0.43** (tcp 0.73) — cross-modal signal is real. (`step1_gate`)
- State loader: joints→sin/cos, tcp→symlog, quat→6D, F/T→symlog, gripper→symlog = 28-dim. (`state`)
- Encoder: **Perceiver** — query `CrossAttention` over vision patch tokens + state token → bottleneck
  latent. (`mm_perceiver`; a minimal MLP version in `mm_jepa` first)
- Loss: **masked latent prediction across MODALITIES** (mask one, predict its EMA-target latent from
  the other; predict-don't-equate) + **per-modal SIGReg** + **joint SIGReg**. Not ×time (that's
  Stage 5); not action-conditioned (#4).
- Eval: cross-modal predictability + RankMe, scene-held-out, multi-seed, with a PCA-256 compression control.

**Outcome — fusion works, and the gain is genuinely cross-modal:**
- Minimal model: cross-modal vision latent beats raw vision **+0.12 R²** (all 5 seeds), no collapse.
- Perceiver: beats raw vision (**+0.27**) *and* beats PCA-256 compression (**+1.80**) on all 5 seeds;
  RankMe 132 (no collapse). Beating compression ⇒ the gain is cross-modal, not dimensionality reduction.
- Caveat: absolute R² came out negative — the probe overfit on the 6.3k-frame cut used to dodge NAS
  congestion. The *relative, all-seed-consistent* ordering is solid; clean positive absolutes need more frames.

**Left:** clean positive absolutes (more frames), Step-6 ablations (bottleneck size, joint-SIGReg
on/off), and **modality × time** (temporal masking → predict future force/contact) = the Stage-5 upgrade.

Design note: masked cross-modal prediction is required, not optional — a shared encoder without it
underperforms single-modality (MJEPA). SIGReg only prevents collapse.

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

`world_tokenizer/` — the pipeline and evals. Runs from the NAS: `source /mnt/nas/data/RH20T/env.sh`
(nothing writes to `/`). Run order and implementation notes in
[`world_tokenizer/README.md`](world_tokenizer/README.md).
