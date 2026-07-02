# World Autoencoder

An encoder that tokenises a robot's sensor feeds (video, state, audio) with LeJEPA on RH20T, so
downstream models (VLAs, world models) can reuse one shared encoder instead of a vision-only one.

## Roadmap — Project 1: World Tokenizer

- Stage 0 — done. Benchmark existing LeJEPA checkpoints.
- Stage 1 — done. LeJEPA finetune on cfg3 video only.
- Stage 2 — next. Modified encoder, cfg3 video + robot_state.
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

## Stage 2 — video + robot_state (next)

Question: does adding robot_state recover the force/contact signal vision can't (R²≈0 → high)?

The eval must be leak-free. robot_state contains F/T, so predict either future force/contact (t+Δ),
or force from a latent built without force (vision + joints + gripper → force). Predicting current
force from an input that already holds it proves nothing.

Steps, one change at a time:
1. Data. Fix the `rh20t_api` aligned getters (`get_joint_angles_aligned`, `get_gripper` return None
   on some scenes; `get_ft_aligned` works). Build the `{frame, state}` loader. Preprocess per
   modality: symlog for unbounded (velocity, position, current, tactile), sin/cos for angles, 6D for
   quaternions.
2. Control, before building the encoder. `[frozen e0 vision emb ⊕ state] → MLP → future/inferred
   force`. Compare vision-only, state-only, fused. If fused beats neither, stop and rethink.
3. Encoder. Query-transformer (Perceiver) fusing ViT tokens + state tokens into world tokens.
   Vision backbone frozen at first. Start with two losses only — cross-modal masked latent
   prediction + per-modal SIGReg. Add joint SIGReg, then action-conditioned prediction, as ablations.
4. Eval. RankMe per-modality and fused; force/contact probes. Success = fused beats vision-alone.

MJEPA finding to respect: a shared encoder without cross-modal prediction underperforms a single
modality. The masked cross-modal prediction loss is required, not optional — SIGReg only prevents
collapse.

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
