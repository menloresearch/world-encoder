# Experiments

Detailed run notes and results per stage. High-level roadmap and design live in the
[README](README.md). Everything runs on the NAS (`source /mnt/nas/data/RH20T/env.sh`).

Standing eval rule (learned in Stage 1): pair **RankMe** (label-free health / collapse detector) with
an **unsaturated, robot-relevant probe** (predict robot state, or contact/force), always
**scene-held-out** and **multi-seed**. One saturated metric will mislead you.

## Stages 0-1 — LeJEPA finetune on cfg3 video only

Pipeline verified end to end: cfg3 → 2.33M frames (799 scenes, 66 tasks) → 240 WebDataset shards →
DDP continue-LeJEPA from `OK-AI/lejepa-vitb16-pretrain-in1k` → eval.

**Result: finetuning on cfg3 video is a no-op — no help, no harm. Keep the warm-start (`e0`).**

- Hot LR (2e-4) collapsed RankMe 300→158. LR 2e-5 fixes it (RankMe ~285).
- The task-id probe drop (0.91→0.74) is a **saturated** metric — it tracks ImageNet appearance, not
  a real regression. (`probe_curve`)
- Robot-relevant eval (contact + force, 5-seed scene-held-out) is flat, and force-R²≈0 for every
  checkpoint. (`robust_robot_eval`, `contact_probe`)
- force-R²≈0 is the point: a single frame doesn't contain force, so vision alone can't encode it —
  which is exactly why Stage 2 adds robot_state.

Scripts: `extract_frames` → `make_shards` → `train` → `eval_lejepa` / `probe_curve` /
`robust_robot_eval` / `contact_probe`; `gate` for the video↔force alignment sanity check.

## Stage 2 — video + robot_state (verified)

Question: does fusing video + robot_state in one encoder learn genuine cross-modal structure — a
latent better than either modality alone — and is the gain *cross-modal*, not just compression?

**What we ran** (single timestep, cfg3 video + robot_state):

- Signal gate: frozen vision → state R² = **0.43** (tcp 0.73) — cross-modal signal is real, so
  building the fusion encoder is justified. (`step1_gate`)
- State loader: joints→sin/cos, tcp→symlog, quat→6D, F/T→symlog, gripper→symlog = 28-dim. (`state`)
- Vision features precomputed once from the frozen `e0` backbone as patch tokens (196×768).
  (`precompute_patch`)
- Encoder: **Perceiver** — learnable queries `CrossAttention` over [vision patch tokens + state
  token] → bottleneck latent. Vision backbone frozen; only the ~2M-param fusion head trains. (`mm_perceiver`)
- Loss: **masked latent prediction across MODALITIES** (mask one, predict its EMA-target latent from
  the other; predict-don't-equate) + **per-modal SIGReg** + **joint SIGReg**. Not ×time (that's
  Stage 5); not action-conditioned (#4).
- Eval: cross-modal predictability + RankMe, scene-held-out, 5 seeds, encoder retrained per split,
  with a PCA-256 compression control. (`train_perceiver`)

**Outcome — fusion works, and the gain is genuinely cross-modal.** Predicting robot state (R²,
5 seeds, scene-held-out, 24k frames, encoder retrained per split):

| feature (→ predict robot state) | R² |
|---|---|
| **Perceiver `z_v` (256, cross-modal, state masked)** | **0.551 ±0.018** |
| raw vision (768, mean-pooled) | 0.257 ±0.075 |
| PCA-256 of LeJEPA vision (compression control) | 0.134 ±0.047 |

- `z_v` beats raw vision by **+0.294 ±0.078** and PCA-256 by **+0.417 ±0.050** — **both on all 5 seeds**.
- Beating the PCA compression control ⇒ the gain is **cross-modal, not dimensionality reduction**.
  A latent 3× smaller than raw vision predicts the robot ~2× better. RankMe 211 (no collapse).
- The vision-only fused latent (`z_v`) is used at eval, so no state leaks in — the encoder has
  *learned* to read robot-relevant structure out of pixels by having been trained alongside state.

**Design note:** masked cross-modal prediction is required, not optional — a shared encoder without
it underperforms single-modality (MJEPA). SIGReg only prevents collapse.

**Left:** ablations (bottleneck size, joint-SIGReg on/off), and **modality × time** (temporal
masking → predict future force/contact) = the Stage-5 upgrade.
