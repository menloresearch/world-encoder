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

## Stage 2 at scale — cfg3+cfg4 (2026-07-04, branch `user/jiaqi-stage2-cfg34`)

First scale-up beyond the cfg3 POC: same recipe (frozen `e0` + Perceiver, single timestep,
legacy 28-dim state — cfg3/4 are both UR5 so the layout still fits), 30 frames/scene →
86,430 frames / 2,881 scenes, 5 seeds, scene-held-out, encoder retrained per split.
Run script: `run_stage2_cfg34.sh`; artifacts: NAS `checkpoints/exp-20260704-032544/`.

| feature (→ predict robot state) | R² |
|---|---|
| **Perceiver `z_v` (256, cross-modal, state masked)** | **0.653 ±0.008** |
| raw vision (768, mean-pooled) | 0.516 ±0.010 |
| PCA-256 of LeJEPA vision (compression control) | 0.418 ±0.015 |

- `z_v` − raw = **+0.137 ±0.007**, `z_v` − PCA-256 = **+0.235 ±0.010** — positive on all
  5 seeds; RankMe 211 (no collapse).
- The cross-modal gain holds at 3.6× the POC's data. Every baseline improves with more
  data (raw 0.257 → 0.516) while the ordering is unchanged — the margin over raw narrows
  in absolute terms but stays decisive vs the compression control.
- Beyond cfg3+4 the 28-dim state path breaks (joint dims differ per robot); the multi-cfg
  path is the chunk packet (`chunk_state.py` → `precompute_chunks.py` → `dataloader.py`),
  used from Phase 1 of [PLAN.md](PLAN.md) onward.
- **Still owed before this result goes external:** the vision-only-*trained* Perceiver
  ablation (isolate cross-modal gain from "trained an in-domain encoder").

## Phase 1 — full-RH20T transfer matrix (2026-07-07, PRELIMINARY — ALL run 4/5 seeds)

Scale-up to all 7 cfgs / 4 embodiments via the robot-agnostic chunk packet (supersedes the
28-dim state). Question: does **one encoder trained on all robots match per-robot
specialists**, and does cross-modal fusion still beat raw vision at full scale?

**What we ran** (single timestep, frozen `e0` + 3-modality Perceiver over the chunk packet):

- Packet: 196 vision patch tokens + 8 motor rows (7 joints + gripper; sin/cos q, symlog dq)
  + 13 ee slots (native-rate F/T + TCP 6D), all masked; **one fixed external camera** per
  scene (wrist excluded). (`chunk_state` → `precompute_chunks` → `dataloader`)
- Encoder: `MMPerceiverChunks` (d=256, 8 queries, ~2M trainable); masked cross-modal latent
  prediction (hide one of vision/motor/ee, predict its EMA-target) + per-modal SIGReg + joint
  SIGReg. (`mm_perceiver2`, `train_chunks`)
- 5 runs × 5 seeds × 40 ep: 4 specialists (flexiv = cfg1+2, ur5 = cfg3+4, franka = cfg5,
  kuka = cfg6+7) + 1 **ALL** (cfg1–7). Each probed on every embodiment's held-out groups →
  5×4 transfer matrix. Baselines fit on train rows only: raw ViT (768), PCA-256. Eval latent
  = **vision-only `z_v`**. Caches on NAS `caches/cfg{1..7}.npz` (~53 GB); `run_matrix.sh`.

**Diagonal — each encoder on its OWN robot** (R², vision-only `z_v`; specialists 5-seed
final, ALL 4-seed preliminary):

| robot | ALL z_v motor | spec z_v motor | raw motor | ALL z_v **ee** | spec z_v ee | raw ee |
|---|---|---|---|---|---|---|
| flexiv | 0.256 | 0.270 ±0.023 | 0.232 | 0.271 | 0.286 ±0.035 | 0.211 |
| ur5 | 0.340 | 0.324 ±0.021 | 0.321 | 0.186 | 0.156 ±0.016 | 0.103 |
| kuka | 0.328 | 0.330 ±0.019 | 0.391 | 0.401 | 0.432 ±0.055 | 0.353 |
| franka | −0.372 | −0.479 ±0.074 | −6.709 | — (no F/T) | — | — |

RankMe (`z_v`) 142–193 across the board — no collapse (flexiv specialist ±48: one seed dipped).

**Findings (preliminary):**
- **Force/EE = the clean cross-modal win.** ALL `z_v` beats raw vision on the F/T + pose
  probe for **every** sensored robot (+0.06 flexiv, +0.08 ur5, +0.05 kuka).
- **Motor (joints) mixed.** ALL beats raw on flexiv/ur5/franka; **loses only on kuka**
  (0.328 vs 0.391). Vision already infers joint pose; force is where fusion earns its keep.
- **"One encoder for all robots" holds (strong form).** ALL matches or beats each specialist
  on its own robot (beats ur5 & franka, ties flexiv & kuka) — and off-diagonal, a specialist
  barely beats raw (franka-specialist → other robots ≈ 0.07–0.15) while ALL stays strong. One
  encoder learned *robots*, not one robot.
- franka own-robot R² is negative for all (tiny, force-blind cfg5), but `z_v` (−0.37) is far
  more stable than raw's catastrophic −6.7.

**Caveats / still owed:** ALL is 4/5 seeds (final mean±std pending completion); the
vision-only-*trained* ablation (owed since Stage 2) and triplet-accuracy eval are not yet
run; **kuka joints** is the one cell where raw beats fusion. Full 5×4 mean±std table to be
finalized when the ALL run completes.
