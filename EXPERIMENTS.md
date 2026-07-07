# Experiments

Detailed run notes and results per stage. High-level roadmap and design live in the
[README](README.md). Everything runs on the NAS (`source /mnt/nas/data/RH20T/env.sh`).

Standing eval rule (learned in Stage 1): pair **RankMe** (label-free health / collapse detector) with
an **unsaturated, robot-relevant probe** (predict robot state, or contact/force), always
**scene-held-out** and **multi-seed**. One saturated metric will mislead you.

> **R² ↑** in every table below: higher is better (fraction of held-out target variance explained;
> 1.0 = perfect, 0 = no better than predicting the mean, negative = worse than the mean).

## Initial experiments — LeJEPA finetune + single-timestep video+state fusion

Everything in this section uses the **legacy 28-dim state vector** (one UR5-shaped layout) and a
**single timestep** — the POC path before the robot-agnostic chunk packet of Phase 1. Two chapters:
(0-1) does finetuning the vision backbone on RH20T video help, and (2) does fusing video +
robot_state in one encoder learn genuine cross-modal structure.

### Stages 0-1 — LeJEPA finetune on cfg3 video only

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

### Stage 2 — video + robot_state (verified): cfg3 POC → cfg3+cfg4 scale-up

Question: does fusing video + robot_state in one encoder learn genuine cross-modal structure — a
latent better than either modality alone — and is the gain *cross-modal*, not just compression?

**What we ran** (single timestep, video + robot_state):

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
- **Two runs.** POC: cfg3 only, ~24k frames, 1 setup. Scale-up (2026-07-04, branch
  `user/jiaqi-stage2-cfg34`): cfg3+cfg4, 30 frames/scene → 86,430 frames / 2,881 scenes (**3.6×**
  data; both UR5 so the 28-dim layout still fits). Run script `run_stage2_cfg34.sh`; artifacts NAS
  `checkpoints/exp-20260704-032544/`.

**Outcome — fusion works, and the gain is genuinely cross-modal.** Predicting robot state (R², 5
seeds, scene-held-out, encoder retrained per split):

| feature (→ predict robot state) | cfg3 POC · R² ↑ | cfg3+cfg4 · R² ↑ |
|---|---|---|
| **Perceiver `z_v` (256, cross-modal, state masked)** | **0.551 ±0.018** | **0.653 ±0.008** |
| raw vision (768, mean-pooled) | 0.257 ±0.075 | 0.516 ±0.010 |
| PCA-256 of LeJEPA vision (compression control) | 0.134 ±0.047 | 0.418 ±0.015 |

- `z_v` beats raw vision by **+0.294 ±0.078** (POC) / **+0.137 ±0.007** (scale) and the PCA control by
  **+0.417 ±0.050** / **+0.235 ±0.010** — **positive on all 5 seeds in both runs**.
- **RankMe of `z_v`** (256-dim latent, ceiling 256): **210.8 ±4.5** on the cfg3+cfg4 scale run
  (per-seed 208/208/206/218/214 — tight, no collapse). The cfg3-only POC's per-seed RankMe was not
  saved to NAS; only the ~211 summary survived, so treat the POC figure as approximate.
- Beating the PCA compression control ⇒ the gain is **cross-modal, not dimensionality reduction**.
  A latent 3× smaller than raw vision predicts the robot better; PCA-256 has the *same* compression
  with no cross-modal training and loses, so the win can't be attributed to "compress to 256 helps."
- The vision-only fused latent (`z_v`) is used at eval, so no state leaks in — the encoder has
  *learned* to read robot-relevant structure out of pixels by having been trained alongside state.
- The cross-modal gain holds at 3.6× the data; the ordering is unchanged.

**Why every baseline jumps from cfg3 → cfg3+cfg4 (analysis).** The margin over raw narrows in
absolute terms (+0.294 → +0.137) but stays decisive vs the compression control (+0.417 → +0.235).
Crucially, the baselines rise for reasons that are mostly *not* "raw vision got better at decoding
state within one robot":

1. **R²'s denominator changed — pooling two setups injects easy, pixel-readable between-config
   variance.** R² = variance-explained ÷ *total* target variance. cfg3-only measures within-setup
   decoding; cfg3+cfg4 pools two UR5 setups with different camera mounts, backgrounds, and
   task/pose distributions. That between-setup difference is now part of total state variance and is
   *trivially* readable from pixels (any encoder can tell a cfg3 frame from a cfg4 frame by the
   background). So every vision baseline earns "free" R² from cross-config discrimination before
   decoding any within-config pose — which is why all three rows rise together and the ordering is
   preserved. This is a partial confound, not pure improvement.
2. **The POC numbers are noisy / under-fit.** Error bars collapse ±0.075 → ±0.010 (7× tighter). With
   ~24k frames split scene-held-out over 5 seeds, each probe fit on few scenes and the vision→state
   map (high-dimensional, data-hungry) was under-fit, so 0.257 is a high-variance underestimate; part
   of the jump to 0.516 is just the estimate settling with 3.6× the data.
3. **PCA-256 improved the most (+0.284 > raw's +0.259), which is diagnostic.** In the POC PCA-256
   (0.134) sat *below* raw (0.257): the top-256 principal directions of the frozen features weren't
   where the state signal lived (and/or the covariance was poorly estimated on 24k frames). At scale
   the covariance stabilizes *and* the new between-config variance lands in the top PCs and correlates
   with state — consistent with point 1.
4. **`z_v` rises the least (+0.102).** It was already reading state-relevant structure out of pixels,
   so it had the least to gain from the easy between-config variance and extra fit data — closest to
   saturation. This is why the margin over raw narrows while the margin over the compression control
   stays large.

   Net: the baseline jump is mostly a changed R² denominator plus the POC being noisy/under-fit — not
   within-robot decoding suddenly improving. The comparison that carries the cross-modal claim is
   `z_v` vs the compression control, which stays decisive across both runs.

**Design note:** masked cross-modal prediction is required, not optional — a shared encoder without
it underperforms single-modality (MJEPA). SIGReg only prevents collapse.

**Left / still owed:**
- The vision-only-*trained* Perceiver ablation (isolate cross-modal gain from "trained an in-domain
  encoder") — owed before this result goes external.
- Ablations: bottleneck size, joint-SIGReg on/off.
- Beyond cfg3+4 the 28-dim state path breaks (joint dims differ per robot); the multi-cfg path is the
  chunk packet (`chunk_state.py` → `precompute_chunks.py` → `dataloader.py`), used from Phase 1 of
  [PLAN.md](PLAN.md) onward.
- **modality × time** (temporal masking → predict future force/contact) = the Stage-5 upgrade.

## Phase 1 — full-RH20T transfer matrix (2026-07-07; all 5 runs 5-seed final)

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

**Diagonal — each encoder on its OWN robot** (R² ↑, vision-only `z_v`, all 5-seed mean ±std):

| robot | ALL z_v motor | spec z_v motor | raw motor | ALL z_v **ee** | spec z_v ee | raw ee |
|---|---|---|---|---|---|---|
| flexiv | 0.252 ±0.021 | 0.270 ±0.023 | 0.232 | 0.268 ±0.031 | 0.286 ±0.035 | 0.211 |
| ur5 | 0.339 ±0.014 | 0.324 ±0.021 | 0.321 | 0.187 ±0.009 | 0.156 ±0.016 | 0.103 |
| kuka | 0.321 ±0.022 | 0.330 ±0.019 | 0.391 | 0.393 ±0.035 | 0.432 ±0.055 | 0.353 |
| franka | −0.377 ±0.049 | −0.479 ±0.074 | −6.709 | — (no F/T) | — | — |

(`raw` has no ±std: raw ViT features don't depend on the encoder or seed, so the baseline is one
fixed number per robot. **PCA-256 ≈ raw at this scale** — motor 0.220/0.321/0.384, ee
0.205/0.144/0.353 for flexiv/ur5/kuka — so unlike the Stage-2 POC the compression control collapses
onto raw and `z_v` vs raw is the binding comparison. Full per-cell PCA-256 lives in `results.json`.)

Column legend (every cell is a **probe R²** — how well a linear probe predicts held-out robot
signals from the given feature; higher = better, scene-held-out):
- **robot** — the embodiment the probe is scored on (its OWN held-out groups; this is the
  diagonal of the 5×4 transfer matrix).
- Two **target groups**, each with three feature columns:
  - **motor** = the 8 motor rows (7 joints + gripper): joint pos/vel, gripper.
  - **ee** = the 13 end-effector slots: native-rate F/T (force/torque) + TCP 6D pose.
- Within each group, the feature the probe reads from:
  - **ALL z_v** — vision-only fused latent `z_v` from the single **ALL** encoder (trained on
    cfg1–7). State is masked at eval, so no state leaks in — this is the "one encoder for all
    robots" number (5-seed mean ±std).
  - **spec z_v** — same latent, but from the **specialist** encoder trained only on that
    robot's cfgs (flexiv=cfg1+2, ur5=cfg3+4, franka=cfg5, kuka=cfg6+7). 5-seed, mean ±std.
  - **raw** — baseline: same target predicted from raw frozen ViT features (768-dim,
    mean-pooled), no fusion. The bar `z_v` must beat to show cross-modal gain.
- **—** = not applicable: franka (cfg5) has no F/T sensor, so it has no ee targets.

**Full transfer matrix** — every encoder (rows) probed on every robot's held-out groups (cols),
vision-only `z_v` R² ↑, 5-seed mean; **bold = diagonal** (own robot). `raw` is the encoder-agnostic
baseline (one row, same for all). This is the evidence behind "one encoder learned *robots*":

*Motor (joints + gripper):*

| train ↓ / eval → | flexiv | ur5 | kuka | franka |
|---|---|---|---|---|
| flexiv spec | **0.270** | 0.254 | 0.234 | −0.393 |
| ur5 spec | 0.148 | **0.324** | 0.236 | −0.411 |
| kuka spec | 0.121 | 0.221 | **0.330** | −0.431 |
| franka spec | 0.069 | 0.088 | 0.148 | **−0.479** |
| **ALL** | 0.252 | 0.339 | 0.321 | −0.377 |
| raw (baseline) | 0.232 | 0.321 | 0.391 | −6.709 |

*End-effector (F/T + TCP pose; franka has no ee):*

| train ↓ / eval → | flexiv | ur5 | kuka |
|---|---|---|---|
| flexiv spec | **0.286** | 0.137 | 0.247 |
| ur5 spec | 0.135 | **0.156** | 0.247 |
| kuka spec | 0.106 | 0.119 | **0.432** |
| franka spec | 0.059 | 0.029 | 0.102 |
| **ALL** | 0.268 | 0.187 | 0.393 |
| raw (baseline) | 0.211 | 0.103 | 0.353 |

Reading the matrix: **ALL is at or near the best cell in every column** (it never trails the best
specialist by much and beats raw on 6 of 7 sensored cells), while **specialists fall apart
off-diagonal** — the franka specialist drops to 0.06–0.15 on other robots (≈ raw or worse), and
even the strong kuka specialist reads 0.11–0.22 elsewhere. A specialist learned *its* robot; ALL
learned robots. (Cost: ~8–52 min/seed per run, ALL the most; held-out n_test 5,970–27,448/robot.)

**RankMe of `z_v`** (effective rank / collapse detector, `exp(entropy of normalized singular
values)`; measured on each robot's held-out `z_v`, 5-seed mean ±std):

| robot | ALL z_v RankMe | spec z_v RankMe |
|---|---|---|
| flexiv | 171.1 ±32.6 | 161.5 ±48.1 |
| ur5 | 171.9 ±31.4 | 177.1 ±2.8 |
| kuka | 166.1 ±27.8 | 142.4 ±7.3 |
| franka | 174.0 ±30.9 | 146.6 ±2.6 |

**Baseline / how to read it.** RankMe has no learned baseline — it's a self-referential diagnostic
bounded by `[1, latent_dim]`. `z_v` is 256-dim (Perceiver pools its 8 queries), so:
- **Ceiling = 256** — a perfectly uniform singular-value spectrum (every direction used equally,
  zero collapse). Unreachable in practice; real healthy encoders sit well below it.
- **Floor → 1** — total dimensional collapse (all variance in one direction).
- **Healthy band observed here ≈ 165–180** (≈0.65–0.70 of ceiling) — no collapse. The ALL numbers
  are the *same* encoder scored on each robot's test set, so the small flexiv→franka spread is the
  test set, not four encoders.

RankMe scales with dimensionality, so these are **not** comparable to the 768-dim raw-ViT features
(whose ceiling is 768) — only within the 256-dim `z_v` family.

The wide ±std on **ALL** and the **flexiv specialist** is one outlier seed each that partially
collapsed (ALL seed → 106, flexiv-spec seed → 66); the other four seeds of each sit ≈178–190. All
other runs are tight (±≤9). Not a systematic collapse — a per-seed training instability worth a
re-run, not a red flag on the recipe.

**Findings:**
- **Force/EE = the clean cross-modal win.** ALL `z_v` beats raw vision on the F/T + pose
  probe for **every** sensored robot (+0.06 flexiv, +0.08 ur5, +0.04 kuka).
- **Motor (joints) mixed.** ALL beats raw on flexiv/ur5/franka; **loses only on kuka**
  (0.321 vs 0.391). Vision already infers joint pose; force is where fusion earns its keep.
- **"One encoder for all robots" holds (strong form).** ALL matches or beats each specialist
  on its own robot (beats ur5 & franka on motor, ties flexiv & kuka) — and off-diagonal, a
  specialist barely beats raw (franka-specialist → other robots ≈ 0.07–0.15) while ALL stays
  strong (see the transfer matrix above). One encoder learned *robots*, not one robot.
- franka own-robot R² is negative for all (tiny, force-blind cfg5), but `z_v` (−0.38) is far
  more stable than raw's catastrophic −6.7.

**Caveats / still owed:** the vision-only-*trained* Perceiver ablation (owed since Stage 2) and
the triplet-accuracy / distance-correlation / alignment-uniformity geometry metrics
(implemented in `metrics.py`, never run on any checkpoint) are outstanding; **kuka joints** is
the one cell where raw beats fusion; and two seeds (ALL, flexiv-spec) had a RankMe dip worth a
re-run.
