# PLAN v2 — Full-RH20T Vis+State Matrix Run → Temporal → Decoder

Rewritten 2026-07-06 after the weekend's work landed on `user/jiaqi` (see DATA.md /
METRICS.md there). Supersedes the 2026-07-03 PLAN (kept in git history). Project context in
[HANDOFF.md](HANDOFF.md); results in [EXPERIMENTS.md](EXPERIMENTS.md).

## What changed since the last PLAN (2026-07-03 → 2026-07-06)

Done over the weekend (on `user/jiaqi`):

- **Preprocessing (was A/B):** all 7 cfgs extracted + sharded on the NAS (54.3M frame
  samples, ~4 TB). `DATA.md` is the per-cfg audit (joint layouts, `_human_2` trap,
  57 scenes missing joint.npy, cfg5 has no physical F/T, tick timing).
- **Robot-agnostic state (was A1's locked "16-dim"):** SUPERSEDED by the tick-anchored
  chunk packet (`chunk_state.py`): motor (8×3, masked — keeps joints via sin/cos q +
  symlog dq instead of dropping them) + ee (13×15 native-rate F/T+TCP, masked) +
  robot_id + ts. Strictly better than the 16-dim design; this doc locks the packet instead.
- **Split (was C1/C2):** frozen `splits/holdout_v1.csv`, held out by (cfg, task, user)
  group, stratified per cfg; `dataloader.py` enforces it and errors on unknown groups.
- **Metrics (was C3):** `metrics/metrics.py` — triplet accuracy, RankMe, probe R²,
  distance correlation, alignment/uniformity. Open: pair/triplet *selection* wiring.
- **Viz (was D):** `visualizer/` local website. (`visualize_stage2.py` was deleted on
  `user/jiaqi` — decide: resurrect for the PCA blog figures, or extend visualizer.)
- **Open-source (was E):** repo is PUBLIC; uv packaging (`pyproject.toml`/`uv.lock`/
  `requirements.txt`; stable-pretraining pinned as a package); env.sh retired
  ("do NOT source env.sh" — its wae-venv is dead on the A6000 VM). Still missing: LICENSE.
- **Bonus result:** Stage-2 recipe scaled to cfg3+cfg4, 5 seeds: z_v→state R²
  **0.653 ±0.008** vs raw 0.516 vs PCA-256 0.418, all seeds positive, RankMe 211.
  Details in EXPERIMENTS.md.

## What actually blocks the full-RH20T run

1. **No trainer consumes the chunk packet.** `mm_perceiver.py` still expects
   (patch, 28-dim state); nothing reads `dataloader.py`'s motor/ee/masks/robot_id.
2. **Chunk caches never computed** — `preprocessing/precompute_chunks.py` has not been
   run; there is no `caches/` on the NAS.
3. Metrics selection wiring (triplet/pair sampling from the test loader).
4. Ops on the A6000 VM: use the repo uv venv (NAS wae-venv is dead there);
   `gh auth login` needed before any push; env.sh's `CUDA_VISIBLE_DEVICES=1` default is
   stale (all 7 GPUs free).

## Code-review findings (2026-07-06 full read of `user/jiaqi`)

Full review of the branch (core pipeline line-by-line + agent sweep of the rest, mask/EMA
logic verified against installed stable_pretraining). **Verdict: solid to build on.** Items:

- `extract_frames.py` still has the `endswith("_human")` bug (documented in DATA.md, not
  fixed) → re-running stage-1 extraction re-leaks 543 `_human_2` scenes. Chunk pipeline
  filters correctly. Fix before any re-extraction; also its resume check treats any
  non-empty frame dir as done (partial extractions silently pass).
- Stage-2 PCA-256 control was fit on train+test rows (train_perceiver.py) — conservative-
  direction leakage (inflates the baseline we beat). Fit PCA on train only in Phase-1 eval.
- metrics.py: math correct, no leakage; add NaN guards for degenerate inputs (all-zero
  latents, single-group splits) when wiring eval.
- Trainer traps (confirmed): chunk masks are True=VALID but CrossAttention masks are
  True=BLOCKED (invert!); `rgb [B,1,196,768]`/`motor [B,1,8,3]` carry a singleton time axis
  their masks lack; `ee_mask` all-False for all of cfg5 + ~⅓ of cfg3 (~12% of data) —
  masked-mean over zero elements = NaN, skip fully-masked samples; `scene_idx`/`group_idx`
  are not stable across different `cfgs=` subsets (matters for matrix runs); dataset holds
  all caches in RAM (~55 GB at 15 chunks/scene) — memmap rework before raising density.
- Nothing on the branch consumes the chunk packet — the Phase-1 trainer is greenfield.

## Phase 1 — Full-RH20T vis+state, single-timestep MATRIX run (this week)

The safe scale-up of the validated Stage-2 recipe: frozen `e0` vision + Perceiver fusion,
single timestep, masking over MODALITY. Do NOT unfreeze vision (Stage-1 evidence).

- [ ] **1.1** Env on the A6000 VM: `uv sync`, dataloader smoke test.
- [ ] **1.2** MMPerceiver v2 + trainer for the chunk packet: three token groups (vision
      patches / motor / ee), masks honored (cfg5's all-False ee_mask doubles as a free
      missing-modality robustness test), masked latent prediction over modality +
      per-modal SIGReg + joint SIGReg. **Every token carries its `ts` from day one** —
      that is the Phase-2 temporal slot; no data or interface rework later.
- [ ] **1.3** Camera choice v1: ONE deterministic *external* camera per scene — exclude
      the wrist cam via the `in_hand` serials in rh20t_api `configs/configs.json`.
      (`chunk_state.py` currently takes `sorted(serials)[0]`, which is sometimes the
      wrist cam = the "multi-camera noise". Camera serials are rig-fixed per cfg —
      verified by sampling.) Multi-view *learning* is parked (see Parked).
- [ ] **1.4** Run `precompute_chunks` for all 7 cfgs in parallel across the 7 A6000s →
      NAS `caches/cfg{1..7}.npz` (~190k chunks at 15/scene ≈ ~56 GB patch fp16 — fits).
- [ ] **1.5** **THE MATRIX:** 4 per-embodiment encoders (flexiv=cfg1+2 6,060 scenes,
      ur5=cfg3+4 2,993, franka=cfg5 1,321, kuka=cfg6+7 2,402) + 1 joint all-7 encoder,
      multi-seed, frozen split. Evaluate every encoder on every embodiment's held-out
      groups → 5×4 transfer matrix. This settles "one encoder vs per-embodiment"
      empirically — the cross-embodiment de-risking and the one-encoder thesis in a single run —
      and is the headline blog figure.
- [ ] **1.6** Claims-protection ablations (cfg3-scale, cheap, BEFORE anything goes
      external): vision-only-*trained* Perceiver (isolates the cross-modal gain — still
      owed from Stage 2), bottleneck size, joint-SIGReg on/off.
- [ ] **1.7** Eval: RankMe + probe R² + triplet accuracy (wire the selection — closes the
      METRICS.md TODO; cross-view / cross-embodiment pairs give the negative tiers),
      per-config breakdown, PCA-256 control, multi-seed error bars.
- [ ] **1.8** Blog figures: PCA of the joint-encoder latent colored by robot / task / cfg.

**Gate:** joint encoder ≥ per-embodiment in-domain; positive cross-embodiment transfer;
no collapse; ablations survive.

## Phase 2 — Temporal (starts the DAY Phase-1 runs launch, not after)

The ×time half of loss #1; Stage-5 kickoff (video + state only; audio stays out). The data
side already exists: tick-anchored chunks, native-rate ee windows (100/125 Hz), irregular
ticks (6.7–14.7 Hz), `ts` cached per chunk. Remaining work is model-side only:

- [ ] **2.1** Continuous-time embedding (Time2Vec / mTAN-style Fourier features of the
      real timestamp) on every token.
- [ ] **2.2** Multi-tick context windows (Δt-based selection via cached `ts`).
- [ ] **2.3** Mask over (modality × time) → predict held-out-time / future ee latents.
- [ ] **2.4** Eval: future force/contact prediction at varying Δt vs the single-timestep
      Phase-1 baseline.

**Gate:** temporal masking beats single-timestep on future-state prediction with RankMe
stable. Loss #4 (action-conditioned forward prediction) only after this gate.

## Phase 3 — Decoder (parallel track)

- [ ] **3.1** robot_state decoder on frozen Phase-1 latents — "the generative decoder is
      our superpowered linear probe": quantifies latent content, cheap, days not weeks.
- [ ] **3.2** PixNeRD → latent diffusion pixel decoder (a viz/probe tool at this stage,
      not a training signal).

## External validation lead — FLARE / GR00T (from 2026-07-06 lit sweep)

NVIDIA GEAR's **FLARE** (arXiv:2505.15659, CoRL 2025, shipped in GR00T N1.5) trains a VLA
with an auxiliary JEPA-style loss: predict the *latent* of the observation 16 steps ahead,
produced by a target encoder g(·). Their ablation shows **g(·) quality is the deciding
factor** (none 43.9% → raw SigLIP-2 49.6% → their learned encoder 55.0%) — and their best
g(·) is vision-language only (no proprio/F-T, single rate) and needs an EMA moving-target
hack. That is a drop-in slot for world-encoder: multimodal, native-rate, SIGReg-frozen.
**Follow-up after the You Liang Tan meeting:** if receptive, define the interface (token
count/dim/rate) and run "our encoder as frozen g(·)" as a downstream benchmark for the
Phase-1 encoder. Prep notes: `~/brain/ishneet/youliangtan-papers.md` (kept out of repo).

## Parked — each with an explicit trigger

| item | trigger |
|---|---|
| Multi-view training objective | Phase-1 analysis: measure same-tick cross-view latent distance with the v1 encoder. If views don't already cluster → add cross-VIEW masked prediction (predict side-view latent from wrist-view + state). Predict-don't-equate — never latent *equality* across views (dual-arm / wrist-cam info-asymmetry objection). Include latent **sum/mean-pool of per-view latents as the fusion baseline** (the neural-codec idea — the additive trick itself doesn't transfer: audio mixes additively at the sensor and codecs are near-lossless, cameras are projections and JEPA latents are lossy by design — but it's the right dumb baseline vs Perceiver fusion with view-tagged tokens). |
| Loss #4 (action-conditioned) | Phase-2 gate passes (needs temporal machinery). |
| Audio (full Stage 5) | Temporal proven. |
| Discrete latent / disentanglement; dual-arm embedding-sum | Microfactory data (Stage 7) — RH20T is single-arm. |
| Loss balancing (Kendall / GradBlend) | Only if the 3-loss balance misbehaves at scale (it hasn't). |
| Theory reading | Rate-distortion → when sizing the bottleneck ablation; PID/synergy → when interpreting the transfer matrix; identifiability/causality → Phase 2+ with actions. |

## Decisions to confirm (blocking)

1. Camera choice v1 = fixed external cam, wrist excluded (proposed above).
2. File ownership: trainer (Ishneet) vs triplet-selection wiring — both touch his code.
3. cfg5 stays in with ee fully masked — yes/no.
4. Push his local work: 2 visualizer commits + 1 tqdm commit are unpushed on his clone.
5. LICENSE — the repo is already public without one (E5 from the old plan, still open).
6. Merge direction: build Phase 1 on top of `user/jiaqi` (it has the dataloader/split);
   `user/ishneet` is stale.
