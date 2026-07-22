# STORY — what we planned, what happened, how it changed

Living narrative doc, updated **2026-07-21**. One page to orient anyone (including future us / JQ):
the chronological plan-vs-actual record. Numbers and build details live in the linked docs — this doc
tells the story and points; it never becomes the source of truth for a number.

## TL;DR today

One encoder for all robots **works and is published** (v0.1 / Kepler paper). Putting **time inside
the encoder failed** its gate twice and is retired; v0.2 re-scoped to **per-frame multi-cam encoder +
time in a separate predictor** — both builds done and green. Downstream on RoboCasa, the
**encoder-as-replacement claim is dead** (a compact latent can't be a policy's only eyes:
0–12% vs baseline 24–32%), so we pivoted to the **FLARE-faithful hybrid** (latent *added to* the
policy's vision, not replacing it) — training now, first number ~today 14:00. The week's decision
number is **baseline-vs-hybrid**.

## 1. Phase 1 / v0.1 — the bet that worked (…–2026-07-07)

- **Plan:** one multimodal JEPA encoder (Perceiver fuse, SIGReg, cross-modal head) over vision +
  robot state, all RH20T robots at once; prove one-encoder-for-all + cross-modal force readout.
- **What happened:** gate PASSED 2026-07-07 (5-seed full-RH20T matrix). One-encoder holds; **force is
  the clean cross-modal win**; transfer matrix in [EXPERIMENTS.md](v0.1/EXPERIMENTS.md). Downstream
  usefulness (surprise AUROC 0.90, state decode, PixNerd pixel decode) shipped.
- **Outcome:** paper on `main` ([paper/](paper/)); reviewer answers + OSS cleanup merged (PR #10/#11).
- **Standing external leads born here:** ARM (edge reference model) and NVIDIA GEAR/FLARE
  (our encoder as their g(·)) — [PLAN.md](PLAN.md) §External.

## 2. Phase 2, original bet — temporal-in-encoder (2026-07-15 → 07-17): FAILED, RETIRED

- **Plan:** fuse a *window of ticks* into one latent — flat Perceiver over all modality×time tokens,
  continuous-time embeddings, no resampling (design doc `TEMPORAL_ARCH.md` — removed in the 07-21 doc
  consolidation, in git history: `git show 8432258:TEMPORAL_ARCH.md`).
- **What happened:** trained healthily but the NH1 gate failed **twice**, under two independent
  objectives (masked-cell, then the ported v0.1 head): temporal ≤ v0.1 on every discriminating cell,
  present-force probe **halved** (0.10 vs 0.21), RankMe 51 vs 134. Diagnostics exonerated pooling,
  masking, time-embed dominance — the temporal *fusion itself* dilutes the per-frame signal
  (a fixed-capacity latent spends its budget summarizing the window instead of nailing the present).
  Full saga: `TEMPORAL_JOURNAL.md` (git history); numbers:
  [results/temporal/RESULTS.md](v0.2/results/temporal/RESULTS.md) §1–3.
- **Why we're confident it's the architecture, not data:** v0.1 wins on the SAME coarse cache.
- **The gate did its job** — caught the regression in days, before any scaling.

## 3. The re-scope — v0.2 = per-frame encoder, time in the predictor (2026-07-17)

- **Decision:** stop making the encoder temporal. v0.2 = per-frame multimodal encoder
  (**Build 1: multi-cam**) + a separate small **next-embedding predictor** on the frozen encoder
  (**Build 2** — time, relocated). Not a retreat: FLARE, LeWM, RoboTTT and JQ's own framing all put
  time *outside* a per-frame g(·). "Fixing v0.2 properly" and "doing FLARE" converge on the same
  artifact. Caveat kept on record: this is right *for our g()→VLA goal*, not a universal law
  (V-JEPA 2 is spatiotemporal because it IS the rollout model). [V0.2.md](v0.2/V0.2.md) — the live doc.

## 4. v0.2 builds — both DONE + GREEN (2026-07-17 → 07-19)

- **Build 2 (predictor) first** (ran on existing caches): beats carry-forward **28–36% at every
  horizon**; pooled ≥ set (superseded a guardrail); action-conditioning gave **no lift** because RH20T
  only records *realized* motion (already in z) — N2 world-model parked until a real commanded-action
  signal (RoboCasa has one). [RESULTS.md](v0.2/results/temporal/RESULTS.md) §4–5.
- **Build 1 (multi-cam, K=4 kuka):** no rank collapse (RankMe 153 vs 136), **force held AND improved
  (0.283 vs 0.251)** — the probe temporal halved — pose 0.717 vs 0.635; same model fed 1 view drops to
  0.132 → views genuinely pay. Caveat: 1-view input is OOD (trained with all 4 present).
  [RESULTS.md](v0.2/results/temporal/RESULTS.md) §5c.

## 5. Embedding analyses for JQ (2026-07-21, kanban brain-internal#1)

- The fused embedding is **view-specific, not view-invariant** (same-instant compositions 1.46×
  further apart than cross-instant); **early fusion > single view > late fusion** (JQ's mean-of-v0.1
  baseline loses to a single view on every probe); convergence-with-views measured (1→2→3 views:
  0.476/0.294/0.140, vs cross-moment ~0.30 → single-view is OOD, not just "less info").
  [RESULTS.md](v0.2/results/temporal/RESULTS.md) §5d.
- **Agreed follow-ups (pending):** v0.2 single-view-encode+average probe, **camera-dropout retrain**
  (~30 min, 1 GPU), per-camera-singleton breakdown; report distances in units of each model's own
  cross-moment distance; run info-retention probes alongside invariance probes on every retrain.
  Full spec: [V0.2.md](v0.2/V0.2.md) Build 1 "JQ follow-ups" bullet.

## 6. N1 downstream — RoboCasa dry run (2026-07-19 → now): replacement dead → HYBRID pivot

- **Plan (JQ's 3-row table):** same Diffusion Policy, swap ONLY the observation encoder — baseline
  ResNet / Kepler e2e (random-init) / Kepler pt-enc (JEPA-pretrained, frozen) — success-vs-epochs on
  RoboCasa365. De-risks the Molmobot/real-microfactory version. [N1_ROBOCASA.md](v0.2/N1_ROBOCASA.md).
- **What happened, in order:**
  1. Infra: frame cache (decode-bound → GPU-bound), Kepler encoder arm in the DP fork, **in-domain
     pretraining gate GREEN** (state R² 0.673 vs raw 0.538, RankMe 190).
  2. **Real bug found + fixed:** DP feeds images in [-1,1], encoder assumed [0,1] → the frozen ViT ate
     OOD inputs in both Kepler arms (~10h of runs tainted; all restarted 07-20 16:29). Lesson: frozen
     backbones turn input-distribution mismatches into silent quality bugs.
  3. **Post-fix results: replacement FAILS.** PnP baseline 24→28→32% (ep 50/100/150) vs e2e 0→12%,
     pt-enc 4→2%; OpenDrawer baseline 74% vs pt-enc 22% (ep 50; earlier drafts misattributed this
     to e2e — e2e OD was paused at ~ep35, no ckpt; see N1_ROBOCASA.md). The two inits don't cleanly rank
     (ep-100 inverted the ep-50 ordering) — the failure is architectural, not the init.
  4. **Probes pinned the cause before more fleet burn:** action-readout probe — pretrained fuse 0.335
     vs 0.413 sitting in the same frozen patches (coarse-spatial ceiling); query width doesn't help
     (q8=q32=q64) → the loss is inherent to fusing under a state-readout objective with no pressure to
     keep fine object detail. **A single compact latent can't be a manipulation policy's only eyes.**
  5. **Pivot (07-21): the FLARE-faithful HYBRID row** — baseline's ResNet untouched **+** our frozen
     latent appended. This is the claim FLARE's evidence actually supports (their latent sits NEXT TO
     vision, never instead of it). Fleet reallocated: hybrid = the decision experiment → DDP-2 on
     GPUs 6+7; replacement arms run only to their ep-150 gate (honest ablation row), then cull → freed
     GPUs go to hybrid seed 1 + baseline seed 1 (±6-7% eval noise → 2 seeds).
- **The fork in the road:** hybrid > baseline ⇒ "pretrained latent adds usable info" lands. Hybrid ≈
  baseline ⇒ the pretraining recipe needs **object-level pressure** (patch recon / DINO-style
  distillation into the fuse) — already scoped as the recipe fix. Either way: learned in sim for two
  days' cost instead of on the microfactory.

## 7. Right now (2026-07-21 ~06:00) and next

- **Live fleet (8 GPUs):** baseline PnP ep~172, baseline OD ep~128, e2e ep~122 + pt-enc ep~128 (cull
  at ep 150, ~08:15), **hybrid ep~8** (ep-50 eval ~14:00 = the number of the week), watcher re-armed,
  durable CSV [results/downstream/n1_results_snapshot_20260721.csv](results/downstream/n1_results_snapshot_20260721.csv).
- **Queue after the cull:** camera-dropout retrain + JQ probes (§5), hybrid/baseline seed 1.
- **Parked with triggers** ([PLAN.md](PLAN.md) §Parked): N2 world-model (needs commanded actions —
  RoboCasa has them, RH20T doesn't), Molmobot port (no F/T), real→sim transfer row.

## Doc map (where a thing lives)

| doc | holds |
|---|---|
| [PLAN.md](PLAN.md) | roadmap + gates + external leads (ARM, GEAR/FLARE) |
| [V0.2.md](v0.2/V0.2.md) | LIVE v0.2: builds 1+2, JQ follow-ups spec |
| [N1_ROBOCASA.md](v0.2/N1_ROBOCASA.md) | LIVE downstream: builds, fleet, results table, pivot record |
| [results/temporal/RESULTS.md](v0.2/results/temporal/RESULTS.md) | every v0.2-era number (§1–3 temporal failure, §4–5 builds, §5d composition) |
| [EXPERIMENTS.md](v0.1/EXPERIMENTS.md) | Phase-1 matrix + v0.1 downstream numbers |
| ~~TEMPORAL_ARCH.md / TEMPORAL_JOURNAL.md~~ | REMOVED 07-21 (retired design / debugging saga) — `git show 8432258:<file>` |
| [DATA.md](DATA.md) | RH20T layout, rates, traps |
| this file | the narrative: plan → actual → change, with pointers |
