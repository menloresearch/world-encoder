# Status & Roadmap — where we are, what's left to prove the architecture

Snapshot as of 2026-07-07, grounded in `world-encoder/PLAN.md`. Companion to
`PHASE1_EXPLAINED.md` (plain-language results). Preliminary where noted (the Phase-1 ALL
run is mid-flight).

## The roadmap at a glance

| stage | what it proves | status |
|---|---|---|
| **Stage 0/1** — foundation + vision-only baseline | *why the project exists*: one frame can't see force (force R² ≈ 0 from vision) | ✅ done |
| **Stage 2** — single-timestep modality fusion (cfg3, cfg3+4) | fusion works in principle: z_v 0.65 > raw 0.52 | ✅ done |
| **Phase 1** — full-RH20T 5×4 transfer matrix | *one encoder for all robots*, at scale | 🔄 running (ablations/triplet still owed) |
| **Phase 2** — Temporal | **the core architectural bet**: multi-rate streams fused by real time | ⬜ next (critical path) |
| **Phase 3** — Decoder | *shows* what the latent knows (probe/demo, not a proof) | ⬜ parallel track |
| **Loss #4** — action-conditioned forward prediction | causality (correlation → cause) | ⬜ after Phase-2 gate |
| **Stage 5** — Audio · **Stage 7** — dual-arm/microfactory | more modalities / embodiments | ⬜ later, gated |
| **FLARE / GR00T** — our encoder as frozen g(·) | external downstream validation | ⬜ opportunistic |

## What's been proven vs what's left

**Proven:** vision alone is insufficient (Stage 1); fusing sensors into one latent by
cross-modal prediction makes vision carry state (Stage 2); at full scale, **one encoder ≈
specialists across 4 robots and force transfers** (Phase 1, preliminary — force is the clean
win, joints marginal).

**Not yet proven — this is what "fully proving the architecture" needs:**
- **Time / multi-rate fusion.** The headline bet — fuse 100 Hz force with ~10 Hz jittery
  vision via real timestamps, no resampling — is untested because Phase 1 is single-timestep.
  Only loss #1's *modality* half is done; its *time* half is Phase 2.
- **Causality.** Same-time alignment is correlational; loss #4 (predict future from actions)
  is what earns "world model." Gated behind temporal.
- **Claims-protection ablations** (esp. the vision-only-*trained* control) — without these
  even the Phase-1 result isn't publishable.

## Temporal or decoder? → Temporal is the critical path; decoder is parallel

- **Temporal (Phase 2)** is what *proves the architecture* — the continuous-time / multi-rate
  differentiator only gets tested here. Phase-1 argues for it: fusion helped force
  (instantaneous) but barely helped joints (motion needs time). Data side is already done
  (tick-anchored chunks, native-rate windows, `ts` cached per token) → **model-side work
  only**: continuous-time embedding + multi-tick windows + mask over (modality × time). This
  is where the **200 ms chunk + 1D-CNN proprioception** design slots in.
- **Decoder (Phase 3)** does *not* prove the architecture — it's a "superpowered linear
  probe" that visualizes/quantifies what's already in the latent (later, a compression demo).
  Runs on frozen Phase-1 latents → proceeds in parallel, doesn't block temporal.

## Immediate sequence

1. **Close out Phase 1:** finish 5 seeds → full 5×4 mean±std table → **vision-only-trained
   ablation** (protects the headline) → wire triplet-accuracy eval → blog figures
   (PCA of joint-encoder latent colored by robot/task/cfg).
2. **Phase 2 — Temporal** (main bet), with the **decoder** running alongside.
3. After the temporal gate: **loss #4** (causality), then **audio**, then **FLARE/GR00T**
   downstream validation.

**One-liner:** we've proven fusion and one-encoder-for-all at a single instant; proving the
*architecture* means proving it across *time* — that's Phase 2, and it's next. The decoder
is a parallel demo, not a proof.

## Phase-2 temporal design

The concrete, authoritative Phase-2 design — continuous-time embedding (mTAN's embedding
only; Perceiver is the other half), Fourier-first→learned-frequency sequencing, the
log-spaced frequency-range crux, the 1D-CNN as the dense-F/T tokenizer, causal-vs-
bidirectional, and the continuous-time-vs-resample ablation — is the checklist in
**`world-encoder/PLAN.md` § Phase 2 (items 2.1–2.5)**. That is the single source of truth;
this doc is the lighter status snapshot and defers to it.

## Open decisions (from PLAN.md)
Camera choice v1 (fixed external, wrist excluded — already applied in code); file ownership
(trainer vs triplet wiring both touch the shared pipeline code); cfg5 stays with ee masked;
push the outstanding unpushed local commits; LICENSE (repo public without one); merge
direction (Phase-1 builds on `user/jiaqi`; `user/ishneet` is stale).
