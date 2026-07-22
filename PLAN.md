# PLAN v2 — Full-RH20T Vis+State Matrix Run → Temporal → Decoder

Rewritten 2026-07-06 after the weekend's work landed on `user/jiaqi` (see DATA.md /
METRICS.md there). Supersedes the 2026-07-03 PLAN (kept in git history). Results in
[EXPERIMENTS.md](EXPERIMENTS.md); data in [DATA.md](DATA.md).

## Current status (2026-07-21)

**The full plan-vs-actual narrative now lives in [STORY.md](STORY.md)** — read that first to orient.
Roadmap-level: **Phase 1 DONE + PUBLISHED** (one-encoder-for-all holds, force = the clean cross-modal
win; [EXPERIMENTS.md](EXPERIMENTS.md)). **Phase 2 temporal-in-encoder RETIRED** (NH1 gate failed twice;
[results/temporal/RESULTS.md](results/temporal/RESULTS.md); full design + saga in git history —
`TEMPORAL_ARCH.md`/`TEMPORAL_JOURNAL.md`, removed 07-21, `git show 8432258:<file>`). **v0.2 re-scoped (07-17) and BOTH BUILDS DONE + GREEN (07-19):** multi-cam holds rank AND
improves force (0.283 vs 0.251); predictor beats carry-forward 28–36%/horizon. Canonical v0.2 doc =
[V0.2.md](V0.2.md) (incl. pending JQ follow-ups: camera-dropout retrain + probes). **N1 downstream is
LIVE on RoboCasa** ([N1_ROBOCASA.md](N1_ROBOCASA.md)): encoder-as-REPLACEMENT is dead (0–12% vs
baseline 24–32%; probes pinned the cause — no object-detail pressure in the objective) → pivoted to
the **FLARE-faithful HYBRID row** (frozen latent ADDED to the policy's vision); baseline-vs-hybrid is
the decision number of the week. Active external leads: **ARM** (edge reference model) and
**FLARE/GR00T** (encoder as g(·), §External).

| stage | proves | status |
|---|---|---|
| Stage 0–2 + Phase 1 matrix | one encoder for all robots (single-timestep) | ✅ DONE + published |
| Downstream (surprise · state/pixel decode) | the encoder is *useful* on the frozen model | ✅ DONE |
| Phase 2 (v0.2) — re-scoped | per-frame multi-cam encoder + time-in-predictor | ✅ both builds DONE + GREEN (07-19); JQ follow-ups pending ([V0.2.md](V0.2.md)) |
| N1 downstream (RoboCasa) | the latent helps a *policy* (task success) | 🔴 replacement arms FAIL → 🔁 HYBRID row training; decision number ~07-21 pm |
| Phase 3 — Decoder (video) | shows what the latent knows | ✅ pipeline done (PixNerd) |
| Loss #4 (action-cond.) · Audio · FLARE g(·) · ARM | causality / modalities / external | ⏸️ N2 unparks on RoboCasa commanded actions; ARM/FLARE external |

## Phase 1, downstream, and the 2026-07 groundwork — DONE (archived)

The downstream-first pivot, the weekend preprocessing log, the `user/jiaqi` code review, the full-RH20T
blockers, and the Phase-1 build checklist all lived here. They are **complete** and were pruned for
brevity — the record is in git history + [EXPERIMENTS.md](EXPERIMENTS.md) + [DATA.md](DATA.md). One-liner:
chunk pipeline → 5×4 matrix + ablations → **gate PASSED 2026-07-07** → paper published; downstream
(surprise AUROC 0.90; state R² 0.45 / joint 0.69; PixNerd pixel decode) all shipped, in EXPERIMENTS.md.

## Phase 2 (v0.2) — RE-SCOPED (2026-07-17): per-frame multi-cam encoder, time in the predictor

**The decision.** The original Phase-2 bet — fuse a *window of ticks* into one latent (time inside the
encoder) — is retired. It failed the NH1 gate under two independent objectives: the flat-Perceiver-over-ticks
→ mean-pool fusion degrades the latent (RankMe 51 vs 134; present-force probe halved 0.10 vs 0.21; P3 dq and
P4 future-force both lose to v0.1 at every horizon). Root cause localized to the temporal *fusion*, not the
head or the masking (both independently ruled out). Numbers: [results/temporal/RESULTS.md](results/temporal/RESULTS.md);
full narrative + design: `TEMPORAL_JOURNAL.md` / `TEMPORAL_ARCH.md` §18/§20 (git history, removed 07-21).

**Why this is the right call, not a retreat.** Every path we care about puts time *outside* a per-frame
encoder: FLARE's g(·) is per-frame (policy does time); LeWM = per-frame encoder + separate next-embedding
predictor; RoboTTT = frozen/per-frame, time in the policy (fast weights); JQ = "the VLA handles the temporal
bit / use JEPA for the time dimension." Our goal is a g(·) feeding a time-handling VLA, so temporal belongs in
the predictor. **"Fixing v0.2 properly" and "doing FLARE" converge on the same artifact** — a next-embedding
predictor on a frozen per-frame encoder. So this is a re-scope, not an abandonment: temporal stays alive, it
just moves from the encoder to the predictor.

**Caveat (don't over-claim).** This is goal-dependent, not a law. V-JEPA 2's encoder is spatiotemporal
(encodes video clips) and works — because it *is* the rollout model. Ours is a g(·) feeding a VLA that already
does time, so per-frame encoder + predictor-side time is the right *division of labor for our downstream*, not
a universal "time can never go in an encoder." Ready answer if cited V-JEPA 2 back: "their encoder is the
rollout model; ours feeds a VLA that already handles time."

### The re-scoped path (one coherent track, in order)

- [x] **2.1 Multi-cam on RH20T — the immediate build.** Single-timestep, spatial-only:
      `[B, 1, n_cam·196, 768]` (the `1` = one tick, explicitly no temporal). Plays to our validated strength
      (v0.1 works at C=1; low architectural risk), and RH20T has **both** multi-view and F/T so it keeps the
      force / cross-modal story alive. Delivers JQ's actual ask: does the Perceiver bottleneck compress N views
      without tanking RankMe / probe R² (Perceiver-compression stress), and can the VLA consume one compressed
      latent instead of `n_cam·196` patch tokens (saves LLM context). JQ's multi-cam file =
      `mm_perceiver3.py`; DONE — the spec is the code (`train_multicam.py`); numbers in RESULTS.md §5c.
- [x] **2.2 Next-embedding predictor on the frozen per-frame encoder — temporal, relocated.** LeWM-style:
      per-frame `z_t` → predict the future latent, keeping the v0.1 cross-modal head + joint-SIGReg. This is
      simultaneously (a) JQ's "JEPA for time," (b) the temporal capability we wanted, and (c) the de-risk for
      the FLARE integration (the cheaper, fully-specified intermediate while FLARE code is unreleased). Target
      structure = the recurrent **carry-forward Perceiver belief-state**: carry `z_t` as a persistent state and
      roll it forward under actions → that IS loss #4 and the encoder→world-model step. Time lives here, not in
      the encoder. **SIGReg-under-time rule still holds:** per-timestep marginal, never a time-pooled latent
      (mirrored in paper §3.5, PR #8).
- [~] **2.3 Downstream validation = FLARE-style integration (ELEVATED); Molmobot bespoke test DEFERRED.**
      The downstream payoff is plugging our per-frame encoder in as FLARE's target encoder g(·): the VLA gets an
      auxiliary loss predicting our future-observation latents. **2.2 is the standalone prototype of the same
      future-latent bet** — the predictor and FLARE are one line of work at two levels (standalone vs in-policy),
      so 2.2 directly de-risks this. Gated on FLARE code being unreleased → reimplement its auxiliary loss on an
      open VLA and/or coordinate with GR00T (§External validation lead). The separate **port-to-Molmobot
      downstream test is parked for now** (needs mp4→ViT-patch precompute + a bytes→JSON h5 loader; Molmobot has
      no force/torque so it can't carry our differentiator) — revisit once 2.2 + the FLARE path are moving.

**Data decision (RH20T vs Molmobot — sequence, don't choose).** Do **2.1 multi-cam on RH20T first** (known-good
data, has F/T), then swap to Molmobot on a validated architecture — one variable at a time (arch change, then
data swap), not both at once. Molmobot **phase5** (`/mnt/nas/datasets/microfactory-phase5-molmobot`, 24 GB sim,
~19k episodes, 181 train + 10 val houses) = **5 cameras** (exo front/overhead/side + L/R wrist), bimanual 14-DOF
(qpos + qvel + `joint_pos` actions), language-annotated, ~29 Hz single-rate; h5 packs each timestep as
JSON-in-uint8. **It has NO force/torque, no ee/TCP pose** → it cannot carry our cross-modal *force* result, and
its single rate removes the native-rate motivation. So: **RH20T = force story + arch validation; Molmobot =
multi-cam-compression + VLA-downstream** where F/T isn't the point. (Not yet checked: `asimov2-arm-molmobot` — a
real-arm rig that may have force; check if the F/T question becomes decisive.)

**Gate (v0.2):** (2.1) multi-cam latent holds RankMe + probe R² vs single-view v0.1 *while* compressing N views;
(2.2) next-embedding predictor beats single-frame on future-state at varying Δt with RankMe stable. Loss #4
(action-conditioned) only after 2.2.

**Execution notes — cheap pre-checks + guardrails (consolidated in [V0.2.md](V0.2.md)).** Key points:
the two builds are **independent** (predictor runs on existing `phase1` + `caches/cfg*.npz`; multi-cam needs a
new K-camera re-precompute) so order is a priority call; **pre-check the predictor NOW** with a simple
`z_t→z_{t+Δ}` fit vs naive carry-forward before building the belief-state; **guardrails** — freeze the encoder
(training the encoder is what broke v0.2), predict the latent *set* not a pooled vector, §2.2 is the
*unconditioned* precursor to loss #4, and watch multi-cam for bottleneck rank-collapse (fix = more latents).
**Pre-check A DONE + GREEN (2026-07-17):** a simple `z_t→z_{t+Δ}` predictor beats carry-forward by ~25–35 %
across all embodiments → §2.2 worth building. Results: [results/temporal/RESULTS.md](results/temporal/RESULTS.md) §4.

### Next steps after 2.1 + 2.2 (detailed recipe consolidated in [V0.2.md](V0.2.md) §Next steps)
Each grounded in a paper template; order = validation first, capability in parallel.
- **N1. Downstream encoder-swap test — the real validation (FLARE ablation discipline).** Fix one policy
  (behavior cloning / small diffusion policy), swap ONLY the observation encoder (raw ViT → SigLIP-2/DINOv2 →
  ours frozen → ours+predictor), measure **task success from rollouts**. Not gated on FLARE code — the fuller
  future-latent-auxiliary-loss version comes when GR00T coordination lands. **Dataset:** Molmobot = usefulness
  test only (has success flags/actions/language/5-cam but **no force/torque**); prefer pretrain-on-Molmobot vs
  transfer-from-RH20T (real→sim confound); force-in-downstream stays an RH20T item.
- **N2. Action-conditioned predictor — world-model capability (LeWM + V-JEPA 2-AC; parallel track).** Separate
  predictor `ẑ_{t+1}=pred(z_t,a_t)`, MSE in latent space, action via AdaLN. **Biggest borrow: DROP the EMA** —
  LeWM shows SIGReg-on-marginal prevents collapse, and SIGReg is already our method (clean diff vs FLARE/V-JEPA).
  Then optional V-JEPA-2-AC-style planning (roll forward → MPC) as the eval. This is loss #4.
- **N3. Denser re-precompute + write-up (EgoScale methodology).** Report prediction loss as a cheap scaling
  proxy but **validate against real downstream success** (never one metric). Re-precompute native-rate frames
  if fine dynamics (contact onset) are needed — our Pre-check A green was long-horizon only (~1.7 s/tick cache).

**Retired temporal-in-encoder — kept for the record, not the roadmap.** The mTAN/Time2Vec continuous-time
embedding, per-stream 1D-CNN tokenizers, continuous-time-vs-resample ablation, and the *window-Perceiver "(a)"*
are in git history (`TEMPORAL_ARCH.md` §18/§20 + `TEMPORAL_JOURNAL.md`, removed 07-21,
`git show 8432258:<file>`). That is the *time-in-encoder* design; it goes live
again only if we ever build a **standalone rollout world-model** (V-JEPA-2-style goal). The recurrent
carry-forward "(b)" survives — it is now §2.2, the predictor.

## Phase 3 — Decoder (parallel track)

- [ ] **3.1** robot_state decoder on frozen Phase-1 latents — "the generative decoder is
      our superpowered linear probe": quantifies latent content, cheap, days not weeks.
- [ ] **3.2** PixNeRD → latent diffusion pixel decoder (a viz/probe tool at this stage,
      not a training signal).

## External validation lead — FLARE / GR00T (NVIDIA GEAR; 2026-07-06 lit sweep)

NVIDIA GEAR's **FLARE** (arXiv:2505.15659, CoRL 2025, shipped in GR00T N1.5) trains a VLA
with an auxiliary JEPA-style loss: predict the *latent* of the observation 16 steps ahead,
produced by a target encoder g(·). Their ablation shows **g(·) quality is the deciding
factor** (none 43.9% → raw SigLIP-2 49.6% → pooled 50.9% → their learned encoder 55.0%) — and
their best g(·) is **vision-language only** (no proprio/F-T, single rate) and needs an EMA
moving-target hack. That is a drop-in slot for world-encoder: **multimodal, native-rate,
SIGReg-frozen** (no EMA). Their own ablation predicts a better g(·) buys policy success, so the
test is concrete — swap our frozen encoder into FLARE's alignment target and measure. Blocked on
FLARE code being unreleased (GR00T N1.5 mentions it; absent from public Isaac-GR00T, #211/#215)
→ needs external coordination; not solo, not now.

**The portfolio gap = our differentiator.** Across GEAR's stack — DreamDojo (44k h human-video
WM), DreamZero (WM = zero-shot policy), DreamGen (WM = data engine), EgoScale (20k h human-video
scaling), FLARE — **every world model is vision(-language) + actions only: no force/torque, no
native-rate proprioception, single-view.** DreamDojo's fast-motion failures and DreamGen's
contact-rich pseudo-label noise are plausibly missing-modality symptoms. Native-rate multimodal
(incl. F/T) fusion is the open lane.

**Borrowable techniques (for the temporal / action-conditioned roadmap):**
- **Latent-action VAE** (DreamDojo): learn continuous latent actions from RH20T *frame pairs* →
  enables an action-conditioned stage (loss #4) *without* action labels. "Reset the conditioning
  layer" = a cheap embodiment-transfer trick.
- **Relative SE(3) deltas** (EgoScale): an embodiment-invariant normalization for our TCP/ee
  streams across RH20T's 4 robots.
- **Prediction-loss as a scaling proxy** (EgoScale: log-linear val-loss→success, R²=0.998;
  AutoEval: val-MSE *anti*-correlates with real success): track LeJEPA prediction loss / latent-
  probe quality as a scaling proxy — and treat any single naive metric as capable of misleading.
- **Compute contrast** (cite DreamZero): dynamics-grounded pretraining beats semantic VLA
  pretraining, but at 14B / GB200 cost; our bet is a ~2M-param fusion head on frozen features +
  latent prediction that captures the robot-relevant part at a fraction of the cost.

## Parked — each with an explicit trigger

| item | trigger |
|---|---|
| Multi-view training objective | Phase-1 analysis: measure same-tick cross-view latent distance with the v1 encoder. If views don't already cluster → add cross-VIEW masked prediction (predict side-view latent from wrist-view + state). Predict-don't-equate — never latent *equality* across views (dual-arm / wrist-cam info-asymmetry objection). Include latent **sum/mean-pool of per-view latents as the fusion baseline** (the neural-codec idea — the additive trick itself doesn't transfer: audio mixes additively at the sensor and codecs are near-lossless, cameras are projections and JEPA latents are lossy by design — but it's the right dumb baseline vs Perceiver fusion with view-tagged tokens). |
| Loss #4 (action-conditioned) | v0.2 §2.2 predictor lands. Target arch = **recurrent carry-forward Perceiver latent** rolled under actions — see §Phase 2 (v0.2) 2.2. |
| Audio (full Stage 5) | Temporal proven. |
| Discrete latent / disentanglement; dual-arm embedding-sum | Microfactory data (Stage 7) — RH20T is single-arm. |
| Loss balancing (Kendall / GradBlend) | Only if the 3-loss balance misbehaves at scale (it hasn't). |
| Theory reading | Rate-distortion → when sizing the bottleneck ablation; PID/synergy → when interpreting the transfer matrix; identifiability/causality → Phase 2+ with actions. |

## Open decisions

1. **LICENSE** — repo is public without one; add a lean Apache-2.0 (deferred TODO from the OSS
   cleanup, PR #11).

*(The 2026-07 "blocking decisions" — camera choice, cfg5-in, merge direction, file ownership — are all
resolved; Phase 1 shipped on those.)*
