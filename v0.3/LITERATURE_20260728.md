# Literature review — where compressed latents pay rent (v0.3 evidence base)

**Date:** 2026-07-28. **Why:** E5 falsified "better representation → better policy" (probe→downstream
inversion), E4 showed dose-dependent aux-loss harm, and E6 now carries the v0.3 thesis — so E6/E4-redesign
decisions should rest on verified literature, not priors.

**Method:** deep-research fan-out (5 angles → 24 primary sources → 120 extracted claims). The automated
3-vote verification panel died on API rate limits, so the **8 decision-critical sources were re-verified
manually** (one agent per source, checked against full paper text, all numbers below confirmed against
tables/quotes unless marked UNVERIFIED). Verification caught real extraction errors — every claim from an
UNVERIFIED source should be treated as single-pass extraction, not fact.

Verified: Patch Policy (2607.18236), FLARE (2505.15659), Du et al. grad-gating (1812.02224),
FoAR (2411.15753, RA-L'25), RDP (2503.02881, RSS'25), FACTR (2502.17432), DINO-WM (2411.04983),
robosuite sensor docs+source (v1.5.2).
Unverified (labeled inline): LfS (2212.05749), SpawnNet (2307.03567), 2304.04591, Theia (2407.20179),
2312.12444, robomimic study, MCR (2410.22325), 2502.03270, REPA-E (2504.10483), FMT (2509.19047),
FILIC (2509.17053), 2512.24497, V-JEPA-2-AC (2506.09985), 2603.12231, WM survey (2606.00113),
swm bench (2605.21800).

---

## Q1 — Has a compressed learned latent ever helped a reactive BC policy? **NO** (high confidence)

- **Patch Policy (VERIFIED):** 40% relative over global-pooled reps (abstract, verbatim). The killer row
  for us: **DynaMo — a dynamics-JEPA pooled latent, i.e. our configuration — scores 0.28/0.27 on Cube vs
  1.68/1.73 for dense patches** (same heads, Table 1). Learned conv compression of patches: 0.69 (256) →
  0.52/0.53/0.51/0.48 (64/16/4/1) — degrades, though NOT strictly monotone (64↔16 inverts; paper never
  says "monotone"). Frozen patches, zero fine-tuning, beat fine-tuned OpenVLA-OFT by 18% at ~0.7% params
  (VQ-BeT variant: 29.5M trainable, ~11ms; the DP variant is ~9M but ~450ms denoising-bound).
  Suites: Push-T, LIBERO Goal, BlockPush, Cube + 3 real Franka tasks. **No RoboCasa/MimicGen** — E2 is
  still the needed calibration on our bench.
  - Encoder nuance (corrected): WebSSL beats V-JEPA 2 on 8/8 task×head combos, but DINOv2 only 4/8;
    worst encoder is SigLIP 2. "Image-SSL > video-JEPA" holds for WebSSL, is a split for DINOv2.
- **SpawnNet (UNVERIFIED):** frozen *pooled* embedding is the bottleneck; dense multi-layer adapter
  fusion improves every backbone; CLS-tiled ablation −24.9 on real Place Bag.
- **LfS (UNVERIFIED):** scratch CNN + augmentation matches frozen PVRs (R3M/MVP/PVR) across BC+RL.
- The apparent counterexamples (Theia, MCR — UNVERIFIED) are **backbone replacements producing dense
  features**, not fused compact latents concatenated beside a vision stream.

**Consensus position:** for reactive BC, detail-destroying compression loses; compression must pay rent
on axes patches don't have (prediction, fusion, planning-compactness). **Our E5 null is the published
pattern** — DynaMo's row is near-exactly our experiment, independently run by NYU.

## Q2 — Aux-loss harm on BC: mechanisms and safe designs (E4 verdict: redesign, one point) (high confidence on facts; medium on transfer)

**FLARE (VERIFIED, all numbers exact):** 70.1% vs 61.9% policy-only on RoboCasa-24; 55.0% vs 44.0% GR-1.
In **every** ablation — λ∈{0.1,…,0.5} (grid is 5 points, not 4), layers 4–8, EMA ρ∈{0.99,…,1.0}, SigLIP2
vs action-aware targets — FLARE **never once dropped below the no-aux baseline**. Worst λ (0.5) still +7.8pp.
Our monotone-harm curve has no analogue anywhere in their paper.

Structural differences from our E4 (the candidate explanations for the sign flip), verified from their
pseudocode:
1. **Loss placement:** cosine loss on MLP-projected activations of **32 dedicated learnable future tokens
   at DiT layer 6/8** — flow-matching and alignment "proceed along separate streams … interact via
   self-attention". **Important correction: this is NOT a stop-grad isolation.** `vl_tokens` are not
   detached; alignment gradients DO reach the policy's vision-language encoder through cross-attention,
   and the encoder IS trained. The isolation is architectural (indirect, diffused path), not hard.
2. **Chaseable target:** target = in-domain action-aware embedding (pretrained WITH action supervision)
   or EMA-tracking (ρ=0.995) of the policy's own evolving encoder; `requires_grad=False` on target side
   only. Our E4 target was a **fixed frozen JEPA z_{t+H}** — an irreducible-floor target the policy can
   never match, so the gradient never anneals. This matches our E4 diagnosis.
3. Layer-4 placement "leads to a notable drop" — but only −2.6pp vs layer 6, still +8.5pp over baseline.

**Du, Czarnecki et al. (VERIFIED — arXiv:1812.02224; ICLR'19 submission, REJECTED — cite as arXiv):**
- Fixed-λ aux loss demonstrably causes sustained negative transfer (ImageNet far pairs; sub-optimal-teacher
  distillation caps RL performance below pure RL).
- **Cosine-gated rule:** apply aux gradient only when cos(∇L_main, ∇L_aux) ≥ 0 (or scale by max(0,cos)).
  **Proposition 1: provably converges to critical points of the main task** — aux provably cannot cause
  permanent pull-away (guarantee is no-divergence, not speedup; Appendix D constructs a slowdown case).
- Mechanism: aux usefulness is non-stationary ("helps initially but hurts later") — no constant λ is
  correct at all times.

**REPA-E (UNVERIFIED, mechanism-relevant):** task gradient into a VAE encoder collapses latent variance
while the training objective looks healthy — published analogue of our "normal action-MSE, harmed
features"; fix is gradient routing (stop-grad) + anchor losses.

**E4 verdict:** the *mechanism* (predictive shaping helps BC) is alive in the literature; our
*implementation* (fixed frozen target + direct obs-encoder loss + constant λ) sits outside the region any
published positive instance occupies, and inside the region Du et al. prove can sustain harm. If the live
w0.01 point doesn't beat baseline: dose-response row closes as "fixed-target-on-encoder harms at all
doses," and the ONE redesigned point worth running is: **alignment on a dedicated policy-internal
token (deep in the DP head), chaseable/EMA target, cos-gated or annealed weight.** If even that fails at
a modest λ, the axis is dead on our stack and the writeup says so with the FLARE deltas as the tested
explanations.

## Q3 — Probe↔downstream inversion: well-precedented (high confidence)

- **robomimic study (UNVERIFIED but canonical):** best-val-loss checkpoint 50–100% worse than best
  rollout checkpoint — proxy/closed-loop divergence is the norm, not an anomaly.
- **2312.12444 (UNVERIFIED):** linear-probe accuracy fails to predict manipulation robustness;
  emergent segmentation (attention-map Jaccard) predicts OOD success; control-pretrained reps (MVP)
  collapse OOD (39.86→6.63).
- **2304.04591 (UNVERIFIED):** representation rankings change across BC/RL/reward regimes; state-readout
  probe loss anti-correlates with BC success (r=−0.78) on Franka-Kitchen.
- **2502.03270 (UNVERIFIED):** mechanism precedent — temporally entangled features map different task
  stages to similar latents (one-to-many action mapping) → good static probes, broken closed loop.
- **DINO-WM-adjacent (swm bench, UNVERIFIED):** prediction error doesn't separate planning success/failure.

**Position for the writeup:** our inversion (geometry probes up, success down) has multiple published
precedents; the metrics that DO track control are task-anchored (state readout, manipulation-centric
attention, temporal structure) — which is what E1's gates are. Caveat honestly: even those correlations
are regime-specific (2304.04591).

## Q4 — Force fusion (E6): the bar is real and genuinely unbeaten (high confidence)

**Has a learned fused latent beaten raw F/T passthrough at matched data? NO published instance.**
- **RDP (VERIFIED, RSS'25):** the cleanest matched comparison — raw 6-D wrench concat **matches or beats**
  optical-tactile representations on all 3 tasks (0.95/0.87/0.70 vs 0.90/0.77/0.48). Correction: the
  optical-tactile rep is **non-learned PCA(15)**, so "raw beats *learned* latents" overstates — but no
  learned tactile encoder wins anywhere in the paper either (tactile-emb ≈ tactile-img, 0.39 vs 0.41).
  Naive tactile-in-obs ≈ no gain vs vision-only (0.39 vs 0.44 / 0.50 vs 0.57 — paper frames as "similar,"
  and it HELPS on Lifting 0.08 vs 0.00) with a contact-oscillation failure mode — our E5 pattern.
  Their win comes from **insertion architecture**: slow latent-chunk policy + fast (24 FPS) tactile into
  the action DECODER. Note: tactile IS also in the slow policy's obs — the fast-path decoder injection is
  the differentiator, not total exclusion.
- **FoAR (VERIFIED, RA-L'25):** contact-predictor-**gated** fusion beats vision-only AND both ungated
  encoded-F/T variants (Wiping 0.875 vs 0.500/0.575/0.475; Peeling 0.756 vs 0.377/0.487/0.524);
  force-concat lands BELOW vision-only on Wiping (0.475<0.500) — ungated force can actively hurt (noise
  during non-contact). Gate: φ(t) from a contact predictor trained with BCE on labels **auto-extracted by
  thresholding F/T magnitude in demos** (free in sim — MuJoCo gives GT contact). Correction: their
  "raw" baselines are *encoded* F/T inserted ungated — literal raw passthrough is untested there.
  Force encoding: 200×100Hz samples → per-timestep tokens → small **transformer** aggregator (MLP-only
  ablation is worse).
- **FACTR (VERIFIED):** naive force+vision 61.2% vs 87.5% with a **vision-corruption curriculum**
  (Gaussian blur, linearly annealed σ) — policies ignore force because it's near-zero most of the time
  and vision suffices on train. Force = joint torque, not wrist F/T; zero sim results.
- **FMT (UNVERIFIED):** 83% vs 22% RGB-only across 6 real tasks, BUT no raw-passthrough baseline; gear
  assembly 0.40→0.95 monotone in F/T sampling rate 30→200Hz — rate-matching F/T down to vision rate
  discards task-critical signal.
- **FILIC (UNVERIFIED):** EE wrench (physics transform) ≫ joint torque as input (+33–53pp over
  vision-only); learned fusion never ablated vs raw concat.

**robosuite facts (VERIFIED against docs + v1.5.2 source):**
- F/T sensor XML (`force_ee`/`torque_ee` at `ft_frame` site) exists in every gripper model, **but there
  is NO default F/T observable** — `robot0_eef_force` does not exist. Access: `env.robots[0].ee_force/
  .ee_torque`, `recent_ee_forcetorques` buffer, or register a custom Observable.
- Default readings are **clean, deterministic, delay-free MuJoCo ground truth**; corrupter/delayer default
  to None; noise/delay helpers exist (`create_gaussian_noise_corrupter` etc.) but ALL parameters are
  caller-supplied — no canonical F/T noise spec anywhere.
- Per-observable `sampling_rate` exists, but the obs dict returns at control_freq — **capturing 100Hz F/T
  under 20Hz control needs a custom filter** (pattern in sim2real docs) or the `recent_ee_forcetorques`
  buffer.

### E6 design deltas (from the above)
1. **Keep the pre-registered bar** (fused latent must beat raw passthrough): the literature confirms it's
   the right bar and that nobody has cleanly cleared it — this is the field's open edge, which is also
   the comms story.
2. **Raw arm = windowed history, not single sample:** ~2s@100Hz window (FoAR recipe), and consider a
   rate ablation (FMT's 30→200Hz monotone result) — a single-tick F/T sample would strawman the raw arm.
3. **Add a 4th arm (or replace "force latent" v1): contact-gated fusion** — the only published mechanism
   that beat ungated force. Contact labels are free in sim. If our fused latent can't beat raw, gated
   fusion is the fallback mechanism that still tests "learned fusion adds."
4. **Expect the FACTR failure mode:** force near-zero most of the episode → policy ignores it. Cheap
   mitigations: vision-corruption curriculum (annealed blur) as an ablation; at minimum, log
   force-attention/usage so "no lift" is diagnosable as ignored-input vs useless-input.
5. **Chunking confound:** DP action chunks create an open-loop window that caps force reactivity (RDP:
   chunk 8→2 collapses grasp 100%→20%; ensembling can't fix). Log chunk length as a limitation; do NOT
   naively shrink chunks.
6. **Sim-fidelity choice, made explicitly:** default F/T is noiseless GT. Either inject Gaussian noise +
   delay (helpers exist, parameters ours to pick and report) or state clean-sim as a scope limit.
   Reviewers/JQ will ask.
7. **Engineering:** must register custom F/T Observables in BOTH demo regeneration and policy training;
   plan for the control-freq sampling trap (use the buffer or custom filter for >control-rate history).
8. Tasks (Square/ToolHang/Coffee) remain on our priors — no external force-sensitivity evidence for
   MimicGen tasks in the verified set; FMT's precision-assembly result is the nearest analogue. Fine, but
   say so in the writeup.

## Q5 — Latent planning (E8): real payoff regime, but NOT for pooled latents (medium-high confidence)

- **DINO-WM (VERIFIED, all numbers exact):** frozen DINOv2 + 19M ViT dynamics + CEM (100×10) on latent-MSE
  goal cost → PushT 0.90 vs DreamerV3 0.30 / IRIS 0.32 / TD-MPC2 0.00. **But dense patches win IN PLANNING
  TOO: 0.90 vs CLS 0.44 / R3M 0.42 / ResNet-pooled 0.20**, same dynamics+planner. Single-vector WMs degrade
  as scene complexity grows. Correction from extraction: the "20–50 step planning horizon" detail was
  fabricated; demos absent at *test* time only (PushT training data = noised expert replays).
- **2603.12231 (UNVERIFIED):** channel compression is nearly free (384→8) if all patch tokens are kept —
  compactness should come from channels, not spatial pooling; latent-planning success needs *trajectory
  straightness*, not probe quality. Evidence limited to 2D toy envs.
- **V-JEPA-2-AC (UNVERIFIED):** zero-shot latent planning beats Octo on real Franka — payoff is in the
  planning regime, consistent.
- **2512.24497 (UNVERIFIED):** tuned JEPA WM beats DINO-WM/V-JEPA-2-AC incl. RoboCasa Reach 25.4 vs
  19.1/16.2; claims LOWER-dim latents ease planning — in direct tension with DINO-WM's dense-wins result;
  unresolved in the field.
- **swm bench (UNVERIFIED):** standardized planner/WM stack exists (DINO-WM/LeWM/PLDM/TD-MPC2 + CEM/MPPI);
  no robosuite/RoboCasa envs; planning brittle under covariate shift.

**E8 verdict: not now.** The planning path is where prediction pays (supported), but the field's own
results say a 256-d Perceiver-pooled latent is the *wrong shape* for it — patch-token dynamics win, and
compactness should be taken on channels. An E8 that would be credible for v0.4: frozen encoder, two arms
(our fused z vs patch tokens), small ViT dynamics, CEM to goal latents on a RoboCasa task (2512.24497
precedent: Reach). Defer until E6 resolves; it competes for the same GPUs and the v0.3 story doesn't need it.

## Comms positioning (brain-internal#4 / JQ Slack)

1. **E5 negative = published pattern:** Patch Policy's DynaMo row (0.27 vs 1.73) is our experiment run
   independently at NYU; SpawnNet ablation, RDP tactile-concat agree in direction. Frame: "concat-into-
   reactive-BC is the one insertion mode neither the world-model line nor the dense-feature line supports;
   we falsified it on RoboCasa with matched data/recipe/checkpoints."
2. **E4 negative ≠ FLARE contradiction:** no published positive instance puts a fixed-frozen-target cosine
   loss directly on the obs encoder; FLARE's own recipe differs on placement (dedicated tokens, layer 6)
   and target (chaseable/EMA) — and Du et al. supply the theory (+ provably-safe gate) for why fixed-λ
   fixed-target can sustain harm. Frame E4 as a *mechanism finding with a theoretical account*, and the
   redesigned point as the discriminating test. Cite Du et al. as arXiv:1812.02224 (not "ICLR").
3. **The v0.1-undermining worry resolves cleanly:** the negatives close the concat mode; the two modes the
   literature leaves open — force fusion beating raw passthrough (nobody has) and prediction-for-planning —
   are exactly v0.1's assets (force probe 0.283 vs 0.251; predictor beats carry 11–36%). E6 sits on the
   field's open edge, pre-registered.
4. **Honesty guardrails:** label single-source claims; swm-bench point that cross-paper "latents help/hurt"
   comparisons are confounded by non-standardized eval → our matched-everything discipline is the
   contribution to emphasize.

## Source table

| # | Source | What it gives us | Status |
|---|---|---|---|
| 1 | Patch Policy, arXiv:2607.18236 (NYU/FAIR) | dense>pooled 40% rel; DynaMo row; compression ablation | **VERIFIED** |
| 2 | FLARE, arXiv:2505.15659 (NVIDIA GEAR) | aux loss +8.2pp RoboCasa-24; never below baseline; token/target design | **VERIFIED** |
| 3 | Du et al., arXiv:1812.02224 (DeepMind) | cos-gated aux gradients; convergence proof; non-stationarity | **VERIFIED** (ICLR'19 rejected — cite arXiv) |
| 4 | FoAR, arXiv:2411.15753, RA-L'25 | contact-gated fusion beats ungated; force-concat < vision-only | **VERIFIED** |
| 5 | RDP, arXiv:2503.02881, RSS'25 | raw wrench ≥ tactile reps; slow-fast decoder insertion; chunking trap | **VERIFIED** |
| 6 | FACTR, arXiv:2502.17432 | force ignored w/o curriculum (61.2→87.5); joint torque; no sim | **VERIFIED** |
| 7 | DINO-WM, arXiv:2411.04983 | dense patches win planning too (0.90 vs 0.44 CLS) | **VERIFIED** |
| 8 | robosuite v1.5.2 docs+source | F/T sensor yes, observable NO; clean-by-default; multi-rate traps | **VERIFIED** |
| 9–24 | LfS 2212.05749 · SpawnNet 2307.03567 · 2304.04591 · Theia 2407.20179 · 2312.12444 · robomimic study · MCR 2410.22325 · 2502.03270 · REPA-E 2504.10483 · FMT 2509.19047 · FILIC 2509.17053 · 2512.24497 · V-JEPA-2-AC 2506.09985 · 2603.12231 · survey 2606.00113 · swm 2605.21800 | context/precedents as cited inline | UNVERIFIED (single-pass extraction) |
