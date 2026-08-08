# World-encoder: literature review and experiment program

## 1. Token insertion into policies — closed, negative

**Literature.**
- arXiv 2605.21414 (PointACT) — inserting representation tokens alongside a trainable vision stack collapses performance (73.2% → 18.6%); the working pattern is a bottleneck-query action expert. Independent replication of our failure mode.
- arXiv 2409.20248 — frozen pretrained encoders underperform end-to-end vision by −42% on average at 160–1000 demonstrations.
- arXiv 2307.03567 (SpawnNet) — the same frozen features yield 11.7% fed directly, 28.3% via LoRA, 86.7% via multi-layer adapters: the insertion mechanism alone moves the outcome 7×.
- arXiv 2405.12213 (Octo) — readout tokens attend to observations but are never attended back, so an added pathway cannot perturb base processing.
- arXiv 2410.10088 (DiT-Block Policy) — per-layer cross-attention and in-context conditioning unstable; adaLN-Zero with zero-initialized conditioning scale is the robust default (+16% from zero-init alone).
- arXiv 2602.17259 (FRAPPE) — future-alignment prefix tokens with a sharp dose–response optimum (λ = 0.05) and a preference for a single target family over multiple simultaneous streams.
- arXiv 2605.04678 — of four ways to add latent actions to a VLA, direct supervision on discrete placeholder tokens is the most effective; early-layer alignment hurts.

**Our experiments (completed).** Three generations of insertion variants (concatenation, gated pooled token, unpooled token set, state-conditioned extraction) at 200 demonstrations: best insertion arm 0.86 vs 0.90 baseline; no lift on a second benchmark (2 seeds); insertion never beats baseline at any demonstration count. Axis closed, consistent with the literature.

## 2. Low-data and replacement regimes — closed as tested; targeted revival planned

**Literature.**
- arXiv 2212.05749 — the frozen-vs-scratch crossover measured directly: frozen wins marginally at 10 demos, learning-from-scratch wins at 100; no single frozen representation is consistently best.
- arXiv 2203.12601 (R3M) — the canonical frozen-representation win, confined to 5–100 demonstrations.
- arXiv 2406.09246 (OpenVLA) — frozen backbone −22.7 pp on a real robot vs finetuned.
- arXiv 2410.18647 — real-world data scaling: performance saturates near 50 demos per setting; frozen DINOv2 scores 0.00 as a diffusion-policy backbone.
- arXiv 2409.18330 (DMC-VB) — no pretrained representation beats end-to-end BC at full data; at 1% data, frozen-pretrained wins appear, largest under distractors.
- arXiv 2505.11528 (LaDi-WM) — world-model gains in manipulation materialize at 10 demos/task, and operate over dense features rather than compressed sets.
- arXiv 2203.03580 (PVR) and arXiv 2303.18240 (VC-1) — the canonical frozen-encoder evaluation protocols; interface choice (which layer, which pooling) flips encoder rankings.

**Our experiments (completed).** Nested low-N study (10/20/40/100 demos, three arms): the predicted crossover did not appear — the token arm degraded faster than baseline as data shrank; full replacement failed at every count.

**Planned.** Time-boxed replacement-viability study: token-capacity sweep (16/32/64 queries), action-relevant extraction objectives, a parameter-efficient adaptation arm, and compute-to-target-success as the primary metric.

## 3. Distribution-shift robustness — positive anchor; extension planned

**Literature.**
- arXiv 2402.08191 (THE COLOSSEUM) — 30–50% degradation from single-axis perturbations, ≥75% combined; worst axes are distractors, object color, lighting.
- arXiv 2509.12531 — the pretrained-vs-scratch advantage appears only out of distribution; from-scratch encoders win in-distribution and collapse under texture shift.
- arXiv 2510.13626 — policies are most brittle to camera pose and object displacement (63–75 pp drops); the one heavily verified robustness lever is training-data diversity.
- arXiv 2506.13867 (ATK) — task-driven compressed keypoints beat dense RGB under combined shift on a real robot; robustness comes from task-driven selection of what to compress.
- arXiv 2312.12444 — an emergent-segmentation metric (Jaccard of ViT CLS-attention maps) predicts out-of-distribution manipulation success at R² = 0.85, where ImageNet probes predict nothing (R² = 0.07); in-distribution success does not predict out-of-distribution success.
- arXiv 2506.19408 — object-centric representations degrade gracefully under distractor shift where pooled representations collapse (−14 pp vs −62 pp).

**Our experiments (completed).** Five-axis shift battery on frozen-token insertion arms: lighting +15.5 pp (4/4 runs — the single confirmed token-arm win), camera −18.7 pp, distractor negative. Frozen-token insertion buys photometric robustness, not geometric — consistent with the view-specific embedding structure found in our composition analysis.

**Planned.** Full photometric battery (brightness/contrast/color/noise) across 2–3 tasks × 3 seeds; representation-drift mechanism analysis; explicit clean-task cost accounting.

## 4. Planning with compressed latents — the active theme

**Literature.**
- arXiv 2603.05438 (CompACT) — learnable queries over frozen patches with 8 discrete tokens support MPC at near-parity with a 784-token latent at ~37× lower planning latency; never benchmarked against dense patch tokens.
- arXiv 2605.07278 (RC-aux) — "predictive but not plannable": accurate short-horizon latent prediction with unusable planning geometry; a reachability-conditioned auxiliary objective recovers much of it; post-fix, compressed latents split results with dense DINO-WM while planning ~130× cheaper.
- arXiv 2411.04983 (DINO-WM) + released code — the reference harness. Protocol from code: CEM 300 samples × 30 iterations; objective = MSE to goal-image latents over all patch tokens plus proprio; frameskip 5; history 3; goals sampled reachable within ~25 steps; success = 20 px / 20°. The published 0.90 is short-horizon goal-reaching, not the full task. Same-harness encoder ablation: DINO patches 0.90, DINO-CLS 0.44, R3M 0.42, ResNet 0.20 — dense spatial tokens beat pooled embeddings for world-model prediction.
- arXiv 2605.14937 (Slot-MPC) — 4×128 slots plan in 0.42 s vs 144 s for dense patches and win; resolves the apparent contradiction with dense-wins-planning results as protocol-specific.
- arXiv 2410.08822 (SOLD) — slot-latent world models beat monolithic latents on relational manipulation.
- arXiv 2506.09985 (V-JEPA 2-AC) — frozen encoder + action-conditioned latent predictor + CEM over goal-image energy, deployed zero-shot on real arms; ~16 s planning per action underscores planning latency as the practical constraint.
- arXiv 2605.06388 — semantic latents beat reconstruction latents for control, but compressing them distorts control geometry.
- Edge context (arXiv 2603.02271, 2506.10100, 2510.24795) — policy-side token count is not a deployment bottleneck (decode/LLM dominates); planning latency is the supported efficiency argument.
- arXiv 2511.08544 (LeJEPA) — a pretraining objective (isotropic-Gaussian regularization); orthogonal to planning, potentially relevant to future encoder training.

**Our experiments (completed).** On an initial minimal harness: compressed tokens ≈ dense patches (0.100 vs 0.094) at 8–17× lower planning cost; search ×3, longer horizon, observation history, and a 5-model ensemble all flat. An oracle control — the same planner given the true simulator and true state — scored 0.198, attributing the failure to the harness (agent-blind objective, under-provisioned optimizer) rather than any representation; a follow-up factorial confirmed the objective as a real defect and falsified the horizon hypothesis.

**In progress / planned.** Full evaluation on the DINO-WM harness: the released checkpoint reproduces (0.86) on our hardware; a frozen 8-token arm and budget-matched dense control under identical protocol; token-count ablation (1/4/8/16); seed replication; an oracle control inside the reference harness. Efficiency measurements recorded: the 8-token predictor trains 2.8× faster, and precomputed-latent training (verified tensor-identical) reduces an epoch from ~14 h to ~18 min — feasible only for compressed latents (19 GB vs ~865 GB dense).

## 5. Force fusion — closed

**Literature.**
- arXiv 2604.01414 — joint-torque concatenation equals vision-only; contact-gated adaptive fusion is what works (30% → 82%), with input gating carrying most of the gain; unfiltered free-space torque noise identified as the failure source.
- arXiv 2505.13982 (AdapTac) — force-as-query modality gating; fusion without a predictive-force auxiliary loses to vision-only (67% vs 73%); the future-force prediction term alone is worth +20 pp.
- arXiv 2506.15953 (ViTacFormer) — most of the headline gain comes from a prediction head plus a scheduled curriculum, not the fusion mechanism; training destabilizes without the curriculum.
- arXiv 2509.16550 (TranTac) — learned wrist-F/T fusion loses to vision-only on average; transient-signal tokenization is what wins.
- arXiv 2502.17432 (FACTR) — anti-dominance curriculum (corrupt the dominant modality, anneal to clean): vision+force 61% → 87.5%.

**Our experiments (completed).** Rung-0 force fusion: double null on two tasks; a pre-registered late-crossover appeal window closed with the null standing. Consistent with the literature: naive force wiring does not work; any future attempt requires a curriculum.

## 6. Encoder design: conditioning, objectives, and the proprioception shortcut

**Literature — conditioning architectures.**
- arXiv 2301.12597 (BLIP-2) / arXiv 2305.06500 (InstructBLIP) — the privileged signal belongs in the query stream, not among the K/V tokens; removing instruction-query interaction costs up to −7.6 out of distribution.
- arXiv 2204.14198 (Flamingo) — gated cross-attention with zero-initialized tanh gating; removing the gate costs 4 points — the strongest published support for zero-initializing a new conditioning path.
- arXiv 2410.24164 (π0), arXiv 2503.14734 (GR00T N1), arXiv 2410.07864 (RDT-1B) — production systems route state on the query side attending over vision, never as extra K/V among image tokens; RDT ships CFG-style condition dropout in code.
- arXiv 2606.24450 — the closest implementation of state-as-query: joint states as query tokens over frozen spatial features, F1 0.93 vs 0.74–0.78 for pooling/concatenation.

**Literature — the proprioception shortcut.**
- arXiv 1905.11979 (causal confusion) and arXiv 2010.14876 (copycat agents) — the founding results: more observation information can worsen imitation via nuisance correlates.
- arXiv 2506.23944 (NADA) — validation loss actively lies in the shortcut's favor (lower val loss, 3% vs 53% rollout success); state dropout/noise recovers performance.
- arXiv 2602.12032 (GAP) — vision+proprio policies 15.8% worse than vision-only across 8/8 sim and 6/6 real tasks; fix = proprio-gradient scaling early in training.
- arXiv 2509.18644 — removing proprioception entirely can improve generalization (0 → 85–100% under height shift), given relative actions and full visual coverage.
- arXiv 2202.05306 — modality laziness: multimodal networks rely on whichever modality is fastest to learn.

**Literature — pretraining objectives and target format.**
- arXiv 2312.13139 (GR-1), arXiv 2412.15109 (Seer) — future-prediction pretraining with clean ablations isolating the future-prediction term.
- arXiv 2410.22325 (MCR) — dynamics-alignment is the largest ablation term in a manipulation-centric objective.
- arXiv 2412.14803 (VPP) — a fine-tuned-then-frozen future predictor consumed as query tokens via cross-attention, with multi-layer aggregation load-bearing.
- arXiv 2305.16985 — single-step inverse dynamics is the only pretraining objective to substantially beat scratch in a controlled same-encoder comparison.
- arXiv 2505.15659 (FLARE) — future-latent alignment auxiliary that never fell below baseline in any ablation.
- arXiv 2502.03270 — time-invariant per-frame encoders violate the Markov assumption BC needs; a trainable cross-attention query over frozen patches is the cheapest published pooling upgrade.
- arXiv 2010.02193 (DreamerV2), arXiv 2503.00653 (DCWM), arXiv 2505.23705 (π0.5-KI) — discrete/categorical targets for representation learning with continuous heads for control; arXiv 2505.04999 (CLAM) as the counter-case where quantization limits fine-grained control.
- arXiv 2011.10566 (SimSiam), arXiv 2212.03319 — collapse in self-predictive learning is prevented by asymmetry mechanisms, not by the loss.

**Our experiments (completed).** State-conditioned extraction encoder (queries conditioned on robot state, zero-initialized gates): shortcut detectors clean by design contrast, offline probes improved, geometry gap closed offline — but downstream Square 0.54 vs 0.90; the pooled-to-unpooled change alone was not the fix either. Dropout-retrain produced representation-space contraction (failed). An FSQ discrete variant of the encoder trains at near-parity with continuous. The state-blind encoder remains the production choice.

## 7. Evaluation methodology — cross-cutting

**Literature.**
- arXiv 2108.03298 (robomimic study) — checkpoint selection by validation loss picks significantly worse policies; success fluctuates across checkpoints; max-over-training with fixed rollout cadence is the defensible protocol.
- arXiv 2310.09289 — sim and real performance nearly uncorrelated across 15 pretrained representations (R² = 32%).
- arXiv 2304.04591 — encoder rankings flip with the downstream policy-learning method.
- arXiv 2303.04137 / arXiv 2405.07503 / arXiv 2510.21991 — reduced-step diffusion inference is safe on easy tasks and collapses catastrophically on precision tasks; few-step behavior is schedule-sensitive.
- arXiv 2503.10966 (STEP) / arXiv 2405.05439 — sequential evaluation with exact Type-I control; UMA confidence bounds for success rates at 40–50 rollouts.
- arXiv 1502.07943 / 1603.06560 / 1810.05934 (Successive Halving, Hyperband, ASHA) and arXiv 2404.04111 — early-discarding theory; one-epoch discarding often loses nothing, with a short-horizon bias caveat.

**Our findings (completed).**
- A planning failure cannot be attributed to the representation without a control. Our initial harness contained design errors, yet from the inside was indistinguishable from a genuine negative result — dynamics beat carry-forward, probes read R² 0.96, a measured exploit gap matched a published failure mechanism, and five remediation experiments were null. An oracle-dynamics control (0.198) reversed the attribution and established that the full task is beyond short-horizon MPC regardless of model quality. Where a validated reference harness exists, baseline reproduction serves the same role; the reviewed papers report neither control.
- Offline probes do not gate downstream success (high probe R² with planning failure; three internal generations plus external corroboration).
- Early-epoch screens cannot rank arms (rank inversions between early and decisive reads; absolute ceilings under-read at intermediate epochs).
- All decision rules (gates, kill rules, contingency branches, prediction forks) are registered before the corresponding numbers exist.
