# Proprio/state shortcut — prevention and detection when state becomes the privileged conditioning signal

**Date:** 2026-07-30. **Question:** v0.4 makes proprio state condition the PerceiverFuse queries that extract
from vision. How do we prevent, and cheaply detect, the known failure where the policy/encoder collapses onto
state-predictable content and ignores vision? **Companion:** REDESIGN_RESEARCH_20260729.md (does not repeat
it; DiT-Block per-dim dropout and FACTR curriculum are extended here with exact numbers from code/paper).

**Method:** WebSearch + WebFetch; every load-bearing claim checked against the fetched source (paper full
text, extracted PDF text, or the project's GitHub code). Two fetches disagreed on D3P — resolved by extracting
the PDF text directly; the "simple averaging, ~82/88%" version was a fetch hallucination, corrected below.
Unfetched claims are marked UNVERIFIED.

## Executive verdict

1. **The shortcut is real, large, and specifically a BC-with-proprio phenomenon.** Cleanest evidence: on
   robomimic Square, BC with vision+proprio gets LOWER validation loss than vision-only (2.94e-2 vs 4.13e-2)
   yet 3.0±3.0% vs 52.8±4.0% rollout success — the shortcut fits the data better and fails at deployment
   (NADA, arXiv 2506.23944). Naive vision+proprio concatenation loses to vision-only on 8/8 GAP benchmark
   tasks (GAP, arXiv 2602.12032, ICLR 2026) and collapses 61.7%→33.3% under randomized initial joint states
   (D3P, arXiv 2511.00555). Removing state entirely takes height generalization 0%→85-100% (State-free
   Policy, arXiv 2509.18644).
2. **Simple mitigations recover most of it; every recipe front-loads suppression of the state pathway.**
   Whole-state dropout p=0.8: 54.6%→74.1% avg (9 tasks); tuned Gaussian state-noise (NADA): 76.1%; GAP
   gradient scaling (λ=0.3, first 50% of epochs): beats vision-only on 8/8. DiT-Block's per-dim proprio
   dropout is `nn.Dropout(p=0.2)` on the raw state vector before tokenization (verified in code). RDT-1B's
   production recipe is CFG-style condition dropout: with p=0.1, replace the whole state with the
   dataset-mean state ("null token"), independently per modality (verified in code).
3. **For the ENCODER side (our new risk), the transferable fix is classifier-free-guidance-style condition
   dropout + zero-init conditioning gates + keeping some queries unconditioned.** CFG's own ablation:
   p_uncond ∈ {0.1, 0.2} equally good, 0.5 consistently worse (arXiv 2207.12598). The analogous
   cross-attention failure in text-to-image diffusion ("catastrophic neglect") is detected via per-token
   cross-attention maps and fixed by forcing attention mass onto neglected condition tokens
   (Attend-and-Excite, arXiv 2301.13826) — the same attention-mass metric runs free on our Perceiver.
4. **Detection is cheap and mostly offline.** Three diagnostics below need no new training runs: (a)
   null-state / null-vision action-MSE sensitivity on held-out demos, (b) a state→z predictability probe on
   the 8 query outputs (the copycat literature's adversary IS this probe, arXiv/NeurIPS 2020), (c)
   attention-mass-on-vision + state-perturbation attention drift. Plus two training-time signatures: the
   val-loss-vs-rollout inversion (NADA) and early per-modality loss/gradient domination (GAP).

---

## 1. The failure mode in the BC literature — papers, mechanisms, effect sizes

### 1.1 NADA / "Adapt Your Body" (arXiv 2506.23944) — the val-loss inversion + tuned state noise

- **Shortcut evidence (verified):** "policies trained with proprioception sometimes achieve lower prediction
  errors on both training and validation datasets, suggesting the model finds a shortcut between
  proprioceptive states and target actions." Square: BC-Full val loss 2.94e-2 < BC-RGB 4.13e-2, but rollout
  success 3.0±3.0% vs 52.8±4.0%. **This is the key diagnostic fact: you cannot see the shortcut in the loss;
  the loss actively lies in the shortcut's favor.**
- **9-task simulation averages (verified):** BC-Full 54.6%, BC-RGB (vision-only) 73.2%, Random Dropout 74.1%,
  NADA 76.1%. Tasks: NutAssemblySquare, PickPlaceCan, Coffee, HammerCleanup, MugCleanup, Stack, StackThree,
  Threading, ThreePieceAssembly (robomimic/MimicGen family — directly adjacent to our RoboCasa setting).
- **The dropout baseline recipe (verified):** "Randomly mask proprioception input to zeros with probability
  80% during training." Per-task: Square 44.8±5.6% (vs 3.0% BC-Full), Can 90.8±1.6% (vs 51.3%), Threading
  92.1±0.5% (vs 73.5%). NADA beats this on 6 of 9 tasks — so p=0.8 whole-vector masking captures most of the
  available lift on its own.
- **NADA's own recipe (verified):** train-time-only Gaussian noise o+ε, ε~N(0, σ*²I), added independently to
  each proprio group (EE pose/vel, gripper pos/vel, joint pos/vel). σ* is selected by minimizing the
  time-conditioned Wasserstein distance between noised-training and policy-rollout state distributions
  (σ search increment δ=0.05, K=10 trajectory intervals, 150 trajectories per side; batch 64, 75k steps,
  lr 2e-4). No noise at eval.
- **Detection metric (verified):** time-conditioned Wasserstein distance between training and rollout proprio
  distributions; Square: 1.05 (BC-Full) → 0.23 (NADA). Needs rollouts, so it is our confirmatory (not early)
  metric.

### 1.2 GAP (arXiv 2602.12032, ICLR 2026) — mechanism diagnosis + gradient modulation

- **Mechanism (verified):** "the policy naturally gravitates toward concise proprioceptive signals that offer
  faster loss reduction when training, thereby dominating the optimization and suppressing the learning of
  the visual modality during motion-transition phases." The failure is phase-localized: sub-phases where
  motion changes and vision must localize the target. Headline: "Vision-Proprioception policies perform
  15.8% worse than Vision-only" (Fig. 1 caption).
- **Vision-only beats naive concat on 8/8 sim tasks (verified, Table 1, 5 seeds):** e.g. Meta-World
  pick-place 91.8±1.5 vs 78.4±1.5; bin-picking 63.2±1.5 vs 48.8±2.2; RoboSuite threading 43.6±1.7 vs
  33.2±1.9. Real robot: vision-only ≥ concat on 6/6 tasks (e.g. "lift lid and pour" 9/20 vs 5/20).
- **The GAP recipe (verified):** scale the gradient of the proprio feature extractor by (1−ρ)·λ where ρ is a
  per-timestep motion-transition-phase indicator: ω_s ← ω_s − λ·(1−ρ)·η∇L_BC. Hyperparameters: λ=0.3
  (insensitive in 0.2–0.4; λ=0.1 too weak ≈ concat baseline; λ=0.8 unstable/collapse); applied **only during
  the first 50 of 100 epochs** ("if the gradient updates for proprioception are suppressed for too many
  epochs, the performance of policies deteriorates"); phases found by Change Point Detection on the proprio
  trajectory (motion distance with α=1 spatial/rotation, β=2e-3 gripper-binary) + an LSTM that soft-estimates
  ρ. For transformer policies (their Octo-VP arm), adjustment is applied "only ... to the proprioception
  feature extractor (i.e., the parameters before the transformer)".
- **A fixed ρ nearly works (verified, Table 7):** fixed ρ=0.5 gets 89/48/75% vs full GAP 91/52/85% (hammer /
  threading / cube) — i.e., a constant 0.5·λ gradient attenuation on the state encoder for the first half of
  training captures most of the effect without any phase machinery.
- **GAP beats all baselines (verified, Table 1):** GAP 94.2/94.2/91.4/73.6/70.4/91.0/77.6/53.0 across the 8
  sim tasks vs Mask (RDT-style proprio-only masking, fixed probability — value not stated in the paper)
  86.4/89.2/83.8/79.2/60.4/78.6/61.8/47.2. Note Mask wins one task (push-wall 79.2 vs 73.6). Aux-loss
  (HumanPlus-style next-frame prediction from visual features) UNDERPERFORMS on precision tasks (threading
  45.4 vs vision-only 43.6 vs GAP 53.0) — forcing vision all the time is also wrong; proprio is genuinely
  useful in motion-consistent phases.
- **Training-time signature (verified, Fig. 6):** with GAP "the loss decreases more slowly during the early
  and middle stages of training. However, the policy ultimately converges to a lower loss." Fast early loss
  = the shortcut fitting; a healthy vision-grounded fit is slower but lower.

### 1.3 State-free Policy (arXiv 2509.18644) — the ceiling of removal + why action space matters

- **Effect sizes (verified, v2 full text):** Pick Pen trained at 80cm table height: state-based 0/30 at both
  72cm and 90cm; state-free 100%/97%. Across three pick-place tasks, height generalization 0→0.93-0.98,
  horizontal 0→0.58-0.70. Fold Shirt 0.183→0.834; whole-body Fetch Bottle 0.117→0.784. In-distribution
  performance is preserved (0.967–1.0).
- **Mitigation comparison on the state arm (verified, Tables I/VIII):** state noise augmentation ±5cm gets
  80% height gen on Pick Pen (vs 0% plain state, vs 93-100% state-free); across the wider Table VIII,
  noise 63.3%, diverse-height data 11.7%, task-mixed or LoRA fine-tuning 0%. **State noising recovers much
  but not all of the removal ceiling.**
- **Preconditions (verified, Tables III/IV):** the state-free result only holds with **relative EE actions**
  (absolute EE, relative joint, absolute joint all → 0% generalization) and **full task-relevant visual
  coverage** (dual wide-angle wrist cams 0.983/0.583 vs overhead-only 0.217/0.133). For us: RoboCasa runs
  delta-EE control and 3 static cams — the static-cam coverage assumption is the weak link; total state
  removal is NOT the right v0.4 move, regularized state is.

### 1.4 D3P (arXiv 2511.00555) — recovery behavior + a per-branch runtime diagnostic

- **Shortcut evidence (verified from PDF text):** on RLBench-style tasks with random initial joint positions,
  DP* (vision+proprio) drops 61.7%→33.3% avg while vision-only DP− holds 58.3%→55.8%; qualitatively, the
  proprio policy "struggles to recover from out-of-distribution joint states, often getting stuck in
  oscillatory behavior", while the vision-only policy "exhibits unexpected failure recovery capability".
  They frame this via Geirhos et al.'s shortcut-learning account (their citation [12]).
- **Design (verified):** dual-branch diffusion policy — one branch conditions on enhanced vision only, one on
  fused vision+proprio; both trained jointly; at inference the overlapping action chunks are aggregated
  using the **DDPM test-time loss as a confidence signal** with temporal weights η=0.97 and a Savitzky-Golay
  smoothing filter. Result: 76.3% fixed-init / 75.8% random-init — the proprio benefit without the OOD
  collapse. (A second WebFetch of this paper returned fabricated numbers/mechanism; corrected against the
  extracted PDF.)
- **Takeaway for us:** maintaining a state-free forward path at all times (their vision-only branch; our
  CFG-null path, §3) both regularizes training and provides a per-timestep runtime shortcut detector — the
  divergence between the two branches' test-time losses localizes exactly where the policy leans on state.

### 1.5 Lineage and scope notes

- **Causal confusion (arXiv 1905.11979, NeurIPS 2019):** the founding statement that MORE observation
  information can yield WORSE imitation because of nuisance correlates; their fix (training over random
  input-mask subgraphs, then intervening to select the mask) is the ancestor of every dropout/masking recipe
  here. [verified at abstract level]
- **Copycat problem (Wen et al., NeurIPS 2020, proceedings.neurips.cc/paper/2020/hash/1b113258af…):** in
  observation-history BC the policy predicts a_{t−1} instead of a_t. Fix (verified from PDF): min-max
  objective V(E,F,D) = E[L(F(e_t),a_t)] + λ·KL(p_E(e_t)‖N(0,I)) − α·L(D(µ_t, a_t), a_{t−1}) — a
  **target-conditioned adversary** that strips a_{t−1} information not shared with a_t, plus an
  **information bottleneck** (KL to unit normal; encoder outputs µ,σ; adversary sees µ, policy sees the
  noisy sample). Baseline lineage: "Dropout-BC" (Bansal et al./ChauffeurNet) = dropout on the confusing
  input subset. Our proprio-conditioned encoder has the identical structure: state is recoverable, actions
  are state-correlated, so z can carry state and nothing else.
- **BC-IB (arXiv 2502.02853):** information bottleneck on the fused latent between (vision ⊕ state ⊕
  language) features and the policy head: L = E[β·I(x_t,z_t) + ‖π(x_t)−a_t‖²], β ∈ [1e-4, 1e-2], β≈1e-4
  "stable improvements". Gains: +7.66 to +10.83pp on LIBERO-Goal/Object variants, 2-10% on CortexBench.
  They explicitly note "previous work has shown that proprioceptive states can lead to overfitting" and keep
  state in but compressed. [verified]
- **Generalist-policy shortcut study (arXiv 2508.06426):** shortcut = reliance on task-irrelevant factors;
  studies viewpoint/background/texture, and **explicitly defers proprioception to future work** — so there is
  no dataset-diversity result for proprio shortcuts yet; the mitigation burden stays on
  regularization/architecture at our scale. [verified]
- **Octo (arXiv 2405.12213):** the base model takes no proprio input; the html v2 text contains **no stated
  causal-confusion rationale** for that choice (folk claims that it does are not supported by the fetched
  text). [verified-negative]

---

## 2. Mitigation recipes — exact numbers, verified

| Mitigation | Exact recipe | Effect size | Source (verified) |
|---|---|---|---|
| Per-dim state dropout | `nn.Dropout(p=0.2)` on raw proprio vector, before the obs tokenizer, train-time only (elementwise Bernoulli, 1/(1−p) scaling); constant, no schedule | part of a "combination of tricks" worth ~40% on long-horizon bimanual tasks (not individually ablated) | DiT-Block Policy paper (arXiv 2410.10088) + `agent.py` in github.com/sudeepdasari/dit-policy (`use_obs="add_token"` path) |
| Whole-state dropout | zero the entire proprio input with p=0.8 per sample | 9-task avg 54.6→74.1%; Square 3.0→44.8% | NADA baseline, arXiv 2506.23944 |
| State noising (tuned) | ε~N(0,σ*²I) per proprio group, train only; σ* by Wasserstein-matching train↔rollout state distributions | 9-task avg 76.1% (best); beats p=0.8 dropout on 6/9 | NADA, arXiv 2506.23944 |
| State noising (SNR knob) | `state_noise_snr` (dB): ε~N(0, state_std/√(10^(SNR/10))) | no published ablation; production flag | RDT-1B code, `train/dataset.py` |
| Condition dropout (CFG-style) | with p=0.1 per sample, replace whole state vector with **dataset-mean state** (and zero the state-element mask); independently mask each camera (replace with background-color image), language, ctrl-freq | no isolated ablation in paper; the GAP-adapted proprio-only version recovers concat→above-concat but < GAP | RDT-1B, arXiv 2410.07864 + `train/dataset.py` (`cond_mask_prob=0.1` default, verified in code) |
| Condition-dropout rate | p_uncond ∈ {0.1, 0.2} equal-best; 0.5 "consistently performs worse … across the entire IS/FID frontier"; drop = null token ∅ | FID 1.55 (p=0.1) vs 1.91 (p=0.5) at w=0.1 | Classifier-Free Guidance, arXiv 2207.12598 |
| Gradient modulation | scale state-encoder grads by λ·(1−ρ), λ=0.3 (0.2–0.4 fine, 0.8 collapses), **first 50% of epochs only**; ρ from CPD+LSTM phase estimate; fixed ρ=0.5 ≈ almost as good | vision+proprio goes from losing 8/8 vs vision-only to winning 8/8; e.g. threading 33.2→53.0 | GAP, arXiv 2602.12032 |
| Anti-dominance curriculum | corrupt the DOMINANT modality early, anneal to clean: σ_n = σ_0(1−n/N) linear or (σ_0/2)(1+cos(nπ/N)) cosine, optional warm-up hold at σ_0; Gaussian blur or maxpool-downsample, pixel or latent space | vision+force 61.2%→87.5% unseen-object success (vision-only 21.3%); schedules within 17-20/25 of each other | FACTR, arXiv 2502.17432 (they corrupt vision to rescue force; §3 for the direction that transfers to us) |
| Information bottleneck | β·I(x;z) on fused latent, β=1e-4…1e-2 (start 1e-4) | +7.7 to +10.8pp LIBERO-Goal/Object | BC-IB, arXiv 2502.02853 |
| Adversarial (targeted) | target-conditioned adversary predicts the nuisance (for us: state) from z, conditioned on a_t; encoder maximizes its error; + KL(z‖N(0,I)) bottleneck | "improves performance significantly" across 6 continuous-control settings (exact deltas not extracted) | Copycat, NeurIPS 2020 |
| Dual path | separate state-free branch + fused branch; aggregate action chunks by DDPM test-time loss (η=0.97 temporal weights, SG filter) | 33.3→75.8% under random init; 61.7→76.3% fixed | D3P, arXiv 2511.00555 |
| Removal (ceiling reference) | no state at all; requires relative-EE actions + full visual coverage (wrist cams) | height gen 0→85-100%; but 0% if absolute actions or poor coverage | State-free Policy, arXiv 2509.18644 |

**Cross-paper regularities worth designing around:**
1. **Front-load the suppression.** GAP applies modulation only in the first half of training and shows late
   suppression hurts; FACTR's curriculum is maximal corruption at step 0 annealing to clean; the shortcut is
   established early (fastest-loss-reduction mechanism), so steady-state regularization can be mild if early
   regularization is strong.
2. **Whole-input dropout rates for a *policy* input are high (0.8); condition-dropout rates for a
   *conditioning* signal are low (0.1–0.2).** These are different mechanisms: p=0.8 dropout makes state
   unavailable most of the time (approaching removal, with state as occasional bonus); p=0.1 condition
   dropout keeps state useful but guarantees a functioning unconditional path. For a signal we WANT the
   encoder to use (v0.4's premise), the CFG-style low rate plus per-dim dropout is the right combination;
   p=0.8 is the fallback if diagnostics still show collapse.
3. **Vision-side aux losses are not a substitute** (GAP's Aux baseline loses on precision tasks), and total
   suppression of state hurts where proprio genuinely carries signal (threading-type tasks). The goal is
   collaboration, not amputation.

---

## 3. The encoder-side subtlety: conditioning collapse in cross-attention

Our risk differs from the policy-side literature above: in v0.4 the state conditions the **queries** of the
PerceiverFuse cross-attention. The failure is not "policy ignores vision" but "state-conditioned queries only
extract visual content that is predictable from state" (e.g., arm-centric patches, table plane) — the
downstream policy then sees a z that is an expensive re-encoding of state. Nothing in the robot-BC literature
tests this exact configuration; the closest analog with a mature fix is condition handling in conditional
diffusion models.

**What transfers:**
- **Condition dropout (the CFG mechanism).** Train the encoder with the state condition randomly replaced by
  a null embedding with p=0.1–0.2 (verified optimal range for CFG, arXiv 2207.12598; RDT-1B independently
  converged on p=0.1 with dataset-mean-state as the null, verified in code). Effect on our architecture: the
  8 queries must produce a useful z with NO state 10-20% of the time, so the vision→z extraction path can
  never degenerate into a state readout. Bonus: this gives a well-defined "null-state" input for the
  detection protocol (§4) and, optionally, CFG-style extrapolation between conditioned and unconditioned
  extraction at inference (untested in this setting — do not enable by default; a search-level note is that
  Pearce et al., arXiv 2301.10677, found guidance can bias sampling toward low-likelihood actions in BC —
  UNVERIFIED, abstract-level only).
- **Catastrophic neglect + attention-mass detection.** In text-to-image diffusion, the model "fails to
  generate one or more of the subjects from the input prompt" and the standard detection/fix operates on
  per-token cross-attention maps, strengthening ("exciting") the attention activations of neglected condition
  tokens (Attend-and-Excite, arXiv 2301.13826, verified at abstract level; the specific
  max-attention-per-token objective detail: UNVERIFIED). Transfer: per-query attention mass over visual K/V
  is the encoder-side health metric, and a training-time floor on vision-attention mass is a possible (novel
  here) intervention if dropout alone fails.
- **Zero-init conditioning gates.** From our verified prior corpus (REDESIGN_RESEARCH_20260729.md §4):
  adaLN-Zero's zero-initialized conditioning scale alone was worth 16% in DiT-Block, and Octo's readout
  isolation shows added pathways should start at zero contribution. Applied here: inject state into the
  queries through a gate initialized to zero (query_i = learned_query_i + gate·f(state), gate=0 at init), so
  the earliest training epochs — precisely when GAP shows the shortcut forms — are purely vision-driven, and
  state influence grows only as it earns gradient.
- **Keep an unconditioned query subset.** Split the 8 queries: 4 state-conditioned, 4 plain learned queries
  (never see state). Analog of Octo's isolated readout tokens and FRAPPE's dedicated prefix tokens (both
  verified in the prior report): the unconditioned subset guarantees a state-independent visual channel into
  the policy regardless of what the conditioned subset collapses to, and the CONTRAST between the two
  subsets is itself a diagnostic (state→z probe R² should differ sharply; if it doesn't, state information
  is leaking through the shared cross-attention — expected and fine — but if the unconditioned tokens'
  vision-attention also collapses, the problem is upstream of conditioning).
- **The FACTR direction, mapped correctly.** FACTR corrupts the modality being over-relied on (vision) to
  force the starved modality (force) to receive gradient. In v0.4 the over-relied modality is STATE; the
  transfer is therefore a **state-corruption curriculum**: start with heavy state noise/high condition-drop
  probability and anneal down to the steady-state values (cosine or linear over the first ~50% of training,
  matching GAP's early-window finding). Concretely: p_cond-drop 0.5→0.1 and/or σ_state σ_0→σ* on FACTR's
  cosine schedule.

---

## 4. Detection protocol — cheap, early, mostly offline

Ordered by cost. D1-D3 need only a checkpoint + held-out demos (minutes on one GPU); T1-T2 are free
training-time monitors. Thresholds are proposals calibrated against the effect sizes above, not published
numbers.

### T1. Val-loss-vs-rollout inversion watch (free, continuous)
The shortcut's signature is a *lower* action-prediction validation loss than a vision-reliant policy would
get (NADA: 2.94e-2 vs 4.13e-2 while rollout success is 3% vs 53%). Concretely: track val action-MSE of the
v0.4 arm against the v0.3 (state-as-policy-input or no-state) baseline arm. **A suspiciously large val-loss
improvement (>20-30% relative) early in training is a red flag, not a win**, until D1/D2 clear it.
[NADA, arXiv 2506.23944]

### T2. Early per-modality gradient/loss domination (free, first epochs)
GAP's mechanism: the proprio pathway wins the early loss-reduction race. Log per-pathway gradient norms
(state-conditioning MLP + query gate vs visual K/V projections) and, if the gate is scalar, the gate
magnitude trajectory. State-pathway gradient norm persistently ≥ vision-pathway norm during epochs 1-10,
or a gate that saturates within the first few epochs, = the GAP failure signature. This lines up with our
ep50-band early-kill discipline: T2 is readable well before ep50. [GAP, arXiv 2602.12032]

### D1. Null-state / null-vision sensitivity sweep (offline action-MSE; the primary gate)
On N≈100 held-out demos, compute policy action-MSE (and encoder z) under four input conditions:
1. intact;
2. **null state** — dataset-mean state (RDT's null convention; exactly in-distribution if we train with
   condition dropout, which is a further reason to adopt it);
3. **permuted state** — state drawn from a random other timestep/episode (distinguishes "uses state
   magnitude" from "uses state identity"; permutation keeps marginals intact);
4. **null vision** — all three cams replaced by the background-color image (RDT's image-null convention).

Read-out: ΔMSE_vision = MSE(null-vision) − MSE(intact); ΔMSE_state analogous.
- **Shortcut verdict:** ΔMSE_vision ≈ 0 while ΔMSE_state is large — the policy is state-driven. Per D3P/GAP
  this is exactly the configuration that collapses under OOD init (61.7→33.3%).
- **Healthy target:** both deltas materially > 0, with ΔMSE_vision dominating during motion-transition
  segments. If GAP-style CPD phase labels are computed (cheap: change-point detection on the demo proprio
  stream), report the deltas per phase — vision sensitivity concentrated in transition phases is the
  pattern GAP identifies as correct collaboration.
- Confirmatory (rollout) version when a run graduates: 20-50 RoboCasa episodes with state frozen to the
  dataset mean at eval vs intact — success-rate delta replaces MSE. GAP's intervention experiments
  (policy-switching by phase) and the State-free spatial-shift evals (train-height vs ±10cm) are the
  published precedents; a ±(5-10)cm object/base placement shift split is the cheapest RoboCasa analog.
  [RDT code; D3P arXiv 2511.00555; GAP arXiv 2602.12032; State-free arXiv 2509.18644]

### D2. State→z predictability probe (encoder-side collapse; the v0.4-specific gate)
Regress from proprio state (current + short history, e.g. 5 steps) to each of the 8 query outputs of the
encoder: ridge regression + 2-layer MLP probe, per-token R². Complement with the inverse-side probe: z →
object pose (information that exists ONLY in vision).
- **Collapse verdict:** state→z R² ≳ 0.9 on the state-conditioned tokens AND z→object-pose R² near the
  state-only baseline (probe from raw state to object pose gives the floor — note in RoboCasa fixed-scene
  episodes state can weakly predict object pose through trajectory correlation, so the floor must be
  measured, not assumed zero).
- **Healthy:** conditioned tokens carry state (moderate R² — expected and desirable) but z→object-pose
  clears the state-only floor by a wide margin, and the 4 unconditioned tokens sit well below the
  conditioned ones on state→z.
- Precedent: this probe is the copycat adversary D(µ_t) → a_{t−1} run as a diagnostic instead of a training
  signal (Wen et al., NeurIPS 2020, verified); their Table 1-style "MSE of predicting the nuisance from the
  latent" is the published template. Runs on cached features in minutes; can be evaluated at every
  checkpoint from epoch 1.

### D3. Attention-mass-on-vision + attention drift under state perturbation
Two numbers from the PerceiverFuse cross-attention, free at any checkpoint:
1. **Vision-attention mass/entropy per query:** if state tokens (or any non-visual tokens) share the K/V,
   the fraction of attention mass on visual tokens; with vision-only K/V, the spatial entropy and the
   overlap of thresholded attention maps with task-object regions. Published precedents for
   attention-map-based health metrics: catastrophic-neglect detection via per-token cross-attention maps
   (Attend-and-Excite, arXiv 2301.13826) and, already verified in our prior report with exact definitions,
   Burns attention-Jaccard (R²=0.85 vs OOD success) and MCR manipulation-centricity (Grad-CAM ∩ SAM-2 masks
   of EE + task objects, R=0.93) — both computable on our checkpoints today.
2. **Attention drift under state shuffle:** recompute the attention maps with within-batch-shuffled state;
   the KL/JS between intact and shuffled attention maps, per query. **Near-zero drift on the conditioned
   queries means conditioning is dead (a different failure — state ignored); very large drift with the
   resulting maps collapsing onto proprioceptively-determined regions (gripper, arm) means the queries
   fetch only state-predictable content.** The healthy middle: state shifts WHERE queries look (e.g., toward
   the target object for the current phase) while maps remain object-anchored under D3's metric 1.

**Protocol placement:** T1/T2 continuously; D1-D3 at first checkpoint (~ep10), at the ep50 kill-gate, and at
graduation. Kill criteria consistent with our early-kill discipline: D1 shortcut verdict + D2 collapse verdict
at ep50 = kill the arm (the literature shows no case of late self-correction; GAP shows the opposite — late
training entrenches whichever pathway won early).

---

## 5. The v0.4 recipe (concrete, each item cited)

Steady-state configuration for the state-conditioned PerceiverFuse encoder + DP policy:

```python
# ---- per training sample ----
s = proprio_state                                  # raw state vector, normalized

# (1) per-dim dropout, DiT-Block recipe [2410.10088 + dit-policy code: nn.Dropout(p=0.2) pre-tokenizer]
s = F.dropout(s, p=0.2, training=True)

# (2) Gaussian state noise, NADA recipe [2506.23944]: sigma* per proprio group via
#     Wasserstein matching of train vs rollout state distributions (search step 0.05);
#     bootstrap value before any rollouts exist: RDT's SNR form [RDT code], e.g. 20dB
s = s + sigma_star * torch.randn_like(s)           # train only, never at eval

# (3) CFG-style condition dropout [2207.12598: p in {0.1,0.2}; RDT code: p=0.1, mean-state null]
if torch.rand(()) < p_cond:                        # p_cond annealed 0.5 -> 0.1 (see schedule)
    cond = null_state_embed                        # learned null token (RDT alt: dataset-mean state)
else:
    cond = state_mlp(s)

# (4) zero-init gated injection into HALF the queries
#     [gate: adaLN-Zero evidence, prior report; isolation: Octo readouts / FRAPPE prefix, prior report]
q_cond   = learned_queries[:4] + self.gate * film(cond)   # self.gate init = 0
q_uncond = learned_queries[4:]                             # never see state
queries  = torch.cat([q_cond, q_uncond], dim=0)            # -> PerceiverFuse cross-attn over ViT tokens
```

Schedules and options:
- **Anneal p_cond 0.5 → 0.1 over the first 50% of training** (cosine), constant 0.1 after. Direction from
  FACTR (corrupt the over-relied modality hard early, anneal down: σ_n=(σ_0/2)(1+cos(nπ/N)), warm-up hold
  supported) [2502.17432]; window from GAP (intervene in the first 50 of 100 epochs only; longer hurts)
  [2602.12032].
- **Optional heavier lever if D1/D2 fail at ep50:** GAP-style constant gradient attenuation — multiply
  state-pathway (state_mlp + film + gate) gradients by λ·(1−ρ) with λ=0.3, fixed ρ=0.5, first half of
  training only (fixed-ρ ablation: 89/48/75 vs full GAP 91/52/85) [2602.12032]. This composes with (1)-(4)
  and needs no phase estimation.
- **Fallback if collapse persists:** whole-state dropout p=0.8 (NADA baseline: recovers 54.6→74.1 avg)
  [2506.23944], or the D3P dual-path consumption — policy takes both {z_cond, z_uncond-only} and action
  chunks are blended by test-time DDPM loss (η=0.97) [2511.00555]. The 4/4 query split above already gives
  us z_uncond for free.
- **Heavier-weight options, only with a demonstrated need:** BC-IB β=1e-4 on the fused latent (+7.7-10.8pp
  on LIBERO variants) [2502.02853]; copycat-style target-conditioned adversary predicting s from z
  conditioned on a_t [NeurIPS 2020].
- **What NOT to do:** don't remove state (State-free's preconditions — wrist cams, relative-EE — are not our
  setup; and threading-class precision tasks need proprio [2509.18644, 2602.12032]); don't rely on a
  vision-side aux loss to fix modality balance (GAP's Aux baseline) [2602.12032]; don't judge the arm by
  val loss (NADA inversion) [2506.23944]; don't apply suppression for the whole run (GAP Table 11)
  [2602.12032].

Detection protocol (from §4): T1+T2 logged continuously; D1 (null/permuted-state and null-vision MSE sweep),
D2 (state→z probe R², both query subsets, vs state→object-pose floor), D3 (vision-attention mass + shuffle
drift) at ep10 / ep50-gate / graduation; NADA-Wasserstein and frozen-state rollouts as confirmatory once an
arm earns rollouts.

---

## Sources

| # | Paper / artifact | ID / URL | What was verified against the fetch |
|---|---|---|---|
| 1 | NADA "Adapt Your Body: Mitigating Proprioception Shifts in IL" | arXiv 2506.23944 (html v1) | val-loss inversion numbers; p=0.8 dropout baseline + per-task/avg results; σ* Wasserstein recipe; train-only noise |
| 2 | GAP "When would Vision-Proprioception Policies Fail…" (ICLR 2026) | arXiv 2602.12032 (PDF, text-extracted) | mechanism quote; Eq. 5 + λ=0.3 (0.2-0.4) ablation; first-50-epochs window; fixed-ρ ablation; Table 1 (8 sim + 6 real tasks); Mask/Aux/MS-Bot baselines; loss-curve figure |
| 3 | State-free Policy "Do You Need Proprioceptive States…" | arXiv 2509.18644 (html v2) | Tables I/III/IV/VIII/IX + Fig. 5 numbers; noise-augmentation arm; action-space and camera preconditions |
| 4 | D3P "Deep Koopman-Boosted Dual-branch Diffusion Policy" | arXiv 2511.00555 (PDF, text-extracted) | Table I row order + 61.7/33.3, 58.3/55.8, 76.3/75.8; test-time-loss aggregation, η=0.97, SG filter; oscillation/recovery quotes; one WebFetch of this PDF hallucinated — corrected |
| 5 | Shortcut Learning in Generalist Robot Policies | arXiv 2508.06426 (html v1) | shortcut definition; proprio explicitly future work; diversity-mitigation effect sizes |
| 6 | Causal Confusion in Imitation Learning | arXiv 1905.11979 | abstract-level: more information → worse imitation; graph-mask + intervention approach |
| 7 | Fighting Copycat Agents in BC from Observation Histories (NeurIPS 2020) | proceedings.neurips.cc …1b113258af…Paper.pdf (text-extracted) | TCA + IB objective verbatim; Dropout-BC baseline lineage; architecture (E/F/D, D sees µ) |
| 8 | DiT-Block Policy | arXiv 2410.10088 (PDF, text-extracted) + github.com/sudeepdasari/dit-policy | "per-dimension observation dropout before tokenization" (paper); `nn.Dropout(p=0.2)` on obs in `data4robotics/agent.py` (code); no schedule; no isolated ablation |
| 9 | RDT-1B | arXiv 2410.07864 + github.com/thu-ml/RoboticsDiffusionTransformer | independent per-modality masking motivation (paper); `cond_mask_prob=0.1` default, mean-state null, background-image null, `state_noise_snr` (code, `train/dataset.py`, `main.py`) |
| 10 | Classifier-Free Diffusion Guidance | arXiv 2207.12598 (ar5iv) | p_uncond ∈ {0.1,0.2,0.5} ablation, 0.5 worse quote, null-token mechanism |
| 11 | FACTR (force curriculum) | arXiv 2502.17432 (html v2) | corruption operators + all four schedule formulas + warm-up; 87.5 vs 61.2 vs 21.3%; scheduler ablation table; near-zero-force explanation |
| 12 | Attend-and-Excite | arXiv 2301.13826 (abstract) | catastrophic-neglect definition; attention-map detection + excitation fix (abstract level; per-token max-attention detail UNVERIFIED) |
| 13 | BC-IB "Rethinking Latent Redundancy in BC" | arXiv 2502.02853 (html v5) | IB placement + loss + β range; LIBERO/CortexBench effect sizes; proprio-overfitting acknowledgment |
| 14 | Octo | arXiv 2405.12213 (html v2) | verified-negative: no proprio in base model inputs discussed, no causal-confusion rationale in text |
| 15 | Pearce et al., Imitating Human Behaviour with Diffusion Models | arXiv 2301.10677 | UNVERIFIED (search snippet only): guidance can bias BC sampling toward low-likelihood trajectories |

**Caveats.** GAP's Mask-baseline probability is unstated in the paper (RDT's code default 0.1 is the natural
assumption, marked as such). DiT-Block's p=0.2 comes from the released code, not a paper table, and has no
isolated ablation. NADA/GAP/D3P are 2025-26 preprints or freshly accepted (GAP: ICLR 2026); effect sizes are
sim-heavy (robomimic/MimicGen, Meta-World/RoboSuite, RLBench-style) — directionally adjacent to RoboCasa but
not calibrated on it. The encoder-side transfer (§3) is an argued analogy: no fetched paper tests condition
dropout on a state-conditioned visual-extraction Perceiver; the components (CFG dropout rate, zero-init
gates, readout isolation, attention-mass metrics) are individually verified, their composition here is ours.
D1-D3 thresholds are proposals, not published numbers.
