# v0.4 research 03 — Which predictive objective(s) pair with state-conditioned extraction

**Date:** 2026-07-30. **Question:** v0.4 makes robot state the privileged cross-attention query. AdapTac's
verified ablation says the wiring alone loses to vision-only — the future-prediction aux is the driver. So
state-as-query must ship paired with a predictive objective **chosen so it forces retention of visual content
that state alone cannot predict**. This report picks that objective.

**Method:** builds on REDESIGN_RESEARCH_20260729.md and LITERATURE_20260728.md (claims from those are tagged
`[internal-verified]` — they went through full per-paper verification on 07-28/29 and are not re-derived).
New material: ~12 direct WebFetch verifications this session (tag `[F]`) plus three parallel verification
sweeps (discrete-vs-continuous targets; state-shortcut/collapse; encoder-stage objective evidence), each of
which fetched every cited source and returned verbatim quotes (tag `[F-sweep]`). Anything not fetched is
marked **UNVERIFIED**.

---

## Executive verdict

1. **Pair state-as-query with FUTURE-VISION LATENT prediction.** It is the only candidate that is (a) not
   satisfiable from state alone, (b) backed at both encoder-pretraining stage (GR-1 ablation 3.33→4.21;
   Seer foresight term 3.31→3.64; V-JEPA 2-AC zero-shot real-robot; DINO-WM) and policy stage (FLARE +8.2pp
   RoboCasa, never below baseline; FRAPPE dose curve), and (c) the published fix for exactly our failure
   mode — HIT added a future image-feature L2 *specifically* "preventing the vision-based skill policies
   from ignoring image features and overfitting to proprioception" `[F-sweep, 2406.10454]`.
2. **Never pair it with future-STATE prediction.** No paper pretrains a vision encoder on future-proprio
   regression (dedicated sweep found zero instances — a real gap), and the assembled shortcut literature
   (Octo causal-confusion note, GAP ICLR'26, copycat, State-free Policy) says the state channel alone
   explains near-future state/actions, so a state-conditioned encoder can drive that loss to floor while
   ignoring vision. Same caution applies, less obviously, to **inverse dynamics with state visible**: in a
   delta/OSC action space a_t ≈ f(s_t, s_{t+1}) — ID is nearly closed-form from the state pair (our
   derivation; no published ablation).
3. **Run target format as a first-class arm: discrete tokens vs continuous.** 2605.04678 (discrete beats
   continuous MLP+MSE in all 8 LIBERO cells, +2.7/+2.2) now has independent representation-stage support
   (π0.5-KI 7.5× convergence; DCWM FSQ+CE > MSE; DreamerV2 canonical) **and** real 2025-26 counter-evidence
   in the latent-action line (CLAM, CoMo, Meta LAWM). Net: discrete wins as a *prediction target for
   representation learning*; continuous wins as a *decode head for control*. We only need the target side.
4. **Where the loss attaches is settled: dedicated prediction/readout tokens, never the 8 output queries.**
   FRAPPE (prefix tokens, layer 21/28, loss only on prefix outputs), FLARE (32 future tokens, layer 6/8),
   Octo readout isolation, GeoPredict track queries (training-only, inference unchanged), 2605.04678
   placeholder positions. No published positive instance puts a fixed-target loss directly on the
   observation pathway's outputs `[internal-verified]` — and our E4 harm sits exactly there.
5. **EMA vs frozen: both work; EMA is a small win only when the target is co-trained.** FLARE ablation:
   ρ=0.995 best, but fully frozen (ρ=1.0) still beats baseline `[F]`. FRAPPE, DINO-WM, V-JEPA 2-AC, and
   2605.04678 all succeed with hard-frozen targets. Rule: target = ground-truth data (pixels/actions/force)
   → learned head fine; target = latents → must be stop-grad frozen or EMA (Tang/SimSiam collapse theory).
6. **Horizon: short.** FRAPPE h=8 > h=16 ≈ h=32; Seer n=3 control steps; GR-1 Δt=3 at finetune; V-JEPA 2-AC
   0.25 s/frame steps. Everything that wins predicts ≲1 s ahead; screen h ∈ {~0.3 s, ~0.8 s} in cache ticks.

---

## 1. Candidate-by-candidate evidence (Q1)

Summary table — "stage" = where the published evidence sits (encoder pretraining vs policy-stage aux);
"state-safe" = NOT satisfiable from state alone when state conditions the encoder (see §2).

| Candidate | Best evidence | Stage | State-safe? | Verdict |
|---|---|---|---|---|
| Future-vision latent (continuous) | GR-1, Seer, V-JEPA 2-AC, DINO-WM, VPP; FLARE, FRAPPE | both | **yes** | **primary** |
| Future-vision discrete tokens | 2605.04678; LAPA, Moto, π0.5-KI, DCWM | both | **yes** | **arm 2 (format)** |
| Inverse dynamics | 2305.16985 (cleanest controlled study) | encoder | only if head is state-blind | additive term, constrained |
| Temporal contrastive | R3M, VIP, MCR term | encoder | yes but weak | not primary |
| Future-force | AdapTac, ViTacFormer | policy-aux | yes (contact info) | conditional on E6 |
| Future-state (proprio) | none found (gap) | — | **NO** | **kill** |

### 1a. Future-vision latent prediction — the strongest and broadest evidence

**Encoder-pretraining stage:**
- **GR-1** (arXiv 2312.13139, ICLR 2024) `[F-sweep]` — GPT-style policy pretrained on Ego4D **video
  prediction in pixel space** ("MSE between the reconstructed and original images"), mask-token readout,
  Δt=1 at pretraining → **Δt=3** at robot finetuning; finetune loss `L_arm + L_gripper + L_video` at equal
  weights. Cleanest pretraining-stage ablation in the corpus: **without video pretraining CALVIN avg length
  3.33 vs 4.21 with; at 10% data, success 0.526 vs 0.778.** ABC→D 85.4% vs best baseline 53.3%.
- **Seer** (arXiv 2412.15109) `[F-sweep]` — predictive inverse dynamics: foresight token "predicts the RGB
  images at the time step t+n", **n=3**, pixel MSE, combined **ℒ = 0.5·ℒ_fore + ℒ_inv**; pretrain on DROID
  then finetune. Ablations isolate the future-prediction term twice: finetune-only 3.31 (neither) → 3.41
  (foresight only) → 3.64 (both); pretraining 3.64 (scratch) → 3.73 (foresight only) → 3.98 (both). Real
  robot 60.0% scratch → 78.4% pretrained.
- **V-JEPA 2 / V-JEPA 2-AC** (arXiv 2506.09985) `[F-sweep]` — encoder pretrained with masked **latent** L1
  to an **EMA target encoder** ("∥Pϕ(Δy,Eθ(x))−sg(Eθ̄(y))∥₁"); then the AC stage **freezes the encoder** and
  trains an action-conditioned predictor on 62 h of raw DROID video (16-frame clips at 4 fps) with L1
  teacher-forcing + rollout losses against **frozen-encoder targets**. Zero-shot real Franka via CEM
  planning: reach 100%, pick-and-place 65–80%. Strongest evidence that latent future prediction over a
  frozen backbone yields real-robot competence — structurally the closest published analogue to our
  frozen-ViT + trained-predictor stack.
- **DINO-WM** (arXiv 2411.04983) `[F-sweep]` — next-step **MSE on frozen DINOv2 patch latents**, history
  H ∈ {1,2,3}; PushT 0.90 vs DreamerV3 0.30. Two decision-relevant ablations: latent-prediction beats
  adding a reconstruction loss (0.92 vs 0.80 PushT), and the same world model over **R3M features gets 0.42
  vs 0.90 over DINOv2 patches** — pooled temporal-contrastive embeddings are a poor substrate for prediction.
- **VPP** (arXiv 2412.14803, ICML 2025) `[internal-verified]` — video-diffusion future predictor consumed
  as query tokens: Calvin ABC-D 4.33 vs VC-1 1.23; +31.6% real dexterous.
- **VLA-JEPA** (arXiv 2602.10098, 2026) `[F-sweep, abs only]` — JEPA pretraining (target encoder on future
  frames, latent-space prediction) then action-head finetune; claims consistent LIBERO/SimplerEnv/real
  gains; numbers UNVERIFIED (abstract only).

**Policy-stage aux (the FLARE/FRAPPE line, both fetched again this session for recipe detail):**
- **FLARE** (arXiv 2505.15659) `[F]` — cosine loss `−cos(f_θ(...), g(ϕ_{t+H}))`, **λ=0.2** (peak of
  {0.1..0.5}; worst λ still +7.8pp `[internal-verified]`), **M=32 learnable future tokens read out at DiT
  layer 6 of 8**, horizon = action-chunk length (16 steps), target = action-aware Q-former embedding (32
  tokens) pretrained with flow-matching then **EMA-updated ρ=0.995**. Target ablation: no-FLARE 43.9% →
  SigLIP2 256-token 49.6% → SigLIP2 pooled-64 50.9% → action-aware 55.0%. **EMA ablation: ρ ∈ {0.99, 0.995,
  0.999, 1.0}; 0.995 best, frozen (1.0) still above baseline.** RoboCasa-24: 70.1% vs 61.9%. Never below
  baseline in any ablation `[internal-verified]`.
- **FRAPPE** (arXiv 2602.17259) `[F]` — cosine to **stop-gradient frozen** VFM embeddings (CLIP 400M +
  DINOv2 142M + ViT 300M, used simultaneously) of the frame **h=8** ahead (h=8: 35.3% > h=16: 35.0% >
  h=32: 29.7%), on **dedicated learnable prefix tokens at DiT layer 21/28**, loss touches prefix outputs
  only. Dose curve λ = 0/0.001/0.02/0.05/0.1/0.5 → 14.0/18.5/26.4/**32.5**/22.0/23.5%. New detail from this
  fetch: with multiple target streams they "observed a mode collapse phenomenon, where a single stream would
  dominate" — one frozen target family is the safer default.

### 1b. Discrete-token targets — deep dive in §3; short version

2605.04678 `[F]` wins all 8 LIBERO cells over a continuous MLP+MSE arm (+2.7/+2.2 avg). Independent
representation-stage support: **π0.5-KI** (arXiv 2505.23705) `[F-sweep]` trains the VLM backbone on
discrete FAST action tokens (DCT → quantization → BPE) while gradient-insulating a continuous action
expert; pure continuous flow-matching needs "~7.5× more training steps" to match. **DCWM** (arXiv
2503.00653) `[F-sweep]`: FSQ (levels {5,3}) latents + stochastic dynamics + cross-entropy beats continuous
latents + MSE on DMControl, and *discrete trained with MSE underperforms discrete trained with CE* — the
loss geometry, not just the codebook, carries the win. **DreamerV2** (arXiv 2010.02193) `[F-sweep]`:
categorical beats Gaussian latents on 42/55 Atari tasks. Counter-evidence exists and is real (§3c).

### 1c. Inverse dynamics — cleanest controlled evidence, but state-unsafe in our wiring

- **Inverse Dynamics Pretraining** (arXiv 2305.16985, NeurIPS 2023) `[F-sweep]` — single-step **k=1** ID,
  action MSE `(a − f(ϕ(o),ϕ(o')))²`, encoder co-learned with the head. Same-encoder comparison vs BC
  pretraining, explicit forward dynamics (pixel recon), implicit forward dynamics (InfoNCE), SimCLR:
  "Inverse dynamics ... is the only method to substantially outperform the baseline of training from
  scratch"; matches the ground-truth-state skyline; scales best with pretraining size. Theory: under
  latent-linear dynamics ID recovers the encoder "up to linear transformation," while the Bayes-optimal BC
  representation can be useless.
- **2606.07687** `[internal-verified]` — ID aux amplifies existing temporal structure, cannot create it
  (V-JEPA-2+ID probe 0.85; Dreamer-4 +0.00).
- Latent-action scaling of the same idea: **LAPA** (2410.11758, ICLR 2025) `[F-sweep]` — VQ latent actions
  (generation space 8⁴), CE prediction, beats an action-labeled SOTA VLA on real tasks; **UniVLA**
  (2505.06111, RSS 2025) `[F-sweep]` — task-centric latent actions in DINO feature space, |C|=16, 4 tokens.
- **The v0.4 catch (our derivation, no published ablation):** all the above condition on *frames only*. If
  the ID head can see state at t and t+k — which it implicitly can once state conditions the extraction
  queries — then in an OSC/delta action space the action is approximately a function of the state pair, and
  the visual pathway is bypassed (copycat mechanism, §2). ID is only admissible as a **state-blind** head.

### 1d. Temporal contrastive — positive but confounded; negative substrate evidence

- **R3M** (arXiv 2203.12601, CoRL 2022) `[F-sweep]` — InfoNCE time-contrastive + video-language + L1/L2
  sparsity (λ_tcn=1, λ_lang=1, λ_sparse=1e-5). +10% avg over prior PVRs, real robot 56% vs CLIP 24%. **No
  ablation removes the time-contrastive term itself**; closest isolation: the language-free variant still
  beats prior SOTA. So "temporal contrastive alone drives policy gains" is *not* actually isolated anywhere.
- **MCR** (2410.22325) `[internal-verified]` — time-contrastive is its *smallest* ablation term (−11.2)
  vs dynamics alignment (−17.0 largest).
- Negatives: 2312.12444 `[internal-verified + F-sweep]` (R3M/VIP/MVP rank poorly under visual shift) and
  DINO-WM's substrate ablation (R3M 0.42 vs DINOv2-patch 0.90). VIP (2210.00030) `[F-sweep]` is implicit
  time-contrastive with real few-shot results but is aimed at reward/goal embeddings, not our extraction.
- Verdict: fine as a tie-breaker term; not the paired objective.

### 1e. Future-force — real driver, wrong gate to decide here

- **AdapTac** (arXiv 2505.13982) `[F]` — diffusion force head "takes the visual and tactile features as
  input and predicts the future net force F_p^n at the next n steps" (control 5 Hz; n = action-chunk
  length), `ℒ = ℒ_π + α·ℒ_ffp`, training-time only (at inference the query uses observed force). Flip
  ablation this fetch: 50% (no FFP/guidance) → 70% (predicting *observed* force) → **90% (predicting
  *future* force)** — the future-ness itself is worth +20pp. Fusion-without-aux loses to vision-only
  `[internal-verified: 67% vs 73% avg]`.
- **ViTacFormer** (2506.15953) `[internal-verified]` — future-tactile L1 head (≈0.08, chaseable, fed back
  as tokens); ablating it drops 100→70 / 100→70 / 90→60 / 90→70.
- Verdict: keep exactly as E6 designed it (separate tokens, late fusion, future-force aux); whether it
  enters v0.4 at all is decided by tonight's tool_hang result, not by this report.

### 1f. Future-state prediction — no direct evidence exists; the nearest neighbors point away

A dedicated sweep found **no paper that pretrains a vision encoder by predicting future proprioceptive
state**. Nearest: **RPT** (arXiv 2306.10007, CoRL 2023) `[F-sweep]` masks and reconstructs interleaved
vision/proprio/action tokens (MSE, best at 70–90% masking; real Pick ~94% vs ~25% scratch) but "does NOT
contain ablations isolating which modality predictions matter most" — and it is in-filling on a trunk, not
future prediction on a vision encoder. **MCR** aligns *vision* (input) to *state-action dynamics* (target)
— the mirror image of our danger case, safe precisely because state is not an input there. GE-Sim 2.0
(2605.27491) reportedly regresses future joint state inside a video simulator — **UNVERIFIED**, and it is a
simulator component regardless. The absence is itself informative: nobody has made future-proprio the
representation target for vision, and §2 explains why it would be a shortcut in our wiring.

---

## 2. The interaction that matters: what state alone CAN satisfy (Q2)

### 2a. The shortcut is published — assembled across five papers

- **Octo** (arXiv 2405.12213, RSS 2024) `[F-sweep, PDF text]`: "Adding Proprioceptive Inputs: Policies
  trained with propioceptive observations seemed generally worse, potentially due to a strong correlation
  between states and future actions. We hypothesize this might be due to a causal confusion between the
  proprioceptive information and the target actions." — the published statement that the state channel
  alone statistically explains near-future targets.
- **GAP** (arXiv 2602.12032, ICLR 2026) `[F-sweep]`: "the policy naturally gravitates toward concise
  proprioceptive signals that offer faster loss reduction when training, thereby dominating the
  optimization and suppressing the learning of the visual modality during motion-transition phases."
  Vision+proprio policies measured **15.8% worse than vision-only** on their tasks; remedy = phase-guided
  down-weighting of proprio gradients. This is our concern stated as an optimization theorem-sketch: proprio
  wins the gradient race even when vision is required.
- **State-free Policy** (arXiv 2509.18644) `[F-sweep]`: removing proprio input entirely: height
  generalization 0% → 85%, horizontal 6% → 64%; "the policy directly associates absolute states with expert
  trajectories ... overfit[s] to the training trajectories."
- **Copycat / causal confusion** (arXiv 2010.14876 NeurIPS 2020 `[F]`; arXiv 1905.11979 NeurIPS 2019
  `[F]`): when the target is strongly autocorrelated and its past is recoverable from an easy channel, BC
  "learns to cheat by predicting the expert's previous action"; "access to more information can yield worse
  performance." The cure is stated in information terms — adversarially remove the nuisance information
  while keeping what predicts the target.
- **Greedy multimodal learning** (arXiv 2202.05306) `[F]`: multimodal nets rely on the modality that is
  fastest to learn and under-fit the rest — the domain-general version (a.k.a. modality laziness).

### 2b. Self-referential latent targets have a second failure mode: collapse

- **Tang et al.** (arXiv 2212.03319, ICML 2023) `[F-sweep]`: for latent self-prediction "trivial
  representations (such as constants) minimize the prediction error"; collapse is avoided only by non-loss
  mechanisms — "a faster paced optimization of the predictor" and "semi-gradient updates."
- **SimSiam** (arXiv 2011.10566) `[F]`: "collapsing solutions do exist for the loss and structure, but a
  stop-gradient operation plays an essential role in preventing collapsing" — EMA not strictly necessary,
  asymmetry is.
- **Voelcker et al.** (arXiv 2406.17718) `[F-sweep]`: latent self-prediction is a helpful *auxiliary* task,
  "while observation reconstruction can provide more useful features when used in isolation" — external,
  observation-grounded targets retain more when the objective must stand alone.
- **BYOL-AC** (arXiv 2406.02035) `[F-sweep]`: action-conditioning changes which low-rank approximation of
  the dynamics the self-predictive fixed point preserves — supplementary theory for action-conditioned
  predictors.

### 2c. Satisfiability-from-state, per target modality

No single paper runs the exact ablation (state-conditioned encoder × aux-target modality) — confirmed gap;
the table below is the assembled verdict, and running that ablation ourselves would be novel evidence.

| Aux target | Satisfiable from state (+actions) alone? | Basis |
|---|---|---|
| Future proprio state | **YES — short-horizon autoregressive; kill** | Octo quote; copycat mechanism; GAP gradient-race |
| Future/next action (ID, state-visible) | **~YES** in delta/OSC spaces: a_t ≈ f(s_t, s_{t+1}) | our derivation from copycat/Octo; UNVERIFIED as published claim |
| Robot-keypoint tracks | **Largely** — FK of future joints | GeoPredict (2512.16811) `[F]` predicts joint+EE keypoints; helps there (+2.4–4.9pp on RoboCasa Human-50) but its policy is NOT state-privileged; in our wiring this target is state-derivable — caution |
| Own future latent (no external anchor) | Degenerately (collapse) | Tang, SimSiam |
| **Future external vision latents** | **NO** — residual beyond the state-predictable component requires reading the scene (object poses, clutter, outcomes of contact) | HIT built the loss for exactly this; FLARE/FRAPPE/Seer/GR-1 all use it |
| Object-centric / any-point tracks | **NO** — arbitrary scene points move with objects, not the arm | ATM (2401.00025) `[F]`: future trajectories of arbitrary points, pretrained on actionless video, "+80% on average" over video-pretraining baselines, 130+ tasks |
| Future force/tactile | **NO** pre-contact — depends on object/scene contact state | AdapTac future-vs-observed force = +20pp on Flip |
| Temporal contrastive (vision frames) | NO, but weak pressure | R3M caveats §1d |

**The design corollary (state it, don't over-claim it):** an aux objective only exerts pressure on the
visual pathway if its target carries information *not derivable from the conditioning inputs*. We found no
single paper stating this as a theorem — treat it as folklore assembled from Octo + copycat + Tang, and cite
those three. HIT (arXiv 2406.10454) `[F-sweep]` is the direct constructive precedent: "By using forward
dynamics prediction on image features, our method enhances performance by regularizing image feature spaces,
preventing the vision-based skill policies from ignoring image features and overfitting to proprioception"
(L2 on predicted future image-feature tokens; no isolated ablation of that term in the paper).

**Published anti-shortcut mitigations to pair with the objective:** GAP phase-guided proprio-gradient
modulation (2602.12032); DiT-Block per-dimension proprio dropout (2410.10088) `[F-sweep + internal-verified]`;
plus our own state-zeroing sensitivity diagnostic (V0.4.md §5 gate).

---

## 3. Discrete vs continuous targets — the deep dive (Q3)

### 3a. What 2605.04678 actually did (fetched full text this session)

"From Pixels to Tokens: A Systematic Study of Latent Action Supervision for VLA Models" (ICML 2026 per repo;
RUCKBReasoning/From_Pixels_to_Tokens) `[F]`:

- **Two token families.** (i) *Image-based latent actions*: VQ-VAE fine-tuned from UniVLA over a **frozen
  DINOv2** feature space, encoding the visual transition (o_t, o_{t+δ}) with **δ=32 frames**, partitioned
  into controllable z^ctrl vs environment z^env latents; standard VQ objective (recon + codebook +
  commitment); image-based codebook size not stated. (ii) *Action-based latent actions*: custom VQ action
  tokenizer, **K=256 codes, d=128**, H tokens per chunk (H=8 LIBERO / 25 RoboTwin / 20 real), encoder = FFT
  temporal-frequency features → three 1-D temporal convs → transformer; plus a latent-consistency masking
  loss for stability.
- **Four integrations, all at VLA policy-finetuning stage** (latent-action models trained separately, then
  **frozen**): LA-Align (cosine at VLM layer k=17/29, continuous), LA-Direct (CE on discrete image-latent
  tokens at dedicated placeholder positions, final layer), LA-Cond (LA-Direct + action decoding causally
  conditioned on predicted latents), LA-Tok (CE on discrete action tokens).
- **The discrete-vs-continuous ablation (Table 4):** continuous arm = "regress the corresponding continuous
  latent representations ... using a two-layer MLP trained with an MSE loss" from the same placeholder
  states. LA-Direct 97.1 vs LA-Direct(C) 94.4; LA-Tok 95.5 vs LA-Tok(C) 93.3; discrete wins **all 8 cells**
  (+2.7 / +2.2 avg). Continuous still beats no-latent substantially — discrete>continuous, not
  continuous-is-harmful `[internal-verified, kill-overturned 07-29]`.
- **λ:** `L = L_action + λ·L_latent`; Appendix E.4: "performance remains stable across different values of
  λ" (exact grid not extracted).
- **Transfer to encoder pretraining: NOT shown.** "All latent action models are kept frozen during VLA
  training" — the comparison lives entirely at policy-finetuning. Using it for our *encoder pretrain* is an
  extrapolation; the bridging evidence is LAPA/Moto/UniVLA (below), which apply discrete-token prediction at
  pretraining scale with real-robot wins, though without a continuous arm.
- **Stated limitation:** "it remains open whether stronger latent representations would amplify or narrow
  the observed gaps between strategies."

### 3b. Replications at the representation-training stage (2025-26)

- **π0.5-KI / Knowledge Insulation** (arXiv 2505.23705, Physical Intelligence) `[F-sweep]` — backbone
  trained on **discrete FAST tokens** (DCT per action dimension → quantization → BPE) while the continuous
  flow head is gradient-insulated; continuous-only training needs **~7.5× more steps**. The largest-scale
  endorsement of "discrete targets for representation, continuous head for control."
- **DCWM** (arXiv 2503.00653) `[F-sweep]` — **FSQ** (2 channels, levels {5,3}) + stochastic dynamics +
  **cross-entropy**: "Discretizing the latent space improves sample efficiency over the continuous latent
  space and formulating stochastic dynamics and training with cross-entropy improves performance further."
  Discrete-with-MSE underperforms discrete-with-CE — the classification loss is part of the mechanism.
- **DreamerV2** (arXiv 2010.02193, ICLR 2021) `[F-sweep]` — categorical vs Gaussian latents: wins 42/55
  Atari games; ablation Gamer-Mean 11.33 → 3.71 without discrete latents. Mechanism hypothesis: "a mixture
  of categoricals is again a categorical," so the prior can fit the aggregate posterior.
- **VQ-BeT** (arXiv 2403.03181, ICML 2024) `[F-sweep]` — residual-VQ action tokens (Nq=2, k=1024) beat
  Diffusion Policy on PushT (0.78 vs 0.74) and Kitchen (3.66 vs 3.44) — policy-stage, included for
  completeness.
- **Nulls:** LAPA (vocab 8⁴), Moto (codebook 128, 8 motion tokens/frame; motion-token loss ablation CALVIN
  3.10 vs ~2.60 without), villa-X (VQ-32; hybrid trick: tokens "discretized via the VQ module but ...
  represented by [their] continuous codebook center"), UniVLA (|C|=16) — all discrete by construction,
  **none ablates the format** `[all F-sweep]`.

### 3c. Contradictions — where continuous wins, and why the contradiction is narrower than it looks

- **CLAM** (arXiv 2505.04999) `[F-sweep]` — continuous latent actions + jointly-trained decoder crush
  discrete on image-based MetaWorld (e.g., Bin Pick 0.82±0.04 vs 0.14±0.03): "applying quantization to the
  latent space severely limits the expressivity of the latent actions for fine-grained manipulation." BUT
  continuous *without* joint training scores only 0.18–0.28 — the win needs the extra machinery.
- **CoMo** (arXiv 2505.17006) `[F-sweep]` — continuous motion latents + temporal-difference and
  temporal-contrastive regularizers: CALVIN 3.070 vs Moto-discrete 2.255. On LIBERO, **plain continuous
  75.2 vs discrete 75.9 — parity/slight discrete win**; only the regularized version reaches 80.1. CoMo
  concedes VQ "can effectively mitigate the shortcut learning problem."
- **Meta latent-action world models in the wild** (arXiv 2601.05230) `[F-sweep, abs only]` — "continuous,
  but constrained, latent actions are able to capture the complexity of actions from in-the-wild videos,
  something that the common vector quantization does not." Full text UNVERIFIED (PDF over fetch limit).
- **V-JEPA** (arXiv 2404.08471) `[F-sweep]` — continuous *feature-space* regression beats *pixel*
  reconstruction (K400 73.7 vs 68.6 frozen) — but has **no discrete-token arm**; it cannot arbitrate our
  question.
- **OpenVLA-OFT** (arXiv 2502.19645) `[F-sweep]` — at the *action-decode* stage, continuous+L1 beats
  discrete tokens 95.3% vs 90.2% on LIBERO; same direction as UniVLA's decode comparison (95.2 vs 73.6).

**Reconciliation.** Split by stage and by what the codebook is for: (1) as a **prediction target for
representation learning**, discrete/CE wins or ties everywhere it is ablated (2605.04678, π0.5-KI, DCWM,
DreamerV2; CoMo's *plain* arms); (2) as an **action decode head**, continuous wins (OpenVLA-OFT, UniVLA);
(3) as the **latent bottleneck of a latent-action model on diverse video**, VQ loses to
*constrained/regularized* continuous latents (CLAM, CoMo-full, Meta LAWM) because a small codebook discards
fine-grained motion. We need case (1) only — v0.4's targets are prediction targets, and the policy head
stays diffusion/continuous regardless. Remaining honest gap: **no paper runs discrete-tokens vs
continuous-feature-regression vs pixels at matched capacity in robot encoder pretraining** — which is
exactly what the v0.4 screen can be.

---

## 4. EMA vs frozen vs learned targets; where the loss attaches (Q4)

### 4a. Target update rule — decision table

| Target type | Update rule with evidence | Evidence |
|---|---|---|
| Ground-truth data (pixels, actions, force, tracks) | co-learned head is fine; no collapse possible | GR-1/Seer pixel MSE `[F-sweep]`; 2305.16985 ID co-learned `[F-sweep]`; AdapTac force `[F]`; GeoPredict tracks `[F]` |
| External pretrained latents | **frozen + stop-grad works, digit-exact** | FRAPPE (3 frozen VFMs) `[F]`; DINO-WM (frozen DINOv2) `[F-sweep]`; 2605.04678 (frozen latent-action models) `[F]`; V-JEPA 2-AC (frozen encoder targets) `[F-sweep]` |
| Co-trained/chaseable embedding | **EMA slightly beats frozen; frozen still > baseline** | FLARE ρ ∈ {0.99, 0.995, 0.999, 1.0}: 0.995 best, 1.0 above baseline `[F]` |
| Model's own latents, no anchor | collapse without stop-grad/EMA/fast-predictor; weakest standalone features | Tang 2212.03319, SimSiam 2011.10566, Voelcker 2406.17718 `[F/F-sweep]` |

For v0.4 the question nearly answers itself: our ViT-B/16 backbones are already frozen, so
**frozen-backbone features of the future frame are free, stable targets** (DINO-WM / V-JEPA 2-AC pattern),
and no EMA machinery is needed. EMA only becomes relevant if we later make the *Perceiver's own* output the
target — which §2b says not to do without stop-grad asymmetry anyway. One caution from FRAPPE: multiple
simultaneous target streams showed "mode collapse ... where a single stream would dominate" — start with
one target family.

### 4b. Attachment point — dedicated tokens win in every published instance

- **FRAPPE** `[F]`: learnable prefix tokens, loss **only** on prefix outputs, at DiT layer 21/28 (~75%
  depth; depth ablated, deep > shallow).
- **FLARE** `[F]`: 32 future tokens, read out at layer 6 of 8; layer-4 readout drops ~3pp; layers 6-8 stable.
- **2605.04678** `[F]`: dedicated placeholder positions at the final layer; early-layer alignment *hurts*
  (LA-Align at early layers −1.5) `[internal-verified]`.
- **Octo** (2405.12213) `[internal-verified]`: readout tokens attend to obs tokens but are never attended
  back — block-mask isolation so the aux pathway cannot perturb base processing.
- **GeoPredict** (2512.16811) `[F]`: learnable future track queries, "used only during training; inference
  behaves exactly like the base VLA policy."
- **HIT** (2406.10454) `[F-sweep]`: predicts future image-feature tokens inside the transformer ("L2
  feature loss on these predicted image features") — same pattern, on the policy trunk.
- **Against on-output attachment:** no published positive instance puts a fixed-frozen-target cosine loss
  directly on the observation encoder's output pathway `[internal-verified, LITERATURE_20260728]`; our E4
  monotone-harm curve is the in-house negative for exactly that placement.

**v0.4 translation:** add **P dedicated predictor queries** to the Perceiver (P ≈ 8-32; FRAPPE gives no
count, FLARE uses 32), which cross-attend to the same frozen K/V (+ state conditioning) but are **not
attended back by the 8 output queries** (Octo mask); predictive loss lands only on the predictor queries'
outputs. Since our Perceiver is shallow, "layer" reduces to: attach at the final cross-attention block, not
the first.

---

## 5. The ranked menu (Q5)

> Units note: source horizons — FRAPPE h=8 frames (best of 8/16/32), Seer n=3 control steps, GR-1 Δt=3,
> V-JEPA 2-AC 4 fps (0.25 s/step), FLARE H=16-step chunk, 2605.04678 image-latents δ=32 frames (long-δ is
> for *latent-action extraction*, not the aux horizon). Consensus: **predict ≲1 s ahead, with the optimum
> near 0.3-0.8 s**. Convert at the robocasa cache tick rate; at 20 Hz control that is h ≈ 6-16 ticks.

### #1 — Action-conditioned future-vision-latent prediction on dedicated predictor queries (continuous)

- **Target modality:** frozen ViT-B/16 patch-pooled features of the frame at t+h, per camera (own backbone
  = free frozen target; DINO-WM / V-JEPA 2-AC pattern; FRAPPE shows frozen VFM targets work as-is).
- **Horizon:** screen h ∈ {~0.3 s, ~0.8 s} in cache ticks (FRAPPE h=8 optimum + Seer n=3 + V-JEPA 2-AC
  0.25 s; FRAPPE's h=32 already degrades).
- **Discrete vs continuous:** continuous for this arm — cosine (FRAPPE/FLARE) or L1 (V-JEPA 2) on the
  latent; cosine preferred (two digit-exact positive recipes).
- **λ:** 0.05-0.2 — FRAPPE dose optimum 0.05 (curve 14.0/18.5/26.4/32.5/22.0/23.5 over
  0/0.001/0.02/0.05/0.1/0.5); FLARE optimum 0.2 of {0.1..0.5}. Non-monotone: screen {0.05, 0.2}.
- **Attachment:** P=8-32 dedicated predictor queries in the Perceiver, final cross-attn block, Octo-style
  read-only isolation; loss never touches the 8 output queries.
- **Target update:** hard-frozen (stop-grad); no EMA needed while the target is the frozen backbone.
- **Conditioning:** predictor queries see state and the action chunk (V-JEPA 2-AC, FLARE action-aware
  target, BYOL-AC theory) — actions explain the motion so the *residual the loss chases is scene content*,
  which is precisely what state alone cannot supply.
- **Why #1:** only family with encoder-stage ablations (GR-1 3.33→4.21; Seer +0.33/+0.34), a
  never-below-baseline policy-stage record (FLARE), a digit-exact dose/horizon/placement recipe (FRAPPE),
  real-robot zero-shot on a frozen encoder (V-JEPA 2-AC), and an explicit anti-proprio-shortcut precedent
  (HIT).

### #2 — Same objective, DISCRETE targets (the format arm, run in the same screen)

- **Tokenizer:** VQ or FSQ over the same frozen backbone features of frame t+h. Cheapest robust choice:
  **FSQ levels {5,3}** (DCWM — no codebook collapse machinery) or **VQ K=256, d=128** (2605.04678's
  action-tokenizer scale); train tokenizer on the cache first, then **freeze** (2605.04678 pattern:
  "all latent action models are kept frozen").
- **Loss:** cross-entropy on the same predictor queries (CE is part of the win — DCWM's discrete+MSE arm
  underperforms).
- **λ:** start in the same 0.05-0.2 band but re-tune by matching aux-gradient norm to the continuous arm
  (CE and cosine scales differ; 2605.04678 reports λ-stability for CE).
- **Horizon:** same {~0.3 s, ~0.8 s} screen. Codebook caution from the anti-VQ line (CLAM/CoMo/Meta LAWM):
  too-small codebooks discard fine-grained motion — if the discrete arm underperforms, scale K before
  killing the arm (LAPA's vocab-scaling 4→8→16 helped).
- **Why #2:** discrete beat continuous in every *ablated* representation-target comparison (2605.04678 all
  8 cells; DCWM; DreamerV2; π0.5-KI at scale), our own E4 failure was a continuous cosine, and the
  discrete-vs-continuous-at-encoder-pretraining cell is empty in the literature — running both arms is the
  experiment the field hasn't done.

### #3 — State-BLIND inverse-dynamics head as an additive term (k=1)

- **Recipe:** predict a_t from (z_t, z_{t+1}) where z are the Perceiver outputs **with the state-conditioning
  pathway masked for this head** (else a_t ≈ f(s_t, s_{t+1}) in a delta action space and the term is
  shortcut-satisfiable — §2c); MSE; co-learned head (target = ground-truth actions, collapse-safe); small
  weight (Seer's α=0.5 is the only published anchor for a combined future+ID recipe; its ID term added
  +0.23-0.25 on top of foresight).
- **Evidence:** 2305.16985 — the only same-encoder controlled comparison, ID the only pretraining objective
  beating scratch, with recovery theory; 2606.07687 — ID amplifies temporal structure (ours will exist once
  #1 is in); Seer — foresight+ID > foresight alone at both stages.
- **Role:** additive term / tie-breaker, not the primary — because with state visible it is unsafe, and
  state-blinding it costs the state-as-query design its premise for that head.
- **Conditional swap:** if E6 tool_hang greenlights force, **future-force prediction** (AdapTac: `ℒ_π +
  α·ℒ_ffp`, next-n-steps at control rate, training-only; future-vs-observed worth +20pp on Flip) replaces
  ID as the #3 term on contact-rich tasks.

### Anti-menu (pre-registered kills)

- **Future-state prediction as the paired objective** — satisfiable from the conditioning input; zero
  published support as a vision-encoder objective; Octo/GAP/copycat mechanism against it. If a state head
  is kept for diagnostics, exclude its gradient from the vision pathway.
- **Temporal contrastive as the primary** — never isolated as the driver (R3M has no tcn-removal ablation);
  substrate evidence against (DINO-WM 0.42 vs 0.90; 2312.12444).
- **Loss on the 8 output queries** — E4's placement harm + zero published positive instances; dedicated
  read-only tokens only.
- **Robot-keypoint-track targets** (GeoPredict-style) under state-conditioning — FK-derivable; if tracks
  are wanted later, use ATM-style *any-point/object* tracks.

### Shortcut gate for every arm (feeds V0.4.md §5)

At ep10 ckpt, before any rollout: (1) state-zeroing delta on the predictive-loss value — if zeroing state
barely moves the aux loss, the predictor is running on vision (good); if zeroing *vision* barely moves it,
the arm is shortcutted (kill); (2) aux-loss-chaseability check à la FLARE/ViTacFormer (loss must visibly
descend — unchaseable targets were the E4/torque-aux failure signature `[internal-verified]`).

---

## Sources

| # | Paper | ID / URL | Used for | Fetch status |
|---|---|---|---|---|
| 1 | From Pixels to Tokens (latent-action supervision study) | arXiv 2605.04678 | Q3 anchor; recipes; Table 4 | **F** (full HTML) |
| 2 | FLARE: Robot Learning with Implicit World Modeling | arXiv 2505.15659 | recipe, EMA ablation, RoboCasa | **F** (full HTML v1) + internal-verified |
| 3 | FRAPPE future-alignment prefix tokens | arXiv 2602.17259 | dose/horizon/placement; frozen targets; mode-collapse note | **F** (full HTML) + internal-verified |
| 4 | AdapTac | arXiv 2505.13982 | future-force aux recipe; Flip ablation | **F** (full HTML v2) + internal-verified |
| 5 | GeoPredict | arXiv 2512.16811 | robot-keypoint track queries; RoboCasa numbers | **F** (full HTML) |
| 6 | ATM: Any-point Trajectory Modeling | arXiv 2401.00025 | object-centric track targets | **F** (abs) |
| 7 | Causal Confusion in Imitation Learning | arXiv 1905.11979 | shortcut theory | **F** (abs) |
| 8 | Fighting Copycat Agents | arXiv 2010.14876 | shortcut theory | **F** (abs) |
| 9 | Greedy multimodal learning | arXiv 2202.05306 | modality-laziness theory | **F** (abs) |
| 10 | SimSiam | arXiv 2011.10566 | stop-grad vs collapse | **F** (abs) |
| 11 | HumanPlus / HIT | arXiv 2406.10454 | future image-feature L2 vs proprio overfit | F-sweep (full HTML) |
| 12 | Octo | arXiv 2405.12213 | proprio causal-confusion quote; readout isolation | F-sweep (PDF) + internal-verified |
| 13 | GAP: vision-proprioception failure | arXiv 2602.12032 (ICLR 2026) | proprio gradient dominance; −15.8% | F-sweep (abs + project page) |
| 14 | State-free Policy | arXiv 2509.18644 | remove-proprio generalization jumps | F-sweep (full HTML) |
| 15 | DiT-Block Policy | arXiv 2410.10088 | per-dim proprio dropout | F-sweep + internal-verified |
| 16 | Tang et al., self-predictive learning | arXiv 2212.03319 | collapse theory | F-sweep (abs) |
| 17 | Voelcker et al., when does self-prediction help | arXiv 2406.17718 | latent-self-pred vs obs-grounded | F-sweep (abs) |
| 18 | BYOL-AC unifying framework | arXiv 2406.02035 | action-conditioned fixed points | F-sweep (abs) |
| 19 | Seer (predictive inverse dynamics) | arXiv 2412.15109 | n=3, α=0.5, both-stage ablations | F-sweep (abs + full HTML) |
| 20 | V-JEPA 2 / 2-AC | arXiv 2506.09985 | latent-L1, EMA→frozen stages, real-robot | F-sweep (abs + full) |
| 21 | V-JEPA | arXiv 2404.08471 | feature > pixel targets; no discrete arm | F-sweep (abs/full) |
| 22 | DINO-WM | arXiv 2411.04983 | frozen-patch MSE targets; R3M substrate ablation | F-sweep (abs + full) + internal-verified |
| 23 | GR-1 | arXiv 2312.13139 (ICLR 2024) | pixel-MSE video pretraining ablation | F-sweep (abs + ar5iv full) |
| 24 | GR-2 | arXiv 2410.06158 | scale datapoint (recipe details UNVERIFIED) | F-sweep (abs) |
| 25 | VPP | arXiv 2412.14803 (ICML 2025) | predictive features beat static | F-sweep (abs) + internal-verified |
| 26 | VLA-JEPA | arXiv 2602.10098 | 2026 JEPA-pretrain-then-finetune (numbers UNVERIFIED) | F-sweep (abs) |
| 27 | Inverse Dynamics Pretraining | arXiv 2305.16985 (NeurIPS 2023) | cleanest controlled encoder comparison; theory | F-sweep (abs + ar5iv full) |
| 28 | LAPO | arXiv 2312.10812 (ICLR 2024) | latent actions from video (Procgen) | F-sweep (abs) |
| 29 | LAPA | arXiv 2410.11758 (ICLR 2025) | VQ latent-action pretraining, vocab 8⁴ | F-sweep (full HTML) |
| 30 | UniVLA | arXiv 2505.06111 (RSS 2025) | DINO-space latent actions; decode comparison | F-sweep (full HTML) |
| 31 | R3M | arXiv 2203.12601 (CoRL 2022) | tcn recipe; NO isolating ablation | F-sweep (ar5iv full) |
| 32 | TCN | arXiv 1704.06888 | historical time-contrastive | F-sweep (abs) |
| 33 | VIP | arXiv 2210.00030 (ICLR 2023) | implicit time-contrastive | F-sweep (abs) |
| 34 | What makes PVRs successful | arXiv 2312.12444 (CoRL 2024) | R3M/VIP/MVP shift brittleness | F-sweep (abs) + internal-verified |
| 35 | RPT | arXiv 2306.10007 (CoRL 2023) | masked sensorimotor prediction; no modality ablation | F-sweep (abs + ar5iv full) |
| 36 | MCR | arXiv 2410.22325 | dynamics-alignment direction (vision→state target) | F-sweep (abs) + internal-verified |
| 37 | π0.5-KI Knowledge Insulation | arXiv 2505.23705 | discrete targets for backbone, 7.5× | F-sweep (full HTML) |
| 38 | Discrete Codebook World Models | arXiv 2503.00653 | FSQ {5,3} + CE > MSE | F-sweep (full HTML) |
| 39 | DreamerV2 | arXiv 2010.02193 (ICLR 2021) | categorical vs Gaussian, 42/55 | F-sweep (PDF read) |
| 40 | VQ-BeT | arXiv 2403.03181 (ICML 2024) | RVQ action tokens beat DP | F-sweep (full HTML) |
| 41 | CLAM | arXiv 2505.04999 | continuous latents win w/ joint decoder | F-sweep (full HTML) |
| 42 | CoMo | arXiv 2505.17006 | plain-continuous parity; regularizers needed | F-sweep (full HTML) |
| 43 | Meta latent-action WMs in the wild | arXiv 2601.05230 | anti-VQ quote (full text UNVERIFIED) | F-sweep (abs only) |
| 44 | OpenVLA-OFT | arXiv 2502.19645 | decode-stage continuous > discrete | F-sweep (full HTML) |
| 45 | villa-X | arXiv 2507.23682 | VQ-with-continuous-centers hybrid | F-sweep (full HTML) |
| 46 | ViTacFormer | arXiv 2506.15953 | future-tactile head ablation | internal-verified |
| 47 | HRP / SpawnNet / Du et al. etc. | (see REDESIGN_RESEARCH / LITERATURE docs) | context only | internal-verified |

**UNVERIFIED leads (surfaced, not fetched):** GE-Sim 2.0 (2605.27491), CLAW (2606.04130), CARE (2601.22467),
LAFP (2606.10517), LAOF (2511.16407), "Why Latent Actions Fail" (2605.20223, abs only), Discrete Diffusion
VLA (2508.20072), FAST tokenizer paper (2501.09747), Moto full recipe beyond ablation (2412.04445 fetched
but format-null), STP (2403.05304), AtomVLA (2603.08519), "Latent Video Prediction Learns Better World
Models" (2605.15618), GR-1 venue listing beyond ar5iv.

**Known gaps (state plainly in the v0.4 doc):** (1) no paper runs state-conditioned-encoder × aux-target-
modality — our screen is novel evidence either way; (2) no matched-capacity discrete-vs-continuous-vs-pixel
comparison at robot encoder pretraining — arms #1 vs #2 fill it; (3) FLARE/FRAPPE evidence is policy-stage —
its transfer to Perceiver pretraining is an extrapolation backed by the encoder-stage GR-1/Seer/V-JEPA-2
results, not identical placement.
