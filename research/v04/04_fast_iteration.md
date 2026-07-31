# Fast-iteration methodology for BC / diffusion-policy experiment screening

**Research memo — v0.4 prep. Date: 2026-07-30.**
**Question:** what do the literature + established practice offer for fast, reliable experiment screening in behavior cloning / robot learning, given our setup (Diffusion Policy transformer, RoboCasa/robomimic sim, A6000s, ~10–16 h per arm today, target ~3–5 h per verdict)?

**Verification convention:** every load-bearing claim carries an arXiv ID or repo link. Claims checked against a fetched source (arXiv page/HTML/PDF, project page, or repo README fetched during this research) are cited plainly. Anything not directly fetched is marked **[UNVERIFIED]**. All statistics in §4 are computed here (exact binomial / Wald formulas + Monte-Carlo check, code in `/tmp/gate_math.py`, reproduced inline).

---

## TL;DR — the proposed loop

| Stage | What | Cost | Decision reliability |
|---|---|---|---|
| 0. Offline pre-filter | Encoder-level attention/Grad-CAM segmentation metrics (Burns Jaccard, MCR manipulation-centricity) — **only for encoder/representation choices, only after in-house calibration** | minutes–1 h, no training | R²≈0.85 / R≈0.93 *across pretrained encoders* in the papers; our own probe inversion says: calibrate before trusting |
| 1. Early-epoch gate | At ~epoch 15/50 (~30% budget): 25 rollouts, DDIM ~10 steps | ~3–4.5 h | Kills only arms ≥20 pts below the band (false-kill ≤5%, exact binomial); <10-pt gaps = tie, continue |
| 2. Reduced-cost final eval | 50 rollouts, DDIM 10–16 steps, sequential stopping (truncated SPRT or STEP) | +6–10 min eval; stops early ~60–80% of the time for clear cases | α=β≈0.05–0.10 vs a decision band; E[N]≈20–35 for 20-pt effects |
| 3. Confirmation (winners only) | 100–200 rollouts, 2 seeds, full-fidelity sampler, Wilson/UMA CIs | ~2–4 h, rare | Certifies 10-pt gaps; publication-grade |

Screening verdict lands at **~3–5 h**; the full 10–16 h path is paid only by survivors.

---

## 1. Offline metrics that predict policy success (without rollouts)

### 1.1 Burns et al. — emergent segmentation (ViT CLS-attention Jaccard)

**Paper:** *What Makes Pre-Trained Visual Representations Successful for Robust Manipulation?* Burns, Witzel, Hamid, Yu, Finn, Hausman. arXiv [2312.12444](https://arxiv.org/abs/2312.12444); CoRL 2024, PMLR v270 ([burns25a](https://proceedings.mlr.press/v270/burns25a.html)). Project: [kayburns.github.io/segmentingfeatures](https://kayburns.github.io/segmentingfeatures/). **Code:** [github.com/stanford-iris-lab/segmenting_feats](https://github.com/stanford-iris-lab/segmenting_feats/tree/eval) (`eval` branch). *(All fetched.)*

**Exact recipe (from the fetched PDF, §5.1):**
> "We evaluate the Jaccard index of an interpolated attention map averaged across heads in the last attention block at the [CLS] token."

1. Take the ViT encoder's **last attention block**, attention from the **[CLS] token** to patch tokens.
2. **Average across heads**, bilinearly **interpolate** to image resolution.
3. Score against ground-truth segmentation masks on the **PASCAL VOC validation set** — the metric is `J = E_D_Pascal[|A∩B| / |A∪B|]` with A = pixels classified positive from the attention map, B = ground-truth mask (mIoU). Binarization specifics live in the released code (`eval` branch), not the paper text.
4. Note: the metric is computed on **Pascal VOC images, not robot images** — it scores an encoder property (emergent objectness), needing **no task data and no training**.

**Validation protocol (fetched, §3/§5.2):** 15 pre-trained models (R3M, MVP ViT-S HOI & ViT-B, VIP, supervised RN/ViT, SIN-SUP, DINO RN/ViT, DINOv2, MoCo-v3 RN/ViT-S/ViT-B, MAE, CLIP ViT-B/16). Frozen encoder + MLP head trained with MSE behavior cloning on **10 demos/task**, 10 tasks (5 FrankaKitchen + 5 Meta-World), 3 seeds × 2 camera angles = 60 policies/model, 11 evals/policy (~9,000 evaluations); each correlation point averages 6,000 eval runs. OOD suite: lighting, texture, distractors (1/3/9 YCB objects).

**Exact numbers (Figure 5, read from the fetched PDF):**

| Metric vs OOD success | ViTs R² | ViTs ρ (Spearman) | ResNets R² | ResNets ρ |
|---|---|---|---|---|
| **Jaccard (segmentation)** | **0.85** | **0.90** | n/a (ViT-only metric) | n/a |
| In-distribution success | 0.64 | 0.73 | 0.43 | 0.60 |
| ImageNet linear probe | 0.07 | 0.21 | 0.80 | −0.50 |
| Shape bias | 0.04 | −0.18 | 0.99 | 1.00 (few points) |

So the task-prompt's "R² ~0.85" is confirmed: **R² = 0.85, ρ = 0.90, for ViTs, against *out-of-distribution* success.** Real-world spot-check: ACT screwdriver pick-up, 50 demos — MVP 0% vs MoCo-v3 40% success (10 test rollouts), rank consistent with Jaccard.

**Known failure modes / scope limits (all stated in or implied by the fetched paper):**
- **ViT-only** (needs CLS attention; shape-bias, not Jaccard, tracked ResNets).
- Predicts **OOD generalization**, not in-distribution success (ID-vs-OOD itself only R²=0.64, with outliers like MVP ViT-S: 39.86 → 6.63).
- Validated **across heterogeneous pretrained encoders** with frozen features + tiny MLP head and 10 demos — *not* across training-recipe variants of a single fine-tuned policy (our usual arm-vs-arm case).
- Object-level-shift counter-argument tested and rejected (Fig. 6: high-Jaccard models keep top rank as distractors increase 1→3→9).
- One real task only; no diffusion heads in the study.

### 1.2 MCR — manipulation centricity (Grad-CAM ∩ SAM-2 Jaccard)

**Paper:** *Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets.* arXiv [2410.22325](https://arxiv.org/abs/2410.22325) (ICLR 2025). Project: [robots-pretrain-robots.github.io](https://robots-pretrain-robots.github.io/). **Code:** [github.com/luccachiang/robots-pretrain-robots](https://github.com/luccachiang/robots-pretrain-robots); checkpoints on HF `GqJiang/robots-pretrain-robots`. *(All fetched.)*

**Exact recipe (from fetched arXiv HTML v2):**
1. Run **Grad-CAM** on the visual encoder over robot-task images → saliency of decision-relevant regions.
2. Build **ground-truth masks with SAM-2** (Ravi et al. 2024) via "manual annotation of key points within the target regions" covering **end-effector + task-relevant objects**.
3. **Binarize** the Grad-CAM map; compute the **Jaccard index** vs the GT mask; **average over the whole evaluation dataset** → "manipulation centricity."

**Exact numbers (fetched):** "a strong correlation between manipulation centricity and downstream task performance, with a **Pearson correlation coefficient of R = 0.93**" (Fig. 2), aggregated over **4 sim domains, 20 tasks** (Robomimic 3, RoboCasa 3, MetaWorld 10, DexArt 4). MCR itself beats the strongest representation baseline by **14.8%** across those benchmarks and lifts 3 real-robot tasks by **76.9%**.

**Gaps / failure modes:**
- The main text does **not** specify the Grad-CAM layer/target or the binarization threshold; and (checked: fetched README) **the released repo contains pre-training code, not the metric code** — you must reimplement from the paper description.
- Same scope caveat as Burns: correlation is **across pretrained representations** (R3M, VIP, MVP, MCR variants…), not across small policy-training deltas.
- Stated failure: embodiment gap — on dexterous-hand tasks, DROID-pretrained variants underperform (gripper-only pretraining data).

### 1.3 Newer (2025–2026): rollout-replacing evaluators rather than rollout-free metrics

The 2025–2026 literature largely gave up on pure representation→success prediction and moved to **cheap surrogate rollouts**:

- **WorldEval** — arXiv [2505.19017](https://arxiv.org/abs/2505.19017) *(abstract fetched)*: turns a video generator into an action-following world model via **Policy2Vec** (latent-action conditioning), then evaluates policies entirely in generated video. Claims: "effectively ranks various robot policies **and individual checkpoints within a single policy**," "strong correlation between policy performance in WorldEval and real-world scenarios," beats real-to-sim. Project: worldeval.github.io.
- **Evaluating Robot Policies in a World Model** — code at [github.com/world-model-eval/world-model-eval](https://github.com/world-model-eval/world-model-eval) *(found via search; paper details UNVERIFIED)*; same thesis, action-conditioned video + success detection. **dWorldEval** (discrete diffusion world model, arXiv 2604.22152) **[UNVERIFIED]**.
- **SureSim** — *Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators*, arXiv [2510.04354](https://arxiv.org/abs/2510.04354) *(abstract fetched)*, Badithela, Snyder, …, Majumdar: formalizes combining **large-scale cheap eval + small-scale expensive eval** as **prediction-powered inference** — paired cheap/expensive evaluations rectify the bias of the cheap evaluator, then non-asymptotic mean-estimation gives confidence intervals on true performance. *This is the statistically correct template for our "DDIM-10 cheap eval + occasional DDPM-100 anchor" scheme (§3, §5).*
- **SIMPLER** — arXiv [2405.05941](https://arxiv.org/abs/2405.05941) *(abstract fetched)*: real-to-sim evaluation; introduced **MMRV (Mean Maximum Rank Violation)** — aggregates worst-case rank violations weighted by the real performance margin — plus Pearson r, as *the* metrics for judging whether a proxy evaluator preserves policy rankings. (MMRV definition itself from the paper via search snippet — **[UNVERIFIED at fetch level]**, but the metric to copy when we validate any proxy.)
- **Demonstration-quality (not policy) screening:** *An Efficient Metric for Data Quality Measurement in Imitation Learning* (arXiv 2605.01544) — frequency-domain signature of control inefficiency, fully offline, no rollouts **[UNVERIFIED]**. Adjacent, useful for dataset triage, not arm ranking.
- **The Temporal Trap** — arXiv [2502.03270](https://arxiv.org/abs/2502.03270) *(abstract fetched)*: pre-trained visual representations are **temporally entangled**; policy success correlates with the latent space capturing *task-progression* cues, and static feature-quality assessments miss this. This is the literature's echo of **our own inverted-probe lesson**: a probe can improve on a static criterion while the property that actually drives closed-loop success (temporal structure) degrades.

### 1.4 Meta-lesson: proxies must be *calibrated*, never assumed

Independent, well-verified evidence that offline losses/probes mis-rank policies:
- **robomimic study** (Mandlekar et al., arXiv [2108.03298](https://arxiv.org/abs/2108.03298), fetched): "policies are trained with surrogate losses," and selecting checkpoints by **lowest validation loss** (or last epoch) instead of online eval makes the selected policy "significantly worse than the best one" — relative drops of **10–100%** (paper text; the project page phrases it as "the best validation policy is 50 to 100% worse"). Also: "policy performance can change significantly from epoch to epoch."
- Burns: ImageNet-probe accuracy — the classic representation probe — was near-useless for ViTs (R²=0.07).
- Both Burns's Jaccard and MCR's centricity were validated **across encoders**, a regime with huge between-model variance. Nothing in either paper licenses using them to rank **small deltas of one architecture** (our typical experiment). Given our local inversion (probes up, success down), the only defensible use is: compute the candidate metric on arms we have *already* fully evaluated, check Pearson + MMRV against known success, and only then admit it as a Stage-0 gate.

---

## 2. Early-epoch ranking: does early performance predict final ranking?

### 2.1 What's established (HPO/bandit literature)

- **Successive Halving** as non-stochastic best-arm identification: Jamieson & Talwalkar, arXiv [1502.07943](https://arxiv.org/abs/1502.07943) *(abstract fetched)* — exploit "the iterative nature of standard machine learning algorithms"; comparable accuracy "an order of magnitude faster than baseline methods."
- **Hyperband**: Li et al., arXiv [1603.06560](https://arxiv.org/abs/1603.06560), JMLR 2018 *(abstract fetched)* — pure-exploration non-stochastic infinite-armed bandit; "over an order-of-magnitude speedup" over baselines; hedges SHA's fixed aggressiveness across brackets.
- **ASHA**: Li et al., arXiv [1810.05934](https://arxiv.org/abs/1810.05934), MLSys 2020 *(abstract via search + arXiv page)* — asynchronous promotion, scales linearly with workers; the practical choice when arms run in parallel across GPUs 0–7.
- **One epoch is often enough:** *The Unreasonable Effectiveness of Early Discarding After One Epoch in NN Hyperparameter Optimization*, Egele et al., arXiv [2404.04111](https://arxiv.org/abs/2404.04111) *(PDF fetched, pp.1–8)*. Findings verbatim: "(1) dynamically allocating resources as done by successive halving or learning curve extrapolation offers minimal (and oftentimes no) utility compared to a constant number of training epochs, and (2) one can often early discard models after only one epoch without losing significant final predictive performance." Caveats (fetched): fully-connected networks on HPOBench-style tabular tasks; the known failure mode is **short-horizon bias** — slow starters get culled (their §2 discusses learning-curve crossings explicitly).
- Rank-correlation folklore in NAS/HPO: Spearman between epoch-t ranking and final ranking "reaches 0.6 only after a few epochs of training and increases steadily" *(from a progressive multi-fidelity evaluation paper surfaced in search — **[UNVERIFIED]**, treat as indicative only)*.

**What fraction of training preserves rank?** There is **no BC-specific published answer**; HPO evidence says rank signal appears very early (ρ≈0.6 within a few epochs, often decision-grade at 25–50% budget with reduction factor 2–4), but that is for supervised losses on i.i.d. data.

### 2.2 The BC-specific complication (verified)

robomimic (arXiv [2108.03298](https://arxiv.org/abs/2108.03298), fetched): success rate **fluctuates significantly across checkpoints**; their protocol is **50 rollouts every 50 epochs (low-dim) / every 20 epochs (image)**, reporting the **max over training**, averaged over 3 seeds. Two consequences:

1. An early gate must be **rollout-based** (success at epoch t), never loss-based — the loss-rank is broken even at convergence (§1.4).
2. A *single* early checkpoint is a noisy draw from a fluctuating curve; band the gate (§4.4) and only kill arms clearly below it, i.e. exactly SHA-with-slack rather than top-k selection. Diffusion-loss curves are even less informative than MSE-BC curves (the loss is a denoising objective, not action accuracy) — another reason to gate on rollouts.

**No published study measures early-epoch → final rank preservation for Diffusion Policy.** We should measure it in-house for free: for every completed 50-epoch arm we already have, re-evaluate saved checkpoints at epochs 10/15/20 with 25 DDIM-10 rollouts and compute Spearman + MMRV against final success. One afternoon of GPU time converts §2.1's folklore into a calibrated local gate. (This doubles as the calibration run for §3.)

---

## 3. Fast diffusion-policy inference for eval

### 3.1 What the original paper actually says (fetched, ar5iv [2303.04137](https://ar5iv.labs.arxiv.org/html/2303.04137))

- Simulation benchmarks: "we used the iDDPM algorithm with the same **100 denoising diffusion iterations for both training and inference**"; **Square Cosine schedule**.
- Real robot: "We used **DDIM** on real-world benchmarks to reduce the inference denoising iterations to **16**"; and "using DDIM with 100 training iterations and **10 inference iterations** enables **0.1 s** inference latency on an Nvidia 3080."
- **No inference-steps-vs-success ablation is reported in the paper.** The DDIM-16/10 settings were shipped for real-robot latency, implying the authors considered the degradation acceptable, but the paper quantifies nothing.

### 3.2 Follow-up numbers (fetched)

**Consistency Policy**, Prasad et al., arXiv [2405.07503](https://arxiv.org/abs/2405.07503) *(HTML fetched)* — the most useful quantified grid (robomimic + Push-T):

| Task | DDPM 100-step | DDiM 15-step | CP 3-step | CP 1-step |
|---|---|---|---|---|
| Lift | 1.00 | – | – | – |
| Can | 0.97 ± 0.01 | 0.82 ± 0.03 | 0.95 ± 0.02 | 0.98 ± 0.01 |
| Square | 0.93 ± 0.02 | 0.85 ± 0.03 | 0.96 ± 0.01 | 0.92 ± 0.02 |
| **Tool Hang** | **0.79 ± 0.03** | **0.14 ± 0.02** | 0.77 ± 0.03 | 0.70 ± 0.03 |
| Push-T | 0.87 ± 0.03 | 0.78 ± 0.03 | 0.84 ± 0.03 | 0.82 ± 0.03 |

Latency: ~1–2 ms vs 11 ms (sim nets), 21 ms vs 192 ms on a laptop GPU real-world (~9×).

Read-through for screening: **reduced-step DDIM costs a few points on easy/medium tasks (Can −15, Square −8, Push-T −9) but can catastrophically collapse on precision tasks (Tool Hang 0.79 → 0.14)**. Whether ranking is preserved is therefore **task-family dependent and must be checked once per task**, exactly the SIMPLER/MMRV calibration of §1.3. (Note: CP's baseline is their EDM-flavored reimplementation; treat absolute values as indicative.)

**Two-Steps Diffusion Policy (GDP)**, arXiv [2510.21991](https://arxiv.org/html/2510.21991) *(HTML fetched)*, Adroit tasks, DDPM sampler, 100 seeds: 100-step baseline 0.68/0.69/0.88/0.87 (Hammer/Relocate/Pen/Door); naive **5-step: 0.85–1.00** (fine, sometimes better); naive **2-step: 0.00–0.13 (collapse)**; their genetic-denoising method recovers 0.91–1.00 at 2 NFE. Confirms: few-step behavior is schedule-sensitive and non-monotone; below ~5 steps vanilla samplers die.

**Distilled one-step policies** (Consistency Policy above; OneDP and similar **[UNVERIFIED]**) match 100-step DDPM within noise — but require an extra distillation training run per arm, which defeats screening. Not recommended for our loop.

### 3.3 Practical recipe (what the evidence supports)

1. **Train unchanged** (100-step DDPM schedule, Square-Cosine) — DDIM decouples train/infer steps by construction (DP paper, fetched).
2. **Evaluate screening rollouts with DDIM 10–16 steps** — the original authors' own real-robot setting.
3. **Calibrate once per task family:** 2 checkpoints × (DDPM-100 vs DDIM-10) × 50 rollouts; require high Pearson and low MMRV before trusting DDIM-10 ranks. Watch for Tool-Hang-like precision tasks.
4. **Never mix samplers across arms in one comparison**; absolute levels shift down a few points, so compare DDIM-10 arms only to DDIM-10 controls (or rectify with SureSim-style paired anchoring, §1.3).
5. Expected eval speedup: denoising dominates rollout compute at 100 steps; 10× fewer steps takes our 50-rollout hour to **~6–10 min** (plus env stepping). Batching envs on one A6000 stacks on top of this.

---

## 4. Rollout-count statistics for ranking policies (the math)

Everything below: exact binomial computation + Wald formulas, Monte-Carlo cross-checked (20k–40k reps). Success counts `X ~ Bin(n, p)`, arms independent.

### 4.1 Precision of a single arm's estimate

SE(p̂) = √(p(1−p)/n); Wilson 95% half-width shown (better than Wald at these n):

| n | p=0.2 | p=0.3 | p=0.5 | p=0.7 |
|---|---|---|---|---|
| 25 | ±0.151 | ±0.169 | ±0.182 | ±0.169 |
| 50 | ±0.109 | ±0.123 | ±0.134 | ±0.123 |
| 100 | ±0.078 | ±0.088 | ±0.096 | ±0.088 |
| 200 | ±0.055 | ±0.063 | ±0.069 | ±0.063 |

**Standing rule this implies: a 50-rollout eval cannot see differences smaller than ~12–13 points; a 25-rollout eval, ~17–18 points.**

### 4.2 Ranking two arms (best-arm selection, no significance claim)

If we just pick the higher p̂ (ties split), the probability the pick is correct is
`P = P(p̂_A > p̂_B) + ½·P(p̂_A = p̂_B)`, computed exactly:

| true gap Δ | base p_B | n=25 | n=50 | n=100 | n=200 |
|---|---|---|---|---|---|
| 0.05 | 0.3 | 0.646 | 0.703 | 0.775 | 0.857 |
| 0.10 | 0.3 | 0.770 | 0.853 | 0.931 | 0.982 |
| 0.15 | 0.3 | 0.864 | 0.940 | 0.986 | 0.999 |
| 0.20 | 0.3 | 0.927 | 0.980 | 0.998 | 1.000 |
| 0.10 | 0.5 | 0.761 | 0.843 | 0.923 | 0.978 |
| 0.20 | 0.5 | 0.927 | 0.980 | 0.998 | 1.000 |

(Base p 0.2–0.5 barely moves these; worst case is p≈0.5.)

**Reading:** with **50 rollouts/arm** you correctly rank a **20-pt gap 98%** of the time and a **15-pt gap 94%**, but a **10-pt gap only ~85%** and 5 pt is a coin flip with lean. To rank 10-pt gaps at ≥98% you need **~200 rollouts/arm**.

### 4.3 Certifying a difference (hypothesis test) is much dearer than picking a winner

One-sided two-proportion test, α=0.05; required n **per arm** for 80% power:
`n = (z_α + z_β)² (p_A q_A + p_B q_B) / Δ²`, z_0.05=1.645, z_0.20=0.842:

| comparison | power @ n=25 | n=50 | n=100 | n/arm for 80% power |
|---|---|---|---|---|
| 0.3 vs 0.4 | 0.18 | 0.28 | 0.44 | **279** |
| 0.3 vs 0.45 | 0.30 | 0.47 | 0.72 | **126** |
| 0.3 vs 0.5 | 0.43 | 0.67 | 0.90 | **72** |
| 0.3 vs 0.6 | 0.72 | 0.94 | ~1.00 | **31** |
| 0.5 vs 0.6 | 0.18 | 0.26 | 0.41 | **303** |
| 0.5 vs 0.7 | 0.43 | 0.67 | 0.90 | **72** |

**Screening should be framed as selection (§4.2) with a kill band, not as significance testing — testing 10-pt claims costs ~280 rollouts/arm.** Reserve testing for final confirmations.

### 4.4 Fixed-n kill gate against a known control level

Control success p₀ known precisely from accumulated history; an arm is "interesting" only if p ≥ p₁. Rule: run n rollouts, **kill if successes ≤ k**, with k the largest integer s.t. `P(Bin(n,p₁) ≤ k) ≤ 0.05` (false-kill of a genuinely good arm ≤5%, exact):

| p₀ (dead) | p₁ (works) | n | kill if ≤ k/n | false-kill(p₁) | false-pass(p₀) |
|---|---|---|---|---|---|
| 0.30 | 0.50 | 25 | 7/25 | 0.022 | 0.488 |
| 0.30 | 0.50 | 50 | 18/50 | 0.032 | 0.141 |
| 0.30 | 0.45 | 50 | 16/50 | 0.043 | 0.316 |
| 0.30 | 0.45 | 100 | 36/100 | 0.043 | 0.080 |
| 0.20 | 0.35 | 50 | 11/50 | 0.034 | 0.289 |
| 0.40 | 0.50 | 100 | 41/100 | 0.044 | 0.377 |

Asymmetry is deliberate: **gates protect good arms (≤5% false-kill) and tolerate letting mediocre arms continue** (they die at the next, cheaper-to-afford stage). This matches early-kill discipline: kill only on decisive evidence.

### 4.5 Sequential testing — Wald SPRT (stop as soon as the answer is clear)

H₀: p=p₀ vs H₁: p=p₁ (indifference-zone framing). After each rollout add to the log-likelihood ratio:
- success: `ln(p₁/p₀)` ; failure: `ln((1−p₁)/(1−p₀))`
- accept H₁ when LLR ≥ a = ln((1−β)/α); accept H₀ when LLR ≤ b = ln(β/(1−α)).

Wald's expected sample sizes, `E[N|H] ≈ (P_accept·a + P_reject·b)/E[Z|H]` with `E[Z|p] = p·ln(p₁/p₀) + (1−p)·ln((1−p₁)/(1−p₀))`; Monte-Carlo (20k reps) confirms:

| p₀ vs p₁ | α=β | E[N|H₀] (Wald / MC) | E[N|H₁] (Wald / MC) | MC error rates | E[N] at midpoint p |
|---|---|---|---|---|---|
| 0.30 / 0.50 | 0.05 | 32.2 / 34.4 | 30.4 / 33.1 | 0.040 / power 0.956 | 56.6 |
| 0.30 / 0.50 | 0.10 | 21.4 / 24.1 | 20.2 / 23.0 | 0.082 / 0.914 | 33.4 |
| 0.30 / 0.45 | 0.05 | 56.2 / 58.8 | 53.2 / 57.1 | 0.044 / 0.956 | 96.4 |
| 0.20 / 0.40 | 0.05 | 29.0 / 31.2 | 25.3 / 28.3 | 0.038 / 0.956 | 49.7 |
| 0.50 / 0.70 | 0.05 | 30.4 / 33.0 | 32.2 / 34.2 | 0.043 / 0.961 | 56.6 |
| 0.40 / 0.50 | 0.05 | 131.6 / 134.3 | 129.8 / 133.2 | 0.045 / 0.929 | 198.2 |

**Reading:** for **20-pt bands** the SPRT decides in **~30 rollouts on average** at α=β=0.05 (vs 72/arm fixed-n for the same power) — a ~2.2× saving; clear-cut arms stop far sooner. The pain point is arms sitting **inside** the band (midpoint column) — cap N.

**Truncated SPRT (N_max = 50; at truncation accept H₁ iff p̂ ≥ (p₀+p₁)/2)** — realized MC properties:

| p₀ / p₁ | α=β nominal | E[N|H₀] | realized type I | E[N|H₁] | realized power |
|---|---|---|---|---|---|
| 0.30 / 0.50 | 0.05 | 29.9 | 0.087 | 29.1 | 0.929 |
| 0.30 / 0.50 | 0.10 | 22.8 | 0.109 | 21.5 | 0.906 |
| 0.30 / 0.45 | 0.05 | 40.4 | 0.145 | 39.0 | 0.872 |
| 0.50 / 0.70 | 0.05 | 29.1 | 0.105 | 29.9 | 0.941 |

I.e., with a hard 50-rollout cap and a 20-pt band you get a ~0.09/0.93 error/power operating point at **E[N]≈30**. Bands of 15 pts under a 50-cap degrade to ~0.15 error — don't promise more than the cap can deliver.

Classic references: Wald's SPRT (1945) — standard math, shown above, no fetch needed; **Hoeffding races** (Maron & Moore, NIPS 1993) and **F-Race** (Birattari et al., GECCO 2002) are the racing-algorithm ancestors **[UNVERIFIED — classic citations, not fetched]**; *Deep RL at the Edge of the Statistical Precipice* (arXiv 2108.13264) for small-sample cross-seed reporting (IQM + bootstrap CIs) **[UNVERIFIED]**.

### 4.6 Robotics-specific sequential evaluation (2024–2026, fetched)

**STEP** — *Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping*, Snyder et al. (TRI/Princeton), arXiv [2503.10966](https://arxiv.org/abs/2503.10966) *(HTML fetched)*, RSS 2025. Code/site: [tri-ml.github.io/step](https://tri-ml.github.io/step/).
- Test: composite H₀: p₁ ≤ p₀ vs H₁: p₁ > p₀ on **paired** trials (one rollout of each policy per round); no known p₀ needed.
- Construction: **alpha-spending** budget f(n) = α\*/N_max across a pre-specified horizon; rejection regions in the cumulative-success state space computed by **linear programming** against worst-case nulls → exact finite-sample **Type-I control at α\***; near-oracle-SPRT sample efficiency. Not anytime-valid beyond N_max (unlike SAVI/betting confidence sequences, one of its fetched baselines alongside Lai's asymptotically optimal test and Barnard's exact batch test — the paper shows naive sequential reuse of batch tests violates Type-I).
- Fetched numbers: 52-pt real gap (28% vs 80%) → decided in **8 trials/policy** of a 50 budget; 36-pt gap (56% vs 92%) → **19 trials**; 16.4-pt sim gap (40% vs 56.4%) → **131 trials** (oracle SPRT 128, Lai 125); "reduces the required number of trials by **up to 32%**" vs feasible sequential baselines; >160 trials saved certifying across three sim tasks. Note how well 131-for-16-pt matches our SPRT table (§4.5: 0.30/0.45 midpoint ≈ 96–130) — the binomial math and the robotics literature agree.

**Vincent et al.** — *How Generalizable Is My Behavior Cloning Policy? A Statistical Approach to Trustworthy Performance Evaluation*, arXiv [2405.05439](https://arxiv.org/abs/2405.05439) *(HTML fetched)*, RA-L 2024. Code: [github.com/TRI-ML/binomial_cis](https://github.com/TRI-ML/binomial_cis), [github.com/TRI-ML/stochastic_verification](https://github.com/TRI-ML/stochastic_verification).
- For binary success: **randomized uniformly-most-accurate (UMA) lower confidence bound** on p (tighter than Clopper–Pearson); for continuous returns: one-sided exact KS-based CDF bound (beats DKW's ε=√(−ln α / 2n)).
- Explicit triad: fix two of {confidence 1−α, tightness, n} → get the third. Their experiments use **40–50 rollouts per policy**; policy comparison via **disjoint one-sided bounds at 0.975 each → joint 95%** ("the chances we come to this conclusion incorrectly are ≤5%"). Use this at Stage 3 (certification), not for screening — disjoint-CI comparison is conservative relative to STEP.

---

## 5. Proposed screening protocol (components + guarantees, all cited)

**Decision target:** per-arm verdict in ~3–5 h; protect truly-good arms (false-kill ≤ ~10% overall); tolerate mediocre arms surviving one extra stage. Effect sizes we act on: ≥15–20 pts at screening; 10 pts only at confirmation. (With p in 0.2–0.7 and ≤50-rollout stages, 5-pt effects are **undecidable** — §4.1–4.2 — and must be batched into confirmation-scale evals or dropped.)

**Stage 0 — offline pre-filter (encoder/representation arms only).**
Compute Burns CLS-attention Jaccard (ViT encoders; repo above) and/or MCR-style Grad-CAM∩SAM-2 centricity (reimplement; §1.2) **on the candidate encoder, before any training**. *Gatekeeping rule:* usable only after in-house calibration — Pearson + MMRV (SIMPLER, [2405.05941](https://arxiv.org/abs/2405.05941)) against ≥8–10 of our own completed arms; given our probe inversion and the Temporal Trap result ([2502.03270](https://arxiv.org/abs/2502.03270)), an uncalibrated offline metric ranks nothing. Basis: Burns R²=0.85/ρ=0.90 (ViT, OOD) [2312.12444]; MCR Pearson R=0.93 [2410.22325]; both across-encoder only.
*Cost: ≤1 h. Guarantee: none intrinsically — inherits the calibration's measured MMRV.*

**Stage 1 — early-epoch gate (ASHA-style, rollout-based).**
Train all arms; at **epoch ~15/50 (~30% budget ≈ 2.7–4.5 h)** run **25 rollouts with DDIM-10** per arm. Kill an arm iff successes fall at/below the §4.4 threshold for (p₀ = control-level, p₁ = control + 0.15–0.20) — e.g. control≈0.3: **kill at ≤7/25** vs p₁=0.5 (false-kill 2.2%) — else continue. Rationale: SHA/ASHA/Hyperband resource allocation ([1502.07943](https://arxiv.org/abs/1502.07943), [1603.06560](https://arxiv.org/abs/1603.06560), [1810.05934](https://arxiv.org/abs/1810.05934)); very-early discarding is empirically near-free in HPO ([2404.04111](https://arxiv.org/abs/2404.04111)); gate on rollouts not loss because BC losses mis-rank (robomimic [2108.03298](https://arxiv.org/abs/2108.03298)) and success fluctuates across epochs (ibid. — hence band + only-clear-kills). *One-off calibration:* Spearman/MMRV of epoch-15 vs final ranks on our historical checkpoints (§2.2) — no published BC number exists; we must own this constant.
*Guarantee (exact binomial): P(kill | arm is ≥p₁ at this epoch) ≤ 5%; short-horizon bias risk borne knowingly and checked by the calibration.*

**Stage 2 — end-of-training screening eval (sequential, reduced-step).**
Survivors train to 50 epochs (evaluate the last ~3 saved checkpoints' mean or the best-of-3 to blunt checkpoint noise, per robomimic's max-over-training practice). Evaluate with **DDIM-10/16** (DP's own real-robot setting, [2303.04137]; per-task calibration per §3.3 — beware Tool-Hang-type collapse [2405.07503]) under a **truncated SPRT** (N_max=50, band = control ± 20 pts, α=β=0.05 nominal → realized ≈0.09/0.93 at E[N]≈30, §4.5), or — for paired arm-vs-control comparisons and exact finite-sample Type-I — **STEP** with α\*=0.05, N_max=50 ([2503.10966], code at tri-ml.github.io/step). Expected cost: **~30 rollouts ≈ 4–6 min** at DDIM-10; decisive arms stop after ~8–20 (STEP's hardware numbers).
*Guarantee: Type-I ≤ α\* exactly (STEP LP construction) or ≈0.09 (truncated SPRT, MC-measured); power ≈0.93 for 20-pt effects.*

**Stage 3 — confirmation (winners only, ~1 in 4–8 arms).**
**100–200 rollouts × 2 seeds**, full-fidelity sampler (DDPM-100, or DDIM if the Stage-2 calibration showed rank+level equivalence; if levels differ, rectify with a SureSim-style paired anchor set, [2510.04354]). Report **Wilson or UMA lower bounds** ([2405.05439], `binomial_cis`); compare vs control via disjoint 0.975 one-sided bounds (joint 95%) or a fixed-n one-sided test. 100/arm certifies Δ=0.2 with power 0.90; 200/arm ranks Δ=0.1 at 98% (§4.2–4.3). Seed-to-seed variance is *not* binomial — keep ≥2 seeds here and report per-seed (cf. Precipice **[UNVERIFIED]**).
*Guarantee: ≤5% joint error on the superiority claim; CI tightness per §4.1 (±0.07 at n=200).*

**Composition (union bound):** P(a truly ≥+20-pt arm is killed anywhere) ≤ 0.05 (S1) + 0.07 (S2) ≈ **12%**, at an expected per-arm cost of **~3–5 h** for kills (which are most arms) instead of 10–16 h; false survivors are cleaned up at Stage 3 for ≤5%. Tune the split (e.g., S1 at 2.5%) if arms are expensive to regenerate.

**Standing invariants**
1. Same sampler/steps/env-set/max-horizon for every arm inside a comparison; change nothing mid-comparison (STEP's alpha-spending assumes the pre-registered N_max; extending a finished batch test is p-hacking — shown formally in [2503.10966]).
2. Every proxy (offline metric, early epoch, reduced steps, world-model eval) gets a Pearson + MMRV calibration against full-fidelity results **before** it gates anything, and a drift re-check when the task family changes. This is the transferable core of SIMPLER/SureSim — and the codified version of our probe-inversion lesson.
3. Decisions <10 pts are out of scope for screening; don't let a 25-rollout number settle a 5-pt argument (§4.1).

---

## Source ledger

| Claim cluster | Source | Fetched? |
|---|---|---|
| Jaccard CLS-attention metric, R²=0.85/ρ=0.90 (ViT, OOD), protocol, Fig. 5/6, ALOHA task | arXiv [2312.12444](https://arxiv.org/abs/2312.12444) + PDF pp.5–9 + [project page](https://kayburns.github.io/segmentingfeatures/) + [repo](https://github.com/stanford-iris-lab/segmenting_feats/tree/eval) | Yes (PDF pages read) |
| MCR centricity recipe, Pearson R=0.93, 4 domains/20 tasks, +14.8%/+76.9% | arXiv [2410.22325](https://arxiv.org/html/2410.22325v2) + [project](https://robots-pretrain-robots.github.io/) + [repo README](https://github.com/luccachiang/robots-pretrain-robots) | Yes (metric code absent from repo — checked) |
| robomimic: 50 rollouts/ckpt, eval every 20–50 epochs, max-over-training, val-loss selection 10–100% worse, epoch-to-epoch fluctuation | arXiv [2108.03298](https://ar5iv.labs.arxiv.org/html/2108.03298) + [study page](https://robomimic.github.io/study/) | Yes |
| DP: DDPM-100 sim, DDIM-16 real, DDIM-10 = 0.1 s, Square-Cosine, no step ablation | arXiv [2303.04137](https://ar5iv.labs.arxiv.org/html/2303.04137) | Yes |
| CP: DDPM-100 vs DDiM-15 vs CP-1/3 table incl. Tool Hang 0.79→0.14 | arXiv [2405.07503](https://arxiv.org/abs/2405.07503) | Yes |
| DDPM 5-step OK / 2-step collapse (Adroit) | arXiv [2510.21991](https://arxiv.org/html/2510.21991) | Yes |
| SHA / Hyperband / ASHA / 1-Epoch discarding | arXiv [1502.07943](https://arxiv.org/abs/1502.07943), [1603.06560](https://arxiv.org/abs/1603.06560), [1810.05934](https://arxiv.org/abs/1810.05934), [2404.04111](https://arxiv.org/abs/2404.04111) | Yes (2404.04111 PDF pages read) |
| STEP: alpha-spending LP test, exact Type-I, 8/19/131-trial examples, ≤32% savings | arXiv [2503.10966](https://arxiv.org/html/2503.10966) + [site](https://tri-ml.github.io/step/) | Yes |
| Vincent: UMA bounds, KS CDF bounds, 40–50 rollouts, disjoint-CI comparison | arXiv [2405.05439](https://arxiv.org/html/2405.05439) + [binomial_cis](https://github.com/TRI-ML/binomial_cis) | Yes |
| WorldEval, SIMPLER, SureSim, Temporal Trap abstract-level claims | arXiv [2505.19017](https://arxiv.org/abs/2505.19017), [2405.05941](https://arxiv.org/abs/2405.05941), [2510.04354](https://arxiv.org/abs/2510.04354), [2502.03270](https://arxiv.org/abs/2502.03270) | Yes (abstracts) |
| MMRV formula details; NAS ρ≈0.6-after-few-epochs; Hoeffding races; F-Race; Precipice; dWorldEval; 2605.01544; OneDP | various | **UNVERIFIED** (marked inline) |
| All §4 tables | computed here (exact binomial + Wald + MC), `/tmp/gate_math.py` | Yes (MC-validated) |
