# v0.5 — Robustness leg (draft section)

**Scope.** Kepler = 8 learnable queries cross-attending a frozen LeJEPA ViT-B/16 patch grid → 8×256
continuous tokens, consumed frozen by (a) a diffusion-policy BC chassis on robomimic Square (200-demo
PH substrate) and (b) latent MPC on the DINO-WM PushT harness. This section covers the shift-robustness
claim only. Planner exploitation is the other half of v0.5.

**Tag legend.** `[VERIFIED]` = real URL + verbatim quote checked this cycle. `[REFUTED]` = claim was
checked and does **not** survive as stated; a corrected form may still be usable and is given.
`[UNVERIFIED]` = plausible, no primary-source check yet — do not put in a paper. `[OURS]` = our own
measurement, internal, not externally checked.

**Cost legend.** `[EVAL-ONLY]` no new weights at all. `[POST-HOC]` new small module or test-time update,
frozen encoder untouched. `[RESAMPLER-RETRAIN]` re-pretrain the 8-query head over the same frozen
backbone. `[ENCODER-RETRAIN]` full Kepler re-pretrain. `[WM-RETRAIN]` dynamics model retrain.

---

## 1. Bottom line

**No — we cannot get a robustness claim that holds on all axes, and we should stop trying.** The axis
asymmetry we measured is a tabulated, field-wide property with a published mechanism (LIBERO-Plus:
camera −37.4pp vs light −11.3pp for OpenVLA-OFT, −78.5pp vs −2.8pp for its third-person variant
`[VERIFIED]`), and the only interventions the literature shows reliably fixing the geometric axis are
geometric canonicalization or a geometry-pretrained backbone — both architecture changes we have ruled
out for this cycle. **The single highest-value change is to reframe the deliverable from "our tokens are
robust" to a calibrated, attributed trade-off curve, and to run one eval-only matrix that produces it:
shift-induced degradation × 4 representation widths (raw patch grid / q16 / q8 / q4) × 4 axes on the
checkpoints we already have.** That one matrix decides whether our camera loss lives in the frozen
backbone (in which case only input canonicalization can help) or in the 8-slot bottleneck (in which case
a locality-prior resampler is the fix), and it simultaneously yields the publishable claim: *compactness
buys dynamics learnability (rollout-MSE ratio 0.494 at q8 vs 0.726 dense `[OURS]`) and pays for it in
geometric robustness*, anchored at one end by DINO-WM's own patch-vs-CLS number (0.90 vs 0.44
`[VERIFIED]`). **Two honest problems must be fixed before any of it is quotable:** our four cells are
severity-uncalibrated across axes and our win/loss cells sit at opposite ends of the success range —
the +15.5pp lighting win is measured where the ResNet baseline has already collapsed to 0.14–0.22, while
the −18.7pp camera loss is measured where the ResNet baseline is still at 0.74 `[OURS]`. **And our own
battery already falsifies our own slogan:** we lose −13pp on distractors, an axis the field calls one of
the *easiest* `[REFUTED-as-stated but directionally safe]`, so "photometric win / geometric loss" is the
wrong frame. The right frame is *which regions get pooled*, developed next.

---

## 2. Why we win photometrically and lose geometrically

### 2.1 The split itself is not novel — say so and cite it

`[VERIFIED]` **LIBERO-Plus** (https://arxiv.org/abs/2510.13626) perturbs 10 open VLA checkpoints across
7 axes over 10,030 evaluation tasks and states the mechanism we would otherwise be claiming as ours:

> Finding 2: Robustness varies considerably by perturbation type. Models are most vulnerable to changes
> in camera viewpoint and robot initial state, which require a high-level understanding of spatial
> geometry and proprioception. In contrast, they show relative resilience to lighting and background
> variations, which constitute more superficial, low-level visual changes.

Per-axis absolute drops: OpenVLA-OFT camera −37.4pp vs light −11.3pp, background −4.7pp; the
third-person-only variant OpenVLA-OFT_w camera −78.5pp vs background −2.8pp (28× asymmetry). Across
models camera −19.1 to −90.9pp, background −2.8 to −64.6pp, light −6.0 to −72.1pp. `[POST-HOC]`
eval-only benchmarking.

**Consequence for the paper:** our contribution is *not* "we discovered an asymmetry". It is "we
reproduce the field-wide asymmetry at the frozen-representation level, and we can attribute it". That is
a smaller claim but a defensible one, and it converts our camera/distractor losses from "our encoder is
broken" into "this is what consuming a captured view costs".

`[VERIFIED]` **COLOSSEUM** (https://arxiv.org/pdf/2402.08191) supplies the architectural half of the
mechanism and the only cure the benchmark literature reports:

> We observe that these model are robust to changes in Camera_Pose, because they do not directly learn
> on captured view. They instead preprocess the input RGBD views into a voxel grid or re-rendered novel
> views.

2D RGB-input models (R3M-MLP, MVP-MLP) list camera pose among their most damaging factors; 3D models
(RVT, PerAct) are robust because they canonicalize geometry. Kepler consumes the captured view, so it
inherits the 2D-PVR failure mode by construction. Write that as scoping, not as a defect — it pre-empts
"why don't you just fix it".

### 2.2 The frame we should actually use: fixed-budget competitive pooling

Our two losses are camera shift (−18.7pp) **and distractors** (−13pp) `[OURS]`. Distractors are not a
geometric axis. FactorWorld's corrected ordering puts distractors, background and lighting in the
**easier** set `[REFUTED-as-originally-stated]`, so we are losing on an axis the field finds easy. A
"photometric easy / geometric hard" story cannot hold both of our losses. One mechanism does:

> **Both losses are failures of *which regions the 8 queries pool*, not of what the backbone features
> encode.** Camera shift moves the salient regions relative to the queries' learned pooling pattern;
> distractors *add* salient regions that compete for a fixed 8-slot budget. Photometric shift changes
> neither — the same regions stay the most salient — which is exactly where we win.

Three verified sources support this reading, and it is testable eval-only.

`[VERIFIED]` **Honeybee** (https://arxiv.org/html/2312.06742v2) measures that a query-resampler — Kepler's
exact abstraction — is worse at spatial understanding than a locality-prior abstractor at matched token
count (Resampler 43.9 vs C-Abstractor 53.5 at M=144), attributing it to

> the abstraction process lacking a locality-aware design

and noting the resampler summarizes only a few regions. That is the "few slots, chosen by saliency"
property our mechanism needs. `[ENCODER-RETRAIN]` for the fix, but the *diagnosis* is free.

`[VERIFIED]` **Burns et al.**, CoRL 2024
(https://kayburns.github.io/segmentingfeatures/static/segmentingfeatures_paper.pdf) give a training-free
predictor of OOD robustness for frozen ViTs — the Jaccard index of the interpolated `[CLS]` attention map
(emergent segmentation):

> There is a strong positive correlation between Jaccard index and OOD performance both in terms of rank
> correlation and the coefficient of determination. These results suggest that while shape-bias may not
> be predictive of the OOD generalization ability of a pre-trained ViT, the segmentation ability is a
> predictive alternative.

High-Jaccard models also degrade least as distractor count rises (1/3/9 YCB objects); ImageNet
linear-probe accuracy and cue-conflict shape bias do **not** predict OOD success for ViTs. Real-robot
ALOHA confirmation with 50 demos: MoCo-v3 40% vs MVP 0%. `[POST-HOC]`/`[EVAL-ONLY]`.

`[VERIFIED]` **DINO-WM** (https://arxiv.org/html/2411.04983v2) — our own harness — states the compression
end of the trade-off:

> patch-based representations better capture spatial information, in contrast to models like R3M, ResNet,
> and DINO CLS, which reduce observations to a single global feature vector, losing crucial spatial
> details necessary for manipulation tasks

with PushT success 0.90 (patch) vs 0.44 (DINO CLS) vs 0.42 (R3M). Our 8×256 tokens sit between those two
poles, which is precisely why a token-budget sweep is the right measurement. `[EVAL-ONLY]`.

### 2.3 The competing hypothesis, and why it must be tested first

The camera loss could instead live in the **backbone**: frozen visual foundation models are
view-consistent but not 3D-consistent. Banani et al., CVPR 2024
(https://openaccess.thecvf.com/content/CVPR2024/papers/Banani_Probing_the_3D_Awareness_of_Visual_Foundation_Models_CVPR_2024_paper.pdf,
arXiv 2404.08636v1) conclude the representations are "view-consistent, not 3D consistent", and report
that cross-view dense correspondence "quickly deteriorates for larger viewpoint changes" and that for
wide-baseline matching "Although DINOv2 performs better than the other models, the absolute performances
for all models are very low". **Important:** the specific per-bin numbers previously circulated in our
notes (≈70–90% at 0–30° collapsing to ≈10–30% at 90–120°) are `[REFUTED]` — Figure 6 is curves only, with
no numeric table, so those percentages are not citable. The paper also explicitly offers view-dependence
only as a hypothesis ("One possibility is that the models are learning view-dependent representations")
against a rival "good image models" explanation, probes dense patch grids rather than compact query
bottlenecks, and does not include LeJEPA. So it **corroborates by analogy** and does not measure our
−18.7pp or our view-vs-content ratio of 1.46.

Backbone-caused and bottleneck-caused camera loss imply different fixes (canonicalization vs resampler
redesign). Section 3.1 separates them for free.

### 2.4 What is genuinely ours

- The view-vs-content ratio 1.46 on a **compact 8-token** bottleneck `[OURS]`. No source we checked
  measures compact query-token view entanglement; Banani and Honeybee are analogies.
- The **coexistence** of higher dynamics learnability and worse geometric robustness in the same tokens
  (rollout-MSE ratio 0.494 q8 / 0.474 q4 / 0.469 q16 vs 0.726 dense) `[OURS]`. Nothing we checked reports
  this pairing.
- Losing on the distractor axis while winning photometrically `[OURS]` — anomalous relative to
  FactorWorld's ordering, and the single most interesting thing in the leg if the pooling mechanism
  explains it.

---

## 3. Ranked interventions

Ordered by (evidence strength) / (cost to us). Ranks 1–4 are runnable this cycle; 5–8 are v0.6 candidates
with the evidence recorded so we don't re-derive it.

### Rank 1 — Attribution + token-budget matrix (the publishable core)

**What.** One eval-only matrix: representation width × shift axis. Widths = raw LeJEPA patch grid, q16,
q8, q4 (artifacts on disk: `/home/menlo/brain/ishneet/dino_wm/plan_outputs/hybrid_q4`,
`.../hybrid_q16`; verify these are encoder variants and not predictor-only before quoting them). Axes =
lighting, colour, camera, distractor, plus a combined cell. Two dependent variables per cell:
(i) downstream success, (ii) representation drift ‖z(clean) − z(shifted)‖ normalized per arm, plus
open-loop rollout MSE ratio for the dynamics side.

**Evidence.** DINO-WM patch 0.90 vs CLS 0.44 vs R3M 0.42 `[VERIFIED]`
(https://arxiv.org/html/2411.04983v2) anchors the wide end; COLOSSEUM's captured-view mechanism
`[VERIFIED]` (https://arxiv.org/pdf/2402.08191) predicts the sign; LIBERO-Plus per-axis drops
`[VERIFIED]` (https://arxiv.org/abs/2510.13626) give free external reference deltas to plot against.

**Cost.** `[EVAL-ONLY]`. No new weights. This is the cheapest thing in the leg and the only one that is
load-bearing for the paper regardless of outcome.

**Exactly what we run.** For each width, freeze everything, encode clean and shifted observation pairs,
report (a) per-axis success delta vs the ResNet chassis, (b) per-axis drift, (c) rollout-MSE ratio. Then
the attribution test: if the **raw patch grid** drifts under camera shift as much as q8 does (normalized),
the loss is in the backbone → only Rank 3 (canonicalization) can help and Ranks 5–7 are dead. If q8
drifts substantially more than the patch grid, the loss is in the bottleneck → Ranks 5–7 become live.
Prediction from the pooling mechanism: **distractor** loss shrinks monotonically with token budget
(q16 < q8 < q4 in loss magnitude) while the photometric **win** is budget-insensitive.

**Kill gate.** If camera-shift degradation is flat across patch-grid/q16/q8/q4 *and* rollout-MSE ratio is
flat (which we already know it is — learnability saturates by 4 tokens `[OURS]`), there is no trade-off
curve, only two independent facts. Then we report the asymmetry as a reproduction of LIBERO-Plus Finding
2 at representation level, drop the trade-off framing, and do not spend a second wave.

### Rank 2 — Emergent-segmentation (Jaccard) probe on the frozen tokens

**What.** Compute Burns et al.'s Jaccard index of interpolated attention maps for (a) the frozen LeJEPA
ViT-B patch grid and (b) each of Kepler's 8 query attention maps. Zero training, zero rollouts.

**Evidence.** `[VERIFIED]` https://kayburns.github.io/segmentingfeatures/static/segmentingfeatures_paper.pdf
— Jaccard correlates strongly with OOD success in both R² and Spearman ρ, while ImageNet accuracy and
shape bias do not for ViTs; high-Jaccard models degrade least as distractors go 1→3→9; real-robot ALOHA
MoCo-v3 40% vs MVP 0%. Same paper's headline negative result is also useful to us: manipulation-pretrained
PVRs do not generalize better than plain ImageNet models under lighting/texture/distractor shift, the best
manipulation-designed model ranks 7th of 15, and MVP ViT-S (HOI) collapses 39.86 → 6.63 average success —
so "frozen robot-pretrained features are robust" is **not** the field's null, and our photometric win is a
real claim rather than a truism.

**Cost.** `[EVAL-ONLY]`.

**Exactly what we run.** Per-query Jaccard on clean frames; then per-query attention-map IoU between clean
and shifted frames for each axis; then attention-mass on distractor pixels as distractor count rises. Two
payoffs: a published-predictor explanation for the photometric win, and a direct test of the pooling
mechanism — if distractors steal query attention mass, our −13pp is a pooling artifact, not a
feature-quality one. Free interpretability figure for the paper either way.

**Kill gate.** If Kepler's queries score *high* Jaccard and their attention maps are *stable* under both
camera shift and distractors, the pooling mechanism is falsified; fall back to the backbone hypothesis
(§2.3) and the paper keeps only Rank 1's descriptive curve.

### Rank 3 — Input canonicalization, oracle-gated

**What.** Stop trying to make the 8 tokens view-invariant. Put a transform in front of the frozen encoder
that maps the shifted observation back to the training viewpoint. Run the **oracle first**.

**Evidence — the learned version.** `[VERIFIED]` Prior-regularized learned canonicalization
(https://arxiv.org/abs/2310.01647): a small equivariant network in front of a **completely frozen** large
model, trained with the prior loss alone — no task loss, no backprop through the frozen network:

> However, we observe that the produced canonical orientations can be misaligned with those of the
> training distribution, hindering performance. Using dataset-dependent priors to inform the
> canonicalization function, we are able to make large pretrained models equivariant while maintaining
> their performance.

Frozen MaskRCNN C4-averaged mask mAP 27.67 → 44.50 while original-val mAP only drops 45.57 → 44.51, with
a 1.9M-parameter canonicalizer (4% of MaskRCNN's 46.4M). Frozen SAM 58.78 → 62.13 C4-Avg mAP. Two
documented failure modes map straight onto our risk: an under-expressive 0.2M canonicalizer collapses
in-distribution mAP to 35.77 (−9.8), and naive (non-prior-regularized) canonicalization costs
in-distribution accuracy (CIFAR10/ResNet50 96.97 → 93.29) because it feeds the frozen network
out-of-distribution poses.

**Evidence — the re-render version.** `[VERIFIED]` Test-time novel-view synthesis back to the training
pose (https://arxiv.org/html/2603.05868v1) — frozen policy, no demos, no finetuning:

> our method requires no additional robot demonstrations or policy fine-tuning

LIBERO large-camera-perturbation success 39.9% (base π0.5) → 94.5% average, at 36.55 ms / 27 FPS on one
RTX 4090. **Caveat to pre-register:** it needs approximate test-time extrinsics, and the same paper reports
that the frozen-geometry-encoder route (GeoAwareVLA, i.e. Rank 8's family) *collapses below 10%* under
wrist-camera perturbation — so Ranks 3 and 8 disagree, and if we run both we must report the disagreement.

**Cost.** Oracle version `[EVAL-ONLY]`; learned canonicalizer `[POST-HOC]` (1.9M-scale module, no encoder
retrain, no gradients through the frozen ViT).

**Exactly what we run, in order.**
1. **PushT analytic oracle (≈1 day, eval-only).** DINO-WM's PushT is an orthographic 2D pygame render at
   512×512 with no camera model at all (`env/pusht/pusht_env.py:110-115`), so our "camera shift" there is
   necessarily a synthetic image warp — which means the inverse warp is **exactly known**. Apply a
   homography perturbation, then canonicalize with its exact inverse, and measure recovered success. This
   is the oracle control the memory rule demands ("oracle controls matter"; our earlier PushT oracle scored
   only 0.198 on the old harness, which is why we migrated). State the limitation plainly: a planar
   homography has no parallax or occlusion change, so PushT can only give an upper bound.
2. **Square with known extrinsics (eval-only).** robomimic Square has a real 3D camera, so re-render the
   shifted view to the training extrinsics using the simulator's ground-truth pose. That is the honest
   oracle for the geometric axis.
3. **Only if 1–2 pass:** train the 1.9M prior-regularized canonicalizer (or drop in a feed-forward NVS
   module) and re-measure, plus the in-distribution tax cell.

**Kill gate.** If the **exact-inverse oracle** does not recover most of the −18.7pp camera loss on Square,
canonicalization is dead for us and we do not train a canonicalizer. Second gate: if in-distribution Square
success drops more than ~2pp under canonicalization, kill — the whole publication bar is "maintained, never
improved", and 2310.01647's own CIFAR10 number (−3.7pp) shows this failure is real.

### Rank 4 — PAD-style inverse-dynamics test-time adaptation

**What.** Keep optimizing Kepler's **existing** inverse-dynamics auxiliary head on the test stream at
deployment. No reward, no labels, no new objective — we pretrained with this head already.

**Evidence.** `[VERIFIED]` https://arxiv.org/abs/2007.04309 — improves generalization in 31 of 36
environments; real Kinova arm push +24pp under a patterned table cloth; simulated joint-dynamics shift
push(all) 48% → 76%:

> Consistent with the real robot results in Section 5, PAD is found to be most effective when changes in
> dynamics are non-trivial, improving by as much as 28% in the push (all) setting, where all 3
> environmental changes are considered jointly.

Negatives from the same paper, all of which scope us: it is mildly positive or slightly harmful where the
baseline is already unaffected (cartpole distractors 776 → 771 and 964 → 960; push(mount) 86% → 84%); the
offline variant that adapts on a single observation and forgets the update is much weaker; and the SSL task
must match the task family (inverse dynamics fails on navigation, where rotation prediction works).

**Cost.** `[POST-HOC]`, but be honest in the writeup: the encoder weights *change at test time*, so this is
not "frozen checkpoint" in the strict sense. The policy stays frozen.

**Exactly what we run.** Online sequential adaptation of the resampler (not the ViT) on the test episode
stream using the inv-dyn head, on the two axes where we currently lose. Their scoping predicts gains on
dynamics/geometry shift and nothing on photometric shift — which is fine, since photometric is where we
already win and there is no headroom.

**Kill gate.** Rung 0 = camera severity 1 on Square, 3 seeds. If adapted success does not beat frozen by
≥5pp, kill. Second gate: if the adapted encoder's *clean* success drops (the space-contraction signature
of our failed dropout retrain), kill immediately — SAR (https://arxiv.org/abs/2302.12400) documents exactly
this collapse mode for test-time updates.

### Rank 5 — Locality-prior resampler (C-Abstractor / D-Abstractor)

**What.** Replace the plain cross-attention resampler with a convolutional (C-Abstractor) or
deformable-attention-with-reference-points (D-Abstractor) head, keeping 8 output tokens and the same
frozen LeJEPA features.

**Evidence.** `[VERIFIED]` https://arxiv.org/html/2312.06742v2 — Resampler 43.9 vs C-Abstractor 53.5 at
M=144 tokens on spatial understanding, attributed to "the abstraction process lacking a locality-aware
design".

**Cost.** `[RESAMPLER-RETRAIN]` — backbone stays frozen, so far cheaper than an encoder retrain, and it is
the principled alternative to the dropout-invariance retrain that already failed by contracting the latent
space `[OURS]`.

**Exactly what we run.** Nothing until Rank 1's attribution test says the loss is in the bottleneck. If it
does: one C-Abstractor variant at q8, matched pretraining budget, then the full battery.

**Kill gate.** Gated *upstream* by Rank 1. Then: if q8-C-Abstractor does not beat q8-Resampler on the
camera axis by ≥5pp at matched clean success, kill — a 9.6-point spatial-understanding gap at M=144 is not
a promise of a policy-level gain at M=8.

### Rank 6 — View-MASKING instead of invariance (only if we ever retrain)

**What.** If Kepler is ever retrained for view robustness, use view-masking + pixel reconstruction on top
of our existing patch-reconstruction head — never a contrastive or dropout invariance term.

**Evidence.** `[VERIFIED]` https://ar5iv.labs.arxiv.org/html/2302.02408 — the invariance route is a
*measured* failure:

> We find that TCN significantly fails to solve most of the tasks in both setups. This shows the critical
> drawback of TCN, which suffers from mode collapse when negative samples are too similar

while view-masking + reconstruction gives 74.7% zero-shot sim-to-real success vs 11.3% for the single-view
masked world model and 7.1% for MAE+WM.

**Cost.** `[ENCODER-RETRAIN]`.

**Value now (free).** This retro-explains our own dropout-based view-invariance retrain failing by latent
contraction — mode collapse under near-identical negatives is the same pathology. That upgrades our failed
retrain from an unexplained null into a **published, citable confirmation**, at zero cost, and it is worth
a paragraph in the paper this cycle even though we won't run the retrain.

**Kill gate.** Not run this cycle. If it is ever run: latent-space rank / effective-dimension monitor with
an automatic abort, since contraction is the known failure.

### Rank 7 — Explicit perspective warp in front of the resampler

**What.** Let the 8 tokens stay view-specific and model the view transform explicitly with a learned
perspective-transform STN before the resampler. Composes with Rank 5.

**Evidence.** `[VERIFIED]` https://arxiv.org/html/2407.15815v1 —

> we modify the affine transformations in the original STN to perspective transformations

Ablations: removing the perspective STN drops multi-task viewpoint-generalization success 86.5% → 65.3%;
removing the multi-view objective drops it to 29.4%; substituting a TCN contrastive loss gives only 65.8%.
Best available number on view-**equivariance** vs view-**invariance** as a design target: the invariance
objective does most of the work, but an explicit view-transform module is worth ~21pp on top, and a pure
contrastive substitute costs ~21pp.

**Cost.** `[RESAMPLER-RETRAIN]` (+ the multi-view objective it ablates against would be
`[ENCODER-RETRAIN]`).

**Kill gate.** Not this cycle. It is also partially redundant with Rank 3 — a learned warp in front of the
resampler and a canonicalizer in front of the backbone are the same idea at two depths, and Rank 3 is
cheaper and better evidenced. Only run this if Rank 3's oracle passes but the *learned* canonicalizer
fails.

### Rank 8 — Geometry-pretrained backbone swap (VGGT)

**What.** Re-pretrain Kepler's resampler over a geometry-pretrained (VGGT) patch grid instead of LeJEPA.

**Evidence.** `[VERIFIED]` https://arxiv.org/html/2509.14117v1 —

> a frozen, pretrained geometric vision model as a feature extractor ... A trainable projection layer then
> adapts these geometrically-rich features

Novel-camera-pose success 37.9% → 82.6% (BAKU, LIBERO) with in-distribution unchanged, 94.2% → 95.2% —
viewpoint robustness bought with **zero in-distribution tax**, which is exactly our publication bar. Their
own ablation shows the frozen-feature **layer** choice is load-bearing: a "Last 4 Layers" configuration
drops in-distribution LIBERO-Long to 60%. Highest ceiling in the whole angle; also the highest cost, and
directly contradicted under wrist views by Rank 3's source (GeoAwareVLA below 10%).

**Cost.** `[ENCODER-RETRAIN]` (new backbone + resampler re-pretrain). Park for v0.6.

**One correction to a plan we were carrying.** The idea of "eval-only first: cross-attend our 8 queries to
an **earlier** LeJEPA block since layer choice is free" is **not** eval-only and will not work as written.
Our resampler was trained against one specific block's feature distribution; swapping the block at
inference feeds it out-of-distribution keys and values, so a null result would be uninformative. The valid
cheap version is a **feature-space layer probe**: measure the view-vs-content ratio and clean-vs-shifted
patch drift *per LeJEPA block*, with no resampler in the loop. That is genuinely `[EVAL-ONLY]`, it is a
one-afternoon job, and if some block is markedly more view-stable it justifies a single
`[RESAMPLER-RETRAIN]` against that block. Fold this into Rank 1.

---

## 4. What the literature says will NOT work

### 4.1 Naive test-time augmentation — measured net harmful on corruption shift

`[VERIFIED]` https://arxiv.org/abs/2110.09506. ImageNet-C mCE (lower better): ResNet-50 76.7 → **77.9**
with TTA (+1.2); DeepAugment+AugMix 53.6 → 55.2 (+1.6); MoEx+CutMix 74.8 → 75.7 (+0.9). Verbatim table row:

> + TTA                            77.9 (+1.2)       61.3 (−2.6)    98.4 (−1.6)

CIFAR-10-C level 5: TTA is worse than no TTA on blur/contrast (defocus 24.1 → 28.3, motion 24.5 → 26.3,
contrast 29.7 → 32.9 test error). **Do not** average Kepler tokens or policy outputs over augmented
crops/flips. If we run TTA at all, run **MEMO** (adapt to minimize marginal entropy over augmentations,
then predict on the clean observation) — 69.9 mCE, the only variant positive on all three benchmarks.

`[VERIFIED]` https://arxiv.org/abs/2011.11156 — the aggregation rule can flip the sign.
Flowers102/MobileNetV2: original 90.28; Mean 90.47 (+0.19); **Max 90.17 (worse than no TTA)**; Greedy
Policy Search 88.28 (−2.0); learned per-augmentation weights 92.62 (+2.34).

> A key finding is that even when test-time augmentation produces a net improvement in accuracy, it can
> change many correct predictions into incorrect predictions.

"In the context of ImageNet and ResNet-18, a little over one third of the labels changed by the standard
TTA policy are incorrect", and "The benefit of simple averaging (Mean) diminishes with larger networks".
If we ever report TTA we must sweep the aggregation rule *and* report corrupted-vs-corrected episode
counts, not just net delta.

### 4.2 Entropy-minimization TTA — collapses in exactly our deployment regime

`[VERIFIED]` https://arxiv.org/abs/2302.12400.

> TTA may fail to improve or even harm the model performance when test data have: 1) mixed distribution
> shifts, 2) small batch sizes, and 3) online imbalanced label distribution shifts, which are quite common
> in practice. In this paper, we investigate the unstable reasons and find that the batch norm layer is a
> crucial factor hindering TTA stability.

ImageNet-C severity 5 under online imbalanced label shift: ResNet50-BN 18.0% unadapted, **Tent 2.1%, EATA
0.9%**; at test batch size 1 both collapse to **0.1%**. ResNet50-GN 30.6% unadapted → 22.0% with Tent, SAR
recovers 37.2%. Even with GN/LN, online entropy minimization "tend[s] to occur collapse, i.e., predicting
all samples to a single class". Closed-loop robot deployment is the worst case on all three of their axes
simultaneously — batch size 1, mixed shift, temporally correlated stream. Ruled out. Usable signal: our
LayerNorm ViT is on the *stable* side of their norm finding, and if we adapt at test time we should use a
reliability filter plus a flat-minimum update — or better, Rank 4's inverse-dynamics objective, which has
no degenerate optimum.

### 4.3 BatchNorm-statistic adaptation — structurally unavailable to us, and a threat to our headline

`[VERIFIED]` https://arxiv.org/abs/2006.16971 —

> Using the corrected statistics, ResNet-50 reaches 62.2% mCE on ImageNet-C compared to 76.7% without
> adaptation.

Improves all 25 models tested; "Even adapting to a single sample improves robustness", with ~16–32 samples
enough to converge. **The cheapest eval-only robustness win in the literature is unavailable to Kepler**:
LeJEPA ViT-B/16 is LayerNorm, so there are no batch statistics to re-estimate. Say that explicitly in the
paper — it is a structural reason a compact-token ViT encoder cannot inherit the standard covariate-shift
gain.

**But it is available to the bare ResNet diffusion-policy baseline we compare against.** Our +15.5pp
lighting win is therefore vulnerable: a reviewer can ask whether it survives when the ResNet arm is allowed
BN-stat adaptation. **This is a must-run defensive control, and it is nearly free.** Treat it as part of
Rank 1's matrix: add a "ResNet + BN-stat adaptation" column to every photometric cell. If the win does not
survive, we need to know before submission, not after.

### 4.4 Photometric augmentation will not fix the geometric axis

`[VERIFIED]` 3D Common Corruptions, CVPR 2022
(https://openaccess.thecvf.com/content/CVPR2022/papers/Kar_3D_Common_Corruptions_and_Data_Augmentation_CVPR_2022_paper.pdf).
Adding 2DCC augmentation cuts 2DCC L1 error 6.43 → 5.78 but only 6.13 → 5.94 on 3DCC, while 3D augmentation
reaches 5.42 — i.e. roughly 0.19 of a 0.65 improvement transfers across the 2D→3D boundary. Do not spend a
wave on colour-jitter retraining hoping to move camera shift.

### 4.5 Do not append a camera-pose token

`[VERIFIED]` https://arxiv.org/pdf/2510.02268 —

> We find that the linear projection method provides no observable benefit. In most cases, early fusion is
> less effective than late fusion.

Per-pixel Plücker ray maps *do* help held-out viewpoints for every policy/task tested (ACT Lift 33.6% →
60.6%, Assembly Square 10.8% → 18.7%), but the naive "project the extrinsic matrix to a single token"
variant gives nothing, and early fusion is worse than late. Two consequences: (i) if we ever condition on
camera pose, encode a per-pixel Plücker map with a small conv net and **late-fuse** before the queries
cross-attend; (ii) expect a tiny effect on our substrate — on Assembly Square the gain is +7.9pp for ACT
and **+0.4pp for Diffusion Policy** (2.0% → 2.4%), so a camera-conditioning leg on Square lands in the
noise. Free bonus control from the same paper: their shortcut finding (policies read camera pose off static
background) means we should re-run camera shift with **randomized backgrounds** — cheap, eval-only,
diagnostic.

### 4.6 Do not claim aug-gain + TTA-gain additively

`[VERIFIED]` TTAB, https://arxiv.org/abs/2306.03536 —

> Interestingly, even though these robust augmentation strategies significantly improve the robustness of
> the base model in the target domain, they only result in a marginal performance increase when combined
> with TTA.

They also find TTA hyperparameters near-impossible to select without prior knowledge of the shift ("even
given the labels of test examples, selecting TTA hyperparameters remains challenging, primarily due to the
batch dependency that arises during online adaptation"), and that even with oracle hyperparameters no
existing TTA method handles correlation or label shift. Practical rule: **one lever per axis**; if we report
two, report the interaction cell (aug-only / TTA-only / aug+TTA). And their oracle-hyperparameter protocol
is the right fit for our early-kill discipline — if TTA does not beat baseline under *oracle*
hyperparameters, kill at rung 0 rather than tuning.

### 4.7 Test-time selection against a learned score exploits it (cross-cutting warning)

`[VERIFIED]` https://arxiv.org/abs/2506.17811 —

> Furthermore, we observe that increasing the number of samples leads to exploitation of the V-GPS value
> function, resulting in performance degradation when sampling more than 8 actions. In contrast,
> RoboMonkey remains robust to reward hacking and demonstrates scalability with increased test-time
> compute.

V-GPS achieves only a 6% action-error reduction vs 21% for a discriminatively trained VLM verifier, and in
SIMPLER "pairing V-GPS with OpenVLA results in worse performance than both standalone OpenVLA and
RoboMonkey"; the verifier framework itself gives +25pp absolute on real-world OOD tasks and +9pp
in-distribution. This is primarily planner-leg material — it is an independent published replication of our
exact signature (predicted cost improves, realized cost worsens as sampling scales) in a system with no
world model — but it also constrains the robustness leg: **do not build a best-of-N robustness wrapper that
selects with a learned score.** Same lesson from MG-Select's corrected numbers (§4.9).

### 4.8 Refutations of ideas we currently hold — read this before writing prose

- **"Photometric easy / geometric hard" as a clean law from FactorWorld** — `[REFUTED]`. The paper's
  cross-setting summary pairs **table texture** with camera as the two hardest factors ("Changing the table
  texture and camera orientation causes the biggest drop, to 52.8% and 45.8%"), and table texture is an
  appearance factor. Corrected form: easier = backgrounds, distractors, lighting; harder = **table textures
  and camera positions**. Also, the "19 tasks / 11 factors / >100 configs per factor" scale is the released
  benchmark's size, not the evidence base for the ordering — the simulated ordering study is on three tasks
  (pick-place, bin-picking, door-open). The 0.4 → <0.1 gap closure is a **training-environment-count**
  result (5 → 100 envs), not a post-hoc fix. URL supplied for this paper only via the augmentation entry:
  arXiv 2307.03659.
- **"Augmentation-class matching is the measured mechanism of the split"** — `[REFUTED]`, and the source
  contradicts it. Xie et al. (arXiv 2307.03659) do say "crop augmentation improves generalization along
  multiple environment factors, most significantly to new camera positions and new table textures", but the
  next sentence is "While the improvement on a spatial factor like camera position is intuitive, we find
  the improvement on a non-spatial factor like table texture surprising." So augmentation class does **not**
  map onto shift class. It is also a policy-training result (`[ENCODER-RETRAIN]`/policy retrain), not a
  property of frozen tokens, so it cannot be evidence for our mechanism at all. Usable only as: crop
  augmentation is a candidate *remedy* for camera shift.
- **"Burns et al. show augmentation choice outweighs supervision"** — `[REFUTED]`, invented attribution.
  Their stated predictor is emergent segmentation ability; the quoted phrase is not verifiable in the
  paper. Cite Burns only for §2.2 / Rank 2 and for "manipulation-specific pretrained representations are
  not automatically robust (best manipulation-designed model ranks 7th of 15)".
- **Banani per-bin correspondence percentages (70–90% → 10–30%)** — `[REFUTED]`, curves only, no numeric
  table. Also do not say the six probed models all encode single-view geometry well: the paper says the
  opposite for CLIP and MAE ("CLIP and MAE features do not encode depth and appear to instead capture rough
  priors"), and the sharp cross-view collapse is specifically StableDiffusion and SAM ("This rapid
  deterioration is not universal, as shown by the wide baseline performance of DINOv2 and DeiT").
  View-dependence is their *hypothesis*, not their measurement.
- **Distracting Control Suite tiers as easy/medium/hard, camera as the harshest axis** — `[REFUTED]`. Tiers
  are easy (β=0.1, 4 videos) / medium (β=0.2, 8 videos) / **blind** (camera turned backwards, a lower-bound
  baseline); "hard" never appears as a level. Half-score points are βcam = 0.2, βrgb = 0.6, **βbg < 0.1**,
  so the ordering is background ≪ camera < colour — **unseen background is the harshest axis**, and the "3×"
  camera-vs-colour figure is our inference from 0.6/0.2, not a stated number. Regime is DM Control pixel-RL
  on locomotion/reaching, so treat the breakpoints as protocol guidance only. Corrected form is still
  usable and is the citation for §5's calibration argument (arXiv 2101.02722).
- **RGB-D-vs-point-cloud as evidence for a photometric/geometric dissociation** — `[REFUTED]` and no URL on
  file. The 0%-at-±0.1-rad figure is Clutter Pick only and applies to both RGB-D models; the 6%-vs-76%
  figure is LiftCube average reward over 42 viewpoint conditions — different task, different metric, not
  "the same policy". Lighting is **not** mildly harmful for RGB-D there (LiftCube RGBD-WM 305 → 126 under
  lighting, ~59%, vs 305 → 73 under viewpoint, 76%), and the fine-grained analysis trained models further
  first, so it is not a clean eval-only sweep. Their own null is worth knowing: for added distractors "we
  see no significant difference between point cloud and RGB-D models."
- **"TTA is the worst-performing family for viewpoint"** — `[REFUTED]` as framing (arXiv 2509.11125v1). The
  numbers are exact (MoVie 8.3, SRM 12.1, MV-MWM 55.9, ReViWo 57.6, Maniwhere 88.9, ManiVID-3D 94.5, over
  10 tasks × 50 episodes, ±60° yaw / ±7.5° pitch, distance 0.9–1.1×), but SRM is **augmentation**, not TTA,
  so the TTA "family" is n=1; and every arm is visual **RL from scratch**, not frozen-encoder BC or latent
  MPC, so it says little about a post-hoc wrapper on our frozen tokens. Note also that both winners
  (Maniwhere, ManiVID-3D) require encoder training with view-varied data — the class we ruled out after the
  dropout contraction failure — so they are not free targets. Directionally safe use: external evidence
  that naive TTA is weak for viewpoint shift.

### 4.9 Two more corrected magnitudes to avoid over-claiming

- **Test-time multi-view action aggregation** (arXiv 2604.00557) — `[REFUTED]` as "+4 to +9pp". Actual
  deltas are Square +1/+4/+4pp (N=10/25/50: 0.17→0.18, 0.29→0.33, 0.47→0.51) and Can +9/+5/+2pp
  (0.30→0.39, 0.68→0.73, 0.83→0.85): **+1 to +9pp, median ~+4pp**, with the weakest cells at the
  low-N and saturated ends. Decisive for us: it was **never** tested on a pure single-view-trained policy
  (the CamF-only row 0.14/0.26/0.42 is only evaluated single-view), the baseline is a DINOv3-base + LoRA
  diffusion policy, and it requires multiple calibrated cameras plus per-view action transforms at test
  time. We have one camera on Square. Not available to us.
- **MG-Select verifier-free best-of-N** (arXiv 2510.05681) — `[REFUTED]` as monotone test-time scaling.
  RoboCasa 100-demo all-tasks: greedy 42.7, sampling N=1 43.8, uniform-reference KL N=4 46.5, likelihood
  N=4 46.8, MG-Select N=4 48.1; but Table 5(b) only goes to N=16 (43.8 / 46.2 / 48.1 / 46.9 / 46.1) and
  **declines after N=4**. There is no N=64. Also the 5.3% → 14.2% (+168%) result is **simulation**
  (RoboCasa pick-and-place, 30 demos), not real-world; only +28% ID / +35% OOD are real-world. Narrow N
  sweet spot, discrete autoregressive-VLA confidence method, not a latent-MPC fix.

---

## 5. Protocol notes — what a reviewer will demand

**5.1 Severity must be calibrated across axes, or the win/loss split means nothing.** Our headline compares
**lighting severity 2** against **camera severity 1**. As written that is indefensible, even though the
direction is *unfavourable* to us (we win at the higher severity index and lose at the lower). Adopt 3DCC's
recipe `[VERIFIED]`:

> we set the corruption level in 3DCC such that for each shift intensity in 2DCC, the average SSIM [72]
> values over all images is the same in both benchmarks.

Concretely: pick per-axis severities with **equal mean SSIM against the clean frame**, then re-report the
split. Distracting Control Suite's corrected breakpoints (βcam 0.2 vs βrgb 0.6 vs βbg < 0.1 half-score
points, arXiv 2101.02722) are the independent argument that nominal severity indices are not comparable
across axes. This single change is what makes our four-axis table survive review.

**5.2 Report percentage change vs each arm's own clean score, not raw absolute pp.** COLOSSEUM's protocol
`[VERIFIED]` (https://arxiv.org/pdf/2402.08191):

> The models are restricted to evaluate over a fixed 25 episodes across 14 different perturbation_factors.
> 4) Models are ranked on a leaderboard based on the percentage change in their performance across these
> factors.

20 tasks × 100 demos, 14 axes, fixed 25 episodes/axis; 5 SOTA models degrade 30–50% per single axis and
≥75% under all perturbations combined; sim-to-real correlation R² = 0.614.

**This is our biggest reporting exposure, and it is not just a units question.** Our lighting cells sit at
ResNet 0.14–0.22 (near floor) while our camera cell sits at ResNet 0.74 (near clean). So "+15.5pp win" is
measured where the baseline has already collapsed, and "−18.7pp loss" is measured where the baseline is
barely touched. **Every cell must report: clean success for both arms, perturbed success for both arms,
absolute delta, and per-arm percentage change from its own clean score.** Anything less and a reviewer will
correctly say the win came off a floor.

**5.3 Add a combined-axes cell.** We currently have none. COLOSSEUM's ≥75% compound degradation is the
justification: a "holds across axes" claim is only really tested under simultaneous shift.

**5.4 Seeds — the single most dismissable thing in this leg.** Our numbers are 4 perturbation worlds on one
task with an unstated seed count; per the ICLR plan, wave-1 and low-N cells are 1-seed fast-eval. The
reviewer-expected bar `[VERIFIED]` (Burns et al.,
https://kayburns.github.io/segmentingfeatures/static/segmentingfeatures_paper.pdf):

> We train 3 different seeds within each task for each of two different camera angles. In total, we learn 60
> policies for each model and perform 11 evaluations per policy, including on the train distribution.

15 models × 10 tasks × 2 camera angles × 3 seeds = 60 policies/model, 11 evaluations/policy, ~6,000
evaluation runs per plotted point. We will not match that scale. **Minimum bar: ≥3 policy seeds per
(encoder, axis, severity) cell, ≥25 episodes/cell (COLOSSEUM's unit), shift-class means with intervals, and
the clean-distribution cell reported alongside every perturbed cell.** Note 25 episodes/cell is the
community norm but is statistically thin — say so rather than letting a reviewer say it.

**5.5 Adopt LIBERO-Plus's axis taxonomy verbatim** (layout, camera viewpoint, robot initial state,
language, light, background, sensor noise) so our table is directly comparable and their per-axis drops
become free reference deltas on the same plot. Drop or rename "distractor" to sit inside their taxonomy
(closest: layout/background) and state which of the 7 we do not test and why.

**5.6 Fairness controls that must ship with the table.** (a) ResNet baseline **with BN-stat adaptation** on
every photometric cell (§4.3). (b) Camera shift with **randomized backgrounds** to test the read-the-pose-
off-the-background shortcut (§4.5). (c) For PushT, state that camera shift is a synthetic 2D warp because
DINO-WM's PushT is an orthographic pygame render with no camera model (`env/pusht/pusht_env.py:110-115`),
so the geometric axis is only honestly testable on Square. (d) If any TTA variant is reported, report
corrupted-vs-corrected episode counts (§4.1).

**5.7 In-distribution tax.** The publication bar is "maintained, never improved", so every intervention
cell needs its clean-task cost reported next to it. Precedents to hold ourselves against: 94.2% → 95.2%
(zero tax, arXiv 2509.14117v1) and 45.57 → 44.51 (−1.1pp, arXiv 2310.01647), against the failure cases
96.97 → 93.29 and in-distribution mAP collapse to 35.77 in the same paper.

---

## 6. Source ledger

| # | URL | What it gives us | Status | Cost class |
|---|-----|------------------|--------|-----------|
| 1 | https://arxiv.org/abs/2510.13626 | LIBERO-Plus: 7-axis taxonomy, 10,030 evals, camera −37.4pp vs light −11.3pp, Finding 2 mechanism | `[VERIFIED]` | eval-only |
| 2 | https://arxiv.org/pdf/2402.08191 | COLOSSEUM: 25-episode/14-axis protocol, %-change reporting, ≥75% compound drop, R²=0.614, 2D-vs-3D captured-view mechanism | `[VERIFIED]` | eval-only |
| 3 | https://kayburns.github.io/segmentingfeatures/static/segmentingfeatures_paper.pdf | Burns et al.: Jaccard/emergent-segmentation predicts OOD; PVRs not automatically robust (7th/15, MVP 39.86→6.63); 3-seed × 2-camera protocol bar | `[VERIFIED]` | eval-only |
| 4 | https://openaccess.thecvf.com/content/CVPR2022/papers/Kar_3D_Common_Corruptions_and_Data_Augmentation_CVPR_2022_paper.pdf | 3DCC: SSIM-matched severity recipe; 2D aug does not transfer to 3D (6.13→5.94 vs 5.42) | `[VERIFIED]` | eval-only |
| 5 | https://arxiv.org/html/2411.04983v2 | DINO-WM (our harness): patch 0.90 / CLS 0.44 / R3M 0.42; global-pooling loses spatial detail | `[VERIFIED]` | eval-only |
| 6 | https://arxiv.org/abs/2310.01647 | Prior-regularized canonicalization in front of a frozen model: 27.67→44.50 C4-Avg, ID 45.57→44.51, 1.9M params, prior loss only; failure modes 0.2M→35.77 and 96.97→93.29 | `[VERIFIED]` | post-hoc |
| 7 | https://arxiv.org/html/2603.05868v1 | Test-time NVS re-render to training pose: LIBERO 39.9%→94.5%, 36.55 ms / 27 FPS, no demos or finetuning; GeoAwareVLA <10% under wrist perturbation | `[VERIFIED]` | post-hoc / eval-only |
| 8 | https://arxiv.org/abs/2007.04309 | PAD: inverse-dynamics TTA, 31/36 envs, real arm +24pp, push(all) 48→76%; negatives (no-shift cells, single-frame variant, SSL-task match) | `[VERIFIED]` | post-hoc |
| 9 | https://arxiv.org/html/2312.06742v2 | Honeybee: Resampler 43.9 vs C-Abstractor 53.5 at M=144; "lacking a locality-aware design" | `[VERIFIED]` | resampler-retrain |
| 10 | https://arxiv.org/html/2509.14117v1 | Geometry-pretrained frozen backbone + light projection: novel-pose 37.9→82.6%, ID 94.2→95.2%; layer choice worth 30pp+ | `[VERIFIED]` | encoder-retrain |
| 11 | https://arxiv.org/html/2407.15815v1 | Perspective-transform STN ablations: 86.5→65.3 (no STN), →29.4 (no multi-view), 65.8 (TCN substitute) | `[VERIFIED]` | resampler/encoder-retrain |
| 12 | https://ar5iv.labs.arxiv.org/html/2302.02408 | View-masking beats invariance: 74.7% vs 11.3% vs 7.1%; TCN "significantly fails", mode collapse — retro-explains our dropout contraction | `[VERIFIED]` | encoder-retrain |
| 13 | https://arxiv.org/pdf/2510.02268 | Plücker ray maps help (ACT Lift 33.6→60.6, Assembly Square 10.8→18.7) but pose-token projection gives nothing, early < late fusion, DP Square +0.4pp; background shortcut control | `[VERIFIED]` | encoder-retrain |
| 14 | https://arxiv.org/abs/2110.09506 | Naive TTA harms corruption robustness (mCE 76.7→77.9); MEMO 69.9; BN adaptation 71.4 | `[VERIFIED]` | eval-only |
| 15 | https://arxiv.org/abs/2011.11156 | TTA aggregation rule flips sign (Mean 90.47 / Max 90.17 / GPS 88.28 / learned 92.62); ~1/3 of changed labels are wrong | `[VERIFIED]` | eval-only |
| 16 | https://arxiv.org/abs/2006.16971 | BN-stat adaptation 76.7→62.2 mCE, all 25 models, 16–32 samples — unavailable to our LayerNorm ViT, available to the ResNet baseline | `[VERIFIED]` | eval-only |
| 17 | https://arxiv.org/abs/2302.12400 | SAR: Tent 18.0→2.1, EATA 0.9, bs=1 → 0.1; GN/LN more stable; BN is the destabilizer | `[VERIFIED]` | eval-only |
| 18 | https://arxiv.org/abs/2306.03536 | TTAB: aug and TTA are substitutes not complements; TTA hyperparameters unselectable without shift knowledge | `[VERIFIED]` | eval-only |
| 19 | https://arxiv.org/abs/2506.17811 | RoboMonkey/V-GPS: sampling exploits a learned value past 8 samples; 6% vs 21% error reduction; verifier +25pp OOD — external replication of our exploitation signature | `[VERIFIED]` | post-hoc |
| 20 | https://openaccess.thecvf.com/content/CVPR2024/papers/Banani_Probing_the_3D_Awareness_of_Visual_Foundation_Models_CVPR_2024_paper.pdf (also arXiv 2404.08636v1) | "view-consistent, not 3D consistent"; wide-baseline absolute performance "very low"; CLIP/MAE do not encode depth | `[REFUTED]` as numbers (curves only); corrected qualitative form usable | eval-only |
| 21 | arXiv 2307.03659 | Xie et al. / FactorWorld: corrected ordering = backgrounds/distractors/lighting easier, **table textures and camera** harder; crop aug also most-helps table texture ("we find ... surprising"); 0.4→<0.1 gap from 5→100 training envs | `[REFUTED]` as "clean photometric/geometric law" and as "augmentation-class matching mechanism" | policy retrain |
| 22 | arXiv 2101.02722 | Distracting Control Suite: βcam 0.2 / βrgb 0.6 / βbg < 0.1 half-score points; tiers easy/medium/**blind**; static≈dynamic | `[REFUTED]` as "easy/medium/hard, camera harshest"; corrected form usable for §5.1 | eval-only |
| 23 | arXiv 2509.11125v1 | ManiVID-3D Table I: MoVie 8.3, SRM 12.1, MV-MWM 55.9, ReViWo 57.6, Maniwhere 88.9, ManiVID-3D 94.5 | `[REFUTED]` as "TTA is the worst family"; RL-from-scratch regime; winners need encoder retraining | encoder-retrain |
| 24 | arXiv 2604.00557 | Multi-view test-time action aggregation: Square 0.17→0.18 / 0.29→0.33 / 0.47→0.51, Can 0.30→0.39 / 0.68→0.73 / 0.83→0.85 | `[REFUTED]` as "+4 to +9pp"; needs multiple calibrated cameras; never tested on a single-view-trained policy | eval-only |
| 25 | arXiv 2510.05681 | MG-Select: RoboCasa 42.7→48.1 at N=4, declines to 46.1 by N=16; 5.3→14.2% is simulation not real | `[REFUTED]` as monotone scaling / real-world | post-hoc |
| 26 | — (no URL on file) | ManiSkill2 RGB-D vs point-cloud viewpoint study: Clutter Pick 0% at ±0.1 rad, LiftCube 6% vs 76% average-reward loss, no point-cloud advantage under distractors | `[REFUTED]` as a photometric/geometric dissociation source; **do not cite until a URL is verified** | eval-only |

**Prior-cycle sources carried forward (planner leg, listed for completeness, verified in earlier cycles):**
Adversarial World Modeling arXiv 2512.09929 `[VERIFIED — prior cycle]` (FGSM-perturbed finetuning, DINO-WM
PushT CEM 78→94; our own version is running); CompACT arXiv 2603.05438 / OpenReview z9KgtDP6LB
`[VERIFIED — prior cycle]`; DCWM/DC-MPC arXiv 2503.00653 `[VERIFIED — prior cycle]`; RC-aux arXiv
2605.07278 `[VERIFIED — prior cycle, downgraded: −0.4pp on Push-T]`; TRM arXiv 2605.22164
`[VERIFIED — prior cycle]`.

**Internal, not externally verified `[OURS]`:** Square robustness battery (lighting sev 2 +15.5pp, 4/4
worlds 0.14→0.44 / 0.18→0.34 / 0.22→0.30 / 0.20→0.28; colour sev 2 +6pp; camera sev 1 0.74→0.553;
distractor sev 1 −13pp); composition analysis view-vs-content ratio 1.46; rollout-MSE ratios q4 0.474 /
q8 0.494 / q16 0.469 vs dense 0.726; dropout view-invariance retrain failure by latent contraction.

---

## 7. Open questions before this section is final

1. Are `plan_outputs/hybrid_q4` and `hybrid_q16` **encoder** variants or predictor-only variants? Rank 1's
   trade-off curve requires the former. If they are predictor-only, the token-budget axis costs three
   resampler pretrains and drops below Rank 3 in priority.
2. What is the actual seed count behind the four lighting worlds and the camera cell? Everything in §5.4
   depends on it, and if it is 1, the headline needs re-running before it can be quoted.
3. Does the Square battery's clean cell exist for both arms at every axis? §5.2's normalized reporting is
   impossible without it.
4. Can we get ground-truth camera extrinsics out of the robomimic Square renderer cheaply? Rank 3's oracle
   depends on it and it is the gate for the entire canonicalization bet.
