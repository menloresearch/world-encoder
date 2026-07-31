# v0.4 research 01 — Wiring robot state as the privileged cross-attention query

**Date:** 2026-07-30. **Question:** how exactly to make proprioceptive state form/condition the
PerceiverFuse queries (8 learned queries, d=256, over frozen ViT-B/16 patch tokens from 3 static cams),
instead of all sensors being equal citizens. **Method:** 18 web fetches across 16 unique sources; every
load-bearing claim below was checked against the fetched full text (arXiv HTML/ar5iv, one PDF extracted
locally). Claims I could not confirm in a fetched source are marked **UNVERIFIED**. Companion docs:
REDESIGN_RESEARCH_20260729.md (insertion mechanisms, AdapTac aux findings), LITERATURE_20260728.md.

---

## Executive summary

1. **AdapTac's "force as query" is narrower than our shorthand suggests.** Its query is ONE vector (net
   force through an MLP); K/V are exactly TWO tokens (one point-cloud feature, one tactile feature); the
   softmax yields two scalar mixing weights. It is a state-driven modality GATE, not state-driven spatial
   attention over patch tokens. Its win decomposes as: concat 40% → +attention-fusion 67% → +future-force
   aux 93%; and fusion WITHOUT the aux (67%) still loses to vision-only RISE (73%). Wiring alone is not
   the mechanism — the chaseable future-signal aux is. Do not port the wiring without the aux.
2. **No paper directly ablates "state conditions the query" vs "state as extra K/V token" in the same
   block.** The honest evidence base is triangulation: InstructBLIP (conditioning the Q-Former queries on
   the privileged signal: +4.3 avg held-in, up to +7.6 OOD, over identical architecture without it),
   NoContactNoWorries (proprio-as-query cross-attention beats MLP concat, F1 0.93 vs 0.74),
   DAB-DETR (structured query conditioning: 45.7 AP @50 epochs vs DETR 43.3 @500), DiT
   (adaLN-Zero > cross-attn > in-context for conditioning), and GAP (the anti-pattern: state CONCAT
   into the feature path makes policies 15.8% WORSE than vision-only).
3. **The scale-proven pattern is "state on the query-side stream, vision as K/V":** pi0 routes state to
   the action expert whose tokens attend over VLM K/V; RDT-1B puts proprio in the main stream with images
   entering only via cross-attention; BLIP-2/InstructBLIP put the privileged signal (text) in the query
   stream via shared self-attention, never in the image K/V. Nobody at scale injects the privileged
   signal as extra K/V next to patch tokens; the one popular K/V-side design (ACT's 2 extra tokens,
   PerAct's tiled concat) is exactly the "equal citizen" default we are moving away from.
4. **GAP is the central risk result for this redesign:** vision+proprioception policies are 15.8% worse
   than vision-only on average (concat wiring; e.g., Meta-World pick-place 78.4±1.5 vs 91.8±1.5), because
   optimization gravitates to the low-dim state signal and suppresses vision learning precisely during
   motion-transition/localization phases — the phase structure of RoboCasa pick-and-place. The design
   consequence: state may set WHERE the queries look, but state content must not flow into the value
   pathway (the fuse output must remain a function of vision values only), and the state pathway must be
   zero-init gated (Flamingo tanh-gate +4.2; RT-1 identity-init FiLM; DiT adaLN-Zero).
5. **Ranked recommendation** (details §5): (1) additive state-conditioned queries with a zero-init tanh
   gate (queries stay learnable; +~0.3–0.5M params; one-line change to the block); (2) Q-Former-style
   state token in the query STREAM via self-attention, masked out of the vision cross-attention and
   dropped from the output (needs a self-attn layer among queries; the strongest large-scale ablation
   backs it); (3) per-sublayer adaLN-Zero/FiLM of the query stream by state (best conditioning mechanism
   in DiT + low-data robot replication, but modulates rather than forms queries). A state-conditioned
   attention BIAS (EE-projection foveation) is a plausible add-on with only indirect evidence — run it
   later, not first.
6. **Pretraining interaction (ours specifically):** if current state becomes an encoder INPUT, our
   state-PREDICTION objective becomes copyable and dies. Switch the aux to FUTURE state (Δ over horizon
   h), mirroring AdapTac, where predicting future (not observed) force was the driver (Flip 50→90; using
   observed force only reached 70).

---

## 1. AdapTac: the exact wiring [verified]

Paper: "Adaptive Visuo-Tactile Fusion with Predictive Force Attention for Dexterous Manipulation"
(Li, Wu, Zhang et al.), arXiv:2505.13982 (fetched HTML v2). Base policy RISE (3D diffusion).

```
net force F_O (sum of taxel forces, camera frame)
        │  MLP g_F  (output dim unspecified in paper)
        ▼
   e_F ──W_Q──►  Q_F                       ← ONE query vector
                          K = [e_pc, e_tac]·W_K     e_pc = g_pc(Z_pc) ∈ R^512
                          V = [e_pc, e_tac]·W_V     e_tac = g_tac(Z_tac) ∈ R^512
   α_pc, α_tac = softmax(Q_F·K^T/√d_k)     ← TWO attention weights (per-modality scalars)
   Z_fuse = α_pc·V_pc + α_tac·V_tac        → diffusion action head
```

- **Where state enters:** only as the query; force content reaches the head only through the two scalar
  weights — a maximal information bottleneck on the privileged signal. [verified]
- **Layer count / heads:** not specified in the paper (single fusion block as described; no multi-layer
  stack is mentioned). [verified absence]
- **Gating:** none beyond the softmax itself — the softmax over exactly two modality tokens IS the gate.
  [verified]
- **Aux:** future net-force prediction over the next n steps (n = action horizon), transformer-based
  diffusion head, L = L_π + α·L_ffp (α unspecified). At inference the PREDICTED future force is
  concatenated into the query: Q_F = g_F([F_O, F_pred])·W_Q. [verified]
- **Ablations (Table II, avg over Open Box / Reorientation / Flip):** concat (no attention, no aux) 40%;
  attention fusion only 67%; full 93%. Table III (Flip): no aux 50%, observed-force prediction+guidance
  70%, future-force prediction 90%, + future-force guidance 90% with episode length 166→113. [verified]
- **Reading for us:** AdapTac supports "state forms the query" as a *modality-arbitration* mechanism and
  supports the future-signal aux strongly. It does NOT demonstrate state-conditioned *spatial* attention
  over patch tokens — our design extrapolates the pattern from 2 K/V tokens to ~600 patch tokens/cam.

## 2. Privileged-conditioning patterns — survey (all fetched unless marked)

### 2.1 Perceiver-IO decoder queries — the design principle [verified]
arXiv:2107.14795 (ar5iv). "We construct queries by combining (concatenating or adding) a set of vectors
into a query vector containing all of the information relevant for one of the O desired outputs." Queries
are learned for classification, positional for spatial outputs, and include INPUT FEATURES when the
output must reflect input content ("for flow we find it helpful to include the input feature at the point
being queried"). Also: "even very simple query features can produce good results." For a robot encoder
whose desired output is "what the policy needs given the current body configuration," the Perceiver-IO
principle says the query should contain the state. This is the cleanest conceptual license for design (a).

### 2.2 BLIP-2 / InstructBLIP — the closest large-scale analogy [verified]
- BLIP-2 (arXiv:2301.12597, ar5iv): Q-Former = 32 learned queries × 768d, 188M params; cross-attention
  to frozen image features inserted every other transformer block; "the queries can additionally interact
  with the text through the same self-attention layers." The privileged signal (text) sits in the QUERY
  STREAM, never in the image K/V. Bottleneck framing: 32×768 outputs vs 257×1024 image features "forces
  queries to extract the most text-relevant visual information."
- InstructBLIP (arXiv:2305.06500, ar5iv): "The instruction interacts with the query embeddings through
  self-attention layers of the Q-Former, and encourages the extraction of task-relevant image features."
  Ablation removing instruction from the Q-Former (FlanT5XL, same everything else): held-in avg 94.1→89.8
  (−4.3); OOD examples: ScienceQA 70.4→63.4, VizWiz 32.7→25.1, iVQA 53.1→47.5. This is the best published
  causal isolation of "condition the queries on the privileged signal vs don't."
- Mapping: text→state, frozen ViT→frozen ViT, 32 queries→8 queries. The wiring is *state tokens
  self-attend with the learned queries between cross-attention layers* — design (e)/Rank-2 below.

### 2.3 Flamingo — resampler + zero-init gating [verified, one detail UNVERIFIED]
arXiv:2204.14198 (ar5iv). 64 learned latent queries cross-attend to visual features (Perceiver
Resampler). Ablations (Table 3, overall score): Perceiver Resampler 70.7 vs vanilla Transformer 66.7 vs
MLP 66.6. Gated cross-attention: output multiplied by tanh(α), "α is a layer-specific learnable scalar
initialized to 0," so "at initialization, the conditioned model yields the same results as the original
language model"; removing tanh gating: 70.7→66.5. **UNVERIFIED:** the appendix detail that resampler K/V
are computed from the concatenation [visual features; latent queries] — could not be confirmed in the
ar5iv fetch (pseudocode not rendered); treat as likely-true but unchecked. Layer count also not
confirmed. The gating result is the strongest published number for "zero-init the new conditioning path."

### 2.4 Octo — readout isolation, and the proprioception episode [verified]
arXiv:2405.12213v2 (HTML). Readout tokens "attend to observation and task tokens ... but [are] not
attended to by any observation or task token — hence, they can only passively read." Diffusion head reads
only readout embeddings. Directly relevant to v0.3 unpooled-tokens (our 8 outputs ≈ readouts), and to
Rank-2's masking discipline. On proprioception: the v2 paper contains NO statement about proprioception
in pretraining and no shortcut/causal-confusion discussion (searched full text incl. appendices; only
mention is force-torque proprioception as a NEW finetuning input on Berkeley Peg Insertion, 70% vs 10%
from-scratch). Note: GAP (§2.8) writes "As reported in the original paper, policies trained with
additional proprioception seemed generally worse than vision-only policies" — that attribution is
**UNVERIFIED** against Octo v2 (not present in the fetched text or the GitHub README; possibly v1 or
docs); GAP's own Octo-VP vs Octo-V experiments do show the effect (their Table 3), and GAP-adjusted
Octo-VP† beats Octo-V by 17% average.

### 2.5 pi0 / GR00T N1 / RDT-1B — state handling in current VLAs [verified]
- pi0 (arXiv:2410.24164, HTML): state is ONE token via linear projection, routed to the small action
  expert (1024 width vs 2048 VLM): "the inputs not seen during VLM pre-training, [q_t, A_t^τ], are routed
  to the action expert." Blockwise attention: images+language | state | actions; state sits in its own
  block, action tokens attend to everything. No stated rationale for the routing. Net: state lives on the
  query-side stream that attends over vision K/V — never as extra K/V among image tokens.
- GR00T N1 (arXiv:2503.14734, HTML): "we use an MLP per embodiment to project [states and actions] to a
  shared embedding dimension as input to the DiT." DiT = alternating self-attn (over state/action tokens)
  and cross-attn to vision-language token embeddings; adaptive layernorm carries the denoising step. Same
  topology: state on the query side, vision as K/V.
- RDT-1B (arXiv:2410.07864, HTML): proprio encoded by MLPs with Fourier features into the main stream;
  images enter ONLY by cross-attention ("to accommodate conditions of varying lengths avoiding the
  information loss"), with image/text injection ALTERNATING across layers because "simultaneous injection
  ... tends to overshadow text-related information." Removing alternating injection: 48%→32% on their
  dexterity benchmark. Two lessons: (i) again state-on-query-side; (ii) when one K/V source has far more
  tokens than another, it drowns the smaller one — the token-count argument AGAINST putting a 1-token
  state into K/V next to ~1800 patch tokens.

### 2.6 The "equal citizen" defaults — ACT, PerAct, HPT [verified]
- ACT (arXiv:2304.13705, ar5iv): "We then append two more features: the current joint positions and the
  'style variable' z," linearly projected to 512, giving a 1202×512 encoder input (1200 image tokens + 2).
  State as extra K/V-ish token works here but was never ablated against alternatives.
- PerAct (arXiv:2209.05451, ar5iv): proprio (4→64d linear) is "tiled in 3D to match the dimensions of the
  patch tensor, and concatenated along the channel" BEFORE the Perceiver (2048 latents × 512, 6 self-attn
  layers); language appended to the input sequence. Proprio as K/V content, tiled everywhere — the
  opposite pole from privileged-query.
- HPT (arXiv:2409.20537, HTML): proprio stem = MLP → sinusoidal PE → cross-attention with 16 learnable
  tokens; vision stem likewise → 16 tokens; equal counts (Np=Nv=16) so the trunk can "treat them in the
  same manner despite large heterogeneity." Scaled across 27 datasets, and GAP cites HPT as a case where
  proprio DID help. So equal-citizen tokenization is viable at scale — the literature is genuinely split,
  and the split correlates with data scale and regularization, not just wiring.

### 2.7 FiLM lineage — RT-1, BC-Z, DiT [verified]
- RT-1 (arXiv:2212.06817, ar5iv): language conditions EfficientNet-B3 via "identity-initialized FiLM
  layers"; "we initialize the weights of the dense layers (fc and hC) which produce the FiLM affine
  transformation to zero, allowing the FiLM layer to initially act as an identity and preserve the
  function of the pretrained weights" — identity-init helped even training from scratch. No FiLM-vs-X
  ablation.
- BC-Z (arXiv:2202.02005, ar5iv): 512-d task embedding → per-channel scale/shift in all 4 ResNet blocks.
  Directly relevant negative result, Appendix L: "Conditioning the policy on proprioceptive information,
  as well as previous robot poses, did not improve performance" — BC-Z dropped state entirely.
- DiT (arXiv:2212.09748, ar5iv): block-design ablation at 400K iters (FID-50K, from Fig. 5): adaLN-Zero
  ~9.5 < adaLN ~10.6 < cross-attention ~11.4 < in-context ~19. "We initialize the MLP to output the
  zero-vector for all α; this initializes the full DiT block as the identity function." Cross-attention
  conditioning adds "roughly a 15% overhead" in Gflops; adaLN(-Zero) negligible. Combined with DiT-Block
  Policy's low-data robot replication (adaLN-Zero > per-layer cross-attn; already verified in
  REDESIGN_RESEARCH_20260729.md §4), this is the evidence base for design (c).

### 2.8 GAP — when privileging state backfires [verified]
"When would Vision-Proprioception Policies Fail in Robotic Manipulation?", arXiv:2602.12032, ICLR 2026
(PDF fetched and text-extracted). Setup: standard joint-learning policy — ResNet-18 vision chunk +
proprio chunk (6D gripper pose + opening degree), features CONCATENATED into the policy head; Meta-World,
RoboSuite (threading, stack, ...), real 6-DoF, 5 seeds. Findings:
- "Vision-Proprioception policies perform 15.8% worse than Vision-only policies" (Fig. 1); e.g.
  Meta-World pick-place 78.4±1.5 (concat) vs 91.8±1.5 (vision-only); assembly 74.6±2.1 vs 82.6±3.0.
- Mechanism: "the policy naturally gravitates toward concise proprioceptive signals that offer faster
  loss reduction ... dominating the optimization and suppressing the learning of the visual modality
  during motion-transition phases." Intervention experiments: swapping in the vision+proprio policy for
  10-step windows barely hurts during motion-consistent phases ("move forward") but degrades during
  motion-transition phases requiring target localization.
- Fixes that work are OPTIMIZATION-side: GAP reduces the magnitude of the proprioception gradient on
  timesteps an LSTM phase model flags as motion-transition; beats aux-loss forcing, fixed-probability
  proprio masking, and MS-Bot; applied to Octo fine-tuning, Octo-VP† averages +17% over Octo-V.
- Reading for us: the failure is state-in-the-VALUE-path plus unconstrained gradients. Query-side wiring
  is precisely the structural version of GAP's fix — state can decide where to look but contributes no
  content the head can regress onto. Still, per-dim state dropout (DiT-Block Policy, prior corpus) and
  the zero-init gate remain mandatory.

### 2.9 Direct state-as-query precedents in robot learning [verified]
- **NoContactNoWorries** (arXiv:2606.24450, HTML): the closest existing implementation to our design.
  Current joints q_t and commanded joints q_t^com are linearly projected to TWO query tokens (D=256);
  K/V = N spatial tokens from a FROZEN RGB-D encoder (conv-downsampled); ONE cross-attention layer, ONE
  head, d=256; the two attended vectors are concatenated → MLP → temporal transformer. Ablation: full
  query-asymmetric design F1 0.93±0.010 vs symmetric (mean-pooled attended tokens) 0.78±0.014 vs MLP over
  concatenated embeddings 0.74±0.023 (real-world 0.84 vs 0.70). Contact estimation + in-hand rotation,
  not BC on pick-and-place — but it is a working existence proof at exactly our dims (d=256, frozen
  vision, single cross-attn layer), and evidence that (i) state-as-query > MLP concat, (ii) WHICH state
  forms the query matters (current vs commanded asymmetry is worth +0.15 F1).
- **3D Diffuser Actor** (arXiv:2402.10885, ar5iv): proprioception = "a short history of predicted
  end-effector keyposes ... represented with learnable feature vectors and their 3D positions used to
  compute positional embeddings"; all tokens (visual, language, proprio, action) jointly contextualized
  with 3D RELATIVE position attention. Ablation: relative attention 81.3% vs absolute 71.3% ("translation
  equivariance through relative attentions is very important"). Evidence for geometry-conditioned
  attention biases (design d) — state enters the attention LOGITS via relative geometry, +10pp.
- **Robot-centric pooling** (arXiv:2411.04331): appeared in search as cross-attention aligning image and
  proprio latents for pooling; both fetch attempts failed (size limit / 404). **UNVERIFIED — do not cite
  numbers from it.**
- **DAB-DETR** (arXiv:2201.12329, ar5iv) as the non-robotics anchor for query conditioning: replacing
  free learned queries with structured, iteratively-updated anchor-box queries: 45.7 AP (R50-DC5, 50
  epochs) vs DETR 43.3 (500 epochs) vs Conditional DETR 43.8 (50 epochs); width/height-modulated
  attention (a conditioned attention bias) and anchor updating (+1.7 AP alone). Message: giving queries
  task-meaningful conditioning instead of leaving them free both speeds convergence and raises the
  ceiling — in a regime (DETR) with vastly more data than ours, where free queries were the bottleneck.

## 3. Does conditioning the QUERY beat state-as-K/V or FiLM-on-features? Evidence status

**No controlled three-way comparison exists in the fetched literature.** What can be said with sources:

| Comparison | Best available evidence | Result |
|---|---|---|
| Query-conditioning vs NO state in encoder | InstructBLIP ablation [verified] | +4.3 held-in avg, up to +7.6 OOD |
| State-as-query cross-attn vs MLP/concat fusion | NoContactNoWorries [verified]; AdapTac Table II [verified] | F1 0.93 vs 0.74; 67% vs 40% (both same-paper, controlled) |
| State concat into feature path vs vision-only | GAP [verified]; BC-Z App. L [verified] | −15.8% avg; "did not improve performance" |
| Privileged signal in K/V next to patch tokens | No positive published instance found at scale; RDT's drowning argument [verified] applies (1 state token vs ~1800 patch tokens) | — |
| FiLM/adaLN vs cross-attn vs in-context tokens (generic conditioning) | DiT Fig. 5 [verified]; DiT-Block Policy [verified, prior corpus, low-data caveat] | adaLN-Zero best, in-context worst |
| Free learned queries vs conditioned queries | DAB-DETR [verified] | conditioned queries: +2.4 AP and 10× faster |
| Zero-init gating of the new path | Flamingo [verified]; RT-1 identity-FiLM [verified]; DiT adaLN-Zero [verified] | +4.2 overall; qualitative; best FID |

Triangulated conclusion: query-side entry is the best-supported placement for a privileged low-dim
signal; K/V-side entry next to a large patch set has no supporting precedent and two arguments against
(token drowning, GAP shortcut); FiLM-style modulation is the best-supported *mechanism* for injecting a
conditioning vector into a transformer stream when you do not need new content, only re-weighting.
These two are compatible: FiLM the QUERY stream (design c) is query-side entry implemented with the
DiT-winning mechanism.

## 4. Design space for PerceiverFuse — evidence and cost per option

Notation: s ∈ R^{d_s} (RoboCasa proprio: joints, gripper, EE pose; d_s ≈ 40–60), Q = 8 learned queries
d=256, K/V = ViT-B/16 patch tokens (3 cams; 768→256 projected), L = current number of cross-attn layers.

### (a) Query-init conditioning: s → MLP → added to the 8 learned queries
`q_i' = q_i + tanh(α_i)·W_i·MLP(s)` (queries stay learnable; α zero-init).
- **For:** Perceiver-IO's stated construction rule ("combining (concatenating or adding) ... all of the
  information relevant for the desired output") [verified]; DAB-DETR conditioned-query gains [verified];
  NoContactNoWorries — state projections literally ARE the queries at d=256 over frozen features, beating
  MLP fusion by +0.19 F1 [verified]; InstructBLIP directionally (conditioning queries on privileged
  signal helps, though via self-attn) [verified].
- **Against / caveats:** if only the INPUT queries are conditioned, the conditioning attenuates through L
  layers (DAB-DETR found iterative re-injection worth +1.7 AP [verified]); AdapTac warns the wiring alone
  may not beat vision-only without a chaseable aux [verified].
- **Cost:** MLP d_s→256→2048 ≈ 0.28–0.55M params; zero code-structure change; zero extra attention cost.

### (b) State token(s) as extra K/V alongside patches
- **For:** ACT ships this way (2 extra tokens among 1202) [verified]; PerAct tiles proprio into every
  voxel patch [verified]; HPT tokenizes proprio into 16 equal tokens and scaled [verified].
- **Against:** this is the "equal citizen" scheme v0.4 is explicitly moving away from; 1–2 state tokens
  vs ~1800 patch K/Vs invites drowning (RDT alternates injection for exactly this failure, 48→32 when
  removed [verified]); GAP/BC-Z show state content reaching the head can actively hurt [verified]; no
  paper shows K/V-side state BEATING an alternative wiring.
- **Cost:** trivial (+1 projection). Worth keeping only as the CONTROL ARM in our experiment matrix.

### (c) FiLM/adaLN-Zero modulation of the query stream, per layer
`h ← γ(s)⊙h + β(s)` (+ zero-init output scale α(s)) applied to query tokens before attn/MLP sublayers.
- **For:** DiT: adaLN-Zero best conditioning block, negligible Gflops, cross-attn +15% Gflops [verified];
  DiT-Block Policy replicated in low-data robot BC [verified, prior corpus, caveat]; RT-1/BC-Z:
  identity-init FiLM is the standard, stable way to condition a pretrained visual pathway on a side
  signal [verified]; per-layer re-injection solves (a)'s attenuation.
- **Against:** modulation cannot ADD state-derived content or re-aim attention as directly as forming the
  query vector (γ,β are per-channel, not per-token-direction); all positive evidence is for
  language/timestep conditioning, none for proprio specifically.
- **Cost:** shared trunk d_s→256 plus per-sublayer heads 256→(2 or 3)·256, zero-init: ≈0.13M per
  modulated sublayer, ~0.3–0.5M for a 2-layer block.

### (d) State-conditioned attention bias
`logits_ij += b_h(geom(patch_j, EE(s)))` — e.g., per-head MLP over patch-to-EE displacement (needs
camera calib; available in RoboCasa sim).
- **For:** 3D Diffuser Actor: relative-geometry attention 81.3 vs 71.3 [verified]; DAB-DETR's w/h-
  modulated attention (conditioned spatial prior) [verified]. Both are geometry-in-the-logits wins.
- **Against:** no instance of specifically EE-conditioned bias in a fusion encoder was found; requires
  calibrated projection machinery per camera; risk of hard-coding a foveation prior that pick-and-place
  localization phases (pre-contact, object far from EE) do NOT want — exactly GAP's failure phase.
- **Cost:** small params (per-head MLP ≈ 10–50K) but the highest engineering cost; needs extrinsics
  plumbing and breaks camera-agnosticism.

### (e) [added] State token(s) in the QUERY stream via self-attention (Q-Former wiring)
State token(s) join the 8 queries; self-attention mixes them between cross-attn layers; state tokens do
NOT cross-attend to vision and are dropped from the output.
- **For:** the InstructBLIP ablation is the only large-scale causal isolation of privileged-signal query
  conditioning (+4.3/+7.6) [verified]; BLIP-2's masking discipline shows how to keep the privileged
  signal out of the image pathway [verified]; pi0/GR00T/RDT all place state in the query-side stream at
  scale [verified]; Octo's readout masking shows how to keep an output read-only [verified].
- **Against:** requires self-attention among queries — if PerceiverFuse currently has none, this adds a
  layer; more moving parts than (a); with only 8+1 tokens, self-attention is cheap but the state token
  output must be discarded to avoid a GAP-style content leak into the head.
- **Cost:** state MLP (≈0.1M) + one self-attn layer at d=256 (≈0.26M + LN) if absent; masks are free.

## 5. Ranked concrete designs for PerceiverFuse v0.4

All three keep: frozen ViT-B/16 per cam; patch tokens of all 3 cams projected to d=256 and concatenated
into ONE K/V set (single joint cross-attention — consistent with our v0.2 early>late fusion finding);
8 output tokens fed UN-POOLED to the Diffusion Policy transformer (v0.3 arm); future-state aux (§6).

### Rank 1 — "StateAdd": additive state-conditioned queries, zero-init gated  ⟵ run first
```
s ─ MLP(d_s→256→8·256, GELU) ─ reshape 8×256 ─┐
                                              ▼
q_i' = q_i  +  tanh(α_i) · u_i        (α ∈ R^8, init 0; q_i learned, kept)
q'  ──► [cross-attn ×L over patch K/V, unchanged] ──► 8 tokens → policy
```
- Where state enters: query formation only, before layer 1 (optionally re-added before each layer,
  DAB-DETR-style, as a +variant). Queries stay learnable. Cross-attn depth unchanged (keep current L;
  1–2 layers is consistent with NoContactNoWorries succeeding at L=1).
- At α=0 this is EXACTLY the v0.3 unpooled encoder → clean A/B, no re-pretraining required to start
  (can even be finetuned onto the existing checkpoint; the gate guarantees no regression at init).
- Param delta: +0.28M (d_s→256→2048 shared MLP) + 8 scalars ≈ +0.3M (~1% of PerceiverFuse-scale module).
- Citations: Perceiver-IO 2107.14795; DAB-DETR 2201.12329; NoContactNoWorries 2606.24450; gating —
  Flamingo 2204.14198, RT-1 2212.06817, DiT 2212.09748.

### Rank 2 — "StateQTok": Q-Former wiring — state token in the query stream
```
s ─ MLP(d_s→256) ─► t_s (1–2 tokens; e.g., t_current, t_command/EE-target)
[q_1..q_8, t_s] ─ self-attn (full) ─► cross-attn: ONLY q_1..q_8 attend to patch K/V
      (t_s masked out of cross-attn, BLIP-2-style)   ×L blocks
output: q_1..q_8 only (t_s DROPPED — Octo-readout-style isolation in reverse)
```
- Where state enters: as content in the query stream, mixed into the 8 queries by self-attention at
  every block — per-layer re-injection for free, and the queries can attend to state *selectively*
  per-head rather than receiving one global additive shift.
- Queries stay learnable; cross-attn depth unchanged; needs 1 self-attn layer per block among 9–10
  tokens (negligible FLOPs).
- Param delta: +0.1M (state MLP) + ~0.26M per added self-attn layer if PerceiverFuse has none ≈ +0.4M.
- Citations: BLIP-2 2301.12597; InstructBLIP 2305.06500 (the +4.3/+7.6 ablation is the single best
  number behind this design); pi0 2410.24164, GR00T 2503.14734, RDT 2410.07864 (state-on-query-side at
  scale); Octo 2405.12213 (attention isolation).
- Why not first: two structural changes at once (self-attn + masks) vs Rank 1's one-liner; the
  InstructBLIP evidence transfers across a bigger domain gap than NoContactNoWorries does for Rank 1.

### Rank 3 — "StateFiLM": adaLN-Zero on the query stream, state as the conditioning vector
```
c = MLP(d_s→256)                     (optionally concat diffusion-free; encoder-side only)
per sublayer (attn, MLP) of the query stream:
   h ← γ_ℓ(c)⊙LN(h) + β_ℓ(c);   sublayer output scaled by α_ℓ(c), α-head zero-init
queries stay learned; cross-attn ×L unchanged; 8 tokens out
```
- Where state enters: multiplicative/additive per-channel modulation of the query stream at every
  sublayer; never any state-derived token content.
- Param delta: 256→3·256 head per sublayer ≈ 0.20M×(2L); for L=2 ≈ +0.8M (heads can be low-rank to
  halve this).
- Citations: DiT 2212.09748 (adaLN-Zero > cross-attn > in-context, ~free Gflops); DiT-Block Policy
  (prior corpus, low-data robot replication, caveat); RT-1 2212.06817 / BC-Z 2202.02005 (FiLM lineage).
- Why third: strongest mechanism evidence but weakest "state specifically" evidence; γ,β re-weight
  channels rather than re-aim queries, which is a weaker implementation of Jia Qi's hypothesis. Best
  used as the comparison arm that tests whether *any* state conditioning suffices vs query FORMATION.

**Control arms for the matrix:** (i) v0.3 unpooled, no state (baseline); (ii) state as 1 extra K/V token
(design b — the equal-citizen control); (iii) optional later: Rank 1 + EE-projection attention bias
(design d) once calibration plumbing exists.

## 6. Cross-cutting requirements (from verified sources)

1. **Zero-init gate on every new state path.** Flamingo tanh(α) (+4.2 overall, 70.7 vs 66.5); RT-1
   identity-init FiLM; DiT adaLN-Zero identity-init. Also gives exact v0.3 equivalence at init — free
   regression protection given our history of insertion-mechanism failures (E5).
2. **Keep state OUT of the value pathway.** The fuse output must be computable from vision values only,
   with state steering attention (AdapTac's bottleneck; GAP's −15.8% concat anti-pattern; BC-Z's negative
   result). The policy already gets raw state separately — the encoder must not become a second, easier
   conduit for it.
3. **Fix the aux-loss interaction.** Our pretraining predicts state from vision; with state as input,
   that objective is copyable. Replace with FUTURE-state prediction (Δs over horizon h≈action horizon),
   the AdapTac pattern where the future-signal aux was the dominant driver (Flip 50→90; observed-signal
   variant only 70) — also consistent with FRAPPE's near-horizon rule (prior corpus).
4. **Which state features form the query matters.** NoContactNoWorries: current+commanded asymmetric
   queries beat symmetric pooling by +0.15 F1; GAP's proprio was 6D EE pose + gripper. For RoboCasa BC:
   EE pose + gripper opening (+ joint angles secondarily); if a commanded/target signal exists at
   encoder time, give it its own query-conditioning slot rather than averaging it in.
5. **Optimization guardrails still needed.** Query-side wiring reduces but does not eliminate the GAP
   shortcut (state still shapes gradients through attention); keep per-dim state dropout (DiT-Block
   Policy, prior corpus) and consider GAP-style gradient down-weighting only if the shortcut signature
   appears (probe: attention entropy over patches collapsing while validation success stalls in the
   reach/localization phase).

## Sources

| # | Paper | ID / URL | What it contributed | Fetch status |
|---|---|---|---|---|
| 1 | AdapTac — Adaptive Visuo-Tactile Fusion w/ Predictive Force Attention | arXiv:2505.13982 (html v2) | exact force-as-query wiring, ablations | verified |
| 2 | NoContactNoWorries (contact via vision+proprio) | arXiv:2606.24450 (html) | direct state-as-query precedent, d=256, ablation | verified |
| 3 | GAP — When would Vision-Proprioception Policies Fail | arXiv:2602.12032 (PDF, text-extracted), ICLR 2026 | −15.8%, shortcut mechanism, phase evidence, Octo-VP | verified |
| 4 | Perceiver IO | arXiv:2107.14795 (ar5iv) | query-construction principle | verified |
| 5 | BLIP-2 | arXiv:2301.12597 (ar5iv) | Q-Former: 32×768 queries, x-attn every other block, text via self-attn | verified |
| 6 | InstructBLIP | arXiv:2305.06500 (ar5iv) | instruction-aware queries ablation +4.3/+7.6 | verified |
| 7 | Flamingo | arXiv:2204.14198 (ar5iv) | 64 latents, resampler ablation 70.7 vs 66.6/66.7, tanh-gate 66.5 | verified (K/V-concat detail & layer count UNVERIFIED) |
| 8 | Octo | arXiv:2405.12213 v2 (html) | readout isolation; NO proprio statement in v2 (checked) | verified |
| 9 | pi0 | arXiv:2410.24164 (html) | state → linear → action-expert block; blockwise attn | verified |
| 10 | GR00T N1 | arXiv:2503.14734 (html) | per-embodiment state MLP; x-attn to VL tokens; adaLN timestep | verified |
| 11 | RDT-1B | arXiv:2410.07864 (html) | proprio MLP in main stream; alternating x-attn injection, 48→32 | verified |
| 12 | HPT | arXiv:2409.20537 (html) | proprio/vision stems → 16 learned tokens each | verified |
| 13 | PerAct | arXiv:2209.05451 (ar5iv) | proprio tiled+concat pre-Perceiver; 2048×512 latents | verified |
| 14 | ACT / ALOHA | arXiv:2304.13705 (ar5iv) | state as 2 extra tokens in 1202×512 | verified |
| 15 | RT-1 | arXiv:2212.06817 (ar5iv) | identity-init FiLM; no proprio input | verified |
| 16 | BC-Z | arXiv:2202.02005 (ar5iv) | FiLM in 4 ResNet blocks; proprio "did not improve" (App. L) | verified |
| 17 | DiT | arXiv:2212.09748 (ar5iv) | adaLN-Zero > x-attn > in-context (Fig. 5, ~9.5/10.6/11.4/19); zero-init | verified (FID values approximate, read from figure) |
| 18 | DAB-DETR | arXiv:2201.12329 (ar5iv) | conditioned queries 45.7@50ep vs 43.3@500ep; modulated attn | verified |
| 19 | 3D Diffuser Actor | arXiv:2402.10885 (ar5iv) | proprio as tokens + relative-geometry attention 81.3 vs 71.3 | verified |
| 20 | Robot-centric pooling | arXiv:2411.04331 | possible state-as-query pooling precedent | **UNVERIFIED** (fetches failed) |

Prior-corpus reuse (verified in REDESIGN_RESEARCH_20260729.md, not re-fetched): DiT-Block Policy
(adaLN-Zero low-data), contact-gated fusion 2604.01414, FRAPPE 2602.17259, AFA 2502.03270.

**Known gaps:** (i) no paper runs the exact three-way ablation (query-conditioning vs K/V token vs FiLM)
in one architecture — our experiment matrix would be the first; (ii) Flamingo resampler internals
partially unverified; (iii) robot-centric pooling unverified; (iv) GAP's attribution of the
proprio-hurts observation to the Octo paper is not supported by the Octo v2 text we fetched.
