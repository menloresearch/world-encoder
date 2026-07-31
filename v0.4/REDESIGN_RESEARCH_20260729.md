# Redesign research — what latent shape / objective / insertion actually pays off downstream

**Date:** 2026-07-29. **Companion:** LITERATURE_20260728.md (force-fusion precedent, FLARE/Du aux-loss
reconciliation, probe↔downstream inversion). This report does NOT repeat that; it answers the REDESIGN
question: latent shape/capacity, objectives with policy-level evidence, insertion mechanisms, why-latents-fail.

**Method:** 5 angles → 22 fetches (21 unique papers) → 110 claims. Verification COMPLETE: one full-paper
agent per source checked every claim against the full text (second pass 2026-07-29; first pass died on
rate limits at 25/110). Tags: `[verified]` confirmed; `[verified, corrected]` detail fixed in place;
`[verified, caveat]` confirmed with a material scope limit. Nothing remains single-pass.

## Executive verdict (final, post-verification)

1. **The single pooled vector is the common loser in every comparison that contains one.** Dense patches,
   slots, and query-token sets each beat pooled somewhere; nothing loses to it. Drop the 8-queries→pool
   step; output a token SET. Survives verification across 6+ sources. [verified; two caveats: SOLD's
   set-vs-pooled is cross-method, ORC's slots are supervised]
2. **Insertion mechanism is now the strongest-verified thread.** Same frozen DINO features: 11.7% fed
   as-is → 8.3% full-finetune → 28.3% LoRA → 51.7% ProgressiveNet → 86.7% SpawnNet adapters — mechanism
   alone moves the number 7x [verified]. AdapTac: attention fusion WITHOUT its aux (67%) loses to
   vision-only (73%) [verified]. adaLN-Zero > cross-attn holds as a low-data default, not law
   [verified, caveat]. Concat no-lift replications hold.
3. **Our E4 aux-loss harm is placement, not concept — fully verified.** FRAPPE: future-latent cosine to a
   FROZEN target on dedicated prefix tokens, h=8, λ=0.05; dose curve and horizon digit-exact. New live
   variable: TARGET FORMAT — discrete latent-token supervision beats continuous regression in all 8
   LIBERO cells (+2.7/+2.2 avg; the prior kill of this claim is OVERTURNED). Our failed aux was
   continuous cosine. [verified]
4. **Objectives: the verified variable is TEMPORAL/dynamics grounding, not masked-recon-vs-not.** The top
   action-R² frozen model IS masked recon (VideoMAE, video); the losers are static image AEs/tokenizers.
   Our PER-FRAME masked recon lacks temporal context — the caution reaches us via the temporal argument,
   not via recon per se. MCR's win is a RECIPE (DROID data + objectives), dynamics-alignment the largest
   ablation term. [verified, reframed]
5. **E6 force arm: do not fuse into the pooled 256-d latent.** Corrected: TranTac's losing baseline is a
   LEARNED MLP+transformer wrist-F/T fusion, losing on AVERAGE only (wins 2 of 4 tasks) — not raw
   passthrough. What wins: separate contact-gated tokens (signal there is joint torque, not wrist F/T),
   late fusion, future-force aux (aux ALONE drives 50→90 on Flip). [verified, corrected]
6. **Slots vs dense split is real but supervision-dependent.** ORC slots trained on ground-truth
   annotations; DOCIR's collapsing baseline had GT masks + target-ID — evidence about GROUPING under
   oracle perception, silent on learned slots. Slot-MPC vs DINO-WM contradiction RESOLVED:
   protocol-specific 0.00s. [verified, caveat]
7. **Two computable diagnostics survive with exact definitions:** Burns Jaccard R²=0.85 (CLS attention,
   all heads, last block, binarized, vs PASCAL VOC masks; ViTs ONLY) and MCR manipulation-centricity
   R=0.93 (binarized Grad-CAM ∩ SAM-2 masks of EE + task objects; N≈6 representations, threshold
   unspecified). Small-N, but free on current checkpoints. [verified, caveat]

---

## 1. Latent shape & capacity (set vs pooled; how many tokens)

**Set/slot latents beat pooled where it's hard:**
- ManiSkill BC study (transformer policy): under distractor-color shift, slot SAVi drops 0.69→0.55 (−14pp)
  while pooled R3M collapses 0.88→0.26. Pick Cube: only slots nonzero (SAVi 0.56±0.01; R3M 0.0; frozen
  DINO 0.0 — and 0.0 on ALL their tasks; Place Cube in Bin: everything fails). [verified; SAVi pretrained
  in-domain, only one OCR method tested] https://arxiv.org/abs/2506.19408
- ORC slot-VLA: 4-28 slot/relation tokens replace 256 dense tokens, competitive on LIBERO+. [verified,
  caveat: slot encoder SUPERVISED on ground-truth object annotations, then frozen — compact-token wins
  may not transfer without that supervision] https://arxiv.org/abs/2511.06754
- SOLD: slot-latent world model beats DreamerV3/TD-MPC2 (monolithic latents) on relational manipulation
  (74.6% vs 33.1/17.3). MBRL, reward curves. [verified, caveat: cross-method comparison — no controlled
  same-encoder set-vs-pooled ablation exists in the paper] https://slot-latent-dynamics.github.io/

**But the split with dense is real:**
- ORC beats dense OpenVLA on object-centric suites (L-Goal 0.86 vs 0.77; L-Object 0.91 vs 0.70) and LOSES
  on long-horizon (L-Long 0.31 vs 0.56) and spatial (0.60 vs 0.72). Paper's own attribution: 4 slots vs
  up to 29 objects — compactness becomes the limitation. [verified] 2511.06754
- Slot-MPC: DINO-WM (dense patches) 0.00 on all four planning tasks vs 4-slot Slot-MPC 0.22–0.64 — but
  the 0.00s are PROTOCOL-SPECIFIC: vision-only (proprio removed) + full-trajectory goal horizons, both
  known DINO-WM killers; under DINO-WM's native H=25 protocol it gets 0.56 on Button Press. Tension with
  yesterday's verified dense-wins-planning result RESOLVED — no contradiction. [verified, corrected]
  https://arxiv.org/abs/2605.14937
- DOCIR: disentangled task-relevant/obstacle/robot streams get Pick 0.95 vs 0.56 vs flat 0.15, Place 0.81
  vs 0.25 vs 0.00. CORRECTED: the collapsing "OCR" baseline (0.56→0.00 at 4 cubes/3 plates) uses
  GROUND-TRUTH per-object masks + a target-ID embedding — even ORACLE object decomposition collapses in
  clutter without interest/obstacle GROUPING. Strengthens task-relevance structure; says NOTHING about
  learned slots. PPO + oracle perception, not BC. [verified, corrected] https://arxiv.org/abs/2503.11565

**Capacity:**
- More slots ≠ better, but not flat either: OC 0.77@4 / 0.65@8 / 0.77@16 (dips, recovers); ORC 0.86@4 →
  0.74@8 → 0.72@16 — "4 beats 16" holds for ORC. Caveats: ablation counts OBJECT slots (ORC "4" ≈ 20
  total tokens with relations); the paper itself blames insufficient capacity on L-Long. [verified,
  corrected] 2511.06754
- Slot-MPC: 4×128 slots = 99.3% dim reduction vs 196×384 patches, plans 0.42±0.01 s vs 144.37±0.83 s,
  and wins. [verified] 2605.14937
- Attentive Feature Aggregation (trainable cross-attn query over frozen patch tokens, replacing CLS/pool):
  in-domain 63.1%→66.4%, up to ~3x robustness under scene perturbation. Cheapest published pool→query
  upgrade. [verified, caveat: measured WITH the TE time encoding concatenated; AFA dropped from the
  paper's v3 split — cite v1/v2] https://arxiv.org/abs/2502.03270

**Shape at consumption:** always a token set in the policy sequence — ORC: slot tokens concat with
language+proprio [2511.06754]; SAVi study: K slots × H frames + learnable [ACT] readout [2506.19408];
Octo: 256+64 patch tokens per camera [https://arxiv.org/html/2405.12213v2]; VPP: learned query tokens
(shape task-dependent; Calvin 16×14×384) via cross-attention, ablation 4.33→3.86
[https://arxiv.org/abs/2412.14803]. All verified.

## 2. Force/tactile fusion (E6-relevant)

All verified; consistent with yesterday's FoAR/RDP/FACTR picture. Sensor heterogeneity: joint-torque
estimates (2604.01414), taxel arrays (AdapTac), camera-based fingertips (ViTacFormer), fingertip IMUs
(TranTac) — only TranTac's BASELINE is wrist 6D F/T.

- **Concat no-lift replications:** joint-torque concat = vision-only (both 30.0% avg; paper's own
  attribution: unfiltered free-space torque noise) [verified] https://arxiv.org/html/2604.01414.
  Visuo-tactile concat 40% vs vision-only 73%, loses all 3 tasks; same visual encoder [verified]
  https://arxiv.org/abs/2505.13982.
- **ViTacFormer, corrected:** headline gap (10/10/9/9 vs naive-fusion ACTw/T 6/6/4/4) is a FULL-SYSTEM
  comparison. Appendix ablations: cross-attn→concat costs only 10-30pt with Peg LEAST affected (100→90 —
  contradicting the main text), while concat + prediction head + curriculum still crushes naive fusion
  (90/70/70/70 vs 60/60/40/40). MOST of the gap = prediction head + curriculum; fusion mechanism is
  second-order. Tactile = camera-based fingertip sensors, policy consumes extracted F/T channels (20 ch);
  ACT-style CVAE, not diffusion; end-to-end per task. [verified, corrected] https://arxiv.org/pdf/2506.15953
- **A LEARNED wrist-F/T fusion can lose to vision-only — on average:** TranTac, four insertion tasks:
  vision + 6D F/T (MLP→512-d + transformer fusion, trained end-to-end, same diffusion head) 50% avg vs
  vision-only 61.25% vs TranTac tactile tokens 78.75% (80/80/65/90). Per-task the F/T arm WINS 2 of 4
  (circle-square 70v60, USB 40v30). NOT raw passthrough — do not cite it as that. Setting is materially
  E6-comparable (wrist 6D F/T + diffusion policy); the bar is real and beatable. [verified, corrected]
  https://arxiv.org/html/2509.16550v1
- **What wins — gating + late fusion:** contact-gated adaptive vision-torque fusion 82% avg vs
  torque-gating-alone 68% vs vision-only/concat 30%. Input gating (no contact → learnable placeholders)
  = MAJORITY of the improvement (30→68); output CFG-style blend ϵ̂final = ϵ̂vision + w(ϵ̂torque − ϵ̂vision)
  adds 68→82. Signal = Franka JOINT external-torque estimates, NOT wrist F/T — wrist applicability is our
  extrapolation. [verified, caveat] 2604.01414
- **Force-as-query cross-attention (AdapTac):** force queries over visual/tactile K/V, per-modality scalar
  mixing before the diffusion head: 93% avg vs vision-only RISE 73%, concat 40%, FoAR 50%. [verified] 2505.13982
- **Aux losses:** torque-as-aux-goal underperforms vision-only (28% vs 30% — one trial in 50; "unchaseable"
  is our gloss) [verified] 2604.01414. Future-force prediction aux is load-bearing AND isolated: AdapTac
  Flip ablation — prediction loss ALONE drives 50→90%; query injection mainly cuts episode length
  (166→113); attention fusion WITHOUT the aux (67%) loses to vision-only (73%) [verified] 2505.13982.
  ViTacFormer future-tactile head (normalized L1 ≈0.08, chaseable, fed back as input tokens): ablating it
  drops 100→70 / 100→70 / 90→60 / 90→70 [verified] 2506.15953.
- ViTacFormer 11-stage task: HNS 0.61 (ACT) → 0.72 (naive tactile) → 0.88 (full), strict completion
  0/10/70%. Needs a 75/25 scheduled-sampling curriculum or training destabilizes. [verified] 2506.15953

## 3. Pretraining objectives with policy-level (not probe) evidence

- **MCR** (frozen encoder + IL — our setting): vision↔state-action-dynamics contrastive + BC-like action
  loss + time-contrastive, trained on DROID. +14.8% over strongest baseline, 4 domains / 20 tasks incl.
  3 RoboCasa tasks; real UR5e 23/30 vs R3M 11/30, VC-1 13/30, MVP 10/30. Ablation digit-exact: w/o
  dynamics-alignment 83.2→66.2 (vs 71.3 no-action-loss, 72.0 no-time-contrastive) — largest single term.
  Wins with a pooled frozen ResNet-50. [verified, caveat: gains attributable to the RECIPE — DROID data +
  objectives — not the objective alone; venue is preprint, NOT ICLR 2025] https://arxiv.org/abs/2410.22325
- **Reconstruction vs action structure — REFRAMED:** at identical ~20 dB PSNR, frozen action-R² spans
  −0.01 to 0.46 — but the TOP scorer IS masked recon (VideoMAE, masked VIDEO patches); the near-zero/
  negative models are STATIC image AEs/tokenizers (SDXL VAE −0.55). The paper's variable is TEMPORAL
  video pretraining vs static/recon-only — do NOT cite it against masked patch recon per se. ID aux
  amplifies existing temporal structure, can't create it: +0.45 on V-JEPA-2 (0.40→0.85), 0.00 on Dreamer 4.
  Recommends video-predictive encoder + small ID head (V-JEPA-2+ID 0.85 vs VideoMAE+ID 0.75, Web-DINO+ID
  0.16). Probe-only, self-flagged; our per-frame masked recon lacks temporal context, so the caution
  applies to us via the temporal argument. [verified, reframed] https://arxiv.org/html/2606.07687v1
- **VPP:** fine-tuned-then-frozen video-diffusion future predictor, consumed as a query-token set via
  cross-attention: Calvin ABC-D avg length 4.33 vs VC-1 1.23 / Voltron 1.54 / Stable-VAE 2.58 (fair swap:
  baselines fine-tuned on the same video data, same head). Multi-layer aggregation matters: 4.33 vs 3.60
  final-layer-only (layers 3/6/9/12: 3.72/3.88/4.29/4.05). +18.6% relative Calvin over prior SOTA;
  +31.6% real dexterous. ICML 2025. [verified] https://arxiv.org/abs/2412.14803
- **HRP, corrected:** affordance fine-tuning (contact points, hand pose, active-object boxes) ≥+15% on
  5 real tasks, 3000+ trials — encoder fine-tuned END-TO-END with BC; ZERO frozen-encoder evaluation
  anywhere in the paper — cannot be cited for or against frozen pipelines. Real loss ablation (Table VIII,
  Ego4D): full 77% avg → 38% no-object / 50% no-hand / 50% no-contact (larger drops than first extracted;
  the 83% was a different table, different eval day). RSS 2024. [verified, corrected]
  https://arxiv.org/abs/2407.18911
- **Negative priors:** 8 frozen PVRs no more sample-efficient than from-scratch in MBRL; scratch wins even
  OOD. Failure traced to missing task/reward info — reward-pred error r=−0.66 p=0.004 vs dynamics error
  r=−0.22 p=0.4, measured on ONE analysis task (Pendulum Swingup) and hedged by the authors ("might",
  "hypothesize"). NeurIPS 2024. [verified, caveat] https://arxiv.org/abs/2411.10175. Manipulation-designed
  PVRs rank 7th/15 under shift (in-distribution they DO win); ImageNet probe predicts nothing (R²=0.07,
  ViTs); emergent segmentation predicts OOD success (R²=0.85, ρ=0.90 — ViTs ONLY); MoCo-v3 40% vs MVP 0%
  on real ALOHA (n=1 pair, ACT likely fine-tuned); MAE masked-PIXEL recon inconsistent under shift.
  CoRL 2024. [verified] https://arxiv.org/abs/2312.12444 — also fetched as the kayburns.github.io project
  PDF, same paper.
- Cheap pre-run diagnostic: MCR manipulation-centricity = avg Jaccard of BINARIZED Grad-CAM vs SAM-2 masks
  of EE + task objects; Pearson R=0.93 aggregate. Reimplementation caveats: threshold unspecified, N≈6
  representations, per-domain strength varies. [verified, caveat] 2410.22325

Net (reworded post-verification): the objective axis with support is temporal/dynamics grounding. Static
or per-frame recon-only objectives have no policy-level wins and probe evidence against them; masked recon
WITH temporal context (VideoMAE) tops the frozen action-R² table. Recipes with policy-level wins inject
action/dynamics signal — MCR contrastive (as part of a data+objective recipe), ID heads, VPP future
prediction. Our per-frame masked recon sits on the wrong side of the temporal split.

## 4. Insertion mechanisms — how an added representation helps instead of hurts

- **Explicit chaseable targets beat implicit alignment:** of four ways to add latent actions to a VLA,
  direct supervision on discrete latent tokens is most effective (LA-Direct 96.6% vs 85.8% baseline
  LIBERO-Long; LA-Tok 78.0% vs 60.5% RoboTwin 2.0; +28/+20pp real). Implicit alignment (LA-Align) also
  helps (+9.0/+10/+21pp) — cosine at layer 17/29; EARLY-layer alignment hurts (−1.5). LA-Cond: zero
  negative transfer across 10 tasks. TARGET FORMAT: discrete supervision beats continuous MSE regression
  +2.7/+2.2 avg, all 8 LIBERO cells — the prior kill of this claim is OVERTURNED (see caveats); scope
  LIBERO-only, continuous still beats no-latent. [verified] https://arxiv.org/abs/2605.04678
- **FRAPPE — the E4 counterfactual, fully verified:** future-latent cosine loss to frozen VFM embeddings
  (CLIP/DINOv2/ViT) of the frame h ahead, on dedicated learnable PREFIX tokens; loss attaches ONLY to
  prefix outputs, at DiT layer 21/28. Dose curve digit-exact, non-monotone with a real optimum:
  λ = 0 / 0.001 / 0.02 / 0.05 / 0.1 / 0.5 → 14.0 / 18.5 / 26.4 / 32.5 / 22.0 / 23.5%. Horizon must be
  near: h=8 35.3%, h=16 35.0%, h=32 29.7%. Beats RDT/π0.5 on RoboTwin (57.5/25.5 vs 47.4/15.1, 45.4/13.3;
  note π0 hits 57.1 Easy — Easy margin +0.4, Hard +11.4). [verified] https://arxiv.org/pdf/2602.17259 —
  Fixed-frozen target like our E4's, yet it helps; even overdosed λ=0.5 beats λ=0. Placement is the
  confirmed variable; target FORMAT (discrete vs continuous) is the new second variable per 2605.04678.
- **DiT-specific conditioning:** per-layer cross-attention and in-context conditioning unstable — 0%/0%
  at 10 DDIM steps vs 50%/100% adaLN-Zero; cross-attn at 100 steps still 38%/70%. Zero-init of the
  conditioning scale alone worth 16% (38%/80% without). Per-camera CNN tokenizers beat conv-stem tokens
  even at 150M vs 115M params (13%/20% vs 50%/100%); per-dim proprio dropout vs shortcut overfit.
  [verified, caveat: LOW-DATA finding — 2 real tasks, ±11-15% error bars; concurrent work hit SOTA WITH
  per-layer cross-attn at 26K episodes. adaLN-Zero = robust default, not law]
  https://dit-policy.github.io/resources/paper.pdf
- **Attention isolation:** Octo readout tokens attend to obs/task tokens, never attended back — enforced
  by a block-wise attention mask, so the added pathway cannot perturb base processing. New modalities at
  finetune = new lightweight encoder + positional embeddings only. Peg insertion 70% vs 10% scratch vs 5%
  VC-1 — CORRECTED framing: that is pretraining-vs-scratch on the same demos (F/T present as a new input),
  NOT an F/T on/off ablation. RSS 2024. [verified, corrected] https://arxiv.org/html/2405.12213v2
- **SpawnNet — the mechanism dose-response (strongest new fact in the corpus):** SAME frozen DINO
  features: 11.7% fed as-is, 8.3% full-finetune, 28.3% LoRA, 51.7% ProgressiveNet, 86.7% SpawnNet
  multi-layer adapters (v2 Table 3) — insertion mechanism alone moves the number 7x. Headline: 86.7±6.4
  vs R3M-frozen-pooled 20.0±7.1 vs scratch+aug 66.7±7.6 (all +d variants, apples-to-apples). CLS-tiled
  ablation −24.9 on Place Bag; last-layer-only −13.3 (sim drop only −1.3 — multi-layer rule rests mainly
  on one real task). [verified] https://arxiv.org/abs/2307.03567

Design rules (post-verification): (1) added signal gets its own tokens/pathway, never mixed into obs
features; (2) gate initialized to zero contribution; (3) aux target chaseable or isolated from the
encoder — and prefer DISCRETE targets where format is a choice; (4) for a DiT at our data scale,
adaLN-Zero over cross-attention (robust default, not law); (5) tap multiple encoder layers, not the last.

## 5. Why learned latents fail policies (contrarian angle)

- **Missing task-relevant information**, not representation quality: PVR-MBRL reward-error correlation
  (r=−0.66) vs weak dynamics-error correlation (r=−0.22); scratch latents organize by reward (UMAP,
  qualitative). Single analysis task (Pendulum Swingup), author-hedged causality. [verified, caveat] 2411.10175
- **Temporal entanglement:** time-invariant per-frame encoders map states needing different actions to
  near-identical latents (Markov violation). Sinusoidal task-progress encoding on frozen features: +16.4%
  Bin Picking; beats FLARE-style latent STACKING (Wilcoxon p≈4.38e-5) — note their "FLARE" is INPUT
  stacking of 3 latents + differences, NOT an aux objective, so the tension with our verified FLARE-aux
  gains largely dissolves. Aggregate significance only (VC-1/iBOT do better with stacking). VIP gains
  ~15pp from the fix. Our per-frame pooled latent sits in this failure class. [verified, caveat: v3 is a
  retitled split paper — cite v1/v2] https://arxiv.org/abs/2502.03270
- **Probe metrics don't transfer; spatial-attention metrics do:** ImageNet linear probe R²=0.07 vs
  emergent-segmentation Jaccard R²=0.85 [verified; metric = CLS attention, all heads, last block,
  binarized, vs PASCAL VOC masks; ViTs ONLY — ResNets differ, probe R²=0.80 on ~3 models] 2312.12444;
  manipulation-centricity R=0.93 [verified, N≈6] 2410.22325. Both computable on our checkpoints today.
- **Frozen-ness is a suspect — but naive unfreezing is worse:** SOLD found finetuning the slot encoder
  necessary on Pick (qualitative recon figures only, NO quantitative frozen-vs-finetuned policy ablation)
  [verified, caveat]; SpawnNet's "frozen may hinder" is verbatim, but its own Table 3 shows full-finetune
  (8.3%) UNDER frozen-as-is (11.7%) — the cure is adapters, not unfreezing [verified] 2307.03567; HRP has
  ZERO frozen-encoder evidence, supports this only via co-adaptation-was-needed [verified, corrected] 2407.18911.
- **Entangled background:** embeddings mix task-relevant and background features, obscuring cues from the
  action decoder — ORC's motivation (partly citing prior work). [verified] 2511.06754

## What this means for v0.3/v0.4 — candidate changes, ranked by evidence

1. **Un-pool: emit the 8 query tokens as a set; policy consumes tokens, not a 256-d vector.** Survives
   full verification (slot/set results here + Patch Policy/DINO-WM yesterday + AFA/VPP/Octo). Capacity is
   NOT the fix — 4 structured tokens beat 16 for ORC (0.86 vs 0.72). Cheapest test: keep the trained
   encoder, drop the pooling op, feed the 8 query outputs to the DiT as extra tokens via a zero-init
   gate. One run, existing checkpoints.
2. **Change the insertion, encoder fixed — now the strongest-verified thread.** SpawnNet: same features,
   7x spread by mechanism alone; AdapTac shows fusion without the right aux loses to vision-only.
   Cheapest: rerun the E5 concat arm with (a) z on read-only tokens (Octo-style), (b) adaLN-Zero — if
   either recovers baseline, the E5 negative was insertion, not information. adaLN-vs-cross-attn carries
   a low-data caveat; both arms are cheap.
3. **Redesigned E4 point (one run): future-latent loss on dedicated prefix tokens in the DiT, h≈8,
   λ≈0.05.** FRAPPE recipe verified digit-exact (placement + dose + horizon; frozen target fine). Second
   variable in the same experiment: DISCRETE target tokens vs continuous cosine — discrete won all 8
   LIBERO cells in 2605.04678, and our failed aux was continuous cosine.
4. **Objective swap for v0.4 pretraining — reworded:** the verified axis is temporal/dynamics grounding,
   not masked-recon removal. Evidence-backed options: dynamics-alignment contrastive (MCR — recipe-level
   evidence, largest ablation term 83.2→66.2), small ID head (amplifies existing temporal structure),
   temporal context in the recon target (VideoMAE-style masked VIDEO recon tops the frozen action-R²
   table; per-frame masked recon is the indicted variant). FIRST, free diagnostics on current
   checkpoints: Burns attention-Jaccard and MCR manipulation-centricity (exact definitions above) — low
   scores = mechanism confirmation before any retrain.
5. **E6 verdict: do NOT run the force arm through the pooled-256d fuse.** Force as separate tokens,
   windowed history (yesterday's FoAR recipe), contact-gated at input (gating = majority of the win,
   30→68 — but that signal is joint torque, not wrist F/T), fused late; future-force-prediction aux
   (isolated as THE driver in AdapTac: 50→90 on Flip; fusion without it loses to vision-only). Keep the
   raw-passthrough bar, reworded: TranTac shows even a LEARNED wrist-F/T fusion can lose to vision-only
   on average — the bar is beatable, and raw passthrough is not the ceiling.
6. **Longer-term (v0.4+): task-relevance structure, not generic slots.** DOCIR's corrected reading is
   STRONGER here: even oracle per-object masks collapse in clutter without interest/obstacle grouping.
   But both slot papers carry supervision caveats (ORC GT annotations; DOCIR GT masks), and ORC's own
   limitation: "struggles to scale in more complex scenarios". Middle path unchanged: per-view spatial
   tokens, compress channels not space (yesterday's 2603.12231, unverified), let policy attention select.
7. **Partial unfreezing — adapters only.** SpawnNet Table 3 reorders this option: full-finetune
   UNDERPERFORMS frozen-as-is (8.3 vs 11.7); LoRA more than doubles it (28.3); multi-layer adapters win
   (86.7). If we unfreeze anything: LoRA/adapters on the Perceiver, backbone frozen — never naive
   finetune at our demo counts.

## Sources

| # | Paper / descriptor | URL | Venue (verified) | Claims | Verified |
|---|---|---|---|---|---|
| 1 | Object-centric vs holistic BC study (SAVi/R3M/DINO, ManiSkill) | https://arxiv.org/abs/2506.19408 | ROBOVIS 2025 | 5 | 5 ok (1 generalization corrected) |
| 2 | ORC slot-token VLA (LIBERO+) | https://arxiv.org/abs/2511.06754 | preprint (NOT ICRA 2026) | 5 | 4 ok + 1 corrected |
| 3 | SOLD slot-latent world model (= arXiv 2410.08822) | https://slot-latent-dynamics.github.io/ | ICML 2025 | 5 | 5 ok, caveats |
| 4 | Latent-action integration study (LA-Direct/Tok/Align/Cond) | https://arxiv.org/abs/2605.04678 | preprint | 5 | 5 ok + kill overturned |
| 5 | Slot-MPC (4×128 slots vs DINO-WM in planning) | https://arxiv.org/abs/2605.14937 | preprint | 5 | 5 ok (0.00s protocol-scoped) |
| 6 | DOCIR disentangled object-centric RL | https://arxiv.org/abs/2503.11565 | preprint | 5 | 4 ok + 1 corrected |
| 7 | TranTac transient tactile tokens | https://arxiv.org/html/2509.16550v1 | preprint | 5 | 4 ok + 1 corrected |
| 8 | Contact-gated vision-torque diffusion fusion | https://arxiv.org/html/2604.01414 | preprint | 5 | 5 ok, caveats |
| 9 | AdapTac force-query cross-attention fusion | https://arxiv.org/abs/2505.13982 | preprint (v2 2025-07-21) | 5 | 5 ok |
| 10 | ViTacFormer visuo-tactile cross-attention | https://arxiv.org/pdf/2506.15953 | no venue (v2 2026-05) | 5 | 4 ok + 1 corrected |
| 11 | MCR manipulation-centric representation | https://arxiv.org/abs/2410.22325 | preprint (NOT ICLR 2025) | 5 | 4 ok + 1 corrected |
| 12 | VPP Video Prediction Policy | https://arxiv.org/abs/2412.14803 | ICML 2025 | 5 | 5 ok |
| 13 | Temporal entanglement + TE/AFA | https://arxiv.org/abs/2502.03270 | preprint (v3 = retitled split; cite v1/v2) | 5 | 5 ok, caveats |
| 14 | Burns et al., emergent segmentation predicts robust manipulation | https://arxiv.org/abs/2312.12444 | CoRL 2024 | 10 (merged w/ #15) | 10 ok |
| 15 | Same paper, second fetch (project PDF) | https://kayburns.github.io/segmentingfeatures/static/segmentingfeatures_paper.pdf | = #14 | — | merged into #14 |
| 16 | SpawnNet dense multi-layer adapters | https://arxiv.org/abs/2307.03567 | unverified (cited ICRA 2024) | 5 | 5 ok |
| 17 | DiT-Block Policy (adaLN-Zero ingredients) | https://dit-policy.github.io/resources/paper.pdf | preprint (arXiv 2410.10088) | 5 | 4 ok + 1 minor corrected |
| 18 | Octo generalist policy (readout tokens) | https://arxiv.org/html/2405.12213v2 | RSS 2024 | 5 | 5 ok (1 reframed) |
| 19 | HRP human affordance pretraining | https://arxiv.org/abs/2407.18911 | RSS 2024 | 5 | 4 ok + 1 corrected |
| 20 | PVRs ineffective for MBRL | https://arxiv.org/abs/2411.10175 | NeurIPS 2024 | 5 | 5 ok |
| 21 | FRAPPE future-alignment prefix tokens | https://arxiv.org/pdf/2602.17259 | preprint | 5 | 5 ok |
| 22 | Action-structure probe study (recon ≠ action R²) | https://arxiv.org/html/2606.07687v1 | preprint | 5 | 5 ok (1 reframed) |

## Methods / caveats

- **Pipeline:** 5 angles, 22 fetches (21 unique papers), 110 extracted claims. First verification pass
  (3-voter adversarial) died on rate limits at 25 claims (7 confirmed, 2 killed, 16 errored). Second pass
  2026-07-29: one full-paper verification agent per source; every claim checked against full text.
  Outcome: ~100 confirmed as stated (many with scope caveats, now inline), 8 corrected in substance
  (TranTac baseline, ORC capacity, ViTacFormer fusion attribution, DOCIR baseline, HRP ablation numbers,
  Octo framing, MCR recipe attribution, DiT-Block boost attribution), 2 reframed (2606.07687 temporal
  argument; Octo peg-insertion), both prior kills adjudicated. Nothing remains single-pass.
- **Recovery-file defect:** the `sources` array had its URL column permuted relative to claim blocks.
  Realigned by content BEFORE verification; the per-paper agents then confirmed every pairing (e.g., VPP
  explicitly not mispaired). Rows 14/15 = same paper fetched twice; effective corpus 21.
- **Killed claim 1 — adjudicated (kill right, reason wrong):** the numbers were exact (R3M 0.88 vs SAVi
  0.69, Push Cube in-distribution) but the generalization fails: the paper attributes R3M's win to
  time-contrastive dynamics pretraining + ~25x more params, NOT slot cost; the pattern REVERSES on harder
  tasks where only slots succeed at all. Cite the adjudicated statement, not the original claim.
- **Killed claim 2 — OVERTURNED:** "discrete supervision consistently outperforms continuous regression,
  +2.7% and +2.2%" is VERBATIM in 2605.04678, and Table 4 confirms discrete wins all 8 cells. The prior
  0-2 panel was wrong (likely could not access the 2026 paper). Scope: LIBERO-only ablation; continuous
  arm = MLP+MSE; continuous still beats no-latent substantially — discrete>continuous, not
  continuous-is-harmful. Directly relevant to the E4 redesign: our failed aux was continuous cosine.
- **General:** several sources are unreviewed 2026 preprints (2602-2606 ids). Force numbers are
  real-robot but sensor-heterogeneous (Sec 2 header); none on MimicGen — transfer to E6 directional, not
  calibrated. The FLARE tensions flagged in the first draft largely dissolve on verification: FRAPPE's
  dose curve is digit-exact, and the TE paper's "FLARE" is input stacking, not the aux loss. Remaining
  soft spots are inline: small n / no seeds in most force papers; Burns metric ViT-only; MCR diagnostic
  N≈6; DiT-Block low-data regime.
