# v0.4 research corpus — all verified research in one file

Merged 2026-08-02 during doc consolidation, each part content-unchanged. Parts, in order:
the post-wave-1 deep-research report (what to run next), the redesign research (latent shape /
objectives / insertion), and the five design-freeze reports 01-05 plus the encoder-swap
protocol literature audit 06.


---

> **APPENDIX (merged 2026-08-02 during doc consolidation, content unchanged). Formerly `research/next/DEEPRESEARCH_NEXT_20260731.md`.**

# DEEP RESEARCH — What gives the compressed latent its best remaining shot (post-wave-1)

**Date:** 2026-07-31 (~1:45pm SGT). **Question registered after** wave-1 closed (ep50 board: hybrid-base 0.90 > sb 0.86 ≈ rc-pool 0.82 ≫ sa-h6 0.54 > rc-cat 0.34 ≫ zo 0.0; claims A/B/F all NO on Square; v0.3 encoder exonerated as tokens; calib footnote landed 0.60 vs 0.64–0.66 band).

**Method / provenance:** 5-angle web sweep → 22 sources fetched → 109 claims extracted → adversarial verification. The original run (104 agents) lost 21 of its top-25 verifications and the entire synthesis to an API rate-limit storm; this report is the recovery: a second verification pass (52 fresh adversarial votes across 35 decision-critical claims, every vote fetched and quote-checked the source) plus synthesis against our ground truth. **Final tally: 39 claims verified (4 at 3-0 in the original run, 35 at 1–2 votes each in the recovery pass), 0 refuted, 0 disputed.** Verifier-added caveats are folded in below as scope notes. Raw pools: original run journal `bc2ecc7a…/subagents/workflows/wf_d81e48eb-b44/journal.jsonl`; recovery `13f334a9…/wf_b1af0a7a-a79/journal.jsonl`.

**⚠ CORRECTION TO THE QUESTION'S PREMISE (data-verified today):** our Square substrate is the robomimic **PH set = 200 demos** (30,154 steps; counted directly in `demo_v141.hdf5` and `image_ft_256.hdf5` on NAS), **not ~1000**. The "1,000 demos" line in PROGRESS.md 07-30 is a carry-over error from the robocasa side (which does use ~1000+ mimicgen demos). All Square conclusions below use 200. This *strengthens* the low-data experiment: the canonical robomimic size-ablation subsets (20%/50% = 40/100 demos) sit directly below our operating point.

---

## 0. TL;DR

**The nulls were the literature's predicted outcome, not an anomaly.** At 160–1000 demos, frozen pretrained encoders lose to end-to-end vision by −42% on average (3-0 verified), and *no* published win exists anywhere for the "extra tokens next to a trainable vision stack" consumption pattern we spent three generations testing. Our token-insertion failure even has an external replication now (EO1+Point: 73.2%→18.6% on RLBench).

**Payoff axes ranked by published support for OUR stack:**

| Rank | Axis | Support | Our cost to test |
|---|---|---|---|
| 1 | **Shift robustness** (Q4) | Multiple independent rollout-level wins, invisible at ID parity — and we already own parity ckpts | **eval-only** |
| 2 | **Low-data ≤100 demos** (Q1) | Every published frozen-rep win lives here; 200 demos is already past the crossover | ~1–2 GPU-nights |
| 3 | **Planning/MPC + edge** (Q3+Q7 merged) | Two 2026 results show 8-token planning works and is 40–130× cheaper; the decisive compressed-vs-dense-patch dynamics comparison in manipulation is *unpublished* — open gap | ~2–3 GPU-nights |
| 4 | Force fusion (Q5-adjacent) | Blocked at our rung-0 double null; needs FACTR-style curriculum first | parked |
| 5 | Policy-side token economy (Q7) | **Dead** — unanimous: token count only matters when an LLM consumes the tokens | — |

**Single most important reframe:** stop trying to *feed z to the policy*. The two live paths are (a) z as the *only* backbone in the ≤100-demo regime, and (b) z as *world-model state for planning* — plus the free option of testing whether the parity tokens we already trained buy robustness under shift.

---

## 1. Verdicts per question

### Q1 — Low-data regime: YES, this alone largely explains the null (as-backbone); test at 10–100 demos
- The paper behind our strongest confirmed claim set (Simple-LfS, arXiv 2212.05749) shows the crossover directly: frozen pretrained reps beat learning-from-scratch **marginally at 10 demos**, LfS **wins at 100** (✓2-0; scope: DMControl, 3 seeds).
- R3M's celebrated frozen win (+20% over scratch, ✓) was measured **exclusively at 5–25 demos (MetaWorld/Kitchen) and 25–100 (Adroit)** — nothing above 100 exists in the paper (✓2-0).
- DMC-VB: at full data **no** pretrained rep beats E2E BC, with or without distractors (✓2-0); cut to **1% (20 trajectories)** and frozen-pretrained wins appear, largest under distractors (✓2-0).
- LaDi-WM's entire manipulation win happens at **10 demos/task** with the world model pretrained on disjoint task-agnostic clips (✓2-0).
- Data scaling laws (UMI, real-world): performance saturates at **~K=50 demos per environment-object pair**; beyond that, more demos ≈ no effect (✓2-0). At 200 demos on a single fixed sim env, the dense baseline is deep in saturation.
- The 2409.20248 null (frozen −42%) was established at **160–1000 demos** and the authors *explicitly flag* the ≤5-demo regime as excluded and likely to flip (✓3-0, original run).
- Counterweight, honestly: the low-N wins above are **as-backbone** (frozen R3M/pretrained encoder IS the vision stack), and several are finetuned rather than frozen (OpenVLA +20.4pp at 10–150 demos is *full-model finetuning*, ✓; frozen DINOv2 as diffusion-policy backbone scored **0.00 vs 0.90 finetuned** in real-world rollouts, ✓2-0 — scope: single Pour-Water ablation). No paper shows a low-N win for tokens-next-to-trainable-vision.

**Fair decisive test:** robomimic's own pre-packaged size-ablation protocol (20%/50% subsets = 40/100 demos of PH, Fig 3 / Tables 27–28, configs auto-generated — ✓2-0) extended down to 10/20. See Experiment 2.

### Q2 — Consumption model: added-tokens has no published win at ANY demo count; the failure is now externally replicated
- **No source in 22 fetched** (nor the prior three research rounds) shows a win for a frozen rep consumed as extra tokens beside a trainable vision stack. The robustness benchmark's own consumption survey confirms: the literature's pattern is always frozen-encoder-AS-backbone + policy head (✓).
- **External replication of our concat kill:** injecting point-cloud tokens into a monolithic pretrained VLA collapses RLBench-10Tasks 73.2% → 18.6% (−54.6pp; our rc-cat was −48pp) (✓2-0, PointACT Table III).
- **The pattern that works when adding a modality:** PointACT's ~300M action expert where **action tokens act as bottleneck queries that cross-attend the new modality's tokens** — 82.3% vs 73.2% baseline vs 18.6% monolithic injection (✓2-0). Note the asymmetry with our design: our gated tokens push z *into the observation stream*; PointACT has the *action head pull from z*. We never tested pull-side consumption → Experiment 4.
- Frozen-as-backbone at scale loses consistently: −42% avg (✓3-0); OpenVLA frozen 47.0 vs 69.7 finetuned, real robot (✓); frozen DINOv2 ≈ 0 in diffusion-policy IL (✓2-0). Finetuned-backbone recovers wins but only for ResNet-class encoders with augmentation (R3M 64.0 frozen → 80.5 finetuned+aug vs LfS 63.1, ✓; ViT/MVP *degrades* when finetuned — verifier scope note).

### Q3 — Claim C / world models: mostly negative, but TWO live 2026 data points and one unpublished decisive comparison we can run
- **CompACT (arXiv 2603.05438)** is architecturally *nearly Kepler* — learnable queries cross-attending frozen DINOv3 patches, **8 discrete tokens** via FSQ, only resampler+decoder trained (✓2-0). Its 8-token latent supports MPC/CEM at near-parity with a 784-token SD-VAE latent on RECON navigation (ATE 1.373 vs 1.262) at **~37× lower planning latency** (4.83s vs 178.78s) (✓2-0).
- **Critical gap (pool, quote-backed):** CompACT is only benchmarked against *reconstruction* tokenizers — never against dense DINO patch tokens as WM state. **The compressed-token-set vs dense-patch dynamics comparison in manipulation is unpublished.** DINO-WM's dense-beats-pooled (0.90 vs 0.44 PushT) remains the only adjacent datum, and it used spatial pooling, not a query token-set. We can be the ones to run the decisive version → Experiment 3.
- **RC-aux (arXiv 2605.07278)**: a compressed reconstruction-free WM is "predictive but not plannable" — accurate short-horizon latent prediction with an unusable planning geometry (✓, pool) — and a **reachability-conditioned aux objective** fixes much of it (Wall 50.4→83.6, 4/5 tasks improved, ✓2-0). Post-fix it splits with dense DINO-WM (loses Wall/Cube, **wins Push-T 90.8 vs 74.0** and Reacher) while planning **~130× cheaper** (35ms vs 4527ms per planner cost call; DINO-WM OOMs at 1024 candidates) (✓2-0).
- Counter-evidence to respect: adapter-compressing semantic latents (1024→96d) distorts control geometry — worse CEM action error, OOD, PCK (✓); and LaDi-WM's manipulation WM win operates over **dense** VFM features, not compressed sets (✓2-0, LIBERO-LONG 68.7% vs DreamerV3 33.5 / TD-MPC2 37.0).
- Net: claim C is **alive specifically as "8 compressed tokens are planning-sufficient and 1–2 orders cheaper"** — which is also the only surviving edge story (Q7).

### Q4 — Generalization/robustness: the strongest remaining axis, and structurally invisible to every eval we've run
- Pretrained-vs-scratch advantage appears **ONLY out-of-distribution** in the MBRL study: from-scratch CNN wins in-distribution on the manipulation task, then collapses −106% relative (below random) under table-texture shift while DINOv2/CLIP encoders hold (✓; scope: "only OOD" is per that task, not paper-wide) (✓).
- COLOSSEUM: SOTA manipulation policies lose **30–50%** to single-axis perturbations, **≥75%** combined (✓); worst axes = **distractor count, object color, lighting**; size matters least (✓).
- **ATK (real robot, diffusion policy — our policy class):** task-driven compressed keypoints vs dense RGB under combined shift: Sushi 0.45 vs **0.00**, Towel 0.70 vs **0.00**; depth/pointcloud also ~0.00 (✓2-0). And the mechanism matters: robustness comes from **task-driven selection of what to compress** — ATK 0.893 vs random-keypoints 0.23–0.34 vs full unfiltered set 0.01–0.12 (✓2-0). Kepler's aux objectives (action-readout, future-latent) make z task-relevance-selected — the right *kind* of compression by this evidence. Scope note: ATK also wins in-distribution on its tasks (pool), so it's not a pure parity-then-crossover story; its wins are at 30–80 demos.
- ID success does not predict OOD success per-encoder (MVP ViT-S: 39.9% ID → 6.6% OOD, ✓) — i.e., **our 0.86-vs-0.90 ID parity says nothing about the shift ordering.** This axis is genuinely untested for Kepler.
- Two warnings: manipulation-specific pretraining does NOT buy OOD robustness per se (best manip model ranked 7/15, below ImageNet-class, ✓); and the one heavily-verified robustness *lever* in the VLA literature is training-data diversity, not representation (pool: OpenVLA-OFT + 20k perturbed trajectories → +37pp camera robustness).
- **Bonus for the meta-lesson:** one offline metric DOES predict OOD rollouts — emergent segmentation (CLS-attention Jaccard), R²=0.85 / ρ=0.90, vs ImageNet probe R²=0.07 (✓). Worth computing on Kepler's cross-attention maps as a pretrain-time gate *for this axis only*.

### Q5 — Escalation task selection: thin coverage; recommendation is to escalate the AXIS, not the task
Honest accounting: the sweep produced no verified corpus on "which sim task saturates below ceiling and is rescued by fusion." What exists: PointACT attributes its residual failures to occlusion/partial views (pool, tangential — a caution that compression doesn't fix occlusion); VLA robustness benchmarks say policies are most brittle to **camera pose and object displacement** (63–75pp drops; positional bias: distractors tolerated, displaced targets fail — pool). Given wave-1's clean chassis and the Q4 evidence, the higher-value escalation is **shift axes on the tasks we already run** (Square/robocasa pnp), not a new task. Force-based escalation stays parked behind rung-0 (see do-not-bother).

### Q6 — StateAdd −32pp mechanism: no direct literature; our result is novel — the fixes to test are FSQ, EMA-targets, and pull-side injection
Nothing in the sweep directly addresses conditioning-induced input non-stationarity harming BC policies at rollout. Closest: 2409.20248 *recommends* task/context-conditioned encoders as the remedy for missing decision-making (pool) — in direct tension with our −32pp, which suggests conditioning must live in *pretraining objectives*, not in policy-time inputs. CompACT's FSQ is the obvious stabilizer (frozen discrete tokens can't drift under the policy) **but the paper makes no stability claim — FSQ is motivated purely by generation efficiency** (pool, quote-backed). So: test, don't assume (folded into Experiments 3/4). Our detector-clean, gates-alive, elevated-train-loss anatomy of the harm is publishable as a finding on its own.

### Q7 — Edge/hardware: policy-side token compression is NOT the bottleneck; the surviving edge story is planning latency
Unanimous across four independent sources (✓ all): ~75% of VLA edge step latency is autoregressive decode, not vision (✓); in a diffusion VLA the language module is ~89% of FLOPs (vision 405G of 4190G) (✓); token-count reduction only pays when a billion-parameter LLM consumes the tokens — not a small diffusion chassis (✓); real edge deployments win via whole-model quantization/graph optimization (ACT 2.86s→0.32s on NXP i.MX 95; int4 OpenVLA halves VRAM at parity, ✓/pool), and the accuracy-critical component is the action expert, not vision (pool). A 120-episode quantized ACT already runs a real task on embedded — no learned world-state latent involved (pool).
**The one compute claim compressed latents can still own:** planner-call cost in latent MPC — 37× (CompACT) to ~130× (RC-aux vs DINO-WM, which OOMs at 1024 candidates) (✓✓). **ARM pivot: sell "compressed world models make MPC tractable on edge," not "fewer policy tokens."** The banked 70×-token-reduction fact survives only as a memory-footprint/bandwidth footnote.

### Meta-lesson, externally corroborated
"Offline probes don't predict rollouts" is now published three independent ways: E2E encoders are decision-makers, not interchangeable feature extractors (✓3-0); best-SSIM VAE latent ↔ worst VLA success (pool); "predictive but not plannable" (✓ pool). The single exception found: CLS-Jaccard → OOD rollout success (✓). Our 3× internal confirmation was ahead of the literature; cite these when writing it up.

---

## 2. Ranked next experiments (all pre-registerable in the fast-screen loop)

**#1 — Shift-robustness battery on the EXISTING wave-1 checkpoints (Q4) — eval-only, run first.**
Design: perturbation wrapper for robomimic Square: (a) distractor objects on the table, (b) nut/peg color swap, (c) lighting (intensity/direction), (d) camera-pose jitter, (e) target-object displacement beyond demo poses. 2–3 severities per axis × 50 fast-eval rollouts. Arms: `hybrid-base` ep50 (0.90) vs `sb` ep50 (0.86) — both already on NAS; optionally `rc-pool` (0.82) as a second latent point.
Comparator: relative degradation at matched ID + absolute success under shift.
Kill gate (pre-register before the first rollout): z-tokens claim the axis iff `sb` beats base by **≥10pp absolute on ≥2 axes** (or ≤half base's relative degradation with overlapping ID). Wash on all axes → the robustness hypothesis for v0.3/v0.4 tokens is dead and the insertion program closes entirely.
Cost: ~0.5 GPU-night of evals + ~1 day eng for the wrapper. Evidence: u4/u5, p11/p12, p6/p7, u16/u17, 2312.12444 ID↛OOD.
Companion (free): compute CLS-Jaccard-style segmentation score on Kepler's cross-attention maps; log against the shift results as a candidate offline gate.

**#2 — Low-N crossover on Square (Q1) — the fair test we never ran.**
Design: subsample PH to **10 / 20 / 40 / 100** demos (40/100 = robomimic's own 20%/50% protocol points). Arms per count: (a) bare hybrid chassis, (b) chassis + sb gated z-tokens, (c) **z-as-backbone** (8 z-tokens + lowdim only — the zo arm, which the corpus says can only win here), 1 seed, ep50 fast-eval (add a seed at any count that gates).
Kill gate: a latent arm claims the axis iff it beats bare chassis by **≥10pp at ≥2 of 4 counts**. Monotone null → low-data axis dead for frozen Kepler, and with Q4 dead too the reactive-BC consumption story ends.
Cost: 12 runs, faster than full-data runs → **~1–2 GPU-nights** with the 7-arms/night loop. Evidence: u0/u9/u10/p1/p2/p3/u3/u8 + the 3-0 2409.20248 low-data-exclusion quote.

**#3 — Claim-C rung 0: 8-token dynamics vs dense-patch dynamics for MPC (Q3+Q7) — the unpublished comparison.**
Design: small transformer dynamics model p(z_{t+1} | z_t, a_t) on frozen Kepler token-sets (two variants: continuous z; FSQ-discretized z per CompACT) vs the same-budget dynamics on dense ViT patch tokens (DINO-WM recipe) — Lift first (cheap env), PushT if portable. CEM/MPPI goal-conditioned planning, 50 episodes, log per-planner-call latency + memory.
Kill gate: z-dynamics ≥**0.8× patch-dynamics success** AND ≥10× cheaper per planner call → claim C alive and the ARM edge story has its number. z ≪ patch (DINO-WM-style gap) on both variants → claim C dead for manipulation; publish the negative alongside CompACT's missing baseline.
Cost: ~2–3 GPU-nights (encoder is trained; dynamics heads are small). Evidence: u19/u20, p5/p13/p14, DINO-WM banked; CompACT-lacks-dense-baseline (pool).

**#4 — Pull-side consumption: action-head cross-attention to z (Q2/Q6) — the one insertion pattern with a published win.**
Design: modify the DP transformer so the action/cond tokens **cross-attend to the 8 z-tokens inside the action head** (PointACT bottleneck pattern) — z never enters the observation stream. Zero-init the cross-attn gate. Run at 200 demos + the best low-N point from #2. Control: identical arch, z replaced by learned constants.
Kill gate: ≥**+8pp over control at ep50** (outside the ±7pp fast-eval band) at either demo count, else the "policy consumes z" family is closed at every insertion site we know of.
Cost: ~1 GPU-night (4 runs). Evidence: u14/u15 (82.3 vs 18.6 is the largest consumption-pattern effect in the corpus); FSQ caveat from Q6 applies (test both continuous and FSQ z if #3's FSQ pretrain exists by then).

**#5 — RC-aux reachability objective on Kepler dynamics (Q3) — conditional on #3.**
Only if #3's z-dynamics lands within striking distance (≥0.5× patch): add multi-horizon open-loop prediction + budget-conditioned reachability with temporal hard negatives to the dynamics training.
Kill gate: +10pp planning success on the #3 task, else close.
Cost: ~1–2 GPU-nights. Evidence: p14 (Wall +33.2pp, 4/5 tasks), p5.

**#6 — (no GPU) ARM/story repositioning + program hygiene.**
Rewrite the ARM collab pitch around #3's planner-latency/memory numbers (37–130× published precedent; DINO-WM OOM datum) instead of policy-token FLOPs. Fold the Q7 evidence into the blog framing. Fix the PROGRESS.md "1,000 demos" error (done in this session's entry) and scope all past Square conclusions to "200-demo PH".

**Sequencing note:** #1 and #2 are independent and can share the fleet tonight (#1 is eval-only); #3 starts after; #4 after #2's low-N points exist; #5 gated on #3. Total to a full verdict on all three live axes: **~5–7 GPU-nights**, well inside the screening-loop budget.

---

## 3. Do-not-bother list (evidence against)

1. **Any further obs-side token insertion at ≥200 demos** (concat, StateAdd flavors, more gate variants): 3 internal generations null-to-harmful + external replication of the collapse (EO1+Point −54.6pp, ✓2-0) + zero published wins for the pattern (✓). Exception: #4's pull-side pattern, once, with a kill gate.
2. **z-only replacing vision at ≥200 demos**: 0/50 twice internally; frozen-backbone-at-scale −42% (✓3-0); frozen DINOv2 ≈ 0.00 in diffusion-policy IL (✓2-0). Only revisit inside #2 at ≤100 demos.
3. **Aux-loss distillation of z into the policy**: monotone dose-dependent harm internally; FLARE corpus already covered.
4. **Raw force wiring without curriculum**: rung-0 double null (Square 66/72→54/54; tool_hang 62/64) — any encoder-fused-force claim stays blocked until a FACTR-style curriculum passes rung-0. Don't re-run rung-0 as-is.
5. **Offline probes as GO gates**: 3× internal + three external corroborations (✓3-0 / pool / pool). Sole exception: CLS-Jaccard for the robustness axis (✓) — adopt as *informational*, never as a GO gate until #1 validates it on our stack.
6. **Edge pitch on policy-side token count**: four independent sources unanimous (✓×4) — decode/LLM dominates; quantization is the proven lever; token economy only matters with LLM-scale consumers.
7. **Expecting manipulation-specific pretraining to buy OOD robustness by itself**: best manip-pretrained model ranked 7/15, below ImageNet-class (✓). Robustness follows segmentation-like properties + task-driven selection, not the pretraining domain.
8. **Assuming FSQ fixes conditioning drift**: the FSQ paper makes no stability claim (pool, quote-backed) — it's a variant inside #3/#4, not a standalone bet.
9. **A new escalation task before the shift battery**: Q5 coverage was thin; the evidence-backed headroom is on shift axes of current tasks (30–50%/≥75% degradation, ✓), not on new-task selection.

---

## 4. Sources (22 fetched; ✓ = claims verified this cycle)

| # | Source | Role |
|---|---|---|
| 1 | arXiv 2409.20248 — E2E encoders vs frozen PVRs (−42%) | Q1/Q2 anchor (✓3-0 ×3 + limitation quote) |
| 2 | arXiv 2212.05749 — Simple LfS baseline | Q1 crossover (✓), finetune-recovers (✓), 3-0 original |
| 3 | arXiv 2410.18647 — Data scaling laws (UMI) | K=50 saturation (✓), frozen DINOv2=0.00 (✓) |
| 4 | arXiv 2203.12601 — R3M | the canonical frozen win, ≤100 demos only (✓✓✓) |
| 5 | arXiv 2406.09246 — OpenVLA | frozen −22.7pp real robot (✓), low-N finetune win (✓) |
| 6 | robomimic v0.1 docs | PH=200 demos (✓), 20%/50% subset protocol (✓) |
| 7 | arXiv 2605.21414 — PointACT | insertion collapse + bottleneck-query fix (✓✓) |
| 8 | openreview A1hpY5RNiH (2312.12444) — robust-manip benchmark | manip-pretrain ≯ ImageNet OOD (✓), CLS-Jaccard→OOD R²=0.85 (✓), consumption survey (✓) |
| 9 | arXiv 2509.12531 — PVMs in MBRL | OOD-only advantage (✓✓), no sample-efficiency gain in RL (✓) |
| 10 | arXiv 2402.08191 — THE COLOSSEUM | 30–50%/≥75% degradation (✓), worst axes (✓) |
| 11 | arXiv 2510.13626 — VLA robustness | camera/init brittleness, data-diversity lever, positional bias (pool) |
| 12 | arXiv 2506.13867 — ATK keypoints | task-driven compressed rep beats dense under shift, diffusion policy, real robot (✓✓) |
| 13 | arXiv 2603.05438 — CompACT "Planning in 8 Tokens" | Kepler-twin arch (✓), 8-token MPC parity @37× (✓), no-dense-baseline caveat (pool) |
| 14 | arXiv 2605.07278 — RC-aux / predictive-not-plannable | reachability aux +33pp (✓), split vs DINO-WM @~130× cheaper (✓) |
| 15 | arXiv 2505.11528 — LaDi-WM | dense-VFM WM wins manipulation, 10-demo regime (✓✓) |
| 16 | arXiv 2605.06388 — WM latent-space study | semantic>recon latents; compression distorts control geometry (✓); SSIM↛success (pool) |
| 17 | arXiv 2409.18330 — DMC-VB | full-data null (✓) + 1%-data crossover (✓) |
| 18 | arXiv 2603.02271 — VLA edge profiling | 75% decode latency (✓) |
| 19 | arXiv 2506.10100 — CogACT profiling | language module 89% FLOPs (✓) |
| 20 | arXiv 2510.24795 — real-time VLA survey | token count matters only w/ LLM backbone (✓); quantization lever (pool) |
| 21 | HF blog: NXP i.MX 95 robotics | edge = whole-model quant; action expert accuracy-critical (pool) |
| 22 | arXiv 2605.07278 planner-cost table | (same as #14) latency/memory numbers (✓) |

*Pool = extracted claim with verbatim quote from the fetched source, not independently re-verified this cycle (109-claim pool retained in the journals). Every claim marked ✓ was adversarially re-checked against the fetched source by 1–3 independent verifiers; all 39 checks passed, 0 refuted.*


---

> **APPENDIX (merged 2026-08-02 during doc consolidation, content unchanged). Formerly `REDESIGN_RESEARCH_20260729.md`.**

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


---

> **APPENDIX (merged 2026-08-02 during doc consolidation, content unchanged). Formerly `research/v04/01_state_as_query.md`.**

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


---

> **APPENDIX (merged 2026-08-02 during doc consolidation, content unchanged). Formerly `research/v04/02_shortcut_mitigation.md`.**

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


---

> **APPENDIX (merged 2026-08-02 during doc consolidation, content unchanged). Formerly `research/v04/03_objectives.md`.**

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


---

> **APPENDIX (merged 2026-08-02 during doc consolidation, content unchanged). Formerly `research/v04/04_fast_iteration.md`.**

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


---

> **APPENDIX (merged 2026-08-02 during doc consolidation, content unchanged). Formerly `research/v04/05_reference_impls.md`.**

# Reference implementations survey — v0.4 encoder redesign

**Date:** 2026-07-30. **Task:** for 10 methods flagged in REDESIGN_RESEARCH_20260729.md, find the official
(or best community) implementation, verify the mechanism-bearing code actually exists in the repo, and
assess portability onto our stack (fork of real-stanford/diffusion_policy + custom PerceiverFuse module,
8 learned queries d=256 over frozen ViT-B/16 patches, robomimic/RoboCasa).

**Method:** every repo below was fetched (repo page, GitHub API metadata/trees, and raw source files for
the mechanism-bearing modules). Claims from raw-file reads are marked; anything not directly fetched is
tagged **UNVERIFIED**. Fetch date 2026-07-30.

## Summary table

| # | Method (paper) | Repo | License | Code status | Mechanism in repo? | Port verdict |
|---|---|---|---|---|---|---|
| 1 | AdapTac (2505.13982) | github.com/kingchou007/adaptac-dex | MIT | Model/policy code present; dataset + some "init code" still TODO | YES — `FFG_policy.py`, `fuse_network.py` | COPY pattern (aux head + force-query attn); encoder is point-cloud, don't port it |
| 2 | FLARE (2505.15659, GR00T N1.5) | none (traces in github.com/NVIDIA/Isaac-GR00T) | Apache-2.0 (GR00T) | **NOT open-sourced** — disabled stubs only | NO — future_tokens embedding exists, loss/MLP absent | BUILD from paper; use FRAPPE code as the working reference |
| 3 | FRAPPE (2602.17259) | github.com/OpenHelix-Team/frappe | **NONE (no license file)** | Official, functional (RDT + RoboTwin) | YES — `flappe_model.py`, `mid_train_runner.py` | REIMPLEMENT (small mechanism; no-license blocks verbatim copy) |
| 4 | MCR (2410.22325) | github.com/luccachiang/robots-pretrain-robots | MIT | Official, complete + HF checkpoints | Losses YES (`mcr/trainer.py`); **metric code ABSENT** | COPY losses; BUILD the Grad-CAM/SAM-2 metric ourselves |
| 5 | SpawnNet (2307.03567) | github.com/johnrso/spawnnet | MIT | Official, complete (sim+real, ckpts) | YES — `simple_bc/encoder/spawnnet/spawnnet.py`, `vit_conv.py` | COPY adapter module; drop the IsaacGym harness |
| 6 | Patch Policy (2607.18236) | github.com/gaoyuezhou/patch_policy | none yet | **PLACEHOLDER** — "Code will be released soon" | NO | BUILD (mechanism trivial); watch repo |
| 7 | DiT-Block Policy (2410.10088) | github.com/sudeepdasari/dit-policy | MIT | Official, complete | YES — `models/diffusion.py` (adaLN-zero), `agent.py` (obs dropout) | COPY directly — cleanest lift of the ten |
| 8 | FACTR (2502.17432) | github.com/RaindragonD/factr | Apache-2.0 | Official (CMU), functional; hardware bits in companion repo | YES — `factr/agent.py` `tokenize_obs()` + `cfg/train_bc.yaml` | COPY — ~50-line curriculum, drop-in |
| 9 | Burns et al. (2312.12444) | github.com/stanford-iris-lab/segmenting_feats + github.com/kayburns/Intriguing-Properties-of-Vision-Transformers | MIT / **none** | Both exist; metric lives in the second | YES — `evaluate_segmentation.py` | RUN as-is on our ViT; adapt for Perceiver queries (no CLS) |
| 10 | Discrete-latent-target aux (2605.04678) | github.com/RUCKBReasoning/From_Pixels_to_Tokens | **NONE (no license file)** | Official (ICML'26), functional, minor TODOs | YES — `exp/train_vla.py` variants + `data_preprocess/*_lam/` | EXTRACT the VQ tokenizer + discrete head idea; harness (Qwen3-VL) irrelevant |

License red flags: FRAPPE, From_Pixels_to_Tokens, and the kayburns metric repo have **no license** —
default all-rights-reserved; reimplement rather than copy verbatim. Everything else MIT/Apache-2.0.

---

## Per-repo notes

### 1. AdapTac — `kingchou007/adaptac-dex` (IROS 2025)

- **URL:** https://github.com/kingchou007/adaptac-dex — MIT. PyTorch; depends on MinkowskiEngine +
  PyTorch3D (RISE-style sparse 3D point-cloud visual encoder), Python 3.8.
- **Mechanism files (verified by raw-file fetch):**
  - `src/adaptac/policy/FFG_policy.py` — class **`FFG`** (Force-Guided Fusion policy).
    `predict_future_force()` computes the aux: `predict_loss, predict_force = self.force_predictor.compute_loss()`,
    weighted into the total as `predict_loss * tactile_weight`. `apply_force_guided_attention()` runs
    `self.force_guided_attn(visual_feat, tactile_feat_reshaped, guide_force_reshaped)` — the *predicted*
    future force is fed back as the attention guide.
  - `src/adaptac/model/fuse_network.py` — **`ForceGuidedCrossAttention`** (predicted force expanded via
    `_expand_force_for_observations()` as the QUERY; `memory = torch.stack([visual, tactile], dim=1)` as
    K/V; implemented as a thin `nn.TransformerDecoderLayer` wrapper), plus `SelfAttentionFusion`,
    `CrossAttentionFusion`. NOTE: the per-modality scalar mixing described in the paper was NOT visible in
    this class on our read — verify before citing that detail (possible paper/code divergence).
  - Decoders: `src/adaptac/model/unet_diffusion.py` (`DiffusionUNetPolicy`, actions) and
    `src/adaptac/model/transformer_diffusion.py` (`DiffusionTransformer` — the future-force predictor is
    itself a small diffusion transformer). Tactile: `src/adaptac/model/tactile/` (MAEGAT GNN over taxels).
- **Completeness:** full file tree fetched; model/policy/dataset/training code present
  (`scripts/command_train.sh`, zarr conversion in `src/adaptac/dataset/gen_data/`). README TODO still
  lists "Release the init code" and "Release the dataset" — checkpoints lean on 3DtacDex.
- **Port assessment:** the two things we want (future-force prediction aux; force-as-query cross-attn
  producing a separate fused stream for the diffusion head) are cleanly isolated in `FFG_policy.py` +
  `fuse_network.py` and are encoder-agnostic — graft onto our diffusion_policy fork with our force window
  replacing their taxel pipeline. Do NOT port the visual side (Minkowski sparse conv over point clouds,
  heavy non-pip dependencies); our frozen-ViT+Perceiver stands in as the K/V source.

### 2. FLARE — no code release (NVIDIA GR00T family)

- **Paper:** https://arxiv.org/abs/2505.15659; project page research.nvidia.com/labs/gear/flare
  (UNVERIFIED — page itself not fetched; URL cited by arXiv/search). arXiv points to
  https://github.com/NVIDIA/Isaac-GR00T as "code".
- **Verified status:** FLARE is **not** in the open GR00T codebase. In tag `n1.5-release`,
  `gr00t/model/action_head/flow_matching_action_head.py` (raw fetch) contains class
  **`FlowmatchingActionHead`** with `self.future_tokens = nn.Embedding(config.num_target_vision_tokens, ...)`
  and the literal comment `return_all_hidden_states=False,  # NOTE (YL): not using flare now`. The
  alignment loss, future-embedding MLP, and target-encoder plumbing are absent. GitHub issues #211
  ("Is FLARE open-sourced at GR00T N1.5") and #215 (asks for release timeline; documents the missing
  pieces) have **no maintainer answer** (both fetched). Current `main` is N1.7 (`gr00t_n1d7/`), Apache-2.0,
  7.7k stars; no flare/flow_matching_action_head hits in the (truncated) main tree — the n1.5-release tag
  is the reference point.
- **Port assessment:** BUILD-not-copy, and that changes little: FLARE's mechanism (a few learnable future
  tokens + cosine alignment of their hidden states to future-frame embeddings) is exactly what FRAPPE
  ships working code for (below). Use the paper for hyperparameters, FRAPPE's repo as the code reference,
  and keep the GR00T `future_tokens` stub as evidence of how NVIDIA slots the tokens into a flow-matching
  action head.

### 3. FRAPPE — `OpenHelix-Team/frappe`

- **URL:** https://github.com/OpenHelix-Team/frappe — official ("Official implementation of FRAPPE"),
  55 stars, created 2026-02, updated 2026-07. **License: NONE** (GitHub API spdx null — treat as
  all-rights-reserved; reimplement, don't paste). Builds on RDT (Robotics Diffusion Transformer) +
  RoboTwin 2.0. 56 .py files; two-stage (mid-train, post-train) entry points
  `main_mid_train.py`/`main_post_train.py`, `finetune_mid.sh`/`finetune_post.sh`.
- **Mechanism files (verified by raw-file fetch):**
  - `models/rdt/flappe_model.py` (sic) — class **`RDT(nn.Module)`**:
    `self.learnable_tokens = nn.Parameter(torch.zeros(1, self.learnable_tokens_num, hidden_size))`
    (std=0.02 init), appended to the sequence via `torch.cat([x, expanded_learnable_tokens], dim=1)`;
    hidden states tapped at a configurable depth: `if encoder_depth is not None and (i + 1) == encoder_depth
    and num_learnable > 0`; projector MLPs (`self.projectors`) map tapped features to teacher spaces.
  - `models/mid_train_runner.py` — **`RDTRunner.compute_loss`** computes the alignment term `proj_loss`:
    L2-normalize then `1 - (z_j * z_tilde_j).sum(dim=-1)` (cosine distance), averaged over
    `len(zs) * bsz`; teacher selection by `enc_type` string matching 'clip' / 'dinov2' / 'theia' / 'vit'
    (encoders in `models/multimodal_encoder/{clip,dinov2,siglip,t5}_encoder.py`). Returned as
    `(loss, proj_loss)`; the λ combination lives in the outer training loop (`train/mid_train.py`,
    UNVERIFIED — not fetched). `models/rdt_runner.py` is the plain-RDT runner (MSE only — verified).
- **Port assessment:** this is the working implementation of our E4 redo (prefix tokens + frozen-VFM
  future-latent cosine at a mid layer). The whole mechanism is ~4 small pieces — token parameter, one-line
  sequence concat, depth tap, projector+cosine — trivially reimplementable inside a diffusion_policy
  transformer/DiT in a day; RDT specifics don't matter. The missing license is the only reason not to
  copy files wholesale.

### 4. MCR — `luccachiang/robots-pretrain-robots` (ICLR 2025)

- **URL:** https://github.com/luccachiang/robots-pretrain-robots — official, MIT, PyTorch,
  ResNet-50 backbone (`torchvision`). Checkpoints on HuggingFace `GqJiang/robots-pretrain-robots`.
- **Mechanism files (verified by raw-file fetch of `mcr/trainer.py`):** class **`Trainer.update()`**
  computes all three losses: time-contrastive (`smoothloss1/2`, InfoNCE over within-video frame triplets),
  action prediction (`bc_loss = model.module.bc_loss(pred_action, b_actions.detach())`), and the
  dynamics-alignment contrastive (`state_align_loss` via `s0loss`/`s2loss`, InfoNCE between image
  embeddings and a state-action encoder). Architecture in `mcr/models/models_mcr.py` (backbone, actor,
  projectors); config `cfgs/config_rep.yaml`; DROID preprocessing in `scripts/` (user-path TODOs).
- **The manipulation-centricity metric (Grad-CAM ∩ SAM-2 Jaccard) is NOT in the repo** — confirmed by
  README/file inspection. If we want the diagnostic (redesign doc §3/§5), we build it: binarized Grad-CAM
  vs SAM-2 masks of EE+objects, threshold unspecified in the paper.
- **Port assessment:** the losses are encoder-pretraining objectives, orthogonal to the policy — portable
  onto Perceiver pretraining wherever we have paired (frames, proprio, action) batches; the
  dynamics-alignment InfoNCE is the piece the ablation says to lift (83.2→66.2 without it). Written for a
  single pooled ResNet embedding, so we adapt the projector heads to our token set (e.g., align a pooled
  readout or per-query projections). Metric = build ourselves (SAM-2 + Grad-CAM, both off-the-shelf).

### 5. SpawnNet — `johnrso/spawnnet` (ICRA 2024)

- **URL:** https://github.com/johnrso/spawnnet — official (linked from project page
  xingyu-lin.github.io/spawnnet; authors Lin, So, Mahalingam, Liu, Abbeel). MIT. PyTorch + Hydra;
  sim experiments use IsaacGym/RLAfford (vendored), real experiments a simple BC harness.
- **Mechanism files (directory listing verified; file internals UNVERIFIED — not read line-by-line):**
  - `simple_bc/encoder/spawnnet/spawnnet.py` — the SpawnNet encoder (two-stream: frozen pretrained ViT
    layers tapped multi-layer, adapter convs fused into a learned stream).
  - `simple_bc/encoder/spawnnet/vit_conv.py` — ViT feature extraction / conv-adapter plumbing.
  - Baselines for the Table-3 mechanism comparison: `simple_bc/encoder/{r3m_encoder,impala,vit_descriptor}.py`.
  - Training/eval: `simple_bc/train.py`, `simple_bc/eval.py`, Hydra confs in `conf/`.
- **Completeness:** production-grade for reproduction — sim checkpoints, real dataset (Drive), debug docs.
- **Port assessment:** the adapter-insertion pattern (tap frozen ViT-B/16 at multiple depths, lightweight
  conv adapters, fuse into a trainable stream) drops straight into our stack as an alternative/addition to
  PerceiverFuse — our Perceiver can consume the adapter pyramid instead of last-layer patches, which is
  precisely the "multi-layer, not last-layer" rule the redesign doc extracted. `simple_bc/encoder/` is
  reasonably self-contained; leave IsaacGym/RLAfford behind.

### 6. Patch Policy — `gaoyuezhou/patch_policy` (arXiv 2607.18236, LeCun/Pinto NYU)

- **URL:** https://github.com/gaoyuezhou/patch_policy — linked from https://patch-policy.github.io/.
  **Placeholder only:** README says "Code will be released soon. Stay tuned!" No license, no description
  (API: null/null), created 2026-07-21, updated 2026-07-29 — repo is 9 days old and being touched, so a
  release may be imminent. 13 MB size is likely media, not code (UNVERIFIED).
- **Port assessment:** no code exists → BUILD. The mechanism is the simplest of the ten (frozen
  DINOv2/WebSSL patch tokens fed densely to a block-causal transformer policy); the paper + our existing
  diffusion_policy transformer suffice. Re-check the repo before implementation freeze — if it lands with
  a permissive license their block-causal masking and token-handling details are worth a diff.

### 7. DiT-Block Policy — `sudeepdasari/dit-policy` (arXiv 2410.10088)

- **URL:** https://github.com/sudeepdasari/dit-policy — official, MIT, PyTorch + Hydra
  (`data4robotics` codebase). Entry point `finetune.py`.
- **Mechanism files (verified by raw-file fetch):**
  - `data4robotics/models/diffusion.py` — the DiT noise net: **`_DiTDecoder`** blocks with
    `_ShiftScaleMod` + **`_ZeroScaleMod`** (adaLN-zero: `nn.init.zeros_(self.scale.weight)` and
    `nn.init.zeros_(self.scale.bias)`), **`_FinalLayer`** with
    `adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(..., 2*hidden_size))`.
  - `data4robotics/agent.py` — **`BaseAgent`**: proprio enters via `use_obs` ∈ {"add_token",
    "pad_img_tokens"}, in both paths through `nn.Dropout(p=0.2)` applied to the raw obs vector
    (`nn.Dropout(p=0.2), nn.Linear(odim, token_dim)` / `self._obs_proc = nn.Dropout(p=0.2)`). Since
    `nn.Dropout` zeroes elements independently, this IS the paper's per-dimension proprio dropout —
    hardcoded p=0.2, no dedicated flag. (Our reading of fetched source; the "per-dim" label is the
    paper's, the code is a plain elementwise Dropout on the state vector.)
  - Per-camera ResNet tokenizer: `data4robotics/models/resnet.py`; trainers in `data4robotics/trainers/bc.py`.
- **Port assessment:** cleanest lift of the survey. adaLN-zero block classes are self-contained
  `nn.Module`s and MIT — copy into our fork for the E5 "zero-init gated insertion" arm; the proprio
  dropout is one line in our obs encoder. Note our diffusion_policy fork's `TransformerForDiffusion`
  differs structurally from their `_DiTDecoder`; copying the two Mod classes + wiring is easier than
  adopting their whole model file.

### 8. FACTR — `RaindragonD/factr` (RSS 2025, CMU)

- **URL:** https://github.com/RaindragonD/factr — official (linked from jasonjzliu.com/factr),
  Apache-2.0. Skeleton is visibly forked from dit-policy/data4robotics (same `agent.py`,
  `models/action_transformer.py`, `trainers/` layout). Companion `factr_teleop` repo (ROS2) holds the
  force-feedback teleop; policy repo is hardware-light.
- **Mechanism files (verified by raw-file fetch):**
  - `factr/agent.py` — **`BaseAgent.tokenize_obs()`** applies the curriculum corruption: pixel space
    `gaussian_2d_smoothing(v, scale)` or `downsample_2d(...)`, latent space `gaussian_1d_smoothing` /
    `downsample_1d` on tokens; intensity from `get_scale(scheduler=..., start=curriculum.start_scale,
    end=curriculum.stop_scale, cur_step=misc.GLOBAL_STEP, max_step=curriculum.max_step)` — i.e., the
    decaying-corruption curriculum is a per-step scale on an image/token smoothing op.
  - `factr/cfg/train_bc.yaml` — `curriculum:` block: `space: pixel|latent`, `operator: blur|downsample`,
    `scheduler: linear` (options no/const/linear/step/exp/cos), `start_scale: 5`, `stop_scale: 0`,
    `max_step: ${max_iterations}`.
  - Force/proprio enters as obs tokens through the same `add_token` path (dropout p=0.2 inherited from
    data4robotics); no exotic force encoder in this repo. Trainer: `factr/train_bc_policy.py`,
    `factr/trainers/bc.py` (plain BC — verified no curriculum logic there; it all lives in the agent).
- **Port assessment:** the entire mechanism is ~50 lines (2 corruption ops + a scale scheduler + a global
  step) sitting in the obs-tokenization path — drop-in for our encoder wrapper on the E6 force arm, fully
  orthogonal to Perceiver/diffusion_policy internals, Apache-licensed. Note a FACTR 2 paper exists
  (arXiv 2606.12406, external force sensing for commodity arms) — not surveyed here.

### 9. Burns et al. attention-Jaccard — two repos

- **Policy/eval harness:** https://github.com/stanford-iris-lab/segmenting_feats — MIT, Stanford IRIS.
  Built on the facebookresearch R3M eval codebase; policy training + kitchen-shift transfer tests under
  `evaluation/r3meval/core/`, `run.sh`/`eval.sh`; project page points at the `eval` branch.
- **The metric itself:** https://github.com/kayburns/Intriguing-Properties-of-Vision-Transformers —
  kayburns' copy of the IP-ViT codebase (GitHub API says `fork: false`, so it's a re-upload, not a fork
  object; last updated 2023-09). **`evaluate_segmentation.py`** (+ `evaluate_segmentation.sh`) computes
  the Jaccard index of binarized ViT attention maps vs PASCAL VOC masks — exactly the R²=0.85 predictor.
  The project page (kayburns.github.io/segmentingfeatures) documents how to add new models to this
  script. **No license** on this repo (and upstream IP-ViT license UNVERIFIED).
- **Port assessment:** run, don't port — it's a standalone diagnostic needing only PASCAL VOC + a model
  hook; wire our frozen ViT-B/16 (and any finetuned variants) into `evaluate_segmentation.py` per the
  project-page instructions. One real adaptation for us: the metric is defined on CLS→patch attention
  (last block, all heads); PerceiverFuse has no CLS, so scoring the *encoder output* means substituting
  query→patch cross-attention maps — that variant is our extrapolation, not Burns'.

### 10. Discrete-latent-target aux — `RUCKBReasoning/From_Pixels_to_Tokens` (arXiv 2605.04678, ICML 2026)

- **URL:** https://github.com/RUCKBReasoning/From_Pixels_to_Tokens — official, 37 stars, created
  2026-03-31. **License: NONE** (API spdx null) — reimplement, don't copy. Stack: Qwen3-VL-2B VLA
  backbone, PyTorch 2.8, RLDS data, LIBERO eval (`experiments/robot/libero/`).
- **Mechanism files (directory/README verified; file internals UNVERIFIED):**
  - `exp/train_vla.py` with `--vla_id ∈ {baseline, la_align, la_direct, la_cond, la_tok}` — all four
    supervision variants from the paper (LA-Direct = discrete latent tokens decoded directly; LA-Align =
    internal-representation alignment; LA-Cond; LA-Tok), implementations under `latentvla/models/vla/`.
  - `data_preprocess/image_based_lam/` — trains the image-based latent-action model (the discrete-token
    target source); `data_preprocess/action_based_lam/` — VQ-style action tokenizer.
- **Completeness:** functional per README (LIBERO support recent); placeholder paths in preprocessing,
  robot constants need manual config.
- **Port assessment:** what transfers to E4-redux is the TARGET FORMAT, not the harness: train a small
  VQ tokenizer over future latents (their `*_lam` recipes) and supervise prefix tokens with cross-entropy
  on discrete codes instead of continuous cosine. The VLA side (Qwen3-VL, autoregressive decoding) shares
  nothing with diffusion_policy — extract the tokenizer training + discrete-supervision head as ideas,
  reimplement (~small VQ-VAE + CE loss) inside our FRAPPE-style aux.

---

## Cross-cutting build-vs-copy summary for our stack

1. **Copy with license clean (MIT/Apache):** DiT-Block adaLN-zero mods + obs dropout (dit-policy),
   FACTR curriculum (~50 lines), SpawnNet encoder module, MCR loss functions, AdapTac aux-head +
   force-query-attention pattern, Burns eval scripts (metric repo license murky — the script is a thin
   layer over VOC eval; worst case rewrite it).
2. **Reimplement from working reference (no license):** FRAPPE prefix-token alignment (the E4 redo core —
   mechanism is 4 small pieces), From_Pixels_to_Tokens discrete-target recipe.
3. **Build from paper (no code exists):** FLARE (use FRAPPE as code proxy; GR00T stub confirms token
   slotting), Patch Policy (trivial mechanism; watch the 9-day-old placeholder repo), MCR
   manipulation-centricity metric (SAM-2 + Grad-CAM, both off-the-shelf).
4. **Biggest practical wins first:** FRAPPE mechanism + dit-policy adaLN-zero cover the two
   highest-ranked redesign changes (E4 prefix-aux, E5 insertion) with the least code; FACTR curriculum
   and the AdapTac aux head cover the E6 force arm; Burns script is a same-day diagnostic on existing
   checkpoints.


---

> **APPENDIX (merged 2026-08-02 during doc consolidation, content unchanged). Formerly `research/v04/06_encoder_swap_protocols_lit.md`.**

# Encoder-swap evaluation protocols in the literature — deep-research report

*Produced 2026-07-30 by the deep-research workflow (runs `wf_e5e6774a-8e4` + completion `wf_e98a0c85-c6d`).*
*Method: 5 search angles -> 19 sources fetched -> 95 claims extracted -> top 25 (central/primary-first) put through 3-vote adversarial verification (2/3 refutations kill) -> 24 confirmed, 1 refuted, 0 unverified -> synthesized into 12 findings.*

## Research question

How do published robot-learning papers design and run "encoder-swap" experiments — evaluating pretrained visual representations by plugging them into downstream policy learning — and what does best practice look like? Cover: (1) the canonical benchmark/protocol papers (PVR/Parisi et al. 2022, R3M, MVP, VIP, VC-1/CortexBench, Voltron eval suite, Burns et al. 2312.12444, MCR, Theia, "Not All Policy Learning Methods are Created Equal" Hu et al. ICML 2023, Dasari "An Unbiased Look at Datasets for Visuo-Motor Pre-Training", Hansen et al. learning-from-scratch baseline, SpawnNet, plus any 2025-2026 successors); (2) the exact protocol dimensions each uses and where they disagree: frozen vs finetuned vs adapter encoders; policy head class (MLP-BC vs diffusion vs ACT) and evidence that conclusions FLIP with head choice; feature interface (CLS vs pooled vs patch tokens; token-count matching); demos per task; seeds; rollouts per eval; checkpoint selection (best-over-checkpoints vs last vs fixed-epoch); per-encoder hyperparameter tuning vs shared config; ID vs OOD splits; compute/data-matched scratch baselines; (3) documented pitfalls/critiques of this methodology (eval variance, simulator vs real transfer of rankings, head-dependence, cherry-picked task suites). Deliverable must END with a comparison table against OUR protocol so we can audit ourselves: frozen encoder + precomputed z-cache swapped into a fixed Diffusion Policy transformer (robomimic square/tool_hang + RoboCasa pick-and-place), 50 rollouts/eval, 1-2 seeds, ±6-7pp noise band treated as tie, ep50/ep100 fixed-epoch gates (not best-over-checkpoint), shared hyperparameters across arms, token-matched comparisons (8 pooled/unpooled query tokens vs 196 dense patch tokens) — state where our protocol matches published best practice, where it deviates, and what specific changes the literature says we should adopt.

## Executive summary

Published encoder-swap evaluations converge on a canonical core protocol — freeze the pretrained encoder, train a lightweight behavior-cloning head on a small demo set, and report success over multiple seeds — established by PVR (Parisi et al. 2022) as the control analogue of linear probing and adopted by R3M, CortexBench/VC-1, and Burns et al. 2312.12444. But the literature also documents four ways this protocol produces unreliable conclusions: encoder rankings flip with the downstream policy-learning method (Hu et al. ICML 2023: R3M best under BC, second-worst under VRF), with the feature interface tapped (PVR: early vs late layers invert across task types), between in-distribution and OOD visual conditions (Burns et al.), and between simulation and real hardware (Dasari: sim-real R²=32%; Hansen: augmented scratch beats best frozen PVR on a real pick task). Protocols disagree most sharply on checkpoint selection (R3M and CortexBench report best-over-checkpoints; Dasari evaluates only the final checkpoint with fixed eval hyperparameters and calls that "maximally fair") and on frozen vs finetuned encoders (Hansen and VC-1 show frozen-only evaluation confounds representation quality with pretraining-domain gap, closable by finetuning with augmentation). Audited against this, our protocol matches best practice on frozen-encoder design, shared hyperparameters, fixed-inits evaluation, token/dimensionality matching, and (per Dasari) fixed-epoch gates, but deviates on seeds (1-2 vs the literature's 3-5), single policy head (diffusion transformer only, with documented head-dependence risk), ID-only sim-only 3-task suite, and the apparent absence of an augmented learning-from-scratch baseline — the full row-by-row audit table is the final finding below.

## Findings

### 1. High confidence (vote: 3-0 on all four constituent claims [0,4,7,18])

The canonical encoder-swap protocol is a FROZEN pretrained encoder feeding a lightweight behavior-cloning head. PVR (Parisi et al., ICML 2022) established it explicitly as the control analogue of the CV linear-probe protocol, deliberately excluding full fine-tuning; CortexBench/VC-1 freezes encoders by default 'to disentangle the effect of learned representations from downstream task learning'; R3M trains only a 2-layer MLP head ([256,256], BatchNorm input, lr 0.001, batch 32, 20k steps) on frozen embedding + proprioception; Burns et al. keep all 15 encoders frozen with an MSE-trained MLP head. Our frozen encoder + precomputed z-cache design is squarely in this tradition (a z-cache is only valid because the encoder is frozen).

**Evidence:** PVR: 'our experiments freeze the vision models... similar in spirit to the linear classification (probe) protocol... We leave evaluation of representations in the full fine-tuning regime to future work.' VC-1: 'we consider frozen visual representations to disentangle the effect of learned representations from downstream task learning.' R3M: 'The downstream policy is a 2 layer MLP with hidden sizes [256,256]... learning rate of 0.001, batch size of 32 for 20000 steps.' Burns: 'The embedding weights are frozen during policy learning, so the pre-trained models receive no task data.'

**Sources:** https://arxiv.org/abs/2203.03580 · https://arxiv.org/abs/2303.18240 · https://arxiv.org/abs/2203.12601 · https://arxiv.org/abs/2312.12444

### 2. High confidence (vote: 3-0 [1,7,9,18]; 2-0 [13])

Concrete protocol dimensions of the canonical suites: PVR — 4 domains (Habitat, DMC x5, Adroit x2, Franka Kitchen x5), 25-100 RL-expert demos per MuJoCo task, MLP head with 3-frame history (LSTM for Habitat), 5 seeds with 95% CIs. R3M — 12 tasks across 3 simulators, 3 seeds per encoder-task pair, results averaged over seeds, camera viewpoints, and demo sizes (5/10/25 MetaWorld+Kitchen; 25/50/100 Adroit), best-over-checkpoints. Burns et al. — 15 encoders, 10 demos/task, 10 tasks in 2 simulators, 3 seeds x 2 cameras = 60 policies per model, ~9,000 zero-shot evaluations including OOD shifts. CortexBench — 17 tasks from 7 benchmarks (Adroit, MetaWorld, DMControl, TriFinger, ObjectNav, ImageNav, MobilePick) spanning IL and RL heads.

**Evidence:** PVR: 'We collect between 25-100 trajectories per task... an MLP with fixed history window in MuJoCo... mean values over five seeds... 95% confidence intervals.' R3M: 'For each visual representation and each task, we run 3 seeds of behavior cloning... average over multiple seeds, viewpoints, and demo dataset sizes.' Burns: 'we learn 60 policies for each model... we run 9,000 different simulated evaluations.' CortexBench: '17 EAI tasks drawn from 7 existing benchmarks.'

**Sources:** https://arxiv.org/abs/2203.03580 · https://arxiv.org/abs/2203.12601 · https://arxiv.org/abs/2312.12444 · https://github.com/facebookresearch/eai-vc · https://arxiv.org/abs/2303.18240

### 3. High confidence (vote: 3-0 [6,8]; 2-1 [23])

Checkpoint selection is a genuine methodological split in the literature. R3M evaluates online every 1,000 steps across 20,000 and reports the BEST success rate achieved; CortexBench reports 'the average rollout performance on the test set for the best intermediate policy during training.' In direct contrast, Dasari et al. (CoRL 2023) evaluate ONLY each policy's final checkpoint, with all evaluation hyperparameters (demo set, rollout count, initial positions, test objects, BC hyperparameters) held fixed within a task — explicitly framed as 'maximally fair evaluation.' Our ep50/ep100 fixed-epoch gates align with the Dasari camp, not the R3M/VC-1 camp; the practical consequence is that our absolute success numbers are not comparable to best-over-checkpoint papers, which are systematically optimistic.

**Evidence:** R3M: 'we train the agent for 20,000 steps, evaluate it online in the environment every 1000 steps, and report the best success rate achieved.' CortexBench: 'we report the average rollout performance on the test set for the best intermediate policy during training.' Dasari: 'Each policy's final checkpoint is evaluated on N test rollouts... All evaluation hyperparameters... are kept fixed within a task. This allows for maximally fair evaluation'; verifier confirmed 'we only evaluate the policy at the end of training (unlike some prior work that evaluated multiple times over the course of training).'

**Sources:** https://arxiv.org/abs/2203.12601 · https://arxiv.org/abs/2303.18240 · https://arxiv.org/abs/2310.09289

### 4. High confidence (vote: 3-0 [3,5,15,20])

No pretrained encoder is universally dominant; rankings are strongly task- and domain-dependent, so conclusions from narrow task suites do not generalize. PVR: no encoder superior across all 4 domains (MoCo > supervised RN50/CLIP on average; crop augmentation matters more than color). CortexBench (17 tasks): R3M best on Adroit/MetaWorld/DMControl, MVP ViT-L best on TriFinger/ImageNav/MobilePick, CLIP best on ObjectNav; even VC-1, best on average, does not dominate, and only reaches competitive-or-better on ALL benchmarks after adaptation (task-specific losses or in-domain data). Hansen et al.: 'no single frozen pre-trained representation is consistently better across all tasks,' even within the visually consistent PixMC benchmark.

**Evidence:** PVR: 'no PVR is clearly superior to any other across all four domains.' VC-1: 'no single PVR is universally dominant. Instead, we find that PVRs tend to work best in the domains... they were originally designed for'; 'when adapting VC-1... VC-1 is competitive with or outperforms state of the art on all benchmark tasks.' Hansen: 'no single frozen pre-trained representation is consistently better across all tasks.'

**Sources:** https://arxiv.org/abs/2203.03580 · https://arxiv.org/abs/2303.18240 · https://github.com/facebookresearch/eai-vc · https://arxiv.org/abs/2212.05749

### 5. High confidence (vote: 3-0 [10,11,14]; 2-0 [12])

Encoder rankings FLIP with the downstream policy head / learning method, so a single-head protocol cannot certify an encoder as 'best.' Hu et al. (ICML 2023, 21 tasks): 'The effectiveness of a pre-trained vision model is highly dependent on the downstream policy learning method' — R3M, the best encoder under BC, drops to second-worst (25% IQM) under visual-reward-function imitation. The same paper shows RL-based downstream evaluation is too high-variance to be a reliable protocol (seed-to-seed differences up to ~25pp for R3M, overlapping bootstrap CIs) and recommends BC and VRF as robust evaluation heads. CortexBench implicitly hedges against head-dependence by aggregating rankings across heterogeneous learners (IL on 5 benchmarks, RL/DD-PPO on ImageNav and MobilePick). Note: no surviving claim tests diffusion or ACT heads specifically — the head-dependence evidence is BC vs RL vs VRF.

**Evidence:** Hu: 'R3M, the star model on BC, only obtains 25% IQM success rate when considering all environments tasks, ranking as the second-worst model'; 'Due to high variability, RL is not a robust evaluation method. We show that the consistent results of VRF and BC make them reliable evaluation methods.' CortexBench table verified: Adroit/MetaWorld/DMControl/TriFinger/ObjectNav = IL; ImageNav/MobilePick = RL (500M steps, DD-PPO/VER).

**Sources:** https://arxiv.org/html/2304.04591 · https://github.com/facebookresearch/eai-vc · https://arxiv.org/abs/2303.18240

### 6. Medium confidence (vote: 2-0 [2] (single primary source))

The feature interface (which layer/tokens are tapped) also flips rankings, and PVR set the precedent for dimensionality-matched comparisons. Early conv-layer features win on fine-grained MuJoCo control while final-layer features win on semantic Habitat navigation; only a full-hierarchy PVR combining layers 3+4+5 — compressed to the last-layer dimensionality explicitly 'to perform fair comparisons' — solves all four domains, sometimes beating ground-truth state. This is direct precedent for our token-matched design (8 pooled/unpooled query tokens vs 196 dense patch tokens) and a warning that any single-interface conclusion (pooled-only or patch-only) is interface-contingent.

**Evidence:** PVR: 'Early convolution layer features are better for fine-grained control tasks (MuJoCo) while later convolution layer features are better for semantic tasks (Habitat ImageNav). ... To ease computations and perform fair comparisons, we compress these representations to the size of the representation at the last layer.' Note: the parallel claim that CortexBench feeds only a CLS/pooled token was REFUTED (0-3), so CortexBench's exact token interface is not established by this evidence.

**Sources:** https://arxiv.org/abs/2203.03580

### 7. Medium confidence (vote: 2-1 [21]; 3-0 [15] (verifier evidence rated high on both))

Frozen-only evaluation confounds representation quality with pretraining-domain gap. Hansen et al. (ICML 2023): frozen PVRs are 'hindered by a significant domain gap between the pre-training datasets and current benchmarks for visuo-motor control, which is alleviated by finetuning' — but finetuning ResNet-based encoders beats frozen ONLY when data augmentation (random shift) is applied during finetuning (and ViT/MVP resists finetuning even with augmentation). VC-1 corroborates: adaptation via task-specific losses or in-domain data yields large gains (e.g., ImageNav +11.3, MobilePick +10.8) that frozen evaluation misses. Implication for us: our frozen-arm conclusions are valid within the frozen regime but should not be read as encoder quality rankings absolute.

**Evidence:** Hansen: 'finetuning ResNet-based representations on task data improves over frozen representations... but only when using data augmentation during finetuning' (Table 5, Adroit). VC-1 README: 'when adapting VC-1 (through task-specific losses or a small amount of in-domain data), VC-1 is competitive with or outperforms state of the art on all benchmark tasks.' PVR explicitly deferred the finetuning regime to future work, so the canonical protocol never resolved this.

**Sources:** https://arxiv.org/abs/2212.05749 · https://arxiv.org/abs/2303.18240 · https://github.com/facebookresearch/eai-vc

### 8. High confidence (vote: 3-0 [19,20]; 3-0 [3])

A compute/data-matched learning-from-scratch baseline with augmentation is mandatory: a shallow ConvNet + random-shift augmentation is competitive with frozen PVR/MVP/R3M across BC, PPO, and DrQ-v2 on 17 tasks in 4 domains (Adroit, DMControl, PixMC, real xArm), and on the real robot the augmented scratch baseline BEAT the best frozen pretrained encoder (Pick: LfS 55.0±5.0 vs max{PVR,MVP,R3M} 35.0±15.0). Hansen et al. explicitly frame this baseline as necessary for 'accurately benchmarking progress.' Note this partially tempers (but within its small-data scope does not contradict) PVR's earlier finding that any frozen PVR beats an UN-augmented from-scratch encoder in the small-demo regime — the augmentation is the load-bearing difference.

**Evidence:** Hansen: 'a simple Learning-from-Scratch (LfS) baseline that incorporates data augmentation and a shallow ConvNet... is surprisingly competitive with recent approaches (PVR, MVP, R3M)... across a variety of algorithms, task domains, and metrics in simulation and on a real robot.' Verified Table 2: Pick LfS(+aug) 55.0±5.0 vs best pretraining 35.0±15.0 (20 trials x 2 seeds), with pretrained methods run using their own public implementations unchanged.

**Sources:** https://arxiv.org/abs/2212.05749 · https://arxiv.org/abs/2203.03580

### 9. High confidence (vote: 3-0 [16,17])

In-distribution-only rankings are misleading: manipulation-specific PVRs (R3M, VIP) comfortably beat standard ImageNet pretraining in-distribution but lose to supervised/self-supervised ImageNet models (e.g., DINO, MoCo-v3) under subtle lighting/texture/distractor shifts (Burns et al., CoRL; on real ALOHA, MoCo-v3 40% grasp vs MVP 0%). Furthermore, among ViTs, emergent segmentation ability (Jaccard index of attention maps) predicts OOD manipulation performance better than ImageNet accuracy, in-domain accuracy, or shape bias — so standard CV metrics should not be used as proxies for policy robustness, and encoder-swap benchmarks should include OOD visual-shift evaluations.

**Evidence:** 'visual representations designed for manipulation and control tasks do not necessarily generalize under subtle changes in lighting and scene texture or the introduction of distractor objects'; 'emergent segmentation ability is a strong predictor of out-of-distribution generalization among ViT models... more predictive than... downstream ImageNet accuracy, in-domain accuracy, or shape-bias.' Scope caveat: obtained with frozen encoders + MLP-BC heads, so per Hu et al. these OOD findings may themselves be head-dependent.

**Sources:** https://arxiv.org/abs/2312.12444

### 10. Medium confidence (vote: 2-1 [22]; 3-0 [20] (two independent primary sources, one split vote; R²=0.32 is 'weakly predictive,' not zero))

Simulation benchmark rankings of pretrained encoders do not transfer reliably to real robots. Dasari et al. (CoRL 2023): across 15 pretrained MAE representations, average sim and average real performance are almost entirely uncorrelated (R²=32%; 34% even cherry-picking the most similar sim/real task pair, RoboMimic block-lift vs real stacking) — the paper's own abstract concludes 'common simulation benchmarks are not a reliable proxy for real world performance.' Hansen et al.'s real-robot flip (scratch beating best frozen PVR on Pick) independently corroborates. Direct relevance: our protocol is sim-only (robomimic + RoboCasa), so encoder conclusions should be scoped as sim-internal until hardware-validated.

**Evidence:** Dasari: 'We find that sim and real performance are almost entirely uncorrelated (a very low R2 = 32%). Even if we were to cherry-pick the two most similar sim/real tasks (RoboMimic's block-lift vs our stacking task) the correlation is still very low: R2 = 34%.' Qualifier from verification: specific to tabletop-manipulation BC with MAE encoders varied by pretraining dataset; should not be universalized to all domains.

**Sources:** https://arxiv.org/abs/2310.09289 · https://arxiv.org/abs/2212.05749

### 11. Medium confidence (vote: 2-1 [23] (verifier evidence rated high; demo-count scoping applied per verifier))

Dasari et al.'s alternative protocol — the main published counterpoint to frozen evaluation — fine-tunes the encoder end-to-end with a 2-layer GMM policy head (Adam, 50K iters), evaluates the FINAL checkpoint only, and holds every evaluation hyperparameter fixed within a task ('maximally fair'). Demo counts were regime-dependent: 50 tele-op demos/task on the real robot, 25 in MetaWorld/Franka Kitchen, but 200 in the RoboSuite sim arm — directly relevant to us since our robomimic tasks are RoboSuite, where 200 demos was Dasari's setting, aligning with robomimic-standard demo counts.

**Evidence:** 'The entire policy network (both p and E) is fine-tuned end-to-end (using Adam for 50K iterations)... Each policy's final checkpoint is evaluated on N test rollouts for every task, with novel initializations... All evaluation hyperparameters... are kept fixed within a task. This allows for maximally fair evaluation.' Verifier correction: '<=50 demos' holds only for real-world (50) and MetaWorld/Kitchen (25); RoboSuite used n=200.

**Sources:** https://arxiv.org/abs/2310.09289

### 12. High confidence (vote: synthesis of findings above)

AUDIT: comparison of OUR protocol (frozen encoder + z-cache into fixed Diffusion Policy transformer; robomimic square/tool_hang + RoboCasa pick-and-place; 50 rollouts/eval; 1-2 seeds; ±6-7pp tie band; ep50/ep100 fixed-epoch gates; shared hyperparameters; token-matched 8 pooled vs 196 patch tokens) against published best practice. Top literature-mandated changes, in priority order: (1) run ≥3 seeds on any decision-relevant comparison and derive the tie band from measured seed CIs rather than a fixed ±6-7pp; (2) add a compute/data-matched shallow-ConvNet + random-shift-augmentation scratch baseline; (3) spot-check the headline encoder ranking with a second policy head (e.g., MLP-BC per R3M recipe) since rankings demonstrably flip with head; (4) add an OOD visual-shift eval (lighting/texture/distractors) before claiming encoder superiority; (5) scope all conclusions as 'frozen-regime, sim-only, tabletop-manipulation, diffusion-BC' — the literature shows each of those qualifiers can flip the ranking.

**Evidence:** | Dimension | Our protocol | Published practice | Verdict | Recommended change |
|---|---|---|---|---|
| Encoder handling | Frozen + precomputed z-cache | Frozen is the canonical default (PVR, R3M, VC-1, Burns); but Hansen/VC-1 show frozen-only confounds quality with domain gap, closable by finetune+aug | MATCHES canon | Keep frozen as primary; add one finetune-with-augmentation arm for the top-2 encoders, or explicitly scope claims to the frozen regime |
| Policy head | Single fixed Diffusion Policy transformer | Canon uses simple MLP-BC (PVR/R3M/Burns); Hu et al. prove rankings flip across heads (R3M: best under BC, 2nd-worst under VRF); CortexBench aggregates over IL+RL heads | DEVIATES (single, heavier-than-canon head) | Spot-check headline comparisons with a cheap MLP-BC head; no published evidence covers diffusion heads specifically, so head-dependence risk is unquantified for our exact setup |
| Feature interface | Token-matched: 8 pooled/unpooled query tokens vs 196 dense patch tokens | PVR compressed multi-layer features to matched dimensionality 'to perform fair comparisons'; interface choice flips rankings (early vs late layers) | MATCHES best practice | Keep; report both interfaces per encoder since single-interface conclusions are interface-contingent |
| Demos per task | robomimic/RoboCasa standard demo sets (~200 PH for RoboSuite tasks) | 10 (Burns), 5-100 (R3M), 25-100 (PVR), 200 for RoboSuite (Dasari) | MATCHES Dasari's RoboSuite setting | Optional: add a low-demo (10-25) arm — encoder differences are largest in the small-data regime the canon was designed for |
| Seeds | 1-2 | 3 (R3M, Burns), 5 + 95% CIs (PVR); Hu documents up to ~25pp seed-to-seed swings (RL; BC more stable) | DEVIATES — weakest point | ≥3 seeds for any comparison that gates a decision; report CIs |
| Rollouts/eval | 50, fixed inits | No fixed standard in verified claims; Dasari fixes N rollouts + initial positions within task; Burns ~9,000 total evals | ROUGHLY MATCHES (fixed inits = Dasari discipline) | Keep 50; keep inits identical across arms |
| Tie/noise band | Fixed ±6-7pp | PVR/Hansen report seed-based 95% CIs; Hansen real-robot error bars ±5-15pp at 2 seeds | PARTIAL match in spirit | Derive band from measured ≥3-seed CIs per task instead of a global constant; with 1-2 seeds the band likely understates variance |
| Checkpoint selection | ep50/ep100 fixed-epoch gates | SPLIT: R3M + CortexBench = best-over-checkpoints; Dasari = final checkpoint, 'maximally fair' | MATCHES Dasari camp; DEVIATES from R3M/VC-1 | Defensible as-is (and consistent with our early-kill discipline); optionally log best-over-checkpoint as a secondary number to check gate sensitivity; never compare our absolute numbers to best-over-checkpoint papers |
| Hyperparameters | Shared across arms | Dasari: BC hparams fixed within task ('maximally fair'); Hansen: competitors' own public hparams unchanged; no canon paper does per-encoder tuning | MATCHES | Keep; note residual bias if the shared config was originally tuned around one arm's encoder family |
| ID vs OOD | ID-only (standard robomimic/RoboCasa inits) | Burns: ID-only rankings misleading; manipulation PVRs lose to DINO/MoCo under lighting/texture/distractor shifts | DEVIATES | Add at least one visual-shift eval (distractors are cheapest in RoboCasa) before any 'encoder X > Y' claim |
| Scratch baseline | None apparent | Hansen: augmented shallow-ConvNet LfS baseline competitive with PVR/MVP/R3M and required for honest benchmarking | DEVIATES (if absent) | Add a data/compute-matched shallow ConvNet + random-shift-aug arm |
| Task suite | 3 sim manipulation tasks | 10-21 tasks across multiple domains (R3M 12, CortexBench 17, Hu 21); rankings demonstrably flip across domains and sim-to-real (R²=32%) | DEVIATES (narrow, sim-only) | Scope conclusions to 'sim tabletop manipulation under diffusion BC'; hardware or added domains needed before any general encoder claim |

**Sources:** https://arxiv.org/abs/2203.03580 · https://arxiv.org/abs/2203.12601 · https://arxiv.org/abs/2303.18240 · https://arxiv.org/html/2304.04591 · https://arxiv.org/abs/2312.12444 · https://arxiv.org/abs/2212.05749 · https://arxiv.org/abs/2310.09289

## Refuted claims

- (vote 0-3) "CortexBench's feature interface feeds only the ViT [CLS] token (or global-average-pooled final conv features for ResNets) into the policy — i.e., a single pooled token, not dense patch tokens — which differs from our 8-token pooled vs 196 dense-patch-token comparison." — https://arxiv.org/abs/2303.18240

## Caveats

Vote and source quality: three constituent claims survived only 2-1 (Hansen domain-gap/finetune-aug [21], Dasari sim-real R² [22], Dasari protocol details [23]) — verifier evidence was rated high on all three, but findings built on them are marked medium. The feature-interface finding rests on a single primary source (PVR, 2-0). The claim that CortexBench feeds only a CLS/pooled token was REFUTED 0-3, so CortexBench's actual token interface — the dimension most relevant to our 8-vs-196-token design — remains uncharacterized by this evidence. Coverage gaps: the research question named Voltron, MCR, Theia, SpawnNet, and 2025-2026 successors, but no claims about them survived verification, so the synthesis tilts toward 2022-2023 protocols; best practice may have shifted (e.g., diffusion/ACT heads are now common in evaluation suites but appear in no surviving claim). Head-dependence evidence (Hu et al.) covers BC vs RL vs VRF, not diffusion heads — extrapolating it to our Diffusion Policy setup is an inference, not a documented result. Dasari's R²=32% implies r≈0.57 (weak-to-moderate, not zero correlation) and is specific to MAE-family encoders on tabletop BC. Burns et al.'s OOD findings were obtained with frozen encoders and MLP-BC heads, so they inherit the same single-head caveat they help motivate. The audit table's characterization of OUR protocol (demo counts, absence of scratch baseline) is taken from the task statement plus robomimic conventions and should be checked against the actual run configs in the we-phase1/v0.3-v0.4 docs.

## Open questions

- Does the documented head-dependence of encoder rankings (BC vs RL vs VRF, Hu et al. 2023) extend to modern generative heads — Diffusion Policy and ACT — i.e., would our frozen-encoder rankings flip under an MLP-BC or ACT head on the same robomimic tasks? No surviving claim tests this directly.
- What feature interface (CLS vs pooled vs dense patch tokens, and any token-count normalization) do CortexBench, Voltron, and the 2024-2026 evaluation suites (Theia, MCR) actually use? The one claim asserting CortexBench's interface was refuted, leaving the literature's token-interface practice unestablished.
- Do fixed-epoch gates vs best-over-checkpoint selection change encoder RANKINGS (not just absolute success rates)? Both camps exist in the literature (Dasari vs R3M/CortexBench) but no verified source measures ranking sensitivity to the selection rule.
- How large a tie band is statistically justified at 50 rollouts and 3 seeds on robomimic-class tasks — is ±6-7pp conservative or optimistic once binomial rollout noise and seed variance are both propagated? The verified sources report CIs but none derive a decision threshold for this regime.

## Sources

- https://arxiv.org/abs/2203.03580 (primary, 5 claims extracted)
- https://arxiv.org/abs/2303.18240 (primary, 5 claims extracted)
- https://arxiv.org/abs/2203.12601 (primary, 5 claims extracted)
- https://arxiv.org/html/2304.04591 (primary, 5 claims extracted)
- https://github.com/facebookresearch/eai-vc (primary, 5 claims extracted)
- https://arxiv.org/abs/2312.12444 (primary, 5 claims extracted)
- https://arxiv.org/abs/2212.05749 (primary, 5 claims extracted)
- https://arxiv.org/abs/2310.09289 (primary, 5 claims extracted)
- https://arxiv.org/abs/2304.04591 (primary, 5 claims extracted)
- https://arxiv.org/abs/2502.03270 (primary, 5 claims extracted)
- https://arxiv.org/abs/2411.10175 (primary, 5 claims extracted)
- https://robomimic.github.io/study/ (primary, 5 claims extracted)
- https://arxiv.org/abs/2303.04137 (primary, 5 claims extracted)
- https://arxiv.org/html/2606.04233v1 (primary, 5 claims extracted)
- https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d (primary, 5 claims extracted)
- https://arxiv.org/html/2607.18236 (primary, 5 claims extracted)
- https://arxiv.org/pdf/2606.14153 (primary, 5 claims extracted)
- https://arxiv.org/abs/2407.20179 (primary, 5 claims extracted)
- https://arxiv.org/abs/2410.22325 (primary, 5 claims extracted)
