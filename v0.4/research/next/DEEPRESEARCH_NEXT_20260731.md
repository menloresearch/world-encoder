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
