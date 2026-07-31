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
