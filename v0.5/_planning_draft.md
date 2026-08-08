# v0.5 — The Planning Leg: what actually fixes compressed-latent planning

**Tag legend.** `[VERIFIED]` = external claim that survived source verification with a real URL and a
verbatim quote (see ledger, §6). `[REFUTED]` = external claim that failed verification; recorded so we
do not re-derive it, never cited as support. `[UNVERIFIED]` = referenced from our own prior-research
ledger but **not** re-verified in this pass — do not put in a paper without a fresh check.
`[MEASURED-OURS]` = our own measurement on our own harness, with the file that produced it. Ours are
not literature; they are the thing the literature has to explain.

Live-state timestamp: 2026-08-05 ~2:30pm SGT (06:30 UTC).

---

## 1. Bottom line

**The encoder is exonerated and the raw Euclidean latent-MSE-to-goal cost is only half the bug; the
other half — and the cheaper half to fix — is the search distribution CEM samples from.** On our own
goal pairs, 99.87% of the quantity CEM minimises on z8 is task-irrelevant, and the task-relevant
subspace is *below* what a random 5-d subspace would carry (0.55x chance), while the dense patch grid
that plans at 0.90 concentrates that same energy 114x above chance — the compact-vs-dense asymmetry is
now measured and held out `[MEASURED-OURS]`. But cost conditioning cannot be the whole story, because
FSQ has 20x-above-chance conditioning plus the best dynamics certificate of any arm and is currently
**0/50** `[MEASURED-OURS]`. What closes the gap is the second measurement: as CEM refines, the plan
walks from Mahalanobis² 13 to **334** against the demo action covariance — the demo scale is ~34-50
and pure isotropic noise is ~354 — i.e. by opt-step 30 our plan is statistically indistinguishable
from noise, and buys a 4.6x predicted-cost reduction doing it `[MEASURED-OURS]`. So: **do not touch
the encoder, do not chase token counts or open-loop certificates; constrain the CEM proposal to the
data-supported action manifold (free, eval-only, five independent papers behind it), keep the
probe-space cost arm as the second factor, and treat the adversarial arms as a gradient-based-planning
enabler rather than a CEM fix, because the AWM authors' own PushT table shows AWM buys exactly zero
under CEM+MPC.**

---

## 2. The diagnosis, cross-checked against the literature

### 2.1 Is our exploitation finding novel? Partly — the phenomenon is old, our *mechanism numbers* are new.

**Known, and named, since 2018.** `[VERIFIED]` Jiang et al. (https://ar5iv.labs.arxiv.org/html/1812.01129)
call it *planner overfitting* and state outright: "the best way to exploit a learned model can be to
exploit it incompletely". They measure a predicted-vs-realized divergence that is structurally our
curve: in Lunar Lander, policy performance **on the learned model** improves monotonically up to 2000
hidden units while performance in the **true environment** peaks near 250 units and then degrades
(averaged over 40 runs); intermediate planning discount γ̌ < γ beats γ̌ = γ in random MDPs; and
intermediate ε-greedy planning beats greedy planning. This is the citation that converts our decay
curves from "a bug we should be embarrassed by" into a named phenomenon with a prescribed remedy
(deliberately weaken the optimizer).

**Measured in CEM-MPC specifically.** `[VERIFIED]` https://ar5iv.labs.arxiv.org/html/2109.14311
measures the model over-estimating future reward at nearly every timestep on humanoid, and reports the
planner trying to "tunnel the ball through the bottom of the cup" on ball-in-cup. Their conclusion is
our conclusion: "As data fit is not the main indicator for planning performance and ensembles are not
sufficient to prevent model exploitation, a promising research direction is adversarial model
learning."

**Measured on PushT with a terminal latent objective — the closest external match.** `[VERIFIED]`
https://arxiv.org/html/2607.12547v1: open-loop CEM attains the **lowest** stage-2 (terminal) MSE
**0.011** while producing the **worst** stage-1 predictions **0.347** (vs 0.081 teacher-forced, 4.3x).
Their sentence is the abstract of our problem: "Unconstrained high-level CEM can search macro-actions
outside the support induced by training trajectories, producing subgoals that appear favorable under
the learned terminal objective but are poor targets for control."

**So what is actually ours.** Three things, none of which appears in any source we verified:

1. **A dose-response in the optimizer knob with realized decay, on a frozen compact latent.**
   `[MEASURED-OURS]` opt_steps 1/3/10/30 → predicted goal cost 1.60/0.98/0.47/0.35 (5x fall) with
   realized cost 1.70/1.73/1.74/1.87 (flat then worse). 2607.12547's diagnostic is about
   *hierarchical macro-actions* whose intermediate subgoals are unusable; it is **not** our
   opt_steps-versus-realized-success curve, and its authors attribute their flat-baseline degradation
   (94.0 → 52.7 → 18.0 at d=25/50/75) to horizon length and optimization difficulty rather than to
   exploitation per se `[REFUTED: "an independent paper measures OUR EXACT signature on the same
   harness" — different harness (LeWorldModel/stable-worldmodel, not DINO-WM), different mode]`.
   We now have this replicated **five** ways: the opt_steps sweep; within-solve rise-then-decay on z8;
   q4's collapse 0.02 → 0.20 → 0.10 → **0.00** held for ten consecutive evals; q16's oscillation
   0.06-0.16; and the ensemble's 0.28-at-127-iterations.
2. **The cost-energy asymmetry, held out, dimension-fair.** See §2.2. Nobody reports it.
3. **The off-support drift of the plan itself, quantified against the demo action covariance.**
   See §2.3. 2607.12547 *asserts* CEM searches outside the training support; we *measure* how far, in
   units of the data covariance, as a function of optimizer effort.

### 2.2 Does anyone report our compact-fails / dense-resists asymmetry? No. It is unclaimed, and we now own the mechanism.

`[MEASURED-OURS]` `world_tokenizer/pusht_cost_rowspace.py`, held-out probe (fit on 1392 of 1989 goal
pairs, evaluated on the rest), results in
`/home/menlo/brain/ishneet/world-encoder/results/pusht/cost_rowspace_heldout.json`. Fraction of the
goal-pair latent MSE — the exact quantity CEM descends — that lives in the rowspace of a ridge probe
latent → true PushT state:

| arm | dim | probe R² (held-out) | task-relevant % of goal cost | enrichment vs chance | closed-loop |
|---|---|---|---|---|---|
| z8 (ours, continuous) | 2048 | 0.979 | **0.134%** | **0.55x — BELOW chance** | 0.08 @ 3 iters |
| q4 | 1024 | 0.979 | 0.221% | 0.45x — below chance | 0.12 @ 3 iters |
| q16 | 4096 | 0.979 | 0.104% | 0.86x — below chance | 0.20 @ 3 iters |
| fsq (discrete) | 2048 | 0.878 | **4.937%** | 20.2x | **0.00 @ 2 iters** |
| **dense DINOv2 patch grid** | 75264 | 0.970 | **0.753%** | **114.1x** | **0.90 @ 2 iters** |

Enrichment (fraction ÷ k/d) is the dimension-fair comparison; raw % is not comparable across arms of
different dimensionality. I independently re-ran the dense arm this session and reproduced it exactly
(0.7527% mean / 0.5859% median; `results/pusht/cost_rowspace_dense.json`).

**Reading it.** The dense features that plan at 0.90 put 5.6x more of the cost energy in
task-relevant directions in absolute terms and **207x** more relative to chance. Our compact
continuous arms are *anti-concentrated* — a random 5-d subspace would carry more of the cost than the
5-d subspace that actually decodes the task state at R² 0.98. That is a stronger version of the TRM
diagnostic (`[UNVERIFIED]` arXiv 2605.22164, prior ledger: XY decodable at R² 0.998 yet the XY-probe
rowspace is <1% of terminal-goal MSE — we are below *chance*, not merely below 1%).

**Nobody else has this.** The nearest external things, and why they are not it:
- `[REFUTED as a causal claim]` DINO-WM (https://ar5iv.labs.arxiv.org/html/2411.04983) Table 2 Push-T:
  DINOv2 **patch** 0.90 vs three single-global-vector encoders DINO CLS 0.44, R3M 0.42, ResNet 0.2,
  under the identical terminal-latent-MSE cost ("The planning cost is defined as the mean squared
  error (MSE) between the current latent state and the goal's latent state, given by
  C=‖z^T − z_g‖²"). But there is **no token-count or patch-count ablation anywhere in the paper**, the
  comparison is confounded by pretraining objective/data/feature dim, the authors only "posit" the
  spatial-detail explanation, and on the simpler PointMaze *all* encoders including single-vector ones
  are near-perfect. It also only tests a single global vector — 8x256 is not a global vector. So it
  cannot adjudicate our 0.08.
- `[VERIFIED]` PRISM (https://arxiv.org/html/2606.07974v1) reports the adjacent fact from the other
  direction: swapping their task-trained ViT-tiny for a frozen DINOv2-base collapses absolute PushT to
  **10-15%**, because "DINOv2's CLS token encodes global semantics rather than the fine 2D spatial
  coordinates PushT requires". That is a *frozen-global-embedding* collapse in the neighbourhood of our
  0.08 — corroborating, but attributed to missing spatial coordinates, not to cost-energy geometry, and
  they never measure the cost decomposition.

**The complication we must state, not bury.** `[MEASURED-OURS]` FSQ has the best absolute conditioning
(4.94%, above dense's 0.75%) **and** the best 5-step certificate of any arm (0.464 vs z8 0.494, dense
0.726) and is at 0/50 over 2 MPC iterations with a within-solve peak of 0.04. So absolute cost
conditioning does **not** rank-order planning success. Two readings survive, and they are
distinguishable by the arms already running: (a) enrichment-relative-to-chance is the right variable
and dense's 114x is qualitatively out of reach for any 2048-d latent; (b) `[UNVERIFIED, our
hypothesis]` in a 75k-d redundant grid a spurious cost-reducing direction has to fight thousands of
correlated features, so redundancy — not conditioning — is what makes the dense cost hard to exploit.
Either way the honest conclusion today is **cost conditioning is necessary-but-not-sufficient**, which
is exactly what pt. 35 pre-registered, and it is why §3.1 (search support) outranks §3.3 (cost).

### 2.3 The mechanism, measured: CEM refinement walks the plan off the demo action manifold

`[MEASURED-OURS]`, two runs this session, no env rollouts needed (trunk + cached latents only).

**(a) The demo action manifold is ~6-dimensional inside our 50-dimensional search space.** From
381,073 chunks over 18,685 train episodes (`/home/menlo/dwm_data/pusht_noise/train/rel_actions.pth`),
in the planner's own normalized units, a chunk being 5 model steps x 10-d = 25 frames x 2 axes:

| quantity | value |
|---|---|
| per-dim std / \|mean\| of real chunks | 1.018 / 0.011 — **the marginal is already matched by normalization** |
| lag-1 / 2 / 3 / 5 / 10 autocorrelation of the action stream | +0.865 / +0.740 / +0.604 / +0.352 / -0.01 |
| chunk covariance eigenvalues | 8.40, 8.00, 6.48, 6.21, 4.20, 3.89, 2.60, 2.37 … 0.077 |
| effective rank (trace/λmax) | **6.18 of 50** |
| mean log-lik of real chunks: N(0,I) vs fitted Gaussian | −71.9 vs −37.4 → **34.4 nats/chunk** |
| Mahalanobis² under the demo covariance: real chunks vs isotropic samples | 50 (= d, by construction) vs 351 → **7.0x** |

This matters because `planning/cem.py:99-105` samples `torch.randn(num_samples, horizon, action_dim) *
sigma + mu` with `var_scale: 1` — an isotropic unit Gaussian. The *marginal* is right; the *temporal
covariance* is absent. Real PushT pushes are smooth (lag-1 0.865); i.i.d. Gaussian chunks are not.
**Every single candidate the world model is ever asked to score sits ~7x further from the demo manifold
than a real action sequence, from opt-step 1.**

**(b) And the optimizer's answer drifts further out the harder it optimizes.** Same CEM as the harness
(300 samples, topk 30, horizon 5), z8 trunk `pusht_dyn_dwm_z8/seed0.pt`, 20 val goal pairs at
(t, t+25):

| opt_steps | predicted goal cost | Mahalanobis² of the returned plan μ | Mahalanobis² of the 30 elites |
|---|---|---|---|
| 1 | 1.604 | **13** | 355 |
| 3 | 0.981 | 40 | 340 |
| 10 | 0.470 | 199 | 345 |
| 30 | 0.351 | **334** | 341 |
| *reference: true demo chunk* | – | **34** (train mean 50 = d) | – |
| *reference: isotropic N(0,I)* | – | **354** | – |

The predicted-cost column reproduces our recorded dose-response (1.60/0.98/0.47/0.35) to three
decimals, which validates this re-implementation against the harness. The new column is the mechanism:
the returned plan starts *inside* the demo support (13, tighter than a real chunk because μ starts at
the distribution centre), crosses the demo scale (~34-50) between opt-step 3 and 10, and by opt-step 30
is at 334 — **statistically indistinguishable from pure noise (354)**. CEM buys its 4.6x predicted-cost
reduction by walking ~10x off-support. Combined with our existing fact that true action sequences
realize ~1.3px error and our model ranks them argmin *under random sampling*, the picture is complete:
the model is accurate on-support, the cost is uninformative off-support, and the optimizer's entire job
as currently configured is to find the off-support directions.

---

## 3. Ranked remedies, by (evidence strength) / (cost to us)

### 3.1 — RUN NEXT, RANK 1: support-constrained CEM proposal + an on-manifold term inside the cost

**Method.** Stop sampling candidates from N(0, I). Sample from the demo action-chunk distribution, and
(second variant) add a log-prior/Mahalanobis penalty **inside** the objective so it survives all 30 opt
steps rather than only shaping the initialization.

**Evidence — five independent sources, all pointing at this one intervention.**
- `[VERIFIED]` https://arxiv.org/html/2607.12547v1, **on PushT**: empirical-macro (data-supported) CEM
  reaches **64.0% at d=50 (+11.3pp)** and **32.7% at d=75 (+14.7pp)** over flat CEM, while
  *unconstrained* hierarchical CEM **hurts** (15.3% vs the flat baseline's 18.0% at d=75) — more
  optimizer power over a learned model is negative. Post-hoc, no retraining. Their own quote names the
  disease: macro-actions "outside the support induced by training trajectories".
- `[VERIFIED]` SAGE, https://arxiv.org/html/2607.17973v1: holding the encoder, latent dynamics **and
  the planning cost** fixed and changing only the action proposal + intermediate targets takes PushT
  from **12.7% → 64.7%** at H=150 (+52pp) and OGBench-Cube 26.7% → 67.3%, at matched CEM budget (300
  candidates / 30 rounds / 30 elites) — "we keep the encoder, latent dynamics, and planning cost fixed,
  and instead improve the local targets and action proposals presented to the frozen world model".
  Their broken baseline (12.7%) is within noise of our 0.08. **Cost caveat `[REFUTED: "entirely
  post-hoc"]`:** SAGE is post-hoc w.r.t. encoder and dynamics but is **not eval-only** — it trains a
  20.71M-param subgoal generator and a 13.61M-param action generator (transformer decoder + 8-mode
  GMM) per task on ~400k aligned windows of that task's expert data, and reports no wall-clock. Also
  `[VERIFIED]` their prior-only ablation is **16.0%**, so refinement through the frozen world model
  still does most of the work — the prior must *constrain* search, not replace it.
- `[VERIFIED]` TAP, https://ar5iv.labs.arxiv.org/html/2208.10291, puts the term **in the objective**:
  discounted predicted reward + α·log min(p(z_1..z_M|s_1), β^M), because "only allowing the
  in-distribution actions prevents the planner from exploiting the weakness of the model by querying
  the actions with high uncertainty whose values are most susceptible to overestimation."
- `[VERIFIED]` MOPP, https://ar5iv.labs.arxiv.org/html/2105.07351: behaviour-cloned proposal + pruning
  candidates by max pairwise ensemble discrepancy, all at operation time on a pre-trained ensemble.
  D4RL medium-expert hopper **95.4±28.0** vs MBOP 55.1±44.3 vs MOPO 23.7±6.0; walker2d 92.9±14.1 vs
  70.2±36.2. Ablations (MOPP-noP, MOPP-noMQ) show the prior and the pruning are **both** needed.
  Caveat: state-input D4RL locomotion, not pixels/compact tokens, and ±28 variance demands ≥3 seeds.
- `[VERIFIED]` TD-MPC2, https://ar5iv.labs.arxiv.org/html/2310.16828: the most successful
  compact-latent planner seeds a fraction of its 512 candidates from a learned policy prior — "To
  accelerate convergence of planning, a fraction of action sequences originate from the policy prior p,
  and we warm-start planning by initializing (μ, σ) as the solution to the previous decision step
  shifted by 1."

**Cost status: POST-HOC / EVAL-ONLY** for the variant we should run first (a fitted Gaussian needs no
network and no gradient step); the GMM/BC-prior upgrade needs a tiny supervised fit on data we already
have, still no encoder or world-model retraining.

**What WE run, concretely, on the DINO-WM PushT harness.**
- Fit once, offline, CPU, minutes: mean + full covariance of the 50-d demo action chunk from
  `/home/menlo/dwm_data/pusht_noise/train/rel_actions.pth` in the planner's normalized units
  (`(a/100 − AM)/AS`, `AM=[-0.0087,0.0068]`, `AS=[0.2019,0.2002]`), plus a k=8 GMM as the upgrade path.
  Persist beside the probe under `/mnt/nas/data/robocasa/kepler_ckpt/`.
- **Arm A (proposal):** in `planning/cem.py:99-105`, replace `torch.randn(...) * sigma[traj] +
  mu[traj]` with `mu + L @ eps` where `L = chol(C_demo)` scaled by `var_scale`, i.e. colored sampling
  in the demo covariance, keeping `action[0] = mu` and the elite update untouched. ~10 lines. Nothing
  else in the harness changes, so it is directly comparable to hybrid v1's 4/50.
- **Arm B (cost, TAP-style):** add `+ lambda * maha2(chunk)` (or `−lambda·log p_GMM`) to the objective
  returned by `planning.objectives.create_objective_fn`. Sweep lambda ∈ {0.01, 0.1, 1} chosen so the
  penalty at the demo scale (~34-50) is ~10% of the typical goal cost.
- **Arm A+B factorially**, 3 seeds, `n_evals=50`, `goal_H=5`, everything else at `conf/plan_pusht.yaml`
  defaults, via `plan_hybrid.py --dyn-ckpt pusht_dyn_dwm_z8/seed0.pt`.

**Kill gate.** Two conditions, both pre-registered: (i) diagnostic — Mahalanobis² of the returned plan
at opt-step 30 must stay ≤ ~50 (the demo scale) instead of 334; (ii) outcome — cumulative
`self.is_success` must exceed **4/50 at 3 MPC iterations** (hybrid v1's matched point). If the
Mahalanobis² is contained but success does not move past 0.08 at matched iterations, the support story
is falsified as a *sufficient* fix and everything shifts to §3.3 + §3.4. If predicted cost stops
falling *and* success does not rise, the proposal is too tight — halve the shrinkage before concluding.

### 3.2 — RUN NEXT, RANK 2: deliberately weaken the optimizer (three knobs, all config-level)

**Method.** Stop optimizing so hard, and give CEM the memory TD-MPC2 gives it.

**Evidence.** `[VERIFIED]` https://ar5iv.labs.arxiv.org/html/1812.01129 — "the best way to exploit a
learned model can be to exploit it incompletely"; intermediate planning discount and intermediate
action-selection noise both beat their greedy limits, and the optimal amount of each knob **increases
as the model becomes more accurate**. `[VERIFIED]` TD-MPC2 deliberately stops at **6** MPPI iterations
with horizon **3** and warm-starts from the previous shifted solution. Ours: 30 opt steps, no prior, and
our own sweep already says opt_steps=1 realizes best (1.70) and 30 is worst (1.87) `[MEASURED-OURS]`.

**Cost status: POST-HOC / EVAL-ONLY.** Pure plan-time knobs on frozen checkpoints.

**What WE run.**
- **Warm-start is currently *inactive* in our config and this is a bug-shaped finding**
  `[MEASURED-OURS]`, read from the code: `planning/mpc.py:94` sets `memo_actions = actions[:,
  n_taken_actions:]`, and `conf/planner/mpc_cem.yaml` has `horizon: 5` with `n_taken_actions: 5`, so
  `memo_actions` is always shape (b, 0, d) and `CEMPlanner.init_mu_sigma` restarts from μ=0, σ=1 at
  every single replan. (`mpc_gd.yaml` uses `n_taken_actions: 1`, so GD *does* warm-start.) Set
  `n_taken_actions=2` (and separately 1) to activate it; report MPC-iteration counts alongside success
  because this multiplies solves per unit env progress.
- **opt_steps U-curve** at fixed everything else: {1, 3, 6, 10, 30}. 6 is TD-MPC2's number. Report the
  U-shape as the headline figure — with 1812.01129 cited, a weaker optimizer is a *result*, not a
  concession.
- **Action-noise floor:** clamp `sigma` to a minimum (e.g. 0.3) so the elite std cannot collapse; this
  is their ε-greedy knob in continuous form.
- **Their falsifiable prediction, tested on our data:** "the optimal amount of each knob increases as
  the model becomes more accurate." Our certificates are q4 0.474 / q8 0.494 / q16 0.469 / fsq 0.464
  `[MEASURED-OURS]`, so the better-certificate arms should tolerate more opt_steps. If they do not,
  say so — it is a clean external-prediction failure on compact latents and worth a paragraph.

**Kill gate.** If the U-curve is monotone-flat (no interior optimum) across all four arms, drop the
whole planner-weakening axis and keep only the single best opt_steps as a budget-fair baseline setting.
If warm-start alone moves cumulative success past 4/50@3 iters, it must be reported as a **harness
configuration** effect, not a Kepler effect — and it retroactively qualifies every compact-arm number
we have.

### 3.3 — KEEP RUNNING, RANK 3: task-resolved (probe / TRM-hybrid) planning cost

**Method.** Encoder, dynamics, planner and protocol frozen; only the *space the MSE is computed in*
changes, from raw all-token latent to the 5-d probe (task-state) projection, or a probe-weighted metric
that keeps a downweighted rowspace-orthogonal residual.

**Evidence.** Our own rowspace numbers (§2.2) are the primary justification: 0.134% and *below chance*
is a defect of the metric, not of the representation, given probe R² 0.979. `[UNVERIFIED]` TRM
(arXiv 2605.22164, prior ledger) is the external precedent — TwoRoom 7%→97%, post-hoc planner-facing
cost — **with the authors' own caveat that PushT gains do not transfer cleanly to closed-loop and that
hybrid rather than replacement costs are recommended**, which is exactly why the `hybrid` mode exists.
`[VERIFIED]` https://arxiv.org/html/2607.16591 supplies the strongest external analogue: post-hoc costs
built from *observable* world signals on the same frozen model and planner cut collisions to 4%/1%
(min-lidar), 12%/11% (time-to-collision), 16%/14% (learned collision predictor) — while the
model-internal proxy made things worse. Their winning move is our move: put a cost term on a quantity
the environment reveals.

**Cost status: POST-HOC / EVAL-ONLY** (probe already fit, R² 0.979 held-out).

**Status and what WE run next.** Already live as `hybrid_cost_probe` and `hybrid_cost_hybrid`
(`models/kepler_hybrid.py`, `cost_space=probe|hybrid`). Current state `[MEASURED-OURS]`: no outer read
yet; within-solve traces 0 → 0.12 (probe) and 0.02 → 0.08 (hybrid) over 8 CEM evals. Next: (i) read
both at 3 MPC iterations against 4/50; (ii) **run the 2x2 of §3.1-Arm-A x probe cost** — this factorial
is the paper, and PRISM is the reason `[VERIFIED]`: a proposal fix on a frozen global embedding still
capped PushT at 10-15% absolute, so proposal-only should give a large relative lift but not 0.90 unless
the cost is also task-resolved; (iii) add a PushT-observable term (T-block overlap / contact) as the
2607.16591-style variant.

**Kill gate.** If probe-space at 3 iters lands ≈0.08, the cost is exonerated as a *sufficient* fix and
the leg narrows to search support + smoothness — which is a publishable negative given §2.2 predicted
the opposite. If probe wins but hybrid loses (or vice versa), report both; do not quietly pick one.

### 3.4 — RANK 4: ACID — inverse-dynamics action-consistency residual in the CEM cost

**Method.** `[REFUTED as "the strongest measured plan-time defence" — the paper makes no such
comparison and never tests alternatives]`. `[VERIFIED, corrected]` ACID
(https://arxiv.org/html/2607.02403v1) adds an IDM action-consistency residual to the CEM cost at
decision time with adaptive weight w_a = λ·σ_g/σ_a, post-hoc on frozen world models. Measured: DINO-WM
Rope Chamfer 1.38 → 0.56, Granular 0.49 → 0.30 (Tab. 2); Le-WM Reacher 76 → 88 (+12pp), Cube 70 → 74,
PushT 96.0 → 100.0; PLDM Reacher 76 → 90 (+14pp), Cube 58 → 68, PushT 72.0 → 76.0 (Tab. 1). Verifier
overhead 8.9% (PLDM) to 39.4% (DINO-WM) of world-model forward latency (Tab. 5). The 10x-sample-
efficiency headline is **one task, one model, one λ** (Le-WM Reacher, Tab. 3 — *not* Tab. 4 — Ours n=30
scores 76 = Original n=300's 76, and the sweep is non-monotonic: 84 at n=50 > 82 at n=150); there is no
PLDM, PushT or DINO-WM budget sweep.

**Cost status: POST-HOC** — no encoder or world-model retraining, but it needs an IDM verifier. **We
already have one:** Kepler was pretrained with an inverse-dynamics auxiliary, so the verifier is a
frozen head we own. That is the reason this ranks above the amortized routes.

**Why only rank 4.** Its PushT numbers are ceiling-regime (+4pp on 96% and 72% baselines) and its
DINO-WM evidence is Chamfer distance on Rope/Granular, not PushT success. It is **untested in a
near-zero-success regime like our 0.08**, and its largest gains (+12/+14pp) are Reacher, not
manipulation-with-contact.

**What WE run.** Add the IDM residual to `planning/objectives.py` with ACID's adaptive weight; λ sweep
{0.03, 0.07, 0.15}; single arm on z8, 3 seeds. Run it *after* §3.1, and if §3.1 lands, run it *on top
of* §3.1 — ACID and a support-constrained proposal are the same family (both penalise implausible
actions) and their interaction is worth one cell.

**Kill gate.** Must beat 4/50@3 iters on its own; if it only helps when stacked on §3.1, report it as
an increment, not a mechanism.

### 3.5 — RANK 5, COMBINED ONLY: MOPP-style ensemble discrepancy *pruning* + behaviour prior

**Method.** Not a mean rollout, and not a variance penalty alone: prune elite candidates whose max
pairwise ensemble discrepancy exceeds a threshold, *and* draw proposals from the behaviour
distribution.

**Evidence, both directions.** `[VERIFIED]` MOPP (numbers in §3.1) with the ablation showing both
halves are needed. `[VERIFIED]` FOWM (https://ar5iv.labs.arxiv.org/html/2310.16029) — return minus
λ·std over the Q-ensemble on a 50-d compact latent, on a real xArm: Reach 89±0 vs 78±12, Pick 50±6 vs
28±6 at 20 online trials; "To mitigate extrapolation errors during online interaction, we propose to
regularize the planner at test-time by balancing estimated returns and (epistemic) model uncertainty."
**Against:** `[VERIFIED]` https://arxiv.org/html/2607.16591 — "Penalising model uncertainty as a risk
signal increases collisions from 26% to 34%: the proxy is anti-correlated with safety in this regime",
AUC 0.60 as a risk predictor, with r<0.15 overlap between dynamics uncertainty and task failure. And
`[VERIFIED]` https://ar5iv.labs.arxiv.org/html/2109.14311 says ensembles are "not sufficient to prevent
model exploitation" — which is precisely what our K=3 mean-rollout showed (0.08 → 0.28, but at ~127 MPC
iterations vs dense's 2) `[MEASURED-OURS]`.

**Cost status: POST-HOC / EVAL-ONLY** on the three trunks we already have.

**What WE run.** One-line change in `models/kepler_hybrid.py:rollout` from mean-of-members to a
per-candidate discrepancy `max_{i,j}‖f^i − f^j‖²`, used as a **hard prune of the elite set** (MOPP), not
as an additive penalty; combined with §3.1's proposal; λ/threshold sweep {p90, p75, p50 of the
candidate discrepancy distribution}. Report the two-way ablation (prior-only, prune-only, both) so we
can say which half carries it.

**Kill gate.** The 2607.16591 negative applies specifically to *additive uncertainty penalties*, so:
run the penalty variant **once** at λ ∈ {0.5} as a confirmation cell and drop the axis if it degrades,
exactly as they predict. Prune-only must also cut the iteration count — if it reaches 0.28 but still
needs >50 MPC iterations, it is not a fix, it is the ensemble result again.

### 3.6 — RANK 6: amortize the planner away (GC-IDM / learned latent planner)

**Evidence.** `[VERIFIED]` https://arxiv.org/html/2606.09311v1: flat latent-world-model CEM on PushT
collapses to **3.52%** at t=75 and **0.00%** under random initialization, while an amortised latent
planner trained on frozen world-model embeddings reaches **91.80% / 82.42%**; planner overhead 2.1±0.1
ms (deterministic) or 242.6±12.2 ms (diffusion) vs **926.6±45.5 ms** for CEM; and it holds at **76.17%
with 40x less data (200 vs 8,318 episodes)** — "The latent planner is trained on latent representations
of successful demonstrations computed by the frozen, pre-trained world model's encoder". `[VERIFIED,
corrected]` GC-IDM (https://arxiv.org/html/2605.08732v1): a ~1.5M-param goal-conditioned inverse
dynamics MLP, ~20 min/env on frozen latents, "matches or outperforms CEM in seven of eight
environment-protocol cells while reducing per-decision cost by 100-130x" (0.71 ms vs ~45,000 predictor
calls) — Two-Room 100.0 vs 84.0, Cube 98.7 vs 67.0, Reacher 99.7 vs 70.3.

**`[REFUTED]` — and this refutation is load-bearing for us.** On **Push-T specifically**, GC-IDM's
advantage evaporates: the single losing cell **is** Push-T (n=50: 84.7±5.0 vs CEM 89.3±6.4), the n=200
cell is a near-tie (84.2 vs 82.5), and CEM's Push-T plateau in the 12-config budget sweep is **90%** —
*above* GC-IDM. The authors: "Push-T is exactly the regime in which longer contact sequences make local
inverse recovery inherently more challenging." Also, GC-IDM optimizes **no** planning objective — it is
supervised action regression, `‖gc-idm(z_t, z_{t+h}, h) − a_t‖²` — so it sidesteps our diagnosis rather
than fixing it.

**Cost status: NOT eval-only.** Encoder and predictor stay frozen, but a new policy head is trained on
the same offline demo data.

**What WE run, and why it is still worth a slot.** Two distinct uses. (1) **Baseline credibility:**
0.00% random-init CEM and 3.52% at t=75 are *published* numbers in our regime — cite them when a
reviewer asks why our 8-token CEM planning is 0.08. (2) **Low-N transfer:** their 200-episode /
76.17% result matters because our Square substrate is 200 demos, so we can claim the amortised route
survives our low-N regime. Run the **deterministic** variant first (2.1 ms overhead ⇒ same-day on the
frozen checkpoint); the speed win transfers to PushT, the accuracy win does not.

**Kill gate.** If the deterministic head does not clear 4/50 within one training run, do not escalate
to the diffusion variant — the PushT-specific evidence says it is the wrong task family for this fix.

### 3.7 — ALREADY RUNNING: what to do with the adversarial arms

**Current state `[MEASURED-OURS]`.** Certificates all improved on clean z8 (0.494) and are **monotone
in ε** — adv-hi 0.474 < adv-mid 0.476 < adv-lo 0.481 — falsifying our registered prediction that large
ε would blunt accuracy. Closed-loop, no outer reads yet; within-solve traces are flat and *inverted*
w.r.t. the certificate: adv-lo 0.02-0.08 (13 evals), adv-mid ~0.06 (19 evals), **adv-hi 0.02-0.04 (18
evals, the worst of the three despite the best certificate)**. Clean-finetune control 0 → 0.06 (6
evals).

**The decisive external fact `[VERIFIED]`, and it is negative for the arm as configured.**
https://arxiv.org/html/2512.09929v1: on PushT under MPC + CEM, AWM delivers **exactly zero** gain —
92% for both DINO-WM and AWM. The entire PushT MPC gain is in gradient-based planning (Adam 76 → 92,
GD 56 → 66); open-loop PushT is GD 38 → 56, Adam 54 → 82, CEM 78 → 94. Their own summary: "In the MPC
setting, Adam GBP with Adversarial World Modeling outperforms CEM with DINO-WM on PointMaze and Wall
and matches CEM on PushT." **Our failing planner is CEM under MPC on PushT — the exact cell where AWM
buys nothing.**

**Three recipe deviations to fix before any re-run** (all from source verification of the paper and the
released code, all one-liners):
1. `[VERIFIED]` **The regression target must shift.** Their objective regresses the *clean absolute*
   next latent: "min_θ E[max_{δ_a∈B_a, δ_z∈B_z} ‖f_θ(Φ_μ(o_t)+δ_z, a_t+δ_a) − Φ_μ(o_{t+1})‖²]". Our
   `world_tokenizer/pusht_dynamics_dwm_adv.py:114` keeps `r = z1 − z0` as the target under
   perturbation, with a comment saying so. For a residual predictor f = z + g, their objective implies
   the residual target is `z1 − z0 − δ_z`, i.e. `r − (z0_adv − z0)`. Holding the true residual fixed
   trains a strictly different, weaker objective — **this is the most likely reason a repro shows no
   gain.**
2. `[VERIFIED]` **No clean/adversarial mixing.** Algorithm 3 trains only on the perturbed batch and the
   released `train.py` takes exactly one backward on the perturbed forward. Our `--adv-weight 0.5`
   halves the adversarial signal, and the paper has **no ablation for clean-mixing in either
   direction** — it is a free variable we introduced.
3. `[VERIFIED]` **ε is a scaling factor on the empirical latent std, not an absolute.** The released
   `conf/train.yaml` literally names them `visual_eps_factor: 0.05 / proprio_eps_factor: 0.02 /
   action_eps_factor: 0.02`. Our cached z8 val latents have mean per-sample std **0.600**, so their
   PushT setting corresponds to an absolute ε_z of **0.030**. Our arms used ε_z = 0.025 / 0.05 / 0.10,
   i.e. λ = 0.042 / 0.083 / 0.167 → **adv-lo is the paper-matched arm and mid/hi are 1.7x/3.3x over**;
   on the action side, with normalized actions at ~unit scale by construction, our ε_a = 0.05/0.10/0.20
   is **2.5x-10x** their 0.02. Also `[VERIFIED]` follow the *code* not Algorithm 3: one
   `autograd.grad`, one sign step, per minibatch over the whole history window — but do add their
   uniform-random init δ ~ U(−ε, ε) with step α = 1.25ε, which we omit.

**Verdict — what to run next and what to drop.**
- **Let the three closed-loop reads finish** (they are already burning GPU) and read them at 3 MPC
  iterations against 4/50. They are cheap and they are the outcome metric; the certificate has never
  once predicted closed-loop.
- **DROP** any wider ε sweep under CEM+MPC. The authors' own PushT table predicts no change, and our
  within-solve traces are already flat with the largest-ε arm worst.
- **RUN** the one arm their evidence actually supports: **AWM finetune + gradient-based planning as a
  joint arm.** We already have `planning/gd.py` and `conf/planner/mpc_gd.yaml` (horizon 5, lr 1,
  opt_steps 1000, `n_taken_actions: 1`, so warm-start is active). Their PushT GBP settings are Adam,
  lr 0.2, 100 opt steps, 10 MPC steps. **Code blocker to fix first:** `KeplerHybridWM.rollout` and
  `encode_obs` are decorated `@torch.no_grad()` (`models/kepler_hybrid.py:86,101`), which silently
  breaks any gradient-based planner — add a grad-enabled path before launching.
- **RE-RUN** one corrected arm (pure adversarial, U(−ε,ε) init + 1.25ε step, shifted residual target,
  λ_z = 0.05 ⇒ ε_z = 0.030, ε_a = 0.02) at one epoch, predictor LR 1e-5, encoder untouched
  `[VERIFIED: "The embedding function Φ_μ is taken to be the pre-trained DINOv2 encoder, and remains
  frozen while finetuning the transition model f_θ"]` — but only *with* the GD planner.
- **Kill gate.** If corrected-AWM + GD does not beat 4/50@3 iters, close the adversarial axis entirely
  and cite 2512.09929's PushT-CEM row as the reason it was never going to work under our planner. That
  is a clean, citable negative.

### 3.8 — ALREADY RUNNING: what to do with the FSQ arm

**Current state `[MEASURED-OURS]`.** Best 5-step certificate of any arm (0.464), best absolute cost
conditioning (4.94%, 20.2x enrichment), and **0/50 cumulative over 2 MPC iterations** with a
within-solve peak of 0.04. Board for the quantized family at matched-ish iterations: q4 6/50 = 0.12 @3,
q16 10/50 = 0.20 @3, fsq 0/50 @2, z8 4/50 @3, dense 45/50 = 0.90 @2.

**Verdict.** Let it reach 3 iterations for the matched comparison against hybrid v1's 4/50, then
**close the discreteness-as-armour hypothesis as unsupported on our task** and do not build further
quantized variants. This is a real finding, not a null: `[UNVERIFIED]` CompACT (arXiv 2603.05438 /
OpenReview z9KgtDP6LB) maintains a 256-token baseline with 16 discrete tokens on RoboMimic Lift under
CEM (56 vs 56) but contains **zero analysis of planner exploitation**; `[UNVERIFIED]` DCWM/DC-MPC
(arXiv 2503.00653) attributes the discrete benefit to **stochasticity** (cross-entropy +
straight-through Gumbel), with deterministic-discrete + MSE underperforming — and our FSQ arm is
deterministic-discrete with an MSE objective, i.e. exactly their failing configuration. Note also
`[VERIFIED]` TD-MPC2 chose bounded SimNorm simplex latents *over* discrete codes for precisely the
gradient-stability reason. **q16 at 0.20 is currently the best compact arm and deserves one honest
sentence, but token count remains a null** (certificates saturate by 4 tokens; outcomes 0.12/0.08/0.20
are all in the same failing band against 0.90).

---

## 4. The decisive question: is raw Euclidean latent MSE to a goal latent ever used *successfully* with compact latents?

**Direct answer: yes, in exactly one documented configuration — and that configuration requires
encoder retraining with explicit latent-geometry regularization, works only at short horizons, and has
no frozen-pretrained-encoder counterpart anywhere in the verified literature. Every compact-latent
planner that works at scale either shapes the latent so Euclidean distance is meaningful, or replaces
the cost with reward/value, or constrains what the optimizer may search.**

The evidence, in order of how much it constrains us:

1. **The cost itself is standard and does work — on dense features.** `[VERIFIED]` DINO-WM's planning
   cost is exactly this: "C=‖z^T − z_g‖²", and it plans PushT at 0.90 with a 75k-d patch grid
   (reproduced on our hardware, 45/50 in two MPC iterations). So the cost is not intrinsically broken.
2. **The one compact success: LeWM.** `[VERIFIED, corrected]` https://arxiv.org/html/2603.19312v1 plans
   on a compact single-vector JEPA latent with a terminal L2 goal cost, CEM at 300 samples x 30 steps,
   top-30 elites. Its two differentiators are decisive for us: **(a) the encoder is trained end-to-end
   on the task data from pixels, not frozen-pretrained; (b) the latent distribution is explicitly
   regularized toward an isotropic Gaussian via SIGReg.** Both require **ENCODER RETRAINING**, so
   neither is available to us post-hoc. `[REFUTED]`: the widely-quoted "~94% on PushT" is **not in the
   paper** — it reports only relative numbers ("achieving an 18% higher success rate on PushT" than
   PLDM) and Figure 6; do not cite 94%.
3. **Even that success is horizon-fragile.** `[VERIFIED]` On the LeWM harness, flat single-level CEM
   over the compact latent with this cost collapses to **3.52%** at a 75-step offset and **0.00%** from
   random init (2606.09311), and **12.7%** at H=150 (SAGE, 2607.17973). `[VERIFIED, corrected]` and
   **94.0% → 52.7% → 18.0%** as the goal offset grows d=25/50/75 (2607.12547) — which its authors
   attribute to horizon length and optimization difficulty. Successful compact-latent planners all plan
   *very short*: TD-MPC2 horizon 3, FF-JEPA 25-step subproblems.
4. **The scale winner does not use latent distance at all.** `[VERIFIED]` TD-MPC2 — 104 tasks, one
   hyperparameter set, 512-d SimNorm-bounded latent — maximizes discounted predicted reward plus a
   terminal Q. Never a goal-latent distance. (The one piece we cannot copy post-hoc: it needs a reward
   model on PushT.)
5. **And the ones that keep a latent cost bolt something on.** `[VERIFIED]` TAP adds
   α·log min(p(z|s), β^M) *inside* the score. `[VERIFIED]` FOWM subtracts λ·ensemble-std. `[VERIFIED]`
   ACID adds an IDM consistency residual. `[VERIFIED]` an energy/density model added as an additive
   plan-time cost on a frozen PETS model raised half-cheetah return 10955±2865 → 13052±2814 (+19%,
   CEM+DEEN) and 12967±3216 (CEM+DAE) — though `[REFUTED]`: that paper has **no plain-Adam baseline**,
   so it does *not* show an on-manifold penalty rescuing a gradient optimizer, its error bars
   (~2800-3200) overlap heavily, and it is one locomotion environment with no manipulation or
   compact-latent evidence.
6. **The strongest single result keeps the cost and fixes the search.** `[VERIFIED]` SAGE: cost held
   **fixed**, PushT 12.7% → 64.7%. That is the empirical answer to "must we replace the cost?" — **no,
   but you must stop the optimizer from leaving the support**, which is exactly what our Mahalanobis
   drift measurement says is happening (§2.3).

**Consequence for Kepler.** For a *frozen, pretrained, compact* encoder there is **no** verified
instance of raw terminal latent MSE planning successfully. Our options are therefore (i) constrain the
search (§3.1, cheapest, best-evidenced, keeps the cost), (ii) make the cost task-resolved (§3.3,
already running), (iii) add an on-manifold/consistency term to the cost (§3.4/§3.5), or (iv) retrain
the encoder with a latent-geometry objective (LeWM's SIGReg route) — which is off the table for v0.5
but is the honest recommendation for a v0.6 pretraining change, and is worth stating in the paper as
the thing our result implies about *how compact encoders should be pretrained if they are to be planned
over*.

---

## 5. What will NOT work — including things we currently believe

1. **Chasing better open-loop rollout certificates. Stop using them as a decision metric.**
   `[VERIFIED]` https://arxiv.org/html/2607.16591 measures a **2.04x MSE gap across model architectures
   producing statistically EQUIVALENT planning (TOST p=0.003-0.042)** and an **8.15x MSE gap giving no
   reward difference (Mann-Whitney p=0.42)**. This directly explains our paradox: our 0.494-vs-0.726
   rollout-MSE advantage should *never* have been expected to buy planning success. Our own data agrees
   four ways `[MEASURED-OURS]`: z8 0.494 beats dense 0.726 yet plans 0.08 vs 0.90; FSQ has the best
   certificate (0.464) and is at 0.00; adv-hi has the best adversarial certificate and the worst
   within-solve trace; and q4/q8/q16 certificates are indistinguishable (0.474/0.494/0.469) while
   outcomes differ. Keep the certificate as a *sanity gate*, never as a *predictor*.
2. **Ensembles, scaled up.** `[VERIFIED]` https://ar5iv.labs.arxiv.org/html/2109.14311 states outright
   that ensembles are "not sufficient to prevent model exploitation" while still helping raw reward
   (ball-in-cup doubles with 5 nets vs 1; walker/finger up to +20%). Our K=3 mean-rollout result (0.08
   → 0.28 at ~127 MPC iterations vs dense's 2) is therefore a **confirmation of published guidance**,
   not a disappointing null — quote it as prior art that predicted our outcome, then move on. Do not
   run K=5 or K=7.
3. **Ensemble-disagreement / model-uncertainty penalties on their own.** `[VERIFIED]`
   https://arxiv.org/html/2607.16591: "Penalising model uncertainty as a risk signal increases
   collisions from 26% to 34%: the proxy is anti-correlated with safety in this regime", AUC 0.60,
   r<0.15 overlap between dynamics uncertainty and task failure — while observable-grounded post-hoc
   costs on the *same* frozen model and planner cut collisions to 4%/1%. Do not spend a wave here;
   only the MOPP *pruning*-plus-prior combination (§3.5) is defensible. Caveat: their setting is 2D
   lidar navigation and a pendulum — no manipulation, no PushT, no JEPA.
4. **Fewer tokens, or a smaller latent dimension.** `[VERIFIED]` 2607.12547 explicitly tried the
   alternatives we would reach for and reported them as marginal: VQ-128 macro-actions needed
   **retraining** to reach 34% at d=75 (vs 32.7% for the free, post-hoc empirical-macro), and a
   **reduced latent dim (8 vs 32) gave NO improvement**. This is directly relevant because our whole
   pitch is 8 tokens: compactness is not what is buying or costing planning performance. Our own
   certificate saturation by 4 tokens says the same.
5. **Multi-step training loss beyond ~5 steps.** `[VERIFIED]` 2109.14311: multi-step training loss
   helps only up to 5 steps. Our certificate protocol is already exactly 5-step — do not extend it and
   do not expect gains from longer-horizon training losses.
6. **Multi-step PGD instead of FGSM.** `[REFUTED, corrected]` 2512.09929 Table 11: PGD "does not
   consistently outperform FGSM" — Wall open-loop FGSM 34 vs 2-step 8 vs 3-step 14, Wall MPC 94/90/94 —
   **but** PointMaze open-loop *favours* PGD (70 vs 80 vs 78) and PointMaze MPC is saturated at
   94/96/94. So PGD buys nothing reliable for 2x the training cost, and it is *not* uniformly worse;
   PushT is absent from Table 11 entirely, so nothing there licenses a PushT conclusion.
7. **Believing AWM will fix our CEM failure.** `[VERIFIED]` See §3.7: 92% vs 92% on PushT MPC+CEM. Also
   `[REFUTED]` the stronger claim that "AWM is a GBP-specific fix that regresses several CEM settings" —
   under closed-loop MPC with CEM it *improves* on Wall (94 vs 82) and PointMaze (98 vs 90); the CEM
   regressions are confined to open-loop and are within 2pp. And `[VERIFIED]` a reproducibility caveat
   worth citing on its own: the AWM authors **could not reproduce DINO-WM's published Wall open-loop
   CEM number**, measuring 32% against the published 74%, and reported DINO-WM's better value.
8. **Expecting AWM evidence to transfer to compact latents at all.** `[VERIFIED, corrected]` The paper
   contains **zero** results on compact or pooled latents — every result uses the frozen dense DINOv2
   patch grid. The only non-DINO architecture is IRIS (VQ-VAE), on Wall only, at near-zero absolute:
   IRIS 0% GD / 4% CEM, IRIS+Online-WM 0%/0%, IRIS+Adversarial-WM 8% GD / 6% CEM (Tab. 8). `[REFUTED]`
   the "196 patches x 384-d, 10-d proprio, 10-d action" specification is **not in the paper** — those
   are DINO-WM codebase details (and 384-d is likely wrong; DINO-WM uses ViT-B/14, 768-d). Do not
   attribute them to this source.
9. **Treating SAGE as a free lunch.** `[REFUTED]` See §3.1: two new modules totalling ~34M params
   trained per task on ~400k expert windows. Post-hoc w.r.t. encoder and dynamics, **not eval-only**,
   and it matters for our low-N regime.
10. **Treating ACID as the best available defence.** `[REFUTED]` The paper never compares against other
    anti-exploitation methods and makes no superiority claim; its PushT deltas are +4pp at ceiling.
11. **Expecting GC-IDM to beat CEM on PushT.** `[REFUTED]` CEM's Push-T plateau is 90%, above GC-IDM's
    84.2%, and Push-T at n=50 is the single cell GC-IDM loses. The speed win transfers to our harness;
    the accuracy win does not.
12. **Encoder-side view-invariance retraining.** Already failed for us (latent space contracted). The
    v0.4 lesson stands and nothing in this research pass contradicts it.
13. **A methodological caution on our own keystone numbers.** The probe fits sit at d ≈ n (z8: 2048 vs
    1989 pairs) or d ≫ n (dense: 75264 vs 1989), so in-sample R² is meaningless there (dense in-sample
    R² is exactly 1.000). The held-out re-fit is what we quote (test R² 0.979 / 0.970), and the
    *direction* is robust — an overfit probe would inflate z8's fraction too, and z8 still lands below
    chance. But before this goes in a paper: re-fit on the **train** split episodes and evaluate the
    fraction on val, and bootstrap the enrichment ratios. `[MEASURED-OURS]`

---

## 6. Source ledger

Every URL below was used in verification for this document; verbatim quotes live in the sections above.

**Verified, live for us**
- Hierarchical planning / support-constrained CEM on PushT — https://arxiv.org/html/2607.12547v1
  (corrected version also cited as arXiv 2607.12547v2, "Mind the Gap: Promises and Pitfalls of
  Hierarchical Planning in LeWorldModel")
- Amortised latent planner, frozen world-model embeddings — https://arxiv.org/html/2606.09311v1
- Planner overfitting / deliberately weakened planners — https://ar5iv.labs.arxiv.org/html/1812.01129
- Dynamics-model design for CEM-MPC; ensembles insufficient — https://ar5iv.labs.arxiv.org/html/2109.14311
- Uncertainty-penalty negative + MSE-vs-planning decoupling (TOST) — https://arxiv.org/html/2607.16591
- MOPP: plan-time pessimism, BC proposal + discrepancy pruning — https://ar5iv.labs.arxiv.org/html/2105.07351
- SAGE: subgoal + action proposal, cost frozen — https://arxiv.org/html/2607.17973v1
- PRISM: prior head on frozen JEPA; DINOv2-CLS PushT collapse — https://arxiv.org/html/2606.07974v1
- TAP: on-manifold log-prior inside the planning score — https://ar5iv.labs.arxiv.org/html/2208.10291
- TD-MPC2: policy-prior seeding, warm start, 6 iterations, no latent distance — https://ar5iv.labs.arxiv.org/html/2310.16828
- FOWM: return minus λ·ensemble-std on a compact latent, real xArm — https://ar5iv.labs.arxiv.org/html/2310.16029
- ACID: IDM action-consistency residual, post-hoc on frozen world models — https://arxiv.org/html/2607.02403v1
- GC-IDM: amortized inverse dynamics on frozen latents — https://arxiv.org/html/2605.08732v1
- Adversarial World Modeling (Parthasarathy et al., "Closing the Train-Test Gap in World Models for
  Gradient-Based Planning") — https://arxiv.org/html/2512.09929v1
- AWM released config (ε as std-scaling factors) — https://github.com/qw3rtman/robust-world-model-planning/blob/main/conf/train.yaml
- AWM released training loop (single grad, single sign step, no clean mixing) — https://github.com/qw3rtman/robust-world-model-planning/blob/main/train.py
- DINO-WM (our harness; terminal latent MSE cost; Table 2 encoder column) — https://ar5iv.labs.arxiv.org/html/2411.04983
- LeWM / LeWorldModel (compact latent + terminal L2, SIGReg, end-to-end) — https://arxiv.org/html/2603.19312v1
- Energy-based plan-time regularization on frozen PETS — https://ar5iv.labs.arxiv.org/html/1910.05527

**Referenced from our prior ledger, NOT re-verified in this pass — check before citing `[UNVERIFIED]`**
- TRM (post-hoc reachability cost; <1% rowspace diagnostic) — arXiv 2605.22164
- CompACT (FSQ discrete tokens; RoboMimic Lift 56 vs 56) — arXiv 2603.05438 / OpenReview z9KgtDP6LB
- DCWM / DC-MPC (benefit attributed to stochasticity) — arXiv 2503.00653
- RC-aux (+33pp Wall only; −0.4pp Push-T) — arXiv 2605.07278

**Our own artifacts referenced above `[MEASURED-OURS]`**
- `/home/menlo/brain/ishneet/world-encoder/world_tokenizer/pusht_cost_rowspace.py`
- `/home/menlo/brain/ishneet/world-encoder/results/pusht/cost_rowspace_heldout.json`
- `/home/menlo/brain/ishneet/world-encoder/results/pusht/cost_rowspace_dense.json` (independent re-run)
- `/home/menlo/brain/ishneet/world-encoder/results/pusht/logs/` (all live arm logs)
- `/home/menlo/brain/ishneet/world-encoder/scripts/pusht_read.py` (sticky-mask outcome reader)
- `/home/menlo/brain/ishneet/world-encoder/world_tokenizer/pusht_dynamics_dwm_adv.py` (adversarial arm)
- `/home/menlo/brain/ishneet/dino_wm/planning/cem.py`, `planning/mpc.py`, `planning/objectives.py`,
  `models/kepler_hybrid.py`, `plan_hybrid.py`, `exploit_test.py`
- Action-geometry and Mahalanobis-drift measurements (§2.3) were run inline this session against
  `/home/menlo/dwm_data/pusht_noise/{train,val}/rel_actions.pth` and
  `/mnt/nas/data/robocasa/kepler_ckpt/pusht_dyn_dwm_z8/seed0.pt`; they should be checked in as
  `world_tokenizer/pusht_action_support.py` before they are cited in a paper.
