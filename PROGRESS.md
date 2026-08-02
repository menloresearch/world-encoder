# Kepler encoder — the tracker

**The live session tracker** — updated every working session (newest notes in the Update log
at the bottom). Main entry point / whole-arc overview: `SUMMARY.md`. Deep details live in ONE
doc per version folder: `v0.1/EXPERIMENTS.md` (paper results log) · `v0.2/V0.2.md` (v0.2 +
N1 downstream + 07-24 update, as appendices) · `v0.3/V0.3.md` (v0.3 experiments + lit review
+ E6 force build + old tracker, as appendices) · `v0.4/V0.4.md` (encoder design LIVE + build
log appendix) · `v0.4/RESEARCH.md` (ALL research: deep-research NEXT, redesign research,
design reports 01-06).

_Last updated: 2026-08-02. All times in this doc are LOCAL (UTC+8); server
logs, run dirs, and the detail docs are in UTC — subtract 8h when cross-referencing._

## What we're building and why

Kepler = an encoder that compresses robot camera images (+ robot state, later force) into a
small latent — 8 tokens instead of ~600 image patches. For it to matter, at least one of these
four claims must come true. Each has a strong baseline it must beat — beating a weak baseline
counts for nothing.

| # | Claim | Must beat | Where it stands |
|---|---|---|---|
| A | Adding our latent makes a policy better | the same policy without it, same budget | ❌ **NO on Square (wave 1, 07-31): best kepler arm = chassis PARITY (sb 0.86 vs hybrid-base 0.90); v0.4 StateAdd = −36pp; nothing beat the baseline at any epoch** |
| B | Latent alone can replace vision (70× fewer tokens) | the dense-patch policy, at parity | ❌ failed v0.2 (14 vs 32%); **failed AGAIN with v0.4 on Square (0/50 at ep10+ep20, killed)** |
| F | Where force matters, force fused by the encoder beats raw force | vision + raw force wired straight in | ❌ rung 0 NULL on BOTH tasks (square 66/72→54/54; tool_hang ep50 62/64) → rung 1 never built, force out of v0.4 |
| C | The latent is better for prediction / planning | dynamics on raw patches | parked — now the live exit per the program kill rule (see wave-1 verdicts) |

**Kill rule for the whole program** (registered before any v0.4 number exists): if the best
v0.4 setup can't win A, B, or F anywhere, "encoder for reactive control" is dead; what's left
is C and the hardware-efficiency story (matching at 70× fewer tokens is already the ARM win).
*[07-31 post-wave-1 note: that parenthetical is a CONDITIONAL, and wave 1 answered it NO —
zo failed the token-matched bar (0/50 vs calib 32-68 at matched epochs). "Same perf at 70×
fewer tokens" is NOT banked and must not appear in any writeup. The hardware story survives
only as the untested claim-C hypothesis (latent-space prediction/planning), which itself has
a lit headwind: DINO-WM, dense patches beat pooled latents for planning too, 0.90 vs 0.44.]*

**Claim F in detail — the ladder and the registered best-case test.** Step 0 (running tonight,
NO encoder involved): raw force vs no force — does force help this task at all? Square said no
(vision-saturated); tool_hang is the last, best sim candidate. Only on a pass do we build the
real test — the 2×2, with the latent always NEXT TO the policy's own vision (replace is dead),
inserted as read-only zero-init gated tokens (concat is dead):

| Arm | What | Role |
|---|---|---|
| a | vision only | floor |
| b | vision + raw force wired straight in | the STRONG baseline ("just add the sensor") |
| c | vision + our latent, no force in it | isolates "does the latent help at all here" |
| d | vision + our latent with force fused in | the hero arm |

Wins needed: **d > b** (fusion beats raw — the headline nobody in the literature has cleared)
and **d > c** (the edge comes from force). Kicker if (d−c) > (b−a): the encoder extracts MORE
from force than raw wiring does — a true fusion claim. Note "d > c alone" proves nothing (any
added sensor helps); arm b is the control that makes it real. Arm d's encoder = force as
separate contact-gated tokens + future-force prediction aux (published wins say the aux is the
driver); the force-annotated tool_hang data for its pretraining already exists (yesterday's regen).

## The story so far, one paragraph per phase

- **v0.1 (paper done):** encoder trained on real robot data (RH20T). Probes showed the latent
  carries useful signal; force looked like the distinctive asset (marginal wins in the transfer
  matrix). Paper written; ARM collab builds on it.
- **v0.2 (done):** multi-camera + force encoder and a future-predictor, both built and green
  offline. First real downstream test (N1 dry run, RoboCasa): adding the latent to a policy
  gave NO lift. That started the "does it actually help downstream?" investigation.
- **v0.3 (closing now):** fixed the encoder's known weakness (fine detail; better probes across
  the board) — and the policy got WORSE. So we systematically tested every way of plugging the
  latent in (scoreboard below). Big lesson: offline metrics said "better" while rollouts said
  "worse" — probes can't be trusted as success signals.
- **v0.4 (starting today):** the last redesign, JQ's direction — the state-as-query encoder —
  plus a much faster experiment loop (verdicts in hours, not days).

## v0.3 scoreboard — what we ran, what happened, what it taught us

| Test | Result | Lesson |
|---|---|---|
| 1. Latent REPLACES vision | ❌ 32% → 14% | latent alone doesn't carry enough for control |
| 2. Latent NEXT TO vision (concat) | ❌ no change (v0.2 enc); WORSE with better v0.3 enc (16 vs 22, 8 vs 32) | the insertion method is suspect; a better offline latent ≠ a better policy |
| 3. Latent as TRAINING SIGNAL (aux loss) | ❌ more weight = worse: 4 / 10-14 / 26 vs 24-26% baseline | the frozen target's gradient distorts the policy's vision; redesigned version folded into v0.4's objective |
| 4a. FORCE step 0, Square: raw force vs none | ❌ null (66/72 → 54/54) | Square is vision-saturated — can't showcase force |
| 4b. FORCE step 0, ToolHang: raw force vs none | ❌ **NULL at ep50: ft 62 / noft 64** (gap 2pp ≪ 12pp gate). Full curves: noft 2/26/52/60/64, ft 0/0/0/40/62 — ft is the same curve DELAYED ~20 epochs, then rapid catch-up. Killed at ep50 read per protocol, 4:46am | force does NOT enter v0.4's objective set (rung-0 null on BOTH tasks: square 66/72→54/54, tool_hang 62/64). Nuance for the record: not "harmful", just slower to fit the 240-dim window. ⚠ ft slope at kill was +22pp/10ep vs noft +4 — a late crossover can't be ruled out; ckpts ep10-50 on local disk, resumable if Ishneet wants a rung-0 appeal |
| 5. UN-POOL: 8 tokens instead of 1 pooled vector | ❌ ep50 = 10% (inside dense 8-32 band) | pooling was not the (only) mistake |
| 6. INSERTION: pooled z as one gated token | ❌ ep50 = 18% (inside dense 8-32 band) | insertion mode alone doesn't recover baseline either → v0.4 is a RESCUE |
| Baseline: dense patches straight to the policy | 66-72% on Square; 8-32% seed spread on robocasa pnp (killed at ep50) | the recipe is fine — robocasa pnp is a NOISY test bed; screens there need calibrated noise bands |

Encoder-side facts: v0.3 encoder settled (λ=2, 8 queries). The old "raw features beat our
latent" readout was a data-split bug — the latent had the action info all along. Offline
probes are now KILL-ONLY signals: they can prove something's broken, never that it works.

## WAVE 1 — COMPLETE (07-31, one night). Final ep50 board (fast-eval DDPM-16, 1 seed, matched recipe)

| Arm | ep10 / 20 / 30 / 40 / 50 | Verdict |
|---|---|---|
| **hybrid-base** (bare ResNet chassis, no kepler) | .56 / .80 / .84 / .84 / **.90** | the bar. Nothing beat it |
| sb — v0.3 tokens, state-blind | .56 / .72 / .82 / .76 / **.86** | PARITY (−4). Encoder content is NOT the problem |
| rc-pool — v0.3 pooled gated token | .54 / .74 / .82 / .82 / **.82** | PARITY (−8). Pooling isn't the problem either (on Square) |
| sa-h6 — v0.4 StateAdd (JQ's state-as-query) | .28 / .36 / .44 / .48 / **.54** | **HURTS: −36 vs chassis, −32 vs its state-blind twin.** Slow-climbing, never plateaued; granted ep50 read (deviation), killed |
| sa-h16 — StateAdd, longer horizon | .20 / .26 → killed ep20 | strict two-read kill; worse than h6 — horizon doesn't rescue |
| rc-cat — v0.3 concat (E5 retro) | .10 / .38 / .46 / .46 / **.34** | **CONFIRMS the pnp concat kill with mechanism: −56 vs chassis, −48 vs same-encoder-as-tokens.** Insertion mode was the killer |
| zo — z-only (claim B) | .00 / .00 → killed ep33 | trained to sibling-level loss, rolled out 0/50 twice. Latent ≠ control-sufficient |
| calibration (dense-ViT chassis) | .32 / .68 / … ep50 ~1pm | dense chassis < ResNet chassis at matched epochs |

**Program read (Square-scoped: frozen-regime, sim-only, diffusion-BC):** A no, B no, F no.
The registered program kill rule now points at C + the hardware story — BUT the pre-registered
escalation caveat applies: Square is vision-saturated; parity arms clustered at the chassis
ceiling, so Square can answer "does it LIFT?" (no) but not "does it help where vision is
insufficient?" (tool_hang/coffee escalation = Ishneet's call). What Square DID discriminate,
cleanly: insertion-mode harm (concat −48 same-encoder) and state-as-query harm (−32 same-chassis)
are real mechanism effects, both with normal train losses — v0.3's "offline probes lie" lesson
held all night. Deviations to ratify: zo hard-kill, sa-h6/rc-cat granted ep50 reads, sa-h16
strict kill, hybrid-base comparator amendment.

## Running right now (2026-07-31 ~7:30am — winding down)

| GPU | What |
|---|---|
| 0 | calibration to ep50 (~1pm, auto-kill armed) + watcher A finishing its eval backlog on GPU 1 |
| 1-7 | FREE. tool_hang closed (NULL), all wave-1 arms closed. Wave-2 slots await morning decisions |

## Superseded overnight board (2:05am, kept for the record)

**Ishneet gave the overnight go-ahead (~1:45am); gates were registered FIRST (V0.4.md §5.1),
then arms launched. All wave-1 ckpts stream to NAS `dp_checkpoints/w1_*` (symlinked).**

| GPU | Arm | Encoder | Run dir (2026.07.30/) | Notes |
|---|---|---|---|---|
| 0 | calibration (dense baseline) | — | `17.49.40_..._densepatch_square_noft` | ckpt-every-10, seed 42 (= 07-29 66% run) → defines ep10-50 kill band |
| 1 | **sa-h6 (LEAD)** | v04sa h6 λ0.05 | `18.00.04_..._stateadd_square_noft` | + fast-eval watcher A (calib/sa-h6/zo) co-resident |
| 2 | zo (claim B) | v04sa h6 λ0.05 | `18.00.12_..._stateadd_square_noft` | z-only: 8 z-tokens + lowdim, no dense vision |
| 3 | sb (state-blind ctrl) | v03 retro | `18.00.20_..._unpool_kepler_square_noft` | + fast-eval watcher B (sb/rc-pool/rc-cat) |
| 4 | rc-pool (retro) | v03 retro | `18.00.28_..._unpool_kepler_square_noft` | pooled-z gated token |
| 5 | rc-cat (retro) | v03 retro | `18.00.36_..._stateadd_square_noft` | concat re-test |
| 6, 7 | tool_hang F rung-0, ft ep~40 / noft ep~42 | — | `07.50.06_...` | **ep30: noft 52 / ft 0.** ep50 ckpts ~3:30/4:15am → autokill; pair read ~5:30-6:30am → watchers killed, then **sa-h16 auto-launches on GPU 6** (deferred launcher armed) |

Timeline: first wave-1 fast-eval reads (ep10, DDPM-16, ~70s each) ~4:15-4:45am; ep20 ~6:30-7am;
ep50 ~1-2pm (Ishneet awake — kill/promote per §5.1 gates). Results CSV =
`results/downstream/wave1_square_20260731.csv`. Trainers run to ep101 ONLY if promoted;
default = kill at ep50 read. All launch scripts in `diffusion_policy/scripts/w1_*`.

**v0.4 wave-0 pnp pretrains: ALL FOUR DONE + DETECTOR-CLEAN (~7pm).** h∈{6,16} × λ∈{0.05,0.2},
40 epochs each (~50 min wall each, 2 GPUs at a time). Every arm: aux chaseable to the end
(0.014-0.018), gates open and settled (|tanh α| ~0.14 cond, 0 uncond), vision-grad ≫ state-grad
throughout. **D1/D2 shortcut detectors: no flags on any arm** — vision moves the latent ~4×
more than state (0.91-0.93 vs 0.22-0.25), aux runs on vision, D2 conditioned tokens R²
0.18-0.76 (max well under the 0.9 collapse line), unconditioned 0.03-0.10 (designed contrast
intact). Offline probes (kill-only): action-readout z 0.63-0.65 > raw-coarse 0.50 > v0.2 0.57;
**SeeSE3 dse3_d1 ~0.49-0.50 = raw patches (0.49)** — the v0.3-diagnosed short-horizon-geometry
gap is CLOSED offline (v0.3 was 0.414). r2_pos 0.80-0.81 actually beats raw 0.77. Rollouts
still the only "works" signal. Ckpts: NAS `kepler_ckpt/pnp_mc3_v04sa_h{6,16}_l{0.05,0.2}/`.

**Un-pool ep50 = 10%, insertion ep50 = 18% (verdicts ~6:45pm)** — both inside the dense 8-32
band, neither recovers baseline convincingly → **E-axis: v0.4 is a RESCUE** (state-conditioned
extraction = lead hypothesis), un-pool alone was not the fix. Calibration note: ep20 rank
(unpool 10 > insertion 6) INVERTED by ep50 (10 < 18) — pnp early-screens cannot rank; ranking
lives on Square as registered. Trainers killed 6:00pm at ep50 ckpt; old watcher retired.

**New adapter code (uncommitted, like everything since 07-27):**
`world_tokenizer/square_hdf5_cache.py` (robomimic hdf5 → patch npz + actions parquets;
featurization = stage_precompute verbatim; state = 9-D eef+gripper lowdim, K=2 cams) and
`robocasa_pretrain_v03.py` gained `--no-gates` (pnp-only gate battery guarded; default
behavior unchanged).

## v0.4 — the new encoder, in plain words

**The idea (JQ's direction):** let the robot's own state steer what the encoder extracts from
the image. Today the 8 queries look at the scene the same way regardless of where the arm is;
in v0.4 the state is injected into the queries — "look at what matters for where I am right
now." The injection is gated and starts at exactly zero, so at init the v0.4 encoder IS the
current un-pool encoder: it can finetune from the existing checkpoint and can't be worse at
the start.

**Trained by predicting the future:** dedicated predictor tokens must predict the visual scene
a beat ahead, given state + planned actions. The loss touches only those extra tokens — never
the output tokens (putting the loss on the outputs is exactly what killed test 3).

**Guard rails:** state is easy to cheat with — the encoder could echo it back and ignore
vision. A stack of dropout/noise/gates prevents that, and cheap detectors catch it at epoch 10
straight from the checkpoint — a cheating arm dies without a single rollout.

**The new experiment loop (screen cheap, confirm once):**
- 1 seed, gate at epoch 10-20, fast eval (~minutes) → a verdict in ~3-5h, 8-12 verdicts/day.
- Screens only see BIG effects (≥12 points). 5-point arguments are banned by construction.
- Only the final winner gets multi-seed, full-length, full-eval treatment (the big run, weekend).
- Calibration first (today): validate fast eval on robocasa; noise bands from today's
  un-pool/insertion curves; new offline metric = "do the 8 queries look at the gripper + target
  object?" scored against the simulator's free ground-truth masks, calibrated on checkpoints
  whose success we already know.

**Which task screens what (registered):**
- **Square = the precision instrument** — ranks mechanisms. Tight seeds, ~70% success (noise
  proportionally small), 13-min epochs, validated 3-min fast eval, paired eval seeds → ONE seed
  per arm is sound for kill/promote. Caveat: baseline 66-72% leaves ~20pts headroom and Square
  is vision-saturated — if arms cluster at ceiling, that means "task too easy", escalate ranking
  to Coffee/ToolHang (pre-registered interpretation).
- **robocasa pnp = transfer check only** (the original claim lives there; 24-pt seed spread makes
  it useless for ranking). One arm stays there: StateAdd finetuned from the existing un-pool ckpt.
- **ToolHang = force claims only** (headroom + contact-rich; slow, fast-eval invalid there).
  **Coffee = optional second opinion.**
- Square-domain encoder pretraining data: mimicgen square_d0 (1,000 demos) downloaded 07-30
  4:37pm → NAS; runs through the existing E6 regen pipeline (incl. F/T for later force arms).

**Wave 1 arms:** StateAdd (lead) · state-in-K/V (control) · state-blind (control) · z-only
(claim B) · discrete-target variant. Force variants join only if tonight's step 0 passes.
**Plus retro-controls on the clean substrate** (Ishneet 07-30: don't trust narrow-gap pnp
conclusions blindly): a CONCAT-z arm and a POOLED-z arm on Square — the broken v0.3 designs,
re-tested where 1 seed + early epochs are readable. If either surprises (e.g. concat works on
Square), the corresponding pnp kill gets revisited. Big-gap kills (replace 14-vs-32, aux 4%)
are NOT re-run — gaps exceed any noise story and carry built-in controls. Related downgrade:
"v0.3 encoder made concat WORSE than v0.2" is underpowered at pnp noise — cite only "concat
is dead", not encoder regression.

## How sure are we the negatives are real? (audit, 07-30)

Each headline kill carries a built-in control a bug would have broken:
- **Aux loss:** w→0 arm recovers baseline EXACTLY through the same code path (harness proven);
  harm is monotone in weight — bugs make noise, not clean dose-response curves. Eval path is
  stock baseline code.
- **Replace (14 vs 32) :** gap too large for any noise story.
- **Dense baseline:** same code = 66-72% on Square (published DP level) → implementation sound,
  pnp spread is the task.
- **Plumbing:** z-cache validated vs live encoding (cos 1.000000); Square force data checked
  against live-stepped traces before training.
- Track record of catching our own artifacts: split-leak readout (caught + corrected), force
  amplitude compression (caught pre-run), vector-env bug (caught in smoke).

Known soft spots (and the cover for each):
1. Early kill gates assumed ±6-7pt noise; real pnp seed spread is 8-32. Doesn't touch the
   huge-gap kills; makes the concat kill's second round individually noisier than treated —
   the insertion arm running TODAY is the re-test, and calibration replaces assumed bands
   with measured ones.
2. Force nulls carry the FACTR confound: naive policies tend to IGNORE force unless trained to
   use it. Pre-registered interpretation: a rung-0 null = "raw force naively wired doesn't
   help", NOT "force is useless". A pass is definitely real.
3. No git audit trail for 07-27→30 code (needs Ishneet's commit go-ahead).
4. brain-internal#1 stale numbers still awaiting the public correction.

## Standing rules
- Kill at the earliest decisive gate. No confirmation tails. Long runs only for positive signals.
- Gates registered before numbers are seen.
- Strong baselines only (dense patches / raw force), never a strawman.
- Probes and val-loss can say "dead", never "works". Only rollouts say "works".
- Every screen logs token-matched numbers — claim-B / ARM evidence for free.

## Waiting on Ishneet
- tool_hang was launched despite the Square null (deviation from square-first gate) — kill on
  sight if you disagree.
- E6 protocol flags: force window = 2s@20Hz (100Hz infeasible from stored demos);
  reset-and-step force regen (nonstandard, restores live force scale).
- brain-internal#1 correction comment (old readout numbers — never cite them).
- Git: everything since 07-27 is uncommitted — commit checkpoint recommended before v0.4 builds.
  *(07-31: committed+pushed through `93475d6`. Now uncommitted: 08-01 doc updates — V0.4.md
  §7.1-7.2 + this file.)*
- JQ encoder rebuild: build go-ahead (spec = V0.4.md §7.2 + §7.3 fusion flag; thread now
  converged 08-02 — no-flatten accepted, MLA dropped, sum-fusion = screenable config);
  it runs in PARALLEL with the wave-2 pick, not instead of it. Wave-2 pick itself still open;
  GPUs 0-7 free. Two ROLLOUT-FREE options can start immediately and don't conflict:
  (a) shift-robustness eval on existing wave-1 ckpts (deep-research #1, ~0.5 GPU-night);
  (b) INTACT family-metric calibration (08-01 addendum below): predicted-vs-expert action-family
  kNN/CKA from small heads on cached z, correlated against the SR numbers already in our CSVs
  (~a day; pre-registered: it can only rank encoders, not insertion modes — success = orders
  v0.2/v0.3/v0.4sa/raw consistently with best-insertion SR; would be the funnel's first
  non-kill-only screening signal).

## Update log (newest first)
- **08-02 (pt. 6b): second-stage consolidation — ONE doc per version folder.** Merged as
  appendices (content unchanged, each marked with a "formerly `<file>`" banner):
  N1_ROBOCASA + UPDATE_20260724 → v0.2/V0.2.md; LITERATURE_20260728 + E6_BUILD_20260729 +
  PROGRESS_20260729 → v0.3/V0.3.md; V04_BUILD_20260730 → v0.4/V0.4.md; DEEPRESEARCH_NEXT +
  REDESIGN_RESEARCH_20260729 + research/v04/01-06 → NEW v0.4/RESEARCH.md (research/ dir
  removed; the 06 report.json moved to v0.4/). Final md set: root README/SUMMARY/PROGRESS/
  PLAN/DATA + v0.1/EXPERIMENTS.md + v0.2/V0.2.md + v0.3/V0.3.md + v0.4/V0.4.md +
  v0.4/RESEARCH.md. All cross-references repointed (nav docs, PLAN, memory). NOTE: the merge was
  mirrored to the docs branch the same day (82ebbda) — both branches carry the same layout.
- **08-02 (pt. 6): docs consolidated to the docs-branch layout.** Root now holds only
  README.md, SUMMARY.md (main entry), PROGRESS.md (this tracker), PLAN.md, DATA.md.
  Version folders mirror the `docs` branch exactly: v0.1/ (EXPERIMENTS.md), v0.2/ (V0.2.md,
  N1_ROBOCASA.md, UPDATE_20260724), v0.3/ (V0.3.md, E6_BUILD, LITERATURE_20260728,
  PROGRESS_20260729), v0.4/ (V0.4.md, V04_BUILD, REDESIGN_RESEARCH, research/v04 +
  research/next). Relative links in moved files fixed (figures/results/metrics now ../).
  Moves are uncommitted plain renames — git will detect them at commit time.
- **08-02 (pt. 5b, final form): SUMMARY.md — technical narrative version.** Structure:
  problem → what has been run with results → diagnosis (5 numbered mechanism sections:
  wrong consumption model, three harm mechanisms, offline/rollout decoupling, objective-level
  collapse threat w/ Enigma+PMAE, the three live axes) → next experiments table w/ gates →
  rebuild spec → do-not-rerun → pending. No person names, no external-collab mentions, no
  ownership tables; papers cited by arXiv id; all numbers preserved. Earlier drafts (table
  overview, plain-voice layman) superseded in place.
- **08-02 (pt. 5): SUMMARY.md created** — one-page plain-language overview of the whole arc
  (v0.2 → now): claims table with verdicts, phase-by-phase results, why the nulls were
  literature-predicted, the Enigma threat model + §7.4 battery, the settled rebuild spec,
  the ordered next-step plan (5 rows + parallel rebuild, ~5-7 GPU-nights to full verdict),
  do-not-bother list, open decisions. PROGRESS.md stays the live tracker; SUMMARY.md is the
  executive view. Then verified against a FULL re-read of every doc (V0.2/V0.3/V0.4/V04_BUILD/
  N1/E6_BUILD/EXPERIMENTS/PLAN/DATA/LITERATURE/REDESIGN/UPDATE_0724/PROGRESS_0729/README) —
  no contradictions; three folds added: (1) CORRECTION to an earlier in-session statement —
  query count IS ablated at pretrain, twice (v0.2-era q8=q32=q64; v0.3 q8>q32>q64, width
  hurts under the detail objective — only below-8 untested; don't tell JQ "never ablated");
  (2) E7 resampler pre-check recorded as the rebuild resampler's validated precursor;
  (3) N1-N4 noted as extensions of the existing v04_detector.py harness.
- **08-02 (pt. 4): two more reads (single-read, NOT adversarially verified).** (1) **PMAE
  (arXiv 2502.06314, ETH):** pixel-recon losses chase high-variance components (color/
  luminance) while class-relevant signal lives in low-variance components that "barely move
  the loss" — the variance-side twin of the Enigma dominant-channel result. Takes: supplies
  the MECHANISM for our discrete-beats-continuous-targets ranking (continuous cosine weights
  dims by variance → low-variance task content invisible; CE on discrete codes flattens the
  weighting — explains E4's continuous-cosine failure); corroborates latent-over-pixel
  targets (no adoption needed); candidate for the FUTURE objective wave (with INTACT
  action-NLL, NOT the JQ rebuild): whitened/variance-normalized latent-target loss —
  screenable middle ground between continuous-cosine (failed) and discrete-CE (ranked);
  sharpens §7.4 N3 (nuisances = high-variance directions). (2) **Explorative Modeling / XM
  (arXiv 2607.27372, Gladstone/Ji/Du):** best-of-K exploration in training; claims 1-step DP
  inference matching 100-step + 80× fewer planning steps vs Diffuser (Maze2D). Verdict:
  orthogonal — policy-training-method research, our chassis stays stock by design; planning
  speedup is trajectory-diffusion, not latent-MPC (#3 unaffected). Watch-list: 1-step DP (if
  it replicates, fast-eval economics change field-wide) + one more planning-latency datum for
  the ARM narrative. No program changes from either.
- **08-02 (pt. 3): Enigma "obsessed encoder" post read (surfaced by Nicole) — DIRECTLY
  RELEVANT: they collapse DINOv3, LeJEPA AND LeWM (our Build-2 recipe source).** Setup: static
  predictable pattern (faint watermark / colored square / RandGoal poses on PushT) constant
  across views/frames, varying across images/episodes. Result: ALLOCATION collapse, not
  distributional — "a 12-bit feature came to dominate a 1024-dim latent while training looked
  perfectly healthy"; SIGReg stays satisfied ("sheet folded until it looks Gaussian");
  collapsed group's prediction loss drops BELOW clean (loss crossover); probes peak-then-decay;
  LeWM planner at chance on RandGoal. Control (pattern re-randomized per view) tracks baseline
  → predictability, not corruption, is the driver. No remedies proposed; code public; repros in
  ~hours on 1×H100. Mapping to us: (1) loss-crossover = our pre-registered T1 val-loss-inversion
  red flag (NADA 2.94e-2 vs 4.13e-2 at 3%-vs-53%) — external replication of the axis-D threat
  model; (2) diversity terms can't fix it — LeJEPA HAS the diversity term (SIGReg) and
  collapses; + INTACT rank inversion (answers Nicole's "diversity loss"); (3) clustering-family
  regularization also collapsed — DINOv3 runs Sinkhorn+KoLeo (answers JQ's clustering);
  (4) our 07-21 view-specificity (ratio 1.46, camera identity dominant) = same failure family
  observed in our own encoder; (5) published-win countermeasures stay: action grounding
  (MCR/INTACT) + masking the predictable channel (axis-D stack). Rebuild exposure: mm_perceiver3
  objective carried unchanged → camera identity / static background / slow force-state
  components are our stagnant-watermark analogs. PROPOSED battery spec now in V0.4.md §7.4
  (N1-N4 nuisance-dominance probes on cached z, N5 static-pattern canary for the rebuild,
  predictor-beats-carry elevated to the named anti-collapse gate, explicit non-adoptions:
  no diversity term / no clustering-as-anti-collapse) — Ishneet to ratify. Caveat: single
  read of a company post (rigorous-looking: code + controls), claims NOT adversarially
  verified; reproducible if we want to check.
- **08-02 (pt. 2): patch-policy thread (Nicole) — proposed 3 arms (patch-policy baseline /
  +kepler tokens as peer patch tokens / kepler tokens as only vision), stock DP action head.**
  Program mapping: arm 2 = added-tokens family (deep-research Q2: no published win at ANY demo
  count; EO1+Point token-native collapse 73.2→18.6 is the direct analog for a patch-native
  transformer), arm 3 = the zo/claim-B arm (0/50 ×2 on Square at 200 demos; frozen-as-backbone
  −42%). Both fall under do-not-bother at ≥200 demos → recommended reshape: same 3 arms at
  10/20/40/100 demos = registered experiment #2 (low-N crossover) on a second, token-native
  chassis. Token-count question (Nicole "why 8", JQ "could be one"): downstream 1-vs-8 already
  read in wave 1 — rc-pool 0.82 vs sb 0.86, inside band, both chassis parity → count moot for
  insertion; count is live ONLY for claim-C planning (CompACT 8-token vs DINO-WM
  dense-beats-pooled 0.90/0.44) → 1/4/8/16 ablation slots inside experiment #3. JQ clustering
  idea = discretization variant already pre-registered in #3/#4 (FSQ/VQ; do-not-bother #8 — no
  stability claim in CompACT); speech distinction: HuBERT-style units win as prediction
  TARGETS (matches our discrete-beats-continuous-targets finding), no published win as policy
  INPUTS; k-means on cached z is post-hoc + rollout-free (same caches as INTACT calibration).
- **08-02: JQ rebuild thread pt. 2 — no-flatten ACCEPTED ("B,R,3n_s,d dooable"), MLA formally
  dropped by both sides; ONE new idea from JQ: sum the modalities after alignment instead of
  cross-modal attention.** Assessment in V0.4.md §7.3: fixed content-independent mixer with a
  slot-correspondence problem, same family as the falsified 07-21 avg-across-views (but earlier
  in the stack, so screenable rather than dead); resolved as a config flag (`fusion: tokens |
  sum`) on the same build — both variants go through the §3 fast loop, and the tokens arm's
  cross-modal attention maps give the "was attention needed?" answer for free. Hide-one point
  still unsurfaced in-thread (attention mask, not zero-slice; under sum it's trivial —
  omit-from-sum). Wave-2 priority unchanged; rebuild spec now effectively converged.
- **08-01 (evening addendum): read INTACT, arXiv 2607.26056 (Zhejiang/Tsinghua AIR, 28 Jul;
  single read — claims transcribed, NOT adversarially verified).** JEPA world model built on
  LeWM (our Build-2 recipe source; same SIGReg family we use) + ONE shared action-likelihood
  operator fed displacement intents — local z_{t+1}−z_t attached, goal sg(z_g)−z_t stop-grad —
  trained end-to-end. Amortizes planning: direct intent→action at 2.9-5.5ms vs CEM 1.48s
  (~300×); search demoted to an optional 128×3 verifier (+16pts over pure CEM at 23× fewer
  samples). Scope caveats: four simple sim tasks (PushT/Cube/Reacher/TwoRoom), offline experts,
  no real robot, no OOD, and they ADMIT the Gaussian-mean actor can hide multimodality exactly
  at contact transitions — not contact-rich evidence. Three takes for us: **(1) the diagnostic**
  — predicted-vs-expert action-FAMILY kNN/CKA (small trained action head, neighborhood overlap)
  tracks closed-loop SR at r≈0.90-0.95 across 45 ckpts, where pointwise action R² is weaker
  (0.815), effective rank INVERTS (higher rank, lower SR), and intent-cluster purity
  anticorrelates. Their inversion results + "a frozen head cannot recover a controllable
  distinction already collapsed by the encoder" = our probes-lie + zo lessons with external
  corroboration (citable in the writeup). The family-metric is calibratable FOR FREE on our
  already-evaluated ckpts (zero rollouts, ~a day on z/patch caches). Pre-registered expectation
  before anyone runs it: an encoder-side metric can only rank ENCODERS, never insertion modes
  (rc-pool vs rc-cat share one encoder yet score 0.82 vs 0.34) — the test is whether it orders
  our encoder families (v0.2 / v0.3 / v0.4sa / raw) consistently with their best-insertion SR.
  If yes → first non-kill-only screening signal the funnel has. **(2) claim-C/ARM ammo:**
  second 2026 precedent (with CompACT) that compact-latent planning is cheap — and now
  amortizable to search-free; fits the "planning latency" ARM pivot. **(3) objective-axis
  candidate for a FUTURE encoder wave (explicitly NOT the JQ rebuild — §7.2 unchanged):**
  action-NLL on displacement intents inside the encoder objective (INTACT + SMWM + MCR now
  triangulate action grounding; mm_perceiver3's objective has zero action pressure). Honest
  headwind: their matched-E1 rows show frozen-rep + post-hoc heads losing big to end-to-end
  (~65 vs ~95 macro) — consistent with the deep-research frozen-encoder warning; if claim C
  gets built, the action term belongs IN pretraining, not bolted on after.
- **08-01: JQ's encoder-rebuild thread converged — spec settled (V0.4.md §7.1-§7.2), nothing
  launched.** JQ's sketch (strict [B,T,S,P], conv stem over time, spatial-then-temporal
  resample, one code path incl. T=1) survives with two changes: **fusion keeps the tokens
  instead of flattening** (the flatten→linear was one lossy 6144→256 contact point — the same
  worry under all three of his follow-up ideas: MLA, layered cross-attn, dual-path), and
  **hide-one becomes an attention mask** (zeroed slices leak a learnable "hidden" symbol
  through biases). Dropped: MLA (we never have a KV cache — one bidirectional pass at any R),
  the extra-wide fusion trunk, dual-path as a build. Dual-path instead gets a pre-registered
  gate: after the first build, the temporal z must predict future force + short-horizon EE
  motion at least as well as plain per-frame z — the exact gate that killed the v0.2 temporal
  encoder — and only a FAIL earns the second path. Honest downstream read: this rebuild does
  NOT re-open claims A/B (wave 1 already showed encoder content at parity; insertion and task
  saturation were the story) — it's the enabling machinery for F rung-1 (multi-rate force) if
  rung-0 ever passes, and mainly for **claim C** (windowed→compact z is the substrate for
  latent-MPC + the predictor line, i.e. the live exit axis). Wave-2 priority unchanged.
- **07-31 (~1:50pm): WAVE 1 CLOSED + deep-research "what next" landed (recovered from a rate-limit crash).**
  Final ep50 board (DDPM-16 fast eval, 50 rollouts): **hybrid-base 0.90 > sb 0.86 ≈ rc-pool
  0.82 ≫ sa-h6 0.54 > rc-cat 0.34 ≫ zo 0.0**; dense-ViT calib footnote: ep50 0.60 vs its
  0.64–0.66 historical band (same seed/recipe — inside fast-eval noise). Claims A/B/F all NO
  on Square, per the pre-registered gates. The overnight deep-research run (104 agents) lost
  its verification + synthesis to API rate limits at 5:17am; recovered this afternoon — 52
  fresh adversarial votes, **39 claims verified / 0 refuted**, report written:
  `v0.4/RESEARCH.md`. Headlines: (1) our nulls are the literature's
  predicted outcome at ≥160 demos (frozen encoders −42% vs E2E, 3-0); (2) added-tokens has NO
  published win at any demo count, and our concat kill now has an external replication
  (EO1+Point 73.2→18.6 on RLBench); (3) live axes ranked: shift-robustness (eval-only on
  existing wave-1 ckpts!) > low-N crossover (10–100 demos) > claim-C latent-MPC (8-token
  planning is 37–130× cheaper in two 2026 papers; the compressed-vs-dense-patch dynamics
  comparison in manipulation is UNPUBLISHED — we can run it); policy-side token economy for
  edge is dead (4 sources unanimous — decode/LLM dominates, ARM story must pivot to planning
  latency). ⚠ **CORRECTION: Square substrate = 200 demos** (PH `demo_v141.hdf5`, counted:
  200 demos / 30,154 steps) — the "1,000 demos" in the 07-30 9:40pm entry below is wrong
  (carry-over from the robocasa side). Scope all Square conclusions to "200-demo PH".
  Ranked experiment list with kill gates + costs in the report §2; nothing launched yet —
  GPUs 0-7 still free, awaiting Ishneet's pick.
- **07-31 (4:50am): TOOL_HANG RUNG-0 CLOSED — NULL, and the ep30 "harm" reading corrected.**
  ep50 pair: **ft 62 / noft 64** (protocol auto-killed both at the read). ft's full curve
  0/0/0/40/62 = noft's curve delayed ~20ep then converging — slow fit of the 240-dim window,
  NOT harm. Claim F rung-0 now null on BOTH tasks → force stays OUT of v0.4 wave 1 (as the
  ladder requires). ⚠ flagged for Ishneet: ft slope at kill +22pp/10ep vs noft +4 — late
  crossover unfalsified; ckpts resumable if he wants an appeal run. **hybrid-base ep10 = 0.56 —
  EQUAL to sb 0.56 / rc-pool 0.54** → v0.3 token arms add nothing over the bare chassis at
  ep10, and sa-h6 (0.28) is 28pp BELOW the no-kepler baseline = kill-read #1 vs the registered
  A-comparator (rc-cat likewise below). hybrid-base ep20 (~5:20am) completes or breaks the kill
  pairs for sa-h6 + rc-cat. **sa-h16 auto-launched GPU 6 4:46am** (encoder strict-load OK, own
  watcher). GPU 7 idle BY CHOICE — second-seed/wave-2 slots are morning decisions.
- **07-31 (4:15am): ep10 round complete, first kill executed, comparator accelerated.**
  Calib ep10 = 0.32 (dense chassis, fast-eval). Board vs it: sb 0.56 (+24) ≈ rc-pool 0.54 (+22,
  and **0.74 at ep20** — above the dense ep50 ceiling already) ≫ sa-h6 0.28 (−4, IN BAND,
  continues; its 28pp gap to sb is chassis-internal: gates all ~|0.05|, train loss 0.099 vs
  0.067 → state-conditioned tokens genuinely slow the early fit, no shortcut/bug signature) ≫
  rc-cat 0.10→0.38@ep20 (recovering, no kill — concat = slow start, not instant death, on this
  chassis) ≫ **zo 0.0/0.0 @ep10+ep20 → KILLED at ep33 per the §5.1 hard-kill clause** (trained
  clean to loss 0.061 = sibling level, rolled out 0/50 twice → the v0.3 thesis verbatim: enough
  for imitation, not for control; worse than v0.2's 14% replace arm). **Claim B: second, cleaner
  failure.** GPU 2 immediately repurposed: **hybrid-base launched there ~4:10am** (2h earlier
  than planned; GPU-7 waiter cancelled) — run `20.10.06_..._hybrid_square_noft`, own watcher,
  ckpts → NAS `w1_hybrid_base`. tool_hang: noft trainer auto-killed at ep50 ckpt 3:50am (ep40 =
  0.60); ep50 DDPM-100 eval in flight; ft ep50 kill imminent.
- **07-31 (2:50am): first reads + comparator fix.** zo ep10 = 0.0 (B-arm, judged vs dense calib
  when its ep10 lands ~4am). **sb ep10 = 0.56** — flagged that the unpool-family chassis (ResNet
  hybrid) is a much stronger Square learner than dense-patch; judging A-arms vs the dense curve
  would conflate chassis with encoder. **§5.1 amendment 1 registered BEFORE any same-chassis arm
  read: new `hybrid-base` stock-DP arm (same recipe/dims, GPU 7 after autokill) = the claim-A
  comparator; dense calib stays as zo's B-comparator + kill floor. sa-h6 vs sb now isolates the
  state-as-query delta.** tool_hang noft ep40 = 0.60 (ft 0.0 through ep30) — rung-0 NO firming up.
- **07-31 (2:05am, night watch): WAVE 1 LAUNCHED — Ishneet's go-ahead, gates registered BEFORE
  any number (V0.4.md §5.1 freeze draft, he ratifies in the morning).** Build: new
  `KeplerStateAddEncoder` (strict full-state_dict load via the v04_detector recipe — the OLD
  loader silently drops the state path; verified tanh(α) matches pretrain exactly) +
  `StateAddKeplerPolicy` (insert_mode unpool / z_only / concat; un-normalizes DP's range-scaled
  lowdim back to raw before the encoder's own μ/σ), config `train_stateadd_bs192.yaml`,
  `w1_fasteval_watcher.py` (eval_fast DDPM-16 shelled to e6 venv). ALL FIVE ARM SMOKES PASS;
  eval path validated end-to-end on a real StateAdd ckpt in the e6 venv (50 rollouts, 71s wall).
  Board: calib GPU0 (launched 1:50am, ckpt-every-10, same seed as the 66% run) + sa-h6/zo/sb/
  rc-pool/rc-cat GPUs 1-5 (launched 2:00am) + 2 fast-eval watchers; sa-h16 auto-launches on
  GPU 6 when the tool_hang autokill frees it. Crash/eval/kill monitor armed in-session.
  CPU budget: arms at 8 workers (72 cores, tool_hang still holds 24 until ~3:30am).
- **07-31 (1:45am, night watch — Ishneet asleep):** Everything from the evening plan verified
  running/complete. (1) **Square pretrain wave: all five DONE** 10:15-10:16pm (~35 min each, faster
  than the 50-70 min estimate), ckpts + results.json on NAS; training monitors healthy end-to-end
  (aux chased 0.056→0.004, conditioned-query α alive q0-3 / exactly 0 q4-7 by design, proj_v grad ≫
  s_mlp throughout). (2) **Post-run detectors on all four Square StateAdd ckpts: NO SHORTCUT FLAGS**
  — dz_state 0.17-0.26 vs dz_vision 0.95-0.97, aux runs on vision, D2 conditioned R² max 0.41 ≪ 0.9
  collapse line. (3) **Deep-research 06 (encoder-swap protocols) landed complete** 10:22pm — 19
  sources, 24/25 claims survived 3-vote adversarial verification, ends with the audit table vs our
  protocol. (4) **tool_hang ep30 pair read: noft 52% / ft 0%** (ft = 0.0 at ep10/20/30). Checked for
  artifacts: ft train loss normal and ≈ noft (0.051 vs 0.055), eval runs in the patched e6 venv, no
  force-wiring errors in eval logs → reads as a REAL rollout failure; rung 0 heading to a decisive
  NO, with the stronger reading that naively-wired raw force is actively harmful on tool_hang (cf.
  FACTR confound note — pre-registered interpretation stands: "raw force naively wired doesn't
  help", not "force is useless"). (5) **ep50 auto-kill ARMED** implementing the pre-registered V0.4
  §4 rule ("kill both arms at the ep50 pair read unless ft−noft ≥ ~12pp"): detached script kills
  each trainer at its ep50 ckpt, kills watchers once both ep50 rows are in the CSV, aborts and
  leaves everything running if ft shows a ≥12pp win at the latest common epoch. (6) GPUs 0-5 left
  FREE deliberately — wave-1 screens wait on design freeze per V0.4 §5 (gates get filled before any
  v0.4 number is seen). Housekeeping: root disk at 94% (34G free) — fine through the ep50 kills
  (ckpts are 1.6G), another argument against ep100 tails. Still waiting on Ishneet: git commit
  go-ahead for 07-27→31 code, brain-internal#1 correction, E6 protocol flags.
- **07-30 (9:40pm):** Square substrate ONLINE — cache adapter finished (31,275 rows / 1,000
  demos → NAS), five Square encoder pretrains launched on GPUs 0-3+5 (v0.4 StateAdd 2×2 +
  v0.3-style retro-control, ~50-70 min each). tool_hang ep20 early-call read arriving: noft
  26%, ft eval in flight. NOTE: the ep20 early-call rule was applied at ep10 but its exact
  text is nowhere in the docs — decisive gate remains the registered ep50 ft−noft ≥ ~12pp;
  ep20 gets recorded, not acted on, unless the pair is unambiguous. Deep-research sweep on
  encoder-swap evaluation protocols (R3M/MVP/VC-1/CortexBench etc.) running in parallel.
- **07-30 (6pm):** v0.4 wave-0 LAUNCHED — all four pre-registered encoder pretrains
  (h∈{6,16} × λ∈{0.05,0.2}) running on GPUs 0/1/2/5. Early logs healthy: aux monotone
  (chaseable), gates opening from zero, vision-grad ≫ state-grad (no GAP signature).
  Un-pool/insertion trainers killed at ep50 ckpt per plan; evals in flight. Conclusions
  audit (Ishneet-reviewed): big-gap kills stand (aux dose-response, replace); narrow-gap
  pnp conclusions (concat family) get Square retro-controls; nothing else re-run.
- **07-30 (pm):** tool_hang pair restarted at ~zero cost with ckpt-every-10 + pre-registered
  early-call rule — first possible force verdict moves from ~2:30am to ~9pm.
- **07-30:** Aux loss (test 3) closed dead — dose-response monotone. E2 baseline killed at the
  ep50-spread verdict (8-32%), no ep100 wait. Square force null; tool_hang step-0 launched.
  Un-pool/insertion gates alive (evals 2→10 / 6→6 by ep20). v0.4 research landed (5 reports);
  design freezing today; claims ledger added (V0.4.md §0.5). Fast eval validated on Square
  (64 vs 66% at ~20× speed). FAST-PIVOT directive: no more ep100-style waits, screen-cheap loop
  is the operating mode. This tracker created (replaces dated PROGRESS files).
- **07-29:** E6 force test built end-to-end in a day (own venv, force-sensor patch, data regen
  with live-scale fix). Square force pair launched. E2 relaunched at 4 seeds. Redesign research
  verified (110 claims, 21 papers). Un-pool + insertion arms built and smoked.
- **07-28:** Aux-loss sweep run and killed (E4). Hybrid rerun killed (E5, no lift). Dense-patch
  baseline built + launched (E2). Literature review (25 sources): no published win for
  compressed latents in reactive control; force bar (beat raw) unbeaten by anyone.
- **≤07-27:** v0.3 encoder settled (λ=2, 8 queries). Old action-readout flip exposed as a
  data-split artifact. Early-kill discipline set (Ishneet): kill at the first decisive gate.
