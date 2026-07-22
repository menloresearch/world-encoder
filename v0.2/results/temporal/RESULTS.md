# Phase 2 (Temporal) — results log

Consolidated temporal-encoder results. Ground-truth numbers + pointers to the saved JSONs here and
the run checkpoints on NAS. Full design (`TEMPORAL_ARCH.md`) + day-by-day debugging narrative
(`TEMPORAL_JOURNAL.md`) are in git history (removed 07-21, `git show 8432258:<file>`). Status one-liner: **temporal-in-encoder retired (NH1 fails to reject, §1–3); v0.2
re-scoped to a per-frame encoder + next-embedding predictor, and Pre-check A (§4) confirms that predictor
direction is worth building** (PLAN.md §Phase 2).

All probes are **vision-only** (the thesis: what vision alone recovers). kuka = cfgs 6,7 (has velocity
+ F/T → the discriminating embodiment). v0.1 reference = `checkpoints/phase1/kuka/seed0.pt`.

## 1. Present-force probe (`diag_present.py`) — does the vision-only latent still carry present F/T?
Compare v0.2 → v0.1 **within a run** (floors differ across runs: win=1 vs win=8 sample different windows).

| run | v0.1 | v0.2 mean | v0.2 max | v0.2 unpooled | raw floor | JSON |
|-----|------|-----------|----------|---------------|-----------|------|
| **C=1** (faithful v01-head port) | 0.253 | **0.251** | 0.240 | 0.248 | 0.103 | `probe_present_kuka_c1.json` |
| **C=8** (temporal, v01 head)     | 0.212 | **0.100** | 0.072 | 0.066 | −0.048 | `probe_present_kuka_c8.json` |

- **C=1**: v0.2 ≈ v0.1 → the §18.12 head fix is a faithful port; force fully recovered (pre-fix was ≈0).
- **C=8**: v0.2 ≈ **half** of v0.1 on the same windows → temporal windowing dilutes present force. NOT
  pooling (max/unpooled no better). See `TEMPORAL_JOURNAL.md` §18.13 (git history).

## 2. NH1 gate (`gate_eval.py`) — dynamics (P3) + future force (P4), C=8 v01-head vs v0.1 vs naive
Source: `gate_kuka_c8_v01.json` (kuka, win 8 str 2, ctx 5, Δ 1/2/3; 5376 train / 2355 test windows).

| metric | v1 (temporal) | v0.1 | naive |
|--------|---------------|------|-------|
| RankMe | 51.3 | **134.0** | — |
| P3 dq (velocity) | −0.023 | **0.037** | — |
| P3 dF/dt | −0.034 | −0.058 | — |
| P4 force @Δ1 | 0.100 | **0.171** | 0.112 |
| P4 force @Δ2 | 0.074 | **0.111** | −0.095 |
| P4 force @Δ3 | 0.047 | **0.058** | −0.164 |

- **P3 dynamics**: temporal at chance (dq negative). No evidence it learned velocity / force-rate.
- **P4 future force**: v1 < v0.1 at every horizon (< naive at Δ1); only beats trivial carry-forward at Δ≥2.
- **RankMe 51 vs 134**: the temporal latent is lower-rank / more collapsed than v0.1's.

## 3. Conclusion (2026-07-16)
**NH1 fails to reject.** The v01-head fix (§18.12) bought *stability* (no collapse; present force recovered
at C=1) but not *capability*: at C=8 the temporal latent is lower-quality (rank 51 vs 134), halves present
force, and loses to single-frame v0.1 on both dynamics and future force. Because near-future force ≈ present
force (autocorrelation), worse present → worse future. Two independent objectives (masked-cell + v01-head)
both fail the gate → suspect = the **temporal fusion** (flat-Perceiver-over-ticks → mean-pool), not the head
or the masking. Redesign motivated by LeWM/RoboTTT: per-frame `z_t` + separate next-embedding predictor
(`TEMPORAL_ARCH.md` §20/§20.1, git history). Caveat: kuka, 1 seed — but margin not borderline (v1 ≤ v0.1 on every cell).

## 4. Pre-check A (`precheck_predict.py`) — is the future LATENT predictable beyond carry-forward? (2026-07-17)
The cheap de-risk for the re-scoped §2.2 predictor (`TEMPORAL_ARCH.md` §21.2, git history): freeze v0.1, encode per-tick
**full multimodal** latents over existing windows, fit a simple `z_t → z_{t+Δ}` predictor, compare to naive
carry-forward (`ẑ=z_t`). Runs on existing data (no precompute). win 8, ctx 4 (input tick 3), stride 2, Δ 1/2/3.

**MSE ratio predictor/naive (<1 = beats carry-forward); linear = Ridge, mlp = early-stopped MLP:**

| embodiment (cfgs) | v0.1 ckpt | n tr/te | RankMe | lin/naive Δ1·2·3 | mlp/naive Δ1·2·3 | JSON |
|---|---|---|---|---|---|---|
| kuka (6,7) | phase1/kuka/seed0 | 5376/2355 | 149 | 0.72 · 0.68 · 0.65 | 0.72 · 0.67 · 0.65 | `precheck_predict_kuka.json` |
| flexiv (1,2) | phase1/flexiv/seed0 | 13696/6041 | 215 | 0.73 · 0.70 · 0.68 | 0.76 · 0.71 · 0.69 | `precheck_predict_flexiv.json` |
| all (1–7) | phase1/all/seed0 | 28544/12256 | 211 | 0.75 · 0.71 · 0.69 | 0.78 · 0.72 · 0.69 | `precheck_predict_all.json` |

**Future force @Δ from the present full latent (fit probe) vs "future force = present force" carry:**

| embodiment | latent R² Δ1·2·3 | carry R² Δ1·2·3 |
|---|---|---|
| kuka | 0.28 · 0.15 · 0.10 | 0.09 · −0.03 · −0.15 |
| flexiv | 0.08 · 0.03 · 0.02 | −0.10 · −0.21 · −0.24 |
| all | 0.22 · 0.15 · 0.12 | 0.05 · −0.06 · −0.14 |

- **GREEN on all three embodiments:** a *simple* predictor cuts latent-prediction MSE ~25–35 % below
  carry-forward at every horizon (positive lin R² 0.10–0.31 while naive R² is ~0/negative → latents move a
  lot across the ~1.7 s ticks, and that motion is learnable). The present latent recovers future force well
  above carrying force forward. → the §2.2 next-embedding predictor is worth building.
- **Reading the caveats honestly:** (1) cache is subsampled (~1.7 s/tick) so Δ1≈1.7 s, Δ3≈5 s — this is
  *long-horizon* predictability; short-horizon (where carry-forward is stronger, less headroom) needs a
  dense re-precompute to test. (2) MLP ≈ linear → the predictable structure is ~**linear** at these horizons,
  so a cheap predictor suffices. (3) the future-force "latent" probe has fit parameters vs the parameter-free
  carry baseline — directionally clear (latent ≫ carry), not a matched comparison. (4) 1 seed / v0.1-seed0.

## 5. Build 2 — trained next-embedding predictor (2026-07-17, `train_predictor.py`)
The real §2.2 artifact (not just the probe): frozen v0.1 → a small predictor `z_t → z_{t+Δ}`, horizon-conditioned,
trained end-to-end. kuka (cfgs 6,7), seed0, best-embodiment-1-seed (full matrix deferred to paper). Metric =
predictor MSE / carry-forward MSE in standardized pooled space (<1 beats carry; comparable to §4 pre-check).

| variant | params | pred/carry MSE Δ1·2·3 | pred R² Δ1·2·3 | RankMe |
|---|---|---|---|---|
| **pooled** | 1.05M | **0.710 · 0.662 · 0.636** | +0.247 · +0.172 · +0.127 | 127 |
| set (8 queries) | 2.44M | 0.724 · 0.674 · 0.651 | +0.230 · +0.156 · +0.104 | 100 |

- **Confirms the direction as a trained artifact:** the predictor beats carry-forward ~28–36% at every horizon,
  reproducing the §4 pre-check (which used a ridge/MLP probe) — so it's not a probe artifact.
- **Design decision — use POOLED, not the set.** pooled ≥ set on every metric (lower MSE ratio, higher R²,
  higher RankMe 127 vs 100) at **half the params**. The "predict the set" guardrail (§21.4) buys nothing here:
  the 8 spatial queries are redundant (the encoder pools them anyway), so predicting them adds params, not
  signal. (This does NOT contradict the temporal-window pooling lesson — that was pooling over *ticks*, this is
  the encoder's own per-frame spatial pooling.)
- **Two bugs found + fixed (transparency):** (1) first run had no standardization + no early stopping → the
  MLP overfit and *lost* to carry-forward (ratio >1) despite low train MSE; fixed with per-dim standardization +
  a val-split early stop → matches the pre-check. (2) two concurrent runs with lance dataloader fork-workers
  **deadlocked** in encoding (0% GPU, no progress) → `--workers 0`.
- Caveats: kuka / 1 seed / coarse cache (long-horizon). JSONs: `predictor_kuka_{pooled,set}.json`;
  checkpoints: `checkpoints/predictor/kuka_{pooled,set}/seed0.pt`.

**N2 (action-conditioning) — first attempt gives NO lift (2026-07-17, `--use-action`).** Conditioned the pooled
predictor on the in-cache "action" = symlog joint velocity (motor ch2, dq) at the input tick, via additive
AdaLN-lite. Result vs the unconditioned baseline (kuka, pooled): **act 0.720/0.670/0.643 vs no-act
0.710/0.662/0.636** (MSE ratio; act marginally *worse*, within noise). JSONs `predictor_kuka_pooled_{act,noact}.json`.
- **Why (honest):** the "action" we have is the *realized* joint velocity, which is **already encoded in z_t**
  (the latent fuses the motor stream) → conditioning re-supplies info the predictor already has. A meaningful
  loss #4 needs a *genuine* action signal — the **commanded** next-motion (not cached) or **DreamDojo-style
  latent actions learned from frame pairs**. So N2 is parked pending a real action source, not "actions don't
  help." (A stronger injection — full AdaLN scale/shift — is a possible follow-up, but redundancy is the likely
  ceiling.)

## 5b. Pre-check B (Build 1 multi-cam) — do views add info? GREEN (2026-07-17, `precheck_multicam.py`)
Before paying for the K-camera precompute: encode each EXTERNAL camera's frame at nearest-matched timestamps
with frozen v0.1 (vision-only latent), compare cross-view vs cross-tick cosine distance. kuka (cfg7), 6 scenes
× 8 cams × ~5 ticks.

| distance | value |
|---|---|
| cross-VIEW (same tick, diff cams) | **0.665** |
| cross-TICK (same cam, diff ticks) | 0.335 |
| raw-ViT cross-view (reference) | 0.147 |
| **ratio view/tick** | **1.98** |

- **GREEN:** two views of the *same instant* are ~2× as far apart in latent space as consecutive time steps →
  views carry **complementary** info, not redundant → multi-cam worth the precompute. The v0.1 latent cross-view
  (0.665) ≫ raw-ViT (0.147) → the encoder is view-*sensitive* (encodes viewpoint content), so fusing views adds
  signal. Caveat: this measures that views *differ* (justifies building), not that the multi-cam encoder will
  *compress* them well (that's the Build-1 experiment). JSON: `precheck_multicam_kuka.json`.

## 5c. Build 1 — multi-cam encoder trained + GATE GREEN (2026-07-19, `train_multicam.py`)
The Build-1 experiment itself: retrain the ~2.2M Perceiver on the K=4-camera kuka cache
(`caches/cfg{6,7}_mc.shard*.npz`, 33,548 chunks, 7-serial cam-id vocab), v0.1 recipe + per-serial camera-id
embedding (`mm_perceiver3.py`), 40 ep single GPU (~30 min). All arms probed on the SAME held-out chunks
(n_test=10,158); v01 = frozen `phase1/kuka/seed0` fed camera slot 0 (the single-view cache's view).

| arm | RankMe | motor R² | ee R² (15-d) | force-only R² | pose-only R² |
|---|---|---|---|---|---|
| **mc4** (ours, 4 views) | **153.5** | **0.400** | **0.544** | **0.283** | **0.717** |
| v01 (single-view v0.1) | 136.2 | 0.351 | 0.482 | 0.251 | 0.635 |
| mc1 (mc model, 1 view) | 155.5 | 0.265 | 0.292 | 0.132 | 0.398 |
| raw4 (pooled ViT, 4 views) | 297.6 | 0.370 | 0.319 | 0.068 | 0.486 |
| raw1 (pooled ViT, 1 view) | 385.0 | 0.390 | 0.353 | 0.100 | 0.521 |

- **GATE GREEN on every read:** (1) no rank collapse under 4 views (153 vs v0.1's 136 — the §17 "compression
  pressure" risk did not materialize at K=4); (2) **force signal held AND improved** (0.283 vs 0.251) — the
  exact probe the temporal design halved (§18.13: 0.101 vs 0.215); (3) extra views genuinely pay (mc4 ≫ mc1
  everywhere; pose +0.08, motor +0.05 over v0.1 — multi-view geometry helps pose most, as expected);
  (4) encoder ≫ raw at matched views (force 0.283 vs 0.068).
- **mc1 caveat:** the mc model fed one view (0.292 ee) is well below v0.1 (0.482) — single-view input is
  out-of-distribution for a model trained always seeing 4. The multi-cam encoder is NOT a drop-in single-view
  replacement; v0.1 stays the 1-cam encoder. (Camera-dropout training is the knob if we ever need one model
  for both.)
- Scope: kuka, seed 0 (dev scoping); full matrix deferred to paper. JSONs:
  `checkpoints/multicam/kuka_mc4/results.json`, `multicam_force_probe.json`.

## 5d. View-composition analysis — is the fused embedding view-invariant? (2026-07-21, `multicam_composition.py`)
JQ's Slack ask (07-21): "we didn't train the multi view camera on single views? Can we calculate some
embedding distance as we increase the number of frames vs different frame?… compare with taking the sum of
individually embedded v0.1 embeddings"; Nicole: "what is the invariance latent density". First, the factual
answer: **correct — no single-view exposure during training** (all 4 cams always present, no camera dropout);
that is exactly why mc1 in §5c is OOD. Then the experiment: 2,000 held-out kuka chunks, cosine distance,
mc4 (`kuka_mc4/seed0`) vs JQ's proposed **late-fusion baseline** = mean of per-view v0.1 embeddings.
JSON: `multicam_composition.json`.

**(1) Convergence — distance to the full-4-view embedding as views are added:**

| views given | mc4 (early fusion) | mean-of-v0.1 (late fusion) |
|---|---|---|
| 1 | 0.476 | 0.261 |
| 2 | 0.294 | 0.115 |
| 3 | 0.140 | 0.043 |

Yes, monotone convergence for both. (Late fusion converges faster *mechanically* — averaging a bigger
subset is trivially closer to averaging all 4; not evidence of quality.)

**(2) Composition sensitivity — same instant, different view-subsets (2-view):**

| | d(same instant, diff camera-pairs) | d(same camera-pair, diff instants) | ratio comp/instant |
|---|---|---|---|
| mc4 | 0.441 | 0.302 | **1.46** |
| mean-of-v0.1 | 0.259 | 0.202 | 1.28 |

**Ratio > 1: different compositions of the SAME moment are further apart than different moments.** The
encoder did NOT learn a view-invariant "canonical scene state" — it aggregates view-specific content. Partly
by design (per-serial camera-id embeddings inject view identity; §5b showed views are complementary, so an
invariant embedding would have to discard that info). This IS Nicole's invariance-density answer: same-instant
compositions are not densely clustered → low invariance. Deployment read: irrelevant for a **fixed rig**
(RoboCasa / Molmobot / microfactory — composition never varies), but for missing-camera robustness
~~camera-dropout training is mandatory~~ — **UPDATE: ran it, it does NOT deliver — see §5e(3)**.

**(3) Late-fusion probe — does averaging v0.1 embeddings substitute for learned fusion? NO:**

| arm | force R² | pose R² | motor R² |
|---|---|---|---|
| mc4 (ours, early fusion) | **0.283** | **0.717** | **0.400** |
| v0.1 single view | 0.251 | 0.635 | 0.351 |
| mean of v0.1 × 4 views (late fusion) | 0.202 | 0.521 | 0.313 |

Late fusion is worse than a *single* v0.1 view on every probe — averaging cancels exactly the view-specific
content that makes multi-view valuable. **Early fusion > single view > late fusion**: fusion has to be
attentional/learned; the Perceiver bottleneck is doing real selection work, not acting like a fancy mean.
(Provenance: rows 1–2 = §5c same-held-out-chunks probes; row 3 = probe run 2026-07-21 ~02:14, numbers
recorded from the run output — the session hit an API rate limit before persisting a JSON for row 3.)

## 5e. JQ follow-ups: MC-encode-per-view+average, per-cam breakdown, camera-dropout retrain (2026-07-21)
The three follow-ups from the §5d Slack thread. #1 ran 10:50 (session died before writeup), #2's retrain
finished 11:27 (probes below ran 16:5x), #3's code landed 11:44 (ran 16:5x). JSONs:
`multicam_avgprobe.json`, `multicam_composition_percam.json`, `multicam_composition_dropout.json`,
NAS `checkpoints/multicam/kuka_mc4_dropout/results.json`.

**(1) MC-encoder-per-view, then average (JQ #1)** — unlike §5d's late fusion this feeds OUR encoder one
view at a time and averages the 4 fused embeddings (§5c results.json probe harness: ee/motor R², same
held-out chunks):

| arm | ee R² | motor R² | RankMe |
|---|---|---|---|
| mc4 full 4-view input | **0.544** | **0.400** | 153 |
| mc-encode-per-view, averaged | 0.425 | 0.346 | 137 |
| best single view (cam3) | 0.478 | 0.325 | 151 |
| worst single view (cam2) | 0.218 | 0.188 | 151 |

Same ordering as §5d: averaging MC single-view embeddings recovers less than the best single view on ee
(0.425 < 0.478) — even with our encoder doing the per-view encoding, the *averaging* step is what loses
the view-specific content. Geometrically the average IS closer to the full embedding than any singleton
(d 0.271 = 0.86× cross-instant, vs 1.22–1.78× for singletons) — close in embedding space, worse on probes:
proximity ≠ information. Early fusion's win is attentional selection, not location.

**OOD caveat resolved (rerun on the dropout ckpt, where single views are in-distribution —
`multicam_avgprobe_dropout.json`):** avg-of-singles is FLAT across ckpts (ee 0.425 → 0.421) and still
below that ckpt's own full fusion (0.461). The §5d/§5e ordering is not an OOD artifact — averaging loses
fusion information, period. (Within the dropout ckpt the avg does beat its best single, 0.421 > 0.378 —
averaging in-distribution embeddings helps locally — but the whole ladder sits below the no-dropout
model: **train-with-all-views + fuse early dominates every dropout variant.**)

**(2) Per-cam breakdown of singleton→full distance (JQ #3, "global vs wrist"):** first the factual answer —
**there is no wrist cam in this set**: `precompute_multicam.py` excludes `IN_HAND_OF_CFG` serials, all 4
cams are external scene views. Among them (normalized by cross-instant distance, pre-dropout ckpt):

| singleton | cam0 | cam1 | cam2 | cam3 |
|---|---|---|---|---|
| d(1-view, full)/d(cross-instant) | 1.35 | 1.59 | 1.78 | **1.22** |

Real spread across viewpoints (cam3 dominates the fusion most, cam2 least — matching their single-view
probe R² ordering: 0.478 vs 0.218), but even the closest singleton sits farther from the full embedding
than a *different moment* does (min 1.22 > 1). No "first view carries most of the world info" — against
JQ's any-view hypothesis, the fusion is genuinely compositional over external views.

**(3) Camera-dropout retrain (JQ #2) — the offered ~30-min knob, now measured: NOT a free fix.**
`kuka_mc4_dropout` = same recipe + per-sample random 1–4 view subset, 40 ep, 35.9 min, seed 0:

| metric | no dropout | dropout | read |
|---|---|---|---|
| mc4 ee / motor R² | **0.544** / **0.400** | 0.461 / 0.356 | full-view quality −0.08 ee |
| mc1 ee / motor R² | 0.292 / 0.265 | 0.327 / 0.271 | single-view +0.035 ee — still ≪ v0.1 (0.482) |
| RankMe (mc4) | 153 | 113 | representation contracts |
| d cross-instant (full) | 0.320 | 0.065 | whole space collapses ~5× |
| d(1-view, full), normalized | 1.49 | 2.04 | *relative* view-invariance got WORSE |
| composition/instant ratio (2-view) | 1.46 | 1.63 | same |

Dropout at this budget shrinks the embedding space globally (cross-instant distances collapse 5×, RankMe
153→113) rather than pulling subsets toward the full-view point: in the model's own scene-discrimination
units, singletons end up *farther* from full. The per-cam rerun (`multicam_avgprobe_dropout.json`) makes
it worse still: within the SAME per-cam harness, dropout drops the best single view 0.478→0.378 (cam3,
the dominant view, takes the biggest hit: −0.12) and the mean over singles 0.343→0.327 — it *equalizes
views downward* (spread narrows 0.218–0.478 → 0.248–0.378); the results.json mc1 "gain" (0.292→0.327) is
the narrowed spread, not lifted information. **Amend the §5d offer before the Slack reply goes out:
dropout-as-recipe'd is not the missing-camera answer — if single-view robustness matters, keep v0.1 per
view; the fused encoder stays a fixed-rig tool.** (Caveats: 1 seed, 40-ep matched budget — a longer/tuned
dropout schedule might trade better; not this week.)

**Implementation verified (07-21 audit, prompted by "did we do the dropout properly"):** (a) end-to-end
identity test on the dropout ckpt — attention-masked dropout (training path) vs physically feeding the
subset (eval path) agree to float noise (max |Δz| ≤ 6e-6 across subset sizes 1/2/3) → no dropped-view
leakage, train/eval consistent; (b) `rand_keep` sampling uniform over sizes 1–4 (.252/.250/.248/.250)
and symmetric across cams (keep rate .625 each); (c) mask sign chain correct (True=BLOCKED → −inf in
SDPA), 196-token groups contiguous per cam, probe scripts subset `cam_ids` with the views; (d) EMA
targets full-view as intended. One architectural note, NOT a bug: the only view-dependent target (vision
EMA) is predicted from the hide-vision pass, so subset contexts are pulled to keep *motor/ee-predictive*
info, never directly toward the full-view embedding — dropout here is the agreed "implicit pressure only"
variant, and the contraction result is that objective's true optimum. The pre-registered escalation
(explicit subset→full consistency loss) remains the correct next lever if invariance is wanted.

## 6. Run inventory (NAS `checkpoints/temporal/`)
- `v01/{kuka_c1_v01,kuka_c8_v01}` — the v01-head fix (§18.12+); **current best**. `gate_kuka_c8_v01.json`.
- `fix/{all,flexiv,ur5,franka,kuka}` (+ seeds) — pre-fix full gate matrix (§18.8, query decoder). `gate.json` per dir.
- `c1/`, `raw/`, `fix2/` — diagnostic ablations (C=1 exoneration §18.11; raw-target divergence §18.12).
- `ur5b/` — first healthy temporal run (§18.7); `smoke/`, `ur5/` — early smoke tests.
- Each run dir has `log_seed*.json` (training curve: loss/inv/sig/RankMe/std per epoch).
