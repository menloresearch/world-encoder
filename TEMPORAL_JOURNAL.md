# Kepler Temporal (Phase 2) — debugging journal

Day-by-day diagnostic narrative for the temporal encoder, extracted from `TEMPORAL_ARCH.md` §18.7–18.14
(2026-07-17) to keep the design doc lean. This is the lab-notebook record of *how* we found the NH1 gate
failure and its cause — kept verbatim for provenance.

- **Design spec + eval protocol:** `TEMPORAL_ARCH.md` (§18.1–18.6 = the hypotheses/protocol).
- **Results + saved JSONs:** `results/temporal/RESULTS.md`.
- **Bottom line:** NH1 fails to reject; the mean-pooled temporal fusion degrades the latent; a redesign
  (per-frame `z_t` + next-embedding predictor) is motivated by LeWM/RoboTTT (`TEMPORAL_ARCH.md` §20/§20.1).

---

### 18.7 First gate result — ur5 (2026-07-16; ONE seed, coarse cache)
`gate_eval.py`: v1 `temporal/ur5b/seed0` vs v0.1 `phase1/ur5/seed0`, cfgs 3+4, window 8, ctx 5,
Δ∈{1,2,3} (~1.7 s/tick).

| test | v1 | v0.1 | naive |
|---|---|---|---|
| P3 dq (velocity) | N/A | N/A | — |
| P3 dF/dt | −0.02 | −0.06 | — |
| P4 force @Δ1 | −0.00 | −0.06 | −0.35 |
| P4 force @Δ2 | −0.02 | −0.09 | −0.39 |
| P4 force @Δ3 | −0.02 | −0.07 | −0.47 |

RankMe: v1 29, v0.1 173.

**Read:** v1 **consistently ≥ v0.1 ≥ naive on every cell** (correct ordering), but **all absolute
R² are ~0/negative** — neither vision-only latent predicts *future* force / force-rate above chance.
⇒ **directional temporal advantage, no absolute future-prediction signal.** Does **not** clearly
reject NH1.

**Caveats / suspects:**
1. **dq (the cleanest test) didn't run** — ur5's joint vector is 6-dim ⇒ `has_vel=False` ⇒ **no
   velocity in ur5's data**. dq is testable only on has-velocity embodiments (flexiv/kuka — verify).
2. **Coarse subsampled cache** (~1.7 s frames, ee = snippets) — future force at 1.7–5 s from
   vision alone is near-impossible and the cache can't give fine force dynamics. Strongest evidence
   yet that the **dense re-precompute** (continuous ee, native-rate frames) may be *required* to
   fairly test temporal.

**Next:** run dq on flexiv/kuka (fleet producing them); multi-seed error bars; and seriously weigh
the dense re-precompute before concluding anything about NH1.

### 18.8 Full gate matrix (2026-07-16) — NH1 FAILS to reject; v0.2 ≤ v0.1
Fleet trained all embodiments (`temporal/fix/`, 40 ep, all healthy: inv bounded, RankMe 20–48, no
collapse). Gate = v0.2 `fix/<emb>/seed0` vs v0.1 `phase1/<emb>/seed0`, vision-only.

| emb | dq: v0.2 / v0.1 | dF/dt: v0.2 / v0.1 | future force Δ1: v0.2 / v0.1 / naive |
|---|---|---|---|
| ur5 | N/A (no vel) | −0.02 / −0.06 | −0.00 / −0.06 / −0.35 |
| franka | −0.01 / −0.09 | N/A (cfg5 no F/T) | N/A |
| flexiv | −0.01 / −0.01 | −0.01 / −0.02 | −0.01 / **+0.02** / −0.12 |
| kuka | −0.00 / **+0.04** | −0.01 / −0.06 | −0.00 / **+0.17** / +0.11 |

RankMe (z_v): v0.2 ~25–35 vs v0.1 ~130–190.

**Conclusion: FAIL to reject NH1 — v0.2 ≤ v0.1 on every discriminating cell.** v0.2's vision-only
latent is at chance (~0) on dynamics + future force everywhere; v0.1 has real signal (kuka
future-force +0.17, dq +0.04; flexiv +0.02). Temporal-as-built does not help and on kuka is worse.

**Critical:** this is **NOT (just) the coarse cache** — v0.1 finds signal on the *same data*, so the
temporal **architecture/objective/eval** is implicated, not the data. Prime suspects: (1) window
mean-pooling **dilutes** the sharp per-frame force cue; (2) the masked-cell objective shapes z_v
less directly for "vision-only → state" than v0.1's objective; (3) v0.2 z_v is much lower rank.

**Next (diagnose before big rebuilds):**
1. Probe **present** force v0.2 vs v0.1 (confirm dilution — the key diagnostic).
2. Eval variant: per-frame / un-pooled readout instead of window mean-pool.
3. Objective: add a v0.1-style direct "vision-only → pooled state" term; and/or the SIGReg-placement
   ablation (§7.3, Dhanoosh).
4. Dense re-precompute is now **secondary** (same-data v0.1 win points at the model, not the data).

### 18.9 Diagnostic — it's the MASKING/objective, not pooling (2026-07-16)
Present-force probe (vision-only) on kuka:

| readout | R² |
|---|---|
| v0.1 (single frame) | **0.213** |
| v0.2 mean-pool (current eval) | −0.01 |
| v0.2 max-pool | −0.08 |
| v0.2 unpooled (PCA-512) | −0.16 |
| raw ViT (floor) | −0.05 |

**Pooling EXONERATED** — un-pooling is *worse*, not better. v0.2's vision-only latent carries **no
present force** (raw-ViT level, any readout); v0.1's carries it (0.21). So v0.2 failed to learn the
cross-modal signal → **the objective is the cause, not the eval.**

**Root cause:** v0.1 hides an **entire modality** → vision is forced to encode force. v0.2 masks a
**random 50% of `(modality × time)` cells**, so a masked ee cell is predictable from **visible ee at
neighboring ticks** (force is temporally autocorrelated) — an easy same-modality shortcut needing no
vision→force learning. Cross-modal pressure got diluted away.

**Fix (cheap, targeted):** add **whole-modality masking** (hide *all* of a modality across the whole
window, v0.1-style) so predicting it *requires* the other modalities — restores cross-modal pressure
while keeping the temporal axis. Masking change + ~15-min retrain, then re-probe present force
(expect it to jump toward ~0.2 if confirmed). Orthogonal to Dhanoosh's SIGReg (rank/shape).

### 18.10 The masking fix FAILED — narrowing the cause (2026-07-16)
Retrained kuka `mask_mode=modality` (hide all C ticks of one modality → forces cross-modal
prediction) → `temporal/fix2/kuka`. Re-probed present force:

| readout | R² (fix2, modality-mask) | R² (before, cell-mask) |
|---|---|---|
| v0.1 (single frame) | **0.213** | 0.213 |
| v0.2 mean-pool | −0.010 | −0.01 |
| v0.2 max-pool | −0.056 | −0.08 |
| v0.2 unpooled (PCA-512) | −0.118 | −0.16 |
| raw ViT (floor) | −0.049 | −0.05 |

**Essentially identical → the §18.9 masking diagnosis was WRONG.** Whole-modality masking (exactly
v0.1's cross-modal pressure) did *not* restore the force signal. Also the run underfit: `inv` stuck
~1.6 (fleet reached 0.5–0.97), `z_v RankMe` ~10 (fleet 20–48) — the harder task barely trained, but
the well-fit cell/mixed fleet *also* gave ~0 force, so underfitting isn't the explanation either.

**Ruled out — time-embed dominance** (`diag_norms.py`): on fix2/kuka, `time(t)` L2 norm 32.7 vs
`proj_v(rgb)` 45.1 (**ratio 0.72**) — comparable but smaller, and a per-frame *constant* shift that
preserves within-frame patch differences (std 46.9, where force cues live). Not swamping vision.

**Remaining suspect — the temporal architecture itself.** The two things v0.2 adds over v0.1 that
survive at C>1: (a) 64 latents summarize C·196 vision tokens (8 frames) instead of one frame's 196
→ the sharp per-frame force cue is diluted across the window belief; (b) the query-predictor can
reconstruct each tick's force from *visible motor* + a time-tagged query without vision contributing.

**Definitive isolation — the C=1 ablation (running):** retrain v0.2 with `window=1` (`temporal/c1`).
At C=1 time collapses to a constant and the 8-frame dilution disappears, reducing v0.2's architecture
(64 latents + query-predictor + single fuse + L2-normalized target) to essentially v0.1's single-tick
setting. **Read:** if C=1 recovers force (→~0.2) the cause is *temporal windowing* (fix = per-tick
readout / fewer latents-per-tick / hierarchical, not flat-64-over-8); if C=1 *still* fails at ~0, the
cause is an architecture/objective detail independent of time (bisect: 64-vs-8 latents, query-decoder
vs per-modality MLP head, L2-normalized vs raw target) and flat-temporal is exonerated as the culprit.

### 18.11 C=1 result — TEMPORAL EXONERATED; it's the predictor/objective (2026-07-16)
Present force, probed at `window=1 stride=1` (bigger single-frame set → raw-ViT floor now +0.103; both
models on the identical set):

| readout | C=1, 64 latents | C=1, 8 latents |
|---|---|---|
| **v0.1** (single frame) | **0.253** | **0.253** |
| v0.2 mean-pool | 0.042 | 0.034 |
| v0.2 unpooled (PCA) | 0.026 | 0.020 |
| **raw ViT (floor)** | **0.103** | **0.103** |

**C=1 does NOT recover force** (0.03–0.04, same as C=8). So the cause is **NOT** temporal windowing,
**NOT** latent width (8 ≈ 64), **NOT** pooling (unpooled fails), **NOT** time-embed (measured, §18.10).
Worse: **v0.2 sits *below* the raw-ViT floor (0.02–0.04 vs 0.103)** — training *destroys* force signal
that's already in the frozen features, while v0.1 *enhances* it (0.25 ≫ 0.103).

**The tell — the invariance loss isn't learning.** inv uses L2-normalized pred+target, so
`inv = 2 − 2·cos θ`. The modality-masked runs sit at inv ≈ 1.6–1.9 → **cos θ ≈ 0.05**, i.e. the
cross-modal prediction is nearly *orthogonal* to the target — the map barely trains. (The fleet's
"good" inv 0.5–0.97 came from *cell* masking's copy-from-neighbor shortcut, which is why it trained
but carried no force.) With the invariance gradient near-noise, SIGReg whitens the vision marginal
*without* cross-modal grounding → force washed out below raw.

**Suspect = v0.2's predictor/objective redesign vs v0.1's proven head:** (a) **L2-normalized cosine
target** (v0.1 = raw MSE); (b) **query cross-attention decoder** (v0.1 = per-modality MLP on the
*pooled* latent); (c) **dropped joint-SIGReg** (v0.1 regularized the fused z too). Top suspect (a):
normalizing a linearly-projected force target onto the unit sphere collapses the separation the head
needs. **Test running:** `--raw-target` flag (unnormalized MSE, out_norm kept so no scale drift) at
C=1 *and* C=8 (`temporal/raw/`). If raw-target C=1 recovers toward ~0.25 → normalization was the
killer, and the temporal (C=8) model just needs the v0.1-style head bolted back on. This also lands
Dhanoosh's point in its correct place: SIGReg isn't the disease (RankMe ~8–10 = it's being
*overpowered*, not over-regularizing); the invariance objective failing to learn is.

### 18.12 The vise + the fix: port v0.1's proven head (2026-07-16)
`--raw-target` alone (query decoder, `temporal/raw/`) **diverged**: mean inv 55→122→142 over 3 epochs,
std collapsing to ~0.02 — even *with* grad-clip 1.0 (added to the trainer). So the query-decoder
predictor is in a **vise**:
- **normalized** cosine target → cross-modal map won't train (cos θ ≈ 0.05, §18.11);
- **raw** target → the fused latent collapses and inv diverges (the original scale-drift, now with no
  normalization to hold it).

v0.1 tolerates the raw target because it has **joint SIGReg on the fused latent** (which v0.2 dropped) —
that's what stops the fused representation from collapsing. So the fix isn't a knob, it's to **restore
v0.1's whole head** on top of the temporal fuse:

**`pred_mode="v01"`** (`mm_perceiver_temporal.py::_forward_v01`): three hide-one-*modality* passes
(hide v / m / e across all ticks), **per-modality MLP on the mean-pooled fused latent** predicting each
modality's **time-pooled raw EMA target**, `--joint-sigreg` on the fused latent for stability. The
temporal fusion is kept (latents still cross-attend all C ticks with time embeddings); only the head
reverts to the proven recipe. At C=1 this is *exactly* v0.1 → sanity bound ~0.25.

Smoke (C=8, v01+joint+raw): inv **drops smoothly 0.49→0.29** in 5 steps, no divergence — the vise is
broken. **Running:** `temporal/v01/` C=1 (expect ~0.25, confirms port) + C=8 (the real temporal fix;
success = present-force ≫ raw-ViT floor 0.10, ideally ≈ v0.1's 0.25). New trainer flags:
`--pred-mode {query,v01}`, `--joint-sigreg`, `--raw-target`, `--grad-clip`.

### 18.13 v01-head confirmation — fix works at C=1, temporal still HALVES present-force at C=8 (2026-07-16)
Both `temporal/v01/` runs trained healthy (kuka, cfgs 6 7, `pred_mode=v01 --joint-sigreg --raw-target`,
d=256 n_lat=64 n_self=4, 40 ep): **C=1** (win=1 str=1) RankMe 35, std 0.44, inv stable ~1.8; **C=8**
(win=8 str=2) RankMe 46, std 0.75, inv ~0.84 — **no collapse** (vs the query decoder's collapsed
8–10 / 0.02). Present-force probe (`diag_present.py`, vision-only, R²; compare v0.2→v0.1 **WITHIN** a run
— floors differ across runs because win=1 vs win=8 sample different windows):

| run | v0.1 | v0.2 mean | max | unpooled | raw floor |
|-----|------|-----------|-----|----------|-----------|
| C=1 (faithful port) | 0.253 | **0.251** | 0.240 | 0.248 | 0.103 |
| C=8 (temporal)      | 0.215 | **0.101** | 0.075 | 0.071 | −0.041 |

**Verdict.** (1) **§18.12 fix CONFIRMED** — at C=1 the ported head recovers force *exactly* (0.251 ≈ v0.1
0.253; pre-fix ≈0). The head/objective was the disease; v0.1's head + joint-SIGReg cures it. (2)
**Temporal windowing STILL halves present-force** — C=1→C=8 with the head **held constant** drops v0.2
0.251→0.101 (= 47% of v0.1's 0.215 on the *same* windows). **NOT pooling** — max/unpooled are no better
(in fact worse) → the latents *entangle time*; the sharp present-tick force cue dilutes across 8 ticks in
the fixed-capacity latent (64×256).

**Should present-force stay flat C=1→C=8? YES, roughly.** The C=8 window is a strict **superset** of the
C=1 input (same last frame + 7 history), so present force is fully available; a good temporal encoder
should *preserve* present-state readout and *add* dynamics. Halving it = the model spends its bounded
latent budget on the window at the expense of the present = a **defect to fix, not an inherent cost**.
For a world-**state** encoder, present-state fidelity is core → temporal must be a *superset* of v0.1,
not a partial replacement. A small dilution (0.25→0.22) is fine; halving is not.

**Fix directions (architectural, separate from the objective):** (i) reserve capacity for the present —
a dedicated "present" latent/query or per-tick allocation; (ii) anchor the present in the objective — a
v0.1-style direct vision→present-state term at the last tick; (iii) more latents (cheap capacity test).

**Decision:** stable enough to **run the gate**. Next = P3 dynamics + P4 future (`gate_eval.py`) on
`temporal/v01/kuka_c8_v01` vs `phase1/kuka` vs naive — kuka **has velocity**, so `dq` (the cleanest P3
discriminator, N/A on ur5) runs here. The present-force fix is iteration-2, gated on whether temporal
even earns its keep.

### 18.14 NH1 GATE on the v01-head fix — STILL FAILS TO REJECT (kuka, 2026-07-16)
Ran `gate_eval.py` on `temporal/v01/kuka_c8_v01` vs `phase1/kuka` vs naive (cfgs 6 7, win 8 str 2, ctx 5,
Δ 1/2/3; 5376/2355 windows), vision-only:

| metric | v1 (temporal) | v0.1 | naive |
|--------|---------------|------|-------|
| RankMe | 51.3 | **134.0** | — |
| P3 dq (velocity) | −0.023 | **0.037** | — |
| P3 dF/dt | −0.034 | −0.058 | — |
| P4 force @Δ1 | 0.100 | **0.171** | 0.112 |
| P4 force @Δ2 | 0.074 | **0.111** | −0.095 |
| P4 force @Δ3 | 0.047 | **0.058** | −0.164 |

**NH1 FAILS TO REJECT — the v01-head fix bought STABILITY, not CAPABILITY.**
- **P3 dynamics: temporal at chance.** dq v1 −0.02 (≤0, below v0.1's near-zero 0.04); dF/dt both ≈0. No
  evidence the temporal latent learned velocity or force-rate.
- **P4 future force: v1 < v0.1 at EVERY horizon** (and < naive at Δ1). Temporal only beats the *trivial*
  carry-forward at Δ≥2 (v1 stays +0.05..0.07 while naive collapses to −0.10..−0.16) — a faint "context
  helps vs future=present," but a low bar, and v0.1 wins outright everywhere.

**Root read = the temporal latent is LOWER QUALITY, not just present-diluted.** RankMe 51 vs 134 — v0.2's
64 latents are more redundant/collapsed than v0.1's 8 queries. Since near-future force is dominated by
present force (autocorrelation), a model that represents present force worse (§18.13: 0.10 vs 0.215) also
predicts *future* force worse — exactly the P4 pattern. This **reproduces the §18.8 negative gate on the
SAME data**: the head fix moved inv/RankMe stability but NOT the gate outcome.

**Implication — the suspect is now the temporal FUSION, not the head or the masking.** Two independent
objectives (masked-cell §18.8; v01-head-on-window §18.12-14) have both failed the gate, while the head
and masking were each ruled out (§18.9-11). What's common to both is **flat-Perceiver-over-all-ticks →
mean-pool → head** — that pooled fuse degrades the latent (rank 51 vs 134). This is exactly the redesign
§20 (LeWM) motivates: **stop mean-pooling the window** — keep a clean per-frame `z_t` + a separate
predictor with **next-embedding prediction** as the temporal term, on top of the v0.1 cross-modal head +
joint-SIGReg.

**Caveat:** kuka, one seed — but it matches the all-embodiment matrix (§18.8) and the margin is not
borderline (v1 ≤ v0.1 on every cell), so more seeds won't flip it. Gate did its job: caught the
regression cheaply, pre-scale. JSON: `temporal/v01/gate_kuka_c8_v01.json`.

---

