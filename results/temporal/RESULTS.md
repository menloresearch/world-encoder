# Phase 2 (Temporal) — results log

Consolidated temporal-encoder results. Ground-truth numbers + pointers to the saved JSONs here and
the run checkpoints on NAS. Full design in `TEMPORAL_ARCH.md`; the day-by-day debugging narrative in
`TEMPORAL_JOURNAL.md`. Status one-liner: **NH1 fails to reject — temporal-as-built does not beat the
single-timestep v0.1; redesign pending** (see Conclusion).

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
  pooling (max/unpooled no better). See `TEMPORAL_JOURNAL.md` §18.13.

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
(TEMPORAL_ARCH §20/§20.1). Caveat: kuka, 1 seed — but margin not borderline (v1 ≤ v0.1 on every cell).

## 4. Run inventory (NAS `checkpoints/temporal/`)
- `v01/{kuka_c1_v01,kuka_c8_v01}` — the v01-head fix (§18.12+); **current best**. `gate_kuka_c8_v01.json`.
- `fix/{all,flexiv,ur5,franka,kuka}` (+ seeds) — pre-fix full gate matrix (§18.8, query decoder). `gate.json` per dir.
- `c1/`, `raw/`, `fix2/` — diagnostic ablations (C=1 exoneration §18.11; raw-target divergence §18.12).
- `ur5b/` — first healthy temporal run (§18.7); `smoke/`, `ur5/` — early smoke tests.
- Each run dir has `log_seed*.json` (training curve: loss/inv/sig/RankMe/std per epoch).
