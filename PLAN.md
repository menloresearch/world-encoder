# PLAN — Open-source + Full-RH20T Vis+State Training Run

Working plan for Ishneet + Jia Qi. Do the execution **on the new VM** — this doc is the checklist.
Context/state in [HANDOFF.md](HANDOFF.md); Stage-2 result in [EXPERIMENTS.md](EXPERIMENTS.md).

## Deadlines
- **By end of tomorrow:** `world-encoder` repo goes **public**.
- **Over the weekend:** full training run of the **Vis+State encoder** on the **full RH20T** train set.

## Locked decisions
- **Modalities:** vision + robot_state only. **Skip audio** for now.
- **Encoder:** frozen `e0` vision (patch tokens) + **Perceiver fusion** (`MMPerceiver`), **single
  timestep** (masking over MODALITY, not time). This is the validated Stage-2 setup.
  - Do **NOT** unfreeze vision (Stage 1 showed video finetuning degrades the encoder).
  - Temporal / continuous-time is **out of scope** for this run (that's Stage 5, later).
- **Data:** full RH20T, **all 7 configs** (already on NAS at `raw/RH20T_cfg{1..7}`).
- **State = ROBOT-AGNOSTIC 16-dim** (works across every robot; joints vary 6/14/21 so they're dropped):
  ```
  state = [ symlog(tcp_pos)   3
            6D(tcp_quat)      6
            symlog(ft)        6
            symlog(gripper)   1 ] = 16
  ```
  - Filter out `*_human` scenes (human demos — no robot state).

## Why these decisions (evidence, so we don't relitigate)
- Joint dim differs by config: **cfg1=14, cfg3=6, cfg7=21** → a fixed joint layout can't span configs.
  TCP pose + F/T + gripper are semantically consistent across all robots → 16-dim is portable.
- `*_human` scenes have **no** `transformed/joint.npy` → must be skipped or the loader crashes.
- Stage 2 (frozen vision + Perceiver, cfg3) already beat raw vision (+0.29 R²) and PCA-256 (+0.42),
  all 5 seeds — so scaling *this* setup is low-risk. Unfreezing/temporal is not.

---

## Workstream A — Preprocessing refactor  *(BLOCKS the run)*
- [ ] **A1.** `state.py` → robot-agnostic 16-dim: drop the joint sin/cos block; keep tcp-pos symlog(3)
      + tcp-quat 6D(6) + ft symlog(6) + gripper symlog(1). Set `STATE_DIM=16`; update `FT_DIMS`
      (F/T now at indices 9..15).
- [ ] **A2.** Scene filter: skip any dir ending `_human`, and any scene missing
      `transformed/tcp_base.npy` / `gripper.npy`. Do this in `precompute_patch` scene loop.
- [ ] **A3.** **Verify F/T exists in cfg2/4/5/6** (human scenes blocked the check earlier — only
      1/3/7 confirmed `ft_base=True`). If a config lacks F/T: zero-fill + a "ft-valid" mask bit, or
      exclude those configs. **Confirm with Jia Qi.**
- [ ] **A4.** Verify `tcp_base.npy` / `gripper.npy` format is consistent across configs (serial keys,
      list-of-dicts). `state.py::_from_list` assumes cfg3's layout — check it holds for cfg1/2/4/5/6/7.
- [ ] **A5.** Per-config sanity: load 1 scene per config, assert a finite 16-dim vector comes out.
- [ ] **A6.** Make the raw path configurable: `RAW` → env var (e.g. `RH20T_RAW`, default
      `raw/RH20T_cfg3`); add a `--configs` arg to iterate multiple configs. (Ties into E1.)

## Workstream B — Data pipeline scaling  *(BLOCKS the run)*
- [ ] **B1.** The `/dev/shm` single-npz cache **won't scale**: patch tokens are 196×768 fp16 ≈ 301 KB
      /frame; full RH20T at ~30 frames/robot-scene is easily 100+ GB → exceeds RAM. **Decide:**
      (a) shard the patch-token cache to **NAS** (compute once, reuse across seeds/epochs) — recommended,
      or (b) run the frozen ViT **on-the-fly** in the dataloader (no big cache, slower per epoch).
- [ ] **B2.** Build the sharded precompute: iterate configs → filter scenes (A2) → sample frames →
      frozen-`e0` patch tokens + 16-dim state + (scene, config, timestamp) → write **shards to NAS**.
- [ ] **B3.** Point `train_perceiver`'s loader at shards instead of the single npz.
- [ ] **B4.** Estimate storage + wall-time; tune frames/scene to fit the weekend window. `log()` any cap.

## Workstream C — Train/test split & eval  *(needed for a credible public result)*
- [ ] **C1.** **Fixed, documented split** saved as a manifest in the repo. Scene-held-out (whole
      scenes to test), stratified across configs + tasks, fixed seed. **Decide granularity with Jia Qi:**
      held-out by *scene* vs by *task* (task-level = harder, tests task generalization).
- [ ] **C2.** No leakage: encoder **and** probe see only train scenes; test scenes never trained on.
- [ ] **C3.** Improve eval: keep predict-robot-state R² (`z_v` vs raw vs PCA-256) + RankMe; add TCP/
      gripper linear probe, contact/force where present, and a **per-config breakdown**.
- [ ] **C4.** Keep multi-seed error bars.

## Workstream D — PCA / embedding data analysis  (the "LeJEPA data-analysis" ask)
- [ ] **D1.** `visualize_stage2.py` is **restored** to the repo (was removed; backup was session-only).
      It produces: R² bar, PCA scatter colored by task/force, scree curves, nearest-neighbor photo
      comparison, photos-laid-out-in-PCA-space. Re-verify it runs against the new cache format.
- [ ] **D2.** Extend to full data: PCA of embeddings colored by **config/robot/task**; show the latent
      separates robots/tasks. This is the figure set for the blog.
- [ ] **D3.** Note: `precompute` must also save `path` + `timestamp` for the photo-map figures (this
      was the small `precompute_patch` diff that got reverted — re-add it when wiring B2).

## Workstream E — Open-source cleanup  *(BLOCKS public, due tomorrow)*
- [ ] **E1.** Remove hardcoded `/mnt/nas/...` paths → env vars / args. **Coordinate with Jia Qi** —
      he owns the new `env.sh` replacement, `requirements.txt`, and the `data/RH20T/raw` layout.
- [ ] **E2.** Install `stable-pretraining` as a **package**, not editable (Ishneet confirmed no local
      edits). Pin it in requirements.
- [ ] **E3.** README already high-level (+ EXPERIMENTS + HANDOFF). Make it runnable by an outsider:
      setup steps, data pointer to `rh20t.github.io`, exact run commands.
- [ ] **E4.** Strip internal-only artifacts (NAS-specific paths, `gate_*.png` refs, scratch outputs).
      Secrets scan.
- [ ] **E5.** Add LICENSE / contributing if going public (Jia Qi?).
- [ ] **E6.** Keep commit authorship consistent (Ishneet, no Claude attribution).

---

## Proposed owner split
- **Jia Qi:** `env.sh` replacement, `requirements.txt`, data-folder consolidation (done → `raw/`),
  repo packaging for public (E1/E5).
- **Ishneet (+ Claude):** state.py robot-agnostic refactor (A), sharded data pipeline (B),
  split + eval (C), PCA viz (D), path configurability (A6 / E1 code side).

## Order of operations on the new VM
1. Mount NAS; source new env; verify `raw/RH20T_cfg{1..7}` + `wae-venv`.
2. **A1–A5** preprocessing refactor + per-config sanity (fast; unblocks everything — do first).
3. **B** sharded precompute (longest job — kick off early, runs while you refactor).
4. **C1** write the fixed split manifest.
5. Launch the **full training run** (weekend) once B + C are ready.
6. **D** visualizations after the run (or run D on the existing cfg3 cache now for the blog).
7. **E** cleanup in parallel to hit tomorrow's public deadline.

## Open questions for Jia Qi (confirm before the run)
1. F/T presence per config (A3) — do all 7 have force/torque, or exclude some?
2. Split granularity (C1) — by-scene or by-task?
3. Frames-per-scene budget given NAS storage (B4).
4. "Full training set" = all 7 configs, or exclude any (e.g. human-heavy configs)?

## Data location note (already changed on NAS)
Raw data moved: `cfg3_raw/RH20T_cfg3` (now empty) → **`raw/RH20T_cfg3`**, and all 7 configs are
present under `raw/`. Code still hardcodes the old path in `precompute_patch.py`, `contact_probe.py`,
`extract_frames.py` — fix via A6. `cfg3_frames/` and `cfg3_shards/` are untouched.
