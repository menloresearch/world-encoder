# N1 downstream encoder-swap — RoboCasa365 dry run (live doc)

**What:** JQ's 3-row table (2026-07-20 Slack) run in sim first: baseline Diffusion Policy /
Kepler **e2e** (random-init encoder) / Kepler **pt-enc** (JEPA-pretrain first, then frozen in
the policy). Same recipe everywhere; ONLY the encoder differs. Success-vs-training-budget
curves (ckpt every 50 ep, 50 rollouts each) test "pt-enc trains faster", not just endpoints.
De-risks the identical Molmobot/real-microfactory version. Sim has NO F/T → these runs test
"encoder helps a policy" + multi-view fusion; the force story stays on RH20T (V0.2.md).

## Where things live (all OUTSIDE this repo)
- `~/brain/ishneet/robocasa` — RoboCasa365 clone + the shared **uv venv** (`.venv`) used by
  everything; assets+datasets symlinked to `/mnt/nas/data/robocasa/` (local disk was 95% full).
- `~/brain/ishneet/robosuite` (master) + `~/brain/ishneet/robomimic` (`robocasa` branch) — deps.
- `~/brain/ishneet/diffusion_policy` — robocasa-benchmark DP fork = the policy trainer.
  RUN EVERYTHING FROM THIS DIR (no `__init__.py` at package root → cwd import only).
- Datasets (lerobot format, 3 cams 256²+proprio+12-D commanded actions): NAS
  `datasets/v1.0/target/atomic/{PickPlaceCounterToCabinet,OpenDrawer,CoffeeSetupMug}` (~500
  human demos each).

## Key builds (2026-07-20)
1. **Frame cache** (`diffusion_policy/build_frame_cache.py`): decode every demo mp4 ONCE →
   uint8 memmap per camera on NAS (`.../frame_cache/`) + patched
   `LerobotDataset.get_video` fast path. Training went decode-bound (~30 min/ep) →
   GPU-bound (~8.5 min/ep 1-GPU, ~5 min DDP-2). Works for ANY lerobot dataset (incl.
   Molmobot VLA training later — told Ishneet to flag to Alex).
2. **Kepler encoder arm** (`diffusion_policy/diffusion_policy/model/vision/kepler_encoder.py`
   + `policy/kepler_diffusion_transformer_policy.py` + config `train_kepler_bs192.yaml`):
   frozen ViTv2 → shared proj + camera-slot emb → ONE PerceiverFuse over 3·196 tokens →
   256-d + lowdim passthrough. ViT frozen in ALL arms (controlled variable). Optimizer
   excludes frozen params. `policy.kepler_ckpt=... policy.freeze_kepler=True` = pt-enc arm.
3. **In-domain pretraining** (`world_tokenizer/robocasa_pretrain.py`, THIS repo): MMPerceiverMC
   (mm_perceiver.py 2-modality JEPA + cam-slot emb = Build-1 trick), vision + 16-d proprio,
   26.6k samples (stride 5). **GATE GREEN: z_v state R² 0.673 vs raw-ViT 0.538, RankMe 190**
   (no collapse). Ckpt `/mnt/nas/data/robocasa/kepler_ckpt/pnp_mc3/seed0.pt`
   (state_dict keys proj_v/cam_emb/mod/fuse load straight into KeplerMultiCamEncoder).
   Row 3 = multicam ARCHITECTURE with fresh in-domain weights — NOT v0.1/kuka_mc4 weights
   (real→sim confound). Cam order everywhere = sorted = (agentview_left, agentview_right,
   eye_in_hand).
4. **Compat patches** (fork-local): diffusers `Union` imports ×2 (DP `lr_scheduler.py`,
   robomimic `torch_utils.py`), vendored gym `AsyncVectorEnv` (shared_memory=False,
   reset seed/options kwargs, `concatenate` arg order), OmegaConf 'eval' resolver clash
   (stable_pretraining vs train.py — guarded in kepler_encoder.py), wandb → `logging.mode=disabled`.
5. **NORMALIZATION BUG (found+fixed 2026-07-20 ~16:20, cost ~10h of 4 runs):** DP's
   `get_image_range_normalizer` maps images [0,1]→**[-1,1]** BEFORE the policy sees them; the
   Kepler encoder assumed [0,1] → the FROZEN ViT ate wildly OOD inputs in both Kepler arms
   (mid-gray pixel → −2.1σ). Symptom: ep-50 PnP success baseline 24% vs e2e 6% / pt-enc 4%,
   pt-enc ≈ e2e (pretraining nullified through a broken ViT). Fix = `x*0.5+0.5` in
   kepler_encoder.forward before ImageNet norm. All 4 Kepler arms RESTARTED from scratch 16:29.
   Lesson: a frozen backbone turns any input-distribution mismatch into a silent quality bug —
   trainable baselines mask it; smoke tests with `torch.rand` can't catch it. Eval evidence, not
   loss curves, exposed it. (Also: eval outputs must use explicit per-run dirs — the ckpt-to-NAS
   symlink made eval_task's `../evals` collide across runs; fixed in eval_one.py/eval_watcher.py.
   And MUJOCO_EGL_DEVICE_ID must pin eval render to the eval GPU.)

## ⚠ 2026-07-22: DISK-FULL KILLED THE FLEET (root fs 100% at 07-21 ~23:39)
All runs below are DEAD; GPUs 0–7 idle. The 10:43/16:22 relaunches saved ckpts locally (2.4G
each; only s0's 04:21 dir kept the NAS symlink) and filled the disk. States at death — hybrid s1
`10.43.21`: ~ep240, latest.ckpt=ep200, ep50–200 eval'd (32/24/24/24%); hybrid s0 `04.21.45`:
~ep126, latest=ep100; baseline-s1 `16.22.47`: ~ep50 — its `epoch=0050` ckpt was truncated
mid-write (deleted; caused watcher2's rc=120 eval loop) but `latest.ckpt` is valid. Cleanup done
07-22: all local ckpts moved to NAS `dp_checkpoints/` (zip-verified) + symlinks swapped — every
run dir is now NAS-symlinked; disk 98%→93% (35G free; the remaining ~366G is root-owned:
40G Qwen3-VL HF cache in /home/root + /var/lib/docker-infra, VM-admin territory).
TODO: resume s1 from ep200 / s0 from ep100 / baseline-s1 from latest; re-arm eval_watcher2.

## Runs — fleet before the crash (as of 2026-07-21 ~06:45, ALL-IN-ON-HYBRID reallocation; tmux on the VM)
**Reallocation 06:35 (Ishneet's call — "all resources to the hybrid"):** baselines STOPPED (their
matched curves through ep 150 already exist — that's all the hybrid comparison needs); freed GPUs →
**hybrid seed 1 immediately** (the delta needs 2 seeds at ±6-7% eval noise; more GPUs on seed-0 would
mean a from-scratch DDP restart — no ckpt before ep 50 — for ~no wall-clock gain).
| tmux | GPUs | arm | task | run dir (`data/outputs/…`) |
|---|---|---|---|---|
| — (STOPPED 06:35 @ep176) | — | baseline DP (evals kept ep50/100/150; resumable from ep-150 ckpt) | PnP | `2026.07.20/04.27.20_…hybrid_pnp_single` |
| — (STOPPED 06:35 @ep133) | — | baseline DP (evals kept ep50/100) | OpenDrawer | `2026.07.20/06.17.57_…hybrid_opendrawer` |
| — (CULLED @ep150) | — | Kepler e2e (ep-150 gate hit: eval'd 14%) | PnP | `2026.07.20/16.29.42_…kepler_dp_pnp` |
| — (CULLED @ep150) | — | Kepler pt-enc (ep-150 gate hit: eval'd 2%) | PnP | `2026.07.20/16.37.35_…kepler_dp_pnp` |
| — (STOPPED) | — | Kepler e2e / pt-enc OD (replacement claim dead; ep-50 ckpt+eval kept) | OpenDrawer | `16.29.16` (paused ~ep35) / `16.37.21` (stopped ep51, eval 22%) |
| **dp_hybrid** | **6,7 DDP-2** | **HYBRID seed 0** (relaunched 07-21 04:21, bs 96×2; ep77 @16:20) | PnP | `2026.07.21/04.21.45_…hybrid_kepler_pnp` |
| **dp_hybrid_s1** | **0–3 DDP-4** | **HYBRID seed 1** — 06:38 launch DIED ~08:30 (zero ckpts, `06.38.26` dir dead); relaunched fresh 10:43 as DDP-4 + `kepler_z_cache` (~3.2 min/ep — ep104 @16:20, laps seed 0) | PnP | `2026.07.21/10.43.21_…hybrid_kepler_pnp` |
| **dp_baseline_s1** | **4** | **baseline DP seed 1** (launched 16:22 by Claude, single-GPU bs192 recipe parity w/ seed 0 + training.seed=1; log `/tmp/dp_baseline_s1.log`, no tmux — setsid-detached pid 2701723) | PnP | `2026.07.21/16.22.47_…hybrid_pnp_single` |
| eval_watcher | 6,7 (eval slots) | re-armed 06:45; still lists dead `06.38.26` (will simply never fire) | — | CSV `results/downstream/n1_results_snapshot_20260721.csv` |
| eval_watcher2 | 5 | launched 16:30 by Claude for the two runs watcher-1 doesn't know: `10.43.21` (s1 ep50+ep100 queued immediately) + `16.22.47`; same CSV (append-mode, safe); log `/tmp/eval_watcher2.log`, pid 2704064 | — | same CSV |

Ep-150 cull done. The camera-dropout retrain turned out to have ALREADY run in the 10:33 session
(ckpt `kuka_mc4_dropout` on NAS 11:27) — post-retrain probes + per-cam breakdown completed ~16:50:
**dropout does NOT deliver** (details RESULTS.md §5e). All three V0.2 JQ follow-ups now done.

("hybrid" in the *baseline* dir names = robomimic's hybrid image+lowdim DP transformer, NOT our
Kepler-hybrid — do not confuse.) 400 epochs (500 steps each — fork caps steps/epoch), ckpt every 50,
checkpoints symlinked to NAS `dp_checkpoints/`. Early note: pt-enc trains FASTER per step than e2e
(1.56 vs 1.42 it/s) — frozen encoder skips its backward.
**Watcher re-armed 07-21 ~04:15** (the rate-limit-killed session had left the hybrid run unwatched and
the CSV in volatile `/tmp`): now watches all 7 runs **including the hybrid**, appends to the durable
CSV in `world-encoder/results/downstream/`. Restart-safety note for the future: the watcher skips any
ckpt that already has an `eval_log.json` and re-queues the rest, but it also RE-APPENDS known results
to the CSV on startup → dedup rows after a restart (`awk '!seen[$0]++'`).

## Results so far (50 rollouts/ckpt, updated 07-22 after the disk-full crash)
**PickPlaceCounterToCabinet:**
| arm | ep 50 | ep 100 | ep 150 | ep 200 |
|---|---|---|---|---|
| baseline DP s0 | 24% | 28% | 32% | — (stopped ep176) |
| baseline DP s1 (`16.22.47`) | (no eval yet — corrupt ep-50 ckpt deleted; state lives in latest.ckpt) | — | — | — |
| Kepler e2e (post-fix) | 0% | 12% | 14% | — |
| Kepler pt-enc (post-fix) | 4% | 2% | 2% | — |
| Kepler HYBRID s0 (`04.21.45`) | 22% | 24% | — (died ep126) | — |
| Kepler HYBRID s1 (`10.43.21`) | **32%** | 24% | 24% | 24% (died ~ep240) |

**Ep-150 gate: closed.** Replacement arms end 14% / 2% vs baseline 32% — replacement claim
falsified on this benchmark, exactly as the action-readout probes predicted. Honest-ablation rows
complete (3 points/arm for JQ's table).
**Hybrid read as of 07-22 (still not the final word):** at matched epochs the hybrid shows NO lift —
ep100: baseline 28% vs hybrid 24%/24% (both seeds); ep150: baseline 32% vs hybrid s1 24%; s1 is FLAT
at 24% from ep100→ep200 while the baseline was still climbing at ep150. The s1 ep50 = 32% spike is a
single point contradicted by s0's 22% at the same epoch — treat as ±6-7% noise until baseline s1's
curve exists. Latent doesn't hurt (unlike replacement) but no evidence it ADDS; if this pattern holds
with both seeds through ep150, the honest table row for JQ is "hybrid ≈ baseline, no added information."
Missing before concluding: baseline s1 curve (zero points), hybrid s0 ep150.

**OpenDrawer:**
| arm | ep 50 | ep 100 |
|---|---|---|
| baseline DP | 74% | 70% |
| Kepler pt-enc (post-fix, `16.37.21`) | **22%** | — (stopped ep 51) |
| Kepler e2e (post-fix, `16.29.16`) | — (paused ~ep35, never reached ckpt) | — |

→ replacement fails on the easy task too (pt-enc ep-50 eval'd 07-21 ~05:45 from the stopped arm's
kept ckpt). **ATTRIBUTION FIX 07-21:** earlier prose here (and the unsent Slack draft) credited the
22% to *e2e* — wrong; `16.37.21`'s hydra config has `kepler_ckpt=…od_mc3/seed0.pt, freeze_kepler=True`
= pt-enc. Fix the Slack draft before sending. Pre-fix tainted rows deleted; baseline rows were never
affected by the norm bug.
**ep-100 wrinkle:** e2e jumped 0→12% while pt-enc dropped 4→2% — the earlier "faint pt-enc > e2e"
signal is INVERTED at ep 100; at these low rates (±6-7% noise at 50 rollouts) the two inits don't
cleanly rank. Both remain miles under baseline (28% at ep 100) → the failure is architectural
(latent-as-only-eyes), not the init. Don't cite the pt-enc≥e2e ordering in updates to JQ.
Baseline PnP is climbing steadily (24→28→32) — the ep-150 comparison gate for the replacement arms
lands when their ep-150 ckpts eval (~07-21 morning, cull after). The **decision number of the week is
baseline-vs-HYBRID at matched epochs** ("does the frozen pretrained latent ADD information").

## Probe-driven pivot (2026-07-20 evening → 07-21)
Cheap-probes-first loop (user-endorsed) before burning more fleet time.

**Sanity gate — state-readout R²** (the encoder's own objective; from Key-builds #3, in-domain
pretrain on 26.6k PnP samples). The encoder is sane at what it's trained for:
| representation | state R² |
|---|---|
| raw ViT patches | 0.538 |
| pretrained fuse z_v | **0.673** (RankMe 190 → no collapse) |

**Action-readout probe — R²** (predict the commanded 12-D×5 sequence from vision;
object-directed → proxy for task-relevant info):
| representation | action R² | read |
|---|---|---|
| proprio only | 0.176 | floor — what vision must add on top of |
| random-init fuse | 0.287 | architecture alone, no pretraining |
| pretrained fuse, q8 (pooled) | **0.335** | set-of-queries 0.337 → pooling is FREE |
| pretrained fuse, q32 / q64 | 0.338 / 0.340 | query width does NOT help |
| raw frozen patches, coarse-spatial (3cam×2×2) | **0.413** | the ceiling sitting in the SAME patches |

Conclusion: the info loss is inherent to cross-attn fusion under a state-predicting JEPA objective
(no pressure to keep fine object detail) — NOT the final pooling, NOT query capacity. Matches evals:
post-norm-fix arms at ep50 = e2e 0%, pt-enc 4% vs baseline 24→28% (ep100). Faint JQ-directional
signal: pt-enc > e2e (ep-50 only — INVERTED at ep 100, see Results; the ordering is noise, don't
build on it).
**DECISION: replacement claim is wrong on this benchmark → build the FLARE-faithful HYBRID row**
(`policy/hybrid_kepler_policy.py`, config `train_hybrid_kepler_bs192.yaml`): stock robomimic
ResNet18-FiLM spatial encoder UNCHANGED ++ frozen pretrained Kepler fused latent (256-d, z only —
lowdim comes via the base) concatenated to its output. Baseline-vs-hybrid isolates "does the latent
ADD information" — the claim FLARE's evidence actually supports (their latent also sits NEXT TO the
visual stream, not instead of it). Replacement arms keep running as the honest ablation row.
**Hybrid smoke-tested 01:51, launched 02:02 single-GPU (~20 min/ep — runs BOTH encoders), RELAUNCHED
04:21 as DDP-2 on GPUs 6+7** (bs 96×2, ~1.4 s/it ≈ 11.5 min/ep → ep-50 ckpt ~14:00 same day; the dead
02.02.18 dir has no ckpts). **Fleet-priority decision (07-21, Ishneet):** the hybrid is the decision
experiment → it gets GPUs first; pt-enc OD stopped (easy task × dead claim — ep-50 ckpt + eval kept);
PnP replacement arms run ONLY to the ep-150 gate (falsification insurance for the probe story + completes
JQ's table 3 points/arm), then get culled → freed GPUs go to hybrid seed 1 + baseline seed 1 (the
baseline-vs-hybrid delta is the deliverable; 50-rollout noise ±6-7% → 2 seeds needed). Future recipe fix
for the encoder itself (not this week): add object/scene pressure to the pretraining objective (patch
reconstruction / DINO-style distillation into the fuse).

## Related result: view-composition analysis (JQ's Slack ask, 2026-07-21)
Ran on the RH20T kuka multicam model (not RoboCasa) — full numbers in
`results/temporal/RESULTS.md` §5d, one-liner: the fused embedding is view-SPECIFIC not view-invariant
(same-instant different-compositions 1.46× further apart than cross-instant); JQ's late-fusion
baseline (mean of per-view v0.1 embeddings) is worse than a single view on every probe →
**early fusion > single view > late fusion**. UPDATE 07-21 ~17:00: all 3 JQ follow-ups DONE
(RESULTS.md §5e) — encode-per-view+avg also loses; per-cam spread real but no wrist cam in the set;
**dropout retrain measured and does NOT deliver** (space contraction, not invariance). Slack reply
drafted + reviewed 07-21 (posting = Ishneet) — **AMEND the draft's "dropout is the offered knob"
line to the §5e(3) result before posting.**

## Next after the table
- Action-conditioned predictor on RoboCasa's COMMANDED 12-D actions (the real "X" RH20T
  lacks) → planner (V-JEPA-2-AC) + FLARE-style aux loss. Ask Alex: log commanded targets
  on the real rig from day one.
- Extra row later: RH20T-pretrained encoder as-is (tests real→sim transfer).
- Port pretrainer to Molmobot (data-loader swap; 5 cams, JSON-in-h5, no F/T).
