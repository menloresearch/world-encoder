# Handoff / Session Context

Everything needed to pick this project up on a fresh machine. Written for a VM switch —
`git clone` this repo, mount the NAS, `source env.sh`, and this doc tells you where everything is
and what's been decided. High-level pitch is in [README.md](README.md); per-stage results in
[EXPERIMENTS.md](EXPERIMENTS.md).

## 1. What this project is

**World Tokenizer** — a robot-first, cross-modal encoder that tokenises a robot's sensor streams
(video + robot_state, later audio + IMU) into one shared latent using **LeJEPA** self-supervision,
so downstream models (VLAs, world models) reuse one encoder instead of a vision-only one, and so
sensor data can be compressed edge→cloud. Base: `galilai-group/lejepa`. Dataset: RH20T **cfg3**.

## 2. Current state (as of this handoff)

| Stage | What | Status |
|---|---|---|
| 0 | Benchmark existing LeJEPA checkpoints | ✅ done |
| 1 | LeJEPA finetune on cfg3 **video only** | ✅ done — **no-op**, keep warm-start `e0` |
| 2 | Perceiver encoder, cfg3 **video + robot_state** | ✅ **verified** — cross-modal latent predicts robot 2× better than vision, beats PCA control |
| 3 | robot_state decoder on Stage-2 latents | not started |
| 5 | scale to video + state + audio, **modality × time** (MJEPA) | not started — this is where continuous-time embedding goes |
| 6 | state + audio decoder on Stage-5 latents | not started |
| 7 | real Microfactory data, same recipe | not started |

### Stage 1 result (why it's a no-op)
Full finetune at LR 2e-4 degraded the encoder (RankMe 300→158, task-id 0.91→0.74). Gentle LR 2e-5
*fixes the collapse* (RankMe ~285) but robot-relevant signal stays **flat** (contact probe
0.69→0.67, force-R²≈0). Conclusion: image-level LeJEPA on cfg3 video adds nothing robot-relevant —
**not** a data-quantity issue; the missing signal is temporal + other modalities. **Keep `e0`.**
The task-id probe is *saturated* (baseline 0.91) — it measures ImageNet appearance, not damage.
Always split **by scene**, never by frame (frame split leaks → 1.000).

### Stage 2 result (the headline)
Perceiver fuses frozen-`e0` vision patch tokens + a robot_state token; trained with cross-modal
masked latent prediction + SIGReg. Predicting robot state (R², 5 seeds, scene-held-out, 24k frames):

| feature → predict robot state | R² |
|---|---|
| **Perceiver `z_v` (256, cross-modal, state masked)** | **0.551 ±0.018** |
| raw vision (768, mean-pooled) | 0.257 ±0.075 |
| PCA-256 of LeJEPA vision (compression control) | 0.134 ±0.047 |

`z_v` beats raw by +0.294 and PCA-256 by +0.417 — **all 5 seeds**, RankMe 211 (no collapse).
Beating the PCA control ⇒ gain is **cross-modal, not compression**. Single timestep, masking over
**modality** (not time). **Honest caveat:** the eval is aligned with the training objective, and
`z_v` is a trained encoder vs frozen baselines — the clean ablation that isolates the cross-modal
gain (vision-only-*trained* Perceiver, no state token) is **not yet run**.

## 3. Environment (NAS — should persist across VMs if the NAS remounts)

Root disk `/` on the old VM was **100% full**; **all work runs on the NAS, never write to `/`**.
Everything lives under `/mnt/nas/data/RH20T`:

- `wae-venv/` — venv built `--system-site-packages` on top of base `/root/miniconda3` (torch
  2.8.0+cu128, numpy, opencv, torchvision); added transformers 4.57.1, timm 1.0.27, lightning
  2.6.5, transforms3d, einops.
- `hf_cache/`, `pip_cache/`, `tmp/` — HF_HOME / PIP_CACHE_DIR / TMPDIR redirected here.
- `RH20T_cfg3.tar.gz` (26 GB), `cfg3_raw/RH20T_cfg3/` (extracted raw), `cfg3_frames/` (extracted jpgs).
- `phase1_ckpt_e{1..10}.pt` — Stage-1 checkpoints (not used; `e0` warm-start is the encoder).
- `env.sh` — **`source /mnt/nas/data/RH20T/env.sh`** sets WAE_ROOT, HF_HOME, PIP_CACHE_DIR,
  PATH (venv), PYTHONPATH (rh20t_api), TMPDIR=/dev/shm/wae_tmp, `CUDA_VISIBLE_DEVICES` default 1.

**Volatile caches go on `/dev/shm`** (NAS is shared with other users and can get congested/slow).
The Stage-2 patch-token cache is `/dev/shm/wae_tmp/mm_patch.npz` (regenerate with `precompute_patch`).

**On the new VM:** confirm `/mnt/nas/data/RH20T` mounts, `source env.sh`, check GPUs with
`nvidia-smi`. If the venv doesn't work (different base conda), recreate `wae-venv` the same way.

## 4. Warm-start checkpoint — how to load it (this bit is fiddly)

Encoder = `OK-AI/lejepa-vitb16-pretrain-in1k` (custom HF "ViTv2", ~86M, DINOv2-style, 768-d).
**`AutoModel.from_pretrained(trust_remote_code=True)` FAILS** (the repo uses absolute imports of its
own sibling modules). Working loader (in `world_tokenizer/model.py`):

```python
import sys, torch
from huggingface_hub import snapshot_download
repo = snapshot_download("OK-AI/lejepa-vitb16-pretrain-in1k", allow_patterns=["*.py","*.json","*.safetensors"])
sys.path.insert(0, repo)
from modelling_vitv2 import ViTv2PretrainedModel
model = ViTv2PretrainedModel.from_pretrained(repo).eval().cuda()
out = model(x)              # x:[N,3,224,224] -> DICT
cls   = out["latent"]       # [N,768]  CLS  (NOT out.cls_tokens / pooler_output / out[0])
patch = out["patch_latent"] # [N,196,768]
```

## 5. Final encoder architecture (the target spec)

Each token = **(value, what, where, when)**:
- **value** — per-modality tokenizer: video→frozen ViT patches; state→symlog/sin-cos/6D→linear proj;
  audio/IMU→strided-conv/codec frontend.
- **what** — modality embedding (learned).
- **where** — spatial position (video patches only).
- **when** — **continuous-time embedding** = Fourier/learned features of the *real timestamp*
  (Time2Vec/mTAN style). This is what lets streams at different native rates fuse **without
  resampling to a common clock and without zero-padding**.
- **fuser** — **Perceiver**: M learnable queries cross-attend over the whole heterogeneous token set
  → fixed-size latent. Cost O(N·M), linear in token count.
- **train** — masked latent prediction over (modality × time) + per-modal SIGReg + joint SIGReg;
  later action-conditioned forward prediction for causality.
- **later (Stage 5+)** — causal time-masking or recurrent/SSM latent memory for streaming + loss #4.

**Status:** modality fusion (Stage 2) ✅ proven; **continuous-time / multi-rate is designed but NOT
built or tested** — Stage 2 was single-timestep, single-camera, state interpolated to the frame.
The continuous-time embedding is the load-bearing missing piece for the "native sampling rate" pitch.

## 6. Losses

| # | loss | role |
|---|------|------|
| 1 | masked latent prediction over (modality × time) tokens | learning signal; predict-don't-equate respects info asymmetry, avoids collapse, robust to missing modalities |
| 2 | per-modality SIGReg | anti-collapse + magnitude standardiser (commensurate modalities before fusion) |
| 3 | joint SIGReg on fused latent | keep fused latent high-rank |
| 4 | action-conditioned forward prediction (if actions matter) | causal engine; same-time alignment is only correlational |

SIGReg = `SlicedEppsPulley` in `stable_pretraining.methods.lejepa` (there is **no** `SIGReg`
symbol). Perceiver cross-attention = `stable_pretraining.backbone.vit.CrossAttention`.

## 7. Literature review — borrowed vs novel (positioning)

**Borrowed (cite, don't claim):**
- Continuous-time embedding mechanism: **mTAN** (`2101.10318`), **Time2Vec** (`1907.05321`),
  **SeFT** (`1909.12064`), **ContiFormer** (`2402.10635`).
- Continuous-time + Perceiver: **COPER** (`2208.03196`) — nearest architectural twin (uses Neural-ODE).
- Multimodal irregular fusion: **FuseMoE** (`2402.03226`), **UTDE** (`2210.12156`) — but all EHR domain.
- Perceiver backbone: **Perceiver IO** (`2107.14795`).
- Cross-modal masked latent prediction: **M3-JEPA** (`2409.05929`), **4M** (`2312.06647`),
  **MultiMAE** (`2204.01678`), **data2vec** (`2202.03555`).
- LeJEPA / SIGReg: **LeJEPA** (`2511.08544`). V-JEPA 2 (`2506.09985`) for action-conditioned latent.
- Robot multi-sensor SSL: **MSDP** (`2511.14427`, closest — but assumes *synchronized* obs),
  See-Hear-Feel (`2212.03858`), M3L (`2311.00924`), Sparsh/Sparsh-X (`2410.24090`/`2506.14754`).
  OpenVLA (`2406.09246`) = the "state bolted on as one token" paradigm we argue against.

**Genuine white space (our contribution):** cross-modal masked **latent** prediction (LeJEPA/SIGReg)
over **3+ heterogeneous robot streams at their native different rates** via continuous-time tokens +
Perceiver. Each ingredient exists; nobody has assembled *this* combination, and the closest robot
SSL encoder (MSDP) explicitly skips the multi-rate part — which is our core thesis.
**Must-reads in order:** mTAN → FuseMoE → COPER → MSDP → V-JEPA 2.
(Caveat: several key papers are Nov 2025–Mar 2026 preprints — real but unreviewed.)

## 8. Repo layout & code (`world_tokenizer/`)

- `model.py` — `load_vitv2` / `LeJEPAVideo` (the checkpoint loader above).
- `state.py` — 28-dim robot_state loader (joints→sin/cos 12, tcp→symlog 3, quat→6D 6, F/T→symlog 6,
  gripper→symlog 1). Reads `transformed/*.npy` directly + interpolates, to dodge an order-dependent
  bug in rh20t_api's aligned getters. `FT_DIMS = range(21,27)`.
- `precompute_patch.py` — cache frozen-`e0` patch tokens (196×768 fp16) + state → `/dev/shm` npz.
- `mm_perceiver.py` — `MMPerceiver` (the Stage-2 encoder): PerceiverFuse + cross-modal masked
  prediction + per-modal/joint SIGReg + EMA target. ~2M trainable params, vision frozen.
- `train_perceiver.py` — Stage-2 train+eval, 5-seed, retrain-per-split, PCA-256 baseline built in.
- `step1_gate.py` — cross-modal signal gate (frozen vision → state R²=0.43).
- Stage 0/1 pipeline: `extract_frames` → `make_shards` → `train` (DDP) → evals
  `eval_lejepa` / `probe_curve` / `contact_probe` / `robust_robot_eval`; `gate` = video↔force sanity.

**DDP note:** use `python -m torch.distributed.run` (NOT the `torchrun` binary — that resolves to
the base env without the venv packages).

## 9. Repo / git

- Remote `origin` = `https://github.com/menloresearch/world-autoencoder.git`, branch **`user/ishneet`**.
- All commits authored **`Ishneet Sukhvinder Singh <85265554+Ishneet0710@users.noreply.github.com>`**
  — **no Claude attribution** (project rule). **Commit/push only when asked.**

## 10. Open threads / next steps

1. **Vision-only-trained ablation** — same Perceiver, no state token, probe → state. Isolates the
   cross-modal gain from "just trained an in-domain encoder." Cheap (~5.6 min/seed). Do this before
   the Stage-2 result goes external as the headline.
2. **Continuous-time embedding (Stage 5 kickoff)** — extend `MMPerceiver` with a Time2Vec/mTAN-style
   time embedding; feed robot_state at its **native ~100 Hz** vs video ~10 Hz (stop interpolating
   down); mask over time → predict held-out-time state/force. First real multi-rate test.
3. **Ablations** — bottleneck size (`n_queries`/`d`), joint-SIGReg on/off.
4. Per-epoch progress logging in `train_perceiver.py`.
5. **Blog / Related-Work section** from the lit review (borrowed-vs-novel split above).

## 11. Memory files (from the old VM's `~/.claude`, folded in above)

These lived at `~/.claude/projects/-root-ishneet/memory/` and won't transfer with the VM. Their
content is captured in this doc: `wae-checkpoint-loader` (§4), `wae-nas-environment` (§3),
`wae-phase1-result` (§2), `wae-stage2-result` (§2). If the new VM keeps the same `~/.claude`, they're
still there; otherwise this file is the source of truth.
