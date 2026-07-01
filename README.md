# World Autoencoder

## Research Plan — Project 1: World Tokenizer

### Roadmap
- **Stage 0** ✅ — Benchmarking existing LeJEPA checkpoints
- **Stage 1** ✅ — LeJEPA trained/finetuned on RH20T_cfg3 **video** data only
- **Stage 2** — LeJEPA training with a modified encoder on RH20T_cfg3 **video + robot_state** data
- **Stage 3** — Train a **robot_state decoder** on the output of stage 2
- **Stage 5** — Scale encoder to **video + state + audio** and adopt the **MJEPA** training strategy
- **Stage 6** — Train a **state + audio decoder** on the output of stage 5
- **Stage 7** — Collect **real data from Microfactory** and train using the strategy developed in stages 1–6

---

## Progress — Stages 0 & 1 (done)

Stage 0 (benchmark the warm-start checkpoint) and Stage 1 (finetune on cfg3 video) are done.
The full pipeline is built and verified; the finetune produced a clear **negative result**.

**Pipeline (verified end-to-end):** RH20T cfg3 → extract frames (`rh20t_api`) →
**2,330,532 jpgs** (799 robot scenes, 66 tasks) → **240 WebDataset shards** →
**DDP×7 continue-LeJEPA** (warm-start `OK-AI/lejepa-vitb16-pretrain-in1k`, LR 2e-4, 10 epochs,
BF16, ~2 h, no loss collapse) → eval. Everything runs on the NAS (see `world_tokenizer/`).

**Result — finetuning *degraded* the encoder.** Scene-held-out probe (66-task id, unseen
scenes, chance 0.015); RankMe = effective rank of embeddings (label-free), max 768:

| ckpt | linear | kNN(20) | RankMe |
|------|--------|---------|--------|
| **e0** (warm-start) | **0.908** | **0.814** | **300** |
| e3 | 0.738 | 0.324 | 217 |
| e10 | 0.717 | (↓) | (↓) |

All three metrics — including **label-free RankMe (300 → 217 = partial dimensional collapse)** —
fall after finetuning. **The warm-start (e0) is the best encoder.**

**Takeaway / next:** continuing plain LeJEPA on one config's video at LR 2e-4 over-adapts and
forgets the strong pretrained features. Try much lower LR (~2e-5) / fewer steps, a
less-saturated eval (contact/gripper from F/T), and keep **e0** as the current-best encoder and
the Stage-2 starting point.

---

## Architecture (scratchpad)

Base: **`galilai-group/lejepa`**

**Dataset** ✅
- **RH20T (cfg3)** — smallest subset, gets us started fastest. Scale to the other subsets later.

**Latent structure**
- continuous, discrete

**Pre-processing** (per modality)
- **symlog** — unbounded quantities (velocity, position, current, tactile)
- **sin/cos** — angles
- **6D / canonicalize** — quaternions

**Encoder** ❓
- LeJEPA

**Decoder** ✅
- **PixNeRD → latent diffusion model**

**Training framework** ❓
- LeJEPA minimal example (no full training script): https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
- Full training example (le-wm): https://github.com/lucas-maes/le-wm/tree/main/config/train
- stable-pretraining (LeJEPA lab; framework for LeJEPA + all SSL — BYOL, DINO, …):
  https://github.com/galilai-group/stable-pretraining/blob/main/stable_pretraining/methods/lejepa.py ·
  [METHODS.md](https://github.com/galilai-group/stable-pretraining/blob/main/METHODS.md)

**Losses** ✅ — LeJEPA loss → SIGReg + prediction (MSE).

The loss combination and why we need it:

| # | Loss | What it does | Why (failure it prevents) |
|---|------|--------------|---------------------------|
| 1 | Masked latent prediction over (modality × time) tokens — *what MJEPA does* | Predict masked embeddings from visible ones (cross-modal + temporal in one mask) | The representation signal. Predict-don't-equate respects info asymmetry → avoids intersection-collapse. Robust to signal loss by default. |
| 2 | Per-modality SIGReg | Push each modality's embedding → isotropic Gaussian | Anti-collapse + magnitude standardizer (makes modalities commensurate before fusion). Joint SIGReg would force cross-modal independence; per-modal still prevents collapse. |
| 3 | Joint SIGReg on the fused latent | Push fused latent → isotropic / high-rank | Keeps the world-model latent expressive, not collapsed. |
| 4 | Action-conditioned forward prediction *(only if actions matter — needed for a robotics model)* | Predict future fused latent from current + action | The causal engine — same-time alignment is only correlational. |

**Post-training experiments:** TBD

---

## Code
`world_tokenizer/` — the LeJEPA + RH20T_cfg3 pipeline (extract → shards → DDP train → eval).
Everything runs from the NAS: `source /mnt/nas/data/RH20T/env.sh`, then see
[`world_tokenizer/README.md`](world_tokenizer/README.md) for the run order.
