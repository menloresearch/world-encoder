# World Autoencoder

## Research Plan — Project 1: World Tokenizer

### Roadmap
- **Stage 0** — Benchmarking existing LeJEPA checkpoints
- **Stage 1** — LeJEPA trained/finetuned on RH20T_cfg3 **video** data only
- **Stage 2** — LeJEPA training with a modified encoder on RH20T_cfg3 **video + robot_state** data
- **Stage 3** — Train a **robot_state decoder** on the output of stage 2
- **Stage 5** — Scale encoder to **video + state + audio** and adopt the **MJEPA** training strategy
- **Stage 6** — Train a **state + audio decoder** on the output of stage 5
- **Stage 7** — Collect **real data from Microfactory** and train using the strategy developed in stages 1–6

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
