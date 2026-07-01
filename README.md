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

**Verdict: finetuning LeJEPA on cfg3 video is a *no-op* — it neither meaningfully helps nor
harms. Keep the warm-start `e0` as the encoder; the real signal is in Stage 2+.**

**Pipeline (verified end-to-end):** RH20T cfg3 → extract frames (`rh20t_api`) → **2,330,532 jpgs**
(799 robot scenes, 66 tasks) → **240 WebDataset shards** → **DDP×7 continue-LeJEPA** (warm-start
`OK-AI/lejepa-vitb16-pretrain-in1k`, BF16, ~2 h/10 epochs, 7.1 steps/s, no loss collapse) → eval.
Everything runs on the NAS (see `world_tokenizer/`).

**How to read the metrics (this is the whole story):**
- **task-id probe** (66-way, chance 0.015): **saturated** — the ImageNet warm-start already scores
  0.91, so it mostly measures drift *away from ImageNet appearance*, not real damage. Misleading.
- **RankMe** (label-free effective rank, max 768): representation *health* / collapse detector.
- **contact / force** (from F/T, scene-held-out, error-barred): *robot-relevant usefulness* — the
  thing we actually care about.

**LR matters — the scary early result was a too-hot LR, not intrinsic:**

| ckpt | task-id lin | task-id kNN | RankMe |
|------|-------------|-------------|--------|
| e0 (warm-start) | 0.908 | 0.814 | 300 |
| LR **2e-4** (hot) e10 | 0.717 | 0.309 | **158** ← rank collapse |
| LR **2e-5** (gentle) e6 | 0.744 | 0.320 | **285** ← healthy, no collapse |

Hot LR genuinely collapsed the rank (300→158); **lowering to 2e-5 fixes it** (RankMe ~285). The
remaining task-id drop is the saturated-metric artifact, not damage.

**Robot-relevant eval (5 scene-split seeds, mean±std) — the decider:**

| ckpt | contact-linear | contact-kNN | force-R² |
|------|----------------|-------------|----------|
| e0 | 0.683 ±0.011 | 0.639 ±0.013 | 0.028 ±0.036 |
| e6 (gentle) | 0.673 ±0.014 | 0.674 ±0.004 | −0.004 ±0.039 |
| e10 (hot) | 0.675 ±0.006 | 0.671 ±0.006 | 0.050 ±0.022 |

Contact is **flat** across recipes; **force-R² ≈ 0 for everyone** — a single RGB frame doesn't
encode force, and finetuning can't create signal that isn't in the data.

**Conclusion:** it's **not a data-quantity or LR problem** — more cfg3 video or lower LR won't
help; the ceiling is that video frames alone lack the robot-relevant signal. **Keep `e0`; Stage 1
is a confirmed no-op. Move to Stage 2** (add robot state / force), where finetuning has new signal.

**Eval lesson:** never trust one *saturated* metric — pair a **health** metric (RankMe) with a
**usefulness** metric on an *unsaturated, task-relevant* target (contact/force), with error bars.
Harness: `world_tokenizer/{eval_lejepa,contact_probe,robust_robot_eval}.py`.

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
