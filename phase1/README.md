# Phase 1 — video-only LeJEPA on RH20T cfg3

Everything runs from the NAS env. **First:** `source /mnt/nas/data/RH20T/env.sh`
(sets the venv, HF/pip/tmp caches, `PYTHONPATH=rh20t_api`, and a free GPU). `/` is full —
nothing here writes to `/`.

## Run order
```bash
source /mnt/nas/data/RH20T/env.sh
cd /root/ishneet/world-autoencoder

# 1. frames for ONE scene (the alignment gate uses these)
python -m phase1.extract_frames --scene task_0001_user_0016_scene_0001_cfg_0003

# 2. debug train: 1 scene, 1 GPU, n_local=0, a few steps
python -m phase1.train --frames-root /mnt/nas/data/RH20T/cfg3_frames/task_0001_user_0016_scene_0001_cfg_0003 \
    --epochs 3 --n-local 0 --max-steps 30

# 3. scale: extract all, then DDP on the 7 free GPUs
python -m phase1.extract_frames --all --num-workers 16
torchrun --nproc_per_node=7 -m phase1.train --frames-root /mnt/nas/data/RH20T/cfg3_frames --epochs 30

# 4. validate
python -m phase1.validate --frames-root /mnt/nas/data/RH20T/cfg3_frames --ckpt /mnt/nas/data/RH20T/phase1_ckpt.pt
```

## Pre-flight facts baked into the code
- **Loader:** `AutoModel + trust_remote_code` FAILS on `OK-AI/lejepa-vitb16-pretrain-in1k`.
  Use `snapshot_download` + `sys.path` + direct `ViTv2PretrainedModel` import (in `model.py`).
  Output is a dict; CLS = `out["latent"]` (768-d).
- **LeJEPA is timm-only**, so `model.py` composes our backbone + projector + the library's
  `SlicedEppsPulley`, reusing `LeJEPA._compute_loss` (loss is imported, not reimplemented).
- **`rh20t_api.extract` has no CLI** — `extract_frames.py` wraps `convert_scene`.

## Gate (do before trusting training)
Extract one scene, open a few frames, and plot F/T magnitude over the episode (via
`RH20TScene.get_ft_aligned`) — the spike must line up with the contact frame. Only then trust
the data. The `validate.py` contact-probe is the natural place to wire `get_ft_aligned` in.

## Notes
- BF16, no GradScaler (fp16-only). LR 2e-4 (fine-tune, not 2e-3). Don't freeze the backbone.
- `n_local=0` for the first run (all 224 crops). Local 96 crops need the ViT's
  variable-resolution path — enable once the 224-only run is clean.
- DDP×8 needs GPU 0 freed; otherwise use the 7 free GPUs.
- IO is the bottleneck off NFS — for the full DDP run, pack frames into WebDataset shards.
