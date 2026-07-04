# world_tokenizer — LeJEPA on RH20T cfg3 (stages 0-1)

## Setup

Python deps (torch/torchvision, timm, stable-pretraining, webdataset, scikit-learn,
matplotlib, pillow) install with pip. One dependency is NOT pip-installable:

- **`rh20t_api`** — the official RH20T toolkit, used from source. Clone
  [github.com/rh20t/rh20t_api](https://github.com/rh20t/rh20t_api) and put its repo
  root on `PYTHONPATH` (it also ships `configs/configs.json`, which the gate/eval
  scripts locate automatically relative to the package).

```bash
git clone https://github.com/rh20t/rh20t_api /path/to/rh20t_api
export PYTHONPATH=/path/to/rh20t_api${PYTHONPATH:+:$PYTHONPATH}
export RH20T=/path/to/RH20T   # data root, laid out as below
```

Data layout (one dir per pipeline stage, one subdir per cfg):
```
$RH20T/
  raw/          RH20T_cfg1 … RH20T_cfg7   (untarred; patch merged into cfg1/2)
  frames/       cfg1/ … cfg7/             (extract_frames output)
  shards/       cfg1/ … cfg7/             (make_shards output; each has count.txt)
  checkpoints/  phase1_*.pt               (training output)
```

All scripts take explicit path flags; their *defaults* point at our NAS
(`/mnt/nas/data/RH20T/...`, same layout). On the menlo box,
`source /mnt/nas/data/RH20T/env.sh` sets the venv, HF/pip/tmp caches, `PYTHONPATH`,
and a free GPU — `/` is full, nothing here may write to `/`.

## Run order

Steps 1+3 for every cfg in one go: `preprocessing/preprocess_all.sh` (env vars
`RAW_ROOT`, `OUT_ROOT`, `CFGS`, `NUM_WORKERS`; resumable, skips finished cfgs).

```bash
# 1. extract frames — mp4 -> timestamped jpgs, all robot scenes
#    (--scene <name> for one; --include-human for human demos)
python -m preprocessing.extract_frames --raw-root $RH20T/raw/RH20T_cfg3 \
    --dest $RH20T/frames/cfg3 --all --num-workers 32

# 2. data gate — F/T <-> video alignment on one scene (sanity, NOT training)
python -m preprocessing.gate --scene task_0001_user_0016_scene_0001_cfg_0003 \
    --raw-root $RH20T/raw/RH20T_cfg3 --frames-root $RH20T/frames/cfg3 --out .

# 3. pack frames into WebDataset shards (fast sequential NFS reads; writes count.txt)
python -m preprocessing.make_shards --frames-root $RH20T/frames/cfg3 \
    --out $RH20T/shards/cfg3 --num-workers 48

# 4. train — DDP via the VENV python (`-m torch.distributed.run`; the `torchrun` binary is
#    base-env and can't see venv packages). GPU 0 busy -> 1..7 (or 0..7 if free).
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=7 \
    -m world_tokenizer.train --shards $RH20T/shards/cfg3 --out $RH20T/checkpoints/phase1_ckpt.pt \
    --epochs 10 --lr 2e-5
#   debug on 1 scene/1 GPU:  --frames-root <scene_dir> --max-steps 30

# 5. eval — scene-held-out, health + robot-relevant (all take --ckpt-template for custom
#    paths; contact/robust also take --raw-root/--frames-root/--conf like the gate)
python -m world_tokenizer.eval_lejepa      --ckpts e0 e3 e6 e10   # linear + kNN + RankMe (task-id)
python -m world_tokenizer.contact_probe    --ckpts e0 e6          # contact from F/T (unsaturated)
python -m world_tokenizer.robust_robot_eval --ckpts e0: e6:<path> --seeds 5  # +force-R2, error bars
#    encoder-only metrics (no predictor): metrics/metrics.py — design notes in metrics/METRICS.md
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

## Notes / findings
- BF16, no GradScaler (fp16-only). AdamW.
- **LR:** 2e-4 is too hot for warm-start → rank collapse (RankMe 300→158). Use **~2e-5**.
- **Result (stages 0-1):** video-only finetune is a **no-op** on robot-relevant signal (contact
  flat, force-R²≈0); task-id drop was a *saturated-metric artifact*. Keep `e0`; go to Stage 2.
- **Eval:** don't trust one saturated metric — pair **RankMe** (health) with an *unsaturated,
  robot-relevant* probe (contact/force), scene-held-out, multi-seed. See `robust_robot_eval.py`.
- `n_local=0` used (2×224 global crops); local 96 crops available but untested at scale.
- DDP×8 needs GPU 0 free; else 1..7. IO bottleneck off NFS → WebDataset shards (`make_shards`).
