# Phase 1 — video-only LeJEPA on RH20T cfg3 (verified)

**Goal:** prove the data pipeline + a single-frame visual encoder *before* adding any other
modality. Green = loss converges, no collapse, linear probe beats chance → Phase 2.

**What it is:** *continue* LeJEPA (image-level SSL, DINOv2-style multi-crop) on cfg3 RGB frames,
warm-started from `OK-AI/lejepa-vitb16-pretrain-in1k` (ViT-B/16, 768-d), backbone **not** frozen.
**Video-only** — no depth/force/proprioception in training (force is used only in the data gate;
real multi-modal inputs come in Phase 2+).

## Data (verified)
- cfg3 = UR5 + WSG-50, fixed rig. **799 robot scenes, 66 tasks, 8 cams/scene.**
- Extracted RGB: **2,330,532 jpgs (640×360)** → packed into **240 WebDataset tar shards**.
- On the NAS (VM root disk is full): `/mnt/nas/data/RH20T/{cfg3_frames,cfg3_shards}`.

## Environment
`source /mnt/nas/data/RH20T/env.sh` — venv (reuses base torch 2.8 / CUDA / 8 GPUs) + transformers /
timm / lightning / webdataset; HF/pip caches + `TMPDIR=/dev/shm` keep everything off `/`.

## Run order (commands that work)
```bash
source /mnt/nas/data/RH20T/env.sh && cd /root/ishneet/world-autoencoder
# 1. extract frames (robot scenes; --include-human adds human demos)
python -m phase1.extract_frames --all --num-workers 32
# 2. data gate — F/T <-> video alignment on one scene (sanity, not training)
python -m phase1.gate --scene task_0001_user_0016_scene_0001_cfg_0003
# 3. pack frames into WebDataset shards (fast sequential NFS reads)
python -m phase1.make_shards --num-workers 48
# 4. train — DDP on the 7 free GPUs (GPU 0 is busy)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=7 \
    -m phase1.train --shards /mnt/nas/data/RH20T/cfg3_shards --epochs 10
# 5. validate (linear probe on CLS embeddings)
python -m phase1.validate --frames-root /mnt/nas/data/RH20T/cfg3_frames --ckpt <ckpt>.pt
```

## Things we verified the hard way
- **Loader:** `AutoModel(trust_remote_code=True)` FAILS (the checkpoint's custom repo uses absolute
  imports). Use `snapshot_download` + `sys.path` + direct `ViTv2PretrainedModel`. Output is a dict;
  CLS = `out["latent"]` (768-d).
- **LeJEPA is timm-only** → we compose our backbone + projector + the library's `SlicedEppsPulley`
  and reuse `LeJEPA._compute_loss` (there is no `SIGReg` symbol; loss imported, not reimplemented).
- **DDP launch:** venv `python -m torch.distributed.run` (the `torchrun` binary is the base env).
- **WebDataset 1.x** needs explicit `nodesplitter=split_by_node` under DDP; `resampled=True` + per-rank seed.
- **`rh20t_api.extract` has no CLI**; some cams lack `color.mp4` → skip them, don't assert.

## Status (measured)
- Data gate: frames decode, `tcp`/`ft` getters sane, force spike aligns with contact. ✅
- Throughput: **7.1 steps/s on 7 GPUs (~12 min/epoch, ~2 h for 10 epochs)** — ~9× vs loose jpgs.
- Training healthy: loss 5.5 → ~0.35, SIGReg 278 → ~16, `emb_std` rising (no collapse).

## Known issue (Phase 2 blocker)
- `get_joint_angles_aligned` / `get_gripper` (the *aligned* getters) hit a `None` bug in `rh20t_api`
  for some scenes; raw F/T + TCP work. Must fix before adding robot state.

## Next
- **Phase 2:** + robot state (aligned getters → `{rgb, joints, torque, tcp, gripper, ft}`) + fusion.
- **Phase 3:** + decoder (PixNerd), encoder frozen first.
