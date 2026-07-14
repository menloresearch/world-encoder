# PixNerd pixel-decode integration (world-encoder z_v -> robot frame)

Files here belong in a clone of https://github.com/MCG-NJU/PixNerd :
- `latent.py`            -> `src/models/conditioner/latent.py`  (feeds z_v as [B,1,256] condition)
- `robot_latent.py`      -> `src/data/dataset/robot_latent.py`  ((frame, z_v) dataset)
- `decode_ur5_128.yaml`  -> `configs_robot/decode_ur5_128.yaml` (small t2i DiT, PixelAE, plain flow-matching)

## Setup
1. `git clone https://github.com/MCG-NJU/PixNerd && cd PixNerd`
2. `uv venv --python 3.11 && uv pip install -r requirements.txt && uv pip install triton setuptools`
3. copy the 3 files above into place
4. build the (frame,z_v) manifest:
   `python -m world_tokenizer.precompute_decode --cfgs 3 4 --ckpt <all/seed0.pt> --out /mnt/nas/data/RH20T/decode/ur5`
5. train (1 GPU — DDP wedged this VM's driver, keep single-GPU):
   `CUDA_VISIBLE_DEVICES=0 python main.py fit --config configs_robot/decode_ur5_128.yaml --trainer.max_steps 50000`

Status: pipeline validated (20-step smoke trained, loss 1.31->1.13). Full run pending a GPU
reset/reboot (8-GPU DDP hung NCCL and wedged the driver 2026-07-08).

## Extra deps + gotchas (learned the hard way)
- `uv pip install triton setuptools matplotlib` (not in requirements; triton needed by torch.compile).
- 8-GPU DDP: set `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` — without it NCCL init hangs and wedges the
  driver on this VM. Probe with a tiny all-reduce before launching.
- Sampling: use the **online `denoiser.*`** weights, NOT `ema_denoiser.*` — EMA (decay 0.9999)
  is ~random at early checkpoints and decodes to pure noise. `sample_decode.py` does this.
