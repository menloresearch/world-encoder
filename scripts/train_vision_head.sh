#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# Vision-only LeJEPA HEAD baseline — the properly-matched control to the
# multimodal Perceiver (which also freezes the ViT and trains a ~2M-param head).
#
# WHAT THIS DOES
#   Step 1: train a FROZEN ViT + trainable proj_v (768->256) head with vision-only
#           LeJEPA (multi-crop, SIGReg + inv loss) on the SAME holdout TRAIN split
#           the multimodal encoder used (all 7 cfgs), via the holdout-aware --cfgs
#           data path. Only proj_v (~200k params) trains.  -> vision_head_all.pt
#   Step 2: full probe eval on all 4 embodiments' held-out groups, comparing
#           head-ft (256) vs raw ViT (768) vs PCA-256 vs (offline) the multimodal
#           z_v — motor/ee ridge R2 + RankMe.  -> vision_head_all/vision_eval.json
#
# EXPECTED RUNTIME (rough, NFS-IO bound)
#   Step 1: ~10-13 min/epoch/GPU at --per-scene 30 (267k train frames); default
#           12 epochs. Head is tiny, so higher LR (1e-3) and more epochs are cheap.
#   Step 2: ~30-45 min (loads each embodiment's cache + encodes all held-out frames).
#
# SAFE TO RE-RUN: Step 1 overwrites its own checkpoints; Step 2 overwrites its JSON.
# Does NOT touch the full-finetune code/checkpoints or the user's phase1_abl runs.
# =============================================================================

# ---- env (edit CUDA_VISIBLE_DEVICES to free GPUs; GPU 0 is often busy) --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # e.g. "1,2,3" for 3-GPU DDP
VENV_PY=/home/menlo/brain/world-encoder/.venv/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"   # ensure `python -m world_tokenizer...` imports THIS worktree's code (has --head-only), not the main checkout's
export PYTHONPATH="${REPO_ROOT}:/mnt/nas/data/RH20T/deps/rh20t_api${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/mnt/nas/data/RH20T/hf_cache
export TMPDIR=/dev/shm/wae_tmp
mkdir -p "$TMPDIR"

CKPT_DIR=/mnt/nas/data/RH20T/checkpoints/phase1_vision
mkdir -p "$CKPT_DIR"
CKPT="${CKPT_DIR}/vision_head_all.pt"

# ---- training hyperparams (head-only: safe to push LR/epochs) -----------------
CFGS="1 2 3 4 5 6 7"     # all embodiments' TRAIN split (per splits/holdout_v1.csv)
PER_SCENE=30             # frames/scene cap (bounds NFS small-file IO)
EPOCHS=12
LR=1e-3                  # tiny head -> high LR is fine (vs 2e-5 for full finetune)
HEAD_DIM=256             # proj_v output dim -> directly comparable to PCA-256 / z_v
BATCH=128               # per-GPU
WORKERS=12
ENC_BS=256

# derive DDP world size from CUDA_VISIBLE_DEVICES
IFS=',' read -ra _GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#_GPUS[@]}
EVAL_GPU="${_GPUS[0]}"   # eval is single-GPU

echo "=============================================================="
echo "[vision-head] GPUs=$CUDA_VISIBLE_DEVICES  NPROC=$NPROC  eval-gpu=$EVAL_GPU"
echo "[vision-head] ckpt -> $CKPT"
echo "=============================================================="

# ---- Step 1: train the ALL head ---------------------------------------------
echo "[vision-head] STEP 1/2: training frozen-ViT + proj_v head (all cfgs, train split)..."
TRAIN_ARGS=(-m world_tokenizer.train
  --head-only --head-dim "$HEAD_DIM"
  --cfgs $CFGS --split train --per-scene "$PER_SCENE"
  --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH"
  --n-local 0 --num-workers "$WORKERS"
  --out "$CKPT")

if [ "$NPROC" -gt 1 ]; then
  "$VENV_PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29541 "${TRAIN_ARGS[@]}"
else
  "$VENV_PY" "${TRAIN_ARGS[@]}"
fi
echo "[vision-head] STEP 1 done. Per-epoch checkpoints: ${CKPT%.pt}_e*.pt ; final: $CKPT"

# ---- Step 2: full probe eval on all 4 embodiments ---------------------------
echo "[vision-head] STEP 2/2: full probe eval (head-256 vs raw-768 vs PCA-256)..."
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV_PY" -m world_tokenizer.train_chunks \
  --vision-ckpt "$CKPT" \
  --tag vision_head_all --out-dir "$CKPT_DIR" \
  --enc-bs "$ENC_BS" --workers "$WORKERS"     # --eval-max 0 (all records) by default

echo "=============================================================="
echo "[vision-head] DONE."
echo "  checkpoint : $CKPT"
echo "  comparison : ${CKPT_DIR}/vision_head_all/vision_eval.json"
echo "  compare vs the multimodal ALL z_v in /mnt/nas/data/RH20T/checkpoints/phase1/all/results.json"
echo "=============================================================="

# =============================================================================
# PER-EMBODIMENT VARIANTS (optional; matches train_chunks EMBODIMENTS). Run after,
# or instead of, the ALL head. Uncomment a block and re-run this script's env first.
#
# for spec in "flexiv 1 2" "ur5 3 4" "franka 5" "kuka 6 7"; do
#   read -r name cfgs <<< "$spec"
#   OUT="${CKPT_DIR}/vision_head_${name}.pt"
#   echo "[vision-head] training ${name} head (cfgs ${cfgs})..."
#   ARGS=(-m world_tokenizer.train --head-only --head-dim "$HEAD_DIM" \
#         --cfgs ${cfgs} --split train --per-scene "$PER_SCENE" --epochs "$EPOCHS" \
#         --lr "$LR" --batch-size "$BATCH" --n-local 0 --num-workers "$WORKERS" --out "$OUT")
#   if [ "$NPROC" -gt 1 ]; then
#     "$VENV_PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29542 "${ARGS[@]}"
#   else
#     "$VENV_PY" "${ARGS[@]}"
#   fi
#   CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV_PY" -m world_tokenizer.train_chunks \
#     --vision-ckpt "$OUT" --tag "vision_head_${name}" --out-dir "$CKPT_DIR" \
#     --enc-bs "$ENC_BS" --workers "$WORKERS"
# done
# =============================================================================
