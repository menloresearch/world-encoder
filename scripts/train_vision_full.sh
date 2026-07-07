#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# FULL-ViT gentle fine-tune (for the ATTENTION-MAP visualization).
#
# WHY THIS EXISTS
#   The head-only baseline (train_vision_head.sh) FREEZES the ViT, so its ViT
#   self-attention (pca_viz panel C) is identical to the pretrained backbone.
#   To visualize a robotics-ADAPTED attention map you must actually fine-tune the
#   backbone. This does a deliberately GENTLE full fine-tune (small LR) from the
#   pretrained LeJEPA ViT-B/16, on the SAME holdout TRAIN split, then evals it.
#
#   Stage-1 finding: LR 2e-4 collapses RankMe (300->158); 2e-5 is the safe ceiling.
#   We default to a small LR and few epochs — enough to shift attention, not wreck rank.
#   Watch the log: emb_std should stay well above 0 (no collapse).
#
# OUTPUT
#   /mnt/nas/data/RH20T/checkpoints/phase1_vision/vision_full_all.pt  (+ per-epoch _e*.pt)
#   Feed it to pca_viz to SEE the adapted attention/PCA:
#     python -m world_tokenizer.pca_viz --ckpt <encoder.pt> \
#        --scenes-csv splits/viz_scenes_default.csv \
#        --vit /mnt/nas/data/RH20T/checkpoints/phase1_vision/vision_full_all.pt \
#        --out pca_viz_out/vit_finetuned.png
#
# EXPECTED RUNTIME: ~10-13 min/epoch/GPU (full backbone; heavier than the head run).
# SAFE TO RE-RUN. Does NOT touch the user's phase1_abl runs or the head checkpoints.
# =============================================================================

# ---- env (edit CUDA_VISIBLE_DEVICES to free GPUs; GPU 0 is often busy) --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # e.g. "1,2,3" for 3-GPU DDP
VENV_PY=/home/menlo/brain/world-encoder/.venv/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"   # import THIS worktree's code (holdout-aware train.py), not the main checkout's
export PYTHONPATH="${REPO_ROOT}:/mnt/nas/data/RH20T/deps/rh20t_api${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/mnt/nas/data/RH20T/hf_cache
export TMPDIR=/dev/shm/wae_tmp
mkdir -p "$TMPDIR"

CKPT_DIR=/mnt/nas/data/RH20T/checkpoints/phase1_vision
mkdir -p "$CKPT_DIR"
CKPT="${CKPT_DIR}/vision_full_all.pt"

# ---- training hyperparams (FULL backbone -> keep LR SMALL) --------------------
CFGS="1 2 3 4 5 6 7"     # all embodiments' TRAIN split (per splits/holdout_v1.csv)
PER_SCENE=30             # frames/scene cap (bounds NFS small-file IO)
EPOCHS=8
LR=2e-5                  # SMALL: Stage-1 showed 2e-4 collapses RankMe; 2e-5 is the safe ceiling.
                         # Go 1e-5 for an even gentler shift. NOT the 1e-3 used for the tiny head.
BATCH=64                # per-GPU (full backbone -> smaller than the head run)
WORKERS=12
ENC_BS=256

IFS=',' read -ra _GPUS <<< "$CUDA_VISIBLE_DEVICES"
NPROC=${#_GPUS[@]}
EVAL_GPU="${_GPUS[0]}"

echo "=============================================================="
echo "[vision-full] GPUs=$CUDA_VISIBLE_DEVICES  NPROC=$NPROC  LR=$LR (gentle full finetune)"
echo "[vision-full] ckpt -> $CKPT"
echo "=============================================================="

# ---- Step 1: gentle full-ViT fine-tune (NO --head-only) ----------------------
echo "[vision-full] STEP 1/2: full-ViT LeJEPA fine-tune (all cfgs, train split, LR $LR)..."
TRAIN_ARGS=(-m world_tokenizer.train
  --cfgs $CFGS --split train --per-scene "$PER_SCENE"
  --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH"
  --n-local 0 --num-workers "$WORKERS"
  --out "$CKPT")
if [ "$NPROC" -gt 1 ]; then
  "$VENV_PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=29551 "${TRAIN_ARGS[@]}"
else
  "$VENV_PY" "${TRAIN_ARGS[@]}"
fi
echo "[vision-full] STEP 1 done. Per-epoch: ${CKPT%.pt}_e*.pt ; final: $CKPT"

# ---- Step 2: probe eval (full-ViT 768 vs raw-768 vs PCA-256) -----------------
echo "[vision-full] STEP 2/2: probe eval..."
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$VENV_PY" -m world_tokenizer.train_chunks \
  --vision-ckpt "$CKPT" \
  --tag vision_full_all --out-dir "$CKPT_DIR" \
  --enc-bs "$ENC_BS" --workers "$WORKERS"

echo "=============================================================="
echo "[vision-full] DONE."
echo "  checkpoint : $CKPT"
echo "  probe eval : ${CKPT_DIR}/vision_full_all/vision_eval.json"
echo "  viz attn   : python -m world_tokenizer.pca_viz --ckpt <enc.pt> \\"
echo "                 --scenes-csv splits/viz_scenes_default.csv --vit $CKPT --out pca_viz_out/vit_finetuned.png"
echo "=============================================================="
