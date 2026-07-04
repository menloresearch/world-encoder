#!/bin/bash
# Stage 2 on cfg3+cfg4 (both UR5 -> same 28-dim state): precompute patch cache, then train.
# Uses the repo's uv .venv — do NOT source env.sh (its wae-venv is root-only).
# Artifacts: $RH20T/checkpoints/exp-<datetime>/ (per-seed perceiver weights + run.log).
set -e
export HF_HOME=/mnt/nas/data/RH20T/hf_cache
export TMPDIR=/dev/shm/wae_tmp
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
mkdir -p "$TMPDIR"
cd "$(dirname "$0")"

PY=/home/menlo/brain/world-encoder/.venv/bin/python
CACHE=/dev/shm/wae_tmp/mm_patch_cfg34.npz
EXP=/mnt/nas/data/RH20T/checkpoints/exp-$(date +%Y%m%d-%H%M%S)
mkdir -p "$EXP"
exec > >(tee -a "$EXP/run.log") 2>&1
echo "experiment dir: $EXP"

echo "=== precompute cfg3+cfg4 (per-scene 30 = the 24k-frame cfg3 POC density) $(date -u) ==="
if [ ! -f "$CACHE" ]; then
    $PY -m world_tokenizer.precompute_patch --cfgs 3 4 --per-scene 30 --out "$CACHE"
    cp "$CACHE" /mnt/nas/data/RH20T/tmp/mm_patch_cfg34.npz  # tmpfs is volatile; durable copy
else
    echo "cache exists, skipping precompute: $CACHE"
fi

echo "=== train_perceiver on combined cache $(date -u) ==="
$PY -m world_tokenizer.train_perceiver --cache "$CACHE" --seeds 5 --out-dir "$EXP"
echo "=== DONE $(date -u) | artifacts in $EXP ==="
