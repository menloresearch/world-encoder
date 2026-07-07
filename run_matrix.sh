#!/usr/bin/env bash
# Phase-1 transfer matrix: 5 training runs (4 specialists + ALL), one per GPU.
# Each run: train MMPerceiverChunks on --train-cfgs (5 seeds x 40 ep), then eval the
# frozen vision-only z_v on every embodiment's held-out groups -> a row of the 5x4
# transfer matrix. Results + checkpoints under $OUT/<tag>/.
set -uo pipefail
cd /home/menlo/brain/ishneet/we-phase1
OUT=/mnt/nas/data/RH20T/checkpoints/phase1
mkdir -p "$OUT" logs

run() {  # <gpu> <tag> <cfgs...>
  local gpu=$1 tag=$2; shift 2
  echo "launching '$tag' (cfgs $*) on GPU$gpu -> logs/train_$tag.log"
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m world_tokenizer.train_chunks \
      --train-cfgs "$@" --tag "$tag" --seeds 5 --epochs 40 \
      --out-dir "$OUT" > "logs/train_$tag.log" 2>&1 &
}

run 0 flexiv 1 2
run 1 ur5    3 4
run 2 franka 5
run 3 kuka   6 7
run 4 all    1 2 3 4 5 6 7
wait
echo "=== ALL TRAINING RUNS DONE $(date -u) ==="
