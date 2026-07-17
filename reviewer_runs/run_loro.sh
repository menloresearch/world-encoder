#!/usr/bin/env bash
# Leave-One-Robot-Out (Q3): train ALL on 3 robots, hold one out. The existing trainer
# already evals on EVERY embodiment, so the held-out robot's cell (an unseen embodiment
# for that encoder) is in each run's results.json for free. No code change.
# Robot -> cfgs:  flexiv=1,2  ur5=3,4  franka=5  kuka=6,7
set -uo pipefail
cd /home/menlo/brain/ishneet/world-encoder
OUT=/mnt/nas/data/RH20T/checkpoints/reviewer/loro
LOG=/home/menlo/brain/ishneet/world-encoder/reviewer_runs/logs
mkdir -p "$OUT" "$LOG"

run() {  # <gpu> <tag> <train-cfgs...>
  local gpu=$1 tag=$2; shift 2
  echo "launching $tag on GPU$gpu (train cfgs $*) -> $LOG/$tag.log"
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m world_tokenizer.train_chunks \
      --train-cfgs "$@" --tag "$tag" --seeds 5 --epochs 40 --out-dir "$OUT" \
      > "$LOG/$tag.log" 2>&1 &
}

run 1 loro_no_flexiv 3 4 5 6 7
run 2 loro_no_ur5    1 2 5 6 7
run 3 loro_no_franka 1 2 3 4 6 7
run 4 loro_no_kuka   1 2 3 4 5
wait
echo "=== LORO ALL DONE $(date -u) ==="
