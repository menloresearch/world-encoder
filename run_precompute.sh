#!/usr/bin/env bash
# Phase-1 precompute: build per-cfg chunk caches for all 7 cfgs, one cfg per GPU.
# Resumable (existing caches/cfgN.npz is skipped). Logs per cfg in logs/.
set -uo pipefail
cd /home/menlo/brain/ishneet/we-phase1
OUT=/mnt/nas/data/RH20T/caches
mkdir -p "$OUT" logs
for n in 1 2 3 4 5 6 7; do
  gpu=$((n-1))
  echo "launching cfg$n on GPU$gpu -> logs/precompute_cfg$n.log"
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python preprocessing/precompute_chunks.py \
      --cfgs $n --out-dir "$OUT" > logs/precompute_cfg$n.log 2>&1 &
done
wait
echo "=== ALL PRECOMPUTE DONE $(date -u) ==="
ls -la "$OUT"
