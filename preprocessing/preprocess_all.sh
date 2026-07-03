#!/usr/bin/env bash
# Preprocess RH20T configs for training: raw scenes -> jpg frames -> WebDataset shards.
#
#   RAW_ROOT/RH20T_cfg<N>/          (untarred RH20T; patch merged into cfg1/2)
#     -> OUT_ROOT/frames/cfg<N>/    (extract_frames; resumable per camera)
#     -> OUT_ROOT/shards/cfg<N>/    (make_shards; skipped if count.txt exists)
#
# Usage:
#   ./preprocessing/preprocess_all.sh                 # all cfgs, NAS defaults
#   CFGS="1 2" NUM_WORKERS=16 ./preprocessing/preprocess_all.sh
#   RAW_ROOT=/data/RH20T OUT_ROOT=/data/RH20T ./preprocessing/preprocess_all.sh
#
# Requires: the project venv active (or PYTHON=/path/to/python), and rh20t_api
# on PYTHONPATH — clone https://github.com/rh20t/rh20t_api and export
# PYTHONPATH=/path/to/rh20t_api (done automatically here if the NAS copy exists).
set -euo pipefail

RAW_ROOT="${RAW_ROOT:-/mnt/nas/data/RH20T/raw}"
OUT_ROOT="${OUT_ROOT:-/mnt/nas/data/RH20T}"
CFGS="${CFGS:-1 2 3 4 5 6 7}"
NUM_WORKERS="${NUM_WORKERS:-32}"

cd "$(dirname "$0")/.."  # repo root, so `python -m world_tokenizer.*` resolves

# pick a python: $PYTHON if set, else the repo venv, else whatever's on PATH
if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -x .venv/bin/python ]]; then PY=".venv/bin/python"
else PY=python3; fi

# rh20t_api is an undeclared source dep; fall back to the NAS copy if not importable
if ! "$PY" -c "import rh20t_api" 2>/dev/null; then
  NAS_API=/mnt/nas/data/RH20T/deps/rh20t_api
  if [[ -d "$NAS_API" ]]; then
    export PYTHONPATH="$NAS_API${PYTHONPATH:+:$PYTHONPATH}"
  fi
  "$PY" -c "import rh20t_api" 2>/dev/null || {
    echo "ERROR: rh20t_api not importable. Clone https://github.com/rh20t/rh20t_api"
    echo "       and export PYTHONPATH=/path/to/rh20t_api"
    exit 1
  }
fi

for n in $CFGS; do
  raw="$RAW_ROOT/RH20T_cfg${n}"
  frames="$OUT_ROOT/frames/cfg${n}"
  shards="$OUT_ROOT/shards/cfg${n}"

  if [[ ! -d "$raw" ]]; then
    echo "!!! cfg${n}: $raw not found, skipping"
    continue
  fi
  if [[ -f "$shards/count.txt" ]]; then
    echo "=== cfg${n}: shards already done ($(cat "$shards/count.txt") samples), skipping"
    continue
  fi

  echo "=== cfg${n}: extracting frames -> $frames ==="
  start=$SECONDS
  "$PY" -m world_tokenizer.extract_frames --raw-root "$raw" --dest "$frames" \
      --all --num-workers "$NUM_WORKERS"

  echo "=== cfg${n}: packing shards -> $shards ==="
  rm -rf "$shards"  # no count.txt = absent or partial; rebuild from scratch
  "$PY" -m world_tokenizer.make_shards --frames-root "$frames" --out "$shards" \
      --num-workers "$NUM_WORKERS"

  echo "=== cfg${n}: done in $(( (SECONDS - start) / 60 ))m ==="
done

echo "ALL CFGS DONE."
