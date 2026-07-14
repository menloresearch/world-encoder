#!/usr/bin/env bash
# Phase-1 claims-protection ablations (1.6), one run per GPU.
#  - vision-only-TRAINED control on all 5 configs (isolates cross-modal gain): *_vo
#  - bottleneck size (d=128 / d=512) + joint-SIGReg off, on cfg3+4 (cheap) per PLAN.
# Compare vs the matrix (phase1/) reference runs. Results in phase1_abl/<tag>/.
set -uo pipefail
cd /home/menlo/brain/ishneet/we-phase1
OUT=/mnt/nas/data/RH20T/checkpoints/phase1_abl
mkdir -p "$OUT" logs
run() {  # <gpu> <tag> <flags> <cfgs...>
  local gpu=$1 tag=$2 flags=$3; shift 3
  echo "launching '$tag' (flags:$flags | cfgs $*) on GPU$gpu -> logs/abl_$tag.log"
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m world_tokenizer.train_chunks \
      $flags --train-cfgs "$@" --tag "$tag" --seeds 5 --epochs 40 \
      --out-dir "$OUT" > "logs/abl_$tag.log" 2>&1 &
}
# vision-only control — mirrors the 5 matrix runs
run 0 flexiv_vo "--vision-only" 1 2
run 1 ur5_vo    "--vision-only" 3 4
run 2 franka_vo "--vision-only" 5
run 3 kuka_vo   "--vision-only" 6 7
run 4 all_vo    "--vision-only" 1 2 3 4 5 6 7
# cheap ablations on cfg3+4 (ur5) — compare to the ur5 matrix run
run 5 ur5_d128  "--d 128" 3 4
run 6 ur5_d512  "--d 512" 3 4
run 7 ur5_nojsr "--no-joint-sigreg" 3 4
wait
echo "=== ALL ABLATIONS DONE $(date -u) ==="
