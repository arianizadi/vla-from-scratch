#!/usr/bin/env bash
# Train a policy on a pickup-sim dataset inside the Foundry container (single GPU).
#   sim/pickup/scripts/train.sh <config> <dataset> <run_name> <total_train_samples> [extra main.py args...]
# e.g. sim/pickup/scripts/train.sh dp_clip pick_cube dp_clip_pick_cube 300000
# Checkpoints land in experiments/pickup/<run_name>/checkpoints; 6 are written per run so a crash
# never costs more than a sixth of the job (resume with --model.resume_from_checkpoint).
set -euo pipefail
cd "$(dirname "$0")/../../.."
cfg=${1:?config name}; dataset=${2:?dataset name}; name=${3:?run name}; samples=${4:?total samples}; shift 4
shards=data/preprocessed/$dataset/shards
mkdir -p experiments/pickup
vla-foundry python vla_foundry/main.py \
  --config_path "sim/pickup/configs/$cfg.yaml" \
  --data.dataset_manifest "['$shards/manifest.jsonl']" \
  --data.dataset_statistics "['$shards/stats.json']" \
  --name "$name" \
  --total_train_samples "$samples" \
  --num_checkpoints 6 \
  "$@" </dev/null 2>&1 | tr '\r' '\n' | grep --line-buffered -vE "step/s|^$" > "experiments/pickup/$name.train.log"
