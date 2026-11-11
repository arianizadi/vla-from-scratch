#!/usr/bin/env bash
# Train a policy on a pickup-sim dataset (single GPU).
#   sim/pickup/scripts/train.sh <config> <dataset> <run_name> <total_train_samples> [extra main.py args...]
# e.g. sim/pickup/scripts/train.sh dp_vit pick_cube_1k dp_vit_pick_cube_1k 1500000
# Checkpoints land in experiments/pickup/<run_name>/checkpoints; 6 are written per run (override with
# PICKUP_NUM_CHECKPOINTS) so a crash never costs more than a sixth of the job. Resume by re-running the
# same command with --model.resume_from_checkpoint <path to checkpoint_N.pt>, or use train_resilient.sh,
# which does that by itself. Environment and GPU selection: see _common.sh.
set -euo pipefail
cfg=${1:?config name}; dataset=${2:?dataset name}; name=${3:?run name}; samples=${4:?total samples}; shift 4
. "$(dirname "$0")/_common.sh"
shards=data/preprocessed/$dataset/shards
mkdir -p experiments/pickup
"${foundry_python[@]}" vla_foundry/main.py \
  --config_path "sim/pickup/configs/$cfg.yaml" \
  --data.dataset_manifest "['$shards/manifest.jsonl']" \
  --data.dataset_statistics "['$shards/stats.json']" \
  --name "$name" \
  --total_train_samples "$samples" \
  --num_checkpoints "${PICKUP_NUM_CHECKPOINTS:-6}" \
  "$@" </dev/null 2>&1 | tr '\r' '\n' | grep --line-buffered -vE "step/s|^$" > "experiments/pickup/$name.train.log"
