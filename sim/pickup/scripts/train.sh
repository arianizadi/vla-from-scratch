#!/usr/bin/env bash
# Train a policy on a pickup-sim dataset (single GPU).
#   sim/pickup/scripts/train.sh <config> <dataset> <run_name> <total_train_samples> [extra main.py args...]
# e.g. sim/pickup/scripts/train.sh dp_vit pick_cube_1k dp_vit_pick_cube_1k 1500000
# Checkpoints land in experiments/pickup/<run_name>/checkpoints; 6 are written per run (override with
# PICKUP_NUM_CHECKPOINTS) so a crash never costs more than a sixth of the job. Resume by re-running the
# same command with --model.resume_from_checkpoint <path to checkpoint_N.pt>, or use train_resilient.sh,
# which does that by itself. Environment and GPU selection: see _common.sh.
# PICKUP_ABOUT="one line on what this run is for" is saved as experiments/pickup/<run_name>.about and
# shown next to the run's name by the monitors.
#
# Several datasets at once (co-training): name them joined by '+' (prune_2k+aim_8k). Each batch then draws from
# them in the proportions PICKUP_DATA_WEIGHTS gives (comma-separated, one per dataset; default equal), and the
# normalisation statistics are merged over all of them. PICKUP_STATS_DATASETS="a b ..." merges the statistics over
# those datasets instead: pretraining on one dataset and fine-tuning on another then normalise states and actions
# the same way (start the fine-tuning with --model.resume_from_checkpoint <checkpoint> --model.resume_weights_only True).
set -euo pipefail
cfg=${1:?config name}; dataset=${2:?dataset name}; name=${3:?run name}; samples=${4:?total samples}; shift 4
. "$(dirname "$0")/_common.sh"
IFS=+ read -r -a datasets <<< "$dataset"
read -r -a stats_from <<< "${PICKUP_STATS_DATASETS:-${datasets[*]}}"
list() { local out="" d; for d in "${@:2}"; do out+="${out:+, }'data/preprocessed/$d/shards/$1'"; done; echo "[$out]"; }
mix=()  # one weight and one modality per dataset when there are several (the config gives them for one)
if [ ${#datasets[@]} -gt 1 ]; then
  modalities=$(printf "'robotics', %.0s" "${datasets[@]}")
  mix=(--data.dataset_weighting "[${PICKUP_DATA_WEIGHTS:-$(printf '1.0,%.0s' "${datasets[@]}" | sed 's/,$//')}]"
    --data.dataset_modality "[${modalities%, }]")
fi
mkdir -p experiments/pickup
if [ -n "${PICKUP_ABOUT:-}" ]; then printf '%s\n' "$PICKUP_ABOUT" > "experiments/pickup/$name.about"; fi
"${foundry_python[@]}" vla_foundry/main.py \
  --config_path "sim/pickup/configs/$cfg.yaml" \
  --data.dataset_manifest "$(list manifest.jsonl "${datasets[@]}")" \
  --data.dataset_statistics "$(list stats.json "${stats_from[@]}")" \
  ${mix[@]+"${mix[@]}"} \
  --name "$name" \
  --total_train_samples "$samples" \
  --num_checkpoints "${PICKUP_NUM_CHECKPOINTS:-6}" \
  "$@" </dev/null 2>&1 | tr '\r' '\n' | grep --line-buffered -vE "step/s|^$" > "experiments/pickup/$name.train.log"
