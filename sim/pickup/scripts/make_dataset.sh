#!/usr/bin/env bash
# Collect scripted demonstrations in the simulator and convert them to training shards.
#   sim/pickup/scripts/make_dataset.sh <task> <episodes> [name] [extra collect.py args...]
# Produces data/lerobot/<name> (LeRobot v2.0) and data/preprocessed/<name>/shards/{manifest.jsonl,stats.json,*.tar}.
# PICKUP_WORKERS sets the number of simulator processes and of preprocessing CPUs (default 8).
# Environment and GPU selection: see _common.sh.
set -euo pipefail
task=${1:?task}; episodes=${2:?episodes}; name=${3:-$task}; shift $(( $# < 3 ? $# : 3 ))
. "$(dirname "$0")/_common.sh"
workers=${PICKUP_WORKERS:-8}
rm -rf "data/lerobot/$name" "data/preprocessed/$name"
"${sim_python[@]}" -m sim.pickup.collect --task "$task" --episodes "$episodes" --out "data/lerobot/$name" \
  --workers "$workers" "$@"
PICKUP_WORKERS=$workers sim/pickup/scripts/preprocess.sh "$name"
