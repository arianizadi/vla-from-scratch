#!/usr/bin/env bash
# Convert a collected LeRobot dataset (data/lerobot/<collection>) to training shards.
#   sim/pickup/scripts/preprocess.sh <collection> [name] [preprocess config]
# Produces data/preprocessed/<name>/shards/{manifest.jsonl,stats.json,*.tar}; name defaults to the collection's.
# The config (sim/pickup/configs/<config>.yaml, default preprocess) sets the windows, among them which past frames
# each sample carries: the same collection can be preprocessed again for a policy that looks further back
# (preprocess_1s: the frame 1 s back instead of the previous one).
# PICKUP_WORKERS sets the number of preprocessing CPUs (default 8); PICKUP_CAMERAS the cameras kept (default
# "['front', 'wrist']"; a twins dataset adds front_twin and wrist_twin). Environment: see _common.sh.
set -euo pipefail
collection=${1:?collection}; name=${2:-$1}; config=${3:-preprocess}
. "$(dirname "$0")/_common.sh"
workers=${PICKUP_WORKERS:-8}
[ -d "data/lerobot/$collection" ] || { echo "no data/lerobot/$collection" >&2; exit 1; }
rm -rf "data/preprocessed/$name"
"${foundry_python[@]}" vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type lerobot \
  --source_episodes "['$root/data/lerobot/$collection/']" \
  --output_dir "$root/data/preprocessed/$name/" \
  --camera_names "${PICKUP_CAMERAS:-['front', 'wrist']}" \
  --observation_keys "['state']" \
  --action_keys "['action']" \
  --samples_per_shard 256 \
  --ray_num_cpus "$workers" \
  --ray_address local \
  --config_path "sim/pickup/configs/$config.yaml" </dev/null 2>&1 | grep -vE "raylet|pid=" | tail -4
echo "dataset ready: data/preprocessed/$name/shards ($(wc -l < "data/preprocessed/$name/shards/manifest.jsonl") shards)"
