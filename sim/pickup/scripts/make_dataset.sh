#!/usr/bin/env bash
# Collect scripted demonstrations in the simulator and convert them to Foundry webdataset shards.
#   sim/pickup/scripts/make_dataset.sh <task> <episodes> [name]
# Produces data/lerobot/<name> (LeRobot v2.0) and data/preprocessed/<name>/shards/{manifest.jsonl,stats.json,*.tar}.
set -euo pipefail
cd "$(dirname "$0")/../../.."
task=${1:?task}; episodes=${2:?episodes}; name=${3:-$task}
sim_python=${PICKUP_SIM_PYTHON:-python3}
rm -rf "data/lerobot/$name" "data/preprocessed/$name"
"$sim_python" -m sim.pickup.collect --task "$task" --episodes "$episodes" --out "data/lerobot/$name" \
  --workers "${PICKUP_WORKERS:-8}"
vla-foundry python vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py \
  --type lerobot \
  --source_episodes "['/workspace/vla_foundry/data/lerobot/$name/']" \
  --output_dir "/workspace/vla_foundry/data/preprocessed/$name/" \
  --camera_names "['front', 'wrist']" \
  --observation_keys "['state']" \
  --action_keys "['action']" \
  --samples_per_shard 256 \
  --config_path sim/pickup/configs/preprocess.yaml </dev/null 2>&1 | grep -vE "raylet|pid=" | tail -4
echo "dataset ready: data/preprocessed/$name/shards ($(cat "data/preprocessed/$name/shards/manifest.jsonl" | wc -l) shards)"
