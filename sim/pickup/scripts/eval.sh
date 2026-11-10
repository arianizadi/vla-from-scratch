#!/usr/bin/env bash
# Closed-loop evaluation: start the policy server in the Foundry container, run the simulator
# client in the robotics environment, then shut the server down.
#   sim/pickup/scripts/eval.sh <experiment_dir> <task> [episodes] [extra eval.py args...]
set -euo pipefail
cd "$(dirname "$0")/../../.."
exp=${1:?experiment dir}; task=${2:?task}; episodes=${3:-50}; shift $(( $# < 3 ? $# : 3 ))
port=${PICKUP_PORT:-6200}
sim_python=${PICKUP_SIM_PYTHON:-python3}
out=${PICKUP_OUT:-results/pickup/$(basename "$exp")/$task}
mkdir -p "$out"
# PICKUP_CHECKPOINT=<path to .pt> evaluates a specific checkpoint instead of the latest one.
vla-foundry python -m sim.pickup.policy_server --checkpoint_dir "$exp" --port "$port" \
  ${PICKUP_CHECKPOINT:+--checkpoint "$PICKUP_CHECKPOINT"} \
  </dev/null >"$out/policy_server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
"$sim_python" -m sim.pickup.eval --task "$task" --episodes "$episodes" --port "$port" --out "$out" \
  --shutdown_server "$@"
wait $server || true
trap - EXIT
