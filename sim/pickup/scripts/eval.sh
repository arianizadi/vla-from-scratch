#!/usr/bin/env bash
# Closed-loop evaluation: start the policy server, run the simulator client against it, then shut the
# server down.
#   sim/pickup/scripts/eval.sh <experiment_dir> <task> [episodes] [extra eval.py args...]
# Environment variables:
#   PICKUP_CHECKPOINT=<path to .pt>  evaluate a specific checkpoint instead of the latest one
#   PICKUP_WEIGHTS=raw|ema           trained or averaged weights of a run trained with EMA (default: EMA)
#   PICKUP_OUT=<dir>                 where results.json and the videos go
#   PICKUP_PORT=<port>               socket port; give each concurrent evaluation its own
# Environment and GPU selection: see _common.sh.
set -euo pipefail
exp=${1:?experiment dir}; task=${2:?task}; episodes=${3:-50}; shift $(( $# < 3 ? $# : 3 ))
. "$(dirname "$0")/_common.sh"
port=${PICKUP_PORT:-6200}
out=${PICKUP_OUT:-results/pickup/$(basename "$exp")/$task}
mkdir -p "$out"
"${foundry_python[@]}" -m sim.pickup.policy_server --checkpoint_dir "$exp" --port "$port" \
  ${PICKUP_CHECKPOINT:+--checkpoint "$PICKUP_CHECKPOINT"} ${PICKUP_WEIGHTS:+--weights "$PICKUP_WEIGHTS"} \
  </dev/null >"$out/policy_server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
"${sim_python[@]}" -m sim.pickup.eval --task "$task" --episodes "$episodes" --port "$port" --out "$out" \
  --shutdown_server "$@"
wait $server || true
trap - EXIT
