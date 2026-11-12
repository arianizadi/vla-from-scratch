#!/usr/bin/env bash
# Closed-loop evaluation: start the policy server, run the simulator client against it, then shut the
# server down.
#   sim/pickup/scripts/eval.sh <experiment_dir> <task> [episodes] [extra eval.py args...]
# Environment variables:
#   PICKUP_CHECKPOINT=<path to .pt>  evaluate a specific checkpoint instead of the latest one
#   PICKUP_WEIGHTS=raw|ema           trained or averaged weights of a run trained with EMA (default: EMA)
#   PICKUP_OUT=<dir>                 where results.json and the videos go
#   PICKUP_PORT=<port>               socket port (default: a free one, so evaluations can run side by side)
# The client refuses to run unless the server reports the checkpoint this evaluation asked for.
# Environment and GPU selection: see _common.sh.
set -euo pipefail
exp=${1:?experiment dir}; task=${2:?task}; episodes=${3:-50}; shift $(( $# < 3 ? $# : 3 ))
. "$(dirname "$0")/_common.sh"
out=${PICKUP_OUT:-results/pickup/$(basename "$exp")/$task}
mkdir -p "$out"
port=${PICKUP_PORT:-$("${sim_python[@]}" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}
expect=(--expect_run "$(basename "$exp")")
if [ -n "${PICKUP_CHECKPOINT:-}" ]; then
  number=$(basename "$PICKUP_CHECKPOINT" .pt)
  expect+=(--expect_number "${number##*_}")
fi
if [ -n "${PICKUP_WEIGHTS:-}" ]; then expect+=(--expect_weights "$PICKUP_WEIGHTS"); fi

: > "$out/policy_server.log"
"${foundry_python[@]}" -m sim.pickup.policy_server --checkpoint_dir "$exp" --port "$port" \
  ${PICKUP_CHECKPOINT:+--checkpoint "$PICKUP_CHECKPOINT"} ${PICKUP_WEIGHTS:+--weights "$PICKUP_WEIGHTS"} \
  </dev/null >"$out/policy_server.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
# Wait until the server has loaded its checkpoint. If it exits first (port taken, bad checkpoint), stop.
until grep -qs "listening on" "$out/policy_server.log"; do
  if ! kill -0 $server 2>/dev/null; then
    echo "policy server exited before it was ready; last lines of $out/policy_server.log:" >&2
    tail -n 5 "$out/policy_server.log" >&2
    exit 1
  fi
  sleep 1
done
"${sim_python[@]}" -m sim.pickup.eval --task "$task" --episodes "$episodes" --port "$port" --out "$out" \
  "${expect[@]}" --shutdown_server "$@"
wait $server || true
trap - EXIT
