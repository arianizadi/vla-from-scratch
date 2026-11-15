#!/usr/bin/env bash
# Closed-loop evaluation of a two-level reasoning policy (tutorial/reasoning-vla.md): a reasoner (a reason_then_act run
# trained without the action loss, e.g. qwen_reasoner.yaml) writes a note every chunk, and the fast policy (e.g.
# dp_vit_lang_reactive_subgoal.yaml) reads the note's sub-goal as its instruction. Starts both policy servers, runs
# eval.py with --subgoals reasoner, shuts both down.
#   sim/pickup/scripts/eval_two_level.sh <fast_experiment_dir> <reasoner_experiment_dir> <task> [episodes] [eval.py args...]
# Environment variables:
#   PICKUP_CHECKPOINT / PICKUP_WEIGHTS                    the fast policy's checkpoint and weights (as for eval.sh)
#   PICKUP_REASONER_CHECKPOINT / PICKUP_REASONER_WEIGHTS  the reasoner's
#   PICKUP_OUT=<dir>   where results.json goes (default results/pickup/<fast run>/<task>_two_level_<reasoner run>)
# Environment and GPU selection: see _common.sh (both servers run on the same GPU).
set -euo pipefail
fast=${1:?fast policy experiment dir}; reasoner=${2:?reasoner experiment dir}; task=${3:?task}; episodes=${4:-50}
shift $(( $# < 4 ? $# : 4 ))
. "$(dirname "$0")/_common.sh"
out=${PICKUP_OUT:-results/pickup/$(basename "$fast")/${task}_two_level_$(basename "$reasoner")}
mkdir -p "$out"
free_port() { "${sim_python[@]}" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
port=$(free_port); rport=$(free_port)
while [ "$rport" = "$port" ]; do rport=$(free_port); done
"${foundry_python[@]}" -m sim.pickup.policy_server --checkpoint_dir "$fast" --port "$port" \
  ${PICKUP_CHECKPOINT:+--checkpoint "$PICKUP_CHECKPOINT"} ${PICKUP_WEIGHTS:+--weights "$PICKUP_WEIGHTS"} \
  </dev/null >"$out/policy_server.log" 2>&1 &
server=$!
"${foundry_python[@]}" -m sim.pickup.policy_server --checkpoint_dir "$reasoner" --port "$rport" \
  ${PICKUP_REASONER_CHECKPOINT:+--checkpoint "$PICKUP_REASONER_CHECKPOINT"} \
  ${PICKUP_REASONER_WEIGHTS:+--weights "$PICKUP_REASONER_WEIGHTS"} </dev/null >"$out/reasoner_server.log" 2>&1 &
rserver=$!
trap 'kill $server $rserver 2>/dev/null || true' EXIT
for log in "$out/policy_server.log" "$out/reasoner_server.log"; do
  until grep -qs "listening on" "$log"; do
    if ! kill -0 $server 2>/dev/null || ! kill -0 $rserver 2>/dev/null; then
      echo "a policy server exited before it was ready; last lines of $log:" >&2
      tail -n 5 "$log" >&2
      exit 1
    fi
    sleep 1
  done
done
"${sim_python[@]}" -m sim.pickup.eval --task "$task" --episodes "$episodes" --port "$port" --out "$out" \
  --expect_run "$(basename "$fast")" --subgoals reasoner --reasoner_port "$rport" \
  --expect_reasoner_run "$(basename "$reasoner")" --shutdown_server "$@"
wait $server $rserver || true
trap - EXIT
