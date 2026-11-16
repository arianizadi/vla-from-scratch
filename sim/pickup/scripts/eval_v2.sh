#!/usr/bin/env bash
# Closed-loop evaluation of architecture v2 (tutorial/architecture-v2.md, sim/pickup/slow_level.py): its slow level
# (a scene memory and a typed program, planned again at every look) gives the fast policy (e.g.
# dp_vit_lang_reactive_subgoal.yaml) its sub-goals as its instruction. Starts the fast policy's server and, when the
# eyes or the program writer are wanted, the slow server (slow_server.py); waits until both listen; runs eval.py with
# --slow_level v2; shuts both down.
#   sim/pickup/scripts/eval_v2.sh <fast_experiment_dir> <task> [episodes] [eval.py args...]
# Environment variables:
#   PICKUP_CHECKPOINT / PICKUP_WEIGHTS  the fast policy's checkpoint and weights (as for eval.sh)
#   PICKUP_SCENE=truth|eyes             what the scene memory takes: the simulator's truth (default) or the eyes'
#                                       percepts
#   PICKUP_EYES=<run dir>               the eyes (a train_eyes.py run), for PICKUP_SCENE=eyes
#   PICKUP_EYES_CHECKPOINT=<k>          their checkpoint number (default: the latest)
#   PICKUP_EYES_WEIGHTS=ema|raw         their averaged (default) or trained weights
#   PICKUP_PROGRAM=gold|writer          the program: the task's own (default), or the one a language model writes
#   PICKUP_WRITER=<model name or path>  that language model, for PICKUP_PROGRAM=writer
#   PICKUP_WRITER_WEIGHTS=<checkpoint>  a reason_then_act checkpoint whose language model is loaded over it
#   PICKUP_OUT=<dir>   where results.json goes (default results/pickup/<fast run>/<task>_v2_<scene>_<program>)
# The slow level's own settings are eval.py's: --slow_stable, --slow_p_min, --patience, --recover.
# Environment and GPU selection: see _common.sh (both servers run on the same GPU).
set -euo pipefail
fast=${1:?fast policy experiment dir}; task=${2:?task}; episodes=${3:-50}
shift $(( $# < 3 ? $# : 3 ))
. "$(dirname "$0")/_common.sh"
scene=${PICKUP_SCENE:-truth}; program=${PICKUP_PROGRAM:-gold}
case $scene in truth|eyes) ;; *) echo "PICKUP_SCENE must be truth or eyes" >&2; exit 1 ;; esac
case $program in gold|writer) ;; *) echo "PICKUP_PROGRAM must be gold or writer" >&2; exit 1 ;; esac
if [ "$scene" = eyes ] && [ -z "${PICKUP_EYES:-}" ]; then
  echo "PICKUP_SCENE=eyes needs PICKUP_EYES=<eyes run dir>" >&2; exit 1
fi
if [ "$program" = writer ] && [ -z "${PICKUP_WRITER:-}" ]; then
  echo "PICKUP_PROGRAM=writer needs PICKUP_WRITER=<language model>" >&2; exit 1
fi
out=${PICKUP_OUT:-results/pickup/$(basename "$fast")/${task}_v2_${scene}_${program}}
mkdir -p "$out"
expect=(--expect_run "$(basename "$fast")")
if [ -n "${PICKUP_CHECKPOINT:-}" ]; then
  number=$(basename "$PICKUP_CHECKPOINT" .pt)
  expect+=(--expect_number "${number##*_}")
fi
if [ -n "${PICKUP_WEIGHTS:-}" ]; then expect+=(--expect_weights "$PICKUP_WEIGHTS"); fi
free_port() { "${sim_python[@]}" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
port=$(free_port)

: > "$out/policy_server.log"
"${foundry_python[@]}" -m sim.pickup.policy_server --checkpoint_dir "$fast" --port "$port" \
  ${PICKUP_CHECKPOINT:+--checkpoint "$PICKUP_CHECKPOINT"} ${PICKUP_WEIGHTS:+--weights "$PICKUP_WEIGHTS"} \
  </dev/null >"$out/policy_server.log" 2>&1 &
server=$!
pids=("$server"); logs=("$out/policy_server.log")
# The slow server only for what needs it: the eyes for the learned scene, the writer for written programs
sport=""; sserver=""
if [ "$scene" = eyes ] || [ "$program" = writer ]; then
  sport=$(free_port)
  while [ "$sport" = "$port" ]; do sport=$(free_port); done
  slow=(-m sim.pickup.slow_server --port "$sport")
  if [ "$scene" = eyes ]; then
    slow+=(--eyes "$PICKUP_EYES")
    if [ -n "${PICKUP_EYES_CHECKPOINT:-}" ]; then slow+=(--eyes_checkpoint "$PICKUP_EYES_CHECKPOINT"); fi
    if [ -n "${PICKUP_EYES_WEIGHTS:-}" ]; then slow+=(--eyes_weights "$PICKUP_EYES_WEIGHTS"); fi
  fi
  if [ "$program" = writer ]; then
    slow+=(--writer "$PICKUP_WRITER")
    if [ -n "${PICKUP_WRITER_WEIGHTS:-}" ]; then slow+=(--writer_weights "$PICKUP_WRITER_WEIGHTS"); fi
  fi
  : > "$out/slow_server.log"
  "${foundry_python[@]}" "${slow[@]}" </dev/null >"$out/slow_server.log" 2>&1 &
  sserver=$!
  pids+=("$sserver"); logs+=("$out/slow_server.log")
fi
trap 'kill $server $sserver 2>/dev/null || true' EXIT
# Wait until each server has loaded what it serves. If one exits first (port taken, bad checkpoint), stop.
for log in "${logs[@]}"; do
  until grep -qs "listening on" "$log"; do
    for i in "${!pids[@]}"; do
      if ! kill -0 "${pids[$i]}" 2>/dev/null; then
        echo "a server exited before it was ready; last lines of ${logs[$i]}:" >&2
        tail -n 5 "${logs[$i]}" >&2
        exit 1
      fi
    done
    sleep 1
  done
done
"${sim_python[@]}" -m sim.pickup.eval --task "$task" --episodes "$episodes" --port "$port" --out "$out" \
  "${expect[@]}" --slow_level v2 --scene "$scene" --program "$program" ${sport:+--slow_port "$sport"} \
  --shutdown_server "$@"
wait $server $sserver || true
trap - EXIT
