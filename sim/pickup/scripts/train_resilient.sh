#!/usr/bin/env bash
# train.sh in a retry loop: if the run dies or hangs, restart it from the newest checkpoint until the
# sample budget is reached. Same arguments as train.sh.
#   sim/pickup/scripts/train_resilient.sh <config> <dataset> <run_name> <total_train_samples> [extra args...]
#
# Why both cases: a crashed run exits, but a data-loader worker that dies (out of memory, a hardware
# fault) leaves the trainer waiting for it forever at 0% GPU. A run whose log has not moved for
# PICKUP_STALL_SECONDS (default 420) is therefore killed and resumed as well. The run is looked at
# every PICKUP_POLL_SECONDS (default 30).
#
# To end a run on purpose, stop this script (Ctrl-C, or kill it): it stops the trainer with it and
# leaves experiments/pickup/<run_name>.stopped, which the monitors show as "stopped" instead of
# "stalled". Put a reason in that file if you like. Starting the run again removes it.
#
# One trainer per run: while this script runs it holds experiments/pickup/<run_name>.pid, and a second
# copy started with the same run name exits with status 2 instead of writing into the same directory.
set -uo pipefail
cd "$(dirname "$0")/../../.."
name=${3:?run name}
log=experiments/pickup/$name.train.log
stopped=experiments/pickup/$name.stopped
pidfile=experiments/pickup/$name.pid
stall=${PICKUP_STALL_SECONDS:-420}
mkdir -p experiments/pickup
claim() { ( set -o noclobber; echo $$ > "$pidfile" ) 2>/dev/null; }
if ! claim; then
  other=$(cat "$pidfile" 2>/dev/null)
  if [ -n "$other" ] && ps -p "$other" -o args= 2>/dev/null | grep -q "train_resilient.sh"; then
    echo "run $name is already being trained by process $other; stop that one first (or pick another name)" >&2
    exit 2
  fi
  rm -f "$pidfile"  # left behind by a copy that was killed outright
  claim || { echo "cannot write $pidfile" >&2; exit 2; }
fi
trap 'rm -f "$pidfile"' EXIT
job=
stop() {
  echo "stopped by hand on $(date -u '+%Y-%m-%d at %H:%M UTC')" > "$stopped"
  if [ -n "$job" ]; then
    kill -TERM -- -$job 2>/dev/null
    sleep 5
    kill -KILL -- -$job 2>/dev/null
  fi
  exit 130
}
trap stop INT TERM
rm -f "$stopped"
for attempt in $(seq 1 "${PICKUP_MAX_ATTEMPTS:-20}"); do
  [ -f "$log" ] && mv "$log" "$log.part$(date +%s)"
  latest=$(ls experiments/pickup/"$name"/checkpoints/checkpoint_*.pt 2>/dev/null | sort -V | tail -1)
  # Own session, so that the whole process tree (trainer, data-loader workers) can be killed at once.
  setsid sim/pickup/scripts/train.sh "$@" ${latest:+--model.resume_from_checkpoint "$latest"} &
  job=$!
  while kill -0 $job 2>/dev/null; do
    sleep "${PICKUP_POLL_SECONDS:-30}" &
    wait $!  # a wait, unlike a sleep in the foreground, is cut short by a signal, so that stop runs at once
    if [ -f "$log" ] && [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt "$stall" ]; then
      echo "attempt $attempt of $name stalled at $(date -u +%H:%M); killing" >> experiments/pickup/"$name".retries.log
      kill -TERM -- -$job 2>/dev/null
      sleep 15
      kill -KILL -- -$job 2>/dev/null
    fi
  done
  grep -q "Model has seen the desired number of samples" "$log" && exit 0
  checkpoints=$(ls experiments/pickup/"$name"/checkpoints 2>/dev/null | grep -c "^checkpoint_")
  echo "attempt $attempt of $name ended at $(date -u +%H:%M) with $checkpoints checkpoints" \
    >> experiments/pickup/"$name".retries.log
  sleep 10
done
exit 1
