#!/usr/bin/env bash
# train.sh in a retry loop: if the run dies or hangs, restart it from the newest checkpoint until the
# sample budget is reached. Same arguments as train.sh.
#   sim/pickup/scripts/train_resilient.sh <config> <dataset> <run_name> <total_train_samples> [extra args...]
#
# Why both cases: a crashed run exits, but a data-loader worker that dies (out of memory, a hardware
# fault) leaves the trainer waiting for it forever at 0% GPU. A run whose log has not moved for
# PICKUP_STALL_SECONDS (default 420) is therefore killed and resumed as well.
set -uo pipefail
cd "$(dirname "$0")/../../.."
name=${3:?run name}
log=experiments/pickup/$name.train.log
stall=${PICKUP_STALL_SECONDS:-420}
mkdir -p experiments/pickup
for attempt in $(seq 1 "${PICKUP_MAX_ATTEMPTS:-20}"); do
  [ -f "$log" ] && mv "$log" "$log.part$(date +%s)"
  latest=$(ls experiments/pickup/"$name"/checkpoints/checkpoint_*.pt 2>/dev/null | sort -V | tail -1)
  # Own session, so that the whole process tree (trainer, data-loader workers) can be killed at once.
  setsid sim/pickup/scripts/train.sh "$@" ${latest:+--model.resume_from_checkpoint "$latest"} &
  job=$!
  while kill -0 $job 2>/dev/null; do
    sleep 30
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
