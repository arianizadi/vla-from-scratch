#!/usr/bin/env bash
# train.sh in a retry loop: if the run dies or hangs (this box has a faulty memory line; a dead
# dataloader worker leaves the trainer waiting forever), restart it from the newest checkpoint until the
# sample budget is reached. Same arguments as train.sh.
#   sim/pickup/scripts/train_resilient.sh <config> <dataset> <run_name> <total_train_samples> [extra args...]
set -uo pipefail
cd "$(dirname "$0")/../../.."
name=${3:?run name}
log=experiments/pickup/$name.train.log
stall=${PICKUP_STALL_SECONDS:-420}
for attempt in $(seq 1 "${PICKUP_MAX_ATTEMPTS:-20}"); do
  [ -f "$log" ] && mv "$log" "$log.part$(date +%s)"
  latest=$(ls experiments/pickup/"$name"/checkpoints/checkpoint_*.pt 2>/dev/null | sort -V | tail -1)
  sim/pickup/scripts/train.sh "$@" ${latest:+--model.resume_from_checkpoint "$latest"} &
  job=$!
  while kill -0 $job 2>/dev/null; do
    sleep 30
    if [ -f "$log" ] && [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt "$stall" ]; then
      echo "attempt $attempt of $name stalled; killing" >> experiments/pickup/"$name".retries.log
      docker ps --no-trunc --format '{{.ID}} {{.Command}}' | grep -- "--name $name " | cut -d' ' -f1 | xargs -r docker kill >/dev/null
    fi
  done
  grep -q "Model has seen the desired number of samples" "$log" && exit 0
  echo "attempt $attempt of $name ended at $(date -u +%H:%M) with $(ls experiments/pickup/"$name"/checkpoints | grep -c checkpoint_) checkpoints" >> experiments/pickup/"$name".retries.log
  sleep 10
done
exit 1
