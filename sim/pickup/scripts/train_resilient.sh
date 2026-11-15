#!/usr/bin/env bash
# train.sh in a retry loop: if the run dies or hangs, restart it from the newest checkpoint until the
# sample budget is reached. Same arguments as train.sh.
#   sim/pickup/scripts/train_resilient.sh <config> <dataset> <run_name> <total_train_samples> [extra args...]
#
# Why both cases: a crashed run exits, but a data-loader worker that dies (out of memory, a hardware
# fault) leaves the trainer waiting for it forever at 0% GPU. A run whose log has not moved for
# PICKUP_STALL_SECONDS (default 420) is therefore killed and resumed as well. The run is looked at
# every PICKUP_POLL_SECONDS (default 30). A checkpoint being written counts as activity: the trainer logs
# nothing while it saves, and on a busy disk a save can take longer than the stall limit.
#
# A run is resumed from its newest whole checkpoint: checkpoint_N.pt, and optimizer_N.pt and ema_N.pt
# where the run writes such files, all complete archives. A checkpoint with a file cut short (a run killed
# while saving) is moved to checkpoints/incomplete/, with any evaluations of it (results/pickup/<run>/*/
# ckpt_N_*), and the run goes on from the one before; the evaluations come again once it is rewritten.
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
ckdir=experiments/pickup/$name/checkpoints
whole() {  # a complete zip archive (torch.save's format) opens; one cut short has no end record
  python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).close()' "$1" 2>/dev/null
}
complete() {  # checkpoint $1: its files are all whole (optimizer and averaged weights where the run writes them)
  local kind
  whole "$ckdir/checkpoint_$1.pt" || return 1
  for kind in optimizer ema; do
    if ls "$ckdir"/${kind}_*.pt >/dev/null 2>&1; then whole "$ckdir/${kind}_$1.pt" || return 1; fi
  done
}
newest_complete() {  # prints the path of the newest whole checkpoint; sets aside newer ones that are not
  local n aside
  for n in $(ls "$ckdir"/checkpoint_*.pt 2>/dev/null | sed 's/.*checkpoint_\([0-9]*\)\.pt$/\1/' | sort -rn); do
    if complete "$n"; then echo "$ckdir/checkpoint_$n.pt"; return; fi
    aside="$ckdir/incomplete/${n}_$(date +%s)"
    mkdir -p "$aside"
    mv "$ckdir"/checkpoint_"$n".pt "$ckdir"/optimizer_"$n".pt "$ckdir"/ema_"$n".pt "$aside"/ 2>/dev/null
    for r in results/pickup/"$name"/*/ckpt_"$n"_*; do [ -e "$r" ] && mv "$r" "$r.incomplete_$(date +%s)"; done
    echo "checkpoint $n of $name has a file cut short; moved to $aside, resuming from an earlier one" \
      >> experiments/pickup/"$name".retries.log
  done
}
active() {  # seconds since the log or a checkpoint file last changed
  local newest
  newest=$(stat -c %Y "$log" "$ckdir"/* 2>/dev/null | sort -n | tail -1)
  echo $(( $(date +%s) - ${newest:-0} ))
}
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
  latest=$(newest_complete)
  # Own session, so that the whole process tree (trainer, data-loader workers) can be killed at once.
  # (resume_weights_only False: a fine-tuning run started with --model.resume_weights_only True continues where it
  # stopped, with its optimiser, schedule and data position, rather than starting over from its own checkpoint)
  setsid sim/pickup/scripts/train.sh "$@" ${latest:+--model.resume_from_checkpoint "$latest" --model.resume_weights_only False} &
  job=$!
  while kill -0 $job 2>/dev/null; do
    sleep "${PICKUP_POLL_SECONDS:-30}" &
    wait $!  # a wait, unlike a sleep in the foreground, is cut short by a signal, so that stop runs at once
    if [ -f "$log" ] && [ "$(active)" -gt "$stall" ]; then
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
