#!/usr/bin/env bash
# Evaluate every checkpoint of a run closed-loop, so that success can be plotted against training.
#   sim/pickup/scripts/eval_checkpoints.sh <experiment_dir> <task> [episodes]
# Results go to results/pickup/<run>/<task>/ckpt_<N>_<raw|ema>/results.json. Runs that saved averaged
# weights (the configs here switch them on) get both their trained (raw) and averaged (ema) weights evaluated. The newest
# checkpoint is evaluated first. A checkpoint that already has results is skipped, so the script can
# be re-run at any time; summarize.py turns the results into one table.
#
# Environment variables (plus those of eval.sh and _common.sh):
#   PICKUP_EVAL_JOBS=<n>  evaluate n checkpoints at a time (default 1). Each takes a free port and
#                         mostly CPU: one simulator process each.
#   PICKUP_WATCH=1        keep polling for new checkpoints until the training log reports the end of
#                         the run. Start it next to train_resilient.sh to evaluate while training.
#   PICKUP_VIDEOS=<n>     rollout videos to save per checkpoint (default 0)
set -uo pipefail
exp=${1:?experiment dir}; task=${2:?task}; episodes=${3:-100}
cd "$(dirname "$0")/../../.."
name=$(basename "$exp")
log=experiments/pickup/$name.train.log
unset PICKUP_PORT  # every evaluation takes its own free port (see eval.sh)

if [ "${PICKUP_EVAL_JOBS:-1}" -gt 1 ] && [ -z "${PICKUP_EVAL_WORKER:-}" ]; then
  # Fan out: every worker runs this script; they share the work through the claim below.
  for slot in $(seq 1 "$PICKUP_EVAL_JOBS"); do
    PICKUP_EVAL_WORKER=1 "$0" "$exp" "$task" "$episodes" &
  done
  wait
  exit 0
fi

stale=${PICKUP_STALE_SECONDS:-1800}
age() { echo $(( $(date +%s) - $(stat -c %Y "$1") )); }
while true; do
  evaluated=0
  for ckpt in $(ls "$exp"/checkpoints/checkpoint_*.pt 2>/dev/null | sort -rV); do
    n=$(basename "$ckpt" .pt); n=${n#checkpoint_}
    for weights in ema raw; do
      file=$ckpt
      [ "$weights" = ema ] && file=$exp/checkpoints/ema_$n.pt
      [ -f "$file" ] || continue
      [ "$(age "$file")" -ge 60 ] || continue  # may still be being written
      out=results/pickup/$name/$task/ckpt_${n}_$weights
      [ -f "$out/results.json" ] && continue
      # Claim the evaluation by creating its directory; mkdir fails if another worker has it.
      # A claim without results after $stale seconds belongs to a worker that died: take it over.
      mkdir -p "$(dirname "$out")"
      if ! mkdir "$out" 2>/dev/null; then
        [ "$(age "$out")" -ge "$stale" ] || continue
        touch "$out"
      fi
      echo "== $name checkpoint $n ($weights)"
      if ! PICKUP_CHECKPOINT=$ckpt PICKUP_WEIGHTS=$weights PICKUP_OUT=$out \
        sim/pickup/scripts/eval.sh "$exp" "$task" "$episodes" --videos "${PICKUP_VIDEOS:-0}" 2>&1 | tail -3; then
        rm -rf "$out"  # give the claim back so the evaluation is tried again
        sleep 20
      fi
      evaluated=$((evaluated + 1))
      break 2  # start over from the newest checkpoint
    done
  done
  [ "$evaluated" -gt 0 ] && continue
  [ "${PICKUP_WATCH:-0}" = 1 ] || break
  if grep -qs "Model has seen the desired number of samples" "$log"; then
    # Training is over: one more pass picks up a final checkpoint that was too fresh to read.
    [ -n "${last_pass:-}" ] && break
    last_pass=1
    sleep 70
    continue
  fi
  sleep 30
done
