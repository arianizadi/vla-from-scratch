#!/usr/bin/env bash
# Evaluate every checkpoint of a run closed-loop, so that success can be plotted against training.
#   sim/pickup/scripts/eval_checkpoints.sh <experiment_dir> <task> [episodes]
# Results go to results/pickup/<run>/<task>/ckpt_<N>_<raw|ema>/results.json. Runs trained with
# --ema.enabled True get both their trained (raw) and averaged (ema) weights evaluated. Checkpoints
# that already have a results.json are skipped, so the script can be re-run at any time.
#
# Environment variables (plus those of eval.sh and _common.sh):
#   PICKUP_WATCH=1    keep polling for new checkpoints until the training log reports the end of
#                     the run. Start it next to train_resilient.sh to evaluate while training.
#   PICKUP_VIDEOS=<n> rollout videos to save per checkpoint (default 0)
#   PICKUP_PORT       give each concurrent evaluator its own port
set -uo pipefail
exp=${1:?experiment dir}; task=${2:?task}; episodes=${3:-100}
cd "$(dirname "$0")/../../.."
name=$(basename "$exp")
log=experiments/pickup/$name.train.log
while true; do
  evaluated=0
  for ckpt in $(ls "$exp"/checkpoints/checkpoint_*.pt 2>/dev/null | sort -V); do
    n=$(basename "$ckpt" .pt); n=${n#checkpoint_}
    for weights in raw ema; do
      file=$ckpt
      [ "$weights" = ema ] && file=$exp/checkpoints/ema_$n.pt
      [ -f "$file" ] || continue
      # Skip files that may still be being written.
      [ $(( $(date +%s) - $(stat -c %Y "$file") )) -ge 60 ] || continue
      out=results/pickup/$name/$task/ckpt_${n}_$weights
      [ -f "$out/results.json" ] && continue
      echo "== $name checkpoint $n ($weights)"
      PICKUP_CHECKPOINT=$ckpt PICKUP_WEIGHTS=$weights PICKUP_OUT=$out \
        sim/pickup/scripts/eval.sh "$exp" "$task" "$episodes" --videos "${PICKUP_VIDEOS:-0}" 2>&1 | tail -2
      evaluated=$((evaluated + 1))
    done
  done
  [ "${PICKUP_WATCH:-0}" = 1 ] || break
  if [ "$evaluated" -eq 0 ] && grep -qs "Model has seen the desired number of samples" "$log"; then
    # Training is over: one more pass picks up a final checkpoint that was too fresh to read.
    sleep 70
    [ -n "${last_pass:-}" ] && break
    last_pass=1
    continue
  fi
  sleep 60
done
