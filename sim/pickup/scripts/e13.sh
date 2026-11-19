#!/usr/bin/env bash
# Architecture v3, E13 (tutorial/architecture-v3.md, step 4): v3's slow level on the simulator's scene with the task's
# own program, over the fast level as it is, on each situation of rungs 1 to 4. Every run goes through eval_v2.sh with
# PICKUP_SLOW_LEVEL=v3, one after the other, each episode starting clear of the plant (--clear_start).
#   sim/pickup/scripts/e13.sh <fast run seed 43> <fast run seed 44> [validation|evaluation|all]
# For each fast run and phase:
#   prune                          the simulator's scene, the whole state, and the scene with the cut gate on
#   order (top_down, bottom_up, pair: the episodes given to each mode)   the same three
#   nothing_to_do                  the scene and the whole state
#   ties, sick, buds               the scene, the whole state, and the scene with the look left undrawn (+undrawn)
# The phases: validation (the seed-43 fast level on validation layouts, seeds from 900,000), then evaluation (both fast
# levels on evaluation layouts, seeds from 1,000,000), and there once more reason_mix with v2's slow level from the new
# starts, so that v2's numbers can be set beside the new ones. A run whose results.json exists is skipped, so the script
# can be started again where it stopped.
# Environment variables:
#   PICKUP_E13_OUT=<dir>          where the runs go (default results/pickup/e13): <dir>/<phase>/<fast run>/<run>
#   PICKUP_E13_VALIDATION=<n>     episodes per run on validation layouts (default 100)
#   PICKUP_E13_EVALUATION=<n>     episodes per run on evaluation layouts (default 200; order: n/2 per mode)
#   PICKUP_E13_ONLY=<regex>       only the runs whose name matches (e.g. 'order|prune')
#   PICKUP_E13_ARGS="<args>"      more eval.py arguments for every v3 run. E13 ran with the defaults (a missing fact held
#                                 for, --inspect hold) and the straight line home (--home_move line); after E13 the runs
#                                 take "--inspect turn" (tutorial/architecture-v3.md, "After E13")
#   PICKUP_WEIGHTS                the fast level's weights (default ema: the final averaged weights)
#   PICKUP_EYES=<eyes run>        also the cells on the eyes' scene (step 6 and E17: <cell>_eyes, every situation and
#                                 reason_mix with v2's slow level), the eyes served by eval_v2.sh (PICKUP_EYES_CHECKPOINT
#                                 and PICKUP_EYES_WEIGHTS as there); PICKUP_E13_ONLY='_eyes$' runs those alone
#   PICKUP_E13_PROGRAM=writer     the program the language model writes (step 5, E17: PICKUP_WRITER and PICKUP_WRITER_ARGS
#                                 as for eval_v2.sh) in place of the task's own, for every v3 run; default gold
#   PICKUP_E13_HELDOUT=1          also the cells of order, ties, sick and buds in held-out sentence frames
#                                 (<cell>_heldout_<scene>: scenarios order_heldout, ties_heldout, ...), for a written
#                                 program; with the task's own they are the training frames' cells
#   PICKUP_GPU and the rest: as for eval_v2.sh and _common.sh
set -euo pipefail
fast43=${1:?fast run, seed 43}; fast44=${2:?fast run, seed 44}; phase=${3:-all}
case $phase in validation|evaluation|all) ;; *) echo "the phase is validation, evaluation or all" >&2; exit 1 ;; esac
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../.."  # the repository's root, as eval_v2.sh runs from it: the results' paths are relative to it
root=${PICKUP_E13_OUT:-results/pickup/e13}
n_val=${PICKUP_E13_VALIDATION:-100}; n_eval=${PICKUP_E13_EVALUATION:-200}
only=${PICKUP_E13_ONLY:-.}
export PICKUP_WEIGHTS=${PICKUP_WEIGHTS:-ema}

# run <phase> <fast run> <name> <level> <scene> <task> <episodes> <first seed> [eval.py args...]
run() {
  local where=$1 fast=$2 name=$3 level=$4 scene=$5 task=$6 episodes=$7 first=$8
  shift 8
  [[ $name =~ $only ]] || return 0
  local out
  out="$root/$where/$(basename "$fast")/$name"
  if [ -f "$out/results.json" ]; then echo "[e13] $out: done already"; return 0; fi
  echo "[e13] $where $(basename "$fast") $name: $episodes episodes from seed $first"
  local more=()
  if [ "$level" = v3 ] && [ -n "${PICKUP_E13_ARGS:-}" ]; then read -r -a more <<< "$PICKUP_E13_ARGS"; fi
  local program=gold
  if [ "$level" = v3 ]; then program=${PICKUP_E13_PROGRAM:-gold}; fi
  PICKUP_SLOW_LEVEL=$level PICKUP_SCENE=$scene PICKUP_PROGRAM=$program PICKUP_OUT=$out \
    "$here/eval_v2.sh" "$fast" "$task" "$episodes" --seed_start "$first" --clear_start \
    ${more[@]+"${more[@]}"} "$@"
}

# situations <phase> <fast run> <episodes> <first seed>
situations() {
  local where=$1 fast=$2 n=$3 first=$4 mode situation
  run "$where" "$fast" prune_truth v3 truth prune "$n" "$first"
  run "$where" "$fast" prune_full v3 full prune "$n" "$first"
  run "$where" "$fast" prune_truth_gate v3 truth prune "$n" "$first" --gate
  for mode in top_down bottom_up pair; do  # the same number of episodes for each way of giving an order
    run "$where" "$fast" "order_${mode}_truth" v3 truth prune "$((n / 2))" "$first" --scenario "order:$mode"
    run "$where" "$fast" "order_${mode}_full" v3 full prune "$((n / 2))" "$first" --scenario "order:$mode"
    run "$where" "$fast" "order_${mode}_truth_gate" v3 truth prune "$((n / 2))" "$first" --scenario "order:$mode" --gate
  done
  run "$where" "$fast" nothing_to_do_truth v3 truth prune "$n" "$first" --scenario nothing_to_do
  run "$where" "$fast" nothing_to_do_full v3 full prune "$n" "$first" --scenario nothing_to_do
  for situation in ties sick buds; do
    run "$where" "$fast" "${situation}_truth" v3 truth prune "$n" "$first" --scenario "$situation"
    run "$where" "$fast" "${situation}_full" v3 full prune "$n" "$first" --scenario "$situation"
    run "$where" "$fast" "${situation}_truth_undrawn" v3 truth prune "$n" "$first" --scenario "$situation+undrawn"
  done
  if [ -n "${PICKUP_EYES:-}" ]; then  # the same cells on the eyes' scene
    run "$where" "$fast" prune_eyes v3 eyes prune "$n" "$first"
    for mode in top_down bottom_up pair; do
      run "$where" "$fast" "order_${mode}_eyes" v3 eyes prune "$((n / 2))" "$first" --scenario "order:$mode"
    done
    run "$where" "$fast" nothing_to_do_eyes v3 eyes prune "$n" "$first" --scenario nothing_to_do
    for situation in ties sick buds; do
      run "$where" "$fast" "${situation}_eyes" v3 eyes prune "$n" "$first" --scenario "$situation"
    done
  fi
  if [ -n "${PICKUP_E13_HELDOUT:-}" ]; then  # held-out sentence frames, on each scene asked for
    local scene
    for scene in truth ${PICKUP_EYES:+eyes}; do
      for mode in top_down bottom_up pair; do
        run "$where" "$fast" "order_${mode}_heldout_$scene" v3 "$scene" prune "$((n / 2))" "$first" \
          --scenario "order_heldout:$mode"
      done
      for situation in ties sick buds; do
        run "$where" "$fast" "${situation}_heldout_$scene" v3 "$scene" prune "$n" "$first" --scenario "${situation}_heldout"
      done
    done
  fi
}

if [ "$phase" != evaluation ]; then
  situations validation "$fast43" "$n_val" 900000
fi
if [ "$phase" != validation ]; then
  for fast in "$fast43" "$fast44"; do
    situations evaluation "$fast" "$n_eval" 1000000
  done
  run evaluation "$fast43" reason_mix_v2_truth v2 truth reason_mix "$n_eval" 1000000
  if [ -n "${PICKUP_EYES:-}" ]; then
    run evaluation "$fast43" reason_mix_v2_eyes v2 eyes reason_mix "$n_eval" 1000000
  fi
fi
echo "[e13] all runs of $phase are in $root"
