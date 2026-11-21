#!/usr/bin/env bash
# The training recipe of tutorial/training-pipeline.md in one command, for a new task: collect demonstrations, draw
# them again in other looks (paired re-rendering), preprocess into shards, train with every checkpoint evaluated as it
# appears, run the report card on the final averaged weights, and clean up. Every stage is an existing script; this one
# only chains them and passes the names along.
#   sim/pickup/scripts/pipeline.sh <task> <episodes> <name> [extra collect.py args...]
# e.g. the robust pruning policy of the lab notebook (step 8, frozen DINOv2 on paired demonstrations):
#   PICKUP_GPU=2 PICKUP_EVAL_GPU=3 PICKUP_EVAL_ARGS=--leaf_fix PICKUP_WORKERS=32 \
#     sim/pickup/scripts/pipeline.sh prune 2000 prune_2k_rs_tail10_leaffix --random_start --tail_steps 10 --leaf_fix
#
# Names. The collection is data/lerobot/<name>; its re-rendered copies data/lerobot/<name>_paired; the shards
# data/preprocessed/<data>_1s (the preprocessing config's suffix), <data> being <name>_paired (or <name> with
# PICKUP_COPIES=0); the run experiments/pickup/<config>_<name>_<samples in M>m[_seed<seed>]; results under
# results/pickup/<run>/, the report card in <task>_scenario_<situation>/ckpt_<final>_ema/.
# Tested: the chain with PICKUP_DRY_RUN=1, and the card's bookkeeping and the cleanup on a run made of empty files. The
# stages themselves are the scripts every run of the lab notebook went through.
#
# Environment variables (plus those of _common.sh, train.sh, eval.sh and eval_checkpoints.sh):
#   PICKUP_STAGES       the stages to run, in order (default "collect rerender preprocess train card cleanup"); a stage
#                       whose output exists is skipped, so the same command picks up where it stopped
#   PICKUP_WORKERS      simulator and preprocessing processes (default 8). rerender.py re-runs the collection with the
#                       same number of workers it was collected with, so keep it the same for collect and rerender
#   PICKUP_COPIES       re-rendered copies per demonstration (default 2; the source is kept too, so a third of the
#                       episodes are in the plain look); 0 trains on the plain demonstrations alone
#   PICKUP_LOOKS        rerender.py --looks (default: its own, the training halves of lighting, camera, bench, distractors)
#   PICKUP_PREPROCESS   preprocessing config (default preprocess_1s: the frame 1 s back, for the *_frames_1s configs)
#   PICKUP_CONFIG       policy config (default dp_dinov2_lang_frames_1s)
#   PICKUP_SAMPLES      training samples (default 20000000)
#   PICKUP_SEED         --hparams.seed and --data.seed (default: the config's, 42)
#   PICKUP_EVAL_GPU     GPU for the checkpoint evaluator and half of the report card (default: PICKUP_GPU)
#   PICKUP_EVAL_EPISODES  episodes per checkpoint evaluation (default 100)
#   PICKUP_EVAL_ARGS    eval.py arguments for every evaluation, checkpoints and card (e.g. --leaf_fix)
#   PICKUP_CARD         the report card's situations (default: plain and the 11 of the lab notebook's step 8)
#   PICKUP_CARD_EPISODES  episodes per situation (default 100)
#   PICKUP_CARD_JOBS    situations evaluated at a time (default 6)
#   PICKUP_SCRATCH      a directory on a scratch disk: the collections and the preprocessing are written there and
#                       linked into data/ (default: none, everything under data/)
#   PICKUP_KEEP_SOURCE  1 (default) keeps the source collection at cleanup (it is what a later re-render starts from);
#                       0 deletes it too. The re-rendered collection and the preprocessing intermediates always go
#   PICKUP_DRY_RUN=1    print the commands instead of running them
set -euo pipefail
task=${1:?task}; episodes=${2:?episodes}; name=${3:?dataset name}; shift 3
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../.."
. "$here/_common.sh"

stages=${PICKUP_STAGES:-collect rerender preprocess train card cleanup}
workers=${PICKUP_WORKERS:-8}
copies=${PICKUP_COPIES:-2}
pre=${PICKUP_PREPROCESS:-preprocess_1s}
cfg=${PICKUP_CONFIG:-dp_dinov2_lang_frames_1s}
samples=${PICKUP_SAMPLES:-20000000}
seed=${PICKUP_SEED:-}
eval_gpu=${PICKUP_EVAL_GPU:-${PICKUP_GPU:-}}
read -r -a eval_args <<< "${PICKUP_EVAL_ARGS:-}"
card=${PICKUP_CARD:-plain lighting camera bench lighting/test camera/test bench/test distractors distractors/test wind bump sticky}
scratch=${PICKUP_SCRATCH:-}
dry=${PICKUP_DRY_RUN:-0}

data=$name; [ "$copies" -gt 0 ] && data=${name}_paired
shards=$data${pre#preprocess}  # preprocess_1s: <data>_1s; preprocess: <data>
run=${cfg}_${name}_$((samples / 1000000))m${seed:+_seed$seed}
exp=experiments/pickup/$run
say() { echo "[pipeline $(date -u +%H:%M:%S)] $*"; }
do_() { if [ "$dry" = 1 ]; then echo "+ $*"; else "$@"; fi; }
has() { [[ " $stages " == *" $1 "* ]]; }
collection_dir() { if [ -n "$scratch" ]; then echo "$scratch/data/lerobot/$1"; else echo "data/lerobot/$1"; fi; }
link_in() {  # link a collection or shard directory on the scratch disk into data/
  [ -n "$scratch" ] || return 0
  do_ mkdir -p "$(dirname "$2")"
  do_ ln -sfn "$1" "$2"
}
say "task $task, $episodes episodes as $name; shards $shards; run $run; stages: $stages"

if has collect; then
  if [ -f "$(collection_dir "$name")/meta/info.json" ]; then say "collect: $name exists, skipped"
  else
    say "collect: $episodes $task demonstrations with $workers workers"
    do_ mkdir -p "$(dirname "$(collection_dir "$name")")"
    do_ "${sim_python[@]}" -m sim.pickup.collect --task "$task" --episodes "$episodes" --out "$(collection_dir "$name")" \
      --workers "$workers" "$@"
    link_in "$(collection_dir "$name")" "data/lerobot/$name"
  fi
fi

if has rerender && [ "$copies" -gt 0 ]; then
  if [ -f "$(collection_dir "$data")/meta/info.json" ]; then say "rerender: $data exists, skipped"
  else
    say "rerender: $copies copies of each demonstration in other looks, the source kept"
    read -r -a looks <<< "${PICKUP_LOOKS:-}"
    do_ "${sim_python[@]}" -m sim.pickup.rerender --src "$(collection_dir "$name")" --out "$(collection_dir "$data")" \
      --copies "$copies" --include_source --collect_workers "$workers" --workers "$workers" \
      ${looks[@]+--looks "${looks[@]}"}
    link_in "$(collection_dir "$data")" "data/lerobot/$data"
  fi
fi

if has preprocess; then
  if [ -f "data/preprocessed/$shards/shards/stats.json" ]; then say "preprocess: $shards exists, skipped"
  else
    say "preprocess: $data into data/preprocessed/$shards ($pre)"
    out_root=$root; [ -n "$scratch" ] && out_root=$scratch
    do_ env PICKUP_ROOT="$out_root" PICKUP_WORKERS="$workers" "$here/preprocess.sh" "$data" "$shards" "$pre"
    # the intermediates: the frames and episode files are not read by training, only the shards
    do_ rm -rf "$out_root/data/preprocessed/$shards/frames" "$out_root/data/preprocessed/$shards/episodes"
    link_in "$scratch/data/preprocessed/$shards" "data/preprocessed/$shards"
  fi
fi

if has train; then
  if grep -qs "Model has seen the desired number of samples" "experiments/pickup/$run.train.log"; then
    say "train: $run finished already, skipped"
  else
    say "train: $cfg on $shards, $samples samples; checkpoints evaluated on GPU ${eval_gpu:-default} as they appear"
    seed_args=(); [ -n "$seed" ] && seed_args=(--hparams.seed "$seed" --data.seed "$seed")
    # the evaluator watches the run until it ends (eval_checkpoints.sh), on its own GPU
    evlog=experiments/pickup/$run.eval.log
    if [ "$dry" = 1 ]; then evlog=/dev/stdout; else mkdir -p experiments/pickup; fi
    do_ env ${eval_gpu:+PICKUP_GPU=$eval_gpu} PICKUP_WATCH=1 PICKUP_EVAL_JOBS="${PICKUP_EVAL_JOBS:-4}" \
      PICKUP_EVAL_ARGS="${eval_args[*]-}" "$here/eval_checkpoints.sh" "$exp" "$task" "${PICKUP_EVAL_EPISODES:-100}" \
      > "$evlog" 2>&1 &
    evaluator=$!
    do_ env PICKUP_NUM_CHECKPOINTS="${PICKUP_NUM_CHECKPOINTS:-10}" PICKUP_ABOUT="${PICKUP_ABOUT:-pipeline.sh: $cfg on $shards}" \
      "$here/train_resilient.sh" "$cfg" "$shards" "$run" "$samples" ${seed_args[@]+"${seed_args[@]}"}
    say "train: done; waiting for the checkpoint evaluator"
    wait "$evaluator" || true
  fi
fi

final() {  # the number of the run's final averaged weights
  { ls "$exp"/checkpoints/ema_*.pt 2>/dev/null || true; } | sed 's/.*ema_\([0-9]*\)\.pt$/\1/' | sort -n | tail -1
}

if has card; then
  n=$(final)
  if [ -z "$n" ] && [ "$dry" = 1 ]; then n=N; fi
  [ -n "$n" ] || { echo "card: $exp has no averaged weights" >&2; exit 1; }
  say "card: checkpoint $n, averaged weights, ${PICKUP_CARD_EPISODES:-100} episodes per situation"
  i=0; pids=(); cardlog=/dev/null; [ "$dry" = 1 ] && cardlog=/dev/stdout
  for s in $card; do
    out=results/pickup/$run/${task}_scenario_${s//[:,@\/]/_}/ckpt_${n}_ema
    [ -f "$out/results.json" ] && continue
    scen=(--scenario "$s"); [ "$s" = plain ] && scen=()
    gpu=${PICKUP_GPU:-}; [ $((i % 2)) = 1 ] && gpu=$eval_gpu
    do_ env ${gpu:+PICKUP_GPU=$gpu} PICKUP_CHECKPOINT="$exp/checkpoints/checkpoint_$n.pt" PICKUP_WEIGHTS=ema \
      PICKUP_OUT="$out" "$here/eval.sh" "$exp" "$task" "${PICKUP_CARD_EPISODES:-100}" \
      ${scen[@]+"${scen[@]}"} ${eval_args[@]+"${eval_args[@]}"} > "$cardlog" &
    pids+=($!); i=$((i + 1))
    if [ ${#pids[@]} -ge "${PICKUP_CARD_JOBS:-6}" ]; then wait "${pids[@]}"; pids=(); fi
  done
  if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}"; fi
  say "card: results in results/pickup/$run/${task}_scenario_*/ckpt_${n}_ema"
  for s in $card; do
    f=results/pickup/$run/${task}_scenario_${s//[:,@\/]/_}/ckpt_${n}_ema/results.json
    if [ -f "$f" ]; then echo "  $s: $(python3 -c 'import json, sys; r = json.load(open(sys.argv[1])); print(r["successes"], "of", r["episodes"])' "$f" 2>/dev/null)"; fi
  done
fi

if has cleanup; then
  n=$(final)
  if [ -n "$n" ]; then
    say "cleanup: $run keeps checkpoint_$n.pt and ema_$n.pt; the other checkpoints and every optimiser state go"
    for f in "$exp"/checkpoints/checkpoint_*.pt "$exp"/checkpoints/ema_*.pt; do
      [ -e "$f" ] || continue
      k=$(basename "$f" .pt); k=${k##*_}
      [ "$k" != "$n" ] && do_ rm -f "$f"
    done
    do_ rm -f "$exp"/checkpoints/optimizer_*.pt
    do_ rm -rf "$exp"/checkpoints/incomplete
  else
    say "cleanup: $run has no averaged weights; its checkpoints are left alone"
  fi
  if [ "$copies" -gt 0 ]; then
    say "cleanup: the re-rendered collection $data (made again from the source by rerender.py if needed)"
    do_ rm -rf "$(collection_dir "$data")" "data/lerobot/$data"
  fi
  if [ "${PICKUP_KEEP_SOURCE:-1}" = 0 ]; then
    say "cleanup: the source collection $name"
    do_ rm -rf "$(collection_dir "$name")" "data/lerobot/$name"
  fi
fi
say "done: run $run, shards data/preprocessed/$shards"
