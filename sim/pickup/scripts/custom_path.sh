#!/usr/bin/env bash
# The from-scratch path, one stage per command: a language model, then a vision-language model built
# on it, then an action policy built on that.
#
#   custom_path.sh data      web text for stage 1 and captioned sim renders for stage 2
#   custom_path.sh llm       stage 1: a 91M-parameter language model on the web text
#   custom_path.sh vlm       stage 2: that language model plus a ViT, on the captioned renders
#   custom_path.sh check     held-out captioning check of the stage-2 model
#   custom_path.sh vla <dataset> <run_name> <samples> [pretrained|scratch] [config]
#                            stage 3: an action policy on the stage-2 model ("pretrained"), or on the
#                            same architecture with random weights ("scratch", the control that shows
#                            what stages 1 and 2 bought). Default config: vla_custom_tokens.
#
# Budgets: PICKUP_LLM_SAMPLES (default 500000), PICKUP_VLM_SAMPLES (default 400000),
# PICKUP_CAPTION_SCENES (default 20000 scenes, two images each).
# Environment and GPU selection: see _common.sh.
set -euo pipefail
stage=${1:?stage: data | llm | vlm | check | vla}; shift
. "$(dirname "$0")/_common.sh"
workers=${PICKUP_WORKERS:-8}

latest() { ls "experiments/pickup/$1"/checkpoints/checkpoint_*.pt 2>/dev/null | sort -V | tail -1; }

train() {  # train <config> <run name> <samples> [extra main.py args...]
  local cfg=$1 name=$2 samples=$3; shift 3
  mkdir -p experiments/pickup
  "${foundry_python[@]}" vla_foundry/main.py --config_path "sim/pickup/configs/$cfg.yaml" --name "$name" \
    --total_train_samples "$samples" --num_checkpoints "${PICKUP_NUM_CHECKPOINTS:-4}" "$@" </dev/null 2>&1 |
    tr '\r' '\n' | grep --line-buffered -vE "step/s|^$" > "experiments/pickup/$name.train.log"
}

case $stage in
  data)
    "${foundry_python[@]}" -m sim.pickup.text_data --out data/custom/dclm
    [ -f data/custom/sim_captions/manifest.jsonl ] || "${sim_python[@]}" -m sim.pickup.caption_data \
      --out data/custom/sim_captions --scenes "${PICKUP_CAPTION_SCENES:-20000}" --workers "$workers"
    [ -f data/custom/sim_captions_heldout/manifest.jsonl ] || "${sim_python[@]}" -m sim.pickup.caption_data \
      --out data/custom/sim_captions_heldout --scenes 1000 --seed0 3000000 --workers "$workers"
    ;;
  llm)
    train custom_llm custom_llm "${PICKUP_LLM_SAMPLES:-500000}" "$@"
    ;;
  vlm)
    llm=$(latest custom_llm)
    [ -n "$llm" ] || { echo "No language-model checkpoint. Run the llm stage first." >&2; exit 1; }
    train custom_vlm custom_vlm "${PICKUP_VLM_SAMPLES:-400000}" --model.transformer.resume_from_checkpoint "$llm" "$@"
    ;;
  check)
    "${foundry_python[@]}" -m sim.pickup.vlm_check --exp experiments/pickup/custom_vlm \
      --shards data/custom/sim_captions_heldout "$@"
    ;;
  vla)
    dataset=${1:?dataset}; name=${2:?run name}; samples=${3:?samples}; init=${4:-pretrained}; cfg=${5:-vla_custom_tokens}
    shift $(( $# < 5 ? $# : 5 ))
    [ -f experiments/pickup/custom_vlm/config_model.yaml ] ||
      { echo "No experiments/pickup/custom_vlm. Run the vlm stage first (the control needs its architecture too)." >&2; exit 1; }
    if [ "$init" = pretrained ]; then
      backbone=(--model.vision_language_backbone.resume_from_checkpoint "$(latest custom_vlm)")
    elif [ "$init" = scratch ]; then
      backbone=(--model.vision_language_backbone.vlm_experiment_dir experiments/pickup/custom_vlm)
    else
      echo "init must be 'pretrained' or 'scratch', got '$init'" >&2; exit 1
    fi
    exec sim/pickup/scripts/train_resilient.sh "$cfg" "$dataset" "$name" "$samples" "${backbone[@]}" "$@"
    ;;
  *)
    echo "unknown stage: $stage (expected data | llm | vlm | check | vla)" >&2; exit 1
    ;;
esac
