#!/usr/bin/env bash
# Takeover demonstrations (DAgger-style): a trained policy drives the first K steps of every episode, then the
# scripted expert takes over from wherever the arm is and finishes the task. Only the expert's part is recorded,
# and only when it succeeds (collect.py --policy_ports). Starts one policy server per simulator worker, runs
# make_dataset.sh with them, and stops them again.
#   sim/pickup/scripts/make_takeover_dataset.sh <experiment_dir> <task> <episodes> <name> [extra collect.py args...]
# e.g. a dataset that holds an earlier one plus 2000 takeover demonstrations from layouts it does not use:
#   sim/pickup/scripts/make_takeover_dataset.sh experiments/pickup/<run> prune 2000 prune_2k_takeover2k \
#     --seed 500000 --takeover_steps 0 700 --include data/lerobot/prune_2k
# Environment variables (plus those of make_dataset.sh and _common.sh):
#   PICKUP_CHECKPOINT=<path to .pt>  serve this checkpoint instead of the latest one
#   PICKUP_WEIGHTS=raw|ema           trained or averaged weights (default: averaged when the run saved them)
#   PICKUP_SERVER_GPUS="2 3"         spread the policy servers over these GPUs (nvidia-smi indices); by default
#                                    they run where everything else does (PICKUP_GPU)
set -euo pipefail
exp=${1:?experiment dir}; task=${2:?task}; episodes=${3:?episodes}; name=${4:?dataset name}; shift 4
. "$(dirname "$0")/_common.sh"
workers=${PICKUP_WORKERS:-8}
read -r -a gpus <<< "${PICKUP_SERVER_GPUS:-}"
logs=data/lerobot/$name.servers
rm -rf "$logs"; mkdir -p "$logs"
# Distinct free ports, all held open at once while they are chosen.
read -r -a ports <<< "$("${sim_python[@]}" -c '
import socket, sys
held = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in held:
    s.bind(("127.0.0.1", 0))
print(*[s.getsockname()[1] for s in held])' "$workers")"
servers=()
trap 'kill "${servers[@]}" 2>/dev/null || true' EXIT
for i in "${!ports[@]}"; do
  if [ ${#gpus[@]} -gt 0 ]; then export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${gpus[$((i % ${#gpus[@]}))]}; fi
  "${foundry_python[@]}" -m sim.pickup.policy_server --checkpoint_dir "$exp" --port "${ports[$i]}" \
    ${PICKUP_CHECKPOINT:+--checkpoint "$PICKUP_CHECKPOINT"} ${PICKUP_WEIGHTS:+--weights "$PICKUP_WEIGHTS"} \
    </dev/null >"$logs/server_$i.log" 2>&1 &
  servers+=($!)
done
if [ -n "${PICKUP_GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$PICKUP_GPU; fi  # the simulator renders here
# Wait until every server has loaded its checkpoint; stop if one exits first.
for i in "${!servers[@]}"; do
  until grep -qs "listening on" "$logs/server_$i.log"; do
    if ! kill -0 "${servers[$i]}" 2>/dev/null; then
      echo "policy server $i exited before it was ready; last lines of $logs/server_$i.log:" >&2
      tail -n 5 "$logs/server_$i.log" >&2
      exit 1
    fi
    sleep 1
  done
done
echo "$workers policy servers ready: $(grep -h loaded "$logs/server_0.log")"
sim/pickup/scripts/make_dataset.sh "$task" "$episodes" "$name" --policy_ports "${ports[@]}" "$@"
