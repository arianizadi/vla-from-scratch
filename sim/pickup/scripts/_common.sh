# Shared setup for the pickup-sim scripts. Sourced, not executed.
#
# By default everything runs in the repository's uv environment:
#     uv sync --group preprocessing --group sim
#
# Overrides (environment variables):
#   PICKUP_FOUNDRY_PYTHON  interpreter for training, preprocessing and the policy server
#                          (default: .venv/bin/python). May be a command with arguments, for
#                          example a container launcher: PICKUP_FOUNDRY_PYTHON="my-launcher python".
#   PICKUP_SIM_PYTHON      interpreter for the simulator side (default: same as above).
#   PICKUP_ROOT            repository path as PICKUP_FOUNDRY_PYTHON sees it (default: the repository
#                          root; set it when that interpreter runs in a container).
#   PICKUP_GPU             GPU index, as nvidia-smi numbers them, for both compute and rendering.
#                          Use it on multi-GPU machines to run several jobs side by side, or to keep
#                          a job off GPU 0. It sets CUDA_VISIBLE_DEVICES (with PCI bus ordering, so
#                          the index means what nvidia-smi shows). MuJoCo 4 renders on the first
#                          CUDA-visible device, so rendering follows. The raw EGL device order can
#                          differ from nvidia-smi's, so do not set MUJOCO_EGL_DEVICE_ID by hand
#                          unless you have checked which GPU it maps to.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

if [ -n "${PICKUP_FOUNDRY_PYTHON:-}" ]; then
  read -r -a foundry_python <<< "$PICKUP_FOUNDRY_PYTHON"
elif [ -x .venv/bin/python ]; then
  foundry_python=("$PWD/.venv/bin/python")
else
  echo "No .venv found. Run: uv sync --group preprocessing --group sim   (or set PICKUP_FOUNDRY_PYTHON)" >&2
  exit 1
fi
if [ -n "${PICKUP_SIM_PYTHON:-}" ]; then
  read -r -a sim_python <<< "$PICKUP_SIM_PYTHON"
else
  sim_python=("${foundry_python[@]}")
fi
root=${PICKUP_ROOT:-$PWD}

export MUJOCO_GL=${MUJOCO_GL:-egl}
if [ -n "${PICKUP_GPU:-}" ]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=$PICKUP_GPU
fi
