#!/usr/bin/env bash
set -Eeuo pipefail
if [[ -f /etc/profile.d/cagap-model-staging.sh ]]; then
  source /etc/profile.d/cagap-model-staging.sh
elif [[ -f /workspace/cagap/cagap-model-staging.env ]]; then
  source /workspace/cagap/cagap-model-staging.env
else
  echo "CAGAP model staging environment not found" >&2
  exit 2
fi

: "${MODEL_REPO:?MODEL_REPO must be set to an HF repo id or local model path}"
: "${SERVED_NAME:?SERVED_NAME must match the model id expected by the batch rows}"
: "${VLLM_API_KEY:?VLLM_API_KEY must be set in env or an env file before launch}"

source "$CAGAP_GPU_VENV/bin/activate"

export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
cuda_runtime_dir="$(find "$CAGAP_GPU_VENV/lib" -type d -path '*/site-packages/nvidia/cuda_runtime/lib' 2>/dev/null | head -n 1)"
if [[ -n "$cuda_runtime_dir" ]]; then
  export LD_LIBRARY_PATH="$cuda_runtime_dir:${LD_LIBRARY_PATH:-}"
fi

port="${PORT:-8000}"
host="${HOST:-0.0.0.0}"
tp="${TP:-1}"
gpu_mem="${GPU_MEM:-0.92}"
dtype="${DTYPE:-auto}"
tool_parser="${TOOL_PARSER:-pythonic}"

args=(
  serve "$MODEL_REPO"
  --host "$host"
  --port "$port"
  --served-model-name "$SERVED_NAME"
  --tensor-parallel-size "$tp"
  --gpu-memory-utilization "$gpu_mem"
  --dtype "$dtype"
  --enable-auto-tool-choice
  --tool-call-parser "$tool_parser"
)

if [[ -n "${REASONING_PARSER:-}" ]]; then
  args+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($EXTRA_VLLM_ARGS)
  args+=("${extra_args[@]}")
fi

exec "$CAGAP_GPU_VENV/bin/vllm" "${args[@]}"
