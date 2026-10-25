#!/bin/bash
# Serve GLM-5.1 (bf16) on an 8x B300 (262 GB/GPU = ~2096 GB) box as the cagap
# JUDGE (judge_a). The judge is a TEXT judge: requests are plain chat
# completions (messages + temperature + max_tokens + stream=false), NO tools,
# label parsed from text. So tool-call parsing is intentionally OFF (vLLM
# default) — we add ONLY the reasoning parser (keeps <think> out of content)
# plus the cagap OSF flag. THINKING IS DISABLED per request (the dispatcher
# sends chat_template_kwargs.enable_thinking=false). The glm45 reasoning parser
# stays ON as a seatbelt: verified (vLLM v0.20.2/main) that with thinking off it
# delegates to IdentityReasoningParser -> verdict in message.content,
# reasoning_content=null. Frozen 2026-05-29: canonical GLM-5.1 judge = thinking OFF.
#
# REQUIREMENTS on the B300 box:
#   - vLLM >= 0.20.2  (min 0.19.0)
#   - transformers >= 5.4.0  (arch glm_moe_dsa; older -> "model type not recognized")
#   - --trust-remote-code
# Per-request (set by the dispatcher, NOT here): temperature=0.0, max_tokens=4096,
#   stream=false. max_attempts=3 for invalid completions (roster).
#
# Required env: VLLM_API_KEY
set -euo pipefail
: "${VLLM_API_KEY:?export VLLM_API_KEY before launching}"

MODEL_DIR="${MODEL_DIR:-/opt/models/GLM-5.1}"
VLLM_BIN="${VLLM_BIN:-/opt/cagap-vllm/bin/vllm}"
TP="${TP:-8}"                              # 8x B300
GPU_MEM="${GPU_MEM:-0.90}"                 # 188.5GB weights/GPU -> ~47GB/GPU KV; 0.92-0.95 safe
MAX_LEN="${MAX_LEN:-32768}"               # judge-protocol parity (native ctx 202752)
SERVED_NAME="${SERVED_NAME:-glm-5.1}"
REASONING_PARSER="${REASONING_PARSER:-glm45}"   # GLM-5.1 reasoning parser (verify; see PITFALLS_GLM51.md)

[ -d "$MODEL_DIR" ] || { echo "FATAL: model dir $MODEL_DIR missing"; exit 1; }
[ -x "$VLLM_BIN" ]  || { echo "FATAL: vLLM not at $VLLM_BIN. Run ready_on_b300.sh first."; exit 1; }

VENV_DIR="$(cd "$(dirname "$VLLM_BIN")/.." && pwd)"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

# DeepGEMM JIT env — GLM-5.1's DSA (DeepSeek Sparse Attention) Indexer needs
# DeepGEMM-compiled CUDA kernels (RuntimeError otherwise at engine init). DeepGEMM
# JIT-compiles at serve time via nvcc AND symbol-dumps each cubin via cuobjdump.
# B300 is sm_103a; nvcc AND cuobjdump MUST be CUDA >=12.9 — 12.8 lacks compute_103a
# and its cuobjdump cannot parse Blackwell-Ultra cubins (-> kernel_runtime.hpp:45
# exit_code==0 assert). We pin 13.0 (matches /usr/local/cuda alternative). 0f5f266
# emits a sm_100f family cubin which runs fine on B300 (verified launch, sm=148).
# These MUST be set on every (re)launch incl. after a spot-eviction restart.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export DG_JIT_NVCC_COMPILER="${DG_JIT_NVCC_COMPILER:-$CUDA_HOME/bin/nvcc}"
export MAX_JOBS="${MAX_JOBS:-16}"

# --- eviction self-heal (narrow): cuobjdump + CUTLASS JIT headers live on the OS
# --- disk / editable-install dir and can vanish on a fresh spot instance; the data
# --- volume keeps /opt/DeepGEMM. Repair ONLY these — never vllm/torch/transformers.
# --- Written set -euo pipefail-safe: explicit guards, loud with the exact path.
if [ ! -x "$CUDA_HOME/bin/cuobjdump" ]; then
  echo "[preflight] cuobjdump missing at $CUDA_HOME/bin -> installing cuda-cuobjdump-13-0"
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-cuobjdump-13-0 || true
fi
if [ ! -x "$CUDA_HOME/bin/cuobjdump" ]; then
  echo "[preflight] FATAL: $CUDA_HOME/bin/cuobjdump still missing -> DeepGEMM JIT asserts at kernel_runtime.hpp:45 (exit_code==0). apt-get install cuda-cuobjdump-13-0." >&2
  exit 3
fi
# FlashInfer JIT-builds the sm_103a FMHA attention kernels and needs CUDA dev
# headers (cublasLt.h, nvrtc.h) that the nvcc-only toolkit install omits ->
# "Ninja build failed ... fatal error: cublasLt.h: No such file". Complete 13.0 dev.
if [ ! -f "$CUDA_HOME/include/cublasLt.h" ] || [ ! -f "$CUDA_HOME/include/nvrtc.h" ]; then
  echo "[preflight] CUDA 13.0 dev headers (cublasLt.h/nvrtc.h) missing -> installing dev packages"
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-nvrtc-dev-13-0 libcublas-dev-13-0 cuda-cudart-dev-13-0 || true
fi
if [ ! -f "$CUDA_HOME/include/cublasLt.h" ] || [ ! -f "$CUDA_HOME/include/nvrtc.h" ]; then
  echo "[preflight] FATAL: $CUDA_HOME/include/{cublasLt.h,nvrtc.h} still missing -> FlashInfer FMHA JIT fails. apt-get install cuda-nvrtc-dev-13-0 libcublas-dev-13-0." >&2
  exit 7
fi
DG_INC=/opt/DeepGEMM/deep_gemm/include
DG_SRC=/opt/DeepGEMM/third-party
if [ ! -f "$DG_INC/cutlass/arch/barrier.h" ]; then
  echo "[preflight] CUTLASS JIT headers absent in $DG_INC -> repairing from $DG_SRC"
  if [ ! -d "$DG_SRC/cutlass/include/cute" ]; then
    echo "[preflight] FATAL: repair source $DG_SRC/cutlass/include/cute absent -> reclone DeepGEMM --recursive." >&2
    exit 4
  fi
  cp -r "$DG_SRC/cutlass/include/cute"    "$DG_INC"/ || { echo "[preflight] FATAL: cute header copy failed" >&2; exit 5; }
  cp -r "$DG_SRC/cutlass/include/cutlass" "$DG_INC"/ || { echo "[preflight] FATAL: cutlass header copy failed" >&2; exit 5; }
  cp -r "$DG_SRC/fmt/include/fmt"         "$DG_INC"/ || { echo "[preflight] FATAL: fmt header copy failed" >&2; exit 5; }
fi
if [ ! -f "$DG_INC/cutlass/arch/barrier.h" ]; then
  echo "[preflight] FATAL: $DG_INC/cutlass/arch/barrier.h still missing after repair." >&2
  exit 6
fi
echo "[preflight] OK: cuobjdump present, CUTLASS JIT headers present ($CUDA_HOME)"
echo "Serving $MODEL_DIR as '$SERVED_NAME' | bf16 | TP=$TP | max_len=$MAX_LEN | reason=$REASONING_PARSER | TEXT-JUDGE (no tools)"

# FLAG NOTES (see PITFALLS_GLM51.md):
#  --reasoning-parser glm45          : seatbelt parser. With thinking OFF it delegates to
#                                      IdentityReasoningParser (verdict -> content, reasoning_content=null).
#                                      VERIFIED safe on vLLM v0.20.2/main. Do NOT substitute 'deepseek_r1'
#                                      (that one breaks with enable_thinking=false: vLLM #17609, content=null).
#  --chat-template-content-format string : keep string content path (judge sends a single string).
#  --no-exclude-tools-when-tool-choice-none : MANDATORY cagap OSF flag.
#  NO --tool-call-parser / NO --enable-auto-tool-choice : judge sends no tools (vLLM default = off).
#  NO MTP : speculative decode breaks </think> detection under structured output.
#  NO --quantization : bf16 auto-detect.
#
#  IF the OSF flag refuses to boot standalone (vLLM ties it to tool support),
#  uncomment the two lines below — glm47 is inert here, present only to satisfy startup:
#    --enable-auto-tool-choice \
#    --tool-call-parser glm47 \
exec "$VLLM_BIN" serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --dtype bfloat16 \
  --reasoning-parser "$REASONING_PARSER" \
  --chat-template-content-format string \
  --no-exclude-tools-when-tool-choice-none \
  --trust-remote-code
