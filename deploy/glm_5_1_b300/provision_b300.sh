#!/bin/bash
# provision_b300.sh — idempotent bring-up of the GLM-5.1 (bf16) judge stack on a
# fresh 8x B300 (sm_103a) box. Collapses the ~1h15m of trial-and-error from the
# 2026-05-30 session into one run (~15 min, mostly DeepGEMM clone + first-boot JIT).
#
# WHY this exists: every fix below was a hand-mutation on a SPOT, OS-only volume.
# A spot eviction or reimage wipes /opt and /usr/local — without this script you
# redo the entire night. Run this the moment a fresh B300 attaches, THEN serve.
#
# Assumes (Verda cagap image already provides these):
#   - NVIDIA driver 580 + fabricmanager up; nvidia-smi works; 8x B300 visible
#   - /opt/cagap-vllm  vLLM 0.19.1+cu130 venv (torch 2.10+cu130, transformers 5.9, flashinfer 0.6.6)
#   - /opt/models/GLM-5.1  bf16 weights (282 safetensors)
#   - NVIDIA CUDA apt repo configured (cuda-ubuntu2404-x86_64.list)
#
# Root cause of ALL the JIT pain: B300 is sm_103a (compute cap 10.3). CUDA 12.8 AND
# 12.9 nvcc do NOT support compute_103 — only CUDA 13.0 does. Every JIT-compiled
# kernel in the stack (DeepGEMM DSA indexer, FlashInfer fmha) must build for 103a,
# so the whole box must be on the CUDA 13.0 toolchain. See PITFALLS_GLM51.md.
set -euo pipefail

VENV="${VENV:-/opt/cagap-vllm}"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
MODEL_DIR="${MODEL_DIR:-/opt/models/GLM-5.1}"
CUDA13="${CUDA13:-/usr/local/cuda-13.0}"
DEEPGEMM_DIR="${DEEPGEMM_DIR:-/opt/DeepGEMM}"
# vLLM 0.19.x's GLM-5.1 DSA call site passes a 1-D context_lens. Newer DeepGEMM
# (>=2.5.0 / commit 891d57b+) mandates 2-D and crashes at CUDA-graph capture with
# `context_lens.dim() == 2`. PIN to the vLLM-expected commit. DO NOT pip-install latest.
DEEPGEMM_REF="${DEEPGEMM_REF:-0f5f2662027f0db05d4e3f6a94e56e2d8fc45c51}"

log(){ printf '\n\033[1;36m[provision]\033[0m %s\n' "$*"; }
die(){ printf '\n\033[1;31m[provision FATAL]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "0. sanity"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing — driver not up"
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
[ "$CC" = "10.3" ] || log "WARNING: compute_cap=$CC (expected 10.3/sm_103a). This script is B300-specific; CUDA-13 may be unnecessary on older arch."
[ -x "$PIP" ] || die "vLLM venv not at $VENV"
[ -d "$MODEL_DIR" ] || die "weights missing at $MODEL_DIR"

# ---------------------------------------------------------------------------
log "1. CUDA 13.0 toolchain (sm_103a) — nvcc + cuobjdump + ALL dev headers, one shot"
# The minimal known-good set (de-duplicated from the session's trial-and-error):
#   cuda-nvcc-13-0        : the only nvcc that lists compute_103          (fixes 'Unsupported gpu architecture compute_103a')
#   cuda-cuobjdump-13-0   : DeepGEMM kernel_runtime.hpp resolves symbols via cuobjdump (fixes kernel_runtime.hpp:45 exit_code!=0)
#   cuda-nvrtc-dev-13-0 } : FlashInfer fmha_gen JIT needs these CUDA-13 dev headers
#   libcublas-dev-13-0  } : (fixes FlashInfer 'Ninja build failed' for arch 103a)
#   cuda-cudart-dev-13-0 / cuda-cccl-13-0 : runtime + CCCL headers for both JITs
export DEBIAN_FRONTEND=noninteractive
apt-get install -y --no-install-recommends \
  cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 \
  cuda-cuobjdump-13-0 cuda-nvrtc-dev-13-0 libcublas-dev-13-0
[ -x "$CUDA13/bin/nvcc" ]      || die "nvcc 13 not at $CUDA13/bin/nvcc"
[ -x "$CUDA13/bin/cuobjdump" ] || die "cuobjdump 13 missing (DeepGEMM will fail at kernel_runtime.hpp:45)"
"$CUDA13/bin/nvcc" --list-gpu-arch | grep -q compute_103 || die "nvcc 13 does not list compute_103 — wrong toolkit"
export CUDA_HOME="$CUDA13"
export PATH="$CUDA13/bin:$PATH"

log "1b. trivial sm_103a build sanity (proves the toolchain before DeepGEMM)"
TMPCU="$(mktemp --suffix=.cu)"; printf '__global__ void k(){}\nint main(){return 0;}\n' > "$TMPCU"
"$CUDA13/bin/nvcc" -gencode=arch=compute_103a,code=sm_103a -o "${TMPCU%.cu}" "$TMPCU" \
  && log "   sm_103a build OK" || die "trivial sm_103a build FAILED — toolchain broken"
rm -f "$TMPCU" "${TMPCU%.cu}"

# ---------------------------------------------------------------------------
log "2. DeepGEMM @ pinned commit $DEEPGEMM_REF (REQUIRED — GLM-5.1 DSA indexer; bf16 too)"
# The PyPI sdist is an ~18KB stub MISSING the CUTLASS submodule -> must source-install.
if [ ! -d "$DEEPGEMM_DIR/.git" ]; then
  git clone --recursive https://github.com/deepseek-ai/DeepGEMM "$DEEPGEMM_DIR"
fi
git -C "$DEEPGEMM_DIR" fetch --all --tags --quiet || true
git -C "$DEEPGEMM_DIR" checkout --quiet "$DEEPGEMM_REF"
git -C "$DEEPGEMM_DIR" submodule update --init --recursive
CUDA_HOME="$CUDA13" "$PIP" install --no-build-isolation --no-deps -e "$DEEPGEMM_DIR"

log "2b. expose CUTLASS/CuTe/fmt headers to the JIT (editable install does NOT)"
# DeepGEMM's JIT only searches deep_gemm/include and has NO env hook for extra
# include paths. Replicate what a full wheel's prepare_includes() would do.
INC="$DEEPGEMM_DIR/deep_gemm/include"
for d in cutlass/include/cutlass cutlass/include/cute fmt/include/fmt; do
  base="$(basename "$d")"
  if [ ! -e "$INC/$base" ]; then
    cp -r "$DEEPGEMM_DIR/third-party/$d" "$INC/" \
      && log "   copied $base -> deep_gemm/include/" \
      || die "missing $DEEPGEMM_DIR/third-party/$d (submodules not checked out for $DEEPGEMM_REF)"
  fi
done

log "2c. clear any stale DeepGEMM JIT cache (forces a clean recompile against this toolchain)"
rm -rf /root/.deep_gemm/cache /root/.deep_gemm/tmp 2>/dev/null || true

# ---------------------------------------------------------------------------
log "3. verify DeepGEMM is importable + vLLM detects it"
CUDA_HOME="$CUDA13" "$PY" - <<'PYEOF' || die "DeepGEMM verification failed"
import deep_gemm
from vllm.utils.deep_gemm import has_deep_gemm
assert has_deep_gemm(), "vllm.utils.deep_gemm.has_deep_gemm() == False"
print("   deep_gemm import OK; vLLM has_deep_gemm() == True")
PYEOF

# ---------------------------------------------------------------------------
cat <<EOF

\033[1;32m[provision DONE]\033[0m Stack ready. CUDA_HOME=$CUDA13

Next:
  export VLLM_API_KEY=...                 # bearer token for clients
  bash $(dirname "$0")/serve_glm51_b300.sh
  # First boot JIT-compiles DeepGEMM DSA + FlashInfer fmha for sm_103a and
  # captures CUDA graphs (~3-4 min incl. 'init engine ... took ~156s'). Caches
  # warm afterward (/root/.cache/flashinfer/0.6.6/103a, torch_compile_cache).

Then SMOKE before any dispatch (thinking-off must yield content + null reasoning):
  VK=\$(cat /opt/serve/vllm_api_key)
  curl -fsS -H "Authorization: Bearer \$VK" -H 'Content-Type: application/json' \\
    http://127.0.0.1:8000/v1/chat/completions \\
    -d '{"model":"glm-5.1","messages":[{"role":"user","content":"Respond exactly: Label: PASS"}],"temperature":0.0,"top_p":1.0,"max_tokens":4096,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}'
  # expect: choices[0].message.content="Label: PASS", reasoning_content=null, finish_reason="stop"
EOF
