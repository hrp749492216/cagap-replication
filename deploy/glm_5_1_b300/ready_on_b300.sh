#!/bin/bash
# One-shot bring-up on the 8x B300 box AFTER this volume is attached.
# Works whether the volume is the root disk or mounted as a secondary disk:
# it auto-locates the GLM-5.1 weights, locates/creates the vLLM venv, starts
# the carbon logger, then launches the serve script.
#
# Required env: VLLM_API_KEY
# Optional:     MODEL_DIR (skip autodetect), VLLM_BIN, GRID_GCO2 (default 79)
set -euo pipefail
: "${VLLM_API_KEY:?export VLLM_API_KEY}"

echo "== [1/5] locate GLM-5.1 weights =="
if [ -z "${MODEL_DIR:-}" ]; then
  for c in /opt/models/GLM-5.1 /mnt/*/opt/models/GLM-5.1 /mnt/*/GLM-5.1 /data/*/GLM-5.1; do
    [ -f "$c/config.json" ] && { MODEL_DIR="$c"; break; }
  done
fi
[ -n "${MODEL_DIR:-}" ] && [ -f "$MODEL_DIR/config.json" ] \
  || { echo "FATAL: could not find GLM-5.1 weights (set MODEL_DIR=...)"; exit 1; }
export MODEL_DIR
echo "   weights: $MODEL_DIR"

echo "== [2/5] locate / create vLLM venv =="
VLLM_BIN="${VLLM_BIN:-/opt/cagap-vllm/bin/vllm}"
if [ ! -x "$VLLM_BIN" ]; then
  echo "   vLLM not found at $VLLM_BIN."
  echo "   Either run the cagap Verda startup script (creates /opt/cagap-vllm),"
  echo "   or install matching the B300 driver, e.g.:"
  echo "     python3 -m venv /opt/cagap-vllm && /opt/cagap-vllm/bin/pip install -U pip"
  echo "     /opt/cagap-vllm/bin/pip install vllm   # pick the build matching CUDA on this host"
  echo "   Then re-run with VLLM_BIN=/opt/cagap-vllm/bin/vllm"
  exit 2
fi
export VLLM_BIN
echo "   vLLM: $($VLLM_BIN --version 2>/dev/null | head -1)"

echo "== [3/5] free GPUs =="
for p in $(lsof /dev/nvidia* 2>/dev/null | awk 'NR>1{print $2}' | sort -u); do kill -9 "$p" 2>/dev/null || true; done
sleep 4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "== [4/5] start carbon logger =="
CARBON_CSV="/opt/carbon/power_log_$(date -u +%Y%m%dT%H%M%SZ).csv"
nohup bash /opt/carbon/gpu_power_logger.sh "$CARBON_CSV" 2 >/opt/carbon/logger.out 2>&1 &
echo "   logger PID $! -> $CARBON_CSV"
echo "$!" > /opt/carbon/logger.pid

echo "== [5/5] launch vLLM serve =="
nohup bash /opt/serve/serve_glm51_b300.sh >/opt/serve/serve_glm51.log 2>&1 &
echo "   serve PID $! -> /opt/serve/serve_glm51.log"
echo
echo "Poll readiness:  curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id'"
echo "After workload drains:"
echo "  kill \$(cat /opt/carbon/logger.pid)"
echo "  python3 /opt/carbon/carbon_report.py $CARBON_CSV --grid-gco2-kwh \${GRID_GCO2:-79} --rows <N> --json /opt/carbon/report.json"
