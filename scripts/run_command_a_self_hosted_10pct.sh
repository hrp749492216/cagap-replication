#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -x /opt/cagap-vllm/bin/python ]; then
  DEFAULT_PYTHON="/opt/cagap-vllm/bin/python"
else
  DEFAULT_PYTHON="python3"
fi

ROOT="${ROOT:-$REPO_ROOT/outputs/main_run/20260525T164059Z/judge_batches}"
ENDPOINT="${CAGAP_VLLM_ENDPOINT:-http://127.0.0.1:8000}"
SERVED_MODEL="${CAGAP_COMMAND_A_SERVED_MODEL:-command-a-03-2025}"
WORKERS="${WORKERS:-128}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.10}"
SAMPLE_SEED="${SAMPLE_SEED:-command-a-self-hosted-v1}"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
SCRIPT="${SCRIPT:-$REPO_ROOT/scripts/dispatch_command_a_self_hosted.py}"
SUMMARY_OUT="${SUMMARY_OUT:-$REPO_ROOT/outputs/main_run/20260525T164059Z/command_a_self_hosted_10pct_summary.json}"
ROUTE_SELECTION_RECORD="${ROUTE_SELECTION_RECORD:-$REPO_ROOT/outputs/main_run/20260525T164059Z/route_selection/command_a_self_hosted_route_selection.json}"
REQUIRE_ROUTE_SELECTION="${REQUIRE_ROUTE_SELECTION:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MODEL_MATCH="${SKIP_MODEL_MATCH:-0}"

EXTRA_ARGS=()
if [ -f "$ROUTE_SELECTION_RECORD" ]; then
  EXTRA_ARGS+=(--route-selection-record "$ROUTE_SELECTION_RECORD")
elif [ "$REQUIRE_ROUTE_SELECTION" = "1" ] && [ "$DRY_RUN" != "1" ]; then
  echo "missing route selection record: $ROUTE_SELECTION_RECORD" >&2
  echo "set ROUTE_SELECTION_RECORD or run scripts/write_command_a_route_selection_record.py before canonical dispatch" >&2
  exit 2
fi
if [ "$DRY_RUN" = "1" ]; then
  EXTRA_ARGS+=(--dry-run --skip-model-match)
elif [ "$SKIP_MODEL_MATCH" = "1" ]; then
  EXTRA_ARGS+=(--skip-model-match)
fi

exec "$PYTHON_BIN" "$SCRIPT" \
  --judge-batches-root "$ROOT" \
  --endpoint "$ENDPOINT" \
  --served-model "$SERVED_MODEL" \
  --model-label command-a-03-2025 \
  --judge command-a-03-2025 \
  --workers "$WORKERS" \
  --max-tokens "$MAX_TOKENS" \
  --sample-fraction "$SAMPLE_FRACTION" \
  --sample-seed "$SAMPLE_SEED" \
  --summary-out "$SUMMARY_OUT" \
  "${EXTRA_ARGS[@]}"
