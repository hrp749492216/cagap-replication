# CAGAP local vLLM serving

`serve_vllm_model.sh` is the generic launcher used for the five local subjects. It sources the
operator's model-staging environment (`/etc/profile.d/cagap-model-staging.sh` or
`/workspace/cagap/cagap-model-staging.env`, which must define `CAGAP_GPU_VENV`), then runs
`vllm serve` with automatic tool choice enabled.

Required variables: `MODEL_REPO` (Hugging Face repo id or local path, pinned in
`config/model_freeze.json`), `SERVED_NAME` (must equal the model id the batch rows expect), and
`VLLM_API_KEY`. Optional: `PORT`, `HOST`, `TP`, `GPU_MEM`, `DTYPE`, `TOOL_PARSER` (default
`pythonic`; use the parser recorded for each subject in `config/roster.json`), `REASONING_PARSER`,
and `EXTRA_VLLM_ARGS` for any further serving flags the roster or runbook records for a subject.

Per-subject wrapper scripts that set these variables were generated on the GPU volume at run time
and are not versioned. The roster and model freeze record the served model names, revisions, and
precision roles they used. Smoke checks before dispatch: `scripts/cloud_vllm_smoke.py` and
`scripts/smoke_canary.py`.
