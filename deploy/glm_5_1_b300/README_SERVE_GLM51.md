# GLM-5.1 serve + carbon — volume handoff

This volume is staged to serve **GLM-5.1 (bf16)** on an **8× B300** box and to
**measure the actual carbon cost** of the run. Everything lives under `/opt/`
so it travels with the volume.

## Layout
```
/opt/models/GLM-5.1/            bf16 weights (282 safetensors, ~1.51 TB)
/opt/hf-venv/                   huggingface_hub + hf_xet (re-download/verify)
/opt/hf_home/                   HF cache + hf_token (chmod 600)
/opt/serve/serve_glm51_b300.sh  vLLM serve (TP=8, OSF flags)
/opt/serve/ready_on_b300.sh     one-shot bring-up after attach
/opt/carbon/gpu_power_logger.sh dependency-free per-GPU energy sampler
/opt/carbon/carbon_report.py    energy CSV -> kWh + gCO2 (FIN grid default)
/opt/carbon/run_with_codecarbon.py  CodeCarbon cross-check wrapper
/opt/carbon-venv/               codecarbon (installed on this volume)
```

## After detaching this volume and attaching to the 8× B300

1. Mount it (if secondary disk) and confirm `config.json` is reachable.
2. Ensure vLLM exists on the B300 (the cagap Verda startup script makes
   `/opt/cagap-vllm`). If not, see step 2 inside `ready_on_b300.sh`.
3. Bring everything up:
   ```bash
   export VLLM_API_KEY=...            # bearer token for clients
   bash /opt/serve/ready_on_b300.sh   # finds weights, starts logger, serves
   ```
4. Wait for readiness, then **smoke-verify the tool parser** (GLM-5.1 may need
   `glm45` instead of `pythonic`):
   ```bash
   curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id'   # expect glm-5.1
   ```
   If `tool_choice=auto` returns 400, relaunch with `TOOL_PARSER=glm45`.

## Carbon accounting

Two independent methods (use the GPU logger as primary, CodeCarbon as cross-check):

- **Primary (GPU-only, exact):** `gpu_power_logger.sh` records each GPU's
  cumulative energy counter; `carbon_report.py` differences start/end →
  `gpu_energy_kwh × PUE × grid_gCO2/kWh = gCO2`. Pass `--rows N` for gCO2/row.
- **Cross-check (whole-machine):** wrap the dispatch with
  `run_with_codecarbon.py` (adds CPU+RAM, country=FIN).

**Grid intensity:** default `79 gCO2/kWh` is the Finland annual average
(low-carbon: nuclear/hydro/wind). For a publishable figure, set
`--grid-gco2-kwh` to the **observed FI zone intensity for the run window**
(e.g. Electricity Maps zone `FI`) and record the source + window. **PUE**
defaults to 1.2 — replace with Verda FIN-03's actual PUE if published.

Example end-to-end:
```bash
# logger started by ready_on_b300.sh -> /opt/carbon/power_log_*.csv
# ... run the GLM-5.1 judge dispatch ...
kill $(cat /opt/carbon/logger.pid)
python3 /opt/carbon/carbon_report.py /opt/carbon/power_log_*.csv \
    --grid-gco2-kwh 79 --pue 1.2 --rows 22851 --json /opt/carbon/report.json
```
