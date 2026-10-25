# How to serve GLM-5.1 (bf16) cleanly on 8× B300

Stand up GLM-5.1 as an OpenAI-compatible text judge (forced verdict, **thinking OFF**) on an
8× B300 (sm_103a) box via vLLM. Three steps: **provision → serve → smoke**.

> For *why* a step exists or what each error means, see `PITFALLS_GLM51.md`. This file is just
> the clean path.

---

## 0. Box prerequisites
- 8× B300 (sm_103a / compute cap 10.3), driver ≥ 580, fabricmanager up → `nvidia-smi` shows 8 GPUs.
- `/opt/cagap-vllm` — vLLM `0.19.1+cu130` venv (torch 2.10+cu130, transformers ≥ 5.4, flashinfer 0.6.6).
- `/opt/models/GLM-5.1` — bf16 weights (282 safetensors).
- NVIDIA CUDA apt repo configured (`cuda-ubuntu2404-x86_64`).
- Recycled spot IP? `ssh-keygen -R <ip>` before the first SSH.

## 1. Provision — once per fresh box (~12 min)
```bash
bash provision_b300.sh
```
Installs the **CUDA 13.0** toolchain (B300 = sm_103a; CUDA 12.8/12.9 nvcc cannot target
`compute_103`) + `cuobjdump-13-0` + CUDA-13 dev headers, and source-installs **DeepGEMM pinned at
`0f5f266`** with CUTLASS/CuTe/fmt headers exposed to its JIT. Idempotent — safe to re-run.

## 2. Serve
```bash
export VLLM_API_KEY=<bearer-token>
bash serve_glm51_b300.sh
```
First boot JIT-compiles the DSA + FMHA kernels for sm_103a and captures CUDA graphs
(`init engine ... ~156s`). Ready when the log prints **`Application startup complete.`**
The script self-heals missing cuobjdump / dev headers / CUTLASS copies on launch and bakes in the
locked flags:

| setting | value |
|---|---|
| tensor-parallel-size | 8 |
| dtype | bfloat16 |
| max-model-len | 32768 |
| gpu-memory-utilization | 0.90 |
| reasoning-parser | `glm45` (→ identity parser when thinking off → verdict in `message.content`) |
| tool-call parser | none (text judge sends no tools) |
| mandatory flag | `--no-exclude-tools-when-tool-choice-none` |
| MTP / quantization | off (bf16 auto-detect) |

## 3. Smoke — gate before any dispatch
```bash
VK=$(cat /opt/serve/vllm_api_key)
curl -fsS -H "Authorization: Bearer $VK" -H 'Content-Type: application/json' \
  http://127.0.0.1:8000/v1/chat/completions -d '{
    "model":"glm-5.1",
    "messages":[{"role":"user","content":"Respond exactly: Label: PASS"}],
    "temperature":0.0,"top_p":1.0,"max_tokens":4096,"stream":false,
    "chat_template_kwargs":{"enable_thinking":false}}'
```
**PASS** = `message.content` non-empty and contains `Label:`, `reasoning_content` is `null`,
`finish_reason` is `"stop"`. (Null `reasoning_content` proves thinking-off is working.)

## Per-request judge body
`temperature 0.0` · `top_p 1.0` · `max_tokens 4096` · `stream false` ·
`chat_template_kwargs.enable_thinking=false`. Parse the label from `message.content`.
Single source of truth + smoke: `glm51_judge_canonical_request.py`.

---

## If it won't boot — 5 most likely causes (one-liners; full detail in `PITFALLS_GLM51.md`)
| symptom | fix |
|---|---|
| `Sparse Attention Indexer ... requires DeepGEMM` | DeepGEMM not installed (DSA needs it, bf16 too) → run `provision_b300.sh`. |
| `context_lens.dim() == 2` crash at graph capture | wrong DeepGEMM — must be commit `0f5f266`, **not** latest. |
| `Unsupported gpu architecture 'compute_103a'` | using CUDA 12.x nvcc → B300 needs **CUDA 13.0**. |
| `kernel_runtime.hpp:45 exit_code != 0` | `cuda-cuobjdump-13-0` missing. |
| FlashInfer `fmha_gen` `Ninja build failed` (103a) | CUDA-13 dev headers missing → `cuda-nvrtc-dev-13-0 libcublas-dev-13-0 cuda-cudart-dev-13-0`. |

## Spot-eviction note
This box is a **single OS volume** — eviction wipes everything, including the ~1.5 TB weights.
- **Same-box reboot:** just `bash serve_glm51_b300.sh` (preflight self-heals; JIT caches warm).
- **Eviction:** re-download weights → `provision_b300.sh` → serve.
- Mirror outputs continuously (`sync_glm_votes_from_b300.sh`) so completed work survives an eviction.
