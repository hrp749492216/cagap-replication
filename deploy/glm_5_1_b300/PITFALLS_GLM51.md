# GLM-5.1 serving — pitfalls & locked config

Scope: **bf16 GLM-5.1 on 8×B300 (262 GB/GPU = ~2096 GB), vLLM, OpenAI-compatible,
LLM-as-judge (judge_a) producing forced text verdicts at temp=0.**
Synthesized from the official vLLM/SGLang recipes, zai-org GitHub, and 2026 vLLM
issues; verified against vLLM source (v0.20.2 / main).

## LOCKED CONFIG (frozen 2026-05-29 — prospective, before canonical dispatch)
```
thinking mode        : OFF  (per request: chat_template_kwargs.enable_thinking=false)
reasoning parser     : glm45  (ON at server — seatbelt, not "go faster")
tool-call parser     : OFF / omitted   (text judge sends no tools)
auto tool choice     : OFF / omitted
chat-template fmt    : string
OSF flag             : --no-exclude-tools-when-tool-choice-none  (inert for no-tools judge)
precision            : bf16     streaming : false     max_tokens : 4096     temperature : 0.0
canonical vote source: local/private GLM-5.1 vLLM route (B300)
```
Record: `outputs/main_run/20260525T164059Z/route_selection/glm_5_1_judge_route_selection.json`.
Request body: `glm51_judge_canonical_request.py` (single source of truth + smoke test).

### Why thinking OFF (judge lane)
Forced-verdict lane wants **stable, parseable, repeatable** verdicts, not deliberation.
Thinking-ON can eat the 4096 budget before the label lands (truncation → Unresolved),
adds variance, and adds channels (`reasoning_content`, separators) that can bleed into
the parsed field. <2 valid judge labels → row goes Unresolved, and GLM-5.1 is a default
judge for most non-Zai/non-Mistral subjects, so a noisy lane hurts the whole panel.
Chosen on parse-stability grounds, NOT on observed labels.

### Why glm45 stays ON with thinking OFF (verified, not assumed)
`glm45` → `DeepSeekV3ReasoningWithThinkingParser`, built **per-request**. With
`enable_thinking=false` it delegates to `IdentityReasoningParser` →
`extract_reasoning` returns `(None, full_text)` → **verdict in `message.content`,
`reasoning_content` null**. (vllm/reasoning/deepseek_v3_reasoning_parser.py +
identity_reasoning_parser.py, serving.py:1116-1126.) Do **not** substitute
`deepseek_r1` — it ignores `enable_thinking` and routes the answer into
`reasoning_content` with `content=null` (vLLM #17609). The pre-injected-`<think>`
trap (#33348) was fixed Jan 2026, long before our 0.20.2 floor.

## Per-route disable mechanism (the route is selected pre-dispatch)
| route | disable thinking via | status |
|---|---|---|
| local vLLM (canonical) | `chat_template_kwargs={"enable_thinking": false}` | verified |
| z.ai native (`api.z.ai/.../paas/v4`) | `{"thinking": {"type": "disabled"}}` | per Z.ai docs; confirm |
| ollama cloud (`glm-5.1:cloud`) | pending verification | pending |
`/no_think` does NOT work for GLM (confirmed by Z-AI). Don't prompt-hack; don't post-strip.

## Hard requirements on the B300 box (empirically verified 2026-05-30)
- vLLM: the cagap custom build **`0.19.1+cu130` already serves GLM-5.1** (its registry has
  `GlmMoeDsaForCausalLM`); stock vLLM needs ≥0.20.2. transformers 5.9.0 present (≥5.4.0). `--trust-remote-code`.
- **DeepGEMM is REQUIRED** (the real blocker, not the vLLM version). GLM-5.1's DSA
  (DeepSeek Sparse Attention) Indexer dies at engine init otherwise:
  `RuntimeError: Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.`
  — and this hits **bf16 too**, not just FP8. Fix (the PyPI sdist is an 18 KB stub MISSING
  the CUTLASS submodule → must source-install):
  ```
  git clone --recursive https://github.com/deepseek-ai/DeepGEMM /opt/DeepGEMM
  CUDA_HOME=/usr/local/cuda /opt/cagap-vllm/bin/pip install --no-build-isolation --no-deps -e /opt/DeepGEMM
  ```
  DeepGEMM **JIT-compiles kernels at serve time via nvcc** — needs `CUDA_HOME` / `DG_JIT_NVCC_COMPILER`
  / `CUDACXX` set (CUDA **12.8 verified working**, "DeepGEMM E8M0 enabled" on Blackwell; DeepGEMM
  docs say 12.9+ but 12.8 sufficed here). Editable install keeps CUTLASS headers
  (`CUTLASS_HOME=/opt/DeepGEMM/third-party/cutlass` as fallback). `serve_glm51_b300.sh` bakes all
  this in so a **spot-eviction restart** doesn't re-crash. Attention backend auto-selects
  `FLASHINFER_MLA_SPARSE`.
- **No MTP** for the judge (speculative decode breaks `</think>` detection under
  structured output, vLLM #34650; and MTP+tools needs vLLM main).
- No `--quantization` (bf16 auto-detect). B300 = sm_100 = supported tier; bf16 sidesteps
  FP8/Blackwell kernel quirks entirely.

## Fit (8×B300 @262)
bf16 1508 GB → 188.5 GB/GPU; ~73 GB/GPU free. MLA KV is tiny (2.74 GiB/seq @32k) →
KV is not the constraint. `--gpu-memory-utilization 0.90` (0.92–0.95 safe).

## Residual pitfalls to watch (reference)
- **Truncation**: thinking is off, but still monitor `finish_reason=length` + Unresolved;
  roster `max_attempts=3` retries invalid completions. 4096 cap is the truncation-risk lever.
- **Streaming tool-parse bugs** (#42400/#31443): N/A — judge is non-streaming + no tools.
- **Determinism**: temp=0 is not bit-deterministic on vLLM (batch variance). Enable Batch
  Invariance only if strict reproducibility is required (perf cost).
- **Smoke before canonical dispatch**: run `glm51_judge_canonical_request.py` →
  assert `content` non-empty, `reasoning_content` empty, `Label:` present.

## What to do next (ordered)
1. Finish bf16 download (in progress on this volume).
2. B300: vLLM ≥0.20.2 + transformers ≥5.4.0 in the serve venv.
3. `bash ready_on_b300.sh` → serves with the locked flags + starts carbon logger.
4. Smoke: `python3 glm51_judge_canonical_request.py --base-url http://127.0.0.1:8000`.
5. Dispatch the judge with the frozen body (`build_judge_body`), non-streaming, validate
   `Label:` client-side (existing parser).
6. After run: `carbon_report.py` for gCO2 (+ gCO2/row). Append runtime provenance to the
   route-selection record (instance_id, model_revision, finish reasons, retries, parsed-label status).

---

## B300 (sm_103a) bring-up — post-mortem & FAST PATH (2026-05-30, empirical)

**One-line root cause of the whole night:** B300 is `sm_103a` (compute cap 10.3).
**CUDA 12.8 AND 12.9 nvcc do not support `compute_103` — only CUDA 13.0 does.** Every
JIT-compiled kernel in the serve path (DeepGEMM DSA indexer, FlashInfer fmha) must build
for 103a, so the whole box must move to the CUDA 13.0 toolchain. We found this one failure
at a time over ~1h15m. The fast path below collapses it to ~15 min.

**FAST PATH:** run `provision_b300.sh` → `serve_glm51_b300.sh` → smoke curl. The script is
idempotent and bakes in every fix below. Manual equivalent:
```bash
# 1. CUDA 13.0 toolchain — nvcc + cuobjdump + ALL dev headers, one shot
apt-get install -y --no-install-recommends \
  cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 \
  cuda-cuobjdump-13-0 cuda-nvrtc-dev-13-0 libcublas-dev-13-0
export CUDA_HOME=/usr/local/cuda-13.0
# 2. DeepGEMM @ vLLM-pinned commit (NOT latest) + expose CUTLASS/CuTe/fmt headers
git clone --recursive https://github.com/deepseek-ai/DeepGEMM /opt/DeepGEMM
cd /opt/DeepGEMM && git checkout 0f5f2662027f0db05d4e3f6a94e56e2d8fc45c51 && git submodule update --init --recursive
CUDA_HOME=/usr/local/cuda-13.0 /opt/cagap-vllm/bin/pip install --no-build-isolation --no-deps -e .
cp -r third-party/cutlass/include/cutlass third-party/cutlass/include/cute third-party/fmt/include/fmt deep_gemm/include/
# 3. serve, then smoke (see provision_b300.sh tail for the exact curl)
```

**Issues hit, in order (symptom → root cause → fix → how to avoid):**

1. **`Sparse Attention Indexer CUDA op requires DeepGEMM to be installed`** (`sparse_attn_indexer.py:324`, every TP worker, GPUs at 0 MiB, generic "Engine core initialization failed").
   - Root: GLM-5.1 DSA (DeepSeek Sparse Attention) indexer needs DeepGEMM — **bf16 too**, not just FP8. PyPI sdist is an ~18 KB stub with **no CUTLASS submodule**.
   - Fix: source-install DeepGEMM (`git clone --recursive`, editable).
   - Avoid: install DeepGEMM in provisioning, before first serve.

2. **`context_lens.dim() == 2` assertion at CUDA-graph capture** (`attention.hpp:195`, kills all TP workers during warmup). ← the most expensive detour.
   - Root: we installed **latest DeepGEMM 2.5.0** (commit 714dd1a). vLLM 0.19's GLM call site passes a **1-D** context_lens; 2.5.0 mandates 2-D. API drift: `0f5f266` accepts 1-D, `891d57b+` requires 2-D.
   - Fix: pin/reinstall DeepGEMM at commit `0f5f266`.
   - Avoid: **never `pip install` latest DeepGEMM.** Pin the vLLM-expected commit from the start.

3. **`Unsupported gpu architecture 'compute_103a'`** (FlashInfer allreduce warning first, then DeepGEMM nvcc).
   - Root: B300 = sm_103a; nvcc 12.8 arch list stops at 100/101/120. **12.9 also lacks 103** (tried — dead end).
   - Fix: `apt install cuda-nvcc-13-0` (+ dev headers); `CUDA_HOME=/usr/local/cuda-13.0`.
   - Avoid: on B300, start on CUDA 13.0.

4. **DeepGEMM JIT `cutlass/arch/barrier.h: No such file`** (`compiler.hpp:220`, "compilation terminated").
   - Root: editable install exposes only `deep_gemm/include` to the JIT, not `third-party/cutlass/include`. DeepGEMM reads only `CUDA_HOME`/`CUDA_PATH`/`DG_USE_NVIDIA_TOOLS` — **no env hook for extra includes** (`CUTLASS_HOME` is ignored by its Python).
   - Fix: copy `third-party/{cutlass/include/cutlass, cutlass/include/cute, fmt/include/fmt}` into `deep_gemm/include/`. (Or a non-editable wheel build, which runs `prepare_includes()`.)
   - Avoid: copy headers immediately after editable install.

5. **DeepGEMM JIT `kernel_runtime.hpp:45 exit_code != 0`** (after "Loading CUBIN"; first misdiagnosed as an sm_100f-vs-sm_103a cubin mismatch — that was wrong).
   - Root: `kernel_runtime.hpp` builds the **cuobjdump** path from cuda_home (→ cuda-13.0) to enumerate symbols; **`cuda-cuobjdump-13-0` was not installed** (the nvcc-13 package doesn't pull it). Assert fired.
   - Fix: `apt install cuda-cuobjdump-13-0`; clear `/root/.deep_gemm/cache`. Then `JIT OK; sm=148`.
   - Avoid: install cuobjdump alongside nvcc.

6. **FlashInfer `fmha_gen` `RuntimeError: Ninja build failed` for `arch=compute_103a,sm_103a`** (`/root/.cache/flashinfer/0.6.6/103a/cached_ops/fmha_gen`, kills EngineCore). ← the final wall.
   - Root: FlashInfer 0.6.6 ships **no prebuilt sm_103a fmha cubin** → JIT-compiles via nvcc 13; that build needs CUDA-13 **dev headers** that weren't all present.
   - Fix: `apt install cuda-nvrtc-dev-13-0 libcublas-dev-13-0 cuda-cudart-dev-13-0`. fmha then compiles + caches under `.../103a`.
   - Avoid: install the full CUDA-13 dev header set up front (fast-path step 1).

**Benign — do NOT chase:**
- `FlashInfer allreduce fusion JIT failed ... compute_103a` *when execution continues past it* → vLLM auto-disables the fused allreduce and runs fine. Only the **fmha** ninja failure (#6) is fatal.
- `temp=0` is not bit-deterministic on vLLM (batch variance) — expected. Self-host buys weight/revision **provenance**, not bit-reproducibility.

**Spot-eviction recovery — CONFIRMED layout (2026-05-30): ONE OS volume `/dev/vda4` (1.8 TB) holds EVERYTHING** — `/`, CUDA 13, `/opt/DeepGEMM`, `/opt/cagap-vllm`, AND `/opt/models/GLM-5.1` (the **~1.5 TB weights**). **There is NO separate data volume.** Two cases:
- **Same-box reboot** (OS disk persists): everything survives, incl. warm JIT caches (`/root/.deep_gemm`, `/root/.cache/flashinfer/0.6.6/103a`, `/root/.cache/vllm/torch_compile_cache` ~2.2 GB). `serve_glm51_b300.sh`'s preflight self-heals any missing cuobjdump / CUDA-13 dev headers / CUTLASS copies inline — just relaunch.
- **Spot eviction / reimage** (OS disk WIPED): EVERYTHING gone, **including the 1.5 TB weights** — recovery is NOT 12 min. Order: (1) recycled IP → `ssh-keygen -R <ip>` (see `verda_recycled_ip_known_hosts`); (2) **re-download weights ~1.5 TB** (hours — see `README_SERVE_GLM51.md` / `/opt/hf-venv`); (3) `bash provision_b300.sh` (~12 min); (4) `export VLLM_API_KEY=...; bash serve_glm51_b300.sh` (re-JITs, ~3–4 min); (5) smoke → dispatch.

**The only real eviction protection is the OUTPUT mirror** — run `sync_glm_votes_from_b300.sh` locally BEFORE dispatch so completed votes are saved continuously. For the grading run, **prefer on-demand over spot** (or snapshot the volume if Verda allows): a mid-run eviction costs a 1.5 TB re-download, not a quick reprovision. CAVEAT: `serve_glm51_b300.sh`'s preflight comment claims "the data volume keeps /opt/DeepGEMM" — **inaccurate on this single-volume box**; on eviction /opt/DeepGEMM is wiped too (its `exit 4` reclone path would fire — but weights are the real loss).

`provision_b300.sh` (full fresh-box rebuild: CUDA 13 + DeepGEMM clone + header copy + verify) vs `serve_glm51_b300.sh` preflight (narrow inline self-heal, assumes /opt/DeepGEMM present) are complementary — the preflight covers reboot; provision covers a wiped box (minus weights).

**Confirmed-good versions** (8× B300 SXM6 AC, 275 GiB/GPU): driver `580.126.09` · vLLM `0.19.1+cu130` · torch `2.10.0+cu130` · transformers `5.9.0` · flashinfer `0.6.6` · DeepGEMM `0f5f266` (editable, headers copied) · nvcc `13.0.88`.

> Clean step-by-step usage (provision → serve → smoke) lives in **`HOWTO_serve_GLM51_B300.md`**. This file is the *why / debug* reference.
