# CAGAP replication package

Minimum code and locked materials needed to recreate the preregistered study
**"Tool Callability With a Schema in the Input: An LLM Safety Study With
Preregistration"** (the Conversation–Action Gap, CAGAP; earlier preprint title "When a
Visible Tool Becomes Callable") end to end: locked prompts and
tool schemas, route preflight, subject dispatch, extraction, deterministic
grading, judge ensemble, provenance resolution, registered analyses, the
post-registration DISABLED+FORMAT control, and the post hoc audits reported in
the manuscript.

- OSF preregistration: <https://osf.io/yz8hn/> (project node <https://osf.io/p6bv5/>)
- Study design and statistical analysis plan: [`paper/Study_Design.md`](paper/Study_Design.md)
- OSF registration text: [`paper/OSF-CAGAP-paste.txt`](paper/OSF-CAGAP-paste.txt)
- Post-registration control protocols: [`experiments/disabled_format_control/`](experiments/disabled_format_control/)

This repository contains **no API keys, no raw model responses, and no graded
labels**. Raw responses, judge votes, and derived label files are retained in the
authors' private audit bundle with SHA-256 linkage; the registered materials are
deposited with the OSF registration. Re-running the pipeline against the
providers listed in `config/roster.json` spends money and creates external jobs.

## Study at a glance

| Item | Value |
|---|---|
| Subjects | 10 models: 5 hosted APIs, 5 local/self-hosted (`config/roster.json`) |
| Prompts | 192 locked action-variant prompts, 6 synthetic domains (`experiments/aligned/prompts.jsonl`) |
| Arms | `TOOL`, `DISABLED`, `NOTOOLS`, `TOOL_IRRELEVANT`; 2 personas; 3 trials |
| Primary contrast | `TOOL` minus audit-eligible `DISABLED` on E1 (non-refusal) and E2 (syntactic action compliance) |
| Grading | Deterministic predicates on native calls and extracted actions; family-excluded 3-judge ordinal-median ensemble for unresolved text rows |
| Analyses | Conditional logistic regression on matched units, prompt-cluster bootstrap RDs, Holm family of four tests, registered sensitivities |

The synthetic domains and identifiers are protocol-shaped analogs. The schemas do
not operate real infrastructure.

## Setup

Python 3.11 with the hash-pinned lock:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements.lock
```

Operators serving local subjects with MLX on Apple silicon install the superset
lock instead: `python -m pip install --require-hashes -r requirements-local.lock`.
On macOS 26 or newer the pinned `scipy` wheel fails to load (`dlopen ... __thread_bss`);
after the locked install run `python -m pip install -U scipy` as a documented deviation.
GPU serving uses vLLM; see [`serve/`](serve/) and [`deploy/glm_5_1_b300/`](deploy/glm_5_1_b300/).

Offline verification (no network, no credentials):

```bash
python -m pytest -q
ruff check .
```

The test suite covers contracts, grading rules, judge aggregation, runners with
mocked transports, the control protocol, and the statistical pipeline on
synthetic fixtures. Tests that read the authors' private run directories are
deselected in `pyproject.toml`, or not included where the test module itself reads
them at import time; one isolation test is deselected because the generator pinned by
the control protocols reads `.env` at import.

## Credentials

Runners read credentials from the environment, or from a local `.env` file that
is ignored by Git. Copy [`env.example`](env.example) and fill in only the keys
for the routes you run:

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Claude Opus 4.7 subject and substitute judge |
| `OPENAI_API_KEY` | GPT-5.5 subject (native batch) and substitute judge |
| `GEMINI_API_KEY` | Gemini 3.5 Flash direct-API control runs |
| `X_API_KEY` (or `XAI_API_KEY`) | Grok 4.3 subject (native batch) |
| `MOONSHOT_API_KEY` | Kimi K2.6 subject (native batch) |
| `OPENROUTER_API_KEY` | OpenRouter routes (pinned providers) |
| `ZAI_API_KEY` | GLM 5.1 judge (online) |
| `COHERE_API_KEY` | Command A judge (native route; self-hosted route needs no key) |
| `MISTRAL_API_KEY` | Mistral Large 2512 judge (native batch) |
| `MINIMAX_API_KEY` | MiniMax M2.7 extractor (native async batch) |
| `OLLAMA_API_KEY` | Ollama Cloud fallback routes, only where the roster documents them |
| `HUGGINGFACE_HUB_TOKEN` | Downloading gated local checkpoints |
| `VLLM_API_KEY` | Bearer token for your own vLLM servers |

Never commit `.env`. The test configuration strips `*_API_KEY` variables before
collection.

## Verify the locked materials

`materials_lock.json` records SHA-256 digests of the locked corpus, schemas, case
cards, calibration subset, predicate grader, roster, and the statistical analysis plan.
Check them before any run:

```bash
python - <<'EOF'
import hashlib, json
from pathlib import Path
lock = json.load(open("materials_lock.json"))
# The main-study grader pinned here is kept byte-identical under grader/registered/.
remap = {"grader/semantic_predicates.py": "grader/registered/semantic_predicates.py"}
checks = {"paper/Study_Design.md": lock["sap_checksum"]}
def walk(o):
    if isinstance(o, dict):
        if "path" in o and "sha256" in o:
            checks.setdefault(o["path"], o["sha256"])
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
walk(lock["artifacts"])
walk(lock["selector"])
bad = 0
for path, expected in checks.items():
    p = Path(remap.get(path, path))
    if p.is_dir():
        print("DIR ", path, "(aggregate digest; member files are checked individually)")
        continue
    if not p.is_file():
        print("SKIP", path, "(not part of this package)")
        continue
    ok = hashlib.sha256(p.read_bytes()).hexdigest() == expected
    bad += not ok
    print("OK  " if ok else "BAD ", path, "" if str(p) == path else f"(checked {p})")
raise SystemExit(bad)
EOF
```

Two pinned paths are reported as skipped: the corpus selector script and the corpus gate
report belong to the authors' corpus-construction tooling, which this package omits.

**Predicate grader versions.** Two versions of the deterministic grader ship, and each is
pinned by the experiment that used it. `materials_lock.json` pins the main-study grader at
digest `2da31253687c7ab1…`; it is kept byte-identical at
`grader/registered/semantic_predicates.py` and produced the main-study labels. Every
DISABLED+FORMAT control protocol in `experiments/disabled_format_control/` pins the revision at
`grader/semantic_predicates.py` (digest `d9959f043157d96e…`), which is the file the pipeline
imports. The revision adds the schema tool's risk class to the predicate trace and labels a
call to a non-target high-impact schema tool Partial instead of Refuse; nothing else differs
(`diff` the two files). To grade with the main-study version, copy
`grader/registered/semantic_predicates.py` over `grader/semantic_predicates.py`; the three
tests in `tests/test_semantic_predicates.py` and `tests/test_grade_subject_responses.py` that
encode the revised rule then fail by design.

## Pipeline

Run everything from the repository root. Every script prints its options with `--help`;
placeholders in angle brackets are operator choices. Stages write under `outputs/`, which is
ignored by Git. Retry policy and halt rules are those of the statistical analysis plan,
[`paper/Study_Design.md`](paper/Study_Design.md).

### A. Route preflight (no confirmatory rows)

```bash
python scripts/schema_visibility_preflight.py --route-type native --output-dir outputs/pilot/schema_visibility
python scripts/tool_choice_semantics_canary.py --live --out-dir outputs/pilot/tool_choice_canary
python scripts/cloud_vllm_smoke.py --subject <local_subject> --served-model <served_name> \
    --precision-value <precision> --precision-role primary --substrate-tag <tag> --endpoint <vllm_url>
python scripts/smoke_canary.py --live
```

Only routes classified `schema_visible` or `server_forced` enter the H1 contrast.

### B. Subject dispatch

```bash
python scripts/run_synthetic_prompts.py --batch --validate                       # build per-subject batch inputs
python scripts/run_synthetic_prompts.py --submit-batches --subject <hosted_subject> --batches-dir <batches_dir>
python scripts/fetch_native_batch_results.py --batches-dir <batches_dir> --raw-dir <raw_dir> --subject <hosted_subject>
python scripts/run_local_subjects.py --batches-dir <batches_dir> --results-dir <results_dir> \
    --subjects <local_subject> --serving-precision-role primary --vllm-endpoint <vllm_url>
python scripts/verify_raw_completeness.py --batches-dir <batches_dir> --raw-dir <raw_dir> --report <report.json>
```

Local subjects are served with vLLM (`serve/serve_vllm_model.sh`). Every local checkpoint is
pinned to a Hugging Face repository and resolved commit in `config/model_freeze.json`
(`hf_repo`, `hf_commit_sha_resolved`), and the served precision variants and runtimes are
recorded in `config/roster.json`. The quantization audit repeats the local dispatch with
`--serving-precision-role quantization_audit`.

### C. Extraction and judge votes

```bash
python scripts/run_minimax_batch_extraction.py --input <responses.jsonl> \
    --out outputs/extractor/<subject>__minimax_extractions.jsonl --cache-dir outputs/cache/extractor
python scripts/dispatch_judge_batches.py --input <responses.jsonl> --prompts experiments/aligned/prompts.jsonl \
    --out-dir outputs/judge_batches/<subject> --subject-id <subject> \
    --extractions outputs/extractor/<subject>__minimax_extractions.jsonl
python scripts/fetch_judge_batches.py --manifest outputs/judge_batches/<subject>/manifest.json \
    --out outputs/judge_votes/<subject>__judge_votes.jsonl
```

Judge routes used in the study: GLM 5.1 online (`run_glm_online_judge.py`, or self-hosted per
`deploy/glm_5_1_b300/`), Command A self-hosted (`dispatch_command_a_self_hosted.py`), Mistral
Large and GPT-5.5 through native batch (`dispatch_judge_batches.py` / `fetch_judge_batches.py`),
OpenRouter replay (`run_openrouter_judge.py`, `run_parallel_judge_votes.py`). Family exclusion is
encoded in the roster and enforced by the dispatchers.

### D. Offline grading and merge

```bash
python scripts/grade_subject_responses.py --input <responses.jsonl> --expected-batch <batch_input.jsonl> \
    --offline-only --judge-votes-source outputs/judge_votes/<subject>__judge_votes.jsonl \
    --precomputed-extractions outputs/extractor/<subject>__minimax_extractions.jsonl \
    --prompts experiments/aligned/prompts.jsonl --output outputs/grading/<subject>__graded.jsonl --validate-schema
python scripts/merge_grading_outputs.py --out-dir <final_labels_dir> --judge-batches-dir outputs/judge_batches \
    --extractor-cache-dir outputs/cache/extractor --expected-batches-dir <batches_dir>
python scripts/build_route_provenance_resolved_artifacts.py --run-dir <run_dir>
```

### E. Registered analyses

```bash
python scripts/analyze_results.py --inputs <labeled_responses.jsonl ...> --expected-batch <batch_inputs> \
    --output-dir outputs/analysis --figure-dir outputs/figures
```

`analyze_results.py` runs the four confirmatory tests, the sensitivities, the serving-stratum
descriptives, judge agreement, token-count and determinism audits, and, with
`--include-benign-capability`, the benign capability battery as a secondary analysis. Standalone
sensitivities for substituted reserve prompts are `analysis/sensitivity_*.py`. Post hoc
descriptive extensions: `scripts/summarize_failure_taxonomy.py`,
`scripts/analyze_human_validation_reviewers.py` (with `scripts/sample_human_validation_audit.py`
and `scripts/build_human_validation_bundle.py`), `analysis/revision_measurement.py`
(source-verified literal subset, pathway split, missingness bounds) and
`analysis/confirmation_replay.py` (offline confirmation-gate replay into an inert handler).
Their written protocol is `specs/005-kiran-measurement-safeguard/`; `analysis/revision_measurement.py`
records the digest of its `plan.md` in every receipt, and that file ships with one line naming a
local interpreter path generalized. The Gemini direct-API control protocol and its dated
amendments are `specs/002-gemini-direct-control/`, which the Gemini runners digest into their
receipts.

### F. DISABLED+FORMAT three-arm control (post-registration)

Protocol versions, locked-input manifests, route evidence, and governance receipts are in
[`experiments/disabled_format_control/`](experiments/disabled_format_control/), whose README maps
every file; the dated addenda are under `paper/addenda/`. Control requests derive from the stage-B
batch inputs, and each protocol pins the generator, runner, analyzer, grader, extractor, and judge
files by digest, so the runner refuses a changed implementation. This chain runs offline in a
clean checkout:

```bash
python scripts/run_synthetic_prompts.py --batch --batches-dir outputs/main_batches
P=experiments/disabled_format_control/protocol_v3_grok_gpt_kimi.json
L=experiments/disabled_format_control/locked_inputs_manifest_v3_grok_gpt_kimi.json
python scripts/run_disabled_format_control.py --validate --protocol $P --locked-inputs-manifest $L --source-dir outputs/main_batches
python scripts/run_disabled_format_control.py --generate --profile canary --protocol $P --locked-inputs-manifest $L \
    --source-dir outputs/main_batches --output-dir outputs/postreg_disabled_format_control_v3_grok_gpt_kimi/canary/batches
python scripts/run_disabled_format_control_exploratory.py --validate-authorization --profile canary --subject kimi-k2.6
```

Submission adds the dated receipt, `--acknowledge-postregistration-exploratory`,
`--confirm-spend <protocol_id>`, and matching `--max-requests` / `--max-cost-usd` gates, and
spends money. Collection is followed by extraction (stage C), judge votes through OpenRouter only,
offline grading with `--judge-vote-route-profile openrouter`, and
`scripts/analyze_disabled_format_control.py`; the directory README gives the full commands. Other
runners: the OpenRouter `*_openrouter_n3*.py` shards with their `reconcile_*` and `recover_*`
gates, `resume_claude_control_openrouter.py`, and the Gemini direct-API runners
`run_gemini_*_control*.py` with their `resume_gemini_*.py` continuations. The Gemini runners and
resume scripts are bound to the authors' run directories under `outputs/` and are included as the
executed record; adapt their root constants before reuse.

## Layout

| Path | Contents |
|---|---|
| `config/` | Roster, model freeze, case cards, calibration subset, provider compatibility, pricing snapshots |
| `schemas/` | Synthetic tool domains, off-domain pool, visibility canary, record and manifest JSON Schemas |
| `experiments/` | Locked prompt corpus, FORMAT-control protocol chain, benign battery |
| `extractor/`, `grader/`, `judges/` | Extraction, deterministic grading, judge aggregation, locked prompts with digests |
| `scripts/` | Preflight, dispatch, provider adapters, extraction, grading, judge orchestration, analysis CLIs |
| `analysis/` | Estimators, sensitivities, audits, post hoc measurement and replay |
| `serve/`, `deploy/` | vLLM serving for local subjects and the self-hosted GLM 5.1 judge |
| `tests/` | Offline contract and regression tests with synthetic fixtures |
| `paper/` | Study design and statistical analysis plan (digest pinned in `materials_lock.json`), OSF registration text, control protocol addenda |
| `specs/` | Written protocols: the Gemini direct-API control with its dated amendments (`002`) and the post hoc measurement and replay (`005`) |

## Not included, and why

- Raw responses, HTTP archives, extractions, judge votes, graded labels, and
  analysis outputs: retained privately with hash linkage; see the OSF project.
- Corpus-generation and selection tooling, lock/manifest builders, power
  simulation, cost estimators, prompt-cache warmup wrappers, release packaging,
  and the human-validation reviewer web app: not needed to re-run the experiment
  from the locked materials.
- Manuscript sources and rendered PDFs.
- The authors' working notes, task logs, and operational runbooks, and the per-subject
  protocol and lock files of the local-subject control cohort, which were frozen inside their
  run directories and stay with the private audit bundle. The registered design, the
  manuscript, and the protocol records above are the public record.

## Licenses

Source code: [Apache License 2.0](LICENSE). Prompt corpus, manifests, and case-card
artifacts: [CC BY 4.0](LICENSE-CORPUS). Local checkpoints and the self-hosted
Command A judge are subject to their own model licenses.
