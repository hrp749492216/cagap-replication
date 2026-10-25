# DISABLED+FORMAT post-registration control

A post-registration exploratory control that asks whether an output-format instruction,
rather than tool callability, explains the main-study gap between the `TOOL` and `DISABLED`
arms. It is outside H1/H2 and their Holm family. This directory holds every protocol version,
locked-input manifest, route-evidence record, and governance receipt; the dated addenda are
under [`../../paper/addenda/`](../../paper/addenda/).

## Design

Every control request is derived from a main-run batch input (stage B of the root README), so
prompt, persona, trial, visible schemas, decoding, and route are held fixed within a cell.

| Arm | Request |
|---|---|
| `DISABLED_BASELINE` | The unchanged `DISABLED` request: six schemas visible, `tool_choice: none`. |
| `DISABLED_FORMAT_JSON` | The same request with the exact frozen JSON-format instruction appended to the system message after two newline characters. |
| `TOOL_CONTEMPORANEOUS` (v3 only) | The paired `TOOL` request, sent through the same route at the same time. |

The v2 protocol (frozen 2026-08-15) is the two-arm contrast on all ten subjects: 11,520 pairs,
23,040 rows, plus 20 excluded canary rows; its primary quantity is the paired E2 risk difference
`DISABLED_FORMAT_JSON − DISABLED_BASELINE`. The v3 protocols add the contemporaneous `TOOL` arm.
Their primary quantity is the residual `TOOL_CONTEMPORANEOUS − DISABLED_FORMAT_JSON`;
`DISABLED_FORMAT_JSON − DISABLED_BASELINE` measures the format contribution and
`TOOL_CONTEMPORANEOUS − DISABLED_BASELINE` the total same-route gap. Analyses report estimates with
prompt-cluster bootstrap intervals, complete-set attrition, and adversarial missingness bounds;
canary rows never enter an endpoint.

## Protocol chain

| File | Role |
|---|---|
| `protocol.json`, `locked_inputs_manifest.json` | v2 two-arm protocol `CAGAP-POSTREG-DISABLED-FORMAT-2026-08-15-v2` on the original routes, with the checksums of its generated inputs. Preserved; superseded by v3. |
| `protocol_v3_draft.json` | Ten-subject all-OpenRouter three-arm draft `…-2026-09-01-v3-DRAFT` (11,520 sets, 34,560 rows, 30 canary rows). Non-executable (`promotable: false`); preserved as history. |
| `preflight_v3_2026-09-01.json` | Live route sweep for the draft: 4 of 10 exact OpenRouter routes passed. |
| `format_pilot_v3_2026-09-02.json`, `kimi_native_batch_pilot_v3_2026-09-03.json`, `openrouter_requalification_v3_2026-09-07.json` | Diagnostic-only pilots on excluded prompts: FORMAT output usability on the passing routes, one native Kimi batch, and a recheck of the four allow-list-blocked routes (all rejected before inference). None enters an endpoint. |
| `protocol_v3_grok_gpt_kimi.json`, `locked_inputs_manifest_v3_grok_gpt_kimi.json` | Frozen three-route successor `…-2026-09-03-v3-GROK-GPT-KIMI`: Kimi K2.6 through the native Moonshot batch API, Grok 4.3 through OpenRouter pinned to `xai/zdr`, GPT-5.5 through OpenRouter pinned to `openai/flex`. 3,456 complete sets, 10,368 rows, 9 canary rows. |
| `osf_filing_manifest_v3_grok_gpt_kimi.json`, `execution_authorization_v3_grok_gpt_kimi_TEMPLATE.json` | Filing packet and receipt template prepared on 2026-09-03 for an OSF-filed activation; superseded by the exploratory activation below and preserved unchanged. |
| `execution_governance_amendment_v3_grok_gpt_kimi_2026-09-05.json`, `execution_authorization_v3_grok_gpt_kimi.json` | Owner decision that the control is exploratory and needs no new OSF filing, and the receipt authorizing the canary profile for Kimi, Grok, and GPT. Approval document: `paper/addenda/DISABLED_FORMAT_CONTROL_EXECUTION_GOVERNANCE_2026-09-05.md`. |
| `protocol_v3_openrouter_n3_eligible.json`, `locked_inputs_manifest_v3_openrouter_n3_eligible.json` | OpenRouter successor `…-2026-09-05-v3-OPENROUTER-N3-ELIGIBLE` for every supportable exact route: Llama 3.3 70B on `coreweave/fp16`, Grok 4.3 on `xai/zdr`, Claude Opus 4.7 on `anthropic`, GPT-5.5 on `openai/flex`. 4,608 complete sets, 13,824 rows, 12 canary rows. Provider fallback and data collection disabled. |
| `execution_governance_amendment_v3_openrouter_n3_eligible_2026-09-05.json`, `execution_authorization_v3_openrouter_n3_eligible.json` | Governance amendment and canary-profile receipt for that successor. Approval document: `paper/addenda/DISABLED_FORMAT_CONTROL_OPENROUTER_N3_EXECUTION_GOVERNANCE_2026-09-05.md`. |
| `canary_review_v3_openrouter_n3_eligible_2026-09-05.json` | Hash-bound review of the twelve canary rows: Grok, Claude, and GPT passed; Llama was stopped because its FORMAT arm returned top-level keys `type/name/parameters` instead of the frozen `name/arguments`. |
| `full_activation_amendment_v3_openrouter_n3_eligible_2026-09-05.json`, `execution_authorization_v3_openrouter_n3_eligible_full.json` | Full-shard authorization for Grok, Claude, and GPT (3,456 rows each); none for Llama. |

Addenda: `DISABLED_FORMAT_CONTROL_2026-08-15.md` (v2), `DISABLED_FORMAT_CONTROL_2026-09-01_DRAFT.md`
(the preserved ten-subject predecessor), `DISABLED_FORMAT_CONTROL_2026-09-03.md` (frozen
three-route design), and the two execution-governance documents of 2026-09-05.

The Gemini subject was collected through Google's direct API under the protocol and dated
amendments in [`../../specs/002-gemini-direct-control/`](../../specs/002-gemini-direct-control/)
with the `run_gemini_*` and `resume_gemini_*` runners. The remaining subjects (the four locally
served models on vLLM and GLM 4.7 Flash through a separate hosted route) ran under per-subject
protocols frozen inside their run directories; those protocol files stay with the private audit
bundle.

## Pinned implementation

Each protocol records SHA-256 digests of the files that generate, grade, and analyze its rows
(`grading_contract_sha256`, `implementation_contract_sha256`), and
`scripts/run_disabled_format_control.py` refuses to validate, generate, or submit when a live file
differs. In this package the live files match the v3 pins: the request generator
`scripts/run_synthetic_prompts.py`, the runner, `scripts/analyze_disabled_format_control.py`,
`tests/test_disabled_format_control.py`, `tests/test_disabled_format_control_v3_activation.py`,
`grader/semantic_predicates.py`, and the other grader, extractor, and judge files.
`tests/fixtures/frozen_control_20260905/` holds byte-exact copies of three of them for tests that
rebuild the frozen repository view. The v2 manifest pins earlier revisions of the runner, analyzer,
and test module that are not shipped, so v2 is inspectable but not re-executable here.

## Offline validation

These commands make no network request and spend nothing. Stage B first regenerates the main-run
batch inputs the control derives from:

```bash
python scripts/run_synthetic_prompts.py --batch --batches-dir outputs/main_batches
P=experiments/disabled_format_control/protocol_v3_grok_gpt_kimi.json
L=experiments/disabled_format_control/locked_inputs_manifest_v3_grok_gpt_kimi.json
python scripts/run_disabled_format_control.py --validate --protocol $P --locked-inputs-manifest $L \
    --source-dir outputs/main_batches
python scripts/run_disabled_format_control.py --dry-run-canary --protocol $P --locked-inputs-manifest $L \
    --source-dir outputs/main_batches
python scripts/run_disabled_format_control.py --generate --profile canary --protocol $P \
    --locked-inputs-manifest $L --source-dir outputs/main_batches \
    --output-dir outputs/postreg_disabled_format_control_v3_grok_gpt_kimi/canary/batches
python scripts/run_disabled_format_control_exploratory.py --validate-authorization --profile canary \
    --subject kimi-k2.6
python scripts/run_disabled_format_control.py --validate-protocol-draft \
    --protocol experiments/disabled_format_control/protocol_v3_draft.json
```

`--validate` reports 3,456 rows per arm and the conservative cost ceiling per subject;
`--validate-authorization` reports `locked_rows_by_subject: {"kimi-k2.6": 3}` and
`network_request_made: false`. Regeneration reproduces the archived main-run inputs byte for byte
for seven subjects. The archived Claude, Gemma, and GLM inputs predate later roster route updates
and differ in route metadata, and the OpenRouter n=3 protocol also binds route-preflight evidence
files; protocols that pin those artifacts validate only against the authors' archived copies.

## Execution, grading, and analysis

Submission requires the matching dated receipt and explicit gates in one command, and it spends
money:

```bash
python scripts/run_disabled_format_control_exploratory.py --submit --profile canary --subject <subject> \
    --protocol $P --locked-inputs-manifest $L \
    --execution-governance-amendment experiments/disabled_format_control/execution_governance_amendment_v3_grok_gpt_kimi_2026-09-05.json \
    --execution-authorization experiments/disabled_format_control/execution_authorization_v3_grok_gpt_kimi.json \
    --acknowledge-postregistration-exploratory --confirm-spend <protocol_id> \
    --max-requests <locked row count> --max-cost-usd <authorized ceiling>
```

Run each subject's canary, review it, then its full shard. Native batches are fetched with
`scripts/fetch_native_batch_results.py`; OpenRouter shards use
`scripts/run_disabled_format_control_openrouter_n3.py` with the `reconcile_*` and `recover_*`
gates. After collection, run per subject:

```bash
python scripts/run_minimax_batch_extraction.py --input <responses.jsonl> \
    --out outputs/<control_root>/extractions/<subject>.jsonl --cache-dir outputs/<control_root>/extractor_cache/<subject>
python scripts/dispatch_judge_batches.py --input <responses.jsonl> --prompts experiments/aligned/prompts.jsonl \
    --out-dir outputs/<control_root>/judge_batches/<subject> --subject-id <subject> \
    --extractions outputs/<control_root>/extractions/<subject>.jsonl
python scripts/run_parallel_judge_votes.py --judge-batches-dir outputs/<control_root>/judge_batches \
    --judges auto --include-command-a-api-in-auto --command-a-route openrouter
python scripts/grade_subject_responses.py --input <responses.jsonl> --expected-batch <control batch input> \
    --precomputed-extractions outputs/<control_root>/extractions/<subject>.jsonl \
    --offline-only --judge-vote-route-profile openrouter \
    --judge <panel judge 1> --judge <panel judge 2> --judge <panel judge 3> \
    --judge-votes-source <votes 1> --judge-votes-source <votes 2> --judge-votes-source <votes 3> \
    --validate-schema --output outputs/<control_root>/final_labels/<subject>/labeled_responses.jsonl
python scripts/analyze_disabled_format_control.py --labels <final label files> --protocol $P \
    --locked-inputs-manifest $L --expected-batches-dir <control batches> --output outputs/<control_root>/analysis/control.json
```

Judge votes must come through the OpenRouter route profile; a native or local vote fails the
grading preflight. The analysis requires the exact locked identity set; `--allow-incomplete-input`
is for an explicitly interim diagnostic only, and absent identities stay in the missingness
denominator.
