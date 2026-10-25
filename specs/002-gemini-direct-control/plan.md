# Implementation Plan: Gemini direct Google control

**Branch**: `002-gemini-direct-control` | **Date**: 2026-10-05 | **Spec**: [spec.md](spec.md)

## Summary
Collect the missing 3,456 Gemini control observations through Google's direct OpenAI-compatible endpoint, retaining FORMAT noncompliance under a prospectively recorded exploratory amendment. Reuse the frozen generator and grading/analysis functions through an isolated adapter; do not edit historical executables.

## Technical Context
**Language/Version**: Python 3.14, existing code-review environment.  
**Primary Dependencies**: requests, jsonschema, pytest/xdist, existing CAGAP modules.  
**Storage**: hashed JSON and physical-LF JSONL, append/fsync ledgers and private raw HTTP bodies.  
**Testing**: pytest -n 4, adapter contract suite plus existing disabled-format and offline grading suites.  
**Target Platform**: local macOS, direct Google HTTPS.  
**Project Type**: research pipeline.  
**Performance Goals**: 12 workers, frozen six-row blocks assigned modulo12, no intra-block reordering; extractor8, bounded provider-specific judges.  
**Constraints**: Exact model gemini-3.5-flash, temperature0/top_p1/max_tokens4096; no route fallback; $100 Google cap, $25 grading cap with ledger verification and trial ceilings.  
**Scale/Scope**: 6 excluded diagnostic requests,3 excluded study canary rows,3456 full rows;1152 expected triplets.

## Constitution Check
I PASS: separate post-registration exploratory namespace; confirmatory artifacts unchanged. II PASS: all three arms contemporaneous on one Google route, request differences validated. III PASS: hash-bound inputs/protocol/executable contracts; operational gates and fail-closed identity checks. IV PASS: tests written before adapter; frozen seeds; bounded xdist. V PASS: exact owner instruction recorded before dispatch, bounded spend, synthetic materials only, no real tool execution. Post-design review PASS. Optional broad git auto-commit hooks skipped to preserve unrelated dirty work.

## Project Structure
- `scripts/run_gemini_direct_control.py`: isolated direct transport, durable reservation/restart safety, freeze and canary gates.
- `tests/test_gemini_direct_control.py`: transport and contract tests.
- `outputs/postreg_disabled_format_control_v3_gemini_google_20261005/`: protocol, batches, route checks, raw archive, responses and receipts.
- `outputs/postreg_disabled_format_control_gemini_grading_20261005/`: frozen sources, extraction, votes, labels and analyses.
- `specs/002-gemini-direct-control/`: spec, research, contracts, tasks, verification.

## Execution and analysis
Direct synchronous transport avoids a new batch conversion and allows completion in this session. Batch pricing is half standard but its asynchronous processing does not preserve the existing dispatch-block execution contract. Standard prices checked at $1.50 input/$9 output per million; conservative input byte ceiling plus4096 output reservations, settled from usage including reasoning. A dispatch reservation is persisted before sending; ambiguous interruptions stop for reconciliation. HTTP429/500/502/503/504 may retry up to3 attempts, all attempts archived and reserved; HTTP200 outcomes never retried. Terminal access errors halt, successful in-flight results drain. Model drift and disabled native calls stop with evidence.

Use existing generator and native run-record normalization. Google endpoint fixed to `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`; tool_choice none with tool schema retained. No reasoning override, response-format constraint or new safety prompt. Verify six-mode schema/callability evidence, then three excluded study canaries. FORMAT noncompliance, refusals and truncation remain outcomes. Full population serialization audit avoids conditioning on compliance.

Reuse MiniMax exact-text cache and frozen judge panel. Gemini family google uses GLM5.1, Command A and Mistral. Continue shared Cohere004 budget687/1000 already used, maximum313 further calls including retries; prefer validated exact judge cache reuse, otherwise existing-authorized OpenRouter Cohere route if available and prospectively recorded. No judge-family substitution. All analysis uses existing seed and10000 bootstrap replicates, with explicit unresolved observations and missingness bounds. No mechanism claim solely from a TOOL–FORMAT residual when serialization is not matched.
