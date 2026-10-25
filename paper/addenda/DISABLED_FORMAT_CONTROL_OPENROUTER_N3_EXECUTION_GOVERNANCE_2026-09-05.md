# Exploratory DISABLED+FORMAT OpenRouter n=3 execution governance

Date: 2026-09-05
Status: frozen before the first affected official response

## Owner decision

The project owner directed the study to complete, through OpenRouter only, the model runs that are operationally supportable now with three trials per cell. The owner also confirmed that this is an exploratory study and that no new OSF filing is required.

This decision creates a new successor execution packet. It does not modify the original ten-model draft or the earlier Kimi/Grok/GPT successor; both remain preserved as historical records.

## Frozen candidate cohort

The successor freezes the four exact OpenRouter routes that already had complete prospective evidence for provider pinning, DISABLED-arm schema visibility, and native TOOL-call support:

- `meta-llama/llama-3.3-70b-instruct` via `coreweave/fp16`
- `x-ai/grok-4.3` via `xai/zdr`
- `anthropic/claude-opus-4.7` via `anthropic`
- `openai/gpt-5.5` via `openai/flex`

The design has 192 prompts, two personas, three trials, and three contemporaneous arms. It therefore contains 4,608 paired analysis sets per arm and 13,824 full-run requests, plus 12 excluded canary requests.

Claude remains a candidate because its exact route passed schema and native-call support checks, but its earlier excluded diagnostic produced five null DISABLED-arm outputs. Its full shard is prohibited unless the new three-row canary yields usable non-null responses in all three arms. The same canary gate applies to every candidate. A failed candidate is stopped and reported; it is not replaced.

## Execution gates

Before any paid request, the launcher must validate the exact protocol, generated request files, locked-input manifest, this approval document, governance amendment, execution receipt, provider pins, and per-profile spending allocations.

The first Llama invocation stopped before credential use because the shared original-study roster gives its historical local substrate precedence. With zero network requests and zero affected responses, the packet was regenerated around `scripts/run_disabled_format_control_openrouter_n3.py`. This hash-pinned adapter changes only the process-local roster view for an exact successor OpenRouter route and delegates all scientific, authorization, transport, budget, and durability behavior to the unchanged frozen runners.

The 12 excluded canary requests run first in frozen Llama, Grok, Claude, GPT order. Full dispatch can begin only after the canaries are retrieved and reviewed for transport success, requested/returned model identity, observed provider identity, completeness, schema/tool invariants, non-null output usability, and provider-reported spend.

After review, passing 3,456-row subject shards may run as independent resumable processes. Every response is durably written, fallbacks remain disabled, data collection remains denied, and no route or model substitution is authorized.

## Spending boundary

The OpenRouter account limit remains USD 140.00. Provider-reported usage before this official successor is USD 1.22860691, leaving USD 138.77139309. Canary and full per-subject allocations sum exactly to that remaining amount. Runtime dispatch checks cumulative provider-reported cost and reserves the next request's conservative ceiling before sending it.

No native-provider spending is authorized by this packet. In particular, the historical Kimi native-batch plan is not part of this all-OpenRouter successor.

## Scientific scope

This study is post-registration exploratory. It is excluded from the preregistered H1/H2 family and does not change the original confirmatory estimands or multiplicity plan. Canary results are operational diagnostics only and never enter analysis.
