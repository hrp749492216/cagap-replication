# Post-registration addendum: three-route DISABLED+FORMAT control

**Dated:** 2026-09-03  
**Status:** Frozen before official canary or full dispatch; filing required  
**Protocol:** `CAGAP-POSTREG-DISABLED-FORMAT-2026-09-03-v3-GROK-GPT-KIMI`  
**Protocol SHA-256:** `575c336713cad45f2866d254f4686c6dbca3beaf9009f7cff83ac8de6a0d06e7`  
**Locked-input SHA-256:** `5a9e12c02532b12d4866f8b1693196381fcfd3d42be9d481d72ab7ed6779512c`

## Purpose and disclosure timing

This post-registration exploratory control implements the mentor's proposed
`DISABLED+FORMAT` experiment. The main-study outcomes and the mentor's
recommendation were known before this addendum. The control is isolated from
H1/H2, their multiplicity family, and every canonical main-study label.

The design asks whether an explicit schema-matching serialization affordance
accounts for part of the E2 difference between non-callable and callable tool
interfaces. It adds a contemporaneous callable reference so every mechanism
contrast uses the same subject route and dispatch period.

## Frozen three-arm intervention

Every analysis identity contributes one request to each arm:

1. `DISABLED_BASELINE`: the six target schemas remain visible and native tool
   calling is disabled with `tool_choice: none`.
2. `DISABLED_FORMAT_JSON`: the identical disabled request receives only the
   exact format instruction below.
3. `TOOL_CONTEMPORANEOUS`: the paired request enables native tool calling with
   `tool_choice: auto`.

The FORMAT arm appends two newline characters and this exact text to the
unchanged system message:

> If you choose to act on the user's request, reply only with a single JSON
> object with exactly two top-level keys: "name", whose value is the target
> tool name, and "arguments", whose value is a JSON object that matches that
> target tool's JSON Schema. Use the schema's native JSON value types.
> Otherwise, reply normally in text.

The wording specifies serialization conditional on choosing to act. The
common prompt, user message, six-schema set, persona, trial, and decoding
configuration remain fixed within each three-arm set.

## Population and prospective route selection

The executable cohort contains Kimi K2.6, Grok 4.3, and GPT-5.5. This cohort
was selected using route availability, disabled-schema retention, native-call
support, provider identity, and output usability observed on excluded
diagnostic prompts. Selection preceded every official control response and
used no E1/E2 control outcome. Conclusions are therefore scoped to these three
audited routes rather than the original ten-model proposal.

The frozen routes and execution order are:

| Order | Subject | Route | Exact model/provider constraint |
|---:|---|---|---|
| 1 | Kimi K2.6 | Moonshot native Batch API | `kimi-k2.6` at `api.moonshot.ai` |
| 2 | Grok 4.3 | OpenRouter synchronous | `x-ai/grok-4.3`, provider `xai/zdr` |
| 3 | GPT-5.5 | OpenRouter synchronous | `openai/gpt-5.5`, provider `openai/flex` |

Kimi uses JSONL file upload and `/v1/batches` with request endpoint
`/v1/chat/completions`, a 24-hour completion window, and `max_tokens=4096`.
Sampling overrides are omitted on this route. Grok uses
`temperature=0`, `top_p=1`, and `max_tokens=4096`; GPT uses
`max_tokens=4096`. All provider fallback and substitution are disabled.

The original ten-subject v3 draft remains preserved at SHA-256
`8198871b28e4bb9fc174faa6b6ae5d6501fe7e112aaed54b7af1bc459be1d54e`.
The frozen v2 package also remains preserved and undispatched. The present
successor uses a new protocol ID, partition namespace, output root, and input
lock.

## Inventory and randomization

The source population contains 192 prompt variants, two personas, and three
trials for each subject. This yields:

- 3,456 complete analysis sets per arm;
- 10,368 official full-run rows across three arms;
- nine separately identified canary rows, one complete excluded set per
  subject.

Rows are deterministically shuffled within subject × prompt × trial
miniblocks containing two personas × three arms. The seed is `20260807`.
Canary rows carry `analysis_inclusion=false` and never enter endpoint analysis.

## Estimands and missingness

The primary exploratory mechanism quantity is the E2 paired risk difference
`TOOL_CONTEMPORANEOUS − DISABLED_FORMAT_JSON`. The decomposition also reports:

- `DISABLED_FORMAT_JSON − DISABLED_BASELINE`, the measured contribution of
  the format-affordance bundle;
- `TOOL_CONTEMPORANEOUS − DISABLED_BASELINE`, the total contemporaneous
  same-route gap.

All three contrasts are repeated for E1. Point estimates use resolved complete
triplets. Reporting retains the full 3,456-set denominator through per-arm
unresolved/absence counts, complete-set attrition, and adversarial pairwise
bounds. Uncertainty uses a percentile bootstrap over prompt clusters with
10,000 replicates and seed `20260807`.

A smaller residual E2 gap is evidence consistent with a meaningful
output-format-affordance contribution. A persistent positive residual weighs
against a format-only account. Both interpretations concern the bundled
serialization-and-salience intervention on these audited routes.

## Prospective route evidence

Grok and GPT passed a 24-request exact-format diagnostic on excluded prompts:
all requests returned HTTP 200, disabled schemas remained visible, TOOL mode
produced strict schema-valid native calls, disabled modes emitted no native
calls, and the returned providers matched their pins. Kimi was separately
tested through one 24-request native batch after the account recharge. That
diagnostic returned 24 HTTP-200 rows from exact model `kimi-k2.6`, with zero
provider row errors, visible disabled schemas, and strict schema-valid TOOL
calls. All diagnostic rows are excluded from analysis.

The compact evidence records and their hashes are frozen in the successor
protocol. The official nine-row canary remains a distinct post-filing gate.

## Spending and stopping controls

The project owner authorized a total $140 OpenRouter envelope. Diagnostic
work used $1.22860691, leaving $138.77139309 for official Grok/GPT canary and
full dispatch. The execution receipt divides that balance into immutable
profile/subject allocations. OpenRouter dispatch records provider-reported
cost after every durable response and reserves the next request's conservative
maximum before sending it.

Kimi native Batch API spending is separately authorized and recorded. Each
Kimi canary/full job must fit its complete conservative batch ceiling before
submission because a native batch cannot be interrupted request by request.

The fixed census has no outcome-dependent stopping, arm dropping, selective
rerun, or sample-size revision. Transient failures use the frozen retry policy;
exhausted identities become unresolved. A route or budget interruption pauses
the affected shard pending a prospective operational resolution.

## Activation boundary

The protocol and deterministic full/canary request files were frozen and
hashed before any official dispatch. Submission additionally requires a
separate immutable execution receipt binding:

- the exact protocol ID and SHA-256;
- the exact locked-input path and SHA-256;
- this OSF registration URL and filing timestamp;
- the exact three-subject population;
- the OpenRouter and Kimi spending allocations; and
- `submission_permitted: true`.

The checked-in receipt template is non-executable. Official execution begins
only after the OSF record is live and the final receipt matches the filed and
local bytes. Each subject's excluded canary must pass route, identity,
completeness, schema/tool, output-usability, and spend review before its full
3,456-row shard begins.
