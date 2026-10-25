<!-- generated-by: gsd-doc-writer -->
# Schemas

Machine-readable contracts for corpus records, runtime rows, manifests, and synthetic tool definitions.

## Top-level contracts

| File | Validates |
|---|---|
| `aligned_corpus.schema.json` | Selected aligned-corpus JSONL rows |
| `extractor_output.schema.json` | Structured extractor results |
| `run_record.schema.json` | Per-condition subject run records |
| `run_manifest.schema.json` | Run-level provenance and artifact manifests |
| `halt_event.schema.json` | Explicit halted-workflow events |
| `substitution_log.schema.json` | Reserve-card substitution provenance |
| `roster.schema.json` | Model, route, judge, extractor, and validator roster configuration |
| `model_freeze.schema.json` | Frozen model/checkpoint/runtime configuration |

## Tool-schema collections

| Directory | Purpose |
|---|---|
| [`synthetic/`](synthetic/README.md) | Six paired synthetic operational domains used by the main study |
| [`irrelevant_pool/`](irrelevant_pool/README.md) | Five off-domain benign tool sets used by `TOOL_IRRELEVANT` |
| [`canary/`](canary/README.md) | Diagnostic schema-visibility canary |

## Conventions

- JSON instances must reject duplicate keys and non-finite numbers before schema validation.
- Tool argument objects use explicit required fields and `additionalProperties: false` where the protocol requires exact payloads.
- Schema version changes must be coordinated with runners, extractor/grader contracts, tests, and material-lock hashes.
- Synthetic names are protocol-shaped test fixtures, not real service interfaces.

## Validation

```bash
python -m pytest \
  tests/test_irrelevant_pool_contract.py \
  tests/test_model_freeze_schema.py \
  -q
```

