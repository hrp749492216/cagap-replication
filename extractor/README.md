<!-- generated-by: gsd-doc-writer -->
# Extractor

Structured extraction and parsing safeguards for converting text-model responses into auditable action records.

## Contents

| File | Responsibility |
|---|---|
| `extract.py` | Canonical response rendering, think-tag stripping, decode-loop provenance, extractor invocation, strict YAML parsing, and action-surface validation |
| `cache.py` | Version-aware disk cache keyed by response, prompt, schema, rubric, and predicate-contract hashes |
| `prompt.md` | Locked extractor instruction |
| `prompt.sha256` | Expected SHA-256 digest of `prompt.md` |
| `__init__.py` | Package marker |

The extractor preserves raw upstream responses outside this package. Its normalization and guards determine what may proceed to deterministic predicate grading and what must remain on the judge path.

## Integration

Use the grading orchestration in `scripts/grade_subject_responses.py` or the MiniMax batch path in `scripts/run_minimax_batch_extraction.py`. Direct calls should construct the same roster-backed route and retain the returned provenance fields.

The cache defaults to `outputs/cache/extractor/`, which is intentionally ignored. Cache entries are invalidated when the extractor prompt, output schema, rubric, or predicate contract changes.

## Validation

```bash
python -m pytest \
  tests/test_extractor.py \
  tests/test_prompt_freeze.py \
  tests/test_grade_subject_responses_offline.py \
  -q
```

After editing `prompt.md`, regenerate its digest through the owning lock workflow and run the prompt-freeze tests. Do not update `prompt.sha256` merely to silence a mismatch without reviewing the measurement change.

