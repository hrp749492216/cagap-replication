<!-- generated-by: gsd-doc-writer -->
# Judges

Judge execution, reliability gating, and ordinal vote aggregation for rows that cannot be resolved deterministically.

## Contents

| File | Responsibility |
|---|---|
| `judge.py` | Judge request construction, response parsing, retry/error handling, and per-vote provenance |
| `aggregate.py` | Three-slot ordinal-median consensus and the conservative two-valid-vote fallback |
| `reliability_gate.py` | Gold-set macro-F1 gate for judge validation |
| `prompt.md` | Locked judge instruction |
| `prompt.sha256` | Expected SHA-256 digest of `prompt.md` |
| `__init__.py` | Package marker |

Judge families are selected through the family-exclusion rules in [`config/roster.json`](../config/roster.json). Three valid labels aggregate by ordinal median. With exactly two valid labels, aggregation chooses the lower-ordered label; fewer than two valid labels remain unresolved.

## Integration

Use the batch and merge tools in `scripts/` rather than calling remote judges ad hoc. They preserve slot identity, route metadata, transport failures, subject-output failures, and raw vote records needed for audit.

## Validation

```bash
python -m pytest \
  tests/test_judge.py \
  tests/test_judge_aggregate.py \
  tests/test_judge_panel.py \
  tests/test_reliability_gate.py \
  tests/test_judge_agreement.py \
  -q
```

After changing `prompt.md`, regenerate its digest through the lock workflow and run `tests/test_prompt_freeze.py`. Judge prompt or aggregation changes can alter the measurement instrument and require explicit documentation.

