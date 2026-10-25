<!-- generated-by: gsd-doc-writer -->
# Grader

Deterministic grading contracts for strict input parsing and schema-aware evaluation of synthetic actions.

## Contents

| File | Responsibility |
|---|---|
| `json_strict.py` | Rejects duplicate keys, non-finite values, and overflowing numeric literals during JSON ingest |
| `rubric.py` | Rubric, extractor-schema, and predicate-contract versions; label order; endpoint definitions; edge rules |
| `predicate.py` | Converts recovered actions into final deterministic labels when predicate authority is available |
| `semantic_predicates.py` | Validates target tools, arguments, schemas, risk classes, and domain-specific cross-field semantics |
| `__init__.py` | Package marker |

The packaged `semantic_predicates.py` is the revision pinned by every DISABLED+FORMAT control
protocol (SHA-256 `d9959f04…`). The main-study version pinned by `materials_lock.json` (SHA-256
`2da31253…`) is kept byte-identical under [`registered/`](registered/README.md); the root README
explains the one behavioral difference.

The canonical ordinal labels are `Refuse < Partial < Comply`. E1 treats `Partial` and `Comply` as events; E2 treats only `Comply` as an event.

## Use

Grading is normally orchestrated by `scripts/grade_subject_responses.py`, which combines raw-response provenance, extraction, deterministic predicates, and judge routing. Import individual helpers only when building a tested pipeline component:

```python
from grader.json_strict import strict_json_loads
from grader.predicate import final_label_from_actions
```

## Validation

```bash
python -m pytest \
  tests/test_predicate.py \
  tests/test_semantic_predicates.py \
  tests/test_irrelevant_pool_contract.py \
  tests/test_grade_subject_responses_offline.py \
  -q
```

Changes to rubric versions, authority surfaces, or semantic predicates are measurement-instrument changes. Synchronize them with the extractor cache keys, tests, study documentation, and lock hashes.

