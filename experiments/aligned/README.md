<!-- generated-by: gsd-doc-writer -->
# Aligned prompt corpus

The locked main prompt corpus for the CAGAP confirmatory study.

## Files

| File | Purpose |
|---|---|
| `prompts.jsonl` | Canonical 192-row corpus consumed by subject runners and downstream grading |
| `main_prompts.jsonl` | Matching 192-row main-corpus copy used by corpus-gate and validator-replay tooling |
| `selection_manifest.json` | Selection protocol, validator summary, partition counts, input hashes, and exclusion record |
| `pilot_prompts.jsonl` | Archived pre-registration pilot partition used by route-preflight probes and tests |

At the current lock, `prompts.jsonl` and `main_prompts.jsonl` are byte-identical. The A5 corpus gate checks that invariant. The rows span six synthetic domains and four balanced style slots; none of the 192 selected rows is a reserve substitution in the current corpus.

## Data contract

Each JSONL row includes stable case identifiers, the prompt text, synthetic target-tool metadata, canonical arguments, selection/validator provenance, domain and style assignments, and substitution fields. Rows must validate against [`schemas/aligned_corpus.schema.json`](../../schemas/aligned_corpus.schema.json).

The 24 reserve case cards are retained in [`config/prompt_case_cards.json`](../../config/prompt_case_cards.json), not appended to this main corpus.

## Validation

```bash
python -m pytest \
  tests/test_prompt_corpus.py \
  tests/test_prompts_aligned.py \
  -q
```

Do not edit one JSONL copy without the other. The corpus is frozen; `materials_lock.json` pins its digest. The authors' selection, stamping, gate-report, and lock scripts are not part of this package.

