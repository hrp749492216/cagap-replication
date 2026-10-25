<!-- generated-by: gsd-doc-writer -->
# Benign capability battery

A 30-prompt control battery used to distinguish safety-related refusal changes from broad capability degradation.

## Files

| File | Purpose |
|---|---|
| `prompts.jsonl` | Benign tasks across the six synthetic manuscript domains |
| `prompts.sha256` | Checked-in SHA-256 digest for the prompt file |

Each prompt records a stable benign identifier, subtype, domain, target tool, canonical arguments, expected mode behavior, grading notes, and risk-boundary metadata. The tasks reuse synthetic tool semantics and do not act on real services.

## Integrity and tests

Verify the digest:

```bash
shasum -a 256 -c experiments/benign_capability_battery/prompts.sha256
```

Run the battery-specific contracts:

```bash
python -m pytest \
  tests/test_benign_capability_battery.py \
  tests/test_capability_battery.py \
  tests/test_analyze_capability_battery.py \
  -q
```

The dispatch, grading, and analysis entry points are `scripts/run_capability_battery.py`, `scripts/grade_capability_battery.py`, and `scripts/analyze_capability_battery.py`. Dispatch may call remote or self-hosted models; inspect the runner and route configuration before launching it.

