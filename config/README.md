<!-- generated-by: gsd-doc-writer -->
# Configuration

Versioned study configuration for model routes, corpus construction, compatibility rules, and pricing assumptions.

## Files

| File | Purpose |
|---|---|
| `roster.json` | Operational source of truth for subject models, judge panel, extractor, route metadata, decoding, and family exclusions |
| `model_freeze.json` | Frozen model, checkpoint, route, and runtime identities |
| `provider_compatibility.json` | Provider capabilities and request-shape compatibility rules |
| `prompt_case_cards.json` | 216 authored case cards: 192 main and 24 same-stratum reserve cards |
| `calibration_prompt_ids.json` | Locked prompt subset used by registered calibration/sensitivity work |
| `pre_osf_prompt_repairs_2026-05-14.json` | Provenance for prompt repairs applied before the OSF lock |
| `api_pricings_native_providers.json` | Native-provider cost assumptions |
| `openrouter_pricing_current.json` | OpenRouter cost assumptions used by estimators and planning tools |

The canonical methodological interpretation of these fields lives in [`paper/Study_Design.md`](../paper/Study_Design.md). JSON structure is additionally constrained by [`schemas/`](../schemas/README.md) and the test suite.

## Working with configuration

- Treat identifiers, route semantics, model revisions, seeds, and decoding values as reproducibility data.
- Keep credentials out of these files. Runners load secrets from environment variables.
- Update pricing snapshots deliberately; they are time-sensitive inputs, not permanent facts.
- Any change to a locked file invalidates `materials_lock.json`; the study materials are frozen.

## Validation

```bash
python -m pytest \
  tests/test_model_freeze_schema.py \
  tests/test_provider_compatibility.py \
  tests/test_case_cards.py \
  -q
```

Do not hand-edit generated lock files to make a validation failure disappear. The authors' lock and manifest tooling is not part of this package.

