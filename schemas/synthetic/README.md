<!-- generated-by: gsd-doc-writer -->
# Synthetic study tool schemas

Paired benign and high-impact tool definitions for six fictional operational domains.

## Domains

- `aerocontrol.json`
- `bankledger.json`
- `cloudorchestrator.json`
- `gridsim.json`
- `medrecordapi.json`
- `trafficcomm.json`

`manifest.json` indexes 108 tools: 54 high-impact tools and 54 benign twins across the six domains. Each entry records its domain, operation family, pair identifier, risk class, tool name, and canonical argument fields.

## Contract

The manifest and domain files are the source of truth for tool names and argument schemas used by case cards, runners, and deterministic semantic predicates. Tool pairs must remain distinguishable by risk class while preserving the paired operation structure required by the design.

All services, identifiers, and actions are synthetic. Do not replace them with real endpoints, credentials, account numbers, patient records, grid assets, or other operational data.

## Validation

```bash
python -m pytest \
  tests/test_case_cards.py \
  tests/test_semantic_predicates.py \
  -q
```

After a schema change, also validate the aligned corpus, case cards, source hashes, and material lock.

