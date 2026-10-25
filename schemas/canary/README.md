<!-- generated-by: gsd-doc-writer -->
# Schema-visibility canary

Diagnostic tool-schema fixture for checking whether a route preserves model-visible schema content when calls are disabled.

`schema_visibility_canary.json` contains a versioned diagnostic prompt, marker token, purpose statement, creation timestamp, and canary tools. It supports the DISABLED-arm route audit; it is not part of the 192-row outcome corpus and must not be treated as behavioral evidence.

Use it through `scripts/schema_visibility_preflight.py`, which also writes the per-route classification (`schema_visibility_classification.json`).

```bash
python -m pytest \
  tests/test_tool_schema_guard.py \
  tests/test_runner_helpers.py \
  -q
```

Do not replace marker tokens or tool definitions without updating the corresponding tests, route evidence, and locked hashes.

