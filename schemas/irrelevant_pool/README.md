<!-- generated-by: gsd-doc-writer -->
# Irrelevant benign tool pool

Off-domain benign tools and capability tasks used by the `TOOL_IRRELEVANT` control arm.

## Contents

| File or group | Purpose |
|---|---|
| `library_catalog.json` | Eight library-catalog tools |
| `garden_simulator.json` | Eight garden-simulation tools |
| `cooking_recipes.json` | Eight cooking/recipe tools |
| `music_player.json` | Eight music-player tools |
| `weather_forecast.json` | Eight weather-forecast tools |
| `manifest.json` | Versioned five-domain inventory and authoring metadata |
| `manifest.schema.json` | Manifest contract |
| `tool_domain.schema.json` | Per-domain tool-file contract |
| `capability_irrelevant_tasks.jsonl` | Forty benign capability tasks, one for each pool tool |

These tools are intentionally irrelevant to the harmful synthetic requests in the main corpus. They let the study separate generic tool availability from relevance and callability of the target tool.

## Validation

```bash
python -m pytest \
  tests/test_irrelevant_pool_contract.py \
  -q
```

Keep each task aligned with its declared tool and expected capability. Changes affect runner payloads, capability checks, expansion hashes, and material locks.

