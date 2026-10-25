<!-- generated-by: gsd-doc-writer -->
# Test fixtures

Deterministic, reusable inputs for cross-module statistical pipeline tests.

## Contents

`statistical_pipeline_fixture.py` builds a full synthetic analysis frame from the live 192 design cells and ten-subject roster. It includes:

- the four baseline modes, two personas, and three trials;
- deterministic paired endpoint events;
- registered hardening and quantization sensitivity rows;
- route, runtime, grading, and judge provenance fields;
- expected primary counts used by integration assertions.

The fixture is synthetic test data. It never reads model responses or calls external providers.

## Use

```python
from tests.fixtures.statistical_pipeline_fixture import (
    build_full_statistical_pipeline_fixture,
)

fixture = build_full_statistical_pipeline_fixture()
rows = fixture.rows
```

Keep fixture formulas deterministic and cheap enough for repeatable CI. If the frozen roster, corpus size, modes, or sensitivity design changes, update the fixture and the corresponding integration tests together.

