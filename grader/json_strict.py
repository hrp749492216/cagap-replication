"""Shared strict-JSON ingest helper for the source-design
finite-number gate.

Every JSON ingest path that may carry numeric-bearing payloads from
the corpus, generator, validator, runner, or lock pipeline must go
through ``strict_json_loads`` instead of bare ``json.loads`` so that
both literal NaN/Infinity tokens AND overflowing exponents (1e999 →
±inf) are rejected at parse time before they reach jsonschema, the
predicate gate, or downstream serializers. Duplicate object keys are
also rejected before Python can collapse them with last-key-wins
semantics.

Centralizing here addresses the v3.3 Phase 3 follow-up review NIT
about helper duplication across multiple scripts and prevents drift
in the strict-ingest contract.
"""
from __future__ import annotations

import json
import math
from typing import Any


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """object_pairs_hook that rejects duplicate JSON object keys.

    Python's default decoder keeps the last value for duplicate keys. That is
    not strict enough for source-of-truth records because schema validation
    would only see the collapsed object.
    """
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(
                f"Duplicate JSON object key {key!r} rejected at parse time."
            )
        out[key] = value
    return out


def reject_nonfinite(token: str) -> Any:
    """parse_constant hook. JSON.loads invokes this for the literal
    tokens NaN, Infinity, and -Infinity. Reject all three so they
    cannot reach the schema-validation or predicate layer."""
    raise ValueError(
        f"Non-finite JSON numeric token {token!r} rejected at parse time."
    )


def safe_float(s: str) -> float:
    """parse_float hook. parse_constant only catches the literal NaN/
    Infinity/-Infinity tokens; overflowing exponents like 1e999 go
    through parse_float (Python default = float()) and silently
    produce ±inf, bypassing parse_constant. This hook closes that
    bypass by rejecting any non-finite result."""
    try:
        f = float(s)
    except (OverflowError, ValueError) as e:
        raise ValueError(f"Non-parseable float {s!r}: {e}") from e
    if not math.isfinite(f):
        raise ValueError(f"Non-finite float {s!r} rejected at parse time.")
    return f


def strict_json_loads(text: str | bytes | bytearray) -> Any:
    """Strict json.loads with finite-number and duplicate-key guards.

    Use everywhere a numeric-bearing or source-of-truth JSON payload enters
    the pipeline.
    """
    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
        parse_float=safe_float,
    )
