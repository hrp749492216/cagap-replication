"""Schema gate for the live model-freeze file."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_live_model_freeze_validates_against_schema():
    freeze = json.loads((REPO_ROOT / "config" / "model_freeze.json").read_text())
    schema = json.loads((REPO_ROOT / "schemas" / "model_freeze.schema.json").read_text())

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(freeze, schema)
