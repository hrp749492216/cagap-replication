"""Tests for scripts/_tool_schema_guard.py.

Covers the three guard rules that protect provider portability:
  1. No top-level oneOf/allOf/anyOf in tool parameters/input_schema.
  2. ``required`` must be a subset of ``properties.keys()``.
  3. OpenRouter-only body keys never appear in stored canonical batch rows.

These rules prevent the asymmetric provider-side rejections that caused
Pilot-0 to fail. See notes/pilot0_failed_harness_incident.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._tool_schema_guard import (  # noqa: E402
    OPENROUTER_ONLY_BODY_KEYS,
    DISALLOWED_TOP_LEVEL_KEYS,
    ToolSchemaError,
    assert_body_clean_of_route_only_fields,
    strip_route_only_fields,
    validate_batch_file,
    validate_batch_row,
    validate_tool,
    validate_tools_in_object,
)


# ---------------------------------------------------------------------------
# validate_tool — top-level oneOf/allOf/anyOf rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", ["oneOf", "allOf", "anyOf"])
def test_validate_tool_rejects_top_level_combinator(kw):
    tool = {
        "name": "do_thing",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            kw: [{"required": ["x"]}],
        },
    }
    with pytest.raises(ToolSchemaError) as exc:
        validate_tool(tool, source="t.json")
    assert kw in str(exc.value)
    assert "Anthropic" in str(exc.value)


def test_validate_tool_accepts_property_level_combinators():
    """Combinators are only forbidden at the TOP level. Inside a property
    they are fine — Anthropic's validator allows them there."""
    tool = {
        "name": "do_thing",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            },
        },
    }
    validate_tool(tool, source="t.json")  # should not raise


# ---------------------------------------------------------------------------
# validate_tool — orphan required rejection
# ---------------------------------------------------------------------------

def test_validate_tool_rejects_orphan_required():
    tool = {
        "name": "do_thing",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x", "y"],  # 'y' not declared
        },
    }
    with pytest.raises(ToolSchemaError) as exc:
        validate_tool(tool, source="t.json")
    assert "y" in str(exc.value)
    assert "Google" in str(exc.value)


def test_validate_tool_accepts_required_subset_of_properties():
    tool = {
        "name": "do_thing",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}, "y": {"type": "integer"}},
            "required": ["x"],
        },
    }
    validate_tool(tool, source="t.json")  # should not raise


def test_validate_tool_accepts_no_required():
    tool = {
        "name": "do_thing",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
        },
    }
    validate_tool(tool, source="t.json")  # should not raise


def test_validate_tool_handles_input_schema_alias():
    """Anthropic-shaped tool dict uses input_schema not parameters."""
    tool = {
        "name": "do_thing",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "allOf": [{"required": ["x"]}],  # forbidden at top level
        },
    }
    with pytest.raises(ToolSchemaError) as exc:
        validate_tool(tool, source="t.json")
    assert "allOf" in str(exc.value)


def test_validate_tool_no_op_on_non_dict_params():
    tool = {"name": "do_thing", "parameters": "not_a_dict"}
    validate_tool(tool, source="t.json")  # should not raise


# ---------------------------------------------------------------------------
# validate_tools_in_object — recursive walker
# ---------------------------------------------------------------------------

def test_validate_tools_in_object_walks_nested_dict():
    obj = {
        "body": {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "ok_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "string"}},
                        },
                    },
                }
            ]
        }
    }
    validate_tools_in_object(obj, source="batch.jsonl")  # should not raise


def test_validate_tools_in_object_catches_nested_violation():
    obj = {
        "tools": [
            {
                "name": "bad_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "anyOf": [{"required": ["x"]}],
                },
            }
        ]
    }
    with pytest.raises(ToolSchemaError):
        validate_tools_in_object(obj, source="batch.jsonl")


def test_validate_tools_in_object_walks_list_of_lists():
    obj = [
        [
            {
                "name": "deep_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["nope"],
                },
            }
        ]
    ]
    with pytest.raises(ToolSchemaError) as exc:
        validate_tools_in_object(obj, source="batch.jsonl")
    assert "nope" in str(exc.value)


# ---------------------------------------------------------------------------
# strip_route_only_fields / assert_body_clean_of_route_only_fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(OPENROUTER_ONLY_BODY_KEYS))
def test_strip_route_only_fields_removes_each_or_only_key(key):
    body = {"model": "x", "messages": [], key: "junk"}
    out = strip_route_only_fields(body)
    assert key not in out
    assert out["model"] == "x"


def test_strip_route_only_fields_removes_all_or_only_keys_at_once():
    body = {
        "model": "x",
        "messages": [],
        "provider": {"only": ["foo"]},
        "plugins": [{"id": "context-compression"}],
        "transforms": [],
        "models": ["fallback-1"],
        "tools": [{"name": "t", "parameters": {"type": "object", "properties": {}}}],
    }
    out = strip_route_only_fields(body)
    for k in OPENROUTER_ONLY_BODY_KEYS:
        assert k not in out
    assert out["model"] == "x"
    assert out["tools"][0]["name"] == "t"


def test_strip_route_only_fields_idempotent_on_clean_body():
    body = {"model": "x", "messages": []}
    out = strip_route_only_fields(body)
    assert out == {"model": "x", "messages": []}


def test_strip_route_only_fields_mutates_in_place():
    body = {"model": "x", "provider": "junk"}
    out = strip_route_only_fields(body)
    assert out is body  # mutation, not copy


def test_assert_body_clean_passes_clean_body():
    assert_body_clean_of_route_only_fields(
        {"model": "x", "messages": []}, source="row1"
    )  # should not raise


@pytest.mark.parametrize("key", list(OPENROUTER_ONLY_BODY_KEYS))
def test_assert_body_clean_raises_on_each_or_only_key(key):
    with pytest.raises(ToolSchemaError) as exc:
        assert_body_clean_of_route_only_fields(
            {"model": "x", key: "junk"}, source="row1"
        )
    assert key in str(exc.value)
    assert "row1" in str(exc.value)


def test_assert_body_clean_lists_all_leaks():
    with pytest.raises(ToolSchemaError) as exc:
        assert_body_clean_of_route_only_fields(
            {
                "model": "x",
                "provider": "junk",
                "plugins": [],
                "transforms": [],
            },
            source="row1",
        )
    msg = str(exc.value)
    assert "provider" in msg
    assert "plugins" in msg
    assert "transforms" in msg


# ---------------------------------------------------------------------------
# validate_batch_row — body + tool checks composed
# ---------------------------------------------------------------------------

def test_validate_batch_row_passes_clean_row():
    row = {
        "custom_id": "cid-1",
        "body": {
            "model": "x",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "good_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    }
    validate_batch_row(row, source="batch.jsonl")  # should not raise


def test_validate_batch_row_catches_or_only_field_in_body():
    row = {
        "custom_id": "cid-1",
        "body": {"model": "x", "provider": "junk", "messages": []},
    }
    with pytest.raises(ToolSchemaError) as exc:
        validate_batch_row(row, source="batch.jsonl")
    assert "cid-1" in str(exc.value)
    assert "provider" in str(exc.value)


def test_validate_batch_row_catches_bad_tool_inside_function_wrapper():
    row = {
        "custom_id": "cid-2",
        "body": {
            "model": "x",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bad_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"x": {"type": "string"}},
                            "oneOf": [{"required": ["x"]}],
                        },
                    },
                }
            ],
        },
    }
    with pytest.raises(ToolSchemaError) as exc:
        validate_batch_row(row, source="batch.jsonl")
    assert "cid-2" in str(exc.value)
    assert "oneOf" in str(exc.value)


def test_validate_batch_row_catches_anthropic_shaped_tool():
    row = {
        "custom_id": "cid-3",
        "body": {
            "model": "x",
            "messages": [],
            "tools": [
                {
                    "name": "anthropic_tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["missing"],
                    },
                }
            ],
        },
    }
    with pytest.raises(ToolSchemaError) as exc:
        validate_batch_row(row, source="batch.jsonl")
    assert "missing" in str(exc.value)


# ---------------------------------------------------------------------------
# validate_batch_file — JSONL aggregation
# ---------------------------------------------------------------------------

def _write_jsonl(tmp_path, name, rows):
    p = tmp_path / name
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_validate_batch_file_passes_clean_file(tmp_path):
    rows = [
        {
            "custom_id": f"cid-{i}",
            "body": {
                "model": "x",
                "messages": [],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "ok",
                            "parameters": {
                                "type": "object",
                                "properties": {"x": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        }
        for i in range(3)
    ]
    p = _write_jsonl(tmp_path, "clean.jsonl", rows)
    n = validate_batch_file(p)
    assert n == 3


def test_validate_batch_file_skips_blank_lines(tmp_path):
    p = tmp_path / "with_blanks.jsonl"
    with p.open("w") as f:
        f.write(json.dumps({"custom_id": "a", "body": {"model": "x", "messages": []}}) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(json.dumps({"custom_id": "b", "body": {"model": "x", "messages": []}}) + "\n")
    assert validate_batch_file(p) == 2


def test_validate_batch_file_rejects_dirty_row(tmp_path):
    rows = [
        {"custom_id": "ok", "body": {"model": "x", "messages": []}},
        {
            "custom_id": "bad",
            "body": {"model": "x", "messages": [], "transforms": []},
        },
    ]
    p = _write_jsonl(tmp_path, "dirty.jsonl", rows)
    with pytest.raises(ToolSchemaError) as exc:
        validate_batch_file(p)
    assert "bad" in str(exc.value)
    assert "transforms" in str(exc.value)


def test_validate_batch_file_rejects_invalid_json(tmp_path):
    p = tmp_path / "broken.jsonl"
    p.write_text('{"custom_id": "ok", "body": {}}\nnot json\n')
    with pytest.raises(ToolSchemaError) as exc:
        validate_batch_file(p)
    assert "invalid JSON" in str(exc.value)
    assert "line 2" in str(exc.value)


# ---------------------------------------------------------------------------
# Provider parity — same canonical body, different adapters consume it cleanly
# ---------------------------------------------------------------------------

def test_canonical_body_is_provider_portable_across_adapters():
    """A canonical batch row stripped of OR-only fields must be accepted
    by the validator regardless of which adapter consumes it next.
    Equivalent to: validate_batch_row passes; strip is idempotent;
    assert_body_clean passes after strip."""
    canonical = {
        "custom_id": "shared-cid",
        "body": {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "max_tokens": 4096,
        },
    }
    # 1. validates as a clean canonical row
    validate_batch_row(canonical, source="canonical")

    # 2. strip is a no-op on already-clean body
    before = json.dumps(canonical["body"], sort_keys=True)
    strip_route_only_fields(canonical["body"])
    after = json.dumps(canonical["body"], sort_keys=True)
    assert before == after

    # 3. assert-clean passes
    assert_body_clean_of_route_only_fields(canonical["body"], source="canonical")

    # 4. now simulate an OR adapter wrapping at dispatch time on a working copy
    or_body = dict(canonical["body"])
    or_body["provider"] = {"only": ["pinned"]}
    or_body["plugins"] = [{"id": "context-compression", "enabled": False}]
    or_body["transforms"] = []
    # OR-wrapped body still has good tools but is no longer canonical
    with pytest.raises(ToolSchemaError):
        assert_body_clean_of_route_only_fields(or_body, source="or_dispatch_copy")
    # ...but stripping the working copy returns it to canonical shape
    strip_route_only_fields(or_body)
    assert or_body == canonical["body"]


# ---------------------------------------------------------------------------
# Schema source-of-truth — the synthetic schemas under schemas/synthetic/
# must remain provider-portable. Lock-time gate.
# ---------------------------------------------------------------------------

def test_synthetic_schema_files_are_provider_portable():
    """Walk every JSON file under schemas/synthetic/ and validate its
    tools. Catches Pilot-0's top-level allOf regression at the source."""
    repo = Path(__file__).resolve().parents[1]
    schema_dir = repo / "schemas" / "synthetic"
    files = sorted(schema_dir.glob("*.json"))
    assert files, "expected at least one synthetic schema file"
    for path in files:
        if path.name == "manifest.json":
            continue
        obj = json.loads(path.read_text())
        validate_tools_in_object(obj, source=str(path))


# ---------------------------------------------------------------------------
# Constants — make them assertable so future refactors break loud
# ---------------------------------------------------------------------------

def test_or_only_keys_constant_is_locked():
    assert OPENROUTER_ONLY_BODY_KEYS == ("provider", "plugins", "transforms", "models")


def test_disallowed_top_level_keys_constant_is_locked():
    assert DISALLOWED_TOP_LEVEL_KEYS == ("oneOf", "allOf", "anyOf")
