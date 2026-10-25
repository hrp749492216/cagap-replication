from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import jsonschema


PROMPTS_PATH = Path("experiments/aligned/prompts.jsonl")
SCHEMAS_DIR = Path("schemas/synthetic")
EXPECTED_STYLE_SLOTS = {
    "outcome_request_no_tool_name",
    "direct_tool_invocation",
    "operator_ticket",
    "urgency_or_policy_bypass_pretext",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_final_aligned_corpus_file_exists():
    assert PROMPTS_PATH.exists()


def test_prompts_count_and_partition():
    prompts = load_jsonl(PROMPTS_PATH)

    assert len(prompts) == 192
    assert Counter(row.get("partition_id") for row in prompts) == {"main": 192}


def test_style_slot_balance():
    prompts = load_jsonl(PROMPTS_PATH)

    assert Counter(row["style_slot"] for row in prompts) == {
        "outcome_request_no_tool_name": 48,
        "direct_tool_invocation": 48,
        "operator_ticket": 48,
        "urgency_or_policy_bypass_pretext": 48,
    }


def test_required_fields_match_current_aligned_corpus_schema():
    required_fields = {
        "case_id",
        "partition_id",
        "source_partition_id",
        "is_substituted",
        "replacement_for",
        "case_card_id",
        "domain",
        "dangerous_tool",
        "benign_twin",
        "style_slot",
        "prompt_text",
        "canonical_args",
        "harm_category",
        "harmful_intent",
        "alignment_rationale",
        "why_safe_model_should_refuse",
        "generator_metadata",
        "validator_a",
        "validator_b",
        "selection_metadata",
        "accepted_at",
    }

    for row in load_jsonl(PROMPTS_PATH):
        missing = required_fields - set(row)
        assert not missing, f"{row.get('case_id', '<unknown>')} missing {sorted(missing)}"


def test_domain_and_tool_names_match_synthetic_schemas():
    schemas = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        if path.name == "manifest.json":
            continue
        schemas[path.stem] = json.loads(path.read_text())

    for row in load_jsonl(PROMPTS_PATH):
        schema = schemas[row["domain"]]
        tool_names = {tool["name"] for tool in schema["tools"]}
        assert row["dangerous_tool"] in tool_names
        assert row["benign_twin"] in tool_names


def test_canonical_args_validate_against_dangerous_tool_schema():
    schemas = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        if path.name == "manifest.json":
            continue
        schemas[path.stem] = json.loads(path.read_text())

    for row in load_jsonl(PROMPTS_PATH):
        schema = schemas[row["domain"]]
        tool_schema = next(
            tool for tool in schema["tools"] if tool["name"] == row["dangerous_tool"]
        )
        jsonschema.Draft202012Validator(tool_schema["parameters"]).validate(
            row["canonical_args"]
        )


def test_case_ids_are_unique_and_style_slots_are_valid():
    prompts = load_jsonl(PROMPTS_PATH)
    case_ids = [row["case_id"] for row in prompts]

    assert len(case_ids) == len(set(case_ids))
    assert {row["style_slot"] for row in prompts} == EXPECTED_STYLE_SLOTS
