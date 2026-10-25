"""Benign capability battery prompt contract."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parent.parent
BATTERY_PATH = REPO_ROOT / "experiments" / "benign_capability_battery" / "prompts.jsonl"
SCHEMAS_DIR = REPO_ROOT / "schemas" / "synthetic"
MANIFEST_PATH = SCHEMAS_DIR / "manifest.json"

EXPECTED_DOMAINS = {
    "aerocontrol",
    "bankledger",
    "cloudorchestrator",
    "gridsim",
    "medrecordapi",
    "trafficcomm",
}
EXPECTED_PROMPTS_PER_DOMAIN = 5
EXPECTED_TOTAL_PROMPTS = 30
EXPECTED_MODE_BEHAVIOR_KEYS = {"TOOL", "DISABLED", "NOTOOLS", "TOOL_IRRELEVANT"}
REQUIRED_METADATA_FIELDS = {
    "domain_original",
    "manuscript_domain",
    "benign_subtype",
    "risk_boundary",
    "expected_mode_behavior",
    "grading_notes",
}
DOMAIN_ORIGINAL = {
    "aerocontrol": "AeroControl",
    "bankledger": "BankLedger",
    "cloudorchestrator": "CloudOrchestrator",
    "gridsim": "GridSim",
    "medrecordapi": "MedRecordAPI",
    "trafficcomm": "TrafficComm",
}
MANUSCRIPT_DOMAIN = {
    "aerocontrol": "avionics",
    "bankledger": "financial ledger",
    "cloudorchestrator": "cloud orchestration",
    "gridsim": "industrial control",
    "medrecordapi": "clinical records",
    "trafficcomm": "smart-city traffic",
}
REQUIRED_GRADING_METRICS = {
    "TOOL_correct_call_rate",
    "TOOL_false_refusal_rate",
    "structured_intent_success_rate",
    "wrong_tool_rate",
    "hallucinated_answer_rate",
    "availability_calibrated_rate",
}
MIN_COMPLEX_ROWS = 9
MIN_NONMUTATING_CHECK_ROWS = 5
MIN_PROMPT_OPENINGS = 12

# The raw JSON Schema currently enforces only the integer range for this field;
# the tool description narrows the fixture values to this discrete set.
ALLOWED_ACTUATOR_IDS = {10, 11, 20, 21, 30, 31}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _battery_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in BATTERY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tool_schemas() -> dict[str, dict[str, dict]]:
    by_domain: dict[str, dict[str, dict]] = {}
    for path in SCHEMAS_DIR.glob("*.json"):
        if path.name == "manifest.json":
            continue
        data = _load_json(path)
        by_domain[data["domain"].lower()] = {
            tool["name"]: tool
            for tool in data["tools"]
        }
    return by_domain


def _benign_tools() -> set[tuple[str, str]]:
    manifest = _load_json(MANIFEST_PATH)
    return {
        (entry["domain"].lower(), entry["tool_name"])
        for entry in manifest
        if entry["risk_class"] == "benign"
    }


def test_benign_capability_battery_shape_and_domains():
    rows = _battery_rows()
    assert len(rows) == EXPECTED_TOTAL_PROMPTS
    assert len({row["benign_id"] for row in rows}) == EXPECTED_TOTAL_PROMPTS

    by_domain = Counter(row["domain"] for row in rows)
    assert set(by_domain) == EXPECTED_DOMAINS
    assert set(by_domain.values()) == {EXPECTED_PROMPTS_PER_DOMAIN}


def test_benign_capability_targets_are_benign_study_tools():
    benign_tools = _benign_tools()
    for row in _battery_rows():
        assert (
            row["domain"],
            row["benign_target_tool"],
        ) in benign_tools, f"{row['benign_id']} targets a non-benign or missing tool"


def test_benign_capability_args_validate_against_tool_schemas():
    schemas = _tool_schemas()
    for row in _battery_rows():
        tool = schemas[row["domain"]][row["benign_target_tool"]]
        jsonschema.Draft202012Validator(tool["parameters"]).validate(row["canonical_args"])


def test_benign_capability_prompt_context_is_self_contained():
    for row in _battery_rows():
        prompt = row["prompt_text"]
        assert "synthetic" in prompt.lower()
        assert "benchmark" in prompt.lower()
        assert row["canonical_args"]["request_context_id"] in prompt


def test_benign_capability_actuator_fixture_uses_documented_id():
    row = next(row for row in _battery_rows() if row["benign_id"] == "BCB-AERO-02")
    assert row["canonical_args"]["actuator_id"] in ALLOWED_ACTUATOR_IDS


def test_benign_capability_rows_include_reviewer_facing_metadata():
    for row in _battery_rows():
        assert REQUIRED_METADATA_FIELDS.issubset(row), row["benign_id"]
        assert row["domain_original"] == DOMAIN_ORIGINAL[row["domain"]]
        assert row["manuscript_domain"] == MANUSCRIPT_DOMAIN[row["domain"]]
        assert row["benign_subtype"]
        assert "synthetic" in row["risk_boundary"]

        mode_behavior = row["expected_mode_behavior"]
        assert set(mode_behavior) == EXPECTED_MODE_BEHAVIOR_KEYS
        assert "exact" in mode_behavior["TOOL"]
        assert "structured intent" in mode_behavior["DISABLED"]
        assert "unavailable" in mode_behavior["NOTOOLS"]
        assert "available but irrelevant" in mode_behavior["TOOL_IRRELEVANT"]
        assert "irrelevant" in mode_behavior["TOOL_IRRELEVANT"]

        assert REQUIRED_GRADING_METRICS.issubset(set(row["grading_notes"]))


def test_benign_capability_prompt_templates_are_varied():
    openings = {
        " ".join(row["prompt_text"].split()[:5]).lower()
        for row in _battery_rows()
    }
    assert len(openings) >= MIN_PROMPT_OPENINGS


def test_benign_capability_sensitive_identifiers_are_clearly_synthetic():
    for row in _battery_rows():
        prompt = row["prompt_text"]
        args = row["canonical_args"]

        if row["domain"] == "bankledger" and any("account" in key for key in args):
            assert "TEST-ACCT-" in prompt

        if row["domain"] == "medrecordapi" and "patient_id" in args:
            assert "SYN-PAT-" in prompt
            assert "de-identified synthetic patient" in prompt.lower()

        if row["domain"] == "cloudorchestrator":
            assert "prod-" not in prompt.lower()
            assert not any(
                isinstance(value, str) and value.startswith("prod-")
                for value in args.values()
            )


def test_benign_capability_includes_richer_argument_and_nonmutating_checks():
    rows = _battery_rows()
    complex_rows = [
        row for row in rows
        if len(row["canonical_args"]) >= 3
        or any(isinstance(value, list) and len(value) >= 2 for value in row["canonical_args"].values())
    ]
    nonmutating_rows = [
        row for row in rows
        if row["benign_subtype"].startswith("nonmutating_")
    ]

    assert len(complex_rows) >= MIN_COMPLEX_ROWS
    assert len(nonmutating_rows) >= MIN_NONMUTATING_CHECK_ROWS
