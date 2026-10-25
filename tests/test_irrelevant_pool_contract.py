"""TOOL_IRRELEVANT pool contract.

The irrelevant pool must be benign, fictional, off-domain, and mechanically
disjoint from the study target tools before the runner can expose it as a
fourth mode.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_DIR = REPO_ROOT / "schemas" / "irrelevant_pool"
TASKS_PATH = POOL_DIR / "capability_irrelevant_tasks.jsonl"
ROOT_TASKS_PATH = REPO_ROOT / "tasks" / "capability_irrelevant.jsonl"

EXPECTED_DOMAINS = {
    "library_catalog",
    "garden_simulator",
    "cooking_recipes",
    "music_player",
    "weather_forecast",
}
EXPECTED_TOOLS_PER_DOMAIN = 8
EXPECTED_TOTAL_TOOLS = 40

BANNED_REAL_TERMS = re.compile(
    r"\b(Modbus|FHIR|CAN|SAE J1939|J1939|DNP3|Kubernetes|SSH|SQL)\b"
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pool_domain_files() -> list[Path]:
    manifest = _load_json(POOL_DIR / "manifest.json")
    return [POOL_DIR / item["file"] for item in manifest["domains"]]


def _pool_tools() -> list[dict]:
    tools: list[dict] = []
    for path in _pool_domain_files():
        data = _load_json(path)
        for tool in data["tools"]:
            tools.append({**tool, "domain": data["domain"]})
    return tools


def _study_tool_and_arg_names() -> set[str]:
    names: set[str] = set()
    case_cards = _load_json(REPO_ROOT / "config" / "prompt_case_cards.json")
    for card in case_cards["case_cards"]:
        names.add(card["dangerous_tool"])
        names.add(card["benign_twin"])
    for path in (REPO_ROOT / "schemas" / "synthetic").glob("*.json"):
        if path.name == "manifest.json":
            continue
        data = _load_json(path)
        for tool in data["tools"]:
            names.add(tool["name"])
            names.update(tool["parameters"].get("properties", {}).keys())
    return names


def test_irrelevant_pool_files_validate_against_schemas():
    manifest_schema = _load_json(POOL_DIR / "manifest.schema.json")
    domain_schema = _load_json(POOL_DIR / "tool_domain.schema.json")
    manifest = _load_json(POOL_DIR / "manifest.json")

    jsonschema.Draft202012Validator(manifest_schema).validate(manifest)
    for path in _pool_domain_files():
        jsonschema.Draft202012Validator(domain_schema).validate(_load_json(path))


def test_irrelevant_pool_counts_and_domains():
    manifest = _load_json(POOL_DIR / "manifest.json")
    domains = {item["domain"] for item in manifest["domains"]}
    assert domains == EXPECTED_DOMAINS

    tools = _pool_tools()
    assert len(tools) == EXPECTED_TOTAL_TOOLS
    by_domain = {domain: 0 for domain in EXPECTED_DOMAINS}
    for tool in tools:
        by_domain[tool["domain"]] += 1
    assert set(by_domain.values()) == {EXPECTED_TOOLS_PER_DOMAIN}
    assert len({tool["name"] for tool in tools}) == EXPECTED_TOTAL_TOOLS


def test_capability_tasks_live_with_irrelevant_pool():
    assert TASKS_PATH.exists()
    assert not ROOT_TASKS_PATH.exists()


def test_irrelevant_pool_is_disjoint_from_study_tools_and_args():
    study_names = _study_tool_and_arg_names()
    overlaps: list[str] = []
    for tool in _pool_tools():
        if tool["name"] in study_names:
            overlaps.append(tool["name"])
        for arg_name in tool["parameters"].get("properties", {}):
            if arg_name in study_names:
                overlaps.append(f"{tool['name']}.{arg_name}")
    assert not overlaps, "irrelevant pool overlaps study tool/arg names: " + ", ".join(overlaps)


def test_capability_tasks_cover_each_irrelevant_tool_once():
    tools = _pool_tools()
    tool_names = {tool["name"] for tool in tools}
    rows = [
        json.loads(line)
        for line in TASKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == EXPECTED_TOTAL_TOOLS
    assert {row["tool_name"] for row in rows} == tool_names
    assert len({row["task_id"] for row in rows}) == EXPECTED_TOTAL_TOOLS
    for row in rows:
        assert row["domain"] in EXPECTED_DOMAINS
        assert row["expected_capability"]
        assert row["prompt"]


def test_irrelevant_pool_text_has_no_real_protocol_leakage():
    paths = _pool_domain_files() + [POOL_DIR / "manifest.json", TASKS_PATH]
    violations = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = BANNED_REAL_TERMS.search(line)
            if match:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {match.group(0)}")
    assert not violations, "real protocol leakage in irrelevant pool:\n" + "\n".join(violations)
