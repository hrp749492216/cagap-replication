from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.schema_visibility_preflight import (
    FORMAT_CONTROL_MODE,
    build_openai_body,
    build_routes,
    diagnostic_prompts,
    family_modes,
    load_format_instruction,
)


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "experiments" / "disabled_format_control" / "protocol_v3_draft.json"
ROSTER = REPO / "config" / "roster.json"


def test_format_probe_uses_exact_frozen_instruction_and_only_changes_system_text() -> None:
    route = next(
        route
        for route in build_routes(ROSTER, control_routes_path=PROTOCOL)
        if route["subject_id"] == "grok-4.3"
    )
    prompt = diagnostic_prompts()[0]
    instruction = load_format_instruction(PROTOCOL)

    baseline, baseline_tools = build_openai_body(prompt, route, "DISABLED")
    formatted, formatted_tools = build_openai_body(
        prompt,
        route,
        FORMAT_CONTROL_MODE,
        format_instruction=instruction,
    )

    assert formatted_tools == baseline_tools
    assert formatted["tool_choice"] == baseline["tool_choice"] == "none"
    assert formatted["tools"] == baseline["tools"]
    assert formatted["messages"][0]["content"] == (
        baseline["messages"][0]["content"] + "\n\n" + instruction
    )

    normalized = json.loads(json.dumps(formatted))
    normalized["messages"][0]["content"] = baseline["messages"][0]["content"]
    assert normalized == baseline


def test_format_probe_requires_protocol_instruction() -> None:
    route = next(
        route
        for route in build_routes(ROSTER, control_routes_path=PROTOCOL)
        if route["subject_id"] == "grok-4.3"
    )
    with pytest.raises(ValueError, match="frozen format instruction"):
        build_openai_body(diagnostic_prompts()[0], route, FORMAT_CONTROL_MODE)


def test_format_mode_is_opt_in() -> None:
    route = next(
        route
        for route in build_routes(ROSTER, control_routes_path=PROTOCOL)
        if route["subject_id"] == "grok-4.3"
    )
    assert FORMAT_CONTROL_MODE not in family_modes(route)
    assert FORMAT_CONTROL_MODE in family_modes(route, include_format_control=True)
