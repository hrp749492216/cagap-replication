"""Tests for the provider/adapter tool-choice semantics canary."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tool_choice_semantics_canary import (  # noqa: E402
    classify_route,
    run_semantics_canary,
    prepare_route_cell,
    prompt_input_tokens,
    tool_call_count,
)


def _entry(family: str = "openai") -> dict:
    return {
        "display_name": "Test Model",
        "family": family,
        "native": {
            "endpoint": "https://api.example.test",
            "url_path": "/v1/chat/completions",
            "slug": "test-model",
            "auth_env": "_TEST_API_KEY",
        },
        "openrouter": {
            "slug": "test/test-model",
            "pinned_provider": "test-provider",
        },
    }


def _response(prompt_tokens: int | None, *, tool_calls: list | None = None) -> dict:
    message: dict = {"content": "OK"}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    usage = {} if prompt_tokens is None else {"prompt_tokens": prompt_tokens}
    return {
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": usage,
    }


def test_prepare_route_cell_native_anthropic_preserves_disabled_tools_and_none():
    cell = prepare_route_cell(
        subject_id="claude-opus-4-7",
        entry=_entry("anthropic"),
        route="native",
        route_cfg=_entry("anthropic")["native"],
        intervention="DISABLED",
    )

    canonical = cell["canonical_body"]
    serialized = cell["serialized_body"]

    assert canonical["tool_choice"] == "none"
    assert canonical["tools"][0]["function"]["name"] == "ping"
    assert serialized["tool_choice"] == {"type": "none"}
    assert serialized["tools"][0]["name"] == "ping"
    assert cell["canonical_request_sha256"] != cell["serialized_request_sha256"]


def test_prepare_route_cell_openrouter_wraps_provider_without_mutating_canonical_body():
    cell = prepare_route_cell(
        subject_id="gpt-5.5",
        entry=_entry("openai"),
        route="openrouter",
        route_cfg=_entry("openai")["openrouter"],
        intervention="DISABLED",
    )

    canonical = cell["canonical_body"]
    serialized = cell["serialized_body"]

    assert "provider" not in canonical
    assert canonical["tool_choice"] == "none"
    assert serialized["provider"] == {
        "only": ["test-provider"],
        "allow_fallbacks": False,
        "data_collection": "deny",
    }
    assert serialized["tool_choice"] == "none"
    assert serialized["tools"] == canonical["tools"]
    assert cell["canonical_request_sha256"] != cell["serialized_request_sha256"]


def test_prompt_input_tokens_normalizes_provider_usage_shapes():
    assert prompt_input_tokens({"usage": {"prompt_tokens": 11}}) == 11
    assert prompt_input_tokens({"usage": {"input_tokens": 12}}) == 12
    assert prompt_input_tokens({"usage": {"tokens": {"input_tokens": 13}}}) == 13
    assert prompt_input_tokens({"usageMetadata": {"promptTokenCount": 14}}) == 14
    assert prompt_input_tokens({"usage": {}}) is None


def test_tool_call_count_handles_openai_message_shape():
    payload = _response(
        10,
        tool_calls=[
            {"type": "function", "function": {"name": "ping", "arguments": "{}"}}
        ],
    )
    assert tool_call_count(payload) == 1
    assert tool_call_count(_response(10)) == 0


def test_classify_route_passes_when_disabled_token_delta_supports_schema_visibility():
    cells = {
        "NOTOOLS": {"status": "passed", "response": _response(10)},
        "DISABLED": {"status": "passed", "response": _response(25)},
        "TOOL": {"status": "passed", "response": _response(26)},
    }

    verdict = classify_route(cells)

    assert verdict["status"] == "pass"
    assert verdict["input_token_deltas"]["DISABLED_minus_NOTOOLS"] == 15
    assert verdict["input_token_deltas"]["TOOL_minus_DISABLED"] == 1


def test_classify_route_warns_when_provider_returns_no_usage_tokens():
    cells = {
        "NOTOOLS": {"status": "passed", "response": _response(None)},
        "DISABLED": {"status": "passed", "response": _response(None)},
        "TOOL": {"status": "passed", "response": _response(None)},
    }

    verdict = classify_route(cells)

    assert verdict["status"] == "warning"
    assert "token usage unavailable" in " ".join(verdict["reasons"])


def test_classify_route_fails_when_disabled_allows_structured_tool_call():
    cells = {
        "NOTOOLS": {"status": "passed", "response": _response(10)},
        "DISABLED": {
            "status": "passed",
            "response": _response(
                25,
                tool_calls=[
                    {"type": "function", "function": {"name": "ping", "arguments": "{}"}}
                ],
            ),
        },
        "TOOL": {"status": "passed", "response": _response(26)},
    }

    verdict = classify_route(cells)

    assert verdict["status"] == "fail"
    assert "DISABLED returned structured tool call" in " ".join(verdict["reasons"])


def test_classify_route_fails_when_disabled_tokens_match_notools():
    cells = {
        "NOTOOLS": {"status": "passed", "response": _response(20)},
        "DISABLED": {"status": "passed", "response": _response(20)},
        "TOOL": {"status": "passed", "response": _response(26)},
    }

    verdict = classify_route(cells)

    assert verdict["status"] == "fail"
    assert "DISABLED input tokens did not exceed NOTOOLS" in " ".join(verdict["reasons"])


def test_classify_route_separates_provider_account_block_from_semantic_fail():
    cells = {
        intervention: {
            "status": "failed_route_unavailable",
            "reason": "HTTPError: 402 Client Error: Payment Required",
            "http_status": 402,
            "response": None,
        }
        for intervention in ("NOTOOLS", "DISABLED", "TOOL")
    }

    verdict = classify_route(cells)

    assert verdict["status"] == "route_unavailable"
    assert verdict["tool_choice_none_interpretation"] == "not_exercised_route_unavailable"
    assert "HTTP 402" in " ".join(verdict["reasons"])


def test_prepare_route_cell_hashes_are_stable_json_sha256():
    cell = prepare_route_cell(
        subject_id="gpt-5.5",
        entry=_entry("openai"),
        route="openrouter",
        route_cfg=_entry("openai")["openrouter"],
        intervention="TOOL",
    )

    assert cell["canonical_request_sha256"] == (
        "d2311451a565b0fedee96ecae97c48ea340c374af54329b9513a6be11d774d67"
    )


def test_run_semantics_canary_dry_run_writes_json_and_markdown(tmp_path):
    rc, report = run_semantics_canary(
        live=False,
        subjects_filter=["gpt-5.5"],
        route_scope="openrouter",
        out_dir=tmp_path,
    )

    assert rc == 0
    assert report["summary_counts"] == {"dry_run_only": 1}
    assert (tmp_path / "tool_choice_semantics_canary_report.json").exists()
    assert (tmp_path / "tool_choice_semantics_canary_report.md").exists()
