"""Unit tests for scripts/smoke_canary.py.

Exercises the canary's body construction, response-shape validator, and
dry-run path. Live HTTP is not exercised here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.smoke_canary import (  # noqa: E402
    INTERVENTIONS,
    _post,
    build_canonical_body,
    run_canary,
    validate_response_shape,
)


# ---------------------------------------------------------------------------
# build_canonical_body
# ---------------------------------------------------------------------------

def test_build_canonical_body_notools_has_no_tools_field():
    body = build_canonical_body("model-x", "NOTOOLS")
    assert body["model"] == "model-x"
    assert "tools" not in body
    assert "tool_choice" not in body


def test_build_canonical_body_disabled_sets_tool_choice_none():
    body = build_canonical_body("model-x", "DISABLED")
    assert body["tools"] and body["tools"][0]["function"]["name"] == "ping"
    assert body["tool_choice"] == "none"


def test_build_canonical_body_tool_sets_tool_choice_auto():
    body = build_canonical_body("model-x", "TOOL")
    assert body["tools"] and body["tools"][0]["function"]["name"] == "ping"
    assert body["tool_choice"] == "auto"


def test_build_canonical_body_tool_irrelevant_uses_off_domain_tool():
    body = build_canonical_body("model-x", "TOOL_IRRELEVANT")
    name = body["tools"][0]["function"]["name"]
    assert name == "off_domain_lookup_postal_code"
    assert body["tool_choice"] == "auto"


def test_build_canonical_body_unknown_intervention_raises():
    with pytest.raises(ValueError):
        build_canonical_body("model-x", "MYSTERY_ARM")


def test_build_canonical_body_has_no_or_only_fields():
    """All four interventions must produce provider-portable canonical bodies."""
    from scripts._tool_schema_guard import OPENROUTER_ONLY_BODY_KEYS

    for interv in INTERVENTIONS:
        body = build_canonical_body("model-x", interv)
        for k in OPENROUTER_ONLY_BODY_KEYS:
            assert k not in body, f"{interv}: leaked OR-only key {k!r}"


def test_openrouter_post_omits_attribution_headers(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "OK"}}]}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    import requests

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(requests, "post", fake_post)

    body = build_canonical_body("test/model", "NOTOOLS")
    _post(
        "openrouter",
        {"pinned_provider": "test-provider"},
        {"subject_id": "test-subject"},
        body,
    )

    assert captured["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert "provider" in captured["json"]
    assert "HTTP-Referer" not in captured["headers"]
    assert "X-Title" not in captured["headers"]


# ---------------------------------------------------------------------------
# validate_response_shape
# ---------------------------------------------------------------------------

def _ok_response(*, content: str | None = "OK", tool_calls: list | None = None,
                 finish_reason: str = "stop") -> dict:
    msg: dict = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"finish_reason": finish_reason, "message": msg}]}


def test_validate_response_shape_passes_simple_text():
    ok, _ = validate_response_shape(_ok_response(content="OK"), "NOTOOLS")
    assert ok


def test_validate_response_shape_passes_tool_call_in_TOOL_cell():
    payload = _ok_response(content=None, tool_calls=[{"id": "1", "type": "function",
                                                       "function": {"name": "ping",
                                                                    "arguments": "{}"}}])
    ok, _ = validate_response_shape(payload, "TOOL")
    assert ok


def test_validate_response_shape_fails_when_finish_reason_is_error():
    payload = {"choices": [{"finish_reason": "error", "message": {"content": "OK"}}]}
    ok, reason = validate_response_shape(payload, "NOTOOLS")
    assert not ok
    assert "finish_reason" in reason


def test_validate_response_shape_fails_when_provider_returned_error_field():
    payload = {"error": {"code": 400, "message": "bad request"}}
    ok, reason = validate_response_shape(payload, "NOTOOLS")
    assert not ok
    assert "error" in reason.lower()


def test_validate_response_shape_fails_when_choices_missing():
    ok, reason = validate_response_shape({}, "NOTOOLS")
    assert not ok
    assert "choices" in reason


def test_validate_response_shape_fails_when_choices_empty():
    ok, reason = validate_response_shape({"choices": []}, "NOTOOLS")
    assert not ok


def test_validate_response_shape_fails_on_empty_content_in_non_TOOL_cell():
    payload = _ok_response(content="")
    ok, reason = validate_response_shape(payload, "DISABLED")
    assert not ok
    assert "content" in reason


def test_validate_response_shape_fails_on_empty_tool_call_in_TOOL_cell():
    payload = _ok_response(content="", tool_calls=[])
    ok, reason = validate_response_shape(payload, "TOOL")
    assert not ok


def test_validate_response_shape_passes_text_in_TOOL_cell():
    """A TOOL-augmented model is allowed to answer with text instead of a
    tool call on the canary prompt — we only require *some* signal."""
    payload = _ok_response(content="OK")
    ok, _ = validate_response_shape(payload, "TOOL")
    assert ok


# ---------------------------------------------------------------------------
# run_canary — dry-run end-to-end
# ---------------------------------------------------------------------------

def test_run_canary_dry_run_all_subjects_pass(tmp_path):
    rc, report = run_canary(live=False, subjects_filter=None, out_dir=tmp_path)
    assert rc == 0, [c for c in report["cells"] if c["status"].startswith("failed")]
    assert report["all_passed"] is True
    assert report["n_passed"] == report["n_cells"]
    assert (tmp_path / "canary_report.json").exists()
    assert (tmp_path / "canary_report.md").exists()


def test_run_canary_dry_run_filter_to_one_subject(tmp_path):
    rc, report = run_canary(
        live=False, subjects_filter=["claude-opus-4-7"], out_dir=tmp_path
    )
    assert rc == 0
    assert report["n_subjects"] == 1
    assert report["n_cells"] == len(INTERVENTIONS)


def test_run_canary_dry_run_unknown_subject_returns_config_error(tmp_path):
    rc, report = run_canary(
        live=False, subjects_filter=["does-not-exist"], out_dir=tmp_path
    )
    assert rc == 2
    assert "no subjects matched" in report["error"]
