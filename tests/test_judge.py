import json
from pathlib import Path

import pytest

from judges.judge import (
    Judge,
    JudgeInfrastructureError,
    MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS,
    MAX_JUDGE_TRANSPORT_ATTEMPTS,
)
from scripts import canary_cohere_judge as cohere_judge


REPO_ROOT = Path(__file__).resolve().parent.parent


class SequencedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, judge_id, entry, messages):
        self.calls.append(
            {"judge_id": judge_id, "entry": entry, "messages": messages}
        )
        if not self.responses:
            raise AssertionError("SequencedClient exhausted")
        return self.responses.pop(0)


def _judge_with_client(client):
    judge = Judge.__new__(Judge)
    judge.judge_id = "glm-5.1"
    judge.entry = {"native": {"slug": "glm-5.1"}}
    judge.prompt_template = (
        "Prompt={{prompt}}\nActions={{actions}}\nResponse={{response}}"
    )
    judge.client = client
    return judge


def test_judge_module_uses_package_imports_without_sys_path_mutation():
    source = Judge.__module__
    assert source == "judges.judge"

    import inspect
    import judges.judge as judge_module

    assert "sys.path.insert" not in inspect.getsource(judge_module)


def test_judge_parser_accepts_case_insensitive_label_field():
    judge = Judge.__new__(Judge)

    label, reason = judge._parse_judge_response(
        "label: c\nreason: The response performs the requested action."
    )

    assert label == "C"
    assert reason == "The response performs the requested action."


def test_judge_parser_accepts_full_label_names():
    judge = Judge.__new__(Judge)

    label, reason = judge._parse_judge_response(
        "Label: Refuse\nReason: Explicit safety refusal."
    )

    assert label == "R"
    assert reason == "Explicit safety refusal."


def test_judge_parser_ignores_embedded_label_mentions():
    judge = Judge.__new__(Judge)

    label, reason = judge._parse_judge_response(
        "Reason: The rubric says not to output Label: C unless complete."
    )

    assert label == "Unresolved"
    assert reason == "The rubric says not to output Label: C unless complete."


def test_judge_prompt_schema_table_covers_high_impact_tools():
    prompt = (REPO_ROOT / "judges" / "prompt.md").read_text(encoding="utf-8")
    manifest = json.loads(
        (REPO_ROOT / "schemas" / "synthetic" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_tools = [
        row["tool_name"] for row in manifest if row["risk_class"] == "high-impact"
    ]

    missing = [tool for tool in expected_tools if f"| `{tool}` |" not in prompt]

    assert missing == []


def test_cohere_judge_uses_global_4096_output_cap(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "cohere-test-key")
    monkeypatch.setattr(cohere_judge, "archive_response_body", lambda *args, **kwargs: None)
    posted: list[dict] = []

    class FakeResponse:
        status_code = 200
        headers = {}
        text = "{}"
        content = b"{}"

        def json(self):
            return {
                "message": {
                    "content": [{"text": "Label: R\nReason: refusal."}]
                },
                "finish_reason": "COMPLETE",
            }

    class FakeSession:
        def post(self, url, **kwargs):
            posted.append({"url": url, **kwargs})
            return FakeResponse()

    client = cohere_judge.CohereJudgeClient(min_interval_seconds=0)
    client.session = FakeSession()

    result = client.evaluate_rendered("judge prompt", custom_id="row-1")

    assert result["label"] == "Refuse"
    assert posted[0]["json"]["max_tokens"] == 4096
    assert posted[0]["json"]["max_tokens"] == cohere_judge.COHERE_JUDGE_MAX_TOKENS


def test_judge_retries_empty_content_and_records_valid_attempt(monkeypatch):
    monkeypatch.setattr("judges.judge._now_utc", lambda: "2026-05-08T00:00:00+00:00")
    client = SequencedClient(
        [
            {
                "_route": "native",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}}
                ],
            },
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "Label: Partial\nReason: hedged action."
                        },
                    }
                ],
            },
        ]
    )
    judge = _judge_with_client(client)

    result = judge.evaluate("prompt text", [], "subject response")

    assert result["label"] == "P"
    assert result["reason"] == "hedged action."
    assert len(client.calls) == 2
    assert result["attempts"] == [
        {
            "judge_id": "glm-5.1",
            "attempt_index": 1,
            "provider_route": "native",
            "timestamp": "2026-05-08T00:00:00+00:00",
            "status": "invalid",
            "http_status": 200,
            "finish_reason": "stop",
            "content_chars": 0,
            "parsed_label": None,
            "invalid_reason": "empty_content",
            "raw_response": {
                "_route": "native",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}}
                ],
            },
        },
        {
            "judge_id": "glm-5.1",
            "attempt_index": 2,
            "provider_route": "native",
            "timestamp": "2026-05-08T00:00:00+00:00",
            "status": "valid",
            "http_status": 200,
            "finish_reason": "stop",
            "content_chars": 37,
            "parsed_label": "Partial",
            "invalid_reason": None,
            "raw_response": {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "Label: Partial\nReason: hedged action."
                        },
                    }
                ],
            },
        },
    ]


def test_judge_stops_after_three_subject_invalid_attempts(monkeypatch):
    """Subject-invalid budget is exhausted by three parseable-but-bad completions."""
    monkeypatch.setattr("judges.judge.time.sleep", lambda _: None)
    client = SequencedClient(
        [
            {
                "_route": "native",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}}
                ],
            },
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "Reason: truncated before label"},
                    }
                ],
            },
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Label: XYZ\nReason: garbage."},
                    }
                ],
            },
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Label: C\nReason: too late."},
                    }
                ],
            },
        ]
    )
    judge = _judge_with_client(client)

    result = judge.evaluate("prompt text", [], "subject response")

    assert result["label"] == "Unresolved"
    assert result["reason"] == "judge_invalid_after_3_attempts"
    assert len(client.calls) == 3
    assert [a["attempt_index"] for a in result["attempts"]] == [1, 2, 3]
    assert [a["invalid_reason"] for a in result["attempts"]] == [
        "empty_content",
        "length_no_label",
        "invalid_or_missing_label",
    ]


def test_judge_transport_errors_do_not_consume_subject_invalid_budget(monkeypatch):
    """A row should still resolve to a valid label after several transport
    failures interleaved with subject-invalid completions, as long as
    neither budget is individually exhausted."""
    monkeypatch.setattr("judges.judge.time.sleep", lambda _: None)
    client = SequencedClient(
        [
            {"_route": "native", "error": "HTTPError 503"},
            {
                "_route": "native",
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}}
                ],
            },
            {"_route": "native", "error": "HTTPError 502"},
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "Reason: truncated"},
                    }
                ],
            },
            {"_route": "native", "error": "HTTPError 429"},
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Label: P\nReason: hedged."},
                    }
                ],
            },
        ]
    )
    judge = _judge_with_client(client)

    result = judge.evaluate("prompt text", [], "subject response")

    assert result["label"] == "P"
    assert len(client.calls) == 6
    reasons = [a["invalid_reason"] for a in result["attempts"]]
    assert reasons == [
        "transport_error",
        "empty_content",
        "transport_error",
        "length_no_label",
        "transport_error",
        None,
    ]


def test_judge_transport_budget_is_six_attempts(monkeypatch):
    """Transport budget is at least MAX_JUDGE_TRANSPORT_ATTEMPTS = 6."""
    assert MAX_JUDGE_TRANSPORT_ATTEMPTS >= 6
    monkeypatch.setattr("judges.judge.time.sleep", lambda _: None)
    client = SequencedClient(
        [{"_route": "native", "error": "HTTPError 502"}] * (MAX_JUDGE_TRANSPORT_ATTEMPTS - 1)
        + [
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Label: R\nReason: refuses."},
                    }
                ],
            }
        ]
    )
    judge = _judge_with_client(client)

    result = judge.evaluate("prompt text", [], "subject response")

    assert result["label"] == "R"
    assert len(client.calls) == MAX_JUDGE_TRANSPORT_ATTEMPTS


def test_judge_raises_infrastructure_error_when_transport_budget_exhausted(monkeypatch):
    """Six consecutive transport failures must raise, not silently mark
    the row Unresolved. The exception carries the attempts log so the
    operator can diagnose and resume."""
    monkeypatch.setattr("judges.judge.time.sleep", lambda _: None)
    client = SequencedClient(
        [{"_route": "native", "error": f"HTTPError 503 attempt {i}"} for i in range(MAX_JUDGE_TRANSPORT_ATTEMPTS)]
    )
    judge = _judge_with_client(client)

    with pytest.raises(JudgeInfrastructureError) as excinfo:
        judge.evaluate("prompt text", [], "subject response")

    err = excinfo.value
    assert err.judge_id == "glm-5.1"
    assert len(err.attempts) == MAX_JUDGE_TRANSPORT_ATTEMPTS
    assert all(a["invalid_reason"] == "transport_error" for a in err.attempts)
    assert "HTTPError 503" in err.last_error


def test_judge_4xx_auth_errors_are_retried_as_transport(monkeypatch):
    """HTTP 401 / 403 / 404 / 402 (OpenRouter credits) are infrastructure
    failures, not subject-output failures. They retry against the
    transport budget and never short-circuit to Unresolved."""
    monkeypatch.setattr("judges.judge.time.sleep", lambda _: None)
    client = SequencedClient(
        [
            {"_route": "openrouter", "error": "HTTPError 402: credits exhausted"},
            {"_route": "openrouter", "error": "HTTPError 401: unauthorized"},
            {"_route": "openrouter", "error": "HTTPError 403: forbidden"},
            {"_route": "openrouter", "error": "HTTPError 404: not found"},
            {"_route": "openrouter", "error": "HTTPError 400: bad request"},
            {
                "_route": "openrouter",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Label: R\nReason: refuses."},
                    }
                ],
            },
        ]
    )
    judge = _judge_with_client(client)

    result = judge.evaluate("prompt text", [], "subject response")

    assert result["label"] == "R"
    assert len(client.calls) == 6
    assert [a["invalid_reason"] for a in result["attempts"][:5]] == ["transport_error"] * 5
    assert result["attempts"][5]["invalid_reason"] is None


def test_judge_raises_infrastructure_error_on_persistent_401(monkeypatch):
    """Six consecutive 401s must raise JudgeInfrastructureError so the
    operator rotates the key, rather than dropping the row into the
    missingness sensitivities."""
    monkeypatch.setattr("judges.judge.time.sleep", lambda _: None)
    client = SequencedClient(
        [{"_route": "openrouter", "error": "HTTPError 401: bad key"}]
        * MAX_JUDGE_TRANSPORT_ATTEMPTS
    )
    judge = _judge_with_client(client)

    with pytest.raises(JudgeInfrastructureError):
        judge.evaluate("prompt text", [], "subject response")


def test_judge_subject_invalid_cap_matches_constant():
    """Pre-registered subject-invalid cap remains 3 (Sen-1 / reliability
    amendment) — only the transport budget changed."""
    assert MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS == 3


def test_judge_never_retries_a_valid_label_value():
    client = SequencedClient(
        [
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Label: R\nReason: refuses."},
                    }
                ],
            },
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Label: C\nReason: would differ."},
                    }
                ],
            },
        ]
    )
    judge = _judge_with_client(client)

    result = judge.evaluate("prompt text", [], "subject response")

    assert result["label"] == "R"
    assert len(client.calls) == 1
    assert result["attempts"][0]["invalid_reason"] is None


def test_judge_retries_429_transport_error():
    client = SequencedClient(
        [
            {"_route": "native", "error": "HTTPError 429: rate limit"},
            {
                "_route": "native",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "Label: C\nReason: recovered."},
                    }
                ],
            },
        ]
    )
    judge = _judge_with_client(client)

    result = judge.evaluate("prompt text", [], "subject response")

    assert result["label"] == "C"
    assert len(client.calls) == 2
    assert result["attempts"][0]["invalid_reason"] == "transport_error"
