from __future__ import annotations

import json

from scripts import run_openrouter_judge as runner


def test_completed_custom_ids_ignore_invalid_or_wrong_route_votes(tmp_path):
    path = tmp_path / "votes.jsonl"
    path.write_text(
        json.dumps(
            {
                "custom_id": "row_invalid",
                "judge": "glm-5.1",
                "label": "Unresolved",
                "provider": "z-ai",
                "model": "z-ai/glm-5.1",
                "judge_prompt_sha256": "current",
            }
        )
        + "\n"
        + json.dumps(
            {
                "custom_id": "row_wrong_judge",
                "judge": "mistral-large-2512",
                "label": "Comply",
                "provider": "z-ai",
                "model": "z-ai/glm-5.1",
                "judge_prompt_sha256": "current",
            }
        )
        + "\n"
        + json.dumps(
            {
                "custom_id": "row_valid",
                "judge": "glm-5.1",
                "label": "Partial",
                "provider": "z-ai",
                "model": "z-ai/glm-5.1",
                "judge_prompt_sha256": "current",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert runner._completed_custom_ids(
        path,
        judge_id="glm-5.1",
        model="z-ai/glm-5.1",
        provider="z-ai",
        prompt_hashes={"row_invalid": "current", "row_wrong_judge": "current", "row_valid": "current"},
    ) == {"row_valid"}


def test_completed_custom_ids_ignore_stale_prompt_hash_votes(tmp_path):
    path = tmp_path / "votes.jsonl"
    path.write_text(
        json.dumps(
            {
                "custom_id": "row_stale",
                "judge": "glm-5.1",
                "label": "Partial",
                "provider": "z-ai",
                "model": "z-ai/glm-5.1",
                "judge_prompt_sha256": "old",
            }
        )
        + "\n"
        + json.dumps(
            {
                "custom_id": "row_current",
                "judge": "glm-5.1",
                "label": "Comply",
                "provider": "z-ai",
                "model": "z-ai/glm-5.1",
                "judge_prompt_sha256": "current",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert runner._completed_custom_ids(
        path,
        judge_id="glm-5.1",
        model="z-ai/glm-5.1",
        provider="z-ai",
        prompt_hashes={"row_stale": "current", "row_current": "current"},
    ) == {"row_current"}


def test_openrouter_judge_raises_infrastructure_error_on_persistent_402(monkeypatch):
    """Six consecutive 402 (credit-exhaustion) responses must raise
    JudgeInfrastructureError, not silently mark the row Unresolved."""
    import pytest
    from judges.judge import JudgeInfrastructureError, MAX_JUDGE_TRANSPORT_ATTEMPTS

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)

    class Response:
        status_code = 402
        headers = {}
        text = "credits exhausted"

        def raise_for_status(self):
            from requests import HTTPError

            raise HTTPError(f"{self.status_code} Client Error", response=self)

        def json(self):  # pragma: no cover - not reached
            return {}

    monkeypatch.setattr(runner.requests, "post", lambda *a, **k: Response())
    client = runner.OpenRouterJudgeClient("glm-5.1")

    with pytest.raises(JudgeInfrastructureError) as excinfo:
        client.evaluate_rendered("judge prompt")
    assert len(excinfo.value.attempts) == MAX_JUDGE_TRANSPORT_ATTEMPTS


def test_openrouter_judge_records_runtime_stamp_on_valid_attempt(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    class Response:
        status_code = 200
        headers = {"x-request-id": "judge_req_1", "authorization": "Bearer secret"}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "judge_gen_1",
                "model": "z-ai/glm-5.1",
                "created": 1770000001,
                "system_fingerprint": "fp_judge",
                "choices": [
                    {
                        "message": {"content": "Label: C\nReason: valid"},
                        "finish_reason": "stop",
                    }
                ],
            }

    def fake_post(*args, **kwargs):
        return Response()

    monkeypatch.setattr(runner.requests, "post", fake_post)
    client = runner.OpenRouterJudgeClient("glm-5.1")

    result = client.evaluate_rendered("judge prompt")

    stamp = result["runtime_stamp"]
    assert stamp["role"] == "judge"
    assert stamp["model_id"] == "glm-5.1"
    assert stamp["route"] == "openrouter"
    assert stamp["provider"] == "z-ai"
    assert stamp["requested_model"] == "z-ai/glm-5.1"
    assert stamp["returned_model"] == "z-ai/glm-5.1"
    assert stamp["response_id"] == "judge_gen_1"
    assert stamp["response_created"] == 1770000001
    assert stamp["system_fingerprint"] == "fp_judge"
    assert stamp["finish_reason"] == "stop"
    assert stamp["response_request_id"] == "judge_req_1"
    assert result["attempts"][0]["runtime_stamp"] == stamp


def test_gpt55_openrouter_judge_uses_openai_pinned_protocol_body(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    captured = {}

    class Response:
        status_code = 200
        headers = {"x-request-id": "judge_req_gpt55"}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "judge_gen_gpt55",
                "model": "openai/gpt-5.5-20260423",
                "choices": [
                    {
                        "message": {"content": "Label: C\nReason: valid"},
                        "finish_reason": "stop",
                    }
                ],
            }

    def fake_post(*args, **kwargs):
        captured["json"] = kwargs["json"]
        return Response()

    monkeypatch.setattr(runner.requests, "post", fake_post)

    client = runner.OpenRouterJudgeClient("gpt-5.5-judge", max_tokens=4096)
    result = client.evaluate_rendered("judge prompt")

    body = captured["json"]
    assert result["label"] == "Comply"
    assert body["model"] == "openai/gpt-5.5"
    assert body["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
        "data_collection": "deny",
    }
    assert body["plugins"] == [{"id": "context-compression", "enabled": False}]
    assert body["transforms"] == []
    assert body["max_completion_tokens"] == 4096
    assert "max_tokens" not in body
    assert "temperature" not in body
    assert "top_p" not in body
    assert "top_k" not in body


def test_main_returns_nonzero_when_all_new_votes_are_invalid(tmp_path, monkeypatch):
    inputs = tmp_path / "judge_inputs.jsonl"
    out = tmp_path / "votes.jsonl"
    inputs.write_text(
        json.dumps({"custom_id": "row_a", "judge_prompt": "prompt a"}) + "\n"
        + json.dumps({"custom_id": "row_b", "judge_prompt": "prompt b"}) + "\n",
        encoding="utf-8",
    )

    class InvalidClient:
        judge_id = "glm-5.1"
        slug = "z-ai/glm-5.1"
        pinned_provider = "z-ai"

        def __init__(self, judge_id, min_interval_seconds=0.0, **kwargs):
            self.judge_id = judge_id

        def evaluate_rendered(self, judge_prompt):
            return {
                "label": "Unresolved",
                "reason": "judge_invalid_after_3_attempts",
                "raw_text": "",
                "attempts": [
                    {
                        "judge_id": self.judge_id,
                        "attempt_index": 1,
                        "status": "invalid",
                        "invalid_reason": "empty_content",
                    }
                ],
            }

    monkeypatch.setattr(runner, "OpenRouterJudgeClient", InvalidClient)

    rc = runner.main(
        [
            "--judge-inputs",
            str(inputs),
            "--out",
            str(out),
            "--judge",
            "glm-5.1",
        ]
    )

    assert rc == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["label"] for row in rows] == ["Unresolved", "Unresolved"]
    assert all(row["judge_prompt_sha256"] for row in rows)


def test_main_returns_nonzero_when_any_new_vote_is_invalid(tmp_path, monkeypatch):
    inputs = tmp_path / "judge_inputs.jsonl"
    out = tmp_path / "votes.jsonl"
    inputs.write_text(
        json.dumps({"custom_id": "row_valid", "judge_prompt": "valid prompt"}) + "\n"
        + json.dumps({"custom_id": "row_invalid", "judge_prompt": "invalid prompt"}) + "\n",
        encoding="utf-8",
    )

    class MixedClient:
        judge_id = "glm-5.1"
        slug = "z-ai/glm-5.1"
        pinned_provider = "z-ai"

        def __init__(self, judge_id, min_interval_seconds=0.0, **kwargs):
            self.judge_id = judge_id

        def evaluate_rendered(self, judge_prompt):
            if "valid" in judge_prompt and "invalid" not in judge_prompt:
                return {
                    "label": "Comply",
                    "reason": "valid",
                    "raw_text": "Label: C\nReason: valid",
                    "attempts": [],
                }
            return {
                "label": "Unresolved",
                "reason": "judge_invalid_after_3_attempts",
                "raw_text": "",
                "attempts": [
                    {
                        "judge_id": self.judge_id,
                        "attempt_index": 1,
                        "status": "invalid",
                        "invalid_reason": "empty_content",
                    }
                ],
            }

    monkeypatch.setattr(runner, "OpenRouterJudgeClient", MixedClient)

    rc = runner.main(
        [
            "--judge-inputs",
            str(inputs),
            "--out",
            str(out),
            "--judge",
            "glm-5.1",
        ]
    )

    assert rc == 1


def test_main_retries_stale_openrouter_vote_without_duplicate_rows(tmp_path, monkeypatch):
    inputs = tmp_path / "judge_inputs.jsonl"
    out = tmp_path / "votes.jsonl"
    inputs.write_text(
        json.dumps({"custom_id": "already-valid", "judge_prompt": "Prompt A"}) + "\n"
        + json.dumps({"custom_id": "stale-row", "judge_prompt": "Prompt B"}) + "\n",
        encoding="utf-8",
    )
    out.write_text(
        json.dumps(
            {
                "custom_id": "already-valid",
                "judge": "glm-5.1",
                "label": "Refuse",
                "provider": "z-ai",
                "model": "z-ai/glm-5.1",
                "judge_prompt_sha256": runner._sha256_text("Prompt A"),
            }
        )
        + "\n"
        + json.dumps(
            {
                "custom_id": "stale-row",
                "judge": "glm-5.1",
                "label": "Partial",
                "provider": "z-ai",
                "model": "z-ai/glm-5.1",
                "judge_prompt_sha256": "old-hash",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class ValidClient:
        judge_id = "glm-5.1"
        slug = "z-ai/glm-5.1"
        pinned_provider = "z-ai"

        def __init__(self, judge_id, min_interval_seconds=0.0, **kwargs):
            self.judge_id = judge_id

        def evaluate_rendered(self, judge_prompt):
            return {
                "label": "Comply",
                "reason": judge_prompt,
                "raw_text": f"Label: C\nReason: {judge_prompt}",
                "attempts": [],
            }

    monkeypatch.setattr(runner, "OpenRouterJudgeClient", ValidClient)

    rc = runner.main(
        [
            "--judge-inputs",
            str(inputs),
            "--out",
            str(out),
            "--judge",
            "glm-5.1",
        ]
    )

    assert rc == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["custom_id"] for row in rows] == ["already-valid", "stale-row"]
    assert rows[1]["label"] == "Comply"
    assert rows[1]["judge_prompt_sha256"] == runner._sha256_text("Prompt B")
    ignored = out.with_suffix(out.suffix + ".ignored.jsonl")
    assert "old-hash" in ignored.read_text(encoding="utf-8")
