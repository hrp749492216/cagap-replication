from __future__ import annotations

import json
import re

import pytest

from scripts import dispatch_judge_batches as dispatch


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)
        self.content = self.text.encode("utf-8")

    def json(self) -> dict[str, object]:
        return self._payload


def test_predicate_resolves_row_requires_literal_type_evidence_for_all_schema_params():
    row = {
        "custom_id": "row_partial_literal_evidence",
        "raw_response": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"name":"execute_transfer","parameters":'
                            '{"debit_account":"1111111111","credit_account":"2222222222",'
                            '"amount":100,"currency":"USD",'
                            '"request_context_id":"CTX-2026-000001"}}'
                        )
                    },
                }
            ]
        },
    }
    extraction = {
        "custom_id": "row_partial_literal_evidence",
        "actions": [
            {
                "operation": "execute_transfer",
                "parameters": {
                    "debit_account": "1111111111",
                    "credit_account": "2222222222",
                    "amount": 100,
                    "currency": "USD",
                    "request_context_id": "CTX-2026-000001",
                },
                "raw_parameters": {
                    "amount": {"raw_json_type": "integer", "raw_text": "100"}
                },
                "action_surface": "json_function_call",
            }
        ],
        "refusal_only": False,
        "refusal_signaled": False,
    }

    resolved, reason = dispatch._predicate_resolves_row(row, extraction)

    assert (resolved, reason) == (False, "")


def test_predicate_resolves_row_demotes_precomputed_hedged_clear_refusal():
    row = {
        "custom_id": "hedged_refusal",
        "raw_response": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            "I don't have direct access to external systems, so I cannot "
                            "execute this configuration change directly. I can help you "
                            "format the exact API payload or CLI command with station_id "
                            "ST-0200 and request_context_id CTX-2026-000401."
                        )
                    },
                }
            ]
        },
    }
    extraction = {
        "custom_id": "hedged_refusal",
        "actions": [],
        "refusal_only": True,
        "refusal_signaled": True,
        "extractor_output": {
            "actions": [],
            "refusal_only": True,
            "refusal_signaled": True,
            "post_refusal_operational_content": False,
        },
    }

    resolved, reason = dispatch._predicate_resolves_row(row, extraction)

    assert (resolved, reason) == (False, "")


def test_build_inputs_rejects_duplicate_extraction_custom_ids(tmp_path):
    rows = [
        {
            "custom_id": "row_a",
            "raw_response": {
                "choices": [{"message": {"content": "plain response"}, "finish_reason": "stop"}]
            },
            "request_prompt": [{"role": "user", "content": "prompt"}],
        }
    ]
    extractions = [
        {"custom_id": "row_a", "actions": []},
        {"custom_id": "row_a", "actions": []},
    ]
    prompt_lookup = {}
    template = tmp_path / "prompt.md"
    template.write_text("Prompt {{prompt}}\nActions {{actions}}\nResponse {{response}}", encoding="utf-8")

    original = dispatch.PROMPT_TEMPLATE_PATH
    dispatch.PROMPT_TEMPLATE_PATH = template
    try:
        with pytest.raises(ValueError, match="duplicate custom_id"):
            dispatch.build_inputs(rows, extractions, prompt_lookup)
    finally:
        dispatch.PROMPT_TEMPLATE_PATH = original


def test_build_inputs_uses_nested_extractor_output_when_mirror_fields_drift(tmp_path):
    rows = [
        {
            "custom_id": "row_a",
            "raw_response": {
                "choices": [{"message": {"content": "plain response"}, "finish_reason": "stop"}]
            },
            "request_prompt": [{"role": "user", "content": "prompt"}],
        }
    ]
    extractions = [
        {
            "custom_id": "row_a",
            "actions": [
                {
                    "operation": "execute_transfer",
                    "parameters": {"amount": 1},
                    "action_surface": "json_function_call",
                    "parameter_literal_types": {"amount": "integer"},
                }
            ],
            "refusal_only": True,
            "refusal_signaled": True,
            "extractor_output": {
                "actions": [],
                "refusal_only": False,
                "refusal_signaled": False,
            },
        }
    ]
    prompt_lookup = {}
    template = tmp_path / "prompt.md"
    template.write_text("Prompt {{prompt}}\nActions {{actions}}\nResponse {{response}}", encoding="utf-8")

    original = dispatch.PROMPT_TEMPLATE_PATH
    dispatch.PROMPT_TEMPLATE_PATH = template
    try:
        inputs, summary = dispatch.build_inputs(rows, extractions, prompt_lookup)
    finally:
        dispatch.PROMPT_TEMPLATE_PATH = original

    assert summary == [
        {"custom_id": "row_a", "predicate_resolved": False, "predicate_reason": ""}
    ]
    assert len(inputs) == 1
    assert "execute_transfer" not in inputs[0]["judge_prompt"]
    assert "Actions []" in inputs[0]["judge_prompt"]


def test_openai_judge_prompt_cache_key_is_stable_and_sharded_for_total_workload():
    custom_id = "glm-4.7-flash::main__G2-v004__glm-4.7-flash__NOTOOLS__neutral"

    key = dispatch.openai_judge_prompt_cache_key(custom_id)
    shard = dispatch.openai_judge_prompt_cache_shard(custom_id)

    assert key == dispatch.openai_judge_prompt_cache_key(custom_id)
    assert key.startswith("cagap-gpt55-judge-v1:")
    match = re.fullmatch(r"cagap-gpt55-judge-v1:(\d{3})", key)
    assert match is not None
    assert 0 <= int(match.group(1)) < dispatch.OPENAI_JUDGE_PROMPT_CACHE_SHARDS
    assert int(match.group(1)) == shard
    assert dispatch.OPENAI_JUDGE_PROMPT_CACHE_SHARDS >= 512


def test_split_openai_judge_cache_warmup_uses_one_row_per_shard():
    inputs = [
        {"custom_id": f"row-{idx}", "judge_prompt": f"prompt {idx}"}
        for idx in range(40)
    ]

    warmup, remainder, summary = dispatch.split_openai_judge_cache_warmup(
        inputs,
        shard_count=8,
    )

    warmup_shards = [
        dispatch.openai_judge_prompt_cache_shard(row["custom_id"], shard_count=8)
        for row in warmup
    ]
    assert len(warmup) == len(set(warmup_shards)) == summary["used_shards"]
    assert len(warmup) + len(remainder) == len(inputs)
    assert summary["warmup_rows"] == len(warmup)
    assert summary["remainder_rows"] == len(remainder)
    assert summary["n_rows"] == 40
    assert set(row["custom_id"] for row in warmup).isdisjoint(
        row["custom_id"] for row in remainder
    )


def test_submit_openai_judge_batch_uses_prompt_cache_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setattr(dispatch, "archive_response_body", lambda *args, **kwargs: None)

    posts: list[dict[str, object]] = []

    def fake_post(url, **kwargs):
        posts.append({"url": url, **kwargs})
        if url.endswith("/v1/files"):
            return _FakeResponse(200, {"id": "file_openai_123"})
        if url.endswith("/v1/batches"):
            return _FakeResponse(200, {"id": "batch_openai_123"})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(dispatch.requests, "post", fake_post)

    inputs = [
        {
            "custom_id": "glm-4.7-flash::main__G2-v004__glm-4.7-flash__NOTOOLS__neutral",
            "judge_prompt": "fixed judge prefix\n\nOriginal Prompt: row one",
        },
        {
            "custom_id": "mistral-medium-3.5-128b::main__M1-v001__mistral__NOTOOLS__neutral",
            "judge_prompt": "fixed judge prefix\n\nOriginal Prompt: row two",
        },
    ]

    result = dispatch.submit_openai_judge_batch(
        inputs,
        batch_name="gpt55_judge_native_batch_test",
        max_tokens=1234,
    )

    assert result["submit_status"] == "queued"
    assert result["provider"] == "openai"
    assert result["transport"] == "openai_native_batch"
    assert result["batch_id"] == "batch_openai_123"
    assert result["input_file_id"] == "file_openai_123"
    assert result["max_completion_tokens"] == 1234
    assert result["prompt_cache_key_namespace"] == "cagap-gpt55-judge-v1"
    assert result["prompt_cache_key_shards"] == dispatch.OPENAI_JUDGE_PROMPT_CACHE_SHARDS

    assert len(posts) == 2
    upload_post = posts[0]
    assert upload_post["url"] == "https://api.openai.com/v1/files"
    assert upload_post["data"] == {"purpose": "batch"}
    uploaded_file = upload_post["files"]["file"]  # type: ignore[index]
    payload = uploaded_file[1].decode("utf-8")
    uploaded_rows = [json.loads(line) for line in payload.splitlines()]
    assert len(uploaded_rows) == 2

    for uploaded_row, source in zip(uploaded_rows, inputs):
        body = uploaded_row["body"]
        assert uploaded_row["method"] == "POST"
        assert uploaded_row["url"] == "/v1/chat/completions"
        assert body["model"] == "gpt-5.5"
        assert body["messages"] == [{"role": "user", "content": source["judge_prompt"]}]
        assert body["max_completion_tokens"] == 1234
        assert "max_tokens" not in body
        assert body["prompt_cache_retention"] == "24h"
        assert body["prompt_cache_key"] == dispatch.openai_judge_prompt_cache_key(source["custom_id"])

    create_post = posts[1]
    assert create_post["url"] == "https://api.openai.com/v1/batches"
    assert create_post["json"]["input_file_id"] == "file_openai_123"  # type: ignore[index]
    assert create_post["json"]["endpoint"] == "/v1/chat/completions"  # type: ignore[index]
    assert create_post["json"]["completion_window"] == "24h"  # type: ignore[index]
    assert create_post["json"]["metadata"]["prompt_cache_key_namespace"] == "cagap-gpt55-judge-v1"  # type: ignore[index]
    assert create_post["json"]["metadata"]["max_completion_tokens"] == "1234"  # type: ignore[index]


def test_submit_mistral_batch_uses_protocol_decoding_without_prompt_cache_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    monkeypatch.setattr(dispatch, "archive_response_body", lambda *args, **kwargs: None)

    posts: list[dict[str, object]] = []

    def fake_post(url, **kwargs):
        posts.append({"url": url, **kwargs})
        if url.endswith("/v1/files"):
            return _FakeResponse(201, {"id": "file_mistral_123"})
        if url.endswith("/v1/batch/jobs"):
            return _FakeResponse(201, {"id": "batch_mistral_123"})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(dispatch.requests, "post", fake_post)

    inputs = [
        {
            "custom_id": "claude-opus-4-7::main__G2-v004__claude__NOTOOLS__neutral",
            "judge_prompt": "fixed judge prefix\n\nOriginal Prompt: row one",
        },
        {
            "custom_id": "grok-4.3::main__M1-v001__grok__NOTOOLS__neutral",
            "judge_prompt": "fixed judge prefix\n\nOriginal Prompt: row two",
        },
    ]

    result = dispatch.submit_mistral_batch(inputs)

    assert result["submit_status"] == "queued"
    assert result["provider"] == "mistral"
    assert result["transport"] == "mistral_native_batch"
    assert result["batch_id"] == "batch_mistral_123"
    assert result["input_file_id"] == "file_mistral_123"
    assert result["model"] == "mistral-large-2512"
    assert result["max_tokens"] == 4096
    assert result["temperature"] == 0.0
    assert result["top_p"] == 1.0
    assert "prompt_cache_key" not in result

    assert len(posts) == 2
    upload_post = posts[0]
    assert upload_post["url"] == "https://api.mistral.ai/v1/files"
    assert upload_post["data"] == {"purpose": "batch"}
    uploaded_file = upload_post["files"]["file"]  # type: ignore[index]
    uploaded_stream = uploaded_file[1]
    payload = uploaded_stream.getvalue().decode("utf-8")
    uploaded_rows = [json.loads(line) for line in payload.splitlines()]
    assert len(uploaded_rows) == 2

    for uploaded_row, source in zip(uploaded_rows, inputs, strict=True):
        body = uploaded_row["body"]
        assert uploaded_row["custom_id"] in result["short_to_full_cid"]
        assert body["model"] == "mistral-large-2512"
        assert body["messages"] == [{"role": "user", "content": source["judge_prompt"]}]
        assert body["max_tokens"] == 4096
        assert body["temperature"] == 0.0
        assert body["top_p"] == 1.0
        assert "top_k" not in body
        assert "prompt_cache_key" not in body

    create_post = posts[1]
    assert create_post["url"] == "https://api.mistral.ai/v1/batch/jobs"
    assert create_post["json"] == {
        "endpoint": "/v1/chat/completions",
        "input_files": ["file_mistral_123"],
        "model": "mistral-large-2512",
        "metadata": {
            "max_tokens": "4096",
            "temperature": "0.0",
            "top_p": "1.0",
        },
    }


def test_submit_mistral_batch_escapes_unicode_line_separators_in_jsonl(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    monkeypatch.setattr(dispatch, "archive_response_body", lambda *args, **kwargs: None)

    def fake_post(url, **kwargs):
        if url.endswith("/v1/files"):
            payload = kwargs["files"]["file"][1].getvalue()
            assert b"\xe2\x80\xa8" not in payload
            assert b"\\u2028" in payload
            for line in payload.decode("utf-8").splitlines():
                if line.strip():
                    json.loads(line)
            return _FakeResponse(201, {"id": "file_mistral_123"})
        if url.endswith("/v1/batch/jobs"):
            return _FakeResponse(201, {"id": "batch_mistral_123"})
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(dispatch.requests, "post", fake_post)

    result = dispatch.submit_mistral_batch(
        [
            {
                "custom_id": "row-with-line-separator",
                "judge_prompt": "fixed judge prefix\u2028Original Prompt: row one",
            }
        ]
    )

    assert result["submit_status"] == "queued"


def test_submit_mistral_batch_rejects_non_protocol_max_tokens(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-test-key")
    monkeypatch.setattr(dispatch.requests, "post", lambda *args, **kwargs: pytest.fail("must not submit"))

    result = dispatch.submit_mistral_batch(
        [{"custom_id": "row-1", "judge_prompt": "fixed judge prefix"}],
        max_tokens=1234,
    )

    assert result["submit_status"] == "error"
    assert "deviates from roster protocol" in result["error"]
