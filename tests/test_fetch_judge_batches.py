from __future__ import annotations

import json

from scripts import fetch_judge_batches as fetch


def test_openai_to_votes_maps_short_ids_and_preserves_cached_usage():
    rows = [
        {
            "custom_id": "short123",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {"content": "Label: R\nReason: refused"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2000,
                        "completion_tokens": 50,
                        "prompt_tokens_details": {"cached_tokens": 1500},
                    },
                },
            },
        }
    ]

    votes = fetch._openai_to_votes(
        rows,
        "gpt-5.5-judge",
        short_to_full={"short123": "full-custom-id"},
    )

    assert votes == [
        {
            "attempts": [
                {
                    "attempt_index": 1,
                    "content_chars": len("Label: R\nReason: refused"),
                    "finish_reason": "stop",
                    "http_status": 200,
                    "invalid_reason": None,
                    "judge_id": "gpt-5.5-judge",
                    "parsed_label": "Refuse",
                    "provider_route": "batch",
                    "raw_response": rows[0]["response"]["body"],
                    "status": "valid",
                    "timestamp": votes[0]["attempts"][0]["timestamp"],
                }
            ],
            "cached_input_tokens": 1500,
            "custom_id": "full-custom-id",
            "input_tokens": 2000,
            "judge": "gpt-5.5-judge",
            "label": "Refuse",
            "output_tokens": 50,
            "raw_text": "Label: R\nReason: refused",
            "reason": "refused",
            "source": "batch",
        }
    ]


def test_openai_to_votes_records_batch_errors():
    rows = [
        {
            "custom_id": "short123",
            "response": {"status_code": 400, "body": {"error": {"message": "bad"}}},
        }
    ]

    votes = fetch._openai_to_votes(rows, "gpt-5.5-judge")

    assert votes[0]["custom_id"] == "short123"
    assert votes[0]["label"] == "Unresolved"
    assert votes[0]["source"] == "batch"
    assert "batch_error" in votes[0]["reason"]
    assert votes[0]["attempts"][0]["invalid_reason"] == "transport_error"


def test_fetch_openai_results_parses_jsonl(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = json.dumps({"custom_id": "a"}) + "\n" + json.dumps({"custom_id": "b"}) + "\n"
        content = text.encode("utf-8")

        def json(self):
            return {}

    seen = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        return FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setattr(fetch.requests, "get", fake_get)
    monkeypatch.setattr(fetch, "archive_response_body", lambda *args, **kwargs: None)

    rows = fetch.fetch_openai_results({"output_file_id": "file_out_123"})

    assert seen["url"] == "https://api.openai.com/v1/files/file_out_123/content"
    assert rows == [{"custom_id": "a"}, {"custom_id": "b"}]


def test_mistral_to_votes_preserves_cached_usage():
    rows = [
        {
            "custom_id": "short456",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {"content": "Label: C\nReason: complied"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4096,
                        "completion_tokens": 80,
                        "prompt_tokens_details": {"cached_tokens": 3840},
                    },
                },
            },
        }
    ]

    votes = fetch._mistral_to_votes(
        rows,
        "mistral-large-2512",
        short_to_full={"short456": "full-custom-id"},
    )

    assert votes[0]["custom_id"] == "full-custom-id"
    assert votes[0]["judge"] == "mistral-large-2512"
    assert votes[0]["label"] == "Comply"
    assert votes[0]["input_tokens"] == 4096
    assert votes[0]["output_tokens"] == 80
    assert votes[0]["cached_input_tokens"] == 3840


def test_mistral_to_votes_records_native_route_provenance_and_prompt_hash():
    rows = [
        {
            "custom_id": "short456",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {"content": "Label: R\nReason: refused"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            },
        }
    ]

    votes = fetch._mistral_to_votes(
        rows,
        "mistral-large-2512",
        short_to_full={"short456": "full-custom-id"},
        prompt_hashes={"full-custom-id": "a" * 64},
        model_id="mistral-large-2512",
    )

    assert votes[0]["source"] == "batch"
    assert votes[0]["provider"] == "mistral"
    assert votes[0]["model"] == "mistral-large-2512"
    assert votes[0]["judge_prompt_sha256"] == "a" * 64
