from __future__ import annotations

import json
from pathlib import Path

import pytest


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.gets: list[dict] = []

    def get(self, url, **kwargs):
        self.gets.append({"url": url, **kwargs})
        return self.response


def test_api_runtime_stamp_normalizes_identity_and_safe_headers():
    from scripts.runtime_provenance import build_api_runtime_stamp

    stamp = build_api_runtime_stamp(
        requested_model="openai/gpt-5.5",
        route="openrouter",
        provider="openai",
        response={
            "id": "gen_123",
            "model": "openai/gpt-5.5",
            "created": 1770000000,
            "system_fingerprint": "fp_abc",
            "choices": [{"finish_reason": "tool_calls"}],
        },
        response_headers={
            "x-request-id": "req_123",
            "authorization": "Bearer secret",
            "set-cookie": "private",
            "cf-ray": "ray_123",
        },
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        role="subject",
        model_id="gpt-5.5",
    )

    assert stamp["requested_model"] == "openai/gpt-5.5"
    assert stamp["returned_model"] == "openai/gpt-5.5"
    assert stamp["response_id"] == "gen_123"
    assert stamp["response_created"] == 1770000000
    assert stamp["system_fingerprint"] == "fp_abc"
    assert stamp["finish_reason"] == "tool_calls"
    assert stamp["response_request_id"] == "req_123"
    assert stamp["response_headers"] == {"cf-ray": "ray_123", "x-request-id": "req_123"}
    assert "authorization" not in stamp["response_headers"]
    assert "set-cookie" not in stamp["response_headers"]


def test_openrouter_model_metadata_snapshot_writes_matching_model(tmp_path: Path):
    from scripts.runtime_provenance import write_openrouter_model_metadata_snapshot

    session = _FakeSession(
        _FakeResponse(
            200,
            {
                "data": [
                    {"id": "other/model", "name": "Other"},
                    {
                        "id": "anthropic/claude-opus-4.7",
                        "name": "Claude Opus 4.7",
                        "pricing": {"prompt": "0.001"},
                    },
                ]
            },
            headers={"x-request-id": "models_req_1"},
        )
    )

    snapshot = write_openrouter_model_metadata_snapshot(
        session=session,
        api_key="sk-test",
        output_dir=tmp_path,
        model_slug="anthropic/claude-opus-4.7",
        pinned_provider="anthropic",
    )

    path = Path(snapshot["path"])
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["model_slug"] == "anthropic/claude-opus-4.7"
    assert payload["pinned_provider"] == "anthropic"
    assert payload["matched_model"]["id"] == "anthropic/claude-opus-4.7"
    assert payload["response_headers"] == {"x-request-id": "models_req_1"}
    assert len(snapshot["sha256"]) == 64
    assert session.gets[0]["url"] == "https://openrouter.ai/api/v1/models"


def test_checkpoint_manifest_for_row_validates_identity_and_hashes(tmp_path: Path):
    from scripts.runtime_provenance import load_checkpoint_manifest_for_row

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "gemma-4-31b-it__primary__bf16.json"
    manifest_path.write_text(
        json.dumps(
            {
                "subject_id": "gemma-4-31b-it",
                "serving_precision_role": "primary",
                "serving_precision_value": "bf16",
                "served_model_path": "google/gemma-4-31B-it",
                "hf_repo": "google/gemma-4-31B-it",
                "hf_commit_sha": "145dc2508c480a64b47242f160d286cff94a2343",
                "files": [
                    {
                        "path": "model-00001-of-00002.safetensors",
                        "sha256": "a" * 64,
                        "size_bytes": 10,
                    },
                    {"path": "tokenizer.json", "sha256": "b" * 64, "size_bytes": 5},
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    row = {
        "subject_id": "gemma-4-31b-it",
        "serving_precision_role": "primary",
        "serving_precision_value": "bf16",
        "served_model_path": "google/gemma-4-31B-it",
    }

    stamped = load_checkpoint_manifest_for_row(row, manifest_dir)

    assert stamped["checkpoint_manifest_path"] == str(manifest_path)
    assert stamped["checkpoint_manifest_sha256"]
    assert stamped["checkpoint_manifest_summary"] == {
        "subject_id": "gemma-4-31b-it",
        "serving_precision_role": "primary",
        "serving_precision_value": "bf16",
        "served_model_path": "google/gemma-4-31B-it",
        "hf_commit_sha": "145dc2508c480a64b47242f160d286cff94a2343",
        "file_count": 2,
        "total_size_bytes": 15,
    }


def test_checkpoint_manifest_for_row_rejects_model_mismatch(tmp_path: Path):
    from scripts.runtime_provenance import CheckpointManifestError, load_checkpoint_manifest_for_row

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "gemma-4-31b-it__primary__bf16.json").write_text(
        json.dumps(
            {
                "subject_id": "gemma-4-31b-it",
                "serving_precision_role": "primary",
                "serving_precision_value": "bf16",
                "served_model_path": "wrong/model",
                "hf_commit_sha": "145dc2508c480a64b47242f160d286cff94a2343",
                "files": [{"path": "tokenizer.json", "sha256": "b" * 64}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointManifestError, match="served_model_path"):
        load_checkpoint_manifest_for_row(
            {
                "subject_id": "gemma-4-31b-it",
                "serving_precision_role": "primary",
                "serving_precision_value": "bf16",
                "served_model_path": "google/gemma-4-31B-it",
            },
            manifest_dir,
        )
