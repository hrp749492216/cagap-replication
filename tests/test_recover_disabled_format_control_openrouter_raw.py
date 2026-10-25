from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_disabled_format_control as control
from scripts.recover_disabled_format_control_openrouter_raw import (
    LATENCY_UNAVAILABLE_SENTINEL_SECONDS,
    _raw_archive_request_sha256,
    build_recovered_records,
    load_manifest_entries,
    write_records_atomic,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "experiments/disabled_format_control/protocol_v3_openrouter_n3_eligible.json"
)
BATCH_PATH = (
    REPO_ROOT
    / "outputs/postreg_disabled_format_control_v3_openrouter_n3_eligible/batches/"
    "input_grok-4.3.jsonl"
)
METADATA_PATH = (
    REPO_ROOT
    / "outputs/postreg_disabled_format_control_v3_openrouter_n3_eligible/responses/"
    "openrouter_model_metadata/x-ai_grok-4.3__xai_zdr.json"
)


def _fixture(tmp_path: Path) -> tuple[dict, dict, dict]:
    protocol, _ = control.load_protocol(PROTOCOL_PATH)
    row = json.loads(BATCH_PATH.read_text(encoding="utf-8").splitlines()[0])
    descriptor = protocol["route_policy"]["routes"]["grok-4.3"]
    route = {
        "slug": descriptor["model_slug"],
        "pinned_provider": descriptor["provider_only"][0],
        "quantizations": descriptor["quantizations"],
        "decoding": descriptor["decoding"],
    }
    request_body = control.build_control_openrouter_body(row, route)
    response = {
        "id": "gen-recovery-test",
        "created": 1788653033,
        "model": "x-ai/grok-4.3",
        "provider": "xAI",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"role": "assistant", "content": "No action taken."},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "cost": 0.00125,
        },
    }
    response_bytes = json.dumps(response, sort_keys=True).encode("utf-8")
    body_path = tmp_path / "response.body"
    body_path.write_bytes(response_bytes)
    entry = {
        "schema_version": "raw_http_archive.v1",
        "captured_at_utc": "2026-09-06T00:03:54.533345Z",
        "source": "requests.Response.content",
        "role": "subject",
        "model_id": "grok-4.3",
        "route": "openrouter",
        "provider": "xai/zdr",
        "custom_id": row["custom_id"],
        "attempt_index": 1,
        "http_status": 200,
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "body_path": str(body_path),
        "body_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "body_bytes": len(response_bytes),
        "body_input_type": "bytes",
        "request_body_sha256": _raw_archive_request_sha256(request_body),
        "_recovery_manifest_sources": [str(tmp_path / "manifest.jsonl")],
    }
    return protocol, row, entry


def test_build_recovered_record_replays_hash_verified_response_without_dispatch(
    tmp_path: Path,
) -> None:
    protocol, row, entry = _fixture(tmp_path)
    records = build_recovered_records(
        rows=[row],
        protocol=protocol,
        subject_id="grok-4.3",
        manifest_entries=[entry],
        metadata_snapshot_path=METADATA_PATH,
        runtime_cost_cap_usd=16.0,
        incident_id="test-incident",
    )

    assert len(records) == 1
    record = records[0]
    assert record["custom_id"] == row["custom_id"]
    assert record["observed_provider"] == "xAI"
    assert record["observed_provider_matches_pin"] is True
    assert record["latency_sec"] == LATENCY_UNAVAILABLE_SENTINEL_SECONDS
    assert record["timestamp"] == "2026-09-06T00:03:54Z"
    assert record["execution_budget"]["cumulative_reported_cost_usd"] == 0.00125
    assert record["artifact_recovery"]["provider_request_reissued"] is False
    assert record["raw_http_capture"]["raw_http_body_sha256"] == entry["body_sha256"]


def test_recovery_rejects_archived_request_hash_drift(tmp_path: Path) -> None:
    protocol, row, entry = _fixture(tmp_path)
    entry["request_body_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="request body hash mismatch"):
        build_recovered_records(
            rows=[row],
            protocol=protocol,
            subject_id="grok-4.3",
            manifest_entries=[entry],
            metadata_snapshot_path=METADATA_PATH,
            runtime_cost_cap_usd=16.0,
            incident_id="test-incident",
        )


def test_manifest_loader_deduplicates_preserved_hardlink_rows(tmp_path: Path) -> None:
    _, _, entry = _fixture(tmp_path)
    entry.pop("_recovery_manifest_sources")
    first = tmp_path / "manifest-one.jsonl"
    second = tmp_path / "manifest-two.jsonl"
    payload = json.dumps(entry, sort_keys=True) + "\n"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")

    loaded = load_manifest_entries([first, second])

    assert len(loaded) == 1
    assert loaded[0]["_recovery_manifest_sources"] == sorted(
        [str(first.resolve()), str(second.resolve())]
    )


def test_atomic_writer_refuses_to_replace_nonempty_artifact(tmp_path: Path) -> None:
    output = tmp_path / "responses.jsonl"
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_records_atomic(output, [{"row": 1}], replace_empty_artifact=True)
