from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_disabled_format_control as control
from scripts.reconcile_disabled_format_control_openrouter_n3 import reconcile_subject


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    REPO_ROOT / "experiments/disabled_format_control/protocol_v3_openrouter_n3_eligible.json"
)
BATCH_PATH = (
    REPO_ROOT
    / "outputs/postreg_disabled_format_control_v3_openrouter_n3_eligible/batches/"
    "input_gpt-5.5.jsonl"
)
METADATA_PATH = (
    REPO_ROOT
    / "outputs/postreg_disabled_format_control_v3_openrouter_n3_eligible/responses/"
    "openrouter_model_metadata/openai_gpt-5.5__openai_flex.json"
)


def _fixture(tmp_path: Path) -> tuple[dict, dict, dict]:
    protocol, _ = control.load_protocol(PROTOCOL_PATH)
    expected = json.loads(BATCH_PATH.read_text(encoding="utf-8").splitlines()[0])
    descriptor = protocol["route_policy"]["routes"]["gpt-5.5"]
    route = {
        "slug": descriptor["model_slug"],
        "pinned_provider": descriptor["provider_only"][0],
        "quantizations": descriptor["quantizations"],
        "decoding": descriptor["decoding"],
    }
    body = control.build_control_openrouter_body(expected, route)
    response = {
        "id": "gen-reconciliation-test",
        "created": 1788653033,
        "model": "openai/gpt-5.5",
        "provider": "OpenAI",
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
            "cost": 0.001,
        },
    }
    raw_bytes = json.dumps(response, sort_keys=True).encode("utf-8")
    raw_path = tmp_path / "response.body"
    raw_path.write_bytes(raw_bytes)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    snapshot = {
        "path": str(METADATA_PATH),
        "sha256": hashlib.sha256(METADATA_PATH.read_bytes()).hexdigest(),
        "matched_model_id": metadata["matched_model"]["id"],
    }
    observed = control.build_control_openrouter_run_record(
        row=expected,
        body=body,
        response=response,
        pinned_provider="openai/flex",
        latency_sec=1.25,
        response_headers={},
        model_metadata_snapshot=snapshot,
        raw_http_capture={
            "raw_http_archive_schema_version": "raw_http_archive.v1",
            "raw_http_body_path": str(raw_path),
            "raw_http_body_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_http_body_bytes": len(raw_bytes),
            "raw_http_manifest_path": "manifest.jsonl",
            "raw_http_captured_at_utc": "2026-09-06T00:03:54Z",
        },
    )
    observed["execution_budget"] = {
        "response_reported_cost_usd": 0.001,
        "cumulative_reported_cost_usd": 0.001,
        "runtime_cost_cap_usd": 33.93139309,
    }
    return expected, observed, descriptor


def test_reconcile_subject_verifies_identity_raw_body_and_cost(tmp_path: Path) -> None:
    expected, observed, descriptor = _fixture(tmp_path)
    result = reconcile_subject(
        expected_rows=[expected],
        observed_rows=[observed],
        descriptor=descriptor,
        subject_id="gpt-5.5",
        runtime_cost_cap_usd=33.93139309,
        recovered_from_raw_archive=False,
    )
    assert result["status"] == "complete_and_reconciled"
    assert result["expected_rows"] == result["observed_rows"] == 1
    assert result["reported_cost_usd"] == 0.001
    assert result["returned_model_counts"] == {"openai/gpt-5.5": 1}
    assert result["observed_provider_counts"] == {"OpenAI": 1}


def test_reconcile_subject_rejects_duplicate_response_id(tmp_path: Path) -> None:
    expected, observed, descriptor = _fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate or missing observed custom ID"):
        reconcile_subject(
            expected_rows=[expected],
            observed_rows=[observed, dict(observed)],
            descriptor=descriptor,
            subject_id="gpt-5.5",
            runtime_cost_cap_usd=33.93139309,
            recovered_from_raw_archive=False,
        )


def test_reconcile_subject_rejects_returned_model_drift(tmp_path: Path) -> None:
    expected, observed, descriptor = _fixture(tmp_path)
    observed["raw_response"]["model"] = "openai/other-model"
    with pytest.raises(ValueError, match="returned model mismatch"):
        reconcile_subject(
            expected_rows=[expected],
            observed_rows=[observed],
            descriptor=descriptor,
            subject_id="gpt-5.5",
            runtime_cost_cap_usd=33.93139309,
            recovered_from_raw_archive=False,
        )
