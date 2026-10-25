from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_disabled_format_control as control
from scripts import run_kimi_batch_format_pilot as kimi_pilot


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL = (
    REPO_ROOT
    / "experiments/disabled_format_control/protocol_v3_grok_gpt_kimi.json"
)


def _native_route() -> dict:
    return {
        "route": "native",
        "model_slug": "kimi-k2.6",
        "provider_only": ["moonshot"],
        "quantizations": [],
        "serving_precision": "provider_managed",
        "decoding": {"max_tokens": 4096},
        "endpoint": "https://api.moonshot.ai",
        "request_path": "/v1/chat/completions",
        "batch_url_path": "/v1/batches",
        "batch_input_files_path": "/v1/files",
        "request_shape": "openai_compatible_batch",
    }


def test_frozen_successor_has_exact_three_subject_inventory() -> None:
    protocol, _protocol_sha256 = control.load_protocol(FROZEN_PROTOCOL)

    assert protocol["population"]["subjects"] == [
        "kimi-k2.6",
        "grok-4.3",
        "gpt-5.5",
    ]
    assert protocol["population"]["expected_analysis_sets"] == 3_456
    assert protocol["population"]["expected_cells_per_arm"] == 3_456
    assert protocol["population"]["expected_rows_total"] == 10_368
    assert protocol["population"]["expected_canary_rows_total"] == 9
    assert protocol["execution_order"] == [
        "kimi-k2.6",
        "grok-4.3",
        "gpt-5.5",
    ]
    routes = protocol["route_policy"]["routes"]
    assert routes["kimi-k2.6"]["route"] == "native"
    assert routes["grok-4.3"]["route"] == "openrouter"
    assert routes["gpt-5.5"]["route"] == "openrouter"


def test_native_kimi_route_applies_the_exact_frozen_batch_contract() -> None:
    protocol = {"route_policy": {"routes": {"kimi-k2.6": _native_route()}}}
    row = {
        "subject_id": "kimi-k2.6",
        "body": {
            "model": "stale-model",
            "messages": [],
            "tools": [],
            "tool_choice": "none",
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 7,
        },
    }

    control._apply_route_metadata(row, protocol)

    assert row["route"] == "native"
    assert row["pinned_provider"] == "moonshot"
    assert row["url"] == "/v1/chat/completions"
    assert row["batch_submit_url"] == "/v1/batches"
    assert row["request_shape"] == "openai_compatible_batch"
    assert row["body"]["model"] == "kimi-k2.6"
    assert row["body"]["max_tokens"] == 4096
    assert "temperature" not in row["body"]
    assert "top_p" not in row["body"]
    assert row["native_body"] == row["body"]


def test_execution_authorization_binds_protocol_lock_osf_and_subject(
    tmp_path: Path,
) -> None:
    protocol = {
        "protocol_id": "protocol-v3",
        "population": {"subjects": ["kimi-k2.6", "grok-4.3", "gpt-5.5"]},
    }
    protocol_sha256 = "a" * 64
    lock_path = tmp_path / "locked.json"
    lock_path.write_text('{"locked":true}\n', encoding="utf-8")
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    authorization_path = tmp_path / "authorization.json"
    authorization = {
        "schema_version": "disabled_format_control.execution_authorization.v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "locked_inputs_manifest": str(lock_path),
        "locked_inputs_manifest_sha256": lock_sha256,
        "osf_registration_url": "https://osf.io/abc12/",
        "osf_filed_at_utc": "2026-09-03T18:00:00Z",
        "osf_filed_protocol_sha256": protocol_sha256,
        "authorized_subjects": protocol["population"]["subjects"],
        "authorized_by": "project_owner",
        "spending_authorization": {
            "openrouter_cap_usd": 140.0,
            "openrouter_spent_before_official_usd": 1.0,
            "openrouter_profile_subject_caps_usd": {
                "canary": {"grok-4.3": 0.25, "gpt-5.5": 0.5},
                "full": {"grok-4.3": 10.0, "gpt-5.5": 100.0},
            },
            "kimi_native_batch": True,
            "kimi_native_batch_cap_usd": 40.0,
            "kimi_native_batch_spent_before_official_usd": 0.04,
            "kimi_native_batch_profile_caps_usd": {
                "canary": 0.25,
                "full": 39.71,
            },
        },
        "submission_permitted": True,
    }
    authorization_path.write_text(
        json.dumps(authorization), encoding="utf-8"
    )

    loaded = control.validate_execution_authorization(
        authorization_path,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        locked_inputs_manifest_path=lock_path,
        subject_id="grok-4.3",
    )
    assert loaded == authorization

    authorization["osf_filed_protocol_sha256"] = "b" * 64
    authorization_path.write_text(
        json.dumps(authorization), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="OSF-filed protocol hash"):
        control.validate_execution_authorization(
            authorization_path,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            locked_inputs_manifest_path=lock_path,
            subject_id="grok-4.3",
        )


def test_execution_authorization_rejects_openrouter_overallocation(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "locked.json"
    lock_path.write_text('{}\n', encoding="utf-8")
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": "disabled_format_control.execution_authorization.v1",
                "protocol_id": "protocol-v3",
                "protocol_sha256": "a" * 64,
                "locked_inputs_manifest": str(lock_path),
                "locked_inputs_manifest_sha256": lock_sha256,
                "osf_registration_url": "https://osf.io/abc12/",
                "osf_filed_at_utc": "2026-09-03T18:00:00Z",
                "osf_filed_protocol_sha256": "a" * 64,
                "authorized_subjects": ["grok-4.3"],
                "authorized_by": "project_owner",
                "spending_authorization": {
                    "openrouter_cap_usd": 140.0,
                    "openrouter_spent_before_official_usd": 1.0,
                    "openrouter_profile_subject_caps_usd": {
                        "canary": {"grok-4.3": 1.0},
                        "full": {"grok-4.3": 139.0},
                    },
                    "kimi_native_batch": False,
                },
                "submission_permitted": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allocations exceed the remaining cap"):
        control.validate_execution_authorization(
            authorization_path,
            protocol={
                "protocol_id": "protocol-v3",
                "population": {"subjects": ["grok-4.3"]},
                "route_policy": {
                    "routes": {"grok-4.3": {"route": "openrouter"}}
                },
            },
            protocol_sha256="a" * 64,
            locked_inputs_manifest_path=lock_path,
            subject_id="grok-4.3",
        )


def test_execution_authorization_template_cannot_enable_submission(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "locked.json"
    lock_path.write_text("{}\n", encoding="utf-8")
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": "disabled_format_control.execution_authorization.v1",
                "submission_permitted": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="submission_permitted=true"):
        control.validate_execution_authorization(
            authorization_path,
            protocol={
                "protocol_id": "protocol-v3",
                "population": {"subjects": ["kimi-k2.6"]},
            },
            protocol_sha256="a" * 64,
            locked_inputs_manifest_path=lock_path,
            subject_id="kimi-k2.6",
        )


def test_successor_submission_requires_post_osf_receipt_before_outputs() -> None:
    protocol = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
    args = Namespace(
        protocol=FROZEN_PROTOCOL,
        output_dir=REPO_ROOT / protocol["artifact_paths"]["canary_batches_dir"],
        locked_inputs_manifest=(
            REPO_ROOT / protocol["artifact_paths"]["locked_inputs_manifest"]
        ),
        execution_authorization=None,
        profile="canary",
        subject=["kimi-k2.6"],
        acknowledge_postregistration_exploratory=True,
        confirm_spend=protocol["protocol_id"],
        max_requests=3,
        max_cost_usd=1.0,
    )

    with pytest.raises(ValueError, match="requires --execution-authorization"):
        control.submit_generated(args)


def test_kimi_batch_pilot_builds_excluded_exact_format_matrix() -> None:
    rows = kimi_pilot.build_pilot_rows(FROZEN_PROTOCOL)

    assert len(rows) == 24
    assert len({row["custom_id"] for row in rows}) == 24
    assert {row["mode"] for row in rows} == {
        "NOTOOLS",
        "DISABLED",
        "DISABLED_FORMAT_JSON",
        "TOOL",
        "TOOL_IRRELEVANT",
        "CANARY_DISABLED",
    }
    for row in rows:
        assert row["subject_id"] == "kimi-k2.6"
        assert row["route"] == "native"
        assert row["analysis_inclusion"] is False
        assert row["diagnostic_only"] is True
        assert row["url"] == "/v1/chat/completions"
        assert row["body"]["model"] == "kimi-k2.6"
        assert row["body"]["max_tokens"] == 4096
        assert "temperature" not in row["body"]
        assert "top_p" not in row["body"]

    format_rows = [row for row in rows if row["mode"] == "DISABLED_FORMAT_JSON"]
    baseline_rows = [row for row in rows if row["mode"] == "DISABLED"]
    assert len(format_rows) == len(baseline_rows) == 4
    instruction = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))["design"][
        "format_instruction_exact"
    ]
    for row in format_rows:
        system = row["body"]["messages"][0]["content"]
        assert system.endswith("\n\n" + instruction)
        assert row["body"]["tool_choice"] == "none"
        assert len(row["body"]["tools"]) == 6


def test_kimi_output_ready_override_is_diagnostic_and_time_gated() -> None:
    batch = {
        "status": "finalizing",
        "request_counts": {"completed": 24, "failed": 0, "total": 24},
        "output_file_id": "file-output",
        "error_file_id": None,
        "errors": None,
        "finalizing_at": 1_000,
    }

    with pytest.raises(ValueError, match="at least 30 minutes"):
        kimi_pilot.validate_output_ready_finalizing(batch, now_epoch=2_799)

    accepted = kimi_pilot.validate_output_ready_finalizing(
        batch, now_epoch=2_800
    )
    assert accepted["diagnostic_only"] is True
    assert accepted["official_canary_policy_unchanged"] is True
    assert accepted["request_counts"] == {
        "completed": 24,
        "failed": 0,
        "total": 24,
    }


def test_openrouter_runtime_budget_reserves_the_next_worst_case_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_synthetic_prompts as runner

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        runner,
        "write_openrouter_model_metadata_snapshot",
        lambda **_kwargs: {"diagnostic": "test"},
    )

    class FakeSubmitter:
        output_dir = tmp_path
        session = object()

        @staticmethod
        def _post_openrouter_with_retries(*_args: object, **_kwargs: object):
            return (
                {
                    "model": "x-ai/grok-4.3",
                    "provider": "xAI",
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                        "cost": 0.001,
                    },
                },
                {},
                None,
            )

    route = {
        "route": "openrouter",
        "model_slug": "x-ai/grok-4.3",
        "provider_only": ["xai/zdr"],
        "quantizations": [],
        "serving_precision": "provider_managed",
        "decoding": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
    }
    protocol = {"route_policy": {"routes": {"grok-4.3": route}}}
    rows = [
        {
            "custom_id": f"row-{index}",
            "subject_id": "grok-4.3",
            "mode": "DISABLED",
            "body": {
                "model": "x-ai/grok-4.3",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 4096,
            },
        }
        for index in range(2)
    ]
    entry = {
        "openrouter": {
            "slug": "x-ai/grok-4.3",
            "pinned_provider": "xai/zdr",
            "quantizations": [],
            "decoding": route["decoding"],
        }
    }

    result = control._submit_v3_openrouter_rows(
        FakeSubmitter(),
        "grok-4.3",
        entry,
        rows,
        protocol=protocol,
        runtime_cost_cap_usd=0.0105,
    )

    assert result["status"] == "paused_budget_cap"
    assert result["responses"] == 1
    assert result["reported_cost_usd"] == pytest.approx(0.001)
    response_path = tmp_path / "responses_grok-4.3_openrouter.jsonl"
    written = [json.loads(line) for line in response_path.read_text().splitlines()]
    assert len(written) == 1
    assert written[0]["execution_budget"]["cumulative_reported_cost_usd"] == 0.001
