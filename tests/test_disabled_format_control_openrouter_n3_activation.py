from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts import run_disabled_format_control as control
from scripts import run_disabled_format_control_exploratory as exploratory
from scripts import run_disabled_format_control_openrouter_n3 as openrouter_n3
from scripts import run_disabled_format_control_openrouter_n3_full as openrouter_n3_full
from scripts.run_synthetic_prompts import select_route


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "experiments/disabled_format_control/protocol_v3_openrouter_n3_eligible.json"
)
LOCKED_INPUTS = (
    REPO_ROOT
    / "experiments/disabled_format_control/locked_inputs_manifest_v3_openrouter_n3_eligible.json"
)
GOVERNANCE_AMENDMENT = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_governance_amendment_v3_openrouter_n3_eligible_2026-09-05.json"
)
EXECUTION_AUTHORIZATION = (
    REPO_ROOT
    / "experiments/disabled_format_control/execution_authorization_v3_openrouter_n3_eligible.json"
)
CANARY_REVIEW = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "canary_review_v3_openrouter_n3_eligible_2026-09-05.json"
)
FULL_EXECUTION_AUTHORIZATION = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_authorization_v3_openrouter_n3_eligible_full.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_openrouter_n3_protocol_freezes_the_four_supported_route_candidates() -> None:
    protocol, _protocol_sha256 = control.load_protocol(PROTOCOL)

    expected_subjects = [
        "llama-3.3-70b-instruct",
        "grok-4.3",
        "claude-opus-4-7",
        "gpt-5.5",
    ]
    assert protocol["status"] == "frozen_before_first_dispatch"
    assert protocol["population"]["subjects"] == expected_subjects
    assert protocol["execution_order"] == expected_subjects
    assert protocol["population"]["trial_ids"] == [0, 1, 2]
    assert protocol["population"]["expected_analysis_sets"] == 4608
    assert protocol["population"]["expected_cells_per_arm"] == 4608
    assert protocol["population"]["expected_rows_total"] == 13824
    assert protocol["population"]["expected_canary_rows_total"] == 12
    assert set(protocol["source_batch_sha256"]) == {
        f"input_{subject}.jsonl" for subject in expected_subjects
    }
    assert set(protocol["route_policy"]["routes"]) == set(expected_subjects)
    assert {
        descriptor["route"]
        for descriptor in protocol["route_policy"]["routes"].values()
    } == {"openrouter"}
    assert protocol["route_policy"]["allow_fallbacks"] is False
    assert protocol["route_policy"]["data_collection"] == "deny"
    assert (
        protocol["design"]["partition_namespace"]
        == "postreg_disabled_format_control_v3_openrouter_n3_eligible"
    )


def test_openrouter_n3_protocol_keeps_the_original_ten_model_draft_historical() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["supersedes_draft_protocol_id"] == (
        "CAGAP-POSTREG-DISABLED-FORMAT-2026-09-01-v3-DRAFT"
    )
    assert (
        "CAGAP-POSTREG-DISABLED-FORMAT-2026-09-03-v3-GROK-GPT-KIMI"
        in protocol["preserved_protocol_ids"]
    )
    assert "exploratory" in protocol["registration_scope"].lower()
    assert "OpenRouter" in protocol["rationale"]


def test_openrouter_n3_locked_inputs_match_every_generated_request_file() -> None:
    lock = json.loads(LOCKED_INPUTS.read_text(encoding="utf-8"))

    assert lock["protocol_sha256"] == _sha256(PROTOCOL)
    assert lock["full"]["row_count"] == 13824
    assert lock["canary"]["row_count"] == 12
    assert lock["determinism"]["independent_generations"] == 2
    assert lock["determinism"]["request_files_byte_identical"] is True
    for profile in ("full", "canary"):
        for name, expected in lock[profile]["output_files"].items():
            path = REPO_ROOT / lock[profile]["batches_dir"] / name
            assert sum(1 for _line in path.open(encoding="utf-8")) == expected["rows"]
            assert _sha256(path) == expected["sha256"]


def test_openrouter_n3_owner_receipt_is_canary_only_and_within_cap() -> None:
    protocol, protocol_sha256 = control.load_protocol(PROTOCOL)
    authorization = exploratory.validate_execution_authorization(
        EXECUTION_AUTHORIZATION,
        governance_amendment_path=GOVERNANCE_AMENDMENT,
        protocol=protocol,
        protocol_path=PROTOCOL,
        protocol_sha256=protocol_sha256,
        locked_inputs_manifest_path=LOCKED_INPUTS,
        subject_id="llama-3.3-70b-instruct",
    )

    assert authorization["authorized_profiles"] == ["canary"]
    spending = authorization["spending_authorization"]
    allocated = sum(
        float(value)
        for profile in spending["openrouter_profile_subject_caps_usd"].values()
        for value in profile.values()
    )
    assert allocated == spending["openrouter_cap_usd"] - spending[
        "openrouter_spent_before_official_usd"
    ]
    assert spending["native_provider_spend_authorized"] is False


def test_protocol_can_route_a_historically_local_subject_through_exact_openrouter_pin() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    descriptor = protocol["route_policy"]["routes"]["llama-3.3-70b-instruct"]

    entry = openrouter_n3.protocol_route_entry(
        "llama-3.3-70b-instruct",
        route="openrouter",
        descriptor=descriptor,
    )
    route, detail = select_route(entry)

    assert route == "openrouter"
    assert detail == {
        "slug": "meta-llama/llama-3.3-70b-instruct",
        "pinned_provider": "coreweave/fp16",
        "quantizations": ["fp16"],
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 4096,
        },
    }
    assert "local_config" not in entry


def test_openrouter_n3_canary_review_stops_llama_and_passes_three_subjects() -> None:
    review = json.loads(CANARY_REVIEW.read_text(encoding="utf-8"))

    assert review["status"] == "complete_before_full_authorization"
    assert review["passing_subjects"] == [
        "grok-4.3",
        "claude-opus-4-7",
        "gpt-5.5",
    ]
    assert review["stopped_subjects"] == ["llama-3.3-70b-instruct"]
    assert review["subjects"]["llama-3.3-70b-instruct"]["verdict"] == (
        "stop_before_full"
    )
    assert review["full_gate"]["total_full_rows_authorized"] == 10368
    assert review["spend_reconciliation"]["reconciliation_delta_usd"] == 0.0


def test_openrouter_n3_full_authorization_accepts_only_canary_passed_subjects() -> None:
    protocol, protocol_sha256 = control.load_protocol(PROTOCOL)

    authorization = openrouter_n3_full.validate_full_execution_authorization(
        FULL_EXECUTION_AUTHORIZATION,
        protocol=protocol,
        protocol_path=PROTOCOL,
        protocol_sha256=protocol_sha256,
        locked_inputs_manifest_path=LOCKED_INPUTS,
        canary_authorization_path=EXECUTION_AUTHORIZATION,
        subject_id="grok-4.3",
    )
    assert authorization["authorized_profiles"] == ["full"]
    assert "llama-3.3-70b-instruct" not in authorization["authorized_subjects"]

    with pytest.raises(ValueError, match="not authorized for full dispatch"):
        openrouter_n3_full.validate_full_execution_authorization(
            FULL_EXECUTION_AUTHORIZATION,
            protocol=protocol,
            protocol_path=PROTOCOL,
            protocol_sha256=protocol_sha256,
            locked_inputs_manifest_path=LOCKED_INPUTS,
            canary_authorization_path=EXECUTION_AUTHORIZATION,
            subject_id="llama-3.3-70b-instruct",
        )
