from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_disabled_format_control as control
from scripts import run_disabled_format_control_exploratory as exploratory


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PROTOCOL = (
    REPO_ROOT
    / "experiments/disabled_format_control/protocol_v3_grok_gpt_kimi.json"
)
LOCKED_INPUTS = (
    REPO_ROOT
    / "experiments/disabled_format_control/locked_inputs_manifest_v3_grok_gpt_kimi.json"
)
GOVERNANCE_AMENDMENT = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_governance_amendment_v3_grok_gpt_kimi_2026-09-05.json"
)
EXECUTION_AUTHORIZATION = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_authorization_v3_grok_gpt_kimi.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_artifacts(tmp_path: Path) -> tuple[dict, Path, Path, Path, Path]:
    protocol = {
        "protocol_id": "exploratory-protocol-v3",
        "status": "frozen_before_first_dispatch",
        "registration_scope": "post-registration exploratory mechanism control",
        "population": {
            "subjects": ["kimi-k2.6", "grok-4.3", "gpt-5.5"],
        },
        "route_policy": {
            "routes": {
                "kimi-k2.6": {"route": "native"},
                "grok-4.3": {"route": "openrouter"},
                "gpt-5.5": {"route": "openrouter"},
            }
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol) + "\n", encoding="utf-8")
    protocol_sha256 = _sha256(protocol_path)
    lock_path = tmp_path / "locked.json"
    lock_path.write_text(
        json.dumps(
            {
                "locked": True,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha256,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    approval_path = tmp_path / "approval.md"
    approval_path.write_text("# Prospective owner approval\n", encoding="utf-8")
    amendment_path = tmp_path / "amendment.json"
    amendment = {
        "schema_version": (
            "disabled_format_control.execution_governance_amendment.v1"
        ),
        "status": "frozen_before_first_official_dispatch",
        "approved_at_utc": "2026-09-05T19:22:31Z",
        "approved_by": "project_owner",
        "registration_status": "post_registration_exploratory",
        "osf_filing_required": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "locked_inputs_manifest": str(lock_path),
        "locked_inputs_manifest_sha256": _sha256(lock_path),
        "approval_document": str(approval_path),
        "approval_document_sha256": _sha256(approval_path),
        "activation_launcher": str(Path(exploratory.__file__).resolve()),
        "activation_launcher_sha256": _sha256(
            Path(exploratory.__file__).resolve()
        ),
        "delegated_frozen_runner": str(Path(control.__file__).resolve()),
        "delegated_frozen_runner_sha256": _sha256(Path(control.__file__).resolve()),
        "official_response_artifacts_at_approval": 0,
        "scientific_contract_unchanged": True,
        "superseded_activation_requirement": "osf_filing_only",
    }
    amendment_path.write_text(json.dumps(amendment) + "\n", encoding="utf-8")
    authorization_path = tmp_path / "authorization.json"
    authorization = {
        "schema_version": "disabled_format_control.execution_authorization.v2",
        "receipt_created_at_utc": "2026-09-05T19:34:19Z",
        "submission_permitted": True,
        "authorization_type": "project_owner_exploratory",
        "registration_status": "post_registration_exploratory",
        "osf_filing_required": False,
        "owner_authorized_at_utc": "2026-09-05T19:22:31Z",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "locked_inputs_manifest": str(lock_path),
        "locked_inputs_manifest_sha256": _sha256(lock_path),
        "execution_governance_amendment": str(amendment_path),
        "execution_governance_amendment_sha256": _sha256(amendment_path),
        "authorized_subjects": protocol["population"]["subjects"],
        "authorized_profiles": ["canary"],
        "authorized_by": "project_owner",
        "authorization_basis": {
            "decision": "Prospective exploratory owner authorization fixture"
        },
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
    }
    authorization_path.write_text(
        json.dumps(authorization) + "\n", encoding="utf-8"
    )
    return protocol, protocol_path, lock_path, amendment_path, authorization_path


def test_exploratory_owner_authorization_accepts_hash_bound_artifacts(
    tmp_path: Path,
) -> None:
    protocol, protocol_path, lock_path, amendment_path, authorization_path = (
        _fixture_artifacts(tmp_path)
    )

    loaded = exploratory.validate_execution_authorization(
        authorization_path,
        governance_amendment_path=amendment_path,
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=_sha256(protocol_path),
        locked_inputs_manifest_path=lock_path,
        subject_id="kimi-k2.6",
    )

    assert loaded["authorization_type"] == "project_owner_exploratory"
    assert loaded["osf_filing_required"] is False
    assert loaded["submission_permitted"] is True


def test_exploratory_owner_authorization_rejects_stale_amendment_hash(
    tmp_path: Path,
) -> None:
    protocol, protocol_path, lock_path, amendment_path, authorization_path = (
        _fixture_artifacts(tmp_path)
    )
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["execution_governance_amendment_sha256"] = "0" * 64
    authorization_path.write_text(json.dumps(authorization) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="governance-amendment hash mismatch"):
        exploratory.validate_execution_authorization(
            authorization_path,
            governance_amendment_path=amendment_path,
            protocol=protocol,
            protocol_path=protocol_path,
            protocol_sha256=_sha256(protocol_path),
            locked_inputs_manifest_path=lock_path,
            subject_id="kimi-k2.6",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("osf_filing_required", True, "must explicitly waive a new OSF filing"),
        (
            "official_response_artifacts_at_approval",
            1,
            "must precede every official response artifact",
        ),
        (
            "scientific_contract_unchanged",
            False,
            "must leave the scientific contract unchanged",
        ),
    ],
)
def test_exploratory_governance_amendment_rejects_invalid_scope(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    protocol, protocol_path, lock_path, amendment_path, _authorization_path = (
        _fixture_artifacts(tmp_path)
    )
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment[field] = value
    amendment_path.write_text(json.dumps(amendment) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        exploratory.validate_execution_governance_amendment(
            amendment_path,
            protocol=protocol,
            protocol_path=protocol_path,
            protocol_sha256=_sha256(protocol_path),
            locked_inputs_manifest_path=lock_path,
        )


def test_checked_in_owner_authorization_is_ready_for_kimi_canary() -> None:
    protocol, protocol_sha256 = control.load_protocol(FROZEN_PROTOCOL)

    loaded = exploratory.validate_execution_authorization(
        EXECUTION_AUTHORIZATION,
        governance_amendment_path=GOVERNANCE_AMENDMENT,
        protocol=protocol,
        protocol_path=FROZEN_PROTOCOL,
        protocol_sha256=protocol_sha256,
        locked_inputs_manifest_path=LOCKED_INPUTS,
        subject_id="kimi-k2.6",
    )

    assert loaded["authorized_subjects"][0] == "kimi-k2.6"
    assert loaded["spending_authorization"][
        "kimi_native_batch_profile_caps_usd"
    ]["canary"] == pytest.approx(0.035798)

    readiness = exploratory.validate_ready(
        Namespace(
            protocol=FROZEN_PROTOCOL,
            locked_inputs_manifest=LOCKED_INPUTS,
            execution_governance_amendment=GOVERNANCE_AMENDMENT,
            execution_authorization=EXECUTION_AUTHORIZATION,
            output_dir=(
                REPO_ROOT
                / "outputs/postreg_disabled_format_control_v3_grok_gpt_kimi/"
                "canary/batches"
            ),
            profile="canary",
            subject=["kimi-k2.6"],
        )
    )
    assert readiness["locked_rows_by_subject"] == {"kimi-k2.6": 3}
    assert readiness["ready_for_submission"] is True
    assert readiness["network_request_made"] is False


def test_exploratory_launcher_delegates_with_v2_receipt_without_claiming_osf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_validator = control.validate_execution_authorization

    def fake_frozen_submit(args: Namespace) -> dict:
        protocol, protocol_sha256 = control.load_protocol(args.protocol)
        compatibility_view = control.validate_execution_authorization(
            args.execution_authorization,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            locked_inputs_manifest_path=args.locked_inputs_manifest,
            subject_id=args.subject[0],
        )
        assert compatibility_view["osf_registration_url"] is None
        return {
            "subject_id": args.subject[0],
            "submission_status": "fixture_only",
            "osf_registration_url": None,
        }

    monkeypatch.setattr(control, "submit_generated", fake_frozen_submit)
    args = Namespace(
        protocol=FROZEN_PROTOCOL,
        locked_inputs_manifest=LOCKED_INPUTS,
        execution_governance_amendment=GOVERNANCE_AMENDMENT,
        execution_authorization=EXECUTION_AUTHORIZATION,
        output_dir=tmp_path,
        profile="canary",
        subject=["kimi-k2.6"],
    )

    result = exploratory.submit_generated(args)

    assert "osf_registration_url" not in result
    assert result["osf_filing_required"] is False
    assert result["authorization_type"] == "project_owner_exploratory"
    assert control.validate_execution_authorization is original_validator
    written = json.loads(
        (tmp_path / "submit_manifest_kimi-k2.6.json").read_text(encoding="utf-8")
    )
    assert written == result


def test_checked_in_authorization_blocks_full_kimi_until_canary_review() -> None:
    with pytest.raises(ValueError, match="profile full is not authorized"):
        exploratory.validate_ready(
            Namespace(
                protocol=FROZEN_PROTOCOL,
                locked_inputs_manifest=LOCKED_INPUTS,
                execution_governance_amendment=GOVERNANCE_AMENDMENT,
                execution_authorization=EXECUTION_AUTHORIZATION,
                output_dir=(
                    REPO_ROOT
                    / "outputs/postreg_disabled_format_control_v3_grok_gpt_kimi/"
                    "batches"
                ),
                profile="full",
                subject=["kimi-k2.6"],
            )
        )
