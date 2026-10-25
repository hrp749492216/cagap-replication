#!/usr/bin/env python3
"""Authorize and submit the frozen v3 control as an exploratory study.

This launcher changes only the activation evidence required by the frozen
2026-09-03 protocol. It validates a prospective owner-governance amendment and
v2 execution receipt, then delegates all scientific input, route, transport,
and spending behavior to the byte-pinned control runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_disabled_format_control as control  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "experiments/disabled_format_control/protocol_v3_grok_gpt_kimi.json"
)
DEFAULT_LOCKED_INPUTS = (
    REPO_ROOT
    / "experiments/disabled_format_control/locked_inputs_manifest_v3_grok_gpt_kimi.json"
)
DEFAULT_GOVERNANCE_AMENDMENT = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_governance_amendment_v3_grok_gpt_kimi_2026-09-05.json"
)
DEFAULT_EXECUTION_AUTHORIZATION = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_authorization_v3_grok_gpt_kimi.json"
)
GOVERNANCE_SCHEMA = "disabled_format_control.execution_governance_amendment.v1"
AUTHORIZATION_SCHEMA = "disabled_format_control.execution_authorization.v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_recorded_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is missing")
    path = Path(value)
    if not path.is_absolute():
        if ".." in path.parts:
            raise ValueError(f"{field_name} must not escape the repository")
        path = REPO_ROOT / path
    return path.resolve()


def _require_bound_file(
    record: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    expected_path: Path | None = None,
    label: str,
) -> Path:
    path = _resolve_recorded_path(record.get(path_field), path_field)
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{label} path mismatch")
    if not path.is_file():
        raise FileNotFoundError(path)
    if record.get(hash_field) != _sha256_file(path):
        raise ValueError(f"{label} hash mismatch")
    return path


def _parse_utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field_name} must be a UTC timestamp")
    return parsed


def validate_execution_governance_amendment(
    path: Path,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    locked_inputs_manifest_path: Path,
) -> dict[str, Any]:
    """Validate the pre-response owner decision that replaces the OSF-only gate."""
    if not path.is_file():
        raise FileNotFoundError(path)
    amendment = _read_json(path)
    if amendment.get("schema_version") != GOVERNANCE_SCHEMA:
        raise ValueError("unsupported execution-governance-amendment schema_version")
    if amendment.get("status") != "frozen_before_first_official_dispatch":
        raise ValueError("execution governance amendment is not prospectively frozen")
    if protocol.get("status") != "frozen_before_first_dispatch":
        raise ValueError("exploratory activation requires a frozen protocol")
    if "exploratory" not in str(protocol.get("registration_scope", "")).lower():
        raise ValueError("protocol is not classified as exploratory")
    if amendment.get("registration_status") != "post_registration_exploratory":
        raise ValueError("governance amendment has the wrong registration status")
    if amendment.get("osf_filing_required") is not False:
        raise ValueError("governance amendment must explicitly waive a new OSF filing")
    if amendment.get("official_response_artifacts_at_approval") != 0:
        raise ValueError("governance amendment must precede every official response artifact")
    if amendment.get("scientific_contract_unchanged") is not True:
        raise ValueError("governance amendment must leave the scientific contract unchanged")
    if amendment.get("superseded_activation_requirement") != "osf_filing_only":
        raise ValueError("governance amendment may supersede only the OSF filing gate")
    _parse_utc_timestamp(amendment.get("approved_at_utc"), "approved_at_utc")
    approved_by = amendment.get("approved_by")
    if approved_by != "project_owner":
        raise ValueError("governance amendment must be approved by project_owner")
    if amendment.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("governance amendment protocol_id mismatch")
    if amendment.get("protocol_sha256") != protocol_sha256:
        raise ValueError("governance amendment protocol hash mismatch")
    _require_bound_file(
        amendment,
        path_field="protocol_path",
        hash_field="protocol_sha256",
        expected_path=protocol_path,
        label="governance-amendment protocol",
    )
    lock_path = _require_bound_file(
        amendment,
        path_field="locked_inputs_manifest",
        hash_field="locked_inputs_manifest_sha256",
        expected_path=locked_inputs_manifest_path,
        label="governance-amendment locked-input",
    )
    locked_inputs = _read_json(lock_path)
    if locked_inputs.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("locked-input manifest protocol_id mismatch")
    if locked_inputs.get("protocol_sha256") != protocol_sha256:
        raise ValueError("locked-input manifest protocol hash mismatch")
    _require_bound_file(
        amendment,
        path_field="approval_document",
        hash_field="approval_document_sha256",
        label="governance-amendment approval document",
    )
    _require_bound_file(
        amendment,
        path_field="activation_launcher",
        hash_field="activation_launcher_sha256",
        expected_path=Path(__file__),
        label="governance-amendment activation launcher",
    )
    _require_bound_file(
        amendment,
        path_field="delegated_frozen_runner",
        hash_field="delegated_frozen_runner_sha256",
        expected_path=Path(control.__file__),
        label="governance-amendment delegated frozen runner",
    )
    return amendment


def _validate_spending_authorization(
    authorization: dict[str, Any],
    *,
    protocol: dict[str, Any],
) -> None:
    spending = authorization.get("spending_authorization")
    if not isinstance(spending, dict) or not spending:
        raise ValueError("execution authorization is missing spending_authorization")

    openrouter_cap = spending.get("openrouter_cap_usd")
    openrouter_spent = spending.get("openrouter_spent_before_official_usd")
    if (
        isinstance(openrouter_cap, bool)
        or not isinstance(openrouter_cap, (int, float))
        or not math.isfinite(float(openrouter_cap))
        or float(openrouter_cap) <= 0
    ):
        raise ValueError("execution authorization needs a positive OpenRouter cap")
    if (
        isinstance(openrouter_spent, bool)
        or not isinstance(openrouter_spent, (int, float))
        or not math.isfinite(float(openrouter_spent))
        or float(openrouter_spent) < 0
        or float(openrouter_spent) > float(openrouter_cap)
    ):
        raise ValueError("execution authorization has invalid prior OpenRouter spend")

    routes = protocol.get("route_policy", {}).get("routes", {})
    if not isinstance(routes, dict):
        raise ValueError("protocol route table is missing")
    openrouter_subjects = {
        str(subject)
        for subject, descriptor in routes.items()
        if isinstance(descriptor, dict) and descriptor.get("route") == "openrouter"
    }
    allocations = spending.get("openrouter_profile_subject_caps_usd")
    if not isinstance(allocations, dict):
        raise ValueError("execution authorization needs OpenRouter profile allocations")
    allocation_total = 0.0
    for profile in ("canary", "full"):
        profile_allocations = allocations.get(profile)
        if not isinstance(profile_allocations, dict):
            raise ValueError(
                f"execution authorization needs OpenRouter {profile} allocations"
            )
        if set(map(str, profile_allocations)) != openrouter_subjects:
            raise ValueError(
                f"execution authorization {profile} OpenRouter subjects mismatch"
            )
        for subject, value in profile_allocations.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    "execution authorization has an invalid OpenRouter allocation "
                    f"for {profile}/{subject}"
                )
            allocation_total += float(value)
    if allocation_total > float(openrouter_cap) - float(openrouter_spent) + 1e-9:
        raise ValueError(
            "execution authorization OpenRouter allocations exceed the remaining cap"
        )

    kimi_descriptor = routes.get("kimi-k2.6")
    kimi_is_native = (
        isinstance(kimi_descriptor, dict) and kimi_descriptor.get("route") == "native"
    )
    if not kimi_is_native:
        return
    if spending.get("kimi_native_batch") is not True:
        raise ValueError("execution authorization does not authorize Kimi native batch")
    kimi_cap = spending.get("kimi_native_batch_cap_usd")
    kimi_spent = spending.get("kimi_native_batch_spent_before_official_usd")
    kimi_allocations = spending.get("kimi_native_batch_profile_caps_usd")
    if (
        isinstance(kimi_cap, bool)
        or not isinstance(kimi_cap, (int, float))
        or not math.isfinite(float(kimi_cap))
        or float(kimi_cap) <= 0
    ):
        raise ValueError("execution authorization needs a positive Kimi batch cap")
    if (
        isinstance(kimi_spent, bool)
        or not isinstance(kimi_spent, (int, float))
        or not math.isfinite(float(kimi_spent))
        or float(kimi_spent) < 0
        or float(kimi_spent) > float(kimi_cap)
    ):
        raise ValueError("execution authorization has invalid prior Kimi spend")
    if not isinstance(kimi_allocations, dict):
        raise ValueError("execution authorization needs Kimi batch profile allocations")
    kimi_total = 0.0
    for profile in ("canary", "full"):
        value = kimi_allocations.get(profile)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError(
                f"execution authorization needs a positive Kimi {profile} allocation"
            )
        kimi_total += float(value)
    if kimi_total > float(kimi_cap) - float(kimi_spent) + 1e-9:
        raise ValueError(
            "execution authorization Kimi allocations exceed the remaining Kimi cap"
        )


def validate_execution_authorization(
    path: Path | None,
    *,
    governance_amendment_path: Path,
    protocol: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    locked_inputs_manifest_path: Path,
    subject_id: str,
) -> dict[str, Any]:
    """Validate the immutable v2 exploratory receipt before any network access."""
    if path is None:
        raise ValueError("exploratory submission requires --execution-authorization")
    if not path.is_file():
        raise FileNotFoundError(path)
    authorization = _read_json(path)
    if authorization.get("submission_permitted") is not True:
        raise ValueError("execution authorization must set submission_permitted=true")
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported exploratory execution-authorization schema_version")
    if authorization.get("authorization_type") != "project_owner_exploratory":
        raise ValueError("execution authorization has the wrong authorization_type")
    if authorization.get("registration_status") != "post_registration_exploratory":
        raise ValueError("execution authorization has the wrong registration status")
    if authorization.get("osf_filing_required") is not False:
        raise ValueError("exploratory execution authorization must set OSF filing false")
    if authorization.get("osf_registration_url") not in (None, ""):
        raise ValueError("exploratory execution authorization must not claim an OSF URL")
    if authorization.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("execution authorization protocol_id mismatch")
    if authorization.get("protocol_sha256") != protocol_sha256:
        raise ValueError("execution authorization protocol hash mismatch")
    _require_bound_file(
        authorization,
        path_field="locked_inputs_manifest",
        hash_field="locked_inputs_manifest_sha256",
        expected_path=locked_inputs_manifest_path,
        label="execution-authorization locked-input",
    )
    actual_amendment_path = _require_bound_file(
        authorization,
        path_field="execution_governance_amendment",
        hash_field="execution_governance_amendment_sha256",
        expected_path=governance_amendment_path,
        label="governance-amendment",
    )
    amendment = validate_execution_governance_amendment(
        actual_amendment_path,
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        locked_inputs_manifest_path=locked_inputs_manifest_path,
    )

    authorized_at = _parse_utc_timestamp(
        authorization.get("owner_authorized_at_utc"), "owner_authorized_at_utc"
    )
    approved_at = _parse_utc_timestamp(amendment.get("approved_at_utc"), "approved_at_utc")
    if authorized_at < approved_at:
        raise ValueError("execution authorization predates its governance amendment")
    receipt_created_at = _parse_utc_timestamp(
        authorization.get("receipt_created_at_utc"), "receipt_created_at_utc"
    )
    if receipt_created_at < authorized_at:
        raise ValueError("execution receipt predates owner authorization")
    if authorization.get("authorized_by") != amendment.get("approved_by"):
        raise ValueError("execution authorization owner does not match its amendment")
    authorization_basis = authorization.get("authorization_basis")
    if not isinstance(authorization_basis, dict) or not authorization_basis:
        raise ValueError("execution authorization is missing authorization_basis")

    frozen_subjects = [str(value) for value in protocol["population"]["subjects"]]
    authorized_subjects = authorization.get("authorized_subjects")
    if not isinstance(authorized_subjects, list) or list(
        map(str, authorized_subjects)
    ) != frozen_subjects:
        raise ValueError("execution authorization subject population mismatch")
    if subject_id not in set(map(str, authorized_subjects)):
        raise ValueError(f"execution authorization excludes subject {subject_id}")
    authorized_profiles = authorization.get("authorized_profiles")
    if (
        not isinstance(authorized_profiles, list)
        or not authorized_profiles
        or len(set(map(str, authorized_profiles))) != len(authorized_profiles)
        or not set(map(str, authorized_profiles)).issubset({"canary", "full"})
    ):
        raise ValueError("execution authorization has invalid authorized_profiles")
    _validate_spending_authorization(authorization, protocol=protocol)
    return authorization


def _require_authorized_profile(
    authorization: dict[str, Any], *, profile: str
) -> None:
    if profile not in set(map(str, authorization["authorized_profiles"])):
        raise ValueError(
            f"profile {profile} is not authorized; complete and review the canary first"
        )


def _decorate_submission_manifest(
    result: dict[str, Any],
    *,
    args: argparse.Namespace,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    decorated = dict(result)
    decorated.pop("osf_registration_url", None)
    decorated.update(
        {
            "authorization_type": authorization["authorization_type"],
            "registration_status": authorization["registration_status"],
            "osf_filing_required": False,
            "owner_authorized_at_utc": authorization["owner_authorized_at_utc"],
            "execution_governance_amendment_path": str(
                args.execution_governance_amendment
            ),
            "execution_governance_amendment_sha256": _sha256_file(
                args.execution_governance_amendment
            ),
            "activation_launcher_sha256": _sha256_file(Path(__file__)),
        }
    )
    manifest_path = args.output_dir / f"submit_manifest_{args.subject[0]}.json"
    manifest_path.write_text(
        json.dumps(
            decorated,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return decorated


def submit_generated(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the exploratory receipt, then delegate to the frozen runner."""
    protocol, protocol_sha256 = control.load_protocol(args.protocol)
    if not args.subject or len(args.subject) != 1:
        raise ValueError("paid submission requires exactly one --subject per invocation")
    subject_id = str(args.subject[0])
    authorization = validate_execution_authorization(
        args.execution_authorization,
        governance_amendment_path=args.execution_governance_amendment,
        protocol=protocol,
        protocol_path=args.protocol,
        protocol_sha256=protocol_sha256,
        locked_inputs_manifest_path=args.locked_inputs_manifest,
        subject_id=subject_id,
    )
    _require_authorized_profile(authorization, profile=str(args.profile))

    original_validator = control.validate_execution_authorization

    def validate_v2_for_frozen_runner(
        path: Path | None,
        *,
        protocol: dict[str, Any],
        protocol_sha256: str,
        locked_inputs_manifest_path: Path,
        subject_id: str,
    ) -> dict[str, Any]:
        validated = validate_execution_authorization(
            path,
            governance_amendment_path=args.execution_governance_amendment,
            protocol=protocol,
            protocol_path=args.protocol,
            protocol_sha256=protocol_sha256,
            locked_inputs_manifest_path=locked_inputs_manifest_path,
            subject_id=subject_id,
        )
        compatibility_view = dict(validated)
        compatibility_view["osf_registration_url"] = None
        return compatibility_view

    control.validate_execution_authorization = validate_v2_for_frozen_runner
    try:
        result = control.submit_generated(args)
    finally:
        control.validate_execution_authorization = original_validator
    return _decorate_submission_manifest(result, args=args, authorization=authorization)


def validate_ready(args: argparse.Namespace) -> dict[str, Any]:
    """Verify activation and locked request artifacts without loading credentials."""
    protocol, protocol_sha256 = control.load_protocol(args.protocol)
    subjects = args.subject or list(protocol["population"]["subjects"])
    profile = str(args.profile)
    artifact_paths = protocol.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        raise ValueError("protocol artifact paths are missing")
    batches_key = "canary_batches_dir" if profile == "canary" else "full_batches_dir"
    output_dir = args.output_dir or REPO_ROOT / str(artifact_paths[batches_key])
    control.validate_protocol_output_target(
        protocol,
        output_dir,
        canary=profile == "canary",
    )
    required_lock = _resolve_recorded_path(
        artifact_paths.get("locked_inputs_manifest"),
        "protocol locked_inputs_manifest",
    )
    if required_lock != args.locked_inputs_manifest.resolve():
        raise ValueError("locked-input manifest path differs from frozen protocol")
    required_authorization = _resolve_recorded_path(
        artifact_paths.get("execution_authorization"),
        "protocol execution_authorization",
    )
    if required_authorization != args.execution_authorization.resolve():
        raise ValueError("execution-authorization path differs from frozen protocol")

    manifest_path = output_dir / "disabled_format_control_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("protocol_sha256") != protocol_sha256:
        raise ValueError("generated manifest protocol hash mismatch")
    if manifest.get("profile") != profile:
        raise ValueError("generated manifest profile mismatch")
    locked_inputs = _read_json(args.locked_inputs_manifest)
    if locked_inputs.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("locked-input manifest protocol_id mismatch")
    if locked_inputs.get("protocol_sha256") != protocol_sha256:
        raise ValueError("locked-input manifest protocol hash mismatch")
    locked_profile = locked_inputs.get(profile)
    if not isinstance(locked_profile, dict):
        raise ValueError(f"locked-input manifest has no {profile!r} profile")

    locked_rows_by_subject: dict[str, int] = {}
    route_by_subject: dict[str, str] = {}
    authorized_cost_cap_by_subject: dict[str, float] = {}
    estimated_cost_ceiling_by_subject: dict[str, float] = {}
    for subject_id in subjects:
        subject = str(subject_id)
        authorization = validate_execution_authorization(
            args.execution_authorization,
            governance_amendment_path=args.execution_governance_amendment,
            protocol=protocol,
            protocol_path=args.protocol,
            protocol_sha256=protocol_sha256,
            locked_inputs_manifest_path=args.locked_inputs_manifest,
            subject_id=subject,
        )
        _require_authorized_profile(authorization, profile=profile)
        rows = control._load_generated_rows(
            output_dir,
            subject,
            manifest,
            locked_profile,
        )
        control.validate_generated_rows(rows, protocol, canary=profile == "canary")
        route, _descriptor = control._route_spec(protocol, subject)
        authorized_cap = control._authorized_profile_subject_cap(
            authorization,
            profile=profile,
            subject_id=subject,
        )
        cost = control._estimated_cost_upper_bound(rows, protocol)
        estimated_cost = float(cost["total_api_usd"])
        if route == "native" and estimated_cost > authorized_cap + 1e-9:
            raise ValueError(
                f"{profile}/{subject} conservative native cost exceeds authorization"
            )
        if route == "native":
            submit_manifest_path = output_dir / f"submit_manifest_{subject}.json"
            if submit_manifest_path.exists():
                raise ValueError(
                    f"native submission ledger already exists at {submit_manifest_path}"
                )
        locked_rows_by_subject[subject] = len(rows)
        route_by_subject[subject] = route
        authorized_cost_cap_by_subject[subject] = authorized_cap
        estimated_cost_ceiling_by_subject[subject] = estimated_cost
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "profile": profile,
        "registration_status": "post_registration_exploratory",
        "osf_filing_required": False,
        "authorized_subjects": list(map(str, subjects)),
        "locked_rows_by_subject": locked_rows_by_subject,
        "route_by_subject": route_by_subject,
        "authorized_cost_cap_usd_by_subject": authorized_cost_cap_by_subject,
        "estimated_cost_ceiling_usd_by_subject": estimated_cost_ceiling_by_subject,
        "execution_authorization_sha256": _sha256_file(
            args.execution_authorization
        ),
        "execution_governance_amendment_sha256": _sha256_file(
            args.execution_governance_amendment
        ),
        "network_request_made": False,
        "ready_for_submission": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--validate-authorization",
        action="store_true",
        help="validate the owner amendment and receipt without API access",
    )
    action.add_argument(
        "--submit",
        action="store_true",
        help="submit one locked subject through the frozen runner",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--locked-inputs-manifest", type=Path, default=DEFAULT_LOCKED_INPUTS
    )
    parser.add_argument(
        "--execution-governance-amendment",
        type=Path,
        default=DEFAULT_GOVERNANCE_AMENDMENT,
    )
    parser.add_argument(
        "--execution-authorization",
        type=Path,
        default=DEFAULT_EXECUTION_AUTHORIZATION,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=("full", "canary"), default="canary")
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument(
        "--acknowledge-postregistration-exploratory", action="store_true"
    )
    parser.add_argument("--confirm-spend", default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    args = parser.parse_args(argv)
    if args.submit and args.output_dir is None:
        parser.error("--submit requires --output-dir")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = submit_generated(args) if args.submit else validate_ready(args)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
