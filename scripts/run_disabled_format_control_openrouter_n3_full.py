#!/usr/bin/env python3
"""Gate full OpenRouter n=3 shards on the completed excluded-canary review."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_disabled_format_control as control  # noqa: E402
from scripts import run_disabled_format_control_exploratory as exploratory  # noqa: E402
from scripts import run_disabled_format_control_openrouter_n3 as openrouter_n3  # noqa: E402


FULL_AUTHORIZATION_SCHEMA = (
    "disabled_format_control.openrouter_n3_full_execution_authorization.v1"
)
FULL_AMENDMENT_SCHEMA = (
    "disabled_format_control.openrouter_n3_full_activation_amendment.v1"
)
DEFAULT_FULL_EXECUTION_AUTHORIZATION = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_authorization_v3_openrouter_n3_eligible_full.json"
)


def _require_exact_bound_file(
    record: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    expected_path: Path | None = None,
    label: str,
) -> Path:
    return exploratory._require_bound_file(
        record,
        path_field=path_field,
        hash_field=hash_field,
        expected_path=expected_path,
        label=label,
    )


def validate_full_execution_authorization(
    path: Path,
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    locked_inputs_manifest_path: Path,
    canary_authorization_path: Path,
    subject_id: str,
) -> dict[str, Any]:
    """Validate the post-canary supplement authorizing one passing full shard."""
    authorization = exploratory._read_json(path)
    if authorization.get("schema_version") != FULL_AUTHORIZATION_SCHEMA:
        raise ValueError("unsupported full execution-authorization schema_version")
    if authorization.get("status") != "frozen_after_canary_before_full_dispatch":
        raise ValueError("full execution authorization is not prospectively frozen")
    if authorization.get("submission_permitted") is not True:
        raise ValueError("full execution authorization is not executable")
    if authorization.get("authorized_profiles") != ["full"]:
        raise ValueError("full execution authorization must authorize only full")
    if authorization.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("full execution authorization protocol_id mismatch")
    if authorization.get("protocol_sha256") != protocol_sha256:
        raise ValueError("full execution authorization protocol hash mismatch")
    _require_exact_bound_file(
        authorization,
        path_field="protocol_path",
        hash_field="protocol_sha256",
        expected_path=protocol_path,
        label="full-authorization protocol",
    )
    _require_exact_bound_file(
        authorization,
        path_field="locked_inputs_manifest",
        hash_field="locked_inputs_manifest_sha256",
        expected_path=locked_inputs_manifest_path,
        label="full-authorization locked-input",
    )
    actual_canary_authorization_path = _require_exact_bound_file(
        authorization,
        path_field="canary_execution_authorization",
        hash_field="canary_execution_authorization_sha256",
        expected_path=canary_authorization_path,
        label="full-authorization canary receipt",
    )
    canary_authorization = exploratory._read_json(actual_canary_authorization_path)
    if canary_authorization.get("authorized_profiles") != ["canary"]:
        raise ValueError("canonical receipt must remain canary-only")
    canary_governance_path = exploratory._resolve_recorded_path(
        canary_authorization.get("execution_governance_amendment"),
        "canary execution_governance_amendment",
    )
    exploratory.validate_execution_authorization(
        actual_canary_authorization_path,
        governance_amendment_path=canary_governance_path,
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        locked_inputs_manifest_path=locked_inputs_manifest_path,
        subject_id=subject_id,
    )

    review_path = _require_exact_bound_file(
        authorization,
        path_field="canary_review",
        hash_field="canary_review_sha256",
        label="full-authorization canary review",
    )
    review = exploratory._read_json(review_path)
    if review.get("status") != "complete_before_full_authorization":
        raise ValueError("canary review is incomplete")
    if review.get("protocol_sha256") != protocol_sha256:
        raise ValueError("canary review protocol hash mismatch")
    passing_subjects = list(map(str, review.get("passing_subjects") or []))
    stopped_subjects = list(map(str, review.get("stopped_subjects") or []))
    if subject_id not in passing_subjects:
        raise ValueError(f"subject {subject_id} is not authorized for full dispatch")
    subject_review = review.get("subjects", {}).get(subject_id)
    if not isinstance(subject_review, dict) or subject_review.get("verdict") != "pass_for_full":
        raise ValueError(f"subject {subject_id} lacks a passing canary verdict")

    amendment_path = _require_exact_bound_file(
        authorization,
        path_field="full_activation_amendment",
        hash_field="full_activation_amendment_sha256",
        label="full-authorization activation amendment",
    )
    amendment = exploratory._read_json(amendment_path)
    if amendment.get("schema_version") != FULL_AMENDMENT_SCHEMA:
        raise ValueError("unsupported full activation-amendment schema_version")
    if amendment.get("status") != "frozen_after_canary_before_full_dispatch":
        raise ValueError("full activation amendment is not prospectively frozen")
    if amendment.get("approved_by") != "project_owner":
        raise ValueError("full activation amendment lacks project-owner authority")
    if amendment.get("official_full_response_artifacts_at_approval") != 0:
        raise ValueError("full activation amendment must precede full responses")
    if amendment.get("protocol_sha256") != protocol_sha256:
        raise ValueError("full activation amendment protocol hash mismatch")
    _require_exact_bound_file(
        amendment,
        path_field="canary_review",
        hash_field="canary_review_sha256",
        expected_path=review_path,
        label="full-amendment canary review",
    )
    _require_exact_bound_file(
        amendment,
        path_field="full_activation_launcher",
        hash_field="full_activation_launcher_sha256",
        expected_path=Path(__file__),
        label="full-amendment launcher",
    )
    reviewed_at = exploratory._parse_utc_timestamp(
        review.get("reviewed_at_utc"), "reviewed_at_utc"
    )
    approved_at = exploratory._parse_utc_timestamp(
        amendment.get("approved_at_utc"), "approved_at_utc"
    )
    created_at = exploratory._parse_utc_timestamp(
        authorization.get("created_at_utc"), "created_at_utc"
    )
    if approved_at < reviewed_at or created_at < approved_at:
        raise ValueError("full activation chronology is invalid")

    authorized_subjects = list(map(str, authorization.get("authorized_subjects") or []))
    if authorized_subjects != passing_subjects:
        raise ValueError("full authorized subjects differ from canary passes")
    if list(map(str, authorization.get("stopped_subjects") or [])) != stopped_subjects:
        raise ValueError("full stopped subjects differ from canary review")
    if subject_id not in authorized_subjects:
        raise ValueError(f"subject {subject_id} is not authorized for full dispatch")

    canonical_caps = canary_authorization["spending_authorization"][
        "openrouter_profile_subject_caps_usd"
    ]["full"]
    spending = authorization.get("spending_authorization")
    if not isinstance(spending, dict):
        raise ValueError("full execution authorization lacks spending authorization")
    full_caps = spending.get("openrouter_full_subject_caps_usd")
    expected_caps = {subject: canonical_caps[subject] for subject in passing_subjects}
    if full_caps != expected_caps:
        raise ValueError("full subject caps differ from the prospectively allocated caps")
    used_after_canary = spending.get("openrouter_usage_after_canary_usd")
    limit = spending.get("openrouter_limit_usd")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (used_after_canary, limit)
    ):
        raise ValueError("full spending ledger is invalid")
    if sum(map(float, full_caps.values())) > float(limit) - float(used_after_canary) + 1e-9:
        raise ValueError("full subject caps exceed the post-canary account envelope")
    return authorization


def _with_full_profile_gate(
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    original = exploratory._require_authorized_profile

    def require_canary_receipt_or_full_supplement(
        authorization: dict[str, Any], *, profile: str
    ) -> None:
        if profile == "full":
            return
        original(authorization, profile=profile)

    exploratory._require_authorized_profile = require_canary_receipt_or_full_supplement
    try:
        return operation()
    finally:
        exploratory._require_authorized_profile = original


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def execute(args: Any, full_authorization_path: Path) -> dict[str, Any]:
    if args.profile != "full":
        raise ValueError("post-canary launcher accepts only --profile full")
    if not args.subject or len(args.subject) != 1:
        raise ValueError("post-canary launcher requires exactly one --subject")
    subject_id = str(args.subject[0])
    protocol, protocol_sha256 = control.load_protocol(args.protocol)
    openrouter_n3.install_protocol_route_entries(protocol)
    authorization = validate_full_execution_authorization(
        full_authorization_path,
        protocol=protocol,
        protocol_path=args.protocol,
        protocol_sha256=protocol_sha256,
        locked_inputs_manifest_path=args.locked_inputs_manifest,
        canary_authorization_path=args.execution_authorization,
        subject_id=subject_id,
    )
    review_path = exploratory._resolve_recorded_path(
        authorization["canary_review"], "canary_review"
    )
    amendment_path = exploratory._resolve_recorded_path(
        authorization["full_activation_amendment"], "full_activation_amendment"
    )
    full_authorization_sha256 = exploratory._sha256_file(full_authorization_path)
    ledger_path = args.output_dir / f"full_activation_{subject_id}.json"
    ledger = {
        "schema_version": "disabled_format_control.openrouter_n3_full_activation_ledger.v1",
        "status": "validation_only" if not args.submit else "submission_started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "protocol_sha256": protocol_sha256,
        "canary_review": str(review_path),
        "canary_review_sha256": exploratory._sha256_file(review_path),
        "full_execution_authorization": str(full_authorization_path),
        "full_execution_authorization_sha256": full_authorization_sha256,
        "full_activation_amendment": str(amendment_path),
        "full_activation_amendment_sha256": exploratory._sha256_file(amendment_path),
    }
    if args.submit:
        _write_json(ledger_path, ledger)
    try:
        result = _with_full_profile_gate(
            lambda: (
                exploratory.submit_generated(args)
                if args.submit
                else exploratory.validate_ready(args)
            )
        )
    except Exception as exc:
        if args.submit:
            _write_json(
                ledger_path,
                {
                    **ledger,
                    "status": "failed_or_indeterminate",
                    "error_type": type(exc).__name__,
                },
            )
        raise
    decorated = {
        **result,
        "canary_review_path": str(review_path),
        "canary_review_sha256": exploratory._sha256_file(review_path),
        "full_execution_authorization_path": str(full_authorization_path),
        "full_execution_authorization_sha256": full_authorization_sha256,
        "full_activation_amendment_path": str(amendment_path),
        "full_activation_amendment_sha256": exploratory._sha256_file(amendment_path),
        "full_activation_launcher_sha256": exploratory._sha256_file(Path(__file__)),
    }
    if args.submit:
        submit_manifest_path = args.output_dir / f"submit_manifest_{subject_id}.json"
        _write_json(submit_manifest_path, decorated)
        _write_json(
            ledger_path,
            {
                **ledger,
                "status": decorated.get("submission_status", "completed"),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "submit_manifest_path": str(submit_manifest_path),
                "reported_cost_usd": decorated.get("reported_cost_usd"),
                "responses": decorated.get("responses"),
                "skipped_existing": decorated.get("skipped_existing"),
            },
        )
    return decorated


def _parse_args(argv: list[str] | None) -> tuple[Any, Path]:
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument(
        "--full-execution-authorization",
        type=Path,
        default=DEFAULT_FULL_EXECUTION_AUTHORIZATION,
    )
    gate_args, remaining = gate.parse_known_args(argv)
    return openrouter_n3._parse_args(remaining), gate_args.full_execution_authorization


def main(argv: list[str] | None = None) -> int:
    args, full_authorization_path = _parse_args(argv)
    try:
        result = execute(args, full_authorization_path)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
