#!/usr/bin/env python3
"""Reconcile the completed OpenRouter n=3 control before downstream use."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_disabled_format_control as control  # noqa: E402
from scripts import run_disabled_format_control_openrouter_n3 as openrouter_n3  # noqa: E402
from scripts import run_disabled_format_control_openrouter_n3_full as full_launcher  # noqa: E402
from scripts.run_synthetic_prompts import redact_reasoning_fields  # noqa: E402


RECONCILIATION_SCHEMA = "disabled_format_control.openrouter_n3_execution_reconciliation.v1"
DEFAULT_RESPONSES_DIR = (
    REPO_ROOT / "outputs/postreg_disabled_format_control_v3_openrouter_n3_eligible/responses"
)
DEFAULT_CANARY_REVIEW = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "canary_review_v3_openrouter_n3_eligible_2026-09-05.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(value)
    return rows


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reconcile_subject(
    *,
    expected_rows: list[dict[str, Any]],
    observed_rows: list[dict[str, Any]],
    descriptor: dict[str, Any],
    subject_id: str,
    runtime_cost_cap_usd: float,
    recovered_from_raw_archive: bool,
) -> dict[str, Any]:
    """Fail closed unless observed rows exactly match one locked subject shard."""
    expected_by_id = {str(row.get("custom_id") or ""): row for row in expected_rows}
    if "" in expected_by_id or len(expected_by_id) != len(expected_rows):
        raise ValueError(f"{subject_id}: locked inputs have duplicate or missing custom IDs")
    observed_by_id: dict[str, dict[str, Any]] = {}
    for row in observed_rows:
        custom_id = str(row.get("custom_id") or "")
        if not custom_id or custom_id in observed_by_id:
            raise ValueError(f"{subject_id}: duplicate or missing observed custom ID {custom_id!r}")
        observed_by_id[custom_id] = row
    missing = sorted(set(expected_by_id) - set(observed_by_id))
    extra = sorted(set(observed_by_id) - set(expected_by_id))
    if missing or extra:
        raise ValueError(
            f"{subject_id}: response identity mismatch; missing={len(missing)}, extra={len(extra)}"
        )

    provider_only = list(descriptor.get("provider_only") or [])
    if descriptor.get("route") != "openrouter" or len(provider_only) != 1:
        raise ValueError(f"{subject_id}: protocol route is not one exact OpenRouter pin")
    pinned_provider = str(provider_only[0])
    model_slug = str(descriptor["model_slug"])
    expected_provider_identity = control._normalized_provider_identity(pinned_provider)
    cumulative_cost = 0.0
    arm_counts: Counter[str] = Counter()
    trial_counts: Counter[str] = Counter()
    returned_models: Counter[str] = Counter()
    observed_providers: Counter[str] = Counter()
    raw_capture_digest_rows: list[dict[str, Any]] = []

    for expected in expected_rows:
        custom_id = str(expected["custom_id"])
        observed = observed_by_id[custom_id]
        for field, expected_value in expected.items():
            if observed.get(field) != expected_value:
                raise ValueError(f"{subject_id}: frozen field {field!r} drifted for {custom_id}")
        if observed.get("route") != "openrouter" or observed.get("pinned_provider") != pinned_provider:
            raise ValueError(f"{subject_id}: route/provider pin drift for {custom_id}")
        observed_provider = str(observed.get("observed_provider") or "")
        if (
            observed.get("observed_provider_matches_pin") is not True
            or control._normalized_provider_identity(observed_provider)
            != expected_provider_identity
        ):
            raise ValueError(f"{subject_id}: observed provider mismatch for {custom_id}")
        response = observed.get("raw_response")
        if not isinstance(response, dict) or response.get("model") != model_slug:
            raise ValueError(f"{subject_id}: returned model mismatch for {custom_id}")
        if observed.get("response") != response:
            raise ValueError(f"{subject_id}: response aliases differ for {custom_id}")
        runtime_stamp = observed.get("runtime_stamp")
        if not isinstance(runtime_stamp, dict) or any(
            (
                runtime_stamp.get("route") != "openrouter",
                runtime_stamp.get("provider") != pinned_provider,
                runtime_stamp.get("requested_model") != model_slug,
                runtime_stamp.get("returned_model") != model_slug,
                runtime_stamp.get("model_id") != subject_id,
            )
        ):
            raise ValueError(f"{subject_id}: runtime identity mismatch for {custom_id}")
        usage = response.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if (
            isinstance(cost, bool)
            or not isinstance(cost, (int, float))
            or not math.isfinite(float(cost))
            or float(cost) < 0
        ):
            raise ValueError(f"{subject_id}: invalid provider-reported cost for {custom_id}")
        cumulative_cost += float(cost)
        budget = observed.get("execution_budget")
        if not isinstance(budget, dict):
            raise ValueError(f"{subject_id}: missing execution budget for {custom_id}")
        if float(budget.get("response_reported_cost_usd", -1)) != float(cost):
            raise ValueError(f"{subject_id}: response cost ledger mismatch for {custom_id}")
        if float(budget.get("cumulative_reported_cost_usd", -1)) != round(cumulative_cost, 8):
            raise ValueError(f"{subject_id}: cumulative cost ledger mismatch for {custom_id}")
        if float(budget.get("runtime_cost_cap_usd", -1)) != runtime_cost_cap_usd:
            raise ValueError(f"{subject_id}: runtime cost cap mismatch for {custom_id}")
        if cumulative_cost > runtime_cost_cap_usd + 1e-9:
            raise ValueError(f"{subject_id}: runtime cost cap exceeded for {custom_id}")

        capture = observed.get("raw_http_capture")
        if not isinstance(capture, dict):
            raise ValueError(f"{subject_id}: missing raw HTTP capture for {custom_id}")
        body_path = Path(str(capture.get("raw_http_body_path") or ""))
        body_path = body_path if body_path.is_absolute() else REPO_ROOT / body_path
        body_bytes = body_path.read_bytes()
        body_sha256 = hashlib.sha256(body_bytes).hexdigest()
        if (
            capture.get("raw_http_archive_schema_version") != "raw_http_archive.v1"
            or capture.get("raw_http_body_sha256") != body_sha256
            or capture.get("raw_http_body_bytes") != len(body_bytes)
        ):
            raise ValueError(f"{subject_id}: raw HTTP capture mismatch for {custom_id}")
        try:
            raw_response = json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{subject_id}: raw response is not JSON for {custom_id}") from exc
        if redact_reasoning_fields(raw_response) != response:
            raise ValueError(f"{subject_id}: normalized response differs from raw body for {custom_id}")

        recovery = observed.get("artifact_recovery")
        if recovered_from_raw_archive:
            if (
                not isinstance(recovery, dict)
                or recovery.get("provider_request_reissued") is not False
                or observed.get("latency_sec") != -1.0
            ):
                raise ValueError(f"{subject_id}: incomplete raw-archive recovery provenance")
        elif recovery is not None or not isinstance(observed.get("latency_sec"), (int, float)):
            raise ValueError(f"{subject_id}: unexpected recovery marker or latency type")

        arm_counts[str(observed.get("control_arm"))] += 1
        trial_counts[str(observed.get("trial_id"))] += 1
        returned_models[str(response["model"])] += 1
        observed_providers[observed_provider] += 1
        raw_capture_digest_rows.append(
            {
                "custom_id": custom_id,
                "body_sha256": body_sha256,
                "body_bytes": len(body_bytes),
                "captured_at_utc": capture.get("raw_http_captured_at_utc"),
            }
        )

    return {
        "status": "complete_and_reconciled",
        "subject_id": subject_id,
        "expected_rows": len(expected_rows),
        "observed_rows": len(observed_rows),
        "missing_rows": 0,
        "extra_rows": 0,
        "duplicate_rows": 0,
        "arm_counts": dict(sorted(arm_counts.items())),
        "trial_counts": dict(sorted(trial_counts.items())),
        "requested_model": model_slug,
        "returned_model_counts": dict(sorted(returned_models.items())),
        "pinned_provider": pinned_provider,
        "observed_provider_counts": dict(sorted(observed_providers.items())),
        "reported_cost_usd": round(cumulative_cost, 8),
        "runtime_cost_cap_usd": runtime_cost_cap_usd,
        "recovered_from_raw_archive": recovered_from_raw_archive,
        "raw_capture_set_sha256": _canonical_digest(raw_capture_digest_rows),
    }


def reconcile(args: argparse.Namespace) -> dict[str, Any]:
    protocol, protocol_sha256 = control.load_protocol(args.protocol)
    openrouter_n3.install_protocol_route_entries(protocol)
    lock = _read_json(args.locked_inputs_manifest)
    if lock.get("protocol_sha256") != protocol_sha256:
        raise ValueError("locked-input manifest protocol hash mismatch")
    locked_full = lock.get("full")
    if not isinstance(locked_full, dict):
        raise ValueError("locked-input manifest has no full profile")
    generated_manifest = _read_json(args.batches_dir / "disabled_format_control_manifest.json")
    review = _read_json(args.canary_review)
    passing_subjects = list(map(str, review.get("passing_subjects") or []))
    stopped_subjects = list(map(str, review.get("stopped_subjects") or []))
    if not passing_subjects or set(passing_subjects) & set(stopped_subjects):
        raise ValueError("invalid canary pass/stop populations")
    if set(passing_subjects) | set(stopped_subjects) != set(protocol["population"]["subjects"]):
        raise ValueError("canary review does not partition the frozen population")

    full_authorization = _read_json(args.full_execution_authorization)
    caps = full_authorization["spending_authorization"]["openrouter_full_subject_caps_usd"]
    recovered_subjects = set(args.recovered_subject)
    if not recovered_subjects.issubset(passing_subjects):
        raise ValueError("a recovered subject is not in the passing population")
    subject_results: dict[str, Any] = {}
    total_cost = 0.0
    for subject in passing_subjects:
        full_launcher.validate_full_execution_authorization(
            args.full_execution_authorization,
            protocol=protocol,
            protocol_path=args.protocol,
            protocol_sha256=protocol_sha256,
            locked_inputs_manifest_path=args.locked_inputs_manifest,
            canary_authorization_path=args.execution_authorization,
            subject_id=subject,
        )
        expected = control._load_generated_rows(
            args.batches_dir,
            subject,
            generated_manifest,
            locked_full,
        )
        control.validate_generated_rows(expected, protocol, canary=False)
        response_path = args.responses_dir / f"responses_{subject}_openrouter.jsonl"
        observed = _read_jsonl(response_path)
        result = reconcile_subject(
            expected_rows=expected,
            observed_rows=observed,
            descriptor=protocol["route_policy"]["routes"][subject],
            subject_id=subject,
            runtime_cost_cap_usd=float(caps[subject]),
            recovered_from_raw_archive=subject in recovered_subjects,
        )
        result["response_path"] = str(response_path)
        result["response_sha256"] = _sha256_file(response_path)
        subject_results[subject] = result
        total_cost += float(result["reported_cost_usd"])

    for subject in stopped_subjects:
        response_path = args.responses_dir / f"responses_{subject}_openrouter.jsonl"
        if response_path.exists() and response_path.stat().st_size:
            raise ValueError(f"stopped subject has a nonempty full response artifact: {subject}")
    expected_usage_after = round(
        float(review["spend_reconciliation"]["observed_usage_after_canary_usd"])
        + total_cost,
        8,
    )
    account_reconciliation: dict[str, Any] | None = None
    if args.openrouter_account_usage_usd is not None:
        delta = round(float(args.openrouter_account_usage_usd) - expected_usage_after, 8)
        account_reconciliation = {
            "expected_usage_after_full_usd": expected_usage_after,
            "observed_usage_after_full_usd": args.openrouter_account_usage_usd,
            "reconciliation_delta_usd": delta,
            "openrouter_limit_usd": args.openrouter_limit_usd,
        }
        if delta != 0:
            raise ValueError(f"OpenRouter account spend does not reconcile: delta={delta}")
    result = {
        "schema_version": RECONCILIATION_SCHEMA,
        "status": "complete_before_extraction_or_grading",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(args.protocol),
        "protocol_sha256": protocol_sha256,
        "locked_inputs_manifest": str(args.locked_inputs_manifest),
        "locked_inputs_manifest_sha256": _sha256_file(args.locked_inputs_manifest),
        "canary_review": str(args.canary_review),
        "canary_review_sha256": _sha256_file(args.canary_review),
        "full_execution_authorization": str(args.full_execution_authorization),
        "full_execution_authorization_sha256": _sha256_file(
            args.full_execution_authorization
        ),
        "passing_subjects": passing_subjects,
        "stopped_subjects": stopped_subjects,
        "authorized_full_rows": sum(item["expected_rows"] for item in subject_results.values()),
        "completed_full_rows": sum(item["observed_rows"] for item in subject_results.values()),
        "full_reported_cost_usd": round(total_cost, 8),
        "subjects": subject_results,
        "account_reconciliation": account_reconciliation,
        "downstream_status": "not_started",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=openrouter_n3.DEFAULT_PROTOCOL)
    parser.add_argument(
        "--locked-inputs-manifest",
        type=Path,
        default=openrouter_n3.DEFAULT_LOCKED_INPUTS,
    )
    parser.add_argument("--batches-dir", type=Path, required=True)
    parser.add_argument("--responses-dir", type=Path, default=DEFAULT_RESPONSES_DIR)
    parser.add_argument("--canary-review", type=Path, default=DEFAULT_CANARY_REVIEW)
    parser.add_argument(
        "--execution-authorization",
        type=Path,
        default=openrouter_n3.DEFAULT_EXECUTION_AUTHORIZATION,
    )
    parser.add_argument(
        "--full-execution-authorization",
        type=Path,
        default=full_launcher.DEFAULT_FULL_EXECUTION_AUTHORIZATION,
    )
    parser.add_argument("--recovered-subject", action="append", default=[])
    parser.add_argument("--openrouter-account-usage-usd", type=float)
    parser.add_argument("--openrouter-limit-usd", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = reconcile(args)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["completed_full_rows"],
                "cost_usd": result["full_reported_cost_usd"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
