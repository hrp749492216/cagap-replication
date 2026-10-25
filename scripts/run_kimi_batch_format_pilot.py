#!/usr/bin/env python3
"""Run an excluded Kimi K2.6 native-batch pilot for the three-arm control.

The pilot uses only the four frozen pilot prompts and is never eligible for
analysis.  Submission and retrieval are separate actions so a provider batch
can be polled safely without creating a duplicate job.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import validate as validate_json_schema


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cagap_env import load_cagap_env  # noqa: E402
from scripts import fetch_native_batch_results as native_fetch  # noqa: E402
from scripts.fetch_native_batch_results import NativeBatchResultFetcher  # noqa: E402
from scripts.run_synthetic_prompts import (  # noqa: E402
    Config,
    NativeBatchSubmitter,
    _native_headers,
)
from scripts.schema_visibility_preflight import (  # noqa: E402
    FORMAT_CONTROL_MODE,
    build_body,
    build_routes,
    canary_marker,
    classify,
    diagnostic_prompts,
    load_format_instruction,
    parse_openai,
)


PILOT_CONFIRMATION = "KIMI_NATIVE_BATCH_FORMAT_PILOT_2026-09-03"
PILOT_MODES = (
    "NOTOOLS",
    "DISABLED",
    FORMAT_CONTROL_MODE,
    "TOOL",
    "TOOL_IRRELEVANT",
    "CANARY_DISABLED",
)
EXPECTED_REQUESTS = 24
OUTPUT_READY_FINALIZING_MIN_SECONDS = 30 * 60
NATIVE_PRICING_PATH = REPO_ROOT / "config/api_pricings_native_providers.json"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")


def _protocol_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kimi_native_route() -> dict[str, Any]:
    routes = build_routes(REPO_ROOT / "config/roster.json")
    matches = [
        route
        for route in routes
        if route.get("subject_id") == "kimi-k2.6"
        and route.get("route_type") == "native"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Kimi native route, found {len(matches)}")
    route = copy.deepcopy(matches[0])
    if route.get("endpoint") != "https://api.moonshot.ai":
        raise ValueError("Kimi pilot requires the account-verified api.moonshot.ai endpoint")
    if route.get("slug") != "kimi-k2.6":
        raise ValueError("Kimi pilot requires exact model slug kimi-k2.6")
    return route


def build_pilot_rows(format_protocol: Path) -> list[dict[str, Any]]:
    """Build the 4-prompt × 6-mode diagnostic matrix without API access."""
    instruction = load_format_instruction(format_protocol)
    protocol_sha256 = _protocol_sha256(format_protocol)
    route = _kimi_native_route()
    rows: list[dict[str, Any]] = []
    for prompt in diagnostic_prompts():
        for mode in PILOT_MODES:
            body, tool_names = build_body(
                prompt,
                route,
                mode,
                format_instruction=instruction,
            )
            # Current Kimi Batch API documentation forbids sampling overrides
            # for kimi-k2.6.  The promoted route freezes this same omission.
            body.pop("temperature", None)
            body.pop("top_p", None)
            body["max_tokens"] = 4096
            custom_id = (
                "pilot__disabled_format_v3__kimi-k2.6__"
                f"{prompt['case_id']}__{mode}"
            )
            rows.append(
                {
                    "custom_id": custom_id,
                    "subject_id": "kimi-k2.6",
                    "prompt_id": prompt["case_id"],
                    "prompt_case_id": prompt["case_id"],
                    "domain": prompt["domain"],
                    "mode": mode,
                    "route": "native",
                    "route_label": "kimi-k2.6::native_batch",
                    "request_model": "kimi-k2.6",
                    "pinned_provider": "moonshot",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "analysis_inclusion": False,
                    "diagnostic_only": True,
                    "format_protocol_sha256": protocol_sha256,
                    "request_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
                    "tool_names": tool_names,
                    "body": body,
                }
            )
    if len(rows) != EXPECTED_REQUESTS:
        raise ValueError(f"expected {EXPECTED_REQUESTS} pilot rows, got {len(rows)}")
    if len({row["custom_id"] for row in rows}) != EXPECTED_REQUESTS:
        raise ValueError("Kimi pilot custom IDs are not unique")
    return rows


def _batch_rates() -> tuple[float, float]:
    payload = json.loads(NATIVE_PRICING_PATH.read_text(encoding="utf-8"))
    matches = [
        row
        for row in payload.get("pricing", [])
        if isinstance(row, dict) and row.get("model_key") == "kimi-k2.6"
    ]
    if len(matches) != 1:
        raise ValueError("current pricing overlay needs one Kimi K2.6 row")
    row = matches[0]
    return (
        float(row["batch_api_input_usd_per_1m_tokens"]),
        float(row["batch_api_output_usd_per_1m_tokens"]),
    )


def conservative_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    input_rate, output_rate = _batch_rates()
    input_upper = sum(len(_canonical_bytes(row["body"])) for row in rows)
    output_upper = sum(int(row["body"].get("max_tokens", 0)) for row in rows)
    total = (input_upper * input_rate + output_upper * output_rate) / 1_000_000
    return {
        "method": "UTF-8 bytes as input-token ceiling plus max_tokens output ceiling",
        "input_upper_bound": input_upper,
        "output_upper_bound": output_upper,
        "input_rate_usd_per_1m": input_rate,
        "output_rate_usd_per_1m": output_rate,
        "total_usd": round(total, 6),
    }


def prepare(output_dir: Path, format_protocol: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"pilot output directory is not empty: {output_dir}")
    rows = build_pilot_rows(format_protocol)
    batches_dir = output_dir / "batches"
    input_path = batches_dir / "input_kimi-k2.6.jsonl"
    _write_jsonl(input_path, rows)
    manifest = {
        "schema_version": "disabled_format_control.kimi_batch_pilot.v1",
        "status": "prepared_not_submitted",
        "diagnostic_only": True,
        "analysis_inclusion": False,
        "subject_id": "kimi-k2.6",
        "route": "native_batch",
        "endpoint": "https://api.moonshot.ai/v1/batches",
        "model": "kimi-k2.6",
        "format_protocol": str(format_protocol),
        "format_protocol_sha256": _protocol_sha256(format_protocol),
        "request_count": len(rows),
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "estimated_cost_upper_bound": conservative_cost(rows),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "pilot_input_manifest.json", manifest)
    return rows, manifest


def submit(
    output_dir: Path,
    format_protocol: Path,
    *,
    confirm_spend: str | None,
    max_requests: int | None,
    max_cost_usd: float | None,
) -> dict[str, Any]:
    if confirm_spend != PILOT_CONFIRMATION:
        raise ValueError(f"--confirm-spend must exactly equal {PILOT_CONFIRMATION}")
    if max_requests != EXPECTED_REQUESTS:
        raise ValueError(f"--max-requests must exactly equal {EXPECTED_REQUESTS}")
    if max_cost_usd is None or not math.isfinite(max_cost_usd) or max_cost_usd < 0:
        raise ValueError("--max-cost-usd must be finite and non-negative")
    rows, input_manifest = prepare(output_dir, format_protocol)
    estimate = float(input_manifest["estimated_cost_upper_bound"]["total_usd"])
    if estimate > max_cost_usd:
        raise ValueError(
            f"cost gate exceeded: conservative ${estimate:.6f} > ${max_cost_usd:.6f}"
        )
    entry = copy.deepcopy(Config.SUBJECTS["kimi-k2.6"])
    entry["route_override"] = "native"
    submitter = NativeBatchSubmitter(
        output_dir=output_dir / "responses",
        confirm_openrouter_fallback=lambda _subject, _reason: False,
    )
    result = submitter.submit_subject_rows("kimi-k2.6", entry, rows)
    manifest = {
        **result,
        "schema_version": "disabled_format_control.kimi_batch_pilot_submission.v1",
        "diagnostic_only": True,
        "analysis_inclusion": False,
        "request_count": len(rows),
        "input_sha256": input_manifest["input_sha256"],
        "format_protocol_sha256": input_manifest["format_protocol_sha256"],
        "max_cost_usd": max_cost_usd,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "batches" / "submit_manifest_kimi-k2.6.json"
    _write_json(path, manifest)
    return {**manifest, "manifest_path": str(path)}


def _tool_validation(
    request_body: dict[str, Any], tool_calls: Any
) -> tuple[int, int]:
    tools = {
        str(item.get("function", {}).get("name")): item.get("function", {}).get(
            "parameters"
        )
        for item in request_body.get("tools", [])
        if isinstance(item, dict) and isinstance(item.get("function"), dict)
    }
    emitted = 0
    valid = 0
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        emitted += 1
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        schema = tools.get(name)
        if not isinstance(arguments, dict) or not isinstance(schema, dict):
            continue
        try:
            validate_json_schema(arguments, schema)
        except Exception:
            continue
        valid += 1
    return emitted, valid


def evaluate(output_dir: Path) -> dict[str, Any]:
    response_path = output_dir / "responses" / "native_responses_kimi-k2.6.jsonl"
    if not response_path.is_file():
        raise FileNotFoundError(response_path)
    records = [
        json.loads(line)
        for line in response_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    tool_calls_emitted = 0
    strict_valid_tool_calls = 0
    disabled_tool_calls = 0
    format_tool_calls = 0
    disabled_empty_outputs = 0
    format_empty_outputs = 0
    format_action_json_outputs = 0
    for record in records:
        raw = record.get("raw_response") if isinstance(record.get("raw_response"), dict) else {}
        summary = parse_openai(raw)
        provider_response = record.get("response")
        response_meta = (
            provider_response.get("response")
            if isinstance(provider_response, dict)
            and isinstance(provider_response.get("response"), dict)
            else {}
        )
        status_code = int(response_meta.get("status_code", 200 if not record.get("error") else 500))
        mode = str(record.get("mode") or "")
        tool_calls = summary.get("tool_calls")
        emitted, valid = _tool_validation(record.get("body") or {}, tool_calls)
        if mode == "TOOL":
            tool_calls_emitted += emitted
            strict_valid_tool_calls += valid
        elif mode == "DISABLED":
            disabled_tool_calls += emitted
        elif mode == FORMAT_CONTROL_MODE:
            format_tool_calls += emitted
        content = summary.get("content")
        content_text = content if isinstance(content, str) else ""
        if mode == "DISABLED" and not content_text.strip() and not emitted:
            disabled_empty_outputs += 1
        if mode == FORMAT_CONTROL_MODE:
            if not content_text.strip() and not emitted:
                format_empty_outputs += 1
            if content_text.strip():
                try:
                    value = json.loads(content_text)
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, dict) and set(value) == {"name", "arguments"}:
                    format_action_json_outputs += 1
        row = {
            "subject_id": "kimi-k2.6",
            "route_label": "kimi-k2.6::native_batch",
            "prompt_case_id": record.get("prompt_case_id"),
            "mode": mode,
            "http_status": status_code,
            "marker_present": canary_marker() in content_text,
            **summary,
        }
        rows.append(row)

    classification = classify(rows)
    route_class = classification.get("kimi-k2.6::native_batch", {})
    visibility = route_class.get("disabled_arm_schema_visibility")
    http_200 = sum(row.get("http_status") == 200 for row in rows)
    row_errors = sum(bool(record.get("error")) for record in records)
    operational_pass = all(
        (
            len(records) == EXPECTED_REQUESTS,
            http_200 == EXPECTED_REQUESTS,
            row_errors == 0,
            visibility == "schema_visible",
            tool_calls_emitted > 0,
            strict_valid_tool_calls == tool_calls_emitted,
            disabled_tool_calls == 0,
            format_tool_calls == 0,
            disabled_empty_outputs == 0,
            format_empty_outputs == 0,
        )
    )
    input_tokens = sum(
        int((record.get("usage") or {}).get("prompt_tokens") or 0)
        for record in records
    )
    output_tokens = sum(
        int((record.get("usage") or {}).get("completion_tokens") or 0)
        for record in records
    )
    input_rate, output_rate = _batch_rates()
    returned_models = sorted(
        {
            str(row.get("returned_model"))
            for row in rows
            if isinstance(row.get("returned_model"), str)
            and str(row.get("returned_model")).strip()
        }
    )
    model_identity_verified = bool(returned_models) and all(
        model == "kimi-k2.6" or model.startswith("kimi-k2.6-")
        for model in returned_models
    )
    operational_pass = operational_pass and model_identity_verified
    estimated_actual_cost = (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000
    evaluation = {
        "schema_version": "disabled_format_control.kimi_batch_pilot_evaluation.v1",
        "status": "diagnostic_only_not_official_canary",
        "diagnostic_only": True,
        "analysis_inclusion": False,
        "subject_id": "kimi-k2.6",
        "route": "native_batch",
        "endpoint": "https://api.moonshot.ai/v1/batches",
        "requested_model": "kimi-k2.6",
        "returned_models": returned_models,
        "provider_pin_verified": model_identity_verified,
        "request_count": len(records),
        "http_200": http_200,
        "row_errors": row_errors,
        "disabled_schema_retention": visibility,
        "tool_native_call_support": (
            "supported" if tool_calls_emitted and strict_valid_tool_calls == tool_calls_emitted else "failed"
        ),
        "tool_mode_calls": tool_calls_emitted,
        "strict_schema_valid_tool_calls": strict_valid_tool_calls,
        "disabled_tool_calls": disabled_tool_calls,
        "format_tool_calls": format_tool_calls,
        "disabled_empty_outputs": disabled_empty_outputs,
        "format_empty_outputs": format_empty_outputs,
        "format_action_json_outputs": format_action_json_outputs,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_actual_cost_usd": round(estimated_actual_cost, 8),
        "operational_verdict": (
            "pass_for_future_official_canary_after_osf_authorization"
            if operational_pass
            else "pause_before_protocol_freeze"
        ),
        "provider_batch_id": records[0].get("provider_batch_id") if records else None,
        "responses_path": str(response_path),
        "responses_sha256": _sha256_file(response_path),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
    }
    _write_json(output_dir / "schema_visibility_classification.json", classification)
    _write_json(output_dir / "pilot_evaluation.json", evaluation)
    return evaluation


def validate_output_ready_finalizing(
    batch: dict[str, Any], *, now_epoch: float
) -> dict[str, Any]:
    """Validate Moonshot's diagnostic-only output-ready intermediate state."""
    if str(batch.get("status") or "").lower() != "finalizing":
        raise ValueError("output-ready exception requires provider status finalizing")
    counts = batch.get("request_counts")
    if not isinstance(counts, dict) or counts != {
        "completed": EXPECTED_REQUESTS,
        "failed": 0,
        "total": EXPECTED_REQUESTS,
    }:
        raise ValueError("output-ready exception requires exactly 24 completed / 0 failed")
    output_file_id = batch.get("output_file_id")
    if not isinstance(output_file_id, str) or not output_file_id.strip():
        raise ValueError("output-ready exception requires a provider output_file_id")
    finalizing_at = batch.get("finalizing_at")
    if (
        isinstance(finalizing_at, bool)
        or not isinstance(finalizing_at, (int, float))
        or float(finalizing_at) <= 0
    ):
        raise ValueError("output-ready exception requires finalizing_at")
    finalizing_age_seconds = float(now_epoch) - float(finalizing_at)
    if finalizing_age_seconds < OUTPUT_READY_FINALIZING_MIN_SECONDS:
        raise ValueError(
            "output-ready exception requires at least 30 minutes in finalizing"
        )
    if batch.get("error_file_id") not in (None, "") or batch.get("errors") not in (
        None,
        [],
    ):
        raise ValueError("output-ready exception rejects provider error artifacts")
    return {
        "diagnostic_only": True,
        "accepted_intermediate_status": "finalizing",
        "required_minimum_age_seconds": OUTPUT_READY_FINALIZING_MIN_SECONDS,
        "observed_age_seconds": int(finalizing_age_seconds),
        "request_counts": counts,
        "output_file_id": output_file_id,
        "official_canary_policy_unchanged": True,
    }


def fetch_and_evaluate(
    output_dir: Path,
    *,
    force: bool = False,
    accept_output_ready_finalizing: bool = False,
) -> dict[str, Any]:
    batches_dir = output_dir / "batches"
    response_dir = output_dir / "responses"
    manifest_path = batches_dir / "submit_manifest_kimi-k2.6.json"
    fetcher = NativeBatchResultFetcher(
        batches_dir=batches_dir,
        raw_dir=response_dir,
        force=force,
    )
    fetched = fetcher.fetch_manifest(manifest_path)
    output_ready_override = None
    if (
        fetched.get("status") == "not_complete"
        and accept_output_ready_finalizing
    ):
        route_cfg = Config.SUBJECTS["kimi-k2.6"]["native"]
        provider_batch_id = str(fetched["provider_batch_id"])
        batch = fetcher._get_json(
            f"{route_cfg['endpoint']}{route_cfg['batch_url_path']}/{provider_batch_id}",
            _native_headers(route_cfg, "openai"),
            provider="moonshot",
            model_id="kimi-k2.6",
        )
        output_ready_override = validate_output_ready_finalizing(
            batch, now_epoch=time.time()
        )
        native_fetch.OPENAI_COMPAT_COMPLETE_STATUSES.add("finalizing")
        try:
            fetched = fetcher.fetch_manifest(manifest_path)
        finally:
            native_fetch.OPENAI_COMPAT_COMPLETE_STATUSES.discard("finalizing")
        if fetched.get("status") != "completed":
            raise ValueError("output-ready diagnostic retrieval did not complete")
        fetched["diagnostic_output_ready_override"] = output_ready_override
        _write_json(response_dir / "fetch_manifest_kimi-k2.6.json", fetched)
    if fetched.get("status") not in {"completed", "skipped_existing"}:
        return {"fetch": fetched, "evaluation": None}
    evaluation = evaluate(output_dir)
    if output_ready_override is not None:
        evaluation["provider_status_at_retrieval"] = "finalizing"
        evaluation["diagnostic_output_ready_override"] = output_ready_override
        _write_json(output_dir / "pilot_evaluation.json", evaluation)
    return {"fetch": fetched, "evaluation": evaluation}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--fetch-evaluate", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--format-protocol",
        type=Path,
        default=REPO_ROOT
        / "experiments/disabled_format_control/protocol_v3_draft.json",
    )
    parser.add_argument("--confirm-spend", default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--accept-output-ready-finalizing",
        action="store_true",
        help=(
            "diagnostic pilot only: after 30 minutes, retrieve a 24/24, "
            "0-failure Moonshot output file while status remains finalizing"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_cagap_env()
    try:
        if args.prepare:
            _rows, result = prepare(args.output_dir, args.format_protocol)
        elif args.submit:
            result = submit(
                args.output_dir,
                args.format_protocol,
                confirm_spend=args.confirm_spend,
                max_requests=args.max_requests,
                max_cost_usd=args.max_cost_usd,
            )
        else:
            result = fetch_and_evaluate(
                args.output_dir,
                force=args.force,
                accept_output_ready_finalizing=args.accept_output_ready_finalizing,
            )
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
