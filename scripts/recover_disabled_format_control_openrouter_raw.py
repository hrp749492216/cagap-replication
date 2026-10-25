#!/usr/bin/env python3
"""Rebuild normalized control records from the restricted raw HTTP archive.

This is an incident-recovery path, not a dispatch path.  It performs no network
requests and refuses to emit an artifact unless every locked input row has one
and only one successful, hash-verified archived response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_disabled_format_control as control  # noqa: E402
from scripts import run_disabled_format_control_openrouter_n3 as openrouter_n3  # noqa: E402


RECOVERY_SCHEMA = "disabled_format_control.raw_http_artifact_recovery.v1"
LATENCY_UNAVAILABLE_SENTINEL_SECONDS = -1.0
DEFAULT_BATCHES_DIR = (
    REPO_ROOT / "outputs/postreg_disabled_format_control_v3_openrouter_n3_eligible/batches"
)
DEFAULT_RAW_MANIFEST = REPO_ROOT / "outputs/raw_http_archive/manifest.jsonl"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_archive_request_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def load_manifest_entries(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read one or more preserved manifest inodes and deduplicate exact rows."""
    unique: dict[str, tuple[dict[str, Any], set[str]]] = {}
    path_list = [Path(path).resolve() for path in paths]
    if not path_list:
        raise ValueError("at least one raw HTTP manifest path is required")
    for path in path_list:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid raw manifest JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(entry, dict):
                    raise ValueError(f"raw manifest row is not an object at {path}:{line_number}")
                key = _canonical_json(entry)
                if key not in unique:
                    unique[key] = (entry, set())
                unique[key][1].add(str(path))
    rows: list[dict[str, Any]] = []
    for entry, sources in unique.values():
        annotated = dict(entry)
        annotated["_recovery_manifest_sources"] = sorted(sources)
        rows.append(annotated)
    return rows


def _resolve_body_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("raw manifest entry has no body_path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _capture_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("raw manifest entry has no captured_at_utc")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid raw capture timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"raw capture timestamp has no timezone: {value!r}")
    return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metadata_snapshot(path: Path, *, model_slug: str, provider: str) -> dict[str, Any]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    matched = snapshot.get("matched_model") if isinstance(snapshot, dict) else None
    if not isinstance(matched, dict):
        raise ValueError("OpenRouter metadata snapshot has no matched_model")
    if snapshot.get("model_slug") != model_slug:
        raise ValueError("OpenRouter metadata snapshot model slug mismatch")
    if snapshot.get("pinned_provider") != provider:
        raise ValueError("OpenRouter metadata snapshot provider mismatch")
    if matched.get("id") != model_slug:
        raise ValueError("OpenRouter metadata snapshot matched model mismatch")
    try:
        recorded_path = str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        recorded_path = str(path.resolve())
    return {
        "path": recorded_path,
        "sha256": _sha256_file(path),
        "model_slug": model_slug,
        "matched_model_id": str(matched["id"]),
    }


def _successful_entries_by_custom_id(
    entries: Iterable[dict[str, Any]],
    *,
    expected_ids: set[str],
    subject_id: str,
    provider: str,
) -> dict[str, dict[str, Any]]:
    successes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        custom_id = str(entry.get("custom_id") or "")
        if custom_id not in expected_ids:
            continue
        if entry.get("schema_version") != "raw_http_archive.v1":
            raise ValueError(f"unsupported raw archive schema for {custom_id}")
        if entry.get("role") != "subject" or entry.get("model_id") != subject_id:
            raise ValueError(f"raw archive subject identity mismatch for {custom_id}")
        if entry.get("route") != "openrouter" or entry.get("provider") != provider:
            raise ValueError(f"raw archive route/provider mismatch for {custom_id}")
        if entry.get("http_status") == 200:
            successes[custom_id].append(entry)
    missing = sorted(expected_ids - set(successes))
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"missing successful raw responses for {len(missing)} rows: {preview}")
    selected: dict[str, dict[str, Any]] = {}
    for custom_id, candidates in successes.items():
        identities = {
            (
                str(entry.get("body_sha256") or ""),
                str(entry.get("request_body_sha256") or ""),
                str(entry.get("captured_at_utc") or ""),
            )
            for entry in candidates
        }
        if len(identities) != 1:
            raise ValueError(f"multiple distinct successful captures for {custom_id}")
        selected[custom_id] = candidates[0]
    return selected


def build_recovered_records(
    *,
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    subject_id: str,
    manifest_entries: list[dict[str, Any]],
    metadata_snapshot_path: Path,
    runtime_cost_cap_usd: float,
    incident_id: str,
    canonical_raw_manifest_path: str = "outputs/raw_http_archive/manifest.jsonl",
) -> list[dict[str, Any]]:
    """Build records in locked-input order from verified archived responses."""
    if not rows:
        raise ValueError("cannot recover an empty locked-input shard")
    custom_ids = [str(row.get("custom_id") or "") for row in rows]
    if not all(custom_ids) or len(custom_ids) != len(set(custom_ids)):
        raise ValueError("locked input rows have missing or duplicate custom IDs")
    descriptor = protocol["route_policy"]["routes"][subject_id]
    if descriptor.get("route") != "openrouter":
        raise ValueError(f"recovery requires a frozen OpenRouter route for {subject_id}")
    provider_only = list(descriptor.get("provider_only") or [])
    if len(provider_only) != 1:
        raise ValueError(f"recovery requires one exact provider pin for {subject_id}")
    provider = str(provider_only[0])
    route = {
        "slug": str(descriptor["model_slug"]),
        "pinned_provider": provider,
        "quantizations": list(descriptor.get("quantizations") or []),
        "decoding": descriptor["decoding"],
    }
    metadata = _metadata_snapshot(
        metadata_snapshot_path,
        model_slug=route["slug"],
        provider=provider,
    )
    successes = _successful_entries_by_custom_id(
        manifest_entries,
        expected_ids=set(custom_ids),
        subject_id=subject_id,
        provider=provider,
    )

    recovered: list[dict[str, Any]] = []
    cumulative_cost = 0.0
    for row, custom_id in zip(rows, custom_ids, strict=True):
        entry = successes[custom_id]
        request_body = control.build_control_openrouter_body(row, route)
        observed_request_sha = _raw_archive_request_sha256(request_body)
        if entry.get("request_body_sha256") != observed_request_sha:
            raise ValueError(f"archived request body hash mismatch for {custom_id}")

        body_path = _resolve_body_path(entry.get("body_path"))
        body_bytes = body_path.read_bytes()
        body_sha256 = hashlib.sha256(body_bytes).hexdigest()
        if body_sha256 != entry.get("body_sha256"):
            raise ValueError(f"archived response body hash mismatch for {custom_id}")
        if len(body_bytes) != entry.get("body_bytes"):
            raise ValueError(f"archived response body size mismatch for {custom_id}")
        try:
            response = json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"archived HTTP 200 body is not JSON for {custom_id}") from exc
        if not isinstance(response, dict) or response.get("error"):
            raise ValueError(f"archived HTTP 200 body is not a successful object for {custom_id}")

        capture = {
            "raw_http_archive_schema_version": str(entry["schema_version"]),
            "raw_http_body_path": str(entry["body_path"]),
            "raw_http_body_sha256": body_sha256,
            "raw_http_body_bytes": len(body_bytes),
            "raw_http_manifest_path": canonical_raw_manifest_path,
            "raw_http_captured_at_utc": str(entry["captured_at_utc"]),
        }
        record = control.build_control_openrouter_run_record(
            row=row,
            body=request_body,
            response=response,
            pinned_provider=provider,
            latency_sec=LATENCY_UNAVAILABLE_SENTINEL_SECONDS,
            response_headers={},
            model_metadata_snapshot=metadata,
            raw_http_capture=capture,
        )
        response_cost = control._openrouter_reported_cost(response, custom_id=custom_id)
        cumulative_cost += response_cost
        if not math.isfinite(cumulative_cost) or cumulative_cost > runtime_cost_cap_usd + 1e-9:
            raise ValueError(f"recovered cost exceeds the runtime cap at {custom_id}")
        record["timestamp"] = _capture_timestamp(entry["captured_at_utc"])
        record["execution_budget"] = {
            "response_reported_cost_usd": response_cost,
            "cumulative_reported_cost_usd": round(cumulative_cost, 8),
            "runtime_cost_cap_usd": runtime_cost_cap_usd,
        }
        record["artifact_recovery"] = {
            "schema_version": RECOVERY_SCHEMA,
            "incident_id": incident_id,
            "provider_request_reissued": False,
            "recovered_from": "byte_for_byte_raw_http_response_archive",
            "request_body_sha256_verified": observed_request_sha,
            "response_body_sha256_verified": body_sha256,
            "raw_manifest_sources": entry.get("_recovery_manifest_sources", []),
            "latency_sec_status": "unavailable_after_external_atomic_pathname_replacement",
            "latency_sec_sentinel": LATENCY_UNAVAILABLE_SENTINEL_SECONDS,
            "response_headers_status": "not_retained_by_restricted_raw_archive_policy",
        }
        recovered.append(record)
    return recovered


def write_records_atomic(
    path: Path,
    records: list[dict[str, Any]],
    *,
    replace_empty_artifact: bool = False,
) -> tuple[str, Path | None]:
    """Durably write records; preserve an explicitly replaced empty artifact."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path: Path | None = None
    if path.exists():
        if not replace_empty_artifact or path.stat().st_size != 0:
            raise FileExistsError(path)
        evidence_path = path.with_name(path.name + ".external-replacement-empty-evidence")
        if evidence_path.exists():
            raise FileExistsError(evidence_path)
    temporary = path.with_name(path.name + f".tmp.recovery.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        if evidence_path is not None:
            path.replace(evidence_path)
        temporary.replace(path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_file(path), evidence_path


def recover(args: argparse.Namespace) -> dict[str, Any]:
    protocol, protocol_sha256 = control.load_protocol(args.protocol)
    openrouter_n3.install_protocol_route_entries(protocol)
    locked = json.loads(args.locked_inputs_manifest.read_text(encoding="utf-8"))
    if locked.get("protocol_sha256") != protocol_sha256:
        raise ValueError("locked-input manifest protocol hash mismatch")
    full_lock = locked.get("full")
    if not isinstance(full_lock, dict):
        raise ValueError("locked-input manifest has no full profile")
    generated_manifest = json.loads(
        (args.batches_dir / "disabled_format_control_manifest.json").read_text(encoding="utf-8")
    )
    rows = control._load_generated_rows(
        args.batches_dir,
        args.subject,
        generated_manifest,
        full_lock,
    )
    control.validate_generated_rows(rows, protocol, canary=False)
    entries = load_manifest_entries(args.raw_manifest or [DEFAULT_RAW_MANIFEST])
    records = build_recovered_records(
        rows=rows,
        protocol=protocol,
        subject_id=args.subject,
        manifest_entries=entries,
        metadata_snapshot_path=args.model_metadata_snapshot,
        runtime_cost_cap_usd=args.runtime_cost_cap_usd,
        incident_id=args.incident_id,
        canonical_raw_manifest_path=args.canonical_raw_manifest_path,
    )
    artifact_sha256, evidence_path = write_records_atomic(
        args.output,
        records,
        replace_empty_artifact=args.replace_empty_artifact,
    )
    return {
        "schema_version": RECOVERY_SCHEMA,
        "status": "recovered_without_provider_redispatch",
        "incident_id": args.incident_id,
        "subject_id": args.subject,
        "protocol_sha256": protocol_sha256,
        "rows": len(records),
        "reported_cost_usd": records[-1]["execution_budget"][
            "cumulative_reported_cost_usd"
        ],
        "output_path": str(args.output.resolve()),
        "output_sha256": artifact_sha256,
        "preserved_empty_evidence_path": (
            str(evidence_path) if evidence_path is not None else None
        ),
        "provider_requests_made": 0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=openrouter_n3.DEFAULT_PROTOCOL)
    parser.add_argument(
        "--locked-inputs-manifest",
        type=Path,
        default=openrouter_n3.DEFAULT_LOCKED_INPUTS,
    )
    parser.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES_DIR)
    parser.add_argument("--raw-manifest", type=Path, action="append", default=[])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--model-metadata-snapshot", type=Path, required=True)
    parser.add_argument("--runtime-cost-cap-usd", type=float, required=True)
    parser.add_argument("--incident-id", required=True)
    parser.add_argument(
        "--canonical-raw-manifest-path",
        default="outputs/raw_http_archive/manifest.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-empty-artifact", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = recover(parse_args(argv))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
