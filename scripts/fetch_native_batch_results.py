"""Fetch completed native provider batch results into local raw JSONL.

This script closes the storage gap between native batch submission manifests
and the grader. It reads ``outputs/batches/submit_manifest_<subject>.json``,
downloads completed provider result files, enriches each result row with the
original request metadata from ``input_<subject>.jsonl``, and writes
``outputs/raw/native_responses_<subject>.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from urllib.parse import urlencode
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from grader.json_strict import strict_json_loads
from run_synthetic_prompts import (  # noqa: E402
    ANTHROPIC_MESSAGES_BATCH_SHAPE,
    OPENAI_COMPAT_BATCH_SHAPE,
    RUN_RECORD_SCHEMA_VERSION,
    XAI_HTTP_BATCH_SHAPE,
    Config,
    _native_headers,
    _run_record_timestamp,
    native_batch_request_shape,
    passive_reasoning_telemetry,
    redact_reasoning_fields,
)
from scripts.runtime_provenance import build_api_runtime_stamp, safe_response_headers  # noqa: E402
from scripts.raw_http_archive import archive_response_body, response_body_bytes  # noqa: E402
from providers import get_adapter  # noqa: E402


OPENAI_COMPAT_COMPLETE_STATUSES = {"completed", "succeeded"}
ANTHROPIC_COMPLETE_STATUSES = {"ended", "completed", "succeeded"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _xai_api_key(route_cfg: dict[str, Any]) -> str:
    return (
        os.environ.get(route_cfg.get("auth_env", ""), "")
        or os.environ.get("XAI_API_KEY", "")
    )


def _xai_headers(route_cfg: dict[str, Any]) -> dict[str, str]:
    key = _xai_api_key(route_cfg)
    if not key:
        raise RuntimeError(
            f"{route_cfg.get('auth_env', 'X_API_KEY')} or XAI_API_KEY is required for xAI native batch"
        )
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _strict_json_object(text: str, *, path: Path | str) -> dict[str, Any]:
    parsed = strict_json_loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return parsed


def _submission_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Return provider fields from flat, pilot, or legacy control wrappers.

    ``run_synthetic_prompts`` writes a flat native-batch manifest, while
    ``run_pilot`` stores that same manifest under a top-level ``manifest`` key
    next to pilot accounting fields. The fetcher only needs the provider
    submission fields, so accept all three shapes.
    """
    if payload.get("final_route") is not None or payload.get("provider_batch_id") is not None:
        return payload
    nested = payload.get("manifest")
    if not isinstance(nested, dict):
        nested = payload.get("provider_result")
    if isinstance(nested, dict):
        manifest = dict(nested)
        if payload.get("subject_id") is not None:
            manifest.setdefault("subject_id", payload.get("subject_id"))
        if payload.get("route") is not None:
            manifest.setdefault("final_route", payload.get("route"))
        return manifest
    return payload


def _iter_jsonl_text(text: str, *, path: Path | str) -> Iterable[dict[str, Any]]:
    import json as _json

    decoder = _json.JSONDecoder(strict=False)
    idx = 0
    row_no = 0
    length = len(text)
    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            parsed, idx = decoder.raw_decode(text, idx)
        except _json.JSONDecodeError as exc:
            snippet = text[max(0, idx - 80): idx + 240]
            raise ValueError(
                f"{path}: could not decode JSON object at char {idx}: {exc}; "
                f"context={snippet!r}"
            ) from exc
        row_no += 1
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}: decoded row {row_no} must be an object.")
        yield parsed


def _iter_jsonl_file(path: Path) -> Iterable[dict[str, Any]]:
    return _iter_jsonl_text(path.read_text(encoding="utf-8"), path=path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(obj: Any) -> str:
    payload = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _request_metadata_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        key: row.get(key)
        for key in (
            "custom_id",
            "subject_id",
            "route",
            "prompt_id",
            "mode",
            "persona",
            "body",
            "hashes",
            "serialized_request_hash",
            "request_prompt",
        )
        if key in row
    }
    return _sha256_json(payload)


def _manifest_input_file_id(manifest: dict[str, Any]) -> str | None:
    for key in ("input_file_id", "provider_input_file_id", "batch_input_file_id"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _batch_input_file_ids(batch: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("input_file_id", "input_file"):
        value = batch.get(key)
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    for key in ("input_file_ids", "input_files"):
        value = batch.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    ids.add(item.strip())
                elif isinstance(item, dict):
                    nested = item.get("id") or item.get("file_id")
                    if isinstance(nested, str) and nested.strip():
                        ids.add(nested.strip())
    return ids


def _validate_batch_input_file_id(
    *,
    manifest: dict[str, Any],
    batch: dict[str, Any],
    provider_batch_id: str,
) -> None:
    expected = _manifest_input_file_id(manifest)
    if expected is None:
        return
    actual_ids = _batch_input_file_ids(batch)
    if actual_ids and expected not in actual_ids:
        raise ValueError(
            f"provider batch {provider_batch_id} input_file_id mismatch: "
            f"manifest expected {expected!r}, provider returned {sorted(actual_ids)!r}"
        )


def _row_error_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("error"))


def _summary_status_for_rows(rows: list[dict[str, Any]]) -> str:
    return "completed_with_errors" if _row_error_count(rows) else "completed"


def _fsync_parent(path: Path) -> None:
    try:
        fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    _fsync_parent(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    _fsync_parent(path)


def _subject_route(subject_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    entry = Config.SUBJECTS.get(subject_id)
    if not isinstance(entry, dict):
        raise ValueError(f"unknown subject_id in manifest: {subject_id!r}")
    route_cfg = entry.get("native")
    if not isinstance(route_cfg, dict):
        raise ValueError(f"subject {subject_id!r} has no native route in roster.")
    shape = native_batch_request_shape(entry, route_cfg)
    if shape is None:
        raise ValueError(f"subject {subject_id!r} does not have an implemented native batch shape.")
    return entry, route_cfg, shape


def _request_rows_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl_file(path):
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id.strip():
            raise ValueError(f"{path}: row missing non-empty custom_id.")
        if custom_id in rows:
            raise ValueError(f"{path}: duplicate custom_id {custom_id!r}.")
        rows[custom_id] = row
    return rows


def _response_text(response: requests.Response) -> str:
    content = getattr(response, "content", b"")
    if isinstance(content, bytes) and content:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            pass
    text = getattr(response, "text", "")
    if isinstance(text, str) and text:
        return text
    if isinstance(content, bytes):
        return content.decode("utf-8")
    return str(content)


def _body_from_openai_compatible_result(row: dict[str, Any]) -> tuple[dict[str, Any], str | None, str]:
    response = row.get("response")
    if not isinstance(response, dict):
        return {}, "missing response object in provider batch result", "missing_response"
    status_code = response.get("status_code")
    body = response.get("body")
    if isinstance(body, dict) and (status_code is None or int(status_code) < 400):
        return body, None, "succeeded"
    error = row.get("error") or response.get("error") or f"provider status_code={status_code}"
    return {}, str(error), "errored"


def _body_from_anthropic_result(
    row: dict[str, Any],
    *,
    entry: dict[str, Any],
) -> tuple[dict[str, Any], str | None, str]:
    result = row.get("result")
    if not isinstance(result, dict):
        return {}, "missing result object in Anthropic batch result", "missing_result"
    result_type = str(result.get("type") or "")
    if result_type == "succeeded" and isinstance(result.get("message"), dict):
        adapter = get_adapter(entry.get("family"))
        message = result["message"]
        if adapter is not None and hasattr(adapter, "from_native_response"):
            message = adapter.from_native_response(message)
        return message, None, "succeeded"
    error = result.get("error") or result
    return {}, json.dumps(error, sort_keys=True, ensure_ascii=False, default=str), result_type or "errored"


def _provider_row_from_xai_http_result(
    item: dict[str, Any],
    *,
    short_to_full: dict[str, str],
) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"custom_id": "", "error": f"non-object xAI result row: {type(item).__name__}"}
    short_id = str(
        item.get("custom_id")
        or item.get("batch_request_id")
        or item.get("id")
        or ""
    )
    custom_id = short_to_full.get(short_id, short_id)
    row = dict(item)
    row["custom_id"] = custom_id
    row["xai_batch_request_id"] = short_id
    return row


def _body_from_xai_http_result(row: dict[str, Any]) -> tuple[dict[str, Any], str | None, str]:
    if row.get("error"):
        return {}, str(row["error"]), "errored"
    response = row.get("response")
    if isinstance(response, dict):
        if "body" in response or "status_code" in response:
            return _body_from_openai_compatible_result(row)
        if "choices" in response:
            return response, None, "succeeded"
        chat_completion = response.get("chat_get_completion")
        if isinstance(chat_completion, dict) and "choices" in chat_completion:
            return chat_completion, None, "succeeded"
    batch_result = row.get("batch_result")
    if isinstance(batch_result, dict):
        batch_response = batch_result.get("response")
        if isinstance(batch_response, dict):
            chat_completion = batch_response.get("chat_get_completion")
            if isinstance(chat_completion, dict) and "choices" in chat_completion:
                return chat_completion, None, "succeeded"
    body = row.get("body")
    if isinstance(body, dict) and "choices" in body:
        return body, None, "succeeded"
    result = row.get("result")
    if isinstance(result, dict) and result.get("error"):
        return {}, json.dumps(result["error"], sort_keys=True, ensure_ascii=False, default=str), "errored"
    return {}, "missing response object in xAI HTTP batch result", "missing_response"


def _native_run_record(
    *,
    request_row: dict[str, Any],
    provider_result: dict[str, Any],
    raw_response: dict[str, Any],
    error: str | None,
    result_status: str,
    subject_id: str,
    provider: str,
    provider_batch_id: str,
) -> dict[str, Any]:
    redacted_raw = redact_reasoning_fields(raw_response) if isinstance(raw_response, dict) else {}
    redacted_result = redact_reasoning_fields(provider_result)
    if isinstance(redacted_result, dict):
        redacted_response = redacted_result.get("response")
        if isinstance(redacted_response, dict) and isinstance(redacted_response.get("headers"), dict):
            redacted_response["headers"] = safe_response_headers(redacted_response["headers"])
    custom_id = str(request_row["custom_id"])
    provider_response = provider_result.get("response") if isinstance(provider_result, dict) else {}
    response_headers = provider_response.get("headers") if isinstance(provider_response, dict) else {}
    body = request_row.get("body") if isinstance(request_row.get("body"), dict) else {}
    runtime_stamp = build_api_runtime_stamp(
        requested_model=str(body.get("model") or ""),
        route="native",
        provider=provider,
        response=redacted_raw,
        response_headers=response_headers,
        provider_result=provider_result,
        endpoint=None,
        role="subject",
        model_id=subject_id,
        provider_batch_id=provider_batch_id,
        http_status=provider_response.get("status_code") if isinstance(provider_response, dict) else None,
    )
    record = dict(request_row)
    record.update(
        {
            "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL, custom_id)),
            "timestamp": _run_record_timestamp(),
            "schema_version": RUN_RECORD_SCHEMA_VERSION,
            "subject_id": subject_id,
            "custom_id": custom_id,
            "route": "native",
            "provider": provider,
            "provider_batch_id": provider_batch_id,
            "native_result_status": result_status,
            "native_result_fetched_at": _now_utc(),
            "response": redacted_result,
            "raw_response": redacted_raw,
            "usage": redacted_raw.get("usage", {}) if isinstance(redacted_raw, dict) else {},
            "runtime_stamp": runtime_stamp,
            "passive_reasoning_telemetry": passive_reasoning_telemetry(raw_response),
        }
    )
    if error:
        record["error"] = error
    return record


class NativeBatchResultFetcher:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        batches_dir: Path = Config.BATCHES_DIR,
        raw_dir: Path = Path("outputs/raw"),
        force: bool = False,
    ) -> None:
        self.session = session or requests.Session()
        self.batches_dir = Path(batches_dir)
        self.raw_dir = Path(raw_dir)
        self.force = force

    def _get_json(
        self,
        url: str,
        headers: dict[str, str],
        *,
        provider: str = "native",
        model_id: str | None = None,
    ) -> dict[str, Any]:
        response = self.session.get(url, headers=headers, timeout=60)
        archive_response_body(
            response_body_bytes(response),
            role="subject_batch_fetch",
            model_id=model_id,
            route="native",
            provider=provider,
            http_status=response.status_code,
            url=url,
            source="requests.Response.content",
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"{url} returned non-object JSON.")
        return data

    def _get_text(
        self,
        url: str,
        headers: dict[str, str],
        *,
        provider: str = "native",
        model_id: str | None = None,
    ) -> str:
        response = self.session.get(url, headers=headers, timeout=120)
        archive_response_body(
            response_body_bytes(response),
            role="subject_batch_fetch",
            model_id=model_id,
            route="native",
            provider=provider,
            http_status=response.status_code,
            url=url,
            source="requests.Response.content",
        )
        response.raise_for_status()
        return _response_text(response)

    def fetch_manifest(self, manifest_path: Path) -> dict[str, Any]:
        payload = _strict_json_object(manifest_path.read_text(encoding="utf-8"), path=manifest_path)
        manifest = _submission_manifest(payload)
        subject_id = str(manifest.get("subject_id") or manifest_path.stem.removeprefix("submit_manifest_"))
        if manifest.get("final_route") != "native":
            return {
                "subject_id": subject_id,
                "status": "skipped_non_native",
                "manifest_path": str(manifest_path),
            }
        provider_batch_id = str(manifest.get("provider_batch_id") or "")
        if not provider_batch_id:
            raise ValueError(f"{manifest_path}: missing provider_batch_id.")

        entry, route_cfg, shape = _subject_route(subject_id)
        if shape == XAI_HTTP_BATCH_SHAPE:
            headers = _xai_headers(route_cfg)
        else:
            headers = _native_headers(route_cfg, "anthropic" if shape == ANTHROPIC_MESSAGES_BATCH_SHAPE else "openai")
        response_path = self.raw_dir / f"native_responses_{subject_id}.jsonl"
        fetch_manifest_path = self.raw_dir / f"fetch_manifest_{subject_id}.json"
        if response_path.exists() and not self.force:
            self._validate_existing_response_file(
                subject_id=subject_id,
                provider_batch_id=provider_batch_id,
                response_path=response_path,
            )
            existing_rows = list(_iter_jsonl_file(response_path))
            summary = {
                "subject_id": subject_id,
                "status": "skipped_existing",
                "manifest_path": str(manifest_path),
                "response_path": str(response_path),
                "rows_written": len(existing_rows),
                "rows_error": _row_error_count(existing_rows),
                "response_sha256": _sha256_file(response_path),
            }
            _write_json_atomic(fetch_manifest_path, summary)
            return summary

        if shape == OPENAI_COMPAT_BATCH_SHAPE:
            summary = self._fetch_openai_compatible(
                subject_id=subject_id,
                entry=entry,
                manifest=manifest,
                route_cfg=route_cfg,
                headers=headers,
                provider_batch_id=provider_batch_id,
                manifest_path=manifest_path,
                response_path=response_path,
            )
        elif shape == ANTHROPIC_MESSAGES_BATCH_SHAPE:
            summary = self._fetch_anthropic(
                subject_id=subject_id,
                entry=entry,
                route_cfg=route_cfg,
                headers=headers,
                provider_batch_id=provider_batch_id,
                manifest_path=manifest_path,
                response_path=response_path,
            )
        elif shape == XAI_HTTP_BATCH_SHAPE:
            summary = self._fetch_xai_http(
                subject_id=subject_id,
                entry=entry,
                route_cfg=route_cfg,
                manifest=manifest,
                headers=headers,
                provider_batch_id=provider_batch_id,
                manifest_path=manifest_path,
                response_path=response_path,
            )
        else:
            raise ValueError(f"unsupported native batch result shape {shape!r}")
        _write_json_atomic(fetch_manifest_path, summary)
        return summary

    def _validate_existing_response_file(
        self,
        *,
        subject_id: str,
        provider_batch_id: str,
        response_path: Path,
    ) -> None:
        request_path = self.batches_dir / f"input_{subject_id}.jsonl"
        request_rows = _request_rows_by_custom_id(request_path)
        expected_ids = set(request_rows)
        seen: set[str] = set()
        for row_no, row in enumerate(_iter_jsonl_file(response_path), start=1):
            custom_id = row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValueError(f"{response_path}: row {row_no} missing non-empty custom_id")
            if custom_id in seen:
                raise ValueError(f"{response_path}: duplicate custom_id {custom_id!r}")
            seen.add(custom_id)
            if row.get("provider_batch_id") != provider_batch_id:
                raise ValueError(
                    f"{response_path}: row {row_no} provider_batch_id "
                    f"{row.get('provider_batch_id')!r} does not match manifest "
                    f"{provider_batch_id!r}"
                )
            request_row = request_rows.get(custom_id)
            if request_row is None:
                continue
            if _request_metadata_fingerprint(row) != _request_metadata_fingerprint(request_row):
                raise ValueError(
                    f"{response_path}: row {row_no} request metadata for "
                    f"custom_id {custom_id!r} does not match {request_path}"
                )
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        if missing or extra:
            raise ValueError(
                f"{response_path}: existing response custom_id set does not match "
                f"{request_path}; missing={len(missing)} extra={len(extra)}"
            )

    def _fetch_openai_compatible(
        self,
        *,
        subject_id: str,
        entry: dict[str, Any],
        manifest: dict[str, Any],
        route_cfg: dict[str, Any],
        headers: dict[str, str],
        provider_batch_id: str,
        manifest_path: Path,
        response_path: Path,
    ) -> dict[str, Any]:
        batch_url = f"{route_cfg['endpoint']}{route_cfg['batch_url_path']}/{provider_batch_id}"
        batch = self._get_json(batch_url, headers)
        _validate_batch_input_file_id(
            manifest=manifest,
            batch=batch,
            provider_batch_id=provider_batch_id,
        )
        status = str(batch.get("status") or "").lower()
        if status not in OPENAI_COMPAT_COMPLETE_STATUSES:
            return {
                "subject_id": subject_id,
                "status": "not_complete",
                "provider_status": status,
                "provider_batch_id": provider_batch_id,
                "manifest_path": str(manifest_path),
            }
        output_file_id = batch.get("output_file_id") or batch.get("result_file_id")
        if not isinstance(output_file_id, str) or not output_file_id.strip():
            raise ValueError(f"completed native batch {provider_batch_id} has no output_file_id.")
        files_path = route_cfg.get("batch_input_files_path", "/v1/files")
        content_url = f"{route_cfg['endpoint']}{files_path}/{output_file_id}/content"
        rows = list(
            self._enriched_rows(
                subject_id=subject_id,
                provider="openai_compatible",
                provider_batch_id=provider_batch_id,
                entry=entry,
                provider_rows=_iter_jsonl_text(self._get_text(content_url, headers), path=content_url),
                body_extractor=lambda row: _body_from_openai_compatible_result(row),
            )
        )
        rows_error = _row_error_count(rows)
        _write_jsonl_atomic(response_path, rows)
        return {
            "subject_id": subject_id,
            "status": _summary_status_for_rows(rows),
            "provider_status": status,
            "provider_batch_id": provider_batch_id,
            "output_file_id": output_file_id,
            "manifest_path": str(manifest_path),
            "response_path": str(response_path),
            "rows_written": len(rows),
            "rows_error": rows_error,
            "response_sha256": _sha256_file(response_path),
        }

    def _fetch_anthropic(
        self,
        *,
        subject_id: str,
        entry: dict[str, Any],
        route_cfg: dict[str, Any],
        headers: dict[str, str],
        provider_batch_id: str,
        manifest_path: Path,
        response_path: Path,
    ) -> dict[str, Any]:
        batch_url = f"{route_cfg['endpoint']}{route_cfg['batch_url_path']}/{provider_batch_id}"
        batch = self._get_json(batch_url, headers)
        status = str(batch.get("processing_status") or batch.get("status") or "").lower()
        if status not in ANTHROPIC_COMPLETE_STATUSES:
            return {
                "subject_id": subject_id,
                "status": "not_complete",
                "provider_status": status,
                "provider_batch_id": provider_batch_id,
                "manifest_path": str(manifest_path),
            }
        results_url = f"{route_cfg['endpoint']}{route_cfg['batch_url_path']}/{provider_batch_id}/results"
        rows = list(
            self._enriched_rows(
                subject_id=subject_id,
                provider="anthropic",
                provider_batch_id=provider_batch_id,
                entry=entry,
                provider_rows=_iter_jsonl_text(self._get_text(results_url, headers), path=results_url),
                body_extractor=lambda row: _body_from_anthropic_result(row, entry=entry),
            )
        )
        rows_error = _row_error_count(rows)
        _write_jsonl_atomic(response_path, rows)
        return {
            "subject_id": subject_id,
            "status": _summary_status_for_rows(rows),
            "provider_status": status,
            "provider_batch_id": provider_batch_id,
            "manifest_path": str(manifest_path),
            "response_path": str(response_path),
            "rows_written": len(rows),
            "rows_error": rows_error,
            "response_sha256": _sha256_file(response_path),
        }

    def _fetch_xai_http(
        self,
        *,
        subject_id: str,
        entry: dict[str, Any],
        route_cfg: dict[str, Any],
        manifest: dict[str, Any],
        headers: dict[str, str],
        provider_batch_id: str,
        manifest_path: Path,
        response_path: Path,
    ) -> dict[str, Any]:
        batch_url = f"{route_cfg['endpoint']}{route_cfg['batch_url_path']}/{provider_batch_id}"
        batch = self._get_json(batch_url, headers, provider="xai", model_id=subject_id)
        state = batch.get("state") if isinstance(batch.get("state"), dict) else {}
        n_pending = int(state.get("num_pending", -1) if isinstance(state, dict) else -1)
        n_total = int(state.get("num_requests", 0) if isinstance(state, dict) else 0)
        n_success = int(state.get("num_success", 0) if isinstance(state, dict) else 0)
        n_error = int(state.get("num_error", 0) if isinstance(state, dict) else 0)
        n_cancelled = int(state.get("num_cancelled", 0) if isinstance(state, dict) else 0)
        provider_status = (
            f"pending={n_pending};total={n_total};success={n_success};"
            f"error={n_error};cancelled={n_cancelled}"
        )
        if not (n_total > 0 and n_pending == 0):
            return {
                "subject_id": subject_id,
                "status": "not_complete",
                "provider_status": provider_status,
                "provider_batch_id": provider_batch_id,
                "manifest_path": str(manifest_path),
            }

        short_to_full = manifest.get("xai_short_to_full_idmap") or {}
        if not isinstance(short_to_full, dict):
            short_to_full = {}
        provider_rows: list[dict[str, Any]] = []
        pagination_token: str | None = None
        while True:
            query_args: dict[str, str] = {"limit": "1000"}
            if pagination_token:
                query_args["pagination_token"] = pagination_token
            result_page = self._get_json(
                f"{route_cfg['endpoint']}{route_cfg['batch_url_path']}/{provider_batch_id}/results?{urlencode(query_args)}",
                headers,
                provider="xai",
                model_id=subject_id,
            )
            items = result_page.get("results") or result_page.get("data") or []
            if not isinstance(items, list):
                raise ValueError(f"xAI batch results returned non-list results: {type(items).__name__}")
            provider_rows.extend(
                _provider_row_from_xai_http_result(item, short_to_full=short_to_full)
                for item in items
            )
            pagination_token = result_page.get("pagination_token") or result_page.get("next_page_token") or None
            if not pagination_token:
                break

        rows = list(
            self._enriched_rows(
                subject_id=subject_id,
                provider="xai",
                provider_batch_id=provider_batch_id,
                entry=entry,
                provider_rows=provider_rows,
                body_extractor=lambda row: _body_from_xai_http_result(row),
            )
        )
        rows_error = _row_error_count(rows)
        _write_jsonl_atomic(response_path, rows)
        return {
            "subject_id": subject_id,
            "status": "completed" if n_error == 0 and n_cancelled == 0 else "completed_with_errors",
            "provider_status": provider_status,
            "provider_batch_id": provider_batch_id,
            "manifest_path": str(manifest_path),
            "response_path": str(response_path),
            "rows_written": len(rows),
            "rows_error": rows_error,
            "response_sha256": _sha256_file(response_path),
        }

    def _enriched_rows(
        self,
        *,
        subject_id: str,
        provider: str,
        provider_batch_id: str,
        entry: dict[str, Any],
        provider_rows: Iterable[dict[str, Any]],
        body_extractor,
    ) -> Iterable[dict[str, Any]]:
        del entry  # Reserved for future provider-specific extractors.
        request_path = self.batches_dir / f"input_{subject_id}.jsonl"
        request_rows = _request_rows_by_custom_id(request_path)
        seen: set[str] = set()
        for provider_row in provider_rows:
            custom_id = provider_row.get("custom_id")
            if not isinstance(custom_id, str) or not custom_id.strip():
                raise ValueError(f"provider result for {subject_id} missing custom_id.")
            if custom_id in seen:
                raise ValueError(f"provider result for {subject_id} duplicated custom_id {custom_id!r}.")
            seen.add(custom_id)
            request_row = request_rows.get(custom_id)
            if request_row is None:
                raise ValueError(f"provider result custom_id {custom_id!r} not found in {request_path}.")
            raw_response, error, result_status = body_extractor(provider_row)
            yield _native_run_record(
                request_row=request_row,
                provider_result=provider_row,
                raw_response=raw_response,
                error=error,
                result_status=result_status,
                subject_id=subject_id,
                provider=provider,
                provider_batch_id=provider_batch_id,
            )
        missing = sorted(set(request_rows) - seen)
        if missing:
            raise ValueError(
                f"missing provider results for {len(missing)} request custom_id(s) "
                f"from {request_path}: {missing[:5]}"
            )


def _discover_manifests(batches_dir: Path, subjects: list[str] | None) -> list[Path]:
    if subjects:
        return [batches_dir / f"submit_manifest_{subject_id}.json" for subject_id in subjects]
    return sorted(batches_dir.glob("submit_manifest_*.json"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-dir", type=Path, default=Config.BATCHES_DIR)
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/raw"))
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--manifest", type=Path, action="append", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing native response JSONL.")
    parser.add_argument(
        "--allow-completed-with-errors",
        action="store_true",
        help=(
            "Return 0 for completed batches that contain provider row errors. "
            "Confirmatory subject fetches should leave this unset so row-level "
            "provider failures are visible."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifests = args.manifest or _discover_manifests(args.batches_dir, args.subject)
    if not manifests:
        print(f"No submit manifests found in {args.batches_dir}.")
        return 2
    fetcher = NativeBatchResultFetcher(
        batches_dir=args.batches_dir,
        raw_dir=args.raw_dir,
        force=args.force,
    )
    summaries = [fetcher.fetch_manifest(path) for path in manifests]
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False))
    incomplete = [s for s in summaries if s.get("status") == "not_complete"]
    errored = [
        s for s in summaries
        if s.get("status") == "completed_with_errors"
        or int(s.get("rows_error") or 0) > 0
    ]
    if incomplete or (errored and not args.allow_completed_with_errors):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
