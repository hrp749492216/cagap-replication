"""Runtime provenance helpers for reproducible model dispatch.

The helpers in this module are intentionally provider-agnostic. They record
the model identity that was requested, the model identity the provider returned
when exposed, stable request IDs, safe response headers, and local checkpoint
manifest hashes without storing secrets.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

_SAFE_HEADER_EXACT = {
    "cf-ray",
    "openai-processing-ms",
    "request-id",
    "x-amzn-requestid",
    "x-request-id",
    "x-trace-id",
}
_SAFE_HEADER_PREFIXES = (
    "anthropic-ratelimit-",
    "ratelimit-",
    "x-ratelimit-",
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class OpenRouterMetadataError(RuntimeError):
    """Raised when OpenRouter model metadata cannot be captured."""


class CheckpointManifestError(RuntimeError):
    """Raised when a local/self-hosted checkpoint checksum manifest is missing or invalid."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Any) -> str:
    return _sha256_bytes(_json_dumps(obj).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_response_headers(headers: Any) -> dict[str, str]:
    """Return non-secret response headers useful for audit/replay.

    Header names are normalized to lowercase. Authorization, cookies, API keys,
    and provider-specific opaque auth material are not copied.
    """
    if not isinstance(headers, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in headers.items():
        lowered = str(key).lower()
        if lowered in _SAFE_HEADER_EXACT or any(
            lowered.startswith(prefix) for prefix in _SAFE_HEADER_PREFIXES
        ):
            out[lowered] = str(value)
    return dict(sorted(out.items()))


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}


def _finish_reason(response: dict[str, Any]) -> str | None:
    choice = _first_choice(response)
    value = choice.get("finish_reason") or response.get("stop_reason")
    return str(value) if value is not None else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _request_id_from_provider_result(provider_result: Any) -> str | None:
    if not isinstance(provider_result, dict):
        return None
    response = provider_result.get("response")
    if isinstance(response, dict):
        value = _first_present(
            response.get("request_id"),
            response.get("id"),
            response.get("requestId"),
        )
        if value is not None:
            return str(value)
    value = _first_present(
        provider_result.get("request_id"),
        provider_result.get("requestId"),
    )
    return str(value) if value is not None else None


def _request_id_from_headers(headers: dict[str, str]) -> str | None:
    for key in ("x-request-id", "request-id", "x-amzn-requestid", "cf-ray", "x-trace-id"):
        value = headers.get(key)
        if value:
            return str(value)
    return None


def build_api_runtime_stamp(
    *,
    requested_model: str | None,
    route: str,
    provider: str,
    response: dict[str, Any] | None,
    response_headers: Any = None,
    provider_result: dict[str, Any] | None = None,
    endpoint: str | None = None,
    role: str | None = None,
    model_id: str | None = None,
    provider_batch_id: str | None = None,
    http_status: int | None = None,
    model_metadata_snapshot: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the normalized identity/runtime block for an API execution."""
    raw_response = response if isinstance(response, dict) else {}
    safe_headers = safe_response_headers(response_headers)
    stamp: dict[str, Any] = {
        "role": role,
        "model_id": model_id,
        "route": route,
        "provider": provider,
        "endpoint": endpoint,
        "requested_model": requested_model,
        "returned_model": raw_response.get("model"),
        "response_id": raw_response.get("id"),
        "response_created": raw_response.get("created"),
        "system_fingerprint": raw_response.get("system_fingerprint"),
        "finish_reason": _finish_reason(raw_response),
        "provider_batch_id": provider_batch_id,
        "http_status": http_status,
        "response_headers": safe_headers,
        "response_request_id": _first_present(
            _request_id_from_headers(safe_headers),
            _request_id_from_provider_result(provider_result),
            raw_response.get("request_id"),
            raw_response.get("requestId"),
        ),
    }
    if model_metadata_snapshot:
        stamp["openrouter_model_metadata_snapshot_path"] = model_metadata_snapshot.get("path")
        stamp["openrouter_model_metadata_snapshot_sha256"] = model_metadata_snapshot.get("sha256")
        stamp["openrouter_model_metadata_matched_id"] = model_metadata_snapshot.get("matched_model_id")
    if extra:
        stamp.update(extra)
    return stamp


def _sanitize_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return safe or "model"


def write_openrouter_model_metadata_snapshot(
    *,
    session: Any,
    api_key: str,
    output_dir: Path,
    model_slug: str,
    pinned_provider: str,
) -> dict[str, Any]:
    """Capture the current OpenRouter /models entry for a routed model."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = session.get(OPENROUTER_MODELS_URL, headers=headers, timeout=60)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise OpenRouterMetadataError("OpenRouter /models response missing data list")
    matched = next(
        (model for model in models if isinstance(model, dict) and model.get("id") == model_slug),
        None,
    )
    if matched is None:
        raise OpenRouterMetadataError(
            f"OpenRouter /models response did not include model slug {model_slug!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_sanitize_filename(model_slug)}__{_sanitize_filename(pinned_provider)}.json"
    path = output_dir / filename
    snapshot = {
        "captured_at": _now_utc(),
        "source_url": OPENROUTER_MODELS_URL,
        "model_slug": model_slug,
        "pinned_provider": pinned_provider,
        "matched_model": matched,
        "response_headers": safe_response_headers(getattr(response, "headers", {})),
        "raw_response_sha256": _sha256_json(payload),
    }
    path.write_text(_json_dumps(snapshot) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "model_slug": model_slug,
        "matched_model_id": str(matched.get("id")),
    }


def _manifest_path_for_row(row: dict[str, Any], manifest_dir: Path) -> Path:
    subject = str(row.get("subject_id") or "")
    role = str(row.get("serving_precision_role") or "primary")
    value = str(row.get("serving_precision_value") or row.get("serving_precision") or "")
    if not subject or not value:
        raise CheckpointManifestError(
            "checkpoint manifest lookup requires subject_id and serving_precision_value"
        )
    return manifest_dir / f"{subject}__{role}__{value}.json"


def _manifest_field(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return value
    return None


def _require_equal(name: str, expected: Any, actual: Any) -> None:
    if str(expected) != str(actual):
        raise CheckpointManifestError(
            f"checkpoint manifest {name} mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_file_entries(files: Any) -> tuple[int, int | None]:
    if not isinstance(files, list) or not files:
        raise CheckpointManifestError("checkpoint manifest files must be a non-empty list")
    total = 0
    total_known = True
    for idx, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise CheckpointManifestError(f"checkpoint manifest files[{idx}] is not an object")
        path = entry.get("path")
        sha = entry.get("sha256") or entry.get("lfs_sha256")
        if not path:
            raise CheckpointManifestError(f"checkpoint manifest files[{idx}] missing path")
        if not isinstance(sha, str) or not _HEX64_RE.fullmatch(sha):
            raise CheckpointManifestError(f"checkpoint manifest files[{idx}] missing sha256")
        size = entry.get("size_bytes")
        if isinstance(size, int):
            total += size
        else:
            total_known = False
    return len(files), total if total_known else None


def load_checkpoint_manifest_for_row(
    row: dict[str, Any],
    manifest_dir: Path | str,
) -> dict[str, Any]:
    """Validate and summarize the checksum manifest for a local/self-hosted row."""
    root = Path(manifest_dir)
    path = _manifest_path_for_row(row, root)
    if not path.exists():
        raise CheckpointManifestError(f"checkpoint manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CheckpointManifestError(f"checkpoint manifest must be a JSON object: {path}")

    subject = str(row.get("subject_id") or "")
    role = str(row.get("serving_precision_role") or "primary")
    value = str(row.get("serving_precision_value") or row.get("serving_precision") or "")
    served_model = str(row.get("served_model_path") or "")
    _require_equal("subject_id", subject, _manifest_field(payload, "subject_id"))
    _require_equal(
        "serving_precision_role",
        role,
        _manifest_field(payload, "serving_precision_role", "precision_role"),
    )
    _require_equal(
        "serving_precision_value",
        value,
        _manifest_field(payload, "serving_precision_value", "precision_value"),
    )
    _require_equal("served_model_path", served_model, _manifest_field(payload, "served_model_path"))
    hf_commit = _manifest_field(payload, "hf_commit_sha", "hf_commit_sha_resolved")
    if not isinstance(hf_commit, str) or not _HF_COMMIT_RE.fullmatch(hf_commit):
        raise CheckpointManifestError("checkpoint manifest missing valid hf_commit_sha")
    file_count, total_size = _validate_file_entries(payload.get("files"))

    summary = {
        "subject_id": subject,
        "serving_precision_role": role,
        "serving_precision_value": value,
        "served_model_path": served_model,
        "hf_commit_sha": hf_commit,
        "file_count": file_count,
        "total_size_bytes": total_size,
    }
    return {
        "checkpoint_manifest_path": str(path),
        "checkpoint_manifest_sha256": _sha256_file(path),
        "checkpoint_manifest_summary": summary,
    }
