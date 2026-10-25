"""Internal byte-for-byte HTTP response body archive.

The normal CAGAP run records keep parsed JSON responses and redact readable
reasoning/thinking text before public-facing artifacts. This module is the
separate restricted archive: it writes the exact HTTP response body bytes first,
then returns a manifest pointer that downstream records can reference.

The archive is intended for internal analysis only and should not be published.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_HTTP_ARCHIVE_DIR_ENV = "CAGAP_RAW_HTTP_ARCHIVE_DIR"
RAW_HTTP_ARCHIVE_DISABLED_ENV = "CAGAP_RAW_HTTP_ARCHIVE_DISABLED"
DEFAULT_RAW_HTTP_ARCHIVE_DIR = REPO_ROOT / "outputs" / "raw_http_archive"

_WRITE_LOCK = threading.Lock()
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe(text: Any, *, max_len: int = 96) -> str:
    value = _SAFE_RE.sub("_", str(text or "unknown")).strip("._")
    return (value or "unknown")[:max_len]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)


def _sha256_json(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _archive_root() -> Path:
    configured = os.environ.get(RAW_HTTP_ARCHIVE_DIR_ENV)
    return Path(configured).expanduser() if configured else DEFAULT_RAW_HTTP_ARCHIVE_DIR


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def raw_http_archive_enabled() -> bool:
    return os.environ.get(RAW_HTTP_ARCHIVE_DISABLED_ENV, "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def response_body_bytes(response: Any) -> bytes:
    """Return best available response body bytes for requests-like fakes.

    Real ``requests.Response.content`` is already the byte-for-byte body. The
    fallbacks keep unit-test doubles and SDK-like objects from crashing the
    capture path; those fallbacks are explicitly not a substitute for raw HTTP
    bytes in live runs.
    """
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8")
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.encode("utf-8")
    json_func = getattr(response, "json", None)
    if callable(json_func):
        try:
            return _json_dumps(json_func()).encode("utf-8")
        except Exception:  # noqa: BLE001
            return b""
    return b""


def archive_response_body(
    raw_body: bytes | str | None,
    *,
    role: str,
    model_id: str | None = None,
    route: str | None = None,
    provider: str | None = None,
    custom_id: str | None = None,
    attempt_index: int | None = None,
    http_status: int | None = None,
    url: str | None = None,
    request_body: Any = None,
    source: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Persist the exact response body bytes and append a manifest row.

    The body file is written before callers parse JSON or apply redaction.
    The manifest contains hashes and routing metadata only; it intentionally
    omits request/response headers because those can contain provider-specific
    sensitive values.
    """
    if not raw_http_archive_enabled() or raw_body is None:
        return None
    if isinstance(raw_body, str):
        body_bytes = raw_body.encode("utf-8")
        body_input_type = "str_utf8"
    else:
        body_bytes = bytes(raw_body)
        body_input_type = "bytes"

    captured_at = _now_utc()
    body_sha = hashlib.sha256(body_bytes).hexdigest()
    request_sha = _sha256_json(request_body)
    root = _archive_root()
    role_dir = _safe(role)
    model_dir = _safe(model_id)
    day = captured_at[:10]
    custom_hash = hashlib.sha256(str(custom_id or "").encode("utf-8")).hexdigest()[:12]
    stamp = captured_at.replace(":", "").replace("-", "").replace(".", "")
    filename = f"{stamp}_{custom_hash}_{body_sha[:16]}.body"
    body_path = root / "bodies" / role_dir / model_dir / day / filename
    manifest_path = root / "manifest.jsonl"

    record: dict[str, Any] = {
        "schema_version": "raw_http_archive.v1",
        "captured_at_utc": captured_at,
        "source": source,
        "role": role,
        "model_id": model_id,
        "route": route,
        "provider": provider,
        "custom_id": custom_id,
        "attempt_index": attempt_index,
        "http_status": http_status,
        "url": url,
        "body_path": _rel(body_path),
        "body_sha256": body_sha,
        "body_bytes": len(body_bytes),
        "body_input_type": body_input_type,
        "request_body_sha256": request_sha,
    }
    if extra:
        record["extra"] = dict(extra)

    with _WRITE_LOCK:
        body_path.parent.mkdir(parents=True, exist_ok=True)
        if not body_path.exists():
            tmp = body_path.with_name(body_path.name + f".tmp.{os.getpid()}")
            with tmp.open("wb") as fh:
                fh.write(body_bytes)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(body_path)
            _fsync_file(body_path)
            _fsync_dir(body_path.parent)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as mh:
            mh.write(_json_dumps(record) + "\n")
            mh.flush()
            os.fsync(mh.fileno())

    return {
        "raw_http_archive_schema_version": record["schema_version"],
        "raw_http_body_path": record["body_path"],
        "raw_http_body_sha256": body_sha,
        "raw_http_body_bytes": len(body_bytes),
        "raw_http_manifest_path": _rel(manifest_path),
        "raw_http_captured_at_utc": captured_at,
    }
