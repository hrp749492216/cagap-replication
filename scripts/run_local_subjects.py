"""Local-route subject dispatcher.

Consumes per-subject batch JSONL files emitted by `run_synthetic_prompts.py`
for local/self-hosted subjects (`route == "local"`). Routes each row to one of two
backends based on `served_model_location`:

  - `external_gpu`  → cloud vLLM HTTP client (OpenAI-compatible)
  - `m5_macbook_128gb` → local MLX-LM serving on Apple Silicon

Output: per-subject response JSONL with `latency_sec`, `usage`, `raw_response`,
and an audit block recording `actual_served_dtype`, `gpu_type`,
`tensor_parallelism`, `context_window`. Each row preserves its
`serving_precision_role` from the input batch (`primary` for BF16/FP16
reference, `quantization_audit` for INT4) so the analyzer's
`_primary_rows` filter behaves correctly.

For parallel same-subject serving, pass lane filters such as
`--serving-precision-role quantization_audit`, `--serving-precision int4_mlx`,
or `--served-model-location m5_macbook_128gb`. Filtered runs write to
lane-suffixed response ledgers, avoiding append collisions with primary lanes
for the same subject.

This is a skeleton for the design freeze. The actual MLX-LM and vLLM
HTTP code paths are stubbed pending: (1) cloud-GPU provider pinning and
auth-env, (2) MLX-LM client choice (mlx-lm CLI vs `mlx_lm.server`).
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from grader.json_strict import strict_json_loads
from scripts.runtime_provenance import (
    CheckpointManifestError,
    load_checkpoint_manifest_for_row,
)
from scripts.raw_http_archive import archive_response_body


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


DEFAULT_BATCHES_DIR = REPO_ROOT / "outputs" / "batches"
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "raw"
FIXED_RUN_TIMESTAMP_ENV = "CAGAP_FIXED_TIMESTAMP"
CHECKPOINT_MANIFEST_DIR_ENV = "CAGAP_CHECKPOINT_MANIFEST_DIR"
ALLOW_MISSING_CHECKPOINT_MANIFEST_ENV = "CAGAP_ALLOW_MISSING_CHECKPOINT_MANIFEST"


def _normalize_filter_values(values: Optional[Iterable[str]]) -> frozenset[str]:
    """Normalize CLI/API filter values, accepting comma-separated strings."""
    if not values:
        return frozenset()
    out: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            text = part.strip()
            if text:
                out.add(text)
    return frozenset(out)


def _path_token(value: str) -> str:
    """Filesystem-safe token for lane-filter output paths."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def local_response_output_path(
    results_dir: Path,
    subject_id: str,
    *,
    serving_precision_roles: Optional[Iterable[str]] = None,
    serving_precisions: Optional[Iterable[str]] = None,
    served_model_locations: Optional[Iterable[str]] = None,
) -> Path:
    """Return the response ledger path for a subject/lane.

    Unfiltered runs preserve the historical ``local_responses_<subject>.jsonl``
    path. Filtered precision/location lanes get a stable suffix so primary and
    quantization-audit runs for the same subject can execute concurrently
    without appending to the same file.
    """
    roles = _normalize_filter_values(serving_precision_roles)
    precisions = _normalize_filter_values(serving_precisions)
    locations = _normalize_filter_values(served_model_locations)
    suffix_parts: list[str] = []
    for label, values in (
        ("location", locations),
        ("precision", precisions),
        ("role", roles),
    ):
        if values:
            suffix_parts.append(
                f"{label}-{'-'.join(_path_token(value) for value in sorted(values))}"
            )
    suffix = "".join(f"__{part}" for part in suffix_parts)
    return results_dir / f"local_responses_{subject_id}{suffix}.jsonl"


def _row_matches_lane_filters(
    row: Dict[str, Any],
    *,
    serving_precision_roles: frozenset[str],
    serving_precisions: frozenset[str],
    served_model_locations: frozenset[str],
) -> bool:
    if serving_precision_roles:
        role = str(row.get("serving_precision_role", "primary"))
        if role not in serving_precision_roles:
            return False
    if serving_precisions:
        row_precisions = {
            str(value)
            for value in (
                row.get("serving_precision"),
                row.get("serving_precision_value"),
            )
            if value is not None
        }
        if row_precisions.isdisjoint(serving_precisions):
            return False
    if served_model_locations:
        location = row.get("served_model_location")
        if location is None or str(location) not in served_model_locations:
            return False
    return True


def _run_record_timestamp() -> str:
    """UTC run-record timestamp, optionally fixed for byte-level replays."""
    fixed = os.environ.get(FIXED_RUN_TIMESTAMP_ENV)
    if fixed:
        return fixed
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _strict_jsonl_object(line: str | bytes, *, path: Path, line_no: int) -> Dict[str, Any]:
    try:
        parsed = strict_json_loads(line)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}:{line_no}: invalid strict JSON row: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}:{line_no}: JSONL row must be an object.")
    return parsed


def _strict_response_object(raw: bytes, *, context: str) -> Dict[str, Any]:
    try:
        parsed = strict_json_loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"{context} not strict JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{context} JSON root must be an object.")
    return parsed


@dataclass
class BackendConfig:
    """Per-row backend selection metadata."""

    role: str               # "primary" | "quantization_audit"
    location: str           # "external_gpu" | "m5_macbook_128gb"
    runtime: str            # "vllm_float16" | "mlx_lm_int4" | etc.
    served_model_path: str  # hf_repo or mlx-community INT4 source
    target_dtype: str       # "fp16" | "bf16" | "int4_mlx"


def select_backend(row: Dict[str, Any]) -> BackendConfig:
    """Pick the backend for a single local-route row.

    The runner stamps `served_model_*` directly from the variant dict
    (see `precision_variants_for_subject` + `_build_request_row`), so the
    `local_inference.*` fallback below is defense-in-depth for rows missing
    served-model stamps or partial replays. The fallback is **role-aware**: a
    `quantization_audit` row falls back through `precision_sensitivity_*`,
    NOT `precision_primary_*` — otherwise an audit row missing its
    `served_model_runtime` would silently inherit the BF16 reference
    runtime (e.g., `vllm_float16`) and dispatch with the wrong runtime.
    """
    role = row.get("serving_precision_role", "primary")
    li = row.get("local_inference") or {}
    if role == "quantization_audit":
        loc_fallback = li.get("precision_sensitivity_location")
        runtime_fallback = li.get("precision_sensitivity_runtime")
    else:
        loc_fallback = li.get("precision_primary_location")
        runtime_fallback = li.get("precision_primary_runtime")
    location = row.get("served_model_location") or loc_fallback
    runtime = row.get("served_model_runtime") or runtime_fallback
    served_model = row.get("served_model_path") or row.get("body", {}).get("model")
    target_dtype = row.get("serving_precision", "bf16")
    if not location:
        raise ValueError(
            f"row {row.get('custom_id')!r} has no served_model_location; "
            "cannot pick local backend."
        )
    if not served_model:
        raise ValueError(
            f"row {row.get('custom_id')!r} has no served_model_path; "
            "runner emitted an incomplete local row."
        )
    return BackendConfig(
        role=role,
        location=location,
        runtime=runtime or "unspecified",
        served_model_path=served_model,
        target_dtype=target_dtype,
    )


# ---------------------------------------------------------------------------
# Backend adapters (stubbed; wire to actual clients in follow-ups)
# ---------------------------------------------------------------------------


RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class ModelMismatchError(RuntimeError):
    """Raised when the server's loaded model does not match the row's
    `served_model_path`. Routes a single row to the error path so the
    operator can spin up the correct server and re-run, rather than
    silently producing wrong-model rows."""


class RuntimeProvenanceError(RuntimeError):
    """Raised when mandatory local/self-hosted runtime provenance is not configured."""


def _runtime_env(key: str) -> str | None:
    val = os.environ.get(key)
    return val.strip() if isinstance(val, str) and val.strip() else None


def runtime_provenance_from_env(*, require: bool = True) -> Dict[str, Any]:
    """Read mandatory local/self-hosted hardware provenance from operator env vars."""
    values = {
        "hardware_name": _runtime_env("CAGAP_RUNTIME_HARDWARE"),
        "driver_version": _runtime_env("CAGAP_RUNTIME_DRIVER"),
        "cuda_version": _runtime_env("CAGAP_RUNTIME_CUDA"),
        "compute_capability": _runtime_env("CAGAP_RUNTIME_COMPUTE_CAP"),
        "tensor_parallel_size": _runtime_env("CAGAP_RUNTIME_TP_SIZE"),
    }
    missing = [key for key, value in values.items() if value is None]
    if require and missing:
        raise RuntimeProvenanceError(
            "Local/self-hosted runtime provenance env vars missing: "
            + ", ".join(missing)
            + ". Export CAGAP_RUNTIME_HARDWARE, CAGAP_RUNTIME_DRIVER, "
            "CAGAP_RUNTIME_CUDA, CAGAP_RUNTIME_COMPUTE_CAP, and "
            "CAGAP_RUNTIME_TP_SIZE before dispatch."
        )
    tp_value = values["tensor_parallel_size"]
    if tp_value is not None:
        if not str(tp_value).isdigit():
            raise RuntimeProvenanceError(
                "CAGAP_RUNTIME_TP_SIZE must be a positive integer string."
            )
        values["tensor_parallel_size"] = int(str(tp_value))
    return values


def _normalize_model_id(value: Optional[str]) -> str:
    """Permissive normalization for model-id comparison. Expands ~,
    strips trailing slashes, lowercases. Filesystem paths and HF repo
    ids are both safe under this rule (HF ids contain `/` but not `~`)."""
    if not value:
        return ""
    expanded = os.path.expanduser(str(value)).rstrip("/\\")
    return expanded.lower()


def _model_paths_match(loaded: str, expected: str) -> bool:
    """Strict equality after expanduser + trailing-slash + case
    normalization. We DO NOT fall back to basename match: BF16 and
    INT4 siblings can share a common basename (e.g.,
    `/srv/models/bf16/Qwen3.6-35B-A3B/` vs
    `/srv/models/int4/Qwen3.6-35B-A3B/`), and a basename rule would
    silently route the audit variant onto the reference server (or
    vice versa). If a deployment uses non-canonical paths, the
    operator must align the roster's `served_model_path` with the
    `--model` arg passed to mlx_lm.server / vllm so the
    `/v1/models` id is bit-identical."""
    if not loaded or not expected:
        return False
    return _normalize_model_id(loaded) == _normalize_model_id(expected)


def _loaded_model_ids(metadata: Dict[str, Any]) -> List[str]:
    """Pull served-model ids out of /v1/models response JSON."""
    data = metadata.get("data") or []
    out: List[str] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        mid = entry.get("id") or entry.get("model")
        if mid:
            out.append(str(mid))
    return out


def assert_loaded_model_matches(
    client: Any,
    expected_model_path: str,
    *,
    skip: bool = False,
) -> Optional[str]:
    """Pre-flight check: confirm the model loaded on the server matches
    the row's `served_model_path`. Returns the matched server-side id
    (for stamping into runtime_stamp), or None if `skip` is set.

    Raises `ModelMismatchError` if the server reports a non-matching model
    or no model at all. Operators can bypass via `--skip-model-match` or
    `CAGAP_SKIP_MODEL_MATCH=1` (ONLY safe when running a single subject
    where the operator just (re)started the correct server)."""
    if skip or client is None:
        return None
    metadata = client.server_metadata() if hasattr(client, "server_metadata") else {}
    loaded_ids = _loaded_model_ids(metadata)
    if not loaded_ids:
        raise ModelMismatchError(
            f"server at {getattr(client, 'endpoint', None) or getattr(client, 'base_url', None)!r} "
            f"reported no loaded model on /v1/models; cannot confirm row "
            f"served_model_path={expected_model_path!r}."
        )
    for lid in loaded_ids:
        if _model_paths_match(lid, expected_model_path):
            return lid
    raise ModelMismatchError(
        f"loaded model(s) {loaded_ids!r} on "
        f"{getattr(client, 'endpoint', None) or getattr(client, 'base_url', None)!r} "
        f"do not match row.served_model_path={expected_model_path!r}; "
        "(re)start the matching server or pass --skip-model-match for testing."
    )


def _post_chat_completions(
    url: str,
    body: Dict[str, Any],
    *,
    headers: Dict[str, str],
    timeout: int,
    max_retries: int = 5,
    backoff_base: float = 1.5,
    archive_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """POST a chat-completions body to an OpenAI-compatible endpoint.

    Uses urllib so the dispatcher has no hard runtime dependency on httpx.
    Retries with exponential backoff on transient HTTP statuses
    (408/425/429/500/502/503/504) and on URLError transport failures.
    Non-retryable HTTP errors (4xx other than 408/425/429) raise RuntimeError
    immediately so the caller can stamp a permanent error row.
    """
    payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=payload, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)
        if "Content-Type" not in headers:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                http_status = getattr(resp, "status", None)
                raw = resp.read()
            raw_capture = archive_response_body(
                raw,
                role=str((archive_metadata or {}).get("role") or "local_subject"),
                model_id=(archive_metadata or {}).get("model_id"),
                route=(archive_metadata or {}).get("route") or "local",
                provider=(archive_metadata or {}).get("provider"),
                custom_id=(archive_metadata or {}).get("custom_id"),
                attempt_index=attempt + 1,
                http_status=int(http_status) if isinstance(http_status, int) else None,
                url=url,
                request_body=body,
                source="urllib.response.read",
            )
            parsed = _strict_response_object(raw, context="chat response")
            if raw_capture is not None:
                parsed["_raw_http_capture"] = raw_capture
            return parsed
        except urllib.error.HTTPError as exc:
            raw_error = exc.read() if hasattr(exc, "read") else b""
            archive_response_body(
                raw_error,
                role=str((archive_metadata or {}).get("role") or "local_subject"),
                model_id=(archive_metadata or {}).get("model_id"),
                route=(archive_metadata or {}).get("route") or "local",
                provider=(archive_metadata or {}).get("provider"),
                custom_id=(archive_metadata or {}).get("custom_id"),
                attempt_index=attempt + 1,
                http_status=exc.code,
                url=url,
                request_body=body,
                source="urllib.HTTPError.read",
            )
            detail = raw_error.decode("utf-8", errors="replace")
            last_exc = RuntimeError(
                f"chat HTTP {exc.code} {exc.reason}: {detail[:500]}"
            )
            if exc.code in RETRYABLE_HTTP_STATUSES and attempt < max_retries:
                delay = backoff_base ** attempt
                logger.warning(
                    "transient HTTP %s on %s (attempt %d/%d); retrying in %.1fs",
                    exc.code, url, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue
            raise last_exc from exc
        except urllib.error.URLError as exc:
            last_exc = RuntimeError(f"chat transport error: {exc}")
            if attempt < max_retries:
                delay = backoff_base ** attempt
                logger.warning(
                    "transport error on %s (attempt %d/%d); retrying in %.1fs",
                    url, attempt + 1, max_retries, delay,
                )
                time.sleep(delay)
                continue
            raise last_exc from exc
    # Should not reach here; loop either returns or raises.
    raise last_exc or RuntimeError("chat retry loop exited without success")


def _chat_completion_with_optional_archive(
    client: Any,
    body: Dict[str, Any],
    archive_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Call client.chat with raw-archive metadata when that client supports it."""
    chat = getattr(client, "chat")
    try:
        signature = inspect.signature(chat)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        accepts_archive = any(
            p.kind == inspect.Parameter.VAR_KEYWORD or p.name == "archive_metadata"
            for p in signature.parameters.values()
        )
        if not accepts_archive:
            return chat(body)
    return chat(body, archive_metadata=archive_metadata)


def _query_models_metadata(
    base_url: str,
    *,
    headers: Dict[str, str],
    timeout: int = 10,
) -> Dict[str, Any]:
    """GET /v1/models on an OpenAI-compatible server. Returns parsed JSON or
    {} on failure (so introspection failures never block dispatch)."""
    url = f"{base_url}/v1/models"
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return _strict_response_object(raw, context="server metadata") if raw else {}
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
        logger.warning("server metadata query failed at %s: %s", url, exc)
        return {}


class VLLMCloudClient:
    """Cloud-GPU vLLM HTTP client (OpenAI-compatible chat-completions).

    Connects to a vLLM serving endpoint that exposes `/v1/chat/completions`.
    Assumes Bearer-token auth via `VLLM_CLOUD_AUTH` (override per-instance).
    Pinned provider list (per roster `vllm_backend_note`): Lambda.ai or
    Verda.com, instance preference B300 > B200 > H200 > H100 > RTX PRO 6000 >
    A100 80GB > A100 40GB (multi-GPU only).
    """

    PROVIDER_LABEL = "cloud_vllm"

    def __init__(
        self,
        endpoint: str,
        auth_env: str = "VLLM_CLOUD_AUTH",
        *,
        max_retries: int = 5,
        backoff_base: float = 1.5,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.auth_env = auth_env
        self.api_key = os.environ.get(auth_env)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        if not self.api_key:
            raise RuntimeError(
                f"{auth_env} env var not set; cloud-GPU vLLM client cannot start. "
                "Set the auth env or pass --vllm-endpoint='' to skip external_gpu rows."
            )
        self._server_metadata: Optional[Dict[str, Any]] = None

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def server_metadata(self) -> Dict[str, Any]:
        """Lazy-cache server metadata (model card, capabilities, served
        dtype if exposed). Called once per client and stamped per row."""
        if self._server_metadata is None:
            self._server_metadata = _query_models_metadata(
                self.endpoint, headers=self._headers()
            )
        return self._server_metadata

    def chat(
        self,
        body: Dict[str, Any],
        *,
        timeout: int = 600,
        archive_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.endpoint}/v1/chat/completions"
        return _post_chat_completions(
            url,
            body,
            headers=self._headers(),
            timeout=timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            archive_metadata=archive_metadata,
        )


class MLXLocalClient:
    """Local MLX-LM serving on Apple Silicon (M5 128 GB).

    Talks to a local `mlx_lm.server` instance over HTTP using the
    OpenAI-compatible chat-completions surface. The served chat template
    must match the model (Llama 3.3 / Gemma 4 / Qwen 3.6) — start the
    server with `mlx_lm.server --model <hf_repo_or_local_dir>
    --chat-template auto --host 127.0.0.1 --port 8080`.
    """

    PROVIDER_LABEL = "local_mlx"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._server_metadata: Optional[Dict[str, Any]] = None

    def server_metadata(self) -> Dict[str, Any]:
        if self._server_metadata is None:
            self._server_metadata = _query_models_metadata(self.base_url, headers={})
        return self._server_metadata

    def chat(
        self,
        body: Dict[str, Any],
        *,
        timeout: int = 600,
        archive_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        return _post_chat_completions(
            url,
            body,
            headers={},
            timeout=timeout,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
            archive_metadata=archive_metadata,
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_INPUT_PASSTHROUGH_FIELDS = (
    "subject_id",
    "route",
    "pinned_provider",
    "mode",
    "order_index",
    "randomized_order_seed",
    "source_randomized_order_seed",
    "persona",
    "prompt_id",
    "trial_id",
    "intervention_condition",
    "corpus_partition_id",
    "domain",
    "serving_precision_value",
    "freeze_date",
    "decoding_applied",
    "prompt_sha256",
    "schema_sha256",
    "irrelevant_pool_sha256",
    "intervention_prompt_sha256",
    "disabled_arm_schema_visibility",
    "route_support_status",
    "schema_visibility_evidence_id",
    "schema_visibility_evidence_sha256",
    "serialized_request_hash",
    "tool_schema_hash",
    "system_prompt_hash",
    "hashes",
    "injected_tool_subset",
    "protocol_id",
    "protocol_sha256",
    "postregistration_exploratory",
    "analysis_family",
    "format_condition",
    "control_arm",
    "format_instruction_present",
    "format_instruction_sha256",
    "analysis_inclusion",
    "operational_canary",
    "partition_id",
    "source_custom_id",
    "source_serialized_request_hash",
    "source_batch_sha256",
    "original_system_prompt_hash",
    "control_route_lock",
    "dispatch_block_id",
    "dispatch_block_sequence_index",
    "dispatch_miniblock_key",
)


def dispatch_row(
    row: Dict[str, Any],
    *,
    vllm_client: Optional[VLLMCloudClient],
    mlx_client: Optional[MLXLocalClient],
    schema_version: str = "run_record",
    skip_model_match: bool = False,
    checkpoint_manifest_dir: Optional[Path] = None,
    require_checkpoint_manifest: bool = False,
) -> Dict[str, Any]:
    """Run one local-route row, returning a response shape conforming to
    `schemas/run_record.schema.json` (so the grader / judge / extractor
    pipeline can consume local rows alongside native and OpenRouter rows)."""
    backend = select_backend(row)
    body = dict(row["body"])
    # Strip OR-only fields so local/native routes don't see provider/plugins/transforms.
    from scripts._tool_schema_guard import strip_route_only_fields
    strip_route_only_fields(body)
    base = {key: row[key] for key in _INPUT_PASSTHROUGH_FIELDS if key in row}
    base.update(
        {
            "run_id": str(uuid.uuid5(uuid.NAMESPACE_URL, str(row["custom_id"]))),
            "timestamp": _run_record_timestamp(),
            "custom_id": row["custom_id"],
            "schema_version": schema_version,
            "model": backend.served_model_path,
            "provider": "local",
            "serving_precision_role": backend.role,
            "serving_precision": backend.target_dtype,
            "serving_precision_value": row.get("serving_precision_value") or backend.target_dtype,
            "served_model_path": backend.served_model_path,
            "served_model_location": backend.location,
            "served_model_runtime": backend.runtime,
            "request_prompt": body.get("messages", []),
        }
    )
    checkpoint_stamp: Dict[str, Any] = {}
    if require_checkpoint_manifest:
        if checkpoint_manifest_dir is None:
            raise CheckpointManifestError("checkpoint manifest directory is required")
        checkpoint_stamp = load_checkpoint_manifest_for_row(row, checkpoint_manifest_dir)
        base.update(checkpoint_stamp)
    runtime_provenance = runtime_provenance_from_env(require=True)
    started = time.monotonic()
    try:
        client = None
        if backend.location == "external_gpu":
            if vllm_client is None:
                raise RuntimeError(
                    "row routed to external_gpu but no VLLMCloudClient configured."
                )
            client = vllm_client
        elif backend.location == "m5_macbook_128gb":
            if mlx_client is None:
                raise RuntimeError(
                    "row routed to m5_macbook_128gb but no MLXLocalClient configured."
                )
            client = mlx_client
        else:
            raise ValueError(
                f"unknown served_model_location {backend.location!r} for row {row.get('custom_id')!r}"
            )
        # Pre-flight: ensure the running server is loaded with the model
        # this row was scheduled against. Catches operator error of
        # forgetting to (re)start the correct server between precision
        # variants. Bypass with `skip_model_match=True` (CLI flag /
        # CAGAP_SKIP_MODEL_MATCH=1) only for tests.
        matched_loaded_id = assert_loaded_model_matches(
            client, backend.served_model_path, skip=skip_model_match,
        )
        resp = _chat_completion_with_optional_archive(
            client,
            body,
            archive_metadata={
                "role": "local_subject",
                "model_id": row.get("subject_id"),
                "route": "local",
                "provider": getattr(client, "PROVIDER_LABEL", "local"),
                "custom_id": row.get("custom_id"),
            },
        )
        raw_http_capture = resp.pop("_raw_http_capture", None)
        elapsed = time.monotonic() - started
        # Per-row runtime stamping: capture what the response itself exposes
        # (system_fingerprint, finish_reason, returned model id) plus a
        # one-shot snapshot of server-side metadata (cached per client) so
        # downstream provenance audits can compare actual served dtype /
        # context window / tensor parallelism against the predeclared
        # roster.local_config values.
        choice = (resp.get("choices") or [{}])[0]
        runtime_stamp = {
            "client_label": getattr(client, "PROVIDER_LABEL", "unknown"),
            "endpoint": getattr(client, "endpoint", None) or getattr(client, "base_url", None),
            "returned_model": resp.get("model"),
            "system_fingerprint": resp.get("system_fingerprint"),
            "finish_reason": choice.get("finish_reason"),
            "server_metadata": client.server_metadata() if client else {},
            "loaded_model_match": matched_loaded_id,
            "model_match_skipped": skip_model_match,
            **runtime_provenance,
            "cuda_version_runtime": runtime_provenance["cuda_version"],
        }
        if checkpoint_stamp:
            runtime_stamp.update(
                {
                    "checkpoint_manifest_path": checkpoint_stamp["checkpoint_manifest_path"],
                    "checkpoint_manifest_sha256": checkpoint_stamp["checkpoint_manifest_sha256"],
                    "checkpoint_manifest_summary": checkpoint_stamp["checkpoint_manifest_summary"],
                }
            )
        base.update(
            {
                "latency_sec": elapsed,
                "usage": resp.get("usage", {}),
                "raw_response": resp,
                "raw_http_capture": raw_http_capture,
                "runtime_stamp": runtime_stamp,
            }
        )
        return base
    except ModelMismatchError as exc:
        elapsed = time.monotonic() - started
        logger.error(
            "row %s model-mismatch: %s", row.get("custom_id"), exc,
        )
        base.update(
            {
                "latency_sec": elapsed,
                "usage": {},
                "raw_response": {},
                "error": f"model_mismatch: {exc}",
            }
        )
        return base
    except (NotImplementedError, RuntimeError) as exc:
        elapsed = time.monotonic() - started
        logger.warning(
            "row %s backend error: %s", row.get("custom_id"), exc,
        )
        base.update(
            {
                "latency_sec": elapsed,
                "usage": {},
                "raw_response": {},
                "error": f"backend_error: {exc}",
            }
        )
        return base
    except Exception as exc:  # transient: socket reset, JSON parse, etc.
        elapsed = time.monotonic() - started
        logger.warning(
            "row %s transient error (%s): %s",
            row.get("custom_id"), type(exc).__name__, exc,
        )
        base.update(
            {
                "latency_sec": elapsed,
                "usage": {},
                "raw_response": {},
                "error": f"transient_{type(exc).__name__}: {exc}",
            }
        )
        return base


def _completed_output_custom_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = _strict_jsonl_object(line, path=path, line_no=line_no)
            custom_id = row.get("custom_id")
            if custom_id is None:
                continue
            custom_id_text = str(custom_id)
            if custom_id_text in completed:
                raise ValueError(f"{path}:{line_no}: duplicate completed custom_id {custom_id_text!r}.")
            completed.add(custom_id_text)
    return completed


def _ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")


def dispatch_file(
    input_path: Path,
    output_path: Path,
    *,
    vllm_client: Optional[VLLMCloudClient] = None,
    mlx_client: Optional[MLXLocalClient] = None,
    skip_model_match: bool = False,
    checkpoint_manifest_dir: Optional[Path] = None,
    require_checkpoint_manifest: bool = False,
    sync_interval_seconds: float = 30.0,
    serving_precision_roles: Optional[Iterable[str]] = None,
    serving_precisions: Optional[Iterable[str]] = None,
    served_model_locations: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Read a per-subject batch JSONL, dispatch each local row, write results."""
    normalized_roles = _normalize_filter_values(serving_precision_roles)
    normalized_precisions = _normalize_filter_values(serving_precisions)
    normalized_locations = _normalize_filter_values(served_model_locations)
    counts = {
        "total": 0,
        "skipped_non_local": 0,
        "skipped_filter": 0,
        "skipped_existing": 0,
        "dispatched": 0,
        "errored": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sync_manifest_path = output_path.with_suffix(output_path.suffix + ".sync.json")
    last_sync = time.monotonic()

    def write_sync_manifest(last_custom_id: str | None = None) -> None:
        payload = {
            "updated_at_utc": _run_record_timestamp(),
            "output_path": str(output_path),
            "raw_http_archive_dir": os.environ.get("CAGAP_RAW_HTTP_ARCHIVE_DIR")
            or str(REPO_ROOT / "outputs" / "raw_http_archive"),
            "sync_interval_seconds": sync_interval_seconds,
            "last_custom_id": last_custom_id,
            "counts": counts,
            "lane_filters": {
                "serving_precision_roles": sorted(normalized_roles),
                "serving_precisions": sorted(normalized_precisions),
                "served_model_locations": sorted(normalized_locations),
            },
            "note": (
                "Completed local/external-GPU response rows are written and "
                "fsynced to this repo immediately. This file is refreshed at "
                "least every sync_interval_seconds when completed rows are available."
            ),
        }
        tmp = sync_manifest_path.with_name(sync_manifest_path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(sync_manifest_path)

    completed_ids = _completed_output_custom_ids(output_path)
    _ensure_trailing_newline(output_path)
    output_mode = "a" if output_path.exists() else "w"
    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        output_mode, encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            counts["total"] += 1
            row = _strict_jsonl_object(line, path=input_path, line_no=line_no)
            if row.get("route") != "local":
                counts["skipped_non_local"] += 1
                continue
            if not _row_matches_lane_filters(
                row,
                serving_precision_roles=normalized_roles,
                serving_precisions=normalized_precisions,
                served_model_locations=normalized_locations,
            ):
                counts["skipped_filter"] += 1
                continue
            custom_id = row.get("custom_id")
            if custom_id is None:
                raise ValueError(f"{input_path}:{line_no}: local row missing custom_id.")
            custom_id_text = str(custom_id)
            if custom_id_text in completed_ids:
                counts["skipped_existing"] += 1
                continue
            try:
                result = dispatch_row(
                    row,
                    vllm_client=vllm_client,
                    mlx_client=mlx_client,
                    skip_model_match=skip_model_match,
                    checkpoint_manifest_dir=checkpoint_manifest_dir,
                    require_checkpoint_manifest=require_checkpoint_manifest,
                )
            except RuntimeProvenanceError:
                raise
            except CheckpointManifestError:
                raise
            except Exception as exc:  # last-resort: never abort the batch
                logger.error(
                    "dispatch_file: row %s raised %s through all handlers: %s",
                    custom_id_text, type(exc).__name__, exc,
                )
                result = {
                    **{
                        key: row[key]
                        for key in _INPUT_PASSTHROUGH_FIELDS
                        if key in row
                    },
                    "custom_id": custom_id_text,
                    "subject_id": row.get("subject_id"),
                    "error": f"unhandled_{type(exc).__name__}: {exc}",
                }
            if "error" in result:
                counts["errored"] += 1
            else:
                counts["dispatched"] += 1
            try:
                dst.write(
                    json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                dst.flush()
                os.fsync(dst.fileno())
            except Exception as exc:
                logger.error(
                    "dispatch_file: failed to serialize row %s (%s); writing minimal error stub",
                    custom_id_text, exc,
                )
                dst.write(json.dumps({"custom_id": custom_id_text, "error": f"serialize_fail: {exc}"}) + "\n")
                dst.flush()
                os.fsync(dst.fileno())
            completed_ids.add(custom_id_text)
            now = time.monotonic()
            if sync_interval_seconds <= 0 or now - last_sync >= sync_interval_seconds:
                write_sync_manifest(custom_id_text)
                last_sync = now
        write_sync_manifest(None)
    return counts


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=DEFAULT_BATCHES_DIR,
        help="Directory containing per-subject input_<subject_id>.jsonl files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to write response JSONL files.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subset of subject IDs (default: all *.jsonl in batches dir).",
    )
    parser.add_argument(
        "--serving-precision-role",
        dest="serving_precision_roles",
        action="append",
        default=None,
        help=(
            "Optional local-row lane filter, repeatable or comma-separated "
            "(for example: primary or quantization_audit). Filtered runs write "
            "to a lane-suffixed response path so same-subject lanes can run in parallel."
        ),
    )
    parser.add_argument(
        "--serving-precision",
        dest="serving_precisions",
        action="append",
        default=None,
        help=(
            "Optional local-row lane filter on serving_precision or "
            "serving_precision_value, repeatable or comma-separated "
            "(for example: bf16, fp16, int4_mlx)."
        ),
    )
    parser.add_argument(
        "--served-model-location",
        dest="served_model_locations",
        action="append",
        default=None,
        help=(
            "Optional local-row lane filter on served_model_location, repeatable "
            "or comma-separated (for example: external_gpu or m5_macbook_128gb)."
        ),
    )
    parser.add_argument(
        "--vllm-endpoint",
        default=None,
        help="Cloud-GPU vLLM endpoint (OpenAI-compatible). If unset, external_gpu rows are skipped.",
    )
    parser.add_argument(
        "--mlx-endpoint",
        default="http://127.0.0.1:8080",
        help="Local MLX-LM server endpoint (OpenAI-compatible).",
    )
    parser.add_argument(
        "--skip-model-match",
        action="store_true",
        default=os.environ.get("CAGAP_SKIP_MODEL_MATCH") == "1",
        help=(
            "Bypass the pre-flight check that the running server has the "
            "row's served_model_path loaded. UNSAFE in batch — only use for "
            "tests or while a single subject's server is verified manually."
        ),
    )
    parser.add_argument(
        "--checkpoint-manifest-dir",
        type=Path,
        default=(
            Path(os.environ[CHECKPOINT_MANIFEST_DIR_ENV])
            if os.environ.get(CHECKPOINT_MANIFEST_DIR_ENV)
            else None
        ),
        help=(
            "Directory containing per-(subject, precision-role, precision-value) "
            "checkpoint checksum manifests named "
            "<subject>__<role>__<precision>.json. Defaults to "
            f"${CHECKPOINT_MANIFEST_DIR_ENV}."
        ),
    )
    parser.add_argument(
        "--allow-missing-checkpoint-manifest",
        action="store_true",
        default=os.environ.get(ALLOW_MISSING_CHECKPOINT_MANIFEST_ENV) == "1",
        help=(
            "Diagnostic bypass for local/self-hosted checksum-manifest enforcement. "
            "Confirmatory local dispatch should leave this unset."
        ),
    )
    parser.add_argument(
        "--sync-interval-seconds",
        type=float,
        default=30.0,
        help=(
            "Refresh a local sync manifest for local/external-GPU output at this interval. "
            "Completed response rows and raw HTTP bodies are still fsynced immediately."
        ),
    )
    args = parser.parse_args(argv)
    serving_precision_roles = _normalize_filter_values(args.serving_precision_roles)
    serving_precisions = _normalize_filter_values(args.serving_precisions)
    served_model_locations = _normalize_filter_values(args.served_model_locations)
    require_checkpoint_manifest = not args.allow_missing_checkpoint_manifest
    if require_checkpoint_manifest and args.checkpoint_manifest_dir is None:
        logger.error(
            "checkpoint manifest directory is required for local dispatch. "
            "Pass --checkpoint-manifest-dir or set %s. Use "
            "--allow-missing-checkpoint-manifest only for diagnostic smoke runs.",
            CHECKPOINT_MANIFEST_DIR_ENV,
        )
        return 2

    vllm_client = VLLMCloudClient(args.vllm_endpoint) if args.vllm_endpoint else None
    mlx_client = MLXLocalClient(args.mlx_endpoint) if args.mlx_endpoint else None

    batch_files: List[Path] = []
    if args.subjects:
        for sid in args.subjects:
            p = args.batches_dir / f"input_{sid}.jsonl"
            if not p.exists():
                logger.warning("no batch file for subject %s at %s; skipping", sid, p)
                continue
            batch_files.append(p)
    else:
        batch_files = sorted(args.batches_dir.glob("input_*.jsonl"))

    if not batch_files:
        logger.error("no batch files found in %s", args.batches_dir)
        return 2

    grand = {
        "total": 0,
        "skipped_non_local": 0,
        "skipped_filter": 0,
        "skipped_existing": 0,
        "dispatched": 0,
        "errored": 0,
    }
    for src in batch_files:
        sid = src.stem.removeprefix("input_")
        dst = local_response_output_path(
            args.results_dir,
            sid,
            serving_precision_roles=serving_precision_roles,
            serving_precisions=serving_precisions,
            served_model_locations=served_model_locations,
        )
        logger.info("dispatching %s -> %s", src, dst)
        c = dispatch_file(
            src,
            dst,
            vllm_client=vllm_client,
            mlx_client=mlx_client,
            skip_model_match=args.skip_model_match,
            checkpoint_manifest_dir=args.checkpoint_manifest_dir,
            require_checkpoint_manifest=require_checkpoint_manifest,
            sync_interval_seconds=args.sync_interval_seconds,
            serving_precision_roles=serving_precision_roles,
            serving_precisions=serving_precisions,
            served_model_locations=served_model_locations,
        )
        for k, v in c.items():
            grand[k] += v
        logger.info("  %s", c)
    logger.info("DONE: %s", grand)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
