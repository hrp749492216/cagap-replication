#!/usr/bin/env python3
"""Prepare Pilot-2 / measurement-replay judge prompts from responses and MiniMax extractions.

Per-subject pipeline:
  Phase A — load precomputed MiniMax batch extractions.
  Phase B — build judge prompts from prompt template + extracted actions + raw response.
  Phase C — write per-judge prompt JSONL files and a dispatch manifest.

Pilot-2 route policy:
  - MiniMax extraction is run before this script via its async/native batch
    path, then supplied with ``--extractions``.
  - This script prepares judge inputs only. Runtime dispatch route is resolved
    later from ``config/roster.json`` and the family-excluded judge panel.

Output:
  outputs/<run_dir>/judge_batches/<subject>/manifest.json
    { "subject_id": ..., "extractions_path": ..., "judges": {...} }
  outputs/<run_dir>/judge_batches/<subject>/extractions.jsonl
    one line per response row; contains custom_id + actions + refusal flags
  outputs/<run_dir>/judge_batches/<subject>/judge_inputs/<judge_id>.jsonl
    rendered judge prompts to send with the per-judge dispatcher scripts

The manifest is consumed by the eventual merge step in
``grade_subject_responses.py`` / ``merge_grading_outputs.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractor.extract import Extractor, subject_response_from_response  # noqa: E402
from scripts.cagap_env import load_cagap_env  # noqa: E402
from scripts.raw_http_archive import archive_response_body, response_body_bytes  # noqa: E402
# Predicate-aware dispatch (Amendment 13.J 2026-05-08): we re-use the same
# helpers grade_subject_responses.py applies in grade_row() to decide which
# rows the predicate path can resolve before judges are needed. Rows that
# satisfy any of these gates do NOT get a judge prompt sent to a batch.
from scripts.grade_subject_responses import (  # noqa: E402
    _demote_clear_refusal_on_operational_tail,
    _extraction_payload_for_routing,
    has_structured_tool_calls,
    is_complete_tool_call_dict,
    load_precomputed_extractions,
)
from scripts import judge_panel as judge_panel_resolver  # noqa: E402

PROMPT_TEMPLATE_PATH = REPO_ROOT / "judges" / "prompt.md"
MISTRAL_BASE = "https://api.mistral.ai"
OPENAI_BASE = "https://api.openai.com"
OPENAI_JUDGE_MODEL = "gpt-5.5"
OPENAI_JUDGE_PROMPT_CACHE_NAMESPACE = "cagap-gpt55-judge-v1"
# The current GPT-5.5 substitute-judge workload is 5,373 rows across the
# GLM-4.7-Flash and Mistral-Medium subjects. 512 shards keeps the average
# requests per cache key below OpenAI's 15/minute guidance even if the full
# combined workload is accepted in a short burst, while preserving repeated
# prefixes within each shard.
OPENAI_JUDGE_PROMPT_CACHE_SHARDS = 512
MISTRAL_MODEL = "mistral-large-2512"
XAI_MODEL = "grok-4.3"
DEFAULT_MAX_TOKENS = 4096

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
load_cagap_env()


def _dispatch_route_hint(judge_id: str, roster: dict[str, Any] | None = None) -> str:
    """Describe prepared judge route options from config/roster.json."""
    if roster is None:
        try:
            roster = judge_panel_resolver.load_roster()
        except Exception:
            roster = {}
    judges = roster.get("judges") if isinstance(roster, dict) else None
    entry = judges.get(judge_id) if isinstance(judges, dict) else None
    if not isinstance(entry, dict):
        return "manual_or_unknown"

    routes: list[str] = []
    if entry.get("native"):
        routes.append("native")
    if entry.get("local"):
        routes.append("local")
    if entry.get("openrouter"):
        routes.append("openrouter")
    if not routes:
        return "manual_or_unknown"
    role = str(entry.get("role", ""))
    prefix = (
        "prepared_substitute_route"
        if role.startswith("substitute")
        else "prepared_route_resolved_by_roster"
    )
    return prefix + "_" + "_".join(routes)


def _roster_decoding_for(model_id: str, roster: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return protocol decoding fields from config/roster.json."""
    if roster is None:
        roster = judge_panel_resolver.load_roster()
    decoding = roster.get("decoding") if isinstance(roster, dict) else {}
    default = decoding.get("default") if isinstance(decoding, dict) else {}
    if not isinstance(default, dict):
        default = {}
    body = dict(default)
    overrides = decoding.get("overrides") if isinstance(decoding, dict) else {}
    model_overrides = overrides.get(model_id) if isinstance(overrides, dict) else {}
    if isinstance(model_overrides, dict):
        for key, value in model_overrides.items():
            if value is None:
                body.pop(key, None)
            else:
                body[key] = value
    return body


# ---------------------------------------------------------------------------
# Subject response helpers (mirror grade_subject_responses.py minimal subset)
# ---------------------------------------------------------------------------

def _raw_response(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_response")
    if not isinstance(raw, dict):
        raw = row.get("raw_api_response")
    # Moonshot/Kimi native batch puts the chat completion at the top of the
    # row under ``response.body`` (not ``raw_response``). Unwrap so the rest
    # of the dispatcher sees an OpenAI-shaped dict.
    if not isinstance(raw, dict) or not (raw.get("choices") or raw.get("content")):
        resp = row.get("response")
        if isinstance(resp, dict):
            body = resp.get("body")
            if isinstance(body, dict) and (body.get("choices") or body.get("content")):
                return body
            if resp.get("choices") or resp.get("content"):
                return resp
    return raw if isinstance(raw, dict) else {}


def _subject_response_text(raw: dict[str, Any]) -> str:
    return subject_response_from_response(raw if isinstance(raw, dict) else {})


def _original_prompt(row: dict[str, Any], prompt_lookup: dict[str, dict[str, Any]]) -> str:
    for key in ("prompt", "prompt_text", "prompt_explicit"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    case_id = row.get("case_card_id")
    style = row.get("style_slot")
    if case_id and style:
        rec = prompt_lookup.get(f"{case_id}_{style}")
        if isinstance(rec, dict):
            for key in ("prompt", "prompt_text", "prompt_explicit"):
                val = rec.get(key)
                if isinstance(val, str) and val:
                    return val
    cid = row.get("custom_id")
    if isinstance(cid, str) and "__" in cid:
        parts = cid.split("__")
        if len(parts) >= 3:
            rec = prompt_lookup.get(parts[1])
            if isinstance(rec, dict):
                for key in ("prompt", "prompt_text", "prompt_explicit"):
                    val = rec.get(key)
                    if isinstance(val, str) and val:
                        return val
    return ""


# ---------------------------------------------------------------------------
# Phase A: extraction
# ---------------------------------------------------------------------------

def _extract_one(extractor: Extractor, custom_id: str, subject_response: str) -> dict[str, Any]:
    if not subject_response.strip():
        return {"custom_id": custom_id, "actions": [], "refusal_only": False,
                "refusal_signaled": False, "extraction_error": "empty_response"}
    try:
        result = extractor.extract(subject_response)
    except Exception as exc:  # noqa: BLE001
        try:
            result = extractor.extract(subject_response, bypass_cache=True)
        except Exception as exc2:  # noqa: BLE001
            return {"custom_id": custom_id, "actions": [], "refusal_only": False,
                    "refusal_signaled": False,
                    "extraction_error": f"{type(exc).__name__}/{type(exc2).__name__}: {exc2}"}
    actions = result.get("actions") if isinstance(result, dict) else []
    if not isinstance(actions, list):
        actions = []
    return {
        "custom_id": custom_id,
        "actions": actions,
        "refusal_only": bool(result.get("refusal_only") if isinstance(result, dict) else False),
        "refusal_signaled": bool(result.get("refusal_signaled") if isinstance(result, dict) else False),
        "extractor_output": result if isinstance(result, dict) else None,
    }


def extract_all(
    rows: list[dict[str, Any]],
    *,
    workers: int = 5,
    extractor: "Extractor | None" = None,
) -> list[dict[str, Any]]:
    """Run Phase A extraction concurrently across ``rows``.

    Pass ``extractor`` to share an instance across multiple Phase A runs
    (e.g. when re-extracting under a non-default cache directory). When
    omitted a default ``Extractor()`` is constructed and not closed by the
    caller.
    """
    owns_extractor = extractor is None
    if owns_extractor:
        extractor = Extractor()
    results: list[dict[str, Any] | None] = [None] * len(rows)
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="extract") as pool:
            futures = {}
            for idx, row in enumerate(rows):
                cid = row.get("custom_id") or f"row_{idx}"
                text = _subject_response_text(_raw_response(row))
                futures[pool.submit(_extract_one, extractor, cid, text)] = idx
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
                done += 1
                if done % 10 == 0 or done == len(rows):
                    logger.info("extracted %d/%d", done, len(rows))
    finally:
        if owns_extractor:
            try:
                extractor.close()
            except Exception:  # noqa: BLE001
                pass
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Phase B: judge prompts
# ---------------------------------------------------------------------------

def render_judge_prompt(template: str, prompt: str, actions: Any, response: str) -> str:
    out = template.replace("{{prompt}}", prompt)
    out = out.replace("{{actions}}", json.dumps(actions, indent=2))
    out = out.replace("{{response}}", response)
    return out


def _predicate_resolves_row(row: dict[str, Any], extraction: dict[str, Any]) -> tuple[bool, str]:
    """True when the grader's predicate path (grade_row) will resolve this
    row without consulting judges. Returns (resolved, reason).

    Mirrors three predicate gates in scripts/grade_subject_responses.py:
      1. ``has_structured_tool_calls`` on raw_response → provider tool calls.
      2. extractor flagged ``refusal_only=True`` AND ``refusal_signaled=True``
         AND no actions → deterministic_clear_refusal.
      3. ``is_complete_tool_call_dict`` on extracted actions → predicate
         authoritative (with or without authoritative-routes resolution; in
         the worst case grader emits ``Unresolved(extractor_error)`` —
         either way judges never fire).
    """
    raw = _raw_response(row)
    if has_structured_tool_calls(raw):
        return True, "structured_tool_calls"
    subject_response = subject_response_from_response(raw)
    payload = _demote_clear_refusal_on_operational_tail(
        _extraction_payload_for_routing(extraction),
        subject_response,
    )
    refusal_only = bool(payload.get("refusal_only"))
    refusal_signaled = bool(payload.get("refusal_signaled"))
    actions = payload.get("actions") or []
    if refusal_only and refusal_signaled and not actions and not payload.get("parse_error"):
        return True, "deterministic_clear_refusal"
    if is_complete_tool_call_dict(actions):
        return True, "complete_tool_call_dict"
    return False, ""


def build_inputs(rows: list[dict[str, Any]],
                 extractions: list[dict[str, Any]],
                 prompt_lookup: dict[str, dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Returns (judge_inputs, predicate_resolved_summary).

    Only rows the predicate path cannot resolve land in ``judge_inputs``.
    The summary records every row's predicate decision so the merge step
    can audit resolution coverage.
    """
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    by_cid: dict[str, dict[str, Any]] = {}
    for idx, extraction in enumerate(extractions, start=1):
        cid = extraction.get("custom_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"extraction row {idx} missing non-empty custom_id")
        if cid in by_cid:
            raise ValueError(f"duplicate custom_id in extractions: {cid!r}")
        by_cid[cid] = extraction
    seen_rows: set[str] = set()
    inputs: list[dict[str, str]] = []
    summary: list[dict[str, Any]] = []
    for row in rows:
        cid = row.get("custom_id")
        if not isinstance(cid, str) or not cid:
            continue
        if cid in seen_rows:
            raise ValueError(f"duplicate custom_id in input rows: {cid!r}")
        seen_rows.add(cid)
        ext = by_cid.get(cid) or {"actions": []}
        resolved, reason = _predicate_resolves_row(row, ext)
        summary.append({"custom_id": cid, "predicate_resolved": resolved,
                        "predicate_reason": reason})
        if resolved:
            continue
        prompt = _original_prompt(row, prompt_lookup)
        response = _subject_response_text(_raw_response(row))
        payload = _extraction_payload_for_routing(ext)
        actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
        rendered = render_judge_prompt(template, prompt, actions, response)
        inputs.append({"custom_id": cid, "judge_prompt": rendered})
    return inputs, summary


# ---------------------------------------------------------------------------
# Phase C1: OpenAI native batch
# ---------------------------------------------------------------------------

def openai_judge_prompt_cache_key(
    custom_id: str,
    *,
    namespace: str = OPENAI_JUDGE_PROMPT_CACHE_NAMESPACE,
    shard_count: int = OPENAI_JUDGE_PROMPT_CACHE_SHARDS,
) -> str:
    """Return a stable, sharded prompt cache key for GPT-5.5 judge batches."""
    shard = openai_judge_prompt_cache_shard(custom_id, shard_count=shard_count)
    width = max(3, len(str(shard_count - 1)))
    return f"{namespace}:{shard:0{width}d}"


def openai_judge_prompt_cache_shard(
    custom_id: str,
    *,
    shard_count: int = OPENAI_JUDGE_PROMPT_CACHE_SHARDS,
) -> int:
    """Return the deterministic prompt-cache shard for a GPT-5.5 judge row."""
    if not isinstance(custom_id, str) or not custom_id:
        raise ValueError("custom_id must be a non-empty string")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    digest = hashlib.sha256(custom_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % shard_count


def split_openai_judge_cache_warmup(
    inputs: list[dict[str, str]],
    *,
    shard_count: int = OPENAI_JUDGE_PROMPT_CACHE_SHARDS,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Split judge rows into one warmup row per cache shard and the remainder."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    warmup_by_shard: dict[int, dict[str, str]] = {}
    remainder: list[dict[str, str]] = []
    per_shard_counts: dict[str, int] = {}
    seen_custom_ids: set[str] = set()
    for entry in inputs:
        custom_id = entry.get("custom_id") or ""
        if not custom_id:
            raise ValueError("judge input missing custom_id")
        if custom_id in seen_custom_ids:
            raise ValueError(f"duplicate judge input custom_id: {custom_id!r}")
        seen_custom_ids.add(custom_id)
        shard = openai_judge_prompt_cache_shard(custom_id, shard_count=shard_count)
        shard_key = str(shard)
        per_shard_counts[shard_key] = per_shard_counts.get(shard_key, 0) + 1
        if shard not in warmup_by_shard:
            warmup_by_shard[shard] = entry
        else:
            remainder.append(entry)

    warmup = [warmup_by_shard[shard] for shard in sorted(warmup_by_shard)]
    counts = list(per_shard_counts.values())
    summary = {
        "n_rows": len(inputs),
        "shard_count": shard_count,
        "used_shards": len(warmup_by_shard),
        "warmup_rows": len(warmup),
        "remainder_rows": len(remainder),
        "missing_shards": [shard for shard in range(shard_count) if shard not in warmup_by_shard],
        "per_shard_min": min(counts) if counts else 0,
        "per_shard_max": max(counts) if counts else 0,
        "per_shard_counts": per_shard_counts,
    }
    return warmup, remainder, summary


def submit_openai_judge_batch(
    inputs: list[dict[str, str]],
    *,
    batch_name: str,
    model_id: str = OPENAI_JUDGE_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompt_cache_namespace: str = OPENAI_JUDGE_PROMPT_CACHE_NAMESPACE,
    prompt_cache_shards: int = OPENAI_JUDGE_PROMPT_CACHE_SHARDS,
) -> dict[str, Any]:
    """Submit pre-rendered GPT-5.5 judge prompts through OpenAI native Batch.

    ``prompt_cache_key`` is attached to every request body so rows that share
    the fixed judge prompt prefix can hit OpenAI prompt caching while still
    spreading the 5,373-row substitute-judge workload across cache-key shards.
    """
    if not inputs:
        return {"submit_status": "error", "error": "no judge inputs to submit"}
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAPI_API_KEY")
    if not key:
        return {"submit_status": "error", "error": "OPENAI_API_KEY not set"}

    short_to_full: dict[str, str] = {}
    lines: list[str] = []
    for entry in inputs:
        custom_id = entry.get("custom_id") or ""
        judge_prompt = entry.get("judge_prompt") or ""
        if not custom_id:
            return {"submit_status": "error", "error": "judge input missing custom_id"}
        digest = hashlib.sha256(custom_id.encode("utf-8")).hexdigest()[:32]
        short_to_full[digest] = custom_id
        lines.append(
            json.dumps(
                {
                    "custom_id": digest,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_id,
                        "messages": [{"role": "user", "content": judge_prompt}],
                        "max_completion_tokens": max_tokens,
                        "prompt_cache_key": openai_judge_prompt_cache_key(
                            custom_id,
                            namespace=prompt_cache_namespace,
                            shard_count=prompt_cache_shards,
                        ),
                        "prompt_cache_retention": "24h",
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    headers = {"Authorization": f"Bearer {key}"}
    upload_url = f"{OPENAI_BASE}/v1/files"
    upload_request_body = {
        "purpose": "batch",
        "n_requests": len(inputs),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "prompt_cache_key_namespace": prompt_cache_namespace,
        "prompt_cache_key_shards": prompt_cache_shards,
    }
    try:
        upload_response = requests.post(
            upload_url,
            headers=headers,
            data={"purpose": "batch"},
            files={"file": (f"{batch_name}.jsonl", payload, "application/jsonl")},
            timeout=60,
        )
    except requests.RequestException as exc:
        return {"submit_status": "error", "error": f"file upload: {type(exc).__name__}: {exc}"}
    archive_response_body(
        response_body_bytes(upload_response),
        role="judge_batch_file_upload",
        model_id=model_id,
        route="native_batch",
        provider="openai",
        http_status=upload_response.status_code,
        url=upload_url,
        request_body=upload_request_body,
        source="requests.Response.content",
    )
    if upload_response.status_code >= 400:
        return {
            "submit_status": "error",
            "error": f"file upload HTTP {upload_response.status_code}: {upload_response.text[:300]}",
        }
    try:
        upload_payload = upload_response.json()
    except ValueError as exc:
        return {"submit_status": "error", "error": f"file upload non-json response: {exc}"}
    input_file_id = upload_payload.get("id") or upload_payload.get("file_id")
    if not isinstance(input_file_id, str) or not input_file_id.strip():
        return {"submit_status": "error", "error": f"file upload missing id: {upload_payload!r}"[:300]}

    create_url = f"{OPENAI_BASE}/v1/batches"
    batch_body = {
        "input_file_id": input_file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {
            "batch_name": batch_name[:128],
            "prompt_cache_key_namespace": prompt_cache_namespace,
            "prompt_cache_key_shards": str(prompt_cache_shards),
            "n_requests": str(len(inputs)),
            "max_completion_tokens": str(max_tokens),
        },
    }
    try:
        create_response = requests.post(
            create_url,
            headers={**headers, "Content-Type": "application/json"},
            json=batch_body,
            timeout=60,
        )
    except requests.RequestException as exc:
        return {
            "submit_status": "error",
            "error": f"batch create: {type(exc).__name__}: {exc}",
            "input_file_id": input_file_id,
        }
    archive_response_body(
        response_body_bytes(create_response),
        role="judge_batch_submit",
        model_id=model_id,
        route="native_batch",
        provider="openai",
        http_status=create_response.status_code,
        url=create_url,
        request_body=batch_body,
        source="requests.Response.content",
    )
    if create_response.status_code >= 400:
        return {
            "submit_status": "error",
            "error": f"batch create HTTP {create_response.status_code}: {create_response.text[:300]}",
            "input_file_id": input_file_id,
        }
    try:
        create_payload = create_response.json()
    except ValueError as exc:
        return {"submit_status": "error", "error": f"batch create non-json response: {exc}", "input_file_id": input_file_id}
    batch_id = create_payload.get("id") or create_payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        return {"submit_status": "error", "error": f"no batch_id from create: {create_payload!r}"[:300]}

    return {
        "submit_status": "queued",
        "provider": "openai",
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "model": model_id,
        "max_completion_tokens": max_tokens,
        "n_requests": len(inputs),
        "submitted_at": int(time.time()),
        "short_to_full_cid": short_to_full,
        "transport": "openai_native_batch",
        "prompt_cache_key_namespace": prompt_cache_namespace,
        "prompt_cache_key_shards": prompt_cache_shards,
        "prompt_cache_retention": "24h",
    }


# ---------------------------------------------------------------------------
# Phase C2: xAI HTTP batch
# ---------------------------------------------------------------------------

def submit_xai_batch(inputs: list[dict[str, str]], *, batch_name: str,
                     model_id: str = XAI_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    key = os.environ.get("X_API_KEY") or os.environ.get("XAI_API_KEY")
    if not key:
        return {"submit_status": "error", "error": "X_API_KEY/XAI_API_KEY not set"}

    short_to_full: dict[str, str] = {}
    lines: list[str] = []
    for entry in inputs:
        # xAI batch custom_id must be unique within the batch and short enough
        # for the provider; map hashed IDs back to full CAGAP custom_ids.
        # Many of our custom_ids share long prefixes that collide under
        # naive truncation, so we map cid → 16-byte sha256 prefix and stash
        # the bidirectional map for the fetcher to undo.
        digest = hashlib.sha256(entry["custom_id"].encode("utf-8")).hexdigest()[:32]
        short_to_full[digest] = entry["custom_id"]
        lines.append(
            json.dumps(
                {
                    "custom_id": digest,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_id,
                        "messages": [{"role": "user", "content": entry["judge_prompt"]}],
                        "max_tokens": max_tokens,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    upload_url = "https://api.x.ai/v1/files"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        upload_response = requests.post(
            upload_url,
            headers=headers,
            data={"purpose": "batch"},
            files={"file": (f"{batch_name}.jsonl", payload, "application/jsonl")},
            timeout=60,
        )
    except requests.RequestException as exc:
        return {"submit_status": "error", "error": f"file upload: {type(exc).__name__}: {exc}"}
    archive_response_body(
        response_body_bytes(upload_response),
        role="judge_batch_file_upload",
        model_id=model_id,
        route="native_batch",
        provider="xai",
        http_status=upload_response.status_code,
        url=upload_url,
        request_body={
            "purpose": "batch",
            "n_requests": len(inputs),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
        source="requests.Response.content",
    )
    if upload_response.status_code >= 400:
        return {"submit_status": "error", "error": f"file upload HTTP {upload_response.status_code}: {upload_response.text[:300]}"}
    try:
        upload_payload = upload_response.json()
    except ValueError as exc:
        return {"submit_status": "error", "error": f"file upload non-json response: {exc}"}
    input_file_id = upload_payload.get("id") or upload_payload.get("file_id")
    if not isinstance(input_file_id, str) or not input_file_id.strip():
        return {"submit_status": "error", "error": f"file upload missing id: {upload_payload!r}"[:300]}

    create_url = "https://api.x.ai/v1/batches"
    batch_body = {"name": batch_name, "input_file_id": input_file_id}
    try:
        create_response = requests.post(
            create_url,
            headers={**headers, "Content-Type": "application/json"},
            json=batch_body,
            timeout=60,
        )
    except requests.RequestException as exc:
        return {
            "submit_status": "error",
            "error": f"batch create: {type(exc).__name__}: {exc}",
            "input_file_id": input_file_id,
        }
    archive_response_body(
        response_body_bytes(create_response),
        role="judge_batch_submit",
        model_id=model_id,
        route="native_batch",
        provider="xai",
        http_status=create_response.status_code,
        url=create_url,
        request_body=batch_body,
        source="requests.Response.content",
    )
    if create_response.status_code >= 400:
        return {
            "submit_status": "error",
            "error": f"batch create HTTP {create_response.status_code}: {create_response.text[:300]}",
            "input_file_id": input_file_id,
        }
    try:
        create_payload = create_response.json()
    except ValueError as exc:
        return {"submit_status": "error", "error": f"batch create non-json response: {exc}", "input_file_id": input_file_id}
    batch_id = create_payload.get("batch_id") or create_payload.get("id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        return {"submit_status": "error", "error": f"no batch_id from create: {create_payload!r}"[:300]}

    return {
        "submit_status": "queued",
        "provider": "xai",
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "model": model_id,
        "n_requests": len(inputs),
        "submitted_at": int(time.time()),
        "short_to_full_cid": short_to_full,
        "transport": "http_file_batch",
    }


# ---------------------------------------------------------------------------
# Phase C3: Mistral batch (OpenAI-compat JSONL via /v1/batch/jobs)
# ---------------------------------------------------------------------------

def submit_mistral_batch(inputs: list[dict[str, str]], *, model_id: str = MISTRAL_MODEL,
                         max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    if not inputs:
        return {"submit_status": "error", "error": "no judge inputs to submit"}
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        return {"submit_status": "error", "error": "MISTRAL_API_KEY not set"}
    headers = {"Authorization": f"Bearer {key}"}

    lines = []
    short_to_full: dict[str, str] = {}
    decoding = _roster_decoding_for(model_id)
    protocol_max_tokens = decoding.get("max_tokens", DEFAULT_MAX_TOKENS)
    if max_tokens != protocol_max_tokens:
        return {
            "submit_status": "error",
            "error": (
                f"max_tokens={max_tokens} deviates from roster protocol "
                f"max_tokens={protocol_max_tokens}"
            ),
        }
    decoding["max_tokens"] = protocol_max_tokens
    for entry in inputs:
        # Mistral allows longer custom_ids than xAI but the same hashing
        # discipline keeps the votes-merge step consistent across providers.
        digest = hashlib.sha256(entry["custom_id"].encode("utf-8")).hexdigest()[:32]
        short_to_full[digest] = entry["custom_id"]
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": entry["judge_prompt"]}],
            **decoding,
        }
        lines.append(
            json.dumps(
                {"custom_id": digest, "body": body},
                ensure_ascii=True,
                sort_keys=True,
                allow_nan=False,
            )
        )
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    try:
        fr = requests.post(
            f"{MISTRAL_BASE}/v1/files",
            headers=headers,
            files={"file": ("judge_input.jsonl", io.BytesIO(payload), "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
        archive_response_body(
            response_body_bytes(fr),
            role="judge_batch_file_upload",
            model_id=model_id,
            route="native",
            provider="mistral",
            http_status=getattr(fr, "status_code", None),
            url=f"{MISTRAL_BASE}/v1/files",
            request_body={
                "purpose": "batch",
                "n_requests": len(lines),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "model": model_id,
                "decoding": decoding,
            },
            source="requests.Response.content",
        )
    except Exception as exc:  # noqa: BLE001
        return {"submit_status": "error", "error": f"file upload: {type(exc).__name__}: {exc}"}
    if fr.status_code not in (200, 201):
        return {"submit_status": "error", "error": f"file HTTP {fr.status_code}: {fr.text[:300]}"}
    file_id = fr.json().get("id")
    if not file_id:
        return {"submit_status": "error", "error": f"file no id: {fr.text[:300]}"}

    try:
        batch_body = {
            "input_files": [file_id],
            "endpoint": "/v1/chat/completions",
            "model": model_id,
            "metadata": {
                "max_tokens": str(decoding.get("max_tokens")),
                "temperature": str(decoding.get("temperature")),
                "top_p": str(decoding.get("top_p")),
            },
        }
        br = requests.post(
            f"{MISTRAL_BASE}/v1/batch/jobs",
            headers={**headers, "Content-Type": "application/json"},
            json=batch_body,
            timeout=120,
        )
        archive_response_body(
            response_body_bytes(br),
            role="judge_batch_submit",
            model_id=model_id,
            route="native",
            provider="mistral",
            http_status=getattr(br, "status_code", None),
            url=f"{MISTRAL_BASE}/v1/batch/jobs",
            request_body=batch_body,
            source="requests.Response.content",
        )
    except Exception as exc:  # noqa: BLE001
        return {"submit_status": "error", "error": f"batch submit: {type(exc).__name__}: {exc}",
                "input_file_id": file_id}
    if br.status_code not in (200, 201):
        return {"submit_status": "error", "error": f"batch HTTP {br.status_code}: {br.text[:400]}",
                "input_file_id": file_id}
    bj = br.json()
    return {
        "submit_status": "queued",
        "provider": "mistral",
        "batch_id": bj.get("id"),
        "input_file_id": file_id,
        "model": model_id,
        "max_tokens": decoding.get("max_tokens"),
        "temperature": decoding.get("temperature"),
        "top_p": decoding.get("top_p"),
        "n_requests": len(lines),
        "submitted_at": int(time.time()),
        "short_to_full_cid": short_to_full,
        "transport": "mistral_native_batch",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="Subject response JSONL (e.g. responses_<subject>.jsonl)")
    p.add_argument("--prompts", required=True, type=Path,
                   help="Prompts file with case_card_id+style_slot lookup")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output dir (e.g. outputs/pilot/2026-05-08/judge_batches/<subject>)")
    p.add_argument("--subject-id", required=True)
    p.add_argument(
        "--extractions",
        type=Path,
        help=(
            "Precomputed MiniMax extraction JSONL to reuse. Pilot-2 expects "
            "this to come from scripts/run_minimax_batch_extraction.py."
        ),
    )
    p.add_argument(
        "--allow-online-extraction",
        action="store_true",
        help=(
            "Developer escape hatch: if --extractions is omitted, run the "
            "inline online Extractor. Pilot-2 should not use this."
        ),
    )
    p.add_argument("--extract-workers", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument(
        "--judges",
        default="auto",
        help=(
            "Comma-separated judges to prepare inputs for, or 'auto' to apply "
            "the roster's subject-level family-excluded judge panel."
        ),
    )
    p.add_argument(
        "--roster",
        type=Path,
        default=REPO_ROOT / "config" / "roster.json",
        help="Roster JSON used when --judges=auto.",
    )
    return p.parse_args()


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(strict=False)
    idx = 0
    row_no = 0
    out: list[dict[str, Any]] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        row, idx = decoder.raw_decode(text, idx)
        row_no += 1
        if not isinstance(row, dict):
            raise ValueError(f"{path}: decoded row {row_no} must be an object")
        out.append(row)
    return out


def load_prompt_lookup(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in _jsonl_objects(path):
        case_id = r.get("case_card_id")
        style = r.get("style_slot")
        if case_id and style:
            out[f"{case_id}_{style}"] = r
        if case_id:
            out.setdefault(case_id, r)
    return out


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    judge_panel: dict[str, Any] | None = None
    if args.judges.strip().lower() == "auto":
        roster = judge_panel_resolver.load_roster(args.roster)
        judge_panel = judge_panel_resolver.resolve_judge_panel(
            roster, args.subject_id
        )
        judges = list(judge_panel["judge_ids"])
    else:
        judges = [j.strip() for j in args.judges.split(",") if j.strip()]

    rows = _jsonl_objects(args.input)
    logger.info("loaded %d rows from %s", len(rows), args.input)
    prompt_lookup = load_prompt_lookup(args.prompts)
    logger.info("loaded %d prompt entries", len(prompt_lookup))

    # Phase A
    row_custom_ids = [
        row.get("custom_id")
        for row in rows
        if isinstance(row.get("custom_id"), str) and row.get("custom_id")
    ]
    if len(row_custom_ids) != len(set(row_custom_ids)):
        dupes = sorted(
            cid for cid in set(row_custom_ids) if row_custom_ids.count(cid) > 1
        )
        raise SystemExit(f"input duplicate custom_id(s): {dupes[:5]}")
    if args.extractions:
        logger.info("phase A: reusing precomputed extractions from %s", args.extractions)
        try:
            extraction_by_cid = load_precomputed_extractions(
                args.extractions,
                input_paths=[args.input],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        extractions = [extraction_by_cid[cid] for cid in row_custom_ids]
    else:
        if not args.allow_online_extraction:
            raise SystemExit(
                "Pilot-2 requires --extractions from MiniMax async batch. "
                "Pass --allow-online-extraction only for developer/debug runs."
            )
        logger.info("phase A: extracting actions (workers=%d)…", args.extract_workers)
        extractions = extract_all(rows, workers=args.extract_workers)
    extractions_path = out_dir / "extractions.jsonl"
    with extractions_path.open("w", encoding="utf-8") as fh:
        for e in extractions:
            fh.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info("wrote %s (%d rows)", extractions_path, len(extractions))

    # Phase B
    logger.info("phase B: building judge prompts (predicate-aware)…")
    inputs, predicate_summary = build_inputs(rows, extractions, prompt_lookup)
    n_predicate = sum(1 for s in predicate_summary if s["predicate_resolved"])
    inputs_dir = out_dir / "judge_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "predicate_decisions.jsonl"
    with summary_path.open("w", encoding="utf-8") as fh:
        for s in predicate_summary:
            fh.write(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n")
    for jid in judges:
        p = inputs_dir / f"{jid}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for entry in inputs:
                fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info("predicate resolved %d/%d rows (skipped from judge dispatch)",
                n_predicate, len(predicate_summary))
    logger.info("wrote %d judge_inputs files (judge_rows=%d)",
                len(judges), len(inputs))

    # Phase C: record the intended per-judge dispatch route. Actual dispatch
    # is performed later by the per-route judge dispatchers.
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "subject_id": args.subject_id,
        "input_path": str(args.input),
        "extractions_path": str(extractions_path),
        "judge_inputs_dir": str(inputs_dir),
        "predicate_decisions_path": str(summary_path),
        "n_rows": len(rows),
        "n_predicate_resolved": n_predicate,
        "n_judge_rows": len(inputs),
        "judges": {},
    }
    if judge_panel is not None:
        manifest["judge_panel"] = judge_panel
    roster_for_routes = judge_panel_resolver.load_roster()
    for jid in judges:
        route_hint = _dispatch_route_hint(jid, roster_for_routes)
        if route_hint != "manual_or_unknown":
            res = {
                "submit_status": "prepared",
                "dispatch_route": route_hint,
                "judge_inputs_path": str(inputs_dir / f"{jid}.jsonl"),
            }
        else:
            res = {
                "submit_status": "prepared",
                "dispatch_route": "manual_or_unknown",
                "judge_inputs_path": str(inputs_dir / f"{jid}.jsonl"),
            }
        manifest["judges"][jid] = res
        logger.info("judge=%s submit_status=%s route=%s",
                    jid, res.get("submit_status"), res.get("dispatch_route"))

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("wrote manifest %s", manifest_path)
    print(json.dumps({"subject_id": args.subject_id, "manifest": str(manifest_path),
                      "judges": {k: v.get("submit_status") for k, v in manifest["judges"].items()}},
                     indent=2, sort_keys=True))
    n_err = sum(1 for v in manifest["judges"].values() if v.get("submit_status") == "error")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
