#!/usr/bin/env python3
"""Poll + fetch judge batches submitted by ``dispatch_judge_batches.py``.

Reads ``manifest.json`` and:
  - polls each judge's batch until terminal state (succeeded/failed/expired/cancelled)
  - downloads results
  - writes per-row votes JSONL: ``judge_votes.jsonl`` with rows of the form
    {"custom_id": ..., "votes": [{"judge": ..., "label": "C|R|P|Unresolved",
                                   "reason": ..., "source": "batch"}, ...]}

Idempotent: re-running picks up already-completed batches and merges in new
ones. The merged votes file accumulates votes across judges.

Per Amendment 13.J extension (2026-05-08).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.raw_http_archive import archive_response_body, response_body_bytes  # noqa: E402

MISTRAL_BASE = "https://api.mistral.ai"
OPENAI_BASE = "https://api.openai.com"
POLL_INTERVAL_S = 30
POLL_TIMEOUT_S = 24 * 3600

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(strict=False)
    idx = 0
    row_no = 0
    rows: list[dict[str, Any]] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        row, idx = decoder.raw_decode(text, idx)
        row_no += 1
        if not isinstance(row, dict):
            raise ValueError(f"{path}: decoded row {row_no} must be an object")
        rows.append(row)
    return rows


def _judge_prompt_hashes(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    hashes: dict[str, str] = {}
    for row in _jsonl_objects(path):
        custom_id = row.get("custom_id")
        judge_prompt = row.get("judge_prompt")
        if isinstance(custom_id, str) and custom_id and isinstance(judge_prompt, str):
            hashes[custom_id] = _sha256_text(judge_prompt)
    return hashes


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_record(
    *,
    judge_id: str,
    provider_route: str,
    label: str,
    text: str,
    raw_response: dict[str, Any],
    finish_reason: Any = None,
    http_status: Any = None,
    invalid_reason: str | None = None,
) -> dict[str, Any]:
    parsed_label = label if label in {"Refuse", "Partial", "Comply"} else None
    if invalid_reason is None and parsed_label is None:
        invalid_reason = "invalid_or_missing_label"
    return {
        "judge_id": judge_id,
        "attempt_index": 1,
        "provider_route": provider_route,
        "timestamp": _now_utc(),
        "status": "valid" if invalid_reason is None else "invalid",
        "http_status": http_status,
        "finish_reason": finish_reason,
        "content_chars": len(text or ""),
        "parsed_label": parsed_label,
        "invalid_reason": invalid_reason,
        "raw_response": raw_response,
    }


def _parse_judge_label(text: str) -> tuple[str, str]:
    label = "Unresolved"
    reason = ""
    if not isinstance(text, str):
        return label, reason
    for ln in text.strip().splitlines():
        s = ln.strip()
        if s.lower().startswith("label:"):
            v = s.split(":", 1)[1].strip().upper()
            if v in ("R", "P", "C"):
                label = {"R": "Refuse", "P": "Partial", "C": "Comply"}[v]
        elif s.lower().startswith("reason:"):
            reason = s.split(":", 1)[1].strip()
    return label, reason


def _xai_headers() -> dict[str, str]:
    key = os.environ.get("X_API_KEY") or os.environ.get("XAI_API_KEY")
    if not key:
        raise RuntimeError("X_API_KEY/XAI_API_KEY not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _openai_headers() -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAPI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------

def _mistral_headers() -> dict[str, str]:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


def poll_mistral(batch_id: str, *, deadline: float) -> dict[str, Any]:
    headers = _mistral_headers()
    while time.time() < deadline:
        r = requests.get(f"{MISTRAL_BASE}/v1/batch/jobs/{batch_id}", headers=headers, timeout=60)
        archive_response_body(
            response_body_bytes(r),
            role="judge_batch_poll",
            model_id="mistral-large-2512",
            route="native_batch",
            provider="mistral",
            http_status=r.status_code,
            url=f"{MISTRAL_BASE}/v1/batch/jobs/{batch_id}",
            source="requests.Response.content",
        )
        if r.status_code != 200:
            logger.warning("mistral poll HTTP %d: %s", r.status_code, r.text[:200])
            time.sleep(POLL_INTERVAL_S)
            continue
        pj = r.json()
        s = pj.get("status")
        logger.info("mistral batch %s state=%s succeeded=%d failed=%d total=%d",
                    batch_id[:12], s,
                    pj.get("succeeded_requests", 0), pj.get("failed_requests", 0),
                    pj.get("total_requests", 0))
        if s in ("SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED", "CANCELLATION_REQUESTED"):
            return pj
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"mistral batch {batch_id} did not terminate")


def fetch_mistral_results(batch_record: dict[str, Any]) -> list[dict[str, Any]]:
    headers = _mistral_headers()
    output_file_id = batch_record.get("output_file")
    if not output_file_id:
        return []
    r = requests.get(f"{MISTRAL_BASE}/v1/files/{output_file_id}/content",
                     headers=headers, timeout=120)
    archive_response_body(
        response_body_bytes(r),
        role="judge_batch_fetch",
        model_id="mistral-large-2512",
        route="native_batch",
        provider="mistral",
        http_status=r.status_code,
        url=f"{MISTRAL_BASE}/v1/files/{output_file_id}/content",
        source="requests.Response.content",
    )
    if r.status_code != 200:
        logger.warning("mistral output file HTTP %d: %s", r.status_code, r.text[:200])
        return []
    rows = []
    for ln in r.text.splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            logger.warning("mistral output bad json: %s", ln[:120])
    return rows


def _mistral_to_votes(rows: list[dict[str, Any]], judge_id: str,
                      short_to_full: dict[str, str] | None = None,
                      prompt_hashes: dict[str, str] | None = None,
                      model_id: str = "mistral-large-2512") -> list[dict[str, Any]]:
    votes = []
    for r in rows:
        short_cid = r.get("custom_id")
        if not isinstance(short_cid, str):
            continue
        cid = (short_to_full or {}).get(short_cid, short_cid)
        # Mistral wraps response under "response.body" (OpenAI-compatible)
        body = ((r.get("response") or {}).get("body") or {})
        http_status = (r.get("response") or {}).get("status_code")
        choices = body.get("choices") or []
        text = ""
        finish_reason = None
        if choices and isinstance(choices[0], dict):
            text = (choices[0].get("message") or {}).get("content") or ""
            finish_reason = choices[0].get("finish_reason")
        err = r.get("error") or body.get("error")
        if err:
            reason = f"batch_error: {json.dumps(err)[:200]}"
            vote = {"custom_id": cid,
                    "judge": judge_id,
                    "label": "Unresolved",
                    "reason": reason,
                    "source": "batch",
                    "provider": "mistral",
                    "model": model_id,
                    "attempts": [
                              _attempt_record(
                                  judge_id=judge_id,
                                  provider_route="batch",
                                  label="Unresolved",
                                  text=text,
                                  raw_response=body or {"error": err},
                                  finish_reason=finish_reason,
                                  http_status=http_status,
                                  invalid_reason="transport_error",
                              )
                          ]}
            prompt_hash = (prompt_hashes or {}).get(cid)
            if prompt_hash:
                vote["judge_prompt_sha256"] = prompt_hash
            votes.append(vote)
            continue
        label, reason = _parse_judge_label(text)
        input_tokens, output_tokens, cached_tokens = _openai_usage_counts(body)
        vote = {"custom_id": cid, "judge": judge_id, "label": label,
                "reason": reason or "(no reason parsed)", "source": "batch",
                "provider": "mistral", "model": model_id,
                "raw_text": text[:1000],
                "attempts": [
                    _attempt_record(
                        judge_id=judge_id,
                        provider_route="batch",
                        label=label,
                        text=text,
                        raw_response=body,
                        finish_reason=finish_reason,
                        http_status=http_status,
                    )
                ]}
        if input_tokens is not None:
            vote["input_tokens"] = input_tokens
        if output_tokens is not None:
            vote["output_tokens"] = output_tokens
        if cached_tokens is not None:
            vote["cached_input_tokens"] = cached_tokens
        prompt_hash = (prompt_hashes or {}).get(cid)
        if prompt_hash:
            vote["judge_prompt_sha256"] = prompt_hash
        votes.append(vote)
    return votes


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def poll_openai(batch_id: str, *, deadline: float) -> dict[str, Any]:
    headers = _openai_headers()
    while time.time() < deadline:
        r = requests.get(f"{OPENAI_BASE}/v1/batches/{batch_id}", headers=headers, timeout=60)
        archive_response_body(
            response_body_bytes(r),
            role="judge_batch_poll",
            model_id="gpt-5.5",
            route="native_batch",
            provider="openai",
            http_status=r.status_code,
            url=f"{OPENAI_BASE}/v1/batches/{batch_id}",
            source="requests.Response.content",
        )
        if r.status_code != 200:
            logger.warning("openai poll HTTP %d: %s", r.status_code, r.text[:200])
            time.sleep(POLL_INTERVAL_S)
            continue
        payload = r.json()
        status = payload.get("status")
        counts = payload.get("request_counts") if isinstance(payload.get("request_counts"), dict) else {}
        logger.info(
            "openai batch %s status=%s completed=%s failed=%s total=%s",
            batch_id[:12],
            status,
            counts.get("completed"),
            counts.get("failed"),
            counts.get("total"),
        )
        if status in {"completed", "failed", "expired", "cancelled"}:
            return payload
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"openai batch {batch_id} did not terminate")


def fetch_openai_results(batch_record: dict[str, Any]) -> list[dict[str, Any]]:
    headers = _openai_headers()
    output_file_id = batch_record.get("output_file_id") or batch_record.get("output_file")
    if not output_file_id:
        return []
    url = f"{OPENAI_BASE}/v1/files/{output_file_id}/content"
    r = requests.get(url, headers=headers, timeout=120)
    archive_response_body(
        response_body_bytes(r),
        role="judge_batch_fetch",
        model_id="gpt-5.5",
        route="native_batch",
        provider="openai",
        http_status=r.status_code,
        url=url,
        source="requests.Response.content",
    )
    if r.status_code != 200:
        logger.warning("openai output file HTTP %d: %s", r.status_code, r.text[:200])
        return []
    rows = []
    for ln in r.text.splitlines():
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            logger.warning("openai output bad json: %s", ln[:120])
    return rows


def _openai_usage_counts(body: dict[str, Any]) -> tuple[Any, Any, Any]:
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = usage.get("input_tokens_details")
    cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    return input_tokens, output_tokens, cached_tokens


def _openai_to_votes(rows: list[dict[str, Any]], judge_id: str,
                     short_to_full: dict[str, str] | None = None) -> list[dict[str, Any]]:
    votes = []
    for r in rows:
        short_cid = r.get("custom_id")
        if not isinstance(short_cid, str):
            continue
        cid = (short_to_full or {}).get(short_cid, short_cid)
        response = r.get("response") if isinstance(r.get("response"), dict) else {}
        body = response.get("body") if isinstance(response.get("body"), dict) else {}
        http_status = response.get("status_code")
        err = r.get("error") or response.get("error") or body.get("error")
        if err or (http_status is not None and int(http_status) >= 400):
            votes.append({"custom_id": cid,
                          "judge": judge_id,
                          "label": "Unresolved",
                          "reason": f"batch_error: {json.dumps(err or {'status_code': http_status})[:200]}",
                          "source": "batch",
                          "attempts": [
                              _attempt_record(
                                  judge_id=judge_id,
                                  provider_route="batch",
                                  label="Unresolved",
                                  text="",
                                  raw_response=body or {"error": err, "status_code": http_status},
                                  http_status=http_status,
                                  invalid_reason="transport_error",
                              )
                          ]})
            continue
        choices = body.get("choices") or []
        text = ""
        finish_reason = None
        if choices and isinstance(choices[0], dict):
            text = (choices[0].get("message") or {}).get("content") or ""
            finish_reason = choices[0].get("finish_reason")
        label, reason = _parse_judge_label(text)
        input_tokens, output_tokens, cached_tokens = _openai_usage_counts(body)
        vote = {"custom_id": cid, "judge": judge_id, "label": label,
                "reason": reason or "(no reason parsed)", "source": "batch",
                "raw_text": (text or "")[:1000],
                "attempts": [
                    _attempt_record(
                        judge_id=judge_id,
                        provider_route="batch",
                        label=label,
                        text=text or "",
                        raw_response=body,
                        finish_reason=finish_reason,
                        http_status=http_status,
                    )
                ]}
        if input_tokens is not None:
            vote["input_tokens"] = input_tokens
        if output_tokens is not None:
            vote["output_tokens"] = output_tokens
        if cached_tokens is not None:
            vote["cached_input_tokens"] = cached_tokens
        votes.append(vote)
    return votes


# ---------------------------------------------------------------------------
# xAI
# ---------------------------------------------------------------------------

def poll_and_fetch_xai(batch_id: str, judge_id: str, *, deadline: float,
                       short_to_full: dict[str, str] | None = None) -> list[dict[str, Any]]:
    headers = _xai_headers()
    base = "https://api.x.ai"

    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/v1/batches/{batch_id}", headers=headers, timeout=60)
        except requests.RequestException as exc:
            logger.warning("xai poll error: %s", exc)
            time.sleep(POLL_INTERVAL_S)
            continue
        archive_response_body(
            response_body_bytes(r),
            role="judge_batch_poll",
            model_id=judge_id,
            route="native_batch",
            provider="xai",
            http_status=r.status_code,
            url=f"{base}/v1/batches/{batch_id}",
            source="requests.Response.content",
        )
        if r.status_code != 200:
            logger.warning("xai poll HTTP %d: %s", r.status_code, r.text[:200])
            time.sleep(POLL_INTERVAL_S)
            continue
        cur = r.json()
        state = cur.get("state") if isinstance(cur.get("state"), dict) else {}
        n_pending = int(state.get("num_pending", -1) if isinstance(state, dict) else -1)
        n_total = int(state.get("num_requests", 0) if isinstance(state, dict) else 0)
        n_success = int(state.get("num_success", 0) if isinstance(state, dict) else 0)
        n_error = int(state.get("num_error", 0) if isinstance(state, dict) else 0)
        logger.info("xai batch %s pending=%d/%d success=%d error=%d",
                    batch_id[:12], n_pending, n_total, n_success, n_error)
        if n_pending == 0 and n_total > 0:
            break
        time.sleep(POLL_INTERVAL_S)
    else:
        raise TimeoutError(f"xai batch {batch_id} did not finish")

    votes: list[dict[str, Any]] = []
    next_page: str | None = None
    while True:
        query_args: dict[str, str] = {"limit": "1000"}
        if next_page:
            query_args["pagination_token"] = next_page
        result_url = f"{base}/v1/batches/{batch_id}/results?{urlencode(query_args)}"
        r = requests.get(result_url, headers=headers, timeout=120)
        archive_response_body(
            response_body_bytes(r),
            role="judge_batch_fetch",
            model_id=judge_id,
            route="native_batch",
            provider="xai",
            http_status=r.status_code,
            url=result_url,
            source="requests.Response.content",
        )
        if r.status_code != 200:
            raise RuntimeError(f"xai result fetch HTTP {r.status_code}: {r.text[:200]}")
        page = r.json()
        items = page.get("results") or page.get("data") or []
        if not isinstance(items, list):
            raise RuntimeError(f"xai result fetch returned non-list results: {type(items).__name__}")
        for item in items:
            if not isinstance(item, dict):
                continue
            short_cid = (
                item.get("custom_id")
                or item.get("batch_request_id")
                or item.get("id")
                or ""
            )
            if not short_cid:
                continue
            cid = (short_to_full or {}).get(short_cid, short_cid)
            response = item.get("response") if isinstance(item.get("response"), dict) else {}
            body = response.get("body") if isinstance(response.get("body"), dict) else None
            if body is None and isinstance(response, dict) and "choices" in response:
                body = response
            if body is None and isinstance(item.get("body"), dict):
                body = item["body"]
            body = body if isinstance(body, dict) else {}
            http_status = response.get("status_code") if isinstance(response, dict) else None
            err = item.get("error") or (response.get("error") if isinstance(response, dict) else None)
            if err or (http_status is not None and int(http_status) >= 400):
                votes.append({"custom_id": cid, "judge": judge_id,
                              "label": "Unresolved",
                              "reason": f"batch_error: {json.dumps(err or {'status_code': http_status})[:200]}",
                              "source": "batch",
                              "attempts": [
                                  _attempt_record(
                                      judge_id=judge_id,
                                      provider_route="batch",
                                      label="Unresolved",
                                      text="",
                                      raw_response=body or {"error": err, "status_code": http_status},
                                      http_status=http_status,
                                      invalid_reason="transport_error",
                                  )
                              ]})
                continue
            choices = body.get("choices") or []
            text = ""
            finish_reason = None
            if choices and isinstance(choices[0], dict):
                text = (choices[0].get("message") or {}).get("content") or ""
                finish_reason = choices[0].get("finish_reason")
            label, reason = _parse_judge_label(text)
            votes.append({"custom_id": cid, "judge": judge_id, "label": label,
                          "reason": reason or "(no reason parsed)", "source": "batch",
                          "raw_text": (text or "")[:1000],
                          "attempts": [
                              _attempt_record(
                                  judge_id=judge_id,
                                  provider_route="batch",
                                  label=label,
                                  text=text or "",
                                  raw_response=body,
                                  finish_reason=finish_reason,
                                  http_status=http_status,
                              )
                          ]})
        next_page = page.get("pagination_token") or page.get("next_page_token") or None
        if not next_page or not items:
            break
    return votes


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--out", default=None, type=Path,
                   help="Output votes JSONL (default: <manifest_dir>/judge_votes.jsonl)")
    p.add_argument("--poll-timeout-s", type=int, default=POLL_TIMEOUT_S)
    p.add_argument("--judges", default=None,
                   help="Comma-separated judges to fetch (default: all in manifest)")
    return p.parse_args()


def merge_votes(existing: list[dict[str, Any]],
                new_votes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group votes by custom_id; later votes for same (cid, judge) overwrite."""
    by_cid: dict[str, dict[str, Any]] = {}
    for v in existing:
        if isinstance(v, dict) and "custom_id" in v:
            by_cid[v["custom_id"]] = v
    for nv in new_votes:
        cid = nv["custom_id"]
        cur = by_cid.setdefault(cid, {"custom_id": cid, "votes": []})
        # remove any existing vote from same judge
        cur["votes"] = [vv for vv in cur.get("votes", []) if vv.get("judge") != nv["judge"]]
        cur["votes"].append({k: v for k, v in nv.items() if k != "custom_id"})
    return list(by_cid.values())


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out_path = args.out or args.manifest.parent / "judge_votes.jsonl"
    deadline = time.time() + args.poll_timeout_s
    requested_judges = (args.judges.split(",") if args.judges else None)

    existing: list[dict[str, Any]] = []
    if out_path.exists():
        for ln in out_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                existing.append(json.loads(ln))
            except json.JSONDecodeError:
                pass

    new_votes: list[dict[str, Any]] = []
    failed_judges: list[str] = []
    for jid, info in (manifest.get("judges") or {}).items():
        if requested_judges and jid not in requested_judges:
            continue
        if info.get("submit_status") != "queued":
            logger.warning("judge=%s skip (submit_status=%s)", jid, info.get("submit_status"))
            failed_judges.append(jid)
            continue
        provider = info.get("provider") or ("xai" if "xai" in jid or "grok" in jid else "mistral")
        batch_id = info.get("batch_id")
        short_to_full = info.get("short_to_full_cid") or {}
        judge_inputs_path = info.get("judge_inputs_path")
        if isinstance(judge_inputs_path, str):
            prompt_hashes = _judge_prompt_hashes(Path(judge_inputs_path))
        else:
            prompt_hashes = _judge_prompt_hashes(
                Path(manifest.get("judge_inputs_dir", "")) / f"{jid}.jsonl"
                if manifest.get("judge_inputs_dir")
                else None
            )
        try:
            if provider == "mistral":
                pj = poll_mistral(batch_id, deadline=deadline)
                rows = fetch_mistral_results(pj)
                votes = _mistral_to_votes(
                    rows,
                    jid,
                    short_to_full=short_to_full,
                    prompt_hashes=prompt_hashes,
                    model_id=str(info.get("model") or "mistral-large-2512"),
                )
            elif provider == "openai":
                pj = poll_openai(batch_id, deadline=deadline)
                rows = fetch_openai_results(pj)
                votes = _openai_to_votes(rows, jid, short_to_full=short_to_full)
            elif provider == "xai":
                votes = poll_and_fetch_xai(batch_id, jid, deadline=deadline,
                                            short_to_full=short_to_full)
            else:
                logger.warning("judge=%s unknown provider %s", jid, provider)
                failed_judges.append(jid)
                continue
        except Exception as exc:  # noqa: BLE001
            logger.error("judge=%s fetch failed: %s", jid, exc)
            failed_judges.append(jid)
            continue
        logger.info("judge=%s fetched %d votes", jid, len(votes))
        new_votes.extend(votes)

    merged = merge_votes(existing, new_votes)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in sorted(merged, key=lambda x: x.get("custom_id", "")):
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    logger.info("wrote %s (%d rows)", out_path, len(merged))
    print(json.dumps({"votes_path": str(out_path), "rows": len(merged),
                      "failed_judges": failed_judges},
                     indent=2, sort_keys=True))
    return 1 if failed_judges else 0


if __name__ == "__main__":
    sys.exit(main())
