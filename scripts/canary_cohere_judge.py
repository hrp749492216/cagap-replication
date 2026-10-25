#!/usr/bin/env python3
"""Run native Cohere Command A judge votes over prepared judge inputs.

The main judge coordinator uses this script for the native Command A lane.
Per-row vote JSONL output is compatible with ``grade_subject_responses.py`` and
``merge_grading_outputs.py``. Responses are archived internally before parsing
when raw HTTP capture is enabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests

from extractor.extract import subject_response_from_response  # noqa: E402
from grader.json_strict import reject_duplicate_keys, reject_nonfinite, safe_float  # noqa: E402
from judges.judge import (  # noqa: E402
    JudgeInfrastructureError,
    MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS,
    MAX_JUDGE_TRANSPORT_ATTEMPTS,
)
from scripts.cagap_env import load_cagap_env  # noqa: E402
from scripts.raw_http_archive import archive_response_body, response_body_bytes  # noqa: E402

load_cagap_env()

PROMPT_TEMPLATE_PATH = REPO_ROOT / "judges" / "prompt.md"
# Backwards-compatible alias for the subject-output budget. Transport
# failures retry against the dedicated MAX_JUDGE_TRANSPORT_ATTEMPTS
# budget and raise JudgeInfrastructureError on exhaustion.
MAX_JUDGE_ATTEMPTS = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS
COHERE_JUDGE_MAX_TOKENS = 4096
VALID_LABELS = {"Refuse", "Partial", "Comply"}

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("canary_cohere_judge")


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(
        strict=False,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
        parse_float=safe_float,
    )
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_hashes(entries: list[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for entry in entries:
        cid = entry.get("custom_id")
        if not isinstance(cid, str) or not cid:
            continue
        if cid in hashes:
            raise ValueError(f"duplicate custom_id in judge inputs: {cid!r}")
        hashes[cid] = _sha256_text(str(entry.get("judge_prompt") or ""))
    return hashes


def _vote_is_valid_for_run(
    row: dict[str, Any],
    *,
    judge_id: str,
    model: str,
    provider: str,
    prompt_hashes: dict[str, str] | None,
) -> bool:
    if row.get("judge") != judge_id:
        return False
    if row.get("model") != model:
        return False
    if row.get("provider") != provider:
        return False
    if row.get("label") not in VALID_LABELS:
        return False
    if prompt_hashes is not None:
        cid = row.get("custom_id")
        if not isinstance(cid, str) or prompt_hashes.get(cid) != row.get("judge_prompt_sha256"):
            return False
    return True


class CohereJudgeClient:
    def __init__(self, model: str = "command-a-03-2025", min_interval_seconds: float = 3.5) -> None:
        self.model = model
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY not set")
        self.endpoint = "https://api.cohere.com/v2/chat"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        self._template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        self._min_interval = max(0.0, float(min_interval_seconds))
        self._pace_lock = threading.Lock()
        self._next_slot = 0.0  # monotonic timestamp of next allowed call

    def _wait_for_slot(self) -> None:
        if self._min_interval <= 0:
            return
        with self._pace_lock:
            now = time.monotonic()
            if now < self._next_slot:
                wait = self._next_slot - now
                time.sleep(wait)
                self._next_slot = time.monotonic() + self._min_interval
            else:
                self._next_slot = now + self._min_interval

    def evaluate(
        self,
        prompt: str,
        actions: Any,
        response: str,
        *,
        custom_id: str | None = None,
    ) -> dict[str, Any]:
        msg = self._template.replace("{{prompt}}", prompt or "")
        msg = msg.replace("{{actions}}", json.dumps(actions, indent=2, ensure_ascii=False))
        msg = msg.replace("{{response}}", response or "")
        return self.evaluate_rendered(msg, custom_id=custom_id)

    def evaluate_rendered(self, judge_prompt: str, *, custom_id: str | None = None) -> dict[str, Any]:
        return self._evaluate_message(judge_prompt, custom_id=custom_id)

    def _evaluate_message(self, msg: str, *, custom_id: str | None = None) -> dict[str, Any]:
        last_exc: str | None = None
        attempts: list[dict[str, Any]] = []
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": msg}],
            "temperature": 0.0,
            "max_tokens": COHERE_JUDGE_MAX_TOKENS,
            "stream": False,
        }
        subject_invalid_count = 0
        transport_count = 0
        last_transport_error = ""
        max_total = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS + MAX_JUDGE_TRANSPORT_ATTEMPTS
        attempt = 0
        while attempt < max_total:
            attempt += 1
            text = ""
            parsed = {"label": "Unresolved", "reason": ""}
            invalid_reason: str | None = None
            raw_response: dict[str, Any] = {}
            finish_reason = None
            retry_after_seconds: float | None = None
            http_status: int | None = None
            raw_http_capture: dict[str, Any] | None = None
            try:
                self._wait_for_slot()
                resp = self.session.post(
                    self.endpoint,
                    headers=self.headers,
                    json=body,
                    timeout=180,
                )
                http_status = resp.status_code
                raw_http_capture = archive_response_body(
                    response_body_bytes(resp),
                    role="judge",
                    model_id=self.model,
                    route="native",
                    provider="cohere",
                    custom_id=custom_id,
                    attempt_index=attempt,
                    http_status=http_status,
                    url=self.endpoint,
                    request_body=body,
                    source="requests.Response.content",
                )
                if resp.headers.get("retry-after"):
                    try:
                        retry_after_seconds = float(resp.headers["retry-after"])
                    except ValueError:
                        retry_after_seconds = None
                if resp.status_code != 200:
                    raw_response = {"status_code": resp.status_code, "body": resp.text[:1000]}
                    invalid_reason = "transport_error"
                else:
                    payload = resp.json()
                    raw_response = payload if isinstance(payload, dict) else {"response": payload}
                    message = raw_response.get("message") if isinstance(raw_response, dict) else {}
                    content = message.get("content") if isinstance(message, dict) else None
                    parts = []
                    if isinstance(content, str):
                        parts.append(content)
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict):
                                t = c.get("text")
                            else:
                                t = getattr(c, "text", None)
                            if isinstance(t, str) and t:
                                parts.append(t)
                    text = "".join(parts)
                    finish_reason = raw_response.get("finish_reason")
                parsed = self._parse(text)
                if invalid_reason is None and not text.strip():
                    invalid_reason = "empty_content"
                elif invalid_reason is None and parsed["label"] == "Unresolved":
                    invalid_reason = "invalid_or_missing_label"
            except Exception as exc:  # noqa: BLE001
                last_exc = f"{type(exc).__name__}: {exc}"
                raw_response = {"error": last_exc}
                invalid_reason = "transport_error"

            attempts.append(
                {
                    "judge_id": self.model,
                    "attempt_index": attempt,
                    "provider_route": "native",
                    "timestamp": _now_utc(),
                    "status": "valid" if invalid_reason is None else "invalid",
                    "http_status": http_status,
                    "finish_reason": finish_reason,
                    "content_chars": len(text),
                    "parsed_label": (
                        parsed["label"] if parsed["label"] != "Unresolved" else None
                    ),
                    "invalid_reason": invalid_reason,
                    "raw_response": raw_response,
                    "raw_http_capture": raw_http_capture,
                }
            )
            if invalid_reason is None:
                parsed["raw_text"] = text[:600]
                parsed["attempts"] = attempts
                return parsed
            if invalid_reason == "transport_error":
                transport_count += 1
                last_transport_error = (
                    str(raw_response.get("error", ""))
                    or (last_exc or f"http_status={http_status}")
                )
                if transport_count >= MAX_JUDGE_TRANSPORT_ATTEMPTS:
                    raise JudgeInfrastructureError(
                        judge_id=self.model,
                        attempts=attempts,
                        last_error=last_transport_error,
                    )
            else:
                subject_invalid_count += 1
                if subject_invalid_count >= MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS:
                    return {
                        "label": "Unresolved",
                        "reason": "judge_invalid_after_3_attempts",
                        "attempts": attempts,
                    }
            sleep_for = min(5 * attempt, 30)
            if retry_after_seconds is not None:
                sleep_for = max(sleep_for, retry_after_seconds)
            time.sleep(sleep_for)
        # Safety net: combined budget exhausted (unreachable under
        # current caps but guarantees a return value).
        return {
            "label": "Unresolved",
            "reason": "judge_invalid_after_3_attempts",
            "attempts": attempts,
        }

    @staticmethod
    def _parse(text: str) -> dict[str, str]:
        label = "Unresolved"
        reason = ""
        for ln in (text or "").strip().splitlines():
            s = ln.strip()
            if s.lower().startswith("label:"):
                v = s.split(":", 1)[1].strip().upper()
                v = v.split()[0] if v else v
                v = v.strip("*` ")
                if v in ("R", "P", "C"):
                    label = {"R": "Refuse", "P": "Partial", "C": "Comply"}[v]
            elif s.lower().startswith("reason:"):
                reason = s.split(":", 1)[1].strip()
        return {"label": label, "reason": reason or "(no reason parsed)"}


def _raw_response(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_response")
    if not isinstance(raw, dict):
        raw = row.get("raw_api_response")
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
    for key in ("prompt", "prompt_text", "prompt_explicit", "user"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    case_id = row.get("case_card_id")
    style = row.get("style_slot")
    if case_id and style:
        rec = prompt_lookup.get(f"{case_id}_{style}")
        if isinstance(rec, dict):
            for key in ("prompt", "prompt_text", "prompt_explicit", "user"):
                val = rec.get(key)
                if isinstance(val, str) and val:
                    return val
    cid = row.get("custom_id")
    if isinstance(cid, str) and "__" in cid:
        parts = cid.split("__")
        if len(parts) >= 3:
            rec = prompt_lookup.get(parts[1])
            if isinstance(rec, dict):
                for key in ("prompt", "prompt_text", "prompt_explicit", "user"):
                    val = rec.get(key)
                    if isinstance(val, str) and val:
                        return val
    return ""


def _judge_one(client, judge_id, row, extractions, prompt_lookup):
    cid = row.get("custom_id") or ""
    ext = extractions.get(cid) or {}
    actions = ext.get("actions") or []
    prompt = _original_prompt(row, prompt_lookup)
    response = _subject_response_text(_raw_response(row))
    try:
        try:
            result = client.evaluate(prompt, actions, response, custom_id=str(cid))
        except TypeError as exc:
            if "custom_id" not in str(exc):
                raise
            result = client.evaluate(prompt, actions, response)
        label = result.get("label") or "Unresolved"
        reason = result.get("reason") or ""
        raw_text = result.get("raw_text") or ""
        attempts = result.get("attempts") or []
    except Exception as exc:  # noqa: BLE001
        label = "Unresolved"
        reason = f"online_error: {type(exc).__name__}: {exc}"
        raw_text = ""
        attempts = []
    return {
        "custom_id": cid,
        "judge": judge_id,
        "label": label,
        "reason": reason,
        "raw_text": raw_text,
        "attempts": attempts,
        "source": "online",
        "provider": "cohere",
        "model": client.model,
    }


def _judge_rendered_one(client, judge_id, entry):
    cid = entry.get("custom_id") or ""
    judge_prompt = entry.get("judge_prompt") or ""
    try:
        try:
            result = client.evaluate_rendered(judge_prompt, custom_id=str(cid))
        except TypeError as exc:
            if "custom_id" not in str(exc):
                raise
            result = client.evaluate_rendered(judge_prompt)
        label = result.get("label") or "Unresolved"
        reason = result.get("reason") or ""
        raw_text = result.get("raw_text") or ""
        attempts = result.get("attempts") or []
    except Exception as exc:  # noqa: BLE001
        label = "Unresolved"
        reason = f"online_error: {type(exc).__name__}: {exc}"
        raw_text = ""
        attempts = []
    return {
        "custom_id": cid,
        "judge": judge_id,
        "label": label,
        "reason": reason,
        "raw_text": raw_text,
        "attempts": attempts,
        "source": "online",
        "provider": "cohere",
        "model": client.model,
        "judge_prompt_sha256": _sha256_text(str(judge_prompt)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--extractions", type=Path)
    p.add_argument("--prompts", type=Path)
    p.add_argument(
        "--judge-inputs",
        type=Path,
        help="Pre-rendered judge_inputs JSONL with {custom_id, judge_prompt}; when set, --input/--extractions/--prompts are not required.",
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--judge", default="command-a-03-2025")
    p.add_argument("--model", default="command-a-03-2025")
    p.add_argument("--row-workers", type=int, default=1)
    p.add_argument("--min-interval-seconds", type=float, default=3.5,
                   help="Global min seconds between Cohere chat calls (rate-limit pacing). Default 3.5s ~= 17 RPM.")
    p.add_argument(
        "--allow-invalid-completions",
        action="store_true",
        help=(
            "Return 0 even if one or more newly completed rows exhausted all "
            "invalid judge attempts. Confirmatory judge dispatch should leave "
            "this unset so partial judge health failures are visible."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt_hashes: dict[str, str] | None = None
    if args.judge_inputs is not None:
        entries = _jsonl_objects(args.judge_inputs)
        prompt_hashes = _prompt_hashes(entries)
        rows: list[dict[str, Any]] = []
        extractions: dict[str, dict[str, Any]] = {}
        prompt_lookup: dict[str, dict[str, Any]] = {}
    else:
        missing = [
            name for name, value in (
                ("--input", args.input),
                ("--extractions", args.extractions),
                ("--prompts", args.prompts),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "missing required arguments without --judge-inputs: " + ", ".join(missing)
            )
        entries = []
        rows = _jsonl_objects(args.input)
        extractions = {}
        for e in _jsonl_objects(args.extractions):
            extractions[e.get("custom_id") or ""] = e

        prompt_lookup = {}
        for r in _jsonl_objects(args.prompts):
            case_id = r.get("case_card_id")
            style = r.get("style_slot")
            if case_id and style:
                prompt_lookup[f"{case_id}_{style}"] = r
            if case_id:
                prompt_lookup.setdefault(case_id, r)

    client = CohereJudgeClient(model=args.model, min_interval_seconds=args.min_interval_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    out_tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    existing_ids: set[str] = set()
    valid_existing_rows: list[dict[str, Any]] = []
    ignored_existing_rows: list[dict[str, Any]] = []
    if args.out.exists():
        for row in _jsonl_objects(args.out):
            cid = row.get("custom_id")
            if (
                isinstance(cid, str)
                and cid
                and _vote_is_valid_for_run(
                    row,
                    judge_id=args.judge,
                    model=args.model,
                    provider="cohere",
                    prompt_hashes=prompt_hashes,
                )
            ):
                existing_ids.add(cid)
                valid_existing_rows.append(row)
            else:
                ignored_existing_rows.append(row)
    if out_tmp.exists():
        for row in _jsonl_objects(out_tmp):
            cid = row.get("custom_id")
            if (
                isinstance(cid, str)
                and cid
                and cid not in existing_ids
                and _vote_is_valid_for_run(
                    row,
                    judge_id=args.judge,
                    model=args.model,
                    provider="cohere",
                    prompt_hashes=prompt_hashes,
                )
            ):
                valid_existing_rows.append(row)
                existing_ids.add(cid)
            else:
                ignored_existing_rows.append(row)
    if ignored_existing_rows:
        ignored_path = args.out.with_suffix(args.out.suffix + ".ignored.jsonl")
        _append_jsonl(ignored_path, ignored_existing_rows)
        logger.warning(
            "ignored %d stale/invalid existing rows from %s; preserved in %s",
            len(ignored_existing_rows),
            args.out,
            ignored_path,
        )
    if ignored_existing_rows or out_tmp.exists():
        _write_jsonl(args.out, valid_existing_rows)
    done_count = 0
    invalid_count = 0
    work_items = entries if args.judge_inputs is not None else rows
    pending_items = [
        item for item in work_items
        if isinstance(item.get("custom_id"), str) and item["custom_id"] not in existing_ids
    ]
    with args.out.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.row_workers, thread_name_prefix="coh-row") as pool:
            if args.judge_inputs is not None:
                futures = {
                    pool.submit(_judge_rendered_one, client, args.judge, entry): i
                    for i, entry in enumerate(pending_items)
                }
            else:
                futures = {
                    pool.submit(_judge_one, client, args.judge, row, extractions, prompt_lookup): i
                    for i, row in enumerate(pending_items)
                }
            for fut in as_completed(futures):
                vote = fut.result()
                with write_lock:
                    fh.write(json.dumps(vote, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                done_count += 1
                invalid_count += 0 if vote.get("label") in VALID_LABELS else 1
                if done_count % 10 == 0 or done_count == len(pending_items):
                    logger.info(
                        "graded %d/%d rows this run (%d skipped existing)",
                        done_count,
                        len(pending_items),
                        len(existing_ids),
                    )
    logger.info("wrote %s (%d new rows, %d skipped existing)", args.out, done_count, len(existing_ids))
    print(json.dumps({
        "out": str(args.out),
        "rows_completed_this_run": done_count,
        "rows_skipped_existing": len(existing_ids),
        "rows_total": len(work_items),
        "judge": args.judge,
        "invalid_rows_completed_this_run": invalid_count,
        "valid_rows_completed_this_run": done_count - invalid_count,
    }, indent=2, sort_keys=True))
    if done_count > 0 and invalid_count > 0 and not args.allow_invalid_completions:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
