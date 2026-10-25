#!/usr/bin/env python3
"""Run MiniMax M2.7 native extraction over response JSONL files.

MiniMax text chat currently accepts ``async: true`` but may return a complete
``chat.completion`` payload directly from the submit call. This script keeps a
submit ledger for auditability, parses immediate completions when present, and
polls only task-only responses. It writes both:

  - an extraction JSONL for judge-input preparation; and
  - a warmed ``ExtractorCache`` directory so ``grade_subject_responses.py`` can
    reuse exactly the same MiniMax output instead of issuing inline calls.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractor.cache import ExtractorCache  # noqa: E402
from extractor.extract import Extractor, subject_response_from_response  # noqa: E402
from scripts.cagap_env import load_cagap_env  # noqa: E402
from scripts.grade_subject_responses import (  # noqa: E402
    extraction_provenance_for_subject_response,
    raw_response_from_row,
)
from scripts.run_synthetic_prompts import Config, decoding_for, select_route  # noqa: E402

load_cagap_env()

TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
COMPLETE_STATES = {"success", "succeeded", "completed", "complete", "done", "finish", "finished"}
PENDING_STATES = {"", "pending", "queued", "queueing", "running", "processing", "started", "in_progress"}
ERROR_STATES = {"failed", "failure", "error", "errored", "cancelled", "canceled", "timeout"}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonl_rows(paths: list[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        decoder = json.JSONDecoder(strict=False)
        idx = 0
        row_no = 0
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            row, idx = decoder.raw_decode(text, idx)
            row_no += 1
            if not isinstance(row, dict):
                raise ValueError(f"{path}: decoded row {row_no} must be an object")
            yield row


def _load_by_custom_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(strict=False)
    idx = 0
    row_no = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        row, idx = decoder.raw_decode(text, idx)
        row_no += 1
        if not isinstance(row, dict):
            raise ValueError(f"{path}: decoded row {row_no} must be an object")
        cid = row.get("custom_id")
        if isinstance(cid, str) and cid:
            if cid in out:
                raise ValueError(f"{path}: duplicate custom_id {cid!r}")
            out[cid] = row
    return out


def _json_response(response: requests.Response) -> dict[str, Any]:
    content = getattr(response, "content", b"")
    if isinstance(content, bytes) and content:
        try:
            text = content.decode("utf-8")
            payload = json.loads(text)
        except UnicodeDecodeError:
            payload = response.json()
    else:
        payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("MiniMax response returned non-object JSON")
    return payload


def _index_rows_by_custom_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, start=1):
        cid = row.get("custom_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"input row {idx}: missing non-empty custom_id")
        if cid in out:
            raise ValueError(f"input row {idx}: duplicate custom_id {cid!r}")
        out[cid] = row
    return out


def _with_extraction_provenance(
    record: dict[str, Any],
    *,
    subject_response: str,
) -> dict[str, Any]:
    return {
        **record,
        **extraction_provenance_for_subject_response(subject_response),
    }


def _repair_completed_records(
    completed: dict[str, dict[str, Any]],
    *,
    rows_by_id: dict[str, dict[str, Any]],
) -> int:
    repaired = 0
    provenance_fields = tuple(
        extraction_provenance_for_subject_response("").keys()
    )
    for cid, record in completed.items():
        row = rows_by_id.get(cid)
        if row is None:
            raise ValueError(f"completed extraction custom_id {cid!r} is not in input rows")
        expected = extraction_provenance_for_subject_response(_row_subject_response(row))
        changed = False
        for field in provenance_fields:
            current = record.get(field)
            if current is None:
                record[field] = expected[field]
                changed = True
            elif current != expected[field]:
                raise ValueError(
                    f"{cid}: {field} mismatch: expected {expected[field]!r} got {current!r}"
                )
        if changed:
            repaired += 1
    return repaired


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _extract_task_id(payload: dict[str, Any]) -> str:
    candidates: list[Any] = [
        payload.get("task_id"),
        payload.get("request_id"),
    ]
    for key in ("data", "result", "task"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.extend([value.get("task_id"), value.get("request_id")])
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError(f"MiniMax async submit response missing task id: {str(payload)[:500]}")


def _status_from_payload(payload: dict[str, Any]) -> str:
    candidates: list[Any] = [
        payload.get("status"),
        payload.get("state"),
        payload.get("task_status"),
        payload.get("status_code"),
    ]
    for key in ("data", "result", "task"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.extend(
                [
                    value.get("status"),
                    value.get("state"),
                    value.get("task_status"),
                    value.get("status_code"),
                ]
            )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _candidate_dicts(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield payload
    for key in ("data", "result", "response", "output", "message"):
        value = payload.get(key)
        if isinstance(value, dict):
            yield value
            yield from _candidate_dicts(value)


def _openai_response_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in _candidate_dicts(payload):
        if isinstance(candidate.get("choices"), list):
            return candidate
    for candidate in _candidate_dicts(payload):
        for key in ("reply", "text", "content"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return {
                    "id": candidate.get("id") or payload.get("task_id"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": value},
                            "finish_reason": candidate.get("finish_reason") or candidate.get("finish_reason_type") or "stop",
                        }
                    ],
                    "usage": candidate.get("usage") or payload.get("usage") or {},
                }
    return None


def _content_from_response(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            finish_reason = choice.get("finish_reason")
            return (
                content if isinstance(content, str) else "",
                finish_reason if isinstance(finish_reason, str) else "",
            )
    return "", ""


class MiniMaxAsyncExtractor:
    def __init__(self, *, cache_dir: Path, timeout: float = 120.0) -> None:
        self.extractor_id = Config.EXTRACTOR_ID
        self.entry = Config.EXTRACTOR[self.extractor_id]
        route, route_cfg = select_route(self.entry)
        if route != "native":
            raise RuntimeError(
                f"MiniMax async extraction requires native route; roster resolved {route}"
            )
        if route_cfg.get("batch_mechanism") != "async_param":
            raise RuntimeError("MiniMax extractor native route is not async_param")
        key = os.environ.get(route_cfg["auth_env"])
        if not key:
            raise RuntimeError(f"{route_cfg['auth_env']} not set")
        self.route_cfg = route_cfg
        self.endpoint = f"{route_cfg['endpoint']}{route_cfg['batch_url_path']}"
        self.query_endpoint = f"{route_cfg['endpoint']}{route_cfg['batch_async_query_path']}"
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.session = requests.Session()
        self.timeout = timeout
        self.prompt_text = (REPO_ROOT / "extractor" / "prompt.md").read_text(encoding="utf-8")
        self.parser = Extractor.__new__(Extractor)
        self.parser.prompt_text = self.prompt_text
        self.cache = ExtractorCache(cache_dir=cache_dir)

    def close(self) -> None:
        self.session.close()

    def submit(self, custom_id: str, subject_response: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.route_cfg["slug"],
            "messages": [
                {"role": "system", "content": self.prompt_text},
                {"role": "user", "content": f"Subject Response:\n{subject_response}"},
            ],
            "async": True,
        }
        body.update(decoding_for(self.extractor_id))
        response = self.session.post(
            self.endpoint,
            headers=self.headers,
            json=body,
            timeout=self.timeout,
        )
        if response.status_code in TRANSIENT_STATUSES:
            raise RuntimeError(f"MiniMax async submit transient HTTP {response.status_code}: {response.text[:300]}")
        response.raise_for_status()
        payload = _json_response(response)
        immediate_response_id = payload.get("id") if _openai_response_from_payload(payload) else None
        task_id = (
            immediate_response_id
            if isinstance(immediate_response_id, str) and immediate_response_id.strip()
            else _extract_task_id(payload)
        )
        return {
            "custom_id": custom_id,
            "task_id": task_id,
            "submitted_at": _now_utc(),
            "submit_response": payload,
        }

    def query(self, task_id: str) -> dict[str, Any]:
        response = self.session.get(
            self.query_endpoint,
            headers=self.headers,
            params={"task_id": task_id},
            timeout=self.timeout,
        )
        if response.status_code in {404, 405}:
            response = self.session.post(
                self.query_endpoint,
                headers=self.headers,
                json={"task_id": task_id},
                timeout=self.timeout,
            )
        if response.status_code in {404, 405}:
            return {
                "status": "poll_endpoint_unavailable",
                "http_status": response.status_code,
                "body": response.text[:300],
            }
        if response.status_code in TRANSIENT_STATUSES:
            return {"_not_complete": True, "status": f"transient_http_{response.status_code}"}
        response.raise_for_status()
        payload = _json_response(response)
        return payload

    def parse_completed(
        self,
        *,
        custom_id: str,
        task_id: str,
        subject_response: str,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        status = _status_from_payload(payload)
        response = _openai_response_from_payload(payload)
        if response is None:
            if status in ERROR_STATES or status == "poll_endpoint_unavailable":
                return _with_extraction_provenance(
                    {
                        "custom_id": custom_id,
                        "actions": [],
                        "refusal_only": False,
                        "refusal_signaled": False,
                        "extraction_error": f"minimax_async_status:{status}",
                        "task_id": task_id,
                        "extractor_route": "minimax_native_async",
                    },
                    subject_response=subject_response,
                )
            if status in PENDING_STATES:
                return None
            return None

        text, finish_reason = _content_from_response(response)
        if not text.strip():
            return _with_extraction_provenance(
                {
                    "custom_id": custom_id,
                    "actions": [],
                    "refusal_only": False,
                    "refusal_signaled": False,
                    "extraction_error": "empty_extractor_completion",
                    "task_id": task_id,
                    "extractor_route": "minimax_native_async",
                    "finish_reason": finish_reason,
                },
                subject_response=subject_response,
            )
        clean_yaml = self.parser._strip_markdown(text)  # noqa: SLF001
        try:
            parsed = self.parser._parse_yaml(  # noqa: SLF001
                clean_yaml, subject_response=subject_response
            )
        except Exception as exc:  # noqa: BLE001
            return _with_extraction_provenance(
                {
                    "custom_id": custom_id,
                    "actions": [],
                    "refusal_only": False,
                    "refusal_signaled": False,
                    "extraction_error": f"parse_error:{type(exc).__name__}: {exc}",
                    "raw_text": text[:600],
                    "task_id": task_id,
                    "extractor_route": "minimax_native_async",
                    "finish_reason": finish_reason,
                },
                subject_response=subject_response,
            )
        self.cache.set(subject_response, self.prompt_text, clean_yaml)
        actions = parsed.get("actions") if isinstance(parsed, dict) else []
        if not isinstance(actions, list):
            actions = []
        return _with_extraction_provenance(
            {
                "custom_id": custom_id,
                "actions": actions,
                "refusal_only": bool(parsed.get("refusal_only") if isinstance(parsed, dict) else False),
                "refusal_signaled": bool(parsed.get("refusal_signaled") if isinstance(parsed, dict) else False),
                "extractor_output": parsed,
                "raw_text": text[:600],
                "task_id": task_id,
                "extractor_route": "minimax_native_async",
                "finish_reason": finish_reason,
            },
            subject_response=subject_response,
        )


def _row_subject_response(row: dict[str, Any]) -> str:
    raw_response, _source = raw_response_from_row(row)
    return subject_response_from_response(raw_response)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-poll-seconds", type=float, default=24 * 3600.0)
    parser.add_argument("--poll-workers", type=int, default=8)
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--poll-only", action="store_true")
    parser.add_argument(
        "--repair-provenance-only",
        action="store_true",
        help=(
            "Only stamp missing current provenance onto an existing complete "
            "extraction JSONL matched to --input, then exit. Mismatched "
            "existing provenance still fails."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks_path = args.tasks or args.out.with_suffix(args.out.suffix + ".tasks.jsonl")
    rows = list(_jsonl_rows(args.input))
    rows_by_id = _index_rows_by_custom_id(rows)
    completed = _load_by_custom_id(args.out)
    tasks = _load_by_custom_id(tasks_path)
    repaired = _repair_completed_records(completed, rows_by_id=rows_by_id)
    if repaired:
        _write_jsonl_atomic(
            args.out,
            [completed[cid] for cid in rows_by_id if cid in completed],
        )
        print(
            json.dumps(
                {
                    "status": "repaired_extraction_provenance",
                    "repaired": repaired,
                    "out": str(args.out),
                },
                sort_keys=True,
            )
        )
    if args.repair_provenance_only:
        print(
            json.dumps(
                {
                    "status": "repair_provenance_only",
                    "rows": len(rows_by_id),
                    "completed": len(completed),
                    "repaired": repaired,
                    "out": str(args.out),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if len(completed) == len(rows_by_id) else 1
    extractor = MiniMaxAsyncExtractor(cache_dir=args.cache_dir)
    try:
        def _parse_task_if_ready(cid: str, task: dict[str, Any]) -> dict[str, Any] | None:
            if cid in completed:
                return None
            row = rows_by_id.get(cid)
            if row is None:
                return None
            submit_response = task.get("submit_response")
            if not isinstance(submit_response, dict):
                return None
            subject_response = _row_subject_response(row)
            if not subject_response.strip():
                return _with_extraction_provenance(
                    {
                        "custom_id": cid,
                        "actions": [],
                        "refusal_only": False,
                        "refusal_signaled": False,
                        "extraction_error": "empty_subject_response",
                        "extractor_route": "minimax_native_async",
                    },
                    subject_response=subject_response,
                )
            task_id_raw = task.get("task_id")
            task_id = task_id_raw if isinstance(task_id_raw, str) else ""
            return extractor.parse_completed(
                custom_id=cid,
                task_id=task_id,
                subject_response=subject_response,
                payload=submit_response,
            )

        recovered = 0
        for cid, task in list(tasks.items()):
            record = _parse_task_if_ready(cid, task)
            if record is None:
                continue
            _append_jsonl(args.out, record)
            completed[cid] = record
            recovered += 1
        if recovered:
            print(
                json.dumps(
                    {
                        "status": "recovered_from_submit_responses",
                        "recovered": recovered,
                        "completed": len(completed),
                        "tasks": len(tasks),
                        "out": str(args.out),
                    },
                    sort_keys=True,
                )
            )

        if not args.poll_only:
            for cid, row in rows_by_id.items():
                if cid in completed or cid in tasks:
                    continue
                subject_response = _row_subject_response(row)
                if not subject_response.strip():
                    record = _with_extraction_provenance(
                        {
                            "custom_id": cid,
                            "actions": [],
                            "refusal_only": False,
                            "refusal_signaled": False,
                            "extraction_error": "empty_subject_response",
                            "extractor_route": "minimax_native_async",
                        },
                        subject_response=subject_response,
                    )
                    _append_jsonl(args.out, record)
                    completed[cid] = record
                    continue
                task = extractor.submit(cid, subject_response)
                _append_jsonl(tasks_path, task)
                tasks[cid] = task
                record = _parse_task_if_ready(cid, task)
                if record is not None:
                    _append_jsonl(args.out, record)
                    completed[cid] = record
                if len(tasks) % 25 == 0:
                    print(json.dumps({"submitted_tasks": len(tasks), "completed": len(completed)}, sort_keys=True))

        if args.submit_only:
            print(
                json.dumps(
                    {
                        "status": "submitted",
                        "rows": len(rows_by_id),
                        "tasks": len(tasks),
                        "completed": len(completed),
                        "tasks_path": str(tasks_path),
                        "out": str(args.out),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        deadline = time.monotonic() + args.max_poll_seconds
        while True:
            pending = [
                (cid, task)
                for cid, task in tasks.items()
                if cid not in completed and isinstance(task.get("task_id"), str)
            ]
            if not pending:
                break
            if time.monotonic() >= deadline:
                print(
                    json.dumps(
                        {
                            "status": "timeout",
                            "pending": len(pending),
                            "completed": len(completed),
                            "out": str(args.out),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 1

            def _poll_one(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
                cid, task = item
                row = rows_by_id[cid]
                subject_response = _row_subject_response(row)
                payload = extractor.query(str(task["task_id"]))
                record = extractor.parse_completed(
                    custom_id=cid,
                    task_id=str(task["task_id"]),
                    subject_response=subject_response,
                    payload=payload,
                )
                return cid, record

            wrote = 0
            with ThreadPoolExecutor(max_workers=args.poll_workers, thread_name_prefix="minimax-poll") as pool:
                futures = {pool.submit(_poll_one, item): item for item in pending}
                for fut in as_completed(futures):
                    cid, record = fut.result()
                    if record is None or cid in completed:
                        continue
                    _append_jsonl(args.out, record)
                    completed[cid] = record
                    wrote += 1
            print(
                json.dumps(
                    {
                        "status": "poll",
                        "completed": len(completed),
                        "pending": len(pending) - wrote,
                        "newly_completed": wrote,
                        "out": str(args.out),
                    },
                    sort_keys=True,
                )
            )
            if wrote == 0:
                time.sleep(args.poll_interval_seconds)
    finally:
        extractor.close()

    print(
        json.dumps(
            {
                "status": "completed",
                "rows": len(rows_by_id),
                "completed": len(completed),
                "tasks": len(tasks),
                "out": str(args.out),
                "tasks_path": str(tasks_path),
                "cache_dir": str(args.cache_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if len(completed) == len(rows_by_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())
