#!/usr/bin/env python3
"""Dispatch Command A judge prompts to a self-hosted vLLM endpoint.

The script consumes prepared judge input files:

  outputs/main_run/<run>/judge_batches/<subject>/judge_inputs/command-a-03-2025.jsonl

and writes resumable vote files in the D6 canonical location:

  outputs/main_run/<run>/judge_batches/<subject>/command_a_self_hosted/votes.jsonl

It is intentionally standalone so it can run on the Verda GPU VM with only the
synced judge_inputs tree and an OpenAI-compatible vLLM server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

VALID_LABELS = {"Refuse", "Partial", "Comply"}
MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS = 3
MAX_JUDGE_TRANSPORT_ATTEMPTS = 6
LABEL_ALIASES = {
    "R": "Refuse",
    "REFUSE": "Refuse",
    "REFUSAL": "Refuse",
    "P": "Partial",
    "PARTIAL": "Partial",
    "PARTIAL/HEDGED": "Partial",
    "HEDGED": "Partial",
    "C": "Comply",
    "COMPLY": "Comply",
    "COMPLIANCE": "Comply",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_route_selection_record(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"route selection record must be a JSON object: {path}")
    return payload, _sha256_file(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


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


def _parse_label(text: str) -> tuple[str, str]:
    label = "Unresolved"
    reason = ""
    for line in (text or "").strip().splitlines():
        field, sep, value = line.partition(":")
        if not sep:
            continue
        field_name = field.strip().lower().lstrip("-* \t").strip("*_` ")
        field_value = value.strip().strip("*_` ")
        if field_name == "label":
            token = field_value.upper().split()[0] if field_value else ""
            token = token.strip("*_` ")
            label = LABEL_ALIASES.get(token, label)
        elif field_name == "reason":
            reason = field_value
    return label, reason or "(no reason parsed)"


def _choice_payload(payload: dict[str, Any]) -> tuple[str, str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "", ""
    choice = choices[0]
    message = choice.get("message")
    content = ""
    if isinstance(message, dict):
        raw_content = message.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            content = "".join(parts)
    finish_reason = choice.get("finish_reason")
    return content, finish_reason if isinstance(finish_reason, str) else ""


def _vote_is_valid_for_run(
    row: dict[str, Any],
    *,
    judge_id: str,
    model_label: str,
    provider: str,
    prompt_hashes: dict[str, str],
) -> bool:
    cid = row.get("custom_id")
    return (
        isinstance(cid, str)
        and cid
        and row.get("judge") == judge_id
        and row.get("model") == model_label
        and row.get("provider") == provider
        and row.get("label") in VALID_LABELS
        and prompt_hashes.get(cid) == row.get("judge_prompt_sha256")
    )


@dataclass(frozen=True)
class SubjectTask:
    subject_id: str
    input_path: Path
    output_path: Path
    entries: list[dict[str, Any]]
    prompt_hashes: dict[str, str]


@dataclass(frozen=True)
class WorkItem:
    subject: SubjectTask
    entry: dict[str, Any]


class VLLMJudgeClient:
    def __init__(
        self,
        *,
        endpoint: str,
        served_model: str,
        model_label: str,
        provider: str,
        source: str,
        api_key_env: str,
        timeout_seconds: float,
        max_tokens: int,
        temperature: float,
        min_interval_seconds: float,
        route_selection_path: Path | None = None,
        route_selection_sha256: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.url = self.endpoint + "/v1/chat/completions"
        self.served_model = served_model
        self.model_label = model_label
        self.provider = provider
        self.source = source
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.route_selection_path = route_selection_path
        self.route_selection_sha256 = route_selection_sha256
        self.headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(api_key_env)
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._pace_lock = threading.Lock()
        self._next_slot = 0.0
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _wait_for_slot(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._pace_lock:
            now = time.monotonic()
            if now < self._next_slot:
                time.sleep(self._next_slot - now)
                now = time.monotonic()
            self._next_slot = now + self.min_interval_seconds

    def check_models(self) -> list[str]:
        response = requests.get(
            self.endpoint + "/v1/models",
            headers=self.headers,
            timeout=min(30.0, self.timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return []
        out: list[str] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                out.append(item["id"])
        return out

    def evaluate(self, judge_prompt: str, *, custom_id: str) -> dict[str, Any]:
        body = {
            "model": self.served_model,
            "messages": [{"role": "user", "content": judge_prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        decoding_parameters = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        attempts: list[dict[str, Any]] = []
        subject_invalid_count = 0
        transport_count = 0
        last_transport_error = ""
        max_total = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS + MAX_JUDGE_TRANSPORT_ATTEMPTS
        for attempt in range(1, max_total + 1):
            payload: dict[str, Any] = {}
            text = ""
            finish_reason = ""
            http_status: int | None = None
            invalid_reason: str | None = None
            started = time.monotonic()
            try:
                self._wait_for_slot()
                response = self._session().post(
                    self.url,
                    headers=self.headers,
                    json=body,
                    timeout=self.timeout_seconds,
                )
                http_status = response.status_code
                if response.status_code != 200:
                    payload = {"error": response.text[:2000]}
                    invalid_reason = "transport_error"
                else:
                    payload = response.json()
                    text, finish_reason = _choice_payload(payload)
            except requests.RequestException as exc:
                payload = {"error": f"{type(exc).__name__}: {exc}"}
                invalid_reason = "transport_error"
            except (json.JSONDecodeError, ValueError) as exc:
                payload = {"error": f"non_json_response: {exc}"}
                invalid_reason = "invalid_or_missing_label"

            label, reason = _parse_label(text)
            if invalid_reason is None:
                if not text.strip():
                    invalid_reason = "empty_content"
                elif finish_reason == "length" and label == "Unresolved":
                    invalid_reason = "length_no_label"
                elif label == "Unresolved":
                    invalid_reason = "invalid_or_missing_label"

            runtime_stamp = {
                "requested_model": self.served_model,
                "model_label": self.model_label,
                "route": "self_hosted",
                "provider": self.provider,
                "endpoint": self.url,
                "role": "judge",
                "http_status": http_status,
                "attempt_index": attempt,
                "wall_seconds": round(time.monotonic() - started, 3),
                "timestamp": _now_utc(),
                "decoding_parameters": decoding_parameters,
                "route_selection_record_path": str(self.route_selection_path) if self.route_selection_path else None,
                "route_selection_record_sha256": self.route_selection_sha256,
            }
            attempts.append(
                {
                    "judge_id": self.model_label,
                    "attempt_index": attempt,
                    "provider_route": "self_hosted",
                    "timestamp": _now_utc(),
                    "status": "valid" if invalid_reason is None else "invalid",
                    "http_status": http_status,
                    "finish_reason": finish_reason,
                    "content_chars": len(text),
                    "parsed_label": label if label != "Unresolved" else None,
                    "invalid_reason": invalid_reason,
                    "raw_response": payload,
                    "runtime_stamp": runtime_stamp,
                }
            )
            if invalid_reason is None:
                return {
                    "label": label,
                    "reason": reason,
                    "raw_text": text[:600],
                    "attempts": attempts,
                    "runtime_stamp": runtime_stamp,
                }
            if invalid_reason == "transport_error":
                transport_count += 1
                last_transport_error = str(payload.get("error", "")) or f"http_status={http_status}"
                if transport_count >= MAX_JUDGE_TRANSPORT_ATTEMPTS:
                    return {
                        "label": "Unresolved",
                        "reason": f"self_hosted_transport_exhausted: {last_transport_error[:500]}",
                        "raw_text": "",
                        "attempts": attempts,
                        "runtime_stamp": runtime_stamp,
                    }
            else:
                subject_invalid_count += 1
                if subject_invalid_count >= MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS:
                    return {
                        "label": "Unresolved",
                        "reason": "judge_invalid_after_3_attempts",
                        "raw_text": "",
                        "attempts": attempts,
                        "runtime_stamp": runtime_stamp,
                    }
            time.sleep(min(2 ** attempt, 20))
        return {
            "label": "Unresolved",
            "reason": "judge_invalid_after_3_attempts",
            "raw_text": "",
            "attempts": attempts,
            "runtime_stamp": attempts[-1]["runtime_stamp"] if attempts else None,
        }


def _discover_subjects(root: Path, requested: list[str]) -> list[Path]:
    if requested:
        return [root / subject for subject in requested]
    return sorted(path for path in root.iterdir() if path.is_dir())


def _deterministic_sample(
    entries: list[dict[str, Any]],
    *,
    fraction: float,
    seed: str,
    subject_id: str,
) -> list[dict[str, Any]]:
    if fraction <= 0.0 or fraction >= 1.0:
        return entries
    target = max(1, round(len(entries) * fraction))

    def key(entry: dict[str, Any]) -> str:
        cid = str(entry.get("custom_id") or "")
        return hashlib.sha256(f"{seed}\0{subject_id}\0{cid}".encode("utf-8")).hexdigest()

    selected_ids = {id(entry) for entry in sorted(entries, key=key)[:target]}
    return [entry for entry in entries if id(entry) in selected_ids]


def _subject_id_from_path(path: Path) -> str:
    return path.name


def _load_subject_task(
    subject_dir: Path,
    *,
    judge_id: str,
    model_label: str,
    provider: str,
) -> tuple[SubjectTask | None, int]:
    input_path = subject_dir / "judge_inputs" / f"{judge_id}.jsonl"
    if not input_path.exists():
        return None, 0
    entries = _load_jsonl(input_path)
    prompt_hashes = _prompt_hashes(entries)
    output_path = subject_dir / "command_a_self_hosted" / "votes.jsonl"
    valid_rows: list[dict[str, Any]] = []
    ignored_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if output_path.exists():
        for row in _load_jsonl(output_path):
            cid = row.get("custom_id")
            if (
                isinstance(cid, str)
                and cid not in seen
                and _vote_is_valid_for_run(
                    row,
                    judge_id=judge_id,
                    model_label=model_label,
                    provider=provider,
                    prompt_hashes=prompt_hashes,
                )
            ):
                valid_rows.append(row)
                seen.add(cid)
            else:
                ignored_rows.append(row)
        if ignored_rows:
            ignored_path = output_path.with_suffix(output_path.suffix + ".ignored.jsonl")
            ignored_path.parent.mkdir(parents=True, exist_ok=True)
            with ignored_path.open("a", encoding="utf-8") as fh:
                for row in ignored_rows:
                    fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            _write_jsonl(output_path, valid_rows)
    task = SubjectTask(
        subject_id=_subject_id_from_path(subject_dir),
        input_path=input_path,
        output_path=output_path,
        entries=entries,
        prompt_hashes=prompt_hashes,
    )
    return task, len(seen)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-batches-root",
        type=Path,
        default=Path("outputs/main_run/20260525T164059Z/judge_batches"),
    )
    parser.add_argument("--subjects", default="", help="Comma-separated subject directory names. Default: all.")
    parser.add_argument("--endpoint", default=os.environ.get("CAGAP_VLLM_ENDPOINT", "http://127.0.0.1:8000"))
    parser.add_argument("--served-model", default=os.environ.get("CAGAP_COMMAND_A_SERVED_MODEL", "command-a-03-2025"))
    parser.add_argument("--model-label", default="command-a-03-2025")
    parser.add_argument("--judge", default="command-a-03-2025")
    parser.add_argument("--provider", default="cohere_self_hosted")
    parser.add_argument("--source", default="self_hosted")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--min-interval-seconds", type=float, default=0.0)
    parser.add_argument("--limit-total", type=int, default=0, help="Debug cap across all subjects; 0 means no cap.")
    parser.add_argument("--limit-per-subject", type=int, default=0, help="Debug cap per subject; 0 means no cap.")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.0,
        help=(
            "Deterministic per-subject sample fraction for timing/cost probes. "
            "Example: 0.10 queues about 10%% of each subject's remaining rows."
        ),
    )
    parser.add_argument("--sample-seed", default="command-a-self-hosted-v1")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional JSON summary path for timing/cost extrapolation.",
    )
    parser.add_argument(
        "--route-selection-record",
        type=Path,
        default=None,
        help=(
            "JSON route-selection record frozen before canonical dispatch. "
            "Its SHA-256 is logged in summaries and each vote runtime stamp."
        ),
    )
    parser.add_argument("--skip-model-match", action="store_true")
    parser.add_argument("--allow-invalid-completions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_fraction < 0.0 or args.sample_fraction > 1.0:
        print("--sample-fraction must be between 0 and 1", file=sys.stderr)
        return 2
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not args.judge_batches_root.exists():
        print(f"judge batches root missing: {args.judge_batches_root}", file=sys.stderr)
        return 2
    try:
        route_selection_record, route_selection_sha256 = _load_route_selection_record(
            args.route_selection_record
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"invalid route selection record: {exc}", file=sys.stderr)
        return 2

    client = VLLMJudgeClient(
        endpoint=args.endpoint,
        served_model=args.served_model,
        model_label=args.model_label,
        provider=args.provider,
        source=args.source,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        min_interval_seconds=args.min_interval_seconds,
        route_selection_path=args.route_selection_record,
        route_selection_sha256=route_selection_sha256,
    )
    if not args.skip_model_match and not args.dry_run:
        served = client.check_models()
        if args.served_model not in served:
            print(
                json.dumps(
                    {
                        "error": "served_model_missing",
                        "expected": args.served_model,
                        "served_models": served,
                        "endpoint": args.endpoint,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2

    tasks: list[SubjectTask] = []
    skipped_existing_total = 0
    for subject_dir in _discover_subjects(args.judge_batches_root, subjects):
        task, skipped = _load_subject_task(
            subject_dir,
            judge_id=args.judge,
            model_label=args.model_label,
            provider=args.provider,
        )
        if task is None:
            continue
        tasks.append(task)
        skipped_existing_total += skipped

    work: list[WorkItem] = []
    per_subject: dict[str, dict[str, int]] = {}
    for task in tasks:
        done_ids = set()
        if task.output_path.exists():
            for row in _load_jsonl(task.output_path):
                cid = row.get("custom_id")
                if isinstance(cid, str) and _vote_is_valid_for_run(
                    row,
                    judge_id=args.judge,
                    model_label=args.model_label,
                    provider=args.provider,
                    prompt_hashes=task.prompt_hashes,
                ):
                    done_ids.add(cid)
        entries = [
            entry
            for entry in task.entries
            if isinstance(entry.get("custom_id"), str) and entry["custom_id"] not in done_ids
        ]
        if args.sample_fraction > 0.0:
            entries = _deterministic_sample(
                entries,
                fraction=args.sample_fraction,
                seed=args.sample_seed,
                subject_id=task.subject_id,
            )
        if args.limit_per_subject > 0:
            entries = entries[: args.limit_per_subject]
        per_subject[task.subject_id] = {
            "input_rows": len(task.entries),
            "skipped_existing": len(done_ids),
            "queued": len(entries),
            "completed": 0,
            "invalid": 0,
        }
        for entry in entries:
            work.append(WorkItem(subject=task, entry=entry))
    if args.limit_total > 0:
        work = work[: args.limit_total]

    print(
        json.dumps(
            {
                "event": "planned",
                "judge_batches_root": str(args.judge_batches_root),
                "subjects": len(tasks),
                "queued_rows": len(work),
                "skipped_existing_rows": skipped_existing_total,
                "endpoint": args.endpoint,
                "served_model": args.served_model,
                "model_label": args.model_label,
                "workers": args.workers,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "dry_run": args.dry_run,
                "sample_fraction": args.sample_fraction,
                "sample_seed": args.sample_seed if args.sample_fraction > 0.0 else None,
                "route_selection_record": str(args.route_selection_record) if args.route_selection_record else None,
                "route_selection_sha256": route_selection_sha256,
                "per_subject": per_subject,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run or not work:
        return 0

    write_locks: dict[Path, threading.Lock] = {task.output_path: threading.Lock() for task in tasks}
    work_queue: queue.Queue[WorkItem | None] = queue.Queue()
    for item in work:
        work_queue.put(item)
    for _ in range(max(1, args.workers)):
        work_queue.put(None)

    state = {
        "completed": 0,
        "valid": 0,
        "invalid": 0,
        "started": time.monotonic(),
        "last_print": time.monotonic(),
    }
    state_lock = threading.Lock()

    def worker() -> None:
        while True:
            item = work_queue.get()
            try:
                if item is None:
                    return
                cid = str(item.entry.get("custom_id") or "")
                prompt = str(item.entry.get("judge_prompt") or "")
                result = client.evaluate(prompt, custom_id=cid)
                vote = {
                    "custom_id": cid,
                    "judge": args.judge,
                    "label": result.get("label") or "Unresolved",
                    "reason": result.get("reason") or "",
                    "raw_text": result.get("raw_text") or "",
                    "attempts": result.get("attempts") or [],
                    "source": args.source,
                    "provider": args.provider,
                    "model": args.model_label,
                    "served_model": args.served_model,
                    "judge_prompt_sha256": _sha256_text(prompt),
                    "runtime_stamp": result.get("runtime_stamp"),
                    "route_selection_record_sha256": route_selection_sha256,
                }
                with write_locks[item.subject.output_path]:
                    _append_jsonl(item.subject.output_path, vote)
                valid = vote["label"] in VALID_LABELS
                with state_lock:
                    state["completed"] += 1
                    state["valid"] += 1 if valid else 0
                    state["invalid"] += 0 if valid else 1
                    per_subject[item.subject.subject_id]["completed"] += 1
                    per_subject[item.subject.subject_id]["invalid"] += 0 if valid else 1
                    now = time.monotonic()
                    if now - state["last_print"] >= 30 or state["completed"] == len(work):
                        elapsed = now - state["started"]
                        rate = state["completed"] / elapsed if elapsed > 0 else 0.0
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "completed": state["completed"],
                                    "queued_rows": len(work),
                                    "valid": state["valid"],
                                    "invalid": state["invalid"],
                                    "rows_per_second": round(rate, 3),
                                    "elapsed_seconds": round(elapsed, 1),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        state["last_print"] = now
            finally:
                work_queue.task_done()

    worker_count = max(1, args.workers)
    threads = [threading.Thread(target=worker, name=f"cmd-a-judge-{i}", daemon=True) for i in range(worker_count)]
    for thread in threads:
        thread.start()
    work_queue.join()
    for thread in threads:
        thread.join(timeout=1)

    elapsed = time.monotonic() - state["started"]
    summary = {
        "event": "done",
        "rows_completed_this_run": state["completed"],
        "valid_rows_completed_this_run": state["valid"],
        "invalid_rows_completed_this_run": state["invalid"],
        "rows_queued_at_start": len(work),
        "elapsed_seconds": round(elapsed, 1),
        "rows_per_second": round(state["completed"] / elapsed, 3) if elapsed > 0 else None,
        "judge_batches_root": str(args.judge_batches_root),
        "endpoint": args.endpoint,
        "served_model": args.served_model,
        "model_label": args.model_label,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "route_selection_record": str(args.route_selection_record) if args.route_selection_record else None,
        "route_selection_sha256": route_selection_sha256,
        "route_selection": route_selection_record,
        "per_subject": per_subject,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if state["invalid"] > 0 and not args.allow_invalid_completions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
