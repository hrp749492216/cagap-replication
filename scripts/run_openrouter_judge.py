#!/usr/bin/env python3
"""Dispatch pre-rendered judge prompts through pinned OpenRouter routes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cagap_env import load_cagap_env  # noqa: E402
from grader.json_strict import reject_duplicate_keys, reject_nonfinite, safe_float  # noqa: E402
from scripts.run_synthetic_prompts import Config, decoding_for  # noqa: E402
from scripts.runtime_provenance import (
    build_api_runtime_stamp,
    write_openrouter_model_metadata_snapshot,
)  # noqa: E402
from scripts.raw_http_archive import archive_response_body, response_body_bytes  # noqa: E402
from scripts.providers import get_adapter  # noqa: E402

load_cagap_env()

from judges.judge import (  # noqa: E402
    JudgeInfrastructureError,
    MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS,
    MAX_JUDGE_TRANSPORT_ATTEMPTS,
)

# Backwards-compatible alias for the subject-output budget.
MAX_JUDGE_ATTEMPTS = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS
# Statuses where the runner SHOULD retry without raise_for_status. Other
# 4xx/5xx codes still count as transport_error (operator-fixable
# infrastructure problems: 400 schema mismatch, 401 auth, 402 credits,
# 403 forbidden, 404 wrong slug) and consume the transport budget rather
# than the subject-output budget.
TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
LABELS = {"R": "Refuse", "P": "Partial", "C": "Comply"}
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder(
        strict=False,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
        parse_float=safe_float,
    )
    idx = 0
    line_no = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        row, idx = decoder.raw_decode(text, idx)
        line_no += 1
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
        rows.append(row)
    return rows


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
    prompt_hashes: dict[str, str],
) -> bool:
    if row.get("judge") != judge_id:
        return False
    if row.get("model") != model:
        return False
    if row.get("provider") != provider:
        return False
    cid = row.get("custom_id")
    if not isinstance(cid, str) or prompt_hashes.get(cid) != row.get("judge_prompt_sha256"):
        return False
    return row.get("label") in set(LABELS.values())


def _completed_custom_ids(
    path: Path,
    *,
    judge_id: str,
    model: str,
    provider: str,
    prompt_hashes: dict[str, str],
) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for row in _load_jsonl(path):
        cid = row.get("custom_id")
        if (
            isinstance(cid, str)
            and cid
            and _vote_is_valid_for_run(
                row,
                judge_id=judge_id,
                model=model,
                provider=provider,
                prompt_hashes=prompt_hashes,
            )
        ):
            done.add(cid)
    return done


def _clean_existing_votes_for_run(
    path: Path,
    *,
    judge_id: str,
    model: str,
    provider: str,
    prompt_hashes: dict[str, str],
) -> set[str]:
    if not path.exists():
        return set()
    valid_rows: list[dict[str, Any]] = []
    ignored_rows: list[dict[str, Any]] = []
    done: set[str] = set()
    for row in _load_jsonl(path):
        cid = row.get("custom_id")
        if (
            isinstance(cid, str)
            and cid
            and cid not in done
            and _vote_is_valid_for_run(
                row,
                judge_id=judge_id,
                model=model,
                provider=provider,
                prompt_hashes=prompt_hashes,
            )
        ):
            valid_rows.append(row)
            done.add(cid)
        else:
            ignored_rows.append(row)
    if ignored_rows:
        ignored_path = path.with_suffix(path.suffix + ".ignored.jsonl")
        _append_jsonl(ignored_path, ignored_rows)
        print(
            json.dumps(
                {
                    "status": "ignored_stale_or_invalid_existing_votes",
                    "ignored": len(ignored_rows),
                    "ignored_path": str(ignored_path),
                    "out": str(path),
                },
                sort_keys=True,
            )
        )
        _write_jsonl(path, valid_rows)
    return done


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
        content = raw_content if isinstance(raw_content, str) else ""
    finish_reason = choice.get("finish_reason")
    return content, finish_reason if isinstance(finish_reason, str) else ""


class OpenRouterJudgeClient:
    def __init__(
        self,
        judge_id: str,
        min_interval_seconds: float = 0.0,
        *,
        max_tokens: int | None = None,
    ) -> None:
        if judge_id not in Config.JUDGES:
            raise ValueError(f"unknown judge id: {judge_id}")
        entry = Config.JUDGES[judge_id]
        route = entry.get("openrouter")
        if not isinstance(route, dict):
            raise ValueError(f"judge {judge_id} has no OpenRouter route")
        api_key = os.environ.get(Config.OPENROUTER_AUTH_ENV)
        if not api_key:
            raise RuntimeError(f"{Config.OPENROUTER_AUTH_ENV} not set")
        self.judge_id = judge_id
        self.entry = entry
        self.slug = route["slug"]
        self.pinned_provider = route["pinned_provider"]
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.max_tokens = max_tokens
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._pace_lock = threading.Lock()
        self._next_slot = 0.0
        self.model_metadata_snapshot: dict[str, Any] | None = None

    def _wait_for_slot(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._pace_lock:
            now = time.monotonic()
            if now < self._next_slot:
                time.sleep(self._next_slot - now)
                now = time.monotonic()
            self._next_slot = now + self.min_interval_seconds

    def capture_model_metadata_snapshot(self, output_dir: Path) -> dict[str, Any]:
        self.model_metadata_snapshot = write_openrouter_model_metadata_snapshot(
            session=requests,
            api_key=os.environ.get(Config.OPENROUTER_AUTH_ENV, ""),
            output_dir=output_dir,
            model_slug=self.slug,
            pinned_provider=self.pinned_provider,
        )
        return self.model_metadata_snapshot

    def evaluate_rendered(self, judge_prompt: str, *, custom_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.slug,
            "messages": [{"role": "user", "content": judge_prompt}],
            "provider": {
                "only": [self.pinned_provider],
                "allow_fallbacks": False,
                "data_collection": "deny",
            },
            "plugins": [{"id": "context-compression", "enabled": False}],
            "transforms": [],
        }
        body.update(decoding_for(self.judge_id))
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.pinned_provider == "openai" and self.slug.startswith("openai/"):
            body = get_adapter("openai").to_native_body(body, self.entry)

        attempts: list[dict[str, Any]] = []
        subject_invalid_count = 0
        transport_count = 0
        last_transport_error = ""
        max_total = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS + MAX_JUDGE_TRANSPORT_ATTEMPTS
        attempt = 0
        while attempt < max_total:
            attempt += 1
            payload: dict[str, Any] = {}
            text = ""
            finish_reason = ""
            http_status: int | None = None
            invalid_reason: str | None = None
            response_headers: dict[str, Any] = {}
            raw_http_capture: dict[str, Any] | None = None
            try:
                self._wait_for_slot()
                response = requests.post(
                    f"{Config.OPENROUTER_BASE_URL}/chat/completions",
                    headers=self.headers,
                    json=body,
                    timeout=180,
                )
                http_status = response.status_code
                raw_http_capture = archive_response_body(
                    response_body_bytes(response),
                    role="judge",
                    model_id=self.judge_id,
                    route="openrouter",
                    provider=self.pinned_provider,
                    custom_id=custom_id,
                    attempt_index=attempt,
                    http_status=http_status,
                    url=f"{Config.OPENROUTER_BASE_URL}/chat/completions",
                    request_body=body,
                    source="requests.Response.content",
                )
                response_headers = dict(getattr(response, "headers", {}) or {})
                if response.status_code in TRANSIENT_STATUSES:
                    payload = {"error": response.text[:1000]}
                    invalid_reason = "transport_error"
                else:
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload.get("error"), dict):
                        invalid_reason = "transport_error"
                    text, finish_reason = _choice_payload(payload)
            except requests.RequestException as exc:
                payload = {"error": f"{type(exc).__name__}: {exc}"}
                invalid_reason = "transport_error"
            except (json.JSONDecodeError, ValueError) as exc:
                payload = {"error": f"non_json_response: {exc}"}
                invalid_reason = "invalid_or_missing_label"

            label, reason = _parse_label(text)
            runtime_stamp = build_api_runtime_stamp(
                requested_model=self.slug,
                route="openrouter",
                provider=self.pinned_provider,
                response=payload,
                response_headers=response_headers,
                endpoint=f"{Config.OPENROUTER_BASE_URL}/chat/completions",
                role="judge",
                model_id=self.judge_id,
                http_status=http_status,
                model_metadata_snapshot=self.model_metadata_snapshot,
                extra={"attempt_index": attempt},
            )
            if invalid_reason is None:
                if not text.strip():
                    invalid_reason = "empty_content"
                elif finish_reason == "length" and label == "Unresolved":
                    invalid_reason = "length_no_label"
                elif label == "Unresolved":
                    invalid_reason = "invalid_or_missing_label"

            attempts.append(
                {
                    "judge_id": self.judge_id,
                    "attempt_index": attempt,
                    "provider_route": "openrouter",
                    "timestamp": _now_utc(),
                    "status": "valid" if invalid_reason is None else "invalid",
                    "http_status": http_status,
                    "finish_reason": finish_reason,
                    "content_chars": len(text),
                    "parsed_label": label if label != "Unresolved" else None,
                    "invalid_reason": invalid_reason,
                    "raw_response": payload,
                    "raw_http_capture": raw_http_capture,
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
                    raise JudgeInfrastructureError(
                        judge_id=self.judge_id,
                        attempts=attempts,
                        last_error=last_transport_error,
                    )
            else:
                subject_invalid_count += 1
                if subject_invalid_count >= MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS:
                    return {
                        "label": "Unresolved",
                        "reason": "judge_invalid_after_3_attempts",
                        "raw_text": "",
                        "attempts": attempts,
                        "runtime_stamp": attempts[-1].get("runtime_stamp") if attempts else None,
                    }
            time.sleep(min(2 ** attempt, 20))
        # Safety net: combined budget exhausted. Should be unreachable
        # because each budget terminates the loop explicitly above.
        return {
            "label": "Unresolved",
            "reason": "judge_invalid_after_3_attempts",
            "raw_text": "",
            "attempts": attempts,
            "runtime_stamp": attempts[-1].get("runtime_stamp") if attempts else None,
        }


def _judge_one(client: OpenRouterJudgeClient, entry: dict[str, Any]) -> dict[str, Any]:
    cid = str(entry.get("custom_id") or "")
    judge_prompt = str(entry.get("judge_prompt") or "")
    prompt_sha256 = _sha256_text(judge_prompt)
    try:
        result = client.evaluate_rendered(judge_prompt, custom_id=cid)
    except TypeError as exc:
        if "custom_id" not in str(exc):
            raise
        result = client.evaluate_rendered(judge_prompt)
    return {
        "custom_id": cid,
        "judge": client.judge_id,
        "label": result.get("label") or "Unresolved",
        "reason": result.get("reason") or "",
        "raw_text": result.get("raw_text") or "",
        "attempts": result.get("attempts") or [],
        "source": "openrouter",
        "provider": client.pinned_provider,
        "model": client.slug,
        "judge_prompt_sha256": prompt_sha256,
        "runtime_stamp": result.get("runtime_stamp"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--row-workers", type=int, default=1)
    parser.add_argument("--min-interval-seconds", type=float, default=0.0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override roster max_tokens for difficult judge retry rows.",
    )
    parser.add_argument(
        "--allow-invalid-completions",
        action="store_true",
        help=(
            "Return 0 even if one or more newly completed rows exhausted all "
            "invalid judge attempts. Confirmatory judge dispatch should leave "
            "this unset so partial judge health failures are visible."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    entries = _load_jsonl(args.judge_inputs)
    prompt_hashes = _prompt_hashes(entries)
    client = OpenRouterJudgeClient(
        args.judge,
        args.min_interval_seconds,
        max_tokens=args.max_tokens,
    )
    done = _clean_existing_votes_for_run(
        args.out,
        judge_id=args.judge,
        model=client.slug,
        provider=client.pinned_provider,
        prompt_hashes=prompt_hashes,
    )
    pending = [
        entry for entry in entries
        if isinstance(entry.get("custom_id"), str) and entry["custom_id"] not in done
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if pending and hasattr(client, "capture_model_metadata_snapshot"):
        client.capture_model_metadata_snapshot(args.out.parent / "openrouter_model_metadata")
    lock = threading.Lock()
    completed = 0
    valid_completed = 0
    invalid_completed = 0
    with args.out.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.row_workers, thread_name_prefix="or-judge") as pool:
            futures = {pool.submit(_judge_one, client, entry): entry for entry in pending}
            for fut in as_completed(futures):
                vote = fut.result()
                with lock:
                    fh.write(json.dumps(vote, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                completed += 1
                if vote.get("label") in set(LABELS.values()):
                    valid_completed += 1
                else:
                    invalid_completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(
                        json.dumps(
                            {
                                "judge": args.judge,
                                "completed_this_run": completed,
                                "pending_at_start": len(pending),
                                "skipped_existing": len(done),
                            },
                            sort_keys=True,
                        )
                    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "judge": args.judge,
                "rows_total": len(entries),
                "rows_completed_this_run": completed,
                "valid_rows_completed_this_run": valid_completed,
                "invalid_rows_completed_this_run": invalid_completed,
                "rows_skipped_existing": len(done),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if completed > 0 and invalid_completed > 0 and not args.allow_invalid_completions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
