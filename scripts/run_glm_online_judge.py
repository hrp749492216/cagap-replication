#!/usr/bin/env python3
"""Online GLM-5.1 judge dispatcher with row-async parallelism.

Reads the same response file + extractions emitted by
``dispatch_judge_batches.py`` and dispatches glm-5.1 prompts through one
partition-level route. Zai-direct is primary; Ollama Cloud is used only when
the zai key is absent before dispatch, not as a row-level fallback after an
invalid zai completion. Results are written to a votes JSONL compatible with
``fetch_judge_batches.py``'s output (so the merge step is uniform).

Per Amendment 13.J extension (2026-05-08): GLM has no usable batch route
(BigModel batch needs identity verification we have not completed) so we
fall back to online with row-async ThreadPoolExecutor.
"""
from __future__ import annotations

import argparse
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

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from judges.judge import (  # noqa: E402
    JudgeInfrastructureError,
    MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS,
    MAX_JUDGE_TRANSPORT_ATTEMPTS,
)
from scripts.raw_http_archive import archive_response_body, response_body_bytes  # noqa: E402

PROMPT_TEMPLATE_PATH = REPO_ROOT / "judges" / "prompt.md"
ZAI_NATIVE_URL = "https://api.z.ai/api/coding/paas/v4/chat/completions"
ZAI_FALLBACK_OLLAMA_URL = "https://ollama.com/v1/chat/completions"
ZAI_FALLBACK_OLLAMA_MODEL = "glm-5.1:cloud"
# Backwards-compatible alias for the subject-output budget. Transport
# failures use the dedicated MAX_JUDGE_TRANSPORT_ATTEMPTS budget and
# raise JudgeInfrastructureError on exhaustion.
MAX_JUDGE_ATTEMPTS = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS
# D5 / 2026-05-27 OSF amendment: standardize GLM judge budget to
# the 4096-token default used by the active judge panel.
GLM_MAX_TOKENS = 4096


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _completed_custom_ids(path: Path) -> set[str]:
    """Return the set of custom_ids already present in the output JSONL.

    Used for resume: rows whose custom_id appears in the existing output file
    are skipped on the next dispatch, so completed work is never redone after
    a crash, laptop shutdown, network drop, or operator cancellation.
    """
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid resume cache row {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(rec, dict):
                raise ValueError(
                    f"Invalid resume cache row {path}:{line_no}: row must be a JSON object."
                )
            cid = rec.get("custom_id")
            if cid is None:
                continue
            cid_text = str(cid)
            if cid_text in completed:
                raise ValueError(
                    f"{path}:{line_no}: duplicate completed custom_id {cid_text!r}; "
                    "remove duplicate before resuming."
                )
            completed.add(cid_text)
    return completed


class GLMClient:
    """Direct GLM client using one partition-level GLM route.

    Z.ai's roster ``url_path`` (``/v1/chat/completions``) 404s; the working
    endpoint is ``/api/coding/paas/v4/chat/completions``. We dispatch
    directly here rather than route through ``RoutedClient`` so the canary
    is unblocked while the roster path is corrected.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.zai_key = os.environ.get("ZAI_API_KEY") or os.environ.get("GLM_API_KEY")
        self.ollama_key = os.environ.get("OLLAMA_API_KEY")
        if self.zai_key:
            self.route_name = "zai-direct"
            self.url = ZAI_NATIVE_URL
            self.model = "glm-5.1"
            self.key = self.zai_key
        elif self.ollama_key:
            self.route_name = "ollama_cloud"
            self.url = ZAI_FALLBACK_OLLAMA_URL
            self.model = ZAI_FALLBACK_OLLAMA_MODEL
            self.key = self.ollama_key
        else:
            self.route_name = "unavailable"
            self.url = ""
            self.model = "glm-5.1"
            self.key = ""

    def evaluate(
        self,
        prompt: str,
        actions: Any,
        response: str,
        *,
        custom_id: str | None = None,
    ) -> dict[str, Any]:
        template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        msg = template.replace("{{prompt}}", prompt)
        msg = msg.replace("{{actions}}", json.dumps(actions, indent=2))
        msg = msg.replace("{{response}}", response)
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": msg}],
            "max_tokens": GLM_MAX_TOKENS,
            "temperature": 0.0,
        }

        if not self.key:
            return {
                "label": "Unresolved",
                "reason": "glm_no_route",
                "attempts": [],
            }

        attempts: list[dict[str, Any]] = []
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        subject_invalid_count = 0
        transport_count = 0
        last_transport_error = ""
        max_total = MAX_JUDGE_SUBJECT_INVALID_ATTEMPTS + MAX_JUDGE_TRANSPORT_ATTEMPTS
        attempt_index = 0
        while attempt_index < max_total:
            attempt_index += 1
            raw: dict[str, Any] = {}
            http_status: int | None = None
            finish_reason = ""
            text = ""
            parsed = {"label": "Unresolved", "reason": ""}
            invalid_reason: str | None = None
            raw_http_capture: dict[str, Any] | None = None
            try:
                r = self.session.post(
                    self.url,
                    headers=headers,
                    json=body,
                    timeout=120,
                )
                http_status = r.status_code
                raw_http_capture = archive_response_body(
                    response_body_bytes(r),
                    role="judge",
                    model_id="glm-5.1",
                    route=self.route_name,
                    provider="zai" if self.route_name == "zai-direct" else self.route_name,
                    custom_id=custom_id,
                    attempt_index=attempt_index,
                    http_status=http_status,
                    url=self.url,
                    request_body=body,
                    source="requests.Response.content",
                )
                if r.status_code != 200:
                    raw = {"error": r.text[:1000]}
                    invalid_reason = "transport_error"
                else:
                    raw = r.json()
                    choices = raw.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        msg_obj = choices[0].get("message") or {}
                        text = msg_obj.get("content") or ""
                        finish_reason = choices[0].get("finish_reason") or ""
                    parsed = self._parse(raw)
                    if not text.strip():
                        invalid_reason = "empty_content"
                    elif finish_reason == "length" and parsed["label"] == "Unresolved":
                        invalid_reason = "length_no_label"
                    elif parsed["label"] == "Unresolved":
                        invalid_reason = "invalid_or_missing_label"
            except requests.RequestException as exc:
                raw = {"error": f"{type(exc).__name__}: {exc}"}
                invalid_reason = "transport_error"
            except (json.JSONDecodeError, ValueError) as exc:
                raw = {"error": f"non_json_response: {exc}"}
                invalid_reason = "invalid_or_missing_label"

            attempts.append(
                {
                    "judge_id": "glm-5.1",
                    "attempt_index": attempt_index,
                    "provider_route": self.route_name,
                    "timestamp": _now_utc(),
                    "status": "valid" if invalid_reason is None else "invalid",
                    "http_status": http_status,
                    "finish_reason": finish_reason,
                    "content_chars": len(text),
                    "parsed_label": (
                        parsed["label"] if parsed["label"] != "Unresolved" else None
                    ),
                    "invalid_reason": invalid_reason,
                    "raw_response": raw,
                    "raw_http_capture": raw_http_capture,
                }
            )
            if invalid_reason is None:
                return {**parsed, "attempts": attempts}
            if invalid_reason == "transport_error":
                transport_count += 1
                last_transport_error = str(raw.get("error", "")) or f"http_status={http_status}"
                if transport_count >= MAX_JUDGE_TRANSPORT_ATTEMPTS:
                    raise JudgeInfrastructureError(
                        judge_id="glm-5.1",
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
            time.sleep(min(2 ** attempt_index, 10))

        # Safety net: combined budget exhausted (unreachable under
        # current caps but guarantees a return value).
        return {
            "label": "Unresolved",
            "reason": "judge_invalid_after_3_attempts",
            "attempts": attempts,
        }

    @staticmethod
    def _parse(payload: dict[str, Any]) -> dict[str, str]:
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return {"label": "Unresolved", "reason": "glm_malformed_response"}
        label = "Unresolved"
        reason = ""
        for ln in (text or "").strip().splitlines():
            s = ln.strip()
            if s.lower().startswith("label:"):
                v = s.split(":", 1)[1].strip().upper()
                if v in ("R", "P", "C"):
                    label = {"R": "Refuse", "P": "Partial", "C": "Comply"}[v]
            elif s.lower().startswith("reason:"):
                reason = s.split(":", 1)[1].strip()
        return {"label": label, "reason": reason or "(no reason parsed)"}

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        text = msg.get("content")
        if isinstance(text, str):
            return text
    content = raw.get("content")
    if isinstance(content, list):
        return "".join(p.get("text") or "" for p in content if isinstance(p, dict))
    if isinstance(content, str):
        return content
    return ""


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


def _judge_one(client: GLMClient, judge_id: str, row: dict[str, Any],
               extractions: dict[str, dict[str, Any]],
               prompt_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cid = row.get("custom_id") or ""
    ext = extractions.get(cid) or {}
    actions = ext.get("actions") or []
    prompt = _original_prompt(row, prompt_lookup)
    response = _subject_response_text(_raw_response(row))
    try:
        result = client.evaluate(prompt, actions, response, custom_id=str(cid))
        label = result.get("label") or "Unresolved"
        reason = result.get("reason") or ""
        attempts = result.get("attempts") or []
    except Exception as exc:  # noqa: BLE001
        label = "Unresolved"
        reason = f"online_error: {type(exc).__name__}: {exc}"
        attempts = []
    return {"custom_id": cid, "judge": judge_id, "label": label,
            "reason": reason, "source": client.route_name, "attempts": attempts}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="Subject response JSONL")
    p.add_argument("--extractions", required=True, type=Path,
                   help="Extractions JSONL from dispatch_judge_batches.py")
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="Output JSONL (per-row votes)")
    p.add_argument("--judge", default="glm-5.1")
    p.add_argument("--row-workers", type=int, default=3,
                   help="Parallel rows (cap to ollama_cloud limit)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    extractions = {}
    for ln in args.extractions.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        extractions[e.get("custom_id") or ""] = e

    prompt_lookup: dict[str, dict[str, Any]] = {}
    for ln in args.prompts.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        case_id = r.get("case_card_id")
        style = r.get("style_slot")
        if case_id and style:
            prompt_lookup[f"{case_id}_{style}"] = r
        if case_id:
            prompt_lookup.setdefault(case_id, r)

    client = GLMClient()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    completed_ids = _completed_custom_ids(args.out)
    pending = [row for row in rows if str(row.get("custom_id") or "") not in completed_ids]
    skipped_existing = len(rows) - len(pending)
    if skipped_existing:
        logger.info(
            "resume: %d/%d rows already present in %s; dispatching %d new rows",
            skipped_existing, len(rows), args.out, len(pending),
        )
    done_count = 0
    with args.out.open("a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.row_workers,
                                 thread_name_prefix="glm-row") as pool:
            futures = {pool.submit(_judge_one, client, args.judge, row, extractions, prompt_lookup): i
                       for i, row in enumerate(pending)}
            for fut in as_completed(futures):
                vote = fut.result()
                with write_lock:
                    fh.write(json.dumps(vote, sort_keys=True, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                done_count += 1
                if done_count % 10 == 0 or done_count == len(pending):
                    logger.info("graded %d/%d rows", done_count, len(pending))
    logger.info(
        "wrote %s (%d new rows, %d skipped existing)",
        args.out, done_count, skipped_existing,
    )
    print(json.dumps({"out": str(args.out), "rows_new": done_count,
                      "rows_skipped_existing": skipped_existing,
                      "judge": args.judge},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
