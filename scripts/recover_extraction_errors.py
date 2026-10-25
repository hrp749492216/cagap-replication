#!/usr/bin/env python3
"""Re-extract rows that have a non-empty subject response but a recorded
``extraction_error`` from a prior Phase A pass.

Empty-response rows (``extraction_error == "empty_response"``) are NOT
re-extracted because the upstream provider returned no content; those
rows correctly route to ``final_label=Unresolved``.

For each surviving error the script issues a fresh extractor call with
``bypass_cache=True`` (extra attempts beyond the pre-registered one-shot
retry are permitted as a recovery patch, not the main-run grading
contract). Successfully recovered rows replace the failing entries in
``extractions.jsonl``; persistent failures are left in place.

Usage:
    python scripts/recover_extraction_errors.py [--subjects qwen3.6-35b-a3b,...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractor.cache import ExtractorCache  # noqa: E402
from extractor.extract import Extractor  # noqa: E402
from scripts.dispatch_judge_batches import _raw_response, _subject_response_text  # noqa: E402

logging.basicConfig(
    level="INFO", format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("recover_extraction_errors")


PILOT_DATE = "2026-05-08"
V1_DIR = REPO_ROOT / "outputs/pilot" / PILOT_DATE / "judge_batches"
V2_DIR = REPO_ROOT / "outputs/pilot" / PILOT_DATE / "extractions_v2"
V2_CACHE_DIR = REPO_ROOT / "outputs/cache/extractor_v2"


PILOT2_SUBJECTS = [
    "claude-opus-4-7",
    "deepseek-v4-pro",
    "grok-4.3",
    "gemma-4-31b-it",
    "gpt-5.5",
    "kimi-k2.6",
    "llama-3.3-70b-instruct",
    "qwen3.6-35b-a3b",
    "qwen3.6-plus",
]


def _resolve_input_path(subject_id: str) -> Path:
    manifest = V1_DIR / subject_id / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    raw = payload.get("input_path")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / raw
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in rows:
            fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _normalize_extracted(extracted: Any) -> dict[str, Any]:
    if not isinstance(extracted, dict):
        return {"actions": [], "refusal_only": False, "refusal_signaled": False}
    actions = extracted.get("actions") or []
    if not isinstance(actions, list):
        actions = []
    return {
        "actions": actions,
        "refusal_only": bool(extracted.get("refusal_only")),
        "refusal_signaled": bool(extracted.get("refusal_signaled")),
        "extractor_output": extracted,
    }


def recover_subject(subject_id: str, max_retries: int = 3) -> dict[str, Any]:
    extractions_path = V2_DIR / subject_id / "extractions.jsonl"
    if not extractions_path.exists():
        return {"subject_id": subject_id, "error": "no_extractions_file"}
    response_path = _resolve_input_path(subject_id)
    rows = _read_jsonl(response_path)
    by_cid = {r.get("custom_id"): r for r in rows if r.get("custom_id")}

    extractions = _read_jsonl(extractions_path)
    cache = ExtractorCache(cache_dir=V2_CACHE_DIR)
    extractor = Extractor(cache=cache)

    recovered: list[str] = []
    persistent: list[dict[str, str]] = []
    skipped_empty: list[str] = []

    try:
        for idx, entry in enumerate(extractions):
            err = entry.get("extraction_error")
            if not err and isinstance(entry.get("extractor_output"), dict):
                nested_parse_error = entry["extractor_output"].get("parse_error")
                if nested_parse_error:
                    err = f"parse_error: {nested_parse_error}"
            if not err:
                continue
            if err == "empty_response":
                skipped_empty.append(entry.get("custom_id", ""))
                continue
            cid = entry.get("custom_id")
            row = by_cid.get(cid) if cid else None
            if row is None:
                persistent.append(
                    {"custom_id": cid or "", "reason": "no_response_row"}
                )
                continue
            text = _subject_response_text(_raw_response(row))
            if not text.strip():
                continue

            extracted: dict[str, Any] | None = None
            last_err = ""
            for attempt in range(1, max_retries + 1):
                try:
                    extracted = extractor.extract(text, bypass_cache=True)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = f"{type(exc).__name__}: {exc}"
                    logger.warning(
                        "%s recovery attempt %d/%d failed: %s",
                        cid, attempt, max_retries, last_err,
                    )

            if extracted is None:
                persistent.append({"custom_id": cid or "", "reason": last_err[:200]})
                continue

            normalized = _normalize_extracted(extracted)
            new_entry = {"custom_id": cid, **normalized}
            extractions[idx] = new_entry
            recovered.append(cid)
    finally:
        try:
            extractor.close()
        except Exception:  # noqa: BLE001
            pass

    if recovered:
        _write_jsonl(extractions_path, extractions)

    summary = {
        "subject_id": subject_id,
        "extractions_path": str(extractions_path.relative_to(REPO_ROOT)),
        "recovered_count": len(recovered),
        "recovered_cids": recovered,
        "persistent_failures": persistent,
        "empty_response_skipped": len(skipped_empty),
    }
    logger.info(
        "%s: recovered=%d persistent=%d empty_skipped=%d",
        subject_id,
        len(recovered),
        len(persistent),
        len(skipped_empty),
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", default="")
    p.add_argument("--max-retries", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    subjects = (
        [s.strip() for s in args.subjects.split(",") if s.strip()]
        or PILOT2_SUBJECTS
    )
    summary: dict[str, Any] = {}
    for subject_id in subjects:
        try:
            summary[subject_id] = recover_subject(subject_id, args.max_retries)
        except Exception as exc:  # noqa: BLE001
            logger.exception("recovery failed for %s: %s", subject_id, exc)
            summary[subject_id] = {"error": str(exc)}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
