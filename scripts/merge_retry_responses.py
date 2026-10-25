#!/usr/bin/env python3
"""Merge original and retry response JSONL files by ``custom_id``.

Retry rows replace originals only when their canonical subject response is
non-empty. The script writes a fresh merged artifact and a small manifest; it
does not mutate either input file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extractor.extract import subject_response_from_response  # noqa: E402
from scripts.grade_subject_responses import raw_response_from_row  # noqa: E402


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.open("r", encoding="utf-8"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row must be an object")
        rows.append(row)
    return rows


def _usable(row: dict[str, Any]) -> bool:
    raw, _source = raw_response_from_row(row)
    return bool(subject_response_from_response(raw).strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--retry", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_rows = _iter_jsonl(args.original)
    retry_rows = _iter_jsonl(args.retry)
    retry_by_id = {
        str(row["custom_id"]): row
        for row in retry_rows
        if isinstance(row.get("custom_id"), str) and _usable(row)
    }
    replaced: list[str] = []
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in original_rows:
        cid = str(row.get("custom_id") or "")
        seen.add(cid)
        if cid in retry_by_id:
            merged.append(retry_by_id[cid])
            replaced.append(cid)
        else:
            merged.append(row)
    extra_retry_ids = sorted(set(retry_by_id) - seen)
    if extra_retry_ids:
        raise SystemExit(f"retry rows not present in original: {extra_retry_ids[:3]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(
            {
                "original": str(args.original),
                "retry": str(args.retry),
                "out": str(args.out),
                "original_rows": len(original_rows),
                "retry_rows": len(retry_rows),
                "retry_rows_usable": len(retry_by_id),
                "replaced_rows": len(replaced),
                "replaced_custom_ids": replaced,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(args.out), "replaced_rows": len(replaced)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
