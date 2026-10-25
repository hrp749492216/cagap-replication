#!/usr/bin/env python3
"""Re-run MiniMax extraction on archived 1152-row pilot subject outputs.

The legacy first-pass extractions written under
``outputs/pilot/2026-05-08/judge_batches/<subject>/extractions.jsonl`` are
preserved untouched. This script writes new extractions to
``outputs/pilot/2026-05-08/extractions_v2/<subject>/extractions.jsonl``
using the second extractor prompt revision (which invalidates the previous extractor cache key) and a
separate cache directory ``outputs/cache/extractor_v2/`` so the previous
cache entries are not modified either. Both old and new extractions can
then be diffed by the extraction-comparison tooling to quantify the impact
of the Tier 1+2+3 fixes.

Usage:
    python scripts/reextract_subject_responses.py [--subjects claude-opus-4-7,...]

When ``--subjects`` is omitted all archived Pilot-2 subject directories are
processed in order.
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
from scripts.dispatch_judge_batches import (  # noqa: E402
    extract_all,
)

logging.basicConfig(
    level="INFO", format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("reextract_subject_responses")


PILOT_DATE = "2026-05-08"
V1_DIR = REPO_ROOT / "outputs/pilot" / PILOT_DATE / "judge_batches"
V2_DIR = REPO_ROOT / "outputs/pilot" / PILOT_DATE / "extractions_v2"
V2_CACHE_DIR = REPO_ROOT / "outputs/cache/extractor_v2"


def _resolve_input_path(subject_id: str) -> Path:
    """Read v1 manifest to find the canonical response file for the subject."""
    manifest = V1_DIR / subject_id / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"v1 manifest missing for {subject_id}; cannot resolve input file: {manifest}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    raw = payload.get("input_path")
    if not raw:
        raise RuntimeError(f"v1 manifest for {subject_id} has no input_path")
    path = REPO_ROOT / raw if not Path(raw).is_absolute() else Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"input file does not exist: {path}")
    return path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def reextract_subject(subject_id: str) -> dict[str, Any]:
    input_path = _resolve_input_path(subject_id)
    rows = _read_rows(input_path)
    out_dir = V2_DIR / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    extractions_path = out_dir / "extractions.jsonl"

    cache = ExtractorCache(cache_dir=V2_CACHE_DIR)
    extractor = Extractor(cache=cache)
    try:
        extractions = extract_all(rows, extractor=extractor, workers=5)
    finally:
        try:
            extractor.close()
        except Exception:  # noqa: BLE001
            pass

    with extractions_path.open("w", encoding="utf-8") as fh:
        for entry in extractions:
            fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "subject_id": subject_id,
        "input_path": str(input_path.relative_to(REPO_ROOT)),
        "extractions_path": str(extractions_path.relative_to(REPO_ROOT)),
        "extractor_prompt_version": "v2",
        "extractor_cache_dir": str(V2_CACHE_DIR.relative_to(REPO_ROOT)),
        "rows": len(rows),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "%s: %d rows extracted -> %s", subject_id, len(rows), extractions_path
    )
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subjects",
        default="",
        help=(
            "Comma-separated subject IDs (default: all archived Pilot-2 subjects). "
            "Subject IDs match v1 directory names under "
            "outputs/pilot/2026-05-08/judge_batches/."
        ),
    )
    return p.parse_args()


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


def main() -> int:
    args = parse_args()
    subjects = (
        [s.strip() for s in args.subjects.split(",") if s.strip()]
        or PILOT2_SUBJECTS
    )
    summary: dict[str, dict[str, Any]] = {}
    for subject_id in subjects:
        try:
            summary[subject_id] = reextract_subject(subject_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("re-extraction failed for %s: %s", subject_id, exc)
            summary[subject_id] = {"error": str(exc)}

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all("error" not in v for v in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
