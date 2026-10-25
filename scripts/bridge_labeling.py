#!/usr/bin/env python3
"""Build a bridge-labeling reliability report from common-sample judge votes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.judge_bridge import build_bridge_report  # noqa: E402


ROSTER_PATH = REPO_ROOT / "config" / "roster.json"


def _jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_no}: expected JSON object")
            rows.append(value)
    return rows


def _default_candidate_judges() -> list[str]:
    roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    judges = roster.get("judges")
    if not isinstance(judges, dict) or len(judges) < 2:
        raise ValueError("config/roster.json must define at least two judges")
    return sorted(str(judge_id) for judge_id in judges)


def _read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_jsonl_objects(path))
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--votes",
        action="append",
        type=Path,
        required=True,
        help=(
            "Bridge vote JSONL file. Repeat for multiple files. Rows may be "
            "long-form {sample_id, judge, label} records or aggregate records "
            "with {custom_id, votes:[...]}. Labels accept R/P/C or full names."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Report JSON path.")
    parser.add_argument(
        "--judge",
        action="append",
        dest="judges",
        help=(
            "Expected candidate judge id. Repeat to set the bridge panel. "
            "Defaults to all judges in config/roster.json."
        ),
    )
    parser.add_argument(
        "--kappa-floor",
        type=float,
        default=0.41,
        help="Linearly weighted pairwise kappa floor. Default: 0.41.",
    )
    parser.add_argument(
        "--require-gold",
        action="store_true",
        help="Fail the gate when no complete bridge sample has gold labels.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        expected_judges = args.judges or _default_candidate_judges()
        report = build_bridge_report(
            _read_records(args.votes),
            expected_judges=expected_judges,
            kappa_floor=args.kappa_floor,
            require_gold=args.require_gold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"bridge_labeling.py: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    status = "PASS" if report["gate"]["passed"] else "FAIL"
    print(
        "bridge-labeling report "
        f"{status}: complete_samples={report['n_complete_samples']} "
        f"judges={','.join(report['expected_judges'])} "
        f"output={args.output}"
    )
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
