"""Verify that stored response ledgers cover every generated batch row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.raw_completeness import (  # noqa: E402
    CompletenessError,
    assert_complete,
    build_completeness_report,
    write_report,
)


def _discover_batches(batches_dir: Path) -> list[Path]:
    return sorted(batches_dir.glob("input_*.jsonl"))


def _discover_responses(raw_dir: Path, batches_dir: Path) -> list[Path]:
    paths = sorted(raw_dir.glob("*.jsonl"))
    paths.extend(sorted(batches_dir.glob("responses_*_openrouter.jsonl")))
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", default=None)
    parser.add_argument("--response", type=Path, action="append", default=None)
    parser.add_argument("--batches-dir", type=Path, default=Path("outputs/batches"))
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/raw"))
    parser.add_argument("--report", type=Path, default=Path("outputs/audit/raw_completeness.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    batches = args.batch or _discover_batches(args.batches_dir)
    responses = args.response or _discover_responses(args.raw_dir, args.batches_dir)
    report = build_completeness_report(batches, responses)
    write_report(args.report, report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
    try:
        assert_complete(batches, responses)
    except CompletenessError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
