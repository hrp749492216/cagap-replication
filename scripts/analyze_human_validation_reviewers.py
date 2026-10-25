#!/usr/bin/env python3
"""Analyze human-validation reviewer completion and agreement statistics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.human_validation_reviewers import build_human_validation_reviewer_report  # noqa: E402


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


def _csv_objects(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _read_label_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            rows.extend(_csv_objects(path))
        else:
            rows.extend(_jsonl_objects(path))
    return rows


def _read_bundle(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    completion = report["completion"]
    assignment = report["assignment_plan"]
    agreement = report["agreement"]
    machine = report["machine_label_agreement"]
    lines = [
        "# Human Validation Reviewer Statistics",
        "",
        "## Design Check",
        "",
        f"- Selected rows: {assignment['selected_row_count']}",
        f"- Assignment edges: {assignment['assignment_edge_count']}",
        f"- Triple-coded rows: {assignment['triple_coded_row_count']}",
        f"- Double-coded rows: {assignment['double_coded_row_count']}",
        "",
        "## Completion",
        "",
    ]
    for reviewer, row in completion["per_reviewer"].items():
        rate = row["completion_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.1%}"
        lines.append(
            f"- {reviewer}: {row['labeled_assigned_count']} / {row['assigned_count']} ({rate_text})"
        )
    lines.extend(
        [
            "",
            "## Agreement",
            "",
            (
                "- Triple-coded complete rows: "
                f"{agreement['triple_coded']['complete_row_count']} / "
                f"{agreement['triple_coded']['assigned_row_count']}"
            ),
            (
                "- Triple-coded unanimous rows: "
                f"{agreement['triple_coded']['unanimous_count']}"
            ),
            (
                "- Double-coded complete rows: "
                f"{agreement['double_coded']['complete_row_count']} / "
                f"{agreement['double_coded']['assigned_row_count']}"
            ),
            (
                "- Ordinal Krippendorff alpha: "
                f"{agreement['krippendorff_alpha_ordinal']}"
            ),
            "",
        ]
    )
    if machine["available"]:
        majority = machine["majority_vote"]
        lines.extend(
            [
                "## Machine Agreement",
                "",
                f"- Selected keyed rows: {machine['keyed_sample_count']}",
                f"- Ignored keyed rows outside reviewer frame: {machine['ignored_keyed_sample_count']}",
                f"- Majority-vote rows used: {majority['n']}",
                (
                    "- Majority-vote rows with insufficient human labels: "
                    f"{majority['insufficient_human_label_count']}"
                ),
                f"- Majority-vote tied/no-majority rows: {majority['tie_or_no_majority_count']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        action="append",
        type=Path,
        required=True,
        help="Admin-export labels file, JSONL or CSV. Repeat for multiple files.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("human_validation_reviewer_app/data/human_validation_5x200_reviewer_app.json"),
        help="Reviewer assignment bundle JSON. Defaults to the 5x200 app bundle.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        help="Optional locked human-validation key JSONL for machine-label comparison.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Report JSON output path.")
    parser.add_argument("--markdown-output", type=Path, help="Optional Markdown summary output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_human_validation_reviewer_report(
            _read_label_rows(args.labels),
            _read_bundle(args.bundle),
            key_rows=_jsonl_objects(args.key) if args.key else None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"analyze_human_validation_reviewers.py: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, args.markdown_output)
    observed = report["completion"]["total_assigned_labels_observed"]
    expected = report["completion"]["total_assigned_labels_expected"]
    print(
        "human-validation reviewer report "
        f"labels={observed}/{expected} "
        f"triple_rows={report['assignment_plan']['triple_coded_row_count']} "
        f"double_rows={report['assignment_plan']['double_coded_row_count']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
