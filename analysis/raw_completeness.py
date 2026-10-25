"""Completeness checks for raw/labeled response ledgers.

The experiment's inferential unit starts from the generated batch JSONL rows.
Before grading or H1 analysis, every expected ``custom_id`` should appear in a
stored response ledger, even if the response itself is an explicit provider
error that later grades to ``Unresolved``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from grader.json_strict import strict_json_loads


class CompletenessError(RuntimeError):
    """Raised when a response ledger is not complete relative to batch inputs."""


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = strict_json_loads(line)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid strict JSON row: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object.")
            yield parsed


def _custom_id(row: dict[str, Any], *, path: Path) -> str:
    value = row.get("custom_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: row missing non-empty string custom_id.")
    return value.strip()


def _has_explicit_error(row: dict[str, Any]) -> bool:
    if row.get("error") or row.get("errors") or row.get("missingness_reason"):
        return True
    response = row.get("response")
    if isinstance(response, dict) and (response.get("error") or response.get("errors")):
        return True
    raw_response = row.get("raw_response")
    if isinstance(raw_response, dict) and (raw_response.get("error") or raw_response.get("errors")):
        return True
    return False


def build_completeness_report(
    expected_paths: Iterable[Path],
    observed_paths: Iterable[Path],
) -> dict[str, Any]:
    expected_seen: dict[str, str] = {}
    observed_seen: dict[str, str] = {}
    duplicate_expected: list[str] = []
    duplicate_observed: list[str] = []
    error_observed = 0

    expected_path_list = [Path(path) for path in expected_paths]
    observed_path_list = [Path(path) for path in observed_paths]

    for path in expected_path_list:
        for row in _iter_jsonl(path):
            custom_id = _custom_id(row, path=path)
            if custom_id in expected_seen:
                duplicate_expected.append(custom_id)
            else:
                expected_seen[custom_id] = str(path)

    for path in observed_path_list:
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            custom_id = _custom_id(row, path=path)
            if custom_id in observed_seen:
                duplicate_observed.append(custom_id)
            else:
                observed_seen[custom_id] = str(path)
            if _has_explicit_error(row):
                error_observed += 1

    expected_ids = set(expected_seen)
    observed_ids = set(observed_seen)
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    duplicate_expected = sorted(set(duplicate_expected))
    duplicate_observed = sorted(set(duplicate_observed))
    complete = not (missing or unexpected or duplicate_expected or duplicate_observed)
    return {
        "complete": complete,
        "expected_count": len(expected_seen),
        "observed_count": len(observed_seen),
        "error_observed_count": error_observed,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "duplicate_expected_count": len(duplicate_expected),
        "duplicate_observed_count": len(duplicate_observed),
        "missing_custom_ids": missing,
        "unexpected_custom_ids": unexpected,
        "duplicate_expected_custom_ids": duplicate_expected,
        "duplicate_observed_custom_ids": duplicate_observed,
        "expected_paths": [str(path) for path in expected_path_list],
        "observed_paths": [str(path) for path in observed_path_list],
    }


def _failure_message(report: dict[str, Any]) -> str:
    parts: list[str] = []
    if report["missing_count"]:
        parts.append(f"missing {report['missing_count']} expected raw row(s)")
    if report["unexpected_count"]:
        parts.append(f"{report['unexpected_count']} unexpected response row(s)")
    if report["duplicate_expected_count"]:
        parts.append(f"{report['duplicate_expected_count']} duplicate expected custom_id(s)")
    if report["duplicate_observed_count"]:
        parts.append(f"{report['duplicate_observed_count']} duplicate observed custom_id(s)")
    return "raw completeness check failed: " + "; ".join(parts)


def assert_complete(
    expected_paths: Iterable[Path],
    observed_paths: Iterable[Path],
) -> dict[str, Any]:
    report = build_completeness_report(expected_paths, observed_paths)
    if not report["complete"]:
        raise CompletenessError(_failure_message(report))
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
