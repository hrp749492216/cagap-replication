#!/usr/bin/env python3
"""Analyze benign capability battery outputs.

This is intentionally separate from ``scripts/analyze_results.py``. The benign
capability battery is an OSF-declared secondary/descriptive diagnostic: it is
not an H1 filter, not part of the Holm family, and cannot rescue or exclude
primary rows. The optional hardening-utility profile is a post-registration
paired descriptive check of baseline versus hardened TOOL rows. Inputs are the
deterministic labels from ``scripts/grade_capability_battery.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GRADED = Path("outputs/benign_capability_battery/graded/labeled_responses.jsonl")
DEFAULT_BATCHES_DIR = Path("outputs/benign_capability_battery/batches")
DEFAULT_OUTPUT_DIR = Path("outputs/benign_capability_battery/analysis")
DEFAULT_HARDENING_UTILITY_GRADED = Path(
    "outputs/benign_capability_battery/hardening_utility_600/graded/labeled_responses.jsonl"
)
DEFAULT_HARDENING_UTILITY_BATCHES_DIR = Path(
    "outputs/benign_capability_battery/hardening_utility_600/batches"
)
DEFAULT_HARDENING_UTILITY_OUTPUT_DIR = Path(
    "outputs/benign_capability_battery/hardening_utility_600/analysis"
)

PLANNED_SUBJECTS = 10
PLANNED_PROMPTS = 30
PLANNED_MODES = ("NOTOOLS", "DISABLED", "TOOL", "TOOL_IRRELEVANT")
PLANNED_ROWS_PER_SUBJECT = PLANNED_PROMPTS * len(PLANNED_MODES)
PLANNED_TOTAL_ROWS = PLANNED_SUBJECTS * PLANNED_ROWS_PER_SUBJECT
DESCRIPTIVE_ANALYSIS_PROFILE = "descriptive_1200"
HARDENING_UTILITY_ANALYSIS_PROFILE = "benign_tool_hardening_utility"
HARDENING_UTILITY_MODES = ("TOOL",)
HARDENING_UTILITY_INTERVENTIONS = ("baseline", "hardened")
HARDENING_UTILITY_ROWS_PER_SUBJECT = (
    PLANNED_PROMPTS * len(HARDENING_UTILITY_MODES) * len(HARDENING_UTILITY_INTERVENTIONS)
)
HARDENING_UTILITY_TOTAL_ROWS = PLANNED_SUBJECTS * HARDENING_UTILITY_ROWS_PER_SUBJECT

NONCALLABLE_OR_IRRELEVANT_MODES = frozenset({"DISABLED", "NOTOOLS", "TOOL_IRRELEVANT"})


@dataclass(frozen=True)
class MetricSpec:
    name: str
    description: str
    eligible_modes: frozenset[str] | None = None


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        "mode_success",
        "Mode-specific descriptive success: exact call in TOOL, structured intent in DISABLED, or availability calibration in non-callable/non-relevant modes.",
    ),
    MetricSpec(
        "TOOL_correct_call",
        "TOOL arm exact benign target-tool call with exact canonical arguments.",
        frozenset({"TOOL"}),
    ),
    MetricSpec(
        "TOOL_false_refusal",
        "TOOL arm false refusal or incorrect tool-unavailable claim.",
        frozenset({"TOOL"}),
    ),
    MetricSpec(
        "structured_intent_success",
        "DISABLED arm non-executed structured intent for the correct benign target and canonical arguments.",
        frozenset({"DISABLED"}),
    ),
    MetricSpec("wrong_tool", "Any executed or emitted wrong tool call."),
    MetricSpec("wrong_args", "Correct target tool with non-canonical arguments."),
    MetricSpec(
        "hallucinated_answer",
        "Non-callable/non-relevant arm answer that fabricates unavailable data.",
        NONCALLABLE_OR_IRRELEVANT_MODES,
    ),
    MetricSpec(
        "availability_calibrated",
        "Non-callable/non-relevant arm response that accurately states tool or data access is unavailable.",
        NONCALLABLE_OR_IRRELEVANT_MODES,
    ),
    MetricSpec(
        "unexpected_executed_call",
        "Exact target call emitted outside the TOOL arm.",
        NONCALLABLE_OR_IRRELEVANT_MODES,
    ),
    MetricSpec("api_error", "Provider or fetch error row."),
)

TABLE_SPECS: dict[str, tuple[str, ...]] = {
    "overall": (),
    "by_subject": ("subject_id",),
    "by_mode": ("mode",),
    "by_intervention_condition": ("intervention_condition",),
    "by_mode_intervention_condition": ("mode", "intervention_condition"),
    "by_subject_intervention_condition": ("subject_id", "intervention_condition"),
    "by_subject_mode": ("subject_id", "mode"),
    "by_route": ("route",),
    "by_route_mode": ("route", "mode"),
    "by_provider": ("provider",),
    "by_provider_mode": ("provider", "mode"),
    "by_domain": ("domain",),
    "by_domain_mode": ("domain", "mode"),
    "by_subject_domain_mode": ("subject_id", "domain", "mode"),
    "by_benign_subtype_mode": ("benign_subtype", "mode"),
    "by_battery_grade": ("battery_grade",),
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_dumps(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        default=str,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            yield row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_dumps(payload, indent=2) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _format_pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.1f}%"


def _format_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "NA"
    return f"{_format_pct(low)} to {_format_pct(high)}"


def _metric_count_rate(row: dict[str, Any], metric_name: str) -> str:
    numerator = int(row.get(f"{metric_name}_n") or 0)
    denominator = int(row.get(f"{metric_name}_denominator") or 0)
    if denominator <= 0:
        return "NA"
    return f"{numerator}/{denominator} ({_format_pct(row.get(f'{metric_name}_rate'))})"


def _metric_ci(row: dict[str, Any], metric_name: str) -> str:
    return _format_ci(
        row.get(f"{metric_name}_ci95_low"),
        row.get(f"{metric_name}_ci95_high"),
    )


def _md_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(_md_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def _metric_value(row: dict[str, Any], metric_name: str) -> bool:
    grade = str(row.get("battery_grade") or "")
    mode = str(row.get("mode") or "")
    if metric_name == "mode_success":
        return _boolish(row.get("mode_success"))
    if metric_name == "TOOL_correct_call":
        return _boolish(row.get("TOOL_correct_call")) or grade == "tool_correct_call"
    if metric_name == "TOOL_false_refusal":
        return _boolish(row.get("TOOL_false_refusal")) or (mode == "TOOL" and grade == "false_refusal")
    if metric_name == "structured_intent_success":
        return _boolish(row.get("structured_intent_success")) or grade == "structured_intent"
    if metric_name == "wrong_tool":
        return _boolish(row.get("wrong_tool")) or grade == "wrong_tool"
    if metric_name == "wrong_args":
        return _boolish(row.get("wrong_args")) or grade == "wrong_args"
    if metric_name == "hallucinated_answer":
        return _boolish(row.get("hallucinated_answer")) or grade == "hallucinated_answer"
    if metric_name == "availability_calibrated":
        return _boolish(row.get("availability_calibrated")) or grade == "availability_calibrated"
    if metric_name == "unexpected_executed_call":
        return grade == "unexpected_executed_call"
    if metric_name == "api_error":
        return grade == "api_error" or bool(row.get("api_error") or row.get("error"))
    raise KeyError(metric_name)


def _eligible_for_metric(row: dict[str, Any], metric: MetricSpec) -> bool:
    if metric.eligible_modes is None:
        return True
    return str(row.get("mode") or "") in metric.eligible_modes


def _wilson_interval(numerator: int, denominator: int, *, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if denominator <= 0:
        return None, None
    p_hat = numerator / denominator
    z2 = z * z
    denom = 1.0 + z2 / denominator
    center = (p_hat + z2 / (2.0 * denominator)) / denom
    half = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * denominator)) / denominator) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _exact_two_sided_binomial_p(successes: int, trials: int) -> float | None:
    if trials <= 0:
        return None
    tail = min(successes, trials - successes)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def _counts(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(str(row.get(column) if row.get(column) is not None else "missing") for row in rows).items()
        )
    )


def _nonmissing_values(rows: list[dict[str, Any]], column: str) -> set[str]:
    return {
        str(row[column])
        for row in rows
        if row.get(column) is not None and str(row.get(column)) != "missing"
    }


def _analysis_profile_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed_modes = _nonmissing_values(rows, "mode")
    observed_interventions = _nonmissing_values(rows, "intervention_condition")
    observed_profiles = _nonmissing_values(rows, "battery_profile")
    observed_battery_ids = _nonmissing_values(rows, "battery_id")

    hardening_profile_detected = (
        HARDENING_UTILITY_ANALYSIS_PROFILE in observed_profiles
        or HARDENING_UTILITY_ANALYSIS_PROFILE in observed_battery_ids
        or (
            observed_modes == set(HARDENING_UTILITY_MODES)
            and observed_interventions == set(HARDENING_UTILITY_INTERVENTIONS)
        )
    )
    if hardening_profile_detected:
        return {
            "analysis_profile": HARDENING_UTILITY_ANALYSIS_PROFILE,
            "registration_status": "post_registration_descriptive_extension",
            "source": (
                "600-row benign hardening-utility rerun using the locked "
                "30-prompt benign battery and fixed ten-subject panel"
            ),
            "planned_subjects": PLANNED_SUBJECTS,
            "planned_prompts": PLANNED_PROMPTS,
            "planned_modes": list(HARDENING_UTILITY_MODES),
            "planned_intervention_conditions": list(HARDENING_UTILITY_INTERVENTIONS),
            "planned_rows_per_subject": HARDENING_UTILITY_ROWS_PER_SUBJECT,
            "planned_total_rows": HARDENING_UTILITY_TOTAL_ROWS,
            "interpretation": (
                "Paired post-registration descriptive utility check: estimates whether "
                "the hardened prompt changes benign callable-tool accuracy."
            ),
            "secondary_rescue_prohibition": (
                "Hardening-utility results do not enter H1, Holm adjustment, primary row "
                "inclusion, or interpretation rescue."
            ),
        }

    return {
        "analysis_profile": DESCRIPTIVE_ANALYSIS_PROFILE,
        "registration_status": "osf_declared_secondary_descriptive",
        "source": "paper/OSF-CAGAP-paste.txt lines declaring benign capability battery; paper/Study_Design.md §18",
        "planned_subjects": PLANNED_SUBJECTS,
        "planned_prompts": PLANNED_PROMPTS,
        "planned_modes": list(PLANNED_MODES),
        "planned_intervention_conditions": ["descriptive"],
        "planned_rows_per_subject": PLANNED_ROWS_PER_SUBJECT,
        "planned_total_rows": PLANNED_TOTAL_ROWS,
        "interpretation": (
            "Capability floor sanity check: descriptive, not used for exclusion in the primary. "
            "Per-subject capability rates are reported alongside primary results."
        ),
        "secondary_rescue_prohibition": (
            "Benign capability results do not enter H1, Holm adjustment, primary row inclusion, "
            "or interpretation rescue."
        ),
    }


def _default_paths_for_profile(profile: str) -> dict[str, Path]:
    if profile == HARDENING_UTILITY_ANALYSIS_PROFILE:
        return {
            "graded": DEFAULT_HARDENING_UTILITY_GRADED,
            "batches_dir": DEFAULT_HARDENING_UTILITY_BATCHES_DIR,
            "output_dir": DEFAULT_HARDENING_UTILITY_OUTPUT_DIR,
        }
    if profile == DESCRIPTIVE_ANALYSIS_PROFILE:
        return {
            "graded": DEFAULT_GRADED,
            "batches_dir": DEFAULT_BATCHES_DIR,
            "output_dir": DEFAULT_OUTPUT_DIR,
        }
    raise ValueError(f"unknown benign capability analysis profile: {profile!r}")


def _group_key(row: dict[str, Any], columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(column) if row.get(column) is not None else "missing") for column in columns)


def _group_key_label(values: tuple[str, ...], columns: tuple[str, ...]) -> str:
    if not columns:
        return "ALL"
    return "|".join(f"{column}={value}" for column, value in zip(columns, values, strict=True))


def _summarize_groups(
    rows: list[dict[str, Any]],
    *,
    group_name: str,
    columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    if columns:
        for row in rows:
            grouped.setdefault(_group_key(row, columns), []).append(row)
    else:
        grouped[()] = list(rows)

    out: list[dict[str, Any]] = []
    for values in sorted(grouped):
        group_rows = grouped[values]
        record: dict[str, Any] = {
            "group_name": group_name,
            "group_key": _group_key_label(values, columns),
            "total_rows": len(group_rows),
            "distinct_subjects": len({str(row.get("subject_id")) for row in group_rows if row.get("subject_id") is not None}),
            "distinct_prompts": len({str(row.get("benign_id") or row.get("prompt_id")) for row in group_rows if row.get("benign_id") is not None or row.get("prompt_id") is not None}),
            "distinct_domains": len({str(row.get("domain")) for row in group_rows if row.get("domain") is not None}),
            "mode_counts": _counts(group_rows, "mode"),
            "battery_grade_counts": _counts(group_rows, "battery_grade"),
            "route_counts": _counts(group_rows, "route"),
            "provider_counts": _counts(group_rows, "provider"),
        }
        for column, value in zip(columns, values, strict=True):
            record[column] = value
        for metric in METRICS:
            eligible = [row for row in group_rows if _eligible_for_metric(row, metric)]
            numerator = sum(1 for row in eligible if _metric_value(row, metric.name))
            denominator = len(eligible)
            ci_low, ci_high = _wilson_interval(numerator, denominator)
            record[f"{metric.name}_n"] = numerator
            record[f"{metric.name}_denominator"] = denominator
            record[f"{metric.name}_rate"] = _rate(numerator, denominator)
            record[f"{metric.name}_ci95_low"] = ci_low
            record[f"{metric.name}_ci95_high"] = ci_high
        out.append(record)
    return out


def _metric_table(summary_tables: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_descriptions = {metric.name: metric.description for metric in METRICS}
    for table_name, table_rows in summary_tables.items():
        for table_row in table_rows:
            for metric in METRICS:
                rows.append(
                    {
                        "table": table_name,
                        "group_name": table_row["group_name"],
                        "group_key": table_row["group_key"],
                        "metric": metric.name,
                        "description": metric_descriptions[metric.name],
                        "total_rows": table_row["total_rows"],
                        "numerator": table_row[f"{metric.name}_n"],
                        "denominator": table_row[f"{metric.name}_denominator"],
                        "rate": table_row[f"{metric.name}_rate"],
                        "ci95_low": table_row[f"{metric.name}_ci95_low"],
                        "ci95_high": table_row[f"{metric.name}_ci95_high"],
                    }
                )
    return rows


def _custom_id_list(rows: Iterable[dict[str, Any]], *, source: Path) -> list[str]:
    ids: list[str] = []
    for row in rows:
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise ValueError(f"{source}: row missing non-empty custom_id")
        ids.append(custom_id)
    return ids


def _duplicate_counts(ids: Iterable[str]) -> dict[str, int]:
    counts = Counter(ids)
    return dict(sorted((key, value) for key, value in counts.items() if value > 1))


def _load_expected_ids(paths: list[Path]) -> list[str]:
    expected: list[str] = []
    for path in paths:
        expected.extend(_custom_id_list(_iter_jsonl(path), source=path))
    return expected


def _completeness_report(
    *,
    rows: list[dict[str, Any]],
    graded_path: Path,
    expected_batch_paths: list[Path],
    strict_completeness: bool,
) -> dict[str, Any]:
    if strict_completeness and not expected_batch_paths:
        raise ValueError(
            "strict completeness requires expected batch file(s); pass "
            "--expected-batch, provide --batches-dir containing input_*.jsonl, "
            "or use --no-strict-completeness for exploratory partial analysis"
        )
    observed_ids = _custom_id_list(rows, source=graded_path)
    observed_duplicates = _duplicate_counts(observed_ids)
    expected_ids = _load_expected_ids(expected_batch_paths) if expected_batch_paths else []
    expected_duplicates = _duplicate_counts(expected_ids)

    observed_set = set(observed_ids)
    expected_set = set(expected_ids)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set) if expected_batch_paths else []
    custom_id_set_matches_expected = bool(expected_batch_paths) and not missing and not extra and not observed_duplicates and not expected_duplicates

    report = {
        "observed_rows": len(rows),
        "observed_unique_custom_ids": len(observed_set),
        "observed_duplicate_custom_ids": observed_duplicates,
        "expected_rows": len(expected_ids) if expected_batch_paths else None,
        "expected_unique_custom_ids": len(expected_set) if expected_batch_paths else None,
        "expected_duplicate_custom_ids": expected_duplicates,
        "expected_batch_paths": [str(path) for path in expected_batch_paths],
        "custom_id_set_matches_expected": custom_id_set_matches_expected,
        "missing_expected_custom_ids_count": len(missing),
        "missing_expected_custom_ids_sample": missing[:10],
        "extra_observed_custom_ids_count": len(extra),
        "extra_observed_custom_ids_sample": extra[:10],
        "observed_subject_count": len({str(row.get("subject_id")) for row in rows if row.get("subject_id") is not None}),
        "observed_prompt_count": len({str(row.get("benign_id") or row.get("prompt_id")) for row in rows if row.get("benign_id") is not None or row.get("prompt_id") is not None}),
        "observed_mode_counts": _counts(rows, "mode"),
        "observed_intervention_condition_counts": _counts(rows, "intervention_condition"),
        "observed_subject_mode_counts": _subject_mode_counts(rows),
    }
    if strict_completeness:
        problems = []
        if observed_duplicates:
            problems.append(f"duplicate observed custom_id(s): {list(observed_duplicates)[:5]}")
        if expected_duplicates:
            problems.append(f"duplicate expected custom_id(s): {list(expected_duplicates)[:5]}")
        if missing:
            problems.append(f"missing expected custom_id(s): {missing[:5]}")
        if extra:
            problems.append(f"extra observed custom_id(s): {extra[:5]}")
        if problems:
            raise ValueError("; ".join(problems))
    return report


def _subject_mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        f"{row.get('subject_id', 'missing')}|{row.get('mode', 'missing')}"
        for row in rows
    )
    return dict(sorted(counts.items()))


def _pair_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    prompt_id = str(row.get("benign_id") or row.get("prompt_id") or "")
    return (
        str(row.get("subject_id") or ""),
        prompt_id,
        str(row.get("persona") or ""),
        str(row.get("trial_id") if row.get("trial_id") is not None else ""),
    )


def _paired_hardening_utility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if str(row.get("mode") or "") == "TOOL"
        and str(row.get("intervention_condition") or "") in {"baseline", "hardened"}
    ]
    if not eligible:
        return {
            "status": "not_applicable",
            "reason": "no baseline/hardened TOOL rows present",
            "scope": "paired_baseline_vs_hardened_tool_rows",
        }

    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = {}
    duplicates: list[dict[str, str]] = []
    for row in eligible:
        key = _pair_key(row)
        condition = str(row.get("intervention_condition"))
        condition_rows = grouped.setdefault(key, {})
        if condition in condition_rows:
            duplicates.append(
                {
                    "subject_id": key[0],
                    "prompt_id": key[1],
                    "persona": key[2],
                    "trial_id": key[3],
                    "intervention_condition": condition,
                }
            )
            continue
        condition_rows[condition] = row

    if duplicates:
        return {
            "status": "error",
            "reason": "duplicate condition rows for paired hardening utility key",
            "duplicate_count": len(duplicates),
            "duplicate_sample": duplicates[:10],
            "scope": "paired_baseline_vs_hardened_tool_rows",
        }

    complete_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    incomplete_pairs = 0
    for condition_rows in grouped.values():
        if {"baseline", "hardened"}.issubset(condition_rows):
            complete_pairs.append((condition_rows["baseline"], condition_rows["hardened"]))
        else:
            incomplete_pairs += 1

    if not complete_pairs:
        return {
            "status": "not_applicable",
            "reason": "no complete baseline/hardened TOOL pairs present",
            "incomplete_pair_count": incomplete_pairs,
            "scope": "paired_baseline_vs_hardened_tool_rows",
        }

    baseline_values = [
        1 if _metric_value(baseline, "TOOL_correct_call") else 0
        for baseline, _hardened in complete_pairs
    ]
    hardened_values = [
        1 if _metric_value(hardened, "TOOL_correct_call") else 0
        for _baseline, hardened in complete_pairs
    ]
    deltas = [hardened - baseline for baseline, hardened in zip(baseline_values, hardened_values, strict=True)]
    n_pairs = len(complete_pairs)
    baseline_success = sum(baseline_values)
    hardened_success = sum(hardened_values)
    hardened_success_baseline_failure = sum(
        1 for baseline, hardened in zip(baseline_values, hardened_values, strict=True)
        if baseline == 0 and hardened == 1
    )
    hardened_failure_baseline_success = sum(
        1 for baseline, hardened in zip(baseline_values, hardened_values, strict=True)
        if baseline == 1 and hardened == 0
    )
    discordant = hardened_success_baseline_failure + hardened_failure_baseline_success
    mean_delta = sum(deltas) / n_pairs
    if n_pairs > 1:
        variance = sum((delta - mean_delta) ** 2 for delta in deltas) / (n_pairs - 1)
        se = math.sqrt(variance / n_pairs)
        ci_low = mean_delta - 1.959963984540054 * se
        ci_high = mean_delta + 1.959963984540054 * se
    else:
        ci_low = None
        ci_high = None

    return {
        "status": "ok",
        "scope": "paired_baseline_vs_hardened_tool_rows",
        "endpoint": "TOOL_correct_call",
        "n_pairs": n_pairs,
        "incomplete_pair_count": incomplete_pairs,
        "baseline_correct_call_n": baseline_success,
        "baseline_correct_call_rate": _rate(baseline_success, n_pairs),
        "hardened_correct_call_n": hardened_success,
        "hardened_correct_call_rate": _rate(hardened_success, n_pairs),
        "risk_difference_hardened_minus_baseline": mean_delta,
        "risk_difference_ci95_low": ci_low,
        "risk_difference_ci95_high": ci_high,
        "risk_difference_ci95_method": "normal_approximation_over_complete_pair_deltas",
        "n_both_success": sum(
            1 for baseline, hardened in zip(baseline_values, hardened_values, strict=True)
            if baseline == 1 and hardened == 1
        ),
        "n_both_failure": sum(
            1 for baseline, hardened in zip(baseline_values, hardened_values, strict=True)
            if baseline == 0 and hardened == 0
        ),
        "n_hardened_success_baseline_failure": hardened_success_baseline_failure,
        "n_hardened_failure_baseline_success": hardened_failure_baseline_success,
        "discordant_pairs": discordant,
        "p_value_exact_binomial": _exact_two_sided_binomial_p(
            hardened_success_baseline_failure,
            discordant,
        ),
        "p_value_method": "two_sided_exact_binomial_over_discordant_pairs",
        "interpretation": (
            "Post-registration descriptive benign-utility check; not an H1 filter, "
            "not part of Holm correction, and not used to rescue or exclude primary rows."
        ),
    }


def _discover_expected_batches(batches_dir: Path) -> list[Path]:
    if not batches_dir.exists():
        return []
    return sorted(batches_dir.glob("input_*.jsonl"))


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    summary_tables: dict[str, list[dict[str, Any]]],
) -> None:
    completeness = summary["completeness"]
    overall = summary_tables["overall"][0]
    by_mode = {
        row["mode"]: row
        for row in summary_tables["by_mode"]
        if isinstance(row.get("mode"), str)
    }
    by_subject = sorted(summary_tables["by_subject"], key=lambda row: str(row.get("subject_id") or ""))

    overall_metrics = [
        ("Mode success", "mode_success"),
        ("TOOL correct call", "TOOL_correct_call"),
        ("TOOL false refusal", "TOOL_false_refusal"),
        ("DISABLED structured intent", "structured_intent_success"),
        ("Availability calibrated", "availability_calibrated"),
        ("Hallucinated answer", "hallucinated_answer"),
        ("Unexpected executed call", "unexpected_executed_call"),
        ("Wrong tool", "wrong_tool"),
        ("Wrong args", "wrong_args"),
        ("API error", "api_error"),
    ]
    mode_order = [mode for mode in PLANNED_MODES if mode in by_mode] + [
        mode for mode in sorted(by_mode) if mode not in PLANNED_MODES
    ]
    mode_rows = [
        [
            mode,
            by_mode[mode]["total_rows"],
            _metric_count_rate(by_mode[mode], "mode_success"),
            _metric_count_rate(by_mode[mode], "TOOL_correct_call"),
            _metric_count_rate(by_mode[mode], "structured_intent_success"),
            _metric_count_rate(by_mode[mode], "availability_calibrated"),
            _metric_count_rate(by_mode[mode], "hallucinated_answer"),
            _metric_count_rate(by_mode[mode], "api_error"),
        ]
        for mode in mode_order
    ]
    subject_rows = [
        [
            row.get("subject_id", "missing"),
            row["total_rows"],
            _metric_count_rate(row, "mode_success"),
            _metric_count_rate(row, "TOOL_correct_call"),
            _metric_count_rate(row, "TOOL_false_refusal"),
            _metric_count_rate(row, "structured_intent_success"),
            _metric_count_rate(row, "availability_calibrated"),
            _metric_count_rate(row, "hallucinated_answer"),
            _metric_count_rate(row, "api_error"),
            _json_dumps(row.get("battery_grade_counts", {})),
        ]
        for row in by_subject
    ]
    scope_sentence = (
        "Scope: post-registration descriptive hardening-utility analysis. These results "
        "are not an H1 filter, do not enter Holm adjustment, and do not rescue or exclude "
        "primary rows."
        if summary.get("analysis_profile") == HARDENING_UTILITY_ANALYSIS_PROFILE
        else (
            "Scope: secondary descriptive capability-floor analysis. These results are "
            "not an H1 filter, do not enter Holm adjustment, and do not rescue or exclude "
            "primary rows."
        )
    )

    lines = [
        "# Benign Capability Battery Analysis",
        "",
        f"Generated at: `{summary['generated_at']}`",
        f"Analysis profile: `{summary['analysis_profile']}`",
        "",
        scope_sentence,
        "",
        "## Completeness",
        "",
        f"- Observed rows: {completeness['observed_rows']}",
        f"- Expected rows: {completeness['expected_rows']}",
        f"- Custom ID set matches expected: {str(completeness['custom_id_set_matches_expected']).lower()}",
        f"- Observed subjects: {completeness['observed_subject_count']}",
        f"- Observed prompts: {completeness['observed_prompt_count']}",
        f"- Observed mode counts: `{_json_dumps(completeness['observed_mode_counts'])}`",
        f"- Observed intervention condition counts: `{_json_dumps(completeness['observed_intervention_condition_counts'])}`",
        "",
        "## Overall Metrics",
        "",
        _markdown_table(
            ["Metric", "n/den", "Rate", "Wilson 95% CI"],
            [
                [
                    label,
                    f"{int(overall.get(f'{metric}_n') or 0)}/{int(overall.get(f'{metric}_denominator') or 0)}",
                    _format_pct(overall.get(f"{metric}_rate")),
                    _metric_ci(overall, metric),
                ]
                for label, metric in overall_metrics
            ],
        ),
        "",
    ]
    hardening_utility = summary.get("hardening_utility", {})
    if hardening_utility.get("status") == "ok":
        lines.extend(
            [
                "## Hardening Utility",
                "",
                (
                    "Paired baseline versus hardened TOOL rows measure whether the locked "
                    "hardened system prompt changes benign callable-tool accuracy. This is "
                    "post-registration descriptive context, not an H1 filter."
                ),
                "",
                _markdown_table(
                    ["Metric", "Value"],
                    [
                        ["Complete pairs", hardening_utility["n_pairs"]],
                        [
                            "baseline TOOL correct-call rate",
                            (
                                f"{hardening_utility['baseline_correct_call_n']}/"
                                f"{hardening_utility['n_pairs']} "
                                f"({_format_pct(hardening_utility['baseline_correct_call_rate'])})"
                            ),
                        ],
                        [
                            "hardened TOOL correct-call rate",
                            (
                                f"{hardening_utility['hardened_correct_call_n']}/"
                                f"{hardening_utility['n_pairs']} "
                                f"({_format_pct(hardening_utility['hardened_correct_call_rate'])})"
                            ),
                        ],
                        [
                            "RD hardened minus baseline",
                            _format_pct(
                                hardening_utility[
                                    "risk_difference_hardened_minus_baseline"
                                ]
                            ),
                        ],
                        [
                            "Discordant pairs",
                            hardening_utility["discordant_pairs"],
                        ],
                        [
                            "Exact binomial p",
                            hardening_utility["p_value_exact_binomial"]
                            if hardening_utility["p_value_exact_binomial"] is not None
                            else "NA",
                        ],
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Mode Summary",
            "",
            _markdown_table(
                [
                    "Mode",
                    "Rows",
                    "Mode success",
                    "TOOL correct",
                    "Structured intent",
                    "Availability calibrated",
                    "Hallucinated answer",
                    "API error",
                ],
                mode_rows,
            ),
            "",
            "## Subject Summary",
            "",
            _markdown_table(
                [
                    "Subject",
                    "Rows",
                    "Mode success",
                    "TOOL correct",
                    "TOOL false refusal",
                    "Structured intent",
                    "Availability calibrated",
                    "Hallucinated answer",
                    "API error",
                    "Grade counts",
                ],
                subject_rows,
            ),
            "",
            "## Artifacts",
            "",
            f"- Graded input: `{summary['graded_input']['path']}`",
            f"- Summary JSON: `{path.with_name('summary.json')}`",
            f"- Metric table CSV: `{path.with_name('metric_table.csv')}`",
            f"- Per-subject CSV: `{path.with_name('by_subject.csv')}`",
            f"- Per-subject-by-mode CSV: `{path.with_name('by_subject_mode.csv')}`",
            *(
                [f"- Hardening utility JSON: `{hardening_utility['json']}`"]
                if hardening_utility.get("json")
                else []
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_capability_battery(
    *,
    graded_path: Path = DEFAULT_GRADED,
    expected_batch_paths: list[Path] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    strict_completeness: bool = True,
) -> dict[str, Any]:
    rows = _read_jsonl(graded_path)
    expected_batch_paths = list(expected_batch_paths or [])
    completeness = _completeness_report(
        rows=rows,
        graded_path=graded_path,
        expected_batch_paths=expected_batch_paths,
        strict_completeness=strict_completeness,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_tables = {
        name: _summarize_groups(rows, group_name=name, columns=columns)
        for name, columns in TABLE_SPECS.items()
    }
    metric_rows = _metric_table(summary_tables)
    hardening_utility = _paired_hardening_utility(rows)
    analysis_profile = _analysis_profile_plan(rows)

    for name, table_rows in summary_tables.items():
        _write_json(output_dir / f"{name}.json", table_rows)
        _write_csv(output_dir / f"{name}.csv", table_rows)
    _write_json(output_dir / "metric_table.json", metric_rows)
    _write_csv(output_dir / "metric_table.csv", metric_rows)
    hardening_utility_path = output_dir / "hardening_utility.json"
    _write_json(hardening_utility_path, hardening_utility)

    summary = {
        "analysis_scope": "secondary_descriptive_not_h1_filter",
        "analysis_profile": analysis_profile["analysis_profile"],
        "generated_at": _now_utc(),
        "graded_input": _file_fingerprint(graded_path),
        "expected_batches": [_file_fingerprint(path) for path in expected_batch_paths],
        "analysis_code": _file_fingerprint(Path(__file__)),
        "osf_alignment": analysis_profile,
        "metric_definitions": {
            metric.name: {
                "description": metric.description,
                "eligible_modes": sorted(metric.eligible_modes) if metric.eligible_modes is not None else "all",
                "interval": "Wilson 95% binomial confidence interval",
            }
            for metric in METRICS
        },
        "completeness": completeness,
        "hardening_utility": {
            **hardening_utility,
            "json": str(hardening_utility_path),
        },
        "tables": {
            name: {
                "rows": len(table_rows),
                "json": str(output_dir / f"{name}.json"),
                "csv": str(output_dir / f"{name}.csv"),
            }
            for name, table_rows in summary_tables.items()
        },
        "metric_table": {
            "rows": len(metric_rows),
            "json": str(output_dir / "metric_table.json"),
            "csv": str(output_dir / "metric_table.csv"),
        },
    }
    report_path = output_dir / "report.md"
    _write_report(report_path, summary=summary, summary_tables=summary_tables)
    summary["report"] = {"markdown": str(report_path)}
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(DESCRIPTIVE_ANALYSIS_PROFILE, HARDENING_UTILITY_ANALYSIS_PROFILE),
        default=DESCRIPTIVE_ANALYSIS_PROFILE,
        help=(
            "Default path profile. Use benign_tool_hardening_utility for the "
            "600-row baseline/hardened TOOL rerun."
        ),
    )
    parser.add_argument("--graded", type=Path, default=None)
    parser.add_argument("--expected-batch", type=Path, action="append", default=None)
    parser.add_argument("--batches-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--no-discover-expected",
        action="store_true",
        help="Do not auto-discover input_*.jsonl batch files when --expected-batch is omitted.",
    )
    parser.add_argument(
        "--no-strict-completeness",
        action="store_true",
        help="Write analysis artifacts even if graded custom_ids do not match expected batches.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile_paths = _default_paths_for_profile(args.profile)
    graded_path = args.graded or profile_paths["graded"]
    batches_dir = args.batches_dir or profile_paths["batches_dir"]
    output_dir = args.output_dir or profile_paths["output_dir"]
    expected = list(args.expected_batch or [])
    if not expected and not args.no_discover_expected:
        expected = _discover_expected_batches(batches_dir)
    summary = analyze_capability_battery(
        graded_path=graded_path,
        expected_batch_paths=expected,
        output_dir=output_dir,
        strict_completeness=not args.no_strict_completeness,
    )
    print(
        "analyzed {rows} benign capability row(s); tables={tables}; output_dir={output_dir}".format(
            rows=summary["completeness"]["observed_rows"],
            tables=len(summary["tables"]),
            output_dir=output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
