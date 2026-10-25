"""Human-validation reviewer statistics for the 5x200 incomplete-overlap audit."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from analysis.judge_agreement import fleiss_kappa, krippendorff_alpha
from analysis.judge_bridge import _linear_weighted_kappa, normalize_label

LABELS = ("Refuse", "Partial", "Comply")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def _label_value(row: dict[str, Any]) -> Any:
    for key in ("label", "selected_label", "human_label"):
        if key in row:
            return row.get(key)
    return None


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _json_number(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _assignment_maps(
    bundle: dict[str, Any],
    expected_reviewers: list[str] | None,
) -> tuple[list[str], dict[str, set[str]], dict[str, set[str]], list[str]]:
    assignments = bundle.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("bundle must contain an assignments array")
    rows = bundle.get("rows")
    if not isinstance(rows, list):
        raise ValueError("bundle must contain a rows array")

    bundle_sample_ids = [
        row.get("sample_id")
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("sample_id"), str)
    ]
    if len(bundle_sample_ids) != len(rows):
        raise ValueError("every bundle row must have a string sample_id")
    selected_sample_ids = list(dict.fromkeys(bundle_sample_ids))
    if len(selected_sample_ids) != len(bundle_sample_ids):
        raise ValueError("bundle rows contain duplicate sample_id values")

    reviewer_order = list(expected_reviewers or [])
    assignments_by_reviewer: dict[str, set[str]] = {}
    sample_to_reviewers: dict[str, set[str]] = {sample_id: set() for sample_id in selected_sample_ids}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("assignment entries must be objects")
        reviewer = assignment.get("reviewer_id")
        sample_ids = assignment.get("sample_ids")
        if not isinstance(reviewer, str) or not reviewer:
            raise ValueError("assignment.reviewer_id must be a non-empty string")
        if not isinstance(sample_ids, list) or not all(isinstance(s, str) for s in sample_ids):
            raise ValueError(f"assignment.sample_ids for {reviewer!r} must be a string list")
        if reviewer not in reviewer_order:
            reviewer_order.append(reviewer)
        unique_sample_ids = set(sample_ids)
        if len(unique_sample_ids) != len(sample_ids):
            raise ValueError(f"assignment for {reviewer!r} contains duplicate sample ids")
        unknown = sorted(unique_sample_ids - set(selected_sample_ids))
        if unknown:
            raise ValueError(f"assignment for {reviewer!r} references unknown sample ids: {unknown[:3]}")
        assignments_by_reviewer[reviewer] = unique_sample_ids
        for sample_id in unique_sample_ids:
            sample_to_reviewers.setdefault(sample_id, set()).add(reviewer)

    for reviewer in reviewer_order:
        assignments_by_reviewer.setdefault(reviewer, set())

    return reviewer_order, assignments_by_reviewer, sample_to_reviewers, selected_sample_ids


def _collect_votes(
    label_rows: Iterable[dict[str, Any]],
    reviewer_order: list[str],
    assignments_by_reviewer: dict[str, set[str]],
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    expected = set(reviewer_order)
    votes_by_sample: dict[str, dict[str, str]] = defaultdict(dict)
    invalid_rows: list[dict[str, Any]] = []
    unassigned_rows: list[dict[str, Any]] = []
    non_primary_rows: list[dict[str, Any]] = []
    n_records = 0
    n_valid_primary = 0

    for row in label_rows:
        n_records += 1
        reviewer = row.get("reviewer_id")
        sample_id = row.get("sample_id")
        if not isinstance(reviewer, str) or reviewer not in expected:
            non_primary_rows.append({"reviewer_id": reviewer, "sample_id": sample_id})
            continue
        label = normalize_label(_label_value(row))
        if label is None:
            invalid_rows.append({"reviewer_id": reviewer, "sample_id": sample_id, "label": _label_value(row)})
            continue
        if not isinstance(sample_id, str) or sample_id not in assignments_by_reviewer[reviewer]:
            unassigned_rows.append({"reviewer_id": reviewer, "sample_id": sample_id, "label": label})
            continue
        if reviewer in votes_by_sample[sample_id]:
            raise ValueError(f"duplicate human label for sample_id={sample_id!r} reviewer={reviewer!r}")
        votes_by_sample[sample_id][reviewer] = label
        n_valid_primary += 1

    ingest = {
        "label_record_count": n_records,
        "valid_primary_label_count": n_valid_primary,
        "invalid_label_count": len(invalid_rows),
        "invalid_label_examples": invalid_rows[:10],
        "unassigned_primary_label_count": len(unassigned_rows),
        "unassigned_primary_label_examples": unassigned_rows[:10],
        "excluded_non_primary_label_count": len(non_primary_rows),
        "excluded_non_primary_label_examples": non_primary_rows[:10],
    }
    return dict(votes_by_sample), ingest


def _assignment_plan(
    reviewer_order: list[str],
    assignments_by_reviewer: dict[str, set[str]],
    sample_to_reviewers: dict[str, set[str]],
    selected_sample_ids: list[str],
) -> dict[str, Any]:
    distribution = Counter(len(sample_to_reviewers[sample_id]) for sample_id in selected_sample_ids)
    triad_counts = Counter(
        tuple(sorted(reviewers))
        for reviewers in sample_to_reviewers.values()
        if len(reviewers) == 3
    )
    pair_counts = Counter(
        tuple(sorted(reviewers))
        for reviewers in sample_to_reviewers.values()
        if len(reviewers) == 2
    )
    return {
        "expected_reviewers": reviewer_order,
        "selected_row_count": len(selected_sample_ids),
        "assignment_edge_count": sum(len(v) for v in assignments_by_reviewer.values()),
        "per_reviewer_assigned_count": {
            reviewer: len(assignments_by_reviewer[reviewer]) for reviewer in reviewer_order
        },
        "sample_coding_distribution": {
            str(coded_count): int(row_count)
            for coded_count, row_count in sorted(distribution.items())
        },
        "triple_coded_row_count": int(distribution.get(3, 0)),
        "double_coded_row_count": int(distribution.get(2, 0)),
        "triad_counts": {"|".join(k): int(v) for k, v in sorted(triad_counts.items())},
        "pair_counts": {"|".join(k): int(v) for k, v in sorted(pair_counts.items())},
    }


def _completion(
    reviewer_order: list[str],
    assignments_by_reviewer: dict[str, set[str]],
    votes_by_sample: dict[str, dict[str, str]],
) -> dict[str, Any]:
    per_reviewer: dict[str, Any] = {}
    for reviewer in reviewer_order:
        assigned = assignments_by_reviewer[reviewer]
        labels = [
            votes_by_sample[sample_id][reviewer]
            for sample_id in assigned
            if reviewer in votes_by_sample.get(sample_id, {})
        ]
        counts = Counter(labels)
        per_reviewer[reviewer] = {
            "assigned_count": len(assigned),
            "labeled_assigned_count": len(labels),
            "missing_count": len(assigned) - len(labels),
            "completion_rate": _safe_rate(len(labels), len(assigned)),
            "label_counts": {label: int(counts[label]) for label in LABELS if counts[label]},
        }
    return {
        "per_reviewer": per_reviewer,
        "total_assigned_labels_expected": sum(len(v) for v in assignments_by_reviewer.values()),
        "total_assigned_labels_observed": sum(
            row["labeled_assigned_count"] for row in per_reviewer.values()
        ),
    }


def _pairwise_agreement(
    reviewer_order: list[str],
    assignments_by_reviewer: dict[str, set[str]],
    votes_by_sample: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reviewer_a, reviewer_b in itertools.combinations(reviewer_order, 2):
        overlap = sorted(assignments_by_reviewer[reviewer_a] & assignments_by_reviewer[reviewer_b])
        labels_a: list[str] = []
        labels_b: list[str] = []
        for sample_id in overlap:
            sample_votes = votes_by_sample.get(sample_id, {})
            if reviewer_a in sample_votes and reviewer_b in sample_votes:
                labels_a.append(sample_votes[reviewer_a])
                labels_b.append(sample_votes[reviewer_b])
        n = len(labels_a)
        raw = _safe_rate(
            sum(a == b for a, b in zip(labels_a, labels_b, strict=True)),
            n,
        )
        rows.append(
            {
                "reviewer_a": reviewer_a,
                "reviewer_b": reviewer_b,
                "assigned_overlap_count": len(overlap),
                "both_labeled_count": n,
                "raw_agreement": raw,
                "linear_weighted_kappa": _json_number(
                    _linear_weighted_kappa(labels_a, labels_b) if n else None
                ),
            }
        )
    return rows


def _count_matrix(label_rows: list[list[str]]) -> np.ndarray:
    matrix = np.zeros((len(label_rows), len(LABELS)), dtype=float)
    for row_index, labels in enumerate(label_rows):
        for label in labels:
            matrix[row_index, LABEL_TO_INDEX[label]] += 1.0
    return matrix


def _coded_subset_agreement(
    sample_to_reviewers: dict[str, set[str]],
    votes_by_sample: dict[str, dict[str, str]],
    coded_count: int,
) -> dict[str, Any]:
    sample_ids = sorted(
        sample_id
        for sample_id, reviewers in sample_to_reviewers.items()
        if len(reviewers) == coded_count
    )
    complete_rows: list[list[str]] = []
    unanimous = 0
    majority_2_1 = 0
    no_majority = 0
    agree_count = 0
    majority_labels: list[str] = []
    for sample_id in sample_ids:
        reviewers = sorted(sample_to_reviewers[sample_id])
        sample_votes = votes_by_sample.get(sample_id, {})
        if not all(reviewer in sample_votes for reviewer in reviewers):
            continue
        labels = [sample_votes[reviewer] for reviewer in reviewers]
        complete_rows.append(labels)
        counts = Counter(labels)
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count]
        if coded_count == 2:
            agree_count += int(max_count == 2)
        if max_count == coded_count:
            unanimous += 1
        elif max_count == 2 and coded_count == 3:
            majority_2_1 += 1
        else:
            no_majority += 1
        if len(winners) == 1:
            majority_labels.append(winners[0])

    fleiss_value = None
    if complete_rows:
        try:
            fleiss_value = fleiss_kappa(_count_matrix(complete_rows))
        except ValueError:
            fleiss_value = None

    result: dict[str, Any] = {
        "assigned_row_count": len(sample_ids),
        "complete_row_count": len(complete_rows),
        "complete_rate": _safe_rate(len(complete_rows), len(sample_ids)),
        "fleiss_kappa": _json_number(fleiss_value),
        "majority_label_counts": dict(Counter(majority_labels)),
    }
    if coded_count == 3:
        result.update(
            {
                "unanimous_count": unanimous,
                "unanimity_rate": _safe_rate(unanimous, len(complete_rows)),
                "majority_2_1_count": majority_2_1,
                "no_majority_count": no_majority,
            }
        )
    elif coded_count == 2:
        result.update(
            {
                "agree_count": agree_count,
                "raw_agreement": _safe_rate(agree_count, len(complete_rows)),
                "disagree_count": len(complete_rows) - agree_count,
            }
        )
    return result


def _krippendorff(
    reviewer_order: list[str],
    selected_sample_ids: list[str],
    votes_by_sample: dict[str, dict[str, str]],
) -> float | None:
    samples_with_overlap = sum(
        1
        for sample_id in selected_sample_ids
        if sum(
            1
            for reviewer in reviewer_order
            if votes_by_sample.get(sample_id, {}).get(reviewer) is not None
        )
        >= 2
    )
    if samples_with_overlap == 0:
        return None
    data = [
        [votes_by_sample.get(sample_id, {}).get(reviewer) for sample_id in selected_sample_ids]
        for reviewer in reviewer_order
    ]
    return _json_number(krippendorff_alpha(np.asarray(data, dtype=object)))


def _machine_agreement(
    key_rows: Iterable[dict[str, Any]] | None,
    reviewer_order: list[str],
    assignments_by_reviewer: dict[str, set[str]],
    sample_to_reviewers: dict[str, set[str]],
    votes_by_sample: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if key_rows is None:
        return {"available": False}
    machine_by_sample: dict[str, str] = {}
    universe_counts: Counter[str] = Counter()
    selected_sample_ids = set(sample_to_reviewers)
    ignored_keyed_sample_count = 0
    for row in key_rows:
        sample_id = row.get("sample_id")
        label = normalize_label(row.get("final_machine_label"))
        if isinstance(sample_id, str) and label is not None:
            if sample_id not in selected_sample_ids:
                ignored_keyed_sample_count += 1
                continue
            machine_by_sample[sample_id] = label
            if isinstance(row.get("audit_universe"), str):
                universe_counts[row["audit_universe"]] += 1

    per_reviewer: dict[str, Any] = {}
    for reviewer in reviewer_order:
        n = 0
        agree = 0
        abs_error = 0
        for sample_id in assignments_by_reviewer[reviewer]:
            machine = machine_by_sample.get(sample_id)
            human = votes_by_sample.get(sample_id, {}).get(reviewer)
            if machine is None or human is None:
                continue
            n += 1
            agree += int(human == machine)
            abs_error += abs(LABEL_TO_INDEX[human] - LABEL_TO_INDEX[machine])
        per_reviewer[reviewer] = {
            "n": n,
            "accuracy": _safe_rate(agree, n),
            "mean_absolute_error": _safe_rate(abs_error, n),
        }

    majority_n = 0
    majority_agree = 0
    majority_abs_error = 0
    insufficient_human_label_count = 0
    tie_or_no_majority_count = 0
    for sample_id, reviewers in sample_to_reviewers.items():
        machine = machine_by_sample.get(sample_id)
        if machine is None:
            continue
        labels = [
            votes_by_sample.get(sample_id, {}).get(reviewer)
            for reviewer in reviewers
            if votes_by_sample.get(sample_id, {}).get(reviewer) is not None
        ]
        if len(labels) < 2:
            insufficient_human_label_count += 1
            continue
        counts = Counter(labels)
        max_count = max(counts.values())
        winners = [label for label, count in counts.items() if count == max_count]
        if len(winners) != 1:
            tie_or_no_majority_count += 1
            continue
        majority = winners[0]
        majority_n += 1
        majority_agree += int(majority == machine)
        majority_abs_error += abs(LABEL_TO_INDEX[majority] - LABEL_TO_INDEX[machine])

    return {
        "available": True,
        "keyed_sample_count": len(machine_by_sample),
        "ignored_keyed_sample_count": ignored_keyed_sample_count,
        "key_universe_counts": dict(universe_counts),
        "per_reviewer": per_reviewer,
        "majority_vote": {
            "n": majority_n,
            "insufficient_human_label_count": insufficient_human_label_count,
            "tie_or_no_majority_count": tie_or_no_majority_count,
            "accuracy": _safe_rate(majority_agree, majority_n),
            "mean_absolute_error": _safe_rate(majority_abs_error, majority_n),
        },
    }


def build_human_validation_reviewer_report(
    label_rows: Iterable[dict[str, Any]],
    assignment_bundle: dict[str, Any],
    *,
    key_rows: Iterable[dict[str, Any]] | None = None,
    expected_reviewers: list[str] | None = None,
) -> dict[str, Any]:
    """Build completion and agreement statistics for human-validation reviewers."""

    reviewer_order, assignments_by_reviewer, sample_to_reviewers, selected_sample_ids = (
        _assignment_maps(assignment_bundle, expected_reviewers)
    )
    votes_by_sample, ingest = _collect_votes(label_rows, reviewer_order, assignments_by_reviewer)
    return {
        "schema_version": "cagap_human_validation_reviewer_stats_v1",
        "purpose": (
            "Descriptive human-validation audit statistics for the 1,000-rating "
            "balanced incomplete-overlap design. These metrics do not update H1 estimates."
        ),
        "assignment_plan": _assignment_plan(
            reviewer_order,
            assignments_by_reviewer,
            sample_to_reviewers,
            selected_sample_ids,
        ),
        "label_ingest": ingest,
        "completion": _completion(reviewer_order, assignments_by_reviewer, votes_by_sample),
        "agreement": {
            "pairwise": _pairwise_agreement(
                reviewer_order,
                assignments_by_reviewer,
                votes_by_sample,
            ),
            "triple_coded": _coded_subset_agreement(sample_to_reviewers, votes_by_sample, 3),
            "double_coded": _coded_subset_agreement(sample_to_reviewers, votes_by_sample, 2),
            "krippendorff_alpha_ordinal": _krippendorff(
                reviewer_order,
                selected_sample_ids,
                votes_by_sample,
            ),
        },
        "machine_label_agreement": _machine_agreement(
            key_rows,
            reviewer_order,
            assignments_by_reviewer,
            sample_to_reviewers,
            votes_by_sample,
        ),
    }
