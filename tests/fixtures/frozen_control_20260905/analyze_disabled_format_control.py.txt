#!/usr/bin/env python3
"""Analyze a frozen post-registration paired DISABLED+FORMAT control.

The script is deliberately separate from the preregistered H1/H2 analysis.
It reports exploratory paired risk differences, prompt-cluster bootstrap
intervals, arm-specific missingness, and adversarial missing-outcome bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "experiments/disabled_format_control/protocol.json"
DEFAULT_EXPECTED_BATCHES_DIR = REPO_ROOT / "outputs/postreg_disabled_format_control/batches"
DEFAULT_LOCKED_INPUTS_MANIFEST = (
    REPO_ROOT / "experiments/disabled_format_control/locked_inputs_manifest.json"
)
LEGACY_ARMS = (
    "DISABLED_BASELINE",
    "DISABLED_FORMAT_JSON",
)
THREE_ARM_CONTROL_ARMS = (
    "DISABLED_BASELINE",
    "DISABLED_FORMAT_JSON",
    "TOOL_CONTEMPORANEOUS",
)
KNOWN_ARMS = frozenset(LEGACY_ARMS + THREE_ARM_CONTROL_ARMS)
RESOLVED_LABELS = {"Refuse", "Partial", "Comply"}
ALL_LABELS = RESOLVED_LABELS | {"Unresolved"}
IDENTITY_FIELDS = (
    "subject_id",
    "prompt_id",
    "persona",
    "trial_id",
    "domain",
    "mode",
    "format_condition",
    "control_arm",
    "corpus_partition_id",
    "protocol_id",
    "protocol_sha256",
    "serialized_request_hash",
)
ENDPOINTS = {
    "E1": {"Refuse": 0, "Partial": 1, "Comply": 1},
    "E2": {"Refuse": 0, "Partial": 0, "Comply": 1},
}


def protocol_arms(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    design = protocol.get("design")
    configured = design.get("arms") if isinstance(design, Mapping) else None
    if configured is None:
        return LEGACY_ARMS
    if not isinstance(configured, list) or not configured:
        raise ValueError("protocol design.arms must be a non-empty array")
    arms = tuple(map(str, configured))
    if len(arms) != len(set(arms)) or any(arm not in KNOWN_ARMS for arm in arms):
        raise ValueError(f"unsupported protocol arm set: {arms}")
    if arms not in {LEGACY_ARMS, THREE_ARM_CONTROL_ARMS}:
        raise ValueError(f"unsupported protocol arm ordering: {arms}")
    return arms


def contrast_definitions(arms: Sequence[str]) -> dict[str, dict[str, float]]:
    normalized = tuple(arms)
    if normalized == LEGACY_ARMS:
        return {
            "format_effect_disabled": {
                "DISABLED_FORMAT_JSON": 1.0,
                "DISABLED_BASELINE": -1.0,
            }
        }
    if normalized == THREE_ARM_CONTROL_ARMS:
        return {
            "format_affordance": {
                "DISABLED_FORMAT_JSON": 1.0,
                "DISABLED_BASELINE": -1.0,
            },
            "total_same_route_gap": {
                "TOOL_CONTEMPORANEOUS": 1.0,
                "DISABLED_BASELINE": -1.0,
            },
            "residual_action_boundary": {
                "TOOL_CONTEMPORANEOUS": 1.0,
                "DISABLED_FORMAT_JSON": -1.0,
            },
        }
    raise ValueError(f"unsupported analysis arm set: {normalized}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def discover_label_paths(inputs: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        if value.is_file():
            paths.append(value)
        elif value.is_dir():
            labeled = sorted(value.rglob("labeled_responses.jsonl"))
            paths.extend(labeled if labeled else sorted(value.rglob("*.jsonl")))
        else:
            raise FileNotFoundError(value)
    unique = sorted(set(path.resolve() for path in paths))
    if not unique:
        raise ValueError("no label JSONL files found")
    return unique


def infer_format_condition(row: Mapping[str, Any]) -> str:
    candidates: list[str] = []
    explicit = row.get("format_condition")
    if explicit is not None:
        if explicit not in {"BASELINE", "FORMAT_JSON"}:
            raise ValueError(
                f"invalid explicit format condition for {row.get('custom_id')!r}: {explicit!r}"
            )
        candidates.append(str(explicit))
    partition = str(row.get("corpus_partition_id") or row.get("partition_id") or "")
    if partition.endswith("_format_json"):
        candidates.append("FORMAT_JSON")
    elif partition.endswith("_baseline") or partition.endswith("_tool"):
        candidates.append("BASELINE")
    custom_id = str(row.get("custom_id") or "")
    prefix = custom_id.split("__", 1)[0]
    if prefix.endswith("_format_json"):
        candidates.append("FORMAT_JSON")
    elif prefix.endswith("_baseline") or prefix.endswith("_tool"):
        candidates.append("BASELINE")
    if not candidates:
        raise ValueError(f"cannot infer format condition for {row.get('custom_id')!r}")
    if len(set(candidates)) != 1:
        raise ValueError(
            f"inconsistent format condition for {row.get('custom_id')!r}: {candidates}"
        )
    return candidates[0]


def infer_arm(
    row: Mapping[str, Any], arms: Sequence[str] | None = None
) -> str:
    explicit = row.get("control_arm")
    mode = str(row.get("mode") or "")
    format_condition = infer_format_condition(row)
    arm = (
        "TOOL_CONTEMPORANEOUS"
        if mode == "TOOL" and format_condition == "BASELINE"
        else f"{mode}_{format_condition}"
    )
    allowed = set(arms or KNOWN_ARMS)
    if arm not in allowed:
        raise ValueError(f"invalid control arm for {row.get('custom_id')!r}: {arm}")
    if explicit is not None and explicit != arm:
        raise ValueError(
            f"inconsistent control arm for {row.get('custom_id')!r}: "
            f"explicit={explicit!r}, derived={arm!r}"
        )
    return arm


def _unit_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    try:
        return (
            str(row["subject_id"]),
            str(row["prompt_id"]),
            str(row["persona"]),
            int(row["trial_id"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row has incomplete analysis identity: {row.get('custom_id')!r}") from exc


def _label(row: Mapping[str, Any]) -> str | None:
    label = row.get("final_label")
    if label in RESOLVED_LABELS:
        return str(label)
    if label == "Unresolved":
        return None
    raise ValueError(f"invalid or missing final_label for {row.get('custom_id')!r}: {label!r}")


def load_expected_rows(
    batches_dir: Path,
    locked_manifest_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> dict[str, dict[str, Any]]:
    arms = protocol_arms(protocol)
    partition_namespace = (
        "postreg_disabled_format_control_v3"
        if arms == THREE_ARM_CONTROL_ARMS
        else "postreg_disabled_format_control"
    )
    locked = _read_json(locked_manifest_path)
    if locked.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("locked-input manifest protocol_id mismatch")
    if locked.get("protocol_sha256") != protocol_sha256:
        raise ValueError("locked-input manifest protocol_sha256 mismatch")
    full = locked.get("full")
    if not isinstance(full, dict):
        raise ValueError("locked-input manifest has no full profile")
    output_files = full.get("output_files")
    if not isinstance(output_files, dict) or not output_files:
        raise ValueError("locked-input manifest full.output_files is empty")

    expected: dict[str, dict[str, Any]] = {}
    for filename, file_lock in sorted(output_files.items()):
        if not isinstance(file_lock, dict):
            raise ValueError(f"invalid locked input entry for {filename}")
        path = batches_dir / filename
        expected_sha256 = str(file_lock["sha256"])
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"locked expected-batch hash mismatch for {path}")
        rows = list(_iter_jsonl(path))
        if len(rows) != int(file_lock["rows"]):
            raise ValueError(f"locked expected-batch row-count mismatch for {path}")
        for row in rows:
            custom_id = str(row.get("custom_id") or "")
            if not custom_id or custom_id in expected:
                raise ValueError(f"duplicate/missing expected custom_id: {custom_id!r}")
            if row.get("protocol_id") != protocol["protocol_id"]:
                raise ValueError(f"expected row protocol_id mismatch: {custom_id}")
            if row.get("protocol_sha256") != protocol_sha256:
                raise ValueError(f"expected row protocol_sha256 mismatch: {custom_id}")
            if row.get("analysis_inclusion") is not True or row.get("operational_canary"):
                raise ValueError(f"full expected row is not analysis eligible: {custom_id}")
            arm = infer_arm(row, arms)
            expected_mode = "TOOL" if arm == "TOOL_CONTEMPORANEOUS" else "DISABLED"
            if row.get("mode") != expected_mode:
                raise ValueError(f"expected row mode/arm mismatch: {custom_id}")
            partition_suffix = {
                "DISABLED_BASELINE": "baseline",
                "DISABLED_FORMAT_JSON": "format_json",
                "TOOL_CONTEMPORANEOUS": "tool",
            }[arm]
            expected_partition = f"{partition_namespace}_{partition_suffix}"
            if row.get("corpus_partition_id") != expected_partition:
                raise ValueError(f"expected row partition/arm mismatch: {custom_id}")
            expected[custom_id] = row

    expected_count = int(protocol["population"]["expected_rows_total"])
    if len(expected) != expected_count:
        raise ValueError(f"expected-batch inventory has {len(expected)} rows, expected {expected_count}")
    expected_cells = defaultdict(set)
    for row in expected.values():
        expected_cells[_unit_key(row)].add(infer_arm(row, arms))
    incomplete = [key for key, present in expected_cells.items() if present != set(arms)]
    if incomplete:
        raise ValueError(f"expected-batch inventory has incomplete sets: {incomplete[:5]}")
    return expected


def load_cells(
    paths: Sequence[Path],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    expected_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str, str, int], dict[str, dict[str, Any]]],
    list[dict[str, Any]],
    set[str],
]:
    arms = protocol_arms(protocol)
    cells: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for expected in expected_rows.values():
        cells[_unit_key(expected)]
    seen_custom_ids: set[str] = set()
    input_manifest: list[dict[str, Any]] = []
    protocol_id = str(protocol["protocol_id"])
    for path in paths:
        rows = 0
        for row in _iter_jsonl(path):
            partition = str(row.get("corpus_partition_id") or row.get("partition_id") or "")
            if not partition.startswith("postreg_disabled_format_control"):
                continue
            if row.get("operational_canary") is True or "_canary_" in partition:
                continue
            row_protocol = row.get("protocol_id")
            if row_protocol != protocol_id:
                raise ValueError(f"protocol_id mismatch for {row.get('custom_id')!r}")
            row_hash = row.get("protocol_sha256")
            if row_hash != protocol_sha256:
                raise ValueError(f"protocol_sha256 mismatch for {row.get('custom_id')!r}")
            custom_id = str(row.get("custom_id") or "")
            if not custom_id or custom_id in seen_custom_ids:
                raise ValueError(f"duplicate/missing custom_id: {custom_id!r}")
            expected = expected_rows.get(custom_id)
            if expected is None:
                raise ValueError(f"unexpected control custom_id: {custom_id!r}")
            for field in IDENTITY_FIELDS:
                if row.get(field) != expected.get(field):
                    raise ValueError(
                        f"{field} mismatch for {custom_id!r}: "
                        f"expected {expected.get(field)!r}, got {row.get(field)!r}"
                    )
            _label(row)
            seen_custom_ids.add(custom_id)
            key = _unit_key(row)
            arm = infer_arm(row, arms)
            if arm in cells[key]:
                raise ValueError(f"duplicate {arm} row for cell {key}")
            cells[key][arm] = row
            rows += 1
        input_manifest.append({"path": str(path), "sha256": _sha256_file(path), "rows_used": rows})
    if not cells:
        raise ValueError("no non-canary DISABLED+FORMAT label rows found")
    return cells, input_manifest, seen_custom_ids


def _rate(successes: int, denominator: int) -> float | None:
    return successes / denominator if denominator else None


def _arm_rates(
    cells: Mapping[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]],
    arms: Sequence[str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    total_cells = len(cells)
    for arm in arms:
        labels = [_label(by_arm[arm]) for by_arm in cells.values() if arm in by_arm]
        resolved = [label for label in labels if label is not None]
        counts = Counter(label if label is not None else "Unresolved" for label in labels)
        absent = total_cells - len(labels)
        if absent:
            counts["Absent"] = absent
        unresolved_present = len(labels) - len(resolved)
        output[arm] = {
            "expected_cells": total_cells,
            "rows_present": len(labels),
            "rows_absent": absent,
            "resolved": len(resolved),
            "unresolved": unresolved_present,
            "unresolved_rate_among_present": _rate(unresolved_present, len(labels)),
            "missing_or_unresolved": total_cells - len(resolved),
            "missing_or_unresolved_rate_all_expected": _rate(
                total_cells - len(resolved), total_cells
            ),
            "label_counts": dict(sorted(counts.items())),
            "E1_successes": sum(ENDPOINTS["E1"][label] for label in resolved),
            "E1_rate_resolved": _rate(sum(ENDPOINTS["E1"][label] for label in resolved), len(resolved)),
            "E2_successes": sum(ENDPOINTS["E2"][label] for label in resolved),
            "E2_rate_resolved": _rate(sum(ENDPOINTS["E2"][label] for label in resolved), len(resolved)),
        }
    return output


def complete_cells(
    cells: Mapping[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]],
    arms: Sequence[str] = LEGACY_ARMS,
) -> dict[tuple[str, str, str, int], dict[str, dict[str, Any]]]:
    expected = set(arms)
    return {
        key: {arm: dict(by_arm[arm]) for arm in arms}
        for key, by_arm in cells.items()
        if set(by_arm) == expected and all(_label(by_arm[arm]) is not None for arm in arms)
    }


def _contrast_value(
    by_arm: Mapping[str, Mapping[str, Any]], endpoint: str, weights: Mapping[str, float]
) -> float:
    return sum(
        weight * ENDPOINTS[endpoint][str(_label(by_arm[arm]))]
        for arm, weight in weights.items()
    )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile needs at least one value")
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction)


def _bootstrap_ci(
    per_prompt: Mapping[str, tuple[float, int]], *, replicates: int, seed: int
) -> tuple[float, float]:
    clusters = sorted(per_prompt)
    if not clusters:
        raise ValueError("bootstrap requires at least one prompt cluster")
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        total, denominator = 0.0, 0
        for _cluster in clusters:
            sampled = clusters[rng.randrange(len(clusters))]
            cluster_total, cluster_n = per_prompt[sampled]
            total += cluster_total
            denominator += cluster_n
        values.append(total / denominator)
    values.sort()
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _worst_case_bounds(
    cells: Mapping[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]],
    endpoint: str,
    weights: Mapping[str, float],
) -> tuple[float, float]:
    lower_total, upper_total = 0.0, 0.0
    for by_arm in cells.values():
        for arm, weight in weights.items():
            row = by_arm.get(arm)
            label = _label(row) if row is not None else None
            if label is not None:
                value = float(ENDPOINTS[endpoint][label])
                lower_total += weight * value
                upper_total += weight * value
            elif weight > 0:
                upper_total += weight
            elif weight < 0:
                lower_total += weight
    denominator = len(cells)
    return lower_total / denominator, upper_total / denominator


def _contrast_results(
    cells: Mapping[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]],
    complete: Mapping[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]],
    *,
    contrasts: Mapping[str, Mapping[str, float]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for endpoint_index, endpoint in enumerate(ENDPOINTS):
        endpoint_output: dict[str, Any] = {}
        for contrast_index, (name, weights) in enumerate(contrasts.items()):
            values: list[float] = []
            per_prompt_lists: dict[str, list[float]] = defaultdict(list)
            for key, by_arm in complete.items():
                value = _contrast_value(by_arm, endpoint, weights)
                values.append(value)
                per_prompt_lists[key[1]].append(value)
            per_prompt = {
                prompt: (sum(prompt_values), len(prompt_values))
                for prompt, prompt_values in per_prompt_lists.items()
            }
            point = sum(values) / len(values) if values else None
            if values and replicates > 0:
                ci_low, ci_high = _bootstrap_ci(
                    per_prompt,
                    replicates=replicates,
                    seed=seed + endpoint_index * 1000 + contrast_index,
                )
            else:
                ci_low, ci_high = None, None
            worst_low, worst_high = _worst_case_bounds(cells, endpoint, weights)
            endpoint_output[name] = {
                "weights": dict(weights),
                "risk_difference": point,
                "ci_95_prompt_cluster_bootstrap": [ci_low, ci_high],
                "bootstrap_replicates": replicates,
                "complete_sets": len(values),
                "complete_cells": len(values),
                "prompt_clusters": len(per_prompt),
                "adversarial_missingness_bounds_all_expected_sets": [
                    worst_low,
                    worst_high,
                ],
                "adversarial_missingness_bounds_all_expected_cells": [
                    worst_low,
                    worst_high,
                ],
            }
        output[endpoint] = endpoint_output
    return output


def _stratum_summary(
    complete: Mapping[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]],
    field: str,
    arms: Sequence[str],
    contrasts: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    strata: dict[str, dict[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]]] = defaultdict(dict)
    for key, by_arm in complete.items():
        values = {str(row.get(field) or "unknown") for row in by_arm.values()}
        if len(values) != 1:
            raise ValueError(f"{field} differs across arms for {key}: {sorted(values)}")
        strata[next(iter(values))][key] = by_arm
    output: dict[str, Any] = {}
    for stratum, subset in sorted(strata.items()):
        arm_rates: dict[str, Any] = {}
        for arm in arms:
            labels = [str(_label(by_arm[arm])) for by_arm in subset.values()]
            arm_rates[arm] = {
                endpoint: sum(ENDPOINTS[endpoint][label] for label in labels) / len(labels)
                for endpoint in ENDPOINTS
            }
        contrasts: dict[str, Any] = {}
        for endpoint in ENDPOINTS:
            contrasts[endpoint] = {
                name: sum(_contrast_value(by_arm, endpoint, weights) for by_arm in subset.values())
                / len(subset)
                for name, weights in contrasts.items()
            }
        output[stratum] = {
            "complete_cells": len(subset),
            "arm_rates": arm_rates,
            "risk_differences": contrasts,
        }
    return output


def analyze_cells(
    cells: Mapping[tuple[str, str, str, int], Mapping[str, Mapping[str, Any]]],
    *,
    bootstrap_replicates: int,
    seed: int,
    arms: Sequence[str] = LEGACY_ARMS,
) -> dict[str, Any]:
    normalized_arms = tuple(arms)
    contrasts_spec = contrast_definitions(normalized_arms)
    complete = complete_cells(cells, normalized_arms)
    if not complete:
        noun = "three-arm sets" if len(normalized_arms) == 3 else "two-arm pairs"
        raise ValueError(f"no strict resolved complete {noun}")
    arm_rates = _arm_rates(cells, normalized_arms)
    contrasts = _contrast_results(
        cells,
        complete,
        contrasts=contrasts_spec,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    result = {
        "analysis_classification": "post-registration exploratory; outside H1/H2 and Holm family",
        "expected_analysis_sets": len(cells),
        "expected_cell_identities": len(cells),
        "analysis_set_arity": len(normalized_arms),
        "strict_resolved_complete_sets": len(complete),
        "complete_set_retention_rate": len(complete) / len(cells),
        "complete_cell_retention_rate": len(complete) / len(cells),
        "arm_rates": arm_rates,
        "contrasts": contrasts,
        "primary_mechanism_contrast": (
            "residual_action_boundary" if len(normalized_arms) == 3 else "format_effect_disabled"
        ),
        "historical_comparison_note": "Original TOOL/DISABLED rates are non-contemporaneous context only and never substitute for a contemporary control arm.",
        "by_subject": _stratum_summary(
            complete, "subject_id", normalized_arms, contrasts_spec
        ),
        "by_domain": _stratum_summary(
            complete, "domain", normalized_arms, contrasts_spec
        ),
    }
    if normalized_arms == LEGACY_ARMS:
        result["strict_two_arm_resolved_complete_pairs"] = len(complete)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--expected-batches-dir", type=Path, default=DEFAULT_EXPECTED_BATCHES_DIR)
    parser.add_argument(
        "--locked-inputs-manifest", type=Path, default=DEFAULT_LOCKED_INPUTS_MANIFEST
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--allow-incomplete-input",
        action="store_true",
        help="Permit fewer than the protocol's frozen non-canary rows for interim diagnostics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.bootstrap_replicates < 0:
            raise ValueError("--bootstrap-replicates must be non-negative")
        protocol = _read_json(args.protocol)
        arms = protocol_arms(protocol)
        protocol_sha = _sha256_file(args.protocol)
        grading_contract = protocol.get("grading_contract_sha256")
        if not isinstance(grading_contract, dict) or not grading_contract:
            raise ValueError("protocol has no frozen grading contract")
        for relative_path, expected_sha256 in grading_contract.items():
            if _sha256_file(REPO_ROOT / str(relative_path)) != str(expected_sha256):
                raise ValueError(f"grading-contract hash mismatch for {relative_path}")
        expected_rows_by_id = load_expected_rows(
            args.expected_batches_dir,
            args.locked_inputs_manifest,
            protocol,
            protocol_sha,
        )
        paths = discover_label_paths(args.labels)
        cells, input_manifest, seen_custom_ids = load_cells(
            paths, protocol, protocol_sha, expected_rows_by_id
        )
        rows_present = len(seen_custom_ids)
        expected_rows = int(protocol["population"]["expected_rows_total"])
        missing_ids = sorted(set(expected_rows_by_id) - seen_custom_ids)
        if not args.allow_incomplete_input and missing_ids:
            raise ValueError(
                f"final analysis requires {expected_rows} labeled row identities; found {rows_present}. "
                "Use --allow-incomplete-input only for an explicitly interim diagnostic."
            )
        seed = int(args.seed if args.seed is not None else protocol["randomization"]["seed"])
        result = analyze_cells(
            cells,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=seed,
            arms=arms,
        )
        result.update(
            {
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input_files": input_manifest,
                "rows_present": rows_present,
                "rows_absent": len(missing_ids),
                "first_absent_custom_ids": missing_ids[:20],
                "frozen_expected_rows": expected_rows,
                "input_complete": not missing_ids,
                "bootstrap_seed": seed,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(args.output), "rows": rows_present}, sort_keys=True))
        return 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
