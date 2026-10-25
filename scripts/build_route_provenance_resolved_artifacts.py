#!/usr/bin/env python3
"""Build non-mutating route/provenance-resolved analysis artifacts.

The locked final-label JSONL files are intentionally left untouched. This
script writes a derived label set that hydrates missing route/provenance fields
from canonical run manifests, batch inputs, and schema-visibility evidence, then
emits targeted route-audit / H1 analysis summaries from the derived inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.raw_completeness import build_completeness_report  # noqa: E402
from grader.json_strict import strict_json_loads  # noqa: E402
from scripts import analyze_results as ar  # noqa: E402


DEFAULT_RUN_DIR = Path("outputs/main_run/20260525T164059Z")
PROVENANCE_FIELDS = (
    "route",
    "provider",
    "pinned_provider",
    "route_family",
    "disabled_arm_schema_visibility",
    "route_support_status",
    "schema_visibility_evidence_id",
    "schema_visibility_evidence_sha256",
    "schema_visibility_route_label",
    "serving_precision",
    "serving_precision_value",
    "serving_precision_role",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = strict_json_loads(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid strict JSON row: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(parsed)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False))
            handle.write("\n")
    os.replace(tmp, path)


def _missing(value: Any) -> bool:
    return value is None or value == ""


def _count_missing(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        field: sum(1 for row in rows if _missing(row.get(field)))
        for field in PROVENANCE_FIELDS
    }


def _load_json(path: Path) -> dict[str, Any]:
    parsed = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected JSON object")
    return parsed


def _classification_sha(run_dir: Path, relative_path: str) -> str:
    path = (run_dir.parents[2] / relative_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return _sha256(path)


def _gpt55_native_overlay(run_dir: Path) -> dict[str, Any]:
    evidence = (
        "outputs/pilot/2026-05-26/schema_visibility_smoke/"
        "20260526T023941Z_gpt55_native_batch/schema_visibility_classification.json"
    )
    full_manifest = _load_json(run_dir / "manifests/gpt-5.5_openai_native_full.json")
    blocked_manifest = _load_json(run_dir / "manifests/gpt-5.5_openai_native_blocked.json")
    return {
        "route": "openai_native_batch",
        "provider": "openai",
        "pinned_provider": "openai",
        "route_family": "native_batch",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_evidence_id": evidence,
        "schema_visibility_evidence_sha256": _classification_sha(run_dir, evidence),
        "schema_visibility_route_label": "gpt-5.5::native_BATCH",
        "serving_precision": "provider_unspecified",
        "serving_precision_value": "provider_unspecified",
        "serving_precision_role": "primary",
        "provenance_resolution_note": (
            "Derived from OpenAI native batch manifests; OpenRouter GPT-5.5 "
            "subject artifacts are superseded by the native batch route."
        ),
        "provenance_resolution_sources": [
            str(run_dir / "post_openai_chain.log"),
            str(run_dir / "manifests/gpt-5.5_openai_native_full.json"),
            str(run_dir / "manifests/gpt-5.5_openai_native_blocked.json"),
            str(run_dir / "responses/responses_gpt-5.5_openai_native_batch.jsonl"),
            "outputs/pilot/2026-05-26/schema_visibility_smoke/"
            "20260526T023941Z_gpt55_native_batch/schema_visibility_classification.json",
        ],
        "provider_batch_ids": [
            full_manifest.get("provider_batch_id"),
            blocked_manifest.get("provider_batch_id"),
        ],
    }


def _subject_overlays(run_dir: Path) -> dict[str, dict[str, Any]]:
    shared_20260519 = (
        "outputs/pilot/2026-05-19/schema_visibility_smoke/"
        "20260519T223440Z/schema_visibility_classification.json"
    )
    kimi_evidence = (
        "outputs/pilot/2026-05-19/schema_visibility_smoke/"
        "20260519T230041Z/schema_visibility_classification.json"
    )
    grok_manifest = _load_json(run_dir / "manifests/grok-4.3_native.json")
    kimi_manifest = _load_json(run_dir / "manifests/kimi-k2.6_native.json")
    return {
        "claude-opus-4-7": {
            "disabled_arm_schema_visibility": "schema_visible",
            "route_support_status": "supported",
            "schema_visibility_evidence_id": shared_20260519,
            "schema_visibility_evidence_sha256": _classification_sha(run_dir, shared_20260519),
            "schema_visibility_route_label": "claude-opus-4-7::openrouter",
            "provenance_resolution_note": (
                "Final labels already carry route/provider fields; this overlay "
                "adds the schema-visibility gate fields from the archived route audit."
            ),
            "provenance_resolution_sources": [
                str(run_dir / "final_labels/claude-opus-4-7/labeled_responses.jsonl"),
                shared_20260519,
            ],
        },
        "gpt-5.5": _gpt55_native_overlay(run_dir),
        "grok-4.3": {
            "route": "native_batch",
            "provider": "xai",
            "pinned_provider": "xai",
            "route_family": "native_batch",
            "disabled_arm_schema_visibility": "schema_visible",
            "route_support_status": "supported",
            "schema_visibility_evidence_id": shared_20260519,
            "schema_visibility_evidence_sha256": _classification_sha(run_dir, shared_20260519),
            "schema_visibility_route_label": "grok-4.3::native",
            "serving_precision": "provider_unspecified",
            "serving_precision_value": "provider_unspecified",
            "serving_precision_role": "primary",
            "provenance_resolution_note": (
                "Derived from xAI native batch manifest and normalized response envelope."
            ),
            "provenance_resolution_sources": [
                str(run_dir / "manifests/grok-4.3_native.json"),
                str(run_dir / "responses/responses_grok-4.3_native_batch.jsonl"),
                shared_20260519,
            ],
            "provider_batch_ids": [grok_manifest.get("provider_batch_id")],
        },
        "kimi-k2.6": {
            "route": "native_batch",
            "provider": "moonshot",
            "pinned_provider": "moonshot",
            "route_family": "native_batch",
            "disabled_arm_schema_visibility": "schema_visible",
            "route_support_status": "supported",
            "schema_visibility_evidence_id": kimi_evidence,
            "schema_visibility_evidence_sha256": _classification_sha(run_dir, kimi_evidence),
            "schema_visibility_route_label": "kimi-k2.6::native",
            "serving_precision": "provider_unspecified",
            "serving_precision_value": "provider_unspecified",
            "serving_precision_role": "primary",
            "provenance_resolution_note": (
                "Derived from Moonshot native batch manifest and normalized response envelope."
            ),
            "provenance_resolution_sources": [
                str(run_dir / "manifests/kimi-k2.6_native.json"),
                str(run_dir / "responses/responses_kimi-k2.6_native_batch.jsonl"),
                kimi_evidence,
            ],
            "provider_batch_ids": [kimi_manifest.get("provider_batch_id")],
        },
    }


def _apply_overlay(
    rows: list[dict[str, Any]],
    overlay: dict[str, Any],
    *,
    overlay_manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filled: Counter[str] = Counter()
    conflicts: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    row_overlay_id = str(overlay_manifest_path)
    fields = {key: value for key, value in overlay.items() if key in PROVENANCE_FIELDS}
    for row in rows:
        derived = dict(row)
        for key, value in fields.items():
            current = derived.get(key)
            if _missing(current):
                derived[key] = value
                filled[key] += 1
            elif current != value:
                conflicts.append(
                    {
                        "custom_id": str(derived.get("custom_id", "")),
                        "field": key,
                        "existing": current,
                        "overlay": value,
                    }
                )
        if fields:
            derived["route_provenance_overlay_manifest"] = row_overlay_id
        out.append(derived)
    return out, {"filled": dict(sorted(filled.items())), "conflicts": conflicts[:20], "conflict_count": len(conflicts)}


def build_derived_labels(run_dir: Path, derived_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    final_labels_dir = run_dir / "final_labels"
    if not final_labels_dir.exists():
        raise FileNotFoundError(final_labels_dir)
    overlay_manifest_path = derived_dir / "route_provenance_overlay_manifest.json"
    overlays = _subject_overlays(run_dir)
    label_paths = sorted(final_labels_dir.glob("*/labeled_responses.jsonl"))
    derived_inputs: list[Path] = []
    per_subject: dict[str, Any] = {}
    source_files: list[dict[str, Any]] = []
    output_files: list[dict[str, Any]] = []

    for label_path in label_paths:
        subject_id = label_path.parent.name
        rows = _read_jsonl(label_path)
        before_missing = _count_missing(rows)
        overlay = overlays.get(subject_id, {})
        derived_rows, overlay_summary = _apply_overlay(
            rows,
            overlay,
            overlay_manifest_path=overlay_manifest_path,
        )
        after_missing = _count_missing(derived_rows)
        out_path = derived_dir / "final_labels_route_provenance_resolved" / subject_id / "labeled_responses.jsonl"
        _write_jsonl(out_path, derived_rows)
        derived_inputs.append(out_path)
        source_files.append(_fingerprint(label_path))
        output_files.append(_fingerprint(out_path))
        per_subject[subject_id] = {
            "rows": len(rows),
            "source": str(label_path),
            "derived": str(out_path),
            "overlay_fields": sorted(key for key in overlay if key in PROVENANCE_FIELDS),
            "before_missing": before_missing,
            "after_missing": after_missing,
            "overlay_summary": overlay_summary,
            "provenance_resolution_note": overlay.get("provenance_resolution_note"),
            "provenance_resolution_sources": overlay.get("provenance_resolution_sources", []),
            "provider_batch_ids": overlay.get("provider_batch_ids", []),
        }

    manifest = {
        "schema_version": 1,
        "artifact_role": "non_mutating_route_provenance_overlay",
        "locked_final_label_files_modified": False,
        "run_dir": str(run_dir),
        "derived_dir": str(derived_dir),
        "source_final_label_files": source_files,
        "derived_final_label_files": output_files,
        "provenance_fields": list(PROVENANCE_FIELDS),
        "per_subject": per_subject,
        "notes": [
            "Derived files are analysis/report inputs only and do not replace locked final-label artifacts.",
            "GPT-5.5 subject provenance uses OpenAI native batch manifests; superseded OpenRouter subject rows are not used as canonical route evidence.",
        ],
    }
    ar._write_json(overlay_manifest_path, manifest)
    return derived_inputs, manifest


def _expected_batch_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "batches").glob("input_*.jsonl"))


def build_targeted_analysis(
    *,
    run_dir: Path,
    derived_inputs: list[Path],
    analysis_dir: Path,
    bootstrap_iter: int,
    bootstrap_seed: int,
    workers: int,
) -> dict[str, Any]:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    expected_batches = _expected_batch_paths(run_dir)
    completeness = build_completeness_report(expected_batches, derived_inputs)
    ar._write_json(analysis_dir / "raw_completeness_report.json", completeness)
    if not completeness["complete"]:
        raise RuntimeError(
            "derived label completeness check failed; see "
            f"{analysis_dir / 'raw_completeness_report.json'}"
        )

    raw_df = ar.load_all_data([str(path) for path in derived_inputs])
    df = ar._normalize_analysis_columns(raw_df)
    outputs = {
        "analysis_exclusions": ar.summarize_analysis_exclusions(raw_df),
        "h1_schema_visibility_gate": ar.build_h1_schema_visibility_gate_summary(df),
        "route_audit_summary": ar.build_route_audit_summary(df),
        "primary_h1": ar.build_primary_h1_results(
            df,
            bootstrap_iter=bootstrap_iter,
            bootstrap_seed=bootstrap_seed,
            workers=workers,
        ),
    }
    parallel_jobs: list[ar.AnalysisJob] = [
        ("domain_homogeneity", "build_domain_homogeneity", (df,), {}),
        ("subject_tier_heterogeneity", "build_subject_tier_heterogeneity", (df,), {}),
        (
            "model_dependence_robustness",
            "build_model_dependence_robustness",
            (df,),
            {
                "bootstrap_iter": 0,
                "bootstrap_seed": bootstrap_seed,
                "primary_h1": outputs["primary_h1"],
            },
        ),
        ("ordinal_gee_secondary", "build_ordinal_gee_secondary", (df,), {}),
        ("threshold_gee_secondary", "build_threshold_gee_secondary", (df,), {}),
        ("contextual_comparisons", "build_contextual_comparisons", (df,), {}),
        ("intervention_h2", "build_intervention_h2", (df,), {}),
        ("sen_q_quantization_audit", "build_sen_q_quantization_audit", (df,), {}),
    ]
    outputs.update(ar._run_analysis_jobs(parallel_jobs, workers=workers))
    for name, payload in outputs.items():
        ar._write_json(analysis_dir / f"{name}.json", payload)

    previous_route = run_dir / "analysis_final/route_audit_summary.json"
    previous_gate = run_dir / "analysis_final/h1_schema_visibility_gate.json"
    comparison = {
        "previous_route_audit_summary": _load_json(previous_route) if previous_route.exists() else None,
        "resolved_route_audit_summary": outputs["route_audit_summary"],
        "previous_h1_schema_visibility_gate": _load_json(previous_gate) if previous_gate.exists() else None,
        "resolved_h1_schema_visibility_gate": outputs["h1_schema_visibility_gate"],
        "derived_inputs": [str(path) for path in derived_inputs],
        "bootstrap_iter": bootstrap_iter,
        "bootstrap_seed": bootstrap_seed,
    }
    ar._write_json(analysis_dir / "route_provenance_resolution_comparison.json", comparison)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/derived_route_provenance_resolved",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/analysis_final_route_provenance_resolved",
    )
    parser.add_argument("--bootstrap-iter", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers for independent analysis jobs; default is CPU count minus two.",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    derived_dir = args.derived_dir or run_dir / "derived_route_provenance_resolved"
    analysis_dir = args.analysis_dir or run_dir / "analysis_final_route_provenance_resolved"

    derived_inputs, manifest = build_derived_labels(run_dir, derived_dir)
    print(f"wrote derived label manifest: {derived_dir / 'route_provenance_overlay_manifest.json'}")
    print(f"derived input files: {len(derived_inputs)}")
    if args.skip_analysis:
        return
    outputs = build_targeted_analysis(
        run_dir=run_dir,
        derived_inputs=derived_inputs,
        analysis_dir=analysis_dir,
        bootstrap_iter=args.bootstrap_iter,
        bootstrap_seed=args.bootstrap_seed,
        workers=ar._normalize_analysis_worker_count(args.workers),
    )
    print(f"wrote targeted analysis outputs: {analysis_dir}")
    print(
        "resolved missing route rows:",
        outputs["route_audit_summary"].get("missing_route_rows"),
    )
    print(
        "resolved H1 schema-gate excluded clusters:",
        outputs["h1_schema_visibility_gate"].get("excluded_clusters"),
    )
    print(
        "subjects with overlays:",
        ", ".join(
            subject
            for subject, info in manifest["per_subject"].items()
            if info["overlay_fields"]
        ),
    )


if __name__ == "__main__":
    main()
