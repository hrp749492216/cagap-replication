#!/usr/bin/env python3
"""Build the blinded 930-row human-validation bundle from the frozen manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.grade_subject_responses import (
    load_prompt_lookup,
    original_prompt_text,
    raw_response_from_row,
    resolve_case_data,
    subject_response_from_response,
)

DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "main_run" / "20260525T164059Z"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "paper"
    / "addenda"
    / "sampling"
    / "human_validation_sampling_manifest_2026-05-26.combined.json"
)
DEFAULT_OUT = DEFAULT_RUN_DIR / "human_validation"
DEFAULT_PROMPTS = REPO_ROOT / "experiments" / "aligned" / "prompts.jsonl"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(2 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(_json_dumps(row) + "\n")


def _write_sha256_sidecar(path: Path) -> None:
    digest = _sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _selected_manifest_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for universe_key, universe_label in (("ullm", "U-LLM"), ("udet", "U-DET")):
        block = manifest.get(universe_key)
        if not isinstance(block, dict):
            raise ValueError(f"manifest missing {universe_key!r} block")
        rows = block.get("selected_with_strata")
        if not isinstance(rows, list):
            raise ValueError(f"manifest {universe_key!r} missing selected_with_strata")
        for row in rows:
            if not isinstance(row, dict):
                continue
            cid = row.get("custom_id")
            tag = row.get("stratum_tag")
            if not isinstance(cid, str) or not isinstance(tag, str):
                raise ValueError(f"bad selected row in {universe_key}: {row!r}")
            if universe_label == "U-DET":
                if tag.endswith("::positive"):
                    universe = "U-DET-POS"
                elif tag.endswith("::negative"):
                    universe = "U-DET-NEG"
                else:
                    universe = "U-DET"
            else:
                universe = universe_label
            selected.append(
                {"custom_id": cid, "stratum_tag": tag, "audit_universe": universe}
            )
    return selected


def _load_labeled_rows(final_labels_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(final_labels_dir.glob("*/labeled_responses.jsonl")):
        for row in _iter_jsonl(path):
            cid = row.get("custom_id")
            if not isinstance(cid, str) or not cid:
                continue
            if cid in out:
                raise ValueError(f"duplicate custom_id in final labels: {cid}")
            out[cid] = row
    return out


def _presentation_sort_key(custom_id: str) -> str:
    return hashlib.sha256(
        f"cagap-human-validation-review-order-v1:{custom_id}".encode("utf-8")
    ).hexdigest()


def _visible_tool_calls(row: dict[str, Any]) -> list[Any]:
    calls = row.get("extracted_tool_calls")
    if isinstance(calls, list):
        return calls
    calls = row.get("provider_tool_calls")
    if isinstance(calls, list):
        return calls
    return []


def _blinded_row(
    *,
    sample_id: str,
    selected: dict[str, str],
    row: dict[str, Any],
    prompt_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw_response, _raw_source = raw_response_from_row(row)
    subject_response = row.get("subject_response")
    if not isinstance(subject_response, str):
        subject_response = subject_response_from_response(raw_response)
    case_data = resolve_case_data(row, prompt_lookup)
    return {
        "schema_version": "cagap_human_validation_blinded_v1",
        "sample_id": sample_id,
        "request_prompt": original_prompt_text(row, case_data),
        "assistant_text": subject_response,
        "tool_calls": _visible_tool_calls(row),
        "rubric_labels": ["Refuse", "Partial", "Comply"],
        "rater_fields": {
            "label": "Refuse|Partial|Comply",
            "evidence_span": "free_text_required",
            "evidence_pointer": "free_text_optional",
            "structured_action_evidence_present": "yes|no|unclear",
            "uncertain": "boolean",
        },
        "review_order_key": _presentation_sort_key(selected["custom_id"]),
    }


def _key_row(sample_id: str, selected: dict[str, str], row: dict[str, Any]) -> dict[str, Any]:
    audit = row.get("grading_audit") if isinstance(row.get("grading_audit"), dict) else {}
    return {
        "schema_version": "cagap_human_validation_key_v1",
        "sample_id": sample_id,
        "custom_id": selected["custom_id"],
        "audit_universe": selected["audit_universe"],
        "stratum_tag": selected["stratum_tag"],
        "prompt_id": row.get("prompt_id"),
        "subject_id": row.get("subject_id"),
        "mode": row.get("mode"),
        "persona": row.get("persona"),
        "trial_id": row.get("trial_id"),
        "intervention_condition": row.get("intervention_condition"),
        "serving_precision_role": row.get("serving_precision_role"),
        "domain": row.get("domain"),
        "dangerous_tool": row.get("dangerous_tool"),
        "final_machine_label": row.get("final_label"),
        "final_label_source": row.get("final_label_source"),
        "grading_route": audit.get("route"),
        "schema_visibility": row.get("disabled_arm_schema_visibility"),
        "route_support_status": row.get("route_support_status"),
        "review_order_key": _presentation_sort_key(selected["custom_id"]),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def out_of_frame_custom_ids(
    selected: list[dict[str, Any]], final_rows: dict[str, dict[str, Any]]
) -> list[str]:
    """custom_ids whose final label is out of the human-validation frame (addendum §4).

    The audit universe is *resolved* machine-graded rows. Unresolved / error rows — e.g.
    provider errors and empty subject responses (``final_label_source == 'empty_response'``,
    which carry ``analysis_inclusion == False``) — are NOT part of the audit unless a
    separately labeled missingness audit is added. Returns the offending custom_ids so a
    bundle build can refuse to silently include them.
    """
    flagged: list[str] = []
    for row in selected:
        final_row = final_rows.get(row["custom_id"], {})
        if (
            final_row.get("analysis_inclusion") is False
            or final_row.get("final_label_source") == "empty_response"
        ):
            flagged.append(row["custom_id"])
    return flagged


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = _selected_manifest_rows(manifest)
    final_rows = _load_labeled_rows(args.final_labels_dir)
    prompt_lookup = load_prompt_lookup(args.prompts)

    missing = [row["custom_id"] for row in selected if row["custom_id"] not in final_rows]
    if missing:
        raise ValueError(f"selected custom_id missing from final labels: {missing[:10]}")

    # Frame guard (addendum §4): refuse to bundle out-of-frame rows (empty_response /
    # analysis_inclusion=False) unless an explicit, clearly-labeled missingness audit is
    # requested. Prevents the empty-response leak that put 12 such rows in the 2026-05-26
    # frozen 930 from recurring on any future regeneration.
    out_of_frame = out_of_frame_custom_ids(selected, final_rows)
    if out_of_frame and not getattr(args, "allow_out_of_frame", False):
        raise ValueError(
            f"{len(out_of_frame)} selected row(s) are out of the human-validation frame "
            f"(empty_response / analysis_inclusion=False) and are excluded by addendum §4; "
            f"e.g. {out_of_frame[:5]}. Pass --allow-out-of-frame only for a clearly-labeled "
            f"missingness audit."
        )

    selected_sorted = sorted(selected, key=lambda row: _presentation_sort_key(row["custom_id"]))
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for idx, selected_row in enumerate(selected_sorted, start=1):
        sample_id = f"HV-{idx:04d}"
        final_row = final_rows[selected_row["custom_id"]]
        blinded_rows.append(
            _blinded_row(
                sample_id=sample_id,
                selected=selected_row,
                row=final_row,
                prompt_lookup=prompt_lookup,
            )
        )
        key_rows.append(_key_row(sample_id, selected_row, final_row))

    if len(blinded_rows) != 930:
        raise ValueError(f"expected 930 blinded rows, got {len(blinded_rows)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    blinded_path = args.out_dir / "human_validation_930_blinded.jsonl"
    key_path = args.out_dir / "human_validation_930_key.jsonl"
    app_json_path = args.out_dir / "human_validation_930_reviewer_app.json"
    key_csv_path = args.out_dir / "human_validation_930_key.csv"
    manifest_out = args.out_dir / "human_validation_930_bundle_manifest.json"

    _write_jsonl(blinded_path, blinded_rows)
    _write_jsonl(key_path, key_rows)
    app_json_path.write_text(
        _json_dumps(
            {
                "schema_version": "cagap_human_validation_reviewer_app_v1",
                "source_manifest": str(args.manifest.relative_to(REPO_ROOT)),
                "source_manifest_sha256": _sha256_file(args.manifest),
                "rows": blinded_rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        key_csv_path,
        key_rows,
        [
            "sample_id",
            "custom_id",
            "audit_universe",
            "stratum_tag",
            "prompt_id",
            "subject_id",
            "mode",
            "persona",
            "trial_id",
            "intervention_condition",
            "serving_precision_role",
            "domain",
            "dangerous_tool",
            "final_machine_label",
            "final_label_source",
            "grading_route",
            "schema_visibility",
            "route_support_status",
        ],
    )

    summary = {
        "schema_version": "cagap_human_validation_bundle_manifest_v1",
        "row_count": len(blinded_rows),
        "source_sampling_manifest": str(args.manifest.relative_to(REPO_ROOT)),
        "source_sampling_manifest_sha256": _sha256_file(args.manifest),
        "final_labels_dir": str(args.final_labels_dir.relative_to(REPO_ROOT)),
        "counts_by_universe": {
            key: sum(1 for row in key_rows if row["audit_universe"] == key)
            for key in ("U-LLM", "U-DET-POS", "U-DET-NEG")
        },
        "counts_by_machine_label": {},
        "artifacts": {
            "blinded_jsonl": str(blinded_path.relative_to(REPO_ROOT)),
            "key_jsonl": str(key_path.relative_to(REPO_ROOT)),
            "key_csv": str(key_csv_path.relative_to(REPO_ROOT)),
            "reviewer_app_json": str(app_json_path.relative_to(REPO_ROOT)),
        },
    }
    for row in key_rows:
        label = str(row.get("final_machine_label"))
        summary["counts_by_machine_label"][label] = (
            summary["counts_by_machine_label"].get(label, 0) + 1
        )
    manifest_out.write_text(_json_dumps(summary) + "\n", encoding="utf-8")
    for path in (blinded_path, key_path, app_json_path, key_csv_path, manifest_out):
        _write_sha256_sidecar(path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--final-labels-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "final_labels",
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--allow-out-of-frame",
        action="store_true",
        help="Permit empty_response / analysis_inclusion=False rows (addendum §4 out-of-frame). "
        "Use ONLY for a separately labeled missingness audit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_bundle(args)
    print(_json_dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
