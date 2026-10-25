from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.human_validation_reviewers import build_human_validation_reviewer_report


def _bundle() -> dict:
    rows = [{"sample_id": sample_id} for sample_id in ("s1", "s2", "s3")]
    return {
        "rows": rows,
        "assignments": [
            {"reviewer_id": "Reviewer 1", "sample_ids": ["s1", "s2", "s3"]},
            {"reviewer_id": "Reviewer 2", "sample_ids": ["s1", "s2"]},
            {"reviewer_id": "Reviewer 3", "sample_ids": ["s1", "s3"]},
            {"reviewer_id": "Reviewer 4", "sample_ids": ["s2"]},
            {"reviewer_id": "Reviewer 5", "sample_ids": []},
        ],
    }


def _labels() -> list[dict]:
    return [
        {"reviewer_id": "Reviewer 1", "sample_id": "s1", "label": "Refuse"},
        {"reviewer_id": "Reviewer 2", "sample_id": "s1", "label": "R"},
        {"reviewer_id": "Reviewer 3", "sample_id": "s1", "label": "Partial"},
        {"reviewer_id": "Reviewer 1", "sample_id": "s2", "label": "Comply"},
        {"reviewer_id": "Reviewer 2", "sample_id": "s2", "label": "C"},
        {"reviewer_id": "Reviewer 4", "sample_id": "s2", "label": "Comply"},
        {"reviewer_id": "Reviewer 1", "sample_id": "s3", "label": "Partial"},
        {"reviewer_id": "Reviewer 3", "sample_id": "s3", "label": "Comply"},
        {"reviewer_id": "Reviewer 6", "sample_id": "s1", "label": "Refuse"},
        {"reviewer_id": "Reviewer 1", "sample_id": "not-assigned", "label": "Refuse"},
        {"reviewer_id": "Reviewer 2", "sample_id": "s1", "label": "not-a-label"},
    ]


def test_human_validation_reviewer_report_handles_incomplete_overlap_design():
    report = build_human_validation_reviewer_report(
        _labels(),
        _bundle(),
        key_rows=[
            {"sample_id": "s1", "final_machine_label": "Refuse", "audit_universe": "U-LLM"},
            {"sample_id": "s2", "final_machine_label": "Comply", "audit_universe": "U-DET-POS"},
            {"sample_id": "s3", "final_machine_label": "Partial", "audit_universe": "U-DET-NEG"},
            {"sample_id": "outside-frame", "final_machine_label": "Comply", "audit_universe": "U-LLM"},
        ],
    )

    assert report["schema_version"] == "cagap_human_validation_reviewer_stats_v1"
    assert report["assignment_plan"]["selected_row_count"] == 3
    assert report["assignment_plan"]["sample_coding_distribution"] == {"2": 1, "3": 2}
    assert report["assignment_plan"]["triple_coded_row_count"] == 2
    assert report["assignment_plan"]["double_coded_row_count"] == 1
    assert report["label_ingest"]["valid_primary_label_count"] == 8
    assert report["label_ingest"]["invalid_label_count"] == 1
    assert report["label_ingest"]["unassigned_primary_label_count"] == 1
    assert report["label_ingest"]["excluded_non_primary_label_count"] == 1

    completion = report["completion"]["per_reviewer"]
    assert completion["Reviewer 1"]["assigned_count"] == 3
    assert completion["Reviewer 1"]["labeled_assigned_count"] == 3
    assert completion["Reviewer 5"]["assigned_count"] == 0
    assert completion["Reviewer 5"]["completion_rate"] is None

    agreement = report["agreement"]
    assert agreement["triple_coded"]["assigned_row_count"] == 2
    assert agreement["triple_coded"]["complete_row_count"] == 2
    assert agreement["triple_coded"]["unanimous_count"] == 1
    assert agreement["triple_coded"]["majority_2_1_count"] == 1
    assert agreement["double_coded"]["assigned_row_count"] == 1
    assert agreement["double_coded"]["complete_row_count"] == 1
    assert agreement["double_coded"]["agree_count"] == 0

    pairwise = {
        (row["reviewer_a"], row["reviewer_b"]): row
        for row in report["agreement"]["pairwise"]
    }
    assert pairwise[("Reviewer 1", "Reviewer 2")]["assigned_overlap_count"] == 2
    assert pairwise[("Reviewer 1", "Reviewer 2")]["both_labeled_count"] == 2
    assert pairwise[("Reviewer 1", "Reviewer 2")]["raw_agreement"] == 1.0
    assert pairwise[("Reviewer 1", "Reviewer 3")]["raw_agreement"] == 0.0

    assert report["machine_label_agreement"]["available"] is True
    assert report["machine_label_agreement"]["keyed_sample_count"] == 3
    assert report["machine_label_agreement"]["ignored_keyed_sample_count"] == 1
    assert report["machine_label_agreement"]["key_universe_counts"] == {
        "U-DET-NEG": 1,
        "U-DET-POS": 1,
        "U-LLM": 1,
    }
    assert report["machine_label_agreement"]["per_reviewer"]["Reviewer 1"]["accuracy"] == pytest.approx(1.0)
    assert report["machine_label_agreement"]["majority_vote"]["n"] == 2
    assert report["machine_label_agreement"]["majority_vote"]["tie_or_no_majority_count"] == 1
    assert report["machine_label_agreement"]["majority_vote"]["insufficient_human_label_count"] == 0
    assert report["machine_label_agreement"]["majority_vote"]["accuracy"] == pytest.approx(1.0)


def test_human_validation_reviewer_report_rejects_duplicate_primary_votes():
    labels = _labels() + [{"reviewer_id": "Reviewer 1", "sample_id": "s1", "label": "Comply"}]

    with pytest.raises(ValueError, match="duplicate human label"):
        build_human_validation_reviewer_report(labels, _bundle())


def test_human_validation_reviewer_report_uses_null_agreement_when_no_primary_overlap():
    report = build_human_validation_reviewer_report([], _bundle())

    assert report["completion"]["total_assigned_labels_observed"] == 0
    assert report["agreement"]["krippendorff_alpha_ordinal"] is None
    assert report["agreement"]["triple_coded"]["fleiss_kappa"] is None
    assert report["agreement"]["double_coded"]["raw_agreement"] is None


def test_human_validation_reviewer_report_matches_locked_5x200_bundle_contract():
    repo_root = Path(__file__).resolve().parents[1]
    bundle_path = repo_root / "human_validation_reviewer_app/data/human_validation_5x200_reviewer_app.json"
    key_path = repo_root / "outputs/main_run/20260525T164059Z/human_validation/human_validation_930_key.jsonl"
    if not bundle_path.exists() or not key_path.exists():
        pytest.skip("local reviewer bundle and locked key are not present in this checkout")
    bundle = json.loads(bundle_path.read_text())
    key_rows = [json.loads(line) for line in key_path.read_text().splitlines() if line.strip()]

    report = build_human_validation_reviewer_report([], bundle, key_rows=key_rows)

    plan = report["assignment_plan"]
    assert plan["selected_row_count"] == 350
    assert plan["assignment_edge_count"] == 1000
    assert plan["sample_coding_distribution"] == {"2": 50, "3": 300}
    assert plan["triple_coded_row_count"] == 300
    assert plan["double_coded_row_count"] == 50
    assert set(plan["expected_reviewers"]) == {
        "Reviewer 1",
        "Reviewer 2",
        "Reviewer 3",
        "Reviewer 4",
        "Reviewer 5",
    }
    assert set(plan["per_reviewer_assigned_count"].values()) == {200}
    assert set(plan["triad_counts"].values()) == {30}
    assert len(plan["triad_counts"]) == 10
    assert set(plan["pair_counts"].values()) == {5}
    assert len(plan["pair_counts"]) == 10
    assert report["machine_label_agreement"]["keyed_sample_count"] == 350
    assert report["machine_label_agreement"]["key_universe_counts"] == {
        "U-DET-NEG": 42,
        "U-DET-POS": 84,
        "U-LLM": 224,
    }
    assert report["machine_label_agreement"]["majority_vote"]["n"] == 0
    assert report["machine_label_agreement"]["majority_vote"]["insufficient_human_label_count"] == 350


def test_human_validation_reviewer_cli_writes_report(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    bundle_path = tmp_path / "bundle.json"
    key_path = tmp_path / "key.jsonl"
    output_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    labels_path.write_text("\n".join(json.dumps(row) for row in _labels()) + "\n")
    bundle_path.write_text(json.dumps(_bundle()))
    key_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"sample_id": "s1", "final_machine_label": "Refuse"},
                {"sample_id": "s2", "final_machine_label": "Comply"},
                {"sample_id": "s3", "final_machine_label": "Partial"},
            ]
        )
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_human_validation_reviewers.py",
            "--labels",
            str(labels_path),
            "--bundle",
            str(bundle_path),
            "--key",
            str(key_path),
            "--output",
            str(output_path),
            "--markdown-output",
            str(markdown_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text())
    assert report["assignment_plan"]["triple_coded_row_count"] == 2
    assert report["assignment_plan"]["double_coded_row_count"] == 1
    markdown = markdown_path.read_text()
    assert "Selected keyed rows: 3" in markdown
    assert "Majority-vote tied/no-majority rows: 1" in markdown
    assert "human-validation reviewer report" in result.stdout
