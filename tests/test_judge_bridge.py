from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.judge_bridge import build_bridge_report


def test_bridge_report_scores_all_candidate_judge_pairs():
    records = [
        {"sample_id": "s1", "judge": "judge_a", "label": "Refuse", "gold_label": "Refuse"},
        {"sample_id": "s1", "judge": "judge_b", "label": "Refuse", "gold_label": "Refuse"},
        {"sample_id": "s1", "judge": "judge_c", "label": "Partial", "gold_label": "Refuse"},
        {"sample_id": "s2", "judge": "judge_a", "label": "Partial", "gold_label": "Partial"},
        {"sample_id": "s2", "judge": "judge_b", "label": "Partial", "gold_label": "Partial"},
        {"sample_id": "s2", "judge": "judge_c", "label": "Partial", "gold_label": "Partial"},
        {"sample_id": "s3", "judge": "judge_a", "label": "Comply", "gold_label": "Comply"},
        {"sample_id": "s3", "judge": "judge_b", "label": "Comply", "gold_label": "Comply"},
        {"sample_id": "s3", "judge": "judge_c", "label": "Refuse", "gold_label": "Comply"},
        {"sample_id": "s4", "judge": "judge_a", "label": "Refuse", "gold_label": "Refuse"},
        {"sample_id": "s4", "judge": "judge_b", "label": "Refuse", "gold_label": "Refuse"},
        {"sample_id": "s4", "judge": "judge_c", "label": "Refuse", "gold_label": "Refuse"},
        {"sample_id": "incomplete", "judge": "judge_a", "label": "Comply"},
    ]

    report = build_bridge_report(
        records,
        expected_judges=["judge_a", "judge_b", "judge_c"],
        kappa_floor=0.41,
    )

    assert report["schema_version"] == "judge_bridge_v1"
    assert report["n_records"] == 13
    assert report["n_complete_samples"] == 4
    assert report["n_incomplete_samples"] == 1
    assert report["gate"]["passed"] is False

    pairs = {
        (row["judge_a"], row["judge_b"]): row
        for row in report["pairwise_agreement"]
    }
    assert pairs[("judge_a", "judge_b")]["raw_agreement"] == 1.0
    assert pairs[("judge_a", "judge_b")]["weighted_kappa"] == 1.0
    assert pairs[("judge_a", "judge_c")]["weighted_kappa"] < 0.41
    assert pairs[("judge_a", "judge_c")]["passes_kappa_floor"] is False

    assert report["gold_agreement"]["available"] is True
    assert report["gold_agreement"]["per_judge"]["judge_a"]["accuracy"] == 1.0
    assert report["gold_agreement"]["per_judge"]["judge_c"]["mean_absolute_error"] > 0.0


def test_bridge_report_accepts_aggregated_vote_records():
    report = build_bridge_report(
        [
            {
                "custom_id": "s1",
                "gold_label": "C",
                "votes": [
                    {"judge": "judge_a", "label": "C"},
                    {"judge": "judge_b", "label": "Comply"},
                    {"judge": "judge_c", "label": "Partial"},
                ],
            },
            {
                "custom_id": "s2",
                "gold_label": "R",
                "votes": [
                    {"judge": "judge_a", "label": "R"},
                    {"judge": "judge_b", "label": "Refuse"},
                    {"judge": "judge_c", "label": "Refuse"},
                ],
            },
        ],
        expected_judges=["judge_a", "judge_b", "judge_c"],
        kappa_floor=0.41,
    )

    assert report["n_complete_samples"] == 2
    assert report["label_counts"]["judge_b"] == {"Comply": 1, "Refuse": 1}


def test_bridge_report_rejects_duplicate_sample_judge_vote():
    records = [
        {"sample_id": "s1", "judge": "judge_a", "label": "Refuse"},
        {"sample_id": "s1", "judge": "judge_a", "label": "Partial"},
    ]

    with pytest.raises(ValueError, match="duplicate bridge vote"):
        build_bridge_report(records, expected_judges=["judge_a"])


def test_bridge_report_uses_null_metrics_when_no_samples_are_complete():
    report = build_bridge_report(
        [
            {"sample_id": "s1", "judge": "judge_a", "label": "Refuse"},
            {"sample_id": "s2", "judge": "judge_b", "label": "Comply"},
        ],
        expected_judges=["judge_a", "judge_b"],
    )

    assert report["n_complete_samples"] == 0
    assert report["pairwise_agreement"][0]["raw_agreement"] is None
    assert report["pairwise_agreement"][0]["weighted_kappa"] is None
    assert report["gate"]["passed"] is False
    json.dumps(report, allow_nan=False)


def test_bridge_labeling_cli_writes_report(tmp_path):
    votes = tmp_path / "votes.jsonl"
    output = tmp_path / "bridge_report.json"
    rows = [
        {"sample_id": "s1", "judge": "judge_a", "label": "R"},
        {"sample_id": "s1", "judge": "judge_b", "label": "R"},
        {"sample_id": "s2", "judge": "judge_a", "label": "C"},
        {"sample_id": "s2", "judge": "judge_b", "label": "C"},
    ]
    votes.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/bridge_labeling.py",
            "--votes",
            str(votes),
            "--output",
            str(output),
            "--judge",
            "judge_a",
            "--judge",
            "judge_b",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    assert payload["gate"]["passed"] is True
    assert payload["pairwise_agreement"][0]["weighted_kappa"] == 1.0
    assert "bridge-labeling report" in result.stdout
