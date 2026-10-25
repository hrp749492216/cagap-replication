from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import analyze_capability_battery as analyze_battery


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    custom_id: str,
    *,
    subject_id: str,
    mode: str,
    domain: str = "bankledger",
    route: str = "openrouter",
    provider: str = "anthropic",
    grade: str = "no_action",
    mode_success: bool = False,
    **flags,
) -> dict:
    row = {
        "custom_id": custom_id,
        "subject_id": subject_id,
        "mode": mode,
        "domain": domain,
        "route": route,
        "provider": provider,
        "battery_grade": grade,
        "mode_success": mode_success,
        "TOOL_correct_call": False,
        "TOOL_false_refusal": False,
        "structured_intent_success": False,
        "wrong_tool": False,
        "wrong_args": False,
        "hallucinated_answer": False,
        "availability_calibrated": False,
    }
    row.update(flags)
    return row


def test_capability_analysis_builds_osf_secondary_tables(tmp_path):
    graded_rows = [
        _row(
            "a-tool-ok",
            subject_id="subject-a",
            mode="TOOL",
            grade="tool_correct_call",
            mode_success=True,
            TOOL_correct_call=True,
        ),
        _row(
            "a-disabled-intent",
            subject_id="subject-a",
            mode="DISABLED",
            grade="structured_intent",
            mode_success=True,
            structured_intent_success=True,
        ),
        _row(
            "a-notools-cal",
            subject_id="subject-a",
            mode="NOTOOLS",
            grade="availability_calibrated",
            mode_success=True,
            availability_calibrated=True,
        ),
        _row(
            "a-irr-wrong",
            subject_id="subject-a",
            mode="TOOL_IRRELEVANT",
            grade="wrong_tool",
            wrong_tool=True,
        ),
        _row(
            "b-tool-refusal",
            subject_id="subject-b",
            mode="TOOL",
            route="native_batch",
            provider="xai",
            grade="false_refusal",
            TOOL_false_refusal=True,
        ),
        _row(
            "b-disabled-hall",
            subject_id="subject-b",
            mode="DISABLED",
            route="native_batch",
            provider="xai",
            grade="hallucinated_answer",
            hallucinated_answer=True,
        ),
        _row(
            "b-notools-cal",
            subject_id="subject-b",
            mode="NOTOOLS",
            route="native_batch",
            provider="xai",
            grade="availability_calibrated",
            mode_success=True,
            availability_calibrated=True,
        ),
        _row(
            "b-irr-cal",
            subject_id="subject-b",
            mode="TOOL_IRRELEVANT",
            route="native_batch",
            provider="xai",
            grade="availability_calibrated",
            mode_success=True,
            availability_calibrated=True,
        ),
    ]
    expected_rows = [{"custom_id": row["custom_id"]} for row in graded_rows]
    graded_path = tmp_path / "graded.jsonl"
    expected_path = tmp_path / "expected.jsonl"
    out_dir = tmp_path / "analysis"
    _write_jsonl(graded_path, graded_rows)
    _write_jsonl(expected_path, expected_rows)

    summary = analyze_battery.analyze_capability_battery(
        graded_path=graded_path,
        expected_batch_paths=[expected_path],
        output_dir=out_dir,
    )

    assert summary["analysis_scope"] == "secondary_descriptive_not_h1_filter"
    assert summary["analysis_profile"] == "descriptive_1200"
    assert summary["osf_alignment"]["analysis_profile"] == "descriptive_1200"
    assert summary["osf_alignment"]["registration_status"] == "osf_declared_secondary_descriptive"
    assert summary["osf_alignment"]["planned_total_rows"] == 1200
    assert summary["completeness"]["custom_id_set_matches_expected"] is True
    assert summary["completeness"]["observed_rows"] == 8
    assert summary["tables"]["by_subject_mode"]["rows"] == 8

    by_subject = {
        row["subject_id"]: row
        for row in json.loads((out_dir / "by_subject.json").read_text(encoding="utf-8"))
    }
    assert by_subject["subject-a"]["mode_success_n"] == 3
    assert by_subject["subject-a"]["mode_success_denominator"] == 4
    assert by_subject["subject-a"]["mode_success_rate"] == pytest.approx(0.75)
    assert by_subject["subject-a"]["TOOL_correct_call_denominator"] == 1
    assert by_subject["subject-a"]["TOOL_correct_call_rate"] == pytest.approx(1.0)
    assert by_subject["subject-b"]["TOOL_false_refusal_rate"] == pytest.approx(1.0)

    by_mode = {
        row["mode"]: row
        for row in json.loads((out_dir / "by_mode.json").read_text(encoding="utf-8"))
    }
    assert by_mode["TOOL"]["TOOL_correct_call_n"] == 1
    assert by_mode["TOOL"]["TOOL_correct_call_denominator"] == 2
    assert by_mode["TOOL"]["TOOL_false_refusal_n"] == 1
    assert by_mode["TOOL"]["TOOL_false_refusal_denominator"] == 2
    assert by_mode["TOOL_IRRELEVANT"]["wrong_tool_rate"] == pytest.approx(0.5)

    assert (out_dir / "summary.json").exists()
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Benign Capability Battery Analysis" in report
    assert "secondary descriptive" in report
    assert "subject-a" in report
    assert "75.0%" in report
    assert (out_dir / "metric_table.csv").exists()
    assert (out_dir / "by_route_mode.csv").exists()
    assert (out_dir / "by_domain_mode.csv").exists()


def test_capability_analysis_reports_paired_hardening_utility(tmp_path):
    graded_rows = [
        _row(
            "s1-p1-base",
            subject_id="subject-a",
            mode="TOOL",
            grade="tool_correct_call",
            mode_success=True,
            TOOL_correct_call=True,
            prompt_id="p1",
            benign_id="p1",
            intervention_condition="baseline",
        ),
        _row(
            "s1-p1-hard",
            subject_id="subject-a",
            mode="TOOL",
            grade="tool_correct_call",
            mode_success=True,
            TOOL_correct_call=True,
            prompt_id="p1",
            benign_id="p1",
            intervention_condition="hardened",
        ),
        _row(
            "s1-p2-base",
            subject_id="subject-a",
            mode="TOOL",
            grade="tool_correct_call",
            mode_success=True,
            TOOL_correct_call=True,
            prompt_id="p2",
            benign_id="p2",
            intervention_condition="baseline",
        ),
        _row(
            "s1-p2-hard",
            subject_id="subject-a",
            mode="TOOL",
            grade="false_refusal",
            TOOL_false_refusal=True,
            prompt_id="p2",
            benign_id="p2",
            intervention_condition="hardened",
        ),
        _row(
            "s2-p1-base",
            subject_id="subject-b",
            mode="TOOL",
            grade="false_refusal",
            TOOL_false_refusal=True,
            prompt_id="p1",
            benign_id="p1",
            intervention_condition="baseline",
        ),
        _row(
            "s2-p1-hard",
            subject_id="subject-b",
            mode="TOOL",
            grade="tool_correct_call",
            mode_success=True,
            TOOL_correct_call=True,
            prompt_id="p1",
            benign_id="p1",
            intervention_condition="hardened",
        ),
    ]
    expected_rows = [{"custom_id": row["custom_id"]} for row in graded_rows]
    graded_path = tmp_path / "graded.jsonl"
    expected_path = tmp_path / "expected.jsonl"
    out_dir = tmp_path / "analysis"
    _write_jsonl(graded_path, graded_rows)
    _write_jsonl(expected_path, expected_rows)

    summary = analyze_battery.analyze_capability_battery(
        graded_path=graded_path,
        expected_batch_paths=[expected_path],
        output_dir=out_dir,
    )

    utility = summary["hardening_utility"]
    assert summary["analysis_profile"] == "benign_tool_hardening_utility"
    assert summary["osf_alignment"]["analysis_profile"] == "benign_tool_hardening_utility"
    assert summary["osf_alignment"]["registration_status"] == "post_registration_descriptive_extension"
    assert summary["osf_alignment"]["planned_total_rows"] == 600
    assert summary["osf_alignment"]["planned_modes"] == ["TOOL"]
    assert summary["osf_alignment"]["planned_intervention_conditions"] == ["baseline", "hardened"]
    assert utility["status"] == "ok"
    assert utility["scope"] == "paired_baseline_vs_hardened_tool_rows"
    assert utility["n_pairs"] == 3
    assert utility["baseline_correct_call_rate"] == pytest.approx(2 / 3)
    assert utility["hardened_correct_call_rate"] == pytest.approx(2 / 3)
    assert utility["risk_difference_hardened_minus_baseline"] == pytest.approx(0.0)
    assert utility["n_hardened_success_baseline_failure"] == 1
    assert utility["n_hardened_failure_baseline_success"] == 1
    assert utility["discordant_pairs"] == 2
    assert utility["risk_difference_ci95_method"] == "normal_approximation_over_complete_pair_deltas"
    assert utility["p_value_method"] == "two_sided_exact_binomial_over_discordant_pairs"
    assert utility["p_value_exact_binomial"] == pytest.approx(1.0)
    assert (out_dir / "hardening_utility.json").exists()
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Hardening Utility" in report
    assert "baseline TOOL correct-call rate" in report
    assert "Analysis profile: `benign_tool_hardening_utility`" in report


def test_capability_analysis_profile_defaults_route_to_hardening_paths():
    args = analyze_battery.parse_args(["--profile", "benign_tool_hardening_utility"])
    paths = analyze_battery._default_paths_for_profile(args.profile)

    assert paths["graded"] == analyze_battery.DEFAULT_HARDENING_UTILITY_GRADED
    assert paths["batches_dir"] == analyze_battery.DEFAULT_HARDENING_UTILITY_BATCHES_DIR
    assert paths["output_dir"] == analyze_battery.DEFAULT_HARDENING_UTILITY_OUTPUT_DIR


def test_capability_analysis_strict_completeness_rejects_missing_expected_rows(tmp_path):
    graded_path = tmp_path / "graded.jsonl"
    expected_path = tmp_path / "expected.jsonl"
    _write_jsonl(
        graded_path,
        [
            _row(
                "present",
                subject_id="subject-a",
                mode="TOOL",
                grade="tool_correct_call",
                mode_success=True,
                TOOL_correct_call=True,
            )
        ],
    )
    _write_jsonl(expected_path, [{"custom_id": "present"}, {"custom_id": "missing"}])

    with pytest.raises(ValueError, match="missing expected custom_id"):
        analyze_battery.analyze_capability_battery(
            graded_path=graded_path,
            expected_batch_paths=[expected_path],
            output_dir=tmp_path / "analysis",
            strict_completeness=True,
        )


def test_capability_analysis_strict_completeness_requires_expected_batches(tmp_path):
    graded_path = tmp_path / "graded.jsonl"
    _write_jsonl(
        graded_path,
        [
            _row(
                "present",
                subject_id="subject-a",
                mode="TOOL",
                grade="tool_correct_call",
                mode_success=True,
                TOOL_correct_call=True,
            )
        ],
    )

    with pytest.raises(ValueError, match="strict completeness requires expected batch"):
        analyze_battery.analyze_capability_battery(
            graded_path=graded_path,
            expected_batch_paths=[],
            output_dir=tmp_path / "analysis",
            strict_completeness=True,
        )


def test_capability_analysis_cli_writes_all_core_artifacts(tmp_path):
    graded_path = tmp_path / "graded.jsonl"
    expected_path = tmp_path / "expected.jsonl"
    out_dir = tmp_path / "analysis"
    rows = [
        _row(
            "cid-tool",
            subject_id="subject-a",
            mode="TOOL",
            grade="tool_correct_call",
            mode_success=True,
            TOOL_correct_call=True,
        )
    ]
    _write_jsonl(graded_path, rows)
    _write_jsonl(expected_path, [{"custom_id": "cid-tool"}])

    rc = analyze_battery.main(
        [
            "--graded",
            str(graded_path),
            "--expected-batch",
            str(expected_path),
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["completeness"]["observed_rows"] == 1
    assert summary["tables"]["overall"]["rows"] == 1
