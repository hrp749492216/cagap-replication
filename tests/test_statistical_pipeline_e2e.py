from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.expansion.sensitivity import PAIRED_LOCAL_SEN_Q_SUBJECT_IDS
from tests.fixtures.statistical_pipeline_fixture import (
    build_full_statistical_pipeline_fixture,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_non_null_primary_result(result: dict, *, require_sane_band: bool = True) -> None:
    assert result["or"] is not None
    assert math.isfinite(float(result["or"]))
    if require_sane_band:
        assert 0.05 < float(result["or"]) < 20.0
    assert result["p_value"] is not None
    assert math.isfinite(float(result["p_value"]))
    assert result["p_holm"] is not None
    assert math.isfinite(float(result["p_holm"]))
    if result["engine"].startswith("conditional_logit"):
        assert result["se"] is not None
        assert math.isfinite(float(result["se"]))
        assert float(result["se"]) > 0


@pytest.mark.slow
@pytest.mark.integration
def test_full_statistical_pipeline_fixture_traces_prereg_section6_outputs(tmp_path):
    fixture = build_full_statistical_pipeline_fixture()
    assert fixture.baseline_row_count == 46_080
    assert fixture.sen_h_row_count > 0
    assert fixture.sen_q_quantization_row_count > 0

    input_path = tmp_path / "full_n3_fixture.jsonl"
    _write_jsonl(input_path, fixture.rows)
    output_dir = tmp_path / "stats"
    figure_dir = tmp_path / "figures"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_results.py",
            "--inputs",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--figure-dir",
            str(figure_dir),
            "--bootstrap-iter",
            "3",
            "--analysis-workers",
            os.environ.get("CAGAP_TEST_ANALYSIS_WORKERS", "4"),
            "--tipping-workers",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr

    primary = _read_json(output_dir / "primary_h1.json")
    assert set(primary["tests"]) == {
        "h1a_erosion",
        "h1a_full_asr",
        "h1b_erosion",
        "h1b_full_asr",
    }
    for endpoint, expected_counts in fixture.expected_primary.items():
        observed = primary["tests"][endpoint]
        _assert_non_null_primary_result(observed)
        for field, expected in expected_counts.items():
            assert observed[field] == expected

    sen_9_summary = _read_json(output_dir / "sensitivities" / "sen_9_summary.json")
    for row in sen_9_summary["table"]:
        assert row["or"] is not None, row
        assert row["p_value"] is not None, row

    sen_h = _read_json(output_dir / "sen_h_hardened_intervention.json")
    assert sen_h["status"] == "ok"
    for result_payload in sen_h["tests"].values():
        _assert_non_null_primary_result(result_payload, require_sane_band=False)

    sen_q = _read_json(output_dir / "sen_q_quantization_audit.json")
    assert sen_q["status"] == "ok"
    assert set(sen_q["subjects"]) == set(PAIRED_LOCAL_SEN_Q_SUBJECT_IDS)
    for subject in sen_q["subjects"].values():
        assert subject["n_pairs"] > 0
        assert subject["ordinal_label_agreement_rate"] is not None
        assert subject["delta_tool_call_rate_quantization_minus_primary"] is not None
    for endpoint_subjects in sen_q["endpoints"].values():
        for subject in endpoint_subjects.values():
            assert subject["delta_quantization_minus_primary"] is not None
            assert subject["matched_or_quantization_vs_primary"] is not None
            assert subject["p_value"] is not None

    sen_d = _read_json(output_dir / "sensitivities" / "sen_d_dispatch_block.json")
    assert sen_d["status"] == "ok"
    assert set(sen_d["aggregate_fits"]) == {"fixed_effect", "stratified"}
    assert sen_d["aggregate_fits"]["fixed_effect"]["status"] == "skipped_high_cardinality"
    for result_payload in sen_d["aggregate_fits"]["stratified"]["tests"].values():
        _assert_non_null_primary_result(result_payload, require_sane_band=False)
    assert sen_d["block_descriptives"]["n_blocks"] >= 2

    ledger_path = output_dir / "prereg_section6_contract_ledger.json"
    assert ledger_path.exists()
    ledger = _read_json(ledger_path)
    assert ledger["resolution_path"] == "prereg_section6_contract_ledger"
    assert ledger["summary"]["missing"] == 0
    assert ledger["summary"]["mismatch"] == 0
    assert ledger["summary"]["match"] > 0
    assert {
        "primary_h1",
        "sen_9",
        "sen_h",
        "sen_q",
        "sen_d",
        "domain_homogeneity",
    } <= {entry["artifact"] for entry in ledger["entries"]}
