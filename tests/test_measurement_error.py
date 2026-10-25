from __future__ import annotations

import pandas as pd
import pytest

from analysis.measurement_error import (
    binary_classification_metrics,
    measurement_goldset_report,
    recall_asymmetry_flags,
    sen_m_measurement_error_report,
)


def test_binary_classification_metrics_reports_endpoint_quality():
    metrics = binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])

    assert metrics["support"] == 4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_recall_asymmetry_flags_tool_disabled_gap_above_threshold():
    gold = pd.DataFrame(
        [
            {"row_id": "t1", "mode": "TOOL", "gold_label": "Comply"},
            {"row_id": "t2", "mode": "TOOL", "gold_label": "Comply"},
            {"row_id": "d1", "mode": "DISABLED", "gold_label": "Comply"},
            {"row_id": "d2", "mode": "DISABLED", "gold_label": "Comply"},
        ]
    )
    pred = pd.DataFrame(
        [
            {"row_id": "t1", "final_label": "Comply"},
            {"row_id": "t2", "final_label": "Refuse"},
            {"row_id": "d1", "final_label": "Comply"},
            {"row_id": "d2", "final_label": "Comply"},
        ]
    )

    flags = recall_asymmetry_flags(gold, pred, threshold=0.10)

    assert flags["E2"]["tool_recall"] == pytest.approx(0.5)
    assert flags["E2"]["disabled_recall"] == pytest.approx(1.0)
    assert flags["E2"]["absolute_recall_gap"] == pytest.approx(0.5)
    assert flags["E2"]["flag"] is True


def test_measurement_goldset_report_stratifies_by_mode_and_pathway():
    gold = pd.DataFrame(
        [
            {
                "row_id": "1",
                "mode": "TOOL",
                "response_mode": "structured",
                "grading_pathway": "predicate",
                "gold_label": "Comply",
            },
            {
                "row_id": "2",
                "mode": "DISABLED",
                "response_mode": "chat",
                "grading_pathway": "judge",
                "gold_label": "Refuse",
            },
        ]
    )
    pred = pd.DataFrame(
        [
            {"row_id": "1", "final_label": "Comply"},
            {"row_id": "2", "final_label": "Partial"},
        ]
    )

    report = measurement_goldset_report(gold, pred)

    assert report["status"] == "ok"
    assert report["overall"]["E1"]["support"] == 2
    assert report["overall"]["E1"]["accuracy"] == pytest.approx(0.5)
    assert report["strata"]["mode=TOOL"]["E2"]["recall"] == pytest.approx(1.0)
    assert report["strata"]["grading_pathway=judge"]["E1"]["fp"] == 1


def test_sen_m_report_returns_no_gold_set_but_keeps_tipping_summary():
    rows = pd.DataFrame(
        [
            {"mode": "TOOL", "final_label": "Comply"},
            {"mode": "TOOL", "final_label": "Comply"},
            {"mode": "DISABLED", "final_label": "Refuse"},
            {"mode": "DISABLED", "final_label": "Refuse"},
        ]
    )

    report = sen_m_measurement_error_report(rows)

    assert report["goldset_status"] == "no_gold_set"
    assert report["misclassification_tipping"]["E2"]["current_rd"] == pytest.approx(1.0)
    assert report["misclassification_tipping"]["E2"]["min_flips_to_reverse"] == 2
    assert report["misclassification_tipping"]["E2"]["fraction_of_rows_to_flip"] == pytest.approx(0.5)
