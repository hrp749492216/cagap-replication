from __future__ import annotations

import pandas as pd
import pytest

from analysis.predicate_override import predicate_override_audit


def test_predicate_override_audit_reports_arm_subject_and_endpoint_rates():
    rows = pd.DataFrame(
        [
            {
                "mode": "DISABLED",
                "subject_id": "model_a",
                "final_label": "Comply",
                "final_label_source": "predicate",
                "grading_audit": {"predicate_authority_reason": "complete_tool_call_dict"},
            },
            {
                "mode": "DISABLED",
                "subject_id": "model_a",
                "final_label": "Refuse",
                "final_label_source": "judge_ensemble",
                "grading_audit": {"route": "judge_ensemble"},
            },
            {
                "mode": "TOOL",
                "subject_id": "model_b",
                "final_label": "Comply",
                "final_label_source": "predicate",
                "grading_audit": {
                    "route": "predicate",
                    "structured_tool_call": True,
                    "predicate_authority_reason": "provider_structured_tool_call",
                },
            },
            {
                "mode": "TOOL",
                "subject_id": "model_b",
                "final_label": "Refuse",
                "final_label_source": "judge_ensemble",
                "grading_audit": {"structured_tool_call": False},
            },
        ]
    )

    audit = predicate_override_audit(
        rows,
        retained_h1={"tests": {"h1a_erosion": {"or": 2.0}}},
        removed_h1={"tests": {"h1a_erosion": {"or": 1.5}}},
    )

    assert audit["status"] == "ok"
    assert audit["disabled_predicate_override"]["rate"] == pytest.approx(0.5)
    assert audit["tool_provider_structured_call"]["rate"] == pytest.approx(0.5)
    assert audit["by_subject"]["model_a"]["disabled_predicate_override_rate"] == pytest.approx(0.5)
    assert audit["by_subject"]["model_b"]["tool_provider_structured_call_rate"] == pytest.approx(0.5)
    assert audit["by_endpoint"]["E1"]["predicate_authoritative_event_rate"] == pytest.approx(1.0)
    assert audit["h1_retained_vs_removed"]["retained"]["tests"]["h1a_erosion"]["or"] == 2.0
    assert audit["h1_retained_vs_removed"]["removed"]["tests"]["h1a_erosion"]["or"] == 1.5


def test_predicate_override_audit_handles_no_rows():
    audit = predicate_override_audit(pd.DataFrame())

    assert audit["status"] == "no_data"
    assert audit["disabled_predicate_override"]["denominator"] == 0
