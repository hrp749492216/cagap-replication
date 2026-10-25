from __future__ import annotations

import pytest

from scripts import run_minimax_batch_extraction as minimax


def test_index_rows_by_custom_id_rejects_missing_custom_id():
    with pytest.raises(ValueError, match="missing non-empty custom_id"):
        minimax._index_rows_by_custom_id([{"subject_id": "subject"}])


def test_index_rows_by_custom_id_rejects_duplicate_custom_id():
    with pytest.raises(ValueError, match="duplicate custom_id"):
        minimax._index_rows_by_custom_id(
            [{"custom_id": "row_a"}, {"custom_id": "row_a"}]
        )


def test_extraction_record_with_provenance_stamps_current_contract_fields():
    record = minimax._with_extraction_provenance(
        {"custom_id": "row_a", "actions": []},
        subject_response="visible response",
    )

    assert record["subject_response_sha256"]
    assert record["extractor_prompt_sha256"]
    assert record["extractor_output_schema_version"] == "v3_raw_literal_types"
    assert record["rubric_version"] == "cagap_rubric_2026_05_09_revised"
    assert record["predicate_contract_version"] == "predicate_measurement_replay2_payload_priority"


def test_repair_completed_records_adds_missing_current_provenance():
    row = {
        "custom_id": "row_a",
        "raw_response": {
            "choices": [{"message": {"content": "visible response"}, "finish_reason": "stop"}]
        },
    }
    completed = {"row_a": {"custom_id": "row_a", "actions": []}}
    repaired = minimax._repair_completed_records(completed, rows_by_id={"row_a": row})

    assert repaired == 1
    assert "subject_response_sha256" in completed["row_a"]


def test_repair_completed_records_rejects_mismatched_existing_provenance():
    row = {
        "custom_id": "row_a",
        "raw_response": {
            "choices": [{"message": {"content": "visible response"}, "finish_reason": "stop"}]
        },
    }
    completed = {
        "row_a": {
            "custom_id": "row_a",
            "actions": [],
            "subject_response_sha256": "0" * 64,
        }
    }

    with pytest.raises(ValueError, match="subject_response_sha256 mismatch"):
        minimax._repair_completed_records(completed, rows_by_id={"row_a": row})
