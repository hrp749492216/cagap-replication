import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_raw_completeness_reports_missing_custom_ids(tmp_path):
    from analysis.raw_completeness import build_completeness_report

    batch_path = tmp_path / "batches" / "input_subject.jsonl"
    response_path = tmp_path / "raw" / "responses_subject.jsonl"
    _write_jsonl(
        batch_path,
        [
            {"custom_id": "row_a", "subject_id": "subject"},
            {"custom_id": "row_b", "subject_id": "subject"},
        ],
    )
    _write_jsonl(response_path, [{"custom_id": "row_a", "raw_response": {"choices": []}}])

    report = build_completeness_report([batch_path], [response_path])

    assert report["complete"] is False
    assert report["expected_count"] == 2
    assert report["observed_count"] == 1
    assert report["missing_custom_ids"] == ["row_b"]


def test_raw_completeness_passes_when_all_expected_ids_are_observed(tmp_path):
    from analysis.raw_completeness import assert_complete

    batch_path = tmp_path / "batches" / "input_subject.jsonl"
    response_path = tmp_path / "raw" / "responses_subject.jsonl"
    _write_jsonl(batch_path, [{"custom_id": "row_a", "subject_id": "subject"}])
    _write_jsonl(response_path, [{"custom_id": "row_a", "error": "provider_error"}])

    report = assert_complete([batch_path], [response_path])

    assert report["complete"] is True
    assert report["error_observed_count"] == 1


def test_raw_completeness_raises_on_missing_before_grading(tmp_path):
    from analysis.raw_completeness import CompletenessError, assert_complete

    batch_path = tmp_path / "batches" / "input_subject.jsonl"
    response_path = tmp_path / "raw" / "responses_subject.jsonl"
    _write_jsonl(batch_path, [{"custom_id": "row_a", "subject_id": "subject"}])
    _write_jsonl(response_path, [])

    with pytest.raises(CompletenessError, match="missing 1 expected raw row"):
        assert_complete([batch_path], [response_path])
