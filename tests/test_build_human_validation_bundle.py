"""Regression guard for the human-validation bundle frame (addendum §4).

In the 2026-05-26 frozen manifest, 12 empty_response / analysis_inclusion=False rows
leaked into the 930-row human-validation set even though §4 excludes unresolved/error
rows from the audit. ``out_of_frame_custom_ids`` is the guard that makes any future
bundle build refuse to silently include such rows.
"""

from scripts.build_human_validation_bundle import out_of_frame_custom_ids


def _selected(*cids):
    return [{"custom_id": c} for c in cids]


def test_flags_empty_response_rows():
    selected = _selected("clean", "empty")
    final_rows = {
        "clean": {"analysis_inclusion": True, "final_label_source": "llm_ensemble"},
        "empty": {"analysis_inclusion": False, "final_label_source": "empty_response"},
    }
    assert out_of_frame_custom_ids(selected, final_rows) == ["empty"]


def test_flags_analysis_excluded_rows_regardless_of_source():
    # e.g. provider_error: analysis_inclusion=False but a different source string.
    selected = _selected("err")
    final_rows = {"err": {"analysis_inclusion": False, "final_label_source": "provider_error"}}
    assert out_of_frame_custom_ids(selected, final_rows) == ["err"]


def test_clean_resolved_rows_are_in_frame():
    selected = _selected("a", "b", "c")
    final_rows = {
        "a": {"analysis_inclusion": True, "final_label_source": "llm_ensemble"},
        "b": {"analysis_inclusion": True, "final_label_source": "predicate"},
        "c": {"analysis_inclusion": True, "final_label_source": "clear_refusal"},
    }
    assert out_of_frame_custom_ids(selected, final_rows) == []


def test_preserves_order_and_catches_multiple():
    selected = _selected("ok1", "bad1", "ok2", "bad2")
    final_rows = {
        "ok1": {"analysis_inclusion": True, "final_label_source": "predicate"},
        "bad1": {"analysis_inclusion": False, "final_label_source": "empty_response"},
        "ok2": {"analysis_inclusion": True, "final_label_source": "llm_ensemble"},
        "bad2": {"analysis_inclusion": False, "final_label_source": "empty_response"},
    }
    assert out_of_frame_custom_ids(selected, final_rows) == ["bad1", "bad2"]
