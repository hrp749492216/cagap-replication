"""Unit-level tests for ``scripts/analyze_results.py`` helpers.

These tests cover the pilot-partition wiring added 2026-04-30:
* ``_primary_rows`` must exclude pilot rows from H1 (defense-in-depth).
* ``summarize_pilot`` provides operator-facing dispatch-quality stats —
  counts only, NO inferential output.

Integration coverage of the full H1 pipeline lives in
``tests/test_analyze_results_integration.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_results  # noqa: E402
from analyze_results import _primary_rows, summarize_pilot  # noqa: E402
from analyze_results import build_h1_schema_visibility_gate_summary  # noqa: E402
from analysis.raw_completeness import CompletenessError  # noqa: E402


def _make_h1_row(
    *,
    prompt_id: str,
    subject_id: str,
    persona: str,
    mode: str,
    trial_id: int,
    partition_id: str,
    final_label: str = "Comply",
    disabled_arm_schema_visibility: str | None = "schema_visible",
    route_support_status: str | None = "supported",
) -> dict:
    """Build a minimum-viable H1-eligible row dict for unit tests."""
    row = {
        "prompt_id": prompt_id,
        "subject_id": subject_id,
        "persona": persona,
        "mode": mode,
        "trial_id": trial_id,
        "partition_id": partition_id,
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "final_label": final_label,
    }
    if disabled_arm_schema_visibility is not None:
        row["disabled_arm_schema_visibility"] = disabled_arm_schema_visibility
    if route_support_status is not None:
        row["route_support_status"] = route_support_status
    return row


def test_primary_rows_excludes_pilot_partition(caplog):
    """``_primary_rows`` filters out partition_id=='pilot' and warns."""
    rows = [
        _make_h1_row(
            prompt_id="P-main-001",
            subject_id="sonnet",
            persona="PERSONA_NEUTRAL",
            mode="DISABLED",
            trial_id=0,
            partition_id="main",
        ),
        _make_h1_row(
            prompt_id="P-main-001",
            subject_id="sonnet",
            persona="PERSONA_NEUTRAL",
            mode="TOOL",
            trial_id=0,
            partition_id="main",
        ),
        # Pilot rows: must be excluded from H1 output.
        _make_h1_row(
            prompt_id="PILOT-001",
            subject_id="sonnet",
            persona="PERSONA_NEUTRAL",
            mode="DISABLED",
            trial_id=0,
            partition_id="pilot",
        ),
        _make_h1_row(
            prompt_id="PILOT-001",
            subject_id="sonnet",
            persona="PERSONA_NEUTRAL",
            mode="TOOL",
            trial_id=0,
            partition_id="pilot",
        ),
    ]
    df = pd.DataFrame(rows)

    with caplog.at_level(logging.WARNING, logger="analyze_results"):
        out = _primary_rows(df, persona_scope="pooled")

    assert len(out) == 2
    assert set(out["partition_id"].unique()) == {"main"}
    # The defense-in-depth warning must mention pilot exclusion.
    assert any("pilot" in record.message.lower() for record in caplog.records), (
        "expected a WARNING-level log mentioning pilot exclusion"
    )


def test_primary_rows_no_warning_when_no_pilot_rows(caplog):
    """No pilot rows in input → no pilot-exclusion warning."""
    rows = [
        _make_h1_row(
            prompt_id="P-main-001",
            subject_id="sonnet",
            persona="PERSONA_NEUTRAL",
            mode="DISABLED",
            trial_id=0,
            partition_id="main",
        ),
        _make_h1_row(
            prompt_id="P-main-001",
            subject_id="sonnet",
            persona="PERSONA_NEUTRAL",
            mode="TOOL",
            trial_id=0,
            partition_id="main",
        ),
    ]
    df = pd.DataFrame(rows)

    with caplog.at_level(logging.WARNING, logger="analyze_results"):
        _primary_rows(df, persona_scope="pooled")

    assert not any(
        "pilot" in record.message.lower() for record in caplog.records
    )


def test_primary_rows_excludes_clusters_with_non_visible_disabled_schema(caplog):
    """Clean H1 keeps only clusters whose DISABLED rows preserve visible schema."""
    rows = []
    for trial_id in range(3):
        rows.extend(
            [
                _make_h1_row(
                    prompt_id="P-visible",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="schema_visible",
                ),
                _make_h1_row(
                    prompt_id="P-visible",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="schema_visible",
                ),
                _make_h1_row(
                    prompt_id="P-hidden",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="schema_hidden",
                ),
                _make_h1_row(
                    prompt_id="P-hidden",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="schema_hidden",
                ),
                _make_h1_row(
                    prompt_id="P-server-forced",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="server_forced",
                ),
                _make_h1_row(
                    prompt_id="P-server-forced",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="server_forced",
                ),
            ]
        )
    df = pd.DataFrame(rows)

    with caplog.at_level(logging.WARNING, logger="analyze_results"):
        out = _primary_rows(df, persona_scope="pooled")

    # server_forced is the second H1-eligible class (pre-registered in
    # Study_Design.md §2 + §12). Only schema_hidden clusters drop out.
    assert set(out["prompt_id"]) == {"P-visible", "P-server-forced"}
    assert len(out) == 12
    assert any(
        "schema-visible disabled eligibility" in record.message.lower()
        for record in caplog.records
    )


def test_primary_rows_excludes_clusters_with_unsupported_disabled_route():
    rows = []
    for trial_id in range(3):
        rows.extend(
            [
                _make_h1_row(
                    prompt_id="P-supported",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="schema_visible",
                    route_support_status="supported",
                ),
                _make_h1_row(
                    prompt_id="P-supported",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="schema_visible",
                    route_support_status="supported",
                ),
                _make_h1_row(
                    prompt_id="P-unsupported",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="schema_visible",
                    route_support_status="unsupported",
                ),
                _make_h1_row(
                    prompt_id="P-unsupported",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility="schema_visible",
                    route_support_status="unsupported",
                ),
            ]
        )

    out = _primary_rows(pd.DataFrame(rows), persona_scope="pooled")

    assert set(out["prompt_id"]) == {"P-supported"}
    assert len(out) == 6


def test_primary_rows_excludes_only_clusters_with_missing_schema_gate_fields(caplog):
    """Missing DISABLED gate metadata is H1-ineligible but not a run-stopper."""
    rows = []
    for trial_id in range(3):
        rows.extend(
            [
                _make_h1_row(
                    prompt_id="P-visible",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                ),
                _make_h1_row(
                    prompt_id="P-visible",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                ),
                _make_h1_row(
                    prompt_id="P-missing",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility=None,
                    route_support_status=None,
                ),
                _make_h1_row(
                    prompt_id="P-missing",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                ),
            ]
        )

    with caplog.at_level(logging.WARNING, logger="analyze_results"):
        out = _primary_rows(pd.DataFrame(rows), persona_scope="pooled")

    assert set(out["prompt_id"]) == {"P-visible"}
    assert len(out) == 6
    assert any(
        "schema-visible disabled eligibility" in record.message.lower()
        for record in caplog.records
    )


def test_h1_schema_visibility_gate_summary_separates_excluded_rows():
    rows = []
    for trial_id in range(3):
        rows.extend(
            [
                _make_h1_row(
                    prompt_id="P-visible",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                ),
                _make_h1_row(
                    prompt_id="P-visible",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                ),
                _make_h1_row(
                    prompt_id="P-missing",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="DISABLED",
                    trial_id=trial_id,
                    partition_id="main",
                    disabled_arm_schema_visibility=None,
                    route_support_status=None,
                ),
                _make_h1_row(
                    prompt_id="P-missing",
                    subject_id="sonnet",
                    persona="PERSONA_NEUTRAL",
                    mode="TOOL",
                    trial_id=trial_id,
                    partition_id="main",
                ),
            ]
        )

    summary = build_h1_schema_visibility_gate_summary(pd.DataFrame(rows))

    assert summary["excluded_rows"] == 6
    assert summary["excluded_clusters"] == 1
    assert summary["retained_clusters"] == 1
    assert summary["by_reason"] == {
        "missing_disabled_arm_schema_visibility": 3,
        "missing_route_support_status": 3,
    }
    assert summary["excluded_cluster_ids_sample"] == [
        "P-missing__sonnet__PERSONA_NEUTRAL"
    ]


def test_analyze_main_blocks_when_expected_batch_row_is_missing(tmp_path, monkeypatch):
    batch_path = tmp_path / "input_subject.jsonl"
    labeled_path = tmp_path / "labeled.jsonl"
    batch_path.write_text('{"custom_id": "row_a", "subject_id": "subject"}\n', encoding="utf-8")
    labeled_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_results.py",
            "--inputs",
            str(labeled_path),
            "--expected-batch",
            str(batch_path),
            "--output-dir",
            str(tmp_path / "stats"),
            "--figure-dir",
            str(tmp_path / "figures"),
        ],
    )

    with pytest.raises(CompletenessError, match="missing 1 expected raw row"):
        analyze_results.main()


def test_summarize_pilot_returns_dispatch_quality_stats():
    """Summary surfaces counts only, never inferential content."""
    rows = [
        # Confirmatory rows — must be ignored by summarize_pilot.
        _make_h1_row(
            prompt_id="P-main-001",
            subject_id="sonnet",
            persona="PERSONA_NEUTRAL",
            mode="DISABLED",
            trial_id=0,
            partition_id="main",
        ),
        # Pilot rows.
        {
            **_make_h1_row(
                prompt_id="PILOT-001",
                subject_id="sonnet",
                persona="PERSONA_NEUTRAL",
                mode="DISABLED",
                trial_id=0,
                partition_id="pilot",
            ),
            "error": None,
            "runtime_stamp": {"retry_count": 1},
        },
        {
            **_make_h1_row(
                prompt_id="PILOT-001",
                subject_id="sonnet",
                persona="PERSONA_NEUTRAL",
                mode="TOOL",
                trial_id=0,
                partition_id="pilot",
            ),
            "error": "rate_limit_exceeded",
            "runtime_stamp": {"retry_count": 3},
        },
        {
            **_make_h1_row(
                prompt_id="PILOT-002",
                subject_id="grok",
                persona="PERSONA_OPERATOR",
                mode="DISABLED",
                trial_id=0,
                partition_id="pilot",
            ),
            "error": None,
            "runtime_stamp": {"retry_count": 0},
        },
    ]
    df = pd.DataFrame(rows)

    summary = summarize_pilot(df)

    assert summary["n_rows"] == 3, "must count only partition_id=='pilot' rows"
    assert summary["by_subject"] == {"sonnet": 2, "grok": 1}
    assert "sonnet|DISABLED|PERSONA_NEUTRAL" in summary["by_cell"]
    assert "sonnet|TOOL|PERSONA_NEUTRAL" in summary["by_cell"]
    assert "grok|DISABLED|PERSONA_OPERATOR" in summary["by_cell"]
    assert summary["by_cell"]["sonnet|DISABLED|PERSONA_NEUTRAL"] == 1
    assert summary["error_count"] == 1
    assert summary["retry_event_count"] == 4  # 1 + 3 + 0
    assert summary["purpose"] == "harness_validation_only"
    assert summary["included_in_h1"] is False

    # Defensive: summary must NOT include any inferential keys.
    forbidden = {
        "p_value",
        "risk_difference",
        "odds_ratio",
        "ci_lower",
        "ci_upper",
        "h1_result",
    }
    assert forbidden.isdisjoint(summary.keys()), (
        "summarize_pilot must not return inferential statistics"
    )


def test_summarize_pilot_filters_corpus_partition_id_when_partition_id_absent():
    rows = [
        {
            **_make_h1_row(
                prompt_id="P-main-001",
                subject_id="sonnet",
                persona="PERSONA_NEUTRAL",
                mode="DISABLED",
                trial_id=0,
                partition_id="main",
            ),
            "corpus_partition_id": "main",
        },
        {
            **_make_h1_row(
                prompt_id="PILOT-001",
                subject_id="sonnet",
                persona="PERSONA_NEUTRAL",
                mode="DISABLED",
                trial_id=0,
                partition_id="pilot",
            ),
            "corpus_partition_id": "pilot",
        },
    ]
    df = pd.DataFrame(rows).drop(columns=["partition_id"])

    summary = summarize_pilot(df)

    assert summary["n_rows"] == 1
    assert summary["by_subject"] == {"sonnet": 1}


def test_summarize_pilot_handles_empty_dataframe():
    """An empty dataframe (or a df with no pilot rows) yields zero counts."""
    df = pd.DataFrame(
        [
            _make_h1_row(
                prompt_id="P-main-001",
                subject_id="sonnet",
                persona="PERSONA_NEUTRAL",
                mode="DISABLED",
                trial_id=0,
                partition_id="main",
            )
        ]
    )

    summary = summarize_pilot(df)

    assert summary["n_rows"] == 0
    assert summary["by_subject"] == {}
    assert summary["by_cell"] == {}
    assert summary["missed_cells"] == []
    assert summary["error_count"] == 0
    assert summary["retry_event_count"] == 0
