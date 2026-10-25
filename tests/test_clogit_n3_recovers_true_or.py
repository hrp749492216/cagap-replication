from __future__ import annotations

import math

import pandas as pd

from scripts._synthetic_n3_fixture import (
    build_concordant_n3_clusters,
    build_synthetic_n3_fixture,
)
from scripts.analyze_results import (
    _fit_primary_endpoint,
    _normalize_analysis_columns,
    _primary_rows,
    _retained_pair_prompt_clusters,
)


def _direct_non_extreme_total_event_count(rows: pd.DataFrame, endpoint_col: str) -> int:
    informative = 0
    for _, group in rows.groupby("cluster_id"):
        arm_counts = group.groupby("mode")[endpoint_col].agg(["sum", "size"])
        if not {"TOOL", "DISABLED"}.issubset(arm_counts.index):
            continue
        total_events = int(arm_counts.loc["TOOL", "sum"]) + int(
            arm_counts.loc["DISABLED", "sum"]
        )
        total_trials = int(arm_counts.loc["TOOL", "size"]) + int(
            arm_counts.loc["DISABLED", "size"]
        )
        if 0 < total_events < total_trials:
            informative += 1
    return informative


def _direct_discordant_pair_counts(rows: pd.DataFrame, endpoint_col: str) -> tuple[int, int]:
    wide = rows.pivot(index="pair_id", columns="mode", values=endpoint_col)
    wide = wide.dropna(subset=["DISABLED", "TOOL"])
    n10 = int(((wide["TOOL"] == 1) & (wide["DISABLED"] == 0)).sum())
    n01 = int(((wide["TOOL"] == 0) & (wide["DISABLED"] == 1)).sum())
    return n10, n01


def test_trial_level_clogit_n3_recovers_known_true_or():
    fixture = build_synthetic_n3_fixture(seed=20260510, prompt_count=48)
    normalized = _normalize_analysis_columns(fixture.frame)
    primary_rows = _primary_rows(normalized, persona_scope="pooled")

    result = _fit_primary_endpoint(primary_rows, "Z_full_asr", bootstrap_iter=10)

    assert result["engine"] == "conditional_logit_3x3_prompt_clustered"
    assert result["n_clusters"] == fixture.expected_primary_clusters
    assert result["informative_clusters"] == _direct_non_extreme_total_event_count(
        primary_rows, "Z_full_asr"
    )
    assert result["informative_clusters"] == fixture.expected_informative_clusters
    n10, n01 = _direct_discordant_pair_counts(primary_rows, "Z_full_asr")
    assert result["n_tool_success_disabled_failure"] == n10
    assert result["n_tool_failure_disabled_success"] == n01
    assert result["discordant_pairs"] == n10 + n01
    assert result["se"] is not None
    assert 0.01 < result["se"] < 0.50
    assert abs(result["beta"] - fixture.true_log_or) <= 1.96 * result["se"]
    assert result["or"] == math.exp(result["beta"])


def test_retained_pair_prompt_clusters_keeps_only_nonzero_variance_strata():
    model_df = pd.DataFrame(
        [
            {"pair_id": "pair_0", "prompt_id": "prompt_0", "Z_full_asr": 0},
            {"pair_id": "pair_0", "prompt_id": "prompt_0", "Z_full_asr": 0},
            {"pair_id": "pair_1", "prompt_id": "prompt_1", "Z_full_asr": 0},
            {"pair_id": "pair_1", "prompt_id": "prompt_1", "Z_full_asr": 1},
            {"pair_id": "pair_2", "prompt_id": "prompt_2", "Z_full_asr": 1},
            {"pair_id": "pair_2", "prompt_id": "prompt_2", "Z_full_asr": 1},
            {"pair_id": "pair_3", "prompt_id": "prompt_3", "Z_full_asr": 1},
            {"pair_id": "pair_3", "prompt_id": "prompt_3", "Z_full_asr": 0},
        ]
    )

    retained = _retained_pair_prompt_clusters(model_df, "Z_full_asr")

    assert retained == ["prompt_1", "prompt_3"]


def test_concordant_n3_clusters_do_not_contribute_to_conditional_or():
    fixture = build_synthetic_n3_fixture(seed=20260510, prompt_count=18)
    normalized = _normalize_analysis_columns(fixture.frame)
    base_rows = _primary_rows(normalized, persona_scope="pooled")
    base = _fit_primary_endpoint(base_rows, "Z_full_asr", bootstrap_iter=10)

    concordant = build_concordant_n3_clusters(start_prompt_index=10_000, cluster_count=12)
    combined = pd.concat([fixture.frame, concordant], ignore_index=True)
    combined_rows = _primary_rows(_normalize_analysis_columns(combined), persona_scope="pooled")
    augmented = _fit_primary_endpoint(combined_rows, "Z_full_asr", bootstrap_iter=10)

    assert augmented["n_clusters"] == base["n_clusters"] + 12
    assert augmented["informative_clusters"] == base["informative_clusters"]
    assert math.isclose(augmented["beta"], base["beta"], abs_tol=1e-4)
    assert math.isclose(augmented["or"], base["or"], rel_tol=1e-4)
