import math

import pandas as pd
from analysis.token_count_audit import analyze_token_counts, stratify_delta_by_tokens

def test_token_count_audit():
    data = [
        {"mode": "TOOL", "input_tokens": 1000, "compliance": 1},
        {"mode": "DISABLED", "input_tokens": 800, "compliance": 0},
        {"mode": "TOOL", "input_tokens": 1200, "compliance": 1},
        {"mode": "DISABLED", "input_tokens": 1000, "compliance": 1}
    ]
    df = pd.DataFrame(data)

    stats = analyze_token_counts(df)
    assert stats.loc['TOOL', 'mean'] == 1100
    assert stats.loc['DISABLED', 'mean'] == 900
    assert stats.loc['TOOL', 'delta_tokens'] == 200

    stratified = stratify_delta_by_tokens(df, n_bins=2)
    assert len(stratified) == 2


def test_token_stratum_missing_mode_arm_is_not_zero_filled():
    df = pd.DataFrame(
        [
            {"mode": "DISABLED", "input_tokens": 10, "compliance": 0},
            {"mode": "DISABLED", "input_tokens": 20, "compliance": 1},
            {"mode": "TOOL", "input_tokens": 100, "compliance": 1},
            {"mode": "TOOL", "input_tokens": 110, "compliance": 1},
        ]
    )

    rows = stratify_delta_by_tokens(df, n_bins=2).sort_values("token_bin")

    first = rows.iloc[0]
    second = rows.iloc[1]
    assert math.isnan(first["tool_rate"])
    assert first["disabled_rate"] == 0.5
    assert math.isnan(first["delta"])
    assert second["tool_rate"] == 1.0
    assert math.isnan(second["disabled_rate"])
    assert math.isnan(second["delta"])


def test_token_stratum_identical_token_counts_still_emits_single_bin():
    df = pd.DataFrame(
        [
            {"mode": "DISABLED", "input_tokens": 100, "compliance": 0},
            {"mode": "TOOL", "input_tokens": 100, "compliance": 1},
        ]
    )

    rows = stratify_delta_by_tokens(df, n_bins=5)

    assert len(rows) == 1
    assert rows.iloc[0]["token_bin"] == 0
    assert rows.iloc[0]["token_range"] == "100-100"
    assert rows.iloc[0]["delta"] == 1.0
