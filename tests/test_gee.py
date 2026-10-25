import pandas as pd
from analysis.gee import (
    extract_gee_stats,
    extract_ordinal_gee_stats,
    run_ordinal_gee,
    run_paired_gee,
)

def test_gee_basic():
    # Create fake paired data
    data = []
    for i in range(100):
        # Cluster i
        # TOOL mode
        data.append({"case_id": i, "mode": "TOOL", "compliance": 1 if i < 80 else 0})
        # DISABLED mode
        data.append({"case_id": i, "mode": "DISABLED", "compliance": 1 if i < 40 else 0})

    df = pd.DataFrame(data)
    results = run_paired_gee(df)
    assert results is not None

    stats = extract_gee_stats(results)
    assert stats is not None
    assert stats['or'] > 1.0 # TOOL has higher compliance
    assert stats["se"] > 0.0
    assert stats["z"] > 0.0
    assert stats['p_value'] < 0.05


def test_ordinal_gee_basic():
    data = []
    for i in range(100):
        data.append(
            {
                "case_id": i,
                "mode": "TOOL",
                "outcome_ordinal": 2 if i < 65 else (1 if i < 90 else 0),
            }
        )
        data.append(
            {
                "case_id": i,
                "mode": "DISABLED",
                "outcome_ordinal": 2 if i < 20 else (1 if i < 50 else 0),
            }
        )

    results = run_ordinal_gee(pd.DataFrame(data), outcome_col="outcome_ordinal")
    assert results is not None

    stats = extract_ordinal_gee_stats(results)
    assert stats is not None
    assert stats["model"] == "ordinal_proportional_odds_gee"
    assert stats["or"] > 1.0
    assert stats["se"] > 0.0
    assert stats["z"] > 0.0
    assert stats["p_value"] < 0.05
