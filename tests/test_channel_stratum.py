import math

import pandas as pd
from analysis.channel_stratum import calculate_delta_by_channel

def test_channel_stratum():
    data = [
        {"subject_id": "model1", "channel": "native", "mode": "TOOL", "compliance": 1},
        {"subject_id": "model1", "channel": "native", "mode": "DISABLED", "compliance": 0},
        {"subject_id": "model2", "channel": "openrouter", "mode": "TOOL", "compliance": 1},
        {"subject_id": "model2", "channel": "openrouter", "mode": "DISABLED", "compliance": 1}
    ]
    df = pd.DataFrame(data)
    results = calculate_delta_by_channel(df)

    # model1 native: 1.0 - 0.0 = 1.0
    # model2 openrouter: 1.0 - 1.0 = 0.0

    assert results[results['subject_id'] == 'model1']['delta'].iloc[0] == 1.0
    assert results[results['subject_id'] == 'model2']['delta'].iloc[0] == 0.0


def test_channel_stratum_missing_mode_arm_is_not_zero_filled():
    df = pd.DataFrame(
        [
            {"subject_id": "model1", "channel": "native", "mode": "TOOL", "compliance": 1},
            {"subject_id": "model1", "channel": "native", "mode": "TOOL", "compliance": 0},
        ]
    )

    result = calculate_delta_by_channel(df).iloc[0]

    assert result["tool_rate"] == 0.5
    assert math.isnan(result["disabled_rate"])
    assert math.isnan(result["delta"])
