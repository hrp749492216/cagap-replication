import pandas as pd


def _mode_rate(counts, mode: str) -> float:
    if mode not in counts.index:
        return float("nan")
    return float(counts.loc[mode])


def calculate_delta_by_channel(df, subject_id_col='subject_id', channel_col='channel', mode_col='mode', outcome_col='compliance'):
    """
    Calculates Delta split by channel (native vs openrouter) for each model.
    """
    results = []

    for (subject_id, channel), group in df.groupby([subject_id_col, channel_col]):
        # Calculate Delta for this (model, channel) pair
        counts = group.groupby(mode_col)[outcome_col].mean()

        tool_rate = _mode_rate(counts, 'TOOL')
        disabled_rate = _mode_rate(counts, 'DISABLED')
        delta = tool_rate - disabled_rate

        results.append({
            "subject_id": subject_id,
            "channel": channel,
            "tool_rate": tool_rate,
            "disabled_rate": disabled_rate,
            "delta": delta,
            "n": len(group)
        })

    return pd.DataFrame(results)
