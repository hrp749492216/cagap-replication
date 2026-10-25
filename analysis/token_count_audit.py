import pandas as pd
import numpy as np


def _mode_rate(counts, mode: str) -> float:
    if mode not in counts.index:
        return float("nan")
    return float(counts.loc[mode])


def analyze_token_counts(df, mode_col='mode', token_col='input_tokens'):
    """
    Analyzes input token delta per mode.
    """
    stats = df.groupby(mode_col)[token_col].agg(['mean', 'std', 'count'])

    # Calculate delta relative to DISABLED
    if 'DISABLED' in stats.index:
        baseline = stats.loc['DISABLED', 'mean']
        stats['delta_tokens'] = stats['mean'] - baseline
    else:
        stats['delta_tokens'] = np.nan

    return stats

def stratify_delta_by_tokens(df, mode_col='mode', outcome_col='compliance', token_col='input_tokens', n_bins=5):
    """
    Calculates Delta stratified by input token count.
    """
    df = df.copy()
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    # Bin token counts
    df['token_bin'] = pd.qcut(df[token_col], q=n_bins, labels=False, duplicates='drop')
    if df['token_bin'].isna().all() and not df.empty:
        df['token_bin'] = 0

    results = []
    for bin_id, group in df.groupby('token_bin'):
        counts = group.groupby(mode_col)[outcome_col].mean()
        tool_rate = _mode_rate(counts, 'TOOL')
        disabled_rate = _mode_rate(counts, 'DISABLED')
        delta = tool_rate - disabled_rate

        results.append({
            "token_bin": bin_id,
            "token_range": f"{group[token_col].min():.0f}-{group[token_col].max():.0f}",
            "tool_rate": tool_rate,
            "disabled_rate": disabled_rate,
            "delta": delta,
            "n": len(group)
        })

    return pd.DataFrame(results)
