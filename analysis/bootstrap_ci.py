import warnings
from typing import Callable, Any, List

import numpy as np
import pandas as pd

def cluster_bootstrap(df: pd.DataFrame,
                      cluster_col: str,
                      stat_fn: Callable[[pd.DataFrame], Any],
                      B: int = 2000,
                      seed: int = 42):
    """
    Performs cluster bootstrap.

    df: Data
    cluster_col: Column identifying clusters (e.g., case_id)
    stat_fn: Function that takes a DataFrame and returns the statistic(s) of interest
    B: Number of bootstrap iterations
    seed: RNG seed for reproducibility
    """
    if cluster_col not in df.columns:
        raise ValueError(f"cluster column {cluster_col!r} not found in dataframe")
    rng = np.random.default_rng(seed)
    df_pos = df.reset_index(drop=True)
    cluster_positions = [
        group.index.to_numpy()
        for _, group in df_pos.groupby(cluster_col, sort=False, dropna=False)
    ]
    n_clusters = len(cluster_positions)
    if n_clusters == 0:
        raise ValueError(
            f"cluster_bootstrap: no clusters found in column {cluster_col!r}; "
            f"cannot resample"
        )

    results = []
    failed_resamples = 0
    first_failure = None
    all_resampled_indices = rng.integers(0, n_clusters, size=(B, n_clusters))

    for b, resampled_indices in enumerate(all_resampled_indices):
        # Build the bootstrap sample from positional rows so repeated clusters
        # become repeated observations with fresh, unique sample labels.
        sample_positions = np.concatenate(
            [cluster_positions[i] for i in resampled_indices]
        )
        sample_df = df_pos.take(sample_positions)
        sample_df.index = pd.RangeIndex(len(sample_df))

        # Calculate statistic. A handful of resamples can legitimately make
        # `stat_fn` raise (e.g. an unstacked groupby that finds only one mode
        # in this draw) — record NaN for that resample rather than crashing
        # the whole bootstrap; `get_bootstrap_ci` already filters non-finite
        # entries before computing percentiles.
        try:
            res = stat_fn(sample_df)
        except Exception as exc:
            failed_resamples += 1
            if first_failure is None:
                first_failure = (b, type(exc).__name__, str(exc))
            res = float("nan")
        results.append(res)

    if failed_resamples > 0:
        first_b, exc_type, exc_message = first_failure
        warnings.warn(
            f"stat_fn failed for {failed_resamples}/{B} bootstrap resamples; "
            f"first failure on resample {first_b}: {exc_type}: {exc_message}",
            RuntimeWarning,
            stacklevel=2,
        )

    return results

def get_bootstrap_ci(bootstrap_results: List[float], confidence: float = 0.95):
    """Calculates percentile CI from bootstrap results.

    Drops NaN/inf entries before taking percentiles so a handful of failed
    resamples do not silently turn the entire CI into NaN.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    arr = np.asarray(bootstrap_results, dtype=float)
    finite = arr[np.isfinite(arr)]
    n_dropped = arr.size - finite.size
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped}/{arr.size} bootstrap resamples were non-finite and "
            f"excluded.",
            RuntimeWarning,
            stacklevel=2,
        )
    if finite.size == 0:
        return float("nan"), float("nan")

    lower_p = (1.0 - confidence) / 2.0
    upper_p = 1.0 - lower_p

    lower = float(np.quantile(finite, lower_p))
    upper = float(np.quantile(finite, upper_p))

    return lower, upper
