import math

import numpy as np
import pandas as pd
import pytest

from analysis.bootstrap_ci import cluster_bootstrap, get_bootstrap_ci


def test_bootstrap_basic():
    data = []
    for i in range(50):
        data.append({"case_id": i, "val": i})
    df = pd.DataFrame(data)

    def mean_fn(d):
        return d['val'].mean()

    results = cluster_bootstrap(df, 'case_id', mean_fn, B=100, seed=42)
    assert len(results) == 100

    lower, upper = get_bootstrap_ci(results)
    assert lower < 24.5 < upper


def test_cluster_bootstrap_empty_clusters_raises():
    df = pd.DataFrame({"case_id": [], "val": []})
    with pytest.raises(ValueError, match="no clusters found"):
        cluster_bootstrap(df, "case_id", lambda d: 0.0, B=10)


def test_cluster_bootstrap_missing_cluster_column_raises_value_error():
    df = pd.DataFrame({"case_id": [1, 2, 3], "val": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="cluster column 'prompt_id' not found"):
        cluster_bootstrap(df, "prompt_id", lambda d: 0.0, B=10)


def test_cluster_bootstrap_stat_fn_raise_yields_nan_resample():
    df = pd.DataFrame({"case_id": [1, 2, 3], "val": [0.0, 1.0, 2.0]})

    def crashing_fn(d):
        raise RuntimeError("intentional")

    with pytest.warns(
        RuntimeWarning,
        match="stat_fn failed for 10/10 bootstrap resamples",
    ):
        results = cluster_bootstrap(df, "case_id", crashing_fn, B=10, seed=0)
    assert len(results) == 10
    assert all(math.isnan(r) for r in results)


def test_cluster_bootstrap_resets_resampled_index_labels():
    df = pd.DataFrame(
        {"case_id": ["a", "a", "b", "b"], "val": [0.0, 1.0, 2.0, 3.0]},
        index=[10, 11, 20, 21],
    )

    def requires_unique_sample_index(sample: pd.DataFrame) -> float:
        assert sample.index.is_unique
        assert sample.index.tolist() == list(range(len(sample)))
        return float(len(sample))

    results = cluster_bootstrap(
        df,
        "case_id",
        requires_unique_sample_index,
        B=1,
        seed=0,
    )

    assert results == [4.0]


def test_get_bootstrap_ci_drops_nan_inf():
    results = [float("nan"), float("inf"), -float("inf"), 0.0, 1.0, 2.0, 3.0, 4.0]
    with pytest.warns(
        RuntimeWarning,
        match="3/8 bootstrap resamples were non-finite",
    ):
        lower, upper = get_bootstrap_ci(results, confidence=0.5)
    assert math.isfinite(lower) and math.isfinite(upper)
    assert lower < upper


def test_get_bootstrap_ci_all_nan_returns_nan_nan():
    results = [float("nan")] * 50
    with pytest.warns(
        RuntimeWarning,
        match="50/50 bootstrap resamples were non-finite",
    ):
        lower, upper = get_bootstrap_ci(results)
    assert math.isnan(lower) and math.isnan(upper)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.1])
def test_get_bootstrap_ci_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        get_bootstrap_ci([0.0, 1.0, 2.0], confidence=confidence)


def test_cluster_bootstrap_preserves_cluster_multiplicity():
    df = pd.DataFrame({"case_id": [1, 1, 2, 2, 3, 3], "val": [0, 1, 0, 1, 0, 1]})
    counts: list[int] = []

    def count_fn(d):
        counts.append(len(d))
        return float(len(d))

    cluster_bootstrap(df, "case_id", count_fn, B=20, seed=42)
    assert all(c == 6 for c in counts)


def test_cluster_bootstrap_prompt_resampling_preserves_within_prompt_rows():
    df = pd.DataFrame(
        [
            {"prompt_id": "p1", "row_id": "p1_a", "event": 1},
            {"prompt_id": "p1", "row_id": "p1_b", "event": 0},
            {"prompt_id": "p2", "row_id": "p2_a", "event": 1},
            {"prompt_id": "p2", "row_id": "p2_b", "event": 1},
            {"prompt_id": "p3", "row_id": "p3_a", "event": 0},
            {"prompt_id": "p3", "row_id": "p3_b", "event": 0},
        ]
    )
    expected_rows = {
        prompt_id: group["row_id"].tolist()
        for prompt_id, group in df.groupby("prompt_id", sort=False)
    }
    observed_samples: list[dict[str, list[str]]] = []

    def record_prompt_rows(sample: pd.DataFrame) -> float:
        observed_samples.append(
            {
                prompt_id: group["row_id"].tolist()
                for prompt_id, group in sample.groupby("prompt_id", sort=False)
            }
        )
        return float(sample["event"].mean())

    cluster_bootstrap(df, "prompt_id", record_prompt_rows, B=20, seed=7)

    saw_duplicate_prompt = False
    for observed in observed_samples:
        for prompt_id, row_ids in observed.items():
            prompt_rows = expected_rows[prompt_id]
            assert len(row_ids) % len(prompt_rows) == 0
            multiplicity = len(row_ids) // len(prompt_rows)
            saw_duplicate_prompt = saw_duplicate_prompt or multiplicity > 1
            assert row_ids == prompt_rows * multiplicity
    assert saw_duplicate_prompt


def test_cluster_bootstrap_reproducible_with_same_seed_and_input_frame():
    df = pd.DataFrame(
        {
            "prompt_id": [f"p{i}" for i in range(8) for _ in range(2)],
            "event": [0, 1, 1, 1, 0, 0, 1, 0] * 2,
        }
    )

    def mean_event(sample: pd.DataFrame) -> float:
        return float(sample["event"].mean())

    first = cluster_bootstrap(df, "prompt_id", mean_event, B=100, seed=20260510)
    second = cluster_bootstrap(df, "prompt_id", mean_event, B=100, seed=20260510)

    assert first == second


def test_cluster_bootstrap_zero_informative_stat_returns_nan_ci():
    df = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p2"],
            "mode": ["DISABLED", "TOOL", "DISABLED", "TOOL"],
            "event": [0, 0, 1, 1],
        }
    )

    def informative_delta(sample: pd.DataFrame) -> float:
        wide = sample.pivot_table(
            index="prompt_id",
            columns="mode",
            values="event",
            aggfunc="mean",
        )
        delta = wide["TOOL"] - wide["DISABLED"]
        informative = delta[delta != 0]
        if informative.empty:
            return float("nan")
        return float(informative.mean())

    results = cluster_bootstrap(df, "prompt_id", informative_delta, B=25, seed=12)
    with pytest.warns(
        RuntimeWarning,
        match="25/25 bootstrap resamples were non-finite",
    ):
        lower, upper = get_bootstrap_ci(results)

    assert all(math.isnan(value) for value in results)
    assert math.isnan(lower) and math.isnan(upper)


def test_cluster_bootstrap_single_prompt_ci_collapses_to_observed_stat():
    df = pd.DataFrame(
        {
            "prompt_id": ["only_prompt"] * 6,
            "mode": ["DISABLED"] * 3 + ["TOOL"] * 3,
            "event": [0, 0, 1, 1, 1, 1],
        }
    )

    def event_rate_difference(sample: pd.DataFrame) -> float:
        rates = sample.groupby("mode")["event"].mean()
        return float(rates["TOOL"] - rates["DISABLED"])

    results = cluster_bootstrap(df, "prompt_id", event_rate_difference, B=25, seed=42)
    lower, upper = get_bootstrap_ci(results)

    assert all(value == pytest.approx(2 / 3) for value in results)
    assert lower == pytest.approx(2 / 3)
    assert upper == pytest.approx(2 / 3)


def test_cluster_bootstrap_all_concordant_data_has_zero_width_ci():
    df = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "mode": ["DISABLED", "TOOL"] * 3,
            "event": [0, 0, 1, 1, 0, 0],
        }
    )

    def event_rate_difference(sample: pd.DataFrame) -> float:
        rates = sample.groupby("mode")["event"].mean()
        return float(rates["TOOL"] - rates["DISABLED"])

    results = cluster_bootstrap(df, "prompt_id", event_rate_difference, B=25, seed=42)
    lower, upper = get_bootstrap_ci(results)

    assert results == [0.0] * 25
    assert lower == 0.0
    assert upper == 0.0


@pytest.mark.slow
def test_percentile_ci_synthetic_coverage_reaches_nominal_floor():
    rng = np.random.default_rng(20260510)
    n_experiments = 5000
    n_clusters = 96
    bootstrap_draws = 199
    batch_size = 100
    disabled_prob = 0.25
    tool_prob = 0.37
    true_delta = tool_prob - disabled_prob
    covered = 0

    for start in range(0, n_experiments, batch_size):
        batch_n = min(batch_size, n_experiments - start)
        disabled = rng.binomial(1, disabled_prob, size=(batch_n, n_clusters))
        tool = rng.binomial(1, tool_prob, size=(batch_n, n_clusters))
        cluster_deltas = tool - disabled
        bootstrap_indices = rng.integers(
            0,
            n_clusters,
            size=(batch_n, bootstrap_draws, n_clusters),
        )
        bootstrap_distribution = np.take_along_axis(
            cluster_deltas[:, None, :],
            bootstrap_indices,
            axis=2,
        ).mean(axis=2)
        ci_low = np.percentile(bootstrap_distribution, 2.5, axis=1)
        ci_high = np.percentile(bootstrap_distribution, 97.5, axis=1)
        covered += int(((ci_low <= true_delta) & (true_delta <= ci_high)).sum())

    assert covered / n_experiments >= 0.93
