from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


METHOD = "subject_mean_rademacher_wild_cluster_bootstrap"


def _not_estimable(reason: str, *, endpoint_col: str, bootstrap_iter: int, seed: int) -> dict[str, Any]:
    return {
        "status": "not_estimable",
        "reason": reason,
        "method": METHOD,
        "endpoint_col": endpoint_col,
        "bootstrap_iterations": int(max(0, bootstrap_iter)),
        "seed": int(seed),
        "n_subjects": 0,
        "n_clusters": 0,
        "subject_ids": [],
        "risk_difference": None,
        "subject_mean_risk_difference": None,
        "cluster_weighted_risk_difference": None,
        "ci_low": None,
        "ci_high": None,
        "p_value_two_sided": None,
        "null_in_ci": None,
    }


def _cluster_deltas(rows: pd.DataFrame, endpoint_col: str) -> pd.DataFrame:
    required = {"cluster_id", "subject_id", "mode", endpoint_col}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"subject-dependence rows missing required columns: {missing}")
    if rows.empty:
        return pd.DataFrame(columns=["cluster_id", "subject_id", "delta"])

    subject_counts = rows.groupby("cluster_id")["subject_id"].nunique(dropna=False)
    mixed_clusters = subject_counts[subject_counts > 1]
    if not mixed_clusters.empty:
        examples = [str(item) for item in mixed_clusters.head(5).index.tolist()]
        raise ValueError(f"cluster_id maps to multiple subject_id values: {examples}")

    complete_counts = (
        rows.dropna(subset=[endpoint_col])
        .groupby(["cluster_id", "mode"])
        .size()
        .unstack(fill_value=0)
    )
    for arm in ("TOOL", "DISABLED"):
        if arm not in complete_counts.columns:
            complete_counts[arm] = 0
    keep_clusters = set(
        complete_counts.index[
            (complete_counts["TOOL"] == 3) & (complete_counts["DISABLED"] == 3)
        ]
    )
    if not keep_clusters:
        return pd.DataFrame(columns=["cluster_id", "subject_id", "delta"])

    filtered = rows[rows["cluster_id"].isin(keep_clusters)].dropna(subset=[endpoint_col])
    arm_means = (
        filtered.groupby(["cluster_id", "subject_id", "mode"], dropna=False)[endpoint_col]
        .mean()
        .unstack("mode")
    )
    if not {"TOOL", "DISABLED"}.issubset(arm_means.columns):
        return pd.DataFrame(columns=["cluster_id", "subject_id", "delta"])
    arm_means = arm_means.dropna(subset=["TOOL", "DISABLED"]).reset_index()
    return pd.DataFrame(
        {
            "cluster_id": arm_means["cluster_id"].astype(str),
            "subject_id": arm_means["subject_id"].astype(str),
            "delta": arm_means["TOOL"].astype(float) - arm_means["DISABLED"].astype(float),
        }
    )


def subject_mean_rademacher_wild_cluster_bootstrap(
    rows: pd.DataFrame,
    endpoint_col: str,
    *,
    bootstrap_iter: int = 999,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Subject-clustered robustness for the paired risk-difference estimator.

    The primary H1 fit clusters its sandwich covariance by prompt. This
    sensitivity collapses complete prompt-subject-persona clusters to a paired
    risk difference, averages those deltas within each subject, and applies
    Rademacher sign weights at the subject level. It is intentionally reported
    as robustness rather than as the primary estimator because the subject panel
    is small.
    """

    bootstrap_iter = int(max(0, bootstrap_iter))
    seed = int(seed)
    if rows.empty:
        return _not_estimable(
            "no_rows",
            endpoint_col=endpoint_col,
            bootstrap_iter=bootstrap_iter,
            seed=seed,
        )

    deltas = _cluster_deltas(rows, endpoint_col)
    if deltas.empty:
        return _not_estimable(
            "no_complete_3x3_clusters",
            endpoint_col=endpoint_col,
            bootstrap_iter=bootstrap_iter,
            seed=seed,
        )

    subject_means = deltas.groupby("subject_id", sort=True)["delta"].mean()
    n_subjects = int(len(subject_means))
    if n_subjects < 2:
        result = _not_estimable(
            "fewer_than_two_subjects",
            endpoint_col=endpoint_col,
            bootstrap_iter=bootstrap_iter,
            seed=seed,
        )
        result.update(
            {
                "n_subjects": n_subjects,
                "n_clusters": int(len(deltas)),
                "subject_ids": [str(item) for item in subject_means.index.tolist()],
            }
        )
        return result

    values = subject_means.to_numpy(dtype=float)
    observed = float(values.mean())
    cluster_weighted = float(deltas["delta"].astype(float).mean())
    clusters_per_subject = (
        deltas.groupby("subject_id", sort=True)["cluster_id"].nunique().astype(int)
    )

    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    null_in_ci: bool | None = None
    if bootstrap_iter > 0:
        rng = np.random.default_rng(seed)
        weights = rng.choice(
            np.array([-1.0, 1.0], dtype=float),
            size=(bootstrap_iter, n_subjects),
        )
        centered = values - observed
        boot_estimates = observed + (weights @ centered) / float(n_subjects)
        null_stats = (weights @ values) / float(n_subjects)
        ci_low = float(np.quantile(boot_estimates, alpha / 2.0))
        ci_high = float(np.quantile(boot_estimates, 1.0 - (alpha / 2.0)))
        threshold = abs(observed) - 1e-12
        extreme = int((np.abs(null_stats) >= threshold).sum())
        p_value = float((extreme + 1) / (bootstrap_iter + 1))
        null_in_ci = bool(ci_low <= 0.0 <= ci_high)

    return {
        "status": "ok",
        "method": METHOD,
        "endpoint_col": endpoint_col,
        "bootstrap_iterations": bootstrap_iter,
        "seed": seed,
        "alpha": float(alpha),
        "n_subjects": n_subjects,
        "n_clusters": int(len(deltas)),
        "subject_ids": [str(item) for item in subject_means.index.tolist()],
        "clusters_per_subject_min": int(clusters_per_subject.min()),
        "clusters_per_subject_max": int(clusters_per_subject.max()),
        "risk_difference": observed,
        "subject_mean_risk_difference": observed,
        "cluster_weighted_risk_difference": cluster_weighted,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value_two_sided": p_value,
        "null_in_ci": null_in_ci,
    }


def _sign(value: Any, *, null_value: float = 0.0) -> int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric == null_value:
        return 0 if numeric == null_value else None
    return 1 if numeric > null_value else -1


def _primary_null_in_ci(primary_result: Mapping[str, Any]) -> bool | None:
    rd_low = primary_result.get("risk_difference_ci_low")
    rd_high = primary_result.get("risk_difference_ci_high")
    if rd_low is not None and rd_high is not None:
        return bool(float(rd_low) <= 0.0 <= float(rd_high))
    ci_low = primary_result.get("ci_low")
    ci_high = primary_result.get("ci_high")
    if ci_low is not None and ci_high is not None:
        return bool(float(ci_low) <= 1.0 <= float(ci_high))
    return None


def compare_subject_dependence_to_primary(
    primary_result: Mapping[str, Any],
    subject_result: Mapping[str, Any],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Flag material disagreements between primary and subject-robust outputs."""

    if subject_result.get("status") != "ok":
        return {
            "primary_engine": primary_result.get("engine"),
            "status": "not_comparable",
            "reason": subject_result.get("reason"),
            "dependence_sensitive": None,
        }

    primary_rd = primary_result.get("risk_difference")
    primary_direction = _sign(primary_rd)
    if primary_direction is None:
        primary_or = primary_result.get("or")
        primary_direction = _sign(primary_or, null_value=1.0)
    subject_direction = _sign(subject_result.get("risk_difference"))
    direction_disagreement = (
        primary_direction is not None
        and subject_direction is not None
        and primary_direction != 0
        and subject_direction != 0
        and primary_direction != subject_direction
    )

    primary_p = primary_result.get("p_holm", primary_result.get("p_value"))
    primary_reject = (
        bool(float(primary_p) <= alpha)
        if primary_p is not None and np.isfinite(float(primary_p))
        else None
    )
    subject_p = subject_result.get("p_value_two_sided")
    subject_reject = (
        bool(float(subject_p) <= alpha)
        if subject_p is not None and np.isfinite(float(subject_p))
        else None
    )
    reject_disagreement = (
        primary_reject is not None
        and subject_reject is not None
        and primary_reject != subject_reject
    )

    primary_null = _primary_null_in_ci(primary_result)
    subject_null = subject_result.get("null_in_ci")
    ci_null_disagreement = (
        primary_null is not None
        and subject_null is not None
        and bool(primary_null) != bool(subject_null)
    )
    flags = [direction_disagreement, reject_disagreement, ci_null_disagreement]

    return {
        "primary_engine": primary_result.get("engine"),
        "primary_p_holm": primary_result.get("p_holm"),
        "primary_holm_reject": primary_result.get("holm_reject", primary_reject),
        "primary_null_in_ci": primary_null,
        "subject_null_in_ci": subject_null,
        "direction_disagreement": bool(direction_disagreement),
        "holm_reject_disagreement": bool(reject_disagreement),
        "ci_null_disagreement": bool(ci_null_disagreement),
        "dependence_sensitive": bool(any(flags)),
    }
