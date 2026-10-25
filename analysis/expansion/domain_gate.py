"""Domain heterogeneity gate helpers."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any, cast

from scipy import stats


SUBJECT_TIERS = ("hosted_api", "local_self_hosted")


def canonical_subject_tier(value: object) -> str:
    """Return the pre-registered two-stratum subject tier label."""
    return str(value)


def evaluate_domain_gate(
    odds_ratios: Mapping[str, float],
    *,
    p_value: float,
    alpha: float = 0.10,
) -> dict[str, object]:
    """Evaluate the pre-specified domain-homogeneity promotion rule.

    The upstream homogeneity test is the matched-discordance chi-square gate
    implemented in `scripts/analyze_results.build_domain_homogeneity` (NOT
    the classical Breslow-Day test of OR homogeneity — Breslow-Day is
    implemented separately in `analysis/breslow_day.py` and is not what
    drives the promotion rule). This helper consumes the stratum odds ratios
    and the homogeneity-test p-value so fixture tests can validate the
    decision rule without invoking a statistical backend.
    """
    values = [float(value) for value in odds_ratios.values()]
    if not values:
        raise ValueError("odds_ratios must contain at least one stratum")
    if any(value <= 0 or not isfinite(value) for value in values):
        raise ValueError("all odds ratios must be positive finite values")
    if not isfinite(p_value) or p_value < 0 or p_value > 1:
        raise ValueError("p_value must be finite and in [0, 1]")
    if not isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("alpha must be finite and in [0, 1]")

    rejected = p_value < alpha
    crosses_null = any(value < 1.0 for value in values) and any(value > 1.0 for value in values)
    same_side = all(value > 1.0 for value in values) or all(value < 1.0 for value in values)

    if not rejected:
        reason = "homogeneity_not_rejected"
        promote = False
    elif crosses_null:
        reason = "directional_heterogeneity"
        promote = True
    elif same_side:
        reason = "same_direction_heterogeneity"
        promote = True
    else:
        reason = "heterogeneity_without_directional_signal"
        promote = False

    return {
        "alpha": alpha,
        "p_value": p_value,
        "promote": promote,
        "directional_heterogeneity": rejected and crosses_null,
        "same_direction": rejected and same_side,
        "reason": reason,
    }


def _discordance_count(record: object, key: str) -> int:
    if isinstance(record, Mapping):
        value = record.get(key)
    else:
        value = getattr(record, key)
    if value is None:
        raise ValueError(f"discordance record missing {key}")
    raw_value = cast(Any, value)
    if isinstance(raw_value, bool):
        raise ValueError(f"{key} must be an integer count")
    count = int(raw_value)
    if count != raw_value:
        raise ValueError(f"{key} must be an integer count")
    if count < 0:
        raise ValueError(f"{key} must be non-negative")
    return count


def evaluate_subject_tier_gate(
    discordance_counts: Mapping[str, object],
    *,
    alpha: float = 0.10,
) -> dict[str, object]:
    """Evaluate matched-discordance heterogeneity across serving strata.

    ``discordance_counts`` maps each tier to counts named
    ``n_tool_success_disabled_failure`` and
    ``n_tool_failure_disabled_success``. The chi-square table is 2 x 2 for
    the pre-registered hosted/API versus local/self-hosted serving strata.
    """
    if not isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("alpha must be finite and in [0, 1]")

    aggregated: dict[str, dict[str, int]] = {}
    for raw_tier, record in discordance_counts.items():
        tier = canonical_subject_tier(raw_tier)
        if tier not in SUBJECT_TIERS:
            continue
        n10 = _discordance_count(record, "n_tool_success_disabled_failure")
        n01 = _discordance_count(record, "n_tool_failure_disabled_success")
        bucket = aggregated.setdefault(
            tier,
            {
                "n_tool_success_disabled_failure": 0,
                "n_tool_failure_disabled_success": 0,
            },
        )
        bucket["n_tool_success_disabled_failure"] += n10
        bucket["n_tool_failure_disabled_success"] += n01

    strata: list[dict[str, Any]] = []
    for tier in SUBJECT_TIERS:
        if tier not in aggregated:
            continue
        record = aggregated[tier]
        n10 = _discordance_count(record, "n_tool_success_disabled_failure")
        n01 = _discordance_count(record, "n_tool_failure_disabled_success")
        odds_ratio = (n10 + 0.5) / (n01 + 0.5)
        strata.append(
            {
                "subject_tier": tier,
                "discordance_table": [n10, n01],
                "n_tool_success_disabled_failure": n10,
                "n_tool_failure_disabled_success": n01,
                "discordant_pairs": n10 + n01,
                "matched_or_haldane_anscombe": float(odds_ratio),
            }
        )

    estimable = [s for s in strata if int(s["discordant_pairs"]) > 0]
    table = [
        [
            int(s["n_tool_success_disabled_failure"]),
            int(s["n_tool_failure_disabled_success"]),
        ]
        for s in estimable
    ]

    if (
        len(estimable) < len(SUBJECT_TIERS)
        or sum(row[0] for row in table) == 0
        or sum(row[1] for row in table) == 0
    ):
        return {
            "status": "not_estimable",
            "reason": "requires both pre-registered serving strata with at least one matched discordance and both global discordance directions",
            "valid_subject_tiers": len(strata),
            "estimable_subject_tiers": len(estimable),
            "subject_tiers": strata,
        }

    chi2, p_value, dfree, _expected = stats.chi2_contingency(table, correction=False)
    gate = evaluate_domain_gate(
        {str(s["subject_tier"]): float(s["matched_or_haldane_anscombe"]) for s in estimable},
        p_value=float(p_value),
        alpha=alpha,
    )
    return {
        "status": "ok",
        "test": "matched_discordance_subject_tier_homogeneity_gate",
        "alpha": alpha,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "df": int(dfree),
        "valid_subject_tiers": len(strata),
        "estimable_subject_tiers": len(estimable),
        "homogeneity_rejected": bool(p_value < alpha),
        "promote_subject_tier_headline": bool(gate["promote"]),
        "directional_heterogeneity": bool(gate["directional_heterogeneity"]),
        "same_direction": bool(gate["same_direction"]),
        "gate_reason": str(gate["reason"]),
        "subject_tiers": strata,
    }
