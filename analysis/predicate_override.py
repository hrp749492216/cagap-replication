"""Predicate-override audit helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


LABEL_TO_ORDINAL = {"Refuse": 0, "Partial": 1, "Comply": 2, "R": 0, "P": 1, "C": 2}


def _audit_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_predicate_authoritative(row: pd.Series) -> bool:
    audit = _audit_dict(row.get("grading_audit"))
    source = str(row.get("final_label_source") or "")
    route = str(audit.get("route") or "")
    reason = str(audit.get("predicate_authority_reason") or "")
    return source == "predicate" or route == "predicate" or bool(reason)


def _is_disabled_extractor_override(row: pd.Series) -> bool:
    if str(row.get("mode")) != "DISABLED":
        return False
    audit = _audit_dict(row.get("grading_audit"))
    reason = str(audit.get("predicate_authority_reason") or "")
    return _is_predicate_authoritative(row) and (
        "complete_tool_call_dict" in reason
        or bool(audit.get("complete_tool_call_dict"))
        or reason == ""
    )


def _is_tool_provider_structured_call(row: pd.Series) -> bool:
    if str(row.get("mode")) != "TOOL":
        return False
    audit = _audit_dict(row.get("grading_audit"))
    reason = str(audit.get("predicate_authority_reason") or "")
    return bool(audit.get("structured_tool_call")) or "provider_structured" in reason


def _endpoint_event(label: Any, endpoint: str) -> int | None:
    if label is None:
        return None
    text = str(label).strip()
    if text not in LABEL_TO_ORDINAL:
        return None
    ordinal = LABEL_TO_ORDINAL[text]
    if endpoint == "E1":
        return int(ordinal >= 1)
    if endpoint == "E2":
        return int(ordinal == 2)
    raise ValueError(f"Unknown endpoint: {endpoint}")


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _arm_rate(rows: pd.DataFrame, mode: str, mask: pd.Series) -> dict[str, Any]:
    mode_mask = rows["mode"].eq(mode) if "mode" in rows.columns else pd.Series(False, index=rows.index)
    denominator = int(mode_mask.sum())
    numerator = int((mode_mask & mask).sum())
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": _rate(numerator, denominator),
    }


def _subject_rates(rows: pd.DataFrame, disabled_mask: pd.Series, tool_mask: pd.Series) -> dict[str, Any]:
    if "subject_id" not in rows.columns:
        return {}
    output: dict[str, Any] = {}
    for subject_id, group in rows.groupby("subject_id", sort=True):
        group_index = group.index
        output[str(subject_id)] = {
            "disabled_predicate_override_rate": _arm_rate(
                group,
                "DISABLED",
                disabled_mask.reindex(group_index, fill_value=False),
            )["rate"],
            "tool_provider_structured_call_rate": _arm_rate(
                group,
                "TOOL",
                tool_mask.reindex(group_index, fill_value=False),
            )["rate"],
        }
    return output


def _endpoint_rates(rows: pd.DataFrame, predicate_mask: pd.Series) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for endpoint in ("E1", "E2"):
        events = pd.Series(
            [_endpoint_event(label, endpoint) for label in rows.get("final_label", [])],
            index=rows.index,
            dtype="float",
        )
        resolved = events.notna()
        event_rows = resolved & events.eq(1)
        predicate_events = event_rows & predicate_mask
        output[endpoint] = {
            "event_rows": int(event_rows.sum()),
            "predicate_authoritative_event_rows": int(predicate_events.sum()),
            "predicate_authoritative_event_rate": _rate(
                int(predicate_events.sum()), int(event_rows.sum())
            ),
        }
    return output


def predicate_override_audit(
    rows: pd.DataFrame,
    *,
    retained_h1: Mapping[str, Any] | None = None,
    removed_h1: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if rows.empty:
        return {
            "status": "no_data",
            "disabled_predicate_override": {"numerator": 0, "denominator": 0, "rate": None},
            "tool_provider_structured_call": {"numerator": 0, "denominator": 0, "rate": None},
            "by_subject": {},
            "by_endpoint": {},
            "h1_retained_vs_removed": {
                "retained": retained_h1 or {"tests": {}},
                "removed": removed_h1 or {"tests": {}},
            },
        }
    work = rows.copy()
    if "mode" not in work.columns:
        work["mode"] = np.nan
    disabled_mask = pd.Series(
        [_is_disabled_extractor_override(row) for _, row in work.iterrows()],
        index=work.index,
    )
    tool_mask = pd.Series(
        [_is_tool_provider_structured_call(row) for _, row in work.iterrows()],
        index=work.index,
    )
    predicate_mask = pd.Series(
        [_is_predicate_authoritative(row) for _, row in work.iterrows()],
        index=work.index,
    )
    return {
        "status": "ok",
        "disabled_predicate_override": _arm_rate(work, "DISABLED", disabled_mask),
        "tool_provider_structured_call": _arm_rate(work, "TOOL", tool_mask),
        "by_subject": _subject_rates(work, disabled_mask, tool_mask),
        "by_endpoint": _endpoint_rates(work, predicate_mask),
        "h1_retained_vs_removed": {
            "retained": retained_h1 or {"tests": {}},
            "removed": removed_h1 or {"tests": {}},
        },
    }
