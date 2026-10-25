"""Sensitivity-analysis row filters for the expansion protocol.

The Sen-7 calibration label is inactive under the global n = 3 design
(preregistration §6.3, amendment 2026-04-30). The repeated-trial
information from the 96-prompt Sen-7 half-sample is absorbed into the
global baseline at trial_id in {0, 1, 2}; the 96-prompt subset is used as
the predeclared substrate for two pre-registered secondary sensitivities
only:

- ``sen_h_hardened_rows`` — Sen-H, hardened-intervention TOOL sub-arm.
- ``sen_q_quantization_audit_rows`` — Sen-Q, paired local/self-hosted BF16-vs-INT4
  quantization audit.

``sen7_calibration_rows`` is maintained as an API-stability shim that
selects main-partition baseline rows; the Sen-7 family of analyses does
not apply to the active design.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from grader.json_strict import strict_json_loads

from .design_constants import BASELINE_INTERVENTION, MODES, TRIAL_IDS

HARDENED_INTERVENTION = "hardened"
QUANTIZATION_AUDIT_ROLE = "quantization_audit"


def _load_paired_local_sen_q_subject_ids() -> tuple[str, ...]:
    roster_path = Path(__file__).resolve().parents[2] / "config" / "roster.json"
    roster = strict_json_loads(roster_path.read_text(encoding="utf-8"))
    return tuple(
        str(subject_id)
        for subject_id, entry in roster["subjects"].items()
        if entry.get("tier") == "local_self_hosted"
        and (entry.get("local_config") or {}).get("precision_sensitivity")
    )


PAIRED_LOCAL_SEN_Q_SUBJECT_IDS = _load_paired_local_sen_q_subject_ids()


def _trial_id(row: dict[str, Any]) -> int:
    value = row.get("trial_id", 0)
    if isinstance(value, bool):
        raise ValueError("trial_id must be an integer")
    try:
        trial_id = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("trial_id must be an integer") from exc
    if trial_id != value:
        raise ValueError("trial_id must be an integer")
    return trial_id


def sen7_calibration_rows(
    rows: Iterable[dict[str, Any]],
    *,
    calibration_prompt_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Return main-partition baseline rows.

    API-stability shim: under the global n = 3 design, this selector
    simply returns the main-partition baseline subset. The Sen-7
    calibration family of analyses is inactive for the active design
    (preregistration §6.3, §7); use ``sen_h_hardened_rows`` for
    hardened-intervention TOOL rows and ``sen_q_quantization_audit_rows``
    for the paired local/self-hosted BF16-vs-INT4 audit. ``calibration_prompt_ids`` is
    accepted for API stability and ignored.
    """
    selected: list[dict[str, Any]] = []
    for row in rows:
        partition_id = str(row.get("corpus_partition_id", row.get("partition_id", "main")))
        if partition_id != "main":
            continue
        if _trial_id(row) not in TRIAL_IDS:
            continue
        if row.get("intervention_condition", BASELINE_INTERVENTION) != BASELINE_INTERVENTION:
            continue
        if row.get("mode") not in MODES:
            continue
        selected.append(row)
    return selected


def sen_h_hardened_rows(
    rows: Iterable[dict[str, Any]],
    *,
    calibration_prompt_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Return Sen-H rows: hardened-intervention TOOL trials on the 96-prompt subset.

    Per preregistration §6.3, every TOOL cell in the 96-prompt
    calibration subset is dispatched a second time under the predeclared
    hardened system-prompt intervention (``intervention_condition =
    "hardened"``) at ``trial_id in {0, 1, 2}``. Sen-H rows are excluded
    from H1 by ``intervention_condition`` filter and analyzed as the
    H2-secondary system-prompt-hardening sensitivity.
    """
    calibration_set = {str(pid) for pid in calibration_prompt_ids or []}
    selected: list[dict[str, Any]] = []
    for row in rows:
        partition_id = str(row.get("corpus_partition_id", row.get("partition_id", "main")))
        if partition_id != "main":
            continue
        prompt_id = str(row.get("prompt_id") or row.get("case_id") or "")
        if prompt_id not in calibration_set:
            continue
        if _trial_id(row) not in TRIAL_IDS:
            continue
        if row.get("intervention_condition") != HARDENED_INTERVENTION:
            continue
        if row.get("mode") != "TOOL":
            continue
        selected.append(row)
    return selected


def sen_q_quantization_audit_rows(
    rows: Iterable[dict[str, Any]],
    *,
    calibration_prompt_ids: Iterable[str],
    sen_q_subject_ids: Iterable[str] = PAIRED_LOCAL_SEN_Q_SUBJECT_IDS,
) -> list[dict[str, Any]]:
    """Return Sen-Q rows: paired local/self-hosted BF16-vs-INT4 audit on the 96-prompt subset.

    Per preregistration §6.3, the paired local/self-hosted Sen-Q subjects
    rerun the same 4 modes x 2 personas x n = 3 trials structure on the
    96-prompt calibration subset under the predeclared 4-bit deployment
    variant. INT4 audit rows carry ``serving_precision_role =
    "quantization_audit"`` and are excluded from H1 by that filter.
    Sen-Q rows are returned regardless of precision role so a caller can
    contrast BF16 reference rows (``serving_precision_role = "primary"``,
    paired local/self-hosted subjects, calibration prompts only) against INT4 audit rows
    (``serving_precision_role = "quantization_audit"``).
    """
    calibration_set = {str(pid) for pid in calibration_prompt_ids or []}
    sen_q_set = {str(s) for s in sen_q_subject_ids}
    selected: list[dict[str, Any]] = []
    for row in rows:
        partition_id = str(row.get("corpus_partition_id", row.get("partition_id", "main")))
        if partition_id != "main":
            continue
        prompt_id = str(row.get("prompt_id") or row.get("case_id") or "")
        if prompt_id not in calibration_set:
            continue
        if _trial_id(row) not in TRIAL_IDS:
            continue
        if row.get("intervention_condition", BASELINE_INTERVENTION) != BASELINE_INTERVENTION:
            continue
        subject_id = str(row.get("subject_id", ""))
        if subject_id not in sen_q_set:
            continue
        if row.get("mode") not in MODES:
            continue
        # accept both BF16 reference (primary) and INT4 audit (quantization_audit)
        # so a caller can run the within-cell BF16 vs INT4 contrast.
        selected.append(row)
    return selected
