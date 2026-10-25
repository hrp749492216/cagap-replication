from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from analysis.expansion.sensitivity import PAIRED_LOCAL_SEN_Q_SUBJECT_IDS
from scripts.corpus_design import build_design_cells
from scripts.run_synthetic_prompts import load_calibration_prompt_ids


BASELINE_MODES = ("NOTOOLS", "DISABLED", "TOOL", "TOOL_IRRELEVANT")
PRIMARY_MODES = {"DISABLED", "TOOL"}
PERSONAS = ("PERSONA_NEUTRAL", "PERSONA_OPERATOR")
TRIAL_IDS = (0, 1, 2)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StatisticalPipelineFixture:
    rows: list[dict[str, Any]]
    baseline_row_count: int
    sen_h_row_count: int
    sen_q_quantization_row_count: int
    expected_primary: dict[str, dict[str, int]]
    calibration_prompt_count: int


def _roster_subjects() -> tuple[tuple[str, str], ...]:
    roster = json.loads((REPO_ROOT / "config/roster.json").read_text(encoding="utf-8"))
    return tuple(
        (str(subject_id), str(entry["tier"]))
        for subject_id, entry in roster["subjects"].items()
    )


def _design_cells() -> list[Any]:
    case_cards = json.loads(
        (REPO_ROOT / "config/prompt_case_cards.json").read_text(encoding="utf-8")
    )
    return build_design_cells(case_cards)


def _paired_events(
    *,
    prompt_idx: int,
    subject_idx: int,
    persona_idx: int,
    trial_id: int,
    salt: int,
) -> tuple[int, int]:
    bucket = (
        prompt_idx * 37
        + subject_idx * 17
        + persona_idx * 11
        + trial_id * 7
        + salt
    ) % 100
    if bucket < 22:
        return 1, 1
    if bucket < 51:
        return 0, 1
    if bucket < 58:
        return 1, 0
    return 0, 0


def _nonprimary_label(
    *,
    prompt_idx: int,
    subject_idx: int,
    persona_idx: int,
    trial_id: int,
    mode: str,
) -> str:
    bucket = (
        prompt_idx * 19 + subject_idx * 13 + persona_idx * 5 + trial_id * 3
    ) % 100
    if mode == "NOTOOLS":
        if bucket < 4:
            return "Comply"
        if bucket < 18:
            return "Partial"
        return "Refuse"
    if bucket < 8:
        return "Comply"
    if bucket < 24:
        return "Partial"
    return "Refuse"


def _label_from_endpoint_events(*, erosion: int, full_asr: int) -> str:
    if full_asr:
        return "Comply"
    if erosion:
        return "Partial"
    return "Refuse"


def _primary_labels(
    *,
    prompt_idx: int,
    subject_idx: int,
    persona_idx: int,
    trial_id: int,
) -> dict[str, str]:
    disabled_erosion, tool_erosion = _paired_events(
        prompt_idx=prompt_idx,
        subject_idx=subject_idx,
        persona_idx=persona_idx,
        trial_id=trial_id,
        salt=0,
    )
    disabled_full, tool_full = _paired_events(
        prompt_idx=prompt_idx,
        subject_idx=subject_idx,
        persona_idx=persona_idx,
        trial_id=trial_id,
        salt=31,
    )
    disabled_erosion = max(disabled_erosion, disabled_full)
    tool_erosion = max(tool_erosion, tool_full)
    return {
        "DISABLED": _label_from_endpoint_events(
            erosion=disabled_erosion,
            full_asr=disabled_full,
        ),
        "TOOL": _label_from_endpoint_events(
            erosion=tool_erosion,
            full_asr=tool_full,
        ),
    }


def _event_for_label(label: str, endpoint: str) -> int:
    if endpoint == "erosion":
        return int(label in {"Partial", "Comply"})
    if endpoint == "full_asr":
        return int(label == "Comply")
    raise ValueError(endpoint)


def _base_row(
    *,
    prompt_id: str,
    case_id: str,
    domain: str,
    subject_id: str,
    subject_tier: str,
    persona: str,
    mode: str,
    trial_id: int,
    final_label: str,
    intervention_condition: str = "baseline",
    serving_precision_role: str = "primary",
    prompt_idx: int,
    subject_idx: int,
) -> dict[str, Any]:
    tool_call_count = int(mode == "TOOL" and final_label == "Comply")
    if serving_precision_role == "quantization_audit" and mode == "TOOL":
        tool_call_count = int(final_label in {"Partial", "Comply"})
    route = "local" if subject_tier == "local_self_hosted" else "native"
    dispatch_block = f"{subject_id}:{prompt_id}:t{trial_id}"
    if intervention_condition == "hardened":
        dispatch_block += ":hardened"
    return {
        "prompt_id": prompt_id,
        "case_id": case_id,
        "subject_id": subject_id,
        "subject_tier": subject_tier,
        "persona": persona,
        "domain": domain,
        "partition_id": "main",
        "trial_id": int(trial_id),
        "intervention_condition": intervention_condition,
        "serving_precision_role": serving_precision_role,
        "serving_precision": (
            "int4_audit" if serving_precision_role == "quantization_audit" else "primary"
        ),
        "mode": mode,
        "final_label": final_label,
        "final_label_status": "resolved",
        "analysis_inclusion": True,
        "final_label_source": "synthetic_statistical_pipeline_fixture",
        "input_tokens": 900 + (prompt_idx % 17),
        "output_tokens": 120 + (subject_idx % 11),
        "judge_a": final_label[0],
        "judge_b": final_label[0],
        "judge_c": final_label[0],
        "gold": final_label[0],
        "channel": route,
        "route": route,
        "dispatch_block_id": dispatch_block,
        "runtime_stamp": {
            "timestamp": f"2026-05-01T12:{(prompt_idx + subject_idx) % 60:02d}:{trial_id:02d}Z",
            "client_label": route,
            "returned_model": subject_id,
        },
        "grading_audit": {
            "route": "predicate" if tool_call_count else "judge_ensemble",
            "structured_tool_call": bool(tool_call_count),
            "complete_tool_call_dict": False,
            "provider_tool_call_count": tool_call_count,
        },
        "provider_tool_calls": [{"name": "synthetic_action"}] if tool_call_count else [],
        "extracted_tool_calls": [{"name": "synthetic_action"}] if tool_call_count else [],
    }


def _quantized_label(label: str, *, prompt_idx: int, subject_idx: int, trial_id: int) -> str:
    bucket = (prompt_idx * 23 + subject_idx * 7 + trial_id * 5) % 12
    if bucket == 0:
        return "Comply" if label == "Refuse" else label
    if bucket == 1:
        return "Partial" if label == "Comply" else label
    return label


def build_full_statistical_pipeline_fixture() -> StatisticalPipelineFixture:
    cells = _design_cells()
    if len(cells) != 192:
        raise AssertionError(f"expected 192 design cells, got {len(cells)}")
    subjects = _roster_subjects()
    if len(subjects) != 10:
        raise AssertionError(f"expected 10 roster subjects, got {len(subjects)}")
    calibration_ids = set(load_calibration_prompt_ids())
    if len(calibration_ids) != 96:
        raise AssertionError(f"expected 96 calibration prompts, got {len(calibration_ids)}")
    sen_q_subjects = set(PAIRED_LOCAL_SEN_Q_SUBJECT_IDS)
    local_subjects = {sid for sid, tier in subjects if tier == "local_self_hosted"}
    if not sen_q_subjects.issubset(local_subjects):
        raise AssertionError("paired local Sen-Q subjects are not all local roster subjects")

    rows: list[dict[str, Any]] = []
    for prompt_idx, cell in enumerate(cells):
        for subject_idx, (subject_id, subject_tier) in enumerate(subjects):
            for persona_idx, persona in enumerate(PERSONAS):
                for trial_id in TRIAL_IDS:
                    primary_labels = _primary_labels(
                        prompt_idx=prompt_idx,
                        subject_idx=subject_idx,
                        persona_idx=persona_idx,
                        trial_id=trial_id,
                    )
                    for mode in BASELINE_MODES:
                        label = (
                            primary_labels[mode]
                            if mode in PRIMARY_MODES
                            else _nonprimary_label(
                                prompt_idx=prompt_idx,
                                subject_idx=subject_idx,
                                persona_idx=persona_idx,
                                trial_id=trial_id,
                                mode=mode,
                            )
                        )
                        rows.append(
                            _base_row(
                                prompt_id=cell.prompt_id,
                                case_id=cell.case_id,
                                domain=cell.domain,
                                subject_id=subject_id,
                                subject_tier=subject_tier,
                                persona=persona,
                                mode=mode,
                                trial_id=trial_id,
                                final_label=label,
                                prompt_idx=prompt_idx,
                                subject_idx=subject_idx,
                            )
                        )
                    if cell.prompt_id in calibration_ids:
                        hardened_label = (
                            "Partial" if primary_labels["TOOL"] == "Comply" else "Refuse"
                        )
                        rows.append(
                            _base_row(
                                prompt_id=cell.prompt_id,
                                case_id=cell.case_id,
                                domain=cell.domain,
                                subject_id=subject_id,
                                subject_tier=subject_tier,
                                persona=persona,
                                mode="TOOL",
                                trial_id=trial_id,
                                final_label=hardened_label,
                                intervention_condition="hardened",
                                prompt_idx=prompt_idx,
                                subject_idx=subject_idx,
                            )
                        )
                    if cell.prompt_id in calibration_ids and subject_id in sen_q_subjects:
                        for mode in BASELINE_MODES:
                            baseline_label = (
                                primary_labels[mode]
                                if mode in PRIMARY_MODES
                                else _nonprimary_label(
                                    prompt_idx=prompt_idx,
                                    subject_idx=subject_idx,
                                    persona_idx=persona_idx,
                                    trial_id=trial_id,
                                    mode=mode,
                                )
                            )
                            rows.append(
                                _base_row(
                                    prompt_id=cell.prompt_id,
                                    case_id=cell.case_id,
                                    domain=cell.domain,
                                    subject_id=subject_id,
                                    subject_tier=subject_tier,
                                    persona=persona,
                                    mode=mode,
                                    trial_id=trial_id,
                                    final_label=_quantized_label(
                                        baseline_label,
                                        prompt_idx=prompt_idx,
                                        subject_idx=subject_idx,
                                        trial_id=trial_id,
                                    ),
                                    serving_precision_role="quantization_audit",
                                    prompt_idx=prompt_idx,
                                    subject_idx=subject_idx,
                                )
                            )

    return StatisticalPipelineFixture(
        rows=rows,
        baseline_row_count=sum(
            1
            for row in rows
            if row["intervention_condition"] == "baseline"
            and row["serving_precision_role"] == "primary"
        ),
        sen_h_row_count=sum(
            1 for row in rows if row["intervention_condition"] == "hardened"
        ),
        sen_q_quantization_row_count=sum(
            1 for row in rows if row["serving_precision_role"] == "quantization_audit"
        ),
        expected_primary=_expected_primary_counts(rows),
        calibration_prompt_count=len(calibration_ids),
    )


def _expected_primary_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    expected: dict[str, dict[str, int]] = {}
    for scope_name, personas in (
        ("h1a", set(PERSONAS)),
        ("h1b", {"PERSONA_NEUTRAL"}),
    ):
        scoped = [
            row
            for row in rows
            if row["intervention_condition"] == "baseline"
            and row["serving_precision_role"] == "primary"
            and row["mode"] in PRIMARY_MODES
            and row["persona"] in personas
        ]
        for endpoint in ("erosion", "full_asr"):
            n10 = n01 = n11 = n00 = 0
            by_pair: dict[tuple[str, str, str, int], dict[str, int]] = {}
            for row in scoped:
                key = (
                    row["prompt_id"],
                    row["subject_id"],
                    row["persona"],
                    int(row["trial_id"]),
                )
                by_pair.setdefault(key, {})[row["mode"]] = _event_for_label(
                    row["final_label"],
                    endpoint,
                )
            for pair in by_pair.values():
                tool = pair["TOOL"]
                disabled = pair["DISABLED"]
                n10 += int(tool == 1 and disabled == 0)
                n01 += int(tool == 0 and disabled == 1)
                n11 += int(tool == 1 and disabled == 1)
                n00 += int(tool == 0 and disabled == 0)
            expected[f"{scope_name}_{endpoint}"] = {
                "n_tool_success_disabled_failure": n10,
                "n_tool_failure_disabled_success": n01,
                "n_both_success": n11,
                "n_both_failure": n00,
                "discordant_pairs": n10 + n01,
            }
    return expected
