"""Subject-level family-excluded judge-panel resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
ROSTER_PATH = REPO_ROOT / "config" / "roster.json"

DEFAULT_PANEL = ("glm-5.1", "command-a-03-2025", "mistral-large-2512")
SUBSTITUTE_PRIORITY = ("gpt-5.5-judge", "claude-opus-4-7-judge")


def load_roster(path: Path = ROSTER_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _panel_config(roster: dict[str, Any]) -> tuple[list[str], list[str]]:
    config = roster.get("judge_family_exclusion")
    if not isinstance(config, dict):
        return list(DEFAULT_PANEL), list(SUBSTITUTE_PRIORITY)
    default_panel = config.get("default_panel")
    substitute_priority = config.get("substitute_priority")
    return (
        list(default_panel) if isinstance(default_panel, list) else list(DEFAULT_PANEL),
        list(substitute_priority)
        if isinstance(substitute_priority, list)
        else list(SUBSTITUTE_PRIORITY),
    )


def _family(section: dict[str, Any], item_id: str, *, section_name: str) -> str:
    entry = section.get(item_id)
    if not isinstance(entry, dict):
        raise ValueError(f"unknown {section_name} id: {item_id!r}")
    family = entry.get("family")
    if not isinstance(family, str) or not family:
        raise ValueError(f"{section_name} {item_id!r} has no family")
    return family


def resolve_judge_panel(roster: dict[str, Any], subject_id: str) -> dict[str, Any]:
    """Resolve the predeclared judge panel for one subject.

    Same-family exclusions are applied at the subject level. The function
    never reintroduces the subject family via a substitute judge.
    """
    subjects = roster.get("subjects")
    judges = roster.get("judges")
    if not isinstance(subjects, dict) or not isinstance(judges, dict):
        raise ValueError("roster must contain subjects and judges objects")

    subject_family = _family(subjects, subject_id, section_name="subject")
    default_panel, substitute_priority = _panel_config(roster)

    resolved: list[str] = []
    excluded: list[str] = []
    substitute_used: str | None = None
    for judge_id in default_panel:
        judge_family = _family(judges, judge_id, section_name="judge")
        if judge_family == subject_family:
            excluded.append(judge_id)
        else:
            resolved.append(judge_id)

    for _excluded_id in excluded:
        for substitute_id in substitute_priority:
            if substitute_id in resolved:
                continue
            substitute_family = _family(judges, substitute_id, section_name="judge")
            if substitute_family == subject_family:
                continue
            resolved.append(substitute_id)
            substitute_used = substitute_id
            break

    if len(resolved) != len(default_panel):
        raise ValueError(
            f"could not fill {len(default_panel)}-judge panel for subject_id={subject_id!r}"
        )

    judge_families = [
        _family(judges, judge_id, section_name="judge") for judge_id in resolved
    ]
    applied = bool(excluded)
    return {
        "judge_ids": resolved,
        "judge_panel_id": "panel__" + "__".join(resolved),
        "judge_families": judge_families,
        "default_panel": default_panel,
        "excluded_default_judges": excluded,
        "judge_family_exclusion_applied": applied,
        "judge_family_exclusion_reason": (
            f"subject_family={subject_family}" if applied else None
        ),
        "substitute_judge_used": substitute_used,
        "subject_family": subject_family,
    }


def panel_provenance_fields(
    roster: dict[str, Any],
    subject_id: str,
    judge_ids: list[str],
) -> dict[str, Any]:
    judges = roster.get("judges")
    if not isinstance(judges, dict):
        raise ValueError("roster must contain judges object")
    subject_family = _family(roster.get("subjects") or {}, subject_id, section_name="subject")
    default_panel, _substitutes = _panel_config(roster)
    judge_families = [
        _family(judges, judge_id, section_name="judge") for judge_id in judge_ids
    ]
    excluded = [
        judge_id
        for judge_id in default_panel
        if judge_id not in judge_ids
        and _family(judges, judge_id, section_name="judge") == subject_family
    ]
    substitute = next(
        (
            judge_id
            for judge_id in judge_ids
            if judge_id not in default_panel
        ),
        None,
    )
    out: dict[str, Any] = {
        "judge_panel_id": "panel__" + "__".join(judge_ids),
        "judge_family_exclusion_applied": bool(excluded),
        "judge_family_exclusion_reason": (
            f"subject_family={subject_family}" if excluded else None
        ),
        "substitute_judge_used": substitute,
    }
    for index, judge_id in enumerate(judge_ids[:3], start=1):
        out[f"judge_slot_{index}_model_id"] = judge_id
        out[f"judge_slot_{index}_family"] = judge_families[index - 1]
    return out
