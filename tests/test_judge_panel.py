from __future__ import annotations

import pytest

from scripts import judge_panel


ROSTER = {
    "subjects": {
        "glm-4.7-flash": {"family": "zai"},
        "mistral-medium-3.5-128b": {"family": "mistral"},
        "gpt-5.5": {"family": "openai"},
    },
    "judges": {
        "glm-5.1": {"family": "zai"},
        "command-a-03-2025": {"family": "cohere"},
        "mistral-large-2512": {"family": "mistral"},
        "gpt-5.5-judge": {"family": "openai"},
        "claude-opus-4-7-judge": {"family": "anthropic"},
    },
    "judge_family_exclusion": {
        "default_panel": ["glm-5.1", "command-a-03-2025", "mistral-large-2512"],
        "substitute_priority": ["gpt-5.5-judge", "claude-opus-4-7-judge"],
    },
}


def test_default_panel_is_used_when_subject_family_is_disjoint():
    panel = judge_panel.resolve_judge_panel(ROSTER, "gpt-5.5")

    assert panel["judge_ids"] == [
        "glm-5.1",
        "command-a-03-2025",
        "mistral-large-2512",
    ]
    assert panel["judge_family_exclusion_applied"] is False
    assert panel["substitute_judge_used"] is None


def test_glm_subject_excludes_glm_judge_and_uses_first_substitute():
    panel = judge_panel.resolve_judge_panel(ROSTER, "glm-4.7-flash")

    assert panel["judge_ids"] == [
        "command-a-03-2025",
        "mistral-large-2512",
        "gpt-5.5-judge",
    ]
    assert panel["judge_family_exclusion_applied"] is True
    assert panel["judge_family_exclusion_reason"] == "subject_family=zai"
    assert panel["substitute_judge_used"] == "gpt-5.5-judge"


def test_mistral_subject_excludes_mistral_judge_and_uses_first_substitute():
    panel = judge_panel.resolve_judge_panel(ROSTER, "mistral-medium-3.5-128b")

    assert panel["judge_ids"] == [
        "glm-5.1",
        "command-a-03-2025",
        "gpt-5.5-judge",
    ]
    assert panel["judge_family_exclusion_applied"] is True
    assert panel["judge_family_exclusion_reason"] == "subject_family=mistral"
    assert panel["substitute_judge_used"] == "gpt-5.5-judge"


def test_subject_family_cannot_be_reintroduced_by_substitute():
    roster = {
        **ROSTER,
        "subjects": {"gpt-5.5": {"family": "openai"}},
        "judge_family_exclusion": {
            "default_panel": ["gpt-5.5-judge"],
            "substitute_priority": ["gpt-5.5-judge"],
        },
    }

    with pytest.raises(ValueError, match="could not fill"):
        judge_panel.resolve_judge_panel(roster, "gpt-5.5")
