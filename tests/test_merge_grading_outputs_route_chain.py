"""Tests for the canonical judge vote-source route chain in merge_grading_outputs.

The chain encodes the predeclared route-preference table plus amendment
D6, which makes the self-hosted Command A lane canonical. Hosted Command A
routes remain available only when a replay explicitly selects them. Stale
canary slugs from earlier pilot runs are intentionally NOT enumerated; they
live in notes/noncanonical_artifacts_excluded_from_lock.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import merge_grading_outputs


def _make_subject_dir(tmp_path: Path, slugs: list[str]) -> Path:
    base = tmp_path / "judge_batches"
    subj_dir = base / "subject"
    for slug in slugs:
        d = subj_dir / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "votes.jsonl").write_text("{}\n", encoding="utf-8")
    return base


def test_glm_chain_picks_zai_direct_first(tmp_path):
    base = _make_subject_dir(
        tmp_path, ["glm_online", "glm_zai_openrouter"]
    )
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, judges=["glm-5.1"]
    )
    glm_paths = [s for s in sources if any("glm" in p for p in s.parts)]
    assert glm_paths == [base / "subject" / "glm_online" / "votes.jsonl"]


def test_glm_chain_falls_back_to_openrouter_pinned(tmp_path):
    base = _make_subject_dir(tmp_path, ["glm_zai_openrouter"])
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, judges=["glm-5.1"]
    )
    glm_paths = [s for s in sources if any("glm" in p for p in s.parts)]
    assert glm_paths == [base / "subject" / "glm_zai_openrouter" / "votes.jsonl"]


def test_glm_chain_excludes_legacy_glm_openrouter_slug(tmp_path):
    base = _make_subject_dir(tmp_path, ["glm_openrouter"])
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, judges=["glm-5.1"]
    )
    # The stale `glm_openrouter` slug must NOT appear; chain prefers the
    # explicit `glm_zai_openrouter` pinned-fallback slug instead.
    assert all("glm_openrouter" not in p for s in sources for p in s.parts)


def test_command_a_chain_picks_native_first(tmp_path):
    base = _make_subject_dir(
        tmp_path,
        [
            "command_a_online",
            "command-a-03-2025",
            "command_a_openrouter",
            "command_a_self_hosted",
        ],
    )
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, command_a_priority="native"
    )
    cmd_paths = [s for s in sources if "command" in str(s)]
    assert cmd_paths == [base / "subject" / "command_a_online" / "votes.jsonl"]


def test_command_a_chain_falls_through_tiers(tmp_path):
    # Skip T1 (both native slugs absent); should land on T2 OpenRouter pinned.
    base = _make_subject_dir(
        tmp_path, ["command_a_openrouter", "command_a_self_hosted"]
    )
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, command_a_priority="native"
    )
    cmd_paths = [s for s in sources if "command" in str(s)]
    assert cmd_paths == [
        base / "subject" / "command_a_openrouter" / "votes.jsonl"
    ]


def test_command_a_chain_can_prefer_openrouter_for_measurement_replay1(tmp_path):
    base = _make_subject_dir(
        tmp_path, ["command_a_online", "command_a_openrouter"]
    )
    sources = merge_grading_outputs._vote_sources(
        "subject",
        judge_batches_dir=base,
        command_a_priority="openrouter",
    )
    cmd_paths = [s for s in sources if "command" in str(s)]
    assert cmd_paths == [
        base / "subject" / "command_a_openrouter" / "votes.jsonl"
    ]


def test_command_a_openrouter_priority_requires_openrouter_file(tmp_path):
    base = _make_subject_dir(tmp_path, ["command_a_online"])

    with pytest.raises(FileNotFoundError, match="command_a_openrouter"):
        merge_grading_outputs._vote_sources(
            "subject",
            judge_batches_dir=base,
            command_a_priority="openrouter",
        )


def test_command_a_chain_uses_self_hosted_tier_3_when_only_option(tmp_path):
    base = _make_subject_dir(tmp_path, ["command_a_self_hosted"])
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, command_a_priority="native"
    )
    cmd_paths = [s for s in sources if "command" in str(s)]
    assert cmd_paths == [
        base / "subject" / "command_a_self_hosted" / "votes.jsonl"
    ]


@pytest.mark.parametrize(
    "stale_slug",
    [
        "cohere_online",
        "cohere_openrouter",
        "command_a_cohere_openrouter",
    ],
)
def test_command_a_chain_excludes_legacy_cohere_canary_slugs(tmp_path, stale_slug):
    base = _make_subject_dir(tmp_path, [stale_slug])
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, judges=["mistral-large-2512"]
    )
    assert all(stale_slug not in p for s in sources for p in s.parts)


def test_mistral_chain_picks_native_first(tmp_path):
    base = _make_subject_dir(
        tmp_path, ["mistral_openrouter", "mistral-large-2512"]
    )
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, judges=["mistral-large-2512"]
    )
    mistral_paths = [s for s in sources if "mistral" in str(s)]
    assert mistral_paths == [
        base / "subject" / "mistral-large-2512" / "votes.jsonl"
    ]


def test_mistral_chain_falls_back_to_openrouter_pinned(tmp_path):
    base = _make_subject_dir(tmp_path, ["mistral_openrouter"])
    sources = merge_grading_outputs._vote_sources(
        "subject", judge_batches_dir=base, judges=["mistral-large-2512"]
    )
    mistral_paths = [s for s in sources if "mistral" in str(s)]
    assert mistral_paths == [
        base / "subject" / "mistral_openrouter" / "votes.jsonl"
    ]


def test_vote_sources_follow_requested_family_excluded_panel(tmp_path):
    base = _make_subject_dir(
        tmp_path,
        ["glm_zai_openrouter", "command_a_self_hosted", "gpt_5_5_openrouter"],
    )

    sources = merge_grading_outputs._vote_sources(
        "subject",
        judge_batches_dir=base,
        judges=["glm-5.1", "command-a-03-2025", "gpt-5.5-judge"],
    )

    assert sources == [
        base / "subject" / "glm_zai_openrouter" / "votes.jsonl",
        base / "subject" / "command_a_self_hosted" / "votes.jsonl",
        base / "subject" / "gpt_5_5_openrouter" / "votes.jsonl",
    ]


def test_openrouter_vote_profile_uses_requested_judges(tmp_path):
    base = _make_subject_dir(
        tmp_path,
        ["glm_zai_openrouter", "command_a_openrouter", "gpt_5_5_openrouter"],
    )

    sources = merge_grading_outputs._vote_sources(
        "subject",
        judge_batches_dir=base,
        judges=["glm-5.1", "command-a-03-2025", "gpt-5.5-judge"],
        vote_route_profile="openrouter",
    )

    assert sources == [
        base / "subject" / "glm_zai_openrouter" / "votes.jsonl",
        base / "subject" / "command_a_openrouter" / "votes.jsonl",
        base / "subject" / "gpt_5_5_openrouter" / "votes.jsonl",
    ]


def test_singleton_judge_votes_are_collected_separately(tmp_path):
    base = tmp_path / "judge_batches"
    subj_dir = base / "subject"
    subj_dir.mkdir(parents=True)
    (subj_dir / "judge_votes.jsonl").write_text("{}\n", encoding="utf-8")
    for slug in ("glm_online", "command_a_self_hosted", "mistral_openrouter"):
        d = subj_dir / slug
        d.mkdir(parents=True)
        (d / "votes.jsonl").write_text("{}\n", encoding="utf-8")
    sources = merge_grading_outputs._vote_sources("subject", judge_batches_dir=base)
    assert subj_dir / "judge_votes.jsonl" in sources
    # Plus the configured judge-family source chains.
    assert len(sources) == 4


def test_grade_subject_passes_offline_only_by_default(tmp_path, monkeypatch):
    judge_batches = _make_subject_dir(
        tmp_path, ["glm_zai_openrouter", "command_a_self_hosted", "mistral_openrouter"]
    )
    subject_dir = judge_batches / "subject"
    input_path = tmp_path / "responses.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    (subject_dir / "input_subject.jsonl").write_text("{}\n", encoding="utf-8")
    (subject_dir / "manifest.json").write_text(
        '{"input_path": "%s"}' % input_path,
        encoding="utf-8",
    )
    (subject_dir / "extractions.jsonl").write_text("{}\n", encoding="utf-8")
    calls = []

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Completed()

    monkeypatch.setattr(merge_grading_outputs.subprocess, "run", fake_run)

    merge_grading_outputs.grade_subject(
        "subject",
        ["glm-5.1", "command-a-03-2025", "mistral-large-2512"],
        tmp_path / "out",
        judge_batches_dir=judge_batches,
        prompts_path=tmp_path / "prompts.jsonl",
        extractor_cache_dir=tmp_path / "cache",
    )

    assert "--offline-only" in calls[0]
    assert "--expected-batch" in calls[0]


def test_grade_subject_requires_expected_batch_by_default(tmp_path, monkeypatch):
    judge_batches = _make_subject_dir(
        tmp_path, ["glm_zai_openrouter", "command_a_self_hosted", "mistral_openrouter"]
    )
    subject_dir = judge_batches / "subject"
    input_path = tmp_path / "responses.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    (subject_dir / "manifest.json").write_text(
        '{"input_path": "%s"}' % input_path,
        encoding="utf-8",
    )
    (subject_dir / "extractions.jsonl").write_text("{}\n", encoding="utf-8")

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess must not run when expected batch is missing")

    monkeypatch.setattr(merge_grading_outputs.subprocess, "run", fail_run)

    with pytest.raises(FileNotFoundError, match="expected batch"):
        merge_grading_outputs.grade_subject(
            "subject",
            ["glm-5.1", "command-a-03-2025", "mistral-large-2512"],
            tmp_path / "out",
            judge_batches_dir=judge_batches,
            prompts_path=tmp_path / "prompts.jsonl",
            extractor_cache_dir=tmp_path / "cache",
        )


def test_openrouter_vote_profile_excludes_singleton_and_requires_or_files(tmp_path):
    base = tmp_path / "judge_batches"
    subj_dir = base / "subject"
    subj_dir.mkdir(parents=True)
    (subj_dir / "judge_votes.jsonl").write_text("{}\n", encoding="utf-8")
    for slug in ("glm_zai_openrouter", "command_a_openrouter", "mistral_openrouter"):
        d = subj_dir / slug
        d.mkdir(parents=True)
        (d / "votes.jsonl").write_text("{}\n", encoding="utf-8")

    sources = merge_grading_outputs._vote_sources(
        "subject",
        judge_batches_dir=base,
        vote_route_profile="openrouter",
    )

    assert subj_dir / "judge_votes.jsonl" not in sources
    assert sources == [
        subj_dir / "glm_zai_openrouter" / "votes.jsonl",
        subj_dir / "command_a_openrouter" / "votes.jsonl",
        subj_dir / "mistral_openrouter" / "votes.jsonl",
    ]


def test_grade_pilot_parse_requires_replay_paths():
    with pytest.raises(SystemExit):
        merge_grading_outputs.parse_args(
            [
                "--subjects",
                "subject",
                "--out-dir",
                "out",
            ]
        )
