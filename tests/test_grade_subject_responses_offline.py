"""Tests for the --offline-only hard-failure gate in grade_subject_responses.

§13.D pre-lock requirement: the main-run final grading pass must NOT silently
dispatch live judges when caches were intended to be authoritative. The gate
runs before any extractor or judge work and reports the full inventory of
missing (custom_id, judge_id) pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import grade_subject_responses as grader


def _write_rows(path: Path, custom_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for cid in custom_ids:
            fh.write(json.dumps({"custom_id": cid, "subject_id": "subj"}) + "\n")


def _stub_judges(judge_ids: list[str]) -> dict[str, object]:
    # _enforce_offline_vote_coverage only inspects judges.keys(); the Judge
    # instances themselves are never called. Use a minimal sentinel object so
    # we can avoid the network-bound Judge constructor.
    return {jid: object() for jid in judge_ids}


def test_offline_only_passes_when_cache_covers_every_pair(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_rows(rows_path, ["c1", "c2"])
    judges = _stub_judges(["j1", "j2"])
    votes_cache = {
        "c1": {"j1": {"label": "Comply"}, "j2": {"label": "Refuse"}},
        "c2": {"j1": {"label": "Comply"}, "j2": {"label": "Comply"}},
    }
    grader._enforce_offline_vote_coverage([rows_path], judges, votes_cache)


def test_offline_only_lists_missing_pairs(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_rows(rows_path, ["c1", "c2", "c3", "c4"])
    judges = _stub_judges(["j1", "j2", "j3"])
    votes_cache = {
        "c1": {"j1": {"label": "Comply"}, "j2": {"label": "Refuse"}, "j3": {"label": "Comply"}},
        "c2": {"j1": {"label": "Comply"}, "j2": {"label": "Comply"}},  # missing j3
        "c3": {"j1": {"label": "Refuse"}, "j2": {"label": "Refuse"}, "j3": {"label": "Refuse"}},
        # c4 absent from cache entirely
    }
    with pytest.raises(RuntimeError) as exc_info:
        grader._enforce_offline_vote_coverage([rows_path], judges, votes_cache)
    msg = str(exc_info.value)
    # c2 missing j3 (1 pair) + c4 missing j1, j2, j3 (3 pairs) = 4 total
    assert "4 (custom_id, judge_id) pair(s)" in msg
    assert "(c2, j3)" in msg
    assert "(c4, j1)" in msg
    assert "(c4, j2)" in msg
    assert "(c4, j3)" in msg


def test_offline_only_requires_votes_cache(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_rows(rows_path, ["c1"])
    judges = _stub_judges(["j1"])
    with pytest.raises(RuntimeError, match="--offline-only requires --judge-votes-source"):
        grader._enforce_offline_vote_coverage([rows_path], judges, votes_cache=None)


def test_offline_only_requires_at_least_one_judge(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    _write_rows(rows_path, ["c1"])
    with pytest.raises(RuntimeError, match="no --judge configured"):
        grader._enforce_offline_vote_coverage(
            [rows_path], judges={}, votes_cache={"c1": {}}
        )


def test_offline_only_truncates_first_20_pairs(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    cids = [f"row_{i:03d}" for i in range(50)]
    _write_rows(rows_path, cids)
    judges = _stub_judges(["j1"])
    votes_cache: dict = {}
    with pytest.raises(RuntimeError) as exc_info:
        grader._enforce_offline_vote_coverage([rows_path], judges, votes_cache)
    msg = str(exc_info.value)
    assert "missing cached judge votes for 50" in msg
    # Only the first 20 pairs are listed by name to keep the message bounded.
    assert "(row_000, j1)" in msg
    assert "(row_019, j1)" in msg
    assert "(row_020, j1)" not in msg


def test_cli_offline_only_flag_is_registered():
    args = grader.parse_args(["--input", "x.jsonl", "--offline-only"])
    assert args.offline_only is True


def test_cli_offline_only_default_false():
    args = grader.parse_args(["--input", "x.jsonl"])
    assert args.offline_only is False


def test_grade_file_offline_only_fails_before_extractor_or_judge_construction(tmp_path, monkeypatch):
    rows_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "out.jsonl"
    _write_rows(rows_path, ["c1"])

    def fail_extractor(*args, **kwargs):
        raise AssertionError("Extractor constructor must not run before offline gate")

    def fail_judges(*args, **kwargs):
        raise AssertionError("Judge constructors must not run before offline gate")

    monkeypatch.setattr(grader, "Extractor", fail_extractor)
    monkeypatch.setattr(grader, "build_judges", fail_judges)

    with pytest.raises(RuntimeError, match="--offline-only requires --precomputed-extractions"):
        grader.grade_file(
            [rows_path],
            output_path,
            prompt_lookup={},
            judge_ids=["j1"],
            votes_cache={},
            offline_only=True,
        )


def test_load_judge_votes_cache_rejects_duplicate_custom_id_judge_pairs(tmp_path):
    vote_a = tmp_path / "votes_a.jsonl"
    vote_b = tmp_path / "votes_b.jsonl"
    vote_a.write_text(
        json.dumps({"custom_id": "c1", "judge": "judge_a", "label": "Comply"}) + "\n",
        encoding="utf-8",
    )
    vote_b.write_text(
        json.dumps({"custom_id": "c1", "judge": "judge_a", "label": "Refuse"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate cached judge vote"):
        grader.load_judge_votes_cache([vote_a, vote_b])


def test_load_judge_votes_cache_hydrates_self_hosted_model_votes_from_judge_inputs(tmp_path):
    subject_dir = tmp_path / "subject"
    vote_dir = subject_dir / "glm_5_1_self_hosted"
    judge_inputs_dir = subject_dir / "judge_inputs"
    vote_dir.mkdir(parents=True)
    judge_inputs_dir.mkdir(parents=True)
    votes = vote_dir / "votes.jsonl"
    judge_inputs = judge_inputs_dir / "glm-5.1.jsonl"
    prompt = "judge prompt bytes"
    cid = "c1"
    votes.write_text(
        json.dumps({"custom_id": cid, "model": "glm-5.1", "label": "Comply"}) + "\n",
        encoding="utf-8",
    )
    judge_inputs.write_text(
        json.dumps({"custom_id": cid, "judge_prompt": prompt}) + "\n",
        encoding="utf-8",
    )

    cache = grader.load_judge_votes_cache([votes])

    vote = cache[cid]["glm-5.1"]
    assert vote["judge"] == "glm-5.1"
    assert vote["judge_prompt_sha256"] == grader._sha256_text(prompt)


def test_load_judge_votes_cache_rejects_wrong_openrouter_route(tmp_path):
    votes = tmp_path / "votes.jsonl"
    votes.write_text(
        json.dumps(
            {
                "custom_id": "c1",
                "judge": "command-a-03-2025",
                "label": "Comply",
                "source": "native",
                "provider": "cohere",
                "model": "command-a-03-2025",
                "judge_prompt_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected vote route"):
        grader.load_judge_votes_cache(
            [votes],
            expected_routes=grader.expected_judge_vote_routes("openrouter"),
        )
