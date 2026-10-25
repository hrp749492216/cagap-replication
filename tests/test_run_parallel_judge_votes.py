from __future__ import annotations

import json
import socket
import subprocess

from scripts import run_parallel_judge_votes as lanes


def _make_inputs(root, subject: str) -> None:
    inputs = root / subject / "judge_inputs"
    inputs.mkdir(parents=True)
    for judge in lanes.JUDGE_IDS:
        (inputs / f"{judge}.jsonl").write_text(
            json.dumps({"custom_id": f"{subject}_{judge}", "judge_prompt": "prompt"}) + "\n",
            encoding="utf-8",
        )


def _allow_network_preflight(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))])


def test_parallel_judge_votes_builds_expected_lane_commands(tmp_path, monkeypatch):
    _allow_network_preflight(monkeypatch)
    _make_inputs(tmp_path, "subject-a")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "env": kwargs.get("env"), "cwd": kwargs.get("cwd")})
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(lanes.subprocess, "run", fake_run)

    rc = lanes.main(
        [
            "--judge-batches-dir",
            str(tmp_path),
            "--subjects",
            "subject-a",
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--glm-row-workers",
            "2",
            "--mistral-row-workers",
            "4",
        ]
    )

    assert rc == 0
    commands = [" ".join(call["cmd"]) for call in calls]
    # D6 moved the canonical Command A lane to the separate self-hosted
    # dispatcher, so the API-lane coordinator defaults to GLM and Mistral.
    assert len(commands) == 2
    assert any("scripts/run_openrouter_judge.py" in c and "--judge glm-5.1" in c for c in commands)
    assert any("scripts/run_openrouter_judge.py" in c and "--judge mistral-large-2512" in c for c in commands)
    assert any("--out " + str(tmp_path / "subject-a" / "glm_zai_openrouter" / "votes.jsonl") in c for c in commands)
    assert any("--out " + str(tmp_path / "subject-a" / "mistral_openrouter" / "votes.jsonl") in c for c in commands)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_subject_runs"] == 0


def test_parallel_judge_votes_continues_other_lanes_after_failure(tmp_path, monkeypatch):
    _allow_network_preflight(monkeypatch)
    _make_inputs(tmp_path, "subject-a")
    _make_inputs(tmp_path, "subject-b")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        text = " ".join(cmd)
        if "glm-5.1" in text and "subject-a" in text:
            return subprocess.CompletedProcess(cmd, 1, stdout="bad", stderr="invalid")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(lanes.subprocess, "run", fake_run)

    rc = lanes.main(
        [
            "--judge-batches-dir",
            str(tmp_path),
            "--subjects",
            "subject-a,subject-b",
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert rc == 1
    # Failure in one GLM subject does not prevent the rest of the matrix.
    assert len(calls) == 4
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_subject_runs"] == 1
    assert summary["lanes"]["glm-5.1"]["subjects_attempted"] == 2


def test_parallel_judge_votes_can_use_command_a_openrouter(tmp_path, monkeypatch):
    _allow_network_preflight(monkeypatch)
    _make_inputs(tmp_path, "subject-a")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(lanes.subprocess, "run", fake_run)

    assert lanes.main(
        [
            "--judge-batches-dir",
            str(tmp_path),
            "--subjects",
            "subject-a",
            "--judges",
            "command-a-03-2025",
            "--command-a-route",
            "openrouter",
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    ) == 0
    command = " ".join(calls[0])
    assert "scripts/run_openrouter_judge.py" in command
    assert "--judge command-a-03-2025" in command
    assert str(tmp_path / "subject-a" / "command_a_openrouter" / "votes.jsonl") in command


def test_parallel_judge_votes_auto_uses_per_subject_inputs(tmp_path, monkeypatch):
    _allow_network_preflight(monkeypatch)
    subject_a = tmp_path / "subject-a" / "judge_inputs"
    subject_a.mkdir(parents=True)
    (subject_a / "glm-5.1.jsonl").write_text(
        json.dumps({"custom_id": "a_glm", "judge_prompt": "prompt"}) + "\n",
        encoding="utf-8",
    )
    subject_b = tmp_path / "subject-b" / "judge_inputs"
    subject_b.mkdir(parents=True)
    (subject_b / "gpt-5.5-judge.jsonl").write_text(
        json.dumps({"custom_id": "b_gpt", "judge_prompt": "prompt"}) + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(lanes.subprocess, "run", fake_run)

    rc = lanes.main(
        [
            "--judge-batches-dir",
            str(tmp_path),
            "--judges",
            "auto",
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert rc == 0
    commands = [" ".join(call) for call in calls]
    assert len(commands) == 2
    assert any("--judge glm-5.1" in command and "subject-a" in command for command in commands)
    assert any("--judge gpt-5.5-judge" in command and "subject-b" in command for command in commands)
    assert not any("--judge glm-5.1" in command and "subject-b" in command for command in commands)
    assert any(str(tmp_path / "subject-b" / "gpt_5_5_openrouter" / "votes.jsonl") in command for command in commands)


def test_parallel_judge_votes_network_preflight_blocks_before_dispatch(tmp_path, monkeypatch):
    _make_inputs(tmp_path, "subject-a")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    def fail_dns(host, port):
        raise socket.gaierror("blocked")

    monkeypatch.setattr(lanes.subprocess, "run", fake_run)
    monkeypatch.setattr(socket, "getaddrinfo", fail_dns)

    rc = lanes.main(
        [
            "--judge-batches-dir",
            str(tmp_path),
            "--subjects",
            "subject-a",
            "--summary-json",
            str(tmp_path / "summary.json"),
        ]
    )

    assert rc == 2
    assert calls == []
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_subject_runs"] == 2
    assert summary["network_preflight"]["ok"] is False
    assert summary["network_preflight"]["failed_hosts"] == ["openrouter.ai"]
