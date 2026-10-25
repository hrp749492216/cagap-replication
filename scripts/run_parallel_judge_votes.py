#!/usr/bin/env python3
"""Run OpenRouter-capable family-excluded judge vote lanes over prepared inputs.

The per-judge dispatchers already provide row-level resume semantics: they
validate existing votes against the current judge prompt hash, keep valid rows,
move stale/invalid rows to ``votes.jsonl.ignored.jsonl``, and only dispatch
missing rows. This coordinator preserves those semantics while running active
judge lanes in parallel:

  - GLM lane: OpenRouter pinned fallback -> ``glm_zai_openrouter/votes.jsonl``
  - Mistral lane: OpenRouter pinned fallback -> ``mistral_openrouter/votes.jsonl``
  - Command A API lanes: opt-in only after D6; canonical Command A dispatch is
    self-hosted and written to ``command_a_self_hosted/votes.jsonl`` by the
    local GPU runner.

Use this coordinator when the run has explicitly chosen the OpenRouter-capable
judge lane profile or is replaying those artifacts. Roster-native GLM/Mistral
routes remain the preferred source where available, and merge precedence treats
these OpenRouter directories as fallbacks.

Each lane processes subjects sequentially to keep per-provider rate limits
predictable; the lanes themselves run concurrently. Subject-level
family-exclusion is resolved before dispatch. A subject-level failure in one
lane is recorded and does not cancel the other lanes.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

JUDGE_IDS = (
    "glm-5.1",
    "mistral-large-2512",
    "command-a-03-2025",
    "gpt-5.5-judge",
    "claude-opus-4-7-judge",
)
ROUTE_HOSTS = {
    "openrouter": "openrouter.ai",
    "cohere_native": "api.cohere.com",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _subjects_from_root(root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    subjects = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "judge_inputs").is_dir():
            subjects.append(path.name)
    return subjects


def _judges_from_inputs(root: Path, subjects: list[str]) -> list[str]:
    seen: set[str] = set()
    for subject_id in subjects:
        inputs = root / subject_id / "judge_inputs"
        if not inputs.exists():
            continue
        for path in sorted(inputs.glob("*.jsonl")):
            seen.add(path.stem)
    return [judge_id for judge_id in JUDGE_IDS if judge_id in seen]


def _task_paths(
    judge_batches_dir: Path,
    subject_id: str,
    judge_id: str,
    *,
    command_a_route: str,
) -> tuple[Path, Path, str]:
    subject_dir = judge_batches_dir / subject_id
    input_path = subject_dir / "judge_inputs" / f"{judge_id}.jsonl"
    if judge_id == "glm-5.1":
        return input_path, subject_dir / "glm_zai_openrouter" / "votes.jsonl", "openrouter"
    if judge_id == "mistral-large-2512":
        return input_path, subject_dir / "mistral_openrouter" / "votes.jsonl", "openrouter"
    if judge_id == "gpt-5.5-judge":
        return input_path, subject_dir / "gpt_5_5_openrouter" / "votes.jsonl", "openrouter"
    if judge_id == "claude-opus-4-7-judge":
        return input_path, subject_dir / "claude_opus_4_7_openrouter" / "votes.jsonl", "openrouter"
    if judge_id == "command-a-03-2025":
        if command_a_route == "native":
            return input_path, subject_dir / "command_a_online" / "votes.jsonl", "cohere_native"
        if command_a_route == "openrouter":
            return input_path, subject_dir / "command_a_openrouter" / "votes.jsonl", "openrouter"
        raise ValueError(f"unknown command_a_route={command_a_route!r}")
    raise ValueError(f"unknown judge_id={judge_id!r}")


def _build_command(
    *,
    judge_id: str,
    route: str,
    input_path: Path,
    out_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    if route == "openrouter":
        row_workers = _openrouter_row_workers(judge_id, args)
        min_interval = _openrouter_min_interval(judge_id, args)
        cmd = [
            sys.executable,
            "scripts/run_openrouter_judge.py",
            "--judge-inputs", str(input_path),
            "--out", str(out_path),
            "--judge", judge_id,
            "--row-workers", str(row_workers),
            "--min-interval-seconds", str(min_interval),
        ]
        max_tokens = _openrouter_max_tokens(judge_id, args)
        if max_tokens is not None:
            cmd += ["--max-tokens", str(max_tokens)]
        if args.allow_invalid_completions:
            cmd.append("--allow-invalid-completions")
        return cmd
    if route == "cohere_native":
        cmd = [
            sys.executable,
            "scripts/canary_cohere_judge.py",
            "--judge-inputs", str(input_path),
            "--out", str(out_path),
            "--judge", judge_id,
            "--model", "command-a-03-2025",
            "--row-workers", str(args.cohere_row_workers),
            "--min-interval-seconds", str(args.cohere_min_interval_seconds),
        ]
        if args.allow_invalid_completions:
            cmd.append("--allow-invalid-completions")
        return cmd
    raise ValueError(f"unknown route={route!r}")


def _openrouter_row_workers(judge_id: str, args: argparse.Namespace) -> int:
    if judge_id == "glm-5.1":
        return args.glm_row_workers
    if judge_id == "mistral-large-2512":
        return args.mistral_row_workers
    if judge_id == "command-a-03-2025":
        return args.command_a_openrouter_row_workers
    return args.substitute_openrouter_row_workers


def _openrouter_min_interval(judge_id: str, args: argparse.Namespace) -> float:
    if judge_id == "glm-5.1":
        return args.glm_min_interval_seconds
    if judge_id == "mistral-large-2512":
        return args.mistral_min_interval_seconds
    if judge_id == "command-a-03-2025":
        return args.command_a_openrouter_min_interval_seconds
    return args.substitute_openrouter_min_interval_seconds


def _openrouter_max_tokens(judge_id: str, args: argparse.Namespace) -> int | None:
    if judge_id == "glm-5.1":
        return args.glm_max_tokens
    if judge_id == "mistral-large-2512":
        return args.mistral_max_tokens
    if judge_id == "command-a-03-2025":
        return args.command_a_openrouter_max_tokens
    return args.substitute_openrouter_max_tokens


def _run_one_subject(
    *,
    judge_batches_dir: Path,
    subject_id: str,
    judge_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    input_path, out_path, route = _task_paths(
        judge_batches_dir,
        subject_id,
        judge_id,
        command_a_route=args.command_a_route,
    )
    record: dict[str, Any] = {
        "subject_id": subject_id,
        "judge_id": judge_id,
        "route": route,
        "judge_inputs": _display_path(input_path),
        "out": _display_path(out_path),
        "expected_rows": _jsonl_count(input_path),
        "started_at": _now_utc(),
    }
    if not input_path.exists():
        if args.skip_missing_judge_inputs:
            record.update(
                {
                    "returncode": 0,
                    "skipped": True,
                    "error": "judge_inputs_missing_skipped",
                    "finished_at": _now_utc(),
                }
            )
            return record
        record.update({"returncode": 2, "error": "judge_inputs_missing", "finished_at": _now_utc()})
        return record
    cmd = _build_command(
        judge_id=judge_id,
        route=route,
        input_path=input_path,
        out_path=out_path,
        args=args,
    )
    env = os.environ.copy()
    if route == "cohere_native" and args.cohere_env_source == "secrets_file":
        env.pop("COHERE_API_KEY", None)
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    record.update(
        {
            "returncode": completed.returncode,
            "finished_at": _now_utc(),
            "stdout_tail": completed.stdout[-3000:] if completed.stdout else "",
            "stderr_tail": completed.stderr[-3000:] if completed.stderr else "",
        }
    )
    return record


def _run_lane(
    *,
    judge_batches_dir: Path,
    subjects: list[str],
    judge_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for subject_id in subjects:
        record = _run_one_subject(
            judge_batches_dir=judge_batches_dir,
            subject_id=subject_id,
            judge_id=judge_id,
            args=args,
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "event": "judge_subject_finished",
                    "judge_id": judge_id,
                    "subject_id": subject_id,
                    "returncode": record.get("returncode"),
                    "out": record.get("out"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if record.get("returncode") != 0 and args.stop_lane_on_error:
            break
    return {
        "judge_id": judge_id,
        "subjects_attempted": sum(1 for r in records if not r.get("skipped")),
        "subjects_failed": sum(1 for r in records if r.get("returncode") != 0),
        "records": records,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_hosts(judges: list[str], *, command_a_route: str) -> list[str]:
    hosts: set[str] = set()
    for judge_id in judges:
        if judge_id in {
            "glm-5.1",
            "mistral-large-2512",
            "gpt-5.5-judge",
            "claude-opus-4-7-judge",
        }:
            hosts.add(ROUTE_HOSTS["openrouter"])
        elif judge_id == "command-a-03-2025":
            if command_a_route == "native":
                hosts.add(ROUTE_HOSTS["cohere_native"])
            else:
                hosts.add(ROUTE_HOSTS["openrouter"])
    return sorted(hosts)


def _network_preflight(hosts: list[str]) -> dict[str, Any]:
    failures: dict[str, str] = {}
    for host in hosts:
        try:
            socket.getaddrinfo(host, 443)
        except OSError as exc:
            failures[host] = f"{type(exc).__name__}: {exc}"
    return {
        "ok": not failures,
        "checked_hosts": hosts,
        "failed_hosts": sorted(failures),
        "failures": failures,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-batches-dir", type=Path, required=True)
    parser.add_argument("--subjects", default="", help="Comma-separated subject ids. Default: all subject dirs.")
    parser.add_argument(
        "--judges",
        default="glm-5.1,mistral-large-2512",
        help=(
            "Comma-separated judges to run, or 'auto' to infer from judge_inputs "
            "files. After D6, auto excludes Command A API lanes unless "
            "--include-command-a-api-in-auto is set."
        ),
    )
    parser.add_argument(
        "--include-command-a-api-in-auto",
        action="store_true",
        help="When --judges=auto, include command-a-03-2025 API lanes despite D6.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--stop-lane-on-error", action="store_true")
    parser.add_argument("--allow-invalid-completions", action="store_true")
    parser.add_argument(
        "--skip-network-preflight",
        action="store_true",
        help="Skip DNS preflight for judge endpoint hosts before launching subprocesses.",
    )

    parser.add_argument("--glm-row-workers", type=int, default=1)
    parser.add_argument("--glm-min-interval-seconds", type=float, default=1.0)
    parser.add_argument("--glm-max-tokens", type=int, default=4096)

    parser.add_argument("--mistral-row-workers", type=int, default=3)
    parser.add_argument("--mistral-min-interval-seconds", type=float, default=0.25)
    parser.add_argument("--mistral-max-tokens", type=int, default=None)

    parser.add_argument("--command-a-route", choices=("native", "openrouter"), default="native")
    parser.add_argument("--cohere-row-workers", type=int, default=1)
    parser.add_argument("--cohere-min-interval-seconds", type=float, default=5.0)
    parser.add_argument(
        "--cohere-env-source",
        choices=("secrets_file", "environment"),
        default="secrets_file",
        help=(
            "secrets_file removes inherited COHERE_API_KEY so scripts/cagap_env.py "
            "loads ~/.cagap-secrets; environment preserves the inherited value."
        ),
    )
    parser.add_argument("--command-a-openrouter-row-workers", type=int, default=1)
    parser.add_argument("--command-a-openrouter-min-interval-seconds", type=float, default=1.0)
    parser.add_argument("--command-a-openrouter-max-tokens", type=int, default=None)
    parser.add_argument("--substitute-openrouter-row-workers", type=int, default=1)
    parser.add_argument("--substitute-openrouter-min-interval-seconds", type=float, default=1.0)
    parser.add_argument("--substitute-openrouter-max-tokens", type=int, default=None)
    parser.set_defaults(skip_missing_judge_inputs=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    judge_batches_dir = (
        args.judge_batches_dir.resolve()
        if not args.judge_batches_dir.is_absolute()
        else args.judge_batches_dir
    )
    subjects = _subjects_from_root(
        judge_batches_dir,
        [s.strip() for s in args.subjects.split(",") if s.strip()] or None,
    )
    if args.judges.strip().lower() == "auto":
        judges = _judges_from_inputs(judge_batches_dir, subjects)
        if not args.include_command_a_api_in_auto:
            judges = [j for j in judges if j != "command-a-03-2025"]
        args.skip_missing_judge_inputs = True
    else:
        judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    unknown = [j for j in judges if j not in JUDGE_IDS]
    if unknown:
        raise SystemExit(f"unknown judge(s): {unknown}; known={JUDGE_IDS}")
    if not subjects:
        raise SystemExit(f"no subjects found under {judge_batches_dir}")

    summary_path = args.summary_json
    if summary_path is None:
        summary_path = judge_batches_dir / "parallel_judge_votes_summary.json"
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path

    if not args.skip_network_preflight:
        preflight = _network_preflight(_required_hosts(judges, command_a_route=args.command_a_route))
        if not preflight["ok"]:
            failed = len(judges) * len(subjects)
            summary = {
                "schema_version": 1,
                "started_or_completed_at": _now_utc(),
                "judge_batches_dir": _display_path(judge_batches_dir),
                "subjects": subjects,
                "judges": judges,
                "command_a_route": args.command_a_route,
                "failed_subject_runs": failed,
                "network_preflight": preflight,
                "lanes": {},
            }
            _write_summary(summary_path, summary)
            print(
                json.dumps(
                    {
                        "event": "parallel_judge_votes_network_preflight_failed",
                        "failed_hosts": preflight["failed_hosts"],
                        "summary_json": _display_path(summary_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 2

    print(
        json.dumps(
            {
                "event": "parallel_judge_votes_start",
                "judge_batches_dir": _display_path(judge_batches_dir),
                "subjects": subjects,
                "judges": judges,
                "command_a_route": args.command_a_route,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    lane_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(judges), thread_name_prefix="judge-lane") as pool:
        futures = {
            pool.submit(
                _run_lane,
                judge_batches_dir=judge_batches_dir,
                subjects=subjects,
                judge_id=judge_id,
                args=args,
            ): judge_id
            for judge_id in judges
        }
        for fut in as_completed(futures):
            judge_id = futures[fut]
            lane_results[judge_id] = fut.result()
            print(
                json.dumps(
                    {
                        "event": "judge_lane_finished",
                        "judge_id": judge_id,
                        "subjects_failed": lane_results[judge_id]["subjects_failed"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    failed = sum(result["subjects_failed"] for result in lane_results.values())
    summary = {
        "schema_version": 1,
        "started_or_completed_at": _now_utc(),
        "judge_batches_dir": _display_path(judge_batches_dir),
        "subjects": subjects,
        "judges": judges,
        "command_a_route": args.command_a_route,
        "failed_subject_runs": failed,
        "lanes": lane_results,
    }
    _write_summary(summary_path, summary)
    print(
        json.dumps(
            {
                "event": "parallel_judge_votes_done",
                "failed_subject_runs": failed,
                "summary_json": _display_path(summary_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
