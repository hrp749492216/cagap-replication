#!/usr/bin/env python3
"""Merge offline grading outputs for pilot and measurement-replay runs.

Uses the current three-slot family-excluded judge panel when vote files are
present. For archived pre-swap pilot artifacts, pass the historical judge ids
explicitly with ``--judges``.

Pass ``--out-dir`` explicitly for every replay so regenerated labels cannot
silently land in an old pilot directory.

Usage:
    python scripts/merge_grading_outputs.py [--subjects claude-opus-4-7,...]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import judge_panel as judge_panel_resolver  # noqa: E402

logging.basicConfig(
    level="INFO", format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("merge_grading_outputs")


PROMPTS_PATH = REPO_ROOT / "experiments/aligned/prompts.jsonl"


PILOT2_SUBJECTS = [
    "claude-opus-4-7",
    "gemini-3.5-flash",
    "grok-4.3",
    "gemma-4-31b-it",
    "gpt-5.5",
    "glm-4.7-flash",
    "kimi-k2.6",
    "llama-3.3-70b-instruct",
    "mistral-medium-3.5-128b",
    "qwen3.6-35b-a3b",
]


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_input_path(subject_id: str, *, judge_batches_dir: Path) -> Path:
    manifest = judge_batches_dir / subject_id / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    raw = payload.get("input_path")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / raw
    return path


def _vote_sources(
    subject_id: str,
    *,
    judge_batches_dir: Path,
    judges: list[str] | None = None,
    command_a_priority: str = "self_hosted",
    vote_route_profile: str = "any",
) -> list[Path]:
    """Return vote-source paths for a subject in main-run route preference order.

    Per CLAUDE.md (Provider-batch wiring gotchas) and preregistration §6.F/6.G:

      GLM 5.1: canonical local self-hosted vLLM route, zai-direct primary,
        OpenRouter pinned-fallback (whole-partition). Stale `glm_openrouter`
        slug is intentionally not enumerated.
      Cohere Command A: D6 canonical self-hosted vLLM from
        CohereLabs/c4ai-command-a-03-2025 on external GPU, with native /v2/chat
        and OpenRouter retained only as non-canonical fallback/provenance unless
        explicitly selected. Legacy `cohere_*` and
        `command_a_cohere_openrouter` canary slugs are intentionally not
        enumerated; the bad OpenRouter Command A pilot artifacts are explicitly
        listed in notes/noncanonical_artifacts_excluded_from_lock.md.
      Mistral Large 3: native route primary, Ollama Cloud approved fallback,
        OpenRouter pinned-provider fallback. Existing offline artifacts may use
        the native slug or the `mistral_openrouter` directory.

    The function consumes the FIRST existing path per judge family ("first
    wins, then break"); ordering encodes the route-preference table.
    """
    requested_judges = judges or [
        "glm-5.1",
        "command-a-03-2025",
        "mistral-large-2512",
    ]
    sources: list[Path] = []
    subject_dir = judge_batches_dir / subject_id
    if vote_route_profile == "openrouter":
        required = [
            _openrouter_vote_path(subject_dir, judge_id)
            for judge_id in requested_judges
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "vote_route_profile='openrouter' requires OpenRouter vote files; "
                f"missing={missing}"
            )
        return required
    if vote_route_profile not in ("any", ""):
        raise ValueError(f"unknown vote_route_profile {vote_route_profile!r}")

    singleton_candidates = (subject_dir / "judge_votes.jsonl",)
    candidate_groups = {
        "glm-5.1": (
            subject_dir / "glm_5_1_self_hosted" / "votes.jsonl",  # B300 local canonical
            subject_dir / "glm_online" / "votes.jsonl",          # T1 zai-direct
            subject_dir / "glm_zai_openrouter" / "votes.jsonl",  # T2 OR pinned-fallback
        ),
        "mistral-large-2512": (
            subject_dir / "mistral-large-2512" / "votes.jsonl",
            subject_dir / "mistral_openrouter" / "votes.jsonl",
        ),
        "gpt-5.5-judge": (
            subject_dir / "gpt_5_5_openrouter" / "votes.jsonl",
            subject_dir / "gpt-5.5-judge" / "votes.jsonl",
        ),
        "claude-opus-4-7-judge": (
            subject_dir / "claude_opus_4_7_openrouter" / "votes.jsonl",
            subject_dir / "claude-opus-4-7-judge" / "votes.jsonl",
        ),
    }
    # Validate an explicitly selected Command A route only when Command A is
    # actually part of this subject's family-excluded panel.  Without this
    # guard, a GLM-only or Mistral-only merge incorrectly fails because a
    # Command A artifact is absent even though that lane was not requested.
    if "command-a-03-2025" in requested_judges:
        if command_a_priority == "self_hosted":
            self_hosted_path = subject_dir / "command_a_self_hosted" / "votes.jsonl"
            if not self_hosted_path.exists():
                raise FileNotFoundError(
                    f"command_a_priority='self_hosted' requires {self_hosted_path}"
                )
            command_a_candidates = (
                self_hosted_path,                                       # D6 canonical HF self-host
                subject_dir / "command_a_online" / "votes.jsonl",       # T1 native provenance
                subject_dir / "command-a-03-2025" / "votes.jsonl",      # T1 native alternate
                subject_dir / "command_a_openrouter" / "votes.jsonl",   # T2 OR pinned-fallback
            )
        elif command_a_priority == "openrouter":
            openrouter_path = subject_dir / "command_a_openrouter" / "votes.jsonl"
            if not openrouter_path.exists():
                raise FileNotFoundError(
                    f"command_a_priority='openrouter' requires {openrouter_path}"
                )
            command_a_candidates = (
                openrouter_path,                                      # Measurement Replay 1 OR path
                subject_dir / "command_a_online" / "votes.jsonl",      # T1 native fallback
                subject_dir / "command-a-03-2025" / "votes.jsonl",     # T1 native alternate
                subject_dir / "command_a_self_hosted" / "votes.jsonl", # D6 HF self-host
            )
        else:
            command_a_candidates = (
                subject_dir / "command_a_online" / "votes.jsonl",       # T1 native
                subject_dir / "command-a-03-2025" / "votes.jsonl",      # T1 native alternate
                subject_dir / "command_a_openrouter" / "votes.jsonl",   # T2 OR pinned-fallback
                subject_dir / "command_a_self_hosted" / "votes.jsonl",  # D6 HF self-host
            )
        candidate_groups["command-a-03-2025"] = command_a_candidates
    for candidate in singleton_candidates:
        if candidate.exists():
            sources.append(candidate)
    for judge_id in requested_judges:
        group = candidate_groups.get(judge_id)
        if group is None:
            group = (
                subject_dir / judge_id / "votes.jsonl",
            )
        for candidate in group:
            if candidate.exists():
                sources.append(candidate)
                break
    return sources


def _openrouter_vote_path(subject_dir: Path, judge_id: str) -> Path:
    paths = {
        "glm-5.1": subject_dir / "glm_zai_openrouter" / "votes.jsonl",
        "command-a-03-2025": subject_dir / "command_a_openrouter" / "votes.jsonl",
        "mistral-large-2512": subject_dir / "mistral_openrouter" / "votes.jsonl",
        "gpt-5.5-judge": subject_dir / "gpt_5_5_openrouter" / "votes.jsonl",
        "claude-opus-4-7-judge": subject_dir / "claude_opus_4_7_openrouter" / "votes.jsonl",
    }
    return paths.get(judge_id, subject_dir / judge_id / "votes.jsonl")


def _judges_from_manifest(subject_id: str, judge_batches_dir: Path) -> list[str] | None:
    manifest = judge_batches_dir / subject_id / "manifest.json"
    if not manifest.exists():
        return None
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    panel = payload.get("judge_panel")
    if isinstance(panel, dict) and isinstance(panel.get("judge_ids"), list):
        return [str(j) for j in panel["judge_ids"]]
    judges = payload.get("judges")
    if isinstance(judges, dict):
        return list(judges.keys())
    return None


def grade_subject(
    subject_id: str,
    judges: list[str],
    out_root: Path,
    *,
    judge_batches_dir: Path,
    prompts_path: Path,
    extractor_cache_dir: Path,
    command_a_priority: str = "self_hosted",
    offline_only: bool = True,
    expected_batches_dir: Path | None = None,
    skip_expected_batch_check: bool = False,
    vote_route_profile: str | None = None,
) -> dict[str, Any]:
    input_path = _resolve_input_path(subject_id, judge_batches_dir=judge_batches_dir)
    out_dir = out_root / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "labeled_responses.jsonl"
    raw_log = out_dir / "raw_api_responses.jsonl"
    votes = _vote_sources(
        subject_id,
        judge_batches_dir=judge_batches_dir,
        judges=judges,
        command_a_priority=command_a_priority,
        vote_route_profile=(
            vote_route_profile
            if vote_route_profile is not None
            else ("openrouter" if command_a_priority == "openrouter" else "any")
        ),
    )
    if not votes:
        return {"subject_id": subject_id, "error": "no_vote_sources"}

    cmd: list[str] = [
        sys.executable,
        "scripts/grade_subject_responses.py",
        "--input", str(input_path),
        "--output", str(output),
        "--raw-response-log", str(raw_log),
        "--extractor-cache-dir", str(extractor_cache_dir),
        "--precomputed-extractions", str(judge_batches_dir / subject_id / "extractions.jsonl"),
        "--prompts", str(prompts_path),
    ]
    expected_batch = None
    if expected_batches_dir is not None:
        expected_batch = expected_batches_dir / f"input_{subject_id}.jsonl"
    elif (judge_batches_dir / subject_id / f"input_{subject_id}.jsonl").exists():
        expected_batch = judge_batches_dir / subject_id / f"input_{subject_id}.jsonl"
    if expected_batch is not None and expected_batch.exists():
        cmd += ["--expected-batch", str(expected_batch)]
    elif skip_expected_batch_check:
        cmd += ["--skip-expected-batch-check"]
    else:
        raise FileNotFoundError(
            f"expected batch file not found for {subject_id!r}; pass "
            "--expected-batches-dir or --skip-expected-batch-check only for "
            "developer/debug partial grading."
        )
    if offline_only:
        cmd.append("--offline-only")
    if command_a_priority == "openrouter" or vote_route_profile == "openrouter":
        cmd += ["--judge-vote-route-profile", "openrouter"]
    for judge in judges:
        cmd += ["--judge", judge]
    for src in votes:
        cmd += ["--judge-votes-source", str(src)]

    logger.info("grading %s (output=%s)", subject_id, output)
    completed = subprocess.run(
        cmd, cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    summary: dict[str, Any] = {
        "subject_id": subject_id,
        "input_path": _display_path(input_path),
        "output_path": _display_path(output),
        "raw_response_log": _display_path(raw_log),
        "judge_batches_dir": _display_path(judge_batches_dir),
        "prompts_path": _display_path(prompts_path),
        "vote_sources": [_display_path(p) for p in votes],
        "offline_only": offline_only,
        "expected_batch": _display_path(expected_batch) if expected_batch else None,
        "skip_expected_batch_check": skip_expected_batch_check,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        "stdout_tail": completed.stdout[-2000:] if completed.stdout else "",
    }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", default="")
    p.add_argument(
        "--judges",
        default="auto",
        help=(
            "Comma-separated judge ids, or 'auto' to use each subject's "
            "family-excluded panel from the judge-batch manifest / roster."
        ),
    )
    p.add_argument(
        "--roster",
        type=Path,
        default=REPO_ROOT / "config" / "roster.json",
        help="Roster JSON used when --judges=auto and no manifest panel exists.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        required=True,
        help=(
            "Output directory root. Required so replay labels cannot "
            "accidentally overwrite or land beside an older pilot pass."
        ),
    )
    p.add_argument(
        "--judge-batches-dir",
        type=Path,
        required=True,
        help=(
            "Root containing per-subject judge batch manifests and vote files. "
            "Required so replay merges cannot silently read stale pilot batches."
        ),
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=PROMPTS_PATH,
        help=(
            "Prompt lookup used by the grader. Defaults to the current main "
            "aligned corpus; pass pilot_prompts.jsonl explicitly for archived "
            "pilot replay."
        ),
    )
    p.add_argument(
        "--extractor-cache-dir",
        type=Path,
        required=True,
        help=(
            "Extractor cache directory passed through to grade_subject_responses.py. "
            "Required so replay merges cannot silently read stale extractor caches."
        ),
    )
    p.add_argument(
        "--command-a-priority",
        choices=("self_hosted", "native", "openrouter"),
        default="self_hosted",
        help=(
            "Command A vote source preference. Default self_hosted follows "
            "D6; use native or openrouter only for explicitly documented "
            "replays/provenance paths."
        ),
    )
    p.add_argument(
        "--allow-live-judge-fallback",
        action="store_true",
        help=(
            "Do not pass --offline-only to grade_subject_responses.py. "
            "Confirmatory replay/main merges should not set this."
        ),
    )
    p.add_argument(
        "--expected-batches-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing input_<subject>.jsonl files for "
            "the raw-completeness check."
        ),
    )
    p.add_argument(
        "--skip-expected-batch-check",
        action="store_true",
        help=(
            "Developer/debug escape hatch: pass --skip-expected-batch-check "
            "through to grade_subject_responses.py when no expected batch file "
            "exists. Confirmatory replay/main merges should not set this."
        ),
    )
    p.add_argument(
        "--judge-vote-route-profile",
        choices=("any", "openrouter"),
        default=None,
        help=(
            "Optional strict route profile for cached judge votes. Defaults to "
            "openrouter when --command-a-priority=openrouter, otherwise any."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subjects = (
        [s.strip() for s in args.subjects.split(",") if s.strip()]
        or PILOT2_SUBJECTS
    )
    explicit_judges = None
    if args.judges.strip().lower() != "auto":
        explicit_judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    roster = None
    if explicit_judges is None:
        roster = judge_panel_resolver.load_roster(args.roster)
    summary: dict[str, Any] = {}
    out_root = args.out_dir.resolve() if not args.out_dir.is_absolute() else args.out_dir
    judge_batches_dir = (
        args.judge_batches_dir.resolve()
        if not args.judge_batches_dir.is_absolute()
        else args.judge_batches_dir
    )
    prompts_path = args.prompts.resolve() if not args.prompts.is_absolute() else args.prompts
    extractor_cache_dir = (
        args.extractor_cache_dir.resolve()
        if not args.extractor_cache_dir.is_absolute()
        else args.extractor_cache_dir
    )
    expected_batches_dir = None
    if args.expected_batches_dir is not None:
        expected_batches_dir = (
            args.expected_batches_dir.resolve()
            if not args.expected_batches_dir.is_absolute()
            else args.expected_batches_dir
        )
    for subject_id in subjects:
        if explicit_judges is None:
            judges = _judges_from_manifest(subject_id, judge_batches_dir)
            if judges is None:
                judges = list(
                    judge_panel_resolver.resolve_judge_panel(roster, subject_id)[
                        "judge_ids"
                    ]
                )
        else:
            judges = explicit_judges
        try:
            summary[subject_id] = grade_subject(
                subject_id,
                judges,
                out_root,
                judge_batches_dir=judge_batches_dir,
                prompts_path=prompts_path,
                extractor_cache_dir=extractor_cache_dir,
                command_a_priority=args.command_a_priority,
                offline_only=not args.allow_live_judge_fallback,
                expected_batches_dir=expected_batches_dir,
                skip_expected_batch_check=args.skip_expected_batch_check,
                vote_route_profile=args.judge_vote_route_profile,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("grading failed for %s: %s", subject_id, exc)
            summary[subject_id] = {"error": str(exc)}

    summary_path = out_root / "grade_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    failed = [
        k for k, v in summary.items()
        if isinstance(v, dict) and (v.get("error") or v.get("returncode"))
    ]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
