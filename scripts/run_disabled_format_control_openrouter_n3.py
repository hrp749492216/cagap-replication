#!/usr/bin/env python3
"""Run the frozen all-OpenRouter n=3 successor through its audited adapter.

The source roster records each model's original-study substrate. This adapter
creates an in-memory runtime view for successor subjects whose prospectively
frozen route is OpenRouter, then delegates every validation, authorization,
budget, request, response, and durability invariant to the existing
exploratory launcher and byte-pinned scientific runner.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_disabled_format_control as control  # noqa: E402
from scripts import run_disabled_format_control_exploratory as exploratory  # noqa: E402
from scripts.run_synthetic_prompts import Config  # noqa: E402


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "experiments/disabled_format_control/protocol_v3_openrouter_n3_eligible.json"
)
DEFAULT_LOCKED_INPUTS = (
    REPO_ROOT
    / "experiments/disabled_format_control/locked_inputs_manifest_v3_openrouter_n3_eligible.json"
)
DEFAULT_GOVERNANCE_AMENDMENT = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_governance_amendment_v3_openrouter_n3_eligible_2026-09-05.json"
)
DEFAULT_EXECUTION_AUTHORIZATION = (
    REPO_ROOT
    / "experiments/disabled_format_control/"
    "execution_authorization_v3_openrouter_n3_eligible.json"
)


def protocol_route_entry(
    subject: str,
    *,
    route: str,
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Return a runtime roster entry controlled by one exact protocol route."""
    if route != "openrouter" or descriptor.get("route") != "openrouter":
        raise ValueError(f"all-OpenRouter adapter rejects {subject} route {route!r}")
    if "model_slug" not in descriptor or not descriptor.get("provider_only"):
        raise ValueError(f"incomplete frozen OpenRouter descriptor for {subject}")
    entry = copy.deepcopy(Config.SUBJECTS[subject])
    entry["route_override"] = "openrouter"
    entry["openrouter"] = {
        "slug": descriptor["model_slug"],
        "pinned_provider": descriptor["provider_only"][0],
        "quantizations": list(descriptor.get("quantizations", [])),
        "decoding": copy.deepcopy(descriptor["decoding"]),
    }
    # Local precedence is correct for the original study, but this successor's
    # route is the separately frozen OpenRouter experimental unit. Remove the
    # original substrate only in this process-local compatibility view.
    if entry.get("tier") == "local_self_hosted" or entry.get("local_config"):
        entry.pop("local_config", None)
        entry["tier"] = "hosted_api"
    return entry


def install_protocol_route_entries(protocol: dict[str, Any]) -> None:
    """Install process-local entries for every frozen successor candidate."""
    routes = protocol.get("route_policy", {}).get("routes")
    subjects = protocol.get("population", {}).get("subjects")
    if not isinstance(routes, dict) or not isinstance(subjects, list):
        raise ValueError("all-OpenRouter protocol population/routes are missing")
    if set(map(str, subjects)) != set(map(str, routes)):
        raise ValueError("all-OpenRouter protocol subject/route population mismatch")
    for raw_subject in subjects:
        subject = str(raw_subject)
        descriptor = routes[subject]
        if not isinstance(descriptor, dict):
            raise ValueError(f"invalid frozen route descriptor for {subject}")
        Config.SUBJECTS[subject] = protocol_route_entry(
            subject,
            route=str(descriptor.get("route") or ""),
            descriptor=descriptor,
        )


def _parse_args(argv: list[str] | None) -> Any:
    args = exploratory.parse_args(argv)
    if args.protocol == exploratory.DEFAULT_PROTOCOL:
        args.protocol = DEFAULT_PROTOCOL
    if args.locked_inputs_manifest == exploratory.DEFAULT_LOCKED_INPUTS:
        args.locked_inputs_manifest = DEFAULT_LOCKED_INPUTS
    if args.execution_governance_amendment == exploratory.DEFAULT_GOVERNANCE_AMENDMENT:
        args.execution_governance_amendment = DEFAULT_GOVERNANCE_AMENDMENT
    if args.execution_authorization == exploratory.DEFAULT_EXECUTION_AUTHORIZATION:
        args.execution_authorization = DEFAULT_EXECUTION_AUTHORIZATION
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        protocol, _protocol_sha256 = control.load_protocol(args.protocol)
        install_protocol_route_entries(protocol)
        result = (
            exploratory.submit_generated(args)
            if args.submit
            else exploratory.validate_ready(args)
        )
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
