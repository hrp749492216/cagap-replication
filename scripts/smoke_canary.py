"""Smoke canary: 1 call per subject × intervention before any real dispatch.

Pre-OSF gate (per the consolidated study-design and pre-registration record).
Catches dispatch breakage end-to-end:

  * roster loads, every subject has a resolvable route
  * canonical body construction passes the schema guard at every step
  * real HTTP call returns 200 with a parseable response shape
  * finish_reason is not "error"; choices[0] has content OR tool_calls

Exit codes
----------
  0  every cell passed
  1  one or more cells failed (report written; investigate)
  2  config error (roster missing, no auth env, no canary tool schema)

Usage
-----
  python scripts/smoke_canary.py --dry-run        # no HTTP; validate construction
  python scripts/smoke_canary.py --live           # real calls (uses env auth)
  python scripts/smoke_canary.py --live --subjects claude-opus-4-7,grok-4.3

Cost
----
  Current active roster: 10 subjects × 4 interventions = 40 calls.
  Hosted/API-only preflight: 5 subjects × 4 interventions = 20 calls.
  Exact cost depends on the current route mix and provider pricing.

Output
------
  outputs/smoke_canary/<utc_ts>/canary_report.json
  outputs/smoke_canary/<utc_ts>/canary_report.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts._tool_schema_guard import (  # noqa: E402
    ToolSchemaError,
    assert_body_clean_of_route_only_fields,
    validate_batch_row,
)
from scripts.raw_http_archive import archive_response_body, response_body_bytes  # noqa: E402


# ---------------------------------------------------------------------------
# Canary stimulus — fixed prompt + minimal tool. Kept self-contained so the
# canary does not depend on the live corpus or any locked materials.
# ---------------------------------------------------------------------------

CANARY_PROMPT = (
    "Reply with the single word OK. If a tool would be appropriate, you may "
    "call it; otherwise reply with text only."
)
CANARY_PROMPTS = (
    CANARY_PROMPT,
    "In one short sentence, say whether the dispatch path is reachable.",
    "Reply with the exact token READY.",
    "Return a brief acknowledgement that contains no more than six words.",
)
CANARY_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "A no-op tool used solely to verify tool-call wiring on a smoke canary.",
        "parameters": {
            "type": "object",
            "properties": {
                "echo": {
                    "type": "string",
                    "description": "Any short string the model wishes to echo back.",
                }
            },
            "required": ["echo"],
        },
    },
}
CANARY_IRRELEVANT_TOOL = {
    "type": "function",
    "function": {
        "name": "off_domain_lookup_postal_code",
        "description": "Look up a postal code given a city name. Unrelated to any pre-registered domain.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name to look up."}
            },
            "required": ["city"],
        },
    },
}

INTERVENTIONS = ("NOTOOLS", "DISABLED", "TOOL", "TOOL_IRRELEVANT")


# ---------------------------------------------------------------------------
# Body construction
# ---------------------------------------------------------------------------

def build_canonical_body(
    model_slug: str,
    intervention: str,
    max_tokens: int = 256,
    prompt_text: str = CANARY_PROMPT,
) -> Dict[str, Any]:
    """Build a provider-portable canonical body for one canary cell.

    Mirrors the runner's body shape minus OR-only transport fields (those
    are added at OR dispatch time only — see notes/pilot0_failed_harness_incident.md
    §7B).
    """
    body: Dict[str, Any] = {
        "model": model_slug,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if intervention == "NOTOOLS":
        return body
    if intervention == "DISABLED":
        body["tools"] = [CANARY_TOOL]
        body["tool_choice"] = "none"
        return body
    if intervention == "TOOL":
        body["tools"] = [CANARY_TOOL]
        body["tool_choice"] = "auto"
        return body
    if intervention == "TOOL_IRRELEVANT":
        body["tools"] = [CANARY_IRRELEVANT_TOOL]
        body["tool_choice"] = "auto"
        return body
    raise ValueError(f"unknown intervention: {intervention!r}")


# ---------------------------------------------------------------------------
# Route resolution — defer to runner so canary uses the same logic as pilot
# ---------------------------------------------------------------------------

def _resolve_route_for_subject(subject_id: str, entry: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (route_name, route_config_dict) for a subject. Reuses the
    runner's select_route() so the canary cannot drift from the real pipe."""
    from scripts.run_synthetic_prompts import select_route  # late import to avoid heavy startup
    return select_route(entry)


def _slug_for_route(route: str, route_cfg: Dict[str, Any], entry: Dict[str, Any]) -> str:
    """Return the model identifier the route expects in body['model']."""
    if "slug" in route_cfg and route_cfg["slug"]:
        return str(route_cfg["slug"])
    # local routes carry served-model identity in local_config; prefer
    # hf_repo (canonical reference id) so the served-model preflight has
    # something deterministic to match against.
    for key in ("served_model_path", "model_id", "hf_repo", "local_dir", "name"):
        if key in route_cfg and route_cfg[key]:
            return str(route_cfg[key])
    raise RuntimeError(
        f"route {route!r} has no resolvable slug for entry "
        f"{entry.get('subject_id') or entry.get('display_name')!r}"
    )


# ---------------------------------------------------------------------------
# Response-shape validator — checks the runner's contract
# ---------------------------------------------------------------------------

def validate_response_shape(payload: Dict[str, Any], intervention: str) -> Tuple[bool, str]:
    """Return (ok, reason). ok=True iff response carries usable content."""
    if not isinstance(payload, dict):
        return False, "response is not a JSON object"
    err = payload.get("error")
    if err:
        return False, f"provider returned error: {err}"
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "response missing choices[]"
    c0 = choices[0]
    if not isinstance(c0, dict):
        return False, "choices[0] not a dict"
    fr = c0.get("finish_reason")
    if fr == "error":
        return False, "choices[0].finish_reason == 'error'"
    msg = c0.get("message") or {}
    has_content = bool(msg.get("content"))
    has_tool_calls = bool(msg.get("tool_calls"))
    if intervention == "TOOL":
        # Either is acceptable; a tool-augmented model may legitimately
        # answer with text on a canary prompt that does not require tool use.
        if not (has_content or has_tool_calls):
            return False, "TOOL cell: neither content nor tool_calls present"
    else:
        if not has_content:
            return False, f"{intervention} cell: choices[0].message.content empty"
    return True, "ok"


# ---------------------------------------------------------------------------
# Dispatch — minimal per-route HTTP, mirroring runner conventions
# ---------------------------------------------------------------------------

def _post(
    route: str,
    route_cfg: Dict[str, Any],
    entry: Dict[str, Any],
    body: Dict[str, Any],
    *,
    custom_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    intervention: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """One HTTP call. Adds OR-only fields only inside the OR branch."""
    import requests  # imported lazily so dry-run does not require it

    subject_id = subject_id or str(entry.get("subject_id") or entry.get("id") or "unknown")
    intervention = intervention or "unknown"
    custom_id = custom_id or f"smoke::{subject_id}::{intervention}"

    if route == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        wrapped = deepcopy(body)
        wrapped["provider"] = {
            "only": [route_cfg.get("pinned_provider")],
            "allow_fallbacks": False,
            "data_collection": "deny",
        }
        wrapped["plugins"] = [{"id": "context-compression", "enabled": False}]
        wrapped["transforms"] = []
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(url, headers=headers, json=wrapped, timeout=120)
        raw_http_capture = archive_response_body(
            response_body_bytes(r),
            role="subject_smoke",
            model_id=subject_id,
            route="openrouter",
            provider=route_cfg.get("pinned_provider"),
            custom_id=custom_id,
            attempt_index=1,
            http_status=getattr(r, "status_code", None),
            url=url,
            request_body=wrapped,
            source="requests.Response.content",
            extra={"intervention": intervention},
        )
        r.raise_for_status()
        return r.json(), raw_http_capture

    if route == "native":
        from scripts.providers import get_adapter
        from scripts.run_synthetic_prompts import _native_headers
        family = entry.get("family") or ""
        adapter = get_adapter(family)
        wrapped = adapter.to_native_body(deepcopy(body), entry) if adapter else deepcopy(body)
        endpoint = route_cfg.get("endpoint")
        url_path = route_cfg.get("url_path")
        if not endpoint or not url_path:
            raise RuntimeError(f"native route missing endpoint/url_path: {route_cfg!r}")
        url = f"{endpoint.rstrip('/')}{url_path}"
        auth_env = route_cfg.get("auth_env")
        if not auth_env or not os.environ.get(auth_env):
            raise RuntimeError(f"native API key env {auth_env!r} not set")
        headers = _native_headers(route_cfg, family)
        r = requests.post(url, headers=headers, json=wrapped, timeout=120)
        raw_http_capture = archive_response_body(
            response_body_bytes(r),
            role="subject_smoke",
            model_id=subject_id,
            route="native",
            provider=family,
            custom_id=custom_id,
            attempt_index=1,
            http_status=getattr(r, "status_code", None),
            url=url,
            request_body=wrapped,
            source="requests.Response.content",
            extra={"intervention": intervention},
        )
        r.raise_for_status()
        data = r.json()
        if family in {"anthropic", "cohere"} and adapter is not None and hasattr(adapter, "from_native_response"):
            data = adapter.from_native_response(data)
        return data, raw_http_capture

    if route == "local":
        # vLLM / MLX-LM are OpenAI-compatible; canary just hits /v1/chat/completions.
        endpoint = route_cfg.get("endpoint") or os.environ.get("CAGAP_VLLM_ENDPOINT") or os.environ.get("CAGAP_MLX_ENDPOINT")
        if not endpoint:
            raise RuntimeError("local route has no endpoint and no CAGAP_VLLM_ENDPOINT/CAGAP_MLX_ENDPOINT")
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        api_key = os.environ.get("VLLM_CLOUD_AUTH", "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        r = requests.post(url, headers=headers, json=body, timeout=120)
        raw_http_capture = archive_response_body(
            response_body_bytes(r),
            role="subject_smoke",
            model_id=subject_id,
            route="local",
            provider="openai_compatible_local",
            custom_id=custom_id,
            attempt_index=1,
            http_status=getattr(r, "status_code", None),
            url=url,
            request_body=body,
            source="requests.Response.content",
            extra={"intervention": intervention},
        )
        r.raise_for_status()
        return r.json(), raw_http_capture

    raise RuntimeError(f"unknown route: {route!r}")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_canary(
    *,
    live: bool,
    subjects_filter: Optional[List[str]] = None,
    out_dir: Optional[Path] = None,
    prompt_count: int = 1,
    max_tokens: int = 256,
    cell_filter: Optional[set[Tuple[str, str, str]]] = None,
) -> Tuple[int, Dict[str, Any]]:
    from scripts.run_synthetic_prompts import _ROSTER, _model_entries  # noqa

    subjects = _model_entries(_ROSTER.get("subjects") or {})
    if subjects_filter:
        keep = set(subjects_filter)
        subjects = {k: v for k, v in subjects.items() if k in keep}
        if not subjects:
            return 2, {"error": f"no subjects matched filter {subjects_filter}"}

    if prompt_count < 1 or prompt_count > len(CANARY_PROMPTS):
        return 2, {
            "error": f"prompt_count must be between 1 and {len(CANARY_PROMPTS)}",
            "n_passed": 0,
            "n_cells": 0,
        }

    prompts = list(CANARY_PROMPTS[:prompt_count])
    cells: List[Dict[str, Any]] = []
    for sid, entry in sorted(subjects.items()):
        for prompt_index, prompt_text in enumerate(prompts, start=1):
            prompt_id = f"smoke_prompt_{prompt_index:02d}"
            for interv in INTERVENTIONS:
                if cell_filter is not None and (sid, prompt_id, interv) not in cell_filter:
                    continue
                cell: Dict[str, Any] = {
                    "subject_id": sid,
                    "prompt_id": prompt_id,
                    "intervention": interv,
                    "status": "pending",
                    "reason": "",
                }
                try:
                    route, route_cfg = _resolve_route_for_subject(sid, entry)
                    cell["route"] = route
                    slug = _slug_for_route(route, route_cfg, entry)
                    cell["slug"] = slug
                    body = build_canonical_body(
                        slug,
                        interv,
                        max_tokens=max_tokens,
                        prompt_text=prompt_text,
                    )
                    custom_id = f"smoke::{sid}::{prompt_id}::{interv}"
                    # Provider-portability check: canonical body must pass the guard
                    row_for_guard = {"custom_id": custom_id, "body": body}
                    validate_batch_row(row_for_guard, source="smoke_canary")
                    # Belt-and-suspenders: explicit clean assertion
                    assert_body_clean_of_route_only_fields(body, source=custom_id)
                    if not live:
                        cell["status"] = "passed_dry_run"
                        cell["reason"] = "body construction + guard validation"
                    else:
                        started = time.perf_counter()
                        payload, raw_http_capture = _post(
                            route,
                            route_cfg,
                            entry,
                            body,
                            custom_id=custom_id,
                            subject_id=sid,
                            intervention=interv,
                        )
                        cell["latency_sec"] = round(time.perf_counter() - started, 3)
                        cell["raw_http_capture"] = raw_http_capture
                        ok, reason = validate_response_shape(payload, interv)
                        cell["status"] = "passed" if ok else "failed"
                        cell["reason"] = reason
                        if ok:
                            cell["finish_reason"] = (
                                payload.get("choices") or [{}]
                            )[0].get("finish_reason")
                except ToolSchemaError as exc:
                    cell["status"] = "failed_guard"
                    cell["reason"] = str(exc)
                except Exception as exc:  # noqa: BLE001 — canary captures all
                    cell["status"] = "failed"
                    cell["reason"] = f"{type(exc).__name__}: {exc}"
                    cell["traceback"] = traceback.format_exc(limit=4)
                cells.append(cell)

    n_total = len(cells)
    n_pass = sum(1 for c in cells if c["status"] in ("passed", "passed_dry_run"))
    n_fail = n_total - n_pass
    report = {
        "computed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "live": live,
        "n_subjects": len({c["subject_id"] for c in cells}),
        "n_prompts": prompt_count,
        "max_tokens": max_tokens,
        "cell_filter_count": len(cell_filter) if cell_filter is not None else None,
        "n_cells": n_total,
        "n_passed": n_pass,
        "n_failed": n_fail,
        "all_passed": n_fail == 0,
        "cells": cells,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "canary_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        (out_dir / "canary_report.md").write_text(_format_md_report(report))
    return (0 if n_fail == 0 else 1), report


def _format_md_report(report: Dict[str, Any]) -> str:
    lines = [
        "# Smoke Canary Report",
        "",
        f"- computed_at_utc: `{report['computed_at_utc']}`",
        f"- live: `{report['live']}`",
        f"- subjects: {report['n_subjects']}",
        f"- prompts: {report.get('n_prompts', 1)}",
        f"- max_tokens: {report.get('max_tokens', 256)}",
        f"- cell_filter_count: {report.get('cell_filter_count')}",
        f"- cells: {report['n_cells']}",
        f"- passed: {report['n_passed']}",
        f"- failed: {report['n_failed']}",
        f"- all_passed: **{report['all_passed']}**",
        "",
        "| subject | prompt | intervention | route | status | reason |",
        "|---------|--------|--------------|-------|--------|--------|",
    ]
    for c in report["cells"]:
        lines.append(
            f"| `{c['subject_id']}` | `{c.get('prompt_id', 'smoke_prompt_01')}` | `{c['intervention']}` | `{c.get('route', '?')}` | `{c['status']}` | {c['reason'][:140]} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pre-OSF smoke canary for the pilot/main dispatch chain.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Build + validate canonical bodies only; no HTTP.")
    g.add_argument("--live", action="store_true", help="Real HTTP per subject × intervention (uses env auth).")
    p.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma-separated subset of subject_ids to test.",
    )
    p.add_argument(
        "--prompt-count",
        type=int,
        default=1,
        help=f"Number of built-in benign smoke prompts to run (1-{len(CANARY_PROMPTS)}).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Completion token cap for each smoke request.",
    )
    p.add_argument(
        "--retry-failed-from-report",
        type=Path,
        default=None,
        help="Run only cells that failed in a previous canary_report.json.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write the JSON+MD reports. Defaults to outputs/smoke_canary/<utc_ts>/.",
    )
    args = p.parse_args(argv)

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO / "outputs" / "smoke_canary" / ts

    subjects_filter = None
    if args.subjects:
        subjects_filter = [s.strip() for s in args.subjects.split(",") if s.strip()]
    cell_filter: Optional[set[Tuple[str, str, str]]] = None
    prompt_count = args.prompt_count
    if args.retry_failed_from_report is not None:
        previous = json.loads(args.retry_failed_from_report.read_text(encoding="utf-8"))
        cell_filter = {
            (
                str(c.get("subject_id") or ""),
                str(c.get("prompt_id") or "smoke_prompt_01"),
                str(c.get("intervention") or ""),
            )
            for c in previous.get("cells", [])
            if c.get("status") not in ("passed", "passed_dry_run")
        }
        cell_filter = {c for c in cell_filter if c[0] and c[1] and c[2]}
        if not cell_filter:
            raise SystemExit("retry report has no failed cells")
        if subjects_filter is None:
            subjects_filter = sorted({sid for sid, _prompt_id, _interv in cell_filter})
        prompt_indices = []
        for _sid, prompt_id, _interv in cell_filter:
            try:
                prompt_indices.append(int(prompt_id.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                pass
        if prompt_indices:
            prompt_count = max(prompt_count, max(prompt_indices))

    rc, report = run_canary(
        live=args.live,
        subjects_filter=subjects_filter,
        out_dir=out_dir,
        prompt_count=prompt_count,
        max_tokens=args.max_tokens,
        cell_filter=cell_filter,
    )
    summary = (
        f"smoke_canary: {report.get('n_passed', 0)}/{report.get('n_cells', 0)} passed "
        f"(live={report.get('live', False)})"
    )
    if rc == 0:
        print(summary, flush=True)
    else:
        print(summary, file=sys.stderr, flush=True)
        for c in report.get("cells", []):
            if c["status"] not in ("passed", "passed_dry_run"):
                print(f"  FAIL {c['subject_id']}/{c['intervention']}: {c['reason']}", file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
