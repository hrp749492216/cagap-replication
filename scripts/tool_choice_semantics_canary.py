"""Provider/adapter preflight for TOOL vs DISABLED semantics.

This is a tiny, benign canary for the OSF causal-contrast requirement. It does
not use main-study prompts. For each selected subject route, it sends or builds
three requests:

  NOTOOLS   -- no tools and no tool_choice
  DISABLED  -- same canary tool, tool_choice="none"
  TOOL      -- same canary tool, tool_choice="auto"

The report archives canonical request bodies, provider-serialized request
bodies, response metadata where available, input-token deltas, and a route-level
classification. The main diagnostic is whether DISABLED preserves the tool
schema at the adapter/transport layer and whether provider token accounting
suggests the schema was included rather than hidden like NOTOOLS.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.runtime_provenance import safe_response_headers  # noqa: E402
from scripts.run_synthetic_prompts import (  # noqa: E402
    Config,
    _ROSTER,
    _disable_openrouter_context_compression,
    _dispatch_with_retry,
    _model_entries,
    _native_headers,
    decoding_for,
    redact_reasoning_fields,
    select_route,
)
from scripts.smoke_canary import CANARY_PROMPT, CANARY_TOOL  # noqa: E402


INTERVENTIONS = ("NOTOOLS", "DISABLED", "TOOL")
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_TOKENS = 64


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(_json_dumps(obj).encode("utf-8")).hexdigest()


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _slug_for_route(route: str, route_cfg: dict[str, Any]) -> str:
    if route_cfg.get("slug"):
        return str(route_cfg["slug"])
    for key in ("served_model_path", "hf_repo", "local_dir", "model_id"):
        if route_cfg.get(key):
            return str(route_cfg[key])
    raise ValueError(f"route {route!r} has no model slug or local model identifier")


def _canary_body(subject_id: str, slug: str, intervention: str, max_tokens: int) -> dict[str, Any]:
    decoding = dict(decoding_for(subject_id)) if subject_id in Config.SUBJECTS else {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    if "max_tokens" in decoding:
        decoding["max_tokens"] = min(int(decoding["max_tokens"]), max_tokens)
    body: dict[str, Any] = {
        "model": slug,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": CANARY_PROMPT},
        ],
    }
    body.update(decoding)
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
    raise ValueError(f"unknown intervention {intervention!r}")


def _openrouter_body(canonical_body: dict[str, Any], route_cfg: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(canonical_body)
    body["model"] = str(route_cfg["slug"])
    body["provider"] = {
        "only": [route_cfg["pinned_provider"]],
        "allow_fallbacks": False,
        "data_collection": "deny",
    }
    _disable_openrouter_context_compression(body)
    return body


def _native_body(
    canonical_body: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    from scripts.providers import get_adapter

    adapter = get_adapter(entry.get("family"))
    if adapter is None:
        return deepcopy(canonical_body)
    return adapter.to_native_body(deepcopy(canonical_body), entry)


def prepare_route_cell(
    *,
    subject_id: str,
    entry: dict[str, Any],
    route: str,
    route_cfg: dict[str, Any],
    intervention: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    """Build canonical and provider-serialized request bodies for one cell."""
    slug = _slug_for_route(route, route_cfg)
    canonical = _canary_body(subject_id, slug, intervention, max_tokens)
    if route == "openrouter":
        serialized = _openrouter_body(canonical, route_cfg)
        endpoint = OPENROUTER_CHAT_URL
    elif route == "native":
        serialized = _native_body(canonical, entry)
        endpoint = f"{route_cfg.get('endpoint', '').rstrip('/')}{route_cfg.get('url_path', '')}"
    elif route == "local":
        serialized = deepcopy(canonical)
        endpoint = str(
            route_cfg.get("endpoint")
            or os.environ.get("CAGAP_VLLM_ENDPOINT")
            or os.environ.get("CAGAP_MLX_ENDPOINT")
            or ""
        )
    else:
        raise ValueError(f"unknown route {route!r}")

    return {
        "subject_id": subject_id,
        "display_name": entry.get("display_name"),
        "family": entry.get("family"),
        "route": route,
        "provider": route_cfg.get("pinned_provider") if route == "openrouter" else entry.get("family"),
        "endpoint": endpoint,
        "intervention": intervention,
        "canonical_body": canonical,
        "serialized_body": serialized,
        "canonical_request_sha256": _sha256_json(canonical),
        "serialized_request_sha256": _sha256_json(serialized),
        "canonical_body_keys": sorted(canonical.keys()),
        "serialized_body_keys": sorted(serialized.keys()),
        "tool_schema_sha256": _sha256_json(canonical.get("tools")) if canonical.get("tools") else None,
    }


def prompt_input_tokens(response: dict[str, Any] | None) -> int | None:
    """Normalize provider input-token accounting across common response shapes."""
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "input_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                return value
        tokens = usage.get("tokens")
        if isinstance(tokens, dict):
            value = tokens.get("input_tokens") or tokens.get("prompt_tokens")
            if isinstance(value, int):
                return value
    metadata = response.get("usageMetadata")
    if isinstance(metadata, dict):
        value = metadata.get("promptTokenCount") or metadata.get("inputTokenCount")
        if isinstance(value, int):
            return value
    return None


def tool_call_count(response: dict[str, Any] | None) -> int:
    if not isinstance(response, dict):
        return 0
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                return len(calls)
    content = response.get("content")
    if isinstance(content, list):
        return sum(
            1
            for block in content
            if isinstance(block, dict) and block.get("type") in {"tool_use", "tool_call"}
        )
    return 0


def _has_disabled_tools(cell: dict[str, Any]) -> bool:
    body = cell.get("serialized_body")
    if not isinstance(body, dict):
        return True
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def _has_disabled_none(cell: dict[str, Any]) -> bool:
    body = cell.get("serialized_body")
    if not isinstance(body, dict):
        return True
    tool_choice = body.get("tool_choice")
    if tool_choice == "none":
        return True
    return isinstance(tool_choice, dict) and tool_choice.get("type") == "none"


def classify_route(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Classify one route from NOTOOLS/DISABLED/TOOL cell records."""
    reasons: list[str] = []
    warnings: list[str] = []
    required = set(INTERVENTIONS)
    missing = sorted(required.difference(cells))
    if missing:
        return {"status": "fail", "reasons": [f"missing cells: {', '.join(missing)}"]}

    route_unavailable_statuses = {"skipped_no_auth", "failed_route_unavailable"}
    if all(cells[intervention].get("status") in route_unavailable_statuses for intervention in INTERVENTIONS):
        http_statuses = sorted(
            {
                int(cells[intervention]["http_status"])
                for intervention in INTERVENTIONS
                if isinstance(cells[intervention].get("http_status"), int)
            }
        )
        status_text = (
            f"HTTP {', '.join(str(value) for value in http_statuses)}"
            if http_statuses
            else "credentials/endpoint unavailable"
        )
        return {
            "status": "route_unavailable",
            "reasons": [f"route not exercised: {status_text}"],
            "input_tokens": {intervention: None for intervention in INTERVENTIONS},
            "input_token_deltas": {
                "DISABLED_minus_NOTOOLS": None,
                "TOOL_minus_DISABLED": None,
            },
            "disabled_tool_call_count": 0,
            "tool_choice_none_interpretation": "not_exercised_route_unavailable",
        }

    for intervention in INTERVENTIONS:
        status = cells[intervention].get("status")
        if status not in {"passed", "passed_dry_run"}:
            reasons.append(f"{intervention} cell status={status}")

    disabled = cells["DISABLED"]
    if not _has_disabled_tools(disabled):
        reasons.append("DISABLED serialized request did not include tools")
    if not _has_disabled_none(disabled):
        reasons.append("DISABLED serialized request did not preserve tool_choice=none")
    if tool_call_count(disabled.get("response")) > 0:
        reasons.append("DISABLED returned structured tool call despite tool_choice=none")

    tokens = {
        intervention: prompt_input_tokens(cells[intervention].get("response"))
        for intervention in INTERVENTIONS
    }
    deltas: dict[str, int | None] = {
        "DISABLED_minus_NOTOOLS": None,
        "TOOL_minus_DISABLED": None,
    }
    if all(tokens[value] is not None for value in INTERVENTIONS):
        assert tokens["DISABLED"] is not None
        assert tokens["NOTOOLS"] is not None
        assert tokens["TOOL"] is not None
        deltas["DISABLED_minus_NOTOOLS"] = tokens["DISABLED"] - tokens["NOTOOLS"]
        deltas["TOOL_minus_DISABLED"] = tokens["TOOL"] - tokens["DISABLED"]
        if deltas["DISABLED_minus_NOTOOLS"] <= 0:
            reasons.append(
                "DISABLED input tokens did not exceed NOTOOLS; provider may hide schema"
            )
        schema_delta = deltas["DISABLED_minus_NOTOOLS"]
        tool_delta = abs(deltas["TOOL_minus_DISABLED"])
        if schema_delta > 0 and tool_delta > max(8, int(schema_delta * 0.5)):
            warnings.append(
                "TOOL and DISABLED input-token counts diverged more than expected"
            )
    else:
        warnings.append("token usage unavailable for one or more cells")

    if reasons:
        status = "fail"
        interpretation = "semantically_non_equivalent_or_route_broken"
    elif warnings:
        status = "warning"
        interpretation = "schema_submitted_but_visibility_not_fully_inferable"
    else:
        status = "pass"
        interpretation = "disabled_schema_visibility_supported_by_request_and_token_delta"
    return {
        "status": status,
        "reasons": reasons + warnings,
        "input_tokens": tokens,
        "input_token_deltas": deltas,
        "disabled_tool_call_count": tool_call_count(disabled.get("response")),
        "tool_choice_none_interpretation": interpretation,
    }


def _auth_available(route: str, route_cfg: dict[str, Any], entry: dict[str, Any]) -> bool:
    if route == "openrouter":
        return bool(os.environ.get(Config.OPENROUTER_AUTH_ENV))
    if route == "native":
        auth_env = str(route_cfg.get("auth_env") or "")
        if auth_env and os.environ.get(auth_env):
            return True
        return entry.get("family") == "xai" and bool(os.environ.get("XAI_API_KEY"))
    if route == "local":
        return bool(
            route_cfg.get("endpoint")
            or os.environ.get("CAGAP_VLLM_ENDPOINT")
            or os.environ.get("CAGAP_MLX_ENDPOINT")
        )
    return False


def _dispatch(cell: dict[str, Any], entry: dict[str, Any], route_cfg: dict[str, Any]) -> dict[str, Any]:
    import requests
    from scripts.providers import get_adapter

    route = str(cell["route"])
    body = deepcopy(cell["serialized_body"])
    headers: dict[str, str]
    url: str
    if route == "openrouter":
        api_key = os.environ.get(Config.OPENROUTER_AUTH_ENV, "")
        if not api_key:
            raise RuntimeError(f"{Config.OPENROUTER_AUTH_ENV} not set")
        url = OPENROUTER_CHAT_URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    elif route == "native":
        family = str(entry.get("family") or "")
        url = f"{route_cfg['endpoint'].rstrip('/')}{route_cfg['url_path']}"
        headers = _native_headers(route_cfg, family)
    elif route == "local":
        endpoint = str(
            route_cfg.get("endpoint")
            or os.environ.get("CAGAP_VLLM_ENDPOINT")
            or os.environ.get("CAGAP_MLX_ENDPOINT")
            or ""
        )
        if not endpoint:
            raise RuntimeError("local endpoint unavailable")
        url = endpoint.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        local_key = os.environ.get("VLLM_CLOUD_AUTH") or os.environ.get("LOCAL_OPENAI_API_KEY")
        if local_key:
            headers["Authorization"] = f"Bearer {local_key}"
    else:
        raise RuntimeError(f"unknown route {route!r}")

    session = requests.Session()
    started = time.perf_counter()
    try:
        response = _dispatch_with_retry(
            session=session,
            url=url,
            headers=headers,
            body=body,
            timeout=120,
            model_id=str(cell["subject_id"]),
        )
        latency_sec = time.perf_counter() - started
        payload = response.json()
        adapter = get_adapter(entry.get("family"))
        if route == "native" and adapter is not None and hasattr(adapter, "from_native_response"):
            payload = adapter.from_native_response(payload)
        return {
            "status": "passed",
            "http_status": response.status_code,
            "latency_sec": round(latency_sec, 3),
            "response": redact_reasoning_fields(payload),
            "response_headers": safe_response_headers(dict(response.headers)),
        }
    finally:
        session.close()


def _routes_for_entry(
    subject_id: str,
    entry: dict[str, Any],
    route_scope: str,
) -> list[tuple[str, dict[str, Any]]]:
    if route_scope == "active":
        route, cfg = select_route(entry)
        return [(route, cfg)]
    routes: list[tuple[str, dict[str, Any]]] = []
    if route_scope in {"all-configured", "native"} and isinstance(entry.get("native"), dict):
        routes.append(("native", entry["native"]))
    if route_scope in {"all-configured", "openrouter"} and isinstance(entry.get("openrouter"), dict):
        routes.append(("openrouter", entry["openrouter"]))
    if route_scope in {"all-configured", "local"} and (
        entry.get("tier") == "local_self_hosted" or isinstance(entry.get("local_config"), dict)
    ):
        routes.append(("local", entry["local_config"]))
    if not routes and route_scope != "all-configured":
        return []
    if route_scope == "all-configured" and not routes:
        route, cfg = select_route(entry)
        routes.append((route, cfg))
    return routes


def run_semantics_canary(
    *,
    live: bool,
    subjects_filter: list[str] | None = None,
    route_scope: str = "all-configured",
    out_dir: Path | None = None,
    require_live: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[int, dict[str, Any]]:
    subjects = _model_entries(_ROSTER.get("subjects") or {})
    if subjects_filter:
        keep = set(subjects_filter)
        subjects = {key: value for key, value in subjects.items() if key in keep}
        if not subjects:
            return 2, {"error": f"no subjects matched filter {subjects_filter}"}

    route_reports: list[dict[str, Any]] = []
    for subject_id, entry in sorted(subjects.items()):
        for route, route_cfg in _routes_for_entry(subject_id, entry, route_scope):
            route_key = f"{subject_id}::{route}"
            cells: dict[str, dict[str, Any]] = {}
            for intervention in INTERVENTIONS:
                cell = prepare_route_cell(
                    subject_id=subject_id,
                    entry=entry,
                    route=route,
                    route_cfg=route_cfg,
                    intervention=intervention,
                    max_tokens=max_tokens,
                )
                try:
                    if not live:
                        cell["status"] = "passed_dry_run"
                        cell["reason"] = "request construction only"
                    elif not _auth_available(route, route_cfg, entry):
                        cell["status"] = "skipped_no_auth"
                        cell["reason"] = "required API key or local endpoint not available"
                    else:
                        cell.update(_dispatch(cell, entry, route_cfg))
                        cell["reason"] = "ok"
                except Exception as exc:  # noqa: BLE001 - audit artifact should capture failures.
                    response = getattr(exc, "response", None)
                    http_status = getattr(response, "status_code", None)
                    if route == "openrouter" and http_status in {401, 402}:
                        cell["status"] = "failed_route_unavailable"
                    else:
                        cell["status"] = "failed"
                    if isinstance(http_status, int):
                        cell["http_status"] = http_status
                    cell["reason"] = f"{type(exc).__name__}: {exc}"
                    cell["traceback"] = traceback.format_exc(limit=5)
                cells[intervention] = cell

            if not live:
                verdict = {
                    "status": "dry_run_only",
                    "reasons": ["live provider semantics not exercised"],
                    "tool_choice_none_interpretation": "request_construction_only",
                }
            else:
                verdict = classify_route(cells)
                if require_live and verdict.get("status") == "route_unavailable":
                    verdict = {
                        **verdict,
                        "status": "fail",
                        "tool_choice_none_interpretation": "required_route_not_exercised",
                    }
            route_reports.append(
                {
                    "route_key": route_key,
                    "subject_id": subject_id,
                    "display_name": entry.get("display_name"),
                    "route": route,
                    "provider": route_cfg.get("pinned_provider") if route == "openrouter" else entry.get("family"),
                    "slug": _slug_for_route(route, route_cfg),
                    "verdict": verdict,
                    "cells": [cells[intervention] for intervention in INTERVENTIONS],
                }
            )

    summary_counts: dict[str, int] = {}
    for report in route_reports:
        status = str(report["verdict"]["status"])
        summary_counts[status] = summary_counts.get(status, 0) + 1
    failed = summary_counts.get("fail", 0)
    report = {
        "schema_version": "tool_choice_semantics_canary.v1",
        "computed_at_utc": _now_utc(),
        "live": live,
        "route_scope": route_scope,
        "require_live": require_live,
        "max_tokens": max_tokens,
        "n_subjects": len({item["subject_id"] for item in route_reports}),
        "n_routes": len(route_reports),
        "summary_counts": dict(sorted(summary_counts.items())),
        "all_required_passed": failed == 0,
        "routes": route_reports,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tool_choice_semantics_canary_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / "tool_choice_semantics_canary_report.md").write_text(
            _format_markdown(report),
            encoding="utf-8",
        )
    return (0 if failed == 0 else 1), report


def _format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Tool Choice Semantics Canary Report",
        "",
        f"- computed_at_utc: `{report['computed_at_utc']}`",
        f"- live: `{report['live']}`",
        f"- route_scope: `{report['route_scope']}`",
        f"- require_live: `{report['require_live']}`",
        f"- routes: {report['n_routes']}",
        f"- summary_counts: `{json.dumps(report['summary_counts'], sort_keys=True)}`",
        "",
        "| subject | route | provider | verdict | interpretation | reasons |",
        "|---------|-------|----------|---------|----------------|---------|",
    ]
    for route in report["routes"]:
        verdict = route["verdict"]
        reasons = "; ".join(verdict.get("reasons") or [])[:180]
        lines.append(
            "| `{subject}` | `{route}` | `{provider}` | `{status}` | `{interp}` | {reasons} |".format(
                subject=route["subject_id"],
                route=route["route"],
                provider=route.get("provider"),
                status=verdict["status"],
                interp=verdict.get("tool_choice_none_interpretation"),
                reasons=reasons,
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tiny provider/adapter canary for DISABLED tool_choice semantics."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Build/audit request bodies only.")
    group.add_argument("--live", action="store_true", help="Send live canary requests.")
    parser.add_argument("--subjects", default=None, help="Comma-separated subject_id filter.")
    parser.add_argument(
        "--route-scope",
        choices=("all-configured", "active", "native", "openrouter", "local"),
        default="all-configured",
        help="Routes to include.",
    )
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Treat no-auth/no-endpoint route skips as failures.",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    if out_dir is None:
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO / "outputs" / "tool_choice_semantics_canary" / stamp
    subjects_filter = None
    if args.subjects:
        subjects_filter = [part.strip() for part in args.subjects.split(",") if part.strip()]

    rc, report = run_semantics_canary(
        live=args.live,
        subjects_filter=subjects_filter,
        route_scope=args.route_scope,
        out_dir=out_dir,
        require_live=args.require_live,
        max_tokens=args.max_tokens,
    )
    print(
        "tool_choice_semantics_canary: "
        f"routes={report.get('n_routes', 0)} summary={report.get('summary_counts', {})} "
        f"out_dir={out_dir}",
        flush=True,
    )
    if rc != 0:
        for route in report.get("routes", []):
            verdict = route.get("verdict", {})
            if verdict.get("status") == "fail":
                print(
                    f"  FAIL {route.get('route_key')}: {'; '.join(verdict.get('reasons') or [])}",
                    file=sys.stderr,
                    flush=True,
                )
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
