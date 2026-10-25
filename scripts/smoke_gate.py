"""Shared smoke-gate verdict helpers for local/self-hosted serving preflights."""

from __future__ import annotations


def evaluate_correctness_smoke_gate(
    records: list[dict],
    *,
    cells_total: int,
    mean_seconds_per_cell: float | None,
    baseline_s_per_cell: float | None = None,
    parser_threshold: float = 0.80,
) -> dict:
    """Return a correctness-only smoke verdict.

    Section 13.H smoke validation is a functional gate: the server must respond
    and TOOL-mode rows must populate structured tool_calls. Latency is retained
    for operational planning, but slow correct runs are not failed.
    """

    ok = [r for r in records if r.get("http_status") == 200]
    http_pass = len(ok) == cells_total
    tool_cells = [r for r in ok if r.get("mode") == "TOOL"]
    tool_calls_populated = sum(1 for r in tool_cells if r.get("has_tool_calls"))
    parser_rate = tool_calls_populated / len(tool_cells) if tool_cells else None
    parser_pass = (not tool_cells) or (parser_rate is not None and parser_rate >= parser_threshold)

    if http_pass and parser_pass:
        verdict = "PASS"
    elif not ok:
        verdict = "FAIL"
    elif http_pass and not parser_pass:
        verdict = "PASS_HTTP_FAIL_PARSER"
    else:
        verdict = "PARTIAL"

    baseline = baseline_s_per_cell or 0
    latency_ratio = None
    if mean_seconds_per_cell is not None and baseline > 0:
        latency_ratio = round(mean_seconds_per_cell / baseline, 2)

    return {
        "verdict": verdict,
        "gate_evaluation": {
            "http_pass": http_pass,
            "parser_pass": parser_pass,
            "tool_cells_total": len(tool_cells),
            "tool_calls_populated": tool_calls_populated,
            "parser_threshold": parser_threshold,
            "parser_observed_rate": round(parser_rate, 3) if parser_rate is not None else None,
            "latency_policy": "descriptive_only_not_gating",
            "latency_mean_s_per_cell": round(mean_seconds_per_cell, 3)
            if mean_seconds_per_cell is not None
            else None,
            "latency_baseline_s_per_cell": baseline_s_per_cell,
            "latency_ratio_vs_baseline": latency_ratio,
        },
    }
