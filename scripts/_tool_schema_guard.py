"""Validators for tool schemas and dispatch bodies.

Enforces provider-portable shape so the same tool inventory can ship to
Anthropic, Google, OpenAI, DeepSeek, Moonshot, and OpenRouter without
asymmetric provider-side rejections that bias the corpus.

Two structural checks (catch synthesizer regressions):
  1. NO top-level ``oneOf`` / ``allOf`` / ``anyOf`` in any tool's
     ``parameters`` / ``input_schema``. Anthropic's input_schema validator
     rejects these. Move the constraint into the description string or
     into property-level keywords.
  2. ``required`` MUST be a subset of ``properties.keys()``. Google's
     GenerateContent strict validator rejects orphan ``required`` entries.

One transport check (catch route-asymmetric body fields):
  3. ``body`` keys ``provider``, ``plugins``, ``transforms``, ``models``
     are OpenRouter-only. They MUST NOT be baked into stored input batch
     rows; they are added at OR-dispatch time and stripped before any
     other route. ``strip_route_only_fields(body)`` returns a body safe
     for native / local-vLLM / MLX dispatch.
"""
from __future__ import annotations

from typing import Any, Iterator


OPENROUTER_ONLY_BODY_KEYS = ("provider", "plugins", "transforms", "models")
DISALLOWED_TOP_LEVEL_KEYS = ("oneOf", "allOf", "anyOf")


class ToolSchemaError(ValueError):
    """Raised when a tool schema or dispatch body violates portability rules."""


def _walk_tools(node: Any) -> Iterator[dict]:
    """Yield every tool-like dict reachable from ``node``."""
    if isinstance(node, dict):
        if "name" in node and ("parameters" in node or "input_schema" in node):
            yield node
        for v in node.values():
            yield from _walk_tools(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_tools(v)


def validate_tool(tool: dict, *, source: str = "<unknown>", tool_index: int | None = None) -> None:
    """Raise ``ToolSchemaError`` if ``tool`` would be rejected by any
    provider in the roster. Source is included in the message for triage."""
    params = tool.get("parameters") or tool.get("input_schema") or {}
    if not isinstance(params, dict):
        return
    name = tool.get("name")
    locator = f"{source}::tool[{tool_index}]({name!r})" if tool_index is not None else f"{source}::tool({name!r})"

    for kw in DISALLOWED_TOP_LEVEL_KEYS:
        if kw in params:
            raise ToolSchemaError(
                f"{locator}: top-level {kw!r} in parameters is rejected by Anthropic; "
                "move the constraint into the description string or into property-level keywords."
            )

    props = set((params.get("properties") or {}).keys())
    req = set(params.get("required") or [])
    orphan = req - props
    if orphan:
        raise ToolSchemaError(
            f"{locator}: required references undeclared properties {sorted(orphan)} — "
            "rejected by Google's GenerateContent strict validator. "
            "Either add the missing properties or prune required."
        )


def validate_tools_in_object(obj: Any, *, source: str = "<unknown>") -> None:
    """Walk ``obj`` and validate every tool-like dict inside."""
    for index, tool in enumerate(_walk_tools(obj)):
        validate_tool(tool, source=source, tool_index=index)


def strip_route_only_fields(body: dict) -> dict:
    """Return ``body`` mutated in place with OR-only fields removed.

    Safe to call before NATIVE / local / vLLM dispatch. Caller adds
    OR-specific fields back after this strip when dispatching via
    OpenRouter.
    """
    for key in OPENROUTER_ONLY_BODY_KEYS:
        body.pop(key, None)
    return body


def assert_body_clean_of_route_only_fields(body: dict, *, source: str = "<unknown>") -> None:
    """Raise if a stored input batch row's body still carries OR-only
    fields. Such fields must be added at dispatch, not baked in at
    synthesis, otherwise downstream native/local routes break."""
    leaks = [k for k in OPENROUTER_ONLY_BODY_KEYS if k in body]
    if leaks:
        raise ToolSchemaError(
            f"{source}: body carries OpenRouter-only fields {leaks} — these must be "
            "added at dispatch time only, not baked into stored batch rows."
        )


def validate_batch_row(row: dict, *, source: str = "<unknown>") -> None:
    """Validate one batch input row end-to-end: body shape + every tool."""
    body = row.get("body") or {}
    cid = row.get("custom_id") or "<unknown_custom_id>"
    src = f"{source}::row[{cid}]"
    assert_body_clean_of_route_only_fields(body, source=src)
    tools = body.get("tools") or []
    for index, t in enumerate(tools):
        # OpenAI shape: {type, function: {name, parameters}} OR Anthropic {name, input_schema}
        if isinstance(t, dict) and "function" in t and isinstance(t["function"], dict):
            validate_tool(t["function"], source=src, tool_index=index)
        else:
            validate_tool(t, source=src, tool_index=index)


def validate_batch_file(path, *, source: str | None = None) -> int:
    """Validate every row in a JSONL batch file. Returns row count.

    Raises ``ToolSchemaError`` on the first violation (fail loud)."""
    import json
    from pathlib import Path
    p = Path(path)
    src = source or str(p)
    n = 0
    with p.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ToolSchemaError(f"{src}:line {line_no}: invalid JSON: {exc}") from exc
            validate_batch_row(row, source=f"{src}:line{line_no}")
            n += 1
    return n
