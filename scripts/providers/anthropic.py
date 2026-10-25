ANTHROPIC_DEFAULT_MAX_TOKENS = 4096

# Claude reasoning generations (Opus 4.x, Sonnet 4.x and beyond) deprecate
# the ``temperature`` and ``top_p`` decoding overrides on /v1/messages and
# return 400 invalid_request_error when either is sent with a non-default
# value. Strip both at the adapter boundary for matching slugs so the
# canonical request body can stay uniform across families.
_REASONING_SLUG_PREFIXES = (
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-haiku-4",
)


def _is_reasoning_slug(slug):
    if not isinstance(slug, str):
        return False
    return any(slug.startswith(p) for p in _REASONING_SLUG_PREFIXES)


def to_native_body(openai_body, entry):
    """
    Translates OpenAI-style body to Anthropic /v1/messages format.
    - Extracts role: system to top-level system parameter.
    - Maps messages to role: user|assistant.
    - Maps tools to Anthropic format (name, description, input_schema).
    - Maps tool_choice to {type: auto|none|tool}.
    """
    messages = []
    system_prompt = ""
    for m in openai_body.get("messages", []):
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            messages.append({"role": m["role"], "content": m["content"]})

    native_body = {
        "model": openai_body["model"],
        "messages": messages,
        "max_tokens": openai_body.get(
            "max_tokens",
            (entry.get("native") or {}).get("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS),
        ),
    }
    if system_prompt:
        native_body["system"] = system_prompt

    if not _is_reasoning_slug(openai_body.get("model")):
        if "temperature" in openai_body:
            native_body["temperature"] = openai_body["temperature"]
        if "top_p" in openai_body:
            native_body["top_p"] = openai_body["top_p"]

    if "tools" in openai_body:
        anthropic_tools = []
        for t in openai_body["tools"]:
            # t is {"type": "function", "function": {...}}
            func = t["function"]
            anthropic_tools.append({
                "name": func["name"],
                "description": func["description"],
                "input_schema": func["parameters"]
            })
        native_body["tools"] = anthropic_tools

        if "tool_choice" in openai_body:
            otc = openai_body["tool_choice"]
            if otc == "auto":
                native_body["tool_choice"] = {"type": "auto"}
            elif otc == "none":
                native_body["tool_choice"] = {"type": "none"}
            elif isinstance(otc, dict) and otc.get("type") == "function":
                native_body["tool_choice"] = {
                    "type": "tool",
                    "name": otc["function"]["name"]
                }
            else:
                # Default to auto if we don't recognize it
                native_body["tool_choice"] = {"type": "auto"}

    return native_body


ANTHROPIC_BATCH_CUSTOM_ID_MAX = 64


def short_custom_id(custom_id: str) -> str:
    """Anthropic /v1/messages/batches enforces 64-char limit on custom_id.
    Runner format routinely exceeds this. Hash to first 40 chars of sha1
    hex (collision-resistant) so each row stays unique. Callers must
    persist a {short -> original} reverse map for retrieval."""
    import hashlib
    if len(custom_id) <= ANTHROPIC_BATCH_CUSTOM_ID_MAX:
        return custom_id
    return hashlib.sha1(custom_id.encode("utf-8")).hexdigest()[:40]


def to_native_batch_row(custom_id, openai_body, entry):
    """
    Anthropic batch format: {custom_id, params: {single_request_body}}
    """
    return {
        "custom_id": short_custom_id(custom_id),
        "params": to_native_body(openai_body, entry)
    }


def from_native_response(native_response):
    """Translate an Anthropic ``/v1/messages`` response to OpenAI shape.

    The Anthropic Messages API returns a response with ``content`` as a
    list of typed blocks (``{"type": "text", ...}`` or
    ``{"type": "tool_use", ...}``) and ``stop_reason`` such as
    ``"tool_use"`` or ``"end_turn"``. The downstream pipeline
    (``extractor/extract.py::subject_response_from_response``,
    ``scripts/grade_subject_responses.py::has_structured_tool_calls`` and
    ``structured_tool_calls_from_response``) all read the OpenAI
    ``choices[0].message.{content,tool_calls}`` shape, so without this
    shim every Anthropic native row gets normalised to ``""`` and routed
    to ``Unresolved(empty_response)`` even when the model returned
    legitimate refusal text plus a benign-twin tool_use. Mirroring into
    OpenAI shape at the dispatch boundary keeps all downstream code
    uniform and preserves the original Anthropic fields for audit.

    Verified 2026-04-30 smoke test: Opus 4.7 returned content=[text+tool_use]
    with stop_reason="tool_use"; orchestrator silently mislabelled the row
    as ``Unresolved(empty_response)`` because the OpenAI-only normalisers
    returned empty.
    """
    if not isinstance(native_response, dict):
        return native_response

    content = native_response.get("content")
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    if isinstance(content, list):
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                value = block.get("text")
                if isinstance(value, str):
                    text_parts.append(value)
            elif block_type == "tool_use":
                name = block.get("name")
                if not isinstance(name, str):
                    continue
                tool_input = block.get("input")
                import json as _json_anthropic_shim
                try:
                    arguments_str = _json_anthropic_shim.dumps(
                        tool_input if isinstance(tool_input, dict) else {},
                        sort_keys=True,
                    )
                except Exception:
                    arguments_str = "{}"
                call_id = block.get("id") or f"toolu_{index}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments_str,
                        },
                    }
                )

    stop_reason = native_response.get("stop_reason")
    finish_reason_map = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_reason_map.get(stop_reason, stop_reason)

    message: dict = {
        "role": native_response.get("role", "assistant"),
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not message["content"]:
            # Mirror the OpenAI/Qwen tool_calls convention: content=null when
            # only a structured tool call was emitted. The extractor's
            # ``subject_response_from_response`` knows how to render that.
            message["content"] = None

    usage_native = native_response.get("usage") or {}
    usage_oai = {
        "prompt_tokens": usage_native.get("input_tokens"),
        "completion_tokens": usage_native.get("output_tokens"),
    }
    if usage_oai["prompt_tokens"] is not None and usage_oai["completion_tokens"] is not None:
        usage_oai["total_tokens"] = (
            usage_oai["prompt_tokens"] + usage_oai["completion_tokens"]
        )

    mirrored = dict(native_response)
    mirrored["choices"] = [
        {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }
    ]
    if usage_oai.get("prompt_tokens") is not None:
        mirrored.setdefault("usage", {})
        if isinstance(mirrored["usage"], dict):
            mirrored["usage"].setdefault("prompt_tokens", usage_oai["prompt_tokens"])
            mirrored["usage"].setdefault(
                "completion_tokens", usage_oai["completion_tokens"]
            )
            if usage_oai.get("total_tokens") is not None:
                mirrored["usage"].setdefault("total_tokens", usage_oai["total_tokens"])
    return mirrored
