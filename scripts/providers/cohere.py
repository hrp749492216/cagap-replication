"""Cohere v2 Chat native adapter.

The Cohere ``/v2/chat`` endpoint accepts chronological ``messages`` and
returns ``message.content`` as a list of typed blocks. Mirror responses to
OpenAI chat-completions shape so downstream judge parsing remains uniform.
"""


def to_native_body(openai_body, entry):
    body = {
        "model": openai_body["model"],
        "messages": openai_body.get("messages", []),
        "stream": False,
    }
    if "max_tokens" in openai_body:
        body["max_tokens"] = openai_body["max_tokens"]
    if "temperature" in openai_body:
        body["temperature"] = openai_body["temperature"]
    if "top_p" in openai_body:
        body["p"] = openai_body["top_p"]
    return body


def _finish_reason(reason):
    mapping = {
        "COMPLETE": "stop",
        "complete": "stop",
        "MAX_TOKENS": "length",
        "max_tokens": "length",
        "STOP_SEQUENCE": "stop",
        "stop_sequence": "stop",
        "TOOL_CALL": "tool_calls",
        "tool_call": "tool_calls",
        "ERROR": "error",
        "error": "error",
        "TIMEOUT": "error",
        "timeout": "error",
    }
    return mapping.get(reason, str(reason or "stop").lower())


def _text_content(message):
    content = (message or {}).get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _usage(native_response):
    tokens = ((native_response.get("usage") or {}).get("tokens") or {})
    prompt = tokens.get("input_tokens")
    completion = tokens.get("output_tokens")
    result = {}
    if prompt is not None:
        result["prompt_tokens"] = prompt
    if completion is not None:
        result["completion_tokens"] = completion
    if prompt is not None and completion is not None:
        result["total_tokens"] = prompt + completion
    return result


def from_native_response(native_response):
    if not isinstance(native_response, dict):
        return native_response
    message = native_response.get("message") or {}
    mirrored = dict(native_response)
    mirrored.setdefault("choices", [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": _text_content(message),
            },
            "finish_reason": _finish_reason(native_response.get("finish_reason")),
        }
    ])
    usage = _usage(native_response)
    if usage:
        mirrored["usage"] = usage
    return mirrored
