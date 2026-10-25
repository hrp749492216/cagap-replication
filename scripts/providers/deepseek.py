def to_native_body(openai_body, entry):
    """DeepSeek is OpenAI-compat. Native /v1/chat/completions rejects the
    OpenRouter-style vendor-prefixed slug ``deepseek/deepseek-v4-pro`` and
    requires the bare slug ``deepseek-v4-pro``. Rewrite body.model to the
    roster's native.slug when an OR-style prefixed slug is present."""
    body = dict(openai_body) if isinstance(openai_body, dict) else openai_body
    native_cfg = (entry or {}).get("native") or {}
    native_slug = native_cfg.get("slug")
    if isinstance(body.get("model"), str) and "/" in body["model"] and native_slug:
        body["model"] = native_slug
    return body


def to_native_batch_row(custom_id, openai_body, entry):
    """
    DeepSeek supports_batch: false in roster.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": openai_body
    }
