def to_native_body(openai_body, entry):
    """
    Alibaba DashScope is OpenAI-compat.
    """
    return openai_body


def to_native_batch_row(custom_id, openai_body, entry):
    """
    DashScope supports_batch: false in roster.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/compatible-mode/v1/chat/completions",
        "body": openai_body
    }
