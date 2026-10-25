def to_native_body(openai_body, entry):
    """
    Google Gemini OpenAI-compat layer is mostly drop-in.
    """
    return openai_body


def to_native_batch_row(custom_id, openai_body, entry):
    """
    Google native batch format not specified in prompt, but roster says
    supports_batch: false for gemini-3-pro-preview.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1beta/openai/chat/completions",
        "body": openai_body
    }
