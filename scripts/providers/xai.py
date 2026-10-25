def to_native_body(openai_body, entry):
    """
    xAI Grok is OpenAI-compat.
    """
    return openai_body


def to_native_batch_row(custom_id, openai_body, entry):
    """
    xAI native batch uses the HTTP file-batch transport. The generic provider
    row remains OpenAI-shaped; NativeBatchSubmitter serializes it to JSONL and
    submits it through /v1/files and /v1/batches.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": openai_body
    }
