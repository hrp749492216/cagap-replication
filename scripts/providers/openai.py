"""OpenAI native adapter.

The chat-completions surface for the GPT-5.x reasoning family enforces two
rename/drop rules that several OpenAI-compatible deployments do not:

- ``max_tokens`` is rejected with ``unsupported_parameter`` and must be
  renamed to ``max_completion_tokens`` (verified 2026-04-30 smoke test).
- ``temperature`` and ``top_p`` are silently rejected on reasoning models
  (only the default values are accepted) and the provider returns 400
  ``invalid_request_error`` when they are present with non-default values.

We strip both at the adapter boundary rather than complicating the
roster's decoding overrides — keeping ``config/roster.json`` semantically
uniform across providers and isolating provider-specific quirks here.

The list of slugs that need this transform is conservative: it covers
gpt-5.x reasoning models seen in the project roster as of 2026-04-30. If
an OpenAI-family slug is added that does NOT need the transform (e.g. a
chat-only non-reasoning model), append it to ``_CHAT_COMPLETIONS_ONLY_SLUGS``
so the rename is skipped.
"""

_REASONING_SLUG_PREFIXES = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
    "kimi-k2",
)

_CHAT_COMPLETIONS_ONLY_SLUGS: set[str] = set()


def _is_reasoning_slug(slug: str) -> bool:
    if slug in _CHAT_COMPLETIONS_ONLY_SLUGS:
        return False
    candidates = [slug]
    if slug.startswith("openai/"):
        candidates.append(slug.split("/", 1)[1])
    return any(
        candidate.startswith(prefix)
        for candidate in candidates
        for prefix in _REASONING_SLUG_PREFIXES
    )


def to_native_body(openai_body, entry):
    """Apply OpenAI-native decoding-param compat transforms in place.

    The rule set is keyed off the request's ``model`` slug rather than the
    roster ``entry`` so that judges and OpenAI-compat siblings (Moonshot,
    Mistral, MiniMax) which reuse this adapter via ``providers/__init__.py``
    do not have their bodies mutated by the GPT-5.x rules.
    """
    slug = openai_body.get("model")
    if not isinstance(slug, str) or not _is_reasoning_slug(slug):
        return openai_body

    if "max_tokens" in openai_body and "max_completion_tokens" not in openai_body:
        openai_body["max_completion_tokens"] = openai_body.pop("max_tokens")
    else:
        openai_body.pop("max_tokens", None)
    openai_body.pop("temperature", None)
    openai_body.pop("top_p", None)
    return openai_body


def to_native_batch_row(custom_id, openai_body, entry):
    """OpenAI Batch JSONL spec: {custom_id, method, url, body}.

    The body is run through ``to_native_body`` first so reasoning-model
    rows in a Batch JSONL inherit the same compat rename as the online
    surface.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": to_native_body(dict(openai_body), entry),
    }
