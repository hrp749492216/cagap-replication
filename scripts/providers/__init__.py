from . import anthropic, openai, google, xai, deepseek, dashscope, cohere

# Mapping from roster 'family' to adapter module
ADAPTERS = {
    "anthropic": anthropic,
    "openai": openai,
    "google": google,
    "xai": xai,
    "deepseek": deepseek,
    "alibaba": dashscope,
    # The following families use OpenAI-compatible request/batch shapes
    "moonshot": openai,
    "zhipu": openai,
    "mistral": openai,
    "meta": openai,
    "minimax": openai,
    "cohere": cohere,
}

def get_adapter(family):
    return ADAPTERS.get(family)
