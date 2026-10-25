import json
from pathlib import Path


COMPATIBILITY_FILE = Path("config/provider_compatibility.json")
ROSTER_FILE = Path("config/roster.json")


def load_compatibility():
    with open(COMPATIBILITY_FILE, "r") as f:
        return json.load(f)


def load_roster():
    with open(ROSTER_FILE, "r") as f:
        return json.load(f)


def active_api_roster_entries():
    roster = load_roster()
    for section in ("subjects", "judges", "extractor"):
        for model_key, entry in roster[section].items():
            if entry.get("native") is not None or entry.get("openrouter") is not None:
                yield section, model_key, entry


def test_provider_compatibility_file_exists():
    assert COMPATIBILITY_FILE.exists()


def test_main_tool_condition_requires_structured_tool_calls():
    data = load_compatibility()

    for record in data["records"]:
        if record["main_tool_condition_allowed"]:
            assert record["structured_tool_calls"], record["decision"]


def test_openrouter_allowed_routes_are_pinned_no_fallbacks():
    """OR routes must have a pinned provider and disable fallbacks. The
    earlier ``require_parameters: True`` requirement was dropped in the
    runner because OR's per-provider parameter-support metadata is
    incomplete and rejects valid bodies for some providers (Anthropic,
    OpenAI) even when the provider's native API accepts the same body.
    Verified against Pilot-1 native-batch successes for those subjects."""
    data = load_compatibility()

    for record in data["records"]:
        if record["provider"] != "openrouter" or not record["main_tool_condition_allowed"]:
            continue

        route = record["provider_route"]
        assert route["only"], record["decision"]
        assert route["allow_fallbacks"] is False, record["decision"]


def test_active_api_roster_has_allowed_provider_compatibility_records():
    data = load_compatibility()
    allowed_records = {
        record["model_key"]: record
        for record in data["records"]
        if record["main_tool_condition_allowed"]
    }

    missing = [
        model_key
        for _, model_key, _ in active_api_roster_entries()
        if model_key not in allowed_records
    ]

    assert missing == []


def test_openrouter_compatibility_routes_match_roster_slugs():
    data = load_compatibility()
    openrouter_slugs = {
        model_key: entry["openrouter"]["slug"]
        for _, model_key, entry in active_api_roster_entries()
        if entry.get("openrouter") is not None
    }
    allowed_openrouter_records = {
        record["model_key"]: record
        for record in data["records"]
        if record["provider"] == "openrouter"
        and record["main_tool_condition_allowed"]
    }

    missing = [
        model_key
        for model_key in openrouter_slugs
        if model_key not in allowed_openrouter_records
    ]

    assert missing == []

    for record in data["records"]:
        if record["provider"] != "openrouter" or not record["main_tool_condition_allowed"]:
            continue

        assert record["model"] == openrouter_slugs[record["model_key"]]
