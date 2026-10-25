# Direct Google transport contract

POST fixed HTTPS endpoint using external GEMINI_API_KEY. Request model gemini-3.5-flash; exact frozen messages/tools/tool_choice and decoding. Strip only documented transport-only fields before request hashing. No server-side tools executed. Accepted returned model aliases restricted to preflight evidence. Disabled arms must retain identical schemas and tool_choice none. Never retry an HTTP200 completion. Keep native Google error evidence. Hash raw bytes, retain LF-safe JSONL, prevent duplicate IDs and ambiguous replay. Native normalized record uses original request_row plus provider_result response status/body. Runtime endpoint and synchronous mode are explicit.

Tests: strict request identity and allowed fields; only arm manipulation differs; model drift/native-disabled calls fail; durable reserve/settle rejects ambiguity/over-budget; duplicate resume fails; Unicode separators survive; deterministic row inventory integration.
