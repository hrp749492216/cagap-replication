#!/usr/bin/env python3
"""Frozen canonical request body + smoke test for the GLM-5.1 judge (vLLM route).

FROZEN 2026-05-29 (prospective, before canonical dispatch):
  thinking      = OFF   (chat_template_kwargs.enable_thinking=false)
  reasoning_parser (server) = glm45  (seatbelt; verdict still lands in content)
  tool parsing  = OFF   (text judge; label parsed from message.content)
  streaming     = False
  max_tokens    = 4096
  temperature   = 0.0

This module is the single source of truth for the REQUEST shape the canonical
GLM-5.1 self-hosted (vLLM/B300) judge dispatcher must send. The dispatcher
(command-a-style) imports build_judge_body() so the freeze can't drift.

For NON-vLLM routes the disable mechanism differs (do NOT send chat_template_kwargs):
  - z.ai native  : extra body {"thinking": {"type": "disabled"}}
  - ollama cloud : (pending verification)
See glm_5_1_judge_route_selection.json -> per_route_disable_mechanism.
"""
from __future__ import annotations

GLM51_JUDGE_MAX_TOKENS = 4096
GLM51_JUDGE_TEMPERATURE = 0.0
GLM51_JUDGE_TOP_P = 1.0  # roster decoding.default; glm-5.1 override keeps it (inert at temp 0)


def build_judge_body(judge_prompt: str, *, served_model: str = "glm-5.1") -> dict:
    """The exact OpenAI-compatible body for the vLLM-route GLM-5.1 judge."""
    return {
        "model": served_model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": GLM51_JUDGE_TEMPERATURE,
        "top_p": GLM51_JUDGE_TOP_P,
        "max_tokens": GLM51_JUDGE_MAX_TOKENS,
        "stream": False,
        # THE LOCK: disable thinking at the template layer (not /no_think, not post-strip).
        "chat_template_kwargs": {"enable_thinking": False},
    }


def extract_label_content(message: dict) -> str | None:
    """Verdict text source. vLLM returns the answer in `content`; with thinking
    off `reasoning_content` is null. Belt-and-suspenders: fall back to
    reasoning_content if a future build ever mis-routes (costs nothing)."""
    return message.get("content") or message.get("reasoning_content")


def smoke_check(message: dict) -> None:
    """Frozen pre-dispatch smoke assertions (raises AssertionError on fail).

    Field note: vLLM's OpenAI message exposes `reasoning_content` (the OpenAI
    python client surfaces it as `.reasoning_content`); some docs call it
    `reasoning`. We assert whichever is present is empty.
    """
    content = message.get("content")
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    assert content not in (None, ""), (
        "FAIL: message.content empty -> parser mis-routed verdict to reasoning_content"
    )
    assert reasoning in (None, ""), (
        "WARN/FAIL: reasoning_content populated -> thinking not actually disabled"
    )


if __name__ == "__main__":
    import argparse, json, sys, urllib.request

    ap = argparse.ArgumentParser(description="GLM-5.1 judge canonical-request smoke test")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--served-model", default="glm-5.1")
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()

    body = build_judge_body(
        "Reply with exactly: Label: PASS", served_model=args.served_model
    )
    req = urllib.request.Request(
        f"{args.base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    msg = resp["choices"][0]["message"]
    print("content           =", repr(msg.get("content")))
    print("reasoning_content =", repr(msg.get("reasoning_content", msg.get("reasoning"))))
    print("finish_reason     =", resp["choices"][0].get("finish_reason"))
    try:
        smoke_check(msg)
        assert "Label:" in (msg.get("content") or ""), "FAIL: 'Label:' not in content"
        print("PASS: thinking off, verdict in content, reasoning empty, Label: present")
    except AssertionError as e:
        print(e)
        sys.exit(1)
