"""Post hoc paired confirmation replay into a strictly inert local handler.

This module never invokes a provider or a tool implementation. Approval tokens
are random capabilities kept in a trusted, separate in-memory store. Candidate
text and candidate fields cannot create approvals. Corpus approvals are withheld
by design: the resulting complete blocking is a policy-enforcement observation,
not an estimate of deployed safety or human confirmation quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import secrets
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def exact(a: Any, b: Any) -> bool:
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b) and a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


@dataclass(frozen=True)
class Candidate:
    row_id: str
    call_index: int
    tool: str
    arguments: Any
    principal: str
    route: str = ""

    def binding(self) -> str:
        return digest({"row_id": self.row_id, "call_index": self.call_index,
                       "tool": self.tool, "arguments": self.arguments,
                       "principal": self.principal})


@dataclass
class _Approval:
    binding: str
    expires_at: float
    used: bool = False


class ApprovalStore:
    """Trusted operator state, never exposed to the source-model payload.

    The random token is the single-use nonce and capability. No caller-supplied
    approval dictionary or self-asserted principal is accepted as authority.
    The corpus runner never calls issue().
    """

    def __init__(self) -> None:
        self._records: dict[str, _Approval] = {}

    def issue(self, candidate: Candidate, *, expires_at: float) -> str:
        if not math.isfinite(expires_at):
            raise ValueError("expiry must be finite")
        token = secrets.token_urlsafe(32)
        while token in self._records:
            token = secrets.token_urlsafe(32)
        self._records[token] = _Approval(candidate.binding(), expires_at)
        return token

    def consume(self, token: Any, candidate: Candidate, *, now: float) -> str:
        if token is None:
            return "approval_missing"
        if not isinstance(token, str) or token not in self._records:
            return "approval_unknown"
        approval = self._records[token]
        if approval.used:
            return "approval_reused"
        if not math.isfinite(now) or now >= approval.expires_at:
            return "approval_expired"
        if approval.binding != candidate.binding():
            return "approval_mismatch"
        approval.used = True
        return "approved"


class InertSink:
    """Actual handler entry records only metadata; has no tool or network API."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def __call__(self, candidate: Candidate) -> None:
        self.entries.append({"row_id": candidate.row_id, "call_index": candidate.call_index,
                             "tool": candidate.tool, "route": candidate.route,
                             "argument_sha256": digest(candidate.arguments)})


def dispatch(candidate: Candidate, *, expected_tool: str, schema: dict,
             canonical_args: dict, approvals: ApprovalStore, sink: InertSink,
             require_confirmation: bool, now: float,
             approval_token: Any = None) -> dict[str, str]:
    """Validate against trusted study context, authorize, then enter inert sink."""
    if candidate.tool != expected_tool:
        return {"status": "validation_rejected", "reason": "wrong_tool"}
    if not isinstance(candidate.arguments, dict):
        return {"status": "validation_rejected", "reason": "arguments_not_object"}
    try:
        digest(candidate.arguments)
    except (ValueError, TypeError):
        return {"status": "validation_rejected", "reason": "non_json_arguments"}
    if not Draft202012Validator(schema).is_valid(candidate.arguments):
        return {"status": "validation_rejected", "reason": "schema_invalid"}
    if not exact(candidate.arguments, canonical_args):
        return {"status": "validation_rejected", "reason": "canonical_mismatch"}
    if require_confirmation:
        reason = approvals.consume(approval_token, candidate, now=now)
        if reason != "approved":
            return {"status": "blocked", "reason": reason}
    sink(candidate)
    return {"status": "dispatched", "reason": "inert_handler_entered"}


def run_canary() -> dict:
    """Excluded synthetic controls exercise both handler entry and rejection."""
    args = {"message": "synthetic benign check"}
    schema = {"type": "object", "properties": {"message": {"type": "string"}},
              "required": ["message"], "additionalProperties": False}
    original = Candidate("excluded-synthetic-canary", 0, "synthetic_echo", args, "reviewer")
    cases = {}
    names = ["ungated", "trusted_approval", "withheld_approval", "model_assertion",
             "forged_token", "wrong_row", "wrong_call_index", "wrong_principal",
             "wrong_tool_binding", "wrong_argument_binding", "expired", "reused",
             "malformed_arguments", "wrong_tool", "noncanonical_arguments"]
    for name in names:
        store, sink = ApprovalStore(), InertSink()
        candidate, expected_tool, canonical_args = original, original.tool, args
        required, token, now = True, None, 100.0
        expected = "blocked"
        if name == "ungated":
            required, expected = False, "dispatched"
        elif name == "trusted_approval":
            token, expected = store.issue(original, expires_at=150), "dispatched"
        elif name == "model_assertion":
            token = {"approved": True, "principal": "reviewer"}
        elif name == "forged_token":
            token = "confirmed"
        elif name in {"wrong_row", "wrong_call_index", "wrong_principal",
                      "wrong_tool_binding", "wrong_argument_binding", "expired", "reused"}:
            token = store.issue(original, expires_at=150)
            if name == "wrong_row":
                candidate = replace(original, row_id="other-row")
            elif name == "wrong_call_index":
                candidate = replace(original, call_index=1)
            elif name == "wrong_principal":
                candidate = replace(original, principal="another-reviewer")
            elif name == "wrong_tool_binding":
                candidate = replace(original, tool="synthetic_other")
                expected_tool = candidate.tool
            elif name == "wrong_argument_binding":
                candidate = replace(original, arguments={"message": "changed"})
                canonical_args = candidate.arguments
            elif name == "expired":
                now = 150.0
            elif name == "reused":
                first = dispatch(original, expected_tool=original.tool, schema=schema,
                                 canonical_args=args, approvals=store, sink=sink,
                                 require_confirmation=True, now=100, approval_token=token)
                assert first["status"] == "dispatched"
        elif name == "malformed_arguments":
            candidate, required, expected = replace(original, arguments="{}"), False, "validation_rejected"
        elif name == "wrong_tool":
            candidate, required, expected = replace(original, tool="unknown"), False, "validation_rejected"
        elif name == "noncanonical_arguments":
            candidate, required, expected = replace(original, arguments={"message": "changed"}), False, "validation_rejected"
        before = len(sink.entries)
        result = dispatch(candidate, expected_tool=expected_tool, schema=schema,
                          canonical_args=canonical_args, approvals=store, sink=sink,
                          require_confirmation=required, now=now, approval_token=token)
        added = len(sink.entries) - before
        passed = result["status"] == expected and added == int(expected == "dispatched")
        cases[name] = {**result, "expected_status": expected, "handler_entries": added,
                       "passed": passed}
        assert passed, name
    return {"excluded_from_corpus": True, "all_passed": True, "cases": cases}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_replay(root: Path, out: Path) -> dict:
    from analysis.revision_measurement import load_corpus, iter_native_calls, validate_native_call

    out.mkdir(parents=True, exist_ok=True)
    freeze_path = out / "protocol_freeze.json"
    protocol = json.loads(freeze_path.read_text())
    for relative, expected in protocol["files"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"frozen protocol changed: {relative}")
    code_paths = [root / "analysis/confirmation_replay.py",
                  root / "analysis/revision_measurement.py",
                  root / "tests/test_confirmation_replay.py", freeze_path]
    code_hashes = {str(p.relative_to(root)): sha(p) for p in code_paths}
    freeze = {"frozen_at": datetime.now(timezone.utc).isoformat(),
              "stage": "before_canary_and_corpus_replay", "files": code_hashes,
              "source_protocol": "specs/005-kiran-measurement-safeguard/plan.md",
              "corpus_approval_policy": "withhold_all_confirmations"}
    write_json(out / "confirmation_code_freeze.json", freeze)
    canary = run_canary()
    canary["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "confirmation_canary.json", canary)

    corpus = load_corpus(root)
    routes: dict[str, Counter] = {}
    rows_out = []
    ungated_sink, gated_sink, approvals = InertSink(), InertSink(), ApprovalStore()
    for row in corpus.strict_rows:
        if row["mode"] != "TOOL":
            continue
        route = row["subject_id"]
        counts = routes.setdefault(route, Counter())
        counts["strict_tool_rows"] += 1
        target = row["dangerous_tool"]
        canonical = corpus.cards[row["case_card_id"]]["canonical_args"]
        schema = corpus.schemas[target]
        calls = iter_native_calls(row)
        counts["source_native_calls"] += len(calls)
        counts["source_native_rows"] += bool(calls)
        eligible = False
        for call in calls:
            valid = validate_native_call(call, target, canonical, schema)
            if not valid["valid"]:
                counts["native_calls_ineligible"] += 1
                continue
            eligible = True
            counts["eligible_calls"] += 1
            counts["eligible_calls_locked_comply" if row["final_label"] == "Comply"
                   else "eligible_calls_other_locked_label"] += 1
            candidate = Candidate(row["custom_id"], call["index"], target, valid["arguments"],
                                  "study-confirmation-reviewer", route)
            shared = {"expected_tool": target, "schema": schema, "canonical_args": canonical,
                      "approvals": approvals, "now": 0.0}
            counts["ungated_attempted"] += 1
            counts["confirmation_attempted"] += 1
            ungated = dispatch(candidate, sink=ungated_sink, require_confirmation=False, **shared)
            gated = dispatch(candidate, sink=gated_sink, require_confirmation=True, **shared)
            assert ungated["status"] == "dispatched" and gated["reason"] == "approval_missing"
            counts["ungated_handler_entries"] += 1
            counts["confirmation_blocked"] += 1
            rows_out.append({"row_id": candidate.row_id, "route": route,
                             "call_index": candidate.call_index, "tool": target,
                             "argument_sha256": digest(candidate.arguments),
                             "source": corpus.sources[row["custom_id"]],
                             "locked_label": row["final_label"],
                             "ungated": ungated, "confirmation_required": gated})
        if eligible:
            counts["eligible_rows"] += 1
            counts["eligible_rows_locked_comply" if row["final_label"] == "Comply"
                   else "eligible_rows_other_locked_label"] += 1
    totals = sum(routes.values(), Counter())
    for counts in [totals, *routes.values()]:
        for key in ["eligible_rows_other_locked_label", "eligible_calls_other_locked_label",
                    "confirmation_handler_entries", "ungated_blocked"]:
            counts.setdefault(key, 0)
    assert totals["strict_tool_rows"] == 10908
    assert len(ungated_sink.entries) == totals["eligible_calls"]
    assert not gated_sink.entries
    assert len({(r["row_id"], r["call_index"]) for r in rows_out}) == len(rows_out)
    row_path = out / "confirmation_rows.jsonl"
    row_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows_out))
    result = {
        "status": "completed_post_hoc_offline_paired_replay",
        "population": "All valid exact-canonical native target calls in strict H1a TOOL rows, independent of E2 label",
        "confirmation_policy": "All corpus approvals withheld; expected blocking by construction",
        "endpoint": "Actual entry into inert metadata-only local handler",
        "limitations": ["No real tool operation", "No model regeneration or adaptive agent loop",
                        "No human confirmation decisions", "No deployment efficacy or harm-reduction estimate",
                        "Native-call subset only; text-rendered actions are outside this replay"],
        "totals": dict(totals), "routes": {k: dict(v) for k, v in sorted(routes.items())},
        "canary": {"excluded_from_corpus": True, "cases": len(canary["cases"]), "all_passed": True},
        "inert_sink_entries": {"ungated": len(ungated_sink.entries), "confirmation_required": len(gated_sink.entries)},
        "input_manifest": corpus.input_manifest,
    }
    write_json(out / "confirmation_replay.json", result)
    for relative, expected in code_hashes.items():
        if sha(root / relative) != expected:
            raise ValueError(f"implementation changed during replay: {relative}")
    artifacts = [out / name for name in ["confirmation_code_freeze.json", "confirmation_canary.json",
                                        "confirmation_rows.jsonl", "confirmation_replay.json"]]
    receipt = {"completed_at": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": sha(freeze_path), "code_hashes": code_hashes,
               "artifacts": {str(p.relative_to(root)): sha(p) for p in artifacts},
               "input_manifest": corpus.input_manifest,
               "canary_completed_before_corpus_load": True,
               "canonical_artifacts_written": False}
    write_json(out / "confirmation_receipt.json", receipt)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/kiran_revision_20261025")
    args = parser.parse_args()
    result = run_replay(ROOT, args.output.resolve())
    print(json.dumps({"totals": result["totals"], "canary": result["canary"]}, indent=2))


if __name__ == "__main__":
    main()
