"""Contracts for a local dispatch boundary; fixtures never operate a real tool."""
from dataclasses import replace

import pytest

from analysis.confirmation_replay import (
    ApprovalStore, Candidate, InertSink, dispatch, run_canary,
)


SCHEMA = {
    "type": "object", "properties": {"message": {"type": "string"}},
    "required": ["message"], "additionalProperties": False,
}
ARGS = {"message": "synthetic benign check"}


@pytest.fixture
def candidate():
    return Candidate("synthetic-row-1", 0, "synthetic_echo", ARGS, "test-reviewer")


def send(candidate, store, sink, token=None, required=True, now=100):
    return dispatch(candidate, expected_tool="synthetic_echo", schema=SCHEMA,
                    canonical_args=ARGS, approvals=store, sink=sink,
                    require_confirmation=required, approval_token=token, now=now)


def test_ungated_calls_handler_but_confirmation_arm_does_not(candidate):
    store, sink = ApprovalStore(), InertSink()
    assert send(candidate, store, sink, required=False)["status"] == "dispatched"
    assert len(sink.entries) == 1
    assert send(candidate, store, sink)["reason"] == "approval_missing"
    assert len(sink.entries) == 1
    assert "arguments" not in sink.entries[0]
    assert len(sink.entries[0]["argument_sha256"]) == 64


def test_trusted_approval_dispatches_once(candidate):
    store, sink = ApprovalStore(), InertSink()
    token = store.issue(candidate, expires_at=150)
    assert send(candidate, store, sink, token)["status"] == "dispatched"
    assert send(candidate, store, sink, token)["reason"] == "approval_reused"
    assert len(sink.entries) == 1


@pytest.mark.parametrize("token", ["confirmed", "true", "forged-token", {"approved": True}])
def test_model_assertion_or_forged_token_is_not_confirmation(candidate, token):
    store, sink = ApprovalStore(), InertSink()
    assert send(candidate, store, sink, token)["status"] == "blocked"
    assert not sink.entries


@pytest.mark.parametrize("change", [
    {"row_id": "another-row"}, {"call_index": 1},
    {"principal": "another-reviewer"},
])
def test_approval_is_bound_to_context(candidate, change):
    store, sink = ApprovalStore(), InertSink()
    token = store.issue(candidate, expires_at=150)
    assert send(replace(candidate, **change), store, sink, token)["reason"] == "approval_mismatch"
    assert not sink.entries


def test_approval_is_bound_to_tool_and_arguments(candidate):
    store, sink = ApprovalStore(), InertSink()
    token = store.issue(candidate, expires_at=150)
    changed = replace(candidate, tool="synthetic_other", arguments={"message": "changed"})
    # Change the validation contract too, isolating approval binding from validation.
    result = dispatch(changed, expected_tool=changed.tool, schema=SCHEMA,
                      canonical_args=changed.arguments, approvals=store, sink=sink,
                      require_confirmation=True, approval_token=token, now=100)
    assert result["reason"] == "approval_mismatch"
    assert not sink.entries


@pytest.mark.parametrize("now", [150, 151])
def test_expired_approval_cannot_dispatch(candidate, now):
    store, sink = ApprovalStore(), InertSink()
    token = store.issue(candidate, expires_at=150)
    assert send(candidate, store, sink, token, now=now)["reason"] == "approval_expired"
    assert not sink.entries


@pytest.mark.parametrize("arguments", [None, "{}", {}, {"message": 1},
                                         {"message": "changed"},
                                         {"message": ARGS["message"], "confirmed": True}])
def test_invalid_or_noncanonical_payload_never_reaches_handler(candidate, arguments):
    store, sink = ApprovalStore(), InertSink()
    result = send(replace(candidate, arguments=arguments), store, sink, required=False)
    assert result["status"] == "validation_rejected"
    assert not sink.entries


def test_wrong_tool_rejected_before_handler(candidate):
    store, sink = ApprovalStore(), InertSink()
    assert send(replace(candidate, tool="unknown"), store, sink, required=False)["reason"] == "wrong_tool"
    assert not sink.entries


def test_boolean_is_not_integer():
    candidate = Candidate("synthetic-row", 0, "synthetic_count", {"n": True}, "reviewer")
    result = dispatch(candidate, expected_tool=candidate.tool,
                      schema={"type": "object", "properties": {"n": {"type": "integer"}}},
                      canonical_args={"n": 1}, approvals=ApprovalStore(), sink=InertSink(),
                      require_confirmation=False, now=100)
    assert result["status"] == "validation_rejected"


def test_canary_checks_entry_and_no_entry_with_explicit_exclusion():
    result = run_canary()
    assert result["excluded_from_corpus"] is True
    assert result["all_passed"] is True
    assert result["cases"]["trusted_approval"]["handler_entries"] == 1
    assert result["cases"]["withheld_approval"]["handler_entries"] == 0
    assert len(result["cases"]) >= 12
