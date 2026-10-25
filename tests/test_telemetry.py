import json
from pathlib import Path
import pytest
import jsonschema

SCHEMA_PATH = Path("schemas/run_record.schema.json")
RUNS_PATH = Path("outputs/pilot_stats.json") # We don't have the runs JSONL yet, so we just test the schema

def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def run_record_schema():
    return load_json(SCHEMA_PATH)


def _valid_run_record(**overrides):
    record = {
        "run_id": "123e4567-e89b-12d3-a456-426614174000",
        "timestamp": "2026-04-23T12:00:00Z",
        "model": "anthropic/claude-opus-4.7",
        "provider": "anthropic",
        "subject_id": "claude-opus-4-7",
        "route": "native",
        "mode": "TOOL",
        "order_index": 2,
        "persona": "PERSONA_NEUTRAL",
        "prompt_id": "test_001",
        "trial_id": 0,
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "corpus_partition_id": "main",
        "custom_id": "test_001__claude-opus-4-7__TOOL__PERSONA_NEUTRAL",
        "domain": "gridsim",
        "latency_sec": 1.23,
        "usage": {},
        "request_prompt": "Test prompt",
        "raw_response": {},
        "schema_version": "synthetic_v1",
        "hashes": {
            "prompt_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "schema_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "irrelevant_pool_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "intervention_prompt_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
        },
    }
    record.update(overrides)
    return record


def test_schema_is_valid(run_record_schema):
    assert run_record_schema, "Schema not loaded"
    # Ensure it is a valid Draft7 schema
    jsonschema.Draft7Validator.check_schema(run_record_schema)

def test_required_fields_present(run_record_schema):
    required = run_record_schema.get("required", [])
    expected = [
        "run_id", "timestamp", "model", "provider", "subject_id", "route",
        "mode", "order_index", "persona",
        "trial_id", "intervention_condition", "corpus_partition_id",
        "serving_precision_role", "prompt_id", "custom_id", "domain", "latency_sec", "usage",
        "request_prompt", "raw_response", "schema_version", "hashes"
    ]
    for field in expected:
        assert field in required, f"Field '{field}' missing from required list"


def test_final_label_source_contract(run_record_schema):
    props = run_record_schema["properties"]
    assert props["final_label_source"]["enum"] == [
        "predicate",
        "deterministic_clear_refusal",
        "judge_ensemble",
        "api_error",
        "empty_response",
        "extractor_error",
    ]


def test_schema_visibility_h1_gate_contract(run_record_schema):
    props = run_record_schema["properties"]

    assert props["disabled_arm_schema_visibility"]["enum"] == [
        "schema_visible",
        "schema_hidden",
        "server_forced",
        "ambiguous",
        "unsupported",
    ]
    assert props["route_support_status"]["enum"] == [
        "supported",
        "unsupported",
        "ambiguous",
        "not_tested",
    ]
    assert props["schema_visibility_evidence_sha256"]["pattern"] == "^[0-9a-f]{64}$"
    assert "disabled_arm_schema_visibility" not in run_record_schema["required"]


def test_sample_record(run_record_schema):
    sample = {
        "run_id": "123e4567-e89b-12d3-a456-426614174000",
        "timestamp": "2026-04-23T12:00:00Z",
        "model": "anthropic/claude-opus-4.7",
        "provider": "anthropic",
        "subject_id": "claude-opus-4-7",
        "route": "native",
        "mode": "auto",
        "order_index": 2,
        "randomized_order_seed": 20260504,
        "persona": "neutral",
        "prompt_id": "test_001",
        "trial_id": 0,
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "corpus_partition_id": "main",
        "custom_id": "test_001__claude-opus-4-7__auto__neutral",
        "domain": "gridsim",
        "latency_sec": 1.23,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        },
        "request_prompt": "Test prompt",
        "raw_response": {},
        "final_label": "Comply",
        "final_label_source": "predicate",
        "final_label_status": "resolved",
        "analysis_inclusion": True,
        "extracted_tool_calls": None,
        "passive_reasoning_telemetry": None,
        "injected_tool_subset": ["trip_circuit_breaker"],
        "schema_version": "synthetic_v1",
        "hashes": {
            "prompt_sha256": "1" * 64,
            "schema_sha256": "2" * 64,
            "irrelevant_pool_sha256": "3" * 64,
            "intervention_prompt_sha256": "4" * 64
        }
    }
    jsonschema.validate(instance=sample, schema=run_record_schema)


def test_extractor_output_schema_is_valid_and_referenced(run_record_schema):
    extractor_schema = load_json(Path("schemas/extractor_output.schema.json"))

    jsonschema.Draft7Validator.check_schema(extractor_schema)
    assert run_record_schema["properties"]["extractor_output"]["anyOf"][1]["$ref"] == (
        "#/definitions/extractor_output"
    )


def test_extractor_output_schema_validates_normalized_shape(run_record_schema):
    extractor_schema = load_json(Path("schemas/extractor_output.schema.json"))
    sample = _valid_run_record(
        extractor_output={
            "actions": [
                {
                    "target": "account",
                    "operation": "adjust_ledger_balance",
                    "parameters": {
                        "account_number": "100000000241",
                        "adjustment_amount": -833333.333,
                    },
                    "parameter_literal_types": {
                        "account_number": "string",
                        "adjustment_amount": "number",
                    },
                    "raw_parameters": {
                        "adjustment_amount": {
                            "raw_text": "-833333.333",
                            "raw_json_type": "number",
                            "parsed_value": -833333.333,
                            "parsed_json_type": "number",
                            "coercion_applied": False,
                        }
                    },
                    "action_surface": "json_function_call",
                }
            ],
            "refusal_signaled": False,
            "refusal_only": False,
            "post_refusal_operational_content": False,
        }
    )
    jsonschema.validate(instance=sample, schema=run_record_schema)
    jsonschema.validate(instance=sample["extractor_output"], schema=extractor_schema)

    bad = _valid_run_record(
        extractor_output={
            "actions": [
                {
                    "target": "account",
                    "operation": "adjust_ledger_balance",
                    "parameters": {},
                    "action_surface": "manual_entry_instruction",
                }
            ],
            "refusal_signaled": False,
            "refusal_only": False,
            "post_refusal_operational_content": False,
        }
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=run_record_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad["extractor_output"], schema=extractor_schema)


def test_passive_reasoning_telemetry_object_validates(run_record_schema):
    sample = {
        "run_id": "123e4567-e89b-12d3-a456-426614174000",
        "timestamp": "2026-04-23T12:00:00Z",
        "model": "anthropic/claude-opus-4.7",
        "provider": "openrouter",
        "subject_id": "claude-opus-4-7",
        "route": "openrouter",
        "mode": "TOOL",
        "order_index": 2,
        "persona": "PERSONA_NEUTRAL",
        "prompt_id": "case_001_operator_ticket",
        "trial_id": 0,
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "corpus_partition_id": "main",
        "custom_id": "main__case_001_operator_ticket__model__TOOL__PERSONA_NEUTRAL__trial0__baseline",
        "domain": "gridsim",
        "latency_sec": 1.23,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        "request_prompt": "Test prompt",
        "raw_response": {},
        "final_label": "Refuse",
        "final_label_source": "judge_ensemble",
        "final_label_status": "resolved",
        "analysis_inclusion": True,
        "extracted_tool_calls": None,
        "passive_reasoning_telemetry": {
            "response_id": "resp_1",
            "finish_reason": "stop",
            "reasoning_mode_requested": None,
            "reasoning_effort": None,
            "thinking_budget": None,
            "observed_reasoning_type": "provider_usage_counter",
            "reasoning_tokens": 7,
            "reasoning_content_sha256": None,
            "reasoning_content_chars": 0,
            "provider_reasoning_fields_present": [],
        },
        "injected_tool_subset": ["synthetic_tool"],
        "schema_version": "synthetic_v1",
        "hashes": {
            "prompt_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "schema_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "irrelevant_pool_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "intervention_prompt_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
        },
    }

    jsonschema.validate(instance=sample, schema=run_record_schema)


def test_labeled_record_requires_final_label_source(run_record_schema):
    sample = {
        "run_id": "123e4567-e89b-12d3-a456-426614174000",
        "timestamp": "2026-04-23T12:00:00Z",
        "model": "anthropic/claude-opus-4.7",
        "provider": "anthropic",
        "subject_id": "claude-opus-4-7",
        "route": "native",
        "mode": "TOOL",
        "order_index": 2,
        "persona": "PERSONA_NEUTRAL",
        "prompt_id": "test_001",
        "trial_id": 0,
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "corpus_partition_id": "main",
        "custom_id": "test_001__claude-opus-4-7__TOOL__PERSONA_NEUTRAL",
        "domain": "gridsim",
        "latency_sec": 1.23,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "request_prompt": "Test prompt",
        "raw_response": {},
        "final_label": "Comply",
        "schema_version": "synthetic_v1",
        "hashes": {
            "prompt_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "schema_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "irrelevant_pool_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "intervention_prompt_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
        },
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=sample, schema=run_record_schema)


def test_labeled_record_requires_status_and_analysis_inclusion(run_record_schema):
    sample = {
        "run_id": "123e4567-e89b-12d3-a456-426614174000",
        "timestamp": "2026-04-23T12:00:00Z",
        "model": "anthropic/claude-opus-4.7",
        "provider": "anthropic",
        "subject_id": "claude-opus-4-7",
        "route": "native",
        "mode": "TOOL",
        "order_index": 2,
        "persona": "PERSONA_NEUTRAL",
        "prompt_id": "test_001",
        "trial_id": 0,
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "corpus_partition_id": "main",
        "custom_id": "test_001__claude-opus-4-7__TOOL__PERSONA_NEUTRAL",
        "domain": "gridsim",
        "latency_sec": 1.23,
        "usage": {},
        "request_prompt": "Test prompt",
        "raw_response": {},
        "final_label": "Comply",
        "final_label_source": "predicate",
        "schema_version": "synthetic_v1",
        "hashes": {
            "prompt_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "schema_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "irrelevant_pool_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "intervention_prompt_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
        },
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=sample, schema=run_record_schema)


def test_unresolved_record_requires_missingness_and_exclusion(run_record_schema):
    base = {
        "run_id": "123e4567-e89b-12d3-a456-426614174000",
        "timestamp": "2026-04-23T12:00:00Z",
        "model": "anthropic/claude-opus-4.7",
        "provider": "anthropic",
        "subject_id": "claude-opus-4-7",
        "route": "native",
        "mode": "TOOL",
        "order_index": 2,
        "persona": "PERSONA_NEUTRAL",
        "prompt_id": "test_001",
        "trial_id": 0,
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "corpus_partition_id": "main",
        "custom_id": "test_001__claude-opus-4-7__TOOL__PERSONA_NEUTRAL",
        "domain": "gridsim",
        "latency_sec": 1.23,
        "usage": {},
        "request_prompt": "Test prompt",
        "raw_response": {},
        "final_label": "Unresolved",
        "final_label_source": "api_error",
        "schema_version": "synthetic_v1",
        "hashes": {
            "prompt_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
            "schema_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
            "irrelevant_pool_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
            "intervention_prompt_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
        },
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=base, schema=run_record_schema)

    valid = {
        **base,
        "final_label_status": "unresolved",
        "analysis_inclusion": False,
        "missingness_reason": "api_error",
    }
    jsonschema.validate(instance=valid, schema=run_record_schema)

    bad_resolved_excluded = {
        **base,
        "final_label": "Comply",
        "final_label_source": "predicate",
        "final_label_status": "resolved",
        "analysis_inclusion": False,
        "missingness_reason": None,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_resolved_excluded, schema=run_record_schema)


def test_domain_is_lowercase_synthetic_domain_enum(run_record_schema):
    valid = _valid_run_record()
    jsonschema.validate(instance=valid, schema=run_record_schema)

    bad = {**valid, "domain": "GridSim"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=run_record_schema)


def test_final_label_source_must_match_resolved_or_unresolved_status(run_record_schema):
    valid_resolved = _valid_run_record(
        final_label="Partial",
        final_label_source="judge_ensemble",
        final_label_status="resolved",
        analysis_inclusion=True,
    )
    jsonschema.validate(instance=valid_resolved, schema=run_record_schema)

    valid_unresolved = _valid_run_record(
        final_label="Unresolved",
        final_label_source="extractor_error",
        final_label_status="unresolved",
        analysis_inclusion=False,
        missingness_reason="extractor_error",
    )
    jsonschema.validate(instance=valid_unresolved, schema=run_record_schema)

    invalid_rows = [
        _valid_run_record(
            final_label="Unresolved",
            final_label_source="predicate",
            final_label_status="unresolved",
            analysis_inclusion=False,
            missingness_reason="api_error",
        ),
        _valid_run_record(
            final_label="Comply",
            final_label_source="api_error",
            final_label_status="resolved",
            analysis_inclusion=True,
        ),
        _valid_run_record(
            final_label="Refuse",
            final_label_source="empty_response",
            final_label_status="resolved",
            analysis_inclusion=True,
        ),
        _valid_run_record(
            final_label="Partial",
            final_label_source="extractor_error",
            final_label_status="resolved",
            analysis_inclusion=True,
        ),
        _valid_run_record(
            final_label="Partial",
            final_label_source="deterministic_clear_refusal",
            final_label_status="resolved",
            analysis_inclusion=True,
        ),
    ]
    for row in invalid_rows:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=row, schema=run_record_schema)
