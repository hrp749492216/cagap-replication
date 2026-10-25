from __future__ import annotations

import copy
import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.analyze_disabled_format_control import (
    analyze_cells,
    infer_format_condition,
    load_cells,
    load_expected_rows,
)
from scripts.fetch_native_batch_results import _submission_manifest
from scripts.grade_subject_responses import expected_judge_vote_routes
from scripts.run_disabled_format_control import (
    DEFAULT_OUTPUT_DIR,
    FORMAT_BASELINE,
    FORMAT_JSON,
    _append_format_instruction,
    _derive_row,
    _estimated_cost_upper_bound,
    _json_bytes,
    build_control_openrouter_body,
    build_control_openrouter_run_record,
    build_rows_from_pairs,
    load_protocol,
    submit_generated,
    validate_generated_rows,
    validate_protocol_output_target,
    validate_protocol_draft,
)
from scripts.schema_visibility_preflight import (
    build_openai_body,
    build_routes,
    classify,
    load_env,
    parse_openai,
    select_control_endpoint,
    validate_openrouter_auth,
)
from scripts.run_synthetic_prompts import _openrouter_run_record


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "experiments/disabled_format_control/protocol.json"
V3_PROTOCOL_PATH = (
    REPO_ROOT / "experiments/disabled_format_control/protocol_v3_draft.json"
)
LOCKED_INPUTS_MANIFEST_PATH = (
    REPO_ROOT / "experiments/disabled_format_control/locked_inputs_manifest.json"
)
ROSTER_PATH = REPO_ROOT / "config/roster.json"
NATIVE_PRICING_PATH = REPO_ROOT / "config/api_pricings_native_providers.json"
OPENROUTER_PRICING_PATH = REPO_ROOT / "config/openrouter_pricing_current.json"
V3_ARMS = (
    "DISABLED_BASELINE",
    "DISABLED_FORMAT_JSON",
    "TOOL_CONTEMPORANEOUS",
)


def test_execution_cost_gate_uses_external_current_pricing_overlays() -> None:
    assert hashlib.sha256(ROSTER_PATH.read_bytes()).hexdigest() == (
        "cd13e61ecc1a083f66b5159555f13461cabda5c55ee2a75664ee671077c8bbe1"
    )

    native = json.loads(NATIVE_PRICING_PATH.read_text(encoding="utf-8"))
    native_rows = {row["model_key"]: row for row in native["pricing"]}
    assert native_rows["grok-4.3"]["batch_api_input_usd_per_1m_tokens"] == pytest.approx(
        1.0
    )
    assert native_rows["grok-4.3"]["batch_api_output_usd_per_1m_tokens"] == pytest.approx(
        2.0
    )
    assert native_rows["mistral-large-2512"][
        "regular_api_input_usd_per_1m_tokens"
    ] == pytest.approx(2.0)
    assert native_rows["mistral-large-2512"][
        "regular_api_output_usd_per_1m_tokens"
    ] == pytest.approx(6.0)
    assert native_rows["mistral-large-2512"][
        "batch_api_input_usd_per_1m_tokens"
    ] == pytest.approx(1.0)
    assert native_rows["mistral-large-2512"][
        "batch_api_output_usd_per_1m_tokens"
    ] == pytest.approx(3.0)

    openrouter = json.loads(OPENROUTER_PRICING_PATH.read_text(encoding="utf-8"))
    mistral = next(
        row for row in openrouter["models"] if row["model_key"] == "mistral-large-2512"
    )
    assert mistral["input_usd_per_1m_tokens"] == pytest.approx(0.5)
    assert mistral["cached_input_usd_per_1m_tokens"] == pytest.approx(0.05)
    assert mistral["output_usd_per_1m_tokens"] == pytest.approx(1.5)

    locked = json.loads(LOCKED_INPUTS_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert locked["full"]["estimated_api_cost_upper_bound_usd"]["grok-4.3"] == pytest.approx(
        28.146564
    )
    assert locked["full"]["estimated_api_cost_upper_bound_usd"]["total"] == pytest.approx(
        601.785098
    )
    assert locked["canary"]["estimated_api_cost_upper_bound_usd"]["total"] == pytest.approx(
        0.520869
    )


def test_cost_estimator_does_not_read_prices_from_frozen_roster() -> None:
    protocol, _ = load_protocol(PROTOCOL_PATH)
    grok_body = {"messages": [], "max_tokens": 4096}
    claude_body = {"messages": [], "max_tokens": 4096}
    rows = [
        {"subject_id": "grok-4.3", "body": grok_body},
        {"subject_id": "claude-opus-4-7", "body": claude_body},
    ]
    estimate = _estimated_cost_upper_bound(rows, protocol)

    expected_grok = round((len(_json_bytes(grok_body)) * 1.0 + 4096 * 2.0) / 1_000_000, 6)
    expected_claude = round(
        (len(_json_bytes(claude_body)) * 5.0 + 4096 * 25.0) / 1_000_000,
        6,
    )
    assert estimate["per_subject_usd"]["grok-4.3"] == pytest.approx(expected_grok)
    assert estimate["per_subject_usd"]["claude-opus-4-7"] == pytest.approx(expected_claude)
    assert estimate["pricing_snapshot"]["frozen_roster_used_for_pricing"] is False
    assert estimate["pricing_snapshot"]["native"]["sha256"] == hashlib.sha256(
        NATIVE_PRICING_PATH.read_bytes()
    ).hexdigest()
    assert estimate["pricing_snapshot"]["openrouter"]["sha256"] == hashlib.sha256(
        OPENROUTER_PRICING_PATH.read_bytes()
    ).hexdigest()


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{index}",
                "description": f"tool {index}",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }
        for index in range(6)
    ]


def _source_row(mode: str) -> dict:
    return {
        "custom_id": f"main__P1__claude-opus-4-7__{mode}__neutral__trial0__baseline",
        "subject_id": "claude-opus-4-7",
        "prompt_id": "P1",
        "persona": "neutral",
        "trial_id": 0,
        "mode": mode,
        "domain": "finance",
        "corpus_partition_id": "main",
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "system_prompt_hash": "source-system-hash",
        "serialized_request_hash": "source-request-hash",
        "randomized_order_seed": 20260504,
        "hashes": {},
        "body": {
            "model": "anthropic/claude-opus-4.7",
            "messages": [
                {"role": "system", "content": "Follow safety policies."},
                {"role": "user", "content": "Perform the requested operation."},
            ],
            "tools": _tools(),
            "tool_choice": "none" if mode == "DISABLED" else "auto",
            "max_tokens": 64,
        },
    }


def _v3_test_protocol(tmp_path: Path) -> dict:
    protocol = json.loads(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    evidence = tmp_path / "route_preflight.json"
    evidence.write_text('{"status":"supported"}\n', encoding="utf-8")
    route = protocol["route_policy"]["routes"]["claude-opus-4-7"]
    route.update(
        {
            "preflight_evidence_id": str(evidence),
            "preflight_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "disabled_schema_retention": "schema_visible",
            "tool_native_call_support": "supported",
            "provider_pin_verified": True,
        }
    )
    return protocol


def test_v3_protocol_declares_three_arm_inventory_and_neutral_instruction() -> None:
    protocol = json.loads(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))

    assert tuple(protocol["design"]["arms"]) == V3_ARMS
    assert protocol["population"]["expected_analysis_sets"] == 11_520
    assert protocol["population"]["expected_cells_per_arm"] == 11_520
    assert protocol["population"]["expected_rows_total"] == 34_560
    assert protocol["population"]["expected_canary_rows_total"] == 30
    assert protocol["design"]["partition_namespace"] == (
        "postreg_disabled_format_control_v3"
    )
    assert protocol["artifact_paths"]["full_batches_dir"].startswith(
        "outputs/postreg_disabled_format_control_v3/"
    )
    instruction = protocol["design"]["format_instruction_exact"]
    assert "If you choose to act" in instruction
    assert "same safety policies" not in instruction
    assert "authorization" not in instruction.lower()


def test_v3_builds_six_row_miniblock_and_three_row_canary(tmp_path: Path) -> None:
    protocol = _v3_test_protocol(tmp_path)
    pairs = {}
    for persona in ("neutral", "operator"):
        disabled = _source_row("DISABLED")
        tool = _source_row("TOOL")
        for row in (disabled, tool):
            row["persona"] = persona
            row["custom_id"] = row["custom_id"].replace("neutral", persona)
        key = ("claude-opus-4-7", "P1", persona, 0)
        pairs[key] = {"DISABLED": disabled, "TOOL": tool}

    rows = build_rows_from_pairs(
        protocol,
        "b" * 64,
        pairs,
        "claude-opus-4-7",
        canary=False,
        source_batch_sha256="a" * 64,
    )
    assert len(rows) == 6
    assert {row["control_arm"] for row in rows} == set(V3_ARMS)
    assert {row["dispatch_block_id"] for row in rows} == {
        "claude-opus-4-7:dfc00000:P1:t0"
    }
    assert sorted(row["order_index"] for row in rows) == list(range(6))
    assert {row["corpus_partition_id"] for row in rows} == {
        "postreg_disabled_format_control_v3_baseline",
        "postreg_disabled_format_control_v3_format_json",
        "postreg_disabled_format_control_v3_tool",
    }
    assert all(
        row["custom_id"].startswith("postreg_disabled_format_control_v3_")
        for row in rows
    )

    canary_rows = build_rows_from_pairs(
        protocol,
        "b" * 64,
        pairs,
        "claude-opus-4-7",
        canary=True,
        source_batch_sha256="a" * 64,
    )
    assert len(canary_rows) == 3
    assert {row["control_arm"] for row in canary_rows} == set(V3_ARMS)
    assert all(row["analysis_inclusion"] is False for row in canary_rows)
    assert {row["corpus_partition_id"] for row in canary_rows} == {
        "postreg_disabled_format_control_v3_canary_baseline",
        "postreg_disabled_format_control_v3_canary_format_json",
        "postreg_disabled_format_control_v3_canary_tool",
    }


def test_v3_triplet_differs_only_by_registered_interventions(tmp_path: Path) -> None:
    protocol = _v3_test_protocol(tmp_path)
    pairs = {
        ("claude-opus-4-7", "P1", "neutral", 0): {
            "DISABLED": _source_row("DISABLED"),
            "TOOL": _source_row("TOOL"),
        }
    }
    rows = build_rows_from_pairs(
        protocol,
        "b" * 64,
        pairs,
        "claude-opus-4-7",
        canary=True,
        source_batch_sha256="a" * 64,
    )
    by_arm = {row["control_arm"]: row for row in rows}
    baseline = by_arm["DISABLED_BASELINE"]
    formatted = by_arm["DISABLED_FORMAT_JSON"]
    tool = by_arm["TOOL_CONTEMPORANEOUS"]

    baseline_plus_instruction = copy.deepcopy(baseline["body"])
    _append_format_instruction(
        baseline_plus_instruction, protocol["design"]["format_instruction_exact"]
    )
    assert baseline_plus_instruction == formatted["body"]
    baseline_without_choice = {
        key: value for key, value in baseline["body"].items() if key != "tool_choice"
    }
    tool_without_choice = {
        key: value for key, value in tool["body"].items() if key != "tool_choice"
    }
    assert baseline_without_choice == tool_without_choice
    assert baseline["body"]["tool_choice"] == "none"
    assert tool["body"]["tool_choice"] == "auto"
    assert {row["route"] for row in rows} == {"openrouter"}
    assert {row["pinned_provider"] for row in rows} == {"anthropic"}
    assert {row["body"]["model"] for row in rows} == {
        "anthropic/claude-opus-4.7"
    }


def test_format_arm_changes_only_frozen_system_instruction() -> None:
    protocol, protocol_sha = load_protocol(PROTOCOL_PATH)
    rows = []
    for mode in ("DISABLED",):
        source = _source_row(mode)
        for fmt in (FORMAT_BASELINE, FORMAT_JSON):
            rows.append(
                _derive_row(
                    source,
                    format_condition=fmt,
                    protocol=protocol,
                    protocol_sha256=protocol_sha,
                    source_batch_sha256="a" * 64,
                    canary=False,
                )
            )

    validate_generated_rows(rows, protocol, canary=False)
    by_arm = {row["control_arm"]: row for row in rows}
    instruction = protocol["design"]["format_instruction_exact"]
    for mode in ("DISABLED",):
        baseline = copy.deepcopy(by_arm[f"{mode}_BASELINE"]["body"])
        formatted = by_arm[f"{mode}_FORMAT_JSON"]["body"]
        _append_format_instruction(baseline, instruction)
        assert baseline == formatted
        assert by_arm[f"{mode}_FORMAT_JSON"]["format_instruction_present"] is True
        assert by_arm[f"{mode}_BASELINE"]["format_instruction_present"] is False
        for row in by_arm.values():
            assert row["randomized_order_seed"] == 20260807
            assert row["source_randomized_order_seed"] == 20260504
            assert row["disabled_arm_schema_visibility"] == "schema_visible"
            assert row["route_support_status"] == "supported"


def _label_row(
    *,
    subject: str,
    prompt: str,
    arm: str,
    label: str,
    domain: str = "finance",
) -> dict:
    if arm == "TOOL_CONTEMPORANEOUS":
        mode, format_condition = "TOOL", "BASELINE"
    else:
        mode, format_condition = arm.split("_", 1)
    return {
        "custom_id": f"postreg__{subject}__{prompt}__{arm}",
        "subject_id": subject,
        "prompt_id": prompt,
        "persona": "neutral",
        "trial_id": 0,
        "domain": domain,
        "mode": mode,
        "format_condition": format_condition,
        "control_arm": arm,
        "final_label": label,
    }


def test_analysis_recovers_paired_format_estimand() -> None:
    cells = {}
    labels = {
        "P1": {
            "DISABLED_BASELINE": "Refuse",
            "DISABLED_FORMAT_JSON": "Comply",
        },
        "P2": {
            "DISABLED_BASELINE": "Refuse",
            "DISABLED_FORMAT_JSON": "Partial",
        },
    }
    for prompt, by_arm_label in labels.items():
        key = ("subject-a", prompt, "neutral", 0)
        cells[key] = {
            arm: _label_row(subject="subject-a", prompt=prompt, arm=arm, label=label)
            for arm, label in by_arm_label.items()
        }

    result = analyze_cells(cells, bootstrap_replicates=50, seed=20260807)
    e2 = result["contrasts"]["E2"]
    assert e2["format_effect_disabled"]["risk_difference"] == pytest.approx(0.5)
    assert result["strict_two_arm_resolved_complete_pairs"] == 2


def test_unresolved_row_is_excluded_and_bounded() -> None:
    complete = {
        arm: _label_row(subject="s", prompt="P1", arm=arm, label="Refuse")
        for arm in (
            "DISABLED_BASELINE",
            "DISABLED_FORMAT_JSON",
        )
    }
    incomplete = {
        arm: _label_row(
            subject="s",
            prompt="P2",
            arm=arm,
            label="Unresolved" if arm == "DISABLED_FORMAT_JSON" else "Refuse",
        )
        for arm in (
            "DISABLED_BASELINE",
            "DISABLED_FORMAT_JSON",
        )
    }
    cells = {
        ("s", "P1", "neutral", 0): complete,
        ("s", "P2", "neutral", 0): incomplete,
    }
    result = analyze_cells(cells, bootstrap_replicates=0, seed=1)
    assert result["strict_two_arm_resolved_complete_pairs"] == 1
    assert result["complete_cell_retention_rate"] == pytest.approx(0.5)
    bounds = result["contrasts"]["E2"]["format_effect_disabled"]
    assert bounds["adversarial_missingness_bounds_all_expected_cells"] == [0.0, 0.5]


def test_absent_pair_remains_in_missingness_denominator() -> None:
    cells = {
        ("s", "P1", "neutral", 0): {
            arm: _label_row(subject="s", prompt="P1", arm=arm, label="Refuse")
            for arm in ("DISABLED_BASELINE", "DISABLED_FORMAT_JSON")
        },
        ("s", "P2", "neutral", 0): {},
    }
    result = analyze_cells(cells, bootstrap_replicates=0, seed=1)
    assert result["expected_cell_identities"] == 2
    assert result["complete_cell_retention_rate"] == pytest.approx(0.5)
    assert result["arm_rates"]["DISABLED_BASELINE"]["rows_absent"] == 1
    assert result["arm_rates"]["DISABLED_BASELINE"][
        "missing_or_unresolved_rate_all_expected"
    ] == pytest.approx(0.5)
    assert result["contrasts"]["E2"]["format_effect_disabled"][
        "adversarial_missingness_bounds_all_expected_cells"
    ] == [-0.5, 0.5]


def test_v3_analysis_recovers_all_three_contrasts() -> None:
    labels = {
        "P1": {
            "DISABLED_BASELINE": "Refuse",
            "DISABLED_FORMAT_JSON": "Partial",
            "TOOL_CONTEMPORANEOUS": "Comply",
        },
        "P2": {
            "DISABLED_BASELINE": "Refuse",
            "DISABLED_FORMAT_JSON": "Comply",
            "TOOL_CONTEMPORANEOUS": "Comply",
        },
    }
    cells = {
        ("subject-a", prompt, "neutral", 0): {
            arm: _label_row(subject="subject-a", prompt=prompt, arm=arm, label=label)
            for arm, label in by_arm.items()
        }
        for prompt, by_arm in labels.items()
    }

    result = analyze_cells(
        cells,
        bootstrap_replicates=50,
        seed=20260807,
        arms=V3_ARMS,
    )
    e2 = result["contrasts"]["E2"]
    assert e2["format_affordance"]["risk_difference"] == pytest.approx(0.5)
    assert e2["total_same_route_gap"]["risk_difference"] == pytest.approx(1.0)
    assert e2["residual_action_boundary"]["risk_difference"] == pytest.approx(0.5)
    assert result["strict_resolved_complete_sets"] == 2
    assert result["analysis_set_arity"] == 3


def test_v3_missing_tool_arm_is_excluded_and_bounded() -> None:
    complete = {
        arm: _label_row(subject="s", prompt="P1", arm=arm, label="Refuse")
        for arm in V3_ARMS
    }
    missing_tool = {
        arm: _label_row(subject="s", prompt="P2", arm=arm, label="Refuse")
        for arm in V3_ARMS
        if arm != "TOOL_CONTEMPORANEOUS"
    }
    cells = {
        ("s", "P1", "neutral", 0): complete,
        ("s", "P2", "neutral", 0): missing_tool,
    }

    result = analyze_cells(cells, bootstrap_replicates=0, seed=1, arms=V3_ARMS)
    assert result["strict_resolved_complete_sets"] == 1
    assert result["complete_set_retention_rate"] == pytest.approx(0.5)
    assert result["arm_rates"]["TOOL_CONTEMPORANEOUS"]["rows_absent"] == 1
    residual = result["contrasts"]["E2"]["residual_action_boundary"]
    assert residual["adversarial_missingness_bounds_all_expected_sets"] == [0.0, 0.5]


def test_v3_unresolved_member_excludes_whole_triplet() -> None:
    triplet = {
        arm: _label_row(
            subject="s",
            prompt="P1",
            arm=arm,
            label="Unresolved" if arm == "DISABLED_FORMAT_JSON" else "Comply",
        )
        for arm in V3_ARMS
    }
    with pytest.raises(ValueError, match="complete three-arm sets"):
        analyze_cells(
            {("s", "P1", "neutral", 0): triplet},
            bootstrap_replicates=0,
            seed=1,
            arms=V3_ARMS,
        )


def test_format_condition_can_be_recovered_after_transport_drops_extra_fields() -> None:
    assert infer_format_condition(
        {"corpus_partition_id": "postreg_disabled_format_control_format_json"}
    ) == "FORMAT_JSON"
    assert infer_format_condition(
        {"custom_id": "postreg_disabled_format_control_baseline__P1__rest"}
    ) == "BASELINE"


def test_inconsistent_format_metadata_fails_closed() -> None:
    with pytest.raises(ValueError, match="inconsistent format condition"):
        infer_format_condition(
            {
                "format_condition": "BASELINE",
                "corpus_partition_id": "postreg_disabled_format_control_format_json",
            }
        )


def test_analyzer_uses_exact_locked_expected_identities(tmp_path: Path) -> None:
    protocol_sha = "b" * 64
    protocol = {
        "protocol_id": "test-protocol",
        "population": {"expected_rows_total": 2},
    }
    expected_rows = []
    for format_condition, arm, suffix in (
        ("BASELINE", "DISABLED_BASELINE", "baseline"),
        ("FORMAT_JSON", "DISABLED_FORMAT_JSON", "format_json"),
    ):
        expected_rows.append(
            {
                "custom_id": f"postreg_disabled_format_control_{suffix}__P1__s__DISABLED__neutral__trial0__baseline",
                "subject_id": "s",
                "prompt_id": "P1",
                "persona": "neutral",
                "trial_id": 0,
                "domain": "gridsim",
                "mode": "DISABLED",
                "format_condition": format_condition,
                "control_arm": arm,
                "corpus_partition_id": f"postreg_disabled_format_control_{suffix}",
                "protocol_id": "test-protocol",
                "protocol_sha256": protocol_sha,
                "serialized_request_hash": ("1" if suffix == "baseline" else "2") * 64,
                "analysis_inclusion": True,
                "operational_canary": False,
            }
        )
    batches_dir = tmp_path / "batches"
    batches_dir.mkdir()
    batch_path = batches_dir / "input_s.jsonl"
    batch_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in expected_rows),
        encoding="utf-8",
    )
    lock_path = tmp_path / "locked.json"
    lock_path.write_text(
        json.dumps(
            {
                "protocol_id": "test-protocol",
                "protocol_sha256": protocol_sha,
                "full": {
                    "output_files": {
                        "input_s.jsonl": {
                            "rows": 2,
                            "sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest(),
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    expected = load_expected_rows(batches_dir, lock_path, protocol, protocol_sha)
    labels_path = tmp_path / "labels.jsonl"
    labeled = [{**row, "final_label": "Refuse"} for row in expected_rows]
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in labeled),
        encoding="utf-8",
    )
    cells, _, seen = load_cells([labels_path], protocol, protocol_sha, expected)
    assert len(seen) == 2
    assert len(cells) == 1

    labeled[0]["persona"] = "operator"
    labels_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in labeled),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="persona mismatch"):
        load_cells([labels_path], protocol, protocol_sha, expected)


def test_legacy_control_submit_manifest_hands_native_fields_to_fetcher() -> None:
    provider_result = {
        "subject_id": "gpt-5.5",
        "final_route": "native",
        "provider_batch_id": "batch-1",
    }
    assert _submission_manifest({"provider_result": provider_result}) == provider_result


def test_all_family_excluded_judges_have_openrouter_route_locks() -> None:
    routes = expected_judge_vote_routes("openrouter")
    assert set(routes) == {
        "glm-5.1",
        "command-a-03-2025",
        "mistral-large-2512",
        "gpt-5.5-judge",
        "claude-opus-4-7-judge",
    }
    assert {route["source"] for route in routes.values()} == {"openrouter"}


def test_openrouter_subject_record_preserves_control_metadata() -> None:
    row = {
        "custom_id": "control-row",
        "subject_id": "claude-opus-4-7",
        "protocol_id": "protocol-v2",
        "protocol_sha256": "a" * 64,
        "format_condition": "FORMAT_JSON",
        "control_arm": "DISABLED_FORMAT_JSON",
        "randomized_order_seed": 20260807,
        "mode": "DISABLED",
        "order_index": 0,
        "persona": "neutral",
        "trial_id": 0,
        "intervention_condition": "baseline",
        "corpus_partition_id": "postreg_disabled_format_control_format_json",
        "serving_precision_role": "primary",
        "prompt_id": "P1",
        "domain": "gridsim",
        "hashes": {},
        "body": {},
    }
    body = {"model": "anthropic/claude-opus-4.7", "messages": []}
    response = {
        "model": "anthropic/claude-opus-4.7",
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
    }
    record = _openrouter_run_record(
        row=row,
        body=body,
        response=response,
        pinned_provider="anthropic",
        latency_sec=0.1,
    )
    assert record["protocol_id"] == "protocol-v2"
    assert record["control_arm"] == "DISABLED_FORMAT_JSON"
    assert record["randomized_order_seed"] == 20260807


def test_v3_draft_validates_structurally_but_cannot_load_for_execution() -> None:
    summary = validate_protocol_draft(V3_PROTOCOL_PATH)
    assert summary["status"] == "draft_pending_route_preflight"
    assert summary["subjects"] == 10
    assert summary["arms"] == 3
    assert summary["expected_rows_total"] == 34_560
    assert summary["expected_canary_rows_total"] == 30
    assert summary["routes_with_complete_evidence"] == 4
    assert summary["promotable"] is False

    with pytest.raises(ValueError, match="draft.*generation or submission"):
        load_protocol(V3_PROTOCOL_PATH)


def test_v3_artifact_paths_cannot_collide_with_frozen_v2() -> None:
    protocol = json.loads(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="refusing colliding output target"):
        validate_protocol_output_target(protocol, DEFAULT_OUTPUT_DIR, canary=False)

    validate_protocol_output_target(
        protocol,
        REPO_ROOT / protocol["artifact_paths"]["full_batches_dir"],
        canary=False,
    )
    validate_protocol_output_target(
        protocol,
        REPO_ROOT / protocol["artifact_paths"]["canary_batches_dir"],
        canary=True,
    )


def test_v3_submission_rejects_draft_before_any_spend_gate() -> None:
    args = Namespace(
        protocol=V3_PROTOCOL_PATH,
        subject=["claude-opus-4-7"],
        acknowledge_postregistration_exploratory=True,
        confirm_spend="CAGAP-POSTREG-DISABLED-FORMAT-2026-09-01-v3-DRAFT",
        max_requests=3,
        max_cost_usd=1.0,
    )
    with pytest.raises(ValueError, match="draft.*generation or submission"):
        submit_generated(args)


def test_v3_control_route_table_is_all_openrouter_and_quantization_aware() -> None:
    routes = build_routes(ROSTER_PATH, control_routes_path=V3_PROTOCOL_PATH)
    assert len(routes) == 10
    assert {route["route_type"] for route in routes} == {"openrouter"}
    assert {route["subject_id"] for route in routes} == set(
        json.loads(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))["population"][
            "subjects"
        ]
    )

    qwen = next(route for route in routes if route["subject_id"] == "qwen3.6-35b-a3b")
    assert qwen["slug"] == "qwen/qwen3.6-35b-a3b"
    assert qwen["pinned_provider"] == "venice/fp8"
    assert qwen["quantizations"] == ["fp8"]
    assert qwen["decoding"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
    }
    body, _ = build_openai_body(
        {"prompt_text": "Diagnostic prompt."},
        qwen,
        "NOTOOLS",
    )
    assert body["provider"] == {
        "only": ["venice/fp8"],
        "quantizations": ["fp8"],
        "allow_fallbacks": False,
        "data_collection": "deny",
    }

    control_body = build_control_openrouter_body(
        {
            "body": {
                "model": "stale/local-model",
                "messages": [],
                "provider": {"only": ["stale-provider"]},
            }
        },
        {
            "slug": qwen["slug"],
            "pinned_provider": qwen["pinned_provider"],
            "quantizations": qwen["quantizations"],
            "decoding": qwen["decoding"],
        },
    )
    assert control_body["model"] == "qwen/qwen3.6-35b-a3b"
    assert control_body["provider"] == body["provider"]
    assert control_body["transforms"] == []

    expected_exact_pins = {
        "gemma-4-31b-it": "venice/bf16",
        "gemini-3.5-flash": "google-ai-studio/flex",
        "glm-4.7-flash": "venice/fp8",
        "gpt-5.5": "openai/flex",
        "grok-4.3": "xai/zdr",
        "kimi-k2.6": "moonshotai/int4",
        "mistral-medium-3.5-128b": "mistral/zdr",
    }
    assert {
        subject_id: next(
            route["pinned_provider"]
            for route in routes
            if route["subject_id"] == subject_id
        )
        for subject_id in expected_exact_pins
    } == expected_exact_pins


def test_v3_preflight_and_transport_share_exact_route_decoding() -> None:
    routes = {
        route["subject_id"]: route
        for route in build_routes(ROSTER_PATH, control_routes_path=V3_PROTOCOL_PATH)
    }
    expected = {
        "claude-opus-4-7": {"max_tokens": 4096},
        "gpt-5.5": {"max_tokens": 4096},
        "kimi-k2.6": {"max_tokens": 4096},
    }
    decoding_keys = {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
    }

    for subject_id, expected_decoding in expected.items():
        route = routes[subject_id]
        preflight_body, _ = build_openai_body(
            {"prompt_text": "Diagnostic prompt."}, route, "NOTOOLS"
        )
        assert {
            key: preflight_body[key]
            for key in decoding_keys
            if key in preflight_body
        } == expected_decoding

        transport_body = build_control_openrouter_body(
            {
                "subject_id": subject_id,
                "body": {
                    "model": "stale-model",
                    "messages": [],
                    "temperature": 1.7,
                    "top_p": 0.2,
                    "max_completion_tokens": 12,
                },
            },
            {
                "slug": route["slug"],
                "pinned_provider": route["pinned_provider"],
                "quantizations": route["quantizations"],
                "decoding": route["decoding"],
            },
        )
        assert {
            key: transport_body[key]
            for key in decoding_keys
            if key in transport_body
        } == expected_decoding


def test_v3_preflight_endpoint_metadata_gate_and_cost_capture() -> None:
    route = {
        "subject_id": "qwen3.6-35b-a3b",
        "slug": "qwen/qwen3.6-35b-a3b",
        "pinned_provider": "venice/fp8",
        "quantizations": ["fp8"],
        "decoding": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
    }
    endpoint = {
        "tag": "venice/fp8",
        "provider_name": "Venice",
        "quantization": "fp8",
        "status": 0,
        "supported_parameters": [
            "tools",
            "tool_choice",
            "temperature",
            "top_p",
            "max_tokens",
        ],
        "supports_tool_choice": {
            "auto": True,
            "function": True,
            "none": True,
            "required": True,
        },
        "pricing": {"prompt": "0.0000001", "completion": "0.000001"},
    }
    assert select_control_endpoint(route, {"data": {"endpoints": [endpoint]}}) == endpoint

    invalid = copy.deepcopy(endpoint)
    invalid["supports_tool_choice"]["none"] = False
    with pytest.raises(ValueError, match="tool_choice:none"):
        select_control_endpoint(route, {"data": {"endpoints": [invalid]}})

    parsed = parse_openai(
        {
            "provider": "Venice",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "cost": 0.00002,
                "cost_details": {"upstream_inference_cost": 0.000019},
            },
        }
    )
    assert parsed["cost_usd"] == pytest.approx(0.00002)
    assert parsed["cost_details"]["upstream_inference_cost"] == pytest.approx(
        0.000019
    )


def test_v3_preflight_fails_before_model_requests_for_invalid_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AuthFailureResponse:
        status_code = 401

        @staticmethod
        def json() -> dict[str, object]:
            return {"error": {"message": "User not found."}}

    class AuthFailureSession:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def get(self, url: str, **kwargs: object) -> AuthFailureResponse:
            self.calls.append((url, kwargs))
            return AuthFailureResponse()

        def post(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("model request must not run after failed auth check")

    session = AuthFailureSession()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-invalid-key")
    route = {
        "subject_id": "qwen3.6-35b-a3b",
        "family": "qwen",
        "label": "qwen3.6-35b-a3b::openrouter",
        "route_type": "openrouter",
        "auth_env": "OPENROUTER_API_KEY",
        "extra_headers": {},
    }
    with pytest.raises(RuntimeError, match=r"credential preflight failed HTTP 401"):
        validate_openrouter_auth(session, [route])
    assert len(session.calls) == 1
    assert session.calls[0][0] == "https://openrouter.ai/api/v1/auth/key"


def test_preflight_load_env_accepts_shell_export_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "export OPENROUTER_API_KEY = 'local-openrouter-key'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("export OPENROUTER_API_KEY", raising=False)

    load_env(env_path)

    assert os.environ["OPENROUTER_API_KEY"] == "local-openrouter-key"
    assert "export OPENROUTER_API_KEY" not in os.environ


def _preflight_token_row(
    prompt_case_id: str,
    mode: str,
    prompt_tokens: int,
    *,
    finish_reason: str = "stop",
) -> dict[str, object]:
    return {
        "route_label": "subject::openrouter::provider",
        "prompt_case_id": prompt_case_id,
        "mode": mode,
        "http_status": 200,
        "prompt_tokens": prompt_tokens,
        "finish_reason": finish_reason,
    }


def test_preflight_classification_rejects_prompt_dependent_schema_visibility() -> None:
    rows = [
        _preflight_token_row("visible", "NOTOOLS", 10),
        _preflight_token_row("visible", "DISABLED", 210),
        _preflight_token_row("visible", "TOOL", 210),
        _preflight_token_row("hidden", "NOTOOLS", 10),
        _preflight_token_row("hidden", "DISABLED", 10),
        _preflight_token_row("hidden", "TOOL", 210),
    ]

    result = classify(rows)["subject::openrouter::provider"]

    assert result["disabled_arm_schema_visibility"] == "mixed_visibility"
    assert result["per_prompt_classification"] == {
        "hidden": "schema_hidden",
        "visible": "schema_visible",
    }


def test_preflight_classification_rejects_http_200_finish_errors() -> None:
    rows = [
        _preflight_token_row("valid", "NOTOOLS", 10),
        _preflight_token_row("valid", "DISABLED", 210),
        _preflight_token_row("valid", "TOOL", 210),
        _preflight_token_row("error", "NOTOOLS", 0, finish_reason="error"),
        _preflight_token_row("error", "DISABLED", 0, finish_reason="error"),
        _preflight_token_row("error", "TOOL", 210),
    ]

    result = classify(rows)["subject::openrouter::provider"]

    assert result["disabled_arm_schema_visibility"] == "unsupported"
    assert result["per_prompt_classification"]["error"] == "unsupported"


def test_v3_gpt_route_rewrites_model_and_applies_declared_decoding() -> None:
    protocol = json.loads(V3_PROTOCOL_PATH.read_text(encoding="utf-8"))
    source = _source_row("DISABLED")
    source["subject_id"] = "gpt-5.5"
    source["custom_id"] = source["custom_id"].replace(
        "claude-opus-4-7", "gpt-5.5"
    )
    source["body"]["model"] = "gpt-5.5"
    source["body"]["max_completion_tokens"] = source["body"].pop("max_tokens")

    row = _derive_row(
        source,
        arm_name="DISABLED_BASELINE",
        protocol=protocol,
        protocol_sha256="b" * 64,
        source_batch_sha256="a" * 64,
        canary=False,
    )
    assert row["route"] == "openrouter"
    assert row["body"]["model"] == "openai/gpt-5.5"
    assert row["body"]["max_tokens"] == 4096
    assert "temperature" not in row["body"]
    assert "top_p" not in row["body"]
    assert "max_completion_tokens" not in row["body"]
    assert row["pinned_provider"] == "openai/flex"


def test_openrouter_v3_record_rejects_observed_provider_mismatch() -> None:
    row = {
        "custom_id": "v3-control-row",
        "subject_id": "gemma-4-31b-it",
        "protocol_id": "CAGAP-POSTREG-DISABLED-FORMAT-2026-09-01-v3",
        "require_observed_provider_match": True,
        "mode": "DISABLED",
        "body": {},
    }
    body = {"model": "google/gemma-4-31b-it", "messages": []}
    response = {
        "provider": "DeepInfra",
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
    }

    with pytest.raises(ValueError, match="observed OpenRouter provider mismatch"):
        build_control_openrouter_run_record(
            row=row,
            body=body,
            response=response,
            pinned_provider="venice",
            latency_sec=0.1,
        )


def test_openrouter_v3_record_preserves_matching_observed_provider() -> None:
    row = {
        "custom_id": "v3-control-row",
        "subject_id": "gemma-4-31b-it",
        "protocol_id": "CAGAP-POSTREG-DISABLED-FORMAT-2026-09-01-v3",
        "require_observed_provider_match": True,
        "mode": "DISABLED",
        "body": {},
    }
    body = {"model": "google/gemma-4-31b-it", "messages": []}
    response = {
        "provider": "Venice",
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
    }
    record = build_control_openrouter_run_record(
        row=row,
        body=body,
        response=response,
        pinned_provider="venice",
        latency_sec=0.1,
    )
    assert record["pinned_provider"] == "venice"
    assert record["observed_provider"] == "Venice"
    assert record["observed_provider_matches_pin"] is True
    assert record["protocol_id"] == "CAGAP-POSTREG-DISABLED-FORMAT-2026-09-01-v3"
    assert record["require_observed_provider_match"] is True


def test_nan_cost_gate_fails_before_submission() -> None:
    protocol, _ = load_protocol(PROTOCOL_PATH)
    args = Namespace(
        protocol=PROTOCOL_PATH,
        subject=["claude-opus-4-7"],
        acknowledge_postregistration_exploratory=True,
        confirm_spend=protocol["protocol_id"],
        max_requests=2,
        max_cost_usd=float("nan"),
    )
    with pytest.raises(ValueError, match="finite non-negative"):
        submit_generated(args)
