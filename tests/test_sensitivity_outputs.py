from __future__ import annotations

import json
import inspect
import subprocess

import pandas as pd
import pytest

import scripts.analyze_results as analyze_results_module
from scripts.analyze_results import (
    _field_status,
    _paired_delta_by_subject,
    build_intervention_h2,
    build_determinism_icc_report,
    build_leave_provider_served_models_out_e1,
    build_model_dependence_robustness,
    build_primary_h1_results,
    build_predicate_override_removal,
    build_repair_round_exclusion,
    build_sen_1_ordinal_median_robustness,
    build_sen_2_judge_alone,
    build_sen_8_dawid_skene,
    build_sen_j_judge_restrictions,
    build_sen_m_measurement_pathway,
    build_sen_h_hardened_intervention,
    build_sen_9a_relaxed_inclusion,
    build_sen_9b_unresolved_as_refuse,
    build_sen_9c_unresolved_as_comply,
    build_sen_9d_tipping_point,
    build_sen_9e_differential_worst_case,
    build_sen_9_summary,
    build_sen_q_quantization_audit,
    build_sen_secondary_summary,
    build_sen_d_dispatch_block,
    build_schema_visibility_route_tally,
    build_substituted_reserve_sensitivities,
    build_tool_persona_interaction,
    build_prereg_section6_contract_ledger,
    build_tier0_repair0_intersection,
    build_validator_tier_exclusion,
)

EventTriple = tuple[int | None, int | None, int | None]


def _label_for_event(event: int | None) -> str:
    if event is None:
        return "Unresolved"
    return "Comply" if event else "Refuse"


def _sensitivity_row(
    *,
    prompt_id: str,
    subject_id: str,
    persona: str,
    trial_id: int,
    mode: str,
    event: int | None,
    route: str = "native",
    intervention_condition: str = "baseline",
    serving_precision_role: str = "primary",
    subject_tier: str = "hosted_api",
    grading_audit: dict | None = None,
) -> dict:
    unresolved = event is None
    row = {
        "prompt_id": prompt_id,
        "case_id": prompt_id,
        "subject_id": subject_id,
        "subject_tier": subject_tier,
        "persona": persona,
        "domain": "GridSim",
        "partition_id": "main",
        "trial_id": trial_id,
        "intervention_condition": intervention_condition,
        "serving_precision_role": serving_precision_role,
        "mode": mode,
        "final_label": _label_for_event(event),
        "final_label_status": "unresolved" if unresolved else "resolved",
        "analysis_inclusion": not unresolved,
        "final_label_source": "missingness" if unresolved else "judge_consensus",
        "missingness_reason": "subject_output_failure" if unresolved else None,
        "dispatch_block_id": f"{subject_id}:{prompt_id}:t{trial_id}",
        "timestamp": f"2026-05-01T04:{trial_id:02d}:00Z",
        "route": route,
    }
    if grading_audit is not None:
        row["grading_audit"] = grading_audit
    return row


def _cluster_rows(
    *,
    cluster_idx: int,
    tool_events: tuple[int | None, int | None, int | None],
    disabled_events: tuple[int | None, int | None, int | None],
    persona: str = "PERSONA_NEUTRAL",
    route: str = "native",
) -> list[dict]:
    prompt_id = f"prompt_{cluster_idx:03d}"
    subject_id = f"model_{cluster_idx % 3}"
    rows = []
    for trial_id, event in enumerate(disabled_events):
        rows.append(
            _sensitivity_row(
                prompt_id=prompt_id,
                subject_id=subject_id,
                persona=persona,
                trial_id=trial_id,
                mode="DISABLED",
                event=event,
                route=route,
            )
        )
    for trial_id, event in enumerate(tool_events):
        rows.append(
            _sensitivity_row(
                prompt_id=prompt_id,
                subject_id=subject_id,
                persona=persona,
                trial_id=trial_id,
                mode="TOOL",
                event=event,
                route=route,
            )
        )
    return rows


def _missingness_fixture_rows() -> list[dict]:
    rows: list[dict] = []
    for cluster_idx in range(36):
        tool_events: EventTriple
        disabled_events: EventTriple
        if cluster_idx == 0:
            tool_events = (1, 1, None)
            disabled_events = (0, 1, None)
        elif cluster_idx < 28:
            tool_events = (1, 1, 0)
            disabled_events = (1, 0, 0)
        else:
            tool_events = (1, 0, 0)
            disabled_events = (1, 1, 0)
        cluster = _cluster_rows(
            cluster_idx=cluster_idx,
            tool_events=tool_events,
            disabled_events=disabled_events,
        )
        for row in cluster:
            row["dispatch_block_id"] = f"{row['subject_id']}:sensitivity_batch_{cluster_idx % 2}:t0"
        rows.extend(cluster)
    return rows


def _sen_h_fixture_rows() -> list[dict]:
    rows: list[dict] = []
    for cluster_idx in range(36):
        if cluster_idx < 28:
            hardened_events = (1, 1, 0)
            disabled_events = (1, 0, 0)
        else:
            hardened_events = (1, 0, 0)
            disabled_events = (1, 1, 0)
        prompt_id = f"prompt_{cluster_idx:03d}"
        subject_id = f"model_{cluster_idx % 3}"
        for trial_id, event in enumerate(disabled_events):
            rows.append(
                _sensitivity_row(
                    prompt_id=prompt_id,
                    subject_id=subject_id,
                    persona="PERSONA_NEUTRAL",
                    trial_id=trial_id,
                    mode="DISABLED",
                    event=event,
                )
            )
        for trial_id, event in enumerate(hardened_events):
            rows.append(
                _sensitivity_row(
                    prompt_id=prompt_id,
                    subject_id=subject_id,
                    persona="PERSONA_NEUTRAL",
                    trial_id=trial_id,
                    mode="TOOL",
                    event=event,
                    intervention_condition="hardened",
                )
            )
        for trial_id in range(3):
            rows.append(
                _sensitivity_row(
                    prompt_id=prompt_id,
                    subject_id=subject_id,
                    persona="PERSONA_NEUTRAL",
                    trial_id=trial_id,
                    mode="TOOL",
                    event=0,
                )
            )
    return rows


def _sen_q_fixture_rows() -> list[dict]:
    rows: list[dict] = []
    local_sen_q_subjects = (
        "llama-3.3-70b-instruct",
        "gemma-4-31b-it",
        "qwen3.6-35b-a3b",
        "glm-4.7-flash",
        "mistral-medium-3.5-128b",
    )
    for subject_idx, subject_id in enumerate(local_sen_q_subjects):
        prompt_id = f"prompt_q_{subject_idx}"
        for trial_id in range(3):
            primary_event = 1 if trial_id == 0 else 0
            quantized_event = 1 if trial_id in {0, 1} else 0
            rows.append(
                _sensitivity_row(
                    prompt_id=prompt_id,
                    subject_id=subject_id,
                    subject_tier="local_weight",
                    persona="PERSONA_NEUTRAL",
                    trial_id=trial_id,
                    mode="TOOL",
                    event=primary_event,
                    grading_audit={
                        "structured_tool_call": trial_id == 0,
                        "provider_tool_call_count": 1 if trial_id == 0 else 0,
                    },
                )
            )
            rows.append(
                _sensitivity_row(
                    prompt_id=prompt_id,
                    subject_id=subject_id,
                    subject_tier="local_weight",
                    persona="PERSONA_NEUTRAL",
                    trial_id=trial_id,
                    mode="TOOL",
                    event=quantized_event,
                    serving_precision_role="quantization_audit",
                    grading_audit={
                        "structured_tool_call": trial_id in {0, 1},
                        "provider_tool_call_count": 1 if trial_id in {0, 1} else 0,
                    },
                )
            )
    return rows


def _assert_h1_shape(payload: dict, resolution_path: str) -> None:
    assert payload["resolution_path"] == resolution_path
    assert set(payload["tests"]) == {
        "h1a_erosion",
        "h1a_full_asr",
        "h1b_erosion",
        "h1b_full_asr",
    }
    h1a = payload["tests"]["h1a_erosion"]
    assert h1a["resolution_path"] == resolution_path
    assert h1a["retained_cluster_count"] >= 35
    assert h1a["informative_cluster_count"] >= 35
    assert h1a["or"] is not None
    assert 0.1 < h1a["or"] < 100.0
    assert h1a["p_value"] is not None
    assert h1a["p_holm"] is not None


def test_sen_9_outputs_apply_each_missingness_rule():
    df = pd.DataFrame(_missingness_fixture_rows())

    sen_9a = build_sen_9a_relaxed_inclusion(df, bootstrap_iter=10)
    sen_9b = build_sen_9b_unresolved_as_refuse(df, bootstrap_iter=10)
    sen_9c = build_sen_9c_unresolved_as_comply(df, bootstrap_iter=10)
    sen_9d = build_sen_9d_tipping_point(df, bootstrap_iter=10)
    sen_9e = build_sen_9e_differential_worst_case(df, bootstrap_iter=10)

    _assert_h1_shape(sen_9a, "sen_9a_relaxed_2_of_3_per_arm")
    _assert_h1_shape(sen_9b, "sen_9b_unresolved_as_refuse")
    _assert_h1_shape(sen_9c, "sen_9c_unresolved_as_comply")
    _assert_h1_shape(sen_9e, "sen_9e_differential_worst_case_against_h1")
    assert sen_9b["imputation"]["imputed_rows"] == 2
    assert sen_9b["imputation"]["final_label_source"] == "imputed_unresolved_refuse"
    assert sen_9c["imputation"]["final_label_source"] == "imputed_unresolved_comply"
    assert sen_9e["imputation"] == {
        "analysis_role": "post_trigger_conservative_bound_not_h1_estimator",
        "imputed_rows": 2,
        "final_label_source": "imputed_unresolved_differential_worst_case",
        "by_mode": {
            "DISABLED": {"imputed_rows": 1, "final_label": "Comply"},
            "TOOL": {"imputed_rows": 1, "final_label": "Refuse"},
        },
        "rule": "DISABLED unresolved -> Comply; TOOL unresolved -> Refuse",
    }

    records = {record["endpoint"]: record for record in sen_9d["records"]}
    assert set(records) == {
        "h1a_erosion",
        "h1a_full_asr",
        "h1b_erosion",
        "h1b_full_asr",
    }
    h1a_tip = records["h1a_erosion"]
    assert h1a_tip["flip_direction"] in {"tool_to_disabled", "disabled_to_tool"}
    assert isinstance(h1a_tip["feasible"], bool)
    if h1a_tip["feasible"]:
        assert 0.0 <= h1a_tip["min_fraction_to_flip"] <= 1.0


def test_sen_1_replays_ordinal_median_three_way_split_rules():
    rows = _missingness_fixture_rows()
    for row in rows[:6]:
        row["judge_a"] = "Refuse"
        row["judge_b"] = "Partial"
        row["judge_c"] = "Comply"
        row["final_label"] = "Partial"
        row["final_label_source"] = "judge_ensemble"

    result = build_sen_1_ordinal_median_robustness(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "ok"
    assert result["ordinal_median_rows"] == 6
    assert set(result["rules"]) == {
        "complete_case_exclude_ordinal_median",
        "impute_refuse",
        "impute_comply",
    }
    assert result["rules"]["complete_case_exclude_ordinal_median"]["tests"][
        "h1a_erosion"
    ]["retained_cluster_count"] == 35
    assert result["rules"]["impute_refuse"]["tests"]["h1a_erosion"][
        "resolution_path"
    ] == "sen_1_impute_refuse"
    assert result["rules"]["impute_comply"]["tests"]["h1a_full_asr"]["p_holm"] is not None


@pytest.mark.slow
def test_sen_2_replays_primary_under_each_judge_alone():
    rows = _missingness_fixture_rows()
    for row in rows:
        row["judge_a"] = row["final_label"]
        row["judge_b"] = "Comply" if row["mode"] == "TOOL" else "Refuse"
        row["judge_c"] = "Refuse"

    result = build_sen_2_judge_alone(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "ok"
    assert set(result["judges"]) == {"judge_a", "judge_b", "judge_c"}
    assert result["judges"]["judge_a"]["rows_replaced"] > 0
    assert result["judges"]["judge_b"]["h1_refit"]["tests"]["h1a_erosion"]["p_holm"] is not None
    assert result["judges"]["judge_c"]["h1_refit"]["resolution_path"] == "sen_2_judge_c_alone"


@pytest.mark.slow
def test_sen_2_reports_named_judge_alone_separately_from_slots():
    rows = _missingness_fixture_rows()
    for row in rows:
        row["judge_a"] = row["final_label"]
        row["judge_b"] = row["final_label"]
        row["judge_c"] = row["final_label"]
        row["judge_slot_1_model_id"] = "glm-5.1"
        row["judge_slot_2_model_id"] = "command-a-03-2025"
        row["judge_slot_3_model_id"] = (
            "gpt-5.5-judge"
            if row["subject_id"] == "model_2"
            else "mistral-large-2512"
        )

    result = build_sen_2_judge_alone(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "ok"
    assert set(result["judges"]) == {"judge_a", "judge_b", "judge_c"}
    assert {
        "glm-5.1",
        "command-a-03-2025",
        "mistral-large-2512",
        "gpt-5.5-judge",
    }.issubset(result["named_judges"])

    slot_c = result["judges"]["judge_c"]
    mistral = result["named_judges"]["mistral-large-2512"]
    gpt_sub = result["named_judges"]["gpt-5.5-judge"]
    assert slot_c["rows_replaced"] > mistral["rows_replaced"]
    assert slot_c["rows_replaced"] > gpt_sub["rows_replaced"]
    assert mistral["judge_path_rows_marked_unresolved"] > 0
    assert gpt_sub["judge_path_rows_marked_unresolved"] > 0
    assert mistral["h1_refit"]["resolution_path"] == "sen_2_named_mistral_large_2512_alone"
    assert gpt_sub["h1_refit"]["resolution_path"] == "sen_2_named_gpt_5_5_judge_alone"


@pytest.mark.slow
def test_sen_2_hydrates_slots_from_saved_judge_votes():
    rows = _missingness_fixture_rows()
    for row in rows:
        if row["final_label"] == "Unresolved":
            continue
        row["judge_votes"] = [
            {"judge": "glm-5.1", "label": row["final_label"]},
            {
                "judge": "command-a-03-2025",
                "label": "Comply" if row["mode"] == "TOOL" else "Refuse",
            },
            {"judge": "mistral-large-2512", "label": "Refuse"},
        ]

    result = build_sen_2_judge_alone(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "ok"
    assert set(result["judges"]) == {"judge_a", "judge_b", "judge_c"}
    assert result["judges"]["judge_a"]["rows_replaced"] > 0
    assert {
        "glm-5.1",
        "command-a-03-2025",
        "mistral-large-2512",
    }.issubset(result["named_judges"])


@pytest.mark.slow
def test_sen_j_judge_restrictions_materialize_all_three_and_conflict_refits():
    rows = _missingness_fixture_rows()
    for row in rows:
        row["final_label"] = "Partial"
        row["final_label_status"] = "resolved"
        row["analysis_inclusion"] = True
        row["final_label_source"] = "judge_ensemble"
        row["judge_a"] = "Refuse"
        row["judge_b"] = "Partial"
        row["judge_c"] = "Comply"
        row["judge_slot_1_model_id"] = "strict-judge"
        row["judge_slot_2_model_id"] = "middle-judge"
        row["judge_slot_3_model_id"] = "lenient-judge"
        row["judge_family_exclusion_applied"] = row["subject_id"] == "model_1"
        row["substitute_judge_used"] = (
            "gpt-5.5-judge" if row["subject_id"] == "model_1" else None
        )
    rows[0]["judge_c"] = None

    result = build_sen_j_judge_restrictions(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "ok"
    all_three = result["all_three_valid_judge_restriction"]
    assert all_three["status"] == "ok"
    assert all_three["judge_path_rows"] == len(rows)
    assert all_three["judge_path_rows_marked_unresolved"] == 1
    assert all_three["h1_refit"]["tests"]["h1a_erosion"]["resolution_path"] == (
        "sen_j_all_three_valid_judge_restriction"
    )
    conflict = result["conflict_subject_excluded"]
    assert conflict["status"] == "ok"
    assert conflict["excluded_subjects"] == ["model_1"]
    assert conflict["h1_refit"]["tests"]["h1a_full_asr"]["resolution_path"] == (
        "sen_j_conflict_subject_excluded"
    )
    strictest = result["strictest_binding_judge_excluded"]
    assert strictest["status"] == "ok"
    assert strictest["strictest_judge_id"] == "strict-judge"
    assert strictest["rows_with_strictest_judge"] == len(rows)
    assert strictest["h1_refit"]["tests"]["h1a_erosion"]["resolution_path"] == (
        "sen_j_strictest_binding_judge_excluded"
    )


def test_substituted_reserve_sensitivities_emit_explicit_zero_substitution_closure():
    rows = _missingness_fixture_rows()
    for row in rows:
        row["is_substituted"] = False
        row["source_partition_id"] = "main"
        row["replacement_for"] = None

    result = build_substituted_reserve_sensitivities(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "not_applicable_zero_substitutions"
    assert result["substituted_baseline_row_count"] == 0
    assert result["substituted_prompt_count"] == 0
    assert result["refit_excluding_substituted_reserves"]["would_equal_primary"] is True
    assert result["tool_by_substitution_interaction"]["status"] == "not_applicable"
    assert result["dispatch_block_balance_audit"]["substituted_dispatch_block_count"] == 0


def test_schema_visibility_route_tally_reports_subject_level_gate_pass():
    rows = _missingness_fixture_rows()
    for row in rows:
        row["disabled_arm_schema_visibility"] = "schema_visible"
        row["route_support_status"] = "supported"
        row["route"] = "native"
        row["provider"] = "provider-x"

    result = build_schema_visibility_route_tally(pd.DataFrame(rows))

    assert result["status"] == "ok"
    assert result["subject_count"] == 3
    assert result["eligible_subject_count"] == 3
    assert result["all_subjects_eligible"] is True
    assert all(subject["eligible"] for subject in result["subjects"].values())


def test_leave_provider_served_models_out_e1_excludes_non_local_subjects():
    rows = []
    for cluster_idx in range(36):
        cluster = _cluster_rows(
            cluster_idx=cluster_idx,
            tool_events=(1, 1, 0),
            disabled_events=(1, 0, 0),
        )
        for row in cluster:
            row["provider"] = "local" if row["subject_id"] != "model_1" else "openai"
        rows.extend(cluster)

    result = build_leave_provider_served_models_out_e1(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "ok"
    assert result["excluded_subjects"] == ["model_1"]
    assert result["retained_subject_count"] == 2
    assert result["h1_refit"]["tests"]["h1a_erosion"]["resolution_path"] == (
        "leave_provider_served_models_out"
    )


def test_sen_8_dawid_skene_inferrs_latent_labels_and_refits_h1():
    rows = _missingness_fixture_rows()
    for row in rows:
        row["judge_a"] = row["final_label"]
        row["judge_b"] = row["final_label"]
        row["judge_c"] = "Partial" if row["final_label"] == "Comply" else row["final_label"]

    result = build_sen_8_dawid_skene(pd.DataFrame(rows), bootstrap_iter=0, max_iter=25)

    assert result["status"] == "ok"
    assert result["dawid_skene"]["n_rows_inferred"] == len(rows) - 2
    assert set(result["dawid_skene"]["confusion_matrices"]) == {
        "judge_a",
        "judge_b",
        "judge_c",
    }
    assert result["h1_refit"]["tests"]["h1a_erosion"]["resolution_path"] == "sen_8_dawid_skene"
    assert result["h1_refit"]["tests"]["h1a_full_asr"]["p_holm"] is not None


def test_sen_8_hydrates_complete_observations_from_saved_judge_votes():
    rows = _missingness_fixture_rows()
    for row in rows:
        if row["final_label"] == "Unresolved":
            continue
        row["judge_votes"] = [
            {"judge": "glm-5.1", "label": row["final_label"]},
            {"judge": "command-a-03-2025", "label": row["final_label"]},
            {
                "judge": "mistral-large-2512",
                "label": "Partial" if row["final_label"] == "Comply" else row["final_label"],
            },
        ]

    result = build_sen_8_dawid_skene(pd.DataFrame(rows), bootstrap_iter=0, max_iter=25)

    assert result["status"] == "ok"
    assert result["dawid_skene"]["n_rows_inferred"] == len(rows) - 2


def test_predicate_override_removal_uses_shadow_judge_labels():
    rows = _missingness_fixture_rows()
    predicate_rows = 0
    for row in rows:
        row["judge_a"] = row["final_label"]
        row["judge_b"] = row["final_label"]
        row["judge_c"] = "Refuse"
        if row["mode"] == "TOOL" and predicate_rows < 6:
            row["final_label_source"] = "predicate"
            row["final_label"] = "Comply"
            row["judge_a"] = "Refuse"
            row["judge_b"] = "Refuse"
            row["judge_c"] = "Partial"
            predicate_rows += 1

    result = build_predicate_override_removal(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "ok"
    assert result["predicate_rows"] == 6
    assert result["rows_replaced_with_shadow_judge_label"] == 6
    assert result["h1_refit"]["tests"]["h1a_full_asr"]["resolution_path"] == (
        "predicate_override_removed"
    )


def test_corpus_construction_sensitivities_exclude_tier1_and_repaired_prompts():
    rows = _missingness_fixture_rows()
    for cluster_idx, start in enumerate(range(0, len(rows), 6)):
        for row in rows[start:start + 6]:
            row["validator_tier"] = (
                "tier1_2of3_tiebreaker" if cluster_idx < 2 else "tier0_dual_pass"
            )
            row["repair_round"] = 1 if 2 <= cluster_idx < 4 else 0

    df = pd.DataFrame(rows)
    tier = build_validator_tier_exclusion(df, bootstrap_iter=0)
    repair = build_repair_round_exclusion(df, bootstrap_iter=0)
    intersection = build_tier0_repair0_intersection(df, bootstrap_iter=0)

    assert tier["status"] == "ok"
    assert tier["excluded_prompt_count"] == 2
    assert tier["h1_refit"]["tests"]["h1a_erosion"]["resolution_path"] == (
        "validator_tier_tier0_only"
    )
    assert repair["excluded_prompt_count"] == 2
    assert intersection["excluded_prompt_count"] == 4
    assert intersection["h1_refit"]["tests"]["h1a_full_asr"]["p_holm"] is not None


def test_sen_m_measurement_pathway_reports_rates_shadow_agreement_and_refit():
    rows = _missingness_fixture_rows()
    shadowed = 0
    for row in rows:
        row["subject_response"] = f"{row['prompt_id']} {row['mode']} trial {row['trial_id']}"
        if row["final_label_status"] == "unresolved":
            row["final_label_source"] = "extractor_error"
        elif row["mode"] == "TOOL" and shadowed < 8:
            row["final_label_source"] = "predicate"
            row["judge_a"] = row["final_label"]
            row["judge_b"] = row["final_label"]
            row["judge_c"] = "Refuse"
            shadowed += 1
        elif row["mode"] == "DISABLED" and shadowed < 12:
            row["final_label_source"] = "deterministic_clear_refusal"
            row["judge_a"] = "Refuse"
            row["judge_b"] = "Refuse"
            row["judge_c"] = "Partial"
            shadowed += 1
        else:
            row["final_label_source"] = "judge_ensemble"

    result = build_sen_m_measurement_pathway(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["status"] == "ok"
    assert result["scope"] == "exploratory_diagnostic_not_holm"
    assert result["pathway_rates_by_arm"]["TOOL"]["predicate_graded"] > 0
    assert result["pathway_rates_by_arm"]["DISABLED"]["clear_refusal_bypass"] > 0
    assert result["shadow_agreement"]["agree_n"] > 0
    assert result["agreement_subset_h1_refit"]["tests"]["h1a_full_asr"]["or"] is not None
    assert result["shadow_h1_refit"]["tests"]["h1a_full_asr"]["or"] is not None


def test_determinism_icc_report_includes_duplicate_rates_icc_and_trial_estimates():
    rows = []
    for mode, responses in {
        "TOOL": ("same", "same", "same"),
        "DISABLED": ("r0", "r1", "r2"),
    }.items():
        for trial_id, response in enumerate(responses):
            rows.append(
                _sensitivity_row(
                    prompt_id="prompt_dup",
                    subject_id="model_dup",
                    persona="PERSONA_NEUTRAL",
                    trial_id=trial_id,
                    mode=mode,
                    event=1 if mode == "TOOL" else 0,
                )
            )
            rows[-1]["subject_response"] = response
    for cluster_idx in range(35):
        rows.extend(
            _cluster_rows(
                cluster_idx=cluster_idx,
                tool_events=(1, 1, 0),
                disabled_events=(1, 0, 0),
            )
        )

    result = build_determinism_icc_report(pd.DataFrame(rows), bootstrap_iter=0)

    assert result["exact_response_duplicate_rate"] is not None
    assert result["label_duplicate_rate"]["E1_erosion"] is not None
    assert "within_cell_icc_e1" in result
    assert "within_cell_icc_e2" in result
    assert set(result["trial_only_estimates"]["E2_full_asr"]) == {
        "trial_0",
        "trial_1",
        "trial_2",
    }


@pytest.mark.slow
def test_model_dependence_robustness_reports_leave_one_out_and_forest_data():
    result = build_model_dependence_robustness(
        pd.DataFrame(_missingness_fixture_rows()),
        bootstrap_iter=0,
        subject_bootstrap_iter=64,
    )

    h1a = result["tests"]["h1a_full_asr"]
    assert set(h1a) == {
        "leave_one_model_out",
        "per_model_risk_difference",
        "subject_dependence_robustness",
    }
    assert h1a["leave_one_model_out"]
    assert h1a["per_model_risk_difference"]
    subject_robustness = h1a["subject_dependence_robustness"]
    assert subject_robustness["status"] == "ok"
    assert subject_robustness["method"] == "subject_mean_rademacher_wild_cluster_bootstrap"
    assert subject_robustness["bootstrap_iterations"] == 64
    assert subject_robustness["n_subjects"] == 3
    assert subject_robustness["p_value_two_sided"] is not None
    assert subject_robustness["comparison_to_primary"]["dependence_sensitive"] in {
        True,
        False,
    }


def test_tool_persona_interaction_reports_exploratory_clogit_term():
    rows = []
    for cluster_idx in range(36):
        rows.extend(
            _cluster_rows(
                cluster_idx=cluster_idx,
                persona="PERSONA_NEUTRAL",
                tool_events=(1, 1, 0),
                disabled_events=(1, 0, 0),
            )
        )
        rows.extend(
            _cluster_rows(
                cluster_idx=cluster_idx + 100,
                persona="PERSONA_OPERATOR",
                tool_events=(1, 1, 1),
                disabled_events=(1, 0, 0),
            )
        )

    result = build_tool_persona_interaction(pd.DataFrame(rows))

    assert result["scope"] == "exploratory_not_holm"
    assert result["tests"]["h1a_full_asr"]["term"] == "tool_x_persona_operator"
    assert result["tests"]["h1a_full_asr"]["p_value"] is not None


def test_sen_9a_keeps_two_of_three_per_arm_cluster_rejected_by_primary():
    df = pd.DataFrame(_missingness_fixture_rows())

    primary = build_primary_h1_results(df, bootstrap_iter=10)
    sen_9a = build_sen_9a_relaxed_inclusion(df, bootstrap_iter=10)

    primary_h1a = primary["tests"]["h1a_erosion"]
    relaxed_h1a = sen_9a["tests"]["h1a_erosion"]
    assert primary_h1a["resolution_path"] == "primary_6_of_6_complete_case"
    assert relaxed_h1a["resolution_path"] == "sen_9a_relaxed_2_of_3_per_arm"
    assert primary_h1a["retained_cluster_count"] == 35
    assert relaxed_h1a["retained_cluster_count"] == 36
    assert relaxed_h1a["n_rows"] == primary_h1a["n_rows"] + 4


def test_sen_9b_and_9c_imputed_labels_enter_cluster_delta():
    rows = _missingness_fixture_rows()
    rows.extend(
        _cluster_rows(
            cluster_idx=900,
            tool_events=(1, None, None),
            disabled_events=(0, 0, 0),
        )
    )
    df = pd.DataFrame(rows)

    sen_9b = build_sen_9b_unresolved_as_refuse(df, bootstrap_iter=10)
    sen_9c = build_sen_9c_unresolved_as_comply(df, bootstrap_iter=10)
    h1b_refuse = sen_9b["tests"]["h1a_full_asr"]
    h1b_comply = sen_9c["tests"]["h1a_full_asr"]

    assert sen_9b["imputation"]["imputed_rows"] == 4
    assert sen_9c["imputation"]["imputed_rows"] == 4
    assert h1b_refuse["retained_cluster_count"] == h1b_comply["retained_cluster_count"]
    assert h1b_refuse["delta_cluster_distribution"]["1"] == (
        h1b_comply["delta_cluster_distribution"]["1"] + 1
    )
    assert h1b_comply["delta_cluster_distribution"]["3"] == 1


def test_sen_9d_reports_no_tipping_point_without_unresolved_rows():
    rows = []
    for cluster_idx in range(36):
        rows.extend(
            _cluster_rows(
                cluster_idx=cluster_idx,
                tool_events=(1, 1, 0),
                disabled_events=(1, 0, 0),
            )
        )

    sen_9d = build_sen_9d_tipping_point(pd.DataFrame(rows), bootstrap_iter=10)

    assert sen_9d["resolution_path"] == "sen_9d_tipping_point"
    for record in sen_9d["records"]:
        assert record["unresolved_rows"] == 0
        assert record["feasible"] is False
        assert record["min_fraction_to_flip"] is None


def test_sen_9d_reports_primary_and_candidate_decision_metadata_deterministically():
    df = pd.DataFrame(_missingness_fixture_rows())

    first = build_sen_9d_tipping_point(df, bootstrap_iter=10, bootstrap_seed=20260510)
    second = build_sen_9d_tipping_point(df, bootstrap_iter=10, bootstrap_seed=20260510)

    assert first == second
    for record in first["records"]:
        assert "primary_or" in record
        assert "primary_p_holm" in record
        assert "primary_holm_reject" in record
        assert "candidate_or_at_flip" in record
        assert "candidate_p_holm_at_flip" in record
        assert "candidate_holm_reject_at_flip" in record
        if record["feasible"]:
            assert record["candidate_or_at_flip"] is not None
            assert record["candidate_p_holm_at_flip"] is not None


def test_default_tipping_worker_count_leaves_two_cores(monkeypatch):
    monkeypatch.setattr(analyze_results_module.os, "cpu_count", lambda: 8)
    assert analyze_results_module._default_tipping_worker_count() == 6

    monkeypatch.setattr(analyze_results_module.os, "cpu_count", lambda: 2)
    assert analyze_results_module._default_tipping_worker_count() == 1

    monkeypatch.setattr(analyze_results_module.os, "cpu_count", lambda: None)
    assert analyze_results_module._default_tipping_worker_count() == 1


def test_default_analysis_worker_count_leaves_two_cores(monkeypatch):
    monkeypatch.setattr(analyze_results_module.os, "cpu_count", lambda: 12)
    assert analyze_results_module._default_analysis_worker_count() == 10

    monkeypatch.setattr(analyze_results_module.os, "cpu_count", lambda: 2)
    assert analyze_results_module._default_analysis_worker_count() == 1


def test_parallel_analysis_job_runner_matches_serial():
    jobs = [
        ("slug_a", "_safe_analysis_slug", ("Model A",), {}),
        ("slug_b", "_safe_analysis_slug", ("GLM-5.1/Judge",), {}),
    ]

    serial = analyze_results_module._run_analysis_jobs(jobs, workers=1)
    parallel = analyze_results_module._run_analysis_jobs(jobs, workers=2)

    assert serial == parallel == {
        "slug_a": "model_a",
        "slug_b": "glm_5_1_judge",
    }


def test_cached_json_artifact_requires_matching_done_manifest(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    signature = {"inputs_sha256": "abc", "bootstrap_iter": 10}
    other_signature = {"inputs_sha256": "different", "bootstrap_iter": 10}

    analyze_results_module._write_cached_json_artifact(
        artifact_path,
        {"value": 7},
        run_signature=signature,
    )

    assert analyze_results_module._load_cached_json_artifact(
        artifact_path,
        run_signature=signature,
    ) == {"value": 7}
    assert analyze_results_module._load_cached_json_artifact(
        artifact_path,
        run_signature=other_signature,
    ) is analyze_results_module._CACHE_MISS


def test_get_or_build_json_artifact_skips_builder_on_matching_cache(tmp_path):
    artifact_path = tmp_path / "artifact.json"
    signature = {"inputs_sha256": "abc", "bootstrap_iter": 10}
    calls = 0

    def build_once():
        nonlocal calls
        calls += 1
        return {"value": calls}

    first = analyze_results_module._get_or_build_json_artifact(
        artifact_path,
        run_signature=signature,
        resume=True,
        builder=build_once,
    )
    second = analyze_results_module._get_or_build_json_artifact(
        artifact_path,
        run_signature=signature,
        resume=True,
        builder=lambda: pytest.fail("builder should not run on cache hit"),
    )

    assert first == second == {"value": 1}
    assert calls == 1


def test_run_analysis_jobs_with_json_cache_skips_completed_jobs(tmp_path):
    signature = {"inputs_sha256": "abc", "bootstrap_iter": 10}
    cached_path = tmp_path / "cached.json"
    fresh_path = tmp_path / "fresh.json"
    analyze_results_module._write_cached_json_artifact(
        cached_path,
        "already_done",
        run_signature=signature,
    )
    jobs = [
        ("cached", "_safe_analysis_slug", ("Should Not Run",), {}),
        ("fresh", "_safe_analysis_slug", ("Fresh Model",), {}),
    ]

    results = analyze_results_module._run_analysis_jobs_with_json_cache(
        jobs,
        artifact_paths={"cached": cached_path, "fresh": fresh_path},
        run_signature=signature,
        resume=True,
        workers=1,
    )

    assert results == {
        "cached": "already_done",
        "fresh": "fresh_model",
    }
    assert analyze_results_module._load_cached_json_artifact(
        fresh_path,
        run_signature=signature,
    ) == "fresh_model"


def test_primary_h1_parallel_workers_match_serial():
    df = pd.DataFrame(_missingness_fixture_rows())

    serial = build_primary_h1_results(
        df,
        bootstrap_iter=0,
        bootstrap_seed=20260510,
        workers=1,
    )
    parallel = build_primary_h1_results(
        df,
        bootstrap_iter=0,
        bootstrap_seed=20260510,
        workers=2,
    )

    assert parallel == serial


def test_sen_9d_parallel_workers_match_serial_records():
    df = pd.DataFrame(_missingness_fixture_rows())

    serial = build_sen_9d_tipping_point(
        df,
        bootstrap_iter=0,
        bootstrap_seed=20260510,
        tipping_workers=1,
    )
    parallel = build_sen_9d_tipping_point(
        df,
        bootstrap_iter=0,
        bootstrap_seed=20260510,
        tipping_workers=2,
    )

    assert parallel["tipping_worker_count"] == 2
    assert parallel["records"] == serial["records"]


def test_sen_9d_uses_endpoint_persona_scope_for_unresolved_denominator(monkeypatch):
    rows = [
        _sensitivity_row(
            prompt_id="p_neutral",
            subject_id="model_a",
            persona="PERSONA_NEUTRAL",
            trial_id=0,
            mode="TOOL",
            event=None,
        ),
        _sensitivity_row(
            prompt_id="p_operator",
            subject_id="model_a",
            persona="PERSONA_OPERATOR",
            trial_id=0,
            mode="DISABLED",
            event=None,
        ),
    ]

    def fake_build_primary_h1_results(
        frame: pd.DataFrame,
        *,
        bootstrap_iter: int = 10000,
        bootstrap_seed: int = 42,
    ) -> dict:
        return {
            "tests": {
                endpoint: {
                    "or": 2.0,
                    "p_value": 0.01,
                    "p_holm": 0.01,
                    "holm_reject": True,
                }
                for endpoint in (
                    "h1a_erosion",
                    "h1a_full_asr",
                    "h1b_erosion",
                    "h1b_full_asr",
                )
            }
        }

    monkeypatch.setattr(
        analyze_results_module,
        "build_primary_h1_results",
        fake_build_primary_h1_results,
    )

    result = build_sen_9d_tipping_point(
        pd.DataFrame(rows),
        bootstrap_iter=0,
        tipping_workers=1,
    )

    unresolved_by_endpoint = {
        record["endpoint"]: record["unresolved_rows"]
        for record in result["records"]
    }
    assert unresolved_by_endpoint["h1a_erosion"] == 2
    assert unresolved_by_endpoint["h1a_full_asr"] == 2
    assert unresolved_by_endpoint["h1b_erosion"] == 1
    assert unresolved_by_endpoint["h1b_full_asr"] == 1


def test_sen_9d_tipping_search_skips_candidate_bootstrap(monkeypatch):
    df = pd.DataFrame(
        [
            _sensitivity_row(
                prompt_id="prompt_unresolved",
                subject_id="model_a",
                persona="PERSONA_NEUTRAL",
                trial_id=0,
                mode="TOOL",
                event=None,
            )
        ]
    )
    bootstrap_calls: list[int] = []

    def fake_build_primary_h1_results(
        frame: pd.DataFrame,
        *,
        bootstrap_iter: int = 10000,
        bootstrap_seed: int = 42,
    ) -> dict:
        bootstrap_calls.append(bootstrap_iter)
        if len(bootstrap_calls) == 1:
            test_result = {"or": 2.0, "p_value": 0.01, "p_holm": 0.01}
        else:
            test_result = {"or": 0.5, "p_value": 0.20, "p_holm": 0.20}
        return {"tests": {"h1a_full_asr": test_result}}

    monkeypatch.setattr(
        analyze_results_module,
        "build_primary_h1_results",
        fake_build_primary_h1_results,
    )

    result = build_sen_9d_tipping_point(
        df,
        bootstrap_iter=123,
        tipping_workers=1,
    )

    assert bootstrap_calls == [123, 0]
    assert result["records"][0]["feasible"] is True


def test_sen_9_summary_reports_all_rules_or_and_adjusted_p_values():
    summary = build_sen_9_summary(
        pd.DataFrame(_missingness_fixture_rows()),
        bootstrap_iter=10,
        bootstrap_seed=20260510,
    )

    assert summary["resolution_path"] == "sen_9_summary"
    assert set(summary["rules"]) == {
        "sen_9a_relaxed_inclusion",
        "sen_9b_unresolved_as_refuse",
        "sen_9c_unresolved_as_comply",
        "sen_9d_tipping_point",
        "sen_9e_differential_worst_case",
    }
    rows = summary["table"]
    assert len(rows) == 20
    assert {
        (row["rule"], row["endpoint"])
        for row in rows
    } == {
        (rule, endpoint)
        for rule in summary["rules"]
        for endpoint in ("h1a_erosion", "h1a_full_asr", "h1b_erosion", "h1b_full_asr")
    }
    for row in rows:
        assert "or" in row
        assert "p_holm" in row
        assert "holm_reject" in row
        if row["rule"] != "sen_9d_tipping_point" or row["tipping_point_feasible"]:
            assert row["or"] is not None
            assert row["p_holm"] is not None
        if row["rule"] == "sen_9e_differential_worst_case":
            assert row["resolution_path"] == "sen_9e_differential_worst_case_against_h1"


def test_sen_h_runs_hardened_tool_vs_baseline_disabled_clogit(monkeypatch):
    rows = _sen_h_fixture_rows()
    monkeypatch.setattr(
        analyze_results_module,
        "_calibration_prompt_ids_for_analysis",
        lambda: {row["prompt_id"] for row in rows},
    )

    result = build_sen_h_hardened_intervention(pd.DataFrame(rows), bootstrap_iter=10)

    assert result["resolution_path"] == "sen_h_hardened_tool_vs_baseline_disabled"
    h1a = result["tests"]["h1a_erosion"]
    assert h1a["resolution_path"] == "sen_h_hardened_tool_vs_baseline_disabled"
    assert h1a["n_rows"] == 216
    assert h1a["retained_cluster_count"] == 36
    assert h1a["or"] is not None
    assert h1a["p_holm"] is not None
    assert result["primary_filter"]["intervention_contrast"] == (
        "hardened_tool_vs_baseline_disabled"
    )


def test_intervention_h2_filters_hardened_keys_without_dataframe_apply(monkeypatch):
    rows = []
    for trial_id in range(3):
        rows.append(
            _sensitivity_row(
                prompt_id="prompt_h2",
                subject_id="model_h2",
                persona="PERSONA_NEUTRAL",
                trial_id=trial_id,
                mode="TOOL",
                event=0,
            )
        )
        rows.append(
            _sensitivity_row(
                prompt_id="prompt_h2",
                subject_id="model_h2",
                persona="PERSONA_NEUTRAL",
                trial_id=trial_id,
                mode="TOOL",
                event=1,
                intervention_condition="hardened",
            )
        )
    monkeypatch.setattr(
        analyze_results_module,
        "_calibration_prompt_ids_for_analysis",
        lambda: {"prompt_h2"},
    )

    def fail_apply(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise AssertionError("build_intervention_h2 must not use DataFrame.apply")

    monkeypatch.setattr(pd.DataFrame, "apply", fail_apply)

    result = build_intervention_h2(pd.DataFrame(rows))

    assert result["status"] == "ok"
    assert result["endpoints"]["full_asr"]["n_pairs"] == 3


def test_calibration_prompt_loader_propagates_runtime_failures(monkeypatch):
    from scripts import run_synthetic_prompts

    def fail_load():
        raise RuntimeError("manifest corrupt")

    monkeypatch.setattr(run_synthetic_prompts, "load_calibration_prompt_ids", fail_load)

    with pytest.raises(RuntimeError, match="manifest corrupt"):
        analyze_results_module._calibration_prompt_ids_for_analysis()


def test_paired_delta_by_subject_forwards_bootstrap_seed(monkeypatch):
    captured_seeds: list[int] = []
    rows = pd.DataFrame(
        [
            {
                "subject_id": "model_a",
                "prompt_id": "prompt_1",
                "pair_id": "prompt_1__trial0",
                "mode": "DISABLED",
                "Z_full_asr": 0,
            },
            {
                "subject_id": "model_a",
                "prompt_id": "prompt_1",
                "pair_id": "prompt_1__trial0",
                "mode": "TOOL",
                "Z_full_asr": 1,
            },
        ]
    )

    def fake_bootstrap(data, cluster_col, statistic_fn, *, B, seed):  # noqa: ANN001, ARG001
        captured_seeds.append(seed)
        return [0.25, 0.75]

    monkeypatch.setattr(analyze_results_module, "cluster_bootstrap", fake_bootstrap)
    monkeypatch.setattr(
        analyze_results_module,
        "get_bootstrap_ci",
        lambda values: (min(values), max(values)),
    )

    result = _paired_delta_by_subject(
        rows,
        bootstrap_iter=9,
        bootstrap_seed=20260511,
    )

    assert captured_seeds == [20260511]
    assert result[0]["lower"] == 0.25
    assert result[0]["upper"] == 0.75


def test_sen_q_reports_label_agreement_and_tool_call_rate_drift(monkeypatch):
    rows = _sen_q_fixture_rows()
    monkeypatch.setattr(
        analyze_results_module,
        "_calibration_prompt_ids_for_analysis",
        lambda: {row["prompt_id"] for row in rows},
    )

    result = build_sen_q_quantization_audit(pd.DataFrame(rows))

    assert result["status"] == "ok"
    llama = result["subjects"]["llama-3.3-70b-instruct"]
    assert llama["n_pairs"] == 3
    assert llama["ordinal_label_agreement_rate"] == pytest.approx(2 / 3)
    assert llama["tool_call_primary_rate"] == pytest.approx(1 / 3)
    assert llama["tool_call_quantization_audit_rate"] == pytest.approx(2 / 3)
    assert llama["delta_tool_call_rate_quantization_minus_primary"] == pytest.approx(1 / 3)
    assert result["endpoints"]["full_asr"]["llama-3.3-70b-instruct"][
        "delta_quantization_minus_primary"
    ] == pytest.approx(1 / 3)


def test_sen_q_quantization_audit_groups_subjects_once():
    source = inspect.getsource(analyze_results_module.build_sen_q_quantization_audit)

    assert source.count('rows.groupby("subject_id")') == 1


@pytest.mark.slow
def test_sen_secondary_summary_reports_primary_h_q_and_d(monkeypatch):
    rows = _sen_h_fixture_rows() + _sen_q_fixture_rows()
    monkeypatch.setattr(
        analyze_results_module,
        "_calibration_prompt_ids_for_analysis",
        lambda: {row["prompt_id"] for row in rows},
    )

    result = build_sen_secondary_summary(
        pd.DataFrame(rows),
        bootstrap_iter=10,
        bootstrap_seed=20260510,
    )

    assert result["resolution_path"] == "sen_secondary_summary"
    assert set(result) >= {"primary_h1", "sen_h", "sen_q", "sen_d"}
    assert result["primary_h1"]["resolution_path"] == "primary_6_of_6_complete_case"
    assert result["sen_h"]["tests"]["h1a_erosion"]["p_holm"] is not None
    assert result["sen_q"]["status"] == "ok"
    assert result["sen_d"]["resolution_path"] == "sen_d_dispatch_block"


def test_sen_secondary_summary_accepts_precomputed_primary(monkeypatch):
    primary_h1 = {"resolution_path": "precomputed_primary", "tests": {}}

    def fail_primary(*args, **kwargs):  # noqa: ANN001, ARG001
        raise AssertionError("primary H1 should not be refit")

    monkeypatch.setattr(analyze_results_module, "build_primary_h1_results", fail_primary)
    monkeypatch.setattr(
        analyze_results_module,
        "build_sen_h_hardened_intervention",
        lambda *args, **kwargs: {"status": "sen_h_stub"},
    )
    monkeypatch.setattr(
        analyze_results_module,
        "build_sen_q_quantization_audit",
        lambda *args, **kwargs: {"status": "sen_q_stub"},
    )
    monkeypatch.setattr(
        analyze_results_module,
        "build_sen_d_dispatch_block",
        lambda *args, **kwargs: {"status": "sen_d_stub"},
    )

    result = build_sen_secondary_summary(
        pd.DataFrame(_missingness_fixture_rows()),
        primary_h1=primary_h1,
        bootstrap_iter=0,
    )

    assert result["primary_h1"] is primary_h1


def test_field_status_treats_nan_as_missing_or_documented_null():
    assert _field_status(float("nan")) == "missing"
    assert _field_status(float("nan"), allow_documented_null=True) == "documented_null"
    assert _field_status(float("inf")) == "mismatch"


def test_prereg_contract_ledger_reads_domain_homogeneity_endpoint_shape():
    primary_h1 = {
        "primary_filter": {
            "partition_id": "main",
            "trial_ids": [0, 1, 2],
            "intervention_condition": "baseline",
            "serving_precision_role": "primary",
            "modes": ["DISABLED", "TOOL"],
        },
        "resolution_path": "primary_6_of_6_complete_case",
        "tests": {},
    }
    domain_homogeneity = {
        "h1b_full_asr": {
            "status": "ok",
            "headline_unit": "pooled",
            "pooled_result_role": "headline",
            "domain_results_role": "descriptive",
            "domains": [
                {
                    "domain": "GridSim",
                    "conditional_logit_or": 2.0,
                    "reporting_role": "descriptive",
                }
            ],
        }
    }

    ledger = build_prereg_section6_contract_ledger(
        primary_h1=primary_h1,
        domain_homogeneity=domain_homogeneity,
        sen_9_summary={"table": []},
        sen_h={"tests": {}},
        sen_q={"subjects": {}, "endpoints": {}},
        sen_d={"aggregate_fits": {}},
    )

    domain_entries = [
        entry for entry in ledger["entries"] if entry["artifact"] == "domain_homogeneity"
    ]
    assert {
        "endpoints.h1b_full_asr.headline_unit",
        "endpoints.h1b_full_asr.pooled_result_role",
        "endpoints.h1b_full_asr.domain_results_role",
        "endpoints.h1b_full_asr.domains[0].conditional_logit_or",
    } <= {entry["field"] for entry in domain_entries}
    assert all(entry["status"] == "match" for entry in domain_entries)


@pytest.mark.slow
def test_sen_d_emits_block_descriptives_and_adjusted_fits():
    rows: list[dict] = []
    for cluster_idx in range(40):
        route = "native" if cluster_idx < 20 else "openrouter"
        if cluster_idx % 4 == 0:
            tool_events = (1, 0, 0)
            disabled_events = (1, 1, 0)
        else:
            tool_events = (1, 1, 0)
            disabled_events = (1, 0, 0)
        cluster = _cluster_rows(
            cluster_idx=cluster_idx,
            tool_events=tool_events,
            disabled_events=disabled_events,
            route=route,
        )
        for row in cluster:
            row["dispatch_block_id"] = "model_a:early_batch:t0" if cluster_idx < 20 else "model_a:late_batch:t0"
        rows.extend(cluster)

    result = build_sen_d_dispatch_block(pd.DataFrame(rows), bootstrap_iter=10)

    assert result["resolution_path"] == "sen_d_dispatch_block"
    assert set(result["aggregate_fits"]) == {"fixed_effect", "stratified"}
    assert result["aggregate_fits"]["fixed_effect"]["tests"]["h1a_erosion"]["p_holm"] is not None
    assert result["aggregate_fits"]["stratified"]["tests"]["h1a_erosion"]["p_holm"] is not None
    assert result["block_descriptives"]["n_blocks"] == 2
    first_block = result["block_descriptives"]["blocks"][0]
    assert first_block["tool_count"] == 60
    assert first_block["disabled_count"] == 60
    assert first_block["timestamp_min"] is not None
    assert first_block["route_mix"]
    shift = result["or_shift_flags"]["h1a_erosion"]["fixed_effect"]
    assert "shift_gt_5_percent" in shift


@pytest.mark.slow
@pytest.mark.integration
def test_analyzer_writes_sensitivity_json_artifacts(tmp_path):
    input_path = tmp_path / "labeled.jsonl"
    rows = _missingness_fixture_rows()
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output_dir = tmp_path / "stats"
    figure_dir = tmp_path / "figures"

    result = subprocess.run(
        [
            "python3",
            "scripts/analyze_results.py",
            "--inputs",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--figure-dir",
            str(figure_dir),
            "--bootstrap-iter",
            "10",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    sensitivity_dir = output_dir / "sensitivities"
    for filename in (
        "sen_1_ordinal_median_robustness.json",
        "sen_2_judge_alone.json",
        "sen_8_dawid_skene.json",
        "sen_9a_relaxed_inclusion.json",
        "sen_9b_unresolved_as_refuse.json",
        "sen_9c_unresolved_as_comply.json",
        "sen_9d_tipping_point.json",
        "sen_9e_differential_worst_case.json",
        "sen_9_summary.json",
        "sen_d_dispatch_block.json",
        "sen_m_measurement_pathway.json",
        "predicate_override_audit.json",
        "predicate_override_removal.json",
        "validator_tier_exclusion.json",
        "repair_round_exclusion.json",
        "tier0_repair0_intersection.json",
    ):
        assert (sensitivity_dir / filename).exists()
    assert (output_dir / "sen_secondary_summary.json").exists()
