#!/usr/bin/env python3
"""Generate, validate, cost-gate, and optionally submit DISABLED+FORMAT sets.

Generation and dry-run validation never call an external API. Paid submission
is a separate action with explicit protocol, request-count, and dollar gates.
Local/self-hosted rows are generated here but dispatched by
``scripts/run_local_subjects.py`` so its checkpoint and served-model guards
remain authoritative.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO_ROOT / "experiments/disabled_format_control/protocol.json"
DEFAULT_LOCKED_INPUTS_MANIFEST = (
    REPO_ROOT / "experiments/disabled_format_control/locked_inputs_manifest.json"
)
DEFAULT_SOURCE_DIR = REPO_ROOT / "outputs/main_run/20260525T164059Z/batches"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/postreg_disabled_format_control/batches"
NATIVE_PRICING_PATH = REPO_ROOT / "config/api_pricings_native_providers.json"
OPENROUTER_PRICING_PATH = REPO_ROOT / "config/openrouter_pricing_current.json"
FORMAT_BASELINE = "BASELINE"
FORMAT_JSON = "FORMAT_JSON"
SOURCE_MODES = ("DISABLED", "TOOL")
CONTROL_MODES = ("DISABLED",)
FORMATS = (FORMAT_BASELINE, FORMAT_JSON)
ARMS = tuple(f"{mode}_{fmt}" for mode in CONTROL_MODES for fmt in FORMATS)
THREE_ARM_CONTROL_ARMS = (
    "DISABLED_BASELINE",
    "DISABLED_FORMAT_JSON",
    "TOOL_CONTEMPORANEOUS",
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
OPENROUTER_DECODING_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield row


def _arm_definitions(protocol: dict[str, Any]) -> tuple[tuple[str, dict[str, str]], ...]:
    design = protocol.get("design")
    if not isinstance(design, dict):
        raise ValueError("protocol design must be an object")
    configured = design.get("arm_definitions")
    if configured is None:
        return (
            (
                "DISABLED_BASELINE",
                {
                    "source_mode": "DISABLED",
                    "format_condition": FORMAT_BASELINE,
                    "tool_choice": "none",
                },
            ),
            (
                "DISABLED_FORMAT_JSON",
                {
                    "source_mode": "DISABLED",
                    "format_condition": FORMAT_JSON,
                    "tool_choice": "none",
                },
            ),
        )
    if not isinstance(configured, dict) or not configured:
        raise ValueError("protocol design.arm_definitions must be a non-empty object")
    ordered_arms = design.get("arms")
    if not isinstance(ordered_arms, list) or not ordered_arms:
        raise ValueError("protocol design.arms must order the arm definitions")
    if len(ordered_arms) != len(set(map(str, ordered_arms))):
        raise ValueError("protocol design.arms contains duplicates")
    if set(map(str, ordered_arms)) != set(map(str, configured)):
        raise ValueError("protocol design.arms and arm_definitions differ")
    output: list[tuple[str, dict[str, str]]] = []
    for raw_name in ordered_arms:
        name = str(raw_name)
        raw_definition = configured.get(name)
        if not isinstance(raw_definition, dict):
            raise ValueError(f"invalid arm definition for {name}")
        definition = {
            "source_mode": str(raw_definition.get("source_mode") or ""),
            "format_condition": str(raw_definition.get("format_condition") or ""),
            "tool_choice": str(raw_definition.get("tool_choice") or ""),
        }
        if definition["source_mode"] not in SOURCE_MODES:
            raise ValueError(f"invalid source_mode for arm {name}")
        if definition["format_condition"] not in FORMATS:
            raise ValueError(f"invalid format_condition for arm {name}")
        expected_choice = "none" if definition["source_mode"] == "DISABLED" else "auto"
        if definition["tool_choice"] != expected_choice:
            raise ValueError(f"invalid tool_choice for arm {name}")
        output.append((name, definition))
    return tuple(output)


def _protocol_arms(protocol: dict[str, Any]) -> tuple[str, ...]:
    return tuple(name for name, _definition in _arm_definitions(protocol))


def _validated_openrouter_decoding(value: Any, subject_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"route descriptor for {subject_id} needs a decoding object")
    allowed = {"temperature", "top_p", "max_tokens"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"route descriptor for {subject_id} has unsupported decoding keys: {unknown}"
        )
    max_tokens = value.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(
            f"route descriptor for {subject_id} needs a positive integer max_tokens"
        )
    has_temperature = "temperature" in value
    has_top_p = "top_p" in value
    if has_temperature != has_top_p:
        raise ValueError(
            f"route descriptor for {subject_id} must declare temperature and top_p together"
        )
    if has_temperature:
        temperature = value["temperature"]
        top_p = value["top_p"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError(f"route descriptor for {subject_id} has invalid temperature")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise ValueError(f"route descriptor for {subject_id} has invalid top_p")
        if not 0 <= float(temperature) <= 2:
            raise ValueError(f"route descriptor for {subject_id} has invalid temperature")
        if not 0 < float(top_p) <= 1:
            raise ValueError(f"route descriptor for {subject_id} has invalid top_p")
    return copy.deepcopy(value)


def _apply_exact_openrouter_decoding(
    body: dict[str, Any], decoding: dict[str, Any]
) -> None:
    for key in OPENROUTER_DECODING_FIELDS:
        body.pop(key, None)
    body.update(copy.deepcopy(decoding))


def _route_descriptor(protocol: dict[str, Any], subject_id: str) -> dict[str, Any]:
    raw = protocol["route_policy"]["routes"][subject_id]
    if isinstance(raw, str):
        route, separator, detail = raw.partition(":")
        if route not in {"openrouter", "native", "local"} or not separator or not detail:
            raise ValueError(f"invalid frozen route spec for {subject_id}: {raw}")
        return {
            "route": route,
            "legacy_detail": detail,
            "provider_only": [detail] if route == "openrouter" else [],
            "quantizations": [],
        }
    if not isinstance(raw, dict):
        raise ValueError(f"invalid route descriptor for {subject_id}")
    route = str(raw.get("route") or "")
    if route not in {"openrouter", "native"}:
        raise ValueError(
            f"object route descriptor for {subject_id} must use openrouter or native, "
            f"got {route!r}"
        )
    model_slug = str(raw.get("model_slug") or "")
    provider_only = raw.get("provider_only")
    quantizations = raw.get("quantizations", [])
    precision = str(raw.get("serving_precision") or "")
    decoding = _validated_openrouter_decoding(raw.get("decoding"), subject_id)
    if not model_slug:
        raise ValueError(f"route descriptor for {subject_id} has no model_slug")
    if not isinstance(provider_only, list) or len(provider_only) != 1:
        raise ValueError(f"route descriptor for {subject_id} needs one provider_only pin")
    if not isinstance(provider_only[0], str) or not provider_only[0].strip():
        raise ValueError(f"route descriptor for {subject_id} has an invalid provider pin")
    if not isinstance(quantizations, list) or any(
        not isinstance(value, str) or not value for value in quantizations
    ):
        raise ValueError(f"route descriptor for {subject_id} has invalid quantizations")
    if not precision:
        raise ValueError(f"route descriptor for {subject_id} has no serving_precision")
    if route == "native":
        native_fields = {
            "endpoint": raw.get("endpoint"),
            "request_path": raw.get("request_path"),
            "batch_url_path": raw.get("batch_url_path"),
            "batch_input_files_path": raw.get("batch_input_files_path"),
            "request_shape": raw.get("request_shape"),
        }
        invalid = sorted(
            name
            for name, value in native_fields.items()
            if not isinstance(value, str) or not value.strip()
        )
        if invalid:
            raise ValueError(
                f"native route descriptor for {subject_id} has missing fields: {invalid}"
            )
        endpoint = str(native_fields["endpoint"])
        if not endpoint.startswith("https://"):
            raise ValueError(
                f"native route descriptor for {subject_id} must use an https endpoint"
            )
        if any(
            not str(native_fields[name]).startswith("/")
            for name in ("request_path", "batch_url_path", "batch_input_files_path")
        ):
            raise ValueError(
                f"native route descriptor for {subject_id} paths must start with '/'"
            )
        if native_fields["request_shape"] != "openai_compatible_batch":
            raise ValueError(
                f"native route descriptor for {subject_id} has unsupported request_shape"
            )
    descriptor = copy.deepcopy(raw)
    descriptor["decoding"] = decoding
    return descriptor


def _validate_protocol_structure(protocol: dict[str, Any]) -> None:
    arms = _protocol_arms(protocol)
    population = protocol["population"]
    subjects = population.get("subjects")
    personas = population.get("personas")
    trial_ids = population.get("trial_ids")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("protocol population.subjects must be non-empty")
    if not isinstance(personas, list) or not personas:
        raise ValueError("protocol population.personas must be non-empty")
    if not isinstance(trial_ids, list) or not trial_ids:
        raise ValueError("protocol population.trial_ids must be non-empty")
    expected_sets = (
        len(subjects)
        * int(population["prompt_variants"])
        * len(personas)
        * len(trial_ids)
    )
    if int(population["expected_cells_per_arm"]) != expected_sets:
        raise ValueError("protocol expected_cells_per_arm does not match its population")
    if int(population["expected_rows_total"]) != expected_sets * len(arms):
        raise ValueError("protocol expected_rows_total does not match population x arms")
    if "expected_analysis_sets" in population and int(
        population["expected_analysis_sets"]
    ) != expected_sets:
        raise ValueError("protocol expected_analysis_sets does not match its population")
    if "expected_canary_rows_total" in population and int(
        population["expected_canary_rows_total"]
    ) != len(subjects) * len(arms):
        raise ValueError("protocol expected_canary_rows_total does not match subjects x arms")
    routes = protocol["route_policy"].get("routes")
    if not isinstance(routes, dict) or set(routes) != set(subjects):
        raise ValueError("protocol route table must contain exactly the frozen subjects")
    for subject_id in subjects:
        _route_descriptor(protocol, str(subject_id))
        raw_route = routes[subject_id]
        if protocol.get("status") == "frozen_before_first_dispatch" and isinstance(
            raw_route, dict
        ):
            evidence_fields = (
                "preflight_evidence_id",
                "preflight_evidence_sha256",
                "disabled_schema_retention",
                "tool_native_call_support",
                "provider_pin_verified",
            )
            if any(raw_route.get(field) is None for field in evidence_fields):
                raise ValueError(
                    f"frozen route descriptor for {subject_id} has incomplete evidence"
                )
    execution_order = protocol.get("execution_order")
    if execution_order is not None and list(map(str, execution_order)) != list(
        map(str, subjects)
    ):
        raise ValueError("protocol execution_order must exactly match population.subjects")
    if any(isinstance(route, dict) for route in routes.values()):
        namespace = protocol["design"].get("partition_namespace")
        if not isinstance(namespace, str) or not namespace.startswith(
            "postreg_disabled_format_control_v3"
        ):
            raise ValueError(
                "three-arm object routes require an isolated v3 partition namespace"
            )
        artifact_paths = protocol.get("artifact_paths")
        required_paths = {
            "full_batches_dir",
            "canary_batches_dir",
            "locked_inputs_manifest",
            "responses_root",
            "canary_responses_root",
            "analysis_root",
        }
        if protocol.get("status") == "frozen_before_first_dispatch":
            required_paths.add("execution_authorization")
        if not isinstance(artifact_paths, dict) or not required_paths.issubset(artifact_paths):
            raise ValueError("three-arm protocol artifact_paths are incomplete")
        for name in required_paths:
            value = Path(str(artifact_paths[name]))
            if value.is_absolute() or ".." in value.parts:
                raise ValueError(f"protocol artifact path {name} must be repository-relative")
        full_batches = Path(str(artifact_paths["full_batches_dir"]))
        canary_batches = Path(str(artifact_paths["canary_batches_dir"]))
        if full_batches.parent.name != namespace:
            raise ValueError("v3 full artifact root must match its partition namespace")
        if canary_batches.parent.parent.name != namespace:
            raise ValueError("v3 canary artifact root must match its partition namespace")
        if full_batches.parent / "responses" != Path(str(artifact_paths["responses_root"])):
            raise ValueError("v3 full response root must be adjacent to its batches directory")
        if canary_batches.parent / "responses" != Path(
            str(artifact_paths["canary_responses_root"])
        ):
            raise ValueError("v3 canary response root must be adjacent to its batches directory")


def load_protocol(
    path: Path, *, allow_draft: bool = False
) -> tuple[dict[str, Any], str]:
    protocol = _read_json(path)
    required = {
        "protocol_id",
        "status",
        "design",
        "population",
        "source_batch_sha256",
        "route_policy",
        "randomization",
        "grading_contract_sha256",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"protocol missing fields: {missing}")
    status = str(protocol["status"])
    if status != "frozen_before_first_dispatch":
        if status == "draft_pending_route_preflight" and not allow_draft:
            raise ValueError("draft protocol cannot be used for generation or submission")
        if status != "draft_pending_route_preflight":
            raise ValueError(f"unsupported protocol status: {status}")
    exact = protocol["design"].get("format_instruction_exact")
    if not isinstance(exact, str) or not exact.strip():
        raise ValueError("protocol format_instruction_exact must be non-empty")
    grading_contract = protocol["grading_contract_sha256"]
    if not isinstance(grading_contract, dict) or not grading_contract:
        raise ValueError("protocol grading_contract_sha256 must be a non-empty object")
    for relative_path, expected_sha256 in grading_contract.items():
        _verify_file_hash(REPO_ROOT / str(relative_path), str(expected_sha256))
    implementation_contract = protocol.get("implementation_contract_sha256")
    if implementation_contract is not None:
        if not isinstance(implementation_contract, dict) or not implementation_contract:
            raise ValueError(
                "protocol implementation_contract_sha256 must be a non-empty object"
            )
        for relative_path, expected_sha256 in implementation_contract.items():
            _verify_file_hash(REPO_ROOT / str(relative_path), str(expected_sha256))
    _validate_protocol_structure(protocol)
    return protocol, _sha256_file(path)


def validate_protocol_draft(path: Path) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(path, allow_draft=True)
    if protocol["status"] != "draft_pending_route_preflight":
        raise ValueError("draft validation requires draft_pending_route_preflight status")
    routes = protocol["route_policy"]["routes"]
    evidence_fields = (
        "preflight_evidence_id",
        "preflight_evidence_sha256",
        "disabled_schema_retention",
        "tool_native_call_support",
        "provider_pin_verified",
    )
    complete = sum(
        1
        for route in routes.values()
        if isinstance(route, dict) and all(route.get(field) is not None for field in evidence_fields)
    )
    arms = _protocol_arms(protocol)
    population = protocol["population"]
    gates = protocol.get("promotion_gates")
    gates_closed = isinstance(gates, dict) and all(value is True for value in gates.values())
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "status": protocol["status"],
        "subjects": len(population["subjects"]),
        "arms": len(arms),
        "arm_names": list(arms),
        "expected_rows_total": int(population["expected_rows_total"]),
        "expected_canary_rows_total": int(population["expected_canary_rows_total"]),
        "routes_with_complete_evidence": complete,
        "promotable": complete == len(routes) and gates_closed,
        "submission_permitted": False,
    }


def _selected_subjects(protocol: dict[str, Any], requested: list[str] | None) -> list[str]:
    frozen = list(protocol["population"]["subjects"])
    if not requested:
        return frozen
    unknown = sorted(set(requested) - set(frozen))
    if unknown:
        raise ValueError(f"subjects not in frozen protocol: {unknown}")
    return [subject for subject in frozen if subject in set(requested)]


def _verify_file_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"source hash mismatch for {path}: expected {expected}, got {actual}")


def _load_gpt_native_rows(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for relative, expected in protocol.get("supplemental_request_source_sha256", {}).items():
        path = REPO_ROOT / relative
        _verify_file_hash(path, expected)
        for row in _iter_jsonl(path):
            custom_id = str(row.get("custom_id") or "")
            if not custom_id or custom_id in rows:
                raise ValueError(f"duplicate/missing supplemental custom_id in {path}: {custom_id!r}")
            rows[custom_id] = row
    if len(rows) != 5184:
        raise ValueError(f"expected 5184 frozen GPT native request rows, found {len(rows)}")
    return rows


def _source_unit_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["subject_id"]),
        str(row["prompt_id"]),
        str(row["persona"]),
        int(row["trial_id"]),
    )


def _load_subject_pairs(
    protocol: dict[str, Any],
    source_dir: Path,
    subject_id: str,
    gpt_native: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, str, int], dict[str, dict[str, Any]]]:
    filename = f"input_{subject_id}.jsonl"
    expected = protocol["source_batch_sha256"][filename]
    path = source_dir / filename
    _verify_file_hash(path, expected)
    population = protocol["population"]
    pairs: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for source_row in _iter_jsonl(path):
        if source_row.get("corpus_partition_id") != population["source_partition_id"]:
            continue
        if source_row.get("intervention_condition") != population["source_intervention_condition"]:
            continue
        if source_row.get("serving_precision_role") != population["serving_precision_role"]:
            continue
        mode = source_row.get("mode")
        if mode not in SOURCE_MODES:
            continue
        row = copy.deepcopy(source_row)
        if subject_id == "gpt-5.5" and gpt_native:
            native = gpt_native.get(str(row["custom_id"]))
            if native is None:
                raise ValueError(f"GPT native request missing for {row['custom_id']}")
            row["body"] = copy.deepcopy(native["body"])
            row["method"] = native.get("method", "POST")
            row["url"] = native.get("url", "/v1/chat/completions")
            row["route"] = "native"
            row["pinned_provider"] = "openai"
            row["batch_submit_url"] = "/v1/batches"
            row["request_shape"] = "openai_compatible_batch"
            row["native_body"] = copy.deepcopy(native["body"])
        key = _source_unit_key(row)
        if mode in pairs[key]:
            raise ValueError(f"duplicate source row for {key} {mode}")
        pairs[key][mode] = row

    expected_per_subject = int(population["expected_cells_per_arm"]) // len(population["subjects"])
    if len(pairs) != expected_per_subject:
        raise ValueError(
            f"{subject_id}: expected {expected_per_subject} paired cells, found {len(pairs)}"
        )
    for key, by_mode in pairs.items():
        if set(by_mode) != set(SOURCE_MODES):
            raise ValueError(f"{subject_id}: incomplete source mode pair for {key}")
        disabled, tool = by_mode["DISABLED"], by_mode["TOOL"]
        if disabled["body"].get("messages") != tool["body"].get("messages"):
            raise ValueError(f"{subject_id}: messages differ across source modes for {key}")
        if disabled["body"].get("tools") != tool["body"].get("tools"):
            raise ValueError(f"{subject_id}: visible tools differ across source modes for {key}")
        d_body = {k: v for k, v in disabled["body"].items() if k != "tool_choice"}
        t_body = {k: v for k, v in tool["body"].items() if k != "tool_choice"}
        if d_body != t_body:
            raise ValueError(f"{subject_id}: source bodies differ beyond tool_choice for {key}")
        if disabled["body"].get("tool_choice") != "none":
            raise ValueError(f"{subject_id}: DISABLED tool_choice is not none for {key}")
        if tool["body"].get("tool_choice") != "auto":
            raise ValueError(f"{subject_id}: TOOL tool_choice is not auto for {key}")
        if len(disabled["body"].get("tools") or []) != 6:
            raise ValueError(f"{subject_id}: expected six visible tools for {key}")
    return pairs


def _route_spec(protocol: dict[str, Any], subject_id: str) -> tuple[str, str]:
    descriptor = _route_descriptor(protocol, subject_id)
    route = str(descriptor["route"])
    if "legacy_detail" in descriptor:
        return route, str(descriptor["legacy_detail"])
    return route, str(descriptor["provider_only"][0])


def _apply_route_metadata(row: dict[str, Any], protocol: dict[str, Any]) -> None:
    subject_id = str(row["subject_id"])
    descriptor = _route_descriptor(protocol, subject_id)
    route, detail = _route_spec(protocol, subject_id)
    row["route"] = route
    row["control_route_lock"] = copy.deepcopy(
        protocol["route_policy"]["routes"][subject_id]
    )
    if route == "openrouter":
        row["method"] = "POST"
        row["url"] = "/chat/completions"
        row["batch_submit_url"] = None
        row["pinned_provider"] = detail
        row.pop("native_body", None)
        row.pop("request_shape", None)
        row.pop("local_inference", None)
        for key in tuple(row):
            if key.startswith("served_model_"):
                row.pop(key, None)
        if "model_slug" in descriptor:
            row["body"]["model"] = str(descriptor["model_slug"])
            _apply_exact_openrouter_decoding(row["body"], descriptor["decoding"])
            row["openrouter_provider_only"] = list(descriptor["provider_only"])
            row["openrouter_quantizations"] = list(descriptor.get("quantizations", []))
            precision = str(descriptor["serving_precision"])
            row["serving_precision"] = precision
            row["serving_precision_value"] = precision
            row["require_observed_provider_match"] = True
    elif route == "native":
        provider = detail.split(":", 1)[0]
        row["method"] = "POST"
        row["url"] = str(descriptor.get("request_path") or "/v1/chat/completions")
        row["pinned_provider"] = provider
        row["batch_submit_url"] = str(
            descriptor.get("batch_url_path") or "/v1/batches"
        )
        row["request_shape"] = str(
            descriptor.get("request_shape")
            or ("xai_http_batch" if subject_id == "grok-4.3" else "openai_compatible_batch")
        )
        if "model_slug" in descriptor:
            row["body"]["model"] = str(descriptor["model_slug"])
            _apply_exact_openrouter_decoding(row["body"], descriptor["decoding"])
            precision = str(descriptor["serving_precision"])
            row["serving_precision"] = precision
            row["serving_precision_value"] = precision
            row["native_endpoint"] = str(descriptor["endpoint"])
            row["native_batch_input_files_path"] = str(
                descriptor["batch_input_files_path"]
            )
        row["native_body"] = copy.deepcopy(row["body"])
    else:
        row["method"] = "LOCAL"
        row["url"] = "local://chat/completions"
        row["pinned_provider"] = "local"
        row["batch_submit_url"] = None


def _apply_schema_visibility_metadata(
    row: dict[str, Any], protocol: dict[str, Any]
) -> None:
    subject_id = str(row["subject_id"])
    route = protocol["route_policy"]["routes"][subject_id]
    if isinstance(route, dict):
        evidence_id = route.get("preflight_evidence_id")
        evidence_sha256 = route.get("preflight_evidence_sha256")
        visibility = route.get("disabled_schema_retention")
        tool_support = route.get("tool_native_call_support")
        pin_verified = route.get("provider_pin_verified")
        if evidence_id is None or evidence_sha256 is None:
            return
        if visibility not in {"schema_visible", "server_forced"}:
            raise ValueError(f"DISABLED schema visibility is not established for {subject_id}")
        if tool_support != "supported":
            raise ValueError(f"TOOL native-call support is not established for {subject_id}")
        if pin_verified is not True:
            raise ValueError(f"provider pin is not verified for {subject_id}")
        evidence_path = Path(str(evidence_id))
        if not evidence_path.is_absolute():
            evidence_path = REPO_ROOT / evidence_path
        _verify_file_hash(evidence_path, str(evidence_sha256))
        row.update(
            {
                "disabled_arm_schema_visibility": visibility,
                "route_support_status": "supported",
                "schema_visibility_evidence_id": str(evidence_id),
                "schema_visibility_evidence_sha256": str(evidence_sha256),
                "tool_native_call_support": tool_support,
                "provider_pin_verified": True,
            }
        )
        return
    overrides = protocol["route_policy"].get("schema_visibility_evidence_overrides", {})
    override = overrides.get(subject_id) if isinstance(overrides, dict) else None
    if override is not None:
        if not isinstance(override, dict):
            raise ValueError(f"invalid schema-visibility override for {subject_id}")
        evidence_id = str(override["schema_visibility_evidence_id"])
        evidence_sha256 = str(override["schema_visibility_evidence_sha256"])
        _verify_file_hash(REPO_ROOT / evidence_id, evidence_sha256)
        row.update(copy.deepcopy(override))


def _append_format_instruction(body: dict[str, Any], instruction: str) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("request body messages must be a list")
    system_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
    if len(system_messages) != 1:
        raise ValueError(f"expected exactly one system message, found {len(system_messages)}")
    content = system_messages[0].get("content")
    if not isinstance(content, str):
        raise ValueError("system message content must be text")
    system_messages[0]["content"] = f"{content}\n\n{instruction}"


def _refresh_request_hashes(row: dict[str, Any]) -> None:
    body = row["body"]
    system_content = next(
        (
            str(message.get("content", ""))
            for message in body.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "system"
        ),
        "",
    )
    serialized = _sha256_json(body)
    system = _sha256_text(system_content) if system_content else EMPTY_SHA256
    tools = body.get("tools")
    tool_schema = _sha256_json(tools) if tools is not None else EMPTY_SHA256
    row["serialized_request_hash"] = serialized
    row["system_prompt_hash"] = system
    row["tool_schema_hash"] = tool_schema
    hashes = copy.deepcopy(row.get("hashes") or {})
    hashes.update(
        {
            "serialized_request_hash": serialized,
            "system_prompt_hash": system,
            "tool_schema_hash": tool_schema,
            "format_instruction_sha256": row["format_instruction_sha256"],
            "protocol_sha256": row["protocol_sha256"],
        }
    )
    row["hashes"] = hashes
    ignored = {"model", "messages", "tools", "tool_choice"}
    row["decoding_applied"] = {
        key: copy.deepcopy(value) for key, value in body.items() if key not in ignored
    }
    if row.get("route") == "native":
        row["native_body"] = copy.deepcopy(body)


def _partition_id(
    protocol: dict[str, Any], arm_name: str, format_condition: str, canary: bool
) -> str:
    if arm_name == "TOOL_CONTEMPORANEOUS":
        suffix = "tool"
    else:
        suffix = "baseline" if format_condition == FORMAT_BASELINE else "format_json"
    namespace = str(
        protocol["design"].get("partition_namespace")
        or "postreg_disabled_format_control"
    )
    prefix = f"{namespace}_canary" if canary else namespace
    return f"{prefix}_{suffix}"


def _make_custom_id(row: dict[str, Any], partition_id: str) -> str:
    return (
        f"{partition_id}__{row['prompt_id']}__{row['subject_id']}__{row['mode']}__"
        f"{row['persona']}__trial{int(row['trial_id'])}__baseline"
    )


def _derive_row(
    source: dict[str, Any],
    *,
    format_condition: str | None = None,
    arm_name: str | None = None,
    protocol: dict[str, Any],
    protocol_sha256: str,
    source_batch_sha256: str,
    canary: bool,
) -> dict[str, Any]:
    definitions = dict(_arm_definitions(protocol))
    if arm_name is None:
        if format_condition is None:
            raise ValueError("derive_row requires format_condition or arm_name")
        arm_name = f"{source['mode']}_{format_condition}"
    definition = definitions.get(arm_name)
    if definition is None:
        raise ValueError(f"unknown protocol arm: {arm_name}")
    if str(source.get("mode") or "") != definition["source_mode"]:
        raise ValueError(
            f"arm {arm_name} requires source mode {definition['source_mode']}, "
            f"got {source.get('mode')!r}"
        )
    if format_condition is not None and format_condition != definition["format_condition"]:
        raise ValueError(f"arm {arm_name} has inconsistent format_condition")
    format_condition = definition["format_condition"]
    row = copy.deepcopy(source)
    body_before = copy.deepcopy(row["body"])
    row["source_custom_id"] = source["custom_id"]
    row["source_serialized_request_hash"] = _sha256_json(body_before)
    row["source_batch_sha256"] = source_batch_sha256
    if "randomized_order_seed" in source:
        row["source_randomized_order_seed"] = source["randomized_order_seed"]
    row["randomized_order_seed"] = int(protocol["randomization"]["seed"])
    row["protocol_id"] = protocol["protocol_id"]
    row["protocol_sha256"] = protocol_sha256
    row["postregistration_exploratory"] = True
    row["analysis_family"] = "postreg_disabled_format_control"
    row["format_condition"] = format_condition
    row["format_instruction_present"] = format_condition == FORMAT_JSON
    instruction = str(protocol["design"]["format_instruction_exact"])
    row["format_instruction_sha256"] = _sha256_text(instruction)
    row["original_system_prompt_hash"] = row.get("system_prompt_hash")
    row["intervention_condition"] = "baseline"
    row["analysis_inclusion"] = not canary
    row["operational_canary"] = canary
    partition_id = _partition_id(protocol, arm_name, format_condition, canary)
    row["corpus_partition_id"] = partition_id
    row["partition_id"] = partition_id
    row["custom_id"] = _make_custom_id(row, partition_id)
    row["control_arm"] = arm_name
    if format_condition == FORMAT_JSON:
        _append_format_instruction(row["body"], instruction)
    _apply_route_metadata(row, protocol)
    _apply_schema_visibility_metadata(row, protocol)
    _refresh_request_hashes(row)
    return row


def _canary_unit(
    pairs: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]], seed: int
) -> tuple[str, str, str, int]:
    def rank(key: tuple[str, str, str, int]) -> str:
        return _sha256_text(f"{seed}|{'|'.join(map(str, key))}")

    return min(pairs, key=rank)


def _shuffle_block(rows: list[dict[str, Any]], seed: int, block_key: str) -> list[dict[str, Any]]:
    local_seed = int(_sha256_text(f"{seed}|{block_key}")[:16], 16)
    rng = random.Random(local_seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    for index, row in enumerate(shuffled):
        row["order_index"] = index
    return shuffled


def build_subject_rows(
    protocol: dict[str, Any],
    protocol_sha256: str,
    source_dir: Path,
    subject_id: str,
    *,
    canary: bool,
    gpt_native: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = _load_subject_pairs(protocol, source_dir, subject_id, gpt_native)
    source_hash = protocol["source_batch_sha256"][f"input_{subject_id}.jsonl"]
    return build_rows_from_pairs(
        protocol,
        protocol_sha256,
        pairs,
        subject_id,
        canary=canary,
        source_batch_sha256=source_hash,
    )


def build_rows_from_pairs(
    protocol: dict[str, Any],
    protocol_sha256: str,
    pairs: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]],
    subject_id: str,
    *,
    canary: bool,
    source_batch_sha256: str,
) -> list[dict[str, Any]]:
    keys = (
        [_canary_unit(pairs, int(protocol["randomization"]["seed"]))]
        if canary
        else list(pairs)
    )
    by_block: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    definitions = _arm_definitions(protocol)
    for key in keys:
        _, prompt_id, persona, trial_id = key
        for arm_name, definition in definitions:
            source = pairs[key][definition["source_mode"]]
            row = _derive_row(
                source,
                arm_name=arm_name,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                source_batch_sha256=source_batch_sha256,
                canary=canary,
            )
            by_block[(subject_id, prompt_id, trial_id)].append(row)

    output: list[dict[str, Any]] = []
    seed = int(protocol["randomization"]["seed"])
    for block_index, block_key in enumerate(sorted(by_block)):
        rows = by_block[block_key]
        expected = len(definitions) if canary else len(definitions) * len(
            protocol["population"]["personas"]
        )
        if len(rows) != expected:
            raise ValueError(f"block {block_key} has {len(rows)} rows, expected {expected}")
        shuffle_key = "|".join(map(str, block_key))
        miniblock_key = f"{block_key[0]}:{block_key[1]}:t{block_key[2]}"
        block_id = f"{subject_id}:dfc{block_index:05d}:{block_key[1]}:t{block_key[2]}"
        for row in _shuffle_block(rows, seed, shuffle_key):
            row["dispatch_block_sequence_index"] = block_index
            row["dispatch_block_id"] = block_id
            row["dispatch_miniblock_key"] = miniblock_key
            output.append(row)
    validate_generated_rows(output, protocol, canary=canary)
    return output


def validate_generated_rows(
    rows: list[dict[str, Any]], protocol: dict[str, Any], *, canary: bool
) -> None:
    if not rows:
        raise ValueError("generated row set is empty")
    custom_ids: set[str] = set()
    cells: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    evidence_files: set[tuple[str, str]] = set()
    instruction = str(protocol["design"]["format_instruction_exact"])
    definitions = dict(_arm_definitions(protocol))
    arms = tuple(definitions)
    expected_seed = int(protocol["randomization"]["seed"])
    for row in rows:
        custom_id = str(row.get("custom_id") or "")
        if not custom_id or custom_id in custom_ids:
            raise ValueError(f"duplicate/missing generated custom_id: {custom_id!r}")
        custom_ids.add(custom_id)
        arm = str(row.get("control_arm") or "")
        if arm not in arms:
            raise ValueError(f"invalid generated control arm: {arm!r}")
        if bool(row.get("analysis_inclusion")) == canary:
            raise ValueError(f"canary analysis-inclusion invariant failed: {custom_id}")
        if row.get("protocol_id") != protocol["protocol_id"]:
            raise ValueError(f"protocol_id mismatch: {custom_id}")
        if row.get("randomized_order_seed") != expected_seed:
            raise ValueError(f"randomized-order seed mismatch: {custom_id}")
        if row.get("route_support_status") != "supported":
            raise ValueError(f"unsupported or unverified route: {custom_id}")
        if row.get("disabled_arm_schema_visibility") not in {
            "schema_visible",
            "server_forced",
        }:
            raise ValueError(f"DISABLED schema visibility is not established: {custom_id}")
        evidence_id = row.get("schema_visibility_evidence_id")
        evidence_sha256 = row.get("schema_visibility_evidence_sha256")
        if not isinstance(evidence_id, str) or not isinstance(evidence_sha256, str):
            raise ValueError(f"schema-visibility evidence is incomplete: {custom_id}")
        evidence_files.add((evidence_id, evidence_sha256))
        body = row.get("body")
        if not isinstance(body, dict):
            raise ValueError(f"missing request body: {custom_id}")
        definition = definitions[arm]
        if row.get("mode") != definition["source_mode"]:
            raise ValueError(f"source mode mismatch for arm {arm}: {custom_id}")
        if row.get("format_condition") != definition["format_condition"]:
            raise ValueError(f"format condition mismatch for arm {arm}: {custom_id}")
        if body.get("tool_choice") != definition["tool_choice"]:
            raise ValueError(f"tool_choice mismatch: {custom_id}")
        if len(body.get("tools") or []) != 6:
            raise ValueError(f"visible six-tool schema invariant failed: {custom_id}")
        raw_route = protocol["route_policy"]["routes"][str(row["subject_id"])]
        if isinstance(raw_route, dict):
            expected_decoding = _validated_openrouter_decoding(
                raw_route.get("decoding"), str(row["subject_id"])
            )
            actual_decoding = {
                key: copy.deepcopy(body[key])
                for key in OPENROUTER_DECODING_FIELDS
                if key in body
            }
            if actual_decoding != expected_decoding:
                raise ValueError(f"route decoding mismatch: {custom_id}")
        system = next(
            str(m.get("content", ""))
            for m in body.get("messages", [])
            if isinstance(m, dict) and m.get("role") == "system"
        )
        has_instruction = system.endswith(f"\n\n{instruction}")
        if has_instruction != (row["format_condition"] == FORMAT_JSON):
            raise ValueError(f"format instruction mismatch: {custom_id}")
        if row["serialized_request_hash"] != _sha256_json(body):
            raise ValueError(f"serialized request hash mismatch: {custom_id}")
        key = _source_unit_key(row)
        if arm in cells[key]:
            raise ValueError(f"duplicate arm {arm} for cell {key}")
        cells[key][arm] = row

    for key, by_arm in cells.items():
        if set(by_arm) != set(arms):
            raise ValueError(f"incomplete generated control set {key}: {sorted(by_arm)}")
        if {"DISABLED_BASELINE", "DISABLED_FORMAT_JSON"}.issubset(by_arm):
            baseline = by_arm["DISABLED_BASELINE"]["body"]
            formatted = by_arm["DISABLED_FORMAT_JSON"]["body"]
            baseline_copy, formatted_copy = copy.deepcopy(baseline), copy.deepcopy(formatted)
            _append_format_instruction(baseline_copy, instruction)
            if baseline_copy != formatted_copy:
                raise ValueError(f"format arm changed more than the frozen instruction for {key}")
        if "TOOL_CONTEMPORANEOUS" in by_arm:
            baseline_row = by_arm["DISABLED_BASELINE"]
            tool_row = by_arm["TOOL_CONTEMPORANEOUS"]
            baseline_body = copy.deepcopy(baseline_row["body"])
            tool_body = copy.deepcopy(tool_row["body"])
            baseline_body.pop("tool_choice", None)
            tool_body.pop("tool_choice", None)
            if baseline_body != tool_body:
                raise ValueError(f"TOOL arm changed more than native callability for {key}")
            invariant_fields = (
                "route",
                "pinned_provider",
                "openrouter_provider_only",
                "openrouter_quantizations",
                "serving_precision",
                "serving_precision_value",
                "tool_schema_hash",
                "decoding_applied",
            )
            for field in invariant_fields:
                values = {_sha256_json(row.get(field)) for row in by_arm.values()}
                if len(values) != 1:
                    raise ValueError(f"cross-arm {field} mismatch for {key}")
    for evidence_id, evidence_sha256 in evidence_files:
        _verify_file_hash(REPO_ROOT / evidence_id, evidence_sha256)


def _normalized_provider_identity(value: Any) -> str:
    provider = str(value or "").split("/", 1)[0]
    return "".join(character for character in provider.lower() if character.isalnum())


def build_control_openrouter_run_record(
    *,
    row: dict[str, Any],
    body: dict[str, Any],
    response: dict[str, Any],
    pinned_provider: str,
    latency_sec: float,
    response_headers: dict[str, Any] | None = None,
    model_metadata_snapshot: dict[str, Any] | None = None,
    raw_http_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a v3 control record and fail closed on provider-pin drift."""
    observed_provider = response.get("provider")
    observed_identity = _normalized_provider_identity(observed_provider)
    pinned_identity = _normalized_provider_identity(pinned_provider)
    if not observed_identity:
        raise ValueError(
            f"OpenRouter response omitted observed provider for {row.get('custom_id')!r}"
        )
    if observed_identity != pinned_identity:
        raise ValueError(
            "observed OpenRouter provider mismatch for "
            f"{row.get('custom_id')!r}: pinned={pinned_provider!r}, "
            f"observed={observed_provider!r}"
        )
    from scripts.run_synthetic_prompts import _openrouter_run_record

    base_record = _openrouter_run_record(
        row=row,
        body=body,
        response=response,
        pinned_provider=pinned_provider,
        latency_sec=latency_sec,
        response_headers=response_headers,
        model_metadata_snapshot=model_metadata_snapshot,
        raw_http_capture=raw_http_capture,
    )
    record = dict(row)
    record.update(base_record)
    record["observed_provider"] = observed_provider
    record["observed_provider_matches_pin"] = True
    return record


def build_control_openrouter_body(
    row: dict[str, Any], openrouter_config: dict[str, Any]
) -> dict[str, Any]:
    """Normalize one frozen v3 row into its exact OpenRouter request body."""
    from scripts._tool_schema_guard import strip_route_only_fields
    from scripts.run_synthetic_prompts import _disable_openrouter_context_compression

    body = copy.deepcopy(row.get("body") or {})
    strip_route_only_fields(body)
    provider = str(openrouter_config["pinned_provider"])
    body["model"] = str(openrouter_config["slug"])
    decoding = _validated_openrouter_decoding(
        openrouter_config.get("decoding"), str(row.get("subject_id") or "unknown")
    )
    _apply_exact_openrouter_decoding(body, decoding)
    body["provider"] = {
        "only": [provider],
        "allow_fallbacks": False,
        "data_collection": "deny",
    }
    quantizations = list(openrouter_config.get("quantizations") or [])
    if quantizations:
        body["provider"]["quantizations"] = quantizations
    _disable_openrouter_context_compression(body)
    return body


def _submit_v3_openrouter_rows(
    submitter: Any,
    subject_id: str,
    entry: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    runtime_cost_cap_usd: float,
) -> dict[str, Any]:
    """Submit pinned OpenRouter rows under a resumable reported-cost cap."""
    from scripts.run_synthetic_prompts import (
        Config,
        OpenRouterFallbackApprovalRequired,
        _completed_openrouter_custom_ids,
        write_openrouter_model_metadata_snapshot,
    )

    api_key = os.environ.get(Config.OPENROUTER_AUTH_ENV, "")
    if not api_key:
        raise OpenRouterFallbackApprovalRequired(
            f"{Config.OPENROUTER_AUTH_ENV} is required for the frozen OpenRouter route"
        )
    route = entry["openrouter"]
    provider = str(route["pinned_provider"])
    out_path = submitter.output_dir / f"responses_{subject_id}_openrouter.jsonl"
    submitter.output_dir.mkdir(parents=True, exist_ok=True)
    completed_ids = _completed_openrouter_custom_ids(out_path)
    reported_cost_usd = _existing_openrouter_spend(out_path)
    needs_dispatch = any(str(row["custom_id"]) not in completed_ids for row in rows)
    snapshot = None
    if needs_dispatch:
        snapshot = write_openrouter_model_metadata_snapshot(
            session=submitter.session,
            api_key=api_key,
            output_dir=submitter.output_dir / "openrouter_model_metadata",
            model_slug=str(route["slug"]),
            pinned_provider=provider,
        )
    completed = 0
    skipped = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            custom_id = str(row["custom_id"])
            if custom_id in completed_ids:
                skipped += 1
                continue
            next_request_ceiling = float(
                _estimated_cost_upper_bound([row], protocol)["total_api_usd"]
            )
            if reported_cost_usd + next_request_ceiling > runtime_cost_cap_usd + 1e-9:
                return {
                    "subject_id": subject_id,
                    "status": "paused_budget_cap",
                    "final_route": "openrouter",
                    "response_path": str(out_path),
                    "responses": completed,
                    "skipped_existing": skipped,
                    "reported_cost_usd": round(reported_cost_usd, 8),
                    "runtime_cost_cap_usd": runtime_cost_cap_usd,
                    "next_request_cost_ceiling_usd": next_request_ceiling,
                    "next_custom_id": custom_id,
                }
            body = build_control_openrouter_body(row, route)
            started = time.perf_counter()
            response, response_headers, raw_http_capture = (
                submitter._post_openrouter_with_retries(
                    api_key,
                    body,
                    custom_id=custom_id,
                    subject_id=subject_id,
                    provider=provider,
                )
            )
            record = build_control_openrouter_run_record(
                row=row,
                body=body,
                response=response,
                pinned_provider=provider,
                latency_sec=time.perf_counter() - started,
                response_headers=response_headers,
                model_metadata_snapshot=snapshot,
                raw_http_capture=raw_http_capture,
            )
            try:
                response_cost = _openrouter_reported_cost(
                    response, custom_id=custom_id
                )
            except ValueError as exc:
                record["execution_budget"] = {
                    "runtime_cost_cap_usd": runtime_cost_cap_usd,
                    "reconciliation_error": str(exc),
                }
                handle.write(
                    json.dumps(
                        record,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                raise
            reported_cost_usd += response_cost
            record["execution_budget"] = {
                "response_reported_cost_usd": response_cost,
                "cumulative_reported_cost_usd": round(reported_cost_usd, 8),
                "runtime_cost_cap_usd": runtime_cost_cap_usd,
            }
            handle.write(
                json.dumps(
                    record,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            completed += 1
            completed_ids.add(custom_id)
    return {
        "subject_id": subject_id,
        "status": "completed",
        "final_route": "openrouter",
        "response_path": str(out_path),
        "responses": completed,
        "skipped_existing": skipped,
        "reported_cost_usd": round(reported_cost_usd, 8),
        "runtime_cost_cap_usd": runtime_cost_cap_usd,
    }


def _estimated_cost_upper_bound(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    native_payload = _read_json(NATIVE_PRICING_PATH)
    openrouter_payload = _read_json(OPENROUTER_PRICING_PATH)
    native_rows = native_payload.get("pricing")
    openrouter_rows = openrouter_payload.get("models")
    if not isinstance(native_rows, list) or not isinstance(openrouter_rows, list):
        raise ValueError("current pricing overlays must contain pricing/models arrays")
    native_prices = {
        str(price["model_key"]): price for price in native_rows if isinstance(price, dict)
    }
    openrouter_prices = {
        str(price["model_key"]): price for price in openrouter_rows if isinstance(price, dict)
    }
    per_subject: dict[str, float] = defaultdict(float)
    token_bounds: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
    for row in rows:
        subject = str(row["subject_id"])
        route, _ = _route_spec(protocol, subject)
        if route == "local":
            input_rate = output_rate = 0.0
        elif route == "native":
            price = native_prices.get(subject)
            if not isinstance(price, dict):
                raise ValueError(f"missing current native price entry for {subject}")
            input_value = price.get("batch_api_input_usd_per_1m_tokens")
            output_value = price.get("batch_api_output_usd_per_1m_tokens")
            if input_value is None:
                input_value = price.get("regular_api_input_usd_per_1m_tokens")
            if output_value is None:
                output_value = price.get("regular_api_output_usd_per_1m_tokens")
            if input_value is None or output_value is None:
                raise ValueError(f"incomplete current native price entry for {subject}")
            input_rate = float(input_value)
            output_rate = float(output_value)
        else:
            price = openrouter_prices.get(subject)
            if not isinstance(price, dict):
                raise ValueError(f"missing current OpenRouter price entry for {subject}")
            input_rate = float(price["input_usd_per_1m_tokens"])
            output_rate = float(price["output_usd_per_1m_tokens"])
        body = row["body"]
        input_upper = len(_json_bytes(body))
        output_upper = int(body.get("max_completion_tokens", body.get("max_tokens", 0)) or 0)
        token_bounds[subject]["input"] += input_upper
        token_bounds[subject]["output"] += output_upper
        per_subject[subject] += (input_upper * input_rate + output_upper * output_rate) / 1_000_000
    return {
        "method": "conservative UTF-8-byte input-token ceiling plus configured maximum output tokens; route-specific current pricing overlays; excludes local GPU rental",
        "pricing_snapshot": {
            "native": {
                "path": str(NATIVE_PRICING_PATH.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(NATIVE_PRICING_PATH),
            },
            "openrouter": {
                "path": str(OPENROUTER_PRICING_PATH.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(OPENROUTER_PRICING_PATH),
            },
            "frozen_roster_used_for_pricing": False,
        },
        "per_subject_usd": {key: round(value, 6) for key, value in sorted(per_subject.items())},
        "total_api_usd": round(sum(per_subject.values()), 6),
        "token_upper_bounds": dict(sorted(token_bounds.items())),
    }


def _required_artifact_path(protocol: dict[str, Any], key: str) -> Path | None:
    artifact_paths = protocol.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        return None
    value = artifact_paths.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"protocol artifact path {key} is missing")
    return REPO_ROOT / value


def validate_execution_authorization(
    path: Path | None,
    *,
    protocol: dict[str, Any],
    protocol_sha256: str,
    locked_inputs_manifest_path: Path,
    subject_id: str,
) -> dict[str, Any]:
    """Validate the immutable, post-OSF receipt required before v3 dispatch."""
    if path is None:
        raise ValueError("v3 submission requires --execution-authorization")
    if not path.is_file():
        raise FileNotFoundError(path)
    authorization = _read_json(path)
    if authorization.get("submission_permitted") is not True:
        raise ValueError("execution authorization must set submission_permitted=true")
    if authorization.get("schema_version") != (
        "disabled_format_control.execution_authorization.v1"
    ):
        raise ValueError("unsupported execution-authorization schema_version")
    if authorization.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("execution authorization protocol_id mismatch")
    if authorization.get("protocol_sha256") != protocol_sha256:
        raise ValueError("execution authorization protocol hash mismatch")
    if authorization.get("osf_filed_protocol_sha256") != protocol_sha256:
        raise ValueError("OSF-filed protocol hash does not match the frozen protocol")

    lock_value = authorization.get("locked_inputs_manifest")
    if not isinstance(lock_value, str) or not lock_value.strip():
        raise ValueError("execution authorization is missing locked_inputs_manifest")
    recorded_lock_path = Path(lock_value)
    if not recorded_lock_path.is_absolute():
        recorded_lock_path = REPO_ROOT / recorded_lock_path
    if recorded_lock_path.resolve() != locked_inputs_manifest_path.resolve():
        raise ValueError("execution authorization locked-input path mismatch")
    actual_lock_sha256 = _sha256_file(locked_inputs_manifest_path)
    if authorization.get("locked_inputs_manifest_sha256") != actual_lock_sha256:
        raise ValueError("execution authorization locked-input hash mismatch")

    osf_url = authorization.get("osf_registration_url")
    if not isinstance(osf_url, str) or not osf_url.startswith("https://osf.io/"):
        raise ValueError("execution authorization needs an https://osf.io/ registration URL")
    filed_at = authorization.get("osf_filed_at_utc")
    if not isinstance(filed_at, str) or not filed_at.strip():
        raise ValueError("execution authorization is missing osf_filed_at_utc")
    try:
        datetime.fromisoformat(filed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("execution authorization has invalid osf_filed_at_utc") from exc

    frozen_subjects = [str(value) for value in protocol["population"]["subjects"]]
    authorized_subjects = authorization.get("authorized_subjects")
    if not isinstance(authorized_subjects, list) or list(
        map(str, authorized_subjects)
    ) != frozen_subjects:
        raise ValueError("execution authorization subject population mismatch")
    if subject_id not in set(map(str, authorized_subjects)):
        raise ValueError(f"execution authorization excludes subject {subject_id}")
    authorized_by = authorization.get("authorized_by")
    if not isinstance(authorized_by, str) or not authorized_by.strip():
        raise ValueError("execution authorization is missing authorized_by")
    spending = authorization.get("spending_authorization")
    if not isinstance(spending, dict) or not spending:
        raise ValueError("execution authorization is missing spending_authorization")
    openrouter_cap = spending.get("openrouter_cap_usd")
    openrouter_spent = spending.get("openrouter_spent_before_official_usd")
    openrouter_allocations = spending.get("openrouter_profile_subject_caps_usd")
    if (
        isinstance(openrouter_cap, bool)
        or not isinstance(openrouter_cap, (int, float))
        or not math.isfinite(float(openrouter_cap))
        or float(openrouter_cap) <= 0
    ):
        raise ValueError("execution authorization needs a positive OpenRouter cap")
    if (
        isinstance(openrouter_spent, bool)
        or not isinstance(openrouter_spent, (int, float))
        or not math.isfinite(float(openrouter_spent))
        or float(openrouter_spent) < 0
        or float(openrouter_spent) > float(openrouter_cap)
    ):
        raise ValueError("execution authorization has invalid prior OpenRouter spend")
    if not isinstance(openrouter_allocations, dict):
        raise ValueError("execution authorization needs OpenRouter profile allocations")
    allocation_total = 0.0
    for profile in ("canary", "full"):
        profile_allocations = openrouter_allocations.get(profile)
        if not isinstance(profile_allocations, dict):
            raise ValueError(
                f"execution authorization needs OpenRouter {profile} allocations"
            )
        for allocated_subject, value in profile_allocations.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(
                    "execution authorization has an invalid OpenRouter allocation "
                    f"for {profile}/{allocated_subject}"
                )
            allocation_total += float(value)
    remaining = float(openrouter_cap) - float(openrouter_spent)
    if allocation_total > remaining + 1e-9:
        raise ValueError(
            "execution authorization OpenRouter allocations exceed the remaining cap"
        )
    routes = protocol.get("route_policy", {}).get("routes", {})
    route_value = routes.get(subject_id) if isinstance(routes, dict) else None
    route_type = (
        str(route_value.get("route") or "")
        if isinstance(route_value, dict)
        else str(route_value or "").partition(":")[0]
    )
    if route_type == "openrouter":
        for profile in ("canary", "full"):
            profile_allocations = openrouter_allocations[profile]
            value = profile_allocations.get(subject_id)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) <= 0
            ):
                raise ValueError(
                    "execution authorization needs a positive OpenRouter allocation "
                    f"for {profile}/{subject_id}"
                )
    if subject_id == "kimi-k2.6" and route_type == "native":
        if spending.get("kimi_native_batch") is not True:
            raise ValueError("execution authorization does not authorize Kimi native batch")
        kimi_cap = spending.get("kimi_native_batch_cap_usd")
        kimi_spent = spending.get("kimi_native_batch_spent_before_official_usd")
        kimi_allocations = spending.get("kimi_native_batch_profile_caps_usd")
        if (
            isinstance(kimi_cap, bool)
            or not isinstance(kimi_cap, (int, float))
            or not math.isfinite(float(kimi_cap))
            or float(kimi_cap) <= 0
        ):
            raise ValueError("execution authorization needs a positive Kimi batch cap")
        if (
            isinstance(kimi_spent, bool)
            or not isinstance(kimi_spent, (int, float))
            or not math.isfinite(float(kimi_spent))
            or float(kimi_spent) < 0
            or float(kimi_spent) > float(kimi_cap)
        ):
            raise ValueError("execution authorization has invalid prior Kimi spend")
        if not isinstance(kimi_allocations, dict):
            raise ValueError("execution authorization needs Kimi batch profile allocations")
        kimi_total = 0.0
        for profile in ("canary", "full"):
            value = kimi_allocations.get(profile)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    f"execution authorization needs a positive Kimi {profile} allocation"
                )
            kimi_total += float(value)
        if kimi_total > float(kimi_cap) - float(kimi_spent) + 1e-9:
            raise ValueError(
                "execution authorization Kimi allocations exceed the remaining Kimi cap"
            )
    return authorization


def _authorized_profile_subject_cap(
    authorization: dict[str, Any], *, profile: str, subject_id: str
) -> float:
    spending = authorization["spending_authorization"]
    if subject_id == "kimi-k2.6" and spending.get("kimi_native_batch") is True:
        return float(spending["kimi_native_batch_profile_caps_usd"][profile])
    allocations = spending["openrouter_profile_subject_caps_usd"]
    return float(allocations[profile][subject_id])


def _openrouter_reported_cost(response: dict[str, Any], *, custom_id: str) -> float:
    usage = response.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
    ):
        raise ValueError(
            f"OpenRouter response {custom_id!r} has no finite non-negative usage.cost"
        )
    return float(cost)


def _existing_openrouter_spend(path: Path) -> float:
    if not path.is_file():
        return 0.0
    total = 0.0
    for record in _iter_jsonl(path):
        total += _openrouter_reported_cost(
            record, custom_id=str(record.get("custom_id") or "unknown")
        )
    return total


def validate_protocol_output_target(
    protocol: dict[str, Any], output_dir: Path, *, canary: bool
) -> None:
    key = "canary_batches_dir" if canary else "full_batches_dir"
    required = _required_artifact_path(protocol, key)
    if required is not None and output_dir.resolve() != required.resolve():
        raise ValueError(
            f"v3 {key} is frozen as {required}; refusing colliding output target {output_dir}"
        )


def generate(
    protocol_path: Path,
    source_dir: Path,
    output_dir: Path | None,
    subjects: list[str] | None,
    *,
    canary: bool,
    write: bool,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(protocol_path)
    if write:
        if output_dir is None:
            raise ValueError("output_dir is required when writing")
        validate_protocol_output_target(protocol, output_dir, canary=canary)
    selected = _selected_subjects(protocol, subjects)
    gpt_native = {}
    if "gpt-5.5" in selected and _route_spec(protocol, "gpt-5.5")[0] == "native":
        gpt_native = _load_gpt_native_rows(protocol)
    all_rows: list[dict[str, Any]] = []
    per_subject: dict[str, list[dict[str, Any]]] = {}
    for subject in selected:
        rows = build_subject_rows(
            protocol,
            protocol_sha,
            source_dir,
            subject,
            canary=canary,
            gpt_native=gpt_native,
        )
        per_subject[subject] = rows
        all_rows.extend(rows)
    arm_counts = Counter(str(row["control_arm"]) for row in all_rows)
    route_counts = Counter(str(row["route"]) for row in all_rows)
    manifest: dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "profile": "canary" if canary else "full",
        "analysis_inclusion": not canary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "subjects": selected,
        "row_count": len(all_rows),
        "per_subject_rows": {key: len(value) for key, value in per_subject.items()},
        "arm_counts": dict(sorted(arm_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "estimated_cost_upper_bound": _estimated_cost_upper_bound(all_rows, protocol),
        "submission_status": "not_submitted",
        "output_files": {},
    }
    if not canary and set(selected) == set(protocol["population"]["subjects"]):
        expected = int(protocol["population"]["expected_rows_total"])
        if len(all_rows) != expected:
            raise ValueError(f"full frozen population expected {expected} rows, got {len(all_rows)}")
    if write:
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        for subject, rows in per_subject.items():
            path = output_dir / f"input_{subject}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(_json_bytes(row).decode("utf-8") + "\n")
            manifest["output_files"][path.name] = {
                "rows": len(rows),
                "sha256": _sha256_file(path),
            }
        manifest_path = output_dir / "disabled_format_control_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        manifest["manifest_path"] = str(manifest_path)
    return manifest


def _load_generated_rows(
    output_dir: Path,
    subject: str,
    manifest: dict[str, Any],
    locked_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    filename = f"input_{subject}.jsonl"
    generated = manifest.get("output_files", {}).get(filename)
    locked = locked_profile.get("output_files", {}).get(filename)
    if not isinstance(generated, dict):
        raise ValueError(f"generated manifest has no locked output for {subject}")
    if not isinstance(locked, dict):
        raise ValueError(f"locked-input manifest has no output for {subject}")
    if generated != locked:
        raise ValueError(f"generated manifest entry does not match locked input for {subject}")
    path = output_dir / filename
    _verify_file_hash(path, str(locked["sha256"]))
    rows = list(_iter_jsonl(path))
    if len(rows) != int(locked["rows"]):
        raise ValueError(f"generated row count mismatch for {subject}")
    return rows


def submit_generated(args: argparse.Namespace) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(args.protocol)
    required_authorization = None
    if isinstance(protocol.get("artifact_paths"), dict):
        validate_protocol_output_target(
            protocol,
            args.output_dir,
            canary=args.profile == "canary",
        )
        required_lock = _required_artifact_path(protocol, "locked_inputs_manifest")
        if (
            required_lock is not None
            and args.locked_inputs_manifest.resolve() != required_lock.resolve()
        ):
            raise ValueError(
                f"v3 locked manifest is frozen as {required_lock}; "
                f"got {args.locked_inputs_manifest}"
            )
        if "execution_authorization" in protocol["artifact_paths"]:
            required_authorization = _required_artifact_path(
                protocol, "execution_authorization"
            )
        supplied_authorization = getattr(args, "execution_authorization", None)
        if required_authorization is not None:
            if supplied_authorization is None:
                raise ValueError("v3 submission requires --execution-authorization")
            if supplied_authorization.resolve() != required_authorization.resolve():
                raise ValueError(
                    f"v3 execution authorization is frozen as {required_authorization}; "
                    f"got {supplied_authorization}"
                )
    if not args.subject or len(args.subject) != 1:
        raise ValueError("paid submission requires exactly one --subject per invocation")
    subject = args.subject[0]
    _selected_subjects(protocol, [subject])
    if not args.acknowledge_postregistration_exploratory:
        raise ValueError("pass --acknowledge-postregistration-exploratory")
    if args.confirm_spend != protocol["protocol_id"]:
        raise ValueError(f"--confirm-spend must exactly equal {protocol['protocol_id']}")
    if args.max_requests is None or args.max_requests <= 0:
        raise ValueError("paid submission requires a positive --max-requests")
    if (
        args.max_cost_usd is None
        or not math.isfinite(args.max_cost_usd)
        or args.max_cost_usd < 0
    ):
        raise ValueError("paid submission requires a finite non-negative --max-cost-usd")
    manifest_path = args.output_dir / "disabled_format_control_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("protocol_sha256") != protocol_sha:
        raise ValueError("generated manifest protocol hash does not match the frozen protocol")
    if manifest.get("profile") != args.profile:
        raise ValueError(
            f"--profile {args.profile!r} does not match generated manifest profile "
            f"{manifest.get('profile')!r}"
        )
    locked_inputs = _read_json(args.locked_inputs_manifest)
    if locked_inputs.get("protocol_id") != protocol["protocol_id"]:
        raise ValueError("locked-input manifest protocol_id does not match the frozen protocol")
    if locked_inputs.get("protocol_sha256") != protocol_sha:
        raise ValueError("locked-input manifest protocol hash does not match the frozen protocol")
    locked_profile = locked_inputs.get(args.profile)
    if not isinstance(locked_profile, dict):
        raise ValueError(f"locked-input manifest has no {args.profile!r} profile")
    authorization = None
    if required_authorization is not None:
        authorization = validate_execution_authorization(
            getattr(args, "execution_authorization", None),
            protocol=protocol,
            protocol_sha256=protocol_sha,
            locked_inputs_manifest_path=args.locked_inputs_manifest,
            subject_id=subject,
        )
    rows = _load_generated_rows(args.output_dir, subject, manifest, locked_profile)
    validate_generated_rows(rows, protocol, canary=args.profile == "canary")
    if len(rows) != args.max_requests:
        raise ValueError(
            f"request gate must exactly equal the locked subject row count: "
            f"{args.max_requests} != {len(rows)}"
        )
    route, _detail = _route_spec(protocol, subject)
    cost = _estimated_cost_upper_bound(rows, protocol)
    authorized_cost_cap = None
    if authorization is not None:
        authorized_cost_cap = _authorized_profile_subject_cap(
            authorization, profile=args.profile, subject_id=subject
        )
        if args.max_cost_usd > authorized_cost_cap + 1e-9:
            raise ValueError(
                f"requested ${args.max_cost_usd:.8f} cost cap exceeds the authorized "
                f"{args.profile}/{subject} cap of ${authorized_cost_cap:.8f}"
            )
    runtime_openrouter_budget = authorization is not None and route == "openrouter"
    if (
        not runtime_openrouter_budget
        and float(cost["total_api_usd"]) > args.max_cost_usd
    ):
        raise ValueError(
            f"cost gate exceeded: conservative ${cost['total_api_usd']:.6f} > ${args.max_cost_usd:.6f}"
        )
    submit_manifest_path = args.output_dir / f"submit_manifest_{subject}.json"
    if route == "native" and submit_manifest_path.exists():
        raise ValueError(
            f"native submission ledger already exists at {submit_manifest_path}; "
            "refusing a potentially duplicate provider batch"
        )
    if route == "local":
        responses_dir = args.output_dir.parent / "responses"
        command = (
            "python3 scripts/run_local_subjects.py "
            f"--batches-dir {args.output_dir} --results-dir {responses_dir} "
            f"--subjects {subject} --serving-precision-role primary "
            "--vllm-endpoint <verified-endpoint> --checkpoint-manifest-dir <verified-manifests>"
        )
        raise ValueError(f"local rows require the checkpoint-guarded local runner: {command}")

    from scripts.cagap_env import load_cagap_env
    from scripts.run_synthetic_prompts import (
        Config,
        NativeBatchCreditExhaustionError,
        NativeBatchSubmissionError,
        NativeBatchSubmitter,
        select_route,
    )

    load_cagap_env()
    entry = copy.deepcopy(Config.SUBJECTS[subject])
    entry["route_override"] = route
    descriptor = _route_descriptor(protocol, subject)
    if "model_slug" in descriptor:
        if route == "openrouter":
            entry["openrouter"] = {
                "slug": descriptor["model_slug"],
                "pinned_provider": descriptor["provider_only"][0],
                "quantizations": list(descriptor.get("quantizations", [])),
                "decoding": copy.deepcopy(descriptor["decoding"]),
            }
        elif route == "native":
            native = copy.deepcopy(entry.get("native") or {})
            expected_native = {
                "endpoint": descriptor["endpoint"],
                "url_path": descriptor["request_path"],
                "batch_url_path": descriptor["batch_url_path"],
                "batch_input_files_path": descriptor["batch_input_files_path"],
                "slug": descriptor["model_slug"],
            }
            for key, expected_value in expected_native.items():
                if native.get(key) != expected_value:
                    raise ValueError(
                        f"frozen native route mismatch for {subject} {key}: "
                        f"protocol={expected_value!r}, roster={native.get(key)!r}"
                    )
            native["decoding"] = copy.deepcopy(descriptor["decoding"])
            entry["native"] = native
    resolved, _ = select_route(entry)
    if resolved != route:
        raise ValueError(f"frozen route {route} resolved as {resolved}; refusing submission")
    submitter = NativeBatchSubmitter(
        output_dir=args.output_dir.parent / "responses",
        confirm_openrouter_fallback=lambda _subject, _reason: False,
    )
    submission_gate = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "subject_id": subject,
        "profile": args.profile,
        "request_gate": args.max_requests,
        "cost_gate_usd": args.max_cost_usd,
        "authorized_cost_cap_usd": authorized_cost_cap,
        "cost_gate_mode": (
            "per_response_reported_cost_with_next_request_reserve"
            if runtime_openrouter_budget
            else "whole_submission_conservative_ceiling"
        ),
        "estimated_cost_upper_bound": cost,
        "locked_input_sha256": locked_profile["output_files"][f"input_{subject}.jsonl"][
            "sha256"
        ],
    }
    if authorization is not None:
        submission_gate.update(
            {
                "execution_authorization_path": str(
                    getattr(args, "execution_authorization")
                ),
                "execution_authorization_sha256": _sha256_file(
                    getattr(args, "execution_authorization")
                ),
                "osf_registration_url": authorization["osf_registration_url"],
            }
        )
    if route == "native":
        submit_manifest_path.write_text(
            json.dumps(
                {**submission_gate, "submission_status": "submission_started"},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    try:
        if route == "native":
            result = submitter.submit_subject_rows(subject, entry, rows)
        elif "model_slug" in descriptor:
            result = _submit_v3_openrouter_rows(
                submitter,
                subject,
                entry,
                rows,
                protocol=protocol,
                runtime_cost_cap_usd=float(args.max_cost_usd),
            )
        else:
            result = submitter._submit_openrouter_rows(subject, entry, rows)
    except (
        NativeBatchCreditExhaustionError,
        NativeBatchSubmissionError,
    ) as exc:
        if route == "native":
            submit_manifest_path.write_text(
                json.dumps(
                    {
                        **submission_gate,
                        "submission_status": "failed_or_indeterminate",
                        "error_type": type(exc).__name__,
                    },
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        raise ValueError(
            f"frozen {route} route failed; no provider substitution was attempted: {exc}"
        ) from exc
    submit_manifest = {
        **result,
        **submission_gate,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "submission_status": result.get("status", "submitted"),
    }
    submit_manifest_path.write_text(
        json.dumps(submit_manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return submit_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate", action="store_true", help="verify frozen inputs only; no writes/API")
    actions.add_argument("--dry-run-canary", action="store_true", help="build one excluded cell per subject in memory; no writes/API")
    actions.add_argument("--generate", action="store_true", help="write frozen request files; no API")
    actions.add_argument("--submit", action="store_true", help="submit one already-generated subject with explicit spend gates")
    actions.add_argument(
        "--validate-protocol-draft",
        action="store_true",
        help="validate a draft protocol structure and closed gates; no generation/API",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--locked-inputs-manifest", type=Path, default=DEFAULT_LOCKED_INPUTS_MANIFEST
    )
    parser.add_argument(
        "--execution-authorization",
        type=Path,
        default=None,
        help="immutable post-OSF execution receipt required by promoted v3 protocols",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", choices=("full", "canary"), default="full")
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--acknowledge-postregistration-exploratory", action="store_true")
    parser.add_argument("--confirm-spend", default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_protocol_draft:
            result = validate_protocol_draft(args.protocol)
        elif args.submit:
            result = submit_generated(args)
        else:
            dry_canary = args.dry_run_canary
            result = generate(
                args.protocol,
                args.source_dir,
                args.output_dir if args.generate else None,
                args.subject,
                canary=dry_canary or args.profile == "canary",
                write=args.generate,
            )
            if args.validate:
                result["validation_only"] = True
            if dry_canary:
                result["dry_run"] = True
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
