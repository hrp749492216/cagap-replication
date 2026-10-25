"""Generate and optionally submit benign capability battery rows.

The benign battery is descriptive and separate from the H1 main-run lane. By
default, this runner expands the locked 30-prompt battery to:

    30 prompts x 10 subjects x 4 modes x 1 neutral persona x 1 trial = 1,200 rows

The ``benign_tool_hardening_utility`` profile reuses the same prompts and
subjects for a post-registration paired utility check:

    30 prompts x 10 subjects x 1 TOOL mode x 2 system prompts = 600 rows

Rows reuse the main synthetic runner's roster, route selection, request body
construction, decoding parameters, route provenance, and native-batch submitter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from scripts.run_synthetic_prompts import (
    BASELINE_INTERVENTION,
    Config as MainConfig,
    EMPTY_SHA256,
    ExperimentRunner,
    HARDENED_INTERVENTION,
    NativeBatchSubmitter,
    _hash_tool_subset,
    _request_tool_schema,
    _system_prompt_for_condition,
    make_custom_id,
    select_irrelevant_tool_subset,
    select_route,
)

logger = logging.getLogger(__name__)

BATTERY_PARTITION_ID = "benign_capability_battery"
BATTERY_INTERVENTION = "descriptive"
BATTERY_PERSONA = "neutral"
BATTERY_TRIAL_ID = 0
DESCRIPTIVE_CAPABILITY_PROFILE = "descriptive_1200"
BENIGN_TOOL_HARDENING_UTILITY_PROFILE = "benign_tool_hardening_utility"
VALID_PROFILES = {
    DESCRIPTIVE_CAPABILITY_PROFILE,
    BENIGN_TOOL_HARDENING_UTILITY_PROFILE,
}

# Locked from outputs/main_run/20260525T164059Z/
# derived_route_provenance_resolved/route_provenance_overlay_manifest.json.
# This is the actual main-run subject route provenance, which differs from
# today's environment-dependent roster selection for Claude/GPT-5.5.
MAIN_RUN_ROUTE_LOCK: dict[str, str] = {
    "claude-opus-4-7": "openrouter",
    "gpt-5.5": "native",
    "gemini-3.5-flash": "openrouter",
    "grok-4.3": "native",
    "kimi-k2.6": "native",
    "llama-3.3-70b-instruct": "local",
    "gemma-4-31b-it": "local",
    "qwen3.6-35b-a3b": "local",
    "glm-4.7-flash": "local",
    "mistral-medium-3.5-128b": "local",
}

MAIN_RUN_ROUTE_LOCK_SOURCE = (
    "outputs/main_run/20260525T164059Z/"
    "derived_route_provenance_resolved/route_provenance_overlay_manifest.json"
)

MAIN_RUN_PROVENANCE_LOCK: dict[str, dict[str, Any]] = {
    "claude-opus-4-7": {
        "route": "openrouter",
        "provider": "anthropic",
        "pinned_provider": "anthropic",
        "model": "anthropic/claude-opus-4.7",
        "serving_precision": "provider_unspecified",
        "serving_precision_value": "provider_unspecified",
        "serving_precision_role": "primary",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": "claude-opus-4-7::openrouter",
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-19/schema_visibility_smoke/"
            "20260519T223440Z/schema_visibility_classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "120c66ab61e8d811530b128b5b176fd16fa056025205ed6d12390e3b001cb260"
        ),
    },
    "gpt-5.5": {
        "route": "openai_native_batch",
        "provider": "openai",
        "pinned_provider": "openai",
        "route_family": "native_batch",
        "serving_precision": "provider_unspecified",
        "serving_precision_value": "provider_unspecified",
        "serving_precision_role": "primary",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": "gpt-5.5::native_BATCH",
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-26/schema_visibility_smoke/"
            "20260526T023941Z_gpt55_native_batch/schema_visibility_classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "6507c7ebe83c8d9d2a0973d152cfffebdf4df8cbf4143294ea241ec66b95b87b"
        ),
    },
    "gemini-3.5-flash": {
        "route": "openrouter",
        "provider": "google-ai-studio",
        "pinned_provider": "google-ai-studio",
        "model": "google/gemini-3.5-flash",
        "serving_precision": "provider_unspecified",
        "serving_precision_value": "provider_unspecified",
        "serving_precision_role": "primary",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": None,
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-20/schema_visibility_smoke/"
            "gemini35_flash_20260520T004107Z/classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "ccc6dae95c9a83ee53a0334ccdabff8b169b083e14db3b8458876ff577b60962"
        ),
    },
    "grok-4.3": {
        "route": "native_batch",
        "provider": "xai",
        "pinned_provider": "xai",
        "route_family": "native_batch",
        "serving_precision": "provider_unspecified",
        "serving_precision_value": "provider_unspecified",
        "serving_precision_role": "primary",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": "grok-4.3::native",
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-19/schema_visibility_smoke/"
            "20260519T223440Z/schema_visibility_classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "120c66ab61e8d811530b128b5b176fd16fa056025205ed6d12390e3b001cb260"
        ),
    },
    "kimi-k2.6": {
        "route": "native_batch",
        "provider": "moonshot",
        "pinned_provider": "moonshot",
        "route_family": "native_batch",
        "serving_precision": "provider_unspecified",
        "serving_precision_value": "provider_unspecified",
        "serving_precision_role": "primary",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": "kimi-k2.6::native",
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-19/schema_visibility_smoke/"
            "20260519T230041Z/schema_visibility_classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "9324da132aa6efcf3576850254311f3432a9d2dd012c381a2f2698956e85a453"
        ),
    },
    "llama-3.3-70b-instruct": {
        "route": "local",
        "provider": "local",
        "pinned_provider": "local",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "serving_precision": "fp16",
        "serving_precision_value": "fp16",
        "serving_precision_role": "primary",
        "served_model_path": "meta-llama/Llama-3.3-70B-Instruct",
        "served_model_location": "external_gpu",
        "served_model_runtime": "vllm_float16",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": None,
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-19/schema_visibility_smoke/"
            "verda_llama_bf16_20260519T235759Z/classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "f943ac1b3dbc80f51c3ef178cea5da08f6ffde2292ad53138cc8d4ce0449b2fe"
        ),
    },
    "gemma-4-31b-it": {
        "route": "local",
        "provider": "local",
        "pinned_provider": "local",
        "model": "google/gemma-4-31B-it",
        "serving_precision": "bf16",
        "serving_precision_value": "bf16",
        "serving_precision_role": "primary",
        "served_model_path": "google/gemma-4-31B-it",
        "served_model_location": "external_gpu",
        "served_model_runtime": "vllm_bf16",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": None,
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-19/schema_visibility_smoke/"
            "verda_gemma_bf16_20260519T234739Z/classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "9539c239160a1d08049035fb435ede767bd14c96d4a138656747913d65b76541"
        ),
    },
    "qwen3.6-35b-a3b": {
        "route": "local",
        "provider": "local",
        "pinned_provider": "local",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "serving_precision": "bf16",
        "serving_precision_value": "bf16",
        "serving_precision_role": "primary",
        "served_model_path": "Qwen/Qwen3.6-35B-A3B",
        "served_model_location": "external_gpu",
        "served_model_runtime": "vllm_bf16",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": None,
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-19/schema_visibility_smoke/"
            "verda_qwen_bf16_20260519T233902Z/classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "1017ea94c2db8265fdf4b624a2cf1d0de885166d7502d2b08c2120e66001be9d"
        ),
    },
    "glm-4.7-flash": {
        "route": "local",
        "provider": "local",
        "pinned_provider": "local",
        "model": "zai-org/GLM-4.7-Flash",
        "serving_precision": "bf16",
        "serving_precision_value": "bf16",
        "serving_precision_role": "primary",
        "served_model_path": "zai-org/GLM-4.7-Flash",
        "served_model_location": "external_gpu",
        "served_model_runtime": "vllm_bf16",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": None,
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-20/schema_visibility_smoke/"
            "lambda_glm-4.7-flash-bf16-tp2-fa2-tritonmoe-verda-b300_20260520T203312Z/"
            "classification_completed_b200_20260520T210531Z.json"
        ),
        "schema_visibility_evidence_sha256": (
            "bad020ea6241f1f58a5b17ab3ae2e53d4d512a1749d307edde9f48fc102616b8"
        ),
    },
    "mistral-medium-3.5-128b": {
        "route": "local",
        "provider": "local",
        "pinned_provider": "local",
        "model": "mistralai/Mistral-Medium-3.5-128B",
        "serving_precision": "fp8_weights_bf16_compute",
        "serving_precision_value": "fp8_weights_bf16_compute",
        "serving_precision_role": "primary",
        "served_model_path": "mistralai/Mistral-Medium-3.5-128B",
        "served_model_location": "external_gpu",
        "served_model_runtime": "vllm_fp8_bf16_compute",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_route_label": None,
        "schema_visibility_evidence_id": (
            "outputs/pilot/2026-05-20/schema_visibility_smoke/"
            "lambda_mistral-medium-3.5-128b-official-fp8-bf16-compute_20260520T192456Z/"
            "classification.json"
        ),
        "schema_visibility_evidence_sha256": (
            "123f877f18b125452f38c7ca84c3761acbaf867ad6da3c9211611cd1d63d8392"
        ),
    },
}


class BatteryConfig:
    PROMPTS_PATH = Path("experiments/benign_capability_battery/prompts.jsonl")
    PROMPTS_SHA256_PATH = Path("experiments/benign_capability_battery/prompts.sha256")
    BATCHES_DIR = Path("outputs/benign_capability_battery/batches")
    HARDENING_UTILITY_BATCHES_DIR = Path(
        "outputs/benign_capability_battery/hardening_utility_600/batches"
    )


def _json_dumps(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_prompt_hash(
    prompts_path: Path = BatteryConfig.PROMPTS_PATH,
    sha256_path: Path = BatteryConfig.PROMPTS_SHA256_PATH,
) -> str:
    expected = sha256_path.read_text(encoding="utf-8").split()[0]
    actual = _sha256_file(prompts_path)
    if actual != expected:
        raise ValueError(
            f"benign battery prompt hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def load_battery_prompts(path: Path = BatteryConfig.PROMPTS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            for key in (
                "benign_id",
                "domain",
                "benign_target_tool",
                "canonical_args",
                "prompt_text",
            ):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing {key}")
            rows.append(row)
    return rows


def benign_domain_tools(
    *,
    schemas: dict[str, list[dict[str, Any]]],
    manifest: list[dict[str, Any]],
    domain: str,
) -> list[dict[str, Any]]:
    domain_lower = domain.lower()
    benign_names = [
        entry["tool_name"]
        for entry in manifest
        if str(entry.get("domain", "")).lower() == domain_lower
        and entry.get("risk_class") == "benign"
    ]
    tools_by_name = {
        tool["name"]: tool
        for tool in schemas.get(domain_lower, [])
        if isinstance(tool.get("name"), str)
    }
    tools = [tools_by_name[name] for name in benign_names if name in tools_by_name]
    missing = sorted(set(benign_names) - set(tools_by_name))
    if missing:
        raise ValueError(f"manifest references missing benign tools for {domain}: {missing}")
    if not tools:
        raise ValueError(f"no benign tools found for {domain}")
    return tools


def _profile_battery_id(profile: str) -> str:
    if profile == DESCRIPTIVE_CAPABILITY_PROFILE:
        return BATTERY_PARTITION_ID
    if profile == BENIGN_TOOL_HARDENING_UTILITY_PROFILE:
        return BENIGN_TOOL_HARDENING_UTILITY_PROFILE
    raise ValueError(f"unknown benign capability battery profile: {profile!r}")


def _profile_mode_names(profile: str) -> tuple[str, ...]:
    if profile == DESCRIPTIVE_CAPABILITY_PROFILE:
        return tuple(MainConfig.MODE_NAMES)
    if profile == BENIGN_TOOL_HARDENING_UTILITY_PROFILE:
        return ("TOOL",)
    raise ValueError(f"unknown benign capability battery profile: {profile!r}")


def _profile_intervention_conditions(profile: str) -> tuple[str, ...]:
    if profile == DESCRIPTIVE_CAPABILITY_PROFILE:
        return (BATTERY_INTERVENTION,)
    if profile == BENIGN_TOOL_HARDENING_UTILITY_PROFILE:
        return (BASELINE_INTERVENTION, HARDENED_INTERVENTION)
    raise ValueError(f"unknown benign capability battery profile: {profile!r}")


def batches_dir_for_profile(profile: str) -> Path:
    if profile == DESCRIPTIVE_CAPABILITY_PROFILE:
        return BatteryConfig.BATCHES_DIR
    if profile == BENIGN_TOOL_HARDENING_UTILITY_PROFILE:
        return BatteryConfig.HARDENING_UTILITY_BATCHES_DIR
    raise ValueError(f"unknown benign capability battery profile: {profile!r}")


def _system_prompt_for_battery_condition(intervention_condition: str) -> str:
    base_prompt = MainConfig.PERSONAS[BATTERY_PERSONA]
    if intervention_condition == BATTERY_INTERVENTION:
        return base_prompt
    return _system_prompt_for_condition(base_prompt, intervention_condition)


def _entry_with_main_run_route_lock(subject_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    locked_route = MAIN_RUN_ROUTE_LOCK.get(subject_id)
    if locked_route is None:
        raise ValueError(f"no main-run route lock for subject {subject_id!r}")
    locked = copy.deepcopy(entry)
    locked["route_override"] = locked_route
    return locked


class _GenerationRouteAuth:
    """Let request-row generation use locked native routes without real keys.

    The main runner's select_route() intentionally requires native auth_envs so
    live dispatch cannot silently fall back. Benign batch generation only needs
    the request shape; no request is sent here and auth material is not stored
    in rows. Submission methods below do not use this bypass.
    """

    def __init__(self, entry: dict[str, Any]) -> None:
        self.auth_env = ""
        self.previous: str | None = None
        self.injected = False
        if entry.get("route_override") == "native":
            native = entry.get("native") or {}
            self.auth_env = str(native.get("auth_env") or "")

    def __enter__(self) -> None:
        if not self.auth_env or os.environ.get(self.auth_env):
            return
        self.injected = True
        self.previous = os.environ.get(self.auth_env)
        os.environ[self.auth_env] = "__capability_battery_generation_only__"

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.injected or not self.auth_env:
            return
        if self.previous is None:
            os.environ.pop(self.auth_env, None)
        else:
            os.environ[self.auth_env] = self.previous


class CapabilityBatteryRunner:
    def __init__(self) -> None:
        self.main_runner = ExperimentRunner()
        self.schemas = self.main_runner.schemas
        self.manifest = self.main_runner.manifest
        self.irrelevant_pool = self.main_runner.irrelevant_pool
        self.irrelevant_pool_sha256 = self.main_runner.irrelevant_pool_sha256

    def close(self) -> None:
        self.main_runner.close()

    def _row_tools_for_mode(
        self,
        prompt: dict[str, Any],
        mode_name: str,
    ) -> tuple[list[dict[str, Any]] | None, str, str]:
        domain = str(prompt["domain"]).lower()
        benign_tools = benign_domain_tools(
            schemas=self.schemas,
            manifest=self.manifest,
            domain=domain,
        )
        target = str(prompt["benign_target_tool"])
        if target not in {tool["name"] for tool in benign_tools}:
            raise ValueError(f"{prompt['benign_id']} target {target!r} is not benign")

        if mode_name == "NOTOOLS":
            return None, _hash_tool_subset([_request_tool_schema(t) for t in benign_tools]), EMPTY_SHA256

        if mode_name == "TOOL_IRRELEVANT":
            item = {
                "prompt_id": prompt["benign_id"],
                "domain": domain,
                "benign_twin": target,
            }
            irrelevant_tools = [
                _request_tool_schema(tool)
                for tool in select_irrelevant_tool_subset(item, self.irrelevant_pool, count=6)
            ]
            return irrelevant_tools, _hash_tool_subset(irrelevant_tools), self.irrelevant_pool_sha256

        request_tools = [_request_tool_schema(tool) for tool in benign_tools]
        return request_tools, _hash_tool_subset(request_tools), EMPTY_SHA256

    def build_rows_for_subject(
        self,
        subject_id: str,
        entry: dict[str, Any],
        prompts: list[dict[str, Any]],
        *,
        profile: str = DESCRIPTIVE_CAPABILITY_PROFILE,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        battery_id = _profile_battery_id(profile)
        mode_names = _profile_mode_names(profile)
        intervention_conditions = _profile_intervention_conditions(profile)
        locked_entry = _entry_with_main_run_route_lock(subject_id, entry)

        for prompt_index, prompt in enumerate(prompts):
            prompt_text = str(prompt["prompt_text"])
            prompt_sha256 = _sha256_text(prompt_text)
            for intervention_index, intervention_condition in enumerate(intervention_conditions):
                system_prompt = _system_prompt_for_battery_condition(intervention_condition)
                for order_index, mode_name in enumerate(mode_names):
                    row_tools, schema_sha256, irrelevant_hash = self._row_tools_for_mode(
                        prompt,
                        mode_name,
                    )
                    tool_choice = MainConfig.MODES[mode_name] if mode_name != "NOTOOLS" else None
                    custom_id = make_custom_id(
                        prompt_id=str(prompt["benign_id"]),
                        subject_id=subject_id,
                        mode_name=mode_name,
                        persona_name=BATTERY_PERSONA,
                        trial_id=BATTERY_TRIAL_ID,
                        corpus_partition_id=battery_id,
                        intervention_condition=intervention_condition,
                    )
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text},
                    ]
                    with _GenerationRouteAuth(locked_entry):
                        row = self.main_runner._build_request_row(
                            custom_id,
                            subject_id,
                            locked_entry,
                            messages,
                            row_tools,
                            tool_choice,
                            prompt_sha256,
                            schema_sha256,
                            irrelevant_hash,
                            EMPTY_SHA256,
                        )
                    row.update(
                        {
                            "battery_id": battery_id,
                            "battery_profile": profile,
                            "corpus_partition_id": battery_id,
                            "intervention_condition": intervention_condition,
                            "dispatch_route": row.get("route"),
                            "main_run_route_lock": MAIN_RUN_ROUTE_LOCK[subject_id],
                            "main_run_route_lock_source": MAIN_RUN_ROUTE_LOCK_SOURCE,
                            "main_run_provenance_lock_source": MAIN_RUN_ROUTE_LOCK_SOURCE,
                            "mode": mode_name,
                            "subject_id": subject_id,
                            "persona": BATTERY_PERSONA,
                            "trial_id": BATTERY_TRIAL_ID,
                            "prompt_id": str(prompt["benign_id"]),
                            "benign_id": str(prompt["benign_id"]),
                            "domain": str(prompt["domain"]).lower(),
                            "domain_original": prompt.get("domain_original"),
                            "manuscript_domain": prompt.get("manuscript_domain"),
                            "benign_target_tool": str(prompt["benign_target_tool"]),
                            "canonical_args": prompt["canonical_args"],
                            "benign_subtype": prompt.get("benign_subtype"),
                            "risk_boundary": prompt.get("risk_boundary"),
                            "expected_mode_behavior": prompt.get("expected_mode_behavior"),
                            "grading_notes": prompt.get("grading_notes"),
                            "order_index": order_index,
                            "intervention_order_index": intervention_index,
                            "randomized_order_seed": MainConfig.RANDOMIZED_ORDER_SEED,
                            "dispatch_block_sequence_index": prompt_index,
                            "dispatch_miniblock_key": (
                                f"{subject_id}:{prompt['benign_id']}:t{BATTERY_TRIAL_ID}:"
                                f"{intervention_condition}"
                            ),
                            "dispatch_block_id": (
                                f"{subject_id}:b{prompt_index:05d}:{prompt['benign_id']}:"
                                f"t{BATTERY_TRIAL_ID}:{intervention_condition}"
                            ),
                            "injected_tool_subset": (
                                [tool["name"] for tool in row_tools]
                                if row_tools is not None
                                else None
                            ),
                        }
                    )
                    if intervention_condition == HARDENED_INTERVENTION:
                        row["hardened_intervention_prompt_sha256"] = (
                            MainConfig.HARDENED_INTERVENTION_SHA256
                        )
                    row.update(copy.deepcopy(MAIN_RUN_PROVENANCE_LOCK[subject_id]))
                    rows.append(row)
        return rows

    def generate_batches(
        self,
        *,
        prompts_path: Path | None = None,
        output_dir: Path | None = None,
        subject_ids: set[str] | None = None,
        verify_hash: bool = True,
        profile: str = DESCRIPTIVE_CAPABILITY_PROFILE,
    ) -> dict[str, Any]:
        if profile not in VALID_PROFILES:
            raise ValueError(f"unknown benign capability battery profile: {profile!r}")
        prompts_path = prompts_path or BatteryConfig.PROMPTS_PATH
        output_dir = output_dir or batches_dir_for_profile(profile)
        if verify_hash:
            prompt_file_sha256 = verify_prompt_hash(prompts_path, BatteryConfig.PROMPTS_SHA256_PATH)
        else:
            prompt_file_sha256 = _sha256_file(prompts_path)

        prompts = load_battery_prompts(prompts_path)
        mode_names = _profile_mode_names(profile)
        intervention_conditions = _profile_intervention_conditions(profile)
        battery_id = _profile_battery_id(profile)
        selected_subjects = {
            sid: entry
            for sid, entry in MainConfig.SUBJECTS.items()
            if subject_ids is None or sid in subject_ids
        }
        if subject_ids is not None:
            missing = sorted(subject_ids - set(selected_subjects))
            if missing:
                raise ValueError(f"unknown subject id(s): {missing}")

        output_dir.mkdir(parents=True, exist_ok=True)
        per_subject_rows: dict[str, int] = {}
        route_counts: dict[str, int] = {}
        dispatch_route_counts: dict[str, int] = {}
        total_rows = 0

        for subject_id, entry in selected_subjects.items():
            rows = self.build_rows_for_subject(subject_id, entry, prompts, profile=profile)
            batch_path = output_dir / f"input_{subject_id}.jsonl"
            with batch_path.open("w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(_json_dumps(row) + "\n")
            locked_entry = _entry_with_main_run_route_lock(subject_id, entry)
            with _GenerationRouteAuth(locked_entry):
                dispatch_route, _route_cfg = select_route(locked_entry)
            canonical_route = str(MAIN_RUN_PROVENANCE_LOCK[subject_id]["route"])
            dispatch_route_counts[dispatch_route] = dispatch_route_counts.get(dispatch_route, 0) + 1
            route_counts[canonical_route] = route_counts.get(canonical_route, 0) + 1
            per_subject_rows[subject_id] = len(rows)
            total_rows += len(rows)
            logger.info("wrote %s rows to %s", len(rows), batch_path)

        summary = {
            "battery_id": battery_id,
            "battery_profile": profile,
            "prompt_file_sha256": prompt_file_sha256,
            "prompt_count": len(prompts),
            "subject_count": len(selected_subjects),
            "mode_names": list(mode_names),
            "persona": BATTERY_PERSONA,
            "trial_id": BATTERY_TRIAL_ID,
            "intervention_conditions": list(intervention_conditions),
            "intervention_condition": (
                intervention_conditions[0] if len(intervention_conditions) == 1 else None
            ),
            "total_rows": total_rows,
            "per_subject_rows": per_subject_rows,
            "route_counts": route_counts,
            "dispatch_route_counts": dispatch_route_counts,
            "main_run_route_lock": MAIN_RUN_ROUTE_LOCK,
            "main_run_provenance_lock": MAIN_RUN_PROVENANCE_LOCK,
            "main_run_route_lock_source": MAIN_RUN_ROUTE_LOCK_SOURCE,
            "output_dir": str(output_dir),
        }
        (output_dir / "capability_battery_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return summary

    def submit_native_batches(
        self,
        *,
        output_dir: Path | None = None,
        subject_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        output_dir = output_dir or BatteryConfig.BATCHES_DIR
        submitter = NativeBatchSubmitter(output_dir=output_dir)
        manifests: list[dict[str, Any]] = []
        selected = subject_ids or set(MainConfig.SUBJECTS)
        for subject_id, entry in MainConfig.SUBJECTS.items():
            if subject_id not in selected:
                continue
            entry = _entry_with_main_run_route_lock(subject_id, entry)
            batch_path = output_dir / f"input_{subject_id}.jsonl"
            if not batch_path.exists():
                raise FileNotFoundError(f"missing generated batch file: {batch_path}")
            route, route_cfg = select_route(entry)
            if route == "local":
                logger.info("local subject %s: generated rows are submitted via local runner", subject_id)
                continue
            if route != "native" or not route_cfg.get("supports_batch"):
                logger.info("subject %s resolved to %s without native batch support", subject_id, route)
                continue
            manifest = submitter.submit_subject_file(subject_id, entry, batch_path)
            manifest_path = output_dir / f"submit_manifest_{subject_id}.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            manifests.append(manifest)
        return manifests

    def submit_batches(
        self,
        *,
        output_dir: Path | None = None,
        subject_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Submit generated HTTP-route batches and leave local jobs untouched."""
        output_dir = output_dir or BatteryConfig.BATCHES_DIR
        submitter = NativeBatchSubmitter(output_dir=output_dir)
        manifests: list[dict[str, Any]] = []
        selected = subject_ids or set(MainConfig.SUBJECTS)
        for subject_id, entry in MainConfig.SUBJECTS.items():
            if subject_id not in selected:
                continue
            entry = _entry_with_main_run_route_lock(subject_id, entry)
            batch_path = output_dir / f"input_{subject_id}.jsonl"
            if not batch_path.exists():
                raise FileNotFoundError(f"missing generated batch file: {batch_path}")
            route, route_cfg = select_route(entry)
            if route == "local":
                logger.info("local subject %s: generated rows are submitted via local runner", subject_id)
                continue
            if route == "native" and route_cfg.get("supports_batch"):
                manifest = submitter.submit_subject_file(subject_id, entry, batch_path)
            elif route == "openrouter":
                with batch_path.open("r", encoding="utf-8") as fh:
                    rows = [json.loads(line) for line in fh if line.strip()]
                manifest = submitter._submit_openrouter_rows(subject_id, entry, rows)
            else:
                logger.info(
                    "subject %s resolved to %s without supported HTTP batch path",
                    subject_id,
                    route,
                )
                continue
            manifest_path = output_dir / f"submit_manifest_{subject_id}.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            manifests.append(manifest)
        return manifests


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=BatteryConfig.PROMPTS_PATH)
    parser.add_argument("--batches-dir", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(VALID_PROFILES),
        default=DESCRIPTIVE_CAPABILITY_PROFILE,
        help=(
            "Batch profile to generate. The default reproduces the 1,200-row "
            "descriptive battery; benign_tool_hardening_utility generates the "
            "600-row baseline/hardened TOOL rerun."
        ),
    )
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--submit-batches", action="store_true")
    parser.add_argument("--submit-native-batches", action="store_true")
    parser.add_argument(
        "--generate-batches",
        action="store_true",
        help="Generate batch JSONL files. This is also the default action.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subjects = set(args.subject) if args.subject else None
    batches_dir = args.batches_dir or batches_dir_for_profile(args.profile)
    runner = CapabilityBatteryRunner()
    try:
        if args.generate_batches or not (args.submit_native_batches or args.submit_batches):
            summary = runner.generate_batches(
                prompts_path=args.prompts,
                output_dir=batches_dir,
                subject_ids=subjects,
                verify_hash=not args.no_verify_hash,
                profile=args.profile,
            )
            print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
        if args.submit_batches:
            manifests = runner.submit_batches(
                output_dir=batches_dir,
                subject_ids=subjects,
            )
            print(json.dumps({"submitted_http_batches": len(manifests)}, sort_keys=True))
        if args.submit_native_batches:
            manifests = runner.submit_native_batches(
                output_dir=batches_dir,
                subject_ids=subjects,
            )
            print(json.dumps({"submitted_native_batches": len(manifests)}, sort_keys=True))
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
