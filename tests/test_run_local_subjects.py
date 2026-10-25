"""Tests for the local-route dispatcher (paired local BF16 + INT4 audit)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from scripts.run_local_subjects import (
    MLXLocalClient,
    RuntimeProvenanceError,
    dispatch_file,
    dispatch_row,
    local_response_output_path,
    select_backend,
)
from scripts.run_synthetic_prompts import precision_variants_for_subject


@pytest.fixture(autouse=True)
def _local_runtime_env(monkeypatch):
    monkeypatch.setenv("CAGAP_RUNTIME_HARDWARE", "test-hardware")
    monkeypatch.setenv("CAGAP_RUNTIME_DRIVER", "test-driver")
    monkeypatch.setenv("CAGAP_RUNTIME_CUDA", "test-cuda")
    monkeypatch.setenv("CAGAP_RUNTIME_COMPUTE_CAP", "test-compute")
    monkeypatch.setenv("CAGAP_RUNTIME_TP_SIZE", "1")


# --------------------------------------------------------------------------
# precision_variants_for_subject
# --------------------------------------------------------------------------


def _api_subject():
    return {
        "tier": "hosted_api",
        "native": {"precision_primary": "provider_unspecified"},
    }


def _local_subject(precision_primary="bf16", sensitivity="int4_mlx", source="mlx-community/test-4bit"):
    return {
        "tier": "local_self_hosted",
        "local_config": {
            "hf_repo": "test/base-bf16",
            "precision_primary": precision_primary,
            "precision_primary_location": "m5_macbook_128gb",
            "precision_primary_runtime": "mlx_lm_bf16",
            "precision_sensitivity": sensitivity,
            "precision_sensitivity_source": source,
            "precision_sensitivity_location": "m5_macbook_128gb",
        },
    }


def test_api_subject_emits_one_primary_variant():
    variants = precision_variants_for_subject(_api_subject())
    assert len(variants) == 1
    assert variants[0]["role"] == "primary"
    assert variants[0]["calibration_only"] is False


def test_local_subject_emits_two_variants_primary_and_audit():
    variants = precision_variants_for_subject(_local_subject())
    assert len(variants) == 2
    assert variants[0]["role"] == "primary"
    assert variants[0]["calibration_only"] is False
    assert variants[0]["model_path"] == "test/base-bf16"
    assert variants[1]["role"] == "quantization_audit"
    assert variants[1]["calibration_only"] is True
    assert variants[1]["model_path"] == "mlx-community/test-4bit"
    assert variants[1]["value"] == "int4_mlx"


def test_local_subject_without_sensitivity_emits_only_primary():
    entry = _local_subject(sensitivity=None)
    entry["local_config"]["precision_sensitivity"] = None
    variants = precision_variants_for_subject(entry)
    assert len(variants) == 1
    assert variants[0]["role"] == "primary"


# --------------------------------------------------------------------------
# select_backend
# --------------------------------------------------------------------------


def _build_row(role="primary", location="m5_macbook_128gb", precision="bf16",
               model="test/model"):
    return {
        "custom_id": "cid",
        "subject_id": "sid",
        "route": "local",
        "serving_precision_role": role,
        "serving_precision": precision,
        "served_model_path": model,
        "served_model_location": location,
        "served_model_runtime": "mlx_lm_bf16" if precision == "bf16" else "mlx_lm_int4",
        "body": {"model": model, "messages": []},
    }


def test_select_backend_primary_local():
    backend = select_backend(_build_row())
    assert backend.role == "primary"
    assert backend.location == "m5_macbook_128gb"
    assert backend.target_dtype == "bf16"


def test_select_backend_audit_local():
    row = _build_row(role="quantization_audit", precision="int4_mlx",
                     model="mlx-community/test-4bit")
    backend = select_backend(row)
    assert backend.role == "quantization_audit"
    assert backend.target_dtype == "int4_mlx"
    assert backend.served_model_path == "mlx-community/test-4bit"


def test_select_backend_external_gpu():
    row = _build_row(location="external_gpu", precision="fp16")
    backend = select_backend(row)
    assert backend.location == "external_gpu"
    assert backend.target_dtype == "fp16"


def test_select_backend_audit_fallback_uses_sensitivity_keys_not_primary():
    """Quantization-audit fallback must read the sensitivity route."""
    row = {
        "custom_id": "cid",
        "subject_id": "sid",
        "route": "local",
        "serving_precision_role": "quantization_audit",
        "served_model_path": "hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        "served_model_location": None,  # force fallback
        "served_model_runtime": None,   # force fallback
        "local_inference": {
            "precision_primary_location": "external_gpu",
            "precision_primary_runtime": "vllm_16bit",
            "precision_sensitivity_location": "external_gpu",
            "precision_sensitivity_runtime": "vllm_awq",
        },
        "body": {"model": "hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"},
    }
    backend = select_backend(row)
    assert backend.role == "quantization_audit"
    assert backend.location == "external_gpu"
    assert backend.runtime == "vllm_awq"  # NOT vllm_16bit


def test_select_backend_primary_fallback_uses_primary_keys():
    """Mirror of the above for primary rows — the fallback must read
    precision_primary_* fields."""
    row = {
        "custom_id": "cid",
        "subject_id": "sid",
        "route": "local",
        "serving_precision_role": "primary",
        "served_model_path": "meta-llama/Llama-3.3-70B-Instruct",
        "served_model_location": None,
        "served_model_runtime": None,
        "local_inference": {
            "precision_primary_location": "external_gpu",
            "precision_primary_runtime": "vllm_16bit",
            "precision_sensitivity_location": "external_gpu",
            "precision_sensitivity_runtime": "vllm_awq",
        },
        "body": {"model": "meta-llama/Llama-3.3-70B-Instruct"},
    }
    backend = select_backend(row)
    assert backend.role == "primary"
    assert backend.runtime == "vllm_16bit"


def test_select_backend_missing_location_raises():
    row = _build_row()
    row["served_model_location"] = None
    row.pop("local_inference", None)
    with pytest.raises(ValueError, match="served_model_location"):
        select_backend(row)


def test_select_backend_missing_model_raises():
    row = _build_row()
    row["served_model_path"] = None
    row["body"]["model"] = None
    with pytest.raises(ValueError, match="served_model_path"):
        select_backend(row)


# --------------------------------------------------------------------------
# dispatch_row schema conformance
# --------------------------------------------------------------------------


class _MockMLX(MLXLocalClient):
    def __init__(self):
        super().__init__(base_url="http://mock:8080", max_retries=0)
        self._server_metadata = {"data": [{"id": "mock", "context_length": 32768}]}

    def chat(self, body, *, timeout=600):
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }


def _full_row(role="quantization_audit", precision="int4_mlx",
              location="m5_macbook_128gb", model="mlx-community/test-4bit"):
    return {
        "custom_id": "cid1",
        "subject_id": "gemma-4-31b-it",
        "route": "local",
        "pinned_provider": "local",
        "mode": "TOOL",
        "persona": "PERSONA_NEUTRAL",
        "prompt_id": "GE1-v001",
        "trial_id": 0,
        "intervention_condition": "baseline",
        "corpus_partition_id": "main",
        "domain": "gridsim",
        "order_index": 1,
        "randomized_order_seed": "seed",
        "serving_precision_role": role,
        "serving_precision": precision,
        "serving_precision_value": precision,
        "served_model_path": model,
        "served_model_location": location,
        "served_model_runtime": "mlx_lm_int4" if precision == "int4_mlx" else "mlx_lm_bf16",
        "body": {"model": model, "messages": [{"role": "user", "content": "hi"}]},
        "hashes": {
            "prompt_sha256": "x", "schema_sha256": "y",
            "irrelevant_pool_sha256": "", "intervention_prompt_sha256": "",
        },
        "prompt_sha256": "x",
        "schema_sha256": "y",
        "irrelevant_pool_sha256": "",
        "intervention_prompt_sha256": "",
        "disabled_arm_schema_visibility": "schema_visible",
        "route_support_status": "supported",
        "schema_visibility_evidence_id": "outputs/schema_visibility/classification.json",
        "schema_visibility_evidence_sha256": "f" * 64,
        "freeze_date": "2026-04-24",
        "decoding_applied": {},
        "injected_tool_subset": None,
    }


def test_dispatch_row_emits_run_record_required_fields():
    row = _full_row()
    result = dispatch_row(
        row, vllm_client=None, mlx_client=_MockMLX(), skip_model_match=True
    )
    required = [
        "run_id", "timestamp", "model", "provider", "subject_id", "route",
        "mode", "order_index", "persona", "trial_id", "intervention_condition",
        "corpus_partition_id", "serving_precision_role", "prompt_id",
        "custom_id", "domain", "latency_sec", "usage", "request_prompt",
        "raw_response", "schema_version", "hashes",
    ]
    missing = [k for k in required if k not in result]
    assert missing == [], f"missing required run_record fields: {missing}"
    assert result["disabled_arm_schema_visibility"] == "schema_visible"
    assert result["route_support_status"] == "supported"
    assert result["schema_visibility_evidence_id"] == "outputs/schema_visibility/classification.json"
    assert result["schema_visibility_evidence_sha256"] == "f" * 64


def test_dispatch_row_uses_stable_run_id_and_fixed_timestamp(monkeypatch):
    monkeypatch.setenv("CAGAP_FIXED_TIMESTAMP", "2026-05-03T00:00:00Z")
    row = _full_row()

    first = dispatch_row(
        row, vllm_client=None, mlx_client=_MockMLX(), skip_model_match=True
    )
    second = dispatch_row(
        row, vllm_client=None, mlx_client=_MockMLX(), skip_model_match=True
    )

    expected_run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, row["custom_id"]))
    assert first["run_id"] == expected_run_id
    assert second["run_id"] == expected_run_id
    assert first["timestamp"] == "2026-05-03T00:00:00Z"


def test_dispatch_row_preserves_serving_precision_role():
    audit = dispatch_row(
        _full_row(role="quantization_audit"),
        vllm_client=None,
        mlx_client=_MockMLX(),
        skip_model_match=True,
    )
    assert audit["serving_precision_role"] == "quantization_audit"
    assert audit["serving_precision"] == "int4_mlx"
    assert audit["serving_precision_value"] == "int4_mlx"
    primary = dispatch_row(
        _full_row(role="primary", precision="bf16",
                  model="test/base-bf16", location="m5_macbook_128gb"),
        vllm_client=None, mlx_client=_MockMLX(), skip_model_match=True,
    )
    assert primary["serving_precision_role"] == "primary"
    assert primary["serving_precision"] == "bf16"
    assert primary["serving_precision_value"] == "bf16"


def test_dispatch_row_records_error_when_backend_unavailable():
    row = _full_row(role="primary", location="external_gpu", precision="fp16",
                    model="meta-llama/Llama-3.3-70B-Instruct")
    result = dispatch_row(row, vllm_client=None, mlx_client=None)
    assert "error" in result
    assert "backend_error" in result["error"]
    # Schema-required identity fields still present even on error rows.
    assert result["custom_id"] == "cid1"
    assert result["serving_precision_role"] == "primary"


# --------------------------------------------------------------------------
# dispatch_file (jsonl input/output)
# --------------------------------------------------------------------------


def test_dispatch_file_skips_non_local_rows(tmp_path: Path):
    src = tmp_path / "input_test.jsonl"
    rows = [
        {"custom_id": "c1", "subject_id": "s", "route": "native", "mode": "TOOL",
         "body": {"model": "x", "messages": []}},
        _full_row(),
    ]
    with src.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    dst = tmp_path / "out.jsonl"
    counts = dispatch_file(
        src, dst, vllm_client=None, mlx_client=_MockMLX(), skip_model_match=True,
    )
    assert counts["total"] == 2
    assert counts["skipped_non_local"] == 1
    assert counts["dispatched"] == 1
    assert dst.exists()
    written = [json.loads(l) for l in dst.read_text().splitlines() if l.strip()]
    assert len(written) == 1
    assert written[0]["serving_precision_role"] == "quantization_audit"


def test_dispatch_file_rejects_duplicate_input_json_keys(tmp_path: Path):
    src = tmp_path / "input_test.jsonl"
    src.write_text(
        '{"custom_id": "c1", "custom_id": "c2", "route": "local"}\n',
        encoding="utf-8",
    )
    dst = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        dispatch_file(
            src, dst, vllm_client=None, mlx_client=_MockMLX(), skip_model_match=True,
        )


def test_dispatch_file_resumes_existing_custom_ids(tmp_path: Path):
    src = tmp_path / "input_test.jsonl"
    first = _full_row()
    second = _full_row()
    second["custom_id"] = "cid2"
    with src.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(first) + "\n")
        fh.write(json.dumps(second) + "\n")

    dst = tmp_path / "out.jsonl"
    existing = {"custom_id": "cid1", "route": "local", "existing": True}
    dst.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    counts = dispatch_file(
        src, dst, vllm_client=None, mlx_client=_MockMLX(), skip_model_match=True,
    )

    assert counts["total"] == 2
    assert counts["skipped_existing"] == 1
    assert counts["dispatched"] == 1
    written = [json.loads(l) for l in dst.read_text().splitlines() if l.strip()]
    assert written[0] == existing
    assert written[1]["custom_id"] == "cid2"


def test_dispatch_file_filters_serving_precision_role_for_parallel_lanes(tmp_path: Path):
    src = tmp_path / "input_gemma-4-31b-it.jsonl"
    primary = _full_row(
        role="primary",
        precision="bf16",
        model="test/base-bf16",
        location="m5_macbook_128gb",
    )
    primary["custom_id"] = "cid_primary"
    audit = _full_row()
    audit["custom_id"] = "cid_audit"
    with src.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(primary) + "\n")
        fh.write(json.dumps(audit) + "\n")

    dst = tmp_path / "out.jsonl"
    counts = dispatch_file(
        src,
        dst,
        vllm_client=None,
        mlx_client=_MockMLX(),
        skip_model_match=True,
        serving_precision_roles={"quantization_audit"},
    )

    assert counts["total"] == 2
    assert counts["skipped_filter"] == 1
    assert counts["dispatched"] == 1
    written = [json.loads(l) for l in dst.read_text().splitlines() if l.strip()]
    assert [row["custom_id"] for row in written] == ["cid_audit"]
    assert written[0]["serving_precision_role"] == "quantization_audit"


def test_local_response_output_path_uses_lane_suffix_for_filtered_runs(tmp_path: Path):
    default_path = local_response_output_path(tmp_path, "gemma-4-31b-it")
    filtered_path = local_response_output_path(
        tmp_path,
        "gemma-4-31b-it",
        serving_precision_roles={"quantization_audit"},
        serving_precisions={"int4_mlx"},
        served_model_locations={"m5_macbook_128gb"},
    )

    assert default_path == tmp_path / "local_responses_gemma-4-31b-it.jsonl"
    assert filtered_path != default_path
    assert filtered_path.name == (
        "local_responses_gemma-4-31b-it"
        "__location-m5_macbook_128gb"
        "__precision-int4_mlx"
        "__role-quantization_audit.jsonl"
    )


def test_dispatch_file_rejects_duplicate_existing_output_keys(tmp_path: Path):
    src = tmp_path / "input_test.jsonl"
    src.write_text(json.dumps(_full_row()) + "\n", encoding="utf-8")
    dst = tmp_path / "out.jsonl"
    dst.write_text('{"custom_id": "cid1", "custom_id": "cid2"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        dispatch_file(
            src, dst, vllm_client=None, mlx_client=_MockMLX(), skip_model_match=True,
        )


# --------------------------------------------------------------------------
# Runtime stamping + retry behavior
# --------------------------------------------------------------------------


def test_dispatch_row_emits_runtime_stamp():
    class _StampingMLX(MLXLocalClient):
        def __init__(self):
            super().__init__(base_url="http://test:8080", max_retries=0)
            self._server_metadata = {"data": [{"id": "m", "context_length": 32768}]}

        def chat(self, body, *, timeout=600):
            return {
                "model": "test-model",
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                "system_fingerprint": "fp_x",
            }

    result = dispatch_row(
        _full_row(), vllm_client=None, mlx_client=_StampingMLX(), skip_model_match=True,
    )
    stamp = result.get("runtime_stamp")
    assert stamp is not None
    assert stamp["client_label"] == "local_mlx"
    assert stamp["endpoint"] == "http://test:8080"
    assert stamp["returned_model"] == "test-model"
    assert stamp["system_fingerprint"] == "fp_x"
    assert stamp["finish_reason"] == "stop"
    assert "data" in stamp["server_metadata"]
    assert stamp["hardware_name"] == "test-hardware"
    assert stamp["driver_version"] == "test-driver"
    assert stamp["cuda_version"] == "test-cuda"
    assert stamp["cuda_version_runtime"] == "test-cuda"
    assert stamp["compute_capability"] == "test-compute"
    assert stamp["tensor_parallel_size"] == 1


def test_dispatch_row_requires_checkpoint_manifest_when_enabled(tmp_path: Path):
    with pytest.raises(Exception, match="checkpoint manifest"):
        dispatch_row(
            _full_row(),
            vllm_client=None,
            mlx_client=_MockMLX(),
            skip_model_match=True,
            checkpoint_manifest_dir=tmp_path,
            require_checkpoint_manifest=True,
        )


def test_dispatch_row_stamps_checkpoint_manifest_when_enabled(tmp_path: Path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "gemma-4-31b-it__quantization_audit__int4_mlx.json"
    manifest_path.write_text(
        json.dumps(
            {
                "subject_id": "gemma-4-31b-it",
                "serving_precision_role": "quantization_audit",
                "serving_precision_value": "int4_mlx",
                "served_model_path": "mlx-community/test-4bit",
                "hf_commit_sha": "145dc2508c480a64b47242f160d286cff94a2343",
                "files": [{"path": "tokenizer.json", "sha256": "a" * 64, "size_bytes": 5}],
            }
        ),
        encoding="utf-8",
    )

    result = dispatch_row(
        _full_row(),
        vllm_client=None,
        mlx_client=_MockMLX(),
        skip_model_match=True,
        checkpoint_manifest_dir=manifest_dir,
        require_checkpoint_manifest=True,
    )

    assert result["checkpoint_manifest_path"] == str(manifest_path)
    assert result["checkpoint_manifest_sha256"]
    assert result["runtime_stamp"]["checkpoint_manifest_sha256"] == result["checkpoint_manifest_sha256"]


def test_dispatch_row_rejects_missing_runtime_provenance(monkeypatch):
    for key in (
        "CAGAP_RUNTIME_HARDWARE",
        "CAGAP_RUNTIME_DRIVER",
        "CAGAP_RUNTIME_CUDA",
        "CAGAP_RUNTIME_COMPUTE_CAP",
        "CAGAP_RUNTIME_TP_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeProvenanceError, match="runtime provenance"):
        dispatch_row(
            _full_row(),
            vllm_client=None,
            mlx_client=_MockMLX(),
            skip_model_match=True,
        )


def test_dispatch_file_rejects_missing_runtime_provenance(monkeypatch, tmp_path: Path):
    for key in (
        "CAGAP_RUNTIME_HARDWARE",
        "CAGAP_RUNTIME_DRIVER",
        "CAGAP_RUNTIME_CUDA",
        "CAGAP_RUNTIME_COMPUTE_CAP",
        "CAGAP_RUNTIME_TP_SIZE",
    ):
        monkeypatch.delenv(key, raising=False)
    src = tmp_path / "input_test.jsonl"
    src.write_text(json.dumps(_full_row()) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeProvenanceError, match="runtime provenance"):
        dispatch_file(
            src,
            tmp_path / "out.jsonl",
            vllm_client=None,
            mlx_client=_MockMLX(),
            skip_model_match=True,
        )


def test_post_chat_completions_retries_on_429(tmp_path: Path):
    """Verify exponential backoff + retry on retryable HTTP statuses."""
    from scripts import run_local_subjects as rls

    call_count = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        call_count["n"] += 1
        if call_count["n"] < 3:
            from urllib.error import HTTPError
            raise HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        # success on 3rd attempt
        class _Resp:
            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}],"usage":{}}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    real_urlopen = rls.urllib.request.urlopen
    real_sleep = rls.time.sleep
    try:
        rls.urllib.request.urlopen = fake_urlopen
        rls.time.sleep = lambda s: sleeps.append(s)
        result = rls._post_chat_completions(
            "http://t/v1/chat/completions",
            {"model": "x", "messages": []},
            headers={},
            timeout=10,
            max_retries=5,
            backoff_base=2.0,
        )
    finally:
        rls.urllib.request.urlopen = real_urlopen
        rls.time.sleep = real_sleep
    assert result["choices"][0]["message"]["content"] == "ok"
    assert call_count["n"] == 3
    # 2 retries → 2 sleeps with exponential delays 2^0, 2^1
    assert sleeps == [1.0, 2.0]


def test_post_chat_completions_rejects_duplicate_response_json_keys():
    from scripts import run_local_subjects as rls

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        class _Resp:
            def read(self):
                return b'{"id": "resp_a", "id": "resp_b", "choices": []}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    real_urlopen = rls.urllib.request.urlopen
    try:
        rls.urllib.request.urlopen = fake_urlopen
        with pytest.raises(RuntimeError, match="strict JSON"):
            rls._post_chat_completions(
                "http://t/v1/chat/completions",
                {"model": "x", "messages": []},
                headers={},
                timeout=10,
                max_retries=0,
            )
    finally:
        rls.urllib.request.urlopen = real_urlopen


# --------------------------------------------------------------------------
# Pre-flight model-match check
# --------------------------------------------------------------------------


class _MatchingMLX(MLXLocalClient):
    """Server reports a model id that matches the row's served_model_path."""

    def __init__(self, model_id: str = "mlx-community/test-4bit"):
        super().__init__(base_url="http://mock:8080", max_retries=0)
        self._server_metadata = {"data": [{"id": model_id, "context_length": 32768}]}

    def chat(self, body, *, timeout=600):
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class _MismatchedMLX(MLXLocalClient):
    """Server reports a *different* model id than the row asks for."""

    def __init__(self):
        super().__init__(base_url="http://mock:8080", max_retries=0)
        self._server_metadata = {"data": [{"id": "wrong/model"}]}

    def chat(self, body, *, timeout=600):  # pragma: no cover - should not be hit
        raise AssertionError("chat must NOT be called when models mismatch")


class _EmptyMetadataMLX(MLXLocalClient):
    """Server returns empty /v1/models payload (no loaded model)."""

    def __init__(self):
        super().__init__(base_url="http://mock:8080", max_retries=0)
        self._server_metadata = {"data": []}

    def chat(self, body, *, timeout=600):  # pragma: no cover - should not be hit
        raise AssertionError("chat must NOT be called when no model is loaded")


def test_dispatch_row_passes_when_loaded_model_matches():
    result = dispatch_row(
        _full_row(),
        vllm_client=None,
        mlx_client=_MatchingMLX("mlx-community/test-4bit"),
    )
    assert "error" not in result
    stamp = result["runtime_stamp"]
    assert stamp["loaded_model_match"] == "mlx-community/test-4bit"
    assert stamp["model_match_skipped"] is False


def test_dispatch_row_rejects_basename_only_match_for_bf16_int4_siblings():
    """review finding 2026-04-30: basename fallback was unsafe — a BF16
    reference and its INT4 sibling can share the same final path
    component (e.g., `/srv/models/bf16/Qwen3.6-35B-A3B` vs
    `/srv/models/int4/Qwen3.6-35B-A3B`). A basename rule would silently
    route the audit variant onto the reference server. The matcher now
    requires strict normalized-path equality, so this case is rejected."""
    result = dispatch_row(
        _full_row(model="/srv/models/int4/Qwen3.6-35B-A3B"),
        vllm_client=None,
        mlx_client=_MatchingMLX("/srv/models/bf16/Qwen3.6-35B-A3B"),
    )
    assert "error" in result
    assert result["error"].startswith("model_mismatch:")


def test_dispatch_row_records_error_on_model_mismatch():
    result = dispatch_row(
        _full_row(),
        vllm_client=None,
        mlx_client=_MismatchedMLX(),
    )
    assert "error" in result
    assert result["error"].startswith("model_mismatch:")


def test_dispatch_row_records_error_when_server_reports_no_model():
    result = dispatch_row(
        _full_row(),
        vllm_client=None,
        mlx_client=_EmptyMetadataMLX(),
    )
    assert "error" in result
    assert result["error"].startswith("model_mismatch:")


def test_dispatch_row_skip_model_match_bypasses_check():
    # Mismatched server, but skip_model_match=True must bypass the check
    # so chat() is reached. Use _MatchingMLX with a clearly-wrong id; the
    # check would normally fail, but skip=True short-circuits it.
    result = dispatch_row(
        _full_row(),
        vllm_client=None,
        mlx_client=_MatchingMLX("definitely/not/the/row/model"),
        skip_model_match=True,
    )
    assert "error" not in result
    assert result["runtime_stamp"]["model_match_skipped"] is True
    assert result["runtime_stamp"]["loaded_model_match"] is None


def test_post_chat_completions_does_not_retry_on_4xx_other():
    """Permanent client errors (e.g., 400 Bad Request) must NOT be retried."""
    from scripts import run_local_subjects as rls

    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        call_count["n"] += 1
        from urllib.error import HTTPError
        raise HTTPError(req.full_url, 400, "Bad Request", {}, None)

    real_urlopen = rls.urllib.request.urlopen
    try:
        rls.urllib.request.urlopen = fake_urlopen
        with pytest.raises(RuntimeError, match="HTTP 400"):
            rls._post_chat_completions(
                "http://t/v1/chat/completions",
                {"model": "x", "messages": []},
                headers={},
                timeout=10,
                max_retries=5,
                backoff_base=2.0,
            )
    finally:
        rls.urllib.request.urlopen = real_urlopen
    assert call_count["n"] == 1
