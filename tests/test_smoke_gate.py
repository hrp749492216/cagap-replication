import subprocess
import sys

from scripts.smoke_gate import evaluate_correctness_smoke_gate


def _record(mode: str, *, status: int = 200, tool_calls: bool = False) -> dict:
    return {
        "mode": mode,
        "http_status": status,
        "has_tool_calls": tool_calls,
    }


def test_slow_correct_smoke_passes_with_descriptive_latency_only():
    records = [
        *[_record("TOOL", tool_calls=True) for _ in range(3)],
        *[_record("DISABLED") for _ in range(3)],
        *[_record("NOTOOLS") for _ in range(3)],
        *[_record("TOOL_IRRELEVANT") for _ in range(3)],
    ]

    result = evaluate_correctness_smoke_gate(
        records,
        cells_total=12,
        mean_seconds_per_cell=12.95,
        baseline_s_per_cell=2.4,
    )

    assert result["verdict"] == "PASS"
    assert result["gate_evaluation"]["http_pass"] is True
    assert result["gate_evaluation"]["parser_pass"] is True
    assert result["gate_evaluation"]["latency_policy"] == "descriptive_only_not_gating"
    assert result["gate_evaluation"]["latency_ratio_vs_baseline"] == 5.4
    assert "latency_pass" not in result["gate_evaluation"]


def test_http_success_still_fails_when_tool_parser_does_not_populate_tool_calls():
    records = [
        *[_record("TOOL", tool_calls=False) for _ in range(3)],
        *[_record("DISABLED") for _ in range(3)],
        *[_record("NOTOOLS") for _ in range(3)],
        *[_record("TOOL_IRRELEVANT") for _ in range(3)],
    ]

    result = evaluate_correctness_smoke_gate(
        records,
        cells_total=12,
        mean_seconds_per_cell=1.0,
        baseline_s_per_cell=1.0,
    )

    assert result["verdict"] == "PASS_HTTP_FAIL_PARSER"
    assert result["gate_evaluation"]["http_pass"] is True
    assert result["gate_evaluation"]["parser_pass"] is False


def test_smoke_scripts_can_run_by_file_path_for_cloud_runners():
    for script in ["scripts/cloud_vllm_smoke.py", "scripts/m5_mlx_smoke.py"]:
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
