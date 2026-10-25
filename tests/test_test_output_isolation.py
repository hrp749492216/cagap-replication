"""Regression checks for fixture leakage into the study provenance store."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import raw_http_archive


def test_archive_fallback_stays_in_temporary_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("CAGAP_RAW_HTTP_ARCHIVE_DIR")
    monkeypatch.delenv("CAGAP_RAW_HTTP_ARCHIVE_DISABLED", raising=False)
    result = raw_http_archive.archive_response_body(
        raw_body=b"fixture", role="subject", model_id="test-subject",
        route="test", provider="fixture", custom_id="isolation-check",
        attempt_index=0, source="test_output_isolation",
    )
    assert result is not None
    assert Path(result["raw_http_body_path"]).is_relative_to(tmp_path)
    assert Path(result["raw_http_manifest_path"]).exists()


def test_writes_to_production_outputs_are_rejected():
    destination = Path(__file__).resolve().parents[1] / "outputs" / "test-leak.txt"
    with pytest.raises(RuntimeError, match="write production output"):
        destination.write_text("must not reach the filesystem")
    assert not destination.exists()


def test_runner_import_does_not_load_credentials():
    result = subprocess.run(
        [sys.executable, "-c", "from unittest.mock import patch; "
         "p = patch('scripts.cagap_env.load_cagap_env', "
         "side_effect=AssertionError('credential load during import')); "
         "p.start(); import scripts.run_synthetic_prompts"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
