"""Deterministic n=3 fixtures for trial-level ConditionalLogit tests.

The fixture simulates the primary 3:3 matched-cluster design under a known
conditional log-odds ratio for TOOL versus DISABLED. It is intentionally small
enough for unit tests while preserving the real study axes: subjects, prompts,
personas, modes, and trial replicates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


SUBJECT_IDS = tuple(f"synthetic_subject_{idx:02d}" for idx in range(9))
PERSONAS = ("PERSONA_NEUTRAL", "PERSONA_OPERATOR")
MODES = ("NOTOOLS", "DISABLED", "TOOL", "TOOL_IRRELEVANT")
TRIAL_IDS = (0, 1, 2)


@dataclass(frozen=True)
class SyntheticN3Fixture:
    frame: pd.DataFrame
    true_log_or: float
    true_or: float
    seed: int
    prompt_count: int
    expected_primary_clusters: int
    expected_informative_clusters: int


def _inv_logit(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _label(event: int) -> str:
    return "Comply" if int(event) else "Refuse"


def _base_row(
    *,
    prompt_id: str,
    subject_id: str,
    persona: str,
    mode: str,
    trial_id: int,
    event: int,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "case_id": prompt_id,
        "subject_id": subject_id,
        "subject_tier": "synthetic",
        "persona": persona,
        "domain": "SyntheticClogit",
        "partition_id": "main",
        "trial_id": int(trial_id),
        "intervention_condition": "baseline",
        "serving_precision_role": "primary",
        "mode": mode,
        "final_label": _label(event),
        "final_label_status": "resolved",
        "analysis_inclusion": True,
        "final_label_source": "synthetic_known_or_fixture",
        "input_tokens": 0,
        "output_tokens": 0,
    }


def _expected_informative_clusters(frame: pd.DataFrame) -> int:
    primary = frame[frame["mode"].isin(["DISABLED", "TOOL"])].copy()
    primary["cluster_id"] = (
        primary["prompt_id"].astype(str)
        + "__"
        + primary["subject_id"].astype(str)
        + "__"
        + primary["persona"].astype(str)
    )
    primary["event"] = (primary["final_label"] == "Comply").astype(int)
    informative = 0
    for _, group in primary.groupby("cluster_id"):
        arm_counts = group.groupby("mode")["event"].agg(["sum", "size"])
        if not {"TOOL", "DISABLED"}.issubset(arm_counts.index):
            continue
        total_events = int(arm_counts.loc["TOOL", "sum"]) + int(
            arm_counts.loc["DISABLED", "sum"]
        )
        total_trials = int(arm_counts.loc["TOOL", "size"]) + int(
            arm_counts.loc["DISABLED", "size"]
        )
        if 0 < total_events < total_trials:
            informative += 1
    return informative


def build_synthetic_n3_fixture(
    *,
    seed: int = 20260510,
    prompt_count: int = 36,
    true_or: float = 2.25,
) -> SyntheticN3Fixture:
    """Build a deterministic primary-design frame generated under true_or.

    For each prompt x subject x persona cluster, the disabled arm uses a
    cluster-specific baseline logit and the TOOL arm adds log(true_or). The
    NOTOOLS and TOOL_IRRELEVANT arms are present to preserve the full row shape
    but are not used by the primary estimator.
    """
    rng = np.random.default_rng(seed)
    true_log_or = math.log(float(true_or))
    rows: list[dict[str, Any]] = []
    for subject_id in SUBJECT_IDS:
        for prompt_idx in range(prompt_count):
            prompt_id = f"synthetic_prompt_{prompt_idx:03d}"
            for persona in PERSONAS:
                baseline_logit = float(rng.normal(loc=-1.0, scale=0.85))
                for trial_id in TRIAL_IDS:
                    for mode in MODES:
                        if mode == "TOOL":
                            eta = baseline_logit + true_log_or
                        elif mode == "TOOL_IRRELEVANT":
                            eta = baseline_logit + 0.10
                        elif mode == "NOTOOLS":
                            eta = baseline_logit - 0.10
                        else:
                            eta = baseline_logit
                        event = int(rng.binomial(1, _inv_logit(eta)))
                        rows.append(
                            _base_row(
                                prompt_id=prompt_id,
                                subject_id=subject_id,
                                persona=persona,
                                mode=mode,
                                trial_id=trial_id,
                                event=event,
                            )
                        )
    frame = pd.DataFrame(rows)
    return SyntheticN3Fixture(
        frame=frame,
        true_log_or=true_log_or,
        true_or=float(true_or),
        seed=seed,
        prompt_count=prompt_count,
        expected_primary_clusters=len(SUBJECT_IDS) * prompt_count * len(PERSONAS),
        expected_informative_clusters=_expected_informative_clusters(frame),
    )


def build_concordant_n3_clusters(
    *,
    start_prompt_index: int,
    cluster_count: int,
) -> pd.DataFrame:
    """Build 0/3-vs-0/3 and 3/3-vs-3/3 clusters for dropout tests."""
    rows: list[dict[str, Any]] = []
    for cluster_idx in range(cluster_count):
        event = 0 if cluster_idx % 2 == 0 else 1
        prompt_id = f"synthetic_concordant_{start_prompt_index + cluster_idx:05d}"
        subject_id = f"synthetic_concordant_subject_{cluster_idx:02d}"
        persona = "PERSONA_NEUTRAL"
        for trial_id in TRIAL_IDS:
            for mode in ("DISABLED", "TOOL"):
                rows.append(
                    _base_row(
                        prompt_id=prompt_id,
                        subject_id=subject_id,
                        persona=persona,
                        mode=mode,
                        trial_id=trial_id,
                        event=event,
                    )
                )
    return pd.DataFrame(rows)
