"""Exploratory TOOL x is_substituted interaction sensitivity.

Fits the H1 conditional-logit endpoint family with the usual TOOL indicator
plus ``TOOL * is_substituted``. A plain ``is_substituted`` main effect is not
included because substitution is cluster-constant and is absorbed by the
conditional-logit group fixed effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.sensitivity_drop_substituted import coerce_substitution_flags
from scripts import analyze_results as ar


TERM = "tool_x_is_substituted"


def _load_jsonl(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_json(path, lines=True) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _empty(engine: str, *, n_rows: int = 0, n_clusters: int = 0) -> dict[str, Any]:
    return {
        "engine": engine,
        "term": TERM,
        "beta": None,
        "se": None,
        "or": None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "n_rows": int(n_rows),
        "n_clusters": int(n_clusters),
    }


def _fit_tool_substituted_interaction(
    rows: pd.DataFrame,
    endpoint_col: str,
) -> dict[str, Any]:
    if rows.empty:
        return _empty("no_data")
    rows = ar._apply_cluster_six_six_filter(rows, endpoint_col=endpoint_col)
    if rows.empty:
        return _empty("no_data_after_six_six_filter")
    model_df = (
        rows.dropna(subset=[endpoint_col])
        .sort_values(["cluster_id", "trial_id", "mode"], kind="mergesort")
        .copy()
    )
    model_df["tool"] = (model_df["mode"] == "TOOL").astype(int)
    substituted = (
        model_df["is_substituted"]
        if "is_substituted" in model_df.columns
        else pd.Series(False, index=model_df.index)
    )
    model_df["is_substituted_bool"] = coerce_substitution_flags(substituted).astype(int)
    model_df[TERM] = model_df["tool"] * model_df["is_substituted_bool"]
    n_clusters = int(model_df["cluster_id"].nunique(dropna=False))
    if model_df[TERM].nunique(dropna=False) < 2:
        return _empty(
            "interaction_not_estimable_no_substituted_tool_variation",
            n_rows=len(model_df),
            n_clusters=n_clusters,
        )
    exog = model_df[["tool", TERM]].copy()
    try:
        model = ar.ConditionalLogit(
            model_df[endpoint_col].astype(int),
            exog,
            groups=model_df["cluster_id"],
        )
        with ar.warnings.catch_warnings():
            ar.warnings.simplefilter("error", ar.ConvergenceWarning)
            fit = model.fit(disp=False)
        params = ar.np.asarray(fit.params, dtype=float)
        retained_clusters = ar._retained_group_prompt_clusters(
            model_df,
            endpoint_col,
            "cluster_id",
        )
        cov = ar._prompt_clustered_sandwich_cov(model, params, retained_clusters)
        beta = float(fit.params[TERM])
        term_idx = list(fit.params.index).index(TERM)
        se = float(ar.np.sqrt(cov[term_idx, term_idx]))
        if se <= 0 or not ar.np.isfinite(se):
            raise ValueError(f"invalid prompt-clustered sandwich SE: {se!r}")
        p_value = 2 * ar.stats.norm.sf(abs(beta / se))
        return {
            "engine": "conditional_logit_tool_is_substituted_interaction_prompt_clustered",
            "term": TERM,
            "beta": beta,
            "se": se,
            "or": float(ar.np.exp(beta)),
            "ci_low": float(ar.np.exp(beta - 1.96 * se)),
            "ci_high": float(ar.np.exp(beta + 1.96 * se)),
            "p_value": float(p_value),
            "n_rows": int(len(model_df)),
            "n_clusters": n_clusters,
            "substituted_clusters": int(
                model_df.loc[
                    model_df["is_substituted_bool"] == 1, "cluster_id"
                ].nunique(dropna=False)
            ),
        }
    except (
        ar.PerfectSeparationError,
        ar.ConvergenceWarning,
        RecursionError,
        ar.np.linalg.LinAlgError,
        ValueError,
    ) as exc:
        result = _empty(
            "interaction_not_estimable",
            n_rows=len(model_df),
            n_clusters=n_clusters,
        )
        result["fit_error"] = str(exc)
        return result


def run_tool_is_substituted_interaction(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    if "is_substituted" not in work.columns:
        work["is_substituted"] = False
    normalized = ar._normalize_analysis_columns(work)
    tests: dict[str, Any] = {}
    for test_name, persona_scope, endpoint_col in ar.H1_TEST_SPECS:
        rows = ar._primary_rows(normalized, persona_scope=persona_scope)
        tests[test_name] = _fit_tool_substituted_interaction(rows, endpoint_col)
    return {
        "sensitivity": "tool_is_substituted_interaction",
        "scope": "exploratory_not_holm",
        "tests": tests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stats/sensitivity_tool_is_substituted_interaction.json"),
    )
    args = parser.parse_args()

    result = run_tool_is_substituted_interaction(_load_jsonl(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
