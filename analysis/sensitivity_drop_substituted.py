"""Drop-substituted reserve sensitivity for H1.

Re-fits the primary H1 family after excluding rows whose final prompt came
from a substituted reserve card (``is_substituted == true``). This preserves
the existing analyzer's complete-case and endpoint definitions; it only
changes which prompt clusters are eligible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts import analyze_results as ar


def coerce_substitution_flags(series: pd.Series) -> pd.Series:
    """Return strict boolean ``is_substituted`` flags.

    JSONL pipeline rows carry real booleans, but ad hoc CSV/JSON transforms can
    materialize them as strings. Pandas ``astype(bool)`` treats every non-empty
    string as true, so parse common boolean spellings explicitly and fail closed
    on ambiguous values.
    """
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n", ""}

    def _coerce(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in true_values:
                return True
            if normalized in false_values:
                return False
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ValueError(f"Invalid is_substituted boolean value: {value!r}")

    return series.map(_coerce).astype(bool)


def _load_jsonl(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_json(path, lines=True) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_drop_substituted(
    df: pd.DataFrame,
    *,
    bootstrap_iter: int = 10000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if "is_substituted" not in df.columns:
        work = df.copy()
        work["is_substituted"] = False
    else:
        work = df.copy()
    substituted = coerce_substitution_flags(work["is_substituted"])
    kept = work.loc[~substituted].copy()
    normalized = ar._normalize_analysis_columns(kept)
    return {
        "sensitivity": "drop_substituted",
        "input_rows": int(len(work)),
        "dropped_rows": int(substituted.sum()),
        "retained_rows": int(len(kept)),
        "primary_h1": ar.build_primary_h1_results(
            normalized,
            bootstrap_iter=bootstrap_iter,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stats/sensitivity_drop_substituted.json"),
    )
    parser.add_argument("--bootstrap-iter", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    result = run_drop_substituted(
        _load_jsonl(args.inputs),
        bootstrap_iter=args.bootstrap_iter,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
