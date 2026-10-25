"""Dispatch-block balance audit for substituted reserve rows.

Reports where substituted-reserve rows fall in the chronological dispatch
blocks used by the primary TOOL-vs-DISABLED analysis. This is descriptive
only; it does not refit H1 or enter the Holm family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.sensitivity_drop_substituted import coerce_substitution_flags
from scripts import analyze_results as ar


MISSING_BLOCK_ID = "<missing_dispatch_block_id>"


def _load_jsonl(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_json(path, lines=True) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _prompt_ids(series: pd.Series) -> list[str]:
    return sorted({str(value) for value in series.dropna().tolist()})


def _block_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block_id, group in rows.groupby("dispatch_block_id", dropna=False, sort=False):
        substituted = group["is_substituted_bool"].astype(bool)
        substituted_rows = int(substituted.sum())
        total_rows = int(len(group))
        substituted_group = group.loc[substituted]
        records.append({
            "dispatch_block_id": str(block_id),
            "total_rows": total_rows,
            "substituted_rows": substituted_rows,
            "non_substituted_rows": int(total_rows - substituted_rows),
            "substituted_clusters": int(
                substituted_group["cluster_id"].nunique(dropna=False)
            ),
            "substituted_prompt_ids": _prompt_ids(substituted_group["prompt_id"]),
            "substitution_row_share": (
                float(substituted_rows / total_rows) if total_rows else 0.0
            ),
        })
    return sorted(
        records,
        key=lambda item: (-int(item["substituted_rows"]), str(item["dispatch_block_id"])),
    )


def run_substitution_dispatch_block_balance(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    if "is_substituted" not in work.columns:
        work["is_substituted"] = False
    if "dispatch_block_id" not in work.columns:
        work["dispatch_block_id"] = MISSING_BLOCK_ID
    else:
        work["dispatch_block_id"] = (
            work["dispatch_block_id"].fillna(MISSING_BLOCK_ID).astype(str)
        )

    work["is_substituted_bool"] = coerce_substitution_flags(work["is_substituted"])
    normalized = ar._normalize_analysis_columns(work)
    primary = ar._primary_rows(normalized, persona_scope="pooled")
    if primary.empty:
        return {
            "sensitivity": "substitution_dispatch_block_balance",
            "scope": "descriptive_not_holm_h1_primary_rows",
            "input_rows": int(len(work)),
            "analysis_rows": 0,
            "has_substitutions": False,
            "substituted_rows": 0,
            "non_substituted_rows": 0,
            "substituted_clusters": 0,
            "substituted_dispatch_blocks": 0,
            "block_table": [],
        }

    substituted = primary["is_substituted_bool"].astype(bool)
    substituted_rows = int(substituted.sum())
    substituted_primary = primary.loc[substituted]
    return {
        "sensitivity": "substitution_dispatch_block_balance",
        "scope": "descriptive_not_holm_h1_primary_rows",
        "input_rows": int(len(work)),
        "analysis_rows": int(len(primary)),
        "has_substitutions": bool(substituted_rows),
        "substituted_rows": substituted_rows,
        "non_substituted_rows": int(len(primary) - substituted_rows),
        "substituted_clusters": int(
            substituted_primary["cluster_id"].nunique(dropna=False)
        ),
        "substituted_dispatch_blocks": int(
            substituted_primary["dispatch_block_id"].nunique(dropna=False)
        ),
        "block_table": _block_records(primary),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stats/sensitivity_substitution_dispatch_block_balance.json"),
    )
    args = parser.parse_args()

    result = run_substitution_dispatch_block_balance(_load_jsonl(args.inputs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
