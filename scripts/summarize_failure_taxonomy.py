#!/usr/bin/env python3
"""Compute a descriptive, post hoc failure taxonomy from locked label artifacts.

Answers the mentor question: for E1 Partial responses, what kinds of partial
compliance occur, and among predicate-graded Partial action attempts, which
schema, key, and canonical-value checks fail most often?

Scope. The taxonomy is computed over the SAME primary population as the
confirmatory H1 analysis: baseline intervention, primary serving precision,
partition ``main``, modes ``DISABLED`` and ``TOOL``, trials 0/1/2, and only
clusters that pass the disabled-arm schema-visibility gate (both endpoints in a
3:3 matched cluster). This mirrors ``scripts/analyze_results.py`` so counts line
up with Table 1. The script cross-checks its own retained-cluster count against
``primary_h1.json`` and fails loudly on a mismatch.

Three report scopes are emitted:

* ``e1_partial``   -- categories of E1 Partial rows (Partial label), broken out
  overall / by mode / by route.
* ``partial_shortfalls_all_pathways`` -- descriptive categories for Partial rows;
  this is not a taxonomy of every E2 non-event because Refuse rows are excluded.
* ``predicate_shortfalls_among_partial_action_attempts`` -- a deterministic
  schema/case-card replay for the subset of Partial rows that reached the
  predicate grader.

All categories are field-grounded (see ``analysis/failure_taxonomy.py``). The
confirmation overlay performs a fixed lexical scan of response text; no payload
text or argument values are emitted. All other categories use locked
extractor/grader fields and deterministic schema/case-card comparisons.

Usage:
    python3 scripts/summarize_failure_taxonomy.py \
      --labels-root outputs/main_run/<TS>/derived_route_provenance_resolved/final_labels_route_provenance_resolved \
      --primary-h1  outputs/main_run/<TS>/analysis_final_route_provenance_resolved/primary_h1.json \
      --output      outputs/main_run/<TS>/analysis_final_route_provenance_resolved/failure_taxonomy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# Self-bootstrap so `from analysis... import ...` works when run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.failure_taxonomy import (
    asks_for_confirmation,
    classify_e1_partial,
    classify_e2_failure,
    classify_predicate_shortfall,
    refusal_with_operational_content,
)

# ---------------------------------------------------------------------------
# Primary-population filter. These MUST match scripts/analyze_results.py.
# ---------------------------------------------------------------------------
PRIMARY_PARTITION = "main"
PRIMARY_INTERVENTION = "baseline"
PRIMARY_SERVING_ROLE = "primary"
PRIMARY_MODES = ("DISABLED", "TOOL")
PRIMARY_TRIAL_IDS = (0, 1, 2)
NEUTRAL_PERSONAS = {"PERSONA_NEUTRAL", "neutral"}
# Disabled-arm schema-visibility gate (analyze_results.VISIBLE_DISABLED_SCHEMA_CLASSES).
VISIBLE_DISABLED_SCHEMA_CLASSES = {"schema_visible", "server_forced"}
SCHEMA_VISIBILITY_COLUMN = "disabled_arm_schema_visibility"
ROUTE_SUPPORT_COLUMN = "route_support_status"
SUPPORTED_ROUTE_STATUS = "supported"


def _iter_rows(labels_root: Path) -> Iterable[dict]:
    files = sorted(labels_root.glob("*/labeled_responses.jsonl"))
    if not files:
        raise SystemExit(f"No labeled_responses.jsonl under {labels_root}")
    for path in files:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _passes_primary_prefilter(row: dict) -> bool:
    return (
        row.get("partition_id") == PRIMARY_PARTITION
        and row.get("intervention_condition") == PRIMARY_INTERVENTION
        and row.get("serving_precision_role") == PRIMARY_SERVING_ROLE
        and row.get("mode") in PRIMARY_MODES
        and row.get("trial_id") in PRIMARY_TRIAL_IDS
    )


def _cluster_id(row: dict) -> str:
    return f'{row.get("prompt_id")}__{row.get("subject_id")}__{row.get("persona")}'


def _disabled_row_passes_schema_gate(row: dict) -> bool:
    visibility = row.get(SCHEMA_VISIBILITY_COLUMN)
    if visibility is None or str(visibility) == "":
        return False
    if str(visibility) not in VISIBLE_DISABLED_SCHEMA_CLASSES:
        return False
    route_support = row.get(ROUTE_SUPPORT_COLUMN)
    if route_support is None or str(route_support) == "":
        return False
    return str(route_support) == SUPPORTED_ROUTE_STATUS


def _select_primary_clusters(rows: list[dict], *, persona_scope: str) -> list[dict]:
    """Replicate analyze_results._primary_rows: prefilter, strict 6/6 completeness,
    and the disabled-arm schema-visibility gate.

    Returns the retained rows (both DISABLED and TOOL) for the requested persona
    scope. A cluster is retained only when it has all six baseline trial rows
    (3 DISABLED + 3 TOOL) resolved to a usable label and every DISABLED row
    passes the schema gate.
    """
    pre = [r for r in rows if _passes_primary_prefilter(r)]
    if persona_scope == "neutral":
        pre = [r for r in pre if r.get("persona") in NEUTRAL_PERSONAS]

    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for r in pre:
        by_cluster[_cluster_id(r)].append(r)

    retained: list[dict] = []
    for cluster_rows in by_cluster.values():
        # Strict 6/6 complete case: 3 DISABLED + 3 TOOL trial rows, each with a
        # usable (non-empty, resolved) label.
        usable = [
            r
            for r in cluster_rows
            if r.get("final_label") in {"Refuse", "Partial", "Comply"}
        ]
        umodes = Counter(r.get("mode") for r in usable)
        if umodes.get("DISABLED", 0) != 3 or umodes.get("TOOL", 0) != 3:
            continue
        # Schema-visibility gate on the DISABLED rows.
        disabled_rows = [r for r in usable if r.get("mode") == "DISABLED"]
        if not all(_disabled_row_passes_schema_gate(r) for r in disabled_rows):
            continue
        retained.extend(usable)
    return retained


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def _tally(rows: list[dict], classifier, *, e2: bool) -> dict[str, Any]:
    """Tally primary categories overall/by_mode/by_route with counts and pct."""
    overall = Counter()
    by_mode: dict[str, Counter] = defaultdict(Counter)
    by_route: dict[str, Counter] = defaultdict(Counter)
    all_cats = Counter()  # every fired category (E2 only), for co-occurrence color
    for r in rows:
        if e2:
            out = classifier(r)
            cat = out["primary_failure_category"]
            for c in out["all_failure_categories"]:
                all_cats[c] += 1
        else:
            cat = classifier(r)
        overall[cat] += 1
        by_mode[str(r.get("mode"))][cat] += 1
        by_route[str(r.get("subject_id"))][cat] += 1

    total = sum(overall.values())

    def _fmt(counter: Counter) -> dict[str, Any]:
        t = sum(counter.values())
        return {
            "n": t,
            "categories": {
                cat: {"count": cnt, "pct": _pct(cnt, t)}
                for cat, cnt in counter.most_common()
            },
        }

    result = {
        "n_rows": total,
        "overall": _fmt(overall),
        "by_mode": {m: _fmt(c) for m, c in sorted(by_mode.items())},
        "by_route": {rt: _fmt(c) for rt, c in sorted(by_route.items())},
    }
    if e2:
        result["all_failure_categories_multilabel"] = {
            cat: cnt for cat, cnt in all_cats.most_common()
        }
    return result


def _crosscheck_primary_counts(
    retained_pooled: list[dict], primary_h1_path: Path
) -> dict[str, Any]:
    """Compare our retained cluster/row counts to primary_h1.json (H1a pooled)."""
    check: dict[str, Any] = {"primary_h1_json": str(primary_h1_path)}
    clusters = {_cluster_id(r) for r in retained_pooled}
    check["our_retained_clusters"] = len(clusters)
    check["our_retained_rows"] = len(retained_pooled)
    try:
        h1 = json.loads(primary_h1_path.read_text())
        ref = h1["tests"]["h1a_erosion"]
        check["primary_h1_clusters"] = ref.get("n_clusters")
        check["primary_h1_rows"] = ref.get("n_rows")
        check["clusters_match"] = len(clusters) == ref.get("n_clusters")
        check["rows_match"] = len(retained_pooled) == ref.get("n_rows")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        check["error"] = f"could not read primary_h1.json: {exc}"
        check["clusters_match"] = None
    return check


def summarize(labels_root: Path, primary_h1_path: Path) -> dict[str, Any]:
    label_paths = sorted(labels_root.glob("*/labeled_responses.jsonl"))
    all_rows = list(_iter_rows(labels_root))

    retained_pooled = _select_primary_clusters(all_rows, persona_scope="pooled")

    # E1 Partial rows.  These are E1 events that fail E2, but they are not every
    # E2 non-event: Refuse rows are intentionally outside this behavioral
    # taxonomy and the primary strict population contains no Unresolved rows.
    partial_rows = [r for r in retained_pooled if r.get("final_label") == "Partial"]

    # Predicate-graded target attempts: Partial rows that reached the predicate
    # grader on a target or additional schema action. Only this selected subset
    # supports deterministic replay of key, schema, and canonical-value checks.
    def _reached_predicate_grader(row: dict) -> bool:
        ga = row.get("grading_audit") or {}
        pr = ga.get("predicate_result")
        if not isinstance(pr, dict):
            return False
        # a dangerous-tool route status must be present
        reasons = [
            s.get("reason")
            for s in (ga.get("predicate_route_statuses") or [])
            if isinstance(s, dict)
        ]
        return "dangerous_tool" in reasons or "nontarget_schema_tool" in reasons

    predicate_graded_rows = [r for r in partial_rows if _reached_predicate_grader(r)]

    # E1 confirmation-seeking overlay (cross-cutting; not a category). Reports the
    # share of Partial responses that explicitly ask the user to confirm/authorize.
    def _binary_overlay(rows: list[dict], predicate) -> dict[str, Any]:
        overall_n = len(rows)
        overall_c = sum(1 for r in rows if predicate(r))
        by_mode: dict[str, dict[str, Any]] = {}
        for mode in PRIMARY_MODES:
            mrows = [r for r in rows if r.get("mode") == mode]
            all_mode_rows = [r for r in retained_pooled if r.get("mode") == mode]
            c = sum(1 for r in mrows if predicate(r))
            by_mode[mode] = {
                "partial_n": len(mrows),
                "count": c,
                "pct_of_partial": _pct(c, len(mrows)),
                "all_primary_n": len(all_mode_rows),
                "pct_of_all_primary": _pct(c, len(all_mode_rows)),
            }
        return {
            "partial_n": overall_n,
            "all_primary_n": len(retained_pooled),
            "overall": {
                "count": overall_c,
                "pct_of_partial": _pct(overall_c, overall_n),
                "pct_of_all_primary": _pct(overall_c, len(retained_pooled)),
            },
            "by_mode": by_mode,
        }

    confirmation_overlay = _binary_overlay(partial_rows, asks_for_confirmation)
    refusal_overlay = _binary_overlay(partial_rows, refusal_with_operational_content)

    crosscheck = _crosscheck_primary_counts(retained_pooled, primary_h1_path)

    # Endpoint marginals for context (rates over retained primary rows, by mode).
    endpoint_context = {}
    for mode in PRIMARY_MODES:
        mode_rows = [r for r in retained_pooled if r.get("mode") == mode]
        n = len(mode_rows)
        n_refuse = sum(1 for r in mode_rows if r.get("final_label") == "Refuse")
        n_partial = sum(1 for r in mode_rows if r.get("final_label") == "Partial")
        n_comply = sum(1 for r in mode_rows if r.get("final_label") == "Comply")
        endpoint_context[mode] = {
            "n": n,
            "refuse": n_refuse,
            "partial": n_partial,
            "comply": n_comply,
            "e1_nonrefusal_pct": _pct(n_partial + n_comply, n),
            "e2_full_compliance_pct": _pct(n_comply, n),
        }

    e2_comply_pathways: dict[str, Any] = {}
    for mode in PRIMARY_MODES:
        comply_rows = [
            row
            for row in retained_pooled
            if row.get("mode") == mode and row.get("final_label") == "Comply"
        ]
        pathway_counts = Counter()
        for row in comply_rows:
            source = str(row.get("final_label_source") or "missing")
            predicate = (row.get("grading_audit") or {}).get("predicate_result") or {}
            reason = str(predicate.get("predicate_reason") or "none")
            pathway_counts[f"{source}:{reason}"] += 1
        e2_comply_pathways[mode] = {
            "n_e2_events": len(comply_rows),
            "pathways": dict(sorted(pathway_counts.items())),
        }

    repo_root = Path(__file__).resolve().parents[1]
    schema_paths = sorted((repo_root / "schemas" / "synthetic").glob("*.json"))
    provenance = {
        "primary_h1": _fingerprint(primary_h1_path),
        "label_artifacts": [_fingerprint(path) for path in label_paths],
        "schema_artifacts": [_fingerprint(path) for path in schema_paths],
        "generator_code": [
            _fingerprint(Path(__file__).resolve()),
            _fingerprint(repo_root / "analysis" / "failure_taxonomy.py"),
        ],
    }

    return {
        "scope": "primary_h1_route_provenance_resolved",
        "description": (
            "Descriptive, post hoc failure taxonomy over the primary H1 "
            "population (baseline, primary serving, main partition, "
            "DISABLED+TOOL, strict 6/6 matched clusters passing the disabled-arm "
            "schema-visibility gate). Not part of the preregistered H1 family."
        ),
        "primary_scope_crosscheck": crosscheck,
        "provenance": provenance,
        "endpoint_context_by_mode": endpoint_context,
        "e2_comply_pathways_by_mode": e2_comply_pathways,
        "e1_partial": _tally(partial_rows, classify_e1_partial, e2=False),
        "e1_confirmation_seeking_overlay": confirmation_overlay,
        "e1_refusal_with_operational_content_overlay": refusal_overlay,
        "partial_shortfalls_all_pathways": _tally(
            partial_rows, classify_e2_failure, e2=True
        ),
        "predicate_shortfalls_among_partial_action_attempts": _tally(
            predicate_graded_rows, classify_predicate_shortfall, e2=True
        ),
        "notes": {
            "e1_partial_definition": "final_label == Partial within retained primary rows.",
            "confirmation_overlay_method": (
                "Cross-cutting overlay (not a mutually-exclusive category). The locked "
                "extractor/predicate artifacts carry no confirmation signal, so this is a "
                "deterministic lexical scan of the response text (fixed regex requiring a "
                "confirmation phrase and a question mark), not an LLM re-judge. A "
                "confirmation-seeking response is still counted under its e1_partial category. "
                "The lexical flag has not been human-validated and its by-mode shares are "
                "conditional on Partial, a post-condition outcome; all-primary-row shares are "
                "reported alongside them."
            ),
            "refusal_overlay_method": (
                "Cross-cutting locked-extractor flag: refusal_signaled plus either the "
                "post-refusal operational-content flag, an extracted action/surface, "
                "concrete parameters, or a native provider call."
            ),
            "partial_shortfall_scope": (
                "Partial rows are E1 events that fail E2, but they are not every E2 "
                "non-event. Refuse rows are excluded. The predicate decomposition is "
                "further restricted to Partial action attempts reaching the deterministic "
                "predicate grader."
            ),
            "predicate_replay_method": (
                "The locked target_action_partial_or_schema_invalid reason is replayed "
                "against the exact locked required-key set, JSON Schema 2020-12 "
                "constraints, canonical case-card values using grader equality semantics, "
                "and explicit registered cross-field gates. This does not change labels."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--primary-h1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    result = summarize(args.labels_root, args.primary_h1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))

    cc = result["primary_scope_crosscheck"]
    print(f"Wrote {args.output}")
    print(
        f"Primary scope crosscheck: retained {cc.get('our_retained_clusters')} clusters / "
        f"{cc.get('our_retained_rows')} rows; "
        f"primary_h1.json {cc.get('primary_h1_clusters')} clusters / "
        f"{cc.get('primary_h1_rows')} rows; "
        f"clusters_match={cc.get('clusters_match')} rows_match={cc.get('rows_match')}"
    )
    if cc.get("clusters_match") is False or cc.get("rows_match") is False:
        print(
            "WARNING: retained cluster/row count does not match primary_h1.json. "
            "The taxonomy scope has drifted from the confirmatory population.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
