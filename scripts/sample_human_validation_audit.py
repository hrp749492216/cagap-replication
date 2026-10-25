#!/usr/bin/env python3
"""Deterministic stratified sampler for the post-registration human
measurement-validation audit (CAGAP_human_validation_addendum_2026-05-26).

Validation universes (per addendum §4):

  - U-LLM:     predicate_resolved == False  (rows whose final label is set
               by the family-excluded LLM judge panel; target n = 600).

  - U-DET-POS: predicate_resolved == True AND predicate_reason in
               {structured_tool_calls, complete_tool_call_dict}  (deterministic
               structured-action / Comply resolutions; target n = 200).

  - U-DET-NEG: predicate_resolved == True AND predicate_reason in
               {deterministic_clear_refusal}  (deterministic Refuse / no-action
               / boundary resolutions; target n = 100).

  U-DET := U-DET-POS ∪ U-DET-NEG. U-LLM and U-DET are disjoint and jointly
  cover every confirmatory-eligible resolved row.

Stratification:
  - U-LLM:     (subject, mode, persona); 48 rows from TOOL/DISABLED + 12 rows
               from NOTOOLS/TOOL_IRRELEVANT per subject. If a primary cell is
               sparse because the deterministic predicate dominates that
               subject's TOOL/DISABLED arms (a real condition on this run for
               llama-3.3-70b, gemma-4-31b, glm-4.7-flash), the per-subject
               deficit spills into the context block within the same subject.
  - U-DET-POS: subject-balanced across the structured-action pool.
  - U-DET-NEG: subject-balanced across the clear-refusal / boundary pool.

Outputs the sampling manifest at:
  paper/addenda/sampling/human_validation_sampling_manifest_<DATE>.json

The manifest records: universe SHA-256, sampling seed, strata, selected
custom_ids per universe with stratum tag, exclusions with reasons, script
SHA-256, predicate-decisions-snapshot hashes, environment fingerprint.

NOTE on timing (addendum §2 + §5):
- Sampling executes against a grading-only snapshot (predicate decisions +
  extracted structured actions). No H1 model estimate, TOOL-vs-DISABLED
  effect, p-value, Holm-adjusted result, or endpoint summary is computed
  or inspected before the sample and manifest are frozen.
- Ensemble judge labels are not yet finalized at sampling time (round-1).
  U-LLM is therefore stratified by (subject, mode, persona); ordinal-label
  balance is reported post-hoc as a diagnostic, not as a re-draw trigger.
- No human rater receives any audit row until both the addendum and the
  sampling manifest are uploaded to OSF and timestamped.

Default subject list excludes claude-opus-4-7 (still dispatching at filing
time). Once Claude is closed and predicate-audited, re-run the script with
  --subjects claude-opus-4-7 --audit-date 2026-05-26-r2
to add Claude's 60 U-LLM + 30 U-DET rows in a round-2 manifest with the
same frozen seed and the round-1 universe definition extended with Claude's
predicate-decisions file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JB = REPO / "outputs" / "main_run" / "20260525T164059Z" / "judge_batches"
DEFAULT_OUT = REPO / "paper" / "addenda" / "sampling"

DEFAULT_SUBJECTS = [
    "llama-3.3-70b-instruct",
    "mistral-medium-3.5-128b",
    "gemma-4-31b-it",
    "qwen3.6-35b-a3b",
    "glm-4.7-flash",
    "gemini-3.5-flash",
    "kimi-k2.6",
    "grok-4.3",
    "gpt-5.5",
]

_VALID_MODES = ("NOTOOLS", "DISABLED", "TOOL", "TOOL_IRRELEVANT")
_VALID_PERSONAS = ("neutral", "operator")
_VALID_ROLES = ("baseline", "hardened")
_QUANT_SUFFIX = "quantization_audit"


def _split_custom_id(cid: str) -> dict | None:
    """Split on `__`, identify the (prompt, subject, mode, persona, trial,
    role) anchors, and reject quantization_audit rows (which have an extra
    trailing `__quantization_audit` segment).

    The custom_id can carry hyphens and single underscores inside the prompt
    and subject segments, so we anchor on the unambiguous mode token instead
    of using a fragile global regex.
    """
    if not cid.startswith("main__"):
        return None
    parts = cid.split("__")
    if parts[0] != "main":
        return None
    if parts[-1] == _QUANT_SUFFIX:
        return None
    if len(parts) < 7:
        return None
    role = parts[-1]
    trial = parts[-2]
    persona = parts[-3]
    mode = parts[-4]
    if role not in _VALID_ROLES:
        return None
    if persona not in _VALID_PERSONAS:
        return None
    if mode not in _VALID_MODES:
        return None
    if not re.fullmatch(r"trial\d+", trial):
        return None
    subject = parts[-5]
    prompt = "__".join(parts[1:-5])
    if not prompt or not subject:
        return None
    return {
        "prompt": prompt,
        "subject": subject,
        "mode": mode,
        "persona": persona,
        "trial": trial,
        "role": role,
    }

PREDICATE_POSITIVE_REASONS = {"structured_tool_calls", "complete_tool_call_dict"}
PREDICATE_NEGATIVE_REASONS = {"deterministic_clear_refusal"}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(2 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_canonical_jsonl(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def parse_custom_id(cid: str) -> dict | None:
    return _split_custom_id(cid)


def load_universe(jb_dir: Path, subjects: list[str]) -> tuple[list[dict], dict]:
    """Returns (rows, per_subject_counts). Each row dict carries:
        subject, mode, persona, trial, role, prompt, custom_id,
        predicate_resolved, predicate_reason.
    Only baseline + hardened roles are kept (primary-eligible).
    """
    rows: list[dict] = []
    per_subject = {}
    for sid in subjects:
        pdec = jb_dir / sid / "predicate_decisions.jsonl"
        if not pdec.exists():
            print(f"[warn] no predicate_decisions for {sid}: {pdec}", file=sys.stderr)
            per_subject[sid] = {"baseline": 0, "hardened": 0, "skipped_role": 0, "unparsed": 0}
            continue
        per_role = defaultdict(int)
        skipped_quant_audit = 0
        unparsed = 0
        with pdec.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                cid = d["custom_id"]
                if cid.endswith(f"__{_QUANT_SUFFIX}"):
                    skipped_quant_audit += 1
                    continue
                parts = parse_custom_id(cid)
                if not parts:
                    unparsed += 1
                    continue
                role = parts["role"]
                per_role[role] += 1
                if role not in ("baseline", "hardened"):
                    continue
                rows.append({
                    "custom_id": cid,
                    "subject": parts["subject"],
                    "mode": parts["mode"],
                    "persona": parts["persona"],
                    "trial": parts["trial"],
                    "role": parts["role"],
                    "prompt": parts["prompt"],
                    "predicate_resolved": bool(d.get("predicate_resolved")),
                    "predicate_reason": d.get("predicate_reason") or "",
                })
        per_subject[sid] = dict(per_role) | {
            "skipped_quant_audit": skipped_quant_audit,
            "unparsed": unparsed,
        }
    rows.sort(key=lambda r: r["custom_id"])
    return rows, per_subject


def _stratum_key_ullm(row: dict) -> tuple:
    return (row["subject"], row["mode"], row["persona"])


def _stratum_key_udet(row: dict) -> tuple:
    if row["predicate_reason"] in PREDICATE_POSITIVE_REASONS:
        bucket = "positive"
    elif row["predicate_reason"] in PREDICATE_NEGATIVE_REASONS:
        bucket = "negative"
    else:
        bucket = "other"
    return (row["subject"], row["mode"], row["persona"], bucket)


MODES_PRIMARY = ("TOOL", "DISABLED")
MODES_CONTEXT = ("NOTOOLS", "TOOL_IRRELEVANT")


def sample_ullm(
    rows: list[dict],
    subjects: list[str],
    rng: random.Random,
    per_subject_target: int = 60,
    primary_ratio: tuple[int, int] = (48, 12),
) -> tuple[list[dict], dict]:
    """Sample U-LLM rows (predicate_resolved == False).

    Strategy:
      1. Aim for n_primary rows from TOOL/DISABLED, balanced across the four
         (mode, persona) cells; if a primary cell is sparse, take all of it.
      2. Aim for n_context rows from NOTOOLS/TOOL_IRRELEVANT, similarly
         balanced; if a context cell is sparse, take all of it.
      3. If after (1) the primary shortfall is large because predicate resolves
         most TOOL/DISABLED rows for that subject, spill the deficit into the
         context block (and vice versa), so the total per-subject draw moves
         toward per_subject_target. Spill direction and counts are logged.
      4. Per-subject shortfall is recorded transparently when even after spill
         the subject pool is too thin.
    """
    pool = [r for r in rows if not r["predicate_resolved"]]
    n_primary, n_context = primary_ratio
    selected: list[dict] = []
    selected_cids: set[str] = set()
    stratum_log: dict = defaultdict(lambda: {"available": 0, "drawn": 0, "shortfall": 0})

    def _draw_from(rs: list[dict], k: int, tag: str) -> int:
        nonlocal selected
        if not rs or k <= 0:
            return 0
        idx = [i for i, r in enumerate(rs) if r["custom_id"] not in selected_cids]
        if not idx:
            return 0
        rng.shuffle(idx)
        drawn = 0
        for i in idx:
            if drawn >= k:
                break
            row = dict(rs[i])
            row["_stratum_tag"] = tag
            selected.append(row)
            selected_cids.add(row["custom_id"])
            drawn += 1
        return drawn

    def _balanced_draw(
        pool_block: list[dict],
        modes: tuple[str, ...],
        target_total: int,
        tag_prefix: str,
    ) -> int:
        """Try to take target_total rows from a (modes × personas) block,
        roughly balanced across the four cells. Return total drawn."""
        cells = []
        for mode in modes:
            for persona in ("neutral", "operator"):
                cell = [r for r in pool_block if r["mode"] == mode and r["persona"] == persona]
                cells.append((mode, persona, cell))
        n_cells = len(cells)
        per_cell = max(1, target_total // n_cells)
        drawn_block = 0
        for mode, persona, cell in cells:
            tag = f"{tag_prefix}::{mode}::{persona}"
            stratum_log[tag]["available"] = len(cell)
            got = _draw_from(cell, per_cell, tag)
            stratum_log[tag]["drawn"] = got
            stratum_log[tag]["shortfall"] = max(0, per_cell - got)
            drawn_block += got
        deficit = max(0, target_total - drawn_block)
        if deficit > 0:
            remaining = [r for r in pool_block if r["custom_id"] not in selected_cids]
            tag = f"{tag_prefix}::REALLOC_WITHIN_BLOCK"
            stratum_log[tag]["available"] = len(remaining)
            got = _draw_from(remaining, deficit, tag)
            stratum_log[tag]["drawn"] = got
            stratum_log[tag]["shortfall"] = max(0, deficit - got)
            drawn_block += got
        return drawn_block

    for sid in subjects:
        sub_pool = [r for r in pool if r["subject"] == sid]
        if not sub_pool:
            stratum_log[f"ULLM::{sid}::__missing__"] = {
                "available": 0, "drawn": 0, "shortfall": per_subject_target,
            }
            continue

        sub_primary = [r for r in sub_pool if r["mode"] in MODES_PRIMARY]
        sub_context = [r for r in sub_pool if r["mode"] in MODES_CONTEXT]

        drawn_primary = _balanced_draw(sub_primary, MODES_PRIMARY, n_primary,
                                       tag_prefix=f"ULLM::{sid}::PRIMARY")
        drawn_context = _balanced_draw(sub_context, MODES_CONTEXT, n_context,
                                       tag_prefix=f"ULLM::{sid}::CONTEXT")

        subject_drawn = drawn_primary + drawn_context
        subject_deficit = max(0, per_subject_target - subject_drawn)

        spill_log_tag = f"ULLM::{sid}::PRIMARY_CONTEXT_SPILL"
        if subject_deficit > 0:
            spill_from_context = [r for r in sub_context if r["custom_id"] not in selected_cids]
            stratum_log[spill_log_tag]["available"] = len(spill_from_context)
            spill_got = _draw_from(spill_from_context, subject_deficit, spill_log_tag)
            stratum_log[spill_log_tag]["drawn"] = spill_got
            stratum_log[spill_log_tag]["shortfall"] = max(0, subject_deficit - spill_got)
            subject_deficit -= spill_got
        else:
            stratum_log[spill_log_tag]["available"] = 0
            stratum_log[spill_log_tag]["drawn"] = 0
            stratum_log[spill_log_tag]["shortfall"] = 0

        if subject_deficit > 0:
            spill_from_primary = [r for r in sub_primary if r["custom_id"] not in selected_cids]
            spill_tag2 = f"ULLM::{sid}::CONTEXT_PRIMARY_SPILL"
            stratum_log[spill_tag2]["available"] = len(spill_from_primary)
            spill_got2 = _draw_from(spill_from_primary, subject_deficit, spill_tag2)
            stratum_log[spill_tag2]["drawn"] = spill_got2
            stratum_log[spill_tag2]["shortfall"] = max(0, subject_deficit - spill_got2)
            subject_deficit -= spill_got2

        subj_summary_tag = f"ULLM::{sid}::__subject_total__"
        stratum_log[subj_summary_tag]["available"] = len(sub_pool)
        stratum_log[subj_summary_tag]["drawn"] = per_subject_target - subject_deficit
        stratum_log[subj_summary_tag]["shortfall"] = subject_deficit

    return selected, dict(stratum_log)


def sample_udet(
    rows: list[dict],
    subjects: list[str],
    rng: random.Random,
    target_positive: int = 200,
    target_negative: int = 100,
) -> tuple[list[dict], dict]:
    """Sample U-DET rows. predicate_resolved == True.

    target_positive rows from {structured_tool_calls, complete_tool_call_dict};
    target_negative rows from {deterministic_clear_refusal}.

    Allocate proportionally across subjects, then within each subject balance
    across (mode, persona).
    """
    pool = [r for r in rows if r["predicate_resolved"]]
    selected: list[dict] = []
    stratum_log: dict = defaultdict(lambda: {"available": 0, "drawn": 0, "shortfall": 0})

    def _alloc_subject(target: int, n_subj: int) -> list[int]:
        base = target // n_subj
        rem = target - base * n_subj
        allocs = [base] * n_subj
        for i in range(rem):
            allocs[i] += 1
        return allocs

    allocs_pos = _alloc_subject(target_positive, len(subjects))
    allocs_neg = _alloc_subject(target_negative, len(subjects))

    def _draw_from(rs: list[dict], k: int, tag: str) -> int:
        nonlocal selected
        if not rs:
            return 0
        idx = list(range(len(rs)))
        rng.shuffle(idx)
        drawn = 0
        for i in idx:
            if drawn >= k:
                break
            row = dict(rs[i])
            row["_stratum_tag"] = tag
            selected.append(row)
            drawn += 1
        return drawn

    for sid, want_pos, want_neg in zip(subjects, allocs_pos, allocs_neg):
        sub_pool = [r for r in pool if r["subject"] == sid]
        pos_pool = [r for r in sub_pool if r["predicate_reason"] in PREDICATE_POSITIVE_REASONS]
        neg_pool = [r for r in sub_pool if r["predicate_reason"] in PREDICATE_NEGATIVE_REASONS]

        tag = f"UDET::{sid}::positive"
        stratum_log[tag]["available"] = len(pos_pool)
        got = _draw_from(pos_pool, want_pos, tag)
        stratum_log[tag]["drawn"] = got
        stratum_log[tag]["shortfall"] = max(0, want_pos - got)

        tag = f"UDET::{sid}::negative"
        stratum_log[tag]["available"] = len(neg_pool)
        got = _draw_from(neg_pool, want_neg, tag)
        stratum_log[tag]["drawn"] = got
        stratum_log[tag]["shortfall"] = max(0, want_neg - got)

    return selected, dict(stratum_log)


def env_fingerprint() -> dict:
    fp = {
        "python_version": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
    }
    try:
        import pkg_resources  # type: ignore

        fp["installed_packages_count"] = len(list(pkg_resources.working_set))
    except Exception:
        pass
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jb-dir", type=Path, default=DEFAULT_JB,
                    help="judge_batches directory containing per-subject predicate_decisions.jsonl")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="output directory for the sampling manifest")
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS,
                    help="subject IDs in scope at sampling time")
    ap.add_argument("--seed", type=int, default=20260526,
                    help="frozen sampling seed (default: 20260526, the filing date)")
    ap.add_argument("--target-ullm-per-subject", type=int, default=60)
    ap.add_argument("--target-udet-positive-total", type=int, default=200)
    ap.add_argument("--target-udet-negative-total", type=int, default=100)
    ap.add_argument("--audit-date", type=str, default="2026-05-26",
                    help="date stamp for the manifest filename")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute the manifest but do not write it to disk")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / f"human_validation_sampling_manifest_{args.audit_date}.json"

    script_path = Path(__file__).resolve()
    script_sha = _sha256_file(script_path)

    rows, per_subject_counts = load_universe(args.jb_dir, args.subjects)
    print(f"[universe] loaded {len(rows)} rows across {len(args.subjects)} subjects "
          f"(baseline+hardened roles only)")
    for sid, counts in per_subject_counts.items():
        print(f"  {sid}: {counts}")

    universe_sha = _sha256_canonical_jsonl(rows)
    print(f"[universe] SHA-256 = {universe_sha}")

    pred_manifest_hashes = {}
    for sid in args.subjects:
        pdec = args.jb_dir / sid / "predicate_decisions.jsonl"
        if pdec.exists():
            pred_manifest_hashes[sid] = _sha256_file(pdec)

    rng = random.Random(args.seed)

    ullm_rows, ullm_strata = sample_ullm(
        rows, args.subjects, rng,
        per_subject_target=args.target_ullm_per_subject,
    )
    udet_rows, udet_strata = sample_udet(
        rows, args.subjects, rng,
        target_positive=args.target_udet_positive_total,
        target_negative=args.target_udet_negative_total,
    )

    print(f"[sample] U-LLM  drawn: {len(ullm_rows)}")
    print(f"[sample] U-DET drawn: {len(udet_rows)}")
    ullm_shortfall_total = sum(s.get("shortfall", 0) for s in ullm_strata.values())
    udet_shortfall_total = sum(s.get("shortfall", 0) for s in udet_strata.values())
    print(f"[sample] U-LLM total shortfall: {ullm_shortfall_total}")
    print(f"[sample] U-DET total shortfall: {udet_shortfall_total}")

    overlap = ({r["custom_id"] for r in ullm_rows}
               & {r["custom_id"] for r in udet_rows})
    if overlap:
        print(f"[error] overlap between U-LLM and U-DET: {len(overlap)} rows", file=sys.stderr)
        sys.exit(1)

    ullm_selection_sha = _sha256_canonical_jsonl(
        sorted(ullm_rows, key=lambda r: r["custom_id"])
    )
    udet_selection_sha = _sha256_canonical_jsonl(
        sorted(udet_rows, key=lambda r: r["custom_id"])
    )

    manifest = {
        "addendum_id": "CAGAP_human_validation_addendum_2026-05-26",
        "osf_registration": "yz8hn",
        "audit_date": args.audit_date,
        "filed_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "subjects_in_scope": list(args.subjects),
        "subjects_not_yet_in_scope": ["claude-opus-4-7"],
        "seed": args.seed,
        "script_path": str(script_path.relative_to(REPO)),
        "script_sha256": script_sha,
        "judge_batches_dir": str(args.jb_dir.relative_to(REPO)),
        "predicate_decisions_sha256_per_subject": pred_manifest_hashes,
        "universe_sha256": universe_sha,
        "universe_row_count": len(rows),
        "universe_per_subject_role_counts": per_subject_counts,
        "ullm": {
            "target_per_subject": args.target_ullm_per_subject,
            "target_total": args.target_ullm_per_subject * len(args.subjects),
            "drawn": len(ullm_rows),
            "selection_sha256": ullm_selection_sha,
            "stratum_log": ullm_strata,
            "selected_custom_ids": [r["custom_id"] for r in ullm_rows],
            "selected_with_strata": [
                {"custom_id": r["custom_id"], "stratum_tag": r["_stratum_tag"]}
                for r in ullm_rows
            ],
        },
        "udet": {
            "target_positive_total": args.target_udet_positive_total,
            "target_negative_total": args.target_udet_negative_total,
            "drawn": len(udet_rows),
            "selection_sha256": udet_selection_sha,
            "stratum_log": udet_strata,
            "selected_custom_ids": [r["custom_id"] for r in udet_rows],
            "selected_with_strata": [
                {"custom_id": r["custom_id"], "stratum_tag": r["_stratum_tag"]}
                for r in udet_rows
            ],
        },
        "environment_fingerprint": env_fingerprint(),
        "interpretation_rule_pointer": "paper/addenda/CAGAP_human_validation_addendum_2026-05-26.md §11",
        "timing_claim": (
            "Sampling executed after OSF registration yz8hn and after main subject "
            "dispatch began, but before human-validation labels were generated, "
            "before Phase D final merged analysis, and before inspection of "
            "human-vs-LLM or human-vs-predicate agreement statistics. U-LLM "
            "stratification omits provisional ensemble-label class because the "
            "ensemble label is not yet finalized; label balance is reported "
            "post-hoc as a diagnostic, not as a re-draw trigger."
        ),
    }

    if args.dry_run:
        print(f"[dry-run] manifest would be written to {manifest_path}")
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
        print(f"[dry-run] manifest size = {len(manifest_text)} chars")
        manifest_sha = _sha256_str(manifest_text)
        print(f"[dry-run] manifest SHA-256 = {manifest_sha}")
        return 0

    text = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path.write_text(text)
    manifest_sha = _sha256_str(text)
    sha_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    sha_path.write_text(manifest_sha + "  " + manifest_path.name + "\n")
    print(f"[done] wrote {manifest_path}")
    print(f"[done] wrote {sha_path}")
    print(f"[done] manifest SHA-256 = {manifest_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
