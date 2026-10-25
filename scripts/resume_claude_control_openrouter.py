#!/usr/bin/env python3
"""Collect only the frozen 952 missing Claude rows into separate response segments."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import requests  # noqa: E402
from scripts import run_disabled_format_control as control  # noqa: E402
from scripts.run_disabled_format_control_openrouter_n3_full import (  # noqa: E402
    validate_full_execution_authorization,
)
from scripts.run_synthetic_prompts import (  # noqa: E402
    Config, NativeBatchSubmitter, OPENROUTER_FALLBACK_MAX_ATTEMPTS,
    write_openrouter_model_metadata_snapshot,
)

ROOT = REPO / "outputs/postreg_disabled_format_control_claude_resume_20261006"
SUBJECT = "claude-opus-4-7"
NON_DISPATCH_FILES = {
    "scripts/analyze_disabled_format_control.py", "tests/test_disabled_format_control.py",
}
HISTORICAL_METADATA = REPO / (
    "outputs/postreg_disabled_format_control_v3_openrouter_n3_eligible/responses/"
    "openrouter_model_metadata/anthropic_claude-opus-4.7__anthropic.json"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_inputs():
    manifest = json.loads((ROOT / "resume_manifest.json").read_text())
    bindings = {**manifest["source_artifacts"], "queue": manifest["resume_queue"]}
    for binding in bindings.values():
        if sha(REPO / binding["path"]) != binding["sha256"]:
            raise ValueError(f"Source hash mismatch: {binding['path']}")
    protocol_path = REPO / bindings["protocol"]["path"]
    protocol = json.loads(protocol_path.read_text())
    exceptions = {}
    for group in ("grading_contract_sha256", "implementation_contract_sha256"):
        for name, expected in protocol[group].items():
            actual = sha(REPO / name)
            if actual != expected:
                if name not in NON_DISPATCH_FILES:
                    raise ValueError(f"Frozen collection dependency changed: {name}")
                exceptions[name] = {"historical_sha256": expected, "current_sha256": actual,
                                    "executed_by_resume_launcher": False}
    control._validate_protocol_structure(protocol)
    auth_path = REPO / bindings["full_execution_authorization"]["path"]
    auth = json.loads(auth_path.read_text())
    validate_full_execution_authorization(
        auth_path, protocol=protocol, protocol_path=protocol_path,
        protocol_sha256=bindings["protocol"]["sha256"],
        locked_inputs_manifest_path=REPO / bindings["locked_inputs_manifest"]["path"],
        canary_authorization_path=REPO / auth["canary_execution_authorization"],
        subject_id=SUBJECT,
    )
    expected = read_rows(REPO / bindings["expected_requests"]["path"])
    prior = read_rows(REPO / bindings["original_responses"]["path"])
    rows = read_rows(REPO / bindings["queue"]["path"])
    ids = [r["custom_id"] for r in prior]
    if len(ids) != len(set(ids)) or len(prior) != 2504 or len(expected) != 3456:
        raise ValueError("Original inventory mismatch")
    seen = set(ids)
    if rows != [r for r in expected if r["custom_id"] not in seen] or len(rows) != 952:
        raise ValueError("Queue is not the exact ordered missing set")
    control.validate_generated_rows(expected, protocol, canary=False)
    return manifest, rows, prior, protocol, exceptions


def number(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid monetary value")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Invalid monetary value")
    return result


def ceiling(row, prices):
    input_rate = max(number(prices[key]) for key in
                     ("prompt", "input_cache_write", "input_cache_write_1h") if key in prices)
    output_rate = number(prices["completion"])
    body = row["body"]
    # Match the existing conservative byte ceiling, with additional wrapping margin.
    input_bytes = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 4096
    return input_bytes * input_rate + int(body.get("max_tokens", 4096)) * output_rate


def response_cost(record, row, prices):
    try:
        return control._openrouter_reported_cost(record, custom_id=row["custom_id"]), "reported"
    except ValueError:
        return ceiling(row, prices), "conservative_reserve"


def new_records(root):
    return [record for path in sorted(root.glob("runs/*/responses.jsonl"))
            for record in read_rows(path)]


def completed_records(root, rows):
    records = new_records(root)
    ids = [r["custom_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate saved custom_id")
    if ids != [r["custom_id"] for r in rows[:len(ids)]]:
        raise ValueError("Saved results are not an ordered queue prefix")
    pending_path = root / "pending.json"
    if pending_path.exists():
        pending = json.loads(pending_path.read_text())
        if pending["custom_id"] not in ids:
            raise ValueError("Unreconciled pending request: inspect pending.json and raw archive before resuming")
    return records


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "collector.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Claude resume collector is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class CountingSession(requests.Session):
    attempts = 0

    def post(self, *args, **kwargs):
        self.attempts += 1
        return super().post(*args, **kwargs)


def collect(*, root, rows, prior_spend, route, prices, api_key, submitter,
            snapshot, cap, available, receipt=None):
    saved = completed_records(root, rows)
    prior_new_spend = sum(number(r["execution_budget"]["accounted_cost_usd"]) for r in saved)
    spent = prior_spend + prior_new_spend
    new_spend = 0.0
    launch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = root / "runs" / launch
    run_dir.mkdir(parents=True)
    write_json(run_dir / "receipt.json", {"created_at_utc": launch, "prior_accounted_usd": spent,
               "subject_cap_usd": cap, "prices": prices, "snapshot": snapshot,
               "missing_cost_policy": "retain outcome; conservatively reserve, never impute actual cost",
               "retry_cost_policy": "reserve previous attempts whose cost is unknown", **(receipt or {})})
    pending_path = root / "pending.json"
    pending_path.unlink(missing_ok=True)  # only after completed_records reconciles it
    count = len(saved)

    def status(state):
        result = {"status": state, "completed": count, "total": len(rows),
                  "remaining": len(rows) - count, "accounted_total_usd": round(spent, 8),
                  "new_accounted_this_launch_usd": round(new_spend, 8), "run_dir": str(run_dir)}
        write_json(root / "status.json", result)
        write_json(run_dir / "status.json", result)
        return result

    with (run_dir / "responses.jsonl").open("a", encoding="utf-8") as handle:
        for row in rows[count:]:
            reserve = ceiling(row, prices)
            max_attempt_reserve = reserve * OPENROUTER_FALLBACK_MAX_ATTEMPTS
            if spent + max_attempt_reserve > cap or new_spend + max_attempt_reserve > available:
                print("Paused at budget/available-credit boundary", flush=True)
                return status("paused_budget")
            body = control.build_control_openrouter_body(row, route)
            write_json(pending_path, {"custom_id": row["custom_id"], "run_dir": str(run_dir),
                       "max_attempt_reserve_usd": max_attempt_reserve,
                       "request_body_sha256": hashlib.sha256(control._json_bytes(body)).hexdigest()})
            submitter.session.attempts = 0
            status("running")
            started = time.perf_counter()
            try:
                response, headers, raw = submitter._post_openrouter_with_retries(
                    api_key, body, custom_id=row["custom_id"], subject_id=SUBJECT,
                    provider=route["pinned_provider"])
                record = control.build_control_openrouter_run_record(
                    row=row, body=body, response=response, pinned_provider=route["pinned_provider"],
                    latency_sec=time.perf_counter() - started, response_headers=headers,
                    model_metadata_snapshot=snapshot, raw_http_capture=raw)
                cost, basis = response_cost(response, row, prices)
                retries = max(0, submitter.session.attempts - 1)
                accounted = cost + retries * reserve
                record["execution_budget"] = {
                    "cost_basis": basis, "response_cost_or_reserve_usd": cost,
                    "unknown_retry_cost_reserve_usd": retries * reserve,
                    "accounted_cost_usd": accounted, "runtime_cost_cap_usd": cap,
                }
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                pending_path.unlink()
            except BaseException:
                status("stopped_pending_reconciliation")
                raise
            count += 1
            spent += accounted
            new_spend += accounted
            status("running")
            print(f"Claude {count}/{len(rows)} saved; accounted total ${spent:.4f}/{cap:.2f}", flush=True)
    return status("completed")


def select_credential(source, *, env_path=None):
    if source == "project":
        from dotenv import dotenv_values

        path = env_path if env_path is not None else REPO / ".env"
        key = dotenv_values(path, interpolate=False).get(Config.OPENROUTER_AUTH_ENV) if path.exists() else None
        label = "project .env"
    else:
        key = os.environ.get(Config.OPENROUTER_AUTH_ENV)
        label = "loaded environment"
    if not key or not key.strip():
        raise ValueError(f"OPENROUTER_API_KEY is missing from {label}; no alternate key was selected")
    if key != key.strip():
        raise ValueError(f"OPENROUTER_API_KEY in {label} has leading/trailing whitespace")
    return key, label


def account_available(session, key, *, source):
    headers = {"Authorization": f"Bearer {key}"}
    data = {}
    for name in ("key", "credits"):
        response = session.get(f"{Config.OPENROUTER_BASE_URL}/{name}", headers=headers, timeout=60)
        if response.status_code in (401, 403):
            if name == "key":
                raise ValueError(
                    f"OpenRouter rejected the key from {source} at /key "
                    f"(HTTP {response.status_code}). Check that credential; no model requests sent."
                )
            raise ValueError(
                f"OpenRouter authenticated the key from {source}, but denied /credits "
                f"(HTTP {response.status_code}); account balance is unverified. No model requests sent."
            )
        response.raise_for_status()
        data[name] = response.json()["data"]
    credits = number(data["credits"]["total_credits"]) - number(data["credits"]["total_usage"])
    key_remaining = data["key"].get("limit_remaining")
    available = credits if key_remaining is None else min(credits, number(key_remaining))
    return max(0.0, available)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Send the remaining frozen requests")
    mode.add_argument("--preflight-only", action="store_true", help="Check authentication, balance and model metadata without sending model requests")
    parser.add_argument("--key-source", choices=("project", "environment"), default="project",
                        help="Credential source; default project .env ignores inherited key overrides")
    args = parser.parse_args()
    os.chdir(REPO)
    with run_lock(ROOT):
        manifest, rows, prior, protocol, exceptions = validate_inputs()
        saved = completed_records(ROOT, rows)
        historical = json.loads(HISTORICAL_METADATA.read_text())["matched_model"]["pricing"]
        prior_spend = sum(response_cost(r, r, historical)[0] for r in prior)
        route = {"slug": manifest["route_lock"]["model_slug"], "pinned_provider": "anthropic",
                 "decoding": manifest["route_lock"]["decoding"], "quantizations": []}
        cap = number(manifest["budget"]["historical_subject_cap_usd"])
        print(f"Verified 952-request queue; {len(saved)} saved, {952-len(saved)} remaining. "
              f"Historical reported cost plus reserve: ${prior_spend:.6f}; cap ${cap:.2f}.", flush=True)
        if not args.execute and not args.preflight_only:
            print("DRY RUN: no network requests. Add --execute to collect.")
            return 0
        if len(saved) == len(rows):
            print("All 952 outcomes already saved; no requests needed.")
            return 0
        key, key_source = select_credential(args.key_source)
        print(f"OpenRouter credential source: {key_source}", flush=True)
        session = CountingSession()
        available = account_available(session, key, source=key_source)
        snapshot_dir = ROOT / "metadata" / uuid.uuid4().hex
        snapshot = write_openrouter_model_metadata_snapshot(
            session=session, api_key=key, output_dir=snapshot_dir,
            model_slug=route["slug"], pinned_provider=route["pinned_provider"])
        prices = json.loads(Path(snapshot["path"]).read_text())["matched_model"]["pricing"]
        if args.preflight_only:
            print(f"Authentication, credit and model checks passed; available ${available:.4f}. No model requests sent.")
            return 0
        submitter = NativeBatchSubmitter(session=session, output_dir=ROOT,
                                        confirm_openrouter_fallback=lambda *_: False)
        receipt = {"owner_instruction": manifest["owner_instruction"],
                   "resume_manifest_sha256": sha(ROOT / "resume_manifest.json"),
                   "launcher_sha256": sha(Path(__file__)),
                   "historical_metadata_sha256": sha(HISTORICAL_METADATA),
                   "non_dispatch_contract_exceptions": exceptions,
                   "credential_source": key_source,
                   "account_available_at_start_usd": available,
                   "protocol_id": protocol["protocol_id"]}
        result = collect(root=ROOT, rows=rows, prior_spend=prior_spend, route=route,
                         prices=prices, api_key=key, submitter=submitter, snapshot=snapshot,
                         cap=cap, available=available, receipt=receipt)
        print(json.dumps(result, indent=2), flush=True)
        return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"STOPPED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
