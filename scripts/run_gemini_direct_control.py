"""Isolated, prospectively locked Gemini direct-Google exploratory collector."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from datetime import datetime, timezone

import requests
import jsonschema

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_disabled_format_control as control
from scripts import schema_visibility_preflight as probe
from scripts.fetch_native_batch_results import _native_run_record

ROOT = REPO / 'outputs/postreg_disabled_format_control_v3_gemini_google_20261005'
PILOT = REPO / 'outputs/pilot/2026-09-21/all_remaining_hosted_routes/20260922T032059Z'
SID = 'gemini-3.5-flash'
ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'
WORKERS = 12
CAP = 100.0
ACCEPTED = {SID, 'models/' + SID, 'gemini-3.5-flash-20260519', 'gemini-3.5-flash-05-2026'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(REPO))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def unique_rows(path):
    rows = read_rows(path)
    found = {r['custom_id']: r for r in rows}
    if len(found) != len(rows):
        raise ValueError('Duplicate response identities')
    return found


def request_body(row):
    body = copy.deepcopy(row['body'])
    expected = {'model': SID, 'temperature': 0, 'top_p': 1, 'max_tokens': 4096}
    for key, value in expected.items():
        if body.get(key) != value:
            raise ValueError(f'Frozen decoding/model mismatch: {key}')
    allowed = set(expected) | {'messages', 'tools', 'tool_choice'}
    if set(body) - allowed:
        raise ValueError(f'Unexpected request fields: {set(body) - allowed}')
    arm = row.get('control_arm')
    if arm and body.get('tool_choice') != ('auto' if arm == 'TOOL_CONTEMPORANEOUS' else 'none'):
        raise ValueError('Arm tool-choice mismatch')
    return body


def operational_error(row, response):
    if response.get('model') not in ACCEPTED:
        return 'returned model identity drift'
    choices = response.get('choices') or []
    if not choices:
        return None  # Empty/refused outcomes remain observations.
    calls = (choices[0].get('message') or {}).get('tool_calls') or []
    if row['body'].get('tool_choice') == 'none' and calls:
        return 'native tool calls emitted in disabled arm'
    return None


def ceiling(body):
    return (len(control._json_bytes(body)) * 1.5 + 4096 * 9) / 1_000_000


def measured_cost(response, fallback):
    u = response.get('usage') or {}
    if isinstance(u.get('prompt_tokens'), int) and isinstance(u.get('completion_tokens'), int):
        return (u['prompt_tokens'] * 1.5 + u['completion_tokens'] * 9) / 1_000_000
    return fallback


class Budget:
    def __init__(self, path, cap):
        self.path, self.cap = Path(path), cap
        self.lock = threading.Lock()
        self.reservations, self.settled = {}, {}
        for item in read_rows(path):
            target = self.reservations if item['event'] == 'reserve' else self.settled
            if item['call_id'] in target:
                raise RuntimeError('Duplicate budget event')
            target[item['call_id']] = item['usd']
        if set(self.reservations) != set(self.settled):
            raise RuntimeError('Unsettled request reservation: reconcile raw response before resume')

    @property
    def used(self):
        return sum(self.settled.values()) + sum(v for k, v in self.reservations.items() if k not in self.settled)

    def reserve(self, call_id, amount):
        with self.lock:
            if call_id in self.reservations or self.used + amount > self.cap:
                raise RuntimeError('Duplicate attempt or spending cap exceeded')
            append(self.path, {'at_utc': now(), 'event': 'reserve', 'call_id': call_id, 'usd': amount})
            self.reservations[call_id] = amount

    def settle(self, call_id, amount):
        with self.lock:
            if call_id not in self.reservations or call_id in self.settled:
                raise RuntimeError('Invalid settlement')
            append(self.path, {'at_utc': now(), 'event': 'settle', 'call_id': call_id, 'usd': amount})
            self.settled[call_id] = amount


def verify_lock():
    lock = json.loads((ROOT / 'execution_lock.json').read_text())
    for filename, expected in lock['files_sha256'].items():
        if sha(REPO / filename) != expected:
            raise RuntimeError(f'Frozen artifact hash mismatch: {filename}')
    control.load_protocol(ROOT / 'protocol.json')
    return lock


def prepare():
    if (ROOT / 'execution_lock.json').exists():
        raise RuntimeError('Run already frozen')
    ROOT.mkdir(exist_ok=True)
    for src, name in [('/tmp/cagap_google_docs_20261005.json', 'google_docs.json'),
                      ('/tmp/cagap_google_pricing_full_20261005.json', 'google_pricing_docs.json'),
                      ('/tmp/cagap_google_model_20261005.json', 'google_model.json')]:
        shutil.copyfile(src, ROOT / name)
    oldplan = json.loads((PILOT / 'plan.json').read_text())
    route = next(x for x in oldplan['routes'] if x['candidate_id'] == 'gemini-3.5-flash__google_native_openai_compatible')
    assert route['url'] == ENDPOINT
    save(ROOT / 'diagnostic_requests.json', route['requests'][:6])
    save(ROOT / 'route_evidence.json', {
        'at_utc': now(), 'historical_screen': rel(PILOT / 'FINAL_QUALIFICATION_REVIEW.json'),
        'historical_screen_sha256': sha(PILOT / 'FINAL_QUALIFICATION_REVIEW.json'),
        'selected_route': {k: v for k, v in route.items() if k != 'requests'},
        'qualification_amendment': 'Operational eligibility requires schema visibility and disabled native calling, not FORMAT compliance. Prior invalid serialization remains observed evidence. No outcome repair/filtering/retry.',
        'fresh_six_mode_and_three_canary_gate_required': True,
    })
    source = REPO / 'experiments/disabled_format_control/protocol_v3_draft.json'
    p = json.loads(source.read_text())
    for k in ['planned_supersedes_protocol_id', 'supersession_condition', 'implementation_contract_pending_freeze', 'promotion_gates']:
        p.pop(k, None)
    p.update(protocol_id='CAGAP-POSTREG-DISABLED-FORMAT-2026-10-05-v3-GEMINI-GOOGLE', status='frozen_before_first_dispatch',
             created_at=now(), frozen_at=now(), registration_scope='Post-registration exploratory; isolated from confirmatory H1/H2.',
             rationale='Owner requests direct Google completion after historical FORMAT qualification failure. FORMAT compliance is observed, not an inclusion gate. All arms and outcomes retained.')
    p['source_protocol'] = {'path': rel(source), 'sha256': sha(source), 'historical_artifact_unchanged': True}
    p['design']['partition_namespace'] = ROOT.name
    p['population'].update(subjects=[SID], expected_analysis_sets=1152, expected_cells_per_arm=1152, expected_rows_total=3456, expected_canary_rows_total=3)
    p['source_batch_sha256'] = {f'input_{SID}.jsonl': p['source_batch_sha256'][f'input_{SID}.jsonl']}
    p['artifact_paths'] = {k: rel(ROOT / v) for k, v in {
        'full_batches_dir': 'batches', 'canary_batches_dir': 'canary/batches', 'locked_inputs_manifest': 'locked_inputs_manifest.json',
        'execution_authorization': 'execution_authorization.json', 'responses_root': 'responses', 'canary_responses_root': 'canary/responses', 'analysis_root': 'analysis'}.items()}
    p['route_policy'] = {'transport': 'Google direct synchronous OpenAI-compatible API; batch paths are generator metadata only', 'allow_fallbacks': False,
        'no_substitution_without_amendment': True, 'routes': {SID: {
            'route': 'native', 'model_slug': SID, 'provider_only': ['google'], 'quantizations': [], 'serving_precision': 'provider_managed',
            'decoding': {'temperature': 0, 'top_p': 1, 'max_tokens': 4096}, 'endpoint': 'https://generativelanguage.googleapis.com',
            'request_path': '/v1beta/openai/chat/completions', 'batch_url_path': '/v1beta/openai/batches', 'batch_input_files_path': '/upload/v1beta/files',
            'request_shape': 'openai_compatible_batch', 'execution_mode': 'synchronous',
            'preflight_evidence_id': rel(ROOT / 'route_evidence.json'), 'preflight_evidence_sha256': sha(ROOT / 'route_evidence.json'),
            'disabled_schema_retention': 'schema_visible', 'tool_native_call_support': 'supported', 'provider_pin_verified': True,
            'accepted_returned_models': sorted(ACCEPTED)}}}
    p['analysis']['expected_analysis_sets'] = 1152
    p['analysis']['serialization_amendment'] = 'Report all-population bare JSON and declared name/arguments compliance in both disabled arms, paired transitions and exact text equality. No outcome-conditioned analysis as primary; residual TOOL–FORMAT contrast alone does not identify a serialization mechanism.'
    p['stopping_and_retry'].update(stopping_rule='Fixed3456 census after6 excluded diagnostics and3 excluded study canaries; no content-dependent reruns.',
        retry_rule='Only explicit HTTP429/500/502/503/504 may retry, maximum3 attempts with2/4 second waits. HTTP200 never retried. Ambiguous transport exceptions halt and reserve worst-case cost.',
        dispatch_failure_rule='Stop on access errors, model drift, native calls in disabled arm or budget cap; drain and preserve in-flight results; no fallback.',
        full_parallelism='12 workers, block_sequence_index modulo12; preserve each six-row randomized block and worker order. Append/fsync before next request.')
    # Preserve the exact current instruments already used by completed hosted grading.
    historical_lock = json.loads((REPO / 'outputs/postreg_disabled_format_control_hosted_grading_20261002/grading_lock.json').read_text())['files_sha256']
    for filename in p['grading_contract_sha256']:
        current = sha(REPO / filename)
        if filename in historical_lock and current != historical_lock[filename]:
            raise RuntimeError(f'Grading instrument drift: {filename}')
        p['grading_contract_sha256'][filename] = current
    p['implementation_contract_sha256'] = {x: sha(REPO / x) for x in ['scripts/run_gemini_direct_control.py', 'scripts/run_disabled_format_control.py', 'scripts/fetch_native_batch_results.py', 'scripts/schema_visibility_preflight.py']}
    p['execution_order'] = [SID]
    save(ROOT / 'protocol.json', p)
    save(ROOT / 'pricing.json', {'pricing': [{'model_key': SID, 'regular_api_input_usd_per_1m_tokens': 1.5, 'regular_api_output_usd_per_1m_tokens': 9.0}], 'source': 'https://ai.google.dev/gemini-api/docs/pricing', 'checked_at_utc': now(), 'batch_discount_used': False})
    control.NATIVE_PRICING_PATH = ROOT / 'pricing.json'
    protocol, psha = control.load_protocol(ROOT / 'protocol.json')
    manifests = {}
    for profile in ['canary', 'full']:
        dest = ROOT / ('canary/batches' if profile == 'canary' else 'batches')
        manifests[profile] = control.generate(ROOT / 'protocol.json', control.DEFAULT_SOURCE_DIR, dest, None, canary=profile == 'canary', write=True)
        generated = control.build_subject_rows(protocol, psha, control.DEFAULT_SOURCE_DIR, SID, canary=profile == 'canary', gpt_native={})
        assert b''.join(control._json_bytes(x) + b'\n' for x in generated) == (dest / f'input_{SID}.jsonl').read_bytes()
        for r in generated:
            assert request_body(r) == r['body']
    save(ROOT / 'locked_inputs_manifest.json', {'schema_version': 'disabled_format_control.locked_inputs.v1', 'status': 'frozen_before_first_official_dispatch',
        'locked_at_utc': now(), 'protocol_path': rel(ROOT / 'protocol.json'), 'protocol_id': p['protocol_id'], 'protocol_sha256': psha,
        'canary': manifests['canary'], 'full': manifests['full'], 'determinism': {'independently_regenerated_byte_identical': True}})
    (ROOT / 'OWNER_AUTHORIZATION.md').write_text('# Gemini exploratory execution authorization\n\nRecorded ' + now() + '.\n\nExact project-owner instruction: “Use the google ai studio/ direct google routes and finish the Gemini part now.”\n\nThis authorizes an isolated direct-Google successor after the explained historical FORMAT blocker. FORMAT noncompliance is retained as an outcome under the analogous local-cohort amendment; no instruction changes, filtering or outcome-dependent retries. All three arms are collected contemporaneously. The operational ceiling is $100 for Google subject collection and $25 for subsequent grading; existing trial-key limits remain binding. These are ceilings, not intended spend or permission for top-ups. New OSF filing is not required for this post-registration exploratory control under constitution2.0.0. No external publication.\n')
    save(ROOT / 'execution_authorization.json', {'created_at_utc': now(), 'submission_permitted': True, 'authorization_type': 'project_owner_exploratory',
        'exact_user_instruction': 'Use the google ai studio/ direct google routes and finish the Gemini part now.', 'protocol_sha256': psha,
        'locked_inputs_sha256': sha(ROOT / 'locked_inputs_manifest.json'), 'owner_document_sha256': sha(ROOT / 'OWNER_AUTHORIZATION.md'),
        'endpoint': ENDPOINT, 'model': SID, 'google_cap_usd': CAP, 'grading_cap_usd': 25, 'full_gate': 'Six diagnostic operational checks and3 study canaries; FORMAT compliance is not eligibility', 'response_count_at_freeze': 0})
    frozen = list(ROOT.glob('*.json')) + [ROOT / 'OWNER_AUTHORIZATION.md', Path(__file__), REPO / 'tests/test_gemini_direct_control.py']
    frozen += list((ROOT / 'batches').glob('*')) + list((ROOT / 'canary/batches').glob('*'))
    frozen += list((REPO / 'specs/002-gemini-direct-control').glob('*.md'))
    frozen = [x for x in frozen if x.is_file() and x.name != 'tasks.md']
    save(ROOT / 'execution_lock.json', {'created_at_utc': now(), 'files_sha256': {rel(x): sha(x) for x in frozen}})
    verify_lock()
    print(json.dumps({'prepared': True, 'full_rows': 3456, 'cap_usd': CAP, 'conservative_full_maximum_usd': manifests['full']['estimated_cost_upper_bound']['total_api_usd']}), flush=True)


def call(row, profile, budget, key):
    body = request_body(row)
    cid = row['custom_id']
    folder = ROOT / 'raw_http_archive' / profile / hashlib.sha256(cid.encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(1, 4):
        call_id = f'{profile}/{cid}/{attempt}'
        budget.reserve(call_id, ceiling(body))
        save(folder / f'{attempt}.request.json', body)
        start = time.monotonic()
        try:
            response = requests.post(ENDPOINT, headers={'Authorization': 'Bearer ' + key}, json=body, timeout=(15, 180))
        except requests.RequestException as exc:
            save(folder / f'{attempt}.transport_error.json', {'at_utc': now(), 'type': type(exc).__name__, 'ambiguous_delivery': True})
            budget.settle(call_id, ceiling(body))
            raise RuntimeError(f'Ambiguous delivery retained at {rel(folder)}; do not automatically resubmit') from None
        rawpath = folder / f'{attempt}.response.body'
        with rawpath.open('wb') as f:
            f.write(response.content); f.flush(); os.fsync(f.fileno())
        try:
            payload = response.json()
        except ValueError:
            payload = {'error': {'message': 'Non-JSON HTTP response archived'}}
        cost = measured_cost(payload, ceiling(body) if response.status_code == 200 else 0)
        budget.settle(call_id, cost)
        save(folder / f'{attempt}.metadata.json', {'at_utc': now(), 'http_status': response.status_code, 'response_sha256': sha(rawpath), 'request_sha256': control._sha256_json(body), 'cost_usd': cost, 'latency_sec': time.monotonic() - start})
        if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
            time.sleep(2 ** attempt)
            continue
        error = f'HTTP{response.status_code}' if response.status_code != 200 else operational_error(row, payload)
        break
    provider_result = {'custom_id': cid, 'response': {'status_code': response.status_code, 'body': payload}}
    record = _native_run_record(request_row=row, provider_result=provider_result, raw_response=payload, error=error,
        result_status='failed' if error else 'succeeded', subject_id=SID, provider='google', provider_batch_id='direct-sync-no-batch')
    record.update(native_transport='synchronous', native_endpoint=ENDPOINT, provider_batch_id=None,
        raw_http_capture={'path': rel(rawpath), 'sha256': sha(rawpath)}, request_transmitted_sha256=control._sha256_json(body), actual_cost_usd=cost)
    return record, error


def diagnostics(budget, key):
    requests_ = json.loads((ROOT / 'diagnostic_requests.json').read_text())
    path = ROOT / 'diagnostics/responses.jsonl'
    existing = unique_rows(path)
    observations = []
    for i, item in enumerate(requests_):
        cid = f'diagnostic_{i:02}_{item["mode"]}'
        row = {'custom_id': cid, 'subject_id': SID, 'body': item['body']}
        if cid in existing:
            record = existing[cid]
        else:
            record, error = call(row, 'diagnostics', budget, key)
            append(path, record)
            if error:
                raise RuntimeError(f'Diagnostic operational failure: {error}')
        if record.get('error'):
            raise RuntimeError('Prior diagnostic failure retained')
        parsed = probe.parse_openai(record['raw_response'])
        parsed.update(subject_id=SID, route_label='google-direct', mode=item['mode'], prompt_case_id=item['prompt_case_id'])
        observations.append(parsed)
    marker = next(x for x in observations if x['mode'] == 'CANARY_DISABLED')
    tool = next(x for x in observations if x['mode'] == 'TOOL')
    tool_request = next(x for x in requests_ if x['mode'] == 'TOOL')
    schemas = {t['function']['name']: t['function']['parameters'] for t in tool_request['body']['tools']}
    for native_call in tool.get('tool_calls') or []:
        fn = native_call['function']
        if fn['name'] not in schemas:
            raise RuntimeError('Diagnostic native tool name not declared')
        jsonschema.validate(json.loads(fn['arguments']), schemas[fn['name']])
    passed = probe.canary_marker() in str(marker.get('content')) and bool(tool.get('tool_calls'))
    review = {'at_utc': now(), 'passed': passed, 'schema_marker_visible': probe.canary_marker() in str(marker.get('content')), 'tool_native_calls_present': bool(tool.get('tool_calls')), 'responses_sha256': sha(path), 'modes': [x['mode'] for x in observations], 'format_compliance_is_outcome_not_gate': True}
    save(ROOT / 'diagnostic_review.json', review)
    print(json.dumps(review), flush=True)
    if not passed:
        raise RuntimeError('Fresh schema visibility/native callability gate failed')


def collect(profile, budget, key):
    diagnostic = json.loads((ROOT / 'diagnostic_review.json').read_text())
    assert diagnostic['passed'] and diagnostic['responses_sha256'] == sha(ROOT / 'diagnostics/responses.jsonl')
    if profile == 'full':
        review = json.loads((ROOT / 'canary_review.json').read_text())
        assert review['passed'] and review['responses_sha256'] == sha(ROOT / 'canary/responses' / f'responses_{SID}_native.jsonl')
    base = ROOT / 'canary' if profile == 'canary' else ROOT
    expected = read_rows(base / 'batches' / f'input_{SID}.jsonl')
    out = base / 'responses' / f'responses_{SID}_native.jsonl'
    existing = unique_rows(out)
    assert set(existing) <= {r['custom_id'] for r in expected}
    if any(x.get('error') for x in existing.values()):
        raise RuntimeError('Prior operational failure retained; requires reconciliation')
    write_lock, stop = threading.Lock(), threading.Event()
    def shard(index):
        for row in expected:
            if profile == 'full' and int(row['dispatch_block_sequence_index']) % WORKERS != index:
                continue
            if row['custom_id'] in existing or stop.is_set():
                continue
            try:
                record, error = call(row, profile, budget, key)
                with write_lock:
                    append(out, record)
                    existing[row['custom_id']] = record
                    if len(existing) % 48 == 0 or profile == 'canary':
                        print(json.dumps({'at_utc': now(), 'profile': profile, 'done': len(existing), 'expected': len(expected), 'spent_usd': round(budget.used, 4)}), flush=True)
                if error:
                    raise RuntimeError(error)
            except Exception:
                stop.set()
                raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS if profile == 'full' else 1) as pool:
        futures = [pool.submit(shard, i) for i in range(WORKERS if profile == 'full' else 1)]
        failures = []
        for f in concurrent.futures.as_completed(futures):
            try: f.result()
            except Exception as exc: failures.append(str(exc))
    if failures:
        save(ROOT / f'{profile}_halt.json', {'at_utc': now(), 'failures': failures, 'retained_rows': len(existing)})
        raise RuntimeError('; '.join(failures))
    assert set(existing) == {r['custom_id'] for r in expected}
    review = {'at_utc': now(), 'passed': True, 'rows': len(existing), 'responses_sha256': sha(out), 'protocol_sha256': sha(ROOT / 'protocol.json'), 'execution_lock_sha256': sha(ROOT / 'execution_lock.json'), 'spent_usd': budget.used, 'format_compliance_is_outcome_not_gate': True}
    save(ROOT / f'{profile}_review.json', review)
    print(json.dumps(review), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'diagnostics', 'canary', 'full', 'verify'])
    args = parser.parse_args()
    os.chdir(REPO)
    if args.stage == 'prepare':
        prepare(); return
    verify_lock()
    if args.stage == 'verify':
        print(json.dumps({'locks_valid': True})); return
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        raise RuntimeError('GEMINI_API_KEY missing from environment')
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = Budget(ROOT / 'calls_ledger.jsonl', CAP)
        if args.stage == 'diagnostics': diagnostics(budget, key)
        else: collect(args.stage, budget, key)


if __name__ == '__main__':
    main()
