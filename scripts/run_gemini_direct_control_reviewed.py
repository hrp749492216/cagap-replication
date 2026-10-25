"""Reviewed operational corrections; original collector/receipts stay frozen.

Schema visibility uses the existing protocol's input-token classifier, not
obedience to the ancillary marker request. Google compatibility usage can omit
reasoning tokens from completion_tokens; bill total-minus-prompt when larger.
"""
import json
import sys
import fcntl
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import run_gemini_direct_control as g


def measured_cost(response, fallback):
    usage = response.get('usage') or {}
    prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return fallback
    total = usage.get('total_tokens')
    output = max(completion, total - prompt) if isinstance(total, int) else completion
    return (prompt * 1.5 + output * 9) / 1_000_000


def visibility(observations):
    return g.probe.classify(observations)['google-direct']['disabled_arm_schema_visibility']


def review():
    g.verify_lock()
    root = g.ROOT
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'operational_review_patch.json').exists():
            raise RuntimeError('Review already recorded')
        requests_ = json.loads((root / 'diagnostic_requests.json').read_text())
        rows = g.read_rows(root / 'diagnostics/responses.jsonl')
        assert len(rows) == len(requests_) == 6
        observations = []
        delta = 0
        for row, item in zip(rows, requests_, strict=True):
            assert not row.get('error')
            obs = g.probe.parse_openai(row['raw_response'])
            obs.update(route_label='google-direct', mode=item['mode'], prompt_case_id=item['prompt_case_id'], http_status=200)
            observations.append(obs)
            delta += measured_cost(row['raw_response'], 0) - row['actual_cost_usd']
        assert visibility(observations) == 'schema_visible'
        tool = next(x for x in observations if x['mode'] == 'TOOL')
        assert tool['tool_calls']
        # Native schemas were validated by the original diagnostic collector.
        assert not any(x.get('tool_calls') for x in observations if x['mode'] in ['DISABLED', 'DISABLED_FORMAT_JSON', 'CANARY_DISABLED'])
        budget = g.Budget(root / 'calls_ledger.jsonl', g.CAP)
        if delta > 0:
            budget.reserve('diagnostics_reasoning_cost_reconciliation', delta)
            budget.settle('diagnostics_reasoning_cost_reconciliation', delta)
        old = root / 'diagnostic_review.json'
        old.rename(root / 'diagnostic_review_initial_marker_gate.json')
        g.save(old, {'at_utc': g.now(), 'passed': True, 'classification': g.probe.classify(observations),
            'schema_marker_visible': False, 'tool_native_calls_present': True,
            'responses_sha256': g.sha(root / 'diagnostics/responses.jsonl'), 'format_compliance_is_outcome_not_gate': True,
            'review_basis': 'Existing frozen schema_visibility_preflight.classify uses retained schema input-token delta; marker obedience is ancillary and not its gate.',
            'supersedes_review': 'diagnostic_review_initial_marker_gate.json', 'new_diagnostic_calls': 0})
        files = [Path(__file__), REPO / 'tests/test_gemini_direct_control_reviewed.py', root / 'diagnostic_review.json', root / 'diagnostic_review_initial_marker_gate.json']
        g.save(root / 'operational_review_patch.json', {'at_utc': g.now(), 'full_observations_at_patch': 0,
            'base_execution_lock_sha256': g.sha(root / 'execution_lock.json'),
            'scope': 'Align fresh eligibility review with existing frozen token-delta schema classifier; correct billing estimate to include reasoning. No model/request/outcome changes and no rerun.',
            'diagnostic_reasoning_cost_correction_usd': delta, 'files_sha256': {g.rel(p): g.sha(p) for p in files}})
        print(json.dumps({'schema_visibility': visibility(observations), 'native_callability': True, 'diagnostic_spent_usd': budget.used}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['review']:
        review()
    else:
        patch = json.loads((g.ROOT / 'operational_review_patch.json').read_text())
        assert patch['base_execution_lock_sha256'] == g.sha(g.ROOT / 'execution_lock.json')
        for path, expected in patch['files_sha256'].items():
            assert g.sha(REPO / path) == expected, path
        g.measured_cost = measured_cost
        g.main()
