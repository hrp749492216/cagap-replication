#!/usr/bin/env python3
"""Cloud-vLLM 12-cell smoke runner per Section 12.H pre-dispatch gate.

Mirror of scripts/m5_mlx_smoke.py for the cloud-vLLM substrate. Caller is
responsible for starting and stopping the vLLM server; this script only
selects 12 cells from the regenerated pilot batch, dispatches them, and
saves the smoke verdict at
``outputs/pilot/<date>/smoke_validation/<subject>_<format>_<substrate>.json``.

The cloud-vLLM substrate honors ``tool_choice="none"`` natively, so no
proxy is needed.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import requests

try:
    from scripts.smoke_gate import evaluate_correctness_smoke_gate
except ModuleNotFoundError:  # direct file-path execution on cloud instances
    from smoke_gate import evaluate_correctness_smoke_gate


def pick_smoke_cells(batch_path: Path, role: str, n: int = 12) -> list[dict]:
    rows = [json.loads(l) for l in open(batch_path) if l.strip()]
    matching = [r for r in rows if r.get('serving_precision_role') == role]
    if not matching:
        matching = rows
    selected = []
    mode_caps: dict[str, int] = {}
    for r in matching:
        m = r.get('mode')
        if mode_caps.get(m, 0) < 3:
            selected.append(r)
            mode_caps[m] = mode_caps.get(m, 0) + 1
        if len(selected) >= n:
            break
    return selected


def dispatch_cells(cells: list[dict], url: str, served_model: str,
                   api_key: str | None, max_tokens: int = 1024) -> list[dict]:
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    out = []
    for i, row in enumerate(cells, 1):
        body = dict(row['body'])
        body['model'] = served_model
        body['max_tokens'] = min(body.get('max_tokens', 256), max_tokens)
        t0 = time.monotonic()
        try:
            r = requests.post(url, json=body, headers=headers, timeout=600)
            wall = time.monotonic() - t0
            rec: dict = {
                'i': i,
                'custom_id': row['custom_id'],
                'mode': row.get('mode'),
                'persona': row.get('persona'),
                'wall_seconds': round(wall, 3),
                'http_status': r.status_code,
            }
            if r.status_code == 200:
                rr = r.json()
                ch = rr['choices'][0]
                msg = ch.get('message', {}) or {}
                rec['finish'] = ch.get('finish_reason')
                rec['has_tool_calls'] = bool(msg.get('tool_calls'))
                content = msg.get('content') or ''
                rec['content_len'] = len(content)
                rec['has_xml_tool'] = '<tool_call>' in content
            else:
                rec['error'] = r.text[:300]
            print(f"  {i:2}/{len(cells)} {rec.get('mode'):16} wall={wall:6.2f}s http={rec['http_status']} finish={rec.get('finish')} tc={rec.get('has_tool_calls')} clen={rec.get('content_len',0)}", flush=True)
            out.append(rec)
        except Exception as e:
            wall = time.monotonic() - t0
            out.append({'i': i, 'custom_id': row['custom_id'], 'wall_seconds': round(wall, 3), 'error': str(e)[:300]})
            print(f"  {i:2}/{len(cells)} ERR: {e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True,
                    choices=[
                        'llama-3.3-70b-instruct',
                        'gemma-4-31b-it',
                        'qwen3.6-35b-a3b',
                        'glm-4.7-flash',
                        'mistral-medium-3.5-128b',
                    ])
    ap.add_argument('--served-model', required=True,
                    help='string passed as model field (must match --served-model-name on vllm serve)')
    ap.add_argument('--precision-value', required=True,
                    help='e.g. bf16 | int4_awq | nvfp4')
    ap.add_argument('--precision-role', required=True,
                    choices=['primary', 'quantization_audit'])
    ap.add_argument('--substrate-tag', required=True,
                    help='e.g. external_gpu_rtxpro6000_blackwell, external_gpu_b300_sxm6')
    ap.add_argument('--endpoint', default='http://127.0.0.1:8000/v1/chat/completions')
    ap.add_argument('--api-key-env', default='VLLM_API_KEY',
                    help='env var holding bearer token; empty/unset = unauthenticated')
    ap.add_argument('--cells', type=int, default=12)
    ap.add_argument('--batch-path', default=None)
    ap.add_argument('--baseline-s-per-cell', type=float, default=None)
    ap.add_argument('--out-dir', default='outputs/pilot/2026-05-07/smoke_validation')
    args = ap.parse_args()

    import os
    api_key = os.environ.get(args.api_key_env) or None

    batch_path = Path(args.batch_path) if args.batch_path else Path(
        f'outputs/pilot/2026-05-07/batches/input_{args.subject}.jsonl'
    )
    if not batch_path.exists():
        print(f"batch missing: {batch_path}", file=sys.stderr)
        return 2

    cells = pick_smoke_cells(batch_path, role=args.precision_role, n=args.cells)
    print(f"smoke {args.subject} | {args.precision_value} | served={args.served_model}")
    print(f"  selected {len(cells)} cells; mode dist: ", end='')
    mc: dict[str, int] = {}
    for c in cells:
        mc[c.get('mode')] = mc.get(c.get('mode'), 0) + 1
    print(mc)

    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.monotonic()
    records = dispatch_cells(cells, url=args.endpoint, served_model=args.served_model,
                             api_key=api_key)
    total = time.monotonic() - t0
    finished_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    ok = [r for r in records if r.get('http_status') == 200]
    mean = sum(r['wall_seconds'] for r in ok) / len(ok) if ok else None
    finishes = {fr: sum(1 for r in ok if r.get('finish') == fr) for fr in {r.get('finish') for r in ok}}
    out = {
        'subject_id': args.subject,
        'precision_role': args.precision_role,
        'precision_value': args.precision_value,
        'served_model_path': args.served_model,
        'serving_substrate': args.substrate_tag,
        'serving_runtime': 'vllm_cloud',
        'cells': len(cells),
        'ok': len(ok),
        'wall_seconds': round(total, 2),
        'mean_seconds_per_cell': round(mean, 3) if mean else None,
        'b300_baseline_s_per_cell': args.baseline_s_per_cell,
        'm5_to_b300_ratio': None,
        'rtx_to_b300_ratio': round(mean / args.baseline_s_per_cell, 2) if mean and args.baseline_s_per_cell else None,
        'finish_reason_distribution': finishes,
        'tool_calls_populated_count': sum(1 for r in ok if r.get('has_tool_calls')),
        'xml_in_content_count': sum(1 for r in ok if r.get('has_xml_tool')),
        'mode_counts': mc,
        'records': records,
        'started_utc': started_utc,
        'finished_utc': finished_utc,
    }
    # Section 13.H smoke gate: HTTP success is necessary but not sufficient.
    # TOOL-mode rows must populate structured tool_calls. Latency is recorded
    # for operational planning only and does not determine PASS/FAIL.
    out.update(evaluate_correctness_smoke_gate(
        records,
        cells_total=len(cells),
        mean_seconds_per_cell=mean,
        baseline_s_per_cell=args.baseline_s_per_cell,
    ))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.subject}_{args.precision_value}_{args.substrate_tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {out_path}")
    print(f"  ok={len(ok)}/{len(cells)} mean={mean}  verdict={out['verdict']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
