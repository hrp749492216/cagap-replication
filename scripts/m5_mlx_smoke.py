#!/usr/bin/env python3
"""M5 MLX smoke runner per Section 12.H pre-dispatch gate.

Brings up mlx_lm.server for one (subject, precision_value, model_path) combo,
dispatches 12 cells (3 prompts x 4 modes from the regenerated pilot batch),
saves outputs/pilot/2026-05-07/smoke_validation/<subject>_<format>_m5_macbook_128gb.json,
kills the server. Sequential one-at-a-time use — caller invokes once per combo.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

try:
    from scripts.smoke_gate import evaluate_correctness_smoke_gate
except ModuleNotFoundError:  # direct file-path execution on local/cloud runners
    from smoke_gate import evaluate_correctness_smoke_gate


def start_server(model_path: str, port: int = 8081, log_path: Path | None = None) -> subprocess.Popen:
    log_path = log_path or Path('logs/mlx') / f'serve_{model_path.split("/")[-1]}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.path.expanduser('~/.local/bin/mlx_lm.server'),
        '--model', model_path,
        '--host', '127.0.0.1',
        '--port', str(port),
        '--log-level', 'INFO',
    ]
    fh = open(log_path, 'wb')
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc


def start_proxy(backend_port: int, public_port: int = 8080,
                log_path: Path | None = None) -> subprocess.Popen:
    log_path = log_path or Path('logs/mlx') / f'proxy_{public_port}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        'scripts/mlx_tool_choice_proxy.py',
        '--backend', f'http://127.0.0.1:{backend_port}',
        '--host', '127.0.0.1',
        '--port', str(public_port),
    ]
    fh = open(log_path, 'wb')
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc


def wait_ready(port: int, model_path: str, timeout: int = 600) -> bool:
    url = f'http://127.0.0.1:{port}/v1/models'
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def pick_smoke_cells(batch_path: Path, role: str, n: int = 12) -> list[dict]:
    rows = [json.loads(l) for l in open(batch_path) if l.strip()]
    matching = [r for r in rows if r.get('serving_precision_role') == role]
    if not matching:
        matching = rows
    selected = []
    mode_caps = {}
    for r in matching:
        m = r.get('mode')
        if mode_caps.get(m, 0) < 3:
            selected.append(r)
            mode_caps[m] = mode_caps.get(m, 0) + 1
        if len(selected) >= n:
            break
    return selected


def dispatch_cells(cells: list[dict], port: int, mlx_model: str, max_tokens: int = 1024) -> list[dict]:
    url = f'http://127.0.0.1:{port}/v1/chat/completions'
    out = []
    for i, row in enumerate(cells, 1):
        body = dict(row['body'])
        body['model'] = mlx_model
        body['max_tokens'] = min(body.get('max_tokens', 256), max_tokens)
        t0 = time.monotonic()
        try:
            r = requests.post(url, json=body, timeout=600)
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


def kill_server(proc: subprocess.Popen, port: int = 8080):
    try:
        proc.terminate()
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    # also free the port if anything else is hanging on it
    subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # M5/MLX fallback is predeclared only for the Gemma and Qwen local
    # subjects. Llama, GLM, and Mistral local smokes use cloud-vLLM paths.
    ap.add_argument('--subject', required=True, choices=['gemma-4-31b-it', 'qwen3.6-35b-a3b'])
    ap.add_argument('--mlx-model', required=True, help='HF repo (e.g. mlx-community/gemma-4-31B-it-bf16)')
    ap.add_argument('--precision-value', required=True, help='e.g. bf16 | int4_mlx')
    ap.add_argument('--precision-role', required=True, choices=['primary', 'quantization_audit'])
    ap.add_argument('--port', type=int, default=8080,
                    help='public-facing port (proxy). dispatcher hits this.')
    ap.add_argument('--backend-port', type=int, default=8081,
                    help='mlx_lm.server backend port behind the proxy.')
    ap.add_argument('--no-proxy', action='store_true',
                    help='skip the tool_choice=none enforcement proxy '
                         '(backwards-compat for raw mlx_lm.server smokes).')
    ap.add_argument('--cells', type=int, default=12)
    ap.add_argument('--batch-path', default=None,
                    help='input batch jsonl; auto-derived if omitted')
    ap.add_argument('--baseline-s-per-cell', type=float, default=None,
                    help='B300 baseline s/cell from Pilot-1a for ratio reporting')
    args = ap.parse_args()

    batch_path = Path(args.batch_path) if args.batch_path else Path(
        f'outputs/pilot/2026-05-07/batches/input_{args.subject}.jsonl'
    )
    if not batch_path.exists():
        print(f"batch missing: {batch_path}", file=sys.stderr)
        return 2

    cells = pick_smoke_cells(batch_path, role=args.precision_role, n=args.cells)
    print(f"smoke {args.subject} | {args.precision_value} | mlx={args.mlx_model}")
    print(f"  selected {len(cells)} cells; mode dist: ", end='')
    mc = {}
    for c in cells:
        mc[c.get('mode')] = mc.get(c.get('mode'), 0) + 1
    print(mc)

    if args.no_proxy:
        backend_proc = start_server(args.mlx_model, port=args.port)
        proxy_proc = None
        print(f"  mlx_lm.server pid={backend_proc.pid} (no-proxy mode)")
        if not wait_ready(args.port, args.mlx_model, timeout=600):
            print("  ERROR: server never came ready", file=sys.stderr)
            kill_server(backend_proc, args.port)
            return 3
    else:
        backend_proc = start_server(args.mlx_model, port=args.backend_port)
        print(f"  mlx_lm.server pid={backend_proc.pid} on backend :{args.backend_port}")
        if not wait_ready(args.backend_port, args.mlx_model, timeout=600):
            print("  ERROR: backend server never came ready", file=sys.stderr)
            kill_server(backend_proc, args.backend_port)
            return 3
        proxy_proc = start_proxy(args.backend_port, public_port=args.port)
        print(f"  mlx_tool_choice_proxy pid={proxy_proc.pid} on :{args.port}")
        # proxy comes up instantly; do a quick health check
        if not wait_ready(args.port, args.mlx_model, timeout=20):
            print("  ERROR: proxy did not come ready", file=sys.stderr)
            kill_server(proxy_proc, args.port)
            kill_server(backend_proc, args.backend_port)
            return 3
    print("  ready; dispatching via proxy" if not args.no_proxy else "  ready; dispatching direct")
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.monotonic()
    records = dispatch_cells(cells, port=args.port, mlx_model=args.mlx_model)
    total = time.monotonic() - t0
    finished_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    if proxy_proc is not None:
        kill_server(proxy_proc, args.port)
    kill_server(backend_proc, args.backend_port if not args.no_proxy else args.port)
    print("  server(s) killed")

    ok = [r for r in records if r.get('http_status') == 200]
    mean = sum(r['wall_seconds'] for r in ok) / len(ok) if ok else None
    finishes = {fr: sum(1 for r in ok if r.get('finish') == fr) for fr in {r.get('finish') for r in ok}}
    out = {
        'subject_id': args.subject,
        'precision_role': args.precision_role,
        'precision_value': args.precision_value,
        'served_model_path': args.mlx_model,
        'serving_substrate': 'm5_macbook_128gb',
        'serving_runtime': 'mlx_lm' + (' + tool_choice_none_proxy' if not args.no_proxy else ''),
        'cells': len(cells),
        'ok': len(ok),
        'wall_seconds': round(total, 2),
        'mean_seconds_per_cell': round(mean, 3) if mean else None,
        'b300_baseline_s_per_cell': args.baseline_s_per_cell,
        'm5_to_b300_ratio': round(mean / args.baseline_s_per_cell, 2) if mean and args.baseline_s_per_cell else None,
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
    out_dir = Path('outputs/pilot/2026-05-07/smoke_validation')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.subject}_{args.precision_value}_m5_macbook_128gb.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  saved: {out_path}")
    print(f"  ok={len(ok)}/{len(cells)} mean={mean}  verdict={out['verdict']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
