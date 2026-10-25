"""Audited continuation after explicit no-generation spend-quota rejections."""
from __future__ import annotations
import fcntl, json, os, shutil, sys, threading, time
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_gemini_generate_content_control as g
REC=g.ROOT/'quota_recovery_01'
WINDOW=600.0
RATE_CAP=8.0

def window_spend(rows, at):
    reserves={r['call_id']:r for r in rows if r['event']=='reserve'}
    settles={r['call_id']:r for r in rows if r['event']=='settle'}
    total=0.0
    for cid,r in reserves.items():
        s=settles.get(cid)
        if s is None:total+=r['usd']
        elif at-datetime.fromisoformat(s['at_utc']).timestamp()<WINDOW:total+=s['usd']
    return total

class PacedBudget(g.Budget):
    def __init__(self,path,cap):
        super().__init__(path,cap)
        self.pace_lock=threading.Lock();self.next_call=0.0
    def reserve(self,cid,amount):
        with self.pace_lock:
            while True:
                with self.lock:
                    rows=g.read_rows(self.path)+g.read_rows(g.PRIOR_ROOT/'calls_ledger.jsonl')
                    spent=window_spend(rows,time.time())
                if spent+amount <= RATE_CAP:break
                time.sleep(5)
            time.sleep(max(0,self.next_call-time.monotonic()))
            super().reserve(cid,amount)
            self.next_call=time.monotonic()+1.0

def prepare():
    g.verify_lock();REC.mkdir(exist_ok=False)
    out=g.ROOT/'responses'/f'responses_{g.SID}_native.jsonl'
    snapshot=REC/'responses_before_resume.jsonl';shutil.copyfile(out,snapshot)
    shutil.copyfile(g.ROOT/'full_halt.json',REC/'full_halt.json')
    shutil.copyfile('/tmp/cagap_google_spend_limits_20261005.json',REC/'google_spend_limits.json')
    failed=[r for r in g.read_rows(snapshot) if r.get('error')]
    assert len(failed)==4
    for row in failed:
        rawpath=g.REPO/row['raw_http_capture']['path']
        assert g.sha(rawpath)==row['raw_http_capture']['sha256']
        raw=json.loads(rawpath.read_text())
        assert row['error']=='HTTP429' and raw['error']['code']==429
        assert 'spend-based rate limit' in raw['error']['message']
        assert not raw.get('candidates')
        g.append(REC/'rejected_rows.jsonl',row)
    g.save(REC/'amendment.json',{'at_utc':g.now(),'reason':'Explicit HTTP429 spend-based quota rejections; no generated candidates. Retry eligibility uses transport metadata only.',
        'successful_observations_retained':1346,'rejected_identities':[r['custom_id'] for r in failed],
        'retry_amendment':'One additional bounded cycle of at most3 attempts for the4 explicit rejected identities after cooldown, and original max3 for undispatched identities. No HTTP200 outcome retry.',
        'pacing':'At least1 second between all dispatches; cap$8 across rolling600seconds including prior-interface measured spend and reserved worst-case in-flight requests; retain12 fixed block streams.',
        'unchanged':'Model, endpoint, prompts, schemas, decoding, arm assignments, within-stream order, full3456population and total$100cap.',
        'archive':'Original1350row checkpoint and all raw attempts retained. Canonical response file retains successful1346rows before continuation; rejected rows archived, then replaced only by first obtained HTTP200 outcome.',
        'new_attempt_namespace':'quota_resume_01'})
    files=[Path(__file__),g.REPO/'tests/test_gemini_quota_resume.py',*REC.glob('*.json'),snapshot,REC/'rejected_rows.jsonl']
    g.save(REC/'execution_lock.json',{'at_utc':g.now(),'files_sha256':{g.rel(p):g.sha(p) for p in files}})
    verify()
    with out.with_suffix('.tmp').open('w') as f:
        for row in g.read_rows(snapshot):
            if not row.get('error'):f.write(json.dumps(row,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
    out.with_suffix('.tmp').replace(out)
    (g.ROOT/'full_halt.json').unlink()
    g.save(REC/'prepared.json',{'at_utc':g.now(),'retained':1346,'resubmission_eligibility':'Four explicit no-generation quota rejections only'})

def verify():
    g.verify_lock()
    for p,h in json.loads((REC/'execution_lock.json').read_text())['files_sha256'].items():assert g.sha(g.REPO/p)==h,p

def run():
    verify()
    with (g.ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=PacedBudget(g.ROOT/'calls_ledger.jsonl',g.CAP)
        original=g.call
        g.call=lambda row,profile,budget,key:original(row,'quota_resume_01',budget,key)
        g.collect('full',budget,os.environ['GEMINI_API_KEY'])
        review=json.loads((g.ROOT/'full_review.json').read_text())
        review.update(quota_recovery_lock_sha256=g.sha(REC/'execution_lock.json'),quota_recovery_applied=True)
        g.save(g.ROOT/'full_review.json',review)
        verify()

if __name__=='__main__':
    {'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
