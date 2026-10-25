"""Resume explicit no-output monthly-quota rejections after capacity update."""
import fcntl,json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import resume_gemini_observe_control as owner
from scripts import run_gemini_generate_content_control as g
from scripts import resume_gemini_generate_content_quota as quota
REC=g.ROOT/'monthly_cap_recovery_01'

def prepare():
    owner.verify();REC.mkdir(exist_ok=False)
    path=g.ROOT/'responses'/f'responses_{g.SID}_native.jsonl'
    shutil.copyfile(path,REC/'responses_before_recovery.jsonl')
    shutil.copyfile(g.ROOT/'full_halt.json',REC/'full_halt.json')
    rows=g.read_rows(path);failed=[r for r in rows if r.get('error')]
    assert len(rows)==3175 and len(failed)==9
    for r in failed:
        assert r['error']=='HTTP429'
        rawpath=g.REPO/r['raw_http_capture']['path'];assert g.sha(rawpath)==r['raw_http_capture']['sha256']
        raw=json.loads(rawpath.read_text());assert raw['error']['code']==429 and 'monthly spending cap' in raw['error']['message'] and not raw.get('candidates')
        g.append(REC/'rejected_attempt_records.jsonl',r)
    g.save(REC/'amendment.json',{'at_utc':g.now(),'reason':'Google project monthly spending cap reached, independently of model behavior. Nine terminal429rejections had no generated output;281identities never dispatched.', 'retained_generated_identities':3166,'remaining_generation_identities':290,'retry_rule':'After explicit capacity-update confirmation, allow one further cycle of at most3transport attempts on9rejected identities and original max3on281undispatched identities. Never retry an HTTP200outcome. Archive all failed attempts and preserve every generated response.', 'owner_behavior_policy':'Unchanged: retain and grade observed native calls despite requested NONE; no behavioral exclusion or restart.', 'spend_cap_usd':100,'grading_cap_usd':25,'new_attempt_namespace':'monthly_cap_resume_01','capacity_update_required':True})
    files=[Path(__file__),*REC.glob('*.json'),REC/'responses_before_recovery.jsonl',REC/'rejected_attempt_records.jsonl']
    g.save(REC/'execution_lock.json',{'at_utc':g.now(),'files_sha256':{g.rel(p):g.sha(p) for p in files}})
    verify()
    tmp=path.with_suffix('.tmp')
    with path.open() as inp,tmp.open('w') as out:
        for line in inp:
            if not json.loads(line).get('error'):out.write(line)
        out.flush();os.fsync(out.fileno())
    tmp.replace(path)
    g.save(REC/'prepared.json',{'at_utc':g.now(),'generated_observations_preserved':3166,'remaining':290,'paused_on_external_spend_cap':True})

def verify():
    owner.verify()
    for p,h in json.loads((REC/'execution_lock.json').read_text())['files_sha256'].items():assert g.sha(g.REPO/p)==h,p

def run():
    verify()
    confirmation=json.loads((REC/'capacity_update_confirmation.json').read_text())
    assert confirmation['capacity_update_confirmed'] is True
    with (g.ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        halt=g.ROOT/'full_halt.json'
        if halt.exists():
            assert g.sha(halt)==g.sha(REC/'full_halt.json')
            halt.unlink()
        budget=quota.PacedBudget(g.ROOT/'calls_ledger.jsonl',g.CAP-owner.qualification_spend())
        original=g.call;g.operational_error=owner.observed_operational_error
        def call(row,profile,budget,key):
            record,error=original(row,'monthly_cap_resume_01',budget,key)
            return owner.observe_record(record),error
        g.call=call;g.collect('full',budget,os.environ['GEMINI_API_KEY'])
        review=json.loads((g.ROOT/'full_review.json').read_text())
        rows=g.read_rows(g.ROOT/'responses'/f'responses_{g.SID}_native.jsonl')
        review.update(owner_amendment_applied=True,owner_observation_amendment_sha256=g.sha(owner.REC/'execution_lock.json'),quota_recovery_lock_sha256=g.sha(quota.REC/'execution_lock.json'),monthly_cap_recovery_lock_sha256=g.sha(REC/'execution_lock.json'),capacity_update_confirmation_sha256=g.sha(REC/'capacity_update_confirmation.json'),disabled_native_calls_observed=sum(owner.observe_record(r)['disabled_native_call_observed'] for r in rows),eligibility='Retain all generated outcomes by requested arm assignment under owner amendment.')
        g.save(g.ROOT/'full_review.json',review);verify()

if __name__=='__main__':{'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
