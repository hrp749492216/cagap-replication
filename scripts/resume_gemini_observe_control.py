"""Owner-amended observation rule: retain native calls despite disabled setting."""
from __future__ import annotations
import copy,fcntl,json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_gemini_generate_content_control as g
from scripts import resume_gemini_generate_content_quota as quota
REC=g.ROOT/'owner_observation_amendment_20261005'
NATIVE_CALL_ERROR='native tool calls emitted in disabled arm'
original_operational_error=g.operational_error

def observed_operational_error(row,response):
    error=original_operational_error(row,response)
    return None if error==NATIVE_CALL_ERROR else error

def observe_record(record):
    row=copy.deepcopy(record)
    choices=(row.get('raw_response')or{}).get('choices')or[]
    calls=((choices[0].get('message')or{}).get('tool_calls')or[]) if choices else []
    row['disabled_native_call_observed']=bool(row.get('body',{}).get('tool_choice')=='none' and calls)
    row['behavioral_observation_amendment']='owner_observation_amendment_20261005'
    if row.get('error')==NATIVE_CALL_ERROR:
        assert row['disabled_native_call_observed']
        row['prior_operational_error']=row.pop('error')
        row['native_result_status']='succeeded'
    return row

def prepare():
    quota.verify();REC.mkdir(exist_ok=False)
    source=g.ROOT/'responses'/f'responses_{g.SID}_native.jsonl'
    shutil.copyfile(source,REC/'responses_before_amendment.jsonl')
    shutil.copyfile(g.ROOT/'full_halt.json',REC/'full_halt.json')
    rows=g.read_rows(source);assert len(rows)==2002
    failures=[r for r in rows if r.get('error')];assert len(failures)==1 and failures[0]['error']==NATIVE_CALL_ERROR
    annotation=observe_record(failures[0]);g.save(REC/'prior_failure_reclassification.json',{'at_utc':g.now(),'custom_id':annotation['custom_id'],'prior_record_sha256':g.control._sha256_json(failures[0]),'amended_record_sha256':g.control._sha256_json(annotation),'raw_capture_unchanged':annotation['raw_http_capture'],'reason':'HTTP200 generated output is an observed behavior, not an API error under owner amendment. No text, call, identity or request changed.'})
    g.save(REC/'amendment.json',{'at_utc':g.now(),'exact_owner_instruction':'No let Gemini do what it does, we do the experiment like we want to do and record what happened. There is nothing we can do. How much more Gemini three arm control runs is remaining?',
        'timing':'Owner amendment after observing2,002native-route rows and one disabled-arm native call; prospective for1,454undispatched identities and retrospective inclusion of every existing native-route observation.',
        'policy':'Treat tool_choice:none/NONE as requested assignment, not a guarantee of provider suppression. Native functionCalls in either disabled arm are retained, annotated and graded by the unchanged content/structured-call rules. Do not filter, replace, or retry HTTP200 outcomes.',
        'cohort':'Retain every existing2,002native-route identity and collect exactly the1,454missing identities, making3,456rows and1,152triplets on one unchanged native route. The separate1,913-row compatibility attempt remains an unpooled historical partial attempt.',
        'safety_gates_retained':'Authentication, explicit transport errors, model identity and spending caps remain enforced. Same endpoint, model, prompts, schemas, decoding, block order and raw archive.',
        'analysis_interpretation':'Post-registration exploratory, with the after-observation amendment disclosed. Estimate requested-interface assignment effects; do not claim physically enforced non-callability or identify an internal mechanism. Report observed native-call counts in every arm.',
        'request_namespace':'owner_observe_01','expected_remaining_by_arm':{'DISABLED_BASELINE':484,'DISABLED_FORMAT_JSON':483,'TOOL_CONTEMPORANEOUS':487},'google_combined_cap_usd':100,'pacing':'Unchanged quota-aware pacing:1second minimum dispatch spacing and$8rolling600second ceiling, including in-flight reservations.'})
    files=[Path(__file__),g.REPO/'tests/test_gemini_observe_control.py',g.REPO/'specs/002-gemini-direct-control/owner-observation-amendment.md',*REC.glob('*.json'),REC/'responses_before_amendment.jsonl']
    g.save(REC/'execution_lock.json',{'at_utc':g.now(),'files_sha256':{g.rel(p):g.sha(p) for p in files}})
    verify()
    # Preserve all other records byte-for-byte; only reclassify the one
    # HTTP200 record previously miscategorized as an operational error.
    tmp=source.with_suffix('.tmp')
    with source.open() as src,tmp.open('w') as out:
        for line in src:
            row=json.loads(line)
            out.write(json.dumps(observe_record(row),ensure_ascii=False)+'\n' if row.get('error')==NATIVE_CALL_ERROR else line)
        out.flush();os.fsync(out.fileno())
    tmp.replace(source)
    (g.ROOT/'full_halt.json').unlink()
    g.save(REC/'prepared.json',{'at_utc':g.now(),'retained':2002,'remaining':1454,'raw_outputs_changed':0,'metadata_records_reclassified':1,'owner_override_applied':True})

def verify():
    quota.verify()
    for p,h in json.loads((REC/'execution_lock.json').read_text())['files_sha256'].items():assert g.sha(g.REPO/p)==h,p

def qualification_spend():
    return sum(r['usd'] for n in ['gemini_vertex_qualification_20261005','gemini_stable_v1_qualification_20261005'] for r in g.read_rows(g.REPO/'outputs'/n/'budget.jsonl') if r['event']=='settle')

def run():
    verify()
    with (g.ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=quota.PacedBudget(g.ROOT/'calls_ledger.jsonl',g.CAP-qualification_spend())
        original=g.call
        g.operational_error=observed_operational_error
        def call(row,profile,budget,key):
            record,error=original(row,'owner_observe_01',budget,key)
            return observe_record(record),error
        g.call=call
        g.collect('full',budget,os.environ['GEMINI_API_KEY'])
        review=json.loads((g.ROOT/'full_review.json').read_text())
        rows=g.read_rows(g.ROOT/'responses'/f'responses_{g.SID}_native.jsonl')
        review.update(owner_observation_amendment_sha256=g.sha(REC/'execution_lock.json'),owner_amendment_applied=True,quota_recovery_lock_sha256=g.sha(quota.REC/'execution_lock.json'),disabled_native_calls_observed=sum(observe_record(r)['disabled_native_call_observed'] for r in rows),eligibility='All outcomes retained by requested arm assignment under explicit owner amendment; suppression is observed, not an exclusion gate.')
        g.save(g.ROOT/'full_review.json',review);verify()

if __name__=='__main__':{'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
