import copy
import pytest
from scripts import run_gemini_direct_control as g


def row(arm='DISABLED_BASELINE'):
    return {'custom_id': 'example', 'control_arm':arm, 'body':{'model':'gemini-3.5-flash','messages':[{'role':'user','content':'test'}], 'tools':[{'type':'function','function':{'name':'noop','parameters':{'type':'object','properties':{}}}}], 'tool_choice':'auto' if arm=='TOOL_CONTEMPORANEOUS' else 'none','temperature':0,'top_p':1,'max_tokens':4096}}


def test_request_preserves_schema_and_does_not_mutate():
    r=row(); original=copy.deepcopy(r)
    assert g.request_body(r)==r['body']
    assert r==original
    for field in ('model','max_tokens','temperature','top_p'):
        bad=copy.deepcopy(r);bad['body'][field]='changed'
        with pytest.raises(ValueError):g.request_body(bad)


def test_disabled_calls_fail_and_format_failure_is_observed():
    response={'model':'gemini-3.5-flash','choices':[{'message':{'content':'```json\n{}\n```'},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':5}}
    assert not g.operational_error(row(),response)
    response['choices'][0]['message']['tool_calls']=[{'function':{'name':'noop','arguments':'{}'}}]
    assert 'disabled' in g.operational_error(row(),response)
    assert not g.operational_error(row('TOOL_CONTEMPORANEOUS'),response)
    response['model']='other'
    assert 'model' in g.operational_error(row('TOOL_CONTEMPORANEOUS'),response)


def test_budget_restart_and_ambiguity(tmp_path):
    p=tmp_path/'ledger.jsonl';b=g.Budget(p,1)
    b.reserve('a',.8)
    with pytest.raises(RuntimeError):b.reserve('b',.3)
    with pytest.raises(RuntimeError):g.Budget(p,1)
    b.settle('a',.1)
    resumed=g.Budget(p,1);resumed.reserve('b',.8);resumed.settle('b',.2)
    assert resumed.used==pytest.approx(.3)
    with pytest.raises(RuntimeError):resumed.reserve('a',.1)


def test_physical_lf_preserves_unicode_and_duplicates_fail(tmp_path):
    p=tmp_path/'r.jsonl';g.append(p,{'custom_id':'a','content':'x\u2028y'})
    assert g.read_rows(p)[0]['content']=='x\u2028y'
    g.append(p,{'custom_id':'a'})
    with pytest.raises(ValueError):g.unique_rows(p)


def test_request_arm_isolation():
    r=row();f=copy.deepcopy(r);f['control_arm']='DISABLED_FORMAT_JSON'
    f['body']['messages'][0]['content']+='\n\nformat'
    assert g.request_body(f)['tools']==g.request_body(r)['tools']
    t=row('TOOL_CONTEMPORANEOUS');a=g.request_body(t);b=g.request_body(r)
    a['tool_choice']=b['tool_choice'];assert a==b
