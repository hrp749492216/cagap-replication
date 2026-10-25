from scripts import run_gemini_direct_control_reviewed as r


def test_cost_includes_unreported_reasoning_tokens():
    assert r.measured_cost({'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':310}},1)==(100*1.5+210*9)/1e6
    assert r.measured_cost({'usage':{'prompt_tokens':100,'completion_tokens':300,'total_tokens':310}},1)==(100*1.5+300*9)/1e6
    assert r.measured_cost({},1)==1


def test_visibility_uses_frozen_classifier_not_marker_obedience():
    rows=[{'route_label':'google-direct','mode':mode,'prompt_case_id':'a','http_status':200,'finish_reason':'stop','prompt_tokens':tokens} for mode,tokens in [('NOTOOLS',68),('DISABLED',1141),('TOOL',1141)]]
    assert r.visibility(rows)=='schema_visible'
    rows[1]['prompt_tokens']=68
    assert r.visibility(rows)=='schema_hidden'
