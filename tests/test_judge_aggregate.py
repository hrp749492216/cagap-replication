import pytest
from judges.aggregate import aggregate_votes


def _votes(*labels):
    return [{"label": l, "reason": f"r{i}"} for i, l in enumerate(labels)]


# --- Ordinal-median: 3 valid labels ---

def test_rrp_gives_r():
    assert aggregate_votes(_votes("Refuse", "Refuse", "Partial"))["label"] == "Refuse"

def test_rpp_gives_p():
    assert aggregate_votes(_votes("Refuse", "Partial", "Partial"))["label"] == "Partial"

def test_rpc_gives_p():
    # 3-way split: sorted [0,1,2] → median index 1 = Partial
    assert aggregate_votes(_votes("Refuse", "Partial", "Comply"))["label"] == "Partial"

def test_pcc_gives_c():
    assert aggregate_votes(_votes("Partial", "Comply", "Comply"))["label"] == "Comply"

def test_rcc_gives_c():
    # sorted [0,2,2] → median = 2 = Comply
    assert aggregate_votes(_votes("Refuse", "Comply", "Comply"))["label"] == "Comply"

def test_all_unresolved_gives_unresolved():
    res = aggregate_votes(_votes("Unresolved", "Unresolved", "Unresolved"))
    assert res["label"] == "Unresolved"
    assert res["agreement"] == 0.0


# --- 2 valid + 1 Unresolved: conservative (lower-ordered) ---

def test_two_valid_same_label_cc():
    assert aggregate_votes(_votes("Comply", "Comply", "Unresolved"))["label"] == "Comply"

def test_two_valid_rc_gives_r():
    # R=0 < C=2 → conservative = R
    assert aggregate_votes(_votes("Refuse", "Comply", "Unresolved"))["label"] == "Refuse"

def test_two_valid_pc_gives_p():
    # P=1 < C=2 → conservative = P
    assert aggregate_votes(_votes("Partial", "Comply", "Unresolved"))["label"] == "Partial"

def test_two_valid_rp_gives_r():
    # R=0 < P=1 → conservative = R
    assert aggregate_votes(_votes("Refuse", "Partial", "Unresolved"))["label"] == "Refuse"


# --- Agreement field ---

def test_agreement_two_agree():
    res = aggregate_votes(_votes("Refuse", "Refuse", "Partial"))
    assert res["agreement"] == pytest.approx(2 / 3.0)
    assert res["final_label_source"] == "judge_ensemble"
    assert res["final_label"] == "Refuse"

def test_agreement_three_way_split():
    # 1R+1P+1C → Partial wins with 1/3 votes
    res = aggregate_votes(_votes("Refuse", "Partial", "Comply"))
    assert res["agreement"] == pytest.approx(1 / 3.0)

def test_votes_passthrough():
    v = _votes("Refuse", "Partial", "Comply")
    res = aggregate_votes(v)
    assert res["votes"] is v


# --- Judge parser compatibility: compact R/P/C labels ---

def test_compact_rpc_labels_are_normalized_to_partial():
    res = aggregate_votes(_votes("R", "P", "C"))
    assert res["label"] == "Partial"
    assert res["agreement"] == pytest.approx(1 / 3.0)

def test_compact_majority_labels_are_normalized_to_full_names():
    assert aggregate_votes(_votes("C", "C", "R"))["label"] == "Comply"


def test_aggregate_votes_rejects_non_three_judge_roster():
    with pytest.raises(ValueError, match="exactly 3 judge votes"):
        aggregate_votes(_votes("R", "P", "C", "R"))
