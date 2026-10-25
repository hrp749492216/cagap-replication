from analysis.determinism_audit import analyze_determinism
import pandas as pd

def test_determinism_audit():
    data = [
        {"subject_id": "s1", "mode": "m1", "persona": "p1", "case_id": "c1", "compliance": 1},
        {"subject_id": "s1", "mode": "m1", "persona": "p1", "case_id": "c1", "compliance": 0}, # Inconsistent
        {"subject_id": "s1", "mode": "m1", "persona": "p1", "case_id": "c2", "compliance": 1},
        {"subject_id": "s1", "mode": "m1", "persona": "p1", "case_id": "c2", "compliance": 1}  # Consistent
    ]
    df = pd.DataFrame(data)
    res = analyze_determinism(df)

    # 2 duplicated groups (c1 and c2).
    # c1 pair is inconsistent. c2 pair is consistent.
    # Total pairs = 2. Inconsistent = 1. Score = 0.5.
    assert res['instability_score'] == 0.5
    assert res['duplicated_n'] == 2


def test_determinism_audit_does_not_rebuild_index_per_duplicate_group(monkeypatch):
    data = []
    for group_idx in range(4):
        for compliance in (1, 0, 1):
            data.append(
                {
                    "subject_id": "s1",
                    "mode": "m1",
                    "persona": "p1",
                    "case_id": f"c{group_idx}",
                    "compliance": compliance,
                }
            )
    calls = 0
    original_set_index = pd.DataFrame.set_index

    def counted_set_index(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_set_index(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "set_index", counted_set_index)

    res = analyze_determinism(pd.DataFrame(data))

    assert res["duplicated_n"] == 4
    assert calls <= 1
