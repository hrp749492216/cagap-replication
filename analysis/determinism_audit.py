
def analyze_determinism(
    df,
    group_cols=("subject_id", "mode", "persona", "case_id"),
    outcome_col="compliance",
):
    """
    Analyzes within-condition instability from duplicated calls.
    Assumes some (subject_id, mode, persona, case_id) combinations have multiple entries.
    """
    group_cols = list(group_cols)
    group_sizes = df.groupby(group_cols, dropna=False).size()
    duplicated_groups = group_sizes[group_sizes > 1]

    if duplicated_groups.empty:
        return {"instability_score": 0.0, "duplicated_n": 0}

    total_pairs = int(((duplicated_groups * (duplicated_groups - 1)) // 2).sum())
    label_counts = df.groupby([*group_cols, outcome_col], dropna=False).size()
    same_label_pairs = int(((label_counts * (label_counts - 1)) // 2).sum())
    instability_count = total_pairs - same_label_pairs

    score = instability_count / total_pairs if total_pairs > 0 else 0.0
    return {
        "instability_score": score,
        "duplicated_n": int(len(duplicated_groups)),
        "total_pairs": total_pairs,
        "instability_count": instability_count,
    }
