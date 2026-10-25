import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def cohen_kappa_per_judge(df, gold_col, judge_cols):
    """
    Calculates Cohen's Kappa for each judge against gold labels.
    """
    results = {}
    for j_col in judge_cols:
        # Drop rows where either judge or gold is missing
        valid = df[[gold_col, j_col]].dropna()
        if len(valid) > 0:
            kappa = cohen_kappa_score(valid[gold_col], valid[j_col])
            results[j_col] = float(kappa)
        else:
            results[j_col] = np.nan
    return results

def fleiss_kappa(matrix):
    """
    Calculates Fleiss' Kappa for a matrix of counts.
    matrix: (N, K) where N is number of subjects, K is number of categories.
    Each row sums to the number of raters (n).
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix must be a non-empty 2D array")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("matrix counts must be non-negative finite values")
    row_totals = matrix.sum(axis=1)
    if not np.all(row_totals == row_totals[0]):
        raise ValueError("each row must have the same number of raters")
    if row_totals[0] < 2:
        raise ValueError("Fleiss kappa requires at least two raters")

    N, k = matrix.shape
    n = np.sum(matrix[0, :]) # Number of raters

    # p_j: proportion of all assignments which were to the j-th category
    p = np.sum(matrix, axis=0) / (N * n)

    # P_i: extent to which raters agree for the i-th subject
    P = (np.sum(matrix**2, axis=1) - n) / (n * (n - 1))

    P_bar = np.sum(P) / N
    P_e_bar = np.sum(p**2)

    if 1 - P_e_bar == 0:
        return 1.0

    return float((P_bar - P_e_bar) / (1 - P_e_bar))

def _ordinal_distance(label_a, label_b):
    """Ordinal R/P/C distance used for Krippendorff alpha."""
    if label_a == label_b:
        return 0.0
    ordinal = {
        "R": 0,
        "Refuse": 0,
        "P": 1,
        "Partial": 1,
        "C": 2,
        "Comply": 2,
    }
    if label_a in ordinal and label_b in ordinal:
        return float(abs(ordinal[label_a] - ordinal[label_b]))
    return 1.0


def krippendorff_alpha(data):
    """
    Calculates Krippendorff's Alpha for ordinal R/P/C data.

    The disagreement distance is |R-P| = |P-C| = 1 and |R-C| = 2.
    data: (n_raters, n_subjects) array of labels. Can contain NaNs.
    """
    data = np.asarray(data, dtype=object)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array of shape (n_raters, n_subjects)")
    # Get all unique labels, preserving the pre-registered R/P/C order where
    # possible so the distance matrix is stable across subsets.
    observed = set(data[~pd.isna(data)])
    preferred_order = ["R", "P", "C", "Refuse", "Partial", "Comply"]
    labels = [label for label in preferred_order if label in observed]
    labels.extend(sorted(observed - set(labels)))
    if len(labels) <= 1:
        return 1.0

    label_to_idx = {l: i for i, l in enumerate(labels)}
    m = len(labels)
    n_raters, n_subjects = data.shape

    # Co-occurrence matrix
    coincidence = np.zeros((m, m))

    for j in range(n_subjects):
        # Items in this column
        column = data[:, j]
        column = column[~pd.isna(column)]
        mu = len(column)
        if mu < 2:
            continue

        for i_idx in range(mu):
            for k_idx in range(mu):
                if i_idx == k_idx:
                    continue
                v_i = label_to_idx[column[i_idx]]
                v_k = label_to_idx[column[k_idx]]
                coincidence[v_i, v_k] += 1.0 / (mu - 1)

    n = coincidence.sum()
    if n == 0:
        return np.nan

    # Marginal totals
    P = coincidence.sum(axis=1)

    # Observed disagreement
    D_o = 0.0
    for i in range(m):
        for k in range(m):
            D_o += coincidence[i, k] * _ordinal_distance(labels[i], labels[k])
    D_o /= n

    # Expected disagreement
    if n <= 1:
        return 1.0
    D_e = 0.0
    for i in range(m):
        for k in range(m):
            D_e += P[i] * P[k] * _ordinal_distance(labels[i], labels[k])
    D_e /= n * (n - 1)

    if D_e == 0:
        return 1.0

    return float(1.0 - (D_o / D_e))

def pairwise_agreement_matrix(df, judge_cols):
    """Calculates pairwise agreement percentage between judges."""
    n = len(judge_cols)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            valid = df[[judge_cols[i], judge_cols[j]]].dropna()
            if len(valid) > 0:
                agreement = (valid[judge_cols[i]] == valid[judge_cols[j]]).mean()
                if isinstance(agreement, pd.Series):
                    agreement = agreement.iloc[0]
                matrix[i, j] = float(agreement)
            else:
                matrix[i, j] = np.nan
    return pd.DataFrame(matrix, index=judge_cols, columns=judge_cols)

def median_judge_fraction(df, judge_cols, labels_order=['R', 'P', 'C']):
    """
    Fraction of decisions where each judge labels the ensemble median.

    Requires an odd number of judges so the median is uniquely defined; the
    paper's design fixes 3 judges (GLM 5.1, Cohere Command A / `command-a-03-2025`,
    Mistral Large 3 / `mistral-large-2512`).
    Even counts would split the median between two label values and produce
    misleading "agreement with median" rates.
    """
    n_judges = len(judge_cols)
    if n_judges < 3 or n_judges % 2 == 0:
        raise ValueError(
            f"median_judge_fraction requires an odd number of judges >= 3; got {n_judges}"
        )
    label_val = {l: i for i, l in enumerate(labels_order)}
    median_idx = n_judges // 2

    counts = {j: 0 for j in judge_cols}
    total = 0

    for _, row in df[judge_cols].iterrows():
        if row.isnull().any():
            continue

        vals = [label_val.get(row[j], -1) for j in judge_cols]
        if any(v == -1 for v in vals):
            continue

        total += 1
        sorted_vals = sorted(vals)
        median_val = sorted_vals[median_idx]

        for j_idx, j_col in enumerate(judge_cols):
            if vals[j_idx] == median_val:
                counts[j_col] += 1

    if total == 0:
        return {j: 0.0 for j in judge_cols}

    return {j: float(c / total) for j, c in counts.items()}
