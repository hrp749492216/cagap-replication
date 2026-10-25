import numpy as np
import pytest
import pandas as pd
from analysis.judge_agreement import (
    cohen_kappa_per_judge,
    fleiss_kappa,
    krippendorff_alpha,
    median_judge_fraction,
)

def test_judge_agreement():
    data = {
        'gold':    ['C', 'R', 'C', 'R', 'C'],
        'judge_a': ['C', 'R', 'C', 'R', 'R'], # 4/5 agreement
        'judge_b': ['C', 'R', 'C', 'R', 'C'], # 5/5 agreement
        'judge_c': ['R', 'C', 'R', 'C', 'R']  # 0/5 agreement
    }
    df = pd.DataFrame(data)

    kappas = cohen_kappa_per_judge(df, 'gold', ['judge_a', 'judge_b', 'judge_c'])
    assert kappas['judge_b'] == 1.0
    assert kappas['judge_a'] > 0
    assert kappas['judge_c'] < 0

def test_median_judge_fraction():
    data = {
        'judge_a': ['C', 'C', 'R'],
        'judge_b': ['C', 'R', 'R'],
        'judge_c': ['R', 'C', 'C']
    }
    df = pd.DataFrame(data)

    # Case 1: C, C, R -> median is C. Judges a and b are median.
    # Case 2: C, R, C -> median is C. Judges a and c are median.
    # Case 3: R, R, C -> median is R. Judges a and b are median.

    # Total median counts: a=3, b=2, c=1
    # Total rows: 3
    # Fractions: a=1.0, b=0.66, c=0.33

    fracs = median_judge_fraction(df, ['judge_a', 'judge_b', 'judge_c'])
    assert fracs['judge_a'] == 1.0
    assert pytest.approx(fracs['judge_b'], 0.01) == 0.666
    assert pytest.approx(fracs['judge_c'], 0.01) == 0.333


def test_median_judge_fraction_rejects_even_judges():
    df = pd.DataFrame({"judge_a": ["C"], "judge_b": ["R"]})
    with pytest.raises(ValueError, match="odd number of judges"):
        median_judge_fraction(df, ["judge_a", "judge_b"])


def test_median_judge_fraction_rejects_single_judge():
    df = pd.DataFrame({"judge_a": ["C"]})
    with pytest.raises(ValueError, match="odd number of judges"):
        median_judge_fraction(df, ["judge_a"])


def test_median_judge_fraction_handles_five_judges():
    # 5 judges: median is index 2 of sorted vals.
    df = pd.DataFrame(
        {
            "j1": ["R"],
            "j2": ["R"],
            "j3": ["P"],
            "j4": ["C"],
            "j5": ["C"],
        }
    )
    fracs = median_judge_fraction(df, ["j1", "j2", "j3", "j4", "j5"])
    # Sorted vals: R(0), R(0), P(1), C(2), C(2). Median = P → only j3 matches.
    assert fracs["j3"] == 1.0
    assert fracs["j1"] == 0.0
    assert fracs["j5"] == 0.0


def test_median_judge_fraction_docstring_names_active_judge_roster():
    assert "GLM 5.1" in median_judge_fraction.__doc__
    assert "command-a-03-2025" in median_judge_fraction.__doc__
    assert "Grok 4.2" not in median_judge_fraction.__doc__
    assert "Kimi K2.6" not in median_judge_fraction.__doc__


def test_krippendorff_alpha_uses_ordinal_distance_for_rpc_labels():
    data = np.array(
        [
            ["R", "R", "R"],
            ["P", "C", "P"],
            ["P", "C", "C"],
        ],
        dtype=object,
    )

    assert krippendorff_alpha(data) == pytest.approx(-1.0 / 9.0)


def test_krippendorff_alpha_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        krippendorff_alpha(np.array(["R", "P", "C"], dtype=object))


def test_fleiss_kappa_rejects_unequal_rater_counts():
    matrix = np.array([[2, 1, 0], [1, 1, 0]])

    with pytest.raises(ValueError, match="same number of raters"):
        fleiss_kappa(matrix)
