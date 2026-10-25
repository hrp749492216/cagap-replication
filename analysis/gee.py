import logging

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import OrdinalGEE

_log = logging.getLogger(__name__)

def run_paired_gee(df, subject_col='case_id', mode_col='mode', outcome_col='compliance', cluster_col='case_id'):
    """
    Runs a GEE for paired binary data.
    Effectively a logistic GEE with exchangeable correlation structure.

    The exchangeable working-correlation structure is a *simplifying*
    assumption: it treats every pair of observations within a cluster as
    equally correlated. The actual data-generating process here is
    nested (prompt → subject → persona) so within-prompt correlations
    differ across subject/persona pairs. Because GEE estimates SE via a
    sandwich estimator, the point estimate is consistent under any working
    correlation; the SE may be slightly off when the working correlation
    is misspecified. This is acceptable for the secondary/descriptive role
    GEE plays in this paper — the primary inference is the prompt-clustered
    ConditionalLogit in scripts/analyze_results.py, which does not depend
    on a working correlation. If a future analysis needs nested clustering
    properly, switch to a mixed-effects model (no first-class Python lib;
    requires R/glmmTMB or PyMC) rather than tweaking the GEE working
    correlation.

    df: DataFrame with at least columns [subject_col, mode_col, outcome_col]
    mode_col: should have 'TOOL' vs 'DISABLED'

    Returns:
        Summary of the model or specific coefficients/p-values.
    """
    # Ensure mode is categorical/binary
    df = df.copy()
    # TOOL=1, DISABLED=0
    df['mode_bin'] = (df[mode_col] == 'TOOL').astype(int)

    # Check if we have both modes
    if df['mode_bin'].nunique() < 2:
        return None

    try:
        fam = sm.families.Binomial()
        ind = sm.cov_struct.Exchangeable()

        # Sort by cluster to ensure GEE works correctly
        df = df.sort_values(cluster_col)

        gee_model = smf.gee(f"{outcome_col} ~ mode_bin",
                            groups=df[cluster_col],
                            data=df,
                            family=fam,
                            cov_struct=ind)
        results = gee_model.fit()
        return results
    except Exception as exc:
        _log.warning("run_paired_gee failed: %s: %s", type(exc).__name__, exc)
        return None

def extract_gee_stats(results):
    """Extracts odds ratio, CI and p-value from GEE results."""
    if results is None:
        return None

    params = results.params
    conf_int = results.conf_int()
    pvalues = results.pvalues
    bse = results.bse
    tvalues = results.tvalues

    # Focus on mode_bin coefficient
    if 'mode_bin' not in params:
        return None

    coef = params['mode_bin']
    lower = conf_int.loc['mode_bin', 0]
    upper = conf_int.loc['mode_bin', 1]
    p_val = pvalues['mode_bin']

    return {
        "log_or": float(coef),
        "se": float(bse['mode_bin']),
        "z": float(tvalues['mode_bin']),
        "or": float(np.exp(coef)),
        "lower_ci": float(np.exp(lower)),
        "upper_ci": float(np.exp(upper)),
        "p_value": float(p_val)
    }


def run_ordinal_gee(
    df,
    mode_col='mode',
    outcome_col='outcome_ordinal',
    cluster_col='case_id',
):
    """
    Runs a proportional-odds ordinal GEE for the TOOL-vs-DISABLED contrast.

    The model is descriptive/secondary. It estimates a single mode coefficient
    across ordinal thresholds for Refuse < Partial < Comply.
    """
    df = df.copy()
    needed = {mode_col, outcome_col, cluster_col}
    if not needed.issubset(df.columns):
        return None
    df = df[df[mode_col].isin(['TOOL', 'DISABLED'])].copy()
    df = df.dropna(subset=[outcome_col, cluster_col])
    if df.empty:
        return None
    df['mode_bin'] = (df[mode_col] == 'TOOL').astype(int)
    if df['mode_bin'].nunique() < 2:
        return None
    if df[outcome_col].nunique() < 3:
        return None

    # Statsmodels' GEE backends require observations to be contiguous within
    # each cluster; without sorting, OrdinalGEE will quietly mis-pair records
    # in some statsmodels versions. Match the ordering done in run_paired_gee.
    df = df.sort_values(cluster_col)
    try:
        model = OrdinalGEE(
            df[outcome_col].astype(int),
            df[['mode_bin']],
            groups=df[cluster_col],
            cov_struct=Exchangeable(),
        )
        return model.fit()
    except Exception as exc:
        _log.warning("run_ordinal_gee failed: %s: %s", type(exc).__name__, exc)
        return None


def extract_ordinal_gee_stats(results):
    """Extracts the TOOL proportional-odds coefficient from OrdinalGEE."""
    if results is None:
        return None

    params = results.params
    if 'mode_bin' not in params:
        return None
    conf_int = results.conf_int()
    pvalues = results.pvalues
    bse = results.bse
    tvalues = results.tvalues
    coef = params['mode_bin']
    lower = conf_int.loc['mode_bin', 0]
    upper = conf_int.loc['mode_bin', 1]
    p_val = pvalues['mode_bin']
    return {
        "model": "ordinal_proportional_odds_gee",
        "log_or": float(coef),
        "se": float(bse['mode_bin']),
        "z": float(tvalues['mode_bin']),
        "or": float(np.exp(coef)),
        "lower_ci": float(np.exp(lower)),
        "upper_ci": float(np.exp(upper)),
        "p_value": float(p_val),
    }
