"""
Statistical analysis utilities for Tavtigian simulation results.

Key analyses
------------
1. C* profile (O_PVSt vs prior): C* has a minimum near prior=0.25 and rises
   in both directions — this means LR+ thresholds become more stringent (higher
   for pathogenic, lower for benign) as the prior deviates from ~0.25.

2. Classification boundary validity: the fixed point thresholds (P≥10, LP=6–9,
   LB=–1 to –6, B≤–7) were derived for prior=0.10. At other priors the same
   boundaries give different posteriors and may fail to satisfy P>0.99, LP>0.90,
   LB<0.10, B<0.001.

3. Feasibility space (Tavtigian 2018 Figure 1): for each (prior, C) pair,
   whether the LP and LB combining rules are simultaneously satisfied. The
   minimum feasible C is 81 at prior=0.25; it rises in both directions.

4. Combining-rule consistency: Path(ii) [2 strong] and LP(i) [1 VeryStrong +
   1 Moderate] are permanently inconsistent regardless of (prior, C).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import (
    TavtigianResult, POINT_VALUES, CLASSIFICATION_BOUNDARIES,
    POSTERIOR_TARGETS, locallrPlus2Posterior,
)
from .suite import SimulationSuite


def _df(suite_or_df) -> pd.DataFrame:
    if isinstance(suite_or_df, SimulationSuite):
        return suite_or_df.to_dataframe()
    return suite_or_df


# ── 1. C* profile ─────────────────────────────────────────────────────────────

def C_star_profile(suite_or_df) -> pd.DataFrame:
    """
    Return (prior, C_star, total_fails) for plotting C* vs prior.

    The paper predicts a minimum near prior=0.25 (C_min=81).  With C_max
    capped at 350, the minimum shifts to ~0.20 but the U-shape is still clear.
    """
    df = _df(suite_or_df)
    return df[["prior", "C_star", "total_fails"]].copy().reset_index(drop=True)


def find_C_star_minimum(suite_or_df) -> Dict:
    """Locate the prior where C* is smallest (framework most 'natural')."""
    df = _df(suite_or_df)
    idx = df["C_star"].idxmin()
    row = df.loc[idx]
    return {
        "prior_at_min_C": float(row["prior"]),
        "C_star_min":     int(row["C_star"]),
        "fails_at_min":   int(row["total_fails"]),
    }


# ── 2. Classification boundary validity ───────────────────────────────────────

def boundary_validity(suite_or_df) -> pd.DataFrame:
    """
    For each prior, report whether the fixed Tavtigian-2020 point thresholds
    deliver the correct posterior.

    Returns a long-form DataFrame with columns:
        prior, boundary, posterior, target, passes, direction
    """
    df = _df(suite_or_df)
    rows = []
    for _, row in df.iterrows():
        for name in CLASSIFICATION_BOUNDARIES:
            target, op = POSTERIOR_TARGETS[name]
            rows.append({
                "prior":     row["prior"],
                "boundary":  name,
                "points":    CLASSIFICATION_BOUNDARIES[name],
                "lr":        row[f"bnd_lr_{name}"],
                "posterior": row[f"bnd_post_{name}"],
                "target":    target,
                "op":        op,
                "passes":    bool(row[f"bnd_pass_{name}"]),
            })
    return pd.DataFrame(rows)


def valid_prior_range(suite_or_df) -> Dict:
    """
    Find the range of priors where ALL six classification boundaries pass.

    Returns a dict with:
        valid_mask         : boolean array aligned to df["prior"]
        valid_prior_lo     : lowest prior where all boundaries pass
        valid_prior_hi     : highest prior where all boundaries pass
        n_valid            : number of priors where all pass
        boundary_pass_rate : per-boundary pass fraction
    """
    df = _df(suite_or_df)
    pass_cols = [f"bnd_pass_{name}" for name in CLASSIFICATION_BOUNDARIES]
    all_pass = df[pass_cols].all(axis=1)

    valid_priors = df.loc[all_pass, "prior"]
    rates = {name: float(df[f"bnd_pass_{name}"].mean())
             for name in CLASSIFICATION_BOUNDARIES}

    return {
        "valid_mask":         all_pass.values,
        "valid_prior_lo":     float(valid_priors.min()) if len(valid_priors) else float("nan"),
        "valid_prior_hi":     float(valid_priors.max()) if len(valid_priors) else float("nan"),
        "n_valid":            int(all_pass.sum()),
        "boundary_pass_rate": rates,
    }


# ── 3. Feasibility space (reproduces Tavtigian 2018 Figure 1) ─────────────────

def feasibility_space(
    priors: Optional[np.ndarray] = None,
    C_range: Optional[np.ndarray] = None,
    original: bool = False,
) -> pd.DataFrame:
    """
    For a 2-D grid of (prior, C) values, compute whether the LP and LB
    combining rules are simultaneously satisfied — reproducing Figure 1 of
    Tavtigian 2018.

    LP satisfied: all LP combining-rule posteriors ≥ 0.90.
    LB satisfied: all LB combining-rule posteriors ≤ 0.10.
    Both satisfied: the feasibility region in the figure.

    Returns a DataFrame with columns:
        prior, C, lp_min_posterior, lb_max_posterior,
        lp_satisfied, lb_satisfied, both_satisfied
    """
    from assay_calibration.fit_utils.evidence_thresholds import (
        likelyPathogenicRulesPosterior,
        likelybenignRulesPosterior,
    )

    if priors is None:
        priors = np.linspace(0.01, 0.45, 120)
    if C_range is None:
        C_range = np.unique(np.concatenate([
            np.arange(81, 200, 2),
            np.arange(200, 500, 5),
            np.arange(500, 1001, 20),
        ]))

    rows = []
    for prior in priors:
        for C in C_range:
            lp_posts = likelyPathogenicRulesPosterior(int(C), prior, original)
            lb_posts = likelybenignRulesPosterior(int(C), prior, original)
            lp_min = float(np.min(lp_posts))
            lb_max = float(np.max(lb_posts))
            rows.append({
                "prior":           prior,
                "C":               int(C),
                "lp_min_post":     lp_min,
                "lb_max_post":     lb_max,
                "lp_satisfied":    lp_min >= 0.90,
                "lb_satisfied":    lb_max <= 0.10,
                "both_satisfied":  (lp_min >= 0.90) and (lb_max <= 0.10),
            })
    return pd.DataFrame(rows)


def minimum_C_curve(
    priors: Optional[np.ndarray] = None,
    original: bool = False,
) -> pd.DataFrame:
    """
    For each prior, find the minimum C that satisfies the LP combining rules
    and the minimum C that satisfies the LB combining rules separately.
    This reproduces the two boundary curves from Tavtigian 2018 Figure 1.
    """
    from assay_calibration.fit_utils.evidence_thresholds import (
        likelyPathogenicRulesPosterior,
        likelybenignRulesPosterior,
    )

    if priors is None:
        priors = np.linspace(0.01, 0.45, 200)

    C_search = np.arange(1, 2001)
    rows = []
    for prior in priors:
        min_C_lp, min_C_lb = None, None
        for C in C_search:
            lp_posts = likelyPathogenicRulesPosterior(int(C), prior, original)
            lb_posts = likelybenignRulesPosterior(int(C), prior, original)
            if min_C_lp is None and np.all(lp_posts >= 0.90):
                min_C_lp = int(C)
            if min_C_lb is None and np.all(lb_posts <= 0.10):
                min_C_lb = int(C)
            if min_C_lp is not None and min_C_lb is not None:
                break
        rows.append({
            "prior":    prior,
            "min_C_LP": min_C_lp,
            "min_C_LB": min_C_lb,
        })
    return pd.DataFrame(rows)


# ── 4. Combining-rule consistency ─────────────────────────────────────────────

def combining_rule_consistency(suite: SimulationSuite) -> pd.DataFrame:
    """
    Annotate each combining-rule posterior with its expected range and whether
    it is one of the two permanently inconsistent rules from Tavtigian 2018:
      - Path(ii):  2 strong → should be P (≥0.99), actually gives ~0.975
      - LP(i):     1 VeryStrong + 1 Moderate → should be LP (<0.99), actually ~0.994

    Returns the long-form combining-rule DataFrame with added columns:
        target_lo, target_hi, in_range, known_inconsistency
    """
    df = suite.combining_rule_dataframe()
    targets = {
        "P":  (0.99, 1.01),
        "LP": (0.90, 0.99),
        "B":  (-0.01, 0.01),
        "LB": (0.01, 0.10),
    }
    # Known inconsistencies in the ORIGINAL ACMG rules (original=True):
    #   P[7]  = 2 Strong   → post ≈ 0.975, labeled P but gives LP-range posterior
    #   LP[5] = 1M + 1VS   → post ≈ 0.994, labeled LP but gives P-range posterior
    # Tavtigian 2018 fixes these by swapping: with original=False (the default)
    # P[7]=1M+1VS and LP[5]=2St are both consistent. Flag based on suite.original.
    if suite.original:
        inconsistent = {("P", 7), ("LP", 5)}
    else:
        inconsistent = set()  # modification fixes both inconsistencies

    rows = []
    for _, row in df.iterrows():
        cat = row["category"]
        lo, hi = targets[cat]
        post = row["posterior"]
        rows.append({
            **row.to_dict(),
            "target_lo":          lo,
            "target_hi":          hi,
            "in_range":           lo <= post <= hi,
            "known_inconsistency": (cat, int(row["rule_idx"])) in inconsistent,
        })
    return pd.DataFrame(rows)


# ── 5. LR+ threshold sensitivity summary ─────────────────────────────────────

def lr_threshold_sensitivity(
    suite_or_df,
    codes: List[int] = (1, 2, 4, 8),
    ref_prior: float = 0.10,
) -> pd.DataFrame:
    """
    Compute log2(LR+_threshold(prior) / LR+_threshold(ref_prior)) for each
    evidence code.  Values > 0 mean the pathogenic threshold is higher
    (more stringent) than at the reference prior; for benign codes,
    values < 0 mean the threshold is lower (more stringent).

    This directly tests the U-shape hypothesis: do thresholds diverge
    symmetrically as prior moves away from the minimum-C prior?
    """
    df = _df(suite_or_df)
    ref_row = df.iloc[(df["prior"] - ref_prior).abs().argsort().iloc[0]]

    rows = []
    for _, row in df.iterrows():
        for k in codes:
            for sign, direction in [(1, "pathogenic"), (-1, "benign")]:
                col = f"log10_lr_p{k}" if sign == 1 else f"log10_lr_b{k}"
                ref_col = col
                delta = row[col] - ref_row[ref_col]
                rows.append({
                    "prior":        row["prior"],
                    "code":         k,
                    "direction":    direction,
                    "log10_lr":     row[col],
                    "delta_log10":  delta,  # positive = more stringent for pathogenic
                    "C_star":       row["C_star"],
                })
    return pd.DataFrame(rows)


# ── 6. Comprehensive summary ──────────────────────────────────────────────────

def summarize_suite(suite: SimulationSuite) -> Dict:
    """
    Run all analyses and return a single nested dict.

    Keys
    ----
    main_df               : suite.to_dataframe()
    C_star_profile        : C* and fails vs prior
    C_star_minimum        : prior and value where C* is smallest
    boundary_validity_df  : long-form boundary pass/fail table
    valid_prior_range     : dict with lo/hi/n_valid
    combining_rules_df    : annotated combining-rule posteriors
    lr_sensitivity_df     : log2(LR+ / LR+_ref) for key tiers
    """
    df = suite.to_dataframe()
    return {
        "main_df":              df,
        "C_star_profile":       C_star_profile(df),
        "C_star_minimum":       find_C_star_minimum(df),
        "boundary_validity_df": boundary_validity(df),
        "valid_prior_range":    valid_prior_range(df),
        "combining_rules_df":   combining_rule_consistency(suite),
        "lr_sensitivity_df":    lr_threshold_sensitivity(df),
    }
