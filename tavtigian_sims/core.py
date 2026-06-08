"""
Core simulation object for the Tavtigian evidence-threshold framework.

Framework summary (Tavtigian 2018, 2020)
-----------------------------------------
For a given prior probability, the framework finds the optimal constant C*
(called O_PVSt in the papers) such that the ACMG/AMP combining rules are as
internally consistent as possible.

Evidence-tier ↔ LR+ mapping (Tavtigian 2020 Table 2):
    Supporting (+1/-1)  : LR+ = C^(1/8)  / C^(-1/8)
    Moderate   (+2/-2)  : LR+ = C^(2/8)  / C^(-2/8)
    Strong     (+4/-4)  : LR+ = C^(4/8)  / C^(-4/8)
    Very Strong(+8/-8)  : LR+ = C^(8/8)  / C^(-8/8)

Intermediate integer codes (3,5,6,7) used by the pipeline map to C^(k/8).

Classification thresholds (Tavtigian 2020 Table 3) — derived for prior=0.10:
    Pathogenic     : total points ≥ 10  →  OP = C^(10/8)
    Likely Pathog. : 6 ≤ points ≤ 9    →  OP ∈ [C^(6/8), C^(9/8))
    VUS            : 0 ≤ points ≤ 5
    Likely Benign  : -6 ≤ points ≤ -1  →  OP ∈ (C^(-6/8), C^(-1/8)]
    Benign         : points ≤ -7       →  OP = C^(-7/8)

Key failure modes exposed by the simulations
---------------------------------------------
1. Because C* changes with prior, the fixed point thresholds no longer
   deliver the promised posteriors (≥0.99 for P, ≥0.90 for LP, etc.)
   when the prior deviates from 0.10.
2. C* has a minimum at prior ≈ 0.25 (absolute minimum = 81 from the paper)
   and rises as the prior moves away in either direction — meaning that
   LR+ thresholds become more stringent (higher for pathogenic, lower for
   benign) for priors far from the minimum.
3. Two ACMG rules are permanently inconsistent regardless of (prior, C*):
   Path(ii) [2 strong] yields Post_P = 0.975 (LP range, not P) and
   LP(i) [1 very-strong + 1 moderate] yields Post_P = 0.994 (P range).
"""

import sys, os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from assay_calibration.fit_utils.evidence_thresholds import (
    get_tavtigian_constant,
    pathogenicRulesPosterior,
    likelyPathogenicRulesPosterior,
    benignRulesPosterior,
    likelybenignRulesPosterior,
    locallrPlus2Posterior,
)

# Evidence codes used by the pipeline (1..8, negatives for benign).
POINT_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]

# The four "named" ACMG tiers and the classification boundary points.
ACMG_TIER_CODES = [1, 2, 4, 8]

# Classification boundary total-point values (Tavtigian 2020, Table 3).
# Posteriors at these boundaries should satisfy the targets below.
CLASSIFICATION_BOUNDARIES: Dict[str, int] = {
    "P_min":  10,   # ≥ 10 pts → Pathogenic;     target posterior ≥ 0.99
    "LP_min":  6,   # ≥  6 pts → Likely Pathog.; target posterior ≥ 0.90
    "LP_max":  9,   # <  9 pts → not yet P;       target posterior < 0.99
    "LB_max": -1,   # ≤ -1 pts → Likely Benign;  target posterior < 0.10
    "LB_min": -6,   # ≥ -6 pts → not yet B;       target posterior > 0.01
    "B_max":  -7,   # ≤ -7 pts → Benign;          target posterior < 0.01
}

POSTERIOR_TARGETS: Dict[str, tuple] = {
    "P_min":  (0.99,   "≥"),
    "LP_min": (0.90,   "≥"),
    "LP_max": (0.99,   "<"),
    "LB_max": (0.10,   "<"),
    "LB_min": (0.01,   ">"),
    "B_max":  (0.01,   "<"),
}


@dataclass
class TavtigianResult:
    """All quantities derived from the Tavtigian framework for a single prior."""

    prior: float
    C_star: int
    original: bool
    strict: bool

    # ── per-code LR+ thresholds (codes ±1..±8) ──────────────────────────────
    # LR+ = C^(|code|/8), benign codes have LR+ < 1.
    lr_threshold:     Dict[int, float] = field(default_factory=dict)
    log_lr_threshold: Dict[int, float] = field(default_factory=dict)   # ln(LR+)
    log10_lr_threshold: Dict[int, float] = field(default_factory=dict) # log10(LR+)
    # Posterior if a variant's LR+ equals *exactly* that tier's threshold.
    posterior_at_tier: Dict[int, float] = field(default_factory=dict)

    # ── classification boundary posteriors ───────────────────────────────────
    # Keys are the strings in CLASSIFICATION_BOUNDARIES (e.g. "P_min", "LP_min"…)
    # These are the posteriors at the combined-evidence thresholds, e.g.
    # P_min: posterior when total points = 10, OP = C^(10/8).
    # This tests whether the fixed Tavtigian-2020 point scale gives the
    # correct posterior at priors other than 0.10.
    boundary_lr:         Dict[str, float] = field(default_factory=dict)
    boundary_posterior:  Dict[str, float] = field(default_factory=dict)
    boundary_passes:     Dict[str, bool]  = field(default_factory=dict)

    # ── combining-rule posteriors ────────────────────────────────────────────
    pathogenic_posteriors:    np.ndarray = field(default_factory=lambda: np.array([]))
    likely_path_posteriors:   np.ndarray = field(default_factory=lambda: np.array([]))
    benign_posteriors:        np.ndarray = field(default_factory=lambda: np.array([]))
    likely_benign_posteriors: np.ndarray = field(default_factory=lambda: np.array([]))

    # ── fail counts ─────────────────────────────────────────────────────────
    path_fails:   int = 0
    lp_fails:     int = 0
    benign_fails: int = 0
    lb_fails:     int = 0
    total_fails:  int = 0


def run_simulation(
    prior: float,
    original: bool = False,
    strict: bool = False,
    C_max: Optional[int] = None,
) -> TavtigianResult:
    """
    Run the Tavtigian framework for *prior* and return a fully populated
    TavtigianResult.

    Parameters
    ----------
    prior : float
        Prior probability of pathogenicity (0 < prior < 1).
    original : bool
        Use original ACMG combining rules (2 strong → P).
        False = Tavtigian 2018 modified rules (PVS1 + 1 moderate → P).
    strict : bool
        Enforce strict LP/LB bound (LP posterior must also stay < 0.99).
    C_max : int or None
        Upper bound for the integer grid search of C*.
        None (default) defers to get_tavtigian_constant's own default (100 000),
        matching thresholds_from_prior in fit.py.
        Pass 350 to reproduce the paper's primary example exactly.
    """
    kwargs = dict(original=original, strict=strict)
    if C_max is not None:
        kwargs["C_max"] = C_max
    C_star = get_tavtigian_constant(prior, **kwargs)
    r = TavtigianResult(prior=prior, C_star=C_star, original=original, strict=strict)

    n_pv = len(POINT_VALUES)  # 8 — the divisor used in thresholds_from_prior

    # ── LR+ thresholds for all pipeline evidence codes ───────────────────────
    for code in range(1, n_pv + 1):
        for signed_code in (code, -code):
            exp = signed_code / n_pv
            lr = float(C_star ** exp)
            r.lr_threshold[signed_code]       = lr
            r.log_lr_threshold[signed_code]   = float(np.log(lr))
            r.log10_lr_threshold[signed_code] = float(np.log10(lr)) if lr > 0 else float("-inf")
            r.posterior_at_tier[signed_code]  = locallrPlus2Posterior(lr, prior)

    # ── classification boundary posteriors ───────────────────────────────────
    for name, pts in CLASSIFICATION_BOUNDARIES.items():
        lr = float(C_star ** (pts / n_pv))
        post = locallrPlus2Posterior(lr, prior)
        r.boundary_lr[name]        = lr
        r.boundary_posterior[name] = post
        target, op = POSTERIOR_TARGETS[name]
        if op == "≥":
            r.boundary_passes[name] = post >= target
        else:
            r.boundary_passes[name] = post < target

    # ── combining-rule posteriors ────────────────────────────────────────────
    r.pathogenic_posteriors    = pathogenicRulesPosterior(C_star, prior, original)
    r.likely_path_posteriors   = likelyPathogenicRulesPosterior(C_star, prior, original)
    r.benign_posteriors        = benignRulesPosterior(C_star, prior)
    r.likely_benign_posteriors = likelybenignRulesPosterior(C_star, prior, original)

    # ── fail counts ─────────────────────────────────────────────────────────
    r.path_fails = int(np.sum(r.pathogenic_posteriors < 0.99))
    lp_mask = r.likely_path_posteriors < 0.90
    if strict:
        lp_mask = lp_mask | (r.likely_path_posteriors > 0.99)
    r.lp_fails     = int(np.sum(lp_mask))
    r.benign_fails = int(np.sum(r.benign_posteriors > 0.01))
    lb_mask = r.likely_benign_posteriors > 0.10
    if strict:
        lb_mask = lb_mask | (r.likely_benign_posteriors < 0.01)
    r.lb_fails    = int(np.sum(lb_mask))
    r.total_fails = r.path_fails + r.lp_fails + r.benign_fails + r.lb_fails
    return r
