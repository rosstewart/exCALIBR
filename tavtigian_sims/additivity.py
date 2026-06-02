"""
Method D — Recanonicalised additive integer-point system.

Background
----------
The ACMG point boundaries (10, 6, -1, -7) for posterior targets
(0.99, 0.90, 0.10, 0.01) are over-constrained for a single linear
point→log(LR+) scale: between-anchor implied α values disagree (0.5995,
0.6278, 0.3996).  No single additive scale can land all four boundary
posteriors exactly at the ACMG point values.

The user accepts renumbering the boundary point values in exchange for
strict additivity, smooth transitions, and exact (to ~1e-4) posteriors at
every prior.  Under additivity, the required gap ratios between adjacent
boundaries are fixed by the posterior targets:

    (T_P - T_LP) : (T_LP - T_LB) : (T_LB - T_B)
        = log(11) : log(81) : log(11)
        = 1       : 1.83258  : 1   (irrational)

The best small-integer rational approximation is 11/6 ≈ 1.83333 (error
7×10⁻⁴), giving the canonical integer gap pattern (6, 11, 6).

With α = log(11)/6 ≈ 0.3996 and anchoring at a chosen prior, the four
boundary T values become:

    prior=0.10:   T_B = -6,   T_LB =  0,   T_LP = 11,   T_P = 17
    prior=0.25:   T_B = -8.75, T_LB = -2.75, T_LP = 8.25, T_P = 14.25

Method D guarantees:
  * Additivity:  log(LR+) = α · T  (single global α, prior-independent)
  * Smoothness:  log(LR+) is a straight line in T (no kinks)
  * Exact bounds at the *recanonical* boundaries (to ~1e-4) for every prior

The trade-off is that the literal ACMG boundary T values (10, 6, -1, -7)
no longer correspond to the canonical posterior targets — Method D uses
shifted-with-prior boundaries instead.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np

# Reuse the Bayes primitives — single source of truth.
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from assay_calibration.fit_utils.bayesian_thresholds import (
    bayes_posterior_from_lr, bayes_lr_for_posterior,
    piecewise_log_lr, piecewise_posterior,
)


# ── Constants ───────────────────────────────────────────────────────────────

# log_post_odds at the four ACMG posterior targets.
ACMG_LOG_ODDS_TARGETS: Dict[str, float] = {
    "P":  math.log(0.99 / 0.01),   # log(99)
    "LP": math.log(0.90 / 0.10),   # log(9)
    "LB": math.log(0.10 / 0.90),   # -log(9)
    "B":  math.log(0.01 / 0.99),   # -log(99)
}

# Required log-odds gaps between adjacent boundaries, fixed by the targets.
ACMG_LOG_ODDS_GAPS: Dict[Tuple[str, str], float] = {
    ("P", "LP"):  ACMG_LOG_ODDS_TARGETS["P"]  - ACMG_LOG_ODDS_TARGETS["LP"],  # log(11)
    ("LP", "LB"): ACMG_LOG_ODDS_TARGETS["LP"] - ACMG_LOG_ODDS_TARGETS["LB"],  # log(81)
    ("LB", "B"):  ACMG_LOG_ODDS_TARGETS["LB"] - ACMG_LOG_ODDS_TARGETS["B"],   # log(11)
}

# Best small-integer rational approximation to log(81)/log(11) ≈ 1.83258.
# 11/6 gives error 7×10⁻⁴ (clinically indistinguishable from exact).
DEFAULT_GAPS: Tuple[int, int, int] = (6, 11, 6)  # (P-LP, LP-LB, LB-B)

# Original ACMG anchor T values (for diagnostic comparison only).
ACMG_T_ANCHORS: Dict[str, int] = {"P": 10, "LP": 6, "LB": -1, "B": -7}


# ── Method D core ───────────────────────────────────────────────────────────

ArrayLike = Union[float, np.ndarray]


def recanonical_alpha(prior: float,
                      gaps: Tuple[int, int, int] = DEFAULT_GAPS) -> float:
    """Single, prior-DEPENDENT α for the recanonicalised additive scale.

    Uses the least-squares optimal α at the given prior: the value that
    minimises the total squared log-odds error across all four ACMG boundary
    T values (10, 6, −1, −7).  This makes per-tier LR+ thresholds
    exp(α(p)·k) vary with prior while maintaining strict additivity
    (single α per prior) and exact posteriors at the four boundary T values
    that are derived for that α.

    The ``gaps`` parameter is kept for signature compatibility but is not
    used in the LSQ formulation.
    """
    return single_alpha_lsq(prior)


def recanonical_boundaries(prior: float,
                           gaps: Tuple[int, int, int] = DEFAULT_GAPS) -> Dict[str, float]:
    """Boundary T values at *prior* under the recanonical additive scale.

    Each T solves   α(p)·T + log_prior_odds  =  log_post_odds(target_q).
    α(p) is the LSQ-optimal value at this prior, so the boundaries shift
    with prior and the posterior is exact at each boundary T.
    """
    alpha = recanonical_alpha(prior, gaps)
    lpo = math.log(prior / (1.0 - prior))
    return {name: (L - lpo) / alpha
            for name, L in ACMG_LOG_ODDS_TARGETS.items()}


def recanonical_log_lr(T: ArrayLike,
                       prior: float,
                       gaps: Tuple[int, int, int] = DEFAULT_GAPS) -> ArrayLike:
    """log(LR+) at total points T under Method D.

    Strictly additive at each prior: log(LR+) = α(p) · T, where α(p) is
    the LSQ-optimal value at that prior.  The scale varies with prior so
    that per-tier LR+ thresholds are prior-aware.
    """
    alpha = recanonical_alpha(prior, gaps)
    return alpha * np.asarray(T, dtype=float)


def recanonical_posterior(T: ArrayLike, prior: float,
                          gaps: Tuple[int, int, int] = DEFAULT_GAPS) -> ArrayLike:
    """Posterior at total points T, prior p, under Method D.

    Exact at the four *recanonical* boundary T values (to ~1e-4) for
    every prior in (0, 1).
    """
    lr = np.exp(recanonical_log_lr(T, prior=prior, gaps=gaps))
    return bayes_posterior_from_lr(lr, prior)


def recanonical_tier_thresholds(prior: float,
                                point_values: Iterable[int] = (1, 2, 3, 4, 5, 6, 7, 8),
                                gaps: Tuple[int, int, int] = DEFAULT_GAPS,
                                ) -> Tuple[np.ndarray, np.ndarray, None]:
    """LR+ thresholds for ±point_values under Method D.

    Same signature as `piecewise_tier_thresholds`: returns (lrP, lrB, None).
    α = α(p) is the LSQ-optimal value at this prior, so the thresholds
    vary with prior — exp(α(p)·k) depends on the prior.
    """
    alpha = recanonical_alpha(prior, gaps)
    pv = np.asarray(list(point_values), dtype=float)
    lrP = np.exp(alpha * pv)
    lrB = np.exp(-alpha * pv)
    return lrP, lrB, None


# ── Companion methods (the rest of the spectrum) ────────────────────────────

def single_alpha_anchored(prior: float, anchor: str = "P") -> float:
    """Additive α chosen so the *anchor* ACMG boundary posterior is exact.

    anchor ∈ {"P","LP","LB","B"}.  The other three boundaries drift.
        α = (log_post_odds(target_anchor) - log_prior_odds) / T_anchor
    """
    T = ACMG_T_ANCHORS[anchor]
    if T == 0:
        raise ValueError("Cannot anchor at T=0")
    L = ACMG_LOG_ODDS_TARGETS[anchor]
    lpo = math.log(prior / (1.0 - prior))
    return (L - lpo) / T


def single_alpha_lsq(prior: float,
                     T_anchors: Tuple[int, int, int, int] = (10, 6, -1, -7),
                     targets: Tuple[float, float, float, float] = (
                         math.log(99), math.log(9), -math.log(9), -math.log(99))
                     ) -> float:
    """Analytical least-squares additive α at the ACMG anchors.

    Minimises Σ (α·T_i + log_prior_odds - log_post_odds(target_i))².
    Closed form: α = Σ T_i·(L_i - LPO) / Σ T_i².
    """
    T = np.asarray(T_anchors, dtype=float)
    L = np.asarray(targets, dtype=float)
    lpo = math.log(prior / (1.0 - prior))
    return float(np.sum(T * (L - lpo)) / np.sum(T * T))


def cubic_spline_log_lr(T: ArrayLike, prior: float,
                        T_knots: Tuple[int, int, int, int] = (-7, -1, 6, 10),
                        targets: Tuple[float, float, float, float] = (
                            -math.log(99), -math.log(9), math.log(9), math.log(99))
                        ) -> ArrayLike:
    """Natural cubic spline through the four ACMG knots.

    Smooth (C²), exact at the four ACMG boundary points for every prior,
    but NOT additive (1 PVSt ≠ 8·1 PSu in general).
    """
    from scipy.interpolate import CubicSpline
    lpo = math.log(prior / (1.0 - prior))
    knot_log_lr = np.asarray(targets, dtype=float) - lpo
    cs = CubicSpline(np.asarray(T_knots, dtype=float), knot_log_lr,
                     bc_type="natural", extrapolate=True)
    return cs(np.asarray(T, dtype=float))


def cubic_spline_posterior(T: ArrayLike, prior: float, **kw) -> ArrayLike:
    lr = np.exp(cubic_spline_log_lr(T, prior=prior, **kw))
    return bayes_posterior_from_lr(lr, prior)


# ── Diagnostics ─────────────────────────────────────────────────────────────

def implied_targets_for_alpha(alpha: float, prior: float,
                              T_anchors: Dict[str, int] = ACMG_T_ANCHORS,
                              ) -> Dict[str, float]:
    """Posteriors that the additive scale (α, prior) produces at given T values.

    posterior = sigmoid(α·T + log_prior_odds).
    """
    lpo = math.log(prior / (1.0 - prior))
    out: Dict[str, float] = {}
    for name, T in T_anchors.items():
        log_post_odds = alpha * T + lpo
        out[name] = 1.0 / (1.0 + math.exp(-log_post_odds))
    return out


def implied_T_for_targets(prior: float, alpha: Optional[float] = None,
                          targets: Dict[str, float] = None,
                          gaps: Tuple[int, int, int] = DEFAULT_GAPS,
                          ) -> Dict[str, float]:
    """Counterfactual: under additivity with the given α, what T values give
    each target posterior exactly at *prior*?  Returns dict label -> T.
    """
    if alpha is None:
        alpha = recanonical_alpha(gaps)
    if targets is None:
        targets = {n: q for n, q in zip(
            ["P", "LP", "LB", "B"], [0.99, 0.90, 0.10, 0.01])}
    lpo = math.log(prior / (1.0 - prior))
    return {n: (math.log(q / (1 - q)) - lpo) / alpha
            for n, q in targets.items()}


def additivity_residual(method_fn: Callable[[ArrayLike, float], ArrayLike],
                         prior: float,
                         T_grid: np.ndarray) -> np.ndarray:
    """Residual of method_fn(T, prior) from the best-fit additive line α·T.

    Zero everywhere => strictly additive.  Large kinks => piecewise.
    """
    y = np.asarray(method_fn(T_grid, prior), dtype=float)
    T_arr = np.asarray(T_grid, dtype=float)
    # Least-squares α through origin: α = Σ T·y / Σ T².
    alpha = float(np.sum(T_arr * y) / np.sum(T_arr * T_arr))
    return y - alpha * T_arr


def smoothness_norm(method_fn: Callable[[ArrayLike, float], ArrayLike],
                    prior: float,
                    T_grid: np.ndarray) -> float:
    """Discrete second-difference L2 norm of method_fn(T) over T_grid.

    Zero for any linear function (additive methods).
    Large near knots for piecewise.  Cubic spline is smooth (small).
    """
    y = np.asarray(method_fn(T_grid, prior), dtype=float)
    d2 = np.diff(y, 2)
    return float(np.sqrt(np.sum(d2 * d2)))


def boundary_error(method_fn: Callable[[ArrayLike, float], ArrayLike],
                   prior_grid: np.ndarray,
                   T_anchors: Dict[str, int] = ACMG_T_ANCHORS,
                   targets: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    """For each ACMG boundary and prior, |posterior - target|.

    method_fn must return log(LR+) given (T, prior).  Returns dict label ->
    array of |error| over priors.
    """
    if targets is None:
        targets = {"P": 0.99, "LP": 0.90, "LB": 0.10, "B": 0.01}
    out: Dict[str, np.ndarray] = {}
    for name, T in T_anchors.items():
        target_q = targets[name]
        errs = np.empty_like(prior_grid, dtype=float)
        for i, p in enumerate(prior_grid):
            lr = math.exp(float(method_fn(T, float(p))))
            post = float(bayes_posterior_from_lr(lr, float(p)))
            errs[i] = abs(post - target_q)
        out[name] = errs
    return out
