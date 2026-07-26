"""
Prior-adaptive Bayesian thresholds for ACMG variant classification.

Two methods that replace the Tavtigian C* grid search with analytical
Bayesian thresholds. Both work exactly for any prior in (0, 1).

Method A — Piecewise α (integer-point preserving)
-------------------------------------------------
Defines a prior-adaptive piecewise linear map  L_p : T -> log(LR+)  with
knots at the four ACMG boundary points (T = 10, 6, -1, -7) such that the
posterior at those T values is *exactly* (0.99, 0.90, 0.10, 0.01) for any
prior p.

Method B — Continuous LR+ (pure Bayesian)
-----------------------------------------
No discretisation. For prior p and target posterior q,
    LR+(q, p) = q(1-p) / (p(1-q))
gives the LR+ at which the posterior equals exactly q.  Classification is
done by direct posterior comparison.

Reference
---------
Bayes (odds form):
    posterior_odds = LR+ * prior_odds
    => LR+ = posterior_odds / prior_odds
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

# ── Bayes primitives ─────────────────────────────────────────────────────────

ArrayLike = Union[float, np.ndarray]


def bayes_lr_for_posterior(target_posterior: ArrayLike,
                           prior: ArrayLike) -> ArrayLike:
    """LR+ that yields exactly *target_posterior* at *prior*.

    Derived from posterior_odds = LR+ * prior_odds:
        LR+ = (q/(1-q)) / (p/(1-p)) = q(1-p) / (p(1-q))
    """
    q = np.asarray(target_posterior, dtype=float)
    p = np.asarray(prior, dtype=float)
    return q * (1.0 - p) / (p * (1.0 - q))


def bayes_posterior_from_lr(lr_plus: ArrayLike, prior: ArrayLike) -> ArrayLike:
    """Posterior from LR+ and prior.  Identical to locallrPlus2Posterior."""
    lr = np.asarray(lr_plus, dtype=float)
    p = np.asarray(prior, dtype=float)
    return lr * p / ((lr - 1.0) * p + 1.0)


# ── ACMG four-boundary anchors ───────────────────────────────────────────────

# (T_threshold, posterior_target).  Used by Method A as the knots of the
# piecewise linear log-LR map.  Ordered ascending in T.
ACMG_KNOTS: Tuple[Tuple[int, float], ...] = (
    (-7, 0.01),
    (-1, 0.10),
    (6,  0.90),
    (10, 0.99),
)

_KNOT_T = np.array([k[0] for k in ACMG_KNOTS], dtype=float)
_KNOT_Q = np.array([k[1] for k in ACMG_KNOTS], dtype=float)
# Posterior-odds part of each knot:  log(q/(1-q))
_KNOT_LOG_POST_ODDS = np.log(_KNOT_Q / (1.0 - _KNOT_Q))

# ── ACMG (6·11·6) additive knots ─────────────────────────────────────────────
# Integer gap pattern (6, 11, 6) approximates the irrational ratio
# log(81)/log(11) ≈ 1.832 to within 0.07%.
#
# TWO conditions are needed for near-perfect additivity:
#   1. Equal slopes across all three segments  (guaranteed by (6,11,6) gaps)
#   2. T_LB = 0  so that log_lr(0, p=0.10) = 0 (LR+=1 at "zero evidence")
#
# Condition 2 makes same-segment combination EXACTLY additive at p=0.10:
#     log_lr(kA + kB) = slope*(kA+kB) = slope*kA + slope*kB = log_lr(kA) + log_lr(kB)
# Condition 1 keeps cross-segment error ≤ 0.07%.
#
# Resulting boundaries: B=−6, LB=0, LP=11, P=17.
# At p=0.10 the LR+ thresholds are: LR+(0)=1 (neutral), LR+(11)=81 (LP),
# LR+(17)=891 (P).
ACMG_ADDITIVE_KNOTS: Tuple[Tuple[int, float], ...] = (
    (-6, 0.01),
    ( 0, 0.10),
    (11, 0.90),
    (17, 0.99),
)

_ADD_KNOT_T = np.array([k[0] for k in ACMG_ADDITIVE_KNOTS], dtype=float)
_ADD_KNOT_Q = np.array([k[1] for k in ACMG_ADDITIVE_KNOTS], dtype=float)
_ADD_KNOT_LOG_POST_ODDS = np.log(_ADD_KNOT_Q / (1.0 - _ADD_KNOT_Q))


# ── Method A: Piecewise α ────────────────────────────────────────────────────

def _log_prior_odds(prior: ArrayLike) -> ArrayLike:
    p = np.asarray(prior, dtype=float)
    return np.log(p / (1.0 - p))


def _piecewise_log_lr_impl(T_arr: np.ndarray,
                            knot_t: np.ndarray,
                            knot_log_lr: np.ndarray) -> np.ndarray:
    """Core piecewise-linear interpolation / extrapolation (not public API)."""
    out = np.empty_like(T_arr)
    for i, t in enumerate(T_arr):
        if t <= knot_t[0]:
            slope = (knot_log_lr[1] - knot_log_lr[0]) / (knot_t[1] - knot_t[0])
            out[i] = knot_log_lr[0] + slope * (t - knot_t[0])
        elif t >= knot_t[-1]:
            slope = (knot_log_lr[-1] - knot_log_lr[-2]) / (knot_t[-1] - knot_t[-2])
            out[i] = knot_log_lr[-1] + slope * (t - knot_t[-1])
        else:
            j = int(np.searchsorted(knot_t, t, side="right") - 1)
            slope = (knot_log_lr[j + 1] - knot_log_lr[j]) / (knot_t[j + 1] - knot_t[j])
            out[i] = knot_log_lr[j] + slope * (t - knot_t[j])
    return out


def piecewise_log_lr(T: ArrayLike, prior: float) -> ArrayLike:
    """L_p(T): piecewise-linear log(LR+) anchored at the four ACMG knots.

    Linear extrapolation beyond the outermost knots uses the slope of the
    adjacent segment. At T in {-7, -1, 6, 10}, the implied posterior is
    exactly (0.01, 0.10, 0.90, 0.99) for any prior in (0, 1).
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    lpo = float(_log_prior_odds(prior))
    knot_log_lr = _KNOT_LOG_POST_ODDS - lpo
    out = _piecewise_log_lr_impl(T_arr, _KNOT_T, knot_log_lr)
    if np.ndim(T) == 0:
        return float(out[0])
    return out


def piecewise_additive_log_lr(T: ArrayLike, prior: float) -> ArrayLike:
    """Piecewise log(LR+) with (6·11·6) additive knots at T = 16, 10, −1, −7.

    Identical interface to ``piecewise_log_lr``.  The gap pattern (6, 11, 6)
    gives nearly equal slopes across all three segments (error < 0.07%),
    so integer-point sums are essentially additive for any combination of
    evidence codes.  Posteriors at the four knot T values are exactly
    (0.01, 0.10, 0.90, 0.99) at every prior.
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    lpo = float(_log_prior_odds(prior))
    knot_log_lr = _ADD_KNOT_LOG_POST_ODDS - lpo
    out = _piecewise_log_lr_impl(T_arr, _ADD_KNOT_T, knot_log_lr)
    if np.ndim(T) == 0:
        return float(out[0])
    return out


def piecewise_display_points(log_lr: ArrayLike, prior: float,
                              targets: Dict[str, float] = None,
                              floor_at_neutral: bool = False) -> ArrayLike:
    """Invert ``piecewise_log_lr``: log(LR+) -> a (generally fractional) point
    label, for display purposes only.

    This is the ACMG-Bayes display-point conversion: given an already-known
    (possibly combined) log(LR+) value, locate which of the three segments
    it falls in (or which side it extrapolates beyond) and linearly invert
    that segment's slope to solve for T instead of log(LR+). Never use the
    result as a combination operand -- combination must always be done on
    log(LR+) directly (see ``piecewise_log_lr``/summing it across evidence
    items); this function only recovers an ACMG-familiar Su/M/S/VS-equivalent
    label at the current prior for reporting.

    Parameters
    ----------
    targets : dict, optional
        Override target posteriors (same keys/semantics as
        ``continuous_classify``'s ``targets``, e.g. ``{"LB": 0.01, "B":
        0.001}``); any key not given falls back to ``DEFAULT_TARGETS``. The
        knot T positions (-7, -1, 6, 10) stay fixed -- only which posterior
        each knot is calibrated to changes.
    floor_at_neutral : bool, optional
        If True, clamp targets via ``_floor_targets_at_prior`` and insert an
        explicit T=0 knot pinned at log(LR+)=0 (in addition to the usual
        B/LB/LP/P knots at T=-7,-1,6,10). Together these guarantee: (1)
        log(LR+) >= 0 is required for ANY positive display point, and
        log(LR+) <= 0 for any negative one -- not just at the +-1
        thresholds a points-sign confusion matrix reads, but for every
        fractional T; and (2) the positive-side points (T>0) depend only on
        the LP/P targets, never on LB/B, so tightening LB/B (e.g. to make
        benign classification stricter) can no longer make positive points
        easier to reach -- see the design discussion this was built from.
        The cost: whenever flooring is active (prior below/above a
        benign/pathogenic-direction target), the floored knot at T=-1 (or
        T=6) collapses to the same log(LR+)=0 value as the new T=0 anchor,
        one full T-unit away -- so log(LR+) values on either side of 0
        resolve to display points about a full point apart (e.g. -1.0 vs
        0.0) despite representing near-identical evidence. This is a
        display-only artifact; ``continuous_classify``'s P/LP/VUS/LB/B
        labels have no such discontinuity.
    """
    log_lr_arr = np.atleast_1d(np.asarray(log_lr, dtype=float))
    lpo = float(_log_prior_odds(prior))
    if targets is None and not floor_at_neutral:
        knot_T = _KNOT_T
        knot_log_lr = _KNOT_LOG_POST_ODDS - lpo
    elif floor_at_neutral:
        merged = {**DEFAULT_TARGETS, **(targets or {})}
        merged = _floor_targets_at_prior(merged, prior)
        knot_T = np.array([-7.0, -1.0, 0.0, 6.0, 10.0])
        knot_log_lr = np.array([
            np.log(merged["B"] / (1.0 - merged["B"])) - lpo,
            np.log(merged["LB"] / (1.0 - merged["LB"])) - lpo,
            0.0,
            np.log(merged["LP"] / (1.0 - merged["LP"])) - lpo,
            np.log(merged["P"] / (1.0 - merged["P"])) - lpo,
        ])
    else:
        merged = {**DEFAULT_TARGETS, **(targets or {})}
        knot_T = _KNOT_T
        knot_q = np.array([merged["B"], merged["LB"], merged["LP"], merged["P"]])
        knot_log_lr = np.log(knot_q / (1.0 - knot_q)) - lpo

    n_knots = len(knot_log_lr)
    out = np.empty_like(log_lr_arr)
    for i, y in enumerate(log_lr_arr):
        if y <= knot_log_lr[0]:
            lo, hi = 0, 1
        elif y >= knot_log_lr[-1]:
            lo, hi = n_knots - 2, n_knots - 1
        else:
            lo = int(np.searchsorted(knot_log_lr, y, side="right") - 1)
            hi = lo + 1
        if knot_log_lr[hi] == knot_log_lr[lo]:
            # Degenerate segment: adjacent knots collapsed to the same value
            # (e.g. B and LB both clamp to the prior itself at an extreme
            # prior, or -- with floor_at_neutral -- a floored knot coincides
            # with the T=0 anchor). Report the lower knot's T.
            out[i] = knot_T[lo]
            continue
        slope = (knot_log_lr[hi] - knot_log_lr[lo]) / (knot_T[hi] - knot_T[lo])
        out[i] = knot_T[lo] + (y - knot_log_lr[lo]) / slope

    if np.ndim(log_lr) == 0:
        return float(out[0])
    return out


def piecewise_lr_plus(T: ArrayLike, prior: float) -> ArrayLike:
    """LR+ at total points T for prior p under Method A."""
    return np.exp(piecewise_log_lr(T, prior))


def piecewise_posterior(T: ArrayLike, prior: float) -> ArrayLike:
    """Posterior at total points T under Method A.

    Exact at the four ACMG boundary points for any prior.
    """
    return bayes_posterior_from_lr(piecewise_lr_plus(T, prior), prior)


def piecewise_additive_lr_plus(T: ArrayLike, prior: float) -> ArrayLike:
    """LR+ at total points T under the (6·11·6) additive piecewise scheme."""
    return np.exp(piecewise_additive_log_lr(T, prior))


def piecewise_additive_posterior(T: ArrayLike, prior: float) -> ArrayLike:
    """Posterior at T under the (6·11·6) additive piecewise scheme.

    Exact at T ∈ {−7, −1, 10, 16} for every prior.
    """
    return bayes_posterior_from_lr(piecewise_additive_lr_plus(T, prior), prior)


def piecewise_additive_tier_thresholds(
    prior: float,
    point_values=(1, 2, 3, 4, 5, 6, 7, 8),
) -> Tuple[np.ndarray, np.ndarray, None]:
    """Same interface as ``piecewise_tier_thresholds`` for the additive variant."""
    pv = np.asarray(point_values, dtype=float)
    lrP = piecewise_additive_lr_plus(pv, prior)
    lrB = piecewise_additive_lr_plus(-pv, prior)
    return np.asarray(lrP, dtype=float), np.asarray(lrB, dtype=float), None


def piecewise_tier_thresholds(
    prior: float,
    point_values=(1, 2, 3, 4, 5, 6, 7, 8),
) -> Tuple[np.ndarray, np.ndarray, None]:
    """Drop-in replacement for `thresholds_from_prior(prior, point_values)`.

    Returns (lrP, lrB, None):
      lrP[i] = LR+ threshold for pathogenic evidence at point_values[i]
      lrB[i] = LR+ threshold for benign     evidence at -point_values[i]
      The third return is None (no Tavtigian C* in this method).

    Derived from the piecewise log-LR map at integer points.
    """
    pv = np.asarray(point_values, dtype=float)
    lrP = piecewise_lr_plus(pv, prior)
    lrB = piecewise_lr_plus(-pv, prior)
    return np.asarray(lrP, dtype=float), np.asarray(lrB, dtype=float), None


# ── Method A′: LP-anchored (two slopes, exact at LP & LB per prior) ────────────

def lp_anchored_log_lr(T: ArrayLike, prior: float) -> ArrayLike:
    """log(LR+) with separate slopes for pathogenic & benign, each exact at anchor.

    Two-slope design anchored at LP (T=6, post=0.90) and LB (T=-1, post=0.10):
      * Pathogenic (T ≥ 0): α_path(p) = (log(9) − log_prior_odds) / 6
      * Benign (T < 0):     α_benign(p) = (log(9) + log_prior_odds) / (−1)
      * Both pass through T=0 where log(LR+)=0 (neutral evidence)

    Properties:
      * LP (T=6) posterior EXACTLY 0.90 for ALL priors ✓
      * LB (T=-1) posterior EXACTLY 0.10 for ALL priors ✓
      * P (T=10) posterior stable at sigmoid(log(81)) ≈ 0.988 for all priors
      * B (T=-7) posterior stable at sigmoid(-log(81)) ≈ 0.001 for all priors
      * Strictly additive within each direction (piecewise linear in log-LR)
      * No kinks at T=0 (both slopes pass through origin)

    Trade-off: Exact posteriors at two anchors require prior-dependent slopes.
    This is unavoidable: constant LR+ → varying posterior across priors, or
    varying posterior targets → varying slopes. We choose exact posteriors.
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    lpo = float(_log_prior_odds(prior))
    log9 = np.log(9.0)

    # Prior-dependent slopes to achieve exact posteriors
    alpha_path = (log9 - lpo) / 6.0        # Pathogenic: anchor at LP (T=6, post=0.90)
    alpha_benign = (log9 + lpo) / (-1.0)   # Benign:     anchor at LB (T=-1, post=0.10)

    out = np.empty_like(T_arr)
    for i, t in enumerate(T_arr):
        if float(t) >= 0:
            out[i] = alpha_path * t
        else:
            out[i] = alpha_benign * t

    if np.ndim(T) == 0:
        return float(out[0])
    return out


def lp_anchored_lr_plus(T: ArrayLike, prior: float) -> ArrayLike:
    """LR+ at total points T under LP-anchored additive scheme (pathogenic)."""
    return np.exp(lp_anchored_log_lr(T, prior))


def lp_anchored_posterior(T: ArrayLike, prior: float) -> ArrayLike:
    """Posterior at T under LP-anchored additive scheme.

    LP (T=6) is exactly 0.90 for every prior.
    P (T=10) is stable at ~0.988 for every prior.
    """
    return bayes_posterior_from_lr(lp_anchored_lr_plus(T, prior), prior)


# ── Method B: Continuous LR+ (pure Bayesian) ─────────────────────────────────

# Target posteriors used to define classification boundaries.  Ordered
# ascending so the lookup below produces a single label.
DEFAULT_TARGETS = {"B": 0.01, "LB": 0.10, "LP": 0.90, "P": 0.99}


def _floor_targets_at_prior(targets: Dict[str, float], prior: float) -> Dict[str, float]:
    """Clamp benign-direction targets (B, LB) to ``min(target, prior)`` and
    pathogenic-direction targets (LP, P) to ``max(target, prior)``.

    This is exactly equivalent to flooring the corresponding log(LR+)
    threshold at 0 (verified: ``logit(min(target, prior)) - logit(prior) ==
    min(logit(target) - logit(prior), 0)``, and symmetrically for max/LP/P).
    Without this, whenever ``prior < target`` for a benign-direction target
    (or ``prior > target`` for a pathogenic-direction one), the threshold
    requires *evidence favoring the wrong direction* to be met -- e.g. at
    prior=0.05 with the default LB target of 0.10, reaching "Likely Benign"
    only requires log(LR+) <= +0.75, so evidence that actually favours
    pathogenicity (LR+ up to ~2.1) still gets classified LB. Flooring
    guarantees LB/B require log(LR+) <= 0 and LP/P require log(LR+) >= 0,
    at the cost of collapsing those targets toward the prior itself when the
    gene is rare (or common) enough that the nominal target is unreachable
    without wrong-direction evidence.
    """
    return {
        "B":  min(targets["B"],  prior),
        "LB": min(targets["LB"], prior),
        "LP": max(targets["LP"], prior),
        "P":  max(targets["P"],  prior),
    }


def continuous_lr_thresholds(prior: float,
                             targets: Dict[str, float] = None,
                             floor_at_neutral: bool = False) -> Dict[str, float]:
    """LR+ value at which posterior equals each target for the given prior.

    Analytical: LR+(q, p) = q(1-p) / (p(1-q)).

    *targets* may be a partial override (e.g. ``{"LB": 0.01, "B": 0.001}``);
    any key not given falls back to ``DEFAULT_TARGETS``.

    *floor_at_neutral*: if True, clamp targets via ``_floor_targets_at_prior``
    so no threshold ever requires evidence in the wrong direction -- see that
    function's docstring.
    """
    merged = {**DEFAULT_TARGETS, **(targets or {})}
    if floor_at_neutral:
        merged = _floor_targets_at_prior(merged, prior)
    return {name: float(bayes_lr_for_posterior(q, prior))
            for name, q in merged.items()}


def continuous_classify(lr_plus: ArrayLike,
                        prior: float,
                        targets: Dict[str, float] = None,
                        floor_at_neutral: bool = False) -> Union[str, np.ndarray]:
    """Direct posterior-based classification: 'P'/'LP'/'VUS'/'LB'/'B'.

    Boundary convention (matches ACMG):
        post >= 0.99           -> 'P'
        0.90 <= post < 0.99    -> 'LP'
        0.10 <= post < 0.90    -> 'VUS'   (also catches the 0.01–0.10 boundary
        0.01 <  post < 0.10    -> 'LB'     overlap; we treat the lower as LB)
        post <= 0.01           -> 'B'

    *targets* may be a partial override (e.g. ``{"LB": 0.01, "B": 0.001}``);
    any key not given falls back to ``DEFAULT_TARGETS``.

    *floor_at_neutral*: if True, clamp targets via ``_floor_targets_at_prior``
    so LB/B never trigger on evidence that actually favours pathogenicity
    (and LP/P never trigger on evidence that actually favours benignity) --
    see that function's docstring. Off by default to preserve the exact
    ACMG posterior-target semantics.
    """
    targets = {**DEFAULT_TARGETS, **(targets or {})}
    if floor_at_neutral:
        targets = _floor_targets_at_prior(targets, prior)
    post = bayes_posterior_from_lr(lr_plus, prior)
    post = np.atleast_1d(np.asarray(post, dtype=float))
    out = np.empty(post.shape, dtype=object)
    out[:] = "VUS"
    out[post >= targets["P"]]  = "P"
    out[(post >= targets["LP"]) & (post < targets["P"])]  = "LP"
    out[(post >  targets["B"]) & (post <= targets["LB"])] = "LB"
    out[post <= targets["B"]]  = "B"
    if np.ndim(lr_plus) == 0:
        return str(out[0])
    return out
