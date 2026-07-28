"""
Simulation suites for the ACMG-Bayes replacement method.

ACMGBayesSuite            : per-code / per-boundary LR+ thresholds, anchored
                             to the exact analytic Bayes threshold at each of
                             the four ACMG posterior targets, for every prior.
ACMGBayesCombinationSuite : multi-evidence combination — per-code log(LR+)
                             values are summed directly (never rounded to an
                             integer point first), so combination error is
                             zero by construction. Subsumes what earlier
                             drafts called "Continuous" (the trivial
                             single-item case: no code interpolation needed
                             when you already have a continuous LR+) and
                             "Ledger" (the multi-item log-LR summation).

Historical variants (`PiecewiseAdditiveSuite`, `LPAnchoredSuite`) are kept
for reference but are not part of the primary Tavtigian-vs-ACMG-Bayes
comparison — see compare.py.

Both primary suites expose a DataFrame interface compatible with the
plotting code in ``compare.py`` so they can be overlaid on the legacy
Tavtigian suite.
"""

from __future__ import annotations

import sys, os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from assay_calibration.fit_utils.bayesian_thresholds import (
    piecewise_lr_plus, piecewise_posterior, piecewise_tier_thresholds,
    piecewise_log_lr,
    piecewise_additive_lr_plus, piecewise_additive_posterior,
    lp_anchored_lr_plus, lp_anchored_posterior,
    continuous_lr_thresholds, bayes_posterior_from_lr,
    ACMG_KNOTS, ACMG_ADDITIVE_KNOTS,
)

from .core import POINT_VALUES, CLASSIFICATION_BOUNDARIES, POSTERIOR_TARGETS

# Classification boundaries for the (6·11·6) additive piecewise variant.
# Same B and LB thresholds as standard ACMG; LP and P shifted to 10 and 16.
ADDITIVE_CLASSIFICATION_BOUNDARIES: dict = {
    "P_min":  17,
    "LP_min": 11,
    "LP_max": 16,   # < 17
    "LB_max":  0,
    "LB_min": -5,   # > -6
    "B_max":  -6,
}

# Posterior targets for the additive variant (same values as standard ACMG).
ADDITIVE_POSTERIOR_TARGETS: dict = {
    "P_min":  (0.99,   "≥"),
    "LP_min": (0.90,   "≥"),
    "LP_max": (0.99,   "<"),
    "LB_max": (0.10,   "<"),
    "LB_min": (0.10,   ">"),
    "B_max":  (0.01,   "<"),
}


# ── Generic piecewise-anchored base ──────────────────────────────────────────

@dataclass
class PiecewiseResult:
    prior: float
    # Per-code LR+ thresholds (codes ±1..±8)
    lr_threshold:       Dict[int, float] = field(default_factory=dict)
    log10_lr_threshold: Dict[int, float] = field(default_factory=dict)
    posterior_at_tier:  Dict[int, float] = field(default_factory=dict)
    # Classification boundary posteriors (exact at the method's own knots)
    boundary_lr:        Dict[str, float] = field(default_factory=dict)
    boundary_posterior: Dict[str, float] = field(default_factory=dict)
    boundary_passes:    Dict[str, bool]  = field(default_factory=dict)


class _PiecewiseSuiteBase:
    """Shared machinery for piecewise-anchored suites."""

    # Subclasses override these three:
    _lr_fn    = staticmethod(piecewise_lr_plus)
    _post_fn  = staticmethod(piecewise_posterior)
    _BOUNDARIES = CLASSIFICATION_BOUNDARIES
    _POST_TARGETS = POSTERIOR_TARGETS
    _METHOD   = "acmg_bayes"

    def __init__(self, priors: np.ndarray):
        self.priors = np.asarray(priors)
        self.results: List[PiecewiseResult] = []

    def _run_one(self, prior: float) -> PiecewiseResult:
        r = PiecewiseResult(prior=prior)
        for k in range(1, 9):
            for signed in (k, -k):
                lr   = float(self._lr_fn(signed, prior))
                r.lr_threshold[signed]       = lr
                r.log10_lr_threshold[signed] = float(np.log10(lr)) if lr > 0 else float("-inf")
                r.posterior_at_tier[signed]  = float(self._post_fn(signed, prior))
        for name, T in self._BOUNDARIES.items():
            lr   = float(self._lr_fn(T, prior))
            post = float(self._post_fn(T, prior))
            r.boundary_lr[name]        = lr
            r.boundary_posterior[name] = post
            target, op = self._POST_TARGETS[name]
            r.boundary_passes[name] = (post >= target) if op == "≥" else (post < target)
        return r

    def run(self) -> "_PiecewiseSuiteBase":
        self.results = [self._run_one(float(p)) for p in self.priors]
        return self

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {"prior": r.prior, "acmg_mapping_method": self._METHOD}
            for k in POINT_VALUES:
                row[f"lr_p{k}"]       = r.lr_threshold[k]
                row[f"log10_lr_p{k}"] = r.log10_lr_threshold[k]
                row[f"post_p{k}"]     = r.posterior_at_tier[k]
                row[f"lr_b{k}"]       = r.lr_threshold[-k]
                row[f"log10_lr_b{k}"] = r.log10_lr_threshold[-k]
                row[f"post_b{k}"]     = r.posterior_at_tier[-k]
            for name in self._BOUNDARIES:
                row[f"bnd_lr_{name}"]   = r.boundary_lr[name]
                row[f"bnd_post_{name}"] = r.boundary_posterior[name]
                row[f"bnd_pass_{name}"] = r.boundary_passes[name]
            rows.append(row)
        return pd.DataFrame(rows)


# ── ACMG-Bayes: exact per-code/boundary thresholds ───────────────────────────

class ACMGBayesSuite(_PiecewiseSuiteBase):
    """Per-code and per-boundary LR+ thresholds, exact at every prior.

    Same interface as the legacy Tavtigian suite (`SimulationSuite`), so the
    two can be overlaid directly. Boundary posteriors are exact by
    construction at every prior (unlike Tavtigian, which is exact only at
    p=0.10). Combining multiple pieces of evidence with this suite's
    per-code thresholds must be done by summing `log(LR+)` values directly —
    see `ACMGBayesCombinationSuite` / `acmg_bayes_log_lr` below — never by
    summing points and re-deriving a threshold at the summed total, which
    reintroduces combination error (quantified in the report's Table 3/4 as
    the cost of the "naive" points-then-remap approach).
    """
    _lr_fn       = staticmethod(piecewise_lr_plus)
    _post_fn     = staticmethod(piecewise_posterior)
    _BOUNDARIES  = CLASSIFICATION_BOUNDARIES
    _POST_TARGETS = POSTERIOR_TARGETS
    _METHOD      = "acmg_bayes"


# ── Historical variant: (6·11·6) additive piecewise ──────────────────────────
# Kept for reference; not part of the primary Tavtigian-vs-ACMG-Bayes
# comparison (see compare.py).

class PiecewiseAdditiveSuite(_PiecewiseSuiteBase):
    """Piecewise-α with (6·11·6) gap-pattern knots at T = 17, 11, 0, −6.

    Two conditions for near-perfect additivity under naive points-then-remap
    combination:
      1. Equal slopes (gap pattern 6·11·6): cross-segment error < 0.07 %.
      2. T_LB = 0: log_lr(0, p=0.10) = 0, making same-segment combination
         exactly additive at the canonical prior.

    Posteriors at the four knot T values are exact for every prior.
    Classification boundaries: B=−6, LB=0, LP=11, P=17.
    Superseded by ACMG-Bayes combination (log-LR summation), which has zero
    combination error under the *standard* ACMG knots without needing a
    special gap pattern.
    """
    _lr_fn        = staticmethod(piecewise_additive_lr_plus)
    _post_fn      = staticmethod(piecewise_additive_posterior)
    _BOUNDARIES   = ADDITIVE_CLASSIFICATION_BOUNDARIES
    _POST_TARGETS = ADDITIVE_POSTERIOR_TARGETS
    _METHOD       = "piecewise_additive"


# ── Historical variant: LP-anchored additive (pathogenic direction only) ─────

class LPAnchoredSuite(_PiecewiseSuiteBase):
    """[DEPRECATED] LP & LB anchored with separate prior-dependent slopes.

    ⚠️  DEPRECATED: superseded by ACMG-Bayes, which achieves additive
    combination (via log-LR summation) AND exact boundary posteriors AND
    prior-independent per-code thresholds simultaneously — the three-way
    tradeoff this class was built to explore no longer applies once
    combination is defined correctly. Kept for reference only.
    """
    _lr_fn        = staticmethod(lp_anchored_lr_plus)
    _post_fn      = staticmethod(lp_anchored_posterior)
    _BOUNDARIES   = CLASSIFICATION_BOUNDARIES
    _POST_TARGETS = POSTERIOR_TARGETS
    _METHOD       = "lp_anchored"


# ── ACMG-Bayes: multi-evidence combination via direct log-LR summation ──────
#
# Per-evidence-item log(LR+) values are summed directly (never rounded to an
# integer point value first), then classified by posterior at the end. This
# is exact Bayesian evidence combination — the "combination error" is zero by
# construction for any set of per-code log-LR values, at every prior,
# regardless of how many segment boundaries the running total would have
# crossed under a points-then-remap scheme. When there is only a single
# piece of evidence (already known as a continuous LR+, not a code), this
# reduces to the trivial case of no summation at all — the "Continuous"
# behaviour of earlier drafts is this suite's one-code special case.
#
# For a fixed set of named evidence codes we take each code's canonical
# log(LR+) from the prior-adaptive anchor (`piecewise_log_lr`) — the same
# per-code values ACMGBayesSuite reports — so combined posteriors are
# directly comparable to the Tavtigian combination-error figures.

def acmg_bayes_log_lr(codes: List[int], prior: float) -> float:
    """Sum of per-code canonical log(LR+) at *prior* — the combination operand.

    Combination is literal addition in log-LR space; no point-tier rounding
    occurs at any stage. A single-item list reduces to that item's own
    canonical log(LR+), i.e. the "continuous" single-assay case.
    """
    return float(sum(piecewise_log_lr(k, prior) for k in codes))


def acmg_bayes_posterior(codes: List[int], prior: float) -> float:
    """Posterior implied by summing *codes*' canonical log-LR at *prior*."""
    lr_total = float(np.exp(acmg_bayes_log_lr(codes, prior)))
    return float(bayes_posterior_from_lr(lr_total, prior))


def acmg_bayes_display_points(log_lr: float, prior: float) -> float:
    """Invert the piecewise knot geometry: log-LR -> fractional point label.

    Display-only conversion, applied to an already-combined log-LR value,
    never used as a combination operand itself. Uses the same four ACMG
    knots as `piecewise_log_lr`, inverted within (or extrapolated beyond)
    the enclosing segment.
    """
    knots_T  = [-7, -1, 6, 10]
    knots_lr = [float(piecewise_log_lr(t, prior)) for t in knots_T]

    if log_lr <= knots_lr[0]:
        lo, hi = 0, 1
    elif log_lr >= knots_lr[-1]:
        lo, hi = 2, 3
    else:
        lo, hi = next(
            (i, i + 1) for i in range(3)
            if knots_lr[i] <= log_lr <= knots_lr[i + 1]
        )

    slope = (knots_lr[hi] - knots_lr[lo]) / (knots_T[hi] - knots_T[lo])
    return knots_T[lo] + (log_lr - knots_lr[lo]) / slope


@dataclass
class ACMGBayesCombinationResult:
    prior: float
    combo: tuple
    log_lr_total: float = 0.0
    posterior: float = 0.0
    display_points: float = 0.0


class ACMGBayesCombinationSuite:
    """Multi-evidence combination sweep for ACMG-Bayes over a fixed set of
    combos.

    For interface parity with `_PiecewiseSuiteBase`, this sweeps the same
    (k_A, k_B) evidence-code pairs used in the combination-error benchmarks
    (see `compare.py`) and records the combined log-LR, posterior, and
    display-point label at every prior. Combination error against the
    per-code LR+ product reference is zero for every combo/prior by
    construction — this suite is for recording/plotting that fact, not for
    detecting it (see `compare.py::compute_ledger_combination_errors` for
    the numerical verification).
    """

    def __init__(self, priors: np.ndarray,
                 combos: Optional[List[tuple]] = None):
        self.priors = np.asarray(priors)
        self.combos = combos or [(4, 4), (2, 4), (2, 2), (6, 6)]
        self.results: List[ACMGBayesCombinationResult] = []

    def run(self) -> "ACMGBayesCombinationSuite":
        self.results = [
            self._run_one(combo, float(p))
            for p in self.priors
            for combo in self.combos
        ]
        return self

    def _run_one(self, combo: tuple, prior: float) -> ACMGBayesCombinationResult:
        log_lr = acmg_bayes_log_lr(list(combo), prior)
        post   = float(bayes_posterior_from_lr(float(np.exp(log_lr)), prior))
        pts    = acmg_bayes_display_points(log_lr, prior)
        return ACMGBayesCombinationResult(prior=prior, combo=combo,
                                           log_lr_total=log_lr,
                                           posterior=post, display_points=pts)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            rows.append({
                "prior": r.prior,
                "acmg_mapping_method": "acmg_bayes",
                "combo": r.combo,
                "log_lr_total": r.log_lr_total,
                "posterior": r.posterior,
                "display_points": r.display_points,
            })
        return pd.DataFrame(rows)
