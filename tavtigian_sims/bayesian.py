"""
Simulation suites for the two Bayesian replacement methods.

PiecewiseSuite  : Method A — prior-adaptive piecewise α (integer points
                  preserved, exact boundary posteriors at every prior).
ContinuousSuite : Method B — continuous LR+ (no discretisation, posterior-
                  based classification).

Both expose a DataFrame interface compatible with the plotting code in
``plots.py`` so they can be overlaid on the legacy Tavtigian suite.
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
    piecewise_additive_lr_plus, piecewise_additive_posterior,
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


# ── Generic piecewise base ────────────────────────────────────────────────────

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
    """Shared machinery for piecewise-α suites."""

    # Subclasses override these three:
    _lr_fn    = staticmethod(piecewise_lr_plus)
    _post_fn  = staticmethod(piecewise_posterior)
    _BOUNDARIES = CLASSIFICATION_BOUNDARIES
    _POST_TARGETS = POSTERIOR_TARGETS
    _METHOD   = "piecewise"

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


# ── Method A: standard piecewise α ───────────────────────────────────────────

class PiecewiseSuite(_PiecewiseSuiteBase):
    """Method A sweep across priors.  Same interface as SimulationSuite."""
    _lr_fn       = staticmethod(piecewise_lr_plus)
    _post_fn     = staticmethod(piecewise_posterior)
    _BOUNDARIES  = CLASSIFICATION_BOUNDARIES
    _POST_TARGETS = POSTERIOR_TARGETS
    _METHOD      = "piecewise"


# ── Method A′: (6·11·6) additive piecewise ───────────────────────────────────

class PiecewiseAdditiveSuite(_PiecewiseSuiteBase):
    """Piecewise-α with (6·11·6) gap-pattern knots at T = 17, 11, 0, −6.

    Two conditions for near-perfect additivity:
      1. Equal slopes (gap pattern 6·11·6): cross-segment error < 0.07 %.
      2. T_LB = 0: log_lr(0, p=0.10) = 0, making same-segment combination
         exactly additive at the canonical prior.

    Posteriors at the four knot T values are exact for every prior.
    Classification boundaries: B=−6, LB=0, LP=11, P=17.
    """
    _lr_fn        = staticmethod(piecewise_additive_lr_plus)
    _post_fn      = staticmethod(piecewise_additive_posterior)
    _BOUNDARIES   = ADDITIVE_CLASSIFICATION_BOUNDARIES
    _POST_TARGETS = ADDITIVE_POSTERIOR_TARGETS
    _METHOD       = "piecewise_additive"


# ── Method B: continuous LR+ suite ───────────────────────────────────────────

@dataclass
class ContinuousResult:
    prior: float
    # LR+ at each ACMG posterior target
    lr_threshold: Dict[str, float] = field(default_factory=dict)


def _run_continuous(prior: float) -> ContinuousResult:
    r = ContinuousResult(prior=prior)
    r.lr_threshold = continuous_lr_thresholds(prior)
    return r


class ContinuousSuite:
    """Method B sweep across priors.

    There are no integer codes in Method B — classification is by posterior
    directly.  This suite just records the LR+ value corresponding to each
    of the four posterior targets (P=0.99, LP=0.90, LB=0.10, B=0.01).
    """

    def __init__(self, priors: np.ndarray):
        self.priors = np.asarray(priors)
        self.results: List[ContinuousResult] = []

    def run(self) -> "ContinuousSuite":
        self.results = [_run_continuous(float(p)) for p in self.priors]
        return self

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {"prior": r.prior, "acmg_mapping_method": "continuous"}
            for name, lr in r.lr_threshold.items():
                row[f"lr_{name}"]       = lr
                row[f"log10_lr_{name}"] = float(np.log10(lr)) if lr > 0 else float("-inf")
                row[f"post_{name}"]     = float(
                    bayes_posterior_from_lr(lr, r.prior)
                )
            rows.append(row)
        return pd.DataFrame(rows)
