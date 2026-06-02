"""
Parallelisable sweep of the Tavtigian framework across a grid of prior
probabilities.

Usage
-----
from tavtigian_sims import SimulationSuite, prior_grid

suite = SimulationSuite(priors=prior_grid("dense"), n_jobs=-1)
suite.run()
df = suite.to_dataframe()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import List, Optional

from .core import TavtigianResult, run_simulation, POINT_VALUES, CLASSIFICATION_BOUNDARIES


# ── prior grids ───────────────────────────────────────────────────────────────

def prior_grid(
    mode: str = "standard",
    n_coarse: int = 80,
    n_fine: int = 200,
    fine_lo: float = 0.05,
    fine_hi: float = 0.50,
) -> np.ndarray:
    """
    Return a deduplicated, sorted array of prior values for sweeping.

    Modes
    -----
    "standard"      : log-spaced (0.01, 0.99) — good default
    "dense"         : log-spaced + linear concentration around (0.05, 0.50)
    "clinical"      : dense around the clinically relevant range (0.05–0.30)
    "full"          : log-spaced (0.001, 0.999) — probes extreme priors
    "paper"         : replicates the Tavtigian 2018 Figure 1 range (0.01–0.45)
    "comprehensive" : log-spaced extremes (0.0001–0.10) + very dense linear
                      (0.10–0.35, centred on the C* minimum) + log-spaced
                      tail (0.35–0.99).  Best overall coverage.
    """
    if mode == "standard":
        pts = np.logspace(np.log10(0.01), np.log10(0.99), n_coarse)
    elif mode == "dense":
        coarse = np.logspace(np.log10(0.01), np.log10(0.99), n_coarse)
        fine   = np.linspace(fine_lo, fine_hi, n_fine)
        pts = np.concatenate([coarse, fine])
    elif mode == "clinical":
        coarse = np.logspace(np.log10(0.01), np.log10(0.99), n_coarse)
        fine   = np.linspace(0.05, 0.30, n_fine)
        pts = np.concatenate([coarse, fine])
    elif mode == "full":
        pts = np.logspace(np.log10(0.001), np.log10(0.999), 300)
    elif mode == "paper":
        # Matches the x-range used in Tavtigian 2018 Figure 1
        pts = np.linspace(0.01, 0.45, 300)
    elif mode == "comprehensive":
        # Log-spaced extremes: 0.0001, 0.001, 0.01, ..., 0.10
        low_tail  = np.logspace(np.log10(0.0001), np.log10(0.10), 60)
        # Very dense linear region centred on the C* minimum (~0.25)
        center    = np.linspace(0.10, 0.35, 500)
        # Log-spaced upper tail: 0.35, ..., 0.50, ..., 0.90, 0.99
        high_tail = np.logspace(np.log10(0.35), np.log10(0.99), 80)
        pts = np.concatenate([low_tail, center, high_tail])
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return np.unique(np.clip(pts, 1e-4, 1 - 1e-4))


# ── SimulationSuite ───────────────────────────────────────────────────────────

class SimulationSuite:
    """
    Sweep the Tavtigian framework over a grid of prior probabilities.

    Parameters
    ----------
    priors : array-like, optional
        Prior probabilities to evaluate.  Defaults to prior_grid("dense").
    original : bool
        Use original ACMG combining rules (default False = Tavtigian 2018).
    strict : bool
        Use strict LP/LB posterior bounds.
    C_max : int or None
        Upper bound for the integer grid search of C*.
        None (default) defers to get_tavtigian_constant's own default (100 000),
        matching the behaviour of thresholds_from_prior in fit.py.
        Pass 350 to reproduce the paper's primary example exactly.
    n_jobs : int
        Number of parallel workers (joblib convention; -1 = all CPUs).
    """

    def __init__(
        self,
        priors: Optional[np.ndarray] = None,
        original: bool = False,
        strict: bool = False,
        C_max: Optional[int] = None,
        n_jobs: int = -1,
    ):
        self.priors   = np.asarray(priors) if priors is not None else prior_grid("dense")
        self.original = original
        self.strict   = strict
        self.C_max    = C_max
        self.n_jobs   = n_jobs
        self.results: List[TavtigianResult] = []

    def run(self) -> "SimulationSuite":
        """Execute all simulations in parallel; populate self.results."""
        raw = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(run_simulation)(p, self.original, self.strict, self.C_max)
            for p in self.priors
        )
        self.results = sorted(raw, key=lambda r: r.prior)
        return self

    # ── export ─────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten all results into a tidy DataFrame.

        Columns include per-tier thresholds, posteriors, and classification
        boundary checks.  See core.py for column-name conventions.
        """
        rows = []
        for r in self.results:
            row: dict = {
                "prior":        r.prior,
                "C_star":       r.C_star,
                "total_fails":  r.total_fails,
                "path_fails":   r.path_fails,
                "lp_fails":     r.lp_fails,
                "benign_fails": r.benign_fails,
                "lb_fails":     r.lb_fails,
            }
            # Per-tier thresholds
            for k in POINT_VALUES:
                row[f"lr_p{k}"]      = r.lr_threshold[k]
                row[f"log10_lr_p{k}"]= r.log10_lr_threshold[k]
                row[f"post_p{k}"]    = r.posterior_at_tier[k]
                row[f"lr_b{k}"]      = r.lr_threshold[-k]
                row[f"log10_lr_b{k}"]= r.log10_lr_threshold[-k]
                row[f"post_b{k}"]    = r.posterior_at_tier[-k]
            # Classification boundary checks
            for name in CLASSIFICATION_BOUNDARIES:
                row[f"bnd_lr_{name}"]     = r.boundary_lr[name]
                row[f"bnd_post_{name}"]   = r.boundary_posterior[name]
                row[f"bnd_pass_{name}"]   = r.boundary_passes[name]
            rows.append(row)
        return pd.DataFrame(rows)

    def combining_rule_dataframe(self) -> pd.DataFrame:
        """Long-form combining-rule posteriors for all priors."""
        rows = []
        for r in self.results:
            for cat, arr in [("P",  r.pathogenic_posteriors),
                              ("LP", r.likely_path_posteriors),
                              ("B",  r.benign_posteriors),
                              ("LB", r.likely_benign_posteriors)]:
                for i, p in enumerate(arr):
                    rows.append({"prior": r.prior, "category": cat,
                                 "rule_idx": i, "posterior": float(p)})
        return pd.DataFrame(rows)
