#!/usr/bin/env python3
"""Compare production kmeans_init_mv (now partial-distance k-means, promoted
from this file's earlier prototype) against the OLD approach it replaced
(complete-rows-only clustering, falling back to global-mean imputation of
every missing entry when too few complete rows exist), for a scenario where
a genuinely necessary separating signal lives in a dimension that gets swept
through increasing missingness -- i.e. a case where losing that dimension's
contribution to cluster ASSIGNMENT should actually matter, unlike a case
where other dimensions already separate the clusters on their own.

This was the decisive validation run before promoting the prototype: at
missing_frac up to ~0.85, the two are indistinguishable (both cluster fine
on whatever complete rows exist); past that, the old approach's global-mean
fallback erases the swept dimension's real signal (identical value across
every row -> zero contribution to which cluster a point is assigned) and
degrades sharply, including a ~1e16-scale numerical blowup at 0.99 missing;
partial-distance k-means degrades gracefully throughout.

Design: 2 true components are only WEAKLY separated in the two always-
observed dimensions (small mean gap, large overlap) but STRONGLY separated
in two additional dimensions that get swept through increasing missingness.

Comparison is via full end-to-end EM recovery accuracy (sim_utils.
score_recovery after a complete tryToFit), not just raw init-label
agreement -- kmeans_init_mv doesn't cleanly expose hard labels for
incomplete rows under the old approach (they were assigned via ad hoc
nearest-center lookup, never stored), and recovery accuracy is the metric
that actually matters (a "prettier" init that doesn't change the final fit
isn't interesting). The "legacy" condition is produced by monkeypatching
kmeans_init_mv to a locally-preserved copy of the pre-promotion function;
the "current" condition needs no patching since it's now the default --
everything else in the fit (EM loop, weight refinement, etc.) is identical
between the two.

Usage:
    python tests/cfusn_simulations/sim_kmeans_missingness_init.py
    python tests/cfusn_simulations/sim_kmeans_missingness_init.py --n-seeds 10
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.assay_calibration.fit_utils.cfusn.fit as CFIT
from src.assay_calibration.fit_utils.fit import tryToFit
from tests.cfusn_simulations.sim_utils import (
    sample_cfusn_mixture, inject_missingness, score_recovery, kmeans_init_mv_legacy,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# dims 0-1: weakly separated (small mean gap, large shared variance).
# dims 2-3: strongly separated, but subject to the missingness sweep.
TRUE_PARAMS = [
    (np.array([-0.3, -0.3, -3.0, -3.0]),
     np.array([[0.1, 0.05], [0.05, -0.05], [0.2, 0.1], [0.15, -0.1]]), 1.0 * np.eye(4)),
    (np.array([0.3, 0.3, 3.0, 3.0]),
     np.array([[-0.1, -0.05], [-0.05, 0.05], [-0.2, -0.1], [-0.15, 0.1]]), 1.0 * np.eye(4)),
]
N_SAMPLES = 3
COMP_PROBS = np.array([[0.7, 0.3], [0.3, 0.7], [0.5, 0.5]])
SWEPT_DIMS = [2, 3]



def _run_one(missing_frac, seed, use_legacy):
    rng = np.random.RandomState(seed)
    sample_sizes = np.full(N_SAMPLES, 200)
    X, sa, _ = sample_cfusn_mixture(TRUE_PARAMS, COMP_PROBS, sample_sizes, rng)
    frac_per_dim = np.zeros(X.shape[1])
    frac_per_dim[SWEPT_DIMS] = missing_frac
    X_missing = inject_missingness(X, frac_per_dim, rng) if missing_frac > 0 else X

    if use_legacy:
        orig = CFIT.kmeans_init_mv
        CFIT.kmeans_init_mv = kmeans_init_mv_legacy
    try:
        result = tryToFit(
            X_missing, sa, num_components=2, constrained=False,
            init_method="kmeans", init_constraint_adjustment="scale",
            multivariate=True, latent_q=2, check_monotonic=False,
            num_fits=1, fit_seed=int(seed), verbose=False, verbose_init=False,
        )
    finally:
        if use_legacy:
            CFIT.kmeans_init_mv = orig

    params = result.get("component_params", [])
    if not params or any(len(p) == 0 for p in params):
        return None
    return score_recovery(TRUE_PARAMS, params)["mean_omega_error"]


def run_sweep(missing_frac_grid, n_seeds):
    rows = []
    for missing_frac in missing_frac_grid:
        for seed in range(n_seeds):
            for use_legacy in (True, False):
                err = _run_one(missing_frac, seed, use_legacy)
                if err is None:
                    continue
                rows.append(dict(
                    missing_frac=missing_frac, seed=seed,
                    variant="legacy" if use_legacy else "current",
                    mean_omega_error=err,
                ))
    return rows


def _report(rows):
    print(f"\n{'═' * 78}")
    print("  KMEANS INIT: legacy (complete-rows/global-mean) vs. current (partial-distance)")
    print(f"{'═' * 78}")
    print(f"  {'missing_frac':>12}  {'legacy err':>13}  {'current err':>14}  {'n':>4}")

    by_frac = {}
    for r in rows:
        by_frac.setdefault(r["missing_frac"], {"legacy": [], "current": []})
        by_frac[r["missing_frac"]][r["variant"]].append(r["mean_omega_error"])

    for frac in sorted(by_frac):
        legacy = np.array(by_frac[frac]["legacy"])
        current = np.array(by_frac[frac]["current"])
        l_mean = legacy.mean() if len(legacy) else float("nan")
        c_mean = current.mean() if len(current) else float("nan")
        print(f"  {frac:>12.2f}  {l_mean:>13.4f}  {c_mean:>14.4f}  {len(legacy):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_kmeans_missingness_init_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--missing-frac-grid", type=float, nargs="+",
                        default=[0.0, 0.5, 0.7, 0.85, 0.90, 0.95, 0.99])
    args = parser.parse_args()

    print(f"True params: {len(TRUE_PARAMS)} components, dims {SWEPT_DIMS} are the "
          f"strongly-separating-but-swept dimensions; dims 0-1 are weakly separating "
          f"and always fully observed.")
    print(f"n_seeds={args.n_seeds}")

    rows = run_sweep(args.missing_frac_grid, args.n_seeds)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
