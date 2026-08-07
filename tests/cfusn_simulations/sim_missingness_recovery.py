#!/usr/bin/env python3
"""Recovery-accuracy (mu/Delta/Gamma/Omega/weight error, sim_utils.score_recovery)
as a function of per-dimension missingness fraction, for a fixed ground-truth
CFUSN(q=2) mixture shaped like a real TP53-scale problem (a couple of
well-covered, informative dimensions + several sparse ones).

This is what empirically grounds discussions like "is KawOligo's real ~1-12%
coverage inside or outside the range where per-dimension recovery holds up" --
the deliverable is an error-vs-missingness curve (printed summary + CSV for
later plotting), not a single anecdotal number.

Usage:
    python tests/cfusn_simulations/sim_missingness_recovery.py
    python tests/cfusn_simulations/sim_missingness_recovery.py --n-seeds 10
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

from src.assay_calibration.fit_utils.fit import tryToFit
from tests.cfusn_simulations.sim_utils import sample_cfusn_mixture, inject_missingness, score_recovery

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# p=6: dims 0-1 informative & always fully observed (mirrors TP53's core
# functional assays), dims 2-5 sparse & subject to the missingness sweep
# (mirrors TP53's KawOligo/log_TempSens/DN_score/etc.).
TRUE_PARAMS = [
    (np.array([-1.5, -1.2, 0.5, -0.3, 0.2, -0.4]),
     np.array([[0.6, 0.2], [0.4, -0.15], [0.3, 0.1], [0.2, -0.1], [-0.15, 0.05], [0.1, 0.2]]),
     0.3 * np.eye(6)),
    (np.array([1.5, 1.2, -0.5, 0.3, -0.2, 0.4]),
     np.array([[-0.6, -0.2], [-0.4, 0.15], [-0.3, -0.1], [-0.2, 0.1], [0.15, -0.05], [-0.1, -0.2]]),
     0.3 * np.eye(6)),
]
N_SAMPLES = 3
COMP_PROBS = np.array([[0.85, 0.15], [0.15, 0.85], [0.5, 0.5]])
SPARSE_DIMS = [2, 3, 4, 5]


def _true_component_weights(sample_sizes):
    sample_sizes = np.asarray(sample_sizes, dtype=float)
    total = sample_sizes.sum()
    return (COMP_PROBS * sample_sizes[:, None]).sum(axis=0) / total


def _fit_component_weights(fit_weights, sample_sizes):
    sample_sizes = np.asarray(sample_sizes, dtype=float)
    total = sample_sizes.sum()
    return (np.asarray(fit_weights) * sample_sizes[:, None]).sum(axis=0) / total


def run_sweep(missing_frac_grid, n_seeds, n_per_sample):
    sample_sizes = np.full(N_SAMPLES, n_per_sample)
    true_weight = _true_component_weights(sample_sizes)
    rows = []

    for missing_frac in missing_frac_grid:
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed * 1000 + int(missing_frac * 100))
            X, sa, _ = sample_cfusn_mixture(TRUE_PARAMS, COMP_PROBS, sample_sizes, rng)
            frac_per_dim = np.zeros(X.shape[1])
            frac_per_dim[SPARSE_DIMS] = missing_frac
            X_missing = inject_missingness(X, frac_per_dim, rng) if missing_frac > 0 else X

            result = tryToFit(
                X_missing, sa, num_components=len(TRUE_PARAMS), constrained=False,
                init_method="kmeans", init_constraint_adjustment="scale",
                multivariate=True, latent_q=2, check_monotonic=False,
                num_fits=1, fit_seed=int(seed), verbose=False, verbose_init=False,
            )
            params = result.get("component_params", [])
            if not params or any(len(p) == 0 for p in params):
                continue
            fit_weight = _fit_component_weights(result["weights"], sample_sizes)

            score = score_recovery(TRUE_PARAMS, params, true_weight, fit_weight)
            for pair in score["pairs"]:
                rows.append(dict(missing_frac=missing_frac, seed=seed, **pair))

    return rows


def _report(rows):
    print(f"\n{'═' * 90}")
    print("  RECOVERY ACCURACY vs. PER-DIMENSION MISSINGNESS FRACTION")
    print(f"{'═' * 90}")
    print(f"  {'missing_frac':>12}  {'mu_err':>10}  {'delta_err':>10}  "
          f"{'gamma_err':>10}  {'omega_err':>10}  {'weight_err':>10}  {'n':>5}")

    by_frac = {}
    for r in rows:
        by_frac.setdefault(r["missing_frac"], []).append(r)

    for frac in sorted(by_frac):
        group = by_frac[frac]
        means = {k: np.mean([g[k] for g in group])
                for k in ("mu_error", "delta_error", "gamma_error", "omega_error", "weight_error")}
        print(f"  {frac:>12.2f}  {means['mu_error']:>10.4f}  {means['delta_error']:>10.4f}  "
              f"{means['gamma_error']:>10.4f}  {means['omega_error']:>10.4f}  "
              f"{means['weight_error']:>10.4f}  {len(group):>5}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_missingness_recovery_{ts}.csv"
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
    parser.add_argument("--n-per-sample", type=int, default=300)
    parser.add_argument("--missing-frac-grid", type=float, nargs="+",
                        default=[0.0, 0.5, 0.7, 0.85, 0.90, 0.95, 0.99])
    args = parser.parse_args()

    print(f"True params: {len(TRUE_PARAMS)} components, {TRUE_PARAMS[0][0].shape[0]}D "
          f"(dims {SPARSE_DIMS} subject to missingness sweep)")
    print(f"n_seeds={args.n_seeds}, n_per_sample={args.n_per_sample}")

    rows = run_sweep(args.missing_frac_grid, args.n_seeds, args.n_per_sample)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
