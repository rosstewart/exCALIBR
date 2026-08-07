#!/usr/bin/env python3
"""Two focused CFUSN (q=2) hyperparameter-sensitivity experiments, kept in one
script since both are "does a hyperparameter matter" questions of the same
shape:

  (1) K-misspecification: fit with num_components above/below the true K,
      score recovery against the closest-matching true components (via
      sim_utils.match_components, which handles the resulting rectangular
      assignment natively), and report both closest-match error and the
      count of unmatched (spurious, when over-fit) or missed (when
      under-fit) components.
  (2) n_mc_truncated sensitivity: sweep the E-step's Monte-Carlo sample
      count for the truncated-normal moments. Note: _mc_truncated_mvn_moments
      (update_steps.py) is actually an analytic-approximation path for low
      q, not literal Monte Carlo sampling -- this experiment explicitly
      checks (and reports) whether n_mc_truncated has ANY measurable effect
      on recovery for production's q=2 default, since a flat curve would
      mean this parameter is effectively a no-op at q=2.

Usage:
    python tests/cfusn_simulations/sim_k_and_mc_sensitivity.py
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
from tests.cfusn_simulations.sim_utils import sample_cfusn_mixture, score_recovery, match_components

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── K-misspecification: true K=3, well-separated so mismatch is attributable
# to K itself, not to genuinely-overlapping components. ──
TRUE_PARAMS_K3 = [
    (np.array([-2.0, -2.0]), np.array([[0.5, 0.1], [0.3, -0.1]]), 0.2 * np.eye(2)),
    (np.array([0.0, 2.0]), np.array([[-0.3, 0.2], [0.4, 0.1]]), 0.2 * np.eye(2)),
    (np.array([2.0, -1.0]), np.array([[0.2, -0.3], [-0.2, 0.2]]), 0.2 * np.eye(2)),
]
COMP_PROBS_K3 = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8],
])
N_SAMPLES = 3

# ── n_mc_truncated sensitivity: simple K=2 problem, fully observed. ──
TRUE_PARAMS_MC = [
    (np.array([-1.5, -1.2]), np.array([[0.6, 0.2], [0.4, -0.15]]), 0.3 * np.eye(2)),
    (np.array([1.5, 1.2]), np.array([[-0.6, -0.2], [-0.4, 0.15]]), 0.3 * np.eye(2)),
]
COMP_PROBS_MC = np.array([[0.85, 0.15], [0.15, 0.85], [0.5, 0.5]])


def run_k_misspecification(k_grid, n_seeds, n_per_sample):
    rows = []
    sample_sizes = np.full(N_SAMPLES, n_per_sample)
    for k_fit in k_grid:
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            X, sa, _ = sample_cfusn_mixture(TRUE_PARAMS_K3, COMP_PROBS_K3, sample_sizes, rng)
            result = tryToFit(
                X, sa, num_components=k_fit, constrained=False,
                init_method="kmeans", init_constraint_adjustment="scale",
                multivariate=True, latent_q=2, check_monotonic=False,
                num_fits=1, fit_seed=int(seed), verbose=False, verbose_init=False,
            )
            params = result.get("component_params", [])
            if not params or any(len(p) == 0 for p in params):
                continue
            score = score_recovery(TRUE_PARAMS_K3, params)
            rows.append(dict(
                k_fit=k_fit, seed=seed,
                mean_omega_error=score["mean_omega_error"],
                n_unmatched_true=len(score["unmatched_true"]),
                n_unmatched_fit=len(score["unmatched_fit"]),
            ))
    return rows


def _report_k_misspecification(rows):
    print(f"\n{'═' * 78}")
    print(f"  K-MISSPECIFICATION (true K={len(TRUE_PARAMS_K3)})")
    print(f"{'═' * 78}")
    print(f"  {'K_fit':>6}  {'mean_omega_err':>16}  {'missed true':>12}  {'spurious fit':>13}  {'n':>4}")
    by_k = {}
    for r in rows:
        by_k.setdefault(r["k_fit"], []).append(r)
    for k in sorted(by_k):
        group = by_k[k]
        mean_err = np.mean([g["mean_omega_error"] for g in group])
        mean_missed = np.mean([g["n_unmatched_true"] for g in group])
        mean_spurious = np.mean([g["n_unmatched_fit"] for g in group])
        print(f"  {k:>6}  {mean_err:>16.4f}  {mean_missed:>12.2f}  {mean_spurious:>13.2f}  {len(group):>4}")


def run_mc_sensitivity(n_mc_grid, n_seeds, n_per_sample):
    rows = []
    sample_sizes = np.full(N_SAMPLES, n_per_sample)
    for n_mc in n_mc_grid:
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            X, sa, _ = sample_cfusn_mixture(TRUE_PARAMS_MC, COMP_PROBS_MC, sample_sizes, rng)
            result = tryToFit(
                X, sa, num_components=len(TRUE_PARAMS_MC), constrained=False,
                init_method="kmeans", init_constraint_adjustment="scale",
                multivariate=True, latent_q=2, check_monotonic=False,
                num_fits=1, fit_seed=int(seed), n_mc_truncated=n_mc,
                verbose=False, verbose_init=False,
            )
            params = result.get("component_params", [])
            if not params or any(len(p) == 0 for p in params):
                continue
            score = score_recovery(TRUE_PARAMS_MC, params)
            rows.append(dict(n_mc_truncated=n_mc, seed=seed,
                             mean_omega_error=score["mean_omega_error"]))
    return rows


def _report_mc_sensitivity(rows):
    print(f"\n{'═' * 78}")
    print("  n_mc_truncated SENSITIVITY (q=2)")
    print(f"{'═' * 78}")
    print(f"  {'n_mc_truncated':>15}  {'mean_omega_err':>16}  {'std':>10}  {'n':>4}")
    by_nmc = {}
    for r in rows:
        by_nmc.setdefault(r["n_mc_truncated"], []).append(r["mean_omega_error"])
    for n_mc in sorted(by_nmc):
        errs = np.array(by_nmc[n_mc])
        print(f"  {n_mc:>15}  {errs.mean():>16.4f}  {errs.std():>10.4f}  {len(errs):>4}")
    all_means = [np.mean(v) for v in by_nmc.values()]
    spread = max(all_means) - min(all_means) if all_means else float("nan")
    print(f"\n  Spread of mean error across n_mc_truncated settings: {spread:.4f}")
    print(f"  (if this is within one setting's std, n_mc_truncated has no "
          f"measurable effect at q=2 -- consistent with _mc_truncated_mvn_moments "
          f"being an analytic approximation for low q, not literal MC sampling)")


def _write_csv(rows, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-per-sample", type=int, default=200)
    parser.add_argument("--k-grid", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--n-mc-grid", type=int, nargs="+", default=[10, 50, 100, 500, 2000])
    args = parser.parse_args()

    k_rows = run_k_misspecification(args.k_grid, args.n_seeds, args.n_per_sample)
    _report_k_misspecification(k_rows)
    _write_csv(k_rows, "sim_k_misspecification")

    mc_rows = run_mc_sensitivity(args.n_mc_grid, args.n_seeds, args.n_per_sample)
    _report_mc_sensitivity(mc_rows)
    _write_csv(mc_rows, "sim_mc_sensitivity")
    print()


if __name__ == "__main__":
    main()
