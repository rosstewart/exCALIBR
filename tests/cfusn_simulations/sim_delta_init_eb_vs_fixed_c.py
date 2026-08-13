#!/usr/bin/env python3
"""Does a POOLED empirical-Bayes shrinkage weight -- derived from the
observed spread of skewness z-scores across a fit's own K components
(positive-part James-Stein: weight = max(0, 1-(m-2)/sum(z_j^2)), m = total
pooled z-scores) -- reduce recovery error compared to a single, manually
swept-and-hardcoded James-Stein c, on the SAME data?

This needs K>=2 components to have more than 2 z-scores to pool from (a
single-component q=2 fit only has 2 columns -- too few for empirical
Bayes to estimate anything meaningful, which is why every earlier script
in this investigation used num_components=1 and could never have tested
this). Deliberately CLEAN, fully-observed data (no missingness) to isolate
"does pooling beat a fixed c" from the separate missingness/projection-
noise issues already investigated at length -- if EB doesn't help even
here, in the easy case, testing it under missingness on top wouldn't be a
fair next step.

Setup: K=3 well-separated components, HETEROGENEOUS true skew across them
(zero / medium / large), p=4, q=2 (same DIRECTION/Gamma structure as the
original clean-data single-component sims for continuity). Compares:
  (a) kmeans_init_mv_fixed_c, c=1.0 (this investigation's best manually-
      calibrated clean-data constant)
  (b) kmeans_init_mv_eb (pooled empirical-Bayes weight, no external c)
at the same n_restarts=4, across several seeds -- no c-grid needed since
there's only one fixed-c value being compared against, per the (correct)
observation that this comparison doesn't require re-sweeping.

Usage:
    python tests/cfusn_simulations/sim_delta_init_eb_vs_fixed_c.py
    python tests/cfusn_simulations/sim_delta_init_eb_vs_fixed_c.py --n-seeds 15
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

import src.assay_calibration.fit_utils.cfusn.fit as CFUSN_FIT_MODULE
from src.assay_calibration.fit_utils.fit import tryToFit
from tests.cfusn_simulations.sim_utils import (
    sample_cfusn_mixture, kmeans_init_mv_fixed_c, kmeans_init_mv_eb, score_recovery,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DIRECTION = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
GAMMA_TRUE = 0.5 * np.eye(4)
# Well-separated component means so k-means reliably recovers cluster
# membership -- the recovery question this script asks is about Delta
# magnitude/sign calibration, not clustering accuracy.
MU_TRUES = [np.zeros(4), 4.0 * np.ones(4), 8.0 * np.ones(4)]
SKEW_MAGNITUDES = [0.0, 0.35, 0.7]  # zero / medium / large, one per component
N_TOTAL = 6000
FIXED_C = 1.0
N_RESTARTS = 4
N_COMPONENTS = 3


def _true_params():
    return [
        (MU_TRUES[k], DIRECTION * SKEW_MAGNITUDES[k], GAMMA_TRUE)
        for k in range(N_COMPONENTS)
    ]


def _run_one(variant, seed, n_restarts):
    true_params = _true_params()
    rng = np.random.RandomState(seed)
    weights_per_sample = np.array([[1 / N_COMPONENTS] * N_COMPONENTS])
    X, sa, true_components = sample_cfusn_mixture(
        true_params, weights_per_sample, [N_TOTAL], rng
    )

    if variant == "fixed_c":
        def _init(X, **kwargs):
            return kmeans_init_mv_fixed_c(X, c=FIXED_C, **kwargs)
    else:
        def _init(X, **kwargs):
            return kmeans_init_mv_eb(X, **kwargs)

    orig = CFUSN_FIT_MODULE.kmeans_init_mv
    CFUSN_FIT_MODULE.kmeans_init_mv = _init
    try:
        best = None
        for i in range(n_restarts):
            result = tryToFit(
                X, sa, num_components=N_COMPONENTS, constrained=False, init_method="kmeans",
                init_constraint_adjustment="scale", multivariate=True, latent_q=2,
                check_monotonic=False, num_fits=1, fit_seed=int(seed * 100 + i),
                lambdaIndex=i, max_em_iters=300, verbose=False, verbose_init=False,
            )
            params = result.get("component_params", [])
            if not params or any(len(p) == 0 for p in params):
                continue
            ll = result["likelihoods"][-1] if len(result.get("likelihoods", [])) else -np.inf
            if best is None or ll > best[0]:
                best = (ll, params)
    finally:
        CFUSN_FIT_MODULE.kmeans_init_mv = orig

    if best is None:
        return None
    ll, params = best
    fit_params = [(p[0], p[1], p[2]) for p in params]
    score = score_recovery(true_params, fit_params)
    if not score["pairs"]:
        return None

    delta_errors = [pr["delta_error"] for pr in score["pairs"]]
    omega_errors = [pr["omega_error"] for pr in score["pairs"]]
    return dict(
        final_ll=float(ll),
        mean_omega_error=score["mean_omega_error"],
        mean_delta_error=float(np.mean(delta_errors)),
        n_unmatched=len(score["unmatched_true"]) + len(score["unmatched_fit"]),
    )


def run_sweep(n_seeds, n_restarts):
    rows = []
    for variant in ("fixed_c", "eb"):
        for seed in range(n_seeds):
            out = _run_one(variant, seed, n_restarts)
            if out is None:
                continue
            rows.append(dict(variant=variant, seed=seed, n_restarts=n_restarts, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  EMPIRICAL-BAYES (pooled across K={N_COMPONENTS} components) vs. "
          f"FIXED c={FIXED_C}: clean data, heterogeneous skew {SKEW_MAGNITUDES}")
    print(f"{'═' * 100}")
    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)

    print(f"\n  {'variant':>10}  {'mean omega_err':>15}  {'mean delta_err':>15}  "
          f"{'mean final_ll':>14}  {'unmatched (sum)':>16}  {'n':>4}")
    for variant in ("fixed_c", "eb"):
        group = by_variant.get(variant, [])
        if not group:
            continue
        omega_errs = np.array([g["mean_omega_error"] for g in group])
        delta_errs = np.array([g["mean_delta_error"] for g in group])
        lls = np.array([g["final_ll"] for g in group])
        unmatched = sum(g["n_unmatched"] for g in group)
        print(f"  {variant:>10}  {omega_errs.mean():>15.4f}  {delta_errs.mean():>15.4f}  "
              f"{lls.mean():>14.5f}  {unmatched:>16}  {len(group):>4}")

    fc = by_variant.get("fixed_c", [])
    eb = by_variant.get("eb", [])
    if fc and eb:
        fc_omega = np.mean([g["mean_omega_error"] for g in fc])
        eb_omega = np.mean([g["mean_omega_error"] for g in eb])
        pct_change = 100 * (eb_omega - fc_omega) / fc_omega if fc_omega > 1e-9 else float("nan")
        print(f"\n  EB vs. fixed-c mean_omega_error change: {pct_change:+.1f}% "
              f"({'EB better' if pct_change < 0 else 'fixed-c better' if pct_change > 0 else 'tied'})")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_eb_vs_fixed_c_{ts}.csv"
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
    parser.add_argument("--n-restarts", type=int, default=N_RESTARTS)
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds, args.n_restarts)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
