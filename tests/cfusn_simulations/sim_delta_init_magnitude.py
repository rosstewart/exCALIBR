#!/usr/bin/env python3
"""Is production's fixed, conservative Delta-init magnitude (0.1 * sqrt(eigval),
same for every restart -- only the *sign* varies across lambdaIndex restarts)
the reason EM converges to a spurious near-zero-skew local optimum instead of
the true, higher-likelihood, larger-skew solution?

Directly established before this script (by hand, not simulated): initializing
a single-component CFUSN(q=2) fit EXACTLY at known large-skew true parameters
converges to a STRICTLY HIGHER likelihood than the same data fit via standard
kmeans_init_mv -- and the M-step formula itself is verified correct (one
M-step starting exactly at truth leaves it essentially unchanged). So this
looks like a real local-optimum problem caused specifically by the init
magnitude being too conservative near the skew=0 region (classically known to
have a near-flat/low-information likelihood ridge for skew-normal-family
models), not a flat-likelihood-ridge or M-step bug.

This script tests the natural fix -- and the natural risk: does using a
LARGER init magnitude reliably converge to the correct solution across BOTH
large-true-skew AND (crucially) near-zero-true-skew cases? Overshooting could
just as easily manufacture spurious skew where none exists as it could find
real skew that a conservative init misses.

Design: single-component CFUSN(q=2) fits (isolates the init/M-step question
cleanly -- no mixture label-switching or responsibility ambiguity), large N
(minimizes finite-sample noise so any remaining error is attributable to
optimization, not data). Sweeps:
  - TRUE skew magnitude: zero, small, medium, large (||Delta_true|| column
    norm 0, 0.15, 0.35, 0.7)
  - init scale_factor: 0.1 (current production default) through 2.0
For each (regime, scale_factor, seed), fits ONE restart (no averaging over
diverse restarts) at that scale -- directly answering "does the limited
single fit we'd actually run converge optimally," per the request, with a
diverse-restarts comparison as a secondary check at the end.

Usage:
    python tests/cfusn_simulations/sim_delta_init_magnitude.py
    python tests/cfusn_simulations/sim_delta_init_magnitude.py --n-seeds 10
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

import src.assay_calibration.fit_utils.cfusn.initializations as INIT
from src.assay_calibration.fit_utils.fit import tryToFit
from tests.cfusn_simulations.sim_utils import sample_cfusn, init_delta_matrix_scaled

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MU_TRUE = np.zeros(4)
GAMMA_TRUE = 0.5 * np.eye(4)
DIRECTION = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # unit-ish directions

SKEW_REGIMES = {
    "zero": 0.0,
    "small": 0.15,
    "medium": 0.35,
    "large": 0.7,
}
SCALE_FACTORS = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
N_OBS = 5000


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _run_one(magnitude, scale_factor, seed, n_restarts=1):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    def _scaled_init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
        return init_delta_matrix_scaled(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                        rng=rng, scale_factor=scale_factor)

    orig = INIT._init_delta_matrix
    INIT._init_delta_matrix = _scaled_init
    try:
        best = None
        for i in range(n_restarts):
            result = tryToFit(
                X, sa, num_components=1, constrained=False, init_method="kmeans",
                init_constraint_adjustment="scale", multivariate=True, latent_q=2,
                check_monotonic=False, num_fits=1, fit_seed=int(seed * 100 + i),
                lambdaIndex=i, max_em_iters=300, verbose=False, verbose_init=False,
            )
            params = result.get("component_params", [])
            if not params or len(params[0]) == 0:
                continue
            ll = result["likelihoods"][-1] if len(result.get("likelihoods", [])) else -np.inf
            if best is None or ll > best[0]:
                best = (ll, params)
    finally:
        INIT._init_delta_matrix = orig

    if best is None:
        return None
    ll, params = best
    Delta_fit = np.asarray(params[0][1])
    return dict(
        true_norm=float(np.linalg.norm(Delta_true)),
        fit_norm=float(np.linalg.norm(Delta_fit)),
        final_ll=float(ll),
    )


def run_sweep(n_seeds, n_restarts_diverse):
    rows = []
    for regime, magnitude in SKEW_REGIMES.items():
        for scale_factor in SCALE_FACTORS:
            for seed in range(n_seeds):
                out = _run_one(magnitude, scale_factor, seed, n_restarts=1)
                if out is None:
                    continue
                rows.append(dict(regime=regime, scale_factor=scale_factor, seed=seed,
                                 n_restarts=1, **out))
                # Secondary check: does combining this scale with several
                # diverse (sign-varied) restarts change the picture?
                out_diverse = _run_one(magnitude, scale_factor, seed, n_restarts=n_restarts_diverse)
                if out_diverse is None:
                    continue
                rows.append(dict(regime=regime, scale_factor=scale_factor, seed=seed,
                                 n_restarts=n_restarts_diverse, **out_diverse))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print("  DELTA INIT MAGNITUDE SWEEP (single-component CFUSN q=2, N=20000)")
    print(f"{'═' * 100}")
    for n_restarts in sorted(set(r["n_restarts"] for r in rows)):
        print(f"\n  --- n_restarts={n_restarts} ---")
        print(f"  {'regime':>8}  {'true_norm':>10}  {'scale':>6}  {'mean fit_norm':>14}  "
              f"{'recovered%':>11}  {'mean final_ll':>14}  {'n':>4}")
        subset = [r for r in rows if r["n_restarts"] == n_restarts]
        by_key = {}
        for r in subset:
            by_key.setdefault((r["regime"], r["scale_factor"]), []).append(r)
        for regime in SKEW_REGIMES:
            for scale_factor in SCALE_FACTORS:
                group = by_key.get((regime, scale_factor), [])
                if not group:
                    continue
                true_norm = group[0]["true_norm"]
                fit_norms = np.array([g["fit_norm"] for g in group])
                lls = np.array([g["final_ll"] for g in group])
                pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
                print(f"  {regime:>8}  {true_norm:>10.3f}  {scale_factor:>6.1f}  "
                      f"{fit_norms.mean():>14.3f}  {pct:>10.0f}%  {lls.mean():>14.5f}  {len(group):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_magnitude_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-restarts-diverse", type=int, default=4,
                        help="Sign-varied restarts for the secondary 'diverse fits' check")
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds, args.n_restarts_diverse)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
