#!/usr/bin/env python3
"""Does a data-driven (method-of-moments) Delta-init magnitude fix what a
fixed scale factor couldn't -- staying small when true skew is ~0 AND
recovering large magnitude when true skew is large, in the SAME run, with
no manual scale tuning?

sim_delta_init_magnitude.py established that NO single fixed scale_factor
works across regimes: small factors under-recover real large skew (8% at
scale=0.1), large factors manufacture spurious skew from nothing (norm 1.5
fitted from a true norm of 0.0 at scale=1.0). The natural fix, mirroring
what the univariate path already does (sn_method_of_moments_init inverts
the Azzalini skew-normal skewness formula from the SAMPLE skewness to size
the shape parameter from data) but was never ported to the CFUSN magnitude
init (_init_delta_matrix only ever used data to pick the *sign*, never the
*magnitude*, which was a fixed 0.1x constant): sim_utils.init_delta_matrix_mom
sizes each column's initial magnitude from that projection's own sample
skewness via the same inversion, instead of a fixed factor.

Compares, across the same zero/small/medium/large true-skew regimes as
sim_delta_init_magnitude.py: production's current fixed scale=0.1 vs. the
method-of-moments init, at both n_restarts=1 (the practically-relevant case)
and n_restarts=4 (diverse-restarts, for reference).

Usage:
    python tests/cfusn_simulations/sim_delta_init_mom.py
    python tests/cfusn_simulations/sim_delta_init_mom.py --n-seeds 10
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
from tests.cfusn_simulations.sim_utils import sample_cfusn, init_delta_matrix_scaled, init_delta_matrix_mom

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MU_TRUE = np.zeros(4)
GAMMA_TRUE = 0.5 * np.eye(4)
DIRECTION = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

SKEW_REGIMES = {
    "zero": 0.0,
    "small": 0.15,
    "medium": 0.35,
    "large": 0.7,
}
N_OBS = 5000
FIXED_SCALE_BASELINE = 0.1  # production's current default, for comparison


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _run_one(magnitude, variant, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    if variant == "fixed":
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_scaled(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                            rng=rng, scale_factor=FIXED_SCALE_BASELINE)
    else:
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_mom(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                         rng=rng, fallback_scale=FIXED_SCALE_BASELINE)

    orig = INIT._init_delta_matrix
    INIT._init_delta_matrix = _init
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


def run_sweep(n_seeds, n_restarts_list):
    rows = []
    for regime, magnitude in SKEW_REGIMES.items():
        for variant in ("fixed", "mom"):
            for n_restarts in n_restarts_list:
                for seed in range(n_seeds):
                    out = _run_one(magnitude, variant, seed, n_restarts)
                    if out is None:
                        continue
                    rows.append(dict(regime=regime, variant=variant, n_restarts=n_restarts,
                                     seed=seed, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  FIXED (scale={FIXED_SCALE_BASELINE}) vs. METHOD-OF-MOMENTS Delta init")
    print(f"{'═' * 100}")
    for n_restarts in sorted(set(r["n_restarts"] for r in rows)):
        print(f"\n  --- n_restarts={n_restarts} ---")
        print(f"  {'regime':>8}  {'true_norm':>10}  {'variant':>8}  {'mean fit_norm':>14}  "
              f"{'recovered%':>11}  {'mean final_ll':>14}  {'n':>4}")
        subset = [r for r in rows if r["n_restarts"] == n_restarts]
        by_key = {}
        for r in subset:
            by_key.setdefault((r["regime"], r["variant"]), []).append(r)
        for regime in SKEW_REGIMES:
            for variant in ("fixed", "mom"):
                group = by_key.get((regime, variant), [])
                if not group:
                    continue
                true_norm = group[0]["true_norm"]
                fit_norms = np.array([g["fit_norm"] for g in group])
                lls = np.array([g["final_ll"] for g in group])
                pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
                print(f"  {regime:>8}  {true_norm:>10.3f}  {variant:>8}  "
                      f"{fit_norms.mean():>14.3f}  {pct:>10.0f}%  {lls.mean():>14.5f}  {len(group):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_mom_{ts}.csv"
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
    parser.add_argument("--n-restarts-list", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds, args.n_restarts_list)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
