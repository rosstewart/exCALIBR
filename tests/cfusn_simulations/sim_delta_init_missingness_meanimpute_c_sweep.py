#!/usr/bin/env python3
"""Does meanimpute's lower excess skewness-estimator noise (confirmed
directly: ~5x the clean-data variance vs. partial-projection's ~7-10x,
across all regimes, in the same isolated single-active-skew-column
diagnostic) translate into a single James-Stein c value that works
reasonably for BOTH clean and TP53-shaped missing data -- succeeding where
partial-projection failed on every tuning lever tried (the shrinkage
constant itself, an effective-sample-size correction, and a
min_weight_frac threshold sweep)?

Same isolated single-active-skew-column setup (p=16, only column 1 -- dims
0,1 -- carries real skew, removing the two-competing-signals confound), same
TP53-shaped block missingness, same James-Stein c grid, same n_restarts=4
as sim_delta_init_missingness_isolated_c_sweep.py -- the only change is
projection_method="meanimpute" instead of "partial".

Usage:
    python tests/cfusn_simulations/sim_delta_init_missingness_meanimpute_c_sweep.py
    python tests/cfusn_simulations/sim_delta_init_missingness_meanimpute_c_sweep.py --n-seeds 10
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
from tests.cfusn_simulations.sim_utils import (
    sample_cfusn, inject_block_missingness, init_delta_matrix_mom_shrunk_projection,
    _shrinkage_james_stein,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

P = 16
MU_TRUE = np.zeros(P)
GAMMA_TRUE = 0.5 * np.eye(P)
_D1 = np.zeros(P); _D1[[0, 1]] = 1.0
DIRECTION = np.zeros((P, 2))
DIRECTION[:, 0] = _D1 / np.linalg.norm(_D1)

BLOCKS = [list(range(0, 8)), [8, 9, 12], [13], [10], [11], [14], [15]]
BLOCK_OBSERVED_FRAC = [0.226, 0.826, 0.789, 0.514, 0.108, 0.046, 0.018]
BLOCK_FRAC_MISSING = [1 - f for f in BLOCK_OBSERVED_FRAC]

SKEW_REGIMES = {
    "zero": 0.0,
    "small": 0.15,
    "medium": 0.35,
    "large": 0.7,
}
Z_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
JS_C_VALUES = [z ** 2 for z in Z_THRESHOLDS]
FALLBACK_SCALE = 0.1
N_OBS = 5000
N_RESTARTS = 4


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _run_one(magnitude, c, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X_full, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    X = inject_block_missingness(X_full, BLOCKS, BLOCK_FRAC_MISSING, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
        return init_delta_matrix_mom_shrunk_projection(
            cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
            fallback_scale=FALLBACK_SCALE, shrinkage_fn=_shrinkage_james_stein, c=c,
            projection_method="meanimpute",
        )

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


def run_sweep(n_seeds, n_restarts):
    rows = []
    for regime, magnitude in SKEW_REGIMES.items():
        for c in JS_C_VALUES:
            for seed in range(n_seeds):
                out = _run_one(magnitude, c, seed, n_restarts)
                if out is None:
                    continue
                rows.append(dict(regime=regime, c=c, seed=seed, n_restarts=n_restarts, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  MEANIMPUTE James-Stein c SWEEP under TP53-shaped missingness "
          f"(isolated single-column truth, n_restarts={N_RESTARTS})")
    print(f"{'═' * 100}")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["regime"], r["c"]), []).append(r)

    for regime in SKEW_REGIMES:
        print(f"\n  --- regime={regime} (true_norm={SKEW_REGIMES[regime]:.2f} scaled) ---")
        print(f"  {'c':>8}  {'mean fit_norm':>14}  {'recovered%':>11}  {'mean final_ll':>14}  {'n':>4}")
        for c in JS_C_VALUES:
            group = by_key.get((regime, c), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            lls = np.array([g["final_ll"] for g in group])
            pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
            print(f"  {c:>8.2f}  {fit_norms.mean():>14.3f}  {pct:>10.0f}%  {lls.mean():>14.5f}  {len(group):>4}")

    print(f"\n  --- summary: combined score per c ---")
    print(f"  (zero-regime score = mean fit_norm itself [want small]; "
          f"other regimes = |recovered% - 100| [want small])")
    print(f"  {'c':>8}  {'zero fit_norm':>14}  {'small |err%|':>13}  "
          f"{'medium |err%|':>14}  {'large |err%|':>13}")
    for c in JS_C_VALUES:
        vals = {}
        for regime in SKEW_REGIMES:
            group = by_key.get((regime, c), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            if true_norm > 1e-9:
                vals[regime] = abs(100 * fit_norms.mean() / true_norm - 100)
            else:
                vals[regime] = fit_norms.mean()
        print(f"  {c:>8.2f}  {vals.get('zero', float('nan')):>14.3f}  "
              f"{vals.get('small', float('nan')):>13.1f}  {vals.get('medium', float('nan')):>14.1f}  "
              f"{vals.get('large', float('nan')):>13.1f}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_missingness_meanimpute_c_sweep_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-restarts", type=int, default=N_RESTARTS)
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds, args.n_restarts)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
