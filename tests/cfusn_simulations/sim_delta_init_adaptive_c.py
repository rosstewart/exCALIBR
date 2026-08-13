#!/usr/bin/env python3
"""Does scaling James-Stein c inversely by a directly-measurable per-column
"information fraction" (fraction of a cluster's rows that actually
contributed to that column's projection) -- c_adaptive = c_base /
info_frac, exactly reducing to c_base at full completeness -- let ONE
formula handle both clean, fully-observed data AND TP53-shaped missing
data well, instead of having to choose between c=1 (excellent on clean,
catastrophic on missing) and c=4 (moderate everywhere, never catastrophic)?

Runs the SAME 4 skew regimes in TWO scenarios:
  "clean"   -- p=4, no missingness (this investigation's original
               clean-data setup, info_frac=1 for every column by
               construction, so c_adaptive should reduce to exactly
               c_base=1.0 here)
  "missing" -- p=16, isolated single-active-skew-column, TP53-shaped
               block missingness, projection_method="partial" (the
               setup sim_delta_init_missingness_isolated_c_sweep.py
               used to find no fixed c does well everywhere)

Compares three variants in each scenario: adaptive (c_base=1.0), fixed
c=1.0, fixed c=4.0 -- no grid sweep needed, so this should be much cheaper
than the earlier c-grid sweeps.

Usage:
    python tests/cfusn_simulations/sim_delta_init_adaptive_c.py
    python tests/cfusn_simulations/sim_delta_init_adaptive_c.py --n-seeds 12
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
    sample_cfusn, inject_block_missingness,
    init_delta_matrix_mom_shrunk_projection, init_delta_matrix_mom_shrunk_adaptive_c,
    _shrinkage_james_stein,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SKEW_REGIMES = {
    "zero": 0.0,
    "small": 0.15,
    "medium": 0.35,
    "large": 0.7,
}
N_OBS = 5000
N_RESTARTS = 4
FALLBACK_SCALE = 0.1
C_BASE = 1.0
FIXED_C_LOW = 1.0
FIXED_C_HIGH = 4.0
VARIANTS = ("adaptive", "fixed_c1", "fixed_c4")

# --- clean scenario ---
CLEAN_P = 4
CLEAN_MU = np.zeros(CLEAN_P)
CLEAN_GAMMA = 0.5 * np.eye(CLEAN_P)
CLEAN_DIRECTION = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

# --- missing scenario (isolated single-active-skew-column, TP53-shaped) ---
MISS_P = 16
MISS_MU = np.zeros(MISS_P)
MISS_GAMMA = 0.5 * np.eye(MISS_P)
_D1 = np.zeros(MISS_P); _D1[[0, 1]] = 1.0
MISS_DIRECTION = np.zeros((MISS_P, 2))
MISS_DIRECTION[:, 0] = _D1 / np.linalg.norm(_D1)
BLOCKS = [list(range(0, 8)), [8, 9, 12], [13], [10], [11], [14], [15]]
BLOCK_OBSERVED_FRAC = [0.226, 0.826, 0.789, 0.514, 0.108, 0.046, 0.018]
BLOCK_FRAC_MISSING = [1 - f for f in BLOCK_OBSERVED_FRAC]


def _make_init(variant, projection_method):
    if variant == "adaptive":
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_mom_shrunk_adaptive_c(
                cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
                fallback_scale=FALLBACK_SCALE, shrinkage_fn=_shrinkage_james_stein,
                projection_method=projection_method, c_base=C_BASE,
            )
        return _init
    c = FIXED_C_LOW if variant == "fixed_c1" else FIXED_C_HIGH

    def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
        return init_delta_matrix_mom_shrunk_projection(
            cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
            fallback_scale=FALLBACK_SCALE, shrinkage_fn=_shrinkage_james_stein, c=c,
            projection_method=projection_method,
        )
    return _init


def _run_one(scenario, magnitude, variant, seed, n_restarts):
    if scenario == "clean":
        Delta_true = CLEAN_DIRECTION * magnitude
        rng = np.random.RandomState(seed)
        X, _ = sample_cfusn(CLEAN_MU, Delta_true, CLEAN_GAMMA, N_OBS, rng)
        projection_method = "complete"
        p = CLEAN_P
    else:
        Delta_true = MISS_DIRECTION * magnitude
        rng = np.random.RandomState(seed)
        X_full, _ = sample_cfusn(MISS_MU, Delta_true, MISS_GAMMA, N_OBS, rng)
        X = inject_block_missingness(X_full, BLOCKS, BLOCK_FRAC_MISSING, rng)
        projection_method = "partial"
        p = MISS_P

    sa = np.ones((N_OBS, 1), dtype=bool)
    _init = _make_init(variant, projection_method)

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
    for scenario in ("clean", "missing"):
        for regime, magnitude in SKEW_REGIMES.items():
            for variant in VARIANTS:
                for seed in range(n_seeds):
                    out = _run_one(scenario, magnitude, variant, seed, n_restarts)
                    if out is None:
                        continue
                    rows.append(dict(scenario=scenario, regime=regime, variant=variant,
                                     seed=seed, n_restarts=n_restarts, **out))
    return rows


def _combined_scores(rows, scenario, variant):
    by_key = {}
    for r in rows:
        if r["scenario"] != scenario or r["variant"] != variant:
            continue
        by_key.setdefault(r["regime"], []).append(r)
    vals = {}
    for regime in SKEW_REGIMES:
        group = by_key.get(regime, [])
        if not group:
            continue
        true_norm = group[0]["true_norm"]
        fit_norms = np.array([g["fit_norm"] for g in group])
        if true_norm > 1e-9:
            vals[regime] = abs(100 * fit_norms.mean() / true_norm - 100)
        else:
            vals[regime] = fit_norms.mean()
    return vals


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  ADAPTIVE c (c_base={C_BASE}) vs. FIXED c={FIXED_C_LOW} vs. FIXED c={FIXED_C_HIGH}: "
          f"clean vs. TP53-shaped missing data")
    print(f"{'═' * 100}")
    for scenario in ("clean", "missing"):
        print(f"\n  === scenario: {scenario} ===")
        print(f"  {'variant':>10}  {'zero':>8}  {'small |err%|':>13}  "
              f"{'medium |err%|':>14}  {'large |err%|':>13}")
        for variant in VARIANTS:
            v = _combined_scores(rows, scenario, variant)
            print(f"  {variant:>10}  {v.get('zero', float('nan')):>8.3f}  "
                  f"{v.get('small', float('nan')):>13.1f}  {v.get('medium', float('nan')):>14.1f}  "
                  f"{v.get('large', float('nan')):>13.1f}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_adaptive_c_{ts}.csv"
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
