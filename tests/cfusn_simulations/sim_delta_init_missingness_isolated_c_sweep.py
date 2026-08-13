#!/usr/bin/env python3
"""Removes the confound found in sim_delta_init_missingness_c_sweep.py's
results: that script's ground truth had TWO comparably-sized skew
directions (columns 1 and 2), and direct inspection showed the real
covariance's top-2 eigenvectors were BLENDS of both true directions, not
cleanly aligned with either -- especially at small/medium true skew (e.g.
at magnitude=0.15, eigvec[0] had dot=-0.40 with column 1 AND -0.20 with
column 2 simultaneously). That means the earlier finding "c=1.0 needs
retuning under missingness" may have been measuring eigenvector confusion
(a known, separate problem -- see sim_delta_init_direction.py), not a
missingness-specific z-score miscalibration.

This script isolates the two candidate explanations by zeroing out column
2's TRUE magnitude entirely (only column 1 carries real skew) -- removing
the "two competing signals" confound -- while keeping everything else
identical (TP53-shaped block missingness, p=16, same magnitude regimes for
column 1, same James-Stein c grid). A direct top-eigenvector check with
this single-column truth (run separately) showed dot products of only
-0.40 to -0.76 with column 1 across regimes -- STILL not close to +-1 --
suggesting missingness-driven covariance ESTIMATION noise (fewer
co-observed rows per pair than clean data's full N) is itself enough to
scramble PCA's eigenvector selection at p=16, even with no second true
signal to compete with.

Runs BOTH the plain z-score ("plain", init_delta_matrix_mom_shrunk_projection
projection_method="partial") and the effective-sample-size-corrected z-score
("ess", init_delta_matrix_mom_shrunk_projection_ess) at the same c grid, so
this also re-tests whether the ESS correction does anything meaningful once
whatever eigenvector PCA actually selects (which may now have diffuse,
genuinely-partial per-row weights across many of the 16 dims, unlike the
degenerate case found before where evec's support fell entirely inside one
missingness block) makes _partial_projection's weight_frac non-trivial.

Usage:
    python tests/cfusn_simulations/sim_delta_init_missingness_isolated_c_sweep.py
    python tests/cfusn_simulations/sim_delta_init_missingness_isolated_c_sweep.py --n-seeds 10
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
    init_delta_matrix_mom_shrunk_projection, init_delta_matrix_mom_shrunk_projection_ess,
    _shrinkage_james_stein,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

P = 16
MU_TRUE = np.zeros(P)
GAMMA_TRUE = 0.5 * np.eye(P)
# Column 2 fixed at exactly zero -- only column 1 (dims 0,1, block A) carries
# real skew, removing the two-competing-signals confound.
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
VARIANTS = ("plain", "ess")


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _run_one(magnitude, variant, c, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X_full, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    X = inject_block_missingness(X_full, BLOCKS, BLOCK_FRAC_MISSING, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    if variant == "plain":
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_mom_shrunk_projection(
                cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
                fallback_scale=FALLBACK_SCALE, shrinkage_fn=_shrinkage_james_stein, c=c,
                projection_method="partial",
            )
    else:
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_mom_shrunk_projection_ess(
                cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
                fallback_scale=FALLBACK_SCALE, shrinkage_fn=_shrinkage_james_stein, c=c,
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
        for variant in VARIANTS:
            for c in JS_C_VALUES:
                for seed in range(n_seeds):
                    out = _run_one(magnitude, variant, c, seed, n_restarts)
                    if out is None:
                        continue
                    rows.append(dict(regime=regime, variant=variant, c=c, seed=seed,
                                     n_restarts=n_restarts, **out))
    return rows


def _combined_scores(rows, variant):
    by_key = {}
    for r in rows:
        if r["variant"] != variant:
            continue
        by_key.setdefault((r["regime"], r["c"]), []).append(r)
    out = {}
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
        out[c] = vals
    return out


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  ISOLATED (single active skew column) James-Stein c SWEEP: plain vs. ESS-corrected")
    print(f"  under TP53-shaped missingness (n_restarts={N_RESTARTS})")
    print(f"{'═' * 100}")

    for variant in VARIANTS:
        label = "plain z-score" if variant == "plain" else "ESS-corrected z-score"
        print(f"\n  === {label} ===")
        scores = _combined_scores(rows, variant)
        print(f"  {'c':>8}  {'zero fit_norm':>14}  {'small |err%|':>13}  "
              f"{'medium |err%|':>14}  {'large |err%|':>13}")
        for c in JS_C_VALUES:
            v = scores.get(c, {})
            print(f"  {c:>8.2f}  {v.get('zero', float('nan')):>14.3f}  "
                  f"{v.get('small', float('nan')):>13.1f}  {v.get('medium', float('nan')):>14.1f}  "
                  f"{v.get('large', float('nan')):>13.1f}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_missingness_isolated_c_sweep_{ts}.csv"
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
