#!/usr/bin/env python3
"""Does column 1's DIRECTION -- currently the top covariance eigenvector
(PCA, variance-maximizing) -- explain why boosting init magnitude sometimes
converged to a WORSE likelihood in sim_delta_init_magnitude.py? If the true
skew axis isn't the direction of greatest variance, PCA's direction is
wrong from the start, and no amount of magnitude tuning fixes a
misdirected Delta.

Stress test: deliberately misalign the true skew axis from the top
covariance eigenvector. Gamma_true has one large, skew-free "noise"
dimension (variance 3.0) and three smaller dimensions (variance 0.3) that
carry the real skew -- so the top PCA eigenvector points mostly at the
noise dimension, while the true column-1 skew direction lies mostly in the
lower-variance dimensions.

Compares direction_method in {"pca" (production default), "mardia",
"projection_pursuit"} at the 4 usual magnitude regimes, holding magnitude
fixed-scale (scale_factor=0.1) throughout so this isolates direction from
the separate magnitude questions in Items 1-3. Column 2 always keeps PCA
(see init_delta_matrix_direction's docstring for why).

Metrics: PRE-EM cosine similarity between the chosen and true column-1
direction (a pure init-quality diagnostic, no EM noise); POST-EM recovered-%
and alignment (does a better-directed init actually reach a better
converged Delta).

Usage:
    python tests/cfusn_simulations/sim_delta_init_direction.py
    python tests/cfusn_simulations/sim_delta_init_direction.py --n-seeds 10
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
from tests.cfusn_simulations.sim_utils import sample_cfusn, init_delta_matrix_direction

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MU_TRUE = np.zeros(4)
# One large noise dimension (idx 0, variance 3.0, carries NO skew) + three
# smaller dimensions that carry the real skew -- deliberately misaligns the
# top-variance (PCA) direction from the true skew direction.
GAMMA_TRUE = np.diag([3.0, 0.3, 0.3, 0.3])
# True column-1 skew direction lives entirely in the low-variance dims,
# normalized; column 2 is a throwaway orthogonal-ish direction (only column
# 1's alignment is being tested here).
_RAW_DIR1 = np.array([0.0, 1.0, 0.3, -0.3])
DIRECTION_1 = _RAW_DIR1 / np.linalg.norm(_RAW_DIR1)
_RAW_DIR2 = np.array([0.0, 0.3, -1.0, 0.3])
DIRECTION_2 = _RAW_DIR2 / np.linalg.norm(_RAW_DIR2)

SKEW_REGIMES = {
    "zero": 0.0,
    "small": 0.15,
    "medium": 0.35,
    "large": 0.7,
}
DIRECTION_METHODS = ["pca", "mardia", "projection_pursuit"]
FIXED_SCALE = 0.1
N_OBS = 5000


def _delta_true(magnitude):
    Delta = np.zeros((4, 2))
    Delta[:, 0] = magnitude * DIRECTION_1
    Delta[:, 1] = magnitude * DIRECTION_2
    return Delta


def _pre_em_cosine(cov, Xc, direction_method, seed):
    """Instantiate init_delta_matrix_direction once (no EM) purely to read
    off which column-1 direction it picked, for the pre-EM diagnostic."""
    rng = np.random.RandomState(seed)
    Delta_init = init_delta_matrix_direction(
        cov, 4, 2, Xc=Xc, cluster_sign_pattern=np.array([1, 1]), rng=rng,
        direction_method=direction_method, scale_factor=FIXED_SCALE,
    )
    col1 = Delta_init[:, 0]
    norm = np.linalg.norm(col1)
    if norm < 1e-12:
        return float("nan")
    col1 = col1 / norm
    return float(abs(np.dot(col1, DIRECTION_1)))  # abs: sign is separately enumerated


def _run_one(magnitude, direction_method, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    cov = np.cov(X, rowvar=False)
    pre_cos = _pre_em_cosine(cov, X, direction_method, seed)

    def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
        return init_delta_matrix_direction(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                           rng=rng, direction_method=direction_method,
                                           scale_factor=FIXED_SCALE)

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
    col1_fit = Delta_fit[:, 0]
    norm = np.linalg.norm(col1_fit)
    post_cos = float(abs(np.dot(col1_fit / norm, DIRECTION_1))) if norm > 1e-12 else float("nan")
    return dict(
        true_norm=float(np.linalg.norm(Delta_true[:, 0])),
        fit_norm=float(norm),
        final_ll=float(ll),
        pre_em_cosine=pre_cos,
        post_em_cosine=post_cos,
    )


def run_sweep(n_seeds, n_restarts):
    rows = []
    for regime, magnitude in SKEW_REGIMES.items():
        for direction_method in DIRECTION_METHODS:
            for seed in range(n_seeds):
                out = _run_one(magnitude, direction_method, seed, n_restarts)
                if out is None:
                    continue
                rows.append(dict(regime=regime, direction_method=direction_method,
                                 n_restarts=n_restarts, seed=seed, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  DIRECTION-FINDER COMPARISON: PCA vs. Mardia vs. projection pursuit "
          f"(misaligned-truth stress test)")
    print(f"{'═' * 100}")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["regime"], r["direction_method"]), []).append(r)

    for regime in SKEW_REGIMES:
        print(f"\n  --- regime={regime} (true col-1 norm={SKEW_REGIMES[regime]:.2f}) ---")
        print(f"  {'method':>20}  {'pre-EM cos':>11}  {'post-EM cos':>12}  "
              f"{'mean fit_norm':>14}  {'recovered%':>11}  {'mean final_ll':>14}  {'n':>4}")
        for method in DIRECTION_METHODS:
            group = by_key.get((regime, method), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            lls = np.array([g["final_ll"] for g in group])
            pre_cos = np.array([g["pre_em_cosine"] for g in group])
            post_cos = np.array([g["post_em_cosine"] for g in group])
            pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
            print(f"  {method:>20}  {np.nanmean(pre_cos):>11.3f}  {np.nanmean(post_cos):>12.3f}  "
                  f"{fit_norms.mean():>14.3f}  {pct:>10.0f}%  {lls.mean():>14.5f}  {len(group):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_direction_{ts}.csv"
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
    parser.add_argument("--n-restarts", type=int, default=1)
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds, args.n_restarts)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
