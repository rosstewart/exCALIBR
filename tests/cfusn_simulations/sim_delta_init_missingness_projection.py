#!/usr/bin/env python3
"""Does replacing _init_delta_matrix's "complete rows across ALL p
dimensions" gate with a projection-aware alternative actually unlock usable
skew sign/magnitude estimation for real multi-assay genes like TP53 --
where a same-scale, same-seed production before/after sanity refit showed
BIT-IDENTICAL results, because 0/9911 real TP53 rows are complete across
all 16 dims (confirmed by direct inspection), silently disabling both the
existing skew-sign heuristic and the new shrinkage-based magnitude
estimate on every cluster, every restart, for the entire history of this
pipeline on TP53?

Stress test mirrors TP53's actual structure (confirmed via pairwise
co-observation inspection: the graph is fully CONNECTED -- every pair of
dims shares real overlap -- yet the joint 16-way intersection is exactly
empty): p=16, block-correlated missingness via inject_block_missingness
with block observed-fractions matching real TP53 approximately (dims 0-7
always jointly observed at ~22.6%; dims {8,9,12} jointly at ~82.6%; dim 13
at ~78.9%; dim 10 at ~51.4%; dim 11 at ~10.8%; dim 14 at ~4.6%; dim 15 at
~1.8%). True skew loads on dims spanning the two largest blocks (0-7 and
8/9/12/13), matching how real biological skew would plausibly appear
across correlated readouts of the same assay(s).

Compares projection_method in {"complete" (production's current, expected
to basically never activate here), "partial", "meanimpute"} at the
already-calibrated James-Stein c=1.0, n_restarts=4, across the same 4 skew
regimes used throughout this investigation. Reports the ACTIVATION RATE
(how often each method produces a usable >=8-point projection at all --
directly answering whether the fix does anything), sign accuracy when
activated, and post-EM recovery.

Usage:
    python tests/cfusn_simulations/sim_delta_init_missingness_projection.py
    python tests/cfusn_simulations/sim_delta_init_missingness_projection.py --n-seeds 10
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
    _shrinkage_james_stein, _partial_projection, _meanimpute_projection,
    _skewness_z_score,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

P = 16
MU_TRUE = np.zeros(P)
GAMMA_TRUE = 0.5 * np.eye(P)

# Column 1 loads on 2 dims of block A (TP53's largest but sparsest block);
# column 2 loads on 2 dims of block B (TP53's densest block). Concentrated
# on 2 dims each (not spread across the whole 8/4-dim block) so the
# induced skew-covariance isn't diluted below GAMMA_TRUE's isotropic noise
# floor -- diagnosed directly: spreading evenly across all 8 of block A's
# dims left Omega's diagonal only ~12% above the 0.5 baseline (0.561 vs
# 0.5), so the actual TOP eigenvector ended up aligned with column 2
# (dot=0.97) rather than column 1 (dot=-0.17) at magnitude=0.7 -- a test
# design flaw, not a projection-method finding; this concentrated version
# restores the same signal-to-noise regime the earlier (successful) 4D
# tests used.
_D1 = np.zeros(P); _D1[[0, 1]] = 1.0
_D2 = np.zeros(P); _D2[[8, 9]] = 1.0
DIRECTION = np.stack([_D1 / np.linalg.norm(_D1), _D2 / np.linalg.norm(_D2)], axis=1)

# Blocks + observed-fractions matching real TP53 (see sim_utils.inject_block_missingness
# docstring for the source inspection); block_frac_missing = 1 - observed_fraction.
BLOCKS = [
    list(range(0, 8)),  # block A
    [8, 9, 12],          # block B
    [13],                 # block C (mostly overlaps B in reality; independent here)
    [10],                  # block D
    [11],                   # block E
    [14],                    # block F
    [15],                     # block G
]
BLOCK_OBSERVED_FRAC = [0.226, 0.826, 0.789, 0.514, 0.108, 0.046, 0.018]
BLOCK_FRAC_MISSING = [1 - f for f in BLOCK_OBSERVED_FRAC]

SKEW_REGIMES = {
    "zero": 0.0,
    "small": 0.15,
    "medium": 0.35,
    "large": 0.7,
}
JS_C = 1.0
FALLBACK_SCALE = 0.1
N_OBS = 5000
N_RESTARTS = 4
PROJECTION_METHODS = ("complete", "partial", "meanimpute")


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _activation_diagnostic(X, Delta_true, method, seed):
    """One-shot (no EM) check of how often this projection method produces
    a usable (>=8-point) projection, and whether its sign matches truth,
    computed directly on the full (single-cluster) dataset -- cheap, no
    monkeypatching needed.

    Deliberately uses the KNOWN TRUE column-1 direction as the projection
    axis here, rather than PCA's estimated top eigenvector: with p=16 and
    12 pure-noise "background" dims, PCA's top-2 eigenvectors can end up as
    a rotated mix of both true skew directions when their induced-variance
    boosts are comparable (confirmed directly: dot products of 0.5-0.8
    with BOTH columns simultaneously, not cleanly separated) -- a known,
    SEPARATE eigenvector-selection problem (see sim_delta_init_direction.py
    / Item 4), not what this script is testing. Isolating the projection
    mechanism from that confound by using the true axis directly answers
    the actual question here: given a reasonable projection axis, does
    "complete"/"partial"/"meanimpute" correctly estimate sign and
    magnitude from it under TP53-shaped missingness. The full EM-based
    recovery metric below still goes through production's real PCA-based
    axis selection unmodified, so the compounded real-world effect is
    still captured there.
    """
    evec = DIRECTION[:, 0].copy()

    if method == "complete":
        complete_rows = ~np.isnan(X).any(axis=1)
        Xc_comp = X[complete_rows]
        projected = Xc_comp @ evec if len(Xc_comp) >= 8 else np.array([])
    elif method == "partial":
        projected = _partial_projection(X, evec, min_weight_frac=0.5)
    else:
        projected = _meanimpute_projection(X, evec)

    activated = len(projected) >= 8
    sign_correct = None
    if activated:
        m3, z = _skewness_z_score(projected)
        # CFUSN's skew always pushes positive along Delta's own columns
        # (T ~ TN_q(0,I,R+^q) is truncated to T>=0), so projecting exactly
        # onto DIRECTION[:,0] itself has a known, trivial true sign of +1
        # whenever magnitude > 0 -- no dot-product needed since evec IS
        # that direction here.
        if abs(m3) > 1e-6:
            sign_correct = bool(int(np.sign(m3)) == 1)
    return activated, sign_correct, len(projected)


def _run_one(magnitude, method, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X_full, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    X = inject_block_missingness(X_full, BLOCKS, BLOCK_FRAC_MISSING, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    activated, sign_correct, n_used = _activation_diagnostic(X, Delta_true, method, seed)

    def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
        return init_delta_matrix_mom_shrunk_projection(
            cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
            fallback_scale=FALLBACK_SCALE, shrinkage_fn=_shrinkage_james_stein, c=JS_C,
            projection_method=method,
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
        activated=activated,
        sign_correct=sign_correct,
        n_used=n_used,
    )


def run_sweep(n_seeds, n_restarts):
    rows = []
    for regime, magnitude in SKEW_REGIMES.items():
        for method in PROJECTION_METHODS:
            for seed in range(n_seeds):
                out = _run_one(magnitude, method, seed, n_restarts)
                if out is None:
                    continue
                rows.append(dict(regime=regime, method=method, n_restarts=n_restarts,
                                 seed=seed, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  PROJECTION-AWARE SKEW ESTIMATION UNDER TP53-SHAPED BLOCK MISSINGNESS "
          f"(p={P}, 0 fully-complete rows by construction)")
    print(f"{'═' * 100}")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["regime"], r["method"]), []).append(r)

    for regime in SKEW_REGIMES:
        print(f"\n  --- regime={regime} (true_norm={SKEW_REGIMES[regime]:.2f} scaled) ---")
        print(f"  {'method':>12}  {'activation%':>11}  {'sign_acc%':>10}  {'mean fit_norm':>14}  "
              f"{'recovered%':>11}  {'mean final_ll':>14}  {'n':>4}")
        for method in PROJECTION_METHODS:
            group = by_key.get((regime, method), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            lls = np.array([g["final_ll"] for g in group])
            act_pct = 100 * np.mean([g["activated"] for g in group])
            sign_vals = [g["sign_correct"] for g in group if g["sign_correct"] is not None]
            sign_pct = 100 * np.mean(sign_vals) if sign_vals else float("nan")
            pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
            print(f"  {method:>12}  {act_pct:>10.0f}%  {sign_pct:>9.0f}%  "
                  f"{fit_norms.mean():>14.3f}  {pct:>10.0f}%  {lls.mean():>14.5f}  {len(group):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_missingness_projection_{ts}.csv"
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
