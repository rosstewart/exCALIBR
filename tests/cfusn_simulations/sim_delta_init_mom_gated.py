#!/usr/bin/env python3
"""Does gating the method-of-moments Delta-init magnitude on skewness
SIGNIFICANCE (z-score vs. the skewness estimator's own standard error) fix
the false-positive problem sim_delta_init_mom.py found -- manufacturing
substantial spurious skew (fitted norm ~0.48) from pure sampling noise when
true skew is exactly 0 -- while keeping the real win for medium/large true
skew (25%->93%, 12%->112% recovered)?

The gate threshold (z_threshold: how many standard errors of skewness-
estimator noise the sample skewness must clear before being trusted) is
swept rather than picked by hand, per the "data-driven, not arbitrary"
requirement -- this script's job is to let the sweep itself show which
threshold (if any) actually balances both regimes well, not to assert one.

Same 4 true-skew regimes (zero/small/medium/large) and N as the previous
two scripts in this chain, for direct comparability.

Usage:
    python tests/cfusn_simulations/sim_delta_init_mom_gated.py
    python tests/cfusn_simulations/sim_delta_init_mom_gated.py --n-seeds 10
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
from tests.cfusn_simulations.sim_utils import sample_cfusn, init_delta_matrix_mom_gated

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
Z_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
N_OBS = 5000
FALLBACK_SCALE = 0.1


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _run_one(magnitude, z_threshold, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
        return init_delta_matrix_mom_gated(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                           rng=rng, fallback_scale=FALLBACK_SCALE,
                                           z_threshold=z_threshold)

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
        for z_threshold in Z_THRESHOLDS:
            for seed in range(n_seeds):
                out = _run_one(magnitude, z_threshold, seed, n_restarts)
                if out is None:
                    continue
                rows.append(dict(regime=regime, z_threshold=z_threshold, seed=seed,
                                 n_restarts=n_restarts, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  SIGNIFICANCE-GATED METHOD-OF-MOMENTS Delta INIT: z_threshold sweep")
    print(f"{'═' * 100}")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["regime"], r["z_threshold"]), []).append(r)

    for regime in SKEW_REGIMES:
        print(f"\n  --- regime={regime} (true_norm={SKEW_REGIMES[regime]:.2f} scaled) ---")
        print(f"  {'z_threshold':>11}  {'mean fit_norm':>14}  {'recovered%':>11}  "
              f"{'mean final_ll':>14}  {'n':>4}")
        for z_threshold in Z_THRESHOLDS:
            group = by_key.get((regime, z_threshold), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            lls = np.array([g["final_ll"] for g in group])
            pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
            print(f"  {z_threshold:>11.1f}  {fit_norms.mean():>14.3f}  {pct:>10.0f}%  "
                  f"{lls.mean():>14.5f}  {len(group):>4}")

    # Summary: for each z_threshold, report the worst-case deviation from
    # "ideal" across regimes (zero should be near 0, others near 100%) --
    # a simple combined score to make the tradeoff legible at a glance.
    print(f"\n  --- summary: combined score per z_threshold ---")
    print(f"  (zero-regime score = mean fit_norm itself [want small]; "
          f"other regimes = |recovered% - 100| [want small])")
    print(f"  {'z_threshold':>11}  {'zero fit_norm':>14}  {'small |err%|':>13}  "
          f"{'medium |err%|':>14}  {'large |err%|':>13}")
    for z_threshold in Z_THRESHOLDS:
        vals = {}
        for regime in SKEW_REGIMES:
            group = by_key.get((regime, z_threshold), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            if true_norm > 1e-9:
                vals[regime] = abs(100 * fit_norms.mean() / true_norm - 100)
            else:
                vals[regime] = fit_norms.mean()
        print(f"  {z_threshold:>11.1f}  {vals.get('zero', float('nan')):>14.3f}  "
              f"{vals.get('small', float('nan')):>13.1f}  {vals.get('medium', float('nan')):>14.1f}  "
              f"{vals.get('large', float('nan')):>13.1f}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_mom_gated_{ts}.csv"
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
    parser.add_argument("--n-restarts", type=int, default=1,
                        help="1 = the practically-relevant single-fit case")
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds, args.n_restarts)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
