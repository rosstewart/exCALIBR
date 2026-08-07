#!/usr/bin/env python3
"""Does warm-starting bootstrap replicate fits from a converged full-dataset
fit -- one extra restart initialized directly at that candidate, on top of
the standard restart set -- meaningfully improve recovery at SMALL
per-replicate N, for the cost of exactly one extra fit?

Motivated by the user's suggestion: before bootstrapping, fit the full
(non-bootstrapped) dataset a few times to find converged Deltas, then use
the best of those as a warm start for the actual bootstrap replicate fits.
single_fit already supports this via its existing initial_params/
initial_weights override (cfusn/fit.py ~lines 109-121) -- no new override
machinery is needed, just a new caller of it.

Design:
  1. Generate one large "population" dataset (N_full=20000) from known true
     params.
  2. Fit it 3-5 times from scratch (lambdaIndex=0..3), keep the highest-LL
     converged (mu, Delta, Gamma) as the warm-start candidate.
  3. Generate bootstrap replicates by RESAMPLING ROWS from that population
     (literal bootstrap, matching production's pattern_stratified_bootstrap
     -- resampling one underlying dataset) at a small replicate N.
  4. Per replicate, compare:
       (A) standard restart set (best-of-N by LL)
       (B) same set PLUS one extra restart initialized directly at the
           warm-start candidate via initial_params/initial_weights --
           best-of-(N+1)
     across the 4 usual skew regimes.

Reports recovery (A) vs (B), how often B's LL >= A's, and how often the
warm-start restart is the one actually selected.

Usage:
    python tests/cfusn_simulations/sim_delta_init_warmstart.py
    python tests/cfusn_simulations/sim_delta_init_warmstart.py --n-seeds 10
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
from tests.cfusn_simulations.sim_utils import sample_cfusn

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
N_FULL = 20000
N_FULL_FIT_RESTARTS = 4
N_REPLICATE = 300
N_REPLICATE_RESTARTS = 4  # standard restart budget for (A); (B) = this + 1


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _fit_once(X, sa, seed, restart_idx, initial_params=None, initial_weights=None):
    kwargs = dict(
        num_components=1, constrained=False, init_method="kmeans",
        init_constraint_adjustment="scale", multivariate=True, latent_q=2,
        check_monotonic=False, num_fits=1, fit_seed=int(seed * 100 + restart_idx),
        lambdaIndex=restart_idx, max_em_iters=300, verbose=False, verbose_init=False,
    )
    if initial_params is not None:
        kwargs["initial_params"] = initial_params
        kwargs["initial_weights"] = initial_weights
    return tryToFit(X, sa, **kwargs)


def _extract(result):
    params = result.get("component_params", [])
    if not params or len(params[0]) == 0:
        return None
    ll = result["likelihoods"][-1] if len(result.get("likelihoods", [])) else -np.inf
    return ll, params


def _find_warm_start_candidate(Delta_true, seed):
    """Fit the full (non-bootstrapped) population dataset N_FULL_FIT_RESTARTS
    times from scratch; return the highest-LL converged (mu, Delta, Gamma)
    as the warm-start candidate, plus the population X for later resampling.
    """
    rng = np.random.RandomState(seed)
    X_full, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_FULL, rng)
    sa_full = np.ones((N_FULL, 1), dtype=bool)

    best = None
    for i in range(N_FULL_FIT_RESTARTS):
        result = _fit_once(X_full, sa_full, seed, i)
        out = _extract(result)
        if out is None:
            continue
        ll, params = out
        if best is None or ll > best[0]:
            best = (ll, params)

    if best is None:
        return None, X_full
    return best[1], X_full


def _run_one_replicate(X_full, Delta_true, warm_start_params, seed, rep_idx):
    rng = np.random.RandomState(seed * 1000 + rep_idx)
    idx = rng.choice(len(X_full), size=N_REPLICATE, replace=True)
    X_rep = X_full[idx]
    sa_rep = np.ones((N_REPLICATE, 1), dtype=bool)

    # (A) standard restart set
    best_a = None
    for i in range(N_REPLICATE_RESTARTS):
        out = _extract(_fit_once(X_rep, sa_rep, seed * 1000 + rep_idx, i))
        if out is None:
            continue
        if best_a is None or out[0] > best_a[0]:
            best_a = out

    # (B) standard set + one extra warm-started restart
    best_b = best_a
    warm_selected = False
    if warm_start_params is not None:
        warm_result = _fit_once(
            X_rep, sa_rep, seed * 1000 + rep_idx, N_REPLICATE_RESTARTS,
            initial_params=warm_start_params,
            initial_weights=np.ones((1, 1)),
        )
        warm_out = _extract(warm_result)
        if warm_out is not None and (best_b is None or warm_out[0] > best_b[0]):
            best_b = warm_out
            warm_selected = True

    if best_a is None or best_b is None:
        return None

    def _delta_norm(best):
        _, params = best
        return float(np.linalg.norm(np.asarray(params[0][1])))

    return dict(
        true_norm=float(np.linalg.norm(Delta_true)),
        fit_norm_a=_delta_norm(best_a),
        fit_norm_b=_delta_norm(best_b),
        final_ll_a=float(best_a[0]),
        final_ll_b=float(best_b[0]),
        b_beats_a=bool(best_b[0] >= best_a[0]),
        warm_selected=warm_selected,
    )


def run_sweep(n_seeds, n_replicates):
    rows = []
    for regime, magnitude in SKEW_REGIMES.items():
        Delta_true = _delta_true(magnitude)
        for seed in range(n_seeds):
            warm_start_params, X_full = _find_warm_start_candidate(Delta_true, seed)
            for rep_idx in range(n_replicates):
                out = _run_one_replicate(X_full, Delta_true, warm_start_params, seed, rep_idx)
                if out is None:
                    continue
                rows.append(dict(regime=regime, seed=seed, rep_idx=rep_idx, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  PRE-BOOTSTRAP FULL-DATASET WARM START: (A) best-of-{N_REPLICATE_RESTARTS} vs. "
          f"(B) best-of-{N_REPLICATE_RESTARTS}+1-warm-started, N_replicate={N_REPLICATE}")
    print(f"{'═' * 100}")
    by_regime = {}
    for r in rows:
        by_regime.setdefault(r["regime"], []).append(r)

    print(f"\n  {'regime':>8}  {'true_norm':>10}  {'A recov%':>9}  {'B recov%':>9}  "
          f"{'A mean_ll':>11}  {'B mean_ll':>11}  {'B>=A %':>8}  {'warm sel %':>11}  {'n':>4}")
    for regime in SKEW_REGIMES:
        group = by_regime.get(regime, [])
        if not group:
            continue
        true_norm = group[0]["true_norm"]
        fn_a = np.array([g["fit_norm_a"] for g in group])
        fn_b = np.array([g["fit_norm_b"] for g in group])
        ll_a = np.array([g["final_ll_a"] for g in group])
        ll_b = np.array([g["final_ll_b"] for g in group])
        b_beats_a = np.array([g["b_beats_a"] for g in group])
        warm_sel = np.array([g["warm_selected"] for g in group])
        pct_a = (100 * fn_a.mean() / true_norm) if true_norm > 1e-9 else float("nan")
        pct_b = (100 * fn_b.mean() / true_norm) if true_norm > 1e-9 else float("nan")
        print(f"  {regime:>8}  {true_norm:>10.3f}  {pct_a:>8.0f}%  {pct_b:>8.0f}%  "
              f"{ll_a.mean():>11.4f}  {ll_b.mean():>11.4f}  {100*b_beats_a.mean():>7.0f}%  "
              f"{100*warm_sel.mean():>10.0f}%  {len(group):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_warmstart_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="number of distinct populations per regime")
    parser.add_argument("--n-replicates", type=int, default=10,
                        help="bootstrap replicates drawn per population")
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds, args.n_replicates)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
