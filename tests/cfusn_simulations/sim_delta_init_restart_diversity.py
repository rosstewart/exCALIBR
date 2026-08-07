#!/usr/bin/env python3
"""Does cycling the Delta-init MAGNITUDE across the restarts that sign
enumeration already requires -- instead of using the same fixed scale=0.1 on
every restart -- improve recovery, at the SAME total fit count?

Phase 1 established two separate facts that motivate this: (1) no single
fixed scale_factor works across skew regimes (sim_delta_init_magnitude.py);
(2) sign-only-diverse restarts (best-of-4 by likelihood, varying only sign
via lambdaIndex) meaningfully help the large-truth case alone (8%->52%
recovered). This asks whether *also* varying magnitude across those same 4
restarts -- not adding any new restarts, just relabeling what each of the
existing 4 already tries -- does better than sign-diversity alone, per the
user's explicit constraint to keep the required restart count minimal.

Compares, at n_restarts=4 (best-of-4 by final LL), across the same
zero/small/medium/large regimes as the other delta_init_* scripts:
  (a) baseline -- fixed scale=0.1 on all 4 sign-diverse restarts (production)
  (b) cycling  -- same 4 restarts, magnitude also cycles per restart index
                  via init_delta_matrix_cycling_magnitude

Usage:
    python tests/cfusn_simulations/sim_delta_init_restart_diversity.py
    python tests/cfusn_simulations/sim_delta_init_restart_diversity.py --n-seeds 10
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
    sample_cfusn, init_delta_matrix_scaled, init_delta_matrix_cycling_magnitude,
)

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
FIXED_SCALE_BASELINE = 0.1
MAGNITUDE_TIERS = (0.1, 0.5, 1.0)
N_OBS = 5000


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _run_one(magnitude, variant, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    if variant == "baseline":
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_scaled(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                            rng=rng, scale_factor=FIXED_SCALE_BASELINE)
        orig = INIT._init_delta_matrix
        INIT._init_delta_matrix = _init
    else:
        # Cycling: magnitude keyed on restart_idx, set via a mutable cell so
        # the monkeypatched _init_delta_matrix (fixed call signature, no
        # restart_idx param) can pick it up per-restart.
        state = {"restart_idx": 0}

        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_cycling_magnitude(
                cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
                restart_idx=state["restart_idx"], magnitude_tiers=MAGNITUDE_TIERS,
            )
        orig = INIT._init_delta_matrix
        INIT._init_delta_matrix = _init

    try:
        best = None
        for i in range(n_restarts):
            if variant == "cycling":
                state["restart_idx"] = i
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
        for variant in ("baseline", "cycling"):
            for seed in range(n_seeds):
                out = _run_one(magnitude, variant, seed, n_restarts)
                if out is None:
                    continue
                rows.append(dict(regime=regime, variant=variant, n_restarts=n_restarts,
                                 seed=seed, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  SIGN-DIVERSE-ONLY (baseline) vs. SIGN+MAGNITUDE-DIVERSE (cycling) restarts, "
          f"same fit count")
    print(f"{'═' * 100}")
    n_restarts = rows[0]["n_restarts"] if rows else "?"
    print(f"\n  n_restarts={n_restarts}, magnitude_tiers={MAGNITUDE_TIERS}")
    print(f"  {'regime':>8}  {'true_norm':>10}  {'variant':>9}  {'mean fit_norm':>14}  "
          f"{'recovered%':>11}  {'mean final_ll':>14}  {'n':>4}")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["regime"], r["variant"]), []).append(r)
    for regime in SKEW_REGIMES:
        for variant in ("baseline", "cycling"):
            group = by_key.get((regime, variant), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            lls = np.array([g["final_ll"] for g in group])
            pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
            print(f"  {regime:>8}  {true_norm:>10.3f}  {variant:>9}  "
                  f"{fit_norms.mean():>14.3f}  {pct:>10.0f}%  {lls.mean():>14.5f}  {len(group):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_restart_diversity_{ts}.csv"
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
    parser.add_argument("--n-restarts", type=int, default=4)
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds, args.n_restarts)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
