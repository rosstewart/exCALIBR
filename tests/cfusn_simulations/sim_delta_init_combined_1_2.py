#!/usr/bin/env python3
"""Does combining Item 1 (restart-indexed magnitude cycling) with Item 2
(continuous James-Stein shrinkage of the MoM magnitude estimate) actually
beat running either alone, at the SAME restart count -- verifying a
hypothesis raised in review rather than asserting it?

Mechanistic argument being tested: init_delta_matrix_mom_shrunk's magnitude
is DETERMINISTIC given the data -- every restart in a fit's batch gets the
exact same shrunk magnitude regardless of lambdaIndex (only the sign
varies). So if that one point estimate is off, no restart in the whole
batch (even at production's min(4**K,100) restart budget) can compensate
via magnitude -- best-of-LL can only rescue you through the sign dimension.
init_delta_matrix_mom_shrunk_cycling instead uses the shrunk estimate as a
CENTER and cycles a multiplier (0.5x/1x/2x) around it per restart_idx --
still zero added restart cost (same index already used for sign decoding)
-- specifically to test whether that hedge helps.

Compares, at n_restarts=4 (matching sim_delta_init_restart_diversity.py's
setup for direct comparability), across the same 4 skew regimes:
  (a) shrunk-only   -- init_delta_matrix_mom_shrunk, James-Stein c=0.25
                        (Item 2's own best single-point compromise), same
                        magnitude on every restart (only sign cycles)
  (b) cycling-only   -- init_delta_matrix_cycling_magnitude, fixed tiers
                        (0.1, 0.5, 1.0) (Item 1 as originally tested)
  (c) combined       -- init_delta_matrix_mom_shrunk_cycling: c=0.25 shrunk
                        center x (0.5, 1.0, 2.0) multiplier cycling
  (d) bimodal        -- init_delta_matrix_mom_shrunk_bimodal: even restarts
                        shrink hard (c=25, "probably no skew"), odd restarts
                        trust the data (c=0, "real skew"), instead of one
                        formula averaging across both hypotheses

Usage:
    python tests/cfusn_simulations/sim_delta_init_combined_1_2.py
    python tests/cfusn_simulations/sim_delta_init_combined_1_2.py --n-seeds 10
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
    sample_cfusn, init_delta_matrix_mom_shrunk, init_delta_matrix_cycling_magnitude,
    init_delta_matrix_mom_shrunk_cycling, init_delta_matrix_mom_shrunk_bimodal,
    _shrinkage_james_stein,
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
FALLBACK_SCALE = 0.1
JS_C = 0.25  # Item 2's best single-point compromise from sim_delta_init_mom_shrinkage.py
FIXED_TIERS = (0.1, 0.5, 1.0)
MULTIPLIER_TIERS = (0.5, 1.0, 2.0)
NULL_C = 25.0
TRUST_C = 0.0
N_OBS = 5000
N_RESTARTS = 4
VARIANTS = ("shrunk_only", "cycling_only", "combined", "bimodal")


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _run_one(magnitude, variant, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    state = {"restart_idx": 0}

    if variant == "shrunk_only":
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_mom_shrunk(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                                rng=rng, fallback_scale=FALLBACK_SCALE,
                                                shrinkage_fn=_shrinkage_james_stein, c=JS_C)
    elif variant == "cycling_only":
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_cycling_magnitude(
                cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
                restart_idx=state["restart_idx"], magnitude_tiers=FIXED_TIERS,
            )
    elif variant == "combined":
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_mom_shrunk_cycling(
                cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
                fallback_scale=FALLBACK_SCALE, shrinkage_fn=_shrinkage_james_stein, c=JS_C,
                restart_idx=state["restart_idx"], multiplier_tiers=MULTIPLIER_TIERS,
            )
    else:  # bimodal
        def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
            return init_delta_matrix_mom_shrunk_bimodal(
                cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng,
                fallback_scale=FALLBACK_SCALE, shrinkage_fn=_shrinkage_james_stein,
                restart_idx=state["restart_idx"], null_c=NULL_C, trust_c=TRUST_C,
            )

    orig = INIT._init_delta_matrix
    INIT._init_delta_matrix = _init
    try:
        best = None
        for i in range(n_restarts):
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
        for variant in VARIANTS:
            for seed in range(n_seeds):
                out = _run_one(magnitude, variant, seed, n_restarts)
                if out is None:
                    continue
                rows.append(dict(regime=regime, variant=variant, n_restarts=n_restarts,
                                 seed=seed, **out))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  ITEM 2 ALONE (shrunk, c={JS_C}) vs. ITEM 1 ALONE (fixed-tier cycling) vs. "
          f"COMBINED (multiplier-dilution) vs. BIMODAL (explicit null/trust restarts)")
    print(f"{'═' * 100}")
    n_restarts = rows[0]["n_restarts"] if rows else "?"
    print(f"\n  n_restarts={n_restarts}")
    print(f"  {'regime':>8}  {'true_norm':>10}  {'variant':>13}  {'mean fit_norm':>14}  "
          f"{'recovered%':>11}  {'mean final_ll':>14}  {'n':>4}")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["regime"], r["variant"]), []).append(r)
    for regime in SKEW_REGIMES:
        for variant in VARIANTS:
            group = by_key.get((regime, variant), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            lls = np.array([g["final_ll"] for g in group])
            pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
            print(f"  {regime:>8}  {true_norm:>10.3f}  {variant:>13}  "
                  f"{fit_norms.mean():>14.3f}  {pct:>10.0f}%  {lls.mean():>14.5f}  {len(group):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_combined_1_2_{ts}.csv"
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
