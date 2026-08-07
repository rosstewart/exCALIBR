#!/usr/bin/env python3
"""Does replacing the hard z_threshold gate (init_delta_matrix_mom_gated) with
a CONTINUOUS shrinkage of the method-of-moments magnitude toward the fixed
fallback -- James-Stein (max(0, 1-c/z**2)) or a sigmoid blend -- trace out a
strictly better zero-regime-false-positive vs. medium/large-regime-error
tradeoff than any single hard threshold could?

sim_delta_init_mom_gated.py's z_threshold sweep found a genuine, irreducible
tradeoff at a single N: no one hard threshold is good in every regime
(stricter gating cuts the zero-regime false positive but degrades medium-
regime recovery, e.g. from 6.6% error to 71.4% error at N=5000). A hard
0/1 cutoff forces a step function on an inherently continuous confidence
signal (z); this script tests whether smoothly blending trust in [0,1]
instead of snapping it to {0,1} lets partial-confidence regimes (small/
medium skew, moderate z) get partial credit instead of being forced into
"trust nothing" or "trust everything."

Per the "data-driven, not arbitrary" requirement, both shrinkage knobs are
swept (not hand-picked), using the SAME grid as the z_threshold sweep
(c = z_threshold**2, so knee-points align directly and are comparable).

Same 4 true-skew regimes (zero/small/medium/large) and N as the previous
scripts in this chain, for direct comparability.

Usage:
    python tests/cfusn_simulations/sim_delta_init_mom_shrinkage.py
    python tests/cfusn_simulations/sim_delta_init_mom_shrinkage.py --n-seeds 10
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
    sample_cfusn, init_delta_matrix_mom_shrunk,
    _shrinkage_james_stein, _shrinkage_sigmoid,
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
Z_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
JS_C_VALUES = [z ** 2 for z in Z_THRESHOLDS]
SIGMOID_Z0_VALUES = Z_THRESHOLDS
SIGMOID_K = 2.0
N_OBS = 5000
FALLBACK_SCALE = 0.1


def _delta_true(magnitude):
    return DIRECTION * magnitude


def _run_one(magnitude, form, knob, seed, n_restarts):
    Delta_true = _delta_true(magnitude)
    rng = np.random.RandomState(seed)
    X, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
    sa = np.ones((N_OBS, 1), dtype=bool)

    if form == "james_stein":
        shrinkage_fn = _shrinkage_james_stein
        shrinkage_kwargs = dict(c=knob)
    else:
        shrinkage_fn = _shrinkage_sigmoid
        shrinkage_kwargs = dict(z0=knob, k=SIGMOID_K)

    def _init(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
        return init_delta_matrix_mom_shrunk(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                            rng=rng, fallback_scale=FALLBACK_SCALE,
                                            shrinkage_fn=shrinkage_fn, **shrinkage_kwargs)

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
        for form, knobs in (("james_stein", JS_C_VALUES), ("sigmoid", SIGMOID_Z0_VALUES)):
            for knob in knobs:
                for seed in range(n_seeds):
                    out = _run_one(magnitude, form, knob, seed, n_restarts)
                    if out is None:
                        continue
                    rows.append(dict(regime=regime, form=form, knob=knob, seed=seed,
                                     n_restarts=n_restarts, **out))
    return rows


def _combined_scores(rows, form):
    """Per-knob combined score dict, mirroring sim_delta_init_mom_gated.py's
    summary table: zero-regime score = mean fit_norm itself (want small);
    other regimes = |recovered% - 100| (want small)."""
    by_key = {}
    for r in rows:
        if r["form"] != form:
            continue
        by_key.setdefault((r["regime"], r["knob"]), []).append(r)

    knobs = sorted(set(r["knob"] for r in rows if r["form"] == form))
    out = {}
    for knob in knobs:
        vals = {}
        for regime in SKEW_REGIMES:
            group = by_key.get((regime, knob), [])
            if not group:
                continue
            true_norm = group[0]["true_norm"]
            fit_norms = np.array([g["fit_norm"] for g in group])
            if true_norm > 1e-9:
                vals[regime] = abs(100 * fit_norms.mean() / true_norm - 100)
            else:
                vals[regime] = fit_norms.mean()
        out[knob] = vals
    return out


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  CONTINUOUS SHRINKAGE (James-Stein / sigmoid) vs. hard z-gate: knob sweep")
    print(f"{'═' * 100}")

    for form in ("james_stein", "sigmoid"):
        label = "James-Stein c" if form == "james_stein" else f"sigmoid z0 (k={SIGMOID_K})"
        print(f"\n  === {label} ===")
        by_key = {}
        for r in rows:
            if r["form"] != form:
                continue
            by_key.setdefault((r["regime"], r["knob"]), []).append(r)
        for regime in SKEW_REGIMES:
            print(f"\n  --- regime={regime} (true_norm={SKEW_REGIMES[regime]:.2f} scaled) ---")
            print(f"  {'knob':>8}  {'mean fit_norm':>14}  {'recovered%':>11}  "
                  f"{'mean final_ll':>14}  {'n':>4}")
            knobs = sorted(set(r["knob"] for r in rows if r["form"] == form))
            for knob in knobs:
                group = by_key.get((regime, knob), [])
                if not group:
                    continue
                true_norm = group[0]["true_norm"]
                fit_norms = np.array([g["fit_norm"] for g in group])
                lls = np.array([g["final_ll"] for g in group])
                pct = (100 * fit_norms.mean() / true_norm) if true_norm > 1e-9 else float("nan")
                print(f"  {knob:>8.2f}  {fit_norms.mean():>14.3f}  {pct:>10.0f}%  "
                      f"{lls.mean():>14.5f}  {len(group):>4}")

        print(f"\n  --- summary: combined score per knob ({label}) ---")
        print(f"  {'knob':>8}  {'zero fit_norm':>14}  {'small |err%|':>13}  "
              f"{'medium |err%|':>14}  {'large |err%|':>13}")
        scores = _combined_scores(rows, form)
        for knob in sorted(scores):
            v = scores[knob]
            print(f"  {knob:>8.2f}  {v.get('zero', float('nan')):>14.3f}  "
                  f"{v.get('small', float('nan')):>13.1f}  {v.get('medium', float('nan')):>14.1f}  "
                  f"{v.get('large', float('nan')):>13.1f}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_mom_shrinkage_{ts}.csv"
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
