#!/usr/bin/env python3
"""Does the partial-distance k-means fix actually help recover a
KawOligo-like dimension specifically -- not just overall recovery, but the
one sparse dimension's own mu/Delta-row/Gamma-diagonal -- across TP53's real
coverage range (KawOligo: ~1-12% observed, i.e. ~88-99% missing)?

Design, shaped like a real TP53-scale problem rather than an abstract 2D toy:
p=6 dimensions. Dims 0-3 are well-observed, strongly-separating "core assay"
dimensions (like WAF1nWT/MDM2nWT/etc.) -- clustering succeeds fine on these
alone, so this isn't testing whether the fit works at all. Dim 4 is a
KawOligo-like dimension: it carries real but modest secondary separating
signal, and gets swept through TP53's actual missingness range. Dim 5 is a
Synonymous-like fully-observed dimension for context/regularization, kept
constant.

Reports two things per missingness fraction, for legacy vs. current
kmeans_init_mv:
  (1) overall mean_omega_error (whole-mixture recovery, as in the other
      sweep scripts)
  (2) error ISOLATED to dim 4 alone: |mu_true[4]-mu_fit[4]|, the Delta row
      error at dim 4 (after resolving the whole matrix's column ambiguity,
      via sim_utils.resolve_delta_ambiguity, then slicing to dim 4), and
      |Gamma_true[4,4]-Gamma_fit[4,4]| -- this is the number that actually
      answers "does the KawOligo-like dimension's own recovered shape get
      better," since a good overall omega_error doesn't guarantee any
      particular dimension's row is well recovered.

Usage:
    python tests/cfusn_simulations/sim_kawoligo_like_recovery.py
    python tests/cfusn_simulations/sim_kawoligo_like_recovery.py --n-seeds 10
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

import src.assay_calibration.fit_utils.cfusn.fit as CFIT
from src.assay_calibration.fit_utils.fit import tryToFit
from tests.cfusn_simulations.sim_utils import (
    sample_cfusn_mixture, inject_missingness, score_recovery, resolve_delta_ambiguity,
    match_components, kmeans_init_mv_legacy,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

KAWOLIGO_DIM = 4
TRUE_PARAMS = [
    (np.array([-1.5, -1.2, 1.0, -0.8, -0.4, 0.2]),
     np.array([[0.6, 0.2], [0.4, -0.15], [-0.3, 0.1], [0.2, 0.15], [0.15, -0.1], [0.1, 0.05]]),
     0.3 * np.eye(6)),
    (np.array([1.5, 1.2, -1.0, 0.8, 0.4, -0.2]),
     np.array([[-0.6, -0.2], [-0.4, 0.15], [0.3, -0.1], [-0.2, -0.15], [-0.15, 0.1], [-0.1, -0.05]]),
     0.3 * np.eye(6)),
]
N_SAMPLES = 3
COMP_PROBS = np.array([[0.85, 0.15], [0.15, 0.85], [0.5, 0.5]])
# TP53's real KawOligo coverage was ~1-12% observed -> ~88-99% missing.
DEFAULT_MISSING_GRID = [0.0, 0.5, 0.7, 0.85, 0.90, 0.95, 0.97, 0.99]


def _dim_recovery(true_params, fit_params):
    """Isolate recovery error to KAWOLIGO_DIM specifically, using the same
    Hungarian component matching as score_recovery, then resolving the
    matched pair's full Delta column ambiguity before slicing to the target
    dimension (column alignment must be resolved on the FULL matrix -- doing
    it on a single row would be underdetermined)."""
    row_ind, col_ind, _ = match_components(true_params, fit_params)
    dim_errors = []
    for ti, fi in zip(row_ind, col_ind):
        mu_t, Delta_t, Gamma_t = true_params[ti]
        mu_f, Delta_f, Gamma_f = fit_params[fi]
        Delta_f_aligned, _, _, _ = resolve_delta_ambiguity(Delta_t, Delta_f)
        dim_errors.append(dict(
            mu_error=float(abs(mu_t[KAWOLIGO_DIM] - mu_f[KAWOLIGO_DIM])),
            delta_row_error=float(np.linalg.norm(
                np.asarray(Delta_t)[KAWOLIGO_DIM] - Delta_f_aligned[KAWOLIGO_DIM]
            )),
            gamma_diag_error=float(abs(
                np.asarray(Gamma_t)[KAWOLIGO_DIM, KAWOLIGO_DIM]
                - np.asarray(Gamma_f)[KAWOLIGO_DIM, KAWOLIGO_DIM]
            )),
        ))
    return {k: float(np.mean([d[k] for d in dim_errors])) for k in dim_errors[0]} if dim_errors else None


def _run_one(missing_frac, seed, use_legacy):
    rng = np.random.RandomState(seed)
    sample_sizes = np.full(N_SAMPLES, 300)
    X, sa, _ = sample_cfusn_mixture(TRUE_PARAMS, COMP_PROBS, sample_sizes, rng)
    frac_per_dim = np.zeros(X.shape[1])
    frac_per_dim[KAWOLIGO_DIM] = missing_frac
    X_missing = inject_missingness(X, frac_per_dim, rng) if missing_frac > 0 else X

    if use_legacy:
        orig = CFIT.kmeans_init_mv
        CFIT.kmeans_init_mv = kmeans_init_mv_legacy
    try:
        result = tryToFit(
            X_missing, sa, num_components=2, constrained=False,
            init_method="kmeans", init_constraint_adjustment="scale",
            multivariate=True, latent_q=2, check_monotonic=False,
            num_fits=1, fit_seed=int(seed), verbose=False, verbose_init=False,
        )
    finally:
        if use_legacy:
            CFIT.kmeans_init_mv = orig

    params = result.get("component_params", [])
    if not params or any(len(p) == 0 for p in params):
        return None

    overall = score_recovery(TRUE_PARAMS, params)["mean_omega_error"]
    dim_specific = _dim_recovery(TRUE_PARAMS, params)
    if dim_specific is None:
        return None
    return dict(overall_omega_error=overall, **dim_specific)


def run_sweep(missing_frac_grid, n_seeds):
    rows = []
    for missing_frac in missing_frac_grid:
        for seed in range(n_seeds):
            for use_legacy in (True, False):
                res = _run_one(missing_frac, seed, use_legacy)
                if res is None:
                    continue
                rows.append(dict(
                    missing_frac=missing_frac, seed=seed,
                    variant="legacy" if use_legacy else "current",
                    **res,
                ))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  KAWOLIGO-LIKE DIMENSION (dim {KAWOLIGO_DIM}) RECOVERY: legacy vs. current kmeans init")
    print(f"{'═' * 100}")
    print(f"  {'missing_frac':>12}  {'overall (L/C)':>16}  {'dim mu_err (L/C)':>18}  "
          f"{'dim delta_err (L/C)':>21}  {'dim gamma_err (L/C)':>21}  {'n':>4}")

    by_frac = {}
    for r in rows:
        by_frac.setdefault(r["missing_frac"], {"legacy": [], "current": []})
        by_frac[r["missing_frac"]][r["variant"]].append(r)

    for frac in sorted(by_frac):
        L = by_frac[frac]["legacy"]
        C = by_frac[frac]["current"]

        def _m(group, key):
            return np.mean([g[key] for g in group]) if group else float("nan")

        print(f"  {frac:>12.2f}  "
              f"{_m(L,'overall_omega_error'):>7.3f}/{_m(C,'overall_omega_error'):>7.3f}  "
              f"{_m(L,'mu_error'):>9.3f}/{_m(C,'mu_error'):>7.3f}  "
              f"{_m(L,'delta_row_error'):>10.3f}/{_m(C,'delta_row_error'):>9.3f}  "
              f"{_m(L,'gamma_diag_error'):>10.3f}/{_m(C,'gamma_diag_error'):>9.3f}  "
              f"{len(L):>4}")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_kawoligo_like_recovery_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--missing-frac-grid", type=float, nargs="+",
                        default=DEFAULT_MISSING_GRID)
    args = parser.parse_args()

    print(f"True params: {len(TRUE_PARAMS)} components, {TRUE_PARAMS[0][0].shape[0]}D. "
          f"Dim {KAWOLIGO_DIM} is the KawOligo-like dimension (real coverage ~1-12%, "
          f"i.e. ~88-99% missing) subject to the sweep; dims 0-3 are always fully observed.")
    print(f"n_seeds={args.n_seeds}")

    rows = run_sweep(args.missing_frac_grid, args.n_seeds)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
