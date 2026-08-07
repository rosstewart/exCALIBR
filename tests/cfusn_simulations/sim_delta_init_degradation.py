#!/usr/bin/env python3
"""Quantify the Delta-init collapse found by instrumenting a real TP53 fit:
initializations.py's _init_delta_matrix used to have a compounding
"shrink Delta until Gamma=cov-Delta@Delta.T is PD" retry loop
(`shrink *= 0.5` every failed retry -> cumulative 0.5*0.25*0.125*...,
annihilating Delta to ~1e-60 within a handful of retries). The loop has
since been fixed here (fixed per-retry decay, Delta *= 0.9, no compounding)
-- this script measures BOTH the old (buggy) and new (fixed) behavior so
the improvement is a concrete before/after number, not a hand-wavy claim.

Two experiments:
  (1) Direct unit-level reproduction: call _init_delta_matrix in isolation
      (no full EM) across a swept grid of synthetic covariance conditioning,
      recording ||Delta||_F after init.
  (2) End-to-end failure rate: generate real CFUSN mixture data, inject
      missingness at a swept fraction (mirroring TP53's real per-dimension
      coverage extremes), run tryToFit(latent_q=2), and record what fraction
      of components initialize with ||Delta||_F below a "collapsed" threshold.

Usage:
    python tests/cfusn_simulations/sim_delta_init_degradation.py
    python tests/cfusn_simulations/sim_delta_init_degradation.py --n-seeds 10
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
from tests.cfusn_simulations.sim_utils import sample_cfusn_mixture, inject_missingness

RESULTS_DIR = Path(__file__).resolve().parent / "results"

COLLAPSE_THRESHOLD = 1e-6


# ── the old, buggy shrink loop (kept here only for before/after comparison) ──

def _init_delta_matrix_buggy(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
    """Exact copy of the pre-fix _init_delta_matrix, for comparison only.

    Do not "fix" this function -- it exists specifically to reproduce the
    old compounding-shrink bug so this script can report a before/after
    failure rate. The real (fixed) implementation lives in
    src/assay_calibration/fit_utils/cfusn/initializations.py.
    """
    import scipy.stats as sps
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        scale = 0.1 * np.sqrt(eigvals[idx])
        evec = eigvecs[:, idx]
        skew_sign = 1
        if Xc is not None:
            complete_rows = ~np.isnan(Xc).any(axis=1)
            Xc_comp = Xc[complete_rows]
            if len(Xc_comp) >= 8:
                sk = sps.skew(Xc_comp @ evec)
                if abs(sk) > 1e-6:
                    skew_sign = int(np.sign(sk))
        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        shrink = 0.5
        for _ in range(20):
            Delta *= shrink
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break
            shrink *= 0.5

    return Delta


# ── Experiment 1: direct unit-level reproduction ────────────────────────────

def _make_ill_conditioned_cov(p, min_eigval, rng):
    """cov = V @ diag(eigvals) @ V.T, one eigenvalue set to min_eigval, rest 1.0."""
    A = rng.standard_normal((p, p))
    V, _ = np.linalg.qr(A)
    eigvals = np.concatenate([[min_eigval], np.ones(p - 1)])
    return V @ np.diag(eigvals) @ V.T


def run_unit_level(p=4, q=2, n_seeds=20, min_eigval_grid=(1e-8, 1e-6, 1e-4, 1e-2)):
    rows = []
    for min_eigval in min_eigval_grid:
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            cov = _make_ill_conditioned_cov(p, min_eigval, rng)
            cluster_sign_pattern = rng.choice([-1, 1], size=q)

            Delta_buggy = _init_delta_matrix_buggy(
                cov, p, q, Xc=None, cluster_sign_pattern=cluster_sign_pattern,
                rng=np.random.RandomState(seed),
            )
            Delta_fixed = INIT._init_delta_matrix(
                cov, p, q, Xc=None, cluster_sign_pattern=cluster_sign_pattern,
                rng=np.random.RandomState(seed),
            )
            rows.append(dict(
                min_eigval=min_eigval, seed=seed,
                delta_norm_buggy=float(np.linalg.norm(Delta_buggy)),
                delta_norm_fixed=float(np.linalg.norm(Delta_fixed)),
            ))
    return rows


def _report_unit_level(rows):
    print(f"\n{'═' * 78}")
    print("  EXPERIMENT 1: direct _init_delta_matrix reproduction")
    print(f"{'═' * 78}")
    print(f"  {'min_eigval':>12}  {'buggy: collapsed%':>20}  {'buggy: median norm':>20}"
          f"  {'fixed: median norm':>20}")
    by_eig = {}
    for r in rows:
        by_eig.setdefault(r["min_eigval"], []).append(r)
    for min_eigval in sorted(by_eig):
        group = by_eig[min_eigval]
        buggy_norms = np.array([g["delta_norm_buggy"] for g in group])
        fixed_norms = np.array([g["delta_norm_fixed"] for g in group])
        collapsed_pct = 100 * (buggy_norms < COLLAPSE_THRESHOLD).mean()
        print(f"  {min_eigval:>12.0e}  {collapsed_pct:>19.1f}%  "
              f"{np.median(buggy_norms):>20.3e}  {np.median(fixed_norms):>20.3e}")


# ── Experiment 2: end-to-end failure rate via tryToFit ──────────────────────

TRUE_PARAMS_E2E = [
    (np.array([-1.5, -1.2, 0.0, 0.0]),
     np.array([[0.6, 0.2], [0.4, -0.15], [0.3, 0.1], [-0.2, 0.05]]), 0.3 * np.eye(4)),
    (np.array([1.5, 1.2, 0.0, 0.0]),
     np.array([[-0.6, -0.2], [-0.4, 0.15], [-0.3, -0.1], [0.2, -0.05]]), 0.3 * np.eye(4)),
]
N_SAMPLES_E2E = 3
COMP_PROBS_E2E = np.array([[0.85, 0.15], [0.15, 0.85], [0.5, 0.5]])


def _run_one_e2e_fit(missing_frac, seed, use_buggy_init):
    rng = np.random.RandomState(seed)
    X, sa, _ = sample_cfusn_mixture(
        TRUE_PARAMS_E2E, COMP_PROBS_E2E, np.full(N_SAMPLES_E2E, 300), rng,
    )
    p = X.shape[1]
    # Only the last 2 (of 4) dimensions get the swept missingness -- mirrors
    # TP53 having some well-covered and some sparse dimensions at once.
    frac_per_dim = np.array([0.0, 0.0, missing_frac, missing_frac])
    X_missing = inject_missingness(X, frac_per_dim, rng)

    if use_buggy_init:
        INIT._init_delta_matrix, _orig = _init_delta_matrix_buggy, INIT._init_delta_matrix
    try:
        result = tryToFit(
            X_missing, sa, num_components=2, constrained=False,
            init_method="kmeans", init_constraint_adjustment="scale",
            multivariate=True, latent_q=2, check_monotonic=False,
            num_fits=1, fit_seed=int(seed), verbose=False, verbose_init=False,
        )
    finally:
        if use_buggy_init:
            INIT._init_delta_matrix = _orig

    init_params = result.get("initial_params", [])
    if not init_params or any(len(p) == 0 for p in init_params):
        return None
    norms = [float(np.linalg.norm(np.asarray(p[1]))) for p in init_params]
    return norms


def run_end_to_end(missing_frac_grid, n_seeds):
    rows = []
    for missing_frac in missing_frac_grid:
        for seed in range(n_seeds):
            for use_buggy in (True, False):
                norms = _run_one_e2e_fit(missing_frac, seed, use_buggy)
                if norms is None:
                    continue
                for c, norm in enumerate(norms):
                    rows.append(dict(
                        missing_frac=missing_frac, seed=seed, component=c,
                        variant="buggy" if use_buggy else "fixed",
                        delta_norm=norm,
                    ))
    return rows


def _report_end_to_end(rows):
    print(f"\n{'═' * 78}")
    print("  EXPERIMENT 2: end-to-end init failure rate via tryToFit")
    print(f"{'═' * 78}")
    print(f"  {'missing_frac':>12}  {'buggy: collapsed%':>20}  {'fixed: collapsed%':>20}")

    by_frac = {}
    for r in rows:
        by_frac.setdefault(r["missing_frac"], {"buggy": [], "fixed": []})
        by_frac[r["missing_frac"]][r["variant"]].append(r["delta_norm"])

    for frac in sorted(by_frac):
        buggy = np.array(by_frac[frac]["buggy"])
        fixed = np.array(by_frac[frac]["fixed"])
        buggy_pct = 100 * (buggy < COLLAPSE_THRESHOLD).mean() if len(buggy) else float("nan")
        fixed_pct = 100 * (fixed < COLLAPSE_THRESHOLD).mean() if len(fixed) else float("nan")
        print(f"  {frac:>12.2f}  {buggy_pct:>19.1f}%  {fixed_pct:>19.1f}%")


def _write_csv(rows, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--missing-frac-grid", type=float, nargs="+",
                        default=[0.0, 0.5, 0.7, 0.85, 0.90, 0.95, 0.99])
    args = parser.parse_args()

    unit_rows = run_unit_level(n_seeds=args.n_seeds)
    _report_unit_level(unit_rows)
    _write_csv(unit_rows, "sim_delta_init_degradation_unit")

    e2e_rows = run_end_to_end(args.missing_frac_grid, n_seeds=max(3, args.n_seeds // 4))
    _report_end_to_end(e2e_rows)
    _write_csv(e2e_rows, "sim_delta_init_degradation_e2e")
    print()


if __name__ == "__main__":
    main()
