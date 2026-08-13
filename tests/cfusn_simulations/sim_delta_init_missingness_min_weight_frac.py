#!/usr/bin/env python3
"""Does raising min_weight_frac (how much of an eigenvector's weight must
be observed for a row to be kept and rescaled by _partial_projection)
reduce the excess skewness-estimator noise diagnosed in the previous
script -- partial-projection's skewness estimate had ~7x higher standard
deviation than a clean full-data projection using the SAME eigenvector,
even though its mean was roughly unbiased, at min_weight_frac=0.5? The
hypothesis: rows kept from as little as 50% of an eigenvector's relevant
weight get rescaled up by sqrt(2)x, and this rescaling disproportionately
amplifies a THIRD-MOMENT (skewness) statistic's noise far more than a
simple effective-sample-size argument captures (confirmed: an
effective-sample-size correction to the z-score's SE formula made
essentially no difference). Raising the threshold trades activation rate
(fewer rows qualify) for lower per-row reconstruction noise -- this script
maps out that tradeoff directly and cheaply (no EM needed).

Same TP53-shaped block missingness, isolated single-active-skew-column
setup (p=16, only column 1 -- dims 0,1 -- carries real skew) as
sim_delta_init_missingness_isolated_c_sweep.py, for continuity. Reports,
per min_weight_frac value and skew regime: activation rate, mean rows
kept, and the skewness estimate's mean/std across many seeds (compared to
the clean full-data-same-eigenvector reference, computed once per regime).

Usage:
    python tests/cfusn_simulations/sim_delta_init_missingness_min_weight_frac.py
    python tests/cfusn_simulations/sim_delta_init_missingness_min_weight_frac.py --n-seeds 50
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

from tests.cfusn_simulations.sim_utils import (
    sample_cfusn, inject_block_missingness, _partial_projection, sps_skew,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

P = 16
MU_TRUE = np.zeros(P)
GAMMA_TRUE = 0.5 * np.eye(P)
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
MIN_WEIGHT_FRAC_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
N_OBS = 5000


def _pairwise_cov(X):
    p = X.shape[1]
    cov = np.zeros((p, p))
    for d1 in range(p):
        for d2 in range(d1, p):
            both = ~np.isnan(X[:, d1]) & ~np.isnan(X[:, d2])
            if both.sum() >= 2:
                cov[d1, d2] = np.cov(X[both, d1], X[both, d2])[0, 1]
            cov[d2, d1] = cov[d1, d2]
    cov += 1e-6 * np.eye(p)
    ev = np.linalg.eigvalsh(cov)
    if ev.min() < 1e-8:
        cov += (1e-8 - ev.min()) * np.eye(p)
    return cov


def _delta_true(magnitude):
    return DIRECTION * magnitude


def run_sweep(n_seeds):
    rows = []
    for regime, magnitude in SKEW_REGIMES.items():
        Delta_true = _delta_true(magnitude)
        clean_skews = []
        per_mwf_data = {mwf: {"activated": [], "n_kept": [], "skews": []} for mwf in MIN_WEIGHT_FRAC_VALUES}

        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            X_full, _ = sample_cfusn(MU_TRUE, Delta_true, GAMMA_TRUE, N_OBS, rng)
            X = inject_block_missingness(X_full, BLOCKS, BLOCK_FRAC_MISSING, rng)
            cov = _pairwise_cov(X)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            evec = eigvecs[:, order[0]]

            clean_skews.append(sps_skew(X_full @ evec))

            for mwf in MIN_WEIGHT_FRAC_VALUES:
                proj = _partial_projection(X, evec, min_weight_frac=mwf)
                activated = len(proj) >= 8
                per_mwf_data[mwf]["activated"].append(activated)
                per_mwf_data[mwf]["n_kept"].append(len(proj))
                if activated:
                    per_mwf_data[mwf]["skews"].append(sps_skew(proj))

        for mwf in MIN_WEIGHT_FRAC_VALUES:
            d = per_mwf_data[mwf]
            skews = np.array(d["skews"])
            rows.append(dict(
                regime=regime, min_weight_frac=mwf,
                activation_pct=100 * np.mean(d["activated"]),
                mean_n_kept=np.mean(d["n_kept"]),
                skew_mean=float(np.mean(skews)) if len(skews) else float("nan"),
                skew_std=float(np.std(skews)) if len(skews) else float("nan"),
                clean_skew_mean=float(np.mean(clean_skews)),
                clean_skew_std=float(np.std(clean_skews)),
            ))
    return rows


def _report(rows):
    print(f"\n{'═' * 100}")
    print(f"  min_weight_frac SWEEP: activation rate vs. skewness-estimator noise")
    print(f"  (TP53-shaped block missingness, isolated single-column truth, p={P})")
    print(f"{'═' * 100}")
    by_regime = {}
    for r in rows:
        by_regime.setdefault(r["regime"], []).append(r)

    for regime in SKEW_REGIMES:
        group = by_regime.get(regime, [])
        if not group:
            continue
        clean_mean = group[0]["clean_skew_mean"]
        clean_std = group[0]["clean_skew_std"]
        print(f"\n  --- regime={regime} (clean full-data skew: mean={clean_mean:.4f}, std={clean_std:.4f}) ---")
        print(f"  {'min_weight_frac':>15}  {'activation%':>11}  {'mean n_kept':>11}  "
              f"{'skew mean':>10}  {'skew std':>9}  {'std ratio vs clean':>18}")
        for r in group:
            std_ratio = r["skew_std"] / clean_std if clean_std > 1e-9 else float("nan")
            print(f"  {r['min_weight_frac']:>15.2f}  {r['activation_pct']:>10.0f}%  "
                  f"{r['mean_n_kept']:>11.0f}  {r['skew_mean']:>10.4f}  {r['skew_std']:>9.4f}  "
                  f"{std_ratio:>17.2f}x")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_missingness_min_weight_frac_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=30)
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds)
    _report(rows)
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
