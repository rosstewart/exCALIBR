#!/usr/bin/env python3
"""When can the data-driven skewness SIGN that _init_delta_matrix already
uses (sign(sps.skew(projection)), hedged today by exhaustive lambdaIndex
sign enumeration -- 4**K restarts for q=2) be trusted at a given per-cluster
sample size N, and at what N does it become reliable enough that enumerating
every sign combination stops being necessary?

Pure sampling + the existing sign-decision logic (no EM needed -- this
question is about the SIGN estimator's own sampling distribution, not about
optimization), so cheap enough to run many seeds per cell. Sweeps N from
values bracketing real KawOligo-like sparse-dimension cluster sizes (~10-40
observed points) up through well-powered clusters, denser at the small-N end
since that's where the accuracy curve is steepest and most decision-relevant.

Also answers, as a confirmatory addendum (no sampling needed -- the z-score
formula depends only on n given a fixed sample skewness), whether the
existing z-gate (_skewness_z_score) already naturally down-weights a sparse
dimension's skewness confidence relative to a well-observed one -- i.e.
whether "equal-weighting dimensions" is already handled here, separately
from the unrelated cross-dimension *scale* standardization already fixed in
Fit.generate_fit_jobs.

Usage:
    python tests/cfusn_simulations/sim_delta_init_sign_reliability.py
    python tests/cfusn_simulations/sim_delta_init_sign_reliability.py --n-seeds 500
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

from tests.cfusn_simulations.sim_utils import sample_cfusn, sps_skew, _skewness_z_score

RESULTS_DIR = Path(__file__).resolve().parent / "results"

SKEW_REGIMES = {
    "zero": 0.0,
    "small": 0.15,
    "medium": 0.35,
    "large": 0.7,
}
N_GRID = [15, 25, 40, 75, 150, 300, 1000, 5000]
TARGET_ACCURACY = 0.95
NOMINAL_SKEW_FOR_Z_ADDENDUM = 0.3


def _run_one(magnitude, n, seed):
    rng = np.random.RandomState(seed)
    X, _ = sample_cfusn(np.zeros(1), np.array([[magnitude]]), np.eye(1), n, rng)
    sk = sps_skew(X[:, 0])
    if np.isnan(sk) or abs(sk) < 1e-9:
        return None
    return int(np.sign(sk))


def run_sweep(n_seeds):
    rows = []
    for regime, magnitude in SKEW_REGIMES.items():
        true_sign = 1 if magnitude > 0 else 0
        for n in N_GRID:
            signs = []
            for seed in range(n_seeds):
                s = _run_one(magnitude, n, seed)
                if s is None:
                    continue
                signs.append(s)
            signs = np.array(signs)
            if regime == "zero":
                # No true sign to be "correct" against -- report instability
                # (fraction positive; expect ~50% if unbiased).
                frac_pos = float((signs > 0).mean()) if len(signs) else float("nan")
                rows.append(dict(regime=regime, magnitude=magnitude, n=n,
                                 n_valid=len(signs), accuracy=float("nan"),
                                 frac_positive=frac_pos))
            else:
                acc = float((signs == true_sign).mean()) if len(signs) else float("nan")
                rows.append(dict(regime=regime, magnitude=magnitude, n=n,
                                 n_valid=len(signs), accuracy=acc, frac_positive=float("nan")))
    return rows


def _report_sign_accuracy(rows):
    print(f"\n{'═' * 100}")
    print(f"  DATA-DRIVEN SKEWNESS SIGN: accuracy vs. per-cluster N")
    print(f"{'═' * 100}")
    by_key = {}
    for r in rows:
        by_key.setdefault(r["regime"], {})[r["n"]] = r

    for regime, magnitude in SKEW_REGIMES.items():
        print(f"\n  --- regime={regime} (magnitude={magnitude:.2f}) ---")
        if regime == "zero":
            print(f"  {'N':>6}  {'frac_positive':>13}  {'n_valid':>8}   "
                  f"(expect ~0.50 -- no systematic sign bias)")
            for n in N_GRID:
                r = by_key[regime][n]
                print(f"  {n:>6}  {r['frac_positive']:>13.3f}  {r['n_valid']:>8}")
        else:
            print(f"  {'N':>6}  {'P(correct sign)':>16}  {'n_valid':>8}")
            first_reliable = None
            for n in N_GRID:
                r = by_key[regime][n]
                print(f"  {n:>6}  {r['accuracy']:>16.3f}  {r['n_valid']:>8}")
                if first_reliable is None and r["accuracy"] >= TARGET_ACCURACY:
                    first_reliable = n
            if first_reliable is not None:
                print(f"  -> smallest N reaching >={TARGET_ACCURACY:.0%} accuracy: {first_reliable}")
            else:
                print(f"  -> {TARGET_ACCURACY:.0%} accuracy NOT reached within tested N range")


def _report_z_addendum():
    print(f"\n{'═' * 100}")
    print(f"  ADDENDUM: does the z-gate already down-weight sparse dimensions?")
    print(f"  (fixed nominal sample skewness = {NOMINAL_SKEW_FOR_Z_ADDENDUM}, "
          f"z = |skew| / SE(n) -- analytic, no sampling)")
    print(f"{'═' * 100}")
    print(f"  {'N':>6}  {'SE(n)':>10}  {'implied z':>10}")
    for n in N_GRID:
        se = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        z = NOMINAL_SKEW_FOR_Z_ADDENDUM / se
        print(f"  {n:>6}  {se:>10.4f}  {z:>10.3f}")
    print(f"\n  Interpretation: the SAME nominal skewness produces a much smaller\n"
          f"  z at small N than at large N -- the z-gate already treats a sparse\n"
          f"  dimension's skewness estimate as less trustworthy without any\n"
          f"  separate equal-weighting mechanism needed.")


def _write_csv(rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"sim_delta_init_sign_reliability_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=200)
    args = parser.parse_args()

    rows = run_sweep(args.n_seeds)
    _report_sign_accuracy(rows)
    _report_z_addendum()
    _write_csv(rows)
    print()


if __name__ == "__main__":
    main()
