#!/usr/bin/env python3
"""Does EM reach the TRUE skew sign, and does the init sign pattern matter?

Supersedes the change-vs-init metric used by sim_sign_flip_cfusn.py and
verify_skew_sign_change_multivariate.py. Whether the fitted Delta differs from
where it started says only that the optimizer moved; it says nothing about
whether it moved somewhere correct. A fit that flips away from a correct init is
counted as a success by that metric and is in fact a failure. What decides
whether sign enumeration is needed is convergence to truth:

    for each initial sign pattern, does the fit end at the true sign?

Only moderate and large skew are simulated. At zero or small skew the sign is
not identifiable from the data at all, every arm is dominated by a symmetric
solution, and the question does not arise.

Reported per latent_q, for each lambdaIndex pattern: the signs the fit ends at
and whether they match truth, after resolving the column PERMUTATION only (see
_resolve_permutation_only -- the per-column sign is identifiable for CFUSN and
must not be resolved away). The summary line -- what fraction of initial
patterns reach truth -- is the number that decides the budget:

    high  -> EM self-corrects, enumeration is redundant, spend restarts elsewhere
    low   -> the init sign is sticky and enumeration is load-bearing

Usage:
    python tests/cfusn_simulations/sim_sign_recovery_to_truth.py
    python tests/cfusn_simulations/sim_sign_recovery_to_truth.py \
        --latent-q 2 --n-components 2 --regimes large --n-seeds 5
"""
import argparse
import csv
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import tryToFit
from src.assay_calibration.fit_utils.cfusn.density_utils import _ensure_matrix_delta
from tests.cfusn_simulations.sim_utils import (
    sample_cfusn_mixture, match_components,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Only regimes where the sign is identifiable. "moderate"/"large" are per-entry
# magnitudes; ||Delta|| per component follows from the dimension.
REGIME_SCALE = {"moderate": 0.7, "large": 1.4}


def build_truth(K, p, q, scale, seed=0):
    """Components with known, mixed sign patterns and LINEARLY INDEPENDENT
    skewness columns.

    An earlier version set D[:, j] = s * scale, making every column a multiple
    of the all-ones vector: rank(Delta) = 1 with anti-parallel columns, i.e. a
    q=1 model in q=2 clothing. With Delta rank-deficient the per-column sign is
    not well defined and any sign-recovery metric computed on it is meaningless.
    Columns are now drawn independently and orthonormalised before scaling, so
    each carries its own identifiable direction and sign.
    """
    rng = np.random.RandomState(20260924 + seed)
    params, signs = [], []
    for c in range(K):
        mu = np.linspace(-2.0, 2.0, K)[c] * np.ones(p)
        B = rng.randn(p, q)
        B, _ = np.linalg.qr(B)                       # orthonormal columns
        D = np.zeros((p, q))
        sg = []
        for j in range(q):
            s = 1.0 if ((c + j) % 2 == 0) else -1.0
            D[:, j] = s * scale * B[:, j] * np.sqrt(p)
            sg.append(int(s))
        params.append((mu, D if q > 1 else D[:, 0], np.eye(p) * 0.6))
        signs.append(tuple(sg))
    return params, signs


def _resolve_permutation_only(Dt, Df):
    """Best column PERMUTATION of Df against Dt -- no sign flips.

    sim_utils.resolve_delta_ambiguity searches permutations AND per-column sign
    flips, which is wrong here in two ways. First, it is wrong on the facts: for
    CFUSN the latent t is truncated to the positive orthant, so flipping a
    column's sign yields a genuinely different distribution (measured: the
    log-density moves by 10-13 nats, whereas swapping columns moves it by
    9.4e-14). The sign is identifiable; only the permutation is not. Second,
    resolving the sign would destroy the very quantity this script measures, by
    re-aligning a sign-wrong fit into agreement with truth.
    """
    q = Dt.shape[1]
    best = None
    for perm in itertools.permutations(range(q)):
        cand = Df[:, perm]
        err = float(np.linalg.norm(np.abs(cand) - np.abs(Dt)))
        if best is None or err < best[0]:
            best = (err, cand)
    return best[1]


def final_signs(params, truth, q):
    """Per-component sign of the fitted Delta columns, relative to truth."""
    row_ind, col_ind, _ = match_components(truth, params)
    out = {}
    for ti, fi in zip(row_ind, col_ind):
        Dt = _ensure_matrix_delta(truth[ti][1])
        Df = _ensure_matrix_delta(params[fi][1])
        Dr = _resolve_permutation_only(Dt, Df)
        sg = []
        for j in range(q):
            col_t, col_f = Dt[:, j], Dr[:, j]
            dot = float(np.dot(col_t, col_f))
            nt = float(np.linalg.norm(col_t))
            sg.append(0 if nt < 1e-9 or abs(dot) < 1e-9
                      else (1 if dot > 0 else -1))
        out[ti] = tuple(sg)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent-q", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--n-components", type=int, default=2)
    ap.add_argument("--n-dims", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=400)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--max-em-iters", type=int, default=1500)
    ap.add_argument("--regimes", nargs="+", default=["moderate", "large"])
    args = ap.parse_args()

    K, p = args.n_components, args.n_dims
    rows = []

    for q in args.latent_q:
        n_patterns = min((2 ** q) ** K, 64)
        for regime in args.regimes:
            scale = REGIME_SCALE[regime]
            truth, true_signs = build_truth(K, p, q, scale)
            print(f"\n{'=' * 66}\n  latent_q={q}  K={K}  p={p}  regime={regime} "
                  f"(per-entry {scale})  patterns={n_patterns}\n{'=' * 66}",
                  flush=True)
            print(f"  true signs per component: {true_signs}", flush=True)
            print(f"\n  {'lambdaIdx':>9} {'seed':>5}  {'final signs':>24}  "
                  f"{'matches truth':>14}", flush=True)

            reached = 0
            total = 0
            for li in range(n_patterns):
                for seed in range(args.n_seeds):
                    rng = np.random.RandomState(500 + seed)
                    W = np.full((2, K), 1.0 / K)
                    X, SA, _ = sample_cfusn_mixture(truth, W,
                                                    [args.n_obs] * 2, rng)
                    res = tryToFit(
                        X, SA, K, False, "kmeans", "scale", multivariate=True,
                        latent_q=q, lambdaIndex=li, num_fits=1,
                        fit_seed=int(seed * 977 + li),
                        max_em_iters=args.max_em_iters, check_monotonic=False,
                        verbose=False, verbose_init=False, raise_on_error=False,
                    )
                    params = res.get("component_params")
                    if not params or any(len(x) == 0 for x in params):
                        continue
                    fs = final_signs(params, truth, q)
                    got = tuple(fs.get(c, tuple([0] * q)) for c in range(K))
                    ok = all(got[c] == true_signs[c] for c in range(K))
                    reached += ok
                    total += 1
                    rows.append(dict(latent_q=q, regime=regime, lambdaIndex=li,
                                     seed=seed, final_signs=str(got),
                                     true_signs=str(true_signs),
                                     matches_truth=int(ok)))
                    if seed == 0:
                        print(f"  {li:9d} {seed:5d}  {str(got):>24}  "
                              f"{'yes' if ok else 'no':>14}", flush=True)
            pct = 100.0 * reached / max(total, 1)
            print(f"\n  REACHED TRUTH: {reached}/{total} ({pct:.0f}%) "
                  f"of initial sign patterns", flush=True)
            print(f"  -> {'EM self-corrects; enumeration adds little' if pct >= 75 else 'init sign is sticky; enumeration is load-bearing'}",
                  flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"sim_sign_recovery_to_truth_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
