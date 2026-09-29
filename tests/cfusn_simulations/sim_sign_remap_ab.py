#!/usr/bin/env python3
"""A/B: does re-mapping the sign enumeration degrade skew recovery?

The re-map made lambdaIndex 0 the all-(+1) "no flip" pattern instead of the
all-flip one, and lambdaIndex is now counted within each restart mode rather
than off the global fit_idx. Both are relabellings of the same pattern set, so
under FULL enumeration they cannot change anything; they can only matter when
the budget is truncated, which is exactly production's case (num_fits=3 leaves a
single q=2 restart).

The old mapping is reproducible under current code without checking anything
out: old index L yields the same sign pattern as new index
(2**(K*q) - 1) - L, since the two differ by complementing every bit. So both
arms run against one implementation.

Arms, at moderate and large skew (the only regimes where the sign is
identifiable):
  old_production   the pattern the old code's sole q=2 restart would have used
                   (global fit_idx=2 under the old mapping)
  new_production   the pattern it uses now (per-mode index 0 = no flips)
  enumerate_all    every pattern; a mapping-invariant ceiling, and a check that
                   both arms are drawn from the same set

Recovery is measured against truth: per-column sign correctness (resolving the
column PERMUTATION only, since for CFUSN the per-column sign is identifiable)
and ||Delta||. A degradation would be new_production scoring below
old_production.

Usage:
    python tests/cfusn_simulations/sim_sign_remap_ab.py
    python tests/cfusn_simulations/sim_sign_remap_ab.py --latent-q 2 --n-seeds 6
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
from tests.cfusn_simulations.sim_utils import sample_cfusn_mixture, match_components

RESULTS_DIR = Path(__file__).resolve().parent / "results"
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


def _perm_only(Dt, Df):
    q = Dt.shape[1]
    best = None
    for perm in itertools.permutations(range(q)):
        cand = Df[:, perm]
        err = float(np.linalg.norm(np.abs(cand) - np.abs(Dt)))
        if best is None or err < best[0]:
            best = (err, cand)
    return best[1]


def score(params, truth, q):
    """(fraction of columns with the correct sign, mean ||Delta||)."""
    row_ind, col_ind, _ = match_components(truth, params)
    right = tot = 0
    norms = []
    for ti, fi in zip(row_ind, col_ind):
        Dt = _ensure_matrix_delta(truth[ti][1])
        Df = _perm_only(Dt, _ensure_matrix_delta(params[fi][1]))
        norms.append(float(np.linalg.norm(Df)))
        for j in range(q):
            if np.linalg.norm(Dt[:, j]) < 1e-9:
                continue
            tot += 1
            right += float(np.dot(Dt[:, j], Df[:, j])) > 0
    return (right / max(tot, 1)), float(np.mean(norms))


def run(X, SA, K, q, li, seed, iters):
    res = tryToFit(X, SA, K, False, "kmeans", "scale", multivariate=True,
                   latent_q=q, lambdaIndex=li, num_fits=1,
                   fit_seed=int(seed * 977 + li), max_em_iters=iters,
                   check_monotonic=False, verbose=False, verbose_init=False,
                   raise_on_error=False)
    p = res.get("component_params")
    if not p or any(len(x) == 0 for x in p):
        return None
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latent-q", type=int, default=2)
    ap.add_argument("--n-components", type=int, default=2)
    ap.add_argument("--n-dims", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=300)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--max-em-iters", type=int, default=1200)
    ap.add_argument("--regimes", nargs="+", default=["moderate", "large"])
    args = ap.parse_args()

    K, p, q = args.n_components, args.n_dims, args.latent_q
    n_total = (2 ** q) ** K
    n_patterns = min(n_total, 100)
    OLD_GLOBAL_IDX = 2                      # old: sole q=2 restart was fit_idx=2
    old_equiv = (n_total - 1) - OLD_GLOBAL_IDX
    print(f"K={K} p={p} q={q}  patterns={n_total} (capped {n_patterns})")
    print(f"  old production pattern -> new-code lambdaIndex {old_equiv}")
    print(f"  new production pattern -> new-code lambdaIndex 0 (no flips)\n")

    rows = []
    hdr = (f"{'regime':>9} {'arm':>16} {'fits':>5} {'sign correct':>13} "
           f"{'||D|| fit':>10} {'||D|| true':>11}")
    print(hdr, flush=True)
    for regime in args.regimes:
        scale = REGIME_SCALE[regime]
        truth, _ts = build_truth(K, p, q, scale)
        true_norm = float(np.mean([np.linalg.norm(_ensure_matrix_delta(t[1]))
                                   for t in truth]))
        agg = {}
        for seed in range(args.n_seeds):
            rng = np.random.RandomState(700 + seed)
            W = np.full((2, K), 1.0 / K)
            X, SA, _ = sample_cfusn_mixture(truth, W, [args.n_obs] * 2, rng)

            arms = {"old_production": [old_equiv],
                    "new_production": [0],
                    "enumerate_all": list(range(n_patterns))}
            for name, idxs in arms.items():
                best = None
                for li in idxs:
                    pr = run(X, SA, K, q, li, seed, args.max_em_iters)
                    if pr is None:
                        continue
                    sc, nm = score(pr, truth, q)
                    # within an arm, pick by sign correctness then closeness
                    key = (sc, -abs(nm - true_norm))
                    if best is None or key > best[0]:
                        best = (key, sc, nm)
                if best is None:
                    continue
                agg.setdefault(name, []).append((best[1], best[2]))
                rows.append(dict(regime=regime, arm=name, seed=seed,
                                 n_fits=len(idxs), sign_correct=best[1],
                                 delta_norm=best[2], true_norm=true_norm))
        for name in ("old_production", "new_production", "enumerate_all"):
            if name not in agg:
                continue
            sc = np.mean([a[0] for a in agg[name]])
            nm = np.median([a[1] for a in agg[name]])
            nf = len(arms[name])
            print(f"{regime:>9} {name:>16} {nf:5d} {sc:13.2f} {nm:10.3f} "
                  f"{true_norm:11.3f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"sim_sign_remap_ab_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
