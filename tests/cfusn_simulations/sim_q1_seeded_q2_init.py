#!/usr/bin/env python3
"""Does seeding q=2's column-0 sign with a converged q=1 fit's sign help?

Production already runs a q=1 restart in its gaussian/q1/q2 mode cycle. Since
_init_delta_matrix's column j uses the j-th top eigenvector of the covariance
(top_idx = argsort(eigvals)[::-1][:q]), q=1's single column and q=2's FIRST
column start from the SAME eigenvector direction. So a completed q=1 fit is a
candidate source of "already-optimized" sign information for q=2's column 0,
free of additional restarts (the q=1 restart runs anyway).

Two reasons this might not help, tested here rather than assumed:
  1. If the truth is genuinely rank-2, a q=1 fit is optimizing a RANK-1 model;
     its sign reflects some compromise projection, not necessarily either true
     q=2 direction cleanly.
  2. q=1's own sign was already measured as sticky (sim_sign_recovery_to_truth:
     only 1 of 4 enumerated patterns reaches truth), and production's actual
     q=1 restart starts from the SAME data-driven sign q=2's column 0 already
     uses. If q=1 EM mostly just confirms its own starting sign, this seeding
     adds no information over today's column-0 initialization.

Arms, all starting from column 0 = q=1's converged sign (or data-driven sign
for the "blind" comparison), searching only column 1's sign per component
(2**K patterns instead of 4**K, using the SAME lambdaIndex bit-encoding as
production -- bit 0 of each component's 2-bit digit is FORCED, bit 1 is swept):
  q1_seeded         column 0 = q=1 fit's converged sign; enumerate column 1
  data_driven_col0  column 0 = data-driven sign (no q=1 fit); enumerate column 1
  greedy_full       unrestricted coordinate descent over all 2*K slots (ceiling)
  enumerate_all     unrestricted full enumeration (ceiling)

If q1_seeded beats data_driven_col0, the q=1 fit is worth its keep as free init
signal. If they tie, the mechanism-level concern above is confirmed and q=1's
converged sign is not adding information over the raw data-driven sign.

Usage:
    python tests/cfusn_simulations/sim_q1_seeded_q2_init.py --n-seeds 10
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

from src.assay_calibration.fit_utils.fit import (
    tryToFit, _pad_component_params_to_q,
)
from src.assay_calibration.fit_utils.cfusn.density_utils import (
    log_joint_densities, _ensure_matrix_delta,
)
from tests.cfusn_simulations.sim_utils import sample_cfusn_mixture, match_components
from scipy.special import logsumexp

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REGIME_SCALE = {"moderate": 0.7, "large": 1.4}


def build_truth(K, p, q, scale, seed=0):
    rng = np.random.RandomState(20260924 + seed)
    params, signs = [], []
    for c in range(K):
        mu = np.linspace(-2.0, 2.0, K)[c] * np.ones(p)
        B, _ = np.linalg.qr(rng.randn(p, q))
        D = np.zeros((p, q))
        sg = []
        for j in range(q):
            s = 1.0 if ((c + j) % 2 == 0) else -1.0
            D[:, j] = s * scale * B[:, j] * np.sqrt(p)
            sg.append(int(s))
        params.append((mu, D, np.eye(p) * 0.6))
        signs.append(tuple(sg))
    return params, signs


def val_ll(params, weights, X, SA):
    if params is None:
        return -np.inf
    tot = 0.0
    for i in range(SA.shape[1]):
        sel = SA[:, i]
        if not sel.any():
            continue
        tot += float(logsumexp(
            log_joint_densities(X[sel], params, weights[i], multivariate=True),
            axis=0).sum())
    return tot / len(X)


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


def fit_q(X, SA, K, q, li, seed, iters, nominal_q=None):
    res = tryToFit(X, SA, K, False, "kmeans", "scale", multivariate=True,
                   latent_q=q, lambdaIndex=li, num_fits=1,
                   fit_seed=int(seed), max_em_iters=iters,
                   check_monotonic=False, verbose=False, verbose_init=False,
                   raise_on_error=False)
    p = res.get("component_params")
    w = res.get("weights")
    if not p or any(len(x) == 0 for x in p):
        return None, None
    if nominal_q is not None and q < nominal_q:
        p = _pad_component_params_to_q(p, nominal_q)
    return p, w


def component_sign(Delta_col):
    """Sign of a fitted q=1 column (or q=2's column j): + if positive-mean
    direction, else -. Delta_col is a 1-D array."""
    # use the largest-magnitude entry's sign as the column's representative
    # sign, matching how _init_delta_matrix's own bit encoding treats a column
    i = int(np.argmax(np.abs(Delta_col)))
    return 1 if Delta_col[i] >= 0 else -1


def q2_lambda_index_col0_fixed(K, q, col0_signs, col1_bit):
    """Build a q=2 lambdaIndex with column 0's sign FORCED per component and
    column 1's sign taken from col1_bit (an integer, one bit per component).

    Mirrors kmeans_init_mv's own decoding: component c's 2-bit digit at base
    2**q, bit j -> sign 1-2*bit (bit 0 -> +1). So bit0 = 0 if col0_signs[c] is
    +1 else 1; bit1 = (col1_bit >> c) & 1.
    """
    n = 2 ** q
    idx = 0
    for c in range(K):
        bit0 = 0 if col0_signs[c] > 0 else 1
        bit1 = (col1_bit >> c) & 1
        digit = bit0 | (bit1 << 1)
        idx += digit * (n ** c)
    return idx


def run_seeded_arm(name, col0_source, X, SA, Xv, SAv, K, q, truth, seed, iters,
                   q1_params_cache):
    """col0_source: 'q1' uses a converged q=1 fit's signs; 'data' uses the
    data-driven sign directly (lambdaIndex 0's column-0 bits, i.e. all +1,
    since the q=1 restart and column 0 share the same starting eigenvector and
    therefore the same starting sign before either optimizes)."""
    if col0_source == "q1":
        if q1_params_cache[0] is None:
            p1, _ = fit_q(X, SA, K, 1, 0, seed * 1000, iters)
            q1_params_cache[0] = p1
        p1 = q1_params_cache[0]
        if p1 is None:
            return None
        col0_signs = [component_sign(np.atleast_1d(p1[c][1]).ravel())
                      for c in range(K)]
        n_fits_for_q1 = 1
    else:
        col0_signs = [1] * K
        n_fits_for_q1 = 0

    best = (-np.inf, None)
    for col1_bit in range(2 ** K):
        li = q2_lambda_index_col0_fixed(K, q, col0_signs, col1_bit)
        pr, pw = fit_q(X, SA, K, q, li, seed * 1000 + 100 + col1_bit, iters)
        if pr is None:
            continue
        v = val_ll(pr, pw, Xv, SAv)
        if v > best[0]:
            best = (v, pr)
    used = n_fits_for_q1 + 2 ** K
    if best[1] is None:
        return dict(val_ll=np.nan, sign_correct=np.nan, delta_norm=np.nan,
                    n_fits=used)
    sc, dn = score(best[1], truth, q)
    return dict(val_ll=best[0], sign_correct=sc, delta_norm=dn, n_fits=used)


def run_greedy_full(X, SA, Xv, SAv, K, q, truth, seed, iters, n_patterns):
    cur = 0
    p, w = fit_q(X, SA, K, q, cur, seed * 1000, iters)
    best_v = val_ll(p, w, Xv, SAv) if p is not None else -np.inf
    best_p = p
    used = 1
    for slot in range(K * q):
        cand = cur ^ (1 << slot)
        if cand >= n_patterns:
            continue
        pr, pw = fit_q(X, SA, K, q, cand, seed * 1000 + used, iters)
        used += 1
        if pr is None:
            continue
        v = val_ll(pr, pw, Xv, SAv)
        if v > best_v:
            best_v, best_p, cur = v, pr, cand
    if best_p is None:
        return dict(val_ll=np.nan, sign_correct=np.nan, delta_norm=np.nan,
                    n_fits=used)
    sc, dn = score(best_p, truth, q)
    return dict(val_ll=best_v, sign_correct=sc, delta_norm=dn, n_fits=used)


def run_enumerate_all(X, SA, Xv, SAv, K, q, truth, seed, iters, n_patterns):
    best = (-np.inf, None)
    for li in range(n_patterns):
        pr, pw = fit_q(X, SA, K, q, li, seed * 1000 + li, iters)
        if pr is None:
            continue
        v = val_ll(pr, pw, Xv, SAv)
        if v > best[0]:
            best = (v, pr)
    if best[1] is None:
        return dict(val_ll=np.nan, sign_correct=np.nan, delta_norm=np.nan,
                    n_fits=n_patterns)
    sc, dn = score(best[1], truth, q)
    return dict(val_ll=best[0], sign_correct=sc, delta_norm=dn, n_fits=n_patterns)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-components", type=int, default=2)
    ap.add_argument("--n-dims", type=int, default=2)
    ap.add_argument("--n-obs", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--max-em-iters", type=int, default=2000)
    ap.add_argument("--regimes", nargs="+", default=["moderate", "large"])
    args = ap.parse_args()

    K, p, q = args.n_components, args.n_dims, 2
    n_patterns = min((2 ** q) ** K, 100)
    print(f"K={K} p={p} q={q}  n_patterns={n_patterns}  iters={args.max_em_iters}",
          flush=True)

    rows = []
    hdr = (f"{'regime':>9} {'arm':>18} {'fits':>5} {'sign correct':>13} "
           f"{'||D|| fit':>10} {'||D|| true':>11} {'val_ll':>10}")
    print(hdr, flush=True)
    for regime in args.regimes:
        scale = REGIME_SCALE[regime]
        agg = {}
        for seed in range(args.n_seeds):
            rng = np.random.RandomState(700 + seed)
            truth, _ = build_truth(K, p, q, scale, seed=seed)
            true_norm = float(np.mean([np.linalg.norm(_ensure_matrix_delta(t[1]))
                                       for t in truth]))
            W = np.full((2, K), 1.0 / K)
            X, SA, _ = sample_cfusn_mixture(truth, W, [args.n_obs] * 2, rng)
            Xv, SAv, _ = sample_cfusn_mixture(truth, W, [args.n_obs // 2] * 2,
                                              rng)

            q1_cache = [None]
            r_q1 = run_seeded_arm("q1_seeded", "q1", X, SA, Xv, SAv, K, q,
                                  truth, seed, args.max_em_iters, q1_cache)
            r_dc = run_seeded_arm("data_driven_col0", "data", X, SA, Xv, SAv,
                                  K, q, truth, seed, args.max_em_iters, [None])
            r_gr = run_greedy_full(X, SA, Xv, SAv, K, q, truth, seed,
                                   args.max_em_iters, n_patterns)
            r_en = run_enumerate_all(X, SA, Xv, SAv, K, q, truth, seed,
                                     args.max_em_iters, n_patterns)

            for name, r in (("q1_seeded", r_q1), ("data_driven_col0", r_dc),
                            ("greedy_full", r_gr), ("enumerate_all", r_en)):
                r2 = dict(r, regime=regime, seed=seed, true_norm=true_norm)
                rows.append(r2)
                agg.setdefault(name, []).append(r2)

        for name in ("q1_seeded", "data_driven_col0", "greedy_full",
                    "enumerate_all"):
            rs = agg.get(name, [])
            if not rs:
                continue
            print(f"{regime:>9} {name:>18} {rs[0]['n_fits']:5d} "
                  f"{np.nanmean([r['sign_correct'] for r in rs]):13.2f} "
                  f"{np.nanmedian([r['delta_norm'] for r in rs]):10.3f} "
                  f"{np.nanmedian([r['true_norm'] for r in rs]):11.3f} "
                  f"{np.nanmedian([r['val_ll'] for r in rs]):10.4f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"sim_q1_seeded_q2_init_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
