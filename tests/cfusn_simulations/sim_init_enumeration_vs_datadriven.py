#!/usr/bin/env python3
"""Does the production restart budget actually recover the true skew?

Distinct from the two existing sims:
  - sim_delta_init_sign_reliability.py asks whether the DATA-DRIVEN SIGN
    ESTIMATOR is accurate, in isolation, with no EM. Its answer: barely better
    than a coin flip (0.49-0.60) except for large skew at large n (0.96 @
    n=5000).
  - sim_delta_init_restart_diversity.py asks whether cycling the init MAGNITUDE
    across an already-sign-diverse set of 4 restarts beats a fixed magnitude.

Neither asks the end-to-end question: at a given restart BUDGET, does the fit
recover the true Delta, and how much does the budget matter? That is what
decides whether production's num_fits=3 is adequate, because of how
Fit.generate_fit_jobs allocates it:

    n_patterns  = min((2**q)**K, 100)
    lambdaIndex = fit_idx % n_patterns

and _restart_mode_sequence makes fit_idx 0 Gaussian (Delta pinned at 0) and
fit_idx 1 latent_q=1 -- so exactly ONE restart explores q=2 sign space, at
lambdaIndex=2. Worse, the sign pattern is a multiplicative FLIP of the
data-driven base sign, so "trust the data sign" is the all-(+1) pattern, which
needs lambdaIndex = sum_c (2**q - 1) * (2**q)**c -- 255 for K=4, q=2, beyond the
cap of 100 and therefore unreachable.

Arms (all selecting best-of-budget by validation log-likelihood, as production
does):
  production     the real 3-restart mode cycle, lambdaIndex = i % n_patterns
  data_driven    one restart at the all-(+1) pattern (trust the data sign)
  enumerate_k    k sign patterns, k swept, all at nominal q
  enumerate_all  every pattern (upper bound on what enumeration can buy)

Reports recovered ||Delta|| against truth and the fraction of components whose
skew DIRECTION is recovered (sign resolved via sim_utils.resolve_delta_ambiguity
after Hungarian component matching, so column permutation/sign ambiguity of the
CFUSN parameterisation is not counted as an error).

Usage:
    python tests/cfusn_simulations/sim_init_enumeration_vs_datadriven.py
    python tests/cfusn_simulations/sim_init_enumeration_vs_datadriven.py \
        --n-seeds 12 --n-obs 400 --latent-q 2 --regimes small large
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import (
    tryToFit, _restart_mode_sequence, _pad_component_params_to_q,
)
from src.assay_calibration.fit_utils.cfusn.density_utils import (
    log_joint_densities, _ensure_matrix_delta,
)
from tests.cfusn_simulations.sim_utils import (
    sample_cfusn_mixture, match_components, resolve_delta_ambiguity,
)
from scipy.special import logsumexp

RESULTS_DIR = Path(__file__).resolve().parent / "results"

REGIME_SCALE = {"zero": 0.0, "small": 0.3, "medium": 0.7, "large": 1.4}


def build_truth(K, p, q, scale, rng):
    """K components with known Delta at the requested skew magnitude.

    Columns are orthonormalised so each skewness direction is independent and
    its sign is identifiable; a generator that made them parallel produced
    rank-1 Delta and meaningless per-column sign metrics.
    """
    params = []
    for c in range(K):
        mu = np.linspace(-2.0, 2.0, K)[c] * np.ones(p)
        Delta = np.zeros((p, q))
        if scale > 0:
            B, _ = np.linalg.qr(rng.randn(p, q))
            for j in range(q):
                # alternate the true sign across components so no single flip
                # pattern is trivially right for all of them
                sgn = 1.0 if ((c + j) % 2 == 0) else -1.0
                Delta[:, j] = sgn * scale * B[:, j] * np.sqrt(p)
        Gamma = np.eye(p) * 0.6
        params.append((mu, Delta if q > 1 else Delta[:, 0], Gamma))
    return params


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


def one_fit(X, SA, K, q, lambda_index, force_gaussian, seed, max_iters,
            nominal_q=None):
    """Run one restart. When q < nominal_q (the mixed q=1 restart in the
    production mode cycle), pad Delta back up to nominal_q columns -- exactly
    what execute_fit_job does via _pad_component_params_to_q before a q=1
    winner is ever handed to a caller. Skipping that here is what crashed
    evaluate() on the shape assertion in resolve_delta_ambiguity: a genuine
    q=1 restart winning selection produced a (p, 1) Delta where every other
    arm returns (p, nominal_q).
    """
    res = tryToFit(
        X, SA, K, False, "kmeans", "scale", multivariate=True,
        latent_q=q, lambdaIndex=lambda_index, force_gaussian=force_gaussian,
        num_fits=1, fit_seed=seed, max_em_iters=max_iters,
        check_monotonic=False, verbose=False, verbose_init=False,
        raise_on_error=False,
    )
    params = res.get("component_params")
    if not params or any(len(p) == 0 for p in params):
        return None, None, -np.inf
    if nominal_q is not None and q < nominal_q:
        params = _pad_component_params_to_q(params, nominal_q)
    return params, res.get("weights"), None


def evaluate(params, truth, q):
    """(recovered ||Delta|| per component, fraction of directions recovered)."""
    row_ind, col_ind, _cost = match_components(truth, params)
    norms, right = [], 0
    for ti, fi in zip(row_ind, col_ind):
        Dt = _ensure_matrix_delta(truth[ti][1])
        Df = _ensure_matrix_delta(params[fi][1])
        res = resolve_delta_ambiguity(Dt, Df)
        Df_res = res[0] if isinstance(res, tuple) else res
        norms.append(float(np.linalg.norm(Df_res)))
        if np.linalg.norm(Dt) > 1e-9:
            # direction recovered if the resolved fit correlates positively
            cos = float(np.sum(Dt * Df_res) /
                        max(np.linalg.norm(Dt) * np.linalg.norm(Df_res), 1e-12))
            right += cos > 0.5
    return float(np.mean(norms)), right / max(len(row_ind), 1)


def run_greedy(X, SA, Xv, SAv, K, q, truth, seed, max_iters, n_patterns,
               passes=1):
    """Coordinate descent over the sign pattern: O(K*q) fits, not (2**q)**K.

    Full enumeration is not a candidate policy at production sizes -- (2**q)**K
    is 256 at K=4 and 4096 at K=6 -- so the practical question is what recovers
    most of enumeration's benefit within a budget that grows linearly. Start
    from the all-(+1) pattern, then sweep each (component, column) slot once,
    flipping it and keeping the flip when validation log-likelihood improves.
    Cost is 1 + K*q fits per pass (9 at K=4, q=2), versus 256 for enumeration.

    Returns (val_ll, params, n_fits_used).
    """
    cur = 0                                   # all-(+1): no flips
    params, w, _ = one_fit(X, SA, K, q, cur, False, seed * 1000, max_iters)
    best_v = val_ll(params, w, Xv, SAv) if params is not None else -np.inf
    best_p = params
    used = 1
    for _ in range(passes):
        improved = False
        for slot in range(K * q):
            cand = cur ^ (1 << slot)          # flip one (component, column)
            if cand >= n_patterns:
                continue
            pr, pw, _ = one_fit(X, SA, K, q, cand, False,
                                seed * 1000 + used, max_iters)
            used += 1
            if pr is None:
                continue
            v = val_ll(pr, pw, Xv, SAv)
            if v > best_v:
                best_v, best_p, cur = v, pr, cand
                improved = True
        if not improved:
            break
    return best_v, best_p, used


def run_arm(name, budget_specs, X, SA, Xv, SAv, K, q, truth, seed, max_iters):
    best = (-np.inf, None, None)
    for j, (li, fg, jq) in enumerate(budget_specs):
        params, w, _ = one_fit(X, SA, K, jq, li, fg, seed * 1000 + j, max_iters,
                               nominal_q=q)
        if params is None:
            continue
        v = val_ll(params, w, Xv, SAv)
        if v > best[0]:
            best = (v, params, w)
    if best[1] is None:
        return dict(arm=name, n_fits=len(budget_specs), val_ll=np.nan,
                    delta_norm=np.nan, dir_recovered=np.nan)
    dn, dr = evaluate(best[1], truth, q)
    return dict(arm=name, n_fits=len(budget_specs), val_ll=best[0],
                delta_norm=dn, dir_recovered=dr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=300)
    ap.add_argument("--latent-q", type=int, default=2)
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument("--n-dims", type=int, default=3)
    ap.add_argument("--max-em-iters", type=int, default=2000)
    ap.add_argument("--regimes", nargs="+",
                    default=["zero", "small", "medium", "large"])
    ap.add_argument("--enumerate-k", nargs="+", type=int, default=[4, 8, 16])
    args = ap.parse_args()

    K, p, q = args.n_components, args.n_dims, args.latent_q
    n_patterns = min((2 ** q) ** K, 100)
    # The all-(+1) "trust the data-driven sign" pattern is lambdaIndex 0: the
    # decoding maps bit 0 -> +1, so index 0 applies no flips. This previously
    # computed the all-bits-set index, which was correct only under the OLD
    # decoding; after the re-map it silently aimed the data_driven arm at the
    # all-FLIP pattern, the opposite of its name.
    all_pos = 0
    modes = _restart_mode_sequence(3)

    print(f"K={K} p={p} q={q}  n_patterns={min((2**q)**K, 100)} "
          f"(uncapped {(2**q)**K})", flush=True)
    print(f"production restart modes for num_fits=3: {modes}", flush=True)
    print(f"all-(+1) 'trust data sign' lambdaIndex={all_pos} "
          f"reachable={all_pos < n_patterns}\n", flush=True)

    prod_specs = []
    for i, m in enumerate(modes):
        jq = 1 if m == "q1" else q
        prod_specs.append((i % min((2 ** jq) ** K, 100), m == "gaussian", jq))

    rows = []
    hdr = (f"{'regime':>8} {'arm':>14} {'fits':>5} {'||D|| fit':>10} "
           f"{'||D|| true':>11} {'dir ok':>8} {'val_ll':>10}")
    print(hdr, flush=True)
    for regime in args.regimes:
        scale = REGIME_SCALE[regime]
        agg = {}
        for seed in range(args.n_seeds):
            rng = np.random.RandomState(1000 + seed)
            truth = build_truth(K, p, q, scale, rng)
            W = np.full((2, K), 1.0 / K)
            X, SA, _ = sample_cfusn_mixture(truth, W, [args.n_obs] * 2, rng)
            Xv, SAv, _ = sample_cfusn_mixture(truth, W, [args.n_obs // 2] * 2, rng)
            true_norm = float(np.mean([np.linalg.norm(_ensure_matrix_delta(t[1]))
                                       for t in truth]))

            arms = [("production", prod_specs),
                    ("data_driven", [(all_pos % n_patterns, False, q)])]
            for k in args.enumerate_k:
                arms.append((f"enumerate_{k}",
                             [(i % n_patterns, False, q) for i in range(k)]))
            if n_patterns <= 64:
                arms.append(("enumerate_all",
                             [(i, False, q) for i in range(n_patterns)]))

            gv, gp, gused = run_greedy(X, SA, Xv, SAv, K, q, truth, seed,
                                       args.max_em_iters, n_patterns)
            if gp is not None:
                gdn, gdr = evaluate(gp, truth, q)
                gr = dict(arm="greedy", n_fits=gused, val_ll=gv,
                          delta_norm=gdn, dir_recovered=gdr,
                          regime=regime, seed=seed, true_norm=true_norm, secs=0.0)
                rows.append(gr)
                agg.setdefault("greedy", []).append(gr)

            for name, specs in arms:
                t0 = time.time()
                r = run_arm(name, specs, X, SA, Xv, SAv, K, q, truth,
                            seed, args.max_em_iters)
                r.update(regime=regime, seed=seed, true_norm=true_norm,
                         secs=time.time() - t0)
                rows.append(r)
                agg.setdefault(name, []).append(r)

        for name in agg:
            rs = agg[name]
            print(f"{regime:>8} {name:>14} {rs[0]['n_fits']:5d} "
                  f"{np.nanmedian([r['delta_norm'] for r in rs]):10.3f} "
                  f"{np.nanmedian([r['true_norm'] for r in rs]):11.3f} "
                  f"{np.nanmean([r['dir_recovered'] for r in rs]):8.2f} "
                  f"{np.nanmedian([r['val_ll'] for r in rs]):10.4f}", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / ("sim_init_enumeration_vs_datadriven_"
                         f"{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=sorted(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
