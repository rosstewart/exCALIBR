#!/usr/bin/env python
"""TEST (not production): per-component families in the anchored all-assayed
fit (hpc/run_anchored_fit_test.py), chosen by forward search.

Per bootstrap and number of free components K, on top of the frozen anchor
and from one fixed k-means start:
  round 0   every free component Gaussian (Delta pinned at zero via
            single_fit's gaussian_components)
  round r   try upgrading each still-Gaussian free component to CFUSN; keep
            the best upgrade if it improves the balanced per-sample held-out
            log-likelihood (mean over P/LP, B/LB, synonymous and phenotype
            samples, each weighted equally), else stop
An all-CFUSN fit from the same start is added for comparison. Config keys:
gauss_<anchor>_k<K>, mixed_<anchor>_k<K>, cfusn_<anchor>_k<K>.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import json
import pickle
import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hpc.run_anchored_fit_test as T
from src.assay_calibration.fit_utils.fit import _weighted_val_ll
from src.assay_calibration.fit_utils.cfusn.initializations import kmeans_init_mv

SCORED = (T.P, T.B, T.S, 4, 5)


def balanced_heldout(bd, params, Wout):
    va_obs, va_sa = bd["obs"][bd["va"]], bd["sa"][bd["va"]]
    vals = []
    for c in SCORED:
        if c < va_sa.shape[1] and va_sa[:, c].any():
            vals.append(float(T.mix_logpdf(va_obs[va_sa[:, c].astype(bool)], params, Wout[c]).mean()))
    return float(np.mean(vals)) if vals else np.nan


def fit_family(args):
    key, i, bd, anchor, k_free, pop, seed, skewed = args
    try:
        obs, sa, cols = T.fit_rows(bd, pop, "tr")
        n_a = len(anchor)
        dens = T.mix_logpdf(obs, anchor, np.full(n_a, 1.0 / n_a))
        worst = np.argsort(dens)[: max(30, int(0.4 * len(obs)))]
        free, _ = kmeans_init_mv(obs[worst], n_clusters=k_free, constrained=False, latent_q=T.LATENT_Q,
                                 rng=np.random.RandomState(seed))
        init = list(anchor) + T._arr(free)
        W0 = np.zeros((sa.shape[1], n_a + k_free))
        for r, c in enumerate(cols):
            a_share = 0.8 if c in (T.B, T.S) else 0.5
            W0[r, :n_a] = a_share / n_a
            W0[r, n_a:] = (1 - a_share) / k_free
        gauss = {n_a + j for j in range(k_free) if j not in skewed}
        kw = dict(latent_q=T.LATENT_Q, fit_seed=seed, verbose=False, max_em_iters=T.MAX_EM_ITERS,
                  rng=np.random.RandomState(seed), initial_params=init, initial_weights=W0,
                  frozen_components=set(range(n_a)))
        if gauss:
            kw["gaussian_components"] = gauss
        f = T.single_fit(obs, sa, n_a + k_free, constrained=False, init_method=None,
                         init_constraint_adjustment=None, multivariate=True, **kw)
        params, W = T._arr(f["component_params"]), np.asarray(f["weights"], float)
        Wout = T.output_weights(params, W, cols, bd)
        vobs, vsa, vcols = T.fit_rows(bd, pop, "va")
        keepv = [vcols.index(c) for c in cols if c in vcols]
        vrow = [cols.index(c) for c in vcols if c in cols]
        val_ll = float(_weighted_val_ll(vobs, vsa[:, keepv], params, W[vrow], mv=True, fit_kwargs={}))
        return key, i, k_free, tuple(sorted(skewed)), params, Wout, balanced_heldout(bd, params, Wout), val_ll, None
    except Exception as e:
        return key, i, k_free, tuple(sorted(skewed)), None, None, np.nan, np.nan, repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", nargs="+", default=["ret"])
    ap.add_argument("--anchor", default="B0.1")
    ap.add_argument("--k-free", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--pool-type", default="all_assayed")
    ap.add_argument("--n-bootstraps", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=100)
    ap.add_argument("--pool-cap", type=int, default=None,
                    help="random subsample of at most this many pool variants per bootstrap")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    T.POOL_MULT = 0.0
    T.POOL_CAP = args.pool_cap
    out = Path(args.output_dir)
    pop = "allassayed"

    data = {k: T.load_gene(k, args.pool_type) for k in args.genes}
    boots = {(k, i): T.bootstrap_data(d, i) for k, d in data.items() for i in range(args.n_bootstraps)}
    results = {}   # (key, i, K) -> {skewed tuple: (params, Wout, bal, val)}
    log = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        anchors = {}
        for k, i, a in ex.map(T.fit_anchors, [(k, i, bd, 2) for (k, i), bd in boots.items()]):
            anchors[(k, i)] = a
        print("anchors done", flush=True)
        units = {(k, i, K): T.anchor_for(args.anchor, anchors[(k, i)])
                 for (k, i) in boots for K in args.k_free}
        units = {u: a for u, a in units.items() if a is not None}
        state = {u: () for u in units}          # current skewed set
        best = {u: -np.inf for u in units}
        active = set(units)
        seed = {u: 31 * u[1] + u[2] for u in units}
        # round 0 (all Gaussian) and the all-CFUSN comparison
        tasks = [(u[0], u[1], boots[u[:2]], units[u], u[2], pop, seed[u], ()) for u in units]
        tasks += [(u[0], u[1], boots[u[:2]], units[u], u[2], pop, seed[u], tuple(range(u[2]))) for u in units]
        rnd = 0
        while tasks:
            for key, i, K, sk, params, Wout, bal, val, err in ex.map(fit_family, tasks, chunksize=1):
                u = (key, i, K)
                log.append(dict(gene=key, bootstrap=i, k_free=K, round=rnd, skewed=list(sk), balanced=bal,
                                val_ll=val, error=err))
                if params is not None:
                    results.setdefault(u, {})[sk] = (params, Wout, bal, val)
            # accept the best improving upgrade per unit
            nxt = []
            for u in sorted(active):
                cur = state[u]
                if rnd == 0:
                    best[u] = results.get(u, {}).get(cur, (None, None, -np.inf))[2]
                else:
                    cands = [(results[u][s][2], s) for s in results.get(u, {})
                             if len(s) == len(cur) + 1 and set(cur) <= set(s) and np.isfinite(results[u][s][2])]
                    if cands and max(cands)[0] > best[u]:
                        best[u], state[u] = max(cands)
                    else:
                        active.discard(u)
                        continue
                remaining = [j for j in range(u[2]) if j not in state[u]]
                if not remaining:
                    active.discard(u)
                    continue
                for j in remaining:
                    s = tuple(sorted(state[u] + (j,)))
                    if s not in results.get(u, {}):
                        nxt.append((u[0], u[1], boots[u[:2]], units[u], u[2], pop, seed[u], s))
            print(f"round {rnd} done; {len(active)} units still searching; next {len(nxt)} fits", flush=True)
            tasks, rnd = nxt, rnd + 1

    pd.DataFrame(log).to_csv(out / "family_search_log.csv", index=False)
    for (key, i), bd in boots.items():
        d = data[key]
        out_dir = out / d["dataset"]
        out_dir.mkdir(parents=True, exist_ok=True)
        entries, diag = {}, []
        for K in args.k_free:
            u = (key, i, K)
            if u not in units:
                continue
            res = results.get(u, {})
            n_a = len(units[u])
            for label, sk in (("gauss", ()), ("mixed", state[u]), ("cfusn", tuple(range(K)))):
                if sk not in res:
                    continue
                params, Wout, bal, val = res[sk]
                name = f"{label}_{args.anchor}_k{K}"
                entries[name] = {"dataset_name": d["dataset"], "bootstrap_seed": i, "num_components": len(params),
                                 "fit_idx": json.dumps(list(sk)), "fit": {"component_params": params, "weights": Wout},
                                 "val_ll": val, "assay_datasets": list(d["ms"].dataset_names),
                                 "calibrated_dims": None, "n_cap_hits": 0}
                diag.append(dict(gene=key, bootstrap=i, model=name, n_anchor=n_a, n_skewed=len(sk), val_ll=val,
                                 balanced_heldout=bal, plp_weight_on_anchor=float(Wout[T.P, :n_a].sum()),
                                 heldout_ll_plp=np.nan, heldout_ll_blb=np.nan))
        with open(out_dir / f"bootstrap_{i}_best_fits.pkl", "wb") as fh:
            pickle.dump(entries, fh)
        pd.DataFrame(diag).to_csv(out_dir / f"diag_bootstrap_{i}.csv", index=False)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
