#!/usr/bin/env python
"""TEST (not production): the anchored all-assayed fit
(hpc/run_anchored_fit_test.py) with the free components' sign pattern chosen
once on the full data.

Full data (all labelled rows and the whole pool, no resampling): fit the
anchor, then search the free components' sign pattern (q=2): the default
pattern and every single (component, direction) flip are fitted in
parallel, the best by full-data log-likelihood is kept, and a second round
of single flips from it follows. The parameter count is the same for every
pattern, so raw likelihood compares them fairly.

Each bootstrap then fits three k-means restarts on top of its own frozen
anchor: one whose free clusters' initial skewness follows the full-data
pattern (kmeans_init_mv reference_params; directions only, never
parameters) and two default restarts with different seeds. Two arms with
equal compute are written:
  struct_<anchor>_k<K>   best (held-out likelihood) of aligned + default #1
  default_<anchor>_k<K>  best of default #1 + default #2
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
from src.assay_calibration.fit_utils.fit import _weighted_val_ll, next_greedy_lambda
from src.assay_calibration.fit_utils.cfusn.initializations import kmeans_init_mv
from src.assay_calibration.fit_utils.cfusn.density_utils import get_likelihood


def full_data(d):
    """bootstrap_data's layout for the full data: every labelled row and the
    whole pool, used both for fitting and as the (unused) held-out set."""
    bd = T.bootstrap_data(d, 0)
    n = len(bd["obs"])
    bd["tr"] = np.arange(n)
    bd["va"] = np.arange(n)
    if d["pool"] is not None:
        bd["pool_tr"] = d["pool"]
        bd["pool_va"] = d["pool"][:0]
    return bd


def fit_free(bd, anchor, k_free, pop, seed, lam=0, ref=None, split_val=True):
    obs, sa, cols = T.fit_rows(bd, pop, "tr")
    n_a = len(anchor)
    dens = T.mix_logpdf(obs, anchor, np.full(n_a, 1.0 / n_a))
    worst = np.argsort(dens)[: max(30, int(0.4 * len(obs)))]
    free, _ = kmeans_init_mv(obs[worst], n_clusters=k_free, constrained=False, latent_q=T.LATENT_Q,
                             rng=np.random.RandomState(seed), lambdaIndex=lam, reference_params=ref)
    init = list(anchor) + T._arr(free)
    W0 = np.zeros((sa.shape[1], n_a + k_free))
    for r, c in enumerate(cols):
        a_share = 0.8 if c in (T.B, T.S) else 0.5
        W0[r, :n_a] = a_share / n_a
        W0[r, n_a:] = (1 - a_share) / k_free
    f = T._fit(obs, sa, n_a + k_free, seed, initial_params=init, initial_weights=W0, frozen=set(range(n_a)))
    params, W = T._arr(f["component_params"]), np.asarray(f["weights"], float)
    if not split_val:
        return params, W, cols, float(get_likelihood(obs, sa, params, W, multivariate=True))
    vobs, vsa, vcols = T.fit_rows(bd, pop, "va")
    keepv = [vcols.index(c) for c in cols if c in vcols]
    vrow = [cols.index(c) for c in vcols if c in cols]
    return params, W, cols, float(_weighted_val_ll(vobs, vsa[:, keepv], params, W[vrow], mv=True, fit_kwargs={}))


def full_candidate(args):
    key, bd, anchor, k_free, pop, lam = args
    try:
        params, W, cols, ll = fit_free(bd, anchor, k_free, pop, 500 + lam, lam=lam, split_val=False)
        return key, lam, params, ll, None
    except Exception as e:
        return key, lam, None, -np.inf, repr(e)


def boot_task(args):
    key, i, bd, anchor, k_free, pop, arm, seed, ref = args
    try:
        params, W, cols, v = fit_free(bd, anchor, k_free, pop, seed, ref=ref)
        return key, i, arm, params, W, cols, v, None
    except Exception as e:
        return key, i, arm, None, None, None, np.nan, repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", nargs="+", default=["ret", "GCK", "BRCA1"])
    ap.add_argument("--anchor", default="B0.1")
    ap.add_argument("--k-free", type=int, default=4)
    ap.add_argument("--pool-type", default="all_assayed")
    ap.add_argument("--n-bootstraps", type=int, default=20)
    ap.add_argument("--max-workers", type=int, default=100)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    T.POOL_MULT = 0.0  # whole pool
    out = Path(args.output_dir)
    pop, K = "allassayed", args.k_free

    data = {k: T.load_gene(k, args.pool_type) for k in args.genes}
    full = {k: full_data(d) for k, d in data.items()}
    boots = {(k, i): T.bootstrap_data(d, i) for k, d in data.items() for i in range(args.n_bootstraps)}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        # stage 1 on the full data and on every bootstrap
        anchors = {}
        jobs = [(k, -1, bd, 3) for k, bd in full.items()] + [(k, i, bd, 2) for (k, i), bd in boots.items()]
        for k, i, a in ex.map(T.fit_anchors, jobs):
            anchors[(k, i)] = a
        print("anchors done", flush=True)

        # full-data sign search for the free components: two rounds of single flips
        n_patterns = min((2 ** T.LATENT_Q) ** K, 100)
        best = {k: (0, None, -np.inf) for k in data}
        tried = {k: set() for k in data}
        search_log = {k: [] for k in data}
        for rnd in range(2):
            tasks = []
            for k in data:
                anc = T.anchor_for(args.anchor, anchors[(k, -1)])
                cur = best[k][0]
                cands = ([cur] if rnd == 0 else []) + [c for s in range(K * T.LATENT_Q)
                                                       if (c := next_greedy_lambda(cur, s, n_patterns)) is not None]
                for lam in cands:
                    if lam not in tried[k]:
                        tried[k].add(lam)
                        tasks.append((k, full[k], anc, K, pop, lam))
            for k, lam, params, ll, err in ex.map(full_candidate, tasks):
                search_log[k].append(dict(round=rnd, lambdaIndex=lam, full_ll=ll, error=err))
                if params is not None and ll > best[k][2]:
                    best[k] = (lam, params, ll)
            print(f"full-data search round {rnd}: " +
                  ", ".join(f"{k} best lambdaIndex={best[k][0]} ll={best[k][2]:.1f}" for k in data), flush=True)

        n_a_full = {k: len(T.anchor_for(args.anchor, anchors[(k, -1)])) for k in data}
        tasks = []
        for (k, i), bd in boots.items():
            anc = T.anchor_for(args.anchor, anchors[(k, i)])
            if anc is None:
                continue
            ref = best[k][1][n_a_full[k]:] if best[k][1] is not None else None
            tasks += [(k, i, bd, anc, K, pop, "aligned", 7000 + i, ref),
                      (k, i, bd, anc, K, pop, "default1", 8000 + i, None),
                      (k, i, bd, anc, K, pop, "default2", 9000 + i, None)]
        print(f"bootstrap fits: {len(tasks)}", flush=True)
        res = list(ex.map(boot_task, tasks, chunksize=1))

    by = {}
    for k, i, arm, params, W, cols, v, err in res:
        if err:
            print(f"  {k} b{i} {arm}: failed {err}", flush=True)
            continue
        by.setdefault((k, i), {})[arm] = (params, W, cols, v)
    wins = {}
    for (k, i), arms in by.items():
        d, bd = data[k], boots[(k, i)]
        out_dir = out / d["dataset"]
        out_dir.mkdir(parents=True, exist_ok=True)
        entries = {}
        for name, members in ((f"struct_{args.anchor}_k{K}", ("aligned", "default1")),
                              (f"default_{args.anchor}_k{K}", ("default1", "default2"))):
            cand = [(arms[m][3], m) for m in members if m in arms and np.isfinite(arms[m][3])]
            if not cand:
                continue
            v, m = max(cand)
            if name.startswith("struct"):
                wins.setdefault(k, {}).setdefault(m, 0)
                wins[k][m] += 1
            params, W, cols, _ = arms[m]
            Wout = T.output_weights(params, W, cols, bd)
            entries[name] = {"dataset_name": d["dataset"], "bootstrap_seed": i, "num_components": len(params),
                             "fit_idx": m, "fit": {"component_params": params, "weights": Wout}, "val_ll": v,
                             "assay_datasets": list(d["ms"].dataset_names), "calibrated_dims": None, "n_cap_hits": 0}
        with open(out_dir / f"bootstrap_{i}_best_fits.pkl", "wb") as fh:
            pickle.dump(entries, fh)
        pd.DataFrame([dict(gene=k, bootstrap=i, model=n, n_anchor=len(T.anchor_for(args.anchor, anchors[(k, i)])),
                           val_ll=e["val_ll"], plp_weight_on_anchor=np.nan, heldout_ll_plp=np.nan,
                           heldout_ll_blb=np.nan) for n, e in entries.items()]).to_csv(
            out_dir / f"diag_bootstrap_{i}.csv", index=False)
    for k in data:
        (out / data[k]["dataset"] / "fullsearch.json").write_text(
            json.dumps(dict(best_lambdaIndex=best[k][0], best_full_ll=best[k][2], log=search_log[k]), default=float))
    print("structured arm: winning restart per bootstrap:", json.dumps(wins), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
