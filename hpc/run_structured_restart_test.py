#!/usr/bin/env python
"""TEST (not production): choose the sign patterns and the component count
once on the full data, then fit each bootstrap with a few k-means restarts
whose initial skewness directions follow that structure.

Full data (all labelled rows, no resampling), for each K in --components:
  q=0  one Gaussian fit
  q=1  1 + K restarts, best by full-data log-likelihood
  q=2  greedy sign search as in production (seed lambdaIndex 0, then one
       flip per (component, direction), kept when the full-data
       log-likelihood improves), best by full-data log-likelihood
Within a (K, q) the parameter count is fixed, so raw likelihood compares
restarts fairly; K is chosen by BIC over the best fit of every (K, q).

Each bootstrap (same resample as production) then fits the chosen K with
four k-means restarts, picked by held-out log-likelihood as in production:
  q=0 Gaussian; q=1 and q=2 with each cluster's initial skewness pointed
  like its matching full-data component (kmeans_init_mv reference_params;
  only directions carry over, never parameters); and one default q=2
  restart (lambdaIndex 0) as a way out when a bootstrap prefers another
  structure.

Writes <output-dir>/<dataset>/bootstrap_{i}_best_fits.pkl with config key
"struct_{K}c" (+ "fullsearch.json" with the full-data choices), ready for
hpc/aggregate_results.py.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import (
    makeOneHot, pattern_stratified_bootstrap, derive_bootstrap_seed, DEFAULT_MASTER_SEED,
    _weighted_val_ll, next_greedy_lambda,
)
from src.assay_calibration.fit_utils.cfusn.fit import single_fit
from src.assay_calibration.fit_utils.cfusn.density_utils import get_likelihood

MAX_EM_ITERS = 2000
GENES = {"ret": "labelseq", "GCK": "integrated", "BRCA1": "integrated", "BRCA1_predictor": "predictor"}


def load(key):
    from mv_analysis import config
    gene, gs = key.split("_")[0], GENES[key]
    ms = config.build_ms_for_gene_cluster(gs, gene)
    return dict(key=key, gene=gene, gs=gs, ms=ms,
                dataset=config.canonical_dataset_name(gene, gs, results_json=config.PAPER_RESULTS_JSON))


def labelled(ms, seed):
    sa = makeOneHot(ms.sample_assignments, rng=np.random.RandomState(seed))
    obs = np.asarray(ms.scores, float)
    keep = sa.any(axis=1) & ~np.isnan(obs).all(axis=1)
    return obs[keep], sa[keep]


def fit_once(obs, sa, K, q, seed, lam=None, ref=None):
    kw = dict(fit_seed=seed, rng=np.random.RandomState(seed), verbose=False, max_em_iters=MAX_EM_ITERS)
    if q == 0:
        kw.update(latent_q=1, force_gaussian=True)
    else:
        kw.update(latent_q=q)
    if lam is not None:
        kw["lambdaIndex"] = lam
    if ref is not None:
        kw["reference_params"] = ref
    return single_fit(obs, sa, K, constrained=False, init_method="kmeans",
                      init_constraint_adjustment=None, multivariate=True, **kw)


def full_ll(obs, sa, f):
    return float(get_likelihood(obs, sa, f["component_params"], f["weights"], multivariate=True))


def n_params(K, p, q, S):
    return K * (p + p * (p + 1) // 2 + p * q) + S * (K - 1)


def full_search(args):
    """One (gene, K, q) unit on the full data."""
    key, obs, sa, K, q = args
    best, best_ll, best_lam, tried = None, -np.inf, None, 0
    try:
        if q == 0:
            cands = [(None, 11)]
        elif q == 1:
            cands = [(None, 100 + r) for r in range(1 + K)]
        else:
            cands = None
        if cands is not None:
            for lam, s in cands:
                try:
                    f = fit_once(obs, sa, K, q, s, lam)
                except Exception:
                    continue
                tried += 1
                ll = full_ll(obs, sa, f)
                if ll > best_ll:
                    best, best_ll, best_lam = f, ll, lam
        else:
            n_patterns = min((2 ** q) ** K, 100)
            cur = 0
            try:
                f = fit_once(obs, sa, K, q, 200, 0)
                best, best_ll, best_lam = f, full_ll(obs, sa, f), 0
                tried += 1
            except Exception:
                pass
            for slot in range(K * q):
                cand = next_greedy_lambda(cur, slot, n_patterns)
                if cand is None:
                    continue
                try:
                    f = fit_once(obs, sa, K, q, 200 + slot + 1, cand)
                except Exception:
                    continue
                tried += 1
                ll = full_ll(obs, sa, f)
                if ll > best_ll:
                    best, best_ll, best_lam, cur = f, ll, cand, cand
    except Exception as e:
        return key, K, q, None, np.nan, None, 0, repr(e)
    return key, K, q, best, best_ll, best_lam, tried, None


def boot_fit(args):
    key, i, tr_obs, tr_sa, va_obs, va_sa, K, q, ref, lam, seed, label = args
    try:
        f = fit_once(tr_obs, tr_sa, K, q, seed, lam, ref)
        v = _weighted_val_ll(va_obs, va_sa, f["component_params"], f["weights"], mv=True, fit_kwargs={})
        return key, i, label, f, float(v), None
    except Exception as e:
        return key, i, label, None, np.nan, repr(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", nargs="+", default=list(GENES))
    ap.add_argument("--components", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--n-bootstraps", type=int, default=20)
    ap.add_argument("--ablate-default", action="store_true",
                    help="give the default CFUSN restart the aligned CFUSN restart's k-means seed (so the two "
                         "differ only in initial skew signs) and save two selections per bootstrap: "
                         "struct4_<K>c (all four restarts) and struct3_<K>c (without the default)")
    ap.add_argument("--fixed-k", action="store_true",
                    help="fit every K in --components (as production does) instead of the BIC choice")
    ap.add_argument("--max-workers", type=int, default=110)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)

    data = {k: load(k) for k in args.genes}
    full = {k: labelled(d["ms"], 12345) for k, d in data.items()}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        units = [(k, full[k][0], full[k][1], K, q) for k in data for K in args.components for q in (0, 1, 2)]
        res = list(ex.map(full_search, units))
        chosen, summary = {}, {}
        for key, K, q, f, ll, lam, tried, err in res:
            obs, sa = full[key]
            bic = -2 * ll + n_params(K, obs.shape[1], q, sa.shape[1]) * np.log(len(obs)) if f is not None else np.inf
            summary.setdefault(key, {})[f"{K}c_q{q}"] = dict(full_ll=ll, bic=bic, lambdaIndex=lam, fits=tried, error=err)
            chosen[(key, K, q)] = f
        picks = {}
        for key in data:
            s = summary[key]
            K_best = min(args.components, key=lambda K: min(s[f"{K}c_q{q}"]["bic"] for q in (0, 1, 2)))
            picks[key] = K_best
            s["chosen_K"] = K_best
            (out / data[key]["dataset"]).mkdir(parents=True, exist_ok=True)
            (out / data[key]["dataset"] / "fullsearch.json").write_text(json.dumps(s, indent=1, default=float))
            print(f"{key}: full-data BIC picks K={K_best}; " +
                  ", ".join(f"{k}: ll={v['full_ll']:.1f} bic={v['bic']:.1f}" for k, v in s.items() if k != "chosen_K"),
                  flush=True)

        tasks = []
        for key, d in data.items():
          for K in (args.components if args.fixed_k else [picks[key]]):
            ref1 = chosen[(key, K, 1)]["component_params"] if chosen[(key, K, 1)] else None
            ref2 = chosen[(key, K, 2)]["component_params"] if chosen[(key, K, 2)] else None
            for i in range(args.n_bootstraps):
                seed = derive_bootstrap_seed(DEFAULT_MASTER_SEED, i)
                obs, sa = labelled(d["ms"], seed)
                tr, va = pattern_stratified_bootstrap(obs, sa, seed)
                base = (key, i, obs[tr], sa[tr], obs[va], sa[va], K)
                label_k = f"_{K}c"
                tasks += [base + (0, None, None, 1000 + i, "q0" + label_k),
                          base + (1, ref1, None, 2000 + i, "q1_ref" + label_k),
                          base + (2, ref2, None, 3000 + i, "q2_ref" + label_k),
                          base + (2, None, 0, (3000 if args.ablate_default else 4000) + i, "q2_default" + label_k)]
        print(f"bootstrap fits: {len(tasks)}", flush=True)
        results = list(ex.map(boot_fit, tasks, chunksize=1))

    selections = {"struct": lambda lab: True}
    if args.ablate_default:
        selections = {"struct4": lambda lab: True, "struct3": lambda lab: not lab.startswith("q2_default")}
    best = {}
    restart_log = []
    for key, i, label, f, v, err in results:
        if err:
            print(f"  {key} b{i} {label}: failed {err}", flush=True)
            continue
        K = int(label.rsplit("_", 1)[1][:-1])
        restart_log.append(dict(gene=key, bootstrap=i, K=K, restart=label.rsplit("_", 1)[0], val_ll=v))
        for sel, keep in selections.items():
            if keep(label) and np.isfinite(v) and ((sel, key, i, K) not in best or v > best[(sel, key, i, K)][1]):
                best[(sel, key, i, K)] = (f, v, label)
    import pandas as pd
    pd.DataFrame(restart_log).to_csv(out / "restart_val_ll.csv", index=False)
    wins, per_file = {}, {}
    for (sel, key, i, K), (f, v, label) in best.items():
        d = data[key]
        wins.setdefault(f"{sel}_{key}_{K}c", {}).setdefault(label, 0)
        wins[f"{sel}_{key}_{K}c"][label] += 1
        entry = {"dataset_name": d["dataset"], "bootstrap_seed": i, "num_components": K, "fit_idx": label,
                 "fit": {"component_params": f["component_params"], "weights": np.asarray(f["weights"])},
                 "val_ll": v, "assay_datasets": list(d["ms"].dataset_names), "calibrated_dims": None,
                 "n_cap_hits": 0}
        per_file.setdefault((d["dataset"], i), {})[f"{sel}_{K}c"] = entry
    for (ds, i), entries in per_file.items():
        with open(out / ds / f"bootstrap_{i}_best_fits.pkl", "wb") as fh:
            pickle.dump(entries, fh)
    print("winning restart per bootstrap:", json.dumps(wins), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
