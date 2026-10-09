#!/usr/bin/env python
"""Grow a gene's converged joint fit with one extra component per phenotype
group, with the phenotype groups entering the fit as their own samples, and
record each grown component's held-out evidence for a data-driven
keep/drop decision.

Per bootstrap (same resample the source fit was trained on, reconstructed
bit-for-bit as in run_grow_component_batch.py), with the source fit's K
components frozen throughout:

  base     source fit + one weight row per phenotype group, estimated on the
           in-bag phenotype variants (phenotype_evidence._posthoc_phenotype_
           weights -- today's post-hoc approach, restricted to in-bag rows)
  +A       base grown by one component seeded from phenotype group A
  +B       base grown by one component seeded from phenotype group B
  +A+B     +A grown by a further component seeded from group B

Every grown model re-estimates all sample weights (P/LP, B/LB, gnomAD,
synonymous and the phenotype groups) with the standard EM; only the new
component's parameters are free. Restarts are chosen by held-out log-
likelihood, as in run_grow_component_batch.py.

Writes, under --output-dir:
  <dataset>/bootstrap_{i}_best_fits.pkl   one config key per model, weights
                                          with the phenotype rows at their
                                          fixed roles (aggregate with
                                          hpc/aggregate_results.py and score
                                          with auxiliary_pathogenic_indices)
  <dataset>/oob_loglik_bootstrap_{i}.csv  per held-out phenotype variant and
                                          model: log-density under its own
                                          phenotype's mixture; plus each
                                          model's mean held-out log-likelihood
                                          over the four standard samples

Seeding: a phenotype cohort can be smaller than the minimum
add_component_to_fit needs to seed a component (RET: 9 MEN2A/B and 7
Hirschsprung variants in total). The seed is then the cohort's in-bag
variants plus their nearest in-bag neighbours (standardized distance over
co-measured readouts), up to --min-seed rows. The seed only sets the new
component's starting point; EM then fits it against every sample.

The new components share the source fit's family (CFUSN, q=2 for the
canonical fits): the fitter has one family per fit, and force_gaussian
would also zero the frozen components' skewness.

Usage
-----
    python hpc/run_grow_phenotype_components.py --gene ret \\
        --source-dir /data/ross/assay_calibration/multivariate/jobs_all_100b_3f_092026 \\
        --source-config 3c_unc --n-bootstraps 20 --num-fits 3 --max-workers 24 \\
        --output-dir /data/ross/assay_calibration/multivariate/jobs_ret_pheno_grow_20b_3f_100726
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import pickle
import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import (
    makeOneHot, pattern_stratified_bootstrap, derive_bootstrap_seed, derive_fit_seed,
    DEFAULT_MASTER_SEED, _weighted_val_ll,
)
from src.assay_calibration.fit_utils.cfusn.fit import add_component_to_fit
from src.assay_calibration.multivariate_analysis.mv_calibration import _sn_logpdf
from hpc.run_grow_component_batch import _load_source_entry

# gene -> (gene-set, dataset name, phenotype groups in growth order with their
# fixed sample roles, mask builder)
PHENOTYPE_GENES = {
    "ret": dict(gene_set="labelseq", dataset="ret_labelseq_mv",
                roles={"MEN2": 4, "Hirschsprung": 5}),
}


def _build(gene):
    spec = PHENOTYPE_GENES[gene]
    if gene == "ret":
        from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
        from mv_analysis.phenotype_evidence import ret_phenotype_masks
        ms = build_labelseq_multiscoresets(genes=["ret"])["ret"]
        return ms, ret_phenotype_masks(ms), spec
    raise ValueError(gene)


def _split(ms, masks, roles, bootstrap_idx, master_seed=DEFAULT_MASTER_SEED):
    """run_grow_component_batch._reconstruct_bootstrap_split, plus one copy
    of each phenotype variant in its phenotype sample, added after the
    resample is drawn (the source fit's resample is drawn from the standard
    samples only, so adding them first would change it). The fitter needs one
    sample per row, so the original row stays in its standard sample (P/LP,
    keeping that sample's composition) and the copy carries the phenotype;
    the copy follows the original into the training or held-out set, with
    the same multiplicity."""
    composition_seed = derive_bootstrap_seed(master_seed, bootstrap_idx)
    onehot_rng = np.random.RandomState(composition_seed)
    sa = makeOneHot(ms.sample_assignments, rng=onehot_rng)
    obs = ms.scores
    include = sa.any(axis=1) & ~np.all(np.isnan(obs), axis=1)
    obs, sa = obs[include], sa[include]
    train_idx, val_idx = pattern_stratified_bootstrap(obs, sa, composition_seed)
    n_roles = max(roles.values()) + 1
    n_cols = max(n_roles, sa.shape[1])
    full = np.zeros((sa.shape[0], n_cols), dtype=sa.dtype)
    full[:, :sa.shape[1]] = sa
    row_ids = np.flatnonzero(include)
    n = len(obs)
    src, copies = [], []
    for name, role in roles.items():
        for j in np.flatnonzero(masks[name][include]):
            row = np.zeros(n_cols, dtype=sa.dtype)
            row[role] = 1
            src.append(j)
            copies.append(row)
    src = np.array(src, dtype=int)
    copy_idx = n + np.arange(len(src))
    obs = np.vstack([obs, obs[src]])
    full = np.vstack([full, np.array(copies).reshape(-1, n_cols)])
    row_ids = np.concatenate([row_ids, row_ids[src]])

    def _with_copies(idx):
        extra = [copy_idx[k] for i in idx for k in np.flatnonzero(src == i)]
        return np.concatenate([np.asarray(idx), np.array(extra, dtype=int)])

    return obs, full, _with_copies(train_idx), _with_copies(val_idx), row_ids, sa.shape[1]


def _as_arrays(fit):
    fit = dict(fit)
    fit["component_params"] = [
        (np.asarray(mu, float), np.asarray(D, float), np.asarray(G, float))
        for mu, D, G in fit["component_params"]]
    fit["weights"] = np.asarray(fit["weights"], float)
    return fit


def _base_with_phenotype_rows(source_fit, train_obs, train_sa, roles):
    from mv_analysis.phenotype_evidence import _posthoc_phenotype_weights
    groups = sorted(roles, key=roles.get)
    X = [train_obs[train_sa[:, roles[g]].astype(bool)] for g in groups]
    if any(len(x) == 0 for x in X):
        return None
    extra = _posthoc_phenotype_weights(source_fit, X)
    w = source_fit["weights"]
    full = np.zeros((train_sa.shape[1], w.shape[1]))
    full[:w.shape[0]] = w
    for g, row in zip(groups, extra):
        full[roles[g]] = row
    out = dict(source_fit)
    out["weights"] = full
    return out


def _mixture_logpdf(x, fit, row):
    params, w = fit["component_params"], fit["weights"][row]
    lp = np.array([_sn_logpdf(x, *p) for p in params])  # (K, n)
    lw = np.log(np.clip(w, 1e-300, None))[:, None]
    m = (lp + lw).max(axis=0)
    return m + np.log(np.exp(lp + lw - m).sum(axis=0))


def _seed_mask(train_obs, train_sa, target_role, phenotype_roles, min_seed):
    """The target cohort's in-bag rows, topped up with their nearest in-bag
    neighbours (excluding other phenotype cohorts) to at least `min_seed`."""
    cohort = train_sa[:, target_role].astype(bool)
    mask = cohort.copy()
    need = min_seed - int(cohort.sum())
    if need <= 0:
        return mask
    sd = np.nanstd(train_obs, axis=0)
    Z = train_obs / np.where(sd > 0, sd, 1.0)
    C = Z[cohort]
    d = np.full(len(Z), np.inf)
    for c in C:
        diff = Z - c
        obs = ~np.isnan(diff)
        n = obs.sum(axis=1)
        dist = np.sqrt(np.nansum(diff ** 2, axis=1) / np.maximum(n, 1))
        dist[n == 0] = np.inf
        d = np.minimum(d, dist)
    other = train_sa[:, [r for r in phenotype_roles if r != target_role]].astype(bool).any(axis=1)
    d[cohort | other] = np.inf
    # bootstrap duplicates share a distance; take distinct candidates in order
    order = np.argsort(d, kind="stable")
    mask[order[:need]] = np.isfinite(d[order[:need]])
    return mask


def _grow(args):
    train_obs, train_sa, val_obs, val_sa, fit, target_role, latent_q, fit_seed, seed_mask = args
    try:
        grown = add_component_to_fit(
            train_obs, train_sa, fit, target_mask=seed_mask,
            constrained=False, multivariate=True, latent_q=latent_q,
            fit_seed=fit_seed, verbose=False, max_em_iters=2000)
    except Exception as e:
        return None, None, str(e)
    val_ll = _weighted_val_ll(val_obs, val_sa, grown["component_params"], grown["weights"],
                              mv=True, fit_kwargs={})
    return _as_arrays(grown), val_ll, None


def _best(results):
    ok = [(g, v, r) for r, (g, v, e) in enumerate(results) if e is None and np.isfinite(v)]
    if not ok:
        return None, None, None
    return max(ok, key=lambda t: t[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene", default="ret", choices=sorted(PHENOTYPE_GENES))
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-config", default="3c_unc")
    ap.add_argument("--n-bootstraps", type=int, default=20)
    ap.add_argument("--num-fits", type=int, default=3)
    ap.add_argument("--latent-q", type=int, default=2)
    ap.add_argument("--max-workers", type=int, default=24)
    ap.add_argument("--min-seed", type=int, default=20,
                    help="minimum seed rows for a new component (cohort plus nearest neighbours)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    ms, masks, spec = _build(args.gene)
    roles = spec["roles"]
    groups = sorted(roles, key=roles.get)
    a, b = groups[0], groups[1]
    src = args.source_config
    keys = {"base": f"{src}_pheno", "A": f"{src}_plus_{a}", "B": f"{src}_plus_{b}",
            "AB": f"{src}_plus_{a}_{b}"}
    out_dir = Path(args.output_dir) / spec["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)

    boots = {}
    for i in range(args.n_bootstraps):
        entry = _load_source_entry(args.source_dir, spec["dataset"], i, src)
        if entry is None:
            print(f"bootstrap {i}: no source fit", flush=True)
            continue
        obs, sa, tr, va, row_ids, n_std = _split(ms, masks, roles, i)
        base = _base_with_phenotype_rows(_as_arrays(entry["fit"]), obs[tr], sa[tr], roles)
        if base is None:
            print(f"bootstrap {i}: a phenotype group has no in-bag variants, skipped", flush=True)
            continue
        boots[i] = dict(entry=entry, obs=obs, sa=sa, tr=tr, va=va, row_ids=row_ids,
                        n_std=n_std, base=base)
    print(f"{len(boots)} bootstraps prepared", flush=True)

    def run_stage(start_key, target_role, k_new):
        tasks, owner = [], []
        for i, d in boots.items():
            start = d["models"][start_key] if start_key != "base" else d["base"]
            K = len(start["component_params"]) + 1
            for r in range(args.num_fits):
                seed = derive_fit_seed(DEFAULT_MASTER_SEED, i, K + 100 * target_role, r)
                tr_obs, tr_sa = d["obs"][d["tr"]], d["sa"][d["tr"]]
                seed_mask = _seed_mask(tr_obs, tr_sa, target_role, list(roles.values()), args.min_seed)
                tasks.append((tr_obs, tr_sa, d["obs"][d["va"]], d["sa"][d["va"]],
                              start, target_role, args.latent_q, seed, seed_mask))
                owner.append(i)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
            res = list(ex.map(_grow, tasks))
        by = {}
        for i, r in zip(owner, res):
            by.setdefault(i, []).append(r)
        for i, rs in by.items():
            g, v, r = _best(rs)
            for e in (x[2] for x in rs if x[2]):
                print(f"  bootstrap {i} {k_new}: restart failed ({e})", flush=True)
            if g is not None:
                boots[i]["models"][k_new] = g
                boots[i].setdefault("fit_idx", {})[k_new] = r

    for d in boots.values():
        d["models"] = {"base": d["base"]}
    run_stage("base", roles[a], "A")
    print("stage +A done", flush=True)
    run_stage("base", roles[b], "B")
    print("stage +B done", flush=True)
    boots_ab = {i: d for i, d in boots.items() if "A" in d["models"]}
    saved = dict(boots)
    boots.clear(); boots.update(boots_ab)
    run_stage("A", roles[b], "AB")
    boots.clear(); boots.update(saved)
    print("stage +A+B done", flush=True)

    for i, d in boots.items():
        va_obs, va_sa = d["obs"][d["va"]], d["sa"][d["va"]]
        rows, out = [], {}
        for m, fit in d["models"].items():
            std_ll = _weighted_val_ll(va_obs, va_sa[:, :d["n_std"]], fit["component_params"],
                                      fit["weights"][:d["n_std"]], mv=True, fit_kwargs={})
            rows.append(dict(bootstrap=i, model=m, kind="standard_mean", group="", row=-1, loglik=std_ll))
            for g in groups:
                sel = va_sa[:, roles[g]].astype(bool)
                if sel.any():
                    ll = _mixture_logpdf(va_obs[sel], fit, roles[g])
                    for rid, v in zip(d["row_ids"][d["va"]][sel], ll):
                        rows.append(dict(bootstrap=i, model=m, kind="phenotype", group=g, row=int(rid), loglik=v))
            out[keys[m]] = {
                "dataset_name": spec["dataset"], "bootstrap_seed": i,
                "num_components": len(fit["component_params"]),
                "fit_idx": d.get("fit_idx", {}).get(m, 0), "fit": fit,
                "val_ll": float(_weighted_val_ll(va_obs, va_sa, fit["component_params"], fit["weights"],
                                                 mv=True, fit_kwargs={})),
                "calibrated_dims": d["entry"].get("calibrated_dims"), "n_cap_hits": 0,
            }
        with open(out_dir / f"bootstrap_{i}_best_fits.pkl", "wb") as f:
            pickle.dump(out, f)
        pd.DataFrame(rows).to_csv(out_dir / f"oob_loglik_bootstrap_{i}.csv", index=False)
    print(f"wrote {len(boots)} bootstraps to {out_dir}\nDone.", flush=True)


if __name__ == "__main__":
    main()
