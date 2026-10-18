#!/usr/bin/env python
"""TEST (not production): joint fits with a frozen "functionally normal"
anchor, and/or an all-assayed population in place of gnomAD for shaping the
components.

Two stages per bootstrap:

  1. Anchor (frozen in stage 2)
       none   no anchor; a standard joint fit with 1 + K_free components
       A      1 component fitted to B/LB + synonymous
       B<t>   2 components fitted jointly to B/LB, synonymous and gnomAD
              (shared components, per-sample weights); every component with
              weight >= t in B/LB or synonymous is kept as the anchor
              (the larger one if none passes)
  2. Joint fit of K_free free components on top of the frozen anchor, over
     every sample, with standard (unweighted) EM. Population sample:
       gnomAD     gnomAD in the fit (as today)
       allassayed a random subsample (2x the gnomAD count) of unlabelled
                  assayed missense SNVs in the fit instead of gnomAD; gnomAD's
                  weights (which set the prior) are then estimated on the
                  frozen fitted components

Phenotype groups (RET: MEN2A/B, Hirschsprung) enter the fit as their own
samples: one copy of each curated variant is added in its phenotype sample
while the original stays in P/LP (the fitter needs one sample per row).

Restarts are chosen by held-out log-likelihood. Writes, under --output-dir:
  <dataset>/bootstrap_{i}_best_fits.pkl  one config key per model, weights
                                         rows P/LP, B/LB, gnomAD, synonymous
                                         (+ phenotype rows), ready for
                                         hpc/aggregate_results.py
  <dataset>/diag_bootstrap_{i}.csv       held-out log-likelihood of the labelled
                                         samples (identical held-out variants
                                         across models), anchor size, and the
                                         P/LP weight on the anchor
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
    makeOneHot, pattern_stratified_bootstrap, derive_bootstrap_seed, DEFAULT_MASTER_SEED, _weighted_val_ll,
)
from src.assay_calibration.fit_utils.cfusn.fit import single_fit
from src.assay_calibration.fit_utils.cfusn.initializations import kmeans_init_mv
from src.assay_calibration.fit_utils.cfusn.update_steps import get_sample_weights
from src.assay_calibration.multivariate_analysis.mv_calibration import _sn_logpdf

P, B, G, S = 0, 1, 2, 3
LATENT_Q = 2
MAX_EM_ITERS = 2000
POOL_MULT = 2.0  # pool subsample size as a multiple of the gnomAD count; <= 0: whole pool
POOL_CAP = None  # optional upper bound on the pool subsample size

GENES = {
    "ret": dict(gene_set="labelseq", pool=True, phenotypes=True),
    "GCK": dict(gene_set="integrated", pool=True, phenotypes=False),
    "BRCA1": dict(gene_set="integrated", pool=True, phenotypes=False),
    "BRCA1_predictor": dict(gene_set="predictor", pool=False, phenotypes=False),
}


# -- data -------------------------------------------------------------------
def load_gene(key, pool_type="all_missense_snv"):
    from mv_analysis import config
    spec = GENES.get(key) or dict(
        gene_set="labelseq" if key.lower() in config.LABELSEQ_GENES else "integrated",
        pool=True, phenotypes=key.lower() == "ret")
    gene = key.split("_")[0]
    gs = spec["gene_set"]
    ms = config.build_ms_for_gene_cluster(gs, gene)
    dataset = config.canonical_dataset_name(gene, gs, results_json=config.PAPER_RESULTS_JSON)
    pool = None
    if spec["pool"]:
        mm = config.build_multiscoresets_for_gene_set(gs, genes=[gene], regularization_type=pool_type)
        rms = mm.get(gene.lower()) or mm.get(gene.upper()) or next(iter(mm.values()))
        # the pool keeps the fit's own readouts, matched by name (a gene's
        # stored fit may use a subset of the readouts the pool builder returns)
        names = list(rms.dataset_names)
        missing = [n for n in ms.dataset_names if n not in names]
        if missing:
            raise ValueError(f"{key}: pool lacks the fit's readouts {missing}")
        cols = [names.index(n) for n in ms.dataset_names]
        s = rms.sample_assignments.astype(bool)
        # every assayed variant of the pool type except labelled controls
        # (P/LP, B/LB, synonymous: already in the fit as their own samples);
        # gnomAD variants stay in, so the pool is not depleted of them
        labelled = [c for c in (P, B, S) if c < s.shape[1] - 1]
        only = s[:, -1] & ~s[:, labelled].any(axis=1)
        pool = np.asarray(rms.scores, float)[only][:, cols]
        pool = pool[~np.isnan(pool).all(axis=1)]
    roles, masks = {}, {}
    if spec["phenotypes"]:
        from mv_analysis.phenotype_evidence import ret_phenotype_masks, RET_PHENOTYPE_ROLES
        masks, roles = ret_phenotype_masks(ms), dict(RET_PHENOTYPE_ROLES)
    return dict(key=key, gene=gene, gene_set=gs, ms=ms, dataset=dataset, pool=pool, roles=roles, masks=masks)


def bootstrap_data(d, i, master_seed=DEFAULT_MASTER_SEED):
    """Labelled rows: the same one-hot resolution and pattern-stratified
    resample the production fits use. Phenotype copies follow their original
    into the training or held-out set. Pool rows: a fixed-size random
    subsample, resampled with replacement for training, the rest held out."""
    ms = d["ms"]
    seed = derive_bootstrap_seed(master_seed, i)
    sa = makeOneHot(ms.sample_assignments, rng=np.random.RandomState(seed))
    obs = np.asarray(ms.scores, float)
    keep = sa.any(axis=1) & ~np.isnan(obs).all(axis=1)
    obs, sa = obs[keep], sa[keep].astype(bool)
    tr, va = pattern_stratified_bootstrap(obs, sa, seed)
    n_std = sa.shape[1]
    roles = d["roles"]
    cols = n_std + len(roles)
    full = np.zeros((len(obs), cols), dtype=bool)
    full[:, :n_std] = sa
    src, prow = [], []
    for name, role in roles.items():
        for j in np.flatnonzero(d["masks"][name][keep]):
            r = np.zeros(cols, dtype=bool)
            r[role] = True
            src.append(j)
            prow.append(r)
    src = np.array(src, dtype=int)
    n = len(obs)
    if len(src):
        copy_idx = n + np.arange(len(src))
        obs = np.vstack([obs, obs[src]])
        full = np.vstack([full, np.array(prow)])

        def add(idx):
            extra = [copy_idx[k] for t in idx for k in np.flatnonzero(src == t)]
            return np.concatenate([np.asarray(idx), np.array(extra, dtype=int)])
        tr, va = add(tr), add(va)
    out = dict(obs=obs, sa=full, tr=np.asarray(tr), va=np.asarray(va), n_std=n_std)
    if d["pool"] is not None:
        rng = np.random.RandomState(seed + 7919)
        n_g = int(sa[:, G].sum())
        size = len(d["pool"]) if POOL_MULT <= 0 else min(int(POOL_MULT * n_g), len(d["pool"]))
        if POOL_CAP:
            size = min(size, POOL_CAP)
        sub = d["pool"][rng.choice(len(d["pool"]), size, replace=False)]
        ptr = rng.randint(0, size, size)
        pva = np.setdiff1d(np.arange(size), ptr)
        out.update(pool_tr=sub[ptr], pool_va=sub[pva])
    return out


def fit_rows(bd, pop, split):
    """(obs, indicators, column -> output weight-row map) for one model's fit:
    every labelled/phenotype sample, with the population sample being gnomAD
    or the all-assayed subsample. Empty columns are dropped (their weight
    rows are estimated afterwards on the frozen components)."""
    idx = bd["tr"] if split == "tr" else bd["va"]
    obs, sa = bd["obs"][idx], bd["sa"][idx]
    cols = list(range(sa.shape[1]))
    if pop == "allassayed":
        rows = ~sa[:, G]
        obs, sa = obs[rows], sa[rows]
        cols.remove(G)
        sa = sa[:, cols]
        p = bd["pool_tr"] if split == "tr" else bd["pool_va"]
        obs = np.vstack([obs, p])
        pool_col = np.zeros((len(p), sa.shape[1] + 1), dtype=bool)
        pool_col[:, -1] = True
        sa = np.vstack([np.hstack([sa, np.zeros((len(sa), 1), dtype=bool)]), pool_col])
        cols = cols + [-1]  # -1: pool, not an output row
    nonempty = sa.any(axis=0)
    return obs, sa[:, nonempty], [c for c, k in zip(cols, nonempty) if k]


# -- fitting helpers ---------------------------------------------------------
def _arr(params):
    return [(np.asarray(m, float), np.asarray(D, float), np.asarray(Gm, float)) for m, D, Gm in params]


def mix_logpdf(X, params, w):
    lp = np.array([_sn_logpdf(X, *p) for p in params])
    lw = np.log(np.clip(np.asarray(w, float), 1e-300, None))[:, None]
    a = lp + lw
    m = a.max(axis=0)
    return m + np.log(np.exp(a - m).sum(axis=0))


def posthoc_weights(params, X, n_iter=500, tol=1e-10):
    K = len(params)
    w = np.full((1, K), 1.0 / K)
    ind = np.ones((len(X), 1), dtype=bool)
    for _ in range(n_iter):
        w_new = get_sample_weights(X, ind, params, w, multivariate=True)
        if np.abs(w_new - w).max() < tol:
            return w_new[0]
        w = w_new
    return w[0]


def _fit(obs, sa, K, seed, initial_params=None, initial_weights=None, frozen=None):
    kw = dict(latent_q=LATENT_Q, fit_seed=seed, verbose=False, max_em_iters=MAX_EM_ITERS,
              rng=np.random.RandomState(seed))
    if initial_params is not None:
        kw.update(initial_params=initial_params, initial_weights=initial_weights,
                  frozen_components=frozen or set())
        return single_fit(obs, sa, K, constrained=False, init_method=None,
                          init_constraint_adjustment=None, multivariate=True, **kw)
    return single_fit(obs, sa, K, constrained=False, init_method="kmeans",
                      init_constraint_adjustment=None, multivariate=True, **kw)


def fit_anchors(args):
    """Stage 1 for one (gene, bootstrap): anchor A and the 2-component fit
    anchor B selects from (best of 3 restarts by held-out likelihood)."""
    key, i, bd, n_restarts = args
    out = {}
    try:
        def benign(split):
            idx = bd["tr"] if split == "tr" else bd["va"]
            sa = bd["sa"][idx]
            cols = [B] + ([S] if bd["n_std"] > S else [])
            rows = sa[:, cols].any(axis=1)
            return bd["obs"][idx][rows]
        Xb, Xbv = benign("tr"), benign("va")
        one = np.ones((len(Xb), 1), dtype=bool)
        fa = _fit(Xb, one, 1, 1000 + i)
        out["A"] = _arr(fa["component_params"])

        def three(split):
            idx = bd["tr"] if split == "tr" else bd["va"]
            sa = bd["sa"][idx]
            cols = [c for c in (B, S, G) if c < bd["n_std"]]
            rows = sa[:, cols].any(axis=1)
            s = sa[rows][:, cols]
            nonempty = s.any(axis=0)
            return bd["obs"][idx][rows], s[:, nonempty], [c for c, k in zip(cols, nonempty) if k]
        X3, s3, c3 = three("tr")
        X3v, s3v, _ = three("va")
        best, best_ll = None, -np.inf
        for r in range(n_restarts):
            try:
                f = _fit(X3, s3, 2, 2000 + 10 * i + r)
            except Exception:
                continue
            ll = _weighted_val_ll(X3v, s3v, f["component_params"], f["weights"], mv=True, fit_kwargs={})
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = f, ll
        if best is not None:
            W = np.asarray(best["weights"], float)
            benign_rows = [k for k, c in enumerate(c3) if c in (B, S)]
            out["B2"] = (_arr(best["component_params"]), W[benign_rows].max(axis=0))
    except Exception as e:
        out["error"] = repr(e)
    return key, i, out


def anchor_for(name, anchors):
    if name == "none":
        return []
    if name == "A":
        return anchors.get("A") or None
    if "B2" not in anchors:
        return None
    t = float(name[1:])
    params, bw = anchors["B2"]
    keep = [k for k in range(len(params)) if bw[k] >= t] or [int(np.argmax(bw))]
    return [params[k] for k in keep]


def stage2(args):
    """Stage 2 for one (gene, bootstrap, model, restart)."""
    key, i, bd, model, anchor, k_free, pop, seed = args
    try:
        obs, sa, cols = fit_rows(bd, pop, "tr")
        vobs, vsa, vcols = fit_rows(bd, pop, "va")
        n_a = len(anchor)
        if n_a == 0:
            f = _fit(obs, sa, 1 + k_free, seed)
        else:
            dens = mix_logpdf(obs, anchor, np.full(n_a, 1.0 / n_a))
            worst = np.argsort(dens)[: max(30, int(0.4 * len(obs)))]
            free, _ = kmeans_init_mv(obs[worst], n_clusters=k_free, constrained=False,
                                     latent_q=LATENT_Q, rng=np.random.RandomState(seed))
            init = list(anchor) + _arr(free)
            W0 = np.zeros((sa.shape[1], n_a + k_free))
            for r, c in enumerate(cols):
                a_share = 0.8 if c in (B, S) else 0.5
                W0[r, :n_a] = a_share / n_a
                W0[r, n_a:] = (1 - a_share) / k_free
            f = _fit(obs, sa, n_a + k_free, seed, initial_params=init, initial_weights=W0,
                     frozen=set(range(n_a)))
        params = _arr(f["component_params"])
        W = np.asarray(f["weights"], float)
        # held-out likelihood on the fit's own samples (restart selection)
        keepv = [vcols.index(c) for c in cols if c in vcols]
        vrow = [cols.index(c) for c in vcols if c in cols]
        val_ll = _weighted_val_ll(vobs, vsa[:, keepv], params, W[vrow], mv=True, fit_kwargs={}) \
            if len(vrow) else np.nan
        return key, i, model, seed, params, W, cols, val_ll, None
    except Exception as e:
        return key, i, model, seed, None, None, None, np.nan, repr(e)


def output_weights(params, W, cols, bd):
    """Weight rows in the scoring layout (P/LP, B/LB, gnomAD, synonymous,
    phenotypes): fitted rows where the sample was in the fit, otherwise
    estimated on the frozen components from in-bag variants (gnomAD under the
    all-assayed population)."""
    n_out = bd["sa"].shape[1]
    out = np.zeros((n_out, len(params)))
    tr_obs, tr_sa = bd["obs"][bd["tr"]], bd["sa"][bd["tr"]]
    for c in range(n_out):
        if c in cols:
            out[c] = W[cols.index(c)]
        else:
            X = tr_obs[tr_sa[:, c]]
            out[c] = posthoc_weights(params, X) if len(X) else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", nargs="+", default=list(GENES))
    ap.add_argument("--n-bootstraps", type=int, default=20)
    ap.add_argument("--num-fits", type=int, default=3)
    ap.add_argument("--k-free", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--anchors", nargs="+", default=["none", "A", "B0.1", "B0.2"])
    ap.add_argument("--max-workers", type=int, default=100)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pool-type", default="all_missense_snv",
                    choices=["all_missense_snv", "all_assayed"])
    ap.add_argument("--pool-mult", type=float, default=2.0,
                    help="pool subsample size as a multiple of the gnomAD count; 0 = whole pool")
    ap.add_argument("--pops", nargs="+", default=["gnomAD", "allassayed"])
    args = ap.parse_args()
    global POOL_MULT
    POOL_MULT = args.pool_mult

    data = {k: load_gene(k, args.pool_type) for k in args.genes}
    boots = {(k, i): bootstrap_data(d, i) for k, d in data.items() for i in range(args.n_bootstraps)}
    for k, d in data.items():
        print(f"{k}: dataset {d['dataset']}, dims {list(d['ms'].dataset_names)}, "
              f"pool {None if d['pool'] is None else len(d['pool'])}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        anchors = {}
        for k, i, out in ex.map(fit_anchors, [(k, i, bd, args.num_fits) for (k, i), bd in boots.items()]):
            if "error" in out:
                print(f"  {k} bootstrap {i}: anchor fit failed: {out['error']}", flush=True)
            anchors[(k, i)] = out
        print("stage 1 (anchors) done", flush=True)

        tasks, aliases = [], {}
        for (k, i), bd in boots.items():
            pops = [p for p in args.pops if p == "gnomAD" or data[k]["pool"] is not None]
            chosen = {}
            for a in args.anchors:
                anc = anchor_for(a, anchors[(k, i)])
                if anc is None:
                    continue
                # thresholds that pick the same anchor share one set of fits
                sig = tuple(id(x) for x in anc) if a.startswith("B") else (a,)
                if sig in chosen:
                    aliases[(k, i, a)] = chosen[sig]
                    continue
                chosen[sig] = a
                for kf in args.k_free:
                    for pop in pops:
                        model = f"{a}_{pop}_k{kf}"
                        for r in range(args.num_fits):
                            tasks.append((k, i, bd, model, anc, kf, pop, 10007 * i + 101 * r + kf))
        print(f"stage 2: {len(tasks)} fits", flush=True)
        results = list(ex.map(stage2, tasks, chunksize=1))

    best = {}
    for k, i, model, seed, params, W, cols, val_ll, err in results:
        for (ak, ai, alias), src in aliases.items():
            if ak == k and ai == i and model.startswith(src + "_"):
                if not err:
                    m2 = alias + model[len(src):]
                    cur = best.get((k, i, m2))
                    if np.isfinite(val_ll) and (cur is None or val_ll > cur[3]):
                        best[(k, i, m2)] = (params, W, cols, val_ll)
        if err:
            print(f"  {k} b{i} {model}: failed {err}", flush=True)
            continue
        cur = best.get((k, i, model))
        if np.isfinite(val_ll) and (cur is None or val_ll > cur[3]):
            best[(k, i, model)] = (params, W, cols, val_ll)

    for (k, i), bd in boots.items():
        d = data[k]
        out_dir = Path(args.output_dir) / d["dataset"]
        out_dir.mkdir(parents=True, exist_ok=True)
        entries, diag = {}, []
        va_obs, va_sa = bd["obs"][bd["va"]], bd["sa"][bd["va"]]
        for (kk, ii, model), (params, W, cols, val_ll) in best.items():
            if kk != k or ii != i:
                continue
            Wout = output_weights(params, W, cols, bd)
            anchor_name = model.split("_")[0]
            n_a = 0 if anchor_name == "none" else len(anchor_for(anchor_name, anchors[(k, i)]))
            row = dict(gene=k, bootstrap=i, model=model, n_anchor=n_a, val_ll=val_ll,
                       plp_weight_on_anchor=float(Wout[P, :n_a].sum()) if n_a else np.nan)
            for name, c in (("plp", P), ("blb", B)):
                sel = va_sa[:, c]
                if bd["sa"].shape[1] > 4:  # P/LP originals only (phenotype copies are separate rows)
                    sel = sel & ~va_sa[:, 4:].any(axis=1)
                row[f"heldout_ll_{name}"] = float(mix_logpdf(va_obs[sel], params, Wout[c]).mean()) \
                    if sel.any() else np.nan
            diag.append(row)
            entries[model] = {
                "dataset_name": d["dataset"], "bootstrap_seed": i, "num_components": len(params),
                "fit_idx": 0, "fit": {"component_params": params, "weights": Wout},
                "val_ll": float(val_ll), "assay_datasets": list(d["ms"].dataset_names),
                "calibrated_dims": None, "n_cap_hits": 0,
            }
        with open(out_dir / f"bootstrap_{i}_best_fits.pkl", "wb") as f:
            pickle.dump(entries, f)
        pd.DataFrame(diag).to_csv(out_dir / f"diag_bootstrap_{i}.csv", index=False)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
