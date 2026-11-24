"""Predictor models on 2 principal components vs the 3 raw predictors
(cluster test jobs_review_predictor_pca2_test_20b_3f_111226, built by
hpc/prepare_predictor_pca_test.py).

Question: with REVEL, MutPred2 and AlphaMissense nearly redundant, does fitting
the joint model on PC1 + PC2 (dropping the MutPred2-vs-others contrast)
reduce bootstrap variance enough to give more usable evidence, without losing
information or calibration?

Variants scored, all by production_oob's rule (out-of-bag splits, p-th / (100-p)-th
percentile of the log LR+ over the fits, median prior):
  pca2          2-PC fits, 20 bootstraps x 3 restarts, held-out K (4; F9 3)
  raw3_K20      the paper's 3-D fits at the same K, bootstrap seeds 0-19
                (same resamples and restart scheme: the like-for-like control)
  raw3_paper20  the paper's 3-D fits at the paper's config, seeds 0-19
  raw3_paper    the paper's 3-D fits at the paper's config, all bootstraps
                (the paper's own predictor results)
The PCA model's data are rebuilt exactly as the job builder did (conditional-
mean imputation of partially observed rows, then the block rotation) and
checked against the stored transform; its out-of-bag rows are read from the
job files (val_variant_indices) and checked against the stored validation
observations.

Outputs (review/predictor_pca/):
  metrics.csv   per gene x variant x p (5, 10): out-of-bag controls, determinate
                rate, errors, MCC, Cllr (median / percentile / points, capped;
                bits), Cllr_min of the median, median bootstrap spread of the
                log LR+ (95th - 5th percentile) on controls and on all variants;
                also for imputed and complete rows separately
  paired.csv    pca2 - raw3_K20 on the controls out of bag in both: Cllr
                differences with stratified-resample intervals, MCC and
                determinate-rate differences
  insample.csv  in-sample points over all variants: share with evidence,
                agreement of levels between pca2 and raw3_K20
  levels.csv    controls per evidence level, pooled over genes, per variant
  fig_predictor_pca.png

Usage: python -m mv_analysis.review_predictor_pca [--workers 8]
"""
import argparse
import contextlib
import copy
import io
import json
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

PCA_ROOT = Path(os.environ.get("MV_PCA_DIR",
                               "/data/ross/assay_calibration/multivariate/jobs_review_predictor_pca2_test_20b_3f_111226"))
PCA_JSON = PCA_ROOT / "tau_0.1" / "bootstrap_results.json.gz"
OUT = R.out_dir("predictor_pca")
NPZ = OUT / "npz"
VARIANTS = ("pca2", "raw3_K20", "raw3_paper20", "raw3_paper")
PS = (5, 10)
N_BOOT = 500


def _info():
    return json.load(open(PCA_ROOT / "tau_0.1" / "predictor_pca.json"))


def pca_ms(key):
    """The 2-PC MultiScoreset for a predictor model, rebuilt as the job builder
    did and checked against the stored transform."""
    from src.assay_calibration.multivariate_data import redundancy_collapse as rc
    d, meta = R.prep(key)
    info = _info()
    m = info["models"][meta["dataset_name"]]
    ms = config.build_ms_for_gene_cluster("predictor", meta["gene"], cluster_idx=0,
                                          results_json=config.PAPER_RESULTS_JSON)
    assert np.array_equal(np.asarray(ms.scores, float), d["X"], equal_nan=True)
    ms = copy.deepcopy(ms)
    n_imp = rc.impute_block_conditional_mean(ms, info["block"])
    tr = rc.collapse_block(ms, info["block"], info["k"], new_names=m["dims"])
    assert n_imp == m["n_imputed_rows"] and ms.n_variants == m["n_variants"]
    assert np.allclose(tr.mean_, m["mean"]) and np.allclose(np.abs(tr.evecs), np.abs(np.asarray(m["loadings"])))
    return ms, m["components"]


def pca_splits(key, ms):
    """{seed: out-of-bag rows} of the PCA fits, from the job files."""
    _, meta = R.prep(key)
    out = {}
    X = np.asarray(ms.scores, float)
    for f in sorted((PCA_ROOT / "jobs").glob("array_*.pkl")):
        for j in pickle.load(open(f, "rb")):
            if j["dataset_name"] != meta["dataset_name"]:
                continue
            sd = j["shared_data"]
            v = np.asarray(sd["val_variant_indices"], int)
            # the job stores observations scaled per dimension
            vo = np.asarray(sd["val_observations"], float)
            scale = np.nanmedian(vo / X[v], axis=0)
            assert np.allclose(X[v] * scale, vo, equal_nan=True)
            out[int(j["bootstrap_seed"])] = v
    return out


def _score(key, variant):
    """Per-split log LR+ matrix etc. for one variant -> npz/<variant>/<key>.npz."""
    from mv_analysis.comparison_methods import pknn_runs as RP
    from mv_analysis.gene_performance_scatter import fast_results_json
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from mv_analysis.production_oob import _single_threaded
    from src.assay_calibration.multivariate_analysis import mv_calibration as mvc
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    _single_threaded()
    warnings.filterwarnings("ignore")
    f = NPZ / variant / f"{key}.npz"
    if f.exists():
        return key, variant, "exists"
    if variant == "raw3_paper":  # the paper's own matrices
        mv = R.mv_npz(key)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **mv)
        return key, variant, "copied"
    d, meta = R.prep(key)
    if variant == "pca2":
        ms, K = pca_ms(key)
        rj, cfg, seeds = str(PCA_JSON), f"{K}c_unc", range(20)
    else:
        ms = config.build_ms_for_gene_cluster("predictor", meta["gene"], cluster_idx=0,
                                              results_json=config.PAPER_RESULTS_JSON)
        K = _info()["models"][meta["dataset_name"]]["components"] if variant == "raw3_K20" \
            else int(meta["config"].split("c")[0])
        rj, cfg, seeds = config.PAPER_RESULTS_JSON, f"{K}c_unc", range(20)
    an = build_gene_set_analysis(ms, meta["gene"].lower(), rj, dataset_name=meta["dataset_name"],
                                 gene_set="predictor")
    an.configs = [cfg]
    an.raw_boots = {k: v for k, v in an.raw_boots.items() if int(k) in set(seeds) and v.get(cfg)}
    order = [int(k) for k in an.raw_boots]
    valid, orig = [], mvc._bootstrap_job

    def _rec(*a, **kw):
        r = orig(*a, **kw)
        valid.append(r is not None and r[0] is not None)
        return r
    mvc._bootstrap_job = _rec
    try:
        with fast_results_json(rj), contextlib.redirect_stdout(io.StringIO()):
            an.run(partial_pattern_mode="trust_global", n_jobs=1, **DEFAULT_RUN_KWARGS)
    finally:
        mvc._bootstrap_job = orig
    r = an.results[cfg]
    kept = [s for s, v in zip(order, valid) if v]
    L = np.asarray(r["lr_matrix"], float)
    assert len(kept) == L.shape[0] == len(r["priors"])
    full = np.zeros(L.shape[1], int)
    for pv in an.point_values:
        full[np.nanpercentile(L, 5, axis=0) >= r["tau_p_log"][pv - 1]] = pv
    for pv in an.point_values:
        full[np.nanpercentile(L, 95, axis=0) <= r["tau_b_log"][pv - 1]] = -pv
    assert np.array_equal(full, np.asarray(r["points"], int))
    if variant == "pca2":
        splits = pca_splits(key, ms)
        # the paper's split generator on the PCA data must give the job files' splits
        chk = RP._paper_oob_splits(ms, kept[:3], K)
        for b in kept[:3]:
            assert np.array_equal(np.sort(chk[b]), np.sort(splits[b])), f"{key}: split {b} differs"
    else:
        splits = RP._paper_oob_splits(ms, kept, K)
    oob = np.zeros((len(kept), L.shape[1]), bool)
    for i, b in enumerate(kept):
        oob[i, splits[b]] = True
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, L=L, priors=np.asarray(r["priors"], float), seeds=np.asarray(kept), oob=oob,
                        point_values=np.asarray(an.point_values), insample_points=np.asarray(r["points"], int))
    return key, variant, f"{len(kept)} valid splits, K={K}"


def _load(key, variant):
    z = np.load(NPZ / variant / f"{key}.npz")
    return {k: z[k] for k in z.files}


def _oob_table(key, variant, p):
    """Per evaluation control out of bag: label, prior, median / lo / hi log LR+,
    points, bootstrap spread."""
    from mv_analysis.production_oob import _points
    d, _ = R.prep(key)
    mv = _load(key, variant)
    L, oob, priors, pv = mv["L"], mv["oob"].astype(bool), mv["priors"], mv["point_values"]
    rows = np.flatnonzero(d["eval_mask"])
    rows = rows[oob[:, rows].any(0)]
    Lc = np.where(oob[:, rows], L[:, rows], np.nan)
    prior = np.array([np.median(priors[oob[:, j]]) for j in rows])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        med = np.nanmedian(Lc, 0)
        lo, hi = np.nanpercentile(Lc, p, 0), np.nanpercentile(Lc, 100 - p, 0)
        s5, s95 = np.nanpercentile(Lc, 5, 0), np.nanpercentile(Lc, 95, 0)
    pts = np.array([_points(a, b, pr, pv) for a, b, pr in zip(lo, hi, prior)], float)
    e = R.evidence_log_lrs(med, lo, hi, prior, pts)
    return pd.DataFrame(dict(row=rows, y=np.asarray(d["plp"], bool)[rows], prior=prior,
                             imputed=np.isnan(d["X"][rows]).any(1), points=R.floor_level(pts),
                             lr_median=e["median"], lr_percentile=e["percentile"], lr_points=e["points"],
                             spread=s95 - s5, n_oob=oob[:, rows].sum(0)))


def _metrics(t):
    from mv_analysis.review_percentile_choice import _cls
    y = t.y.to_numpy(bool)
    out = dict(n_plp=int(y.sum()), n_blb=int((~y).sum()), **_cls(t.points, y),
               median_spread=float(np.nanmedian(t.spread)))
    for s in ("median", "percentile", "points"):
        out[f"cllr_{s}"] = R.cllr(t[f"lr_{s}"], y) if y.any() and (~y).any() else np.nan
    out["cllr_min"] = R.cllr_min(t.lr_median, y) if y.any() and (~y).any() else np.nan
    return out


def metrics():
    rows = []
    for key, m in R.paper_keys("predictor"):
        for v in VARIANTS:
            mv = _load(key, v)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spread_all = float(np.nanmedian(np.nanpercentile(mv["L"], 95, 0) - np.nanpercentile(mv["L"], 5, 0)))
            for p in PS:
                t = _oob_table(key, v, p)
                for subset, tt in (("all", t), ("complete", t[~t.imputed]), ("imputed", t[t.imputed])):
                    if not len(tt):
                        continue
                    rows.append(dict(key=key, gene=m["gene"], variant=v, percentile=p, rows=subset,
                                     n_fits=int(mv["L"].shape[0]), median_spread_all_variants=spread_all,
                                     **_metrics(tt)))
        print(key, flush=True)
    return pd.DataFrame(rows)


def paired(rng):
    """pca2 - raw3_K20 on the controls out of bag in both."""
    out = []
    for key, m in R.paper_keys("predictor"):
        for p in PS:
            a, b = (_oob_table(key, v, p).set_index("row") for v in ("pca2", "raw3_K20"))
            c = a.index.intersection(b.index)
            a, b = a.loc[c], b.loc[c]
            y = a.y.to_numpy(bool)
            if not y.any() or y.all():
                continue
            ip, ib = np.flatnonzero(y), np.flatnonzero(~y)
            idx = [np.concatenate([rng.choice(ip, len(ip)), rng.choice(ib, len(ib))]) for _ in range(N_BOOT)]
            rec = dict(key=key, gene=m["gene"], percentile=p, n_common=len(c), n_plp=len(ip), n_blb=len(ib),
                       n_imputed=int(a.imputed.sum()))
            for s in ("median", "percentile", "points"):
                d0 = R.cllr(a[f"lr_{s}"], y) - R.cllr(b[f"lr_{s}"], y)
                db = [R.cllr(a[f"lr_{s}"], y, ix) - R.cllr(b[f"lr_{s}"], y, ix) for ix in idx]
                rec |= {f"d_cllr_{s}": d0, f"d_cllr_{s}_lo": np.percentile(db, 2.5),
                        f"d_cllr_{s}_hi": np.percentile(db, 97.5)}
            ma, mb = _metrics(a), _metrics(b)
            rec |= dict(mcc_pca2=ma["mcc"], mcc_raw3=mb["mcc"], det_pca2=ma["det_rate"], det_raw3=mb["det_rate"],
                        errors_pca2=ma["errors"], errors_raw3=mb["errors"],
                        spread_pca2=ma["median_spread"], spread_raw3=mb["median_spread"],
                        same_level=float((a.points.to_numpy() == b.points.to_numpy()).mean()),
                        more_evidence_pca2=int((np.abs(a.points.to_numpy()) > np.abs(b.points.to_numpy())).sum()),
                        less_evidence_pca2=int((np.abs(a.points.to_numpy()) < np.abs(b.points.to_numpy())).sum()))
            out.append(rec)
    return pd.DataFrame(out)


def insample():
    out = []
    for key, m in R.paper_keys("predictor"):
        d, _ = R.prep(key)
        pts = {v: R.floor_level(_load(key, v)["insample_points"]) for v in VARIANTS}
        lab = np.asarray(d["labelled"], bool)
        for v in VARIANTS:
            out.append(dict(key=key, gene=m["gene"], variant=v, n=len(pts[v]),
                            share_evidence_unlabelled=float((pts[v][~lab] != 0).mean()),
                            share_path_unlabelled=float((pts[v][~lab] > 0).mean()),
                            share_benign_unlabelled=float((pts[v][~lab] < 0).mean()),
                            same_level_as_raw3_K20=float((pts[v] == pts["raw3_K20"]).mean()),
                            same_level_as_raw3_paper=float((pts[v] == pts["raw3_paper"]).mean())))
    return pd.DataFrame(out)


def levels():
    out = []
    for v in VARIANTS:
        for p in PS:
            t = pd.concat([_oob_table(k, v, p) for k, _ in R.paper_keys("predictor")])
            for lv in (-8, -4, -2, -1, 0, 1, 2, 4, 8):
                x = t[t.points == lv]
                out.append(dict(variant=v, percentile=p, level=lv, n_plp=int(x.y.sum()), n_blb=int((~x.y).sum())))
    return pd.DataFrame(out)


def figure(mt, pr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = mt[(mt.rows == "all")]
    genes = sorted(x.gene.unique())
    cols = {"pca2": "C3", "raw3_K20": "C0", "raw3_paper20": "C9", "raw3_paper": "0.4"}
    panels = [("cllr_median", "Cllr median (bits)"), ("cllr_percentile", "Cllr percentile (bits)"),
              ("cllr_points", "Cllr points (bits)"), ("mcc", "MCC"), ("det_rate", "determinate rate"),
              ("median_spread", "median 5-95 spread of log LR+")]
    fig, axes = plt.subplots(len(PS), len(panels), figsize=(3.0 * len(panels), 3.0 * len(PS)), squeeze=False)
    for i, p in enumerate(PS):
        for j, (c, lab) in enumerate(panels):
            ax = axes[i, j]
            for k, v in enumerate(VARIANTS):
                y = x[(x.variant == v) & (x.percentile == p)].set_index("gene").reindex(genes)[c]
                ax.scatter(np.arange(len(genes)) + (k - 1.5) * 0.15, y, s=12, color=cols[v], label=v)
            ax.set_xticks(range(len(genes)))
            ax.set_xticklabels(genes, rotation=90, fontsize=6)
            ax.set_title(f"{lab}, p = {p}", fontsize=7)
            ax.tick_params(labelsize=6)
            if i == 0 and j == 0:
                ax.legend(fontsize=5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_predictor_pca.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    keys = [k for k, _ in R.paper_keys("predictor")]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_score, *zip(*[(k, v) for k in keys for v in VARIANTS])):
            print(*r, flush=True)
    mt = metrics()
    mt.to_csv(OUT / "metrics.csv", index=False)
    pr = paired(np.random.default_rng(0))
    pr.to_csv(OUT / "paired.csv", index=False)
    ins = insample()
    ins.to_csv(OUT / "insample.csv", index=False)
    lv = levels()
    lv.to_csv(OUT / "levels.csv", index=False)
    figure(mt, pr)
    pd.set_option("display.width", 250)
    cols = ["gene", "variant", "percentile", "n_plp", "n_blb", "det_rate", "errors", "mcc", "cllr_median",
            "cllr_percentile", "cllr_points", "cllr_min", "median_spread"]
    print(mt[mt.rows == "all"][cols].round(3).to_string(index=False))
    print(pr.round(3).to_string(index=False))
    print(ins.round(3).to_string(index=False))
    print(lv.pivot_table(index=["variant", "percentile"], columns="level", values=["n_plp", "n_blb"]).to_string())


if __name__ == "__main__":
    main()
