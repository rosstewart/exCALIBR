#!/usr/bin/env python3
"""Out-of-bag evaluation under PRODUCTION scoring, on the paper's own
bootstrap splits, for the joint model (MV), P-KNN and supervised classifier
baselines.

Splits: each joint fit's bootstrap sample is regenerated with the paper's job
generator and seed (Fit.generate_fit_jobs, master seed 0); recomputing a
stored fit's validation log-likelihood on its regenerated split reproduces the
stored val_ll exactly. A variant is out of bag (OOB) for a split when that
fit never saw it.

Scoring (identical for every method): each variant collects log LR+ values
only from splits where it is OOB; pathogenic evidence when the 5th percentile
reaches the pathogenic thresholds, benign when the 95th percentile reaches the
benign thresholds (the production rule). The thresholds are set by the prior
estimated from the TRAINING data only: the median of the joint model's
per-split prior estimates over the variant's OOB splits (P-KNN and the
classifiers cannot estimate a prior; their likelihood ratios do not depend on
one).
  MV: the canonical config's per-split log LR+ (MVCalibrationAnalysis with the
      production run settings).
  P-KNN: the per-gene TP53 protocol (N_calibration 10, regularization = the
      gene's labelled variants) calibrated on each split's in-bag labelled
      variants; its 100 internal bootstrap posteriors per split are converted
      to log LR+ and pooled over the variant's OOB splits.
  Classifiers (comparison_methods/supervised.py): trained on each split's
      in-bag labelled variants; one log LR+ per OOB split.
Metrics: the paper's evaluation rows (report._eval_labels) at the
evidence-direction threshold. Variants in every bootstrap sample (never OOB)
are excluded for all methods.

Driven by the analysis notebook's "production-oob" section (run_all);
standalone: nice -n 19 python -m mv_analysis.production_oob {mv,splits,collect,all}
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import contextlib
import io
import json
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mv_analysis import config  # noqa: E402
from mv_analysis.comparison_methods import pknn_runs as RP  # noqa: E402

OUT = Path(config.PKNN_DIR) / "production_oob"
N_CAL = 10


def _single_threaded():
    """Pool workers forked from a process that already loaded NumPy inherit
    its full BLAS/OpenMP thread pools (128 threads each); cap them at one."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def keys_with_both_classes(keys=None):
    out = []
    for p in sorted((Path(config.PKNN_DIR) / "prep").glob("*.json")):
        m = json.load(open(p))
        if (keys is None or m["key"] in keys) and m["eval_P"] > 0 and m["eval_B"] > 0:
            out.append(m["key"])
    return out


def _mv_one(key):
    """MV per-split log LR+, per-split priors and OOB masks -> <key>/mv.npz."""
    _single_threaded()
    from mv_analysis.gene_performance_scatter import fast_results_json
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from src.assay_calibration.multivariate_analysis import mv_calibration as mvc
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    warnings.filterwarnings("ignore")
    f = OUT / key / "mv.npz"
    if f.exists():
        return key, "exists"
    d, meta = RP.load_prep(key)
    gs, gene, cl, cfg = meta["gene_set"], meta["gene"], meta["cluster_idx"], meta["config"]
    rj = config.PAPER_RESULTS_JSON
    ms = config.build_ms_for_gene_cluster(gs, gene, cluster_idx=cl, results_json=rj)
    assert np.array_equal(np.asarray(ms.scores, float), d["X"], equal_nan=True)
    an = build_gene_set_analysis(ms, gene.lower(), rj, dataset_name=meta["dataset_name"], gene_set=gs)
    an.configs = [cfg]
    seeds = [int(k) for k in an.raw_boots]
    # run() silently drops invalid bootstrap fits; record which (n_jobs=1 keeps
    # the worker in-process so the recorder sees every call, in order)
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
    kept = [s for s, v in zip(seeds, valid) if v]
    L = np.asarray(r["lr_matrix"], float)
    assert len(kept) == L.shape[0] == len(r["priors"])
    # the production rule on the full matrix must reproduce the stored points
    full = np.zeros(L.shape[1], int)
    for pv in an.point_values:
        full[np.nanpercentile(L, 5, axis=0) >= r["tau_p_log"][pv - 1]] = pv
    for pv in an.point_values:
        full[np.nanpercentile(L, 95, axis=0) <= r["tau_b_log"][pv - 1]] = -pv
    assert np.array_equal(full, np.asarray(r["points"], int)), "log LR+ matrix does not reproduce production points"
    K = int(cfg.split("c")[0])
    splits = RP._paper_oob_splits(ms, kept, K)
    oob = np.zeros((len(kept), L.shape[1]), bool)
    for i, b in enumerate(kept):
        oob[i, splits[b]] = True
    (OUT / key).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, L=L, priors=np.asarray(r["priors"], float), seeds=np.asarray(kept), oob=oob,
                        point_values=np.asarray(an.point_values), insample_points=np.asarray(r["points"], int))
    return key, f"{len(kept)} valid splits"


def _split_one(key, i, inner):
    """P-KNN posteriors and classifier log LR+ for split i's OOB evaluation
    rows, trained on its in-bag labelled rows -> <key>/split<i>.npz."""
    _single_threaded()
    from mv_analysis.comparison_methods import pknn, supervised
    warnings.filterwarnings("ignore")
    f = OUT / key / f"split{i}.npz"
    if f.exists():
        return key, i
    d, meta = RP.load_prep(key)
    mv = np.load(OUT / key / "mv.npz")
    oob = mv["oob"][i]
    X, lab, ev = d["X"], d["labelled"], d["eval_mask"]
    y = d["plp"].astype(int)
    tr, te = lab & ~oob, ev & oob
    rows = np.flatnonzero(te)
    out = {"rows": rows, "prior": float(mv["priors"][i])}
    if len(rows) and y[tr].sum() >= 2 and (y[tr] == 0).sum() >= 2:
        post, _ = pknn.bootstrap_posteriors(X[tr], y[tr], X[te], X[lab], out["prior"], N_CAL, n_jobs=inner)
        out["pknn_post"] = post.astype(np.float32)
    for m in supervised.MODELS:
        v = supervised.fit_predict(m, X[tr], y[tr], X[te], seed=int(mv["seeds"][i])) if len(rows) else None
        if v is not None:
            out[f"sup_{m}"] = v.astype(np.float32)
    np.savez_compressed(f, **out)
    return key, i


@lru_cache(maxsize=None)
def _log_c(prior_rounded):
    from src.assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant
    return float(np.log(get_tavtigian_constant(prior_rounded)))


def _points(lo, hi, prior, point_values):
    """Production rule: lo (5th percentile log LR+) against the pathogenic
    thresholds, hi (95th percentile) against the benign ones, at `prior`."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    lc = _log_c(round(float(prior), 4))
    pts = 0
    for pv in point_values:
        if lo >= pv / 8 * lc:
            pts = pv
    for pv in point_values:
        if hi <= -pv / 8 * lc:
            pts = -pv
    return pts


def _collect_one(key):
    from mv_analysis.comparison_methods import supervised
    d, meta = RP.load_prep(key)
    mv = np.load(OUT / key / "mv.npz")
    L, oob, priors, pvals = mv["L"], mv["oob"], mv["priors"], mv["point_values"]
    ev = d["eval_mask"]
    n = oob.shape[0]
    pk = {j: [] for j in np.flatnonzero(ev)}
    sup = {m: {j: [] for j in np.flatnonzero(ev)} for m in supervised.MODELS}
    for i in range(n):
        s = np.load(OUT / key / f"split{i}.npz")
        rows = s["rows"]
        if "pknn_post" in s.files:
            p = np.clip(s["pknn_post"].astype(float), 1e-12, 1 - 1e-12)
            pr = float(s["prior"])
            llr = np.log(p / (1 - p)) - np.log(pr / (1 - pr))  # prior-free: P-KNN weights benign counts by the prior
            for j, row in zip(rows, llr):
                pk[j].append(row)
        for m in supervised.MODELS:
            if f"sup_{m}" in s.files:
                for j, v in zip(rows, s[f"sup_{m}"]):
                    sup[m][j].append(float(v))
    recs = []
    for j in np.flatnonzero(ev):
        o = oob[:, j]
        if not o.any():
            recs.append(dict(row=j, n_oob=0))
            continue
        prior = float(np.median(priors[o]))
        r = dict(row=j, n_oob=int(o.sum()), oob_prior=prior)
        lv = L[o, j]
        r["MV"] = _points(np.nanpercentile(lv, 5), np.nanpercentile(lv, 95), prior, pvals)
        if pk[j]:
            # P-KNN returns NaN for internal bootstraps whose resample lacks a
            # class (undefined mutual-information weights); those are ignored
            v = np.concatenate(pk[j])
            r["P-KNN"] = _points(np.nanpercentile(v, 5), np.nanpercentile(v, 95), prior, pvals)
            r["pknn_nan_frac"] = float(np.isnan(v).mean())
        for m in supervised.MODELS:
            if sup[m][j]:
                v = np.asarray(sup[m][j])
                r[m] = _points(np.percentile(v, 5), np.percentile(v, 95), prior, pvals)
        recs.append(r)
    out = pd.DataFrame(recs)
    out.insert(0, "variant", d["variants"][out["row"]])
    out.insert(1, "label", d["plp"][out["row"]].astype(int))
    out["mv_insample"] = mv["insample_points"][out["row"]]
    out.to_csv(OUT / f"{key}.csv", index=False)
    return key


def _pool(fn, tasks, workers, label):
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *t): t for t in tasks}
        for k, f in enumerate(as_completed(futs)):
            try:
                res = f.result()
                if label == "mv" or k % 200 == 0:
                    print(f"  [{label}] {k + 1}/{len(tasks)} {res}", flush=True)
            except Exception:
                print(f"  [{label}] {futs[f]} FAILED\n{traceback.format_exc()}", flush=True)


def run_all(workers=12, inner=2, keys=None):
    """MV, P-KNN and classifier out-of-bag evaluation for every dataset with
    both classes, skipping finished outputs; returns the per-variant tables'
    directory."""
    OUT.mkdir(parents=True, exist_ok=True)
    keys = keys_with_both_classes(keys)
    _pool(_mv_one, [(k,) for k in keys if not (OUT / k / "mv.npz").exists()], max(1, workers // 2), "mv")
    tasks = [(k, i, inner) for k in keys if (OUT / k / "mv.npz").exists()
             for i in range(np.load(OUT / k / "mv.npz")["oob"].shape[0]) if not (OUT / k / f"split{i}.npz").exists()]
    _pool(_split_one, tasks, workers, "splits")
    for k in keys:
        if (OUT / k / "mv.npz").exists():
            _collect_one(k)
    return OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["mv", "splits", "collect", "all"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--keys", nargs="*")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = keys_with_both_classes(args.keys)
    if args.stage in ("mv", "all"):
        _pool(_mv_one, [(k,) for k in keys if not (OUT / k / "mv.npz").exists()], args.workers, "mv")
    if args.stage in ("splits", "all"):
        tasks = [(k, i, args.inner) for k in keys if (OUT / k / "mv.npz").exists()
                 for i in range(np.load(OUT / k / "mv.npz")["oob"].shape[0])
                 if not (OUT / k / f"split{i}.npz").exists()]
        print(f"splits: {len(tasks)} tasks", flush=True)
        _pool(_split_one, tasks, args.workers, "splits")
    if args.stage in ("collect", "all"):
        for k in keys:
            try:
                _collect_one(k)
            except Exception:
                print(f"  [collect] {k} FAILED\n{traceback.format_exc()}", flush=True)
        print(f"collected {len(keys)} datasets -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
