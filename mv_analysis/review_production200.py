"""Preliminary comparison of the review production run (200 bootstraps x greedy
restarts; hpc/review_jobs.sh) with the paper's fits (100 bootstraps x 3 restarts).

Both runs draw bootstrap seed s the same way, so seeds 0-99 are the same
resamples and validation splits in both. That separates the two budget changes:
  restarts    paper (100 seeds, 3 restarts) vs new run seeds 0-99 (greedy)
  bootstraps  new run seeds 0-99 vs all of its seeds (up to 200)

Per paper model present in the new results (canonical config of the paper):
  restarts.csv     paired held-out log-likelihood per bootstrap (new - paper),
                   per variant of the validation set where the fits record it;
                   share of bootstraps the best restart is a greedy candidate
                   (fit_idx >= 3: 0 Gaussian, 1 MSN seed, 2 CFUSN seed)
  performance.csv  out-of-bag MCC / determinate rate / errors on the paper's
                   evaluation controls, production rule at p = 5, for
                   paper / new_first100 / new_all; agreement of the in-sample
                   points with the paper's
  components.csv   the held-out component rule (median paired val_ll 4c - 3c > 0)
                   on each run
The new run's per-split matrices are built by production_oob._mv_one with the
new results file (npz under review/production200/npz/, never the paper's).
TP53's new fits pool the categorical prior over gnomAD only (the default);
the paper's TP53 fits pooled all samples, so TP53 differs in that too.

Usage: python -m mv_analysis.review_production200 <results.json.gz> [--workers N]
"""
import argparse
import gzip
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

OUT = R.out_dir("production200")
NPZ = OUT / "npz"
GREEDY_FIRST_IDX = 3


def _init(rj):
    os.environ.pop("CFUSN_CAT_POOL", None)  # new fits: the default (gnomAD-only) pool
    from mv_analysis import production_oob as PO
    config.PAPER_RESULTS_JSON = rj
    PO.OUT = NPZ


def _build(key):
    from mv_analysis import production_oob as PO
    try:
        return PO._mv_one(key)
    except Exception as e:  # noqa: BLE001
        return key, f"FAILED {type(e).__name__}: {e}"


def _mv(dirpath):
    z = np.load(dirpath / "mv.npz")
    return {k: z[k] for k in z.files}


def _oob_points(key, mv, keep_rows, p=5):
    from mv_analysis.production_oob import _points
    d, _ = R.prep(key)
    L, oob, priors, pv = mv["L"][keep_rows], mv["oob"][keep_rows].astype(bool), mv["priors"][keep_rows], mv["point_values"]
    rows = np.flatnonzero(d["eval_mask"])
    pts = np.full(len(rows), np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for i, j in enumerate(rows):
            o = oob[:, j]
            if o.any():
                prior = float(np.median(priors[o]))
                lv = L[o, j]
                pts[i] = _points(np.nanpercentile(lv, p), np.nanpercentile(lv, 100 - p), prior, pv)
    return rows, pts


def _insample(mv, keep_rows, p=5):
    from mv_analysis.production_oob import _points
    L, priors, pv = mv["L"][keep_rows], mv["priors"][keep_rows], mv["point_values"]
    prior = float(np.median(priors))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        lo, hi = np.nanpercentile(L, p, axis=0), np.nanpercentile(L, 100 - p, axis=0)
    return np.array([_points(a, b, prior, pv) for a, b in zip(lo, hi)], float)


def _metrics(pts, y):
    from mv_analysis.review_priors import _metrics as M
    return M(pts, y)


def restarts(new, paper, keys):
    rows = []
    for key, m in keys:
        ds, cfg = m["dataset_name"], m["config"]
        a, b = new.get(ds, {}), paper.get(ds, {})
        for s in sorted(set(a) & set(b), key=int):
            ea, eb = a[s].get(cfg), b[s].get(cfg)
            if not ea or not eb:
                continue
            rows.append(dict(key=key, panel=m["panel"], seed=int(s), val_ll_new=ea["val_ll"], val_ll_paper=eb["val_ll"],
                             fit_idx_new=ea["fit_idx"]))
        for s in sorted(a, key=int):
            if int(s) >= 100 and a[s].get(cfg):
                rows.append(dict(key=key, panel=m["panel"], seed=int(s), val_ll_new=a[s][cfg]["val_ll"],
                                 val_ll_paper=np.nan, fit_idx_new=a[s][cfg]["fit_idx"]))
    t = pd.DataFrame(rows)
    t["diff"] = t.val_ll_new - t.val_ll_paper
    t.to_csv(OUT / "restarts_per_bootstrap.csv", index=False)
    def _summ(x):
        dd = x["diff"].dropna()  # paired seeds only (0-99)
        return pd.Series(dict(
            n_paired=len(dd), median_diff=dd.median(), mean_diff=dd.mean(), p90_diff=dd.quantile(0.9),
            frac_better=(dd > 1e-6).mean(), frac_worse=(dd < -1e-6).mean(),
            frac_better_by_0p01=(dd > 0.01).mean(),
            frac_greedy_wins=(x.fit_idx_new >= GREEDY_FIRST_IDX).mean(), n_boots_new=len(x)))
    s = t.groupby(["key", "panel"]).apply(_summ).reset_index()
    s.to_csv(OUT / "restarts.csv", index=False)
    return s


def components(new, paper, keys):
    rows = []
    for key, m in keys:
        ds = m["dataset_name"]
        for run, d in (("paper", paper), ("new", new)):
            if ds not in d:
                continue
            diffs = [d[ds][s]["4c_unc"]["val_ll"] - d[ds][s]["3c_unc"]["val_ll"] for s in d[ds]
                     if d[ds][s].get("4c_unc") and d[ds][s].get("3c_unc")]
            med = float(np.median(diffs)) if diffs else np.nan
            rows.append(dict(key=key, panel=m["panel"], run=run, paper_config=m["config"], n=len(diffs),
                             median_4c_minus_3c=med, heldout_k=4 if med > 0 else 3))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "components.csv", index=False)
    return t


def performance(keys):
    rows = []
    for key, m in keys:
        f = NPZ / key / "mv.npz"
        if not f.exists():
            continue
        d, _ = R.prep(key)
        y_all = np.asarray(d["plp"], bool)
        new = _mv(NPZ / key)
        pap = R.mv_npz(key)
        seeds = new["seeds"]
        sets = {"paper": (pap, np.arange(len(pap["seeds"]))),
                "new_first100": (new, np.flatnonzero(seeds < 100)),
                "new_all": (new, np.arange(len(seeds)))}
        ins_ref = None
        for name, (mv, keep) in sets.items():
            rows_, pts = _oob_points(key, mv, keep)
            ok = ~np.isnan(pts)
            ins = _insample(mv, keep)
            if ins_ref is None:
                ins_ref = ins
            rec = dict(key=key, panel=m["panel"], gene=m["gene"], set=name, n_splits=len(keep),
                       n_controls=int(ok.sum()), insample_agree_with_paper=float(np.mean(ins == ins_ref)),
                       insample_det_rate_all_rows=float(np.mean(ins != 0)),
                       **_metrics(pts[ok], y_all[rows_][ok]))
            rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "performance.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    rj = str(Path(a.results).resolve())
    new = json.load(gzip.open(rj, "rt"))
    paper = json.load(gzip.open(config.PAPER_RESULTS_JSON, "rt"))
    keys = [(k, m) for k, m in R.paper_keys() if m["dataset_name"] in new]
    print(f"{len(keys)} paper models in the new results:", ", ".join(k for k, _ in keys), flush=True)
    (OUT / "source.txt").write_text(rj + "\n")

    s = restarts(new, paper, keys)
    print(s.round(4).to_string(index=False), flush=True)
    c = components(new, paper, keys)
    print(c.pivot_table(index="key", columns="run", values="heldout_k").to_string(), flush=True)

    NPZ.mkdir(exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(rj,)) as ex:
        futs = [ex.submit(_build, k) for k, _ in keys]
        for f in as_completed(futs):
            print(*f.result(), flush=True)
    t = performance(keys)
    print(t.pivot_table(index=["panel", "key"], columns="set", values=["mcc", "det_rate", "errors"]).round(3).to_string())
    print(t.groupby(["panel", "set"])[["mcc", "det_rate", "errors", "insample_agree_with_paper"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
