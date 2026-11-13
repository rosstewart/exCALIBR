"""Shared helpers for the review analyses (mv_analysis/review_*.py).

Every review analysis writes under REVIEW_DIR (PAPER_FIGURES_DIR/review/) and
never into the paper's own output directories.

oob_points(key, p) gives the joint model's out-of-bag points at any
percentile p by production_oob.py's rule (the p-th percentile of the variant's
log LR+ over the fits that did not contain it against the pathogenic
thresholds, the (100-p)-th against the benign ones, at the median prior of
those fits). At p = 5 it reproduces production_oob's MV column exactly.
"""
import functools
import gzip
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config

# Alternative fits (e.g. the review production run, 200 bootstraps x greedy
# restarts): MV_REVIEW_FITS=<bootstrap_results.json.gz> scores the joint model
# from those fits instead of the paper's (config.PAPER_RESULTS_JSON is pointed
# at them for everything that rescores). MV_REVIEW_SUBSET_FROM=<results json>
# restricts the analyses to the paper models present in that file (default:
# MV_REVIEW_FITS), so a partial run can be compared with the paper on the same
# models. MV_REVIEW_DIR sets the output dir (it must differ from the paper's
# review/ whenever either variable is set).
ALT_FITS = os.environ.get("MV_REVIEW_FITS") or None
SUBSET_FROM = os.environ.get("MV_REVIEW_SUBSET_FROM") or ALT_FITS
REVIEW_DIR = Path(os.environ.get("MV_REVIEW_DIR") or Path(config.PAPER_FIGURES_DIR) / "review")
if (ALT_FITS or SUBSET_FROM) and REVIEW_DIR == Path(config.PAPER_FIGURES_DIR) / "review":
    raise RuntimeError("MV_REVIEW_FITS / MV_REVIEW_SUBSET_FROM need their own MV_REVIEW_DIR")
if ALT_FITS:
    config.PAPER_RESULTS_JSON = ALT_FITS
    # new fits use the default (gnomAD-only) categorical pool; CFUSN_CAT_POOL=all
    # only re-scores the paper's older TP53 fits
    os.environ.pop("CFUSN_CAT_POOL", None)
NPZ_DIR = REVIEW_DIR / "npz"  # alternative fits' per-split matrices
PERCENTILES = (5, 10, 15, 20, 25, 50)
LEVELS = (1, 2, 4, 8)  # standard strengths; points are floored to these
GENOME_WIDE_PRIOR = 0.0441  # Chen et al.'s calibration prior


def out_dir(name):
    d = REVIEW_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


@functools.lru_cache(maxsize=None)
def log_c(prior):
    """log of the Tavtigian constant at `prior` (rounded to 4 decimals, as production_oob)."""
    from mv_analysis.production_oob import _log_c
    return _log_c(round(float(prior), 4))


def floor_level(points):
    """Points floored to the standard levels (+-1, 2, 4, 8), keeping sign; 0 stays 0."""
    p = np.asarray(points, float)
    out = np.zeros_like(p)
    for lv in LEVELS:
        out[p >= lv] = lv
        out[p <= -lv] = -lv
    return np.where(np.isnan(p), np.nan, out)


@functools.lru_cache(maxsize=None)
def prep(key):
    from mv_analysis.comparison_methods import pknn_runs as RP
    return RP.load_prep(key)


@functools.lru_cache(maxsize=None)
def mv_npz(key):
    from mv_analysis import production_oob as PO
    if ALT_FITS:
        f = NPZ_DIR / key / "mv.npz"
        if not f.exists():  # production_oob._mv_one on the alternative fits
            out, PO.OUT = PO.OUT, NPZ_DIR
            try:
                PO._mv_one(key)
            finally:
                PO.OUT = out
    else:
        f = PO.OUT / key / "mv.npz"
    z = np.load(f)
    return {k: z[k] for k in z.files}


def insample_paper_points(key):
    """The joint model's in-sample points at p = 5 for every model row: the
    paper's stored points, or the alternative fits' by the same rule."""
    return insample_points(key, 5) if ALT_FITS else np.asarray(prep(key)[0]["mv_points"], float)


@functools.lru_cache(maxsize=1)
def _subset_datasets():
    return set(json.load(gzip.open(SUBSET_FROM, "rt"))) if SUBSET_FROM else None


def oob_points(key, p=5, rows=None):
    """DataFrame (variant, label, n_oob, oob_prior, points) of the joint model's
    out-of-bag points at percentile p for the evaluation controls (or `rows`)."""
    from mv_analysis.production_oob import _points
    d, _ = prep(key)
    mv = mv_npz(key)
    L, oob, priors, pv = mv["L"], mv["oob"].astype(bool), mv["priors"], mv["point_values"]
    rows = np.flatnonzero(d["eval_mask"]) if rows is None else np.asarray(rows)
    recs = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for j in rows:
            o = oob[:, j]
            if not o.any():
                recs.append(dict(row=j, n_oob=0, oob_prior=np.nan, points=np.nan))
                continue
            prior = float(np.median(priors[o]))
            lv = L[o, j]
            recs.append(dict(row=j, n_oob=int(o.sum()), oob_prior=prior,
                             points=_points(np.nanpercentile(lv, p), np.nanpercentile(lv, 100 - p), prior, pv)))
    out = pd.DataFrame(recs)
    out.insert(0, "variant", np.asarray(d["variants"])[out["row"]])
    out.insert(1, "label", d["plp"][out["row"]].astype(int))
    return out


def insample_points(key, p=5):
    """The joint model's in-sample points at percentile p for every model row:
    percentiles over all bootstrap fits, thresholds at the median prior of all
    fits (the analysis's rule). At p = 5 equal to the paper's points."""
    from mv_analysis.production_oob import _points
    mv = mv_npz(key)
    L, priors, pv = mv["L"], mv["priors"], mv["point_values"]
    prior = float(np.median(priors))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        lo, hi = np.nanpercentile(L, p, axis=0), np.nanpercentile(L, 100 - p, axis=0)
    return np.array([_points(a, b, prior, pv) for a, b in zip(lo, hi)], float)


def check_oob_points(key):
    """At p = 5, oob_points must equal production_oob's stored MV column."""
    from mv_analysis.production_oob import OUT
    f = OUT / f"{key}.csv"
    if not f.exists():
        return None
    a = oob_points(key, 5).set_index("variant")
    b = pd.read_csv(f).set_index("variant")
    b = b[b["n_oob"] > 0]
    x, y = a.loc[b.index, "points"].to_numpy(float), b["MV"].to_numpy(float)
    assert np.array_equal(np.nan_to_num(x), np.nan_to_num(y)), f"{key}: oob_points(5) differs from production_oob"
    return True


def paper_keys(panel=None):
    """Evaluated paper models: [(key, meta)] from the P-KNN prep."""
    out = []
    subset = _subset_datasets()
    for f in sorted((Path(config.PKNN_DIR) / "prep").glob("*.json")):
        m = json.load(open(f))
        if config.paper_includes(m["key"]) and (panel is None or m["panel"] == panel) \
                and (subset is None or m["dataset_name"] in subset):
            out.append((m["key"], m))
    return out


def strength_points():
    """The paper's strength_points.csv with its keys in the prep's form
    (the predictor models are keyed "predictor-mv__<gene>" there)."""
    sp = pd.read_csv(Path(config.PAPER_STRENGTH_DIR) / "strength_points.csv", low_memory=False)
    sp["key"] = sp["key"].str.replace("^predictor-mv__", "predictor__", regex=True)
    return sp


MIN_PER_CLASS = 20  # below this many controls of either class, no per-model readout decisions


def evidence_log_lrs(med, lo, hi, prior, points):
    """The three log LR+ versions scored by Cllr, per control, all capped at
    +-log c(prior) (the Very Strong LR: 8 points, the most evidence the
    pipeline can assign; beyond it mixture-tail LRs never become evidence):
      median      the model's estimate of the LR (information)
      percentile  the one-sided bound the evidence rule uses: the 5th
                  percentile if it is above 0, the 95th if below 0, else 0
                  (LR = 1); folds in the bootstrap variance
      points      the LR implied by the points assigned, points/8 x log c
                  (the evidence as applied: step function and prior included)"""
    lc = np.array([log_c(p) if np.isfinite(p) else np.nan for p in np.atleast_1d(prior)], float)
    med, lo, hi = (np.asarray(v, float) for v in (med, lo, hi))
    pct = np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0))
    pct = np.where(np.isfinite(lo) & np.isfinite(hi), pct, np.nan)
    return dict(median=np.clip(med, -lc, lc), percentile=np.clip(pct, -lc, lc),
                points=np.asarray(points, float) / 8 * lc)


def cllr_costs(log_lr, y):
    """Per-control log-likelihood-ratio cost terms (bits) and their weights.

    Cllr (Brummer & du Preez 2006) = 1/2 [mean over P/LP of log2(1 + 1/LR)
    + mean over B/LB of log2(1 + LR)]: the log loss at even prior odds with
    the two classes weighted equally, so it scores the LR alone (no prior),
    is proper (overstated LRs are penalized) and equals 1 bit for no evidence
    (LR = 1). log_lr is the natural log LR+ (NaN: no evidence, LR = 1).
    Returns (cost per control, weight per control) with Cllr = sum(w * cost);
    the weights are 1/(2 n_class)."""
    lr = np.nan_to_num(np.asarray(log_lr, float))
    y = np.asarray(y, bool)
    cost = np.where(y, np.logaddexp(0, -lr), np.logaddexp(0, lr)) / np.log(2)
    n_p, n_b = y.sum(), (~y).sum()
    w = np.where(y, 0.5 / max(n_p, 1), 0.5 / max(n_b, 1))
    return cost, w


def cllr(log_lr, y, idx=None):
    """Cllr in bits (see cllr_costs); idx: a stratified resample of the controls."""
    c, w = cllr_costs(log_lr, y)
    if idx is None:
        return float((c * w).sum())
    yy = np.asarray(y, bool)[idx]
    w = np.where(yy, 0.5 / max(yy.sum(), 1), 0.5 / max((~yy).sum(), 1))
    return float((c[idx] * w).sum())


def cllr_min(log_lr, y):
    """Cllr after the optimal monotone recalibration of the LRs (pool-adjacent
    violators on the controls): the discrimination part of Cllr; Cllr -
    Cllr_min is the calibration loss."""
    from sklearn.isotonic import IsotonicRegression
    lr = np.nan_to_num(np.asarray(log_lr, float))
    y = np.asarray(y, bool)
    n_p, n_b = y.sum(), (~y).sum()
    if not n_p or not n_b:
        return np.nan
    # PAV on class-balanced weights gives the posterior at even prior odds
    w = np.where(y, 0.5 / n_p, 0.5 / n_b)
    q = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(lr, y.astype(float), sample_weight=w).predict(lr)
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return cllr(np.log(q / (1 - q)), y)


def mh_lr(n_p, n_b, tot_p, tot_b):
    """Mantel-Haenszel-style pooled likelihood ratio over genes: sum n_p*B/N over
    sum n_b*P/N, the odds of a level among P/LP vs B/LB controls standardized
    within gene. Arrays over genes."""
    n_p, n_b, tot_p, tot_b = (np.asarray(v, float) for v in (n_p, n_b, tot_p, tot_b))
    N = tot_p + tot_b
    num, den = (n_p * tot_b / N).sum(), (n_b * tot_p / N).sum()
    return num / den if den > 0 else np.inf if num > 0 else np.nan
