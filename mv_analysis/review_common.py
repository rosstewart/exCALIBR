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
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from mv_analysis import config

REVIEW_DIR = Path(config.PAPER_FIGURES_DIR) / "review"
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
    from mv_analysis.production_oob import OUT
    z = np.load(OUT / key / "mv.npz")
    return {k: z[k] for k in z.files}


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
    for f in sorted((Path(config.PKNN_DIR) / "prep").glob("*.json")):
        m = json.load(open(f))
        if config.paper_includes(m["key"]) and (panel is None or m["panel"] == panel):
            out.append((m["key"], m))
    return out


def strength_points():
    """The paper's strength_points.csv with its keys in the prep's form
    (the predictor models are keyed "predictor-mv__<gene>" there)."""
    sp = pd.read_csv(Path(config.PAPER_STRENGTH_DIR) / "strength_points.csv", low_memory=False)
    sp["key"] = sp["key"].str.replace("^predictor-mv__", "predictor__", regex=True)
    return sp


def mh_lr(n_p, n_b, tot_p, tot_b):
    """Mantel-Haenszel-style pooled likelihood ratio over genes: sum n_p*B/N over
    sum n_b*P/N, the odds of a level among P/LP vs B/LB controls standardized
    within gene. Arrays over genes."""
    n_p, n_b, tot_p, tot_b = (np.asarray(v, float) for v in (n_p, n_b, tot_p, tot_b))
    N = tot_p + tot_b
    num, den = (n_p * tot_b / N).sum(), (n_b * tot_p / N).sum()
    return num / den if den > 0 else np.inf if num > 0 else np.nan
