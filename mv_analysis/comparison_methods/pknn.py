"""P-KNN (Lin & Brandes 2026, Genet Med; doi 10.1016/j.gim.2026.102692) as a
comparison method for ExCALIBR-MV.

A thin wrapper around the authors' CPU implementation (P-KNN/P_KNN/P_KNN_CPU.py,
imported unmodified): their per-bootstrap scorer and point conversion are
used as-is. Differences from their `get_bootstrap_KNN_score` are execution
only:
  - a bounded worker pool (theirs hard-codes joblib n_jobs=-1);
  - np.random seeded per bootstrap, so mutual_info_classif's tie-breaking
    noise (random_state=None) is reproducible;
  - leave-one-out / k-fold drivers for per-gene evaluation, following their
    TP53 protocol (Supplementary Methods, "Integration of different evidence
    sources": leave-one-out, N_calibration = 10, the full labelled set,
    query included, as the regularization set, gene-specific prior).
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np

PKNN_SRC = Path(__file__).resolve().parent / "P-KNN" / "P_KNN"
if str(PKNN_SRC) not in sys.path:
    sys.path.insert(0, str(PKNN_SRC))

import P_KNN_CPU as _pk  # noqa: E402

N_BOOTSTRAP = 100
BOOTSTRAP_ALPHA = 0.05
FRAC_REGULARIZATION = 0.03


def get_logbase(prior):
    return _pk.get_logbase(prior)


def _one_bootstrap(cal_X, cal_y, query_X, reg_X, w, n_cal, frac_reg, seed):
    """Their get_one_bootstrap_KNN_score with np.random seeded first, plus
    the number of feature columns that survived KNN imputation (sklearn's
    KNNImputer drops columns that are entirely NaN in the resampled set)."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(cal_X.shape[0], size=cal_X.shape[0], replace=True)
    dropped = int(np.isnan(cal_X[idx]).all(axis=0).sum())
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        post = _pk.get_one_bootstrap_KNN_score(
            cal_X, query_X, reg_X, cal_y, w, n_calibration_in_window=n_cal,
            frac_regularization_in_window=frac_reg, normalization="rank", impute=True,
            mi_scaling=True, seed=seed)
    return post, dropped


def bootstrap_posteriors(cal_X, cal_y, query_X, reg_X, prior, n_cal, n_boot=N_BOOTSTRAP,
                         frac_reg=FRAC_REGULARIZATION, n_jobs=1):
    """(n_query, n_boot) posterior probabilities of pathogenicity, one column
    per bootstrap of the labelled set (seeds 0..n_boot-1, as in their code),
    and the per-bootstrap count of dropped all-NaN columns.

    The benign weight is computed once from the full labelled set, as in
    their get_bootstrap_KNN_score. n_cal is capped at n_labelled - 1 (their
    window indexes the n_cal-th sorted distance)."""
    cal_X = np.asarray(cal_X, float)
    cal_y = np.asarray(cal_y, int)
    query_X = np.atleast_2d(np.asarray(query_X, float))
    reg_X = np.asarray(reg_X, float)
    w = _pk.get_weight(cal_y, prior)
    n_cal = min(int(n_cal), cal_X.shape[0] - 1)
    if n_jobs == 1:
        res = [_one_bootstrap(cal_X, cal_y, query_X, reg_X, w, n_cal, frac_reg, s) for s in range(n_boot)]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(_one_bootstrap)(cal_X, cal_y, query_X, reg_X, w, n_cal, frac_reg, s)
                                      for s in range(n_boot))
    post = np.column_stack([r[0] for r in res])
    return post, np.array([r[1] for r in res])


def to_points(post, prior, alpha=BOOTSTRAP_ALPHA, logbase=None):
    """Their get_P_KNN_ACMG_score: (P_path, P_benign, points), each side the
    alpha-quantile bootstrap bound, LR base derived from the prior unless
    given (their genome-wide analyses pass Pejaver et al.'s 1124), clipped to
    [0, 8] per side, points = pathogenic - benign."""
    return _pk.get_P_KNN_ACMG_score(np.asarray(post, float), alpha, prior, logbase or get_logbase(prior))


def _reg_for(reg, X):
    return X if isinstance(reg, str) and reg == "labelled" else np.asarray(reg, float)


def leave_one_out(X, y, prior, n_cal, reg="labelled", query_mask=None, n_boot=N_BOOTSTRAP,
                  frac_reg=FRAC_REGULARIZATION, n_jobs=1, folds=None, seed=0):
    """Out-of-sample P-KNN posteriors for the labelled rows of X.

    Each held-out row (or fold) is scored by P-KNN calibrated on the other
    labelled rows. reg="labelled": the full labelled set, held-out rows
    included, is the regularization set (their TP53 protocol); otherwise an
    array of unlabelled rows (e.g. the gene's gnomAD variants).
    query_mask: rows to score (default all); calibration always uses every
    other labelled row. folds=None is leave-one-out; an int k uses stratified
    k-fold. Returns (post (n, n_boot) with NaN rows outside query_mask,
    dropped-column counts per scored unit)."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n = len(y)
    query_mask = np.ones(n, bool) if query_mask is None else np.asarray(query_mask, bool)
    reg_X = _reg_for(reg, X)
    if folds is None:
        units = [np.array([i]) for i in np.flatnonzero(query_mask)]
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=seed)
        units = [te[query_mask[te]] for _, te in skf.split(X, y)]
        units = [u for u in units if len(u)]

    def _unit(te):
        tr = np.ones(n, bool)
        tr[te] = False
        assert not tr[te].any()
        post, dropped = bootstrap_posteriors(X[tr], y[tr], X[te], reg_X, prior, n_cal, n_boot=n_boot,
                                             frac_reg=frac_reg, n_jobs=1)
        return te, post, int(dropped.max())

    if n_jobs == 1:
        res = [_unit(te) for te in units]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(_unit)(te) for te in units)
    out = np.full((n, n_boot), np.nan)
    drops = []
    for te, post, d in res:
        out[te] = post
        drops.append(d)
    return out, np.array(drops)


def fit_and_score(X, y, query_X, prior, n_cal, reg="labelled", n_boot=N_BOOTSTRAP,
                  frac_reg=FRAC_REGULARIZATION, n_jobs=1):
    """P-KNN calibrated on all labelled rows, scoring unlabelled queries
    (e.g. VUS) by inference only. reg="labelled": the labelled set is the
    regularization set, as in leave-one-out; the queries take no part in it,
    so a query's score does not depend on which other queries are scored."""
    X = np.asarray(X, float)
    query_X = np.atleast_2d(np.asarray(query_X, float))
    reg_X = _reg_for(reg, X)
    return bootstrap_posteriors(X, y, query_X, reg_X, prior, n_cal, n_boot=n_boot, frac_reg=frac_reg,
                                n_jobs=n_jobs)
