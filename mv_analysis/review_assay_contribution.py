"""What each readout contributes to the joint model's evidence, by
marginalizing it out of the fitted model (no refit).

score_masked(key, masked) re-scores every model row with the readouts in
`masked` set missing, through the paper's own analysis (trust_global
marginalization of the fitted mixtures, per-fit priors re-estimated on the
population sample as usual), and returns the per-fit log LR+ matrix with the
bootstrap seeds it kept. Out-of-bag points then use the paper's bootstrap
splits (the canonical mv.npz masks, matched by seed). Results are cached in
review/assay_contribution/cache/.

  drop_one.csv   per model and readout: out-of-bag MCC, determinate rate,
                 errors and mean |points| on the controls with that readout
                 marginalized out, the changes from the full model, and the
                 out-of-bag log loss of the posterior (a proper score; the
                 posterior from the median out-of-bag log LR+ and the out-of-bag
                 prior)
  descriptors.csv per model and readout: coverage (variants, controls with a
                 score), univariate separation of the controls (AUC), the
                 readout's own univariate prior, max |Spearman r| with the
                 model's other readouts

Usage: python -m mv_analysis.review_assay_contribution [--workers N] [--keys ...]
"""
import argparse
import contextlib
import hashlib
import io
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis import review_common as R

OUT = R.out_dir("assay_contribution")
CACHE = OUT / "cache"


class _MaskedMS:
    """A model's MultiScoreset with some readouts set missing for every row."""

    def __init__(self, ms, masked):
        self._base = ms
        names = [str(n) for n in ms.dataset_names]
        X = np.asarray(ms.scores, float).copy()
        for n in masked:
            X[:, names.index(n)] = np.nan
        self.scores = X

    def __getattr__(self, name):
        return getattr(self._base, name)


def _single_threaded():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


@__import__("functools").lru_cache(maxsize=8)
def _ms(key):
    d, m = R.prep(key)
    with contextlib.redirect_stdout(io.StringIO()):
        ms = config.build_ms_for_gene_cluster(m["gene_set"], m["gene"], cluster_idx=m["cluster_idx"],
                                              results_json=config.PAPER_RESULTS_JSON)
    return ms


def score_masked(key, masked=()):
    """(L, priors, seeds) with readouts `masked` marginalized out."""
    from mv_analysis.gene_performance_scatter import fast_results_json
    from mv_analysis.mv_cockpit import DEFAULT_RUN_KWARGS
    from src.assay_calibration.multivariate_analysis import mv_calibration as mvc
    from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
    masked = tuple(sorted(masked))
    if not masked:
        mv = R.mv_npz(key)
        return mv["L"], mv["priors"], mv["seeds"]
    tag = hashlib.md5("|".join(masked).encode()).hexdigest()[:12]
    f = CACHE / key / f"{tag}.npz"
    if f.exists():
        z = np.load(f)
        return z["L"], z["priors"], z["seeds"]
    _, m = R.prep(key)
    ms = _MaskedMS(_ms(key), masked)
    rj = config.PAPER_RESULTS_JSON
    an = build_gene_set_analysis(ms, m["gene"].lower(), rj, dataset_name=m["dataset_name"], gene_set=m["gene_set"])
    an.configs = [m["config"]]
    seeds = [int(k) for k in an.raw_boots]
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
    r = an.results[m["config"]]
    kept = np.asarray([s for s, v in zip(seeds, valid) if v])
    L, priors = np.asarray(r["lr_matrix"], float), np.asarray(r["priors"], float)
    assert len(kept) == L.shape[0], f"{key} {masked}: kept fits do not match the LR matrix"
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, L=L, priors=priors, seeds=kept, masked=np.asarray(masked))
    return L, priors, kept


def oob_eval(key, L, priors, seeds, p=5):
    """Out-of-bag points and posteriors of the evaluation controls under (L,
    priors, seeds), using the canonical splits' masks matched by seed."""
    from mv_analysis.production_oob import _points
    d, _ = R.prep(key)
    mv = R.mv_npz(key)
    pos = {int(s): i for i, s in enumerate(mv["seeds"])}
    idx = np.array([pos.get(int(s), -1) for s in seeds])
    ok = idx >= 0
    L, priors, oob = L[ok], priors[ok], mv["oob"][idx[ok]].astype(bool)
    # the controls are those out of bag in the paper's own fits, so every
    # readout subset is evaluated on the same controls (NaN where a subset's
    # kept fits leave a control without out-of-bag fits)
    rows = np.flatnonzero(d["eval_mask"])
    rows = rows[mv["oob"][:, rows].any(0)]
    pts, post = [], []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for j in rows:
            o = oob[:, j]
            if not o.any():
                pts.append(np.nan)
                post.append(np.nan)
                continue
            lv, pr = L[o, j], float(np.median(priors[o]))
            pts.append(_points(np.nanpercentile(lv, p), np.nanpercentile(lv, 100 - p), pr, mv["point_values"]))
            med = np.nanmedian(lv)
            post.append(1 / (1 + (1 - pr) / pr * np.exp(-med)) if np.isfinite(med) else pr)
    y = np.asarray(d["plp"], bool)[rows]
    return np.asarray(pts, float), np.asarray(post, float), y


def _metrics(pts, post, y):
    from mv_analysis.review_priors import _metrics as M
    eps = 1e-6
    q = np.clip(post, eps, 1 - eps)
    out = M(pts, y)
    out.update(mean_abs_points=float(np.mean(np.abs(np.nan_to_num(pts)))),
               log_loss=float(np.nanmean(-np.where(y, np.log(q), np.log(1 - q)))))
    return out


def _drop_one(key):
    _single_threaded()
    warnings.filterwarnings("ignore")
    _, m = R.prep(key)
    names = [str(n) for n in _ms(key).dataset_names]
    rows = [dict(key=key, panel=m["panel"], gene=m["gene"], dropped="(none)",
                 **_metrics(*oob_eval(key, *score_masked(key))))]
    if len(names) > 1:
        for n in names:
            rows.append(dict(key=key, panel=m["panel"], gene=m["gene"], dropped=n,
                             **_metrics(*oob_eval(key, *score_masked(key, (n,))))))
    return rows


def _descriptors(key):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    d, m = R.prep(key)
    X = np.asarray(d["X"], float)
    names = [str(n) for n in d["dims"]]
    ev, y = np.asarray(d["eval_mask"], bool), np.asarray(d["plp"], bool)
    rows = []
    for j, n in enumerate(names):
        x = X[:, j]
        has = ~np.isnan(x)
        c = has & ev
        auc = roc_auc_score(y[c], x[c]) if c.any() and 0 < y[c].sum() < c.sum() else np.nan
        rs = []
        for k in range(len(names)):
            if k != j:
                both = has & ~np.isnan(X[:, k])
                if both.sum() > 10:
                    rs.append(abs(spearmanr(x[both], X[both, k]).correlation))
        rows.append(dict(key=key, panel=m["panel"], gene=m["gene"], readout=n, n_variants=int(has.sum()),
                         n_controls=int(c.sum()), n_plp=int((c & y).sum()), n_blb=int((c & ~y).sum()),
                         auc_separation=max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
                         max_abs_spearman_other=max(rs) if rs else np.nan))
    return rows


def run(workers=8, keys=None):
    warnings.filterwarnings("ignore")
    ks = [k for k, m in R.paper_keys() if keys is None or k in keys]
    desc = [r for k in ks for r in _descriptors(k)]
    pd.DataFrame(desc).to_csv(OUT / "descriptors.csv", index=False)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_drop_one, ks):
            rows += res
            print(res[0]["key"], flush=True)
    t = pd.DataFrame(rows)
    full = t[t["dropped"] == "(none)"].set_index("key")
    for c in ("mcc", "det_rate", "errors", "mean_abs_points", "log_loss"):
        t[f"delta_{c}"] = t[c] - t["key"].map(full[c])
    t.to_csv(OUT / "drop_one.csv", index=False)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keys", nargs="*")
    a = ap.parse_args()
    run(a.workers, a.keys)
