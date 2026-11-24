"""Choosing the evidence percentile by information (plan A, information view).

The evidence rule claims, for each variant, the p-th percentile of its log
LR+ over the bootstrap fits against the pathogenic thresholds and the
(100-p)-th against the benign ones. Small p is conservative: the bound shrinks
toward LR = 1 as the fits vary, and more readouts make them vary more (the
dimension-driven conservatism). p = 50 is the median, the model's estimate.

For every model and p in PCTS, on the out-of-bag evaluation controls (the
production rule, review_common.oob_points' fits and priors), Cllr (bits;
review_common.cllr) of
  percentile  the one-sided bound itself (5th/95th -> p-th/(100-p)-th), and
  points      the LR implied by the points it earns at the out-of-bag prior,
both capped at the Very Strong LR (review_common.evidence_log_lrs). Lower is
better; 1 bit = no evidence. Too conservative a p loses information; too
liberal a p is penalized once the controls show the evidence overstated
(Cllr is proper, but with few, well-separated controls it can only see gross
overstatement, so the calibration check per level (review_percentile_sets)
remains the guard against overstating).

  by_model.csv  Cllr per model and p, with the p that minimizes each, and
                stratified-resample intervals of Cllr(p) - Cllr(5)
  by_panel.csv  per panel and p: median over models and the share of models
                where p beats p = 5; models with too few controls excluded
  fig_cllr_by_percentile.png

Usage: python -m mv_analysis.review_percentile_info
"""
import warnings

import numpy as np
import pandas as pd

from mv_analysis import review_common as R

OUT = R.out_dir("percentile_info")
PCTS = (2.5, 5, 10, 15, 20, 25, 30, 40, 50)
N_BOOT = 200


def model_lrs(key):
    """{p: evidence_log_lrs dict} on the evaluation controls with out-of-bag fits."""
    from mv_analysis.production_oob import _points
    d, _ = R.prep(key)
    mv = R.mv_npz(key)
    L, oob, priors, pv = mv["L"], mv["oob"].astype(bool), mv["priors"], mv["point_values"]
    rows = np.flatnonzero(d["eval_mask"])
    rows = rows[oob[:, rows].any(0)]
    y = np.asarray(d["plp"], bool)[rows]
    Lc = np.where(oob[:, rows], L[:, rows], np.nan)
    prior = np.array([np.median(priors[oob[:, j]]) for j in rows])
    out = {}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        med = np.nanmedian(Lc, axis=0)
        for p in PCTS:
            lo, hi = np.nanpercentile(Lc, p, axis=0), np.nanpercentile(Lc, 100 - p, axis=0)
            pts = np.array([_points(a, b, pr, pv) for a, b, pr in zip(lo, hi, prior)], float)
            out[p] = R.evidence_log_lrs(med, lo, hi, prior, pts)
    return out, y


def run():
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    rows = []
    for key, m in R.paper_keys():
        lrs, y = model_lrs(key)
        if not y.any() or y.all():
            continue
        idx = [np.concatenate([rng.choice(np.flatnonzero(y == c), (y == c).sum()) for c in (True, False)])
               for _ in range(N_BOOT)]
        for score in ("percentile", "points"):
            ref = R.cllr(lrs[5][score], y)
            ref_b = np.array([R.cllr(lrs[5][score], y, ix) for ix in idx])
            for p in PCTS:
                c = R.cllr(lrs[p][score], y)
                diff_b = np.array([R.cllr(lrs[p][score], y, ix) for ix in idx]) - ref_b
                rows.append(dict(key=key, panel=m["panel"], gene=m["gene"], score=score, percentile=p, cllr=c,
                                 cllr_minus_p5=c - ref, diff_lo=np.percentile(diff_b, 2.5),
                                 diff_hi=np.percentile(diff_b, 97.5), n_plp=int(y.sum()), n_blb=int((~y).sum()),
                                 few_controls=bool(min(y.sum(), (~y).sum()) < R.MIN_PER_CLASS)))
        print(key, flush=True)
    t = pd.DataFrame(rows)
    best = t.loc[t.groupby(["key", "score"])["cllr"].idxmin(), ["key", "score", "percentile"]].rename(
        columns={"percentile": "best_percentile"})
    t = t.merge(best, on=["key", "score"])
    t.to_csv(OUT / "by_model.csv", index=False)
    ok = t[~t.few_controls]
    s = ok.groupby(["panel", "score", "percentile"]).agg(
        n_models=("key", "nunique"), median_cllr=("cllr", "median"),
        median_minus_p5=("cllr_minus_p5", "median"),
        share_better_than_p5=("cllr_minus_p5", lambda x: float((x < -1e-9).mean())),
        share_sig_better=("diff_hi", lambda x: float((x < 0).mean())),
        share_sig_worse=("diff_lo", lambda x: float((x > 0).mean()))).reset_index()
    s.to_csv(OUT / "by_panel.csv", index=False)
    _figure(ok)
    print(s.round(3).to_string(index=False))
    print(ok.drop_duplicates(["key", "score"]).groupby(["panel", "score"])["best_percentile"]
          .value_counts().unstack(fill_value=0).to_string())


def _figure(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = sorted(t.panel.unique())
    fig, axes = plt.subplots(2, len(panels), figsize=(3.2 * len(panels), 5.5), sharex=True)
    for j, panel in enumerate(panels):
        for i, score in enumerate(("percentile", "points")):
            ax = axes[i, j]
            x = t[(t.panel == panel) & (t.score == score)]
            for _, g in x.groupby("key"):
                ax.plot(g.percentile, g.cllr_minus_p5, color="0.6", lw=0.7)
            med = x.groupby("percentile").cllr_minus_p5.median()
            ax.plot(med.index, med.values, color="k", lw=2)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"{panel}: {score}", fontsize=8)
            ax.set_xlabel("percentile p")
            ax.set_ylabel("Cllr(p) - Cllr(5) (bits)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cllr_by_percentile.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
