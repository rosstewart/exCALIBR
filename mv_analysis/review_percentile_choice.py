"""Choosing the evidence percentile per panel by several criteria (plan A).

No single score decides: lower percentiles protect against errors on harder,
unclassified variants and against biases in how controls were ascertained
(easier cases, labels partly from predictors), which the controls cannot
measure and which make every control-based metric favour higher p. So every
criterion is reported against p and the choice is made by eye.

Per panel and p in review_percentile_info.PCTS, on the out-of-bag evaluation
controls (production rule; points floored to +-1, 2, 4, 8):

  metrics.csv     pooled over the panel's controls and as the median over its
                  models (models with fewer than MIN_PER_CLASS controls of
                  either class excluded from the medians):
                  determinate rate, errors, MCC, accuracy (overall and among
                  determinate calls), PPV = P/LP share of pathogenic calls,
                  NPV = B/LB share of benign calls, diagnostic odds ratio
                  (TP x TN / FP x FN, 0.5 added to every cell when one is 0),
                  Cllr of the capped median / percentile / points log LR+
                  (bits; information). PPV and NPV depend on the controls'
                  class balance, which is not the population's; compare them
                  across p, not with other studies.
  calibration.csv per evidence level (+-1, 2, 4, 8): P/LP and B/LB counts,
                  the observed LR pooled over genes (Mantel-Haenszel, so each
                  gene's class balance does not leak in) with a 95% interval
                  from 500 resamples of the controls within gene and class,
                  the claimed LR (c^(level/8) at each control's prior; median),
                  and two judgements:
                    not_refuted   the interval reaches the claim (upper bound
                                  >= claimed for pathogenic levels, lower bound
                                  <= claimed for benign ones)
                    demonstrated  the whole interval is at least as strong as
                                  the claim (lower bound >= claimed, or upper
                                  <= claimed for benign); the ClinGen-style
                                  standard (Pejaver et al. 2022)
                  plus n_wrong_side and max_demonstrable_lr: the strongest LR
                  these counts could demonstrate even with no wrong-side
                  control (rule of three on the opposite class), i.e. whether
                  the level can be tested at all
  fig_percentile_choice.png  every criterion against p, one column per panel

Usage: python -m mv_analysis.review_percentile_choice
"""
import warnings

import numpy as np
import pandas as pd

from mv_analysis import review_common as R
from mv_analysis.review_percentile_info import PCTS, model_lrs

OUT = R.out_dir("percentile_choice")
N_BOOT = 500
LEVELS = (-8, -4, -2, -1, 1, 2, 4, 8)


def _cls(pts, y):
    pts = np.nan_to_num(np.asarray(pts, float))
    tp, fp = int(((pts >= 1) & y).sum()), int(((pts >= 1) & ~y).sum())
    tn, fn = int(((pts <= -1) & ~y).sum()), int(((pts <= -1) & y).sum())
    det = tp + tn + fp + fn
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    a, b, c, d = (tp, fp, fn, tn) if min(tp, fp, fn, tn) > 0 else (tp + .5, fp + .5, fn + .5, tn + .5)
    return dict(n=len(pts), det_rate=det / len(pts), errors=fp + fn,
                mcc=(tp * tn - fp * fn) / den if den else np.nan,
                accuracy=(tp + tn) / len(pts), accuracy_determinate=(tp + tn) / det if det else np.nan,
                ppv=tp / (tp + fp) if tp + fp else np.nan, npv=tn / (tn + fn) if tn + fn else np.nan,
                dor=(a * d) / (b * c))


def controls():
    """One row per (model, control, p): gene, label, prior, floored points and
    the three capped log LR+ versions."""
    rows = []
    for key, m in R.paper_keys():
        lrs, y = model_lrs(key)
        d, _ = R.prep(key)
        mv = R.mv_npz(key)
        r = np.flatnonzero(d["eval_mask"])
        r = r[mv["oob"][:, r].any(0)]
        oob = mv["oob"][:, r].astype(bool)
        prior = np.array([np.median(mv["priors"][oob[:, j]]) for j in range(len(r))])
        few = bool(min(y.sum(), (~y).sum()) < R.MIN_PER_CLASS)
        for p, e in lrs.items():
            pts = np.round(e["points"] / np.array([R.log_c(x) for x in prior]) * 8)
            rows.append(pd.DataFrame(dict(key=key, panel=m["panel"], gene=m["gene"], p=p, y=y, prior=prior,
                                          points=R.floor_level(pts), lr_median=e["median"],
                                          lr_percentile=e["percentile"], lr_points=e["points"], few=few)))
        print(key, flush=True)
    return pd.concat(rows, ignore_index=True)


def metrics(t):
    out = []
    for (panel, p), x in t.groupby(["panel", "p"]):
        y = x.y.to_numpy(bool)
        pooled = _cls(x.points, y)
        per = pd.DataFrame([dict(_cls(g.points, g.y.to_numpy(bool)),
                                 **{f"cllr_{s}": R.cllr(g[f"lr_{s}"], g.y.to_numpy(bool))
                                    for s in ("median", "percentile", "points")})
                            for _, g in x[~x.few].groupby("key")])
        rec = dict(panel=panel, percentile=p, n_models=x.key.nunique(), n_models_in_medians=len(per),
                   **{f"pooled_{k}": v for k, v in pooled.items()},
                   **{f"median_{k}": per[k].median() for k in per.columns if k != "n"})
        out.append(rec)
    return pd.DataFrame(out)


def calibration(t, rng):
    out = []
    for (panel, p), x in t.groupby(["panel", "p"]):
        genes = x.key.to_numpy()
        ug, gi = np.unique(genes, return_inverse=True)
        y = x.y.to_numpy(bool)
        pts = x.points.to_numpy(float)
        G = len(ug)
        tot_p = np.bincount(gi[y], minlength=G).astype(float)
        tot_b = np.bincount(gi[~y], minlength=G).astype(float)
        for lv in LEVELS:
            inl = pts == lv
            n_p = np.bincount(gi[y & inl], minlength=G).astype(float)
            n_b = np.bincount(gi[~y & inl], minlength=G).astype(float)
            if n_p.sum() + n_b.sum() == 0:
                continue
            obs = R.mh_lr(n_p, n_b, tot_p, tot_b)
            # resample controls within gene and class: binomial on the level counts
            bs = []
            for _ in range(N_BOOT):
                bp = rng.binomial(tot_p.astype(int), np.divide(n_p, tot_p, out=np.zeros(G), where=tot_p > 0))
                bb = rng.binomial(tot_b.astype(int), np.divide(n_b, tot_b, out=np.zeros(G), where=tot_b > 0))
                bs.append(R.mh_lr(bp, bb, tot_p, tot_b))
            bs = np.nan_to_num(np.array(bs, float), nan=np.nan, posinf=1e12)
            lo, hi = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
            hi = np.inf if hi >= 1e12 else hi
            lo = np.inf if lo >= 1e12 else lo
            claimed = float(np.median(np.exp(lv / 8 * np.array([R.log_c(v) for v in x.prior[inl]]))))
            path = lv > 0
            # strongest LR demonstrable with no wrong-side control (rule of three)
            share_right = (n_p.sum() / tot_p.sum()) if path else (n_b.sum() / tot_b.sum())
            wrong_upper = 3 / (tot_b.sum() if path else tot_p.sum())
            maxdem = share_right / wrong_upper if path else wrong_upper / share_right if share_right else np.nan
            n_wrong = n_b.sum() if path else n_p.sum()
            if n_wrong == 0:
                # resampling cannot bound the open side when no control is on
                # the wrong side; use the rule-of-three bound there
                if path:
                    lo, hi = maxdem, np.inf
                else:
                    lo, hi = 0.0, maxdem
            out.append(dict(panel=panel, percentile=p, level=lv, n_plp=int(n_p.sum()), n_blb=int(n_b.sum()),
                            n_wrong_side=int(n_b.sum() if path else n_p.sum()),
                            observed_lr=obs, lr_lo=lo, lr_hi=hi, claimed_lr=claimed,
                            not_refuted=bool(hi >= claimed) if path else bool(lo <= claimed),
                            demonstrated=bool(lo >= claimed) if path else bool(hi <= claimed),
                            max_demonstrable_lr=maxdem,
                            testable=bool(maxdem >= claimed) if path else bool(maxdem <= claimed)))
    return pd.DataFrame(out)


def figure(mt, cal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = sorted(mt.panel.unique())
    rows = [("pooled_mcc", "MCC (pooled)"), ("pooled_det_rate", "determinate rate"), ("pooled_errors", "errors"),
            ("pooled_ppv", "PPV"), ("pooled_npv", "NPV"), ("pooled_dor", "DOR (log)"),
            ("median_cllr_points", "Cllr of points (bits, model median)"), ("cal", "levels not refuted / demonstrated")]
    fig, axes = plt.subplots(len(rows), len(panels), figsize=(3.2 * len(panels), 1.7 * len(rows)), sharex=True)
    for j, panel in enumerate(panels):
        x = mt[mt.panel == panel].sort_values("percentile")
        for i, (col, lab) in enumerate(rows):
            ax = axes[i, j]
            if col == "cal":
                c = cal[cal.panel == panel].groupby("percentile")
                ax.plot(c.not_refuted.mean().index, c.not_refuted.mean().values, "o-", ms=3, label="not refuted")
                ax.plot(c.demonstrated.mean().index, c.demonstrated.mean().values, "s-", ms=3, label="demonstrated")
                ax.set_ylim(-0.05, 1.05)
                if j == 0:
                    ax.legend(fontsize=6, frameon=False)
            else:
                ax.plot(x.percentile, x[col], "o-", ms=3, color="k")
                if col == "pooled_dor":
                    ax.set_yscale("log")
            ax.axvline(5, color="0.7", lw=0.6, ls="--")
            if j == 0:
                ax.set_ylabel(lab, fontsize=7)
            if i == 0:
                ax.set_title(panel, fontsize=8)
            ax.tick_params(labelsize=6)
        axes[-1, j].set_xlabel("percentile p")
    fig.tight_layout()
    fig.savefig(OUT / "fig_percentile_choice.png", dpi=180)
    plt.close(fig)


def run():
    warnings.filterwarnings("ignore")
    t = controls()
    mt = metrics(t)
    mt.to_csv(OUT / "metrics.csv", index=False)
    cal = calibration(t, np.random.default_rng(0))
    cal.to_csv(OUT / "calibration.csv", index=False)
    figure(mt, cal)
    cols = ["panel", "percentile", "pooled_mcc", "pooled_det_rate", "pooled_errors", "pooled_accuracy", "pooled_ppv",
            "pooled_npv", "pooled_dor", "median_mcc", "median_det_rate", "median_cllr_median", "median_cllr_points"]
    print(mt[cols].round(3).to_string(index=False))
    print(cal.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
