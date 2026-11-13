"""Calibration slope of the claimed evidence, per panel and percentile (plan A:
borrowing strength across evidence levels, so sparse levels need not be
tested one by one).

Per panel and p, on the out-of-bag evaluation controls
(review_percentile_choice.controls):
    logit P(P/LP | x, gene) = alpha_gene + beta * x
x = claimed log LR = points / 8 x log c(prior) (0 for no evidence, itself a
claim of LR = 1). By Bayes' rule the gene intercept absorbs each gene's class
balance among its controls and beta compares claimed with observed log LRs:
beta = 1 calibrated, < 1 overstated (a claimed LR behaves like LR^beta),
> 1 conservative. Firth-penalized logistic regression (finite under the
near-separation of these controls); genes with a single class carry no slope
information and are dropped. Interval: 200 resamples of the controls within
gene and class.

Linearity (the assumption that lets the slope speak for sparse levels): the
same model with one coefficient per evidence level in place of beta*x; the
penalized likelihood-ratio statistic against the linear model, and each
level's fitted log LR (its coefficient: the observed log LR of that level
relative to no evidence, gene-adjusted) next to the claimed one.

  slope.csv   beta, interval, n controls / genes, linearity test
  levels.csv  per level: claimed log LR (mean x), fitted log LR from the
              level model with its interval, and the log LR the slope implies
              at its lower bound (exp(beta_lo x): the strength the claim
              keeps if calibration is as bad as the slope allows)
  fig_calibration_slope.png  fitted vs claimed log LR per level, with the
              slope line, per panel and p

Usage: python -m mv_analysis.review_calibration_slope
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

from mv_analysis import review_common as R
from mv_analysis.review_percentile_choice import controls

OUT = R.out_dir("calibration_slope")
N_BOOT = 200
SHOW_P = (5, 10, 15, 25, 50)


def firth(X, y, iters=100, tol=1e-8):
    """Firth-penalized logistic regression. Returns (coef, penalized loglik,
    covariance)."""
    b = np.zeros(X.shape[1])

    def pll(b):
        eta = X @ b
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        sign, logdet = np.linalg.slogdet(info)
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        return ll + 0.5 * logdet, p, w, info
    cur, p, w, info = pll(b)
    for _ in range(iters):
        inv = np.linalg.pinv(info)
        h = w * np.einsum("ij,jk,ik->i", X, inv, X)
        step = inv @ (X.T @ (y - p + h * (0.5 - p)))
        t = 1.0
        while True:
            nb = b + t * step
            new, np_, nw, ninfo = pll(nb)
            if new >= cur - 1e-10 or t < 1e-4:
                break
            t /= 2
        b, p, w, info = nb, np_, nw, ninfo
        if abs(new - cur) < tol and np.max(np.abs(t * step)) < 1e-6:
            cur = new
            break
        cur = new
    return b, cur, np.linalg.pinv(info)


def _design(x, genes, levels=None):
    ug = np.unique(genes)
    G = (genes[:, None] == ug[None, :]).astype(float)
    if levels is None:
        return np.column_stack([G, x])
    lv = [lv for lv in np.unique(levels) if lv != 0]
    return np.column_stack([G] + [(levels == lv).astype(float) for lv in lv]), lv


def fit_panel(x, y, genes, lev, rng):
    keep = np.zeros(len(y), bool)
    for g in np.unique(genes):
        m = genes == g
        if y[m].any() and (~y[m]).any():
            keep |= m
    x, y, genes, lev = x[keep], y[keep].astype(float), genes[keep], lev[keep]
    Xl = _design(x, genes)
    b, pll_lin, cov = firth(Xl, y)
    beta = b[-1]
    # resample within gene and class
    strata = [np.flatnonzero((genes == g) & (y == c)) for g in np.unique(genes) for c in (0, 1)]
    bs = []
    for _ in range(N_BOOT):
        ix = np.concatenate([rng.choice(s, len(s)) for s in strata if len(s)])
        bs.append(firth(_design(x[ix], genes[ix]), y[ix])[0][-1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # level model: one coefficient per evidence level
    Xv, lvs = _design(x, genes, lev)
    bv, pll_lev, covv = firth(Xv, y)
    lrt = 2 * (pll_lev - pll_lin)
    df = len(lvs) - 1
    ng = len(np.unique(genes))
    levels = []
    for i, lv in enumerate(lvs):
        m = lev == lv
        se = np.sqrt(covv[ng + i, ng + i])
        xc = float(np.mean(x[m]))
        levels.append(dict(level=int(lv), n_plp=int(y[m].sum()), n_blb=int((1 - y[m]).sum()),
                           claimed_log_lr=xc, fitted_log_lr=bv[ng + i],
                           fitted_lo=bv[ng + i] - 1.96 * se, fitted_hi=bv[ng + i] + 1.96 * se,
                           implied_log_lr_at_beta_lo=lo * xc, implied_log_lr_at_beta=beta * xc))
    return dict(beta=beta, beta_lo=lo, beta_hi=hi, n_controls=int(len(y)), n_genes=ng,
                linearity_lrt=lrt, linearity_df=df, linearity_p=float(chi2.sf(max(lrt, 0), df)) if df > 0 else np.nan), levels


def run():
    warnings.filterwarnings("ignore")
    t = controls()
    rng = np.random.default_rng(0)
    S, L = [], []
    for (panel, p), x in t.groupby(["panel", "p"]):
        lc = np.array([R.log_c(v) for v in x.prior])
        xx = (x.points.to_numpy(float) / 8) * lc
        ok = np.isfinite(xx)
        s, lv = fit_panel(xx[ok], x.y.to_numpy(bool)[ok], x.key.to_numpy()[ok], x.points.to_numpy(float)[ok], rng)
        S.append(dict(panel=panel, percentile=p, **s))
        L += [dict(panel=panel, percentile=p, **r) for r in lv]
        print(panel, p, {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()}, flush=True)
    s, lv = pd.DataFrame(S), pd.DataFrame(L)
    s.to_csv(OUT / "slope.csv", index=False)
    lv.to_csv(OUT / "levels.csv", index=False)
    figure(s, lv)


def figure(s, lv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = sorted(s.panel.unique())
    ps = [p for p in SHOW_P if p in set(s.percentile)]
    fig, axes = plt.subplots(len(ps), len(panels), figsize=(3.2 * len(panels), 2.8 * len(ps)), squeeze=False)
    for j, panel in enumerate(panels):
        for i, p in enumerate(ps):
            ax = axes[i, j]
            x = lv[(lv.panel == panel) & (lv.percentile == p)]
            r = s[(s.panel == panel) & (s.percentile == p)].iloc[0]
            lim = max(8, np.nanmax(np.abs(x[["claimed_log_lr"]].to_numpy())) + 1)
            g = np.linspace(-lim, lim, 10)
            ax.plot(g, g, color="0.7", lw=0.8, ls="--", label="calibrated")
            ax.plot(g, r.beta * g, color="k", lw=1, label=f"slope {r.beta:.2f} [{r.beta_lo:.2f}, {r.beta_hi:.2f}]")
            ax.errorbar(x.claimed_log_lr, x.fitted_log_lr,
                        yerr=[x.fitted_log_lr - x.fitted_lo, x.fitted_hi - x.fitted_log_lr],
                        fmt="o", ms=3, color="C0", lw=0.8)
            for _, rr in x.iterrows():
                ax.annotate(f"{rr.level:+d}\n{rr.n_plp}/{rr.n_blb}", (rr.claimed_log_lr, rr.fitted_log_lr),
                            fontsize=4.5, xytext=(3, -8), textcoords="offset points")
            ax.set_ylim(-lim * 1.6, lim * 1.6)
            ax.set_title(f"{panel}, p = {p}  (linearity p = {r.linearity_p:.2g})", fontsize=7)
            ax.set_xlabel("claimed log LR", fontsize=6)
            ax.set_ylabel("fitted log LR (gene-adjusted)", fontsize=6)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_calibration_slope.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
