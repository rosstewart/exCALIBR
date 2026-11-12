"""Supplementary figures of the calibration-quality simulation, with a design diagram and per-panel cartoons.

    S3  make_design_figure       design diagram (A), prior reliability (B) and evidence retained (C) across the grid
    S4  make_overstated_figure   ExCALIBR vs acmgscaler: overstated evidence on points (Tavtigian, lp_anchored) and on
                                 the posterior, by controls per class (true prior 0.10) and by true prior
    S5  make_posterior_figure    precision of pathogenic / benign calls by posterior threshold, with the evidence
                                 points marked; by true prior; call rate
    S6  make_diagnostics_figure  the bootstrap prior spread as a flag for unreliable priors and for little usable
                                 evidence; real calibrations
    S7  make_misfit_figure       well-fit, overfit and underfit examples; in-sample vs held-out Yang distance;
                                 overstated evidence vs the fitted-vs-true distance
Panels that need metrics added after the first grid run (log-LR retention, fitted-vs-true distance) fall back to the
points-based version or say so on the panel.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
import scipy.stats as sps

from analysis.calibration_quality.figures import (CAT, G_ORDER, GRID, INK, K_ORDER, MUTED, N_ORDER, RICH_LABEL, SEQ,
                                                  _counts, _heat_block, _letter, _style, precision_curve,
                                                  retention_by_cell)
from analysis.prior_simulation.figures import BENIGN, PATH_C, _bar, _cartoon_axes

EXC, ACM = CAT[0], CAT[1]
EXC_LIGHT, ACM_LIGHT = "#86b6ef", "#f4a37e"  # same hues, lighter steps (fewer controls)
UNMEAS = "#d9a3aa"  # pathogenic, unmeasured mechanism (scores like benign)


def _save(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def _title(ax, i, text, pad=4):
    ax.set_title(f"{_letter(i)} {text}", loc="left", color=INK, fontweight="bold", pad=pad)


# ── cartoons ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _cartoon_scores(ax, kappas=(1.5, 2.5, 4.0), gamma_note=True):
    x = np.linspace(-8, 4, 300)
    ax.fill_between(x, sps.norm.pdf(x), color=BENIGN, alpha=0.25, lw=0)
    ax.plot(x, sps.norm.pdf(x), color=BENIGN, lw=1.1)
    for i, (k, a) in enumerate(zip(kappas, (0.4, 0.7, 1.0))):
        ax.plot(x, sps.norm.pdf(x, -k), color=PATH_C, lw=1.1, alpha=a)
        ax.text(-k, 0.415 + 0.045 * (len(kappas) - 1 - i), f"κ = {k:g}", fontsize=5.2, color=PATH_C, ha="center")
    ax.plot(x, 0.97 * sps.norm.pdf(x), color=UNMEAS, lw=1.0, ls="--")
    ax.text(0.9, 0.36, "benign", fontsize=5.6, color=BENIGN)
    if gamma_note:
        ax.text(1.6, 0.2, "pathogenic,\nunmeasured\n(1 − γ)", fontsize=5.0, color="#a8434f", va="center")
        ax.text(-7.8, 0.36, "pathogenic,\nmeasured (γ)", fontsize=5.0, color=PATH_C)
    ax.set_xlim(-8, 4)
    ax.set_ylim(0, 0.53)
    ax.set_yticks([])
    ax.set_xticks([-4, 0])
    ax.set_xticklabels(["abnormal", "normal"], fontsize=5.8)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)


def _cartoon_samples(ax):
    _cartoon_axes(ax)
    g = 0.6
    _bar(ax, 3, [(0.7 * g, PATH_C, ""), (0.7 * (1 - g), UNMEAS, ""), (0.3, BENIGN, "////")], "P/LP",
         "5–80; mix γ_P/LP, benign ε")
    _bar(ax, 2, [(0.8, BENIGN, ""), (0.2, PATH_C, "////")], "B/LB", "0–80; pathogenic η")
    _bar(ax, 1, [(0.25 * g, PATH_C, ""), (0.25 * (1 - g), UNMEAS, ""), (0.75, BENIGN, "")], "population",
         "50 / 200 / 1000; prior π")
    _bar(ax, 0, [(1, BENIGN, "")], "synonymous", "0 / 100 / 500")
    ax.set_ylim(-0.6, 3.6)


def _cartoon_arms(ax, methods=True):
    ax.axis("off")
    rows = [("Truth", "true likelihood ratio and true prior", INK),
            ("True components", "the pipeline with the true score distributions", "#8a919c"),
            ("ExCALIBR", "fit of all four samples, estimated prior", EXC)]
    if methods:
        rows.append(("acmgscaler", "P/LP and B/LB only, prior fixed at 0.10", ACM))
    ax.text(0, 1.0, "Each replicate, scored on held-out\ntruly pathogenic and benign variants:", fontsize=5.8,
            color=MUTED, va="top", transform=ax.transAxes)
    for i, (n, d, col) in enumerate(rows):
        y = 0.62 - i * 0.19
        ax.add_patch(Rectangle((0, y - 0.05), 0.05, 0.1, color=col, transform=ax.transAxes, clip_on=False))
        ax.text(0.08, y, n, fontsize=6.2, color=INK, fontweight="bold", va="center", transform=ax.transAxes)
        ax.text(0.08, y - 0.085, d, fontsize=5.4, color=MUTED, va="center", transform=ax.transAxes)


def _cartoon_counts(ax, counts=N_ORDER, label="controls per class"):
    ax.axis("off")
    full = np.log10(max(counts) + 1)
    for i, n in enumerate(counts):
        h = np.log10(n + 1) / full
        ax.add_patch(Rectangle((i - 0.2, 0), 0.4, h, color=PATH_C, transform=ax.transData))
        ax.add_patch(Rectangle((i + 0.2, 0), 0.4, h, color=BENIGN, transform=ax.transData))
    ax.set_xlim(-0.6, len(counts))
    ax.set_ylim(0, 1.25)
    ax.text(len(counts) / 2 - 0.3, 1.08, f"{label}: P/LP (red), B/LB (gray), log scale", fontsize=5.4, color=MUTED,
            ha="center")


def _cartoon_priors(ax, priors=(0.05, 0.10, 0.25, 0.40)):
    ax.axis("off")
    for i, p in enumerate(priors):
        ax.add_patch(Rectangle((i * 1.2, 0), 1.0, p * 2.2, color=PATH_C))
        ax.add_patch(Rectangle((i * 1.2, p * 2.2), 1.0, (1 - p) * 0.55, color=BENIGN, alpha=0.6))
        ax.text(i * 1.2 + 0.5, -0.12, f"{p:g}", fontsize=5.4, color=MUTED, ha="center", va="top")
    ax.set_xlim(-0.2, len(priors) * 1.2)
    ax.set_ylim(-0.35, 1.5)
    ax.text(len(priors) * 0.6, 1.33, "true prior: pathogenic share of the population", fontsize=5.4, color=MUTED,
            ha="center")


def _cartoon_overstated(ax):
    """Definitions: points (an evidence level beyond the truth's, or the wrong direction) and posterior (how far the
    posterior goes beyond the truth's, in the direction its evidence claims)."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.0, 0.98, "Overstated evidence", fontsize=6.6, color=INK, fontweight="bold", va="top")
    lv = [-8, -4, -2, -1, 0, 1, 2, 4, 8]
    xs = np.linspace(0.04, 0.96, len(lv))
    y = 0.62
    ax.plot([xs[0], xs[-1]], [y, y], color=MUTED, lw=0.8)
    for x, v in zip(xs, lv):
        ax.plot([x, x], [y - 0.02, y + 0.02], color=MUTED, lw=0.8)
        ax.text(x, y - 0.05, f"{v:+d}" if v else "0", fontsize=5, color=MUTED, ha="center", va="top")
    ax.plot(xs[6], y - 0.13, marker="^", color=INK, ms=4.5, ls="")
    ax.text(xs[6], y - 0.2, "truth", fontsize=5.2, color=INK, ha="center", va="top")
    for x, ok, lab in ((xs[5], True, "weaker"), (xs[7], False, "stronger"), (xs[2], False, "wrong\ndirection")):
        ax.plot(x, y + 0.06, marker="o", ms=4.5, color="#c0392b" if not ok else EXC, mfc="white" if ok else "#c0392b",
                ls="")
        ax.text(x, y + 0.11, lab, fontsize=4.8, color=MUTED, ha="center", va="bottom")
    ax.text(0.0, 0.3, "Points: share of variants whose level is beyond\nthe truth's (filled), "
            "on the ACMG/AMP levels", fontsize=5.1, color=MUTED, va="top")
    ax.text(0.0, 0.11, "Posterior: mean amount (percentage points) the\nposterior goes past the truth's, in the\n"
            "direction its evidence claims", fontsize=5.1, color=MUTED, va="top")


def _cartoon_priors_boot(ax):
    """Three bootstrap prior distributions: tight (reliable), wide (unreliable), piled at the 0.5 start."""
    ax.axis("off")
    rng = np.random.default_rng(1)
    specs = [("tight: reliable", rng.normal(0.2, 0.02, 400)), ("wide: unreliable", rng.normal(0.25, 0.15, 400)),
             ("at the 0.5 start:\nnot identified", np.r_[np.full(250, 0.5), rng.normal(0.3, 0.12, 150)])]
    for i, (lab, v) in enumerate(specs):
        sub = ax.inset_axes([0.02 + 0.33 * i, 0.22, 0.29, 0.62])
        sub.hist(np.clip(v, 0, 1), bins=np.linspace(0, 1, 26), color=EXC, alpha=0.85)
        sub.axvline(0.2, color=INK, lw=0.8, ls="--")
        sub.set_xlim(0, 1)
        sub.set_yticks([])
        sub.set_xticks([0, 0.5, 1])
        sub.tick_params(labelsize=5)
        for s in ("left", "top", "right"):
            sub.spines[s].set_visible(False)
        sub.set_title(lab, fontsize=5.4, color=MUTED, pad=1)
    ax.text(0.5, 0.0, "prior estimates across bootstrap iterations (dashed: truth)", fontsize=5.4, color=MUTED,
            ha="center", transform=ax.transAxes)


# ── S3: design + degradation ─────────────────────────────────────────────────────────────────────────────────────

def llr_retention_by_cell(g):
    """Usable evidence retained: total capped |reported log LR| (same direction as the truth) over truly pathogenic
    and benign variants, as a share of the truth's capped |log LR|; falls back to points when the log-LR columns are
    absent."""
    cols = ["fit_pathogenic_llr_rep_abs_dir_cap", "fit_benign_llr_rep_abs_dir_cap",
            "fit_pathogenic_llr_true_abs_cap", "fit_benign_llr_true_abs_cap"]
    if not all(c in g for c in cols) or g[cols].isna().all().all():
        return retention_by_cell(g)
    num = g[cols[0]].sum() + g[cols[1]].sum()
    den = g[cols[2]].sum() + g[cols[3]].sum()
    return float(num / den) if den > 0 else np.nan


def make_design_figure(df, out_path):
    has_llr = "fit_pathogenic_llr_rep_abs_dir_cap" in df and df["fit_pathogenic_llr_rep_abs_dir_cap"].notna().any()
    fig = plt.figure(figsize=(7.2, 9.6))
    outer = fig.add_gridspec(3, 2, height_ratios=[0.5, 1, 1], width_ratios=[1, 0.035], hspace=0.32, wspace=0.04,
                             left=0.12, right=0.92, top=0.965, bottom=0.05)
    top = outer[0, :].subgridspec(1, 3, width_ratios=[1, 1.25, 1.05], wspace=0.35)
    a1, a2, a3 = (fig.add_subplot(top[0, i]) for i in range(3))
    _cartoon_scores(a1)
    a1.set_title(f"{_letter(0)} Simulated scores", loc="left", color=INK, fontweight="bold", pad=4, x=-0.05)
    _cartoon_samples(a2)
    a2.set_title("Samples (red: pathogenic; hatched: mislabeled)", loc="left", color=MUTED, fontsize=6.2, pad=4)
    _cartoon_arms(a3)
    im = _heat_block(fig, outer[1, 0], df, lambda g: float(g["prior_bad"].mean()), 1.0,
                     "Prior more than 10 percentage points off", _letter(1), lambda v: f"{100 * v:.0f}", "")
    cb = fig.colorbar(im, cax=fig.add_subplot(outer[1, 1]))
    cb.set_label("% of replicates", fontsize=6.5)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["0", "50", "100"])
    cb.outline.set_visible(False)
    title = ("Usable evidence retained (log likelihood ratio)" if has_llr
             else "Evidence retained (points; log-LR version after the rerun)")
    im = _heat_block(fig, outer[2, 0], df, llr_retention_by_cell, 1.0, title, _letter(2), lambda v: f"{100 * v:.0f}", "")
    cb = fig.colorbar(im, cax=fig.add_subplot(outer[2, 1]))
    cb.set_label("% of the truth's evidence", fontsize=6.5)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["0", "50", "100"])
    cb.outline.set_visible(False)
    return _save(fig, out_path)


# ── S4: overstated evidence ──────────────────────────────────────────────────────────────────────────────────────

METRICS = [  # (title, ExCALIBR columns, acmgscaler columns)
    ("Points, Tavtigian", ("fit_vs_truth_pathogenic_tier_over", "fit_vs_truth_benign_tier_over"),
     ("acmg_vs_truth_pathogenic_tier_over", "acmg_vs_truth_benign_tier_over")),
    ("Points, lp_anchored", ("lp_fit_vs_truth_pathogenic_tier_over", "lp_fit_vs_truth_benign_tier_over"),
     ("lp_acmg_vs_truth_pathogenic_tier_over", "lp_acmg_vs_truth_benign_tier_over")),
    ("Posterior", ("fit_pathogenic_post_over_mean", "fit_benign_post_over_mean"),
     ("acmg_pathogenic_post_over_mean", "acmg_benign_post_over_mean")),
]


def _mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def _over(d, cols):
    return 100 * (d[cols[0]] + d[cols[1]]) / 2


def make_overstated_figure(merged, out_path):
    fig = plt.figure(figsize=(7.2, 7.4))
    outer = fig.add_gridspec(5, 3, height_ratios=[0.95, 0.2, 1, 0.2, 1], hspace=0.38, wspace=0.34,
                             left=0.08, right=0.98, top=0.965, bottom=0.07)
    a = fig.add_subplot(outer[0, 0:2])
    sub = a.get_subplotspec().subgridspec(1, 2, wspace=0.25)
    a.remove()
    ax_m = fig.add_subplot(sub[0, 0])
    ax_m.axis("off")
    ax_m.set_title(f"{_letter(0)} Compared methods", loc="left", color=INK, fontweight="bold", pad=2)
    for i, (n, d1, d2, col) in enumerate((("ExCALIBR", "fits P/LP, B/LB, population", "and synonymous scores;",
                                           EXC),
                                          ("acmgscaler", "kernel densities of P/LP and", "B/LB scores only;", ACM))):
        y = 0.8 - i * 0.48
        ax_m.add_patch(Rectangle((0, y - 0.07), 0.06, 0.14, color=col, transform=ax_m.transAxes))
        ax_m.text(0.1, y, n, fontsize=6.6, fontweight="bold", color=INK, va="center", transform=ax_m.transAxes)
        ax_m.text(0.1, y - 0.15, f"{d1}\n{d2}\n" + ("prior estimated from the population" if i == 0
                                                     else "prior fixed at 0.10"), fontsize=5.5, color=MUTED,
                  va="top", transform=ax_m.transAxes)
    _cartoon_scores(fig.add_subplot(sub[0, 1]), kappas=(2.5,), gamma_note=False)
    _cartoon_overstated(fig.add_subplot(outer[0, 2]))

    m10 = merged[(merged["experiment"] == "core") & np.isclose(merged["pi"], 0.10)]
    pr = merged[merged["experiment"].isin(["core", "prior"]) & (merged["richness"] == "mid") &
                merged["n_plp"].isin([20, 80]) & merged["kappa"].isin([2.5, 4.0]) & merged["gamma"].isin([0.25, 1.0])]
    for j, (title, exc_cols, acm_cols) in enumerate(METRICS):
        _cartoon_counts(fig.add_subplot(outer[1, j]))
        ax = fig.add_subplot(outer[2, j])
        for cols, col, lab, need in ((exc_cols, EXC, "ExCALIBR", False), (acm_cols, ACM, "acmgscaler", True)):
            if not all(c in m10 for c in cols):
                continue
            ys = []
            for n in N_ORDER:
                g = m10[m10["n_plp"] == n]
                if need:
                    g = g[g["acmg_ok"]]
                ys.append((n, *_mean_ci(_over(g, cols))) if len(g) else (n, np.nan, np.nan))
            x, y, e = (np.array(v, float) for v in zip(*ys))
            ok = np.isfinite(y)
            ax.errorbar(np.arange(len(x))[ok], y[ok], yerr=e[ok], color=col, marker="o", ms=3.5, lw=1.3, capsize=0,
                        label=lab)
        ax.set_xticks(range(len(N_ORDER)))
        ax.set_xticklabels(N_ORDER)
        ax.set_ylim(bottom=0)
        _style(ax, "controls per class", "variants overstated (%)" if j < 2 else "posterior excess (pp)")
        _title(ax, 1 + j, f"{title}, true prior 0.10")
        if j == 0:
            ax.legend(frameon=False, fontsize=5.8, loc="upper right")
            ax.text(0.03, 0.42, "acmgscaler\nneeds ≥ 10", fontsize=5.2, color=MUTED, transform=ax.transAxes, va="top")
        _cartoon_priors(fig.add_subplot(outer[3, j]))
        ax = fig.add_subplot(outer[4, j])
        for cols, base, light, lab, need in ((exc_cols, EXC, EXC_LIGHT, "ExCALIBR", False),
                                             (acm_cols, ACM, ACM_LIGHT, "acmgscaler", True)):
            if not all(c in pr for c in cols):
                continue
            for n, col in ((20, light), (80, base)):
                g0 = pr[pr["n_plp"] == n]
                if need:
                    g0 = g0[g0["acmg_ok"]]
                st = [(pi, *_mean_ci(_over(g0[np.isclose(g0["pi"], pi)], cols))) for pi in sorted(g0["pi"].unique())]
                if not st:
                    continue
                x, y, e = (np.array(v, float) for v in zip(*st))
                ax.errorbar(x, y, yerr=e, color=col, marker="o", ms=3.2, lw=1.2, capsize=0, label=f"{lab}, {n}")
        ax.axvline(0.10, color=MUTED, ls="--", lw=0.7)
        ax.set_ylim(bottom=0)
        _style(ax, "true prior", "variants overstated (%)" if j < 2 else "posterior excess (pp)")
        _title(ax, 4 + j, f"{title}, by true prior")
        if j == 0:
            ax.legend(frameon=False, fontsize=5.4, loc="upper center", ncol=2, title="method, controls per class",
                      title_fontsize=5.4)
    return _save(fig, out_path)


# ── S5: posterior calibration ────────────────────────────────────────────────────────────────────────────────────

def _point_posteriors(prior, mapping):
    from src.assay_calibration.fit_utils.fit import thresholds_from_prior
    pts = [1, 2, 3, 4, 5, 6, 7, 8]
    lp, lb, _ = thresholds_from_prior(prior, pts, acmg_mapping_method=mapping)
    post = lambda lr: prior * lr / (prior * lr + 1 - prior)
    return dict(zip(pts, post(np.asarray(lp)))), dict(zip(pts, post(np.asarray(lb))))


def _mark_points(ax, prior, side, xlim, mapping="tavtigian"):
    """Dotted lines at the posteriors that +1, +2, +4, +6 and +8 points (or their benign counterparts) correspond to
    at `prior`. At a prior of 0.10 the Tavtigian and lp_anchored mappings coincide."""
    p, b = _point_posteriors(prior, mapping)
    marks = p if side == "path" else b
    for k, v in marks.items():
        if xlim[0] <= v <= xlim[1] and k in (1, 2, 4, 6, 8):
            ax.axvline(v, color=MUTED, lw=0.5, ls=":")
            ax.text(v, 0.02, f"{'+' if side == 'path' else '−'}{k}", transform=ax.get_xaxis_transform(),
                    fontsize=5.2, color=INK, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))


def truth_called(cells, n=40000, threshold=0.9):
    """Share of population variants whose TRUE posterior is at least `threshold`, averaged over the given rows'
    settings (exact up to Monte Carlo error with n draws per setting)."""
    from analysis.calibration_quality import generate as G
    out = []
    for cid, g in cells.groupby("cell_id"):
        c = g.iloc[0].to_dict()
        rng = np.random.default_rng(zlib.crc32(cid.encode()))
        x = G.sample_draws(rng, c, 2, n)
        lr = np.exp(G.true_log_lr(x, c))
        post = c["pi"] * lr / (c["pi"] * lr + 1 - c["pi"])
        out.append(((post >= threshold).mean(), len(g)))
    v, w = zip(*out)
    return float(np.average(v, weights=w))


def make_posterior_figure(merged, out_path):
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42, left=0.08, right=0.98, top=0.94, bottom=0.09)
    base = merged[(merged["experiment"] == "core") & np.isclose(merged["pi"], 0.10)]
    series = []
    for n, (ce, ca) in ((10, (EXC_LIGHT, ACM_LIGHT)), (80, (EXC, ACM))):
        b = base[base["n_plp"] == n]
        series += [("fit", ce, f"ExCALIBR, {n}", b), ("acmg", ca, f"acmgscaler, {n}", b[b["acmg_ok"]])]

    ax = fig.add_subplot(gs[0, 0])
    for pre, col, lab, d in series:
        pc = precision_curve(d, pre)
        if pc is None or not len(d):
            continue
        s = pc[pc["t"] >= 0.8]
        ax.plot(s["t"], 100 * s["ppv"], marker="o", ms=3, lw=1.3, color=col, label=lab)
    ax.plot([0.8, 1], [80, 100], color=MUTED, lw=0.8, ls="--")
    _style(ax, "posterior threshold t", "pathogenic among posterior ≥ t (%)")
    ax.set_xlim(0.79, 1.0)
    ax.set_ylim(70, 101)
    _mark_points(ax, 0.10, "path", (0.79, 1.0))
    ax.legend(frameon=False, fontsize=5.2, loc="lower right", title="method, controls per class", title_fontsize=5.2)
    _title(ax, 0, "Pathogenic calls, prior 0.10")

    ax = fig.add_subplot(gs[0, 1])
    for pre, col, lab, d in series:
        pc = precision_curve(d, pre)
        if pc is None or not len(d):
            continue
        s = pc[pc["t"] <= 0.1]
        ax.plot(s["t"], 100 * s["npv"], marker="o", ms=3, lw=1.3, color=col)
    tt = np.geomspace(1e-4, 0.1, 50)
    ax.plot(tt, 100 * (1 - tt), color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log")
    _style(ax, "posterior threshold t", "benign among posterior ≤ t (%)")
    ax.set_xlim(5e-4, 0.12)
    ax.set_ylim(85, 100.5)
    _mark_points(ax, 0.10, "ben", (5e-4, 0.12))
    _title(ax, 1, "Benign calls, prior 0.10")

    ax = fig.add_subplot(gs[0, 2])
    pr = merged[merged["experiment"].isin(["core", "prior"]) & (merged["richness"] == "mid") &
                merged["n_plp"].isin([20, 80]) & merged["kappa"].isin([2.5, 4.0]) & merged["gamma"].isin([0.25, 1.0])]
    for pre, (light, dark), lab in (("fit", (EXC_LIGHT, EXC), "ExCALIBR"), ("acmg", (ACM_LIGHT, ACM), "acmgscaler")):
        for n, col in ((20, light), (80, dark)):
            d0 = pr[pr["n_plp"] == n]
            if pre == "acmg":
                d0 = d0[d0["acmg_ok"]]
            ys = []
            for pi in sorted(d0["pi"].unique()):
                pc = precision_curve(d0[np.isclose(d0["pi"], pi)], pre)
                if pc is not None:
                    ys.append((pi, 100 * float(pc.loc[np.isclose(pc["t"], 0.9), "ppv"].iloc[0])))
            if ys:
                x, y = zip(*ys)
                ax.plot(x, y, marker="o", ms=3.5, lw=1.3, color=col, label=f"{lab}, {n}")
    ax.axhline(90, color=MUTED, ls="--", lw=0.8)
    ax.axvline(0.10, color=MUTED, ls=":", lw=0.7)
    _style(ax, "true prior", "pathogenic among posterior ≥ 0.9 (%)")
    ax.legend(frameon=False, fontsize=5.2, loc="lower right")
    _title(ax, 2, "Precision at 0.9")

    # (D) call rate: share of population variants with posterior >= 0.9, vs the truth's
    ax = fig.add_subplot(gs[1, 0])
    for pre, col, lab in (("fit", EXC, "ExCALIBR"), ("acmg", ACM, "acmgscaler"), ("truth", INK, "truth")):
        ys = []
        for pi in sorted(pr["pi"].unique()):
            s = pr[np.isclose(pr["pi"], pi)]
            if pre == "truth":
                v = 100 * truth_called(s)
            else:
                s2 = s[s["acmg_ok"]] if pre == "acmg" else s
                pc = precision_curve(s2, pre)
                v = 100 * float(pc.loc[np.isclose(pc["t"], 0.9), "called_path"].iloc[0]) if pc is not None else np.nan
            ys.append((pi, v))
        x, y = zip(*ys)
        ax.plot(x, y, marker="o", ms=3.5, lw=1.3, color=col, label=lab, ls="--" if pre == "truth" else "-")
    _style(ax, "true prior", "population variants called (%)")
    ax.legend(frameon=False, fontsize=5.4, loc="upper left")
    _title(ax, 3, "Posterior ≥ 0.9 calls")

    # (E, F) precision at 0.9 by controls per class at true priors 0.05 and 0.25
    for j, pi in enumerate((0.05, 0.25)):
        ax = fig.add_subplot(gs[1, 1 + j])
        s = merged[merged["experiment"].isin(["core", "prior"]) & np.isclose(merged["pi"], pi)]
        for pre, col, lab in (("fit", EXC, "ExCALIBR"), ("acmg", ACM, "acmgscaler")):
            ys = []
            for n in N_ORDER:
                g = s[s["n_plp"] == n]
                if pre == "acmg":
                    g = g[g["acmg_ok"]]
                if not len(g):
                    continue
                pc = precision_curve(g, pre)
                if pc is not None:
                    ys.append((N_ORDER.index(n), 100 * float(pc.loc[np.isclose(pc["t"], 0.9), "ppv"].iloc[0])))
            if ys:
                x, y = zip(*ys)
                ax.plot(x, y, marker="o", ms=3.5, lw=1.3, color=col, label=lab)
        ax.axhline(90, color=MUTED, ls="--", lw=0.8)
        ax.set_xticks(range(len(N_ORDER)))
        ax.set_xticklabels(N_ORDER)
        _style(ax, "controls per class", "pathogenic among posterior ≥ 0.9 (%)" if j == 0 else "")
        _title(ax, 4 + j, f"By controls, true prior {pi:g}")
    return _save(fig, out_path)


# ── S6: diagnostics ──────────────────────────────────────────────────────────────────────────────────────────────

def _replicate_retention(d):
    """Per-replicate evidence retained: capped log-LR where available, else points."""
    if "fit_pathogenic_llr_rep_abs_dir_cap" in d and d["fit_pathogenic_llr_rep_abs_dir_cap"].notna().any():
        num = d["fit_pathogenic_llr_rep_abs_dir_cap"] + d["fit_benign_llr_rep_abs_dir_cap"]
        den = d["fit_pathogenic_llr_true_abs_cap"] + d["fit_benign_llr_true_abs_cap"]
    else:
        num = d["fit_vs_truth_pathogenic_mean_abs"] + d["fit_vs_truth_benign_mean_abs"]
        den = d["truth_pathogenic_mean_abs"] + d["truth_benign_mean_abs"]
    return num / den.where(den > 0.5)


def make_diagnostics_figure(df, out_path, sd_real=None):
    d = df[~df["experiment"].isin(["mech_bias", "misfit"])].copy()
    d["retention"] = _replicate_retention(d)
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1], hspace=0.55, wspace=0.42, left=0.08, right=0.98, top=0.94,
                          bottom=0.12)
    ax = fig.add_subplot(gs[0, :])
    _cartoon_priors_boot(ax)
    _title(ax, 0, "What the bootstrap prior spread shows", pad=2)

    ax = fig.add_subplot(gs[1, 0])
    bins = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0]
    lab = ["<0.02", "0.02–0.05", "0.05–0.1", "0.1–0.15", "0.15–0.2", "0.2–0.3", ">0.3"]
    b = pd.cut(d["fit_prior_sd"], bins, labels=lab, include_lowest=True)
    share = d.groupby(b, observed=False)["prior_bad"].mean() * 100
    ax.bar(range(len(lab)), share.to_numpy(), color=EXC, width=0.7)
    for i, s in enumerate(share):
        ax.text(i, s + 1.5, f"{s:.0f}", ha="center", fontsize=5.4, color=INK)
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=5.6)
    ax.set_ylim(0, 100)
    _style(ax, "SD of the prior across bootstraps", "prior > 10 pp off (% of replicates)")
    _title(ax, 1, "Unreliable prior")

    ax = fig.add_subplot(gs[1, 1])
    pb = [0, 0.05, 0.15, 0.3, 0.5, 1.01]
    plab = ["<0.05", "0.05–0.15", "0.15–0.3", "0.3–0.5", "≥0.5"]
    bp = pd.cut(d["fit_prior"], pb, labels=plab, right=False)
    low = (d["retention"] < 0.25).astype(float).where(d["retention"].notna())
    s1 = d.assign(low=low).groupby(bp, observed=False)["low"].mean() * 100
    ax.bar(range(len(plab)), s1.to_numpy(), color=EXC, width=0.7)
    for i, s in enumerate(s1):
        ax.text(i, (s if np.isfinite(s) else 0) + 1.5, f"{s:.0f}" if np.isfinite(s) else "", ha="center",
                fontsize=5.4, color=INK)
    ax.set_xticks(range(len(plab)))
    ax.set_xticklabels(plab, rotation=35, ha="right", fontsize=5.6)
    ax.set_ylim(0, 100)
    _style(ax, "estimated prior", "< 25% of the truth's evidence (% of replicates)")
    _title(ax, 2, "Little usable evidence")

    ax = fig.add_subplot(gs[1, 2])
    if sd_real is not None and len(sd_real):
        v = np.asarray(list(sd_real), float)
        ax.hist(v, bins=np.linspace(0, max(0.35, v.max() + 0.01), 25), color=EXC)
        ax.axvline(0.10, color=MUTED, ls="--", lw=0.8)
        ax.text(0.105, ax.get_ylim()[1] * 0.9, f"{int((v > 0.10).sum())} of {len(v)}\nabove 0.10", fontsize=5.6,
                color=INK)
        _style(ax, "approximate SD of the bootstrap priors", "real calibrations")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "real calibrations: drawn in the notebook", ha="center", fontsize=6, color=MUTED)
    _title(ax, 3, "Real calibrations")
    return _save(fig, out_path)


# ── S7: misfit and Yang distance ─────────────────────────────────────────────────────────────────────────────────

def make_misfit_figure(df, examples, out_path):
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.55, wspace=0.38, left=0.08, right=0.98, top=0.93,
                          bottom=0.1)
    for j, ex in enumerate(examples[:3]):
        ax = fig.add_subplot(gs[0, j])
        s = ex["samples"]["Pathogenic/Likely Pathogenic"]
        x = np.asarray(ex["x"])
        ax.hist(s["scores"], bins=np.linspace(x.min(), x.max(), 30), density=True, color="#d7dbe0", label="observed")
        ax.fill_between(x, s["fit_lo"], s["fit_hi"], color=EXC, alpha=0.2, lw=0)
        ax.plot(x, s["fit"], color=EXC, lw=1.3, label="fit (median, 5–95%)")
        ax.plot(x, s["true"], color=INK, lw=1.0, ls="--", label="truth")
        yv = s["yang_val"]
        ax.text(0.03, 0.97, f"in-sample {s['yang_train']:.3f}\nheld out {'n/a' if yv != yv else f'{yv:.3f}'}\n"
                f"vs truth {s['yang_true']:.3f}", transform=ax.transAxes, fontsize=5.4, va="top", color=INK)
        ax.set_yticks([])
        ax.set_xlim(-10, 4)
        _style(ax, "score", "P/LP density" if j == 0 else "")
        short = {"well fit": "Well fit", "overfit": "Overfit (5 controls per class)", "underfit": "Underfit (two modes)"}
        _title(ax, j, next((v for k, v in short.items() if ex["name"].startswith(k)), ex["name"]))
        if j == 0:
            ax.legend(frameon=False, fontsize=5.2, loc="upper right")

    d = df.copy()
    d["regime"] = np.where(d["experiment"] == "misfit", "misspecified truth",
                           np.where(d["n_plp"] <= 10, "correct, ≤ 10 controls", "correct, ≥ 20 controls"))
    cols = {"correct, ≥ 20 controls": "#b8bec7", "correct, ≤ 10 controls": EXC, "misspecified truth": ACM}
    ax = fig.add_subplot(gs[1, 0])
    for r, col in cols.items():
        g = d[d["regime"] == r].sample(min(2500, (d["regime"] == r).sum()), random_state=0)
        ax.scatter(g["yang_train_plp"], g["yang_val_plp"], s=3, color=col, lw=0, alpha=0.6, label=r)
    lim = [0.005, 1]
    ax.plot(lim, lim, color=MUTED, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.axhline(0.2, color=MUTED, lw=0.7, ls="--")
    _style(ax, "in-sample Yang distance (P/LP)", "held-out Yang distance (P/LP)")
    ax.legend(frameon=False, fontsize=5.2, loc="upper left", markerscale=2.5)
    _title(ax, 3, "Overfit: held out ≫ in sample")

    has_true = "yang_true_plp" in d and d["yang_true_plp"].notna().any()
    d["over"] = 100 * (d["fit_vs_truth_pathogenic_over"] + d["fit_vs_truth_benign_over"]) / 2
    for j, (xcol, xl, t) in enumerate((("yang_val_plp", "held-out Yang distance (P/LP)", "Held-out distance"),
                                       ("yang_true_plp", "fitted vs true distance (P/LP)", "Distance to the truth"))):
        ax = fig.add_subplot(gs[1, 1 + j])
        if xcol == "yang_true_plp" and not has_true:
            ax.axis("off")
            ax.text(0.5, 0.5, "fitted-vs-true distance:\navailable after the rerun", ha="center", va="center",
                    fontsize=6.5, color=MUTED, transform=ax.transAxes)
            _title(ax, 4 + j, t)
            continue
        bins = [0, 0.02, 0.05, 0.1, 0.2, 10]
        blab = ["<0.02", "0.02–0.05", "0.05–0.1", "0.1–0.2", ">0.2"]
        for r, col in cols.items():
            g = d[d["regime"] == r]
            bb = pd.cut(g[xcol], bins, labels=blab)
            m = g.groupby(bb, observed=False)["over"].mean()
            ax.plot(range(len(blab)), m.to_numpy(), marker="o", ms=3.2, lw=1.2, color=col, label=r)
        ax.set_xticks(range(len(blab)))
        ax.set_xticklabels(blab, rotation=35, ha="right", fontsize=5.6)
        ax.set_ylim(bottom=0)
        _style(ax, xl, "variants overstated, points (%)" if j == 0 else "")
        _title(ax, 4 + j, t)
    return _save(fig, out_path)
