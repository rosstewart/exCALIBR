"""Supplementary figures of the calibration-quality simulation.

make_degradation_figure: share of replicates whose prior is more than 10 percentage points off (A) and the evidence
    retained relative to the truth (B), across the core grid: controls per class (rows) x separation (columns) in
    small multiples by the fraction of pathogenic variants the assay measures (gamma) and the population/synonymous
    sample size.
make_diagnostics_figure: overstated evidence for ExCALIBR vs acmgscaler (C, D), the prior diagnostics (E) and the
    Yang distance under deliberately misspecified truths (F).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

INK, MUTED, GRID = "#14171c", "#565d68", "#e6e8ec"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential, light->dark
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"]  # categorical slots 1-4, fixed order
RICH_LABEL = {"low": "population 50, synonymous 0", "mid": "population 200, synonymous 100",
              "high": "population 1000, synonymous 500"}
N_ORDER = [5, 10, 20, 40, 80]
K_ORDER = [1.5, 2.5, 4.0]
G_ORDER = [0.25, 0.5, 1.0]

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def _letter(i):
    return f"({'ABCDEFGHIJ'[i]})"


MIN_TRUTH_POINTS = 0.5  # cells where the truth itself gives less evidence per variant are not rated for retention


def _fit_points(df):
    return df["fit_vs_truth_pathogenic_mean_abs"] + df["fit_vs_truth_benign_mean_abs"]


def _truth_points(df):
    return df["truth_pathogenic_mean_abs"] + df["truth_benign_mean_abs"]


def retention_by_cell(g):
    """Evidence retained in a cell: total mean |points| of the fit over truly pathogenic and truly benign variants,
    summed over replicates, as a share of the truth's (a ratio of totals, stable where the truth gives little
    evidence); NaN where the truth averages under MIN_TRUTH_POINTS per class pair."""
    tru = _truth_points(g)
    return float(_fit_points(g).sum() / tru.sum()) if tru.mean() >= MIN_TRUTH_POINTS else np.nan


def _heat_block(fig, gs, df, cell_value, vmax, title, letter, fmt, label):
    core = df[df["experiment"] == "core"]
    sub = gs.subgridspec(3, 3, hspace=0.35, wspace=0.12)
    im = None
    for i, g in enumerate(G_ORDER):
        for j, r in enumerate(["low", "mid", "high"]):
            ax = fig.add_subplot(sub[i, j])
            t = (core[(core["gamma"] == g) & (core["richness"] == r)].groupby(["n_plp", "kappa"])
                 .apply(cell_value).unstack("kappa").reindex(index=N_ORDER, columns=K_ORDER))
            im = ax.imshow(t.to_numpy(float), cmap=SEQ, vmin=0, vmax=vmax, aspect="auto", origin="lower")
            for (a, b), v in np.ndenumerate(t.to_numpy(float)):
                ax.text(b, a, fmt(v) if np.isfinite(v) else "–", ha="center", va="center", fontsize=5.6,
                        color="white" if np.isfinite(v) and v > 0.55 * vmax else (INK if np.isfinite(v) else MUTED))
            ax.set_xticks(range(3))
            ax.set_xticklabels([f"{k:g}" for k in K_ORDER] if i == 2 else [])
            ax.set_yticks(range(5))
            ax.set_yticklabels(N_ORDER if j == 0 else [])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(RICH_LABEL[r], fontsize=6.2, color=MUTED, pad=3)
            if j == 0:
                ax.set_ylabel(f"γ = {g:g}\ncontrols per class", fontsize=6.3)
            if i == 2:
                ax.set_xlabel("separation κ", fontsize=6.3)
            if i == 0 and j == 0:
                ax.annotate(f"{letter} {title}", xy=(0, 1), xycoords="axes fraction", xytext=(-38, 22),
                            textcoords="offset points", fontsize=8, fontweight="bold", color=INK, va="bottom",
                            annotation_clip=False)
    return im


def make_degradation_figure(df, out_path):
    fig = plt.figure(figsize=(7.2, 8.6))
    outer = fig.add_gridspec(2, 2, width_ratios=[1, 0.035], hspace=0.3, wspace=0.04,
                             left=0.12, right=0.92, top=0.94, bottom=0.06)
    im1 = _heat_block(fig, outer[0, 0], df, lambda g: float(g["prior_bad"].mean()), 1.0,
                      "Prior more than 10 percentage points off", _letter(0), lambda v: f"{100 * v:.0f}", "")
    cb = fig.colorbar(im1, cax=fig.add_subplot(outer[0, 1]))
    cb.set_label("% of replicates", fontsize=6.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(["0", "25", "50", "75", "100"])
    cb.outline.set_visible(False)
    im2 = _heat_block(fig, outer[1, 0], df, retention_by_cell, 1.0, "Evidence retained relative to the truth",
                      _letter(1),
                      lambda v: f"{100 * v:.0f}", "")
    cb = fig.colorbar(im2, cax=fig.add_subplot(outer[1, 1]))
    cb.set_label("% of the truth's mean |points|", fontsize=6.5)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cb.set_ticklabels(["0", "25", "50", "75", "100"])
    cb.outline.set_visible(False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _over(d, pre):
    return (d[f"{pre}_vs_truth_pathogenic_tier_over"] + d[f"{pre}_vs_truth_benign_tier_over"]) / 2


def _mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def make_diagnostics_figure(merged, out_path, misfit=None, core_ref=None):
    """merged: results joined with the acmgscaler rows (summarize.acmgscaler_compare). misfit: results of the
    deliberately misspecified cells (optional panel F); core_ref: Gaussian replicates shown for reference in F."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), gridspec_kw=dict(hspace=0.5, wspace=0.32))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.12)
    m = merged[merged["experiment"].isin(["core", "prior"])]

    # (A) overstated evidence vs controls per class
    ax = axes[0, 0]
    for pre, col, lab in (("fit", CAT[0], "ExCALIBR"), ("acmg", CAT[1], "acmgscaler (prior 0.10)")):
        pts = []
        for n in N_ORDER:
            g = m[m["n_plp"] == n]
            if pre == "acmg":
                g = g[g["acmg_ok"]]
            mu, ci = _mean_ci(100 * _over(g, pre)) if len(g) else (np.nan, np.nan)
            pts.append((n, mu, ci))
        x, y, e = (np.array(v, float) for v in zip(*pts))
        ok = np.isfinite(y)
        ax.errorbar(np.arange(len(x))[ok], y[ok], yerr=e[ok], color=col, marker="o", ms=4, lw=1.4, capsize=0,
                    label=lab)
    ax.text(0, 4.2, "acmgscaler\nnot run (<10)", fontsize=5.6, color=MUTED, ha="center")
    ax.set_xticks(range(len(N_ORDER)))
    ax.set_xticklabels(N_ORDER)
    _style(ax, "controls per class", "variants with overstated evidence (%)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_title(f"{_letter(0)} Overstated evidence", loc="left", color=INK, fontweight="bold")

    # (B) overstated evidence vs true prior (same cells for every prior)
    ax = axes[0, 1]
    sel = m[(m["richness"] == "mid") & m["n_plp"].isin([20, 80]) & m["kappa"].isin([2.5, 4.0]) &
            m["gamma"].isin([0.25, 1.0])]
    for pre, col, lab in (("fit", CAT[0], "ExCALIBR"), ("acmg", CAT[1], "acmgscaler (prior 0.10)")):
        g0 = sel[sel["acmg_ok"]] if pre == "acmg" else sel
        stats = [(pi, *_mean_ci(100 * _over(g0[g0["pi"] == pi], pre))) for pi in sorted(g0["pi"].unique())]
        x, y, e = (np.array(v, float) for v in zip(*stats))
        ax.errorbar(x, y, yerr=e, color=col, marker="o", ms=4, lw=1.4, capsize=0, label=lab)
    ax.axvline(0.10, color=MUTED, ls="--", lw=0.8)
    ax.text(0.105, ax.get_ylim()[1] * 0.92, "acmgscaler's prior", fontsize=5.6, color=MUTED)
    _style(ax, "true prior", "variants with overstated evidence (%)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{_letter(1)} Overstated evidence by true prior", loc="left", color=INK, fontweight="bold")

    # (C) prior diagnostic: share of replicates with a prior > 10 pp off, by the bootstrap SD of the prior
    ax = axes[1, 0]
    d = merged[merged["experiment"] != "mech_bias"]
    bins = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0]
    lab = ["<0.02", "0.02–0.05", "0.05–0.1", "0.1–0.15", "0.15–0.2", "0.2–0.3", ">0.3"]
    b = pd.cut(d["fit_prior_sd"], bins, labels=lab, include_lowest=True)
    share = d.groupby(b, observed=False)["prior_bad"].mean() * 100
    count = d.groupby(b, observed=False).size()
    ax.bar(range(len(lab)), share.to_numpy(), color=CAT[0], width=0.7)
    for i, (s, c) in enumerate(zip(share, count)):
        ax.text(i, s + 1.5, f"{s:.0f}%\n(n={c:,})", ha="center", va="bottom", fontsize=5.4, color=INK,
                linespacing=1.1)
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    _style(ax, "SD of the prior across bootstrap iterations", "prior > 10 pp off (% of replicates)")
    ax.set_title(f"{_letter(2)} Prior spread flags unreliable priors", loc="left", color=INK, fontweight="bold")

    # (D) Yang distance under misspecified truths
    ax = axes[1, 1]
    if misfit is not None and len(misfit):
        if core_ref is not None:
            ax.scatter(core_ref["yang_val_max"], 100 * (core_ref["fit_vs_truth_pathogenic_over"] +
                       core_ref["fit_vs_truth_benign_over"]) / 2, s=3, color="#b8bec7", lw=0, alpha=0.5,
                       label="correctly specified (core grid)")
        for (shape, col) in zip(["skew", "bimodal", "heavy"], CAT[1:]):
            g = misfit[misfit["shape"] == shape]
            ax.scatter(g["yang_val_max"], 100 * (g["fit_vs_truth_pathogenic_over"] +
                       g["fit_vs_truth_benign_over"]) / 2, s=7, color=col, lw=0, alpha=0.8,
                       label={"skew": "skewed classes", "bimodal": "bimodal abnormal class",
                              "heavy": "heavy-tailed classes"}[shape])
        ax.axvline(0.2, color=MUTED, ls="--", lw=0.8)
        ax.legend(frameon=False, fontsize=5.6, loc="upper left", markerscale=1.6)
    _style(ax, "Yang distance, max over samples (held out)", "variants with overstated evidence (%)")
    ax.set_title(f"{_letter(3)} Misfit and Yang distance", loc="left", color=INK, fontweight="bold")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path
