"""Simplified supplementary figures of the calibration-quality simulation.

    S3  make_fewer_data_figure    fewer data: less evidence, not wrong evidence (true prior 0.25; gamma = 1 first;
                                  controls per class on x, one line per gnomAD/synonymous sample size), then by separation and
                                  by the measured fraction gamma
    S4  make_methods_figure       ExCALIBR vs acmgscaler: overconfidence by controls per class and by true prior,
                                  precision of confident pathogenic calls by true prior
    S5  make_reliability_figure   calibration quality metrics: the prior reliability score (how it is computed, its
                                  calibration in the simulation, the real calibrations) and the fit-quality check
                                  (largest in-sample Yang distance) on the real calibrations
    S2+ make_prior_scarcity_panel prior error (percentage points) when every sample is scarce, for the prior-simulation
                                  figure

Panels that will use the per-variant log-LR error once it is in the results (analysis/calibration_quality/run.py)
take their metric as an argument (METRICS); until then they show the posterior error, which is stored.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from analysis.calibration_quality.figures import GRID, INK, MUTED, N_ORDER, _style, precision_curve
from analysis.calibration_quality.figures_v2 import _save, _title
# Manuscript palette (analysis/figure4/panels.py SAMPLE_COLORS, STRENGTH_COLOR, CARTOON_FIT_COLORS, CMAP; methods as
# analysis/comparison_diagnostics: tab10 in canonical order, ExCALIBR first, acmgscaler third)
PLP, BLB, GNOMAD, SYN = "#CA7682", "#1D7AAB", "#A0A0A0", "#6BAA75"
PATH_C, UNMEAS = "#B1535F", "#e6b1b8"  # pathogenic, measured / unmeasured mechanism (pink strength ramp)
PATH_RAMP = {4.0: "#943744", 2.5: "#ca7682", 1.5: "#e6b1b8"}
BENIGN = BLB
FIT_DARK = "#8B3A47"  # fitted curve over P/LP scores, as in the manuscript's sample-fit cartoons
LEVELS = ("#c49a5a", "#8b6fb0", "#3b2d52")  # tan -> purple, the ends of the manuscript heatmap colormap

EXC, ACM = "#1f77b4", "#2ca02c"
EXC_LIGHT, ACM_LIGHT = "#aec7e8", "#98df8a"
RICH = [("low", "50, 0", LEVELS[0]), ("mid", "200, 100", LEVELS[1]), ("high", "1000, 500", LEVELS[2])]
GNOMAD_SYN = "population, synonymous variants"
MAIN_KAPPA = 4.0  # as in the prior simulation (Section 3) and Figure 1; 2.5 and 1.5 shown alongside
MAIN_PI = 0.25  # the prior simulation (Figures S1/S2) uses 0.25

# metric -> (per-replicate value from columns {pre}_{class}_..., y label, scale); the log-LR error metrics slot in
# here once run.py stores them
METRICS = {
    "post_over": (lambda d, pre: (d[f"{pre}_pathogenic_post_over_mean"] + d[f"{pre}_benign_post_over_mean"]) / 2,
                  "mean overshoot per variant (pp)", 100),
    "post_over_share": (lambda d, pre: (d[f"{pre}_pathogenic_post_over"] + d[f"{pre}_benign_post_over"]) / 2,
                        "variants overshooting by > 1 pp (%)", 100),
    # from the S3 local rerun (run.py post_over_5/_10/_any; merge with merge_overshoot_rerun first)
    "post_over_share5": (lambda d, pre: (d[f"{pre}_pathogenic_post_over_5"] + d[f"{pre}_benign_post_over_5"]) / 2,
                         "variants overshooting by > 5 pp (%)", 100),
    "post_over_share10": (lambda d, pre: (d[f"{pre}_pathogenic_post_over_10"] + d[f"{pre}_benign_post_over_10"]) / 2,
                          "variants overshooting by > 10 pp (%)", 100),
    "post_over_any": (lambda d, pre: (d[f"{pre}_pathogenic_post_over_any"] + d[f"{pre}_benign_post_over_any"]) / 2,
                      "variants overshooting at all (%)", 100),
    "post_mae": (lambda d, pre: (d[f"{pre}_pathogenic_post_mae"] + d[f"{pre}_benign_post_mae"]) / 2,
                 "mean |posterior error| (percentage points)", 100),
}


def merge_overshoot_rerun(df, rerun_csv):
    """Add the rerun's new overshoot columns (post_over_any/_5/_10) to `df` by (cell_id, rep); rows the rerun did not
    cover get NaN. Stored columns are taken from `df` (the rerun reproduces them)."""
    r = pd.read_csv(rerun_csv, low_memory=False)
    new = [c for c in r.columns if c.endswith(("post_over_any", "post_over_5", "post_over_10"))]
    return df.drop(columns=[c for c in new if c in df]).merge(r[["cell_id", "rep"] + new], on=["cell_id", "rep"],
                                                               how="left")


def llr_retention(d, pre="fit"):
    """Per-replicate share of the truth's capped |log LR| that the reported LR keeps in the truth's direction."""
    num = d[f"{pre}_pathogenic_llr_rep_abs_dir_cap"] + d[f"{pre}_benign_llr_rep_abs_dir_cap"]
    den = d[f"{pre}_pathogenic_llr_true_abs_cap"] + d[f"{pre}_benign_llr_true_abs_cap"]
    return num / den.where(den > 0)


def _mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


CI_STYLE = "bars"  # "bars": error bars on the points; "band": shaded 95% CI band behind the line


def _line(ax, xs, ys, es, col, lab, **kw):
    xs, ys, es = (np.asarray(v, float) for v in (xs, ys, es))
    if CI_STYLE == "band":
        ok = np.isfinite(ys) & np.isfinite(es)
        ax.fill_between(xs[ok], np.clip(ys[ok] - es[ok], 0, None), ys[ok] + es[ok], color=col, alpha=0.18, lw=0,
                        zorder=1)
        ax.plot(xs, ys, color=col, marker="o", ms=3, lw=1.2, label=lab, zorder=2, **kw)
    else:
        ax.errorbar(xs, ys, yerr=es, color=col, marker="o", ms=3, lw=1.2, capsize=1.5, elinewidth=0.7, label=lab,
                    **kw)


def _controls_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(N_ORDER)
    ax.set_xticklabels([str(n) for n in N_ORDER])
    ax.minorticks_off()


def _core(df, gamma=1.0, kappa=MAIN_KAPPA):
    return df[(df["experiment"] == "core") & np.isclose(df["gamma"], gamma) & np.isclose(df["kappa"], kappa)]


def _by_controls(ax, d, value, scale=1.0):
    for r, lab, col in RICH:
        g = d[d["richness"] == r]
        st = [(n, *_mean_ci(scale * value(g[g["n_plp"] == n]))) for n in N_ORDER]
        _line(ax, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st], col, lab)
    _controls_axis(ax)


# ── design cartoons (S3A) ──────────────────────────────────────────────────────────────────────────────────

def _design_scores(ax):
    """Benign N(0, 1) and measured pathogenic N(-kappa, 1) scores for the three separations; labels above peaks."""
    x = np.linspace(-7.5, 3.5, 400)
    pdf = lambda m: np.exp(-0.5 * (x - m) ** 2) / np.sqrt(2 * np.pi)
    ax.fill_between(x, pdf(0), color=BENIGN, alpha=0.18, lw=0)
    ax.plot(x, pdf(0), color=BENIGN, lw=1.2)
    for k, a, yl in ((4.0, 1.0, 0.43), (2.5, 0.65, 0.49), (1.5, 0.35, 0.55)):  # staggered labels
        ax.plot(x, pdf(-k), color=PATH_RAMP[k], lw=1.3)
        ax.text(-k, yl, f"$\\kappa={k:g}$", ha="center", va="bottom", fontsize=5.6, color=PATH_RAMP[k])
    ax.text(0.3, 0.43, "$N(0,1)$", ha="left", va="bottom", fontsize=5.6, color=BENIGN)
    # pathogenic variants acting through unmeasured mechanisms (1 - gamma) score like benign ones
    ax.plot(x, 0.92 * pdf(0), color=UNMEAS, lw=1.1, ls=(0, (3, 2)))
    ax.text(2.2, 0.27, "$1-\\gamma$", ha="left", va="center", fontsize=7, color=UNMEAS)
    ax.annotate("", xy=(0.95, 0.24), xytext=(2.15, 0.27), arrowprops=dict(arrowstyle="-", color=UNMEAS, lw=0.6))
    ax.set_ylim(0, 0.62)
    ax.set_xlim(-7.5, 3.5)
    ax.set_yticks([])
    ax.set_xticks([-6, -4, -2, 0, 2])
    ax.tick_params(labelsize=5.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("assay score", fontsize=6.2)


def _design_samples(ax):
    """Composition of the four samples (pathogenic red, benign grey) with their simulated sizes."""
    rows = [("P/LP", 1.0, "5–80"), ("B/LB", 0.0, "5–80"), ("population", 0.25, "50 / 200 / 1000"),
            ("synonymous", 0.0, "0 / 100 / 500")]
    g = 0.6  # drawn measured fraction gamma
    ys = [3.3, 2.3, 1.0, 0.0]  # extra gap above the population bar for the P(Y = 1) bracket
    for (lab, frac, n), y in zip(rows, ys):
        base = {"P/LP": PLP, "B/LB": BLB, "population": GNOMAD, "synonymous": SYN}[lab]
        ax.barh(y, g * frac, color=PATH_C, height=0.6)
        ax.barh(y, (1 - g) * frac, left=g * frac, color=UNMEAS, height=0.6)
        ax.barh(y, 1 - frac, left=frac, color=base, height=0.6, alpha=0.85)
        if lab == "P/LP":
            ax.text(g / 2, y, "$\\gamma$", ha="center", va="center", fontsize=6.5, color="white")
            ax.text(g + (1 - g) / 2, y, "$1-\\gamma$", ha="center", va="center", fontsize=6.0, color=INK)
        ax.text(-0.04, y, lab, ha="right", va="center", fontsize=6.0, color=INK)
        ax.text(1.04, y, n, ha="left", va="center", fontsize=5.6, color=MUTED)
    # bracket over the whole pathogenic part of gnomAD (measured and unmeasured): P(Y = 1)
    yb = 1 + 0.36
    ax.plot([0, 0, 0.25, 0.25], [yb - 0.05, yb, yb, yb - 0.05], color=PATH_C, lw=0.7, clip_on=False)
    ax.text(0.125, yb + 0.03, "$P(Y=1)$", ha="center", va="bottom", fontsize=5.2, color=PATH_C)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-0.6, ys[0] + 0.6)
    ax.axis("off")


# ── S3 ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def make_fewer_data_figure(df, out_path, metric="post_over", pi=MAIN_PI):
    """True prior `pi` (0.25, as in the prior simulation), core grid. (A) design; (B) log-LR retained and (C) `metric`
    by controls per class, gamma = 1, kappa = MAIN_KAPPA, one line per gnomAD/synonymous size; (D) log-LR retained by
    separation kappa (gamma = 1) and (E) by the measured fraction gamma (kappa = MAIN_KAPPA), both with 200 gnomAD and
    100 synonymous variants."""
    value, ylab, scale = METRICS[metric]
    fig = plt.figure(figsize=(7.2, 9.4))
    outer = fig.add_gridspec(4, 1, height_ratios=[0.5, 1, 1, 1], hspace=0.5, left=0.09, right=0.98, top=0.97,
                             bottom=0.05)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 1.1], wspace=0.35)
    a1, a2 = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    _design_scores(a1)
    _title(a1, 0, "Design")
    _design_samples(a2)

    core = df[(df["experiment"] == "core") & np.isclose(df["pi"], pi)]
    d = _core(core)
    mid = outer[1].subgridspec(1, 2, wspace=0.3)
    ax = fig.add_subplot(mid[0, 0])
    _by_controls(ax, d, lambda g: llr_retention(g), 100)
    ax.set_ylim(0, 100)
    _style(ax, "controls per class (P/LP and B/LB)", "evidence retained (% of the truth's log LR)")
    _title(ax, 1, f"Evidence retained (κ = {MAIN_KAPPA:g}, γ = 1)")
    ax.legend(frameon=False, fontsize=5.3, loc="lower right", title=GNOMAD_SYN, title_fontsize=5.5)
    ax = fig.add_subplot(mid[0, 1])
    _by_controls(ax, d, lambda g: value(g, "fit"), scale)
    ax.set_ylim(0, max(1, ax.get_ylim()[1]))  # at least 0-1; wider if the metric needs it
    _style(ax, "controls per class (P/LP and B/LB)", ylab)
    _title(ax, 2, f"Overconfidence (κ = {MAIN_KAPPA:g}, γ = 1)")
    ax.legend(frameon=False, fontsize=5.3, loc="upper right", title=GNOMAD_SYN, title_fontsize=5.5)

    # rows 3-4: by separation (gamma = 1) and by measured fraction (kappa = MAIN_KAPPA); columns: retained | overconf.
    m = core[core["richness"] == "mid"]
    factors = (("kappa", [1.5, 2.5, 4.0], ("gamma", 1.0), "separation κ", "separation (γ = 1)"),
               ("gamma", [0.25, 0.5, 1.0], ("kappa", MAIN_KAPPA), "measured fraction γ",
                f"measured fraction (κ = {MAIN_KAPPA:g})"))
    for i, (col, levels, fixed, lab, what) in enumerate(factors):
        row = outer[2 + i].subgridspec(1, 2, wspace=0.3)
        g0 = m[np.isclose(m[fixed[0]], fixed[1])]
        for j, (metric_fn, sc, ylim, ytxt, head) in enumerate((
                (lambda g: llr_retention(g), 100, (0, 100), "evidence retained (%)", "Evidence retained"),
                (lambda g: value(g, "fit"), scale, None, ylab, "Overconfidence"))):
            ax = fig.add_subplot(row[0, j])
            for lv, c in zip(levels, LEVELS):
                g = g0[np.isclose(g0[col], lv)]
                st = [(n, *_mean_ci(sc * metric_fn(g[g["n_plp"] == n]))) for n in N_ORDER]
                _line(ax, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st], c, f"{lv:g}")
            _controls_axis(ax)
            ax.set_ylim(*ylim) if ylim else ax.set_ylim(bottom=0)
            _style(ax, "controls per class", ytxt)
            _title(ax, 3 + 2 * i + j, f"{head} by {what}")
            ax.legend(frameon=False, fontsize=5.3, loc="lower right" if j == 0 else "upper right", title=lab,
                      title_fontsize=5.5)
    return _save(fig, out_path)


# ── S2 addition: prior error when every sample is scarce ───────────────────────────────────────────────────────

def prior_scarcity_stats(df, pi=MAIN_PI):
    """Tidy table for the S2 joint-scarcity panel: mean |fitted - true prior| (probability units) and 95% CI half-width
    by controls per class (n_plp = n_blb) and population/synonymous level; core grid, gamma = 1, kappa = MAIN_KAPPA."""
    d = _core(df[np.isclose(df["pi"], pi)])
    rows = []
    for r, lab, col in RICH:
        g = d[d["richness"] == r]
        for n in N_ORDER:
            m, e = _mean_ci((g[g["n_plp"] == n]["fit_prior"] - g[g["n_plp"] == n]["pi"]).abs())
            rows.append(dict(level=lab, color=col, n=n, mean=m, ci=e, n_reps=int((g["n_plp"] == n).sum())))
    return pd.DataFrame(rows)


def draw_prior_scarcity(ax, stats, legend_kw=None, colors=None):
    """Draw `prior_scarcity_stats` output: one line per population/synonymous level, controls per class on x.
    `colors` overrides the level colours (e.g. greys where tan/purple/dark already mean assays)."""
    for i, (lab, g) in enumerate(stats.groupby("level", sort=False)):
        _line(ax, g["n"], g["mean"], g["ci"], colors[i] if colors else g["color"].iloc[0], lab)
    _controls_axis(ax)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, title=GNOMAD_SYN, **(legend_kw or dict(fontsize=5.2, title_fontsize=5.4)))


def make_prior_scarcity_panel(df, out_path, pi=MAIN_PI):
    """Mean |fitted - true prior| (percentage points) by controls per class, one line per gnomAD/synonymous sample size,
    true prior `pi` (core grid, gamma = 1, kappa = MAIN_KAPPA)."""
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    d = _core(df[np.isclose(df["pi"], pi)])
    _by_controls(ax, d, lambda g: (g["fit_prior"] - g["pi"]).abs(), 100)
    ax.set_ylim(bottom=0)
    _style(ax, "controls per class", "|prior error| (percentage points)")
    ax.set_title(f"All samples scarce (true prior {pi:.2f})", loc="left", color=INK, fontweight="bold", pad=4)
    ax.legend(frameon=False, fontsize=5.2, loc="upper right", title=GNOMAD_SYN, title_fontsize=5.4)
    fig.tight_layout()
    return _save(fig, out_path)


# ── review only: overconfidence metric comparison on S3's panels ─────────────────────────────────────────────

def make_overshoot_metric_comparison(df, out_path, pi=MAIN_PI,
                                     metrics=("post_over", "post_over_any", "post_over_share", "post_over_share5",
                                              "post_over_share10")):
    """Rows: S3 C (by controls, kappa 4, gamma 1, lines = population/synonymous size), E (by kappa, gamma 1) and G
    (by gamma, kappa 4), both 200/100; columns: overconfidence metrics. Needs merge_overshoot_rerun for _any/_5/_10."""
    core = df[(df["experiment"] == "core") & np.isclose(df["pi"], pi)]
    mid = core[core["richness"] == "mid"]
    rows = (("C: by controls (κ = 4, γ = 1)", _core(core), "richness", [r for r, _, _ in RICH],
             [lab for _, lab, _ in RICH], GNOMAD_SYN),
            ("E: by separation (γ = 1)", mid[np.isclose(mid["gamma"], 1)], "kappa", [1.5, 2.5, 4.0], ["1.5", "2.5", "4"],
             "separation κ"),
            ("G: by measured fraction (κ = 4)", mid[np.isclose(mid["kappa"], MAIN_KAPPA)], "gamma", [0.25, 0.5, 1.0],
             ["0.25", "0.5", "1"], "measured fraction γ"))
    heads = {"post_over": "Mean overshoot (pp)", "post_over_any": "Share overshooting at all (%)",
             "post_over_share": "Share > 1 pp (%)", "post_over_share5": "Share > 5 pp (%)",
             "post_over_share10": "Share > 10 pp (%)"}
    fig, axes = plt.subplots(len(rows), len(metrics), figsize=(2.1 * len(metrics), 6.6), squeeze=False,
                             gridspec_kw=dict(hspace=0.55, wspace=0.42))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.07)
    for i, (rlab, d, col, levels, labs, ltitle) in enumerate(rows):
        for j, met in enumerate(metrics):
            value, _, scale = METRICS[met]
            ax = axes[i, j]
            for lv, lab, c in zip(levels, labs, LEVELS):
                g = d[d[col] == lv] if col == "richness" else d[np.isclose(d[col], lv)]
                st = [(n, *_mean_ci(scale * value(g[g["n_plp"] == n], "fit"))) for n in N_ORDER]
                _line(ax, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st], c, lab)
            _controls_axis(ax)
            ax.set_ylim(bottom=0)
            _style(ax, "controls per class", heads[met] if j == 0 or True else "")
            ax.set_ylabel(heads[met], fontsize=5.6)
            if i == 0:
                ax.set_title(heads[met], loc="left", color=INK, fontweight="bold", fontsize=6.4, pad=4)
            if j == 0:
                ax.text(-0.42, 1.1, rlab, transform=ax.transAxes, fontsize=6.6, fontweight="bold", color=INK)
                ax.legend(frameon=False, fontsize=4.8, title=ltitle, title_fontsize=5.0, loc="upper right")
    return _save(fig, out_path)


# ── S4 ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def make_methods_figure(merged, out_path, metric="post_over", pi=0.10):
    """ExCALIBR (200 gnomAD, 100 synonymous variants) vs acmgscaler (prior 0.10, >= 10 controls per class), same replicates,
    gamma = 1, kappa = MAIN_KAPPA. (A) overconfidence (`metric`) by controls per class at true prior `pi` (core grid);
    `pi` is 0.10, acmgscaler's own prior, so the control-count comparison is fair to it.
    (B) the same and (C) the share pathogenic among population variants with posterior >= 0.9 by true prior (prior
    slices and core), 20 (light) and 80 (dark) controls per class."""
    value, ylab, scale = METRICS[metric]
    arms = (("fit", (EXC_LIGHT, EXC), "ExCALIBR"), ("acmg", (ACM_LIGHT, ACM), "acmgscaler (prior 0.10)"))
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.08, right=0.98, top=0.88, bottom=0.2)
    sel = lambda d: d[np.isclose(d["gamma"], 1) & np.isclose(d["kappa"], MAIN_KAPPA) & (d["richness"] == "mid")]
    base = sel(merged[(merged["experiment"] == "core") & np.isclose(merged["pi"], pi)])
    ax = fig.add_subplot(gs[0, 0])
    for pre, (_, col), lab in arms:
        d = base[base["acmg_ok"]] if pre == "acmg" else base
        st = [(n, *_mean_ci(scale * value(d[d["n_plp"] == n], pre))) for n in N_ORDER]
        st = [s for s in st if np.isfinite(s[1])]
        _line(ax, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st], col, lab)
    _controls_axis(ax)
    ax.set_ylim(bottom=0)
    _style(ax, "controls per class", ylab)
    _title(ax, 0, f"By controls (true prior {pi:.2f})")
    ax.legend(frameon=False, fontsize=5.4, loc="upper right")

    sl = sel(merged[merged["experiment"].isin(["core", "prior"]) & merged["n_plp"].isin([20, 80])])
    priors = sorted(sl["pi"].unique())
    ax, ax2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ppv_all = []
    for pre, (light, dark), lab in arms:
        for n, col in ((20, light), (80, dark)):
            d = sl[sl["n_plp"] == n]
            d = d[d["acmg_ok"]] if pre == "acmg" else d
            st = [(p, *_mean_ci(scale * value(d[np.isclose(d["pi"], p)], pre))) for p in priors]
            _line(ax, [s[0] for s in st], [s[1] for s in st], [s[2] for s in st], col, f"{lab.split(' ')[0]}, {n}")
            pp = []
            for p in priors:
                pc = precision_curve(d[np.isclose(d["pi"], p)], pre)
                pp.append(100 * float(pc.set_index("t").loc[0.9, "ppv"]) if pc is not None else np.nan)
            ax2.plot(priors, pp, color=col, marker="o", ms=3, lw=1.2, label=f"{lab.split(' ')[0]}, {n}")
            ppv_all.extend(v for v in pp if np.isfinite(v))
    for a in (ax, ax2):
        a.set_xscale("log")
        a.set_xticks(priors)
        a.set_xticklabels([f"{p:g}" for p in priors], rotation=35, ha="right", fontsize=5.8)
        a.minorticks_off()
        a.axvline(0.10, color=MUTED, lw=0.6, ls=":")
    ax.set_ylim(bottom=0)
    _style(ax, "true prior", ylab)
    _title(ax, 1, "Overconfidence by true prior")
    ax.legend(frameon=False, fontsize=5.0, ncol=1, loc="upper right", title="method, controls per class",
              title_fontsize=5.2)
    lo = 5 * np.floor((min(ppv_all) - 3) / 5) if ppv_all else 0  # observed range, not 0-100
    ax2.set_ylim(lo, 101)
    _style(ax2, "true prior", "pathogenic among posterior ≥ 0.9 (%)")
    _title(ax2, 2, "Precision by true prior")
    ax2.legend(frameon=False, fontsize=5.0, ncol=1, loc="lower right", title="method, controls per class",
               title_fontsize=5.2)
    return _save(fig, out_path)


def _sub(ax, text):
    """Muted second title line under a panel title (the settings it shows)."""
    ax.text(0, 1.015, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8, color=MUTED)


# ── S5 ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, text, col="#eef3fb", ec="#9ec5f4", fs=5.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=col, ec=ec, lw=0.7,
                                transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, transform=ax.transAxes,
            fontweight="bold" if bold else None)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))


def _icon(ax, rect):
    """An inset axes in `ax`'s axes coordinates, with no frame or ticks."""
    a = ax.inset_axes(rect)
    a.set_xticks([])
    a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    return a


def _reliability_schematic(ax):
    """Feature cartoons (bootstrap priors, fit to the data, separation) -> boosted trees -> reliability in [0, 1]."""
    ax.axis("off")
    _title(ax, 0, "Prior reliability score", pad=2)
    rng = np.random.default_rng(3)
    x = np.linspace(-4, 4, 200)
    pdf = lambda m, s=1.0: np.exp(-0.5 * ((x - m) / s) ** 2)
    ys = (0.70, 0.38, 0.06)
    # X_q: spread of the bootstrap priors
    a = _icon(ax, [0.04, ys[0], 0.13, 0.26])
    a.hist(rng.normal(0.2, 0.05, 300), bins=14, color=GNOMAD)
    a.text(0.03, 0.95, r"$\hat q_b$", transform=a.transAxes, ha="left", va="top", fontsize=6.5, color=INK)
    # X_fit: fitted density over the observed scores
    a = _icon(ax, [0.04, ys[1], 0.13, 0.26])
    a.hist(rng.normal(0, 1, 300), bins=14, color=PLP, alpha=0.5, density=True)
    a.plot(x, pdf(0) / np.sqrt(2 * np.pi), color=FIT_DARK, lw=1.2)
    a.set_xlim(-4, 4)
    # X_sep: P/LP vs benign reference
    a = _icon(ax, [0.04, ys[2], 0.13, 0.26])
    a.fill_between(x, pdf(1.0), color=BLB, alpha=0.5, lw=0)
    a.fill_between(x, pdf(-1.2), color=PLP, alpha=0.55, lw=0)
    for y, lab in zip(ys, (r"$X_q$", r"$X_{\mathrm{fit}}$", r"$X_{\mathrm{sep}}$")):
        ax.text(0.19, y + 0.13, lab, transform=ax.transAxes, ha="left", va="center", fontsize=8, color=INK)
        _arrow(ax, 0.25, y + 0.13, 0.36, 0.5)
    # boosted trees: three small stacked trees
    for j, dx in enumerate((0.0,)):
        j = 2
        a = _icon(ax, [0.4, 0.2, 0.16, 0.62])
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        nodes = {(0.5, 0.88): [(0.27, 0.55), (0.73, 0.55)], (0.27, 0.55): [(0.13, 0.2), (0.4, 0.2)],
                 (0.73, 0.55): [(0.6, 0.2), (0.87, 0.2)]}
        for (x0, y0), kids in nodes.items():
            for x1, y1 in kids:
                a.plot([x0, x1], [y0, y1], color=MUTED, lw=0.8, alpha=0.5 + 0.25 * j)
        pts = [(0.5, 0.88), (0.27, 0.55), (0.73, 0.55), (0.13, 0.2), (0.4, 0.2), (0.6, 0.2), (0.87, 0.2)]
        a.scatter(*zip(*pts), s=22, color="#b3a9c4", edgecolor="#4a3d5f", lw=0.6, zorder=3, alpha=0.5 + 0.25 * j)
    _arrow(ax, 0.6, 0.5, 0.68, 0.5)
    # output: reliability bar from 0 to 1
    a = _icon(ax, [0.7, 0.38, 0.27, 0.22])
    a.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap="RdYlGn", extent=(0, 1, 0, 1), alpha=0.85)
    a.plot([0.8, 0.8], [-0.15, 1.15], color=INK, lw=1.4, clip_on=False)
    a.set_xticks([0, 1])
    a.set_xticklabels(["0", "1"], fontsize=6.5)
    a.tick_params(length=0)
    ax.text(0.835, 0.72, r"$R$", transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=INK)


def make_reliability_figure(model, real, out_path, yang_t=0.2):
    """(A) how the score is computed; (B) predicted vs observed reliability in grouped cross-validation (deciles);
    (C) the real calibrations ranked by score; (D) their fit-quality check: the largest in-sample Yang distance
    over the samples (diagnostics.real_features), against `yang_t`."""
    from analysis.calibration_quality.diagnostics import ANG_REF
    cv = model["cv"]
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.42, 1], hspace=0.42, wspace=0.42, left=0.08, right=0.98,
                          top=0.95, bottom=0.09)
    ax = fig.add_subplot(gs[0, :])
    _reliability_schematic(ax)

    ax = fig.add_subplot(gs[1, 0])
    rel_p = np.clip(1 - cv["pred_err"] / ANG_REF, 0, 1)
    rel_o = np.clip(1 - cv["err"] / ANG_REF, 0, 1)
    q = pd.qcut(rel_p, 10, labels=False, duplicates="drop")
    b = pd.DataFrame(dict(p=rel_p, o=rel_o, q=q)).groupby("q").agg(p=("p", "mean"), o=("o", "mean"),
                                                                   s=("o", "sem"))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":")
    ax.errorbar(b["p"], b["o"], yerr=1.96 * b["s"], color=EXC, marker="o", ms=3, lw=1.0, capsize=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, "predicted reliability", "observed (simulation)")
    _title(ax, 1, "Prior reliability: calibration")

    ax = fig.add_subplot(gs[1, 1])
    t = real.dropna(subset=["prior_reliability"]).sort_values("prior_reliability").reset_index(drop=True)
    ax.scatter(np.arange(len(t)), t["prior_reliability"], s=8, color=EXC, lw=0, clip_on=False)
    # raw dataset names for the five lowest, to the right at about their own height (spread to >= 0.09 apart) so
    # leader lines run near-horizontal and never cross a label or each other
    low = t.head(5)
    ly, prev = [], -np.inf
    for v in low["prior_reliability"]:
        prev = max(v, prev + 0.09, 0.03)
        ly.append(prev)
    for (i, r), yl in zip(low.iterrows(), ly):
        ax.annotate(r["dataset"], (i, r["prior_reliability"]), xytext=(12, yl), textcoords="data",
                    fontsize=4.6, color=INK, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=1.5, shrinkB=0,
                                    relpos=(0, 0.5)))
    ax.set_ylim(-0.02, 1)
    _style(ax, f"real calibrations, ranked (n = {len(t)})", "prior reliability")
    _title(ax, 2, "Prior reliability: real data")

    ax = fig.add_subplot(gs[1, 2])
    fit = real[["dataset", "yang_train_max", "yang_val_max"]].copy()
    fit["fit_metric"] = fit["yang_train_max"]
    fit = fit.dropna(subset=["fit_metric"]).sort_values("fit_metric").reset_index(drop=True)
    flag = fit["fit_metric"] > yang_t
    ax.scatter(np.arange(len(fit)), fit["fit_metric"], s=8, lw=0, color=np.where(flag, PATH_C, EXC))
    ax.axhline(yang_t, color=MUTED, lw=0.8, ls="--")
    ax.set_ylim(0, max(0.3, float(fit["fit_metric"].max()) * 1.1))
    _style(ax, f"real calibrations, ranked (n = {len(fit)})", "largest in-sample Yang distance")
    _title(ax, 3, "Fit quality: real data")
    return _save(fig, out_path)
