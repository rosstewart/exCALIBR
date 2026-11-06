"""Supplementary figure for the prior-estimation simulation.

(a) schematic of the simulation with each assay's class score densities and
its prior estimates; (b-d) clean controls in three control-size regimes;
(e-g) label noise and P/LP mechanism mix; (h-j) separation and correlated
mechanisms. Reads paired_replicates.csv (test_priors.py) via summary.load.

Run:
    python -m analysis.prior_simulation.figures [--data-dir DIR] [--out PATH]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from analysis.prior_simulation.simulate_priors import (ASSAYS, CLASSES, DEFAULT_OUT_DIR, POP_M1_FRAC, REGIMES,
                                                       class_density, predicted_prior)
from analysis.prior_simulation.summary import load, sel

# Categorical slots 1-3 of the validated default palette, in fixed order.
COLORS = dict(zip(ASSAYS, ["#2a78d6", "#eb6834", "#1baf7a"]))
SHORT = dict(zip(ASSAYS, ["A1", "A2", "A3"]))
LONG = dict(zip(ASSAYS, ["M1 only", "M2 only", "M1 + M2"]))
DETECTED = dict(zip(ASSAYS, [POP_M1_FRAC, 1 - POP_M1_FRAC, 1.0]))
INK, MUTED, GRID, BOX = "#14171c", "#565d68", "#e6e8ec", "#f4f5f7"
BENIGN = "#7a828d"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def _letter(i):
    """Panel label: capital letter in parentheses, e.g. (A)."""
    return f"({'ABCDEFGHIJ'[i]})"


def _panel_letter(ax, letter, x=-0.18, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


def _style(ax, title, xlabel, ylim, ylabel="Estimated prior"):
    ax.set_title(title, loc="left", color=INK, pad=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _points(ax, df, xcol, span, experiment=None):
    """Mean and 95% CI per assay; with `experiment`, also the closed-form large-sample prediction of
    Proposition 3 (`predicted_prior`) as a line."""
    for i, assay in enumerate(ASSAYS):
        s = df[df.assay == assay].sort_values(xcol)
        if s.empty:
            continue
        base = (i - 1) * 0.03 * span
        c = COLORS[assay]
        for arm, p, dx in (("oracle", "o", -0.009), ("fit", "f", 0.009)):
            x = s[xcol].to_numpy() + base + dx * span
            m = s[f"{p}_m"].to_numpy()
            err = np.vstack([m - s[f"{p}_lo"].to_numpy(), s[f"{p}_hi"].to_numpy() - m])
            ax.errorbar(x, m, yerr=err, fmt="o", ms=3.6, lw=1.1, capsize=0, color=c, zorder=3,
                        mfc=c if arm == "oracle" else "white", mec=c, mew=1.0)
        if experiment is not None:
            pred = np.array([predicted_prior(experiment, assay, r.true_prior, eps_p=r.eps_p, delta_b=r.delta_b,
                                             lab_m1_frac=r.lab_m1_frac) for r in s.itertuples()])
            ax.plot(s[xcol].to_numpy() + base, pred, color=c, lw=1.0, alpha=0.45, zorder=2)


PATH_C = "#a8434f"  # pathogenic (any mechanism) in the sample cartoons


def _clouds(ax, kappa=4.0, rho=0.0, n=90, seed=3, lim=(-7.5, 3)):
    """Latent effects (e1, e2): benign at (0, 0), pathogenic via M1 at (-kappa, -rho kappa), via M2 at
    (-rho kappa, -kappa); unit variance, correlation rho."""
    rng = np.random.default_rng(seed)
    cov = [[1, rho], [rho, 1]]
    for mu, col, k in (((0, 0), BENIGN, n), ((-kappa, -rho * kappa), COLORS[ASSAYS[0]], n // 3),
                       ((-rho * kappa, -kappa), COLORS[ASSAYS[1]], n)):
        e = rng.multivariate_normal(mu, cov, k)
        ax.scatter(e[:, 0], e[:, 1], s=1.6, color=col, alpha=0.7, lw=0)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlabel("e1", labelpad=1, fontsize=6)
    ax.set_ylabel("e2", labelpad=1, fontsize=6)


def _schematic(ax, base):
    """Panel a: latent effects -> three assays -> estimates."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.8,
                                     connectionstyle="arc3,rad=0"))

    # Latent values of every variant (square panel), and the mechanism split of pathogenic variants below it.
    fw, fh = ax.figure.get_size_inches()
    pos = ax.get_position()
    aspect = (pos.height * fh) / (pos.width * fw)  # axes-fraction width of a square of unit axes-fraction height
    hc = 0.66
    wc = hc * aspect
    xc = 0.1
    c_ax = ax.inset_axes([xc, 0.25, wc, hc], transform=ax.transAxes)
    _clouds(c_ax, lim=(-7.5, 5))
    c_ax.set_title("Latent values of each variant", fontsize=6.8, color=INK, loc="center", pad=3, fontweight="bold")
    c_ax.text(2.9, 0.0, "benign", fontsize=5.8, color=BENIGN, ha="left", va="center")
    c_ax.text(-4, 2.6, "via M1", fontsize=5.8, color=COLORS[ASSAYS[0]], ha="center", va="bottom")
    c_ax.text(2.9, -4, "via M2", fontsize=5.8, color=COLORS[ASSAYS[1]], ha="left", va="center")
    b_ax = ax.inset_axes([xc, 0.03, wc, 0.07], transform=ax.transAxes)
    b_ax.barh([0], [POP_M1_FRAC], color=COLORS[ASSAYS[0]], height=1)
    b_ax.barh([0], [1 - POP_M1_FRAC], left=POP_M1_FRAC, color=COLORS[ASSAYS[1]], height=1)
    b_ax.text(POP_M1_FRAC / 2, 0, f"{100 * POP_M1_FRAC:.0f}%", color="white", fontsize=5.6, ha="center",
              va="center")
    b_ax.text(POP_M1_FRAC + (1 - POP_M1_FRAC) / 2, 0, f"{100 * (1 - POP_M1_FRAC):.0f}%", color="white",
              fontsize=5.6, ha="center", va="center")
    b_ax.text(-0.04, 0, "pathogenic:\nM1 / M2", fontsize=5.8, color=MUTED, ha="right", va="center",
              transform=b_ax.transData)
    b_ax.set_xlim(0, 1)
    b_ax.axis("off")
    x_arrow = xc + wc + 0.03

    x0, w, h, gap = 0.42, 0.58, 0.29, 0.03
    xs = np.linspace(-8, 3.5, 300)
    for i, assay in enumerate(ASSAYS):
        y = 1 - (i + 1) * h - i * gap
        score = {ASSAYS[0]: "score = e1", ASSAYS[1]: "score = e2", ASSAYS[2]: "score = min(e1, e2)"}[assay]
        ax.text(x0 + 0.01, y + h - 0.022, f"{SHORT[assay]}: reads {LONG[assay]}", fontsize=6.8,
                fontweight="bold", color=COLORS[assay], va="top")
        ax.text(x0 + 0.19, y + h - 0.022, f"({score})", fontsize=6.1, color=MUTED, va="top")
        arrow((x_arrow, 0.58), (x0 - 0.01, y + h / 2))
        d_ax = ax.inset_axes([x0 + 0.012, y + 0.06, 0.23, h - 0.15], transform=ax.transAxes)
        for c, style, col in (("B", "-", BENIGN), ("M1", "--", COLORS[ASSAYS[0]]), ("M2", ":", COLORS[ASSAYS[1]])):
            d_ax.plot(xs, class_density(xs, c, assay, 4.0, 0.0), ls=style, color=col, lw=1.1)
        d_ax.set_yticks([])
        d_ax.set_xticks([-4, 0], ["abnormal", "normal"])
        d_ax.tick_params(length=0, labelsize=6)
        d_ax.spines["left"].set_visible(False)
        d_ax.set_xlim(-8, 3.5)
        d_ax.set_ylim(0, 0.45)
        s_ax = ax.inset_axes([x0 + 0.3, y + 0.105, 0.265, h - 0.2], transform=ax.transAxes)
        r = base[base.assay == assay].iloc[0]
        c = COLORS[assay]
        s_ax.axvline(0.25, color=MUTED, ls="--", lw=0.8)
        s_ax.plot([0.25 * DETECTED[assay]], [0.15], marker="D", ms=4, mfc="white", mec=INK, mew=0.9, ls="")
        for yy, p, filled in ((0.7, "o", True), (0.42, "f", False)):
            s_ax.errorbar([r[f"{p}_m"]], [yy], xerr=[[r[f"{p}_m"] - r[f"{p}_lo"]], [r[f"{p}_hi"] - r[f"{p}_m"]]],
                          fmt="o", ms=3.8, lw=1.2, capsize=0, color=c, mfc=c if filled else "white", mec=c)
        s_ax.set_xlim(0, 0.5)
        s_ax.set_ylim(0, 1)
        s_ax.set_yticks([])
        s_ax.spines["left"].set_visible(False)
        s_ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        s_ax.tick_params(labelsize=6)
        s_ax.set_title("estimated prior (true 0.25)", fontsize=6, color=MUTED, loc="right", pad=1.5)
    handles = [
        plt.Line2D([], [], color=BENIGN, lw=1.1, label="benign"),
        plt.Line2D([], [], color=COLORS[ASSAYS[0]], ls="--", lw=1.1, label="pathogenic via M1"),
        plt.Line2D([], [], color=COLORS[ASSAYS[1]], ls=":", lw=1.1, label="pathogenic via M2"),
        plt.Line2D([], [], marker="D", mfc="white", mec=INK, ls="", ms=4,
                   label="fraction of population pathogenic and scoring abnormal"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.42, 0.0), ncol=2, frameon=False, fontsize=5.9,
              handlelength=2.2, columnspacing=1.2)


# Sample cartoons above panels b-j: what each experiment changes.
_SAMPLES = ("P/LP", "B/LB", "population", "synonymous")


def _bar(ax, y, parts, label, note=""):
    """One horizontal sample bar of unit length split into (fraction, color, hatch) parts."""
    left = 0.0
    for frac, col, hatch in parts:
        ax.barh([y], [frac], left=left, color=col, height=0.62, hatch=hatch, edgecolor="white", lw=0.4)
        left += frac
    ax.text(-0.03, y, label, ha="right", va="center", fontsize=5.8, color=MUTED)
    if note:
        ax.text(1.03, y, note, ha="left", va="center", fontsize=5.6, color=MUTED)


def _cartoon_axes(ax):
    ax.set_xlim(-0.42, 1.55)
    ax.axis("off")


def _cartoon_sizes(ax, counts):
    """Bars proportional to log10(n + 1) on a common scale (500 = full), one per sample, coloured by class."""
    _cartoon_axes(ax)
    full = np.log10(501)
    comp = {"P/LP": [(1, PATH_C, "")], "B/LB": [(1, BENIGN, "")],
            "population": [(0.25, PATH_C, ""), (0.75, BENIGN, "")], "synonymous": [(1, BENIGN, "")]}
    for k, (name, n) in enumerate(zip(_SAMPLES, counts)):
        L = np.log10(n + 1) / full
        _bar(ax, 3 - k, [(f * L, c, h) for f, c, h in comp[name]], name, str(n))
    ax.set_ylim(-0.6, 3.6)


def _cartoon_noise(ax, which):
    _cartoon_axes(ax)
    if which == "plp":
        _bar(ax, 1, [(0.8, PATH_C, ""), (0.2, BENIGN, "////")], "P/LP", "benign: 0–20%")
        _bar(ax, 0, [(1, BENIGN, "")], "B/LB")
    else:
        _bar(ax, 1, [(1, PATH_C, "")], "P/LP")
        _bar(ax, 0, [(0.8, BENIGN, ""), (0.2, PATH_C, "////")], "B/LB", "pathogenic: 0–20%")
    ax.set_ylim(-0.6, 1.6)


def _cartoon_mech(ax):
    _cartoon_axes(ax)
    m1, m2 = COLORS[ASSAYS[0]], COLORS[ASSAYS[1]]
    _bar(ax, 1, [(0.75, m1, "////"), (0.25, m2, "")], "P/LP", "M1: 0–100%")
    _bar(ax, 0, [(POP_M1_FRAC, m1, ""), (1 - POP_M1_FRAC, m2, "")], "population\npathogenic", "M1: 25%")
    ax.set_ylim(-0.6, 1.6)


def _cartoon_shift(ax):
    xs = np.linspace(-6, 3, 200)
    g = lambda m: np.exp(-0.5 * (xs - m) ** 2)
    ax.plot(xs, g(0), color=BENIGN, lw=1.0)
    for k, a in ((1.5, 1.0), (2.5, 0.6), (4.0, 0.35)):
        ax.plot(xs, g(-k), color=PATH_C, lw=1.0, alpha=a)
    ax.annotate("", xy=(-4, 1.12), xytext=(0, 1.12), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.7))
    ax.text(-2, 1.2, "κ = 1.5, 2.5, 4", fontsize=5.8, color=MUTED, ha="center", va="bottom")
    ax.set_xlim(-8.5, 5)
    ax.set_ylim(0, 1.6)
    ax.axis("off")


def _cartoon_corr(ax, kappa):
    ax.axis("off")
    for k, rho in enumerate((0.0, 0.6)):
        sub = ax.inset_axes([0.2 + 0.33 * k, 0.2, 0.28, 0.85])
        _clouds(sub, kappa=kappa, rho=rho, n=70)
        sub.set_xlabel("")
        sub.set_ylabel("")
        for s in sub.spines.values():
            s.set_color(GRID)
        sub.text(0.5, -0.04, f"ρ = {rho:g}", transform=sub.transAxes, fontsize=5.8, color=MUTED, ha="center",
                 va="top")


def make_figure(summ, out_path):
    fig = plt.figure(figsize=(7.2, 8.4))
    outer = fig.add_gridspec(4, 1, height_ratios=[1.25, 1.3, 1.3, 1.3], hspace=0.40,
                             left=0.08, right=0.985, top=0.975, bottom=0.085)
    ax_a = fig.add_subplot(outer[0])
    _schematic(ax_a, sel(summ, experiment="1_clean", regime="rich", true_prior=0.25))
    _panel_letter(ax_a, _letter(0), x=-0.05, y=0.98)

    grid, carts = [], []
    for r in (1, 2, 3):
        inner = outer[r].subgridspec(2, 3, height_ratios=[0.36, 1], hspace=0.14, wspace=0.34)
        carts += [fig.add_subplot(inner[0, c]) for c in range(3)]
        grid += [fig.add_subplot(inner[1, c]) for c in range(3)]
    clean = [("rich", "Rich samples"), ("TPK1-like", "Sparse samples"), ("KCNH2_Kozek-like", "Very sparse samples")]
    for ax, cax, (reg, title) in zip(grid[:3], carts[:3], clean):
        df = sel(summ, experiment="1_clean", regime=reg)
        ax.plot([0, 0.45], [0, 0.45], color=MUTED, ls="--", lw=0.8, zorder=1)
        _points(ax, df, "true_prior", 0.45)
        ax.set_xlim(0, 0.45)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        _style(ax, "", "True prior", (0, 0.6))
        _cartoon_sizes(cax, REGIMES[reg])
        cax.set_title(title, loc="left", color=INK, pad=2, fontsize=7.5)
    for assay in ASSAYS:
        grid[0].plot([0, 0.45], [0, 0.45 * DETECTED[assay]], color=COLORS[assay], ls=":", lw=0.9, zorder=1)
    grid[0].text(0.01, 0.56, "dotted: fraction pathogenic\nand scoring abnormal", fontsize=5.8, color=MUTED, va="top")

    specs = [
        (grid[3], "2_noise_plp", "eps_p", "Benign variants labeled P/LP", "Fraction of P/LP that are benign",
         (0.03, 0.22), (0.1, 0.45), [0.05, 0.1, 0.15, 0.2], {}, lambda c: _cartoon_noise(c, "plp")),
        (grid[4], "2_noise_blb", "delta_b", "Pathogenic variants labeled B/LB", "Fraction of B/LB that are pathogenic",
         (0.03, 0.22), (0.1, 0.45), [0.05, 0.1, 0.15, 0.2], {}, lambda c: _cartoon_noise(c, "blb")),
        (grid[5], "3_mech_bias", "lab_m1_frac", "P/LP mechanism mix", "Fraction of P/LP via M1 (population: 0.25)",
         (-0.07, 1.07), (0, 0.85), [0, 0.25, 0.5, 0.75, 1], {}, _cartoon_mech),
        (grid[6], "4_separation", "delta", "Separation", "Abnormal shift κ (SD)",
         (1.2, 4.3), (0.1, 0.45), [1.5, 2.5, 4], {"true_prior": 0.25}, _cartoon_shift),
        (grid[7], "5_correlation", "rho", "Correlated mechanisms, κ = 4", "Correlation ρ",
         (-0.08, 0.68), (0.1, 0.45), [0, 0.3, 0.6], {"true_prior": 0.25, "delta": 4.0},
         lambda c: _cartoon_corr(c, 4.0)),
        (grid[8], "5_correlation", "rho", "Correlated mechanisms, κ = 2.5", "Correlation ρ",
         (-0.08, 0.68), (0.1, 0.45), [0, 0.3, 0.6], {"true_prior": 0.25, "delta": 2.5},
         lambda c: _cartoon_corr(c, 2.5)),
    ]
    for (ax, exp, xcol, title, xlabel, xlim, ylim, xticks, extra, cart), cax in zip(specs, carts[3:]):
        df = sel(summ, experiment=exp, **extra)
        ax.axhline(0.25, color=MUTED, ls="--", lw=0.8, zorder=1)
        _points(ax, df, xcol, xlim[1] - xlim[0],
                experiment=exp if exp in ("2_noise_plp", "2_noise_blb", "3_mech_bias") else None)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        _style(ax, "", xlabel, ylim)
        cart(cax)
        cax.set_title(title, loc="left", color=INK, pad=2, fontsize=7.5)
    for k, cax in enumerate(carts):
        cax.text(-0.02, 1.02, _letter(k + 1), transform=cax.transAxes, fontsize=9, fontweight="bold", color=INK,
                 va="bottom", ha="right")

    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[a], mec="white", ms=5,
                          label=f"{SHORT[a]}: {LONG[a]}") for a in ASSAYS]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=4.5, label="true components"),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", ms=4.5, label="ExCALIBR fit"),
        plt.Line2D([], [], color=MUTED, alpha=0.45, lw=1.0, label="closed-form prediction (Proposition 3)"),
        plt.Line2D([], [], color=MUTED, ls="--", lw=0.8, label="true prior"),
        plt.Rectangle((0, 0), 1, 1, color=PATH_C, label="pathogenic (sample cartoons)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.3, columnspacing=1.0,
               handletextpad=0.4, bbox_to_anchor=(0.5, 0.003))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def make_sample_size_figure(ss, out_path):
    """Supplementary figure: bias (a-d) and SD (e-h) of the true-component prior estimate as one control sample at a
    time is reduced (true prior 0.25; others at 100/100/500/200). Filled: all estimates (before the discard rule);
    open: estimates kept by the discard rule (prior <= 0.001 or >= 0.999 dropped). Lines: closed-form predictions
    (delta-method P/LP bias; asymptotic SD, Proposition 2)."""
    from analysis.prior_simulation.simulate_priors import predicted_plp_bias, predicted_sd
    from analysis.prior_simulation.summary import BASE_COUNTS, SAMPLES
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), gridspec_kw=dict(hspace=0.55, wspace=0.38))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.17)
    for j, sample in enumerate(SAMPLES):
        g = ss[ss["sample"] == sample]
        sizes = sorted(g["size"].unique(), reverse=True)
        pos = {n: k for k, n in enumerate(sizes)}
        for row, (col, lo, hi, ylabel, ylim) in enumerate((("bias", "bias_lo", "bias_hi", "Bias (estimate − 0.25)",
                                                            (-0.1, 0.32)),
                                                           ("sd", "sd_lo", "sd_hi", "SD of the estimate", (0, 0.32)))):
            ax = axes[row, j]
            if row == 0:
                ax.axhline(0, color=MUTED, ls="--", lw=0.8, zorder=1)
            for i, assay in enumerate(ASSAYS):
                c = COLORS[assay]
                for arm, filled, dx in (("oracle_raw", True, -0.07), ("oracle", False, 0.07)):
                    if row == 1 and arm == "oracle":
                        continue
                    r = g[(g.assay == assay) & (g.arm == arm)].sort_values("size", ascending=False)
                    x = np.array([pos[n] for n in r["size"]]) + (i - 1) * 0.16 + (dx if row == 0 else 0)
                    m = r[col].to_numpy()
                    err = np.vstack([m - r[lo].to_numpy(), r[hi].to_numpy() - m])
                    ax.errorbar(x, m, yerr=err, fmt="o", ms=3.2, lw=1.0, capsize=0, color=c, zorder=3,
                                mfc=c if filled else "white", mec=c, mew=0.9)
                xs = np.arange(len(sizes)) + (i - 1) * 0.16
                counts = [{**BASE_COUNTS, sample: n} for n in sizes]
                if row == 0 and sample == "P/LP":
                    # delta-method approximation: drawn only where >= 2.5 P/LP are expected to act through the
                    # measured mechanism(s)
                    ok = [k["P/LP"] * DETECTED[assay] >= 2.5 for k in counts]
                    ax.plot(xs[ok], [predicted_plp_bias(assay, 0.25, k["P/LP"]) for k, o in zip(counts, ok) if o],
                            color=c, lw=1.0, alpha=0.45, zorder=2)
                if row == 1 and sample in ("P/LP", "gnomAD"):
                    ax.plot(xs, [predicted_sd(assay, 0.25, k["gnomAD"], k["P/LP"]) for k in counts], color=c,
                            lw=1.0, alpha=0.45, zorder=2)
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([str(int(n)) for n in sizes])
            ax.set_xlim(-0.5, len(sizes) - 0.5)
            name = "population" if sample == "gnomAD" else sample
            _style(ax, f"{name} reduced" if row == 0 else "", f"{name} variants", ylim, ylabel=ylabel)
            _panel_letter(ax, _letter(row * 4 + j), x=-0.3, y=1.04)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[a], mec="white", ms=5,
                          label=f"{SHORT[a]}: {LONG[a]}") for a in ASSAYS]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=4.5, label="all estimates"),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", ms=4.5, label="after discard rule"),
        plt.Line2D([], [], color=MUTED, alpha=0.45, lw=1.0, label="closed-form prediction"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.3, columnspacing=1.0,
               handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "supp_prior_simulation.png")
    args = ap.parse_args()
    _, _, summ = load(args.data_dir)
    print(f"wrote {make_figure(summ, args.out)}")


if __name__ == "__main__":
    main()
