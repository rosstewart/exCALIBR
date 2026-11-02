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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from analysis.prior_simulation.simulate_priors import ASSAYS, CLASSES, DEFAULT_OUT_DIR, POP_M1_FRAC, class_density
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


def _panel_letter(ax, letter, x=-0.18, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


def _style(ax, title, xlabel, ylim, ylabel="Estimated prior"):
    ax.set_title(title, loc="left", color=INK, pad=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _points(ax, df, xcol, span, expected=True):
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
        if expected:
            ax.plot(s[xcol].to_numpy() + base, s["expected"].to_numpy(), color=c, lw=1.0, alpha=0.45, zorder=2)


def _schematic(ax, base):
    """Panel a: data generation -> assays -> estimates."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, lines, step=0.062):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    fc=BOX, ec=GRID, lw=0.8))
        ax.text(x + 0.01, y + h - 0.025, title, fontsize=7.2, fontweight="bold", color=INK, va="top")
        for i, ln in enumerate(lines):
            ax.text(x + 0.01, y + h - 0.1 - step * i, ln, fontsize=6.1, color=MUTED, va="top")

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=MUTED, lw=0.8,
                                     connectionstyle="arc3,rad=0"))

    box(0.0, 0.62, 0.225, 0.38, "Variants in the gene", [
        "benign: probability P(Y=0)", "pathogenic: probability P(Y=1)",
        "   via M1: 25% of pathogenic", "   via M2: 75% of pathogenic", "true P(Y=1) = 0.25 (varied)"])
    box(0.0, 0.08, 0.225, 0.5, "Samples", [
        "P/LP: pathogenic (100)", "   + mislabels, mechanism mix",
        "B/LB: benign (100)", "   + pathogenic mislabels",
        "gnomAD: pathogenic w.p. P(Y=1) (500)", "synonymous: benign (200)", "sparse: 8/1/50/20, 3/0/10/6"])
    box(0.255, 0.2, 0.2, 0.62, "Latent effect (e1, e2)", [
        "e ~ N(μ, Σ),  Σ = [[1, ρ], [ρ, 1]]", "benign:  μ = (0, 0)", "via M1:  μ = (−Δ, −ρΔ)",
        "via M2:  μ = (−ρΔ, −Δ)", "Δ = 4 SD (1.5, 2.5)", "ρ = 0 (0.3, 0.6)", "each variant is read",
        "by all three assays"])
    arrow((0.225, 0.5), (0.255, 0.5))

    x0, w, h, gap = 0.49, 0.51, 0.3, 0.035
    xs = np.linspace(-8, 3.5, 300)
    for i, assay in enumerate(ASSAYS):
        y = 1 - (i + 1) * h - i * gap
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    fc="white", ec=GRID, lw=0.8))
        score = {ASSAYS[0]: "score = e1", ASSAYS[1]: "score = e2", ASSAYS[2]: "score = min(e1, e2)"}[assay]
        ax.text(x0 + 0.01, y + h - 0.022, f"{SHORT[assay]}: reads {LONG[assay]}", fontsize=6.8,
                fontweight="bold", color=COLORS[assay], va="top")
        ax.text(x0 + 0.17, y + h - 0.022, f"({score})", fontsize=6.1, color=MUTED, va="top")
        arrow((0.455, 0.5), (x0, y + h / 2))
        # Class densities.
        d_ax = ax.inset_axes([x0 + 0.012, y + 0.06, 0.2, h - 0.15], transform=ax.transAxes)
        for c, style, col in (("B", "-", BENIGN), ("M1", "--", COLORS[ASSAYS[0]]), ("M2", ":", COLORS[ASSAYS[1]])):
            d_ax.plot(xs, class_density(xs, c, assay, 4.0, 0.0), ls=style, color=col, lw=1.1)
        d_ax.set_yticks([])
        d_ax.set_xticks([-4, 0], ["abnormal", "normal"])
        d_ax.tick_params(length=0, labelsize=6)
        d_ax.spines["left"].set_visible(False)
        d_ax.set_xlim(-8, 3.5)
        d_ax.set_ylim(0, 0.45)
        # Estimate strip.
        s_ax = ax.inset_axes([x0 + 0.26, y + 0.075, 0.235, h - 0.19], transform=ax.transAxes)
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
        s_ax.set_title("estimated prior (true 0.25)", fontsize=6, color=MUTED, loc="left", pad=1.5)
    # Legend for panel a.
    handles = [
        plt.Line2D([], [], color=BENIGN, lw=1.1, label="benign"),
        plt.Line2D([], [], color=COLORS[ASSAYS[0]], ls="--", lw=1.1, label="pathogenic via M1"),
        plt.Line2D([], [], color=COLORS[ASSAYS[1]], ls=":", lw=1.1, label="pathogenic via M2"),
        plt.Line2D([], [], marker="D", mfc="white", mec=INK, ls="", ms=4,
                   label="fraction of gnomAD pathogenic and scoring abnormal"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.035), ncol=4, frameon=False, fontsize=6.2,
              handlelength=2.2, columnspacing=1.2)


def make_figure(summ, out_path):
    fig = plt.figure(figsize=(7.2, 10.0))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.55, 1, 1, 1], hspace=0.62, wspace=0.34,
                          left=0.08, right=0.985, top=0.975, bottom=0.075)
    ax_a = fig.add_subplot(gs[0, :])
    _schematic(ax_a, sel(summ, experiment="1_clean", regime="rich", true_prior=0.25))
    _panel_letter(ax_a, "a", x=-0.05, y=0.98)

    grid = [fig.add_subplot(gs[r, c]) for r in (1, 2, 3) for c in range(3)]
    clean = [("rich", "Rich controls (100/100/500/200)"), ("TPK1-like", "TPK1-like controls (8/1/50/20)"),
             ("KCNH2_Kozek-like", "KCNH2-like controls (3/0/10/6)")]
    for ax, (reg, title) in zip(grid[:3], clean):
        df = sel(summ, experiment="1_clean", regime=reg)
        ax.plot([0, 0.45], [0, 0.45], color=MUTED, ls="--", lw=0.8, zorder=1)
        _points(ax, df, "true_prior", 0.45, expected=False)
        ax.set_xlim(0, 0.45)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
        _style(ax, title, "True prior", (0, 0.6))
    for assay in ASSAYS:
        grid[0].plot([0, 0.45], [0, 0.45 * DETECTED[assay]], color=COLORS[assay], ls=":", lw=0.9, zorder=1)
    grid[0].text(0.01, 0.56, "dotted: fraction pathogenic\nand scoring abnormal", fontsize=5.8, color=MUTED, va="top")

    specs = [
        (grid[3], "2_noise_plp", "eps_p", "Benign variants labelled P/LP", "Fraction of P/LP that are benign",
         (0.03, 0.22), (0.1, 0.45), [0.05, 0.1, 0.15, 0.2], {}),
        (grid[4], "2_noise_blb", "delta_b", "Pathogenic variants labelled B/LB", "Fraction of B/LB that are pathogenic",
         (0.03, 0.22), (0.1, 0.45), [0.05, 0.1, 0.15, 0.2], {}),
        (grid[5], "3_mech_bias", "lab_m1_frac", "P/LP mechanism mix", "Fraction of P/LP via M1 (gnomAD: 0.25)",
         (-0.07, 1.07), (0, 0.85), [0, 0.25, 0.5, 0.75, 1], {}),
        (grid[6], "4_separation", "delta", "Separation (true prior 0.25)", "Abnormal shift Δ (SD)",
         (1.2, 4.3), (0.1, 0.45), [1.5, 2.5, 4], {"true_prior": 0.25}),
        (grid[7], "5_correlation", "rho", "Correlated mechanisms, Δ = 4", "Correlation ρ",
         (-0.08, 0.68), (0.1, 0.45), [0, 0.3, 0.6], {"true_prior": 0.25, "delta": 4.0}),
        (grid[8], "5_correlation", "rho", "Correlated mechanisms, Δ = 2.5", "Correlation ρ",
         (-0.08, 0.68), (0.1, 0.45), [0, 0.3, 0.6], {"true_prior": 0.25, "delta": 2.5}),
    ]
    for ax, exp, xcol, title, xlabel, xlim, ylim, xticks, extra in specs:
        df = sel(summ, experiment=exp, **extra)
        ax.axhline(0.25, color=MUTED, ls="--", lw=0.8, zorder=1)
        _points(ax, df, xcol, xlim[1] - xlim[0], expected=exp in ("2_noise_plp", "2_noise_blb", "3_mech_bias"))
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        _style(ax, title, xlabel, ylim)
    for ax, letter in zip(grid, "bcdefghij"):
        _panel_letter(ax, letter)

    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[a], mec="white", ms=5,
                          label=f"{SHORT[a]}: {LONG[a]}") for a in ASSAYS]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=4.5, label="true components"),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", ms=4.5, label="ExCALIBR fit"),
        plt.Line2D([], [], color=MUTED, alpha=0.45, lw=1.0, label="large-sample expectation"),
        plt.Line2D([], [], color=MUTED, ls="--", lw=0.8, label="true prior"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=6.3, columnspacing=1.0,
               handletextpad=0.4, bbox_to_anchor=(0.5, 0.005))
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
