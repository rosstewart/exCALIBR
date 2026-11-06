"""Figure for the prior-estimation simulation (reads results.csv).

Run:
    python -m analysis.prior_simulation.plot_priors [--out-dir DIR]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.prior_simulation.simulate_priors import ASSAYS, DEFAULT_OUT_DIR, POP_M1_FRAC

# Categorical slots 1-3 of the validated default palette, in fixed order.
COLORS = dict(zip(ASSAYS, ["#2a78d6", "#eb6834", "#1baf7a"]))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OFFSETS = dict(zip(ASSAYS, [-1.5, 0, 1.5]))
DETECTED = dict(zip(ASSAYS, [POP_M1_FRAC, 1 - POP_M1_FRAC, 1.0]))


def _style(ax, title, xlabel, ylabel="Estimated prior"):
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)
    ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def _arm(ax, sub, xcol, assay, dx, prefix, filled):
    if f"{prefix}_median" not in sub or sub[f"{prefix}_median"].isna().all():
        return
    sub = sub.sort_values(xcol)
    x = sub[xcol].to_numpy() + dx
    y = sub[f"{prefix}_median"].to_numpy()
    err = np.vstack([y - sub[f"{prefix}_p5"], sub[f"{prefix}_p95"] - y])
    c = COLORS[assay]
    ax.errorbar(x, y, yerr=err, fmt="o", ms=5, lw=1.1, capsize=0, color=c,
                mfc=c if filled else "white", mec=c if not filled else "white", mew=1.2)


def _panel(ax, df, title, xcol, xlabel, truth=0.25, **filters):
    sub = df
    for k, v in filters.items():
        sub = sub[np.isclose(sub[k], v)] if isinstance(v, float) else sub[sub[k] == v]
    span = np.ptp(sub[xcol]) or 1.0
    for assay in ASSAYS:
        s = sub[sub.assay == assay]
        dx = OFFSETS[assay] * 0.012 * span
        _arm(ax, s, xcol, assay, dx - 0.006 * span, "oracle_boot", True)
        _arm(ax, s, xcol, assay, dx + 0.006 * span, "fit_boot", False)
        s = s.sort_values(xcol)
        ax.plot(s[xcol] + dx, s.expected_prior, "_", ms=11, mew=1.5, color=COLORS[assay])
    if xcol == "true_prior":
        ax.plot([0, 0.45], [0, 0.45], color=MUTED, lw=1, ls="--", zorder=0)
        ax.set_xlim(0, 0.45)
    else:
        ax.axhline(truth, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_ylim(0, 1.02)
    _style(ax, title, xlabel)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    df = pd.read_csv(args.out_dir / "results.csv")

    fig, axes = plt.subplots(3, 3, figsize=(13, 11.5))
    ax = axes.flat
    _panel(ax[0], df, "a  Clean labels, rich (100/100/500/200)", "true_prior", "True prior",
           experiment="1_clean", regime="rich")
    for assay in ASSAYS:
        # The reviewer's quantity: fraction of gnomAD variants that score abnormal.
        x = np.array([0, 0.45])
        ax[0].plot(x, x * DETECTED[assay], ls=":", lw=1.2, color=COLORS[assay])
    ax[0].text(0.44, 0.02, "dotted: fraction of gnomAD that is\npathogenic AND detected by the assay",
               fontsize=7, color=MUTED, ha="right")
    _panel(ax[1], df, "b  Clean labels, TPK1-like (8/1/50/20)", "true_prior", "True prior",
           experiment="1_clean", regime="TPK1-like")
    _panel(ax[2], df, "c  Clean labels, KCNH2_Kozek-like (3/0/10/6)", "true_prior", "True prior",
           experiment="1_clean", regime="KCNH2_Kozek-like")
    _panel(ax[3], df, "d  Benign variants labeled P/LP", "eps_p", "Fraction of P/LP that are benign",
           experiment="2_noise_plp")
    _panel(ax[4], df, "e  Pathogenic variants labeled B/LB", "delta_b",
           "Fraction of B/LB that are pathogenic", experiment="2_noise_blb")
    _panel(ax[5], df, "f  Mechanism-biased P/LP (gnomAD: 25% M1)", "lab_m1_frac",
           "Fraction of P/LP acting via M1", experiment="3_mech_bias")
    _panel(ax[6], df, "g  Assay separation (true prior 0.25)", "delta",
           "Abnormal shift (SD units)", experiment="4_separation", true_prior=0.25)
    _panel(ax[7], df, "h  Correlated mechanisms, shift 4 SD", "rho",
           "Correlation between mechanisms", experiment="5_correlation", true_prior=0.25, delta=4.0)
    _panel(ax[8], df, "i  Correlated mechanisms, shift 2.5 SD", "rho",
           "Correlation between mechanisms", experiment="5_correlation", true_prior=0.25, delta=2.5)

    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[a], mec="white", ms=6, label=a)
               for a in ASSAYS]
    handles += [
        plt.Line2D([], [], marker="o", ls="", color=MUTED, mec="white", ms=6, label="Fixed true components"),
        plt.Line2D([], [], marker="o", ls="", color=MUTED, mfc="white", ms=6, label="Production 2-component fit"),
        plt.Line2D([], [], marker="_", ls="", color=MUTED, ms=11, mew=1.5, label="Large-sample expectation"),
        plt.Line2D([], [], ls="--", color=MUTED, label="True prior"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.text(0.01, 0.035, "Points: median and 5-95% range of bootstrap prior estimates, pooled over replicate "
             "datasets (fixed components: 10 x 100; fit: 3 x 100).", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"prior_simulation.{ext}", dpi=200)
    print(f"wrote {args.out_dir / 'prior_simulation.png'}")


if __name__ == "__main__":
    main()
