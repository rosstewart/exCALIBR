"""Schematic of the consequence-cell simulation, drawn from generate.py's parameters (so it cannot drift from the code).

Top: the generative tree for one variant with the default (anchored) numbers. Bottom: for each consequence and assay
type, the true score densities split by true pathogenicity, with the share of each class that is functionally
abnormal in that assay, and the true prior of the kept population.

Run: python -m analysis.consequence_cells_sim.schematic [--out PATH]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.consequence_cells_sim import generate as G

PATH, BEN = "#b2182b", "#2166ac"


def tree_text(c):
    mm = c["mis_mech"]
    return "\n".join([
        "consequence c (population shares: synonymous {synonymous:.2f}, missense {missense:.2f}, PTV {PTV:.2f})".format(**c["pop_share"]),
        "  SpliceAI max(DS_AG/AL/DG/DL) >= 0.2 with P = h_c (syn {synonymous:.3f}, mis {missense:.3f}, PTV {PTV:.3f})".format(**c["h"]),
        "  splice disruption: P = d_c if DS >= 0.2 (syn {synonymous:.3f}, mis {missense:.3f}, PTV {PTV:.3f}), ".format(**c["d_hi"]) +
        f"{c['d_lo']:.3f} otherwise;  pathogenic if disrupted with P = {c['q_sp']:.2f}",
        "  protein-mechanism pathogenic with P = pi_prot (syn {synonymous:.2f}, mis {missense:.3f}, PTV {PTV:.2f})".format(**c["pi_prot"]),
        f"     missense mechanism: stability {mm['stability']:.2f} | activity {mm['activity']:.2f} | other (GoF etc.) {mm['other']:.2f};"
        f"  PTV: LoF {c['tau_ptv']:.2f} | partial {1 - c['tau_ptv']:.2f}",
        "  pathogenic = splice-pathogenic OR protein-pathogenic",
        "  functionally abnormal in the assay:",
        "     fitness (SGE-like, endogenous locus):  splice disruption, missense stability/activity, PTV LoF",
        "     abundance (VAMP-like, cDNA):            missense stability, PTV LoF;  rows with DS > 0.2 removed (production filter)",
        f"  score: abnormal N(-kappa, 1), normal N(0, 1)   (kappa grid {G.KAPPAS}; hypomorph slice adds N(-kappa/2, 1))",
    ])


def main(out):
    base = G.make_cell("schematic")
    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.75, 1, 1], hspace=0.5, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
    ax0.text(0, 1, tree_text(base), family="monospace", fontsize=8.3, va="top")
    ax0.set_title("Generative model for one variant (defaults anchored on step 0)", loc="left", fontsize=10, fontweight="bold")
    x = np.linspace(-8, 4, 600)
    for i, assay in enumerate(("fitness", "abundance")):
        c = G.make_cell("schematic", assay=assay)
        for j, csq in enumerate(G.CONSEQUENCES):
            ax = fig.add_subplot(gs[i + 1, j])
            t = G.truth(c, csq)
            fb = np.exp(G._mix_logpdf(x, c, t["B"]))
            ax.plot(x, fb, color=BEN, lw=1.6, label=f"benign: {100 * t['B']['abn']:.1f}% abnormal")
            if t["pi"] > 0:
                fp = np.exp(G._mix_logpdf(x, c, t["P"]))
                ax.plot(x, fp, color=PATH, lw=1.6, label=f"pathogenic: {100 * t['P']['abn']:.0f}% abnormal")
            else:
                ax.text(0.03, 0.55, "no pathogenic variants\nremain visible or kept", transform=ax.transAxes, fontsize=7.5)
            ax.set_title(f"{assay} · {csq}   π = {t['pi']:.3g}", fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="upper left")
            ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
            if i == 1:
                ax.set_xlabel("assay score (κ = %.1f)" % c["kappa"])
    fig.suptitle("Consequence-cell simulation: true pathogenicity vs functional abnormality vs score", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/data/ross/assay_calibration/consequence_cells_sim/schematic.png")
    main(ap.parse_args().out)
