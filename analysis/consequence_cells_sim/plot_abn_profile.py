"""Example score distributions behind the abnormal-profile overlap test (experimental, 2026-11-19).

Real data: three rows (overlap declared / not declared, bound 0.2-0.45 / clearly different, median TV > 0.4), three
datasets each, from the large-labeled cell tables. Simulation: one replicate per case from the cells-only refits,
with the true pathogenic PTV and missense densities overlaid.

    python -m analysis.consequence_cells_sim.plot_abn_profile
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/data/ross/assay_calibration")
CELLS = ROOT / "missense_only_calibrations/cells/hpc_large_labeled/labeled/cells"
REAL = ROOT / "consequence_cells_sim/real_overlap/abn_profile_real_with_outcomes.csv"
SIM = ROOT / "consequence_cells_sim/hpc_cellfits"
# fixed categorical order (slots 1-3 of the reference palette)
COL = {"B/LB": "#2a78d6", "P/LP missense": "#eb6834", "P/LP PTV": "#1baf7a"}
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"
PTV = ("stop_gained", "frameshift_variant")

ROWS = [("Overlap declared (bound < 0.2)",
         ["BRCA2_Huang_2025_SGE", "BRCA1_Findlay_2018_clinvar_2018", "TP53_Funk_2025_clinvar_2018"]),
        ("Not declared: uncertain (bound 0.2-0.45, median distance < 0.2)",
         ["KCNH2_O_Neill_2024_surface_expression", "HMBS_van_Loggerenberg_2023_erythroid", "FKRP_Ma_2024"]),
        ("Clearly different (median distance > 0.4)",
         ["LDLR_Tabet_2025_abundance", "GCK_Gersing_2024_abundance", "TP53_Boettcher_2019_clinvar_2018"])]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=2)
    ax.set_yticks([])


def _hist(ax, x, bins, label, lw=1.6):
    if len(x) == 0:
        return
    h, e = np.histogram(x, bins=bins, density=True)
    ax.stairs(h, e, color=COL[label], lw=lw)


def _groups_real(ds):
    from analysis.membership_cells.cells import Cells
    c = Cells.load(CELLS / f"{ds}.npz")
    t = c.table
    mask = np.array([list(m) for m in t["mask"]])[c.cell_of_row]
    csq = t["consequence"].to_numpy()[c.cell_of_row]
    sc = c.scores
    return {"B/LB": sc[mask[:, 1]],
            "P/LP missense": sc[mask[:, 0] & (csq == "missense_variant")],
            "P/LP PTV": sc[mask[:, 0] & np.isin(csq, PTV)]}


def _legend(fig, extra=None):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=COL[k], lw=2, label=k) for k in COL]
    if extra:
        hs.append(Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 2)), label=extra))
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))


def _n(g):
    return " / ".join(str(len(g[k])) for k in COL) + " (B/LB / missense / PTV)"


def real_figure(out):
    from analysis.plot_common import display_name
    r = pd.read_csv(REAL).set_index("dataset")
    fig = plt.figure(figsize=(12, 10.5), layout="constrained")
    subs = fig.subfigures(3, 1, hspace=0.04)
    for i, (title, dss) in enumerate(ROWS):
        subs[i].suptitle(title, fontsize=10.5, color=INK, fontweight="bold", x=0.01, ha="left")
        row_axes = subs[i].subplots(1, 3)
        for j, ds in enumerate(dss):
            ax = row_axes[j]
            g = _groups_real(ds)
            allx = np.concatenate(list(g.values()))
            lo, hi = np.nanpercentile(allx, [0.5, 99.5])
            bins = np.linspace(lo, hi, 30)
            for k in ("B/LB", "P/LP missense", "P/LP PTV"):
                _hist(ax, g[k][(g[k] >= lo) & (g[k] <= hi)], bins, k)
            _style(ax)
            x = r.loc[ds]
            ax.set_title(f"{display_name(ds).replace('_', ' ')}\n"
                         f"distance {x.tv_med:.2f} (bound {x.tv_up:.2f}); n = {_n(g)}\n"
                         f"P/LP missense with benign evidence: {int(x.fn_std_all)} pooled, {int(x.fn_std_mis)} missense-only",
                         fontsize=8, color=INK, loc="left")
            ax.set_xlabel("Assay score", fontsize=8, color=MUTED)
    _legend(fig)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def sim_figure(out):
    from scipy.stats import norm
    from analysis.consequence_cells_sim import generate as G
    a = pd.read_csv(SIM / "abn_profile.csv")
    picks = [("Clearly different: PTVs 2.5x further out (abundance, κ 2, large)",
              "ptvloc__abundance_large_realistic_k2_ptv_loc2.5", None),
             ("Different but mild: PTVs 1.5x further out (fitness, κ 3.5, large)",
              "ptvloc__fitness_large_realistic_k3.5_ptv_loc1.5", None),
             ("Different, pooling costs: PTVs 1.5x further out (abundance, κ 3.5, large)",
              "ptvloc__abundance_large_realistic_k3.5_ptv_loc1.5", None),
             ("Only false overlap call (hypomorph slice, fitness)",
              "slice__fitness_mid_realistic_k3.5_hypomorphTrue_nu0.3", "overlap")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), layout="constrained")
    for ax, (title, cid, how) in zip(axes.ravel(), picks):
        x = a[a.cell_id == cid]
        row = x[x.overlap].iloc[0] if how == "overlap" else x.iloc[(x.tv_med - x.tv_med.median()).abs().argsort().iloc[0]]
        c = next(cc for cc in G.design() if cc["cell_id"] == cid)
        d = G.make_data(G.rng_for(c, int(row.rep)), c)
        sc, m, csq = d["score"], d["mask"], d["csq"]
        g = {"B/LB": sc[m[:, 1]], "P/LP missense": sc[m[:, 0] & (csq == "missense")],
             "P/LP PTV": sc[m[:, 0] & (csq == "PTV")]}
        bins = np.linspace(-c["kappa"] * c.get("ptv_loc", 1.0) - 3, 3.5, 32)
        for k in g:
            _hist(ax, g[k], bins, k)
        xs = np.linspace(bins[0], bins[-1], 400)
        loc = dict(ptv=-c["kappa"] * c.get("ptv_loc", 1.0), abn=-c["kappa"], int=-c["kappa"] / 2, nor=0.0)
        for csq_name, lab in (("PTV", "P/LP PTV"), ("missense", "P/LP missense")):
            P = G.truth(c, csq_name)["P"]
            ax.plot(xs, sum(P[k] * norm.pdf(xs, loc[k], 1) for k in P), color=COL[lab], lw=1, ls=(0, (3, 2)))
        _style(ax)
        pb = row["pooled__missense_path_benign"]
        cost = (f"\nP/LP missense with benign evidence: {pb:.0%} pooled, "
                f"{row['cell_mis__missense_path_benign']:.0%} cell-model missense") if np.isfinite(pb) else ""
        ax.set_title(f"{title}\nreplicate {int(row.rep)}: distance {row.tv_med:.2f} (bound {row.tv_up:.2f}), "
                     f"true {row.true_tv:.2f}; n = {_n(g)}{cost}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("Assay score", fontsize=8, color=MUTED)
    _legend(fig, "true pathogenic density")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    out = ROOT / "consequence_cells_sim/real_overlap"
    real_figure(out / "abn_profile_examples_real.png")
    sim_figure(SIM / "abn_profile_examples_sim.png")
    print(out / "abn_profile_examples_real.png", SIM / "abn_profile_examples_sim.png")


if __name__ == "__main__":
    main()
