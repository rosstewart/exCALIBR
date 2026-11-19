"""Real-data figure for delta-gamma and detected-density TV (experimental, 2026-11-19).

Panel A: every testable canonical dataset at (delta gamma, TV_D), bootstrap 5-95% ranges, marker area = P/LP missense
moving to benign evidence when PTVs are pooled (cell_all minus cell_mis). Panels B-G: six datasets at the extremes, each
with the score histograms (B/LB, P/LP missense, P/LP PTV), the unmixed detected densities f_D (pointwise bootstrap
median, dashed), and the evidence of missense controls under the cell model with all P/LP (cell_all) vs missense P/LP
(cell_mis).

    python -m analysis.consequence_cells_sim.plot_mech_overlap
"""
from __future__ import annotations

import glob
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.consequence_cells_sim import mech_overlap as MO  # noqa: E402

COL = {"B/LB": "#2a78d6", "P/LP missense": "#eb6834", "P/LP PTV": "#1baf7a"}
EV = {"benign": "#2a78d6", "indeterminate": "#c9c8c0", "pathogenic": "#e34948"}
INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"
DG_CUT, TV_CUT = -0.1, 0.2
EXAMPLES = [
    ("BRCA2_Huang_2025_SGE", "Both small: pooling is harmless"),
    ("LDLR_Tabet_2025_abundance", "Missense detected far less (Δγ ≪ 0)"),
    ("DDX3X_Radford_2023", "Moderate on both, not flagged"),
    ("GCK_Gersing_2024_abundance", "Detected densities differ (TV_D high)"),
    ("TP53_Giacomelli_2018_combined_score_clinvar_2018", "PTVs detected less (Δγ ≫ 0), same f_D"),
    ("TP53_Boettcher_2019_clinvar_2018", "Both large, opposite direction"),
]


def _style(ax, yticks=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=2)
    if not yticks:
        ax.set_yticks([])


def load():
    from analysis.discovery import DEFAULT_EXCLUDED_DATASETS as EX
    d = pd.read_csv(MO.OUT_REAL / "mech_overlap_real.csv")
    d = d[(d.call != "untestable") & ~d.dataset.isin(EX)].copy()
    d["flag_gamma"] = d.d_gamma_q95 < DG_CUT
    d["flag_tv"] = d.tv_D_q05 > TV_CUT
    d["flag"] = d.flag_gamma | d.flag_tv
    d["dfn_cell"] = d.fn_cell_all - d.fn_cell_mis
    return d


def fd_curves(ds):
    if ds not in fd_curves.cache:
        fd_curves.cache[ds] = _fd_curves(ds)
    return fd_curves.cache[ds]


fd_curves.cache = {}


def _fd_curves(ds):
    """Scores by group, grid, pointwise-median f_D for PTV and missense, and gamma medians."""
    from analysis.membership_cells.cells import Cells
    c = Cells.load(MO.LARGE / "cells" / f"{ds}.npz")
    t = c.table
    n = t["n"].to_numpy(float)
    m = np.array([list(x) for x in t["mask"]])
    csq = t["consequence"].to_numpy()
    sel = dict(ptv=m[:, 0] & np.isin(csq, MO.REAL_PTV), mis=m[:, 0] & (csq == "missense_variant"), blb=m[:, 1],
               syn=m[:, 3])
    rows, rc = m[c.cell_of_row], csq[c.cell_of_row]
    groups = {"B/LB": c.scores[rows[:, 1]], "P/LP missense": c.scores[rows[:, 0] & (rc == "missense_variant")],
              "P/LP PTV": c.scores[rows[:, 0] & np.isin(rc, MO.REAL_PTV)]}
    scores = dict(ptv=groups["P/LP PTV"], mis=groups["P/LP missense"])
    lo, hi = np.nanpercentile(c.scores, [0.5, 99.5])
    pad = 0.05 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 600)
    K = MO._real_k(ds)
    bm = json.load(open(MO.LARGE / f"config_{K}.json")).get(ds, {}).get("benign_method", "avg")
    fd = {"ptv": [], "mis": []}
    for f in sorted(glob.glob(str(MO.LARGE / "raw" / f"{ds}__{K}__*.json.gz"))):
        for rec in json.load(gzip.open(f)).values():
            if not rec or "error" in rec:
                continue
            fit = rec[K]["fit"]
            o = MO.one_fit(fit["component_params"], np.asarray(fit["weights"], float), n, sel, scores, grid, bm)
            for k in fd:
                if o[f"fD_{k}"] is not None:
                    fd[k].append(o[f"fD_{k}"])
    med = {k: (np.median(np.array(v), axis=0) if v else None) for k, v in fd.items()}
    return groups, grid, med, (lo, hi), K


def evidence(ds, K):
    """Counts of benign / indeterminate / pathogenic in-bag points for missense P/LP and B/LB, cell_all vs cell_mis."""
    from analysis.membership_cells.cells import Cells
    from analysis.membership_cells.metrics import _pts
    c = Cells.load(MO.LARGE / "cells" / f"{ds}.npz")
    t = c.table
    m = np.array([list(x) for x in t["mask"]])[c.cell_of_row]
    mis = np.array([str(x).startswith("missense_variant") for x in t["consequence"]])[c.cell_of_row]
    ids = pd.Index(c.variant_ids)
    out = {}
    for arm in ("cell_all", "cell_mis"):
        p = _pts(MO.LARGE / f"calib_{arm}_{K}", ds)
        for lab, col in (("P/LP", 0), ("B/LB", 1)):
            x = p.reindex(ids[m[:, col] & mis]).dropna() if p is not None else pd.Series(dtype=float)
            out[(lab, arm)] = dict(benign=int((x < 0).sum()), indeterminate=int((x == 0).sum()),
                                   pathogenic=int((x > 0).sum()))
    return out


def overview(ax, d):
    from analysis.plot_common import display_name
    ax.axvline(DG_CUT, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.axhline(TV_CUT, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.axvline(0, color=GRID, lw=0.8)
    for _, r in d.iterrows():
        c = "#e34948" if r.flag else "#8a8980"
        ax.plot([r.d_gamma_q05, r.d_gamma_q95], [r.tv_D_med] * 2, color=c, lw=0.8, alpha=0.5)
        ax.plot([r.d_gamma_med] * 2, [r.tv_D_q05, r.tv_D_q95], color=c, lw=0.8, alpha=0.5)
        size = 18 + 2.2 * max(r.dfn_cell, 0) if np.isfinite(r.dfn_cell) else 18
        ax.scatter(r.d_gamma_med, r.tv_D_med, s=size, color=c, edgecolor="white", linewidth=1, zorder=3)
    ex = dict(EXAMPLES)
    for _, r in d.iterrows():
        if r.dataset in ex or r.flag:
            lab = chr(ord("B") + list(ex).index(r.dataset)) + "  " if r.dataset in ex else ""
            ax.annotate(f"{lab}{display_name(r.dataset).replace('_', ' ')}", (r.d_gamma_med, r.tv_D_med),
                        xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK)
    ax.text(DG_CUT - 0.01, 0.62, "missense detected less ←", ha="right", fontsize=7.5, color=MUTED)
    ax.text(0.01, 0.62, "→ PTVs detected less", ha="left", fontsize=7.5, color=MUTED)
    ax.set_xlabel("Δγ = γ missense − γ PTV  (share of P/LP whose mechanism the assay detects)", fontsize=8.5, color=INK)
    ax.set_ylabel("TV_D  (detected pathogenic densities, PTV vs missense)", fontsize=8.5, color=INK)
    ax.set_ylim(-0.03, 1.08)
    _style(ax, yticks=True)
    f, nf = d[d.flag], d[~d.flag]
    return (f"A   Flag = Δγ upper bound < {DG_CUT} or TV_D lower bound > {TV_CUT} (red): "
                 f"{len(f)} of {len(d)} datasets.  P/LP missense with benign evidence, cell_all vs cell_mis: "
                 f"flagged {int(f.fn_cell_all.sum())} vs {int(f.fn_cell_mis.sum())}, "
                 f"not flagged {int(nf.fn_cell_all.sum())} vs {int(nf.fn_cell_mis.sum())}.\n"
                 "Point = bootstrap median, lines = 5-95%; area = extra P/LP missense with benign evidence when PTVs are pooled.")


def example(ax, axb, ds, title, letter, r):
    from analysis.plot_common import display_name
    groups, grid, med, (lo, hi), K = fd_curves(ds)
    bins = np.linspace(lo, hi, 30)
    for k, x in groups.items():
        x = x[(x >= lo) & (x <= hi)]
        if len(x):
            h, e = np.histogram(x, bins=bins, density=True)
            ax.stairs(h, e, color=COL[k], lw=1.4, alpha=0.75)
    for k, lab in (("ptv", "P/LP PTV"), ("mis", "P/LP missense")):
        if med[k] is not None:
            ax.plot(grid, med[k], color=COL[lab], lw=1.8, ls=(0, (4, 2)))
    _style(ax)
    ax.set_xlim(grid[0], grid[-1])
    ax.tick_params(labelbottom=False)
    ax.set_title(f"{letter}   {display_name(ds).replace('_', ' ')}  ({K})\n{title}\n"
                 f"γ PTV {r.gamma_ptv_med:.2f}, γ missense {r.gamma_mis_med:.2f}  →  Δγ {r.d_gamma_med:+.2f} "
                 f"[{r.d_gamma_q05:+.2f}, {r.d_gamma_q95:+.2f}]\n"
                 f"TV_D {r.tv_D_med:.2f} [{r.tv_D_q05:.2f}, {r.tv_D_q95:.2f}]   n = {len(groups['B/LB'])} / "
                 f"{len(groups['P/LP missense'])} / {len(groups['P/LP PTV'])}",
                 fontsize=7.8, color=INK, loc="left")
    ev = evidence(ds, K)
    from src.assay_calibration.plot_utils.utils import _draw_evidence_strip
    allx = np.concatenate(list(groups.values()))
    g = lambda lab, arm: "/".join(str(ev[(lab, arm)][k]) for k in ("benign", "indeterminate", "pathogenic"))  # noqa: E731
    rows = [("cell_all", "pooled (all P/LP)", f"missense P/LP {g('P/LP', 'cell_all')}, B/LB {g('B/LB', 'cell_all')}; "
                                              f"PTV P/LP {ptv_counts(ds, K, 'cell_all')}"),
            ("cell_mis", "missense P/LP", f"missense P/LP {g('P/LP', 'cell_mis')}, B/LB {g('B/LB', 'cell_mis')}"),
            ("cell_ptv", "PTV P/LP, pooled prior", f"PTV P/LP {ptv_counts(ds, K, 'cell_ptv')}")]
    for axs, (arm, name, counts) in zip(axb, rows):
        pr = {k: [list(r) for r in v] for k, v in point_ranges(ds, K, arm).items()}
        _draw_evidence_strip(axs, pr, grid[0], grid[-1], allx)
        axs.set_xticks([])
        for sp in axs.spines.values():
            sp.set_visible(False)
        axs.text(0.0, 1.04, f"{name}:  {counts}", transform=axs.transAxes, fontsize=6.8, color=INK, va="bottom")
    axb[-1].set_xticks(ax.get_xticks())
    axb[-1].set_xlim(grid[0], grid[-1])
    axb[-1].tick_params(colors=MUTED, labelsize=7.5, length=2)


def _stored(ds, K, arm):
    f = glob.glob(str(MO.LARGE / f"calib_{arm}_{K}" / ds / "*_calibration.json"))
    return json.load(open(f[0])) if f else {}


_PTV_CACHE = {}


def point_ranges(ds, K, arm):
    if arm == "cell_ptv":     # PTV P/LP cells, prior fixed at the stored cell_all prior
        if ds not in _PTV_CACHE:
            cal = MO.cell_arm_calibration(ds, "ptv", manual_prior=float(_stored(ds, K, "cell_all")["prior"]))
            _PTV_CACHE[ds] = cal["point_ranges"] if cal else {}
        pr = _PTV_CACHE[ds]
    else:
        pr = _stored(ds, K, arm).get("point_ranges", {})
    return {int(k): [(float(a), float(b)) for a, b in v] for k, v in pr.items() if v}


def ptv_counts(ds, K, arm):
    """benign/indeterminate/pathogenic counts of P/LP PTV controls under an arm's point ranges."""
    from src.assay_calibration.fit_utils.fit import assign_points
    groups = fd_curves.cache[ds][0]
    x = groups["P/LP PTV"]
    pts = np.asarray(assign_points(x, point_ranges(ds, K, arm)), int) if len(x) else np.array([], int)
    return f"{int((pts < 0).sum())}/{int((pts == 0).sum())}/{int((pts > 0).sum())}"


def main():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    d = load().set_index("dataset", drop=False)
    fig = plt.figure(figsize=(15, 17), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 2.05], hspace=0.03)
    ax0 = top.subplots()
    txt = overview(ax0, d)
    ax0.text(0.01, 0.97, txt.replace("A   ", "").replace("datasets.  ", "datasets.\n").replace(": flagged", ":\nflagged"),
             transform=ax0.transAxes, fontsize=8.5, color=INK, va="top", ha="left")
    ax0.set_title("A", fontsize=10, color=INK, loc="left", fontweight="bold")
    subs = bottom.subfigures(2, 3, wspace=0.02, hspace=0.04)
    for i, (ds, title) in enumerate(EXAMPLES):
        sf = subs.ravel()[i]
        axes = sf.subplots(4, 1, gridspec_kw=dict(height_ratios=[3, 0.36, 0.36, 0.36], hspace=0.55))
        example(axes[0], axes[1:], ds, title, chr(ord("B") + i), d.loc[ds])
    hs = [Line2D([], [], color=COL[k], lw=1.6, label=k) for k in COL]
    hs += [Line2D([], [], color=MUTED, lw=1.8, ls=(0, (4, 2)), label="f_D (unmixed detected density)")]
    from src.assay_calibration.plot_utils.utils import EVIDENCE_STRENGTH_COLORS as ESC
    hs += [Patch(color=ESC[v], label=f"{v:+d}" if v else "0") for v in (-8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8)]
    leg = ax0.legend(handles=hs, loc="center right", ncol=2, frameon=False, fontsize=8, title="score histograms, f_D, evidence points",
                     title_fontsize=8, bbox_to_anchor=(1.0, 0.7))
    leg._legend_box.align = "left"
    fig.get_layout_engine().set(rect=(0, 0.02, 1, 1))
    out = MO.OUT_REAL / "mech_overlap_examples_real.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
