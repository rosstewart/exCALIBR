"""Preliminary figures for the consequence paper, built from existing results only (2026-11-20).

All four are marked PRELIMINARY. P1 and P4 use the experimental cell-model runs: in-bag, lp_anchored, 500 bootstraps
x 8 restarts, Δγ from the boundary-at-points estimator. P2 and P3 use the production ExCALIBR calibrations (OOB,
Tavtigian), i.e. the manuscript numbers.

    python -m csq_analysis.prelim [p1 p2 p3 p4]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from csq_analysis import config as C  # noqa: E402

INK, MUTED, GRID = "#2b2b2a", "#6b6a64", "#e4e3de"
# consequence identity: fixed order, slots of the reference categorical palette
CSQ_COL = {"missense": "#eb6834", "PTV": "#1baf7a", "synonymous": "#2a78d6", "splice": "#9b59b6",
           "intronic": "#b8860b", "UTR": "#8c8c8c", "inframe": "#d4a017", "other": "#5f5e5a"}
EV = {"benign": "#2a78d6", "indeterminate": "#c9c8c0", "pathogenic": "#e34948"}
TAG = "PRELIMINARY"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=2)


def _stamp(fig, text):
    fig.get_layout_engine().set(rect=(0, 0.035, 1, 0.965))
    fig.text(0.995, 0.003, f"{TAG}: {text}", ha="right", va="bottom", fontsize=7, color=MUTED)


# ── P1 ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def p1_cases():
    from analysis.consequence_cells_sim import plot_mech_overlap as P
    out = C.PRELIM / "P1_cases.png"
    P.main(out=out)
    return out


# ── P2 ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def composition_table(rows=None):
    """Per dataset x sample x consequence counts of production scoreset rows."""
    from csq_analysis.annotate import load_rows
    r = load_rows() if rows is None else rows
    parts = []
    for samp, col in (("P/LP", "plp"), ("B/LB", "blb"), ("gnomAD", "pop"), ("assayed", None)):
        sel = r[r[col]] if col else r
        parts.append(sel.groupby(["dataset", "csq"]).size().rename("n").reset_index().assign(sample=samp))
    t = pd.concat(parts, ignore_index=True)
    full = pd.MultiIndex.from_product([sorted(r.dataset.unique()), ["P/LP", "B/LB", "gnomAD", "assayed"],
                                       C.CLASS_ORDER], names=["dataset", "sample", "csq"])
    t = t.set_index(["dataset", "sample", "csq"]).reindex(full, fill_value=0).reset_index()
    return t


def p2_composition():
    from analysis.plot_common import display_name  # noqa: F401  (datasets are not named in this figure)
    t = composition_table()
    t.to_csv(C.PRELIM / "P2_composition.csv", index=False)
    n_ds = t.dataset.nunique()
    show = {"P/LP": ["missense", "PTV", "splice", "synonymous", "other"],
            "B/LB": ["missense", "synonymous", "intronic", "splice", "other"],
            "gnomAD": ["missense", "PTV", "synonymous", "intronic"],
            "assayed": ["missense", "PTV", "synonymous"]}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True, layout="constrained")
    xs = np.unique(np.round(np.logspace(0, 3.5, 60)).astype(int))
    for ax, (samp, csqs) in zip(axes, show.items()):
        for c in csqs:
            v = t[(t["sample"] == samp) & (t.csq == c)].n.to_numpy()
            ax.step(xs, [(v >= x).sum() for x in xs], where="post", color=CSQ_COL[c], lw=1.8, label=c)
            ax.annotate(f"{int((v >= 20).sum())}", xy=(20, (v >= 20).sum()), xytext=(3, 2),
                        textcoords="offset points", fontsize=7, color=CSQ_COL[c])
        for x in (3, 10, 20):
            ax.axvline(x, color=GRID, lw=1, zorder=0)
            ax.text(x, n_ds + 1.5, f"n={x}", fontsize=7, color=MUTED, ha="center")
        ax.set_xscale("log")
        ax.set_xlim(1, 3000)
        ax.set_ylim(0, n_ds + 5)
        ax.set_title(samp if samp != "assayed" else "all assayed", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("minimum variants per dataset", fontsize=8.5, color=MUTED)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
        ax.grid(axis="y", color=GRID, lw=0.5)
        _style(ax)
    axes[0].set_ylabel(f"datasets (of {n_ds}) with at least n", fontsize=8.5, color=MUTED)
    fig.suptitle("Control availability by consequence (labels at n = 20: datasets with ≥ 20)", fontsize=11,
                 color=INK, x=0.01, ha="left")
    _stamp(fig, "production scoreset rows after splice filtering; ClinVar 2025 (2018 for 4 genes)")
    out = C.PRELIM / "P2_composition.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ── P3 ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def production_variants():
    """Production OOB points of every canonical dataset (as analyze_pipeline_output A1d), joined to the per-row
    consequence annotations. Cached."""
    f = C.CACHE / "production_variants.pkl"
    if f.exists():
        return pd.read_pickle(f)
    import json
    from pathlib import Path
    from analysis import config as acfg
    from analysis.discovery import discover_outputs, load_all_variants
    from analysis.evidence import build_clinvar_multilabel
    from analysis.plot_common import effective_points
    from csq_analysis.annotate import load_rows
    tree, ms, cals = discover_outputs(Path(acfg.OUTPUT_DIR))
    dc = json.load(open(acfg.DATASET_CONFIGS))
    df = load_all_variants(tree=tree, model_selections=ms, dataset_configs=dc, methods_filter=None,
                           datasets_filter=None, calibrations=cals, min_controls=0)
    pm = sorted(df["method"].unique())[0]
    parts = []
    for d in sorted(df["dataset"].unique()):
        s = df[(df["method"] == pm) & (df["dataset"] == d)]
        if s.empty:
            continue
        lab = build_clinvar_multilabel(s)[:, :5].astype(bool)
        parts.append(pd.DataFrame(dict(dataset=d, variant_id=s["variant_id"].to_numpy(),
                                       points=effective_points(s, True, label=d, context="csq_prelim").values.astype(int),
                                       blb=lab[:, 0], vus=lab[:, 1], plp=lab[:, 2], popn=lab[:, 3], syn=lab[:, 4])))
    v = pd.concat(parts, ignore_index=True)
    rows = load_rows()[["dataset", "variant_id", "csq", "simplified_consequence"]]
    n0 = len(v)
    v = v.merge(rows.drop_duplicates(["dataset", "variant_id"]), on=["dataset", "variant_id"], how="left")
    assert len(v) == n0
    miss = v.csq.isna().sum()
    print(f"production variants: {len(v)} rows, {v.dataset.nunique()} datasets, {miss} without consequence")
    v["csq"] = v.csq.fillna("unannotated")
    C.CACHE.mkdir(parents=True, exist_ok=True)
    v.to_pickle(f)
    return v


def p3_errors():
    v = production_variants()
    v = v[v.dataset.isin(C.canonical_datasets())]
    ev = np.where(v.points < 0, "benign", np.where(v.points > 0, "pathogenic", "indeterminate"))
    v = v.assign(ev=ev)
    tab = (v[v.plp | v.blb].assign(label=np.where(v[v.plp | v.blb].plp, "P/LP", "B/LB"))
           .groupby(["label", "csq", "ev"]).size().unstack("ev", fill_value=0))
    tab.to_csv(C.PRELIM / "P3_errors_by_consequence.csv")
    fn, fp = int((v.plp & (v.points < 0)).sum()), int((v.blb & (v.points > 0)).sum())
    print(f"P3 check: FN {fn} (manuscript 586), FP {fp} (manuscript 255)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, lab, bad in zip(axes, ("P/LP", "B/LB"), ("benign", "pathogenic")):
        t = tab.loc[lab].reindex(columns=["pathogenic", "indeterminate", "benign"], fill_value=0)
        t = t[t.sum(axis=1) > 0]
        t = t.loc[[c for c in C.CLASS_ORDER + ["unannotated"] if c in t.index]][::-1]
        tot = t.sum(axis=1)
        left = np.zeros(len(t))
        for e in ("pathogenic", "indeterminate", "benign"):
            frac = (t[e] / tot).to_numpy()
            ax.barh(np.arange(len(t)), frac, left=left, color=EV[e], height=0.7, edgecolor="white", linewidth=1.5,
                    label=e)
            left += frac
        for i, (c, row) in enumerate(t.iterrows()):
            ax.text(1.01, i, f"n={int(tot[c])}  {('FN' if lab == 'P/LP' else 'FP')} {int(row[bad])}",
                    va="center", fontsize=8, color=INK)
        ax.set_yticks(np.arange(len(t)), t.index, fontsize=9, color=INK)
        ax.set_xlim(0, 1.32)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("share of controls by OOB evidence direction", fontsize=8.5, color=MUTED)
        wrong = "benign evidence (FN)" if lab == "P/LP" else "pathogenic evidence (FP)"
        ax.set_title(f"{lab} controls: {int(t[bad].sum())} with {wrong}", fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.05), ncol=3)
    fig.suptitle("Production ExCALIBR evidence on ClinVar controls, by variant consequence (87 datasets)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    _stamp(fig, "production calibrations, OOB points, Tavtigian; counts are per measurement (a variant in k assays counts k times)")
    out = C.PRELIM / "P3_errors_by_consequence.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ── P4 ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def p4_pooling():
    from analysis.consequence_cells_sim.plot_mech_overlap import load
    from analysis.plot_common import display_name
    d = load()
    d = d[d.dataset.isin(C.canonical_datasets())]
    d.to_csv(C.PRELIM / "P4_pooling_cost.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1.6]))
    groups = [("flagged", d[d.flag]), ("not flagged", d[~d.flag])]
    for ax, (metric, name) in zip(axes[:2], (("fn", "P/LP missense with benign evidence"),
                                             ("fp", "B/LB missense with pathogenic evidence"))):
        w = 0.36
        for i, (g, s) in enumerate(groups):
            a, b = int(s[f"{metric}_cell_all"].sum()), int(s[f"{metric}_cell_mis"].sum())
            ax.bar(i - w / 2, a, w * 0.94, color="#5f5e5a", label="pooled P/LP (cell_all)" if i == 0 else None)
            ax.bar(i + w / 2, b, w * 0.94, color="#eb6834", label="missense P/LP (cell_mis)" if i == 0 else None)
            for x, y in ((i - w / 2, a), (i + w / 2, b)):
                ax.text(x, y, f"{y}", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks([0, 1], [f"{g}\n({len(s)} datasets)" for g, s in groups], fontsize=8.5, color=INK)
        ax.set_title(name, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[2]
    s = d[d.flag | (d.fn_cell_all - d.fn_cell_mis).abs().ge(5)].sort_values("dfn_cell")
    y = np.arange(len(s))
    ax.hlines(y, s.fn_cell_mis, s.fn_cell_all, color=GRID, lw=2)
    ax.scatter(s.fn_cell_all, y, s=40, color="#5f5e5a", zorder=3, label="pooled")
    ax.scatter(s.fn_cell_mis, y, s=40, color="#eb6834", zorder=3, label="missense P/LP")
    ax.set_yticks(y, [display_name(x).replace("_", " ") + (" *" if f else "") for x, f in zip(s.dataset, s.flag)],
                  fontsize=7.5, color=INK)
    ax.set_xlabel("P/LP missense with benign evidence", fontsize=8.5, color=MUTED)
    ax.set_title("Datasets flagged (*) or with |change| ≥ 5", fontsize=9.5, color=INK, loc="left")
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlim(-0.5, max(s.fn_cell_all.max(), s.fn_cell_mis.max()) * 1.4)
    ax.set_xticks([0, 5, 10, 50, 100, 200], ["0", "5", "10", "50", "100", "200"])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.suptitle("Cost of pooling PTV P/LP for missense controls, by overlap flag (flag: Δγ q95 < −0.1 or TV_D q05 > 0.2)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    _stamp(fig, "cell model, in-bag, lp_anchored, 500 bootstraps x 8 restarts; Δγ from the boundary-at-points estimator")
    out = C.PRELIM / "P4_pooling_cost.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    C.PRELIM.mkdir(parents=True, exist_ok=True)
    todo = sys.argv[1:] or ["p1", "p2", "p3", "p4"]
    for k in todo:
        print(k, globals()[{"p1": "p1_cases", "p2": "p2_composition", "p3": "p3_errors", "p4": "p4_pooling"}[k]]())


if __name__ == "__main__":
    main()
