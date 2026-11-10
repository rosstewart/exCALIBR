"""Per-dataset metric violins for membership-cell model configurations, in the style of manuscript
Extended Data Figure 4 (analysis.comparison_diagnostics.plot_comparison_boxplots).

Groups (each compared on its own matched datasets). Missense-P/LP models are evaluated on missense controls only;
all-P/LP models on both control sets:
  Missense controls          std_all, std_mis, cell_all / cell_mis at labeled and assay scale (large runs, 500 bootstraps)
  All controls               std_all, cell_all at labeled and assay scale
  K test, cell_all, missense / all controls, and K test, cell_mis, missense controls:
                             labeled K = 3, 4 and assay K = 3, 4, 5 (11 datasets, 20 bootstraps)
In-bag standard points, lp_anchored mapping, splice-filtered rows excluded (splice == False).

Outputs (in <large labeled>/): cell_config_violins.png (accuracy, DOR, determinate %) and
cell_config_violins_mcc.png (MCC, sensitivity and specificity among determinate calls).
Usage:  python -m analysis.membership_cells.plot_violins
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import analysis.comparison_diagnostics as cd

C = Path("/data/ross/assay_calibration/missense_only_calibrations/cells")
LAB = C / "hpc_large_labeled" / "labeled"
ASSAY = C / "hpc_large_assay" / "assay"
KTEST = C / "k_test"

DISPLAY = {
    "std_all": "Standard, all P/LP", "std_mis": "Standard, missense P/LP",
    "cell_all_lab": "Cells, all P/LP, labeled", "cell_mis_lab": "Cells, missense P/LP, labeled",
    "cell_all_assay": "Cells, all P/LP, assay", "cell_mis_assay": "Cells, missense P/LP, assay",
    "lab_K3": "labeled, K = 3", "lab_K4": "labeled, K = 4",
    "assay_K3": "assay, K = 3", "assay_K4": "assay, K = 4", "assay_K5": "assay, K = 5",
}


def _matrix(r):
    """2x3 confusion matrix in compute_classification_metrics' layout: rows B/LB, P/LP;
    columns Benign, Indeterminate, Pathogenic."""
    if r is None or (r["n_plp"] + r["n_blb"]) == 0:
        return None
    return pd.DataFrame([[r["tn"], r["ir_blb"], r["fp"]], [r["fn"], r["ir_plp"], r["tp"]]],
                        index=["B/LB", "P/LP"], columns=["Benign", "Indeterminate", "Pathogenic"])


def _lookup(df, keys):
    return {tuple(k): r for k, r in zip(df[keys].itertuples(index=False), df.to_dict("records"))}


def build_groups():
    lab = pd.read_csv(LAB / "metrics_inbag.csv")
    assay = pd.read_csv(ASSAY / "metrics_inbag.csv")
    lab, assay = lab[~lab.splice.astype(bool)], assay[~assay.splice.astype(bool)]
    kt = pd.read_csv(KTEST / "eval_k.csv")
    L, A = _lookup(lab, ["dataset", "arm", "controls"]), _lookup(assay, ["dataset", "arm", "controls"])
    K = _lookup(kt, ["dataset", "arm", "scale", "K", "controls"])
    datasets = sorted(set(lab.dataset) | set(kt.dataset))

    groups = {}
    src = {"std_all": (L, "std_all"), "std_mis": (L, "std_mis"),
           "cell_all_lab": (L, "cell_all"), "cell_mis_lab": (L, "cell_mis"),
           "cell_all_assay": (A, "cell_all"), "cell_mis_assay": (A, "cell_mis")}
    # missense-P/LP models (std_mis, cell_mis) are evaluated on missense controls only
    for ctl, title, keep in (("missense", "Missense controls", list(src)),
                             ("all", "All controls (all-P/LP models only)", ["std_all", "cell_all_lab", "cell_all_assay"])):
        groups[title] = {m: [_matrix(src[m][0].get((ds, src[m][1], ctl))) for ds in datasets] for m in keep}
    runs = (("labeled", "K3"), ("labeled", "K4"), ("assay", "K3"), ("assay", "K4"), ("assay", "K5"))
    for arm, ctl, title in (("cell_all", "missense", "K test: cells, all P/LP; missense controls"),
                            ("cell_all", "all", "K test: cells, all P/LP; all controls"),
                            ("cell_mis", "missense", "K test: cells, missense P/LP; missense controls")):
        groups[title] = {f"{'lab' if s == 'labeled' else 'assay'}_{k}": [_matrix(K.get((ds, arm, s, k, ctl))) for ds in datasets]
                         for s, k in runs}
    return groups, datasets


def _mcc_figure(groups, datasets):
    import matplotlib.pyplot as plt
    metrics = (("mcc", "MCC"), ("sens", "Sensitivity (determinate)"), ("spec", "Specificity (determinate)"))
    fig, axs = plt.subplots(len(metrics), 1, figsize=(24, 10.5), sharex=True)
    x, ticks, labels, bounds = 1.0, [], [], []
    colors = cd._method_colors([m for g in groups.values() for m in g])
    placed = []
    for gi, (title, mats) in enumerate(groups.items()):
        methods = list(mats)
        idx = [i for i in range(len(datasets)) if all(mats[m][i] is not None for m in methods)]
        pos = [x + j for j in range(len(methods))]
        placed.append((title, methods, idx, pos))
        x = pos[-1] + 1.6
        if gi < len(groups) - 1:
            bounds.append(pos[-1] + 0.8)
    for (key, ylab), ax in zip(metrics, axs):
        for title, methods, idx, pos in placed:
            for m, p in zip(methods, pos):
                vals, agg = [], np.zeros(4)
                for i in idx:
                    M = groups[title][m][i].values
                    tn, fp, fn, tp = M[0, 0], M[0, 2], M[1, 0], M[1, 2]
                    agg += (tn, fp, fn, tp)
                    vals.append(_rate(key, tn, fp, fn, tp))
                vals = np.array([v for v in vals if np.isfinite(v)])
                if len(vals) > 1:
                    vp = ax.violinplot([vals], positions=[p], showmedians=True, widths=0.8)
                    cd._style_violin(vp, [m], colors)
                a = _rate(key, *agg)
                med = np.median(vals) if len(vals) else np.nan
                ax.text(p, 1.04, f"median: {med:.2f}\nagg.: {a:.2f}", ha="center", va="bottom", fontsize=6.5,
                        transform=ax.get_xaxis_transform())
                if key == "mcc":
                    ticks.append(p)
                    labels.append(f"{DISPLAY[m]}\n({len(idx)} datasets)")
            if key == "mcc":
                ax.text(np.mean(pos), 1.22, title, ha="center", va="bottom", fontsize=9, fontweight="bold",
                        transform=ax.get_xaxis_transform())
        for b in bounds:
            ax.axvline(b, color="lightgrey", lw=1.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.05 if key == "mcc" else 0, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axs[-1].set_xticks(ticks)
    axs[-1].set_xticklabels(labels, rotation=20, ha="right", rotation_mode="anchor", fontsize=8)
    fig.tight_layout(h_pad=3.5)
    return fig


def _group_titles(fig, groups):
    """plot_comparison_boxplots draws no group titles; add them above the top row, using its own x layout."""
    ax = fig.axes[0]
    x, gap = 1.0, 1.6
    for title, mats in groups.items():
        pos = [x + i for i in range(len(mats))]
        ax.text(np.mean(pos), 1.32, title, ha="center", va="bottom", fontsize=10, fontweight="bold",
                transform=ax.get_xaxis_transform())
        x = pos[-1] + 1 + gap


def _rate(key, tn, fp, fn, tp):
    if key == "sens":
        return tp / (tp + fn) if tp + fn else np.nan
    if key == "spec":
        return tn / (tn + fp) if tn + fp else np.nan
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / den if den else np.nan


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cd._METHOD_DISPLAY.update(DISPLAY)
    # one fixed, distinct colour per configuration (tab10 alone wraps and repeats)
    t20 = plt.get_cmap("tab20").colors
    cd._GLOBAL_METHOD_COLOR.update({m: t20[i] for i, m in enumerate(
        ["std_all", "std_mis", "cell_all_lab", "cell_mis_lab", "cell_all_assay", "cell_mis_assay"])})
    cd._GLOBAL_METHOD_COLOR.update({m: t20[i] for i, m in enumerate(
        ["lab_K3", "lab_K4", "assay_K3", "assay_K4", "assay_K5"], start=6)})
    groups, datasets = build_groups()
    fig = cd.plot_comparison_boxplots(groups, datasets, len(datasets))
    _group_titles(fig, groups)
    fig.savefig(LAB / "cell_config_violins.png", dpi=150, bbox_inches="tight")
    fig = _mcc_figure(groups, datasets)
    fig.savefig(LAB / "cell_config_violins_mcc.png", dpi=150, bbox_inches="tight")
    for title, mats in groups.items():
        n = sum(all(mats[m][i] is not None for m in mats) for i in range(len(datasets)))
        print(f"{title}: {n} matched datasets")
    print(LAB / "cell_config_violins.png")
    print(LAB / "cell_config_violins_mcc.png")


if __name__ == "__main__":
    main()
