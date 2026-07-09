"""
Figure helpers for pipeline-native analysis.

Wraps existing plot functions from src.assay_calibration.plot_utils.utils and
provides a method-comparison scatter matching the style of
plot_gene_level_performance_comparison (accuracy variant).
"""
from __future__ import annotations

import sys
import gzip
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.plot_utils.utils import (
    plot_aggregate_confusion_matrices,
    plot_combined_evidence_distributions,
    plot_evidence_by_clinvar_class_with_stats,
    compute_classification_metrics,
)
from src.assay_calibration.fit_utils.fit import thresholds_from_prior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, path: Path, dpi: int = 300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def _pretty(method: str) -> str:
    return {
        "tavtigian": "Tavtigian",
        "piecewise": "Piecewise",
        "continuous": "Continuous",
        "strict_additive": "Strict Additive",
        "default": "ExCALIBR",
    }.get(method, method.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Fig 1 – Aggregate confusion heatmap
# ---------------------------------------------------------------------------

def make_confusion_figure(
    danzs_m1: List,
    danzs_m2: List,
    dataset_names: List[str],
    label1: str,
    label2: str,
    figure_dir: Path,
    tag: str = "",
):
    """Aggregate confusion heatmap comparing two sets of per-dataset matrices.

    Uses plot_aggregate_confusion_matrices from plot_utils.utils with letters=True.
    The left panel is re-labelled label1, right panel label2.
    """
    fig, _, _ = plot_aggregate_confusion_matrices(
        danzs_m1, danzs_m2, dataset_names, letters=True
    )
    axes = [ax for ax in fig.get_axes() if hasattr(ax, "get_title")]
    if len(axes) >= 2:
        axes[0].set_title(_pretty(label1), fontsize=18, fontweight="bold", pad=10)
        axes[1].set_title(_pretty(label2), fontsize=18, fontweight="bold", pad=10)

    tag_suffix = f"_{tag}" if tag else ""
    _save(fig, figure_dir / f"confusion_heatmap_{label1}_vs_{label2}{tag_suffix}.png")


def make_single_confusion_figure(
    matrices: List,
    dataset_names: List[str],
    label: str,
    figure_dir: Path,
    tag: str = "",
):
    """Single-panel aggregate confusion heatmap (ClinVar rows x evidence-direction columns).

    Unlike make_confusion_figure, doesn't require a second matrix (author labels or
    another method) to compare against — used whenever a calibration confusion matrix
    should be plotted on its own.
    """
    aggregate = None
    n_datasets = 0
    for mat in matrices:
        if mat is None:
            continue
        aggregate = mat.copy() if aggregate is None else aggregate + mat
        n_datasets += 1

    if aggregate is None:
        print(f"  SKIP single confusion figure for {label}: no valid matrices")
        return

    metrics = compute_classification_metrics(aggregate)

    label_map = {"PLP": "P/LP", "BLB": "B/LB", "IR": "Indeterminate",
                 "Normal": "Benign", "Abnormal": "Pathogenic"}
    xlabels = [label_map.get(str(c), str(c)) for c in aggregate.columns]
    ylabels = [label_map.get(str(r), str(r)) for r in aggregate.index]

    colors = ["whitesmoke", "purple"]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("nature_purple", colors)

    max_val = aggregate.values.max()
    annot = aggregate.astype(str)
    for row in range(len(aggregate)):
        for col in range(len(aggregate.columns)):
            annot.iloc[row, col] = f"{aggregate.iloc[row, col]:,}"

    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        aggregate, annot=annot, fmt="", cmap=cmap, vmin=0, vmax=max_val,
        ax=ax, cbar_kws={"label": "Count"}, linewidths=2.5, linecolor="white",
        annot_kws={"fontsize": 13, "ha": "center", "va": "center"},
    )
    for text_obj in ax.texts:
        x, y = text_obj.get_position()
        row, col = int(y), int(x)
        if row < len(aggregate) and col < len(aggregate.columns):
            value = aggregate.iloc[row, col]
            text_obj.set_color("white" if value / max_val > 0.45 else "black")

    ax.set_xticklabels(xlabels, rotation=0, fontsize=11)
    ax.set_yticklabels(ylabels, rotation=0, fontsize=11)
    ax.set_xlabel("Evidence Direction", fontsize=12)
    ax.set_ylabel("ClinVar Classification", fontsize=12)
    ax.set_title(f"{_pretty(label)} ({n_datasets} datasets)", fontsize=14, fontweight="bold", pad=10)
    ax.text(
        0.5, -0.22,
        f"DOR: {metrics['dor_standard']:.1f}  |  Coverage: {100 * metrics['coverage']:.1f}%",
        transform=ax.transAxes, fontsize=10, ha="center", va="top", color="#555555",
    )

    tag_suffix = f"_{tag}" if tag else ""
    _save(fig, figure_dir / f"confusion_heatmap_{label}{tag_suffix}.png")


# ---------------------------------------------------------------------------
# Fig 2 – Combined evidence distribution (author + sample panels)
# ---------------------------------------------------------------------------

def make_evidence_figure(
    all_danz: np.ndarray,
    all_author: Optional[np.ndarray],
    all_clinvar: np.ndarray,
    label: str,
    figure_dir: Path,
):
    """Combined two-panel evidence distribution figure for one method.

    Top panel: author annotations (Normal / Indeterminate / Abnormal).
    Bottom panel: sample categories (B/LB, VUS, P/LP, gnomAD, Synonymous).

    When author labels are unavailable (all_author is None), falls back to a
    single-panel ClinVar/sample figure.
    """
    has_author = (
        all_author is not None
        and len(all_author) > 0
        and not np.all(all_author == 1)   # not entirely Indeterminate
    )

    if has_author:
        fig = plot_combined_evidence_distributions(
            author_assignments=all_danz,
            author_annotations=all_author,
            clinvar_assignments=all_danz,
            clinvar_classes=all_clinvar,
        )
    else:
        fig = plot_evidence_by_clinvar_class_with_stats(all_danz, all_clinvar)

    fig.suptitle(
        _pretty(label), fontsize=14, fontweight="bold",
        y=1.01 if has_author else 1.02,
    )
    _save(fig, figure_dir / f"evidence_distribution_{label}.png")


# ---------------------------------------------------------------------------
# Fig 3 – Per-gene accuracy scatter (method1 vs method2)
# ---------------------------------------------------------------------------

def make_scatter_figure(
    conf_by_method: Dict[str, List],
    dataset_names: List[str],
    method1: str,
    method2: str,
    figure_dir: Path,
    metric: str = "accuracy",
    tag: str = "",
):
    """Scatter of per-gene accuracy for method1 (y) vs method2 (x).

    Matches the style of plot_gene_level_performance_comparison with
    metric='accuracy'.  Point size scales with total variant count per gene.
    """
    if method1 not in conf_by_method or method2 not in conf_by_method:
        print(f"  SKIP scatter: missing matrices for {method1} or {method2}")
        return

    mats1 = conf_by_method[method1]
    mats2 = conf_by_method[method2]

    # --- Aggregate per gene ---
    gene_m1: Dict[str, pd.DataFrame] = {}
    gene_m2: Dict[str, pd.DataFrame] = {}
    gene_total: Dict[str, int] = {}

    for mat1, mat2, name in zip(mats1, mats2, dataset_names):
        if mat1 is None or mat2 is None:
            continue
        gene = name.split("_")[0]
        gene_m1[gene] = gene_m1[gene] + mat1 if gene in gene_m1 else mat1.copy()
        gene_m2[gene] = gene_m2[gene] + mat2 if gene in gene_m2 else mat2.copy()
        gene_total[gene] = gene_total.get(gene, 0) + int(mat1.values.sum())

    gene_results = []
    for gene in gene_m1:
        try:
            met1 = compute_classification_metrics(gene_m1[gene])
            met2 = compute_classification_metrics(gene_m2[gene])
        except Exception:
            continue
        gene_results.append({
            "gene": gene,
            "m1": met1[metric],
            "m2": met2[metric],
            "total": gene_total.get(gene, 100),
        })

    if not gene_results:
        print("  SKIP scatter: no valid gene results")
        return

    finite = [r for r in gene_results if r["m1"] not in (0, float("inf")) and r["m2"] not in (0, float("inf"))]
    undefined = [r for r in gene_results if r["m1"] == 0 or r["m2"] == 0]

    if not finite:
        print("  SKIP scatter: no finite gene results")
        return

    # --- Axis limits ---
    min_v = min(min(r["m1"] for r in finite), min(r["m2"] for r in finite))
    max_v = max(max(r["m1"] for r in finite), max(r["m2"] for r in finite))
    plot_min = max(0.5, min_v - 0.05)
    plot_max = min(1.0, max_v + 0.02)
    undefined_pos = plot_min * 0.97

    # --- Colours ---
    gene_list = sorted({r["gene"] for r in gene_results})
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(gene_list), 1)))
    gene_colors = {g: colors[i] for i, g in enumerate(gene_list)}

    totals = [r["total"] for r in gene_results]
    min_t, max_t = min(totals), max(totals)

    def _ms(total):
        if max_t == min_t:
            return 300
        return 150 + 600 * (total - min_t) / (max_t - min_t)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot([plot_min, plot_max], [plot_min, plot_max], "k--", alpha=0.35, lw=1.5, zorder=1)

    texts = []
    for r in finite:
        ax.scatter(
            r["m2"], r["m1"],
            s=_ms(r["total"]),
            c=[gene_colors[r["gene"]]],
            alpha=0.75,
            edgecolors="white",
            linewidth=2,
            zorder=3,
        )
        texts.append(
            ax.text(r["m2"], r["m1"], r["gene"],
                    fontsize=9, ha="center", va="center",
                    fontweight="bold", zorder=4)
        )

    for i, r in enumerate(undefined):
        x_pos = plot_min + i * 0.02
        ax.scatter(x_pos, undefined_pos, s=_ms(r["total"]),
                   facecolors="none", edgecolors=gene_colors[r["gene"]],
                   linewidth=3, alpha=0.9, zorder=3)
        texts.append(ax.text(x_pos, undefined_pos, r["gene"], fontsize=8,
                             ha="center", va="center", fontweight="bold"))

    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=1, alpha=0.6),
                    expand_points=(1.5, 1.5))
    except ImportError:
        pass

    metric_label = "Accuracy"
    ax.set_xlabel(f"{_pretty(method2)} {metric_label}", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"{_pretty(method1)} {metric_label}", fontsize=14, fontweight="bold")
    ax.set_xlim(plot_min * 0.95, plot_max)
    ax.set_ylim(plot_min * 0.95, plot_max)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor("#FAFAFA")

    # Size legend
    size_vals = [min_t, int((min_t + max_t) / 2), max_t]
    legend_elements = [
        plt.scatter([], [], s=_ms(v), c="#888", alpha=0.6,
                    edgecolors="white", linewidth=2,
                    label=f"n={v:,}")
        for v in size_vals if v > 0
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, title="Control variants", loc="lower right",
                  framealpha=0.9, fontsize=10)

    fig.tight_layout()
    tag_suffix = f"_{tag}" if tag else ""
    _save(fig, figure_dir / f"per_gene_scatter_{method1}_vs_{method2}{tag_suffix}.png")


# ---------------------------------------------------------------------------
# LR values loading
# ---------------------------------------------------------------------------

def load_lr_values(output_dir: Path, dataset_name: str, method: Optional[str], comp: str) -> Optional[Dict]:
    """Load *_lr_values.json.gz for a given dataset/method/component.

    Returns dict with keys: score_range, log_lr_plus, prior, scoreset_flipped.
    Returns None if the file doesn't exist or can't be loaded.
    """
    if method:
        fname = f"{dataset_name}_{method}_{comp}_lr_values.json.gz"
    else:
        fname = f"{dataset_name}_{comp}_lr_values.json.gz"
    # Search recursively — same pattern as calibration JSON discovery
    candidates = sorted(output_dir.rglob(fname))
    if not candidates:
        return None
    try:
        with gzip.open(candidates[0], "rt", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "score_range": np.array(data["score_range"]),
            "log_lr_plus": np.array(data["log_lr_plus"]),
            "prior": float(data["prior"]),
            "scoreset_flipped": bool(data.get("scoreset_flipped", False)),
        }
    except Exception as e:
        print(f"  WARNING: could not load LR values from {candidates[0]}: {e}")
        return None


# ---------------------------------------------------------------------------
# Fig 4 – Log LR+ curves with thresholds, one subplot per dataset
# ---------------------------------------------------------------------------

# Method colours: tavtigian=blue, piecewise=orange, others cycle further
_METHOD_COLORS = {
    "tavtigian": "#1f77b4",
    "piecewise": "#ff7f0e",
    "continuous": "#2ca02c",
    "strict_additive": "#9467bd",
    "default": "#1f77b4",
}
_FALLBACK_COLORS = ["#e377c2", "#8c564b", "#bcbd22", "#17becf"]


def make_log_lr_figure(
    lr_by_method: Dict[str, Dict[str, Dict]],
    dataset_names: List[str],
    point_ranges_by_method: Dict[str, Dict[str, Dict]],
    figure_dir: Path,
    tag: str = "",
    ncols: int = 5,
):
    """Grid of log LR+ plots, one subplot per dataset, all methods overlaid.

    Parameters
    ----------
    lr_by_method : {method: {dataset_name: {score_range, log_lr_plus, prior, scoreset_flipped}}}
    dataset_names : ordered list of dataset names to plot
    point_ranges_by_method : {method: {dataset_name: point_ranges_dict}}
        Used to derive threshold lines.  Point ranges come from the calibration
        JSON; pass {} to skip thresholds.
    figure_dir : output directory
    tag : appended to the filename
    ncols : subplot columns
    """
    methods = [m for m in lr_by_method if any(d in lr_by_method[m] for d in dataset_names)]
    if not methods:
        print("  SKIP log_lr figure: no LR data loaded")
        return

    # Assign colours
    fallback_iter = iter(_FALLBACK_COLORS)
    method_color = {m: _METHOD_COLORS.get(m) or next(fallback_iter, "#333333") for m in methods}

    nrows = (len(dataset_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})

    legend_handles: Dict[str, object] = {}

    for idx, ds in enumerate(dataset_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        plotted = False
        for m in methods:
            lr = lr_by_method[m].get(ds)
            if lr is None:
                continue

            score_range = lr["score_range"]
            log_lr_plus = lr["log_lr_plus"]
            prior = lr["prior"]
            flipped = lr["scoreset_flipped"]
            color = method_color[m]

            llr_curves = np.nanpercentile(log_lr_plus, [5, 50, 95], axis=0)

            # Median solid, percentile band shaded
            line, = ax.plot(score_range, llr_curves[1], color=color,
                            linewidth=1.6, label=_pretty(m))
            ax.fill_between(score_range, llr_curves[0], llr_curves[2],
                            color=color, alpha=0.15)
            legend_handles[m] = line
            plotted = True

            # Threshold lines from this method's point_ranges (use first method's)
            pr_dict = (point_ranges_by_method.get(m) or {}).get(ds)
            if pr_dict and m == methods[0]:
                point_values = sorted({abs(int(k)) for k in pr_dict})
                if point_values and prior > 0:
                    try:
                        tauP, tauB, _ = list(map(np.log,
                            thresholds_from_prior(prior, point_values + [10])))
                        for tp in tauP[:-1]:
                            ax.axhline(tp, color="red", linestyle="--",
                                       linewidth=0.7, alpha=0.55)
                        for tb in tauB[:-1]:
                            ax.axhline(tb, color="steelblue", linestyle="--",
                                       linewidth=0.7, alpha=0.55)
                    except Exception:
                        pass

        if not plotted:
            ax.set_visible(False)
            continue

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(ds.replace("_", " "), fontsize=7, pad=3)
        ax.set_xlabel("Score", fontsize=7)
        ax.set_ylabel("log LR+", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    # Hide unused subplots
    for idx in range(len(dataset_names), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Single shared legend at top
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            [_pretty(m) for m in legend_handles],
            loc="upper center",
            ncol=len(legend_handles),
            fontsize=9,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 1.01),
        )

    tag_suffix = f"_{tag}" if tag else ""
    method_str = "_".join(methods)
    _save(fig, figure_dir / f"log_lr_plus_{method_str}{tag_suffix}.png")


# ---------------------------------------------------------------------------
# Fig 5 – Per-dataset calibration details (replaces make_log_lr_figure)
# Mirrors the layout of plot_scoreset_best_config: histograms / point ranges /
# LR+ curves in three rows, one column per sample present in the variants CSV.
# Works from saved pipeline outputs only — no GMM fits required.
# ---------------------------------------------------------------------------

_SAMPLE_ORDER = [
    "Pathogenic/Likely Pathogenic",
    "Benign/Likely Benign",
    "gnomAD",
    "Synonymous",
]
_SAMPLE_COLORS = {
    "Pathogenic/Likely Pathogenic": "#CA7682",
    "Benign/Likely Benign":         "#1D7AAB",
    "gnomAD":                       "#A0A0A0",
    "Synonymous":                   "#6BAA75",
}
_SAMPLE_ALPHAS = {
    "Pathogenic/Likely Pathogenic": 0.6,
    "Benign/Likely Benign":         0.6,
    "gnomAD":                       0.3,
    "Synonymous":                   0.5,
}
_SAMPLE_LABELS = {
    "Pathogenic/Likely Pathogenic": "P/LP",
    "Benign/Likely Benign":         "B/LB",
    "gnomAD":                       "gnomAD",
    "Synonymous":                   "Synonymous",
}
_PV_LINESTYLES = {1: "dotted", 2: "dashed", 4: "dashdot", 8: (5, (10, 3))}


def make_calibration_figure(
    df_variants: pd.DataFrame,
    calibrations_by_method: Dict[str, Optional[Dict]],
    lr_by_method: Dict[str, Optional[Dict]],
    dataset: str,
    figure_dir: Path,
    tag: str = "",
):
    """Per-dataset calibration details figure: histograms / point ranges / LR+ curves.

    Mirrors the 3-row × n_samples-column layout of plot_scoreset_best_config:
      Row 0 : score histogram per sample (no GMM overlay — works from variants CSV)
              with vertical threshold lines from the first method's calibration
      Row 1 : point-assignment score ranges from calibration JSON, one line per method
      Row 2 : log-LR+ percentile curves (5 / 50 / 95 pct) with ACMG threshold lines

    Saves one PNG per dataset: ``calibration_{dataset}[_{tag}].png``.
    """
    samples_present = [
        s for s in _SAMPLE_ORDER if (df_variants["sample"] == s).any()
    ]
    n_samples = len(samples_present)
    if n_samples == 0:
        print(f"  SKIP calibration figure for {dataset}: no sample data")
        return

    methods = [m for m in calibrations_by_method if calibrations_by_method.get(m) is not None]
    if not methods:
        return

    first_cal = calibrations_by_method[methods[0]]
    flipped = bool(first_cal.get("scoreset_flipped", False))
    pr_first = {int(k): v for k, v in first_cal.get("point_ranges", {}).items()}

    fig, ax = plt.subplots(
        3, n_samples,
        figsize=(5 * n_samples, 14),
        squeeze=False,
        gridspec_kw={"hspace": 0.38, "wspace": 0.30},
    )
    fig.suptitle(dataset.replace("_", " "), fontsize=13, fontweight="bold", y=1.01)

    # ------------------------------------------------------------------
    # Row 0 : Score histograms (one column per sample; shared across methods)
    # ------------------------------------------------------------------
    for col, sname in enumerate(samples_present):
        a = ax[0, col]
        df_s = df_variants[df_variants["sample"] == sname]
        color = _SAMPLE_COLORS.get(sname, "#888888")
        alpha = _SAMPLE_ALPHAS.get(sname, 0.5)
        if not df_s.empty:
            if _HAS_SNS:
                sns.histplot(df_s["score"], stat="density", ax=a, alpha=alpha, color=color)
            else:
                a.hist(df_s["score"], density=True, alpha=alpha, color=color, bins=30)
            # Threshold lines from first method's calibration
            for pv, ranges in pr_first.items():
                ls = _PV_LINESTYLES.get(abs(pv), "solid")
                col_th = "red" if pv > 0 else "steelblue"
                for lo, hi in ranges:
                    # Boundary of the point region facing the uninformative zone
                    thresh = lo if (pv > 0) != flipped else hi
                    a.axvline(thresh, color=col_th, linestyle=ls, linewidth=0.9, alpha=0.6)
        label = _SAMPLE_LABELS.get(sname, sname)
        a.set_title(f"{label}\n(n={len(df_s):,})", fontsize=9)
        a.set_xlabel("Score", fontsize=8)
        if col == 0:
            a.set_ylabel("Density", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(True, alpha=0.2, linewidth=0.5)

    # ------------------------------------------------------------------
    # Row 1 : Point-assignment ranges (same content per column)
    # ------------------------------------------------------------------
    all_pv: set = set()
    for m in methods:
        cal = calibrations_by_method.get(m) or {}
        all_pv.update(int(k) for k in cal.get("point_ranges", {}))
    sorted_pv = sorted(all_pv)
    pv_to_row = {pv: i for i, pv in enumerate(sorted_pv)}

    for col in range(n_samples):
        a = ax[1, col]
        for midx, m in enumerate(methods):
            cal = calibrations_by_method.get(m) or {}
            pr = {int(k): v for k, v in cal.get("point_ranges", {}).items()}
            mc = _METHOD_COLORS.get(m, "#333333")
            labelled = False
            for pv in sorted(pr):
                row_y = pv_to_row.get(pv, 0) + midx * 0.12
                for lo, hi in pr[pv]:
                    a.plot(
                        [lo, hi], [row_y, row_y],
                        color=mc, linewidth=2.5, alpha=0.8,
                        label=_pretty(m) if not labelled else "",
                    )
                    labelled = True
        a.set_yticks(range(len(sorted_pv)))
        a.set_yticklabels(
            [f"{pv:+d}" for pv in sorted_pv] if col == 0 else [""] * len(sorted_pv),
            fontsize=7,
        )
        if col == 0:
            a.set_ylabel("Points", fontsize=8)
            a.set_title("Point Assignment Ranges", fontsize=9)
            if len(methods) > 1:
                a.legend(fontsize=7, loc="best")
        a.set_xlabel("Score", fontsize=8)
        a.grid(True, alpha=0.2, linewidth=0.5)

    # ------------------------------------------------------------------
    # Row 2 : Log LR+ percentile curves with ACMG threshold lines
    # ------------------------------------------------------------------
    for col in range(n_samples):
        a = ax[2, col]
        for midx, m in enumerate(methods):
            lr = lr_by_method.get(m)
            if lr is None:
                continue
            sr = np.asarray(lr["score_range"])
            llr = np.asarray(lr["log_lr_plus"])
            prior = float(lr["prior"])
            mc = _METHOD_COLORS.get(m, "#333333")
            pct = np.nanpercentile(llr, [5, 50, 95], axis=0)
            a.plot(sr, pct[1], color=mc, linewidth=1.5, label=_pretty(m))
            a.fill_between(sr, pct[0], pct[2], color=mc, alpha=0.15)
            # Threshold lines from first method only (avoid clutter)
            if midx == 0:
                cal = calibrations_by_method.get(m) or {}
                pvs = sorted({abs(int(k)) for k in cal.get("point_ranges", {})})
                if pvs and prior > 0:
                    try:
                        tauP, tauB, _ = list(map(np.log, thresholds_from_prior(prior, pvs + [10])))
                        for tp in tauP[:-1]:
                            a.axhline(tp, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
                        for tb in tauB[:-1]:
                            a.axhline(tb, color="steelblue", linestyle="--", linewidth=0.7, alpha=0.5)
                    except Exception:
                        pass
        a.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        a.set_xlabel("Score", fontsize=8)
        if col == 0:
            a.set_ylabel("log LR+", fontsize=8)
            a.set_title("Log LR+  (5th / median / 95th pct)", fontsize=9)
            if len(methods) > 1:
                a.legend(fontsize=7)
        a.tick_params(labelsize=7)
        a.grid(True, alpha=0.2, linewidth=0.5)

    tag_suffix = f"_{tag}" if tag else ""
    _save(fig, figure_dir / f"calibration_{dataset}{tag_suffix}.png")
