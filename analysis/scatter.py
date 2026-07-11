"""
Per-gene accuracy scatter (method1 vs method2), pipeline-native.

Deliberately mirrors the visual idiom of
src.assay_calibration.plot_utils.utils.plot_gene_level_performance_comparison
(same marker/size/color/adjustText scheme) but is kept as its own
implementation rather than extending that function, since the src.py version
hardcodes DanZ-vs-author semantics used by other call sites (confusion.py /
legacy scripts) that must not change. Moved verbatim from
analysis/plot_utils.py — no visual changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.assay_calibration.plot_utils.utils import compute_classification_metrics


from analysis.plot_common import save_and_show, pretty_method as _pretty
_save = save_and_show


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
    metric='accuracy'. Point size scales with total variant count per gene.
    """
    if method1 not in conf_by_method or method2 not in conf_by_method:
        print(f"  SKIP scatter: missing matrices for {method1} or {method2}")
        return

    mats1 = conf_by_method[method1]
    mats2 = conf_by_method[method2]

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

    min_v = min(min(r["m1"] for r in finite), min(r["m2"] for r in finite))
    max_v = max(max(r["m1"] for r in finite), max(r["m2"] for r in finite))
    plot_min = max(0.5, min_v - 0.05)
    plot_max = min(1.0, max_v + 0.02)
    undefined_pos = plot_min * 0.97

    gene_list = sorted({r["gene"] for r in gene_results})
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(gene_list), 1)))
    gene_colors = {g: colors[i] for i, g in enumerate(gene_list)}

    totals = [r["total"] for r in gene_results]
    min_t, max_t = min(totals), max(totals)

    def _ms(total):
        if max_t == min_t:
            return 300
        return 150 + 600 * (total - min_t) / (max_t - min_t)

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
