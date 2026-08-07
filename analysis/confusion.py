"""
Confusion-matrix construction and plotting for pipeline-native variants.

Building matrices (build_confusion_matrix / build_author_confusion_matrix) is
pipeline-native logic. Plotting is a thin pass-through to
src.assay_calibration.plot_utils.utils, whose figures this module must match
exactly (colors/layout/annotations unchanged) — only the calling convention
is new.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from src.assay_calibration.plot_utils.utils import (
    plot_aggregate_confusion_matrices,
    compute_classification_metrics,
)
from analysis.plot_common import effective_points as _effective_points, sample_matches

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

def build_confusion_matrix(df_sub: pd.DataFrame, use_oob: bool = True, label: str = "") -> Optional[pd.DataFrame]:
    """Build a 2×3 DataFrame from P/LP and B/LB variants.

    Rows: [BLB, PLP]   Cols: [Normal, IR, Abnormal]

    `label` (e.g. "{dataset}/{method}") is included in the printed in-bag
    fallback log line, if any, for traceability.
    """
    df_plp = df_sub[sample_matches(df_sub, "Pathogenic/Likely Pathogenic")]
    df_blb = df_sub[sample_matches(df_sub, "Benign/Likely Benign")]

    plp = _effective_points(df_plp, use_oob, label=label, context="PLP")
    blb = _effective_points(df_blb, use_oob, label=label, context="BLB")

    if len(plp) == 0 and len(blb) == 0:
        return None

    def _counts(pts):
        if len(pts) == 0:
            return [0, 0, 0]
        return [int((pts < 0).sum()), int((pts == 0).sum()), int((pts > 0).sum())]

    mat = pd.DataFrame(
        [_counts(blb), _counts(plp)],
        index=["BLB", "PLP"],
        columns=["Normal", "IR", "Abnormal"],
    )
    return mat


def build_author_confusion_matrix(df_sub: pd.DataFrame, use_oob: bool = True) -> Optional[pd.DataFrame]:
    """Build the author-annotation matrix for the SAME ClinVar-labeled variant
    population as build_confusion_matrix — matching
    test/plot_author_calibration_confusion.py::calculate_confusion_mat_oob
    exactly: rows are ClinVar ground truth (BLB/PLP), columns are what the
    *author* called those same variants (Normal/IR/Abnormal). This is what
    makes the ExCALIBR-vs-author comparison apples-to-apples — both matrices
    describe the identical set of P/LP + B/LB variants, just tallied by a
    different column (evidence-direction from points vs. the author's own
    call), rather than by two independently-filtered variant sets.

    Rows: [BLB, PLP]   Cols: [Normal, IR, Abnormal]  (author's classification)

    `use_oob` is accepted for signature symmetry with build_confusion_matrix
    but is a no-op here — author labels don't depend on OOB vs in-bag scoring.
    """
    if "auth_label" not in df_sub.columns:
        return None

    df_plp = df_sub[sample_matches(df_sub, "Pathogenic/Likely Pathogenic")]
    df_blb = df_sub[sample_matches(df_sub, "Benign/Likely Benign")]

    if len(df_plp) == 0 and len(df_blb) == 0:
        return None

    indeterminate_codes = {"NOT SPECIFIED", "INDETERMINATE", "IGNORE"}

    def _counts(sub):
        upper = sub["auth_label"].str.upper()
        norm = int((upper == "NORMAL").sum())
        abnorm = int((upper == "ABNORMAL").sum())
        ir = int((upper.isin(indeterminate_codes) | sub["auth_label"].isna()).sum())
        return [norm, ir, abnorm]

    mat = pd.DataFrame(
        [_counts(df_blb), _counts(df_plp)],
        index=["BLB", "PLP"],
        columns=["Normal", "IR", "Abnormal"],
    )
    return mat


# ---------------------------------------------------------------------------
# Plotting — thin wrappers around src/assay_calibration/plot_utils/utils.py
# ---------------------------------------------------------------------------

from analysis.plot_common import save_and_show, pretty_method as _pretty
_save = save_and_show


def make_confusion_figure(
    danzs_m1: List,
    danzs_m2: List,
    dataset_names: List[str],
    label1: str,
    label2: str,
    figure_dir: Path,
    tag: str = "",
    xlabel: Optional[str] = None,
    xticklabels: Optional[List[str]] = None,
    title1: Optional[str] = None,
    title2: Optional[str] = None,
):
    """Aggregate confusion heatmap comparing two sets of per-dataset matrices.

    Calls plot_aggregate_confusion_matrices from plot_utils.utils unmodified
    (letters=True); only re-labels the two panel titles.

    *xlabel*/*xticklabels*, if given, override
    plot_aggregate_confusion_matrices' hardcoded "Evidence Direction" axis
    label and its Normal/IR/Abnormal -> Benign/Indeterminate/Pathogenic
    tick-label mapping on both panels -- additive only (default None
    preserves the exact existing look), for callers whose column semantics
    aren't the Normal/IR/Abnormal evidence-direction convention (e.g. a
    points-sign family, where the columns are Negative/Indeterminate/
    Positive rather than a ClinVar-style label).

    *title1*/*title2*, if given, override the panel title text shown on the
    image itself, independent of *label1*/*label2* (which still control the
    output filename) -- for callers that want a different display name (e.g.
    "Posterior-exact") than the identifier used elsewhere for file naming.
    """
    fig, _, _ = plot_aggregate_confusion_matrices(
        danzs_m1, danzs_m2, dataset_names, letters=True
    )
    axes = [ax for ax in fig.get_axes() if hasattr(ax, "get_title")]
    if len(axes) >= 2:
        axes[0].set_title(title1 if title1 is not None else _pretty(label1),
                           fontsize=18, fontweight="bold", pad=10)
        axes[1].set_title(title2 if title2 is not None else _pretty(label2),
                           fontsize=18, fontweight="bold", pad=10)
    if xlabel is not None or xticklabels is not None:
        for ax in axes[:2]:
            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=14)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, rotation=0, fontsize=12)

    tag_suffix = f"_{tag}" if tag else ""
    _save(fig, figure_dir / f"confusion_heatmap_{label1}_vs_{label2}{tag_suffix}.png")


# Diverging Blue(Benign)/Gray(Indeterminate)/Red(Pathogenic) row-normalized
# color scheme, column-keyed -- extracted from
# plot_aggregate_confusion_matrices's letters=True branch (the two-panel
# ExCALIBR-vs-author comparison figure's style) so single-panel figures
# match it too, instead of the flat purple gradient used previously.
_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "blue_gradient", ['#F0F8FC', '#99C8DC', '#7AB5D1', '#4B91A6', '#2E6B7E'])
_RED_CMAP = LinearSegmentedColormap.from_list(
    "red_gradient", ['#FCF0F2', '#E6B1B8', '#D68F99', '#B85C6B', '#943744'])
_GRAY_CMAP = LinearSegmentedColormap.from_list(
    "gray_gradient", ['#F5F5F5', '#CCCCCC', '#999999', '#666666'])


def _column_cmap(col_name: str) -> LinearSegmentedColormap:
    name = str(col_name)
    if any(tag in name for tag in ("Benign", "Normal", "BLB", "B/LB")):
        return _BLUE_CMAP
    if any(tag in name for tag in ("Pathogenic", "Abnormal", "PLP", "P/LP")):
        return _RED_CMAP
    return _GRAY_CMAP  # IR/Indeterminate, and any other column


def draw_diverging_confusion_heatmap(ax, aggregate: pd.DataFrame, fontsize: int = 14):
    """Draw one row-normalized diverging confusion heatmap into `ax`.

    Same color logic as plot_aggregate_confusion_matrices's letters=True
    heatmap (src/assay_calibration/plot_utils/utils.py), generalized to a
    single matrix/axis so it can be reused both for single-panel figures
    (make_single_confusion_figure) and multi-panel grids
    (analysis/path_percentile_confusion.py) without duplicating the color
    logic. Caller is responsible for ticks/labels/title/caption -- this only
    draws the cells.
    """
    n_rows, n_cols = len(aggregate), len(aggregate.columns)
    for i in range(n_rows):
        row_max = aggregate.iloc[i].max()
        for j, col_name in enumerate(aggregate.columns):
            value = aggregate.iloc[i, j]
            normalized = value / row_max if row_max > 0 else 0
            facecolor = _column_cmap(col_name)(normalized)
            ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1, facecolor=facecolor, edgecolor="white", linewidth=2.5,
            ))
            text_color = "white" if (row_max > 0 and value / row_max > 0.45) else "black"
            if "IR" in str(col_name) or "Indeterminate" in str(col_name):
                text_color = "black"
            ax.text(j + 0.5, i + 0.5, f"{value:,}", ha="center", va="center",
                     fontsize=fontsize, color=text_color)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_facecolor("#F9F9F9")


def make_single_confusion_figure(
    matrices: List,
    dataset_names: List[str],
    label: str,
    figure_dir: Path,
    tag: str = "",
):
    """Single-panel aggregate confusion heatmap (ClinVar rows x evidence-direction columns).

    Uses the diverging Blue/Gray/Red row-normalized style (see
    draw_diverging_confusion_heatmap) -- the same idiom
    plot_aggregate_confusion_matrices uses for its two-panel
    ExCALIBR-vs-author comparison -- for the case where there's no second
    matrix (author labels / other method) to compare against.
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

    fig, ax = plt.subplots(figsize=(5, 4.2))
    draw_diverging_confusion_heatmap(ax, aggregate, fontsize=13)

    ax.set_xticks(np.arange(len(xlabels)) + 0.5)
    ax.set_yticks(np.arange(len(ylabels)) + 0.5)
    ax.set_xticklabels(xlabels, rotation=0, fontsize=11)
    ax.set_yticklabels(ylabels, rotation=0, fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
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
