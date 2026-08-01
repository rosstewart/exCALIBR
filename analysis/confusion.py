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
from matplotlib.colors import LinearSegmentedColormap

from src.assay_calibration.plot_utils.utils import (
    plot_aggregate_confusion_matrices,
    compute_classification_metrics,
)
from analysis.plot_common import effective_points as _effective_points, sample_matches

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

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
    """
    fig, _, _ = plot_aggregate_confusion_matrices(
        danzs_m1, danzs_m2, dataset_names, letters=True
    )
    axes = [ax for ax in fig.get_axes() if hasattr(ax, "get_title")]
    if len(axes) >= 2:
        axes[0].set_title(_pretty(label1), fontsize=18, fontweight="bold", pad=10)
        axes[1].set_title(_pretty(label2), fontsize=18, fontweight="bold", pad=10)
    if xlabel is not None or xticklabels is not None:
        for ax in axes[:2]:
            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=14)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, rotation=0, fontsize=12)

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

    Reproduces the same visual idiom as plot_aggregate_confusion_matrices's
    letters=False single-panel branch (purple gradient + DOR/coverage caption),
    for the case where there's no second matrix (author labels / other method)
    to compare against.
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
