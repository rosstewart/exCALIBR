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

    Returns None (rather than an all-IR matrix) for a dataset with zero
    determinate (Normal/Abnormal) author calls -- that pattern means the
    author functional classification was never recorded for this dataset
    (auth_label all NaN), not that the author genuinely called every P/LP
    and B/LB variant indeterminate. Counting those as real IR would inflate
    the aggregate's denominator with "indeterminate" rows that are actually
    just missing data, dragging down the author panel's coverage/DOR against
    datasets ExCALIBR itself classifies fine -- matches
    test/plot_author_calibration_confusion.py's own comment ("do NOT save
    author one, it has artificial indeterminates") about this same failure
    mode, there worked around by hand-curating a `reported_list` of datasets
    with real author data instead of filtering programmatically.
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
    if (mat["Normal"].sum() + mat["Abnormal"].sum()) == 0:
        return None
    return mat


def build_both_determinate_confusion_matrices(
    df_sub: pd.DataFrame, use_oob: bool = True, label: str = "",
) -> tuple:
    """(excalibr_matrix, author_matrix) -- the SAME two [BLB,PLP] x
    [Normal,IR,Abnormal] matrices build_confusion_matrix/
    build_author_confusion_matrix already produce for the unrestricted 3a
    panel, just scoped down to the subset of P/LP+B/LB variants where BOTH
    ExCALIBR (points != 0) AND the author (auth_label NORMAL/ABNORMAL, not
    an indeterminate code or missing) made a determinate call.

    Restricting to that shared subset before building each matrix (rather
    than building them independently and comparing afterward) is what makes
    this an apples-to-apples "both methods actually took a position here"
    view: every row/column in both returned matrices describes the exact
    same set of variants, so the IR column is necessarily all-zero on both
    sides.
    """
    if "auth_label" not in df_sub.columns:
        return None, None

    pts = _effective_points(df_sub, use_oob, label=label, context="both_determinate")
    upper = df_sub["auth_label"].str.upper()
    both_det_mask = (pts != 0) & upper.isin({"NORMAL", "ABNORMAL"})
    df_restricted = df_sub[both_det_mask]

    excalibr_mat = build_confusion_matrix(df_restricted, use_oob=use_oob, label=label)
    author_mat = build_author_confusion_matrix(df_restricted, use_oob=use_oob)
    return excalibr_mat, author_mat


def build_vus_coverage(df_sub: pd.DataFrame, use_oob: bool = True, label: str = "") -> Optional[tuple]:
    """(n_determinate, n_vus) among `df_sub`'s ClinVar-VUS variants, by the
    sign of ExCALIBR's own effective evidence points (nonzero = determinate,
    same convention as section 9's per-scoreset VUS breakdown in
    analyze_pipeline_output.py). None if `is_vus` isn't present or there are
    no VUS rows -- callers aggregate a list of these across datasets before
    turning the pair into a percentage, so a per-dataset None must be
    droppable rather than treated as (0, 0)."""
    if "is_vus" not in df_sub.columns:
        return None
    df_vus = df_sub[df_sub["is_vus"].fillna(False).astype(bool)]
    if df_vus.empty:
        return None
    pts = _effective_points(df_vus, use_oob, label=label, context="VUS")
    return int((pts != 0).sum()), len(pts)


def build_author_vus_coverage(df_sub: pd.DataFrame) -> Optional[tuple]:
    """(n_determinate, n_vus) among `df_sub`'s ClinVar-VUS variants, by the
    author's own call -- determinate meaning `auth_label` is NORMAL/ABNORMAL
    rather than one of the indeterminate codes or missing. Mirrors
    build_author_confusion_matrix's indeterminate-code handling."""
    if "is_vus" not in df_sub.columns or "auth_label" not in df_sub.columns:
        return None
    df_vus = df_sub[df_sub["is_vus"].fillna(False).astype(bool)]
    if df_vus.empty:
        return None
    indeterminate_codes = {"NOT SPECIFIED", "INDETERMINATE", "IGNORE"}
    upper = df_vus["auth_label"].str.upper()
    n_determinate = int((~(upper.isin(indeterminate_codes) | df_vus["auth_label"].isna())).sum())
    return n_determinate, len(df_vus)


def _aggregate_coverage_pct(coverages: Optional[List]) -> Optional[float]:
    """coverages: list of (n_determinate, n_total)-or-None (one per dataset,
    same shape as the matrices lists build_confusion_matrix produces).
    Returns the pooled determinate percentage, or None if every entry was
    None/empty -- lets callers show "no VUS in scope" rather than a fake 0%."""
    if not coverages:
        return None
    pairs = [c for c in coverages if c is not None]
    if not pairs:
        return None
    n_det = sum(p[0] for p in pairs)
    n_tot = sum(p[1] for p in pairs)
    return 100 * n_det / n_tot if n_tot > 0 else None


def _dor_coverage_text(dor: float, controls_pct: float, vus_pct: Optional[float]) -> str:
    """'DOR: X.X\\nDeterminate: Controls Y.Y%[, VUS Z.Z%]' -- the VUS clause is
    only appended when a real pooled VUS coverage was supplied (see
    _aggregate_coverage_pct); omitted rather than faked when no VUS data was
    in scope for the panel."""
    line2 = f"Determinate: Controls {controls_pct:.1f}%"
    if vus_pct is not None:
        line2 += f", VUS {vus_pct:.1f}%"
    return f"DOR: {dor:.1f}\n{line2}"


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
    vus_coverages_m1: Optional[List] = None,
    vus_coverages_m2: Optional[List] = None,
    filename: Optional[str] = None,
):
    """Aggregate confusion heatmap comparing two sets of per-dataset matrices.

    Calls plot_aggregate_confusion_matrices from plot_utils.utils (letters=True)
    then re-labels the two panel titles and draws a
    "DOR: X.X / Determinate: Controls Y.Y%[, VUS Z.Z%]" caption under each
    panel -- plot_aggregate_confusion_matrices itself only builds this caption
    for its (unused by any current caller) letters=False styling, so it's
    added here instead of duplicating panel-lookup logic in that function.

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

    *vus_coverages_m1*/*vus_coverages_m2*, if given, are per-dataset
    (n_determinate, n_vus)-or-None lists (same shape/order as
    danzs_m1/danzs_m2, from build_vus_coverage / build_author_vus_coverage)
    pooled into the caption's VUS percentage; omitted from the caption
    (rather than shown as 0%) when not supplied.

    *filename*, if given, overrides the auto-built
    "confusion_heatmap_{label1}_vs_{label2}{tag}.png" name -- for callers
    that want a more descriptive name once `figure_dir` already encodes the
    comparison via a subfolder (e.g. analyze_pipeline_output.py's
    clinvar_comparisons/{author,acmgscaler,...}/ layout).
    """
    fig, danz_metrics, auth_metrics = plot_aggregate_confusion_matrices(
        danzs_m1, danzs_m2, dataset_names, letters=True
    )
    axes = [ax for ax in fig.get_axes() if hasattr(ax, "get_title")]
    if len(axes) >= 2:
        axes[0].set_title(title1 if title1 is not None else _pretty(label1),
                           fontsize=18, fontweight="bold", pad=10)
        axes[1].set_title(title2 if title2 is not None else _pretty(label2),
                           fontsize=18, fontweight="bold", pad=10)
        vus_pct_1 = _aggregate_coverage_pct(vus_coverages_m1)
        vus_pct_2 = _aggregate_coverage_pct(vus_coverages_m2)
        axes[0].text(
            0.5, -0.24, _dor_coverage_text(danz_metrics["dor_standard"], 100 * danz_metrics["coverage"], vus_pct_1),
            transform=axes[0].transAxes, fontsize=11, ha="center", va="top", color="#555555",
        )
        axes[1].text(
            0.5, -0.24, _dor_coverage_text(auth_metrics["dor_standard"], 100 * auth_metrics["coverage"], vus_pct_2),
            transform=axes[1].transAxes, fontsize=11, ha="center", va="top", color="#555555",
        )
    if xlabel is not None or xticklabels is not None:
        for ax in axes[:2]:
            if xlabel is not None:
                ax.set_xlabel(xlabel, fontsize=14)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, rotation=0, fontsize=12)

    tag_suffix = f"_{tag}" if tag else ""
    _save(fig, figure_dir / (filename or f"confusion_heatmap_{label1}_vs_{label2}{tag_suffix}.png"))


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
    vus_coverages: Optional[List] = None,
    title_suffix: Optional[str] = None,
    filename: Optional[str] = None,
):
    """Single-panel aggregate confusion heatmap (ClinVar rows x evidence-direction columns).

    Uses the diverging Blue/Gray/Red row-normalized style (see
    draw_diverging_confusion_heatmap) -- the same idiom
    plot_aggregate_confusion_matrices uses for its two-panel
    ExCALIBR-vs-author comparison -- for the case where there's no second
    matrix (author labels / other method) to compare against.

    *vus_coverages*, if given, is a per-dataset (n_determinate, n_vus)-or-None
    list (same shape/order as `matrices`, from build_vus_coverage) pooled
    into the caption's VUS percentage; omitted (rather than shown as 0%)
    when not supplied.

    *title_suffix*, if given, replaces the default "(N datasets)" title
    parenthetical -- for callers where `matrices` isn't one entry per
    dataset (e.g. a single pre-aggregated gene-deduplicated matrix, where
    "(1 datasets)" would be misleading).

    *filename*, if given, overrides the auto-built
    "confusion_heatmap_{label}{tag}.png" name.
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
    ax.set_title(f"{_pretty(label)} {title_suffix if title_suffix is not None else f'({n_datasets} datasets)'}",
                  fontsize=14, fontweight="bold", pad=10)
    vus_pct = _aggregate_coverage_pct(vus_coverages)
    ax.text(
        0.5, -0.24,
        _dor_coverage_text(metrics["dor_standard"], 100 * metrics["coverage"], vus_pct),
        transform=ax.transAxes, fontsize=10, ha="center", va="top", color="#555555",
    )

    tag_suffix = f"_{tag}" if tag else ""
    _save(fig, figure_dir / (filename or f"confusion_heatmap_{label}{tag_suffix}.png"))


def make_confusion_grid_figure(
    panels: List[tuple],
    figure_dir: Path,
    filename: str,
    suptitle: Optional[str] = None,
):
    """N-panel confusion-matrix grid, one panel per (panel_label, matrices)
    pair -- same diverging Blue/Gray/Red row-normalized style as
    make_single_confusion_figure/draw_diverging_confusion_heatmap, laid out
    side by side for direct visual comparison (e.g. ExCALIBR vs acmgscaler
    vs a manually-fixed-prior ExCALIBR rerun -- see analyze_pipeline_output.py
    section 3a2).

    *panels*: list of (panel_label, matrices) pairs, each `matrices` a
    per-dataset list-or-None (same shape build_confusion_matrix's callers
    already produce) -- aggregated internally exactly like
    make_single_confusion_figure. A pair with no valid matrix in scope is
    dropped rather than failing the whole grid.
    """
    valid = []
    for panel_label, matrices in panels:
        aggregate = None
        n_datasets = 0
        for mat in matrices:
            if mat is None:
                continue
            aggregate = mat.copy() if aggregate is None else aggregate + mat
            n_datasets += 1
        if aggregate is not None:
            valid.append((panel_label, aggregate, n_datasets))

    if not valid:
        print(f"  SKIP confusion grid {filename}: no panel had a valid matrix")
        return

    label_map = {"PLP": "P/LP", "BLB": "B/LB", "IR": "Indeterminate",
                 "Normal": "Benign", "Abnormal": "Pathogenic"}

    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 4.6))
    if len(valid) == 1:
        axes = [axes]

    for ax, (panel_label, aggregate, n_datasets) in zip(axes, valid):
        metrics = compute_classification_metrics(aggregate)
        xlabels = [label_map.get(str(c), str(c)) for c in aggregate.columns]
        ylabels = [label_map.get(str(r), str(r)) for r in aggregate.index]

        draw_diverging_confusion_heatmap(ax, aggregate, fontsize=12)
        ax.set_xticks(np.arange(len(xlabels)) + 0.5)
        ax.set_yticks(np.arange(len(ylabels)) + 0.5)
        ax.set_xticklabels(xlabels, rotation=0, fontsize=10)
        ax.set_yticklabels(ylabels, rotation=0, fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Evidence Direction", fontsize=11)
        ax.set_ylabel("ClinVar Classification", fontsize=11)
        ax.set_title(f"{_pretty(panel_label)} ({n_datasets} datasets)", fontsize=13, fontweight="bold", pad=10)
        ax.text(
            0.5, -0.26, _dor_coverage_text(metrics["dor_standard"], 100 * metrics["coverage"], None),
            transform=ax.transAxes, fontsize=9, ha="center", va="top", color="#555555",
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=1.04)
    fig.tight_layout()
    _save(fig, figure_dir / filename)
