"""
SpliceAI-threshold / VEP-splice-consequence-filter ablation: how much does
ExCALIBR's calibration performance change as the splice-variant exclusion
rules used by `Scoreset.splicing_filter`
(src/assay_calibration/data_utils/dataset.py) are relaxed or removed, for
assays that don't themselves detect splice effects?

Each condition here is a FULL, independent `hpc/prepare.py pillar_project`
rerun (own output dir, same shape as analysis.config.OUTPUT_DIR/
SKEW_LOCKED_OUTPUT_DIR), produced by analysis/build_splice_ablation_jobs.py:
  {SPLICE_ABLATION_ROOT}/thresh_0.1/  ...  thresh_0.9/   (VEP filter ON,
      SpliceAI threshold = that value)
  {SPLICE_ABLATION_ROOT}/keep_all/                       (VEP filter OFF,
      SpliceAI thresholding disabled -- no splice-variant rows dropped)

Unlike analysis/robustness.py's downsample/discordance conditions (which are
suffixed sub-populations of ONE dataset dir, requiring point_ranges to be
re-applied to a fixed reference population), each condition here is a
complete, independently-fit ExCALIBR output tree -- discovered/loaded with
analysis.discovery exactly like OUTPUT_DIR itself, same as
analyze_pipeline_output.py section 3a3's skew-locked comparison. No
reference-population indirection is needed: each condition's own
*_variants.csv/*_calibration.json already reflect that condition's own
splice-filtered population.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import config as cfg
from analysis.discovery import discover_outputs, load_all_variants
from analysis.confusion import build_confusion_matrix
from analysis.plot_common import save_and_show

# Matches "thresh_0.1" .. "thresh_0.9" (build_splice_ablation_jobs.py's own
# naming) -- anything else (e.g. "keep_all", a stray "logs/" dir) is handled
# separately by discover_splice_ablation_conditions.
_THRESH_RE = re.compile(r"^thresh_(?P<value>\d+\.\d+)$")
_KEEP_ALL_LABEL = "keep_all"


def discover_splice_ablation_conditions(
    root: Optional[str] = None,
) -> List[Tuple[str, Optional[float], Path]]:
    """[(condition_label, spliceai_threshold_or_None, condition_dir), ...]
    for every condition subdirectory found under `root`, sorted by threshold
    ascending with "keep_all" last. [] if root doesn't exist."""
    root_path = Path(root or cfg.SPLICE_ABLATION_ROOT)
    if not root_path.is_dir():
        return []

    conditions: List[Tuple[str, Optional[float], Path]] = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        if child.name == _KEEP_ALL_LABEL:
            conditions.append((child.name, None, child))
            continue
        m = _THRESH_RE.match(child.name)
        if m is not None:
            conditions.append((child.name, float(m.group("value")), child))
    conditions.sort(key=lambda c: (c[1] is None, c[1]))
    return conditions


def load_splice_ablation_variants(
    condition_dir: Path,
    dataset_configs: Optional[Dict] = None,
    datasets_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """One condition's variants, loaded exactly like a normal pipeline
    output tree (analysis.discovery.discover_outputs + load_all_variants) --
    same pattern analyze_pipeline_output.py section 3a3 uses for
    SKEW_LOCKED_OUTPUT_DIR. Empty DataFrame if nothing discovered."""
    tree, model_selections, calibrations = discover_outputs(condition_dir)
    if not tree:
        return pd.DataFrame()
    return load_all_variants(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=datasets_filter, calibrations=calibrations,
        min_controls=0,
    )


def compute_splice_ablation_confusion_matrices(
    condition_dir: Path,
    dataset_configs: Optional[Dict] = None,
    datasets_filter: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """{dataset: confusion_matrix} for one condition, use_oob=False (these
    reruns don't carry oob_* columns, same convention as the skew-locked/
    GMM-baseline comparisons)."""
    df = load_splice_ablation_variants(condition_dir, dataset_configs, datasets_filter)
    if df.empty:
        return {}
    matrices = {}
    for dataset in sorted(df["dataset"].unique()):
        df_ds = df[df["dataset"] == dataset]
        mat = build_confusion_matrix(df_ds, use_oob=False, label=f"{dataset}/{condition_dir.name}")
        if mat is not None:
            matrices[dataset] = mat
    return matrices


def run_splice_ablation_analysis(
    root: Optional[str] = None,
    dataset_configs: Optional[Dict] = None,
    datasets_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convenience wrapper: discover every condition, build per-dataset
    confusion matrices, compute classification metrics. Returns one row per
    (condition_label, dataset) with columns: condition_label,
    spliceai_threshold, dataset, then every key from
    compute_classification_metrics (accuracy, coverage, dor_standard,
    sensitivity, specificity, mcc, lr_plus_*, ...). Empty DataFrame if no
    conditions are found on disk.
    """
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics

    conditions = discover_splice_ablation_conditions(root)
    if not conditions:
        print(f"  SKIP splice ablation analysis: no conditions found under "
              f"{root or cfg.SPLICE_ABLATION_ROOT}")
        return pd.DataFrame()

    rows = []
    for condition_label, spliceai_threshold, condition_dir in conditions:
        matrices = compute_splice_ablation_confusion_matrices(condition_dir, dataset_configs, datasets_filter)
        if not matrices:
            print(f"  SKIP {condition_label}: no confusion matrices (no variants discovered under {condition_dir})")
            continue
        for dataset, mat in matrices.items():
            metrics = compute_classification_metrics(mat)
            rows.append({
                "condition_label": condition_label, "spliceai_threshold": spliceai_threshold,
                "dataset": dataset, **metrics,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_METRIC_YLABELS = {
    "accuracy": "Accuracy", "coverage": "Coverage", "dor_standard": "DOR",
    "sensitivity": "Sensitivity", "specificity": "Specificity",
    "lr_plus_standard": "LR+", "lr_plus_pathogenic": "LR+ (pathogenic)",
    "lr_plus_benign": "LR+ (benign)", "mcc": "MCC",
}


def plot_splice_ablation_curve(
    summary_df: pd.DataFrame,
    metrics: List[str] = ("accuracy", "coverage", "dor_standard"),
    figure_dir: Optional[Path] = None,
    label: str = "all_datasets",
):
    """Median line + IQR ribbon (across datasets) vs. spliceai_threshold
    (linear x-axis, 0.1-0.9 -- unlike robustness.py's log2 downsample-N
    axis, threshold has no natural log scale), individual dataset values as
    scatter points, dashed reference line + separate marker for the
    "keep_all" condition (spliceai_threshold is None there, so it can't sit
    on the same linear axis -- drawn as a horizontal line instead, same
    role as robustness.py's own reference line). One subplot per metric.
    """
    sub = summary_df[summary_df["spliceai_threshold"].notna()].copy()
    keep_all = summary_df[summary_df["spliceai_threshold"].isna()]
    if sub.empty:
        print(f"  SKIP splice ablation curve for {label}: no thresholded conditions")
        return None

    thresholds = sorted(sub["spliceai_threshold"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        medians, p25s, p75s, xs_scatter, ys_scatter = [], [], [], [], []
        for t in thresholds:
            vals = sub.loc[sub["spliceai_threshold"] == t, metric].values
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                medians.append(np.nan); p25s.append(np.nan); p75s.append(np.nan)
                continue
            p25, p50, p75 = np.percentile(vals, [25, 50, 75])
            medians.append(p50); p25s.append(p25); p75s.append(p75)
            xs_scatter.extend([t] * len(vals))
            ys_scatter.extend(vals.tolist())

        ax.fill_between(thresholds, p25s, p75s, alpha=0.25, color="C0")
        ax.plot(thresholds, medians, marker="o", color="C0", label="median (thresholded)")
        ax.scatter(xs_scatter, ys_scatter, alpha=0.4, s=15, color="C0")

        if not keep_all.empty:
            keep_all_vals = keep_all[metric].values
            keep_all_vals = keep_all_vals[np.isfinite(keep_all_vals)]
            if len(keep_all_vals):
                ax.axhline(np.median(keep_all_vals), linestyle="--", color="black", alpha=0.6,
                            label="keep_all (median)")
                ax.scatter([max(thresholds)] * len(keep_all_vals), keep_all_vals,
                           alpha=0.3, s=15, color="black", marker="x")

        ax.set_xlabel("SpliceAI threshold")
        ax.set_ylabel(_METRIC_YLABELS.get(metric, metric))
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(linewidth=0.5, alpha=0.3)

    fig.suptitle(f"{label}: SpliceAI threshold / VEP splice-filter ablation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if figure_dir is not None:
        save_and_show(fig, Path(figure_dir) / f"splice_ablation_{label}.png")
    return fig
