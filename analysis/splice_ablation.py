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
from analysis.plot_common import save_and_show, sample_matches

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
    spliceai_threshold: Optional[float] = 0.2,
    vep_splice_filter: bool = True,
) -> pd.DataFrame:
    """One condition's variants, loaded exactly like a normal pipeline
    output tree (analysis.discovery.discover_outputs + load_all_variants) --
    same pattern analyze_pipeline_output.py section 3a3 uses for
    SKEW_LOCKED_OUTPUT_DIR. Empty DataFrame if nothing discovered.

    `spliceai_threshold`/`vep_splice_filter` MUST be this condition's own
    values (from discover_splice_ablation_conditions -- the second tuple
    element is exactly spliceai_threshold; vep_splice_filter is True for
    every thresholded condition and False only for "keep_all") -- otherwise
    load_all_variants rebuilds each dataset's Scoreset with its own
    hardcoded defaults (0.2/True) regardless of which condition's
    calibration.json is being applied to it, silently making every
    condition's "population" identical and defeating the whole point of
    this ablation. Callers here always pass the condition's real values;
    nothing in this module relies on the (0.2, True) defaults below.
    """
    tree, model_selections, calibrations = discover_outputs(condition_dir)
    if not tree:
        return pd.DataFrame()
    return load_all_variants(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=datasets_filter, calibrations=calibrations,
        min_controls=0, spliceai_threshold=spliceai_threshold, vep_splice_filter=vep_splice_filter,
    )


def compute_splice_ablation_confusion_matrices(
    condition_dir: Path,
    dataset_configs: Optional[Dict] = None,
    datasets_filter: Optional[List[str]] = None,
    spliceai_threshold: Optional[float] = 0.2,
    vep_splice_filter: bool = True,
) -> Dict[str, pd.DataFrame]:
    """{dataset: confusion_matrix} for one condition, use_oob=False (these
    reruns don't carry oob_* columns, same convention as the skew-locked/
    GMM-baseline comparisons). `spliceai_threshold`/`vep_splice_filter` MUST
    be this condition's own values -- see load_splice_ablation_variants."""
    df = load_splice_ablation_variants(
        condition_dir, dataset_configs, datasets_filter, spliceai_threshold, vep_splice_filter,
    )
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
        vep_splice_filter = spliceai_threshold is not None
        matrices = compute_splice_ablation_confusion_matrices(
            condition_dir, dataset_configs, datasets_filter, spliceai_threshold, vep_splice_filter,
        )
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
# Aggregate (all-datasets-pooled) summary tables
# ---------------------------------------------------------------------------

_SAMPLE_COUNT_CATEGORIES = {
    "n_plp": "Pathogenic/Likely Pathogenic",
    "n_blb": "Benign/Likely Benign",
    # The per-variant "sample" column's real category string for the
    # background/gnomAD-derived population is "population" (see
    # analysis.plot_common.sample_matches's own docstring example) --
    # "gnomAD" is only the --population-type config value that determines
    # which rows get labeled "population" in the first place, never a
    # literal category string in the data itself.
    "n_gnomad": "population",
    "n_synonymous": "Synonymous",
}

_PCT_CHANGE_METRICS = [
    "accuracy", "coverage", "dor_standard", "sensitivity", "specificity", "mcc",
    "lr_plus_standard", "lr_plus_pathogenic", "lr_plus_benign",
]


def compute_splice_ablation_aggregate_summary(
    root: Optional[str] = None,
    dataset_configs: Optional[Dict] = None,
    datasets_filter: Optional[List[str]] = None,
    baseline_condition: str = "thresh_0.2",
) -> pd.DataFrame:
    """Table A: one row per condition, ALL datasets pooled into a single
    combined confusion matrix (summed, not per-dataset) -> one aggregate
    classification-metrics row per condition, same pooling
    print_aggregate_performance already applies to the main pipeline's
    results elsewhere in analyze_pipeline_output.py -- just with "condition"
    as the row axis instead of one aggregate row.

    Also carries each condition's raw PLP/BLB/gnomAD/Synonymous variant
    counts (pooled across every dataset via analysis.plot_common.
    sample_matches), plus every metric/count's percent change relative to
    `baseline_condition` (default "thresh_0.2", matching
    Scoreset.splicing_filter's own hardcoded defaults --
    spliceai_threshold=0.2, vep_splice_filter=True).

    Since each condition genuinely uses its own, differently-filtered
    population (that's the point of the ablation -- see this module's own
    docstring), a metric's percent change here mixes two effects: the
    control population itself changed, AND the calibration was refit to
    that different population. compute_splice_ablation_fixed_population_
    summary isolates just the second effect, holding the population fixed.
    """
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics

    conditions = discover_splice_ablation_conditions(root)
    if not conditions:
        print(f"  SKIP splice ablation aggregate summary: no conditions found under "
              f"{root or cfg.SPLICE_ABLATION_ROOT}")
        return pd.DataFrame()

    rows = []
    for condition_label, spliceai_threshold, condition_dir in conditions:
        vep_splice_filter = spliceai_threshold is not None
        df = load_splice_ablation_variants(
            condition_dir, dataset_configs, datasets_filter, spliceai_threshold, vep_splice_filter,
        )
        if df.empty:
            print(f"  SKIP {condition_label}: no variants discovered under {condition_dir}")
            continue

        matrices = [
            build_confusion_matrix(df[df["dataset"] == ds], use_oob=False, label=f"{ds}/{condition_label}")
            for ds in sorted(df["dataset"].unique())
        ]
        matrices = [m for m in matrices if m is not None]
        if not matrices:
            print(f"  SKIP {condition_label}: no confusion matrices")
            continue
        aggregate_matrix = sum(matrices[1:], matrices[0])
        metrics = compute_classification_metrics(aggregate_matrix)

        row = {"condition_label": condition_label, "spliceai_threshold": spliceai_threshold, **metrics}
        for col, category in _SAMPLE_COUNT_CATEGORIES.items():
            row[col] = int(sample_matches(df, category).sum())
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty or baseline_condition not in summary_df["condition_label"].values:
        print(f"  NOTE: baseline condition '{baseline_condition}' not in scope -- no pct-change columns")
        return summary_df

    baseline_row = summary_df.loc[summary_df["condition_label"] == baseline_condition].iloc[0]
    for col in _PCT_CHANGE_METRICS + list(_SAMPLE_COUNT_CATEGORIES.keys()):
        base_val = baseline_row[col]
        summary_df[f"pct_change_{col}"] = np.where(
            base_val != 0, 100 * (summary_df[col] - base_val) / base_val, np.nan,
        )
    return summary_df


def compute_splice_ablation_fixed_population_summary(
    root: Optional[str] = None,
    dataset_configs: Optional[Dict] = None,
    datasets_filter: Optional[List[str]] = None,
    baseline_condition: str = "thresh_0.2",
) -> pd.DataFrame:
    """Table B: isolates the calibration-only effect from population drift
    (see compute_splice_ablation_aggregate_summary's docstring for the
    confound this addresses). Fixes the variant population to
    `baseline_condition`'s own (default "thresh_0.2", matching
    Scoreset.splicing_filter's hardcoded defaults), and for every OTHER
    condition re-applies THAT condition's own calibration.json point_ranges
    to this SAME fixed population via
    analysis.discovery.recompute_points_from_calibration -- the same
    primitive analysis.robustness's own _iter_robustness_scored uses to
    apply a condition's point_ranges to a fixed reference population.

    PLP/BLB/gnomAD/Synonymous counts are therefore identical across every
    row here by construction (always baseline_condition's own counts) --
    any metric difference reflects purely a calibration-threshold shift,
    not population composition drift (that's Table A's job).
    """
    from analysis.discovery import resolve_component, recompute_points_from_calibration
    from src.assay_calibration.plot_utils.utils import compute_classification_metrics

    conditions = discover_splice_ablation_conditions(root)
    condition_dirs = {label: d for label, _, d in conditions}
    condition_thresholds = {label: t for label, t, _ in conditions}
    if baseline_condition not in condition_dirs:
        print(f"  SKIP splice ablation fixed-population summary: baseline condition "
              f"'{baseline_condition}' not found under {root or cfg.SPLICE_ABLATION_ROOT}")
        return pd.DataFrame()

    baseline_threshold = condition_thresholds[baseline_condition]
    baseline_df = load_splice_ablation_variants(
        condition_dirs[baseline_condition], dataset_configs, datasets_filter,
        baseline_threshold, baseline_threshold is not None,
    )
    if baseline_df.empty:
        print(f"  SKIP splice ablation fixed-population summary: no variants for baseline "
              f"condition '{baseline_condition}'")
        return pd.DataFrame()
    primary_method = sorted(baseline_df["method"].unique())[0]
    baseline_df = baseline_df[baseline_df["method"] == primary_method]

    rows = []
    for condition_label, spliceai_threshold, condition_dir in conditions:
        tree, model_selections, calibrations = discover_outputs(condition_dir)
        scored_frames = []
        for dataset in sorted(baseline_df["dataset"].unique()):
            df_ds = baseline_df[baseline_df["dataset"] == dataset]
            if dataset not in tree:
                continue
            comp = resolve_component(dataset, list(tree[dataset].keys()), model_selections, dataset_configs)
            methods_here = list(tree[dataset].get(comp, {}).keys())
            if not methods_here:
                continue
            cal_path = (calibrations or {}).get(dataset, {}).get(sorted(methods_here)[0], {}).get(comp)
            if cal_path is None:
                continue
            try:
                scored = recompute_points_from_calibration(df_ds, cal_path)
            except (ValueError, FileNotFoundError, KeyError) as e:
                print(f"  SKIP {dataset}/{condition_label} (fixed-population): {e}")
                continue
            scored["dataset"] = dataset
            scored_frames.append(scored)
        if not scored_frames:
            print(f"  SKIP {condition_label}: no matching calibrations for baseline population")
            continue
        scored_df = pd.concat(scored_frames, ignore_index=True)

        matrices = [
            build_confusion_matrix(
                scored_df[scored_df["dataset"] == ds], use_oob=False,
                label=f"{ds}/{condition_label}(fixed-pop)",
            )
            for ds in sorted(scored_df["dataset"].unique())
        ]
        matrices = [m for m in matrices if m is not None]
        if not matrices:
            continue
        aggregate_matrix = sum(matrices[1:], matrices[0])
        metrics = compute_classification_metrics(aggregate_matrix)
        rows.append({"condition_label": condition_label, "spliceai_threshold": spliceai_threshold, **metrics})

    summary_df = pd.DataFrame(rows)
    if summary_df.empty or baseline_condition not in summary_df["condition_label"].values:
        return summary_df
    baseline_row = summary_df.loc[summary_df["condition_label"] == baseline_condition].iloc[0]
    for col in _PCT_CHANGE_METRICS:
        base_val = baseline_row[col]
        summary_df[f"pct_change_{col}"] = np.where(base_val != 0, 100 * (summary_df[col] - base_val) / base_val, np.nan)
    return summary_df


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
