"""
Compare ExCALIBR-vs-ClinVar confusion matrices across the pathogenic-percentile
sweep run via `run_igvf_batch.py --auto-bidirectional --pathogenic-percentile
{5,10,15,25,50} --benign-percentile 95`.

Each percentile's output tree is loaded exactly like
`analysis/analyze_pipeline_output.py`'s `main()` (discover_outputs ->
load_all_variants -> per-dataset build_confusion_matrix, primary method),
except for one input that must NOT come from analysis/config.py's defaults:
these five runs all used
`--dataset /data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_merged_89datasets.tsv.gz`,
not config.DATASET_TSV (a different file) -- passed explicitly below as
DATASET_TSV so load_all_variants doesn't silently fall back to the wrong one.
`--dataset-configs` does match config.DATASET_CONFIGS's current default
(dataset_configs_jul_2026.json), so that one is reused. `--precomputed-fits`
isn't needed here -- only the density-overlay plots use it, not confusion
matrices.

Run as a script:
    python analysis/path_percentile_confusion.py [--figure-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import config
from analysis.discovery import discover_outputs, load_all_variants
from analysis.confusion import build_confusion_matrix, draw_diverging_confusion_heatmap
from analysis.plot_common import save_and_show, pretty_method
from src.assay_calibration.plot_utils.utils import compute_classification_metrics

# --- Inputs specific to these five runs (see module docstring) -------------
DATASET_TSV = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_merged_89datasets.tsv.gz"

# percentile label -> output dir. No "_pXX" suffix == the standard/default
# pathogenic-percentile-5 run.
PERCENTILE_OUTPUT_DIRS = {
    "5": "/data/ross/assay_calibration/exc_pp_bidirectional_test",
    "10": "/data/ross/assay_calibration/exc_pp_bidirectional_test_p10",
    "15": "/data/ross/assay_calibration/exc_pp_bidirectional_test_p15",
    "25": "/data/ross/assay_calibration/exc_pp_bidirectional_test_p25",
    "50": "/data/ross/assay_calibration/exc_pp_bidirectional_test_p50",
}


def load_percentile_aggregate_matrix(output_dir: Path, dataset_configs: dict):
    """Discover + load one percentile's output tree, then sum its per-dataset
    ExCALIBR-vs-ClinVar confusion matrices (primary method, use_oob=True) into
    one aggregate 2x3 DataFrame. Returns (aggregate_matrix_or_None, n_datasets,
    method_label)."""
    tree, model_selections, calibrations = discover_outputs(output_dir)
    if not tree:
        print(f"  SKIP {output_dir}: no *_variants.csv / *_calibration.json found")
        return None, 0, None

    df = load_all_variants(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=None, calibrations=calibrations,
        min_controls=0, dataset_tsv=DATASET_TSV,
    )
    if df.empty:
        print(f"  SKIP {output_dir}: no variants loaded")
        return None, 0, None

    method = sorted(df["method"].unique())[0]
    df_m = df[df["method"] == method]

    aggregate = None
    n_datasets = 0
    for dataset in sorted(df_m["dataset"].unique()):
        df_ds = df_m[df_m["dataset"] == dataset]
        mat = build_confusion_matrix(df_ds, use_oob=True, label=f"{dataset}/{method}")
        if mat is None:
            continue
        aggregate = mat.copy() if aggregate is None else aggregate + mat
        n_datasets += 1

    return aggregate, n_datasets, method


def _draw_confusion_panel(ax, aggregate: pd.DataFrame, title: str, n_datasets: int):
    """Draw one heatmap panel using the diverging Blue(Benign)/Gray(IR)/
    Red(Pathogenic) row-normalized style shared with
    analysis.confusion.make_single_confusion_figure (see
    draw_diverging_confusion_heatmap), adapted to plot into a pre-existing
    `ax` (a subplot) instead of creating its own figure."""
    metrics = compute_classification_metrics(aggregate)

    label_map = {"PLP": "P/LP", "BLB": "B/LB", "IR": "Indeterminate",
                 "Normal": "Benign", "Abnormal": "Pathogenic"}
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
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.text(
        0.5, -0.28,
        f"DOR: {metrics['dor_standard']:.1f}  |  Coverage: {100 * metrics['coverage']:.1f}%  |  n={n_datasets}",
        transform=ax.transAxes, fontsize=9, ha="center", va="top", color="#555555",
    )


def plot_percentile_confusion_grid(results: dict, figure_dir: Path, method_label: str = ""):
    """results: {percentile_label: (aggregate_matrix_or_None, n_datasets)}.

    One figure, one row, one panel per percentile with a valid matrix.
    """
    valid = [(pct, mat, n) for pct, (mat, n) in results.items() if mat is not None]
    if not valid:
        print("  SKIP percentile confusion grid: no percentile produced a matrix")
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 4.6))
    if len(valid) == 1:
        axes = [axes]

    for ax, (pct, mat, n_datasets) in zip(axes, valid):
        _draw_confusion_panel(ax, mat, title=f"Pathogenic percentile: {pct}", n_datasets=n_datasets)

    fig.suptitle(
        f"ExCALIBR vs. ClinVar confusion matrix by pathogenic percentile ({pretty_method(method_label)})"
        if method_label else "ExCALIBR vs. ClinVar confusion matrix by pathogenic percentile",
        fontsize=15, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    save_and_show(fig, figure_dir / "path_percentile_confusion_grid.png")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ExCALIBR-vs-ClinVar confusion matrices across a pathogenic-percentile sweep",
    )
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--dataset-configs", default=config.DATASET_CONFIGS)
    args = parser.parse_args()

    figure_dir = Path(args.figure_dir) if args.figure_dir else Path(config.FIGURE_DIR)
    figure_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_configs) as f:
        dataset_configs = json.load(f)

    print("=" * 80)
    print("PATH-PERCENTILE CONFUSION COMPARISON")
    print("=" * 80)

    results = {}
    primary_method = ""
    for pct, output_dir in PERCENTILE_OUTPUT_DIRS.items():
        print(f"\n--- pathogenic-percentile {pct} ({output_dir}) ---")
        aggregate, n_datasets, method = load_percentile_aggregate_matrix(Path(output_dir), dataset_configs)
        if aggregate is not None:
            primary_method = primary_method or method
            metrics = compute_classification_metrics(aggregate)
            print(f"  Loaded {n_datasets} datasets, method={method}")
            print(f"  Aggregate matrix:\n{aggregate}")
            print(f"  DOR={metrics['dor_standard']:.2f}  Coverage={100 * metrics['coverage']:.1f}%  "
                  f"Accuracy={100 * metrics['accuracy']:.1f}%")
        results[pct] = (aggregate, n_datasets)

    plot_percentile_confusion_grid(results, figure_dir, method_label=primary_method)

    print(f"\n{'=' * 80}\nDONE\n{'=' * 80}")
    print(f"Figures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
    sys.exit(0)
