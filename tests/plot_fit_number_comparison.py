#!/usr/bin/env python3
"""
Standalone plotting for tests/benchmark_num_fits_dataframe.py's summary.csv:
median + IQR ribbon of best-of-N degradation vs. restart count (num_fits),
pooled across every (dataset, n_c) row.

The actual plotting logic lives in
analysis.robustness.plot_fit_number_comparison_curve so it's usable both
from here and from the analysis/ notebook-style pipeline
(analyze_pipeline_output.py) -- this script is just a thin CLI around it.

Usage:
    python tests/plot_fit_number_comparison.py \\
        --summary-csv /tmp/benchmark_num_fits_dataframe_full/summary.csv \\
        --figure-dir /tmp/benchmark_num_fits_dataframe_full/figures
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")

from analysis.robustness import plot_fit_number_comparison_curve


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", required=True,
                        help="summary.csv from tests/benchmark_num_fits_dataframe.py")
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--metric", default="delta",
                        help="Column to plot vs. num_fits (default: delta = "
                             "mean_best - baseline train_ll)")
    parser.add_argument("--per-dataset", action="store_true",
                        help="Also plot one curve per dataset (in addition to the pooled "
                             "cross-dataset curve)")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    print(f"Loaded {len(df)} rows, {df['dataset'].nunique()} datasets, "
          f"num_fits levels: {sorted(df['num_fits'].unique())}")

    plot_fit_number_comparison_curve(
        df, metric=args.metric, figure_dir=args.figure_dir, label="all_datasets",
    )

    if args.per_dataset:
        for (dataset, n_c), sub in df.groupby(["dataset", "n_c"]):
            plot_fit_number_comparison_curve(
                sub, metric=args.metric, figure_dir=args.figure_dir,
                label=f"{dataset}_{n_c}c",
            )


if __name__ == "__main__":
    main()
