#!/usr/bin/env python3
"""
Standalone plotting for tests/benchmark_num_fits_dataframe.py's summary.csv:
median + IQR ribbon of best-of-N degradation vs. restart count (num_fits),
pooled across every (dataset, n_c) row.

The actual plotting logic lives in
analysis.robustness.plot_fit_number_comparison_curve so it's usable both
from here and from the analysis/ notebook-style pipeline
(analyze_pipeline_output.py) -- this script is just a thin CLI around it.

Raw "delta" (train_ll difference) isn't interpretable across datasets with
different likelihood scales, so when --train-lls-json is given (the sibling
train_lls.json tests/benchmark_num_fits_dataframe.py also writes), this adds
two normalized metrics instead:
  - delta_std (recommended/default): delta divided by that (dataset, n_c)'s
    own restart-to-restart standard deviation -- a dimensionless "how many
    SDs of this dataset's own noise" measure. Intuitive regardless of the
    raw log-likelihood's sign/scale.
  - geometric_mean_lr_pct: 100*exp(delta) -- mathematically bounded in (0%,
    100%] (delta is a difference of two per-observation average
    log-densities, so exp(delta) is scale-invariant regardless of the raw
    LL values' sign), but do NOT read this as a "quality percentage":
    densities aren't probabilities (they can exceed 1), so e.g. 90% here
    does NOT mean "90% as many correct classifications" or any other
    intuitively-linear notion of quality -- it is specifically a
    geometric-mean likelihood *ratio*. Kept available but secondary; see
    analysis.robustness.compute_delta_std_column's docstring for the full
    reasoning.

Usage:
    python tests/plot_fit_number_comparison.py \\
        --summary-csv /tmp/benchmark_num_fits_dataframe_full/summary.csv \\
        --train-lls-json /tmp/benchmark_num_fits_dataframe_full/train_lls.json \\
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

from analysis.robustness import (
    plot_fit_number_comparison_curve, compute_delta_std_column, summarize_delta_std_table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", required=True,
                        help="summary.csv from tests/benchmark_num_fits_dataframe.py")
    parser.add_argument("--train-lls-json", default=None,
                        help="train_lls.json from the same run -- enables the "
                             "delta_std/geometric_mean_lr_pct normalized metrics")
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--metric", default=None,
                        help="Column to plot vs. num_fits. Default: delta_std if "
                             "--train-lls-json is given, else raw delta")
    parser.add_argument("--per-dataset", action="store_true",
                        help="Also plot one curve per dataset (in addition to the pooled "
                             "cross-dataset curve)")
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    print(f"Loaded {len(df)} rows, {df['dataset'].nunique()} datasets, "
          f"num_fits levels: {sorted(df['num_fits'].unique())}")

    metric = args.metric
    if args.train_lls_json:
        df = compute_delta_std_column(df, args.train_lls_json)
        metric = metric or "delta_std"
        for m in ("delta_std", "geometric_mean_lr_pct"):
            table = summarize_delta_std_table(df, metric=m)
            print(f"\n{m} -- median (IQR: 25th-75th) by restart count:")
            print(table.to_string(index=False))
    metric = metric or "delta"

    plot_fit_number_comparison_curve(
        df, metric=metric, figure_dir=args.figure_dir, label="all_datasets",
    )

    if args.per_dataset:
        for (dataset, n_c), sub in df.groupby(["dataset", "n_c"]):
            plot_fit_number_comparison_curve(
                sub, metric=metric, figure_dir=args.figure_dir,
                label=f"{dataset}_{n_c}c",
            )


if __name__ == "__main__":
    main()
