#!/usr/bin/env python3
"""
CLI driver for post-fit multivariate analysis, now with an optional
comparison against existing UV (univariate ExCALIBR) calibration baselines.

Given an aggregated bootstrap_results.json.gz (see hpc/aggregate_results.py)
and a --gene-set/--gene, builds the matching ms via hpc/prepare.py's
pluggable ingestion (reused via mv_analysis.build, not duplicated), runs
MVCalibrationAnalysis across every fitted config (3c/4c/5c/6c/...) AND every
partial_pattern_mode, and prints one flat comparison table -- MV rows plus,
when a UV source exists for this gene-set (see mv_analysis/uv_sources.py),
'UV non-conflicting' and 'UV max' rows using identical metric definitions.

Usage
-----
    python run_mv_analysis.py --results-json /path/bootstrap_results.json.gz \\
        --gene-set tp53

    python run_mv_analysis.py --results-json /path/bootstrap_results.json.gz \\
        --gene-set labelseq --gene braf

    python run_mv_analysis.py --results-json /path/bootstrap_results.json.gz \\
        --gene-set fgfr   # combined FGFR1-4 scoreset by default; UV comparison
                          # skipped (pending data), see mv_analysis/README.md
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mv_analysis.build import build_ms
from mv_analysis.report import build_comparison_table
from src.assay_calibration.multivariate_analysis.mv_calibration import _PARTIAL_PATTERN_MODES

_GENE_SET_CHOICES = ["fgfr", "tp53", "labelseq", "card11", "predictor-mv", "combined"]
_AUX_INDICES = {"tp53": [4], "card11": [4, 5]}
_NO_UV_GENE_SETS = {"fgfr"}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-json", required=True,
                    help="Path to aggregated bootstrap_results.json.gz")
    ap.add_argument("--gene-set", required=True, choices=_GENE_SET_CHOICES)
    ap.add_argument("--gene", default=None,
                    help="Required for labelseq/predictor-mv/combined (multi-gene "
                         "gene-sets); ignored for tp53/card11 (always one gene) and "
                         "fgfr (combined by default -- use --fgfr-separate + --gene "
                         "to pick one of FGFR1-4)")
    ap.add_argument("--genes", nargs="+", default=None)
    ap.add_argument("--exclude-genes", nargs="+", default=None)
    ap.add_argument("--dataframe", default=None)
    ap.add_argument("--predictor-data-dir",
                    default="/data/ross/assay_calibration/predictor_calibrations/"
                            "single_gene_calibration_data")
    ap.add_argument("--data-dir", default=None,
                    help="[predictor-mv] defaults to --predictor-data-dir if unset")
    ap.add_argument("--rpvs-all", action="store_true")
    ap.add_argument("--kawoligo-seed", type=int, default=0)
    ap.add_argument("--kawoligo-jitter-sigma", type=float, default=0.1,
                     help="[tp53] required by hpc/prepare.py's TP53 dispatch; "
                          "matches multivariate_data/tp53.py's own default")
    ap.add_argument("--fgfr-separate", action="store_true")
    ap.add_argument("--modes", nargs="+", default=list(_PARTIAL_PATTERN_MODES),
                    choices=list(_PARTIAL_PATTERN_MODES))
    ap.add_argument("--path-percentile", type=float, default=5)
    ap.add_argument("--min-valid-boots", type=int, default=1)
    ap.add_argument("--aux-path-percentile", type=float, default=None)
    ap.add_argument("--aux-ben-percentile", type=float, default=None)
    ap.add_argument("--csv-out", default=None,
                    help="Optional path to save the comparison table as CSV")
    ap.add_argument("--compare-uv", action=argparse.BooleanOptionalAction, default=None,
                    help="Compare against existing UV calibrations (default: on, "
                         "except for --gene-set fgfr/predictor-mv where no UV source "
                         "exists/is bridgeable yet -- see mv_analysis/uv_sources.py)")
    args = ap.parse_args()

    gene, ms, dataset_name = build_ms(args)
    aux_idx = _AUX_INDICES.get(args.gene_set)

    compare_uv = args.compare_uv
    if compare_uv is None:
        compare_uv = args.gene_set not in _NO_UV_GENE_SETS

    run_kwargs = dict(
        path_percentile=args.path_percentile,
        min_valid_boots=args.min_valid_boots,
        reestimate_marginal_weights=False,
        enforce_marginal_monotonicity=False,
        liberal_marginal_monotonicity=False,
    )
    if args.aux_path_percentile is not None:
        run_kwargs["aux_path_percentile"] = args.aux_path_percentile
    if args.aux_ben_percentile is not None:
        run_kwargs["aux_ben_percentile"] = args.aux_ben_percentile

    table, uv_dataset_names = build_comparison_table(
        gene, args.gene_set, ms, args.results_json,
        dataset_name=dataset_name, auxiliary_pathogenic_indices=aux_idx,
        modes=args.modes, compare_uv=compare_uv, **run_kwargs,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"\n=== {gene} ({args.gene_set}) ===")
    if uv_dataset_names is not None:
        print(f"UV comparison built from {len(uv_dataset_names)} dataset(s): {uv_dataset_names}")
    print(table.to_string(index=False))
    if args.csv_out:
        table.to_csv(args.csv_out, index=False)
        print(f"\nSaved to {args.csv_out}")


if __name__ == "__main__":
    main()
