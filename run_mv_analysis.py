#!/usr/bin/env python3
"""
Thin CLI driver for post-fit multivariate analysis.

Given an aggregated bootstrap_results.json.gz (see hpc/aggregate_results.py)
and a --gene-set/--gene, builds the matching ms via the exact same pluggable
ingestion hpc/prepare.py uses (reused directly, not duplicated), runs
MVCalibrationAnalysis across every fitted config (3c/4c/5c/6c/...) AND every
partial_pattern_mode ("none"/"old_gate"/"pu_unmix"/"conservative" -- see
src/assay_calibration/multivariate_analysis/gene_set_analysis.py's
MODE_DISPLAY_NAMES), and prints one flat comparison table.

Usage
-----
    python run_mv_analysis.py --results-json /path/bootstrap_results.json.gz \\
        --gene-set tp53

    python run_mv_analysis.py --results-json /path/bootstrap_results.json.gz \\
        --gene-set labelseq --gene braf

    python run_mv_analysis.py --results-json /path/bootstrap_results.json.gz \\
        --gene-set fgfr   # combined FGFR1-4 scoreset by default
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "hpc"))

import prepare  # hpc/prepare.py -- reuse its ingestion dispatch, don't duplicate it
from src.assay_calibration.multivariate_data.common import gene_set_dataset_label
from src.assay_calibration.multivariate_analysis.gene_set_analysis import (
    build_gene_set_analysis, report_configs_and_modes,
)
from src.assay_calibration.multivariate_analysis.mv_calibration import _PARTIAL_PATTERN_MODES

_GENE_SET_CHOICES = ["fgfr", "tp53", "labelseq", "card11", "predictor-mv", "combined"]
_AUX_INDICES = {"tp53": [4], "card11": [4, 5]}


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
    ap.add_argument("--fgfr-separate", action="store_true")
    ap.add_argument("--modes", nargs="+", default=list(_PARTIAL_PATTERN_MODES),
                    choices=list(_PARTIAL_PATTERN_MODES))
    ap.add_argument("--path-percentile", type=float, default=5)
    ap.add_argument("--min-valid-boots", type=int, default=1)
    ap.add_argument("--aux-path-percentile", type=float, default=None)
    ap.add_argument("--aux-ben-percentile", type=float, default=None)
    ap.add_argument("--csv-out", default=None,
                    help="Optional path to save the comparison table as CSV")
    args = ap.parse_args()

    if args.gene_set == "predictor-mv":
        from src.assay_calibration.multivariate_data.predictors import (
            load_predictor_ms, predictor_dataset_label,
        )
        if not args.gene:
            raise SystemExit("--gene required for --gene-set predictor-mv")
        ms = load_predictor_ms(args.gene, args.data_dir or args.predictor_data_dir)
        dataset_name = predictor_dataset_label(args.gene)
        gene = args.gene
    else:
        gene_ms_map = prepare._load_and_filter_gene_ms_map(args.gene_set, args)
        if args.gene_set == "tp53":
            gene = "TP53"
        elif args.gene_set == "card11":
            gene = "CARD11"
        elif args.gene_set == "fgfr" and not args.fgfr_separate:
            gene = "FGFR_combined"
        else:
            if not args.gene:
                raise SystemExit(
                    f"--gene required for --gene-set {args.gene_set} "
                    f"(available: {sorted(gene_ms_map)})"
                )
            gene = args.gene
        if gene not in gene_ms_map:
            raise SystemExit(f"{gene!r} not found; available: {sorted(gene_ms_map)}")
        ms = gene_ms_map[gene]
        dataset_name = gene_set_dataset_label(gene, args.gene_set)

    aux_idx = _AUX_INDICES.get(args.gene_set)

    analysis = build_gene_set_analysis(
        ms, gene, args.results_json, dataset_name=dataset_name,
        auxiliary_pathogenic_indices=aux_idx,
    )

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

    table = report_configs_and_modes(analysis, modes=args.modes, **run_kwargs)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"\n=== {gene} ({args.gene_set}) ===")
    print(table.to_string(index=False))
    if args.csv_out:
        table.to_csv(args.csv_out, index=False)
        print(f"\nSaved to {args.csv_out}")


if __name__ == "__main__":
    main()
