#!/usr/bin/env python
"""
CLI runner for analysis.all_variant_evidence — builds, from run_igvf_batch.py /
run_pipeline.py output, the variant-assay and variant-aggregate in-bag
evidence tables (every variant an assay measured, regardless of keep_mask/
sample membership — see analysis/all_variant_evidence.py's module docstring
for why these are always in-bag, never OOB), plus (optionally) a
predictor-only table stripped from analysis.config.PREDICTOR_EVIDENCE_SOURCE_CSV.

Both evidence tables carry, per row, the integer ACMG points/evidence code AND a
continuous `lr_plus`/`posterior` (plus the raw `lr_plus_p5`/`lr_plus_p95` bounds
they were combined from). `lr_plus` is reconciled against that row's own
postprocessed points, so the point scale and the Bayesian scale never disagree.

Examples
--------
# All three tables, written under --output-dir/tables:
python analysis/run_build_all_variant_evidence_tables.py --output-dir /path/to/pipeline/output

# Skip the predictor-only table (e.g. PREDICTOR_EVIDENCE_SOURCE_CSV not on this machine):
python analysis/run_build_all_variant_evidence_tables.py --no-predictor-table

# Restrict to a subset of datasets (e.g. to test timing before a full run):
python analysis/run_build_all_variant_evidence_tables.py --datasets MSH2_Jia_2021 BAP1_Waters_2024
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis import config as cfg
from analysis.discovery import discover_outputs, load_all_variants


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=None, help=f"default: {cfg.OUTPUT_DIR}")
    parser.add_argument("--dataset-tsv", default=None, help=f"default: {cfg.DATASET_TSV}")
    parser.add_argument("--dataset-configs", default=None, help=f"default: {cfg.DATASET_CONFIGS}")
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="Restrict to these dataset names (default: every dataset discovered under --output-dir)")
    parser.add_argument("--out-dir", default=None,
                         help=f"Directory to write the CSVs to (default: {cfg.FIGURE_DIR}/tables)")
    parser.add_argument("--no-predictor-table", action="store_true",
                         help="Skip the predictor-only table (analysis.config.PREDICTOR_EVIDENCE_SOURCE_CSV)")
    parser.add_argument("--no-dataframe-with-points", action="store_true",
                         help="Skip dataframe_with_points.csv.gz (the input dataframe + evidence columns)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or cfg.OUTPUT_DIR)
    dataset_tsv = args.dataset_tsv or cfg.DATASET_TSV
    dataset_configs_path = args.dataset_configs or cfg.DATASET_CONFIGS
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.FIGURE_DIR) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tree, model_selections, calibrations = discover_outputs(output_dir)
    if not tree:
        print(f"ERROR: no *_variants.csv or *_calibration.json found under {output_dir}")
        sys.exit(1)

    with open(dataset_configs_path) as f:
        dataset_configs = json.load(f)

    df = load_all_variants(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=args.datasets, calibrations=calibrations, min_controls=0,
    )
    if df.empty:
        print("ERROR: no variants loaded — check --datasets filter / output-dir")
        sys.exit(1)

    datasets = sorted(df["dataset"].unique())
    primary_method = sorted(df["method"].unique())[0]
    print(f"{len(datasets)} dataset(s), primary_method={primary_method!r}")

    from analysis.all_variant_evidence import (
        build_all_variant_groups_table, build_dataframe_with_points,
        build_predictor_only_table, build_variant_aggregate_table,
        build_variant_assay_table, merge_variant_aggregate_with_predictors,
    )
    from analysis.discovery import load_master_df

    print("\nBuilding all-variant-groups population (in-bag, keep_mask-independent) ...")
    df_all_groups = build_all_variant_groups_table(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        calibrations=calibrations, dataset_tsv=dataset_tsv, datasets=datasets, primary_method=primary_method,
    )
    print(f"  {len(df_all_groups):,} measured variants "
          f"(one row per measurement; {int(df_all_groups['group_n_rows'].sum()):,} input rows behind them)")

    # Cached per process (analysis.discovery._master_df_cache), so this does
    # not re-read the file build_all_variant_groups_table already loaded.
    df_input = load_master_df(dataset_tsv)

    print("\nBuilding variant-assay table (one row per input dataframe row) ...")
    variant_assay_df = build_variant_assay_table(df_all_groups, df_input=df_input)
    variant_assay_path = out_dir / "variant_assay_specific_evidence.csv.gz"
    variant_assay_df.to_csv(variant_assay_path, index=False, compression="gzip")
    print(f"  {len(variant_assay_df):,} rows, {variant_assay_df['dataset'].nunique()} datasets, "
          f"{variant_assay_df['gene_symbol'].nunique()} genes -> {variant_assay_path}")

    print("\nBuilding variant-aggregate table (one row per nucleotide coordinate) ...")
    variant_aggregate_df = build_variant_aggregate_table(variant_assay_df)
    variant_aggregate_path = out_dir / "variant_aggregated_functional_evidence.csv.gz"
    variant_aggregate_df.to_csv(variant_aggregate_path, index=False, compression="gzip")
    print(f"  {len(variant_aggregate_df):,} unique variants, "
          f"{variant_aggregate_df['gene_symbol'].nunique()} genes -> {variant_aggregate_path}")

    if not args.no_dataframe_with_points:
        print("\nBuilding dataframe-with-points (input dataframe + evidence columns) ...")
        df_with_points = build_dataframe_with_points(df_input, variant_assay_df)
        dfp_path = out_dir / "dataframe_with_points.csv.gz"
        df_with_points.to_csv(dfp_path, index=False, compression="gzip")
        n_ev = int(df_with_points["excalibr_points"].notna().sum())
        print(f"  {len(df_with_points):,} rows ({n_ev:,} with evidence, "
              f"{len(df_with_points) - n_ev:,} filtered) -> {dfp_path}")
        counts = df_with_points["excalibr_filter_reason"].value_counts()
        for reason, n in counts.items():
            if reason:
                print(f"    excalibr_filter_reason={reason}: {n:,}")

    if not args.no_predictor_table and not cfg.warn_if_missing(
        cfg.PREDICTOR_EVIDENCE_SOURCE_CSV, "predictor-only evidence table",
    ):
        print("\nBuilding predictor-only table ...")
        predictor_df = build_predictor_only_table()
        predictor_path = out_dir / "predictor_only_evidence.csv.gz"
        predictor_df.to_csv(predictor_path, index=False, compression="gzip")
        print(f"  {len(predictor_df):,} rows -> {predictor_path}")

        print("\nMerging variant-aggregate table with predictor-only table ...")
        merged_df = merge_variant_aggregate_with_predictors(variant_aggregate_df, predictor_df)
        merged_path = out_dir / "variant_aggregated_experimental_predictive_evidence.csv.gz"
        merged_df.to_csv(merged_path, index=False, compression="gzip")
        print(f"  {len(merged_df):,} rows -> {merged_path}")


if __name__ == "__main__":
    main()
