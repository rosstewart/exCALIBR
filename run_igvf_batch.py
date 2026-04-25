#!/usr/bin/env python
"""
IGVF Batch Processing - Run calibration pipeline across multiple datasets
with per-dataset configurations loaded from a JSON config file.

Input:
  - A CSV/TSV containing all datasets, distinguished by a "Dataset" column
  - A JSON config file mapping dataset names to [n_c, benign_method, {overrides}]
  - Precomputed bootstrap fits (gzipped JSON) keyed by dataset name

Example:
  python run_igvf_batch.py \\
      --dataset data/integrated_variant_effect_dataset.tsv.gz \\
      --dataset-configs src/igvf_configs/dataset_configs_jan_2026.json \\
      --precomputed-fits /data/ross/assay_calibration/results.json.gz \\
      --output-dir ./igvf_output \\
      --oob --splits-file /data/ross/assay_calibration/splits.pkl
"""
import os
import sys
import json
import argparse
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from src.assay_calibration.pipeline.config import PipelineConfig
from src.assay_calibration.pipeline.visualize import (
    generate_visualizations,
    load_precomputed_fits,
    process_component_fits,
)
from src.assay_calibration.pipeline.model_selection import bootstrap_paired_test
from src.assay_calibration.pipeline.variant_evidence import compute_variant_table
from src.assay_calibration.pipeline.utils import (
    setup_logging,
    save_results,
    load_dataset_from_df,
)


def parse_dataset_config(
    config_entry,
) -> Tuple[str, str, Dict]:
    """
    Parse a dataset config entry.

    Supports two formats:
      - ["3c", "avg"]                          -> n_c="3c", benign="avg", overrides={}
      - ["3c", "avg", {"liberal_monotonicity": false}] -> with overrides
    """
    if isinstance(config_entry, dict):
        n_c = config_entry["n_c"]
        benign = config_entry.get("benign_method", "avg")
        overrides = {k: v for k, v in config_entry.items()
                     if k not in ("n_c", "benign_method")}
        return n_c, benign, overrides

    # List/tuple format
    n_c = config_entry[0]
    benign = config_entry[1] if len(config_entry) > 1 else "avg"
    overrides = config_entry[2] if len(config_entry) > 2 and isinstance(config_entry[2], dict) else {}
    return n_c, benign, overrides


def run_single_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    bootstrap_results: Dict,
    n_c: str,
    benign_method: str,
    overrides: Dict,
    args,
    dataset_splits: Optional[Dict] = None,
) -> Optional[Dict]:
    """Run calibration for a single dataset within the batch.

    Parameters
    ----------
    n_c : str
        Component key, e.g. ``"2c"`` or ``"3c"``.  The special value
        ``"all"`` processes both 2c and 3c (with model selection saved
        but all components output).
    """

    dataset_df = df[df["Dataset"] == dataset_name.replace("_clinvar_2018", "")]
    if len(dataset_df) == 0:
        print(f"  SKIP {dataset_name}: no rows in dataset CSV")
        return None

    # Determine ClinVar release
    clinvar_release = "2018" if "clinvar_2018" in dataset_name else "2025"
    if "not_clinvar_2018" in dataset_name:
        clinvar_release = "2025"

    # Build component list
    if n_c == "all":
        component_list = [2, 3]
    else:
        component_list = [int(n_c.replace("c", ""))]

    # Build per-dataset config
    config = PipelineConfig(
        dataset_csv=args.dataset,
        dataset_name=dataset_name,
        output_dir=os.path.join(args.output_dir, dataset_name),
        components=component_list,
        use_median_prior=True,
        use_2c_equation=False,
        liberal_monotonicity=overrides.get("liberal_monotonicity", True),
        benign_method=benign_method,
        scoreset_flipped_override=overrides.get("scoreset_flipped_override", None),
        compute_oob=args.oob,
        oob_min_samples=args.oob_min_samples,
        n_jobs=args.n_jobs_inner,
        auto_select_model=False,
        clinvar_release=clinvar_release,
        min_clinvar_star=args.min_clinvar_star,
        population_type=args.population_type,
        point_values=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_names=args.sample_names if hasattr(args, "sample_names") else None,
        debug=args.debug if hasattr(args, "debug") else False,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(output_dir), dataset_name)

    try:
        # Load scoreset
        scoreset = load_dataset_from_df(dataset_df, config)

        n_samples = len([s for s in scoreset.samples])
        logger.info(f"Dataset {dataset_name}: {len(scoreset.scores)} variants, "
                     f"{n_samples} samples, components={component_list} {benign_method}")

        # Build selected_components dict
        selected_components = {f"{c}c": c for c in component_list}

        # Model selection (when processing multiple components)
        selected_k = None
        if len(component_list) > 1:
            try:
                test_result = bootstrap_paired_test(
                    bootstrap_results, verbose=False
                )
                selected_k = test_result["conservative_k"]
                logger.info(f"  Model selection: {selected_k}c")

                # Save model selection result
                test_file = output_dir / f"{dataset_name}_model_selection.json"
                with open(test_file, "w") as f:
                    json.dump(test_result, f, indent=2)
            except Exception as e:
                logger.warning(f"  Model selection failed: {e}")

        # Generate calibration for ALL components
        results = generate_visualizations(
            bootstrap_results=bootstrap_results,
            config=config,
            selected_components=selected_components,
            logger=logger,
            scoreset=scoreset,
        )

        if not results:
            logger.warning(f"  No results for {dataset_name}")
            return None

        # Per-variant evidence table
        for comp_key, calibration in results.items():
            variant_df = compute_variant_table(
                scoreset=scoreset,
                calibration=calibration,
                config=config,
                dataset_splits=dataset_splits,
                logger=logger,
            )
            table_path = output_dir / f"{dataset_name}_{comp_key}_variants.csv"
            variant_df.to_csv(table_path, index=False)
            logger.info(f"  Saved: {table_path} ({len(variant_df)} variants)")

        # Save calibration
        save_results(
            results=results,
            bootstrap_results=None,
            config=config,
            logger=logger,
            selected_k=selected_k,
        )

        return results

    except Exception as e:
        logger.error(f"  FAILED {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="IGVF Batch Calibration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset", required=True,
                       help="Path to input CSV/TSV with all datasets (must have 'Dataset' column)")
    parser.add_argument("--dataset-configs", default=None,
                       help="Path to JSON config mapping dataset names to [n_c, benign_method, {overrides}]. "
                            "If omitted, all datasets in --precomputed-fits are processed with "
                            "default settings (2c+3c, avg, model selection enabled).")
    parser.add_argument("--precomputed-fits", required=True,
                       help="Path to precomputed bootstrap fits (gzipped JSON)")
    parser.add_argument("--output-dir", default="./igvf_output",
                       help="Output directory (default: ./igvf_output)")

    # OOB
    parser.add_argument("--oob", action="store_true",
                       help="Compute OOB per-variant evidence")
    parser.add_argument("--splits-file", default=None,
                       help="Path to precomputed splits pickle for OOB")
    parser.add_argument("--oob-min-samples", type=int, default=10,
                       help="Min OOB samples per variant (default: 10)")

    # Filtering
    parser.add_argument("--datasets", nargs="*", default=None,
                       help="Only process these dataset names (default: all in config)")

    # Parallelization
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of datasets to process in parallel (default: 1)")
    parser.add_argument("--n-jobs-inner", type=int, default=-1,
                       help="Number of parallel jobs within each dataset (default: -1 = all CPUs)")

    # ClinVar
    parser.add_argument("--clinvar-release", default="2025", choices=["2025", "2018"])
    parser.add_argument("--min-clinvar-star", type=int, default=1)
    parser.add_argument("--population-type", default="gnomAD",
                       choices=["all_variants", "all_nsSNV", "all_missense_nsSNV",
                                "gnomAD", "gnomAD_nsSNV", "gnomAD_missense_nsSNV"])

    # Model selection override
    parser.add_argument("--auto-select-model", action="store_true",
                       help="Use bootstrap paired test to auto-select n_c instead of config")
    parser.add_argument("--sample-names", type=str, nargs="+", default=None,
                       help="Explicit sample names matching column order in data "
                            "(applied to all datasets; e.g. 'Pathogenic/Likely Pathogenic' "
                            "'Benign/Likely Benign' gnomAD Synonymous)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging (component params, flip detection, point ranges)")

    args = parser.parse_args()

    print("=" * 80)
    print("BATCH CALIBRATION PIPELINE")
    print("=" * 80)

    # Load precomputed fits
    print(f"\nLoading precomputed fits from {args.precomputed_fits}...")
    with gzip.open(args.precomputed_fits, "rt", encoding="utf-8") as f:
        all_bootstrap_results = json.load(f)
    print(f"Loaded fits for {len(all_bootstrap_results)} datasets")

    # Load dataset configs (or auto-discover from precomputed fits)
    if args.dataset_configs is not None:
        with open(args.dataset_configs, "r") as f:
            dataset_configs = json.load(f)
        print(f"\nLoaded {len(dataset_configs)} dataset configurations")
    else:
        # No config provided: auto-discover all datasets from precomputed fits
        # Default: process both 2c and 3c with model selection, avg benign method
        print(f"\nNo --dataset-configs provided; using defaults for all "
              f"{len(all_bootstrap_results)} datasets (both 2c+3c, model selection)")
        dataset_configs = {
            name: ["all", "avg"]  # "all" = process 2c+3c with model selection
            for name in all_bootstrap_results
        }

    # Filter to requested datasets
    if args.datasets:
        dataset_configs = {k: v for k, v in dataset_configs.items() if k in args.datasets}
        print(f"Filtered to {len(dataset_configs)} requested datasets")

    # Load input data
    sep = "\t" if args.dataset.endswith((".tsv", ".tsv.gz")) else ","
    print(f"\nLoading dataset CSV from {args.dataset}...")
    df = pd.read_csv(args.dataset, sep=sep)
    print(f"Loaded {len(df)} rows")

    # Load splits if provided
    all_splits = None
    if args.splits_file and args.oob:
        print(f"\nLoading splits from {args.splits_file}...")
        with open(args.splits_file, "rb") as f:
            all_splits = pickle.load(f)
        print(f"Loaded splits for {len(all_splits)} datasets")

    # Process datasets
    os.makedirs(args.output_dir, exist_ok=True)

    datasets_to_process = []
    for dataset_name, config_entry in dataset_configs.items():
        if dataset_name not in all_bootstrap_results:
            print(f"  SKIP {dataset_name}: not in precomputed fits")
            continue

        n_c, benign_method, overrides = parse_dataset_config(config_entry)

        # Auto-select model if requested (only when config specifies a single n_c;
        # "all" already handles model selection inside run_single_dataset)
        if args.auto_select_model and n_c != "all":
            try:
                test_result = bootstrap_paired_test(
                    all_bootstrap_results[dataset_name], verbose=False
                )
                n_c = f"{test_result['conservative_k']}c"
            except Exception:
                pass  # fallback to config

        # Get splits for this dataset
        dataset_splits = None
        if all_splits and dataset_name in all_splits:
            dataset_splits = all_splits[dataset_name]

        datasets_to_process.append((
            dataset_name,
            all_bootstrap_results[dataset_name],
            n_c, benign_method, overrides, dataset_splits,
        ))

    print(f"\nProcessing {len(datasets_to_process)} datasets...")

    if args.n_jobs == 1:
        # Sequential processing
        for i, (name, boot_results, n_c, benign, ovr, splits) in enumerate(datasets_to_process):
            print(f"\n{'='*80}")
            print(f"[{i+1}/{len(datasets_to_process)}] {name} ({n_c}, {benign})")
            print(f"{'='*80}")
            run_single_dataset(name, df, boot_results, n_c, benign, ovr, args, splits)
    else:
        # Parallel dataset processing
        Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_single_dataset)(
                name, df, boot_results, n_c, benign, ovr, args, splits
            )
            for name, boot_results, n_c, benign, ovr, splits in datasets_to_process
        )

    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
