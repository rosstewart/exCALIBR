#!/usr/bin/env python
"""
Main entry point for Assay Calibration Pipeline
"""
import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict

from src.assay_calibration.pipeline.config import PipelineConfig
from src.assay_calibration.pipeline.fit_bootstrap import BootstrapRunner
from src.assay_calibration.pipeline.model_selection import bootstrap_paired_test
from src.assay_calibration.pipeline.visualize import generate_visualizations
from src.assay_calibration.pipeline.variant_evidence import compute_variant_table
from src.assay_calibration.pipeline.utils import setup_logging, save_results, load_dataset_from_df
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def main():
    parser = argparse.ArgumentParser(
        description="Assay Calibration Pipeline - Bootstrap fitting and calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (2c and 3c, parallel execution)
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021
  
  # Run only 2-component fits on SLURM
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 --components 2 --mode slurm
  
  # Run with custom bootstrap count and save fits
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 --n-bootstraps 500 --save-fits
  
  # Single-threaded execution (slow)
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 --mode single
        """
    )
    
    # Required arguments
    parser.add_argument("--dataset", required=True, help="Path to input CSV dataset")
    parser.add_argument("--name", required=True, help="Dataset name (used for output files)")
    
    # Output options
    parser.add_argument("--output-dir", default="./calibration_output", 
                       help="Output directory (default: ./calibration_output)")
    parser.add_argument("--save-fits", action="store_true",
                       help="Save all bootstrap fit results (large files)")
    
    # Model parameters
    parser.add_argument("--components", type=int, nargs="+", choices=[2, 3, 4],
                       default=[2, 3], help="Component counts to fit (default: 2 3)")
    parser.add_argument("--benign-method", choices=["benign", "avg", "synonymous"],
                       default="avg", help="Method for benign sample (default: avg)")
    parser.add_argument("--no-median-prior", action="store_true",
                       help="Use 5th percentile thresholds instead of median prior")
    parser.add_argument("--use-equation", action="store_true",
                       help="Use equation for 2c prior (instead of EM estimation)")
    parser.add_argument("--conservative-monotonicity", action="store_true",
                       help="Conservative enforcement of monotonicity on evidence thresholds")
    parser.add_argument("--manual-prior", type=float, default=None,
                       help="Manually set prior probability (0-1). Skips empirical estimation.")
    parser.add_argument("--population-type", default="gnomAD",
                       choices=["all_variants", "all_nsSNV", "all_missense_nsSNV",
                                "gnomAD", "gnomAD_nsSNV", "gnomAD_missense_nsSNV"],
                       help="Population type for dataset loading (default: gnomAD)")
    
    # OOB evidence
    parser.add_argument("--oob", action="store_true",
                       help="Compute out-of-bag per-variant evidence (slower)")
    parser.add_argument("--oob-min-samples", type=int, default=1,
                       help="Min OOB bootstrap samples per variant (default: 1)")
    
    # Bootstrap parameters
    parser.add_argument("--n-bootstraps", type=int, default=1000,
                       help="Number of bootstrap iterations (default: 1000)")
    parser.add_argument("--fits-per-bootstrap", type=int, default=100,
                       help="Fits per bootstrap iteration (default: 100)")
    
    # Execution mode
    parser.add_argument("--mode", choices=["slurm", "parallel", "single"],
                       default="parallel", help="Execution mode (default: parallel)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs (-1 = all CPUs, default: -1)")
    
    # SLURM options
    slurm = parser.add_argument_group("SLURM options")
    slurm.add_argument("--slurm-account", default="default", help="SLURM account")
    slurm.add_argument("--slurm-partition", default="short", help="SLURM partition")
    slurm.add_argument("--slurm-time", type=int, default=23, help="SLURM time (hours)")
    slurm.add_argument("--slurm-mem", type=int, default=8, help="SLURM memory (GB)")
    slurm.add_argument("--slurm-cpus", type=int, default=12, help="CPUs per SLURM task")
    slurm.add_argument("--slurm-conda-env", default="assay_calibration",
                       help="Conda environment name")
    slurm.add_argument("--slurm-modules", nargs="*",
                       help="Module load commands (e.g. 'module load anaconda3/2024.06')")
    
    # Model selection
    parser.add_argument("--no-auto-select", action="store_true",
                       help="Disable automatic model selection (use all fitted models)")
    parser.add_argument("--selection-alpha", type=float, default=0.05,
                       help="Significance level for model selection (default: 0.05)")
    parser.add_argument("--no-conservative", action="store_true",
                       help="Use p-value test instead of conservative 5th percentile")
    
    # ClinVar options
    parser.add_argument("--clinvar-release", choices=["2025", "2018"], default="2025", help="ClinVar release year")
    parser.add_argument("--min-clinvar-star", type=int, default=1,
                       help="Minimum ClinVar review stars (default: 1)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        dataset_csv=args.dataset,
        dataset_name=args.name,
        output_dir=args.output_dir,
        n_bootstraps=args.n_bootstraps,
        num_fits_per_bootstrap=args.fits_per_bootstrap,
        components=args.components,
        use_median_prior=not args.no_median_prior,
        use_2c_equation=args.use_equation,
        liberal_monotonicity=not args.conservative_monotonicity,
        benign_method=args.benign_method,
        manual_prior=args.manual_prior,
        population_type=args.population_type,
        compute_oob=args.oob,
        oob_min_samples=args.oob_min_samples,
        execution_mode=args.mode,
        n_jobs=args.n_jobs,
        slurm_account=args.slurm_account,
        slurm_partition=args.slurm_partition,
        slurm_time_hours=args.slurm_time,
        slurm_mem_gb=args.slurm_mem,
        slurm_cpus_per_task=args.slurm_cpus,
        slurm_conda_env=args.slurm_conda_env,
        slurm_module_commands=args.slurm_modules,
        auto_select_model=not args.no_auto_select,
        model_selection_alpha=args.selection_alpha,
        use_conservative_selection=not args.no_conservative,
        save_bootstrap_fits=args.save_fits,
        clinvar_release=args.clinvar_release,
        min_clinvar_star=args.min_clinvar_star,
    )
    
    # Run pipeline
    run_calibration_pipeline(config)

def run_calibration_pipeline(config: PipelineConfig):
    """Main pipeline execution"""
    
    # Setup
    logger = setup_logging(config.output_dir, config.dataset_name)
    logger.info("="*80)
    logger.info("ASSAY CALIBRATION PIPELINE")
    logger.info("="*80)
    logger.info(f"\nDataset: {config.dataset_name}")
    logger.info(f"Input: {config.dataset_csv}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Components: {config.components}")
    logger.info(f"Bootstraps: {config.n_bootstraps}")
    logger.info(f"Execution: {config.execution_mode}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Step 1: Run bootstrap fits
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Bootstrap Fitting")
    logger.info("="*80)
    
    runner = BootstrapRunner(config)
    bootstrap_results, dataset_splits = runner.run()
    
    logger.info(f"\nCompleted {len(bootstrap_results)} bootstrap iterations")
    
    # Step 2: Model selection (if fitting multiple components)
    selected_components = {}
    if len(config.components) > 1 and config.auto_select_model:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Model Selection")
        logger.info("="*80)
        
        if 2 in config.components and 3 in config.components and len(config.components) == 2:
            test_result = bootstrap_paired_test(
                bootstrap_results,
                k_range=[2, 3],
                alpha=config.model_selection_alpha,
                verbose=True
            )
            
            # Save test results
            test_file = Path(config.output_dir) / f"{config.dataset_name}_model_selection.json"
            with open(test_file, 'w') as f:
                json.dump(test_result, f, indent=2)
            
            if config.use_conservative_selection:
                selected_k = test_result['conservative_k']
                logger.info(f"\nUsing conservative selection (5th percentile): {selected_k}c")
            else:
                selected_k = test_result['selected_k']
                logger.info(f"\nUsing p-value selection: {selected_k}c")
            
            selected_components = {f"{selected_k}c": selected_k}
        else:
            # Just use all fitted components
            selected_components = {f"{c}c": c for c in config.components}
    else:
        # Use all fitted components
        selected_components = {f"{c}c": c for c in config.components}
        logger.info(f"\nUsing fitted components: {list(selected_components.keys())}")
    
    # Step 3a: Generate visualizations and export
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Visualization and Export")
    logger.info("="*80)
    
    results = generate_visualizations(
        bootstrap_results=bootstrap_results,
        config=config,
        selected_components=selected_components,
        logger=logger
    )
    
    # Step 3b: Per-variant evidence table
    logger.info("\n" + "="*80)
    logger.info("STEP 3b: Per-Variant Evidence Table")
    logger.info("="*80)
    
    df_vt = pd.read_csv(config.dataset_csv)
    scoreset_vt = load_dataset_from_df(df_vt, config)
    
    for component_key, calibration in results.items():
        variant_df = compute_variant_table(
            scoreset=scoreset_vt,
            calibration=calibration,
            config=config,
            dataset_splits=dataset_splits if config.compute_oob else None,
            logger=logger,
        )
        table_path = Path(config.output_dir) / f"{config.dataset_name}_{component_key}_variants.csv"
        variant_df.to_csv(table_path, index=False)
        logger.info(f"  Saved variant table: {table_path} ({len(variant_df)} variants)")
    
    # Step 4: Save results
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Saving Results")
    logger.info("="*80)
    
    save_results(
        results=results,
        bootstrap_results=bootstrap_results if config.save_bootstrap_fits else None,
        config=config,
        logger=logger
    )
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {config.output_dir}")
    logger.info(f"  - Calibration: {config.dataset_name}_<component>_calibration.json")
    logger.info(f"  - Visualization: {config.dataset_name}_<component>_visualization.png")
    logger.info(f"  - Variant table: {config.dataset_name}_<component>_variants.csv")
    if config.save_bootstrap_fits:
        logger.info(f"  - Bootstrap fits: {config.dataset_name}_bootstrap_fits.pkl")

if __name__ == "__main__":
    main()
