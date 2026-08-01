#!/usr/bin/env python
"""
Main entry point for Assay Calibration Pipeline
"""
import os
import sys
import argparse
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.assay_calibration.pipeline.config import PipelineConfig, resolve_prior_mode
from src.assay_calibration.pipeline.fit_bootstrap import BootstrapRunner
from src.assay_calibration.pipeline.model_selection import bootstrap_paired_test
from src.assay_calibration.pipeline.visualize import (
    generate_visualizations,
    load_precomputed_fits,
)
from src.assay_calibration.pipeline.variant_evidence import compute_variant_table
from src.assay_calibration.pipeline.utils import setup_logging, save_results, load_dataset_from_df
from src.assay_calibration.pipeline.progress import ProgressReporter
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# (n_bootstraps, fits_per_bootstrap) shorthands -- see docs/configuration.md#quality-vs-speed-presets
BOOTSTRAP_PRESETS = {
    "light": (20, 8),
    "medium": (100, 8),
    "large": (500, 8),
    "xl": (1000, 8),
    "finest": (1000, 100),
}

def main():
    parser = argparse.ArgumentParser(
        description="Assay Calibration Pipeline - Bootstrap fitting and calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (3c only, parallel execution)
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021

  # Also fit 2c (e.g. to compare against 3c)
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 \\
      --components 2 3

  # Run from precomputed bootstrap fits
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 \\
      --precomputed-fits /path/to/results.json.gz --components 3

  # Run with OOB evidence using precomputed splits
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 \\
      --precomputed-fits /path/to/results.json.gz --oob --splits-file /path/to/splits.pkl

  # Higher-quality run via a preset shorthand (light/medium/large/xl/finest)
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 \\
      --preset large

  # Run with custom bootstrap count and save fits
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 \\
      --n-bootstraps 500 --save-fits

  # Single-threaded execution (slow)
  python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021 --mode single
        """
    )

    # Required arguments
    parser.add_argument("--dataset", required=True, help="Path to input CSV dataset")
    parser.add_argument("--name", required=True, help="Dataset name (used for output files)")

    # Precomputed fits
    parser.add_argument("--precomputed-fits", default=None,
                       help="Path to precomputed bootstrap fits (gzipped JSON). "
                            "Skips bootstrap fitting step.")
    parser.add_argument("--splits-file", default=None,
                       help="Path to precomputed bootstrap splits pickle (for OOB with precomputed fits)")

    # Model parameters
    parser.add_argument("--components", type=int, nargs="+",
                       default=[3], metavar="K",
                       help="Component counts to fit (default: 3 only -- assumed at least as good "
                            "as 2c for most assays, not always true but a reasonable default; pass "
                            "e.g. --components 2 3 to fit both and compare). Integers 2-10.")

    # Bootstrap parameters. --preset is the primary quality/speed control;
    # --n-bootstraps/--fits-per-bootstrap are an advanced escape hatch for
    # setting the two underlying numbers directly (each overrides --preset
    # for that one setting only, so e.g. --preset large --n-bootstraps 300
    # uses 300 bootstraps at large's 8 fits-per-bootstrap).
    parser.add_argument("--preset", choices=list(BOOTSTRAP_PRESETS), default="light",
                       help="Quality/speed level (default: light, fastest). medium/large/xl/finest "
                            "are progressively slower and more stable -- see "
                            "docs/configuration.md#quality-vs-speed-presets for what each buys you. "
                            "Most users should just pick a preset; --n-bootstraps/"
                            "--fits-per-bootstrap below are for manual control instead.")
    parser.add_argument("--n-bootstraps", type=int, default=None,
                       help="(Advanced) number of bootstrap iterations, overriding --preset's value. "
                            "Default: whatever --preset implies (20 for 'light').")
    parser.add_argument("--fits-per-bootstrap", type=int, default=None,
                       help="(Advanced) fits per bootstrap iteration, overriding --preset's value. "
                            "Default: whatever --preset implies (8 for 'light').")

    # Execution mode
    # NOTE: for running many datasets across a SLURM cluster, use the separate
    # batch HPC workflow (slurm/prepare.py + slurm/submit_array.sh) documented
    # in the README instead -- these two modes are for running this single
    # dataset directly, in this one process.
    parser.add_argument("--mode", choices=["parallel", "single"],
                       default="parallel",
                       help="'parallel' (default): fit bootstraps using --n-jobs worker "
                            "processes. 'single': fit bootstraps one at a time in this "
                            "process (slower; only useful for debugging).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                       help="Number of parallel jobs (-1 = all CPUs, default: -1)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                       help="cpu (default): existing per-job joblib path. "
                            "gpu: batch fits through src/assay_calibration/fit_utils/jax_batch and "
                            "run on GPU instead. "
                            "Untested on GPU as of authoring -- validate with "
                            "tests/test_batch_em_parity.py first.")

    # Output options
    parser.add_argument("--output-dir", default="./calibration_output",
                       help="Output directory (default: ./calibration_output)")
    parser.add_argument("--save-fits", action="store_true",
                       help="(Deprecated, bootstrap fits are now always saved when computed fresh)")
    parser.add_argument("--viz-only", action="store_true",
                       help="Regenerate visualizations only — skip variant tables and calibration JSON save")

    # --- Algorithm modifications: knobs that change how the calibration is computed ---

    # Prior
    parser.add_argument("--manual-prior", type=float, default=None,
                       help="Manually set prior probability (0-1). Skips empirical estimation.")
    parser.add_argument("--no-median-prior", action="store_true",
                       help="Use 5th percentile thresholds instead of median prior")
    parser.add_argument("--use-equation", action="store_true",
                       help="Use equation for 2c prior (instead of EM estimation)")

    # Benign sample / postprocessing
    parser.add_argument("--benign-method", choices=["benign", "avg", "synonymous"],
                       default="avg", help="Method for benign sample (default: avg)")
    parser.add_argument("--conservative-monotonicity", action="store_true",
                       help="Conservative enforcement of monotonicity on evidence thresholds")
    parser.add_argument("--no-postprocess", action="store_true",
                       help="Skip all point-range postprocessing (monotonicity enforcement "
                            "and extend-to-limits) and return raw LR-threshold-crossing "
                            "intervals exactly as fitted. Mainly a debugging/inspection tool "
                            "for looking at unprocessed evidence -- NOT needed for "
                            "bidirectional assays (e.g. LoF/GoF in one assay), which are "
                            "already handled automatically by --auto-bidirectional.")

    # Percentile thresholds
    parser.add_argument("--pathogenic-percentile", type=float, default=5.0,
                       help="Conservative (lower-bound/pathogenic-direction) percentile used for "
                            "all bootstrap LR+/threshold percentile calculations (conservative "
                            "thresholds, C-range, OOB LR percentiles, per-variant LR percentiles). "
                            "Paired by default with 100-p as the upper (benign-direction) bound -- "
                            "override that independently with --benign-percentile. "
                            "Default: 5.0 (matches prior hardcoded 5th/95th behavior).")
    parser.add_argument("--benign-percentile", type=float, default=None,
                       help="Upper (benign-direction) percentile, independent of "
                            "--pathogenic-percentile. Omit to keep the historical symmetric "
                            "pairing (100 - pathogenic-percentile); set explicitly to decouple "
                            "the two, e.g. to sweep --pathogenic-percentile while always keeping "
                            "the benign-direction bound at the 95th percentile.")

    # Bidirectional-assay auto-detection (majority vote across bootstrap
    # fits) instead of requiring --no-postprocess by hand. On by default. See
    # PipelineConfig.auto_bidirectional and BIDIRECTIONAL_VOTE_THRESHOLD /
    # clean_benign_fragments_no_extend / clean_bidirectional_pathogenic_evidence
    # in fit_utils/point_ranges.py.
    parser.add_argument("--auto-bidirectional", dest="auto_bidirectional",
                       action="store_true", default=True,
                       help="(default: on) Auto-detect bidirectional assays from mixture-component "
                            "weights (a pathogenic-like component on each side of a benign-like "
                            "component) and, when a majority of bootstrap fits show the pattern, "
                            "skip standard monotonicity enforcement for this dataset's pathogenic "
                            "tiers (benign tiers still get cleaned up, just without extend-to-limits) "
                            "instead of the standard monotonicity/extend-to-xlims postprocessing. "
                            "Only applies for n_c >= 3.")
    parser.add_argument("--no-auto-bidirectional", dest="auto_bidirectional",
                       action="store_false",
                       help="Disable automatic bidirectional-assay detection; use standard "
                            "monotonicity postprocessing unconditionally (or combine with "
                            "--no-postprocess to disable postprocessing entirely).")

    # Pathomechanism prior (PN/PU mode only)
    parser.add_argument("--pathomechanism-prior",
                       dest="pathomechanism_prior_enabled", action="store_true", default=None,
                       help="[PN/PU mode only] Enable the dual LR+/prior pathomechanism "
                            "approach: decompose the pathogenic-labeled sample's score "
                            "density into gamma*f_D(x) + (1-gamma)*f_N(x), where f_N is FIXED to "
                            "an anchor density (benign/synonymous blend in PN mode, raw gnomAD in "
                            "PU mode -- anchored, no label-switching), gamma is the estimated "
                            "fraction of PLP-labeled variants whose disease mechanism this assay "
                            "actually measures, and f_D ('assay-relevant pathogenic' density) "
                            "feeds a genuinely separate pathogenic-direction (PS3) prior+LR+ pair "
                            "(reported as pathomechanism_prior, i.e. rho = P(pathogenic AND "
                            "mechanism-detectable)) in the calibration JSON. The benign-direction "
                            "(BS3) prior+LR+ pair (reported as prior, i.e. pi = P(pathogenic, any "
                            "mechanism)) always stays raw/mechanism-agnostic. "
                            "PLP_frac_pathomechanism_measured (the gamma estimate) and stability "
                            "flags are reported in the calibration JSON. Mutually exclusive with "
                            "--filter-pathogenic-sample-by-lr. Default: off (raw single LR+/prior).")
    parser.add_argument("--no-pathomechanism-prior",
                       dest="pathomechanism_prior_enabled", action="store_false", default=None,
                       help=argparse.SUPPRESS)
    # EXPERIMENTAL, hidden: LR-filter cleaning of the pathogenic sample. Kept for
    # experimentation only -- mutually exclusive with --pathomechanism-prior above.
    # Parses with default=None; resolve_prior_mode reconciles it after parsing.
    parser.add_argument("--filter-pathogenic-sample-by-lr",
                       dest="filter_pathogenic_sample_by_lr", action="store_true", default=None,
                       help=argparse.SUPPRESS)
    parser.add_argument("--no-filter-pathogenic-sample-by-lr",
                       dest="filter_pathogenic_sample_by_lr", action="store_false", default=None,
                       help=argparse.SUPPRESS)
    # EXPERIMENTAL, hidden: which f_D construction --pathomechanism-prior uses.
    # Only meaningful when --pathomechanism-prior is enabled; defaults to
    # 'subtraction' (per-component excess max(0, w_P - w_N), renormalized) when
    # unset. 'masking' keeps w_P wherever it exceeds w_N, zero elsewhere,
    # renormalized -- more generous toward borderline components, less smooth
    # at the w_P==w_N boundary. Kept hidden: an advanced tuning knob, not a
    # top-level user-facing choice.
    parser.add_argument("--pathomechanism-method",
                       dest="pathomechanism_method", choices=["off", "subtraction", "masking"],
                       default=None,
                       help=argparse.SUPPRESS)

    # Model selection (only relevant when fitting multiple --components)
    parser.add_argument("--no-auto-select", action="store_true",
                       help="Disable automatic model selection (use all fitted models)")
    parser.add_argument("--selection-alpha", type=float, default=0.05,
                       help="Significance level for model selection (default: 0.05)")
    parser.add_argument("--no-conservative", action="store_true",
                       help="Use p-value test instead of conservative 5th percentile")

    # ClinVar / ACMG-mapping options
    parser.add_argument("--clinvar-release", choices=["2026", "2025", "2018"], default="2026", help="ClinVar release year")
    parser.add_argument("--min-clinvar-star", type=int, default=1,
                       help="Minimum ClinVar review stars (default: 1)")
    parser.add_argument("--acmg-mapping-method",
                       choices=["tavtigian", "acmg_bayes", "all"],
                       default="tavtigian",
                       help="ACMG-mapping method (default: tavtigian). "
                            "'acmg_bayes' = prior-adaptive posterior-based classification, "
                            "exact at the four ACMG boundaries for every prior, with a "
                            "display-only ACMG point label derived from the same log(LR+) "
                            "for reporting (supersedes the former separate 'piecewise'/"
                            "'continuous'/'strict_additive' options); "
                            "'all' = run both and write per-method output files.")
    parser.add_argument("--acmg-bayes-targets", type=json.loads, default=None,
                       help="Override acmg_bayes target posteriors as a JSON object, "
                            "e.g. '{\"LB\": 0.01, \"B\": 0.001}'. Any key not given falls "
                            "back to the standard targets (P=0.99, LP=0.90, LB=0.10, "
                            "B=0.01). Ignored when acmg-mapping-method=tavtigian.")
    parser.add_argument("--acmg-bayes-floor-at-neutral", action="store_true",
                       help="Clamp LB/B/LP/P targets against the gene's own prior so no "
                            "threshold ever requires evidence in the wrong direction. "
                            "Ignored when acmg-mapping-method=tavtigian.")

    # OOB evidence
    parser.add_argument("--oob", action="store_true",
                       help="Compute out-of-bag per-variant evidence (slower)")
    parser.add_argument("--oob-min-samples", type=int, default=1,
                       help="Min OOB bootstrap samples per variant (default: 1)")

    # --- Utilities ---
    parser.add_argument("--population-type", default="gnomAD",
                       choices=["all_variants", "all_nsSNV", "all_missense_nsSNV",
                                "gnomAD", "gnomAD_nsSNV", "gnomAD_missense_nsSNV"],
                       help="Population type for dataset loading (default: gnomAD)")
    parser.add_argument("--scoreset-flipped-override", type=str, default=None,
                       choices=["true", "false"],
                       help="Override automatic scoreset flip detection")
    parser.add_argument("--sample-names", type=str, nargs="+", default=None,
                       help="Explicit sample names matching column order in data "
                            "(e.g. 'Pathogenic/Likely Pathogenic' 'Benign/Likely Benign' gnomAD Synonymous)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Master seed for full reproducibility (train/val splits, EM "
                            "initializations, and E-step Monte Carlo draws are all derived "
                            "from this). Omit for the historical unseeded behavior.")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging (component params, weights, flip detection, point ranges)")
    parser.add_argument("--progress-file", default=None,
                        help="Path to write JSON progress updates (used by web backend). "
                            "Has no effect on pipeline output when omitted.")


    args = parser.parse_args()

    # --n-bootstraps/--fits-per-bootstrap, if explicitly passed, override the
    # --preset shorthand for that one setting only.
    preset_n_bootstraps, preset_fits_per_bootstrap = BOOTSTRAP_PRESETS[args.preset]
    if args.n_bootstraps is None:
        args.n_bootstraps = preset_n_bootstraps
    if args.fits_per_bootstrap is None:
        args.fits_per_bootstrap = preset_fits_per_bootstrap

    # Combine the visible --pathomechanism-prior on/off toggle with the hidden
    # --pathomechanism-method sub-choice into the single value resolve_prior_mode
    # expects, defaulting the sub-choice to "subtraction" when enabled.
    if args.pathomechanism_prior_enabled:
        pm_flag = args.pathomechanism_method or "subtraction"
    elif args.pathomechanism_prior_enabled is False:
        pm_flag = "off"
    else:
        pm_flag = args.pathomechanism_method  # None unless explicitly set

    # Reconcile the mutually-exclusive prior-cleaning flags (defaults to off:
    # raw single LR+/prior).
    args.filter_pathogenic_sample_by_lr, args.pathomechanism_method = resolve_prior_mode(
        args.filter_pathogenic_sample_by_lr, pm_flag
    )

    # Parse scoreset_flipped_override
    flipped_override = None
    if args.scoreset_flipped_override is not None:
        flipped_override = args.scoreset_flipped_override.lower() == "true"

    # Create configuration
    config = PipelineConfig(
        dataset_csv=args.dataset,
        dataset_name=args.name,
        output_dir=args.output_dir,
        precomputed_fits=args.precomputed_fits,
        splits_file=args.splits_file,
        n_bootstraps=args.n_bootstraps,
        num_fits_per_bootstrap=args.fits_per_bootstrap,
        seed=args.seed,
        components=args.components,
        use_median_prior=not args.no_median_prior,
        use_2c_equation=args.use_equation,
        liberal_monotonicity=not args.conservative_monotonicity,
        postprocess_point_ranges=not args.no_postprocess,
        auto_bidirectional=args.auto_bidirectional,
        benign_method=args.benign_method,
        manual_prior=args.manual_prior,
        population_type=args.population_type,
        scoreset_flipped_override=flipped_override,
        compute_oob=args.oob,
        oob_min_samples=args.oob_min_samples,
        execution_mode=args.mode,
        n_jobs=args.n_jobs,
        device=args.device,
        auto_select_model=not args.no_auto_select,
        model_selection_alpha=args.selection_alpha,
        use_conservative_selection=not args.no_conservative,
        save_bootstrap_fits=args.save_fits,
        clinvar_release=args.clinvar_release,
        min_clinvar_star=args.min_clinvar_star,
        sample_names=args.sample_names,
        debug=args.debug,
        viz_only=args.viz_only,
        progress_file=args.progress_file,
        pathogenic_percentile=args.pathogenic_percentile,
        benign_percentile=args.benign_percentile,
        filter_pathogenic_sample_by_lr=args.filter_pathogenic_sample_by_lr,
        pathomechanism_method=args.pathomechanism_method,
        acmg_mapping_method=(args.acmg_mapping_method
                              if args.acmg_mapping_method != "all" else "tavtigian"),
        acmg_bayes_targets=args.acmg_bayes_targets,
        acmg_bayes_floor_at_neutral=args.acmg_bayes_floor_at_neutral,
    )

    # Resolve ACMG-mapping methods to run
    acmg_mapping_methods = (["tavtigian", "acmg_bayes"]
                             if args.acmg_mapping_method == "all"
                             else [args.acmg_mapping_method])

    # Run pipeline once per ACMG-mapping method.  For 'all', share Step 1/2
    # across methods by computing fits once and re-running Step 3+ per method.
    if len(acmg_mapping_methods) == 1:
        config.acmg_mapping_method = acmg_mapping_methods[0]
        run_calibration_pipeline(config)
    else:
        run_calibration_pipeline_multi(config, acmg_mapping_methods)

def run_calibration_pipeline_multi(config: PipelineConfig, acmg_mapping_methods):
    """Run the calibration pipeline once per ACMG-mapping method, sharing fits.

    The first method computes (or loads) bootstrap fits and saves them to
    disk; subsequent methods reload those fits from the saved path so we
    don't refit.  Outputs are distinguished by a "_<acmg_mapping_method>"
    suffix on the dataset name, e.g. ``MYGENE_tavtigian_3c_calibration.json``.
    """
    original_name = config.dataset_name
    original_precomputed = config.precomputed_fits
    saved_fits_path = None

    for i, acmg_mapping_method in enumerate(acmg_mapping_methods):
        config.acmg_mapping_method = acmg_mapping_method
        config.dataset_name = f"{original_name}_{acmg_mapping_method}"

        if i == 0:
            # Pass original_name so precomputed fits are looked up without the
            # acmg-method suffix (fits are identical across methods).
            run_calibration_pipeline(config, fits_name=original_name)
            # Subsequent methods reuse the saved fits regardless of how Step 1 ran.
            cand = os.path.join(
                config.output_dir,
                f"{config.dataset_name}_bootstrap_fits.json.gz",
            )
            if os.path.exists(cand):
                saved_fits_path = cand
        else:
            if saved_fits_path is not None:
                config.precomputed_fits = saved_fits_path
            run_calibration_pipeline(config, fits_name=original_name)

    # restore for any caller that holds the config
    config.dataset_name = original_name
    config.precomputed_fits = original_precomputed


def run_calibration_pipeline(config: PipelineConfig, fits_name: Optional[str] = None):
    """Main pipeline execution.

    fits_name: override the dataset name used to look up precomputed fits.
    Useful when running --acmg-mapping-method all, where config.dataset_name
    is suffixed (e.g. 'gof_benta_tavtigian') but fits are stored under the
    original name ('gof_benta').
    """

    # Setup
    logger = setup_logging(config.output_dir, config.dataset_name)
    reporter = ProgressReporter(config.progress_file)
    logger.info("="*80)
    logger.info("ASSAY CALIBRATION PIPELINE")
    logger.info("="*80)
    logger.info(f"\nDataset: {config.dataset_name}")
    logger.info(f"Input: {config.dataset_csv}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Components: {config.components}")
    if config.precomputed_fits:
        logger.info(f"Precomputed fits: {config.precomputed_fits}")
    else:
        logger.info(f"Bootstraps: {config.n_bootstraps}")
    logger.info(f"Execution: {config.execution_mode}")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Step 1: Get bootstrap results (either from precomputed or fresh fitting)
    dataset_splits = None
    fits_are_fresh = False  # Track whether we computed fits (vs loaded precomputed)

    if config.precomputed_fits:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Loading Precomputed Bootstrap Fits")
        logger.info("="*80)

        bootstrap_results = load_precomputed_fits(
            config.precomputed_fits, fits_name or config.dataset_name
        )
        logger.info(f"\nLoaded {len(bootstrap_results)} bootstrap iterations")

        # Load precomputed splits if provided (for OOB)
        if config.splits_file and config.compute_oob:
            logger.info(f"Loading splits from {config.splits_file}")
            with open(config.splits_file, 'rb') as f:
                all_splits = pickle.load(f)
            # splits_file may contain {dataset: {seed: split}} or {seed: split}
            if config.dataset_name in all_splits:
                dataset_splits = all_splits[config.dataset_name]
            else:
                dataset_splits = all_splits
            logger.info(f"  Loaded splits for {len(dataset_splits)} bootstrap seeds")
        elif config.compute_oob:
            # Recover splits from bootstrap seeds using the Fit class
            logger.info("Recovering splits from bootstrap seeds...")
            dataset_splits = _recover_splits_from_seeds(
                bootstrap_results, config, logger
            )
    else:
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Bootstrap Fitting")
        logger.info("="*80)

        runner = BootstrapRunner(config, reporter=reporter)
        bootstrap_results, dataset_splits = runner.run()
        fits_are_fresh = True

        logger.info(f"\nCompleted {len(bootstrap_results)} bootstrap iterations")

    # Count valid fits across all bootstraps and components (both paths)
    valid_fits = sum(
        1 for seed_results in bootstrap_results.values()
        for v in seed_results.values() if v is not None
    )
    total_fits = len(bootstrap_results) * len(config.components)


    # Step 2: Model selection (if fitting multiple components)
    # Always process ALL fitted components; model selection annotates the
    # preferred one (saved to *_model_selection.json) but does not filter.
    selected_components = {f"{c}c": c for c in config.components}
    selected_k = None

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
                logger.info(f"\nModel selection (conservative): {selected_k}c")
            else:
                selected_k = test_result['selected_k']
                logger.info(f"\nModel selection (p-value): {selected_k}c")

            logger.info(f"  Processing all components: {list(selected_components.keys())}")
    else:
        logger.info(f"\nUsing fitted components: {list(selected_components.keys())}")

    reporter.stage("model_selection",
        selected_model=f"{selected_k}c" if selected_k else None,
        components_fitted=list(selected_components.keys()),
    )

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

    if getattr(config, "viz_only", False):
        logger.info("\n--viz-only: skipping variant tables and calibration save")
        logger.info("\nPIPELINE COMPLETE (visualizations only)")
        return

    # Step 3b: Per-variant evidence table
    logger.info("\n" + "="*80)
    logger.info("STEP 3b: Per-Variant Evidence Table")
    logger.info("="*80)

    sep = "\t" if config.dataset_csv.endswith((".tsv", ".tsv.gz")) else ","
    df_vt = pd.read_csv(config.dataset_csv, sep=sep)
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

        reporter.stage("visualization",
           component=component_key,
           prior=calibration.get("prior"),
           scoreset_flipped=calibration.get("scoreset_flipped"),
           valid_fits=valid_fits,
           total_fits=total_fits,
       )
        
    reporter.stage("variant_table",
        variants_assigned=sum(len(pd.read_csv(
            Path(config.output_dir) / f"{config.dataset_name}_{k}_variants.csv"
        )) for k in results),
    )

    # Simpler alternative (avoids re-reading CSVs):
    reporter.stage("variant_table")

    # Step 4: Save results
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Saving Results")
    logger.info("="*80)

    save_results(
        results=results,
        bootstrap_results=bootstrap_results if fits_are_fresh else None,
        config=config,
        logger=logger,
        selected_k=selected_k,
    )

    reporter.complete(
        selected_model=f"{selected_k}c" if selected_k else None,
        components=list(results.keys()),
    )

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {config.output_dir}")
    logger.info(f"  - Calibration: {config.dataset_name}_<component>_calibration.json")
    logger.info(f"  - LR values: {config.dataset_name}_<component>_lr_values.json.gz")
    logger.info(f"  - Visualization: {config.dataset_name}_<component>_visualization.png")
    logger.info(f"  - Variant table: {config.dataset_name}_<component>_variants.csv")
    if fits_are_fresh:
        logger.info(f"  - Bootstrap fits: {config.dataset_name}_bootstrap_fits.json.gz")


def _recover_splits_from_seeds(
    bootstrap_results: Dict,
    config: PipelineConfig,
    logger,
) -> Dict:
    """
    Recover train/val splits from bootstrap seeds by re-running the bootstrap
    sampling with the same seeds (deterministic).
    """
    from src.assay_calibration.fit_utils.fit import Fit

    df = pd.read_csv(config.dataset_csv)
    scoreset = load_dataset_from_df(df, config)
    fitter = Fit(scoreset)

    splits = {}
    seeds = sorted(bootstrap_results.keys(), key=lambda k: int(k))
    logger.info(f"  Recovering splits for {len(seeds)} bootstrap seeds...")

    for seed_key in seeds:
        seed = int(seed_key)
        # generate_fit_jobs with any component count to get the train/val split
        n_c = config.components[0]
        jobs = fitter.generate_fit_jobs(
            component_range=[n_c],
            bootstrap_seed=seed,
            check_monotonic=True,
            num_fits=1,  # only need 1 to get the split
        )
        if jobs:
            splits[seed] = {
                'val_observations': jobs[0]['val_observations'],
                'val_sample_assignments': jobs[0]['val_sample_assignments'],
                'val_variant_indices': jobs[0].get('val_variant_indices'),
            }

    logger.info(f"  Recovered splits for {len(splits)} seeds")
    return splits


if __name__ == "__main__":
    main()
