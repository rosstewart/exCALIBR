"""
Visualization and calibration result generation
"""
import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging

from ..fit_utils.fit import calculate_score_ranges, thresholds_from_prior
from ..fit_utils.two_sample import density_utils
from ..fit_utils.point_ranges import (
    enforce_monotonicity_point_ranges,
    prior_equation_2c,
    prior_invalid,
    get_fit_prior,
    get_bootstrap_score_ranges,
    remove_insufficient_bootstrap_converage_points,
    check_thresholds_reached,
    extend_points_to_xlims,
    compute_single_fit_log_densities
)
from ..data_utils.dataset import Scoreset, BasicScoreset
from ..fit_utils.utils import serialize_dict
from ..plot_utils.utils import plot_scoreset_example_publication

from .config import PipelineConfig
from .utils import load_dataset_from_df

def load_precomputed_fits(fits_path: str, dataset_name: str) -> Dict:
    """
    Load precomputed bootstrap fits from gzipped JSON.

    The expected format (matching assign_points.py) is::

        {dataset_name: {bootstrap_key: {"2c": {...}, "3c": {...}}, ...}}

    Each fit entry has ``fit``, ``val_ll``, etc.
    Returns the inner dict: ``{bootstrap_key: {"2c": fit, "3c": fit}}``.
    """
    import gzip

    with gzip.open(fits_path, 'rt', encoding='utf-8') as f:
        all_results = json.load(f)

    if dataset_name not in all_results:
        available = list(all_results.keys())[:10]
        raise KeyError(
            f"Dataset '{dataset_name}' not found in precomputed fits. "
            f"Available: {available}{'...' if len(all_results) > 10 else ''}"
        )

    return all_results[dataset_name]


def generate_visualizations(
    bootstrap_results: Dict,
    config: PipelineConfig,
    selected_components: Dict,
    logger: logging.Logger,
    scoreset=None,
) -> Dict:
    """
    Generate visualizations and calibration results

    Args:
        bootstrap_results: Dict of bootstrap fits
        config: Pipeline configuration
        selected_components: Dict mapping component keys to component counts
        logger: Logger instance
        scoreset: Pre-loaded Scoreset (optional; loaded from CSV if None)

    Returns:
        Dict with calibration results
    """

    # Load dataset if not provided
    if scoreset is None:
        df = pd.read_csv(config.dataset_csv)
        scoreset = load_dataset_from_df(df, config)
    results = {}

    # Process each selected component count
    for component_key, n_c in selected_components.items():
        logger.info(f"\nProcessing {component_key}...")

        # Extract fits for this component count
        fits = []
        bootstrap_seeds = []
        for bootstrap_idx in sorted(bootstrap_results.keys(), key=lambda k: int(k)):
            if component_key in bootstrap_results[bootstrap_idx]:
                fit_result = bootstrap_results[bootstrap_idx][component_key]
                if fit_result is not None:
                    fits.append(fit_result)
                    bootstrap_seeds.append(int(bootstrap_idx))
        bootstrap_seeds = np.array(bootstrap_seeds)

        
        logger.info(f"  Valid fits: {len(fits)}/{len(bootstrap_results)}")
        
        if len(fits) == 0:
            logger.warning(f"  No valid fits for {component_key}, skipping")
            continue
        
        # Process fits and generate calibration
        calibration = process_component_fits(
            fits=fits,
            scoreset=scoreset,
            config=config,
            n_c=n_c,
            logger=logger,
            bootstrap_seeds=bootstrap_seeds,
        )

        
        results[component_key] = calibration
        
        # Generate visualization
        output_dir = Path(config.output_dir)
        figure_path = output_dir / f"{config.dataset_name}_{component_key}_visualization.png"
        
        try:
            fig = plot_scoreset_example_publication(
                dataset=config.dataset_name,
                scoreset=scoreset,
                indv_summary=calibration,
                fits=fits,
                score_range=calibration['score_range'],
                config=f"({config.benign_method})",
                n_c=component_key,
                n_samples=len([s for s in scoreset.samples]),
                relax=False,
                flipped=calibration.get('scoreset_flipped', False)
            )
            
            fig.savefig(figure_path, bbox_inches='tight', dpi=300)
            logger.info(f"  Saved visualization: {figure_path}")
            
        except Exception as e:
            logger.error(f"  Failed to generate visualization: {e}")
    
    return results

def process_component_fits(
    fits: List[Dict],
    scoreset: Scoreset,
    config: PipelineConfig,
    n_c: int,
    logger: logging.Logger,
    bootstrap_seeds: np.ndarray = None,
) -> Dict:
    """
    Process bootstrap fits to generate calibration thresholds
    
    This is the core calibration logic adapted from the notebooks
    """
    
    # Identify sample indices - MORE FLEXIBLE DETECTION
    pathogenic_idx, benign_idx, gnomad_idx, synonymous_idx = None, None, None, None
    
    for i, sample_name in enumerate(scoreset.sample_names):
        if scoreset.sample_counts[i] == 0:
            continue
        
        if sample_name == "Pathogenic/Likely Pathogenic":
            pathogenic_idx = i
        elif sample_name == "Benign/Likely Benign":
            benign_idx = i
        elif sample_name == "gnomAD" or sample_name == "population":
            gnomad_idx = i
        elif sample_name == "Synonymous":
            synonymous_idx = i
        else:
            raise ValueError(f"Invalid sample name: {sample_name}")
    
    # More flexible validation - allow missing pathogenic OR missing benign/synonymous
    if gnomad_idx is None:
        raise ValueError("Missing required gnomAD/population sample")
    
    if pathogenic_idx is None and benign_idx is None and synonymous_idx is None:
        raise ValueError("Must have at least pathogenic OR (benign/synonymous)")
    
    # Adjust benign_method based on available samples
    if synonymous_idx is None and (config.benign_method == 'avg' or config.benign_method == 'synonymous'):
        logger.warning(f"  No synonymous sample, setting benign_method from {config.benign_method} to benign")
        config.benign_method = 'benign'
    elif benign_idx is None and (config.benign_method == 'avg' or config.benign_method == 'benign'):
        logger.warning(f"  No benign sample, setting benign_method from {config.benign_method} to synonymous")
        config.benign_method = 'synonymous'

    # Adjust indices to match fit weight arrays.
    # Scoreset (IGVF format) always has columns for all 4 sample types, but
    # fits only include weights for non-empty samples, so indices must shift.
    # BasicScoreset columns already match fit weights directly — no adjustment.
    if not isinstance(scoreset, BasicScoreset):
        if benign_idx is None:
            gnomad_idx -= 1
            if synonymous_idx is not None:
                synonymous_idx -= 1
        if pathogenic_idx is None:
            gnomad_idx -= 1
            if benign_idx is not None:
                benign_idx -= 1
            if synonymous_idx is not None:
                synonymous_idx -= 1

    # Validate indices against actual fit weight length
    n_weights = len(fits[0]['fit']['weights'])
    for name, idx in [("pathogenic", pathogenic_idx), ("benign", benign_idx),
                      ("gnomad", gnomad_idx), ("synonymous", synonymous_idx)]:
        if idx is not None and idx >= n_weights:
            logger.warning(f"  {name}_idx={idx} exceeds fit weights length ({n_weights}), setting to None")
            if name == "synonymous":
                synonymous_idx = None
            elif name == "benign":
                benign_idx = None

    # Re-check benign_method after validation
    if synonymous_idx is None and config.benign_method == 'avg':
        config.benign_method = 'benign'
    elif benign_idx is None and config.benign_method == 'avg':
        config.benign_method = 'synonymous'
    
    logger.info(f"  Sample indices: P={pathogenic_idx}, B={benign_idx}, G={gnomad_idx}, S={synonymous_idx}")
    
    if config.manual_prior is not None:
        # Use manually specified prior — skip estimation
        prior = config.manual_prior
        fit_priors = np.full(len(fits), prior)
        valid_bootstrap_seeds = bootstrap_seeds  # no filtering
        valid_mask = np.ones(len(fits), dtype=bool)
        logger.info(f"  Using manual prior: {prior:.6f}")
    else:
        # Compute priors - PASS ALL INDICES
        if not config.use_2c_equation or n_c != 2:
            # Use EM estimation
            n_cores = config.n_jobs if config.n_jobs > 0 else (os.cpu_count() or 1)
            fit_priors = np.array(Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
                delayed(get_fit_prior)(
                    fit, scoreset, config.benign_method,
                    pathogenic_idx=pathogenic_idx,
                    benign_idx=benign_idx,
                    gnomad_idx=gnomad_idx,
                    synonymous_idx=synonymous_idx
                )
                for fit in fits
            ))
        else:
            # Use 2c equation
            fit_priors = []
            for fit in fits:
                weights = fit['fit']['weights']
                
                if len(weights) == 3:
                    w_p = weights[pathogenic_idx] if pathogenic_idx is not None else None
                    w_g = weights[gnomad_idx]
                    if benign_idx is not None:
                        w_b = weights[benign_idx]
                        w_s = w_b
                    else:
                        w_b = weights[synonymous_idx]
                        w_s = w_b
                elif len(weights) == 4:
                    w_p, w_b, w_g, w_s = weights
                else:
                    raise ValueError(f"Unexpected number of samples: {len(weights)}")
                
                if config.benign_method == 'synonymous':
                    fit_priors.append(prior_equation_2c(w_p, w_s, w_g))
                elif config.benign_method == 'avg':
                    w_bs = (np.array(w_b) + np.array(w_s)) / 2
                    fit_priors.append(prior_equation_2c(w_p, w_bs, w_g))
                else:
                    fit_priors.append(prior_equation_2c(w_p, w_b, w_g))
            
            fit_priors = np.array(fit_priors)
        
        # Filter invalid priors
        valid_mask = ~(np.isnan(fit_priors) | (fit_priors <= 0) | (fit_priors >= 1))
        fit_priors = fit_priors[valid_mask]
        fits = np.array(fits)[valid_mask].tolist()
        valid_bootstrap_seeds = bootstrap_seeds[valid_mask] if bootstrap_seeds is not None else np.array([])
        
        logger.info(f"  Valid priors: {len(fit_priors)}")
        
        # Compute prior
        prior = np.nanmedian(fit_priors)
        logger.info(f"  Prior: {prior:.6f}")
    
    # Setup score range
    observed_scores = scoreset.scores[scoreset._sample_assignments.any(1)]
    score_range = np.linspace(*np.percentile(observed_scores, [0, 100]), 10000)
    
    # Compute log likelihood ratios
    n_cores = config.n_jobs if config.n_jobs > 0 else (os.cpu_count() or 1)
    results_fpfb = Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
        delayed(compute_single_fit_log_densities)(
            fit, prior, score_range, config.benign_method,
            pathogenic_idx=pathogenic_idx,
            benign_idx=benign_idx,
            gnomad_idx=gnomad_idx,
            synonymous_idx=synonymous_idx
        )
        for fit, prior in zip(fits, fit_priors)
    )
    
    # Unpack results
    _log_fp = np.array([r[0] if r[0] is not None else np.full(len(score_range), np.nan) 
                        for r in results_fpfb])
    _log_fb = np.array([r[1] if r[1] is not None else np.full(len(score_range), np.nan) 
                        for r in results_fpfb])
    
    # Get bootstrap score ranges
    results = Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
        delayed(get_bootstrap_score_ranges)(
            fitIdx, fit, fp, fb, score_range, fit_priors, config.point_values
        )
        for fitIdx, (fit, fp, fb) in enumerate(zip(fits, _log_fp, _log_fb))
    )
    
    # Aggregate results
    log_fp = np.full((len(fits), len(score_range)), np.nan)
    log_fb = np.full((len(fits), len(score_range)), np.nan)
    ranges_pathogenic, ranges_benign = [], []
    Cs = []
    
    for fitIdx, log_fp_local, log_fb_local, ranges_p, ranges_b, C in results:
        log_fp[fitIdx] = log_fp_local
        log_fb[fitIdx] = log_fb_local
        ranges_pathogenic.append({key: np.array(value).reshape(-1) for key, value in ranges_p.items()})
        ranges_benign.append({key: np.array(value).reshape(-1) for key, value in ranges_b.items()})
        Cs.append(C)
    
    log_lr_plus = log_fp - log_fb
    
    # Compute C range
    C = np.array([np.nanpercentile(Cs, 5), np.nanpercentile(Cs, 95)])
    
    # Detect if scoreset is flipped - USE MEAN SCORES INSTEAD OF WEIGHTS
    scoreset_flipped = False
    
    if pathogenic_idx is not None:
        path_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:, pathogenic_idx]])
    else:
        path_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:, gnomad_idx]])
    
    if benign_idx is not None or synonymous_idx is not None:
        if config.benign_method == 'avg' and benign_idx is not None and synonymous_idx is not None:
            ben_mean_score = (
                np.mean(scoreset.scores[scoreset._sample_assignments[:, benign_idx]]) +
                np.mean(scoreset.scores[scoreset._sample_assignments[:, synonymous_idx]])
            ) / 2
        elif config.benign_method == 'synonymous' and synonymous_idx is not None:
            ben_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:, synonymous_idx]])
        else:
            ben_mean_score = (
                np.mean(scoreset.scores[scoreset._sample_assignments[:, benign_idx]])
                if benign_idx is not None
                else np.mean(scoreset.scores[scoreset._sample_assignments[:, synonymous_idx]])
            )
    else:
        ben_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:, gnomad_idx]])
    
    if path_mean_score > ben_mean_score:
        scoreset_flipped = True

    # Per-dataset override (e.g. TARDBP_Bolognesi_Faure_2019)
    if config.scoreset_flipped_override is not None:
        scoreset_flipped = config.scoreset_flipped_override

    logger.info(f"  Scoreset flipped: {scoreset_flipped}")
    
    # Compute point ranges - exclude score positions where ALL fits are NaN
    nan_counts = np.isnan(log_lr_plus).sum(0)
    range_subset = nan_counts < log_lr_plus.shape[0]
    
    if config.use_median_prior:
        # Use median prior for unified thresholds
        logger.info("  Using median prior for unified thresholds")
        point_ranges_pathogenic, point_ranges_benign, C = calculate_score_ranges(
            np.nanpercentile(log_lr_plus[:, range_subset], 5, axis=0),
            np.nanpercentile(log_lr_plus[:, range_subset], 95, axis=0),
            prior,
            score_range[range_subset],
            config.point_values,
        )
        point_ranges = {**point_ranges_pathogenic, **point_ranges_benign}
        if prior <= 0 or prior >= 1:
            for point in point_ranges:
                point_ranges[point] = []
    else:
        # Use 5th percentile conservative thresholds
        logger.info("  Using 5th percentile conservative thresholds")
        p_xaxis_5percentile_conservative = 5 if not scoreset_flipped else 95
        b_xaxis_5percentile_conservative = 95 if not scoreset_flipped else 5
        
        p_max = max if not scoreset_flipped else min
        b_min = min if not scoreset_flipped else max
        p_inf = -np.inf if not scoreset_flipped else np.inf
        b_inf = np.inf if not scoreset_flipped else -np.inf
        
        conservative_thresholds = {}
        
        for point_value in config.point_values:
            conservative_thresholds[point_value] = np.nanpercentile(
                [p_max(ranges_p[point_value]) if len(ranges_p[point_value]) > 0 else p_inf
                 for ranges_p in ranges_pathogenic],
                p_xaxis_5percentile_conservative
            )
            
            conservative_thresholds[-1 * point_value] = np.nanpercentile(
                [b_min(ranges_b[-1 * point_value]) if len(ranges_b[-1 * point_value]) > 0 else b_inf
                 for ranges_b in ranges_benign],
                b_xaxis_5percentile_conservative
            )
        
        # Convert thresholds to ranges
        point_ranges = {}
        valid_scores = score_range[range_subset]
        
        for point_value, threshold in conservative_thresholds.items():
            if np.isnan(threshold) or np.isinf(threshold):
                point_ranges[point_value] = []
                continue
            
            if (point_value > 0 and not scoreset_flipped) or (point_value < 0 and scoreset_flipped):
                # Pathogenic or flipped benign
                if abs(point_value) == max(config.point_values):
                    point_ranges[point_value] = [[valid_scores[0], threshold]]
                else:
                    lower_lim = conservative_thresholds[point_value + 1 if point_value > 0 else point_value - 1]
                    if np.isnan(lower_lim):
                        point_ranges[point_value] = [[valid_scores[0], threshold]]
                    else:
                        point_ranges[point_value] = [[lower_lim, threshold]]
            else:
                # Benign
                if abs(point_value) == max(config.point_values):
                    point_ranges[point_value] = [[threshold, valid_scores[-1]]]
                else:
                    upper_lim = conservative_thresholds[point_value - 1 if point_value < 0 else point_value + 1]
                    if np.isnan(upper_lim):
                        point_ranges[point_value] = [[threshold, valid_scores[-1]]]
                    else:
                        point_ranges[point_value] = [[threshold, upper_lim]]
    
    # Check for insufficient bootstrap coverage
    percent_no_evidence = {point: 0.0 for point in config.point_values + list(-1 * np.array(config.point_values))}
    
    # Enforce monotonicity (first pass)
    enforce_monotonicity_point_ranges(
        point_ranges,
        config.point_values,
        score_range[range_subset],
        scoreset_flipped=scoreset_flipped,
        liberal=config.liberal_monotonicity
    )

    # Extend to limits
    extend_points_to_xlims(
        point_ranges,
        config.point_values,
        score_range[range_subset],
        scoreset_flipped
    )

    # Enforce monotonicity again after extending (matches assign_points.py)
    enforce_monotonicity_point_ranges(
        point_ranges,
        config.point_values,
        score_range[range_subset],
        scoreset_flipped=scoreset_flipped,
        liberal=config.liberal_monotonicity
    )

    logger.info(f"  Final point ranges computed: {len([k for k, v in point_ranges.items() if v])} non-empty")
    
    # Serialize and return
    return serialize_dict({
        'prior': prior,
        'priors': fit_priors,
        'valid_bootstrap_seeds': valid_bootstrap_seeds,
        'valid_mask': valid_mask,
        'point_ranges': point_ranges,
        'score_range': score_range[range_subset],
        'log_lr_plus': log_lr_plus[:, range_subset],
        'log_fp': log_fp[:, range_subset],
        'log_fb': log_fb[:, range_subset],
        'C': C,
        'scoreset_flipped': scoreset_flipped,
        'n_valid_fits': len(fits),
    })