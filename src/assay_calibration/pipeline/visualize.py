"""
Visualization and calibration result generation
"""
import sys
import os
import json
import signal
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from joblib import Parallel, delayed
import logging

from ..fit_utils.fit import calculate_score_ranges, thresholds_from_prior, calculate_score_ranges_dual
from ..fit_utils.cfusn import density_utils
from ..fit_utils.point_ranges import (
    enforce_monotonicity_point_ranges,
    prior_equation_2c,
    prior_invalid,
    get_fit_prior,
    compute_lr_filtered_pathogenic_mask,
    get_bootstrap_score_ranges,
    get_bootstrap_score_ranges_dual,
    remove_insufficient_bootstrap_converage_points,
    check_thresholds_reached,
    extend_points_to_xlims,
    compute_single_fit_log_densities,
    compute_pathomechanism_lr_curves,
    resolve_pathomechanism_anchor,
    _compute_log_fp_only,
)
from ..data_utils.dataset import Scoreset, BasicScoreset
from ..fit_utils.utils import serialize_dict
import logging as _logging
_logging.getLogger("matplotlib").setLevel(_logging.ERROR)
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
import matplotlib as _mpl
_mpl.use('Agg')
import matplotlib.pyplot as plt
from ..plot_utils.utils import (
    plot_scoreset_best_config, sample_density, add_thresholds, log_thresholds_with_ylim_pad,
)

from .config import PipelineConfig
from .utils import load_dataset_from_df

_precomputed_fits_cache: Dict[str, Dict] = {}


def load_precomputed_fits(fits_path: str, dataset_name: str) -> Dict:
    """
    Load precomputed bootstrap fits from gzipped JSON.

    The expected format (matching assign_points.py) is::

        {dataset_name: {bootstrap_key: {"2c": {...}, "3c": {...}}, ...}}

    Each fit entry has ``fit``, ``val_ll``, etc.
    Returns the inner dict: ``{bootstrap_key: {"2c": fit, "3c": fit}}``.

    The full decompressed JSON (every dataset, not just `dataset_name`) is
    cached in-process keyed by `fits_path` -- callers that fetch several
    datasets from the same fits file in one session (e.g. an analysis
    notebook building more than one Scoreset off the same PRECOMPUTED_FITS)
    would otherwise re-decompress and re-parse the whole file on every call.
    """
    import gzip

    if fits_path not in _precomputed_fits_cache:
        with gzip.open(fits_path, 'rt', encoding='utf-8') as f:
            _precomputed_fits_cache[fits_path] = json.load(f)
    all_results = _precomputed_fits_cache[fits_path]

    # Two supported formats:
    #   (a) {dataset_name: {seed: {"2c": ..., "3c": ...}}}
    #   (b) flat {seed: {"2c": ..., "3c": ...}}  (as written by save_results)
    if dataset_name in all_results:
        return all_results[dataset_name]

    # Detect flat format: top-level keys look like bootstrap seeds (digit strings)
    # and their values look like component dicts.
    keys = list(all_results.keys())
    if keys and all(str(k).lstrip('-').isdigit() for k in keys):
        return all_results

    available = keys[:10]
    raise KeyError(
        f"Dataset '{dataset_name}' not found in precomputed fits. "
        f"Available: {available}{'...' if len(keys) > 10 else ''}"
    )


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
        sep = "\t" if config.dataset_csv.endswith((".tsv", ".tsv.gz")) else ","
        df = pd.read_csv(config.dataset_csv, sep=sep)
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

        if calibration is None:
            continue

        results[component_key] = calibration

        # Generate visualization in a forked child so a hung C renderer can be
        # killed from the parent.  signal.SIGALRM cannot interrupt C extensions
        # (e.g. matplotlib's Agg backend), so fork + waitpid is the only reliable
        # timeout mechanism on Linux.
        output_dir = Path(config.output_dir)
        figure_path = output_dir / f"{config.dataset_name}_{component_key}_visualization.png"

        child_pid = os.fork()
        if child_pid == 0:
            try:
                fig = plot_scoreset_best_config(
                    dataset=config.dataset_name,
                    scoreset=scoreset,
                    indv_summary=calibration,
                    fits=fits,
                    score_range=np.asarray(calibration['score_range']),
                    config=f"({config.benign_method})",
                    n_c=component_key,
                    n_samples=len([s for s in scoreset.samples]),
                    relax=False,
                    flipped=calibration.get('scoreset_flipped', False)
                )
                fig.savefig(figure_path, dpi=150)
                os._exit(0)
            except Exception:
                os._exit(1)
        else:
            import time
            start = time.monotonic()
            exit_status = None
            while time.monotonic() - start < 600:
                pid, st = os.waitpid(child_pid, os.WNOHANG)
                if pid != 0:
                    exit_status = st
                    break
                time.sleep(1)
            if exit_status is None:
                os.kill(child_pid, signal.SIGKILL)
                os.waitpid(child_pid, 0)
                logger.warning(f"  Visualization timed out after 600s, skipping")
            elif os.WIFEXITED(exit_status) and os.WEXITSTATUS(exit_status) == 0:
                logger.info(f"  Saved visualization: {figure_path}")
            else:
                logger.error(f"  Visualization process failed")
    
    return results

def process_component_fits(
    fits: List[Dict],
    scoreset: Scoreset,
    config: PipelineConfig,
    n_c: int,
    logger: logging.Logger,
    bootstrap_seeds: np.ndarray = None,
    precomputed_log_fp_all: np.ndarray = None,
    return_log_fp_all: bool = False,
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
        # Extra samples beyond the standard four are tracked in _sample_assignments
        # for variant-table labelling but are not used in calibration — skip them.
    
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

    # Adjust raw column indices to fit-weight indices.
    # Both Scoreset and BasicScoreset keep all 4 standard columns in
    # _sample_assignments (empty ones included), but the sample_assignments
    # property and the fit weights only cover non-empty columns.  Shift each
    # index down by the number of lower-indexed standard columns that are empty.
    if benign_idx is None:
        if gnomad_idx is not None:
            gnomad_idx -= 1
        if synonymous_idx is not None:
            synonymous_idx -= 1
    if pathogenic_idx is None:
        if gnomad_idx is not None:
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

    # Save pre-filter snapshot for log_fp sharing with the benign combo
    all_fits_prefilter = list(fits)
    n_fits_prefilter = len(fits)

    # Debug: sample names, counts, and mean scores per sample
    if config.debug:
        logger.info("  [DEBUG] Sample detection:")
        for i, name in enumerate(scoreset.sample_names):
            count = scoreset.sample_counts[i]
            if count > 0:
                mean_sc = np.mean(scoreset.scores[scoreset._sample_assignments[:, i]])
                logger.info(f"    col {i}: '{name}' — {count} variants, mean score = {mean_sc:.4f}")
            else:
                logger.info(f"    col {i}: '{name}' — 0 variants (empty)")
        logger.info(f"    isinstance(BasicScoreset): {isinstance(scoreset, BasicScoreset)}")
        logger.info(f"    benign_method: {config.benign_method}")

    filter_flag = getattr(config, "filter_pathogenic_sample_by_lr", False)
    if filter_flag:
        if config.use_2c_equation and n_c == 2:
            raise ValueError(
                "filter_pathogenic_sample_by_lr is not supported with use_2c_equation "
                "(n_c=2 equation path never calls get_fit_prior, so the filtered "
                "pathogenic weights it computes would have no effect)."
            )
        if pathogenic_idx is None:
            logger.warning("  filter_pathogenic_sample_by_lr has no effect (no pathogenic "
                            "sample for this dataset/combo); computing prior normally")
            filter_flag = False
        if config.manual_prior is not None:
            logger.warning("  filter_pathogenic_sample_by_lr is ignored because manual_prior is set")

    # EXPERIMENTAL, opt-in alternative to filter_flag: pathomechanism-corrected
    # f_D (see get_fit_prior / compute_pathomechanism_pathogenic_density[_masked]
    # in fit_utils/point_ranges.py) feeding a genuinely separate
    # pathogenic-direction (PS3) prior+LR+ pair (rho, LR_D), paired with a
    # mechanism-agnostic benign-direction (BS3) prior+LR+ pair (pi, LR_PB)
    # that always stays raw/uncorrected -- see PipelineConfig.pathomechanism_method's
    # docstring for the rho/pi derivation and why this pairing (not e.g. pi
    # with the mechanism-specific curve) is the mathematically consistent one.
    # Needs a pathogenic-labeled sample; anchors against benign/synonymous
    # when present (PN/standard mode) or raw gnomAD directly (PU-only mode).
    # Mutually exclusive with filter_flag (already enforced at
    # PipelineConfig construction, re-checked here for callers that mutate
    # config post-construction).
    pathomechanism_method = getattr(config, "pathomechanism_method", None)
    gamma_flag = pathomechanism_method is not None
    if gamma_flag:
        if filter_flag:
            raise ValueError(
                "pathomechanism_method and filter_pathogenic_sample_by_lr are mutually "
                "exclusive -- disable one."
            )
        if config.use_2c_equation and n_c == 2:
            raise ValueError(
                "pathomechanism_method is not supported with use_2c_equation "
                "(n_c=2 equation path never calls get_fit_prior)."
            )
        if getattr(config, "acmg_mapping_method", "tavtigian") == "acmg_bayes":
            raise NotImplementedError(
                "pathomechanism_method is not yet supported with "
                "acmg_mapping_method='acmg_bayes' (no dual-prior sibling for "
                "calculate_classification_ranges); use another acmg_mapping_method "
                "or set pathomechanism_method=None."
            )
        if pathogenic_idx is None:
            logger.warning("  pathomechanism_method has no effect (no pathogenic-labeled "
                            "sample for this dataset/combo); computing prior normally")
            gamma_flag = False
            pathomechanism_method = None
        if config.manual_prior is not None:
            logger.warning("  pathomechanism_method is ignored because manual_prior is set")

    # True unmixing (PU/NU) mode, per get_fit_prior's own mode selection --
    # pathogenic XOR benign/synonymous present. Used below to flag prior
    # instability specific to that estimator (its saturation floor).
    unmixing_mode = not (
        pathogenic_idx is not None and (benign_idx is not None or synonymous_idx is not None)
    )
    prior_unstable = False
    prior_pathogenic_unstable = False  # rho instability -- only meaningful when gamma_flag
    gamma_hat = np.nan
    gamma_unstable = False
    pct_pathogenic_rows_kept = None  # LR-filter row-level retention (filter_flag only)
    pct_bootstraps_kept = None  # fraction of bootstrap fits used for LR+/prior computation

    if config.manual_prior is not None:
        # Use manually specified prior — skip estimation. Both directions
        # collapse to the same manual value (dual pathomechanism curves never
        # trigger below since gamma_flag stays whatever it resolved to above,
        # but this whole branch bypasses get_fit_prior/curve-building
        # entirely, so there's nothing for it to act on).
        prior = config.manual_prior
        prior_pathogenic = config.manual_prior
        fit_priors = np.full(len(fits), prior)
        fit_priors_pathogenic = fit_priors
        fit_gammas = np.full(len(fits), np.nan)
        valid_bootstrap_seeds = bootstrap_seeds  # no filtering
        valid_mask = np.ones(len(fits), dtype=bool)
        pct_bootstraps_kept = 1.0
        logger.info(f"  Using manual prior: {prior:.6f}")
    else:
        # Pre-extract arrays so loky memory-maps them rather than pickling
        # the full Scoreset object once per task.
        scores_arr = np.asarray(scoreset.scores)
        sa_arr = np.asarray(scoreset.sample_assignments)
        n_cores = config.n_jobs if config.n_jobs > 0 else (os.cpu_count() or 1)

        pathogenic_row_mask = None
        if filter_flag:
            pathogenic_row_mask, n_labeled, n_kept = compute_lr_filtered_pathogenic_mask(
                all_fits_prefilter, scores_arr, sa_arr,
                config.benign_method, pathogenic_idx, benign_idx, synonymous_idx,
                gnomad_idx=gnomad_idx,
                percentile=config.pathogenic_percentile,
                n_jobs=n_cores,
            )
            logger.info(f"  filter_pathogenic_sample_by_lr: kept {n_kept}/{n_labeled} "
                        f"pathogenic-labeled variants (percentile={config.pathogenic_percentile})")
            pct_pathogenic_rows_kept = (n_kept / n_labeled) if n_labeled > 0 else None
            if n_kept == 0:
                logger.warning("  filter_pathogenic_sample_by_lr: 0 variants survived filtering; "
                                "falling back to unfiltered pathogenic sample for this combo")
                pathogenic_row_mask = None
                prior_unstable = True

        # Compute priors - PASS ALL INDICES
        used_get_fit_prior = not config.use_2c_equation or n_c != 2
        if used_get_fit_prior:
            def _run_get_fit_prior(method):
                return Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
                    delayed(get_fit_prior)(
                        fit, scores_arr, config.benign_method,
                        pathogenic_idx=pathogenic_idx,
                        benign_idx=benign_idx,
                        gnomad_idx=gnomad_idx,
                        synonymous_idx=synonymous_idx,
                        sample_assignments=sa_arr,
                        pathogenic_row_mask=pathogenic_row_mask,
                        pathomechanism_method=method,
                    )
                    for fit in fits
                )

            # Benign-direction / mechanism-agnostic prior (pi): always the raw,
            # uncorrected estimate -- this is today's existing single-prior
            # computation, unchanged in meaning regardless of pathomechanism_method
            # (see PipelineConfig.pathomechanism_method's docstring).
            prior_gamma_results_benign = _run_get_fit_prior(None)
            fit_priors_benign = np.array([r[0] for r in prior_gamma_results_benign])
            fit_gammas_benign = np.array([r[1] for r in prior_gamma_results_benign])  # always nan

            if gamma_flag:
                # Pathogenic-direction / mechanism-specific prior (rho).
                prior_gamma_results_path = _run_get_fit_prior(pathomechanism_method)
                fit_priors_pathogenic = np.array([r[0] for r in prior_gamma_results_path])
                fit_gammas = np.array([r[1] for r in prior_gamma_results_path])

                # pathomechanism_method discards (rather than silently falls back
                # on) any individual bootstrap fit whose excess/mask is degenerate
                # everywhere (see compute_pathomechanism_pathogenic_density[_masked]).
                # If that wipes out every fit for this combo, there's no
                # pathogenic-direction correction to apply at all -- fall back to
                # the benign-direction's already-computed raw estimate for rho too
                # (degrades both directions to the raw baseline for this combo)
                # rather than discarding it outright.
                if not np.any(np.isfinite(fit_priors_pathogenic)):
                    logger.warning(
                        "  pathomechanism_method: 0/%d fits had any assay-relevant "
                        "excess/mask (degenerate on every fit); falling back to raw "
                        "pathogenic density for the pathogenic-direction prior on "
                        "this combo",
                        n_fits_prefilter,
                    )
                    fit_priors_pathogenic = fit_priors_benign.copy()
                    fit_gammas = fit_gammas_benign.copy()
                    prior_unstable = True
                    prior_pathogenic_unstable = True
                    gamma_unstable = True
            else:
                fit_priors_pathogenic = fit_priors_benign
                fit_gammas = fit_gammas_benign

            # 'fit_priors'/'prior' stay benign-direction (pi) throughout this
            # function for backward compatibility -- see PipelineConfig.pathomechanism_method.
            fit_priors = fit_priors_benign
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
            fit_gammas = np.full(len(fits), np.nan)
            fit_priors_pathogenic = fit_priors  # 2c-equation path never supports pathomechanism_method

        # Filter invalid priors. A fit only contributes downstream if BOTH
        # directions produced a valid prior for it (when gamma_flag is off,
        # fit_priors_pathogenic is the exact same array as fit_priors, so this
        # intersection reduces to the original single-array condition with no
        # behavior change). Fits invalid in only one direction are dropped
        # entirely rather than partially retained -- simpler and lower-risk
        # than tracking two independently-filtered fit sets through the rest
        # of this function, at the cost of a small, documented efficiency loss.
        valid_mask = ~(
            np.isnan(fit_priors) | (fit_priors <= 0) | (fit_priors >= 1) |
            np.isnan(fit_priors_pathogenic) | (fit_priors_pathogenic <= 0) | (fit_priors_pathogenic >= 1)
        )
        fit_priors = fit_priors[valid_mask]
        fit_priors_pathogenic = fit_priors_pathogenic[valid_mask]
        fit_gammas = fit_gammas[valid_mask]
        fits = np.array(fits)[valid_mask].tolist()
        valid_bootstrap_seeds = bootstrap_seeds[valid_mask] if bootstrap_seeds is not None else np.array([])

        logger.info(f"  Valid priors: {len(fit_priors)}")
        pct_bootstraps_kept = (len(fit_priors) / n_fits_prefilter) if n_fits_prefilter > 0 else None
        logger.info(f"  pct_bootstraps_kept (for LR+/prior computation): {pct_bootstraps_kept:.4f}"
                    if pct_bootstraps_kept is not None else "  pct_bootstraps_kept: n/a (0 fits)")

        if len(fits) == 0:
            logger.warning(f"  No valid priors; skipping {component_key}")
            return None

        # Compute prior(s). 'prior' (pi, benign-direction) is unchanged in
        # meaning/name for backward compatibility; 'prior_pathogenic' (rho) is
        # new, only meaningful when gamma_flag.
        prior = np.nanmedian(fit_priors)
        prior_pathogenic = np.nanmedian(fit_priors_pathogenic) if gamma_flag else np.nan
        logger.info(f"  Prior (benign-direction, pi): {prior:.6f}")
        if gamma_flag:
            logger.info(f"  Prior (pathogenic-direction, rho): {prior_pathogenic:.6f}")

        # PU/NU unmixing floors saturating-low fits at 0.01 (get_fit_prior)
        # rather than discarding them (see its mode-gated saturation handling).
        # If the median itself lands exactly on that floor, the majority of
        # bootstrap fits saturated -- the estimate is a floor artifact, not a
        # real prior.
        if used_get_fit_prior and unmixing_mode and np.isclose(prior, 0.01):
            prior_unstable = True
        if used_get_fit_prior and unmixing_mode and gamma_flag and np.isclose(prior_pathogenic, 0.01):
            prior_pathogenic_unstable = True

        # gamma_hat: median assay-mechanistic-coverage estimate across bootstrap
        # fits (mirrors how `prior` itself is a nanmedian over fit_priors).
        # Unstable when no bootstrap fit produced a finite gamma at all -- no
        # arbitrary threshold here, just "did the EM produce anything usable".
        if gamma_flag:
            finite_gammas = fit_gammas[np.isfinite(fit_gammas)]
            if len(finite_gammas) > 0:
                gamma_hat = float(np.nanmedian(finite_gammas))
                logger.info(
                    f"  PLP_frac_pathomechanism_measured: {gamma_hat:.6f} (estimated "
                    "fraction of PLP-labeled variants with the disease mechanism this "
                    "assay measures)"
                )
            else:
                gamma_unstable = True
                logger.warning("  PLP_frac_pathomechanism_measured: no finite gamma "
                               "estimates across bootstrap fits")

    # Indices into all_fits_prefilter that survived prior filtering
    valid_original_indices = np.where(valid_mask)[0]

    # Setup score range
    observed_scores = scoreset.scores[scoreset._sample_assignments.any(1)]
    score_range = np.linspace(*np.percentile(observed_scores, [0, 100]), config.score_range_points)

    # Build per-fit precomputed log_fp array (shape: n_fits_prefilter × score_range_points)
    # when pathogenic samples exist.  Two cases:
    #   1. Caller passed precomputed_log_fp_all — reuse it directly (benign combo path).
    #   2. return_log_fp_all=True — compute upfront so the result can be handed to the
    #      matching benign combo (avg combo path).
    # When neither applies (normal single-combo usage) log_fp_prefilter stays None and
    # compute_single_fit_log_densities computes log_fp internally as before.
    log_fp_prefilter = None
    n_cores = config.n_jobs if config.n_jobs > 0 else (os.cpu_count() or 1)
    if pathogenic_idx is not None:
        if precomputed_log_fp_all is not None:
            log_fp_prefilter = np.asarray(precomputed_log_fp_all)
        elif return_log_fp_all:
            log_fp_prefilter = np.array(Parallel(
                n_jobs=min(n_fits_prefilter, n_cores), verbose=0
            )(
                delayed(_compute_log_fp_only)(fit, score_range, pathogenic_idx)
                for fit in all_fits_prefilter
            ))

    acmg_mapping_method = getattr(config, "acmg_mapping_method", "tavtigian")

    log_lr_pathogenic = None  # set below only when gamma_flag; stays None otherwise
    C_pathogenic = np.array([np.nan, np.nan])

    if gamma_flag:
        # Two-LR/two-prior pathomechanism scheme: pathogenic-direction (PS3)
        # curves/ranges from (rho, LR_D = log_fD - log_fN_pathogenic);
        # benign-direction (BS3) from (pi, LR_PB = log_fP_raw - log_fN_benign)
        # -- see compute_pathomechanism_lr_curves, get_bootstrap_score_ranges_dual.
        # PN mode: log_fN_benign == log_fN_pathogenic (no unmixing, one shared
        # anchor). PU mode: population is a mixture itself, so each direction
        # is unmixed separately against its own native (prior, density) pair
        # (pi/f_P_raw for benign, rho/f_D for pathogenic) -- sharing one raw,
        # unpurified anchor between both (this function's earlier behavior)
        # understated both LRs; confirmed on PTEN_Mighell_2018_clinvar_2018.
        anchors_is_pu = [
            resolve_pathomechanism_anchor(
                fit['fit']['weights'], config.benign_method, benign_idx, synonymous_idx, gnomad_idx
            )
            for fit in fits
        ]
        results_dual_curves = Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
            delayed(compute_pathomechanism_lr_curves)(
                fit, score_range, fit['fit']['weights'][pathogenic_idx], w_N_anchor,
                pathomechanism_method, is_pu=is_pu,
                prior_pathogenic=fit_priors_pathogenic[i], prior_benign=fit_priors_benign[i],
            )
            for i, (fit, (w_N_anchor, is_pu)) in enumerate(zip(fits, anchors_is_pu))
        )

        results = Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
            delayed(get_bootstrap_score_ranges_dual)(
                fitIdx, fit, log_fD, log_fN_benign, log_fP_raw, score_range,
                fit_priors_pathogenic, fit_priors_benign, config.point_values,
                acmg_mapping_method=acmg_mapping_method, log_fN_pathogenic=log_fN_pathogenic,
            )
            for fitIdx, (fit, (log_fD, log_fN_benign, log_fP_raw, log_fN_pathogenic))
            in enumerate(zip(fits, results_dual_curves))
        )

        log_fD_arr = np.full((len(fits), len(score_range)), np.nan)
        log_fN_arr = np.full((len(fits), len(score_range)), np.nan)
        log_fN_pathogenic_arr = np.full((len(fits), len(score_range)), np.nan)
        log_fPraw_arr = np.full((len(fits), len(score_range)), np.nan)
        ranges_pathogenic, ranges_benign = [], []
        Cs_pathogenic, Cs = [], []  # 'Cs' stays benign-direction for backward-compat aggregation below

        for (fitIdx, log_fD_local, log_fN_local, log_fP_raw_local, ranges_p, ranges_b, C_p, C_b,
             log_fN_pathogenic_local) in results:
            log_fD_arr[fitIdx] = log_fD_local
            log_fN_arr[fitIdx] = log_fN_local
            log_fN_pathogenic_arr[fitIdx] = log_fN_pathogenic_local
            log_fPraw_arr[fitIdx] = log_fP_raw_local
            ranges_pathogenic.append({key: np.array(value).reshape(-1) for key, value in ranges_p.items()})
            ranges_benign.append({key: np.array(value).reshape(-1) for key, value in ranges_b.items()})
            Cs_pathogenic.append(C_p)
            Cs.append(C_b)

        log_lr_pathogenic = log_fD_arr - log_fN_pathogenic_arr
        # Backward-compat aliases: 'log_fp'/'log_fb'/'log_lr_plus' (and the
        # result dict's 'prior'/'C') always represent the benign-direction
        # (pi, mechanism-agnostic) curve, matching 'prior'=pi.
        log_fp = log_fPraw_arr
        log_fb = log_fN_arr
        log_lr_plus = log_fp - log_fb

        Cs_valid_p = [c for c in Cs_pathogenic if isinstance(c, (int, float)) and not (isinstance(c, float) and np.isnan(c))]
        if Cs_valid_p:
            C_pathogenic = np.array([
                np.nanpercentile(Cs_valid_p, config.pathogenic_percentile),
                np.nanpercentile(Cs_valid_p, 100 - config.pathogenic_percentile),
            ])
    else:
        # Legacy single-prior/single-LR path, unchanged.
        results_fpfb = Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
            delayed(compute_single_fit_log_densities)(
                fit, prior, score_range, config.benign_method,
                pathogenic_idx=pathogenic_idx,
                benign_idx=benign_idx,
                gnomad_idx=gnomad_idx,
                synonymous_idx=synonymous_idx,
                precomputed_log_fp=(
                    log_fp_prefilter[valid_original_indices[j]]
                    if log_fp_prefilter is not None else None
                ),
            )
            for j, (fit, prior) in enumerate(zip(fits, fit_priors))
        )

        # Unpack results
        _log_fp = np.array([r[0] if r[0] is not None else np.full(len(score_range), np.nan)
                            for r in results_fpfb])
        _log_fb = np.array([r[1] if r[1] is not None else np.full(len(score_range), np.nan)
                            for r in results_fpfb])

        # Get bootstrap score ranges
        results = Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
            delayed(get_bootstrap_score_ranges)(
                fitIdx, fit, fp, fb, score_range, fit_priors, config.point_values,
                acmg_mapping_method=acmg_mapping_method,
                acmg_bayes_targets=getattr(config, "acmg_bayes_targets", None),
                acmg_bayes_floor_at_neutral=getattr(
                    config, "acmg_bayes_floor_at_neutral", False),
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

    if config.debug:
        logger.info(f"  [DEBUG] LR+ stats: shape={log_lr_plus.shape}")
        logger.info(f"    log_lr_plus[0] range: [{np.nanmin(log_lr_plus[0]):.4f}, {np.nanmax(log_lr_plus[0]):.4f}]")
        logger.info(f"    Prior: {prior:.6f}, fit_priors range: [{np.nanmin(fit_priors):.6f}, {np.nanmax(fit_priors):.6f}]")

    # Compute C range (piecewise mapping returns C=None — skip those)
    Cs_valid = [c for c in Cs if isinstance(c, (int, float)) and not (isinstance(c, float) and np.isnan(c))]
    if Cs_valid:
        C = np.array([
            np.nanpercentile(Cs_valid, config.pathogenic_percentile),
            np.nanpercentile(Cs_valid, 100 - config.pathogenic_percentile),
        ])
    else:
        C = np.array([np.nan, np.nan])

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

    if config.debug:
        logger.info(f"  [DEBUG] Flip detection: path_mean={path_mean_score:.4f}, ben_mean={ben_mean_score:.4f}")
        logger.info(f"    path_mean > ben_mean => flipped={path_mean_score > ben_mean_score}")
        if config.scoreset_flipped_override is not None:
            logger.info(f"    Override applied: {config.scoreset_flipped_override}")
        # Example fit params
        fit0 = fits[0]['fit']
        logger.info(f"  [DEBUG] Example fit (bootstrap 0):")
        logger.info(f"    weights: {fit0['weights']}")
        logger.info(f"    component_params: {fit0.get('component_params', 'N/A')}")
    
    # Compute point ranges - exclude score positions where ALL fits are NaN
    nan_counts = np.isnan(log_lr_plus).sum(0)
    range_subset = nan_counts < log_lr_plus.shape[0]
    
    if config.use_median_prior:
        # Use median prior for unified thresholds
        logger.info(f"  Using median prior for unified thresholds (method={acmg_mapping_method})")
        if gamma_flag:
            # acmg_mapping_method == "acmg_bayes" is already rejected together
            # with gamma_flag at flag-validation time above (no dual-prior
            # sibling for calculate_classification_ranges yet), so only the
            # tavtigian path is reachable here.
            point_ranges_pathogenic, point_ranges_benign, C_p_dual, C_b_dual = calculate_score_ranges_dual(
                np.nanpercentile(log_lr_pathogenic[:, range_subset], config.pathogenic_percentile, axis=0),
                np.nanpercentile(log_lr_plus[:, range_subset], 100 - config.pathogenic_percentile, axis=0),
                prior_pathogenic, prior,
                score_range[range_subset],
                config.point_values,
                acmg_mapping_method=acmg_mapping_method,
            )
            point_ranges = {**point_ranges_pathogenic, **point_ranges_benign}
            if not np.isfinite(prior_pathogenic) or prior_pathogenic <= 0 or prior_pathogenic >= 1:
                for point in point_ranges_pathogenic:
                    point_ranges[point] = []
            if prior <= 0 or prior >= 1:
                for point in point_ranges_benign:
                    point_ranges[point] = []
            C = C_b_dual  # backward-compat: 'C' stays benign-direction, matches 'prior'=pi
            C_pathogenic = C_p_dual
        elif acmg_mapping_method == "acmg_bayes":
            from ..fit_utils.fit import calculate_classification_ranges
            cls_p, cls_b, lr_thresholds = calculate_classification_ranges(
                np.nanpercentile(log_lr_plus[:, range_subset], config.pathogenic_percentile, axis=0),
                np.nanpercentile(log_lr_plus[:, range_subset], 100 - config.pathogenic_percentile, axis=0),
                prior,
                score_range[range_subset],
                targets=getattr(config, "acmg_bayes_targets", None),
                floor_at_neutral=getattr(config, "acmg_bayes_floor_at_neutral", False),
            )
            point_ranges = {**cls_p, **cls_b}
            if prior <= 0 or prior >= 1:
                for k in point_ranges:
                    point_ranges[k] = []
            C = lr_thresholds  # dict of LR+ thresholds, not an integer C*
        else:
            point_ranges_pathogenic, point_ranges_benign, C = calculate_score_ranges(
                np.nanpercentile(log_lr_plus[:, range_subset], config.pathogenic_percentile, axis=0),
                np.nanpercentile(log_lr_plus[:, range_subset], 100 - config.pathogenic_percentile, axis=0),
                prior,
                score_range[range_subset],
                config.point_values,
                acmg_mapping_method=acmg_mapping_method,
            )
            point_ranges = {**point_ranges_pathogenic, **point_ranges_benign}
            if prior <= 0 or prior >= 1:
                for point in point_ranges:
                    point_ranges[point] = []
    else:
        # Use pathogenic_percentile-th conservative thresholds
        logger.info(f"  Using {config.pathogenic_percentile:g}th percentile conservative thresholds")
        p_xaxis_5percentile_conservative = (
            config.pathogenic_percentile if not scoreset_flipped else 100 - config.pathogenic_percentile
        )
        b_xaxis_5percentile_conservative = (
            100 - config.pathogenic_percentile if not scoreset_flipped else config.pathogenic_percentile
        )
        
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

    # Monotonicity enforcement and extend-to-xlims are integer-point-specific.
    # Continuous uses string-keyed ranges but still needs the outermost P and B
    # ranges extended to ±inf so scores beyond the observed range are classified.
    if acmg_mapping_method == "acmg_bayes":
        # Extend outermost pathogenic range (P, or LP if P is empty) to ±inf.
        for label in ("P", "LP"):
            ranges = point_ranges.get(label)
            if ranges:
                if not scoreset_flipped:
                    ranges[0][0] = -np.inf   # low score = pathogenic
                else:
                    ranges[-1][-1] = np.inf  # high score = pathogenic
                break
        # Extend outermost benign range (B, or LB if B is empty) to ±inf.
        for label in ("B", "LB"):
            ranges = point_ranges.get(label)
            if ranges:
                if not scoreset_flipped:
                    ranges[-1][-1] = np.inf  # high score = benign
                else:
                    ranges[0][0] = -np.inf   # low score = benign
                break
        if config.debug:
            logger.info("  [DEBUG] Continuous classification ranges:")
            for k, v in point_ranges.items():
                logger.info(f"    {k}: {v}")
    else:
        if config.debug:
            logger.info("  [DEBUG] Point ranges BEFORE enforce_monotonicity:")
            for k in sorted(point_ranges.keys(), key=lambda x: -x):
                logger.info(f"    {k:+d}: {point_ranges[k]}")

        if getattr(config, "postprocess_point_ranges", True):
            # Enforce monotonicity (first pass)
            enforce_monotonicity_point_ranges(
                point_ranges,
                config.point_values,
                score_range[range_subset],
                scoreset_flipped=scoreset_flipped,
                liberal=config.liberal_monotonicity
            )

            if config.debug:
                logger.info("  [DEBUG] Point ranges AFTER first enforce_monotonicity:")
                for k in sorted(point_ranges.keys(), key=lambda x: -x):
                    logger.info(f"    {k:+d}: {point_ranges[k]}")

            # Extend to limits
            extend_points_to_xlims(
                point_ranges,
                config.point_values,
                score_range[range_subset],
                scoreset_flipped,
                inf=True
            )

            if config.debug:
                logger.info("  [DEBUG] Point ranges AFTER extend_points_to_xlims:")
                for k in sorted(point_ranges.keys(), key=lambda x: -x):
                    logger.info(f"    {k:+d}: {point_ranges[k]}")

            # Enforce monotonicity again after extending (matches assign_points.py)
            enforce_monotonicity_point_ranges(
                point_ranges,
                config.point_values,
                score_range[range_subset],
                scoreset_flipped=scoreset_flipped,
                liberal=config.liberal_monotonicity
            )

            if config.debug:
                logger.info("  [DEBUG] Point ranges AFTER second enforce_monotonicity (final):")
                for k in sorted(point_ranges.keys(), key=lambda x: -x):
                    logger.info(f"    {k:+d}: {point_ranges[k]}")
        elif config.debug:
            logger.info("  [DEBUG] --no-postprocess: skipping monotonicity enforcement and extend-to-xlims")

    logger.info(f"  Final ranges computed: {len([k for k, v in point_ranges.items() if v])} non-empty")
    
    # Serialize and return
    result = {
        # 'prior' (pi) and every legacy field below stay benign-direction /
        # mechanism-agnostic for backward compatibility -- see
        # PipelineConfig.pathomechanism_method's docstring. 'pathomechanism_prior'
        # (rho) is the new pathogenic-direction/mechanism-specific prior,
        # populated only when pathomechanism_method is active.
        'prior': prior,
        'prior_unstable': bool(prior_unstable),
        'pathomechanism_prior': prior_pathogenic if gamma_flag else None,
        'pathomechanism_prior_unstable': bool(prior_pathogenic_unstable) if gamma_flag else False,
        'pathomechanism_method': pathomechanism_method,
        # PLP_frac_pathomechanism_measured (a.k.a. gamma_hat): estimated fraction
        # of PLP-labeled variants whose disease mechanism this assay actually
        # measures (a 0-1 fraction, median over bootstrap fits).
        'PLP_frac_pathomechanism_measured': gamma_hat,
        'PLP_frac_pathomechanism_measured_description': (
            'estimated fraction (0-1) of PLP-labeled variants with the disease '
            'mechanism this assay measures'),
        'PLP_frac_pathomechanism_measured_unstable': bool(gamma_unstable),
        'PLP_frac_pathomechanism_measured_per_fit': fit_gammas,
        'priors': fit_priors,
        'priors_pathogenic': fit_priors_pathogenic if gamma_flag else None,
        'pct_bootstraps_kept': pct_bootstraps_kept,
        'pct_pathogenic_rows_kept': pct_pathogenic_rows_kept,
        'valid_bootstrap_seeds': valid_bootstrap_seeds,
        'valid_mask': valid_mask,
        'point_ranges': point_ranges,
        'score_range': score_range[range_subset],
        'log_lr_plus': log_lr_plus[:, range_subset],
        'log_lr_pathogenic': (
            log_lr_pathogenic[:, range_subset] if log_lr_pathogenic is not None else None
        ),
        'log_fp': log_fp[:, range_subset],
        'log_fb': log_fb[:, range_subset],
        'C': C,
        'C_pathogenic': C_pathogenic,
        'scoreset_flipped': scoreset_flipped,
        'n_valid_fits': len(fits),
        'acmg_mapping_method': acmg_mapping_method,
    }
    # Include pre-filter log_fp so callers can share it with a sibling benign combo.
    # Only populated when return_log_fp_all=True and precomputed_log_fp_all was not given.
    if return_log_fp_all and precomputed_log_fp_all is None and log_fp_prefilter is not None:
        result['log_fp_all_fits'] = log_fp_prefilter
    return serialize_dict(result)


def _plot_confusion_panel(
    ax: "plt.Axes",
    m: Dict,
    is_selected: bool,
    best_metrics: Optional[Dict] = None,
) -> None:
    """Draw a 2×3 confusion matrix styled like plot_aggregate_confusion_matrices.

    Blue/Gray/Red column-based colormaps, row-normalized intensity.
    Each metric (MCC, Acc, Cov) is bolded independently where it is best across configs.
    """
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    if best_metrics is None:
        best_metrics = {}

    data = [
        [m["TN"],     m["IR_BLB"], m["FP"]],   # B/LB row
        [m["FN"],     m["IR_PLP"], m["TP"]],    # P/LP row
    ]
    col_labels = ["Benign", "Indet.", "Pathogenic"]
    row_labels = ["B/LB", "P/LP"]

    blue_cmap = LinearSegmentedColormap.from_list(
        "blue_g", ["#F0F8FC", "#99C8DC", "#7AB5D1", "#4B91A6", "#2E6B7E"])
    red_cmap  = LinearSegmentedColormap.from_list(
        "red_g",  ["#FCF0F2", "#E6B1B8", "#D68F99", "#B85C6B", "#943744"])
    gray_cmap = LinearSegmentedColormap.from_list(
        "gray_g", ["#F5F5F5", "#CCCCCC", "#999999", "#666666"])
    col_cmaps = [blue_cmap, gray_cmap, red_cmap]

    for r in range(2):
        row_max = max(data[r]) or 1
        for c in range(3):
            val = int(data[r][c])
            norm = val / row_max
            color = col_cmaps[c](norm)
            ax.add_patch(mpatches.Rectangle(
                (c, r), 1, 1,
                facecolor=color, edgecolor="white", linewidth=2.5,
            ))
            txt_color = "black" if c == 1 else ("white" if norm > 0.45 else "black")
            ax.text(c + 0.5, r + 0.5, str(val),
                    ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_facecolor("#F9F9F9")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if is_selected:
        ax.patch.set_edgecolor("#006400")
        ax.patch.set_linewidth(2.5)
        ax.set_facecolor("#F0FFF0")

    # Per-metric labels — bold independently where best
    metric_items = [
        (f"MCC={m['MCC']:.3f}",              best_metrics.get("MCC", False)),
        (f"Acc={m.get('accuracy', 0.0):.3f}", best_metrics.get("accuracy", False)),
        (f"Cov={m['coverage']:.3f}",          best_metrics.get("coverage", False)),
    ]
    for i, (text, is_best) in enumerate(metric_items):
        ax.text(
            (i + 0.5) / 3, 1.14, text,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=7.5,
            fontweight="bold" if is_best else "normal",
        )



def plot_all_configs_comparison(
    dataset_name: str,
    scoreset,
    fits_by_nc: Dict,
    calibrations: Dict,
    selected_config: tuple,
    metrics: Optional[Dict] = None,
) -> plt.Figure:
    """
    Grid comparison of all (2c/3c) × (avg/benign) calibration configs.

    Layout (3 rows per n_c):
      density row : [sample density + component fits] … | blank | blank
      avg row     : [density + avg threshold shading] … | avg LR+ | avg confusion
      benign row  : [density + benign shading]        … | benign LR+ | benign confusion

    The selected config panels are highlighted with a green border and bold title.
    """
    import seaborn as sns

    BENIGN_METHODS = ["avg", "benign"]
    n_components = [nc for nc in ["2c", "3c"]
                    if nc in fits_by_nc and any(f"{nc}_{b}" in calibrations for b in BENIGN_METHODS)]
    if not n_components:
        return None

    n_samples = scoreset.n_samples
    n_cols = n_samples + 2          # sample columns + LR+ + confusion
    n_rows_per_nc = 3               # density, avg, benign
    n_rows = len(n_components) * n_rows_per_nc
    hist_colors = ["#CA7682", "#1D7AAB", "#A0A0A0", "#6BAA75"]

    height_ratios = []
    for _ in n_components:
        height_ratios.extend([1.4, 1.0, 1.0])
    total_height = sum(height_ratios) * 3.5
    fig_width = 4.5 * n_cols

    fig, ax = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, total_height),
        squeeze=False,
        gridspec_kw={"hspace": 0.55, "wspace": 0.38, "height_ratios": height_ratios},
    )

    # Compute which config is best per metric (may tie → multiple bolded)
    _best_metrics: Dict[str, Dict[str, bool]] = {}
    if metrics:
        best_mcc = max((v.get("MCC", -999) for v in metrics.values()), default=-999)
        best_acc = max((v.get("accuracy", -999) for v in metrics.values()), default=-999)
        best_cov = max((v.get("coverage", -999) for v in metrics.values()), default=-999)
        for ck, mv in metrics.items():
            _best_metrics[ck] = {
                "MCC":      mv.get("MCC", -999) == best_mcc,
                "accuracy": mv.get("accuracy", -999) == best_acc,
                "coverage": mv.get("coverage", -999) == best_cov,
            }

    for nc_idx, n_c_str in enumerate(n_components):
        row_density = nc_idx * n_rows_per_nc
        row_avg     = nc_idx * n_rows_per_nc + 1
        row_benign  = nc_idx * n_rows_per_nc + 2
        fits = fits_by_nc.get(n_c_str, [])

        # Score range reference (first available calibration for this n_c)
        score_range_ref = None
        for b in BENIGN_METHODS:
            calib = calibrations.get(f"{n_c_str}_{b}")
            if calib is not None:
                score_range_ref = np.asarray(calib["score_range"])
                break

        # Blank right columns in density row
        ax[row_density, n_samples].axis("off")
        ax[row_density, n_samples + 1].axis("off")

        # --- Row 0: density + component fits (shared across methods) ---
        xlim = None
        if score_range_ref is not None and fits:
            s_col = 0  # filtered column index into sample_assignments / ax
            for s_raw in range(len(scoreset.sample_counts)):
                if scoreset.sample_counts[s_raw] == 0:
                    continue
                ax_fit = ax[row_density, s_col]
                scores_s = scoreset.scores[scoreset.sample_assignments[:, s_col].astype(bool)]
                n = len(scores_s)
                q1, q3 = np.percentile(scores_s, [25, 75])
                iqr = q3 - q1
                fd_w = 2 * iqr * n ** (-1 / 3) if iqr > 0 else 0
                w = scores_s.max() - scores_s.min()
                bins = max(min(100, int(w / fd_w)) if fd_w > 0 else 50, 10)
                sns.histplot(scores_s, bins=bins, stat="density", ax=ax_fit,
                             alpha=0.5, color=hist_colors[s_raw % len(hist_colors)])
                density = sample_density(score_range_ref, fits, s_col)
                for c in range(density.shape[1]):
                    d = np.nanpercentile(density[:, c, :], [5, 50, 95], axis=0)
                    ax_fit.plot(score_range_ref, d[1], color=f"C{c}", linestyle="--",
                                label=f"Comp {c+1}", linewidth=1.2)
                d_tot = np.nansum(density, axis=1)
                d_pct = np.nanpercentile(d_tot, [5, 50, 95], axis=0)
                ax_fit.plot(score_range_ref, d_pct[1], color="black", alpha=0.5)
                ax_fit.fill_between(score_range_ref, d_pct[0], d_pct[2], color="gray", alpha=0.2)
                ax_fit.set_title(f"{n_c_str}: {scoreset.sample_names[s_raw]}\n(n={n:,d})", fontsize=8)
                ax_fit.legend(fontsize=6)
                ax_fit.grid(linewidth=0.5, alpha=0.3)
                ax_fit.set_xlabel("Score", fontsize=7)
                s_col += 1
            xlim = ax[row_density, 0].get_xlim()

        # Fallback xlim from score_range_ref when fits were absent (viz-only)
        if xlim is None and score_range_ref is not None:
            xlim = (float(score_range_ref[0]), float(score_range_ref[-1]))

        # --- Rows 1 & 2: per-method (avg, benign) ---
        for benign_method, row_m in [("avg", row_avg), ("benign", row_benign)]:
            comp_key = f"{n_c_str}_{benign_method}"
            calib = calibrations.get(comp_key)
            is_selected = (n_c_str, benign_method) == selected_config
            sel_color = "#006400" if is_selected else "black"

            if calib is None:
                for col in range(n_cols):
                    ax[row_m, col].axis("off")
                continue

            score_range = np.asarray(calib["score_range"])
            if "log_lr_pct" in calib:
                llr = np.asarray(calib["log_lr_pct"])
            else:
                llr = np.nanpercentile(np.asarray(calib["log_lr_plus"]), [5, 50, 95], axis=0)
            prior = float(calib["prior"])

            def _sel_spine(a):
                if is_selected:
                    for sp in a.spines.values():
                        sp.set_edgecolor("#006400")
                        sp.set_linewidth(2.0)

            # Per-sample: point ranges (mirrors plot_scoreset_best_config Row 1)
            point_ranges = sorted([(int(k), v) for k, v in calib["point_ranges"].items()])
            point_values = [pr[0] for pr in point_ranges]
            # Always use the calibration's full score_range for xlim so that
            # pathogenic/benign ranges are never clipped by sample 0's histogram xlim
            if score_range is not None and len(score_range) > 1:
                eff_xlim = (float(score_range[0]), float(score_range[-1]))
            elif xlim is not None:
                eff_xlim = xlim
            else:
                eff_xlim = None

            s_col = 0  # filtered column index into sample_assignments / ax
            for s_raw in range(len(scoreset.sample_counts)):
                if scoreset.sample_counts[s_raw] == 0:
                    continue
                ax_m = ax[row_m, s_col]

                for pointIdx, (pointVal, scoreRanges) in enumerate(point_ranges):
                    for sr in scoreRanges:
                        x0 = eff_xlim[0] if (eff_xlim and np.isneginf(sr[0])) else sr[0]
                        x1 = eff_xlim[1] if (eff_xlim and np.isposinf(sr[1])) else sr[1]
                        if eff_xlim:
                            x0 = max(x0, eff_xlim[0])
                            x1 = min(x1, eff_xlim[1])
                        ax_m.plot([x0, x1], [pointIdx, pointIdx],
                                  color="red" if pointVal > 0 else "blue",
                                  linestyle="-", alpha=0.7, linewidth=2)

                ax_m.set_ylim(-1, len(point_values))
                ax_m.set_yticks(range(len(point_values)))
                ax_m.set_yticklabels([f"{v:+d}" if v != 0 else "0" for v in point_values], fontsize=7)
                ax_m.set_xlabel("Score", fontsize=7)
                ax_m.set_ylabel("Points", fontsize=7)
                if eff_xlim:
                    ax_m.set_xlim(eff_xlim)
                ax_m.grid(linewidth=0.5, alpha=0.3)
                mark = " ✓" if is_selected else ""
                ax_m.set_title(f"Point Assignments [{benign_method}{mark}]",
                               fontsize=7.5, fontweight="bold" if is_selected else "normal",
                               color=sel_color)
                _sel_spine(ax_m)
                s_col += 1

            # LR+ panel
            ax_lr = ax[row_m, n_samples]
            for curve, color, lbl in zip(llr, ["red", "black", "blue"], ["5th", "Med", "95th"]):
                ax_lr.plot(score_range, curve, color=color, label=lbl, linewidth=1.5)
            pt_vals = sorted({abs(int(k)) for k in calib["point_ranges"]})
            if 0 < prior < 1 and pt_vals:
                try:
                    tauP, tauB, ylim_top, ylim_bottom = log_thresholds_with_ylim_pad(prior, pt_vals)
                    add_thresholds(tauP, tauB, ax_lr)
                    ax_lr.set_ylim([ylim_bottom, ylim_top])
                except Exception:
                    pass
            if "priors_pct" in calib:
                pp = calib["priors_pct"]
            elif "priors" in calib:
                priors_arr = np.atleast_1d(calib["priors"])
                pp = np.percentile(priors_arr, [5, 50, 95]) if len(priors_arr) > 1 else [prior] * 3
            else:
                pp = [prior, prior, prior]
            ax_lr.set_title(f"{n_c_str} {benign_method} LR+\nprior {pp[1]:.3f} ({pp[0]:.3f}–{pp[2]:.3f})",
                            fontsize=8, fontweight="bold" if is_selected else "normal", color=sel_color)
            ax_lr.legend(fontsize=6, loc="best")
            ax_lr.grid(linewidth=0.5, alpha=0.3)
            ax_lr.set_xlabel("Score", fontsize=7)
            ax_lr.set_ylabel("Log LR+", fontsize=7)
            if xlim:
                ax_lr.set_xlim(xlim)
            _sel_spine(ax_lr)

            # Confusion panel
            ax_conf = ax[row_m, n_samples + 1]
            m_data = metrics.get(comp_key) if metrics else None
            if m_data is not None:
                _plot_confusion_panel(ax_conf, m_data, is_selected, _best_metrics.get(comp_key, {}))
            else:
                ax_conf.axis("off")

    sel_str = f"{selected_config[0]} {selected_config[1]}" if selected_config else "none"
    fig.suptitle(f"{dataset_name}  —  all configs (selected: {sel_str})",
                 fontsize=13, y=1.002)
    return fig
