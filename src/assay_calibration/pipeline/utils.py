"""
Utility functions for the calibration pipeline
"""
import os
import sys
import json
import pickle
import gzip
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from .config import PipelineConfig

from ..fit_utils.utils import serialize_dict
from ..data_utils.dataset import Scoreset, BasicScoreset

def setup_logging(output_dir: str, dataset_name: str) -> logging.Logger:
    """Setup logging to both file and console"""
    
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    log_file = log_dir / f"{dataset_name}_pipeline.log"
    
    # Use a per-call name so concurrent threads get independent loggers
    # and don't accumulate each other's handlers.
    logger = logging.getLogger(f'calibration_pipeline.{dataset_name}')
    logger.setLevel(logging.INFO)
    
    # Fix both problems
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_results(
    results: Dict,
    bootstrap_results: Optional[Dict],
    config: PipelineConfig,
    logger: logging.Logger,
    selected_k: Optional[int] = None,
):
    """Save calibration results and optionally bootstrap fits.

    Always saves per-component:
      - ``<name>_<comp>_calibration.json`` — compact point-range output
        matching the format of ``example/BRCA1_Findlay_2018.json``
      - ``<name>_<comp>_lr_values.json.gz`` — likelihood-ratio interpolation
        over the full score range (``score_range`` + ``log_lr_plus`` matrix)

    When *bootstrap_results* is not ``None`` (i.e., fits were computed fresh,
    not loaded from precomputed), also saves:
      - ``<name>_bootstrap_fits.json.gz`` — gzipped JSON in the same format
        as the precomputed-fits input (``{seed: {"2c": {...}, "3c": {...}}}``).
    """

    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Per-component outputs
    # ------------------------------------------------------------------
    for component_key, calibration in results.items():
        # 1. Compact calibration JSON (matches example/BRCA1_Findlay_2018.json)
        calibration_compact = {
            'prior': calibration['prior'],
            'prior_unstable': calibration.get('prior_unstable', False),
            # pathomechanism_prior (rho = P(pathogenic AND mechanism-detectable))
            # is the pathogenic-direction/mechanism-specific prior, paired with
            # the pathogenic-direction LR+ curve (log_lr_pathogenic). 'prior'
            # above stays the benign-direction/mechanism-agnostic prior (pi),
            # unchanged in meaning. Both None/False when pathomechanism_method
            # is off for this run -- see PipelineConfig.pathomechanism_method.
            'pathomechanism_prior': calibration.get('pathomechanism_prior', None),
            'pathomechanism_prior_unstable': calibration.get('pathomechanism_prior_unstable', False),
            'pathomechanism_method': calibration.get('pathomechanism_method', None),
            'PLP_frac_pathomechanism_measured': calibration.get(
                'PLP_frac_pathomechanism_measured', None),
            'PLP_frac_pathomechanism_measured_unstable': calibration.get(
                'PLP_frac_pathomechanism_measured_unstable', False),
            'pct_bootstraps_kept': calibration.get('pct_bootstraps_kept', None),
            'pct_pathogenic_rows_kept': calibration.get('pct_pathogenic_rows_kept', None),
            'point_ranges': calibration['point_ranges'],
            'dataset': config.dataset_name,
            'relax': True,
            'liberal_monotonicity': config.liberal_monotonicity,
            'n_c': component_key,
            'benign_method': config.benign_method,
            'clinvar_2018': config.clinvar_release == '2018',
            'scoreset_flipped': calibration.get('scoreset_flipped', False),
            'uncalibratable_reason': None,
            # component_key may be bare ("3c") or compound ("3c_avg") depending
            # on the caller -- compare only the n_c portion so this stays
            # correct either way.
            'model_selected': (component_key.split('_', 1)[0] == f"{selected_k}c") if selected_k is not None else None,
        }

        json_file = output_dir / f"{config.dataset_name}_{component_key}_calibration.json"
        with open(json_file, 'w') as f:
            json.dump(serialize_dict(calibration_compact), f, indent=2)
        logger.info(f"  Saved calibration: {json_file}")

        # 2. LR interpolation over score range (always saved)
        # Store precomputed percentile curves (5th/50th/95th) rather than all
        # 1000 bootstrap rows — reduces file size ~300× with no downstream loss.
        if 'score_range' in calibration and 'log_lr_plus' in calibration:
            import numpy as _np
            llr = _np.asarray(calibration['log_lr_plus'])
            pct = _np.nanpercentile(llr, [5, 50, 95], axis=0)
            priors_arr = _np.asarray(calibration.get('priors', calibration['prior']))
            priors_pct = _np.nanpercentile(priors_arr, [5, 50, 95]).tolist()
            lr_data = {
                'dataset': config.dataset_name,
                'n_c': component_key,
                'score_range': calibration['score_range'],
                # Legacy keys, always benign-direction/mechanism-agnostic
                # (pi, LR_PB) for backward compatibility with existing
                # consumers -- unchanged in meaning even when
                # pathomechanism_method is active.
                'log_lr_plus_p5': pct[0].tolist(),
                'log_lr_plus_p50': pct[1].tolist(),
                'log_lr_plus_p95': pct[2].tolist(),
                'priors_pct': priors_pct,
                'prior': calibration['prior'],
                'log_lr_benign_p5': pct[0].tolist(),
                'log_lr_benign_p50': pct[1].tolist(),
                'log_lr_benign_p95': pct[2].tolist(),
                'prior_benign_pct': priors_pct,
                'scoreset_flipped': calibration.get('scoreset_flipped', False),
            }
            # Pathogenic-direction (rho, LR_D) curve+prior percentiles, only
            # when pathomechanism_method produced one for this combo.
            if calibration.get('log_lr_pathogenic') is not None:
                llr_p = _np.asarray(calibration['log_lr_pathogenic'])
                pct_p = _np.nanpercentile(llr_p, [5, 50, 95], axis=0)
                rho_arr = _np.asarray(calibration.get('priors_pathogenic', calibration.get('pathomechanism_prior')))
                lr_data['log_lr_pathogenic_p5'] = pct_p[0].tolist()
                lr_data['log_lr_pathogenic_p50'] = pct_p[1].tolist()
                lr_data['log_lr_pathogenic_p95'] = pct_p[2].tolist()
                lr_data['prior_pathogenic_pct'] = _np.nanpercentile(rho_arr, [5, 50, 95]).tolist()

            lr_file = output_dir / f"{config.dataset_name}_{component_key}_lr_values.json.gz"
            with gzip.open(lr_file, 'wt', encoding='utf-8') as f:
                json.dump(serialize_dict(lr_data), f)
            lr_size_mb = lr_file.stat().st_size / 1e6
            logger.info(f"  Saved LR values: {lr_file} ({lr_size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # Bootstrap fits (gzipped JSON, only when computed fresh)
    # ------------------------------------------------------------------
    if bootstrap_results is not None:
        fits_file = output_dir / f"{config.dataset_name}_bootstrap_fits.json.gz"
        with gzip.open(fits_file, 'wt', encoding='utf-8') as f:
            json.dump(serialize_dict(bootstrap_results), f)

        fits_size_mb = fits_file.stat().st_size / 1e6
        logger.info(f"  Saved bootstrap fits: {fits_file} ({fits_size_mb:.1f} MB)")

def collect_slurm_results(jobs_dir: Path, config: PipelineConfig) -> Dict:
    """Collect results from completed SLURM jobs"""
    
    import glob
    
    results_dir = jobs_dir.parent
    result_files = glob.glob(str(results_dir / "results_array_*.pkl"))
    
    if not result_files:
        raise FileNotFoundError(f"No SLURM results found in {results_dir}")
    
    print(f"Found {len(result_files)} result files")
    
    # Aggregate all results
    all_results = {}
    for result_file in sorted(result_files):
        with open(result_file, 'rb') as f:
            array_results = pickle.load(f)
        
        for result in array_results:
            bootstrap_seed = result['bootstrap_seed']
            all_results[bootstrap_seed] = {
                k: v for k, v in result.items()
                if k != 'bootstrap_seed'
            }
    
    print(f"Collected {len(all_results)} bootstrap results")
    
    return all_results

def validate_dataset(df, dataset_name: str) -> bool:
    """Validate that dataset has required columns and samples"""
    
    required_cols = ['score', 'sample']  # Adjust based on your data format
    
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'")
            return False
    
    # Check for required samples
    samples = df['sample'].unique()
    
    required_samples = ['Pathogenic/Likely Pathogenic', 'gnomAD']
    has_benign = 'Benign/Likely Benign' in samples
    has_synonymous = 'Synonymous' in samples
    
    if not has_benign and not has_synonymous:
        print("Error: Must have either 'Benign/Likely Benign' or 'Synonymous' sample")
        return False
    
    for sample in required_samples:
        if sample not in samples:
            print(f"Error: Missing required sample '{sample}'")
            return False
    
    return True

def load_dataset_from_df(df, config):
    # Filter to specific dataset if needed
    if "Dataset" in df.columns:
        df = df[df["Dataset"] == config.dataset_name]

    # IGVF-formatted datasets have "auth_reported_score" and full metadata
    # for Scoreset preprocessing. Everything else is a BasicScoreset.
    if "auth_reported_score" in df.columns:
        return Scoreset(
            df,
            clinvar_release=config.clinvar_release,
            min_clinvar_star=config.min_clinvar_star,
            population_type=config.population_type,
            synonymous_exclusive=getattr(config, "synonymous_exclusive", False),
        )

    if "score" not in df.columns or "sample_assignments" not in df.columns:
        raise KeyError(
            f"Dataset must have either 'auth_reported_score' (IGVF format) or "
            f"'score' + 'sample_assignments' columns. Found: {list(df.columns)}"
        )

    kwargs = {}
    if config.sample_names is not None:
        kwargs["sample_names"] = config.sample_names
    return BasicScoreset(df["score"].values, df["sample_assignments"].values, **kwargs)
