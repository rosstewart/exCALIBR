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
    
    # Create logger
    logger = logging.getLogger('calibration_pipeline')
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
    logger: logging.Logger
):
    """Save calibration results and optionally bootstrap fits"""
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save calibration JSON for each component
    for component_key, calibration in results.items():
        # Prepare calibration output (compact version with only essential data)
        calibration_compact = {
            'dataset': config.dataset_name,
            'component': component_key,
            'prior': calibration['prior'],
            'point_ranges': calibration['point_ranges'],
            'scoreset_flipped': calibration.get('scoreset_flipped', False),
            'n_valid_fits': calibration.get('n_valid_fits', 0),
            'config': {
                'n_bootstraps': config.n_bootstraps,
                'num_fits_per_bootstrap': config.num_fits_per_bootstrap,
                'use_median_prior': config.use_median_prior,
                'use_2c_equation': config.use_2c_equation,
                'benign_method': config.benign_method,
                'clinvar_release': config.clinvar_release,
            }
        }
        
        # Save compact JSON
        json_file = output_dir / f"{config.dataset_name}_{component_key}_calibration.json"
        with open(json_file, 'w') as f:
            json.dump(serialize_dict(calibration_compact), f, indent=2)
        
        logger.info(f"  Saved calibration: {json_file}")
        
        # Save full results (if requested)
        if config.save_bootstrap_fits:
            # Convert numpy arrays to lists for JSON compatibility
            calibration_full = serialize_dict(calibration)
            
            # Save as compressed JSON
            json_gz_file = output_dir / f"{config.dataset_name}_{component_key}_full.json.gz"
            with gzip.open(json_gz_file, 'wt', encoding='utf-8') as f:
                json.dump(serialize_dict(calibration_full), f, indent=2)
            
            logger.info(f"  Saved full results: {json_gz_file}")
    
    # Save bootstrap fits (if requested)
    if bootstrap_results is not None:
        pkl_file = output_dir / f"{config.dataset_name}_bootstrap_fits.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(bootstrap_results, f)
        
        logger.info(f"  Saved bootstrap fits: {pkl_file}")
        logger.info(f"    Size: {pkl_file.stat().st_size / 1e6:.1f} MB")

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
        
    if "scores" in df.columns and "sample_assignments" in df.columns:
        return BasicScoreset(df["scores"], df["sample_assignments"])
        
    return Scoreset(
        df,
        clinvar_release=config.clinvar_release,
        min_clinvar_star=config.min_clinvar_star,
        population_type=config.population_type,
    )
