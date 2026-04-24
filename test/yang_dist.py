from joblib import Parallel, delayed
import pickle
import numpy as np
from scipy import stats as sps
from scipy.interpolate import interp1d


def load_splits_dict(splits_path="/data/ross/assay_calibration/dataset_splits_recovered.pkl"):
    # Load splits
    with open(splits_path, 'rb') as f:
        dataset_to_splits = pickle.load(f)
    return dataset_to_splits

def compute_yang_distance_p2(empirical_samples, learned_cdf_func, n_points=1000):
    """
    Compute Yang distance (p=2) between empirical and learned CDFs.
    D_2(F, G) = sqrt(integral_0^1 |F^{-1}(u) - G^{-1}(u)|^2 du)
    """
    if len(empirical_samples) == 0:
        return np.nan
    
    sorted_samples = np.sort(empirical_samples)
    u_grid = np.linspace(0, 1, n_points)
    
    # Empirical quantiles
    empirical_quantiles = np.quantile(sorted_samples, u_grid)
    
    # Learned quantiles - invert the CDF
    x_min = sorted_samples.min() - 3 * sorted_samples.std()
    x_max = sorted_samples.max() + 3 * sorted_samples.std()
    x_grid = np.linspace(x_min, x_max, 10000)
    cdf_values = learned_cdf_func(x_grid)
    
    # Handle non-monotonic CDFs
    unique_indices = np.where(np.diff(cdf_values) > 1e-10)[0]
    if len(unique_indices) < 2:
        return np.nan
    
    try:
        learned_quantile_func = interp1d(
            cdf_values[unique_indices], x_grid[unique_indices],
            bounds_error=False, fill_value='extrapolate'
        )
        learned_quantiles = learned_quantile_func(u_grid)
    except:
        return np.nan
    
    # Yang distance
    squared_diffs = (empirical_quantiles - learned_quantiles) ** 2
    yang_distance = np.sqrt(np.trapz(squared_diffs, u_grid))
    
    return yang_distance


def _process_single_bootstrap(boot_idx, dataset, n_c, fits_dict, dataset_splits, sample_map):
    """Worker function to process a single bootstrap iteration."""
    
    if int(boot_idx) not in dataset_splits:
        return {k: np.nan for k in sample_map.keys()}
    
    # Get train split
    boot_dict = dataset_splits[int(boot_idx)]
    train_obs = boot_dict["train_observations"]
    train_assign = boot_dict["train_sample_assignments"]
    
    # Calculate score range for this bootstrap (for normalization)
    score_range = train_obs.max() - train_obs.min()
    if score_range == 0:
        score_range = 1.0  # Avoid division by zero
    
    # Separate by sample
    train_by_sample = {k: [] for k in sample_map.keys()}
    
    for obs, assign in zip(train_obs, train_assign):
        class_idx = np.where(assign)[0][0]  # One-hot to index
        sample_key = list(sample_map.keys())[class_idx]
        train_by_sample[sample_key].append(obs)
    
    # Convert to arrays
    for k in train_by_sample:
        train_by_sample[k] = np.array(train_by_sample[k])
    
    # Get fit parameters
    if n_c in fits_dict[boot_idx]:
        fit = fits_dict[boot_idx][n_c]
    else:
        fit = fits_dict[boot_idx]
    params = fit['fit']['component_params']
    weights = fit['fit']['weights']
    
    # Compute Yang distance for each sample
    boot_distances = {}
    for sample_key, sample_idx in sample_map.items():
        if len(train_by_sample[sample_key]) == 0:
            boot_distances[sample_key] = np.nan
            continue
        
        # Learned CDF for this sample
        def learned_cdf(x):
            cdfs = np.array([
                w * sps.skewnorm.cdf(x, a, loc, scale)
                for (a, loc, scale), w in zip(params, weights[sample_idx])
            ])
            return cdfs.sum(axis=0)
        
        # Compute raw Yang distance
        distance = compute_yang_distance_p2(
            train_by_sample[sample_key], learned_cdf
        )
        
        # Normalize by this bootstrap's score range
        normalized_distance = distance / score_range if not np.isnan(distance) else np.nan
        boot_distances[sample_key] = normalized_distance
    
    return boot_distances


def compute_bootstrap_yang_distances_parallel(dataset, n_c, fits, scoreset, dataset_to_splits, n_jobs=-1, normalize=True):
    """
    Compute Yang distances for all bootstrap iterations across all samples (parallelized).
    
    Parameters:
    -----------
    dataset : str
        Dataset name
    n_c : str
        Component model ('2c' or '3c')
    fits : dict
        Dictionary of fits indexed by bootstrap iteration
    scoreset : object
        Scoreset object (not used but kept for API compatibility)
    dataset_to_splits : dict
        Dictionary of train/val splits per dataset and bootstrap
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    normalize : bool
        If True, normalize distances by each bootstrap's score range (default: True)
        This makes distances unitless and comparable across bootstraps
    
    Returns:
    --------
    dict with keys ['pathogenic', 'benign', 'gnomad', 'synonymous'],
    each containing array of normalized distances per bootstrap (unitless, roughly 0-1 range)
    """
    
    if dataset not in dataset_to_splits:
        raise ValueError(f"Dataset {dataset} not found in splits")
    
    sample_map = {'pathogenic': 0, 'benign': 1, 'gnomad': 2, 'synonymous': 3}
    fits_dict = {i: fits[i] for i in range(len(fits))} if isinstance(fits, list) else fits
    boot_indices = sorted(fits_dict.keys())
    
    norm_str = "normalized " if normalize else ""
    print(f"{dataset}: computing {norm_str}Yang distances for {len(boot_indices)} bootstrap iterations on {n_jobs if n_jobs > 0 else 'all'} cores...")
    
    # Parallel processing
    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_process_single_bootstrap)(
                boot_idx, dataset, n_c, fits_dict, dataset_to_splits[dataset], sample_map
            ) for boot_idx in boot_indices
        )
    else:
        results = [_process_single_bootstrap(
                boot_idx, dataset, n_c, fits_dict, dataset_to_splits[dataset], sample_map
            ) for boot_idx in boot_indices]
    
    # Reorganize results
    yang_distances = {k: [] for k in sample_map.keys()}
    for boot_result in results:
        for sample_key in sample_map.keys():
            yang_distances[sample_key].append(boot_result[sample_key])
    
    # Convert to arrays
    yang_distances = {k: np.array(v) for k, v in yang_distances.items()}
    
    return yang_distances