"""
Yang distance (p=2) between empirical bootstrap-training data and the fitted
skew-normal mixture CDF — a bootstrap-level goodness-of-fit diagnostic.

Moved from test/yang_dist.py. The only change: ``load_splits_dict`` (which
read a multi-GB ``dataset_splits_recovered.pkl``) is replaced by
``build_splits_for_dataset``, which regenerates train/val splits on demand via
``Fit.generate_fit_jobs(bootstrap_seed=idx)`` — confirmed deterministic given
just (dataset, bootstrap_idx) and the original scoreset, so results are
identical without needing the pickle.
"""
from joblib import Parallel, delayed
import numpy as np
from scipy import stats as sps
from scipy.interpolate import interp1d


def build_splits_for_dataset(scoreset, n_c: int, bootstrap_indices) -> dict:
    """Regenerate {boot_idx: {"train_observations", "train_sample_assignments"}}
    for the given bootstrap indices, deterministically, from `scoreset`.

    Equivalent to looking up `dataset_to_splits[dataset]` in the legacy
    `dataset_splits_recovered.pkl`, but computed fresh — no pickle required.

    Deliberately serial: each generate_fit_jobs call is cheap on its own
    (~0.04s measured on BAP1_Waters_2024, so ~1000 calls is on the order of
    40s), but joblib-parallelizing this loop was measured to be *slower*
    (110s vs 12s for 300 calls on the same dataset) because each task has to
    re-pickle the whole Fit/Scoreset object to a worker process, which
    dwarfs the actual per-split compute cost. Don't parallelize this again
    without re-measuring on a case where the scoreset is large enough that
    the compute-per-task genuinely exceeds pickling overhead.
    """
    from src.assay_calibration.fit_utils.fit import Fit

    fitter = Fit(scoreset)
    splits = {}
    for idx in bootstrap_indices:
        jobs = fitter.generate_fit_jobs(
            component_range=[n_c], bootstrap_seed=int(idx),
            check_monotonic=True, num_fits=1,
        )
        if jobs:
            splits[int(idx)] = {
                "train_observations": jobs[0]["train_observations"],
                "train_sample_assignments": jobs[0]["train_sample_assignments"],
            }
    return splits


def compute_yang_distance_p2(empirical_samples, learned_cdf_func, n_points=1000, n_grid=10000):
    """
    Compute Yang distance (p=2) between empirical and learned CDFs.
    D_2(F, G) = sqrt(integral_0^1 |F^{-1}(u) - G^{-1}(u)|^2 du)

    `n_grid` is the resolution of the x-grid the learned CDF is evaluated on
    before being inverted via interpolation — the dominant per-call cost
    (n_grid skewnorm.cdf evaluations x n_components). The default (10000)
    matches the original diagnostic script; for a bootstrap-median summary
    statistic (as used by compute_bootstrap_yang_distances_parallel for
    excalibr_datasets.csv), a coarser grid (e.g. 2000) changes individual
    Yang-distance values negligibly relative to bootstrap-to-bootstrap
    variance, at a large constant-factor speedup.
    """
    if len(empirical_samples) == 0:
        return np.nan

    sorted_samples = np.sort(empirical_samples)
    u_grid = np.linspace(0, 1, n_points)

    empirical_quantiles = np.quantile(sorted_samples, u_grid)

    x_min = sorted_samples.min() - 3 * sorted_samples.std()
    x_max = sorted_samples.max() + 3 * sorted_samples.std()
    x_grid = np.linspace(x_min, x_max, n_grid)
    cdf_values = learned_cdf_func(x_grid)

    unique_indices = np.where(np.diff(cdf_values) > 1e-10)[0]
    if len(unique_indices) < 2:
        return np.nan

    try:
        learned_quantile_func = interp1d(
            cdf_values[unique_indices], x_grid[unique_indices],
            bounds_error=False, fill_value='extrapolate'
        )
        learned_quantiles = learned_quantile_func(u_grid)
    except Exception:
        return np.nan

    squared_diffs = (empirical_quantiles - learned_quantiles) ** 2
    yang_distance = np.sqrt(np.trapz(squared_diffs, u_grid))

    return yang_distance


def _resolve_sample_indices(scoreset) -> dict:
    """Map canonical Yang-distance sample keys -> the *compacted* column index
    used by train_sample_assignments/the fit's `weights`, or None if that
    category has zero variants / isn't present at all for this dataset.

    Critical: Scoreset.sample_assignments (what Fit.generate_fit_jobs actually
    uses -- see src/assay_calibration/data_utils/dataset.py) is
    ``self._sample_assignments[:, self.sample_counts > 0]`` -- a boolean mask
    that DROPS every zero-count category's column and compacts/reindexes the
    rest, rather than keeping a fixed-width [pathogenic, benign, gnomad,
    synonymous] layout with empty columns for absent categories. So the
    correct index for a present category is its position among only the
    *other present* categories, in original relative order -- not its
    position in the uncompacted scoreset.sample_names list, and DEFINITELY
    not a hardcoded [0,1,2,3]. Verified against TARDBP_Bolognesi_Faure_2019
    (benign and synonymous both zero-count): the real
    train_sample_assignments has only 2 columns, not 4. Under the old
    hardcoded map, 'benign' (index 1) would have silently pulled gnomAD's
    training data (whatever landed in the compacted index 1), and 'gnomad'
    (index 2) would have read past the end of the array, always returning an
    empty group -- so the two non-last-missing categories would misattribute
    or falsely null out entirely, not just be individually absent.
    """
    name_to_key = {
        "Pathogenic/Likely Pathogenic": "pathogenic",
        "Benign/Likely Benign": "benign",
        "gnomAD": "gnomad",
        "population": "gnomad",
        "Synonymous": "synonymous",
    }
    indices = {"pathogenic": None, "benign": None, "gnomad": None, "synonymous": None}
    sample_counts = getattr(scoreset, "sample_counts", None)
    compact_idx = 0
    for i, name in enumerate(scoreset.sample_names):
        present = sample_counts is None or sample_counts[i] > 0
        if not present:
            continue  # dropped from the compacted sample_assignments columns
        key = name_to_key.get(name)
        if key is not None:
            indices[key] = compact_idx
        compact_idx += 1
    return indices


def _process_single_bootstrap(
    boot_idx, dataset, n_c, fits_dict, dataset_splits, sample_indices, n_points=1000, n_grid=10000,
):
    """Worker function to process a single bootstrap iteration.

    `sample_indices` : {canonical_key: column_index_or_None} from
    _resolve_sample_indices -- column_index indexes both train_assign's
    columns and the fit's `weights` array (both keyed to the same scoreset
    column ordering), never a hardcoded canonical position.
    """
    all_keys = list(sample_indices.keys())

    if int(boot_idx) not in dataset_splits:
        return {k: np.nan for k in all_keys}

    boot_dict = dataset_splits[int(boot_idx)]
    train_obs = boot_dict["train_observations"]
    train_assign = boot_dict["train_sample_assignments"]

    score_range = train_obs.max() - train_obs.min()
    if score_range == 0:
        score_range = 1.0

    # train_assign is one-hot per row (each training observation belongs to
    # exactly one fit-time sample class) — argmax over the whole matrix at
    # once replaces what used to be a Python-level np.where(...)[0][0] call
    # per row, which dominated this function's cost at ~1000 bootstraps/call.
    class_idx = train_assign.argmax(axis=1)

    if n_c in fits_dict[boot_idx]:
        fit = fits_dict[boot_idx][n_c]
    else:
        fit = fits_dict[boot_idx]
    params = fit['fit']['component_params']
    weights = fit['fit']['weights']

    boot_distances = {}
    for sample_key, col_idx in sample_indices.items():
        if col_idx is None:
            boot_distances[sample_key] = np.nan
            continue

        train_for_sample = train_obs[class_idx == col_idx]
        if len(train_for_sample) == 0:
            boot_distances[sample_key] = np.nan
            continue

        def learned_cdf(x, col_idx=col_idx):
            cdfs = np.array([
                w * sps.skewnorm.cdf(x, a, loc, scale)
                for (a, loc, scale), w in zip(params, weights[col_idx])
            ])
            return cdfs.sum(axis=0)

        distance = compute_yang_distance_p2(
            train_for_sample, learned_cdf, n_points=n_points, n_grid=n_grid,
        )

        normalized_distance = distance / score_range if not np.isnan(distance) else np.nan
        boot_distances[sample_key] = normalized_distance

    return boot_distances


def compute_bootstrap_yang_distances_parallel(
    dataset, n_c, fits, scoreset, dataset_to_splits=None, n_jobs=-1, normalize=True,
    n_points=1000, n_grid=10000,
):
    """
    Compute Yang distances for all bootstrap iterations across all samples (parallelized).

    Parameters
    ----------
    dataset : str
        Dataset name
    n_c : str
        Component model ('2c' or '3c')
    fits : dict
        Dictionary of fits indexed by bootstrap iteration
    scoreset : object
        Scoreset used to (re)derive train/val splits when dataset_to_splits is
        not provided (see build_splits_for_dataset).
    dataset_to_splits : dict, optional
        Precomputed {dataset: {boot_idx: {...}}} splits (legacy pickle format).
        If None, splits are regenerated on demand from `scoreset` — deterministic,
        so results are identical either way.
    n_jobs : int
        Number of parallel jobs (-1 for all cores) for the per-bootstrap
        distance computation. Split regeneration (build_splits_for_dataset)
        is always serial — see its docstring for why parallelizing it is
        counterproductive.
    normalize : bool
        If True, normalize distances by each bootstrap's score range (default: True)
    n_points, n_grid : int
        Quantile-grid / learned-CDF-grid resolution passed to
        compute_yang_distance_p2 — see its docstring. Defaults match the
        original diagnostic script exactly; pass smaller values (e.g.
        n_grid=2000) for a large constant-factor speedup when only a
        bootstrap-median summary statistic is needed (as in
        analysis.build_dataset_summary.compute_yang_distances_all), not a
        pixel-identical figure. Always run over every bootstrap in `fits`
        (no subsampling) -- the whole point of computing this over ~1000
        bootstraps is to characterize the full distribution/variance of the
        goodness-of-fit statistic, not just its central tendency, so cutting
        the bootstrap count defeats the purpose even if only the median is
        reported downstream.

    Returns
    -------
    dict with keys ['pathogenic', 'benign', 'gnomad', 'synonymous'],
    each containing array of normalized distances per bootstrap (unitless, roughly 0-1 range)
    """
    sample_indices = _resolve_sample_indices(scoreset)
    missing = [k for k, v in sample_indices.items() if v is None]
    if missing:
        print(f"  {dataset}: no {', '.join(missing)} sample -- those Yang distances will be all-NaN")
    fits_dict = {i: fits[i] for i in range(len(fits))} if isinstance(fits, list) else fits
    boot_indices = sorted(fits_dict.keys())

    n_c_int = int(str(n_c).rstrip("c"))
    if dataset_to_splits is not None and dataset in dataset_to_splits:
        splits = dataset_to_splits[dataset]
    else:
        splits = build_splits_for_dataset(scoreset, n_c_int, boot_indices)

    norm_str = "normalized " if normalize else ""
    print(f"{dataset}: computing {norm_str}Yang distances for {len(boot_indices)} bootstrap iterations on {n_jobs if n_jobs > 0 else 'all'} cores...")

    if n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_process_single_bootstrap)(
                boot_idx, dataset, n_c, fits_dict, splits, sample_indices, n_points, n_grid,
            ) for boot_idx in boot_indices
        )
    else:
        results = [_process_single_bootstrap(
                boot_idx, dataset, n_c, fits_dict, splits, sample_indices, n_points, n_grid,
            ) for boot_idx in boot_indices]

    yang_distances = {k: [] for k in sample_indices.keys()}
    for boot_result in results:
        for sample_key in sample_indices.keys():
            yang_distances[sample_key].append(boot_result[sample_key])

    yang_distances = {k: np.array(v) for k, v in yang_distances.items()}

    return yang_distances
