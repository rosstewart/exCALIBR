"""
Fitting routines for skew-normal mixture models.

Supports univariate, restricted multivariate (q=1), and CFUSN (q>=1).

To use CFUSN, pass latent_q=2 (or any q>1) in kwargs to Fit.run() or
Fit.generate_fit_jobs(). This propagates to initialization and EM steps.

Key kwargs for CFUSN:
    latent_q : int (default 1) — latent dimension q
    n_mc_truncated : int (default 500) — MC samples for truncated MVN moments
"""

import sys
from pathlib import Path
import os
import json

sys.path.append(str(Path(__file__).resolve().parents[2]))
from assay_calibration.data_utils.dataset import Scoreset, BasicScoreset, MultiScoreset
from scipy.stats import skewnorm
import numpy as np
from typing import Tuple, List, Dict
from .evidence_thresholds import get_tavtigian_constant
import logging
import sys
from joblib import Parallel, delayed
from .cfusn.fit import single_fit, compute_sample_weights
from .cfusn.density_utils import (
    get_likelihood, get_sample_likelihood,
    msn_logpdf_alternate_missing, cfusn_logpdf_alternate_missing,
    mixture_pdf, log_joint_densities, _ensure_matrix_delta, get_q,
)
from .utils import serialize_dict
import time
from tqdm.auto import tqdm

logging.basicConfig()
logging.root.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


def tryToFit(observations, sample_indicators, num_components, constrained,
             init_method, init_constraint_adjustment, multivariate=False, **kwargs):
    try:
        fit_results = single_fit(
            observations, sample_indicators, num_components, constrained,
            init_method, init_constraint_adjustment, multivariate=multivariate, **kwargs
        )
        return fit_results
    except Exception as e:
        if kwargs.get("raise_on_error", False):
            raise
        import traceback
        import warnings
        warnings.warn(f"tryToFit failed: {e}\n{traceback.format_exc()}")
        return dict(
            component_params=[[] for _ in range(num_components)],
            weights=None,
            likelihoods=[-np.inf],
            xlims=None,
            times_submerged=[],
        )


def get_bootstrap_indices(dataset_size):
    indices = np.arange(dataset_size)
    train_indices = np.random.choice(indices, size=dataset_size, replace=True)
    test_indices = np.setdiff1d(indices, train_indices)
    return train_indices, test_indices


def sample_specific_bootstrap(sample_assignments, bootstrap_seed=None):
    train_indices = []
    eval_indices = []
    rng = np.random.RandomState(bootstrap_seed)
    for sample_num in range(sample_assignments.shape[1]):
        sample_indices = np.where(sample_assignments[:, sample_num])[0]
        if not len(sample_indices):
            continue
        sample_eval = []
        fails = 0
        if len(sample_indices) == 1:
            sample_train = sample_indices
            sample_eval = []
        else:
            sample_train = []
            while not len(sample_eval) and fails < 100:
                sample_train = rng.choice(sample_indices, size=len(sample_indices), replace=True)
                sample_eval = np.setdiff1d(sample_indices, sample_train)
                fails += 1
            if fails >= 100:
                raise ValueError("Failed to generate bootstrap split")
        train_indices.append(sample_train)
        if len(sample_eval):
            eval_indices.append(sample_eval)
    train_indices = np.concatenate(train_indices)
    eval_indices = np.concatenate(eval_indices)
    return train_indices, eval_indices


def pattern_stratified_bootstrap(observations, sample_assignments, bootstrap_seed=None):
    """Bootstrap stratified by (sample class × missingness pattern).

    For multivariate data with missing values, plain sample-specific
    bootstrapping lets the proportion of complete vs partial observations
    vary randomly across bootstraps.  That variation directly changes how
    much joint vs marginal information the EM sees, inflating variance in
    the fitted parameters and in downstream LR estimates.

    This function adds a second stratification level: within each sample
    class the variants are grouped by their missingness pattern (which
    dimensions are observed), and bootstrap resampling is done
    independently within each (sample, pattern) stratum.  Every bootstrap
    therefore has the same composition of complete/partial cases per
    sample class (up to ±1 due to integer rounding), isolating LR
    variance to genuine parameter uncertainty rather than data-composition
    fluctuations.

    Parameters
    ----------
    observations : (N, D) ndarray — may contain NaN
    sample_assignments : (N, S) bool array — one-hot per row
    bootstrap_seed : int or None

    Returns
    -------
    train_indices, eval_indices : 1-D int arrays
    """
    rng = np.random.RandomState(bootstrap_seed)
    N = observations.shape[0]

    # Missingness pattern for each variant: tuple of which dims are observed
    obs_mask = ~np.isnan(observations)
    patterns = [tuple(obs_mask[i].tolist()) for i in range(N)]

    train_indices = []
    eval_indices = []

    for sample_num in range(sample_assignments.shape[1]):
        sample_idx = np.where(sample_assignments[:, sample_num])[0]
        if not len(sample_idx):
            continue

        # Group within this sample by missingness pattern
        strata: dict = {}
        for idx in sample_idx:
            strata.setdefault(patterns[idx], []).append(idx)

        for group in strata.values():
            group = np.array(group)
            if len(group) == 1:
                train_indices.append(group)
                continue

            # Resample within stratum, require at least one OOB sample
            fails = 0
            while fails < 100:
                g_train = rng.choice(group, size=len(group), replace=True)
                g_eval = np.setdiff1d(group, g_train)
                if len(g_eval):
                    break
                fails += 1
            else:
                g_train = group  # last resort: use all, no eval from this stratum
                g_eval = np.array([], dtype=int)

            train_indices.append(g_train)
            if len(g_eval):
                eval_indices.append(g_eval)

    train_out = np.concatenate(train_indices)
    eval_out = np.concatenate(eval_indices) if eval_indices else np.array([], dtype=int)
    return train_out, eval_out


def _weighted_val_ll(val_observations, val_sample_assignments, params, weights, mv, fit_kwargs):
    """Validation log-likelihood with optional per-sample weighting.

    When sample_balance_beta=0 and sample_proportions=None (defaults), returns the
    standard per-observation average LL (existing behaviour, n-based).

    When sample_balance_beta > 0 or sample_proportions is given, computes the
    per-sample mean LL and combines them with the same weights used in the M-step,
    so that the val score reflects the same objective the EM optimised.

    Parameters
    ----------
    fit_kwargs : dict
        Expects sample_balance_beta (float) and/or sample_proportions (array-like).
    """
    if not fit_kwargs.get("weighted_val_ll", False):
        return get_likelihood(
            val_observations, val_sample_assignments, params, weights, multivariate=mv
        ) / len(val_sample_assignments)

    beta = float(fit_kwargs.get("sample_balance_beta", 0.0))
    proportions = fit_kwargs.get("sample_proportions", None)

    if proportions is None and beta == 0.0:
        return get_likelihood(
            val_observations, val_sample_assignments, params, weights, multivariate=mv
        ) / len(val_sample_assignments)

    N_samples = val_sample_assignments.shape[1]
    n_per_sample = val_sample_assignments.sum(axis=0).astype(float)

    sample_lls = get_sample_likelihood(
        val_observations, val_sample_assignments, params, weights, multivariate=mv
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        per_sample_mean_ll = np.where(n_per_sample > 0, sample_lls / n_per_sample, 0.0)

    if proportions is not None:
        w = np.asarray(proportions, dtype=float)
        if len(w) != N_samples:
            raise ValueError(
                f"sample_proportions length {len(w)} != N_samples {N_samples}"
            )
    else:
        n_ref = n_per_sample[n_per_sample > 0].min() if (n_per_sample > 0).any() else 1.0
        w = np.where(n_per_sample > 0, (n_ref / np.maximum(n_per_sample, 1.0)) ** beta, 0.0)

    w = np.where(n_per_sample > 0, w, 0.0)
    w_sum = w.sum()
    if w_sum == 0:
        return -np.inf
    w = w / w_sum
    return float((w * per_sample_mean_ll).sum())


def derive_fit_seed(master_seed, bootstrap_idx, num_components, fit_idx):
    """Derive an independent, collision-free seed for one (bootstrap, component-count,
    restart) triple from a single top-level master_seed.

    Uses SeedSequence rather than e.g. simple addition/multiplication of the inputs so
    that streams for different (bootstrap_idx, num_components, fit_idx) combinations
    don't collide or correlate. Returns None (→ unseeded) when master_seed is None,
    preserving historical behaviour for callers that don't opt into reproducibility.
    """
    if master_seed is None:
        return None
    ss = np.random.SeedSequence([master_seed, bootstrap_idx, num_components, fit_idx])
    return int(ss.generate_state(1)[0])


class Fit:
    def __init__(self, scoreset):
        """Accept Scoreset, BasicScoreset, or MultiScoreset."""
        self.scoreset = scoreset
        self.fit_result = {}
        self.multivariate = (
            isinstance(scoreset, MultiScoreset)
            or getattr(scoreset, 'is_multivariate', False)
            or (hasattr(scoreset, 'scores') and scoreset.scores.ndim == 2 and scoreset.scores.shape[1] > 1)
        )

    @classmethod
    def from_dict(cls, scoreset, fit_dict):
        model = MulticomponentCalibrationModel.from_params(
            fit_dict["skewness"], fit_dict["locs"],
            fit_dict["scales"], fit_dict["sample_weights"],
        )
        obj = cls(scoreset)
        obj.model = model
        obj._eval_metrics = fit_dict["eval_metrics"]
        return obj

    def run(self, component_range, **kwargs):
        """
        Run fits. For MultiScoreset, automatically routes to multivariate EM.

        For CFUSN, pass latent_q=2 (or higher) in kwargs.

        Thin wrapper around generate_fit_jobs(): job construction (one-hot
        sample assignment, bootstrap split, init-method selection, per-restart
        fit_seed derivation) is shared with the BootstrapRunner pipeline path
        instead of being duplicated here with its own (previously unseeded)
        copy of the same logic. Each job is executed via tryToFit (not
        execute_fit_job) so the full result dict — including "likelihoods",
        used by some callers for iteration counts — is preserved, matching
        this method's historical return contract.

        Returns
        -------
        models : list of fit result dicts
        best_idx : int or None
        best_val_ll : float or None
        """
        core_limit = kwargs.pop("core_limit", -1)
        verbose = kwargs.get("verbose", False)

        jobs = self.generate_fit_jobs(component_range, **kwargs)
        if not jobs:
            return [], None, None

        def _run_job(job):
            return tryToFit(
                job["train_observations"], job["train_sample_assignments"],
                job["num_components"], job["constrained"],
                job["init_method"], job["init_constraint_adjustment"],
                multivariate=job["multivariate"], **job["kwargs"],
            )

        if core_limit == 1:
            models = [_run_job(job) for job in jobs]
        else:
            verbosity = kwargs.get("verbose_level", 20) if verbose else 0
            if verbose:
                print(f"Running {len(jobs)} fits, cores={core_limit}")
            models = Parallel(n_jobs=core_limit, batch_size=1, verbose=verbosity)(
                delayed(_run_job)(job) for job in jobs
            )

        mv = jobs[0]["multivariate"]

        def _safe_ll(obs, sa, m):
            params = m.get("component_params")
            weights = m.get("weights")
            if params is None or weights is None:
                return -np.inf
            if isinstance(params, list) and (
                len(params) == 0
                or any(isinstance(p, (list, tuple)) and len(p) == 0 for p in params)
            ):
                return -np.inf
            try:
                return _weighted_val_ll(obs, sa, params, weights, mv, kwargs)
            except Exception:
                return -np.inf

        val_observations = jobs[0]["val_observations"]
        val_sample_assignments = jobs[0]["val_sample_assignments"]

        if kwargs.get("bootstrap", True) and val_observations is not None:
            val_lls = [_safe_ll(val_observations, val_sample_assignments, m) for m in models]
            best_idx = int(np.nanargmax(val_lls))
            return models, best_idx, val_lls[best_idx]

        train_observations = jobs[0]["train_observations"]
        train_sample_assignments = jobs[0]["train_sample_assignments"]
        train_lls = [_safe_ll(train_observations, train_sample_assignments, m) for m in models]
        best_idx = int(np.nanargmax(train_lls))
        return models, best_idx, train_lls[best_idx]

    # ──────────────────────────────────────
    # Density evaluation
    # ──────────────────────────────────────

    def joint_densities(self, x, sampleNum):
        """Weighted pdfs of each mixture component.

        Parameters
        ----------
        x : np.ndarray
            (n,) for univariate, (n, D) for multivariate.
        sampleNum : int

        Returns
        -------
        np.ndarray (K, n)
        """
        weights = self.fit_result["weights"][sampleNum]
        params = self.fit_result["component_params"]

        if self.multivariate:
            results = []
            for k, (p, w) in enumerate(zip(params, weights)):
                mu, Delta, Gamma = p
                Delta_arr = np.asarray(Delta)

                if Delta_arr.ndim == 2 and Delta_arr.shape[1] > 1:
                    # CFUSN path
                    from .cfusn.density_utils import cfusn_logpdf_alternate_missing
                    x_2d = np.atleast_2d(x)
                    if np.isnan(x_2d).any():
                        log_pdf = cfusn_logpdf_alternate_missing(x_2d, mu, Delta, Gamma)
                    else:
                        from .cfusn.density_utils import cfusn_logpdf_alternate
                        log_pdf = cfusn_logpdf_alternate(x_2d, mu, Delta, Gamma)
                    results.append(w * np.exp(log_pdf))
                else:
                    # Restricted MSN (q=1)
                    if not (isinstance(p, (list, tuple)) and len(p) == 3):
                        raise ValueError(
                            f"Component {k} params malformed: expected (mu, Delta, Gamma), "
                            f"got {type(p)} of length {len(p) if hasattr(p, '__len__') else '?'}"
                        )
                    from .cfusn.density_utils import msn_logpdf_alternate_missing
                    Delta_vec = np.asarray(Delta).ravel()
                    log_pdf = msn_logpdf_alternate_missing(np.atleast_2d(x), mu, Delta_vec, Gamma)
                    results.append(w * np.exp(log_pdf))
            return np.array(results)
        else:
            return np.array([
                w * skewnorm.pdf(x, a, loc, scale)
                for (a, loc, scale), w in zip(params, weights)
            ])

    def _fit_eval(self):
        self._eval_metrics = {}
        for sampleNum, (sample_scores, sample_name) in enumerate(self.scoreset.samples):
            if self.multivariate:
                self._eval_metrics[sample_name] = {"note": "multivariate — per-dim eval TBD"}
            else:
                u = np.unique(sample_scores)
                u.sort()
                self._eval_metrics[sample_name] = {
                    "empirical_cdf": empirical_cdf(u),
                    "model_cdf": get_sample_cdf(
                        self.fit_result["component_params"],
                        self.fit_result["weights"], u, sampleNum
                    ),
                }
                self._eval_metrics[sample_name]["cdf_dist"] = yang_dist(
                    self._eval_metrics[sample_name]["empirical_cdf"],
                    self._eval_metrics[sample_name]["model_cdf"],
                )

    def get_prior_estimate(self, population_sample, **kwargs):
        pathogenic_idx = kwargs.get("pathogenic_idx", 0)
        benign_idx = kwargs.get("benign_idx", 1)
        pathogenic_density = self.joint_densities(population_sample, pathogenic_idx).sum(0)
        benign_density = self.joint_densities(population_sample, benign_idx).sum(0)
        prior_estimate = 0.5
        tolerance = kwargs.get("tolerance", 1e-6)
        max_em_steps = kwargs.get("max_em_steps", 10000)
        for step in range(max_em_steps):
            posteriors = 1 / (
                1 + (1 - prior_estimate) / prior_estimate
                * benign_density / pathogenic_density
            )
            new_prior = np.nanmean(posteriors)
            if abs(new_prior - prior_estimate) < tolerance:
                prior_estimate = new_prior
                break
            prior_estimate = new_prior
            if prior_estimate <= 0 or prior_estimate >= 1:
                raise ValueError(f"Invalid prior estimate: {prior_estimate}")
        return prior_estimate

    def get_log_lrPlus(self, x, pathogenic_idx=0, controls_idx=1):
        fP = self.joint_densities(x, pathogenic_idx)
        fB = self.joint_densities(x, controls_idx)
        return np.log(fP) - np.log(fB)

    def get_score_thresholds(self, prior, point_values,
                              acmg_mapping_method="tavtigian", **kwargs):
        if prior <= 0 or prior >= 1:
            raise ValueError(f"Prior must be in (0,1), got {prior:.4f}")
        point_values = np.array(point_values)
        if (point_values <= 0).any():
            raise ValueError(f"point_values must be positive, got {point_values}")

        scores = self.scoreset.scores
        if self.multivariate:
            log_LR = self.get_log_lrPlus(scores)
            log_lr_sum = log_LR.sum(axis=0)
            order = np.argsort(log_lr_sum)
            uscores_1d = np.arange(len(order))
            score_ranges_p, score_ranges_b, C = calculate_score_ranges(
                log_lr_sum[order], log_lr_sum[order],
                prior, uscores_1d,
                point_values, acmg_mapping_method=acmg_mapping_method,
            )
            return score_ranges_p, score_ranges_b
        else:
            uscores = np.linspace(scores.min(), scores.max(), 1000)
            log_LR = self.get_log_lrPlus(uscores)
            score_ranges_p, score_ranges_b, C = calculate_score_ranges(
                log_LR, log_LR, prior, uscores, point_values,
                acmg_mapping_method=acmg_mapping_method,
            )
            return score_ranges_p, score_ranges_b

    def get_continuous_classification(self, scores, prior, targets=None):
        """Method B (continuous LR+): per-score posterior + ACMG label.

        Bypasses integer points entirely.  Returns a dict with arrays
        'lr_plus', 'posterior', 'label' aligned to *scores*.
        """
        from .bayesian_thresholds import (
            bayes_posterior_from_lr, continuous_classify,
        )
        if prior <= 0 or prior >= 1:
            raise ValueError(f"Prior must be in (0,1), got {prior:.4f}")
        log_lr = self.get_log_lrPlus(np.asarray(scores))
        if log_lr.ndim > 1:
            log_lr = log_lr.sum(axis=0)
        lr = np.exp(log_lr)
        post = bayes_posterior_from_lr(lr, prior)
        labels = continuous_classify(lr, prior, targets=targets)
        return {"lr_plus": lr, "posterior": post, "label": labels}

    def to_dict(self, **kwargs):
        model_params = serialize_dict(self.fit_result)
        return {
            **model_params,
            "multivariate": self.multivariate,
            "latent_q": self._get_latent_q(),
            "eval_metrics": {
                k: v for k, v in getattr(self, "_eval_metrics", {}).items()
            },
        }

    def _get_latent_q(self):
        """Determine latent dimension q from fitted params."""
        params = self.fit_result.get("component_params", [])
        if params and self.multivariate:
            return get_q(params)
        return 1

    @staticmethod
    def _select_calibration_dims(observations, min_overlap_rows=30):
        """Find the largest subset of assay dimensions with sufficient pairwise overlap.

        Two dimensions overlap if at least ``min_overlap_rows`` variants have
        non-NaN scores in both assays simultaneously.

        Returns
        -------
        list of int
            Column indices to use.  Empty list means no pair has sufficient
            overlap — caller should fall back to per-assay 1-D calibration.
        """
        from itertools import combinations
        D = observations.shape[1]
        if D == 1:
            return [0]

        overlap = np.zeros((D, D), dtype=int)
        for i in range(D):
            for j in range(i + 1, D):
                both = ~np.isnan(observations[:, i]) & ~np.isnan(observations[:, j])
                overlap[i, j] = overlap[j, i] = both.sum()

        # adjacency: edge between i and j if overlap is sufficient
        adj = overlap >= min_overlap_rows
        np.fill_diagonal(adj, True)

        all_dims = list(range(D))
        if all(adj[i, j] for i in all_dims for j in all_dims):
            return all_dims

        # Find largest clique (brute force; fine for D <= ~10)
        for size in range(D - 1, 1, -1):
            for subset in combinations(range(D), size):
                if all(adj[i, j] for i in subset for j in subset):
                    return list(subset)

        return []  # no pair has sufficient overlap

    # ──────────────────────────────────────
    # Job generation for distributed fitting
    # ──────────────────────────────────────

    def generate_fit_jobs(self, component_range, **kwargs):
        observations = self.scoreset.scores
        mv = self.multivariate
        if not mv and observations.ndim == 2 and observations.shape[1] > 1:
            mv = True

        if mv:
            _latent_q = kwargs.get("latent_q", 2)
            _max_K = max(component_range)
            _default_fits = min((2 ** _latent_q) ** _max_K, 100)
            NUM_FITS = kwargs.get("num_fits", _default_fits)
        else:
            NUM_FITS = kwargs.get("num_fits", 100)

        if not mv:
            kwargs["score_min"] = min(
                kwargs.get("score_min", observations.min()), observations.min()
            )
            kwargs["score_max"] = max(
                kwargs.get("score_max", observations.max()), observations.max()
            )

        bootstrap_seed = kwargs.get("bootstrap_seed", None)
        onehot_rng = np.random.RandomState(bootstrap_seed)
        sample_assignments = self.scoreset.sample_assignments
        sample_assignments = makeOneHot(sample_assignments, rng=onehot_rng)

        if mv:
            include = sample_assignments.any(axis=1) & ~np.all(np.isnan(observations), axis=1)
        else:
            include = sample_assignments.any(axis=1) & ~np.isnan(observations)

        observations = observations[include]
        sample_assignments = sample_assignments[include]

        # ── Overlap check: reduce dimensions if assays don't co-occur enough ──
        verbose_overlap = kwargs.get("verbose_overlap", True)
        log_prefix = kwargs.get("log_prefix", "")
        calibrated_dims = list(range(observations.shape[1])) if mv else None
        if mv:
            min_overlap = kwargs.get("min_overlap_rows", 30)
            calibrated_dims = Fit._select_calibration_dims(observations, min_overlap)
            if len(calibrated_dims) == 0:
                if verbose_overlap:
                    print(
                        f"{log_prefix}[OVERLAP] No assay pair has ≥{min_overlap} shared "
                        f"observations. Cannot run multivariate calibration — fall back "
                        f"to per-assay 1D calibrations."
                    )
                return []
            elif len(calibrated_dims) < observations.shape[1]:
                excluded = sorted(set(range(observations.shape[1])) - set(calibrated_dims))
                if verbose_overlap:
                    print(
                        f"{log_prefix}[OVERLAP] Assay dim(s) {excluded} lack ≥{min_overlap} "
                        f"shared rows with the rest. Running {len(calibrated_dims)}D "
                        f"calibration on dims {calibrated_dims}; use existing 1D "
                        f"calibrations for dim(s) {excluded}."
                    )
                observations = observations[:, calibrated_dims]
                valid = (
                    ~np.all(np.isnan(observations), axis=1)
                    & sample_assignments.any(axis=1)
                )
                observations = observations[valid]
                sample_assignments = sample_assignments[valid]
            kwargs["calibrated_dims"] = calibrated_dims

        train_indices = np.arange(len(observations))
        val_indices = np.array([], dtype=int)
        bootstrap_seed = kwargs.get("bootstrap_seed", None)
        if kwargs.get("bootstrap", True):
            if mv:
                train_indices, val_indices = pattern_stratified_bootstrap(
                    observations, sample_assignments, bootstrap_seed
                )
            else:
                train_indices, val_indices = sample_specific_bootstrap(
                    sample_assignments, bootstrap_seed
                )

        constrained = kwargs.get("check_monotonic", True)

        init_method = kwargs.get("init_strategy", "random")
        if init_method != "random":
            init_methods = np.full(NUM_FITS, init_method)
        else:
            if mv:
                init_methods = np.full(NUM_FITS, "kmeans")
            else:
                init_method_rng = np.random.RandomState(bootstrap_seed)
                init_methods = init_method_rng.choice(["kmeans", "method_of_moments"], size=NUM_FITS)

        init_constraint_adjustment = "scale"
        init_constraint_adjustments = np.full(NUM_FITS, init_constraint_adjustment)

        _latent_q_jobs = kwargs.get("latent_q", 2)
        master_seed = kwargs.get("master_seed", None)

        jobs = []
        for i in range(NUM_FITS):
            for num_components in component_range:
                if mv:
                    n_patterns = min((2 ** _latent_q_jobs) ** num_components, 100)
                else:
                    n_patterns = 2 ** num_components
                kwargs["lambdaIndex"] = i % n_patterns
                kwargs["fit_seed"] = derive_fit_seed(master_seed, bootstrap_seed, num_components, i)
                job = {
                    "job_id": f"b{bootstrap_seed}_f{i}_c{num_components}",
                    "bootstrap_seed": bootstrap_seed,
                    "fit_idx": i,
                    "num_components": num_components,
                    "train_observations": observations[train_indices],
                    "train_sample_assignments": sample_assignments[train_indices],
                    "val_observations": observations[val_indices] if len(val_indices) else None,
                    "val_sample_assignments": sample_assignments[val_indices] if len(val_indices) else None,
                    "val_variant_indices": val_indices if len(val_indices) else None,
                    "constrained": constrained,
                    "init_method": init_methods[i],
                    "init_constraint_adjustment": init_constraint_adjustments[i],
                    "multivariate": mv,
                    "calibrated_dims": calibrated_dims,
                    "kwargs": kwargs.copy(),
                }
                jobs.append(job)
        return jobs

    @staticmethod
    def execute_fit_job(job):
        try:
            mv = job.get("multivariate", False)
            result = tryToFit(
                job["train_observations"],
                job["train_sample_assignments"],
                job["num_components"],
                job["constrained"],
                job["init_method"],
                job["init_constraint_adjustment"],
                multivariate=mv,
                verbose=False,
                **job["kwargs"],
            )
            result.pop("history", None)
            result.pop("likelihoods", None)

            # Guard: if init failed, component_params contains empty lists
            params = result.get("component_params", [])
            weights = result.get("weights")
            init_failed = (
                weights is None
                or not params
                or any(
                    isinstance(p, (list, tuple)) and len(p) == 0
                    for p in params
                )
            )
            if init_failed:
                return {
                    "dataset_name": job.get("dataset_name"),
                    "bootstrap_seed": job["bootstrap_seed"],
                    "num_components": job["num_components"],
                    "fit_idx": job["fit_idx"],
                    "fit": result,
                    "val_ll": -np.inf,
                    "calibrated_dims": job.get("calibrated_dims"),
                }

            val_ll = None
            if job["val_observations"] is not None:
                val_ll = _weighted_val_ll(
                    job["val_observations"],
                    job["val_sample_assignments"],
                    result["component_params"],
                    result["weights"],
                    mv,
                    job.get("kwargs", {}),
                )

            return {
                "dataset_name": job.get("dataset_name"),
                "bootstrap_seed": job["bootstrap_seed"],
                "num_components": job["num_components"],
                "fit_idx": job["fit_idx"],
                "fit": result,
                "val_ll": val_ll,
                "calibrated_dims": job.get("calibrated_dims"),
            }
        except Exception as e:
            print(f"Failed: b{job['bootstrap_seed']} c{job['num_components']} f{job['fit_idx']}: {e}")


# ══════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════

def prior_from_weights(weights, population_idx=2, controls_idx=1, pathogenic_idx=0, inverted=False):
    print("This method does not produce very good estimates for 2 component mixture")
    w_idx = 1 if inverted else 0
    prior = (
        (weights[population_idx, w_idx] - weights[controls_idx, w_idx])
        / (weights[pathogenic_idx, w_idx] - weights[controls_idx, w_idx])
    ).item()
    return prior if 0 < prior < 1 else np.nan


def thresholds_from_prior(prior, point_values,
                           acmg_mapping_method="tavtigian", **kwargs):
    """LR+ tier thresholds at *prior* for the given integer point codes.

    acmg_mapping_method : str
        "tavtigian" (default): legacy C*-based thresholds via
            get_tavtigian_constant.  lrP = C^(pv/len(pv)).
        "piecewise": prior-adaptive Bayesian thresholds with exact posteriors
            at the four ACMG boundary points (T = 10, 6, -1, -7).  See
            assay_calibration.fit_utils.bayesian_thresholds.

    Returns
    -------
    lrP, lrB, C
        lrP[i], lrB[i] are LR+ thresholds for pathogenic/benign evidence at
        point_values[i].  C is the Tavtigian constant for
        acmg_mapping_method='tavtigian' and None for 'piecewise'.
    """
    if acmg_mapping_method == "tavtigian":
        C = get_tavtigian_constant(prior, **kwargs)
        pv = np.array(point_values)
        lrP = C ** (pv / len(pv))
        lrB = 1 / lrP
        return lrP, lrB, C
    elif acmg_mapping_method == "piecewise":
        from .bayesian_thresholds import piecewise_tier_thresholds
        return piecewise_tier_thresholds(prior, point_values)
    elif acmg_mapping_method == "strict_additive":
        try:
            from tavtigian_sims.additivity import recanonical_tier_thresholds
        except ImportError:
            import sys, os as _os
            _root = _os.path.normpath(
                _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "..", "..", "..", ".."))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from tavtigian_sims.additivity import recanonical_tier_thresholds
        return recanonical_tier_thresholds(prior, point_values)
    else:
        raise ValueError(
            f"Unknown acmg_mapping_method {acmg_mapping_method!r}; "
            f"expected 'tavtigian', 'piecewise', or 'strict_additive'"
        )


def assign_p(lr, tau, points):
    for i, t in enumerate(tau):
        if lr >= t and (i == len(tau) - 1 or lr < tau[i + 1]):
            return points[i]
    return 0


def assign_b(lr, tau, points):
    for i, t in enumerate(tau):
        if lr <= t and (i == len(tau) - 1 or lr > tau[i + 1]):
            return points[i]
    return 0


def get_point_ranges(scores, lrPlus, tau, point_values, pathogenicOrBenign):
    point_values = np.array(point_values, int)
    assign = assign_p
    if pathogenicOrBenign == "benign":
        point_values = -1 * point_values
        assign = assign_b
    point_ranges = {int(p): [] for p in point_values}
    range_open = np.nan
    range_point = np.nan
    for si, li in zip(scores, lrPlus):
        pt = assign(li, tau, point_values)
        if pt != range_point:
            if range_point not in {np.nan, 0}:
                point_ranges[int(range_point)].append(sorted(list(map(float, (range_open, si)))))
            range_open = si
            range_point = pt
    if range_point not in {np.nan, 0}:
        point_ranges[int(range_point)].append(sorted(list(map(float, (range_open, scores[-1])))))
    return point_ranges


def calculate_score_ranges(log_lrPlusLow, log_lrPlusHigh, prior, scores, point_values,
                            acmg_mapping_method="tavtigian", **kwargs):
    lrP, lrB, C = thresholds_from_prior(
        prior, point_values, acmg_mapping_method=acmg_mapping_method, **kwargs
    )
    tauP = np.log(lrP)
    tauB = np.log(lrB)
    pathogenic_ranges = get_point_ranges(scores, log_lrPlusLow, tauP, point_values, "pathogenic")
    benign_ranges = get_point_ranges(scores, log_lrPlusHigh, tauB, point_values, "benign")
    return pathogenic_ranges, benign_ranges, C


def calculate_score_ranges_dual(log_lr_pathogenic, log_lr_benign, prior_pathogenic, prior_benign,
                                 scores, point_values, acmg_mapping_method="tavtigian", **kwargs):
    """Dual-prior sibling of calculate_score_ranges for the pathomechanism
    two-LR/two-prior PS3/BS3 scheme: pathogenic-direction tier thresholds are
    derived from prior_pathogenic (rho) alone, benign-direction thresholds
    from prior_benign (pi) alone -- two independent calls to
    thresholds_from_prior, each contributing only its own direction's half.

    This is a deliberate deviation from strict Tavtigian-2018, where a single
    constant C is jointly calibrated to satisfy ALL FOUR rule-combination
    categories (pathogenic/likely-pathogenic/benign/likely-benign)
    simultaneously under ONE prior (see get_tavtigian_constant). Here, C_p is
    independently searched using only prior_pathogenic (and only its
    pathogenic/likely-pathogenic halves are used, via lrP/tauP) and C_b is
    independently searched using only prior_benign (only its
    benign/likely-benign halves are used, via lrB/tauB). This is accepted as
    the necessary cost of having two independently-meaningful priors for the
    two evidence directions (see module-level pathomechanism plan/docs for
    the rho vs pi derivation) -- PS3 and BS3 are separate ACMG evidence codes
    with separately awarded points in this pipeline already, not combined
    into one continuous score, so calibrating them under two different
    constants does not break anything downstream that assumed a single C.

    Returns (pathogenic_ranges, benign_ranges, C_pathogenic, C_benign).
    """
    lrP, _, C_p = thresholds_from_prior(
        prior_pathogenic, point_values, acmg_mapping_method=acmg_mapping_method, **kwargs
    )
    _, lrB, C_b = thresholds_from_prior(
        prior_benign, point_values, acmg_mapping_method=acmg_mapping_method, **kwargs
    )
    tauP = np.log(lrP)
    tauB = np.log(lrB)
    pathogenic_ranges = get_point_ranges(scores, log_lr_pathogenic, tauP, point_values, "pathogenic")
    benign_ranges = get_point_ranges(scores, log_lr_benign, tauB, point_values, "benign")
    return pathogenic_ranges, benign_ranges, C_p, C_b


def calculate_classification_ranges(log_lrPlusLow, log_lrPlusHigh, prior, scores,
                                     targets=None, floor_at_neutral=False):
    """Continuous-method (Method B) analogue of calculate_score_ranges.

    Returns score intervals keyed by classification label rather than by
    integer points.  Boundaries are exact: a score falls in label "P" iff
    the posterior from its LR+ is >= 0.99 at the given prior, etc.

    Parameters
    ----------
    targets : dict, optional
        Override target posteriors, e.g. ``{"LB": 0.01, "B": 0.001}``.
        Any key not given falls back to
        ``bayesian_thresholds.DEFAULT_TARGETS`` (P=0.99, LP=0.90, LB=0.10,
        B=0.01). LP/P are rarely changed; LB/B are the common override for
        stricter benign-direction targets.
    floor_at_neutral : bool, optional
        If True, clamp targets so LB/B never trigger on evidence that
        actually favours pathogenicity (and LP/P never trigger on evidence
        favouring benignity) -- see
        ``bayesian_thresholds._floor_targets_at_prior``. Otherwise a target
        below (resp. above) the prior can be satisfied by wrong-direction
        evidence, since reaching that posterior doesn't require overcoming
        the prior in that direction.

    Returns
    -------
    pathogenic_ranges : {"P": [[lo,hi], ...], "LP": [...]}
    benign_ranges     : {"B": [[lo,hi], ...], "LB": [...]}
    thresholds        : dict label -> LR+ threshold (continuous)
    """
    from .bayesian_thresholds import continuous_lr_thresholds, DEFAULT_TARGETS
    merged_targets = {**DEFAULT_TARGETS, **(targets or {})}
    thresholds = continuous_lr_thresholds(prior, merged_targets,
                                          floor_at_neutral=floor_at_neutral)

    def _ranges(log_lr_arr, labels_in_order, lr_thresholds, scores):
        # labels_in_order: ascending in LR+ stringency, e.g. ["LP", "P"] for path
        # lr_thresholds[label] gives the LR+ at which posterior reaches that label.
        tau = np.log(np.array([lr_thresholds[l] for l in labels_in_order]))
        out = {l: [] for l in labels_in_order}

        def _assign(lr):
            for i in range(len(tau) - 1, -1, -1):
                if lr >= tau[i]:
                    return labels_in_order[i]
            return None

        range_open = np.nan
        range_label = None
        for si, li in zip(scores, log_lr_arr):
            lab = _assign(li)
            if lab != range_label:
                if range_label is not None:
                    out[range_label].append(sorted([float(range_open), float(si)]))
                range_open = si
                range_label = lab
        if range_label is not None:
            out[range_label].append(sorted([float(range_open), float(scores[-1])]))
        return out

    def _ranges_benign(log_lr_arr, labels_in_order, lr_thresholds, scores):
        # labels_in_order: ascending in LR+ stringency *for benign* (B more
        # stringent than LB), so we compare against -log(lr) flipped.
        tau = np.log(np.array([lr_thresholds[l] for l in labels_in_order]))
        # For benign, lr below threshold means evidence is "at least this benign".
        out = {l: [] for l in labels_in_order}

        def _assign(lr):
            for i in range(len(tau) - 1, -1, -1):
                if lr <= tau[i]:
                    return labels_in_order[i]
            return None

        range_open = np.nan
        range_label = None
        for si, li in zip(scores, log_lr_arr):
            lab = _assign(li)
            if lab != range_label:
                if range_label is not None:
                    out[range_label].append(sorted([float(range_open), float(si)]))
                range_open = si
                range_label = lab
        if range_label is not None:
            out[range_label].append(sorted([float(range_open), float(scores[-1])]))
        return out

    # Pathogenic side: P is more stringent than LP (higher LR+)
    pathogenic_ranges = _ranges(log_lrPlusLow, ["LP", "P"], thresholds, scores)
    # Benign side: B is more stringent than LB (lower LR+)
    benign_ranges = _ranges_benign(log_lrPlusHigh, ["LB", "B"], thresholds, scores)
    return pathogenic_ranges, benign_ranges, thresholds


def makeOneHot(sample_assignments, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    assert np.all(sample_assignments.any(axis=0))
    sample_assignments = np.array(sample_assignments)
    onehot = np.zeros_like(sample_assignments)
    max_attempts = 10_000
    for _ in range(max_attempts):
        for i in range(sample_assignments.shape[0]):
            true_indices = np.where(sample_assignments[i])[0]
            if len(true_indices) > 0:
                selected = rng.choice(true_indices)
                onehot[i] = False
                onehot[i, selected] = True
        if np.all(np.any(onehot, axis=0)):
            break
    else:
        uncovered = np.where(~np.any(onehot, axis=0))[0].tolist()
        raise ValueError(
            f"makeOneHot: could not cover sample(s) {uncovered} after {max_attempts} attempts. "
            f"These samples have only shared variants and there are not enough shared variants "
            f"to guarantee coverage. Each sample should have at least one exclusive variant."
        )
    assert np.all(np.any(onehot, axis=0))
    assert np.all(onehot.sum(axis=1) <= 1)
    return onehot


def assign_points(scores, point_score_ranges):
    points = np.zeros_like(scores, dtype=int) if scores.ndim == 1 else np.zeros(len(scores), dtype=int)
    for points_key, score_ranges in point_score_ranges.items():
        for score_range in score_ranges:
            if scores.ndim == 1:
                mask = (scores >= score_range[0]) & (scores <= score_range[1])
            else:
                mask = (np.arange(len(scores)) >= score_range[0]) & (np.arange(len(scores)) <= score_range[1])
            points[mask] = points_key
    return points