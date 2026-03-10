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
from .multivariate.fit import single_fit
from .multivariate.density_utils import (
    get_likelihood, msn_logpdf_alternate_missing, mixture_pdf, log_joint_densities,
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

        Returns
        -------
        models : list of fit result dicts
        best_idx : int or None
        best_val_ll : float or None
        """
        NUM_FITS = kwargs.get("num_fits", 100)
        observations = self.scoreset.scores
        mv = self.multivariate

        # Fallback: detect from data shape
        if not mv and observations.ndim == 2 and observations.shape[1] > 1:
            mv = True
            self.multivariate = True

        if not mv:
            kwargs["score_min"] = min(
                kwargs.get("score_min", observations.min()), observations.min()
            )
            kwargs["score_max"] = max(
                kwargs.get("score_max", observations.max()), observations.max()
            )
        # else: xlims are per-dimension, handled inside single_fit

        sample_assignments = self.scoreset.sample_assignments
        if kwargs.get("verbose", False):
            print(f"sample counts: {sample_assignments.sum(0)}")
        sample_assignments = makeOneHot(sample_assignments)
        if kwargs.get("verbose", False):
            print(f"sample counts (after one-hot): {sample_assignments.sum(0)}")

        # Filter invalid rows
        if mv:
            # Keep rows where at least one dimension is non-NaN AND assigned to a sample
            include = sample_assignments.any(axis=1) & ~np.all(np.isnan(observations), axis=1)
        else:
            include = sample_assignments.any(axis=1) & ~np.isnan(observations)

        observations = observations[include]
        sample_assignments = sample_assignments[include]

        train_indices = np.arange(len(observations))
        val_indices = np.array([], dtype=int)
        bootstrap_seed = kwargs.get("bootstrap_seed", None)
        if kwargs.get("bootstrap", True):
            train_indices, val_indices = sample_specific_bootstrap(
                sample_assignments, bootstrap_seed
            )

        constrained = kwargs.get("check_monotonic", True)

        init_method = kwargs.get("init_strategy", "random")
        if init_method != "random":
            init_methods = np.full(NUM_FITS, init_method)
        else:
            if mv:
                init_methods = np.full(NUM_FITS, "kmeans")  # MoM not yet for MV
            else:
                init_methods = np.random.choice(
                    ["kmeans", "method_of_moments"], size=NUM_FITS
                )

        init_constraint_adjustment = kwargs.get("init_constraint_adjustment_param", "scale")
        if init_constraint_adjustment != "random":
            init_constraint_adjustments = np.full(NUM_FITS, init_constraint_adjustment)
        else:
            init_constraint_adjustments = np.random.choice(["skew", "scale"], size=NUM_FITS)

        val_observations = observations[val_indices] if len(val_indices) else None
        val_sample_assignments = sample_assignments[val_indices] if len(val_indices) else None
        train_observations = observations[train_indices]
        train_sample_assignments = sample_assignments[train_indices]

        core_limit = kwargs.get("core_limit", -1)

        if core_limit == 1:
            models = []
            for num_components in component_range:
                for i in range(NUM_FITS):
                    kwargs["lambdaIndex"] = i % (2 ** num_components)
                    models.append(tryToFit(
                        train_observations, train_sample_assignments,
                        num_components, constrained,
                        init_methods[i], init_constraint_adjustments[i],
                        multivariate=mv, **kwargs,
                    ))
        else:
            verbosity = kwargs.get("verbose_level", 20) if kwargs.get("verbose", False) else 0
            if kwargs.get("verbose", False):
                print(f"Running {NUM_FITS} fits × {len(component_range)} components, cores={core_limit}")
            models = Parallel(n_jobs=core_limit, batch_size=1, verbose=verbosity)(
                delayed(tryToFit)(
                    train_observations, train_sample_assignments,
                    num_components, constrained,
                    init_methods[i], init_constraint_adjustments[i],
                    **{**kwargs, "multivariate": mv,
                       "lambdaIndex": i % (2 ** num_components)}
                )
                for i in range(NUM_FITS)
                for num_components in component_range
            )

        # Select best model
        def _safe_ll(obs, sa, m):
            """Compute likelihood, returning -inf for failed fits."""
            params = m.get("component_params")
            weights = m.get("weights")
            if params is None or weights is None:
                return -np.inf
            # Check for empty or malformed params
            if isinstance(params, list) and (
                len(params) == 0
                or any(isinstance(p, (list, tuple)) and len(p) == 0 for p in params)
            ):
                return -np.inf
            try:
                return get_likelihood(
                    obs, sa, params, weights, multivariate=mv
                ) / len(sa)
            except Exception:
                return -np.inf

        if kwargs.get("bootstrap", True) and val_observations is not None:
            val_lls = [_safe_ll(val_observations, val_sample_assignments, m) for m in models]
            best_idx = int(np.nanargmax(val_lls))
            return models, best_idx, val_lls[best_idx]

        # No bootstrap: select by training likelihood
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
                # p should be (mu, Delta, Gamma) — a 3-tuple of arrays
                if not (isinstance(p, (list, tuple)) and len(p) == 3):
                    raise ValueError(
                        f"Component {k} params malformed: expected (mu, Delta, Gamma), "
                        f"got {type(p)} of length {len(p) if hasattr(p, '__len__') else '?'}"
                    )
                mu, Delta, Gamma = p
                log_pdf = msn_logpdf_alternate_missing(np.atleast_2d(x), mu, Delta, Gamma)
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
                # Evaluate CDF per dimension is not straightforward for MV;
                # store per-dimension marginal CDFs
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

    def get_score_thresholds(self, prior, point_values, **kwargs):
        if prior <= 0 or prior >= 1:
            raise ValueError(f"Prior must be in (0,1), got {prior:.4f}")
        point_values = np.array(point_values)
        if (point_values <= 0).any():
            raise ValueError(f"point_values must be positive, got {point_values}")

        scores = self.scoreset.scores
        if self.multivariate:
            # For MV: evaluate LR along observed scores (N, D)
            log_LR = self.get_log_lrPlus(scores)
            # Sum log-LR across components, result is (K, N) → sum over K → (N,)
            log_lr_sum = log_LR.sum(axis=0)
            # Sort by aggregate LR for threshold finding
            order = np.argsort(log_lr_sum)
            uscores_1d = np.arange(len(order))  # proxy 1-d scores
            score_ranges_p, score_ranges_b, C = calculate_score_ranges(
                log_lr_sum[order], log_lr_sum[order],
                prior, scores[order] if not self.multivariate else uscores_1d,
                point_values
            )
            return score_ranges_p, score_ranges_b
        else:
            uscores = np.linspace(scores.min(), scores.max(), 1000)
            log_LR = self.get_log_lrPlus(uscores)
            score_ranges_p, score_ranges_b, C = calculate_score_ranges(
                log_LR, log_LR, prior, uscores, point_values
            )
            return score_ranges_p, score_ranges_b

    def to_dict(self, **kwargs):
        model_params = serialize_dict(self.fit_result)
        return {
            **model_params,
            "multivariate": self.multivariate,
            "eval_metrics": {
                k: v for k, v in getattr(self, "_eval_metrics", {}).items()
            },
        }

    # ──────────────────────────────────────
    # Job generation for distributed fitting
    # ──────────────────────────────────────

    def generate_fit_jobs(self, component_range, **kwargs):
        NUM_FITS = kwargs.get("num_fits", 100)
        observations = self.scoreset.scores
        mv = self.multivariate
        if not mv and observations.ndim == 2 and observations.shape[1] > 1:
            mv = True

        if not mv:
            kwargs["score_min"] = min(
                kwargs.get("score_min", observations.min()), observations.min()
            )
            kwargs["score_max"] = max(
                kwargs.get("score_max", observations.max()), observations.max()
            )

        sample_assignments = self.scoreset.sample_assignments
        sample_assignments = makeOneHot(sample_assignments)

        if mv:
            include = sample_assignments.any(axis=1) & ~np.all(np.isnan(observations), axis=1)
        else:
            include = sample_assignments.any(axis=1) & ~np.isnan(observations)

        observations = observations[include]
        sample_assignments = sample_assignments[include]

        train_indices = np.arange(len(observations))
        val_indices = np.array([], dtype=int)
        bootstrap_seed = kwargs.get("bootstrap_seed", None)
        if kwargs.get("bootstrap", True):
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
                np.random.seed(bootstrap_seed)
                init_methods = np.random.choice(["kmeans", "method_of_moments"], size=NUM_FITS)

        init_constraint_adjustment = "scale"
        init_constraint_adjustments = np.full(NUM_FITS, init_constraint_adjustment)

        jobs = []
        for i in range(NUM_FITS):
            for num_components in component_range:
                kwargs["lambdaIndex"] = i % (2 ** num_components)
                job = {
                    "job_id": f"b{bootstrap_seed}_f{i}_c{num_components}",
                    "bootstrap_seed": bootstrap_seed,
                    "fit_idx": i,
                    "num_components": num_components,
                    "train_observations": observations[train_indices],
                    "train_sample_assignments": sample_assignments[train_indices],
                    "val_observations": observations[val_indices] if len(val_indices) else None,
                    "val_sample_assignments": sample_assignments[val_indices] if len(val_indices) else None,
                    "constrained": constrained,
                    "init_method": init_methods[i],
                    "init_constraint_adjustment": init_constraint_adjustments[i],
                    "multivariate": mv,
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

            val_ll = None
            if job["val_observations"] is not None:
                val_ll = get_likelihood(
                    job["val_observations"],
                    job["val_sample_assignments"],
                    result["component_params"],
                    result["weights"],
                    multivariate=mv,
                ) / len(job["val_sample_assignments"])

            return {
                "dataset_name": job.get("dataset_name"),
                "bootstrap_seed": job["bootstrap_seed"],
                "num_components": job["num_components"],
                "fit_idx": job["fit_idx"],
                "fit": result,
                "val_ll": val_ll,
            }
        except Exception as e:
            print(f"Failed: b{job['bootstrap_seed']} c{job['num_components']} f{job['fit_idx']}: {e}")


# ══════════════════════════════════════════════
# Utility functions (unchanged from original)
# ══════════════════════════════════════════════

def prior_from_weights(weights, population_idx=2, controls_idx=1, pathogenic_idx=0, inverted=False):
    print("This method does not produce very good estimates for 2 component mixture")
    w_idx = 1 if inverted else 0
    prior = (
        (weights[population_idx, w_idx] - weights[controls_idx, w_idx])
        / (weights[pathogenic_idx, w_idx] - weights[controls_idx, w_idx])
    ).item()
    return prior if 0 < prior < 1 else np.nan


def thresholds_from_prior(prior, point_values, **kwargs):
    C = get_tavtigian_constant(prior, **kwargs)
    pv = np.array(point_values)
    lrP = C ** (pv / len(pv))
    lrB = 1 / lrP
    return lrP, lrB, C


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


def calculate_score_ranges(log_lrPlusLow, log_lrPlusHigh, prior, scores, point_values, **kwargs):
    lrP, lrB, C = thresholds_from_prior(prior, point_values, **kwargs)
    tauP = np.log(lrP)
    tauB = np.log(lrB)
    pathogenic_ranges = get_point_ranges(scores, log_lrPlusLow, tauP, point_values, "pathogenic")
    benign_ranges = get_point_ranges(scores, log_lrPlusHigh, tauB, point_values, "benign")
    return pathogenic_ranges, benign_ranges, C


def makeOneHot(sample_assignments):
    assert np.all(sample_assignments.any(axis=0))
    sample_assignments = np.array(sample_assignments)
    onehot = np.zeros_like(sample_assignments)
    while not np.all(np.any(onehot, axis=0)):
        for i in range(sample_assignments.shape[0]):
            true_indices = np.where(sample_assignments[i])[0]
            if len(true_indices) > 0:
                selected = np.random.choice(true_indices)
                onehot[i] = False
                onehot[i, selected] = True
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
                # For multivariate, this is an index-based range
                mask = (np.arange(len(scores)) >= score_range[0]) & (np.arange(len(scores)) <= score_range[1])
            points[mask] = points_key
    return points