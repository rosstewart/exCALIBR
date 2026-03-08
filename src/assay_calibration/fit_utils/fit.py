import sys
from pathlib import Path
import os
import json

sys.path.append(str(Path(__file__).resolve().parents[2]))
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from assay_calibration.data_utils.dataset import Scoreset, BasicScoreset
from scipy.stats import skewnorm
import numpy as np
from typing import Tuple
from .evidence_thresholds import get_tavtigian_constant
import logging
import sys
from typing import List, Dict,Tuple
from joblib import Parallel, delayed
from .two_sample.fit import single_fit
from .two_sample.density_utils import get_likelihood
from .utils import serialize_dict
import time
from tqdm.auto import tqdm
logging.basicConfig()
logging.root.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)



def tryToFit(observations, sample_indicators, num_components, constrained, init_method, init_constraint_adjustment, **kwargs):
    fit_results = single_fit(observations, sample_indicators, num_components, constrained, init_method, init_constraint_adjustment, **kwargs)
    return fit_results


def get_bootstrap_indices(dataset_size):
    """
    Generate a bootstrap split of a dataset of a given size

    Required Arguments:
    --------------------------------
    dataset_size -- int
        The size of the dataset to generate the bootstrap split for

    Returns:
    --------------------------------
    train_indices -- np.ndarray
        The indices of the training set
    test_indices -- np.ndarray
        The indices of the test set
    """
    indices = np.arange(dataset_size)
    train_indices = np.random.choice(indices, size=dataset_size, replace=True)
    test_indices = np.setdiff1d(indices, train_indices)
    return train_indices, test_indices


def sample_specific_bootstrap(sample_assignments, bootstrap_seed=None):
    """
    Bootstrap each sample separately

    Required Arguments:
    --------------------------------
    sample_assignments -- np.ndarray (NSamples, NComponents)
        The one-hot encoded sample assignments

    Returns:
    --------------------------------
    train_indices -- np.ndarray
        The indices of the training set
    eval_indices -- np.ndarray
        The indices of the eval set
    """
    train_indices = []
    eval_indices = []
    rng = np.random.RandomState(bootstrap_seed) # seeds are different on each subsequent call but the same across runs
    
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
    def __init__(self, scoreset: Scoreset | BasicScoreset):
        self.scoreset = scoreset
        self.fit_result = {}

    @classmethod
    def from_dict(cls, scoreset, fit_dict):
        model = MulticomponentCalibrationModel.from_params(
            fit_dict["skewness"],
            fit_dict["locs"],
            fit_dict["scales"],
            fit_dict["sample_weights"],
        )
        cls = cls(scoreset)
        cls.model = model
        cls._eval_metrics = fit_dict["eval_metrics"]
        return cls

    def run(self, component_range, **kwargs):
        """
        Run a single fit on a bootstrapped sample of the data

        Required Arguments:
        --------------------------------
        component_range -- list of int
            The range of components to fit

        Optional Arguments:
        --------------------------------
        num_fits -- int (default 100)
            The number of fits to run
        core_limit -- int (default -1)
            The number of cores to use for parallel processing
        bootstrap -- bool (default True)
            Whether to use bootstrap sampling
        - bootstrap_seed : int (default None)
            Seed to randomly sample during bootstrap
        - check_convergence : bool (default True)
            If True, check for convergence in the log likelihood
        - verbose : bool (default False)
            If True, print progress messages.
        - max_em_iters : int (default 10,000)
            Maximum number of iterations to run the EM algorithm.
        - tol : float (default 1e-6)
            Tolerance for convergence in the log likelihood.
        - check_monotonic : bool (default True)
            If True, check for monotonicity between each pair of neighboring components
        - submerge_steps : int (default None)
            Max number of steps to explore without constraint. Halves after every limit hit. check_monotonic must be True.
        - init_constraint_adjustment_param : str (default "skew")
            Param to adjust during intialization to satisfy constraint. Either "skew", "scale", or "random".
        - init_strategy : str (default random)
            pick one - (kmeans, method_of_moments, random)
        - score_min : float | int (default None)
            Minimum score to consider when checking for monotonicity.
        - score_max : float | int (default None)
            Maximum score to consider when checking for monotonicity.
        """
        NUM_FITS = kwargs.get("num_fits", 100)
        observations = self.scoreset.scores
        kwargs["score_min"] = min(
            kwargs.get("score_min", observations.min()), self.scoreset.scores.min()
        )
        kwargs["score_max"] = max(
            kwargs.get("score_max", observations.max()), self.scoreset.scores.max()
        )
        sample_assignments = self.scoreset.sample_assignments
        if kwargs.get("verbose", False):
            print(f"sample counts: {sample_assignments.sum(0)}")
        sample_assignments = makeOneHot(sample_assignments)
        if kwargs.get("verbose", False):
            print(f"sample counts: {sample_assignments.sum(0)}")
        include = sample_assignments.any(axis=1) & ~np.isnan(observations)
        observations = observations[include]
        sample_assignments = sample_assignments[include]
        if kwargs.get("verbose", False):
            print(f"sample counts: {sample_assignments.sum(0)}")
        train_indices = np.arange(len(observations))
        val_indices = np.array([], dtype=int)
        bootstrap_seed = kwargs.get("bootstrap_seed", None)
        if kwargs.get("bootstrap", True):
            train_indices, val_indices = sample_specific_bootstrap(sample_assignments, bootstrap_seed)
        constrained = kwargs.get("check_monotonic", True)

        # init_methods contains the desired method for each fit
        init_method = kwargs.get("init_strategy", "random")
        if init_method != 'random':
            init_methods = np.full(NUM_FITS, init_method) 
        else:
            init_methods = np.random.choice(["kmeans", "method_of_moments"], size=NUM_FITS)

        # adjust skew or scale during initial constraint adjustment
        init_constraint_adjustment = kwargs.get("init_constraint_adjustment_param", "scale")
        if init_constraint_adjustment != 'random':
            init_constraint_adjustments = np.full(NUM_FITS, init_constraint_adjustment) 
        else:
            init_constraint_adjustments = np.random.choice(["skew", "scale"], size=NUM_FITS)

        val_observations = observations[val_indices]
        val_sample_assignments = sample_assignments[val_indices]
        train_observations = observations[train_indices]
        train_sample_assignments = sample_assignments[train_indices]
        

        core_limit = kwargs.get("core_limit", -1)

        if core_limit == 1:
            if kwargs.get("verbose", False):
                print(
                    f"Running {NUM_FITS} fits for each of {len(component_range)} components sequentially"
                )
            # models = [
            #     tryToFit(
            #         train_observations,
            #         train_sample_assignments,
            #         num_components,
            #         constrained,
            #         init_methods[i],
            #         init_constraint_adjustments[i],
            #         **kwargs,
            #     )
            #     for i in range(NUM_FITS)
            #     for num_components in component_range
            # ]
            models = []
            for num_components in component_range:
                for i in range(NUM_FITS):
                    kwargs["lambdaIndex"] = i%(2**num_components)
                    models.append(tryToFit(
                        train_observations,
                        train_sample_assignments,
                        num_components,
                        constrained,
                        init_methods[i],
                        init_constraint_adjustments[i],
                        **kwargs,
                    ))
        else:
            verbosity = 0
            if kwargs.get("verbose", False):
                print(
                    f"Running {NUM_FITS} fits for each of {len(component_range)} components with {core_limit} cores"
                )
                verbosity = kwargs.get("verbose_level", 20)
            models = Parallel(n_jobs=core_limit, batch_size=1, verbose=verbosity)(
                delayed(tryToFit)(
                    train_observations,
                    train_sample_assignments,
                    num_components,
                    constrained,
                    init_methods[i],
                    init_constraint_adjustments[i],
                    **{**kwargs, "lambdaIndex": i % (2**num_components)}  # Merge dicts
                )
                for i in range(NUM_FITS)
                for num_components in component_range
            )

        # calculate best model and val LL
        if kwargs.get("bootstrap", True):
            val_lls = [get_likelihood(val_observations, val_sample_assignments, m['component_params'], m['weights']) / len(val_sample_assignments) for m in models]
            best_idx = np.nanargmax(val_lls)
            best_fit = models[best_idx]
            best_val_ll = val_lls[best_idx]

            return models, best_idx, best_val_ll
        
        return models, None, None


    def generate_fit_jobs(self, component_range, **kwargs):
        """
        Generate job specifications for fits without executing them.
        This allows for better parallelization strategies.
        
        Returns:
            list of dict: Job specifications containing all parameters needed for each fit
        """
        NUM_FITS = kwargs.get("num_fits", 100)
        observations = self.scoreset.scores
        kwargs["score_min"] = min(
            kwargs.get("score_min", observations.min()), self.scoreset.scores.min()
        )
        kwargs["score_max"] = max(
            kwargs.get("score_max", observations.max()), self.scoreset.scores.max()
        )
        
        sample_assignments = self.scoreset.sample_assignments
        sample_assignments = makeOneHot(sample_assignments)
        include = sample_assignments.any(axis=1) & ~np.isnan(observations)
        observations = observations[include]
        sample_assignments = sample_assignments[include]
        
        train_indices = np.arange(len(observations))
        val_indices = np.array([], dtype=int)
        bootstrap_seed = kwargs.get("bootstrap_seed", None)
        do_bootstrap = kwargs.get("bootstrap", True)
        
        if bootstrap_seed is not None:
            assert do_bootstrap
        
        if do_bootstrap:
            train_indices, val_indices = sample_specific_bootstrap(sample_assignments, bootstrap_seed)
        
        constrained = kwargs.get("check_monotonic", True)
        
        # Initialize methods
        init_method = kwargs.get("init_strategy", "random")
        if init_method != 'random':
            init_methods = np.full(NUM_FITS, init_method)
        else:
            np.random.seed(bootstrap_seed)  # Ensure reproducibility
            init_methods = np.random.choice(["kmeans", "method_of_moments"], size=NUM_FITS)
        
        init_constraint_adjustment = "scale"#kwargs.get("init_constraint_adjustment_param", "skew")
        if init_constraint_adjustment != 'random':
            init_constraint_adjustments = np.full(NUM_FITS, init_constraint_adjustment)
        else:
            np.random.seed(bootstrap_seed)
            init_constraint_adjustments = np.random.choice(["skew", "scale"], size=NUM_FITS)
        
        # Generate job specifications
        jobs = []
        for i in range(NUM_FITS):
            for num_components in component_range:
                kwargs["lambdaIndex"] = i%(2**num_components)
                job = {
                    'job_id': f"b{bootstrap_seed}_f{i}_c{num_components}",
                    'bootstrap_seed': bootstrap_seed,
                    'fit_idx': i,
                    'num_components': num_components,
                    'train_observations': observations[train_indices],
                    'train_sample_assignments': sample_assignments[train_indices],
                    'val_observations': observations[val_indices] if len(val_indices) > 0 else None,
                    'val_sample_assignments': sample_assignments[val_indices] if len(val_indices) > 0 else None,
                    'constrained': constrained,
                    'init_method': init_methods[i],
                    'init_constraint_adjustment': init_constraint_adjustments[i],
                    'kwargs': kwargs.copy()
                }
                jobs.append(job)
        
        return jobs
    
    @staticmethod
    def execute_fit_job(job):
        """Execute a single fit job and save immediately."""
        # Check if already completed (for resumability)
        #save_path = f"{job['save_dir']}/{job['dataset_name']}_b{job['bootstrap_seed']}_c{job['num_components']}_f{job['fit_idx']}.pkl"
        #if os.path.exists(save_path):
        #    # print(f"Skipping existing: {save_path}")
        #    return None

        # print(f"Running {save_path}...",flush=True)
        
        try:
            result = tryToFit(
                job['train_observations'],
                job['train_sample_assignments'],
                job['num_components'],
                job['constrained'],
                job['init_method'],
                job['init_constraint_adjustment'],
                verbose=False,
                **job['kwargs']
            )

            result.pop('history', None) # free up space
            result.pop('likelihoods', None) # free up space
            
            # Calculate validation likelihood
            val_ll = None
            if job['val_observations'] is not None:
                val_ll = get_likelihood(
                    job['val_observations'],
                    job['val_sample_assignments'],
                    result['component_params'],
                    result['weights']
                ) / len(job['val_sample_assignments'])
            
            # Save to disk
            save_data = {
                'dataset_name': job['dataset_name'],
                'bootstrap_seed': job['bootstrap_seed'],
                'num_components': job['num_components'],
                'fit_idx': job['fit_idx'],
                'fit': result,
                'val_ll': val_ll
            }
            
            # with open(save_path, 'wb') as f:
            #     pickle.dump(save_data, f)

            return save_data
            
        except Exception as e:
            print(f"Failed: {job['dataset_name']} b{job['bootstrap_seed']} c{job['num_components']} f{job['fit_idx']}: {e}")


        
        
    def joint_densities(self, x, sampleNum):
        """
        weighted pdfs of a mixture of skew normal distributions

        Parameters
        ----------
        x : np.array (n,)
            values at which to evaluate the pdf
        sampleNum : int
            index of the sample to use

        Returns
        -------
        np.array (k, n)
            joint density for each component of the mixture
        """
        weights = self.fit_result["weights"][sampleNum]
        return np.array(
            [
                w * skewnorm.pdf(x, a, loc, scale)
                for (a, loc, scale), w in zip(
                    self.fit_result["component_params"], weights
                )
            ]
        )  # type: ignore

    def _fit_eval(self):
        """
        Evaluate the fit quality
        """
        self._eval_metrics = {}
        for sampleNum, (sample_scores, sample_name) in enumerate(self.scoreset.samples):  # type: ignore
            u = np.unique(sample_scores)
            u.sort()
            self._eval_metrics[sample_name] = {}
            self._eval_metrics[sample_name]["empirical_cdf"] = empirical_cdf(
                u
            )
            self._eval_metrics[sample_name]["model_cdf"] = get_sample_cdf(
                self.fit_result['component_params'],
                 self.fit_result['weights'],
                   u,
                   sampleNum
            )
            self._eval_metrics[sample_name]["cdf_dist"] = yang_dist(
                self._eval_metrics[sample_name]["empirical_cdf"],
                self._eval_metrics[sample_name]["model_cdf"],
            )

    def get_prior_estimate(self, population_sample: np.ndarray, **kwargs) -> float:
        """
        Get the prior estimate for a given sample

        Required Arguments:
        --------------------------------
        population_sample -- np.ndarray
            Observations from the population

        Optional Arguments:
        --------------------------------
        pathogenic_idx -- int (default 0)
            The index of the pathogenic component in the weights matrix
        benign_idx -- int (default 1)
            The index of the benign component in the weights matrix
        tolerance -- float (default 1e-6)
            The tolerance for convergence of the prior estimate
        max_em_steps -- int (default 10000)
            Maximum number of steps to run EM update for prior estimate

        Returns:
        --------------------------------
        prior -- float
            The prior probability of pathogenicity
        """
        pathogenic_idx = kwargs.get("pathogenic_idx", 0)
        benign_idx = kwargs.get("benign_idx", 1)
        pathogenic_density = self.joint_densities(population_sample,pathogenic_idx).sum(0)
        benign_density = self.joint_densities(population_sample, benign_idx).sum(0)
        # Initialize values for MLLS
        prior_estimate = 0.5
        converged = False
        tolerance = kwargs.get("tolerance", 1e-6)
        em_steps = 0
        max_em_steps = kwargs.get("max_em_steps",10000)
        while not converged:
            em_steps += 1
            posteriors = 1 / (
                1
                + (1 - prior_estimate)
                / prior_estimate
                * benign_density # type: ignore
                / pathogenic_density
            )
            new_prior = np.nanmean(posteriors)
            prior_estimate = new_prior
            if prior_estimate < 0 or prior_estimate > 1:
                raise ValueError(f"Invalid prior estimate obtained, {prior_estimate}")
            if em_steps >= max_em_steps:
                print(f"EM prior estimate algorithm failed to converge after {max_em_steps:,d} iterations. ")
                break
        return prior_estimate


    def get_log_lrPlus(self, x, pathogenic_idx=0, controls_idx=1):
        fP = self.joint_densities(x,pathogenic_idx)
        fB = self.joint_densities(x,controls_idx)
        return np.log(fP) - np.log(fB)

    def get_score_thresholds(self, prior, point_values,**kwargs):
        """
        Calculate the point to score range mappings for this model's estimated pathogenic and benign
        class conditional score distributions and the given prior
        """
        if (prior <= 0) or (prior >= 1):
            raise ValueError(f'Prior must be in the range (0,1), received {prior:.4f}')
        point_values = np.array(point_values)
        if (point_values <= 0).any():
            raise ValueError(f"point_values must be a list of positive integers; received {point_values}")
        uscores = np.linspace(
            self.scoreset.scores.min(), self.scoreset.scores.max(), 1000
        )
        log_LR = self.get_log_lrPlus(uscores)
        (
            score_ranges_pathogenic,
            score_ranges_benign,
            C
        ) = calculate_score_ranges(
            log_LR, log_LR, prior, uscores, point_values
        )
        return score_ranges_pathogenic, score_ranges_benign

    def to_dict(self, **kwargs):
        # model_params = {k: v.tolist() for k, v in self.model.get_params().items()}
        model_params = serialize_dict(self.fit_result)
        extra = {}
        return {
            **model_params,
            **extra,
            "eval_metrics": {
                k: {
                    "empirical_cdf": v["empirical_cdf"].tolist(),
                    "model_cdf": v["model_cdf"].tolist(),
                    "cdf_dist": v["cdf_dist"],
                }
                for k, v in self._eval_metrics.items()
            },
        }

def prior_from_weights(
    weights: np.ndarray,
    population_idx: int = 2,
    controls_idx: int = 1,
    pathogenic_idx: int = 0,
    inverted: bool = False,
) -> float:
    """
    Calculate the prior probability of an observation from the population being pathogenic

    Required Arguments:
    --------------------------------
    weights -- Ndarray (NSamples, NComponents)
        The mixture weights of each sample

    Optional Arguments:
    --------------------------------
    population_idx -- int (default 2)
        The index of the population component in the weights matrix

    controls_idx -- int (default 1)
        The index of the controls (i.e. benign) component in the weights matrix

    pathogenic_idx -- int (default 0)
        The index of the pathogenic component in the weights matrix

    Returns:
    --------------------------------
    prior -- float
        The prior probability of an observation from the population being pathogenic
    """
    print(
        "This method does not produce very good estimates for 2 component mixture and is invalid for more than 2 components"
    )
    if inverted:
        w_idx = 1
    else:
        w_idx = 0
    prior = (
        (weights[population_idx, w_idx] - weights[controls_idx, w_idx])
        / (weights[pathogenic_idx, w_idx] - weights[controls_idx, w_idx])
    ).item()
    if prior <= 0 or prior >= 1:
        return np.nan
    return prior


def thresholds_from_prior(prior, point_values,**kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the evidence thresholds (LR+ values) for each point value given a prior

    Parameters
    ----------
    prior : float
        The prior probability of pathogenicity


    """
    C = get_tavtigian_constant(prior,**kwargs)

    lrThresholdP = C**(np.array(point_values)/len(point_values))
    lrThresholdB = 1/(C**(np.array(point_values)/len(point_values)))

    return lrThresholdP, lrThresholdB, C
    
    
def assign_p(lr, tau,points):
    for i, t in enumerate(tau):
        if lr >= t and (i == len(tau)-1 or lr < tau[i+1]):
            return points[i]
    return 0

def assign_b(lr, tau, points):
    for i, t in enumerate(tau):
        if lr <= t and (i == len(tau)-1 or lr > tau[i+1]):
            return points[i]
    return 0

def get_point_ranges(scores, lrPlus, tau, point_values, pathogenicOrBenign)->Dict[int, List[Tuple[float,float]]]:
    point_values = np.array(point_values,int)
    assign = assign_p
    if pathogenicOrBenign == "benign":
        point_values = int(-1) * point_values
        assign = assign_b
    point_ranges = {int(p) : [] for p in point_values}
    range_open = np.nan
    range_point = np.nan
    for si, li in list(zip(scores, lrPlus)):
        point_i = assign(li, tau, point_values)
        if point_i != range_point:
            if range_point not in {np.nan,0}:
                point_ranges[int(range_point)].append( sorted(list(map(float,(range_open, si) ))))
            range_open = si
            range_point = point_i
    if range_point not in {np.nan,0}:
        point_ranges[int(range_point)].append( sorted(list(map(float,(range_open, scores[-1]) ))))
    return point_ranges

def calculate_score_ranges(log_lrPlusLow, log_lrPlusHigh,prior, scores,point_values,**kwargs) -> Tuple[Dict[int,List[Tuple[float,float]]],Dict[int,List[Tuple[float,float]]]]:
    """
    {point_value: int -> (range_start:float, range_end:float)}
    """
    lrThresholdP,lrThresholdB,C = thresholds_from_prior(prior, point_values,**kwargs)
    tauP = np.log(lrThresholdP)
    tauB = np.log(lrThresholdB)
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
                selected_index = np.random.choice(true_indices)
                onehot[i] = False
                onehot[i, selected_index] = True
    assert np.all(np.any(onehot, axis=0))
    assert np.all(onehot.sum(axis=1) <= 1)
    # print(onehot)
    return onehot

def assign_points(scores, point_score_ranges: Dict[int, List[Tuple[float,float]]]):
    """
    Assign points to each score based on the thresholds

    Required Arguments:
    --------------------------------
    scores -- np.ndarray (n,)
        The scores to assign points to

    point_score_ranges -- Dict[int, List[Tuple[float,float]]]
        The score ranges for evidence points
    """
    
    points = np.zeros_like(scores, dtype=int)
    for points_key, score_ranges in point_score_ranges.items():
        for score_range in score_ranges:
            mask = (scores >= score_range[0]) & \
                    (scores <= score_range[1])
            points[mask] = points_key
    return points
