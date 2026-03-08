from .update_steps import em_iteration, get_sample_weights
from .density_utils import get_likelihood
from .initializations import kmeans_init, methodOfMomentsInit
from . import constraints

import numpy as np
import logging
from tqdm.auto import tqdm
import warnings


def single_fit(
    observations, sample_indicators, N_components, constrained, init_method, init_constraint_adjustment, **kwargs
):
    """
    Fit a two-component mixture model to the observations using the EM algorithm.

    Parameters
    ----------
    observations : np.ndarray
        1D array of observations (e.g., scores).
    sample_indicators : np.ndarray
        2D one-hot encoded array indicating sample membership for each observation.
    constrained : bool
        Whether to enforce component-pair density ratio constraints
    init_method : kmeans or method_of_moments

    Optional Parameters (kwargs)
    -------------------------


    max_em_iters : int, default=10000
        Maximum number of EM iterations.

    verbose : bool, default=True
        Whether to display a progress bar.

    initial_weights : np.ndarray, optional
        Optional initial weights for the samples. If provided along with initial_params, these will be used to initialize the EM algorithm.
            Otherwise, the model will be initialized using k-means initialization.

    initial_params : list of np.ndarray, optional
        Optional initial parameters for the mixture components. If provided along with initial_weights, these will be used to initialize the EM algorithm.
            Otherwise, the model will be initialized using k-means initialization.

    early_stopping : bool, default=True
        Whether to stop the EM algorithm early if the likelihood converges.

    check_submerged_duration : bool, default=False
        Whether to check if the unconstrained model goes underwater. Useful for learning model behavior.

    DEPRECATED : submerge_steps : int, optional
        Optional max number of initial steps to explore without constraint. constrained must be set to true.

    Returns
    -------
    dict
        A dictionary containing:
        - 'component_params': List of parameters for each mixture component.
        - 'weights': Final weights for each sample.
        - 'likelihoods': Array of likelihood values at each iteration.
        - 'history': List of dictionaries containing component parameters and weights at each iteration.
        - 'kmeans': KMeans object used for initialization (if applicable), or None.
        - 'xlims': bounds of scores
        - 'times_submerged': 'list containing each underwater duration taken by the model'
    """
    MAX_EM_ITERS = kwargs.get("max_em_iters", 10000)
    verbose = kwargs.get("verbose", True)
    check_submerged_duration = kwargs.get("check_submerged_duration", False)
    MIN_SCALE = 1e-100
    
    submerge_steps = kwargs.get("submerge_steps", None)
    if submerge_steps is not None:# and not constrained:
        # raise ValueError("constrained must be True when submerge_steps is not None.")
        raise NotImplementedError('submerge_steps is deprecated')
    
    xlims = (observations.min(), observations.max())
    N_samples = sample_indicators.shape[1]
    if (
        kwargs.get("initial_weights", None) is not None
        and kwargs.get("initial_params", None) is not None
    ):
        kmeans = None
        # Start with user provided initialization
        initial_params = kwargs.get("initial_params", [])
        W = np.array(kwargs.get("initial_weights"))
        if W.shape != (N_samples, N_components):
            raise ValueError(
                f"Initial weights shape {W.shape} does not match number of samples {N_samples}"
            )
        if len(initial_params) != N_components:
            raise ValueError(
                f"Initial params length {len(initial_params)} does not match number of components {N_components}"
            )
    else:
        W = np.ones((N_samples, N_components)) / N_components

        assert init_method == "method_of_moments" or init_method == "kmeans"
        initial_params = None
        
        if init_method == "method_of_moments":
            kmeans = "method_of_moments"
            initial_params = methodOfMomentsInit(observations, N_components, constrained, init_constraint_adjustment=init_constraint_adjustment, **kwargs)

        # init_method is kmeans or was a failed method of moments (fall back to kmeans)
        if initial_params is None:
            if init_method == "method_of_moments" and verbose:
                print('failed method of moments, falling back to kmeans')
                
            # Run Initialization
            try:
                initial_params, kmeans = kmeans_init(
                    observations, n_clusters=N_components, constrained=constrained, init_constraint_adjustment=init_constraint_adjustment, **kwargs
                )
            except ValueError:
                logging.warning("Failed to initialize")
                return dict(
                    component_params=[[] for _ in range(N_components)],
                    weights=W,
                    likelihoods=[-1 * np.inf],
                    xlims=xlims,
                    times_submerged=[],
                )
            
        W = get_sample_weights(observations, sample_indicators, initial_params, W)
    history = [dict(component_params=initial_params, weights=W)]
    # initial likelihood
    likelihoods = np.array(
        [
            get_likelihood(observations, sample_indicators, initial_params, W)
            / len(sample_indicators),
        ]
    )
    # Check for bad initialization
    try:
        updated_component_params, updated_weights = em_iteration(
            observations,
            sample_indicators,
            initial_params,
            W,
            constrained,
            xlims,
            iterNum=0,
        )
    except ZeroDivisionError:
        logging.warning("ZeroDivisionError")
        return dict(
            component_params=initial_params,
            weights=W,
            likelihoods=[*likelihoods, -1 * np.inf],
            kmeans=kmeans,
            xlims=xlims,
            times_submerged=[],
        )
    likelihoods = np.array(
        [
            *likelihoods,
            get_likelihood(
                observations,
                sample_indicators,
                updated_component_params,
                updated_weights,
            )
            / len(sample_indicators),
        ]
    )
    # Run the EM algorithm
    if verbose:
        pbar = tqdm(total=MAX_EM_ITERS, leave=False, desc="EM Iteration")

    try: # return failed fit upon ValueError or ZeroDivisionError
    
        underwater_time = 0
        times_submerged = [] # only append when coming back up
        if not constrained and check_submerged_duration:
            is_underwater = constraints.multicomponent_density_constraint_violated(updated_component_params, xlims)
            if is_underwater:
                underwater_time += 1 # already did step 0 before for loop
    
        for i in range(MAX_EM_ITERS):
            history.append(
                dict(component_params=updated_component_params, weights=updated_weights)
            )
            if np.isnan(likelihoods).any():
                raise ValueError()
            if np.isnan(np.concatenate(updated_component_params)).any():
                raise ValueError()
            if np.isnan(updated_weights).any():
                raise ValueError()
            if np.isnan(np.concatenate(updated_component_params)).any():
                raise ValueError(
                    f"NaN in updated component params at iteration {i}\n{updated_component_params}"
                )
            if np.isnan(updated_weights).any():
                raise ValueError(
                    f"NaN in updated weights at iteration {i}\n{updated_weights}"
                )
            updated_component_params, updated_weights = em_iteration(
                observations,
                sample_indicators,
                updated_component_params,
                updated_weights,
                constrained,
                xlims,
                iterNum=i + 1,
            )
            # enforce minimum scale to accommodate numerical errors
            for i, (a, loc, scale) in enumerate(updated_component_params):
                if scale < MIN_SCALE:
                    updated_component_params[i] = (a, loc, max(scale, MIN_SCALE))
    
            # check underwater duration
            if not constrained and check_submerged_duration:
                violated = constraints.multicomponent_density_constraint_violated(updated_component_params, xlims)
    
                if is_underwater and violated: # stayed underwater
                    underwater_time += 1
                elif is_underwater and not violated: # resurfaced by chance
                    is_underwater = False
                    times_submerged.append(underwater_time)
                    underwater_time = 0
                elif not is_underwater and violated: # went back underwater
                    is_underwater = True
                    underwater_time += 1
                elif not is_underwater and not violated: # stayed above water
                    pass
    
            likelihoods = np.array(
                [
                    *likelihoods,
                    get_likelihood(
                        observations,
                        sample_indicators,
                        updated_component_params,
                        updated_weights,
                    )
                    / len(sample_indicators),
                ]
            )
            if i > 0 and (likelihoods[-1] < likelihoods[-2]):
                decrease = likelihoods[-2] - likelihoods[-1]
                
                relative_decrease = decrease / abs(likelihoods[-2])
                
                is_numerical_error = decrease < 1e-13
                
                if not is_numerical_error:
                    raise ValueError(f"Iteration {i}: Likelihood ({likelihoods[-2]}->{likelihoods[-1]}) decreased by {decrease:.2e} (relative: {relative_decrease:.2e}) - (numerical rounding?)\nParams: {history[-1]['component_params']}-->{updated_component_params}\nWeights: {history[-1]['weights']}-->{updated_weights}")
                
            if kwargs.get("verbose", True):
                pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})  # type: ignore
                pbar.update(1)  # type: ignore
            if (
                kwargs.get("early_stopping", True)
                and i >= 1
                and (np.abs(likelihoods[-1] - likelihoods[-2]) / abs(likelihoods[-2]) < 1e-8).all() # relative difference < 1e-8
            ):
                break
    
        
        # check final if resurfaced
        if not constrained and check_submerged_duration:
            violated = constraints.multicomponent_density_constraint_violated(updated_component_params, xlims)
    
            if is_underwater and not violated: # resurfaced by chance on last iteration
                is_underwater = False
                times_submerged.append(underwater_time)
                underwater_time = 0
        
        history.append(
            dict(component_params=updated_component_params, weights=updated_weights)
        )
        if kwargs.get("verbose", True):
            pbar.close()  # type: ignore
        if constrained and constraints.multicomponent_density_constraint_violated(
            updated_component_params, xlims
        ):
            raise ValueError("Final parameters violate density constraint")


    
    except (ValueError, ZeroDivisionError) as e:
        warnings.warn(f'Failed fit: {e}')
        return dict(
            component_params=updated_component_params,
            weights=updated_weights,
            likelihoods=[*likelihoods, -1 * np.inf],
            kmeans=kmeans,
            xlims=xlims,
            times_submerged=[],
        )

    return dict(
        component_params=updated_component_params,
        weights=updated_weights,
        likelihoods=likelihoods,
        history=history,
        kmeans=kmeans,
        xlims=xlims,
        times_submerged=times_submerged,
        initial_params = initial_params
    )
