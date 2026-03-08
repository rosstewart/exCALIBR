from . import density_utils
from .constraints import multicomponent_density_constraint_violated
from typing import List, Tuple, Any
import numpy as np
import scipy.stats as sps


def em_iteration(
    observations,
    sample_indicators,
    current_component_params,
    current_weights,
    constrained,
    xlims,
    **kwargs,
):
    """
    Perform one iteration of the EM algorithm with the density constraint

    Arguments:
    observations: np.array (N,) : observed instances
    sample_indicators: np.array (N, S) : indicator matrix of which sample each observation belongs to
    current_component_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : skewness, location, scale parameters of each component
    current_weights: np.array (S, K) : weights of each component for each sample
    constrained: bool : whether to enforce the density constraint
    xlims: tuple : (xmin, xmax) : range of x values to check the density ratio

    Returns:
    updated_component_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : updated skewness, location, scale parameters of each component
    updated_weights: np.array (S, K) : updated weights of each component for each sample
    """
    if constrained and multicomponent_density_constraint_violated(
        current_component_params, xlims
    ):
        raise ValueError("density constraint violated at start of em iteration,")
    N, S = sample_indicators.shape
    K = len(current_component_params)
    assert current_weights.shape == (S, K)
    sample_indicators = validate_indicators(sample_indicators)
    assert sample_indicators.shape == (N, S)
    responsibilities = sample_specific_responsibilities(
        observations, sample_indicators, current_component_params, current_weights
    )
    updated_component_params: List[Tuple[Any]] = [
        (None,),
    ] * K
    for component_num in range(K):
        candidate_location = get_location_update(
            observations,
            responsibilities[component_num],
            current_component_params[component_num],
        )
        if constrained:
            constrained_updated_loc = get_constrained_location_update(
                candidate_location,
                component_num,
                current_component_params,
                updated_component_params,
                xlims,
                **kwargs,
            )
        else:
            constrained_updated_loc = candidate_location
        candidateDelta = get_Delta_update(
            constrained_updated_loc,
            observations,
            responsibilities[component_num],
            current_component_params[component_num],
        )
        if constrained:
            constrained_updated_Delta = get_constrained_Delta_update(
                candidateDelta,
                constrained_updated_loc,
                component_num,
                current_component_params,
                updated_component_params,
                xlims,
                **kwargs,
            )
        else:
            constrained_updated_Delta = candidateDelta

        candidateGamma = get_Gamma_update(
            constrained_updated_loc,
            constrained_updated_Delta,
            observations,
            responsibilities[component_num],
            current_component_params[component_num],
        )
        if constrained:
            constrained_updated_Gamma = get_constrained_Gamma_update(
                candidateGamma,
                constrained_updated_loc,
                constrained_updated_Delta,
                component_num,
                current_component_params,
                updated_component_params,
                xlims,
                **kwargs,
            )
        else:
            constrained_updated_Gamma = candidateGamma

        updated_component_params[component_num] = density_utils.alternate_to_canonical(  # type: ignore
            constrained_updated_loc,
            constrained_updated_Delta,
            constrained_updated_Gamma,
        )
        if constrained and multicomponent_density_constraint_violated(
            [
                *updated_component_params[: component_num + 1],
                *current_component_params[component_num + 1 :],
            ],
            xlims,
        ):
            raise ValueError(
                f"constraint violated after updating component {component_num} iter {kwargs.get('iterNum',-1)}\n{updated_component_params}\n{current_component_params}"
            )

    updated_weights = get_sample_weights(
        observations, sample_indicators, updated_component_params, current_weights
    )
    return updated_component_params, updated_weights


def get_constrained_location_update(
    candidate_location,
    component_num,
    current_component_params,
    updated_component_params,
    xlims,
    **kwargs,
):
    """
    Perform binary search to get a location update near the unconstrained update that satisfies the density constraint
    """
    # if component_num > 0:
    #         bsearch_params = [updated_component_params[0], *current_component_params[1:]]
    # else:
    #     bsearch_params = [*current_component_params]
    bsearch_params = []
    for ki in range(len(current_component_params)):
        if ki < component_num:
            bsearch_params.append(updated_component_params[ki])
        else:
            bsearch_params.append(current_component_params[ki])
    constrained_updated_loc = binary_search(
        candidate_location,
        bsearch_params,
        component_num,
        0,
        xlims,
        msg=f"loc_{component_num} iter {kwargs.get('iterNum',-1)}",
    )
    return constrained_updated_loc


def get_constrained_Delta_update(
    candidate_Delta,
    constrained_updated_loc,
    component_num,
    current_component_params,
    updated_component_params,
    xlims,
    **kwargs,
):
    K = len(current_component_params)
    bsearch_params = []
    for ki in range(K):
        if ki < component_num:
            bsearch_params.append(updated_component_params[ki])
        elif ki > component_num:
            bsearch_params.append(current_component_params[ki])
        else:
            _, Delta, Gamma = density_utils.canonical_to_alternate(
                *current_component_params[ki]
            )
            bsearch_params.append(
                density_utils.alternate_to_canonical(
                    constrained_updated_loc, Delta, Gamma
                )
            )
    constrained_updated_Delta = binary_search(
        candidate_Delta,
        bsearch_params,
        component_num,
        1,
        xlims,
        msg=f"Delta_{component_num} iter {kwargs.get('iterNum',-1)}",
    )
    return constrained_updated_Delta


def get_constrained_Gamma_update(
    candidate_Gamma,
    constrained_updated_loc,
    constrained_updated_Delta,
    component_num,
    current_component_params,
    updated_component_params,
    xlims,
    **kwargs,
):
    K = len(current_component_params)
    bsearch_params = []
    for ki in range(K):
        if ki < component_num:
            bsearch_params.append(updated_component_params[ki])
        elif ki > component_num:
            bsearch_params.append(current_component_params[ki])
        else:
            _, _, Gamma = density_utils.canonical_to_alternate(
                *current_component_params[ki]
            )
            bsearch_params.append(
                density_utils.alternate_to_canonical(
                    constrained_updated_loc, constrained_updated_Delta, Gamma
                )
            )
    constrained_updated_Gamma = binary_search(
        candidate_Gamma,
        bsearch_params,
        component_num,
        2,
        xlims,
        msg=f"Gamma_{component_num} iter {kwargs.get('iterNum',-1)}",
    )
    return constrained_updated_Gamma


def get_sample_weights(
    observations, sample_indicators, updated_component_params, current_weights
):
    updated_weights = np.zeros_like(current_weights)
    for i in range(current_weights.shape[0]):  # for each sample
        sample_observations = observations[
            sample_indicators[:, i]
        ]  # get all observations for this sample
        posts = density_utils.component_posteriors(
            sample_observations, updated_component_params, current_weights[i]
        )  # Get the component posteriors (responsibilities) using the updated component params and current weights
        updatedWeight = posts.mean(1)
        if np.isnan(updatedWeight).any():
            nanLocs = np.where(np.isnan(posts.T))[0]
            raise ValueError(
                f"about to set updated weight to {updatedWeight}\n{sample_observations[nanLocs]}\n{updated_component_params}\n{current_weights[i]}\n{nanLocs}\n{posts.T[nanLocs]}"
            )
        updated_weights[i] = updatedWeight
    return updated_weights


def sample_specific_responsibilities(
    observations, sample_indicators, component_params, weights
):
    """
    For each observation calculate the posteriors with respect to each component and that observation's sample's component weights

    Arguments:
    observations: np.array (N,) : observed instances
    sample_indicators: np.array (N, S) : indicator matrix of which sample each observation belongs to
    component_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : skewness, location, scale parameters of each component
    weights: np.array (S, K) : weights of each component for each sample

    Returns:
    responsibilities: np.array (K, N) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    """
    N_samples = sample_indicators.shape[1]
    N_components = len(component_params)
    N_observations = len(observations)
    assert weights.shape == (N_samples, N_components)

    responsibilities = np.zeros((N_components, N_observations))
    for i, sample_mask in enumerate(sample_indicators.T):
        X = observations[sample_mask]
        responsibilities[:, sample_mask] = density_utils.component_posteriors(
            X, component_params, weights[i]
        )
    return responsibilities


def validate_indicators(Indicators):
    assert Indicators.ndim == 2
    assert (Indicators.sum(1) == 1).all()
    assert np.isin(Indicators, [0, 1]).all()
    return Indicators.astype(bool)


def get_location_update(observations, responsibilities, component_params):
    """
    Calculate the location update for the given component

    Arguments:
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_loc: float : updated location parameter
    """
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    (_, Delta, Gamma) = density_utils.canonical_to_alternate(*component_params)
    m = observations - v * Delta
    return (m * responsibilities).sum() / responsibilities.sum()


def get_Delta_update(updated_loc, observations, responsibilities, component_params):
    """
    Calculate the Delta update for the given component

    Arguments:
    updated_loc: float : updated location parameter from this iteration
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_Delta: float : updated Delta parameter
    """

    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    d = v * (observations - updated_loc)
    return (d * responsibilities).sum() / (w * responsibilities).sum()


def get_Gamma_update(
    updated_loc, updated_Delta, observations, responsibilities, component_params
):
    """
    Calculate the Gamma update for the given component

    Arguments:
    updated_loc: float : updated location parameter from this iteration
    updated_Delta: float : updated Delta parameter from this iteration
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_Gamma: float : updated Gamma parameter
    """
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    g = (
        (observations - updated_loc) ** 2
        - (2 * updated_Delta * v * (observations - updated_loc))
        + (updated_Delta**2 * w)
    )
    return (g * responsibilities).sum() / responsibilities.sum()


def trunc_norm_moments(mu, sigma):
    """Array trunc norm moments"""
    cdf = sps.norm.cdf(mu / sigma)
    flags = cdf == 0
    pdf = sps.norm.pdf(mu / sigma)
    p = np.zeros_like(pdf)
    p[~flags] = pdf[~flags] / cdf[~flags]
    p[flags] = abs(mu[flags] / sigma)

    m1 = mu + sigma * p
    m2 = mu**2 + sigma**2 + sigma * mu * p
    return m1, m2


def get_truncated_normal_moments(observations, component_params):
    _delta = density_utils._get_delta(component_params)
    loc, scale = component_params[1:]
    truncated_normal_loc = _delta / scale * (observations - loc)
    truncated_normal_scale = np.sqrt(1 - _delta**2)
    v, w = trunc_norm_moments(truncated_normal_loc, truncated_normal_scale)
    return v, w


def binary_search(
    candidate_value, current_params, component_index, parameter_index, xlims, msg=""
):
    """
    Perform a binary search to find the value of the parameter that satisfies the pairwise component density ratio constraint

    Arguments:
    candidate_value: float : candidate value of the parameter ** IN ALTERNATE PARAMETERIZATION **
    current_params: list [tuple : (a, loc, scale) ]: skewness, location, scale parameters for each of the K components
    component_index: int : index of the component to update
    parameter_index: int : index of the parameter to update
    xlims: tuple : (xmin, xmax) : range of x values to check the density ratio

    Returns:
    float : updated parameter value
    """
    if multicomponent_density_constraint_violated(current_params, xlims):
        raise ValueError(f"constraint already violated before bsearch {msg}")
    current_alternate_params = [
        list(density_utils.canonical_to_alternate(*param)) for param in current_params
    ]
    # print(f'old version bsearch candidate {candidate_value} param {parameter_index}, alt params: {current_alternate_params}')
    lower_bound = current_alternate_params[component_index][parameter_index]
    upper_bound = candidate_value
    while abs(upper_bound - lower_bound) > 1e-4:
        midpoint = (upper_bound + lower_bound) / 2
        updated_params = current_alternate_params.copy() # shallow copy
        updated_params = [list(p) for p in current_alternate_params]
        updated_params[component_index][parameter_index] = midpoint
        if multicomponent_density_constraint_violated(
            list(map(lambda tup: density_utils.alternate_to_canonical(*tup), updated_params)),  # type: ignore
            xlims,
        ):
            upper_bound = midpoint
        else:
            lower_bound = midpoint
    verify_binary_search_result(
        lower_bound, current_params, component_index, parameter_index, xlims
    )
    return lower_bound


def verify_binary_search_result(
    constrained_Alternate_parameter_value,
    current_canonical_params,
    component_index,
    update_index,
    xlims,
):
    test_params = [p for p in current_canonical_params] # copy of original to not modify in place
    mu, Delta, Gamma = density_utils.canonical_to_alternate(
        *current_canonical_params[component_index]
    )
    if update_index == 0:
        # updated mu
        test_params[
            component_index
        ] = density_utils.alternate_to_canonical(
            constrained_Alternate_parameter_value, Delta, Gamma
        )
    elif update_index == 1:
        test_params[
            component_index
        ] = density_utils.alternate_to_canonical(
            mu, constrained_Alternate_parameter_value, Gamma
        )
    else:
        # updated Gamma
        test_params[
            component_index
        ] = density_utils.alternate_to_canonical(
            mu, Delta, constrained_Alternate_parameter_value
        )
    # after making the update to current_canonical params, these should still satisfy the constraint
    if multicomponent_density_constraint_violated(test_params, xlims):
        raise ValueError(
            f"The binary search result for the {update_index } parameter update for component {component_index} does not satify the constraint"
        )

