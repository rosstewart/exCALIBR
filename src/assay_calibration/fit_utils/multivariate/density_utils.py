import numpy as np
import scipy.stats as sps
from scipy.special import logsumexp


def mixture_pdf(x, params, weights):
    """
    returns log: pdf of a mixture of skew normal distributions

    $ f(x) = a * f_1(x) + (1-a) f_0(x) $
    """
    return logsumexp(log_joint_densities(x, params, weights), axis=0)

def get_log_fPB(X, params, weights):
    return np.log(joint_densities(X, params, weights).sum(0))

def joint_densities(x, params, weights):
    """
    weighted pdfs of a mixture of skew normal distributions
    """
    return np.array(
        [
            w * sps.skewnorm.pdf(x, a, loc, scale)
            for (a, loc, scale), w in zip(params, weights)
        ]
    )

def log_joint_densities(x, params, weights):
    """
    Return log of weighted pdfs to avoid overflow
    Returns: log(w * pdf(x)) for each component
    """
    weights = np.asarray(weights)
    
    # Compute all log PDFs
    log_pdfs = np.array([
        sps.skewnorm.logpdf(x, a, loc, scale) 
        for (a, loc, scale) in params
    ])
    
    # Add log weights
    with np.errstate(divide='ignore'):
        log_weights = np.log(weights)
    log_weights[weights == 0] = -np.inf
    
    # Return log(w * pdf) = log(w) + log(pdf)
    return log_weights[:, None] + log_pdfs


def component_posteriors(x, canonical_params, individual_sample_weights):
    individual_sample_weights = np.array(individual_sample_weights)[:, None]
    assert len(canonical_params) == individual_sample_weights.shape[0]
    log_pdfs = np.stack(
        [sps.skewnorm.logpdf(x.ravel(), *p) for p in canonical_params], axis=0
    )
    numerators = np.zeros_like(log_pdfs)
    with np.errstate(divide='ignore'): # ignore zero sample weight warning
        numerators = log_pdfs + np.log(individual_sample_weights)
    d = np.zeros_like(numerators[0])
    d = logsumexp(numerators, axis=0)
    P = np.exp(numerators - d[None])  # type: ignore
    P[np.isnan(P)] = 0
    return P


def canonical_to_alternate(a, loc, scale):
    """
    convert canonical parameters to alternate parameters

    Arguments:
    a: skewness parameter
    loc: location parameter
    scale: scale parameter

    Returns:
    Delta
    Gamma
    """
    Delta = 0
    Gamma = 0

    _delta = a / np.sqrt(1 + a**2)
    Delta = scale * _delta
    Gamma = scale**2 - Delta**2

    return tuple(map(float, (loc, Delta, Gamma)))


def alternate_to_canonical(loc, Delta, Gamma):
    """
    convert alternate parameters to canonical parameters

    Arguments:
    loc: location parameter
    Delta
    Gamma

    Returns:
    a: skewness parameter
    loc: location parameter
    scale: scale parameter
    """
    try:
        a = np.sign(Delta) * np.sqrt(Delta**2 / Gamma)
    except ZeroDivisionError:
        raise ZeroDivisionError(
            f"Invalid skewness parameter: {Delta * np.sqrt(1 / Gamma)} from Delta: {Delta}, Gamma: {Gamma}"
        )
    if np.isinf(a) or np.isnan(a):
        raise ZeroDivisionError(
            f"Invalid skewness parameter: {a} from Delta: {Delta}, Gamma: {Gamma}"
        )
    scale = np.sqrt(Gamma + Delta**2)
    
    return tuple(map(float, (a, loc, scale)))



def _get_delta(params):
    a = params[0]
    return a / np.sqrt(1 + a**2)


def get_likelihood(observations, sample_indicators, component_params, weights):
    if component_params is None or weights is None:
        return -1 * np.inf
    Likelihood = 0.0
    for sample_num, sample_mask in enumerate(sample_indicators.T):
        X = observations[sample_mask]
        # sample_likelihood = joint_densities(
        #     X, component_params, weights[sample_num]
        # ).sum(axis=0)
        # Likelihood += np.log(sample_likelihood).sum().item()
        log_weighted_pdfs = log_joint_densities(X, component_params, weights[sample_num])
        # Sum across components using logsumexp
        log_sample_likelihood = logsumexp(log_weighted_pdfs, axis=0)
        Likelihood += log_sample_likelihood.sum()
    return Likelihood # LOG LIKELIHOOD


def get_sample_likelihood(observations, sample_indicators, component_params, weights):
    if component_params is None or weights is None:
        return [-1 * np.inf]*len(sample_indicators.T)
    Likelihoods = [0]*len(sample_indicators.T)
    for sample_num, sample_mask in enumerate(sample_indicators.T):
        X = observations[sample_mask]
        # sample_likelihood = joint_densities(
        #     X, component_params, weights[sample_num]
        # ).sum(axis=0)
        # Likelihood += np.log(sample_likelihood).sum().item()
        log_weighted_pdfs = log_joint_densities(X, component_params, weights[sample_num])
        # Sum across components using logsumexp
        log_sample_likelihood = logsumexp(log_weighted_pdfs, axis=0)
        Likelihoods[sample_num] += log_sample_likelihood.sum()
    return np.array(Likelihoods) # LOG LIKELIHOODS
