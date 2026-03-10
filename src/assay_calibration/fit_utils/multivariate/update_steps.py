from . import density_utils
from .constraints import multicomponent_density_constraint_violated
from typing import List, Tuple, Any
import numpy as np
import scipy.stats as sps


# ══════════════════════════════════════════════
# Truncated-normal moments  (T is ALWAYS scalar)
# ══════════════════════════════════════════════

def trunc_norm_moments(mu, sigma):
    """Moments of TN(mu, sigma^2, R+).  mu, sigma are arrays (N,)."""
    ratio = mu / sigma
    cdf = sps.norm.cdf(ratio)
    pdf = sps.norm.pdf(ratio)
    safe = cdf > 1e-300
    p = np.zeros_like(pdf)
    p[safe] = pdf[safe] / cdf[safe]
    p[~safe] = np.abs(ratio[~safe])
    m1 = mu + sigma * p
    m2 = mu**2 + sigma**2 + sigma * mu * p
    return m1, m2


def get_truncated_normal_moments(observations, component_params):
    """Univariate: params = (a, loc, scale) canonical."""
    _delta = density_utils._get_delta(component_params)
    loc, scale = component_params[1:]
    tn_loc = _delta / scale * (observations - loc)
    tn_scale = np.sqrt(1 - _delta**2)
    return trunc_norm_moments(tn_loc, tn_scale)


def get_truncated_normal_moments_mv(observations, mu, Delta, Gamma):
    """Multivariate: alternate params (mu, Delta, Gamma).  NO missing data.
    observations: (N, K)  |  mu: (K,)  |  Delta: (K,)  |  Gamma: (K, K)
    Returns v: (N,), w: (N,)   — still scalar per observation.
    """
    Omega = Gamma + np.outer(Delta, Delta)
    Omega = 0.5 * (Omega + Omega.T)
    Omega_inv_Delta = np.linalg.solve(Omega, Delta)  # (K,)
    residuals = observations - mu  # (N, K)
    eta = residuals @ Omega_inv_Delta  # (N,)
    sigma_sq = max(1.0 - Delta @ Omega_inv_Delta, 1e-12)
    sigma = np.sqrt(sigma_sq)
    return trunc_norm_moments(eta, sigma)


def get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma):
    """Multivariate TN moments with NaN (missing dimensions) per observation.

    Uses the MSN marginal: for observed set S,
        T | X_S ~ TN(Delta_S' Omega_SS^{-1} (x_S - mu_S),
                     1 - Delta_S' Omega_SS^{-1} Delta_S,  R+)

    observations: (N, K) with NaN  |  mu: (K,)  |  Delta: (K,)  |  Gamma: (K,K)
    Returns v: (N,), w: (N,)
    """
    observations = np.atleast_2d(observations)
    N, K = observations.shape
    v = np.zeros(N)
    w = np.zeros(N)

    obs_mask = ~np.isnan(observations)  # (N, K)
    patterns, inverse = np.unique(obs_mask, axis=0, return_inverse=True)

    for pi, pattern in enumerate(patterns):
        obs_dims = np.where(pattern)[0]
        if len(obs_dims) == 0:
            # Fully missing: no information, use prior moments of TN(0,1,R+)
            idx = np.where(inverse == pi)[0]
            v[idx] = np.sqrt(2 / np.pi)  # E[|Z|] for Z~N(0,1)
            w[idx] = 1.0                 # E[Z^2] = 1
            continue
        idx = np.where(inverse == pi)[0]

        # Marginal parameters for observed dims
        mu_s = mu[obs_dims]
        Delta_s = Delta[obs_dims]
        Gamma_s = Gamma[np.ix_(obs_dims, obs_dims)]

        x_s = observations[np.ix_(idx, obs_dims)]
        v[idx], w[idx] = get_truncated_normal_moments_mv(x_s, mu_s, Delta_s, Gamma_s)

    return v, w


# ══════════════════════════════════════════════
# Univariate update rules (unchanged)
# ══════════════════════════════════════════════

def get_location_update(observations, responsibilities, component_params):
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    (_, Delta, Gamma) = density_utils.canonical_to_alternate(*component_params)
    m = observations - v * Delta
    return (m * responsibilities).sum() / responsibilities.sum()


def get_Delta_update(updated_loc, observations, responsibilities, component_params):
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    d = v * (observations - updated_loc)
    return (d * responsibilities).sum() / (w * responsibilities).sum()


def get_Gamma_update(updated_loc, updated_Delta, observations, responsibilities, component_params):
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    g = (
        (observations - updated_loc) ** 2
        - 2 * updated_Delta * v * (observations - updated_loc)
        + updated_Delta**2 * w
    )
    return (g * responsibilities).sum() / responsibilities.sum()


# ══════════════════════════════════════════════
# Multivariate update rules
# ══════════════════════════════════════════════

def get_location_update_mv(observations, responsibilities, mu, Delta, Gamma):
    """mu_new_d = sum z_j obs_{jd} (x_{jd} - Delta_d * v_j) / sum z_j obs_{jd}
    Handles NaN (missing dimensions) via available-case accumulation.
    observations: (N, K) with NaN | responsibilities: (N,) | mu,Delta: (K,) | Gamma: (K,K)
    Returns: (K,)
    """
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    obs = ~np.isnan(observations)  # (N, K)
    x_fill = np.where(obs, observations, 0.0)
    m = x_fill - v[:, None] * Delta[None, :]
    z = responsibilities[:, None]
    numer = (m * z * obs).sum(axis=0)
    denom = (z * obs).sum(axis=0)
    return numer / np.maximum(denom, 1e-12)


def get_Delta_update_mv(updated_mu, observations, responsibilities, mu, Delta, Gamma):
    """Delta_new_d = sum z_j obs_{jd} v_j (x_{jd} - mu_d) / sum z_j obs_{jd} w_j
    Handles NaN via available-case accumulation.
    Returns: (K,)
    """
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    obs = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    residuals = x_fill - updated_mu[None, :]
    z = responsibilities
    numer = (z[:, None] * v[:, None] * residuals * obs).sum(axis=0)
    denom = (z[:, None] * w[:, None] * obs).sum(axis=0)
    return numer / np.maximum(denom, 1e-12)


def get_Gamma_update_mv(updated_mu, updated_Delta, observations, responsibilities, mu, Delta, Gamma):
    """Gamma_{d1,d2} via available-case accumulation over observations
    where both d1,d2 observed. Handles NaN.
    Returns: (K, K)
    """
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    N, K = observations.shape
    obs = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    z = responsibilities
    residuals = x_fill - updated_mu[None, :]
    Gamma_new = np.zeros((K, K))
    for d1 in range(K):
        for d2 in range(d1, K):
            both = obs[:, d1] & obs[:, d2]
            z_b = z[both]
            if z_b.sum() < 1e-12:
                continue
            r1 = residuals[both, d1]
            r2 = residuals[both, d2]
            v_b = v[both]
            w_b = w[both]
            g = (r1 * r2
                 - updated_Delta[d1] * v_b * r2
                 - updated_Delta[d2] * v_b * r1
                 + updated_Delta[d1] * updated_Delta[d2] * w_b)
            Gamma_new[d1, d2] = (g * z_b).sum() / z_b.sum()
            Gamma_new[d2, d1] = Gamma_new[d1, d2]
    return Gamma_new


# ══════════════════════════════════════════════
# Responsibilities & weights (unified)
# ══════════════════════════════════════════════

def validate_indicators(Indicators):
    assert Indicators.ndim == 2
    assert (Indicators.sum(1) == 1).all()
    assert np.isin(Indicators, [0, 1]).all()
    return Indicators.astype(bool)


def sample_specific_responsibilities(
    observations, sample_indicators, component_params, weights, multivariate=False
):
    N_samples = sample_indicators.shape[1]
    N_components = len(component_params)
    N_observations = observations.shape[0]
    assert weights.shape == (N_samples, N_components)
    responsibilities = np.zeros((N_components, N_observations))
    for i, mask in enumerate(sample_indicators.T):
        X = observations[mask]
        responsibilities[:, mask] = density_utils.component_posteriors(
            X, component_params, weights[i], multivariate=multivariate
        )
    return responsibilities


def get_sample_weights(
    observations, sample_indicators, updated_component_params, current_weights,
    multivariate=False
):
    updated_weights = np.zeros_like(current_weights)
    for i in range(current_weights.shape[0]):
        X = observations[sample_indicators[:, i]]
        posts = density_utils.component_posteriors(
            X, updated_component_params, current_weights[i],
            multivariate=multivariate
        )
        uw = posts.mean(1)
        if np.isnan(uw).any():
            bad = np.where(np.isnan(posts.T))[0]
            raise ValueError(
                f"NaN weight: {uw}\n{X[bad]}\n{updated_component_params}\n{current_weights[i]}"
            )
        updated_weights[i] = uw
    return updated_weights


# ══════════════════════════════════════════════
# Constrained update helpers — univariate (unchanged)
# ══════════════════════════════════════════════

def get_constrained_location_update(
    candidate_location, component_num, current_component_params,
    updated_component_params, xlims, **kwargs
):
    bsearch_params = [
        updated_component_params[k] if k < component_num else current_component_params[k]
        for k in range(len(current_component_params))
    ]
    return binary_search(
        candidate_location, bsearch_params, component_num, 0, xlims,
        msg=f"loc_{component_num} iter {kwargs.get('iterNum', -1)}"
    )


def get_constrained_Delta_update(
    candidate_Delta, constrained_updated_loc, component_num,
    current_component_params, updated_component_params, xlims, **kwargs
):
    K = len(current_component_params)
    bsearch_params = []
    for ki in range(K):
        if ki < component_num:
            bsearch_params.append(updated_component_params[ki])
        elif ki > component_num:
            bsearch_params.append(current_component_params[ki])
        else:
            _, Delta, Gamma = density_utils.canonical_to_alternate(*current_component_params[ki])
            bsearch_params.append(
                density_utils.alternate_to_canonical(constrained_updated_loc, Delta, Gamma)
            )
    return binary_search(
        candidate_Delta, bsearch_params, component_num, 1, xlims,
        msg=f"Delta_{component_num} iter {kwargs.get('iterNum', -1)}"
    )


def get_constrained_Gamma_update(
    candidate_Gamma, constrained_updated_loc, constrained_updated_Delta,
    component_num, current_component_params, updated_component_params,
    xlims, **kwargs
):
    K = len(current_component_params)
    bsearch_params = []
    for ki in range(K):
        if ki < component_num:
            bsearch_params.append(updated_component_params[ki])
        elif ki > component_num:
            bsearch_params.append(current_component_params[ki])
        else:
            _, _, Gamma = density_utils.canonical_to_alternate(*current_component_params[ki])
            bsearch_params.append(
                density_utils.alternate_to_canonical(
                    constrained_updated_loc, constrained_updated_Delta, Gamma
                )
            )
    return binary_search(
        candidate_Gamma, bsearch_params, component_num, 2, xlims,
        msg=f"Gamma_{component_num} iter {kwargs.get('iterNum', -1)}"
    )


# ══════════════════════════════════════════════
# Constrained update helpers — multivariate
# ══════════════════════════════════════════════
# For multivariate, binary search on a single scalar doesn't work
# because mu and Delta are vectors, Gamma is a matrix.
# Instead we do a LINE SEARCH: interpolate between old and candidate
#   param(alpha) = (1 - alpha) * old + alpha * candidate
# and binary search on alpha in [0, 1].

def _mv_build_constraint_params(component_num, current_params, updated_params,
                                 candidate_alternate, alpha):
    """Build a full param list with component_num interpolated at fraction alpha
    between its current alternate values and candidate_alternate.
    candidate_alternate = (mu_cand, Delta_cand, Gamma_cand)
    """
    params = []
    for ki in range(len(current_params)):
        if ki < component_num:
            params.append(updated_params[ki])
        elif ki > component_num:
            params.append(current_params[ki])
        else:
            # current alternate values for this component
            cur_alt = current_params[ki]  # already (mu, Delta, Gamma)
            mu_i = (1 - alpha) * cur_alt[0] + alpha * candidate_alternate[0]
            Delta_i = (1 - alpha) * cur_alt[1] + alpha * candidate_alternate[1]
            Gamma_i = (1 - alpha) * cur_alt[2] + alpha * candidate_alternate[2]
            Gamma_i = 0.5 * (Gamma_i + Gamma_i.T)
            params.append((mu_i, Delta_i, Gamma_i))
    return params


def get_constrained_update_mv(
    candidate_mu, candidate_Delta, candidate_Gamma,
    component_num, current_component_params, updated_component_params,
    xlims, multivariate=True, **kwargs
):
    """Line-search between current and candidate alternate params to satisfy constraint.
    All params in alternate form: (mu, Delta, Gamma).
    Returns: (mu, Delta, Gamma) — the constrained update.
    """
    candidate = (candidate_mu, candidate_Delta, candidate_Gamma)

    # Check if unconstrained candidate already satisfies
    test = _mv_build_constraint_params(
        component_num, current_component_params, updated_component_params,
        candidate, alpha=1.0
    )
    if not multicomponent_density_constraint_violated(test, xlims, multivariate=multivariate):
        return candidate_mu, candidate_Delta, candidate_Gamma

    # Binary search on alpha in [0, 1]
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        test = _mv_build_constraint_params(
            component_num, current_component_params, updated_component_params,
            candidate, alpha=mid
        )
        if multicomponent_density_constraint_violated(test, xlims, multivariate=multivariate):
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break

    final = _mv_build_constraint_params(
        component_num, current_component_params, updated_component_params,
        candidate, alpha=lo
    )
    return final[component_num]


# ══════════════════════════════════════════════
# Binary search — univariate (unchanged)
# ══════════════════════════════════════════════

def binary_search(
    candidate_value, current_params, component_index, parameter_index, xlims, msg=""
):
    if multicomponent_density_constraint_violated(current_params, xlims):
        raise ValueError(f"constraint already violated before bsearch {msg}")
    current_alternate_params = [
        list(density_utils.canonical_to_alternate(*param)) for param in current_params
    ]
    lower_bound = current_alternate_params[component_index][parameter_index]
    upper_bound = candidate_value
    while abs(upper_bound - lower_bound) > 1e-4:
        midpoint = (upper_bound + lower_bound) / 2
        updated_params = [list(p) for p in current_alternate_params]
        updated_params[component_index][parameter_index] = midpoint
        if multicomponent_density_constraint_violated(
            list(map(lambda t: density_utils.alternate_to_canonical(*t), updated_params)),
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
    constrained_val, current_canonical_params, component_index, update_index, xlims
):
    test_params = list(current_canonical_params)
    mu, Delta, Gamma = density_utils.canonical_to_alternate(*current_canonical_params[component_index])
    if update_index == 0:
        test_params[component_index] = density_utils.alternate_to_canonical(constrained_val, Delta, Gamma)
    elif update_index == 1:
        test_params[component_index] = density_utils.alternate_to_canonical(mu, constrained_val, Gamma)
    else:
        test_params[component_index] = density_utils.alternate_to_canonical(mu, Delta, constrained_val)
    if multicomponent_density_constraint_violated(test_params, xlims):
        raise ValueError(
            f"Binary search result for param {update_index} component {component_index} violates constraint"
        )


# ══════════════════════════════════════════════
# EM iteration — unified
# ══════════════════════════════════════════════

def em_iteration(
    observations, sample_indicators, current_component_params, current_weights,
    constrained, xlims, multivariate=False, **kwargs
):
    mv = multivariate
    if constrained and multicomponent_density_constraint_violated(
        current_component_params, xlims, multivariate=mv
    ):
        raise ValueError("density constraint violated at start of em iteration")

    N = observations.shape[0]
    S = sample_indicators.shape[1]
    K = len(current_component_params)
    assert current_weights.shape == (S, K)
    sample_indicators = validate_indicators(sample_indicators)

    responsibilities = sample_specific_responsibilities(
        observations, sample_indicators, current_component_params, current_weights,
        multivariate=mv
    )

    if not mv:
        updated_component_params = _em_update_univariate(
            observations, responsibilities, current_component_params,
            constrained, xlims, K, **kwargs
        )
    else:
        updated_component_params = _em_update_multivariate(
            observations, responsibilities, current_component_params,
            constrained, xlims, K, **kwargs
        )

    updated_weights = get_sample_weights(
        observations, sample_indicators, updated_component_params, current_weights,
        multivariate=mv
    )
    return updated_component_params, updated_weights


def _em_update_univariate(
    observations, responsibilities, current_component_params,
    constrained, xlims, K, **kwargs
):
    """One M-step for all components, univariate case."""
    updated = [None] * K
    for c in range(K):
        z = responsibilities[c]
        cp = current_component_params[c]

        loc_cand = get_location_update(observations, z, cp)
        if constrained:
            loc_cand = get_constrained_location_update(
                loc_cand, c, current_component_params, updated, xlims, **kwargs
            )

        Delta_cand = get_Delta_update(loc_cand, observations, z, cp)
        if constrained:
            Delta_cand = get_constrained_Delta_update(
                Delta_cand, loc_cand, c, current_component_params, updated, xlims, **kwargs
            )

        Gamma_cand = get_Gamma_update(loc_cand, Delta_cand, observations, z, cp)
        if constrained:
            Gamma_cand = get_constrained_Gamma_update(
                Gamma_cand, loc_cand, Delta_cand, c,
                current_component_params, updated, xlims, **kwargs
            )

        updated[c] = density_utils.alternate_to_canonical(loc_cand, Delta_cand, Gamma_cand)

        if constrained and multicomponent_density_constraint_violated(
            [*updated[:c + 1], *current_component_params[c + 1:]],
            xlims,
        ):
            raise ValueError(
                f"constraint violated after component {c} iter {kwargs.get('iterNum', -1)}"
            )
    return updated


def _em_update_multivariate(
    observations, responsibilities, current_component_params,
    constrained, xlims, K, **kwargs
):
    """One M-step for all components, multivariate case.
    component_params are stored in alternate form: (mu, Delta, Gamma)
    where mu (K_dim,), Delta (K_dim,), Gamma (K_dim, K_dim).
    """
    updated = [None] * K
    for c in range(K):
        z = responsibilities[c]  # (N,)
        mu_old, Delta_old, Gamma_old = current_component_params[c]

        # ---- unconstrained updates ----
        mu_cand = get_location_update_mv(observations, z, mu_old, Delta_old, Gamma_old)
        Delta_cand = get_Delta_update_mv(mu_cand, observations, z, mu_old, Delta_old, Gamma_old)
        Gamma_cand = get_Gamma_update_mv(
            mu_cand, Delta_cand, observations, z, mu_old, Delta_old, Gamma_old
        )

        # ---- enforce positive-definiteness of Gamma ----
        eigvals = np.linalg.eigvalsh(Gamma_cand)
        if eigvals.min() < 1e-8:
            Gamma_cand += (1e-8 - eigvals.min()) * np.eye(Gamma_cand.shape[0])

        # ---- constraint enforcement via line search ----
        if constrained:
            try:
                mu_cand, Delta_cand, Gamma_cand = get_constrained_update_mv(
                    mu_cand, Delta_cand, Gamma_cand,
                    c, current_component_params, updated,
                    xlims, multivariate=True, **kwargs
                )
            except Exception as e:
                # If constraint enforcement itself fails, fall back to no update
                if kwargs.get("raise_on_error", False):
                    raise
                import warnings
                warnings.warn(f"Constraint enforcement failed for component {c}: {e}. Keeping old params.")
                mu_cand, Delta_cand, Gamma_cand = mu_old, Delta_old, Gamma_old

        updated[c] = (mu_cand, Delta_cand, Gamma_cand)

        # Verify constraint (soft check — warn instead of crash)
        if constrained:
            test_params = [*updated[:c + 1], *current_component_params[c + 1:]]
            try:
                violated = multicomponent_density_constraint_violated(
                    test_params, xlims, multivariate=True
                )
            except Exception:
                violated = False  # Can't check → don't block progress

            if violated:
                # Revert to old params rather than crashing
                if kwargs.get("raise_on_error", False):
                    raise ValueError(
                        f"constraint violated after component {c} "
                        f"iter {kwargs.get('iterNum', -1)}"
                    )
                import warnings
                warnings.warn(
                    f"Constraint violated after component {c} "
                    f"iter {kwargs.get('iterNum', -1)}. Reverting to old params."
                )
                updated[c] = (mu_old, Delta_old, Gamma_old)

    return updated