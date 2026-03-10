import numpy as np
import scipy.stats as sps
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn, norm
from scipy.stats._multivariate import _squeeze_output
from scipy.linalg import sqrtm, inv as linalg_inv


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def is_multivariate(component_params):
    """Detect whether component_params are multivariate.
    Univariate params: (scalar, scalar, scalar)
    Multivariate params: (ndarray, ndarray, ndarray_2d)
    """
    if len(component_params) == 0:
        return False
    first = component_params[0]
    return isinstance(first[0], np.ndarray) and first[0].ndim >= 1


def _get_delta(params):
    """Univariate only: extract delta from canonical (a, loc, scale)."""
    a = params[0]
    return a / np.sqrt(1 + a**2)


# ──────────────────────────────────────────────
# Parametrization conversions — univariate
# ──────────────────────────────────────────────

def canonical_to_alternate(a, loc, scale):
    _delta = a / np.sqrt(1 + a**2)
    Delta = scale * _delta
    Gamma = scale**2 - Delta**2
    return tuple(map(float, (loc, Delta, Gamma)))


def alternate_to_canonical(loc, Delta, Gamma):
    try:
        a = np.sign(Delta) * np.sqrt(Delta**2 / Gamma)
    except ZeroDivisionError:
        raise ZeroDivisionError(
            f"Invalid skewness parameter from Delta={Delta}, Gamma={Gamma}"
        )
    if np.isinf(a) or np.isnan(a):
        raise ZeroDivisionError(
            f"Invalid skewness parameter: {a} from Delta={Delta}, Gamma={Gamma}"
        )
    scale = np.sqrt(Gamma + Delta**2)
    return tuple(map(float, (a, loc, scale)))


# ──────────────────────────────────────────────
# Parametrization conversions — multivariate
# ──────────────────────────────────────────────
# Canonical: (Lambda, mu, Omega)  Lambda (K,), mu (K,), Omega (K,K)
# Alternate: (mu, Delta, Gamma)   mu (K,), Delta (K,), Gamma (K,K)

def canonical_to_alternate_mv(Lambda, mu, Omega):
    Lambda, mu, Omega = np.asarray(Lambda), np.asarray(mu), np.asarray(Omega)
    Omega_half = np.real(sqrtm(Omega))
    delta = Lambda / np.sqrt(1 + Lambda @ Lambda)
    Delta = Omega_half @ delta
    Gamma = Omega - np.outer(Delta, Delta)
    Gamma = 0.5 * (Gamma + Gamma.T)
    return mu, Delta, Gamma


def alternate_to_canonical_mv(mu, Delta, Gamma):
    mu, Delta, Gamma = np.asarray(mu), np.asarray(Delta), np.asarray(Gamma)
    Omega = Gamma + np.outer(Delta, Delta)
    Omega = 0.5 * (Omega + Omega.T)
    Omega_half_inv = np.real(linalg_inv(sqrtm(Omega)))
    c = 1.0 - Delta @ np.linalg.solve(Omega, Delta)
    c = max(c, 1e-12)
    Lambda = Omega_half_inv @ Delta / np.sqrt(c)
    return Lambda, mu, Omega


# ──────────────────────────────────────────────
# Multivariate skew-normal density
# ──────────────────────────────────────────────

class multivariate_skewnorm:
    """MSN in canonical form (shape_vec, loc_vec, cov_matrix)."""

    def __init__(self, a, loc, cov=None):
        try:
            self.dim = len(a)
        except TypeError:
            self.dim = 1
        self.loc = np.asarray(loc)
        self.shape = np.asarray(a)
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim) if cov is None else np.asarray(cov)

    @classmethod
    def pdf(cls, x, a, loc, cov=None):
        return cls(a, loc, cov)._pdf(x)

    @classmethod
    def logpdf(cls, x, a, loc, cov=None):
        return cls(a, loc, cov)._logpdf(x)

    def _pdf(self, x):
        return np.exp(self._logpdf(x))

    def _logpdf(self, x):
        x = mvn._process_quantiles(x, self.dim)
        y = x - self.loc
        log_phi = mvn(self.mean, self.cov).logpdf(y)
        log_cdf = norm(0, 1).logcdf(np.dot(y, self.shape))
        return _squeeze_output(np.log(2) + log_phi + log_cdf)

    def rvs_fast(self, size=1):
        aCa = self.shape @ self.cov @ self.shape
        delta = (1 / np.sqrt(1 + aCa)) * self.cov @ self.shape
        cov_star = np.block(
            [[np.ones(1), delta], [delta[:, None], self.cov]]
        )
        x = mvn(np.zeros(self.dim + 1), cov_star).rvs(size)
        x0, x1 = x[:, 0], x[:, 1:]
        x1[x0 <= 0] *= -1
        return x1 + self.loc


def msn_logpdf_alternate(x, mu, Delta, Gamma):
    """Compute MSN log-density from *alternate* params (mu, Delta, Gamma).
    x: (N, K) or (K,)  |  mu: (K,)  |  Delta: (K,)  |  Gamma: (K,K)
    No missing data — all dimensions must be present.
    Returns: (N,) or scalar
    """
    mu = np.asarray(mu, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)

    # Handle scalar/1-d case
    if mu.ndim == 0:
        mu = mu.reshape(1)
    if Delta.ndim == 0:
        Delta = Delta.reshape(1)
    if Gamma.ndim < 2:
        Gamma = Gamma.reshape(1, 1)

    K = len(mu)
    Omega = Gamma + np.outer(Delta, Delta)
    Omega = 0.5 * (Omega + Omega.T)

    # Regularize Omega to ensure positive-definiteness
    eigvals = np.linalg.eigvalsh(Omega)
    if eigvals.min() < 1e-10:
        Omega += (1e-10 - eigvals.min() + 1e-10) * np.eye(K)

    # Check for any NaN/Inf in params — bail early
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(Omega))):
        x_arr = np.atleast_2d(x)
        return np.full(x_arr.shape[0], -np.inf)

    x_arr = np.atleast_2d(x)
    N = x_arr.shape[0]

    try:
        log_phi = mvn(mean=mu, cov=Omega, allow_singular=True).logpdf(x_arr)
    except (ValueError, np.linalg.LinAlgError):
        return np.full(N, -np.inf)

    try:
        Omega_inv_Delta = np.linalg.solve(Omega, Delta)
    except np.linalg.LinAlgError:
        return np.full(N, -np.inf)

    residuals = x_arr - mu
    eta = residuals @ Omega_inv_Delta
    sigma_sq = 1.0 - Delta @ Omega_inv_Delta
    sigma_sq = max(sigma_sq, 1e-12)
    log_Phi = norm.logcdf(eta / np.sqrt(sigma_sq))

    result = np.log(2) + log_phi + log_Phi

    # Ensure output is clean
    result = np.asarray(result, dtype=float).ravel()
    result[~np.isfinite(result)] = -np.inf

    if N == 1:
        return result[0]
    return result


def msn_logpdf_alternate_missing(x, mu, Delta, Gamma):
    """Compute MSN log-density handling NaN (missing dimensions) per observation.

    Uses the MSN marginal property: for observed set S,
        X_S ~ MSN(mu_S, Delta_S, Gamma_{SS})

    x: (N, K) with NaN for missing dimensions
    mu: (K,)  |  Delta: (K,)  |  Gamma: (K, K)
    Returns: (N,)
    """
    mu, Delta, Gamma = np.asarray(mu, dtype=float), np.asarray(Delta, dtype=float), np.asarray(Gamma, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    N, K = x.shape
    log_pdf = np.zeros(N)  # fully-missing rows get log-density 0 (density 1)

    obs_mask = ~np.isnan(x)  # (N, K)
    # Group by observed pattern for efficiency
    patterns = {}
    for j in range(N):
        key = tuple(obs_mask[j])
        if key not in patterns:
            patterns[key] = []
        patterns[key].append(j)

    for pattern_key, indices in patterns.items():
        obs_dims = np.array([i for i, o in enumerate(pattern_key) if o])
        if len(obs_dims) == 0:
            continue
        idx = np.array(indices)

        mu_s = mu[obs_dims]
        Delta_s = Delta[obs_dims]
        Gamma_s = Gamma[np.ix_(obs_dims, obs_dims)]

        x_s = x[np.ix_(idx, obs_dims)]

        try:
            lp = msn_logpdf_alternate(x_s, mu_s, Delta_s, Gamma_s)
            lp = np.atleast_1d(np.asarray(lp, dtype=float))
            lp[~np.isfinite(lp)] = -np.inf
            log_pdf[idx] = lp
        except Exception:
            log_pdf[idx] = -np.inf

    return log_pdf


# ──────────────────────────────────────────────
# Mixture density helpers (unified)
# ──────────────────────────────────────────────

def _single_component_logpdf(x, params, multivariate=False):
    """Log-pdf of a single component for either parameterization.
    For multivariate with missing data (NaN), uses marginal density.
    """
    if not multivariate:
        return sps.skewnorm.logpdf(x, *params)
    else:
        if isinstance(params[2], np.ndarray) and params[2].ndim == 2:
            # alternate form (mu, Delta, Gamma)
            x_arr = np.atleast_2d(x)
            if np.isnan(x_arr).any():
                return msn_logpdf_alternate_missing(x_arr, *params)
            return msn_logpdf_alternate(x_arr, *params)
        else:
            # canonical form (Lambda, mu, Omega)
            return multivariate_skewnorm.logpdf(x, *params)


def mixture_pdf(x, params, weights, multivariate=False):
    """Log-pdf of the full mixture."""
    return logsumexp(
        log_joint_densities(x, params, weights, multivariate=multivariate), axis=0
    )


def get_log_fPB(X, params, weights, multivariate=False):
    return np.log(
        joint_densities(X, params, weights, multivariate=multivariate).sum(0)
    )


def joint_densities(x, params, weights, multivariate=False):
    return np.array([
        w * np.exp(_single_component_logpdf(x, p, multivariate))
        for p, w in zip(params, weights)
    ])


def log_joint_densities(x, params, weights, multivariate=False):
    weights = np.asarray(weights)
    log_pdfs = np.array([
        _single_component_logpdf(x, p, multivariate) for p in params
    ])
    with np.errstate(divide="ignore"):
        log_w = np.log(weights)
    log_w[weights == 0] = -np.inf
    return log_w[:, None] + log_pdfs


# ──────────────────────────────────────────────
# Responsibilities
# ──────────────────────────────────────────────

def component_posteriors(X, canonical_params, individual_sample_weights, multivariate=False):
    individual_sample_weights = np.array(individual_sample_weights)[:, None]
    assert len(canonical_params) == individual_sample_weights.shape[0]

    if not multivariate:
        log_pdfs = np.stack(
            [sps.skewnorm.logpdf(X.ravel(), *p) for p in canonical_params], axis=0
        )
    else:
        X_2d = np.atleast_2d(X)
        N = X_2d.shape[0]
        log_pdfs_list = []
        for p in canonical_params:
            lp = _single_component_logpdf(X_2d, p, multivariate=True)
            lp = np.atleast_1d(np.asarray(lp, dtype=float)).ravel()
            # Ensure correct length
            if len(lp) != N:
                lp = np.full(N, -np.inf)
            lp[~np.isfinite(lp)] = -np.inf
            log_pdfs_list.append(lp)
        log_pdfs = np.stack(log_pdfs_list, axis=0)  # (K, N)

    with np.errstate(divide="ignore"):
        numerators = log_pdfs + np.log(individual_sample_weights)
    denom = logsumexp(numerators, axis=0)
    P = np.exp(numerators - denom[None])
    P[np.isnan(P)] = 0
    return P


# ──────────────────────────────────────────────
# Likelihood
# ──────────────────────────────────────────────

def get_likelihood(observations, sample_indicators, component_params, weights, multivariate=False):
    if component_params is None or weights is None:
        return -np.inf
    L = 0.0
    for s, mask in enumerate(sample_indicators.T):
        X = observations[mask]
        lw = log_joint_densities(X, component_params, weights[s], multivariate=multivariate)
        L += logsumexp(lw, axis=0).sum()
    return L


def get_sample_likelihood(observations, sample_indicators, component_params, weights, multivariate=False):
    if component_params is None or weights is None:
        return [-np.inf] * sample_indicators.shape[1]
    Ls = [0.0] * sample_indicators.shape[1]
    for s, mask in enumerate(sample_indicators.T):
        X = observations[mask]
        lw = log_joint_densities(X, component_params, weights[s], multivariate=multivariate)
        Ls[s] += logsumexp(lw, axis=0).sum()
    return np.array(Ls)