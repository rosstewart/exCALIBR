"""
Density utilities for mixtures of multivariate skew-normal distributions.

Supports both the restricted MSN (q=1, Delta is a vector) and the CFUSN
(q>=1, Delta is a matrix) parameterizations.

CFUSN alternate parameterization:
    mu    : (p,)      location
    Delta : (p, q)    skewness matrix  (reduces to (p,) vector when q=1)
    Gamma : (p, p)    residual covariance

Omega = Gamma + Delta @ Delta.T
f(x) = 2^q * phi_p(x; mu, Omega) * Phi_q(Delta.T @ Omega^{-1} (x-mu); 0, D)
where D = I_q - Delta.T @ Omega^{-1} @ Delta

References:
    Lin (2009) - MLE for multivariate skew normal mixture models
    Jain et al. (2019) - Identifiability of two-component skew normal mixtures
    Sahu et al. (2003) - A new class of multivariate skew distributions
"""

import numpy as np
import scipy.stats as sps
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn, norm
from scipy.stats._multivariate import _squeeze_output
from scipy.linalg import sqrtm, inv as linalg_inv
from scipy.linalg import cho_factor, cho_solve

_LOG2PI = np.log(2 * np.pi)
_LOG2   = np.log(2.0)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _delta_ndim(Delta):
    """Return the latent dimension q from Delta."""
    Delta = np.asarray(Delta)
    if Delta.ndim <= 1:
        return 1
    return Delta.shape[1]


def _ensure_matrix_delta(Delta):
    """Ensure Delta is (p, q). If (p,), reshape to (p, 1)."""
    Delta = np.asarray(Delta, dtype=float)
    if Delta.ndim == 1:
        return Delta.reshape(-1, 1)
    return Delta


def is_multivariate(component_params):
    """Detect whether component_params are multivariate.
    Univariate params: (scalar, scalar, scalar)
    Multivariate params: (ndarray, ndarray, ndarray_2d)
    """
    if len(component_params) == 0:
        return False
    first = component_params[0]
    return isinstance(first[0], np.ndarray) and first[0].ndim >= 1


def is_cfusn(component_params):
    """Detect whether params use CFUSN (q > 1) based on Delta shape."""
    if not is_multivariate(component_params):
        return False
    first = component_params[0]
    Delta = np.asarray(first[1])
    return Delta.ndim == 2 and Delta.shape[1] > 1


def get_q(component_params):
    """Get latent dimension q from component params."""
    if not is_multivariate(component_params):
        return 1
    first = component_params[0]
    return _delta_ndim(first[1])


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
# For q=1:  canonical (Lambda_vec, mu, Omega) <-> alternate (mu, Delta_vec, Gamma)
# For q>1 (CFUSN): canonical (Lambda_mat, mu, Omega) <-> alternate (mu, Delta_mat, Gamma)
#   Lambda_mat: (p, q),  Omega: (p, p)
#   Delta_mat:  (p, q),  Gamma: (p, p)

def canonical_to_alternate_mv(Lambda, mu, Omega):
    """Convert canonical to alternate. Works for both q=1 (vector) and q>1 (matrix)."""
    Lambda = np.asarray(Lambda, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Omega = np.asarray(Omega, dtype=float)

    if Lambda.ndim == 1:
        # q = 1 case (restricted MSN)
        Omega_half = np.real(sqrtm(Omega))
        delta = Lambda / np.sqrt(1 + Lambda @ Lambda)
        Delta = Omega_half @ delta
        Gamma = Omega - np.outer(Delta, Delta)
        Gamma = 0.5 * (Gamma + Gamma.T)
        return mu, Delta, Gamma
    else:
        # q > 1 case (CFUSN)
        # From stochastic representation: X = mu + Lambda*tau + U, U ~ N(0, Sigma)
        # Omega = Sigma + Lambda @ Lambda.T
        # In alternate form: Delta = Lambda, Gamma = Sigma = Omega - Lambda @ Lambda.T
        # (This follows from the Sahu et al. parameterization)
        Delta = Lambda.copy()
        Gamma = Omega - Lambda @ Lambda.T
        Gamma = 0.5 * (Gamma + Gamma.T)
        return mu, Delta, Gamma


def alternate_to_canonical_mv(mu, Delta, Gamma):
    """Convert alternate to canonical. Works for both q=1 and q>1."""
    mu = np.asarray(mu, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)

    if Delta.ndim == 1:
        # q = 1
        Omega = Gamma + np.outer(Delta, Delta)
        Omega = 0.5 * (Omega + Omega.T)
        Omega_half_inv = np.real(linalg_inv(sqrtm(Omega)))
        c = 1.0 - Delta @ np.linalg.solve(Omega, Delta)
        c = max(c, 1e-12)
        Lambda = Omega_half_inv @ Delta / np.sqrt(c)
        return Lambda, mu, Omega
    else:
        # q > 1 (CFUSN): Lambda = Delta, Omega = Gamma + Delta @ Delta.T
        Lambda = Delta.copy()
        Omega = Gamma + Delta @ Delta.T
        Omega = 0.5 * (Omega + Omega.T)
        return Lambda, mu, Omega


# ──────────────────────────────────────────────
# Multivariate normal CDF helpers
# ──────────────────────────────────────────────

def _mvn_logcdf(upper, mean, cov):
    """Log of multivariate normal CDF P(X <= upper) for X ~ N(mean, cov).
    upper: (q,) or scalar
    Returns: scalar log-probability
    """
    upper = np.atleast_1d(np.asarray(upper, dtype=float))
    mean = np.atleast_1d(np.asarray(mean, dtype=float))
    q = len(upper)
    if q == 1:
        sigma = np.sqrt(np.asarray(cov).ravel()[0]) if np.asarray(cov).ndim > 0 else np.sqrt(float(cov))
        return norm.logcdf((upper[0] - mean[0]) / max(sigma, 1e-15))
    # For q >= 2, use scipy's mvn CDF (not log, so take log after)
    try:
        val = mvn(mean=mean, cov=cov, allow_singular=True).cdf(upper)
        if val <= 0:
            return -np.inf
        return np.log(val)
    except Exception:
        return -np.inf



def _mvn_logcdf_batch(uppers, mean, cov):
    """Batch log-CDF  P(X <= uppers[j])  for X ~ N(mean, cov).

    uppers : (N, q)  |  mean : (q,)  |  cov : (q, q)
    Returns : (N,)

    KEY FIX: scipy mvn.cdf is vectorized over its first axis — pass the
    entire (N, q) array in one call instead of looping over N rows.
    """
    N, q = uppers.shape
    if q == 1:
        s = np.sqrt(max(float(np.asarray(cov).ravel()[0]), 1e-15))
        return norm.logcdf((uppers[:, 0] - mean[0]) / s)
    try:
        rv   = mvn(mean=mean, cov=cov, allow_singular=True)
        vals = rv.cdf(uppers)                        # (N,) — one call, no loop
        return np.log(np.maximum(vals, 1e-300))
    except Exception:
        return np.full(N, -np.inf)



# ──────────────────────────────────────────────
# CFUSN density (generalized, handles q >= 1)
# ──────────────────────────────────────────────

def cfusn_logpdf_alternate(x, mu, Delta, Gamma):
    """CFUSN log-density in alternate params.

    KEY FIXES vs original:
      1. Single cho_factor reused for log|Ω|, Mahalanobis, and Ω⁻¹Δ.
      2. eigvalsh deferred to exception path (only run when Cholesky fails).
      3. _mvn_logcdf_batch now vectorized (no N-loop inside).
      4. No scipy mvn object created for the Gaussian part.
    """
    mu    = np.asarray(mu,    dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)

    if Delta.ndim <= 1:                              # q=1 fast-path unchanged
        return msn_logpdf_alternate(x, mu, Delta.ravel(), Gamma)

    p, q = Delta.shape
    if Gamma.ndim < 2:
        Gamma = Gamma.reshape(1, 1)

    Omega  = Gamma + Delta @ Delta.T
    Omega += Omega.T; Omega *= 0.5

    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    N = x_arr.shape[0]

    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(Omega))):
        return np.full(N, -np.inf)

    # ── single Cholesky covers log|Ω|, maha, and Ω⁻¹Δ ──────────────────
    try:
        cf = cho_factor(Omega, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        emin = np.linalg.eigvalsh(Omega).min()       # only on failure
        Omega += (abs(emin) + 1e-10) * np.eye(p)
        try:
            cf = cho_factor(Omega, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.full(N, -np.inf)

    log_det         = 2.0 * np.log(np.diag(cf[0])).sum()
    residuals       = x_arr - mu                                       # (N, p)
    sol             = cho_solve(cf, residuals.T, check_finite=False)   # (p, N)
    maha            = np.einsum('ni,in->n', residuals, sol)             # (N,)
    log_phi         = -0.5 * (p * _LOG2PI + log_det + maha)

    Omega_inv_Delta = cho_solve(cf, Delta, check_finite=False)          # (p, q)
    D               = np.eye(q) - Delta.T @ Omega_inv_Delta             # (q, q)
    D               = 0.5 * (D + D.T)
    eig_D           = np.linalg.eigvalsh(D)
    if eig_D.min() < 1e-10:
        D += (1e-10 - eig_D.min() + 1e-10) * np.eye(q)

    eta     = residuals @ Omega_inv_Delta                               # (N, q)
    log_Phi = _mvn_logcdf_batch(eta, np.zeros(q), D)                   # (N,) vectorized

    result = np.asarray(q * _LOG2 + log_phi + log_Phi, dtype=float).ravel()
    result[~np.isfinite(result)] = -np.inf
    return result[0] if N == 1 else result



def cfusn_logpdf_alternate_missing(x, mu, Delta, Gamma):
    """Compute CFUSN log-density handling NaN (missing dims) per observation.

    Uses the CFUSN marginal property: for observed set S,
        X_S ~ CFUSN(mu_S, Delta_S, Gamma_{SS})
    where Delta_S is rows of Delta corresponding to S.

    Parameters
    ----------
    x : (N, p) with NaN for missing
    mu : (p,)
    Delta : (p, q) or (p,) for q=1
    Gamma : (p, p)

    Returns
    -------
    (N,) log-density
    """
    mu = np.asarray(mu, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)

    if Delta.ndim <= 1:
        return msn_logpdf_alternate_missing(x, mu, Delta, Gamma)

    x = np.atleast_2d(np.asarray(x, dtype=float))
    N, p_dim = x.shape
    log_pdf = np.zeros(N)

    # Vectorised pattern partition (was a per-row Python dict loop).
    obs_mask = ~np.isnan(x)
    patterns, inverse = np.unique(obs_mask, axis=0, return_inverse=True)

    for pi, pattern in enumerate(patterns):
        obs_dims = np.where(pattern)[0]
        if len(obs_dims) == 0:
            continue
        idx = np.where(inverse == pi)[0]

        mu_s = mu[obs_dims]
        Delta_s = Delta[obs_dims, :]          # (|S|, q)
        Gamma_s = Gamma[np.ix_(obs_dims, obs_dims)]
        x_s = x[np.ix_(idx, obs_dims)]

        try:
            lp = cfusn_logpdf_alternate(x_s, mu_s, Delta_s, Gamma_s)
            lp = np.atleast_1d(np.asarray(lp, dtype=float))
            lp[~np.isfinite(lp)] = -np.inf
            log_pdf[idx] = lp
        except Exception:
            log_pdf[idx] = -np.inf

    return log_pdf


# ──────────────────────────────────────────────
# Original q=1 MSN density (kept for backward compat & speed)
# ──────────────────────────────────────────────

# ─── msn_logpdf_alternate (q=1) — same Cholesky treatment ───────────────────
def msn_logpdf_alternate(x, mu, Delta, Gamma):
    """q=1 MSN. Cholesky-based, no scipy MVN object, eigvalsh only on failure."""
    mu    = np.asarray(mu,    dtype=float).ravel()
    Delta = np.asarray(Delta, dtype=float).ravel()
    Gamma = np.atleast_2d(np.asarray(Gamma, dtype=float))
    K     = len(mu)

    Omega  = Gamma + np.outer(Delta, Delta)
    Omega += Omega.T; Omega *= 0.5

    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    N = x_arr.shape[0]

    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(Omega))):
        return np.full(N, -np.inf)

    try:
        cf = cho_factor(Omega, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        emin = np.linalg.eigvalsh(Omega).min()
        Omega += (abs(emin) + 1e-10) * np.eye(K)
        try:
            cf = cho_factor(Omega, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return np.full(N, -np.inf)

    log_det         = 2.0 * np.log(np.diag(cf[0])).sum()
    residuals       = x_arr - mu
    sol             = cho_solve(cf, residuals.T, check_finite=False)
    maha            = np.einsum('ni,in->n', residuals, sol)
    log_phi         = -0.5 * (K * _LOG2PI + log_det + maha)

    Omega_inv_Delta = cho_solve(cf, Delta, check_finite=False)
    eta             = residuals @ Omega_inv_Delta
    sigma_sq        = max(1.0 - Delta @ Omega_inv_Delta, 1e-12)
    log_Phi         = norm.logcdf(eta / np.sqrt(sigma_sq))

    result = np.asarray(_LOG2 + log_phi + log_Phi, dtype=float).ravel()
    result[~np.isfinite(result)] = -np.inf
    return result[0] if N == 1 else result


def msn_logpdf_alternate_missing(x, mu, Delta, Gamma):
    """MSN log-density with NaN handling for q=1. Delta is (p,) vector."""
    mu = np.asarray(mu, dtype=float)
    Delta = np.asarray(Delta, dtype=float).ravel()
    Gamma = np.asarray(Gamma, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    N, K = x.shape
    log_pdf = np.zeros(N)

    # Vectorised pattern partition (was a per-row Python dict loop).
    obs_mask = ~np.isnan(x)
    patterns, inverse = np.unique(obs_mask, axis=0, return_inverse=True)

    for pi, pattern in enumerate(patterns):
        obs_dims = np.where(pattern)[0]
        if len(obs_dims) == 0:
            continue
        idx = np.where(inverse == pi)[0]
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
# Unified single-component log-pdf
# ──────────────────────────────────────────────

def _single_component_logpdf(x, params, multivariate=False):
    """Log-pdf of a single component.

    For multivariate, params = (mu, Delta, Gamma) in alternate form.
    Delta can be (p,) for q=1 or (p, q) for CFUSN.
    """
    if not multivariate:
        return sps.skewnorm.logpdf(x, *params)
    else:
        mu, Delta, Gamma = params
        Delta_arr = np.asarray(Delta)
        if isinstance(Gamma, np.ndarray) and Gamma.ndim == 2:
            x_arr = np.atleast_2d(x)
            has_nan = np.isnan(x_arr).any()
            if Delta_arr.ndim == 2 and Delta_arr.shape[1] > 1:
                # CFUSN path
                if has_nan:
                    return cfusn_logpdf_alternate_missing(x_arr, mu, Delta, Gamma)
                return cfusn_logpdf_alternate(x_arr, mu, Delta, Gamma)
            else:
                # q=1 restricted MSN path
                Delta_vec = Delta_arr.ravel()
                if has_nan:
                    return msn_logpdf_alternate_missing(x_arr, mu, Delta_vec, Gamma)
                return msn_logpdf_alternate(x_arr, mu, Delta_vec, Gamma)
        else:
            # canonical form fallback
            return multivariate_skewnorm.logpdf(x, *params)


# ──────────────────────────────────────────────
# Multivariate skew-normal class (canonical, q=1 only)
# ──────────────────────────────────────────────

class multivariate_skewnorm:
    """MSN in canonical form (shape_vec, loc_vec, cov_matrix). q=1 only."""

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


# ──────────────────────────────────────────────
# Mixture density helpers (unified)
# ──────────────────────────────────────────────

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

def component_posteriors(X, canonical_params, individual_sample_weights,
                          multivariate=False, cached_log_pdfs=None):
    """Compute responsibilities P(component | x).

    Parameters
    ----------
    cached_log_pdfs : (K, N_s) array or None
        If provided, skips the density evaluation (which is the dominant
        cost) and reuses the supplied log-pdfs. Caller is responsible for
        ensuring the cache matches ``X`` and ``canonical_params``.
    """
    individual_sample_weights = np.array(individual_sample_weights)[:, None]
    assert len(canonical_params) == individual_sample_weights.shape[0]

    if cached_log_pdfs is not None:
        log_pdfs = cached_log_pdfs
    elif not multivariate:
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
            if len(lp) != N:
                lp = np.full(N, -np.inf)
            lp[~np.isfinite(lp)] = -np.inf
            log_pdfs_list.append(lp)
        log_pdfs = np.stack(log_pdfs_list, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # invalid='ignore' suppresses the (-inf) - (-inf) = NaN warning that
        # fires when an observation has zero density under all components;
        # the `P[np.isnan(P)] = 0` line below is the canonical fix.
        numerators = log_pdfs + np.log(individual_sample_weights)
        denom = logsumexp(numerators, axis=0)
        P = np.exp(numerators - denom[None])
    P[np.isnan(P)] = 0
    return P


# ──────────────────────────────────────────────
# Likelihood
# ──────────────────────────────────────────────

def get_likelihood(observations, sample_indicators, component_params, weights,
                   multivariate=False, sample_weights=None):
    """Total log-likelihood under the mixture.

    When ``sample_weights`` (per-obs, length N) is given, returns the
    sample-weighted LL  ``Σ_n sw_n · log Σ_k W[s(n),k] f_k(x_n)``. With
    ``sw_n = 1`` this reduces to the standard LL — bit-identical
    behaviour for callers that don't pass the kwarg.

    Used by EM under ``sample_balance_beta>0``: the M-step then
    maximises the same weighted Q-function we evaluate here, restoring
    the standard EM monotonicity guarantee on this objective.
    """
    if component_params is None or weights is None:
        return -np.inf
    L = 0.0
    for s, mask in enumerate(sample_indicators.T):
        X = observations[mask]
        lw = log_joint_densities(X, component_params, weights[s], multivariate=multivariate)
        log_mix = logsumexp(lw, axis=0)
        if sample_weights is not None:
            L += float(np.sum(np.asarray(sample_weights)[mask] * log_mix))
        else:
            L += float(log_mix.sum())
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