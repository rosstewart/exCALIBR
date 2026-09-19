"""
EM update steps for mixtures of multivariate skew-normal distributions.

Supports both restricted MSN (q=1) and CFUSN (q>=1).

For CFUSN, the E-step requires moments of a q-dimensional truncated normal:
    T | X=x ~ TN_q(m, S, R^q_+)
    eta  = E[T | x]           shape (q,)
    Psi  = E[T T' | x]        shape (q, q)

These are computed via Monte Carlo (rejection sampling + Gibbs fallback).

M-step follows Lin (2009) equations:
    mu_new  = (sum z_j y_j - Delta sum z_j eta_j) / sum z_j
    Delta_new = (sum z_j (y_j - mu) eta_j') (sum z_j Psi_j)^{-1}
    Gamma_new = (1/sum z_j) sum z_j [(y-mu-Delta eta)(y-mu-Delta eta)' + Delta(Psi - eta eta')Delta']

References:
    Lin (2009) - MLE for multivariate skew normal mixture models, §3
    Tallis (1961) - Moments of truncated multi-normal
"""

from . import density_utils
from . import separation
from .constraints import (
    multicomponent_density_constraint_violated,
    build_constraint_grids,
    _logpdf_for_check,
    _adjacent_pair_violated,
)
from typing import List, Tuple, Any
import numpy as np
import scipy.stats as sps
from scipy.stats import truncnorm
from scipy.special import logsumexp, owens_t
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm


def _bvn_cdf_owens(h, k, rho):
    """Exact standard bivariate normal CDF P(Z1<=h, Z2<=k), corr=rho, via
    Owen's T function (Owen 1956). Vectorized over h/k; rho scalar/shared.

    Validated to ~2.22e-16 against scipy.stats.multivariate_normal.cdf over
    500 random cases, and ~40-4900x faster per call at the batch sizes real
    fits use (this is what makes the exact q=2 E-step below cheap enough to
    run every EM iteration instead of resorting to Monte Carlo).
    """
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    denom = np.sqrt(max(1 - rho ** 2, 1e-300))

    # The T-function argument a1=(k-rho*h)/(h*denom) is genuinely singular
    # at h=0 (diverges to +-inf when k!=0, or needs both h,k->0 together
    # when k==0 too) -- substituting a tiny epsilon for an exactly-zero h
    # must be done in BOTH the numerator and denominator (not just the
    # denominator) so the substitution reproduces the correct limiting
    # value of a1 instead of silently zeroing out the correlation term.
    # (Caught via a symmetric mean=(0,0) sanity check: using the literal
    # zero in the numerator gave Phi_2(0,0;0)=0.5 instead of the true
    # 0.25 -- alpha=0 is a common, not edge-case, state during EM.)
    h_safe = np.where(h == 0, 1e-12, h)
    k_safe = np.where(k == 0, 1e-12, k)

    a1 = (k_safe - rho * h_safe) / (h_safe * denom)
    a2 = (h_safe - rho * k_safe) / (k_safe * denom)
    T1 = owens_t(h, a1)
    T2 = owens_t(k, a2)

    hk = h * k
    delta = np.where((hk > 0) | ((hk == 0) & (h + k >= 0)), 0.0, 0.5)
    return 0.5 * (norm.cdf(h) + norm.cdf(k)) - T1 - T2 - delta


# ══════════════════════════════════════════════
# Truncated-normal moments — scalar (q=1, unchanged)
# ══════════════════════════════════════════════

def _psd_floor_or_reject(Gamma_cand, floor=1e-8):
    """Enforce positive-definiteness of a candidate Gamma, or signal that
    the candidate is unusable.

    Returns (Gamma_fixed, ok). ok=False means Gamma_cand is degenerate
    (non-finite, or LAPACK's eigensolver itself failed to converge on it --
    both observed in practice on ill-conditioned M-step candidates, e.g.
    from a near-zero-responsibility component) and the caller should reject
    this candidate and keep the component's previous parameters, rather
    than propagate garbage (or an uncaught LinAlgError) forward.

    np.linalg.eigvalsh does NOT reliably raise on bad input -- e.g. an
    all-Inf matrix returns [nan, nan] silently rather than raising, which a
    bare `except LinAlgError` would miss (`nan < floor` is False, so the
    ridge-fix below wouldn't even trigger) -- hence the explicit
    isfinite checks before AND after the eigendecomposition.
    """
    if not np.all(np.isfinite(Gamma_cand)):
        return Gamma_cand, False
    try:
        eigvals = np.linalg.eigvalsh(Gamma_cand)
    except np.linalg.LinAlgError:
        return Gamma_cand, False
    if not np.all(np.isfinite(eigvals)):
        return Gamma_cand, False
    if eigvals.min() < floor:
        Gamma_cand = Gamma_cand + (floor - eigvals.min()) * np.eye(Gamma_cand.shape[0])
    return Gamma_cand, True


def trunc_norm_moments(mu, sigma):
    """Moments of TN(mu, sigma^2, R+). mu, sigma are arrays (N,)."""
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
    """Univariate SN: params = (a, loc, scale) canonical."""
    _delta = density_utils._get_delta(component_params)
    loc, scale = component_params[1:]
    tn_loc = _delta / scale * (observations - loc)
    tn_scale = np.sqrt(1 - _delta**2)
    return trunc_norm_moments(tn_loc, tn_scale)


def get_truncated_normal_moments_mv(observations, mu, Delta, Gamma):
    """Restricted MSN (q=1): alternate params, Delta is (p,) vector.
    Returns v: (N,), w: (N,) — scalar moments per observation.
    """
    Delta = np.asarray(Delta).ravel()
    Omega = Gamma + np.outer(Delta, Delta)
    Omega = 0.5 * (Omega + Omega.T)
    Omega_inv_Delta = np.linalg.solve(Omega, Delta)
    residuals = observations - mu
    eta = residuals @ Omega_inv_Delta
    sigma_sq = max(1.0 - Delta @ Omega_inv_Delta, 1e-12)
    sigma = np.sqrt(sigma_sq)
    return trunc_norm_moments(eta, sigma)


def get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma):
    """Restricted MSN (q=1) with NaN handling. Delta is (p,) vector.
    Returns v: (N,), w: (N,).
    """
    Delta = np.asarray(Delta).ravel()
    observations = np.atleast_2d(observations)
    N, K = observations.shape
    v = np.zeros(N)
    w = np.zeros(N)

    obs_mask = ~np.isnan(observations)
    patterns, inverse = np.unique(obs_mask, axis=0, return_inverse=True)

    for pi, pattern in enumerate(patterns):
        obs_dims = np.where(pattern)[0]
        if len(obs_dims) == 0:
            idx = np.where(inverse == pi)[0]
            v[idx] = np.sqrt(2 / np.pi)
            w[idx] = 1.0
            continue
        idx = np.where(inverse == pi)[0]
        mu_s = mu[obs_dims]
        Delta_s = Delta[obs_dims]
        Gamma_s = Gamma[np.ix_(obs_dims, obs_dims)]
        x_s = observations[np.ix_(idx, obs_dims)]
        v[idx], w[idx] = get_truncated_normal_moments_mv(x_s, mu_s, Delta_s, Gamma_s)

    return v, w


# ══════════════════════════════════════════════
# Truncated multivariate normal moments (q >= 1)
#   T | X=x ~ TN_q(m, S, R^q_+)
#   Uses Monte Carlo with rejection sampling + Gibbs fallback
# ══════════════════════════════════════════════

def _gibbs_sample_tn_q(mean, cov, n_samples, n_burnin=50, rng=None):
    """Gibbs sampler for TN_q(mean, cov, R^q_+).

    Parameters
    ----------
    mean : (q,)
    cov : (q, q)
    n_samples : int
    n_burnin : int

    Returns
    -------
    samples : (n_samples, q)
    """
    if rng is None:
        rng = np.random.RandomState()
    q = len(mean)
    samples = np.zeros((n_samples, q))
    # Start in positive orthant
    x = np.maximum(mean, 0.01 * np.ones(q))

    # Precompute conditional parameters
    # For j-th component given rest:
    # x_j | x_{-j} ~ TN( mu_cond, var_cond, [0, inf) )
    cov_inv = np.linalg.inv(cov + 1e-12 * np.eye(q))

    total = n_burnin + n_samples
    for t in range(total):
        for j in range(q):
            others = [i for i in range(q) if i != j]
            # Conditional mean and variance
            var_j = 1.0 / cov_inv[j, j]
            mu_j = mean[j] - var_j * cov_inv[j, others] @ (x[others] - mean[others])
            sd_j = np.sqrt(max(var_j, 1e-15))
            # Sample from TN(mu_j, var_j, [0, inf))
            a = -mu_j / sd_j
            x[j] = truncnorm.rvs(a, 1e10, loc=mu_j, scale=sd_j, random_state=rng)

        if t >= n_burnin:
            samples[t - n_burnin] = x.copy()

    return samples


# def _mc_truncated_mvn_moments(means, cov, n_mc=500, rng=None):
#     """Compute E[T] and E[TT'] for T ~ TN_q(mean_j, cov, R^q_+)
#     for each observation j.

#     Parameters
#     ----------
#     means : (N, q) — conditional means per observation
#     cov : (q, q)   — shared conditional covariance
#     n_mc : int      — MC samples per observation

#     Returns
#     -------
#     eta : (N, q)     — E[T | x_j]
#     Psi : (N, q, q)  — E[TT' | x_j]
#     """
#     if rng is None:
#         rng = np.random.RandomState(42)
#     N, q = means.shape
#     eta = np.zeros((N, q))
#     Psi = np.zeros((N, q, q))

#     # Cholesky for sampling
#     try:
#         L = np.linalg.cholesky(cov + 1e-12 * np.eye(q))
#     except np.linalg.LinAlgError:
#         eigv = np.linalg.eigvalsh(cov)
#         cov_reg = cov + (1e-8 - min(eigv.min(), 0)) * np.eye(q)
#         L = np.linalg.cholesky(cov_reg)

#     # Try vectorized rejection sampling first
#     # Generate a large batch of standard normals
#     oversample = max(n_mc * 10, 2000)
#     Z_batch = rng.randn(oversample, q)
#     candidates_base = Z_batch @ L.T  # (oversample, q) — zero-mean samples

#     for j in range(N):
#         m = means[j]
#         candidates = candidates_base + m  # shift to mean
#         valid = (candidates > 0).all(axis=1)
#         n_valid = valid.sum()

#         if n_valid >= n_mc:
#             # Use rejection samples
#             good = candidates[valid][:n_mc]
#         elif n_valid >= max(20, n_mc // 5):
#             # Use what we have (smaller sample)
#             good = candidates[valid]
#         else:
#             # Fall back to Gibbs
#             good = _gibbs_sample_tn_q(m, cov, n_mc, n_burnin=50, rng=rng)

#         eta[j] = good.mean(axis=0)
#         Psi[j] = (good.T @ good) / len(good)

#     return eta, Psi


def _mc_truncated_mvn_moments(means, cov, n_mc=500, rng=None):
    """E[T] and E[TT'] for T ~ TN_q(mean_j, cov, R^q_+).

    q=2 fast path: EXACT closed form (Kan & Robotti 2017, JCGS, Theorem 1
    recursion for F^n_kappa(a,b;mu,Sigma), specialized to n=2, first
    orthant), not an approximation -- see _bvn_cdf_owens above for the
    bivariate-CDF piece that makes this cheap enough to call every EM
    iteration. Validated to ~1e-4 against 2M-sample Monte Carlo across a
    range of means/correlations/truncation severity.

    This replaces an earlier ad-hoc formula for the cross moment
    E[T1 T2] (independent-marginals mean product + a linear correlation
    correction) that was not guaranteed to satisfy the Cauchy-Schwarz
    bound every genuine second-moment matrix must satisfy
    (Psi[0,1]^2 <= Psi[0,0]*Psi[1,1]) -- confirmed on real TP53 production
    data that ~30% of individual observations violated it, occasionally
    producing a responsibility-weighted Psi_sum with a negative eigenvalue
    and causing the M-step's Delta update to blow up by orders of
    magnitude. The exact formula is PSD by construction (it's a genuine
    second moment, not an approximation to one), so no post-hoc clipping
    is needed for that failure mode; it also no longer systematically
    suppresses fitted skew the way the old approximation did.

    For q != 2 the original MC path is used unchanged.

    Mirrored in fit_utils/jax_batch/batch_em_cfusn.py's
    `_component_moments_and_logpdf` (JAX port, for the opt-in GPU path) --
    keep both in sync if this formula ever changes.
    """
    if rng is None:
        rng = np.random.RandomState()

    N, q = means.shape

    # ── q=2 fast path — EXACT closed form ───────────────────────────────
    if q == 2:
        cov  = 0.5 * (cov + cov.T)
        eig  = np.linalg.eigvalsh(cov)
        if eig.min() < 1e-12:
            cov = cov + (1e-12 - eig.min() + 1e-12) * np.eye(2)

        mu1, mu2   = means[:, 0], means[:, 1]
        s1sq, s2sq = cov[0, 0], cov[1, 1]
        s12        = cov[0, 1]
        s1, s2     = np.sqrt(s1sq), np.sqrt(s2sq)
        rho        = np.clip(s12 / (s1 * s2), -0.999999, 0.999999)

        alpha1, alpha2 = -mu1 / s1, -mu2 / s2
        # L = P(X1>0, X2>0) computed as a single direct bivariate-CDF call
        # (via the sign-flip identity P(Z1>a1,Z2>a2) = Phi_2(-a1,-a2;rho)),
        # NOT as 1-Phi(a1)-Phi(a2)+Phi_2(a1,a2;rho): that difference-of-four-
        # terms form catastrophically cancels in the truncation-heavy tail
        # (each term near 0 or 1, true result near 0), and the resulting
        # floating-point noise is a REAL, non-negligible fraction of the
        # true value for a large range of realistic (mu, cov) -- confirmed
        # directly (deep-tail stress test, N=3000 draws): the old form's
        # negative-noise excursions have median magnitude ~1e-16 regardless
        # of the true L's scale (i.e. can swamp a true L of 1e-8 or 1e-10
        # entirely), while this direct form's excursions are ~1e-20,
        # genuine machine-epsilon dust only when the true L is already
        # unrepresentable. This was the root cause of a real production
        # failure mode: as a poorly-supported component's shape drifted
        # toward extreme skew, a growing fraction of rows hit the old
        # floor, and that accumulating floor-induced bias eventually
        # flipped the EM step's net likelihood delta negative, tripping
        # the hard likelihood-decrease guard (fit.py) and discarding the
        # whole fit.
        L = _bvn_cdf_owens(-alpha1, -alpha2, rho)
        # Floor only against genuine floating-point noise (this form's own
        # residual negative dust, ~1e-20 scale -- see above), not as a
        # large-magnitude safety net; downstream overflow protection is
        # the CLIP/nan_to_num on eta/Psi below, so this floor no longer
        # needs to double as that safety net.
        L = np.maximum(L, 1e-300)

        # Conditional params for removing dim 2 -> tilde params of dim 1,
        # and vice versa (Kan & Robotti's c_kappa terms need these).
        mu_t1  = mu2 - s12 * mu1 / s1sq
        s_t1   = np.sqrt(max(s2sq - s12 ** 2 / s1sq, 1e-300))
        mu_t2  = mu1 - s12 * mu2 / s2sq
        s_t2   = np.sqrt(max(s1sq - s12 ** 2 / s2sq, 1e-300))

        def phi1_at0(mu, s):
            return norm.pdf(0.0, loc=mu, scale=s)

        def F1_0(mu, s):
            return 1 - norm.cdf(-mu / s)

        def F1_1(mu, s):
            a = -mu / s
            return mu * F1_0(mu, s) + s * norm.pdf(a)

        c00_1 = phi1_at0(mu1, s1) * F1_0(mu_t1, s_t1)
        c00_2 = phi1_at0(mu2, s2) * F1_0(mu_t2, s_t2)
        F10   = mu1 * L + s1sq * c00_1 + s12 * c00_2   # E[T1] numerator
        F01   = mu2 * L + s12 * c00_1 + s2sq * c00_2   # E[T2] numerator

        c10_2      = phi1_at0(mu2, s2) * F1_1(mu_t2, s_t2)
        F20        = mu1 * F10 + s1sq * L + s12 * c10_2         # E[T1^2] numerator
        F11_route1 = mu2 * F10 + s12 * L + s2sq * c10_2         # E[T1 T2] via kappa=(1,0)

        c01_1      = phi1_at0(mu1, s1) * F1_1(mu_t1, s_t1)
        F02        = mu2 * F01 + s12 * c01_1 + s2sq * L         # E[T2^2] numerator
        F11_route2 = mu1 * F01 + s1sq * c01_1 + s12 * L         # E[T1 T2] via kappa=(0,1)

        eta          = np.stack([F10 / L, F01 / L], axis=1)
        # Both routes derive the same true cross moment via independent
        # recursion paths; averaging cancels floating-point drift between
        # them (they agree to ~1e-17 in validation, so this is not masking
        # a real discrepancy).
        cross        = 0.5 * (F11_route1 + F11_route2) / L
        diag0        = F20 / L
        diag1        = F02 / L

        # Defense-in-depth: when a component's shape drifts so far that the
        # true positive-orthant probability L rounds to (numerically
        # indistinguishable from) 0, F10/F20/etc./L divisions produce
        # unbounded garbage. A NAIVE element-wise clip on eta/Psi -- clamping
        # each entry independently to a fixed range -- does NOT guarantee
        # the resulting Psi stays a valid second-moment matrix: e.g.
        # clamping a garbage-negative diagonal down to -CLIP**2 instead of
        # up to 0 (E[Ti^2] can never be negative) produces a Psi with a
        # large-magnitude negative eigenvalue, which is exactly the
        # mechanism that made the M-step's Delta update blow up on
        # degenerate near-zero-L rows (confirmed via a direct reproducer:
        # mean=[184.5,-35.4], corr=0.986, true orthant probability
        # numerically 0 -> old symmetric clip gave Psi=[[1e8,1e8],
        # [1e8,-1e8]], eigenvalues [-1.41e8, 1.41e8]). Fix: clip the
        # diagonal to [0, CLIP**2] (its true valid range) and the
        # off-diagonal to the Cauchy-Schwarz bound |Psi01|<=sqrt(Psi00*Psi11)
        # derived FROM the (already-clipped) diagonal -- this guarantees
        # every returned Psi is PSD by construction, not just individually
        # bounded, mirroring the same PSD-guarantee approach used for the
        # old approximate formula's cross term.
        CLIP = 1e4
        eta   = np.nan_to_num(eta, nan=0.0, posinf=CLIP, neginf=-CLIP)
        eta   = np.clip(eta, -CLIP, CLIP)
        diag0 = np.nan_to_num(diag0, nan=0.0, posinf=CLIP ** 2, neginf=0.0)
        diag1 = np.nan_to_num(diag1, nan=0.0, posinf=CLIP ** 2, neginf=0.0)
        diag0 = np.clip(diag0, 0.0, CLIP ** 2)
        diag1 = np.clip(diag1, 0.0, CLIP ** 2)
        cross = np.nan_to_num(cross, nan=0.0, posinf=CLIP ** 2, neginf=-CLIP ** 2)
        cross_bound = np.sqrt(diag0 * diag1)
        cross = np.clip(cross, -cross_bound, cross_bound)

        Psi          = np.zeros((N, 2, 2))
        Psi[:, 0, 0] = diag0
        Psi[:, 1, 1] = diag1
        Psi[:, 0, 1] = cross
        Psi[:, 1, 0] = cross
        return eta, Psi

    # ── q != 2: original MC path unchanged ──────────────────────────────
    eta = np.zeros((N, q))
    Psi = np.zeros((N, q, q))
    try:
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(q))
    except np.linalg.LinAlgError:
        eigv    = np.linalg.eigvalsh(cov)
        cov_reg = cov + (1e-8 - min(eigv.min(), 0)) * np.eye(q)
        L       = np.linalg.cholesky(cov_reg)

    oversample      = max(n_mc * 10, 2000)
    Z_batch         = rng.randn(oversample, q)
    candidates_base = Z_batch @ L.T

    for j in range(N):
        m          = means[j]
        candidates = candidates_base + m
        valid      = (candidates > 0).all(axis=1)
        n_valid    = valid.sum()
        if n_valid >= n_mc:
            good = candidates[valid][:n_mc]
        elif n_valid >= max(20, n_mc // 5):
            good = candidates[valid]
        else:
            good = _gibbs_sample_tn_q(m, cov, n_mc, n_burnin=50, rng=rng)
        eta[j] = good.mean(axis=0)
        Psi[j] = (good.T @ good) / len(good)

    return eta, Psi


def get_truncated_normal_moments_cfusn(observations, mu, Delta, Gamma, n_mc=500, rng=None):
    """Compute truncated normal moments for CFUSN (q >= 1).

    T | X=x ~ TN_q(m, S, R^q_+)
    where m = Delta' Omega^{-1} (x - mu), S = I_q - Delta' Omega^{-1} Delta

    Parameters
    ----------
    observations : (N, p)  — may contain NaN
    mu : (p,)
    Delta : (p, q)
    Gamma : (p, p)
    n_mc : int

    Returns
    -------
    eta : (N, q)     — E[T | x_j]
    Psi : (N, q, q)  — E[TT' | x_j]
    """
    Delta = np.asarray(Delta, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)

    if Delta.ndim == 1:
        Delta = Delta.reshape(-1, 1)

    p, q = Delta.shape
    observations = np.atleast_2d(observations)
    N = observations.shape[0]

    if rng is None:
        rng = np.random.RandomState()

    has_missing = np.isnan(observations).any()

    if not has_missing:
        # All dimensions observed — single covariance S shared across obs
        Omega = Gamma + Delta @ Delta.T
        Omega = 0.5 * (Omega + Omega.T)
        Omega_inv_Delta = np.linalg.solve(Omega, Delta)  # (p, q)
        S = np.eye(q) - Delta.T @ Omega_inv_Delta  # (q, q)
        S = 0.5 * (S + S.T)
        # Regularize
        eig = np.linalg.eigvalsh(S)
        if eig.min() < 1e-10:
            S += (1e-10 - eig.min() + 1e-10) * np.eye(q)

        residuals = observations - mu  # (N, p)
        means = residuals @ Omega_inv_Delta  # (N, q)

        if q == 1:
            # Fast scalar path
            sigma = np.sqrt(max(S[0, 0], 1e-12))
            v, w = trunc_norm_moments(means[:, 0], np.full(N, sigma))
            return v.reshape(N, 1), w.reshape(N, 1, 1)

        eta, Psi = _mc_truncated_mvn_moments(means, S, n_mc=n_mc, rng=rng)
        return eta, Psi

    # Handle missing data: group by observation pattern
    eta = np.zeros((N, q))
    Psi = np.zeros((N, q, q))
    # Default for fully missing: prior moments of TN_q(0, I_q, R^q_+)
    prior_eta = np.full(q, np.sqrt(2 / np.pi))
    prior_Psi = np.eye(q)  # E[TT'] for standard half-normal

    obs_mask = ~np.isnan(observations)
    patterns, inverse = np.unique(obs_mask, axis=0, return_inverse=True)

    for pi, pattern in enumerate(patterns):
        obs_dims = np.where(pattern)[0]
        idx = np.where(inverse == pi)[0]

        if len(obs_dims) == 0:
            eta[idx] = prior_eta
            Psi[idx] = prior_Psi
            continue

        # Marginal params for observed dims
        mu_s = mu[obs_dims]
        Delta_s = Delta[obs_dims, :]  # (|S|, q)
        Gamma_s = Gamma[np.ix_(obs_dims, obs_dims)]
        x_s = observations[np.ix_(idx, obs_dims)]

        Omega_s = Gamma_s + Delta_s @ Delta_s.T
        Omega_s = 0.5 * (Omega_s + Omega_s.T)
        eig_O = np.linalg.eigvalsh(Omega_s)
        if eig_O.min() < 1e-10:
            Omega_s += (1e-10 - eig_O.min() + 1e-10) * np.eye(len(obs_dims))

        Omega_s_inv_Delta_s = np.linalg.solve(Omega_s, Delta_s)  # (|S|, q)
        S_s = np.eye(q) - Delta_s.T @ Omega_s_inv_Delta_s  # (q, q)
        S_s = 0.5 * (S_s + S_s.T)
        eig_S = np.linalg.eigvalsh(S_s)
        if eig_S.min() < 1e-10:
            S_s += (1e-10 - eig_S.min() + 1e-10) * np.eye(q)

        residuals_s = x_s - mu_s
        means_s = residuals_s @ Omega_s_inv_Delta_s  # (n_idx, q)

        if q == 1:
            sigma = np.sqrt(max(S_s[0, 0], 1e-12))
            v, w = trunc_norm_moments(means_s[:, 0], np.full(len(idx), sigma))
            eta[idx] = v.reshape(-1, 1)
            Psi[idx] = w.reshape(-1, 1, 1)
        else:
            e, P = _mc_truncated_mvn_moments(means_s, S_s, n_mc=n_mc, rng=rng)
            eta[idx] = e
            Psi[idx] = P

    return eta, Psi


# ══════════════════════════════════════════════
# Univariate update rules (unchanged)
# ══════════════════════════════════════════════

def get_location_update(observations, responsibilities, component_params,
                        sample_weights=None):
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    (_, Delta, Gamma) = density_utils.canonical_to_alternate(*component_params)
    m = observations - v * Delta
    r = responsibilities if sample_weights is None else responsibilities * sample_weights
    return (m * r).sum() / r.sum()


def get_Delta_update(updated_loc, observations, responsibilities, component_params,
                     sample_weights=None):
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    d = v * (observations - updated_loc)
    r = responsibilities if sample_weights is None else responsibilities * sample_weights
    return (d * r).sum() / (w * r).sum()


def get_Gamma_update(updated_loc, updated_Delta, observations, responsibilities, component_params,
                    sample_weights=None):
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, component_params)
    g = (
        (observations - updated_loc) ** 2
        - 2 * updated_Delta * v * (observations - updated_loc)
        + updated_Delta**2 * w
    )
    r = responsibilities if sample_weights is None else responsibilities * sample_weights
    return (g * r).sum() / r.sum()


# ══════════════════════════════════════════════
# CFUSN M-step updates (q >= 1)
#
# Following Lin (2009) M-step equations:
#   mu  = (sum z_j y_j - Delta sum z_j eta_j) / sum z_j
#   Delta = (sum z_j (y-mu) eta') (sum z_j Psi)^{-1}
#   Gamma = (1/Z) sum z_j [(y-mu-D*eta)(y-mu-D*eta)' + D(Psi-eta eta')D']
#
# Here eta_j = E[T|x_j], Psi_j = E[TT'|x_j], z_j = responsibility
# ══════════════════════════════════════════════

# ── CFUSN M-step helpers — vectorized over p ────────────────────────────────

def _regularized_spd(A, floor=1e-10):
    """Symmetrize and floor the eigenvalues of A so it can be solved against.

    Needed because Gamma_oo for a given missingness pattern can be genuinely
    near-singular on real data (collinear assay dimensions), not just
    numerically noisy.
    """
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    eig = np.linalg.eigvalsh(A)
    if eig.min() < floor:
        A = A + (floor - eig.min() + floor) * np.eye(A.shape[0])
    return A


def _completed_data_moments(observations, mu, Delta, Gamma, eta, Psi):
    """Posterior moments of the COMPLETE data vector x under missingness.

    The M-step maximises Q = E[log p(x, T) | x_obs], so when entries are
    missing it needs posterior moments of the full x, not just its observed
    part. Conditional on T=t we have x ~ N(mu + Delta t, Gamma), so for a row
    with observed dims o and missing dims m:

        x_m | x_o, t ~ N(a_m + B_m t,  C_mm)
        K    = Gamma_mo Gamma_oo^-1
        a_m  = mu_m + K (x_o - mu_o)
        B_m  = Delta_m - K Delta_o
        C_mm = Gamma_mm - K Gamma_om

    This is linear in t, so E[x | x_o] = a + B eta and the second moments
    follow by the tower property, reusing eta = E[T | x_o] and
    Psi = E[TT' | x_o] (already computed against the correct observed-dims
    marginal by get_truncated_normal_moments_cfusn).

    Why this matters: summing only over observed entries -- available-case
    estimation -- is the exact M-step ONLY when Gamma is diagonal, because it
    discards precisely the observed-missing cross terms that Gamma^-1 couples.
    With a full Gamma it does not maximise Q, so EM loses its ascent guarantee
    and the observed-data likelihood can decrease. Confirmed directly: on data
    where the available-case M-step decreased the likelihood at iteration 1,
    forcing Gamma diagonal restored monotone convergence.

    Returns (Ex, Ext, Exx) with shapes (N, p), (N, p, q), (N, p, p).
    """
    observations = np.atleast_2d(np.asarray(observations, dtype=float))
    N, p = observations.shape
    q = eta.shape[1]
    mu = np.asarray(mu, dtype=float)
    Delta = density_utils._ensure_matrix_delta(Delta)
    Gamma = np.asarray(Gamma, dtype=float)

    a = np.where(np.isnan(observations), 0.0, observations)
    B = np.zeros((N, p, q))
    C = np.zeros((N, p, p))
    q_idx = np.arange(q)

    obs_mask = ~np.isnan(observations)
    patterns, inverse = np.unique(obs_mask, axis=0, return_inverse=True)
    for pi, pattern in enumerate(patterns):
        idx = np.where(inverse == pi)[0]
        o = np.where(pattern)[0]
        m = np.where(~pattern)[0]
        if len(m) == 0:
            continue                       # fully observed: a = x, B = 0, C = 0
        if len(o) == 0:
            # Nothing observed: fall back to the component's own prior moments.
            a[np.ix_(idx, m)] = mu[m]
            B[np.ix_(idx, m, q_idx)] = Delta[m]
            C[np.ix_(idx, m, m)] = Gamma[np.ix_(m, m)]
            continue

        G_oo = _regularized_spd(Gamma[np.ix_(o, o)])
        K = np.linalg.solve(G_oo, Gamma[np.ix_(m, o)].T).T        # (|m|, |o|)
        a[np.ix_(idx, m)] = mu[m] + (observations[np.ix_(idx, o)] - mu[o]) @ K.T
        B[np.ix_(idx, m, q_idx)] = Delta[m] - K @ Delta[o]
        C_mm = Gamma[np.ix_(m, m)] - K @ Gamma[np.ix_(o, m)]
        C[np.ix_(idx, m, m)] = 0.5 * (C_mm + C_mm.T)

    B_eta = np.einsum('npq,nq->np', B, eta)
    Ex = a + B_eta
    Ext = a[:, :, None] * eta[:, None, :] + np.einsum('npi,nij->npj', B, Psi)
    Exx = (a[:, :, None] * a[:, None, :]
           + a[:, :, None] * B_eta[:, None, :]
           + B_eta[:, :, None] * a[:, None, :]
           + np.einsum('npi,nij,nqj->npq', B, Psi, B)
           + C)
    return Ex, Ext, Exx


def get_location_update_cfusn(observations, responsibilities, mu, Delta, Gamma,
                               n_mc=500, rng=None, sample_weights=None,
                               return_completed=False):
    """CFUSN location M-step. Returns (mu_new, eta, Psi) — moments reused downstream.

    With ``return_completed=True`` also returns the completed-data moments
    (or None when nothing is missing) so the Delta/Gamma steps can reuse them
    instead of recomputing the per-pattern conditionals three times.
    """
    Delta      = density_utils._ensure_matrix_delta(Delta)
    eta, Psi   = get_truncated_normal_moments_cfusn(
        observations, mu, Delta, Gamma, n_mc=n_mc, rng=rng
    )
    obs    = ~np.isnan(observations)                 # (N, p)
    x_fill = np.where(obs, observations, 0.0)
    z      = responsibilities                         # (N,)
    z_eff  = z if sample_weights is None else z * sample_weights

    if obs.all():
        # Complete data: available-case masking is exact, so this path is left
        # exactly as it was (bit-identical results for fully-observed inputs).
        #
        # KEY FIX: single matrix op replaces per-dimension loop
        # Delta_eta[j, d] = sum_r Delta[d, r] * eta[j, r]
        Delta_eta = eta @ Delta.T                         # (N, p)
        obs_z     = obs * z_eff[:, None]                  # (N, p)
        numer     = (obs_z * (x_fill - Delta_eta)).sum(0) # (p,)
        denom     = obs_z.sum(0)                           # (p,)
        mu_new    = np.where(denom > 1e-12, numer / np.maximum(denom, 1e-12), mu)
        completed = None
    else:
        completed = _completed_data_moments(observations, mu, Delta, Gamma, eta, Psi)
        Ex = completed[0]
        # mu maximising Q is the full-vector weighted mean of E[x] - Delta E[T]
        # (Gamma^-1 cancels for an unconstrained mu, exactly as in the complete
        # -data case -- but E[x] must be the COMPLETED x, not the observed part).
        denom = float(z_eff.sum())
        if denom > 1e-12:
            mu_new = (z_eff[:, None] * (Ex - eta @ Delta.T)).sum(0) / denom
        else:
            mu_new = mu

    if return_completed:
        return mu_new, eta, Psi, completed
    return mu_new, eta, Psi


# Minimum Gamma eigenvalue, as a fraction of the data's own mean per-dimension
# variance. Guards the classic unbounded-likelihood degeneracy of
# Gaussian/skew-normal mixtures: LL rises without bound as a component's
# covariance collapses onto a lower-dimensional subspace. Measured directly on
# simulated data whose TRUE Gamma is 0.3*I (min eigenvalue 0.3): an
# unconstrained fit under 50% MCAR drove min eig(Gamma) to 1.06e-08 while the
# likelihood was still climbing after 30,000 iterations and never met the 1e-8
# convergence test -- EM working correctly on an ill-posed objective.
#
# Deliberately scaled to the GLOBAL data variance, not the component's own
# responsibility-weighted variance: a component-local scale shrinks as the
# component collapses, so the bound would chase the degeneracy down instead of
# containing it (the same failure mode documented for the Delta magnitude cap
# below). Set low (1e-6) because real data can have genuinely near-singular
# Gamma from assay collinearity -- confirmed on real TP53, where every
# component legitimately has a near-zero eigenvalue -- so this must only catch
# outright collapse, not real structure.
MIN_GAMMA_EIGVAL_FRAC = 1e-6


# Inverse-Wishart (ridge) penalty on Gamma -- the smooth alternative to the hard
# eigenvalue floor below. With an IW(Psi, nu) prior the penalised M-step stays
# closed form:
#
#     Gamma = (S_c + Psi) / (n_c + nu + p + 1)
#
# where S_c is the responsibility-weighted second-moment sum this update already
# computes and n_c = sum of responsibilities. Unlike clipping, this makes the
# OBJECTIVE bounded rather than truncating the parameter space: there is a real
# interior maximum, so EM converges to it instead of grinding along a boundary.
# It also shrinks automatically with n_c -- a component with plenty of effective
# mass is barely affected, a collapsing one is pulled back hardest.
#
# Psi = GAMMA_RIDGE_FRAC * (mean per-dim variance) * I, nu = p + 2 (weakly
# informative, finite prior mean). The implied smallest eigenvalue is
# ~Psi/(n_c + nu + p + 1), i.e. it adapts to component mass rather than being a
# fixed constant like MIN_GAMMA_EIGVAL_FRAC.
GAMMA_RIDGE_FRAC = 1e-3

# Which regularisation the multivariate M-step applies to Gamma.
# False -> hard eigenvalue floor (_floor_gamma_eigenvalues); True -> IW ridge.
#
# Default True: the hard floor turned out to be load-bearing rather than
# insurance -- on real TP53 every component came to rest exactly on it
# (1.000e-06 for all 6 at K=6, 2 of 3 at K=3), so the fitted covariance was
# determined by the constant rather than by the data. The ridge left Gamma
# data-determined (1.8e-05 - 4.6e-05) with zero pinned components in every
# arm tested, at the cost of slightly lower raw likelihood, which is the
# expected price of shrinkage.
USE_GAMMA_RIDGE = True


def _ridge_gamma(Gamma_new, n_eff, observations, frac=GAMMA_RIDGE_FRAC, nu=None):
    """Inverse-Wishart MAP shrinkage of a Gamma M-step candidate.

    ``Gamma_new`` is the unpenalised candidate S_c / n_c, so the penalised
    estimate is recovered as (Gamma_new * n_c + Psi) / (n_c + nu + p + 1).

    Monotonicity: this is the exact maximiser of the penalised Q, so EM remains
    monotone in (Q + log prior). The reported observed-data likelihood is NOT
    the objective being maximised any more -- see the caller for why that
    matters for val_ll-based model selection.
    """
    G = 0.5 * (np.asarray(Gamma_new, dtype=float) + np.asarray(Gamma_new, dtype=float).T)
    p = G.shape[0]
    if nu is None:
        nu = p + 2
    with np.errstate(invalid="ignore"):
        scale = float(np.nanmean(np.nanvar(np.atleast_2d(observations), axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        return G
    n_eff = max(float(n_eff), 0.0)
    Psi = frac * scale * np.eye(p)
    return (G * n_eff + Psi) / (n_eff + nu + p + 1)


def gamma_log_prior_total(component_params, observations, multivariate=True):
    """Total log IW(Psi, nu) prior over every component's Gamma.

    Returns 0.0 when the ridge is disabled, so callers can add this
    unconditionally and stay bit-identical in the floor/no-ridge configuration.

    This is the term that makes the ridge's objective differ from the raw
    observed-data likelihood: the penalised M-step maximises
    (log L + this), so a monotonicity check run against log L alone will see
    spurious decreases -- which is exactly what happened before this existed
    (a ridge fit failing with "Likelihood decreased by 3.9e+00" at iteration 8
    while the penalised objective was still increasing).

        log p(Gamma) = -(nu + p + 1)/2 * log|Gamma| - 1/2 * tr(Psi Gamma^-1)

    (dropping the Gamma-independent normalising constant, which cancels in
    every iteration-to-iteration comparison.)
    """
    if not USE_GAMMA_RIDGE or not multivariate:
        return 0.0
    with np.errstate(invalid="ignore"):
        scale = float(np.nanmean(np.nanvar(np.atleast_2d(observations), axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        return 0.0
    total = 0.0
    for prm in component_params:
        if len(prm) < 3:
            continue
        G = np.asarray(prm[2], dtype=float)
        if G.ndim != 2:
            continue
        G = 0.5 * (G + G.T)
        p_dim = G.shape[0]
        nu = p_dim + 2
        Psi = GAMMA_RIDGE_FRAC * scale * np.eye(p_dim)
        sign, logdet = np.linalg.slogdet(G)
        if sign <= 0 or not np.isfinite(logdet):
            return -np.inf
        try:
            tr_term = float(np.trace(np.linalg.solve(G, Psi)))
        except np.linalg.LinAlgError:
            return -np.inf
        total += -0.5 * (nu + p_dim + 1) * logdet - 0.5 * tr_term
    return float(total)


def _regularize_gamma_candidate(Gamma_new, n_eff, observations):
    """Dispatch to whichever Gamma regularisation is enabled."""
    if USE_GAMMA_RIDGE:
        return _ridge_gamma(Gamma_new, n_eff, observations)
    return _floor_gamma_eigenvalues(Gamma_new, observations)


def _floor_gamma_eigenvalues(Gamma, observations, frac=MIN_GAMMA_EIGVAL_FRAC):
    """Clip Gamma's eigenvalues from below at frac * (mean per-dim variance).

    Monotonicity-safe: for a covariance M-step of the form
    Q ∝ -0.5[log|Gamma| + tr(Gamma^-1 S)], the maximiser subject to a minimum
    eigenvalue constraint is exactly S with its eigenvalues clipped at that
    minimum (standard result for constrained Gaussian-mixture MLE). The update
    here returns precisely such an S, so clipping yields the constrained
    maximiser rather than an arbitrary projection -- ECM ascent is preserved on
    the constrained parameter space.
    """
    Gamma = 0.5 * (np.asarray(Gamma, dtype=float) + np.asarray(Gamma, dtype=float).T)
    with np.errstate(invalid="ignore"):
        scale = float(np.nanmean(np.nanvar(np.atleast_2d(observations), axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        return Gamma
    floor = frac * scale
    eig, vec = np.linalg.eigh(Gamma)
    if eig.min() >= floor:
        return Gamma
    return vec @ np.diag(np.maximum(eig, floor)) @ vec.T


# How the Delta magnitude cap's per-dimension bound is scaled.
#   "local"  -- SAFETY_FACTOR * sqrt(component's own responsibility-weighted
#               variance). Original behaviour. Flaw: that variance is exactly
#               what shrinks when a component's Gamma collapses, so the bound
#               tightens as the component degenerates and the cap ends up
#               binding routinely instead of rarely (measured on real TP53:
#               70% of component-iterations at K=3, up to 95% at K=2).
#   "global" -- SAFETY_FACTOR * sqrt(the DATA's per-dimension variance), which
#               does not move as a component collapses. Same reasoning as
#               _floor_gamma_eigenvalues scaling to global rather than
#               component-local variance.
DELTA_CAP_SCALE = "local"

# Include the (1 - 2/pi) truncated-normal variance factor in the cap's
# positive-definiteness bound.
#
# The original bound came from the standard factor-analysis identity
# Sigma = Lambda Lambda' + Psi, i.e. Gamma_dd = cov_dd - ||Delta_d||^2, which
# holds when the latent factors have UNIT variance. CFUSN's factor is truncated
# to the positive orthant -- T ~ TN_q(0, I, R_+^q), a half-normal with
# E[T] = sqrt(2/pi) and Var(T) = 1 - 2/pi ~ 0.3634 -- so the correct identity is
#
#     Var(X_d) = Gamma_dd + (1 - 2/pi) * ||Delta_d||^2
#
# and positive Gamma_dd requires ||Delta_d|| < sqrt(Var_d / (1 - 2/pi)), i.e.
# 1.66 * sqrt(Var_d). Assuming Var(T) = 1 makes the bound sqrt(0.3634) = 0.603x
# too small, so the cap clamped at ~57% of the positive-definiteness limit it
# cites as its own justification -- which is why it bound on 64-98% of
# component-iterations on real data under every scaling tried. Verified
# numerically: regressing empirical Var(X_d) on ||Delta_d||^2 across 8
# (Delta, Gamma) settings gives c = 0.355-0.365 vs (1 - 2/pi) = 0.3634.
#
# Set False to reproduce the original (too strict) bound.
DELTA_CAP_PD_CORRECTION = True


def _global_var_d(observations):
    """Per-dimension variance of the data itself (NaN-aware), used by the
    "global" cap scaling so the bound does not shrink with a collapsing
    component."""
    with np.errstate(invalid="ignore"):
        v = np.nanvar(np.atleast_2d(np.asarray(observations, dtype=float)), axis=0)
    return np.where(np.isfinite(v), v, 0.0)


def _apply_delta_cap(Delta_new, weighted_var_d, Delta_prev=None, global_var_d=None):
    """Defense-in-depth magnitude cap; see get_Delta_update_cfusn for the full
    rationale on why the bound is SAFETY_FACTOR * sqrt(local variance).

    Two behaviours:

    - ``Delta_prev`` given (the production path): a binding cap REJECTS the
      candidate and keeps the previous Delta for this component. This is
      monotonicity-safe -- Q decomposes as a sum over components and the M-step
      is a sequence of conditional maximisations (ECM), so leaving one block at
      its previous value is a no-op and still gives Q_new >= Q_old.
    - ``Delta_prev`` None (legacy): rescale the offending rows. Retained for
      callers that have no previous Delta to fall back on, but note that a
      rescaled Delta is neither the maximiser nor the previous value, which is
      exactly why it can DECREASE the observed-data likelihood (confirmed: a
      fit failing with a 1.51e-04 decrease at iteration 1476 converged
      monotonically once the rescale was removed).

    Returns (Delta, capped) where ``capped`` reports whether the cap bound.
    """
    SAFETY_FACTOR = 0.95
    var_d = weighted_var_d
    if DELTA_CAP_SCALE == "global" and global_var_d is not None:
        var_d = global_var_d
    # Positive-definiteness limit. For X = mu + Delta T + e with
    # T ~ TN_q(0, I, R_+^q), Var(T_i) = 1 - 2/pi, so
    #     Var(X_d) = Gamma_dd + (1 - 2/pi) * ||Delta_d||^2
    # and Gamma_dd > 0 requires ||Delta_d|| < sqrt(Var_d / (1 - 2/pi))
    #                                       = 1.66 * sqrt(Var_d).
    # The original bound omitted the (1 - 2/pi) factor and so clamped at
    # 0.95*sqrt(Var_d) -- about 57% of the actual limit, which is why the cap
    # bound on 64-98% of component-iterations on real data under every
    # scaling. Verified numerically: regressing Var_emp on ||Delta_d||^2
    # across 8 (Delta, Gamma) settings gives c = 0.355-0.365 vs
    # (1 - 2/pi) = 0.3634.
    var_d = np.maximum(var_d, 0.0)
    if DELTA_CAP_PD_CORRECTION:
        var_d = var_d / (1.0 - 2.0 / np.pi)
    max_norm_d = SAFETY_FACTOR * np.sqrt(var_d)
    row_norms = np.linalg.norm(Delta_new, axis=1)
    over = row_norms > max_norm_d
    if not over.any():
        return Delta_new, False
    if Delta_prev is not None:
        return np.asarray(Delta_prev, dtype=float), True
    scale = np.where(over, max_norm_d / np.maximum(row_norms, 1e-12), 1.0)
    return Delta_new * scale[:, None], True


def _delta_update_completed(mu_new, z_eff, eta, Psi, completed, p, q, Delta_prev=None,
                            global_var_d=None):
    """Delta M-step from completed-data moments (missing-data case).

    With E[x] completed, every row contributes to every dimension, so the
    per-dimension (p separate q x q) solves of the available-case path collapse
    to a single q x q solve -- the same system the complete-data M-step solves.
    """
    Ex, Ext, Exx = completed
    RIDGE_FLOOR = 1e-3   # same floor/rationale as the available-case path below

    # numer[d, r] = sum_n z_n (E[x_n t_n']_{d,r} - mu_d eta_{n,r})
    numer = (np.einsum('n,ndr->dr', z_eff, Ext)
             - np.outer(mu_new, (z_eff[:, None] * eta).sum(0)))
    Psi_sum = np.einsum('n,nij->ij', z_eff, Psi)          # (q, q)

    eig = np.linalg.eigvalsh(0.5 * (Psi_sum + Psi_sum.T))
    if eig.min() < RIDGE_FLOOR:
        Psi_sum = Psi_sum + (RIDGE_FLOOR - eig.min() + RIDGE_FLOOR) * np.eye(q)
    try:
        Delta_new = np.linalg.solve(Psi_sum, numer.T).T   # (p, q)
    except np.linalg.LinAlgError:
        Delta_new = numer / np.maximum(np.diag(Psi_sum), 1e-12)[None, :]

    # Same defense-in-depth magnitude cap as the available-case path (see its
    # comment block for the full rationale); the per-dimension local variance
    # is now the completed second moment E[(x_d - mu_d)^2] rather than an
    # observed-only variance.
    denom = float(z_eff.sum())
    if denom > 1e-8:
        var_rows = (np.einsum('ndd->nd', Exx)
                    - 2.0 * Ex * mu_new[None, :]
                    + (mu_new ** 2)[None, :])                # (N, p)
        weighted_var_d = (z_eff[:, None] * var_rows).sum(0) / denom
    else:
        weighted_var_d = np.zeros(p)
    Delta_new, _capped = _apply_delta_cap(Delta_new, weighted_var_d, Delta_prev,
                                          global_var_d=global_var_d)
    return Delta_new


def get_Delta_update_cfusn(mu_new, observations, responsibilities, eta, Psi,
                           sample_weights=None, completed=None, Delta_prev=None):
    """CFUSN Delta M-step.  Returns (p, q).

    KEY FIX: replace per-d einsum loop with two batched einsums, then
    solve per-dimension (p=2 → only 2 solves of 2×2 systems).

    ``completed`` is the (Ex, Ext, Exx) tuple from _completed_data_moments,
    supplied only when the data has missing entries; when it is None the
    original available-case path runs unchanged (exact for complete data).
    """
    obs    = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    z      = responsibilities
    z_eff  = z if sample_weights is None else z * sample_weights
    N, p   = observations.shape
    q      = eta.shape[1]

    if completed is not None:
        return _delta_update_completed(mu_new, z_eff, eta, Psi, completed, p, q,
                                       Delta_prev=Delta_prev,
                                       global_var_d=_global_var_d(observations))

    obs_z    = obs * z_eff[:, None]                   # (N, p)
    residuals = x_fill - mu_new                       # (N, p)

    # numer[d, r] = sum_n obs_z[n,d] * residuals[n,d] * eta[n,r]
    numer    = (obs_z * residuals).T @ eta             # (p, q) — one matmul

    # Psi_sum[d, i, j] = sum_n obs_z[n,d] * Psi[n, i, j]
    Psi_sum  = np.einsum('nd,nij->dij', obs_z, Psi)   # (p, q, q)

    # Ridge floor for Psi_sum[d]'s conditioning check. Was 1e-10 -- far too
    # small for the ~unit-scale (per-dimension-standardized, see
    # Fit.generate_fit_jobs) data this actually runs on: a near-singular 2x2
    # solve with an O(1e-10) floor but O(1)-scale numer[d] can return
    # results inflated by a factor of ~1/floor, i.e. ~1e10 -- confirmed
    # directly (real TP53 production data, stochastic across BLAS
    # summation-order noise around the singularity: identical inputs
    # produced Delta row norms in the single digits on most runs, but
    # ~1.7e9 on others). 1e-3 is still far below any well-conditioned
    # Psi_sum's eigenvalues on this data (verified: 15 varied-seed TP53
    # trials post-fix, max Delta row norm ~1.4, no blowups), so this only
    # changes behavior for genuinely near-singular per-dimension systems
    # (e.g. a dimension with ~zero effective observations for this
    # component), which is exactly the case that needs damping rather than
    # amplification.
    RIDGE_FLOOR = 1e-3
    Delta_new = np.zeros((p, q))
    for d in range(p):                                 # p=2 → 2 iterations
        Ps = Psi_sum[d]
        eig = np.linalg.eigvalsh(Ps)
        if eig.min() < RIDGE_FLOOR:
            Ps = Ps + (RIDGE_FLOOR - eig.min() + RIDGE_FLOOR) * np.eye(q)
        try:
            Delta_new[d] = np.linalg.solve(Ps, numer[d])
        except np.linalg.LinAlgError:
            Delta_new[d] = numer[d] / np.maximum(np.diag(Ps), 1e-12)

    # ── Defense-in-depth magnitude cap ──────────────────────────────────
    # The Psi PSD-violation fix (see _mc_truncated_mvn_moments) and the
    # ridge floor above eliminate outright numerical blowups (was: up to
    # ~1.7e9; after those two fixes alone: a non-catastrophic but still
    # real residual tail, up to ~47 on real TP53 K=6 data, 13/960 fits
    # exceeding row-norm 10). This cap eliminates that remaining tail
    # (verified: 0/960 exceed row-norm 10 afterward, max row-norm ~4.5)
    # while costing negligible real recovery accuracy, for a mathematical
    # reason specific to the skew-normal family, not an arbitrary limit:
    #
    # Gamma[d,d] = cov[d,d] - ||Delta[d,:]||^2 must stay positive for Gamma
    # to be PD at all, so ||Delta[d,:]||^2 approaching cov[d,d] is already
    # a sign of a poorly-conditioned estimate. But more importantly, the
    # skew-normal family this reduces to marginally (a single active
    # skewing direction plus independent Gaussian noise) has a hard
    # ceiling on Pearson skewness of ~0.995 (Azzalini's skew-normal:
    # skewness -> (4-pi)/2 * (sqrt(2/pi))^3 / (1-2/pi)^1.5 ~= 0.995 as the
    # shape parameter -> infinity, and NEVER exceeds it at any finite
    # value). Past a moderate Delta magnitude, further increases buy
    # almost no additional real skewness -- the likelihood becomes very
    # flat in Delta (many different large values are nearly
    # indistinguishable), a known skew-normal MLE identifiability
    # pathology -- while linearly increasing ||Delta[d,:]||^2's share of
    # cov[d,d], which is exactly the poorly-conditioned regime this whole
    # investigation traced these blowups back to. Capping at
    # SAFETY_FACTOR=0.95 of the dimension's own LOCAL (responsibility-
    # weighted) variance leaves Gamma headroom and sits well past where
    # real, identifiable skewness has already saturated.
    #
    # Simulated directly (tests/cfusn_simulations/sim_delta_magnitude_cap.py):
    # negligible cost at realistic large true skew (94% vs 93% recovered
    # at Delta column norm 0.7, this investigation's "large" regime
    # throughout) but does measurably reduce recovery at a deliberately
    # unrealistic stress magnitude (98% vs 86% recovered at norm 1.5 --
    # by design, since that magnitude implies ~41% of a standardized
    # dimension's variance from skew alone, deep in the saturated-
    # skewness regime no real single-factor skew-normal-family signal
    # should reach). If real data genuinely needs skewness beyond what
    # ANY skew-normal-family Delta could represent (e.g. a hard assay
    # floor/ceiling/detection limit), this cap does not create that
    # limitation -- the skew-normal family's ~0.995 ceiling already caps
    # it, with or without this guard; that case needs a different
    # likelihood (e.g. genuinely censored/truncated), not a larger Delta.
    denom_d = obs_z.sum(axis=0)  # (p,)
    weighted_var_d = np.where(
        denom_d > 1e-8,
        (obs_z * residuals ** 2).sum(axis=0) / np.maximum(denom_d, 1e-8),
        0.0,
    )
    Delta_new, _capped = _apply_delta_cap(Delta_new, weighted_var_d, Delta_prev,
                                          global_var_d=_global_var_d(observations))

    return Delta_new


def _gamma_update_completed(mu_new, Delta_new, z_eff, eta, Psi, completed):
    """Gamma M-step from completed-data moments (missing-data case).

    Gamma = (1/sum z) * sum_n z_n E[u u'],  u = x - mu - Delta t, expanded into
    the completed sufficient statistics:

        E[uu'] = E[xx'] - E[xt']D' - D E[tx'] + D Psi D'
                 - mu E[x]' - E[x] mu' + mu E[t]'D' + D E[t] mu' + mu mu'

    The E[xx'] term carries the Cov[x_miss | x_obs] correction from
    _completed_data_moments, which is exactly what available-case masking drops.
    """
    Ex, Ext, Exx = completed
    denom = float(z_eff.sum())
    if denom <= 1e-12:
        return np.zeros((len(mu_new), len(mu_new)))

    S_xx = np.einsum('n,nab->ab', z_eff, Exx)
    S_xt = np.einsum('n,nar->ar', z_eff, Ext)
    S_x  = np.einsum('n,na->a', z_eff, Ex)
    S_tt = np.einsum('n,nij->ij', z_eff, Psi)
    S_t  = np.einsum('n,ni->i', z_eff, eta)

    D = Delta_new
    Gamma_new = (
        S_xx
        - S_xt @ D.T - D @ S_xt.T
        + D @ S_tt @ D.T
        - np.outer(mu_new, S_x) - np.outer(S_x, mu_new)
        + np.outer(mu_new, S_t @ D.T) + np.outer(D @ S_t, mu_new)
        + denom * np.outer(mu_new, mu_new)
    ) / denom
    return 0.5 * (Gamma_new + Gamma_new.T)


def get_Gamma_update_cfusn(mu_new, Delta_new, observations, responsibilities, eta, Psi,
                           sample_weights=None, completed=None):
    """CFUSN Gamma M-step.  Returns (p, p).

    KEY FIX: precompute Psi_minus once; use vectorized outer-product ops
    for term1 and a single batched einsum for term2, replacing repeated
    per-(d1,d2) einsums inside the double loop.

    ``completed`` is supplied only when the data has missing entries; when it
    is None the original available-case path runs unchanged.
    """
    if completed is not None:
        z = responsibilities
        z_eff = z if sample_weights is None else z * sample_weights
        return _gamma_update_completed(mu_new, Delta_new, z_eff, eta, Psi, completed)

    obs    = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    z      = responsibilities
    z_eff  = z if sample_weights is None else z * sample_weights
    N, p   = observations.shape
    q      = eta.shape[1]

    Delta_eta = eta @ Delta_new.T                             # (N, p)
    residuals = x_fill - mu_new - Delta_eta                   # (N, p)
    Psi_minus = Psi - np.einsum('ni,nj->nij', eta, eta)      # (N, q, q) — once

    obs_f   = obs.astype(residuals.dtype)                     # (N, p)
    Z_b     = (z_eff[:, None, None] * obs[:, :, None] * obs[:, None, :]).sum(0)  # (p, p)

    # term1[a,b] = sum_n z[n] · obs[n,a] · obs[n,b] · r[n,a] · r[n,b]
    #           = (z · obs · r).T @ (obs · r)         — gemm, no (N,p,p) intermediate
    masked_r = obs_f * residuals                              # (N, p), zero where unobserved
    term1    = (z_eff[:, None] * masked_r).T @ masked_r       # (p, p)

    # Psi_corr[a,b,i,j] = sum_n z[n] · obs[n,a] · obs[n,b] · Psi_minus[n,i,j]
    z_ob     = z_eff[:, None, None] * obs[:, :, None] * obs[:, None, :]  # (N, p, p)
    Psi_corr = np.einsum('nab,nij->abij', z_ob, Psi_minus)    # (p,p,q,q)
    # term2[a,b] = Delta[a,:] @ Psi_corr[a,b,:,:] @ Delta[b,:]
    term2    = np.einsum('ai,abij,bj->ab', Delta_new, Psi_corr, Delta_new)  # (p,p)

    safe      = Z_b > 1e-12
    Gamma_new = np.where(safe, (term1 + term2) / np.maximum(Z_b, 1e-12), 0.0)
    Gamma_new = 0.5 * (Gamma_new + Gamma_new.T)
    return Gamma_new


# ══════════════════════════════════════════════
# Restricted MSN (q=1) multivariate update rules (kept for backward compat)
# ══════════════════════════════════════════════

def _mv_completed_moments(observations, mu, Delta, Gamma):
    """q=1 wrapper around _completed_data_moments.

    The restricted MSN is just the CFUSN with q=1, so the same completed-data
    moments apply with eta = v (N,1) and Psi = w (N,1,1). See
    _completed_data_moments for why available-case masking is not the exact
    M-step whenever Gamma is non-diagonal -- that flaw was present in this q=1
    path exactly as it was in the CFUSN one.
    """
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    eta = v[:, None]                                  # (N, 1)
    Psi = w[:, None, None]                            # (N, 1, 1)
    completed = _completed_data_moments(observations, mu, Delta, Gamma, eta, Psi)
    return v, w, eta, Psi, completed


def get_location_update_mv(observations, responsibilities, mu, Delta, Gamma,
                           sample_weights=None):
    """mu update for q=1 restricted MSN. Delta is (p,) vector."""
    obs = ~np.isnan(observations)
    r = responsibilities if sample_weights is None else responsibilities * sample_weights
    Delta_vec = np.asarray(Delta).ravel()

    if not obs.all():
        v, w, eta, Psi, completed = _mv_completed_moments(observations, mu, Delta, Gamma)
        Ex = completed[0]
        denom = float(r.sum())
        if denom <= 1e-12:
            return np.asarray(mu, dtype=float)
        return (r[:, None] * (Ex - v[:, None] * Delta_vec[None, :])).sum(0) / denom

    # Complete data: available-case masking is exact -- original path, unchanged.
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    x_fill = np.where(obs, observations, 0.0)
    m = x_fill - v[:, None] * Delta_vec[None, :]
    z = r[:, None]
    numer = (m * z * obs).sum(axis=0)
    denom = (z * obs).sum(axis=0)
    return numer / np.maximum(denom, 1e-12)


def get_Delta_update_mv(updated_mu, observations, responsibilities, mu, Delta, Gamma,
                       sample_weights=None):
    """Delta update for q=1 restricted MSN. Returns (p,) vector."""
    obs = ~np.isnan(observations)
    z = responsibilities if sample_weights is None else responsibilities * sample_weights

    if not obs.all():
        v, w, eta, Psi, completed = _mv_completed_moments(observations, mu, Delta, Gamma)
        Ex, Ext, Exx = completed
        # Delta_d = sum_n z (E[x_d t] - mu_d E[t]) / sum_n z E[t^2]; with the
        # completed moments every row informs every dimension, so the
        # denominator is a single scalar rather than one per dimension.
        numer = (z[:, None] * Ext[:, :, 0]).sum(0) - updated_mu * float((z * v).sum())
        denom = float((z * w).sum())
        return numer / max(denom, 1e-12)

    # Complete data: available-case masking is exact -- original path, unchanged.
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    x_fill = np.where(obs, observations, 0.0)
    residuals = x_fill - updated_mu[None, :]
    numer = (z[:, None] * v[:, None] * residuals * obs).sum(axis=0)
    denom = (z[:, None] * w[:, None] * obs).sum(axis=0)
    return numer / np.maximum(denom, 1e-12)


def get_Gamma_update_mv(updated_mu, updated_Delta, observations, responsibilities, mu, Delta, Gamma,
                        sample_weights=None):
    """Gamma update for q=1 restricted MSN. Returns (p, p)."""
    N, K = observations.shape
    obs = ~np.isnan(observations)
    z = responsibilities if sample_weights is None else responsibilities * sample_weights
    updated_Delta = np.asarray(updated_Delta).ravel()

    if not obs.all():
        v, w, eta, Psi, completed = _mv_completed_moments(observations, mu, Delta, Gamma)
        return _gamma_update_completed(updated_mu, updated_Delta.reshape(-1, 1),
                                       z, eta, Psi, completed)

    # Complete data: available-case masking is exact -- original path, unchanged.
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    x_fill = np.where(obs, observations, 0.0)
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
# Responsibilities & weights (unified, unchanged)
# ══════════════════════════════════════════════

def validate_indicators(Indicators):
    assert Indicators.ndim == 2
    assert (Indicators.sum(1) == 1).all()
    assert np.isin(Indicators, [0, 1]).all()
    return Indicators.astype(bool)


def sample_specific_responsibilities(
    observations, sample_indicators, component_params, weights, multivariate=False,
    cached_log_pdfs=None,
):
    """E-step responsibilities.

    Parameters
    ----------
    cached_log_pdfs : list[ndarray] or None
        Optional per-sample (K, N_s) log-pdf matrices, indexed parallel to
        ``sample_indicators.T``. When supplied, density re-evaluation is
        skipped — used to reuse the log-pdfs computed by the previous
        iteration's ``get_sample_weights_and_ll`` (which are computed on
        the same iterate that this E-step needs).
    """
    N_samples = sample_indicators.shape[1]
    N_components = len(component_params)
    N_observations = observations.shape[0]
    assert weights.shape == (N_samples, N_components)
    responsibilities = np.zeros((N_components, N_observations))
    for i, mask in enumerate(sample_indicators.T):
        X = observations[mask]
        cached_i = (cached_log_pdfs[i]
                    if cached_log_pdfs is not None else None)
        responsibilities[:, mask] = density_utils.component_posteriors(
            X, component_params, weights[i], multivariate=multivariate,
            cached_log_pdfs=cached_i,
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


# ── Fused weight update + LL ─────────────────────────────────────────────────

def get_sample_weights_and_ll(observations, sample_indicators, updated_params,
                               current_weights, multivariate=False,
                               return_log_pdfs=False, sample_weights=None):
    """Compute updated weights AND normalised log-likelihood in a single density pass.

    The original code calls get_sample_weights (density eval) then fit.py calls
    get_likelihood (identical density eval).  This function caches the log_pdfs
    from the weight pass and re-weights with the fresh weights to get LL —
    eliminating one complete density evaluation per EM iteration.

    Parameters
    ----------
    return_log_pdfs : bool
        If True, additionally returns the per-sample list of (K, N_s)
        log-pdf matrices computed on ``updated_params``. The caller can
        feed these into the *next* iteration's
        :func:`sample_specific_responsibilities` (via its
        ``cached_log_pdfs`` arg) to avoid redoing the same density work.

    Returns
    -------
    updated_weights : (S, K)
    normalised_ll   : float   (LL / N, ready to append to likelihoods array)
    log_pdfs_per_sample : list[ndarray]   (only if return_log_pdfs=True)
    """
    S, Kc  = current_weights.shape
    N      = observations.shape[0]
    upd_w  = np.zeros_like(current_weights)
    cache  = [None] * S                              # cache (Kc, N_s) log_pdfs

    for i in range(S):
        mask = sample_indicators[:, i]
        X    = observations[mask]
        N_s  = X.shape[0]

        if not multivariate:
            log_pdfs = np.stack(
                [sps.skewnorm.logpdf(X.ravel(), *p) for p in updated_params], axis=0
            )
        else:
            X_2d = np.atleast_2d(X)
            lp_list = []
            for p in updated_params:
                lp = density_utils._single_component_logpdf(X_2d, p, multivariate=True)
                lp = np.atleast_1d(np.asarray(lp, dtype=float)).ravel()
                if len(lp) != N_s:
                    lp = np.full(N_s, -np.inf)
                lp[~np.isfinite(lp)] = -np.inf
                lp_list.append(lp)
            log_pdfs = np.stack(lp_list, axis=0)      # (Kc, N_s)

        with np.errstate(divide='ignore', invalid='ignore'):
            # invalid='ignore' suppresses (-inf) - (-inf) = NaN when an
            # observation has zero density under all components; the
            # `P[np.isnan(P)] = 0.0` below is the canonical fix.
            log_w = np.where(current_weights[i] > 0,
                             np.log(current_weights[i]), -np.inf)
            nums  = log_pdfs + log_w[:, None]
            denom = logsumexp(nums, axis=0)
            P     = np.exp(nums - denom[None])
        P[np.isnan(P)] = 0.0

        uw = P.mean(1)
        if np.isnan(uw).any():
            bad = np.where(np.isnan(P.T))[0]
            raise ValueError(
                f"NaN weight: {uw}\n{X[bad]}\n{updated_params}\n{current_weights[i]}"
            )
        upd_w[i] = uw
        cache[i] = log_pdfs

    # ── LL with updated weights — zero extra density evals ──────────────
    # When sample_weights is given, evaluate the per-obs *weighted* LL
    # Σ_n sw_n · log Σ_k W[s(n),k] f_k(x_n) — this matches the M-step's
    # Q-function under sample_balance_beta>0, so EM monotonicity holds.
    # Note: the W[s,k] update above (P.mean(1)) is unchanged because
    # sw_n is constant within sample s, so it factors out of the W
    # argmax and the standard mean-responsibility formula remains
    # optimal under the weighted Q.
    sw = None if sample_weights is None else np.asarray(sample_weights)
    ll = 0.0
    for i in range(S):
        with np.errstate(divide='ignore'):
            log_w = np.where(upd_w[i] > 0, np.log(upd_w[i]), -np.inf)
        log_mix = logsumexp(cache[i] + log_w[:, None], axis=0)
        if sw is not None:
            ll += float(np.sum(sw[sample_indicators[:, i]] * log_mix))
        else:
            ll += float(log_mix.sum())

    if return_log_pdfs:
        return upd_w, ll / N, cache
    return upd_w, ll / N


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
# Constrained update — CFUSN (line search on alpha)
# ══════════════════════════════════════════════

def _mv_build_constraint_params(component_num, current_params, updated_params,
                                 candidate_alternate, alpha):
    """Build param list with component interpolated at fraction alpha."""
    params = []
    for ki in range(len(current_params)):
        if ki < component_num:
            params.append(updated_params[ki])
        elif ki > component_num:
            params.append(current_params[ki])
        else:
            cur_alt = current_params[ki]
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
    """Line-search constraint enforcement (q=1 and q>1).

    Optimisations vs. the previous implementation:
      1. Probe grids built once from the alpha=1 candidate (rather than
         re-derived from the test param set on every binary-search step).
      2. Log-pdfs of the K-1 *static* components (not being modified by
         this M-step pass) computed once and cached.  Each binary-search
         step only re-evaluates the variable component.  This is roughly
         a K× speedup on the density-eval cost — the dominant CFUSN cost.
      3. Binary search capped at 20 iterations with 1e-4 tolerance
         (was 50/1e-6).  The EM doesn't need atomic-precision frontier.

    ``constraint_mode`` (kwarg, default 'line') is passed via kwargs;
    'marginal' enables the per-dim CFUSN marginal check from
    constraints.build_constraint_grids.
    """
    candidate = (candidate_mu, candidate_Delta, candidate_Gamma)
    K = len(current_component_params)
    constraint_mode = kwargs.get("constraint_mode", "line")
    max_iters = kwargs.get("constraint_bsearch_max_iters", 20)
    tol = kwargs.get("constraint_bsearch_tol", 1e-4)
    # Match multicomponent_density_constraint_violated's default: skip
    # near-duplicate adjacent pairs in marginal mode (1.0 nat ≈ 2.7×
    # density ratio range — components below this look the same).
    min_log_ratio_range = kwargs.get(
        "constraint_min_log_ratio_range",
        1.0 if constraint_mode == "marginal" else 0.0,
    )

    # Build the candidate-at-alpha=1 param set; used both for the early-exit
    # check and as the basis for fixing the probe grid.
    test1 = _mv_build_constraint_params(
        component_num, current_component_params, updated_component_params,
        candidate, alpha=1.0
    )

    if not multivariate:
        # Univariate path is handled by binary_search elsewhere; defensive
        # fallback that preserves previous behaviour if invoked here.
        if not multicomponent_density_constraint_violated(
            test1, xlims, multivariate=False
        ):
            return candidate_mu, candidate_Delta, candidate_Gamma

    # Probe grid(s) — fixed across the binary search.  In 'line' mode the
    # direction technically depends on means (which include the candidate);
    # we freeze it at alpha=1 so static-component logpdfs can be cached.
    grids = build_constraint_grids(test1, xlims, mode=constraint_mode)

    # Pre-compute log-pdfs for components other than ``component_num``.
    # These are alpha-invariant: ki<c uses updated_params[ki], ki>c uses
    # current_params[ki]; both are fixed during this binary search.
    static_lps = {}  # {grid_id: {k: logpdf_array}}
    for grid_id, x_grid in grids:
        per_k = {}
        for k in range(K):
            if k == component_num:
                continue
            p_k = (
                updated_component_params[k] if k < component_num
                else current_component_params[k]
            )
            try:
                lp = _logpdf_for_check(x_grid, p_k, multivariate=True)
                lp = np.asarray(lp, dtype=float)
                lp[~np.isfinite(lp)] = -np.inf
            except Exception:
                # If a static component's density blows up, treat the whole
                # check as violated → roll back to alpha=0.
                return current_component_params[component_num]
            per_k[k] = lp
        static_lps[grid_id] = per_k

    cur_alt = current_component_params[component_num]

    def _interp_at(alpha):
        mu_i = (1 - alpha) * cur_alt[0] + alpha * candidate[0]
        Delta_i = (1 - alpha) * cur_alt[1] + alpha * candidate[1]
        Gamma_i = (1 - alpha) * cur_alt[2] + alpha * candidate[2]
        Gamma_i = 0.5 * (Gamma_i + Gamma_i.T)
        return (mu_i, Delta_i, Gamma_i)

    def _violated_at(alpha):
        var_p = _interp_at(alpha)
        for grid_id, x_grid in grids:
            try:
                var_lp = _logpdf_for_check(x_grid, var_p, multivariate=True)
                var_lp = np.asarray(var_lp, dtype=float)
                var_lp[~np.isfinite(var_lp)] = -np.inf
            except Exception:
                return True
            log_pdfs = []
            for k in range(K):
                log_pdfs.append(
                    var_lp if k == component_num else static_lps[grid_id][k]
                )
            if _adjacent_pair_violated(
                log_pdfs, min_log_ratio_range=min_log_ratio_range,
            ):
                return True
        return False

    if not _violated_at(1.0):
        return candidate_mu, candidate_Delta, candidate_Gamma

    lo, hi = 0.0, 1.0
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        if _violated_at(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break

    # The binary search used a probe grid frozen at alpha=1.  The outer
    # verification rebuilds the grid from the actual alpha=lo params (which
    # may point in a different direction in line mode) and can detect
    # violations the frozen-grid search missed.  Do a final proper check
    # here so the outer code doesn't have to revert.
    def _proper_violated_at(a):
        p = _interp_at(a)
        test = [
            (updated_component_params[ki] if ki < component_num
             else (p if ki == component_num
                   else current_component_params[ki]))
            for ki in range(K)
        ]
        return multicomponent_density_constraint_violated(
            test, xlims, multivariate=True, mode=constraint_mode,
            min_log_ratio_range=min_log_ratio_range,
        )

    for _ in range(10):
        if not _proper_violated_at(lo):
            break
        lo *= 0.5
    else:
        return current_component_params[component_num]

    return _interp_at(lo)


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
# EM iteration — unified (supports q=1 and q>1)
# ══════════════════════════════════════════════

def resolve_separation_config(constrained, multivariate, constraint_mode):
    """Separation feature config for a fit, or None when not applicable.

    Returns ``None`` for unconstrained fits and for univariate constrained
    fits (which keep the 1-D density-ratio constraint). For multivariate
    constrained fits returns ``{"tempering", "repulsion", "mode"}`` and raises
    on the deprecated 'line'/'marginal' modes.
    """
    if not (constrained and multivariate):
        return None
    return separation.resolve_constraint_mode(constraint_mode)


def em_iteration(observations, sample_indicators, current_component_params,
                 current_weights, constrained, xlims, multivariate=False,
                 cached_log_pdfs=None, return_log_pdfs=False, **kwargs):
    """EM iteration.

    Returns (updated_params, updated_weights, normalised_ll) by default,
    or (updated_params, updated_weights, normalised_ll, new_log_pdfs)
    when ``return_log_pdfs=True``. The LL is computed for free alongside
    the weight update — callers in fit.py should NOT call get_likelihood
    separately after this function.

    Parameters
    ----------
    cached_log_pdfs : list[ndarray] or None
        Per-sample (K, N_s) log-pdf matrices computed on
        ``current_component_params`` by the previous iteration's weight/LL
        pass. When supplied, the E-step skips the redundant density
        evaluation. Pass ``None`` for the first iteration or after a
        backtracking revert.
    return_log_pdfs : bool
        When True, also returns the freshly computed per-sample log-pdf
        matrices on ``updated_params`` so the caller can thread them
        forward as ``cached_log_pdfs`` for the next iteration.
    """
    mv = multivariate
    constraint_mode = kwargs.get("constraint_mode", separation.DEFAULT_CONSTRAINT_MODE)

    # Multivariate constrained fits use the separation features (responsibility
    # tempering + Bhattacharyya repulsion); the deprecated 'line'/'marginal'
    # density-ratio modes raise here. Univariate constrained fits retain the
    # working 1-D density-ratio constraint (binary search), unaffected by mode.
    sep_cfg = resolve_separation_config(constrained, mv, constraint_mode)

    if constrained and not mv and multicomponent_density_constraint_violated(
        current_component_params, xlims, multivariate=False
    ):
        import warnings
        warnings.warn(
            f"density constraint violated at start of em iteration "
            f"{kwargs.get('iterNum', -1)}; "
            "continuing — per-component line search will attempt to restore it"
        )

    S = sample_indicators.shape[1]
    K = len(current_component_params)
    assert current_weights.shape == (S, K)
    sample_indicators = validate_indicators(sample_indicators)

    responsibilities = sample_specific_responsibilities(
        observations, sample_indicators, current_component_params, current_weights,
        multivariate=mv, cached_log_pdfs=cached_log_pdfs,
    )

    # (1) Responsibility tempering — sharpen before the M-step.
    if sep_cfg is not None and sep_cfg["tempering"]:
        beta = separation.tempering_beta(kwargs.get("iterNum", 0), **kwargs)
        responsibilities = separation.apply_responsibility_tempering(
            responsibilities, beta
        )

    # The legacy per-component density line-search runs only for the retained
    # univariate constraint; MV separation disables it (sep_cfg is not None).
    mstep_constrained = constrained and sep_cfg is None
    # Stabilisers (covariance ridge + low-mass freeze) guard against the
    # degeneracy that responsibility sharpening exacerbates.
    kwargs["_stabilize_separation"] = sep_cfg is not None

    if not mv:
        updated_params = _em_update_univariate(
            observations, responsibilities, current_component_params,
            mstep_constrained, xlims, K, **kwargs
        )
    else:
        q = density_utils.get_q(current_component_params)
        if q > 1:
            updated_params = _em_update_cfusn(
                observations, responsibilities, current_component_params,
                mstep_constrained, xlims, K, q=q, **kwargs
            )
        else:
            updated_params = _em_update_multivariate(
                observations, responsibilities, current_component_params,
                mstep_constrained, xlims, K, **kwargs
            )

    # (2) Bhattacharyya repulsion — push overlapping components apart in joint
    # space after the closed-form M-step.
    if sep_cfg is not None and sep_cfg["repulsion"]:
        updated_params = separation.bhattacharyya_repulsion_step(
            updated_params, xlims, multivariate=mv, **kwargs
        )

    sw = kwargs.get("sample_weights", None)
    if return_log_pdfs:
        updated_weights, ll, new_log_pdfs = get_sample_weights_and_ll(
            observations, sample_indicators, updated_params, current_weights,
            multivariate=mv, return_log_pdfs=True, sample_weights=sw,
        )
        return updated_params, updated_weights, ll, new_log_pdfs

    updated_weights, ll = get_sample_weights_and_ll(
        observations, sample_indicators, updated_params, current_weights,
        multivariate=mv, sample_weights=sw,
    )
    return updated_params, updated_weights, ll



def _em_update_univariate(
    observations, responsibilities, current_component_params,
    constrained, xlims, K, **kwargs
):
    """One M-step for all components, univariate case."""
    sample_weights = kwargs.get("sample_weights", None)
    updated = [None] * K
    for c in range(K):
        z = responsibilities[c]
        cp = current_component_params[c]

        loc_cand = get_location_update(observations, z, cp,
                                       sample_weights=sample_weights)
        if constrained:
            loc_cand = get_constrained_location_update(
                loc_cand, c, current_component_params, updated, xlims, **kwargs
            )

        if kwargs.get("force_gaussian"):
            Delta_cand = 0.0
        else:
            Delta_cand = get_Delta_update(loc_cand, observations, z, cp,
                                          sample_weights=sample_weights)
            if constrained:
                Delta_cand = get_constrained_Delta_update(
                    Delta_cand, loc_cand, c, current_component_params, updated, xlims, **kwargs
                )

        Gamma_cand = get_Gamma_update(loc_cand, Delta_cand, observations, z, cp,
                                      sample_weights=sample_weights)
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
    """One M-step for all components, restricted MSN (q=1) case.
    component_params in alternate form: (mu, Delta_vec, Gamma_mat).
    """
    sample_weights = kwargs.get("sample_weights", None)
    stabilize = kwargs.get("_stabilize_separation", False)
    # Low-mass freeze: independently togglable from the separation/tempering
    # stabilizers above (`stabilize`) -- experimental, default off to match
    # existing committed behavior. See low_mass_freeze usage below.
    low_mass_freeze = stabilize or kwargs.get("low_mass_freeze", False)
    min_mass = separation.min_component_mass(**kwargs) if low_mass_freeze else 0.0
    updated = [None] * K
    for c in range(K):
        z = responsibilities[c]
        mu_old, Delta_old, Gamma_old = current_component_params[c]

        # Low-mass freeze (see _em_update_cfusn).
        if low_mass_freeze:
            z_mass = float((z if sample_weights is None else z * sample_weights).sum())
            if z_mass < min_mass:
                updated[c] = (mu_old, Delta_old, Gamma_old)
                continue

        mu_cand = get_location_update_mv(observations, z, mu_old, Delta_old, Gamma_old,
                                         sample_weights=sample_weights)
        Delta_cand = get_Delta_update_mv(mu_cand, observations, z, mu_old, Delta_old, Gamma_old,
                                         sample_weights=sample_weights)
        Gamma_cand = get_Gamma_update_mv(
            mu_cand, Delta_cand, observations, z, mu_old, Delta_old, Gamma_old,
            sample_weights=sample_weights,
        )

        # Same covariance-collapse regularisation as the CFUSN path.
        _n_eff = float((z if sample_weights is None else z * sample_weights).sum())
        Gamma_cand = _regularize_gamma_candidate(Gamma_cand, _n_eff, observations)

        if stabilize:
            Gamma_cand = separation.regularize_gamma(Gamma_cand, xlims, **kwargs)

        # Enforce positive-definiteness of Gamma. Unconditional (not gated
        # behind `stabilize`, which only ever turns on for constrained
        # multivariate fits -- see resolve_separation_config): an
        # ill-conditioned/degenerate M-step candidate (e.g. from a
        # near-zero-responsibility component) can make LAPACK's eigensolver
        # itself fail to converge, not just return small/negative
        # eigenvalues. Unconstrained multivariate fits (this pipeline's
        # default everywhere) had zero protection against this -- one
        # non-convergent candidate used to kill the entire tryToFit attempt.
        Gamma_cand, gamma_ok = _psd_floor_or_reject(Gamma_cand)
        if not gamma_ok:
            updated[c] = (mu_old, Delta_old, Gamma_old)
            continue

        # Sanity guard against Δ blowups (see _em_update_cfusn).
        if stabilize and not separation.params_sane(
            mu_cand, Delta_cand, Gamma_cand, xlims, **kwargs
        ):
            updated[c] = (mu_old, Delta_old, Gamma_old)
            continue

        if constrained:
            try:
                mu_cand, Delta_cand, Gamma_cand = get_constrained_update_mv(
                    mu_cand, Delta_cand, Gamma_cand,
                    c, current_component_params, updated,
                    xlims, multivariate=True, **kwargs
                )
            except Exception as e:
                if kwargs.get("raise_on_error", False):
                    raise
                import warnings
                warnings.warn(f"Constraint enforcement failed for component {c}: {e}.")
                mu_cand, Delta_cand, Gamma_cand = mu_old, Delta_old, Gamma_old

        updated[c] = (mu_cand, Delta_cand, Gamma_cand)

        if constrained:
            test_params = [*updated[:c + 1], *current_component_params[c + 1:]]
            try:
                violated = multicomponent_density_constraint_violated(
                    test_params, xlims, multivariate=True,
                    mode=kwargs.get("constraint_mode", "line"),
                )
            except Exception:
                violated = False
            if violated:
                if kwargs.get("raise_on_error", False):
                    raise ValueError(
                        f"constraint violated after component {c} "
                        f"iter {kwargs.get('iterNum', -1)}"
                    )
                import warnings
                warnings.warn(
                    f"Constraint violated after component {c} "
                    f"iter {kwargs.get('iterNum', -1)}. Reverting."
                )
                updated[c] = (mu_old, Delta_old, Gamma_old)

    return updated


def _em_update_cfusn(
    observations, responsibilities, current_component_params,
    constrained, xlims, K, q=2, **kwargs
):
    """One M-step for all components, CFUSN (q > 1) case.
    component_params in alternate form: (mu, Delta_mat, Gamma_mat)
    where Delta_mat is (p, q).
    """
    n_mc = kwargs.get("n_mc_truncated", 500)
    sample_weights = kwargs.get("sample_weights", None)
    stabilize = kwargs.get("_stabilize_separation", False)
    # Low-mass freeze: independently togglable from the separation/tempering
    # stabilizers above (`stabilize`) -- experimental, default off to match
    # existing committed behavior. See low_mass_freeze usage below.
    low_mass_freeze = stabilize or kwargs.get("low_mass_freeze", False)
    min_mass = separation.min_component_mass(**kwargs) if low_mass_freeze else 0.0
    updated = [None] * K

    for c in range(K):
        z = responsibilities[c]  # (N,)
        mu_old, Delta_old, Gamma_old = current_component_params[c]
        Delta_old = density_utils._ensure_matrix_delta(Delta_old)

        # Low-mass freeze: a component that has lost (almost) all of its
        # responsibility mass has an ill-defined mean — keep it put rather than
        # letting the M-step send it to a degenerate spike outside the data.
        if low_mass_freeze:
            z_mass = float((z if sample_weights is None else z * sample_weights).sum())
            if z_mass < min_mass:
                updated[c] = (mu_old, Delta_old, Gamma_old)
                continue

        rng = kwargs.get("rng") or np.random.RandomState()

        # --- M-step using Lin (2009) equations ---
        # Step 1: location + compute eta, Psi
        # `completed` is None for fully-observed data (the M-step is then
        # exactly as before); otherwise it carries the completed-data moments
        # so all three steps maximise Q rather than an available-case proxy.
        mu_cand, eta, Psi, completed = get_location_update_cfusn(
            observations, z, mu_old, Delta_old, Gamma_old, n_mc=n_mc, rng=rng,
            sample_weights=sample_weights, return_completed=True,
        )

        # Step 2: Delta (p, q)
        Delta_cand = get_Delta_update_cfusn(mu_cand, observations, z, eta, Psi,
                                            sample_weights=sample_weights,
                                            completed=completed,
                                            Delta_prev=Delta_old)

        # Step 3: Gamma (p, p)
        Gamma_cand = get_Gamma_update_cfusn(mu_cand, Delta_cand, observations, z, eta, Psi,
                                            sample_weights=sample_weights,
                                            completed=completed)

        # Bound the unbounded-likelihood degeneracy (a component's covariance
        # collapsing onto a lower-dimensional subspace). Applied to every
        # multivariate fit, not just constrained ones: separation.regularize_gamma
        # below is gated behind `stabilize`, and _psd_floor_or_reject only
        # rescues an already non-PSD matrix, so unconstrained fits -- this
        # pipeline's default -- previously had no covariance floor at all.
        _n_eff = float((z if sample_weights is None else z * sample_weights).sum())
        Gamma_cand = _regularize_gamma_candidate(Gamma_cand, _n_eff, observations)

        # Covariance ridge: floor the scale so a sharpened component can't
        # collapse to a near-degenerate spike.
        if stabilize:
            Gamma_cand = separation.regularize_gamma(Gamma_cand, xlims, **kwargs)

        # Enforce positive-definiteness of Gamma. Unconditional (not gated
        # behind `stabilize`, which only turns on for constrained
        # multivariate fits -- see resolve_separation_config): an
        # ill-conditioned/degenerate M-step candidate can make LAPACK's
        # eigensolver itself fail to converge, not just return small/
        # negative eigenvalues. Unconstrained multivariate fits (this
        # pipeline's default everywhere) had zero protection against this --
        # one non-convergent candidate used to kill the entire tryToFit
        # attempt.
        Gamma_cand, gamma_ok = _psd_floor_or_reject(Gamma_cand)
        if not gamma_ok:
            updated[c] = (mu_old, Delta_old, Gamma_old)
            continue

        # Sanity guard: a degenerate per-component moment matrix can drive the
        # Δ solve to extreme values (centroid flung to ~1e9). Reject such a
        # candidate and keep the previous params for this component.
        if stabilize and not separation.params_sane(
            mu_cand, Delta_cand, Gamma_cand, xlims, **kwargs
        ):
            updated[c] = (mu_old, Delta_old, Gamma_old)
            continue

        # Constraint enforcement via line search
        if constrained:
            try:
                mu_cand, Delta_cand, Gamma_cand = get_constrained_update_mv(
                    mu_cand, Delta_cand, Gamma_cand,
                    c, current_component_params, updated,
                    xlims, multivariate=True, **kwargs
                )
            except Exception as e:
                if kwargs.get("raise_on_error", False):
                    raise
                import warnings
                warnings.warn(f"CFUSN constraint failed component {c}: {e}.")
                mu_cand, Delta_cand, Gamma_cand = mu_old, Delta_old, Gamma_old

        updated[c] = (mu_cand, Delta_cand, Gamma_cand)

        # Verify constraint
        if constrained:
            test_params = [*updated[:c + 1], *current_component_params[c + 1:]]
            try:
                violated = multicomponent_density_constraint_violated(
                    test_params, xlims, multivariate=True,
                    mode=kwargs.get("constraint_mode", "line"),
                )
            except Exception:
                violated = False
            if violated:
                if kwargs.get("raise_on_error", False):
                    raise ValueError(
                        f"CFUSN constraint violated after component {c}"
                    )
                import warnings
                warnings.warn(f"CFUSN constraint violated component {c}. Reverting.")
                updated[c] = (mu_old, Delta_old, Gamma_old)

    return updated