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
from scipy.special import logsumexp
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm


# ══════════════════════════════════════════════
# Truncated-normal moments — scalar (q=1, unchanged)
# ══════════════════════════════════════════════

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

    q=2 KEY FIX: fully vectorized over N — zero Python loop.
    The approximation (independent marginals + correlation correction for
    the off-diagonal of Psi) is unchanged from the original; only the loop
    is removed.

    For q != 2 the original MC path is used unchanged.
    """
    if rng is None:
        rng = np.random.RandomState()

    N, q = means.shape

    # ── q=2 fast path — vectorized ──────────────────────────────────────
    if q == 2:
        cov  = 0.5 * (cov + cov.T)
        eig  = np.linalg.eigvalsh(cov)
        if eig.min() < 1e-12:
            cov = cov + (1e-12 - eig.min() + 1e-12) * np.eye(2)

        std  = np.sqrt(np.diag(cov))                          # (2,)
        corr = np.clip(cov[0, 1] / (std[0] * std[1] + 1e-15), -0.9999, 0.9999)

        # All (N, 2) operations — no Python loop
        alpha   = means / std[None, :]                        # (N, 2)
        phi_a   = norm.pdf(alpha)                             # (N, 2)
        Phi_a   = norm.cdf(alpha)                             # (N, 2)
        safe    = Phi_a > 1e-12
        ratio   = np.where(safe,
                           phi_a / np.where(safe, Phi_a, 1.0),
                           np.abs(alpha))                     # (N, 2)

        eta     = means + std[None, :] * ratio                # (N, 2)
        diag_Ps = std[None, :]**2 + means**2 + std[None, :] * means * ratio  # (N, 2)
        cross   = eta[:, 0] * eta[:, 1] + corr * std[0] * std[1]             # (N,)

        Psi             = np.zeros((N, 2, 2))
        Psi[:, 0, 0]    = diag_Ps[:, 0]
        Psi[:, 1, 1]    = diag_Ps[:, 1]
        Psi[:, 0, 1]    = cross
        Psi[:, 1, 0]    = cross
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

def get_location_update_cfusn(observations, responsibilities, mu, Delta, Gamma,
                               n_mc=500, rng=None, sample_weights=None):
    """CFUSN location M-step. Returns (mu_new, eta, Psi) — moments reused downstream."""
    Delta      = density_utils._ensure_matrix_delta(Delta)
    eta, Psi   = get_truncated_normal_moments_cfusn(
        observations, mu, Delta, Gamma, n_mc=n_mc, rng=rng
    )
    obs    = ~np.isnan(observations)                 # (N, p)
    x_fill = np.where(obs, observations, 0.0)
    z      = responsibilities                         # (N,)
    z_eff  = z if sample_weights is None else z * sample_weights

    # KEY FIX: single matrix op replaces per-dimension loop
    # Delta_eta[j, d] = sum_r Delta[d, r] * eta[j, r]
    Delta_eta = eta @ Delta.T                         # (N, p)
    obs_z     = obs * z_eff[:, None]                  # (N, p)
    numer     = (obs_z * (x_fill - Delta_eta)).sum(0) # (p,)
    denom     = obs_z.sum(0)                           # (p,)
    mu_new    = np.where(denom > 1e-12, numer / np.maximum(denom, 1e-12), mu)

    return mu_new, eta, Psi


def get_Delta_update_cfusn(mu_new, observations, responsibilities, eta, Psi,
                           sample_weights=None):
    """CFUSN Delta M-step.  Returns (p, q).

    KEY FIX: replace per-d einsum loop with two batched einsums, then
    solve per-dimension (p=2 → only 2 solves of 2×2 systems).
    """
    obs    = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    z      = responsibilities
    z_eff  = z if sample_weights is None else z * sample_weights
    N, p   = observations.shape
    q      = eta.shape[1]

    obs_z    = obs * z_eff[:, None]                   # (N, p)
    residuals = x_fill - mu_new                       # (N, p)

    # numer[d, r] = sum_n obs_z[n,d] * residuals[n,d] * eta[n,r]
    numer    = (obs_z * residuals).T @ eta             # (p, q) — one matmul

    # Psi_sum[d, i, j] = sum_n obs_z[n,d] * Psi[n, i, j]
    Psi_sum  = np.einsum('nd,nij->dij', obs_z, Psi)   # (p, q, q)

    Delta_new = np.zeros((p, q))
    for d in range(p):                                 # p=2 → 2 iterations
        Ps = Psi_sum[d]
        eig = np.linalg.eigvalsh(Ps)
        if eig.min() < 1e-10:
            Ps = Ps + (1e-10 - eig.min() + 1e-10) * np.eye(q)
        try:
            Delta_new[d] = np.linalg.solve(Ps, numer[d])
        except np.linalg.LinAlgError:
            Delta_new[d] = numer[d] / np.maximum(np.diag(Ps), 1e-12)

    return Delta_new


def get_Gamma_update_cfusn(mu_new, Delta_new, observations, responsibilities, eta, Psi,
                           sample_weights=None):
    """CFUSN Gamma M-step.  Returns (p, p).

    KEY FIX: precompute Psi_minus once; use vectorized outer-product ops
    for term1 and a single batched einsum for term2, replacing repeated
    per-(d1,d2) einsums inside the double loop.
    """
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

def get_location_update_mv(observations, responsibilities, mu, Delta, Gamma,
                           sample_weights=None):
    """mu update for q=1 restricted MSN. Delta is (p,) vector."""
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    obs = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    Delta_vec = np.asarray(Delta).ravel()
    m = x_fill - v[:, None] * Delta_vec[None, :]
    r = responsibilities if sample_weights is None else responsibilities * sample_weights
    z = r[:, None]
    numer = (m * z * obs).sum(axis=0)
    denom = (z * obs).sum(axis=0)
    return numer / np.maximum(denom, 1e-12)


def get_Delta_update_mv(updated_mu, observations, responsibilities, mu, Delta, Gamma,
                       sample_weights=None):
    """Delta update for q=1 restricted MSN. Returns (p,) vector."""
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    obs = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    residuals = x_fill - updated_mu[None, :]
    z = responsibilities if sample_weights is None else responsibilities * sample_weights
    numer = (z[:, None] * v[:, None] * residuals * obs).sum(axis=0)
    denom = (z[:, None] * w[:, None] * obs).sum(axis=0)
    return numer / np.maximum(denom, 1e-12)


def get_Gamma_update_mv(updated_mu, updated_Delta, observations, responsibilities, mu, Delta, Gamma,
                        sample_weights=None):
    """Gamma update for q=1 restricted MSN. Returns (p, p)."""
    v, w = get_truncated_normal_moments_mv_missing(observations, mu, Delta, Gamma)
    N, K = observations.shape
    obs = ~np.isnan(observations)
    x_fill = np.where(obs, observations, 0.0)
    z = responsibilities if sample_weights is None else responsibilities * sample_weights
    updated_Delta = np.asarray(updated_Delta).ravel()
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
    min_mass = separation.min_component_mass(**kwargs) if stabilize else 0.0
    updated = [None] * K
    for c in range(K):
        z = responsibilities[c]
        mu_old, Delta_old, Gamma_old = current_component_params[c]

        # Low-mass freeze (see _em_update_cfusn).
        if stabilize:
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

        if stabilize:
            Gamma_cand = separation.regularize_gamma(Gamma_cand, xlims, **kwargs)

        eigvals = np.linalg.eigvalsh(Gamma_cand)
        if eigvals.min() < 1e-8:
            Gamma_cand += (1e-8 - eigvals.min()) * np.eye(Gamma_cand.shape[0])

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
    min_mass = separation.min_component_mass(**kwargs) if stabilize else 0.0
    updated = [None] * K

    for c in range(K):
        z = responsibilities[c]  # (N,)
        mu_old, Delta_old, Gamma_old = current_component_params[c]
        Delta_old = density_utils._ensure_matrix_delta(Delta_old)

        # Low-mass freeze: a component that has lost (almost) all of its
        # responsibility mass has an ill-defined mean — keep it put rather than
        # letting the M-step send it to a degenerate spike outside the data.
        if stabilize:
            z_mass = float((z if sample_weights is None else z * sample_weights).sum())
            if z_mass < min_mass:
                updated[c] = (mu_old, Delta_old, Gamma_old)
                continue

        rng = kwargs.get("rng") or np.random.RandomState()

        # --- M-step using Lin (2009) equations ---
        # Step 1: location + compute eta, Psi
        mu_cand, eta, Psi = get_location_update_cfusn(
            observations, z, mu_old, Delta_old, Gamma_old, n_mc=n_mc, rng=rng,
            sample_weights=sample_weights,
        )

        # Step 2: Delta (p, q)
        Delta_cand = get_Delta_update_cfusn(mu_cand, observations, z, eta, Psi,
                                            sample_weights=sample_weights)

        # Step 3: Gamma (p, p)
        Gamma_cand = get_Gamma_update_cfusn(mu_cand, Delta_cand, observations, z, eta, Psi,
                                            sample_weights=sample_weights)

        # Covariance ridge: floor the scale so a sharpened component can't
        # collapse to a near-degenerate spike.
        if stabilize:
            Gamma_cand = separation.regularize_gamma(Gamma_cand, xlims, **kwargs)

        # Enforce positive-definiteness of Gamma
        eigvals = np.linalg.eigvalsh(Gamma_cand)
        if eigvals.min() < 1e-8:
            Gamma_cand += (1e-8 - eigvals.min()) * np.eye(Gamma_cand.shape[0])

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