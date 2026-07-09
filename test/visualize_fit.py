"""
Visualize multivariate calibration fit using bootstrap-percentile densities
and pre-computed evidence assignments from MVCalibrationAnalysis.run().

Supports both restricted MSN (q=1, Delta is a p-vector) and
CFUSN (q>=1, Delta is a p×q matrix) parameterizations.
Detection is automatic based on Delta shape in component params.

All evidence points displayed come directly from analysis.results[config]['points'].
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as mvn, norm
from scipy.ndimage import uniform_filter1d
from joblib import Parallel, delayed
from src.assay_calibration.fit_utils.cfusn.density_utils import get_q


# ──────────────────────────────────────
# Shared constants
# ──────────────────────────────────────
SAMPLE_COLORS = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
SAMPLE_NAMES_DEFAULT = ['Pathogenic/Likely Pathogenic', 'Benign/Likely Benign',
                        'population', 'Synonymous']
SAMPLE_MARKERS = ['o', 's', '^', 'D']
# Thin borders for row-0 scatter: dark enough to read against the evidence colormap
_SAMPLE_EDGE_COLORS = ['#8B0000', '#00008B', '#303030', '#1A5E2A']  # dark red/blue/gray/green

POINT_CMAP_COLORS = [
    (0.0, '#08306b'), (0.3, '#4292c6'), (0.45, '#c6dbef'),
    (0.5, '#f7f7f7'),
    (0.55, '#fcbba1'), (0.7, '#ef3b2c'), (1.0, '#67000d'),
]
POINT_CMAP = LinearSegmentedColormap.from_list('evidence', POINT_CMAP_COLORS)


# ──────────────────────────────────────
# Density helpers — unified for q=1 and q>1
# ──────────────────────────────────────

def _mvn_logcdf_batch(uppers, mean, cov):
    """Batch log-CDF of multivariate normal for (N, q) upper limits."""
    N, q = uppers.shape
    if q == 1:
        sigma = np.sqrt(max(float(np.asarray(cov).ravel()[0]), 1e-15))
        return norm.logcdf((uppers[:, 0] - mean[0]) / sigma)
    try:
        rv = mvn(mean=mean, cov=cov, allow_singular=True)
        vals = rv.cdf(uppers)   # vectorised: (N,) in one C-level call
        return np.log(np.maximum(vals, 1e-300))
    except Exception:
        return np.full(N, -np.inf)


def _sn_logpdf(x, mu, Delta, Gamma):
    """Unified skew-normal log-density. Auto-dispatches based on Delta shape.

    Parameters
    ----------
    x : (N, p) with possible NaN
    mu : (p,)
    Delta : (p,) for q=1  OR  (p, q) for CFUSN q>1
    Gamma : (p, p)

    Returns
    -------
    (N,) log-density
    """
    Delta = np.asarray(Delta, dtype=float)
    if Delta.ndim == 2 and Delta.shape[1] > 1:
        return _cfusn_logpdf(x, mu, Delta, Gamma)
    else:
        return _msn_logpdf_q1(x, mu, np.atleast_1d(Delta).ravel(), Gamma)


def _msn_logpdf_q1(x, mu, Delta, Gamma):
    """MSN log-density for q=1 (Delta is a p-vector). Handles NaN via marginals."""
    mu = np.asarray(mu, float)
    Delta = np.asarray(Delta, float).ravel()
    Gamma = np.asarray(Gamma, float)
    x = np.atleast_2d(np.asarray(x, float))
    N, K = x.shape
    log_pdf = np.zeros(N)
    obs_mask = ~np.isnan(x)
    patterns = {}
    for j in range(N):
        patterns.setdefault(tuple(obs_mask[j]), []).append(j)
    for pat, indices in patterns.items():
        obs_dims = np.array([i for i, o in enumerate(pat) if o])
        if len(obs_dims) == 0:
            continue
        idx = np.array(indices)
        mu_s, Delta_s = mu[obs_dims], Delta[obs_dims]
        Gamma_s = Gamma[np.ix_(obs_dims, obs_dims)]
        Omega_s = Gamma_s + np.outer(Delta_s, Delta_s)
        Omega_s = 0.5 * (Omega_s + Omega_s.T)
        eigv = np.linalg.eigvalsh(Omega_s)
        if eigv.min() < 1e-10:
            Omega_s += (1e-10 - eigv.min() + 1e-10) * np.eye(len(obs_dims))
        try:
            log_phi = mvn(mean=mu_s, cov=Omega_s, allow_singular=True).logpdf(
                x[np.ix_(idx, obs_dims)])
            Oid = np.linalg.solve(Omega_s, Delta_s)
            eta = (x[np.ix_(idx, obs_dims)] - mu_s) @ Oid
            s2 = max(1.0 - Delta_s @ Oid, 1e-12)
            lp = np.log(2) + log_phi + norm.logcdf(eta / np.sqrt(s2))
            lp = np.atleast_1d(np.asarray(lp, float))
            lp[~np.isfinite(lp)] = -np.inf
            log_pdf[idx] = lp
        except Exception:
            log_pdf[idx] = -np.inf
    return log_pdf


def _cfusn_logpdf(x, mu, Delta, Gamma):
    """CFUSN log-density for q>=1 (Delta is p×q matrix). Handles NaN via marginals.

    f(x) = 2^q * phi_p(x; mu, Omega) * Phi_q(Delta' Omega^{-1} (x-mu); 0, D)
    where Omega = Gamma + Delta Delta', D = I_q - Delta' Omega^{-1} Delta

    Marginal property: for observed set S,
        X_S ~ CFUSN(mu_S, Delta_S, Gamma_{SS})
    where Delta_S is the row-subset of Delta.
    """
    mu = np.asarray(mu, float)
    Delta = np.asarray(Delta, float)
    Gamma = np.asarray(Gamma, float)
    x = np.atleast_2d(np.asarray(x, float))
    N, p = x.shape
    q = Delta.shape[1]
    log_pdf = np.zeros(N)

    obs_mask = ~np.isnan(x)
    patterns = {}
    for j in range(N):
        patterns.setdefault(tuple(obs_mask[j]), []).append(j)

    for pat, indices in patterns.items():
        obs_dims = np.array([i for i, o in enumerate(pat) if o])
        if len(obs_dims) == 0:
            continue
        idx = np.array(indices)
        p_s = len(obs_dims)

        mu_s = mu[obs_dims]
        Delta_s = Delta[obs_dims, :]  # (p_s, q)
        Gamma_s = Gamma[np.ix_(obs_dims, obs_dims)]

        Omega_s = Gamma_s + Delta_s @ Delta_s.T
        Omega_s = 0.5 * (Omega_s + Omega_s.T)
        eigv = np.linalg.eigvalsh(Omega_s)
        if eigv.min() < 1e-10:
            Omega_s += (1e-10 - eigv.min() + 1e-10) * np.eye(p_s)

        try:
            log_phi = mvn(mean=mu_s, cov=Omega_s, allow_singular=True).logpdf(
                x[np.ix_(idx, obs_dims)])
        except Exception:
            log_pdf[idx] = -np.inf
            continue

        try:
            Omega_s_inv_Delta_s = np.linalg.solve(Omega_s, Delta_s)  # (p_s, q)
        except np.linalg.LinAlgError:
            log_pdf[idx] = -np.inf
            continue

        D_s = np.eye(q) - Delta_s.T @ Omega_s_inv_Delta_s  # (q, q)
        D_s = 0.5 * (D_s + D_s.T)
        eig_D = np.linalg.eigvalsh(D_s)
        if eig_D.min() < 1e-10:
            D_s += (1e-10 - eig_D.min() + 1e-10) * np.eye(q)

        residuals = x[np.ix_(idx, obs_dims)] - mu_s  # (n_idx, p_s)
        eta = residuals @ Omega_s_inv_Delta_s  # (n_idx, q)

        log_Phi = _mvn_logcdf_batch(eta, np.zeros(q), D_s)

        lp = q * np.log(2) + log_phi + log_Phi
        lp = np.atleast_1d(np.asarray(lp, float))
        lp[~np.isfinite(lp)] = -np.inf
        log_pdf[idx] = lp

    return log_pdf


# ──────────────────────────────────────
# Bootstrap density computations
# ──────────────────────────────────────

def _reconstitute_fit_dict(fit_raw):
    """Standalone reconstitution — no analysis object needed (safe to pickle)."""
    inner = fit_raw.get('fit', fit_raw)
    if inner is None:
        return None
    cp = inner.get('component_params', [])
    if not cp or any(len(p) == 0 for p in cp):
        return None
    params = []
    for p in cp:
        mu    = np.array(p[0], dtype=float)
        Delta = np.array(p[1], dtype=float)
        Gamma = np.array(p[2], dtype=float)
        if Delta.ndim == 2 and Delta.shape[1] == 1:
            Delta = Delta.ravel()
        params.append((mu, Delta, Gamma))
    weights = np.array(inner['weights'], dtype=float)
    return {'component_params': params, 'weights': weights,
            'xlims': inner.get('xlims'),
            'latent_q': inner.get('latent_q', get_q(params))}


def _collect_valid_fits(analysis, config):
    """Collect all valid reconstituted fits for a config (parallel, loky)."""
    candidates = [
        boot_data[config]
        for boot_data in analysis.raw_boots.values()
        if config in boot_data and boot_data[config] is not None
    ]
    results = Parallel(n_jobs=-1)(
        delayed(_reconstitute_fit_dict)(fit_raw)
        for fit_raw in candidates
    )
    return [r for r in results if r is not None]


def _eval_all_marginals(fit, x_2d_list, p_idx, b_idx, s_idx, benign_method, S,
                        g_idx=None, prior=None):
    """Evaluate marginal LR+ and sample densities for one bootstrap across all dimensions.

    x_2d_list : list of D arrays, each (n_grid, total_dims) with one column set.
    Returns   : list of D results, each the output of _eval_marginal_fit.
    """
    return [
        _eval_marginal_fit(fit, x_2d, p_idx, b_idx, s_idx, benign_method, S,
                           g_idx=g_idx, prior=prior)
        for x_2d in x_2d_list
    ]


def _eval_fit_on_grid(fit, grid_pts, p_idx, b_idx, s_idx, benign_method,
                      g_idx=None, prior=None):
    """Evaluate LR+ for one bootstrap fit on a pre-built grid of points."""
    params = fit['component_params']
    weights = fit['weights']
    K = len(params)
    n_w = len(weights)

    nu_mode = p_idx is None and (b_idx is not None or s_idx is not None) and g_idx is not None
    pu_mode = p_idx is not None and b_idx is None and s_idx is None and g_idx is not None

    comp_log = [_sn_logpdf(grid_pts, *params[c]) for c in range(K)]

    s_valid = s_idx is not None and s_idx < n_w
    b_valid = b_idx is not None and b_idx < n_w

    def _benign_log():
        if s_valid and benign_method == 'synonymous':
            w_b = weights[s_idx]
        elif s_valid and b_valid and benign_method == 'avg':
            w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
        elif b_valid:
            w_b = weights[b_idx]
        elif s_valid:
            w_b = weights[s_idx]
        else:
            return None
        return logsumexp([np.log(w_b[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)

    if nu_mode:
        log_fb = _benign_log()
        if log_fb is None:
            return np.zeros(len(grid_pts))
        log_fpop = logsumexp([np.log(weights[g_idx][c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        _p = prior if prior is not None else 0.1
        fp = np.maximum((np.exp(log_fpop) - (1 - _p) * np.exp(log_fb)) / _p,
                        np.exp(log_fpop) * 1e-10)
        return np.log(fp) - log_fb
    elif pu_mode:
        w_p = weights[p_idx]
        log_fp = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        log_fpop = logsumexp([np.log(weights[g_idx][c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        _p = prior if prior is not None else 0.1
        fb = np.maximum((np.exp(log_fpop) - _p * np.exp(log_fp)) / (1 - _p),
                        np.exp(log_fpop) * 1e-10)
        return log_fp - np.log(fb)
    else:
        if p_idx is None or p_idx >= n_w:
            return np.zeros(len(grid_pts))
        w_p = weights[p_idx]
        log_fb = _benign_log()
        if log_fb is None:
            return np.zeros(len(grid_pts))
        log_fp = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        return log_fp - log_fb


def _eval_fit_on_grid_batch_p(fit, grid_pts, p_indices, b_idx, s_idx, benign_method,
                               g_idx=None, prior=None):
    """Evaluate LR+ on a grid for multiple pathogenic indices in one shot.

    Component log-densities (_sn_logpdf) are computed ONCE and shared across
    all p_indices, avoiding redundant density evaluations.

    Returns list of len(p_indices) flat lr arrays.
    """
    params = fit['component_params']
    weights = fit['weights']
    K = len(params)
    n_w = len(weights)

    comp_log = [_sn_logpdf(grid_pts, *params[c]) for c in range(K)]

    s_valid = s_idx is not None and s_idx < n_w
    b_valid = b_idx is not None and b_idx < n_w

    if s_valid and benign_method == 'synonymous':
        w_b = weights[s_idx]
    elif s_valid and b_valid and benign_method == 'avg':
        w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
    elif b_valid:
        w_b = weights[b_idx]
    elif s_valid:
        w_b = weights[s_idx]
    else:
        w_b = None

    log_fb = (logsumexp([np.log(w_b[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
              if w_b is not None else None)
    log_fpop = (logsumexp([np.log(weights[g_idx][c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
                if g_idx is not None and g_idx < n_w else None)
    _p = prior if prior is not None else 0.1

    out = []
    for p_idx in p_indices:
        nu = p_idx is None and log_fb is not None and log_fpop is not None
        pu = p_idx is not None and w_b is None and log_fpop is not None and (p_idx < n_w)
        if nu:
            fp = np.maximum((np.exp(log_fpop) - (1 - _p) * np.exp(log_fb)) / _p,
                            np.exp(log_fpop) * 1e-10)
            out.append(np.log(fp) - log_fb)
        elif pu:
            w_p = weights[p_idx]
            log_fp = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
            fb = np.maximum((np.exp(log_fpop) - _p * np.exp(log_fp)) / (1 - _p),
                            np.exp(log_fpop) * 1e-10)
            out.append(log_fp - np.log(fb))
        elif p_idx is None or p_idx >= n_w or log_fb is None:
            out.append(np.full(len(grid_pts), np.nan))
        else:
            w_p = weights[p_idx]
            log_fp = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
            out.append(log_fp - log_fb)
    return out


def _eval_all_marginals_batch_p(fit, x_2d_list, p_indices, b_idx, s_idx, benign_method, S,
                                g_idx=None, prior=None):
    """Evaluate marginals for multiple pathogenic indices in one shot.

    Component log-densities computed ONCE per dimension and shared across all
    p_indices.  sample_logs / comp_logs are also identical across p_indices.

    Returns list of len(p_indices), each a list of D results
    (lr_1d, sample_logs, comp_logs_per_sample).
    """
    params = fit['component_params']
    weights = fit['weights']
    K = len(params)
    n_w = len(weights)

    s_valid = s_idx is not None and s_idx < n_w
    b_valid = b_idx is not None and b_idx < n_w

    # Benign weights for standard/NU modes
    if s_valid and benign_method == 'synonymous':
        w_b = weights[s_idx]
    elif s_valid and b_valid and benign_method == 'avg':
        w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
    elif b_valid:
        w_b = weights[b_idx]
    elif s_valid:
        w_b = weights[s_idx]
    else:
        w_b = None  # PU mode: no benign available

    w_ps = [weights[p_idx] if (p_idx is not None and p_idx < n_w) else None for p_idx in p_indices]
    results_per_p = [[] for _ in p_indices]

    for x_2d in x_2d_list:
        comp_log = [_sn_logpdf(x_2d, *params[c]) for c in range(K)]

        sample_logs, comp_logs_per_sample = [], []
        for s in range(S):
            if s >= n_w:
                sample_logs.append(None); comp_logs_per_sample.append(None)
                continue
            w_s = weights[s]
            log_d = logsumexp([np.log(w_s[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
            sample_logs.append(log_d)
            comp_logs_per_sample.append([np.log(w_s[c] + 1e-300) + comp_log[c] for c in range(K)])

        log_fb = (logsumexp([np.log(w_b[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
                  if w_b is not None else None)
        log_fpop = (logsumexp([np.log(weights[g_idx][c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
                    if g_idx is not None and g_idx < n_w else None)
        _p = prior if prior is not None else 0.1

        for pi, w_p in enumerate(w_ps):
            p_idx_i = p_indices[pi]
            nu = p_idx_i is None and log_fb is not None and log_fpop is not None
            pu = p_idx_i is not None and w_p is not None and w_b is None and log_fpop is not None
            if nu:
                fp = np.maximum((np.exp(log_fpop) - (1 - _p) * np.exp(log_fb)) / _p,
                                np.exp(log_fpop) * 1e-10)
                results_per_p[pi].append((np.log(fp) - log_fb, sample_logs, comp_logs_per_sample))
            elif pu:
                log_fp = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
                fb = np.maximum((np.exp(log_fpop) - _p * np.exp(log_fp)) / (1 - _p),
                                np.exp(log_fpop) * 1e-10)
                results_per_p[pi].append((log_fp - np.log(fb), sample_logs, comp_logs_per_sample))
            elif w_p is None or log_fb is None:
                results_per_p[pi].append((np.full(len(x_2d), np.nan), sample_logs, comp_logs_per_sample))
            else:
                log_fp = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
                results_per_p[pi].append((log_fp - log_fb, sample_logs, comp_logs_per_sample))

    return results_per_p


def _eval_fit_sample_densities(fit, grid_pts, n_samples):
    """Evaluate per-sample log-density for one bootstrap fit on a grid.

    Returns list of length n_samples, each a 1-D log-density array.
    """
    params = fit['component_params']
    weights = fit['weights']
    K = len(params)
    comp_logs = [_sn_logpdf(grid_pts, *params[c]) for c in range(K)]
    result = []
    for s in range(n_samples):
        w_s = weights[s]
        result.append(logsumexp(
            [np.log(w_s[c] + 1e-300) + comp_logs[c] for c in range(K)],
            axis=0,
        ))
    return result


def _eval_marginal_fit(fit, x_2d, p_idx, b_idx, s_idx, benign_method, S,
                       g_idx=None, prior=None):
    """Evaluate LR+ and per-sample densities for one bootstrap on a 1-D marginal grid.

    Module-level so joblib loky workers can pickle it by name.

    Returns
    -------
    lr_1d : (n,) log-LR array
    sample_logs : list of S arrays, each (n,), or None if sample absent in this fit
    comp_logs_per_sample : list of S lists (or None for absent samples)
    """
    params = fit['component_params']
    weights = fit['weights']
    K = len(params)
    n_w = len(weights)

    comp_log = [_sn_logpdf(x_2d, *params[c]) for c in range(K)]

    sample_logs = []
    comp_logs_per_sample = []
    for s in range(S):
        if s >= n_w:
            sample_logs.append(None)
            comp_logs_per_sample.append(None)
            continue
        w_s = weights[s]
        log_d = logsumexp([np.log(w_s[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        sample_logs.append(log_d)
        comp_logs_per_sample.append([np.log(w_s[c] + 1e-300) + comp_log[c] for c in range(K)])

    nu_mode = p_idx is None and (b_idx is not None or s_idx is not None) and g_idx is not None
    pu_mode = p_idx is not None and b_idx is None and s_idx is None and g_idx is not None

    s_valid = s_idx is not None and s_idx < n_w
    if nu_mode:
        if s_valid and benign_method == 'synonymous':
            w_b = weights[s_idx]
        elif s_valid and b_idx is not None and b_idx < n_w and benign_method == 'avg':
            w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
        elif b_idx is not None and b_idx < n_w:
            w_b = weights[b_idx]
        else:
            w_b = weights[s_idx]
        log_fb   = logsumexp([np.log(w_b[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        log_fpop = logsumexp([np.log(weights[g_idx][c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        _p = prior if prior is not None else 0.1
        fp = np.maximum((np.exp(log_fpop) - (1 - _p) * np.exp(log_fb)) / _p,
                        np.exp(log_fpop) * 1e-10)
        lr_1d = np.log(fp) - log_fb
    elif pu_mode:
        w_p      = weights[p_idx]
        log_fp   = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        log_fpop = logsumexp([np.log(weights[g_idx][c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        _p = prior if prior is not None else 0.1
        fb = np.maximum((np.exp(log_fpop) - _p * np.exp(log_fp)) / (1 - _p),
                        np.exp(log_fpop) * 1e-10)
        lr_1d = log_fp - np.log(fb)
    else:
        if p_idx is None or p_idx >= n_w:
            lr_1d = np.zeros(x_2d.shape[0])
        else:
            w_p = weights[p_idx]
            if s_valid and benign_method == 'synonymous':
                w_b = weights[s_idx]
            elif s_valid and b_idx is not None and b_idx < n_w and benign_method == 'avg':
                w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
            elif b_idx is not None and b_idx < n_w:
                w_b = weights[b_idx]
            elif s_valid:
                w_b = weights[s_idx]
            else:
                lr_1d = np.zeros(x_2d.shape[0])
                return lr_1d, sample_logs, comp_logs_per_sample
            log_fp = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
            log_fb = logsumexp([np.log(w_b[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
            lr_1d = log_fp - log_fb

    return lr_1d, sample_logs, comp_logs_per_sample


def _compute_conservative_lr_grid(analysis, config, all_fits, x1g, x2g,
                                   dim_i=0, dim_j=1, total_dims=None,
                                   p_idx_override=None):
    """
    Compute conservative discrete point grid from bootstrap LR+ percentiles.

    For each grid cell:
      - Compute LR+ across all bootstraps
      - Use path_percentile (e.g. 5th) for positive LR+ → pathogenic evidence
      - Use ben_percentile (e.g. 95th) for negative LR+ → benign evidence
      - Convert to discrete points using pre-computed thresholds

    dim_i, dim_j  : which dimensions to plot on x and y axes
    total_dims    : full model dimensionality (inferred from first fit if None)
    p_idx_override: if given, use this effective index as pathogenic instead of analysis.p_idx

    Returns
    -------
    grid_points : (n1, n2) int array — discrete evidence points
    lr_conservative : (n1, n2) — the conservative LR+ used for assignment
    """
    r = analysis.results[config]
    tau_p_log = r['tau_p_log']
    tau_b_log = r['tau_b_log']
    path_pctile = r.get('path_percentile', 5)
    ben_pctile = r.get('ben_percentile', 95)

    if total_dims is None:
        total_dims = analysis.ms.scores.shape[1]

    X1, X2 = np.meshgrid(x1g, x2g, indexing='ij')
    grid_pts = np.full((X1.size, total_dims), np.nan)
    grid_pts[:, dim_i] = X1.ravel()
    grid_pts[:, dim_j] = X2.ravel()
    n_grid = len(grid_pts)
    grid_shape = (len(x1g), len(x2g))

    p_idx = p_idx_override if p_idx_override is not None else analysis.p_idx
    b_idx = analysis.b_idx
    s_idx = getattr(analysis, 's_idx', None)
    g_idx = getattr(analysis, 'g_idx', None)
    prior = r.get('median_prior', None)
    benign_method = analysis.benign_method

    lr_all = Parallel(n_jobs=-1)(
        delayed(_eval_fit_on_grid)(fit, grid_pts, p_idx, b_idx, s_idx, benign_method,
                                   g_idx=g_idx, prior=prior)
        for fit in all_fits
    )
    lr_arr = np.array(lr_all)

    lr_p5 = np.nanpercentile(lr_arr, path_pctile, axis=0)
    lr_p95 = np.nanpercentile(lr_arr, ben_pctile, axis=0)

    grid_points_flat = np.zeros(n_grid, dtype=int)
    for pv in analysis.point_values:
        grid_points_flat[lr_p5 >= tau_p_log[pv - 1]] = pv
    for pv in analysis.point_values:
        grid_points_flat[lr_p95 <= tau_b_log[pv - 1]] = -pv

    lr_conservative = np.where(lr_p5 > 0, lr_p5, np.where(lr_p95 < 0, lr_p95, 0.0))

    return grid_points_flat.reshape(grid_shape), lr_conservative.reshape(grid_shape)


def _compute_all_sample_density_grids(all_fits, S, x1g, x2g,
                                       dim_i=0, dim_j=1, total_dims=None,
                                       max_fits=20):
    """Compute mean density grids for all S samples in a single pass over fits.

    Uses at most max_fits bootstrap fits (subsampled evenly) — density contours
    are visually indistinguishable with 20 vs 100 bootstraps and it's S× faster
    than calling _compute_sample_density_grid separately for each sample.

    Returns list of length S, each a dict {'mean': array, 'std': array} or None.
    """
    if not all_fits:
        return [None] * S
    if total_dims is None:
        total_dims = all_fits[0]['component_params'][0][0].shape[0]

    X1, X2 = np.meshgrid(x1g, x2g, indexing='ij')
    grid_pts = np.full((X1.size, total_dims), np.nan)
    grid_pts[:, dim_i] = X1.ravel()
    grid_pts[:, dim_j] = X2.ravel()
    grid_shape = (len(x1g), len(x2g))
    n_grid_pts = X1.size

    # Subsample fits evenly
    step = max(1, len(all_fits) // max_fits)
    fits = all_fits[::step]

    # Accumulate log-densities per sample: S lists of arrays
    log_densities = [[] for _ in range(S)]

    for fit in fits:
        params  = fit['component_params']
        weights = fit['weights']
        K = len(params)
        n_w = len(weights)
        # Component log-densities computed once, shared across all samples
        comp_log = [_sn_logpdf(grid_pts, *params[c]) for c in range(K)]
        for s in range(min(S, n_w)):
            w_s = weights[s]
            log_d = logsumexp([np.log(w_s[c] + 1e-300) + comp_log[c]
                               for c in range(K)], axis=0)
            log_densities[s].append(log_d)

    results = []
    for s in range(S):
        ld = log_densities[s]
        if not ld:
            results.append(None)
            continue
        arr = np.array(ld)
        mean_log = logsumexp(arr, axis=0) - np.log(arr.shape[0])
        linear = np.exp(arr)
        results.append({
            'mean': np.exp(mean_log).reshape(grid_shape),
            'std':  np.std(linear, axis=0).reshape(grid_shape),
        })
    return results


def _compute_sample_density_grid(all_fits, s_idx, x1g, x2g, dim_i=0, dim_j=1, total_dims=None):
    """Compute mean density for sample s_idx on a 2D grid."""
    if total_dims is None and all_fits:
        total_dims = all_fits[0]['component_params'][0][0].shape[0]
    if total_dims is None:
        total_dims = 2
    X1, X2 = np.meshgrid(x1g, x2g, indexing='ij')
    grid_pts = np.full((X1.size, total_dims), np.nan)
    grid_pts[:, dim_i] = X1.ravel()
    grid_pts[:, dim_j] = X2.ravel()
    grid_shape = (len(x1g), len(x2g))

    densities = []
    for fit in all_fits:
        params = fit['component_params']
        weights = fit['weights']
        if s_idx >= len(weights):
            continue
        K = len(params)
        w_s = weights[s_idx]
        log_d = logsumexp([
            np.log(w_s[c] + 1e-300) + _sn_logpdf(grid_pts, *params[c])
            for c in range(K)
        ], axis=0)
        densities.append(log_d)

    if not densities:
        return None
    arr = np.array(densities)
    mean_log = logsumexp(arr, axis=0) - np.log(arr.shape[0])
    linear = np.exp(arr)
    return {
        'mean': np.exp(mean_log).reshape(grid_shape),
        'std': np.std(linear, axis=0).reshape(grid_shape),
    }


def _compute_bootstrap_marginal_lr(analysis, config, dim, x_grid):
    """
    Compute per-bootstrap marginal LR+ on a 1D grid, then return
    percentiles (5th, 50th, 95th).

    Returns
    -------
    lr_percentiles : dict with 'p5', 'p50', 'p95' arrays of shape (n,)
    sample_marginals : dict of {s_idx: {'mean': (n,), 'std': (n,)}}
    component_marginals : dict of {s_idx: list of {'mean': (n,)}}
    n_used : int
    """
    n = len(x_grid)
    D = analysis.ms.scores.shape[1]
    x_2d = np.full((n, D), np.nan)
    x_2d[:, dim] = x_grid

    S = analysis.ms.sample_assignments.shape[1]
    path_pctile = analysis.results[config].get('path_percentile', 5)
    ben_pctile = analysis.results[config].get('ben_percentile', 95)

    p_idx = analysis.p_idx
    b_idx = analysis.b_idx
    s_idx = getattr(analysis, 's_idx', None)
    benign_method = analysis.benign_method

    # Pre-collect valid reconstituted fits (fast — no heavy computation)
    valid_fits = []
    for boot_key, boot_data in analysis.raw_boots.items():
        fit_raw = boot_data.get(config)
        if fit_raw is None:
            continue
        inner = fit_raw.get('fit', fit_raw)
        fit = analysis._reconstitute_params(inner)
        if fit is not None:
            valid_fits.append(fit)

    n_used = len(valid_fits)
    if n_used == 0:
        return None, None, None, 0

    # Parallel evaluation across bootstraps
    results = Parallel(n_jobs=-1)(
        delayed(_eval_marginal_fit)(fit, x_2d, p_idx, b_idx, s_idx, benign_method, S)
        for fit in valid_fits
    )

    # Aggregate results (some fits may have fewer weight rows than S)
    lr_list = [r[0] for r in results]
    sample_logs = {s: [r[1][s] for r in results if r[1][s] is not None] for s in range(S)}
    component_logs = {s: {} for s in range(S)}
    for r in results:
        comp_logs_per_sample = r[2]
        for s in range(S):
            if comp_logs_per_sample[s] is None:
                continue
            for c, clog in enumerate(comp_logs_per_sample[s]):
                component_logs[s].setdefault(c, []).append(clog)

    lr_arr = np.array(lr_list)
    lr_percentiles = {
        'p5': np.nanpercentile(lr_arr, path_pctile, axis=0),
        'p50': np.nanpercentile(lr_arr, 50, axis=0),
        'p95': np.nanpercentile(lr_arr, ben_pctile, axis=0),
    }

    sample_marginals = {}
    for s in range(S):
        logs = sample_logs[s]
        if not logs:
            sample_marginals[s] = None
            continue
        arr = np.array(logs)
        mean_log = logsumexp(arr, axis=0) - np.log(arr.shape[0])
        linear = np.exp(arr)
        sample_marginals[s] = {
            'mean': np.exp(mean_log),
            'std': np.std(linear, axis=0),
        }

    component_marginals = {}
    for s in range(S):
        if not component_logs[s]:
            component_marginals[s] = None
            continue
        component_marginals[s] = []
        K = len(component_logs[s])
        for c in range(K):
            arr = np.array(component_logs[s][c])
            mean_log = logsumexp(arr, axis=0) - np.log(arr.shape[0])
            component_marginals[s].append({'mean': np.exp(mean_log)})

    return lr_percentiles, sample_marginals, component_marginals, n_used


def _compute_all_marginals(analysis, config, x_grids, p_idx_override=None):
    """Compute marginal LR+ and densities for all dimensions in one parallel sweep.

    x_grids      : list of D 1-D grids (one per dimension).
    p_idx_override: if given, use this effective index as pathogenic instead of analysis.p_idx.
    Returns      : list of D marginal_data dicts (same structure as _compute_bootstrap_marginal_lr).
    """
    D = len(x_grids)
    total_dims = analysis.ms.scores.shape[1]
    S = analysis.ms.sample_assignments.shape[1]

    x_2d_list = []
    for dim, x_grid in enumerate(x_grids):
        x_2d = np.full((len(x_grid), total_dims), np.nan)
        x_2d[:, dim] = x_grid
        x_2d_list.append(x_2d)

    path_pctile = analysis.results[config].get('path_percentile', 5)
    ben_pctile  = analysis.results[config].get('ben_percentile', 95)
    p_idx = p_idx_override if p_idx_override is not None else analysis.p_idx
    b_idx = analysis.b_idx
    s_idx = getattr(analysis, 's_idx', None)
    g_idx = getattr(analysis, 'g_idx', None)
    prior = analysis.results[config].get('median_prior', None)
    benign_method = analysis.benign_method

    valid_fits = []
    for boot_data in analysis.raw_boots.values():
        fit_raw = boot_data.get(config)
        if fit_raw is None:
            continue
        inner = fit_raw.get('fit', fit_raw)
        fit = analysis._reconstitute_params(inner)
        if fit is not None:
            valid_fits.append(fit)

    n_used = len(valid_fits)
    if n_used == 0:
        return [None] * D

    # One parallel call: each worker processes all D marginals for one bootstrap
    all_results = Parallel(n_jobs=-1)(
        delayed(_eval_all_marginals)(fit, x_2d_list, p_idx, b_idx, s_idx, benign_method, S,
                                     g_idx=g_idx, prior=prior)
        for fit in valid_fits
    )
    # all_results[b][d] = (lr_1d, sample_logs, comp_logs_per_sample)

    marginal_data = []
    for dim in range(D):
        dim_results = [all_results[b][dim] for b in range(n_used)]

        lr_list = [r[0] for r in dim_results]
        sample_logs   = {s: [r[1][s] for r in dim_results if r[1][s] is not None] for s in range(S)}
        component_logs = {s: {} for s in range(S)}
        for r in dim_results:
            for s in range(S):
                if r[2][s] is None:
                    continue
                for c, clog in enumerate(r[2][s]):
                    component_logs[s].setdefault(c, []).append(clog)

        lr_arr = np.array(lr_list)
        lr_percentiles = {
            'p5':  np.nanpercentile(lr_arr, path_pctile, axis=0),
            'p50': np.nanpercentile(lr_arr, 50, axis=0),
            'p95': np.nanpercentile(lr_arr, ben_pctile, axis=0),
        }

        sample_marginals = {}
        for s in range(S):
            logs = sample_logs[s]
            if not logs:
                sample_marginals[s] = None
                continue
            arr = np.array(logs)
            mean_log = logsumexp(arr, axis=0) - np.log(arr.shape[0])
            sample_marginals[s] = {
                'mean': np.exp(mean_log),
                'std':  np.std(np.exp(arr), axis=0),
            }

        component_marginals = {}
        for s in range(S):
            if not component_logs[s]:
                component_marginals[s] = None
                continue
            component_marginals[s] = []
            for c in range(len(component_logs[s])):
                arr = np.array(component_logs[s][c])
                mean_log = logsumexp(arr, axis=0) - np.log(arr.shape[0])
                component_marginals[s].append({'mean': np.exp(mean_log)})

        marginal_data.append({
            'x': x_grids[dim], 'lr': lr_percentiles,
            'sample': sample_marginals, 'components': component_marginals,
            'n': n_used,
        })

    return marginal_data


def _aggregate_marginal_dim(dim_results, x_grid, path_pctile, ben_pctile, S):
    """Aggregate per-bootstrap marginal results for one dim into a marginal_data dict."""
    lr_list = [r[0] for r in dim_results]
    sample_logs    = {s: [r[1][s] for r in dim_results if r[1][s] is not None] for s in range(S)}
    component_logs = {s: {} for s in range(S)}
    for r in dim_results:
        for s in range(S):
            if r[2][s] is None:
                continue
            for c, clog in enumerate(r[2][s]):
                component_logs[s].setdefault(c, []).append(clog)

    lr_arr = np.array(lr_list)
    lr_percentiles = {
        'p5':  np.nanpercentile(lr_arr, path_pctile, axis=0),
        'p50': np.nanpercentile(lr_arr, 50,          axis=0),
        'p95': np.nanpercentile(lr_arr, ben_pctile,  axis=0),
    }
    sample_marginals = {}
    for s in range(S):
        logs = sample_logs[s]
        if not logs:
            sample_marginals[s] = None
            continue
        arr = np.array(logs)
        mean_log = logsumexp(arr, axis=0) - np.log(arr.shape[0])
        sample_marginals[s] = {'mean': np.exp(mean_log), 'std': np.std(np.exp(arr), axis=0)}

    component_marginals = {}
    for s in range(S):
        if not component_logs[s]:
            component_marginals[s] = None
            continue
        component_marginals[s] = []
        for c in range(len(component_logs[s])):
            arr = np.array(component_logs[s][c])
            mean_log = logsumexp(arr, axis=0) - np.log(arr.shape[0])
            component_marginals[s].append({'mean': np.exp(mean_log)})

    return {'x': x_grid, 'lr': lr_percentiles,
            'sample': sample_marginals, 'components': component_marginals,
            'n': len(dim_results)}


def _compute_all_marginals_batch_p(all_fits, x_grids, p_indices,
                                    b_idx, s_idx, benign_method, S,
                                    path_pctile, ben_pctile,
                                    g_idx=None, prior=None, n_jobs=-1):
    """Compute marginals for all p_indices in one parallel sweep.

    Component densities are computed ONCE per (bootstrap, dimension) and shared
    across all p_indices.  Replaces N sequential calls to _compute_all_marginals.

    Returns list of len(p_indices), each a list of D marginal_data dicts.
    """
    D = len(x_grids)
    total_dims = len(x_grids)   # will be overridden below from first fit
    if all_fits:
        total_dims = all_fits[0]['component_params'][0][0].shape[0]

    x_2d_list = []
    for dim, x_grid in enumerate(x_grids):
        x_2d = np.full((len(x_grid), total_dims), np.nan)
        x_2d[:, dim] = x_grid
        x_2d_list.append(x_2d)

    n_used = len(all_fits)
    if n_used == 0:
        return [[None] * D for _ in p_indices]

    all_results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_all_marginals_batch_p)(
            fit, x_2d_list, p_indices, b_idx, s_idx, benign_method, S,
            g_idx=g_idx, prior=prior)
        for fit in all_fits
    )
    # all_results[boot][pi][dim] = (lr_1d, sample_logs, comp_logs)

    return [
        [
            _aggregate_marginal_dim(
                [all_results[b][pi][dim] for b in range(n_used)],
                x_grids[dim], path_pctile, ben_pctile, S,
            )
            for dim in range(D)
        ]
        for pi in range(len(p_indices))
    ]


def _compute_lr_grids_for_all_p(all_fits, x1g, x2g, total_dims,
                                  b_idx, s_idx, benign_method, p_configs,
                                  g_idx=None, prior=None):
    """Compute LR+ grids for multiple pathogenic indices in one parallel sweep.

    p_configs : list of dicts, each with:
        p_idx, tau_p_log, tau_b_log, path_pctile, ben_pctile, point_values

    Returns list of (grid_points, lr_conservative) tuples, one per p_config.
    """
    X1, X2 = np.meshgrid(x1g, x2g, indexing='ij')
    grid_pts = np.full((X1.size, total_dims), np.nan)
    grid_pts[:, 0] = X1.ravel()
    grid_pts[:, 1] = X2.ravel()
    n_grid    = len(grid_pts)
    grid_shape = (len(x1g), len(x2g))
    p_indices  = [pc['p_idx'] for pc in p_configs]

    lr_all = Parallel(n_jobs=-1)(
        delayed(_eval_fit_on_grid_batch_p)(fit, grid_pts, p_indices, b_idx, s_idx, benign_method,
                                           g_idx=g_idx, prior=prior)
        for fit in all_fits
    )
    # lr_all[boot][pi] = flat lr array

    results = []
    for pi, pc in enumerate(p_configs):
        lr_arr = np.array([lr_all[b][pi] for b in range(len(all_fits))])
        lr_p5  = np.nanpercentile(lr_arr, pc['path_pctile'], axis=0)
        lr_p95 = np.nanpercentile(lr_arr, pc['ben_pctile'],  axis=0)
        gp = np.zeros(n_grid, dtype=int)
        for pv in pc['point_values']:
            gp[lr_p5  >= pc['tau_p_log'][pv - 1]] = pv
        for pv in pc['point_values']:
            gp[lr_p95 <= pc['tau_b_log'][pv - 1]] = -pv
        lr_con = np.where(lr_p5 > 0, lr_p5, np.where(lr_p95 < 0, lr_p95, 0.0))
        results.append((gp.reshape(grid_shape), lr_con.reshape(grid_shape)))
    return results


# ──────────────────────────────────────
# Main plot function
# ──────────────────────────────────────

def build_heatmap_data(analysis, config):
    """Minimal precomputed dict for plot_variant_evidence_heatmap — no grid computation.

    Per-dimension LR cells will show '—'; Total cells and aux_total='lr' work fully
    since those values are read directly from analysis.results[config].
    """
    return {
        'analysis':                    analysis,
        'config':                      config,
        'marginal_data':               {},
        'aux_marginal_data':           {},
        'aux_vs_primary_marginal_data': {},
    }

def precompute_mv_plot_data(analysis, config, n_grid=120, pad=0.5, n_jobs=-1,
                            projection='umap', pivot_dim='activity_No_treatment'):
    """Precompute all expensive data needed for plot_mv_calibration.

    Returns a dict that can be passed directly to render_mv_plot_data.
    Separating precomputation from rendering means you can tweak plot
    aesthetics (figsize, contour_levels, first_row_only, etc.) without
    rerunning the parallel bootstrap sweeps.
    """
    r = analysis.results.get(config)
    if r is None:
        raise ValueError(f"Config {config} has no results. Run analysis first.")

    ms = analysis.ms
    scores = ms.scores
    sa = ms.sample_assignments
    N, D = scores.shape
    S = sa.shape[1]
    dataset_names = getattr(ms, 'dataset_names', [f'Dim {d}' for d in range(D)])

    r_cfg = analysis.results[config]
    path_pctile = r_cfg.get('path_percentile', 5)
    ben_pctile  = 100 - path_pctile

    x_grids = [
        np.linspace(np.nanmin(scores[:, d]) - pad, np.nanmax(scores[:, d]) + pad, 500)
        for d in range(D)
    ]
    # Separate coarser grids for 2D pairwise display (n_grid × n_grid, not 500 × 500)
    x_grids_plot = [
        np.linspace(np.nanmin(scores[:, d]) - pad, np.nanmax(scores[:, d]) + pad, n_grid)
        for d in range(D)
    ]

    print(f"  Collecting bootstrap fits...")
    all_fits = _collect_valid_fits(analysis, config)
    n_boots_used = len(all_fits)

    aux_p_entries = getattr(analysis, 'aux_p_entries', [])

    # Resolve aux-specific percentile (stored in aux_results when aux_path_percentile was set)
    _aux_res_cfg = r_cfg.get('aux_results', {})
    _first_aux_r = next(iter(_aux_res_cfg.values()), {}) if _aux_res_cfg else {}
    aux_path_pctile = int(_first_aux_r.get('path_percentile', path_pctile))
    aux_ben_pctile  = int(_first_aux_r.get('ben_percentile', ben_pctile))

    print(f"  Computing primary marginals (1 pathogenic × {D} dims, {n_boots_used} boots)...")
    primary_marginals = _compute_all_marginals_batch_p(
        all_fits, x_grids, [analysis.p_idx],
        analysis.b_idx, getattr(analysis, 's_idx', None),
        analysis.benign_method,
        ms.sample_assignments.shape[1],
        path_pctile, ben_pctile,
        g_idx=getattr(analysis, 'g_idx', None),
        prior=r.get('median_prior', None),
        n_jobs=n_jobs,
    )
    marginal_data = {d: primary_marginals[0][d] for d in range(D)}

    aux_marginal_data = {}
    if aux_p_entries:
        aux_eff_indices = [eff for _, eff in aux_p_entries]
        print(f"  Computing aux marginals ({len(aux_eff_indices)} pathogenic × {D} dims, "
              f"p{aux_path_pctile}/p{aux_ben_pctile})...")
        aux_marginals = _compute_all_marginals_batch_p(
            all_fits, x_grids, aux_eff_indices,
            analysis.b_idx, getattr(analysis, 's_idx', None),
            analysis.benign_method,
            ms.sample_assignments.shape[1],
            aux_path_pctile, aux_ben_pctile,
            g_idx=getattr(analysis, 'g_idx', None),
            prior=r.get('median_prior', None),
            n_jobs=n_jobs,
        )
        for ai, (fixed_idx, _) in enumerate(aux_p_entries):
            aux_marginal_data[fixed_idx] = {d: aux_marginals[ai][d] for d in range(D)}

    # Aux-vs-primary marginals: numerator=aux_eff_idx, denominator=primary_p (treated as benign)
    aux_vs_primary_marginal_data = {}
    if aux_p_entries:
        aux_eff_indices_vsp = [eff for _, eff in aux_p_entries]
        print(f"  Computing aux-vs-primary marginals ({len(aux_eff_indices_vsp)} "
              f"aux × {D} dims, p{aux_path_pctile}/p{aux_ben_pctile})...")
        avp_all = _compute_all_marginals_batch_p(
            all_fits, x_grids, aux_eff_indices_vsp,
            b_idx=analysis.p_idx, s_idx=None,
            benign_method=analysis.benign_method,
            S=ms.sample_assignments.shape[1],
            path_pctile=aux_path_pctile, ben_pctile=aux_ben_pctile,
            g_idx=None, prior=None, n_jobs=n_jobs,
        )
        for ai, (fixed_idx, _) in enumerate(aux_p_entries):
            aux_vs_primary_marginal_data[fixed_idx] = {d: avp_all[ai][d] for d in range(D)}

    # ── 2D point-region grid (D=2 only) ──────────────────────────────────────
    lr_grids_2d = None
    if D == 2:
        x1g = np.linspace(np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad, n_grid)
        x2g = np.linspace(np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad, n_grid)
        r_cfg = analysis.results[config]
        _p_configs = [{
            'p_idx':        analysis.p_idx,
            'tau_p_log':    r_cfg['tau_p_log'],
            'tau_b_log':    r_cfg['tau_b_log'],
            'path_pctile':  r_cfg.get('path_percentile', 5),
            'ben_pctile':   r_cfg.get('ben_percentile', 95),
            'point_values': analysis.point_values,
        }]
        for _, eff_idx in aux_p_entries:
            _ar = r_cfg.get('aux_results', {})
            _ar_entry = next((v for k, v in _ar.items() if v.get('eff_idx') == eff_idx), {})
            _p_configs.append({
                'p_idx':        eff_idx,
                'tau_p_log':    _ar_entry.get('tau_p_log', r_cfg['tau_p_log']),
                'tau_b_log':    _ar_entry.get('tau_b_log', r_cfg['tau_b_log']),
                'path_pctile':  r_cfg.get('path_percentile', 5),
                'ben_pctile':   r_cfg.get('ben_percentile', 95),
                'point_values': analysis.point_values,
            })
        print(f"  Computing LR+ grid ({n_grid}×{n_grid}) for {len(_p_configs)} "
              f"pathogenic index(es)...")
        lr_grids_2d = _compute_lr_grids_for_all_p(
            all_fits, x1g, x2g, scores.shape[1],
            analysis.b_idx, getattr(analysis, 's_idx', None),
            analysis.benign_method, _p_configs,
            g_idx=getattr(analysis, 'g_idx', None),
            prior=r.get('median_prior', None),
        )

    # ── Aux-vs-primary 2D grids ───────────────────────────────────────────────
    aux_vs_primary_grids_2d = None
    if D == 2 and aux_p_entries:
        _avp_configs = [{
            'p_idx':        eff_idx,
            'tau_p_log':    r_cfg['tau_p_log'],
            'tau_b_log':    r_cfg['tau_b_log'],
            'path_pctile':  r_cfg.get('path_percentile', 5),
            'ben_pctile':   r_cfg.get('ben_percentile', 95),
            'point_values': analysis.point_values,
        } for _, eff_idx in aux_p_entries]
        print(f"  Computing aux-vs-primary LR+ grids ({n_grid}×{n_grid})...")
        aux_vs_primary_grids_2d = _compute_lr_grids_for_all_p(
            all_fits, x1g, x2g, scores.shape[1],
            b_idx=analysis.p_idx, s_idx=None,
            benign_method=analysis.benign_method,
            p_configs=_avp_configs,
            g_idx=None, prior=None,
        )

    # ── Per-sample density grids (D=2 row 1 contours) ───────────────────────
    density_grids_2d = None
    if D == 2:
        x1g = np.linspace(np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad, n_grid)
        x2g = np.linspace(np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad, n_grid)
        print(f"  Computing sample density grids ({S} samples, ≤20 bootstrap fits)...")
        density_grids_2d = _compute_all_sample_density_grids(all_fits, S, x1g, x2g)

    # ── Pairwise LR+ grids ───────────────────────────────────────────────────
    pairwise_grids_hd = None
    pairwise_dim_pairs = None
    _do_pairwise = (D > 2 and projection == 'pairwise') or projection in ('activity_pairs', 'pairwise_dim0')
    if _do_pairwise:
        from itertools import combinations as _combinations
        if projection == 'pairwise':
            pairwise_dim_pairs = list(_combinations(range(D), 2))
        elif projection == 'pairwise_dim0':
            pairwise_dim_pairs = [(0, d) for d in range(1, D)]
        else:  # activity_pairs
            act_dims = [d for d, n in enumerate(dataset_names) if pivot_dim.lower() in n.lower()]
            assert len(act_dims) == 1, \
                f"Expected exactly one '{pivot_dim}' dimension, found {len(act_dims)}: " \
                f"{[dataset_names[d] for d in act_dims]}"
            act_d = act_dims[0]
            pairwise_dim_pairs = [(act_d, d) for d in range(D) if d != act_d]
        pairwise_grids_hd = []
        for dim_i, dim_j in pairwise_dim_pairs:
            print(f"  Computing pairwise LR+ grid: {dataset_names[dim_i]} vs {dataset_names[dim_j]}...")
            pairwise_grids_hd.append(_compute_conservative_lr_grid(
                analysis, config, all_fits,
                x_grids_plot[dim_i], x_grids_plot[dim_j],
                dim_i=dim_i, dim_j=dim_j, total_dims=D,
            ))

    # ── Aux pairwise grids (aux-vs-benign, one per pair per aux) ────────────
    aux_pairwise_grids_hd = None   # {fixed_idx: [grid per pair]}
    if _do_pairwise and aux_p_entries and pairwise_dim_pairs:
        aux_pairwise_grids_hd = {}
        for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
            _sn_tmp = getattr(ms, 'sample_names', None) or []
            aux_name = _sn_tmp[fixed_idx] if fixed_idx < len(_sn_tmp) else f'Sample {fixed_idx}'
            aux_pairwise_grids_hd[fixed_idx] = []
            _ar = r_cfg.get('aux_results', {})
            _ar_entry = next((v for k, v in _ar.items() if v.get('eff_idx') == eff_idx), {})
            _pc = {
                'p_idx':        eff_idx,
                'tau_p_log':    _ar_entry.get('tau_p_log', r_cfg['tau_p_log']),
                'tau_b_log':    _ar_entry.get('tau_b_log', r_cfg['tau_b_log']),
                'path_pctile':  r_cfg.get('path_percentile', 5),
                'ben_pctile':   r_cfg.get('ben_percentile', 95),
                'point_values': analysis.point_values,
            }
            for dim_i, dim_j in pairwise_dim_pairs:
                print(f"  Computing aux pairwise grid ({aux_name}): "
                      f"{dataset_names[dim_i]} vs {dataset_names[dim_j]}...")
                grids = _compute_lr_grids_for_all_p(
                    all_fits, x_grids_plot[dim_i], x_grids_plot[dim_j], D,
                    b_idx=analysis.b_idx, s_idx=getattr(analysis, 's_idx', None),
                    benign_method=analysis.benign_method,
                    p_configs=[_pc],
                    g_idx=getattr(analysis, 'g_idx', None),
                    prior=r.get('median_prior', None),
                )
                aux_pairwise_grids_hd[fixed_idx].append(grids[0])

    # ── UMAP embedding for D>2 ───────────────────────────────────────────────
    umap_data = None
    if D > 2 and projection == 'umap':
        try:
            import sys, types
            if 'importlib.metadata' not in sys.modules:
                _meta = types.ModuleType('importlib.metadata')
                _meta.version = lambda _pkg: '0.0.0'
                class _PNF(Exception): pass
                _meta.PackageNotFoundError = _PNF
                sys.modules['importlib.metadata'] = _meta
            import umap as umap_lib

            from src.assay_calibration.data_utils.dataset import BasicMultiScoreset

            if isinstance(ms, BasicMultiScoreset):
                # BasicMultiScoreset already exposes the full score matrix as
                # _scores — no dataframe or union-find key matching needed.
                full_scores = ms._scores.copy()   # (N, D)
                N_full = N
                sample_indices = np.arange(N)
                valid_sample = np.ones(N, dtype=bool)
            else:
                # Build full score matrix from all variants in each scoreset
                # (including VUS/unclassified, not just the four calibration samples)
                all_scores_by_key = {}   # hgvs_key -> {dim: score}
                for d, scoreset in enumerate(ms.scoresets):
                    df_s = scoreset.dataframe
                    sc = getattr(scoreset, 'score_col', 'auth_reported_score')
                    for _, row in df_s.iterrows():
                        # Use the same key strategy as MultiScoreset._get_variant_keys
                        strategy = getattr(scoreset, '_code_strategy', 'genomic')
                        if strategy == 'hgvs_p':
                            key = str(row.get('hgvs_p', ''))
                        elif strategy == 'hgvs_c':
                            key = str(row.get('hgvs_c', ''))
                        else:
                            key = str(tuple(row.get(c, '') for c in ms.group_cols))
                        if key and key != 'nan':
                            if key not in all_scores_by_key:
                                all_scores_by_key[key] = {}
                            val = pd.to_numeric(row.get(sc), errors='coerce')
                            if not np.isnan(val):
                                all_scores_by_key[key][d] = float(val)

                all_keys = list(all_scores_by_key.keys())
                N_full = len(all_keys)
                full_scores = np.full((N_full, D), np.nan)
                for i, key in enumerate(all_keys):
                    for d, v in all_scores_by_key[key].items():
                        full_scores[i, d] = v

                # Map ms._variants_kept back to indices in all_keys.
                # ms._variants_kept contains canonical union-find atoms — try all
                # plausible string representations to find a match.
                key_to_idx = {k: i for i, k in enumerate(all_keys)}
                sample_indices = []
                for vk in ms._variants_kept:
                    idx = -1
                    # vk is a tuple; try element 0 first (hgvs_p / hgvs_c strategy)
                    candidates = [vk[0] if len(vk) >= 1 else None,
                                  str(vk),
                                  str(vk[0]) if len(vk) >= 1 else None]
                    for cand in candidates:
                        if cand is not None and cand in key_to_idx:
                            idx = key_to_idx[cand]
                            break
                    sample_indices.append(idx)
                sample_indices = np.array(sample_indices)
                valid_sample = sample_indices >= 0
                n_missing = (~valid_sample).sum()
                if n_missing > 0:
                    print(f"  Warning: {n_missing}/{N} sample variants not matched "
                          f"in full scoreset — they will be absent from UMAP.")

            print(f"  Computing pairwise distance matrix "
                  f"({N_full} total variants incl. VUS, D={D})...")
            # For each pair of variants, compute Euclidean distance using only
            # dims both have observed, normalized by the number of shared dims.
            # Variants with no shared dims get max distance (handled below).
            observed = ~np.isnan(full_scores)   # (N_full, D)
            # Z-score each dim using only observed entries so dims with
            # different scales contribute equally to pairwise distance.
            full_scores_z = full_scores.copy()
            for d in range(D):
                vals = full_scores[observed[:, d], d]
                if len(vals) > 1:
                    mu, sigma = vals.mean(), vals.std()
                    if sigma > 0:
                        full_scores_z[observed[:, d], d] = (vals - mu) / sigma
            dist = np.zeros((N_full, N_full), dtype=np.float32)
            for d in range(D):
                obs_d = observed[:, d]
                both = obs_d[:, None] & obs_d[None, :]
                diff = np.nan_to_num(full_scores_z[:, d], nan=0.0)
                sq_diff = (diff[:, None] - diff[None, :]) ** 2
                dist += sq_diff * both.astype(np.float32)

            shared_dims = (observed[:, None, :] & observed[None, :, :]).sum(axis=2)
            shared_dims = np.maximum(shared_dims, 1)
            dist = np.sqrt(dist / shared_dims)
            no_shared = (observed[:, None, :] | observed[None, :, :]).sum(axis=2) == 0
            dist[no_shared] = dist[~no_shared].max() if (~no_shared).any() else 0.0
            np.fill_diagonal(dist, 0.0)

            print(f"  Fitting UMAP on precomputed distances...")
            reducer = umap_lib.UMAP(n_components=2, metric='precomputed',
                                    random_state=42, n_jobs=1)
            full_embedding = reducer.fit_transform(dist)   # (N_full, 2)

            # Extract embedding rows for the calibration sample variants
            embedding = np.full((N, 2), np.nan)
            embedding[valid_sample] = full_embedding[sample_indices[valid_sample]]
            imputed_mask = np.isnan(scores).any(axis=1)

            umap_data = {
                'embedding':    embedding,      # (N, 2) — sample variants only
                'imputed_mask': imputed_mask,   # (N,) — partial obs among samples
            }
            print(f"  UMAP done.")
        except Exception as _umap_err:
            print(f"  UMAP skipped: {type(_umap_err).__name__}: {_umap_err}")

    return {
        'analysis':                     analysis,
        'config':                       config,
        'all_fits':                     all_fits,
        'marginal_data':                marginal_data,
        'aux_marginal_data':            aux_marginal_data,
        'aux_vs_primary_marginal_data': aux_vs_primary_marginal_data,
        'x_grids':                      x_grids,
        'x_grids_plot':                 x_grids_plot,
        'lr_grids_2d':                  lr_grids_2d,
        'aux_vs_primary_grids_2d':      aux_vs_primary_grids_2d,
        'density_grids_2d':             density_grids_2d,
        'pairwise_grids_hd':            pairwise_grids_hd,
        'aux_pairwise_grids_hd':        aux_pairwise_grids_hd,
        'pairwise_dim_pairs':           pairwise_dim_pairs,
        'umap_data':                    umap_data,
        'n_boots_used':                 n_boots_used,
        'n_grid':                       n_grid,
        'pad':                          pad,
    }


def render_mv_plot_data(precomputed, figsize=None, contour_levels=6,
                        first_row_only=False, max_lr_pairs=10,
                        sample_names=None, projection='umap'):
    """Render a plot from precomputed data returned by precompute_mv_plot_data.

    All expensive bootstrap computation is already done; this only runs the
    matplotlib drawing code so you can iterate on aesthetics cheaply.
    """
    analysis      = precomputed['analysis']
    config        = precomputed['config']
    all_fits      = precomputed['all_fits']
    marginal_data = precomputed['marginal_data']
    aux_marginal_data = precomputed['aux_marginal_data']
    aux_vs_primary_marginal_data = precomputed.get('aux_vs_primary_marginal_data', {})
    x_grids       = precomputed['x_grids']
    n_boots_used  = precomputed['n_boots_used']
    n_grid        = precomputed['n_grid']
    pad           = precomputed['pad']

    r = analysis.results[config]
    ms = analysis.ms
    scores = ms.scores
    sa = ms.sample_assignments
    N, D = scores.shape
    S = sa.shape[1]
    dataset_names = getattr(ms, 'dataset_names', [f'Dim {d}' for d in range(D)])

    # Map effective sample indices to fixed roles (0=P/LP, 1=B/LB, 2=gnomAD, 3=Syn)
    _eff_to_fixed = {}
    for fixed_idx, eff_idx in [(0, analysis.p_idx), (1, analysis.b_idx),
                                (2, analysis.g_idx), (3, analysis.s_idx)]:
        if eff_idx is not None:
            _eff_to_fixed[eff_idx] = fixed_idx

    if sample_names is None:
        _sn_raw = getattr(ms, 'sample_names', None) or SAMPLE_NAMES_DEFAULT
        sample_names = [_sn_raw[i] if i < len(_sn_raw) else f'Sample {i}' for i in range(S)]
    else:
        # custom names indexed by fixed role; map to effective indices via _eff_to_fixed
        sample_names = [sample_names[_eff_to_fixed.get(i, i)] if _eff_to_fixed.get(i, i) < len(sample_names)
                        else f'Sample {i}' for i in range(S)]

    latent_q    = r.get('latent_q', getattr(analysis, '_latent_q', 1))
    model_label = f"CFUSN q={latent_q}" if latent_q > 1 else "MSN q=1"
    points       = r['points']
    tau_p_log    = r['tau_p_log']
    tau_b_log    = r['tau_b_log']
    median_prior = r['median_prior']
    C_path       = r.get('C_path', r.get('C', '?'))
    C_ben        = r.get('C_ben', '?')
    path_pctile  = r.get('path_percentile', 5)
    ben_pctile   = r.get('ben_percentile', 95)
    max_pt       = max(analysis.point_values)
    pt_norm      = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)
    ylim_bound   = max(abs(tau_p_log[-1]), abs(tau_b_log[-1]))
    missing_frac = 1.0 - (~np.isnan(scores).any(axis=1)).mean()

    if first_row_only:
        marginal_data_render           = {d: None for d in range(D)}
        aux_marginal_data_render       = {fid: {d: None for d in range(D)}
                                          for fid, _ in getattr(analysis, 'aux_p_entries', [])}
        aux_vs_primary_marginal_render = {fid: {d: None for d in range(D)}
                                          for fid, _ in getattr(analysis, 'aux_p_entries', [])}
    else:
        marginal_data_render           = marginal_data
        aux_marginal_data_render       = aux_marginal_data
        aux_vs_primary_marginal_render = aux_vs_primary_marginal_data

    sample_style = {
        'colors':  [SAMPLE_COLORS[_eff_to_fixed.get(i, i) % len(SAMPLE_COLORS)]
                    for i in range(S)],
        'edges':   [_SAMPLE_EDGE_COLORS[_eff_to_fixed.get(i, i) % len(_SAMPLE_EDGE_COLORS)]
                    for i in range(S)],
        'markers': [SAMPLE_MARKERS[_eff_to_fixed.get(i, i) % len(SAMPLE_MARKERS)]
                    for i in range(S)],
    }

    aux_p_entries = getattr(analysis, 'aux_p_entries', [])
    gene = getattr(ms, 'scoreset_name', '')
    suptitle = (
        f'{gene} — {config}\n'
        f'{n_boots_used} bootstraps, prior={median_prior:.4f}, '
        f'missing={missing_frac*100:.1f}%'
    )

    if D == 2 and projection != 'activity_pairs':
        fig, info = _plot_mv_2d(
            analysis, config, all_fits, marginal_data_render, x_grids,
            scores, sa, N, D, S, dataset_names, sample_names,
            points, tau_p_log, tau_b_log, median_prior, C_path, C_ben,
            path_pctile, ben_pctile, max_pt, pt_norm, ylim_bound, missing_frac,
            model_label, n_boots_used, pad, n_grid, contour_levels,
            figsize, suptitle,
            aux_p_entries=aux_p_entries, aux_marginal_data=aux_marginal_data_render,
            aux_vs_primary_marginal_data=aux_vs_primary_marginal_render,
            first_row_only=first_row_only, sample_style=sample_style,
            lr_grids_2d=precomputed.get('lr_grids_2d'),
            aux_vs_primary_grids_2d=precomputed.get('aux_vs_primary_grids_2d'),
            density_grids_2d=precomputed.get('density_grids_2d'),
        )
    else:  # D>2, or D==2 with activity_pairs
        umap_data = precomputed.get('umap_data')
        use_umap = (projection == 'umap' and umap_data is not None)
        if use_umap:
            fig, info = _plot_mv_umap(
                analysis, config, scores, sa, N, D, S,
                dataset_names, sample_names, points, max_pt, pt_norm,
                model_label, n_boots_used, missing_frac,
                umap_data, figsize, suptitle, sample_style,
                first_row_only=first_row_only,
            )
        else:
            fig, info = _plot_mv_hd(
                analysis, config, all_fits, marginal_data_render,
                precomputed.get('x_grids_plot', x_grids),
                scores, sa, N, D, S, dataset_names, sample_names,
                points, tau_p_log, tau_b_log, median_prior, C_path, C_ben,
                path_pctile, ben_pctile, max_pt, pt_norm, ylim_bound,
                model_label, n_boots_used, pad, n_grid, contour_levels,
                figsize, suptitle,
                aux_p_entries=aux_p_entries, aux_marginal_data=aux_marginal_data_render,
                aux_vs_primary_marginal_data=aux_vs_primary_marginal_render,
                sample_style=sample_style,
                max_lr_pairs=(None if projection == 'pairwise_dim0' else max_lr_pairs),
                pairwise_grids_hd=precomputed.get('pairwise_grids_hd'),
                aux_pairwise_grids_hd=precomputed.get('aux_pairwise_grids_hd'),
                pairwise_dim_pairs=precomputed.get('pairwise_dim_pairs'),
                first_row_only=first_row_only,
            )
    return fig, info


def plot_mv_calibration(analysis, config, figsize=None, n_grid=120,
                        contour_levels=6, first_row_only=False, max_lr_pairs=10):
    """Convenience wrapper: precompute + render in one call."""
    precomputed = precompute_mv_plot_data(analysis, config, n_grid=n_grid)
    return render_mv_plot_data(precomputed, figsize=figsize,
                               contour_levels=contour_levels,
                               first_row_only=first_row_only,
                               max_lr_pairs=max_lr_pairs)


def plot_variant_evidence_heatmap(precomputed, sample_idx=0,
                                  variant_ids=None, figsize=None,
                                  cell_fontsize=6, row_height=0.38,
                                  show_aux='auto',
                                  aux_total='points'):
    """Heatmap of per-dimension marginal evidence for variants in one or more samples.

    Rows   = variants, grouped by sample with a separator header row between groups.
    Cols   = assay dimensions + 'Total' + optional aux-vs-benign and aux-vs-primary cols.
    Color  = marginal evidence points for that dimension (POINT_CMAP scale).
    Text   = path-pct LR on top line, ben-pct LR on bottom line.
             Missing dimensions shown as gray with '—'.

    Parameters
    ----------
    precomputed  : dict from precompute_mv_plot_data
    sample_idx   : int or list of int — sample column(s) to display
    variant_ids  : list of str, optional — explicit variant list (ignores sample_idx)
    figsize      : (w, h) or None (auto)
    cell_fontsize: font size inside cells
    row_height   : inches per data row (separator rows count as 0.5×)
    show_aux     : 'auto'        — show aux columns if aux data available
                   'benign'      — only aux-vs-benign columns
                   'pathogenic'  — only aux-vs-primary-pathogenic columns
                   'both'        — both aux comparison column groups
                   'none'        — only primary evidence columns
    aux_total    : 'points'      — Total cell shows pre-computed Tavtigian joint points (default)
                   'lr'          — Total cell shows joint log LR+ (p5/median/p95 across bootstraps)
    """
    analysis      = precomputed['analysis']
    config        = precomputed['config']
    marginal_data = precomputed['marginal_data']
    aux_marginal_data          = precomputed.get('aux_marginal_data', {})
    aux_vs_primary_marginal_data = precomputed.get('aux_vs_primary_marginal_data', {})

    ms     = analysis.ms
    scores = ms.scores
    _sa_raw = ms._sample_assignments   # (N, n_raw_cols) — fixed column positions
    N, D    = scores.shape
    S_raw   = _sa_raw.shape[1]

    r           = analysis.results[config]
    joint_pts   = r['points']
    tau_p_log   = r['tau_p_log']
    tau_b_log   = r['tau_b_log']
    path_pctile = r.get('path_percentile', 5)
    ben_pctile  = r.get('ben_percentile', 95)
    max_pt      = max(analysis.point_values)
    pt_norm     = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)

    dataset_names    = getattr(ms, 'dataset_names', [f'Dim {d}' for d in range(D)])
    _sn_raw          = getattr(ms, 'sample_names', None) or SAMPLE_NAMES_DEFAULT
    sample_names_all = [_sn_raw[i] if i < len(_sn_raw) else f'Sample {i}'
                        for i in range(S_raw)]

    variants_kept = list(ms._variants_kept)

    # ── Resolve rows ─────────────────────────────────────────────────────────
    # Each entry in `render_rows` is either:
    #   ('variant', vi, label)          — a data row
    #   ('separator', None, group_name) — a group header row
    render_rows = []

    if variant_ids is not None:
        missing = [v for v in variant_ids if v not in variants_kept]
        if missing:
            print(f"Warning: {len(missing)} variant(s) not found: {missing[:5]}")
        for v in variant_ids:
            if v in variants_kept:
                render_rows.append(('variant', variants_kept.index(v), v))
    else:
        indices = sample_idx if isinstance(sample_idx, (list, tuple)) else [sample_idx]
        for si in indices:
            if si >= S_raw:
                print(f"sample_idx={si} out of range (raw cols: {S_raw}), skipping")
                continue
            mask = _sa_raw[:, si].astype(bool)
            if not mask.any():
                print(f"Sample {si} ({sample_names_all[si]}) is empty, skipping")
                continue
            sname = sample_names_all[si] if si < len(sample_names_all) else f'Sample {si}'
            if len(indices) > 1:
                render_rows.append(('separator', None, sname))
            for vi in np.where(mask)[0].tolist():
                render_rows.append(('variant', vi, variants_kept[vi]))

    if not any(kind == 'variant' for kind, _, _ in render_rows):
        print("No variants to display.")
        return None

    # ── Resolve aux comparison column groups ─────────────────────────────────
    aux_p_entries = getattr(analysis, 'aux_p_entries', [])
    r_cfg = analysis.results[config]
    _aux_res = r_cfg.get('aux_results', {})
    _sn_raw2 = getattr(ms, 'sample_names', None) or SAMPLE_NAMES_DEFAULT

    # Determine which aux column groups to show
    has_aux = bool(aux_p_entries)
    _show_b  = show_aux in ('auto', 'benign', 'both')     and has_aux
    _show_p  = show_aux in ('auto', 'pathogenic', 'both') and has_aux
    if show_aux == 'none':
        _show_b = _show_p = False

    # aux_col_groups: 8-tuple per group
    #   (label, fixed_idx, data_dict, tau_p, tau_b, col_color, group_type, n_cols)
    # group_type: 'vs_b' (D+1 wide), 'vs_p' (1 wide, Total only), 'pb' (1 wide, Total only)
    aux_col_groups = []
    for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
        aux_name = (_sn_raw2[fixed_idx] if fixed_idx < len(_sn_raw2) else f'Sample {fixed_idx}')
        aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
        aux_r = _aux_res.get(fixed_idx, {})
        p_name = (_sn_raw2[analysis.p_idx] if analysis.p_idx is not None
                  and analysis.p_idx < len(_sn_raw2) else 'P/LP')
        if _show_b:
            aux_col_groups.append((
                f'{aux_name}\nvs B', fixed_idx,
                aux_marginal_data.get(fixed_idx, {}),
                aux_r.get('tau_p_log', tau_p_log),
                aux_r.get('tau_b_log', tau_b_log),
                aux_color, 'vs_b', D + 1,
            ))
        if _show_p:
            aux_col_groups.append((
                f'{aux_name}\nvs {p_name}', fixed_idx,
                {}, tau_p_log, tau_b_log,
                aux_color, 'vs_p', 1,
            ))

    n_aux_col_groups = len(aux_col_groups)
    n_cols_primary = D + 1
    # col_start for each aux group computed explicitly (groups have variable width)
    col_starts = []
    cur = n_cols_primary
    for *_, n_cols_grp in aux_col_groups:
        col_starts.append(cur)
        cur += n_cols_grp
    n_cols_total = cur

    # ── Compute per-cell values (variant rows only) ───────────────────────────
    variant_rows = [(i, vi, lbl) for i, (kind, vi, lbl) in enumerate(render_rows)
                    if kind == 'variant']
    n_variant_rows = len(variant_rows)
    n_sep_rows     = sum(1 for kind, _, _ in render_rows if kind == 'separator')

    n_total = len(render_rows)
    color_grid   = np.zeros((n_total, n_cols_total))
    text_grid    = [[''] * n_cols_total for _ in range(n_total)]
    missing_grid = np.zeros((n_total, n_cols_total), dtype=bool)
    col_header_color = ['black'] * n_cols_total  # per-column header color

    def _fill_single_lr_cell(row_pos, col, vi, lr_5th, lr_median, lr_95th, tp, tb, col_c):
        """Fill one Total-only cell (for vs_p and pb 1-wide groups)."""
        col_header_color[col] = col_c
        if lr_5th is not None and vi < len(lr_5th) and not np.isnan(lr_5th[vi]):
            lrp = float(lr_5th[vi])
            lrm = float(lr_median[vi]) if lr_median is not None else float('nan')
            lrb = float(lr_95th[vi])   if lr_95th  is not None else float('nan')
            mpt = 0
            for pv in range(max_pt, 0, -1):
                if pv - 1 < len(tp) and lrp >= tp[pv - 1]: mpt = pv; break
                if pv - 1 < len(tb) and lrb <= tb[pv - 1]: mpt = -pv; break
            color_grid[row_pos, col] = mpt
            sp = '+' if lrp >= 0 else ''
            sm = '+' if lrm >= 0 else ''
            sb = '+' if lrb >= 0 else ''
            text_grid[row_pos][col] = f'{sp}{lrp:.2f}\n{sm}{lrm:.2f}\n{sb}{lrb:.2f}'
        else:
            missing_grid[row_pos, col] = True
            text_grid[row_pos][col] = '—'

    def _fill_lr_cols(row_pos, vi, col_start, dim_mdata, tp, tb, col_c,
                      aux_pts_arr, aux_lr_5th=None, aux_lr_median=None, aux_lr_95th=None):
        """Fill D + 1 cells starting at col_start for one aux comparison group.

        The last cell (col_start + D) is a Total column.  Its content depends on
        aux_total: 'points' shows the pre-computed Tavtigian joint points;
        'lr' shows the joint log LR+ (p5/median/p95 across bootstraps).
        """
        col_header_color[col_start:col_start + D + 1] = [col_c] * (D + 1)
        for d in range(D):
            col = col_start + d
            sc = scores[vi, d]
            md = dim_mdata.get(d) if dim_mdata else None
            if np.isnan(sc) or md is None or md.get('lr') is None:
                missing_grid[row_pos, col] = True
                text_grid[row_pos][col] = '—'
                continue
            lrp = float(np.interp(sc, md['x'], md['lr']['p5']))
            lrm = float(np.interp(sc, md['x'], md['lr']['p50']))
            lrb = float(np.interp(sc, md['x'], md['lr']['p95']))
            mpt = 0
            for pv in range(max_pt, 0, -1):
                if pv - 1 < len(tp) and lrp >= tp[pv - 1]:
                    mpt = pv; break
                if pv - 1 < len(tb) and lrb <= tb[pv - 1]:
                    mpt = -pv; break
            color_grid[row_pos, col] = mpt
            sp = '+' if lrp >= 0 else ''
            sm = '+' if lrm >= 0 else ''
            sb = '+' if lrb >= 0 else ''
            text_grid[row_pos][col] = f'{sp}{lrp:.2f}\n{sm}{lrm:.2f}\n{sb}{lrb:.2f}'
        # Total column
        total_col = col_start + D
        if aux_total == 'lr' and aux_lr_5th is not None and vi < len(aux_lr_5th):
            lrp = float(aux_lr_5th[vi])
            lrm = float(aux_lr_median[vi]) if aux_lr_median is not None else float('nan')
            lrb = float(aux_lr_95th[vi])   if aux_lr_95th  is not None else float('nan')
            mpt = 0
            for pv in range(max_pt, 0, -1):
                if pv - 1 < len(tp) and lrp >= tp[pv - 1]:
                    mpt = pv; break
                if pv - 1 < len(tb) and lrb <= tb[pv - 1]:
                    mpt = -pv; break
            color_grid[row_pos, total_col] = mpt
            sp = '+' if lrp >= 0 else ''
            sm = '+' if lrm >= 0 else ''
            sb = '+' if lrb >= 0 else ''
            text_grid[row_pos][total_col] = f'{sp}{lrp:.2f}\n{sm}{lrm:.2f}\n{sb}{lrb:.2f}'
        elif aux_pts_arr is not None and vi < len(aux_pts_arr):
            jp = int(aux_pts_arr[vi])
            color_grid[row_pos, total_col] = jp
            text_grid[row_pos][total_col] = f'{jp:+d}'
        else:
            missing_grid[row_pos, total_col] = True
            text_grid[row_pos][total_col] = '—'

    is_sep = [kind == 'separator' for kind, _, _ in render_rows]

    for row_pos, vi, _ in variant_rows:
        # Primary evidence columns
        for d in range(D):
            sc = scores[vi, d]
            md = marginal_data.get(d)
            if np.isnan(sc) or md is None or md.get('lr') is None:
                missing_grid[row_pos, d] = True
                text_grid[row_pos][d]    = '—'
                continue
            x   = md['x']
            lrp = float(np.interp(sc, x, md['lr']['p5']))
            lrb = float(np.interp(sc, x, md['lr']['p95']))
            mpt = 0
            for pv in range(max_pt, 0, -1):
                if pv - 1 < len(tau_p_log) and lrp >= tau_p_log[pv - 1]:
                    mpt = pv; break
                if pv - 1 < len(tau_b_log) and lrb <= tau_b_log[pv - 1]:
                    mpt = -pv; break
            color_grid[row_pos, d] = mpt
            sign_p = '+' if lrp >= 0 else ''
            sign_b = '+' if lrb >= 0 else ''
            text_grid[row_pos][d]  = f'{sign_p}{lrp:.2f}\n{sign_b}{lrb:.2f}'
        jp = int(joint_pts[vi])
        color_grid[row_pos, D] = jp
        text_grid[row_pos][D]  = f'{jp:+d}'

        # Aux comparison column groups
        for gi, (glabel, fixed_idx, dim_mdata, tp, tb, gc, group_type, _) in enumerate(aux_col_groups):
            col_start = col_starts[gi]
            _ar = _aux_res.get(fixed_idx, {})
            if group_type == 'vs_b':
                aux_pts_arr     = _ar.get('points')   if aux_total == 'points' else None
                aux_lr_5th_arr  = _ar.get('lr_5th')   if aux_total == 'lr' else None
                aux_lr_med_arr  = _ar.get('lr_median') if aux_total == 'lr' else None
                aux_lr_95th_arr = _ar.get('lr_95th')  if aux_total == 'lr' else None
                _fill_lr_cols(row_pos, vi, col_start, dim_mdata, tp, tb, gc,
                              aux_pts_arr, aux_lr_5th_arr, aux_lr_med_arr, aux_lr_95th_arr)
            elif group_type == 'vs_p':
                _vsp = (_ar.get('vsp_results') or {})
                _fill_single_lr_cell(row_pos, col_start, vi,
                                     _vsp.get('lr_5th'), _vsp.get('lr_median'), _vsp.get('lr_95th'),
                                     tp, tb, gc)

    # ── DataFrame (variant rows only, no separators) ──────────────────────────
    table_records = []
    for row_pos, vi, lbl in variant_rows:
        rec = {'variant': lbl}
        for d in range(D):
            dn = dataset_names[d]
            if missing_grid[row_pos, d]:
                rec[f'{dn}__score'] = rec[f'{dn}__lr_path'] = rec[f'{dn}__lr_ben'] = rec[f'{dn}__pts'] = np.nan
            else:
                sc  = scores[vi, d]
                md  = marginal_data.get(d)
                lrp = float(np.interp(sc, md['x'], md['lr']['p5']))
                lrb = float(np.interp(sc, md['x'], md['lr']['p95']))
                rec[f'{dn}__score']   = sc
                rec[f'{dn}__lr_path'] = lrp
                rec[f'{dn}__lr_ben']  = lrb
                rec[f'{dn}__pts']     = int(color_grid[row_pos, d])
        rec['joint_pts'] = int(color_grid[row_pos, D])
        for gi, (glabel, fixed_idx, dim_mdata, tp, tb, gc, group_type, _) in enumerate(aux_col_groups):
            col_start = col_starts[gi]
            if group_type == 'vs_b':
                for d in range(D):
                    dn = dataset_names[d]
                    col = col_start + d
                    if missing_grid[row_pos, col]:
                        rec[f'{glabel}__{dn}__lr_path'] = rec[f'{glabel}__{dn}__lr_ben'] = np.nan
                    else:
                        md2 = (dim_mdata or {}).get(d)
                        if md2 and md2.get('lr') is not None:
                            sc = scores[vi, d]
                            rec[f'{glabel}__{dn}__lr_path'] = float(np.interp(sc, md2['x'], md2['lr']['p5']))
                            rec[f'{glabel}__{dn}__lr_ben']  = float(np.interp(sc, md2['x'], md2['lr']['p95']))
                        else:
                            rec[f'{glabel}__{dn}__lr_path'] = rec[f'{glabel}__{dn}__lr_ben'] = np.nan
            total_col = col_start + (D if group_type == 'vs_b' else 0)
            rec[f'{glabel}__total'] = (
                int(color_grid[row_pos, total_col]) if not missing_grid[row_pos, total_col] else np.nan
            )
        table_records.append(rec)
    table_df = pd.DataFrame(table_records).set_index('variant')

    # ── Draw ─────────────────────────────────────────────────────────────────
    # Column labels
    col_labels = list(dataset_names) + ['Total\n(joint)']
    for glabel, _, _, _, _, _, group_type, _ in aux_col_groups:
        if group_type == 'vs_b':
            col_labels += [f'{glabel}\n{dn}' for dn in dataset_names] + [f'{glabel}\nTotal']
        else:
            col_labels += [glabel]

    n_cols = n_cols_total
    col_w  = max(1.4, 7.0 / min(n_cols, D + 1))
    total_height = n_variant_rows * row_height + n_sep_rows * row_height * 0.5
    if figsize is None:
        figsize = (col_w * n_cols + 2.5, total_height + 1.5)

    # Map render_rows to y-coordinates (separator rows are half-height)
    y_coords = []
    y = 0.0
    row_heights_list = []
    for kind, _, _ in render_rows:
        h = row_height * 0.5 if kind == 'separator' else row_height
        y_coords.append(y)
        row_heights_list.append(h)
        y += h
    total_y = y

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, total_y)
    ax.invert_yaxis()
    ax.set_aspect('auto')
    ax.axis('off')

    # Column headers (above y=0)
    _total_cols = {D} | {
        col_starts[gi] + (D if gt == 'vs_b' else 0)
        for gi, (*_, gt, _) in enumerate(aux_col_groups)
    }
    for col, lbl in enumerate(col_labels):
        is_total = col in _total_cols
        is_aux   = (col >= n_cols_primary)
        color    = col_header_color[col] if is_aux else 'black'
        weight   = 'bold' if is_total else 'normal'
        ax.text(col + 0.5, -row_height * 0.5, lbl, ha='center', va='center',
                fontsize=cell_fontsize + 1, fontweight=weight, rotation=45,
                rotation_mode='anchor', color=color)
    ax.text(-0.05, -row_height * 0.5, 'Variant', ha='right', va='center',
            fontsize=cell_fontsize + 1, fontweight='bold')

    # Draw vertical separator lines between column groups
    _sep_cols = [n_cols_primary] + list(col_starts[1:])
    for sc_x in _sep_cols:
        ax.axvline(sc_x, color='#888888', lw=1.2, alpha=0.6, zorder=5)

    for row_pos, (kind, vi, lbl) in enumerate(render_rows):
        y0 = y_coords[row_pos]
        rh = row_heights_list[row_pos]

        if kind == 'separator':
            rect = plt.Rectangle((0, y0), n_cols, rh,
                                  facecolor='#404040', edgecolor='none')
            ax.add_patch(rect)
            ax.text(n_cols / 2, y0 + rh / 2, lbl,
                    ha='center', va='center',
                    fontsize=cell_fontsize + 1, fontweight='bold', color='white')
            continue

        ax.text(-0.05, y0 + rh / 2, lbl, ha='right', va='center',
                fontsize=cell_fontsize - 1, color='black')

        for col in range(n_cols):
            if missing_grid[row_pos, col]:
                face = '#d0d0d0'; tc = '#666666'
            else:
                face = POINT_CMAP(pt_norm(color_grid[row_pos, col]))
                lum  = 0.299 * face[0] + 0.587 * face[1] + 0.114 * face[2]
                tc   = 'white' if lum < 0.5 else 'black'
            rect = plt.Rectangle((col, y0), 1, rh,
                                  facecolor=face, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(col + 0.5, y0 + rh / 2, text_grid[row_pos][col],
                    ha='center', va='center',
                    fontsize=cell_fontsize, color=tc, linespacing=1.3)

    if variant_ids is not None:
        title_label = 'selected variants'
    elif isinstance(sample_idx, (list, tuple)):
        title_label = ' + '.join(
            sample_names_all[si] if si < len(sample_names_all) else f'Sample {si}'
            for si in sample_idx
        )
    else:
        title_label = (sample_names_all[sample_idx]
                       if sample_idx < len(sample_names_all) else f'Sample {sample_idx}')
    aux_note = ''
    if aux_col_groups:
        aux_note = f'  |  aux cols: {", ".join(g[0].replace(chr(10), " ") for g in aux_col_groups)}'
    ax.set_title(
        f'{title_label}  |  color = evidence points  |  '
        f'text = p{path_pctile} / p{ben_pctile} LR\n'
        f'Total = full joint evidence{aux_note}',
        fontsize=cell_fontsize + 2, pad=20
    )
    plt.tight_layout()
    return fig, table_df


def plot_rpv_quadrant(precomputed, fixed_idx=None,
                      pctile='median',
                      samples=None,
                      label_variants=None,
                      tau_lines=True,
                      show_classes=False,
                      pad=0.5,
                      xlim=None,
                      ylim=None,
                      sample_names=None,
                      gvsr=None,
                      gvsr_log=True,
                      figsize=None, ax=None):
    """Scatter of (aux vs B LR, aux vs P LR) for variants in selected samples.

    Quadrant interpretation
    -----------------------
    top-right    (aux_vs_B > 0, aux_vs_P ≈ 0 / positive)  → true RPV, reduced penetrance
    bottom-right (aux_vs_B > 0, aux_vs_P << 0)            → PLP-like, high penetrance
    left         (aux_vs_B ≤ 0)                           → benign-like, no RPV signal

    Parameters
    ----------
    precomputed    : dict from precompute_mv_plot_data (or build_heatmap_data)
    fixed_idx      : which aux entry to plot; defaults to the first
    pctile         : 'median' | 'conservative' | 'liberal'
    samples        : list of raw sample indices to include; None = all
    label_variants : list of variant IDs to annotate
    tau_lines      : draw Tavtigian threshold lines on both axes
    show_classes   : shade classification regions (low_pen_rpv / plp_like / benign_like)
                     using the first Tavtigian threshold on each axis as the boundary
    pad            : fractional padding around the data range for axis limits
    gvsr           : pd.Series or dict mapping variant ID → GVSr value.
                     When provided, only aux-sample (fixed_idx) variants with a
                     GVSr value are plotted; color encodes GVSr magnitude.
                     Other samples are drawn in gray as reference.
    gvsr_log       : apply log10 scale to GVSr colormap (default True)
    """
    analysis = precomputed['analysis']
    config   = precomputed['config']
    r        = analysis.results[config]
    ms       = analysis.ms
    _sa_raw  = ms._sample_assignments
    N        = ms.scores.shape[0]

    variants_kept = list(ms._variants_kept)
    aux_p_entries = getattr(analysis, 'aux_p_entries', [])
    if not aux_p_entries:
        print("No aux entries configured.")
        return None

    if fixed_idx is None:
        fixed_idx = aux_p_entries[0][0]

    aux_res = r.get('aux_results', {}).get(fixed_idx, {})
    vsp_res = aux_res.get('vsp_results') or {}

    if pctile == 'median':
        x_arr = aux_res.get('lr_median')
        y_arr = vsp_res.get('lr_median')
    elif pctile == 'conservative':
        x_arr = aux_res.get('lr_5th')
        y_arr = vsp_res.get('lr_95th')
    else:
        x_arr = aux_res.get('lr_95th')
        y_arr = vsp_res.get('lr_5th')

    if x_arr is None or y_arr is None:
        print("LR arrays missing — rerun run() to populate vsp_results.")
        return None

    x_arr = np.asarray(x_arr, float)
    y_arr = np.asarray(y_arr, float)

    tau_p     = r['tau_p_log']
    tau_b     = r['tau_b_log']
    aux_tau_p = aux_res.get('tau_p_log', tau_p)
    aux_tau_b = aux_res.get('tau_b_log', tau_b)

    _sn_raw  = sample_names or getattr(ms, 'sample_names', None) or SAMPLE_NAMES_DEFAULT
    S_raw    = _sa_raw.shape[1]
    sample_names_all = [_sn_raw[i] if i < len(_sn_raw) else f'Sample {i}'
                        for i in range(S_raw)]

    _sc_fixed = [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(4)]

    if samples is None:
        samples = list(range(S_raw))
    # aux sample always drawn last (on top)
    draw_order = [si for si in samples if si != fixed_idx]
    if fixed_idx in samples:
        draw_order.append(fixed_idx)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (7, 6))
    else:
        fig = ax.get_figure()

    # ── Collect points that will be plotted (for axis limits) ─────────────
    plotted_x, plotted_y = [], []
    for si in draw_order:
        if si >= S_raw:
            continue
        mask = _sa_raw[:, si].astype(bool)
        xi = x_arr[mask]; yi = y_arr[mask]
        valid = np.isfinite(xi) & np.isfinite(yi)
        plotted_x.append(xi[valid]); plotted_y.append(yi[valid])

    if not any(len(v) for v in plotted_x):
        print("No valid (finite) LR values for selected samples.")
        return None

    all_x = np.concatenate(plotted_x); all_y = np.concatenate(plotted_y)
    if xlim is None:
        xspan = all_x.max() - all_x.min() or 1.0
        xlim  = (all_x.min() - pad * xspan, all_x.max() + pad * xspan)
    if ylim is None:
        yspan = all_y.max() - all_y.min() or 1.0
        ylim  = (all_y.min() - pad * yspan, all_y.max() + pad * yspan)

    # ── Scatter ────────────────────────────────────────────────────────────
    gvsr_mode = gvsr is not None
    if gvsr_mode:
        # normalise to dict {variant_id: value}
        if hasattr(gvsr, 'to_dict'):
            gvsr_dict = gvsr.to_dict()
        else:
            gvsr_dict = dict(gvsr)

    for si, xi, yi in zip(draw_order, plotted_x, plotted_y):
        if gvsr_mode:
            if si == fixed_idx:
                # aux sample: color by GVSr for variants that have it, skip others
                mask = _sa_raw[:, si].astype(bool)
                vi_indices = np.where(mask)[0]
                gvsr_vals, gvsr_xi, gvsr_yi = [], [], []
                for vi, xv, yv in zip(vi_indices, xi, yi):
                    vid = variants_kept[vi]
                    if vid in gvsr_dict and np.isfinite(gvsr_dict[vid]):
                        gvsr_vals.append(gvsr_dict[vid])
                        gvsr_xi.append(xv); gvsr_yi.append(yv)
                if gvsr_vals:
                    import matplotlib.colors as mcolors
                    vals = np.array(gvsr_vals, dtype=float)
                    cvals = np.log10(np.clip(vals, 1e-6, None)) if gvsr_log else vals
                    norm  = mcolors.Normalize(vmin=cvals.min(), vmax=cvals.max())
                    cmap  = plt.cm.plasma
                    sc = ax.scatter(gvsr_xi, gvsr_yi, s=50, c=cvals, cmap=cmap,
                                    norm=norm, alpha=0.85, zorder=4,
                                    edgecolors='white', linewidths=0.4,
                                    label=sample_names_all[si])
                    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
                    clbl = f'log₁₀(GVSr)' if gvsr_log else 'GVSr'
                    cbar.set_label(clbl, fontsize=8)
                    cbar.ax.tick_params(labelsize=7)
            else:
                # other samples: gray reference
                ax.scatter(xi, yi, s=20, color='#cccccc', alpha=0.35, zorder=1,
                           edgecolors='none', label=sample_names_all[si])
        else:
            color = _AUX_COLORS[(si - 4) % len(_AUX_COLORS)] if si >= 4 else _sc_fixed[si % 4]
            ax.scatter(xi, yi, s=30, color=color, alpha=0.65, zorder=2,
                       edgecolors='white', linewidths=0.4, label=sample_names_all[si])

    # ── Axis limits (tight around plotted data) ────────────────────────────
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    # ── Classification region shading ──────────────────────────────────────
    _class_legend_patches = None
    if show_classes:
        import matplotlib.patches as mpatches
        x_rpv_thresh = float(aux_tau_p[0]) if len(aux_tau_p) else 0.0  # vs_B supporting
        x_ben_thresh = float(aux_tau_b[0]) if len(aux_tau_b) else 0.0  # vs_B benign
        y_rpv_thresh = float(tau_p[0])     if len(tau_p)     else 0.0  # vs_P >= 1 pt (low pen)
        y_plp_thresh = float(tau_b[0])     if len(tau_b)     else 0.0  # vs_P <= -1 pt (plp-like)

        rx0 = x_rpv_thresh
        # low_pen_rpv: x > x_rpv_thresh, y > y_rpv_thresh (vsp_points >= 1)
        ax.add_patch(mpatches.Rectangle(
            (rx0, y_rpv_thresh), xlim[1] - rx0, ylim[1] - y_rpv_thresh,
            color='#3cb371', alpha=0.07, zorder=0, linewidth=0))
        # plp_like: x > x_rpv_thresh, y < y_plp_thresh (vsp_points <= -1)
        ax.add_patch(mpatches.Rectangle(
            (rx0, ylim[0]), xlim[1] - rx0, y_plp_thresh - ylim[0],
            color='#e05c00', alpha=0.07, zorder=0, linewidth=0))
        # benign_like: x < x_ben_thresh
        ax.add_patch(mpatches.Rectangle(
            (xlim[0], ylim[0]), x_ben_thresh - xlim[0], ylim[1] - ylim[0],
            color='#2c7bb6', alpha=0.07, zorder=0, linewidth=0))

        _class_legend_patches = [
            mpatches.Patch(color='#3cb371', alpha=0.5, label='low-pen RPV'),
            mpatches.Patch(color='#e05c00', alpha=0.5, label='PLP-like'),
            mpatches.Patch(color='#2c7bb6', alpha=0.5, label='benign-like'),
        ]

    # ── Quadrant + threshold lines ─────────────────────────────────────────
    ax.axvline(0, color='#555555', lw=1.0, ls='--', alpha=0.6, zorder=3)
    ax.axhline(0, color='#555555', lw=1.0, ls='--', alpha=0.6, zorder=3)

    if tau_lines:
        for pv_i, (tp_v, tb_v) in enumerate(zip(aux_tau_p, aux_tau_b)):
            a = min(0.12 + 0.06 * (pv_i + 1), 0.4)
            ax.axvline(tp_v, color='#d7191c', lw=0.7, ls=':', alpha=a, zorder=3)
            ax.axvline(tb_v, color='#2c7bb6', lw=0.7, ls=':', alpha=a, zorder=3)
        for pv_i, (tp_v, tb_v) in enumerate(zip(tau_p, tau_b)):
            a = min(0.12 + 0.06 * (pv_i + 1), 0.4)
            ax.axhline(tp_v, color='#d7191c', lw=0.7, ls=':', alpha=a, zorder=3)
            ax.axhline(tb_v, color='#2c7bb6', lw=0.7, ls=':', alpha=a, zorder=3)

    # ── Quadrant labels (in axes-fraction coords, unaffected by data range) ─
    def _qlabel(fx, fy, text, ha, va):
        ax.text(fx, fy, text, ha=ha, va=va, fontsize=7, color='#333333',
                fontstyle='italic', alpha=0.70, transform=ax.transAxes)

    _qlabel(0.52, 0.97, 'RPV\n(reduced pen.)',  ha='left',  va='top')
    _qlabel(0.52, 0.03, 'PLP-like\n(high pen.)', ha='left', va='bottom')
    _qlabel(0.02, 0.97, 'benign-like',           ha='left',  va='top')

    # ── Labels ────────────────────────────────────────────────────────────
    aux_name = (_sn_raw[fixed_idx] if fixed_idx < len(_sn_raw) else f'Sample {fixed_idx}')
    p_name   = (_sn_raw[analysis.p_idx] if analysis.p_idx is not None
                and analysis.p_idx < len(_sn_raw) else 'P/LP')
    ax.set_xlabel(f'log LR⁺  {aux_name} vs B  ({pctile})', fontsize=9)
    ax.set_ylabel(f'log LR⁺  {aux_name} vs {p_name}  ({pctile})', fontsize=9)
    ax.set_title(f'RPV quadrant — {aux_name}', fontsize=10)

    if label_variants:
        for vid in label_variants:
            if vid not in variants_kept:
                continue
            vi = variants_kept.index(vid)
            xv, yv = float(x_arr[vi]), float(y_arr[vi])
            if np.isfinite(xv) and np.isfinite(yv):
                ax.annotate(vid, (xv, yv), fontsize=6,
                            xytext=(4, 4), textcoords='offset points',
                            color='black', zorder=5)

    sample_leg = ax.legend(fontsize=7, markerscale=1.2, framealpha=0.7,
                           bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    if _class_legend_patches is not None:
        import matplotlib.patches as mpatches
        ax.add_artist(sample_leg)
        ax.legend(handles=_class_legend_patches, fontsize=7, framealpha=0.7,
                  bbox_to_anchor=(1.01, 0), loc='lower left', borderaxespad=0)
    ax.grid(lw=0.3, alpha=0.3)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    return fig, ax


def inspect_variant(variant_id, precomputed, figsize=None):
    """Print per-dimension log LR+ and evidence points for one variant,
    then plot a marginal row per observed dimension.

    Parameters
    ----------
    variant_id : str
        Must match an entry in ms._variants_kept.
    precomputed : dict
        Output of precompute_mv_plot_data.
    """
    analysis      = precomputed['analysis']
    config        = precomputed['config']
    marginal_data = precomputed['marginal_data']
    aux_marginal_data          = precomputed.get('aux_marginal_data', {})
    aux_vs_primary_marginal_data = precomputed.get('aux_vs_primary_marginal_data', {})

    ms     = analysis.ms
    scores = ms.scores           # (N, D)
    sa     = ms.sample_assignments  # (N, S)
    N, D   = scores.shape
    S      = sa.shape[1]

    # ── Find the variant ──────────────────────────────────────────────────────
    variants_kept = list(ms._variants_kept)
    if variant_id not in variants_kept:
        print(f"Variant '{variant_id}' not found in ms._variants_kept")
        print(f"  First 10: {variants_kept[:10]}")
        return None
    idx = variants_kept.index(variant_id)

    r           = analysis.results[config]
    v_scores    = scores[idx]          # (D,)
    v_points    = int(r['points'][idx])
    tau_p_log   = r['tau_p_log']
    tau_b_log   = r['tau_b_log']
    path_pctile = r.get('path_percentile', 5)
    ben_pctile  = r.get('ben_percentile', 95)
    ylim_bound  = max(abs(tau_p_log[-1]), abs(tau_b_log[-1]))
    max_pt      = max(analysis.point_values)

    dataset_names = getattr(ms, 'dataset_names', [f'Dim {d}' for d in range(D)])
    _sn_raw       = getattr(ms, 'sample_names', None) or SAMPLE_NAMES_DEFAULT
    sample_names_all = [_sn_raw[i] if i < len(_sn_raw) else f'Sample {i}' for i in range(S)]

    _eff_to_fixed = {}
    for fixed_idx, eff_idx in [(0, analysis.p_idx), (1, analysis.b_idx),
                                (2, analysis.g_idx), (3, analysis.s_idx)]:
        if eff_idx is not None:
            _eff_to_fixed[eff_idx] = fixed_idx

    # ── Text summary ──────────────────────────────────────────────────────────
    sample_membership = [sample_names_all[i] for i in range(S) if sa[idx, i]]
    print(f"Variant : {variant_id}")
    print(f"Points  : {v_points:+d}")
    print(f"Sample  : {', '.join(sample_membership) if sample_membership else 'none (VUS/unclassified)'}")
    print()
    print(f"{'Dim':<4}  {'Assay':<30}  {'Score':>8}  "
          f"{'logLR+(path)':>13}  {'logLR+(med)':>11}  {'logLR+(ben)':>11}")
    print("-" * 90)

    obs_dims = []
    for d in range(D):
        sc = v_scores[d]
        if np.isnan(sc):
            print(f"{d:<4}  {dataset_names[d]:<30}  {'—':>8}")
            continue
        obs_dims.append(d)
        md = marginal_data.get(d)
        if md is None or md.get('lr') is None:
            print(f"{d:<4}  {dataset_names[d]:<30}  {sc:>8.4f}  (no LR+ data)")
            continue
        x     = md['x']
        lrp5  = float(np.interp(sc, x, md['lr']['p5']))
        lrp50 = float(np.interp(sc, x, md['lr']['p50']))
        lrp95 = float(np.interp(sc, x, md['lr']['p95']))
        print(f"{d:<4}  {dataset_names[d]:<30}  {sc:>8.4f}  "
              f"{lrp5:>+13.3f}  {lrp50:>+11.3f}  {lrp95:>+11.3f}")

    if not obs_dims:
        print("\nNo observed dimensions — nothing to plot.")
        return None

    # ── Build aux LR panel specs ──────────────────────────────────────────────
    aux_p_entries = getattr(analysis, 'aux_p_entries', [])
    r_cfg = analysis.results[config]
    _aux_res = r_cfg.get('aux_results', {})
    _sn_raw2 = getattr(ms, 'sample_names', None) or SAMPLE_NAMES_DEFAULT
    _sn_fixed = [_sn_raw2[i] if i < len(_sn_raw2) else f'Sample {i}' for i in range(S)]

    # Each aux entry: two extra LR panels (vs benign, vs primary pathogenic)
    aux_lr_specs = []   # list of (title, dim_mdata_dict, tau_p, tau_b, color, rug_top, rug_bot)
    for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
        aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
        aux_name = _sn_fixed[fixed_idx] if fixed_idx < len(_sn_fixed) else f'Sample {fixed_idx}'
        aux_r = _aux_res.get(fixed_idx, {})
        p_name = _sn_fixed[analysis.p_idx] if analysis.p_idx is not None and analysis.p_idx < len(_sn_fixed) else 'P/LP'
        aux_lr_specs.append({
            'title':    f'{aux_name} vs B',
            'mdata':    aux_marginal_data.get(fixed_idx, {}),
            'tau_p':    aux_r.get('tau_p_log', tau_p_log),
            'tau_b':    aux_r.get('tau_b_log', tau_b_log),
            'color':    aux_color,
            'rug_top':  eff_idx,
            'rug_bot':  analysis.b_idx,
        })
        aux_lr_specs.append({
            'title':    f'{aux_name} vs {p_name}',
            'mdata':    aux_vs_primary_marginal_data.get(fixed_idx, {}),
            'tau_p':    tau_p_log,
            'tau_b':    tau_b_log,
            'color':    aux_color,
            'rug_top':  eff_idx,
            'rug_bot':  analysis.p_idx,
        })

    # ── Figure ────────────────────────────────────────────────────────────────
    n_hist_cols  = min(S, 4)
    n_extra_lr   = len(aux_lr_specs)
    n_cols       = n_hist_cols + 1 + n_extra_lr  # density + primary LR + extra LR
    n_rows       = len(obs_dims)
    _sc_fixed    = [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(4)]

    if figsize is None:
        figsize = (4.5 * n_cols, 3.2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False,
                             gridspec_kw={'hspace': 0.50, 'wspace': 0.32})

    for row, d in enumerate(obs_dims):
        sc = v_scores[d]
        md = marginal_data.get(d)

        # histogram panels
        for col in range(n_hist_cols):
            s_idx  = col
            ax     = axes[row, col]
            fi     = _eff_to_fixed.get(s_idx, s_idx)
            color  = _sc_fixed[fi % len(_sc_fixed)]

            if s_idx < S:
                obs_s = scores[sa[:, s_idx].astype(bool), d]
                obs_s = obs_s[~np.isnan(obs_s)]
                if len(obs_s) > 0:
                    ax.hist(obs_s, bins=40, density=True, alpha=0.3,
                            color=color, edgecolor='white', linewidth=0.3)

                if md is not None and md.get('sample') and md['sample'].get(s_idx) is not None:
                    s_data = md['sample'][s_idx]
                    ax.plot(md['x'], s_data['mean'], color=color, lw=1.5)
                    ax.fill_between(md['x'],
                                    np.maximum(s_data['mean'] - s_data['std'], 0),
                                    s_data['mean'] + s_data['std'],
                                    color=color, alpha=0.15)
                n_obs_s = int((~np.isnan(scores[sa[:, s_idx].astype(bool), d])).sum()) if s_idx < S else 0
            else:
                n_obs_s = 0

            ax.axvline(sc, color='black', lw=1.5, ls='--', alpha=0.85, zorder=5)
            ax.set_xlabel(dataset_names[d], fontsize=7)
            ax.set_ylabel('Density', fontsize=7)
            ax.set_title(f'{sample_names_all[s_idx]} (n={n_obs_s})',
                         fontsize=7, color=color, fontweight='bold')
            ax.tick_params(labelsize=6)
            ax.grid(lw=0.2, alpha=0.2)

        # Primary LR+ panel
        ax = axes[row, n_hist_cols]
        if md is not None and md.get('lr') is not None:
            x_marg = md['x']
            p5, p50, p95 = md['lr']['p5'], md['lr']['p50'], md['lr']['p95']
            ax.plot(x_marg, p50, color='black', lw=1.5, label='Median')
            ax.plot(x_marg, p5,  color='#d7191c', lw=1.0, label=f'{path_pctile}th')
            ax.plot(x_marg, p95, color='#2c7bb6', lw=1.0, label=f'{ben_pctile}th')
            ax.fill_between(x_marg, p5, p95, color='gray', alpha=0.06)
            for pv in range(1, len(tau_p_log) + 1):
                alpha = min(0.15 + 0.05 * pv, 0.5)
                ax.axhline(tau_p_log[pv - 1], color='red',  ls=':', lw=0.5, alpha=alpha)
                ax.axhline(tau_b_log[pv - 1], color='blue', ls=':', lw=0.5, alpha=alpha)
            for pv in [1, 4, max_pt]:
                if pv - 1 < len(tau_p_log):
                    ax.text(x_marg[0], tau_p_log[pv - 1], f' +{pv}', fontsize=5,
                            color='red',  va='bottom')
                    ax.text(x_marg[0], tau_b_log[pv - 1], f' -{pv}', fontsize=5,
                            color='blue', va='top')
            ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
            lr_med = float(np.interp(sc, x_marg, p50))
            lr_pth = float(np.interp(sc, x_marg, p5))
            ax.axvline(sc, color='black', lw=1.5, ls='--', alpha=0.85, zorder=5)
            ax.plot(sc, lr_med, 'ko', ms=6, zorder=6)
            ax.annotate(f'med={lr_med:+.2f}\npath={lr_pth:+.2f}',
                        xy=(sc, lr_med), xytext=(6, 4),
                        textcoords='offset points', fontsize=6,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
            ax.set_ylim(-ylim_bound, ylim_bound)
        ax.set_xlabel(dataset_names[d], fontsize=7)
        ax.set_ylabel('log LR+', fontsize=7)
        ax.set_title(f'LR+ (P vs B) — {dataset_names[d]}', fontsize=8, fontweight='bold')
        ax.legend(fontsize=5, framealpha=0.6)
        ax.tick_params(labelsize=6)
        ax.grid(lw=0.2, alpha=0.2)

        # Extra aux LR panels
        for ei, spec in enumerate(aux_lr_specs):
            ax = axes[row, n_hist_cols + 1 + ei]
            emd = spec['mdata'].get(d) if spec['mdata'] else None
            ec = spec['color']
            if emd is not None and emd.get('lr') is not None:
                xm = emd['x']
                ep5, ep50, ep95 = emd['lr']['p5'], emd['lr']['p50'], emd['lr']['p95']
                ax.plot(xm, ep50, color=ec,   lw=1.5, label='Median')
                ax.plot(xm, ep5,  color=ec,   lw=1.0, ls='--', label=f'{path_pctile}th')
                ax.plot(xm, ep95, color=ec,   lw=1.0, ls=':',  label=f'{ben_pctile}th')
                ax.fill_between(xm, ep5, ep95, color=ec, alpha=0.08)
                tp, tb = spec['tau_p'], spec['tau_b']
                for pv in range(1, len(tp) + 1):
                    alpha = min(0.15 + 0.05 * pv, 0.5)
                    ax.axhline(tp[pv - 1], color='red',  ls=':', lw=0.5, alpha=alpha)
                    ax.axhline(tb[pv - 1], color='blue', ls=':', lw=0.5, alpha=alpha)
                ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
                lr_med = float(np.interp(sc, xm, ep50))
                lr_pth = float(np.interp(sc, xm, ep5))
                ax.axvline(sc, color='black', lw=1.5, ls='--', alpha=0.85, zorder=5)
                ax.plot(sc, lr_med, 'o', color=ec, ms=6, zorder=6)
                ax.annotate(f'med={lr_med:+.2f}\np{path_pctile}={lr_pth:+.2f}',
                            xy=(sc, lr_med), xytext=(6, 4),
                            textcoords='offset points', fontsize=6,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
                # Rug ticks
                rt, rb = spec.get('rug_top'), spec.get('rug_bot')
                if rt is not None and rt < S:
                    ot = scores[sa[:, rt], d]; ot = ot[~np.isnan(ot)]
                    if len(ot): ax.plot(ot, np.full(len(ot), ylim_bound), '|', color=ec, alpha=0.4, ms=3, mew=0.3)
                if rb is not None and rb < S:
                    ob = scores[sa[:, rb], d]; ob = ob[~np.isnan(ob)]
                    if len(ob): ax.plot(ob, np.full(len(ob), -ylim_bound), '|', color='#888888', alpha=0.3, ms=3, mew=0.3)
                ax.set_ylim(-ylim_bound, ylim_bound)
            ax.set_xlabel(dataset_names[d], fontsize=7)
            ax.set_ylabel('log LR', fontsize=7)
            ax.set_title(f'{spec["title"]} — {dataset_names[d]}', fontsize=7,
                         fontweight='bold', color=ec)
            ax.legend(fontsize=5, framealpha=0.6)
            ax.tick_params(labelsize=6)
            ax.grid(lw=0.2, alpha=0.2)

    fig.suptitle(f'{variant_id}  |  total points = {v_points:+d}',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


_AUX_COLORS = ['#e6a817', '#8B4513', '#006400', '#800080', '#FF6600', '#4B0082']


def _draw_marginal_row(fig, gs, row, dim, md, scores, sa, S, n_cols,
                       dataset_names, sample_names, tau_p_log, tau_b_log,
                       ylim_bound, path_pctile, ben_pctile, analysis,
                       sample_style=None, extra_lr_cols=None):
    """Draw one row: S sample density panels + primary LR+ + optional extra LR+ columns.

    extra_lr_cols : list of dicts, each with keys:
        md        : marginal_data dict for this dim (or None)
        title     : panel title string
        color     : line color for this comparison
        tau_p     : tau_p_log array for threshold lines
        tau_b     : tau_b_log array for threshold lines
        rug_top   : sample index whose scores to show as rug ticks at top (numerator)
        rug_bot   : sample index whose scores to show as rug ticks at bottom (denominator)
    """
    extra_lr_cols = extra_lr_cols or []
    n_extra = len(extra_lr_cols)
    # Primary LR+ column is at index S; extras follow at S+1, S+2, ...
    lr_col = S

    x_marg = md['x']
    _sc = sample_style['colors']  if sample_style else [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(S)]

    for s_idx in range(S):
        ax = fig.add_subplot(gs[row, s_idx])
        color = _sc[s_idx]

        s_data = md['sample'][s_idx] if md['sample'] is not None else None
        if s_data is not None:
            total_mean = s_data['mean']
            total_std  = s_data['std']
            ax.plot(x_marg, total_mean, color=color, lw=1.5, zorder=3)
            ax.fill_between(x_marg,
                            np.maximum(total_mean - total_std, 0),
                            total_mean + total_std, color=color, alpha=0.08)
            c_data = md['components'][s_idx] if md['components'] is not None else None
            if c_data is not None:
                n_comp = len(c_data)
                comp_colors = plt.cm.Set2(np.linspace(0, 1, max(n_comp, 3)))
                for c in range(n_comp):
                    c_mean = c_data[c]['mean']
                    if c_mean.max() < total_mean.max() * 0.005:
                        continue
                    ax.plot(x_marg, c_mean, color=comp_colors[c], lw=0.8, ls='--', alpha=0.7)

        obs = scores[sa[:, s_idx], dim]
        obs = obs[~np.isnan(obs)]
        if len(obs) > 0:
            ax.hist(obs, bins=40, density=True, alpha=0.2, color=color,
                    edgecolor='white', linewidth=0.3)

        n_s_dim = (~np.isnan(scores[sa[:, s_idx], dim])).sum()
        ax.set_xlabel(dataset_names[dim], fontsize=7)
        ax.set_ylabel('Density', fontsize=7)
        ax.set_title(f'{sample_names[s_idx]} — {dataset_names[dim]} (n={n_s_dim})',
                     fontsize=7, fontweight='bold', color=color)
        ax.grid(lw=0.2, alpha=0.2)

    # Primary LR+ panel (column S)
    ax = fig.add_subplot(gs[row, lr_col])
    if md['lr'] is not None:
        p5, p50, p95 = md['lr']['p5'], md['lr']['p50'], md['lr']['p95']
        ax.plot(x_marg, p50, color='black', lw=1.5, label='Median')
        ax.plot(x_marg, p5,  color='#d7191c', lw=1.0, label=f'{path_pctile}th (path)')
        ax.plot(x_marg, p95, color='#2c7bb6', lw=1.0, label=f'{ben_pctile}th (ben)')
        ax.fill_between(x_marg, p5, p95, color='gray', alpha=0.06)
        for pv in range(1, len(tau_p_log) + 1):
            alpha = min(0.15 + 0.05 * pv, 0.5)
            ax.axhline(tau_p_log[pv - 1], color='red',  ls=':', lw=0.4, alpha=alpha)
            ax.axhline(tau_b_log[pv - 1], color='blue', ls=':', lw=0.4, alpha=alpha)
        for pv in [1, 4, max(analysis.point_values)]:
            if pv - 1 < len(tau_p_log):
                ax.text(x_marg[0], tau_p_log[pv - 1], f' +{pv}', fontsize=5, color='red',  va='bottom')
                ax.text(x_marg[0], tau_b_log[pv - 1], f' -{pv}', fontsize=5, color='blue', va='top')
        ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
        if analysis.p_idx is not None:
            obs_p = scores[sa[:, analysis.p_idx], dim]
            obs_p = obs_p[~np.isnan(obs_p)]
            if len(obs_p):
                ax.plot(obs_p, np.full(len(obs_p), ylim_bound),
                        '|', color='#CA7682', alpha=0.3, ms=3, mew=0.3)
        neg_idx = analysis.b_idx if analysis.b_idx is not None else analysis.s_idx
        if neg_idx is not None:
            obs_b = scores[sa[:, neg_idx], dim]
            obs_b = obs_b[~np.isnan(obs_b)]
            if len(obs_b):
                ax.plot(obs_b, np.full(len(obs_b), -ylim_bound),
                        '|', color='#1D7AAB', alpha=0.3, ms=3, mew=0.3)
        ax.set_ylim(-ylim_bound, ylim_bound)
    ax.set_xlabel(dataset_names[dim], fontsize=7)
    ax.set_ylabel('log LR+', fontsize=7)
    ax.set_title(f'Marginal LR+ — {dataset_names[dim]}', fontsize=8, fontweight='bold')
    ax.legend(fontsize=5, framealpha=0.6)
    ax.grid(lw=0.2, alpha=0.2)

    # Extra LR+ columns (aux comparisons)
    for ei, ec in enumerate(extra_lr_cols):
        ax = fig.add_subplot(gs[row, lr_col + 1 + ei])
        emd = ec.get('md')
        if emd is None or emd.get('lr') is None:
            ax.axis('off')
            continue
        ec_col  = ec['color']
        ec_tp   = ec.get('tau_p', tau_p_log)
        ec_tb   = ec.get('tau_b', tau_b_log)
        xm = emd['x']
        ep5, ep50, ep95 = emd['lr']['p5'], emd['lr']['p50'], emd['lr']['p95']
        ax.plot(xm, ep50, color=ec_col,   lw=1.5, label='Median')
        ax.plot(xm, ep5,  color=ec_col,   lw=1.0, ls='--', label=f'{path_pctile}th')
        ax.plot(xm, ep95, color=ec_col,   lw=1.0, ls=':',  label=f'{ben_pctile}th')
        ax.fill_between(xm, ep5, ep95, color=ec_col, alpha=0.08)
        for pv in range(1, len(ec_tp) + 1):
            alpha = min(0.15 + 0.05 * pv, 0.5)
            ax.axhline(ec_tp[pv - 1], color='red',  ls=':', lw=0.4, alpha=alpha)
            ax.axhline(ec_tb[pv - 1], color='blue', ls=':', lw=0.4, alpha=alpha)
        for pv in [1, 4, max(analysis.point_values)]:
            if pv - 1 < len(ec_tp):
                ax.text(xm[0], ec_tp[pv - 1], f' +{pv}', fontsize=5, color='red',  va='bottom')
                ax.text(xm[0], ec_tb[pv - 1], f' -{pv}', fontsize=5, color='blue', va='top')
        ax.axhline(0, color='gray', lw=0.8, alpha=0.5)
        rug_top = ec.get('rug_top')
        rug_bot = ec.get('rug_bot')
        if rug_top is not None and rug_top < sa.shape[1]:
            obs_t = scores[sa[:, rug_top], dim]; obs_t = obs_t[~np.isnan(obs_t)]
            if len(obs_t):
                ax.plot(obs_t, np.full(len(obs_t), ylim_bound),
                        '|', color=ec_col, alpha=0.4, ms=3, mew=0.3)
        if rug_bot is not None and rug_bot < sa.shape[1]:
            obs_b2 = scores[sa[:, rug_bot], dim]; obs_b2 = obs_b2[~np.isnan(obs_b2)]
            if len(obs_b2):
                ax.plot(obs_b2, np.full(len(obs_b2), -ylim_bound),
                        '|', color='#888888', alpha=0.3, ms=3, mew=0.3)
        ax.set_ylim(-ylim_bound, ylim_bound)
        ax.set_xlabel(dataset_names[dim], fontsize=7)
        ax.set_ylabel('log LR', fontsize=7)
        ax.set_title(ec['title'], fontsize=7, fontweight='bold', color=ec_col)
        ax.legend(fontsize=5, framealpha=0.6)
        ax.grid(lw=0.2, alpha=0.2)


def _draw_aux_lr_row(fig, gs, row, dim, aux_md, aux_name, aux_color,
                     eff_idx, aux_tau_p_log, aux_tau_b_log,
                     ylim_bound, path_pctile, ben_pctile,
                     dataset_names, analysis, n_cols):
    """Draw one aux-sample LR+ row: blank density columns + aux LR+ in last column."""
    for c in range(n_cols - 1):
        fig.add_subplot(gs[row, c]).axis('off')

    ax = fig.add_subplot(gs[row, n_cols - 1])
    if aux_md is None or aux_md.get('lr') is None:
        ax.axis('off')
        return

    x_marg = aux_md['x']
    p5  = aux_md['lr']['p5']
    p50 = aux_md['lr']['p50']
    p95 = aux_md['lr']['p95']

    ax.plot(x_marg, p50, color=aux_color, lw=1.5, label='Median')
    ax.plot(x_marg, p5,  color=aux_color, lw=1.0, ls='--', label=f'{path_pctile}th')
    ax.plot(x_marg, p95, color=aux_color, lw=1.0, ls='--', label=f'{ben_pctile}th')
    ax.fill_between(x_marg, p5, p95, color=aux_color, alpha=0.10)

    for pv in range(1, len(aux_tau_p_log) + 1):
        alpha = min(0.15 + 0.05 * pv, 0.5)
        ax.axhline(aux_tau_p_log[pv - 1], color='red',  ls=':', lw=0.4, alpha=alpha)
        ax.axhline(aux_tau_b_log[pv - 1], color='blue', ls=':', lw=0.4, alpha=alpha)
    for pv in [1, 4, max(analysis.point_values)]:
        if pv - 1 < len(aux_tau_p_log):
            ax.text(x_marg[0], aux_tau_p_log[pv - 1], f' +{pv}', fontsize=5, color='red',  va='bottom')
            ax.text(x_marg[0], aux_tau_b_log[pv - 1], f' -{pv}', fontsize=5, color='blue', va='top')
    ax.axhline(0, color='gray', lw=0.8, alpha=0.5)

    # Rug ticks for the aux sample's own variants
    scores = analysis.ms.scores
    sa = analysis.ms.sample_assignments
    if eff_idx is not None and eff_idx < sa.shape[1]:
        obs_aux = scores[sa[:, eff_idx], dim]
        obs_aux = obs_aux[~np.isnan(obs_aux)]
        if len(obs_aux):
            ax.plot(obs_aux, np.full(len(obs_aux), ylim_bound),
                    '|', color=aux_color, alpha=0.4, ms=3, mew=0.3)
    neg_idx = analysis.b_idx if analysis.b_idx is not None else analysis.s_idx
    if neg_idx is not None:
        obs_b = scores[sa[:, neg_idx], dim]
        obs_b = obs_b[~np.isnan(obs_b)]
        if len(obs_b):
            ax.plot(obs_b, np.full(len(obs_b), -ylim_bound),
                    '|', color='#1D7AAB', alpha=0.3, ms=3, mew=0.3)

    ax.set_ylim(-ylim_bound, ylim_bound)
    ax.set_xlabel(dataset_names[dim], fontsize=7)
    ax.set_ylabel('log LR+', fontsize=7)
    ax.set_title(f'Marginal LR+ — {aux_name} ({dataset_names[dim]})',
                 fontsize=8, fontweight='bold', color=aux_color)
    ax.legend(fontsize=5, framealpha=0.6)
    ax.grid(lw=0.2, alpha=0.2)


def _add_marginal_strips(ax, scores_2d, mask, ev_vals, dim_x, dim_y,
                         pt_norm, cmap, marker, x_range, y_range,
                         dim_x_name='', dim_y_name=''):
    """Attach 1D rug strips for partial-observation variants.

    Top strip: variants observed only in dim_x (x score known, y missing).
      → dots at fixed y=0, no y-axis (y is meaningless).
    Right strip: variants observed only in dim_y (y score known, x missing).
      → dots at fixed x=0, no x-axis (x is meaningless).
    Both colored by evidence value.
    """
    has_x = ~np.isnan(scores_2d[:, dim_x])
    has_y = ~np.isnan(scores_2d[:, dim_y])
    partial_x = mask & has_x & ~has_y
    partial_y = mask & has_y & ~has_x

    if not partial_x.any() and not partial_y.any():
        return None

    divider = make_axes_locatable(ax)
    rng = np.random.default_rng(42)
    ax_top = None

    if partial_x.any():
        ax_top = divider.append_axes('top', size='10%', pad=0.03, sharex=ax)
        ax_top.set_xlim(x_range)
        jitter = rng.uniform(-0.15, 0.15, partial_x.sum())
        ax_top.scatter(scores_2d[partial_x, dim_x], jitter,
                       c=ev_vals[partial_x], cmap=cmap, norm=pt_norm,
                       s=14, alpha=0.85, linewidths=0.3,
                       edgecolors='#333333', marker=marker)
        ax_top.set_ylim(-0.5, 0.5)
        ax_top.set_yticks([])
        ax_top.tick_params(labelbottom=False, bottom=False, left=False)
        label_x = dim_x_name or f'dim {dim_x}'
        ax_top.set_ylabel(f'{label_x} only\n(n={partial_x.sum()})', fontsize=5,
                          rotation=0, labelpad=44, va='center')
        ax_top.set_facecolor('#f4f4f4')
        for sp in ax_top.spines.values():
            sp.set_linewidth(0.3); sp.set_color('#cccccc')

    if partial_y.any():
        ax_right = divider.append_axes('right', size='10%', pad=0.03, sharey=ax)
        ax_right.set_ylim(y_range)
        jitter = rng.uniform(-0.15, 0.15, partial_y.sum())
        ax_right.scatter(jitter, scores_2d[partial_y, dim_y],
                         c=ev_vals[partial_y], cmap=cmap, norm=pt_norm,
                         s=14, alpha=0.85, linewidths=0.3,
                         edgecolors='#333333', marker=marker)
        ax_right.set_xlim(-0.5, 0.5)
        ax_right.set_xticks([])
        ax_right.tick_params(labelleft=False, left=False, bottom=False)
        label_y = dim_y_name or f'dim {dim_y}'
        ax_right.set_xlabel(f'{label_y} only\n(n={partial_y.sum()})', fontsize=5, labelpad=2)
        ax_right.set_facecolor('#f4f4f4')
        for sp in ax_right.spines.values():
            sp.set_linewidth(0.3); sp.set_color('#cccccc')

    return ax_top


def _plot_mv_2d(analysis, config, all_fits, marginal_data, x_grids,
                scores, sa, N, D, S, dataset_names, sample_names,
                points, tau_p_log, tau_b_log, median_prior, C_path, C_ben,
                path_pctile, ben_pctile, max_pt, pt_norm, ylim_bound, missing_frac,
                model_label, n_boots_used, pad, n_grid, contour_levels,
                figsize, suptitle,
                aux_p_entries=None, aux_marginal_data=None,
                aux_vs_primary_marginal_data=None,
                first_row_only=False, sample_style=None, lr_grids_2d=None,
                aux_vs_primary_grids_2d=None,
                density_grids_2d=None):
    """Layout for D=2: 2D grid, density contours, marginals."""
    aux_p_entries = aux_p_entries or []
    aux_marginal_data = aux_marginal_data or {}
    aux_vs_primary_marginal_data = aux_vs_primary_marginal_data or {}
    n_aux = len(aux_p_entries)
    _sc = sample_style['colors']  if sample_style else [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(S)]
    _se = sample_style['edges']   if sample_style else [_SAMPLE_EDGE_COLORS[i % len(_SAMPLE_EDGE_COLORS)] for i in range(S)]
    _sm = sample_style['markers'] if sample_style else [SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)] for i in range(S)]

    x1g = np.linspace(np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad, n_grid)
    x2g = np.linspace(np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad, n_grid)
    x1_range = (x1g[0], x1g[-1])
    x2_range = (x2g[0], x2g[-1])
    complete = ~np.isnan(scores).any(axis=1)

    if lr_grids_2d is not None:
        _all_grids = lr_grids_2d
    else:
        r_cfg      = analysis.results[config]
        _aux_res   = r_cfg.get('aux_results', {})
        _p_configs = [{
            'p_idx':       analysis.p_idx,
            'tau_p_log':   r_cfg['tau_p_log'],
            'tau_b_log':   r_cfg['tau_b_log'],
            'path_pctile': r_cfg.get('path_percentile', 5),
            'ben_pctile':  r_cfg.get('ben_percentile', 95),
            'point_values': analysis.point_values,
        }]
        for _, eff_idx in aux_p_entries:
            _ar = next((v for k, v in _aux_res.items() if v.get('eff_idx') == eff_idx), {})
            _p_configs.append({
                'p_idx':       eff_idx,
                'tau_p_log':   _ar.get('tau_p_log', r_cfg['tau_p_log']),
                'tau_b_log':   _ar.get('tau_b_log', r_cfg['tau_b_log']),
                'path_pctile': r_cfg.get('path_percentile', 5),
                'ben_pctile':  r_cfg.get('ben_percentile', 95),
                'point_values': analysis.point_values,
            })
        print(f"  Computing LR+ grid ({n_grid}×{n_grid}) for {len(_p_configs)} "
              f"pathogenic index(es)...")
        _all_grids = _compute_lr_grids_for_all_p(
            all_fits, x1g, x2g, scores.shape[1],
            analysis.b_idx, getattr(analysis, 's_idx', None),
            analysis.benign_method, _p_configs,
            g_idx=getattr(analysis, 'g_idx', None),
            prior=analysis.results[config].get('median_prior', None),
        )
    grid_points, lr_conservative = _all_grids[0]

    # ── first_row_only: col=sample, single row (D=2) ─────────────────────────
    if first_row_only:
        n_cols_s = S
        if figsize is None:
            figsize = (4.8 * n_cols_s, 5.5)
        # constrained_layout=False: make_axes_locatable (for marginal strips)
        # is incompatible with tight_layout/constrained_layout
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs = gridspec.GridSpec(1, n_cols_s, figure=fig,
                               hspace=0.4, wspace=0.45,
                               left=0.08, right=0.97, top=0.88, bottom=0.22)

        # Map each variant to its calibrated evidence value via nearest grid cell
        from scipy.interpolate import RegularGridInterpolator
        _gp_interp = RegularGridInterpolator(
            (x1g, x2g), grid_points.astype(float),
            method='nearest', bounds_error=False, fill_value=0,
        )

        for s_idx in range(S):
            ax = fig.add_subplot(gs[0, s_idx])
            mask_any = sa[:, s_idx]  # all variants in this sample
            mask = mask_any & complete  # only complete (both dims observed)
            n_s = int(mask_any.sum())

            # Evidence for all variants in this sample via nearest-grid lookup
            ev_all = np.zeros(len(scores), dtype=int)
            ev_all[mask_any] = points[mask_any]  # use pre-computed point assignments

            if mask.any():
                # Scatter complete variants colored by evidence
                ax.scatter(scores[mask, 0], scores[mask, 1],
                           c=ev_all[mask], cmap=POINT_CMAP, norm=pt_norm,
                           s=14, alpha=0.85, linewidths=0.3,
                           edgecolors='#333333', zorder=2,
                           marker=_sm[s_idx])

            # Faint evidence region background + ±1 boundary contours
            if grid_points.any():
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ax.contourf(x1g, x2g, grid_points.T,
                                levels=[-8.5, -0.5, 0.5, 8.5],
                                colors=['#2c7bb6', '#f7f7f7', '#d7191c'],
                                alpha=0.07, zorder=1)
                    for lvl, col in [(0.5, '#d7191c'), (-0.5, '#2c7bb6')]:
                        try:
                            ax.contour(x1g, x2g, grid_points.T,
                                       levels=[lvl], colors=[col],
                                       linewidths=0.7, alpha=0.5, zorder=3)
                        except Exception:
                            pass

            # Marginal strips for partial-observation variants
            ax_top_strip = _add_marginal_strips(
                ax, scores, mask_any, ev_all,
                0, 1, pt_norm, POINT_CMAP,
                _sm[s_idx], x1_range, x2_range,
                dim_x_name=dataset_names[0],
                dim_y_name=dataset_names[1])

            ax.set_xlim(x1_range); ax.set_ylim(x2_range)
            ax.set_xlabel(dataset_names[0], fontsize=8)
            ax.set_ylabel(dataset_names[1], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_facecolor('#f8f8f8')
            ax.grid(lw=0.15, alpha=0.3, zorder=0, color='white')
            # Set title on the top strip if it exists, otherwise on the main axes
            title_ax = ax_top_strip if ax_top_strip is not None else ax
            title_ax.set_title(f'{sample_names[s_idx]}  (n={n_s})',
                               fontsize=9, fontweight='bold', color=_sc[s_idx],
                               pad=3)

        # Legends: samples (top row) + evidence points (bottom row)
        sample_handles = [
            Line2D([0], [0], marker=_sm[i], color='#333333',
                   markerfacecolor=_sc[i], markersize=7,
                   linewidth=0, markeredgewidth=0.4,
                   label=f"{sample_names[i]} (n={int(sa[:, i].sum())})")
            for i in range(min(S, 4)) if sa[:, i].sum() > 0
        ]
        present_evs = sorted({int(v) for v in np.unique(grid_points) if -8 <= v <= 8})
        ev_handles = [
            Patch(facecolor=POINT_CMAP(pt_norm(pv)), edgecolor='#333333',
                  linewidth=0.3, label=f"{'+' if pv > 0 else ''}{pv}")
            for pv in present_evs
        ]
        fig.suptitle(suptitle, fontsize=10, fontweight='bold')
        # Single combined legend: sample handles on left, separator, evidence on right
        sep = [Line2D([0], [0], color='none', label='  ')]
        all_handles = sample_handles + sep + ev_handles
        fig.legend(handles=all_handles, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), ncol=len(all_handles),
                   fontsize=7.5, frameon=True, title_fontsize=8,
                   columnspacing=0.8, handletextpad=0.4)
        info = {'grid_points': grid_points, 'lr_conservative': lr_conservative,
                'n_boots_used': n_boots_used}
        return fig, info

    # ── full layout ────────────────────────────────────────────────────────────
    # n_aux extra LR cols (aux vs benign) + n_aux extra LR cols (aux vs primary)
    n_extra_lr = n_aux * 2
    n_grid_cols = 1 + n_aux          # row 0 grid panels
    # marginal rows: S density cols + 1 primary LR col + n_extra_lr extra LR cols
    n_cols = max(n_grid_cols, S + 1 + n_extra_lr)

    n_marg_rows = D                  # one row per dimension (aux now in extra cols)
    height_ratios = [1.2, 1.2] + [0.8] * D
    n_total_rows = 2 + D

    if figsize is None:
        figsize = (5.5 * n_cols, 5.0 + 4.0 * D)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_total_rows, n_cols, figure=fig,
                           height_ratios=height_ratios,
                           hspace=0.45, wspace=0.25)

    # Row 0 col 0: primary 2D point grid
    ax = fig.add_subplot(gs[0, 0])
    for s_idx in range(min(S, 4) - 1, -1, -1):   # descending: 3→0 so P/LP on top
        mask = sa[:, s_idx] & complete
        if not mask.any(): continue
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   color=_sc[s_idx], s=14, alpha=0.7,
                   edgecolors=_se[s_idx], linewidths=0.5, zorder=1,
                   marker=_sm[s_idx])
    im = ax.pcolormesh(x1g, x2g, grid_points.T, cmap=POINT_CMAP,
                       norm=pt_norm, shading='auto', alpha=0.72, zorder=2)
    plt.colorbar(im, ax=ax, label='Evidence Points', shrink=0.8)
    ax.set_xlabel(dataset_names[0], fontsize=8); ax.set_ylabel(dataset_names[1], fontsize=8)
    ax.set_xlim(x1_range); ax.set_ylim(x2_range)
    ax.set_aspect('auto')
    ax.set_title(f'Point Regions (P vs B)\nprior={median_prior:.4f}', fontsize=9, fontweight='bold')
    ax.grid(lw=0.2, alpha=0.3, zorder=0)

    # Row 0 cols 1..n_aux: aux-vs-benign 2D point grids
    _aux_results = analysis.results.get(config, {}).get('aux_results', {})
    for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
        ax = fig.add_subplot(gs[0, 1 + ai])
        aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
        aux_name = (sample_names[fixed_idx] if fixed_idx < len(sample_names)
                    else f'Sample {fixed_idx}')
        aux_gp, aux_lr_con = _all_grids[1 + ai]
        for s_idx in range(min(S, 4) - 1, -1, -1):
            mask = sa[:, s_idx] & complete
            if not mask.any(): continue
            ax.scatter(scores[mask, 0], scores[mask, 1],
                       color=_sc[s_idx], s=14, alpha=0.7,
                       edgecolors=_se[s_idx], linewidths=0.5, zorder=1,
                       marker=_sm[s_idx])
        im2 = ax.pcolormesh(x1g, x2g, aux_gp.T, cmap=POINT_CMAP,
                            norm=pt_norm, shading='auto', alpha=0.72, zorder=2)
        plt.colorbar(im2, ax=ax, label='Evidence Points', shrink=0.8)
        ax.set_xlabel(dataset_names[0], fontsize=8); ax.set_ylabel(dataset_names[1], fontsize=8)
        ax.set_xlim(x1_range); ax.set_ylim(x2_range)
        ax.set_aspect('auto')
        _ar = _aux_results.get(fixed_idx, {})
        _ap = _ar.get('median_prior', float('nan'))
        ax.set_title(f'Aux vs B: {aux_name}\nprior={_ap:.4f}',
                     fontsize=9, fontweight='bold', color=aux_color)
        ax.grid(lw=0.2, alpha=0.3)

    # Fill any unused row-0 columns
    for c_idx in range(n_grid_cols, n_cols):
        fig.add_subplot(gs[0, c_idx]).axis('off')

    # ── Bottom legends ──────────────────────────────────────────────────────
    sample_handles = [
        Line2D([0], [0], marker=_sm[s_idx],
               color=_se[s_idx],
               markerfacecolor=_sc[s_idx],
               markersize=9, linewidth=0, markeredgewidth=0.8,
               label=f"{sample_names[s_idx]} (n={int(sa[:, s_idx].sum())})")
        for s_idx in range(min(S, 4))
    ]
    all_grid_arrays = [grid_points] + [_all_grids[1 + ai][0] for ai in range(n_aux)]
    present_evs = sorted({int(v) for gp in all_grid_arrays
                           for v in np.unique(gp) if -8 <= v <= 8})
    ev_handles = [
        Patch(facecolor=POINT_CMAP(pt_norm(pv)), edgecolor='gray', linewidth=0.3,
              label=f"{'+' if pv > 0 else ''}{pv}")
        for pv in present_evs
    ]
    leg_ev = fig.legend(handles=ev_handles, loc='lower center',
                        bbox_to_anchor=(0.5, 0.0), fontsize=7.5,
                        frameon=True, title='Evidence points', title_fontsize=8,
                        ncol=len(present_evs))
    leg_sample = fig.legend(handles=sample_handles, loc='lower center',
                             bbox_to_anchor=(0.5, 0.10), fontsize=7.5,
                             frameon=True, title='Samples', title_fontsize=8,
                             ncol=len(sample_handles))
    fig.subplots_adjust(bottom=0.25)

    if not first_row_only:
        # Row 1: per-sample 2D density contours
        if all_fits:
            for s_idx in range(min(S, n_cols)):
                ax = fig.add_subplot(gs[1, s_idx])
                d = (density_grids_2d[s_idx] if density_grids_2d is not None
                     else _compute_sample_density_grid(all_fits, s_idx, x1g, x2g))
                if d is not None:
                    d_mean, d_std = d['mean'], d['std']
                    levels = np.linspace(d_mean.max() * 0.01, d_mean.max() * 0.95, contour_levels)
                    cmap_name = 'Greens' if s_idx >= 2 else ('Reds' if s_idx == 0 else 'Blues')
                    if levels[-1] > levels[0]:
                        ax.contourf(x1g, x2g, d_mean.T, levels=levels, cmap=cmap_name, alpha=0.4)
                        ax.contour(x1g, x2g, d_mean.T, levels=levels,
                                   colors=_sc[s_idx], linewidths=0.5, alpha=0.6)
                    outer = levels[1] if len(levels) > 1 else levels[0]
                    for bound, ls in [(np.maximum(d_mean - d_std, 0), ':'), (d_mean + d_std, ':')]:
                        ax.contour(x1g, x2g, bound.T, levels=[outer],
                                   colors=_sc[s_idx], linewidths=0.3, linestyles=ls, alpha=0.3)
                mask = sa[:, s_idx] & complete
                if mask.any():
                    ax.scatter(scores[mask, 0], scores[mask, 1],
                               c=_sc[s_idx], s=4, alpha=0.3, edgecolors='none')
                _plot_component_means(ax, all_fits, s_idx)
                ax.set_xlim(x1_range); ax.set_ylim(x2_range)
                ax.set_xlabel(dataset_names[0], fontsize=7); ax.set_ylabel(dataset_names[1], fontsize=7)
                n_s = sa[:, s_idx].sum()
                ax.set_title(f'{sample_names[s_idx]} (n={n_s})', fontsize=8, fontweight='bold',
                             color=_sc[s_idx])
                ax.grid(lw=0.2, alpha=0.2)
        for c_idx in range(S, n_cols):
            fig.add_subplot(gs[1, c_idx]).axis('off')

        # Rows 2+: marginals — one row per dim, aux comparisons as extra LR columns
        for dim in range(D):
            base_row = 2 + dim
            md = marginal_data[dim]
            if md is None:
                for c in range(n_cols):
                    fig.add_subplot(gs[base_row, c]).axis('off')
                continue

            # Build extra_lr_cols: aux-vs-benign then aux-vs-primary for each aux
            extra_cols = []
            for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
                aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
                aux_name = (sample_names[fixed_idx] if fixed_idx < len(sample_names)
                            else f'Sample {fixed_idx}')
                aux_r = _aux_results.get(fixed_idx, {})
                # aux vs benign
                extra_cols.append({
                    'md':      (aux_marginal_data.get(fixed_idx) or {}).get(dim),
                    'title':   f'{aux_name}\nvs Benign — {dataset_names[dim]}',
                    'color':   aux_color,
                    'tau_p':   aux_r.get('tau_p_log', tau_p_log),
                    'tau_b':   aux_r.get('tau_b_log', tau_b_log),
                    'rug_top': eff_idx,
                    'rug_bot': analysis.b_idx,
                })
                # aux vs primary pathogenic
                p_name = sample_names[analysis.p_idx] if analysis.p_idx is not None else 'P/LP'
                extra_cols.append({
                    'md':      (aux_vs_primary_marginal_data.get(fixed_idx) or {}).get(dim),
                    'title':   f'{aux_name}\nvs {p_name} — {dataset_names[dim]}',
                    'color':   aux_color,
                    'tau_p':   tau_p_log,
                    'tau_b':   tau_b_log,
                    'rug_top': eff_idx,
                    'rug_bot': analysis.p_idx,
                })

            _draw_marginal_row(fig, gs, base_row, dim, md, scores, sa, S, n_cols,
                               dataset_names, sample_names, tau_p_log, tau_b_log,
                               ylim_bound, path_pctile, ben_pctile, analysis,
                               sample_style=sample_style, extra_lr_cols=extra_cols)

    fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=1.02)
    info = {
        'marginal_data': marginal_data, 'x1g': x1g, 'x2g': x2g,
        'grid_points': grid_points, 'lr_conservative': lr_conservative,
        'n_boots_used': n_boots_used, 'latent_q': analysis._latent_q,
    }
    return fig, info


def _plot_mv_umap(analysis, config, scores, sa, N, D, S,
                  dataset_names, sample_names, points, max_pt, pt_norm,
                  model_label, n_boots_used, missing_frac,
                  umap_data, figsize, suptitle, sample_style,
                  first_row_only=False):
    """UMAP projection plot for D>2.  Points colored by evidence; partial
    observations (imputed dims) drawn with a hollow marker to distinguish them."""
    embedding    = umap_data['embedding']     # (N, 2)
    imputed_mask = umap_data['imputed_mask']  # (N,) bool

    _sc = sample_style['colors']  if sample_style else [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(S)]
    _se = sample_style['edges']   if sample_style else [_SAMPLE_EDGE_COLORS[i % len(_SAMPLE_EDGE_COLORS)] for i in range(S)]
    _sm = sample_style['markers'] if sample_style else [SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)] for i in range(S)]

    complete = ~np.isnan(scores).any(axis=1)

    # Global axis limits across all sample variants (ignoring NaN rows)
    valid_emb = embedding[~np.isnan(embedding).any(axis=1)]
    pad_umap  = 0.5
    xlim = (valid_emb[:, 0].min() - pad_umap, valid_emb[:, 0].max() + pad_umap)
    ylim = (valid_emb[:, 1].min() - pad_umap, valid_emb[:, 1].max() + pad_umap)

    if first_row_only:
        n_cols = S
        if figsize is None:
            figsize = (4.8 * n_cols, 5.5)
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs  = gridspec.GridSpec(1, n_cols, figure=fig,
                                hspace=0.4, wspace=0.45,
                                left=0.08, right=0.97, top=0.88, bottom=0.22)

        for s_idx in range(S):
            ax = fig.add_subplot(gs[0, s_idx])
            mask_any  = sa[:, s_idx].astype(bool)
            mask_comp = mask_any & complete
            mask_part = mask_any & ~complete
            n_s = int(mask_any.sum())

            if mask_any.any():
                ax.scatter(embedding[mask_any, 0], embedding[mask_any, 1],
                           c=points[mask_any], cmap=POINT_CMAP, norm=pt_norm,
                           s=14, alpha=0.85, linewidths=0.3,
                           edgecolors='#333333', zorder=2, marker=_sm[s_idx])

            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel('UMAP 1', fontsize=8)
            ax.set_ylabel('UMAP 2', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_facecolor('#f8f8f8')
            ax.grid(lw=0.15, alpha=0.3, zorder=0, color='white')
            ax.set_title(f'{sample_names[s_idx]}  (n={n_s})',
                         fontsize=9, fontweight='bold', color=_sc[s_idx], pad=3)

        # Legend: samples + evidence patches + partial marker explanation
        sample_handles = [
            Line2D([0], [0], marker=_sm[i], color='#333333',
                   markerfacecolor=_sc[i], markersize=7,
                   linewidth=0, markeredgewidth=0.4,
                   label=f"{sample_names[i]} (n={int(sa[:, i].sum())})")
            for i in range(min(S, 4)) if sa[:, i].sum() > 0
        ]
        present_evs = sorted({int(v) for v in np.unique(points) if -8 <= v <= 8})
        ev_handles = [
            Patch(facecolor=POINT_CMAP(pt_norm(pv)), edgecolor='#333333',
                  linewidth=0.3, label=f"{'+' if pv > 0 else ''}{pv}")
            for pv in present_evs
        ]
        sep = [Line2D([0], [0], color='none', label='  ')]
        all_handles = sample_handles + sep + ev_handles
        fig.suptitle(suptitle, fontsize=10, fontweight='bold')
        fig.legend(handles=all_handles, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), ncol=len(all_handles),
                   fontsize=7.5, frameon=True, columnspacing=0.8, handletextpad=0.4)
        return fig, {'n_boots_used': n_boots_used}

    # ── full layout: one panel per sample, arranged in a grid ────────────────
    ncols = min(S, 3)
    nrows = int(np.ceil(S / ncols))
    if figsize is None:
        figsize = (4.8 * ncols, 5.0 * nrows)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                            hspace=0.45, wspace=0.4,
                            left=0.08, right=0.97, top=0.90, bottom=0.14)

    for s_idx in range(S):
        row, col = divmod(s_idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        mask_any  = sa[:, s_idx].astype(bool)
        mask_comp = mask_any & complete
        mask_part = mask_any & ~complete
        n_s = int(mask_any.sum())

        if mask_comp.any():
            ax.scatter(embedding[mask_comp, 0], embedding[mask_comp, 1],
                       c=points[mask_comp], cmap=POINT_CMAP, norm=pt_norm,
                       s=14, alpha=0.85, linewidths=0.3,
                       edgecolors='#333333', zorder=2, marker=_sm[s_idx])
        if mask_part.any():
            ax.scatter(embedding[mask_part, 0], embedding[mask_part, 1],
                       c=points[mask_part], cmap=POINT_CMAP, norm=pt_norm,
                       s=18, alpha=0.75, linewidths=1.0,
                       edgecolors=_sc[s_idx], facecolors='none',
                       zorder=2, marker=_sm[s_idx])

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel('UMAP 1', fontsize=8)
        ax.set_ylabel('UMAP 2', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor('#f8f8f8')
        ax.grid(lw=0.15, alpha=0.3, zorder=0, color='white')
        ax.set_title(f'{sample_names[s_idx]}  (n={n_s})',
                     fontsize=9, fontweight='bold', color=_sc[s_idx])

    present_evs = sorted({int(v) for v in np.unique(points) if -8 <= v <= 8})
    ev_handles = [
        Patch(facecolor=POINT_CMAP(pt_norm(pv)), edgecolor='#333333',
              linewidth=0.3, label=f"{'+' if pv > 0 else ''}{pv}")
        for pv in present_evs
    ]
    partial_handle = Line2D([0], [0], marker='o', color='#888888',
                            markerfacecolor='none', markersize=7,
                            linewidth=0, markeredgewidth=1.0,
                            label='partial obs. (imputed)')
    sep = [Line2D([0], [0], color='none', label='  ')]
    fig.suptitle(suptitle, fontsize=10, fontweight='bold')
    fig.legend(handles=ev_handles + sep + [partial_handle],
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=len(ev_handles) + 2, fontsize=7.5, frameon=True,
               columnspacing=0.8, handletextpad=0.4)
    return fig, {'n_boots_used': n_boots_used}


def _plot_mv_hd(analysis, config, all_fits, marginal_data, x_grids,
                scores, sa, N, D, S, dataset_names, sample_names,
                points, tau_p_log, tau_b_log, median_prior, C_path, C_ben,
                path_pctile, ben_pctile, max_pt, pt_norm, ylim_bound,
                model_label, n_boots_used, pad, n_grid, contour_levels,
                figsize, suptitle,
                aux_p_entries=None, aux_marginal_data=None,
                aux_vs_primary_marginal_data=None,
                max_lr_pairs=10,
                sample_style=None, pairwise_grids_hd=None, aux_pairwise_grids_hd=None,
                pairwise_dim_pairs=None,
                first_row_only=False):
    """Layout for D>2.

    Row 0: pairwise LR+ grids — all C(D,2) combinations up to max_lr_pairs,
           followed by aux pairwise grids for each aux sample (dim 0 vs dim 1 only).
    Rows 1..D: per-dimension marginals — [S density cols] + [primary LR+] +
               [aux-vs-benign LR+] + [aux-vs-primary LR+] for each aux.
    """
    from itertools import combinations as _combinations
    aux_p_entries = aux_p_entries or []
    aux_marginal_data = aux_marginal_data or {}
    aux_vs_primary_marginal_data = aux_vs_primary_marginal_data or {}
    n_aux = len(aux_p_entries)
    _aux_results = analysis.results.get(config, {}).get('aux_results', {})
    _sc = sample_style['colors']  if sample_style else [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(S)]
    _se = sample_style['edges']   if sample_style else [_SAMPLE_EDGE_COLORS[i % len(_SAMPLE_EDGE_COLORS)] for i in range(S)]
    _sm = sample_style['markers'] if sample_style else [SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)] for i in range(S)]

    # Use precomputed pairs if provided, else fall back to all combinations.
    # max_lr_pairs=None means no limit.
    if pairwise_dim_pairs is not None:
        all_dim_pairs = pairwise_dim_pairs if max_lr_pairs is None else pairwise_dim_pairs[:max_lr_pairs]
    else:
        base = list(_combinations(range(D), 2))
        all_dim_pairs = base if max_lr_pairs is None else base[:max_lr_pairs]
    n_pairs = len(all_dim_pairs)
    complete = ~np.isnan(scores).any(axis=1)

    def _get_pair_grid(k, dim_i, dim_j):
        if pairwise_grids_hd is not None and k < len(pairwise_grids_hd):
            return pairwise_grids_hd[k]
        print(f"  Computing LR+ grid: dim {dim_i} vs dim {dim_j}...")
        return _compute_conservative_lr_grid(
            analysis, config, all_fits, x_grids[dim_i], x_grids[dim_j],
            dim_i=dim_i, dim_j=dim_j, total_dims=D,
        )

    # ── first_row_only: rows=dim pairs, cols=samples + aux-vs-B ─────────────
    if first_row_only:
        n_rows_f = n_pairs
        n_cols_f = S + n_aux
        if figsize is None:
            figsize = (4.8 * n_cols_f, 5.0 * n_rows_f)
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs = gridspec.GridSpec(n_rows_f, n_cols_f, figure=fig,
                               hspace=0.5, wspace=0.45,
                               left=0.08, right=0.97, top=0.93, bottom=0.22)

        def _get_aux_pair_grid(fixed_idx, k):
            """Return (grid_points, lr_conservative) for aux fixed_idx, dim pair k."""
            if aux_pairwise_grids_hd and fixed_idx in aux_pairwise_grids_hd:
                grids = aux_pairwise_grids_hd[fixed_idx]
                if k < len(grids):
                    return grids[k]
            return None, None

        for k, (dim_i, dim_j) in enumerate(all_dim_pairs):
            xig, xjg = x_grids[dim_i], x_grids[dim_j]
            grid_points, lr_conservative = _get_pair_grid(k, dim_i, dim_j)
            from scipy.interpolate import RegularGridInterpolator
            _gp_interp = RegularGridInterpolator(
                (xig, xjg), grid_points.astype(float),
                method='nearest', bounds_error=False, fill_value=0,
            )
            has_i = ~np.isnan(scores[:, dim_i])
            has_j = ~np.isnan(scores[:, dim_j])
            complete_ij = has_i & has_j

            for s_idx in range(S):
                ax = fig.add_subplot(gs[k, s_idx])
                mask_any = sa[:, s_idx]
                mask = mask_any & complete_ij
                n_s = int(mask_any.sum())

                ev_all = points.copy()

                if mask.any():
                    ax.scatter(scores[mask, dim_i], scores[mask, dim_j],
                               c=ev_all[mask], cmap=POINT_CMAP, norm=pt_norm,
                               s=14, alpha=0.85, linewidths=0.3,
                               edgecolors='#333333', zorder=2,
                               marker=_sm[s_idx])

                if grid_points.any():
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        ax.contourf(xig, xjg, grid_points.T,
                                    levels=[-8.5, -0.5, 0.5, 8.5],
                                    colors=['#2c7bb6', '#f7f7f7', '#d7191c'],
                                    alpha=0.07, zorder=1)
                        for lvl, col in [(0.5, '#d7191c'), (-0.5, '#2c7bb6')]:
                            try:
                                ax.contour(xig, xjg, grid_points.T,
                                           levels=[lvl], colors=[col],
                                           linewidths=0.7, alpha=0.5, zorder=3)
                            except Exception:
                                pass

                ax_top_strip = _add_marginal_strips(
                    ax, scores, mask_any, ev_all,
                    dim_i, dim_j, pt_norm, POINT_CMAP,
                    _sm[s_idx],
                    (xig[0], xig[-1]), (xjg[0], xjg[-1]),
                    dim_x_name=dataset_names[dim_i],
                    dim_y_name=dataset_names[dim_j])

                ax.set_xlim(xig[0], xig[-1]); ax.set_ylim(xjg[0], xjg[-1])
                ax.set_xlabel(dataset_names[dim_i], fontsize=7)
                ax.set_ylabel(dataset_names[dim_j], fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_facecolor('#f8f8f8')
                ax.grid(lw=0.15, alpha=0.3, zorder=0, color='white')
                if k == 0:
                    title_ax = ax_top_strip if ax_top_strip is not None else ax
                    title_ax.set_title(f'{sample_names[s_idx]} (n={n_s})',
                                       fontsize=8, fontweight='bold',
                                       color=_sc[s_idx], pad=3)
                if s_idx == 0:
                    ax.set_ylabel(
                        f'{dataset_names[dim_i]} vs {dataset_names[dim_j]}\n{dataset_names[dim_j]}',
                        fontsize=7)

            # Aux-vs-B columns (columns S..S+n_aux-1)
            for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
                aux_col = S + ai
                aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
                aux_name = sample_names[eff_idx] if eff_idx < len(sample_names) else f'Aux {ai}'
                agp, _ = _get_aux_pair_grid(fixed_idx, k)
                ax = fig.add_subplot(gs[k, aux_col])
                mask_aux = sa[:, eff_idx] if eff_idx < S else np.zeros(N, dtype=bool)
                mask_aux_ij = mask_aux & complete_ij

                if mask_aux_ij.any():
                    ax.scatter(scores[mask_aux_ij, dim_i], scores[mask_aux_ij, dim_j],
                               c=[aux_color] * int(mask_aux_ij.sum()),
                               s=14, alpha=0.85, linewidths=0.3,
                               edgecolors='#333333', zorder=2,
                               marker=_sm[eff_idx % len(_sm)])

                if agp is not None and agp.any():
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        ax.contourf(xig, xjg, agp.T,
                                    levels=[-8.5, -0.5, 0.5, 8.5],
                                    colors=['#2c7bb6', '#f7f7f7', '#d7191c'],
                                    alpha=0.07, zorder=1)
                        for lvl, col in [(0.5, '#d7191c'), (-0.5, '#2c7bb6')]:
                            try:
                                ax.contour(xig, xjg, agp.T,
                                           levels=[lvl], colors=[col],
                                           linewidths=0.7, alpha=0.5, zorder=3)
                            except Exception:
                                pass

                ax.set_xlim(xig[0], xig[-1]); ax.set_ylim(xjg[0], xjg[-1])
                ax.set_xlabel(dataset_names[dim_i], fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_facecolor('#f8f8f8')
                ax.grid(lw=0.15, alpha=0.3, zorder=0, color='white')
                if k == 0:
                    ax.set_title(f'{aux_name}\nvs Benign (n={int(mask_aux.sum())})',
                                 fontsize=8, fontweight='bold', color=aux_color, pad=3)
                if ai == 0:
                    ax.set_ylabel(
                        f'{dataset_names[dim_i]} vs {dataset_names[dim_j]}\n{dataset_names[dim_j]}',
                        fontsize=7)

        all_gp = np.concatenate([_get_pair_grid(k, di, dj)[0].ravel()
                                  for k, (di, dj) in enumerate(all_dim_pairs)])
        present_evs = sorted({int(v) for v in np.unique(all_gp) if -8 <= v <= 8})
        ev_handles = [
            Patch(facecolor=POINT_CMAP(pt_norm(pv)), edgecolor='#333333',
                  linewidth=0.3, label=f"{'+' if pv > 0 else ''}{pv}")
            for pv in present_evs
        ]
        sample_handles = [
            Line2D([0], [0], marker=_sm[i], color='#333333',
                   markerfacecolor=_sc[i], markersize=7,
                   linewidth=0, markeredgewidth=0.4,
                   label=f"{sample_names[i]} (n={int(sa[:, i].sum())})")
            for i in range(min(S, 4)) if sa[:, i].sum() > 0
        ]
        fig.suptitle(suptitle, fontsize=10, fontweight='bold')
        sep = [Line2D([0], [0], color='none', label='  ')]
        all_handles = sample_handles + sep + ev_handles
        fig.legend(handles=all_handles, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), ncol=len(all_handles),
                   fontsize=7.5, frameon=True, title_fontsize=8,
                   columnspacing=0.8, handletextpad=0.4)
        return fig, {'n_boots_used': n_boots_used}

    # ── full layout ────────────────────────────────────────────────────────────
    n_extra_lr = n_aux * 2   # per dim: n_aux (vs benign) + n_aux (vs primary)
    # Row 0 needs n_pairs + n_aux cols; marginal rows need S + 1 + n_extra_lr cols
    n_cols  = max(n_pairs + n_aux, S + 1 + n_extra_lr)
    # One row per dimension (aux comparisons are extra columns, not extra rows)
    n_rows  = 1 + D
    height_ratios = [1.6] + [0.9] * D
    if figsize is None:
        figsize = (max(4.5 * n_cols, 4.5 * (n_pairs + n_aux)),
                   4.0 + 3.0 * D)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           height_ratios=height_ratios,
                           hspace=0.50, wspace=0.35)

    # ═══════════════════════════════════════
    # Row 0: primary pairwise grids then aux grids (dim 0 vs dim 1 for each aux)
    # ═══════════════════════════════════════
    for k, (dim_i, dim_j) in enumerate(all_dim_pairs):
        xig = x_grids[dim_i]
        xjg = x_grids[dim_j]
        xi_range = (xig[0], xig[-1])
        xj_range = (xjg[0], xjg[-1])

        grid_points, lr_conservative = _get_pair_grid(k, dim_i, dim_j)

        ax = fig.add_subplot(gs[0, k])
        im = ax.pcolormesh(xig, xjg, grid_points.T, cmap=POINT_CMAP,
                           norm=pt_norm, shading='auto', alpha=0.7)
        plt.colorbar(im, ax=ax, label='Points', shrink=0.8)

        for s_idx in range(S):
            mask = sa[:, s_idx] & complete
            if not mask.any():
                continue
            ax.scatter(scores[mask, dim_i], scores[mask, dim_j],
                       c=points[mask], cmap=POINT_CMAP, norm=pt_norm,
                       s=8, alpha=0.5,
                       edgecolors=_sc[s_idx], linewidths=0.3,
                       marker=_sm[s_idx])

        ax.set_xlabel(dataset_names[dim_i], fontsize=7)
        ax.set_ylabel(dataset_names[dim_j], fontsize=7)
        ax.set_xlim(xi_range); ax.set_ylim(xj_range)
        ax.set_title(f'{dataset_names[dim_i]} vs {dataset_names[dim_j]}',
                     fontsize=8, fontweight='bold')
        ax.grid(lw=0.2, alpha=0.3)

    # Aux pairwise grids (dim 0 vs dim 1 only, placed after primary pairs)
    if n_aux > 0 and D >= 2:
        xig, xjg = x_grids[0], x_grids[1]
        xi_range = (xig[0], xig[-1])
        xj_range = (xjg[0], xjg[-1])
        for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
            col = n_pairs + ai
            if col >= n_cols:
                break
            ax = fig.add_subplot(gs[0, col])
            aux_name = (sample_names[fixed_idx] if fixed_idx < len(sample_names)
                        else f'Sample {fixed_idx}')
            aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
            _aux_precomputed = (aux_pairwise_grids_hd or {}).get(fixed_idx, [])
            if _aux_precomputed:
                aux_gp, aux_lr_con = _aux_precomputed[0]
            else:
                print(f"  Computing aux LR+ grid for {aux_name} (idx={fixed_idx})...")
                aux_gp, aux_lr_con = _compute_conservative_lr_grid(
                    analysis, config, all_fits, xig, xjg,
                    dim_i=0, dim_j=1, total_dims=D, p_idx_override=eff_idx,
                )
            im2 = ax.pcolormesh(xig, xjg, aux_gp.T, cmap=POINT_CMAP,
                                norm=pt_norm, shading='auto', alpha=0.7)
            plt.colorbar(im2, ax=ax, label='Points', shrink=0.8)
            aux_r = analysis.results[config].get('aux_results', {}).get(fixed_idx, {})
            aux_pts = aux_r.get('points', points)
            for s_idx in range(S):
                mask = sa[:, s_idx] & complete
                if not mask.any(): continue
                ax.scatter(scores[mask, 0], scores[mask, 1],
                           c=aux_pts[mask], cmap=POINT_CMAP, norm=pt_norm,
                           s=8, alpha=0.5,
                           edgecolors=_sc[s_idx], linewidths=0.3,
                           marker=_sm[s_idx])
            ax.set_xlabel(dataset_names[0], fontsize=7)
            ax.set_ylabel(dataset_names[1], fontsize=7)
            ax.set_xlim(xi_range); ax.set_ylim(xj_range)
            _ar = _aux_results.get(fixed_idx, {})
            _ap = _ar.get('median_prior', float('nan'))
            _ac = _ar.get('C_path', float('nan'))
            ax.set_title(f'Aux: {aux_name}\nprior={_ap:.4f}, C_p={_ac:.1f}',
                         fontsize=8, fontweight='bold', color=aux_color)
            ax.grid(lw=0.2, alpha=0.3)

    # blank remaining columns in row 0
    for c_idx in range(n_pairs + n_aux, n_cols):
        fig.add_subplot(gs[0, c_idx]).axis('off')

    # ═══════════════════════════════════════
    # Rows 1+: per-dimension marginals — one row per dim, aux as extra LR columns
    # ═══════════════════════════════════════
    for dim in range(D):
        base_row = 1 + dim
        md = marginal_data[dim]
        if md is None:
            for c in range(n_cols):
                fig.add_subplot(gs[base_row, c]).axis('off')
            continue

        extra_cols = []
        for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
            aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
            aux_name = (sample_names[fixed_idx] if fixed_idx < len(sample_names)
                        else f'Sample {fixed_idx}')
            aux_r = _aux_results.get(fixed_idx, {})
            # aux vs benign
            extra_cols.append({
                'md':      (aux_marginal_data.get(fixed_idx) or {}).get(dim),
                'title':   f'{aux_name}\nvs Benign — {dataset_names[dim]}',
                'color':   aux_color,
                'tau_p':   aux_r.get('tau_p_log', tau_p_log),
                'tau_b':   aux_r.get('tau_b_log', tau_b_log),
                'rug_top': eff_idx,
                'rug_bot': analysis.b_idx,
            })
            # aux vs primary pathogenic
            p_name = sample_names[analysis.p_idx] if analysis.p_idx is not None else 'P/LP'
            extra_cols.append({
                'md':      (aux_vs_primary_marginal_data.get(fixed_idx) or {}).get(dim),
                'title':   f'{aux_name}\nvs {p_name} — {dataset_names[dim]}',
                'color':   aux_color,
                'tau_p':   tau_p_log,
                'tau_b':   tau_b_log,
                'rug_top': eff_idx,
                'rug_bot': analysis.p_idx,
            })

        _draw_marginal_row(fig, gs, base_row, dim, md, scores, sa, S, n_cols,
                           dataset_names, sample_names, tau_p_log, tau_b_log,
                           ylim_bound, path_pctile, ben_pctile, analysis,
                           sample_style=sample_style, extra_lr_cols=extra_cols)

    fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=1.01)
    info = {
        'marginal_data': marginal_data,
        'n_boots_used': n_boots_used,
        'latent_q': analysis._latent_q,
    }
    return fig, info


def _plot_component_means(ax, all_fits, s_idx):
    """Plot average component locations across bootstraps as stars."""
    K_counts = {}
    K_mu_sums = {}

    for fit in all_fits:
        params = fit['component_params']
        weights = fit['weights']
        for c, (mu, Delta, Gamma) in enumerate(params):
            w = weights[s_idx][c] if s_idx < weights.shape[0] else 0
            if w < 0.01:
                continue
            K_counts.setdefault(c, 0)
            K_mu_sums.setdefault(c, np.zeros_like(mu))
            K_counts[c] += 1
            K_mu_sums[c] = K_mu_sums[c] + mu

    for c in K_counts:
        if K_counts[c] > 0:
            mean_mu = K_mu_sums[c] / K_counts[c]
            if len(mean_mu) >= 2:
                ax.plot(mean_mu[0], mean_mu[1], '*', color='black',
                        markersize=6, markeredgecolor='white',
                        markeredgewidth=0.3, zorder=5)