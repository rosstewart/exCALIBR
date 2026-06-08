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
    all_p_indices = [analysis.p_idx] + [eff for _, eff in aux_p_entries]

    print(f"  Computing marginals for {len(all_p_indices)} pathogenic index(es) "
          f"× {D} dims ({n_boots_used} boots)...")
    all_marginals = _compute_all_marginals_batch_p(
        all_fits, x_grids, all_p_indices,
        analysis.b_idx, getattr(analysis, 's_idx', None),
        analysis.benign_method,
        ms.sample_assignments.shape[1],
        path_pctile, ben_pctile,
        g_idx=getattr(analysis, 'g_idx', None),
        prior=r.get('median_prior', None),
        n_jobs=n_jobs,
    )
    marginal_data = {d: all_marginals[0][d] for d in range(D)}
    aux_marginal_data = {}
    for ai, (fixed_idx, _) in enumerate(aux_p_entries):
        aux_marginal_data[fixed_idx] = {d: all_marginals[ai + 1][d] for d in range(D)}

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
    _do_pairwise = (D > 2 and projection == 'pairwise') or projection == 'activity_pairs'
    if _do_pairwise:
        from itertools import combinations as _combinations
        if projection == 'pairwise':
            pairwise_dim_pairs = list(_combinations(range(D), 2))
        else:
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
            dist = np.zeros((N_full, N_full), dtype=np.float32)
            for d in range(D):
                obs_d = observed[:, d]
                both = obs_d[:, None] & obs_d[None, :]
                diff = np.nan_to_num(full_scores[:, d], nan=0.0)
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
        'analysis':           analysis,
        'config':             config,
        'all_fits':           all_fits,
        'marginal_data':      marginal_data,
        'aux_marginal_data':  aux_marginal_data,
        'x_grids':            x_grids,
        'x_grids_plot':       x_grids_plot,
        'lr_grids_2d':        lr_grids_2d,
        'density_grids_2d':   density_grids_2d,
        'pairwise_grids_hd':  pairwise_grids_hd,
        'pairwise_dim_pairs': pairwise_dim_pairs,
        'umap_data':          umap_data,
        'n_boots_used':       n_boots_used,
        'n_grid':             n_grid,
        'pad':                pad,
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
        marginal_data_render   = {d: None for d in range(D)}
        aux_marginal_data_render = {fid: {d: None for d in range(D)}
                                    for fid, _ in getattr(analysis, 'aux_p_entries', [])}
    else:
        marginal_data_render      = marginal_data
        aux_marginal_data_render  = aux_marginal_data

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
            first_row_only=first_row_only, sample_style=sample_style,
            lr_grids_2d=precomputed.get('lr_grids_2d'),
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
                sample_style=sample_style,
                max_lr_pairs=max_lr_pairs,
                pairwise_grids_hd=precomputed.get('pairwise_grids_hd'),
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


_AUX_COLORS = ['#e6a817', '#8B4513', '#006400', '#800080', '#FF6600', '#4B0082']


def _draw_marginal_row(fig, gs, row, dim, md, scores, sa, S, n_cols,
                       dataset_names, sample_names, tau_p_log, tau_b_log,
                       ylim_bound, path_pctile, ben_pctile, analysis,
                       sample_style=None):
    """Draw one row: S sample density panels + primary pathogenic LR+ panel."""
    x_marg = md['x']
    _sc = sample_style['colors']  if sample_style else [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(S)]

    for s_idx in range(min(S, n_cols - 1)):
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

    # Primary LR+ panel (last column)
    ax = fig.add_subplot(gs[row, n_cols - 1])
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
                first_row_only=False, sample_style=None, lr_grids_2d=None,
                density_grids_2d=None):
    """Layout for D=2: 2D grid, density contours, marginals."""
    aux_p_entries = aux_p_entries or []
    aux_marginal_data = aux_marginal_data or {}
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
    n_grid_cols = 1 + n_aux
    n_cols = max(S, n_grid_cols)

    # Each of D=2 dimensions gets 1 primary row + n_aux aux-LR rows
    n_marg_rows = D * (1 + n_aux)
    height_ratios = [1.2, 1.2] + [0.8] + [0.5] * n_aux + [0.8] + [0.5] * n_aux
    n_total_rows = 2 + n_marg_rows

    if figsize is None:
        figsize = (5.5 * n_cols, 5.0 + 4.0 * D + 2.0 * D * n_aux)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_total_rows, n_cols, figure=fig,
                           height_ratios=height_ratios,
                           hspace=0.45, wspace=0.25)

    # Row 0 col 0: primary 2D point grid
    ax = fig.add_subplot(gs[0, 0])
    # Points rendered first (behind evidence colormap)
    for s_idx in range(min(S, 4) - 1, -1, -1):   # descending: 3→0 so P/LP on top
        mask = sa[:, s_idx] & complete
        if not mask.any(): continue
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   color=_sc[s_idx], s=14, alpha=0.7,
                   edgecolors=_se[s_idx], linewidths=0.5, zorder=1,
                   marker=_sm[s_idx])
    # Evidence colormap on top (semi-transparent so points show through)
    im = ax.pcolormesh(x1g, x2g, grid_points.T, cmap=POINT_CMAP,
                       norm=pt_norm, shading='auto', alpha=0.72, zorder=2)
    plt.colorbar(im, ax=ax, label='Evidence Points', shrink=0.8)
    ax.set_xlabel(dataset_names[0], fontsize=8); ax.set_ylabel(dataset_names[1], fontsize=8)
    ax.set_xlim(x1_range); ax.set_ylim(x2_range)
    ax.set_aspect('auto')
    ax.set_title(f'Point Regions\nprior={median_prior:.4f}', fontsize=9, fontweight='bold')
    ax.grid(lw=0.2, alpha=0.3, zorder=0)

    # Row 0 cols 1..n_aux: aux 2D point grids
    for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
        ax = fig.add_subplot(gs[0, 1 + ai])
        aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
        aux_name = (sample_names[fixed_idx] if fixed_idx < len(sample_names)
                    else f'Sample {fixed_idx}')
        aux_gp, aux_lr_con = _all_grids[1 + ai]
        # Points first, evidence on top
        for s_idx in range(min(S, 4) - 1, -1, -1):   # descending: 3→0
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
        _ar = analysis.results.get(config, {}).get('aux_results', {}).get(fixed_idx, {})
        _ap = _ar.get('median_prior', float('nan'))
        ax.set_title(f'Aux: {aux_name}\nprior={_ap:.4f}',
                     fontsize=9, fontweight='bold', color=aux_color)
        ax.grid(lw=0.2, alpha=0.3)

    # Fill any unused row-0 columns (density rows may need more cols)
    for c_idx in range(n_grid_cols, n_cols):
        fig.add_subplot(gs[0, c_idx]).axis('off')

    # ── Bottom legends (below all panels) ──────────────────────────────────
    # Sample handles — match actual scatter style
    sample_handles = [
        Line2D([0], [0], marker=_sm[s_idx],
               color=_se[s_idx],
               markerfacecolor=_sc[s_idx],
               markersize=9, linewidth=0, markeredgewidth=0.8,
               label=f"{sample_names[s_idx]} (n={int(sa[:, s_idx].sum())})")
        for s_idx in range(min(S, 4))
    ]

    # Evidence handles — only levels actually present across any panel, including 0
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
                                   colors=_sc[s_idx],
                                   linewidths=0.5, alpha=0.6)
                    outer = levels[1] if len(levels) > 1 else levels[0]
                    for bound, ls in [(np.maximum(d_mean - d_std, 0), ':'), (d_mean + d_std, ':')]:
                        ax.contour(x1g, x2g, bound.T, levels=[outer],
                                   colors=_sc[s_idx],
                                   linewidths=0.3, linestyles=ls, alpha=0.3)
                mask = sa[:, s_idx] & complete
                if mask.any():
                    ax.scatter(scores[mask, 0], scores[mask, 1],
                               c=_sc[s_idx], s=4, alpha=0.3,
                               edgecolors='none')
                _plot_component_means(ax, all_fits, s_idx)
                ax.set_xlim(x1_range); ax.set_ylim(x2_range)
                ax.set_xlabel(dataset_names[0], fontsize=7); ax.set_ylabel(dataset_names[1], fontsize=7)
                n_s = sa[:, s_idx].sum()
                ax.set_title(f'{sample_names[s_idx]} (n={n_s})', fontsize=8, fontweight='bold',
                             color=_sc[s_idx])
                ax.grid(lw=0.2, alpha=0.2)
        for c_idx in range(S, n_cols):
            fig.add_subplot(gs[1, c_idx]).axis('off')

        # Rows 2+: marginals — one primary row + n_aux aux rows per dimension
        _aux_results = analysis.results.get(config, {}).get('aux_results', {})
        for dim in range(D):
            base_row = 2 + dim * (1 + n_aux)
            md = marginal_data[dim]
            if md is None:
                for r_ in range(1 + n_aux):
                    for c in range(n_cols):
                        fig.add_subplot(gs[base_row + r_, c]).axis('off')
                continue
            _draw_marginal_row(fig, gs, base_row, dim, md, scores, sa, S, n_cols,
                               dataset_names, sample_names, tau_p_log, tau_b_log,
                               ylim_bound, path_pctile, ben_pctile, analysis,
                               sample_style=sample_style)
            for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
                aux_md_dim = (aux_marginal_data.get(fixed_idx) or {}).get(dim)
                aux_name = (sample_names[fixed_idx] if fixed_idx < len(sample_names)
                            else f'Sample {fixed_idx}')
                aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
                aux_r = _aux_results.get(fixed_idx, {})
                aux_tau_p = aux_r.get('tau_p_log', tau_p_log)
                aux_tau_b = aux_r.get('tau_b_log', tau_b_log)
                _draw_aux_lr_row(fig, gs, base_row + 1 + ai, dim,
                                 aux_md_dim, aux_name, aux_color,
                                 eff_idx, aux_tau_p, aux_tau_b,
                                 ylim_bound, path_pctile, ben_pctile,
                                 dataset_names, analysis, n_cols)

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
                aux_p_entries=None, aux_marginal_data=None, max_lr_pairs=10,
                sample_style=None, pairwise_grids_hd=None, pairwise_dim_pairs=None,
                first_row_only=False):
    """Layout for D>2.

    Row 0: pairwise LR+ grids — all C(D,2) combinations up to max_lr_pairs,
           followed by aux pairwise grids for each aux sample (dim 0 vs dim 1 only).
    Rows 1..D: per-dimension marginals — [S sample density panels] + [marginal LR+].
    """
    from itertools import combinations as _combinations
    aux_p_entries = aux_p_entries or []
    aux_marginal_data = aux_marginal_data or {}
    n_aux = len(aux_p_entries)
    _aux_results = analysis.results.get(config, {}).get('aux_results', {})
    _sc = sample_style['colors']  if sample_style else [SAMPLE_COLORS[i % len(SAMPLE_COLORS)] for i in range(S)]
    _se = sample_style['edges']   if sample_style else [_SAMPLE_EDGE_COLORS[i % len(_SAMPLE_EDGE_COLORS)] for i in range(S)]
    _sm = sample_style['markers'] if sample_style else [SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)] for i in range(S)]

    # Use precomputed pairs if provided, else fall back to all combinations
    if pairwise_dim_pairs is not None:
        all_dim_pairs = pairwise_dim_pairs[:max_lr_pairs]
    else:
        all_dim_pairs = list(_combinations(range(D), 2))[:max_lr_pairs]
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

    # ── first_row_only: rows=dim pairs, cols=samples ──────────────────────────
    if first_row_only:
        n_rows_f = n_pairs
        n_cols_f = S
        if figsize is None:
            figsize = (4.8 * n_cols_f, 5.0 * n_rows_f)
        fig = plt.figure(figsize=figsize, constrained_layout=False)
        gs = gridspec.GridSpec(n_rows_f, n_cols_f, figure=fig,
                               hspace=0.5, wspace=0.45,
                               left=0.08, right=0.97, top=0.93, bottom=0.22)

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
    n_cols  = max(max(S, 2) + 1, n_pairs + n_aux)
    # Each dimension: 1 primary row + n_aux aux-LR rows
    n_rows  = 1 + D * (1 + n_aux)
    height_ratios = [1.6] + ([0.9] + [0.5] * n_aux) * D
    if figsize is None:
        figsize = (max(4.5 * n_cols, 4.5 * (n_pairs + n_aux)),
                   4.0 + 3.0 * D + 1.5 * D * n_aux)
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
    # Rows 1+: per-dimension marginals — primary row + n_aux aux-LR rows each
    # ═══════════════════════════════════════
    for dim in range(D):
        base_row = 1 + dim * (1 + n_aux)
        md = marginal_data[dim]
        if md is None:
            for r_ in range(1 + n_aux):
                for c in range(n_cols):
                    fig.add_subplot(gs[base_row + r_, c]).axis('off')
            continue
        _draw_marginal_row(fig, gs, base_row, dim, md, scores, sa, S, n_cols,
                           dataset_names, sample_names, tau_p_log, tau_b_log,
                           ylim_bound, path_pctile, ben_pctile, analysis,
                           sample_style=sample_style)
        for ai, (fixed_idx, eff_idx) in enumerate(aux_p_entries):
            aux_md_dim = (aux_marginal_data.get(fixed_idx) or {}).get(dim)
            aux_name = (sample_names[fixed_idx] if fixed_idx < len(sample_names)
                        else f'Sample {fixed_idx}')
            aux_color = _AUX_COLORS[ai % len(_AUX_COLORS)]
            aux_r = _aux_results.get(fixed_idx, {})
            aux_tau_p = aux_r.get('tau_p_log', tau_p_log)
            aux_tau_b = aux_r.get('tau_b_log', tau_b_log)
            _draw_aux_lr_row(fig, gs, base_row + 1 + ai, dim,
                             aux_md_dim, aux_name, aux_color,
                             eff_idx, aux_tau_p, aux_tau_b,
                             ylim_bound, path_pctile, ben_pctile,
                             dataset_names, analysis, n_cols)

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