"""
Visualize multivariate calibration fit using bootstrap-percentile densities
and pre-computed evidence assignments from MVCalibrationAnalysis.run().

Supports both restricted MSN (q=1, Delta is a p-vector) and
CFUSN (q>=1, Delta is a p×q matrix) parameterizations.
Detection is automatic based on Delta shape in component params.

All evidence points displayed come directly from analysis.results[config]['points'].
"""

import numpy as np
import matplotlib.pyplot as plt
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


def _eval_all_marginals(fit, x_2d_list, p_idx, b_idx, s_idx, benign_method, S):
    """Evaluate marginal LR+ and sample densities for one bootstrap across all dimensions.

    x_2d_list : list of D arrays, each (n_grid, total_dims) with one column set.
    Returns   : list of D results, each the output of _eval_marginal_fit.
    """
    return [
        _eval_marginal_fit(fit, x_2d, p_idx, b_idx, s_idx, benign_method, S)
        for x_2d in x_2d_list
    ]


def _eval_fit_on_grid(fit, grid_pts, p_idx, b_idx, s_idx, benign_method):
    """Evaluate LR+ for one bootstrap fit on a pre-built grid of points."""
    params = fit['component_params']
    weights = fit['weights']
    K = len(params)

    w_p = weights[p_idx]
    s_valid = s_idx is not None and s_idx < len(weights)
    if s_valid and benign_method == 'synonymous':
        w_b = weights[s_idx]
    elif s_valid and benign_method == 'avg':
        w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
    else:
        w_b = weights[b_idx]

    log_fp = logsumexp(
        [np.log(w_p[c] + 1e-300) + _sn_logpdf(grid_pts, *params[c]) for c in range(K)],
        axis=0,
    )
    log_fb = logsumexp(
        [np.log(w_b[c] + 1e-300) + _sn_logpdf(grid_pts, *params[c]) for c in range(K)],
        axis=0,
    )
    return log_fp - log_fb


def _eval_fit_on_grid_batch_p(fit, grid_pts, p_indices, b_idx, s_idx, benign_method):
    """Evaluate LR+ on a grid for multiple pathogenic indices in one shot.

    Component log-densities (_sn_logpdf) are computed ONCE and shared across
    all p_indices, avoiding redundant density evaluations.

    Returns list of len(p_indices) flat lr arrays.
    """
    params = fit['component_params']
    weights = fit['weights']
    K = len(params)
    n_w = len(weights)

    # Component densities computed once, shared across all p_indices
    comp_log = [_sn_logpdf(grid_pts, *params[c]) for c in range(K)]

    s_valid = s_idx is not None and s_idx < n_w
    if s_valid and benign_method == 'synonymous':
        w_b = weights[s_idx]
    elif s_valid and benign_method == 'avg':
        w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
    else:
        w_b = weights[b_idx]
    log_fb = logsumexp([np.log(w_b[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)

    out = []
    for p_idx in p_indices:
        if p_idx >= n_w:
            out.append(np.full(len(grid_pts), np.nan))
            continue
        w_p = weights[p_idx]
        log_fp = logsumexp([np.log(w_p[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)
        out.append(log_fp - log_fb)
    return out


def _eval_all_marginals_batch_p(fit, x_2d_list, p_indices, b_idx, s_idx, benign_method, S):
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
    if s_valid and benign_method == 'synonymous':
        w_b = weights[s_idx]
    elif s_valid and benign_method == 'avg':
        w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
    else:
        w_b = weights[b_idx]

    w_ps = [weights[p_idx] if p_idx < n_w else None for p_idx in p_indices]
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

        log_fb = logsumexp([np.log(w_b[c] + 1e-300) + comp_log[c] for c in range(K)], axis=0)

        for pi, w_p in enumerate(w_ps):
            if w_p is None:
                results_per_p[pi].append((np.full(len(x_2d), np.nan), sample_logs, comp_logs_per_sample))
                continue
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


def _eval_marginal_fit(fit, x_2d, p_idx, b_idx, s_idx, benign_method, S):
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

    w_p = weights[p_idx]
    s_valid = s_idx is not None and s_idx < len(weights)
    if s_valid and benign_method == 'synonymous':
        w_b = weights[s_idx]
    elif s_valid and benign_method == 'avg':
        w_b = (np.array(weights[b_idx]) + np.array(weights[s_idx])) / 2
    else:
        w_b = weights[b_idx]

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
    benign_method = analysis.benign_method

    lr_all = Parallel(n_jobs=-1)(
        delayed(_eval_fit_on_grid)(fit, grid_pts, p_idx, b_idx, s_idx, benign_method)
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
        delayed(_eval_all_marginals)(fit, x_2d_list, p_idx, b_idx, s_idx, benign_method, S)
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
                                    path_pctile, ben_pctile):
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

    all_results = Parallel(n_jobs=-1)(
        delayed(_eval_all_marginals_batch_p)(
            fit, x_2d_list, p_indices, b_idx, s_idx, benign_method, S)
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
                                  b_idx, s_idx, benign_method, p_configs):
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
        delayed(_eval_fit_on_grid_batch_p)(fit, grid_pts, p_indices, b_idx, s_idx, benign_method)
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

def plot_mv_calibration(analysis, config, figsize=None, n_grid=120,
                        contour_levels=6, first_row_only=False, max_lr_pairs=10):
    """
    Multivariate calibration visualization. Layout adapts to dimensionality.

    D=2: 2D point-region grid + per-sample density contours + marginals
    D>2: LR+ distribution violin + per-dimension marginals for all D dims
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
    _sn_raw = getattr(ms, 'sample_names', None) or SAMPLE_NAMES_DEFAULT
    sample_names = [_sn_raw[i] if i < len(_sn_raw) else f'Sample {i}' for i in range(S)]

    latent_q   = r.get('latent_q', getattr(analysis, '_latent_q', 1))
    model_label = f"CFUSN q={latent_q}" if latent_q > 1 else "MSN q=1"

    points      = r['points']
    tau_p_log   = r['tau_p_log']
    tau_b_log   = r['tau_b_log']
    median_prior = r['median_prior']
    C_path      = r.get('C_path', r.get('C', '?'))
    C_ben       = r.get('C_ben', '?')
    path_pctile = r.get('path_percentile', 5)
    ben_pctile  = r.get('ben_percentile', 95)
    max_pt      = max(analysis.point_values)
    pt_norm     = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)
    ylim_bound  = max(abs(tau_p_log[-1]), abs(tau_b_log[-1]))
    missing_frac = 1.0 - (~np.isnan(scores).any(axis=1)).mean()
    pad = 0.5

    # ── Marginals (all dims, single parallel sweep) ──
    x_grids = [
        np.linspace(np.nanmin(scores[:, d]) - pad, np.nanmax(scores[:, d]) + pad, 500)
        for d in range(D)
    ]
    print(f"  Collecting bootstrap fits...")
    all_fits     = _collect_valid_fits(analysis, config)
    n_boots_used = len(all_fits)

    # Auxiliary pathogenic marginals
    aux_p_entries = getattr(analysis, 'aux_p_entries', [])

    # Marginals — skip entirely when only the top row is needed
    all_p_indices = [analysis.p_idx] + [eff for _, eff in aux_p_entries]
    r_cfg = analysis.results[config]
    path_pctile = r_cfg.get('path_percentile', 5)
    ben_pctile  = 100 - path_pctile
    if first_row_only:
        marginal_data   = {d: None for d in range(D)}
        aux_marginal_data = {fixed_idx: {d: None for d in range(D)}
                             for fixed_idx, _ in aux_p_entries}
    else:
        print(f"  Computing marginals for {len(all_p_indices)} pathogenic index(es) "
              f"× {D} dims ({n_boots_used} boots)...")
        all_marginals = _compute_all_marginals_batch_p(
            all_fits, x_grids, all_p_indices,
            analysis.b_idx, getattr(analysis, 's_idx', None),
            analysis.benign_method,
            ms.sample_assignments.shape[1],
            path_pctile, ben_pctile,
        )
        marginal_data = {d: all_marginals[0][d] for d in range(D)}
        aux_marginal_data = {}
        for ai, (fixed_idx, _) in enumerate(aux_p_entries):
            aux_marginal_data[fixed_idx] = {d: all_marginals[ai + 1][d] for d in range(D)}

    gene = getattr(ms, 'scoreset_name', '')
    suptitle = (
        f'{gene} — {config}\n'
        f'{n_boots_used} bootstraps, prior={median_prior:.4f}, '
        f'missing={missing_frac*100:.1f}%'
    )

    info = {'marginal_data': marginal_data, 'n_boots_used': n_boots_used, 'latent_q': latent_q}

    if D == 2:
        fig, info = _plot_mv_2d(
            analysis, config, all_fits, marginal_data, x_grids,
            scores, sa, N, D, S, dataset_names, sample_names,
            points, tau_p_log, tau_b_log, median_prior, C_path, C_ben,
            path_pctile, ben_pctile, max_pt, pt_norm, ylim_bound, missing_frac,
            model_label, n_boots_used, pad, n_grid, contour_levels,
            figsize, suptitle,
            aux_p_entries=aux_p_entries, aux_marginal_data=aux_marginal_data,
            first_row_only=first_row_only,
        )
    else:
        fig, info = _plot_mv_hd(
            analysis, config, all_fits, marginal_data, x_grids,
            scores, sa, N, D, S, dataset_names, sample_names,
            points, tau_p_log, tau_b_log, median_prior, C_path, C_ben,
            path_pctile, ben_pctile, max_pt, pt_norm, ylim_bound,
            model_label, n_boots_used, pad, n_grid, contour_levels,
            figsize, suptitle,
            aux_p_entries=aux_p_entries, aux_marginal_data=aux_marginal_data,
            max_lr_pairs=max_lr_pairs,
        )

    return fig, info


_AUX_COLORS = ['#e6a817', '#8B4513', '#006400', '#800080', '#FF6600', '#4B0082']


def _draw_marginal_row(fig, gs, row, dim, md, scores, sa, S, n_cols,
                       dataset_names, sample_names, tau_p_log, tau_b_log,
                       ylim_bound, path_pctile, ben_pctile, analysis):
    """Draw one row: S sample density panels + primary pathogenic LR+ panel."""
    x_marg = md['x']

    for s_idx in range(min(S, n_cols - 1)):
        ax = fig.add_subplot(gs[row, s_idx])
        color = SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)]

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


def _plot_mv_2d(analysis, config, all_fits, marginal_data, x_grids,
                scores, sa, N, D, S, dataset_names, sample_names,
                points, tau_p_log, tau_b_log, median_prior, C_path, C_ben,
                path_pctile, ben_pctile, max_pt, pt_norm, ylim_bound, missing_frac,
                model_label, n_boots_used, pad, n_grid, contour_levels,
                figsize, suptitle,
                aux_p_entries=None, aux_marginal_data=None,
                first_row_only=False):
    """Layout for D=2: 2D grid, density contours, marginals."""
    aux_p_entries = aux_p_entries or []
    aux_marginal_data = aux_marginal_data or {}
    n_aux = len(aux_p_entries)

    x1g = np.linspace(np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad, n_grid)
    x2g = np.linspace(np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad, n_grid)
    x1_range = (x1g[0], x1g[-1])
    x2_range = (x2g[0], x2g[-1])
    complete = ~np.isnan(scores).any(axis=1)

    # Build p_configs for primary + all aux in one batch
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
        _ar = _aux_res.get(eff_idx, {})   # keyed by eff_idx in aux_results
        # aux_results is keyed by fixed_idx; find it
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
    )
    grid_points, lr_conservative = _all_grids[0]

    # Row 0: primary grid + aux grids (no separate legend column — legend goes below)
    n_grid_cols = 1 + n_aux
    n_cols = n_grid_cols if first_row_only else max(S, n_grid_cols)

    # Each of D=2 dimensions gets 1 primary row + n_aux aux-LR rows
    n_marg_rows = D * (1 + n_aux)
    if first_row_only:
        height_ratios = [1.2]
        n_total_rows = 1
    else:
        height_ratios = [1.2, 1.2] + [0.8] + [0.5] * n_aux + [0.8] + [0.5] * n_aux
        n_total_rows = 2 + n_marg_rows

    if figsize is None:
        if first_row_only:
            figsize = (5.5 * n_grid_cols, 6.5)   # extra vertical room for bottom legend
        else:
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
                   color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                   s=14, alpha=0.7,
                   edgecolors=_SAMPLE_EDGE_COLORS[s_idx % len(_SAMPLE_EDGE_COLORS)],
                   linewidths=0.5, zorder=1,
                   marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)])
    # Evidence colormap on top (semi-transparent so points show through)
    im = ax.pcolormesh(x1g, x2g, grid_points.T, cmap=POINT_CMAP,
                       norm=pt_norm, shading='auto', alpha=0.72, zorder=2)
    plt.colorbar(im, ax=ax, label='Evidence Points', shrink=0.8)
    ax.contour(x1g, x2g, lr_conservative.T, levels=[0], colors='black', linewidths=1, zorder=3)
    ax.set_xlabel(dataset_names[0], fontsize=8); ax.set_ylabel(dataset_names[1], fontsize=8)
    ax.set_xlim(x1_range); ax.set_ylim(x2_range)
    ax.set_aspect('equal', adjustable='box')
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
                       color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                       s=14, alpha=0.7,
                       edgecolors=_SAMPLE_EDGE_COLORS[s_idx % len(_SAMPLE_EDGE_COLORS)],
                       linewidths=0.5, zorder=1,
                       marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)])
        im2 = ax.pcolormesh(x1g, x2g, aux_gp.T, cmap=POINT_CMAP,
                            norm=pt_norm, shading='auto', alpha=0.72, zorder=2)
        plt.colorbar(im2, ax=ax, label='Evidence Points', shrink=0.8)
        ax.contour(x1g, x2g, aux_lr_con.T, levels=[0], colors='black', linewidths=1, zorder=3)
        ax.set_xlabel(dataset_names[0], fontsize=8); ax.set_ylabel(dataset_names[1], fontsize=8)
        ax.set_xlim(x1_range); ax.set_ylim(x2_range)
        ax.set_aspect('equal', adjustable='box')
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
        Line2D([0], [0], marker=SAMPLE_MARKERS[s_idx],
               color=_SAMPLE_EDGE_COLORS[s_idx % len(_SAMPLE_EDGE_COLORS)],
               markerfacecolor=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
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
                d = _compute_sample_density_grid(all_fits, s_idx, x1g, x2g)
                if d is not None:
                    d_mean, d_std = d['mean'], d['std']
                    levels = np.linspace(d_mean.max() * 0.01, d_mean.max() * 0.95, contour_levels)
                    cmap_name = 'Greens' if s_idx >= 2 else ('Reds' if s_idx == 0 else 'Blues')
                    if levels[-1] > levels[0]:
                        ax.contourf(x1g, x2g, d_mean.T, levels=levels, cmap=cmap_name, alpha=0.4)
                        ax.contour(x1g, x2g, d_mean.T, levels=levels,
                                   colors=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                                   linewidths=0.5, alpha=0.6)
                    outer = levels[1] if len(levels) > 1 else levels[0]
                    for bound, ls in [(np.maximum(d_mean - d_std, 0), ':'), (d_mean + d_std, ':')]:
                        ax.contour(x1g, x2g, bound.T, levels=[outer],
                                   colors=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                                   linewidths=0.3, linestyles=ls, alpha=0.3)
                mask = sa[:, s_idx] & complete
                if mask.any():
                    ax.scatter(scores[mask, 0], scores[mask, 1],
                               c=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)], s=4, alpha=0.3,
                               edgecolors='none')
                _plot_component_means(ax, all_fits, s_idx)
                ax.set_xlim(x1_range); ax.set_ylim(x2_range)
                ax.set_xlabel(dataset_names[0], fontsize=7); ax.set_ylabel(dataset_names[1], fontsize=7)
                n_s = sa[:, s_idx].sum()
                ax.set_title(f'{sample_names[s_idx]} (n={n_s})', fontsize=8, fontweight='bold',
                             color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)])
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
                               ylim_bound, path_pctile, ben_pctile, analysis)
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


def _plot_mv_hd(analysis, config, all_fits, marginal_data, x_grids,
                scores, sa, N, D, S, dataset_names, sample_names,
                points, tau_p_log, tau_b_log, median_prior, C_path, C_ben,
                path_pctile, ben_pctile, max_pt, pt_norm, ylim_bound,
                model_label, n_boots_used, pad, n_grid, contour_levels,
                figsize, suptitle,
                aux_p_entries=None, aux_marginal_data=None, max_lr_pairs=10):
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

    all_dim_pairs = list(_combinations(range(D), 2))[:max_lr_pairs]
    n_pairs = len(all_dim_pairs)
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

    complete = ~np.isnan(scores).any(axis=1)

    # ═══════════════════════════════════════
    # Row 0: primary pairwise grids then aux grids (dim 0 vs dim 1 for each aux)
    # ═══════════════════════════════════════
    for k, (dim_i, dim_j) in enumerate(all_dim_pairs):
        xig = x_grids[dim_i]
        xjg = x_grids[dim_j]
        xi_range = (xig[0], xig[-1])
        xj_range = (xjg[0], xjg[-1])

        print(f"  Computing LR+ grid: dim {dim_i} vs dim {dim_j}...")
        grid_points, lr_conservative = _compute_conservative_lr_grid(
            analysis, config, all_fits, xig, xjg,
            dim_i=dim_i, dim_j=dim_j, total_dims=D,
        )

        ax = fig.add_subplot(gs[0, k])
        im = ax.pcolormesh(xig, xjg, grid_points.T, cmap=POINT_CMAP,
                           norm=pt_norm, shading='auto', alpha=0.7)
        plt.colorbar(im, ax=ax, label='Points', shrink=0.8)
        ax.contour(xig, xjg, lr_conservative.T, levels=[0],
                   colors='black', linewidths=0.8)

        for s_idx in range(S):
            mask = sa[:, s_idx] & complete
            if not mask.any():
                continue
            ax.scatter(scores[mask, dim_i], scores[mask, dim_j],
                       c=points[mask], cmap=POINT_CMAP, norm=pt_norm,
                       s=8, alpha=0.5,
                       edgecolors=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                       linewidths=0.3,
                       marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)])

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
            ax.contour(xig, xjg, aux_lr_con.T, levels=[0], colors='black', linewidths=0.8)
            aux_r = analysis.results[config].get('aux_results', {}).get(fixed_idx, {})
            aux_pts = aux_r.get('points', points)
            for s_idx in range(S):
                mask = sa[:, s_idx] & complete
                if not mask.any(): continue
                ax.scatter(scores[mask, 0], scores[mask, 1],
                           c=aux_pts[mask], cmap=POINT_CMAP, norm=pt_norm,
                           s=8, alpha=0.5,
                           edgecolors=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                           linewidths=0.3,
                           marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)])
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
                           ylim_bound, path_pctile, ben_pctile, analysis)
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