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


# ──────────────────────────────────────
# Shared constants
# ──────────────────────────────────────
SAMPLE_COLORS = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
SAMPLE_NAMES_DEFAULT = ['Pathogenic/Likely Pathogenic', 'Benign/Likely Benign',
                        'population', 'Synonymous']
SAMPLE_MARKERS = ['o', 's', '^', 'D']

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

def _collect_valid_fits(analysis, config):
    """Collect all valid reconstituted fits for a config."""
    all_fits = []
    for boot_key, boot_data in analysis.raw_boots.items():
        fit_raw = boot_data.get(config)
        if fit_raw is None:
            continue
        inner = fit_raw.get('fit', fit_raw)
        fit = analysis._reconstitute_params(inner)
        if fit is not None:
            all_fits.append(fit)
    return all_fits


def _eval_fit_on_grid(fit, grid_pts, p_idx, b_idx, s_idx, benign_method):
    """Evaluate LR+ for one bootstrap fit on a pre-built grid of points.

    Module-level so joblib loky workers can pickle it by name.
    Returns (lr_flat,) — 1-D array of length n_grid_pts.
    """
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


def _compute_conservative_lr_grid(analysis, config, all_fits, x1g, x2g):
    """
    Compute conservative discrete point grid from bootstrap LR+ percentiles.

    For each grid cell:
      - Compute LR+ across all bootstraps
      - Use path_percentile (e.g. 5th) for positive LR+ → pathogenic evidence
      - Use ben_percentile (e.g. 95th) for negative LR+ → benign evidence
      - Convert to discrete points using pre-computed thresholds

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

    X1, X2 = np.meshgrid(x1g, x2g, indexing='ij')
    grid_pts = np.column_stack([X1.ravel(), X2.ravel()])
    n_grid = len(grid_pts)
    grid_shape = (len(x1g), len(x2g))

    p_idx = analysis.p_idx
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


def _compute_sample_density_grid(all_fits, s_idx, x1g, x2g):
    """Compute mean density for sample s_idx on a 2D grid."""
    X1, X2 = np.meshgrid(x1g, x2g, indexing='ij')
    grid_pts = np.column_stack([X1.ravel(), X2.ravel()])
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


# ──────────────────────────────────────
# Main plot function
# ──────────────────────────────────────

def plot_mv_calibration(analysis, config, figsize=None, n_grid=120,
                        contour_levels=6):
    """
    Comprehensive multivariate calibration visualization.

    Supports both restricted MSN (q=1) and CFUSN (q>1) models.
    All evidence points come from analysis.results[config]['points'].

    Layout:
      Row 0: [Observations by evidence (2D)]  [Observations by sample]
      Row 1: [Per-sample 2D density contours] × S
      Row 2: [Per-sample marginal dim0 + components] × S  [Marginal LR+ dim0]
      Row 3: [Per-sample marginal dim1 + components] × S  [Marginal LR+ dim1]
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

    # Detect model type for display
    latent_q = r.get('latent_q', getattr(analysis, '_latent_q', 1))
    model_label = f"CFUSN q={latent_q}" if latent_q > 1 else "MSN q=1"

    # ── Pre-computed results (authoritative) ──
    points = r['points']
    tau_p_log = r['tau_p_log']
    tau_b_log = r['tau_b_log']
    median_prior = r['median_prior']
    C_path = r.get('C_path', r.get('C', '?'))
    C_ben = r.get('C_ben', '?')
    path_pctile = r.get('path_percentile', 5)
    ben_pctile = r.get('ben_percentile', 95)
    max_pt = max(analysis.point_values)
    pt_norm = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)
    n_valid = r['n_valid']
    ylim_bound = max(abs(tau_p_log[-1]), abs(tau_b_log[-1])) * 1.05

    complete = ~np.isnan(scores).any(axis=1)
    missing_frac = 1.0 - complete.mean()

    # Grid ranges
    pad = 0.5
    x1_range = (np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad)
    x2_range = (np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad)
    x1g = np.linspace(*x1_range, n_grid)
    x2g = np.linspace(*x2_range, n_grid)

    # Collect bootstrap fits
    print(f"  Collecting bootstrap fits...")
    all_fits = _collect_valid_fits(analysis, config)
    n_boots_used = len(all_fits)

    # Compute conservative 2D point grid
    print(f"  Computing conservative LR+ grid ({n_grid}×{n_grid}, "
          f"{n_boots_used} boots, {model_label})...")
    grid_points, lr_conservative = _compute_conservative_lr_grid(
        analysis, config, all_fits, x1g, x2g)

    # Marginals with percentile LR+
    marginal_data = {}
    for dim in range(min(D, 2)):
        x_marg = np.linspace(
            np.nanmin(scores[:, dim]) - pad,
            np.nanmax(scores[:, dim]) + pad, 500)
        lr_pctiles, sm, cm, nm = _compute_bootstrap_marginal_lr(
            analysis, config, dim, x_marg)
        marginal_data[dim] = {
            'x': x_marg, 'lr': lr_pctiles, 'sample': sm,
            'components': cm, 'n': nm,
        }

    # ── Figure layout ──
    n_cols = max(S, 2) + 1
    if figsize is None:
        figsize = (4.5 * n_cols, 16)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, n_cols, figure=fig,
                           height_ratios=[1.2, 1.2, 0.8, 0.8],
                           hspace=0.45, wspace=0.35)

    # ═══════════════════════════════════════
    # Row 0, Col 0: Conservative point regions + observations
    # ═══════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0])
    im = ax.pcolormesh(x1g, x2g, grid_points.T, cmap=POINT_CMAP,
                       norm=pt_norm, shading='auto', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Evidence Points', shrink=0.8)
    ax.contour(x1g, x2g, lr_conservative.T, levels=[0], colors='black',
               linewidths=1, linestyles='-')
    for s_idx in range(S):
        mask = sa[:, s_idx] & complete
        if not mask.any():
            continue
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   c=points[mask], cmap=POINT_CMAP, norm=pt_norm,
                   s=10, alpha=0.6,
                   edgecolors=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                   linewidths=0.3,
                   marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)])
    ax.set_xlabel(dataset_names[0], fontsize=8)
    ax.set_ylabel(dataset_names[1], fontsize=8)
    ax.set_xlim(x1_range); ax.set_ylim(x2_range)
    ax.set_title(f'Point Regions ({model_label})\nprior={median_prior:.4f}, '
                 f'C_p={C_path:.1f}, C_b={C_ben:.1f}',
                 fontsize=9, fontweight='bold')
    ax.grid(lw=0.2, alpha=0.3)

    # ═══════════════════════════════════════
    # Row 0, Col 1: Observations colored by sample
    # ═══════════════════════════════════════
    ax = fig.add_subplot(gs[0, 1])
    for s_idx in range(S):
        mask = sa[:, s_idx] & complete
        if not mask.any():
            continue
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   c=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)], s=8, alpha=0.4,
                   edgecolors='none',
                   marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)],
                   label=f"{sample_names[s_idx]} ({mask.sum()})")
    ax.legend(fontsize=5, framealpha=0.6, loc='upper right')
    ax.set_xlabel(dataset_names[0], fontsize=8)
    ax.set_ylabel(dataset_names[1], fontsize=8)
    ax.set_xlim(x1_range); ax.set_ylim(x2_range)
    ax.set_title('Observations by Sample', fontsize=9, fontweight='bold')
    ax.grid(lw=0.2, alpha=0.3)

    for c_idx in range(2, n_cols):
        fig.add_subplot(gs[0, c_idx]).axis('off')

    # ═══════════════════════════════════════
    # Row 1: Per-sample 2D density contours
    # ═══════════════════════════════════════
    if all_fits:
        for s_idx in range(min(S, n_cols)):
            ax = fig.add_subplot(gs[1, s_idx])
            d = _compute_sample_density_grid(all_fits, s_idx, x1g, x2g)

            if d is not None:
                d_mean = d['mean']
                d_std = d['std']

                levels = np.linspace(d_mean.max() * 0.01, d_mean.max() * 0.95,
                                     contour_levels)
                cmap_name = 'Greens' if s_idx >= 2 else ('Reds' if s_idx == 0 else 'Blues')
                if levels[-1] > levels[0]:
                    ax.contourf(x1g, x2g, d_mean.T, levels=levels,
                                cmap=cmap_name, alpha=0.4)
                    ax.contour(x1g, x2g, d_mean.T, levels=levels,
                               colors=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                               linewidths=0.5, alpha=0.6)

                outer = levels[1] if len(levels) > 1 else levels[0]
                for bound, ls in [(np.maximum(d_mean - d_std, 0), ':'),
                                  (d_mean + d_std, ':')]:
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
            ax.set_xlabel(dataset_names[0], fontsize=7)
            ax.set_ylabel(dataset_names[1], fontsize=7)
            n_s = sa[:, s_idx].sum()
            ax.set_title(f'{sample_names[s_idx]} (n={n_s})',
                         fontsize=8, fontweight='bold',
                         color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)])
            ax.grid(lw=0.2, alpha=0.2)

    for c_idx in range(S, n_cols):
        fig.add_subplot(gs[1, c_idx]).axis('off')

    # ═══════════════════════════════════════
    # Rows 2–3: Marginal densities + LR+ percentiles
    # ═══════════════════════════════════════
    for dim in range(min(D, 2)):
        row = 2 + dim
        md = marginal_data[dim]
        x_marg = md['x']

        for s_idx in range(min(S, n_cols - 1)):
            ax = fig.add_subplot(gs[row, s_idx])

            s_data = md['sample'][s_idx] if md['sample'] is not None else None
            if s_data is not None:
                total_mean = s_data['mean']
                total_std = s_data['std']
                ax.plot(x_marg, total_mean,
                        color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                        lw=1.5, zorder=3)
                ax.fill_between(x_marg,
                                np.maximum(total_mean - total_std, 0),
                                total_mean + total_std,
                                color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                                alpha=0.08)

                c_data = (md['components'][s_idx]
                          if md['components'] is not None else None)
                if c_data is not None:
                    n_comp = len(c_data)
                    comp_colors = plt.cm.Set2(np.linspace(0, 1, max(n_comp, 3)))
                    for c in range(n_comp):
                        c_mean = c_data[c]['mean']
                        if c_mean.max() < total_mean.max() * 0.005:
                            continue
                        ax.plot(x_marg, c_mean, color=comp_colors[c],
                                lw=0.8, ls='--', alpha=0.7)

            obs_dim = scores[sa[:, s_idx], dim]
            obs_dim = obs_dim[~np.isnan(obs_dim)]
            if len(obs_dim) > 0:
                ax.hist(obs_dim, bins=40, density=True, alpha=0.2,
                        color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)], edgecolor='white',
                        linewidth=0.3)

            n_s_dim = (~np.isnan(scores[sa[:, s_idx], dim])).sum()
            ax.set_xlabel(dataset_names[dim], fontsize=7)
            ax.set_ylabel('Density', fontsize=7)
            ax.set_title(f'{sample_names[s_idx]} — {dataset_names[dim]} (n={n_s_dim})',
                         fontsize=7, fontweight='bold',
                         color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)])
            ax.grid(lw=0.2, alpha=0.2)

        # ── Last column: Marginal LR+ percentiles ──
        ax = fig.add_subplot(gs[row, n_cols - 1])
        if md['lr'] is not None:
            p5 = md['lr']['p5']
            p50 = md['lr']['p50']
            p95 = md['lr']['p95']

            ax.plot(x_marg, p50, color='black', lw=1.5,
                    label=f'Median LR+')
            ax.plot(x_marg, p5, color='#d7191c', lw=1.0,
                    label=f'{path_pctile}th pctile (path)')
            ax.plot(x_marg, p95, color='#2c7bb6', lw=1.0,
                    label=f'{ben_pctile}th pctile (ben)')

            for pv in analysis.point_values:
                alpha = 0.15 + 0.05 * pv
                ax.axhline(tau_p_log[pv - 1], color='red', ls=':',
                           lw=0.4, alpha=min(alpha, 0.5))
                ax.axhline(tau_b_log[pv - 1], color='blue', ls=':',
                           lw=0.4, alpha=min(alpha, 0.5))
            for pv in [1, 4, max_pt]:
                if pv - 1 < len(tau_p_log):
                    ax.text(x_marg[0], tau_p_log[pv - 1], f' +{pv}',
                            fontsize=5, color='red', va='bottom')
                    ax.text(x_marg[0], tau_b_log[pv - 1], f' -{pv}',
                            fontsize=5, color='blue', va='top')

            ax.axhline(0, color='gray', lw=0.8, alpha=0.5)

            obs_p = scores[sa[:, analysis.p_idx], dim]
            obs_p = obs_p[~np.isnan(obs_p)]
            if len(obs_p) > 0:
                ax.plot(obs_p, np.full(len(obs_p), ylim_bound * 0.97),
                        '|', color='#CA7682', alpha=0.3, ms=3, mew=0.3)
            # Use benign if present, fall back to synonymous for bottom rug ticks
            neg_rug_idx = analysis.b_idx if analysis.b_idx is not None else analysis.s_idx
            if neg_rug_idx is not None:
                obs_b = scores[sa[:, neg_rug_idx], dim]
                obs_b = obs_b[~np.isnan(obs_b)]
                if len(obs_b) > 0:
                    ax.plot(obs_b, np.full(len(obs_b), -ylim_bound * 0.97),
                            '|', color='#1D7AAB', alpha=0.3, ms=3, mew=0.3)

            ax.set_ylim(-ylim_bound, ylim_bound)

        ax.set_xlabel(dataset_names[dim], fontsize=7)
        ax.set_ylabel('log LR+', fontsize=7)
        ax.set_title(f'Marginal LR+ — {dataset_names[dim]}',
                     fontsize=8, fontweight='bold')
        ax.legend(fontsize=5, framealpha=0.6)
        ax.grid(lw=0.2, alpha=0.2)

    # ── Suptitle ──
    gene = getattr(ms, 'scoreset_name', '')
    fig.suptitle(
        f'{gene} — {config} ({model_label})\n'
        f'{n_boots_used} bootstraps, '
        f'prior={median_prior:.4f}, '
        f'C_p={C_path:.1f}, C_b={C_ben:.1f}, '
        f'missing={missing_frac*100:.1f}%',
        fontsize=11, fontweight='bold', y=1.02)

    info = {
        'marginal_data': marginal_data,
        'x1g': x1g, 'x2g': x2g,
        'grid_points': grid_points,
        'lr_conservative': lr_conservative,
        'n_boots_used': n_boots_used,
        'latent_q': latent_q,
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