import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal as mvn, norm

def msn_logpdf_grid(x1, x2, mu, Delta, Gamma):
    """Evaluate MSN log-density on a meshgrid."""
    mu, Delta, Gamma = np.asarray(mu,float), np.asarray(Delta,float), np.asarray(Gamma,float)
    pts = np.column_stack([x1.ravel(), x2.ravel()])
    Omega = Gamma + np.outer(Delta, Delta)
    Omega = 0.5 * (Omega + Omega.T)
    eigv = np.linalg.eigvalsh(Omega)
    if eigv.min() < 1e-10:
        Omega += (1e-10 - eigv.min() + 1e-10) * np.eye(2)
    log_phi = mvn(mean=mu, cov=Omega, allow_singular=True).logpdf(pts)
    Oid = np.linalg.solve(Omega, Delta)
    eta = (pts - mu) @ Oid
    s2 = max(1.0 - Delta @ Oid, 1e-12)
    log_Phi = norm.logcdf(eta / np.sqrt(s2))
    return (np.log(2) + log_phi + log_Phi).reshape(x1.shape)


def plot_mv_fit(ms, fit_result, figsize=(16, 10)):
    """
    Visualize a 2D skew-normal mixture fit.
    
    Parameters
    ----------
    ms : MultiScoreset
    fit_result : dict with 'component_params' and 'weights'
    """
    params = fit_result['component_params']
    weights = fit_result['weights']
    scores = ms.scores  # (N, 2) with NaN
    K = len(params)
    
    sample_names = ms.sample_names
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
    sample_markers = ['o', 's', '^', 'D']
    
    # Grid for contours
    pad = 0.5
    x1_range = (np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad)
    x2_range = (np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad)
    x1g = np.linspace(*x1_range, 200)
    x2g = np.linspace(*x2_range, 200)
    X1, X2 = np.meshgrid(x1g, x2g)
    
    # ─── Figure 1: Data + mixture contours per sample ───
    n_samples = weights.shape[0]
    fig, axes = plt.subplots(1, n_samples + 1, figsize=figsize, 
                              gridspec_kw={'width_ratios': [1]*n_samples + [1]})
    
    for s_idx in range(n_samples):
        ax = axes[s_idx]
        w_s = weights[s_idx]
        
        # Mixture density for this sample
        log_mix = None
        for c in range(K):
            lp = msn_logpdf_grid(X1, X2, *params[c])
            weighted = np.log(w_s[c] + 1e-300) + lp
            if log_mix is None:
                log_mix = weighted
            else:
                # logsumexp
                mx = np.maximum(log_mix, weighted)
                log_mix = mx + np.log(np.exp(log_mix - mx) + np.exp(weighted - mx))
        
        # Contour plot
        levels = np.linspace(np.nanpercentile(log_mix[np.isfinite(log_mix)], 5),
                             np.nanmax(log_mix[np.isfinite(log_mix)]), 15)
        ax.contourf(X1, X2, log_mix, levels=levels, cmap='Blues', alpha=0.4)
        ax.contour(X1, X2, log_mix, levels=levels, colors='steelblue', 
                   linewidths=0.5, alpha=0.6)
        
        # Scatter data for this sample
        mask = ms._sample_assignments[:, s_idx] if s_idx < ms._sample_assignments.shape[1] else np.zeros(len(scores), bool)
        s_scores = scores[mask]
        
        # Plot complete cases
        complete = ~np.isnan(s_scores).any(axis=1)
        if complete.any():
            ax.scatter(s_scores[complete, 0], s_scores[complete, 1],
                      c=sample_colors[s_idx % len(sample_colors)], 
                      marker=sample_markers[s_idx % len(sample_markers)],
                      s=15, alpha=0.6, edgecolors='none',
                      label=f'Complete ({complete.sum()})')
        
        # Plot partial cases as ticks on axes
        only_d0 = np.isnan(s_scores[:, 1]) & ~np.isnan(s_scores[:, 0])
        only_d1 = np.isnan(s_scores[:, 0]) & ~np.isnan(s_scores[:, 1])
        if only_d0.any():
            ax.scatter(s_scores[only_d0, 0], 
                      np.full(only_d0.sum(), x2_range[0] + 0.05),
                      c=sample_colors[s_idx % len(sample_colors)],
                      marker='|', s=30, alpha=0.5,
                      label=f'Dim 0 only ({only_d0.sum()})')
        if only_d1.any():
            ax.scatter(np.full(only_d1.sum(), x1_range[0] + 0.05),
                      s_scores[only_d1, 1],
                      c=sample_colors[s_idx % len(sample_colors)],
                      marker='_', s=30, alpha=0.5,
                      label=f'Dim 1 only ({only_d1.sum()})')
        
        # Mark component centers
        for c in range(K):
            mu_c = params[c][0]
            ax.plot(mu_c[0], mu_c[1], 'k*', markersize=12, zorder=5, alpha=w_s[c])
            ax.annotate(f'C{c} (w={w_s[c]:.2f})', mu_c, fontsize=8,
                       xytext=(5, 5), textcoords='offset points')
        
        name = sample_names[s_idx] if s_idx < len(sample_names) else f'Sample {s_idx}'
        ax.set_title(f'{name}\n(n={mask.sum()})', fontsize=10)
        ax.set_xlabel(ms.dataset_names[0] if hasattr(ms, 'dataset_names') else 'Dim 0')
        ax.set_ylabel(ms.dataset_names[1] if hasattr(ms, 'dataset_names') else 'Dim 1')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(x1_range)
        ax.set_ylim(x2_range)
    
    # ─── Last panel: all samples together ───
    ax = axes[-1]
    # Overall mixture (equal weights)
    log_mix = None
    w_avg = weights.mean(axis=0)
    for c in range(K):
        lp = msn_logpdf_grid(X1, X2, *params[c])
        weighted = np.log(w_avg[c] + 1e-300) + lp
        if log_mix is None:
            log_mix = weighted
        else:
            mx = np.maximum(log_mix, weighted)
            log_mix = mx + np.log(np.exp(log_mix - mx) + np.exp(weighted - mx))
    
    levels = np.linspace(np.nanpercentile(log_mix[np.isfinite(log_mix)], 5),
                         np.nanmax(log_mix[np.isfinite(log_mix)]), 15)
    ax.contourf(X1, X2, log_mix, levels=levels, cmap='Greys', alpha=0.3)
    ax.contour(X1, X2, log_mix, levels=levels, colors='gray', linewidths=0.5, alpha=0.5)
    
    for s_idx in range(n_samples):
        mask = ms._sample_assignments[:, s_idx] if s_idx < ms._sample_assignments.shape[1] else np.zeros(len(scores), bool)
        s_scores = scores[mask]
        complete = ~np.isnan(s_scores).any(axis=1)
        name = sample_names[s_idx] if s_idx < len(sample_names) else f'Sample {s_idx}'
        if complete.any():
            ax.scatter(s_scores[complete, 0], s_scores[complete, 1],
                      c=sample_colors[s_idx % len(sample_colors)],
                      marker=sample_markers[s_idx % len(sample_markers)],
                      s=15, alpha=0.5, edgecolors='none', label=name)
    
    for c in range(K):
        mu_c = params[c][0]
        ax.plot(mu_c[0], mu_c[1], 'k*', markersize=12, zorder=5)
    
    ax.set_title('All Samples', fontsize=10)
    ax.set_xlabel(ms.dataset_names[0] if hasattr(ms, 'dataset_names') else 'Dim 0')
    ax.set_ylabel(ms.dataset_names[1] if hasattr(ms, 'dataset_names') else 'Dim 1')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)
    
    plt.tight_layout()
    plt.show()
    
    # ─── Figure 2: Marginal densities per dimension ───
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 6), squeeze=False)
    
    for dim in range(2):
        x_grid = np.linspace(np.nanmin(scores[:, dim]) - pad,
                             np.nanmax(scores[:, dim]) + pad, 500)
        # For marginal: just use 1-d params (mu[dim], Delta[dim], Gamma[dim,dim])
        for s_idx in range(n_samples):
            ax = axes[dim, s_idx]
            w_s = weights[s_idx]
            
            mask = ms._sample_assignments[:, s_idx] if s_idx < ms._sample_assignments.shape[1] else np.zeros(len(scores), bool)
            s_scores = scores[mask, dim]
            s_scores = s_scores[~np.isnan(s_scores)]
            
            ax.hist(s_scores, bins=30, density=True, alpha=0.4,
                   color=sample_colors[s_idx % len(sample_colors)],
                   edgecolor='white', linewidth=0.5)
            
            # Marginal mixture density
            mix_pdf = np.zeros_like(x_grid)
            for c in range(K):
                mu_d = params[c][0][dim]
                Delta_d = params[c][1][dim]
                Gamma_d = params[c][2][dim, dim]
                # 1-d MSN: f(x) = 2/omega * phi((x-mu)/omega) * Phi(lambda*(x-mu)/omega)
                omega = np.sqrt(Gamma_d + Delta_d**2)
                delta = Delta_d / omega
                lam = delta / np.sqrt(1 - delta**2 + 1e-12)
                from scipy.stats import skewnorm
                comp_pdf = w_s[c] * skewnorm.pdf(x_grid, lam, loc=mu_d, scale=omega)
                ax.plot(x_grid, comp_pdf, '--', alpha=0.7, linewidth=1,
                       label=f'C{c} (w={w_s[c]:.2f})')
                mix_pdf += comp_pdf
            
            ax.plot(x_grid, mix_pdf, 'k-', linewidth=1.5, label='Mixture')
            
            name = sample_names[s_idx] if s_idx < len(sample_names) else f'Sample {s_idx}'
            dim_name = ms.dataset_names[dim] if hasattr(ms, 'dataset_names') else f'Dim {dim}'
            ax.set_title(f'{name}\n{dim_name} (n={len(s_scores)})', fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
    
    plt.tight_layout()
    plt.show()
    
    # ─── Print summary ───
    print("\n" + "="*60)
    print("FIT SUMMARY")
    print("="*60)
    for c in range(K):
        mu, Delta, Gamma = params[c]
        Omega = Gamma + np.outer(Delta, Delta)
        print(f"\nComponent {c}:")
        print(f"  mu    = {mu}")
        print(f"  Delta = {Delta}")
        print(f"  Omega = \n    {Omega[0]}\n    {Omega[1]}")
        print(f"  Gamma eigenvalues = {np.linalg.eigvalsh(Gamma)}")
    print(f"\nWeights:")
    for s_idx in range(n_samples):
        name = sample_names[s_idx] if s_idx < len(sample_names) else f'Sample {s_idx}'
        print(f"  {name}: {weights[s_idx]}")
    ll = fit_result.get('likelihoods', [])
    if len(ll):
        print(f"\nFinal log-likelihood: {ll[-1]:.6f}")
        print(f"Iterations: {len(ll)}")




"""
Multivariate calibration visualization.

Shows:
  - 2D log LR+ heatmap with point-assignment contour boundaries
  - Per-sample 2D scatter colored by assigned points
  - Marginal 1D LR+ curves per dimension
  - Per-sample density fits with component decomposition
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import multivariate_normal as mvn, norm, skewnorm

# ──────────────────────────────────────
# Config
# ──────────────────────────────────────
SAMPLE_COLORS = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
SAMPLE_NAMES_DEFAULT = [
    "Pathogenic/Likely Pathogenic",
    "Benign/Likely Benign",
    "Population",
    "Synonymous",
]
POINT_CMAP_COLORS = [
    (0.0, '#08306b'),   # strong benign (dark blue)
    (0.3, '#4292c6'),   # moderate benign
    (0.45, '#c6dbef'),  # weak benign
    (0.5, '#f7f7f7'),   # neutral (white)
    (0.55, '#fcbba1'),  # weak pathogenic
    (0.7, '#ef3b2c'),   # moderate pathogenic
    (1.0, '#67000d'),   # strong pathogenic (dark red)
]
POINT_CMAP = LinearSegmentedColormap.from_list('evidence', POINT_CMAP_COLORS)
SAMPLE_MARKERS = ['o', 's', '^', 'D']


def _msn_logpdf_grid(X1, X2, mu, Delta, Gamma):
    mu, Delta, Gamma = np.asarray(mu, float), np.asarray(Delta, float), np.asarray(Gamma, float)
    pts = np.column_stack([X1.ravel(), X2.ravel()])
    Omega = Gamma + np.outer(Delta, Delta)
    Omega = 0.5 * (Omega + Omega.T)
    eigv = np.linalg.eigvalsh(Omega)
    if eigv.min() < 1e-10:
        Omega += (1e-10 - eigv.min() + 1e-10) * np.eye(2)
    try:
        log_phi = mvn(mean=mu, cov=Omega, allow_singular=True).logpdf(pts)
    except Exception:
        return np.full(X1.shape, -np.inf)
    try:
        Oid = np.linalg.solve(Omega, Delta)
    except Exception:
        return np.full(X1.shape, -np.inf)
    eta = (pts - mu) @ Oid
    s2 = max(1.0 - Delta @ Oid, 1e-12)
    log_Phi = norm.logcdf(eta / np.sqrt(s2))
    result = (np.log(2) + log_phi + log_Phi).reshape(X1.shape)
    result[~np.isfinite(result)] = -np.inf
    return result


def _mixture_logpdf_grid(X1, X2, params, weights):
    log_mix = None
    for p, w in zip(params, weights):
        if w < 1e-300:
            continue
        lp = _msn_logpdf_grid(X1, X2, *p) + np.log(w)
        if log_mix is None:
            log_mix = lp.copy()
        else:
            mx = np.maximum(log_mix, lp)
            log_mix = mx + np.log(np.exp(log_mix - mx) + np.exp(lp - mx))
    return log_mix if log_mix is not None else np.full(X1.shape, -np.inf)


def _marginal_params(mu, Delta, Gamma, dim):
    mu_d = mu[dim]
    Delta_d = Delta[dim]
    Gamma_d = Gamma[dim, dim]
    omega = np.sqrt(Gamma_d + Delta_d ** 2)
    if omega < 1e-12:
        return 0.0, mu_d, 1e-6
    delta = np.clip(Delta_d / omega, -0.9999, 0.9999)
    lam = delta / np.sqrt(1 - delta ** 2 + 1e-12)
    return lam, mu_d, omega


def _log_lr_plus_grid(X1, X2, params, weights, pathogenic_idx, benign_idx, benign_method='benign',
                       synonymous_idx=None):
    """Compute log LR+ = log f_path(x) - log f_benign(x) on a 2D grid."""
    w_p = weights[pathogenic_idx]
    if benign_method == 'synonymous' and synonymous_idx is not None:
        w_b = weights[synonymous_idx]
    elif benign_method == 'avg' and benign_idx is not None and synonymous_idx is not None:
        w_b = (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
    else:
        w_b = weights[benign_idx]
    log_fp = _mixture_logpdf_grid(X1, X2, params, w_p)
    log_fb = _mixture_logpdf_grid(X1, X2, params, w_b)
    return log_fp - log_fb, log_fp, log_fb


def _assign_points_2d(log_lr, tau_p_log, tau_b_log, point_values):
    """Assign integer point values to each grid cell based on LR+ thresholds."""
    points = np.zeros_like(log_lr, dtype=int)
    # Pathogenic: LR+ >= threshold
    for pv in point_values:
        mask = log_lr >= tau_p_log[pv - 1]
        points[mask] = pv
    # Benign: LR+ <= threshold
    for pv in point_values:
        mask = log_lr <= tau_b_log[pv - 1]
        points[mask] = -pv
    return points


def compute_mv_prior(fit_result, ms, pathogenic_idx=0, benign_idx=1, gnomad_idx=2,
                      benign_method='benign', synonymous_idx=3, **kwargs):
    """Estimate prior from multivariate model using EM on population scores."""
    from src.assay_calibration.fit_utils.multivariate.density_utils import msn_logpdf_alternate_missing
    params = fit_result['component_params']
    weights = fit_result['weights']
    K = len(params)

    pop_scores = ms.scores[ms._sample_assignments[:, gnomad_idx]]

    # Compute densities under pathogenic and benign mixture
    w_p = weights[pathogenic_idx]
    if benign_method == 'synonymous' and synonymous_idx is not None:
        w_b = weights[synonymous_idx]
    elif benign_method == 'avg':
        w_b = (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
    else:
        w_b = weights[benign_idx]

    # log f_path and log f_benign for each population observation
    log_fp_list = [np.log(w_p[c] + 1e-300) + msn_logpdf_alternate_missing(pop_scores, *params[c])
                   for c in range(K)]
    log_fb_list = [np.log(w_b[c] + 1e-300) + msn_logpdf_alternate_missing(pop_scores, *params[c])
                   for c in range(K)]

    from scipy.special import logsumexp
    log_fp = logsumexp(np.array(log_fp_list), axis=0)
    log_fb = logsumexp(np.array(log_fb_list), axis=0)
    fp = np.exp(log_fp)
    fb = np.exp(log_fb)

    prior = 0.5
    for _ in range(kwargs.get('max_em_steps', 10000)):
        with np.errstate(divide='ignore', invalid='ignore'):
            posteriors = 1 / (1 + (1 - prior) / prior * fb / fp)
        new_prior = np.nanmean(posteriors)
        if abs(new_prior - prior) < kwargs.get('tolerance', 1e-6):
            prior = new_prior
            break
        prior = new_prior
    return prior


def plot_mv_calibration(ms, fit_result, prior=None, point_values=None, 
                         pathogenic_idx=0, benign_idx=1, gnomad_idx=2, synonymous_idx=3,
                         benign_method='benign', figsize=None, title=None):
    """
    Full multivariate calibration visualization.

    Parameters
    ----------
    ms : MultiScoreset
    fit_result : dict with 'component_params' and 'weights'
    prior : float, optional. If None, computed via EM.
    point_values : list of int, optional. Default [1..8].
    """
    from src.assay_calibration.fit_utils.fit import thresholds_from_prior

    params = fit_result['component_params']
    weights = fit_result['weights']
    scores = ms.scores
    K = len(params)
    n_samples = weights.shape[0]
    sa = ms._sample_assignments

    if point_values is None:
        point_values = list(range(1, 9))
    if prior is None:
        prior = compute_mv_prior(fit_result, ms, pathogenic_idx, benign_idx, gnomad_idx,
                                  benign_method, synonymous_idx)

    dataset_names = getattr(ms, 'dataset_names', ['Dim 0', 'Dim 1'])
    sample_names = getattr(ms, 'sample_names', SAMPLE_NAMES_DEFAULT[:n_samples])

    # Thresholds
    tau_p, tau_b, C = thresholds_from_prior(prior, point_values)
    tau_p_log = np.log(tau_p)
    tau_b_log = np.log(tau_b)

    # Grid
    pad = 0.5
    x1r = (np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad)
    x2r = (np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad)
    x1g = np.linspace(*x1r, 250)
    x2g = np.linspace(*x2r, 250)
    X1, X2 = np.meshgrid(x1g, x2g)

    # Compute LR+ on grid
    log_lr, log_fp_grid, log_fb_grid = _log_lr_plus_grid(
        X1, X2, params, weights, pathogenic_idx, benign_idx,
        benign_method, synonymous_idx
    )

    # Point assignments on grid
    point_grid = _assign_points_2d(log_lr, tau_p_log, tau_b_log, point_values)

    # Per-observation point assignment (for complete-case obs)
    complete = ~np.isnan(scores).any(axis=1)
    obs_points = np.full(len(scores), np.nan)
    if complete.any():
        from src.assay_calibration.fit_utils.multivariate.density_utils import msn_logpdf_alternate_missing
        from scipy.special import logsumexp
        w_p = weights[pathogenic_idx]
        if benign_method == 'synonymous' and synonymous_idx is not None:
            w_b = weights[synonymous_idx]
        elif benign_method == 'avg':
            w_b = (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
        else:
            w_b = weights[benign_idx]
        lfp = logsumexp([np.log(w_p[c] + 1e-300) + msn_logpdf_alternate_missing(scores[complete], *params[c])
                         for c in range(K)], axis=0)
        lfb = logsumexp([np.log(w_b[c] + 1e-300) + msn_logpdf_alternate_missing(scores[complete], *params[c])
                         for c in range(K)], axis=0)
        obs_lr = lfp - lfb
        obs_pts = np.zeros(complete.sum(), dtype=int)
        for pv in point_values:
            obs_pts[obs_lr >= tau_p_log[pv - 1]] = pv
        for pv in point_values:
            obs_pts[obs_lr <= tau_b_log[pv - 1]] = -pv
        obs_points[complete] = obs_pts

    # ════════════════════════════════════
    # Layout
    # ════════════════════════════════════
    # Row 0: [LR+ heatmap] [Point regions] [Obs by points]
    # Row 1: Per-sample 2D scatter+contours (n_samples panels)
    # Row 2: Marginal dim0 per sample + LR+ marginal dim0
    # Row 3: Marginal dim1 per sample + LR+ marginal dim1

    n_top = 2
    n_bot = max(n_samples, 2) + 1  # +1 for marginal LR+
    n_cols = max(n_top, n_bot)
    if figsize is None:
        figsize = (5.5 * n_cols, 20)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, n_cols, figure=fig, height_ratios=[1.3, 1.1, 0.7, 0.7],
                           hspace=0.38, wspace=0.35)

    # ─── Row 0, Col 0: Point assignment regions (full) ───
    ax = fig.add_subplot(gs[0, 0])
    max_pt = max(point_values)
    pt_norm = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)
    im = ax.pcolormesh(X1, X2, point_grid, cmap=POINT_CMAP, norm=pt_norm, shading='auto')
    cb = plt.colorbar(im, ax=ax, label='Evidence Points', shrink=0.8)
    cb.set_ticks(list(range(-max_pt, max_pt + 1, 2)))
    # Contour boundaries between point levels
    for pv in point_values:
        ax.contour(X1, X2, point_grid, levels=[pv - 0.5], colors='red', linewidths=0.5, alpha=0.4)
        ax.contour(X1, X2, point_grid, levels=[-pv + 0.5], colors='blue', linewidths=0.5, alpha=0.4)
    for c in range(K):
        ax.plot(*params[c][0], 'k*', ms=10, zorder=5)
    ax.set_xlabel(dataset_names[0])
    ax.set_ylabel(dataset_names[1])
    ax.set_title(f'Point Regions\nprior={prior:.4f}, C={C:.1f}', fontsize=10, fontweight='bold')
    ax.set_xlim(x1r); ax.set_ylim(x2r)

    # ─── Row 0, Col 1: Observations colored by points ───
    ax = fig.add_subplot(gs[0, 1])
    ax.pcolormesh(X1, X2, point_grid, cmap=POINT_CMAP, norm=pt_norm, shading='auto', alpha=0.15)
    for s_idx in range(n_samples):
        mask = sa[:, s_idx] & complete
        if not mask.any():
            continue
        s_pts = obs_points[mask]
        s_sc = scores[mask]
        name = sample_names[s_idx] if s_idx < len(sample_names) else f'S{s_idx}'
        ax.scatter(s_sc[:, 0], s_sc[:, 1], c=s_pts, cmap=POINT_CMAP, norm=pt_norm,
                   s=18, alpha=0.7, edgecolors=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                   linewidths=0.6, marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)],
                   label=f'{name} ({mask.sum()})')
    ax.legend(fontsize=6, loc='upper left', framealpha=0.7)
    ax.set_xlabel(dataset_names[0])
    ax.set_ylabel(dataset_names[1])
    ax.set_title('Observations by Evidence', fontsize=10, fontweight='bold')
    ax.set_xlim(x1r); ax.set_ylim(x2r)

    # Hide extra cols in row 0
    for c_idx in range(2, n_cols):
        fig.add_subplot(gs[0, c_idx]).axis('off')

    # ─── Row 1: Per-sample 2D density + scatter ───
    for s_idx in range(n_samples):
        ax = fig.add_subplot(gs[1, s_idx])
        w_s = weights[s_idx]
        mask = sa[:, s_idx] if s_idx < sa.shape[1] else np.zeros(len(scores), bool)
        s_scores = scores[mask]

        # Mixture density contours only (no per-component overlays)
        log_mix = _mixture_logpdf_grid(X1, X2, params, w_s)
        finite = log_mix[np.isfinite(log_mix)]
        if len(finite) > 10:
            lvls = np.linspace(np.percentile(finite, 10), np.max(finite), 12)
            ax.contourf(X1, X2, log_mix, levels=lvls, cmap='Blues', alpha=0.35)
            ax.contour(X1, X2, log_mix, levels=lvls, colors='steelblue',
                       linewidths=0.4, alpha=0.5)

        # Scatter complete cases
        comp = ~np.isnan(s_scores).any(axis=1)
        if comp.any():
            ax.scatter(s_scores[comp, 0], s_scores[comp, 1],
                       c=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                       marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)],
                       s=10, alpha=0.5, edgecolors='none')

        # Partial observations
        only0 = np.isnan(s_scores[:, 1]) & ~np.isnan(s_scores[:, 0])
        if only0.any():
            yt = x2r[0] + 0.04 * (x2r[1] - x2r[0])
            ax.scatter(s_scores[only0, 0], np.full(only0.sum(), yt),
                       c=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)], marker='|', s=20, alpha=0.35)
        only1 = np.isnan(s_scores[:, 0]) & ~np.isnan(s_scores[:, 1])
        if only1.any():
            xt = x1r[0] + 0.04 * (x1r[1] - x1r[0])
            ax.scatter(np.full(only1.sum(), xt), s_scores[only1, 1],
                       c=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)], marker='_', s=20, alpha=0.35)

        # Component centers
        for c in range(K):
            mu_c = params[c][0]
            ax.plot(mu_c[0], mu_c[1], 'k*', ms=11, zorder=5, alpha=max(w_s[c], 0.2))
            ax.annotate(f'C{c} ({w_s[c]:.2f})', mu_c, fontsize=6, xytext=(4, 4),
                        textcoords='offset points', fontweight='bold', alpha=max(w_s[c], 0.3))

        name = sample_names[s_idx] if s_idx < len(sample_names) else f'S{s_idx}'
        ax.set_title(f'{name} (n={mask.sum()})', fontsize=9, fontweight='bold',
                     color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)])
        ax.set_xlabel(dataset_names[0], fontsize=8)
        ax.set_ylabel(dataset_names[1], fontsize=8)
        ax.set_xlim(x1r); ax.set_ylim(x2r)
        ax.grid(linewidth=0.2, alpha=0.3)

    for c_idx in range(n_samples, n_cols):
        fig.add_subplot(gs[1, c_idx]).axis('off')

    # ─── Rows 2–3: Marginal densities + marginal LR+ per dimension ───
    SAMPLE_MARKERS_MPL = ['o', 's', '^', 'D']
    for dim in range(2):
        x_grid = np.linspace(np.nanmin(scores[:, dim]) - pad,
                              np.nanmax(scores[:, dim]) + pad, 500)

        # Per-sample marginal density
        for s_idx in range(min(n_samples, n_cols - 1)):
            ax = fig.add_subplot(gs[dim + 2, s_idx])
            w_s = weights[s_idx]
            mask = sa[:, s_idx] if s_idx < sa.shape[1] else np.zeros(len(scores), bool)
            s_sc = scores[mask, dim]
            s_sc = s_sc[~np.isnan(s_sc)]

            if len(s_sc) > 1:
                ax.hist(s_sc, bins=min(30, max(8, len(s_sc) // 5)), density=True, alpha=0.3,
                        color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)], edgecolor='white', linewidth=0.5)

            mix_pdf = np.zeros_like(x_grid)
            for c in range(K):
                lam, mu_d, omega = _marginal_params(*params[c], dim)
                cpdf = w_s[c] * skewnorm.pdf(x_grid, lam, loc=mu_d, scale=omega)
                ax.plot(x_grid, cpdf, '--', color=f'C{c}', alpha=0.6, lw=0.9,
                        label=f'C{c} ({w_s[c]:.2f})')
                mix_pdf += cpdf
            ax.plot(x_grid, mix_pdf, 'k-', lw=1.3, alpha=0.7, label='Mix')

            name = sample_names[s_idx] if s_idx < len(sample_names) else f'S{s_idx}'
            ax.set_title(f'{name} — {dataset_names[dim]} (n={len(s_sc)})', fontsize=8,
                         color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)])
            ax.legend(fontsize=5, loc='upper right', framealpha=0.6)
            ax.set_xlabel('Score', fontsize=7)
            ax.set_ylabel('Density', fontsize=7)
            ax.grid(lw=0.2, alpha=0.3)

        # Marginal LR+ for this dimension (last column)
        ax = fig.add_subplot(gs[dim + 2, n_cols - 1])

        # Compute marginal LR+ by integrating out the other dimension
        # Approximate: evaluate LR+ along a line at median of other dim
        other_dim = 1 - dim
        other_median = np.nanmedian(scores[:, other_dim])
        if dim == 0:
            lr_marginal = np.array([
                _log_lr_plus_grid(
                    np.array([[xi]]), np.array([[other_median]]),
                    params, weights, pathogenic_idx, benign_idx, benign_method, synonymous_idx
                )[0].item() for xi in x_grid
            ])
        else:
            lr_marginal = np.array([
                _log_lr_plus_grid(
                    np.array([[other_median]]), np.array([[xi]]),
                    params, weights, pathogenic_idx, benign_idx, benign_method, synonymous_idx
                )[0].item() for xi in x_grid
            ])

        ax.plot(x_grid, lr_marginal, 'k-', lw=1.5, label=f'LR+ (other={other_median:.2f})')
        ax.axhline(0, color='gray', lw=0.8, ls='-', alpha=0.5)

        # Thresholds
        for pv in [1, 4, max(point_values)]:
            if pv <= max(point_values):
                ax.axhline(tau_p_log[pv - 1], color='red', ls='--', lw=0.6, alpha=0.5)
                ax.axhline(tau_b_log[pv - 1], color='blue', ls='--', lw=0.6, alpha=0.5)
                ax.text(x_grid[-1], tau_p_log[pv - 1], f'+{pv}', fontsize=6, ha='right',
                        va='bottom', color='red')
                ax.text(x_grid[-1], tau_b_log[pv - 1], f'-{pv}', fontsize=6, ha='right',
                        va='top', color='blue')

        # Rug plot of observations
        for s_idx in range(n_samples):
            mask_s = sa[:, s_idx]
            s_sc = scores[mask_s, dim]
            s_sc = s_sc[~np.isnan(s_sc)]
            if len(s_sc) > 0:
                ax.plot(s_sc, np.full_like(s_sc, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -5),
                        '|', color=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                        ms=4, alpha=0.3)

        ax.set_title(f'Marginal LR+ — {dataset_names[dim]}', fontsize=8, fontweight='bold')
        ax.set_xlabel('Score', fontsize=7)
        ax.set_ylabel('log LR+', fontsize=7)
        ax.legend(fontsize=5, framealpha=0.6)
        ylim_max = min(max(tau_p_log[-1] + 2, 5), 15)
        ylim_min = max(min(tau_b_log[-1] - 2, -5), -15)
        ax.set_ylim(ylim_min, ylim_max)
        ax.grid(lw=0.2, alpha=0.3)

    # ── Title ──
    ll = fit_result.get('likelihoods', [])
    ll_str = f", LL={ll[-1]:.4f}" if len(ll) and np.isfinite(ll[-1]) else ""
    suptitle = title or getattr(ms, 'scoreset_name', 'Multivariate Calibration')
    n_miss = np.isnan(scores).sum()
    n_total = scores.size
    fig.suptitle(
        f"{suptitle}\n{K} components, prior={prior:.4f}, C={C:.1f}{ll_str}, "
        f"missing={100 * n_miss / n_total:.1f}%",
        fontsize=12, fontweight='bold', y=1.02
    )

    return fig, {'prior': prior, 'C': C, 'tau_p_log': tau_p_log, 'tau_b_log': tau_b_log,
                 'obs_points': obs_points, 'point_grid': point_grid}