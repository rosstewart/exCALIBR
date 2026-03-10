import scipy.stats as sps
import numpy as np
from . import density_utils
from typing import Tuple


def density_constraint_violated(params_1, params_2, xlims: Tuple[float, float],
                                 multivariate=False) -> bool:
    """
    Check if the density ratio of distribution 1 to distribution 2 is monotonic.
    Works for both univariate and multivariate (evaluated along 1-d grid for MV).
    """
    x_grid = np.linspace(*xlims, 1000)

    if not multivariate:
        log_pdf_1 = sps.skewnorm.logpdf(x_grid, *params_1)
        log_pdf_2 = sps.skewnorm.logpdf(x_grid, *params_2)
    else:
        log_pdf_1 = density_utils.msn_logpdf_alternate(x_grid[:, None], *params_1)
        log_pdf_2 = density_utils.msn_logpdf_alternate(x_grid[:, None], *params_2)

    ix = np.logical_and(log_pdf_1 > -7.0, log_pdf_2 > -7.0)
    if np.sum(ix) < 2:
        return False
    diffs = np.diff(log_pdf_1[ix] - log_pdf_2[ix])
    return not np.all(diffs < 0)


def multicomponent_density_constraint_violated(
    param_sets, xlims, multivariate=False, D=None, tolerance=0
) -> bool:
    """
    For each pair of adjacent distributions, check density-ratio monotonicity.

    For univariate: xlims = (xmin, xmax), D is ignored.
    For multivariate: xlims can be (xmin, xmax) applied to first dimension,
        or a dict {d: (xmin_d, xmax_d)} for per-dimension checks.
        If D is None, defaults to checking along a 1-d slice through the
        component means.
    """
    if not multivariate:
        # ---- univariate path (unchanged logic) ----
        if D is not None:
            # legacy: D is an iterable of dimension indices
            for d in D:
                x_d = np.linspace(*xlims[d], 1000)
                log_pdfs = [sps.skewnorm.logpdf(x_d, *p) for p in param_sets]
                for i in range(len(log_pdfs) - 1):
                    ix = np.logical_and(log_pdfs[i] > -7.0, log_pdfs[i + 1] > -7.0)
                    if np.sum(ix) < 2:
                        continue
                    diffs = np.diff(log_pdfs[i][ix] - log_pdfs[i + 1][ix])
                    if np.any(diffs > tolerance):
                        return True
            return False
        else:
            x_grid = np.linspace(*xlims, 1000)
            log_pdfs = [sps.skewnorm.logpdf(x_grid, *p) for p in param_sets]
            for i in range(len(log_pdfs) - 1):
                ix = np.logical_and(log_pdfs[i] > -7.0, log_pdfs[i + 1] > -7.0)
                if np.sum(ix) < 2:
                    continue
                diffs = np.diff(log_pdfs[i][ix] - log_pdfs[i + 1][ix])
                if np.any(diffs > tolerance):
                    return True
            return False

    # ---- multivariate path ----
    # Evaluate along a 1-d line connecting component means (first two),
    # or along first coordinate axis if means aren't available.
    mus = [np.asarray(p[0]) for p in param_sets]  # p[0] = mu in alternate form
    K_dim = mus[0].shape[0]

    # Build a 1-d probe direction: from first to last mean
    direction = mus[-1] - mus[0]
    dir_norm = np.linalg.norm(direction)
    if dir_norm < 1e-12:
        direction = np.zeros(K_dim)
        direction[0] = 1.0
    else:
        direction = direction / dir_norm

    # Project all means onto this direction to find range
    projections = [mu @ direction for mu in mus]
    proj_min = min(projections) - 3.0 * max(np.sqrt(np.abs(np.diag(np.asarray(p[2]))).max()) for p in param_sets)
    proj_max = max(projections) + 3.0 * max(np.sqrt(np.abs(np.diag(np.asarray(p[2]))).max()) for p in param_sets)
    t_grid = np.linspace(proj_min, proj_max, 1000)
    # Points along the line: mu_center + t * direction
    center = mus[0]
    x_grid = center[None, :] + t_grid[:, None] * direction[None, :]  # (1000, K_dim)

    log_pdfs = []
    for p in param_sets:
        try:
            lp = density_utils.msn_logpdf_alternate(x_grid, *p)
            lp = np.asarray(lp, dtype=float)
            lp[~np.isfinite(lp)] = -np.inf
            log_pdfs.append(lp)
        except Exception:
            # If density evaluation fails, we can't check — assume violated
            return True

    for i in range(len(log_pdfs) - 1):
        ix = np.logical_and(log_pdfs[i] > -7.0, log_pdfs[i + 1] > -7.0)
        if np.sum(ix) < 2:
            continue
        diffs = np.diff(log_pdfs[i][ix] - log_pdfs[i + 1][ix])
        if np.any(diffs > tolerance):
            return True
    return False


def positive_likelihood_ratio_montonicity_constraint_violated(
    component_param_sets, weights_pathogenic, weights_benign,
    xlims: Tuple[float, float], epsilon=1e-12, multivariate=False,
):
    x_values = np.linspace(*xlims, 1000)
    if multivariate:
        # use 1-d slice along first coordinate
        K_dim = len(component_param_sets[0][0])
        x_grid = np.zeros((1000, K_dim))
        x_grid[:, 0] = x_values
        f_path = density_utils.mixture_pdf(x_grid, component_param_sets, weights_pathogenic, multivariate=True)
        f_benign = density_utils.mixture_pdf(x_grid, component_param_sets, weights_benign, multivariate=True)
    else:
        f_path = density_utils.mixture_pdf(x_values, component_param_sets, weights_pathogenic)
        f_benign = density_utils.mixture_pdf(x_values, component_param_sets, weights_benign)
    log_lr_plus = f_path - f_benign  # already log
    return (np.diff(log_lr_plus) > 0).any()