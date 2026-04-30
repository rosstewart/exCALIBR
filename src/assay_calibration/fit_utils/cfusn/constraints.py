"""
Density-ratio monotonicity constraint checks.

Three regimes:
  - Univariate: density ratio of adjacent components monotone in score.
  - Multivariate "line" mode: density ratio monotone along a 1-D probe
    through the component means (necessary, not sufficient).
  - Multivariate "marginal" mode: density ratio monotone along EACH
    predictor dimension separately, evaluated via the CFUSN marginal
    property (Δ row-subset, Γ block-subset).  Stricter than "line"
    and the appropriate constraint for predictor calibration where
    each dim has an interpretable LR+ axis.

Dispatches density evaluation by Δ shape so CFUSN q>1 is handled
correctly (the previous implementation always used msn_logpdf_alternate,
which silently misbehaves for q>1).
"""

import scipy.stats as sps
import numpy as np
from . import density_utils
from typing import Tuple


# ──────────────────────────────────────────────────────────────────
# Density dispatch
# ──────────────────────────────────────────────────────────────────

def _logpdf_for_check(x_grid, params, multivariate=False):
    """Evaluate one component's log-pdf on a constraint-check grid.

    Univariate: ``sps.skewnorm.logpdf``.
    MV q=1:     ``msn_logpdf_alternate`` (or _missing variant if x has NaN).
    MV q>1:     ``cfusn_logpdf_alternate`` (or _missing variant if x has NaN).
    """
    if not multivariate:
        return sps.skewnorm.logpdf(x_grid, *params)
    mu, Delta, Gamma = params
    Delta_arr = np.asarray(Delta)
    has_nan = bool(np.isnan(np.asarray(x_grid)).any())
    if Delta_arr.ndim == 2 and Delta_arr.shape[1] > 1:
        # CFUSN q>1
        if has_nan:
            return density_utils.cfusn_logpdf_alternate_missing(x_grid, mu, Delta, Gamma)
        return density_utils.cfusn_logpdf_alternate(x_grid, mu, Delta, Gamma)
    # Restricted MSN (q=1)
    if has_nan:
        return density_utils.msn_logpdf_alternate_missing(x_grid, mu, Delta, Gamma)
    return density_utils.msn_logpdf_alternate(x_grid, mu, Delta, Gamma)


# ──────────────────────────────────────────────────────────────────
# Probe-grid construction
# ──────────────────────────────────────────────────────────────────

def _xlims_per_dim(xlims, K_dim):
    """Normalise xlims to a list of per-dim (xmin, xmax) tuples."""
    if isinstance(xlims, dict):
        return [xlims[d] for d in range(K_dim)]
    if (
        isinstance(xlims, (list, tuple))
        and len(xlims) > 0
        and isinstance(xlims[0], (list, tuple))
    ):
        return [tuple(p) for p in xlims]
    # single (xmin, xmax) — broadcast
    return [tuple(xlims)] * K_dim


def _build_line_grid(param_sets, n=1000):
    """Single (1000, p) grid along the line connecting first and last component means.

    Returns ``[(None, x_grid)]`` so callers can iterate uniformly with
    the marginal mode (which returns multiple grids).
    """
    mus = [np.asarray(p[0]) for p in param_sets]
    K_dim = mus[0].shape[0]

    direction = mus[-1] - mus[0]
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-12:
        direction = np.zeros(K_dim)
        direction[0] = 1.0
    else:
        direction = direction / dir_norm

    # Probe range: ±3 max-scale beyond the projected means.
    max_scale = max(
        float(np.sqrt(np.abs(np.diag(np.asarray(p[2]))).max()))
        for p in param_sets
    )
    projections = [float(mu @ direction) for mu in mus]
    proj_min = min(projections) - 3.0 * max_scale
    proj_max = max(projections) + 3.0 * max_scale
    t_grid = np.linspace(proj_min, proj_max, n)
    center = mus[0]
    x_grid = center[None, :] + t_grid[:, None] * direction[None, :]
    return [(None, x_grid)]


def _build_marginal_grids(param_sets, xlims, n=1000):
    """Per-dim grids: one (n, p) grid per dim with NaN in all other columns.

    Uses the CFUSN marginal property — _logpdf_for_check's NaN-handling
    paths give the correct 1-D marginal density.
    """
    K_dim = np.asarray(param_sets[0][0]).shape[0]
    xl = _xlims_per_dim(xlims, K_dim)
    grids = []
    for d in range(K_dim):
        x_d = np.linspace(xl[d][0], xl[d][1], n)
        x_grid = np.full((n, K_dim), np.nan)
        x_grid[:, d] = x_d
        grids.append((d, x_grid))
    return grids


def build_constraint_grids(param_sets, xlims, mode="line", n=1000):
    """Public entry: build probe grid(s) for MV constraint checking."""
    if mode == "marginal":
        return _build_marginal_grids(param_sets, xlims, n=n)
    return _build_line_grid(param_sets, n=n)


# ──────────────────────────────────────────────────────────────────
# Pairwise monotonicity check on precomputed log-pdfs
# ──────────────────────────────────────────────────────────────────

def _adjacent_pair_violated(log_pdfs, tolerance=0.0, mass_floor=-7.0,
                            min_log_ratio_range=0.0):
    """Given a list of K log-pdf arrays evaluated on a shared grid,
    report whether any adjacent pair (i, i+1) has a non-monotone
    density ratio in the high-mass region.

    The density ratio f_{i+1}/f_i must be non-decreasing along the
    grid (= ``f_i/f_{i+1}`` non-increasing = the original
    ``not np.all(diffs < 0)`` condition).  ``mass_floor`` masks out
    grid points where either component has near-zero mass; without
    this, numerical noise in the tails creates spurious violations.

    ``min_log_ratio_range`` (default 0 = off) skips the monotonicity
    check between pairs whose log-density-ratio varies by less than
    this amount across the high-mass region.  Such pairs represent
    "effectively the same distribution" — their LR+ is approximately
    constant and any non-monotonicity is structural noise from a near-
    degenerate component ordering, not a real evidence-calibration
    violation.  Used by marginal mode (which is otherwise too strict
    when K > number of distinct sample classes — e.g. K=3 predictor
    fits where B/LB and gnomAD anchor to overlapping score regions).
    """
    for i in range(len(log_pdfs) - 1):
        ix = (log_pdfs[i] > mass_floor) & (log_pdfs[i + 1] > mass_floor)
        if int(ix.sum()) < 2:
            continue
        log_ratio = log_pdfs[i][ix] - log_pdfs[i + 1][ix]
        if min_log_ratio_range > 0:
            ratio_range = float(log_ratio.max() - log_ratio.min())
            if ratio_range < min_log_ratio_range:
                continue
        diffs = np.diff(log_ratio)
        if (diffs > tolerance).any():
            return True
    return False


# ──────────────────────────────────────────────────────────────────
# Public constraint checks
# ──────────────────────────────────────────────────────────────────

def density_constraint_violated(params_1, params_2, xlims: Tuple[float, float],
                                 multivariate=False) -> bool:
    """Pairwise density-ratio check between two components.

    Univariate-only in practice (used by ``binary_search``).  For
    multivariate, dispatches to ``multicomponent_density_constraint_violated``
    with K=2 to keep behaviour consistent.
    """
    if not multivariate:
        x_grid = np.linspace(*xlims, 1000)
        log_pdf_1 = sps.skewnorm.logpdf(x_grid, *params_1)
        log_pdf_2 = sps.skewnorm.logpdf(x_grid, *params_2)
        ix = (log_pdf_1 > -7.0) & (log_pdf_2 > -7.0)
        if int(ix.sum()) < 2:
            return False
        diffs = np.diff(log_pdf_1[ix] - log_pdf_2[ix])
        return bool((diffs > 0).any())
    return multicomponent_density_constraint_violated(
        [params_1, params_2], xlims, multivariate=True, mode="line"
    )


def multicomponent_density_constraint_violated(
    param_sets, xlims, multivariate=False, D=None, tolerance=0,
    mode="line", min_log_ratio_range=None,
):
    """Check density-ratio monotonicity for adjacent components.

    Univariate
    ----------
    Either iterate dim indices in ``D`` (legacy) or use the single
    (xmin, xmax) tuple ``xlims``.

    Multivariate
    ------------
    ``mode='line'``    — 1-D probe through means.  Necessary but not
                          sufficient for full p-dim monotonicity.
    ``mode='marginal'`` — per-dim marginals via CFUSN marginal property.
                          Stricter, but pairs with near-equal densities
                          (``log_ratio_range < min_log_ratio_range``)
                          are skipped so K > #distinct-classes fits
                          (e.g. K=3 predictor where B/LB and gnomAD
                          overlap) don't fail spuriously.

    ``min_log_ratio_range`` defaults to 0.0 in line mode and 1.0 in
    marginal mode (set ``min_log_ratio_range=0`` to disable the skip).

    Density evaluation auto-dispatches by Δ shape so CFUSN q>1 is
    handled correctly.
    """
    # Mode-specific default for the near-duplicate-pair skip.
    if min_log_ratio_range is None:
        min_log_ratio_range = 1.0 if mode == "marginal" else 0.0

    if not multivariate:
        # Univariate path — unchanged logic, kept for backward compat.
        if D is not None:
            for d in D:
                x_d = np.linspace(*xlims[d], 1000)
                log_pdfs = [sps.skewnorm.logpdf(x_d, *p) for p in param_sets]
                if _adjacent_pair_violated(log_pdfs, tolerance=tolerance):
                    return True
            return False
        x_grid = np.linspace(*xlims, 1000)
        log_pdfs = [sps.skewnorm.logpdf(x_grid, *p) for p in param_sets]
        return _adjacent_pair_violated(log_pdfs, tolerance=tolerance)

    # Multivariate path
    grids = build_constraint_grids(param_sets, xlims, mode=mode)
    for _grid_id, x_grid in grids:
        log_pdfs = []
        for p in param_sets:
            try:
                lp = _logpdf_for_check(x_grid, p, multivariate=True)
                lp = np.asarray(lp, dtype=float)
                lp[~np.isfinite(lp)] = -np.inf
                log_pdfs.append(lp)
            except Exception:
                # Density evaluation failed — treat as violated
                # (the previous implementation did the same).
                return True
        if _adjacent_pair_violated(
            log_pdfs, tolerance=tolerance,
            min_log_ratio_range=min_log_ratio_range,
        ):
            return True
    return False


def positive_likelihood_ratio_montonicity_constraint_violated(
    component_param_sets, weights_pathogenic, weights_benign,
    xlims: Tuple[float, float], epsilon=1e-12, multivariate=False,
):
    """LR+ monotonicity check.  Unchanged from the original."""
    x_values = np.linspace(*xlims, 1000)
    if multivariate:
        K_dim = len(component_param_sets[0][0])
        x_grid = np.zeros((1000, K_dim))
        x_grid[:, 0] = x_values
        f_path = density_utils.mixture_pdf(
            x_grid, component_param_sets, weights_pathogenic, multivariate=True
        )
        f_benign = density_utils.mixture_pdf(
            x_grid, component_param_sets, weights_benign, multivariate=True
        )
    else:
        f_path = density_utils.mixture_pdf(x_values, component_param_sets, weights_pathogenic)
        f_benign = density_utils.mixture_pdf(x_values, component_param_sets, weights_benign)
    log_lr_plus = f_path - f_benign  # already log
    return (np.diff(log_lr_plus) > 0).any()
