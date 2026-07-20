"""
Analyze multivariate bootstrap calibration results.

Supports both restricted MSN (q=1, Delta is a p-vector) and
CFUSN (q>=1, Delta is a p×q matrix) parameterizations.

Detection is automatic: if Delta in component params is 2-D, the CFUSN
density path is used; otherwise the original q=1 MSN path is used.

Usage:
    from analyze_mv_results import MVCalibrationAnalysis
    analysis = MVCalibrationAnalysis(ms, results_path)
    analysis.run()
    analysis.plot_comparison()
"""

import json
import gzip
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import multivariate_normal as mvn, norm, skewnorm
from scipy.special import logsumexp
from collections import defaultdict, Counter
import pandas as pd
import warnings
import os
from joblib import Parallel, delayed

# Make canonical src/ importable so this module reuses the production
# density and threshold helpers instead of maintaining parallel copies.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant
from src.assay_calibration.fit_utils.cfusn.density_utils import (
    _single_component_logpdf,
    get_q,
)
from src.assay_calibration.fit_utils.point_ranges import (
    estimate_prior_from_class_densities,
    compute_pathomechanism_pathogenic_density,
    _get_benign_reference_weights,
)


# ──────────────────────────────────────
# Constants
# ──────────────────────────────────────
SAMPLE_COLORS = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
SAMPLE_MARKERS = ['o', 's', '^', 'D']
POINT_CMAP_COLORS = [
    (0.0, '#08306b'), (0.3, '#4292c6'), (0.45, '#c6dbef'),
    (0.5, '#f7f7f7'),
    (0.55, '#fcbba1'), (0.7, '#ef3b2c'), (1.0, '#67000d'),
]
POINT_CMAP = LinearSegmentedColormap.from_list('evidence', POINT_CMAP_COLORS)
CONFIG_COLORS = {
    '2c_con': '#2ca02c', '2c_unc': '#98df8a',
    '3c_con': '#d62728', '3c_unc': '#ff9896',
}
CONFIG_LABELS = {
    '2c_con': '2c constrained', '2c_unc': '2c unconstrained',
    '3c_con': '3c constrained', '3c_unc': '3c unconstrained',
}


# ──────────────────────────────────────
# Parallel worker (module-level for joblib pickling)
# ──────────────────────────────────────

def _bootstrap_job(fit_raw, scores, sa,
                   p_idx, b_idx, g_idx, s_idx, benign_method,
                   partial_patterns, reestimate_marginal_weights,
                   extra_p_indices=None, pathomechanism_prior=True):
    """Process one bootstrap fit.  Standalone so joblib loky workers don't
    need to pickle the full MVCalibrationAnalysis object."""
    if fit_raw is None:
        return None, "fit_raw is None"
    inner = fit_raw.get('fit', fit_raw)
    if inner is None:
        return None, "inner fit is None"

    # ---- reconstitute params ----
    cp = inner.get('component_params', [])
    if not cp or any(len(p) == 0 for p in cp):
        return None, "empty component_params"
    params = []
    for p in cp:
        mu = np.array(p[0], dtype=float)
        Delta = np.array(p[1], dtype=float)
        Gamma = np.array(p[2], dtype=float)
        if Delta.ndim == 2 and Delta.shape[1] == 1:
            Delta = Delta.ravel()
        params.append((mu, Delta, Gamma))
    weights = np.array(inner['weights'], dtype=float)

    # Soft-clamp indices to the rows available in this fit's weights.
    # Fits trained with fewer samples (e.g. no synonymous) have fewer weight rows;
    # any index beyond that is treated as absent for this bootstrap.
    n_w = weights.shape[0]
    n_sa = sa.shape[1]
    eff_p = p_idx if p_idx is not None and p_idx < n_w else None
    eff_b = b_idx if b_idx is not None and b_idx < n_w else None
    eff_g = g_idx if (g_idx is not None and g_idx < n_w and g_idx < n_sa) else None
    eff_s = s_idx if s_idx is not None and s_idx < n_w else None

    # Validate extra_p_indices against weight rows
    eff_extra_p = []
    if extra_p_indices:
        for ep in extra_p_indices:
            if ep is not None and ep < n_w:
                eff_extra_p.append(ep)

    aux_lrs = {}   # {eff_idx: {'lr': array, 'prior': float}}

    nu_mode = eff_p is None and (eff_b is not None or eff_s is not None) and eff_g is not None
    pu_mode = eff_p is not None and eff_b is None and eff_s is None and eff_g is not None

    if eff_p is None and not nu_mode:
        return None, f"pathogenic index {p_idx} out of range for weights with {n_w} rows"
    if eff_b is None and eff_s is None and not pu_mode:
        return None, (
            f"neither benign ({b_idx}) nor synonymous ({s_idx}) valid for "
            f"weights with {n_w} rows"
        )

    def _sample_supports(sample_idx, obs_dims):
        """Does the given sample have any variant observed on every dim
        in ``obs_dims``? Determines whether its mixing weights are
        identified for the caller's pattern.
        """
        if sample_idx is None or sample_idx >= sa.shape[1]:
            return False
        dims = list(obs_dims)
        if not dims:
            return False
        has_obs = np.all(~np.isnan(scores[:, dims]), axis=1)
        return bool(np.any(sa[:, sample_idx].astype(bool) & has_obs))

    def _benign_w(w, b_sup=True, s_sup=True):
        b_ok = eff_b is not None and eff_b < len(w) and b_sup
        s_ok = eff_s is not None and eff_s < len(w) and s_sup
        if s_ok and benign_method == 'synonymous':
            return w[eff_s]
        if s_ok and b_ok and benign_method == 'avg':
            return (np.array(w[eff_b]) + np.array(w[eff_s])) / 2
        if b_ok:
            return w[eff_b]
        if s_ok:
            return w[eff_s]
        return None  # neither sample supports this pattern

    # ---- prior EM ----
    try:
        if eff_g is None:
            prior = 0.5
        else:
            pop_scores = scores[sa[:, eff_g].astype(bool)]
            N_pop = pop_scores.shape[0]
            K = len(params)
            w_pop = weights[eff_g]

            # ---- pathomechanism correction (default on; standard/PU only) ----
            # Replaces the raw pathogenic-labeled weight vector with a
            # corrected "assay-relevant" excess density before prior
            # estimation, mirroring get_fit_prior's pathomechanism_prior
            # (src/assay_calibration/fit_utils/point_ranges.py). Standard
            # mode anchors against the UNRESTRICTED benign-method blend (no
            # per-pattern support-gating -- the correction characterizes the
            # pathogenic-labeled sample's overall mixture composition once,
            # not per identifiability pattern); PU mode anchors against the
            # raw gnomAD/population weights, same as UV. Computed once (not
            # per pattern group): the missing-data-aware density functions
            # (_single_component_logpdf's *_missing dispatch) already handle
            # heterogeneous NaN patterns within one batched call. Not applied
            # in NU mode (no pathogenic sample to correct) or to auxiliary
            # pathogenic samples (open question, not a silent gap: aux
            # samples represent different diseases and may warrant their own
            # correction, but that's not implemented here).
            pathomechanism_no_pu_support = False
            f1_full = None
            if pathomechanism_prior and eff_p is not None and not nu_mode:
                w_N_anchor = weights[eff_g] if pu_mode else _get_benign_reference_weights(
                    weights, benign_method, eff_b, eff_s)
                labeled_scores = scores[sa[:, eff_p].astype(bool)]
                corrected, _gamma_hat = compute_pathomechanism_pathogenic_density(
                    labeled_scores, pop_scores, params, weights[eff_p], w_N_anchor,
                    multivariate=True,
                )
                if corrected is None:
                    return None, "pathomechanism correction: no assay-relevant excess"
                f1_full = corrected
                if pu_mode:
                    pathomechanism_no_pu_support = not np.any(f1_full > 1e-300)

            log_f1 = np.full(N_pop, np.nan)  # "labeled class" density on pop observations
            log_f2 = np.full(N_pop, np.nan)  # "reference class" density on pop observations
            pop_obs = ~np.isnan(pop_scores)
            pop_groups = {}
            for j in range(N_pop):
                key = frozenset(np.where(pop_obs[j])[0])
                if not key:
                    continue
                pop_groups.setdefault(key, []).append(j)

            for obs_key, ilist in pop_groups.items():
                jdx = np.array(ilist)
                x_sub = pop_scores[jdx]
                if nu_mode:
                    # NU: labeled class = benign, reference = population.
                    # Skip _sample_supports check: prior estimation only needs
                    # the benign density evaluated at gnomAD points — we don't
                    # need strict pattern overlap (unlike LR computation).
                    w_b = _benign_w(weights, b_sup=True, s_sup=True)
                    if w_b is None:
                        continue
                    log_f1[jdx] = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                             for c in range(K)], axis=0)
                    log_f2[jdx] = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                             for c in range(K)], axis=0)
                elif pu_mode:
                    # PU: labeled class = pathogenic, reference = population
                    if f1_full is None:
                        w_p = weights[eff_p]
                        log_f1[jdx] = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                                 for c in range(K)], axis=0)
                    log_f2[jdx] = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                             for c in range(K)], axis=0)
                else:
                    # Standard: pathogenic vs benign
                    b_sup = _sample_supports(eff_b, obs_key)
                    s_sup = _sample_supports(eff_s, obs_key)
                    w_b = _benign_w(weights, b_sup=b_sup, s_sup=s_sup)
                    if w_b is None:
                        continue
                    if f1_full is None:
                        w_p = weights[eff_p]
                        log_f1[jdx] = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                                 for c in range(K)], axis=0)
                    log_f2[jdx] = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                             for c in range(K)], axis=0)

            # When the pathomechanism correction ran, f1 comes from the
            # single precomputed f1_full array (already linear-space, indexed
            # like pop_scores) rather than from log_f1 -- see the comment
            # above on why this is computed once, not per pattern group.
            # `valid` is still keyed off log_f2's finiteness either way, so
            # a population row whose pattern never got a supported benign
            # reference is excluded from both f1 and f2 identically.
            if f1_full is not None:
                valid = np.isfinite(log_f2)
                f1, f2 = f1_full[valid], np.exp(log_f2[valid])
            else:
                valid = np.isfinite(log_f1) & np.isfinite(log_f2)
                f1, f2 = np.exp(log_f1[valid]), np.exp(log_f2[valid])

            # Shared with UV's get_fit_prior (src/assay_calibration/fit_utils/
            # point_ranges.py::estimate_prior_from_class_densities) -- same
            # EM loop / boundary-estimator math, same floor/discard policy
            # for PU/NU.
            if nu_mode:
                prior = estimate_prior_from_class_densities(f1, f2, mode='negative_unlabeled')
            elif pu_mode:
                no_signal = 0.01 if pathomechanism_no_pu_support else 0.1
                prior = estimate_prior_from_class_densities(
                    f1, f2, mode='positive_unlabeled', no_signal_prior=no_signal)
            else:
                prior = estimate_prior_from_class_densities(f1, f2, mode='standard')
    except Exception as e:
        import traceback
        return None, f"prior EM: {e}\n{traceback.format_exc()}"

    if not np.isfinite(prior) or prior <= 0.001 or prior >= 0.999:
        return None, f"invalid prior {prior}"

    # ---- aux prior EM (one per auxiliary pathogenic sample) ----
    # Each aux sample represents a different disease, so its prior must be
    # estimated independently using that sample's mixing weights against gnomad.
    aux_priors = {}
    if eff_extra_p:
        K_em = len(params)
        for aux_eff_p in eff_extra_p:
            if eff_g is None or eff_g >= sa.shape[1]:
                aux_priors[aux_eff_p] = 0.5
                continue
            try:
                pop_scores_a = scores[sa[:, eff_g].astype(bool)]
                N_pop_a = pop_scores_a.shape[0]
                w_p_a = weights[aux_eff_p]
                log_fp_a = np.full(N_pop_a, np.nan)
                log_fb_a = np.full(N_pop_a, np.nan)
                pop_obs_a = ~np.isnan(pop_scores_a)
                pop_groups_a = {}
                for j in range(N_pop_a):
                    key = frozenset(np.where(pop_obs_a[j])[0])
                    if not key:
                        continue
                    pop_groups_a.setdefault(key, []).append(j)
                for obs_key, ilist in pop_groups_a.items():
                    jdx = np.array(ilist)
                    b_sup = _sample_supports(eff_b, obs_key)
                    s_sup = _sample_supports(eff_s, obs_key)
                    w_b = _benign_w(weights, b_sup=b_sup, s_sup=s_sup)
                    if w_b is None:
                        continue
                    x_sub = pop_scores_a[jdx]
                    log_fp_a[jdx] = logsumexp(
                        [np.log(w_p_a[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                         for c in range(K_em)], axis=0)
                    log_fb_a[jdx] = logsumexp(
                        [np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                         for c in range(K_em)], axis=0)
                valid_a = np.isfinite(log_fp_a) & np.isfinite(log_fb_a)
                fp_a = np.exp(log_fp_a[valid_a])
                fb_a = np.exp(log_fb_a[valid_a])
                # Shared with UV's get_fit_prior and this file's main prior-EM
                # loop above (estimate_prior_from_class_densities). Not
                # pathomechanism-corrected -- see the comment on the main
                # loop's f1_full: aux samples represent different diseases
                # and may warrant their own correction, but that's not
                # implemented here. Acceptance policy below (open (0,1),
                # else 0.5) is this loop's own, distinct from the main loop's
                # 0.001/0.999 gate -- kept as-is, not harmonized.
                aux_prior = estimate_prior_from_class_densities(fp_a, fb_a, mode='standard')
                if np.isfinite(aux_prior) and 0 < aux_prior < 1:
                    aux_priors[aux_eff_p] = aux_prior
                else:
                    aux_priors[aux_eff_p] = 0.5
            except Exception:
                aux_priors[aux_eff_p] = 0.5

    # ---- LR computation ----
    try:
        N, D = scores.shape
        obs_mask = ~np.isnan(scores)
        all_dims = frozenset(range(D))

        K = len(params)
        if reestimate_marginal_weights and partial_patterns:
            S = weights.shape[0]
            marginal_w = {}
            for pattern in partial_patterns:
                obs_dims = np.array(sorted(pattern))
                has_obs = np.all(~np.isnan(scores[:, obs_dims]), axis=1)
                mw = np.zeros_like(weights)
                for s in range(S):
                    s_mask = sa[:, s] & has_obs
                    if not s_mask.any():
                        mw[s] = weights[s]
                        continue
                    x_s = scores[s_mask]
                    log_dens = np.zeros((K, s_mask.sum()))
                    for c in range(K):
                        xm = np.full_like(x_s, np.nan)
                        xm[:, obs_dims] = x_s[:, obs_dims]
                        log_dens[c] = _sn_logpdf(xm, *params[c])
                    lw = log_dens + np.log(weights[s] + 1e-300)[:, None]
                    resp = np.exp(lw - logsumexp(lw, axis=0))
                    nw = resp.mean(axis=1)
                    mw[s] = nw / nw.sum()
                marginal_w[pattern] = mw

            log_lr = np.full(N, np.nan)
            pat_groups = {}
            for j in range(N):
                key = frozenset(np.where(obs_mask[j])[0])
                pat_groups.setdefault(key, []).append(j)
            for obs_key, indices in pat_groups.items():
                if not obs_key:
                    continue
                idx = np.array(indices)
                w = weights if obs_key == all_dims else marginal_w.get(obs_key, weights)
                x_sub = scores[idx]
                if nu_mode:
                    b_sup = _sample_supports(eff_b, obs_key)
                    s_sup = _sample_supports(eff_s, obs_key)
                    w_b = _benign_w(w, b_sup=b_sup, s_sup=s_sup)
                    if w_b is None:
                        continue
                    lfb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                     for c in range(K)], axis=0)
                    lfpop = logsumexp([np.log(w[eff_g][c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                      for c in range(K)], axis=0)
                    fp = np.maximum((np.exp(lfpop) - (1 - prior) * np.exp(lfb)) / prior,
                                    np.exp(lfpop) * 1e-10)
                    log_lr[idx] = np.log(fp) - lfb
                elif pu_mode:
                    w_p = w[eff_p]
                    lfp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                     for c in range(K)], axis=0)
                    lfpop = logsumexp([np.log(w[eff_g][c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                      for c in range(K)], axis=0)
                    fb = np.maximum((np.exp(lfpop) - prior * np.exp(lfp)) / (1 - prior),
                                    np.exp(lfpop) * 1e-10)
                    log_lr[idx] = lfp - np.log(fb)
                else:
                    b_sup = _sample_supports(eff_b, obs_key)
                    s_sup = _sample_supports(eff_s, obs_key)
                    w_b = _benign_w(w, b_sup=b_sup, s_sup=s_sup)
                    if w_b is None:
                        continue
                    w_p = w[eff_p]
                    lfp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                     for c in range(K)], axis=0)
                    lfb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                     for c in range(K)], axis=0)
                    log_lr[idx] = lfp - lfb
            lr = log_lr
        else:
            log_lr = np.full(N, np.nan)
            pat_groups = {}
            for j in range(N):
                key = frozenset(np.where(obs_mask[j])[0])
                if not key:
                    continue
                pat_groups.setdefault(key, []).append(j)
            for obs_key, indices in pat_groups.items():
                idx = np.array(indices)
                x_sub = scores[idx]
                if nu_mode:
                    b_sup = _sample_supports(eff_b, obs_key)
                    s_sup = _sample_supports(eff_s, obs_key)
                    w_b = _benign_w(weights, b_sup=b_sup, s_sup=s_sup)
                    if w_b is None:
                        continue
                    lfb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                     for c in range(K)], axis=0)
                    lfpop = logsumexp([np.log(weights[eff_g][c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                      for c in range(K)], axis=0)
                    fp = np.maximum((np.exp(lfpop) - (1 - prior) * np.exp(lfb)) / prior,
                                    np.exp(lfpop) * 1e-10)
                    log_lr[idx] = np.log(fp) - lfb
                elif pu_mode:
                    w_p = weights[eff_p]
                    lfp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                     for c in range(K)], axis=0)
                    lfpop = logsumexp([np.log(weights[eff_g][c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                      for c in range(K)], axis=0)
                    fb = np.maximum((np.exp(lfpop) - prior * np.exp(lfp)) / (1 - prior),
                                    np.exp(lfpop) * 1e-10)
                    log_lr[idx] = lfp - np.log(fb)
                else:
                    b_sup = _sample_supports(eff_b, obs_key)
                    s_sup = _sample_supports(eff_s, obs_key)
                    w_b = _benign_w(weights, b_sup=b_sup, s_sup=s_sup)
                    if w_b is None:
                        continue
                    w_p = weights[eff_p]
                    lfp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                     for c in range(K)], axis=0)
                    lfb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                     for c in range(K)], axis=0)
                    log_lr[idx] = lfp - lfb
            lr = log_lr

        # ---- aux LR computation (same benign reference, different pathogenic weights) ----
        for aux_eff_p in eff_extra_p:
            aux_log_lr = np.full(N, np.nan)
            w_p_aux = weights[aux_eff_p]
            for obs_key, indices in pat_groups.items():
                if not obs_key:
                    continue
                idx = np.array(indices)
                b_sup = _sample_supports(eff_b, obs_key)
                s_sup = _sample_supports(eff_s, obs_key)
                w_b = _benign_w(weights, b_sup=b_sup, s_sup=s_sup)
                if w_b is None:
                    continue
                x_sub = scores[idx]
                lfp = logsumexp([np.log(w_p_aux[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                 for c in range(K)], axis=0)
                lfb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                 for c in range(K)], axis=0)
                aux_log_lr[idx] = lfp - lfb
            aux_lrs[aux_eff_p] = {
                'lr':     aux_log_lr,
                'vsp_lr': aux_log_lr - log_lr,              # aux vs primary P
                'prior':  aux_priors.get(aux_eff_p, 0.5),
            }

    except Exception as e:
        import traceback
        return None, f"LR computation: {e}\n{traceback.format_exc()}"

    return (prior, lr, aux_lrs), None


# ──────────────────────────────────────
# Density helpers
# ──────────────────────────────────────

def _sn_logpdf(x, mu, Delta, Gamma):
    """Skew-normal log-density wrapper.

    Delegates to the canonical implementation in
    ``src/assay_calibration/fit_utils/cfusn/density_utils.py`` so the
    analysis path uses the exact same density evaluation as the EM
    fitter. Auto-dispatches between q=1 (restricted MSN) and q>1
    (CFUSN) based on Delta shape, and handles NaN via the CFUSN
    marginal property.
    """
    lp = _single_component_logpdf(x, (mu, Delta, Gamma), multivariate=True)
    return np.atleast_1d(np.asarray(lp, dtype=float))


def _marginal_params(mu, Delta, Gamma, dim):
    """Extract univariate marginal skew-normal params for a single dimension.

    For q=1 (Delta is vector): standard marginal extraction.
    For q>1 (Delta is matrix): the marginal X_d is a CFUSN_1,q which
    reduces to a 1-D skew-normal only when q=1. For q>1, we approximate
    by projecting onto the dominant skewness direction.
    """
    Delta = np.asarray(Delta, dtype=float)

    if Delta.ndim == 2 and Delta.shape[1] > 1:
        # CFUSN: Delta is (p, q). Marginal of X_d is CFUSN_{1,q}(mu_d, Delta_d, Gamma_dd)
        # where Delta_d = Delta[dim, :] is (q,).
        # For visualization/approximation, collapse to effective scalar skewness:
        #   effective_Delta = ||Delta[dim, :]||_2 * sign(sum(Delta[dim, :]))
        # This preserves the overall skewness magnitude and direction.
        Delta_d = Delta[dim, :]  # (q,)
        effective_Delta = np.linalg.norm(Delta_d) * np.sign(np.sum(Delta_d))
        Gamma_d = Gamma[dim, dim]
        mu_d = mu[dim]
        omega = np.sqrt(Gamma_d + effective_Delta ** 2)
        if omega < 1e-12:
            return 0.0, mu_d, 1e-6
        delta = np.clip(effective_Delta / omega, -0.9999, 0.9999)
        lam = delta / np.sqrt(1 - delta ** 2 + 1e-12)
        return lam, mu_d, omega
    else:
        # q=1: standard path
        Delta_vec = np.atleast_1d(Delta).ravel()
        mu_d = mu[dim]
        Delta_d = Delta_vec[dim]
        Gamma_d = Gamma[dim, dim]
        omega = np.sqrt(Gamma_d + Delta_d ** 2)
        if omega < 1e-12:
            return 0.0, mu_d, 1e-6
        delta = np.clip(Delta_d / omega, -0.9999, 0.9999)
        lam = delta / np.sqrt(1 - delta ** 2 + 1e-12)
        return lam, mu_d, omega


def _thresholds(prior, point_values):
    """
    Compute log LR+ thresholds for each point value using the
    Tavtigian framework.
    """
    C = get_tavtigian_constant(prior)
    pv = np.array(point_values)
    tau_p = C ** (pv / 8)
    tau_b = 1.0 / tau_p
    return np.log(tau_p), np.log(tau_b), C, C


def _assign_points_vec(log_lr, tau_p_log, tau_b_log, point_values):
    """Assign integer points to each observation from log LR+ values."""
    pts = np.zeros(len(log_lr), dtype=int)
    for pv in point_values:
        pts[log_lr >= tau_p_log[pv - 1]] = pv
    for pv in point_values:
        pts[log_lr <= tau_b_log[pv - 1]] = -pv
    return pts


# ──────────────────────────────────────
# Main analysis class
# ──────────────────────────────────────

class MVCalibrationAnalysis:
    """
    Analyze multivariate bootstrap calibration results.

    Automatically supports both restricted MSN (q=1) and CFUSN (q>1)
    based on the shape of Delta in the stored component parameters.
    """

    def __init__(self, ms, gene, results_path,
                 point_values=None,
                 pathogenic_idx=0, benign_idx=1,
                 gnomad_idx=2, synonymous_idx=3,
                 auxiliary_pathogenic_indices=None,
                 benign_method='benign',
                 dataset_name=None, dataset_suffix='_mv'):
        """
        Parameters
        ----------
        dataset_name : str or None
            Explicit results-JSON key. Overrides the auto-built
            ``f"{gene}{dataset_suffix}"`` if given.
        dataset_suffix : str
            Suffix appended to ``gene`` to form the results key when
            ``dataset_name`` is not provided. Use ``'_predictors_mv'``
            for predictor-driven runs (matches predictor_mv_utils.
            DATASET_SUFFIX) and ``'_mv'`` (default) for assay runs.
        """
        self.ms = ms
        self.point_values = point_values or list(range(1, 9))
        self.benign_method = benign_method
        self.pathomechanism_prior = True  # overridden by run()'s kwarg of the same name

        # Map fixed role indices (0=path,1=ben,2=gnomad,3=syn) to effective indices
        # in ms.sample_assignments (public property, drops empty-sample columns).
        # This ensures our indices match the weight rows in stored fits.
        present = np.where(ms.sample_counts > 0)[0]  # fixed-role indices that have data

        def _eff_idx(role_name, fixed_idx):
            pos = np.searchsorted(present, fixed_idx)
            if pos < len(present) and present[pos] == fixed_idx:
                return int(pos)
            warnings.warn(f"{role_name} (role {fixed_idx}) has no observations — treating as absent")
            return None

        self.p_idx = _eff_idx("pathogenic", pathogenic_idx)
        self.b_idx = _eff_idx("benign", benign_idx)
        self.g_idx = _eff_idx("gnomad", gnomad_idx)
        self.s_idx = _eff_idx("synonymous", synonymous_idx)

        # Determine inference mode based on which labeled classes are present.
        # NU (negative-unlabeled): pathogenic absent, benign/syn + gnomad present → derive fp from pop
        # PU (positive-unlabeled): benign+syn absent, pathogenic + gnomad present → derive fb from pop
        # Standard: both pathogenic and benign/syn present
        _nu_mode = self.p_idx is None and (self.b_idx is not None or self.s_idx is not None) and self.g_idx is not None
        _pu_mode = self.p_idx is not None and self.b_idx is None and self.s_idx is None and self.g_idx is not None

        if self.p_idx is None and not _nu_mode:
            raise ValueError(
                f"Pathogenic sample (role {pathogenic_idx}) has no observations and NU mode "
                f"is unavailable (need gnomad + benign/synonymous)."
            )
        if self.b_idx is None and self.s_idx is None and not _pu_mode:
            raise ValueError(
                f"Neither benign (role {benign_idx}) nor synonymous (role {synonymous_idx}) "
                f"has observations and PU mode is unavailable (need gnomad + pathogenic)."
            )

        self.nu_mode = _nu_mode
        self.pu_mode = _pu_mode

        if _nu_mode:
            warnings.warn(
                f"Pathogenic sample (role {pathogenic_idx}) absent — using negative-unlabeled "
                f"(NU) mode: fp derived from population unmixing."
            )
        if _pu_mode:
            warnings.warn(
                f"Benign/synonymous samples absent — using positive-unlabeled (PU) mode: "
                f"fb derived from population unmixing."
            )

        # Auxiliary pathogenic samples — each gets its own point regions using
        # the same benign reference as the primary pathogenic sample.
        self.aux_p_entries = []  # list of (fixed_idx, eff_idx)
        for fixed_idx in (auxiliary_pathogenic_indices or []):
            eff = _eff_idx(f"aux pathogenic {fixed_idx}", fixed_idx)
            if eff is not None:
                self.aux_p_entries.append((fixed_idx, eff))

        print(f"Loading results from {results_path}...")
        with gzip.open(results_path, 'rt', encoding='utf-8') as f:
            raw = json.load(f)
        print(f"Genes available: {raw.keys()}")

        self.dataset_name = dataset_name or f"{gene}{dataset_suffix}"
        if self.dataset_name not in raw:
            raise KeyError(
                f"Results JSON has no key '{self.dataset_name}'. "
                f"Available: {sorted(raw.keys())}"
            )
        self.raw_boots = raw[self.dataset_name]
        print(f"  Using gene key: {self.dataset_name}")

        sample_boot = next(iter(self.raw_boots.values()))
        self.configs = sorted([k for k in sample_boot.keys() if sample_boot[k] is not None])
        print(f"  Configs found: {self.configs}")
        print(f"  Bootstraps: {len(self.raw_boots)}")

        # Detect latent_q from the first valid fit
        self._latent_q = self._detect_latent_q()

        self.results = {}

    def _detect_latent_q(self):
        """Detect latent dimension q from the first valid fit in results."""
        for boot_data in self.raw_boots.values():
            for config in self.configs:
                fit_raw = boot_data.get(config)
                if fit_raw is None:
                    continue
                inner = fit_raw.get('fit', fit_raw)
                fit = self._reconstitute_params(inner)
                if fit is not None:
                    return get_q(fit['component_params'])
        return 1

    def _reconstitute_params(self, fit_dict):
        """Convert JSON lists back to numpy arrays for alternate-form params.

        Handles both q=1 (Delta stored as list) and q>1 (Delta stored as
        list-of-lists / 2D array).
        """
        if fit_dict is None:
            return None
        cp = fit_dict.get('component_params', [])
        if not cp or any(len(p) == 0 for p in cp):
            return None
        params = []
        for p in cp:
            mu = np.array(p[0], dtype=float)
            Delta = np.array(p[1], dtype=float)
            Gamma = np.array(p[2], dtype=float)
            # Delta: if it deserialized as 2D with shape (p, q) where q>1, keep as matrix.
            # If 1D or (p, 1), keep as vector for backward compat with q=1 code paths.
            if Delta.ndim == 2 and Delta.shape[1] == 1:
                Delta = Delta.ravel()
            params.append((mu, Delta, Gamma))
        weights = np.array(fit_dict['weights'], dtype=float)
        return {'component_params': params, 'weights': weights,
                'xlims': fit_dict.get('xlims'),
                'latent_q': fit_dict.get('latent_q', get_q(params))}

    def _sample_supports_pattern(self, sample_idx, obs_dims):
        """Does ``sample_idx`` have at least one variant observed on every
        dim in ``obs_dims``? Used to decide whether that sample's mixing
        weights are identified for variants with this observation pattern.
        """
        if sample_idx is None:
            return False
        sa = self.ms.sample_assignments
        if sample_idx >= sa.shape[1]:
            return False
        dims = list(obs_dims)
        if not dims:
            return False
        scores = self.ms.scores
        has_obs = np.all(~np.isnan(scores[:, dims]), axis=1)
        return bool(np.any(sa[:, sample_idx] & has_obs))

    def _get_benign_weights(self, weights, b_supports=True, s_supports=True):
        """Return benign-class mixing weights, dropping any sample whose
        mixing weights are not identified for the caller's observation
        pattern (``*_supports=False``). Falls back to whichever sample is
        supported; raises if neither is.
        """
        n_w = len(weights)
        b_valid = self.b_idx is not None and self.b_idx < n_w and b_supports
        s_valid = self.s_idx is not None and self.s_idx < n_w and s_supports

        if s_valid and self.benign_method == 'synonymous':
            return weights[self.s_idx]
        if s_valid and b_valid and self.benign_method == 'avg':
            return (np.array(weights[self.b_idx]) + np.array(weights[self.s_idx])) / 2
        if b_valid:
            return weights[self.b_idx]
        if s_valid:
            return weights[self.s_idx]
        raise ValueError(
            f"No supported benign/synonymous sample for this pattern "
            f"(b_idx={self.b_idx}, s_idx={self.s_idx}, weights rows={n_w}, "
            f"b_supports={b_supports}, s_supports={s_supports})"
        )

    def _compute_prior_em(self, params, weights):
        """EM prior estimation using population scores.

        Groups population variants by observation pattern so the
        benign-class weights drop unsupported samples on a per-pattern
        basis (mirrors the LR computation).
        """
        if self.g_idx is None or self.g_idx >= weights.shape[0]:
            return 0.5  # no population sample — neutral prior
        pop_mask = self.ms.sample_assignments[:, self.g_idx]
        pop_scores = self.ms.scores[pop_mask]
        N_pop = pop_scores.shape[0]
        K = len(params)
        w_pop = weights[self.g_idx]

        # ---- pathomechanism correction (see _bootstrap_job for full
        # rationale) -- default on via self.pathomechanism_prior (set by
        # run()'s kwarg of the same name), standard/PU only, computed once
        # rather than per pattern group. ----
        pathomechanism_no_pu_support = False
        f1_full = None
        if (getattr(self, "pathomechanism_prior", True) and self.p_idx is not None
                and not self.nu_mode):
            w_N_anchor = weights[self.g_idx] if self.pu_mode else _get_benign_reference_weights(
                weights, self.benign_method, self.b_idx, self.s_idx)
            labeled_scores = self.ms.scores[self.ms.sample_assignments[:, self.p_idx]]
            corrected, _gamma_hat = compute_pathomechanism_pathogenic_density(
                labeled_scores, pop_scores, params, weights[self.p_idx], w_N_anchor,
                multivariate=True,
            )
            if corrected is None:
                return np.nan  # no assay-relevant excess -- discard this fit
            f1_full = corrected
            if self.pu_mode:
                pathomechanism_no_pu_support = not np.any(f1_full > 1e-300)

        log_f1 = np.full(N_pop, np.nan)  # labeled-class density
        log_f2 = np.full(N_pop, np.nan)  # reference density
        obs_mask = ~np.isnan(pop_scores)
        pat_groups = {}
        for j in range(N_pop):
            key = frozenset(np.where(obs_mask[j])[0])
            if not key:
                continue
            pat_groups.setdefault(key, []).append(j)

        for obs_key, indices in pat_groups.items():
            idx = np.array(indices)
            x_sub = pop_scores[idx]
            if self.nu_mode:
                try:
                    w_b = self._get_benign_weights(weights, b_supports=True, s_supports=True)
                except ValueError:
                    continue
                log_f1[idx] = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                         for c in range(K)], axis=0)
                log_f2[idx] = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                         for c in range(K)], axis=0)
            elif self.pu_mode:
                if f1_full is None:
                    w_p = weights[self.p_idx]
                    log_f1[idx] = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                             for c in range(K)], axis=0)
                log_f2[idx] = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                         for c in range(K)], axis=0)
            else:
                b_sup = self._sample_supports_pattern(self.b_idx, obs_key)
                s_sup = self._sample_supports_pattern(self.s_idx, obs_key)
                try:
                    w_b = self._get_benign_weights(weights, b_supports=b_sup, s_supports=s_sup)
                except ValueError:
                    continue
                if f1_full is None:
                    w_p = weights[self.p_idx]
                    log_f1[idx] = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                             for c in range(K)], axis=0)
                log_f2[idx] = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                         for c in range(K)], axis=0)

        # See _bootstrap_job's identical f1_full handling: valid is keyed off
        # log_f2 either way, so this is a no-op swap when the correction is off.
        if f1_full is not None:
            valid = np.isfinite(log_f2)
            f1_at_pop, f2_at_pop = f1_full[valid], np.exp(log_f2[valid])
        else:
            valid = np.isfinite(log_f1) & np.isfinite(log_f2)
            f1_at_pop, f2_at_pop = np.exp(log_f1[valid]), np.exp(log_f2[valid])

        # Shared with UV's get_fit_prior and this file's _bootstrap_job
        # (src/assay_calibration/fit_utils/point_ranges.py::
        # estimate_prior_from_class_densities). Intentionally NO discard/
        # bounds check on the returned value here (unlike _bootstrap_job's
        # outer 0.001/0.999 gate) -- this call site has never had one; do
        # not add one without a deliberate, separate decision to do so.
        if self.nu_mode:
            return estimate_prior_from_class_densities(f1_at_pop, f2_at_pop, mode='negative_unlabeled')
        if self.pu_mode:
            no_signal = 0.01 if pathomechanism_no_pu_support else 0.1
            return estimate_prior_from_class_densities(
                f1_at_pop, f2_at_pop, mode='positive_unlabeled', no_signal_prior=no_signal)
        return estimate_prior_from_class_densities(f1_at_pop, f2_at_pop, mode='standard')

    def _compute_variant_lr(self, params, weights, prior=None):
        """Compute log LR+ for every variant in ms.scores.

        Groups variants by their observation pattern so the benign-class
        weights are built only from samples that actually have data on
        the variant's observed dimensions.
        """
        K = len(params)
        scores = self.ms.scores
        N, _ = scores.shape
        obs_mask = ~np.isnan(scores)
        w_pop = weights[self.g_idx] if self.g_idx is not None else None

        log_lr = np.full(N, np.nan)
        pat_groups = {}
        for j in range(N):
            key = frozenset(np.where(obs_mask[j])[0])
            if not key:
                continue
            pat_groups.setdefault(key, []).append(j)

        for obs_key, indices in pat_groups.items():
            idx = np.array(indices)
            x_sub = scores[idx]
            if self.nu_mode:
                b_sup = self._sample_supports_pattern(self.b_idx, obs_key)
                s_sup = self._sample_supports_pattern(self.s_idx, obs_key)
                try:
                    w_b = self._get_benign_weights(weights, b_supports=b_sup, s_supports=s_sup)
                except ValueError:
                    continue
                lfb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                 for c in range(K)], axis=0)
                lfpop = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                   for c in range(K)], axis=0)
                _p = prior if prior is not None else 0.1
                fp = np.maximum((np.exp(lfpop) - (1 - _p) * np.exp(lfb)) / _p,
                                np.exp(lfpop) * 1e-10)
                log_lr[idx] = np.log(fp) - lfb
            elif self.pu_mode:
                w_p = weights[self.p_idx]
                lfp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                 for c in range(K)], axis=0)
                lfpop = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                   for c in range(K)], axis=0)
                _p = prior if prior is not None else 0.1
                fb = np.maximum((np.exp(lfpop) - _p * np.exp(lfp)) / (1 - _p),
                                np.exp(lfpop) * 1e-10)
                log_lr[idx] = lfp - np.log(fb)
            else:
                w_p = weights[self.p_idx]
                b_sup = self._sample_supports_pattern(self.b_idx, obs_key)
                s_sup = self._sample_supports_pattern(self.s_idx, obs_key)
                try:
                    w_b = self._get_benign_weights(weights, b_supports=b_sup, s_supports=s_sup)
                except ValueError:
                    continue
                log_fp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                    for c in range(K)], axis=0)
                log_fb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                    for c in range(K)], axis=0)
                log_lr[idx] = log_fp - log_fb
        return log_lr

    def _cap_lr_by_density_fraction(self, lr_matrix, config, density_fraction_threshold=0.05):
        """Cap LR+ to prevent evidence in regions where opposing class has density."""
        r = self.results[config]
        median_prior = r['median_prior']

        rep_fit = None
        best_dist = np.inf
        for bk, bd in self.raw_boots.items():
            fr = bd.get(config)
            if fr is None:
                continue
            inner = fr.get('fit', fr)
            fit = self._reconstitute_params(inner)
            if fit is None:
                continue
            try:
                p = self._compute_prior_em(fit['component_params'], fit['weights'])
                if abs(p - median_prior) < best_dist:
                    best_dist = abs(p - median_prior)
                    rep_fit = fit
            except Exception:
                continue

        if rep_fit is None:
            warnings.warn(f"Could not find representative fit for {config}, skipping density cap")
            return lr_matrix, np.full(lr_matrix.shape[1], 0.5)

        params = rep_fit['component_params']
        weights = rep_fit['weights']
        scores = self.ms.scores
        N, _ = scores.shape
        K = len(params)

        w_pop = weights[self.g_idx] if self.g_idx is not None else None
        log_fp = np.full(N, np.nan)
        log_fb = np.full(N, np.nan)
        obs_mask = ~np.isnan(scores)
        pat_groups = {}
        for j in range(N):
            key = frozenset(np.where(obs_mask[j])[0])
            if not key:
                continue
            pat_groups.setdefault(key, []).append(j)

        for obs_key, indices in pat_groups.items():
            idx = np.array(indices)
            x_sub = scores[idx]
            if self.nu_mode:
                b_sup = self._sample_supports_pattern(self.b_idx, obs_key)
                s_sup = self._sample_supports_pattern(self.s_idx, obs_key)
                try:
                    w_b = self._get_benign_weights(weights, b_supports=b_sup, s_supports=s_sup)
                except ValueError:
                    continue
                lfb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                 for c in range(K)], axis=0)
                lfpop = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                   for c in range(K)], axis=0)
                fp = np.maximum((np.exp(lfpop) - (1 - median_prior) * np.exp(lfb)) / median_prior,
                                np.exp(lfpop) * 1e-10)
                log_fp[idx] = np.log(fp)
                log_fb[idx] = lfb
            elif self.pu_mode:
                w_p = weights[self.p_idx]
                lfp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                 for c in range(K)], axis=0)
                lfpop = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                   for c in range(K)], axis=0)
                fb = np.maximum((np.exp(lfpop) - median_prior * np.exp(lfp)) / (1 - median_prior),
                                np.exp(lfpop) * 1e-10)
                log_fp[idx] = lfp
                log_fb[idx] = np.log(fb)
            else:
                w_p = weights[self.p_idx]
                b_sup = self._sample_supports_pattern(self.b_idx, obs_key)
                s_sup = self._sample_supports_pattern(self.s_idx, obs_key)
                try:
                    w_b = self._get_benign_weights(weights, b_supports=b_sup, s_supports=s_sup)
                except ValueError:
                    continue
                log_fp[idx] = logsumexp([
                    np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                    for c in range(K)
                ], axis=0)
                log_fb[idx] = logsumexp([
                    np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                    for c in range(K)
                ], axis=0)

        log_total = np.logaddexp(log_fp, log_fb)
        path_fraction = np.exp(log_fp - log_total)

        lr_capped = lr_matrix.copy()
        mixed_path = path_fraction > density_fraction_threshold
        lr_capped[:, mixed_path] = np.maximum(lr_capped[:, mixed_path], 0.0)
        mixed_ben = (1 - path_fraction) > density_fraction_threshold
        lr_capped[:, mixed_ben] = np.minimum(lr_capped[:, mixed_ben], 0.0)

        n_path_capped = mixed_path.sum()
        n_ben_capped = mixed_ben.sum()
        n_both = (mixed_path & mixed_ben).sum()
        print(f"  Density cap: {n_path_capped} variants floored (path signal), "
              f"{n_ben_capped} ceilinged (ben signal), {n_both} both (→ no evidence)")

        return lr_capped, path_fraction

    def _reestimate_marginal_weights(self, params, weights, obs_dims):
        """Re-estimate sample-specific component weights using marginal density."""
        obs_dims = np.atleast_1d(obs_dims)
        sa = self.ms.sample_assignments
        scores = self.ms.scores
        K = len(params)
        S = weights.shape[0]

        has_obs = np.all(~np.isnan(scores[:, obs_dims]), axis=1)
        marginal_weights = np.zeros_like(weights)

        for s in range(S):
            s_mask = sa[:, s] & has_obs
            n_s = s_mask.sum()
            if n_s == 0:
                marginal_weights[s] = weights[s]
                continue

            x_s = scores[s_mask]
            log_densities = np.zeros((K, n_s))
            for c in range(K):
                x_marginal = np.full_like(x_s, np.nan)
                x_marginal[:, obs_dims] = x_s[:, obs_dims]
                log_densities[c] = _sn_logpdf(x_marginal, *params[c])

            log_weighted = log_densities + np.log(weights[s] + 1e-300)[:, None]
            log_total = logsumexp(log_weighted, axis=0)
            log_resp = log_weighted - log_total[None, :]
            resp = np.exp(log_resp)
            new_w = resp.mean(axis=1)
            new_w = new_w / new_w.sum()
            marginal_weights[s] = new_w

        return marginal_weights

    def _compute_variant_lr_reestimated(self, params, weights, marginal_weights_by_pattern, prior=None):
        """Compute log LR+ per variant, using re-estimated marginal weights for partials."""
        scores = self.ms.scores
        N, D = scores.shape
        K = len(params)
        log_lr = np.full(N, np.nan)

        obs_mask = ~np.isnan(scores)
        patterns = {}
        for j in range(N):
            key = frozenset(np.where(obs_mask[j])[0])
            patterns.setdefault(key, []).append(j)

        all_dims = frozenset(range(D))

        for obs_key, indices in patterns.items():
            if len(obs_key) == 0:
                continue
            idx = np.array(indices)

            if obs_key == all_dims:
                w = weights
            else:
                w = marginal_weights_by_pattern.get(obs_key, weights)

            x_sub = scores[idx]
            w_pop = w[self.g_idx] if self.g_idx is not None else None
            if self.nu_mode:
                b_sup = self._sample_supports_pattern(self.b_idx, obs_key)
                s_sup = self._sample_supports_pattern(self.s_idx, obs_key)
                try:
                    w_b = self._get_benign_weights(w, b_supports=b_sup, s_supports=s_sup)
                except ValueError:
                    continue
                lfb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                 for c in range(K)], axis=0)
                lfpop = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                   for c in range(K)], axis=0)
                _p = prior if prior is not None else 0.1
                fp = np.maximum((np.exp(lfpop) - (1 - _p) * np.exp(lfb)) / _p,
                                np.exp(lfpop) * 1e-10)
                log_lr[idx] = np.log(fp) - lfb
            elif self.pu_mode:
                w_p = w[self.p_idx]
                lfp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                 for c in range(K)], axis=0)
                lfpop = logsumexp([np.log(w_pop[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                   for c in range(K)], axis=0)
                _p = prior if prior is not None else 0.1
                fb = np.maximum((np.exp(lfpop) - _p * np.exp(lfp)) / (1 - _p),
                                np.exp(lfpop) * 1e-10)
                log_lr[idx] = lfp - np.log(fb)
            else:
                b_sup = self._sample_supports_pattern(self.b_idx, obs_key)
                s_sup = self._sample_supports_pattern(self.s_idx, obs_key)
                w_p = w[self.p_idx]
                try:
                    w_b = self._get_benign_weights(w, b_supports=b_sup, s_supports=s_sup)
                except ValueError:
                    continue
                log_fp = logsumexp([np.log(w_p[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                    for c in range(K)], axis=0)
                log_fb = logsumexp([np.log(w_b[c] + 1e-300) + _sn_logpdf(x_sub, *params[c])
                                    for c in range(K)], axis=0)
                log_lr[idx] = log_fp - log_fb

        return log_lr

    def _process_one_bootstrap(self, fit_raw, partial_patterns, reestimate_marginal_weights):
        """Process a single bootstrap fit.

        Returns (prior, lr_array) on success, or (None, error_str) on failure.
        Safe to call from multiple threads: all operations on self.ms.scores and
        self.ms.sample_assignments are read-only, and scipy/numpy release the GIL.
        """
        if fit_raw is None:
            return None, "fit_raw is None"
        inner = fit_raw.get('fit', fit_raw)
        fit = self._reconstitute_params(inner)
        if fit is None:
            return None, "fit is None"

        params = fit['component_params']
        weights = fit['weights']

        try:
            prior = self._compute_prior_em(params, weights)
        except Exception as e:
            return None, str(e)
        if not np.isfinite(prior) or prior <= 0 or prior >= 1:
            return None, f"invalid prior {prior}"

        try:
            if reestimate_marginal_weights and partial_patterns:
                marginal_w = {}
                for pattern in partial_patterns:
                    obs_dims = np.array(sorted(pattern))
                    marginal_w[pattern] = self._reestimate_marginal_weights(
                        params, weights, obs_dims)
                lr = self._compute_variant_lr_reestimated(params, weights, marginal_w, prior=prior)
            else:
                lr = self._compute_variant_lr(params, weights, prior=prior)
        except Exception as e:
            return None, str(e)

        return (prior, lr), None

    # ──────────────────────────────────────────────────────────────
    # run() — unchanged logic, uses _sn_logpdf throughout
    # ──────────────────────────────────────────────────────────────

    def run(self, min_valid_boots=50,
            cap_by_density=False, density_fraction_threshold=0.05,
            enforce_monotonicity=False, liberal_monotonicity=False,
            path_percentile=5,
            aux_path_percentile=None,
            aux_ben_percentile=None,
            reestimate_marginal_weights=True,
            enforce_marginal_monotonicity=True,
            liberal_marginal_monotonicity=True,
            pathomechanism_prior=True,
            n_jobs=-1):
        """Run analysis for all configs.

        pathomechanism_prior : bool, default True
            Estimate the scalar prior from the pathomechanism-corrected
            pathogenic density (excess over the benign/gnomAD anchor) rather
            than the raw pathogenic-labeled density -- mirrors get_fit_prior's
            pathomechanism_prior default in the univariate pipeline
            (src/assay_calibration/fit_utils/point_ranges.py). Pass False to
            reproduce this file's pre-pathomechanism behavior exactly.
        """
        self.pathomechanism_prior = pathomechanism_prior
        ben_percentile = 100 - path_percentile
        _aux_path_pct = aux_path_percentile if aux_path_percentile is not None else path_percentile
        _aux_ben_pct  = aux_ben_percentile  if aux_ben_percentile  is not None else (100 - _aux_path_pct)
        scores = self.ms.scores
        sa = self.ms.sample_assignments
        N, D = scores.shape

        obs_mask = ~np.isnan(scores)
        all_dims = frozenset(range(D))
        partial_patterns = set()
        for j in range(N):
            key = frozenset(np.where(obs_mask[j])[0])
            if key and key != all_dims:
                partial_patterns.add(key)

        if reestimate_marginal_weights and partial_patterns:
            print(f"\nMarginal weight re-estimation enabled for patterns: "
                  f"{[sorted(p) for p in partial_patterns]}")

        directions = None
        if enforce_monotonicity:
            directions = self._infer_score_directions()
            mode = "liberal" if liberal_monotonicity else "conservative"
            print(f"\nMonotonicity enforcement: {mode}")
            dim_names = getattr(self.ms, 'dataset_names',
                                [f'Dim {d}' for d in range(D)])
            for d, (dir_, name) in enumerate(zip(directions, dim_names)):
                print(f"  {name}: {'higher' if dir_ > 0 else 'lower'} = pathogenic")

        for config in self.configs:
            print(f"\n{'='*50}")
            print(f"Config: {config}")
            print(f"{'='*50}")

            boot_items = list(self.raw_boots.items())
            n_total = len(boot_items)
            scores = self.ms.scores
            sa = self.ms.sample_assignments
            aux_eff_indices = [eff for _, eff in self.aux_p_entries]
            eff_to_fixed = {eff: fixed for fixed, eff in self.aux_p_entries}

            print(f"  Processing {n_total} bootstraps (n_jobs={n_jobs})...")
            if aux_eff_indices:
                print(f"  Aux pathogenic samples (effective indices): {aux_eff_indices}")
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_bootstrap_job)(
                    bd.get(config), scores, sa,
                    self.p_idx, self.b_idx, self.g_idx, self.s_idx,
                    self.benign_method, partial_patterns,
                    reestimate_marginal_weights,
                    extra_p_indices=aux_eff_indices,
                    pathomechanism_prior=pathomechanism_prior,
                )
                for _, bd in boot_items
            )

            priors = []
            lr_matrix = []
            aux_lr_matrices     = {fixed: [] for fixed, _ in self.aux_p_entries}
            aux_vsp_lr_matrices = {fixed: [] for fixed, _ in self.aux_p_entries}
            aux_prior_collections = {fixed: [] for fixed, _ in self.aux_p_entries}
            n_valid = 0
            for result, err in parallel_results:
                if result is None:
                    if err:
                        warnings.warn(err)
                    continue
                prior, lr, aux_lrs = result
                priors.append(prior)
                lr_matrix.append(lr)
                n_valid += 1
                for eff_idx, aux_data in aux_lrs.items():
                    fixed_idx = eff_to_fixed.get(eff_idx)
                    if fixed_idx is not None:
                        aux_lr_matrices[fixed_idx].append(aux_data['lr'])
                        aux_vsp_lr_matrices[fixed_idx].append(aux_data['vsp_lr'])
                        aux_prior_collections[fixed_idx].append(aux_data['prior'])

            print(f"  Valid bootstraps: {n_valid}/{n_total}")

            if n_valid < min_valid_boots:
                print(f"  Skipping: too few valid bootstraps")
                self.results[config] = None
                continue

            priors = np.array(priors)
            lr_matrix = np.array(lr_matrix)

            # ── Prior stability across bootstraps ─────────────────────
            print(f"  Prior across boots: median={np.nanmedian(priors):.4f}, "
                  f"mean={np.nanmean(priors):.4f}, std={np.nanstd(priors):.4f}, "
                  f"5/95={np.nanpercentile(priors, 5):.4f}/"
                  f"{np.nanpercentile(priors, 95):.4f}, "
                  f"min/max={np.nanmin(priors):.4f}/{np.nanmax(priors):.4f}")

            # ── LR+ underflow / non-finite diagnostics ───────────────
            n_nan_lr  = int(np.isnan(lr_matrix).sum())
            n_neg_inf = int(np.isneginf(lr_matrix).sum())
            n_pos_inf = int(np.isposinf(lr_matrix).sum())
            if n_nan_lr or n_neg_inf or n_pos_inf:
                n_var_any  = int((~np.isfinite(lr_matrix)).any(axis=0).sum())
                n_var_all  = int((~np.isfinite(lr_matrix)).all(axis=0).sum())
                print(f"  LR+ matrix non-finite: {n_nan_lr} NaN, "
                      f"{n_neg_inf} -inf, {n_pos_inf} +inf "
                      f"(of {lr_matrix.size} entries)")
                print(f"    variants with any non-finite LR+: {n_var_any}, "
                      f"all non-finite: {n_var_all}")
            # Magnitude of finite extremes — flags log_fp/log_fb underflow
            finite = lr_matrix[np.isfinite(lr_matrix)]
            if finite.size:
                print(f"  LR+ finite range: min={finite.min():.2f}, "
                      f"max={finite.max():.2f} "
                      f"(|LR+|>50: {int((np.abs(finite) > 50).sum())} entries)")

            median_prior = np.nanmedian(priors)
            tau_p_log, tau_b_log, C_path, C_ben = _thresholds(
                median_prior, self.point_values)
            print(f"  LR+ percentiles: path={path_percentile}th, ben={ben_percentile}th")

            lr_matrix_raw = lr_matrix

            path_fraction = None
            if cap_by_density:
                print(f"  Applying density cap (threshold={density_fraction_threshold})...")
                self.results[config] = {'median_prior': median_prior}
                lr_matrix, path_fraction = self._cap_lr_by_density_fraction(
                    lr_matrix, config, density_fraction_threshold)
                n_capped = (lr_matrix != lr_matrix_raw).any(axis=0).sum()
                print(f"  Variants with any LR+ capped: {n_capped}/{N}")

                if self.p_idx is not None:
                    path_mask_check = sa[:, self.p_idx]
                    raw_min = np.nanmin(lr_matrix_raw[:, path_mask_check], axis=0)
                    capped_min = np.nanmin(lr_matrix[:, path_mask_check], axis=0)
                    n_plp_protected = ((raw_min < 0) & (capped_min >= 0)).sum()
                    print(f"  P/LP variants protected from benign LR+: {n_plp_protected}")

            lr_median = np.nanmedian(lr_matrix, axis=0)
            lr_5th = np.nanpercentile(lr_matrix, path_percentile, axis=0)
            lr_95th = np.nanpercentile(lr_matrix, ben_percentile, axis=0)

            if enforce_monotonicity:
                print("  Applying monotonicity enforcement...")
                lr_5th_orig = lr_5th.copy()
                lr_95th_orig = lr_95th.copy()
                lr_median_orig = lr_median.copy()
                lr_5th = self._enforce_monotonicity_mv(
                    lr_5th, liberal=liberal_monotonicity, directions=directions)
                lr_95th = self._enforce_monotonicity_mv(
                    lr_95th, liberal=liberal_monotonicity, directions=directions)
                lr_median = self._enforce_monotonicity_mv(
                    lr_median, liberal=liberal_monotonicity, directions=directions)
                n_changed_5 = np.sum(~np.isclose(lr_5th, lr_5th_orig, atol=1e-6))
                n_changed_95 = np.sum(~np.isclose(lr_95th, lr_95th_orig, atol=1e-6))
                print(f"  Adjusted: {n_changed_5} variants (5th pctl), "
                      f"{n_changed_95} variants (95th pctl)")

            if enforce_marginal_monotonicity:
                directions = self._infer_score_directions()
                mode = "liberal" if liberal_marginal_monotonicity else "conservative"
                print(f"  Applying marginal monotonicity enforcement ({mode})...")
                lr_5th, lr_95th, n_adj = self._enforce_marginal_monotonicity(
                    lr_5th, lr_95th, tau_p_log, tau_b_log,
                    liberal=liberal_marginal_monotonicity,
                    directions=directions)
                print(f"  Marginal monotonicity: {n_adj} partial variants adjusted")

            points = np.zeros(N, dtype=int)
            for pv in self.point_values:
                points[lr_5th >= tau_p_log[pv - 1]] = pv
            for pv in self.point_values:
                points[lr_95th <= tau_b_log[pv - 1]] = -pv

            path_mask = (sa[:, self.p_idx] if self.p_idx is not None
                         else np.zeros(N, bool))
            ben_mask = (sa[:, self.b_idx] if self.b_idx is not None
                        else np.zeros(N, bool))
            syn_mask = (sa[:, self.s_idx] if self.s_idx is not None
                        else np.zeros(N, bool))
            neg_mask = ben_mask | syn_mask

            path_correct = ((points[path_mask] > 0).mean()
                            if path_mask.any() else np.nan)
            path_wrong = ((points[path_mask] < 0).mean()
                          if path_mask.any() else np.nan)
            neg_correct = ((points[neg_mask] < 0).mean()
                           if neg_mask.any() else np.nan)
            neg_wrong = ((points[neg_mask] > 0).mean()
                         if neg_mask.any() else np.nan)

            print(f"  Prior: {median_prior:.4f} "
                  f"[{np.percentile(priors, 5):.4f}, "
                  f"{np.percentile(priors, 95):.4f}]")
            print(f"  C_path: {C_path:.2f}, C_ben: {C_ben:.2f}")
            print(f"  Pathogenic: {path_correct*100:.1f}% correct, "
                  f"{path_wrong*100:.1f}% wrong")
            print(f"  Benign+Syn: {neg_correct*100:.1f}% correct, "
                  f"{neg_wrong*100:.1f}% wrong")
            print(f"  Point distribution: "
                  f"{dict(sorted(Counter(points).items()))}")

            if cap_by_density:
                lr_5th_raw = np.nanpercentile(lr_matrix_raw, path_percentile, axis=0)
                lr_95th_raw = np.nanpercentile(lr_matrix_raw, ben_percentile, axis=0)
                points_raw = np.zeros(N, dtype=int)
                for pv in self.point_values:
                    points_raw[lr_5th_raw >= tau_p_log[pv - 1]] = pv
                for pv in self.point_values:
                    points_raw[lr_95th_raw <= tau_b_log[pv - 1]] = -pv

                changed = points != points_raw
                n_changed = changed.sum()
                fn_before = ((points_raw[path_mask] < 0).sum()
                             if path_mask.any() else 0)
                fn_after = ((points[path_mask] < 0).sum()
                            if path_mask.any() else 0)
                fp_before = ((points_raw[neg_mask] > 0).sum()
                             if neg_mask.any() else 0)
                fp_after = ((points[neg_mask] > 0).sum()
                            if neg_mask.any() else 0)
                print(f"\n  ── Density cap impact ──")
                print(f"  Variants with changed points: {n_changed}")
                print(f"  FN (P/LP→ben): {fn_before} → {fn_after}")
                print(f"  FP (B/LB→path): {fp_before} → {fp_after}")
                print(f"  Uncapped points: {dict(sorted(Counter(points_raw).items()))}")
                print(f"  Capped points:   {dict(sorted(Counter(points).items()))}")

            if reestimate_marginal_weights and partial_patterns:
                is_partial = ~np.all(obs_mask, axis=1)
                n_partial = is_partial.sum()
                partial_path = is_partial & path_mask
                partial_neg = is_partial & neg_mask
                if partial_path.any():
                    p_correct = (points[partial_path] > 0).mean()
                    p_wrong = (points[partial_path] < 0).mean()
                    print(f"\n  ── Partial variants (marginal re-estimated) ──")
                    print(f"  N partial: {n_partial}")
                    print(f"  Partial P/LP: {p_correct*100:.1f}% correct, "
                          f"{p_wrong*100:.1f}% wrong")
                if partial_neg.any():
                    n_correct = (points[partial_neg] < 0).mean()
                    n_wrong = (points[partial_neg] > 0).mean()
                    print(f"  Partial B/LB+Syn: {n_correct*100:.1f}% correct, "
                          f"{n_wrong*100:.1f}% wrong")

            self.results[config] = {
                'priors': priors,
                'median_prior': median_prior,
                'C_path': C_path,
                'C_ben': C_ben,
                'tau_p_log': tau_p_log,
                'tau_b_log': tau_b_log,
                'lr_matrix': lr_matrix,
                'lr_median': lr_median,
                'lr_5th': lr_5th,
                'lr_95th': lr_95th,
                'points': points,
                'n_valid': n_valid,
                'path_correct': path_correct,
                'path_wrong': path_wrong,
                'neg_correct': neg_correct,
                'neg_wrong': neg_wrong,
                'monotonicity_enforced': enforce_monotonicity,
                'monotonicity_liberal': liberal_monotonicity,
                'density_capped': cap_by_density,
                'density_fraction_threshold': density_fraction_threshold,
                'path_percentile': path_percentile,
                'ben_percentile': ben_percentile,
                'marginal_weights_reestimated': reestimate_marginal_weights,
                'marginal_monotonicity_enforced': enforce_marginal_monotonicity,
                'marginal_monotonicity_liberal': liberal_marginal_monotonicity,
                'latent_q': self._latent_q,
            }
            if cap_by_density:
                self.results[config]['lr_matrix_raw'] = lr_matrix_raw
                self.results[config]['points_raw'] = points_raw
                self.results[config]['lr_5th_raw'] = lr_5th_raw
                self.results[config]['lr_95th_raw'] = lr_95th_raw
            elif enforce_monotonicity:
                self.results[config]['lr_5th_raw'] = lr_5th_orig
                self.results[config]['lr_95th_raw'] = lr_95th_orig
                self.results[config]['lr_median_raw'] = lr_median_orig
            if path_fraction is not None:
                self.results[config]['path_fraction'] = path_fraction

            # ── Auxiliary pathogenic results ──────────────────────────
            aux_results = {}
            for fixed_idx, eff_idx in self.aux_p_entries:
                mats = aux_lr_matrices.get(fixed_idx, [])
                if not mats:
                    continue
                aux_lr_mat = np.array(mats)
                aux_lr_5th = np.nanpercentile(aux_lr_mat, _aux_path_pct, axis=0)
                aux_lr_95th = np.nanpercentile(aux_lr_mat, _aux_ben_pct, axis=0)
                aux_lr_median = np.nanmedian(aux_lr_mat, axis=0)

                # Each aux sample has its own disease-specific prior and thresholds
                aux_prior_arr = np.array(aux_prior_collections.get(fixed_idx, []))
                aux_median_prior = float(np.nanmedian(aux_prior_arr)) if len(aux_prior_arr) else 0.5
                aux_tau_p_log, aux_tau_b_log, aux_C_path, aux_C_ben = _thresholds(
                    aux_median_prior, self.point_values)

                if enforce_marginal_monotonicity:
                    aux_lr_5th, aux_lr_95th, _ = self._enforce_marginal_monotonicity(
                        aux_lr_5th, aux_lr_95th, aux_tau_p_log, aux_tau_b_log,
                        liberal=liberal_marginal_monotonicity,
                        directions=self._infer_score_directions())

                aux_points = np.zeros(N, dtype=int)
                for pv in self.point_values:
                    aux_points[aux_lr_5th >= aux_tau_p_log[pv - 1]] = pv
                for pv in self.point_values:
                    aux_points[aux_lr_95th <= aux_tau_b_log[pv - 1]] = -pv

                aux_path_mask = sa[:, eff_idx] if eff_idx < sa.shape[1] else np.zeros(N, bool)
                aux_path_correct = ((aux_points[aux_path_mask] > 0).mean()
                                    if aux_path_mask.any() else np.nan)
                aux_path_wrong = ((aux_points[aux_path_mask] < 0).mean()
                                  if aux_path_mask.any() else np.nan)
                aux_neg_correct = ((aux_points[neg_mask] < 0).mean()
                                   if neg_mask.any() else np.nan)
                aux_neg_wrong   = ((aux_points[neg_mask] > 0).mean()
                                   if neg_mask.any() else np.nan)

                print(f"\n  ── Aux pathogenic sample {fixed_idx} ──")
                print(f"  Percentiles: path={_aux_path_pct}th, ben={_aux_ben_pct}th")
                print(f"  Prior: {aux_median_prior:.4f}  C_path: {aux_C_path:.2f}  C_ben: {aux_C_ben:.2f}")
                print(f"  vs Benign — Path correct: {aux_path_correct*100:.1f}%  wrong: {aux_path_wrong*100:.1f}%")
                print(f"  vs Benign — Ben+Syn correct: {aux_neg_correct*100:.1f}%  wrong: {aux_neg_wrong*100:.1f}%")
                print(f"  vs Benign point dist: {dict(sorted(Counter(aux_points).items()))}")

                # ── Aux vs primary-P comparison ───────────────────────────
                # LR = P(x | aux_P) / P(x | primary_P); use primary thresholds
                vsp_mats = aux_vsp_lr_matrices.get(fixed_idx, [])
                vsp_results = None
                if vsp_mats:
                    vsp_lr_mat    = np.array(vsp_mats)
                    vsp_lr_5th    = np.nanpercentile(vsp_lr_mat, _aux_path_pct, axis=0)
                    vsp_lr_95th   = np.nanpercentile(vsp_lr_mat, _aux_ben_pct,  axis=0)
                    vsp_lr_median = np.nanmedian(vsp_lr_mat, axis=0)

                    if enforce_marginal_monotonicity:
                        vsp_lr_5th, vsp_lr_95th, _ = self._enforce_marginal_monotonicity(
                            vsp_lr_5th, vsp_lr_95th, tau_p_log, tau_b_log,
                            liberal=liberal_marginal_monotonicity,
                            directions=self._infer_score_directions())

                    vsp_points = np.zeros(N, dtype=int)
                    for pv in self.point_values:
                        vsp_points[vsp_lr_5th >= tau_p_log[pv - 1]] = pv
                    for pv in self.point_values:
                        vsp_points[vsp_lr_95th <= tau_b_log[pv - 1]] = -pv

                    # aux variants more like aux_P than primary_P → positive (correct)
                    vsp_aux_correct = ((vsp_points[aux_path_mask] > 0).mean()
                                       if aux_path_mask.any() else np.nan)
                    vsp_aux_wrong   = ((vsp_points[aux_path_mask] < 0).mean()
                                       if aux_path_mask.any() else np.nan)
                    # primary P variants more like primary_P than aux_P → negative
                    vsp_path_neg    = ((vsp_points[path_mask] < 0).mean()
                                       if path_mask.any() else np.nan)
                    vsp_path_pos    = ((vsp_points[path_mask] > 0).mean()
                                       if path_mask.any() else np.nan)

                    print(f"  vs Primary P — aux correct: {vsp_aux_correct*100:.1f}%  "
                          f"wrong: {vsp_aux_wrong*100:.1f}%  |  "
                          f"P/LP neg: {vsp_path_neg*100:.1f}%  pos: {vsp_path_pos*100:.1f}%")
                    print(f"  vs Primary P point dist: "
                          f"{dict(sorted(Counter(vsp_points).items()))}")

                    vsp_results = {
                        'lr_matrix': vsp_lr_mat,
                        'lr_median': vsp_lr_median,
                        'lr_5th':    vsp_lr_5th,
                        'lr_95th':   vsp_lr_95th,
                        'points':    vsp_points,
                        'aux_correct': vsp_aux_correct,
                        'aux_wrong':   vsp_aux_wrong,
                        'path_neg':    vsp_path_neg,
                        'path_pos':    vsp_path_pos,
                    }


                aux_results[fixed_idx] = {
                    'lr_matrix': aux_lr_mat,
                    'lr_median': aux_lr_median,
                    'lr_5th': aux_lr_5th,
                    'lr_95th': aux_lr_95th,
                    'points': aux_points,
                    'priors': aux_prior_arr,
                    'median_prior': aux_median_prior,
                    'C_path': aux_C_path,
                    'C_ben': aux_C_ben,
                    'tau_p_log': aux_tau_p_log,
                    'tau_b_log': aux_tau_b_log,
                    'path_correct': aux_path_correct,
                    'path_wrong': aux_path_wrong,
                    'neg_correct': aux_neg_correct,
                    'neg_wrong': aux_neg_wrong,
                    'path_percentile': _aux_path_pct,
                    'ben_percentile': _aux_ben_pct,
                    'eff_idx': eff_idx,
                    'vsp_results': vsp_results,
                }
            self.results[config]['aux_results'] = aux_results

    def summary_table(self):
        """Return a pandas DataFrame summarizing all configs."""
        rows = []
        for config in self.configs:
            r = self.results.get(config)
            if r is None:
                rows.append({'config': config, 'status': 'failed'})
                continue
            rows.append({
                'config': config,
                'n_boots': r['n_valid'],
                'prior': f"{r['median_prior']:.4f}",
                'C_path': f"{r['C_path']:.1f}",
                'C_ben': f"{r['C_ben']:.1f}",
                'path_correct%': f"{r['path_correct']*100:.1f}",
                'path_wrong%': f"{r['path_wrong']*100:.1f}",
                'neg_correct%': f"{r['neg_correct']*100:.1f}",
                'neg_wrong%': f"{r['neg_wrong']*100:.1f}",
                'point_dist': dict(sorted(Counter(r['points']).items())),
                'latent_q': r.get('latent_q', 1),
            })
        return pd.DataFrame(rows)

    def score_rpv_penetrance(self, config, fixed_idx,
                             rpv_min_points=1, pen_min_points=0):
        """Return a DataFrame of RPV-ness and penetrance scores for every variant.

        Parameters
        ----------
        config          : str   — result config key (e.g. '3c_unc')
        fixed_idx       : int   — sample-assignment column index for the aux/RPV sample
        rpv_min_points  : int   — minimum aux Tavtigian points to be RPV-eligible (default 1 = supporting)
        pen_min_points  : int   — vsp_points threshold separating low_pen_rpv from plp_like
                                  (default 0: requires vsp_points >= 1 for low_pen_rpv;
                                   vsp_points = 0 falls to intermediate)

        Returns
        -------
        pd.DataFrame indexed by variant string with columns:
          vs_B_5th, vs_B_median, vs_B_95th  — log LR+: RPV vs benign
          vs_P_5th, vs_P_median, vs_P_95th  — log LR+: RPV vs primary-P
          rpv_points   — Tavtigian points for vs_B (aux-specific thresholds, conservative)
          vsp_points   — Tavtigian points for vs_P (aux/RPV thresholds, conservative)
          rpv_class    — 'low_pen_rpv' | 'plp_like' | 'benign_like' | 'intermediate'

        Classification rules (applied in priority order):
          benign_like   : rpv_points <= -1
          plp_like      : rpv_points >= rpv_min_points AND vsp_points <= pen_min_points
          low_pen_rpv   : rpv_points >= rpv_min_points AND vsp_points > pen_min_points
          intermediate  : everything else (rpv_points == 0, or ambiguous vsp)
        """
        r  = self.results[config]
        ar = r['aux_results'][fixed_idx]
        vsp = ar.get('vsp_results') or {}
        variants = list(self.ms._variants_kept)
        N = len(variants)

        rpv_pts = ar['points']

        # Recompute vsp_points using aux thresholds (derived from the RPV sample's own
        # prior) rather than the primary prior stored in vsp_results['points'].
        # The primary prior (disease prevalence) is wrong for the RPV-vs-P comparison;
        # the RPV prior better reflects "what fraction of functional variants are RPV-like".
        _aux_tau_p = ar.get('tau_p_log', r['tau_p_log'])
        _aux_tau_b = ar.get('tau_b_log', r['tau_b_log'])
        _path_pct  = ar.get('path_percentile', 5)
        _ben_pct   = ar.get('ben_percentile',  95)
        _vsp_mat   = vsp.get('lr_matrix')
        if _vsp_mat is not None and len(_vsp_mat):
            _vsp_5th  = np.nanpercentile(_vsp_mat, _path_pct, axis=0)
            _vsp_95th = np.nanpercentile(_vsp_mat, _ben_pct,  axis=0)
            vsp_pts = np.zeros(N, dtype=int)
            for pv in self.point_values:
                vsp_pts[_vsp_5th  >= _aux_tau_p[pv - 1]] =  pv
            for pv in self.point_values:
                vsp_pts[_vsp_95th <= _aux_tau_b[pv - 1]] = -pv
        else:
            vsp_pts = vsp.get('points', np.full(N, 0, dtype=int))

        cls = np.full(N, 'intermediate', dtype=object)
        cls[rpv_pts <= -1] = 'benign_like'
        plp_mask = (rpv_pts >= rpv_min_points) & (vsp_pts <= pen_min_points)
        rpv_mask = (rpv_pts >= rpv_min_points) & (vsp_pts >  pen_min_points)
        cls[plp_mask] = 'plp_like'
        cls[rpv_mask] = 'low_pen_rpv'

        # penetrance_score: defined only for rpv_points >= 1 (disease-relevant variants).
        # Scaled 0–1 from vs_P_median across that subset; flipped so that
        # high vs_P (RPV > P, low penetrance) → 0 and low vs_P (P > RPV, high penetrance) → 1.
        vsp_med   = np.asarray(vsp.get('lr_median', np.full(N, np.nan)), dtype=float)
        eligible  = rpv_pts >= 1
        pen_score = np.full(N, np.nan)
        if eligible.sum() > 0:
            vals = vsp_med[eligible]
            vmin = np.nanmin(vals); vmax = np.nanmax(vals)
            if vmax > vmin:
                pen_score[eligible] = 1.0 - (vals - vmin) / (vmax - vmin)

        return pd.DataFrame({
            'vs_B_5th':         ar.get('lr_5th'),
            'vs_B_median':      ar.get('lr_median'),
            'vs_B_95th':        ar.get('lr_95th'),
            'vs_P_5th':         vsp.get('lr_5th'),
            'vs_P_median':      vsp.get('lr_median'),
            'vs_P_95th':        vsp.get('lr_95th'),
            'rpv_points':       rpv_pts,
            'vsp_points':       vsp_pts,
            'rpv_class':        cls,
            'penetrance_score': pen_score,
        }, index=variants)

    # ──────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────

    def plot_comparison(self, figsize=None):
        """4-panel comparison across configs."""
        valid_configs = [c for c in self.configs if self.results.get(c) is not None]
        if not valid_configs:
            print("No valid configs to plot")
            return

        n_configs = len(valid_configs)
        scores = self.ms.scores
        sa = self.ms.sample_assignments
        dataset_names = getattr(self.ms, 'dataset_names', ['Dim 0', 'Dim 1'])

        if figsize is None:
            figsize = (max(5 * n_configs, 14), 18)
        fig = plt.figure(figsize=figsize)

        n_cols = max(n_configs, 2)
        gs = gridspec.GridSpec(3, n_cols, figure=fig, height_ratios=[0.8, 1.2, 0.7],
                               hspace=0.4, wspace=0.35)

        # Row 0, Col 0: Prior distributions
        ax = fig.add_subplot(gs[0, 0])
        for ci, config in enumerate(valid_configs):
            r = self.results[config]
            color = CONFIG_COLORS.get(config, f'C{ci}')
            ax.hist(r['priors'], bins=30, alpha=0.4, color=color,
                    label=f"{CONFIG_LABELS.get(config, config)} (med={r['median_prior']:.3f})",
                    density=True, edgecolor='white', linewidth=0.5)
            ax.axvline(r['median_prior'], color=color, ls='--', lw=1.5)
        ax.set_xlabel('Prior')
        ax.set_ylabel('Density')
        q_label = f" (CFUSN q={self._latent_q})" if self._latent_q > 1 else ""
        ax.set_title(f'Prior Distributions{q_label}', fontweight='bold')
        ax.legend(fontsize=7, framealpha=0.7)
        ax.grid(lw=0.2, alpha=0.3)

        # Row 0, Col 1: Accuracy bar chart
        ax = fig.add_subplot(gs[0, 1])
        x = np.arange(n_configs)
        width = 0.2
        for ci, config in enumerate(valid_configs):
            r = self.results[config]
            color = CONFIG_COLORS.get(config, f'C{ci}')
            ax.bar(ci - width, r['path_correct'] * 100, width, color=color, alpha=0.8,
                   label='Path correct' if ci == 0 else '')
            ax.bar(ci, r['neg_correct'] * 100, width, color=color, alpha=0.5,
                   label='Neg correct' if ci == 0 else '')
            ax.bar(ci + width, r['path_wrong'] * 100, width, color='red', alpha=0.3,
                   label='Path wrong' if ci == 0 else '')
            ax.text(ci - width, r['path_correct'] * 100 + 1,
                    f"{r['path_correct']*100:.0f}", ha='center', fontsize=7)
            ax.text(ci, r['neg_correct'] * 100 + 1,
                    f"{r['neg_correct']*100:.0f}", ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in valid_configs],
                           rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('% Correct Direction')
        ax.set_title('Accuracy by Config', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(lw=0.2, alpha=0.3, axis='y')

        for c_idx in range(2, n_cols):
            fig.add_subplot(gs[0, c_idx]).axis('off')

        # Row 1: 2D point assignment maps per config
        pad = 0.5
        x1r = (np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad)
        x2r = (np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad)

        max_pt = max(self.point_values)
        pt_norm = TwoSlopeNorm(vmin=-max_pt, vcenter=0, vmax=max_pt)

        for ci, config in enumerate(valid_configs):
            ax = fig.add_subplot(gs[1, ci])
            r = self.results[config]

            complete = ~np.isnan(scores).any(axis=1)
            for s_idx in range(sa.shape[1]):
                mask = sa[:, s_idx] & complete
                if not mask.any():
                    continue
                s_pts = r['points'][mask]
                s_sc = scores[mask]
                ax.scatter(s_sc[:, 0], s_sc[:, 1], c=s_pts, cmap=POINT_CMAP, norm=pt_norm,
                           s=12, alpha=0.6,
                           edgecolors=SAMPLE_COLORS[s_idx % len(SAMPLE_COLORS)],
                           linewidths=0.4,
                           marker=SAMPLE_MARKERS[s_idx % len(SAMPLE_MARKERS)])

            label = CONFIG_LABELS.get(config, config)
            q_str = f", q={r.get('latent_q', 1)}" if r.get('latent_q', 1) > 1 else ""
            ax.set_title(f'{label}\nprior={r["median_prior"]:.3f}, n={r["n_valid"]}{q_str}',
                         fontsize=9, fontweight='bold',
                         color=CONFIG_COLORS.get(config, 'black'))
            ax.set_xlabel(dataset_names[0], fontsize=8)
            ax.set_ylabel(dataset_names[1], fontsize=8)
            ax.set_xlim(x1r)
            ax.set_ylim(x2r)
            ax.grid(lw=0.2, alpha=0.3)

        for c_idx in range(n_configs, n_cols):
            fig.add_subplot(gs[1, c_idx]).axis('off')

        # Row 2: Marginal LR+ comparison per dimension
        for dim in range(min(2, n_cols)):
            ax = fig.add_subplot(gs[2, dim])

            for ci, config in enumerate(valid_configs):
                r = self.results[config]
                dim_scores = scores[:, dim]
                valid = ~np.isnan(dim_scores) & np.isfinite(r['lr_median'])
                if not valid.any():
                    continue
                order = np.argsort(dim_scores[valid])
                sorted_scores = dim_scores[valid][order]
                sorted_lr_5 = r['lr_5th'][valid][order]
                sorted_lr_50 = r['lr_median'][valid][order]
                sorted_lr_95 = r['lr_95th'][valid][order]

                win = max(len(sorted_scores) // 50, 5)
                from scipy.ndimage import uniform_filter1d
                smooth_5 = uniform_filter1d(sorted_lr_5.astype(float), win)
                smooth_50 = uniform_filter1d(sorted_lr_50.astype(float), win)
                smooth_95 = uniform_filter1d(sorted_lr_95.astype(float), win)

                color = CONFIG_COLORS.get(config, f'C{ci}')
                label = CONFIG_LABELS.get(config, config)
                ax.plot(sorted_scores, smooth_50, color=color, lw=1.2, label=label)
                ax.fill_between(sorted_scores, smooth_5, smooth_95,
                                color=color, alpha=0.1)

            ax.axhline(0, color='gray', lw=0.8, ls='-', alpha=0.5)
            if valid_configs:
                ref = self.results[valid_configs[0]]
                for pv in [1, 4, max(self.point_values)]:
                    if pv <= max(self.point_values):
                        ax.axhline(ref['tau_p_log'][pv-1], color='red', ls=':', lw=0.5, alpha=0.4)
                        ax.axhline(ref['tau_b_log'][pv-1], color='blue', ls=':', lw=0.5, alpha=0.4)

            ax.set_title(f'Marginal LR+ — {dataset_names[dim]}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Score', fontsize=8)
            ax.set_ylabel('log LR+', fontsize=8)
            ax.legend(fontsize=6, framealpha=0.6)
            ax.grid(lw=0.2, alpha=0.3)

        for c_idx in range(2, n_cols):
            fig.add_subplot(gs[2, c_idx]).axis('off')

        fig.suptitle(f'{self.dataset_name} — Multivariate Calibration Comparison',
                     fontsize=13, fontweight='bold', y=1.01)
        return fig

    def precompute_plot_data(self, config, n_grid=120):
        """Precompute expensive bootstrap data for plot_config_detail.

        Returns a dict that can be passed to render_plot_data (or
        visualize_fit.render_mv_plot_data) repeatedly without redoing
        the parallel bootstrap sweeps.
        """
        from visualize_fit import precompute_mv_plot_data
        return precompute_mv_plot_data(self, config, n_grid=n_grid)

    def render_plot_data(self, precomputed, figsize=None, contour_levels=6,
                         first_row_only=False, max_lr_pairs=10):
        """Render a plot from data returned by precompute_plot_data.

        Cheap — only runs matplotlib drawing, no bootstrap computation.
        """
        from visualize_fit import render_mv_plot_data
        return render_mv_plot_data(precomputed, figsize=figsize,
                                   contour_levels=contour_levels,
                                   first_row_only=first_row_only,
                                   max_lr_pairs=max_lr_pairs)

    def plot_config_detail(self, config, figsize=None, n_grid=120, contour_levels=6,
                           first_row_only=False, max_lr_pairs=10):
        """Detailed visualization using bootstrap-averaged densities.

        Parameters
        ----------
        first_row_only : bool
            If True, render only the top row (2D point-region grids + legend)
            and skip the density-contour and marginal rows.  Much faster when
            you only need a quick look at the evidence map.
        max_lr_pairs : int
            Maximum number of dimension-pair LR+ grids to show in row 0 of the
            D>2 layout.  Defaults to 10.  All C(D,2) pairs are shown in order
            of (dim_i, dim_j), not just pairs anchored to dim 0.
        """
        from visualize_fit import plot_mv_calibration
        return plot_mv_calibration(self, config, figsize=figsize,
                                   n_grid=n_grid, contour_levels=contour_levels,
                                   first_row_only=first_row_only,
                                   max_lr_pairs=max_lr_pairs)


    # ──────────────────────────────────────────────────────────────
    # 2D grid / monotonicity helpers (module-level functions used
    # as static within the class context)
    # ──────────────────────────────────────────────────────────────

    from scipy.ndimage import label as nd_label
    from scipy.interpolate import RegularGridInterpolator


    def _make_lr_grid(params, weights_p, weights_b, x1_range, x2_range, n=150):
        """Evaluate log LR+ on a 2D grid. Uses _sn_logpdf for auto q dispatch."""
        x1g = np.linspace(*x1_range, n)
        x2g = np.linspace(*x2_range, n)
        X1, X2 = np.meshgrid(x1g, x2g, indexing='ij')
        pts = np.column_stack([X1.ravel(), X2.ravel()])
        K = len(params)
        log_fp = logsumexp(
            [np.log(weights_p[c] + 1e-300) + _sn_logpdf(pts, *params[c]) for c in range(K)],
            axis=0
        )
        log_fb = logsumexp(
            [np.log(weights_b[c] + 1e-300) + _sn_logpdf(pts, *params[c]) for c in range(K)],
            axis=0
        )
        lr_grid = (log_fp - log_fb).reshape(n, n)
        return x1g, x2g, lr_grid


    def _connected_component_of_max(binary_grid):
        """Return boolean mask keeping only the largest connected component."""
        from scipy.ndimage import label as nd_label
        labeled, n_comp = nd_label(binary_grid)
        if n_comp == 0:
            return binary_grid.copy()
        comp_sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
        dominant = np.argmax(comp_sizes) + 1
        return labeled == dominant


    def enforce_monotonicity_2d(
        lr_percentile_grid,
        threshold,
        direction,
    ):
        """2D monotonicity: keep only dominant connected component of threshold region."""
        if direction == 'path':
            candidate = lr_percentile_grid >= threshold
        else:
            candidate = lr_percentile_grid <= threshold
        if not candidate.any():
            return candidate
        return _connected_component_of_max(candidate)


    def _build_variant_interpolators(x1g, x2g, valid_grids_path, valid_grids_benign, point_values):
        """Build per-point-value interpolators for monotonicity-enforced regions."""
        from scipy.interpolate import RegularGridInterpolator
        interp_path, interp_benign = {}, {}
        for pv in point_values:
            interp_path[pv] = RegularGridInterpolator(
                (x1g, x2g), valid_grids_path[pv].astype(float),
                method='nearest', bounds_error=False, fill_value=0.0,
            )
            interp_benign[pv] = RegularGridInterpolator(
                (x1g, x2g), valid_grids_benign[pv].astype(float),
                method='nearest', bounds_error=False, fill_value=0.0,
            )
        return interp_path, interp_benign


    def assign_points_monotone_2d(
        scores, lr_5th, lr_95th,
        tau_p_log, tau_b_log, point_values,
        params, weights_p, weights_b,
        lr_5th_grid=None, lr_95th_grid=None,
        x1g=None, x2g=None, n_grid=150,
        x1_range=None, x2_range=None,
    ):
        """Assign discrete points with 2D monotonicity enforcement."""
        N = len(scores)
        has_0 = ~np.isnan(scores[:, 0])
        has_1 = ~np.isnan(scores[:, 1])
        complete = has_0 & has_1

        if lr_5th_grid is None or x1g is None:
            pad = 0.3
            if x1_range is None:
                x1_range = (np.nanmin(scores[:, 0]) - pad, np.nanmax(scores[:, 0]) + pad)
            if x2_range is None:
                x2_range = (np.nanmin(scores[:, 1]) - pad, np.nanmax(scores[:, 1]) + pad)
            x1g, x2g, lr_med_grid = _make_lr_grid(
                params, weights_p, weights_b, x1_range, x2_range, n=n_grid)
            lr_5th_grid = lr_med_grid
            lr_95th_grid = lr_med_grid

        valid_path = {}
        valid_benign = {}
        for pv in point_values:
            vp = enforce_monotonicity_2d(lr_5th_grid, tau_p_log[pv - 1], 'path')
            vb = enforce_monotonicity_2d(lr_95th_grid, tau_b_log[pv - 1], 'benign')
            if pv > point_values[0]:
                prev = point_values[point_values.index(pv) - 1]
                vp = vp & valid_path[prev]
                vb = vb & valid_benign[prev]
            valid_path[pv] = vp
            valid_benign[pv] = vb

        interp_path, interp_benign = _build_variant_interpolators(
            x1g, x2g, valid_path, valid_benign, point_values)

        pts = np.zeros(N, dtype=int)
        for pv in point_values:
            qualify_p = complete & (lr_5th >= tau_p_log[pv - 1])
            qualify_b = complete & (lr_95th <= tau_b_log[pv - 1])
            if qualify_p.any():
                in_region = interp_path[pv](scores[qualify_p]) > 0.5
                idx = np.where(qualify_p)[0][in_region]
                pts[idx] = pv
            if qualify_b.any():
                in_region = interp_benign[pv](scores[qualify_b]) > 0.5
                idx = np.where(qualify_b)[0][in_region]
                pts[idx] = -pv

        only_0 = has_0 & ~has_1
        only_1 = ~has_0 & has_1
        for pv in point_values:
            pts[only_0 & (lr_5th >= tau_p_log[pv - 1])] = pv
            pts[only_0 & (lr_95th <= tau_b_log[pv - 1])] = -pv
            pts[only_1 & (lr_5th >= tau_p_log[pv - 1])] = pv
            pts[only_1 & (lr_95th <= tau_b_log[pv - 1])] = -pv

        return pts, valid_path, valid_benign, x1g, x2g, lr_5th_grid


    # ──────────────────────────────────────────────────────────────
    # PAVA / monotonicity enforcement
    # ──────────────────────────────────────────────────────────────

    import numpy as np
    from collections import Counter

    def _pava_increasing(y, weights=None):
        """Isotonic regression via PAVA enforcing non-decreasing output."""
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n <= 1:
            return y.copy()
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        result = y.copy()
        blocks = []
        for i in range(n):
            blocks.append([result[i] * w[i], w[i], i, i])
            while len(blocks) > 1:
                cur = blocks[-1]
                prev = blocks[-2]
                if prev[0] / prev[1] > cur[0] / cur[1]:
                    prev[0] += cur[0]
                    prev[1] += cur[1]
                    prev[3] = cur[3]
                    blocks.pop()
                else:
                    break
        for wsum, wtot, start, end in blocks:
            result[start:end + 1] = wsum / wtot
        return result

    def _pava_decreasing(y, weights=None):
        """Isotonic regression enforcing non-increasing output."""
        return -_pava_increasing(-np.asarray(y, dtype=float), weights)

    def _infer_score_directions(self):
        """Determine pathogenic direction (+1 or -1) for each score dimension."""
        scores = self.ms.scores
        sa = self.ms.sample_assignments
        K = scores.shape[1]
        path_mask = (sa[:, self.p_idx] if self.p_idx is not None
                     else np.zeros(len(scores), bool))
        ben_mask = sa[:, self.b_idx] if self.b_idx is not None else np.zeros(len(scores), bool)
        if self.s_idx is not None:
            ben_mask = ben_mask | sa[:, self.s_idx]
        directions = np.ones(K)
        for dim in range(K):
            valid_p = path_mask & ~np.isnan(scores[:, dim])
            valid_b = ben_mask & ~np.isnan(scores[:, dim])
            if valid_p.any() and valid_b.any():
                mean_p = np.nanmean(scores[valid_p, dim])
                mean_b = np.nanmean(scores[valid_b, dim])
                directions[dim] = 1.0 if mean_p > mean_b else -1.0
        return directions

    def _enforce_monotonicity_mv(self, lr_values, liberal=False, directions=None):
        """Enforce monotonicity of log-LR+ along each score dimension."""
        scores = self.ms.scores
        N, K = scores.shape
        if directions is None:
            directions = self._infer_score_directions()
        adjusted_per_dim = np.full((K, N), np.nan)

        for dim in range(K):
            valid = ~np.isnan(scores[:, dim])
            n_valid = valid.sum()
            if n_valid < 10:
                adjusted_per_dim[dim, valid] = lr_values[valid]
                continue
            dim_scores = scores[valid, dim] * directions[dim]
            dim_lr = lr_values[valid].copy()
            order = np.argsort(dim_scores)
            sorted_lr = dim_lr[order]
            sorted_path = np.maximum(sorted_lr, 0.0)
            adjusted_path = _pava_increasing(sorted_path)
            sorted_ben = np.maximum(-sorted_lr, 0.0)
            adjusted_ben = _pava_decreasing(sorted_ben)
            if not liberal:
                adjusted_path = _remove_islands_conservative(adjusted_path)
                adjusted_ben = _remove_islands_conservative(adjusted_ben[::-1])[::-1]
            combined = adjusted_path - adjusted_ben
            result = np.empty(n_valid)
            result[order] = combined
            adjusted_per_dim[dim, valid] = result

        if liberal:
            lr_adjusted = np.full(N, np.nan)
            for i in range(N):
                vals = adjusted_per_dim[:, i]
                obs = ~np.isnan(vals)
                if not obs.any():
                    lr_adjusted[i] = lr_values[i]
                else:
                    observed = vals[obs]
                    if np.any(observed > 0):
                        lr_adjusted[i] = np.max(observed)
                    elif np.any(observed < 0):
                        lr_adjusted[i] = np.max(observed)
                    else:
                        lr_adjusted[i] = 0.0
        else:
            lr_adjusted = np.full(N, np.nan)
            for i in range(N):
                vals = adjusted_per_dim[:, i]
                obs = ~np.isnan(vals)
                if not obs.any():
                    lr_adjusted[i] = lr_values[i]
                else:
                    observed = vals[obs]
                    has_path = np.any(observed > 0)
                    has_ben = np.any(observed < 0)
                    if has_path and has_ben:
                        lr_adjusted[i] = 0.0
                    elif has_path:
                        lr_adjusted[i] = np.min(observed[observed > 0])
                    elif has_ben:
                        lr_adjusted[i] = np.max(observed[observed < 0])
                    else:
                        lr_adjusted[i] = 0.0
        return lr_adjusted


    def _remove_islands_conservative(y, min_island_frac=0.02):
        """Remove isolated islands of non-zero evidence in a sorted LR+ array."""
        result = y.copy()
        n = len(result)
        if n == 0:
            return result
        min_size = max(int(n * min_island_frac), 2)
        nonzero = result > 0
        if not nonzero.any():
            return result
        changes = np.diff(nonzero.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if nonzero[0]:
            starts = np.concatenate([[0], starts])
        if nonzero[-1]:
            ends = np.concatenate([ends, [n]])
        for s, e in zip(starts, ends):
            touches_start = (s == 0)
            touches_end = (e == n)
            size = e - s
            if not touches_start and not touches_end:
                result[s:e] = 0.0
            elif size < min_size and not touches_start and not touches_end:
                result[s:e] = 0.0
        return result


    def _assign_points_from_lr(self, lr_values, tau_p_log, tau_b_log):
        """Assign integer evidence points from log-LR+ values."""
        N = len(lr_values)
        pts = np.zeros(N, dtype=int)
        for pv in self.point_values:
            pts[lr_values >= tau_p_log[pv - 1]] = pv
        for pv in self.point_values:
            pts[lr_values <= tau_b_log[pv - 1]] = -pv
        return pts

    def _enforce_marginal_monotonicity(self, lr_5th, lr_95th, tau_p_log, tau_b_log,
                                        liberal=True, directions=None):
        """Enforce monotonicity for partial observations only."""
        scores = self.ms.scores
        N, D = scores.shape
        obs_mask = ~np.isnan(scores)
        complete = obs_mask.all(axis=1)
        if directions is None:
            directions = self._infer_score_directions()
        lr_5th_out = lr_5th.copy()
        lr_95th_out = lr_95th.copy()
        n_adjusted = 0

        for dim in range(D):
            other_dims = [d for d in range(D) if d != dim]
            has_this = obs_mask[:, dim]
            missing_other = np.any(~obs_mask[:, other_dims], axis=1) if other_dims else np.ones(N, bool)
            partial = has_this & missing_other & ~complete
            if partial.sum() < 5:
                continue

            idx = np.where(partial)[0]
            dim_scores = scores[idx, dim] * directions[dim]
            order = np.argsort(dim_scores)
            idx_sorted = idx[order]
            n_sorted = len(idx_sorted)

            pts_path = _assign_path_points(lr_5th[idx_sorted], tau_p_log, self.point_values)
            pts_ben = _assign_ben_points(lr_95th[idx_sorted], tau_b_log, self.point_values)

            all_dim_scores = scores[~np.isnan(scores[:, dim]), dim]
            all_oriented = all_dim_scores * directions[dim]
            oriented_min = np.min(all_oriented)
            oriented_max = np.max(all_oriented)
            range_span = oriented_max - oriented_min
            tail_tol = range_span * 0.05 if range_span > 0 else 0.1
            path_tail_min = oriented_max - tail_tol
            ben_tail_max = oriented_min + tail_tol

            path_ranges = {}
            for pv in self.point_values:
                path_ranges[pv] = _find_runs(pts_path >= pv)
            ben_ranges = {}
            for pv in self.point_values:
                ben_ranges[-pv] = _find_runs(pts_ben <= -pv)

            sorted_oriented = dim_scores[order]
            _enforce_monotonicity_ranges(
                path_ranges, ben_ranges, self.point_values,
                n_sorted, liberal=liberal,
                sorted_scores=sorted_oriented,
                path_tail_min=path_tail_min,
                ben_tail_max=ben_tail_max)

            pts_path_clean = np.zeros(n_sorted, dtype=int)
            for pv in self.point_values:
                for start, end in path_ranges[pv]:
                    pts_path_clean[start:end] = np.maximum(pts_path_clean[start:end], pv)
            pts_ben_clean = np.zeros(n_sorted, dtype=int)
            for pv in self.point_values:
                for start, end in ben_ranges[-pv]:
                    pts_ben_clean[start:end] = np.minimum(pts_ben_clean[start:end], -pv)

            for i in range(n_sorted):
                vi = idx_sorted[i]
                changed = False
                if pts_path_clean[i] < pts_path[i]:
                    if pts_path_clean[i] == 0:
                        lr_5th_out[vi] = min(lr_5th_out[vi], tau_p_log[0] - 1e-6)
                    else:
                        next_pv = pts_path_clean[i]
                        if next_pv < len(tau_p_log):
                            lr_5th_out[vi] = min(lr_5th_out[vi], tau_p_log[next_pv] - 1e-6)
                    changed = True
                if abs(pts_ben_clean[i]) < abs(pts_ben[i]):
                    if pts_ben_clean[i] == 0:
                        lr_95th_out[vi] = max(lr_95th_out[vi], tau_b_log[0] + 1e-6)
                    else:
                        next_pv = abs(pts_ben_clean[i])
                        if next_pv < len(tau_b_log):
                            lr_95th_out[vi] = max(lr_95th_out[vi], tau_b_log[next_pv] + 1e-6)
                    changed = True
                if changed:
                    n_adjusted += 1

        return lr_5th_out, lr_95th_out, n_adjusted

    @staticmethod
    def _points_from_lr(lr_vals, tau_p_log):
        pts = np.zeros(len(lr_vals), dtype=int)
        for pv_idx, tau in enumerate(tau_p_log):
            pts[lr_vals >= tau] = pv_idx + 1
        return pts

    @staticmethod
    def _points_from_lr_ben(lr_vals, tau_b_log):
        pts = np.zeros(len(lr_vals), dtype=int)
        for pv_idx, tau in enumerate(tau_b_log):
            pts[lr_vals <= tau] = -(pv_idx + 1)
        return pts


# ──────────────────────────────────────────────────────────────
# Static helpers (outside the class)
# ──────────────────────────────────────────────────────────────

def _assign_path_points(lr_vals, tau_p_log, point_values):
    pts = np.zeros(len(lr_vals), dtype=int)
    for pv in point_values:
        pts[lr_vals >= tau_p_log[pv - 1]] = pv
    return pts


def _assign_ben_points(lr_vals, tau_b_log, point_values):
    pts = np.zeros(len(lr_vals), dtype=int)
    for pv in point_values:
        pts[lr_vals <= tau_b_log[pv - 1]] = -pv
    return pts


def _find_runs(bool_array):
    runs = []
    n = len(bool_array)
    i = 0
    while i < n:
        if bool_array[i]:
            start = i
            while i < n and bool_array[i]:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    return runs


def _enforce_monotonicity_ranges(path_ranges, ben_ranges, point_values,
                                  n_sorted, liberal=True,
                                  sorted_scores=None,
                                  path_tail_min=None,
                                  ben_tail_max=None):
    """Enforce monotonicity on point ranges, matching the univariate logic."""

    def _run_touches_path_tail(run):
        if sorted_scores is None:
            return run[1] == n_sorted
        last_score = sorted_scores[min(run[1] - 1, n_sorted - 1)]
        return last_score >= path_tail_min

    def _run_touches_ben_tail(run):
        if sorted_scores is None:
            return run[0] == 0
        first_score = sorted_scores[run[0]]
        return first_score <= ben_tail_max

    if liberal:
        for pv in point_values:
            if path_ranges[pv]:
                path_ranges[pv] = [path_ranges[pv][-1]]
        for pv in point_values:
            if ben_ranges[-pv]:
                ben_ranges[-pv] = [ben_ranges[-pv][0]]
        for i_idx, i in enumerate(point_values):
            for j in point_values:
                if j <= i:
                    continue
                if path_ranges[i] and path_ranges[j]:
                    ri = path_ranges[i][0]
                    rj = path_ranges[j][0]
                    if rj[0] <= ri[0] and rj[1] >= ri[1]:
                        path_ranges[i] = []
                if ben_ranges[-i] and ben_ranges[-j]:
                    ri = ben_ranges[-i][0]
                    rj = ben_ranges[-j][0]
                    if rj[0] <= ri[0] and rj[1] >= ri[1]:
                        ben_ranges[-i] = []
    else:
        max_path_killed = None
        max_ben_killed = None
        for pv in point_values:
            if max_path_killed is not None:
                path_ranges[pv] = []
            elif path_ranges[pv]:
                touches_tail = any(_run_touches_path_tail(r) for r in path_ranges[pv])
                if not touches_tail:
                    path_ranges[pv] = []
                    max_path_killed = pv
                    continue
                if pv == 1 and len(path_ranges[pv]) > 0:
                    next_pv = 2 if 2 in path_ranges else None
                    has_higher = next_pv is not None and len(path_ranges.get(next_pv, [])) > 0
                    if not has_higher:
                        if not any(_run_touches_path_tail(r) for r in path_ranges[pv]) and \
                           not any(_run_touches_ben_tail(r) for r in path_ranges[pv]):
                            path_ranges[pv] = []
                            max_path_killed = pv
                            continue
                if len(path_ranges[pv]) > 1:
                    next_pv = pv + 1
                    if next_pv in path_ranges and path_ranges[next_pv]:
                        higher_start = path_ranges[next_pv][0][0]
                        higher_end = path_ranges[next_pv][-1][1]
                        keep = []
                        for r in path_ranges[pv]:
                            if r[1] == higher_start or r[0] == higher_end:
                                keep.append(r)
                        if not keep:
                            path_ranges[pv] = []
                            max_path_killed = pv
                        else:
                            path_ranges[pv] = keep
                    if len(path_ranges[pv]) > 1:
                        path_ranges[pv] = [(path_ranges[pv][0][0],
                                            path_ranges[pv][-1][1])]

            if max_ben_killed is not None:
                ben_ranges[-pv] = []
            elif ben_ranges[-pv]:
                touches_tail = any(_run_touches_ben_tail(r) for r in ben_ranges[-pv])
                if not touches_tail:
                    ben_ranges[-pv] = []
                    max_ben_killed = -pv
                    continue
                if pv == 1 and len(ben_ranges[-pv]) > 0:
                    next_pv = -2
                    has_higher = len(ben_ranges.get(next_pv, [])) > 0
                    if not has_higher:
                        if not any(_run_touches_ben_tail(r) for r in ben_ranges[-pv]) and \
                           not any(_run_touches_path_tail(r) for r in ben_ranges[-pv]):
                            ben_ranges[-pv] = []
                            max_ben_killed = -pv
                            continue
                if len(ben_ranges[-pv]) > 1:
                    next_pv = -(pv + 1)
                    if next_pv in ben_ranges and ben_ranges[next_pv]:
                        higher_start = ben_ranges[next_pv][0][0]
                        higher_end = ben_ranges[next_pv][-1][1]
                        keep = []
                        for r in ben_ranges[-pv]:
                            if r[1] == higher_start or r[0] == higher_end:
                                keep.append(r)
                        if not keep:
                            ben_ranges[-pv] = []
                            max_ben_killed = -pv
                        else:
                            ben_ranges[-pv] = keep
                    if len(ben_ranges[-pv]) > 1:
                        ben_ranges[-pv] = [(ben_ranges[-pv][0][0],
                                            ben_ranges[-pv][-1][1])]