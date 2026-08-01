"""Batched JAX initialization for skew-normal mixture EM.

Replaces the sequential CPU init loops in interop.py (kmeans + MoM) with
fully batched JAX implementations.  Calling batch_init_univariate /
batch_init_cfusn once for a whole chunk eliminates the Python for-loop over
individual fit jobs that was responsible for 97-99% of total wall time.

K, S, p, q are static Python ints at JIT trace time (one chunk = one
(dataset, num_components) group, so these are constant across the batch).
Any K >= 2 is supported — there is no upper limit; XLA unrolls Python `for`
loops at trace time.

jax_enable_x64 must be set before importing this module (done in interop.py).
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax

from .constraints_jax import constraint_violated, build_grid
from .batch_em import _e_step_responsibilities
from .batch_em_cfusn import _densities_all_components, _responsibilities_from_logpdfs


# ─────────────────────────────────────────────────────────────────────────────
# Univariate init
# ─────────────────────────────────────────────────────────────────────────────

def _kmeans_1d(obs, K, n_lloyd=20):
    """obs: (batch, N) → locs (batch, K), scales (batch, K).  Deterministic.

    Initializes K centers at evenly-spaced quantile positions of sorted_obs,
    then runs n_lloyd Lloyd iterations via lax.fori_loop (not a Python for loop)
    so XLA compiles a single loop body rather than 20 unrolled copies — critical
    for keeping JIT compilation time under a minute at large batch sizes.
    """
    batch, N = obs.shape
    sorted_obs = jnp.sort(obs, axis=-1)
    idxs = jnp.array([int(N * (k + 0.5) / K) for k in range(K)], dtype=jnp.int32)
    centers = sorted_obs[:, idxs]               # (batch, K)
    scales = jnp.ones((batch, K), dtype=obs.dtype)

    def lloyd_step(_, carry):
        centers_c, scales_c = carry
        dists = jnp.abs(obs[:, :, None] - centers_c[:, None, :])  # (batch, N, K)
        asgn = jnp.argmin(dists, axis=-1)                          # (batch, N)
        # Vectorised update over K clusters using one-hot masks
        asgn_oh = jax.nn.one_hot(asgn, K, dtype=obs.dtype)         # (batch, N, K)
        n_k = jnp.maximum(asgn_oh.sum(1), 1.0)                     # (batch, K)
        new_centers = (obs[:, :, None] * asgn_oh).sum(1) / n_k     # (batch, K)
        dev = obs[:, :, None] - new_centers[:, None, :]             # (batch, N, K)
        var_k = (dev ** 2 * asgn_oh).sum(1) / jnp.maximum(n_k - 1.0, 1.0)
        new_scales = jnp.maximum(jnp.sqrt(var_k), 1e-6)
        return new_centers, new_scales

    centers, scales = lax.fori_loop(0, n_lloyd, lloyd_step, (centers, scales))

    sort_idx = jnp.argsort(centers, axis=-1)
    locs = jnp.take_along_axis(centers, sort_idx, axis=-1)
    scales = jnp.take_along_axis(scales, sort_idx, axis=-1)
    return locs, scales


def _lambda_sign_batch(fit_idx, K):
    """(batch,) int fit_idx -> (batch, K) deterministic +/-1 skew signs,
    enumerating all 2**K sign patterns in fit_idx order.

    GPU counterpart of initializations._lambda_signs's lambdaIndex path
    (cfusn/initializations.py:27-38): lambdaIndex = fit_idx % 2**K, then bit
    `k` of lambdaIndex (MSB-first, i.e. bit position K-1-k) gives component
    k's sign -- this ordering is chosen to match
    itertools.product([-1, 1], repeat=K)'s enumeration exactly, so restart i
    gets the identical sign pattern on GPU and CPU (values still differ --
    magnitude is still an independent random draw each restart, see callers
    below, and interop.py's per-chunk shared PRNGKey vs. CPU's per-restart
    RandomState seeding means the literal magnitude numbers won't match --
    but the sign *pattern* and its period-2**K cycling do).
    """
    n_patterns = 2 ** K
    lam_idx = jnp.mod(fit_idx, n_patterns).astype(jnp.int32)      # (batch,)
    bit_positions = jnp.arange(K - 1, -1, -1)                     # MSB..LSB
    bits = (lam_idx[:, None] >> bit_positions[None, :]) & 1       # (batch, K)
    return 2.0 * bits.astype(jnp.float64) - 1.0


def _mom_1d(obs, K, key, fit_idx):
    """obs: (batch, N) → a, locs, scales (batch, K), valid (batch,) bool.

    Quantile-split MoM: evenly-spaced percentile cut-points, then
    sn_method_of_moments_init formula per slice.  valid[b] is True iff all K
    slices were large enough and produced finite params.
    """
    batch, N = obs.shape
    sorted_obs = jnp.sort(obs, axis=-1)
    cutpoints = [int(N * (k + 1) / K) for k in range(K - 1)]
    min_size = max(10, int(0.05 * N))

    a1 = jnp.sqrt(2.0 / jnp.pi)
    c_coef = (4.0 - jnp.pi) / 2.0
    key_fb = key

    a_list, loc_list, scale_list, ok_list = [], [], [], []
    for k in range(K):
        lo = 0 if k == 0 else cutpoints[k - 1]
        hi = N if k == K - 1 else cutpoints[k]
        n_slice = hi - lo
        static_ok = n_slice >= min_size     # baked into trace at compile time

        sl = sorted_obs[:, lo:hi]           # (batch, n_slice)
        m1 = sl.mean(-1)
        centered = sl - m1[:, None]
        m2 = jnp.maximum((centered ** 2).mean(-1), 1e-10)
        std = jnp.sqrt(m2)
        m3_norm = (centered ** 3).mean(-1) / jnp.maximum(std ** 3, 1e-15)

        abs_m3 = jnp.abs(m3_norm)
        ratio_23 = (c_coef / jnp.maximum(abs_m3, 1e-10)) ** (2.0 / 3.0)
        delta = jnp.sign(m3_norm) / jnp.sqrt(jnp.maximum(a1 ** 2 * (1.0 + ratio_23), 1e-10))
        delta = jnp.clip(delta, -0.98, 0.98)

        sigma = jnp.maximum(
            jnp.sqrt(m2 / jnp.maximum(1.0 - a1 ** 2 * delta ** 2, 1e-10)), 1e-6
        )
        mu_sn = m1 - a1 * delta * sigma
        alpha = delta / jnp.sqrt(jnp.maximum(1.0 - delta ** 2, 1e-12))

        # Fallback when skewness is near-zero
        fallback_a = jax.random.uniform(key_fb, (batch,), dtype=jnp.float64,
                                         minval=-0.25, maxval=0.25)
        near_zero = abs_m3 < 1e-10
        alpha = jnp.where(near_zero, fallback_a, alpha)
        mu_sn = jnp.where(near_zero, m1, mu_sn)
        sigma = jnp.where(near_zero, jnp.maximum(std, 1e-6), sigma)

        finite = jnp.isfinite(alpha) & jnp.isfinite(mu_sn) & jnp.isfinite(sigma)
        if static_ok:
            ok_list.append(finite)
        else:
            ok_list.append(jnp.zeros(batch, dtype=jnp.bool_))

        a_list.append(alpha)
        loc_list.append(mu_sn)
        scale_list.append(sigma)

    a_mom = jnp.stack(a_list, axis=-1)      # (batch, K)
    loc_mom = jnp.stack(loc_list, axis=-1)
    scale_mom = jnp.stack(scale_list, axis=-1)
    valid = jnp.stack(ok_list, axis=-1).all(-1)  # (batch,)

    # Override a-signs with the enumerated lambdaIndex pattern (mirrors
    # methodOfMomentsInit's `params[0] = lambdas[i] * abs(params[0])`,
    # initializations.py:143-147) -- magnitude stays the MoM-estimated
    # |alpha|, unchanged, only the sign is replaced.
    signs = _lambda_sign_batch(fit_idx, K)
    a_mom = signs * jnp.abs(a_mom)

    return a_mom, loc_mom, scale_mom, valid


def _fix_constraint(a, locs, scales, xmin, xmax, K, max_iters=100):
    """Shrink scales (per violated batch element) until constraint satisfied.

    a/locs/scales: (batch, K); xmin/xmax: (batch,) -> scales: (batch, K).
    Uses fori_loop (fixed iteration count) to avoid device-host sync on each
    step — mirrors the binary_search implementation in constraints_jax.py.
    """
    grid = build_grid(xmin, xmax)           # (batch, 1000)

    def body(itr, scales_c):
        params = [(a[:, k], locs[:, k], scales_c[:, k]) for k in range(K)]
        violated = constraint_violated(params, grid)    # (batch,)
        above_floor = scales_c.min(-1) > 1e-6          # (batch,)
        shrink = violated & above_floor
        return jnp.where(shrink[:, None], scales_c * 0.95, scales_c)

    return lax.fori_loop(0, max_iters, body, scales)


def _initial_weights_uv(obs, sample_idx, a0, loc0, scale0, S):
    """One E-step from uniform prior -> W (batch, S, K)."""
    batch, K = a0.shape
    W_uni = jnp.ones((batch, S, K), dtype=obs.dtype) / K
    resp = _e_step_responsibilities(obs, sample_idx, a0, loc0, scale0, W_uni)
    resp_bnk = jnp.moveaxis(resp, 1, 2)    # (batch, N, K)
    W_rows = []
    for s in range(S):
        mask = (sample_idx == s).astype(obs.dtype)
        n_s = jnp.maximum(mask.sum(-1, keepdims=True), 1.0)
        W_rows.append((resp_bnk * mask[:, :, None]).sum(1) / n_s)
    return jnp.stack(W_rows, axis=1)       # (batch, S, K)


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def batch_init_univariate(obs, sample_idx, S, K, constrained, xmin, xmax, key, fit_idx):
    """Batched univariate init: k-means + MoM (with k-means fallback) + W.

    obs: (batch, N); sample_idx: (batch, N) int; xmin/xmax: (batch,);
    key: a single JAX PRNGKey for this batch (created from fit_seeds in
    interop.py) used only for magnitude draws now; fit_idx: (batch,) int,
    each restart's index within its (dataset, num_components) job list --
    drives the deterministic lambdaIndex-style skew SIGN enumeration
    (_lambda_sign_batch) that replaces the old random-sign draw, matching
    the CPU path's lambdaIndex behavior (cfusn/initializations.py:27-38).

    Returns a0, loc0, scale0: (batch, K);  W0: (batch, S, K);
            init_failed: (batch,) bool (True when constraint cannot be satisfied).
    """
    batch = obs.shape[0]
    key_km_mag, key_mom = jax.random.split(key)

    # Deterministic k-means + enumerated (lambdaIndex-style) a signs
    locs_km, scales_km = _kmeans_1d(obs, K)
    signs_km = _lambda_sign_batch(fit_idx, K)
    mags_km = jax.random.uniform(key_km_mag, (batch, K), dtype=jnp.float64, minval=0.0, maxval=0.25)
    a_km = signs_km * mags_km

    # Method of moments (falls back to k-means when slices too small / MoM fails)
    a_mom, locs_mom, scales_mom, mom_valid = _mom_1d(obs, K, key_mom, fit_idx)

    a0 = jnp.where(mom_valid[:, None], a_mom, a_km)
    loc0 = jnp.where(mom_valid[:, None], locs_mom, locs_km)
    scale0 = jnp.where(mom_valid[:, None], scales_mom, scales_km)

    # Equalise scales to max per fit (mirrors methodOfMomentsInit's max_scale trick)
    max_scale = scale0.max(-1, keepdims=True)
    scale0 = jnp.where(mom_valid[:, None], max_scale * jnp.ones_like(scale0), scale0)

    if constrained:
        scale0 = _fix_constraint(a0, loc0, scale0, xmin, xmax, K)

    W0 = _initial_weights_uv(obs, sample_idx, a0, loc0, scale0, S)

    init_failed = constrained & (scale0.min(-1) < 1e-6)
    return a0, loc0, scale0, W0, init_failed


# ─────────────────────────────────────────────────────────────────────────────
# CFUSN (multivariate) init
# ─────────────────────────────────────────────────────────────────────────────

def _kmeans_mv(obs, obs_mask, K, n_lloyd=20):
    """obs: (batch, N, p); obs_mask: (batch, N, p) bool ->
    centers (batch, K, p), asgn (batch, N).  Deterministic.

    Missing values are imputed with column means for distance computation;
    NaN-aware means are used for cluster center updates.
    """
    batch, N, p = obs.shape
    # Column-mean imputation (NaN-aware)
    col_sum = (obs * obs_mask).sum(1)
    col_cnt = jnp.maximum(obs_mask.astype(obs.dtype).sum(1), 1.0)
    obs_imp = jnp.where(obs_mask, obs, (col_sum / col_cnt)[:, None, :])  # (batch, N, p)

    # Quantile-based init: sort by first dimension, pick K evenly-spaced rows
    sort_r = jnp.argsort(obs_imp[:, :, 0], axis=-1)                      # (batch, N)
    ci = jnp.array([int(N * (k + 0.5) / K) for k in range(K)], dtype=jnp.int32)
    centers = obs_imp[jnp.arange(batch)[:, None], sort_r[:, ci], :]      # (batch, K, p)

    def lloyd_step_mv(_, centers_c):
        diff = obs_imp[:, :, None, :] - centers_c[:, None, :, :]          # (batch, N, K, p)
        asgn = jnp.argmin((diff ** 2).sum(-1), axis=-1)                   # (batch, N)
        asgn_oh = jax.nn.one_hot(asgn, K, dtype=obs.dtype)                # (batch, N, K)
        m_k = obs_mask[:, :, None, :] & (asgn_oh[:, :, :, None] > 0)     # (batch, N, K, p)
        n_k = jnp.maximum(m_k.astype(obs.dtype).sum(1), 1.0)             # (batch, K, p)
        c_k = (obs[:, :, None, :] * m_k).sum(1) / n_k                    # (batch, K, p)
        has_members = (asgn_oh.sum(1) > 0)[:, :, None]                   # (batch, K, 1)
        return jnp.where(has_members, c_k, centers_c)

    centers = lax.fori_loop(0, n_lloyd, lloyd_step_mv, centers)

    diff = obs_imp[:, :, None, :] - centers[:, None, :, :]
    asgn = jnp.argmin((diff ** 2).sum(-1), axis=-1)
    return centers, asgn


def _nan_aware_cov(obs, obs_mask, in_cluster, p):
    """NaN-aware pairwise covariance for observations in `in_cluster`.

    obs: (batch, N, p); obs_mask: (batch, N, p); in_cluster: (batch, N) bool.
    Returns (batch, p, p) with 1e-6*I regularisation and PD correction.
    The double loop over (d1, d2) pairs unrolls at trace time; for p<=4 that
    is at most 16 iterations.
    """
    batch = obs.shape[0]
    rows = []
    for d1 in range(p):
        row = []
        for d2 in range(p):
            both = obs_mask[:, :, d1] & obs_mask[:, :, d2] & in_cluster  # (batch, N)
            n_b = jnp.maximum(both.astype(obs.dtype).sum(-1), 1.0)
            x1 = obs[:, :, d1]
            x2 = obs[:, :, d2]
            mu1 = (x1 * both).sum(-1) / n_b
            mu2 = (x2 * both).sum(-1) / n_b
            r1 = (x1 - mu1[:, None]) * both
            r2 = (x2 - mu2[:, None]) * both
            cov_val = (r1 * r2).sum(-1) / jnp.maximum(n_b - 1.0, 1.0)
            cov_val = jnp.where(n_b >= 2.0, cov_val, 1e-2)
            row.append(cov_val)
        rows.append(jnp.stack(row, axis=-1))    # (batch, p)
    cov = jnp.stack(rows, axis=-2)              # (batch, p, p) -- rows x cols

    cov = cov + 1e-6 * jnp.eye(p)[None, :, :]
    # PD correction
    min_ev = jnp.linalg.eigvalsh(cov).min(-1)  # (batch,)
    corr = jnp.maximum(-min_ev + 1e-8, 0.0)
    cov = cov + corr[:, None, None] * jnp.eye(p)[None, :, :]
    return cov


def _init_delta_matrix_jax(cov, obs, obs_mask, in_cluster, q, key):
    """Delta (batch, p, q) from top-q eigenvectors of cov.

    Skewness signs are derived from cluster data when enough complete rows
    exist, then multiplied by an enumerated sign from `key`.  A small noise
    term is added; the result is shrunk until Gamma = cov - Delta@Delta.T is PD.
    """
    batch, p, _ = cov.shape
    eigvals, eigvecs = jnp.linalg.eigh(cov)    # ascending: (batch,p), (batch,p,p)

    Delta = jnp.zeros((batch, p, q), dtype=cov.dtype)
    for j in range(q):
        idx = p - 1 - j                         # index of j-th largest eigenvalue
        ev = eigvecs[:, :, idx]                 # (batch, p)
        el = eigvals[:, idx]                    # (batch,)
        scale_j = 0.1 * jnp.sqrt(jnp.maximum(el, 0.0))  # (batch,)

        # Marginal skewness of cluster data projected onto eigenvector j
        complete = (~jnp.isnan(obs).any(-1)) & in_cluster  # (batch, N)
        proj = (obs * ev[:, None, :]).sum(-1)               # (batch, N)
        n_proj = jnp.maximum(complete.astype(cov.dtype).sum(-1), 1.0)
        pmean = (proj * complete).sum(-1) / n_proj
        cent = (proj - pmean[:, None]) * complete
        m2 = (cent ** 2).sum(-1) / jnp.maximum(n_proj - 1.0, 1.0)
        m3 = (cent ** 3).sum(-1) / jnp.maximum(n_proj - 1.0, 1.0)
        skew = m3 / jnp.maximum(m2 ** 1.5, 1e-12)
        skew_sign = jnp.where(jnp.abs(skew) < 1e-6, 1.0, jnp.sign(skew).astype(cov.dtype))
        skew_sign = jnp.where(n_proj < 8.0, 1.0, skew_sign)

        # Enumerated sign from key (fold_in with j so each direction gets its own key)
        key_j = jax.random.fold_in(key, j)
        enum_sign = 2.0 * jax.random.randint(key_j, (batch,), 0, 2).astype(cov.dtype) - 1.0

        col_j = (skew_sign * enum_sign * scale_j)[:, None] * ev  # (batch, p)
        Delta = Delta.at[:, :, j].set(col_j)

    # Small noise
    key_noise = jax.random.fold_in(key, q)
    diag_sc = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))  # (batch, p)
    noise = jax.random.uniform(key_noise, (batch, p, q), dtype=cov.dtype, minval=-0.05, maxval=0.05)
    Delta = Delta + noise * diag_sc[:, :, None]

    # Shrink Delta until Gamma = cov - Delta @ Delta.T is PD
    def shrink_body(_, Delta_c):
        Gamma_c = cov - jnp.matmul(Delta_c, jnp.swapaxes(Delta_c, -1, -2))
        needs = jnp.linalg.eigvalsh(Gamma_c).min(-1) < 1e-6   # (batch,)
        return jnp.where(needs[:, None, None], Delta_c * 0.5, Delta_c)

    Delta = lax.fori_loop(0, 20, shrink_body, Delta)
    return Delta


def _initial_weights_mv(obs, obs_mask, sample_idx, mu0, Delta0, Gamma0, S):
    """One CFUSN E-step from uniform prior -> W (batch, S, K)."""
    batch, K = mu0.shape[:2]
    W_uni = jnp.ones((batch, S, K), dtype=obs.dtype) / K
    log_pdfs, _, _ = _densities_all_components(obs, obs_mask, mu0, Delta0, Gamma0)
    P = _responsibilities_from_logpdfs(log_pdfs, sample_idx, W_uni)    # (batch, N, K)
    W_rows = []
    for s in range(S):
        mask = (sample_idx == s).astype(obs.dtype)
        n_s = jnp.maximum(mask.sum(-1, keepdims=True), 1.0)
        W_rows.append((P * mask[:, :, None]).sum(1) / n_s)
    return jnp.stack(W_rows, axis=1)


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def batch_init_cfusn(obs, obs_mask, sample_idx, S, K, q, key):
    """Batched CFUSN init (unconstrained, non-anchored only).

    obs: (batch, N, p); obs_mask: (batch, N, p) bool; sample_idx: (batch, N);
    key: JAX PRNGKey for this batch.  q must equal 2.

    Returns mu0 (batch,K,p), Delta0 (batch,K,p,q), Gamma0 (batch,K,p,p),
            W0 (batch,S,K), init_failed (batch,) bool.
    """
    assert q == 2, "batch_init_cfusn only supports q=2"
    batch, N, p = obs.shape

    # Global covariance (fallback for small clusters)
    global_cov = _nan_aware_cov(obs, obs_mask, jnp.ones((batch, N), dtype=jnp.bool_), p)
    global_cov_per_cluster = global_cov / K

    # K-means cluster assignments
    _, asgn = _kmeans_mv(obs, obs_mask, K)   # asgn: (batch, N)

    mu_list, Delta_list, Gamma_list = [], [], []
    min_cluster_size = float(max(10, p + 2))

    for k in range(K):
        in_k = (asgn == k)                                                # (batch, N)
        n_k = in_k.astype(obs.dtype).sum(-1)                             # (batch,)

        # NaN-aware cluster mean
        m_k = obs_mask & in_k[:, :, None]
        cnt_k = jnp.maximum(m_k.astype(obs.dtype).sum(1), 1.0)          # (batch, p)
        mu_k = (obs * m_k).sum(1) / cnt_k                                # (batch, p)

        # Use global cov scaled down when cluster is too small for reliable cov
        small = n_k < min_cluster_size
        cov_k = _nan_aware_cov(obs, obs_mask, in_k, p)
        cov_k = jnp.where(small[:, None, None], global_cov_per_cluster, cov_k)

        key_k = jax.random.fold_in(key, k)
        Delta_k = _init_delta_matrix_jax(cov_k, obs, obs_mask, in_k, q, key_k)

        Gamma_k = cov_k - jnp.matmul(Delta_k, jnp.swapaxes(Delta_k, -1, -2))
        Gamma_k = 0.5 * (Gamma_k + jnp.swapaxes(Gamma_k, -1, -2))
        min_ev_k = jnp.linalg.eigvalsh(Gamma_k).min(-1)
        corr_k = jnp.maximum(-min_ev_k + 1e-8, 0.0)
        Gamma_k = Gamma_k + corr_k[:, None, None] * jnp.eye(p)[None, :, :]

        mu_list.append(mu_k)
        Delta_list.append(Delta_k)
        Gamma_list.append(Gamma_k)

    mu0 = jnp.stack(mu_list, axis=1)        # (batch, K, p)
    Delta0 = jnp.stack(Delta_list, axis=1)  # (batch, K, p, q)
    Gamma0 = jnp.stack(Gamma_list, axis=1)  # (batch, K, p, p)

    # Sort components by first dimension of mu (matches CPU kmeans_init_mv)
    sort_idx = jnp.argsort(mu0[:, :, 0], axis=-1)                       # (batch, K)
    mu0 = jnp.take_along_axis(mu0, jnp.broadcast_to(sort_idx[:, :, None], mu0.shape), axis=1)
    Delta0 = jnp.take_along_axis(
        Delta0, jnp.broadcast_to(sort_idx[:, :, None, None], Delta0.shape), axis=1)
    Gamma0 = jnp.take_along_axis(
        Gamma0, jnp.broadcast_to(sort_idx[:, :, None, None], Gamma0.shape), axis=1)

    # Initial weights
    W0 = _initial_weights_mv(obs, obs_mask, sample_idx, mu0, Delta0, Gamma0, S)
    W0 = jnp.take_along_axis(
        W0, jnp.broadcast_to(sort_idx[:, None, :], W0.shape), axis=2)

    # Failed if any Gamma is not PD after construction
    min_ev_all = jnp.stack([
        jnp.linalg.eigvalsh(Gamma0[:, k]).min(-1) for k in range(K)
    ], axis=-1).min(-1)                                                  # (batch,)
    init_failed = min_ev_all < 1e-8

    return mu0, Delta0, Gamma0, W0, init_failed
