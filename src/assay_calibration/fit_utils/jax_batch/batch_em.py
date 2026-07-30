"""Batched univariate skew-normal mixture EM, mirroring
``cfusn/fit.py::single_fit`` + ``cfusn/update_steps.py::_em_update_univariate``
for the ``multivariate=False`` path.

One call fits an entire batch of independent (bootstrap_seed, fit_idx) jobs
for the *same dataset and num_components* at once — every array below has a
leading ``batch`` axis instead of being a single unbatched fit. See
``interop.py`` for how job pickles are packed into these batched arrays and
unpacked back into the existing per-job result-dict format.

Known simplifications vs. the NumPy reference (validate in
``tests/test_batch_em_parity.py``):
  - No cross-iteration log-pdf caching (NumPy threads `cached_log_pdfs`
    forward as a performance optimization only — recomputing every
    iteration here is mathematically equivalent, just wastes some FLOPs
    XLA can likely fuse away anyway).
  - The exact iteration-count bookkeeping (NumPy calls one extra
    `em_iteration` before its main loop starts) isn't reproduced 1:1 — both
    implementations still run EM to the same relative-likelihood-change
    convergence criterion, so the fixed point should match even though the
    iteration at which convergence is *declared* may be off by one or two.
  - `sample_balance_beta` / `sample_proportions` (M-step reweighting) are
    not supported; only the default (unweighted) M-step is implemented.
  - A NumPy fit that raises (likelihood decrease beyond floating-point
    noise, degenerate params) is discarded entirely by `tryToFit`/
    `execute_fit_job` (component_params/weights reset, val_ll=-inf) rather
    than salvaged — this batched path matches that by marking the batch
    element `failed` and letting the interop layer report val_ll=-inf,
    not by trying to recover the best iterate before the failure.
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from .skewnorm_jax import (
    skewnorm_logpdf, canonical_to_alternate, alternate_to_canonical,
    truncated_moments_from_canonical,
)
from .constraints_jax import build_grid, binary_search

_REL_TOL = 1e-8
_MIN_SCALE = 1e-100


def _log_pdfs(observations, a, loc, scale):
    """observations: (batch, N); a/loc/scale: (batch, K) -> (batch, K, N)."""
    K = a.shape[1]
    return jnp.stack([
        skewnorm_logpdf(observations, a[:, k:k + 1], loc[:, k:k + 1], scale[:, k:k + 1])
        for k in range(K)
    ], axis=1)


def _gather_sample_weights(W, sample_idx):
    """W: (batch, S, K); sample_idx: (batch, N) int -> (batch, N, K) = W[b, sample_idx[b,n], :]."""
    batch = W.shape[0]
    return W[jnp.arange(batch)[:, None], sample_idx, :]


def _e_step_responsibilities(observations, sample_idx, a, loc, scale, W):
    """Matches sample_specific_responsibilities / component_posteriors.
    Returns (batch, K, N) responsibilities.
    """
    log_pdfs = _log_pdfs(observations, a, loc, scale)          # (batch,K,N)
    log_pdfs_bnk = jnp.moveaxis(log_pdfs, 1, 2)                 # (batch,N,K)
    w_n = _gather_sample_weights(W, sample_idx)                 # (batch,N,K)
    log_w = jnp.where(w_n > 0, jnp.log(jnp.where(w_n > 0, w_n, 1.0)), -jnp.inf)
    numer = log_pdfs_bnk + log_w
    denom = logsumexp(numer, axis=-1, keepdims=True)
    P = jnp.exp(numer - denom)
    P = jnp.nan_to_num(P, nan=0.0)
    return jnp.moveaxis(P, 2, 1)                                # (batch,K,N)


def _weights_and_ll(observations, sample_idx, a_new, loc_new, scale_new, W_old, n_samples):
    """Matches get_sample_weights_and_ll: new weights from OLD W as prior mixing
    weight against NEW-param densities, then LL using the NEWLY updated weights.
    """
    log_pdfs = _log_pdfs(observations, a_new, loc_new, scale_new)   # (batch,K,N)
    log_pdfs_bnk = jnp.moveaxis(log_pdfs, 1, 2)                      # (batch,N,K)
    w_n_old = _gather_sample_weights(W_old, sample_idx)              # (batch,N,K)
    log_w_old = jnp.where(w_n_old > 0, jnp.log(jnp.where(w_n_old > 0, w_n_old, 1.0)), -jnp.inf)
    numer = log_pdfs_bnk + log_w_old
    denom = logsumexp(numer, axis=-1, keepdims=True)
    P = jnp.nan_to_num(jnp.exp(numer - denom), nan=0.0)              # (batch,N,K)

    N = observations.shape[1]
    W_new_list = []
    for s in range(n_samples):
        mask = (sample_idx == s).astype(observations.dtype)          # (batch,N)
        denom_s = jnp.maximum(mask.sum(axis=-1), 1e-12)               # (batch,)
        num_s = jnp.einsum('bn,bnk->bk', mask, P)                    # (batch,K)
        W_new_list.append(num_s / denom_s[:, None])
    W_new = jnp.stack(W_new_list, axis=1)                             # (batch,S,K)

    w_n_new = _gather_sample_weights(W_new, sample_idx)                # (batch,N,K)
    log_w_new = jnp.where(w_n_new > 0, jnp.log(jnp.where(w_n_new > 0, w_n_new, 1.0)), -jnp.inf)
    log_mix = logsumexp(log_pdfs_bnk + log_w_new, axis=-1)            # (batch,N)
    ll = log_mix.sum(axis=-1) / N                                     # (batch,)
    return W_new, ll


def _m_step(observations, resp, a, loc, scale, constrained, x_grid, force_gaussian):
    """One M-step for all K components. Mirrors _em_update_univariate."""
    K = a.shape[1]
    # (loc, Delta, Gamma) per component, batch-shaped, mutated in place as
    # components finalize — mirrors NumPy's `updated`/`current_component_params` mix.
    alt = list(canonical_to_alternate(a[:, k], loc[:, k], scale[:, k]) for k in range(K))

    new_a, new_loc, new_scale = [None] * K, [None] * K, [None] * K
    for c in range(K):
        z = resp[:, c, :]                                              # (batch,N)
        denom = jnp.maximum(z.sum(-1), 1e-12)
        cur_a, cur_loc, cur_scale = a[:, c], loc[:, c], scale[:, c]
        v, w = truncated_moments_from_canonical(
            observations, cur_a[:, None], cur_loc[:, None], cur_scale[:, None]
        )
        _, Delta_old, _ = canonical_to_alternate(cur_a, cur_loc, cur_scale)

        m = observations - v * Delta_old[:, None]
        loc_cand = (m * z).sum(-1) / denom
        if constrained:
            loc_final = binary_search(loc_cand, alt, c, 0, x_grid)
        else:
            loc_final = loc_cand
        alt[c] = (loc_final, alt[c][1], alt[c][2])

        if force_gaussian:
            Delta_final = jnp.zeros_like(loc_final)
        else:
            d = v * (observations - loc_final[:, None])
            Delta_cand = (d * z).sum(-1) / jnp.maximum((w * z).sum(-1), 1e-12)
            if constrained:
                Delta_final = binary_search(Delta_cand, alt, c, 1, x_grid)
            else:
                Delta_final = Delta_cand
        alt[c] = (alt[c][0], Delta_final, alt[c][2])

        resid = observations - loc_final[:, None]
        g = resid ** 2 - 2 * Delta_final[:, None] * v * resid + Delta_final[:, None] ** 2 * w
        Gamma_cand = (g * z).sum(-1) / denom
        if constrained:
            Gamma_final = binary_search(Gamma_cand, alt, c, 2, x_grid)
        else:
            Gamma_final = Gamma_cand
        alt[c] = (alt[c][0], alt[c][1], Gamma_final)

        a_c, loc_c, scale_c = alternate_to_canonical(loc_final, Delta_final, Gamma_final)
        scale_c = jnp.maximum(scale_c, _MIN_SCALE)
        new_a[c], new_loc[c], new_scale[c] = a_c, loc_c, scale_c

    return jnp.stack(new_a, axis=1), jnp.stack(new_loc, axis=1), jnp.stack(new_scale, axis=1)


@functools.partial(
    jax.jit,
    static_argnums=(2,),
    static_argnames=("constrained", "force_gaussian", "max_em_iters"),
)
def fit_batch(observations, sample_idx, n_samples, a0, loc0, scale0, W0,
              xmin, xmax, constrained=True, force_gaussian=False,
              max_em_iters=10000):
    """Batched EM fit for a group of univariate jobs sharing (dataset, num_components).

    Parameters
    ----------
    observations : (batch, N) float — training observations, one row per
        (bootstrap_seed, fit_idx) job. Same N across the batch (bootstrap
        resampling always draws N points with replacement from the dataset).
    sample_idx : (batch, N) int — which of the `n_samples` classes each
        point belongs to (argmax of the one-hot `train_sample_assignments`).
    a0, loc0, scale0 : (batch, K) — per-component initial canonical params
        (from kmeans_init / method_of_moments, computed on CPU — see
        interop.py; batching does not extend to initialization).
    W0 : (batch, n_samples, K) — initial per-sample mixing weights.
    xmin, xmax : (batch,) — score range for the constraint grid (only used
        when constrained=True).
    constrained, force_gaussian : Python bool, assumed uniform across the
        batch (true for every job sharing one `prepare.py` invocation).

    Returns
    -------
    a, loc, scale : (batch, K) final canonical params.
    W : (batch, n_samples, K) final per-sample weights.
    failed : (batch,) bool — True where the fit should be treated as failed
        (matches NumPy's `tryToFit` exception path: discard, val_ll=-inf).
    it_final : () int32 — actual iteration count at exit (diagnostic).
    """
    x_grid = build_grid(xmin, xmax) if constrained else None
    batch, K = a0.shape

    def cond(state):
        it, *_ , done = state
        return jnp.logical_and(it < max_em_iters, jnp.logical_not(jnp.all(done)))

    def body(state):
        it, a, loc, scale, W, ll_prev, failed, done = state

        resp = _e_step_responsibilities(observations, sample_idx, a, loc, scale, W)
        a2, loc2, scale2 = _m_step(observations, resp, a, loc, scale,
                                    constrained, x_grid, force_gaussian)
        W2, ll2 = _weights_and_ll(observations, sample_idx, a2, loc2, scale2, W, n_samples)

        bad = jnp.isnan(ll2) | jnp.any(jnp.isnan(W2), axis=(1, 2))
        bt_threshold = 1e-8 * jnp.abs(ll_prev)
        decreased = (it > 0) & (ll2 < ll_prev - bt_threshold)
        failed_now = (decreased | bad) & jnp.logical_not(done)
        new_failed = failed | failed_now

        rel_change = jnp.abs(ll2 - ll_prev) / jnp.maximum(jnp.abs(ll_prev), 1e-300)
        converged = (it >= 1) & (rel_change < _REL_TOL) & jnp.logical_not(failed_now)

        keep = done[:, None]
        a_out = jnp.where(keep, a, jnp.where(failed_now[:, None], a, a2))
        loc_out = jnp.where(keep, loc, jnp.where(failed_now[:, None], loc, loc2))
        scale_out = jnp.where(keep, scale, jnp.where(failed_now[:, None], scale, scale2))
        keep3 = done[:, None, None]
        W_out = jnp.where(keep3, W, jnp.where(failed_now[:, None, None], W, W2))
        ll_out = jnp.where(done, ll_prev, jnp.where(failed_now, ll_prev, ll2))

        new_done = done | failed_now | converged
        return (it + 1, a_out, loc_out, scale_out, W_out, ll_out, new_failed, new_done)

    init_state = (
        jnp.zeros((), dtype=jnp.int32),
        a0, loc0, scale0, W0,
        jnp.full((batch,), -jnp.inf),
        jnp.zeros((batch,), dtype=bool),
        jnp.zeros((batch,), dtype=bool),
    )
    it_final, a, loc, scale, W, _ll, failed, done = lax.while_loop(cond, body, init_state)
    return a, loc, scale, W, failed, it_final, done
