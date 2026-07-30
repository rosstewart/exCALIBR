"""Batched CFUSN (q=2) multivariate skew-normal mixture EM — **unconstrained
only**, mirroring ``cfusn/update_steps.py::_em_update_cfusn`` /
``get_truncated_normal_moments_cfusn``'s ``q==2`` fast path.

Per the plan this was built from: the multivariate constraint/separation
code path is unused in practice, so it is not ported here at all — this
module only implements the plain (``constrained=False``) M-step, with the
likelihood-decrease *backtracking* NumPy still applies in that case
(``single_fit``'s backtrack block runs for multivariate fits regardless of
``constrained``, interpolating up to 10 halving steps between old and new
params — it does not immediately fail the fit the way the univariate path
does). `latent_q` is assumed to be exactly 2 (asserted), matching real usage.

**Highest-risk module in this package — validate hardest here.** Two
substitutions replace the NumPy reference's exact behavior with an
approximation that needs confirming in ``tests/test_batch_em_parity.py``:

1. Missingness handling ("big-M" trick): the NumPy reference groups
   observations by missingness pattern (`np.unique` on the observed mask)
   and solves a reduced-dimension system per pattern — a data-dependent
   shape that can't be traced/batched. This module instead augments the
   missing dimensions of `Omega`/`Gamma` with a large placeholder variance
   (`_BIG_M`) and zero correlation to the observed dims, decoupling them, so
   every observation solves the same-shape `(p, p)` system. This changes the
   *normalizing constant* of the per-component density by an additive
   constant that depends only on the observation's own missingness pattern
   (not on which component or which candidate params are being evaluated),
   which cancels out in (a) E-step responsibilities (softmax over
   components, same point), (b) EM likelihood-convergence/backtracking
   checks (same point set every iteration), and (c) `val_ll` comparisons
   across `fit_idx` within one bootstrap (all fits share the same
   `val_observations`). It does **not** cancel in an absolute
   log-likelihood value compared *across* different bootstraps/datasets —
   don't consume this module's `ll` output as an absolute quantity without
   re-deriving it from the NumPy path.
2. Bivariate normal CDF: `jax`/`jax.scipy.stats` has no multivariate-normal
   CDF. `_bvn_cdf` below implements the standard integral representation
   Phi_2(h,k;rho) = Phi(h)Phi(k) + (1/2pi) * integral_0^rho of the bivariate
   density in the correlation parameter, evaluated via fixed-order
   Gauss-Legendre quadrature (accurate and differentiable, but an
   approximation — not scipy's `multivariate_normal.cdf`).

Given both of the above, per-fit `val_ll` values from this module should be
expected to differ slightly (not bit-identically) from the NumPy path even
when the fitted component params match closely; component params and
relative fit ranking within a bootstrap are the parity signal to trust most.

Memory note: peak memory during the E-step/M-step is `O(batch * N * p^2)`
per component (the augmented `(p,p)` system is built per-observation) —
much larger than the univariate path's `O(batch * N)`. Use a smaller
`max_batch_size` for CFUSN groups in `interop.py` than for univariate ones.
"""
import functools

import numpy as _np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.stats import norm as jnorm
from jax.scipy.special import logsumexp

_LOG2PI = float(jnp.log(2 * jnp.pi))
_LOG2 = float(jnp.log(2.0))
_BIG_M = 1e8
_EPS = 1e-8
_REL_TOL = 1e-8


def _bvn_cdf(h, k, rho, n_quad=24):
    """Phi_2(h, k; rho) via Gauss-Legendre quadrature. h, k, rho: same shape."""
    nodes = jnp.asarray(_np.polynomial.legendre.leggauss(n_quad)[0])
    weights = jnp.asarray(_np.polynomial.legendre.leggauss(n_quad)[1])
    rho = jnp.clip(rho, -0.999, 0.999)

    t = 0.5 * rho[..., None] * (nodes[None, ...] + 1.0)          # (..., n_quad)
    jac = 0.5 * rho
    h_ = h[..., None]
    k_ = k[..., None]
    denom = jnp.maximum(1.0 - t ** 2, 1e-8)
    integrand = jnp.exp(-(h_ ** 2 - 2 * t * h_ * k_ + k_ ** 2) / (2 * denom)) / jnp.sqrt(denom)
    integral = jac * jnp.sum(weights[None, ...] * integrand, axis=-1)
    return jnorm.cdf(h) * jnorm.cdf(k) + integral / (2 * jnp.pi)


def _augment_omega(Omega, obs_mask):
    """Omega: (..., p, p); obs_mask: (..., p) bool. Decouple missing dims
    with a large placeholder variance so every observation gets a
    fixed-shape (p, p) system regardless of its missingness pattern.
    """
    p = Omega.shape[-1]
    miss = jnp.logical_not(obs_mask)
    row_miss = miss[..., :, None]
    col_miss = miss[..., None, :]
    keep = jnp.logical_not(jnp.logical_or(row_miss, col_miss))
    eye = jnp.eye(p)
    diag_miss = miss[..., :, None] * eye
    return jnp.where(keep, Omega, 0.0) + _BIG_M * diag_miss


def _component_moments_and_logpdf(observations, obs_mask, mu, Delta, Gamma):
    """observations/obs_mask: (batch, N, p); mu: (batch, p); Delta: (batch, p, 2);
    Gamma: (batch, p, p). Returns eta, Psi (batch,N,2[,2]) and log_pdf (batch,N).
    """
    p = mu.shape[-1]
    Omega = Gamma + jnp.einsum('bpq,brq->bpr', Delta, Delta)
    Omega = 0.5 * (Omega + jnp.swapaxes(Omega, -1, -2))
    Omega_aug = _augment_omega(Omega[:, None, :, :], obs_mask)     # (batch,N,p,p)

    Delta_bc = jnp.broadcast_to(Delta[:, None, :, :], Omega_aug.shape[:-1] + (2,))
    Omega_inv_Delta = jnp.linalg.solve(Omega_aug, Delta_bc)        # (batch,N,p,2)

    D = jnp.eye(2) - jnp.einsum('bnpq,bnpr->bnqr', Delta_bc, Omega_inv_Delta)
    D = 0.5 * (D + jnp.swapaxes(D, -1, -2)) + _EPS * jnp.eye(2)

    resid = jnp.where(obs_mask, observations - mu[:, None, :], 0.0)   # (batch,N,p)
    means = jnp.einsum('bnp,bnpq->bnq', resid, Omega_inv_Delta)       # (batch,N,2)

    # ── density (log_phi + log_Phi) ──────────────────────────────────────
    logdet = jnp.linalg.slogdet(Omega_aug)[1]                          # (batch,N)
    Omega_aug_inv_resid = jnp.linalg.solve(Omega_aug, resid[..., None])[..., 0]
    maha = jnp.einsum('bnp,bnp->bn', resid, Omega_aug_inv_resid)
    log_phi = -0.5 * (p * _LOG2PI + logdet + maha)

    std = jnp.sqrt(jnp.diagonal(D, axis1=-2, axis2=-1))                # (batch,N,2)
    corr = jnp.clip(D[..., 0, 1] / (std[..., 0] * std[..., 1] + 1e-15), -0.999, 0.999)
    log_Phi = jnp.log(jnp.maximum(
        _bvn_cdf(means[..., 0] / std[..., 0], means[..., 1] / std[..., 1], corr), 1e-300
    ))
    log_pdf = 2 * _LOG2 + log_phi + log_Phi
    log_pdf = jnp.where(jnp.isfinite(log_pdf), log_pdf, -jnp.inf)

    # ── truncated-normal moments, q=2 closed form (update_steps.py fast path) ──
    alpha = means / std
    phi_a = jnorm.pdf(alpha)
    Phi_a = jnorm.cdf(alpha)
    safe = Phi_a > 1e-12
    ratio = jnp.where(safe, phi_a / jnp.where(safe, Phi_a, 1.0), jnp.abs(alpha))
    eta = means + std * ratio                                          # (batch,N,2)
    diag_Ps = std ** 2 + means ** 2 + std * means * ratio
    cross = eta[..., 0] * eta[..., 1] + corr * std[..., 0] * std[..., 1]
    Psi = jnp.zeros(means.shape[:-1] + (2, 2))
    Psi = Psi.at[..., 0, 0].set(diag_Ps[..., 0])
    Psi = Psi.at[..., 1, 1].set(diag_Ps[..., 1])
    Psi = Psi.at[..., 0, 1].set(cross)
    Psi = Psi.at[..., 1, 0].set(cross)

    return eta, Psi, log_pdf


def _densities_all_components(observations, obs_mask, mu, Delta, Gamma):
    """mu:(batch,K,p) Delta:(batch,K,p,2) Gamma:(batch,K,p,p).
    Returns log_pdfs (batch,K,N), etas/psis: list length K of (batch,N,2[,2]).
    """
    K = mu.shape[1]
    log_pdfs, etas, psis = [], [], []
    for k in range(K):
        eta_k, psi_k, lp_k = _component_moments_and_logpdf(
            observations, obs_mask, mu[:, k], Delta[:, k], Gamma[:, k]
        )
        log_pdfs.append(lp_k)
        etas.append(eta_k)
        psis.append(psi_k)
    return jnp.stack(log_pdfs, axis=1), etas, psis


def _responsibilities_from_logpdfs(log_pdfs, sample_idx, W):
    """log_pdfs:(batch,K,N); W:(batch,S,K) -> P:(batch,N,K)."""
    batch = W.shape[0]
    w_n = W[jnp.arange(batch)[:, None], sample_idx, :]                  # (batch,N,K)
    log_w = jnp.where(w_n > 0, jnp.log(jnp.where(w_n > 0, w_n, 1.0)), -jnp.inf)
    numer = jnp.moveaxis(log_pdfs, 1, 2) + log_w                        # (batch,N,K)
    denom = logsumexp(numer, axis=-1, keepdims=True)
    return jnp.nan_to_num(jnp.exp(numer - denom), nan=0.0)


def _weights_and_ll(log_pdfs, sample_idx, W_old, n_samples):
    """Matches get_sample_weights_and_ll: weights from OLD W as prior against
    NEW-param densities, LL from the newly updated weights.
    """
    P = _responsibilities_from_logpdfs(log_pdfs, sample_idx, W_old)      # (batch,N,K)
    W_list = []
    for s in range(n_samples):
        mask = (sample_idx == s).astype(P.dtype)                        # (batch,N)
        denom_s = jnp.maximum(mask.sum(axis=-1), 1e-12)
        num_s = jnp.einsum('bn,bnk->bk', mask, P)
        W_list.append(num_s / denom_s[:, None])
    W_new = jnp.stack(W_list, axis=1)                                    # (batch,S,K)

    batch = W_new.shape[0]
    N = sample_idx.shape[1]
    w_n_new = W_new[jnp.arange(batch)[:, None], sample_idx, :]
    log_w_new = jnp.where(w_n_new > 0, jnp.log(jnp.where(w_n_new > 0, w_n_new, 1.0)), -jnp.inf)
    log_mix = logsumexp(jnp.moveaxis(log_pdfs, 1, 2) + log_w_new, axis=-1)
    ll = log_mix.sum(axis=-1) / N
    return W_new, ll


def _m_step(observations, obs_mask, resp, mu, Delta, etas, psis):
    """One CFUSN M-step for all K components (unconstrained only).
    mu:(batch,K,p) Delta:(batch,K,p,2). etas/psis: from OLD params (§ E-step).
    """
    K, p = mu.shape[1], mu.shape[2]
    obs_f = obs_mask.astype(observations.dtype)
    x_fill = jnp.where(obs_mask, observations, 0.0)

    new_mu, new_Delta, new_Gamma = [], [], []
    for c in range(K):
        z = resp[:, c, :]                                               # (batch,N)
        eta_c, Psi_c = etas[c], psis[c]                                  # (batch,N,2),(batch,N,2,2)
        mu_old_c, Delta_old_c = mu[:, c], Delta[:, c]                    # (batch,p),(batch,p,2)

        # location
        Delta_eta = jnp.einsum('bnq,bpq->bnp', eta_c, Delta_old_c)       # (batch,N,p)
        obs_z = obs_f * z[..., None]                                     # (batch,N,p)
        numer_mu = (obs_z * (x_fill - Delta_eta)).sum(axis=1)            # (batch,p)
        denom_mu = obs_z.sum(axis=1)
        mu_new_c = jnp.where(denom_mu > 1e-12, numer_mu / jnp.maximum(denom_mu, 1e-12), mu_old_c)

        # Delta (solve p independent 2x2 systems)
        residuals = x_fill - mu_new_c[:, None, :]                        # (batch,N,p)
        numer_pq = jnp.einsum('bnp,bnq->bpq', obs_z * residuals, eta_c)  # (batch,p,2)
        Psi_sum = jnp.einsum('bnp,bnij->bpij', obs_z, Psi_c)             # (batch,p,2,2)
        Delta_new_dims = []
        for d in range(p):
            Ps_d = Psi_sum[:, d] + _EPS * jnp.eye(2)
            Delta_new_dims.append(jnp.linalg.solve(Ps_d, numer_pq[:, d][..., None]).squeeze(-1))
        Delta_new_c = jnp.stack(Delta_new_dims, axis=1)                  # (batch,p,2)

        # Gamma
        Delta_eta_new = jnp.einsum('bnq,bpq->bnp', eta_c, Delta_new_c)
        residuals2 = x_fill - mu_new_c[:, None, :] - Delta_eta_new
        Psi_minus = Psi_c - jnp.einsum('bni,bnj->bnij', eta_c, eta_c)
        masked_r = obs_f * residuals2                                    # (batch,N,p)
        term1 = jnp.einsum('zna,znr->zar', z[:, :, None] * masked_r, masked_r)
        z_ob = z[:, :, None, None] * obs_f[:, :, :, None] * obs_f[:, :, None, :]  # (batch,N,p,p)
        Psi_corr = jnp.einsum('znab,znij->zabij', z_ob, Psi_minus)
        term2 = jnp.einsum('zai,zabij,zbj->zab', Delta_new_c, Psi_corr, Delta_new_c)
        Z_b = z_ob.sum(axis=1)                                            # (batch,p,p)
        Gamma_new_c = jnp.where(Z_b > 1e-12, (term1 + term2) / jnp.maximum(Z_b, 1e-12), 0.0)
        Gamma_new_c = 0.5 * (Gamma_new_c + jnp.swapaxes(Gamma_new_c, -1, -2))
        Gamma_new_c = Gamma_new_c + 1e-8 * jnp.eye(p)

        new_mu.append(mu_new_c)
        new_Delta.append(Delta_new_c)
        new_Gamma.append(Gamma_new_c)

    return jnp.stack(new_mu, axis=1), jnp.stack(new_Delta, axis=1), jnp.stack(new_Gamma, axis=1)


def _interpolate(old, new, alpha):
    return tuple(o + alpha * (n - o) for o, n in zip(old, new))


@functools.partial(
    jax.jit,
    static_argnums=(3,),
    static_argnames=("max_em_iters", "n_backtrack"),
)
def fit_batch_cfusn(observations, obs_mask, sample_idx, n_samples,
                     mu0, Delta0, Gamma0, W0, max_em_iters=2000, n_backtrack=10):
    """Batched, unconstrained-only CFUSN (q=2) EM fit.

    observations, obs_mask : (batch, N, p) — NaNs pre-filled as 0 in
        `observations`; `obs_mask` carries which entries were actually observed.
    sample_idx : (batch, N) int.
    mu0 : (batch, K, p); Delta0 : (batch, K, p, 2); Gamma0 : (batch, K, p, p).
    W0 : (batch, n_samples, K).

    Returns mu, Delta, Gamma, W, failed (batch,) — `failed` is only set on
    NaN/non-finite breakdowns (see module docstring: multivariate fits
    backtrack on a plain likelihood decrease rather than failing outright).
    """
    batch = mu0.shape[0]

    def cond(state):
        it, *_, done = state
        return jnp.logical_and(it < max_em_iters, jnp.logical_not(jnp.all(done)))

    def body(state):
        it, mu, Delta, Gamma, W, ll_prev, failed, done = state

        log_pdfs_old, etas, psis = _densities_all_components(observations, obs_mask, mu, Delta, Gamma)
        resp = _responsibilities_from_logpdfs(log_pdfs_old, sample_idx, W)   # (batch,N,K) -> need (batch,K,N)
        resp = jnp.moveaxis(resp, 2, 1)

        mu2, Delta2, Gamma2 = _m_step(observations, obs_mask, resp, mu, Delta, etas, psis)
        log_pdfs2, _, _ = _densities_all_components(observations, obs_mask, mu2, Delta2, Gamma2)
        W2, ll2 = _weights_and_ll(log_pdfs2, sample_idx, W, n_samples)

        bt_threshold = 1e-8 * jnp.abs(ll_prev)
        decreased = (it > 0) & (ll2 < ll_prev - bt_threshold)

        # Backtracking: up to n_backtrack halving steps interpolating toward
        # the old params (mirrors single_fit's alpha-halving block for mv fits).
        def backtrack_step(alpha, carry):
            mu_bt, Delta_bt, Gamma_bt, ll_bt, resolved = carry
            mu_try, Delta_try, Gamma_try = _interpolate((mu, Delta, Gamma), (mu2, Delta2, Gamma2), alpha)
            log_pdfs_try, _, _ = _densities_all_components(observations, obs_mask, mu_try, Delta_try, Gamma_try)
            _, ll_try = _weights_and_ll(log_pdfs_try, sample_idx, W, n_samples)
            ok = (ll_try >= ll_prev - 1e-13) & jnp.logical_not(resolved)
            mu_out = jnp.where(ok[:, None, None], mu_try, mu_bt)
            Delta_out = jnp.where(ok[:, None, None, None], Delta_try, Delta_bt)
            Gamma_out = jnp.where(ok[:, None, None, None], Gamma_try, Gamma_bt)
            ll_out = jnp.where(ok, ll_try, ll_bt)
            return mu_out, Delta_out, Gamma_out, ll_out, resolved | ok

        bt_carry = (mu, Delta, Gamma, ll_prev, jnp.zeros((batch,), dtype=bool))
        alpha = 0.5
        for _ in range(n_backtrack):
            bt_carry = backtrack_step(alpha, bt_carry)
            alpha *= 0.5
        mu_bt, Delta_bt, Gamma_bt, ll_bt, resolved = bt_carry
        # Backtracking exhausted without recovery -> revert fully to old params/ll.
        mu_bt = jnp.where(resolved[:, None, None], mu_bt, mu)
        Delta_bt = jnp.where(resolved[:, None, None, None], Delta_bt, Delta)
        Gamma_bt = jnp.where(resolved[:, None, None, None], Gamma_bt, Gamma)
        ll_bt = jnp.where(resolved, ll_bt, ll_prev)

        mu_final = jnp.where(decreased[:, None, None], mu_bt, mu2)
        Delta_final = jnp.where(decreased[:, None, None, None], Delta_bt, Delta2)
        Gamma_final = jnp.where(decreased[:, None, None, None], Gamma_bt, Gamma2)
        ll_final = jnp.where(decreased, ll_bt, ll2)
        W_final = W2  # weights recompute is cheap enough to skip re-deriving under backtrack

        bad = jnp.isnan(ll_final) | jnp.any(jnp.isnan(mu_final), axis=(1, 2))
        new_failed = failed | (bad & jnp.logical_not(done))

        rel_change = jnp.abs(ll_final - ll_prev) / jnp.maximum(jnp.abs(ll_prev), 1e-300)
        converged = (it >= 1) & (rel_change < _REL_TOL) & jnp.logical_not(bad)

        keep = done
        mu_out = jnp.where(keep[:, None, None], mu, mu_final)
        Delta_out = jnp.where(keep[:, None, None, None], Delta, Delta_final)
        Gamma_out = jnp.where(keep[:, None, None, None], Gamma, Gamma_final)
        W_out = jnp.where(keep[:, None, None], W, W_final)
        ll_out = jnp.where(keep, ll_prev, ll_final)

        new_done = done | new_failed | converged
        return (it + 1, mu_out, Delta_out, Gamma_out, W_out, ll_out, new_failed, new_done)

    init_state = (
        jnp.zeros((), dtype=jnp.int32),
        mu0, Delta0, Gamma0, W0,
        jnp.full((batch,), -jnp.inf),
        jnp.zeros((batch,), dtype=bool),
        jnp.zeros((batch,), dtype=bool),
    )
    it_final, mu, Delta, Gamma, W, _ll, failed, _done = lax.while_loop(cond, body, init_state)
    return mu, Delta, Gamma, W, failed, it_final
