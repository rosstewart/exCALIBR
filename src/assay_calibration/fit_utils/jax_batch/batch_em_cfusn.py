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
   approximation — not scipy's `multivariate_normal.cdf`). `_bvn_cdf` is
   also now reused as the exact-CDF piece of the truncated-normal moments
   recursion in `_component_moments_and_logpdf` (see below) — the moments
   themselves are the exact closed form (Kan & Robotti 2017), not an
   approximation; `_bvn_cdf`'s quadrature error is the only source of
   deviation from a true closed-form value, and it is PSD by construction
   (unlike the ad-hoc cross-term formula this replaced, which needed a
   defensive clip and is now gone).

**Not independently re-validated against real GPU hardware or
`tests/test_batch_em_parity.py` in the session that ported the exact
moments recursion below** (no `jax` install was available there) — this
mirrors the already MC-validated NumPy formula in `update_steps.py`
line-for-line, but re-run `test_batch_em_parity.py` on a machine with JAX
installed before trusting this path in production.

Given the above, per-fit `val_ll` values from this module should be
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
from jax.scipy.special import logsumexp, log_ndtr

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


# ── Shared log-space orthant probability and moment ratios ─────────────────
#
# These are line-for-line ports of cfusn/update_steps.py's
# _log_bvn_orthant_quad / _log_phi_at0 / _log_f1_0 / _log_f1_1, deliberately
# using the SAME algorithm rather than a second one that has to be kept in
# agreement. The previous arrangement -- Owen's T on CPU, a Drezner
# rho-integral here -- was not a mirror at all: it clipped rho at 0.999 vs the
# CPU's 0.999999, floored 1-t^2 at 1e-8 with no CPU analogue, used 24 nodes,
# and in the deep tail returned Phi(h)Phi(k) + integral with both terms
# underflowed to 0. Keep these in sync with update_steps.py by construction:
# if one changes, change the other identically.
#
# Constants must match update_steps.py exactly.
_LOG_SQRT_2PI = 0.5 * float(_np.log(2.0 * _np.pi))
_BVN_QUAD_NODES = 64
_BVN_QUAD_PANELS = 4
_BVN_QUAD_LOG_RANGE = 140.0
_F1_1_ASYMPTOTIC_T = 25.0

# Gauss-Legendre nodes are data-independent, so resolve them at trace time with
# numpy rather than carrying them as traced values.
_GL_NODES, _GL_WEIGHTS = _np.polynomial.legendre.leggauss(_BVN_QUAD_NODES)


# Owen's T, and the Owen's-T form of Phi_2, so the GPU can run the SAME hybrid
# as the CPU. JAX has no owens_t primitive, so it is computed by quadrature on a
# bounded interval -- validated against scipy.special.owens_t (the function the
# CPU actually calls) rather than against a second guess at the same thing:
#
#     regime              max abs err
#     |a| <= 1              4.2e-17
#     1 < |a| <= 50         1.1e-16
#     |a| > 50              5.6e-17
#     |a| huge (1e9)        8.3e-17
#     |h| huge              0.0
#     |h| tiny, |a| huge    7.0e-12   <- the one loose corner; T is O(0.25)
#                                        there, so ~3e-11 relative
#     h == 0 / a == 0       <= 2.8e-17
#
# Two branches, both bounded-interval, chosen so each is used only where it is
# machine-precision (n=64 x 6 panels; a crossover at |a| = 50 is where they
# overlap -- see the measurements that fixed it there):
#   |a| <= 50 : T = (1/2pi) int_0^{arctan a} exp(-h^2 sec^2(th)/2) dth
#   |a| >  50 : T = 0.5*Phi(-|h|) - (1/2pi) int_0^{arctan(1/a)}
#                                            exp(-h^2/(2 sin^2(ph))) dph
#               i.e. the complement of T(h, inf) = 0.5*Phi(-|h|); the remaining
#               interval shrinks like 1/a, so large a is easy rather than hard.
# T is odd in a.
_OWENS_T_NODES = 64
_OWENS_T_PANELS = 6
_OWENS_T_XOVER = 50.0
_OWENS_MIN_RELIABLE = 1e-10          # must match update_steps.py
_OT_NODES, _OT_WEIGHTS = _np.polynomial.legendre.leggauss(_OWENS_T_NODES)


def _owens_t_panel_sum(h, upper, kind):
    x = jnp.asarray(_OT_NODES)
    w = jnp.asarray(_OT_WEIGHTS)
    tot = jnp.zeros(jnp.broadcast_shapes(jnp.shape(h), jnp.shape(upper)))
    for i in range(_OWENS_T_PANELS):
        lo = upper * i / _OWENS_T_PANELS
        hi = upper * (i + 1) / _OWENS_T_PANELS
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        th = mid[..., None] + half[..., None] * x
        if kind == "sec":
            e = jnp.exp(-0.5 * h[..., None] ** 2 / jnp.cos(th) ** 2)
        else:
            e = jnp.exp(-0.5 * h[..., None] ** 2
                        / jnp.maximum(jnp.sin(th) ** 2, 1e-300))
        tot = tot + half * jnp.sum(w * e, axis=-1)
    return tot / (2.0 * jnp.pi)


def _owens_t(h, a):
    """Owen's T(h, a). Mirrors scipy.special.owens_t, which the CPU calls."""
    sgn = jnp.sign(a)
    aa = jnp.abs(a)
    small = aa <= _OWENS_T_XOVER
    # clamp each branch's argument so the unselected branch cannot produce a
    # nan/inf that would propagate through the where
    direct = _owens_t_panel_sum(h, jnp.arctan(jnp.where(small, aa, 1.0)), "sec")
    compl = (0.5 * jnorm.cdf(-jnp.abs(h))
             - _owens_t_panel_sum(
                 h, jnp.arctan(1.0 / jnp.where(small, _OWENS_T_XOVER, aa)), "sin"))
    return sgn * jnp.where(small, direct, compl)


def _bvn_cdf_owens(h, k, rho):
    """Phi_2(h, k; rho) via Owen's T. Mirrors update_steps._bvn_cdf_owens,
    including the h==0/k==0 substitution in BOTH numerator and denominator (a
    literal zero there silently zeroes the correlation term)."""
    denom = jnp.sqrt(jnp.maximum(1.0 - rho ** 2, 1e-300))
    h_safe = jnp.where(h == 0, 1e-12, h)
    k_safe = jnp.where(k == 0, 1e-12, k)
    a1 = (k_safe - rho * h_safe) / (h_safe * denom)
    a2 = (h_safe - rho * k_safe) / (k_safe * denom)
    hk = h * k
    delta = jnp.where(jnp.logical_or(hk > 0,
                                     jnp.logical_and(hk == 0, h + k >= 0)),
                      0.0, 0.5)
    return (0.5 * (jnorm.cdf(h) + jnorm.cdf(k))
            - _owens_t(h, a1) - _owens_t(k, a2) - delta)


def _log_bvn_cdf(h, k, rho):
    """log Phi_2(h, k; rho): the SAME hybrid as update_steps._log_bvn_cdf.

    Owen's T above L = 1e-10 (accurate at high |rho|, where the quadrature is
    weakest), the log-space quadrature below (accurate arbitrarily deep, where
    Owen's T catastrophically cancels). Both branches are evaluated for every
    element -- jnp.where cannot skip work -- so the tail form is clamped to stay
    finite where it is not selected.
    """
    fast = _bvn_cdf_owens(h, k, rho)
    log_fast = jnp.log(jnp.maximum(fast, 1e-300))
    log_quad = _log_bvn_orthant_quad(h, k, rho)
    return jnp.where(fast > _OWENS_MIN_RELIABLE, log_fast, log_quad)


def _log_bvn_orthant_quad(h, k, rho,
                          log_range=_BVN_QUAD_LOG_RANGE):
    """log P(Z1 <= h, Z2 <= k), standard bivariate normal, correlation rho.

    Mirrors update_steps._log_bvn_orthant_quad. Cancellation-free: every term in
    the accumulation is positive, so relative accuracy survives arbitrarily deep
    into the tail.

        P = int_0^inf phi(z) * Phi((k' - rho z)/sqrt(1-rho^2)) dz,  z = h' - u

    with (h', k') ordered so the more negative variable is the outer one.
    rho is an array here (per-observation), where the CPU takes a shared scalar;
    the arithmetic is otherwise identical.
    """
    x = jnp.asarray(_GL_NODES)
    w = jnp.asarray(_GL_WEIGHTS)

    lo_v = jnp.minimum(h, k)
    hi_v = jnp.maximum(h, k)
    sd = jnp.sqrt(jnp.maximum(1.0 - rho * rho, 1e-300))

    H = jnp.maximum(-lo_v, 0.0)
    U = jnp.sqrt(H ** 2 + 2.0 * log_range) - H
    U = jnp.maximum(U, 1e-3)
    U = jnp.where(lo_v > 0, jnp.maximum(U, 12.0), U)

    parts = []
    for i in range(_BVN_QUAD_PANELS):
        e0 = i / _BVN_QUAD_PANELS
        e1 = (i + 1) / _BVN_QUAD_PANELS
        half = 0.5 * U * (e1 - e0)
        mid = 0.5 * U * (e1 + e0)
        u = mid[..., None] + half[..., None] * x
        z = lo_v[..., None] - u
        log_phi = -0.5 * z * z - _LOG_SQRT_2PI
        log_Phi = log_ndtr((hi_v[..., None] - rho[..., None] * z) / sd[..., None])
        log_w = jnp.log(jnp.maximum(w * half[..., None], 1e-300))
        parts.append(logsumexp(log_w + log_phi + log_Phi, axis=-1))
    return logsumexp(jnp.stack(parts, axis=-1), axis=-1)


def _log_phi_at0(mu, s):
    """log of the N(mu, s^2) density at 0. Mirrors update_steps._log_phi_at0."""
    return -0.5 * (mu / s) ** 2 - jnp.log(s) - _LOG_SQRT_2PI


def _log_f1_0(mu, s):
    """log P(X > 0) = log Phi(mu/s). Mirrors update_steps._log_f1_0.

    NOT 1 - Phi(-mu/s): that form annihilates once -mu/s > 8.
    """
    return log_ndtr(mu / s)


def _log_f1_1(mu, s):
    """log E[X 1{X>0}] for X ~ N(mu, s^2). Mirrors update_steps._log_f1_1.

    Three regimes, same thresholds as the CPU:
      t >= -1      direct t*Phi(t) + phi(t) -- no cancellation, no overflow.
      -T < t < -1  factor out phi(t), bracket 1 + t*u, u = Phi(t)/phi(t).
                   u ~ exp(t^2/2) so this OVERFLOWS for t beyond about +38 and
                   must never be applied to positive t (that bug returned inf
                   and destroyed real fits on the CPU side).
      t <= -T      asymptotic series for the bracket.

    Evaluated with jnp.where rather than boolean indexing, so all three
    expressions are computed for every element -- hence each is clamped to stay
    finite on the branches where it is not selected, otherwise a NaN/inf from an
    unselected branch would propagate through the where.
    """
    t = mu / s
    T = _F1_1_ASYMPTOTIC_T

    # direct branch (used for t >= -1)
    g_direct = (t * jnp.exp(log_ndtr(t))
                + jnp.exp(-0.5 * t ** 2 - _LOG_SQRT_2PI))
    log_direct = jnp.log(jnp.maximum(g_direct, 1e-300))

    # bracket branch (used for -T < t < -1); clamp the exponent so the
    # unselected positive-t elements cannot overflow to inf
    expo = jnp.minimum(log_ndtr(t) + 0.5 * t ** 2 + _LOG_SQRT_2PI, 300.0)
    u = jnp.exp(expo)
    log_mid = (-0.5 * t ** 2 - _LOG_SQRT_2PI
               + jnp.log(jnp.maximum(1.0 + t * u, 1e-300)))

    # series branch (used for t <= -T); guard 1/t^2 at t == 0
    t_safe = jnp.where(jnp.abs(t) < 1.0, -1.0, t)
    it2 = 1.0 / (t_safe * t_safe)
    series = it2 * (1.0 + it2 * (-3.0 + it2 * (15.0 + it2
                    * (-105.0 + it2 * 945.0))))
    log_far = (-0.5 * t ** 2 - _LOG_SQRT_2PI
               + jnp.log(jnp.maximum(series, 1e-300)))

    out = jnp.where(t >= -1.0, log_direct,
                    jnp.where(t > -T, log_mid, log_far))
    return out + jnp.log(s)


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


# Ported to match cfusn/update_steps.py's log-space rewrite (fixed tail
# cancellation in Phi_2, F1_0 = 1 - Phi(-t) annihilation, the t*Phi(t) + phi(t)
# cancellation, and division by a possibly-zero L). The shared helpers above are
# the SAME algorithm as the CPU path, not a second one -- see their comment.
#
# NOT YET EXECUTED: there is no JAX in this environment, so this port is
# validated only by a numpy-backed shim that runs these formulas against the CPU
# implementation. Before trusting the GPU path, run tests/test_batch_em_parity.py
# on a JAX machine.

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
    corr = jnp.clip(D[..., 0, 1] / (std[..., 0] * std[..., 1] + 1e-15),
                -0.999999, 0.999999)   # matches update_steps.py
    # Same stable log orthant probability as the moments use below; the old
    # _bvn_cdf underflowed to 0 here in the tail and then got floored to 1e-300,
    # silently truncating the density for exactly the observations whose
    # moments were also wrong.
    log_Phi = _log_bvn_cdf(means[..., 0] / std[..., 0],
                           means[..., 1] / std[..., 1], corr)
    log_pdf = 2 * _LOG2 + log_phi + log_Phi
    log_pdf = jnp.where(jnp.isfinite(log_pdf), log_pdf, -jnp.inf)

    # ── truncated-normal moments, q=2 EXACT closed form (Kan & Robotti 2017,
    # JCGS, Theorem 1 recursion) -- mirrors update_steps.py's
    # _mc_truncated_mvn_moments q=2 branch exactly (NumPy there, JAX here);
    # keep both in sync if this formula ever changes. Replaces an earlier
    # ad-hoc cross-term approximation that was not guaranteed PSD (see the
    # NumPy version's docstring for the failure mode this caused). Reuses
    # this file's own _bvn_cdf (above) as the exact bivariate-CDF piece --
    # JAX has no Owen's-T primitive, but this file already needed a
    # precise bivariate normal CDF for log_Phi above, so no new numerical
    # primitive is required here.
    mu1, mu2 = means[..., 0], means[..., 1]
    s1, s2 = std[..., 0], std[..., 1]
    s1sq, s2sq = s1 ** 2, s2 ** 2
    s12 = corr * s1 * s2

    alpha1, alpha2 = -mu1 / s1, -mu2 / s2
    # Everything below is a RATIO to L, and those ratios stay finite as L -> 0,
    # so work in logs and never form L itself. Mirrors update_steps.py's
    # log-space recursion exactly -- see the shared helpers above for why the
    # value-space form was unusable (median relative error 2.3e+05 and a
    # non-positive result 64% of the time once L < 1e-16).
    log_L = _log_bvn_cdf(-alpha1, -alpha2, corr)

    mu_t1 = mu2 - s12 * mu1 / s1sq
    s_t1 = jnp.sqrt(jnp.maximum(s2sq - s12 ** 2 / s1sq, 1e-300))
    mu_t2 = mu1 - s12 * mu2 / s2sq
    s_t2 = jnp.sqrt(jnp.maximum(s1sq - s12 ** 2 / s2sq, 1e-300))

    log_p1 = _log_phi_at0(mu1, s1)
    log_p2 = _log_phi_at0(mu2, s2)
    r00_1 = jnp.exp(log_p1 + _log_f1_0(mu_t1, s_t1) - log_L)
    r00_2 = jnp.exp(log_p2 + _log_f1_0(mu_t2, s_t2) - log_L)

    e1 = mu1 + s1sq * r00_1 + s12 * r00_2
    e2 = mu2 + s12 * r00_1 + s2sq * r00_2

    r10_2 = jnp.exp(log_p2 + _log_f1_1(mu_t2, s_t2) - log_L)
    r01_1 = jnp.exp(log_p1 + _log_f1_1(mu_t1, s_t1) - log_L)

    eta = jnp.stack([e1, e2], axis=-1)                                 # (batch,N,2)
    diag0 = mu1 * e1 + s1sq + s12 * r10_2                              # E[T1^2]
    diag1 = mu2 * e2 + s12 * r01_1 + s2sq                              # E[T2^2]
    x_route1 = mu2 * e1 + s12 + s2sq * r10_2                           # kappa=(1,0)
    x_route2 = mu1 * e2 + s1sq * r01_1 + s12                           # kappa=(0,1)
    cross = 0.5 * (x_route1 + x_route2)

    # Defense-in-depth against a component's shape drifting so far that L
    # rounds to (numerically indistinguishable from) 0 -- mirrors
    # update_steps.py's PSD-preserving fix exactly (see its comment for the
    # reproducer that shows why a naive element-wise clip on Psi is unsafe:
    # it can leave the diagonal negative or the off-diagonal exceeding the
    # Cauchy-Schwarz bound, producing a Psi with a large-magnitude negative
    # eigenvalue). Clip the diagonal to its true valid range [0, CLIP**2]
    # and the off-diagonal to sqrt(diag0*diag1) so the result is guaranteed
    # PSD by construction, not just individually bounded.
    CLIP = 1e4
    eta = jnp.nan_to_num(eta, nan=0.0, posinf=CLIP, neginf=-CLIP)
    eta = jnp.clip(eta, -CLIP, CLIP)
    diag0 = jnp.nan_to_num(diag0, nan=0.0, posinf=CLIP ** 2, neginf=0.0)
    diag1 = jnp.nan_to_num(diag1, nan=0.0, posinf=CLIP ** 2, neginf=0.0)
    diag0 = jnp.clip(diag0, 0.0, CLIP ** 2)
    diag1 = jnp.clip(diag1, 0.0, CLIP ** 2)
    cross = jnp.nan_to_num(cross, nan=0.0, posinf=CLIP ** 2, neginf=-CLIP ** 2)
    cross_bound = jnp.sqrt(diag0 * diag1)
    cross = jnp.clip(cross, -cross_bound, cross_bound)

    Psi = jnp.zeros(means.shape[:-1] + (2, 2))
    Psi = Psi.at[..., 0, 0].set(diag0)
    Psi = Psi.at[..., 1, 1].set(diag1)
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



# ── Completed-data moments (port of cfusn/update_steps._completed_data_moments) ──

# Gamma regularisation and Delta-cap settings, mirroring the CPU path's
# USE_GAMMA_RIDGE / GAMMA_RIDGE_FRAC / DELTA_CAP_PD_CORRECTION so both
# implementations fit the same model. See cfusn/update_steps.py for the full
# derivations; the short version:
#   * available-case masking is the exact M-step only for diagonal Gamma, so
#     the M-step below uses posterior moments of the COMPLETE x instead;
#   * the unpenalised mixture likelihood is unbounded as a component's Gamma
#     collapses, so Gamma gets an inverse-Wishart ridge;
#   * the Delta magnitude cap's positive-definiteness bound needs the
#     (1 - 2/pi) truncated-normal variance factor.
_USE_GAMMA_RIDGE = True
_GAMMA_RIDGE_FRAC = 1e-3
_DELTA_CAP_PD_CORRECTION = True
_DELTA_CAP_SAFETY_FACTOR = 0.95
_TN_VAR = 1.0 - 2.0 / jnp.pi          # Var(T_i) for T ~ TN(0,1) on R_+


def _augment_gamma(Gamma, obs_mask):
    """Gamma: (batch,p,p) -> (batch,N,p,p) with missing dims decoupled.

    Same device as _augment_omega: the observed block is kept, cross terms to
    missing dims are zeroed and missing diagonals get _BIG_M. Solving against
    this yields Gamma_oo^-1 applied to the observed coordinates and ~0 on the
    missing ones, which is what the conditional formulas below need without
    materialising a different-shaped system per missingness pattern.
    """
    p = Gamma.shape[-1]
    G = Gamma[:, None, :, :]                                        # (batch,1,p,p)
    miss = jnp.logical_not(obs_mask)                                # (batch,N,p)
    keep = jnp.logical_not(jnp.logical_or(miss[..., :, None], miss[..., None, :]))
    diag_miss = miss[..., :, None] * jnp.eye(p)
    return jnp.where(keep, G, 0.0) + _BIG_M * diag_miss


def _completed_moments(observations, obs_mask, mu, Delta, Gamma, eta, Psi):
    """Posterior moments of the COMPLETE x given x_obs, for one component.

    Conditional on T=t, x ~ N(mu + Delta t, Gamma), so for missing dims m and
    observed dims o (with K = Gamma_mo Gamma_oo^-1):

        x_m | x_o, t ~ N(a_m + B_m t, C_mm)
        a_m = mu_m + K (x_o - mu_o),  B_m = Delta_m - K Delta_o,
        C_mm = Gamma_mm - K Gamma_om

    Returns Ex (batch,N,p), Ext (batch,N,p,q), Exx (batch,N,p,p).
    """
    p = mu.shape[-1]
    q = Delta.shape[-1]
    G_aug = _augment_gamma(Gamma, obs_mask)                          # (batch,N,p,p)
    Gb = Gamma[:, None, :, :]                                        # (batch,1,p,p)
    miss = jnp.logical_not(obs_mask)                                 # (batch,N,p)

    # a: observed entries keep x; missing entries take mu_m + K (x_o - mu_o).
    resid_o = jnp.where(obs_mask, observations - mu[:, None, :], 0.0)     # (batch,N,p)
    w = jnp.linalg.solve(G_aug, resid_o[..., None])[..., 0]               # (batch,N,p)
    K_resid = jnp.einsum('bnpr,bnr->bnp', jnp.broadcast_to(Gb, G_aug.shape), w)
    a = jnp.where(obs_mask, observations, mu[:, None, :] + K_resid)

    # B: zero on observed rows, Delta_m - K Delta_o on missing rows.
    Delta_o = jnp.where(obs_mask[..., None], Delta[:, None, :, :], 0.0)   # (batch,N,p,q)
    sol_D = jnp.linalg.solve(G_aug, Delta_o)                              # (batch,N,p,q)
    K_Delta = jnp.einsum('bnpr,bnrq->bnpq', jnp.broadcast_to(Gb, G_aug.shape), sol_D)
    B = jnp.where(miss[..., None], Delta[:, None, :, :] - K_Delta, 0.0)

    # C: Gamma_mm - K Gamma_om on the missing-missing block, zero elsewhere.
    G_om = jnp.where(obs_mask[..., :, None] & miss[..., None, :], Gb, 0.0)
    sol_G = jnp.linalg.solve(G_aug, G_om)                                 # (batch,N,p,p)
    K_Gom = jnp.einsum('bnpr,bnrc->bnpc', jnp.broadcast_to(Gb, G_aug.shape), sol_G)
    C = jnp.where(miss[..., :, None] & miss[..., None, :], Gb - K_Gom, 0.0)
    C = 0.5 * (C + jnp.swapaxes(C, -1, -2))

    B_eta = jnp.einsum('bnpq,bnq->bnp', B, eta)
    Ex = a + B_eta
    Ext = a[..., :, None] * eta[..., None, :] + jnp.einsum('bnpi,bnij->bnpj', B, Psi)
    Exx = (a[..., :, None] * a[..., None, :]
           + a[..., :, None] * B_eta[..., None, :]
           + B_eta[..., :, None] * a[..., None, :]
           + jnp.einsum('bnpi,bnij,bnrj->bnpr', B, Psi, B)
           + C)
    return Ex, Ext, Exx


def _ridge_gamma_jax(Gamma_new, n_eff, data_var):
    """Inverse-Wishart shrinkage: (S_c + Psi) / (n_c + nu + p + 1), with
    Gamma_new = S_c / n_c. Mirrors cfusn/update_steps._ridge_gamma."""
    p = Gamma_new.shape[-1]
    nu = p + 2
    Psi = _GAMMA_RIDGE_FRAC * data_var[:, None, None] * jnp.eye(p)
    n = jnp.maximum(n_eff, 0.0)[:, None, None]
    return (Gamma_new * n + Psi) / (n + nu + p + 1)


def _m_step(observations, obs_mask, resp, mu, Delta, Gamma, etas, psis, col_mask=None):
    """One CFUSN M-step for all K components (unconstrained only).
    mu:(batch,K,p) Delta:(batch,K,p,2). etas/psis: from OLD params (§ E-step).
    col_mask: (batch,2), [1,0] for q1-mode rows, [0,0] for Gaussian-mode
    rows, else [1,1] -- see fit_batch_cfusn's docstring. Applied to
    Delta_new_c BEFORE it's used in the Gamma update (not just on the final
    returned Delta), so a masked row's Gamma reflects only its active
    direction(s)' residual variance, matching genuine independent lower-q
    EM exactly rather than transiently crediting the soon-to-be-discarded
    column(s).
    """
    K, p = mu.shape[1], mu.shape[2]
    q = Delta.shape[-1]
    obs_f = obs_mask.astype(observations.dtype)

    # Per-dimension data variance, for the Delta cap's PD bound and the ridge
    # scale. NaN-free because `observations` arrives with missing entries
    # already zero-filled and `obs_mask` carries which are real.
    cnt = jnp.maximum(obs_f.sum(axis=1), 1.0)                            # (batch,p)
    mean_d = (obs_f * observations).sum(axis=1) / cnt
    var_d = jnp.maximum(
        (obs_f * (observations - mean_d[:, None, :]) ** 2).sum(axis=1) / cnt, 0.0)
    data_var = var_d.mean(axis=-1)                                        # (batch,)

    new_mu, new_Delta, new_Gamma = [], [], []
    for c in range(K):
        z = resp[:, c, :]                                                # (batch,N)
        eta_c, Psi_c = etas[c], psis[c]                                  # (batch,N,q),(batch,N,q,q)
        mu_old_c, Delta_old_c, Gamma_old_c = mu[:, c], Delta[:, c], Gamma[:, c]

        # Posterior moments of the COMPLETE x -- the whole point of this port.
        Ex, Ext, Exx = _completed_moments(
            observations, obs_mask, mu_old_c, Delta_old_c, Gamma_old_c, eta_c, Psi_c)

        z_sum = jnp.maximum(z.sum(axis=1), 1e-12)                         # (batch,)

        # location: full-vector weighted mean of E[x] - Delta E[T]
        Delta_eta = jnp.einsum('bnq,bpq->bnp', eta_c, Delta_old_c)        # (batch,N,p)
        mu_new_c = jnp.einsum('bn,bnp->bp', z, Ex - Delta_eta) / z_sum[:, None]

        # Delta: single q x q solve (completed moments make every row inform
        # every dimension, so the per-dimension systems of the available-case
        # version collapse into one).
        numer = (jnp.einsum('bn,bnpq->bpq', z, Ext)
                 - mu_new_c[:, :, None] * jnp.einsum('bn,bnq->bq', z, eta_c)[:, None, :])
        # Conditional ridge floor, matching get_Delta_update_cfusn's RIDGE_FLOOR:
        # only damp a genuinely near-singular system (an unconditional _EPS would
        # leave a near-singular solve to amplify numer by ~1/_EPS, which is the
        # blow-up that floor was introduced for).
        Psi_sum = jnp.einsum('bn,bnij->bij', z, Psi_c)
        Psi_sum = 0.5 * (Psi_sum + jnp.swapaxes(Psi_sum, -1, -2))
        _RIDGE_FLOOR = 1e-3
        eig_min = jnp.linalg.eigvalsh(Psi_sum)[..., 0]                    # (batch,)
        bump = jnp.where(eig_min < _RIDGE_FLOOR,
                         _RIDGE_FLOOR - eig_min + _RIDGE_FLOOR, 0.0)
        Psi_sum = Psi_sum + bump[:, None, None] * jnp.eye(q)
        Delta_new_c = jnp.linalg.solve(
            Psi_sum[:, None, :, :], numer[..., None]).squeeze(-1)         # (batch,p,q)
        if col_mask is not None:
            Delta_new_c = Delta_new_c * col_mask[:, None, :]

        # Defense-in-depth magnitude cap, with the (1 - 2/pi) PD correction:
        # Var(x_d) = Gamma_dd + (1 - 2/pi)||Delta_d||^2, so Gamma_dd > 0 needs
        # ||Delta_d|| < sqrt(Var_d / (1 - 2/pi)).
        var_for_cap = var_d / _TN_VAR if _DELTA_CAP_PD_CORRECTION else var_d
        max_norm_d = _DELTA_CAP_SAFETY_FACTOR * jnp.sqrt(jnp.maximum(var_for_cap, 0.0))
        row_norms = jnp.linalg.norm(Delta_new_c, axis=-1)                 # (batch,p)
        over = row_norms > max_norm_d
        scale = jnp.where(over, max_norm_d / jnp.maximum(row_norms, 1e-12), 1.0)
        Delta_new_c = Delta_new_c * scale[..., None]

        # Gamma from completed sufficient statistics:
        # (1/sum z) sum_n z [ E[xx'] - E[xt']D' - D E[tx'] + D Psi D'
        #                     - mu E[x]' - E[x] mu' + mu E[t]'D' + D E[t] mu' + mu mu' ]
        S_xx = jnp.einsum('bn,bnac->bac', z, Exx)
        S_xt = jnp.einsum('bn,bnaq->baq', z, Ext)
        S_x = jnp.einsum('bn,bna->ba', z, Ex)
        S_tt = jnp.einsum('bn,bnij->bij', z, Psi_c)
        S_t = jnp.einsum('bn,bni->bi', z, eta_c)
        D = Delta_new_c
        Gamma_new_c = (
            S_xx
            - jnp.einsum('baq,bcq->bac', S_xt, D) - jnp.einsum('baq,bcq->bca', S_xt, D)
            + jnp.einsum('bai,bij,bcj->bac', D, S_tt, D)
            - mu_new_c[:, :, None] * S_x[:, None, :] - S_x[:, :, None] * mu_new_c[:, None, :]
            + mu_new_c[:, :, None] * jnp.einsum('bi,bci->bc', S_t, D)[:, None, :]
            + jnp.einsum('bai,bi->ba', D, S_t)[:, :, None] * mu_new_c[:, None, :]
            + z_sum[:, None, None] * mu_new_c[:, :, None] * mu_new_c[:, None, :]
        ) / z_sum[:, None, None]
        Gamma_new_c = 0.5 * (Gamma_new_c + jnp.swapaxes(Gamma_new_c, -1, -2))

        if _USE_GAMMA_RIDGE:
            Gamma_new_c = _ridge_gamma_jax(Gamma_new_c, z.sum(axis=1), data_var)
        else:
            Gamma_new_c = Gamma_new_c + _EPS * jnp.eye(p)

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
                     mu0, Delta0, Gamma0, W0, mode=None,
                     max_em_iters=10000, n_backtrack=10):
    """Batched, unconstrained-only CFUSN (q=2) EM fit.

    observations, obs_mask : (batch, N, p) — NaNs pre-filled as 0 in
        `observations`; `obs_mask` carries which entries were actually observed.
    sample_idx : (batch, N) int.
    mu0 : (batch, K, p); Delta0 : (batch, K, p, 2); Gamma0 : (batch, K, p, p).
    W0 : (batch, n_samples, K).
    mode : (batch,) int, or None (= all 0). Mixed q=1/q=2/Gaussian restarts
        (see fit.py::generate_fit_jobs): every restart uses uniform q=2-
        shaped tensors, but rows marked here must behave as a lower-q-
        equivalent fit throughout EM, not just at init -- so the relevant
        Delta column(s) are re-zeroed after every M-step update (a q=1
        component IS a q=2 component with a zero second column, and a
        Gaussian component IS one with BOTH columns zero; see
        fit.py::_pad_component_params_to_q's docstring for the algebraic
        proof, applied once for q=1 and twice for Gaussian). 0 = full q=2,
        1 = q=1-equivalent (second column zeroed), 2 = Gaussian-equivalent
        (both columns zeroed). Without this, an M-step would immediately
        start moving a masked restart's column(s) away from zero, silently
        turning it into an ordinary unconstrained q=2 restart after one
        iteration.

    Returns mu, Delta, Gamma, W, failed (batch,) — `failed` is only set on
    NaN/non-finite breakdowns (see module docstring: multivariate fits
    backtrack on a plain likelihood decrease rather than failing outright).
    """
    batch = mu0.shape[0]
    if mode is None:
        mode = jnp.zeros((batch,), dtype=jnp.int32)
    # (batch, 2): [1,0] for q1-mode rows, [0,0] for Gaussian-mode rows,
    # [1,1] otherwise -- multiplying Delta by this after every M-step keeps
    # the masked column(s) at exactly zero for the rest of EM.
    col_mask = jnp.where(
        (mode == 1)[:, None], jnp.array([1.0, 0.0]),
        jnp.where((mode == 2)[:, None], jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])),
    )

    def cond(state):
        it, *_, done = state
        return jnp.logical_and(it < max_em_iters, jnp.logical_not(jnp.all(done)))

    def body(state):
        it, mu, Delta, Gamma, W, ll_prev, failed, done = state

        log_pdfs_old, etas, psis = _densities_all_components(observations, obs_mask, mu, Delta, Gamma)
        resp = _responsibilities_from_logpdfs(log_pdfs_old, sample_idx, W)   # (batch,N,K) -> need (batch,K,N)
        resp = jnp.moveaxis(resp, 2, 1)

        mu2, Delta2, Gamma2 = _m_step(observations, obs_mask, resp, mu, Delta, Gamma,
                                       etas, psis, col_mask=col_mask)
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
    it_final, mu, Delta, Gamma, W, _ll, failed, done = lax.while_loop(cond, body, init_state)
    return mu, Delta, Gamma, W, failed, it_final, done
