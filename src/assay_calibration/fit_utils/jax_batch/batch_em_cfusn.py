"""Batched CFUSN multivariate skew-normal mixture EM -- **unconstrained
only** -- run the way ``cfusn/fit.py::single_fit`` runs a multivariate fit:
the same model, the same E-step/M-step formulas and numerical guards, and the
same loop control. Differences from the CPU path are meant to be confined to
initialization (when ``interop.run_gpu`` uses its batched GPU init) and
floating-point order.

What is mirrored, and from where (cfusn/update_steps.py unless noted):

* E-step, per row, on its observed marginal: the CFUSN q=2 density
  (density_utils.cfusn_logpdf_alternate) and exact Kan & Robotti moments
  (_mc_truncated_mvn_moments' q=2 branch), with the CPU's conditional
  Omega/S/covariance floors; q=1 restarts use the restricted-MSN density and
  scalar moments (msn_logpdf_alternate, get_truncated_normal_moments_mv);
  Gaussian restarts are q=2 with Delta pinned at zero. Per-missingness-
  pattern marginals become one fixed-shape (p, p) system by embedding each
  row's observed block in the identity (_embed_observed), which is exact.
* M-step: completed-data moments (_completed_data_moments), location, the
  q=2 Delta solve with DELTA_SOLVE_RIDGE_FLOOR or the q=1 scalar update, the
  "local" Delta magnitude cap that keeps the previous Delta when it binds,
  the Gamma inverse-Wishart ridge, the measurement-noise floor, and
  _psd_floor_or_reject (rejections keep the component's previous params and
  are counted like guard_stats); per-observation sample weights throughout.
* Loop: the objective is the (weighted) likelihood plus the Gamma IW
  log-prior; single_fit's decrease handling (revert-and-stop / fail), its
  1e-8 convergence test, its 50000-iteration cap, and its best-iterate rule.

Not ported (interop.py refuses fits that need them): constrained/separation
fits, categorical dims, weight_divergence, tied_population, dim weights,
per-component gaussian/frozen components, low-mass freeze.

Memory note: peak memory during the E-step/M-step is `O(batch * N * p^2)`
per component (the embedded `(p,p)` system is built per-observation) --
much larger than the univariate path's `O(batch * N)`. Use a smaller
`max_batch_size` for CFUSN groups in `interop.py` than for univariate ones.
"""
import functools

import numpy as _np
import jax
import jax.numpy as jnp
from jax import lax
import jax.scipy.linalg
from jax.scipy.stats import norm as jnorm
from jax.scipy.special import logsumexp, log_ndtr

_LOG2PI = float(jnp.log(2 * jnp.pi))
_LOG2 = float(jnp.log(2.0))


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


def _sym(A):
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def _floor_min_eig(A, floor, diag_mask=None):
    """The CPU's conditional eigenvalue floor: where lambda_min(A) < floor, add
    (floor - lambda_min + floor) on the diagonal (on the `diag_mask` dims only,
    when A embeds an observed block -- see _embed_observed)."""
    lam = jnp.linalg.eigvalsh(A)[..., 0]
    bump = jnp.where(lam < floor, floor - lam + floor, 0.0)
    d = jnp.eye(A.shape[-1]) if diag_mask is None else diag_mask[..., None, :] * jnp.eye(A.shape[-1])
    return A + bump[..., None, None] * d


def _embed_observed(A, obs_f):
    """A (..., p, p) restricted to each row's observed block, embedded in the
    identity on its missing dims with zero cross terms.

    This is how the CPU's per-missingness-pattern marginals (np.unique over the
    mask, then Omega[o, o] / Gamma[o, o]) become one fixed-shape system: the
    identity block contributes nothing to a log-determinant, a solve against a
    zero right-hand side, or (being >= 1) a lambda_min below the floors above,
    so every quantity computed on the embedded matrix equals the observed-block
    one exactly. (This replaces a large-placeholder-variance device, which left
    an O(log big-M) constant in each row's log-density.)
    """
    keep = obs_f[..., :, None] * obs_f[..., None, :]
    return keep * A + (1.0 - obs_f)[..., None, :] * jnp.eye(A.shape[-1])


def _trunc_norm_moments(m, sigma):
    """Mirrors update_steps.trunc_norm_moments: moments of TN(m, sigma^2, R+)."""
    ratio = m / sigma
    cdf = jnorm.cdf(ratio)
    pdf = jnorm.pdf(ratio)
    safe = cdf > 1e-300
    pr = jnp.where(safe, pdf / jnp.where(safe, cdf, 1.0), jnp.abs(ratio))
    return m + sigma * pr, m ** 2 + sigma ** 2 + sigma * m * pr


def _tn2_moments(means, cov):
    """E[T], E[TT'] for T ~ TN_2(means, cov, R^2_+), row-wise. Mirrors the q=2
    branch of update_steps._mc_truncated_mvn_moments (Kan & Robotti 2017,
    log-space), including its 1e-12 covariance floor, its rho clip (used only
    in log L -- s12 itself stays unclipped there too) and its PSD-preserving
    clip of degenerate rows."""
    cov = _floor_min_eig(_sym(cov), 1e-12)
    mu1, mu2 = means[..., 0], means[..., 1]
    s1sq, s2sq, s12 = cov[..., 0, 0], cov[..., 1, 1], cov[..., 0, 1]
    s1, s2 = jnp.sqrt(s1sq), jnp.sqrt(s2sq)
    rho = jnp.clip(s12 / (s1 * s2), -0.999999, 0.999999)

    alpha1, alpha2 = -mu1 / s1, -mu2 / s2
    log_L = _log_bvn_cdf(-alpha1, -alpha2, rho)

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

    eta = jnp.stack([e1, e2], axis=-1)
    diag0 = mu1 * e1 + s1sq + s12 * r10_2                              # E[T1^2]
    diag1 = mu2 * e2 + s12 * r01_1 + s2sq                              # E[T2^2]
    x_route1 = mu2 * e1 + s12 + s2sq * r10_2                           # kappa=(1,0)
    x_route2 = mu1 * e2 + s1sq * r01_1 + s12                           # kappa=(0,1)
    cross = 0.5 * (x_route1 + x_route2)

    CLIP = 1e4
    eta = jnp.nan_to_num(eta, nan=0.0, posinf=CLIP, neginf=-CLIP)
    eta = jnp.clip(eta, -CLIP, CLIP)
    diag0 = jnp.clip(jnp.nan_to_num(diag0, nan=0.0, posinf=CLIP ** 2, neginf=0.0), 0.0, CLIP ** 2)
    diag1 = jnp.clip(jnp.nan_to_num(diag1, nan=0.0, posinf=CLIP ** 2, neginf=0.0), 0.0, CLIP ** 2)
    cross = jnp.nan_to_num(cross, nan=0.0, posinf=CLIP ** 2, neginf=-CLIP ** 2)
    cross_bound = jnp.sqrt(diag0 * diag1)
    cross = jnp.clip(cross, -cross_bound, cross_bound)
    Psi = jnp.stack([jnp.stack([diag0, cross], -1), jnp.stack([cross, diag1], -1)], -2)
    return eta, Psi


def _component_estep(observations, obs_mask, mu, Delta, Gamma, mode, has_missing):
    """Log-density and E-step moments of one component, for every row.

    observations/obs_mask: (batch, N, p) (missing entries zero-filled);
    mu: (batch, p); Delta: (batch, p, 2); Gamma: (batch, p, p); mode: (batch,)
    int (0 = q=2, 1 = q=1 with Delta's second column zero, 2 = Gaussian with
    Delta zero); has_missing: (batch,) bool, whether that fit's training data
    has any missing entry. Returns log_pdf (batch, N), eta (batch, N, 2),
    Psi (batch, N, 2, 2).

    Mirrors, row by row on each row's observed marginal:
      log_pdf  density_utils.cfusn_logpdf_alternate (q=2 / Gaussian rows) and
               msn_logpdf_alternate (q=1 rows): no Omega floor except the
               Cholesky-failure fallback, D floored at 1e-10 (q=2) / sigma^2 at
               1e-12 (q=1); 0 for a row with nothing observed.
      eta, Psi update_steps.get_truncated_normal_moments_cfusn (q=2 rows; it
               floors Omega_s at 1e-10 only when the fit's data has missing
               entries, which is when it takes its per-pattern branch) and
               get_truncated_normal_moments_mv_missing (q=1 rows: scalar
               moments, second latent coordinate zero); the TN prior moments
               for a row with nothing observed.
    """
    p = mu.shape[-1]
    obs_f = obs_mask.astype(observations.dtype)
    n_obs = obs_f.sum(-1)                                              # (batch,N)
    none_obs = n_obs == 0
    q1 = (mode == 1)[:, None]                                          # (batch,1)

    Omega = _sym(Gamma + jnp.einsum('bpq,brq->bpr', Delta, Delta))
    Om = _embed_observed(Omega[:, None], obs_f)                        # (batch,N,p,p)
    D_o = Delta[:, None] * obs_f[..., None]                            # (batch,N,p,2)
    resid = jnp.where(obs_mask, observations - mu[:, None, :], 0.0)   # (batch,N,p)

    # ── density ────────────────────────────────────────────────────────
    # cho_factor, and on failure the CPU's one retry with
    # (|lambda_min| + 1e-10) added to the observed block.
    L = jnp.linalg.cholesky(Om)
    chol_bad = jnp.any(jnp.isnan(L), axis=(-1, -2))
    lam = jnp.linalg.eigvalsh(Om)[..., 0]
    Om_d = jnp.where(chol_bad[..., None, None],
                     Om + (jnp.abs(lam) + 1e-10)[..., None, None] * (obs_f[..., None, :] * jnp.eye(p)),
                     Om)
    L = jnp.linalg.cholesky(Om_d)
    logdet = 2.0 * jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
    maha = jnp.einsum('bnp,bnp->bn', resid, jnp.linalg.solve(Om_d, resid[..., None])[..., 0])
    log_phi = -0.5 * (n_obs * _LOG2PI + logdet + maha)
    OiD = jnp.linalg.solve(Om_d, D_o)                                  # (batch,N,p,2)
    DtOiD = jnp.einsum('bnpq,bnpr->bnqr', D_o, OiD)                    # (batch,N,2,2)
    m_d = jnp.einsum('bnp,bnpq->bnq', resid, OiD)                      # (batch,N,2)

    Dm = _floor_min_eig(_sym(jnp.eye(2) - DtOiD), 1e-10)
    sd = jnp.sqrt(jnp.maximum(jnp.diagonal(Dm, axis1=-2, axis2=-1), 1e-300))
    rho_d = jnp.clip(Dm[..., 0, 1] / (sd[..., 0] * sd[..., 1]), -0.999999, 0.999999)
    lp2 = 2 * _LOG2 + log_phi + _log_bvn_cdf(m_d[..., 0] / sd[..., 0], m_d[..., 1] / sd[..., 1], rho_d)
    sig2_q1 = jnp.maximum(1.0 - DtOiD[..., 0, 0], 1e-12)
    lp1 = _LOG2 + log_phi + log_ndtr(m_d[..., 0] / jnp.sqrt(sig2_q1))
    log_pdf = jnp.where(q1, lp1, lp2)
    log_pdf = jnp.where(jnp.isfinite(log_pdf), log_pdf, -jnp.inf)
    log_pdf = jnp.where(none_obs, 0.0, log_pdf)

    # ── moments, q=2 / Gaussian rows ───────────────────────────────────
    Om_m = jnp.where(has_missing[:, None, None, None],
                     _floor_min_eig(Om, 1e-10, diag_mask=obs_f), Om)
    OiD_m = jnp.linalg.solve(Om_m, D_o)
    S = _floor_min_eig(_sym(jnp.eye(2) - jnp.einsum('bnpq,bnpr->bnqr', D_o, OiD_m)), 1e-10)
    eta2, Psi2 = _tn2_moments(jnp.einsum('bnp,bnpq->bnq', resid, OiD_m), S)

    # ── moments, q=1 rows (unfloored Omega, no Cholesky retry) ──────────
    OiD0 = jnp.linalg.solve(Om, D_o)
    sig2_m1 = jnp.maximum(1.0 - jnp.einsum('bnp,bnp->bn', D_o[..., 0], OiD0[..., 0]), 1e-12)
    v, w = _trunc_norm_moments(jnp.einsum('bnp,bnp->bn', resid, OiD0[..., 0]), jnp.sqrt(sig2_m1))
    z0 = jnp.zeros_like(v)
    eta1 = jnp.stack([v, z0], axis=-1)
    Psi1 = jnp.stack([jnp.stack([w, z0], -1), jnp.stack([z0, z0], -1)], -2)

    eta = jnp.where(q1[..., None], eta1, eta2)
    Psi = jnp.where(q1[..., None, None], Psi1, Psi2)

    # Nothing observed: the truncated-normal prior moments, as on the CPU
    # (q=2: eta = sqrt(2/pi) 1, Psi = I; q=1: v = sqrt(2/pi), w = 1).
    eta_prior = jnp.where(q1, jnp.array([_SQRT_2_OVER_PI, 0.0]), jnp.array([_SQRT_2_OVER_PI, _SQRT_2_OVER_PI]))
    Psi_prior = jnp.where(q1[..., None], jnp.array([[1.0, 0.0], [0.0, 0.0]]), jnp.eye(2))
    eta = jnp.where(none_obs[..., None], eta_prior[:, None, :], eta)
    Psi = jnp.where(none_obs[..., None, None], Psi_prior[:, None, :, :], Psi)
    return log_pdf, eta, Psi


def _densities_all_components(observations, obs_mask, mu, Delta, Gamma, mode=None, has_missing=None):
    """mu:(batch,K,p) Delta:(batch,K,p,2) Gamma:(batch,K,p,p).
    Returns log_pdfs (batch,K,N), etas (batch,K,N,2), psis (batch,K,N,2,2).
    mode/has_missing: see _component_estep; default all-q=2 / from obs_mask.
    """
    batch, K = mu.shape[:2]
    if mode is None:
        mode = jnp.zeros((batch,), dtype=jnp.int32)
    if has_missing is None:
        has_missing = jnp.logical_not(jnp.all(obs_mask, axis=(1, 2)))
    outs = [_component_estep(observations, obs_mask, mu[:, k], Delta[:, k], Gamma[:, k],
                             mode, has_missing) for k in range(K)]
    return tuple(jnp.stack(x, axis=1) for x in zip(*outs))


def _responsibilities_from_logpdfs(log_pdfs, sample_idx, W):
    """log_pdfs:(batch,K,N); W:(batch,S,K) -> P:(batch,N,K)."""
    batch = W.shape[0]
    w_n = W[jnp.arange(batch)[:, None], sample_idx, :]                  # (batch,N,K)
    log_w = jnp.where(w_n > 0, jnp.log(jnp.where(w_n > 0, w_n, 1.0)), -jnp.inf)
    numer = jnp.moveaxis(log_pdfs, 1, 2) + log_w                        # (batch,N,K)
    denom = logsumexp(numer, axis=-1, keepdims=True)
    return jnp.nan_to_num(jnp.exp(numer - denom), nan=0.0)


def _weights_and_ll(log_pdfs, sample_idx, W_old, n_samples, sample_w=None):
    """Matches get_sample_weights_and_ll: weights from OLD W as prior against
    NEW-param densities, LL from the newly updated weights.

    sample_w: (batch,N) per-observation M-step weights (cfusn/fit.py's
    compute_sample_weights), or None. The W step is unchanged by them (they
    are constant within a sample, so they factor out of its argmax); the LL
    is the weighted sum_n w_n log f(x_n) the weighted M-step increases.
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
    if sample_w is not None:
        log_mix = sample_w * log_mix
    ll = log_mix.sum(axis=-1) / N
    return W_new, ll



# ── Completed-data moments (port of cfusn/update_steps._completed_data_moments) ──

# Gamma regularisation and Delta-cap settings, mirroring the CPU path's
# USE_GAMMA_RIDGE / GAMMA_RIDGE_FRAC / DELTA_SOLVE_RIDGE_FLOOR /
# DELTA_CAP_SCALE="local" / DELTA_CAP_PD_CORRECTION so both implementations
# fit the same model. See cfusn/update_steps.py for the derivations.
_USE_GAMMA_RIDGE = True
_GAMMA_RIDGE_FRAC = 1e-3
_DELTA_SOLVE_RIDGE_FLOOR = 1e-3
_DELTA_CAP_PD_CORRECTION = True
_DELTA_CAP_SAFETY_FACTOR = 0.95
_TN_VAR = 1.0 - 2.0 / jnp.pi          # Var(T_i) for T ~ TN(0,1) on R_+
_SQRT_2_OVER_PI = float(_np.sqrt(2.0 / _np.pi))

# cfusn/fit.py::single_fit's loop constants (multivariate).
MV_MAX_EM_ITERS = 50000
_PLATEAU_REL_TOL = 1e-8
_CATASTROPHIC_REL_TOL = 1e-1
_CONVERGENCE_REL_TOL = 1e-8


def _completed_moments(observations, obs_mask, mu, Delta, Gamma, eta, Psi):
    """Posterior moments of the COMPLETE x given x_obs, for one component.

    Conditional on T=t, x ~ N(mu + Delta t, Gamma), so for missing dims m and
    observed dims o (with K = Gamma_mo Gamma_oo^-1):

        x_m | x_o, t ~ N(a_m + B_m t, C_mm)
        a_m = mu_m + K (x_o - mu_o),  B_m = Delta_m - K Delta_o,
        C_mm = Gamma_mm - K Gamma_om

    Gamma_oo carries the CPU's 1e-10 floor (_regularized_spd); a row with
    nothing observed gets the component's own prior moments (a = mu,
    B = Delta, C = Gamma), as on the CPU.
    Returns Ex (batch,N,p), Ext (batch,N,p,q), Exx (batch,N,p,p).
    """
    obs_f = obs_mask.astype(observations.dtype)
    Gb = Gamma[:, None, :, :]                                        # (batch,1,p,p)
    G_aug = _floor_min_eig(_embed_observed(_sym(Gb), obs_f), 1e-10, diag_mask=obs_f)
    miss = jnp.logical_not(obs_mask)                                 # (batch,N,p)
    G_full = jnp.broadcast_to(Gb, G_aug.shape)

    # a: observed entries keep x; missing entries take mu_m + K (x_o - mu_o).
    resid_o = jnp.where(obs_mask, observations - mu[:, None, :], 0.0)
    K_resid = jnp.einsum('bnpr,bnr->bnp', G_full, jnp.linalg.solve(G_aug, resid_o[..., None])[..., 0])
    a = jnp.where(obs_mask, observations, mu[:, None, :] + K_resid)

    # B: zero on observed rows, Delta_m - K Delta_o on missing rows.
    Delta_o = jnp.where(obs_mask[..., None], Delta[:, None, :, :], 0.0)
    K_Delta = jnp.einsum('bnpr,bnrq->bnpq', G_full, jnp.linalg.solve(G_aug, Delta_o))
    B = jnp.where(miss[..., None], Delta[:, None, :, :] - K_Delta, 0.0)

    # C: Gamma_mm - K Gamma_om on the missing-missing block, zero elsewhere.
    G_om = jnp.where(obs_mask[..., :, None] & miss[..., None, :], Gb, 0.0)
    K_Gom = jnp.einsum('bnpr,bnrc->bnpc', G_full, jnp.linalg.solve(G_aug, G_om))
    C = jnp.where(miss[..., :, None] & miss[..., None, :], Gb - K_Gom, 0.0)
    C = _sym(C)

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
    Gamma_new = S_c / n_c. Mirrors cfusn/update_steps._ridge_gamma, including
    leaving Gamma (symmetrized) unchanged when the data variance is not a
    positive finite number."""
    p = Gamma_new.shape[-1]
    nu = p + 2
    G = _sym(Gamma_new)
    ok = jnp.isfinite(data_var) & (data_var > 0)
    Psi = _GAMMA_RIDGE_FRAC * jnp.where(ok, data_var, 1.0)[:, None, None] * jnp.eye(p)
    n = jnp.maximum(n_eff, 0.0)[:, None, None]
    return jnp.where(ok[:, None, None], (G * n + Psi) / (n + nu + p + 1), G)


def noise_floor_factors(noise, p):
    """Per-fit inputs for _floor_gamma_at_noise_jax, built once in NumPy
    exactly as cfusn/update_steps._floor_gamma_at_noise builds them.

    noise: (p,p) noise covariance, length-p variances, or None. Returns
    (L, f): L (p,p) is the Cholesky factor of Sigma_FF embedded in the
    identity on the unfloored readouts U, and f (p,) is 1.0 on the floored
    readouts F (finite, positive noise variance). None/no F gives (I, 0),
    for which the floor is the identity map.
    """
    L_full = _np.eye(p)
    f = _np.zeros(p)
    if noise is None:
        return L_full, f
    S = _np.asarray(noise, dtype=float)
    if S.ndim == 1:
        S = _np.diag(S)
    var = _np.diag(S)
    F = _np.flatnonzero(_np.isfinite(var) & (var > 0))
    if not len(F):
        return L_full, f
    SF = _np.nan_to_num(S[_np.ix_(F, F)], nan=0.0)
    SF = 0.5 * (SF + SF.T)
    try:
        L = _np.linalg.cholesky(SF)
    except _np.linalg.LinAlgError:
        L = _np.linalg.cholesky(SF + 1e-9 * _np.mean(_np.diag(SF)) * _np.eye(len(F)))
    L_full[_np.ix_(F, F)] = L
    f[F] = 1.0
    return L_full, f


def _floor_gamma_at_noise_jax(Gamma, noise_L, floor_mask):
    """Fixed-shape port of cfusn/update_steps._floor_gamma_at_noise (see its
    docstring for the derivation): Gamma_FF >= Sigma_FF via generalized-
    eigenvalue clipping, with the increase propagated to the unfloored
    readouts through their regression on F.

    Gamma: (batch,p,p); noise_L, floor_mask: stacked noise_floor_factors.
    The data-dependent F/U index sets become masks: Gamma_FF is embedded in
    the identity on U, so the whitened matrix's U block is exactly I, its
    eigenvalues there are 1, and max(eig, 1) - eig adds nothing outside F.
    A row with no floored readouts is returned symmetrized and otherwise
    unchanged, like the NumPy function.
    """
    p = Gamma.shape[-1]
    I = jnp.eye(p)
    f = floor_mask.astype(Gamma.dtype)                                  # (batch,p)
    u = 1.0 - f
    FF = f[:, :, None] * f[:, None, :]
    G = 0.5 * (Gamma + jnp.swapaxes(Gamma, -1, -2))
    G_F = FF * G + u[:, None, :] * I                                    # blockdiag(G_FF, I)
    Li = jax.scipy.linalg.solve_triangular(noise_L, jnp.broadcast_to(I, G.shape), lower=True)
    W = Li @ G_F @ jnp.swapaxes(Li, -1, -2)
    W = 0.5 * (W + jnp.swapaxes(W, -1, -2))
    eig, vec = jnp.linalg.eigh(W)
    clip = jnp.maximum(eig, 1.0) - eig
    delta = noise_L @ (vec * clip[:, None, :]) @ jnp.swapaxes(vec, -1, -2) @ jnp.swapaxes(noise_L, -1, -2)
    delta = 0.5 * (delta + jnp.swapaxes(delta, -1, -2))
    # E = [I; B] with B = G_UF G_FF^-1: rows F are the identity, rows U the
    # regression; columns U are zero (delta has no U entries to carry).
    G_FU = f[:, :, None] * u[:, None, :] * G
    E = f[:, :, None] * I + jnp.swapaxes(jnp.linalg.solve(G_F, G_FU), -1, -2)
    out = G + E @ delta @ jnp.swapaxes(E, -1, -2)
    return 0.5 * (out + jnp.swapaxes(out, -1, -2))


def _psd_floor_or_reject_jax(G, floor=1e-8, reject_rel=1e-8):
    """Mirrors update_steps._psd_floor_or_reject: returns (G_fixed, ok).
    Non-finite candidates, and ones with a genuinely negative eigenvalue
    (lambda_min < -reject_rel * max|lambda|), are rejected; round-off dust
    below `floor` is floored."""
    finite = jnp.all(jnp.isfinite(G), axis=(-1, -2))
    G_safe = jnp.where(finite[:, None, None], G, jnp.eye(G.shape[-1]))
    eig = jnp.linalg.eigvalsh(G_safe)
    lam_min = eig[..., 0]
    lam_max = jnp.max(jnp.abs(eig), axis=-1)
    ok = finite & jnp.all(jnp.isfinite(eig), axis=-1)
    low = lam_min < floor
    ok = ok & jnp.logical_not(low & (lam_min < -reject_rel * jnp.maximum(lam_max, 1e-300)))
    G_fixed = jnp.where(low[:, None, None], G + (floor - lam_min)[:, None, None] * jnp.eye(G.shape[-1]), G)
    return G_fixed, ok


def _m_step(observations, obs_mask, resp, mu, Delta, Gamma, etas, psis, mode, data_var,
            sample_w=None, neff_kish=False, noise_L=None, floor_mask=None):
    """One unconstrained M-step for all K components, mirroring
    update_steps._em_update_cfusn (q=2 and Gaussian rows) and
    _em_update_multivariate (q=1 rows) component by component.

    resp: (batch,K,N) responsibilities; etas/psis: (batch,K,N,2[,2]) E-step
    moments at the OLD params (shared by the location, Delta and Gamma steps,
    as on the CPU). mode: (batch,) see _component_estep. data_var: (batch,)
    mean per-dimension data variance (ridge scale). sample_w: (batch,N) per-
    observation weights or None -- every sufficient statistic uses
    z * sample_w, the CPU's z_eff, and so does the ridge's n_eff (Kish
    effective size when neff_kish, matching _component_n_eff). noise_L /
    floor_mask: noise_floor_factors per row, or None for no floor.

    Returns mu, Delta, Gamma (batch,K,...), plus per-row (batch,) counts of
    Gamma-rejection events and the largest n_eff a rejection hit -- the CPU's
    guard_stats["gamma_rejected"] / ["gamma_rejected_max_mass"].
    """
    K, p = mu.shape[1], mu.shape[2]
    q = Delta.shape[-1]
    is_q1 = (mode == 1)
    is_gauss = (mode == 2)

    new_mu, new_Delta, new_Gamma = [], [], []
    n_rej = jnp.zeros(mu.shape[0], dtype=jnp.int32)
    rej_mass = jnp.zeros(mu.shape[0])
    for c in range(K):
        z = resp[:, c, :]                                                # (batch,N)
        if sample_w is not None:
            z = z * sample_w
        eta_c, Psi_c = etas[:, c], psis[:, c]                            # (batch,N,q),(batch,N,q,q)
        mu_old, Delta_old, Gamma_old = mu[:, c], Delta[:, c], Gamma[:, c]

        Ex, Ext, Exx = _completed_moments(
            observations, obs_mask, mu_old, Delta_old, Gamma_old, eta_c, Psi_c)
        z_sum = z.sum(axis=1)                                            # (batch,)
        z_div = jnp.maximum(z_sum, 1e-12)

        # location (get_location_update_cfusn / _mv): weighted mean of
        # E[x] - Delta E[T]; an unbacked component keeps its previous mu.
        Delta_eta = jnp.einsum('bnq,bpq->bnp', eta_c, Delta_old)
        mu_new = jnp.einsum('bn,bnp->bp', z, Ex - Delta_eta) / z_div[:, None]
        mu_new = jnp.where((z_sum > 1e-12)[:, None], mu_new, mu_old)

        # Delta, q=2 (_delta_update_completed): one q x q solve against
        # sum z E[TT'], conditionally ridged as DELTA_SOLVE_RIDGE_FLOOR.
        numer = (jnp.einsum('bn,bnpq->bpq', z, Ext)
                 - mu_new[:, :, None] * jnp.einsum('bn,bnq->bq', z, eta_c)[:, None, :])
        Psi_sum = jnp.einsum('bn,bnij->bij', z, Psi_c)
        eig_min = jnp.linalg.eigvalsh(_sym(Psi_sum))[..., 0]
        bump = jnp.where(eig_min < _DELTA_SOLVE_RIDGE_FLOOR,
                         _DELTA_SOLVE_RIDGE_FLOOR - eig_min + _DELTA_SOLVE_RIDGE_FLOOR, 0.0)
        Psi_sum = Psi_sum + bump[:, None, None] * jnp.eye(q)
        Delta_q2 = jnp.swapaxes(jnp.linalg.solve(Psi_sum, jnp.swapaxes(numer, -1, -2)), -1, -2)

        # Delta, q=1 (get_Delta_update_mv): scalar denominator sum z E[T^2].
        w_sum = jnp.einsum('bn,bn->b', z, Psi_c[..., 0, 0])
        d1 = numer[..., 0] / jnp.where(w_sum > 1e-12, w_sum, 1.0)[:, None]
        d1 = jnp.where((w_sum > 1e-12)[:, None], d1, Delta_old[..., 0])
        Delta_q1 = jnp.stack([d1, jnp.zeros_like(d1)], axis=-1)

        Delta_cand = jnp.where(is_q1[:, None, None], Delta_q1, Delta_q2)

        # Magnitude cap (_apply_delta_cap with Delta_prev, DELTA_CAP_SCALE=
        # "local"): bound each row by the component's own responsibility-
        # weighted completed variance; a binding cap rejects the candidate and
        # keeps the previous Delta.
        var_rows = (jnp.diagonal(Exx, axis1=-2, axis2=-1)
                    - 2.0 * Ex * mu_new[:, None, :] + (mu_new ** 2)[:, None, :])
        wvar = jnp.einsum('bn,bnp->bp', z, var_rows) / jnp.maximum(z_sum, 1e-8)[:, None]
        wvar = jnp.where((z_sum > 1e-8)[:, None], wvar, 0.0)
        wvar = jnp.maximum(wvar, 0.0)
        if _DELTA_CAP_PD_CORRECTION:
            wvar = wvar / _TN_VAR
        max_norm = _DELTA_CAP_SAFETY_FACTOR * jnp.sqrt(wvar)
        capped = jnp.any(jnp.linalg.norm(Delta_cand, axis=-1) > max_norm, axis=-1)
        Delta_new = jnp.where(capped[:, None, None], Delta_old, Delta_cand)
        # Gaussian rows pin Delta at zero (force_gaussian), uncapped.
        Delta_new = jnp.where(is_gauss[:, None, None], 0.0, Delta_new)

        # Gamma from completed sufficient statistics (_gamma_update_completed).
        S_xx = jnp.einsum('bn,bnac->bac', z, Exx)
        S_xt = jnp.einsum('bn,bnaq->baq', z, Ext)
        S_x = jnp.einsum('bn,bna->ba', z, Ex)
        S_tt = jnp.einsum('bn,bnij->bij', z, Psi_c)
        S_t = jnp.einsum('bn,bni->bi', z, eta_c)
        D = Delta_new
        Gamma_new = (
            S_xx
            - jnp.einsum('baq,bcq->bac', S_xt, D) - jnp.einsum('baq,bcq->bca', S_xt, D)
            + jnp.einsum('bai,bij,bcj->bac', D, S_tt, D)
            - mu_new[:, :, None] * S_x[:, None, :] - S_x[:, :, None] * mu_new[:, None, :]
            + mu_new[:, :, None] * jnp.einsum('bi,bci->bc', S_t, D)[:, None, :]
            + jnp.einsum('bai,bi->ba', D, S_t)[:, :, None] * mu_new[:, None, :]
            + z_sum[:, None, None] * mu_new[:, :, None] * mu_new[:, None, :]
        ) / z_div[:, None, None]
        Gamma_new = jnp.where((z_sum > 1e-12)[:, None, None], _sym(Gamma_new), 0.0)

        # Ridge, then the measurement-noise floor, then PSD floor-or-reject.
        n_eff = z_sum
        if neff_kish and sample_w is not None:
            denom = (z * sample_w).sum(axis=1)
            n_eff = jnp.where(denom > 0, z_sum ** 2 / jnp.where(denom > 0, denom, 1.0), 0.0)
        if _USE_GAMMA_RIDGE:
            Gamma_new = _ridge_gamma_jax(Gamma_new, n_eff, data_var)
        if noise_L is not None:
            Gamma_new = _floor_gamma_at_noise_jax(Gamma_new, noise_L, floor_mask)
        Gamma_new, ok = _psd_floor_or_reject_jax(Gamma_new)

        n_rej = n_rej + jnp.logical_not(ok).astype(jnp.int32)
        rej_mass = jnp.where(ok, rej_mass, jnp.maximum(rej_mass, n_eff))
        new_mu.append(jnp.where(ok[:, None], mu_new, mu_old))
        new_Delta.append(jnp.where(ok[:, None, None], Delta_new, Delta_old))
        new_Gamma.append(jnp.where(ok[:, None, None], Gamma_new, Gamma_old))

    return (jnp.stack(new_mu, axis=1), jnp.stack(new_Delta, axis=1),
            jnp.stack(new_Gamma, axis=1), n_rej, rej_mass)


def _gamma_log_prior(Gamma, data_var, n_obs):
    """update_steps.gamma_log_prior_total / N: the IW log-prior the ridged
    M-step maximises alongside the likelihood. -inf for a row where any
    component's Gamma is not positive definite; 0 when the data variance is
    not a positive finite number."""
    p = Gamma.shape[-1]
    nu = p + 2
    ok = jnp.isfinite(data_var) & (data_var > 0)
    G = _sym(Gamma)
    sign, logdet = jnp.linalg.slogdet(G)                               # (batch,K)
    Psi = _GAMMA_RIDGE_FRAC * jnp.where(ok, data_var, 1.0)[:, None, None, None] * jnp.eye(p)
    tr = jnp.trace(jnp.linalg.solve(G, jnp.broadcast_to(Psi, G.shape)), axis1=-2, axis2=-1)
    terms = -0.5 * (nu + p + 1) * logdet - 0.5 * tr
    bad = jnp.any((sign <= 0) | jnp.logical_not(jnp.isfinite(logdet)), axis=-1)
    total = jnp.where(bad, -jnp.inf, terms.sum(-1))
    if not _USE_GAMMA_RIDGE:
        return jnp.zeros_like(data_var)
    return jnp.where(ok, total, 0.0) / n_obs


@functools.partial(
    jax.jit,
    static_argnums=(3,),
    static_argnames=("max_em_iters", "neff_kish"),
)
def fit_batch_cfusn(observations, obs_mask, sample_idx, n_samples,
                     mu0, Delta0, Gamma0, W0, mode=None,
                     sample_w=None, noise_L=None, floor_mask=None, neff_kish=False,
                     max_em_iters=MV_MAX_EM_ITERS):
    """Batched, unconstrained CFUSN EM fit, run exactly as cfusn/fit.py's
    single_fit runs a multivariate fit.

    observations, obs_mask : (batch, N, p) -- NaNs pre-filled as 0 in
        `observations`; `obs_mask` carries which entries were actually observed.
    sample_idx : (batch, N) int.
    mu0 : (batch, K, p); Delta0 : (batch, K, p, 2); Gamma0 : (batch, K, p, p).
    W0 : (batch, n_samples, K).
    mode : (batch,) int, or None (= all 0): 0 = q=2 restart, 1 = q=1 restart
        (Delta's second column held at zero; E-step, density and Delta update
        are the CPU's restricted-MSN ones, not a q=2 fit with a column
        zeroed after the fact), 2 = Gaussian restart (Delta held at zero).
    sample_w : (batch, N) per-observation M-step weights, or None (= 1).
    noise_L, floor_mask : (batch, p, p), (batch, p) from noise_floor_factors,
        or None for no measurement-noise floor.
    neff_kish : static; weight_neff="kish" for the ridge's n_eff.
    max_em_iters : static; single_fit's MAX_EM_ITERS (multivariate default
        50000) -- the loop runs one initial EM step plus up to this many more.

    Loop control mirrors single_fit: the objective is the (weighted)
    likelihood plus the Gamma IW log-prior; no decrease check on the first
    two steps; after that a relative decrease >= 1e-1 fails the fit, one
    > 1e-8 reverts to the previous iterate and stops, and a relative change
    < 1e-8 stops; the best-objective iterate seen is returned if the last one
    is worse. NaN weights fail immediately, NaN likelihoods at the next step.

    Returns mu, Delta, Gamma, W, failed, it_final, done, hit_cap, n_rejected,
    rejected_max_mass -- `failed`, `hit_cap` (loop ran out of iterations),
    `n_rejected` and `rejected_max_mass` per row (batch,); it_final the
    number of EM steps taken by the slowest row.
    """
    batch, K = mu0.shape[:2]
    N = observations.shape[1]
    if mode is None:
        mode = jnp.zeros((batch,), dtype=jnp.int32)
    has_missing = jnp.logical_not(jnp.all(obs_mask, axis=(1, 2)))

    # Ridge / prior scale: nanmean over dims of the per-dim (population)
    # variance of the observed entries, as _ridge_gamma computes it.
    obs_f = obs_mask.astype(observations.dtype)
    cnt = obs_f.sum(axis=1)                                              # (batch,p)
    mean_d = (obs_f * observations).sum(axis=1) / jnp.maximum(cnt, 1.0)
    var_d = (obs_f * (observations - mean_d[:, None, :]) ** 2).sum(axis=1) / jnp.maximum(cnt, 1.0)
    var_d = jnp.where(cnt > 0, var_d, jnp.nan)
    data_var = jnp.nanmean(var_d, axis=-1)                               # (batch,)

    def estep(mu, Delta, Gamma):
        return _densities_all_components(observations, obs_mask, mu, Delta, Gamma, mode, has_missing)

    def em_step(mu, Delta, Gamma, W, log_pdfs, etas, psis):
        """One em_iteration: E-step at the current params (its densities and
        moments carried from the previous step), M-step, then the W update
        and LL at the new params. Returns the new iterate, its objective,
        E-step quantities for the next step and the guard counts."""
        resp = jnp.moveaxis(_responsibilities_from_logpdfs(log_pdfs, sample_idx, W), 2, 1)
        mu2, Delta2, Gamma2, n_rej, rej_mass = _m_step(
            observations, obs_mask, resp, mu, Delta, Gamma, etas, psis, mode, data_var,
            sample_w=sample_w, neff_kish=neff_kish, noise_L=noise_L, floor_mask=floor_mask)
        log_pdfs2, etas2, psis2 = estep(mu2, Delta2, Gamma2)
        W2, ll2 = _weights_and_ll(log_pdfs2, sample_idx, W, n_samples, sample_w)
        obj2 = ll2 + _gamma_log_prior(Gamma2, data_var, N)
        nan_w = jnp.any(jnp.isnan(W2), axis=(1, 2))
        return (mu2, Delta2, Gamma2, W2), ll2, obj2, (log_pdfs2, etas2, psis2), n_rej, rej_mass, nan_w

    def where_rows(m, new, old):
        return jax.tree_util.tree_map(
            lambda a, b: jnp.where(m.reshape(m.shape + (1,) * (a.ndim - 1)), a, b), new, old)

    # Initial likelihood (likelihoods[0]): only its NaN-ness is ever used.
    cache0 = estep(mu0, Delta0, Gamma0)
    batch_ix = jnp.arange(batch)[:, None]
    lw0 = W0[batch_ix, sample_idx, :]
    log_mix0 = logsumexp(jnp.moveaxis(cache0[0], 1, 2)
                         + jnp.where(lw0 > 0, jnp.log(jnp.where(lw0 > 0, lw0, 1.0)), -jnp.inf), axis=-1)
    if sample_w is not None:
        log_mix0 = sample_w * log_mix0
    ll_nan = jnp.isnan(log_mix0.sum(-1))

    # First EM step: no checks; it seeds the best-iterate tracking.
    params1, ll1, obj1, cache1, n_rej1, mass1, nan_w1 = em_step(mu0, Delta0, Gamma0, W0, *cache0)
    ll_nan = ll_nan | jnp.isnan(ll1)

    init_state = dict(
        it=jnp.zeros((), dtype=jnp.int32),
        cur=params1, cache=cache1, obj=obj1, best=params1, best_obj=obj1,
        ll_nan=ll_nan, failed=nan_w1, done=nan_w1,
        hit_cap=jnp.zeros((batch,), dtype=bool),
        n_rej=n_rej1, rej_mass=mass1,
    )

    def cond(st):
        return jnp.logical_and(st["it"] < max_em_iters, jnp.logical_not(jnp.all(st["done"])))

    def body(st):
        it = st["it"]
        active = jnp.logical_not(st["done"])
        # Top of single_fit's loop: NaN in any likelihood so far fails the fit.
        nan_fail = active & st["ll_nan"]
        active = active & jnp.logical_not(nan_fail)

        new, ll2, obj2, cache2, n_rej, rej_mass, nan_w = em_step(*st["cur"], *st["cache"])
        nan_fail = nan_fail | (active & nan_w)
        active = active & jnp.logical_not(nan_w)

        obj_old = st["obj"]
        improved = active & (obj2 > st["best_obj"])
        best = where_rows(improved, new, st["best"])
        best_obj = jnp.where(improved, obj2, st["best_obj"])

        rel_dec = (obj_old - obj2) / jnp.maximum(jnp.abs(obj_old), 1e-300)
        decreased = active & (it > 0) & (obj2 < obj_old)
        catastrophic = decreased & (rel_dec >= _CATASTROPHIC_REL_TOL)
        revert = decreased & jnp.logical_not(catastrophic) & (rel_dec > _PLATEAU_REL_TOL)
        advance = active & jnp.logical_not(catastrophic | revert)
        converged = advance & (it >= 1) & (jnp.abs(obj2 - obj_old) / jnp.abs(obj_old) < _CONVERGENCE_REL_TOL)

        cur = where_rows(advance, new, st["cur"])
        cache = where_rows(advance, cache2, st["cache"])
        obj = jnp.where(advance, obj2, obj_old)
        failed = st["failed"] | nan_fail | catastrophic
        stopped = nan_fail | catastrophic | revert | converged
        last = it == max_em_iters - 1
        hit_cap = st["hit_cap"] | (active & jnp.logical_not(stopped) & last)
        return dict(
            it=it + 1, cur=cur, cache=cache, obj=obj, best=best, best_obj=best_obj,
            ll_nan=st["ll_nan"] | (advance & jnp.isnan(ll2)),
            failed=failed, done=st["done"] | stopped, hit_cap=hit_cap,
            n_rej=st["n_rej"] + jnp.where(active, n_rej, 0),
            rej_mass=jnp.where(active, jnp.maximum(st["rej_mass"], rej_mass), st["rej_mass"]),
        )

    st = lax.while_loop(cond, body, init_state)
    # Return the best iterate when the last one is worse (single_fit's
    # best_state rule).
    use_best = st["obj"] < st["best_obj"]
    mu, Delta, Gamma, W = where_rows(use_best, st["best"], st["cur"])
    return (mu, Delta, Gamma, W, st["failed"], st["it"] + 1, st["done"] | st["hit_cap"],
            st["hit_cap"], st["n_rej"], st["rej_mass"])
