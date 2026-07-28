"""JAX-traceable univariate skew-normal mixture math.

Every function here operates on arrays with a leading ``batch`` axis (one
independent EM fit per batch element) instead of a single unbatched fit —
"batching" is just an extra vectorized axis, not ``jax.vmap`` over a
per-fit function, which keeps this code a straightforward port of the
elementwise/broadcast NumPy in
``cfusn/density_utils.py`` and ``cfusn/update_steps.py``.

Mirrors (NumPy reference, for parity testing):
  - density_utils.canonical_to_alternate / alternate_to_canonical
  - update_steps.trunc_norm_moments / get_truncated_normal_moments
  - scipy.stats.skewnorm.logpdf (re-derived, not JAX-traceable upstream)
"""
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

_LOG2 = float(jnp.log(2.0))


def skewnorm_logpdf(x, a, loc, scale):
    """log-pdf of skew-normal(a, loc, scale) — matches scipy.stats.skewnorm.logpdf."""
    z = (x - loc) / scale
    return _LOG2 - jnp.log(scale) + jnorm.logpdf(z) + jnorm.logcdf(a * z)


def canonical_to_alternate(a, loc, scale):
    """(a, loc, scale) -> (loc, Delta, Gamma). Matches density_utils.canonical_to_alternate."""
    delta = a / jnp.sqrt(1.0 + a ** 2)
    Delta = scale * delta
    Gamma = scale ** 2 - Delta ** 2
    return loc, Delta, Gamma


def alternate_to_canonical(loc, Delta, Gamma, eps=1e-12):
    """(loc, Delta, Gamma) -> (a, loc, scale). Matches density_utils.alternate_to_canonical.

    NumPy raises ZeroDivisionError on Gamma<=0 / non-finite `a`; the batched
    path can't raise per-element, so Gamma is floored at `eps` instead —
    degenerate components should instead be caught by the caller via
    likelihood/NaN checks, matching the effect (not the exact exception) of
    the NumPy behavior.
    """
    Gamma_safe = jnp.maximum(Gamma, eps)
    a = jnp.sign(Delta) * jnp.sqrt(jnp.maximum(Delta ** 2, 0.0) / Gamma_safe)
    scale = jnp.sqrt(Gamma_safe + Delta ** 2)
    return a, loc, scale


def trunc_norm_moments(mu, sigma):
    """Moments of TN(mu, sigma^2, R+). Matches update_steps.trunc_norm_moments."""
    ratio = mu / sigma
    cdf = jnorm.cdf(ratio)
    pdf = jnorm.pdf(ratio)
    safe = cdf > 1e-300
    p = jnp.where(safe, pdf / jnp.where(safe, cdf, 1.0), jnp.abs(ratio))
    m1 = mu + sigma * p
    m2 = mu ** 2 + sigma ** 2 + sigma * mu * p
    return m1, m2


def truncated_moments_from_canonical(x, a, loc, scale):
    """Matches update_steps.get_truncated_normal_moments (univariate SN, canonical params).

    x : (..., N) observations, broadcastable against a/loc/scale's leading dims
        (typically (batch, 1) so the result broadcasts to (batch, N)).
    """
    delta = a / jnp.sqrt(1.0 + a ** 2)
    tn_loc = delta / scale * (x - loc)
    tn_scale = jnp.sqrt(jnp.maximum(1.0 - delta ** 2, 1e-12))
    return trunc_norm_moments(tn_loc, tn_scale)
