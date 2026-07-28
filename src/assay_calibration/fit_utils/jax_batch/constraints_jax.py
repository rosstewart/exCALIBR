"""Batched univariate density-ratio monotonicity constraint, mirroring
``cfusn/constraints.py``'s univariate path (``multicomponent_density_constraint_violated``
with ``multivariate=False``) and ``cfusn/update_steps.py``'s ``binary_search``.

Known simplification vs. the NumPy reference (flag for parity testing):
``_adjacent_pair_violated`` there computes ``np.diff`` on the *compacted*
sequence of grid points passing the mass-floor mask (so it correctly skips
gaps of excluded points when checking monotonicity). This batched version
instead only compares directly-adjacent grid indices that are *both* valid,
which differs when a mass-floor-excluded gap sits between two valid regions
in the middle of the grid. For a unimodal skew-normal component this gap is
expected to be rare/nonexistent in practice (mass typically forms one
contiguous region), but this needs confirming against the NumPy reference in
the parity tests, especially for near-degenerate fits.
"""
import jax.numpy as jnp
from jax import lax

from .skewnorm_jax import skewnorm_logpdf, alternate_to_canonical

_GRID_N = 1000
_MASS_FLOOR = -7.0


def build_grid(xmin, xmax, n=_GRID_N):
    """xmin, xmax: (batch,) -> (batch, n) grid, matching np.linspace(*xlims, 1000) per fit."""
    t = jnp.linspace(0.0, 1.0, n)
    return xmin[:, None] + t[None, :] * (xmax - xmin)[:, None]


def _adjacent_pair_violated(log_pdfs, mass_floor=_MASS_FLOOR):
    """log_pdfs: (K, batch, n) -> (batch,) bool, any adjacent-pair violation.

    See module docstring for the one known deviation from the NumPy
    reference (compacted-diff vs. adjacent-masked-diff).
    """
    batch = log_pdfs.shape[1]
    violated = jnp.zeros(batch, dtype=bool)
    for i in range(log_pdfs.shape[0] - 1):
        ix = (log_pdfs[i] > mass_floor) & (log_pdfs[i + 1] > mass_floor)  # (batch, n)
        log_ratio = log_pdfs[i] - log_pdfs[i + 1]
        diffs = jnp.diff(log_ratio, axis=-1)  # (batch, n-1)
        both_valid = ix[:, :-1] & ix[:, 1:]
        pair_violated = jnp.any((diffs > 0.0) & both_valid, axis=-1)
        n_valid = ix.sum(axis=-1)
        violated = violated | (pair_violated & (n_valid >= 2))
    return violated


def constraint_violated(canonical_params, x_grid):
    """canonical_params: list of K (a, loc, scale) each (batch,); x_grid: (batch, n).

    Mirrors constraints.multicomponent_density_constraint_violated(multivariate=False).
    """
    log_pdfs = jnp.stack([
        skewnorm_logpdf(x_grid, a[:, None], loc[:, None], scale[:, None])
        for a, loc, scale in canonical_params
    ], axis=0)  # (K, batch, n)
    return _adjacent_pair_violated(log_pdfs)


def binary_search(candidate_value, current_alt_params, component_index, param_index,
                   x_grid, max_iters=40):
    """Batched bisection matching update_steps.binary_search's contract.

    current_alt_params : list of K tuples (loc, Delta, Gamma), each entry a
        (batch,) array — the "current_alternate_params" the NumPy version
        builds from whatever mix of already-updated/not-yet-updated
        component params the caller assembled (see batch_em.py, which
        mirrors get_constrained_{location,Delta,Gamma}_update's bsearch_params
        construction).
    component_index, param_index : Python ints (static) — which component and
        which of (loc=0, Delta=1, Gamma=2) is being bisected.
    candidate_value : (batch,) unconstrained M-step candidate for that param.

    Returns the constrained (batch,) value, i.e. the NumPy `lower_bound` at
    convergence. NumPy loops `while abs(upper-lower) > 1e-4`; this instead
    runs a fixed `max_iters` (40 halvings comfortably exceeds the precision
    reached by that while-loop for any realistic parameter range), which is
    required for lax.fori_loop/JIT tracing.
    """
    lower0 = current_alt_params[component_index][param_index]
    upper0 = candidate_value

    def _trial_canonical(mid):
        params = []
        for c, (loc, Delta, Gamma) in enumerate(current_alt_params):
            if c == component_index:
                vals = [loc, Delta, Gamma]
                vals[param_index] = mid
                loc, Delta, Gamma = vals
            params.append(alternate_to_canonical(loc, Delta, Gamma))
        return params

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        violated = constraint_violated(_trial_canonical(mid), x_grid)
        new_hi = jnp.where(violated, mid, hi)
        new_lo = jnp.where(violated, lo, mid)
        return (new_lo, new_hi)

    lo, hi = lax.fori_loop(0, max_iters, body, (lower0, upper0))
    return lo
