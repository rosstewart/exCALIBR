"""Batched univariate density-ratio monotonicity constraint, mirroring
``cfusn/constraints.py``'s univariate path (``multicomponent_density_constraint_violated``
with ``multivariate=False``) and ``cfusn/update_steps.py``'s ``binary_search``.
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

    Matches the CPU reference exactly: detects any increase in log_ratio among
    valid (above mass-floor) grid points, including across sub-floor gaps.

    Uses lax.associative_scan (parallel prefix max) to compute the maximum
    valid log_ratio to the right of each position in O(log n) XLA depth —
    no sequential scan primitive, so compilation inside nested loops is fast.

    Correctness: violation exists iff any valid position i has a valid j > i
    with log_ratio[j] > log_ratio[i].  This is equivalent to the CPU's
    np.diff-on-compacted check because a non-increasing compacted sequence
    has no such pair, and vice versa.
    """
    batch = log_pdfs.shape[1]
    violated = jnp.zeros(batch, dtype=bool)
    for i in range(log_pdfs.shape[0] - 1):
        valid = (log_pdfs[i] > mass_floor) & (log_pdfs[i + 1] > mass_floor)  # (batch, n)
        log_ratio = log_pdfs[i] - log_pdfs[i + 1]                            # (batch, n)
        n_valid = valid.sum(axis=-1)                                          # (batch,)

        # Max valid log_ratio to the RIGHT of each position (exclusive).
        # Mask invalid positions with -inf, then parallel suffix-max via
        # reverse + associative_scan(max) + reverse.
        masked = jnp.where(valid, log_ratio, -jnp.inf)                       # (batch, n)
        rev_cum_max = lax.associative_scan(
            jnp.maximum, masked[:, ::-1], axis=-1
        )                                                                     # (batch, n)
        # right_max[i] = max valid ratio strictly right of i  (shift by 1)
        right_max = rev_cum_max[:, ::-1][:, 1:]                              # (batch, n-1)

        pair_violated = jnp.any(
            valid[:, :-1] & (right_max > log_ratio[:, :-1]), axis=-1
        )
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
                   x_grid, max_iters=20):
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
    runs a fixed `max_iters` halvings. 20 halvings gives precision
    initial_range/2^20 ≈ 1e-6 for a unit range — tighter than NumPy's 1e-4
    stopping criterion. Required for lax.fori_loop/JIT tracing.

    Performance: only the bisected component's log_pdf is recomputed on each
    halving. The other K-1 components' log_pdfs are constant across all
    halvings; they are computed once before the fori_loop and captured as
    compile-time constants in the XLA graph — giving a K× speedup over
    recomputing all K components every iteration.
    """
    K = len(current_alt_params)  # static Python int
    lower0 = current_alt_params[component_index][param_index]
    upper0 = candidate_value

    # Pre-compute log_pdfs for all components EXCEPT component_index.
    # These are loop-invariant (don't depend on mid) and are hoisted here so
    # XLA embeds them as constants in the fori_loop body rather than
    # recomputing on every halving.
    fixed_log_pdfs = {}  # c (int) -> (batch, n) float64
    for c in range(K):
        if c != component_index:
            a_c, loc_c, scale_c = alternate_to_canonical(*current_alt_params[c])
            fixed_log_pdfs[c] = skewnorm_logpdf(
                x_grid, a_c[:, None], loc_c[:, None], scale_c[:, None]
            )  # (batch, n)

    # Alternate params for the bisected component (minus the param being varied)
    c_alt = list(current_alt_params[component_index])

    def body(_, carry):
        lo, hi = carry
        converged = jnp.abs(hi - lo) <= 1e-4   # match NumPy while-loop stop condition
        mid = 0.5 * (lo + hi)

        # Recompute log_pdf only for the bisected component
        params_mid = c_alt.copy()
        params_mid[param_index] = mid
        a_bi, loc_bi, scale_bi = alternate_to_canonical(*params_mid)
        lp_bi = skewnorm_logpdf(
            x_grid, a_bi[:, None], loc_bi[:, None], scale_bi[:, None]
        )  # (batch, n)

        # Assemble full (K, batch, n) from cached fixed + new bisected slice
        all_lps = [fixed_log_pdfs[c] if c != component_index else lp_bi for c in range(K)]
        log_pdfs = jnp.stack(all_lps, axis=0)  # (K, batch, n)

        violated = _adjacent_pair_violated(log_pdfs)
        new_hi = jnp.where(converged, hi, jnp.where(violated, mid, hi))
        new_lo = jnp.where(converged, lo, jnp.where(violated, lo, mid))
        return (new_lo, new_hi)

    lo, _ = lax.fori_loop(0, max_iters, body, (lower0, upper0))
    return lo
