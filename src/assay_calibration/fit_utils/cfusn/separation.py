"""
Component-separation features for *multivariate* CFUSN mixtures.

Background
----------
The univariate fits enforce non-overlap with a density-ratio monotonicity
constraint (``constraints.py`` + ``binary_search``).  That works on the line
because ℝ is totally ordered: a monotone likelihood ratio is a single
threshold that partitions the score axis.  In ℝ² there is no canonical total
order, so the 1-D intuition does not generalise — the old multivariate
``mode='line'`` and ``mode='marginal'`` checks are projection-based and can
neither certify nor induce joint non-overlap.  They are deprecated; see
:func:`resolve_constraint_mode`.

Replacement (objective-level separation, not a feasibility projection)
----------------------------------------------------------------------
(1) **Responsibility tempering** — sharpen E-step responsibilities
    ``r ← r^(1+β) / Σ r^(1+β)`` (annealed β) before the M-step.  This is the
    descent direction of a co-membership / Gini overlap penalty
    ``Σ_n Σ_{i<j} r_{ni} r_{nj}`` and drives responsibilities toward one-hot,
    i.e. a confident soft partition.  Geometry-free, so it works unchanged for
    CFUSN q≥1.

(2) **Bhattacharyya repulsion** — after the closed-form M-step, nudge each
    component's location apart along pairwise directions, weighted by the
    Bhattacharyya overlap ``exp(-D_B)`` of the Gaussian envelopes
    ``N(m_i, Ω_i)`` (skew-adjusted centroid ``m_i``, ``Ω_i = Γ_i + Δ_iΔ_iᵀ``).
    Affine-invariant geometric spacing in joint space; self-limiting because
    the weight vanishes once components separate.

``constraint_mode`` values for multivariate fits:
    ``"repulsion"`` (default)            → (2) only
    ``"tempering"``                      → (1) only
    ``"separation"`` / ``"tempering+repulsion"`` → (1)+(2)
    ``"line"`` / ``"marginal"``          → DEPRECATED, raises.

Choosing the default — empirical basis
--------------------------------------
A sweep over overlap levels (ARI vs. ground-truth labels ↑, contested-point
fraction ↓) showed a sharp qualitative split:

  * **Repulsion (2)** improves ARI over the unconstrained baseline in *every*
    regime (high/medium/low overlap) and never fabricates structure. When
    components genuinely overlap it honestly leaves contested mass high rather
    than forcing a partition.
  * **Tempering (1)** (and **combined**, which it dominates) is excellent when
    a real separation exists (near-perfect ARI on well-separated data) but
    *catastrophic* when overlap is genuine: it drives contested → 0 while ARI
    collapses to ≈ 0, i.e. a confident-but-wrong partition. This is **not**
    tunable away — even β=0.1 collapses, because any sharpening iterated to
    convergence tends to hard assignment. The danger is that the overlap
    metric looks great while the clustering is meaningless.

So the default is **repulsion** (robust, honest). Use ``"tempering"`` or
``"separation"`` only when the data is known to contain genuinely separable
clusters and maximal overlap reduction is wanted.
"""

import numpy as np


DEFAULT_CONSTRAINT_MODE = "repulsion"

_LEGACY_MODES = {"line", "marginal"}
_TEMPERING_ONLY = {"tempering"}
_REPULSION_ONLY = {"repulsion"}
_FULL_MODES = {"separation", "tempering+repulsion"}

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


class DeprecatedConstraintError(ValueError):
    """Raised when a deprecated multivariate constraint mode is requested."""


# ──────────────────────────────────────────────────────────────────
# Mode resolution / validation
# ──────────────────────────────────────────────────────────────────

def is_legacy_mode(constraint_mode) -> bool:
    return constraint_mode in _LEGACY_MODES


def resolve_constraint_mode(constraint_mode):
    """Validate a multivariate constraint mode and return its feature config.

    Returns ``{"tempering": bool, "repulsion": bool, "mode": str}``.
    Raises :class:`DeprecatedConstraintError` for ``"line"``/``"marginal"`` and
    ``ValueError`` for unknown names.
    """
    if constraint_mode in _LEGACY_MODES:
        raise DeprecatedConstraintError(
            f"constraint_mode={constraint_mode!r} is deprecated and removed for "
            "multivariate CFUSN fits: projection-based density-ratio "
            "constraints ('line'/'marginal') do not induce joint non-overlap in "
            "2D. Use constraint_mode='repulsion' (default; Bhattacharyya "
            "repulsion), 'tempering' (responsibility tempering), or 'separation' "
            "(both)."
        )
    if constraint_mode in _TEMPERING_ONLY:
        return {"tempering": True, "repulsion": False, "mode": "tempering"}
    if constraint_mode in _REPULSION_ONLY:
        return {"tempering": False, "repulsion": True, "mode": "repulsion"}
    if constraint_mode in _FULL_MODES:
        return {"tempering": True, "repulsion": True, "mode": "separation"}
    raise ValueError(
        f"Unknown constraint_mode={constraint_mode!r}. Valid: 'separation', "
        "'tempering', 'repulsion' ('line'/'marginal' are deprecated)."
    )


def validate_constraint_mode(constraint_mode, multivariate):
    """Fail fast on a user-specified mode.

    Deprecated modes raise whenever they are specified; other names are only
    range-checked for multivariate fits (constraint_mode is unused for the
    univariate density-ratio constraint, which is retained).
    """
    if constraint_mode in _LEGACY_MODES:
        resolve_constraint_mode(constraint_mode)  # raises DeprecatedConstraintError
    if multivariate:
        resolve_constraint_mode(constraint_mode)  # range-check name


def separation_min_iters(**kwargs) -> int:
    """Iterations the annealing schedule needs before convergence may be tested."""
    warmup = int(kwargs.get("tempering_warmup", 5))
    ramp = int(kwargs.get("tempering_ramp", 20))
    return warmup + ramp


# ──────────────────────────────────────────────────────────────────
# M-step stabilisers (prevent the degeneracy that sharpened assignment
# exacerbates: components losing all mass → undefined mean / spike collapse).
# ──────────────────────────────────────────────────────────────────

def data_scale(xlims) -> float:
    """Mean per-dimension span of the data box."""
    if xlims is None:
        return 1.0
    spans = [float(hi) - float(lo) for lo, hi in xlims]
    return float(np.mean(spans)) if spans else 1.0


def gamma_ridge(xlims, **kwargs) -> float:
    """Additive variance floor for Γ, as ``(cov_ridge_frac · data_scale)²``.

    Keeps components from collapsing to near-degenerate spikes when tempering
    sharpens responsibilities. ``cov_ridge_frac=0`` disables it.
    """
    frac = float(kwargs.get("cov_ridge_frac", 0.05))
    if frac <= 0.0:
        return 0.0
    return (frac * data_scale(xlims)) ** 2


def regularize_gamma(Gamma, xlims, **kwargs):
    """Add the ridge floor to Γ's diagonal (no-op if ridge is 0)."""
    ridge = gamma_ridge(xlims, **kwargs)
    if ridge <= 0.0:
        return Gamma
    Gamma = np.asarray(Gamma, dtype=float)
    return Gamma + ridge * np.eye(Gamma.shape[0])


def min_component_mass(**kwargs) -> float:
    """Effective responsibility mass below which a component update is frozen."""
    return float(kwargs.get("min_component_mass", 2.0))


def params_sane(mu, Delta, Gamma, xlims, **kwargs):
    """Reject pathological M-step candidates (the source of Δ→∞ blowups).

    A candidate is sane iff all parameters are finite, the skew-adjusted
    centroid lies within the data box expanded by ``param_sanity_margin``
    scales, and the envelope scale does not exceed ``param_sanity_scale_mult``
    times the data scale. Used (under ``_stabilize_separation``) to keep a
    degenerate per-component moment matrix from sending Δ/Γ to extreme values.
    """
    mu = np.asarray(mu, dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(Delta))
            and np.all(np.isfinite(Gamma))):
        return False
    scale = data_scale(xlims)
    margin = float(kwargs.get("param_sanity_margin", 3.0))
    scale_mult = float(kwargs.get("param_sanity_scale_mult", 25.0))
    cent = _skew_adjusted_centroid(mu, Delta)
    if xlims is not None:
        for d, (lo, hi) in enumerate(xlims):
            if cent[d] < lo - margin * scale or cent[d] > hi + margin * scale:
                return False
    env = _envelope_cov(Delta, Gamma)
    if np.trace(env) > (scale_mult * scale) ** 2:
        return False
    return True


# ──────────────────────────────────────────────────────────────────
# (1) Responsibility tempering
# ──────────────────────────────────────────────────────────────────

def tempering_beta(iter_num, **kwargs) -> float:
    """Annealed sharpening exponent β.

    β = 0 for the first ``tempering_warmup`` iters (let plain EM find a
    sensible basin), then ramps linearly to ``tempering_beta_max`` over
    ``tempering_ramp`` iters and holds.
    """
    warmup = int(kwargs.get("tempering_warmup", 5))
    ramp = max(int(kwargs.get("tempering_ramp", 20)), 1)
    # Gentle default: strong enough to separate genuinely overlapping
    # components, mild enough to avoid disrupting good optima / collapsing
    # components (β=2 over-sharpens; β≈0.5 is the empirical sweet spot).
    beta_max = float(kwargs.get("tempering_beta_max", 0.5))
    if iter_num < warmup:
        return 0.0
    frac = min(1.0, (iter_num - warmup) / ramp)
    return frac * beta_max


def apply_responsibility_tempering(responsibilities, beta):
    """Sharpen responsibilities ``r ← r^(1+β) / Σ_k r^(1+β)``.

    ``responsibilities`` is ``(K, N)``; normalisation is over the component
    axis. β ≤ 0 is a no-op (returns the input unchanged).
    """
    if beta <= 0.0:
        return responsibilities
    with np.errstate(divide="ignore"):
        log_r = np.log(np.clip(responsibilities, 1e-300, None))
    sharp = np.exp((1.0 + beta) * log_r)
    denom = sharp.sum(axis=0, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return sharp / denom


# ──────────────────────────────────────────────────────────────────
# (2) Bhattacharyya repulsion
# ──────────────────────────────────────────────────────────────────

def _skew_adjusted_centroid(mu, Delta):
    """E[X] ≈ μ + √(2/π) · Δ·1_q  (CFUSN mean under TN_q(0, I, R^q_+) latent)."""
    mu = np.asarray(mu, dtype=float).ravel()
    Delta = np.atleast_2d(np.asarray(Delta, dtype=float))
    if Delta.shape[0] != mu.shape[0] and Delta.shape[1] == mu.shape[0]:
        Delta = Delta.T
    return mu + _SQRT_2_OVER_PI * Delta.sum(axis=1)


def _envelope_cov(Delta, Gamma):
    """Gaussian-envelope covariance Ω = Γ + ΔΔᵀ (PD by construction)."""
    Delta = np.atleast_2d(np.asarray(Delta, dtype=float))
    Gamma = np.asarray(Gamma, dtype=float)
    if Delta.shape[0] != Gamma.shape[0] and Delta.shape[1] == Gamma.shape[0]:
        Delta = Delta.T
    Om = Gamma + Delta @ Delta.T
    return 0.5 * (Om + Om.T)


def bhattacharyya_distance(mi, Oi, mj, Oj):
    """Bhattacharyya distance between N(mi, Oi) and N(mj, Oj).

    Returns ``np.inf`` if the averaged covariance is not usable (treated as a
    fully separated pair → zero repulsion weight).
    """
    O = 0.5 * (Oi + Oj)
    dm = mi - mj
    try:
        Oinv_dm = np.linalg.solve(O, dm)
        sign_O, logdet_O = np.linalg.slogdet(O)
        sign_i, logdet_i = np.linalg.slogdet(Oi)
        sign_j, logdet_j = np.linalg.slogdet(Oj)
    except np.linalg.LinAlgError:
        return np.inf
    if sign_O <= 0 or sign_i <= 0 or sign_j <= 0:
        return np.inf
    term1 = 0.125 * float(dm @ Oinv_dm)
    term2 = 0.5 * (logdet_O - 0.5 * (logdet_i + logdet_j))
    return term1 + max(term2, 0.0)


def repulsion_strength(iter_num, **kwargs) -> float:
    """Decaying repulsion gain: a symmetry-breaking force that fades to zero.

    Repulsion is an *early* force — it nudges overlapping components apart while
    the data-driven M-step still settles their final, data-supported locations.
    It is held at ``repulsion_lr`` from ``repulsion_warmup`` and decays linearly
    to 0 by ``repulsion_decay_iters`` (default = warmup + tempering_ramp). This
    avoids the runaway-to-boundary behaviour of a constant post-hoc mean push,
    which has no opposing data force within the step.
    """
    warmup = int(kwargs.get("repulsion_warmup", 5))
    if iter_num < warmup:
        return 0.0
    lr = float(kwargs.get("repulsion_lr", 0.1))
    decay_iters = int(kwargs.get(
        "repulsion_decay_iters",
        warmup + int(kwargs.get("tempering_ramp", 20)),
    ))
    span = max(decay_iters - warmup, 1)
    frac = max(0.0, 1.0 - (iter_num - warmup) / span)
    return lr * frac


def bhattacharyya_repulsion_step(updated_params, xlims, multivariate=True, **kwargs):
    """Push overlapping components apart by nudging their locations (early, decaying).

    For each component i, shift μ_i (centroid moves with it; Δ, Γ unchanged) by
        Δμ_i = clip( g · Σ_{j≠i} exp(-D_B(i,j)) · s_ij · û_ij ,  ‖·‖ ≤ cap·s_i )
    where û_ij is the unit centroid-separation direction, s_ij the mean
    component scale, exp(-D_B) the Bhattacharyya overlap weight (large when
    components overlap, ≈0 once separated), and ``g`` the decaying gain from
    :func:`repulsion_strength`. The per-step displacement is capped at
    ``repulsion_max_step_frac · s_i`` so a component cannot be flung off its
    data; the M-step re-anchors μ to the data each iteration.

    No-op for univariate fits (repulsion is a joint-space operation), before
    ``repulsion_warmup``, and once the gain has decayed to zero.
    """
    if not multivariate:
        return updated_params
    K = len(updated_params)
    if K < 2:
        return updated_params

    gain = repulsion_strength(int(kwargs.get("iterNum", 0)), **kwargs)
    if gain <= 0.0:
        return updated_params

    cap_frac = float(kwargs.get("repulsion_max_step_frac", 0.15))

    mus = [np.asarray(p[0], dtype=float).ravel() for p in updated_params]
    cents = [_skew_adjusted_centroid(p[0], p[1]) for p in updated_params]
    covs = [_envelope_cov(p[1], p[2]) for p in updated_params]
    p_dim = mus[0].shape[0]
    scales = [float(np.sqrt(max(np.trace(O) / p_dim, 1e-12))) for O in covs]

    rng = kwargs.get("rng") or np.random.RandomState()
    new_params = []
    for i, (mu, Delta, Gamma) in enumerate(updated_params):
        shift = np.zeros(p_dim)
        for j in range(K):
            if i == j:
                continue
            diff = cents[i] - cents[j]
            norm = float(np.linalg.norm(diff))
            if norm < 1e-12:
                u = rng.normal(size=p_dim)
                u /= max(np.linalg.norm(u), 1e-12)
            else:
                u = diff / norm
            d_b = bhattacharyya_distance(cents[i], covs[i], cents[j], covs[j])
            weight = float(np.exp(-d_b)) if np.isfinite(d_b) else 0.0
            shift += gain * weight * 0.5 * (scales[i] + scales[j]) * u

        # Cap displacement to a fraction of this component's own scale.
        cap = cap_frac * scales[i]
        snorm = float(np.linalg.norm(shift))
        if snorm > cap and snorm > 0:
            shift *= cap / snorm

        mu_new = mus[i] + shift
        if xlims is not None:
            for d in range(p_dim):
                lo, hi = xlims[d]
                mu_new[d] = min(max(mu_new[d], lo), hi)
        new_params.append((mu_new, Delta, Gamma))
    return new_params
