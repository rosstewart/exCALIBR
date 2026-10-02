from .update_steps import em_iteration, get_sample_weights, gamma_log_prior_total
from .density_utils import get_likelihood, get_q, _ensure_matrix_delta
from .initializations import (
    kmeans_init, methodOfMomentsInit, kmeans_init_mv, kmeans_init_mv_anchored,
    kmeans_init_anchored,
)
from . import constraints
from . import separation

import numpy as np
import logging
from tqdm.auto import tqdm
import warnings


def compute_sample_weights(sample_indicators, **kwargs):
    """Per-observation M-step weights from sample_proportions, sample_weight_transform,
    or sample_balance_beta.

    sample_proportions : array-like of length S
        Explicit relative weight for each sample (e.g. [2,1,1,1] upweights
        sample 0 twice as much as the others). Takes precedence over
        sample_weight_transform and sample_balance_beta when supplied.
        Values are normalised internally so only their ratios matter.

    sample_weight_transform : {None, "log2", "ln"} (default None)
        Each sample's TOTAL M-step contribution is made proportional to
        log2(N_s) or ln(N_s) instead of the standard N_s (i.e. per-observation
        weight = log(N_s) / N_s) -- a gentler correction than full balancing
        (sample_balance_beta=1 makes every sample's total contribution
        EQUAL regardless of N_s; this only compresses the disparity
        logarithmically). Takes precedence over sample_balance_beta when
        both are supplied.

        NOTE: "log2" and "ln" produce IDENTICAL fitted parameters, not just
        similar ones -- log2(N) = ln(N)/ln(2), a single global constant
        factor applied identically across every sample class, and every
        M-step update consuming this weight is a NORMALIZED weighted average
        ((x*r).sum()/r.sum()) or an argmax that factors the per-class-constant
        weight out entirely (see update_steps.py's get_sample_weights_and_ll,
        whose own comment confirms "sw_n is constant within sample s, so it
        factors out of the W argmax"). A uniform rescale of all weights by
        the same constant cannot change any normalized-average or argmax
        result. Both options are kept for API clarity/future generality
        (e.g. if a non-uniform transform were ever added), but running both
        as separate experiments is redundant -- pick either one.

    sample_balance_beta : float (default 0)
        Continuous balance parameter (power law).  beta=0 → standard EM (no
        reweighting); beta=1 → each sample contributes equally regardless of
        size. NOT the same functional form as sample_weight_transform (power
        law vs. logarithmic) -- no value of beta reproduces a log-weighted fit.

    Precedence: sample_proportions > sample_weight_transform > sample_balance_beta > none.
    Returns None when nothing requests any reweighting (byte-for-byte the
    original no-op behavior).
    """
    proportions = kwargs.get("sample_proportions", None)
    transform = kwargs.get("sample_weight_transform", None)
    beta = float(kwargs.get("sample_balance_beta", 0.0))
    N_samples = sample_indicators.shape[1]
    N_per_sample = sample_indicators.sum(axis=0).astype(float)

    if proportions is not None:
        proportions = np.asarray(proportions, dtype=float)
        if len(proportions) != N_samples:
            raise ValueError(
                f"sample_proportions length {len(proportions)} != N_samples {N_samples}"
            )
        proportions = proportions / proportions.sum()
        # per-obs weight p_s / N_s so the total M-step contribution of sample s ∝ p_s
        per_sample_w = np.where(
            N_per_sample > 0, proportions / np.maximum(N_per_sample, 1.0), 0.0
        )
    elif transform in ("log2", "ln") and N_samples > 1:
        log_fn = np.log2 if transform == "log2" else np.log
        # per-obs weight log(N_s) / N_s so the total M-step contribution of
        # sample s ∝ log(N_s), mirroring the sample_proportions branch's
        # "target total contribution / N_s" shape above.
        target = np.where(N_per_sample > 0, log_fn(np.maximum(N_per_sample, 1.0)), 0.0)
        per_sample_w = np.where(N_per_sample > 0, target / np.maximum(N_per_sample, 1.0), 0.0)
    elif beta > 0 and N_samples > 1:
        N_ref = N_per_sample[N_per_sample > 0].min() if (N_per_sample > 0).any() else 1.0
        per_sample_w = np.where(
            N_per_sample > 0, (N_ref / np.maximum(N_per_sample, 1.0)) ** beta, 0.0
        )
    else:
        return None

    return (sample_indicators.astype(float) * per_sample_w).sum(axis=1)


def compute_dim_weights(observations, **kwargs):
    """Per-dimension M-step weights from dim_proportions, dim_weight_transform,
    or dim_balance_beta -- the per-DIMENSION analog of compute_sample_weights
    above, motivated by sparse dimensions (e.g. TP53's KawOligo at ~1.5% row
    coverage) getting too little pull on the shared EM fit relative to dense
    dimensions. EXPERIMENTAL / opt-in only: returns None (byte-identical
    no-op) unless a caller explicitly requests reweighting, and only Delta's
    M-step actually consumes the result -- for q>1 restarts,
    update_steps.py's _delta_update_completed / get_Delta_update_cfusn; for
    q=1 restricted-MSN restarts, get_Delta_update_mv (same treatment, scalar
    instead of (q,q) shared normalizer). mu and Gamma are not currently weighted
    by this (mu is algebraically inert to this kind of per-dimension
    reweighting, and Gamma's M-step is a ratio of weighted sums over the
    same per-cell weight, so a naive per-dimension-pair outer-product
    weighting would cancel exactly; see the design discussion this was
    implemented from).

    IMPORTANT asymmetry vs. compute_sample_weights: sample reweighting there
    is anchored at the SMALLEST sample (weights <= 1, damping large samples'
    pull toward the rarest one), because every sample's weighted rows feed
    the SAME shared per-component parameters and directly compete for them.
    Here, each dimension's Delta row is fit independently of every other
    dimension's (see get_Delta_update_cfusn/_delta_update_completed), so
    down-weighting dense dimensions would do nothing at all for a sparse
    one -- the only way this mechanism can affect a sparse dimension's own
    fit is to weight IT specifically. So dim_weights here is anchored at the
    DENSEST dimension (weight == 1 there, i.e. unweighted/no-op for the best-
    covered dimension) and weights are >= 1 for sparser ones, boosting how
    much a dimension's genuinely-observed rows count relative to its
    EM-imputed (missing-data-completed) rows in its own Delta estimate.

    dim_proportions : array-like of length p
        Explicit per-dimension weight (directly, not derived from
        coverage) -- 1.0 means unweighted. Takes precedence over
        dim_weight_transform and dim_balance_beta when supplied.

    dim_weight_transform : {None, "log2", "ln"} (default None)
        A dimension's real-observed-row weight is anchored so its
        log2/ln-scaled effective mass (log_fn(coverage_d)) matches the
        densest dimension's effective mass at coverage_d==cov_max -- the
        gentle option, mirroring sample_weight_transform's log-compression
        rationale but anchored at the max per the asymmetry above. Takes
        precedence over dim_balance_beta when both are supplied.

    dim_balance_beta : float (default 0)
        Continuous balance parameter (power law): weight_d =
        (cov_max / coverage_d) ** beta. beta=0 -> no reweighting; beta=1 ->
        every dimension's real-observed rows are boosted so their total
        effective mass equals the densest dimension's raw coverage (full
        equalisation). NOT the same functional form as dim_weight_transform.

    Precedence: dim_proportions > dim_weight_transform > dim_balance_beta > none.
    Returns None when nothing requests any reweighting.
    """
    proportions = kwargs.get("dim_proportions", None)
    transform = kwargs.get("dim_weight_transform", None)
    beta = float(kwargs.get("dim_balance_beta", 0.0))
    observations = np.asarray(observations, dtype=float)
    p = observations.shape[1]
    coverage = (~np.isnan(observations)).sum(axis=0).astype(float)  # (p,)
    valid = coverage > 0

    if proportions is not None:
        proportions = np.asarray(proportions, dtype=float)
        if len(proportions) != p:
            raise ValueError(
                f"dim_proportions length {len(proportions)} != N_dims {p}"
            )
        dim_w = proportions
    elif transform in ("log2", "ln") and p > 1 and valid.any():
        log_fn = np.log2 if transform == "log2" else np.log
        cov_max = coverage[valid].max()
        log_cov_max = max(float(log_fn(max(cov_max, 1.0))), 1e-12)
        dim_w = np.where(
            valid,
            (cov_max / log_cov_max) * log_fn(np.maximum(coverage, 1.0)) / np.maximum(coverage, 1.0),
            0.0,
        )
    elif beta > 0 and p > 1 and valid.any():
        cov_max = coverage[valid].max()
        dim_w = np.where(valid, (cov_max / np.maximum(coverage, 1.0)) ** beta, 0.0)
    else:
        return None

    return dim_w


def single_fit(
    observations, sample_indicators, N_components, constrained,
    init_method, init_constraint_adjustment, multivariate=False, **kwargs
):
    # Iteration cap. Multivariate fits get a much larger budget than
    # univariate: with the exact q=2 E-step's tail accuracy fixed, MV fits that
    # used to thrash in a numerically-degenerate regime now genuinely converge,
    # and some need far more than 10000 iterations to get there (measured on
    # kras_labelseq_mv 4c: 1166 iterations before the fix, still climbing
    # monotonically at 10001 after it, i.e. the old cap was binding). Hitting
    # the cap is no longer destructive -- the fit returns the best iterate it
    # reached -- so a larger budget buys convergence rather than risking a
    # worse answer. Univariate fits converge in far fewer iterations and are
    # left on the historical cap so their behaviour is unchanged.
    _default_max_iters = 50000 if multivariate else 10000
    MAX_EM_ITERS = kwargs.get("max_em_iters", _default_max_iters)
    verbose = kwargs.get("verbose", True)
    check_submerged_duration = kwargs.get("check_submerged_duration", False)
    MIN_SCALE = 1e-100
    # Relative-to-|LL| thresholds splitting a likelihood decrease into
    # plateau noise / converged-with-a-tiny-backstep / real M-step overshoot.
    # See the decrease check in the EM loop below for the measured values
    # these are set from.
    PLATEAU_REL_TOL = 1e-8
    OVERSHOOT_REL_TOL = 1e-5
    # Above this relative decrease the iterate itself is presumed corrupt (not
    # merely overshot) and the fit is failed outright rather than rolled back.
    CATASTROPHIC_REL_TOL = 1e-1
    mv = multivariate
    latent_q = kwargs.get("latent_q", 2)
    constraint_mode = kwargs.get("constraint_mode", separation.DEFAULT_CONSTRAINT_MODE)

    # Single RNG for this fit (init draws + every EM iteration's MC/repulsion
    # steps), seeded from fit_seed for reproducibility. Stored back into kwargs
    # so it's picked up by every init routine via kwargs.get("rng") without
    # having to edit each call site. None fit_seed (unseeded pipeline runs)
    # falls back to RandomState(None) — the historical unseeded behaviour.
    rng = kwargs.get("rng") or np.random.RandomState(kwargs.get("fit_seed"))
    kwargs["rng"] = rng

    # Fail fast on deprecated 'line'/'marginal' modes (multivariate only); the
    # univariate density-ratio constraint is retained and ignores the mode.
    separation.validate_constraint_mode(constraint_mode, mv)
    # Multivariate constrained fits induce non-overlap via the separation
    # features (Bhattacharyya repulsion by default; optionally responsibility
    # tempering) rather than a feasibility projection; this deliberately trades
    # likelihood for separation, so the EM-monotonicity machinery (backtracking,
    # final density check) is relaxed below.
    separation_active = bool(constrained and mv)
    sep_min_iters = separation.separation_min_iters(**kwargs)

    if kwargs.get("submerge_steps") is not None:
        raise NotImplementedError("submerge_steps is deprecated")

    if mv:
        xlims = tuple(
            (float(np.nanmin(observations[:, d])), float(np.nanmax(observations[:, d])))
            for d in range(observations.shape[1])
        )
    else:
        xlims = (observations.min(), observations.max())

    N_samples = sample_indicators.shape[1]

    # ---- Per-observation sample-balance weights (M-step reweighting) ----
    # compute_sample_weights handles both sample_proportions (explicit per-sample
    # target proportions) and sample_balance_beta (continuous balance exponent).
    # Returns None → standard EM with no reweighting.
    sample_weights_per_obs = compute_sample_weights(sample_indicators, **kwargs)

    # ---- Per-dimension Delta-reweighting (EXPERIMENTAL, opt-in only) ----
    # See compute_dim_weights' docstring. mv-only (a univariate fit has one
    # dimension, nothing to reweight); returns None unless the caller passed
    # dim_proportions/dim_weight_transform/dim_balance_beta explicitly.
    dim_weights_per_dim = compute_dim_weights(observations, **kwargs) if mv else None

    # ---- Monitoring-objective weights for dim_weights (EXPERIMENTAL) ----
    # Never active by default (None unless dim_weights_per_dim is set, which
    # itself defaults to None). A dim-weighted M-step has no obligation to
    # increase the PLAIN likelihood every iteration (dimension-level
    # reweighting isn't a simple per-row reweighting of the joint density
    # the way sample_weights is, so there's no exact closed-form "weighted
    # Q" to track the way sample_balance_beta's get_likelihood(...,
    # sample_weights=...) call already does) -- without this, the EM loop's
    # overshoot/plateau/best-iterate bookkeeping below watches a quantity
    # the M-step was never trying to increase, and spuriously reverts
    # early, promising, in-progress fits (confirmed on real TP53 data: a
    # dim-weighted q2 CFUSN restart reverted after 8 iterations on a
    # false-alarm "overshoot" that a matched baseline fit ran >1000
    # iterations past without issue). _dim_monitor_row_weights gives each
    # row a weight equal to the MEAN dim_weights value over that row's own
    # observed dimensions (1.0 for every row when dim_weights_per_dim is
    # None/all-ones, so production behavior is bit-identical) -- a
    # well-defined, directionally-consistent proxy objective (rows with
    # more of the upweighted/sparse dimensions observed count for more),
    # not a formal re-derivation of Q under dimension reweighting (which
    # Gamma's non-diagonal coupling makes intractable in closed form; see
    # _delta_update_completed's docstring on diagonal-Gamma being the only
    # case available-case masking is exact). Combined multiplicatively with
    # any sample_weights_per_obs so both reweighting mechanisms, if both are
    # ever used together, are reflected in the single monitored series.
    monitor_weights = sample_weights_per_obs
    if dim_weights_per_dim is not None:
        obs_mask = ~np.isnan(observations)
        n_obs_per_row = obs_mask.sum(axis=1)
        row_dim_w = np.where(
            n_obs_per_row > 0,
            (obs_mask * dim_weights_per_dim[None, :]).sum(axis=1) / np.maximum(n_obs_per_row, 1),
            1.0,
        )
        monitor_weights = row_dim_w if monitor_weights is None else monitor_weights * row_dim_w

    # ---- Initialization ----
    if (
        kwargs.get("initial_weights") is not None
        and kwargs.get("initial_params") is not None
    ):
        kmeans = None
        initial_params = kwargs["initial_params"]
        W = np.array(kwargs["initial_weights"])
        if W.shape != (N_samples, N_components):
            raise ValueError(f"Initial weights shape {W.shape} mismatch")
        if len(initial_params) != N_components:
            raise ValueError(f"Initial params length {len(initial_params)} mismatch")
        if mv and latent_q > 1:
            initial_params = _ensure_cfusn_params(initial_params, latent_q)
    else:
        W = np.ones((N_samples, N_components)) / N_components

        initial_params = None
        if init_method == "method_of_moments" and not mv:
            kmeans = "method_of_moments"
            initial_params = methodOfMomentsInit(
                observations, N_components, constrained,
                init_constraint_adjustment=init_constraint_adjustment, **kwargs
            )

        if initial_params is None:
            if init_method == "method_of_moments" and verbose and not mv:
                print("failed method of moments, falling back to kmeans")
            if verbose:
                q_str = f", latent_q={latent_q}" if mv and latent_q > 1 else ""
                print(f"[INIT] mv={mv}, method={'kmeans_mv' if mv else 'kmeans'}, "
                      f"obs shape={observations.shape}, "
                      f"NaN count={np.isnan(observations).sum()}, "
                      f"xlims={xlims}{q_str}")
            try:
                if mv and init_method == "anchored":
                    initial_params, kmeans = kmeans_init_mv_anchored(
                        observations, sample_indicators,
                        n_clusters=N_components,
                        constrained=constrained,
                        init_constraint_adjustment=init_constraint_adjustment,
                        **kwargs
                    )
                elif mv:
                    initial_params, kmeans = kmeans_init_mv(
                        observations, n_clusters=N_components,
                        constrained=constrained,
                        init_constraint_adjustment=init_constraint_adjustment,
                        **kwargs
                    )
                elif init_method == "anchored":
                    initial_params, kmeans = kmeans_init_anchored(
                        observations, sample_indicators,
                        n_clusters=N_components,
                        constrained=constrained,
                        init_constraint_adjustment=init_constraint_adjustment,
                        **kwargs
                    )
                else:
                    initial_params, kmeans = kmeans_init(
                        observations, n_clusters=N_components,
                        constrained=constrained,
                        init_constraint_adjustment=init_constraint_adjustment,
                        **kwargs
                    )
            except ValueError as e:
                if kwargs.get("raise_on_error", True):
                    raise
                if kwargs.get("verbose_init", True):
                    print(f"[INIT FAILED] {e}")
                return dict(
                    component_params=[[] for _ in range(N_components)],
                    weights=W,
                    likelihoods=[-np.inf],
                    xlims=xlims,
                    times_submerged=[],
                )

        if mv and latent_q > 1:
            initial_params = _ensure_cfusn_params(initial_params, latent_q)

        if mv and kwargs.get("force_gaussian"):
            # Force the very first E-step to already see Delta=0, not
            # whatever nonzero value kmeans/method-of-moments init produced
            # -- the M-step will zero it out from iteration 1 onward
            # regardless (see _em_update_cfusn/_em_update_multivariate's
            # force_gaussian handling), but starting there directly avoids
            # a spurious first-iteration likelihood swing between a
            # nonzero-Delta E-step and the immediately-following zeroed one.
            initial_params = [
                (mu_, np.zeros_like(Delta_), Gamma_)
                for mu_, Delta_, Gamma_ in initial_params
            ]

        W = get_sample_weights(
            observations, sample_indicators, initial_params, W, multivariate=mv
        )

    em_kwargs = {}

    # Collector for M-step guard events (currently: Gamma candidates rejected
    # as indefinite). Reported in the result so a fit whose components were
    # frozen by repeated rejections is visible to best-fit selection rather
    # than silently competing on val_ll alone.
    guard_stats = {}
    em_kwargs["_guard_stats"] = guard_stats
    em_kwargs["constraint_mode"] = constraint_mode
    em_kwargs["rng"] = rng
    em_kwargs["force_gaussian"] = kwargs.get("force_gaussian", False)
    if mv and latent_q > 1:
        em_kwargs["n_mc_truncated"] = kwargs.get("n_mc_truncated", 500)
    if sample_weights_per_obs is not None:
        em_kwargs["sample_weights"] = sample_weights_per_obs
    if dim_weights_per_dim is not None:
        em_kwargs["dim_weights"] = dim_weights_per_dim
    if kwargs.get("frozen_components"):
        # Explicit, caller-chosen set of component indices that never update
        # in the M-step for the whole fit -- e.g. growing a fit by adding one
        # new component while holding every pre-existing component fixed
        # (see add_component_to_fit below). Threaded through the same way
        # sample_weights is: single_fit doesn't forward **kwargs wholesale
        # to em_kwargs, so anything the M-step needs must be listed here
        # explicitly.
        em_kwargs["frozen_components"] = set(kwargs["frozen_components"])

    # Set when the EM loop stopped on a real (non-plateau) overshoot and rolled
    # back to the previous iterate; surfaced in the result dict so a rolled-back
    # fit is visibly different from one that converged normally.
    overshoot_stopped = None

    history = [dict(component_params=initial_params, weights=W)]
    # Initial likelihood: no em_iteration has run yet so we must evaluate explicitly.
    # Pass sample_weights so the initial LL is on the same (weighted) objective
    # the M-step will optimise — preserves monotonicity under β>0.
    likelihoods = np.array([
        get_likelihood(
            observations, sample_indicators, initial_params, W,
            multivariate=mv, sample_weights=monitor_weights,
        ) / len(sample_indicators)
    ])

    # Penalised objective: when the Gamma ridge is active the M-step maximises
    # (log L + log p(Gamma)), not log L alone, so BOTH the decrease check and the
    # convergence test below must run against this series or they will see
    # spurious decreases in a fit that is improving correctly. With the ridge
    # disabled gamma_log_prior_total returns 0.0, so `objectives` is elementwise
    # equal to `likelihoods` and behaviour is bit-identical.
    def _penalty(params):
        return gamma_log_prior_total(params, observations, multivariate=mv) / len(sample_indicators)

    objectives = np.array([likelihoods[0] + _penalty(initial_params)])

    # ---- First EM iteration ----
    # em_iteration now also returns the per-sample log_pdf cache computed on
    # the *updated* params; we feed it back as cached_log_pdfs to the next
    # iteration's E-step (whose current_params == this iteration's
    # updated_params), eliminating one full density pass per iteration.
    try:
        updated_component_params, updated_weights, ll, cached_log_pdfs = em_iteration(
            observations, sample_indicators, initial_params, W,
            constrained, xlims, multivariate=mv, iterNum=0,
            return_log_pdfs=True, **em_kwargs,
        )
    except ZeroDivisionError as e:
        if kwargs.get("raise_on_error", True):
            raise
        print(f"[FIRST EM ITER FAILED] ZeroDivisionError: {e}")
        return dict(
            component_params=initial_params, weights=W,
            likelihoods=[*likelihoods, -np.inf],
            kmeans=kmeans, xlims=xlims, times_submerged=[]
        )

    if dim_weights_per_dim is not None:
        # em_iteration's own `ll` was computed against em_kwargs["sample_weights"]
        # (the TRUE M-step weight) -- recompute against monitor_weights instead,
        # so the EM loop's bookkeeping below tracks the dim-weighted proxy
        # objective, not a quantity the M-step has no obligation to increase.
        ll = get_likelihood(
            observations, sample_indicators, updated_component_params, updated_weights,
            multivariate=mv, sample_weights=monitor_weights,
        ) / len(sample_indicators)
    likelihoods = np.append(likelihoods, ll)
    objectives = np.append(objectives, ll + _penalty(updated_component_params))

    # Best-iterate tracking. EM is monotone in `objectives`, so for a healthy
    # fit the best iterate IS the last one and this is a no-op. It matters on
    # the exit paths where that does not hold: a sub-PLATEAU_REL_TOL decrease
    # falls through the decrease check without reverting, and the fit then
    # stops on the early-stopping test right below it (same 1e-8 threshold, on
    # |delta|), so without this we would return an iterate known to be worse
    # than one we already had. Cheap insurance that the returned parameters are
    # the best ones the fit ever reached, whichever exit it takes.
    best_obj = objectives[-1]
    best_state = (updated_component_params, updated_weights, len(objectives))

    if verbose:
        q_label = f" (CFUSN q={latent_q})" if mv and latent_q > 1 else ""
        pbar = tqdm(total=MAX_EM_ITERS, leave=False, desc=f"EM Iteration{q_label}")

    try:
        underwater_time = 0
        times_submerged = []
        if not constrained and check_submerged_duration:
            is_underwater = constraints.multicomponent_density_constraint_violated(
                updated_component_params, xlims, multivariate=mv, mode="line",
            )
            if is_underwater:
                underwater_time += 1

        for it in range(MAX_EM_ITERS):
            history.append(dict(
                component_params=updated_component_params,
                weights=updated_weights
            ))
            if np.isnan(likelihoods).any():
                raise ValueError("NaN in likelihoods")
            if np.isnan(updated_weights).any():
                raise ValueError(f"NaN in weights at iteration {it}")

            # em_iteration returns (params, weights, ll, log_pdfs_cache) — no
            # separate get_likelihood needed. The cache is the log_pdfs on the
            # *just-updated* params, which is exactly what next iter's E-step
            # needs (its current_params == this iter's updated_params).
            updated_component_params, updated_weights, ll, cached_log_pdfs = em_iteration(
                observations, sample_indicators,
                updated_component_params, updated_weights,
                constrained, xlims, multivariate=mv, iterNum=it + 1,
                cached_log_pdfs=cached_log_pdfs,
                return_log_pdfs=True, **em_kwargs,
            )

            if dim_weights_per_dim is not None:
                # See the first-iteration block above for why this
                # recomputation is needed under dim_weights.
                ll = get_likelihood(
                    observations, sample_indicators, updated_component_params, updated_weights,
                    multivariate=mv, sample_weights=monitor_weights,
                ) / len(sample_indicators)

            if not mv:
                for i, (a, loc, scale) in enumerate(updated_component_params):
                    if scale < MIN_SCALE:
                        updated_component_params[i] = (a, loc, max(scale, MIN_SCALE))
                # Univariate: scale clamp invalidates log-pdf cache for that comp
                cached_log_pdfs = None

            if not constrained and check_submerged_duration:
                violated = constraints.multicomponent_density_constraint_violated(
                    updated_component_params, xlims, multivariate=mv
                )
                if is_underwater and violated:
                    underwater_time += 1
                elif is_underwater and not violated:
                    is_underwater = False
                    times_submerged.append(underwater_time)
                    underwater_time = 0
                elif not is_underwater and violated:
                    is_underwater = True
                    underwater_time += 1

            likelihoods = np.append(likelihoods, ll)
            objectives = np.append(objectives, ll + _penalty(updated_component_params))

            if objectives[-1] > best_obj:
                best_obj = objectives[-1]
                best_state = (updated_component_params, updated_weights,
                              len(objectives))

            # Separation (tempering + repulsion) deliberately trades likelihood
            # for non-overlap, so the penalised objective is not EM-monotone in
            # the raw LL. Skip the decrease check for separation fits and accept
            # the decrease; convergence is governed by the LL plateau after the
            # annealing schedule completes (see sep_min_iters guard below).
            #
            # dim_weights gets the same treatment, for a different reason:
            # under dim_weights, Delta's M-step maximises a per-dimension-
            # reweighted pseudo-Q while mu/Gamma still maximise the TRUE Q --
            # confirmed directly (not just a monitoring-mismatch artifact:
            # even the matched monitor_weights-weighted objective genuinely
            # rises then falls iteration-to-iteration on real data) -- so the
            # three M-step pieces no longer jointly ascend one consistent
            # objective and EM's monotonicity guarantee does not hold. Without
            # this, a dim-weighted fit reverted-and-stopped after as few as
            # 2-8 iterations on its very first non-monotone step (observed on
            # both synthetic and real TP53 data), discarding the entire
            # restart budget. best_state/best_obj tracking above (already
            # keyed off the matched weighted `objectives` series) is the
            # safety net here, exactly as it already is for separation fits.
            dim_weighted_active = dim_weights_per_dim is not None
            if it > 0 and objectives[-1] < objectives[-2] and not separation_active \
                    and not dim_weighted_active:
                decrease = objectives[-2] - objectives[-1]
                # A decrease here is one of two very different things, and the
                # magnitude relative to the LL separates them cleanly:
                #
                #   < PLATEAU_REL_TOL   floating-point noise on a converged
                #                       plateau -- ignore and keep iterating
                #                       (the original 1e-13 absolute threshold
                #                       fired on every such iteration).
                #   < OVERSHOOT_REL_TOL a real but tiny backward step, seen on
                #                       weakly-identified multivariate fits
                #                       that have already converged (measured:
                #                       5e-7 @ iter 113, 6e-7 @ iter 275, on
                #                       LL ~ -7). Failing the whole fit here
                #                       discarded perfectly good converged
                #                       parameters, which is the common case on
                #                       low-separation real data. Treat it as
                #                       convergence: revert to the last good
                #                       iterate and stop.
                #   >= OVERSHOOT_REL_TOL genuine M-step overshoot (measured:
                #                       3.4e-3 @ iter 4, 2.0e-4 @ iter 20).
                #                       Also revert-and-stop, but flagged --
                #                       see the branch below for why.
                #   >= CATASTROPHIC_REL_TOL parameter divergence; the iterate is
                #                       presumed corrupt, so fail the fit and
                #                       let the caller retry with another seed.
                rel_decrease = decrease / max(abs(objectives[-2]), 1e-300)
                if rel_decrease >= OVERSHOOT_REL_TOL:
                    # A decrease this large is a real M-step overshoot, but that
                    # does NOT make the fit worthless: history[-1] was reached by
                    # a monotone sequence and holds the best objective seen, so
                    # raising here threw away every iteration of it. Measured on
                    # a real production run: of 40 discarded restarts, 16 (40%)
                    # died after 1000+ iterations and 6 past iteration 5000 --
                    # essentially converged fits rejected for one bad final step,
                    # then handed -inf so best-of-num_fits could never select
                    # them. With num_fits=3 that silently drops a third of the
                    # restart budget.
                    #
                    # So keep the last good iterate unless the blow-up is severe
                    # enough that it is likely corrupt too (same run: early
                    # failures reached decreases of 2.04e+06, which is parameter
                    # divergence rather than a step-size problem). Only those
                    # still fail the fit.
                    if rel_decrease >= CATASTROPHIC_REL_TOL:
                        raise ValueError(
                            f"Iteration {it}: Likelihood decreased by {decrease:.2e}"
                        )
                    overshoot_stopped = (it, float(decrease), float(rel_decrease))
                if rel_decrease > PLATEAU_REL_TOL:
                    # history[-1] was appended at the top of this iteration and
                    # holds the params whose LL is likelihoods[-2] -- i.e. the
                    # better of the two iterates.
                    updated_component_params = history[-1]["component_params"]
                    updated_weights = history[-1]["weights"]
                    likelihoods = likelihoods[:-1]
                    objectives = objectives[:-1]
                    break

            if verbose:
                pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})
                pbar.update(1)

            # Don't converge before the tempering schedule has finished annealing
            # — early LL plateaus during warmup would stop the fit before
            # separation pressure is applied.
            allow_early_stop = not (separation_active and it < sep_min_iters)
            if kwargs.get("early_stopping", True) and it >= 1 and allow_early_stop:
                # Suppress invalid-subtract warning when both LLs are -inf
                # (NaN propagates harmlessly: NaN < 1e-8 → False, no break;
                # the np.isnan(likelihoods).any() guard above will trip on
                # the next iter and surface the real failure).
                with np.errstate(invalid='ignore'):
                    rel_change = (
                        np.abs(objectives[-1] - objectives[-2])
                        / abs(objectives[-2])
                    )
                if rel_change < 1e-8:
                    break

        # Return the best iterate rather than the last. These agree for a
        # monotone fit; they differ when the loop exited on a small decrease
        # (see best_state's definition above), and there the last iterate is
        # strictly worse than one we already computed. Truncate the LL/objective
        # series to match so the reported likelihood describes the parameters
        # actually returned.
        if best_state is not None and objectives[-1] < best_obj:
            updated_component_params, updated_weights, best_len = best_state
            likelihoods = likelihoods[:best_len]
            objectives = objectives[:best_len]

        if not constrained and check_submerged_duration:
            violated = constraints.multicomponent_density_constraint_violated(
                updated_component_params, xlims, multivariate=mv, mode="line",
            )
            if is_underwater and not violated:
                times_submerged.append(underwater_time)

        history.append(dict(
            component_params=updated_component_params, weights=updated_weights
        ))
        if verbose:
            pbar.close()
        # The final density-ratio check applies only to the retained univariate
        # constraint; multivariate separation does not use a density constraint.
        if constrained and not separation_active and \
                constraints.multicomponent_density_constraint_violated(
                    updated_component_params, xlims, multivariate=mv,
                ):
            raise ValueError("Final parameters violate density constraint")

    except (ValueError, ZeroDivisionError) as e:
        if kwargs.get("raise_on_error", True):
            raise
        import traceback
        warnings.warn(f"Failed fit: {e}\n{traceback.format_exc()}")
        return dict(
            component_params=updated_component_params,
            weights=updated_weights,
            likelihoods=[*likelihoods, -np.inf],
            kmeans=kmeans, xlims=xlims, times_submerged=[]
        )

    if overshoot_stopped is not None:
        _it, _dec, _rel = overshoot_stopped
        warnings.warn(
            f"EM stopped on a likelihood overshoot at iteration {_it} "
            f"(decrease {_dec:.2e}, relative {_rel:.2e}); reverted to the "
            f"previous iterate. The fit is usable but did not converge "
            f"normally."
        )

    return dict(
        component_params=updated_component_params,
        weights=updated_weights,
        likelihoods=likelihoods,
        overshoot_stopped=overshoot_stopped,
        guard_stats=dict(guard_stats),
        history=history,
        kmeans=kmeans,
        xlims=xlims,
        times_submerged=times_submerged,
        initial_params=initial_params,
        latent_q=latent_q,
    )


def _init_component_from_class(observations, sample_indicators, target_sample_idx,
                                latent_q, rng, old_params=None, old_weights=None,
                                residual_frac=0.4, multivariate=True):
    """Method-of-moments init for ONE new multivariate/CFUSN component,
    seeded from `target_sample_idx`'s (e.g. P/LP) own RESIDUAL data -- the
    subset of its observations the existing (frozen) fit explains worst --
    not from its raw class mean.

    Why residual, not raw mean: tried seeding from the plain class mean
    first and smoke-tested it on synthetic data with a known minority
    subpopulation (90 points from one cluster + 30 from a separate,
    displaced one, both labeled the same class). The raw mean is dominated
    by whatever majority subcluster the existing components ALREADY fit
    well, so the new component just rediscovers that majority and never
    reaches the actual gap -- confirmed directly (new component's fitted
    mean landed on the majority cluster, not the held-out minority one).
    Seeding from the worst-explained residual_frac (lowest total mixture
    log-density under the existing fit, via density_utils.log_joint_
    densities/logsumexp) fixes this: the new component starts where the
    existing fit is actually failing, which is the entire point of adding
    one.

    If `old_params`/`old_weights` are not given (e.g. direct unit-testing
    of this helper in isolation), falls back to the raw-class-mean
    behaviour over the whole class.

    Returns (mu, Delta, Gamma) in alternate form -- component_params for
    multivariate/CFUSN fits are stored natively in this form (see
    _em_update_multivariate/_em_update_cfusn's own `updated[c] = (mu_cand,
    Delta_cand, Gamma_cand)`), no canonical-form conversion needed.
    """
    mask = sample_indicators[:, target_sample_idx].astype(bool)
    Xc = np.asarray(observations)[mask]
    keep = ~np.all(np.isnan(Xc), axis=1)
    Xc = Xc[keep]
    p = Xc.shape[1]
    if len(Xc) < max(10, p + 2):
        raise ValueError(
            f"add_component_to_fit: target class {target_sample_idx} has only "
            f"{len(Xc)} usable observations (need at least {max(10, p + 2)})"
        )

    if old_params is not None and old_weights is not None:
        from .density_utils import log_joint_densities
        from scipy.special import logsumexp
        class_weights = np.asarray(old_weights)[target_sample_idx]
        lw = log_joint_densities(Xc, old_params, class_weights, multivariate=multivariate)
        log_mix = logsumexp(lw, axis=0)  # (Nc,) total mixture log-density per point
        n_residual = max(max(10, p + 2), int(np.ceil(residual_frac * len(Xc))))
        n_residual = min(n_residual, len(Xc))
        worst_idx = np.argsort(log_mix)[:n_residual]
        Xc = Xc[worst_idx]

    mu = np.nanmean(Xc, axis=0)
    cov = np.zeros((p, p))
    for d1 in range(p):
        for d2 in range(d1, p):
            both = ~np.isnan(Xc[:, d1]) & ~np.isnan(Xc[:, d2])
            if both.sum() >= 2:
                cov[d1, d2] = np.cov(Xc[both, d1], Xc[both, d2])[0, 1]
            cov[d2, d1] = cov[d1, d2]
    cov += 1e-6 * np.eye(p)
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-8:
        cov += (1e-8 - eigvals.min()) * np.eye(p)

    if latent_q == 1:
        Delta = rng.uniform(-0.1, 0.1, size=p) * np.sqrt(np.diag(cov))
        Gamma = cov - np.outer(Delta, Delta)
    else:
        Delta = rng.uniform(-0.1, 0.1, size=(p, latent_q)) * np.sqrt(np.diag(cov))[:, None]
        Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-8:
        Gamma += (1e-8 - eigvals_G.min()) * np.eye(p)

    return mu, Delta, Gamma


def add_component_to_fit(
    observations, sample_indicators, fit_result, target_sample_idx,
    constrained=False, multivariate=True, latent_q=2, **kwargs
):
    """Grow an already-converged fit by ONE new component, initialized from
    and primarily driven by `target_sample_idx`'s (e.g. P/LP) own data,
    while every pre-existing component is FROZEN (held byte-identical)
    throughout -- see update_steps.py's `frozen_components` kwarg, added
    specifically for this use case.

    Rationale: a uniform sample-weighting scheme (sample_balance_beta/
    sample_weight_transform) reweights the WHOLE EM objective and can
    degrade the fit's quality on the bulk/majority classes to help a small
    one (confirmed on real data: log/ln-weighted predictor-mv fits showed
    flat marginal LR+ and density curves that didn't track the real
    per-class histograms). Growing the fit instead -- keep the normal,
    unweighted fit's K components exactly as they converged, then add one
    new (K+1-th) component -- cannot degrade the existing classes' fit at
    all, since their parameters never change; it can only add explanatory
    power for whichever class the new component ends up being responsible
    for.

    Mechanics: the new component is seeded via method-of-moments directly
    on `target_sample_idx`'s own observations (see
    _init_component_from_class) and given a modest initial per-class
    mixing weight (manually set below, not re-estimated from data, since
    there's no existing weight for a component that didn't exist before).
    The subsequent EM iterations use the STANDARD (unmodified) E-step --
    responsibilities and per-class mixing weights are computed across all
    K+1 components for every class, not just the target one -- so if other
    classes also end up assigning the new component real responsibility,
    the mixing-weight re-estimation naturally reflects that; only the new
    component's own M-step is live (via frozen_components), everything
    else is standard, already-validated EM machinery.

    Parameters
    ----------
    fit_result : dict
        A converged single_fit(...) return value (K components) -- reused
        as the K+1 fit's starting point. Not re-validated for convergence
        here; pass in whatever fit_result you trust.
    target_sample_idx : int
        Sample-indicator column (e.g. the P/LP class) whose data seeds the
        new component's initialization and starting mixing weight.
    **kwargs : forwarded to single_fit (e.g. fit_seed, max_em_iters,
        num_fits-level concerns are the caller's responsibility -- this
        runs exactly one fit, not a multi-restart search).

    Returns the same dict shape as single_fit, now with K+1 components.
    """
    old_params = list(fit_result["component_params"])
    old_weights = np.asarray(fit_result["weights"], dtype=float)
    K = len(old_params)
    S, K_check = old_weights.shape
    if K_check != K:
        raise ValueError(f"fit_result weights shape {old_weights.shape} inconsistent with {K} components")

    rng = kwargs.get("rng") or np.random.RandomState(kwargs.get("fit_seed"))

    new_component = _init_component_from_class(
        observations, sample_indicators, target_sample_idx, latent_q, rng,
        old_params=old_params, old_weights=old_weights, multivariate=multivariate,
    )
    new_params = old_params + [new_component]

    # Expand weights from (S, K) to (S, K+1): the target class starts with
    # a modest share on the new component (taken proportionally from its
    # existing K-column weights, which are rescaled down so the row still
    # sums to 1); every other class starts at exactly 0 on the new
    # component -- not a hard constraint (the E-step/weight-reestimation
    # is standard and can move any class's weight onto the new component
    # if the data supports it), just a neutral starting point.
    INITIAL_TARGET_SHARE = 0.3
    new_col = np.zeros((S, 1))
    new_col[target_sample_idx, 0] = INITIAL_TARGET_SHARE
    expanded_weights = np.concatenate([old_weights, new_col], axis=1)
    expanded_weights[target_sample_idx, :K] *= (1.0 - INITIAL_TARGET_SHARE)

    return single_fit(
        observations, sample_indicators, K + 1,
        constrained=constrained,
        init_method=None, init_constraint_adjustment=None,
        multivariate=multivariate,
        initial_params=new_params,
        initial_weights=expanded_weights,
        frozen_components=set(range(K)),
        latent_q=latent_q,
        rng=rng,
        **kwargs,
    )


def _ensure_cfusn_params(params, latent_q):
    """Ensure all Delta in params are (p, q) matrices.

    If Delta is a (p,) vector and latent_q > 1, expand to (p, q) by
    placing the vector in column 0 and filling the rest with zeros.
    """
    result = []
    for mu, Delta, Gamma in params:
        Delta = np.asarray(Delta, dtype=float)
        if Delta.ndim == 1 and latent_q > 1:
            p = len(Delta)
            Delta_mat = np.zeros((p, latent_q))
            Delta_mat[:, 0] = Delta
            Delta = Delta_mat
        elif Delta.ndim == 2 and Delta.shape[1] != latent_q:
            p = Delta.shape[0]
            Delta_new = np.zeros((p, latent_q))
            q_copy = min(Delta.shape[1], latent_q)
            Delta_new[:, :q_copy] = Delta[:, :q_copy]
            Delta = Delta_new
        result.append((np.asarray(mu), Delta, np.asarray(Gamma)))
    return result