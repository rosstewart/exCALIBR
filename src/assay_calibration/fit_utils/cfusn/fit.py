from .update_steps import em_iteration, get_sample_weights
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
    """Per-observation M-step weights from sample_proportions or sample_balance_beta.

    sample_proportions : array-like of length S
        Explicit relative weight for each sample (e.g. [2,1,1,1] upweights
        sample 0 twice as much as the others).  Takes precedence over
        sample_balance_beta when both are supplied.  Values are normalised
        internally so only their ratios matter.

    sample_balance_beta : float (default 0)
        Continuous balance parameter.  beta=0 → standard EM (no reweighting);
        beta=1 → each sample contributes equally regardless of size.

    Returns None when neither parameter requests any reweighting.
    """
    proportions = kwargs.get("sample_proportions", None)
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
    elif beta > 0 and N_samples > 1:
        N_ref = N_per_sample[N_per_sample > 0].min() if (N_per_sample > 0).any() else 1.0
        per_sample_w = np.where(
            N_per_sample > 0, (N_ref / np.maximum(N_per_sample, 1.0)) ** beta, 0.0
        )
    else:
        return None

    return (sample_indicators.astype(float) * per_sample_w).sum(axis=1)


def single_fit(
    observations, sample_indicators, N_components, constrained,
    init_method, init_constraint_adjustment, multivariate=False, **kwargs
):
    MAX_EM_ITERS = kwargs.get("max_em_iters", 10000)
    verbose = kwargs.get("verbose", True)
    check_submerged_duration = kwargs.get("check_submerged_duration", False)
    MIN_SCALE = 1e-100
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
                if kwargs.get("raise_on_error", False):
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

        W = get_sample_weights(
            observations, sample_indicators, initial_params, W, multivariate=mv
        )

    em_kwargs = {}
    em_kwargs["constraint_mode"] = constraint_mode
    em_kwargs["rng"] = rng
    if mv and latent_q > 1:
        em_kwargs["n_mc_truncated"] = kwargs.get("n_mc_truncated", 500)
    if sample_weights_per_obs is not None:
        em_kwargs["sample_weights"] = sample_weights_per_obs

    history = [dict(component_params=initial_params, weights=W)]
    # Initial likelihood: no em_iteration has run yet so we must evaluate explicitly.
    # Pass sample_weights so the initial LL is on the same (weighted) objective
    # the M-step will optimise — preserves monotonicity under β>0.
    likelihoods = np.array([
        get_likelihood(
            observations, sample_indicators, initial_params, W,
            multivariate=mv, sample_weights=sample_weights_per_obs,
        ) / len(sample_indicators)
    ])

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
        if kwargs.get("raise_on_error", False):
            raise
        print(f"[FIRST EM ITER FAILED] ZeroDivisionError: {e}")
        return dict(
            component_params=initial_params, weights=W,
            likelihoods=[*likelihoods, -np.inf],
            kmeans=kmeans, xlims=xlims, times_submerged=[]
        )

    likelihoods = np.append(likelihoods, ll)

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

            # Separation (tempering + repulsion) deliberately trades likelihood
            # for non-overlap, so the penalised objective is not EM-monotone in
            # the raw LL. Skip backtracking for separation fits and accept the
            # decrease; convergence is governed by the LL plateau after the
            # annealing schedule completes (see sep_min_iters guard below).
            if it > 0 and likelihoods[-1] < likelihoods[-2] and not separation_active:
                decrease = likelihoods[-2] - likelihoods[-1]
                # Trigger backtracking only on a decrease that exceeds floating-
                # point noise on the LL. The original 1e-13 absolute threshold
                # fires on every plateau iteration once EM has converged,
                # wasting up to 10 LL evaluations per iteration on noise.
                bt_threshold = 1e-8 * abs(likelihoods[-2])
                if decrease > bt_threshold:
                    if mv:
                        # Backtracking: get_likelihood is kept here because it
                        # evaluates candidate params that aren't stored anywhere
                        # and are different on each alpha step.
                        old_params = history[-1]['component_params']
                        old_weights = history[-1]['weights']
                        alpha = 0.5
                        for _ in range(10):
                            bt_params = _interpolate_params(
                                old_params, updated_component_params, alpha
                            )
                            bt_weights = (1 - alpha) * old_weights + alpha * updated_weights
                            bt_ll = get_likelihood(
                                observations, sample_indicators,
                                bt_params, bt_weights, multivariate=mv,
                                sample_weights=sample_weights_per_obs,
                            ) / len(sample_indicators)
                            if bt_ll >= likelihoods[-2] - 1e-13:
                                updated_component_params = bt_params
                                updated_weights = bt_weights
                                likelihoods[-1] = bt_ll
                                break
                            alpha *= 0.5
                        else:
                            updated_component_params = old_params
                            updated_weights = old_weights
                            likelihoods[-1] = likelihoods[-2]
                        # Backtracking modified the iterate after em_iteration's
                        # weight/LL pass cached log_pdfs on pre-backtrack params.
                        # Force next iter's E-step to recompute density.
                        cached_log_pdfs = None
                    else:
                        raise ValueError(
                            f"Iteration {it}: Likelihood decreased by {decrease:.2e}"
                        )

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
                        np.abs(likelihoods[-1] - likelihoods[-2])
                        / abs(likelihoods[-2])
                    )
                if rel_change < 1e-8:
                    break

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
        if kwargs.get("raise_on_error", False):
            raise
        import traceback
        warnings.warn(f"Failed fit: {e}\n{traceback.format_exc()}")
        return dict(
            component_params=updated_component_params,
            weights=updated_weights,
            likelihoods=[*likelihoods, -np.inf],
            kmeans=kmeans, xlims=xlims, times_submerged=[]
        )

    return dict(
        component_params=updated_component_params,
        weights=updated_weights,
        likelihoods=likelihoods,
        history=history,
        kmeans=kmeans,
        xlims=xlims,
        times_submerged=times_submerged,
        initial_params=initial_params,
        latent_q=latent_q,
    )


def _interpolate_params(old_params, new_params, alpha):
    """Interpolate between old and new component params.

    Works for both q=1 (Delta is vector) and q>1 (Delta is matrix).
    """
    bt_params = []
    for c in range(len(old_params)):
        mu_o, D_o, G_o = old_params[c]
        mu_n, D_n, G_n = new_params[c]

        mu_o, mu_n = np.asarray(mu_o), np.asarray(mu_n)
        D_o, D_n = np.asarray(D_o, dtype=float), np.asarray(D_n, dtype=float)
        G_o, G_n = np.asarray(G_o), np.asarray(G_n)

        mu_bt = (1 - alpha) * mu_o + alpha * mu_n
        D_bt = (1 - alpha) * D_o + alpha * D_n
        G_bt = (1 - alpha) * G_o + alpha * G_n
        G_bt = 0.5 * (G_bt + G_bt.T)
        bt_params.append((mu_bt, D_bt, G_bt))
    return bt_params


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