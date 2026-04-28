from .update_steps import em_iteration, get_sample_weights
from .density_utils import get_likelihood, get_q, _ensure_matrix_delta
from .initializations import kmeans_init, methodOfMomentsInit, kmeans_init_mv
from . import constraints

import numpy as np
import logging
from tqdm.auto import tqdm
import warnings


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
                if mv:
                    initial_params, kmeans = kmeans_init_mv(
                        observations, n_clusters=N_components,
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
                print(f"[INIT FAILED] {e}")
                return dict(
                    component_params=[[] for _ in range(N_components)],
                    weights=W,
                    likelihoods=[-np.inf],
                    xlims=xlims,
                    times_submerged=[],
                )

        W = get_sample_weights(
            observations, sample_indicators, initial_params, W, multivariate=mv
        )

    em_kwargs = {}
    if mv and latent_q > 1:
        em_kwargs["n_mc_truncated"] = kwargs.get("n_mc_truncated", 500)

    history = [dict(component_params=initial_params, weights=W)]
    # Initial likelihood: no em_iteration has run yet so we must evaluate explicitly.
    likelihoods = np.array([
        get_likelihood(observations, sample_indicators, initial_params, W, multivariate=mv)
        / len(sample_indicators)
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
                updated_component_params, xlims, multivariate=mv
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

            if it > 0 and likelihoods[-1] < likelihoods[-2]:
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
                                bt_params, bt_weights, multivariate=mv
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

            if (
                kwargs.get("early_stopping", True)
                and it >= 1
                and np.abs(likelihoods[-1] - likelihoods[-2]) / abs(likelihoods[-2]) < 1e-8
            ):
                break

        if not constrained and check_submerged_duration:
            violated = constraints.multicomponent_density_constraint_violated(
                updated_component_params, xlims, multivariate=mv
            )
            if is_underwater and not violated:
                times_submerged.append(underwater_time)

        history.append(dict(
            component_params=updated_component_params, weights=updated_weights
        ))
        if verbose:
            pbar.close()
        if constrained and constraints.multicomponent_density_constraint_violated(
            updated_component_params, xlims, multivariate=mv
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