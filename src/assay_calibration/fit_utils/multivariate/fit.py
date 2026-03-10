from .update_steps import em_iteration, get_sample_weights
from .density_utils import get_likelihood
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
    """
    Fit a mixture model using EM.

    Parameters
    ----------
    observations : np.ndarray
        (N,) for univariate, (N, K) for multivariate.
    sample_indicators : np.ndarray
        (N, S) one-hot sample membership.
    N_components : int
    constrained : bool
    init_method : str  ("kmeans" or "method_of_moments")
    init_constraint_adjustment : str
    multivariate : bool
        Whether data/model is multivariate.

    Returns
    -------
    dict with component_params, weights, likelihoods, history, etc.
    """
    MAX_EM_ITERS = kwargs.get("max_em_iters", 10000)
    verbose = kwargs.get("verbose", True)
    check_submerged_duration = kwargs.get("check_submerged_duration", False)
    MIN_SCALE = 1e-100
    mv = multivariate

    if kwargs.get("submerge_steps") is not None:
        raise NotImplementedError("submerge_steps is deprecated")

    # Determine xlims
    if mv:
        # per-dimension bounds — use nanmin/nanmax since data has NaN
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
                print(f"[INIT] mv={mv}, method={'kmeans_mv' if mv else 'kmeans'}, "
                      f"obs shape={observations.shape}, "
                      f"NaN count={np.isnan(observations).sum()}, "
                      f"xlims={xlims}")
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

    history = [dict(component_params=initial_params, weights=W)]
    likelihoods = np.array([
        get_likelihood(observations, sample_indicators, initial_params, W, multivariate=mv)
        / len(sample_indicators)
    ])

    # First iteration
    try:
        updated_component_params, updated_weights = em_iteration(
            observations, sample_indicators, initial_params, W,
            constrained, xlims, multivariate=mv, iterNum=0
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

    likelihoods = np.array([
        *likelihoods,
        get_likelihood(
            observations, sample_indicators,
            updated_component_params, updated_weights, multivariate=mv
        ) / len(sample_indicators)
    ])

    if verbose:
        pbar = tqdm(total=MAX_EM_ITERS, leave=False, desc="EM Iteration")

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

            updated_component_params, updated_weights = em_iteration(
                observations, sample_indicators,
                updated_component_params, updated_weights,
                constrained, xlims, multivariate=mv, iterNum=it + 1
            )

            # enforce minimum scale (univariate only)
            if not mv:
                for i, (a, loc, scale) in enumerate(updated_component_params):
                    if scale < MIN_SCALE:
                        updated_component_params[i] = (a, loc, max(scale, MIN_SCALE))

            # check underwater duration
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

            likelihoods = np.array([
                *likelihoods,
                get_likelihood(
                    observations, sample_indicators,
                    updated_component_params, updated_weights, multivariate=mv
                ) / len(sample_indicators)
            ])

            if it > 0 and likelihoods[-1] < likelihoods[-2]:
                decrease = likelihoods[-2] - likelihoods[-1]
                if decrease > 1e-13:
                    if mv:
                        # Available-case M-step doesn't guarantee monotonicity.
                        # Backtrack: interpolate between old and new params.
                        old_params = history[-1]['component_params']
                        old_weights = history[-1]['weights']
                        alpha = 0.5
                        for _ in range(10):
                            bt_params = []
                            for c in range(len(updated_component_params)):
                                mu_o, D_o, G_o = old_params[c]
                                mu_n, D_n, G_n = updated_component_params[c]
                                mu_bt = (1 - alpha) * mu_o + alpha * mu_n
                                D_bt = (1 - alpha) * D_o + alpha * D_n
                                G_bt = (1 - alpha) * G_o + alpha * G_n
                                G_bt = 0.5 * (G_bt + G_bt.T)
                                bt_params.append((mu_bt, D_bt, G_bt))
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
                            # Backtracking failed — revert to old params
                            updated_component_params = old_params
                            updated_weights = old_weights
                            likelihoods[-1] = likelihoods[-2]
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

        # final underwater check
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
    )
