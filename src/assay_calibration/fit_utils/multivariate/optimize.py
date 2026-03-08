import numpy as np
from scipy import optimize, stats


def project_to_monotone_lrPlus(
    params,
    xmin,
    xmax,
    grid_n=120,
    eps_density=1e-12,
    maxiter=200,
    skew_bounds=(-5, 5),
):
    """
    Find the parameter set Q that minimizes KL(P || Q) where P is the current
    skew-normal mixture defined by `params`, subject to the constraint that
    the positive likelihood ratio log fp(x) - log fb(x) *decreasing* on [xmin,xmax].

    Inputs
    ------
    params : tuple or list
        ((skew1, loc1, scale1),
         (skew2, loc2, scale2),
         ...,
         (skewK, locK, scaleK),
         ((wp0, wp1, ..., wpk),(wb0, wb1, ...,wbk)))
    xmin, xmax : floats
        domain over which monotonicity must hold and (also) used for numeric KL integral.
    grid_n : int
        number of grid points to discretize [xmin, xmax] for constraints & integration.
    eps_density : float
        small floor to densities to avoid log(0).
    maxiter : int
        maximum iterations for optimizer.

    Returns
    -------
    result_params, info
        result_params: ((skew1, loc1, scale1), (skew2, loc2, scale2), (skew3, loc3, scale3), w_adj)
        info: dict containing optimizer success flag and message
    """
    # print("optimizing")
    # --- unpack input params ---
    component_param_sets = params[:-1]
    wP = np.array(params[-1][0], dtype=float)
    wB = np.array(params[-1][1], dtype=float)
    # Grid for constraints and integration
    x_grid = np.linspace(xmin, xmax, grid_n)

    # helper: build skew-normal pdf (scipy uses 'a' as skew)
    def skew_pdf(a, loc, scale, x):
        # scipy.stats.skewnorm.pdf handles a, loc, scale
        return stats.skewnorm.pdf(x, a, loc=loc, scale=scale)

    def mixture_pdf(param_sets, w, x):
        return np.stack(
            [w[i] * skew_pdf(p[0], p[1], p[2], x) for i, p in enumerate(param_sets)]
        ).sum(0)

    P_grids = [
        np.maximum(skew_pdf(p[0], p[1], p[2], x_grid), eps_density)
        for p in component_param_sets
    ]  # reference distributions for each skew normal component

    # --- parameter vectorization for optimizer ---
    # vector: [skew1, mu1, log scale1, skew2, loc2, log scale2, ..., skewK, locK, log scaleK, logit_wP1, logit_wP2,..., logit_wP_K-1, logit_wB1, logit_wB2,...,logit_wB_K-1]
    def packVector(component_param_sets, wPB):
        log_scales = np.log([p[2] for p in component_param_sets])
        # convert w (length 3) to two logits
        # use inverse-softmax: for w=[w1,w2,w3], logits = log(w[:2]/w3)
        wP = wPB[0]
        wB = wPB[1]
        wP = np.maximum(wP, 1e-12)
        wB = np.maximum(wB, 1e-12)
        K = len(component_param_sets)
        lP = [np.log(wP[i]) - np.log(wP[-1]) for i in range(K - 1)]
        lB = [np.log(wB[i]) - np.log(wB[-1]) for i in range(K - 1)]
        pack = []
        for compNum in range(len(component_param_sets)):
            pack.extend(
                [
                    component_param_sets[compNum][0],
                    component_param_sets[compNum][1],
                    log_scales[compNum],
                ]
            )
        pack.extend([*lP, *lB])
        assert len(pack) == 13
        return np.array(pack, dtype=float)

    def unpack_vector(vec):
        # [
        #  skew1, loc1, log scale1,
        #  skew2, loc2, log scale2,
        #  ...,
        #  skewK, locK, log scaleK,
        #  lP1, lP2,...,lP_K-1,
        #  lB1, lB2, ...,lB_K-1
        # ]
        # 3 * 3 + 2 * 2
        # |n| = 3 * K + 2 * (K-1)
        # |n| = 5 * K - 2
        # K = (|n| + 2) / 5
        n_components = int((len(vec) + 2) / 5)
        component_param_sets = []
        for compNum in range(n_components):
            start_idx = 3 * compNum
            skew, loc, log_scale = vec[start_idx : 3 + start_idx]
            component_param_sets.append([skew, loc, np.exp(log_scale)])
        lB = vec[-(n_components - 1) :]
        lP = vec[-2 * (n_components - 1) : -(n_components - 1)]

        # softmax-like conversion to weights (via logits relative to component 3)
        logsP = np.array([*lP, 0.0])
        exP = np.exp(logsP - np.max(logsP))
        wP = exP / np.sum(exP)
        logsB = np.array([*lB, 0.0])
        exB = np.exp(logsB - np.max(logsB))
        wB = exB / np.sum(exB)
        wPB = np.stack([wP, wB])
        # assert wPB.shape == (2,3)
        packed = [*component_param_sets, wPB]
        # assert len(packed) == 4
        return packed

    x_grid_local = x_grid  # closure
    # --- objective: KL(P || Q) over x_grid (trapezoidal integration) ---
    def objective(vec):
        component_param_sets = unpack_vector(vec)[:-1]
        Q_grids = [
            np.maximum(skew_pdf(p[0], p[1], p[2], x_grid_local), eps_density)
            for p in component_param_sets
        ]

        # KL(P||Q) approx = integral P * log(P/Q)
        integrands = [
            P_grid * np.log(P_grid / Q_grid) for P_grid, Q_grid in zip(P_grids, Q_grids)
        ]
        integrand = np.stack(integrands).sum(0)
        return np.trapezoid(integrand, x_grid_local)

    # --- monotonicity constraints as vector valued inequality ---
    # positive likelihood ratio should be monotonically decreasing

    def lrPlus_diff_constraints(vec):
        unpacked = unpack_vector(vec)
        component_param_sets = unpacked[:-1]
        wPB = unpacked[-1]
        wP = wPB[0]
        wB = wPB[1]
        fPath = mixture_pdf(component_param_sets, wP, x_grid_local)
        fBenign = mixture_pdf(component_param_sets, wB, x_grid_local)
        lrPlus = fPath / fBenign
        diffs = lrPlus[:-1] - lrPlus[1:]
        return diffs

    # --- bounds & initial guess ---
    # initial vector from current params
    x0 = packVector(component_param_sets, np.stack([wP, wB]))
    print("x0 dim", len(x0))
    # Bounds:
    # - skews: unbounded (but we can put wide bounds)
    # - mus: unbounded (put wide bounds)
    # - logs: bound scales to avoid extremely small or huge values
    # - logits: unbounded (but clamp to wide bounds)
    big = 100
    min_log_scale = np.log(1e-6)
    max_log_scale = np.log(1e3)
    n_components = len(component_param_sets)
    bnds = [skew_bounds, (-big, big), (min_log_scale, max_log_scale)] * n_components
    bnds += [
        *(
            2
            * [
                (-big, big),
            ]
            * (n_components - 1)
        )
    ]

    # Nonlinear inequality constraint: ratio_diff_constraints(vec) >= 0
    cons = ({"type": "ineq", "fun": lrPlus_diff_constraints},)

    options = {"maxiter": maxiter, "ftol": 1e-9}
    # print("running optimization")
    res = optimize.minimize(
        objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options=options
    )
    result_params = unpack_vector(res.x)
    # print("optimization done")
    lrPlusDiffs = lrPlus_diff_constraints(res.x)
    message = {
        "montonic": (lrPlusDiffs >= 0).all(),
        "lrPlusDiffs": lrPlusDiffs,
        "KL Penalty": res.fun,
    }
    if not res.success:
        message["success"] = False
        message["message"] = res.message
        print("optimization failed")
        # If failed, return original params
        return result_params, message
    # return unpacked optimized vector
    message["success"] = True
    message["message"] = res.message

    return result_params, message
