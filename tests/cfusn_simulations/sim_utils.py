"""Shared machinery for the CFUSN (q>=2) simulation-study scripts in this
package: a generative sampler matching the EM's own assumed model, a
per-dimension MCAR missingness injector, and a ground-truth recovery scorer
(Hungarian component matching + Delta column sign/permutation resolution).

The partial-distance k-means prototype that used to live here has been
promoted to production (initializations.py's kmeans_init_mv) -- see
sim_kmeans_missingness_init.py, which now compares production against a
locally-preserved copy of the old approach instead.

Not a pytest module -- imported by the sim_*.py scripts, which are
standalone argparse/__main__ scripts (see tests/verify_skew_sign_change_*.py
for the existing convention this package extends to q>=2 CFUSN).
"""
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Sampling ─────────────────────────────────────────────────────────────────

def _ensure_matrix_delta(Delta):
    Delta = np.asarray(Delta, dtype=float)
    if Delta.ndim == 1:
        return Delta.reshape(-1, 1)
    return Delta


def _safe_chol(Gamma, floor=1e-8):
    """Cholesky with a small ridge fallback if Gamma isn't quite PD."""
    Gamma = np.asarray(Gamma, dtype=float)
    Gamma = 0.5 * (Gamma + Gamma.T)
    try:
        return np.linalg.cholesky(Gamma)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(Gamma)
        ridge = floor - eigvals.min() + floor
        return np.linalg.cholesky(Gamma + ridge * np.eye(Gamma.shape[0]))


def sample_cfusn(mu, Delta, Gamma, n, rng, max_reject_rounds=200):
    """Draw n iid observations from CFUSN(mu, Delta, Gamma).

    Generative model (matches the EM's own E-step assumption in
    get_truncated_normal_moments_cfusn):
        X = mu + Delta @ T + chol(Gamma) @ Z
        T ~ TN_q(0, I_q, R_+^q)   (independent standard normal, truncated to
                                    the positive orthant, via rejection)
        Z ~ N(0, I_p)             independent of T

    Using this as the ground-truth generator makes recovery experiments a
    closed-loop test of whether EM recovers parameters under the model it
    assumes, not a test of model misspecification.

    Parameters
    ----------
    mu : (p,)
    Delta : (p,) or (p, q)
    Gamma : (p, p), SPD (or near enough -- see _safe_chol)
    n : int
    rng : np.random.RandomState
    max_reject_rounds : int -- rejection-sampling batches to attempt before
        raising. Acceptance rate for T is 2**-q (25% at q=2, the only value
        production uses -- cheap through q~4-5; for larger q, swap in the
        existing _gibbs_sample_tn_q from update_steps.py, which samples this
        exact target distribution as a special case with mean=0, cov=I).

    Returns
    -------
    X : (n, p)
    T : (n, q) -- the true latent skew variable per observation, returned
        for ground-truth diagnostics (e.g. checking eta recovery), not just X.
    """
    mu = np.asarray(mu, dtype=float)
    Delta = _ensure_matrix_delta(Delta)
    p, q = Delta.shape
    L = _safe_chol(Gamma)

    accepted = []
    n_needed = n
    for _ in range(max_reject_rounds):
        if n_needed <= 0:
            break
        batch = max(4 * n_needed, 10000)
        cand = rng.standard_normal((batch, q))
        keep = cand[np.all(cand > 0, axis=1)]
        accepted.append(keep)
        n_needed -= len(keep)
    T = np.concatenate(accepted, axis=0)
    if len(T) < n:
        raise RuntimeError(
            f"sample_cfusn: only accepted {len(T)}/{n} truncated-normal draws "
            f"after {max_reject_rounds} rejection rounds (q={q}); this should "
            f"not happen for q<=4-5 -- check q isn't unexpectedly large."
        )
    T = T[:n]

    Z = rng.standard_normal((n, p)) @ L.T
    X = mu[None, :] + T @ Delta.T + Z
    return X, T


def sample_cfusn_mixture(component_params, weights_per_sample, sample_sizes, rng):
    """K-component, multi-sample-class CFUSN mixture, mirroring the existing
    verify_skew_sign_change_*.py scripts' _generate_data.

    Parameters
    ----------
    component_params : list of (mu, Delta, Gamma), the K TRUE components
    weights_per_sample : (S, K) array -- mixing weights per sample class
        (rows should each sum to 1)
    sample_sizes : (S,) int array -- number of observations to draw per
        sample class
    rng : np.random.RandomState

    Returns
    -------
    X : (N, p)
    sample_indicators : (N, S) bool, one-hot sample-class membership
    true_components : (N,) int, which of the K components generated each row
        (ground truth, for diagnostics only -- not seen by the fitter)
    """
    K = len(component_params)
    S = weights_per_sample.shape[0]
    p = np.asarray(component_params[0][0]).shape[0]

    X_parts, sa_parts, comp_parts = [], [], []
    for s in range(S):
        n = int(sample_sizes[s])
        if n == 0:
            continue
        probs = np.asarray(weights_per_sample[s], dtype=float)
        probs = probs / probs.sum()
        comp_idx = rng.choice(K, size=n, p=probs)
        Xs = np.empty((n, p))
        for c in range(K):
            mask = comp_idx == c
            n_c = int(mask.sum())
            if n_c == 0:
                continue
            mu, Delta, Gamma = component_params[c]
            Xs[mask], _ = sample_cfusn(mu, Delta, Gamma, n_c, rng)
        sa = np.zeros((n, S), dtype=bool)
        sa[:, s] = True
        X_parts.append(Xs)
        sa_parts.append(sa)
        comp_parts.append(comp_idx)

    X = np.concatenate(X_parts, axis=0)
    sample_indicators = np.concatenate(sa_parts, axis=0)
    true_components = np.concatenate(comp_parts, axis=0)
    return X, sample_indicators, true_components


# ── Missingness injection ───────────────────────────────────────────────────

def inject_missingness(X, frac_per_dim, rng):
    """NaN out frac_per_dim[d] of rows independently per dimension d (MCAR).

    Deliberately simple (independent per-dimension Bernoulli masks) rather
    than modeling assay-specific missingness correlation -- MCAR isolates
    the statistical effect of sparsity from any particular missingness
    mechanism, which is the question these simulations are asking.

    Parameters
    ----------
    X : (N, p) fully-observed array
    frac_per_dim : (p,) array-like, fraction of rows to NaN per dimension
    rng : np.random.RandomState

    Returns
    -------
    X_missing : (N, p) copy of X with NaNs injected
    """
    X = np.asarray(X, dtype=float).copy()
    N, p = X.shape
    frac_per_dim = np.asarray(frac_per_dim, dtype=float)
    assert len(frac_per_dim) == p
    assert np.all((frac_per_dim >= 0) & (frac_per_dim < 1)), \
        "frac_per_dim must be in [0, 1) -- a dimension can't be 100% missing"

    for d in range(p):
        if frac_per_dim[d] <= 0:
            continue
        n_missing = int(round(frac_per_dim[d] * N))
        idx = rng.choice(N, size=n_missing, replace=False)
        X[idx, d] = np.nan

    # No hard requirement that fully-observed rows exist: kmeans_init_mv
    # already has a global-column-mean fallback for exactly this case
    # (needed to test realistically extreme missingness, e.g. TP53's
    # KawOligo, which is ~99% missing and has essentially zero rows that
    # are simultaneously observed across every sparse dimension at once).
    return X


def inject_block_missingness(X, blocks, block_frac_missing, rng):
    """Like inject_missingness, but each block of columns is jointly
    observed or jointly missing together per row (one Bernoulli draw per
    row per block), instead of independent per-dimension MCAR.

    Mirrors real multi-assay datasets like TP53's (confirmed via direct
    inspection of ms.scores): dims 0-7 there are ALWAYS co-observed
    together (2244/2244 pairwise co-observation -- one 8-readout assay a
    variant either has or doesn't), dims {8,9,12} likewise (8188/8188).
    Independent per-dimension MCAR (inject_missingness) cannot reproduce
    this correlated structure, which matters here: with p=16 blocks like
    this, EVERY pairwise overlap can be large (the co-observation graph is
    fully connected -- not "disjoint" dimensions) while the JOINT
    intersection across all p dims is still exactly empty (0/9911 rows
    fully observed for real TP53), which is the condition this whole
    investigation is about.

    Parameters
    ----------
    X : (N, p) fully-observed array
    blocks : list of column-index lists (may overlap or partition p; real
        data can have both jointly-missing groups and individually-sparse
        singleton "blocks" of size 1)
    block_frac_missing : list of floats, one per block, each in [0, 1)
    rng : np.random.RandomState
    """
    X = np.asarray(X, dtype=float).copy()
    N = X.shape[0]
    assert len(blocks) == len(block_frac_missing)
    for cols, frac in zip(blocks, block_frac_missing):
        if frac <= 0:
            continue
        n_missing = int(round(frac * N))
        idx = rng.choice(N, size=n_missing, replace=False)
        for c in cols:
            X[idx, c] = np.nan
    return X


# ── Projection-aware skewness estimation under pervasive missingness ───────
# ── (no row need be fully observed for these to return a value) ───────────

def _partial_projection(Xc, evec, min_weight_frac=0.5, return_weights=False):
    """Per-row projection onto evec using only that row's OBSERVED entries,
    renormalized by the fraction of evec's squared weight actually present
    -- the projection analog of _partial_distance_sq's per-row available-
    case normalization (already promoted to production for k-means
    clustering under missingness).

    Rows where the observed dimensions carry less than min_weight_frac of
    evec's total squared weight are dropped (too little of the
    projection's actual signal is present to trust the renormalized
    value) -- unlike _partial_distance_sq, which uses every row regardless
    of how little is observed (fine for a distance/argmin comparison, but
    a skewness estimate from a mostly-imputed-away projection would be
    unreliable in a way a clustering label wouldn't be).

    Returns a 1-D array of valid renormalized projections (may be shorter
    than len(Xc)), or an empty array if none qualify. If return_weights,
    also returns each kept row's weight_frac (how much of evec's weight
    was actually observed for it, in [min_weight_frac, 1]) -- a per-row
    information/reliability weight for _skewness_z_score_weighted's
    effective-sample-size correction, since a row rescaled up from e.g.
    50% observed weight carries genuinely less information than a fully-
    observed row, even though both contribute one value to the array.
    """
    Xc = np.asarray(Xc, dtype=float)
    evec = np.asarray(evec, dtype=float)
    obs = ~np.isnan(Xc)
    total_weight = float(np.sum(evec ** 2))
    if total_weight < 1e-12:
        return (np.array([]), np.array([])) if return_weights else np.array([])
    observed_weight = (obs * (evec ** 2)[None, :]).sum(axis=1)
    weight_frac = observed_weight / total_weight
    keep = weight_frac >= min_weight_frac
    if not keep.any():
        return (np.array([]), np.array([])) if return_weights else np.array([])
    Xf = np.where(obs, Xc, 0.0)
    raw_proj = Xf[keep] @ evec
    # Renormalize: an all-observed row's projection has "scale" sqrt(total_weight)
    # in the sense that Var(evec.X) ~ sum(evec_k^2) under iid-unit-variance dims;
    # a partially-observed row's raw dot product only accumulates
    # observed_weight worth of that -- rescale so partial and full rows are
    # on a comparable scale before computing skewness across them together.
    scale_correction = np.sqrt(total_weight / np.maximum(observed_weight[keep], 1e-12))
    projections = raw_proj * scale_correction
    if return_weights:
        return projections, weight_frac[keep]
    return projections


def _meanimpute_projection(Xc, evec):
    """Per-row projection onto evec after filling NaN with that column's
    own mean -- every row contributes a full-length projection (imputed
    entries contribute exactly the column mean, i.e. zero deviation along
    that axis, diluting rather than distorting the projection), so this
    never drops rows the way _partial_projection's min_weight_frac can,
    at the cost of systematically shrinking the estimated skewness toward
    0 in proportion to how much of each row was imputed.
    """
    Xc = np.asarray(Xc, dtype=float)
    evec = np.asarray(evec, dtype=float)
    col_means = np.nanmean(Xc, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    Xfilled = np.where(np.isnan(Xc), col_means[None, :], Xc)
    return Xfilled @ evec


# ── Ground-truth recovery scoring ───────────────────────────────────────────

def _omega(Delta, Gamma):
    Delta = _ensure_matrix_delta(Delta)
    Gamma = np.asarray(Gamma, dtype=float)
    return Gamma + Delta @ Delta.T


def match_components(true_params, fit_params):
    """Hungarian assignment of fit components to true components.

    Cost[i,j] = normalized ||mu_true_i - mu_fit_j||_2
              + normalized ||Omega_true_i - Omega_fit_j||_F
    where Omega = Gamma + Delta@Delta.T (sign-invariant in Delta, so this
    avoids conflating "wrong component match" with "right component,
    ambiguous Delta sign"). Generalizes to arbitrary K, unlike a location-
    only mu[0]-sort heuristic (which only works for well-separated 1-D-ish
    problems).

    Parameters
    ----------
    true_params, fit_params : lists of (mu, Delta, Gamma)

    Returns
    -------
    row_ind, col_ind : matched (true_idx, fit_idx) pairs, from
        scipy.optimize.linear_sum_assignment (length = min(K_true, K_fit))
    cost_matrix : (K_true, K_fit)
    """
    mu_true = np.array([np.asarray(p[0], dtype=float) for p in true_params])
    mu_fit = np.array([np.asarray(p[0], dtype=float) for p in fit_params])
    omega_true = np.array([_omega(p[1], p[2]) for p in true_params])
    omega_fit = np.array([_omega(p[1], p[2]) for p in fit_params])

    K_true, K_fit = len(true_params), len(fit_params)

    if K_true > 1:
        pairwise_mu = [np.linalg.norm(mu_true[i] - mu_true[j])
                       for i in range(K_true) for j in range(i + 1, K_true)]
        mu_scale = np.median(pairwise_mu) or 1.0
        pairwise_om = [np.linalg.norm(omega_true[i] - omega_true[j])
                       for i in range(K_true) for j in range(i + 1, K_true)]
        omega_scale = np.median(pairwise_om) or 1.0
    else:
        mu_scale, omega_scale = 1.0, 1.0

    cost = np.zeros((K_true, K_fit))
    for i in range(K_true):
        for j in range(K_fit):
            cost[i, j] = (
                np.linalg.norm(mu_true[i] - mu_fit[j]) / mu_scale
                + np.linalg.norm(omega_true[i] - omega_fit[j], ord="fro") / omega_scale
            )

    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, cost


def resolve_delta_ambiguity(Delta_true, Delta_fit):
    """Find the (permutation, sign) of Delta_fit's q columns minimizing
    Frobenius error against Delta_true.

    CFUSN's latent columns are only identified up to independent per-column
    sign flip and permutation (T and a relabeled/sign-flipped T' induce the
    same Delta@T distribution under the corresponding relabeling of Delta's
    columns) -- EM has no reason to recover any particular labeling. This
    answers "how close did EM get, allowing the best relabeling," which is
    the right question for a recovery-accuracy metric: the alternative
    (comparing raw column order) would report spuriously huge error whenever
    two columns happen to have swapped, which is expected/harmless.

    Brute-forced over q! permutations x 2**q signs -- exact and cheap for
    q=2 (production's value: 2*4=8 combinations); documented as fine
    through q~4 (24*16=384), factorial growth would need a smarter
    (e.g. Hungarian-on-|correlation|) approach for larger q, not needed now.

    Returns
    -------
    Delta_fit_aligned : (p, q)
    perm : tuple, best column permutation of Delta_fit
    signs : tuple of +/-1, best per-column sign flip (applied after perm)
    error : float, Frobenius error at the best alignment
    """
    Delta_true = _ensure_matrix_delta(Delta_true)
    Delta_fit = _ensure_matrix_delta(Delta_fit)
    q = Delta_true.shape[1]
    assert Delta_fit.shape[1] == q

    best = None
    for perm in itertools.permutations(range(q)):
        permuted = Delta_fit[:, perm]
        for signs in itertools.product([-1, 1], repeat=q):
            candidate = permuted * np.array(signs)[None, :]
            err = float(np.linalg.norm(Delta_true - candidate, ord="fro"))
            if best is None or err < best[3]:
                best = (candidate, perm, signs, err)
    return best


def score_recovery(true_params, fit_params, true_weight=None, fit_weight=None):
    """Match fit components to true components and score recovery error.

    Parameters
    ----------
    true_params, fit_params : lists of (mu, Delta, Gamma)
    true_weight, fit_weight : optional (K,) arrays of overall (not
        per-sample) mixing proportion per component, for weight_error.
        Callers with per-sample weight matrices should pass e.g. the
        sample-size-weighted average across samples.

    Returns
    -------
    dict with:
        'pairs' : list of per-matched-pair dicts (true_idx, fit_idx,
            mu_error, delta_error, gamma_error, omega_error, weight_error)
        'unmatched_true' : true component indices with no fit match
            (K_fit < K_true)
        'unmatched_fit' : fit component indices with no true match
            (K_fit > K_true -- spurious/extra components)
        'mean_omega_error' : mean Omega_error across matched pairs (primary
            headline recovery metric -- needs no ambiguity resolution)
    """
    row_ind, col_ind, _ = match_components(true_params, fit_params)
    K_true, K_fit = len(true_params), len(fit_params)

    pairs = []
    for ti, fi in zip(row_ind, col_ind):
        mu_t, Delta_t, Gamma_t = true_params[ti]
        mu_f, Delta_f, Gamma_f = fit_params[fi]
        mu_t = np.asarray(mu_t, dtype=float)
        mu_f = np.asarray(mu_f, dtype=float)
        Gamma_t = np.asarray(Gamma_t, dtype=float)
        Gamma_f = np.asarray(Gamma_f, dtype=float)

        _, _, _, delta_error = resolve_delta_ambiguity(Delta_t, Delta_f)

        entry = {
            "true_idx": int(ti),
            "fit_idx": int(fi),
            "mu_error": float(np.linalg.norm(mu_t - mu_f)),
            "delta_error": delta_error,
            "gamma_error": float(np.linalg.norm(Gamma_t - Gamma_f, ord="fro")),
            "omega_error": float(np.linalg.norm(
                _omega(Delta_t, Gamma_t) - _omega(Delta_f, Gamma_f), ord="fro"
            )),
        }
        if true_weight is not None and fit_weight is not None:
            entry["weight_error"] = float(abs(true_weight[ti] - fit_weight[fi]))
        pairs.append(entry)

    unmatched_true = [i for i in range(K_true) if i not in set(row_ind)]
    unmatched_fit = [j for j in range(K_fit) if j not in set(col_ind)]
    mean_omega_error = float(np.mean([p["omega_error"] for p in pairs])) if pairs else float("nan")

    return {
        "pairs": pairs,
        "unmatched_true": unmatched_true,
        "unmatched_fit": unmatched_fit,
        "mean_omega_error": mean_omega_error,
    }


# ── Legacy k-means init (pre-promotion reference, for comparison only) ──────

def kmeans_init_mv_legacy(X, **kwargs):
    """Exact copy of kmeans_init_mv as it existed before promoting
    partial-distance clustering to production (initializations.py):
    complete-rows-only sklearn KMeans, falling back to global-mean
    imputation of every missing entry (for every row, not just the missing
    ones) once too few complete rows exist. Do not "fix" this -- it exists
    specifically to reproduce the old behavior for before/after comparison
    scripts (sim_kmeans_missingness_init.py, sim_kawoligo_like_recovery.py).
    """
    import warnings
    from sklearn.cluster import KMeans
    from src.assay_calibration.fit_utils.cfusn.initializations import _init_delta_matrix

    rng = kwargs.get("rng") or np.random.RandomState()
    n_clusters = kwargs.get("n_clusters", 2)
    latent_q = kwargs.get("latent_q", 1)
    lambdaIndex = kwargs.get("lambdaIndex", 0)
    n_sign_per_cluster = 2 ** latent_q
    N, K_dim = X.shape

    complete_mask = ~np.isnan(X).any(axis=1)
    X_complete = X[complete_mask]
    min_needed = n_clusters * max(10, K_dim + 2)
    if len(X_complete) < min_needed:
        col_means = np.nanmean(X, axis=0)
        X_complete = np.where(np.isnan(X), col_means[None, :], X)
        complete_mask = np.ones(N, dtype=bool)

    global_cov = np.zeros((K_dim, K_dim))
    for d1 in range(K_dim):
        for d2 in range(d1, K_dim):
            both = ~np.isnan(X_complete[:, d1]) & ~np.isnan(X_complete[:, d2])
            if both.sum() >= 2:
                global_cov[d1, d2] = np.cov(X_complete[both, d1], X_complete[both, d2])[0, 1]
            global_cov[d2, d1] = global_cov[d1, d2]
    global_cov += 1e-6 * np.eye(K_dim)
    _ev = np.linalg.eigvalsh(global_cov)
    if _ev.min() < 1e-8:
        global_cov += (1e-8 - _ev.min()) * np.eye(K_dim)

    for _attempt in range(100):
        try:
            kmeans = KMeans(n_clusters=n_clusters, init=kwargs.get("kmeans_init", "random"),
                            n_init=1, random_state=rng)
            kmeans.fit(X_complete)
            labels = np.full(N, -1, dtype=int)
            labels[complete_mask] = kmeans.predict(X_complete)
            centers = kmeans.cluster_centers_

            for j in np.where(~complete_mask)[0]:
                obs = ~np.isnan(X[j])
                if not obs.any():
                    labels[j] = rng.randint(n_clusters)
                else:
                    labels[j] = np.argmin([
                        np.sum((X[j, obs] - centers[c, obs]) ** 2)
                        for c in range(n_clusters)
                    ])

            component_parameters = []
            for c in range(n_clusters):
                Xc = X[labels == c]
                small_cluster = len(Xc) < max(10, K_dim + 2)
                if len(Xc) > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mu = np.nanmean(Xc, axis=0)
                    missing_dims = np.isnan(mu)
                    if missing_dims.any():
                        global_mu = np.nanmean(X, axis=0)
                        mu[missing_dims] = global_mu[missing_dims]
                else:
                    mu = np.nanmean(X, axis=0)

                if small_cluster:
                    cov = global_cov / n_clusters
                else:
                    cov = np.zeros((K_dim, K_dim))
                    for d1 in range(K_dim):
                        for d2 in range(d1, K_dim):
                            both = ~np.isnan(Xc[:, d1]) & ~np.isnan(Xc[:, d2])
                            if both.sum() < 2:
                                cov[d1, d2] = 1e-2
                            else:
                                cov[d1, d2] = np.cov(Xc[both, d1], Xc[both, d2])[0, 1]
                            cov[d2, d1] = cov[d1, d2]
                    cov += 1e-6 * np.eye(K_dim)
                    eigvals = np.linalg.eigvalsh(cov)
                    if eigvals.min() < 1e-8:
                        cov += (1e-8 - eigvals.min()) * np.eye(K_dim)

                cluster_pattern_idx = (lambdaIndex // (n_sign_per_cluster ** c)) % n_sign_per_cluster
                cluster_sign_pattern = np.array([
                    ((cluster_pattern_idx >> j) & 1) * 2 - 1 for j in range(latent_q)
                ])
                Delta = _init_delta_matrix(cov, K_dim, latent_q,
                                           Xc=None if small_cluster else Xc,
                                           cluster_sign_pattern=cluster_sign_pattern, rng=rng)
                Gamma = cov - Delta @ Delta.T
                Gamma = 0.5 * (Gamma + Gamma.T)
                eigvals_G = np.linalg.eigvalsh(Gamma)
                if eigvals_G.min() < 1e-8:
                    Gamma += (1e-8 - eigvals_G.min()) * np.eye(K_dim)
                component_parameters.append((mu, Delta, Gamma))

            component_parameters.sort(key=lambda p: p[0][0])
            if len(component_parameters) == n_clusters:
                return component_parameters, kmeans
        except Exception:
            continue

    raise ValueError("Failed legacy init after 100 attempts")


# ── Gibbs-sampled reference E-step moments (diagnostic only, not for production) ──

def gibbs_truncated_mvn_moments(means, cov, n_mc=500, rng=None,
                                n_gibbs_samples=2000, n_burnin=200):
    """Drop-in replacement for update_steps._mc_truncated_mvn_moments (same
    signature/return shape: (N,q) means, (q,q) cov -> (N,q) eta, (N,q,q) Psi)
    using genuine Gibbs sampling (update_steps._gibbs_sample_tn_q) of the
    EXACT posterior T|X=x ~ TN_q(m, S, R_+^q) instead of production's q=2
    fast path, which computes each marginal exactly (closed-form truncated-
    normal formula) but approximates the cross term E[T1*T2|x] as
    independent-marginals-plus-a-linear-correlation-correction rather than
    the true bivariate truncated-normal cross-moment (which has no simple
    closed form). This function is that true cross-moment, at MCMC
    precision -- the reference to measure the analytic approximation's bias
    against.

    Deliberately slow (real Gibbs sampling with burn-in, looped per
    observation) -- for diagnostic comparisons on small N only, never a
    production substitute. n_mc is accepted but unused (kept only so this
    matches _mc_truncated_mvn_moments's call signature for monkeypatching).
    """
    from src.assay_calibration.fit_utils.cfusn.update_steps import _gibbs_sample_tn_q

    if rng is None:
        rng = np.random.RandomState()
    N, q = means.shape
    eta = np.zeros((N, q))
    Psi = np.zeros((N, q, q))
    for j in range(N):
        samples = _gibbs_sample_tn_q(means[j], cov, n_gibbs_samples,
                                     n_burnin=n_burnin, rng=rng)
        eta[j] = samples.mean(axis=0)
        Psi[j] = (samples.T @ samples) / len(samples)
    return eta, Psi


# ── Configurable-magnitude Delta init (diagnostic: is the default init scale too small?) ──

def init_delta_matrix_scaled(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None,
                             scale_factor=0.1):
    """Exact copy of initializations._init_delta_matrix, with the per-direction
    initial magnitude (`0.1 * sqrt(eigval)` in production) exposed as
    `scale_factor` instead of hardcoded. Used to test whether production's
    fixed 0.1x scale is systematically too conservative -- diagnosed via
    sim_delta_init_magnitude.py after finding that EM initialized exactly at
    known large-skew true parameters converges to a strictly higher
    likelihood than EM initialized via the standard (small-magnitude) path,
    on the same data (a genuine local-optimum, not a flat-likelihood-ridge
    or M-step-formula issue -- both were directly ruled out).
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        scale = scale_factor * np.sqrt(eigvals[idx])
        evec = eigvecs[:, idx]

        skew_sign = 1
        if Xc is not None:
            import scipy.stats as sps
            complete_rows = ~np.isnan(Xc).any(axis=1)
            Xc_comp = Xc[complete_rows]
            if len(Xc_comp) >= 8:
                sk = sps.skew(Xc_comp @ evec)
                if abs(sk) > 1e-6:
                    skew_sign = int(np.sign(sk))

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


# ── Restart-indexed magnitude cycling (Item 1: diversify magnitude on the ──
# ── SAME restart budget the sign enumeration already pays for) ──────────

def init_delta_matrix_cycling_magnitude(cov, p, q, Xc=None, cluster_sign_pattern=None,
                                        rng=None, restart_idx=0,
                                        magnitude_tiers=(0.1, 0.5, 1.0)):
    """Exact copy of init_delta_matrix_scaled, except scale_factor is chosen
    by cycling through magnitude_tiers keyed on restart_idx (`restart_idx %
    len(magnitude_tiers)`) instead of a single fixed value.

    This relabels the magnitude used by each of the sign-enumeration
    restarts that _init_delta_matrix already requires (lambdaIndex's 4**K
    sign combinations for q=2) -- it does not add any new restarts. The
    caller is responsible for passing a distinct restart_idx per restart
    (e.g. the same loop counter already used to pick lambdaIndex).
    """
    rng = rng or np.random.RandomState()
    scale_factor = magnitude_tiers[restart_idx % len(magnitude_tiers)]
    return init_delta_matrix_scaled(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                    rng=rng, scale_factor=scale_factor)


# ── Data-driven Delta magnitude init (method-of-moments, ported from the ──
# ── univariate path's sn_method_of_moments_init) ─────────────────────────

def _mom_delta_magnitude(projected_data):
    """Method-of-moments estimate of |delta| in [0,1) (the standardized
    skew-normal shape magnitude) from a 1-D projection's sample skewness --
    the same Azzalini skew-normal skewness-inversion sn_method_of_moments_init
    already uses for the univariate path, here reused to size the CFUSN
    Delta init's MAGNITUDE (not just its sign, which _init_delta_matrix
    already estimates this way) from the data instead of a fixed constant.

    Returns None if there isn't enough data or the inversion degenerates
    (caller should fall back to a fixed default in that case).
    """
    if len(projected_data) < 8:
        return None
    m3 = sps_skew(projected_data)
    if np.isnan(m3) or np.abs(m3) < 1e-10:
        return 0.0
    a1 = np.sqrt(2 / np.pi)
    c = (4 - np.pi) / 2
    try:
        delta = 1.0 / np.sqrt(a1 ** 2 * (1 + (c / np.abs(m3)) ** (2 / 3)))
    except (ZeroDivisionError, FloatingPointError):
        return None
    if np.isnan(delta) or delta >= 0.99:
        return None
    return float(delta)


def sps_skew(x):
    import scipy.stats as sps
    return sps.skew(x)


def init_delta_matrix_mom(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None,
                          fallback_scale=0.1):
    """Like init_delta_matrix_scaled, but sizes each column's magnitude from
    the data's own method-of-moments skewness estimate (_mom_delta_magnitude)
    instead of a fixed scale_factor -- so init is naturally small when the
    data shows little real skew and naturally larger when it shows a lot,
    rather than one constant that's simultaneously too small for real large
    skew and (if raised) prone to manufacturing spurious skew where none
    exists (see sim_delta_init_magnitude.py's fixed-scale sweep).

    Falls back to `fallback_scale * sqrt(eigval)` (matching production's
    current fixed-0.1 behavior) when there isn't enough data in Xc for a
    reliable moment estimate (small_cluster callers already pass Xc=None
    for exactly this reason).
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        evec = eigvecs[:, idx]

        skew_sign = 1
        delta_mag = None
        if Xc is not None:
            complete_rows = ~np.isnan(Xc).any(axis=1)
            Xc_comp = Xc[complete_rows]
            if len(Xc_comp) >= 8:
                projected = Xc_comp @ evec
                sk = sps_skew(projected)
                if not np.isnan(sk) and abs(sk) > 1e-6:
                    skew_sign = int(np.sign(sk))
                delta_mag = _mom_delta_magnitude(projected)

        if delta_mag is None:
            scale = fallback_scale * np.sqrt(eigvals[idx])
        else:
            scale = delta_mag * np.sqrt(eigvals[idx])

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


# ── Skewness-seeking direction finders (Item 4: is PCA's variance-maximizing ──
# ── column-1 direction the reason boosting magnitude sometimes backfires?) ──

def mardia_skew_vector(Xc):
    """Whitened third-moment direction: the vector generalization of the
    scalar building block of Mardia's multivariate skewness statistic
    b1,p = mean_i,j(g_ij**3), where g_ij = (x_i-xbar)'S^-1(x_j-xbar).

        g_i = (x_i - xbar)' S^-1 (x_i - xbar)      (Mahalanobis-sq, per row)
        v = mean_i( g_i * S^-1 (x_i - xbar) )

    Unlike PCA's top covariance eigenvector (variance-maximizing), this
    points toward the direction of greatest third-moment asymmetry --
    closed-form, deterministic (no restart-to-restart RNG jitter unlike
    projection pursuit), and cheap (reuses the same S^-1 the caller already
    has via cov).

    Returns a unit vector (p,), or None if S is singular or Xc has too few
    complete rows.
    """
    Xc = np.asarray(Xc, dtype=float)
    complete_rows = ~np.isnan(Xc).any(axis=1)
    Xf = Xc[complete_rows]
    if len(Xf) < 8:
        return None
    xbar = Xf.mean(axis=0)
    centered = Xf - xbar[None, :]
    S = np.cov(centered, rowvar=False)
    S = np.atleast_2d(S) + 1e-8 * np.eye(Xf.shape[1])
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None
    Sinv_centered = centered @ Sinv.T  # (n, p), each row = S^-1 (x_i - xbar)
    g = np.einsum("ij,ij->i", centered, Sinv_centered)  # (n,) Mahalanobis-sq
    v = (g[:, None] * Sinv_centered).mean(axis=0)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    return v / norm


def projection_pursuit_direction(Xc, candidate_dirs):
    """Among candidate_dirs (each a (p,) vector, need not be unit norm),
    return the (unit-normalized) one whose 1-D projection has the largest
    |sample skewness|. Cheap, direct alternative to Mardia's closed form --
    the tradeoff is that its natural generalization to q=2's second column
    (sequential deflation to find a second skew-maximizing direction
    orthogonal to the first) is a more edge-case-prone search, out of scope
    to generalize fully here (see init_delta_matrix_direction, which only
    ever uses this for column 1).
    """
    Xc = np.asarray(Xc, dtype=float)
    complete_rows = ~np.isnan(Xc).any(axis=1)
    Xf = Xc[complete_rows]
    if len(Xf) < 8:
        return None
    best = None
    for d in candidate_dirs:
        d = np.asarray(d, dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            continue
        d = d / norm
        sk = abs(sps_skew(Xf @ d))
        if np.isnan(sk):
            continue
        if best is None or sk > best[0]:
            best = (sk, d)
    return best[1] if best is not None else None


def init_delta_matrix_direction(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None,
                                direction_method="pca", scale_factor=0.1):
    """Like init_delta_matrix_scaled, but column 1's DIRECTION (not
    magnitude, which stays fixed-scale here to isolate the direction
    question from the magnitude questions in Items 1-3) can come from
    {"pca" (production default), "mardia", "projection_pursuit"} instead of
    always the top covariance eigenvector. Column 2 (q=2's second latent
    direction) always keeps the PCA heuristic -- generalizing the
    alternative direction-finders to a second, deflated direction is out of
    scope for this diagnostic.

    Falls back to the PCA direction for column 1 whenever the requested
    method can't produce one (not enough data, singular S, degenerate
    skewness) -- same fallback semantics as the other init_delta_matrix_*
    variants.
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    complete_rows = None
    Xc_comp = None
    if Xc is not None:
        complete_rows = ~np.isnan(Xc).any(axis=1)
        Xc_comp = Xc[complete_rows]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        scale = scale_factor * np.sqrt(eigvals[idx])
        pca_evec = eigvecs[:, idx]
        evec = pca_evec

        if j == 0 and direction_method != "pca" and Xc_comp is not None and len(Xc_comp) >= 8:
            if direction_method == "mardia":
                v = mardia_skew_vector(Xc_comp)
            elif direction_method == "projection_pursuit":
                candidate_dirs = [eigvecs[:, k] for k in top_idx] + [
                    eigvecs[:, k] for k in np.argsort(eigvals)[::-1]
                ]
                v = projection_pursuit_direction(Xc_comp, candidate_dirs)
            else:
                raise ValueError(f"unknown direction_method: {direction_method!r}")
            if v is not None:
                # Orient the alternative direction consistently with the PCA
                # eigenvector's sign convention (arbitrary otherwise) so the
                # downstream skew_sign logic (which assumes evec's own
                # arbitrary orientation is corrected for separately) isn't
                # silently flipped by the direction-finder's own sign choice.
                if np.dot(v, pca_evec) < 0:
                    v = -v
                evec = v

        skew_sign = 1
        if Xc_comp is not None and len(Xc_comp) >= 8:
            sk = sps_skew(Xc_comp @ evec)
            if not np.isnan(sk) and abs(sk) > 1e-6:
                skew_sign = int(np.sign(sk))

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


# ── Significance-gated method-of-moments Delta magnitude init ──────────────

def _skewness_z_score(projected_data):
    """|sample skewness| / its standard error under the null of no skew, for
    a sample of size n: SE = sqrt(6n(n-1) / ((n-2)(n+1)(n+3))) (the standard
    finite-sample formula for the SE of the skewness estimator, not the
    large-n approximation sqrt(6/n), since per-cluster n here can be modest).
    This is the data-driven quantity a significance gate thresholds against,
    instead of trusting the raw method-of-moments point estimate
    unconditionally (which sim_delta_init_mom.py showed manufactures
    meaningful spurious skew from pure sampling noise when true skew is 0).
    """
    n = len(projected_data)
    if n < 8:
        return 0.0, 0.0
    m3 = sps_skew(projected_data)
    if np.isnan(m3):
        return 0.0, 0.0
    se = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    z = abs(m3) / se if se > 0 else 0.0
    return float(m3), float(z)


def _skewness_z_score_weighted(projected_data, weights=None):
    """Like _skewness_z_score, but uses an EFFECTIVE sample size (Kish's
    formula: n_eff = (sum w)^2 / sum(w^2)) in the SE formula instead of the
    raw row count, when per-row weights are given.

    Motivation: _partial_projection's rescaled rows are NOT all equally
    informative -- a row reconstructed from 50% of an eigenvector's weight
    was rescaled up by ~1.4x and is genuinely noisier than a fully-observed
    row, but the plain _skewness_z_score counts every row the same,
    treating a batch of low-information reconstructed rows as if it were a
    clean iid sample of the same size. This was diagnosed directly: the
    SAME James-Stein c=1.0 calibrated on clean data produced far worse
    zero/small-regime false positives under TP53-shaped missingness
    (sim_delta_init_missingness_c_sweep.py), consistent with z being
    systematically inflated (SE under-estimated) when weights are ignored.
    Kish's n_eff shrinks toward the count of "worth one clean data point"
    when weights are uneven/small, so it correctly returns a SMALLER,
    more honest sample size (hence bigger SE, smaller z) when most kept
    rows carry only partial information -- without needing a separately
    retuned c for the missing-data case.

    weights=None (or all-equal) reduces exactly to _skewness_z_score's
    plain n (n_eff == n in that case).
    """
    n = len(projected_data)
    if n < 8:
        return 0.0, 0.0
    m3 = sps_skew(projected_data)
    if np.isnan(m3):
        return 0.0, 0.0
    if weights is None:
        n_eff = float(n)
    else:
        w = np.asarray(weights, dtype=float)
        sw = w.sum()
        sw2 = (w ** 2).sum()
        n_eff = (sw ** 2 / sw2) if sw2 > 1e-12 else float(n)
    if n_eff < 8:
        return 0.0, 0.0
    se = np.sqrt(6 * n_eff * (n_eff - 1) / ((n_eff - 2) * (n_eff + 1) * (n_eff + 3)))
    z = abs(m3) / se if se > 0 else 0.0
    return float(m3), float(z)


# ── Continuous shrinkage (Item 2): smooth alternatives to the hard z-gate ──

def _shrinkage_james_stein(z, c):
    """max(0, 1 - c/z**2) -- exactly 0 at z=0, ->1 as z->inf, single knob c
    (a squared-z threshold -- c=z_threshold**2 makes the knee comparable to
    init_delta_matrix_mom_gated's hard cutoff at the same z_threshold).
    """
    if z <= 0:
        return 0.0
    return max(0.0, 1.0 - c / (z * z))


def _shrinkage_sigmoid(z, z0, k=2.0):
    """Smooth logistic shrinkage weight in (0, 1), centered at z0 with slope
    k -- unlike the James-Stein form, never exactly 0 (small residual trust
    even at z=0), but smooth everywhere including near z=0.
    """
    return 1.0 / (1.0 + np.exp(-k * (z - z0)))


def init_delta_matrix_mom_shrunk(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None,
                                 fallback_scale=0.1, shrinkage_fn=_shrinkage_james_stein,
                                 **shrinkage_kwargs):
    """Like init_delta_matrix_mom_gated, but instead of a hard 0/1 cutoff on
    z, blends the MoM magnitude toward fallback_scale continuously via
    shrinkage_fn(z, **shrinkage_kwargs) -> weight in [0,1]:

        scale = weight * mom_scale + (1 - weight) * fallback_scale

    This is the fix sim_delta_init_mom_gated.py's z_threshold sweep pointed
    at: a hard gate forces one threshold to serve both "don't manufacture
    skew from noise" (wants a high threshold) and "don't miss real medium
    skew" (wants a low threshold) -- there is no single value that's good at
    both. A continuous blend lets partial trust scale smoothly with z instead
    of an all-or-nothing cutoff, which sim_delta_init_mom_shrinkage.py's
    Pareto-frontier comparison tests directly against the hard gate.
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        evec = eigvecs[:, idx]
        fallback = fallback_scale * np.sqrt(eigvals[idx])

        skew_sign = 1
        scale = fallback
        if Xc is not None:
            complete_rows = ~np.isnan(Xc).any(axis=1)
            Xc_comp = Xc[complete_rows]
            if len(Xc_comp) >= 8:
                projected = Xc_comp @ evec
                m3, z = _skewness_z_score(projected)
                if abs(m3) > 1e-6:
                    skew_sign = int(np.sign(m3))
                delta_mag = _mom_delta_magnitude(projected)
                if delta_mag is not None:
                    mom_scale = delta_mag * np.sqrt(eigvals[idx])
                    weight = float(shrinkage_fn(z, **shrinkage_kwargs))
                    weight = min(max(weight, 0.0), 1.0)
                    scale = weight * mom_scale + (1 - weight) * fallback

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


def init_delta_matrix_mom_shrunk_cycling(cov, p, q, Xc=None, cluster_sign_pattern=None,
                                         rng=None, fallback_scale=0.1,
                                         shrinkage_fn=_shrinkage_james_stein,
                                         restart_idx=0, multiplier_tiers=(0.5, 1.0, 2.0),
                                         **shrinkage_kwargs):
    """Item 1 + Item 2 combined: use the James-Stein/sigmoid-shrunk MoM
    magnitude (init_delta_matrix_mom_shrunk) as a data/confidence-informed
    CENTER, then cycle a multiplier around that center by restart_idx (the
    same free relabeling init_delta_matrix_cycling_magnitude uses) instead
    of cycling among data-blind fixed absolute scale_factors.

    This is deliberately NOT "run Item 1 and Item 2 as two independent
    magnitude-setters" -- that would be redundant (Item 1's fixed tiers
    ignore the data-driven estimate; Item 2's shrunk estimate never varies
    across restarts on its own, since it's deterministic given the data,
    which is exactly why a fixed-magnitude Item 2 can't use its restart
    budget to hedge against its own point estimate being wrong). Instead
    the shrunk estimate sets the center and cycling only varies AROUND it,
    still at zero added restart cost (same restart_idx already used for
    sign-pattern lambdaIndex decoding).
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]
    multiplier = multiplier_tiers[restart_idx % len(multiplier_tiers)]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        evec = eigvecs[:, idx]
        fallback = fallback_scale * np.sqrt(eigvals[idx])

        skew_sign = 1
        center = fallback
        if Xc is not None:
            complete_rows = ~np.isnan(Xc).any(axis=1)
            Xc_comp = Xc[complete_rows]
            if len(Xc_comp) >= 8:
                projected = Xc_comp @ evec
                m3, z = _skewness_z_score(projected)
                if abs(m3) > 1e-6:
                    skew_sign = int(np.sign(m3))
                delta_mag = _mom_delta_magnitude(projected)
                if delta_mag is not None:
                    mom_scale = delta_mag * np.sqrt(eigvals[idx])
                    weight = float(shrinkage_fn(z, **shrinkage_kwargs))
                    weight = min(max(weight, 0.0), 1.0)
                    center = weight * mom_scale + (1 - weight) * fallback

        scale = multiplier * center

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


def init_delta_matrix_mom_shrunk_bimodal(cov, p, q, Xc=None, cluster_sign_pattern=None,
                                         rng=None, fallback_scale=0.1,
                                         shrinkage_fn=_shrinkage_james_stein,
                                         restart_idx=0,
                                         null_c=25.0, trust_c=0.0,
                                         **shrinkage_kwargs):
    """Alternative to init_delta_matrix_mom_shrunk_cycling: rather than
    diluting ONE shrunk-magnitude center with a symmetric multiplier around
    it (which sim_delta_init_combined_1_2.py showed actively hurts
    medium/large-regime recovery -- the shrunk center was already a good
    calibrated estimate there, and multiplying it by 0.5x/2x just adds noise
    for best-of-LL to occasionally pick a worse local optimum from), give
    the restart budget to the two competing HYPOTHESES explicitly instead of
    asking one shrinkage constant to average across both:

      - even restart_idx -> "probably no real skew": shrink hard (null_c,
        a high James-Stein c -- defaults to 25.0, the top of the swept
        z_threshold**2 grid, i.e. close to pure fallback_scale)
      - odd restart_idx  -> "trust the data's own estimate": shrink little
        or not at all (trust_c, defaults to 0.0 -- the raw, ungated MoM
        magnitude)

    No shrinkage formula is asked to be simultaneously right about both
    "truly no skew" and "real skew" at once; the SIGN dimension still
    enumerates via cluster_sign_pattern exactly as in every other
    init_delta_matrix_* variant, orthogonal to this restart_idx-driven
    magnitude-hypothesis toggle. Still zero added restart cost (same
    restart_idx already used for sign-pattern lambdaIndex decoding).
    """
    c = null_c if (restart_idx % 2 == 0) else trust_c
    return init_delta_matrix_mom_shrunk(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern,
                                        rng=rng, fallback_scale=fallback_scale,
                                        shrinkage_fn=shrinkage_fn, c=c)


def init_delta_matrix_mom_shrunk_projection(cov, p, q, Xc=None, cluster_sign_pattern=None,
                                            rng=None, fallback_scale=0.1,
                                            shrinkage_fn=_shrinkage_james_stein,
                                            projection_method="complete",
                                            min_weight_frac=0.5,
                                            **shrinkage_kwargs):
    """Like init_delta_matrix_mom_shrunk, but generalizes how each column's
    1-D projection is computed from Xc, via projection_method:

      "complete"   -- production's current behavior: only rows complete
                       across ALL p dimensions (Xc[~isnan(Xc).any(axis=1)]).
                       Confirmed via direct inspection of real TP53 data
                       (9911 rows x 16 dims, 0 rows fully observed even
                       though the pairwise co-observation graph is fully
                       CONNECTED -- see inject_block_missingness's
                       docstring) that this can be permanently 0 rows for
                       real multi-assay genes, silently disabling BOTH the
                       sign estimate and the shrunk magnitude estimate on
                       every cluster, every restart -- confirmed via a
                       bit-identical before/after TP53 sanity refit.
      "partial"    -- _partial_projection: per-row projection using only
                       that row's observed entries, renormalized, dropping
                       rows with too little of evec's weight observed
                       (min_weight_frac).
      "meanimpute" -- _meanimpute_projection: every row contributes (NaN
                       filled with column mean), diluting rather than
                       dropping.

    Sign and magnitude both come from whichever projection this produces
    (same downstream logic as init_delta_matrix_mom_shrunk otherwise).
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        evec = eigvecs[:, idx]
        fallback = fallback_scale * np.sqrt(eigvals[idx])

        skew_sign = 1
        scale = fallback
        if Xc is not None:
            if projection_method == "complete":
                complete_rows = ~np.isnan(Xc).any(axis=1)
                Xc_comp = Xc[complete_rows]
                projected = Xc_comp @ evec if len(Xc_comp) >= 8 else np.array([])
            elif projection_method == "partial":
                projected = _partial_projection(Xc, evec, min_weight_frac=min_weight_frac)
            elif projection_method == "meanimpute":
                projected = _meanimpute_projection(Xc, evec)
            else:
                raise ValueError(f"unknown projection_method: {projection_method!r}")

            if len(projected) >= 8:
                m3, z = _skewness_z_score(projected)
                if abs(m3) > 1e-6:
                    skew_sign = int(np.sign(m3))
                delta_mag = _mom_delta_magnitude(projected)
                if delta_mag is not None:
                    mom_scale = delta_mag * np.sqrt(eigvals[idx])
                    weight = float(shrinkage_fn(z, **shrinkage_kwargs))
                    weight = min(max(weight, 0.0), 1.0)
                    scale = weight * mom_scale + (1 - weight) * fallback

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


def init_delta_matrix_mom_shrunk_adaptive_c(cov, p, q, Xc=None, cluster_sign_pattern=None,
                                            rng=None, fallback_scale=0.1,
                                            shrinkage_fn=_shrinkage_james_stein,
                                            projection_method="complete",
                                            min_weight_frac=0.5, c_base=1.0,
                                            min_info_frac=0.01):
    """Like init_delta_matrix_mom_shrunk_projection, but instead of using a
    single externally-fixed James-Stein c for every column regardless of
    how much missingness degraded its projection, scales c UP inversely
    with how much of the cluster's data actually contributed to that
    column's projection:

        info_frac = (# rows used in the projection) / (# rows in cluster)
        c_adaptive = c_base / max(info_frac, min_info_frac)

    At info_frac=1 (a fully-observed cluster/column -- e.g. clean data, or
    projection_method="complete" when completeness happens to hold), this
    reduces EXACTLY to c_base (this investigation's clean-data-calibrated
    value, c_base=1.0) by construction. As missingness shrinks info_frac
    (e.g. TP53-shaped data, where far fewer rows qualify for a "partial"
    projection than exist in the cluster), c rises automatically -- no
    separate, hand-picked "missing-data constant" (like the c=4 explored
    as a fixed universal compromise) is needed; the SAME formula anchored
    at the SAME clean-data c_base handles both regimes, with the degree of
    missingness (directly measurable from the data) doing the interpolation
    instead of a manual choice between two fixed candidates.

    min_info_frac floors c_adaptive's growth (avoiding a divide-by-near-
    zero blowup when only a handful of rows qualify).
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]
    n_cluster = len(Xc) if Xc is not None else 0

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        evec = eigvecs[:, idx]
        fallback = fallback_scale * np.sqrt(eigvals[idx])

        skew_sign = 1
        scale = fallback
        if Xc is not None and n_cluster > 0:
            if projection_method == "complete":
                complete_rows = ~np.isnan(Xc).any(axis=1)
                Xc_comp = Xc[complete_rows]
                projected = Xc_comp @ evec if len(Xc_comp) >= 8 else np.array([])
            elif projection_method == "partial":
                projected = _partial_projection(Xc, evec, min_weight_frac=min_weight_frac)
            elif projection_method == "meanimpute":
                projected = _meanimpute_projection(Xc, evec)
            else:
                raise ValueError(f"unknown projection_method: {projection_method!r}")

            if len(projected) >= 8:
                m3, z = _skewness_z_score(projected)
                if abs(m3) > 1e-6:
                    skew_sign = int(np.sign(m3))
                delta_mag = _mom_delta_magnitude(projected)
                if delta_mag is not None:
                    mom_scale = delta_mag * np.sqrt(eigvals[idx])
                    info_frac = max(len(projected) / n_cluster, min_info_frac)
                    c_adaptive = c_base / info_frac
                    weight = float(shrinkage_fn(z, c=c_adaptive))
                    weight = min(max(weight, 0.0), 1.0)
                    scale = weight * mom_scale + (1 - weight) * fallback

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


def init_delta_matrix_mom_shrunk_projection_ess(cov, p, q, Xc=None, cluster_sign_pattern=None,
                                                rng=None, fallback_scale=0.1,
                                                shrinkage_fn=_shrinkage_james_stein,
                                                min_weight_frac=0.5,
                                                **shrinkage_kwargs):
    """Like init_delta_matrix_mom_shrunk_projection(projection_method=
    "partial"), but feeds _partial_projection's per-row weight_frac into
    _skewness_z_score_weighted's effective-sample-size correction instead
    of the plain row-count z-score -- the fix for
    sim_delta_init_missingness_c_sweep.py's finding that the SAME
    clean-data-calibrated c=1.0 produced far worse zero/small-regime false
    positives under TP53-shaped missingness (a partially-reconstructed row
    was being counted as fully informative, inflating z). The goal is a
    SINGLE c that works reasonably for both clean and highly-missing data,
    rather than needing two separately-calibrated constants.

    Only implemented for projection_method="partial" (the method that
    actually produces meaningful per-row weights to correct with;
    "complete" and "meanimpute" don't have a natural analog -- "complete"
    rows are already all full-weight, "meanimpute" rows are all nominally
    "full length" but diluted rather than reweighted).
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        evec = eigvecs[:, idx]
        fallback = fallback_scale * np.sqrt(eigvals[idx])

        skew_sign = 1
        scale = fallback
        if Xc is not None:
            projected, weights = _partial_projection(
                Xc, evec, min_weight_frac=min_weight_frac, return_weights=True
            )

            if len(projected) >= 8:
                m3, z = _skewness_z_score_weighted(projected, weights=weights)
                if abs(m3) > 1e-6:
                    skew_sign = int(np.sign(m3))
                delta_mag = _mom_delta_magnitude(projected)
                if delta_mag is not None:
                    mom_scale = delta_mag * np.sqrt(eigvals[idx])
                    weight = float(shrinkage_fn(z, **shrinkage_kwargs))
                    weight = min(max(weight, 0.0), 1.0)
                    scale = weight * mom_scale + (1 - weight) * fallback

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


# ── Multi-component whole-init routines: empirical-Bayes-pooled vs. a ──────
# ── single manually-chosen c, for a direct, apples-to-apples comparison ────
# ── (needs K>=2 components so there's more than 2 z-scores to pool -- a ────
# ── single-component fit's 2 columns are too few for EB to have anything ───
# ── meaningful to estimate from). Clean, fully-observed data only -- this ──
# ── isolates "does pooling beat a fixed c" from the separate missingness/ ──
# ── projection-noise issues already investigated (init_delta_matrix_mom_ ──
# ── shrunk_projection and friends). ─────────────────────────────────────

def _cluster_column_stats(X, labels, n_clusters, latent_q):
    """First pass shared by both multi-component init routines below:
    k-means already ran (labels given); for each cluster, eigendecompose
    its covariance and compute each of the top-q eigenvectors' skewness
    z-score and method-of-moments magnitude estimate. Returns a list (one
    dict per cluster) with cov/mu/per-column stats, plus the flat list of
    every (cluster, column) z-score -- the pool empirical Bayes draws from.
    """
    K_dim = X.shape[1]
    cluster_info = []
    for c in range(n_clusters):
        Xc = X[labels == c]
        mu = Xc.mean(axis=0)
        cov = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(K_dim)
        eigvals_cov = np.linalg.eigvalsh(cov)
        if eigvals_cov.min() < 1e-8:
            cov = cov + (1e-8 - eigvals_cov.min()) * np.eye(K_dim)
        eigvals, eigvecs = np.linalg.eigh(cov)
        top_idx = np.argsort(eigvals)[::-1][:latent_q]
        cols = []
        for idx in top_idx:
            evec = eigvecs[:, idx]
            proj = Xc @ evec
            m3, z = _skewness_z_score(proj)
            delta_mag = _mom_delta_magnitude(proj)
            cols.append(dict(evec=evec, eigval=eigvals[idx], m3=m3, z=z, delta_mag=delta_mag))
        cluster_info.append(dict(mu=mu, cov=cov, cols=cols))
    return cluster_info


def _build_delta_from_weights(cluster_info, n_clusters, latent_q, lambdaIndex,
                              weights, fallback_scale, rng):
    """Second pass shared by both routines: given a per-(cluster,column)
    shrinkage weight already decided (however it was derived), build each
    cluster's (mu, Delta, Gamma). weights is a flat list in the same
    (cluster, column) order _cluster_column_stats produced.
    """
    n_sign_per_cluster = 2 ** latent_q
    K_dim = cluster_info[0]["mu"].shape[0]
    component_parameters = []
    w_idx = 0
    for c in range(n_clusters):
        info = cluster_info[c]
        Delta = np.zeros((K_dim, latent_q))
        cluster_pattern_idx = (lambdaIndex // (n_sign_per_cluster ** c)) % n_sign_per_cluster
        cluster_sign_pattern = [((cluster_pattern_idx >> j) & 1) * 2 - 1 for j in range(latent_q)]
        for j, colinfo in enumerate(info["cols"]):
            weight = weights[w_idx]
            w_idx += 1
            fallback = fallback_scale * np.sqrt(colinfo["eigval"])
            skew_sign = int(np.sign(colinfo["m3"])) if abs(colinfo["m3"]) > 1e-6 else 1
            if colinfo["delta_mag"] is not None:
                mom_scale = colinfo["delta_mag"] * np.sqrt(colinfo["eigval"])
                scale = weight * mom_scale + (1 - weight) * fallback
            else:
                scale = fallback
            enum_sign = cluster_sign_pattern[j]
            Delta[:, j] = skew_sign * enum_sign * scale * colinfo["evec"]

        Delta += rng.uniform(-0.05, 0.05, size=(K_dim, latent_q)) * np.sqrt(np.diag(info["cov"]))[:, None]
        Gamma = info["cov"] - Delta @ Delta.T
        Gamma = 0.5 * (Gamma + Gamma.T)
        eigvals_G = np.linalg.eigvalsh(Gamma)
        if eigvals_G.min() < 1e-6:
            for _ in range(20):
                Delta *= 0.9
                Gamma = info["cov"] - Delta @ Delta.T
                if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                    break
        component_parameters.append((info["mu"], Delta, Gamma))

    component_parameters.sort(key=lambda p: p[0][0])
    return component_parameters


def kmeans_init_mv_fixed_c(X, **kwargs):
    """Multi-component whole-init routine using a single, externally
    chosen, manually-tuned James-Stein c applied independently to every
    (cluster, column) -- the status quo approach this session has been
    calibrating via simulation sweeps. Clean fully-observed data only
    (plain sklearn KMeans); kwargs: n_clusters, latent_q, lambdaIndex, rng,
    c (the manually-chosen constant, default 1.0), fallback_scale.
    """
    from sklearn.cluster import KMeans

    rng = kwargs.get("rng") or np.random.RandomState()
    n_clusters = kwargs.get("n_clusters", 2)
    latent_q = kwargs.get("latent_q", 2)
    lambdaIndex = kwargs.get("lambdaIndex", 0)
    c = kwargs.get("c", 1.0)
    fallback_scale = kwargs.get("fallback_scale", 0.1)

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=rng)
    labels = km.fit_predict(X)

    cluster_info = _cluster_column_stats(X, labels, n_clusters, latent_q)
    weights = []
    for info in cluster_info:
        for colinfo in info["cols"]:
            weights.append(min(max(_shrinkage_james_stein(colinfo["z"], c=c), 0.0), 1.0))

    component_parameters = _build_delta_from_weights(
        cluster_info, n_clusters, latent_q, lambdaIndex, weights, fallback_scale, rng
    )
    return component_parameters, (labels, km)


def kmeans_init_mv_eb(X, **kwargs):
    """Multi-component whole-init routine using a POOLED empirical-Bayes
    shrinkage weight instead of an externally chosen c: computes every
    (cluster, column) z-score within this SAME init call (needs no EM --
    only the k-means clustering that already happened), then derives ONE
    shared shrinkage weight from their pooled spread via the classical
    positive-part James-Stein rule:

        weight = max(0, 1 - (m-2) / sum_j(z_j^2))

    where m = total number of pooled z-scores and the sum runs over every
    (cluster, column) pair. If the pooled z^2 values are mostly close to
    their null expectation (~1 each, under no real skew anywhere), m-2 is
    comparable to the sum and weight collapses toward 0 (correctly
    distrusting the individual per-column MoM estimates); if several
    z-scores are large (real skew present), the sum dominates m-2 and
    weight rises toward 1 -- entirely from the observed data, no
    externally chosen constant. (This is the single-shared-weight,
    positive-part form of the classical James-Stein estimator for
    simultaneously shrinking several estimates toward zero; a fuller
    version could shrink each cluster's own subset differently, but the
    single pooled weight is the natural first thing to test against a
    single global manually-chosen c, since that's exactly what it's
    replacing.)

    kwargs: same as kmeans_init_mv_fixed_c, minus c (not used -- derived).
    """
    from sklearn.cluster import KMeans

    rng = kwargs.get("rng") or np.random.RandomState()
    n_clusters = kwargs.get("n_clusters", 2)
    latent_q = kwargs.get("latent_q", 2)
    lambdaIndex = kwargs.get("lambdaIndex", 0)
    fallback_scale = kwargs.get("fallback_scale", 0.1)

    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=rng)
    labels = km.fit_predict(X)

    cluster_info = _cluster_column_stats(X, labels, n_clusters, latent_q)
    all_z = [colinfo["z"] for info in cluster_info for colinfo in info["cols"]]
    m = len(all_z)
    S = float(sum(z ** 2 for z in all_z))
    if m > 2 and S > 1e-9:
        eb_weight = max(0.0, 1.0 - (m - 2) / S)
    else:
        eb_weight = 0.0
    weights = [eb_weight] * m

    component_parameters = _build_delta_from_weights(
        cluster_info, n_clusters, latent_q, lambdaIndex, weights, fallback_scale, rng
    )
    return component_parameters, (labels, km)


def init_delta_matrix_mom_gated(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None,
                                fallback_scale=0.1, z_threshold=2.0):
    """Like init_delta_matrix_mom, but only trusts the method-of-moments
    magnitude estimate when the projected data's sample skewness is
    "significant" relative to its own sampling noise (z-score >=
    z_threshold, using _skewness_z_score) -- otherwise falls back to
    fallback_scale, same as when there isn't enough data at all. This is
    the fix for sim_delta_init_mom.py's finding that plain MoM sizing
    manufactures spurious skew (from pure noise) when true skew is ~0:
    the un-gated estimator can't distinguish "small real skewness" from
    "sampling noise around zero skewness" using the point estimate alone,
    but the z-score can.
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        evec = eigvecs[:, idx]

        skew_sign = 1
        delta_mag = None
        if Xc is not None:
            complete_rows = ~np.isnan(Xc).any(axis=1)
            Xc_comp = Xc[complete_rows]
            if len(Xc_comp) >= 8:
                projected = Xc_comp @ evec
                m3, z = _skewness_z_score(projected)
                if z >= z_threshold:
                    if abs(m3) > 1e-6:
                        skew_sign = int(np.sign(m3))
                    delta_mag = _mom_delta_magnitude(projected)

        if delta_mag is None:
            scale = fallback_scale * np.sqrt(eigvals[idx])
        else:
            scale = delta_mag * np.sqrt(eigvals[idx])

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


# Module-level config for forced_skew_init_delta_matrix, set by the caller
# (e.g. a diagnostic script) before monkeypatching production's
# _init_delta_matrix with this function. Kept as globals rather than extra
# args since this needs the EXACT _init_delta_matrix(cov, p, q, Xc=None,
# cluster_sign_pattern=None, rng=None) signature to drop in via monkeypatch.
FORCED_SKEW_DIMS = []          # column indices to force
FORCED_SKEW_SIGNS = {}         # {dim_idx: +1 or -1}, sign of forced skew
FORCED_SKEW_SCALE_FRAC = 0.95  # fraction of sqrt(cov[d,d]) to force onto
# Must be set by the caller to the ORIGINAL (pre-monkeypatch) production
# _init_delta_matrix callable -- NOT re-imported by name at call time, since
# once monkeypatched, the module-level name resolves back to this wrapper
# and would recurse infinitely.
FORCED_SKEW_BASE_FN = None


def forced_skew_init_delta_matrix(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
    """Diagnostic-only variant of production's _init_delta_matrix (see
    src/assay_calibration/fit_utils/cfusn/initializations.py): runs the
    normal data-driven init for every dimension, then FORCES the dimensions
    listed in FORCED_SKEW_DIMS onto a large single-axis skew (magnitude
    FORCED_SKEW_SCALE_FRAC * sqrt(cov[d,d]), matching the M-step's own
    magnitude-cap ceiling, sign from FORCED_SKEW_SIGNS) regardless of what
    the cluster's own local data says.

    Purpose: distinguish whether the real CFUSN mixture's near-symmetric
    fitted components (measured on real TP53 v3 fits: alpha_ratio ~0.07-1.4
    for dimensions where an unconditional 1D skew-normal fit finds ~0.995)
    reflects a genuine MLE optimum, or a local-optimum/init deficiency --
    by seeding EM at the opposite (large-skew) extreme and checking, via
    held-out val_ll, whether it converges back down (current low skew is
    correct) or stays skewed with equal/better val_ll (init was the bug).
    See the Phase 4 plan for full context.
    """
    base_fn = FORCED_SKEW_BASE_FN
    if base_fn is None:
        from src.assay_calibration.fit_utils.cfusn.initializations import _init_delta_matrix as base_fn

    Delta = base_fn(cov, p, q, Xc=Xc, cluster_sign_pattern=cluster_sign_pattern, rng=rng)

    for d in FORCED_SKEW_DIMS:
        if d >= p:
            continue
        sign = FORCED_SKEW_SIGNS.get(d, 1)
        mag = FORCED_SKEW_SCALE_FRAC * np.sqrt(max(cov[d, d], 0.0))
        Delta[d, :] = 0.0
        Delta[d, 0] = sign * mag

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


# Module-level config for pooled_fallback_init_delta_matrix (Phase 4 Step 2,
# Approach A). POOLED_DELTA_VEC is a (p,) array: one skew-informed magnitude
# (signed) per raw dimension, precomputed once per bootstrap from a pooled
# (all-cluster) scipy.stats.skewnorm.fit on the standardized training data
# for that dimension -- NOT per-cluster (per-cluster subsets are exactly
# where the sample-size/missingness instability this shrinkage exists to
# guard against already lives; see the Phase 4 plan's Step 2 rationale).
POOLED_DELTA_VEC = None


def pooled_fallback_init_delta_matrix(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
    """Approach A (Phase 4 Step 2): identical to production's
    _init_delta_matrix except for ONE change -- the JS-shrinkage fallback
    target. Production shrinks weakly-supported clusters toward
    _FALLBACK_SCALE * sqrt(eigval) (an essentially symmetric, near-zero-skew
    constant); this variant shrinks them toward this dimension's own known
    population-level skew (from POOLED_DELTA_VEC, projected onto the
    cluster's eigenvector), so an under-supported cluster's fallback is "the
    population's known skew level" instead of "no skew." Well-supported
    clusters are unaffected -- the adaptive-c weighting already lets their
    own local MoM estimate dominate.
    """
    global POOLED_DELTA_VEC
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]
    n_cluster = len(Xc) if Xc is not None else 0

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        evec = eigvecs[:, idx]
        if POOLED_DELTA_VEC is not None:
            fallback = float(abs(POOLED_DELTA_VEC @ evec))
        else:
            fallback = _FALLBACK_SCALE_LOCAL * np.sqrt(eigvals[idx])

        skew_sign = 1
        scale = fallback
        if Xc is not None and n_cluster > 0:
            projected = _partial_projection(Xc, evec, min_weight_frac=0.5)
            if len(projected) >= 8:
                m3, z = _skewness_z_score(projected)
                if abs(m3) > 1e-6:
                    skew_sign = int(np.sign(m3))
                delta_mag = _mom_delta_magnitude(projected)
                if delta_mag is not None:
                    mom_scale = delta_mag * np.sqrt(eigvals[idx])
                    info_frac = max(len(projected) / n_cluster, 0.01)
                    c_adaptive = 1.0 / info_frac
                    weight = _shrinkage_james_stein(z, c=c_adaptive)
                    weight = min(max(weight, 0.0), 1.0)
                    scale = weight * mom_scale + (1 - weight) * fallback

        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


_FALLBACK_SCALE_LOCAL = 0.1


def compute_pooled_delta_vec(X):
    """Compute POOLED_DELTA_VEC from (N, p) data (may contain NaN): per
    dimension, scipy.stats.skewnorm.fit on all non-NaN values, converted to
    a signed Delta magnitude via delta = alpha/sqrt(1+alpha^2), magnitude =
    delta*scale (same decomposition sample_cfusn's generative model uses:
    X = mu + Delta@T + chol(Gamma)@Z, T ~ half-normal per axis for q=1 --
    matches Azzalini's loc/scale/alpha parameterization with mu=loc,
    Delta=scale*delta, Gamma=scale^2*(1-delta^2)).
    """
    import scipy.stats as sps_stats
    p = X.shape[1]
    vec = np.zeros(p)
    for d in range(p):
        x = X[:, d]
        x = x[~np.isnan(x)]
        if len(x) < 20:
            continue
        try:
            a, loc, scale = sps_stats.skewnorm.fit(x)
        except Exception:
            continue
        a = np.clip(a, -1e4, 1e4)
        delta = a / np.sqrt(1 + a ** 2)
        vec[d] = delta * scale
    return vec
