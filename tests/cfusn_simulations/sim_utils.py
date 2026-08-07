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
