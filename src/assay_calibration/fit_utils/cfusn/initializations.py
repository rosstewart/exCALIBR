"""
Initialization routines for multivariate skew-normal mixture models.

Supports both restricted MSN (q=1, Delta is a vector) and CFUSN (q>=1, Delta is a matrix).

For CFUSN initialization:
    - Delta is initialized as (p, q) matrix with small random entries
    - Gamma is initialized from cluster covariance minus Delta @ Delta.T
    - The latent dimension q is passed via kwargs["latent_q"]
"""

from .constraints import (
    density_constraint_violated,
    multicomponent_density_constraint_violated,
)
from . import separation
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats as sps
import itertools
import warnings


# ══════════════════════════════════════════════
# Univariate initializations (unchanged)
# ══════════════════════════════════════════════

def _lambda_signs(n_components, kwargs, rng):
    """Return a length-n_components tuple of +/-1 skew signs.

    Sourced from the enumerated sign pattern at kwargs["lambdaIndex"] (so that,
    across a bootstrap's restarts, every 2^n_components sign combination is
    covered exactly once) when lambdaIndex is supplied; otherwise falls back
    to a random +/-1 draw per component.
    """
    if "lambdaIndex" in kwargs:
        LambdasTable = list(itertools.product([-1, 1], repeat=n_components))
        return LambdasTable[kwargs["lambdaIndex"] % len(LambdasTable)]
    return tuple(rng.choice([-1, 1]) for _ in range(n_components))


def kmeans_init(X, **kwargs):
    rng = kwargs.get("rng") or np.random.RandomState()
    repeat = 0
    while repeat < 1000:
        n_clusters = kwargs.get("n_clusters", 2)
        init = kwargs.get("kmeans_init", "random")
        kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=rng)
        X = np.array(X).reshape((-1, 1))
        kmeans.fit(X)
        cluster_assignments = kmeans.predict(X)
        component_parameters = []
        force_gaussian = kwargs.get("force_gaussian", False)
        lambdas = _lambda_signs(n_clusters, kwargs, rng)
        for i in range(n_clusters):
            X_cluster = X[cluster_assignments == i]
            loc, scale = sps.norm.fit(X_cluster)
            a = 0.0 if force_gaussian else lambdas[i] * rng.uniform(0, 0.25)
            component_parameters.append((a, float(loc), float(scale)))
        component_parameters = sorted(component_parameters, key=lambda x: x[1])
        if kwargs.get("constrained", True):
            component_parameters = fix_to_satisfy_density_constraint(
                component_parameters,
                (X.min(), X.max()),
                **kwargs,
            )
        if len(component_parameters) == 0 or any(
            len(p) == 0 for p in component_parameters
        ):
            repeat += 1
        else:
            return component_parameters, kmeans
    raise ValueError("Failed to initialize")


def sn_method_of_moments_init(X, rng=None, force_gaussian=False):
    rng = rng or np.random.RandomState()
    m1 = np.mean(X)
    m2 = np.var(X)
    if m2 < 1e-10:
        return []
    if force_gaussian:
        return 0.0, m1, max(np.sqrt(m2), 1e-6)
    m3 = sps.skew(X)
    a1 = np.sqrt(2 / np.pi)
    c = (4 - np.pi) / 2  # skewness coefficient: gamma1 = c*(a1*delta)^3 / (1-a1^2*delta^2)^(3/2)
    try:
        if np.abs(m3) < 1e-10:
            return rng.uniform(-0.25, 0.25), m1, max(np.sqrt(m2), 1e-6)
        # Invert the SN skewness formula for delta (dimensionless, no m2):
        #   delta = sign(gamma1) / sqrt(a1^2 * (1 + (c/|gamma1|)^(2/3)))
        delta = np.sign(m3) / np.sqrt(a1**2 * (1 + (c / np.abs(m3)) ** (2/3)))
        if np.isnan(delta) or np.abs(delta) >= 0.99:
            return rng.uniform(-0.25, 0.25), m1, max(np.sqrt(m2), 1e-6)
        denom = 1 - a1**2 * delta**2
        if denom <= 1e-10:
            return rng.uniform(-0.25, 0.25), m1, max(np.sqrt(m2), 1e-6)
        sigma = max(np.sqrt(m2 / denom), 1e-6)
        mu = m1 - a1 * delta * sigma
        alpha = delta / np.sqrt(max(1 - delta**2, 1e-12))
        if np.any(np.isnan([mu, sigma, alpha])) or np.any(np.isinf([mu, sigma, alpha])):
            return []
        return alpha, mu, sigma
    except (ZeroDivisionError, RuntimeWarning):
        return []


def methodOfMomentsInit(X, n_components, constrained, max_attempts=1000, **kwargs):
    rng = kwargs.get("rng") or np.random.RandomState()
    lambdas = _lambda_signs(n_components, kwargs, rng)

    for attempt in range(max_attempts):
        if rng.random() < 0.7:
            base = np.linspace(0, 100, n_components + 1)[1:-1]
            iqr = np.percentile(X, 75) - np.percentile(X, 25)
            jitter = rng.normal(0, iqr * 0.1, len(base))
            cutPoints = np.percentile(X, np.sort(np.clip(base + jitter, 1, 99)))
        else:
            cutPoints = np.sort(
                rng.uniform(
                    np.percentile(X, 5), np.percentile(X, 95), n_components - 1
                )
            )

        component_parameters = []
        success = True
        for i in range(n_components):
            if i == 0:
                Xc = X[X <= cutPoints[0]]
            elif i == n_components - 1:
                Xc = X[X > cutPoints[-1]]
            else:
                Xc = X[(X > cutPoints[i - 1]) & (X <= cutPoints[i])]
            if len(Xc) < max(10, int(0.05 * len(X))):
                success = False
                break
            params = sn_method_of_moments_init(
                Xc, rng=rng, force_gaussian=kwargs.get("force_gaussian", False)
            )
            if len(params) == 0:
                success = False
                break
            params = list(params)
            # Keep the moment-estimated magnitude; override only the sign with
            # lambdas[i] so restarts still cover all 2^n_components sign
            # patterns. (force_gaussian already zeroed params[0] above, and
            # lambdas[i] * abs(0.0) stays 0.)
            params[0] = lambdas[i] * abs(params[0])
            component_parameters.append(tuple(params))

        if success and all(len(p) > 0 for p in component_parameters):
            max_scale = max(p[2] for p in component_parameters)
            component_parameters = [
                (p[0], p[1], max_scale) for p in component_parameters
            ]
            if constrained:
                component_parameters = fix_to_satisfy_density_constraint(
                    component_parameters, (X.min(), X.max()), **kwargs
                )
            if len(component_parameters) and all(
                len(p) > 0 for p in component_parameters
            ):
                return component_parameters
    print("MoM constraint failed")
    return None


def fix_to_satisfy_density_constraint(component_parameters, xlims, **kwargs):
    n_components = len(component_parameters)
    param_to_adjust = kwargs.get("init_constraint_adjustment", "scale")
    assert param_to_adjust == "scale"
    if any(len(p) == 0 for p in component_parameters):
        return [[] for _ in range(n_components)]
    component_parameters = sorted(component_parameters, key=lambda x: x[1])
    for i in range(n_components):
        if len(component_parameters[i]) >= 3 and component_parameters[i][2] < 1e-6:
            component_parameters[i] = list(component_parameters[i])
            component_parameters[i][2] = 1e-6
    trial = 0
    while (
        multicomponent_density_constraint_violated(
            xlims=xlims, param_sets=component_parameters
        )
        and trial < 300
    ):
        for i in range(n_components):
            p = list(component_parameters[i])
            p[2] *= 0.95
            component_parameters[i] = tuple(p)
            trial += 1
        if min(p[2] for p in component_parameters) < 1e-6:
            break
    if multicomponent_density_constraint_violated(
        xlims=xlims, param_sets=component_parameters
    ):
        return [[] for _ in range(n_components)]
    return component_parameters


# ══════════════════════════════════════════════
# Multivariate initialization — unified for q=1 and q>1
# ══════════════════════════════════════════════

def _partial_distance_sq(X, centers):
    """(N, K) squared partial-distance from each row of X to each center,
    using only that row's observed dimensions, normalized by how many of
    them there are (so rows with different missingness patterns remain
    comparable). Reduces to ordinary squared Euclidean distance (up to a
    constant per-row scale that doesn't affect argmin) when X has no
    missing values.
    """
    N, p = X.shape
    K = centers.shape[0]
    obs = ~np.isnan(X)
    X_fill = np.where(obs, X, 0.0)
    n_obs = np.maximum(obs.sum(axis=1), 1)
    dist = np.empty((N, K))
    for k in range(K):
        diff = (X_fill - centers[k][None, :]) * obs
        dist[:, k] = (diff ** 2).sum(axis=1) / n_obs
    return dist


def partial_distance_kmeans_labels(X, n_clusters, rng, max_iter=100, n_init=5):
    """Lloyd's-algorithm k-means using partial (available-case) distance.

    Unlike the previous approach (cluster on fully-observed rows only, or --
    when too few of those exist -- replace every missing entry with its
    column's global mean for every row before clustering), this natively
    supports NaN: every pairwise point-to-center distance uses only the
    dimensions that point actually has observed. A highly-missing dimension
    (e.g. TP53's KawOligo, ~1-15% coverage) still contributes proportionally
    to its real information for whichever comparisons happen to have it,
    rather than being either fully used post-hoc or fully erased by
    global-mean imputation once complete rows run out -- the latter was
    empirically shown (tests/cfusn_simulations/sim_kmeans_missingness_init.py)
    to degrade sharply (and in one case produce a ~1e16-scale numerical
    blowup) past ~85-90% missingness, while partial distance degrades
    gracefully.

    Only cluster *centers* get column-mean-filled where a cluster has zero
    real observations in some dimension (needed so later steps have a
    number, not NaN); the distance computation itself is imputation-free.

    Returns (labels, centers) for the best (lowest total partial-distance)
    of n_init random restarts.
    """
    N, p = X.shape
    col_means = np.nanmean(X, axis=0)
    best = None

    for _ in range(n_init):
        idx = rng.choice(N, size=n_clusters, replace=False)
        centers = np.where(np.isnan(X[idx]), col_means[None, :], X[idx])
        labels = None
        for _ in range(max_iter):
            dist = _partial_distance_sq(X, centers)
            new_labels = np.argmin(dist, axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            new_centers = centers.copy()
            for k in range(n_clusters):
                mask = labels == k
                if not mask.any():
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    cm = np.nanmean(X[mask], axis=0)
                missing = np.isnan(cm)
                cm[missing] = col_means[missing]
                new_centers[k] = cm
            centers = new_centers

        dist = _partial_distance_sq(X, centers)
        inertia = float(dist[np.arange(N), labels].sum())
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)

    return best[0], best[1]


def kmeans_init_mv(X, **kwargs):
    """K-means based initialization for multivariate skew-normal mixtures.

    For CFUSN (q > 1), pass latent_q=q in kwargs. Default q=1 (restricted MSN).

    Parameters
    ----------
    X : (N, p) data array, may contain NaN
    **kwargs :
        n_clusters : int
        constrained : bool
        latent_q : int (default 1) — latent dimension q
        lambdaIndex : int — encodes sign patterns for all clusters; for q=2,
            cluster c uses pattern (lambdaIndex // 4**c) % 4, with bits mapping
            to +/-1 per latent direction. Defaults to 0.

    Returns
    -------
    component_parameters : list of (mu, Delta, Gamma) tuples
        Delta is (p,) for q=1, (p, q) for q>1
    kmeans : (labels, centers) from partial_distance_kmeans_labels (stands
        in for the historical fitted-sklearn-KMeans return value; unused
        downstream except as init metadata)
    """
    rng = kwargs.get("rng") or np.random.RandomState()
    n_clusters = kwargs.get("n_clusters", 2)
    constrained = kwargs.get("constrained", True)
    latent_q = kwargs.get("latent_q", 1)
    lambdaIndex = kwargs.get("lambdaIndex", 0)
    n_sign_per_cluster = 2 ** latent_q  # 4 for q=2
    N, K_dim = X.shape

    # Fallback covariance computed once — used when a cluster is too small
    # to estimate its own covariance reliably. Computed directly from the
    # raw (possibly NaN) X via pairwise-observed masking.
    global_cov = np.zeros((K_dim, K_dim))
    for d1 in range(K_dim):
        for d2 in range(d1, K_dim):
            both = ~np.isnan(X[:, d1]) & ~np.isnan(X[:, d2])
            if both.sum() >= 2:
                global_cov[d1, d2] = np.cov(X[both, d1], X[both, d2])[0, 1]
            global_cov[d2, d1] = global_cov[d1, d2]
    global_cov += 1e-6 * np.eye(K_dim)
    _ev = np.linalg.eigvalsh(global_cov)
    if _ev.min() < 1e-8:
        global_cov += (1e-8 - _ev.min()) * np.eye(K_dim)

    last_error = None
    for attempt in range(100):
        try:
            labels, centers = partial_distance_kmeans_labels(X, n_clusters, rng)
            kmeans = (labels, centers)

            component_parameters = []
            for c in range(n_clusters):
                Xc = X[labels == c]
                small_cluster = len(Xc) < max(10, K_dim + 2)

                if len(Xc) > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mu = np.nanmean(Xc, axis=0)
                    # A sparse dimension (e.g. an assay with real coverage on
                    # only a small fraction of variants -- TP53's KawOligo is
                    # ~2% of variants) can easily have ZERO non-missing
                    # entries among the specific rows k-means assigned to
                    # this cluster, even though the cluster itself is
                    # non-empty. np.nanmean silently returns NaN for that
                    # column (hence the warning) -- and a NaN component mean
                    # poisons every downstream EM computation for this
                    # component, not just this initialization. Fall back to
                    # that dimension's GLOBAL mean instead of leaving NaN.
                    missing_dims = np.isnan(mu)
                    if missing_dims.any():
                        global_mu = np.nanmean(X, axis=0)
                        mu[missing_dims] = global_mu[missing_dims]
                else:
                    mu = np.nanmean(X, axis=0)

                if small_cluster:
                    # Too few points for a reliable per-cluster covariance —
                    # use global covariance scaled down so components don't overlap.
                    cov = global_cov / n_clusters
                else:
                    # Compute covariance handling NaN
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

                if latent_q == 1:
                    # Restricted MSN: Delta is (p,) vector
                    Delta = rng.uniform(-0.1, 0.1, size=K_dim) * np.sqrt(np.diag(cov))
                    Gamma = cov - np.outer(Delta, Delta)
                    eigvals_G = np.linalg.eigvalsh(Gamma)
                    if eigvals_G.min() < 1e-8:
                        Gamma += (1e-8 - eigvals_G.min()) * np.eye(K_dim)
                else:
                    # Decode lambdaIndex for cluster c: each cluster uses q bits of
                    # an n_sign_per_cluster-ary digit at position c.
                    cluster_pattern_idx = (lambdaIndex // (n_sign_per_cluster ** c)) % n_sign_per_cluster
                    cluster_sign_pattern = np.array([
                        ((cluster_pattern_idx >> j) & 1) * 2 - 1
                        for j in range(latent_q)
                    ])
                    # CFUSN: Delta is (p, q) matrix.
                    # Pass Xc=None for small clusters so _init_delta_matrix uses
                    # the enumerated sign pattern rather than unreliable skewness.
                    Delta = _init_delta_matrix(cov, K_dim, latent_q,
                                               Xc=None if small_cluster else Xc,
                                               cluster_sign_pattern=cluster_sign_pattern,
                                               rng=rng)
                    Gamma = cov - Delta @ Delta.T
                    Gamma = 0.5 * (Gamma + Gamma.T)
                    eigvals_G = np.linalg.eigvalsh(Gamma)
                    if eigvals_G.min() < 1e-8:
                        Gamma += (1e-8 - eigvals_G.min()) * np.eye(K_dim)

                component_parameters.append((mu, Delta, Gamma))

            component_parameters.sort(key=lambda p: p[0][0])

            if constrained:
                result = fix_to_satisfy_density_constraint_mv(
                    component_parameters, X, **kwargs
                )
                if result is None:
                    last_error = "Constraint failed"
                    continue
                component_parameters = result

            if len(component_parameters) == n_clusters:
                return component_parameters, kmeans

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue

    raise ValueError(f"Failed init after 100 attempts: {last_error}")


def default_anchor_groups(K, S):
    """Sample-index groups whose observations init each component.

    Returns a list of length min(K, S); element c is the list of sample
    column indices to pool when initialising component c.

    Special case: full assay layout (S == 4 → [pathogenic, benign,
    population, synonymous]) with K <= 3 merges synonymous (idx 3) into
    the benign anchor (idx 1). Both represent neutral variants and
    benefit from being modelled by the same component when there are
    fewer components than samples.

    All other (K, S) configurations use 1:1 anchoring.
    """
    if S == 4 and K <= 3:
        return [[0], [1, 3], [2]][:K]
    return [[c] for c in range(min(K, S))]


def _format_vec(x):
    a = np.atleast_1d(np.asarray(x, dtype=float))
    return "[" + ", ".join(f"{v:+.3f}" for v in a) + "]"


def _adaptive_anchor_centroid(
    Xc, mode="contrastive", direction_ref=None,
    bic_delta=6.0, min_n=20, dim_label=None, comp_idx=None, verbose=False, rng=None,
):
    """Pick the anchor centroid μ for one component.

    Modes
    -----
    "mean":         np.nanmean(Xc, axis=0).  Override that disables BIC
                    selection entirely — restores the original anchored-init
                    behaviour.  Use as a debug knob.
    "dominant":     1- vs 2-component Gaussian mixture by BIC.  If 2c wins
                    (ΔBIC > bic_delta), use the dominant cluster's mean;
                    else use the simple mean.
    "contrastive":  default.  Same BIC selection, but if 2c wins AND a
                    well-defined ``direction_ref`` is given, pick the GMM
                    cluster whose mean has the larger projection along
                    direction_ref ("most distinctive of this anchor sample
                    relative to the others") instead of the dominant cluster.
                    Falls back to "dominant" when direction_ref is degenerate.

    Returns ``(mu, label)`` where mu is a (p,) array (1-D inputs are
    reshaped internally and a (1,) array is returned) and label is a
    short string describing the chosen branch (used for debug logging).
    """
    Xc = np.asarray(Xc, dtype=float)
    is_1d = (Xc.ndim == 1)
    if is_1d:
        Xc = Xc.reshape(-1, 1)
    complete = ~np.isnan(Xc).any(axis=1)
    Xf = Xc[complete]
    nanmean = np.nanmean(Xc, axis=0)

    def _log(branch, msg):
        if not verbose:
            return
        tag = f"[anchor-init {dim_label or '?'} comp {comp_idx if comp_idx is not None else '?'}]"
        print(f"  {tag} {branch}: {msg}")

    if mode == "mean":
        _log("mean", f"override → mu={_format_vec(nanmean)}")
        return nanmean, "mean(override)"

    if len(Xf) < min_n:
        _log("mean", f"n_complete={len(Xf)} < min_n={min_n} → mu={_format_vec(nanmean)}")
        return nanmean, "mean(small-n)"

    try:
        from sklearn.mixture import GaussianMixture
        gmm_rng = rng or np.random.RandomState()
        gm1 = GaussianMixture(
            n_components=1, covariance_type="full", reg_covar=1e-4,
            random_state=gmm_rng,
        ).fit(Xf)
        gm2 = GaussianMixture(
            n_components=2, covariance_type="full", reg_covar=1e-4,
            n_init=3, random_state=gmm_rng,
        ).fit(Xf)
    except Exception as e:
        _log("mean", f"GMM fit failed ({e}) → mu={_format_vec(nanmean)}")
        return nanmean, "mean(gmm-fail)"

    bic1, bic2 = float(gm1.bic(Xf)), float(gm2.bic(Xf))
    delta_bic = bic1 - bic2  # positive → 2c better
    if delta_bic <= bic_delta:
        _log(
            "mean",
            f"unimodal (ΔBIC={delta_bic:+.2f} ≤ {bic_delta}) → mu={_format_vec(nanmean)}",
        )
        return nanmean, f"mean(unimodal,ΔBIC={delta_bic:+.2f})"

    means_2 = gm2.means_         # (2, p)
    weights_2 = gm2.weights_     # (2,)

    if mode == "contrastive" and direction_ref is not None:
        d = np.asarray(direction_ref, dtype=float).ravel()
        d_norm = float(np.linalg.norm(d))
        ref_tol = 1e-6 * max(1.0, float(np.linalg.norm(nanmean)))
        if d_norm > ref_tol:
            scores = means_2 @ d
            pick = int(np.argmax(scores))
            picked_mu = means_2[pick]
            _log(
                "contrastive-2cGMM",
                f"bimodal (ΔBIC={delta_bic:+.2f}); pick mode {pick}/2 "
                f"weight={weights_2[pick]:.2f}, projection={scores[pick]:+.3f} "
                f"(other={scores[1 - pick]:+.3f}) → mu={_format_vec(picked_mu)}",
            )
            return picked_mu, (
                f"contrastive-2cGMM(mode={pick},w={weights_2[pick]:.2f},"
                f"ΔBIC={delta_bic:+.2f})"
            )
        _log(
            "contrastive",
            f"direction_ref ‖·‖={d_norm:.2e} ≤ tol={ref_tol:.2e} → falling back to dominant",
        )

    pick = int(np.argmax(weights_2))
    picked_mu = means_2[pick]
    _log(
        "dominant-2cGMM",
        f"bimodal (ΔBIC={delta_bic:+.2f}); dominant mode {pick}/2 "
        f"weight={weights_2[pick]:.2f} → mu={_format_vec(picked_mu)}",
    )
    return picked_mu, f"dominant-2cGMM(mode={pick},w={weights_2[pick]:.2f},ΔBIC={delta_bic:+.2f})"


def _anchor_direction_ref(X, sample_indicators, anchor_groups, c, multivariate):
    """Direction vector pointing from 'other anchored samples' toward 'this
    anchor group's samples'. Returns None if no other anchored samples exist.
    """
    own = set(anchor_groups[c])
    others = set()
    for cc, grp in enumerate(anchor_groups):
        if cc == c:
            continue
        others.update(grp)
    if not others:
        return None

    N = X.shape[0]
    own_mask = np.zeros(N, dtype=bool)
    for s in own:
        own_mask |= sample_indicators[:, s].astype(bool)
    other_mask = np.zeros(N, dtype=bool)
    for s in others:
        other_mask |= sample_indicators[:, s].astype(bool)

    X_own = X[own_mask]
    X_oth = X[other_mask]
    if multivariate:
        X_own = X_own[~np.all(np.isnan(X_own), axis=1)]
        X_oth = X_oth[~np.all(np.isnan(X_oth), axis=1)]
        if len(X_own) == 0 or len(X_oth) == 0:
            return None
        return np.nanmean(X_own, axis=0) - np.nanmean(X_oth, axis=0)
    else:
        X_own = X_own[~np.isnan(X_own)]
        X_oth = X_oth[~np.isnan(X_oth)]
        if len(X_own) == 0 or len(X_oth) == 0:
            return None
        return np.array([float(np.mean(X_own)) - float(np.mean(X_oth))])


def kmeans_init_mv_anchored(X, sample_indicators, **kwargs):
    """Anchored multivariate init: each component is initialised from
    the union of observations across its anchor sample(s).

    Parameters
    ----------
    X : (N, p) ndarray, may contain NaN
    sample_indicators : (N, S) bool ndarray (one-hot per row)
    **kwargs :
        n_clusters : int (K)
        constrained : bool
        latent_q : int (default 1) — latent dimension q
        lambdaIndex : int — encodes per-cluster sign patterns (see
            kmeans_init_mv).
        anchor_groups : list[list[int]] (optional) — override default
            mapping from component index to sample indices.
        anchor_centroid_mode : {"contrastive","dominant","mean"}
            How to derive each component's μ from its anchor obs.  See
            ``_adaptive_anchor_centroid``.  Default "contrastive": detect
            bimodal anchor samples via BIC and pick the mode most
            distinct from the *other* anchor samples (handles cases
            where the anchor sample's larger mode lies in
            wrong-class-territory).
        anchor_centroid_verbose : bool — print which branch chose the μ.

    For component c with group [s_a, s_b, ...], pools
    X[sample_indicators[:, s_a] | sample_indicators[:, s_b] | ...] and
    computes (mu, Delta, Gamma) from that pool. Components with index >=
    min(K, S) are filled in from the global k-means init on remaining
    observations.

    Returns
    -------
    component_parameters : list of (mu, Delta, Gamma) tuples
    kmeans : fitted sklearn KMeans (or None if all components anchored)
    """
    rng = kwargs.get("rng") or np.random.RandomState()
    n_clusters = kwargs.get("n_clusters", 2)
    constrained = kwargs.get("constrained", True)
    latent_q = kwargs.get("latent_q", 1)
    lambdaIndex = kwargs.get("lambdaIndex", 0)
    n_sign_per_cluster = 2 ** latent_q
    anchor_mode = kwargs.get("anchor_centroid_mode", "contrastive")
    anchor_verbose = kwargs.get("anchor_centroid_verbose", False)

    N, K_dim = X.shape
    S = sample_indicators.shape[1]
    anchor_groups = kwargs.get("anchor_groups") or default_anchor_groups(n_clusters, S)

    component_parameters = []
    for c in range(min(n_clusters, len(anchor_groups))):
        group = anchor_groups[c]
        # Pool observations from all samples in this anchor group
        mask = np.zeros(N, dtype=bool)
        for s in group:
            mask |= sample_indicators[:, s].astype(bool)
        Xc = X[mask]
        # Drop rows where every dim is NaN
        Xc = Xc[~np.all(np.isnan(Xc), axis=1)]
        if len(Xc) < max(10, K_dim + 2):
            raise ValueError(
                f"Anchored init: component {c} group {group} has only "
                f"{len(Xc)} usable observations (need at least {max(10, K_dim+2)})"
            )

        direction_ref = (
            _anchor_direction_ref(X, sample_indicators, anchor_groups, c, multivariate=True)
            if anchor_mode == "contrastive" else None
        )
        mu, _branch = _adaptive_anchor_centroid(
            Xc, mode=anchor_mode, direction_ref=direction_ref,
            dim_label="MV", comp_idx=c, verbose=anchor_verbose, rng=rng,
        )

        # NaN-aware pairwise covariance (mirrors kmeans_init_mv)
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

        if latent_q == 1:
            Delta = rng.uniform(-0.1, 0.1, size=K_dim) * np.sqrt(np.diag(cov))
            Gamma = cov - np.outer(Delta, Delta)
            eigvals_G = np.linalg.eigvalsh(Gamma)
            if eigvals_G.min() < 1e-8:
                Gamma += (1e-8 - eigvals_G.min()) * np.eye(K_dim)
        else:
            cluster_pattern_idx = (lambdaIndex // (n_sign_per_cluster ** c)) % n_sign_per_cluster
            cluster_sign_pattern = np.array([
                ((cluster_pattern_idx >> j) & 1) * 2 - 1
                for j in range(latent_q)
            ])
            Delta = _init_delta_matrix(
                cov, K_dim, latent_q, Xc=Xc,
                cluster_sign_pattern=cluster_sign_pattern,
                rng=rng,
            )
            Gamma = cov - Delta @ Delta.T
            Gamma = 0.5 * (Gamma + Gamma.T)
            eigvals_G = np.linalg.eigvalsh(Gamma)
            if eigvals_G.min() < 1e-8:
                Gamma += (1e-8 - eigvals_G.min()) * np.eye(K_dim)

        component_parameters.append((mu, Delta, Gamma))

    # Fill any unanchored components (c >= min(K, S)) via plain kmeans init
    kmeans = None
    if len(component_parameters) < n_clusters:
        n_remaining = n_clusters - len(component_parameters)
        # Reuse kmeans_init_mv on the full data with reduced n_clusters; we'll
        # take its first n_remaining components (sort-by-mu downstream gives
        # a consistent ordering).
        extra_kwargs = {**kwargs, "n_clusters": n_remaining}
        try:
            extra_params, kmeans = kmeans_init_mv(X, **extra_kwargs)
            component_parameters.extend(extra_params[:n_remaining])
        except Exception:
            # If unanchored init fails, raise (caller will catch and retry)
            raise

    # Equalise Γ across components — analog of UV methodOfMomentsInit's
    # ``max_scale`` trick.  Different anchor samples produce different
    # covariance shapes (e.g. P/LP cov from ~50 variants vs gnomAD cov from
    # ~2000 variants).  With unequal scales, the log-density-ratio between
    # any pair of components is quadratic in x — opens upward as x → ±∞ —
    # which is fundamentally non-monotone, no matter how much
    # ``fix_to_satisfy_density_constraint_mv`` shrinks Γ uniformly.  Equal
    # scales make the log-ratio linear → monotone.  Δ is kept per-component
    # (carries the per-anchor skew direction); we only rescale it if needed
    # to keep Γ_common − Δ Δᵀ positive-definite.  Pass
    # ``anchor_equalize_scale=False`` in kwargs to skip.
    if kwargs.get("anchor_equalize_scale", True) and component_parameters:
        Γs = [np.asarray(p[2], dtype=float) for p in component_parameters]
        traces = [float(np.trace(G)) for G in Γs]
        Γ_common = Γs[int(np.argmax(traces))].copy()
        equalised = []
        for mu, Delta, _Γ_old in component_parameters:
            Δ = np.asarray(Delta, dtype=float)
            DDT = np.outer(Δ, Δ) if Δ.ndim == 1 else Δ @ Δ.T
            if np.linalg.eigvalsh(Γ_common - DDT).min() < 1e-8:
                # Binary-search scale s so Γ_common − s²·ΔΔᵀ is PD.
                lo, hi = 0.0, 1.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    test = Γ_common - mid * mid * DDT
                    if np.linalg.eigvalsh(test).min() > 1e-8:
                        lo = mid
                    else:
                        hi = mid
                Δ = Δ * lo
            equalised.append((mu, Δ, Γ_common.copy()))
        component_parameters = equalised

    component_parameters.sort(key=lambda p: p[0][0])

    if constrained:
        result = fix_to_satisfy_density_constraint_mv(
            component_parameters, X, **kwargs
        )
        if result is None:
            raise ValueError("Anchored init: constraint enforcement failed")
        component_parameters = result

    return component_parameters, kmeans


def kmeans_init_anchored(X, sample_indicators, **kwargs):
    """Anchored univariate init: each component initialised from its
    anchor sample(s).

    Mirrors :func:`kmeans_init_mv_anchored`'s structure but produces
    univariate ``(α, loc, scale)`` skew-normal parameter tuples:
      - ``loc`` from the adaptive anchor centroid (BIC-driven; see
        ``_adaptive_anchor_centroid``).
      - ``scale`` from the std of the anchor sample's observations,
        floored to a small positive value.
      - ``α`` (skewness) from sample skewness of the anchor obs, with
        the sign overridden by ``lambdaIndex`` enumeration so that the
        per-bootstrap fit set still covers all ``2^K`` sign patterns.

    Components beyond the anchor groups are filled in from the global
    1-D kmeans init on ``X``.

    Parameters
    ----------
    X : (N,) ndarray
    sample_indicators : (N, S) bool ndarray (one-hot per row)
    **kwargs :
        Same as ``kmeans_init`` plus ``anchor_groups`` /
        ``anchor_centroid_mode`` / ``anchor_centroid_verbose`` /
        ``lambdaIndex`` as in the MV anchored init.
    """
    rng = kwargs.get("rng") or np.random.RandomState()
    n_clusters = kwargs.get("n_clusters", 2)
    constrained = kwargs.get("constrained", True)
    lambdaIndex = kwargs.get("lambdaIndex", 0)
    anchor_mode = kwargs.get("anchor_centroid_mode", "contrastive")
    anchor_verbose = kwargs.get("anchor_centroid_verbose", False)

    X1 = np.asarray(X).ravel()
    N = X1.shape[0]
    S = sample_indicators.shape[1]
    anchor_groups = kwargs.get("anchor_groups") or default_anchor_groups(n_clusters, S)

    LambdasTable = list(itertools.product([-1, 1], repeat=n_clusters))
    lambdas = LambdasTable[lambdaIndex % len(LambdasTable)]

    component_parameters = []
    for c in range(min(n_clusters, len(anchor_groups))):
        group = anchor_groups[c]
        mask = np.zeros(N, dtype=bool)
        for s in group:
            mask |= sample_indicators[:, s].astype(bool)
        Xc = X1[mask]
        Xc = Xc[~np.isnan(Xc)]
        if len(Xc) < 10:
            raise ValueError(
                f"Anchored UV init: component {c} group {group} has only "
                f"{len(Xc)} usable observations (need at least 10)"
            )

        direction_ref = (
            _anchor_direction_ref(X1, sample_indicators, anchor_groups, c, multivariate=False)
            if anchor_mode == "contrastive" else None
        )
        mu_arr, _branch = _adaptive_anchor_centroid(
            Xc, mode=anchor_mode, direction_ref=direction_ref,
            dim_label="UV", comp_idx=c, verbose=anchor_verbose, rng=rng,
        )
        loc = float(np.atleast_1d(mu_arr)[0])

        # (α, scale) from data via skew-normal MoM on the anchor obs.
        # Falls back to empirical std + a small uniform skewness when
        # MoM fails (very small Xc, ~zero skewness, etc.).  We then
        # override α's *sign* with lambdas[c] to preserve the
        # bootstrap-wide 2^K sign-pattern coverage from lambdaIndex —
        # the magnitude stays data-driven.
        force_gaussian = kwargs.get("force_gaussian", False)
        sn_params = sn_method_of_moments_init(Xc, rng=rng) if len(Xc) > 1 else []
        if sn_params and len(sn_params) == 3:
            alpha_data, _loc_unused, scale_data = sn_params
            a_mag = max(abs(float(alpha_data)), 0.05)
            scale = max(float(scale_data), 1e-3)
        else:
            a_mag = float(rng.uniform(0.05, 0.4))
            scale = max(float(np.std(Xc, ddof=1)), 1e-3) if len(Xc) > 1 else 1e-3
        alpha = 0.0 if force_gaussian else float(lambdas[c]) * a_mag

        component_parameters.append((alpha, loc, scale))

    # Fill any unanchored components via plain kmeans init on full X.
    kmeans = None
    if len(component_parameters) < n_clusters:
        n_remaining = n_clusters - len(component_parameters)
        extra_kwargs = {**kwargs, "n_clusters": n_remaining,
                        "constrained": False}  # we'll constraint-fix below
        try:
            extra_params, kmeans = kmeans_init(X1, **extra_kwargs)
            component_parameters.extend(extra_params[:n_remaining])
        except Exception:
            raise

    # Equalise scales to the max — mirrors methodOfMomentsInit. The
    # density constraint is much easier to satisfy when components
    # share a scale (otherwise narrow P/LP scale + wide gnomAD scale
    # produces overlapping densities that the constraint fixer rejects).
    if component_parameters:
        max_scale = max(p[2] for p in component_parameters)
        component_parameters = [
            (p[0], p[1], max_scale) for p in component_parameters
        ]
    component_parameters = sorted(component_parameters, key=lambda p: p[1])

    if constrained:
        component_parameters = fix_to_satisfy_density_constraint(
            component_parameters,
            (float(X1.min()), float(X1.max())),
            **kwargs,
        )
        if (
            len(component_parameters) == 0
            or any(len(p) == 0 for p in component_parameters)
        ):
            raise ValueError("Anchored UV init: constraint enforcement failed")

    return component_parameters, kmeans


def _init_delta_matrix(cov, p, q, Xc=None, cluster_sign_pattern=None, rng=None):
    """Initialize Delta (p, q) matrix for CFUSN.

    Uses the top-q eigenvectors of cov as skewness directions. Each column j
    gets a sign determined by:
      1. Base sign: sign of marginal skewness of cluster data projected onto
         eigenvector j (uses complete rows of Xc; falls back to +1 if too few).
      2. Enumerated flip: multiplied by cluster_sign_pattern[j] ∈ {-1, +1},
         which comes from decoding lambdaIndex in the caller.

    If cluster_sign_pattern is None, a random sign is used (original behaviour).
    """
    rng = rng or np.random.RandomState()
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_idx = np.argsort(eigvals)[::-1][:q]

    Delta = np.zeros((p, q))
    for j, idx in enumerate(top_idx):
        scale = 0.1 * np.sqrt(eigvals[idx])
        evec = eigvecs[:, idx]

        # Base sign from marginal skewness of cluster data along this eigenvector
        skew_sign = 1
        if Xc is not None:
            complete_rows = ~np.isnan(Xc).any(axis=1)
            Xc_comp = Xc[complete_rows]
            if len(Xc_comp) >= 8:
                sk = sps.skew(Xc_comp @ evec)
                if abs(sk) > 1e-6:
                    skew_sign = int(np.sign(sk))

        # Enumerated flip from lambdaIndex (or random if not provided)
        enum_sign = (
            int(cluster_sign_pattern[j])
            if cluster_sign_pattern is not None
            else rng.choice([-1, 1])
        )
        Delta[:, j] = skew_sign * enum_sign * scale * evec

    # Add small random perturbation
    Delta += rng.uniform(-0.05, 0.05, size=(p, q)) * np.sqrt(np.diag(cov))[:, None]

    # Verify Gamma = cov - Delta @ Delta.T is PD; shrink Delta if needed.
    # Fixed per-retry decay (not compounding -- `shrink *= 0.5` here used to
    # make the cumulative multiplier 0.5*0.25*0.125*... i.e. doubly-
    # exponential, annihilating Delta to ~1e-60 within a handful of retries
    # whenever the first attempt or two didn't succeed. Worst case now:
    # 0.9**20 ~= 0.12, so Delta always retains a meaningful fraction of its
    # original scale even after the full retry budget.
    Gamma = cov - Delta @ Delta.T
    eigvals_G = np.linalg.eigvalsh(Gamma)
    if eigvals_G.min() < 1e-6:
        for _ in range(20):
            Delta *= 0.9
            Gamma = cov - Delta @ Delta.T
            if np.linalg.eigvalsh(Gamma).min() > 1e-6:
                break

    return Delta


def fix_to_satisfy_density_constraint_mv(component_parameters, X, **kwargs):
    """Shrink scale matrices until the (legacy) density constraint is satisfied.

    Only applies to the deprecated 'line'/'marginal' density-ratio modes. The
    current separation features (tempering + repulsion) induce non-overlap
    during EM, not at initialization, so this is a no-op for them — shrinking
    the init scales would only hurt the starting basin.
    """
    constraint_mode = kwargs.get("constraint_mode", separation.DEFAULT_CONSTRAINT_MODE)
    if not separation.is_legacy_mode(constraint_mode):
        return component_parameters

    n_components = len(component_parameters)
    K_dim = component_parameters[0][0].shape[0]

    xlims = tuple(
        (float(np.nanmin(X[:, d])), float(np.nanmax(X[:, d])))
        for d in range(K_dim)
    )

    trial = 0
    while trial < 50:
        if not multicomponent_density_constraint_violated(
            component_parameters, xlims, multivariate=True, mode=constraint_mode,
        ):
            return component_parameters
        # Shrink: scale Gamma and Delta toward zero
        shrunk = []
        for mu, Delta, Gamma in component_parameters:
            Delta_shrunk = Delta * 0.95
            Gamma_shrunk = Gamma * 0.95 + 0.05 * np.eye(K_dim) * np.trace(Gamma) / K_dim
            shrunk.append((mu, Delta_shrunk, Gamma_shrunk))
        component_parameters = shrunk
        trial += 1

    if multicomponent_density_constraint_violated(
        component_parameters, xlims, multivariate=True, mode=constraint_mode,
    ):
        return None
    return component_parameters