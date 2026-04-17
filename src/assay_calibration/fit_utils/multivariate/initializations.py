from .constraints import (
    density_constraint_violated,
    multicomponent_density_constraint_violated,
)
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats as sps
import itertools


# ══════════════════════════════════════════════
# Univariate initializations (unchanged)
# ══════════════════════════════════════════════

def kmeans_init(X, **kwargs):
    repeat = 0
    while repeat < 1000:
        n_clusters = kwargs.get("n_clusters", 2)
        init = kwargs.get("kmeans_init", "random")
        kmeans = KMeans(n_clusters=n_clusters, init=init)
        X = np.array(X).reshape((-1, 1))
        kmeans.fit(X)
        cluster_assignments = kmeans.predict(X)
        component_parameters = []
        for i in range(n_clusters):
            X_cluster = X[cluster_assignments == i]
            loc, scale = sps.norm.fit(X_cluster)
            a = np.random.uniform(-0.25, 0.25)
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


def sn_method_of_moments_init(X):
    m1 = np.mean(X)
    m2 = np.var(X)
    m3 = sps.skew(X)
    if m2 < 1e-10:
        return []
    a1 = np.sqrt(2 / np.pi)
    c = (4 - np.pi) / 2
    try:
        if np.abs(m3) < 1e-10:
            return np.random.uniform(-0.25, 0.25), m1, max(np.sqrt(m2), 1e-6)
        delta = np.sign(m3) / np.sqrt(a1**2 * (1 + (c / np.abs(m3)) ** (2/3)))
        if np.isnan(delta) or np.abs(delta) >= 0.99:
            return np.random.uniform(-0.25, 0.25), m1, max(np.sqrt(m2), 1e-6)
        denom = 1 - a1**2 * delta**2
        if denom <= 1e-10:
            return np.random.uniform(-0.25, 0.25), m1, max(np.sqrt(m2), 1e-6)
        sigma = max(np.sqrt(m2 / denom), 1e-6)
        mu = m1 - a1 * delta * sigma
        alpha = delta / np.sqrt(max(1 - delta**2, 1e-12))
        if np.any(np.isnan([mu, sigma, alpha])) or np.any(np.isinf([mu, sigma, alpha])):
            return []
        return alpha, mu, sigma
    except (ZeroDivisionError, RuntimeWarning):
        return []


def methodOfMomentsInit(X, n_components, constrained, max_attempts=1000, **kwargs):
    LambdasTable = list(itertools.product([-1, 1], repeat=n_components))
    if "lambdaIndex" in kwargs:
        lambdas = LambdasTable[kwargs["lambdaIndex"]]

    for attempt in range(max_attempts):
        if np.random.rand() < 0.7:
            base = np.linspace(0, 100, n_components + 1)[1:-1]
            iqr = np.percentile(X, 75) - np.percentile(X, 25)
            jitter = np.random.normal(0, iqr * 0.1, len(base))
            cutPoints = np.percentile(X, np.sort(np.clip(base + jitter, 1, 99)))
        else:
            cutPoints = np.sort(
                np.random.uniform(
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
            params = sn_method_of_moments_init(Xc)
            if len(params) == 0:
                success = False
                break
            params = list(params)
            params[0] = lambdas[i]
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
# Multivariate initialization
# ══════════════════════════════════════════════

def kmeans_init_mv(X, **kwargs):
    n_clusters = kwargs.get("n_clusters", 2)
    constrained = kwargs.get("constrained", True)
    N, K_dim = X.shape

    complete_mask = ~np.isnan(X).any(axis=1)
    X_complete = X[complete_mask]
    min_needed = n_clusters * max(10, K_dim + 2)
    if len(X_complete) < min_needed:
        raise ValueError(f"Only {len(X_complete)}/{N} complete rows, need {min_needed}")

    last_error = None
    for attempt in range(1000):
        try:
            kmeans = KMeans(n_clusters=n_clusters, init=kwargs.get("kmeans_init","random"), n_init=1)
            kmeans.fit(X_complete)
            labels = np.full(N, -1, dtype=int)
            labels[complete_mask] = kmeans.predict(X_complete)
            centers = kmeans.cluster_centers_
            for j in np.where(~complete_mask)[0]:
                obs = ~np.isnan(X[j])
                if not obs.any():
                    labels[j] = np.random.randint(n_clusters)
                else:
                    labels[j] = np.argmin([np.sum((X[j,obs]-centers[c,obs])**2) for c in range(n_clusters)])

            component_parameters = []
            ok = True
            for c in range(n_clusters):
                Xc = X[labels == c]
                if len(Xc) < max(10, K_dim+2):
                    ok = False; last_error = f"Cluster {c}: {len(Xc)} pts"; break
                mu = np.nanmean(Xc, axis=0)
                cov = np.zeros((K_dim, K_dim))
                for d1 in range(K_dim):
                    for d2 in range(d1, K_dim):
                        both = ~np.isnan(Xc[:,d1]) & ~np.isnan(Xc[:,d2])
                        if both.sum() < 2:
                            cov[d1,d2] = 1e-2
                        else:
                            cov[d1,d2] = np.cov(Xc[both,d1], Xc[both,d2])[0,1]
                        cov[d2,d1] = cov[d1,d2]
                cov += 1e-6 * np.eye(K_dim)
                eigvals = np.linalg.eigvalsh(cov)
                if eigvals.min() < 1e-8:
                    cov += (1e-8 - eigvals.min()) * np.eye(K_dim)
                Delta = np.random.uniform(-0.1, 0.1, size=K_dim) * np.sqrt(np.diag(cov))
                Gamma = cov - np.outer(Delta, Delta)
                eigvals = np.linalg.eigvalsh(Gamma)
                if eigvals.min() < 1e-8:
                    Gamma += (1e-8 - eigvals.min()) * np.eye(K_dim)
                component_parameters.append((mu, Delta, Gamma))
            if not ok:
                continue
            component_parameters.sort(key=lambda p: p[0][0])
            if constrained:
                result = fix_to_satisfy_density_constraint_mv(component_parameters, X, **kwargs)
                if result is None:
                    last_error = "Constraint failed"; continue
                component_parameters = result
            if len(component_parameters) == n_clusters:
                return component_parameters, kmeans
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"; continue
    raise ValueError(f"Failed init after 1000 attempts: {last_error}")


def fix_to_satisfy_density_constraint_mv(component_parameters, X, **kwargs):
    """Shrink scale matrices until constraint is satisfied.
    component_parameters: list of (mu, Delta, Gamma) in alternate form.
    X: (N, K_dim) data (may contain NaN), used to compute xlims.
    """
    n_components = len(component_parameters)
    K_dim = component_parameters[0][0].shape[0]

    xlims = tuple(
        (float(np.nanmin(X[:, d])), float(np.nanmax(X[:, d])))
        for d in range(K_dim)
    )

    trial = 0
    while trial < 300:
        if not multicomponent_density_constraint_violated(
            component_parameters, xlims, multivariate=True
        ):
            return component_parameters
        # Shrink: scale Gamma and Delta toward zero
        shrunk = []
        for mu, Delta, Gamma in component_parameters:
            shrunk.append((mu, Delta * 0.95, Gamma * 0.95 + 0.05 * np.eye(K_dim) * np.trace(Gamma) / K_dim))
        component_parameters = shrunk
        trial += 1

    # Check if final satisfies
    if multicomponent_density_constraint_violated(
        component_parameters, xlims, multivariate=True
    ):
        return None
    return component_parameters