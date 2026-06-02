"""Focused follow-up: at HIGH overlap, does a gentler tempering beta avoid the
ARI collapse? Decides whether combined-with-low-beta can be a safe default."""
import warnings
import numpy as np
from sklearn.metrics import adjusted_rand_score
import sys
sys.path.append('..')
from src.assay_calibration.fit_utils.cfusn.fit import single_fit
from src.assay_calibration.fit_utils.cfusn import density_utils as D


def sample_cfusn(mu, Delta, Gamma, n):
    T = np.abs(np.random.randn(n, Delta.shape[1]))
    L = np.linalg.cholesky(Gamma)
    return mu + T @ Delta.T + np.random.randn(n, Delta.shape[0]) @ L.T


def pad(Dl):
    Dl = np.atleast_2d(Dl)
    if Dl.shape[1] == 1:
        out = np.zeros((Dl.shape[0], 2)); out[:, 0] = Dl[:, 0]; return out
    return Dl


BASE = [(np.array([-1.3, -0.5]), pad(np.array([[0.7], [0.2]]))),
        (np.array([0.0, 1.3]),   np.array([[0.2, 0.5], [0.5, 0.1]])),
        (np.array([1.3, -0.5]),  pad(np.array([[-0.6], [0.3]])))]
GAMMA = 0.7 * np.eye(2)
N_PER = 130
SEP = 0.9   # high overlap


def make_data(seed):
    np.random.seed(seed)
    Xs, ys = [], []
    for k, (mu, Dl) in enumerate(BASE):
        Xs.append(sample_cfusn(mu * SEP, Dl, GAMMA, N_PER)); ys.append(np.full(N_PER, k))
    return np.vstack(Xs), np.concatenate(ys)


COMMON = dict(multivariate=True, latent_q=2, verbose=False,
              max_em_iters=70, n_mc_truncated=100, raise_on_error=False)


def evaluate(res, X, y):
    if res["likelihoods"][-1] == -np.inf or not res["component_params"][0]:
        return np.nan, np.nan
    P = D.component_posteriors(X, res["component_params"], res["weights"][0], multivariate=True)
    ari = adjusted_rand_score(y, P.argmax(0))
    contested = float(np.mean(np.sort(P, axis=0)[-2, :] > 0.15))
    return ari, contested


def run(con, cfg, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = single_fit(X, np.ones((X.shape[0], 1), dtype=int), 3, con, "kmeans", "scale",
                       **COMMON, **cfg)
    return evaluate(r, X, y)


data = [make_data(s) for s in range(5)]


def agg(rows):
    return np.nanmean([r[0] for r in rows]), np.nanmean([r[1] for r in rows])


print("HIGH overlap (sep=0.9), 5 seeds — ARI / contested")
print("baseline        ", "%5.2f / %5.2f" % agg([run(False, {}, X, y) for X, y in data]))
print("repulsion lr=.10", "%5.2f / %5.2f" % agg([run(True, {"constraint_mode": "repulsion"}, X, y) for X, y in data]))
for b in [0.1, 0.2, 0.3, 0.5]:
    a, c = agg([run(True, {"constraint_mode": "tempering", "tempering_beta_max": b}, X, y) for X, y in data])
    print(f"tempering b={b:<4} ", "%5.2f / %5.2f" % (a, c))
for b in [0.1, 0.2, 0.3]:
    a, c = agg([run(True, {"constraint_mode": "separation", "tempering_beta_max": b}, X, y) for X, y in data])
    print(f"combined  b={b:<4} ", "%5.2f / %5.2f" % (a, c))
