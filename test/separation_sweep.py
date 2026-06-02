"""Bounded experiment: tempering (1) vs repulsion (2) vs combined (1+2),
across overlap levels, plus knob sweeps — to choose the default.

Metrics (paired across methods per (overlap, seed)):
  ARI       : adjusted Rand index of argmax-responsibility vs true labels (↑)
  contested : fraction of points with top-2 responsibilities both > 0.15 (↓)
"""
import warnings
import numpy as np
from sklearn.metrics import adjusted_rand_score
import sys
sys.path.append('..')
from src.assay_calibration.fit_utils.cfusn.fit import single_fit
from src.assay_calibration.fit_utils.cfusn import density_utils as D
from src.assay_calibration.fit_utils.cfusn import separation as S


def sample_cfusn(mu, Delta, Gamma, n):
    T = np.abs(np.random.randn(n, Delta.shape[1]))
    L = np.linalg.cholesky(Gamma)
    return mu + T @ Delta.T + np.random.randn(n, Delta.shape[0]) @ L.T


def pad(Dl):
    Dl = np.atleast_2d(Dl)
    if Dl.shape[1] == 1:
        out = np.zeros((Dl.shape[0], 2)); out[:, 0] = Dl[:, 0]; return out
    return Dl


# Base skewed 3-component geometry; `sep` scales the centers (smaller = more overlap).
BASE = [(np.array([-1.3, -0.5]), pad(np.array([[0.7], [0.2]]))),
        (np.array([0.0, 1.3]),   np.array([[0.2, 0.5], [0.5, 0.1]])),
        (np.array([1.3, -0.5]),  pad(np.array([[-0.6], [0.3]])))]
GAMMA = 0.7 * np.eye(2)
N_PER = 130


def make_data(sep, seed):
    np.random.seed(seed)
    Xs, ys = [], []
    for k, (mu, Dl) in enumerate(BASE):
        Xs.append(sample_cfusn(mu * sep, Dl, GAMMA, N_PER))
        ys.append(np.full(N_PER, k))
    return np.vstack(Xs), np.concatenate(ys)


COMMON = dict(multivariate=True, latent_q=2, verbose=False,
              max_em_iters=70, n_mc_truncated=100, raise_on_error=False)


def evaluate(res, X, y):
    if res["likelihoods"][-1] == -np.inf or not res["component_params"][0]:
        return np.nan, np.nan
    P = D.component_posteriors(X, res["component_params"], res["weights"][0],
                               multivariate=True)  # (K, N)
    pred = P.argmax(0)
    ari = adjusted_rand_score(y, pred)
    Ps = np.sort(P, axis=0)
    contested = float(np.mean(Ps[-2, :] > 0.15))
    return ari, contested


def run(con, cfg, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = single_fit(X, np.ones((X.shape[0], 1), dtype=int), 3, con,
                       "kmeans", "scale", **COMMON, **cfg)
    return evaluate(r, X, y)


def agg(rows):
    a = np.array([r[0] for r in rows]); c = np.array([r[1] for r in rows])
    return np.nanmean(a), np.nanmean(c)


SEEDS = range(5)
OVERLAP = [("high", 0.9), ("med", 1.4), ("low", 2.2)]

METHODS = {
    "baseline":          (False, {}),
    "tempering (1)":     (True, {"constraint_mode": "tempering"}),
    "repulsion (2)":     (True, {"constraint_mode": "repulsion"}),
    "combined (1+2)":    (True, {"constraint_mode": "separation"}),
}

print("=" * 64)
print("MAIN: methods x overlap   (ARI↑ / contested↓, mean over 5 seeds)")
print("=" * 64)
print(f"{'method':18s} " + "  ".join(f"{lbl:>14s}" for lbl, _ in OVERLAP))
# Pre-generate data once per (overlap, seed) for pairing.
data = {(lbl, s): make_data(sep, s) for lbl, sep in OVERLAP for s in SEEDS}
for m, (con, cfg) in METHODS.items():
    cells = []
    for lbl, _ in OVERLAP:
        rows = [run(con, cfg, *data[(lbl, s)]) for s in SEEDS]
        cells.append(agg(rows))
    print(f"{m:18s} " + "  ".join(f"{a:5.2f}/{c:5.2f}" for a, c in cells))

print("\n" + "=" * 64)
print("KNOB SWEEP @ medium overlap (ARI↑ / contested↓)")
print("=" * 64)
med = [data[("med", s)] for s in SEEDS]

print("-- tempering(1): beta_max --")
for b in [0.3, 0.5, 1.0]:
    rows = [run(True, {"constraint_mode": "tempering", "tempering_beta_max": b}, X, y)
            for X, y in med]
    a, c = agg(rows); print(f"  beta={b:<4}  ARI={a:5.2f}  contested={c:5.2f}")

print("-- combined(1+2): beta_max x repulsion_lr --")
for b in [0.3, 0.5, 1.0]:
    for lr in [0.05, 0.15]:
        rows = [run(True, {"constraint_mode": "separation",
                           "tempering_beta_max": b, "repulsion_lr": lr}, X, y)
                for X, y in med]
        a, c = agg(rows)
        print(f"  beta={b:<4} lr={lr:<5}  ARI={a:5.2f}  contested={c:5.2f}")

print("-- repulsion(2): repulsion_lr --")
for lr in [0.05, 0.15, 0.3]:
    rows = [run(True, {"constraint_mode": "repulsion", "repulsion_lr": lr}, X, y)
            for X, y in med]
    a, c = agg(rows); print(f"  lr={lr:<5}  ARI={a:5.2f}  contested={c:5.2f}")
