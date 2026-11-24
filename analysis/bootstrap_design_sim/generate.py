"""Bootstrap-design simulation (Arm A): synthetic multivariate design grid.

Generative model, D readouts split into functional (first ceil(D/2)) and
predictor (rest) dimensions, all with unit variance and within-class
correlation rho (Sigma = (1 - rho) I + rho 11'):
    benign                       N(0, Sigma)
    pathogenic, measured         N(-kappa 1, Sigma)   (a fraction gamma)
    pathogenic, unmeasured       N(0, Sigma)
so the true LR of an observed sub-vector x_S is
    [gamma N(x_S; -kappa 1_S, Sigma_SS) + (1 - gamma) N(x_S; 0, Sigma_SS)] / N(x_S; 0, Sigma_SS).

Strata (consequence):
  P/LP missense           pathogenic mechanism mix gamma; every readout available
  P/LP PTV (share ptv)    always "measured" (null) on functional readouts; no predictor scores
  B/LB missense           benign
  population missense     pathogenic with prob pi (mechanism mix gamma), else benign
  synonymous              benign, functional readouts only

Missingness of the functional readouts (predictors are always observed for
missense):
  mcar     each functional readout observed with probability 0.7
  region   each variant has a position u ~ U(0,1) -- pathogenic ones from
           Beta(4, 4), concentrated mid-protein -- and functional readout j
           covers its own interval, so coverage is design-driven and
           correlated with pathogenicity
  scarce   functional readout 0 observed for 7% of variants, the others 0.8

Cells: one factor at a time from the baseline (D 4, region, sparse, kappa
2.5, gamma 1, rho 0.5, PTV share 0.4, pi 0.25). Every cell has a stable id;
replicate r draws from default_rng([crc32(cell id), r]), as
analysis/calibration_quality/generate.py.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy.stats import multivariate_normal as mvn

SIZES = {"rich": (80, 80, 1000, 500), "sparse": (20, 20, 200, 100), "very_sparse": (5, 5, 50, 0)}
BASE = dict(D=4, missing="region", size="sparse", kappa=2.5, gamma=1.0, rho=0.5, ptv=0.4, pi=0.25)
SLICES = dict(D=(2, 4, 8), missing=("mcar", "region", "scarce"), size=("rich", "sparse", "very_sparse"),
              kappa=(1.5, 2.5, 4.0), gamma=(0.25, 1.0), rho=(0.0, 0.5), ptv=(0.0, 0.4, 0.7))
N_EVAL = 2000  # held-out draws per class (missense), for oracle comparisons


def cell_id(c):
    return (f"D{c['D']}_{c['missing']}_{c['size']}_k{c['kappa']:g}_g{c['gamma']:g}_r{c['rho']:g}"
            f"_ptv{c['ptv']:g}_pi{c['pi']:g}")


def design():
    cells = {}
    for factor, levels in SLICES.items():
        for lv in levels:
            c = dict(BASE, **{factor: lv})
            c["cell_id"] = cell_id(c)
            cells[c["cell_id"]] = c
    return list(cells.values())


def rng_for(cell, rep):
    return np.random.default_rng([zlib.crc32(cell["cell_id"].encode()), int(rep)])


def _sigma(c):
    D, rho = c["D"], c["rho"]
    return (1 - rho) * np.eye(D) + rho * np.ones((D, D))


def n_functional(c):
    return int(np.ceil(c["D"] / 2))


def _draw(rng, n, c, path_frac, mech_gamma, force_measured=False):
    S = _sigma(c)
    is_path = rng.random(n) < path_frac
    measured = is_path & (force_measured | (rng.random(n) < mech_gamma))
    X = rng.multivariate_normal(np.zeros(c["D"]), S, size=n)
    X[measured] -= c["kappa"]
    return X, is_path


def _mask(rng, n, c, is_path, functional_only=False):
    D, F = c["D"], n_functional(c)
    obs = np.ones((n, D), dtype=bool)
    if c["missing"] == "mcar":
        obs[:, :F] = rng.random((n, F)) < 0.7
    elif c["missing"] == "scarce":
        obs[:, 0] = rng.random(n) < 0.07
        obs[:, 1:F] = rng.random((n, F - 1)) < 0.8
    else:  # region
        u = np.where(is_path, rng.beta(4, 4, n), rng.random(n))
        for j in range(F):
            a = j / (F + 1) * 0.6
            obs[:, j] = (u >= a) & (u <= a + 0.6)
    if functional_only:
        obs[:, F:] = False
    empty = ~obs.any(axis=1)
    obs[empty, 0] = True  # every row observed somewhere
    return obs


def make_data(rng, c):
    """Rows of every sample with their stratum, plus held-out missense draws.
    Returns dict(X, stratum, members (N, 4) bool over P/LP, B/LB, population,
    synonymous), eval_X, eval_y, consequence)."""
    n_p, n_b, n_g, n_s = SIZES[c["size"]]
    n_ptv = int(round(n_p * c["ptv"]))
    n_pm = n_p - n_ptv
    parts = []
    X, ip = _draw(rng, n_pm, c, 1.0, c["gamma"])
    parts.append((X, _mask(rng, n_pm, c, ip), (1, 0, 0, 0), "missense"))
    X, ip = _draw(rng, n_ptv, c, 1.0, 1.0, force_measured=True)
    parts.append((X, _mask(rng, n_ptv, c, ip, functional_only=True), (1, 0, 0, 0), "PTV"))
    X, ip = _draw(rng, n_b, c, 0.0, c["gamma"])
    parts.append((X, _mask(rng, n_b, c, ip), (0, 1, 0, 0), "missense"))
    X, ip = _draw(rng, n_g, c, c["pi"], c["gamma"])
    parts.append((X, _mask(rng, n_g, c, ip), (0, 0, 1, 0), "missense"))
    if n_s:
        X, ip = _draw(rng, n_s, c, 0.0, c["gamma"])
        parts.append((X, _mask(rng, n_s, c, ip, functional_only=True), (0, 0, 0, 1), "synonymous"))
    Xs, M, csq = [], [], []
    for X, obs, memb, k in parts:
        X = X.copy()
        X[~obs] = np.nan
        Xs.append(X)
        M.append(np.tile(np.array(memb, bool), (len(X), 1)))
        csq += [k] * len(X)
    out = dict(X=np.vstack(Xs), members=np.vstack(M), consequence=np.array(csq))
    # held-out missense: pathogenic (mix gamma) and benign, with the design's missingness
    Xp, ip = _draw(rng, N_EVAL, c, 1.0, c["gamma"])
    Xb, ib = _draw(rng, N_EVAL, c, 0.0, c["gamma"])
    op, ob = _mask(rng, N_EVAL, c, ip), _mask(rng, N_EVAL, c, ib)
    Xp[~op] = np.nan
    Xb[~ob] = np.nan
    out["eval_X"] = np.vstack([Xp, Xb])
    out["eval_y"] = np.r_[np.ones(N_EVAL, int), np.zeros(N_EVAL, int)]
    return out


def true_log_lr(X, c):
    """Closed-form log LR (missense pathogenic vs benign) on each row's
    observed readouts."""
    S = _sigma(c)
    out = np.full(len(X), np.nan)
    obs = ~np.isnan(X)
    pats = {}
    for i, o in enumerate(obs):
        pats.setdefault(tuple(o), []).append(i)
    for pat, idx in pats.items():
        d = np.flatnonzero(pat)
        if len(d) == 0:
            continue
        Xs = X[np.ix_(idx, d)]
        Sd = S[np.ix_(d, d)]
        l0 = mvn(np.zeros(len(d)), Sd).logpdf(Xs)
        l1m = mvn(-c["kappa"] * np.ones(len(d)), Sd).logpdf(Xs)
        g = c["gamma"]
        l1 = np.logaddexp(np.log(g) + l1m, np.log1p(-g) + l0) if g < 1 else l1m
        out[idx] = np.atleast_1d(l1 - l0)
    return out
