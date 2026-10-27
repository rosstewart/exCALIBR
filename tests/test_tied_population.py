"""Tied population weights (cfusn/tied_population.py) and the Kish ridge count."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.assay_calibration.fit_utils.cfusn import tied_population as tp  # noqa: E402
from src.assay_calibration.fit_utils.cfusn.fit import single_fit  # noqa: E402
from src.assay_calibration.fit_utils.cfusn.update_steps import _component_n_eff  # noqa: E402

SPEC = {"primary": 0, "benign": [1, 3], "population": 2}


def _data(seed=0, pi=0.2):
    rng = np.random.default_rng(seed)
    path = lambda n: rng.normal([-3, -3], 0.6, (n, 2))
    ben = lambda n: rng.normal([0, 0], 0.8, (n, 2))
    n_pop = 400
    k = rng.binomial(n_pop, pi)
    X = np.vstack([path(12), ben(80), np.vstack([path(k), ben(n_pop - k)]), ben(150)])
    counts = [12, 80, n_pop, 150]
    S = np.zeros((len(X), 4), bool)
    S[np.arange(len(X)), np.repeat(np.arange(4), counts)] = True
    return X, S, k / n_pop


def test_tie_and_pi_roundtrip():
    W = np.array([[0.9, 0.1], [0.1, 0.9], [0.3, 0.7], [0.2, 0.8]])
    T = tp.tie(W, SPEC)
    assert np.allclose(T[1], T[3])
    pi = tp.estimate_pi(T, SPEC)
    assert np.allclose(T[2], pi * T[0] + (1 - pi) * T[1])


def test_update_keeps_tied_form_and_rows_normalised():
    rng = np.random.default_rng(1)
    W = tp.tie(rng.dirichlet(np.ones(3), 4), SPEC)
    nbar = rng.uniform(1, 50, (4, 3))
    U = tp.tied_weight_update(nbar, W, SPEC)
    assert np.allclose(U.sum(1), 1)
    pi = tp.estimate_pi(U, SPEC)
    assert np.allclose(U[2], pi * U[0] + (1 - pi) * U[1]) and np.allclose(U[1], U[3])


def test_spec_from_roles():
    assert tp.spec_from_roles(["primary", "benign", "population", "synonymous"]) == SPEC
    assert tp.spec_from_roles(["benign", "population", "synonymous"]) is None  # no P/LP
    assert tp.spec_from_roles(["primary", "population"]) is None  # no benign reference


def test_tied_fit_recovers_pi():
    # Well-separated data already satisfy containment, so the tied fit should
    # match the free one and recover the simulated pi. (The likelihood trace is
    # not the EM objective -- the Gamma ridge's prior is added to it -- so
    # monotonicity is not asserted on it here.)
    X, S, true_pi = _data()
    kw = dict(multivariate=True, latent_q=1, verbose=False, max_em_iters=3000)
    tied = single_fit(X, S, 2, False, "kmeans", False, tied_population=SPEC, rng=np.random.default_rng(0), **kw)
    free = single_fit(X, S, 2, False, "kmeans", False, rng=np.random.default_rng(0), **kw)
    W = np.asarray(tied["weights"])
    pi = tp.estimate_pi(W, SPEC)
    assert np.allclose(W[2], pi * W[0] + (1 - pi) * W[1], atol=1e-8)
    assert abs(pi - true_pi) < 0.05
    assert abs(tied["likelihoods"][-1] - free["likelihoods"][-1]) < 1e-3


def test_kish_neff():
    z = np.array([1, 1, 1, 1, 1, 1, 1, .5, .2])
    w = np.array([6.64] * 7 + [1, 1])
    assert np.isclose(_component_n_eff(z, w, {}), (z * w).sum())
    assert 7 < _component_n_eff(z, w, {"weight_neff": "kish"}) < 7.7
    assert np.isclose(_component_n_eff(z, np.ones(9), {"weight_neff": "kish"}), z.sum())


def test_noise_floor_clips_only_below_noise():
    from src.assay_calibration.fit_utils.cfusn.update_steps import _floor_gamma_at_noise
    noise = np.array([0.04, 0.09])
    wide = np.array([[0.5, 0.1], [0.1, 0.4]])
    assert np.allclose(_floor_gamma_at_noise(wide, noise), wide)          # already above: unchanged
    spike = np.array([[1e-4, 0.0], [0.0, 1e-4]])
    F = _floor_gamma_at_noise(spike, noise)
    assert np.all(np.linalg.eigvalsh(F / np.sqrt(np.outer(noise, noise))) >= 1 - 1e-9)
    assert _floor_gamma_at_noise(spike, None) is spike


def test_noise_floor_partial_readouts():
    from src.assay_calibration.fit_utils.cfusn.update_steps import _floor_gamma_at_noise
    spike = np.array([[1e-4, 2e-5], [2e-5, 1e-4]])
    F = _floor_gamma_at_noise(spike, np.array([0.04, np.nan]))
    cond = lambda G: (G[1, 0] / G[0, 0], G[1, 1] - G[1, 0] ** 2 / G[0, 0])  # readout 1 | readout 0
    assert F[0, 0] >= 0.04 - 1e-12 and np.allclose(cond(F), cond(spike))  # only readout 0 floored
    assert np.all(np.linalg.eigvalsh(F) > 0)                            # stays PSD
    # constrained maximiser of q(G) = -log|G| - tr(G^-1 A): never below any
    # other feasible G (here: the old block-add answer and random feasible ones)
    rng = np.random.default_rng(0)
    A = np.array([[0.02, 0.015, 0.01], [0.015, 0.03, 0.02], [0.01, 0.02, 0.5]])
    noise = np.array([0.04, 0.06, np.nan])
    q = lambda G: -np.linalg.slogdet(G)[1] - np.trace(np.linalg.solve(G, A))
    best = _floor_gamma_at_noise(A, noise)
    Li = np.linalg.inv(np.linalg.cholesky(np.diag(noise[:2])))
    assert np.linalg.eigvalsh(Li @ best[:2, :2] @ Li.T).min() >= 1 - 1e-9
    block = A.copy()
    block[:2, :2] = best[:2, :2]                    # F block floored, cross terms kept
    assert q(block) < q(best) - 1e-6
    for _ in range(200):
        M = rng.normal(size=(3, 3)) * 0.1
        S = 0.5 * (M + M.T)
        S[:2, :2] = M[:2, :2] @ M[:2, :2].T          # stays feasible on F
        G = best + 0.1 * S
        if np.linalg.eigvalsh(G).min() > 0:
            assert q(G) <= q(best) + 1e-9
    full = _floor_gamma_at_noise(spike, np.array([0.04, 0.09]))
    assert np.all(np.linalg.eigvalsh(full / np.sqrt(np.outer([0.04, 0.09], [0.04, 0.09]))) >= 1 - 1e-9)


def test_noise_floor_full_covariance():
    from src.assay_calibration.fit_utils.cfusn.update_steps import _floor_gamma_at_noise
    var = np.array([0.04, 0.09])
    # zero correlation: identical to the diagonal floor
    spike = np.array([[1e-4, 2e-5], [2e-5, 1e-4]])
    assert np.allclose(_floor_gamma_at_noise(spike, np.diag(var)), _floor_gamma_at_noise(spike, var))
    # correlated noise allows a component narrower along the low-noise direction
    r = 0.8
    S = np.array([[var[0], r * np.sqrt(var[0] * var[1])], [r * np.sqrt(var[0] * var[1]), var[1]]])
    G = 0.95 * S                                     # just under the full-covariance floor
    F = _floor_gamma_at_noise(G, S)
    assert np.allclose(F, S, atol=1e-10)             # raised exactly to Sigma
    Li = np.linalg.inv(np.linalg.cholesky(S))
    assert np.linalg.eigvalsh(Li @ F @ Li.T).min() >= 1 - 1e-9
    D = _floor_gamma_at_noise(G, var)                # the diagonal floor over-widens it
    assert np.linalg.eigvalsh(Li @ D @ Li.T).max() > 1.5
    # NaN readout: same as the partial vector floor
    P = _floor_gamma_at_noise(spike, np.array([[0.04, np.nan], [np.nan, np.nan]]))
    assert P[0, 0] >= 0.04 - 1e-12 and np.allclose(P, _floor_gamma_at_noise(spike, np.array([0.04, np.nan])))
