"""Tests for the opt-in SQUAREM EM accelerator (cfusn/squarem.py).

    source activate excalibr
    OMP_NUM_THREADS=1 nice -n 19 pytest tests/test_squarem.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "cfusn_simulations"))

from src.assay_calibration.fit_utils.cfusn import categorical as cat  # noqa: E402
from src.assay_calibration.fit_utils.cfusn import squarem as sq  # noqa: E402
from src.assay_calibration.fit_utils.cfusn.fit import single_fit  # noqa: E402
from src.assay_calibration.fit_utils.cfusn.update_steps import gamma_log_prior_total  # noqa: E402
from sim_utils import sample_cfusn_mixture, sample_categorical_labels, inject_missingness  # noqa: E402

FIT_COMMON = dict(constrained=False, init_method="kmeans", init_constraint_adjustment=None,
                  multivariate=True, verbose=False)
PARAMS_3D = [
    (np.array([0.0, 0.0, 0.0]), np.array([[0.8, 0.0], [0.0, 0.3], [0.2, 0.1]]), np.eye(3) * 0.5),
    (np.array([2.0, 1.5, 1.0]), np.array([[-0.5, 0.0], [0.0, 0.4], [0.1, -0.3]]), np.eye(3) * 0.5),
]
W_2 = np.array([[0.7, 0.3], [0.3, 0.7]])
PI = [np.array([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])]


def _make(seed=0, with_cat=True, miss=0.25):
    rng = np.random.RandomState(seed)
    X, S, comp = sample_cfusn_mixture(PARAMS_3D, W_2, [250, 250], rng)
    X = inject_missingness(X, [0.0, miss, miss], rng)
    if not with_cat:
        return X, S, None
    labels = inject_missingness(sample_categorical_labels(comp, PI, rng), [0.3], rng)
    obs = np.hstack([X, labels])
    spec = cat.resolve_cat_spec({X.shape[1]: ["l0", "l1", "l2"]}, obs.shape[1])
    return obs, S, spec


def _objective_series(res, X_cont, N):
    hist = res["history"][:len(res["likelihoods"])]
    out = []
    for ll, h in zip(res["likelihoods"], hist):
        o = ll + gamma_log_prior_total(h["component_params"], X_cont, multivariate=True) / N
        if "cat_probs" in h:
            o += cat.cat_log_prior(h["cat_probs"], res["cat_alpha"]) / N
        out.append(o)
    return np.array(out)


def test_pack_unpack_roundtrip():
    rng = np.random.RandomState(0)
    p = 4
    params = []
    for D in (rng.randn(p, 2), rng.randn(p)):
        A = rng.randn(p, p)
        params.append((rng.randn(p), D, A @ A.T + np.eye(p)))
    W = np.array([[0.2, 0.8], [0.0, 1.0], [0.5, 0.5]])
    probs = [np.array([[0.1, 0.3, 0.6], [0.5, 0.25, 0.25]])]
    theta = (params, W, probs)
    pk = sq.Packer([theta, theta, theta])
    back = pk.unpack(pk.pack(theta))
    for (m, D, G), (m2, D2, G2) in zip(params, back[0]):
        assert np.allclose(m, m2) and np.allclose(D, D2) and np.allclose(G, G2)
        assert np.shape(D) == np.shape(D2)
    assert np.allclose(W, back[1]) and back[1][1, 0] == 0.0
    assert np.allclose(probs[0], back[2][0])


def test_extrapolated_gamma_is_positive_definite():
    rng = np.random.RandomState(1)
    p = 3
    A = rng.randn(p, p)
    G0 = A @ A.T + np.eye(p)
    thetas = []
    for k in range(3):
        # three nearby iterates, as consecutive EM steps are
        E = 0.02 * k * rng.randn(p, p)
        thetas.append(([(rng.randn(p), rng.randn(p, 2), G0 + 0.5 * (E + E.T))],
                       np.array([[1.0]]), None))
    pk = sq.Packer(thetas)
    x0, x1, x2 = (pk.pack(t) for t in thetas)
    r, v = x1 - x0, x2 - 2 * x1 + x0
    for alpha in (1.0, 2.0, 4.0):
        G = pk.unpack(x0 + 2 * alpha * r + alpha ** 2 * v)[0][0][2]
        assert np.linalg.eigvalsh(G).min() > 0


@pytest.mark.parametrize("latent_q,force_gaussian,with_cat", [
    (2, False, True), (1, False, True), (2, True, False), (2, False, False)])
def test_squarem_monotone_and_not_worse(latent_q, force_gaussian, with_cat):
    obs, S, spec = _make(seed=3, with_cat=with_cat)
    kw = dict(latent_q=latent_q, force_gaussian=force_gaussian, fit_seed=5, max_em_iters=3000)
    if spec is not None:
        kw["cat_spec"] = spec
    plain = single_fit(obs, S, 2, **kw, **FIT_COMMON)
    acc = single_fit(obs, S, 2, em_accelerator="squarem", **kw, **FIT_COMMON)
    X_cont = obs if spec is None else cat.split_observations(obs, spec)[0]
    obj = _objective_series(acc, X_cont, len(S))
    rel = np.diff(obj) / np.abs(obj[:-1])
    assert rel.min() >= -1e-10, f"accepted iterates not monotone: {rel.min():.3e}"
    assert np.isclose(acc["final_objective"], obj[-1], rtol=0, atol=1e-12)
    # never worse than plain EM from the same start (by more than the stop tolerance)
    assert acc["final_objective"] >= plain["final_objective"] - 1e-8 * abs(plain["final_objective"])
    assert acc["em_accelerator"] == "squarem" and acc["squarem_stats"]["cycles"] > 0
    assert acc["n_em_iters"] <= plain["n_em_iters"]


def test_budget_counts_every_em_call():
    obs, S, _ = _make(seed=2, with_cat=False)
    res = single_fit(obs, S, 2, latent_q=2, fit_seed=1, max_em_iters=25, early_stopping=False,
                     em_accelerator="squarem", **FIT_COMMON)
    assert res["n_em_iters"] <= 26 and res["stop_reason"] == "cap"


def test_default_is_plain_em():
    obs, S, _ = _make(seed=2, with_cat=False)
    a = single_fit(obs, S, 2, latent_q=2, fit_seed=1, max_em_iters=60, **FIT_COMMON)
    b = single_fit(obs, S, 2, latent_q=2, fit_seed=1, max_em_iters=60, em_accelerator="none",
                   **FIT_COMMON)
    assert np.array_equal(a["likelihoods"], b["likelihoods"])
    assert "em_accelerator" not in a


@pytest.mark.parametrize("bad", [dict(latent_q=3), dict(constrained=True),
                                 dict(dim_proportions=[0.5, 0.3, 0.2])])
def test_refused_configurations(bad):
    obs, S, _ = _make(seed=2, with_cat=False)
    kw = dict(FIT_COMMON, latent_q=2, fit_seed=1, max_em_iters=5, em_accelerator="squarem")
    kw.update(bad)
    with pytest.raises((NotImplementedError, ValueError)):
        single_fit(obs, S, 2, raise_on_error=True, **kw)


def test_unknown_accelerator_rejected():
    with pytest.raises(ValueError):
        sq.resolve({"em_accelerator": "anderson"})


# ── univariate (EXPERIMENTAL, opt-in) ───────────────────────────────────────

UV_COMMON = dict(init_method="kmeans", init_constraint_adjustment="scale", multivariate=False,
                 verbose=False, check_monotonic=True)


def _make_uv(seed=0):
    from scipy.stats import skewnorm
    rng = np.random.RandomState(seed)
    path = skewnorm.rvs(-2, loc=-2.0, scale=1.2, size=120, random_state=rng)
    ben = skewnorm.rvs(2, loc=1.0, scale=1.0, size=200, random_state=rng)
    pop = np.concatenate([skewnorm.rvs(-2, loc=-2.0, scale=1.2, size=60, random_state=rng),
                          skewnorm.rvs(2, loc=1.0, scale=1.0, size=240, random_state=rng)])
    x = np.concatenate([path, ben, pop])
    S = np.zeros((len(x), 3), dtype=bool)
    S[:120, 0] = True
    S[120:320, 1] = True
    S[320:, 2] = True
    return x, S


def test_univariate_pack_unpack_roundtrip():
    params = [(-1.5, -2.0, 1.2), (0.0, 0.5, 0.8), (2.0, 1.0, 1.0)]
    W = np.array([[0.7, 0.2, 0.1], [0.0, 0.4, 0.6]])
    pk = sq.UnivariatePacker([(params, W, None)] * 3)
    p2, W2, _ = pk.unpack(pk.pack((params, W, None)))
    assert np.allclose(np.asarray(p2), np.asarray(params), atol=1e-12)
    assert np.allclose(W2, W) and W2[1, 0] == 0.0


@pytest.mark.parametrize("constrained", [True, False])
@pytest.mark.parametrize("seed", [0, 1])
def test_univariate_squarem_monotone_feasible_and_close(constrained, seed):
    from src.assay_calibration.fit_utils.cfusn import constraints
    x, S = _make_uv(seed)
    kw = dict(UV_COMMON, constrained=constrained, fit_seed=seed, max_em_iters=10000)
    a = single_fit(x, S, 2, **kw)
    b = single_fit(x, S, 2, em_accelerator="squarem", **kw)
    assert b["em_accelerator"] == "squarem"
    # accepted iterates are monotone in the objective (gamma prior is 0 univariate)
    L = np.asarray(b["likelihoods"])
    assert np.all(np.diff(L) >= -1e-8 * np.abs(L[:-1]))
    if constrained:
        assert not constraints.multicomponent_density_constraint_violated(
            b["component_params"], b["xlims"], multivariate=False)
    # same fixed point as plain EM, to within the stopping tolerance
    assert b["final_objective"] >= a["final_objective"] - 1e-6 * abs(a["final_objective"])
    assert np.allclose(np.asarray(b["component_params"]), np.asarray(a["component_params"]), atol=5e-2)
