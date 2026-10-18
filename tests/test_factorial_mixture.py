"""Tests for the experimental factorial (axis-tied) mixture
(src/assay_calibration/fit_utils/factorial/).

    source activate excalibr
    OMP_NUM_THREADS=4 nice -n 10 taskset -c 0-7 pytest tests/test_factorial_mixture.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.assay_calibration.fit_utils.factorial import FactorialMixture, weight_step, total_correlation
from src.assay_calibration.fit_utils.factorial import export as fx
from src.assay_calibration.fit_utils.factorial import axis_search as fs
from src.assay_calibration.fit_utils.factorial.attribution import attribution_table
from src.assay_calibration.fit_utils.cfusn import categorical as cat
from src.assay_calibration.fit_utils.cfusn.density_utils import get_likelihood


def _synthetic(seed=0, n=(220, 220, 400), missing=0.15, with_cat=True):
    """Two independent axes: A = dims 0,1 ; B = dims 2,3 (+ categorical dim 4 on B).
    Samples: P (defective on A or B), B (normal on both), pop (mixture)."""
    rng = np.random.default_rng(seed)
    means = {("A", 0): [0, 0], ("A", 1): [3, 2.5], ("B", 0): [0, 0], ("B", 1): [-2.5, 3]}
    pdef = {"P": (0.6, 0.6), "B": (0.05, 0.05), "pop": (0.3, 0.3)}
    X, S = [], []
    for s, (name, size) in enumerate(zip(("P", "B", "pop"), n)):
        a = rng.random(size) < pdef[name][0]
        b = rng.random(size) < pdef[name][1]
        xa = np.array([rng.normal(means[("A", int(t))], 0.6) for t in a])
        xb = np.array([rng.normal(means[("B", int(t))], 0.6) for t in b])
        lev = np.where(b, rng.choice(3, size, p=[0.1, 0.2, 0.7]), rng.choice(3, size, p=[0.8, 0.15, 0.05]))
        cols = [xa, xb] + ([lev[:, None].astype(float)] if with_cat else [])
        X.append(np.hstack(cols))
        sa = np.zeros((size, 3), bool)
        sa[:, s] = True
        S.append(sa)
    X, S = np.vstack(X), np.vstack(S)
    m = rng.random(X.shape) < missing
    m[np.arange(len(X)), rng.integers(0, X.shape[1], len(X))] = False   # keep >= 1 observed
    X[m] = np.nan
    return X, S


def _axes(kind, with_cat=True):
    b = dict(cont=[2, 3], cats=[4], levels=[3]) if with_cat else dict(cont=[2, 3])
    return [dict(cont=[0, 1], n_states=2, kind=kind), dict(b, n_states=2, kind=kind)]


@pytest.mark.parametrize("kind", ["gaussian", "msn", "cfusn"])
def test_objective_monotone(kind):
    X, S = _synthetic(seed=1)
    m = FactorialMixture(_axes(kind), tau=5.0, alpha_w=0.05, seed=0, n_mc=100).fit(X, S, max_iter=60, tol=0)
    d = np.diff(m.objectives) / np.maximum(np.abs(m.objectives[:-1]), 1)
    assert m.overshoot is None, m.overshoot
    assert d.min() >= -1e-10, d.min()
    np.testing.assert_allclose(m.W.sum(1), 1.0)
    assert np.all(m.W > 0), "Dirichlet floor must remove exact zeros"


def test_weight_step_never_decreases():
    rng = np.random.default_rng(0)
    shape = (3, 2, 2)
    for _ in range(20):
        W_old = rng.dirichlet(np.ones(12))
        counts = rng.gamma(1.0, 5.0, 12) + 0.05
        from src.assay_calibration.fit_utils.factorial.model import weight_objective
        for tau in (0.0, 5.0, 50.0):
            W = weight_step(counts, W_old, tau, shape)
            assert weight_objective(W, counts, tau, shape) >= weight_objective(W_old, counts, tau, shape) - 1e-12


@pytest.mark.parametrize("kind", ["gaussian", "msn", "cfusn"])
def test_export_is_exact(kind):
    X, S = _synthetic(seed=2)
    m = FactorialMixture(_axes(kind), seed=0, n_mc=100).fit(X, S, max_iter=30)
    spec = {"names": ["c"], "positions": [4], "level_names": [["a", "b", "c"]], "cont_positions": [0, 1, 2, 3]}
    mean, std = np.zeros(5), np.ones(5)
    fit = fx.to_fit_dict(m, mean, std, cat_spec=spec)
    params = cat.wrap_params([tuple(np.asarray(v) for v in comp) for comp in fit["component_params"]], fit)
    assert isinstance(params, fx.FactorialParams)
    # analysis path (categorical.component_logpdf_full) == model's own density, per cell
    from src.assay_calibration.multivariate_analysis.mv_calibration import _comp_logpdf
    lp_model = m.cell_logpdfs(X)
    lp_hook = np.array([_comp_logpdf(X, params, c) for c in range(len(params))])
    np.testing.assert_allclose(lp_hook, lp_model, atol=1e-9)
    # the stored joint (mu, Delta, Gamma) tuples are exact too (block-diagonal CFUSN)
    from src.assay_calibration.fit_utils.cfusn.density_utils import _single_component_logpdf
    Xc, labels, _ = cat.split_observations(X[:40], cat.CatSpec.from_dict(spec))
    # (latent dim = sum of axes' q; above 2 the production CFUSN density uses a
    # numerical orthant probability, so the joint tuple is exact only up to that
    # integration error -- the analysis path above uses the exact block sum)
    atol = 1e-6 if int(fit["latent_q"]) <= 2 else 1e-3
    for c in range(len(params)):
        joint = np.atleast_1d(_single_component_logpdf(Xc, params[c], True)).ravel()
        np.testing.assert_allclose(joint, params.cont_logpdf(Xc, c), atol=atol)
    # full mixture LL through the production likelihood (full-width rows; the
    # wrapped params add the categorical term themselves) equals the model's
    ll = get_likelihood(X, S, params, np.asarray(fit["weights"]), multivariate=True)
    np.testing.assert_allclose(ll, m.loglik(X, S), rtol=1e-9)


def test_comeasurement_rule_blocks_disjoint_dims():
    X, S = _synthetic(seed=3, with_cat=False)
    X = np.hstack([X, np.full((len(X), 1), np.nan)])
    X[:300, 4] = np.random.default_rng(0).normal(size=300)
    X[:300, 2:4] = np.nan                          # dim 4 never co-observed with dims 2, 3
    ok = fs.comeasurement_ok(X, S[:, :2].any(1))
    assert not ok[2, 4] and not ok[3, 4] and ok[0, 1]
    D = np.ones((5, 5))
    for part in fs.candidate_partitions(D, ok):
        for axis in part:
            assert fs.axis_allowed(axis, ok), part


def test_search_recovers_true_axes():
    splits = []
    for seed in (10, 11, 12):
        X, S = _synthetic(seed=seed, with_cat=False)
        idx = np.random.default_rng(seed).permutation(len(X))
        tr, va = idx[: int(0.7 * len(X))], idx[int(0.7 * len(X)):]
        splits.append((X[tr], S[tr], X[va], S[va]))
    X0, S0 = _synthetic(seed=10, with_cat=False)
    srch = fs.AxisSearch(splits, [], [], X0, S0[:, :2].any(1), kind="gaussian", states_range=(1, 2, 3),
                         seeds=(0,), workers=4, log=lambda *a: None)
    part, ns, kind = srch.run(k_uniform=(2,), max_sweeps=2)
    assert fs.canon(part) == ((0, 1), (2, 3)), (part, ns)
    assert tuple(ns) == (2, 2), ns


def test_attribution_flags_single_axis_defect():
    X, S = _synthetic(seed=4, with_cat=False)
    m = FactorialMixture(_axes("gaussian", with_cat=False), seed=0).fit(X, S)
    probe = np.array([[3.0, 2.5, 0.0, 0.0],      # defective on A only
                      [0.0, 0.0, -2.5, 3.0],     # defective on B only
                      [0.0, 0.0, 0.0, 0.0],      # normal
                      [3.0, 2.5, -2.5, 3.0]])    # both
    df, _ = attribution_table(m, probe, ["A", "B", "none", "both"], p_idx=0, b_idx=1, axis_names=["A", "B"])
    assert df.single_axis_defect.tolist() == [True, True, False, False]
    assert df.defective_axis.tolist()[:2] == ["A", "B"]


def test_total_correlation_zero_for_product():
    a, b = np.array([0.3, 0.7]), np.array([0.2, 0.5, 0.3])
    assert abs(total_correlation(np.outer(a, b).ravel(), (2, 3))) < 1e-12
