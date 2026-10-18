"""Tests for the experimental weight-divergence penalty on the mixing weights
(src/assay_calibration/fit_utils/cfusn/weight_divergence.py) and the
min_rows path of add_component_to_fit.

The central requirement: with the penalty on, EM must stay MONOTONE in the
tracked (penalised) objective on every M-step path, exactly as the plain fit
is in its own objective.

Run with (throttled -- the server is shared):
    source activate excalibr
    OMP_NUM_THREADS=4 nice -n 10 taskset -c 0-7 pytest tests/test_weight_divergence.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "cfusn_simulations"))

from src.assay_calibration.fit_utils.cfusn import categorical as cat
from src.assay_calibration.fit_utils.cfusn import weight_divergence as wd
from src.assay_calibration.fit_utils.cfusn.fit import single_fit, add_component_to_fit
from src.assay_calibration.fit_utils.cfusn.density_utils import get_likelihood
from src.assay_calibration.fit_utils.cfusn.update_steps import gamma_log_prior_total
from sim_utils import sample_cfusn_mixture, sample_categorical_labels

FIT_COMMON = dict(constrained=False, init_method="kmeans", init_constraint_adjustment=None,
                  multivariate=True, verbose=False, n_mc_truncated=50)

PARAMS = [
    (np.array([0.0, 0.0]), np.array([[0.6, 0.0], [0.0, 0.3]]), np.eye(2) * 0.5),
    (np.array([2.5, 1.5]), np.array([[-0.5, 0.0], [0.0, 0.4]]), np.eye(2) * 0.5),
    (np.array([0.5, 3.0]), np.array([[0.3, 0.0], [0.0, -0.4]]), np.eye(2) * 0.4),
]
SAMPLE_NAMES = ["P", "B", "gnomAD", "R"]
W_TRUE = np.array([[0.10, 0.85, 0.05],     # P
                   [0.85, 0.10, 0.05],     # B
                   [0.50, 0.40, 0.10],     # gnomAD (population mixture, unpenalised)
                   [0.15, 0.15, 0.70]])    # R: small, distinct
SIZES = [250, 250, 400, 30]
GROUPS = [["P"], ["B", "Syn"], ["R"]]


def _make(seed=0):
    rng = np.random.RandomState(seed)
    X, S, comp = sample_cfusn_mixture(PARAMS, W_TRUE, SIZES, rng)
    return X, S, comp


def _spec(lam):
    return wd.resolve_weight_divergence({"lambda": lam, "groups": GROUPS}, SAMPLE_NAMES, len(SAMPLE_NAMES))


def _eq(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float):
        return (a == b) or (np.isnan(a) and np.isnan(b))
    return a == b


# ── spec resolution ─────────────────────────────────────────────────────────
def test_resolve_groups_and_exclusions():
    spec = _spec(0.1)
    # P=0, B=1, R=3; gnomAD (2) in no group; "Syn" absent -> dropped.
    assert spec == {"lambda": 0.1, "pairs": [(0, 1), (0, 3), (1, 3)]}
    names = ["P", "B", "gnomAD", "Syn", "R"]
    s2 = wd.resolve_weight_divergence({"lambda": 0.1, "groups": GROUPS}, names, 5)
    assert (1, 3) not in s2["pairs"] and (3, 1) not in s2["pairs"], "B/LB vs Synonymous must never be penalised"
    assert all(2 not in p for p in s2["pairs"]), "gnomAD must not be penalised"
    assert wd.resolve_weight_divergence({"lambda": 0.0, "groups": GROUPS}, SAMPLE_NAMES, 4) is None
    assert wd.resolve_weight_divergence(None, SAMPLE_NAMES, 4) is None
    with pytest.raises(ValueError):
        wd.resolve_weight_divergence({"lambda": 0.1, "groups": [["P"], ["P", "B"]]}, SAMPLE_NAMES, 4)


# ── gradient and the GEM W step ─────────────────────────────────────────────
def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(5), size=4)
    W[3, 1] = 1e-6
    pairs = _spec(1.0)["pairs"]
    G = wd.divergence_grad(W, pairs)
    for i in range(4):
        for k in range(5):
            h = 1e-4 * W[i, k]                               # relative step: log-curvature ~1/W
            Wp, Wm = W.copy(), W.copy()
            Wp[i, k] += h
            Wm[i, k] -= h
            fd = (wd.divergence(Wp, pairs) - wd.divergence(Wm, pairs)) / (2 * h)
            assert abs(fd - G[i, k]) < 1e-5 * max(1.0, abs(fd)), (i, k, fd, G[i, k])


@pytest.mark.parametrize("seed", range(20))
def test_weight_step_never_decreases_minorizer(seed):
    rng = np.random.default_rng(seed)
    S, K = 4, 5
    W_old = rng.dirichlet(np.full(K, 0.5), size=S)
    W_old[rng.random((S, K)) < 0.15] = 0.0                    # exact zeros
    W_old[:, 0] = np.maximum(W_old[:, 0], 1e-9)
    W_old /= W_old.sum(1, keepdims=True)
    n = rng.integers(3, 300, size=S)
    nbar = rng.dirichlet(np.ones(K), size=S) * n[:, None] * (W_old > 0)
    pairs = _spec(1.0)["pairs"]
    for lam_total in (0.0, 1.0, 30.0, 1000.0):
        W_new = wd.penalized_weight_update(nbar, W_old, pairs, lam_total)
        f_old = wd.minorizer(W_old, nbar, pairs, lam_total)
        f_new = wd.minorizer(W_new, nbar, pairs, lam_total)
        assert f_new >= f_old - 1e-12 * max(1.0, abs(f_old)), (lam_total, f_old, f_new)
        np.testing.assert_allclose(W_new.sum(1), 1.0, atol=1e-12)
        assert np.all(W_new[W_old == 0] == 0), "exact zeros must stay zero"
        closed = nbar / nbar.sum(1, keepdims=True)
        np.testing.assert_array_equal(W_new[2], closed[2])   # unpenalised row (gnomAD) untouched


# ── full fits: monotone in the tracked penalised objective ──────────────────
MODES = {"gaussian": dict(latent_q=2, force_gaussian=True), "q1": dict(latent_q=1), "q2": dict(latent_q=2)}


def _check_monotone(res, X, S, lam, pairs, extra=None, cat_alpha=None):
    assert res["overshoot_stopped"] is None, f"overshoot revert: {res['overshoot_stopped']}"
    obj = np.asarray(res["objectives"])
    rel = np.diff(obj) / np.maximum(np.abs(obj[:-1]), 1.0)
    assert rel.min() >= -1e-10, f"penalised objective decreased: min rel diff {rel.min():.3e}"
    # Independent recomputation of the tracked objective at the returned iterate.
    N = len(S)
    Xc, ex = X, None
    if extra is not None:
        Xc, labels, _ = cat.split_observations(X, extra)
        ex = cat.cat_log_pdfs(labels, res["cat_probs"])
    L = get_likelihood(Xc, S, res["component_params"], res["weights"], multivariate=True,
                       extra_log_pdfs=ex) / N
    L += gamma_log_prior_total(res["component_params"], Xc, multivariate=True) / N
    if extra is not None:
        L += cat.cat_log_prior(res["cat_probs"], res["cat_alpha"]) / N
    L += lam * wd.divergence(res["weights"], pairs)
    np.testing.assert_allclose(L, obj.max(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("lam", [0.03, 0.3, 1.0])
def test_penalised_fit_is_monotone(mode, lam):
    X, S, _ = _make(seed=3)
    spec = _spec(lam)
    res = single_fit(X, S, 3, fit_seed=5, max_em_iters=60, early_stopping=False,
                     weight_divergence=spec, **MODES[mode], **FIT_COMMON)
    _check_monotone(res, X, S, lam, spec["pairs"])


@pytest.mark.parametrize("mode", ["gaussian", "q1"])
def test_penalised_fit_is_monotone_with_sample_balance(mode):
    X, S, _ = _make(seed=4)
    spec = _spec(0.3)
    res = single_fit(X, S, 3, fit_seed=2, max_em_iters=60, early_stopping=False,
                     weight_divergence=spec, sample_balance_beta=0.5, **MODES[mode], **FIT_COMMON)
    # The tracked LL is sample-weighted here, so only the series itself is checked.
    assert res["overshoot_stopped"] is None
    obj = np.asarray(res["objectives"])
    rel = np.diff(obj) / np.maximum(np.abs(obj[:-1]), 1.0)
    assert rel.min() >= -1e-10, f"penalised objective decreased: min rel diff {rel.min():.3e}"


def test_penalised_fit_is_monotone_with_categorical_dim():
    rng = np.random.RandomState(9)
    X, S, comp = sample_cfusn_mixture(PARAMS, W_TRUE, SIZES, rng)
    pi = [np.array([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])]
    labels = sample_categorical_labels(comp, pi, rng).astype(float)
    labels[rng.random(len(labels)) < 0.5] = np.nan
    obs = np.hstack([X, labels])
    cspec = cat.resolve_cat_spec({2: ["a", "b", "c"]}, 3)
    spec = _spec(0.3)
    res = single_fit(obs, S, 3, fit_seed=1, max_em_iters=60, early_stopping=False, cat_spec=cspec,
                     weight_divergence=spec, latent_q=1, **FIT_COMMON)
    _check_monotone(res, obs, S, 0.3, spec["pairs"], extra=cspec)


def test_zero_lambda_is_bitwise_identical():
    X, S, _ = _make(seed=1)
    kw = dict(fit_seed=3, max_em_iters=30, latent_q=1, **FIT_COMMON)
    base = single_fit(X, S, 3, **kw)
    off = single_fit(X, S, 3, weight_divergence={"lambda": 0.0, "groups": GROUPS}, **kw)
    assert _eq(base, off)


def test_penalty_increases_sample_divergence():
    gains = []
    for seed in range(5):
        X, S, _ = _make(seed=10 + seed)
        kw = dict(fit_seed=seed, max_em_iters=150, latent_q=1, **FIT_COMMON)
        spec = _spec(1.0)
        r0 = single_fit(X, S, 3, **kw)
        r1 = single_fit(X, S, 3, weight_divergence=spec, **kw)
        gains.append(wd.divergence(r1["weights"], spec["pairs"]) - wd.divergence(r0["weights"], spec["pairs"]))
    assert np.mean(gains) > 0 and sum(g >= -1e-6 for g in gains) >= 4, gains


def test_separation_and_dim_weights_are_refused():
    X, S, _ = _make(seed=0)
    with pytest.raises(ValueError):
        single_fit(X, S, 2, fit_seed=0, max_em_iters=3, weight_divergence=_spec(0.1), latent_q=1,
                   **{**FIT_COMMON, "constrained": True})


def test_jax_path_refuses_weight_divergence():
    pytest.importorskip("jax")
    from src.assay_calibration.fit_utils.jax_batch import interop
    with pytest.raises(NotImplementedError):
        interop._init_cfusn({"kwargs": {"weight_divergence": {"lambda": 0.1, "pairs": [(0, 1)]}}})


# ── dedicated component for a tiny sample (add_component_to_fit min_rows) ───
def test_add_component_min_rows_four_row_target():
    X, S, _ = _make(seed=7)
    r_rows = np.where(S[:, 3])[0]
    keep = np.ones(len(X), bool)
    keep[r_rows[4:]] = False                                   # R reduced to 4 rows
    X, S = X[keep], S[keep]
    base = single_fit(X, S, 2, fit_seed=0, max_em_iters=60, latent_q=1, **FIT_COMMON)
    with pytest.raises(ValueError):
        add_component_to_fit(X, S, base, target_sample_idx=3, latent_q=1, fit_seed=0, max_em_iters=20,
                             verbose=False, n_mc_truncated=50)
    grown = add_component_to_fit(X, S, base, target_sample_idx=3, latent_q=1, fit_seed=0, max_em_iters=40,
                                 verbose=False, n_mc_truncated=50, min_rows=3)
    W = np.asarray(grown["weights"])
    assert W.shape == (4, 3)
    assert np.all(W[:3, 2] == 0), "other samples must stay at exactly 0 on the dedicated component"
    assert W[3, 2] > 0
    assert all(np.all(np.isfinite(np.asarray(a))) for p in grown["component_params"] for a in p)
    for k in range(2):                                         # base components frozen
        for a, b in zip(grown["component_params"][k], base["component_params"][k]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_generate_fit_jobs_resolves_and_runs():
    from types import SimpleNamespace
    from src.assay_calibration.fit_utils.fit import Fit
    X, S, _ = _make(seed=2)
    ss = SimpleNamespace(scores=X, sample_assignments=S, dataset_names=["a", "b"], sample_names=SAMPLE_NAMES)
    jobs = Fit(ss).generate_fit_jobs([3], num_fits=1, bootstrap=True, check_monotonic=False, master_seed=0,
                                     bootstrap_seed=0, max_em_iters=30, n_mc_truncated=50,
                                     weight_divergence={"lambda": 0.3, "groups": GROUPS})
    job = jobs[0]
    assert job["kwargs"]["weight_divergence"] == {"lambda": 0.3, "pairs": [(0, 1), (0, 3), (1, 3)]}
    res = Fit.execute_fit_job(job)
    assert np.isfinite(res["val_ll"])
    assert res["fit"]["weight_divergence"]["pairs"] == [(0, 1), (0, 3), (1, 3)]
    off = Fit(ss).generate_fit_jobs([3], num_fits=1, bootstrap=True, check_monotonic=False, master_seed=0,
                                    bootstrap_seed=0, max_em_iters=30, n_mc_truncated=50,
                                    weight_divergence={"lambda": 0.0, "groups": GROUPS})[0]
    assert "weight_divergence" not in off["kwargs"]
