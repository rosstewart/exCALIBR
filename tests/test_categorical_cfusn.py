"""Tests for categorical (nominal) dimensions in the CFUSN mixture
(location model -- see src/assay_calibration/fit_utils/cfusn/categorical.py).

Run with (throttled -- the server is shared):
    source activate excalibr
    OMP_NUM_THREADS=8 taskset -c 0-7 pytest tests/test_categorical_cfusn.py -v
"""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "cfusn_simulations"))
sys.path.insert(0, str(REPO_ROOT / "tests" / "golden"))

from src.assay_calibration.fit_utils.cfusn import categorical as cat
from src.assay_calibration.fit_utils.cfusn import fit as cfit
from src.assay_calibration.fit_utils.cfusn.fit import single_fit, add_component_to_fit
from src.assay_calibration.fit_utils.cfusn.density_utils import (
    get_likelihood, component_posteriors, _single_component_logpdf,
)
from src.assay_calibration.fit_utils.cfusn.update_steps import gamma_log_prior_total
from src.assay_calibration.fit_utils.fit import Fit
from sim_utils import sample_cfusn_mixture, sample_categorical_labels, inject_missingness

FIT_COMMON = dict(constrained=False, init_method="kmeans", init_constraint_adjustment=None,
                  multivariate=True, verbose=False, n_mc_truncated=50)

PARAMS_2D = [
    (np.array([0.0, 0.0]), np.array([[0.8, 0.0], [0.0, 0.3]]), np.eye(2) * 0.5),
    (np.array([2.0, 1.5]), np.array([[-0.5, 0.0], [0.0, 0.4]]), np.eye(2) * 0.5),
]
W_2 = np.array([[0.7, 0.3], [0.3, 0.7]])
PI_TRUE = [np.array([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]])]


def _make(seed=0, sizes=(300, 300), params=PARAMS_2D, weights=W_2, pi=PI_TRUE, cat_missing=0.3):
    rng = np.random.RandomState(seed)
    X, S, comp = sample_cfusn_mixture(params, weights, list(sizes), rng)
    labels = sample_categorical_labels(comp, pi, rng)
    labels = inject_missingness(labels, [cat_missing] * labels.shape[1], rng)
    obs = np.hstack([X, labels])
    spec = cat.resolve_cat_spec({X.shape[1] + c: ["l0", "l1", "l2"] for c in range(len(pi))},
                                obs.shape[1])
    return obs, S, comp, spec


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


# ── (a) M-step correctness ──────────────────────────────────────────────────

def test_cat_mstep_matches_bruteforce_and_maximises():
    rng = np.random.RandomState(1)
    N, K, L = 200, 3, 4
    labels = rng.randint(0, L, size=(N, 1))
    labels[rng.rand(N) < 0.25, 0] = -1
    z = rng.dirichlet(np.ones(K), size=N).T
    sw = rng.uniform(0.5, 2.0, size=N)
    alpha = [rng.uniform(0.1, 1.0, size=L)]
    old = [rng.dirichlet(np.ones(L), size=K)]

    P = cat.cat_mstep(labels, z, alpha, old_probs=old, frozen={1}, sample_weights=sw)[0]

    for k in range(K):
        if k == 1:
            np.testing.assert_array_equal(P[k], old[0][k])
            continue
        counts = np.array([sum(z[k, n] * sw[n] for n in range(N) if labels[n, 0] == l)
                           for l in range(L)])
        expect = (alpha[0] + counts) / (alpha[0].sum() + counts.sum())
        np.testing.assert_allclose(P[k], expect, rtol=0, atol=1e-12)

    def objective(Pk, k):
        counts = np.array([np.sum(z[k] * sw * (labels[:, 0] == l)) for l in range(L)])
        return np.sum((counts + alpha[0]) * np.log(Pk))

    for k in (0, 2):
        best = objective(P[k], k)
        for _ in range(1000):
            alt = P[k] * np.exp(rng.normal(0, 0.05, size=L))
            alt /= alt.sum()
            assert objective(alt, k) <= best + 1e-12


# ── (b) missing categorical contributes nothing ─────────────────────────────

def test_missing_categorical_is_exact_marginalisation():
    obs, S, _, spec = _make(cat_missing=0.0)
    X_cont, labels, _ = cat.split_observations(obs, spec)
    probs = [np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])]
    all_missing = np.full_like(labels, -1)
    extra = cat.cat_log_pdfs(all_missing, probs)
    assert np.all(extra == 0.0)
    W = np.array([[0.6, 0.4], [0.5, 0.5]])
    base = get_likelihood(X_cont, S, PARAMS_2D, W, multivariate=True)
    with_extra = get_likelihood(X_cont, S, PARAMS_2D, W, multivariate=True, extra_log_pdfs=extra)
    assert base == with_extra

    partial = labels.copy()
    partial[::2] = -1
    ex = cat.cat_log_pdfs(partial, probs)
    assert np.all(ex[:, ::2] == 0.0)
    np.testing.assert_allclose(ex[:, 1::2], np.log(probs[0][:, labels[1::2, 0]]))


def test_split_rejects_jittered_codes():
    obs, S, _, spec = _make()
    obs = obs.copy()
    obs[:, spec.positions[0]] += 0.01
    with pytest.raises(ValueError, match="integer codes"):
        cat.split_observations(obs, spec)


# ── (c) regression / opt-in guarantees ──────────────────────────────────────

def test_no_spec_is_bitwise_identical_to_golden():
    from categorical_regression_cases import run_regression_cases, GOLDEN_PATH
    with open(GOLDEN_PATH, "rb") as f:
        gold = pickle.load(f)
    new = run_regression_cases()
    for key in gold:
        assert _eq(gold[key], new[key]), f"regression case {key!r} changed"


def test_spec_naming_absent_dim_takes_plain_path():
    obs, S, _, _ = _make()
    X = obs[:, :2]
    assert cat.resolve_cat_spec({"not_a_dim": ["a", "b"]}, 2, ["d0", "d1"]) is None
    a = single_fit(X, S, 2, latent_q=1, fit_seed=0, max_em_iters=15, **FIT_COMMON)
    b = single_fit(X, S, 2, latent_q=1, fit_seed=0, max_em_iters=15, cat_spec=None, **FIT_COMMON)
    assert _eq(a["component_params"], b["component_params"])
    assert _eq(a["likelihoods"], b["likelihoods"])
    assert "cat_probs" not in a


def test_component_independent_categorical_matches_dropped_dim():
    """With a huge prior, every component's pi is the same, so the
    categorical term is constant across components per row and must cancel
    in responsibilities: continuous params/W match the fit without it."""
    obs, S, _, spec = _make(seed=3)
    kw = dict(latent_q=1, fit_seed=5, max_em_iters=20, early_stopping=False, **FIT_COMMON)
    with_cat = single_fit(obs, S, 2, cat_spec=spec, cat_tau=1e12, **kw)
    dropped = single_fit(obs[:, spec.cont_positions], S, 2, **kw)
    for (m1, d1, g1), (m2, d2, g2) in zip(with_cat["component_params"], dropped["component_params"]):
        np.testing.assert_allclose(m1, m2, atol=1e-8)
        np.testing.assert_allclose(d1, d2, atol=1e-8)
        np.testing.assert_allclose(g1, g2, atol=1e-8)
    np.testing.assert_allclose(with_cat["weights"], dropped["weights"], atol=1e-8)


# ── (d) monotone objective on the exact (q=1 / Gaussian) M-step paths ───────

@pytest.mark.parametrize("mode", ["q1", "gaussian"])
def test_objective_monotone(mode):
    obs, S, _, spec = _make(seed=4)
    obs = obs.copy()
    obs[:20, :2] = np.nan  # include categorical-only rows
    extra_kw = dict(latent_q=1) if mode == "q1" else dict(latent_q=2, force_gaussian=True)
    n_iter = 40
    res = single_fit(obs, S, 2, cat_spec=spec, fit_seed=6, max_em_iters=n_iter,
                     early_stopping=False, **extra_kw, **FIT_COMMON)
    assert res["overshoot_stopped"] is None
    assert len(res["likelihoods"]) == n_iter + 2, "loop broke early on a likelihood decrease"
    X_cont, _, _ = cat.split_observations(obs, spec)
    N = len(S)
    hist = res["history"][:len(res["likelihoods"])]
    obj = np.array([
        ll + gamma_log_prior_total(h["component_params"], X_cont, multivariate=True) / N
        + cat.cat_log_prior(h["cat_probs"], res["cat_alpha"]) / N
        for ll, h in zip(res["likelihoods"], hist)
    ])
    rel = np.diff(obj) / np.abs(obj[:-1])
    assert rel.min() >= -1e-10, f"objective decreased: min rel diff {rel.min():.3e}"


# ── (e) synthetic recovery, and the categorical dim helps ───────────────────

def _accuracy(res, obs, S, comp, spec=None):
    """Fraction of rows whose MAP component (best of the 2 label permutations) matches truth."""
    if spec is not None:
        X_cont, labels, _ = cat.split_observations(obs, spec)
        extra = cat.cat_log_pdfs(labels, res["cat_probs"])
    else:
        X_cont, extra = obs, None
    post = np.zeros((2, len(obs)))
    for s in range(S.shape[1]):
        m = S[:, s]
        post[:, m] = component_posteriors(
            X_cont[m], res["component_params"], res["weights"][s], multivariate=True,
            extra_log_pdfs=None if extra is None else extra[:, m])
    pred = post.argmax(0)
    return max(np.mean(pred == comp), np.mean(pred == 1 - comp))


PI_RECOVERY = [np.array([[0.85, 0.10, 0.05], [0.05, 0.15, 0.80]])]
W_RECOVERY = np.array([[0.6, 0.4], [0.4, 0.6]])


def test_recovery_categorical_probs_when_components_identifiable():
    """Well-separated continuous components: pi_hat must match the
    finite-sample frequencies of each TRUE component's labels (isolates the
    estimator from component-identification error)."""
    sep = [
        (np.array([0.0, 0.0]), np.array([[0.6, 0.0], [0.0, 0.3]]), np.eye(2) * 0.4),
        (np.array([3.0, 2.5]), np.array([[-0.4, 0.0], [0.0, 0.3]]), np.eye(2) * 0.4),
    ]
    for seed in range(5):
        obs, S, comp, spec = _make(seed=10 + seed, sizes=(1000, 1000), params=sep,
                                   weights=W_RECOVERY, pi=PI_RECOVERY, cat_missing=0.2)
        _, labels, _ = cat.split_observations(obs, spec)
        emp = np.array([[np.mean(labels[(comp == k) & (labels[:, 0] >= 0), 0] == l)
                         for l in range(3)] for k in range(2)])
        rc = single_fit(obs, S, 2, cat_spec=spec, latent_q=1, fit_seed=seed,
                        max_em_iters=150, **FIT_COMMON)
        P = rc["cat_probs"][0]
        err_emp = min(np.abs(P - emp).max(), np.abs(P[::-1] - emp).max())
        err_true = min(np.abs(P - PI_RECOVERY[0]).max(), np.abs(P[::-1] - PI_RECOVERY[0]).max())
        assert err_emp < 0.02, f"seed {seed}: max |pi_hat - empirical| = {err_emp:.3f}"
        assert err_true < 0.06, f"seed {seed}: max |pi_hat - pi| = {err_true:.3f}"


def test_categorical_improves_assignment_when_continuous_overlaps():
    """Heavily overlapping continuous components: the categorical dim must
    materially improve recovery of which component generated each row."""
    overlap = [
        (np.array([0.0, 0.0]), np.array([[0.6, 0.0], [0.0, 0.3]]), np.eye(2) * 0.8),
        (np.array([0.9, 0.7]), np.array([[-0.4, 0.0], [0.0, 0.3]]), np.eye(2) * 0.8),
    ]
    wins, gains = 0, []
    for seed in range(5):
        obs, S, comp, spec = _make(seed=10 + seed, sizes=(1000, 1000), params=overlap,
                                   weights=W_RECOVERY, pi=PI_RECOVERY, cat_missing=0.2)
        kw = dict(latent_q=1, fit_seed=seed, max_em_iters=150, **FIT_COMMON)
        rc = single_fit(obs, S, 2, cat_spec=spec, **kw)
        rd = single_fit(obs[:, spec.cont_positions], S, 2, **kw)
        acc_c = _accuracy(rc, obs, S, comp, spec)
        acc_d = _accuracy(rd, obs[:, spec.cont_positions], S, comp)
        gains.append(acc_c - acc_d)
        wins += acc_c > acc_d
    assert wins >= 4, f"categorical dim helped assignment in only {wins}/5 seeds"
    assert np.mean(gains) > 0.1, f"mean accuracy gain only {np.mean(gains):.3f}"


# ── (f) sparse categorical: shrinkage and the lambda floor ──────────────────

def test_sparse_categorical_shrinks_unsupported_component_to_prior():
    sep = [
        (np.array([0.0, 0.0]), np.zeros((2, 2)), np.eye(2) * 0.3),
        (np.array([6.0, 6.0]), np.zeros((2, 2)), np.eye(2) * 0.3),
    ]
    rng = np.random.RandomState(7)
    X, S, comp = sample_cfusn_mixture(sep, np.array([[0.5, 0.5]]), [1000], rng)
    labels = np.full((len(X), 1), np.nan)
    idx0 = np.flatnonzero(comp == 0)[:20]          # only ~2% observed, all from component 0
    labels[idx0, 0] = rng.choice(2, size=len(idx0), p=[0.7, 0.3])  # level 2 never appears
    obs = np.hstack([X, labels])
    spec = cat.resolve_cat_spec({2: ["a", "b", "c"]}, 3)
    res = single_fit(obs, S, 2, cat_spec=spec, latent_q=1, fit_seed=0, max_em_iters=60, **FIT_COMMON)
    P = res["cat_probs"][0]
    alpha = res["cat_alpha"][0]
    prior = alpha / alpha.sum()
    k1 = int(np.argmax([p[0][0] for p in res["component_params"]]))  # the component near (6, 6)
    assert 0.5 * np.abs(P[k1] - prior).sum() < 0.02
    assert np.all(P > 0)
    held_out = np.array([[2]])
    assert np.all(np.isfinite(cat.cat_log_pdfs(held_out, res["cat_probs"])))


# ── (g) rows observed only on the categorical dim ───────────────────────────

def test_categorical_only_rows():
    obs, S, _, spec = _make(seed=8)
    obs = obs.copy()
    only = np.zeros(len(obs), bool)
    only[::10] = True
    obs[only, :2] = np.nan
    obs[only, 2] = np.where(np.isnan(obs[only, 2]), 1.0, obs[only, 2])
    res = single_fit(obs, S, 2, cat_spec=spec, latent_q=2, fit_seed=1, max_em_iters=60, **FIT_COMMON)
    for mu, D, G in res["component_params"]:
        assert np.all(np.isfinite(mu)) and np.all(np.isfinite(D)) and np.all(np.isfinite(G))
    X_cont, labels, has_cont = cat.split_observations(obs, spec)
    assert not has_cont[only].any()
    for n in np.flatnonzero(only)[:5]:
        s = int(np.argmax(S[n]))
        lp = np.array([_single_component_logpdf(X_cont[n:n + 1], p, True) for p in res["component_params"]]).ravel()
        assert np.allclose(lp, 0.0)
        expect = res["weights"][s] * res["cat_probs"][0][:, labels[n, 0]]
        expect = expect / expect.sum()
        got = component_posteriors(X_cont[n:n + 1], res["component_params"], res["weights"][s],
                                   multivariate=True,
                                   extra_log_pdfs=cat.cat_log_pdfs(labels[n:n + 1], res["cat_probs"]))
        np.testing.assert_allclose(got[:, 0], expect, atol=1e-12)


# ── (h) frozen components keep params AND categorical probs ─────────────────

def test_add_component_keeps_frozen_categorical_probs():
    obs, S, _, spec = _make(seed=9)
    base = single_fit(obs, S, 2, cat_spec=spec, latent_q=2, fit_seed=0, max_em_iters=40, **FIT_COMMON)
    grown = add_component_to_fit(obs, S, base, target_sample_idx=1, latent_q=2, cat_spec=spec,
                                 max_em_iters=15, n_mc_truncated=50, verbose=False, fit_seed=2)
    assert len(grown["component_params"]) == 3
    assert grown["cat_probs"][0].shape == (3, 3)
    for k in range(2):
        np.testing.assert_array_equal(grown["cat_probs"][0][k], base["cat_probs"][0][k])
        for a, b in zip(grown["component_params"][k], base["component_params"][k]):
            np.testing.assert_array_equal(a, b)


# ── (i) generate_fit_jobs / execute_fit_job round trip + serialization ──────

def _scoreset(obs, S, names, categorical_dims):
    return SimpleNamespace(scores=obs, sample_assignments=S, dataset_names=names,
                           categorical_dims=categorical_dims)


def test_fit_jobs_roundtrip():
    obs, S, _, _ = _make(seed=11)
    obs = obs.copy()
    obs[:, :2] = obs[:, :2] * np.array([50.0, 0.1]) + np.array([200.0, -3.0])  # very different raw scales
    names = ["cont_a", "cont_b", "kaw"]
    ss = _scoreset(obs, S, names, {"kaw": ["Tetramer", "Dimer", "Monomer"]})
    jobs = Fit(ss).generate_fit_jobs([2], num_fits=3, bootstrap=True, check_monotonic=False,
                                     master_seed=0, bootstrap_seed=0, max_em_iters=25,
                                     n_mc_truncated=50)
    job = jobs[0]
    spec = job["kwargs"]["cat_spec"]
    assert spec["names"] == ["kaw"] and spec["positions"] == [2]
    col = job["train_observations"][:, 2]
    vals = col[~np.isnan(col)]
    assert set(np.unique(vals)) <= {0.0, 1.0, 2.0}, "categorical codes must not be standardized"
    res = Fit.execute_fit_job(job)
    fit = res["fit"]
    assert np.isfinite(res["val_ll"])
    assert all(len(mu) == 2 for mu, _, _ in fit["component_params"])
    mus = np.array([mu for mu, _, _ in fit["component_params"]])
    assert mus[:, 0].min() > 50, "continuous params should be back in raw units"
    blob = json.dumps({"cat_spec": fit["cat_spec"],
                       "cat_probs": [P.tolist() for P in fit["cat_probs"]]})
    back = json.loads(blob)
    assert back["cat_spec"] == fit["cat_spec"]
    np.testing.assert_array_equal(np.array(back["cat_probs"][0]), fit["cat_probs"][0])
    assert _eq(pickle.loads(pickle.dumps(fit["cat_probs"])), fit["cat_probs"])


# ── (j) name-keyed spec survives overlap gating ─────────────────────────────

def test_spec_resolves_after_gating_drops_a_dim():
    obs, S, _, _ = _make(seed=12)
    isolated = np.full(len(obs), np.nan)
    rows = np.arange(10)
    isolated[rows] = 1.0
    obs = obs.copy()
    obs[rows, :] = np.nan            # isolated dim never co-occurs with the others
    full = np.hstack([isolated[:, None], obs])      # columns: iso, cont_a, cont_b, kaw
    ss = _scoreset(full, S, ["iso", "cont_a", "cont_b", "kaw"], {"kaw": ["a", "b", "c"]})
    jobs = Fit(ss).generate_fit_jobs([2], num_fits=1, bootstrap=False, check_monotonic=False,
                                     master_seed=0, bootstrap_seed=0, max_em_iters=5,
                                     verbose_overlap=False)
    assert jobs[0]["calibrated_dims"] == [1, 2, 3]
    assert jobs[0]["kwargs"]["cat_spec"]["positions"] == [2]
    assert jobs[0]["kwargs"]["cat_spec"]["names"] == ["kaw"]


def test_tp53_categorical_ingestion_and_collapse():
    from src.assay_calibration.multivariate_data import tp53
    from src.assay_calibration.multivariate_data import redundancy_collapse as rc
    jit = tp53.build_tp53_multiscoreset()
    jit2 = tp53.build_tp53_multiscoreset(kawoligo_mode="jitter")
    assert np.array_equal(jit.scores, jit2.scores, equal_nan=True), "default mode must be unchanged"
    assert jit.categorical_dims == {}

    ms = tp53.build_tp53_multiscoreset(kawoligo_mode="categorical")
    assert ms.categorical_dims == {"KawOligo": ["Tetramer", "Dimer", "Monomer"]}
    k = list(ms.dataset_names).index("KawOligo")
    vals = ms.scores[:, k]
    assert set(np.unique(vals[~np.isnan(vals)])) == {0.0, 1.0, 2.0}
    others = [i for i in range(ms.scores.shape[1]) if i != k]
    assert np.array_equal(ms.scores[:, others], jit.scores[:, others], equal_nan=True)

    rc.apply_preset({"TP53": ms}, "tp53_kato_pca2", verbose=False)
    k2 = list(ms.dataset_names).index("KawOligo")
    assert ms.categorical_dims == {"KawOligo": ["Tetramer", "Dimer", "Monomer"]}
    np.testing.assert_array_equal(ms.scores[:, k2], vals)
    with pytest.raises(ValueError, match="categorical"):
        rc.collapse_block(ms, ["KawOligo", "DN_score"], 1)
    rc.select_dims(ms, [i for i, n in enumerate(ms.dataset_names) if n != "KawOligo"])
    assert ms.categorical_dims == {}


# ── (k) revert path keeps params and probs consistent ───────────────────────

def test_overshoot_revert_returns_matching_probs(monkeypatch):
    obs, S, _, spec = _make(seed=13)
    real = cfit.em_iteration
    calls = {"n": 0}

    def fake(*a, **k):
        out = list(real(*a, **k))
        calls["n"] += 1
        if calls["n"] == 6:
            out[2] = out[2] - 1e-3          # inject an overshoot-sized decrease
        return tuple(out)

    monkeypatch.setattr(cfit, "em_iteration", fake)
    res = single_fit(obs, S, 2, cat_spec=spec, latent_q=1, fit_seed=0, max_em_iters=30, **FIT_COMMON)
    assert res["overshoot_stopped"] is not None
    match = [h for h in res["history"][:-1] if h["component_params"] is res["component_params"]]
    assert match, "returned params not found in history"
    assert match[0]["cat_probs"] is res["cat_probs"]


# ── (l) guards ──────────────────────────────────────────────────────────────

def test_dim_weights_with_categorical_raises():
    obs, S, _, spec = _make()
    with pytest.raises(ValueError, match="dim_weights"):
        single_fit(obs, S, 2, cat_spec=spec, latent_q=1, dim_weight_transform="log2",
                   max_em_iters=5, **FIT_COMMON)


def test_jax_path_refuses_categorical():
    pytest.importorskip("jax")
    from src.assay_calibration.fit_utils.jax_batch import interop
    with pytest.raises(NotImplementedError):
        interop._init_cfusn({"kwargs": {"cat_spec": {"names": ["x"]}}})


# ── Phase 3: downstream (analysis-side) evaluation ──────────────────────────
def _small_cat_fit(seed=21):
    obs, S, _, _ = _make(seed=seed)
    ss = _scoreset(obs, S, ["cont_a", "cont_b", "kaw"], {"kaw": ["Tetramer", "Dimer", "Monomer"]})
    job = Fit(ss).generate_fit_jobs([2], num_fits=1, bootstrap=False, check_monotonic=False,
                                    master_seed=0, bootstrap_seed=0, max_em_iters=25,
                                    n_mc_truncated=50)[0]
    return obs, S, Fit.execute_fit_job(job)["fit"]


def test_downstream_component_logpdf_matches_fitter():
    from src.assay_calibration.multivariate_analysis import mv_calibration as mvc
    from src.assay_calibration.multivariate_analysis import visualize_fit as vf
    from scipy.special import logsumexp

    obs, S, fit = _small_cat_fit()
    # Through JSON, as the HPC aggregate stores it, then reconstituted.
    fit = json.loads(json.dumps({"component_params": [[np.asarray(a).tolist() for a in p]
                                                      for p in fit["component_params"]],
                                 "weights": np.asarray(fit["weights"]).tolist(),
                                 "cat_spec": fit["cat_spec"],
                                 "cat_probs": [np.asarray(P).tolist() for P in fit["cat_probs"]]}))
    plain = [tuple(np.asarray(a, dtype=float) for a in p) for p in fit["component_params"]]
    params = cat.wrap_params(plain, fit)
    assert isinstance(params, cat.MixtureParams) and cat.total_dims(params) == 3
    assert cat.wrap_params(plain, {}) is plain

    spec = params.cat_spec
    X_cont, labels, _ = cat.split_observations(obs, spec)
    logpi = np.log(np.asarray(fit["cat_probs"][0]))
    for c in range(len(params)):
        base = mvc._sn_logpdf(X_cont, *params[c])
        expected = base + np.where(labels[:, 0] >= 0, logpi[c, np.maximum(labels[:, 0], 0)], 0.0)
        np.testing.assert_allclose(mvc._comp_logpdf(obs, params, c), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(vf._comp_logpdf(obs, params, c), expected, rtol=0, atol=1e-12)

    # Analysis-side mixture log-likelihood == the fitter's get_likelihood + categorical extra.
    W = np.asarray(fit["weights"])
    comp = np.array([mvc._comp_logpdf(obs, params, c) for c in range(len(params))])
    ll_analysis = sum(logsumexp(np.log(W[s])[:, None] + comp[:, S[:, s]], axis=0).sum()
                      for s in range(S.shape[1]))
    extra = cat.cat_log_pdfs(labels, [np.asarray(P) for P in fit["cat_probs"]])
    ll_fitter = get_likelihood(X_cont, S, plain, W, multivariate=True, extra_log_pdfs=extra)
    np.testing.assert_allclose(ll_analysis, ll_fitter, rtol=1e-9)

    back = pickle.loads(pickle.dumps(params))
    assert isinstance(back, cat.MixtureParams) and back.cat_spec.names == ["kaw"]
    assert _eq(back.cat_probs, params.cat_probs)


def test_downstream_monotonicity_skips_nominal_dims():
    from src.assay_calibration.multivariate_analysis.mv_calibration import MVCalibrationAnalysis
    ms = SimpleNamespace(dataset_names=["a", "kaw", "b"], categorical_dims={"kaw": ["x", "y"]})
    assert MVCalibrationAnalysis._nominal_dims(SimpleNamespace(ms=ms)) == {1}
    ms.categorical_dims = {}
    assert MVCalibrationAnalysis._nominal_dims(SimpleNamespace(ms=ms)) == set()


# ── single fully-observed row: get_likelihood must score the mixture, not (K, K) ──
@pytest.mark.parametrize("Delta", [np.array([0.5, 0.0]), np.array([[0.5, 0.0], [0.0, 0.3]]), np.zeros(2)],
                         ids=["q1", "q2", "gaussian"])
def test_single_row_likelihood_matches_duplicated(Delta):
    from scipy.special import logsumexp
    P = [(np.zeros(2), Delta, np.eye(2)), (np.ones(2), Delta, np.eye(2))]
    W = np.array([[0.3, 0.7]])
    x = np.array([[0.4, -0.2]])
    lp = np.array([_single_component_logpdf(x, p, True) for p in P])
    assert lp.shape == (2, 1)
    expected = float(logsumexp(np.log(W[0])[:, None] + lp, axis=0)[0])
    one = get_likelihood(x, np.ones((1, 1), bool), P, W, multivariate=True)
    two = get_likelihood(np.vstack([x, x]), np.ones((2, 1), bool), P, W, multivariate=True)
    np.testing.assert_allclose(one, expected, rtol=1e-12)
    np.testing.assert_allclose(two, 2 * expected, rtol=1e-12)


# ── measured-variant marginals (visualize_fit, opt-in) ──────────────────────
def test_measured_marginals_reweight_toward_measured_subset():
    from scipy.special import logsumexp
    from src.assay_calibration.multivariate_analysis import visualize_fit as vf
    G = np.eye(2) * 0.3
    params = [(np.array([0.0, 0.0]), np.zeros((2, 2)), G), (np.array([3.0, 3.0]), np.zeros((2, 2)), G)]
    W = np.array([[0.8, 0.2]])
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.5, (80, 2)), rng.normal(3, 0.5, (20, 2))])
    sa = np.ones((100, 1), bool)
    X[:80, 1] = np.nan                       # dim 1 measured only on the comp-2-like rows
    grids = [np.linspace(-2, 5, 50), np.linspace(-2, 5, 50)]
    out = vf._measured_marginals_for_fit({"component_params": params, "weights": W}, X, sa, grids)
    x2 = np.full((50, 2), np.nan)
    x2[:, 1] = grids[1]
    whole = logsumexp(np.array([vf._comp_logpdf(x2, params, c) for c in range(2)]) + np.log(W[0])[:, None], axis=0)
    meas = out[1][0]
    hi = grids[1] > 2
    assert np.exp(meas)[hi].sum() > 3 * np.exp(whole)[hi].sum(), "measured subset is comp-2-like"
    np.testing.assert_allclose(np.trapz(np.exp(out[0][0]), grids[0]), 1.0, atol=0.02)
    # the weights are exactly the mean component posterior over the measured rows
    rows = np.where(~np.isnan(X[:, 1]))[0]
    lp = np.array([vf._comp_logpdf(X[rows], params, c) for c in range(2)]) + np.log(W[0])[:, None]
    rbar = np.exp(lp - logsumexp(lp, axis=0)).mean(1)
    expect = logsumexp(np.array([vf._comp_logpdf(x2, params, c) for c in range(2)]) + np.log(rbar)[:, None], axis=0)
    np.testing.assert_allclose(meas, expect, atol=1e-12)
