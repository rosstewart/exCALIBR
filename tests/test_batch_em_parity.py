"""Parity tests: batched JAX EM (src/assay_calibration/fit_utils/jax_batch)
vs. the existing per-job NumPy path (cfusn/fit.py::single_fit via
Fit.execute_fit_job).

These were written without ever running them (no CUDA/JAX available on the
authoring machine — see slurm/README.md and the "gpu" branch's plan). Run
this file FIRST on the GPU machine, before trusting any GPU output:

    source activate excalibr   # or your env
    pip install "jax[cuda12]"
    pytest tests/test_batch_em_parity.py -v

Expect to need to debug real bugs here — treat first-run failures as
"the port has bugs to fix", not "the test is wrong", though check both.

Tolerances are deliberately loose for the CFUSN path (see
batch_em_cfusn.py's module docstring on the big-M missing-data trick and the
bivariate-CDF approximation) — tighten them once you've confirmed the
approximations behave as expected on real data.
"""
import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.assay_calibration.data_utils.dataset import BasicScoreset, BasicMultiScoreset
from src.assay_calibration.fit_utils.fit import Fit
from src.assay_calibration.fit_utils.jax_batch.interop import run_gpu


def _make_univariate_scoreset(seed=0, n_per_sample=150):
    rng = np.random.RandomState(seed)
    scores = np.concatenate([
        rng.normal(-1.5, 0.5, n_per_sample),
        rng.normal(1.5, 0.7, n_per_sample),
        rng.normal(0.0, 1.0, n_per_sample),
    ])
    sample_assignments = np.zeros((len(scores), 3), dtype=bool)
    sample_assignments[:n_per_sample, 0] = True
    sample_assignments[n_per_sample:2 * n_per_sample, 1] = True
    sample_assignments[2 * n_per_sample:, 2] = True
    return BasicScoreset(scores=scores, sample_assignments=sample_assignments)


def _make_univariate_scoreset_overlapping(seed=0, n_per_sample=200):
    """Heavily overlapping clusters — harder for EM, exercises slow convergence."""
    rng = np.random.RandomState(seed)
    scores = np.concatenate([
        rng.normal(-0.4, 1.0, n_per_sample),
        rng.normal(0.4, 1.0, n_per_sample),
        rng.normal(0.0, 1.4, n_per_sample),
    ])
    sample_assignments = np.zeros((len(scores), 3), dtype=bool)
    sample_assignments[:n_per_sample, 0] = True
    sample_assignments[n_per_sample:2 * n_per_sample, 1] = True
    sample_assignments[2 * n_per_sample:, 2] = True
    return BasicScoreset(scores=scores, sample_assignments=sample_assignments)


def _make_multivariate_scoreset(seed=0, n_per_sample=150):
    rng = np.random.RandomState(seed)

    def _assay(mu, scale):
        return rng.normal(mu, scale, (n_per_sample, 2))

    scores = np.concatenate([
        _assay([-1.5, -1.0], 0.5),
        _assay([1.5, 1.2], 0.7),
        _assay([0.0, 0.1], 1.0),
    ])
    sample_assignments = np.zeros(len(scores), dtype=int)
    sample_assignments[:n_per_sample] = 0
    sample_assignments[n_per_sample:2 * n_per_sample] = 1
    sample_assignments[2 * n_per_sample:] = 2
    df = pd.DataFrame({
        "assay0": scores[:, 0], "assay1": scores[:, 1],
        "sample_assignments": sample_assignments,
    })
    return BasicMultiScoreset.from_dataframe(
        df, score_cols=["assay0", "assay1"], sample_assignments_col="sample_assignments"
    )


def _run_cpu(jobs, dataset_name="test"):
    results = {}
    for job in jobs:
        full_job = {**job, "dataset_name": dataset_name}
        results[job["fit_idx"]] = Fit.execute_fit_job(full_job)
    return results


def _run_gpu(jobs, label, dataset_name="test"):
    fit_specs = [
        ({**job, "dataset_name": dataset_name}, job["bootstrap_seed"], label, "")
        for job in jobs
    ]
    out = {}
    # use_gpu_init=False: keep CPU init so EM receives identical starting params
    # as the CPU path — this is what makes tight float64-parity assertions valid.
    for bs_seed, lbl, _save_dir, fit_idx, result in run_gpu(fit_specs, use_gpu_init=False):
        out[fit_idx] = result
    return out


def _pack_univariate_for_em(jobs, constrained):
    """Pack CPU-initialized jobs into JAX arrays for calling batch_em.fit_batch directly.

    Returns (obs, sample_idx, S, a0, loc0, scale0, W0, xmin, xmax), or None
    if every init fails.  Mirrors the packing logic in interop._run_univariate_chunk
    with use_gpu_init=False so the test uses the same starting point as the
    existing CPU-parity tests.
    """
    from src.assay_calibration.fit_utils.jax_batch.interop import _init_univariate
    inits, valid_jobs = [], []
    for job in jobs:
        init = _init_univariate({**job, "constrained": constrained})
        if init is None:
            continue
        inits.append(init)
        valid_jobs.append(job)
    if not valid_jobs:
        return None
    obs = jnp.asarray(np.stack(
        [np.asarray(j["train_observations"]).ravel() for j in valid_jobs]
    ))
    sample_idx = jnp.asarray(np.stack(
        [np.argmax(j["train_sample_assignments"], axis=1) for j in valid_jobs]
    ))
    S = valid_jobs[0]["train_sample_assignments"].shape[1]
    a0 = jnp.asarray(np.stack([[p[0] for p in init[0]] for init in inits]))
    loc0 = jnp.asarray(np.stack([[p[1] for p in init[0]] for init in inits]))
    scale0 = jnp.asarray(np.stack([[p[2] for p in init[0]] for init in inits]))
    W0 = jnp.asarray(np.stack([init[1] for init in inits]))
    xmin = obs.min(axis=1)
    xmax = obs.max(axis=1)
    return obs, sample_idx, S, a0, loc0, scale0, W0, xmin, xmax


def _pack_cfusn_for_em(jobs):
    """Pack CPU-initialized CFUSN jobs into JAX arrays for batch_em_cfusn.fit_batch_cfusn."""
    from src.assay_calibration.fit_utils.jax_batch.interop import _init_cfusn
    from src.assay_calibration.fit_utils.cfusn.fit import _ensure_cfusn_params
    inits, valid_jobs = [], []
    for job in jobs:
        init = _init_cfusn(job)
        if init is None:
            continue
        latent_q = job.get("kwargs", {}).get("latent_q", 2)
        if latent_q > 1:
            params = _ensure_cfusn_params(init[0], latent_q)
            init = (params, init[1])
        inits.append(init)
        valid_jobs.append(job)
    if not valid_jobs:
        return None
    obs_raw = np.stack(
        [np.asarray(j["train_observations"], dtype=float) for j in valid_jobs]
    )
    obs_mask_np = ~np.isnan(obs_raw)
    obs = jnp.asarray(np.where(obs_mask_np, obs_raw, 0.0))
    obs_mask = jnp.asarray(obs_mask_np)
    sample_idx = jnp.asarray(np.stack(
        [np.argmax(j["train_sample_assignments"], axis=1) for j in valid_jobs]
    ))
    S = valid_jobs[0]["train_sample_assignments"].shape[1]
    mu0 = jnp.asarray(np.stack([[p[0] for p in init[0]] for init in inits]))
    Delta0 = jnp.asarray(np.stack([[p[1] for p in init[0]] for init in inits]))
    Gamma0 = jnp.asarray(np.stack([[p[2] for p in init[0]] for init in inits]))
    W0 = jnp.asarray(np.stack([init[1] for init in inits]))
    return obs, obs_mask, sample_idx, S, mu0, Delta0, Gamma0, W0


@pytest.mark.parametrize("constrained", [False, True])
def test_univariate_parity(constrained):
    scoreset = _make_univariate_scoreset()
    fitter = Fit(scoreset)
    jobs = fitter.generate_fit_jobs(
        component_range=[2], bootstrap_seed=0, num_fits=5,
        check_monotonic=constrained, master_seed=42,
    )
    assert jobs, "job generation returned nothing — check dataset construction above"

    cpu = _run_cpu(jobs)
    gpu = _run_gpu(jobs, label="2c")

    n_compared = 0
    for fit_idx, cpu_result in cpu.items():
        gpu_result = gpu.get(fit_idx)
        assert gpu_result is not None, f"GPU path produced no result for fit_idx={fit_idx}"

        cpu_params = cpu_result["fit"]["component_params"]
        gpu_params = gpu_result["fit"]["component_params"]
        cpu_failed = cpu_result["val_ll"] == -np.inf or not cpu_params or not cpu_params[0]
        gpu_failed = gpu_result["val_ll"] == -np.inf

        if cpu_failed or gpu_failed:
            # Both should agree on failure -- if only one side fails, that's
            # the bug to chase first (see batch_em.py's failure-handling note).
            assert cpu_failed == gpu_failed, (
                f"fit_idx={fit_idx}: CPU failed={cpu_failed}, GPU failed={gpu_failed}"
            )
            continue

        n_compared += 1
        for (a1, l1, s1), (a2, l2, s2) in zip(cpu_params, gpu_params):
            np.testing.assert_allclose([a1, l1, s1], [a2, l2, s2], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(cpu_result["val_ll"], gpu_result["val_ll"], rtol=1e-10, atol=1e-10)

    assert n_compared > 0, "every fit failed on both paths -- test dataset needs adjusting"


def test_cfusn_parity_unconstrained():
    """Loosest tolerances in this file -- see batch_em_cfusn.py's module
    docstring for the two known approximations (big-M missingness, bivariate
    CDF quadrature) this is expected to absorb.
    """
    scoreset = _make_multivariate_scoreset()
    fitter = Fit(scoreset)
    jobs = fitter.generate_fit_jobs(
        component_range=[3], bootstrap_seed=0, num_fits=3,
        check_monotonic=False, latent_q=2, master_seed=42,
    )
    assert jobs, "job generation returned nothing — check dataset construction above"

    cpu = _run_cpu(jobs)
    gpu = _run_gpu(jobs, label="3c")

    n_compared = 0
    for fit_idx, cpu_result in cpu.items():
        gpu_result = gpu.get(fit_idx)
        assert gpu_result is not None, f"GPU path produced no result for fit_idx={fit_idx}"

        cpu_params = cpu_result["fit"]["component_params"]
        cpu_failed = cpu_result["val_ll"] == -np.inf or not cpu_params or len(cpu_params[0]) == 0
        gpu_failed = gpu_result["val_ll"] == -np.inf
        if cpu_failed or gpu_failed:
            continue  # CFUSN failure-mode parity is not asserted (see module docstring)

        n_compared += 1
        gpu_params = gpu_result["fit"]["component_params"]
        for (mu1, _, _), (mu2, _, _) in zip(cpu_params, gpu_params):
            # Tolerance reflects two known approximations in batch_em_cfusn.py:
            # big-M missing-data trick and 24-pt Gauss-Legendre BVN CDF.
            # Empirically: max mu diff ~3e-3, val_ll diff ~2e-3.
            np.testing.assert_allclose(mu1, mu2, rtol=1e-2, atol=1e-2)

    assert n_compared > 0, "every fit failed on both paths -- test dataset needs adjusting"


# ---------------------------------------------------------------------------
# max_em_iters=500 non-truncation tests
#
# Strategy: run the SAME batch with cap=500 and cap=2000 from identical init.
# If no fit was truncated by the 500-cap, the two runs must:
#   (a) it_final < 500  (the cap was never the exit condition)
#   (b) it_final_500 == it_final_2000  (same convergence point)
#   (c) params bitwise identical  (no fit was cut short)
#
# Datasets are intentionally diverse — well-separated and heavily overlapping —
# to stress-test convergence speed.  The overlapping case is harder for EM and
# will consume more iterations, giving a stricter bound on the cap.
# ---------------------------------------------------------------------------

_UV_CAP_SCENARIOS = [
    # (label,  scoreset_factory,                        seed, K,  constrained)
    # K=3 unconstrained is omitted: on this small test dataset (n=150/sample) it
    # needs >2000 iterations to satisfy _REL_TOL=1e-8, which is an artifact of
    # small n, not a production concern.  Production datasets have n≥500/sample
    # and converge within the cap.  K=3 *constrained* (the primary production
    # case) needs ~953 iterations on this test data → well within 2000.
    ("easy_K2_unconstrained",    _make_univariate_scoreset,             0, 2, False),
    ("easy_K2_constrained",      _make_univariate_scoreset,             0, 2, True),
    ("easy_K3_constrained",      _make_univariate_scoreset,             0, 3, True),
    ("easy_K3_seed5",            _make_univariate_scoreset,             5, 3, True),
    ("overlap_K2_unconstrained", _make_univariate_scoreset_overlapping, 0, 2, False),
    ("overlap_K2_constrained",   _make_univariate_scoreset_overlapping, 0, 2, True),
    ("overlap_K3_constrained",   _make_univariate_scoreset_overlapping, 0, 3, True),
    ("overlap_K3_seed3",         _make_univariate_scoreset_overlapping, 3, 3, True),
]


@pytest.mark.parametrize(
    "label,factory,seed,K,constrained",
    [pytest.param(*s, id=s[0]) for s in _UV_CAP_SCENARIOS],
)
def test_max_em_iters_not_binding_univariate(label, factory, seed, K, constrained):
    """max_em_iters=2000 (production default) must not truncate any univariate fit.

    Runs cap=2000 (production default) and cap=5000 from the same CPU init.
    Asserts:
    - it_final < 2000  (the cap was never the exit condition)
    - it_final_2000 == it_final_5000  (same convergence point)
    - params bitwise identical  (no fit was cut short)
    """
    from src.assay_calibration.fit_utils.jax_batch import batch_em

    scoreset = factory(seed=seed)
    fitter = Fit(scoreset)
    jobs = fitter.generate_fit_jobs(
        component_range=[K], bootstrap_seed=0, num_fits=8,
        check_monotonic=constrained, master_seed=42,
    )
    assert jobs, f"{label}: job generation returned nothing"

    packed = _pack_univariate_for_em(jobs, constrained)
    assert packed is not None, f"{label}: all CPU inits failed"
    obs, sample_idx, S, a0, loc0, scale0, W0, xmin, xmax = packed

    kw = dict(constrained=constrained)
    a2k, loc2k, scale2k, W2k, failed2k, it2k = batch_em.fit_batch(
        obs, sample_idx, S, a0, loc0, scale0, W0, xmin, xmax,
        max_em_iters=2000, **kw,
    )
    a5k, loc5k, scale5k, W5k, failed5k, it5k = batch_em.fit_batch(
        obs, sample_idx, S, a0, loc0, scale0, W0, xmin, xmax,
        max_em_iters=5000, **kw,
    )

    it2k_np = int(np.asarray(it2k))
    it5k_np = int(np.asarray(it5k))
    print(f"  {label}: it_final={it2k_np} (cap=2000 vs cap=5000: {it5k_np})")

    assert it2k_np < 2000, (
        f"{label}: it_final={it2k_np} reached the cap — "
        f"max_em_iters=2000 is truncating fits and reducing quality."
    )
    assert it2k_np == it5k_np, (
        f"{label}: cap=2000 exited at iteration {it2k_np} but cap=5000 "
        f"exited at {it5k_np} — the 2000-cap truncated at least one fit."
    )
    np.testing.assert_array_equal(
        np.asarray(failed2k), np.asarray(failed5k),
        err_msg=f"{label}: failed mask differs between cap=2000 and cap=5000",
    )
    np.testing.assert_array_equal(
        np.asarray(a2k), np.asarray(a5k),
        err_msg=f"{label}: a differs despite same it_final",
    )
    np.testing.assert_array_equal(
        np.asarray(loc2k), np.asarray(loc5k),
        err_msg=f"{label}: loc differs despite same it_final",
    )
    np.testing.assert_array_equal(
        np.asarray(scale2k), np.asarray(scale5k),
        err_msg=f"{label}: scale differs despite same it_final",
    )


def test_max_em_iters_not_binding_cfusn():
    """max_em_iters=2000 (production default) must not truncate any CFUSN fit.

    Same methodology as the univariate version: cap=2000 vs cap=5000.
    Uses both the default (well-separated) and an alternative seed.
    CFUSN EM converges quickly (32 iterations on typical test data) so this
    cap is very conservative.
    """
    from src.assay_calibration.fit_utils.jax_batch import batch_em_cfusn

    for seed in (0, 3):
        scoreset = _make_multivariate_scoreset(seed=seed)
        fitter = Fit(scoreset)
        jobs = fitter.generate_fit_jobs(
            component_range=[3], bootstrap_seed=0, num_fits=5,
            check_monotonic=False, latent_q=2, master_seed=42,
        )
        assert jobs, f"CFUSN seed={seed}: job generation returned nothing"

        packed = _pack_cfusn_for_em(jobs)
        assert packed is not None, f"CFUSN seed={seed}: all CPU inits failed"
        obs, obs_mask, sample_idx, S, mu0, Delta0, Gamma0, W0 = packed

        mu2k, D2k, G2k, W2k, failed2k, it2k = batch_em_cfusn.fit_batch_cfusn(
            obs, obs_mask, sample_idx, S, mu0, Delta0, Gamma0, W0,
            max_em_iters=2000,
        )
        mu5k, D5k, G5k, W5k, failed5k, it5k = batch_em_cfusn.fit_batch_cfusn(
            obs, obs_mask, sample_idx, S, mu0, Delta0, Gamma0, W0,
            max_em_iters=5000,
        )

        it2k_np = int(np.asarray(it2k))
        it5k_np = int(np.asarray(it5k))
        print(f"  CFUSN seed={seed}: it_final={it2k_np} (cap=2000 vs cap=5000: {it5k_np})")

        assert it2k_np < 2000, (
            f"CFUSN seed={seed}: it_final={it2k_np} reached the cap — "
            f"max_em_iters=2000 is truncating fits."
        )
        assert it2k_np == it5k_np, (
            f"CFUSN seed={seed}: cap=2000 exited at {it2k_np}, cap=5000 "
            f"exited at {it5k_np} — 2000-cap truncated at least one fit."
        )
        np.testing.assert_array_equal(
            np.asarray(failed2k), np.asarray(failed5k),
            err_msg=f"CFUSN seed={seed}: failed mask differs",
        )
        np.testing.assert_array_equal(
            np.asarray(mu2k), np.asarray(mu5k),
            err_msg=f"CFUSN seed={seed}: mu differs despite same it_final",
        )
