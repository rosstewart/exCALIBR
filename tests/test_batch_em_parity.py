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
    for bs_seed, lbl, _save_dir, fit_idx, result in run_gpu(fit_specs):
        out[fit_idx] = result
    return out


@pytest.mark.parametrize("constrained", [False, True])
def test_univariate_parity(constrained):
    scoreset = _make_univariate_scoreset()
    fitter = Fit(scoreset)
    jobs = fitter.generate_fit_jobs(
        component_range=[2], bootstrap_seed=0, num_fits=5,
        check_monotonic=constrained,
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
            np.testing.assert_allclose([a1, l1, s1], [a2, l2, s2], rtol=1e-2, atol=1e-2)
        np.testing.assert_allclose(cpu_result["val_ll"], gpu_result["val_ll"], rtol=1e-2, atol=1e-2)

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
        check_monotonic=False, latent_q=2,
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
            # Component means are the least-approximated quantity here (the
            # big-M/bvn-cdf approximations bias density values, not the
            # location updates) -- start parity-checking with these.
            np.testing.assert_allclose(mu1, mu2, rtol=0.1, atol=0.2)

    assert n_compared > 0, "every fit failed on both paths -- test dataset needs adjusting"
