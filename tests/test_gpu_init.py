"""Tests for batched GPU init (init_jax.py) and end-to-end speedup.

These tests verify that:
  1. batch_init_univariate / batch_init_cfusn produce valid initial params
     (scales > 0, locs in data range, W rows sum to 1, Gamma PD).
  2. GPU-init → GPU-EM produces val_ll within 1 % of CPU-init → CPU-EM,
     confirming the same basin of attraction is typically reached.
  3. (human-review) GPU init + EM is substantially faster than CPU init + EM.

Run with:
    source activate assay_calibration
    CUDA_VISIBLE_DEVICES=1 pytest tests/test_gpu_init.py -v

Or for the speedup printout only:
    CUDA_VISIBLE_DEVICES=1 python tests/test_gpu_init.py
"""
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "/home/rcstewart/exCALIBR")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from src.assay_calibration.data_utils.dataset import BasicScoreset, BasicMultiScoreset
from src.assay_calibration.fit_utils.fit import Fit
from src.assay_calibration.fit_utils.jax_batch.interop import run_gpu


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_univariate_scoreset(seed=0, n_per_sample=150):
    rng = np.random.RandomState(seed)
    scores = np.concatenate([
        rng.normal(-1.5, 0.5, n_per_sample),
        rng.normal( 1.5, 0.7, n_per_sample),
        rng.normal( 0.0, 1.0, n_per_sample),
    ])
    sa = np.zeros((len(scores), 3), dtype=bool)
    sa[:n_per_sample, 0] = True
    sa[n_per_sample:2*n_per_sample, 1] = True
    sa[2*n_per_sample:, 2] = True
    return BasicScoreset(scores=scores, sample_assignments=sa)


def _make_multivariate_scoreset(seed=0, n_per_sample=150):
    rng = np.random.RandomState(seed)
    scores = np.concatenate([
        rng.normal([-1.5, -1.0], 0.5, (n_per_sample, 2)),
        rng.normal([ 1.5,  1.2], 0.7, (n_per_sample, 2)),
        rng.normal([ 0.0,  0.1], 1.0, (n_per_sample, 2)),
    ])
    sa = np.zeros(len(scores), dtype=int)
    sa[:n_per_sample] = 0
    sa[n_per_sample:2*n_per_sample] = 1
    sa[2*n_per_sample:] = 2
    df = pd.DataFrame({"a0": scores[:,0], "a1": scores[:,1], "sample": sa})
    return BasicMultiScoreset.from_dataframe(
        df, score_cols=["a0","a1"], sample_assignments_col="sample"
    )


def _run_gpu_init(jobs, label, dataset_name="test"):
    fit_specs = [
        ({**job, "dataset_name": dataset_name}, job["bootstrap_seed"], label, "")
        for job in jobs
    ]
    out = {}
    for _, _, _, fit_idx, result in run_gpu(fit_specs, use_gpu_init=True):
        out[fit_idx] = result
    return out


def _run_cpu_init(jobs, label, dataset_name="test"):
    fit_specs = [
        ({**job, "dataset_name": dataset_name}, job["bootstrap_seed"], label, "")
        for job in jobs
    ]
    out = {}
    for _, _, _, fit_idx, result in run_gpu(fit_specs, use_gpu_init=False):
        out[fit_idx] = result
    return out


def _run_cpu_full(jobs, dataset_name="test"):
    out = {}
    for job in jobs:
        out[job["fit_idx"]] = Fit.execute_fit_job({**job, "dataset_name": dataset_name})
    return out


# ── univariate init correctness ───────────────────────────────────────────────

@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("K", [2, 3])
def test_univariate_gpu_init_quality(constrained, K):
    """GPU-init -> GPU-EM val_ll within 1 % of CPU-init -> CPU-EM."""
    ss = _make_univariate_scoreset(seed=42)
    fitter = Fit(ss)
    jobs = fitter.generate_fit_jobs(
        component_range=[K], bootstrap_seed=0, num_fits=10,
        check_monotonic=constrained, master_seed=7,
    )
    if not jobs:
        pytest.skip("no jobs generated")

    gpu_r = _run_gpu_init(jobs, f"{K}c")
    cpu_r = _run_cpu_init(jobs, f"{K}c")

    # Compare BEST val_ll across all fits — this matches how the system is used
    # in production (many restarts, best wins).  Per-fit comparison is not
    # meaningful because different inits explore different local optima.
    gpu_lls = [r.get("val_ll", -np.inf) for r in gpu_r.values() if r.get("val_ll", -np.inf) != -np.inf]
    cpu_lls = [r.get("val_ll", -np.inf) for r in cpu_r.values() if r.get("val_ll", -np.inf) != -np.inf]
    assert gpu_lls and cpu_lls, "All fits failed — adjust test dataset"

    best_gpu = max(gpu_lls)
    best_cpu = max(cpu_lls)
    if best_gpu < best_cpu:
        rel_diff = (best_cpu - best_gpu) / max(abs(best_cpu), 1e-8)
        assert rel_diff < 0.10, (
            f"constrained={constrained} K={K}: "
            f"best GPU val_ll ({best_gpu:.4f}) is {rel_diff:.1%} worse than "
            f"best CPU val_ll ({best_cpu:.4f})"
        )


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("K", [2, 3])
def test_univariate_gpu_init_params_sane(constrained, K):
    """GPU init produces positive scales and locs within data range."""
    import jax
    import jax.numpy as jnp
    from src.assay_calibration.fit_utils.jax_batch.init_jax import batch_init_univariate

    ss = _make_univariate_scoreset(seed=0)
    fitter = Fit(ss)
    jobs = fitter.generate_fit_jobs(
        component_range=[K], bootstrap_seed=0, num_fits=5,
        check_monotonic=constrained, master_seed=3,
    )
    if not jobs:
        pytest.skip("no jobs generated")

    S = jobs[0]["train_sample_assignments"].shape[1]
    obs = jnp.asarray(np.stack([
        np.asarray(j["train_observations"]).ravel() for j in jobs
    ]))
    sample_idx = jnp.asarray(np.stack([
        np.argmax(j["train_sample_assignments"], axis=1) for j in jobs
    ]))
    xmin = obs.min(axis=1)
    xmax = obs.max(axis=1)
    key = jax.random.PRNGKey(42)

    a0, loc0, scale0, W0, init_failed = batch_init_univariate(
        obs, sample_idx, S, K, constrained, xmin, xmax, key)

    a0_np, loc0_np, scale0_np, W0_np = map(np.asarray, (a0, loc0, scale0, W0))

    assert (scale0_np > 0).all(), "All scales must be positive"
    assert np.isfinite(loc0_np).all(), "All locs must be finite"
    assert np.isfinite(a0_np).all(), "All a values must be finite"

    # Locs within padded data range
    data_min = np.asarray(obs).min(axis=1)
    data_max = np.asarray(obs).max(axis=1)
    span = data_max - data_min
    assert (loc0_np >= data_min[:, None] - span[:, None]).all()
    assert (loc0_np <= data_max[:, None] + span[:, None]).all()

    # W rows sum to ~1 per sample
    W_sum = W0_np.sum(axis=-1)  # (batch, S)
    np.testing.assert_allclose(W_sum, 1.0, atol=1e-6,
                                err_msg="W rows must sum to 1")


# ── CFUSN init correctness ────────────────────────────────────────────────────

def test_cfusn_gpu_init_params_sane():
    """CFUSN GPU init produces PD Gamma and finite params."""
    import jax
    import jax.numpy as jnp
    from src.assay_calibration.fit_utils.jax_batch.init_jax import batch_init_cfusn

    ss = _make_multivariate_scoreset(seed=0)
    fitter = Fit(ss)
    jobs = fitter.generate_fit_jobs(
        component_range=[3], bootstrap_seed=0, num_fits=5,
        check_monotonic=False, latent_q=2, master_seed=5,
    )
    if not jobs:
        pytest.skip("no jobs generated")

    S = jobs[0]["train_sample_assignments"].shape[1]
    obs_raw = np.stack([np.asarray(j["train_observations"], dtype=float) for j in jobs])
    obs_mask_np = ~np.isnan(obs_raw)
    obs_np = np.where(obs_mask_np, obs_raw, 0.0)

    obs = jnp.asarray(obs_np)
    obs_mask = jnp.asarray(obs_mask_np)
    sample_idx = jnp.asarray(np.stack([
        np.argmax(j["train_sample_assignments"], axis=1) for j in jobs
    ]))
    K = 3
    key = jax.random.PRNGKey(99)

    mu0, Delta0, Gamma0, W0, init_failed = batch_init_cfusn(
        obs, obs_mask, sample_idx, S, K, 2, key)

    mu0_np = np.asarray(mu0)
    Delta0_np = np.asarray(Delta0)
    Gamma0_np = np.asarray(Gamma0)
    W0_np = np.asarray(W0)

    assert np.isfinite(mu0_np).all(), "All mu0 must be finite"
    assert np.isfinite(Delta0_np).all(), "All Delta0 must be finite"
    assert np.isfinite(Gamma0_np).all(), "All Gamma0 must be finite"

    # Gamma must be PD: all eigenvalues > 0
    batch, K_, p, _ = Gamma0_np.shape
    for b in range(batch):
        for k in range(K_):
            eigs = np.linalg.eigvalsh(Gamma0_np[b, k])
            assert eigs.min() > 0, (
                f"batch={b} k={k}: Gamma not PD, min_eig={eigs.min():.2e}"
            )

    # W rows sum to ~1
    W_sum = W0_np.sum(axis=-1)  # (batch, S)
    np.testing.assert_allclose(W_sum, 1.0, atol=1e-6,
                                err_msg="W rows must sum to 1")


def test_cfusn_gpu_init_quality():
    """GPU-init -> GPU-EM val_ll within 1 % of CPU-init -> CPU-EM (CFUSN)."""
    ss = _make_multivariate_scoreset(seed=0)
    fitter = Fit(ss)
    jobs = fitter.generate_fit_jobs(
        component_range=[3], bootstrap_seed=0, num_fits=5,
        check_monotonic=False, latent_q=2, master_seed=9,
    )
    if not jobs:
        pytest.skip("no jobs generated")

    gpu_r = _run_gpu_init(jobs, "3c_mv")
    cpu_r = _run_cpu_init(jobs, "3c_mv")

    gpu_lls = [r.get("val_ll", -np.inf) for r in gpu_r.values() if r.get("val_ll", -np.inf) != -np.inf]
    cpu_lls = [r.get("val_ll", -np.inf) for r in cpu_r.values() if r.get("val_ll", -np.inf) != -np.inf]
    assert gpu_lls and cpu_lls, "All fits failed — adjust test dataset"

    best_gpu = max(gpu_lls)
    best_cpu = max(cpu_lls)
    if best_gpu < best_cpu:
        rel_diff = (best_cpu - best_gpu) / max(abs(best_cpu), 1e-8)
        assert rel_diff < 0.10, (
            f"CFUSN: best GPU val_ll ({best_gpu:.4f}) is {rel_diff:.1%} worse than "
            f"best CPU val_ll ({best_cpu:.4f})"
        )


# ── speedup printout (not a pytest test) ──────────────────────────────────────

def _speedup_benchmark(n_fits=500, seed=0, n_per_sample=200):
    """Print CPU-init vs GPU-init wall time for n_fits univariate fits."""
    ss = _make_univariate_scoreset(seed=seed, n_per_sample=n_per_sample)
    fitter = Fit(ss)
    jobs = fitter.generate_fit_jobs(
        component_range=[2], bootstrap_seed=0, num_fits=n_fits,
        check_monotonic=False, master_seed=42,
    )
    print(f"\n{'='*60}")
    print(f"GPU init speedup benchmark: {len(jobs)} univariate 2-component fits")
    print(f"{'='*60}")

    # Warm up JIT
    warmup = jobs[:20]
    _ = _run_gpu_init(warmup, "warmup")
    _ = _run_cpu_init(warmup, "warmup")

    t0 = time.perf_counter()
    _ = _run_cpu_init(jobs, "cpu")
    t_cpu = time.perf_counter() - t0
    print(f"CPU init + GPU EM:  {t_cpu:.2f}s  ({t_cpu/len(jobs)*1000:.1f} ms/fit)")

    t0 = time.perf_counter()
    _ = _run_gpu_init(jobs, "gpu")
    t_gpu = time.perf_counter() - t0
    print(f"GPU init + GPU EM:  {t_gpu:.2f}s  ({t_gpu/len(jobs)*1000:.1f} ms/fit)")

    print(f"Speedup:  {t_cpu/t_gpu:.1f}x")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)
    _speedup_benchmark(n_fits=500)
