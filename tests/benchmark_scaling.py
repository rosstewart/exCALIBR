"""Measure GPU kernel time vs batch size, separate from CPU init overhead.

Run with:
    source activate assay_calibration
    CUDA_VISIBLE_DEVICES=1 python tests/benchmark_scaling.py
"""
import sys, time
import numpy as np

sys.path.insert(0, "/home/rcstewart/exCALIBR")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from src.assay_calibration.data_utils.dataset import BasicScoreset
from src.assay_calibration.fit_utils.fit import Fit
from src.assay_calibration.fit_utils.jax_batch.interop import _init_univariate
from src.assay_calibration.fit_utils.jax_batch import batch_em
from src.assay_calibration.fit_utils.jax_batch.constraints_jax import build_grid


def make_jobs(n_fits, seed=0, n_per_sample=200):
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
    ss = BasicScoreset(scores=scores, sample_assignments=sa)
    fitter = Fit(ss)
    return fitter.generate_fit_jobs(
        component_range=[2], bootstrap_seed=0, num_fits=n_fits,
        check_monotonic=False, master_seed=42,
    )


def pack_arrays(jobs):
    """Run CPU init and pack into JAX-ready arrays. Returns (arrays, init_time)."""
    t0 = time.perf_counter()
    inits, obs_list, sidx_list = [], [], []
    for job in jobs:
        init = _init_univariate(job)
        if init is None:
            continue
        inits.append(init)
        obs_list.append(np.asarray(job["train_observations"]).ravel())
        sidx_list.append(np.argmax(job["train_sample_assignments"], axis=1))
    init_time = time.perf_counter() - t0

    obs   = jnp.asarray(np.stack(obs_list))
    sidx  = jnp.asarray(np.stack(sidx_list))
    a0    = jnp.asarray(np.stack([[p[0] for p in init[0]] for init in inits]))
    loc0  = jnp.asarray(np.stack([[p[1] for p in init[0]] for init in inits]))
    scale0= jnp.asarray(np.stack([[p[2] for p in init[0]] for init in inits]))
    W0    = jnp.asarray(np.stack([init[1] for init in inits]))
    S     = inits[0][1].shape[0]
    xmin  = obs.min(axis=1)
    xmax  = obs.max(axis=1)
    return (obs, sidx, S, a0, loc0, scale0, W0, xmin, xmax), init_time


def time_kernel(arrays, n_repeats=3):
    """Time just the JAX fit_batch call (kernel + H2D/D2H). Returns median seconds."""
    obs, sidx, S, a0, loc0, scale0, W0, xmin, xmax = arrays
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = batch_em.fit_batch(obs, sidx, S, a0, loc0, scale0, W0,
                                    xmin, xmax, constrained=False)
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ── JIT warm-up ────────────────────────────────────────────────────────────────
print("Warming up JIT...")
warmup_jobs = make_jobs(20)
warmup_arrays, _ = pack_arrays(warmup_jobs)
time_kernel(warmup_arrays, n_repeats=2)
print("  done\n")

# ── Scaling sweep ──────────────────────────────────────────────────────────────
BATCH_SIZES = [50, 200, 500, 1000, 2000, 5000, 10_000]
CPU_MS_PER_FIT = 2759.5   # from earlier benchmark (single-threaded)
CPU_CORES_700 = 700

print(f"{'batch':>8}  {'init_s':>7}  {'kernel_s':>9}  {'total_s':>8}  "
      f"{'ms/fit':>7}  {'fits/s_gpu':>11}  {'fits/s_700cpu':>14}  {'gpu_vs_700cpu':>14}")
print("-" * 100)

big_jobs = make_jobs(max(BATCH_SIZES))

for B in BATCH_SIZES:
    jobs = big_jobs[:B]
    arrays, init_t = pack_arrays(jobs)
    kernel_t = time_kernel(arrays, n_repeats=3)
    total_t = init_t + kernel_t
    ms_per_fit = total_t / B * 1000
    fits_per_s_gpu = B / total_t
    fits_per_s_700 = CPU_CORES_700 / (CPU_MS_PER_FIT / 1000)
    ratio = fits_per_s_gpu / fits_per_s_700

    print(f"{B:>8}  {init_t:>7.2f}  {kernel_t:>9.2f}  {total_t:>8.2f}  "
          f"{ms_per_fit:>7.1f}  {fits_per_s_gpu:>11.1f}  {fits_per_s_700:>14.1f}  "
          f"{ratio:>13.3f}x")

print()
print("Notes:")
print(f"  CPU single-core reference: {CPU_MS_PER_FIT:.0f} ms/fit")
print(f"  700-core throughput: {fits_per_s_700:.0f} fits/s (assumed perfect parallelism)")
print(f"  'kernel_s' = pure JAX fit_batch (jax.block_until_ready), excludes init")
print(f"  'init_s'   = sequential CPU kmeans+MoM (the current bottleneck)")
print(f"  GPU init would reduce init_s to ~0; kernel_s is the hard floor")
