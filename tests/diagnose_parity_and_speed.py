"""Full parity diagnosis and wall-clock benchmark for the JAX GPU path.

Run with:
    source activate assay_calibration
    CUDA_VISIBLE_DEVICES=1 python tests/diagnose_parity_and_speed.py

Prints:
  1. Per-fit, per-component parameter differences (a, loc, scale) and val_ll
     for unconstrained and constrained univariate fits, and CFUSN fits.
  2. Summary statistics (max/mean absolute and relative differences).
  3. Wall-clock benchmark: CPU vs GPU on a larger synthetic batch.
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/rcstewart/exCALIBR")

import jax
jax.config.update("jax_enable_x64", True)

from src.assay_calibration.data_utils.dataset import BasicScoreset, BasicMultiScoreset
from src.assay_calibration.fit_utils.fit import Fit
from src.assay_calibration.fit_utils.jax_batch.interop import run_gpu


# ── helpers ───────────────────────────────────────────────────────────────────

def make_univariate_scoreset(seed=0, n_per_sample=150):
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


def make_multivariate_scoreset(seed=0, n_per_sample=150):
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


def run_cpu(jobs, dataset_name="test"):
    out = {}
    for job in jobs:
        r = Fit.execute_fit_job({**job, "dataset_name": dataset_name})
        out[job["fit_idx"]] = r
    return out


def run_gpu_dict(jobs, label, dataset_name="test"):
    specs = [({**j, "dataset_name": dataset_name}, j["bootstrap_seed"], label, "") for j in jobs]
    out = {}
    for _, _, _, fit_idx, r in run_gpu(specs):
        out[fit_idx] = r
    return out


def _param_rows(cpu_result, gpu_result, label):
    """Return list of dict rows comparing each component parameter."""
    rows = []
    cp = cpu_result.get("fit", {}).get("component_params", [])
    gp = gpu_result.get("fit", {}).get("component_params", [])
    cpu_ll = cpu_result.get("val_ll", float("nan"))
    gpu_ll = gpu_result.get("val_ll", float("nan"))
    for k, (c_params, g_params) in enumerate(zip(cp, gp)):
        if len(c_params) == 3 and len(g_params) == 3:
            c0, c1, c2 = float(c_params[0]), float(c_params[1]), float(c_params[2])
            g0, g1, g2 = float(g_params[0]), float(g_params[1]), float(g_params[2])
            rows.append({
                "label": label,
                "fit_idx": cpu_result.get("fit_idx"),
                "comp": k,
                "cpu_p0": c0, "gpu_p0": g0, "diff_p0": abs(c0 - g0),
                "cpu_p1": c1, "gpu_p1": g1, "diff_p1": abs(c1 - g1),
                "cpu_p2": c2, "gpu_p2": g2, "diff_p2": abs(c2 - g2),
                "cpu_ll": cpu_ll, "gpu_ll": gpu_ll, "diff_ll": abs(cpu_ll - gpu_ll),
            })
    return rows


def print_parity(rows, p0_name="a", p1_name="loc", p2_name="scale"):
    if not rows:
        print("  (no rows)")
        return
    df = pd.DataFrame(rows)
    # per-row table
    pd.set_option("display.float_format", "{:.6f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)
    show_cols = ["fit_idx","comp",
                 f"cpu_{p0_name}", f"gpu_{p0_name}", f"diff_{p0_name}",
                 f"cpu_{p1_name}", f"gpu_{p1_name}", f"diff_{p1_name}",
                 f"cpu_{p2_name}", f"gpu_{p2_name}", f"diff_{p2_name}",
                 "diff_ll"]
    # rename diff cols for printing
    rename = {"diff_p0": f"diff_{p0_name}", "diff_p1": f"diff_{p1_name}", "diff_p2": f"diff_{p2_name}",
              "cpu_p0": f"cpu_{p0_name}", "gpu_p0": f"gpu_{p0_name}",
              "cpu_p1": f"cpu_{p1_name}", "gpu_p1": f"gpu_{p1_name}",
              "cpu_p2": f"cpu_{p2_name}", "gpu_p2": f"gpu_{p2_name}"}
    df2 = df.rename(columns=rename)
    avail = [c for c in show_cols if c in df2.columns]
    print(df2[avail].to_string(index=False))
    print()
    # summary stats
    for col in ["diff_p0","diff_p1","diff_p2","diff_ll"]:
        nice = col.replace("diff_p0", f"diff_{p0_name}").replace("diff_p1", f"diff_{p1_name}").replace("diff_p2", f"diff_{p2_name}")
        vals = df[col].values
        print(f"  {nice:15s}  max={vals.max():.2e}  mean={vals.mean():.2e}  "
              f"rel_max={( vals / (np.abs(df[col.replace('diff_','cpu_')].values) + 1e-12) ).max():.2e}")


# ── Section 1: parity ─────────────────────────────────────────────────────────

print("=" * 70)
print("SECTION 1: PARITY — unconstrained univariate (5 fits, 2 components)")
print("=" * 70)

ss = make_univariate_scoreset()
fitter = Fit(ss)
jobs = fitter.generate_fit_jobs(component_range=[2], bootstrap_seed=0,
                                 num_fits=5, check_monotonic=False, master_seed=42)
cpu_r = run_cpu(jobs)
gpu_r = run_gpu_dict(jobs, "2c")

rows_unc = []
for fi in sorted(cpu_r):
    rows_unc.extend(_param_rows(cpu_r[fi], gpu_r[fi], "unconstrained"))
print_parity(rows_unc, p0_name="a", p1_name="loc", p2_name="scale")

print()
print("=" * 70)
print("SECTION 2: PARITY — constrained univariate (5 fits, 2 components)")
print("=" * 70)

jobs_c = fitter.generate_fit_jobs(component_range=[2], bootstrap_seed=0,
                                   num_fits=5, check_monotonic=True, master_seed=42)
cpu_rc = run_cpu(jobs_c)
gpu_rc = run_gpu_dict(jobs_c, "2c_con")

rows_con = []
for fi in sorted(cpu_rc):
    rows_con.extend(_param_rows(cpu_rc[fi], gpu_rc[fi], "constrained"))
print_parity(rows_con, p0_name="a", p1_name="loc", p2_name="scale")

print()
print("=" * 70)
print("SECTION 3: PARITY — CFUSN unconstrained (3 fits, 3 components)")
print("  (mu, Delta[0,0], Gamma[0,0] for first dim only)")
print("=" * 70)

ss_mv = make_multivariate_scoreset()
fitter_mv = Fit(ss_mv)
jobs_mv = fitter_mv.generate_fit_jobs(component_range=[3], bootstrap_seed=0,
                                       num_fits=3, check_monotonic=False,
                                       latent_q=2, master_seed=42)
cpu_mv = run_cpu(jobs_mv)
gpu_mv = run_gpu_dict(jobs_mv, "3c_mv")

rows_mv = []
for fi in sorted(cpu_mv):
    cp = cpu_mv[fi].get("fit", {}).get("component_params", [])
    gp = gpu_mv[fi].get("fit", {}).get("component_params", [])
    cpu_ll = cpu_mv[fi].get("val_ll", float("nan"))
    gpu_ll = gpu_mv[fi].get("val_ll", float("nan"))
    for k, (c_p, g_p) in enumerate(zip(cp, gp)):
        if len(c_p) == 3 and len(g_p) == 3:
            # c_p = (mu, Delta, Gamma)
            c_mu = np.asarray(c_p[0])
            g_mu = np.asarray(g_p[0])
            rows_mv.append({
                "fit_idx": fi, "comp": k,
                "cpu_p0": c_mu[0], "gpu_p0": g_mu[0], "diff_p0": abs(c_mu[0]-g_mu[0]),
                "cpu_p1": c_mu[1], "gpu_p1": g_mu[1], "diff_p1": abs(c_mu[1]-g_mu[1]),
                "cpu_p2": float(cpu_ll), "gpu_p2": float(gpu_ll), "diff_p2": abs(cpu_ll-gpu_ll),
                "cpu_ll": cpu_ll, "gpu_ll": gpu_ll, "diff_ll": abs(cpu_ll-gpu_ll),
            })

print_parity(rows_mv, p0_name="mu0", p1_name="mu1", p2_name="ll")

# ── Section 2: wall-clock benchmark ──────────────────────────────────────────

print()
print("=" * 70)
print("SECTION 4: WALL-CLOCK BENCHMARK")
print("=" * 70)

N_FITS = 200   # jobs per batch size to benchmark
N_SAMPLES = 3

def make_bench_jobs(n_fits, constrained=False, seed=7):
    ss_b = make_univariate_scoreset(seed=seed, n_per_sample=200)
    f = Fit(ss_b)
    return f.generate_fit_jobs(component_range=[2], bootstrap_seed=0,
                                num_fits=n_fits, check_monotonic=constrained,
                                master_seed=99)

print(f"\nGenerating {N_FITS} unconstrained 2-component jobs (n=600 per fit)...")
bench_jobs = make_bench_jobs(N_FITS, constrained=False)
print(f"  {len(bench_jobs)} jobs generated")

# warm up JAX JIT
print("Warming up JAX JIT (first call compiles the kernel)...")
warmup_jobs = make_bench_jobs(10)
_ = run_gpu_dict(warmup_jobs, "warmup")
print("  JIT warm-up done")

# CPU timing
print(f"\nCPU path ({N_FITS} fits, sequential via execute_fit_job)...")
t0 = time.perf_counter()
for job in bench_jobs:
    Fit.execute_fit_job({**job, "dataset_name": "bench"})
cpu_time = time.perf_counter() - t0
print(f"  CPU wall time: {cpu_time:.2f}s  ({cpu_time/N_FITS*1000:.1f} ms/fit)")

# GPU timing
print(f"\nGPU path ({N_FITS} fits, batched JAX)...")
t0 = time.perf_counter()
_ = run_gpu_dict(bench_jobs, "bench")
gpu_time = time.perf_counter() - t0
print(f"  GPU wall time: {gpu_time:.2f}s  ({gpu_time/N_FITS*1000:.1f} ms/fit)")

print(f"\n  Speedup: {cpu_time/gpu_time:.1f}×  (CPU {cpu_time:.1f}s vs GPU {gpu_time:.1f}s)")
print()
print("Note: GPU time includes Python overhead (init + unpack) but JIT is already")
print("compiled. CPU is single-threaded (no ProcessPoolExecutor parallelism here).")
print("Real SLURM speedup depends on how many CPU cores vs GPU are available.")
