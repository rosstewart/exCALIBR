#!/usr/bin/env python3
"""
Quality degradation analysis: how does best val_ll degrade with fewer fits/bootstrap?

For each selected dataset/bootstrap, runs all 100 fits through GPU EM, collects
every fit's val_ll, then estimates E[best of N] for N in {100,64,32,16,8} via
Monte Carlo subsampling without replacement (2000 reps per N).

Results are saved to tests/benchmark_num_fits/ (or --results-dir):
  val_lls.json     — raw val_ll arrays per (task, dataset, bootstrap, label)
  summary.csv      — mean/std Δval_ll per (dataset, bootstrap, label, N)
  run.log          — full stdout (also printed live)

Usage:
    python tests/benchmark_num_fits.py <output_dir> [options]

Examples:
    CUDA_VISIBLE_DEVICES=3 python tests/benchmark_num_fits.py /data/ross/.../my_run
    CUDA_VISIBLE_DEVICES=3 python tests/benchmark_num_fits.py /data/ross/.../my_run \\
        --tasks 25,95,238,432,517,654,913 --bootstraps 3
"""
import argparse
import csv
import json
import pickle
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR_DEFAULT = Path(__file__).resolve().parent / "benchmark_num_fits"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
jax.config.update("jax_enable_x64", True)

NUM_FITS_GRID = [100, 64, 32, 16, 8]
N_SUBSAMPLES = 2000
DEFAULT_TASKS = [25, 95, 238, 432, 517, 654, 913]


# ---------------------------------------------------------------------------

def _build_fit_specs(job_dict, output_dir):
    dataset_name = job_dict["dataset_name"]
    bs_seed = job_dict["bootstrap_seed"]
    shared = job_dict["shared_data"]
    is_mv = job_dict.get("multivariate", False)
    save_dir = str(output_dir / dataset_name)

    fit_specs = []
    for nc_key in sorted(k for k in job_dict if k.startswith("jobs_") and isinstance(job_dict[k], list)):
        label = nc_key[len("jobs_"):]
        for minimal_job in job_dict[nc_key]:
            full_job = {**minimal_job, **shared, "dataset_name": dataset_name, "save_dir": save_dir}
            if is_mv:
                full_job["multivariate"] = True
            if "kwargs" in full_job:
                full_job["kwargs"].pop("multivariate", None)
            fit_specs.append((full_job, bs_seed, label, save_dir))
    return fit_specs


def run_all_fits(job_dict, output_dir):
    """Run all fits for one bootstrap; return {label: np.array of val_lls}."""
    from src.assay_calibration.fit_utils.jax_batch.interop import run_gpu
    fit_specs = _build_fit_specs(job_dict, output_dir)
    results = run_gpu(fit_specs)
    by_label = defaultdict(list)
    for _bs, label, _sd, _idx, result in results:
        vll = result.get("val_ll", -np.inf) if result else -np.inf
        by_label[label].append(vll)
    return {lbl: np.array(vs) for lbl, vs in by_label.items()}


def best_of_n_stats(val_lls, n_fits_grid, rng):
    """Monte Carlo estimate of E[best of N] for each N in n_fits_grid.

    For each N: draw N fits without replacement 2000 times, take max each time,
    return mean ± std. Fits with val_ll=-inf (failed) are excluded first.
    """
    valid = val_lls[np.isfinite(val_lls)]
    n_valid = len(valid)
    baseline = float(valid.max()) if n_valid > 0 else -np.inf

    stats = {}
    for N in n_fits_grid:
        if N >= n_valid:
            stats[N] = (baseline, 0.0)
            continue
        bests = np.array([
            valid[rng.choice(n_valid, size=N, replace=False)].max()
            for _ in range(N_SUBSAMPLES)
        ])
        stats[N] = (float(bests.mean()), float(bests.std()))
    return baseline, n_valid, stats


# ---------------------------------------------------------------------------

class Tee:
    """Write to both a file and stdout simultaneously."""
    def __init__(self, path):
        self._file = open(path, "w")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir")
    parser.add_argument("--tasks", default=",".join(map(str, DEFAULT_TASKS)),
                        help="Comma-separated array task indices (default: diverse sample)")
    parser.add_argument("--bootstraps", type=int, default=3,
                        help="Bootstraps per task (first N from the pkl, default: 3)")
    parser.add_argument("--results-dir", default=str(_RESULTS_DIR_DEFAULT),
                        help="Directory to save results (default: tests/benchmark_num_fits/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    jobs_dir = output_dir / "jobs"
    task_ids = [int(t) for t in args.tasks.split(",")]
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    tee = Tee(results_dir / "run.log")
    sys.stdout = tee

    print(f"num_fits grid : {NUM_FITS_GRID}")
    print(f"Monte Carlo N : {N_SUBSAMPLES} subsamples per (bootstrap, label, num_fits)")
    print(f"Tasks         : {task_ids}")
    print(f"Bootstraps    : {args.bootstraps} per task")
    print(f"Results dir   : {results_dir}")
    print()

    # Raw storage for val_ll arrays — keyed (task_id, dataset, bs_seed, label)
    raw_val_lls = {}

    # Accumulate Δval_ll per label across all bootstraps for the summary
    degradation = defaultdict(lambda: defaultdict(list))

    # CSV rows: one row per (task, dataset, bootstrap, label, N)
    csv_rows = []

    for task_id in task_ids:
        pkl_path = jobs_dir / f"array_{task_id:04d}.pkl"
        if not pkl_path.exists():
            print(f"Task {task_id:4d}: pkl not found — skipping")
            continue

        jobs = pickle.load(open(pkl_path, "rb"))
        selected = jobs[:args.bootstraps]
        dataset = jobs[0]["dataset_name"]
        print(f"Task {task_id:4d}  {dataset}")
        print(f"{'─' * 60}")

        for job_dict in selected:
            bs_seed = job_dict["bootstrap_seed"]
            print(f"  bootstrap {bs_seed}", flush=True)

            val_lls_by_label = run_all_fits(job_dict, output_dir)

            for label, val_lls in sorted(val_lls_by_label.items()):
                key = (task_id, dataset, int(bs_seed), label)
                raw_val_lls[str(key)] = val_lls.tolist()

                baseline, n_valid, stats = best_of_n_stats(val_lls, NUM_FITS_GRID, rng)
                n_failed = len(val_lls) - n_valid
                failed_str = f"  ({n_failed} failed)" if n_failed else ""
                print(f"    [{label}]  baseline (best/100) = {baseline:.4f}{failed_str}")
                for N in NUM_FITS_GRID:
                    mean, std = stats[N]
                    delta = mean - baseline
                    bar = "█" * int(max(0, -delta * 200))  # 0.005 per block
                    print(f"      fits={N:3d}:  {mean:.4f} ± {std:.4f}  Δ={delta:+.4f}  {bar}")
                    csv_rows.append({
                        "task_id": task_id,
                        "dataset": dataset,
                        "bootstrap": int(bs_seed),
                        "label": label,
                        "num_fits": N,
                        "baseline": round(baseline, 6),
                        "mean_best": round(mean, 6),
                        "std_best": round(std, 6),
                        "delta": round(delta, 6),
                        "n_valid": n_valid,
                        "n_failed": n_failed,
                    })
                    degradation[label][N].append(delta)
                print()

        print()

    # ── Save raw val_lls ──────────────────────────────────────────────────────
    with open(results_dir / "val_lls.json", "w") as f:
        json.dump(raw_val_lls, f, indent=2)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if csv_rows:
        with open(results_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # ── Print summary ─────────────────────────────────────────────────────────
    if not degradation:
        sys.stdout = tee._stdout
        tee.close()
        return

    print("=" * 60)
    print("SUMMARY — mean Δval_ll (best_of_N − best_of_100) across all bootstraps")
    print("Negative = degradation from using fewer fits")
    print("=" * 60)

    all_labels = sorted(degradation.keys())
    header = f"{'label':>6}  " + "  ".join(f"fits={N:3d}" for N in NUM_FITS_GRID)
    print(header)
    print("─" * len(header))
    for label in all_labels:
        row = f"{label:>6}  "
        for N in NUM_FITS_GRID:
            deltas = degradation[label][N]
            row += f"  {np.mean(deltas):+.4f}" if deltas else "       —"
        print(row)

    all_deltas = {N: [] for N in NUM_FITS_GRID}
    for label in all_labels:
        for N in NUM_FITS_GRID:
            all_deltas[N].extend(degradation[label][N])
    print("─" * len(header))
    row = f"{'ALL':>6}  "
    for N in NUM_FITS_GRID:
        row += f"  {np.mean(all_deltas[N]):+.4f}" if all_deltas[N] else "       —"
    print(row)

    print()
    print(f"Saved: {results_dir / 'val_lls.json'}")
    print(f"Saved: {results_dir / 'summary.csv'}")
    print(f"Saved: {results_dir / 'run.log'}")

    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
