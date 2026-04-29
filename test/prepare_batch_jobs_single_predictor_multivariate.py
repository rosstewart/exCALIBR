"""
Generate batch jobs for multivariate calibration using single-predictor scores.

For each gene with all three predictors (REVEL, MutPred2, AlphaMissense)
available, build a 3D BasicMultiScoreset and produce SLURM array jobs
analogous to prepare_batch_jobs_multivariate.py, but driven by predictor
score CSVs rather than ClinVar-versioned assay scoresets.

Per-predictor CSV layout (same format as prepare_batch_jobs_single_predictor.py):
    {data_dir}/{gene}/{gene}_{predictor}.csv.gz
    columns: protein_variant, score, sample_assignments

Each gene's BasicMultiScoreset is the union of variants by protein_variant ID
across the three predictors; missing dims are NaN and sample assignments are
ORed across predictors.
"""

import sys
import os
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.append("..")

from src.assay_calibration.fit_utils.fit import Fit
from predictor_mv_utils import (
    PREDICTORS,
    PREDICTOR_DATASET_NAMES,
    SAMPLE_COLUMNS,
    DATASET_SUFFIX,
    predictor_dataset_label,
    load_predictor_data,
    build_basic_multi_scoreset,
)


# ============================================================================
# STEP 2: Per-gene job generation (mirrors prepare_batch_jobs_multivariate)
# ============================================================================

def process_gene(gene, predictor_dfs, output_dir, N_BOOTSTRAPS, NUM_FITS,
                 component_range, constraint_modes, latent_q=2,
                 init_strategy="anchored", sample_balance_beta=0.5):
    gene_label = predictor_dataset_label(gene)
    save_dir = f"{output_dir}/{gene_label}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Gene: {gene}")
    print(f"{'=' * 60}")

    ms, info = build_basic_multi_scoreset(gene, predictor_dfs)
    if ms is None:
        print(f"  Skipping {gene}: {info}")
        return None

    K_dim = ms.n_assays
    print(f"  BasicMultiScoreset: {ms.n_variants} variants, {K_dim} dimensions")
    print(f"  Missing: {ms.missing.mean() * 100:.1f}%")
    print(f"  Samples: {dict(zip(ms.sample_names, ms.sample_counts.tolist()))}")

    constrained_flags = []
    if "con" in constraint_modes:
        constrained_flags.append(True)
    if "unc" in constraint_modes:
        constrained_flags.append(False)

    fitter = Fit(ms)
    all_jobs = []

    for bootstrap_iter in range(N_BOOTSTRAPS):
        jobs_by_mode = {}
        for nc in component_range:
            for constrained in constrained_flags:
                mode_key = f"{nc}c_{'con' if constrained else 'unc'}"
                fit_kwargs = {
                    "latent_q": latent_q,
                    "init_strategy": init_strategy,
                    "sample_balance_beta": sample_balance_beta,
                }
                if NUM_FITS is not None:
                    fit_kwargs["num_fits"] = NUM_FITS
                try:
                    jobs = fitter.generate_fit_jobs(
                        component_range=[nc],
                        bootstrap_seed=bootstrap_iter,
                        check_monotonic=constrained,
                        **fit_kwargs,
                    )
                except Exception as e:
                    print(f"  Bootstrap {bootstrap_iter}, {mode_key}: "
                          f"job generation failed ({e})")
                    jobs = []
                jobs_by_mode[mode_key] = jobs

        shared_data = None
        for jobs in jobs_by_mode.values():
            if jobs:
                first = jobs[0]
                shared_data = {
                    "train_observations": first["train_observations"],
                    "train_sample_assignments": first["train_sample_assignments"],
                    "val_observations": first["val_observations"],
                    "val_sample_assignments": first["val_sample_assignments"],
                }
                break
        if shared_data is None:
            continue

        minimal_by_mode = {}
        for mode_key, jobs in jobs_by_mode.items():
            minimal = []
            for job in jobs:
                minimal.append({
                    "job_id": job["job_id"],
                    "bootstrap_seed": job["bootstrap_seed"],
                    "fit_idx": job["fit_idx"],
                    "num_components": job["num_components"],
                    "constrained": job["constrained"],
                    "init_method": job["init_method"],
                    "init_constraint_adjustment": job["init_constraint_adjustment"],
                    "multivariate": True,
                    "kwargs": job["kwargs"],
                })
            minimal_by_mode[mode_key] = minimal

        total_fits = sum(len(v) for v in minimal_by_mode.values())

        consolidated_job = {
            "dataset_name": gene_label,
            "gene": gene,
            "predictors": list(PREDICTOR_DATASET_NAMES[p] for p in PREDICTORS),
            "n_dimensions": K_dim,
            "save_dir": save_dir,
            "bootstrap_seed": bootstrap_iter,
            "shared_data": shared_data,
            "multivariate": True,
            "num_fits_total": total_fits,
        }
        for mode_key, minimal in minimal_by_mode.items():
            consolidated_job[f"jobs_{mode_key}"] = minimal

        all_jobs.append(consolidated_job)

    modes_str = " × ".join([f"{nc}c" for nc in component_range])
    constraint_str = "/".join(constraint_modes)
    if NUM_FITS is None:
        fits_str = f"dynamic ({', '.join(f'K={nc}:{min((2**latent_q)**nc,100)}' for nc in component_range)})"
    else:
        fits_str = str(NUM_FITS)
    print(f"  Generated {len(all_jobs)} bootstrap jobs "
          f"({fits_str} fits × [{modes_str}] × {{{constraint_str}}} each)")
    return all_jobs


# ============================================================================
# STEP 3: Manifest
# ============================================================================

def generate_manifest(output_dir, data_dir, target_array_size=1000, n_jobs=30,
                      genes=None, component_range=None, constraint_modes=None,
                      N_BOOTSTRAPS=200, NUM_FITS=None, latent_q=2,
                      init_strategy="anchored", sample_balance_beta=0.5):
    if component_range is None:
        component_range = [2, 3]
    if constraint_modes is None:
        constraint_modes = ["con", "unc"]

    jobs_dir = f"{output_dir}/jobs"
    os.makedirs(jobs_dir, exist_ok=True)

    print(f"Loading predictor CSVs from {data_dir}...")
    by_gene = load_predictor_data(data_dir, genes=genes)
    if not by_gene:
        print("No predictor data found.")
        return 0, 0

    print(f"\nDiscovered predictor coverage:")
    for gene in sorted(by_gene):
        avail = sorted(by_gene[gene].keys())
        flag = "OK" if all(p in avail for p in PREDICTORS) else "INCOMPLETE"
        print(f"  {gene}: {avail} [{flag}]")

    print(f"\nConfiguration:")
    print(f"  Components: {component_range}")
    print(f"  Constraints: {constraint_modes}")
    print(f"  Bootstraps: {N_BOOTSTRAPS}")
    if NUM_FITS is None:
        print(f"  Fits per config: dynamic (min(4^K,100) for q={latent_q}; "
              + ", ".join(f"K={nc}→{min((2**latent_q)**nc,100)}" for nc in component_range) + ")")
    else:
        print(f"  Fits per config: {NUM_FITS} (override)")
    print(f"  Init strategy: {init_strategy}")
    print(f"  Sample-balance β: {sample_balance_beta}")

    print(f"\nGenerating jobs ({n_jobs} workers)...")
    all_jobs_by_gene = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_gene)(
            gene, predictor_dfs, output_dir, N_BOOTSTRAPS, NUM_FITS,
            component_range, constraint_modes, latent_q=latent_q,
            init_strategy=init_strategy, sample_balance_beta=sample_balance_beta,
        )
        for gene, predictor_dfs in by_gene.items()
    )

    all_jobs = []
    for jobs in all_jobs_by_gene:
        if jobs is not None:
            all_jobs.extend(jobs)

    total_jobs = len(all_jobs)
    if total_jobs == 0:
        print("No valid jobs generated.")
        return 0, 0

    print(f"\nTotal consolidated jobs: {total_jobs:,}")

    jobs_per_array = max(1, total_jobs // target_array_size)
    num_arrays = (total_jobs + jobs_per_array - 1) // jobs_per_array
    print(f"Jobs per array task: {jobs_per_array}")
    print(f"Number of array tasks: {num_arrays}")

    print("\nSaving job files...")
    job_index = []
    for array_idx in range(num_arrays):
        start = array_idx * jobs_per_array
        end = min(start + jobs_per_array, total_jobs)
        array_jobs = all_jobs[start:end]
        with open(f"{jobs_dir}/array_{array_idx:04d}.pkl", "wb") as f:
            pickle.dump(array_jobs, f)
        for local_idx, job in enumerate(array_jobs):
            job_index.append({
                "array_idx": array_idx,
                "local_idx": local_idx,
                "global_idx": start + local_idx,
                "dataset_name": job["dataset_name"],
                "gene": job["gene"],
                "n_dimensions": job["n_dimensions"],
                "bootstrap_seed": job["bootstrap_seed"],
                "num_fits_total": job["num_fits_total"],
            })

    with open(f"{output_dir}/job_index.json", "w") as f:
        json.dump({
            "total_jobs": total_jobs,
            "num_arrays": num_arrays,
            "jobs_per_array": jobs_per_array,
            "component_range": component_range,
            "constraint_modes": constraint_modes,
            "fits_per_component": NUM_FITS if NUM_FITS is not None else "dynamic",
            "predictors": list(PREDICTORS),
            "job_index": job_index,
        }, f, indent=2)
    print(f"Job index saved to: {output_dir}/job_index.json")

    print("\nSummary by gene:")
    counts = defaultdict(int)
    for j in all_jobs:
        counts[j["gene"]] += 1
    for gene, cnt in sorted(counts.items()):
        print(f"  {gene}: {cnt:,} jobs")

    # Use dynamic expected fits for timing estimate when NUM_FITS not overridden
    timing_fits = NUM_FITS if NUM_FITS is not None else max(
        min((2 ** latent_q) ** nc, 100) for nc in component_range
    )
    create_slurm_script(output_dir, num_arrays, jobs_per_array, timing_fits,
                        component_range)
    return total_jobs, num_arrays


# ============================================================================
# STEP 4: SLURM and worker scripts
# ============================================================================

def create_slurm_script(output_dir, num_arrays, jobs_per_array, num_fits,
                        component_range):
    fits_per_array = jobs_per_array * num_fits * len(component_range)
    minutes_per_array = int(fits_per_array / 60) + 30
    hours = min(minutes_per_array // 60, 11)
    minutes = minutes_per_array % 60
    time_str = f"{hours:02d}:{minutes:02d}:00"

    slurm_script = f"""#!/bin/bash
#SBATCH --account=predrag
#SBATCH --job-name=mv_predictors
#SBATCH --output={output_dir}/logs/array_%A_%a.out
#SBATCH --error={output_dir}/logs/array_%A_%a.err
#SBATCH --array=0-{num_arrays - 1}
#SBATCH --time={time_str}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=short

mkdir -p {output_dir}/logs

module load anaconda3/2024.06
source $HOME/.bashrc
conda activate pillar_project

python {output_dir}/run_mv_predictor_array_task.py {output_dir}/jobs $SLURM_ARRAY_TASK_ID

echo "Array task $SLURM_ARRAY_TASK_ID completed"
"""
    script_path = f"{output_dir}/submit_mv_predictor_array.sh"
    with open(script_path, "w") as f:
        f.write(slurm_script)
    os.chmod(script_path, 0o755)

    worker_script = '''import sys
import pickle
import os
sys.path.append("..")
from src.assay_calibration.fit_utils.fit import Fit


def run_mv_predictor_array_task(jobs_dir, array_idx):
    array_file = f"{jobs_dir}/array_{array_idx:04d}.pkl"
    if not os.path.exists(array_file):
        print(f"Error: {array_file} not found")
        sys.exit(1)

    with open(array_file, "rb") as f:
        jobs = pickle.load(f)

    print(f"Array task {array_idx}: {len(jobs)} consolidated MV predictor jobs")

    for job_idx, cjob in enumerate(jobs):
        print(f"\\nJob {job_idx + 1}/{len(jobs)}: {cjob['dataset_name']} "
              f"({cjob['n_dimensions']}D, bootstrap={cjob['bootstrap_seed']})")
        shared = cjob["shared_data"]
        for nc_key in sorted(k for k in cjob if k.startswith("jobs_")):
            nc_jobs = cjob[nc_key]
            mode = nc_key.replace("jobs_", "")
            print(f"  Running {len(nc_jobs)} fits for {mode}...")
            for fit_idx, mjob in enumerate(nc_jobs):
                try:
                    full_job = {**mjob, **shared,
                                "dataset_name": cjob["dataset_name"]}
                    result = Fit.execute_fit_job(full_job)
                    if result is not None:
                        save_path = os.path.join(
                            cjob["save_dir"],
                            f"{mode}_bootstrap_{cjob['bootstrap_seed']}"
                            f"_fit_{fit_idx}.pkl"
                        )
                        with open(save_path, "wb") as f:
                            pickle.dump(result, f)
                    if (fit_idx + 1) % 25 == 0:
                        print(f"    {fit_idx + 1}/{len(nc_jobs)} done")
                except Exception as e:
                    print(f"    Fit {fit_idx} failed: {e}")
                    continue
        print(f"  Done bootstrap {cjob['bootstrap_seed']}")

    print(f"\\nArray task {array_idx} complete!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_mv_predictor_array_task.py <jobs_dir> <array_idx>")
        sys.exit(1)
    run_mv_predictor_array_task(sys.argv[1], int(sys.argv[2]))
'''
    worker_path = f"{output_dir}/run_mv_predictor_array_task.py"
    with open(worker_path, "w") as f:
        f.write(worker_script)

    print(f"\nSLURM script: {script_path}")
    print(f"Worker script: {worker_path}")
    print(f"Estimated time per array task: {time_str}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup HPC job array for multivariate predictor calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All genes with all 3 predictors, 2c/3c, both constraint modes:
  python prepare_batch_jobs_single_predictor_multivariate.py

  # BRCA1 only, 3 components, unconstrained only:
  python prepare_batch_jobs_single_predictor_multivariate.py \\
      --genes BRCA1 --components 3 --constraints unc
        """,
    )
    parser.add_argument("--genes", nargs="+", default=None,
                        help="Specific gene(s) to process. Default: all.")
    parser.add_argument("--components", nargs="+", type=int, default=None,
                        help="Component counts to fit (default: 2 3).")
    parser.add_argument("--constraints", nargs="+",
                        choices=["con", "unc", "both"], default=["both"],
                        help="Constraint modes (default: both).")
    parser.add_argument("--target-array-size", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=30)
    parser.add_argument("--data-dir", type=str,
                        default="/data/ross/assay_calibration/predictor_scores/"
                                "single_gene_calibration_data",
                        help="Directory containing {gene}/{gene}_{predictor}.csv.gz")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: auto from args).")
    parser.add_argument("--n-bootstraps", type=int, default=200)
    parser.add_argument("--num-fits", type=int, default=None,
                        help="Override dynamic NUM_FITS (default: min(4^K,100) per component count).")
    parser.add_argument("--init-strategy", type=str, default="anchored",
                        choices=["anchored", "kmeans"],
                        help="Initialization strategy. 'anchored' (default) initialises each "
                             "component from its anchor sample; 'kmeans' uses joint k-means.")
    parser.add_argument("--sample-balance-beta", type=float, default=0.5,
                        help="Sample-balanced M-step strength β ∈ [0,1]. 0=off (status quo), "
                             "1=each sample contributes equally to component params. Default: 0.5.")
    args = parser.parse_args()

    constraint_modes = set()
    for c in args.constraints:
        if c == "both":
            constraint_modes.update(["con", "unc"])
        else:
            constraint_modes.add(c)
    constraint_modes = sorted(constraint_modes)

    component_range = args.components or [2, 3]

    base_dir = "/data/ross/assay_calibration/explorer_jobs_predictors_multivariate"
    gene_part = ("_".join(sorted(g.upper() for g in args.genes))
                 if args.genes else "all")
    comp_part = "c" + "-".join(str(c) for c in sorted(component_range))
    const_part = "+".join(sorted(constraint_modes))
    run_name = f"{gene_part}_{comp_part}_{const_part}"

    output_dir = args.output_dir or f"{base_dir}/{run_name}"

    print("=" * 80)
    print("HPC Job Array Setup — Multivariate Predictor Calibration")
    print("=" * 80)
    print(f"  Genes:       {args.genes or 'all'}")
    print(f"  Components:  {component_range}")
    print(f"  Constraints: {constraint_modes}")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Output:      {output_dir}")
    print()

    total_jobs, num_arrays = generate_manifest(
        output_dir=output_dir,
        data_dir=args.data_dir,
        target_array_size=args.target_array_size,
        n_jobs=args.n_jobs,
        genes=args.genes,
        component_range=component_range,
        constraint_modes=constraint_modes,
        N_BOOTSTRAPS=args.n_bootstraps,
        NUM_FITS=args.num_fits,
        init_strategy=args.init_strategy,
        sample_balance_beta=args.sample_balance_beta,
    )

    print(f"\n{'=' * 80}")
    print("Setup complete!")
    print(f"{'=' * 80}")
    print(f"  Total consolidated jobs: {total_jobs:,}")
    print(f"  Array tasks: {num_arrays:,}")
    print(f"\nNext steps:")
    print(f"  1. Review: {output_dir}/submit_mv_predictor_array.sh")
    print(f"  2. Submit: cd {output_dir} && sbatch submit_mv_predictor_array.sh")
    print(f"  3. Monitor: check {output_dir}/logs/")
