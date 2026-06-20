import sys
import pickle
import numpy as np
import os
import concurrent.futures

sys.path.append("/home/stewart.ro/assay_calibration")
from src.assay_calibration.fit_utils.fit import Fit


def run_consolidated(consolidated_job, job_idx, total_jobs):
    print(f"\n{'='*70}")
    print(f"Job {job_idx+1}/{total_jobs}")
    print(f"  Dataset: {consolidated_job['dataset_name']}")
    print(f"  Bootstrap: {consolidated_job['bootstrap_seed']}")
    print(f"  Total fits: {consolidated_job['num_fits_total']}")
    is_mv = consolidated_job.get('multivariate', False)
    if is_mv:
        print(f"  Mode: MULTIVARIATE ({consolidated_job.get('n_dimensions', '?')}D)")
        print(f"  Assays: {consolidated_job.get('assay_datasets', [])}")
    else:
        print(f"  Mode: univariate")
    print('=' * 70)

    # Uncalibratable datasets
    skip_datasets = ['F9_Popp_2025_model']
    if consolidated_job['dataset_name'] in skip_datasets:
        return

    # Adjust save path for explorer
    consolidated_job['save_dir'] = consolidated_job['save_dir'].replace(
        '/data/ross', '/projects/pedjas_lab/stewart.ro'
    )
    shared_data = consolidated_job['shared_data']
    save_dir = consolidated_job['save_dir']
    bootstrap_seed = consolidated_job['bootstrap_seed']

    save_file = f"{save_dir}/bootstrap_{bootstrap_seed}_best_fits.pkl"

    # Discover which modes are in this job
    # Supports both old format (jobs_2c, jobs_3c) and new (jobs_2c_con, jobs_2c_unc, etc.)
    nc_keys = sorted([
        k for k in consolidated_job
        if k.startswith("jobs_") and isinstance(consolidated_job[k], list)
    ])
    # e.g. ['jobs_2c', 'jobs_3c'] or ['jobs_2c_con', 'jobs_2c_unc', 'jobs_3c_con', 'jobs_3c_unc']

    # Derive result labels: "jobs_2c" -> "2c", "jobs_3c_con" -> "3c_con"
    nc_labels = [k.replace("jobs_", "") for k in nc_keys]

    # Check for existing results
    old_results = {}
    if os.path.exists(save_file):
        try:
            with open(save_file, 'rb') as f:
                old_results = pickle.load(f)
        except Exception:
            old_results = {}

        # Check if all modes already have results
        all_done = all(
            label in old_results and old_results[label] is not None
            for label in nc_labels
        )
        if all_done:
            return

    # Initialize best results from old or fresh
    best_results = {}
    best_val_lls = {}
    for label in nc_labels:
        if label in old_results and old_results[label] is not None:
            best_results[label] = old_results[label]
            best_val_lls[label] = old_results[label].get('val_ll', -np.inf)
        else:
            best_results[label] = None
            best_val_lls[label] = -np.inf

    # Run fits for each mode
    for nc_key, label in zip(nc_keys, nc_labels):
        # Skip if already have result
        if best_results[label] is not None:
            continue

        jobs_list = consolidated_job[nc_key]
        if not jobs_list:
            continue

        print(f"\n  Running {len(jobs_list)} fits for {label}...")

        for fit_idx, minimal_job in enumerate(jobs_list):
            try:
                full_job = {**minimal_job, **shared_data}
                full_job['dataset_name'] = consolidated_job['dataset_name']
                full_job['save_dir'] = consolidated_job['save_dir']
                if 'dataset_file' in consolidated_job:
                    full_job['dataset_file'] = consolidated_job['dataset_file']
                # Set multivariate at top level (execute_fit_job reads this)
                if is_mv:
                    full_job['multivariate'] = True
                # Remove from kwargs to avoid duplicate keyword arg in tryToFit
                if 'kwargs' in full_job and 'multivariate' in full_job['kwargs']:
                    del full_job['kwargs']['multivariate']

                result = Fit.execute_fit_job(full_job)

                if result is not None and result.get('val_ll') is not None:
                    if result['val_ll'] > best_val_lls[label]:
                        best_results[label] = result
                        best_val_lls[label] = result['val_ll']
                elif result is not None and best_results[label] is None:
                    best_results[label] = result

                if (fit_idx + 1) % 10 == 0:
                    print(f"    Completed {fit_idx+1}/{len(jobs_list)} ({label})")
            except Exception as e:
                print(f"    ✗ {label} fit {fit_idx} failed: {e}")
                continue

    # Save
    os.makedirs(save_dir, exist_ok=True)
    with open(save_file, 'wb') as f:
        pickle.dump(best_results, f)
    print(f"\n  ✓ Saved bootstrap {bootstrap_seed} -> {save_file}")


def run_array_task(jobs_dir, array_idx):
    array_file = f"{jobs_dir}/array_{array_idx:04d}.pkl"
    if not os.path.exists(array_file):
        print(f"Error: Array file {array_file} not found")
        sys.exit(1)

    with open(array_file, 'rb') as f:
        jobs = pickle.load(f)

    print(f"Array task {array_idx}: Processing {len(jobs)} consolidated jobs")
    total_fits = sum(job['num_fits_total'] for job in jobs)
    print(f"Total fits to run: {total_fits}")

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as ex:
        futures = []
        for job_idx, job in enumerate(jobs):
            futures.append(ex.submit(run_consolidated, job, job_idx, len(jobs)))
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"Job failed with error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_array_task.py <jobs_dir> <array_idx>")
        sys.exit(1)

    jobs_dir = sys.argv[1]
    array_idx = int(sys.argv[2])
    run_array_task(jobs_dir, array_idx)
