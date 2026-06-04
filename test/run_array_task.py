import sys
import pickle
import numpy as np
import os
import concurrent.futures
from collections import defaultdict

sys.path.append("/home/stewart.ro/assay_calibration")
from src.assay_calibration.fit_utils.fit import Fit


def run_one_fit(fit_spec):
    """Run a single fit. Module-level so ProcessPoolExecutor can pickle it.

    Returns (bootstrap_seed, label, save_dir, result_or_None).
    """
    full_job, bootstrap_seed, label, save_dir = fit_spec
    try:
        result = Fit.execute_fit_job(full_job)
        return (bootstrap_seed, label, save_dir, result)
    except Exception as e:
        return (bootstrap_seed, label, save_dir, None)


def run_array_task(jobs_dir, array_idx):
    array_file = f"{jobs_dir}/array_{array_idx:04d}.pkl"
    if not os.path.exists(array_file):
        print(f"Error: Array file {array_file} not found")
        sys.exit(1)

    with open(array_file, 'rb') as f:
        jobs = pickle.load(f)

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    total_fits = sum(job['num_fits_total'] for job in jobs)
    print(f"Array task {array_idx}: {len(jobs)} bootstraps, {total_fits} fits, {n_cpus} CPUs")

    skip_datasets = ['F9_Popp_2025_model']

    # Flatten all individual fits into a single list so every CPU stays busy.
    # The old structure ran 8 fits serially inside each ProcessPoolExecutor worker,
    # wasting 11/12 CPUs whenever one bootstrap finished early.
    fit_specs = []
    fits_per_bootstrap = defaultdict(int)  # (bootstrap_seed, save_dir) -> n_fits remaining

    for cjob in jobs:
        if cjob['dataset_name'] in skip_datasets:
            continue

        save_dir = cjob['save_dir'].replace('/data/ross', '/projects/pedjas_lab/stewart.ro')
        shared   = cjob['shared_data']
        bs_seed  = cjob['bootstrap_seed']
        is_mv    = cjob.get('multivariate', False)

        # Skip modes already saved on disk (resume support)
        save_file = f"{save_dir}/bootstrap_{bs_seed}_best_fits.pkl"
        existing = {}
        if os.path.exists(save_file):
            try:
                with open(save_file, 'rb') as f:
                    existing = pickle.load(f)
            except Exception:
                existing = {}

        nc_keys = sorted(k for k in cjob if k.startswith('jobs_') and isinstance(cjob[k], list))
        for nc_key in nc_keys:
            label = nc_key.replace('jobs_', '')
            if label in existing and existing[label] is not None:
                continue  # already have a good result for this mode

            for minimal_job in cjob[nc_key]:
                full_job = {**minimal_job, **shared,
                            'dataset_name': cjob['dataset_name'],
                            'save_dir': save_dir}
                if is_mv:
                    full_job['multivariate'] = True
                if 'kwargs' in full_job and 'multivariate' in full_job.get('kwargs', {}):
                    del full_job['kwargs']['multivariate']

                fit_specs.append((full_job, bs_seed, label, save_dir))
                fits_per_bootstrap[(bs_seed, save_dir)] += 1

    if not fit_specs:
        print("All fits already completed — nothing to do.")
        return

    print(f"Submitting {len(fit_specs)} fits across {len(fits_per_bootstrap)} bootstraps")

    # best[(bs_seed, label, save_dir)] = (val_ll, result)
    best = {}
    remaining = dict(fits_per_bootstrap)  # copy — counts down as fits complete

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as ex:
        futures = [ex.submit(run_one_fit, spec) for spec in fit_specs]

        for fut in concurrent.futures.as_completed(futures):
            try:
                bs_seed, label, save_dir, result = fut.result()
            except Exception as e:
                print(f"Fit raised: {e}")
                continue

            val_ll = result.get('val_ll', -np.inf) if result else -np.inf
            key = (bs_seed, label, save_dir)
            if val_ll > best.get(key, (-np.inf, None))[0]:
                best[key] = (val_ll, result)

            bs_key = (bs_seed, save_dir)
            remaining[bs_key] -= 1

            # Save as soon as all fits for this bootstrap are done
            if remaining[bs_key] == 0:
                save_file = f"{save_dir}/bootstrap_{bs_seed}_best_fits.pkl"
                label_results = {
                    lbl: res
                    for (b, lbl, sd), (_, res) in best.items()
                    if b == bs_seed and sd == save_dir
                }
                # Merge with any pre-existing results (from earlier runs)
                existing = {}
                if os.path.exists(save_file):
                    try:
                        with open(save_file, 'rb') as f:
                            existing = pickle.load(f)
                    except Exception:
                        existing = {}
                existing.update(label_results)
                os.makedirs(save_dir, exist_ok=True)
                with open(save_file, 'wb') as f:
                    pickle.dump(existing, f)
                print(f"  ✓ Saved bootstrap {bs_seed} -> {save_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_array_task.py <jobs_dir> <array_idx>")
        sys.exit(1)

    run_array_task(sys.argv[1], int(sys.argv[2]))
