import sys
sys.path.append("..")
from pathlib import Path
import numpy as np
from scipy import stats
import importlib
import src.assay_calibration.fit_utils.cfusn.fit
from src.assay_calibration.fit_utils.fit import Fit
importlib.reload(src.assay_calibration.fit_utils.cfusn.fit)
importlib.reload(src.assay_calibration.fit_utils.fit)
from src.assay_calibration.fit_utils.cfusn.fit import single_fit
from src.assay_calibration.fit_utils.cfusn import (density_utils,constraints, optimize)
import scipy.stats as sps
import matplotlib
matplotlib.set_loglevel("warning")
import numpy as np
from tqdm.auto import trange
import os
sys.path.append(str(Path(os.getcwd()).parent))
# from src.assay_calibration.data_utils.dataset_clinvar_version_functionality import (
#     Scoreset as Scoreset_clinvar_func,
# )
from src.assay_calibration.data_utils.dataset import (
    Scoreset,
)
import json
import glob
import pickle
from joblib import Parallel, delayed
import pandas as pd

# ============================================================================
# STEP 1: Generate consolidated job manifest
# ============================================================================

def process_dataset(df, dataset, output_dir, N_BOOTSTRAPS, NUM_FITS, clinvar_release="2026", selected_components=None, population_type=None):
    """Process a single dataset and return consolidated jobs (one per bootstrap)."""
    if clinvar_release == "2026":
        dataset_name = dataset
    else:
        dataset_name = f"{dataset}_clinvar_{clinvar_release}"
    save_dir = f'{output_dir}/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)

    if selected_components is None:
        selected_components = [2, 3]
    print(f"{dataset_name}: components={selected_components}")

    try:
        scoreset_kwargs = dict(clinvar_release=clinvar_release, min_clinvar_star=1)
        if population_type is not None:
            scoreset_kwargs["population_type"] = population_type
        ds = Scoreset(df[df["Dataset"] == dataset], **scoreset_kwargs)
    except ValueError as e:
        print(f"{dataset} skipping: {e}")
        return

    n_samples = len([s for s in ds.samples])
    if n_samples < 2:
        print(f"{dataset} skipping: insufficient samples")
        return

    fitter = Fit(ds)

    all_jobs = []

    # Create one job per bootstrap iteration
    for bootstrap_iter in range(N_BOOTSTRAPS):
        jobs_2c = []
        if 2 in selected_components:
            jobs_2c = fitter.generate_fit_jobs(
                component_range=[2],
                bootstrap_seed=bootstrap_iter,
                check_monotonic=True,
                num_fits=NUM_FITS
            )

        jobs_3c = []
        if 3 in selected_components:
            jobs_3c = fitter.generate_fit_jobs(
                component_range=[3],
                bootstrap_seed=bootstrap_iter,
                check_monotonic=True,
                num_fits=NUM_FITS
            )

        jobs_4c = None
        # if 'VHL_Buckley_2024' in dataset_name:
        #     jobs_4c = fitter.generate_fit_jobs(
        #         component_range=[4],
        #         bootstrap_seed=bootstrap_iter,
        #         check_monotonic=True,
        #         num_fits=NUM_FITS
        #     )
        
        # Extract shared data from first available job (all jobs in a bootstrap share train/val splits)
        reference_jobs = jobs_2c or jobs_3c
        if reference_jobs:
            shared_data = {
                'train_observations': reference_jobs[0]['train_observations'],
                'train_sample_assignments': reference_jobs[0]['train_sample_assignments'],
                'val_observations': reference_jobs[0]['val_observations'],
                'val_sample_assignments': reference_jobs[0]['val_sample_assignments'],
            }
        else:
            shared_data = None
        
        # Strip out redundant data from individual jobs
        jobs_2c_minimal = []
        for job in jobs_2c:
            minimal_job = {
                'job_id': job['job_id'],
                'bootstrap_seed': job['bootstrap_seed'],
                'fit_idx': job['fit_idx'],
                'num_components': job['num_components'],
                'constrained': job['constrained'],
                'init_method': job['init_method'],
                'init_constraint_adjustment': job['init_constraint_adjustment'],
                'kwargs': job['kwargs']
            }
            jobs_2c_minimal.append(minimal_job)
        
        jobs_3c_minimal = []
        for job in jobs_3c:
            minimal_job = {
                'job_id': job['job_id'],
                'bootstrap_seed': job['bootstrap_seed'],
                'fit_idx': job['fit_idx'],
                'num_components': job['num_components'],
                'constrained': job['constrained'],
                'init_method': job['init_method'],
                'init_constraint_adjustment': job['init_constraint_adjustment'],
                'kwargs': job['kwargs']
            }
            jobs_3c_minimal.append(minimal_job)

        
        jobs_4c_minimal = []
        if jobs_4c is not None:
            for job in jobs_4c:
                minimal_job = {
                    'job_id': job['job_id'],
                    'bootstrap_seed': job['bootstrap_seed'],
                    'fit_idx': job['fit_idx'],
                    'num_components': job['num_components'],
                    'constrained': job['constrained'],
                    'init_method': job['init_method'],
                    'init_constraint_adjustment': job['init_constraint_adjustment'],
                    'kwargs': job['kwargs']
                }
                jobs_4c_minimal.append(minimal_job)
        
        # Consolidate into one job containing shared data + minimal job specs
        consolidated_job = {
            'dataset_name': dataset_name,
            'dataset_file': "",
            'save_dir': save_dir,
            'bootstrap_seed': bootstrap_iter,
            'shared_data': shared_data,  # Shared train/val splits
            'jobs_2c': jobs_2c_minimal,  # Just the fit parameters
            'jobs_3c': jobs_3c_minimal,  # Just the fit parameters
            'jobs_4c': jobs_4c_minimal,  # Just the fit parameters
            'num_fits_total': len(jobs_2c_minimal) + len(jobs_3c_minimal) + len(jobs_4c_minimal)
        }
        
        all_jobs.append(consolidated_job)
    
    return all_jobs

def requires_2018(df, dataset):
    genes_2018 = ['BRCA1', 'PTEN', 'MSH2', 'TP53']
    gene = df[df["Dataset"] == dataset]["Gene"].iloc[0]

    return gene in genes_2018
    

def generate_job_manifest(output_dir, target_array_size=1000, n_jobs=30, run_downsample_discordance=False,
                          config_file=None, dataframe_path=None, population_type=None):
    """
    Generate job manifest optimized for MAX_ARRAY_SIZE limit.

    Args:
        output_dir: Directory where jobs and results are written
        target_array_size: Target number of array tasks (default: 1000, your cluster limit)
        n_jobs: Number of parallel workers for job generation
        config_file: Path to dataset config JSON (default: src/igvf_configs/dataset_configs_jan_2026.json)
    """
    if config_file is None:
        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "src", "igvf_configs", "dataset_configs_jan_2026.json")

    with open(config_file) as f:
        dataset_config = json.load(f)

    # Build new->old name mapping (strip _clinvar_2018 suffix before lookup)
    name_map_df = pd.read_csv("/data/ross/assay_calibration/dataframe/new_dataset_names.csv")
    new_to_old = dict(zip(name_map_df["New_names"], name_map_df["Old_names"]))

    def components_for(dataset_name):
        entry = dataset_config.get(dataset_name)
        if entry is None:
            # Strip _clinvar_2018 suffix for the name-mapping lookup, but keep
            # dataset_name itself unchanged (only used to find the config entry)
            base = dataset_name.removesuffix("_clinvar_2018")
            old_name = new_to_old.get(base)
            if old_name is not None:
                # Re-attach the suffix if it was present
                lookup = old_name + ("_clinvar_2018" if dataset_name.endswith("_clinvar_2018") else "")
                entry = dataset_config.get(lookup)
        if entry is None:
            return [2, 3]  # default: both
        nc_str = entry[0]  # "2c" or "3c"
        return [int(nc_str[0])]

    jobs_dir = f"{output_dir}/jobs"
    os.makedirs(jobs_dir, exist_ok=True)

    N_BOOTSTRAPS = 1000
    NUM_FITS = 100  # fits per bootstrap per component

    if dataframe_path is None:
        dataframe_path = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz"
    sep = "\t" if dataframe_path.endswith(".tsv.gz") or dataframe_path.endswith(".tsv") else ","
    df = pd.read_csv(dataframe_path, sep=sep)
    datasets = df.Dataset.unique()

    print(f"Target array size: {target_array_size}")
    print(f"Parallel workers: {n_jobs if n_jobs > 0 else 'all CPUs'}")
    print(f"Config file: {config_file}")

    # Print component selection summary
    for dataset in sorted(datasets):
        clinvar_release = "2018" if requires_2018(df, dataset) else "2026"
        dataset_name = dataset if clinvar_release == "2026" else f"{dataset}_clinvar_{clinvar_release}"
        comps = components_for(dataset_name)
        print(f"  {dataset_name}: {comps}c")

    # Process all datasets in parallel
    print("\nLoading datasets and generating jobs...")
    all_jobs_by_dataset = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_dataset)(
            df, dataset, output_dir, N_BOOTSTRAPS, NUM_FITS,
            clinvar_release="2026" if not requires_2018(df, dataset) else "2018",
            selected_components=components_for(
                dataset if not requires_2018(df, dataset) else f"{dataset}_clinvar_2018"
            ),
            population_type=population_type,
        ) for dataset in datasets
    )
    
    # Flatten
    print("\nFlattening and organizing jobs...")
    all_jobs = []
    for jobs in all_jobs_by_dataset:
        if jobs is not None:
            all_jobs.extend(jobs)
    
    total_jobs = len(all_jobs)
    print(f"Total consolidated jobs: {total_jobs:,}")
    print(f"  (Each job runs up to {NUM_FITS} fits per selected component)")
    
    # Calculate optimal jobs per array task
    jobs_per_array = max(1, total_jobs // target_array_size)
    num_arrays = (total_jobs + jobs_per_array - 1) // jobs_per_array
    
    print(f"\nOptimal arrangement:")
    print(f"  Jobs per array task: {jobs_per_array}")
    print(f"  Number of array tasks: {num_arrays}")
    print(f"  Fits per array task: {jobs_per_array * NUM_FITS * 2:,}")
    
    # Assign global indices and save individual job files
    print("\nSaving job files...")
    job_index = []
    
    for array_idx in range(num_arrays):
        start_idx = array_idx * jobs_per_array
        end_idx = min(start_idx + jobs_per_array, total_jobs)
        
        # Get jobs for this array task
        array_jobs = all_jobs[start_idx:end_idx]
        
        # Save to file
        job_file = f"{jobs_dir}/array_{array_idx:04d}.pkl"
        with open(job_file, 'wb') as f:
            pickle.dump(array_jobs, f)
        
        # Track metadata
        for local_idx, job in enumerate(array_jobs):
            job_index.append({
                'array_idx': array_idx,
                'local_idx': local_idx,
                'global_idx': start_idx + local_idx,
                'dataset_name': job['dataset_name'],
                'bootstrap_seed': job['bootstrap_seed'],
                'num_fits_total': job['num_fits_total']
            })
    
    # Save lightweight index
    index_file = f"{output_dir}/job_index.json"
    with open(index_file, 'w') as f:
        json.dump({
            'total_jobs': total_jobs,
            'num_arrays': num_arrays,
            'jobs_per_array': jobs_per_array,
            'fits_per_job': NUM_FITS * 2,
            'job_index': job_index
        }, f, indent=2)
    
    print(f"Job index saved to: {index_file}")
    
    # Create SLURM script
    create_slurm_script(output_dir, num_arrays, jobs_per_array, NUM_FITS)
    
    return total_jobs, num_arrays


def create_slurm_script(output_dir, num_arrays, jobs_per_array, num_fits):
    """Create SLURM job array script."""
    
    # Estimate time per array task
    # Assume ~1 second per fit, plus overhead
    fits_per_array = jobs_per_array * num_fits * 2  # 2 components
    minutes_per_array = int(fits_per_array / 60) + 20  # Add buffer
    hours = min(minutes_per_array // 60, 11)  # Cap at 11 hours for safety
    minutes = minutes_per_array % 60
    time_str = f"{hours:02d}:{minutes:02d}:00"
    
    slurm_script = f"""#!/bin/bash
#SBATCH --account=predrag
#SBATCH --job-name=calibration_bootstrap
#SBATCH --output={output_dir}/logs/array_%A_%a.out
#SBATCH --error={output_dir}/logs/array_%A_%a.err
#SBATCH --array=0-{num_arrays-1}
#SBATCH --time={time_str}
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short

# Create logs directory
mkdir -p {output_dir}/logs

module load anaconda3/2024.06
source $HOME/.bashrc
conda activate pillar_project

# Run the worker script
python run_array_task.py {output_dir}/jobs $SLURM_ARRAY_TASK_ID

echo "Array task $SLURM_ARRAY_TASK_ID completed"
"""
    
    script_path = f"{output_dir}/submit_array.sh"
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    os.chmod(script_path, 0o755)
    print(f"\nSLURM script saved to: {script_path}")
    print(f"Estimated time per array task: {time_str}")


# ============================================================================
# STEP 2: Worker script
# ============================================================================

def create_worker_script(output_dir):
    """Create the worker script that runs array tasks."""
    
    worker_script = """import sys
import pickle
import os
sys.path.append("..")
from src.assay_calibration.fit_utils.fit import Fit

def run_array_task(jobs_dir, array_idx):
    \"\"\"Execute all jobs in an array task.\"\"\"
    
    array_file = f"{jobs_dir}/array_{array_idx:04d}.pkl"
    
    if not os.path.exists(array_file):
        print(f"Error: Array file {array_file} not found")
        sys.exit(1)
    
    # Load all jobs for this array task
    with open(array_file, 'rb') as f:
        jobs = pickle.load(f)
    
    print(f"Array task {array_idx}: Processing {len(jobs)} consolidated jobs")
    total_fits = sum(job['num_fits_total'] for job in jobs)
    print(f"Total fits to run: {total_fits}")
    
    for job_idx, consolidated_job in enumerate(jobs):
        print(f"\\n{'='*70}")
        print(f"Job {job_idx+1}/{len(jobs)}")
        print(f"  Dataset: {consolidated_job['dataset_name']}")
        print(f"  Bootstrap: {consolidated_job['bootstrap_seed']}")
        print(f"  Total fits: {consolidated_job['num_fits_total']}")
        print('='*70)
        
        # Get shared data for this bootstrap
        shared_data = consolidated_job['shared_data']
        
        # Run all 2c fits - reconstruct full job by merging shared data
        print(f"\\nRunning {len(consolidated_job['jobs_2c'])} fits for 2 components...")
        for fit_idx, minimal_job in enumerate(consolidated_job['jobs_2c']):
            try:
                # Reconstruct full job by adding shared data
                full_job = {**minimal_job, **shared_data}
                Fit.execute_fit_job(full_job)
                if (fit_idx + 1) % 10 == 0:
                    print(f"  Completed {fit_idx+1}/{len(consolidated_job['jobs_2c'])} (2c)")
            except Exception as e:
                print(f"  ✗ 2c fit {fit_idx} failed: {e}")
                continue
        
        # Run all 3c fits - reconstruct full job by merging shared data
        print(f"\\nRunning {len(consolidated_job['jobs_3c'])} fits for 3 components...")
        for fit_idx, minimal_job in enumerate(consolidated_job['jobs_3c']):
            try:
                # Reconstruct full job by adding shared data
                full_job = {**minimal_job, **shared_data}
                Fit.execute_fit_job(full_job)
                if (fit_idx + 1) % 10 == 0:
                    print(f"  Completed {fit_idx+1}/{len(consolidated_job['jobs_3c'])} (3c)")
            except Exception as e:
                print(f"  ✗ 3c fit {fit_idx} failed: {e}")
                continue
        
        print(f"\\n✓ Completed all fits for bootstrap {consolidated_job['bootstrap_seed']}")
    
    print(f"\\n{'='*70}")
    print(f"Array task {array_idx} complete!")
    print('='*70)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_array_task.py <jobs_dir> <array_idx>")
        sys.exit(1)
    
    jobs_dir = sys.argv[1]
    array_idx = int(sys.argv[2])
    
    run_array_task(jobs_dir, array_idx)
"""
    
    worker_path = f"{output_dir}/run_array_task.py"
    with open(worker_path, 'w') as f:
        f.write(worker_script)
    
    print(f"Worker script saved to: {worker_path}")


# ============================================================================
# STEP 3: Status checker
# ============================================================================

def create_status_checker(output_dir):
    """Create a script to check job completion status."""
    
    status_script = f"""import json
import os
from collections import defaultdict

index_file = "{output_dir}/job_index.json"
output_dir = "{output_dir}"

with open(index_file, 'r') as f:
    metadata = json.load(f)

print(f"Total consolidated jobs: {{metadata['total_jobs']:,}}")
print(f"Array tasks: {{metadata['num_arrays']:,}}")
print(f"Jobs per array: {{metadata['jobs_per_array']}}")
print(f"Fits per job: {{metadata['fits_per_job']}}")
print(f"Total fits: {{metadata['total_jobs'] * metadata['fits_per_job']:,}}")

# Check completed fits
completed_2c = 0
completed_3c = 0
total_expected_2c = 0
total_expected_3c = 0
by_dataset = defaultdict(lambda: {{'completed_2c': 0, 'completed_3c': 0, 'total_bootstraps': 0}})

for job_info in metadata['job_index']:
    dataset_name = job_info['dataset_name']
    bootstrap_seed = job_info['bootstrap_seed']
    
    by_dataset[dataset_name]['total_bootstraps'] += 1
    
    save_dir = f"{{output_dir}}/{{dataset_name}}"
    
    # Count expected fits (half are 2c, half are 3c)
    fits_per_component = job_info['num_fits_total'] // 2
    total_expected_2c += fits_per_component
    total_expected_3c += fits_per_component
    
    # Check 2c fits
    for fit_idx in range(fits_per_component):
        output_file = f"{{save_dir}}/2c_bootstrap_{{bootstrap_seed}}_fit_{{fit_idx}}.pkl"
        if os.path.exists(output_file):
            completed_2c += 1
            by_dataset[dataset_name]['completed_2c'] += 1
    
    # Check 3c fits
    for fit_idx in range(fits_per_component):
        output_file = f"{{save_dir}}/3c_bootstrap_{{bootstrap_seed}}_fit_{{fit_idx}}.pkl"
        if os.path.exists(output_file):
            completed_3c += 1
            by_dataset[dataset_name]['completed_3c'] += 1

total_completed = completed_2c + completed_3c
total_expected = total_expected_2c + total_expected_3c

print(f"\\nOverall Progress:")
print(f"  2c fits: {{completed_2c:,}} / {{total_expected_2c:,}} ({{completed_2c/total_expected_2c*100:.1f}}%)")
print(f"  3c fits: {{completed_3c:,}} / {{total_expected_3c:,}} ({{completed_3c/total_expected_3c*100:.1f}}%)")
print(f"  Total: {{total_completed:,}} / {{total_expected:,}} ({{total_completed/total_expected*100:.1f}}%)")

print(f"\\nBy Dataset:")
for dataset, stats in sorted(by_dataset.items()):
    total_per_dataset = stats['total_bootstraps'] * (metadata['fits_per_job'] // 2)
    completed_per_dataset = stats['completed_2c'] + stats['completed_3c']
    pct = completed_per_dataset / (total_per_dataset * 2) * 100 if total_per_dataset > 0 else 0
    print(f"  {{dataset}}: {{completed_per_dataset:,}}/{{total_per_dataset*2:,}} fits ({{pct:.1f}}%)")
"""
    
    status_path = f"{output_dir}/check_status.py"
    with open(status_path, 'w') as f:
        f.write(status_script)
    
    print(f"Status checker saved to: {status_path}")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Setup HPC job array for bootstrap fits')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory where jobs and results are written')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to dataset config JSON (default: src/igvf_configs/dataset_configs_jan_2026.json)')
    parser.add_argument('--target-array-size', type=int, default=1000,
                       help='Target number of array tasks (default: 1000, cluster MAX_ARRAY_SIZE)')
    parser.add_argument('--n-jobs', type=int, default=30,
                       help='Number of parallel workers for job generation (default: 30)')
    parser.add_argument('--run_downsample_discordance', action='store_true',
                   help='Generate jobs for downsampling and discordance analyses (default: False)')
    parser.add_argument('--dataframe', type=str, default=None,
                       help='Path to input dataframe (default: integrated_variant_effect_dataset.tsv.gz). '
                            'CSV (.csv/.csv.gz) or TSV (.tsv/.tsv.gz) are auto-detected by extension.')
    parser.add_argument('--population-type', type=str, default=None,
                       help="Population type passed to Scoreset (e.g. 'all_missense_nsSNV'). Omit to use Scoreset default.")
    args = parser.parse_args()

    output_dir = args.output_dir

    print("="*80)
    print("HPC Job Array Setup - Consolidated Bootstrap Fits")
    print("="*80)

    # Generate all jobs and scripts
    total_jobs, num_arrays = generate_job_manifest(
        output_dir=output_dir,
        target_array_size=args.target_array_size,
        n_jobs=args.n_jobs,
        run_downsample_discordance=args.run_downsample_discordance,
        config_file=args.config_file,
        dataframe_path=args.dataframe,
        population_type=args.population_type,
    )
    # create_worker_script(output_dir)
    # create_status_checker(output_dir)

    print("\n" + "="*80)
    print("Setup complete!")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Total consolidated jobs: {total_jobs:,}")
    print(f"  Array tasks: {num_arrays:,}")
    print(f"  Jobs per array task: {total_jobs // num_arrays}")
    print(f"\nNext steps:")
    print(f"1. Review: {output_dir}/submit_array.sh")
    print(f"2. Submit: cd {output_dir} && sbatch submit_array.sh")
    print(f"3. Monitor: python {output_dir}/check_status.py")
    print(f"4. Logs: {output_dir}/logs/")
