"""
Bootstrap fitting engine for assay calibration
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
import subprocess

from ..fit_utils.fit import Fit

from .config import PipelineConfig
from .utils import load_dataset_from_df

class BootstrapRunner:
    """Handles bootstrap fitting with different execution modes"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.dataset = None
        self.fitter = None
        
    def run(self) -> Tuple[Dict, Optional[Dict]]:
        """Main entry point for bootstrap fitting.
        
        Returns
        -------
        (bootstrap_results, dataset_splits)
            dataset_splits is None when execution_mode == 'slurm'.
        """
        
        # Load dataset
        self._load_dataset()
        
        # Choose execution strategy
        if self.config.execution_mode == "slurm":
            return self._run_slurm()
        elif self.config.execution_mode == "parallel":
            return self._run_parallel()
        else:  # single
            return self._run_single()
    
    def _load_dataset(self):
        """Load dataset from CSV"""
        df = pd.read_csv(self.config.dataset_csv)
        
        self.dataset = load_dataset_from_df(df, self.config)
        
        n_samples = len([s for s in self.dataset.samples])
        print(f"Loaded dataset: {self.config.dataset_name}")
        print(f"  Samples: {n_samples}")
        print(f"  Variants: {len(self.dataset.scores)}")
        
        if n_samples < 3:
            raise ValueError(f"Insufficient samples: {n_samples} < 3")
        
        self.fitter = Fit(self.dataset)
    
    def _run_parallel(self) -> Tuple[Dict, Dict]:
        """Run bootstrap fits in parallel using joblib, flattened across bootstraps × fits."""
        print(f"\nRunning {self.config.n_bootstraps} bootstraps in parallel...")
        print(f"  Jobs: {self.config.n_jobs if self.config.n_jobs > 0 else 'all CPUs'}")
        print(f"  Components: {self.config.components}")

        # Generate all bootstrap jobs
        all_jobs = []
        for bootstrap_idx in range(self.config.n_bootstraps):
            all_jobs.append(self._generate_bootstrap_job(bootstrap_idx))

        # Extract dataset splits for OOB (before execution, zero overhead)
        dataset_splits = self._extract_splits(all_jobs)

        # Flatten to one task per individual fit across all bootstraps × components × fits
        flat_tasks = [
            (bootstrap_job['bootstrap_seed'], component_key, minimal_job, job_data['shared_data'])
            for bootstrap_job in all_jobs
            for component_key, job_data in bootstrap_job['component_jobs'].items()
            for minimal_job in job_data['jobs']
        ]

        # Execute all individual fits in parallel
        flat_results = Parallel(n_jobs=self.config.n_jobs, verbose=10)(
            delayed(BootstrapRunner._execute_single_fit)(
                minimal_job, shared_data, self.config.dataset_name
            )
            for _, _, minimal_job, shared_data in flat_tasks
        )

        # Aggregate: keep best fit by val_ll per (bootstrap_seed, component_key)
        best_fits: Dict = {}
        for (bootstrap_seed, component_key, _, _), result in zip(flat_tasks, flat_results):
            if result is None:
                continue
            key = (bootstrap_seed, component_key)
            if key not in best_fits or result['val_ll'] > best_fits[key]['val_ll']:
                best_fits[key] = result

        # Reconstruct the same {seed: {component_key: best_result}} structure
        aggregated = {}
        for bootstrap_idx in range(self.config.n_bootstraps):
            entry = {}
            for n_c in self.config.components:
                component_key = f"{n_c}c"
                entry[component_key] = best_fits.get((bootstrap_idx, component_key))
            aggregated[bootstrap_idx] = entry

        return aggregated, dataset_splits

    @staticmethod
    def _execute_single_fit(minimal_job: Dict, shared_data: Dict, dataset_name: str) -> Optional[Dict]:
        """Execute one fit job; returns None on failure."""
        try:
            full_job = {**minimal_job, **shared_data, 'dataset_name': dataset_name}
            return Fit.execute_fit_job(full_job)
        except Exception as e:
            print(f"  ✗ Fit failed: {e}")
            return None
    
    def _run_single(self) -> Tuple[Dict, Dict]:
        """Run bootstrap fits single-threaded (for debugging)"""
        print(f"\nRunning {self.config.n_bootstraps} bootstraps (single-threaded)...")
        print("Warning: This will be slow. Consider using --mode parallel")
        
        # Generate all jobs first to extract splits
        all_jobs = []
        for bootstrap_idx in range(self.config.n_bootstraps):
            all_jobs.append(self._generate_bootstrap_job(bootstrap_idx))
        
        dataset_splits = self._extract_splits(all_jobs)
        
        results = []
        for bootstrap_idx, bootstrap_job in enumerate(all_jobs):
            print(f"\nBootstrap {bootstrap_idx + 1}/{self.config.n_bootstraps}")
            
            result = self._execute_bootstrap_job(bootstrap_job)
            results.append(result)
            
            if (bootstrap_idx + 1) % 10 == 0:
                print(f"  Completed {bootstrap_idx + 1}/{self.config.n_bootstraps}")
        
        return self._aggregate_results(results), dataset_splits

    
    def _run_slurm(self) -> Dict:
        """Run bootstrap fits on SLURM cluster"""
        print(f"\nPreparing SLURM job array...")
        
        # Create jobs directory
        jobs_dir = Path(self.config.output_dir) / "slurm_jobs"
        jobs_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate and save all jobs
        print(f"Generating {self.config.n_bootstraps} bootstrap jobs...")
        jobs_per_array = max(1, self.config.n_bootstraps // 1000)  # Max 1000 array tasks
        num_arrays = (self.config.n_bootstraps + jobs_per_array - 1) // jobs_per_array
        
        for array_idx in range(num_arrays):
            start_idx = array_idx * jobs_per_array
            end_idx = min(start_idx + jobs_per_array, self.config.n_bootstraps)
            
            array_jobs = []
            for bootstrap_idx in range(start_idx, end_idx):
                job = self._generate_bootstrap_job(bootstrap_idx)
                array_jobs.append(job)
            
            # Save array job file
            job_file = jobs_dir / f"array_{array_idx:04d}.pkl"
            with open(job_file, 'wb') as f:
                pickle.dump(array_jobs, f)
        
        # Create SLURM submission script
        slurm_script = self._create_slurm_script(jobs_dir, num_arrays)
        script_path = jobs_dir / "submit.sh"
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        os.chmod(script_path, 0o755)
        
        # Create worker script
        worker_script = self._create_worker_script()
        worker_path = jobs_dir / "run_array_task.py"
        with open(worker_path, 'w') as f:
            f.write(worker_script)
        
        print(f"\nSLURM setup complete:")
        print(f"  Jobs directory: {jobs_dir}")
        print(f"  Array tasks: {num_arrays}")
        print(f"  Jobs per task: {jobs_per_array}")
        print(f"\nTo submit:")
        print(f"  cd {jobs_dir}")
        print(f"  sbatch submit.sh")
        if self.config.compute_oob:
            print(f"\nNote: OOB evidence is not supported with SLURM execution.")
        
        print(f"\nAfter completion, run:")
        print(f"  python run_pipeline.py --dataset {self.config.dataset_csv} " +
              f"--name {self.config.dataset_name} --collect-slurm {jobs_dir}")
        
        sys.exit(0)  # Exit after setup
        
        sys.exit(0)  # Exit after setup
    
    @staticmethod
    def _extract_splits(all_jobs: List[Dict]) -> Dict[int, Dict]:
        """Extract val observations/assignments from pre-generated jobs for OOB."""
        splits = {}
        for job in all_jobs:
            seed = job['bootstrap_seed']
            first_key = next(iter(job['component_jobs']))
            shared = job['component_jobs'][first_key]['shared_data']
            splits[seed] = {
                'val_observations': shared['val_observations'],
                'val_sample_assignments': shared['val_sample_assignments'],
            }
        return splits
    
    def _generate_bootstrap_job(self, bootstrap_idx: int) -> Dict:

        """Generate jobs for a single bootstrap iteration"""
        
        # Generate fit jobs for each component count
        all_jobs = {}
        for n_components in self.config.components:
            fit_kwargs = dict(
                component_range=[n_components],
                bootstrap_seed=bootstrap_idx,
                check_monotonic=True,
                num_fits=self.config.num_fits_per_bootstrap,
            )
            if self.config.sample_balance_beta != 0.0:
                fit_kwargs["sample_balance_beta"] = self.config.sample_balance_beta
            if self.config.sample_proportions is not None:
                fit_kwargs["sample_proportions"] = self.config.sample_proportions
            if self.config.weighted_val_ll:
                fit_kwargs["weighted_val_ll"] = True
            jobs = self.fitter.generate_fit_jobs(**fit_kwargs)
            
            # Extract shared data (train/val splits)
            if jobs:
                shared_data = {
                    'train_observations': jobs[0]['train_observations'],
                    'train_sample_assignments': jobs[0]['train_sample_assignments'],
                    'val_observations': jobs[0]['val_observations'],
                    'val_sample_assignments': jobs[0]['val_sample_assignments'],
                }
                
                # Create minimal job specs (without redundant data)
                minimal_jobs = []
                for job in jobs:
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
                    minimal_jobs.append(minimal_job)
                
                all_jobs[f"{n_components}c"] = {
                    'shared_data': shared_data,
                    'jobs': minimal_jobs
                }
        
        return {
            'bootstrap_seed': bootstrap_idx,
            'dataset_name': self.config.dataset_name,
            'component_jobs': all_jobs
        }
    
    def _execute_bootstrap_job(self, bootstrap_job: Dict) -> Dict:
        """Execute all fits for a single bootstrap iteration"""
        
        bootstrap_seed = bootstrap_job['bootstrap_seed']
        results = {'bootstrap_seed': bootstrap_seed}
        
        # Execute fits for each component count
        for component_key, job_data in bootstrap_job['component_jobs'].items():
            shared_data = job_data['shared_data']
            best_val_ll = -np.inf
            best_result = None
            
            # Run all fits for this component count
            for minimal_job in job_data['jobs']:
                try:
                    # Reconstruct full job
                    full_job = {**minimal_job, **shared_data}
                    full_job['dataset_name'] = self.config.dataset_name
                    
                    # Execute fit
                    result = Fit.execute_fit_job(full_job)
                    
                    # Track best fit
                    if result['val_ll'] > best_val_ll:
                        best_result = result
                        best_val_ll = result['val_ll']
                
                except Exception as e:
                    print(f"  ✗ Fit failed: {e}")
                    continue
            
            results[component_key] = best_result
        
        return results
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate bootstrap results into final structure"""
        
        aggregated = {}
        for result in results:
            bootstrap_seed = result['bootstrap_seed']
            aggregated[bootstrap_seed] = {
                k: v for k, v in result.items() 
                if k != 'bootstrap_seed'
            }
        
        return aggregated
    
    def _create_slurm_script(self, jobs_dir: Path, num_arrays: int) -> str:
        """Create SLURM submission script"""
        
        logs_dir = jobs_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Build module load commands
        if self.config.slurm_module_commands:
            module_commands = "\n".join(self.config.slurm_module_commands)
        else:
            module_commands = "# No module commands specified - set via --slurm-modules"
        
        # Build conda activation
        if self.config.slurm_conda_env:
            conda_activate = f"""
        source $HOME/.bashrc
        conda activate {self.config.slurm_conda_env}
        """
        else:
            conda_activate = "# No conda environment specified"

        return f"""#!/bin/bash
#SBATCH --account={self.config.slurm_account}
#SBATCH --job-name=calibration_{self.config.dataset_name}
#SBATCH --output={logs_dir}/array_%A_%a.out
#SBATCH --error={logs_dir}/array_%A_%a.err
#SBATCH --array=0-{num_arrays-1}
#SBATCH --time={self.config.slurm_time_hours}:00:00
#SBATCH --mem={self.config.slurm_mem_gb}G
#SBATCH --cpus-per-task={self.config.slurm_cpus_per_task}
#SBATCH --partition={self.config.slurm_partition}

# Load required modules (customize via --slurm-modules)
{module_commands}
{conda_activate}

python run_array_task.py {jobs_dir} $SLURM_ARRAY_TASK_ID

echo "Array task $SLURM_ARRAY_TASK_ID completed"
"""
    
    def _create_worker_script(self) -> str:
        """Create worker script for SLURM array tasks"""
        
        return """import sys
import pickle
import os
import concurrent.futures
from pathlib import Path

# Add package to path
sys.path.append(str(Path(__file__).parent.parent))
from fit_bootstrap import BootstrapRunner

def run_array_task(jobs_dir, array_idx):
    array_file = Path(jobs_dir) / f"array_{array_idx:04d}.pkl"
    
    if not array_file.exists():
        print(f"Error: Array file {array_file} not found")
        sys.exit(1)
    
    with open(array_file, 'rb') as f:
        jobs = pickle.load(f)
    
    print(f"Array task {array_idx}: Processing {len(jobs)} bootstrap iterations")
    
    # Execute jobs in parallel
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    
    # Create dummy runner for execution
    from config import PipelineConfig
    config = PipelineConfig(
        dataset_csv="",  # Not needed for execution
        dataset_name=jobs[0]['dataset_name']
    )
    runner = BootstrapRunner(config)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as ex:
        futures = [ex.submit(runner._execute_bootstrap_job, job) for job in jobs]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    # Save results
    output_file = Path(jobs_dir).parent / f"results_array_{array_idx:04d}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Array task {array_idx} complete!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_array_task.py <jobs_dir> <array_idx>")
        sys.exit(1)
    
    jobs_dir = sys.argv[1]
    array_idx = int(sys.argv[2])
    run_array_task(jobs_dir, array_idx)
"""
