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
from .progress import ProgressReporter

from ..fit_utils.fit import Fit

from .config import PipelineConfig
from .utils import load_dataset_from_df

class BootstrapRunner:
    """Handles bootstrap fitting with different execution modes"""
    
    def __init__(self, config: PipelineConfig, reporter: "ProgressReporter | None" = None):
        self.config = config
        self.reporter = reporter
        self.dataset = None
        self.fitter = None
        
    def run(self) -> Tuple[Dict, Optional[Dict]]:
        """Main entry point for bootstrap fitting.

        Returns
        -------
        (bootstrap_results, dataset_splits)
        """

        # Load dataset
        self._load_dataset()

        # Choose execution strategy
        if self.config.execution_mode == "parallel":
            return self._run_parallel()
        else:  # single
            return self._run_single()
    
    def _load_dataset(self):
        """Load dataset from CSV"""
        sep = "\t" if self.config.dataset_csv.endswith((".tsv", ".tsv.gz")) else ","
        df = pd.read_csv(self.config.dataset_csv, sep=sep)
        self.dataset = load_dataset_from_df(df, self.config)
        
        n_samples = len([s for s in self.dataset.samples])
        n_variants = len(self.dataset.scores)
        print(f"Loaded dataset: {self.config.dataset_name}")
        print(f"  Samples: {n_samples}")
        print(f"  Variants: {n_variants}")
        
        if n_samples < 3:
            raise ValueError(f"Insufficient samples: {n_samples} < 3")
        
        self.fitter = Fit(self.dataset)

        # Report dataset info — n_fits_total computed here so reporter can show an accurate total from the start.
        n_fits_total = (
            self.config.n_bootstraps
            * len(self.config.components)
            * self.config.num_fits_per_bootstrap
        )
        if self.reporter is not None:
            self.reporter.start(
                n_bootstraps=self.config.n_bootstraps,
                n_fits_total=n_fits_total,
                n_variants=n_variants,
                n_samples=n_samples,
            )
    
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

        if getattr(self.config, "device", "cpu") == "gpu":
            flat_results = self._execute_flat_tasks_gpu(flat_tasks)
        else:
            # Run all fits; return_as="generator" is joblib >=1.2.0 only so we
            # collect as a plain list and accept that per-fit progress is coarser.
            flat_results = Parallel(n_jobs=self.config.n_jobs, verbose=0)(
                delayed(BootstrapRunner._execute_single_fit)(
                    minimal_job, shared_data, self.config.dataset_name
                )
                for _, _, minimal_job, shared_data in flat_tasks
            )
        if self.reporter is not None:
            # reporter.track is a no-op here but keeps the interface intact
            flat_results = list(self.reporter.track(iter(flat_results), total=len(flat_tasks)))

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

    def _execute_flat_tasks_gpu(self, flat_tasks: List[Tuple]) -> List[Optional[Dict]]:
        """GPU counterpart of the joblib dispatch above: batches flat_tasks
        through jax_batch.interop.run_gpu (grouped by num_components/
        multivariate) instead of one job per CPU worker.

        run_gpu doesn't preserve flat_tasks' order (it groups/batches jobs),
        so results are re-keyed by (bootstrap_seed, component_key, fit_idx)
        and reassembled in the caller's original order — the aggregation
        loop in _run_parallel does a positional `zip(flat_tasks, flat_results)`
        and would silently misattribute results if this weren't re-sorted.
        """
        from ..fit_utils.jax_batch.interop import run_gpu

        fit_specs = [
            ({**minimal_job, **shared_data, 'dataset_name': self.config.dataset_name},
             bootstrap_seed, component_key, "")
            for bootstrap_seed, component_key, minimal_job, shared_data in flat_tasks
        ]
        by_key = {
            (bs_seed, label, fit_idx): result
            for bs_seed, label, _save_dir, fit_idx, result in run_gpu(fit_specs)
        }
        return [
            by_key.get((bootstrap_seed, component_key, minimal_job['fit_idx']))
            for bootstrap_seed, component_key, minimal_job, _shared_data in flat_tasks
        ]

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
                'val_variant_indices': shared.get('val_variant_indices'),
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
                master_seed=self.config.seed,
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
                    'val_variant_indices': jobs[0].get('val_variant_indices'),
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
                        'multivariate': job.get('multivariate', False),
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
    
