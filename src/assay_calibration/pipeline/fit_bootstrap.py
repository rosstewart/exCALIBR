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
    """Handles bootstrap fitting"""

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

        return self._run_parallel()
    
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
        
        # 2 is the minimum viable sample count: gnomAD/population plus at
        # least one of Pathogenic/LP or Benign/LB (or Synonymous) -- this is
        # exactly the PU/NU case (see docs/input-formats.md#pnpunu-modes-
        # missing-class-inference), which fit_utils/point_ranges.py already
        # supports via density unmixing. Matches the `< 2` threshold used
        # elsewhere for the same check (hpc/prepare.py, multivariate_data/
        # combined.py, multivariate_data/common.py).
        if n_samples < 2:
            raise ValueError(f"Insufficient samples: {n_samples} < 2")
        
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
            import jax
            gpu_devices = jax.devices("gpu")
            print(f"  Device: GPU ({len(gpu_devices)}x {gpu_devices[0].device_kind})")
            flat_results = self._execute_flat_tasks_gpu(flat_tasks)
        else:
            print(f"  Jobs: {self.config.n_jobs if self.config.n_jobs > 0 else 'all CPUs'}")
            # Run all fits; return_as="generator" would let progress stream in
            # as each fit finishes, but isn't available in the pinned joblib
            # (checked: no `return_as` param as of 1.2.0), so this still
            # blocks until every fit completes and we accept that per-fit
            # progress is coarser than true streaming. verbose=51 (just above
            # joblib's own >50 stdout threshold -- below that it writes to
            # stderr instead) prints ~50 evenly-spaced "Done X out of Y |
            # elapsed ... remaining ..." lines regardless of how many total
            # fits there are (frequency scales with the verbose value, see
            # joblib.Parallel.print_progress), so this doesn't get spammier
            # on --preset xl/finest's much larger fit counts.
            flat_results = Parallel(n_jobs=self.config.n_jobs, verbose=51)(
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
    
