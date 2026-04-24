import glob
import pickle
from collections import defaultdict
from joblib import Parallel, delayed
import os
from typing import Dict, Set, Tuple, List



def load_single_pkl_file(filepath: str, target_datasets: Set[str], 
                         max_splits: int, current_counts: Dict[str, int]) -> Dict:
    """
    Load a single pkl file and extract relevant jobs.
    
    Returns dict of {dataset_name: {bootstrap_seed: shared_data}}
    """
    results = defaultdict(dict)
    
    try:
        with open(filepath, "rb") as f:
            job_array = pickle.load(f)
        
        for job in job_array:
            job_dataset = job.get("dataset_name", "")
            
            # Skip if not a target dataset
            if job_dataset not in target_datasets:
                continue
            
            # Skip if already have enough splits (use atomic check)
            if current_counts.get(job_dataset, 0) >= max_splits:
                continue
            
            bootstrap_seed = job.get("bootstrap_seed")
            shared_data = job.get("shared_data")
            
            if bootstrap_seed is not None and shared_data is not None:
                if bootstrap_seed not in results[job_dataset]:
                    results[job_dataset][bootstrap_seed] = shared_data
    
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
    
    return dict(results)



def get_replicate_directory(dataset: str) -> str:
    """
    Determine which replicate directory a non-keep_old dataset belongs to.
    
    Returns the specific directory name.
    """
    if dataset.endswith('_clinvar_2018'):
        return "explorer_jobs_replicate_run_2018"
    elif dataset in ["TARDBP_Bolognesi_Faure_2019", "SGCB_Li_2023", "SFPQ_unpublished"]:
        return "explorer_jobs_replicate_run_PU"
    elif dataset == "TSC2_combined_unpublished":
        return "explorer_jobs_tsc2"
    elif dataset in ['BAP1_Waters_2024','RAD51C_Olvera-León_2024_z_score_D4_D14']:
        return "explorer_jobs_bap1_rad51c"
    else:
        return "explorer_jobs_replicate_run"


def process_directory_parallel(mode: str, target_datasets: Set[str], 
                               base_path: str, max_splits: int,
                               jobs_subdir: str = "jobs",
                               n_jobs: int = -1) -> Dict:
    """
    Process all pkl files in a directory in parallel.
    
    Returns dict of {dataset_name: {bootstrap_seed: shared_data}}
    """
    jobs_dir = f"{base_path}/{mode}/{jobs_subdir}"
    
    if not os.path.exists(jobs_dir):
        print(f"  Directory not found: {jobs_dir}")
        return {}
    
    # Get all pkl files
    pkl_files = sorted(glob.glob(f"{jobs_dir}/array_*.pkl"))
    
    if not pkl_files:
        print(f"  No pkl files found in {jobs_dir}")
        return {}
    
    print(f"  {mode}: Loading {len(pkl_files)} pkl files with {n_jobs if n_jobs > 0 else 'all'} jobs...")
    
    # Track current counts to enable early stopping
    current_counts = defaultdict(int)
    all_results = defaultdict(dict)
    
    # Process in batches to allow early stopping
    batch_size = 100
    for batch_start in range(0, len(pkl_files), batch_size):
        batch_files = pkl_files[batch_start:batch_start + batch_size]
        
        # Load batch in parallel
        batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(load_single_pkl_file)(
                filepath, target_datasets, max_splits, current_counts
            )
            for filepath in batch_files
        )
        
        # Merge results
        for result in batch_results:
            for dataset, splits in result.items():
                for bootstrap_seed, shared_data in splits.items():
                    if bootstrap_seed not in all_results[dataset]:
                        all_results[dataset][bootstrap_seed] = shared_data
                        current_counts[dataset] += 1
        
        # Check if all datasets are complete
        if all(current_counts.get(d, 0) >= max_splits for d in target_datasets):
            print(f"    All target datasets complete, stopping early after {batch_start + len(batch_files)} files")
            break
        
        # Progress update
        complete = sum(1 for d in target_datasets if current_counts.get(d, 0) >= max_splits)
        print(f"    Batch {batch_start//batch_size + 1}: {complete}/{len(target_datasets)} datasets complete")
    
    return dict(all_results)

def get_dataset_directory(dataset: str, keep_old_list: List[str], 
                          dataset_relax_configs: Dict) -> str:
    """Determine the correct directory for a keep_old dataset."""
    if dataset not in keep_old_list:
        raise ValueError(f"Dataset {dataset} is not in keep_old_list")
    
    if dataset in dataset_relax_configs:
        return "explorer_jobs_semifinal_run"
    elif dataset.endswith('_clinvar_2018'):
        return "explorer_jobs_old_clinvar"
    else:
        return "explorer_jobs"


def get_all_keep_old_directories() -> List[str]:
    """Get all possible directories that might contain keep_old datasets."""
    return [
        "explorer_jobs",
        "explorer_jobs_old_clinvar",
        "explorer_jobs_semifinal_run",
        "explorer_jobs_12_01_25_rerun",
        "explorer_jobs_12_04_25_rerun",
        "explorer_jobs_circular_clinvar"
    ]


def recover_all_datasets_optimized(
    dataset_configs: Dict,
    keep_old_list: List[str],
    dataset_relax_configs: Dict,
    base_path: str = "/data/ross/assay_calibration",
    max_splits: int = 1000,
    n_jobs: int = -1
) -> Dict:
    """
    Efficiently recover splits for all datasets.
    For _clinvar_2018 datasets, falls back to keep_old directories if not found in replicate.
    """
    
    print("="*80)
    print("OPTIMIZED SPLIT RECOVERY")
    print("="*80)
    
    # Separate datasets into keep_old and non_keep_old
    keep_old_datasets = set(d for d in dataset_configs if d in keep_old_list)
    non_keep_old_datasets = set(d for d in dataset_configs if d not in keep_old_list)
    
    print(f"\nDatasets to process:")
    print(f"  Keep old: {len(keep_old_datasets)}")
    print(f"  New (non-keep_old): {len(non_keep_old_datasets)}")
    
    all_results = {}
    
    # ========================================================================
    # PART 1: Process keep_old datasets
    # ========================================================================
    if keep_old_datasets:
        print(f"\n{'='*80}")
        print("PART 1: Processing keep_old datasets")
        print('='*80)
        
        # Group keep_old datasets by directory
        datasets_by_dir = defaultdict(set)
        for dataset in keep_old_datasets:
            directory = get_dataset_directory(dataset, keep_old_list, dataset_relax_configs)
            datasets_by_dir[directory].add(dataset)
        
        print(f"\nGrouped into {len(datasets_by_dir)} directories:")
        for directory, datasets in sorted(datasets_by_dir.items()):
            print(f"  {directory}: {len(datasets)} datasets")
        
        # Process each directory
        for directory, datasets in datasets_by_dir.items():
            print(f"\nProcessing {directory}...")
            
            results = process_directory_parallel(
                mode=directory,
                target_datasets=datasets,
                base_path=base_path,
                max_splits=max_splits,
                jobs_subdir="jobs",
                n_jobs=n_jobs
            )
            
            # Merge results
            for dataset, splits in results.items():
                all_results[dataset] = splits
                print(f"  {dataset}: {len(splits)}/{max_splits} splits")
        
        # Check additional directories for any missing keep_old datasets
        missing_keep_old = [d for d in keep_old_datasets if len(all_results.get(d, {})) < max_splits]
        
        if missing_keep_old:
            print(f"\n{'='*80}")
            print(f"Checking additional directories for {len(missing_keep_old)} incomplete keep_old datasets")
            print('='*80)
            
            additional_dirs = [
                "explorer_jobs_12_01_25_rerun",
                "explorer_jobs_12_04_25_rerun",
                "explorer_jobs_circular_clinvar"
            ]
            
            for directory in additional_dirs:
                incomplete = [d for d in missing_keep_old if len(all_results.get(d, {})) < max_splits]
                
                if not incomplete:
                    break
                
                print(f"\n{directory}: Checking {len(incomplete)} datasets...")
                
                results = process_directory_parallel(
                    mode=directory,
                    target_datasets=set(incomplete),
                    base_path=base_path,
                    max_splits=max_splits,
                    jobs_subdir="jobs",
                    n_jobs=n_jobs
                )
                
                for dataset, splits in results.items():
                    if len(splits) > len(all_results.get(dataset, {})):
                        all_results[dataset] = splits
                        print(f"  ✓ {dataset}: {len(splits)}/{max_splits} splits")
    
    # ========================================================================
    # PART 2: Process non_keep_old datasets
    # ========================================================================
    if non_keep_old_datasets:
        print(f"\n{'='*80}")
        print("PART 2: Processing non-keep_old datasets")
        print('='*80)
        
        # Group non-keep_old datasets by directory
        datasets_by_dir = defaultdict(set)
        for dataset in non_keep_old_datasets:
            directory = get_replicate_directory(dataset)
            datasets_by_dir[directory].add(dataset)
        
        print(f"\nGrouped into {len(datasets_by_dir)} directories:")
        for directory, datasets in sorted(datasets_by_dir.items()):
            print(f"  {directory}: {len(datasets)} datasets")
        
        # Process each directory
        for directory, datasets in datasets_by_dir.items():
            print(f"\nProcessing {directory}...")
            
            if directory == "explorer_jobs_replicate_run_PU":
                jobs_ext = "jobs_PU"
            elif directory == "explorer_jobs_replicate_run_2018":
                jobs_ext = "jobs_2018"
            else:
                jobs_ext = "jobs"
            
            results = process_directory_parallel(
                mode=directory,
                target_datasets=datasets,
                base_path=base_path,
                max_splits=max_splits,
                jobs_subdir=jobs_ext,
                n_jobs=n_jobs
            )
            
            # Merge results
            for dataset, splits in results.items():
                all_results[dataset] = splits
                print(f"  {dataset}: {len(splits)}/{max_splits} splits")
    
    # ========================================================================
    # PART 3: Fallback for _clinvar_2018 datasets not found in replicate
    # ========================================================================
    print(f"\n{'='*80}")
    print("PART 3: Checking fallback for missing _clinvar_2018 datasets")
    print('='*80)
    
    # Find _clinvar_2018 datasets that weren't found or are incomplete
    missing_clinvar_2018 = [
        d for d in dataset_configs 
        if d.endswith('_clinvar_2018') 
        and d not in keep_old_list
        and len(all_results.get(d, {})) < max_splits
    ]
    
    if missing_clinvar_2018:
        print(f"\nFound {len(missing_clinvar_2018)} _clinvar_2018 datasets to check in keep_old directories:")
        for d in missing_clinvar_2018:
            print(f"  {d}: {len(all_results.get(d, {}))}/{max_splits}")
        
        # Try to find them in keep_old directories (including additional directories)
        fallback_dirs = [
            "explorer_jobs_old_clinvar",
            "explorer_jobs",
            "explorer_jobs_12_01_25_rerun",
            "explorer_jobs_12_04_25_rerun",
            "explorer_jobs_circular_clinvar"
        ]
        
        for directory in fallback_dirs:
            # Only check datasets still incomplete
            incomplete = [d for d in missing_clinvar_2018 if len(all_results.get(d, {})) < max_splits]
            
            if not incomplete:
                print(f"\nAll datasets complete, skipping remaining directories")
                break
            
            print(f"\n{directory}: Checking {len(incomplete)} datasets as fallback...")
            
            results = process_directory_parallel(
                mode=directory,
                target_datasets=set(incomplete),
                base_path=base_path,
                max_splits=max_splits,
                jobs_subdir="jobs",
                n_jobs=n_jobs
            )
            
            # Merge results (only if we don't have better results already)
            for dataset, splits in results.items():
                if len(splits) > len(all_results.get(dataset, {})):
                    all_results[dataset] = splits
                    print(f"  ✓ {dataset}: {len(splits)}/{max_splits} splits (fallback)")
    else:
        print("\nNo missing _clinvar_2018 datasets to check")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("RECOVERY COMPLETE")
    print("="*80)
    
    complete = sum(1 for splits in all_results.values() if len(splits) == max_splits)
    partial = sum(1 for splits in all_results.values() if 0 < len(splits) < max_splits)
    empty = sum(1 for splits in all_results.values() if len(splits) == 0)
    missing = len(dataset_configs) - len(all_results)
    
    print(f"\nResults:")
    print(f"  Complete ({max_splits} splits): {complete}")
    print(f"  Partial: {partial}")
    print(f"  Empty: {empty}")
    print(f"  Missing: {missing}")
    
    # Show problematic datasets
    if partial > 0 or empty > 0 or missing > 0:
        print("\nDatasets with issues:")
        
        for dataset in sorted(dataset_configs.keys()):
            splits = all_results.get(dataset, {})
            n_splits = len(splits)
            
            if n_splits < max_splits:
                # Determine which directory this dataset should be in
                if dataset in keep_old_list:
                    directory = get_dataset_directory(dataset, keep_old_list, dataset_relax_configs)
                    full_path = f"{directory}/jobs"
                else:
                    directory = get_replicate_directory(dataset)
                    if directory == "explorer_jobs_replicate_run_PU":
                        jobs_ext = "jobs_PU"
                    elif directory == "explorer_jobs_replicate_run_2018":
                        jobs_ext = "jobs_2018"
                    else:
                        jobs_ext = "jobs"
                    full_path = f"{directory}/{jobs_ext}"
                
                print(f"  {dataset}: {n_splits}/{max_splits} ({full_path})")
    
    return all_results


# Usage:
# if __name__ == "__main__":
    
#     dataset_to_splits = recover_all_datasets_optimized(
#         dataset_configs=new_dataset_configs,
#         keep_old_list=keep_old_list,
#         dataset_relax_configs=dataset_relax_configs,
#         base_path="/data/ross/assay_calibration",
#         max_splits=1000,
#         n_jobs=20  # Parallel loading of pkl files
#     )
    
#     # Save results
#     output_file = "/data/ross/assay_calibration/dataset_splits_recovered.pkl"
#     with open(output_file, "wb") as f:
#         pickle.dump(dataset_to_splits, f)
    
#     print(f"\nResults saved to {output_file}")


# output_file = "/data/ross/assay_calibration/dataset_splits_recovered.pkl"
# with open(output_file, "rb") as f:
#     dataset_to_splits = pickle.load(f)

# print("len(dataset_to_splits)", len(dataset_to_splits))

# dataset_to_add = "TP53_Fayer_2021_meta_clinvar_2018"

# dataset_to_splits = {**dataset_to_splits, **recover_all_datasets_optimized(
#     dataset_configs={dataset_to_add: new_dataset_configs[dataset_to_add]},
#     keep_old_list=keep_old_list,
#     dataset_relax_configs=dataset_relax_configs,
#     base_path="/data/ross/assay_calibration",
#     max_splits=1000,
#     n_jobs=20  # Parallel loading of pkl files
# )}

# print("len(dataset_to_splits)", len(dataset_to_splits))
