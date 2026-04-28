import sys
sys.path.append("..")
from pathlib import Path
import numpy as np
import importlib
import os
import json
import glob
import pickle
import pandas as pd
from joblib import Parallel, delayed
from collections import defaultdict

from src.assay_calibration.fit_utils.fit import Fit
from src.assay_calibration.data_utils.dataset import Scoreset, MultiScoreset

# keep dedup_same_aa_sub false for the univariate case, since dedup happens
def generate_predictor_datasets(gene=None, predictor=None):
    from collections import defaultdict

    wd = '/data/ross/assay_calibration'
    df_with_predictor_scores = pd.read_csv(f"{wd}/dataframe/integrated_variant_effect_dataset_analysis.csv.gz")
    
    # contains score col and training set label col
    predictor_info = {
        "REVEL": ("REVEL", "REVEL_train"),
        "MutPred2": ("MutPred2", "MP2_train"),
        "AlphaMissense": ("AM_score", None)
    }

    if predictor is not None:
        assert predictor in ["REVEL", "MutPred2", "AlphaMissense"]
        predictor_info = {predictor: predictor_info[predictor]}
    
    predictor_datasets = defaultdict(dict)
    for predictor, (score_col, training_set_col) in predictor_info.items():

        if gene is None:
            sg_calibration_genes = ["BRCA1","BRCA2","JAG1","MSH2","SCN5A","TP53","TSC2"]
            if predictor != "MutPred2":
                sg_calibration_genes.insert(2, "F9")
        else:
            sg_calibration_genes = [gene]
    
        df_predictor = df_with_predictor_scores[~df_with_predictor_scores[score_col].isna()]
        mask = df_predictor.Gene.isin(sg_calibration_genes)
        if training_set_col is not None:
            mask &= (df_predictor[training_set_col] == False)
        df_predictor = df_predictor[mask]

        # make IDs consistent within the same protein variant
        # dataset.py will ensure they are correctly:
        #    - deduplicated in single calibration
        #    - matched across all variant instances in multivariate, e.g. not deduplicated with SGE but deduplicated with VAMP-Seq
        missense_group_cols = ["Gene", "Ensembl Transcript ID", "aa_pos", "aa_ref", "aa_alt"]
        df_predictor["ID"] = (
            df_predictor[missense_group_cols]
            .astype(str)
            .agg("|".join, axis=1)
        )

        # also ensure the dataset is updated
        df_predictor["Dataset"] = (
            df_predictor["Gene"].astype(str) + f"_{predictor}"
        )
                
        for gene in sg_calibration_genes:
            predictor_datasets[predictor][gene] = {"df": df_predictor[df_predictor.Gene == gene], "score_col": score_col}

    return predictor_datasets
    

# ============================================================================
# STEP 0: Discover multi-assay gene groups
# ============================================================================

def discover_gene_groups(df, max_dimensions=None, manual_groups=None):
    if manual_groups is not None:
        all_datasets = set(df["Dataset"].unique())
        validated = {}
        for name, datasets in manual_groups.items():
            missing = [d for d in datasets if d not in all_datasets]
            if missing:
                print(f"  Warning: {name} — missing datasets: {missing}")
            present = [d for d in datasets if d in all_datasets]
            if len(present) > 1:
                validated[name] = present
            else:
                print(f"  Warning: {name} — fewer than 2 valid datasets, skipping")
        return validated

    dataset_gene = (
        df.groupby("Dataset")["Gene"]
        .first()
        .reset_index()
        .rename(columns={"Dataset": "dataset", "Gene": "gene"})
    )

    meta_analyses = ["F9_Popp_2025_model",
                     # "HMBS_van_Loggerenberg_2023_combined",
                     "TP53_Fayer_2021_meta",]
                     # "TP53_Fortuno_2021_Kato_meta",
                     # "TP53_Giacomelli_2018_combined_score"]

    gene_to_datasets = defaultdict(list)
    for _, row in dataset_gene.iterrows():
        ds_name = row["dataset"]
        if ds_name in meta_analyses:
            print(f"{ds_name}: SKIPPING (meta-analysis)")
            continue
        gene_to_datasets[row["gene"]].append(row["dataset"])

    multi_assay = {}
    for gene, datasets in gene_to_datasets.items():
        if len(datasets) <= 1:
            continue
        if max_dimensions is not None and len(datasets) > max_dimensions:
            print(f"  {gene}: {len(datasets)} datasets exceeds max_dimensions={max_dimensions}, skipping. "
                  f"Use manual_groups to specify which to combine.")
            print(f"    Datasets: {sorted(datasets)}")
            continue
        multi_assay[gene] = sorted(datasets)

    return multi_assay


def print_gene_groups(gene_groups):
    print(f"\nFound {len(gene_groups)} genes with multiple datasets:")
    print("-" * 70)
    for gene, datasets in sorted(gene_groups.items()):
        print(f"  {gene} ({len(datasets)} assays, {len(datasets)}-dimensional):")
        for ds in datasets:
            print(f"    - {ds}")
    print("-" * 70)
    print(f"Total: {sum(len(v) for v in gene_groups.values())} datasets across {len(gene_groups)} genes")


# ============================================================================
# STEP 1: Build MultiScoreset and generate jobs for one gene
# ============================================================================

def build_multi_scoreset(df, gene, datasets, clinvar_release="2025", **kwargs):
    scoresets = []
    dataset_names = []

    for ds_name in datasets:
        try:
            ds = Scoreset(
                df[df["Dataset"] == ds_name],
                clinvar_release=clinvar_release,
                min_clinvar_star=1,
            )
            n_samples = sum(1 for _ in ds.samples)
            if n_samples < 2:
                print(f"  {ds_name}: skipping (only {n_samples} sample(s))")
                continue
            scoresets.append(ds)
            dataset_names.append(ds_name)
        except (ValueError, KeyError) as e:
            print(f"  {ds_name}: skipping ({e})")
            continue

    if len(scoresets) < 2:
        print(f"  {gene}: fewer than 2 valid datasets, skipping")
        return None, dataset_names

    try:
        ms = MultiScoreset(scoresets, dataset_names=dataset_names)
    except (ValueError, KeyError) as e:
        print(f"  {gene}: MultiScoreset construction failed ({e})")
        return None, dataset_names

    return ms, dataset_names


def process_gene_group(df, gene, datasets, output_dir, N_BOOTSTRAPS, NUM_FITS,
                       clinvar_release="2025", component_range=None,
                       constraint_modes=None):
    """
    Process a single gene group: build MultiScoreset and generate consolidated jobs.

    Parameters
    ----------
    constraint_modes : list of str
        Which constraint modes to run. Subset of ['con', 'unc'].
        Default: ['con', 'unc'] (both).
    """
    if constraint_modes is None:
        constraint_modes = ['con', 'unc']

    gene_label = f"{gene}_mv{'_clinvar_' + clinvar_release if clinvar_release != '2025' else ''}"
    save_dir = f"{output_dir}/{gene_label}"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Gene: {gene} ({len(datasets)} assays)")
    print(f"{'='*60}")

    ms, valid_datasets = build_multi_scoreset(df, gene, datasets, clinvar_release)
    if ms is None:
        return None

    K_dim = ms.n_assays
    print(f"  MultiScoreset: {ms.n_variants} variants, {K_dim} dimensions")
    print(f"  Missing: {ms.missing.mean() * 100:.1f}%")
    print(f"  Samples: {dict(zip(ms.sample_names, ms.sample_counts.tolist()))}")

    if component_range is None:
        component_range = [2, 3]

    constrained_flags = []
    if 'con' in constraint_modes:
        constrained_flags.append(True)
    if 'unc' in constraint_modes:
        constrained_flags.append(False)

    fitter = Fit(ms)

    all_jobs = []

    for bootstrap_iter in range(N_BOOTSTRAPS):
        jobs_by_mode = {}

        for nc in component_range:
            for constrained in constrained_flags:
                mode_key = f"{nc}c_{'con' if constrained else 'unc'}"
                try:
                    jobs = fitter.generate_fit_jobs(
                        component_range=[nc],
                        bootstrap_seed=bootstrap_iter,
                        check_monotonic=constrained,
                        num_fits=NUM_FITS,
                    )
                except Exception as e:
                    print(f"  Bootstrap {bootstrap_iter}, {mode_key}: job generation failed ({e})")
                    jobs = []

                jobs_by_mode[mode_key] = jobs

        # Extract shared data from first available job
        shared_data = None
        for mode_key, jobs in jobs_by_mode.items():
            if jobs:
                first_job = jobs[0]
                shared_data = {
                    "train_observations": first_job["train_observations"],
                    "train_sample_assignments": first_job["train_sample_assignments"],
                    "val_observations": first_job["val_observations"],
                    "val_sample_assignments": first_job["val_sample_assignments"],
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
            "assay_datasets": valid_datasets,
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
    print(f"  Generated {len(all_jobs)} bootstrap jobs "
          f"({NUM_FITS} fits × [{modes_str}] × {{{constraint_str}}} each)")
    return all_jobs


# ============================================================================
# STEP 2: Generate full manifest
# ============================================================================

def generate_mv_job_manifest(target_array_size=1000, n_jobs=30,
                              max_dimensions=np.inf, manual_groups=None,
                              genes=None, component_range=None,
                              constraint_modes=None, output_dir=None):
    """
    Generate job manifest for multivariate calibration.

    Parameters
    ----------
    genes : list of str or None
        If provided, only process these genes. Otherwise discover all.
    component_range : list of int or None
        Component counts to fit (default [2, 3]).
    constraint_modes : list of str or None
        Subset of ['con', 'unc'] (default both).
    output_dir : str or None
        Output directory. Auto-generated if None.
    """
    if output_dir is None:
        output_dir = "/data/ross/assay_calibration/explorer_jobs_multivariate/default"
    jobs_dir = f"{output_dir}/jobs"
    os.makedirs(jobs_dir, exist_ok=True)

    N_BOOTSTRAPS = 200
    NUM_FITS = 100
    if component_range is None:
        component_range = [2, 3]
    if constraint_modes is None:
        constraint_modes = ['con', 'unc']

    # Load data
    df = pd.read_csv(
        "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz",
        sep="\t",
    )

    # Discover multi-assay genes
    gene_groups = discover_gene_groups(df, max_dimensions=max_dimensions,
                                       manual_groups=manual_groups)

    # Filter to requested genes
    if genes is not None:
        filtered = {}
        for g in genes:
            g_upper = g.upper()
            # Match case-insensitively
            matches = {k: v for k, v in gene_groups.items() if k.upper() == g_upper}
            if matches:
                filtered.update(matches)
            else:
                print(f"  Warning: gene '{g}' not found in multi-assay groups")
        gene_groups = filtered

    print_gene_groups(gene_groups)

    if not gene_groups:
        print("No multi-assay genes found. Exiting.")
        return 0, 0

    modes_str = ", ".join([f"{nc}c" for nc in component_range])
    constraint_str = ", ".join(constraint_modes)
    print(f"\nConfiguration:")
    print(f"  Components: [{modes_str}]")
    print(f"  Constraints: [{constraint_str}]")
    print(f"  Bootstraps: {N_BOOTSTRAPS}")
    print(f"  Fits per config: {NUM_FITS}")

    genes_2018 = ["BRCA1", "MSH2", "PTEN", "TP53"]
    
    # Process each gene group
    print(f"\nGenerating jobs ({n_jobs} workers)...")
    all_jobs_by_gene = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_gene_group)(
            df, gene, datasets, output_dir, N_BOOTSTRAPS, NUM_FITS,
            clinvar_release="2025" if gene not in genes_2018 else "2018", component_range=component_range,
            constraint_modes=constraint_modes,
        )
        for gene, datasets in gene_groups.items()
    )

    # Flatten
    all_jobs = []
    for jobs in all_jobs_by_gene:
        if jobs is not None:
            all_jobs.extend(jobs)

    total_jobs = len(all_jobs)
    if total_jobs == 0:
        print("No valid jobs generated. Exiting.")
        return 0, 0

    print(f"\nTotal consolidated jobs: {total_jobs:,}")

    # Partition into array tasks
    jobs_per_array = max(1, total_jobs // target_array_size)
    num_arrays = (total_jobs + jobs_per_array - 1) // jobs_per_array

    print(f"Jobs per array task: {jobs_per_array}")
    print(f"Number of array tasks: {num_arrays}")

    # Save job files
    print("\nSaving job files...")
    job_index = []

    for array_idx in range(num_arrays):
        start = array_idx * jobs_per_array
        end = min(start + jobs_per_array, total_jobs)
        array_jobs = all_jobs[start:end]

        job_file = f"{jobs_dir}/array_{array_idx:04d}.pkl"
        with open(job_file, "wb") as f:
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

    # Save index
    index_file = f"{output_dir}/job_index.json"
    with open(index_file, "w") as f:
        json.dump(
            {
                "total_jobs": total_jobs,
                "num_arrays": num_arrays,
                "jobs_per_array": jobs_per_array,
                "component_range": component_range,
                "constraint_modes": constraint_modes,
                "fits_per_component": NUM_FITS,
                "gene_groups": {g: ds for g, ds in gene_groups.items()},
                "job_index": job_index,
            },
            f,
            indent=2,
        )
    print(f"Job index saved to: {index_file}")

    # Summary by gene
    print("\nSummary by gene:")
    gene_job_counts = defaultdict(int)
    for j in all_jobs:
        gene_job_counts[j["gene"]] += 1
    for gene, count in sorted(gene_job_counts.items()):
        ndim = gene_groups[gene]
        print(f"  {gene}: {count:,} jobs ({len(ndim)} assays, {len(ndim)}D)")

    # SLURM script
    create_mv_slurm_script(output_dir, num_arrays, jobs_per_array,
                           NUM_FITS, component_range)

    return total_jobs, num_arrays


# ============================================================================
# STEP 3: SLURM script
# ============================================================================

def create_mv_slurm_script(output_dir, num_arrays, jobs_per_array, num_fits, component_range):
    fits_per_array = jobs_per_array * num_fits * len(component_range)
    minutes_per_array = int(fits_per_array / 60) + 30
    hours = min(minutes_per_array // 60, 11)
    minutes = minutes_per_array % 60
    time_str = f"{hours:02d}:{minutes:02d}:00"

    slurm_script = f"""#!/bin/bash
#SBATCH --account=predrag
#SBATCH --job-name=mv_calibration
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

python {output_dir}/run_mv_array_task.py {output_dir}/jobs $SLURM_ARRAY_TASK_ID

echo "Array task $SLURM_ARRAY_TASK_ID completed"
"""

    script_path = f"{output_dir}/submit_mv_array.sh"
    with open(script_path, "w") as f:
        f.write(slurm_script)
    os.chmod(script_path, 0o755)

    worker_script = f'''import sys
import pickle
import os
sys.path.append("..")
from src.assay_calibration.fit_utils.fit import Fit

def run_mv_array_task(jobs_dir, array_idx):
    array_file = f"{{jobs_dir}}/array_{{array_idx:04d}}.pkl"
    if not os.path.exists(array_file):
        print(f"Error: {{array_file}} not found")
        sys.exit(1)

    with open(array_file, "rb") as f:
        jobs = pickle.load(f)

    print(f"Array task {{array_idx}}: {{len(jobs)}} consolidated MV jobs")

    for job_idx, cjob in enumerate(jobs):
        print(f"\\nJob {{job_idx+1}}/{{len(jobs)}}: {{cjob['dataset_name']}} "
              f"({{cjob['n_dimensions']}}D, bootstrap={{cjob['bootstrap_seed']}})")

        shared = cjob["shared_data"]

        for nc_key in sorted(k for k in cjob if k.startswith("jobs_")):
            nc_jobs = cjob[nc_key]
            mode = nc_key.replace("jobs_", "")
            print(f"  Running {{len(nc_jobs)}} fits for {{mode}}...")

            for fit_idx, mjob in enumerate(nc_jobs):
                try:
                    full_job = {{**mjob, **shared, "dataset_name": cjob["dataset_name"]}}
                    result = Fit.execute_fit_job(full_job)

                    if result is not None:
                        save_path = os.path.join(
                            cjob["save_dir"],
                            f"{{mode}}_bootstrap_{{cjob['bootstrap_seed']}}_fit_{{fit_idx}}.pkl"
                        )
                        with open(save_path, "wb") as f:
                            pickle.dump(result, f)

                    if (fit_idx + 1) % 25 == 0:
                        print(f"    {{fit_idx+1}}/{{len(nc_jobs)}} done")
                except Exception as e:
                    print(f"    Fit {{fit_idx}} failed: {{e}}")
                    continue

        print(f"  Done bootstrap {{cjob['bootstrap_seed']}}")

    print(f"\\nArray task {{array_idx}} complete!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_mv_array_task.py <jobs_dir> <array_idx>")
        sys.exit(1)
    run_mv_array_task(sys.argv[1], int(sys.argv[2]))
'''

    worker_path = f"{output_dir}/run_mv_array_task.py"
    with open(worker_path, "w") as f:
        f.write(worker_script)

    print(f"\nSLURM script: {script_path}")
    print(f"Worker script: {worker_path}")
    print(f"Estimated time per array task: {time_str}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup HPC job array for multivariate bootstrap fits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All multi-assay genes, default 2c/3c, both constrained and unconstrained:
  python prepare_batch_jobs_multivariate.py

  # GCK only, 4-6 components, unconstrained only:
  python prepare_batch_jobs_multivariate.py --genes GCK --components 4 5 6 --constraints unc

  # Multiple genes, 3 components, both modes:
  python prepare_batch_jobs_multivariate.py --genes GCK ASPA TP53 --components 3

  # List available genes without generating jobs:
  python prepare_batch_jobs_multivariate.py --list-only
        """,
    )
    parser.add_argument(
        "--genes", nargs="+", default=None,
        help="Specific gene(s) to process. If omitted, all multi-assay genes are used."
    )
    ## TODO: ADD DATASET-SPECIFICITY PER GENE (ALONG WITH PREDICTORS)
    parser.add_argument(
        "--components", nargs="+", type=int, default=None,
        help="Component counts to fit (default: 2 3). E.g. --components 4 5 6"
    )
    parser.add_argument(
        "--constraints", nargs="+", choices=["con", "unc", "both"],
        default=["both"],
        help="Constraint modes: 'con' (constrained), 'unc' (unconstrained), "
             "'both' (default). Multiple can be specified."
    )
    parser.add_argument(
        "--target-array-size", type=int, default=1000,
        help="Target number of array tasks (default: 1000)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=30,
        help="Parallel workers for job generation (default: 30)"
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="Only list gene groups, don't generate jobs"
    )
    parser.add_argument(
        "--max-dimensions", type=int, default=None,
        help="Skip genes with more datasets than this"
    )
    parser.add_argument(
        "--predictors", type=str, default=None,
        help="Add REVEL, MutPred2, or AlphaMissense scores as additional dimensions"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: auto-generated from arguments)"
    )
    args = parser.parse_args()

    # Resolve constraint modes
    constraint_modes = set()
    for c in args.constraints:
        if c == "both":
            constraint_modes.update(["con", "unc"])
        else:
            constraint_modes.add(c)
    constraint_modes = sorted(constraint_modes)

    # Resolve component range
    component_range = args.components or [2, 3]

    max_dim = args.max_dimensions if args.max_dimensions else np.inf

    # Auto-determine output directory name
    base_dir = "/data/ross/assay_calibration"

    gene_part = "_".join(sorted(g.upper() for g in args.genes)) if args.genes else "all"
    comp_part = "c" + "-".join(str(c) for c in sorted(component_range))
    const_part = "+".join(sorted(constraint_modes))
    run_name = f"{gene_part}_{comp_part}_{const_part}"

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{base_dir}/{run_name}"

    print("=" * 80)
    print("HPC Job Array Setup — Multivariate Bootstrap Fits")
    print("=" * 80)
    print(f"  Genes:       {args.genes or 'all multi-assay'}")
    print(f"  Components:  {component_range}")
    print(f"  Constraints: {constraint_modes}")
    print(f"  Output:      {output_dir}")
    print()

    df = pd.read_csv(
        "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz",
        sep="\t",
    )

    gene_groups = discover_gene_groups(df, max_dimensions=max_dim)

    # Filter to requested genes
    if args.genes is not None:
        filtered = {}
        for g in args.genes:
            g_upper = g.upper()
            matches = {k: v for k, v in gene_groups.items() if k.upper() == g_upper}
            if matches:
                filtered.update(matches)
            else:
                print(f"  Warning: gene '{g}' not found in multi-assay groups")
        gene_groups = filtered

    print_gene_groups(gene_groups)

    if args.list_only:
        sys.exit(0)

    total_jobs, num_arrays = generate_mv_job_manifest(
        target_array_size=args.target_array_size,
        n_jobs=args.n_jobs,
        max_dimensions=max_dim,
        genes=args.genes,
        component_range=component_range,
        constraint_modes=constraint_modes,
        output_dir=output_dir,
    )

    print(f"\n{'=' * 80}")
    print("Setup complete!")
    print(f"{'=' * 80}")
    print(f"  Total consolidated jobs: {total_jobs:,}")
    print(f"  Array tasks: {num_arrays:,}")
    print(f"\nNext steps:")
    print(f"  1. Review: {output_dir}/submit_mv_array.sh")
    print(f"  2. Submit: cd {output_dir} && sbatch submit_mv_array.sh")
    print(f"  3. Monitor: check {output_dir}/logs/")