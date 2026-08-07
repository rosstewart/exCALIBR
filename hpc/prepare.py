#!/usr/bin/env python3
"""
ExCALIBR job manifest generator — unified entry point.

Sub-commands
------------
  default       Univariate Scoreset from integrated TSV (ClinVar-versioned)
  mavedb        Univariate Scoreset.from_mavedb (MaveDB / label-seq CSV)
  basicscoreset Univariate BasicScoreset from CSV with sample_assignments column
  multivariate  MultiScoreset for multi-assay genes (--gene-set selects the
                ingestion source: integrated [default] / fgfr / tp53 /
                labelseq / card11 / combined)
  predictor-mv  BasicMultiScoreset for predictor-score data
  mv_all        Merge FGFR/TP53/LABEL-seq/CARD11/predictor-mv/combined (--pipelines
                selects which, in priority order) into ONE job array/manifest

Usage
-----
  python hpc/prepare.py default       --output-dir DIR [options]
  python hpc/prepare.py mavedb        --output-dir DIR --score-cols COL ... [options]
  python hpc/prepare.py basicscoreset --output-dir DIR [--dataframe F | --data-dir D]
  python hpc/prepare.py multivariate  --output-dir DIR [options]
  python hpc/prepare.py predictor-mv  --output-dir DIR --data-dir D [options]
  python hpc/prepare.py mv_all        --output-dir DIR [--pipelines ... ] [options]

After running, submit with:
  bash hpc/submit_array.sh <output_dir>
"""

import sys
import os
import json
import glob
import pickle
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib
matplotlib.set_loglevel("warning")

# ── Repo root on sys.path so src/ imports work regardless of cwd ──────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ── Default hyperparameters ───────────────────────────────────────────────────
_DEFAULT_N_BOOTSTRAPS = 1000
_DEFAULT_NUM_FITS = 100


def _available_memory_mb() -> float:
    """Return available system memory in MB (Linux /proc/meminfo MemAvailable)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024   # kB → MB
    except OSError:
        pass
    # macOS / other: rough estimate from sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 1024 / 1024 * 0.5   # assume 50% free
    except Exception:
        return 8192   # 8 GB safe fallback


def _estimate_worker_mb(df_slice: "pd.DataFrame | None" = None) -> int:
    """Estimate peak RSS per worker in MB.

    Each worker imports src.assay_calibration (~800 MB) and holds one dataset
    slice (typically a few hundred rows — negligible).  The bulk is the import.
    """
    base_mb = 800
    if df_slice is not None:
        # Rough in-memory size of the slice (bytes → MB), with 3× pandas overhead
        try:
            slice_mb = df_slice.memory_usage(deep=True).sum() / 1024 / 1024 * 3
            return int(base_mb + slice_mb)
        except Exception:
            pass
    return base_mb + 50   # generous slice overhead default


def _resolve_n_jobs(n_jobs: int, sample_slice: "pd.DataFrame | None" = None) -> int:
    """Cap n_jobs by both available CPUs and available memory."""
    # CPU cap
    if n_jobs == -1:
        try:
            cpu_cap = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_cap = os.cpu_count() or 1
    else:
        cpu_cap = n_jobs

    # Memory cap
    try:
        avail_mb = _available_memory_mb()
        worker_mb = _estimate_worker_mb(sample_slice)
        mem_cap = max(1, int(avail_mb * 0.85 / worker_mb))
        resolved = min(cpu_cap, mem_cap)
        print(f"Workers: {resolved}  "
              f"(CPU cap: {cpu_cap}, "
              f"memory cap: {mem_cap} "
              f"[{avail_mb:,.0f} MB avail / ~{worker_mb} MB per worker])")
        return resolved
    except Exception:
        return cpu_cap


# =============================================================================
# Shared helpers
# =============================================================================

def _strip_minimal(job: dict) -> dict:
    return {k: job[k] for k in (
        "job_id", "bootstrap_seed", "fit_idx", "num_components",
        "constrained", "init_method", "init_constraint_adjustment", "kwargs"
    )}


def _shared_data(jobs: list) -> dict | None:
    if not jobs:
        return None
    j = jobs[0]
    return {
        "train_observations": j["train_observations"],
        "train_sample_assignments": j["train_sample_assignments"],
        "val_observations": j["val_observations"],
        "val_sample_assignments": j["val_sample_assignments"],
    }


def _save_manifest(output_dir: str, all_jobs: list, target_array_size: int,
                   n_bootstraps: int, extra_index_fields: dict | None = None) -> tuple[int, int]:
    jobs_dir = os.path.join(output_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    total_jobs = len(all_jobs)
    jobs_per_array = max(1, total_jobs // target_array_size)
    num_arrays = (total_jobs + jobs_per_array - 1) // jobs_per_array

    print(f"\nOptimal arrangement:")
    print(f"  Jobs per array task : {jobs_per_array}")
    print(f"  Number of array tasks: {num_arrays}")

    job_index = []
    print("\nSaving job files...")
    for array_idx in range(num_arrays):
        start = array_idx * jobs_per_array
        end = min(start + jobs_per_array, total_jobs)
        array_jobs = all_jobs[start:end]
        with open(os.path.join(jobs_dir, f"array_{array_idx:04d}.pkl"), "wb") as f:
            pickle.dump(array_jobs, f)
        for local_idx, job in enumerate(array_jobs):
            entry = {
                "array_idx": array_idx,
                "local_idx": local_idx,
                "global_idx": start + local_idx,
                "dataset_name": job["dataset_name"],
                "bootstrap_seed": job["bootstrap_seed"],
                "num_fits_total": job["num_fits_total"],
            }
            # propagate optional fields (gene, n_dimensions, etc.)
            for field in ("gene", "n_dimensions"):
                if field in job:
                    entry[field] = job[field]
            job_index.append(entry)

    index: dict = {
        "total_jobs": total_jobs,
        "num_arrays": num_arrays,
        "jobs_per_array": jobs_per_array,
        "n_bootstraps": n_bootstraps,
        "job_index": job_index,
    }
    if extra_index_fields:
        index.update(extra_index_fields)

    index_file = os.path.join(jobs_dir, "job_index.json")
    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Job index saved to   : {index_file}")

    dataset_names = sorted({e["dataset_name"] for e in job_index})
    datasets_file = os.path.join(output_dir, "datasets.txt")
    with open(datasets_file, "w") as f:
        f.write("\n".join(dataset_names) + "\n")
    print(f"Dataset list saved to: {datasets_file}")

    return total_jobs, num_arrays


def _print_next_steps(output_dir: str, total_jobs: int, num_arrays: int) -> None:
    slurm_dir = str(_SCRIPT_DIR)
    print(f"\n{'='*80}")
    print("Setup complete!")
    print(f"{'='*80}")
    print(f"  Total consolidated jobs : {total_jobs:,}")
    print(f"  Array tasks             : {num_arrays:,}")
    print(f"\nNext steps:")
    print(f"  1. Submit:    bash {slurm_dir}/submit_array.sh {output_dir}")
    print(f"     Override SLURM params via env vars, e.g.:")
    print(f"     SLURM_MEM=8G SLURM_CPUS=24 bash {slurm_dir}/submit_array.sh {output_dir}")
    print(f"  2. Monitor:   python {slurm_dir}/count_bootstraps.py {output_dir}")
    print(f"  3. Aggregate: python {slurm_dir}/aggregate_results.py {output_dir}")
    print(f"  4. Logs:      {output_dir}/logs/")
    print(f"\nSee hpc/README.md for full workflow and tunable parameters.")


# =============================================================================
# Mode: default / mavedb  (univariate Scoreset)
# =============================================================================

DEFAULT_MASTER_SEED = 0  # keep in sync with fit_utils.fit.DEFAULT_MASTER_SEED


def _parse_seed(s: str):
    """argparse type= for --seed. Deliberately duplicated (not imported) from
    src.assay_calibration.pipeline.config.parse_master_seed -- this module
    avoids top-level src.assay_calibration imports (~800 MB per worker, see
    the module docstring) so argparse setup in the parent process stays
    light; the 4 generate_fit_jobs call sites this feeds already do their own
    lazy `from ...fit_utils.fit import Fit` inside each worker function.
    """
    if s.lower() in ("none", "random"):
        return None
    return int(s)


def _requires_2018(df: pd.DataFrame, dataset: str) -> bool:
    genes_2018 = {"BRCA1", "PTEN", "MSH2", "TP53"}
    return df[df["Dataset"] == dataset]["Gene"].iloc[0] in genes_2018


def _resolve_dataset_filter(args) -> "set[str] | None":
    """Union of --datasets and --datasets-file, or None if neither was given
    (meaning: no filter, process every dataset in --dataframe)."""
    names: "set[str]" = set()
    if getattr(args, "datasets", None):
        names.update(args.datasets)
    datasets_file = getattr(args, "datasets_file", None)
    if datasets_file:
        with open(datasets_file) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    names.add(line)
    return names or None


def _components_from_config(config_file: str | None, new_to_old: dict) -> dict:
    """Return {dataset_name → [component_ints]} from a config JSON."""
    if config_file is None:
        config_file = str(_REPO_ROOT / "src" / "igvf_configs" / "dataset_configs_jan_2026.json")
    with open(config_file) as f:
        cfg = json.load(f)

    def lookup(dataset_name):
        entry = cfg.get(dataset_name)
        if entry is None:
            base = dataset_name.removesuffix("_clinvar_2018")
            old = new_to_old.get(base)
            if old is not None:
                key = old + ("_clinvar_2018" if dataset_name.endswith("_clinvar_2018") else "")
                entry = cfg.get(key)
        if entry is None:
            return [2, 3]
        # Two config schemas in the wild: dict entries (current, e.g.
        # dataset_configs_jul_2026.json: {"n_c": "2c", "benign_method": "avg"})
        # and older list/tuple entries (e.g. ["3c", "avg", {...}]) where the
        # n_c token is the first element. Same "n_c" extraction as
        # analysis/discovery.py's _parse_dataset_config_entry.
        n_c = str(entry.get("n_c", "3c")) if isinstance(entry, dict) else str(entry[0])
        if n_c == "all":
            return [2, 3]
        return [int(n_c[0])]

    return lookup


def _process_default_dataset(df_ds, dataset, output_dir, N_BOOTSTRAPS, NUM_FITS,
                              clinvar_release, selected_components, population_type,
                              force_gaussian=False, default_release="2026",
                              master_seed=DEFAULT_MASTER_SEED):
    from src.assay_calibration.data_utils.dataset import Scoreset
    from src.assay_calibration.fit_utils.fit import Fit

    dataset_name = dataset if clinvar_release == default_release else f"{dataset}_clinvar_{clinvar_release}"
    save_dir = os.path.join(output_dir, dataset_name)

    if selected_components is None:
        selected_components = [2, 3]
    print(f"  {dataset_name}: components={selected_components}")

    try:
        kw = dict(clinvar_release=clinvar_release, min_clinvar_star=1)
        if population_type:
            kw["population_type"] = population_type
        ds = Scoreset(df_ds, **kw)
    except (ValueError, KeyError) as e:
        print(f"  {dataset_name} skipping: {e}")
        return None

    if sum(1 for _ in ds.samples) < 2:
        print(f"  {dataset_name} skipping: insufficient samples")
        return None

    fitter = Fit(ds)
    all_jobs = []
    for bs in range(N_BOOTSTRAPS):
        jobs_by_key = {}
        ref_jobs = None
        for nc in selected_components:
            jobs = fitter.generate_fit_jobs(
                component_range=[nc], bootstrap_seed=bs, master_seed=master_seed,
                check_monotonic=True, num_fits=NUM_FITS,
                force_gaussian=force_gaussian)
            jobs_by_key[f"jobs_{nc}c"] = [_strip_minimal(j) for j in jobs]
            if ref_jobs is None:
                ref_jobs = jobs

        cjob = {
            "dataset_name": dataset_name,
            "dataset_file": "",
            "save_dir": save_dir,
            "bootstrap_seed": bs,
            "shared_data": _shared_data(ref_jobs or []),
            "num_fits_total": sum(len(v) for v in jobs_by_key.values()),
        }
        cjob.update(jobs_by_key)
        all_jobs.append(cjob)

    return all_jobs


def _process_mavedb_dataset(df_gene, gene, score_col, dataset_col, output_dir,
                             N_BOOTSTRAPS, NUM_FITS, selected_components,
                             population_type, regularization_type, splice_measure,
                             master_seed=DEFAULT_MASTER_SEED):
    from src.assay_calibration.data_utils.dataset import Scoreset
    from src.assay_calibration.fit_utils.fit import Fit

    dataset_name = f"{gene}_{score_col.replace(' ', '_')}"
    save_dir = os.path.join(output_dir, dataset_name)

    if selected_components is None:
        selected_components = [2, 3]
    print(f"  {dataset_name}: components={selected_components}")

    try:
        kw = {}
        if population_type:
            kw["population_type"] = population_type
        if regularization_type:
            kw["regularization_type"] = regularization_type
        ds = Scoreset.from_mavedb(
            df_gene,
            dataset_col=dataset_col,
            score_col=score_col,
            splice_measure=splice_measure,
            **kw,
        )
    except (ValueError, KeyError) as e:
        print(f"  {dataset_name} skipping: {e}")
        return None

    if sum(1 for _ in ds.samples) < 2:
        print(f"  {dataset_name} skipping: insufficient samples")
        return None

    fitter = Fit(ds)
    all_jobs = []
    for bs in range(N_BOOTSTRAPS):
        jobs_by_key = {}
        ref_jobs = None
        for nc in selected_components:
            jobs = fitter.generate_fit_jobs(
                component_range=[nc], bootstrap_seed=bs, master_seed=master_seed,
                check_monotonic=True, num_fits=NUM_FITS)
            jobs_by_key[f"jobs_{nc}c"] = [_strip_minimal(j) for j in jobs]
            if ref_jobs is None:
                ref_jobs = jobs

        cjob = {
            "dataset_name": dataset_name,
            "dataset_file": "",
            "save_dir": save_dir,
            "bootstrap_seed": bs,
            "shared_data": _shared_data(ref_jobs or []),
            "num_fits_total": sum(len(v) for v in jobs_by_key.values()),
        }
        cjob.update(jobs_by_key)
        all_jobs.append(cjob)

    return all_jobs


def run_default(args):
    from src.assay_calibration.data_utils.dataset import Scoreset
    from src.assay_calibration.fit_utils.fit import Fit

    N_BOOTSTRAPS = _DEFAULT_N_BOOTSTRAPS
    NUM_FITS = _DEFAULT_NUM_FITS
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "jobs"), exist_ok=True)

    # "default" mode. "pillar_project" mode. (see main()) reuses this same
    # function with clinvar_default_release="2025" instead -- every dataset
    # that isn't forced to 2018 uses this release, rather than 2026.
    default_release = getattr(args, "clinvar_default_release", "2026")

    df_path = args.dataframe or "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz"
    sep = "\t" if df_path.endswith((".tsv.gz", ".tsv")) else ","
    df = pd.read_csv(df_path, sep=sep)

    # Component selection: explicit flag > config file > default [2, 3]
    if args.components is not None:
        _comp_for = lambda name: args.components  # noqa: E731
        print(f"Components: {args.components} (explicit --components)")
    elif args.config_file is not None:
        name_map_df = pd.read_csv("/data/ross/assay_calibration/dataframe/new_dataset_names.csv")
        new_to_old = dict(zip(name_map_df["New_names"], name_map_df["Old_names"]))
        _comp_for = _components_from_config(args.config_file, new_to_old)
        print(f"Components: per-dataset from {args.config_file}")
    else:
        _comp_for = lambda name: [2, 3]  # noqa: E731
        print("Components: [2, 3] for all datasets (no --config-file or --components given)")

    datasets = df.Dataset.unique()
    dataset_filter = _resolve_dataset_filter(args)
    if dataset_filter is not None:
        # --datasets/--datasets-file entries may carry the "_clinvar_2018"
        # suffix used for pipeline *output* naming (see _requires_2018
        # below) -- the raw Dataset column itself never does, so strip it
        # before matching.
        normalized_filter = {name.removesuffix("_clinvar_2018") for name in dataset_filter}
        missing = sorted(normalized_filter - set(datasets))
        if missing:
            print(f"  WARNING: {len(missing)} requested dataset(s) not found in "
                  f"--dataframe's Dataset column: {missing}")
        datasets = [ds for ds in datasets if ds in normalized_filter]
        print(f"Datasets: {len(datasets)} (filtered via --datasets/--datasets-file "
              f"from {len(dataset_filter)} requested)")
    else:
        print(f"Datasets: {len(datasets)}")
    for ds in sorted(datasets):
        rel = "2018" if _requires_2018(df, ds) else default_release
        name = ds if rel == default_release else f"{ds}_clinvar_{rel}"
        print(f"  {name}: {_comp_for(name)}c")

    partitions = {ds: df[df["Dataset"] == ds] for ds in datasets}

    print("\nGenerating jobs...")
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_default_dataset)(
            partitions[ds], ds, output_dir, N_BOOTSTRAPS, NUM_FITS,
            clinvar_release="2018" if _requires_2018(df, ds) else default_release,
            default_release=default_release,
            selected_components=_comp_for(
                ds if not _requires_2018(df, ds) else f"{ds}_clinvar_2018"
            ),
            population_type=args.population_type,
            force_gaussian=args.force_gaussian,
            master_seed=args.seed,
        )
        for ds in datasets
    )

    all_jobs = [j for r in results if r for j in r]
    total_jobs, num_arrays = _save_manifest(output_dir, all_jobs, args.target_array_size, N_BOOTSTRAPS)
    _print_next_steps(output_dir, total_jobs, num_arrays)


def run_mavedb(args):
    N_BOOTSTRAPS = _DEFAULT_N_BOOTSTRAPS
    NUM_FITS = _DEFAULT_NUM_FITS
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "jobs"), exist_ok=True)

    sep = "\t" if args.dataframe.endswith((".tsv.gz", ".tsv")) else ","
    df = pd.read_csv(args.dataframe, sep=sep)

    genes = df[args.dataset_col].unique()
    combinations = [(gene, sc) for gene in sorted(genes) for sc in args.score_cols]
    print(f"MaveDB mode: {len(genes)} genes × {len(args.score_cols)} score cols = {len(combinations)} combinations")

    gene_partitions = {gene: df[df[args.dataset_col] == gene] for gene in genes}

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_mavedb_dataset)(
            gene_partitions[gene], gene, score_col, args.dataset_col, output_dir, N_BOOTSTRAPS, NUM_FITS,
            selected_components=args.components,
            population_type=args.population_type,
            regularization_type=args.regularization_type,
            splice_measure=args.splice_measure,
            master_seed=args.seed,
        )
        for gene, score_col in combinations
    )

    all_jobs = [j for r in results if r for j in r]
    total_jobs, num_arrays = _save_manifest(output_dir, all_jobs, args.target_array_size, N_BOOTSTRAPS)
    _print_next_steps(output_dir, total_jobs, num_arrays)


# =============================================================================
# Mode: basicscoreset  (univariate BasicScoreset)
# =============================================================================

def _load_basicscoreset_config(config_file: str) -> dict:
    """Load dataset_configs JSON → {dataset_name: {"n_c": int, ...}}."""
    with open(config_file) as f:
        raw = json.load(f)
    # **v must come first so the explicit int override wins
    return {k: {**v, "n_c": int(v["n_c"][0])} for k, v in raw.items()}


def _lookup_nc(dataset_name: str, cfg: dict, name_strip_re, fallback_components: list) -> list:
    """Return [nc] from config, stripping robustness suffix when needed.

    Tries in order:
      1. Exact match
      2. After stripping the robustness suffix (e.g. _ds4_s2, _disc0.10_s3)
      3. Stripped name + "_clinvar_2018" (for historic ClinVar-versioned keys)
      4. fallback_components
    """
    if cfg is None:
        return fallback_components

    for candidate in [
        dataset_name,
        name_strip_re.sub("", dataset_name) if name_strip_re else None,
        (name_strip_re.sub("", dataset_name) + "_clinvar_2018") if name_strip_re else None,
    ]:
        if candidate and candidate in cfg:
            return [cfg[candidate]["n_c"]]

    return fallback_components


def _process_basicscoreset_dataset(dataset_name, dataset_df, output_dir,
                                   N_BOOTSTRAPS, num_fits_by_nc, selected_components,
                                   master_seed=DEFAULT_MASTER_SEED):
    from src.assay_calibration.data_utils.dataset import BasicScoreset
    from src.assay_calibration.fit_utils.fit import Fit

    save_dir = os.path.join(output_dir, dataset_name)

    max_idx = (
        dataset_df["sample_assignments"].dropna()
        .str.split(",").explode().astype(int).max()
    )
    all_cols = [str(i) for i in range(max_idx + 1)]
    onehot = (
        dataset_df["sample_assignments"]
        .str.get_dummies(sep=",")
        .reindex(columns=all_cols, fill_value=0)
        .astype(bool)
        .to_numpy()
    )

    try:
        ds = BasicScoreset(scores=dataset_df["score"], sample_assignments=onehot)
    except ValueError as e:
        print(f"  {dataset_name} skipping: {e}")
        return None

    n_samples = sum(1 for _ in ds.samples)
    if n_samples < 2:
        print(f"  {dataset_name} skipping: only {n_samples} sample(s)")
        return None
    fits_desc = ", ".join(f"{nc}c×{num_fits_by_nc.get(nc, num_fits_by_nc.get('default', 8))}"
                          for nc in selected_components)
    print(f"  {dataset_name}: {n_samples} samples, components=[{fits_desc}]")

    fitter = Fit(ds)
    all_jobs = []
    for bs in range(N_BOOTSTRAPS):
        jobs_by_key = {}
        ref_jobs = None
        for nc in selected_components:
            num_fits = num_fits_by_nc.get(nc, num_fits_by_nc.get("default", 8))
            jobs = fitter.generate_fit_jobs(
                component_range=[nc], bootstrap_seed=bs, master_seed=master_seed,
                check_monotonic=True, num_fits=num_fits)
            jobs_by_key[f"jobs_{nc}c"] = [_strip_minimal(j) for j in jobs]
            if ref_jobs is None:
                ref_jobs = jobs

        cjob = {
            "dataset_name": dataset_name,
            "dataset_file": "",
            "save_dir": save_dir,
            "bootstrap_seed": bs,
            "shared_data": _shared_data(ref_jobs or []),
            "num_fits_total": sum(len(v) for v in jobs_by_key.values()),
        }
        cjob.update(jobs_by_key)
        all_jobs.append(cjob)

    return all_jobs


def run_basicscoreset(args):
    import re
    N_BOOTSTRAPS = getattr(args, "n_bootstraps", _DEFAULT_N_BOOTSTRAPS)
    # Per-nc num_fits: 2^nc covers all lambda sign patterns exactly once.
    fits_3c = getattr(args, "fits_3c", 8)
    fits_2c = getattr(args, "fits_2c", 4)
    num_fits_by_nc = {2: fits_2c, 3: fits_3c, "default": fits_3c}

    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "jobs"), exist_ok=True)

    # Config-file based per-dataset n_c lookup
    cfg = None
    name_strip_re = None
    if getattr(args, "config_file", None):
        cfg = _load_basicscoreset_config(args.config_file)
        print(f"Loaded per-dataset config from {args.config_file} ({len(cfg)} entries)")
    if getattr(args, "name_strip", None):
        name_strip_re = re.compile(args.name_strip)

    fallback_components = args.components or [2, 3]

    if args.data_dir is not None:
        paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv*")))
        partitions = {os.path.basename(p).split(".")[0]: pd.read_csv(p) for p in paths}
        print(f"Found {len(partitions)} CSV files in {args.data_dir}")
    else:
        sep = "\t" if args.dataframe.endswith((".tsv.gz", ".tsv")) else ","
        df = pd.read_csv(args.dataframe, sep=sep)
        if "Dataset" not in df.columns:
            raise ValueError("--dataframe must have a 'Dataset' column or use --data-dir")
        partitions = {ds: df[df["Dataset"] == ds] for ds in df["Dataset"].unique()}
        print(f"Found {len(partitions)} datasets in {args.dataframe}")

    print(f"\nGenerating jobs for {len(partitions)} datasets "
          f"({N_BOOTSTRAPS} boots, fits/boot: 2c={fits_2c} 3c={fits_3c})...")
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_basicscoreset_dataset)(
            name, df_ds, output_dir, N_BOOTSTRAPS, num_fits_by_nc,
            _lookup_nc(name, cfg, name_strip_re, fallback_components),
            master_seed=args.seed,
        )
        for name, df_ds in partitions.items()
    )

    all_jobs = [j for r in results if r for j in r]
    total_jobs, num_arrays = _save_manifest(output_dir, all_jobs, args.target_array_size, N_BOOTSTRAPS)
    _print_next_steps(output_dir, total_jobs, num_arrays)


# =============================================================================
# Mode: multivariate  (MultiScoreset for multi-assay genes)
# =============================================================================

def _discover_gene_groups(df, max_dimensions=None, manual_groups=None):
    if manual_groups is not None:
        all_datasets = set(df["Dataset"].unique())
        validated = {}
        for name, datasets in manual_groups.items():
            present = [d for d in datasets if d in all_datasets]
            missing = [d for d in datasets if d not in all_datasets]
            if missing:
                print(f"  Warning: {name} — missing: {missing}")
            if len(present) > 1:
                validated[name] = present
            else:
                print(f"  Warning: {name} — fewer than 2 valid datasets, skipping")
        return validated

    meta_analyses = {"F9_Popp_2025_model", "TP53_Fayer_2021_meta"}
    gene_to_datasets = defaultdict(list)
    for _, row in df.groupby("Dataset")["Gene"].first().reset_index().iterrows():
        if row["Dataset"] in meta_analyses:
            continue
        gene_to_datasets[row["Gene"]].append(row["Dataset"])

    result = {}
    for gene, datasets in gene_to_datasets.items():
        if len(datasets) <= 1:
            continue
        if max_dimensions is not None and len(datasets) > max_dimensions:
            print(f"  {gene}: {len(datasets)} datasets > max_dimensions={max_dimensions}, skipping")
            continue
        result[gene] = sorted(datasets)
    return result


def _generate_bootstrap_fit_jobs(ms, dataset_label, gene, save_dir, N_BOOTSTRAPS, NUM_FITS,
                                  component_range, constraint_modes, latent_q, init_strategy,
                                  min_overlap_rows=30, sample_balance_beta=None,
                                  extra_cjob_fields=None, master_seed=DEFAULT_MASTER_SEED):
    """Shared bootstrap x component x constraint job-generation loop, backed
    by Fit.generate_fit_jobs. Used by every multivariate pipeline in this
    file: the integrated-dataframe multi-assay genes (BRCA1/PTEN/MSH2/
    pillar-project), predictor-mv, and the FGFR/TP53/LABEL-seq/CARD11/
    combined gene-sets -- the only per-pipeline difference is how ``ms`` was
    built (see multivariate_data/*.py), not how jobs are generated from it.

    ``min_overlap_rows`` defaults to Fit.generate_fit_jobs's own default (30,
    i.e. unchanged behavior for the pre-existing integrated-dataframe
    pipeline). Callers standardizing on pattern_stratified_bootstrap's
    guarantee against sparse-dimension signal loss (FGFR/TP53/LABEL-seq/
    CARD11/predictor-mv/combined, per the consolidation plan) should pass
    ``min_overlap_rows=1`` explicitly.
    """
    from src.assay_calibration.fit_utils.fit import Fit

    constrained_flags = []
    if "con" in constraint_modes:
        constrained_flags.append(True)
    if "unc" in constraint_modes:
        constrained_flags.append(False)

    fitter = Fit(ms)
    all_jobs = []
    for bs in range(N_BOOTSTRAPS):
        jobs_by_key = {}
        ref_jobs = None
        for nc in component_range:
            for constrained in constrained_flags:
                mode_key = f"jobs_{nc}c_{'con' if constrained else 'unc'}"
                fit_kwargs = {
                    "latent_q": latent_q, "init_strategy": init_strategy,
                    "min_overlap_rows": min_overlap_rows,
                    "log_prefix": f"  {dataset_label}: ",
                    "verbose_overlap": bs == 0 and nc == component_range[0]
                                       and constrained == constrained_flags[0],
                }
                if sample_balance_beta is not None:
                    fit_kwargs["sample_balance_beta"] = sample_balance_beta
                if NUM_FITS is not None:
                    fit_kwargs["num_fits"] = NUM_FITS
                try:
                    jobs = fitter.generate_fit_jobs(
                        component_range=[nc], bootstrap_seed=bs, master_seed=master_seed,
                        check_monotonic=constrained, **fit_kwargs)
                except Exception as e:
                    print(f"  {dataset_label} bs={bs} {mode_key}: {e}")
                    jobs = []
                minimal = [{**_strip_minimal(j), "multivariate": True} for j in jobs]
                jobs_by_key[mode_key] = minimal
                if ref_jobs is None and jobs:
                    ref_jobs = jobs

        shared = _shared_data(ref_jobs or [])
        if shared is None:
            continue

        cjob = {
            "dataset_name": dataset_label,
            "gene": gene,
            "n_dimensions": ms.n_assays,
            "save_dir": save_dir,
            "bootstrap_seed": bs,
            "shared_data": shared,
            "multivariate": True,
            "num_fits_total": sum(len(v) for v in jobs_by_key.values()),
        }
        if extra_cjob_fields:
            cjob.update(extra_cjob_fields)
        cjob.update(jobs_by_key)
        all_jobs.append(cjob)

    return all_jobs


def _process_multivariate_gene(df_gene, gene, datasets, output_dir, N_BOOTSTRAPS, NUM_FITS,
                                clinvar_release, component_range, constraint_modes,
                                latent_q, init_strategy, population_type,
                                master_seed=DEFAULT_MASTER_SEED):
    from src.assay_calibration.multivariate_data.common import build_multiscoreset_from_long_dataframe

    gene_label = f"{gene}_mv{'_clinvar_' + clinvar_release if clinvar_release != '2026' else ''}"
    save_dir = os.path.join(output_dir, gene_label)

    scoreset_kwargs = dict(clinvar_release=clinvar_release, min_clinvar_star=1)
    if population_type:
        scoreset_kwargs["population_type"] = population_type

    ms = build_multiscoreset_from_long_dataframe(
        df_gene, gene, datasets, scoreset_kwargs=scoreset_kwargs,
    )
    if ms is None:
        print(f"  {gene}: fewer than 2 valid datasets, skipping")
        return None

    print(f"  {gene_label}: {ms.n_variants} variants, {ms.n_assays}D, "
          f"missing={ms.missing.mean()*100:.1f}%")

    return _generate_bootstrap_fit_jobs(
        ms, gene_label, gene, save_dir, N_BOOTSTRAPS, NUM_FITS,
        component_range, constraint_modes, latent_q, init_strategy,
        extra_cjob_fields={"assay_datasets": list(ms.dataset_names)},
        master_seed=master_seed,
    )


def _build_gene_set_ms_map(gene_set, args):
    """Dispatch to the gene-set-specific ingestion module for
    ``--gene-set`` values other than "integrated" (the pre-existing,
    untouched BRCA1/PTEN/MSH2/pillar-project path in run_multivariate).
    Returns {gene: MultiScoreset/BasicMultiScoreset}.
    """
    if gene_set == "fgfr":
        from src.assay_calibration.multivariate_data.fgfr import build_fgfr_multiscoresets
        return build_fgfr_multiscoresets(
            genes=args.genes, exclude_genes=args.exclude_genes,
            combine_genes=not getattr(args, "fgfr_separate", False),
        )

    if gene_set == "tp53":
        from src.assay_calibration.multivariate_data.tp53 import build_tp53_multiscoreset
        return {"TP53": build_tp53_multiscoreset(
            RPVS_ALL=args.rpvs_all, kawoligo_seed=args.kawoligo_seed,
            kawoligo_jitter_sigma=args.kawoligo_jitter_sigma,
        )}

    if gene_set == "labelseq":
        from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
        return build_labelseq_multiscoresets()

    if gene_set == "card11":
        from src.assay_calibration.multivariate_data.card11 import build_card11_multiscoreset
        return {"CARD11": build_card11_multiscoreset()}

    if gene_set == "combined":
        from src.assay_calibration.multivariate_data.combined import (
            build_functional_scoresets, build_combined_multiscoreset,
            get_functionally_assayed_protein_variants, DEFAULT_INTEGRATED_DATAFRAME,
        )
        from src.assay_calibration.multivariate_data.predictors import DEFAULT_GENES
        from src.assay_calibration.multivariate_data.common import resolve_clinvar_release

        df_path = args.dataframe or DEFAULT_INTEGRATED_DATAFRAME
        sep = "\t" if df_path.endswith((".tsv.gz", ".tsv")) else ","
        df = pd.read_csv(df_path, sep=sep, low_memory=False)
        genes = args.genes if args.genes else list(DEFAULT_GENES)

        result = {}
        for gene in genes:
            datasets = sorted(df[df["Gene"] == gene]["Dataset"].unique())
            if not datasets:
                print(f"  {gene}: no functional datasets in {df_path}, skipping")
                continue
            # Same ClinVar-release convention as the 'integrated' gene-set
            # (BRCA1/MSH2/PTEN/TP53 use 2018, matching their published
            # calibration) -- this used to default to "2026" unconditionally
            # here, silently diverging from 'integrated' for the 3 of these
            # 4 genes that are also predictor-mv DEFAULT_GENES.
            functional_scoresets = build_functional_scoresets(
                df, gene, datasets, clinvar_release=resolve_clinvar_release(gene),
            )
            # Variants with functional data must use the functional side's
            # ClinVar label (fixed at that release); predictor CSVs carry
            # their own, unversioned ClinVar label that would otherwise
            # leak a different vintage into the P/LP/B/LB control set for
            # any variant the functional assay also measured. Predictor-
            # only variants are unaffected and keep their own label.
            functionally_assayed = get_functionally_assayed_protein_variants(df, gene, datasets)
            ms = build_combined_multiscoreset(
                gene, functional_scoresets, datasets, args.predictor_data_dir,
                functionally_assayed_variants=functionally_assayed,
            )
            if ms is None:
                print(f"  {gene}: no predictor CSVs under {args.predictor_data_dir}, skipping")
                continue
            result[gene] = ms
        return result

    raise ValueError(f"Unknown --gene-set {gene_set!r}")


def _load_and_filter_gene_ms_map(gene_set, args):
    """Build + filter + print {gene: ms} for one --gene-set value. Split out
    of run_multivariate_gene_set so `all` (below) can build several
    gene-sets' ms maps without duplicating the filtering/printing logic.
    """
    gene_ms_map = _build_gene_set_ms_map(gene_set, args)

    # 'combined' and 'fgfr' already filter by --genes at the ingestion source
    # above (fgfr's combine_genes=True default returns one "FGFR_combined"
    # key, not gene-named, so this dict-key filter wouldn't even apply
    # correctly to it).
    if args.genes and gene_set not in ("combined", "fgfr"):
        upper = {g.upper(): g for g in args.genes}
        gene_ms_map = {g: ms for g, ms in gene_ms_map.items() if g.upper() in upper}
    if args.exclude_genes and gene_set != "fgfr":  # fgfr applies exclude_genes at the source too
        excluded = {g.upper() for g in args.exclude_genes}
        gene_ms_map = {g: ms for g, ms in gene_ms_map.items() if g.upper() not in excluded}

    print(f"\n[{gene_set}] {len(gene_ms_map)} genes:")
    for gene, ms in sorted(gene_ms_map.items()):
        print(f"  {gene}: {ms.n_variants} variants, {ms.n_assays}D, "
              f"sample_names={ms.sample_names}")
    return gene_ms_map


def _generate_jobs_for_gene_ms_map(gene_set, gene_ms_map, args):
    """Job-generation loop only (ms map already built/filtered/printed by
    _load_and_filter_gene_ms_map) -- shared between run_multivariate_gene_set
    (one gene-set, own output_dir/manifest) and run_mv_all (many gene-sets,
    one shared output_dir/manifest). Reuses the exact same
    _generate_bootstrap_fit_jobs loop as the integrated-dataframe path and
    predictor-mv, with min_overlap_rows=1 and pattern_stratified_bootstrap
    (via Fit.generate_fit_jobs's own mv=True default) standardized across
    all of them, per the consolidation plan.

    Returns (all_jobs, extra_index_fields).
    """
    from src.assay_calibration.multivariate_data.common import gene_set_dataset_label

    output_dir = args.output_dir
    N_BOOTSTRAPS = args.n_bootstraps
    NUM_FITS = args.num_fits
    component_range = args.components
    constraint_modes = _resolve_constraints(args.constraints)
    latent_q = 2

    def process_one(gene, ms):
        label = gene_set_dataset_label(gene, gene_set)
        save_dir = os.path.join(output_dir, label)
        print(f"  {label}: {ms.n_variants} variants, {ms.n_assays}D, "
              f"missing={ms.missing.mean()*100:.1f}%")
        return _generate_bootstrap_fit_jobs(
            ms, label, gene, save_dir, N_BOOTSTRAPS, NUM_FITS,
            component_range, constraint_modes, latent_q, args.init_strategy,
            min_overlap_rows=1,  # pattern_stratified_bootstrap already guards sparse dims
            extra_cjob_fields={"gene_set": gene_set, "assay_datasets": list(ms.dataset_names)},
            master_seed=args.seed,
        )

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_one)(gene, ms) for gene, ms in gene_ms_map.items()
    )

    all_jobs = [j for r in results if r for j in r]
    extra = {"gene_set": gene_set, "genes": sorted(gene_ms_map.keys())}
    return all_jobs, extra


def run_multivariate_gene_set(args):
    gene_set = args.gene_set
    gene_ms_map = _load_and_filter_gene_ms_map(gene_set, args)

    if args.list_only:
        return

    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "jobs"), exist_ok=True)

    all_jobs, extra = _generate_jobs_for_gene_ms_map(gene_set, gene_ms_map, args)
    extra.update({
        "component_range": args.components,
        "constraint_modes": _resolve_constraints(args.constraints),
    })
    total_jobs, num_arrays = _save_manifest(
        output_dir, all_jobs, args.target_array_size, args.n_bootstraps, extra)
    _print_next_steps(output_dir, total_jobs, num_arrays)


def _load_and_filter_gene_groups(args):
    """Load the integrated dataframe + discover/filter multi-assay gene
    groups. Split out of run_multivariate so `all` (below) can build the
    integrated pipeline's jobs alongside other pipelines' in one manifest.
    """
    df_path = args.dataframe or "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset.tsv.gz"
    sep = "\t" if df_path.endswith((".tsv.gz", ".tsv")) else ","
    df = pd.read_csv(df_path, sep=sep)

    max_dim = args.max_dimensions  # may be None
    gene_groups = _discover_gene_groups(df, max_dimensions=max_dim)

    if args.genes:
        upper = {g.upper(): g for g in args.genes}
        filtered = {g: v for g, v in gene_groups.items() if g.upper() in upper}
        for req in upper:
            if not any(g.upper() == req for g in filtered):
                print(f"  Warning: gene '{req}' not found in multi-assay groups")
        gene_groups = filtered

    if args.exclude_genes:
        excluded = {g.upper() for g in args.exclude_genes}
        gene_groups = {g: v for g, v in gene_groups.items() if g.upper() not in excluded}

    print(f"\nFound {len(gene_groups)} gene groups:")
    for g, ds in sorted(gene_groups.items()):
        print(f"  {g} ({len(ds)} assays): {ds}")
    return df, gene_groups


def _generate_integrated_jobs(df, gene_groups, args):
    """Job-generation loop only (gene groups already built/filtered/printed
    by _load_and_filter_gene_groups) -- shared between run_multivariate
    (own output_dir/manifest) and run_mv_all (one shared output_dir/manifest).
    Returns (all_jobs, extra_index_fields).
    """
    from src.assay_calibration.multivariate_data.common import resolve_clinvar_release

    output_dir = args.output_dir
    N_BOOTSTRAPS = args.n_bootstraps
    NUM_FITS = args.num_fits
    component_range = args.components
    constraint_modes = _resolve_constraints(args.constraints)
    latent_q = 2

    # Pass only the rows relevant to each gene group
    gene_partitions = {
        gene: df[df["Dataset"].isin(datasets)]
        for gene, datasets in gene_groups.items()
    }

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_multivariate_gene)(
            gene_partitions[gene], gene, datasets, output_dir, N_BOOTSTRAPS, NUM_FITS,
            clinvar_release=resolve_clinvar_release(gene),
            component_range=component_range,
            constraint_modes=constraint_modes,
            latent_q=latent_q,
            init_strategy=args.init_strategy,
            population_type=args.population_type,
            master_seed=args.seed,
        )
        for gene, datasets in gene_groups.items()
    )

    all_jobs = [j for r in results if r for j in r]
    extra = {"gene_groups": {g: ds for g, ds in gene_groups.items()}}
    return all_jobs, extra


def run_multivariate(args):
    if getattr(args, "gene_set", "integrated") != "integrated":
        return run_multivariate_gene_set(args)

    df, gene_groups = _load_and_filter_gene_groups(args)

    if args.list_only:
        return

    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "jobs"), exist_ok=True)

    all_jobs, extra = _generate_integrated_jobs(df, gene_groups, args)
    extra.update({
        "component_range": args.components,
        "constraint_modes": _resolve_constraints(args.constraints),
    })
    total_jobs, num_arrays = _save_manifest(
        output_dir, all_jobs, args.target_array_size, args.n_bootstraps, extra)
    _print_next_steps(output_dir, total_jobs, num_arrays)


# =============================================================================
# Mode: predictor-mv  (BasicMultiScoreset from predictor CSVs)
# =============================================================================

def _resolve_predictor_data_dir(args):
    """predictor-mv's own subcommand uses --data-dir (required there);
    `all` uses --predictor-data-dir (shared with 'combined'). Prefer
    whichever is actually set so both call sites work unmodified.
    """
    return getattr(args, "data_dir", None) or getattr(args, "predictor_data_dir", None)


def _load_predictor_mv_by_gene(args):
    from src.assay_calibration.multivariate_data.predictors import load_predictor_data

    data_dir = _resolve_predictor_data_dir(args)
    print(f"Loading predictor CSVs from {data_dir}...")
    by_gene = load_predictor_data(data_dir, genes=args.genes)
    if args.exclude_genes:
        excluded = {g.upper() for g in args.exclude_genes}
        by_gene = {g: v for g, v in by_gene.items() if g.upper() not in excluded}
    return by_gene


def _generate_predictor_mv_jobs(by_gene, args):
    """Job-generation loop only (by_gene already loaded/filtered by
    _load_predictor_mv_by_gene) -- shared between run_predictor_mv (own
    output_dir/manifest) and run_mv_all (one shared output_dir/manifest).
    Returns (all_jobs, extra_index_fields).
    """
    from src.assay_calibration.multivariate_data.predictors import (
        PREDICTORS, build_basic_multi_scoreset, predictor_dataset_label,
    )

    N_BOOTSTRAPS = args.n_bootstraps
    NUM_FITS = args.num_fits
    output_dir = args.output_dir
    component_range = args.components
    constraint_modes = _resolve_constraints(args.constraints)
    latent_q = 2

    def process_one(gene, predictor_dfs):
        label = predictor_dataset_label(gene)
        save_dir = os.path.join(output_dir, label)
        ms, info = build_basic_multi_scoreset(gene, predictor_dfs)
        if ms is None:
            print(f"  {gene}: skipping — {info}")
            return None
        print(f"  {label}: {ms.n_variants} variants, {ms.n_assays}D")

        return _generate_bootstrap_fit_jobs(
            ms, label, gene, save_dir, N_BOOTSTRAPS, NUM_FITS,
            component_range, constraint_modes, latent_q, args.init_strategy,
            min_overlap_rows=1,  # pattern_stratified_bootstrap already guards sparse dims
            sample_balance_beta=args.sample_balance_beta,
            master_seed=args.seed,
        )

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_one)(gene, predictor_dfs)
        for gene, predictor_dfs in by_gene.items()
    )

    all_jobs = [j for r in results if r for j in r]
    extra = {"predictors": list(PREDICTORS)}
    return all_jobs, extra


def run_predictor_mv(args):
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "jobs"), exist_ok=True)

    by_gene = _load_predictor_mv_by_gene(args)
    if not by_gene:
        print("No predictor data found. Exiting.")
        return

    all_jobs, extra = _generate_predictor_mv_jobs(by_gene, args)
    extra.update({
        "component_range": args.components,
        "constraint_modes": _resolve_constraints(args.constraints),
    })
    total_jobs, num_arrays = _save_manifest(
        output_dir, all_jobs, args.target_array_size, args.n_bootstraps, extra)
    _print_next_steps(output_dir, total_jobs, num_arrays)


# =============================================================================
# Mode: all  (merge every new multivariate pipeline into ONE job array,
# in priority order)
# =============================================================================

_MV_ALL_PIPELINE_CHOICES = ["integrated", "fgfr", "tp53", "labelseq", "card11", "predictor-mv", "combined"]
_MV_ALL_GENE_SET_PIPELINES = {"fgfr", "tp53", "labelseq", "card11", "combined"}


def run_mv_all(args):
    """Build jobs for every pipeline in --pipelines (default order:
    labelseq tp53 card11 fgfr predictor-mv combined) and merge them into
    ONE _save_manifest call for a single output_dir.

    Ordering: _save_manifest chunks `all_jobs` into array files preserving
    list order, so whichever pipeline's jobs are concatenated first occupy
    the lowest array indices. Combined with MAX_CONCURRENT (SLURM) or
    a bounded xargs -P concurrency (run_local_array.sh), earlier-listed
    pipelines' jobs tend to be scheduled and finish first -- a soft
    priority bias, not a hard dependency (nothing blocks a later
    pipeline's jobs from starting before an earlier one finishes; for a
    hard sequential guarantee, run separate `multivariate --gene-set X`
    manifests and chain them with SLURM --dependency, or invoke
    run_local_array.sh once per gene-set's own output_dir in sequence).
    """
    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, "jobs"), exist_ok=True)

    all_jobs = []
    per_pipeline = {}

    for pipeline in args.pipelines:
        print(f"\n{'=' * 80}\nPipeline: {pipeline}  (array indices will start at job #{len(all_jobs)})\n{'=' * 80}")

        if pipeline == "integrated":
            df, gene_groups = _load_and_filter_gene_groups(args)
            jobs, pextra = _generate_integrated_jobs(df, gene_groups, args)
        elif pipeline == "predictor-mv":
            by_gene = _load_predictor_mv_by_gene(args)
            if not by_gene:
                print("  No predictor data found, skipping.")
                jobs, pextra = [], {}
            else:
                jobs, pextra = _generate_predictor_mv_jobs(by_gene, args)
        elif pipeline in _MV_ALL_GENE_SET_PIPELINES:
            gene_ms_map = _load_and_filter_gene_ms_map(pipeline, args)
            jobs, pextra = _generate_jobs_for_gene_ms_map(pipeline, gene_ms_map, args)
        else:
            raise ValueError(f"Unknown pipeline {pipeline!r}")

        print(f"  {pipeline}: {len(jobs)} jobs")
        per_pipeline[pipeline] = {**pextra, "n_jobs": len(jobs),
                                   "job_index_range": [len(all_jobs), len(all_jobs) + len(jobs)]}
        all_jobs.extend(jobs)

    extra = {
        "component_range": args.components,
        "constraint_modes": _resolve_constraints(args.constraints),
        "pipeline_order": list(args.pipelines),
        "per_pipeline": per_pipeline,
    }
    total_jobs, num_arrays = _save_manifest(
        output_dir, all_jobs, args.target_array_size, args.n_bootstraps, extra)
    _print_next_steps(output_dir, total_jobs, num_arrays)
    print(f"\nPriority order: {args.pipelines} -- earlier pipelines occupy lower array "
          f"indices, so they're scheduled/tend to finish first under bounded concurrency "
          f"(MAX_CONCURRENT for SLURM, or the concurrency arg to run_local_array.sh).")


# =============================================================================
# CLI
# =============================================================================

def _resolve_constraints(constraint_list: list[str]) -> list[str]:
    modes = set()
    for c in constraint_list:
        if c == "both":
            modes.update(["con", "unc"])
        else:
            modes.add(c)
    return sorted(modes)


def _add_common_args(p):
    p.add_argument("--output-dir", required=True, help="Directory for jobs and results")
    p.add_argument("--target-array-size", type=int, default=1000,
                   help="Target SLURM array size (default: 1000)")
    p.add_argument("--n-jobs", type=int, default=30,
                   help="Parallel workers for job generation (default: 30)")
    p.add_argument("--seed", type=_parse_seed, default=DEFAULT_MASTER_SEED,
                   help="Master reproducibility seed (default: 0, matches historical "
                        "bootstrap composition). Pass a non-zero int to also perturb "
                        "bootstrap composition, or 'none' to opt out entirely (true "
                        "entropy, non-reproducible).")


def main():
    parser = argparse.ArgumentParser(
        description="ExCALIBR batch job manifest generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── default / pillar_project ─────────────────────────────────────────────
    def _add_default_mode_args(p):
        _add_common_args(p)
        p.add_argument("--dataframe", default=None,
                       help="Input TSV/CSV (default: integrated_variant_effect_dataset.tsv.gz)")
        p.add_argument("--config-file", default=None,
                       help="Dataset config JSON (default: dataset_configs_jan_2026.json)")
        p.add_argument("--population-type", default=None)
        p.add_argument("--components", nargs="+", type=int, default=None,
                       help="Override component counts (default: per-config)")
        p.add_argument("--force-gaussian", action="store_true",
                       help="Ablation: lock skew=0 throughout init and every EM "
                            "M-step, collapsing skew-normal components to plain "
                            "Gaussians while keeping the rest of the pipeline "
                            "(bootstraps, fit counts, constraints) identical")
        p.add_argument("--datasets", nargs="+", default=None,
                       help="Only generate jobs for these Dataset name(s) (as they "
                            "appear in --dataframe's Dataset column), instead of "
                            "every unique dataset found there. Combinable with "
                            "--datasets-file (union of both).")
        p.add_argument("--datasets-file", default=None,
                       help="Path to a text file with one Dataset name per line "
                            "(blank lines and '#'-prefixed comments ignored) to "
                            "restrict job generation to, instead of every unique "
                            "dataset in --dataframe. Combinable with --datasets "
                            "(union of both).")
        return p

    p_def = sub.add_parser("default", help="Univariate Scoreset from integrated TSV")
    _add_default_mode_args(p_def)
    p_def.set_defaults(func=run_default, clinvar_default_release="2026")

    p_pillar = sub.add_parser(
        "pillar_project",
        help="Same as 'default', but ClinVar 2025 in place of ClinVar 2026 "
             "everywhere (datasets forced to 2018 are unaffected)",
    )
    _add_default_mode_args(p_pillar)
    p_pillar.set_defaults(func=run_default, clinvar_default_release="2025")

    # ── mavedb ───────────────────────────────────────────────────────────────
    p_mv = sub.add_parser("mavedb", help="Univariate Scoreset.from_mavedb")
    _add_common_args(p_mv)
    p_mv.add_argument("--dataframe", required=True)
    p_mv.add_argument("--score-cols", nargs="+", required=True,
                      help="Score columns to iterate over")
    p_mv.add_argument("--dataset-col", default="gene_symbol")
    p_mv.add_argument("--population-type", default=None)
    p_mv.add_argument("--regularization-type", default=None)
    p_mv.add_argument("--components", nargs="+", type=int, default=None)
    p_mv.add_argument("--splice-measure",
                      type=lambda x: x.lower() in ("yes", "true", "1"),
                      required=True, metavar="yes|no")
    p_mv.set_defaults(func=run_mavedb)

    # ── basicscoreset ────────────────────────────────────────────────────────
    p_bs = sub.add_parser("basicscoreset", help="Univariate BasicScoreset")
    _add_common_args(p_bs)
    grp = p_bs.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataframe", default=None,
                     help="CSV/TSV with Dataset + score + sample_assignments columns")
    grp.add_argument("--data-dir", default=None,
                     help="Directory of per-dataset CSV files")
    p_bs.add_argument("--components", nargs="+", type=int, default=None,
                      help="Fallback component counts when no config-file match (default: 2 3)")
    p_bs.add_argument("--n-bootstraps", type=int, default=_DEFAULT_N_BOOTSTRAPS,
                      help=f"Bootstrap iterations (default: {_DEFAULT_N_BOOTSTRAPS})")
    p_bs.add_argument("--fits-3c", type=int, default=_DEFAULT_NUM_FITS,
                      help=f"Fits per bootstrap for 3c (default: {_DEFAULT_NUM_FITS}; "
                           "set to 8 to cover exactly the 2^3 lambda sign patterns)")
    p_bs.add_argument("--fits-2c", type=int, default=_DEFAULT_NUM_FITS,
                      help=f"Fits per bootstrap for 2c (default: {_DEFAULT_NUM_FITS}; "
                           "set to 4 to cover exactly the 2^2 lambda sign patterns)")
    p_bs.add_argument("--config-file", default=None,
                      help="dataset_configs JSON for per-dataset n_c lookup")
    p_bs.add_argument("--name-strip", default=None,
                      help="Regex to strip robustness suffix before config lookup "
                           r"(e.g. '_ds[0-9]+_s[0-9]+$|_disc[0-9.]+_s[0-9]+$')")
    p_bs.set_defaults(func=run_basicscoreset)

    # ── multivariate ─────────────────────────────────────────────────────────
    p_multi = sub.add_parser("multivariate", help="MultiScoreset for multi-assay genes")
    _add_common_args(p_multi)
    p_multi.add_argument("--dataframe", default=None)
    p_multi.add_argument("--genes", nargs="+", default=None,
                         help="Specific genes to process (default: all multi-assay)")
    p_multi.add_argument("--exclude-genes", nargs="+", default=None,
                         help="Genes to drop from the run")
    p_multi.add_argument("--components", nargs="+", type=int, default=[4],
                         help="Component counts (default: 4)")
    p_multi.add_argument("--constraints", nargs="+",
                         choices=["con", "unc", "both"], default=["unc"],
                         help="Constraint modes (default: unc)")
    p_multi.add_argument("--init-strategy", default="kmeans",
                         choices=["kmeans", "anchored"])
    p_multi.add_argument("--max-dimensions", type=int, default=None,
                         help="Skip genes with more datasets than this")
    p_multi.add_argument("--population-type", default=None)
    p_multi.add_argument("--n-bootstraps", type=int, default=1000)
    p_multi.add_argument("--num-fits", type=int, default=100,
                         help="Override dynamic NUM_FITS (default: 100)")
    p_multi.add_argument("--list-only", action="store_true",
                         help="Print gene groups and exit")
    p_multi.add_argument("--gene-set", default="integrated",
                         choices=["integrated", "fgfr", "tp53", "labelseq", "card11", "combined"],
                         help="Ingestion source (default: integrated, the pre-existing "
                              "BRCA1/PTEN/MSH2/pillar-project path via --dataframe/--datasets). "
                              "Other values use the pluggable multivariate_data/*.py ingestion "
                              "modules and standardize on min_overlap_rows=1.")
    p_multi.add_argument("--rpvs-all", action="store_true",
                         help="[--gene-set tp53] Use the full RPV list instead of the "
                              "high-confidence subset (default: high-confidence only)")
    p_multi.add_argument("--kawoligo-seed", type=int, default=0,
                         help="[--gene-set tp53] Seed for KawOligo's jitter (default: 0)")
    p_multi.add_argument("--kawoligo-jitter-sigma", type=float, default=0.1,
                         help="[--gene-set tp53] Gaussian jitter sigma for KawOligo's "
                              "{Tetramer,Dimer,Monomer} -> {0,0.5,1} encoding (default: 0.1)")
    p_multi.add_argument("--fgfr-separate", action="store_true",
                         help="[--gene-set fgfr] Fit FGFR1/2/3/4 as separate MultiScoresets "
                              "(one per gene) instead of the default: pooling all target "
                              "genes' P/LP/B/LB/gnomAD/Synonymous variants into one combined "
                              "MultiScoreset")
    p_multi.add_argument("--predictor-data-dir",
                         default="/data/ross/assay_calibration/predictor_calibrations/"
                                 "single_gene_calibration_data",
                         help="[--gene-set combined] Directory containing "
                              "{gene}/{gene}_{predictor}.csv.gz")
    p_multi.set_defaults(func=run_multivariate)

    # ── predictor-mv ─────────────────────────────────────────────────────────
    p_pred = sub.add_parser("predictor-mv", help="BasicMultiScoreset from predictor CSVs")
    _add_common_args(p_pred)
    p_pred.add_argument("--data-dir", required=True,
                        help="Directory containing {gene}/{gene}_{predictor}.csv.gz")
    p_pred.add_argument("--genes", nargs="+", default=None)
    p_pred.add_argument("--exclude-genes", nargs="+", default=None,
                        help="Genes to drop from the run")
    p_pred.add_argument("--components", nargs="+", type=int, default=[4])
    p_pred.add_argument("--constraints", nargs="+",
                        choices=["con", "unc", "both"], default=["unc"])
    p_pred.add_argument("--init-strategy", default="kmeans",
                        choices=["anchored", "kmeans"])
    p_pred.add_argument("--n-bootstraps", type=int, default=1000)
    p_pred.add_argument("--num-fits", type=int, default=100)
    p_pred.add_argument("--sample-balance-beta", type=float, default=0,
                        help="Sample-balanced M-step β ∈ [0,1] (default: 0)")
    p_pred.set_defaults(func=run_predictor_mv)

    # ── mv_all ───────────────────────────────────────────────────────────────
    p_mv_all = sub.add_parser(
        "mv_all",
        help="Merge FGFR/TP53/LABEL-seq/CARD11/predictor-mv/combined (and, "
             "opt-in, the integrated pillar-project path) into ONE job array, "
             "in priority order",
    )
    _add_common_args(p_mv_all)
    p_mv_all.add_argument("--pipelines", nargs="+", choices=_MV_ALL_PIPELINE_CHOICES,
                       default=["labelseq", "tp53", "card11", "fgfr", "predictor-mv", "combined"],
                       help="Priority order -- earlier pipelines get lower array indices "
                            "(default: labelseq tp53 card11 fgfr predictor-mv combined). "
                            "'integrated' (BRCA1/PTEN/MSH2/pillar-project) is opt-in.")
    p_mv_all.add_argument("--dataframe", default=None,
                       help="Integrated variant-effect TSV/CSV, used by the 'integrated' "
                            "and 'combined' pipelines")
    p_mv_all.add_argument("--predictor-data-dir",
                       default="/data/ross/assay_calibration/predictor_calibrations/"
                               "single_gene_calibration_data",
                       help="Directory containing {gene}/{gene}_{predictor}.csv.gz, used "
                            "by 'predictor-mv' and 'combined'")
    p_mv_all.add_argument("--genes", nargs="+", default=None,
                       help="Restrict every pipeline to these genes (default: each "
                            "pipeline's own full gene set)")
    p_mv_all.add_argument("--exclude-genes", nargs="+", default=None)
    p_mv_all.add_argument("--components", nargs="+", type=int, default=[4])
    p_mv_all.add_argument("--constraints", nargs="+",
                       choices=["con", "unc", "both"], default=["unc"])
    p_mv_all.add_argument("--init-strategy", default="kmeans",
                       choices=["kmeans", "anchored"])
    p_mv_all.add_argument("--n-bootstraps", type=int, default=1000)
    p_mv_all.add_argument("--num-fits", type=int, default=100)
    p_mv_all.add_argument("--sample-balance-beta", type=float, default=0,
                       help="[predictor-mv] Sample-balanced M-step β ∈ [0,1] (default: 0)")
    p_mv_all.add_argument("--max-dimensions", type=int, default=None,
                       help="['integrated' pipeline only] skip genes with more datasets than this")
    p_mv_all.add_argument("--population-type", default=None,
                       help="['integrated' pipeline only]")
    p_mv_all.add_argument("--rpvs-all", action="store_true",
                       help="[tp53] use the full RPV list instead of the high-confidence subset")
    p_mv_all.add_argument("--kawoligo-seed", type=int, default=0,
                       help="[tp53] seed for KawOligo's jitter (default: 0)")
    p_mv_all.add_argument("--kawoligo-jitter-sigma", type=float, default=0.1,
                       help="[tp53] Gaussian jitter sigma for KawOligo's "
                            "{Tetramer,Dimer,Monomer} -> {0,0.5,1} encoding (default: 0.1)")
    p_mv_all.add_argument("--fgfr-separate", action="store_true",
                       help="[fgfr] fit FGFR1/2/3/4 as separate MultiScoresets instead of "
                            "the default: one pooled MultiScoreset across all target genes")
    p_mv_all.set_defaults(func=run_mv_all)

    args = parser.parse_args()

    # Sample one dataset slice to estimate per-worker memory more accurately
    _sample_slice = None
    df_path = getattr(args, "dataframe", None)
    if df_path and os.path.exists(df_path):
        try:
            sep = "\t" if df_path.endswith((".tsv.gz", ".tsv")) else ","
            _df_peek = pd.read_csv(df_path, sep=sep)
            if "Dataset" in _df_peek.columns:
                first_ds = _df_peek["Dataset"].iloc[0]
                _sample_slice = _df_peek[_df_peek["Dataset"] == first_ds]
            else:
                _sample_slice = _df_peek.iloc[:500]
        except Exception:
            pass

    args.n_jobs = _resolve_n_jobs(args.n_jobs, sample_slice=_sample_slice)
    print("=" * 80)
    print(f"ExCALIBR job manifest — mode: {args.mode}  (workers: {args.n_jobs})")
    print("=" * 80)
    args.func(args)


if __name__ == "__main__":
    main()
