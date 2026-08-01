#!/usr/bin/env python3
"""
Standalone plotting for tests/benchmark_bootstrap_reduction.py's output:
one figure per dataset, overlaying every bootstrap-count level's own
[p5,p50,p95] LR+ band, point-range boundaries, and (if --precomputed-fits
is given) mixture-density curves.

The actual plotting logic lives in
analysis.robustness.plot_bootstrap_reduction_config_summary so it's usable
both from here and from the analysis/ notebook-style pipeline
(analyze_pipeline_output.py) -- this script is just the data-assembly layer
that knows tests/benchmark_bootstrap_reduction.py's on-disk directory
layout ({output_dir}/{dataset}/level_{N}/{dataset}_{n_c}_{benign}_
calibration.json) and turns it into that function's plain
{N: {"calib_path", "lr_path", "fits"}} input.

Usage:
    python tests/plot_bootstrap_reduction_config.py \\
        --bootstrap-reduction-dir /tmp/bootstrap_reduction_full3 \\
        --figure-dir /tmp/bootstrap_reduction_full3/figures
    python tests/plot_bootstrap_reduction_config.py \\
        --bootstrap-reduction-dir /tmp/bootstrap_reduction_full3 \\
        --figure-dir /tmp/figures --datasets BAP1_Waters_2024 --no-density
"""
import argparse
import gzip
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")

from run_igvf_batch import parse_dataset_config
from src.assay_calibration.pipeline.config import PipelineConfig
from src.assay_calibration.pipeline.utils import load_dataset_from_df
from tests.benchmark_bootstrap_reduction import (
    build_dataset_df, _DEFAULT_PRECOMPUTED_FITS, _DEFAULT_DATAFRAME, _DEFAULT_CONFIG,
)
from analysis.robustness import plot_bootstrap_reduction_config_summary


def build_reference_df(scoreset) -> pd.DataFrame:
    """{"score", "sample"} DataFrame matching analysis.plot_common.sample_matches'
    expected pipe-separated multi-label "sample" column -- a variant can
    belong to more than one sample category at once (e.g. both "Synonymous"
    and "population"), so this must NOT be a single-label assignment."""
    sample_names = [name for _, name in scoreset.samples]
    sa = np.asarray(scoreset.sample_assignments)
    sample_col = [
        "|".join(sample_names[j] for j in range(sa.shape[1]) if sa[i, j])
        for i in range(sa.shape[0])
    ]
    return pd.DataFrame({"score": np.asarray(scoreset.scores, dtype=float), "sample": sample_col})


def _process_one_dataset(dataset_name, ds_dir, dataset_configs, dataframe_path, dataset_df,
                         all_bootstrap_results, figure_dir):
    """One dataset's worth of work: resolve levels on disk, build its
    Scoreset (real per-dataset cost -- ClinVar/splicing-filter lookups, not
    "cheap" preprocessing), and render its config-summary figure.

    Top-level (not a closure) so joblib/loky can dispatch it to a worker
    process -- this is what makes the per-dataset loop in main() actually
    parallel instead of building all ~90 Scoresets one at a time in the main
    process, which was the exact bottleneck already found and fixed in
    tests/benchmark_num_fits_dataframe.py (see build_scoreset_for_dataset's
    docstring there for the original diagnosis).

    Any exception past this point is caught and turned into a (None) skip
    rather than propagating: joblib's default Parallel() is fail-fast -- one
    dataset hitting an unexpected pipeline edge case previously aborted the
    entire batch, cancelling every other still-pending dataset's plot, not
    just the one that failed (same class of bug already fixed for
    tests/benchmark_bootstrap_reduction.py's run_one_level). Row 0's density
    overlay already has its own finer-grained per-level try/except inside
    plot_bootstrap_reduction_config_summary; this is the last-resort net for
    anything else.
    """
    level_dirs = sorted(
        (d for d in ds_dir.iterdir() if d.is_dir() and d.name.startswith("level_")),
        key=lambda d: int(d.name.replace("level_", "")), reverse=True,
    )
    if not level_dirs:
        return dataset_name, "no level_* dirs found"

    config_entry = dataset_configs.get(dataset_name, ["3c", "avg"])
    n_c, benign_method, _ = parse_dataset_config(config_entry)
    if n_c in ("", "all", None):
        n_c = "3c"
    n_c_str = n_c if n_c.endswith("c") else f"{n_c}c"
    comp_key = f"{n_c_str}_{benign_method}"

    levels_data = {}
    for level_dir in level_dirs:
        N = int(level_dir.name.replace("level_", ""))
        calib_path = level_dir / f"{dataset_name}_{comp_key}_calibration.json"
        lr_path = level_dir / f"{dataset_name}_{comp_key}_lr_values.json.gz"
        if not (calib_path.exists() and lr_path.exists()):
            continue
        fits = None
        if all_bootstrap_results is not None:
            boot = all_bootstrap_results.get(dataset_name)
            if boot is not None:
                seeds_sorted = sorted(boot, key=lambda k: int(k))
                fits = [
                    boot[s][n_c_str] for s in seeds_sorted[:N]
                    if isinstance(boot[s], dict) and boot[s].get(n_c_str) is not None
                ] or None
        levels_data[N] = {"calib_path": calib_path, "lr_path": lr_path, "fits": fits}

    if not levels_data:
        return dataset_name, "no complete levels found"

    clinvar_release = "2018" if dataset_name.endswith("_clinvar_2018") else "2025"
    cfg = PipelineConfig(
        dataset_csv=dataframe_path, dataset_name=dataset_name,
        output_dir=str(ds_dir), components=[int(n_c_str[0])],
        benign_method=benign_method, clinvar_release=clinvar_release,
        min_clinvar_star=1, population_type="gnomAD",
    )
    try:
        scoreset = load_dataset_from_df(dataset_df, cfg)
        reference_df = build_reference_df(scoreset)

        print(f"{dataset_name}: {len(levels_data)} levels ({sorted(levels_data.keys(), reverse=True)})",
              flush=True)
        plot_bootstrap_reduction_config_summary(
            dataset_name, reference_df, levels_data, figure_dir=figure_dir, show=False,
        )
    except Exception as e:
        return dataset_name, f"{type(e).__name__}: {e}"
    return dataset_name, None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-reduction-dir", required=True,
                        help="Output dir from tests/benchmark_bootstrap_reduction.py")
    parser.add_argument("--precomputed-fits", default=_DEFAULT_PRECOMPUTED_FITS,
                        help="Optional: enables Row 0 density overlay by re-slicing each "
                             "level's own bootstrap-fit pool from this file")
    parser.add_argument("--dataframe", default=_DEFAULT_DATAFRAME)
    parser.add_argument("--config-file", default=_DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--no-density", action="store_true",
                        help="Skip Row 0 density overlay even if --precomputed-fits is given "
                             "(faster; density needs decompressing+re-slicing the fits file)")
    args = parser.parse_args()

    br_dir = Path(args.bootstrap_reduction_dir)
    datasets = sorted(
        d.name for d in br_dir.iterdir()
        if d.is_dir() and any(d.glob("level_*"))
    )
    if args.datasets:
        requested = set(args.datasets)
        datasets = [d for d in datasets if d in requested]
    print(f"Found {len(datasets)} datasets with bootstrap-reduction output")

    with open(args.config_file) as f:
        dataset_configs = json.load(f)

    sep = "\t" if args.dataframe.endswith((".tsv", ".tsv.gz")) else ","
    df = pd.read_csv(args.dataframe, sep=sep)

    all_bootstrap_results = None
    if args.precomputed_fits and not args.no_density:
        print(f"Loading {args.precomputed_fits} for Row 0 density overlay...")
        with gzip.open(args.precomputed_fits, "rt", encoding="utf-8") as f:
            all_bootstrap_results = json.load(f)

    # Pre-filter each dataset's (small) slice in the main process first --
    # cheap, vectorized -- so each worker gets a small pickle instead of the
    # full multi-tens-of-MB dataframe; the *actual* per-dataset cost
    # (Scoreset construction) still happens inside the parallel dispatch
    # below, in _process_one_dataset.
    per_dataset_df = {d: build_dataset_df(d, df) for d in datasets}

    print(f"Dispatching {len(datasets)} datasets across {mp.cpu_count()} CPUs...", flush=True)
    results = Parallel(n_jobs=-1, batch_size=1, backend="loky", verbose=10)(
        delayed(_process_one_dataset)(
            dataset_name, br_dir / dataset_name, dataset_configs, args.dataframe,
            per_dataset_df[dataset_name], all_bootstrap_results, args.figure_dir,
        )
        for dataset_name in datasets
    )
    for dataset_name, err in results:
        if err is not None:
            print(f"{dataset_name}: SKIP ({err})")


if __name__ == "__main__":
    main()
