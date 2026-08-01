"""Speed benchmark for run_pipeline.py's default parameters on the example dataset.

Times Step 1 (bootstrap fitting) separately from the rest of calibration
(model selection, visualization, per-variant evidence, saving) for both a
2-component and a 3-component fit, using run_pipeline.py's actual default
bootstrap settings (--n-bootstraps 20 --fits-per-bootstrap 8).

Reports wall-clock time and a derived "fits per core-second" throughput
figure for Step 1 (the dominant, CPU-bound cost), so the result can be
extrapolated to any --n-bootstraps/--fits-per-bootstrap/--n-jobs combination
via:

    estimated_step1_seconds = (n_bootstraps * fits_per_bootstrap) / (fits_per_core_second * n_jobs)

Run with:
    source activate excalibr
    python tests/benchmark_run_pipeline_speed.py --n-jobs 64
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.assay_calibration.pipeline.config import PipelineConfig
from src.assay_calibration.pipeline.fit_bootstrap import BootstrapRunner
from src.assay_calibration.pipeline.visualize import generate_visualizations
from src.assay_calibration.pipeline.variant_evidence import compute_variant_table
from src.assay_calibration.pipeline.utils import setup_logging, save_results, load_dataset_from_df

EXAMPLE_DATASET = str(Path(__file__).resolve().parent.parent / "example" / "MSH2_Jia_2021.csv")
DATASET_NAME = "MSH2_Jia_2021"   # must match the CSV's "Dataset" column value
DEFAULT_N_BOOTSTRAPS = 20        # matches run_pipeline.py --n-bootstraps default
DEFAULT_FITS_PER_BOOTSTRAP = 8   # matches run_pipeline.py --fits-per-bootstrap default


def time_one_component_count(n_c: int, n_jobs: int, output_dir: str) -> dict:
    config = PipelineConfig(
        dataset_csv=EXAMPLE_DATASET,
        dataset_name=DATASET_NAME,
        output_dir=output_dir,
        n_bootstraps=DEFAULT_N_BOOTSTRAPS,
        num_fits_per_bootstrap=DEFAULT_FITS_PER_BOOTSTRAP,
        components=[n_c],
        n_jobs=n_jobs,
        clinvar_release="2026",
        auto_select_model=False,  # single component count; model selection N/A
    )
    logger = setup_logging(config.output_dir, config.dataset_name)

    # Step 1: bootstrap fitting (the dominant, CPU-bound cost)
    t0 = time.perf_counter()
    runner = BootstrapRunner(config)
    bootstrap_results, dataset_splits = runner.run()
    step1_seconds = time.perf_counter() - t0

    valid_fits = sum(
        1 for seed_results in bootstrap_results.values()
        for v in seed_results.values() if v is not None
    )
    total_fits = len(bootstrap_results) * len(config.components)

    # Steps 2-4: model selection (skipped, single n_c) + visualization/export +
    # per-variant evidence table + save -- "the rest of calibration".
    t0 = time.perf_counter()
    selected_components = {f"{n_c}c": n_c}
    results = generate_visualizations(
        bootstrap_results=bootstrap_results,
        config=config,
        selected_components=selected_components,
        logger=logger,
    )
    df_vt = pd.read_csv(config.dataset_csv)
    scoreset_vt = load_dataset_from_df(df_vt, config)
    for component_key, calibration in results.items():
        variant_df = compute_variant_table(
            scoreset=scoreset_vt, calibration=calibration, config=config,
            dataset_splits=None, logger=logger,
        )
        variant_df.to_csv(Path(config.output_dir) / f"{config.dataset_name}_{component_key}_variants.csv",
                           index=False)
    save_results(results=results, bootstrap_results=bootstrap_results, config=config,
                 logger=logger, selected_k=None)
    rest_seconds = time.perf_counter() - t0

    n_fit_units = DEFAULT_N_BOOTSTRAPS * DEFAULT_FITS_PER_BOOTSTRAP
    return {
        "n_c": n_c,
        "n_jobs": n_jobs,
        "valid_fits": valid_fits,
        "total_fits": total_fits,
        "step1_bootstrap_fitting_seconds": step1_seconds,
        "rest_of_calibration_seconds": rest_seconds,
        "total_seconds": step1_seconds + rest_seconds,
        "fits_per_core_second_step1": n_fit_units / (step1_seconds * n_jobs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=64,
                        help="Parallel jobs for bootstrap fitting (default: 64)")
    parser.add_argument("--output-dir", default="/tmp/benchmark_run_pipeline_speed",
                        help="Scratch output directory")
    parser.add_argument("--results-json", default=None,
                        help="Optional path to write results as JSON")
    args = parser.parse_args()

    print(f"Dataset: {EXAMPLE_DATASET}")
    print(f"n_bootstraps={DEFAULT_N_BOOTSTRAPS}, fits_per_bootstrap={DEFAULT_FITS_PER_BOOTSTRAP}, "
          f"n_jobs={args.n_jobs}\n")

    results = []
    for n_c in (3, 2):
        print(f"=== {n_c}c ===")
        r = time_one_component_count(n_c, args.n_jobs, args.output_dir)
        results.append(r)
        print(f"  Step 1 (bootstrap fitting): {r['step1_bootstrap_fitting_seconds']:.1f}s "
              f"({r['valid_fits']}/{r['total_fits']} valid fits, "
              f"{r['fits_per_core_second_step1']:.3f} fits/core-second)")
        print(f"  Rest of calibration:        {r['rest_of_calibration_seconds']:.1f}s")
        print(f"  Total:                      {r['total_seconds']:.1f}s\n")

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.results_json}")


if __name__ == "__main__":
    main()
