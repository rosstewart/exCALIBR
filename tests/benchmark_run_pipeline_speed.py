"""Speed benchmark for run_pipeline.py's default parameters on the example dataset.

Times Step 1 (bootstrap fitting) separately from the rest of calibration
(model selection, visualization, per-variant evidence, saving).

CPU mode benchmarks both 2c and 3c fits using run_pipeline.py's default
bootstrap settings (--n-bootstraps 20 --fits-per-bootstrap 8) and reports
"fits per core-second" for Step 1, which can be extrapolated:

    estimated_step1_seconds = (n_bootstraps * fits_per_bootstrap) / (fits_per_core_second * n_jobs)

GPU mode benchmarks 3c only (the default component count) with two passes:
  - jit+run: JAX JIT-compiles the EM kernel then runs; this is what
    run_pipeline.py users see (new process = recompile every time)
  - steady-state: compiled kernel is already cached; this is what the HPC
    batch runner (hpc/run_local_array_gpu.sh) sees after the first dataset,
    since it keeps one Python process alive across many datasets
The Step 1 delta between the two passes is printed as JIT compilation overhead.

Run with:
    source activate excalibr

    # CPU (default)
    python tests/benchmark_run_pipeline_speed.py --n-jobs 64

    # GPU (default device)
    python tests/benchmark_run_pipeline_speed.py --device gpu

    # GPU pinned to card 1
    python tests/benchmark_run_pipeline_speed.py --device cuda:1
"""
import argparse
import json
import os
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


def _parse_device(device: str) -> str:
    """Same logic as run_pipeline._parse_device; must run before any JAX import."""
    d = device.strip().lower()
    if d == "cpu":
        return "cpu"
    if d == "gpu":
        return "gpu"
    if d.startswith("cuda:"):
        idx = d[len("cuda:"):]
        if not idx.isdigit():
            raise SystemExit(f"--device: expected cuda:N (integer index), got '{device}'")
        os.environ["CUDA_VISIBLE_DEVICES"] = idx
        return "gpu"
    raise SystemExit(f"--device: unrecognised value '{device}'. Use cpu, gpu, or cuda:N.")


def time_one_component_count(n_c: int, n_jobs: int, device: str,
                              output_dir: str) -> dict:
    config = PipelineConfig(
        dataset_csv=EXAMPLE_DATASET,
        dataset_name=DATASET_NAME,
        output_dir=output_dir,
        n_bootstraps=DEFAULT_N_BOOTSTRAPS,
        num_fits_per_bootstrap=DEFAULT_FITS_PER_BOOTSTRAP,
        components=[n_c],
        n_jobs=n_jobs,
        device=device,
        clinvar_release="2026",
        auto_select_model=False,
    )
    logger = setup_logging(config.output_dir, config.dataset_name)

    # Step 1: bootstrap fitting (the dominant cost)
    t0 = time.perf_counter()
    runner = BootstrapRunner(config)
    bootstrap_results, dataset_splits = runner.run()
    step1_seconds = time.perf_counter() - t0

    valid_fits = sum(
        1 for seed_results in bootstrap_results.values()
        for v in seed_results.values() if v is not None
    )
    total_fits = len(bootstrap_results) * len(config.components)

    # Steps 2-4
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
        variant_df.to_csv(
            Path(config.output_dir) / f"{config.dataset_name}_{component_key}_variants.csv",
            index=False,
        )
    save_results(results=results, bootstrap_results=bootstrap_results,
                 config=config, logger=logger, selected_k=None)
    rest_seconds = time.perf_counter() - t0

    n_fit_units = DEFAULT_N_BOOTSTRAPS * DEFAULT_FITS_PER_BOOTSTRAP
    result = {
        "n_c": n_c,
        "n_jobs": n_jobs,
        "device": device,
        "valid_fits": valid_fits,
        "total_fits": total_fits,
        "step1_bootstrap_fitting_seconds": step1_seconds,
        "rest_of_calibration_seconds": rest_seconds,
        "total_seconds": step1_seconds + rest_seconds,
    }
    if device == "cpu":
        result["fits_per_core_second_step1"] = n_fit_units / (step1_seconds * n_jobs)
    else:
        result["fits_per_second_step1"] = n_fit_units / step1_seconds
    return result


def print_result(r: dict, device: str) -> None:
    label = f" [{r['run_label']}]" if r.get("run_label") else ""
    print(f"  Step 1 (bootstrap fitting){label}: {r['step1_bootstrap_fitting_seconds']:.1f}s "
          f"({r['valid_fits']}/{r['total_fits']} valid fits)", end="")
    if device == "cpu":
        print(f", {r['fits_per_core_second_step1']:.3f} fits/core-second)")
    else:
        print(f", {r['fits_per_second_step1']:.2f} fits/second)")
    print(f"  Rest of calibration:               {r['rest_of_calibration_seconds']:.1f}s")
    print(f"  Total:                             {r['total_seconds']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=64,
                        help="Parallel jobs for CPU bootstrap fitting (default: 64; "
                             "ignored when --device gpu/cuda:N)")
    parser.add_argument("--device", default="cpu",
                        help="cpu (default), gpu, or cuda:N. "
                             "GPU mode benchmarks 3c only; timing includes JAX JIT "
                             "compilation (which always runs when a new process starts).")
    parser.add_argument("--output-dir", default="/tmp/benchmark_run_pipeline_speed",
                        help="Scratch output directory")
    parser.add_argument("--results-json", default=None,
                        help="Optional path to write results as JSON")
    args = parser.parse_args()

    # Must set CUDA_VISIBLE_DEVICES before any JAX import (happens inside BootstrapRunner).
    device = _parse_device(args.device)

    print(f"Dataset : {EXAMPLE_DATASET}")
    print(f"Device  : {device}" + (f" (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
                                    if "CUDA_VISIBLE_DEVICES" in os.environ else ""))
    print(f"n_bootstraps={DEFAULT_N_BOOTSTRAPS}, fits_per_bootstrap={DEFAULT_FITS_PER_BOOTSTRAP}", end="")
    if device == "cpu":
        print(f", n_jobs={args.n_jobs}")
    else:
        print()
        print("Note: first GPU pass includes JAX JIT compilation; "
              "second pass is steady-state throughput.\n")

    # GPU: benchmark 3c twice — first pass includes JAX JIT compilation,
    # second pass reuses the compiled kernel (steady-state). The difference
    # is the compilation overhead. run_pipeline.py users always experience
    # the first-pass (jit+run) time; the HPC batch runner
    # (hpc/run_local_array_gpu.sh) keeps one process alive across datasets
    # so subsequent datasets see steady-state throughput.
    # CPU: benchmark both 3c and 2c (no JIT, single pass each).
    component_counts = (3,) if device == "gpu" else (3, 2)

    all_results = []
    for n_c in component_counts:
        print(f"\n=== {n_c}c ===")
        if device == "gpu":
            r1 = time_one_component_count(n_c, args.n_jobs, device, args.output_dir)
            r1["run_label"] = "jit+run"
            all_results.append(r1)
            print_result(r1, device)

            r2 = time_one_component_count(n_c, args.n_jobs, device, args.output_dir)
            r2["run_label"] = "steady-state"
            all_results.append(r2)
            print_result(r2, device)

            compile_s = r1["step1_bootstrap_fitting_seconds"] - r2["step1_bootstrap_fitting_seconds"]
            print(f"  JIT compilation overhead (Step 1 delta): {compile_s:.1f}s")
        else:
            r = time_one_component_count(n_c, args.n_jobs, device, args.output_dir)
            all_results.append(r)
            print_result(r, device)

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.results_json}")


if __name__ == "__main__":
    main()
