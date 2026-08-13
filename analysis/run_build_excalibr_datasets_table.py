#!/usr/bin/env python
"""
CLI runner for analysis.build_dataset_summary.build_excalibr_datasets_table —
reconstructs excalibr_datasets.csv from run_igvf_batch.py output.

Runs in two stages:
  1. Calibration ranges + sample counts + metadata (fast, seconds/dataset).
  2. Yang distance goodness-of-fit (slow, ~1-3 min/dataset at full 1000
     bootstraps/full grid resolution -- see analysis/yang_distance.py) --
     skipped by default; pass --with-yang to include it.

Examples
--------
# Fast table only (no Yang distances):
python analysis/run_build_excalibr_datasets_table.py --output excalibr_datasets.csv

# Full table including Yang distances (all 1000 bootstraps/dataset; n_grid=2000
# by default -- <0.1% change in reported medians vs the original diagnostic's
# n_grid=10000, ~4-5x faster per bootstrap -- see analysis/yang_distance.py's
# compute_yang_distance_p2 docstring):
python analysis/run_build_excalibr_datasets_table.py --with-yang --output excalibr_datasets.csv

# Same, but with the exact original (slower) CDF grid resolution:
python analysis/run_build_excalibr_datasets_table.py --with-yang --yang-n-grid 10000 --output excalibr_datasets.csv

# Restrict to a subset of datasets (e.g. to test timing before a full run):
python analysis/run_build_excalibr_datasets_table.py --with-yang --datasets MSH2_Jia_2021 BAP1_Waters_2024 --output test.csv

# Resumable: write each dataset's Yang distance to --yang-checkpoint as it
# finishes, and skip any dataset already present there on a rerun (e.g. after
# killing a long run, or after a code fix -- delete just the affected rows
# from the checkpoint CSV first if only some datasets need recomputing):
python analysis/run_build_excalibr_datasets_table.py --with-yang \\
    --yang-checkpoint excalibr_datasets_yang_checkpoint.csv --output excalibr_datasets.csv
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis import config as cfg
from analysis.discovery import discover_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=None, help=f"default: {cfg.OUTPUT_DIR}")
    parser.add_argument("--dataset-tsv", default=None, help=f"default: {cfg.DATASET_TSV}")
    parser.add_argument("--dataset-configs", default=None, help=f"default: {cfg.DATASET_CONFIGS}")
    parser.add_argument("--precomputed-fits", default=None, help=f"default: {cfg.PRECOMPUTED_FITS}")
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="Restrict to these dataset names (default: every dataset discovered under --output-dir)")
    parser.add_argument("--with-yang", action="store_true",
                         help="Also compute Yang distance goodness-of-fit (slow -- see module docstring)")
    parser.add_argument("--yang-n-grid", type=int, default=2000,
                         help="CDF-grid resolution for Yang distance (default 2000: ~4-5x faster than the original "
                              "diagnostic's 10000 for <0.1%% change in reported medians; pass 10000 for the exact "
                              "original resolution)")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for Yang distance (-1 = all cores)")
    parser.add_argument("--yang-checkpoint", default=None,
                         help="CSV path to write each dataset's Yang distance to incrementally, and to "
                              "resume from (skip datasets already present) on a rerun -- see module docstring")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    output_dir = args.output_dir or cfg.OUTPUT_DIR

    if args.datasets:
        dataset_list = args.datasets
    else:
        tree, _, _ = discover_outputs(Path(output_dir))
        dataset_list = sorted(tree.keys())
    print(f"{len(dataset_list)} dataset(s): {dataset_list[:5]}{'...' if len(dataset_list) > 5 else ''}")

    from analysis.build_dataset_summary import (
        build_excalibr_datasets_table, compute_yang_distances_all,
    )

    yang_df = None
    if args.with_yang:
        print(f"\nComputing Yang distances (n_grid={args.yang_n_grid}, all bootstraps/dataset) ...")
        yang_df = compute_yang_distances_all(
            dataset_list, output_dir=output_dir, dataset_tsv=args.dataset_tsv,
            precomputed_fits=args.precomputed_fits, dataset_configs_path=args.dataset_configs,
            n_jobs=args.n_jobs, n_grid=args.yang_n_grid, checkpoint_path=args.yang_checkpoint,
        )
        print(f"Yang distances computed for {len(yang_df)}/{len(dataset_list)} dataset(s)")

    print("\nBuilding excalibr_datasets table (calibration ranges + sample counts + metadata) ...")
    table = build_excalibr_datasets_table(
        output_dir=output_dir, dataset_tsv=args.dataset_tsv, dataset_list=dataset_list,
        yang_distances_df=yang_df,
    )

    table.to_csv(args.output, index=False)
    print(f"\nWrote {len(table)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
