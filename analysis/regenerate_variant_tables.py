#!/usr/bin/env python
"""
Regenerate missing *_variants.csv files from already-computed calibration JSONs.

Some pipeline output directories contain calibration.json / lr_values.json.gz for
a dataset's selected (n_c, benign_method) combo but never got a variants.csv,
because run_igvf_batch.py's sweep path (_run_one_combo) intentionally skips
compute_variant_table to save time, and the final "selected config" run
(run_single_dataset) was never executed for these datasets.

This script reuses the pipeline's own compute_variant_table() — not a
reimplementation — against a freshly-loaded Scoreset and the on-disk
calibration.json, so it reproduces standard_points exactly. It cannot recover
oob_points, since that requires the original per-seed bootstrap fits
(bootstrap_fits.json.gz), which don't exist for these datasets.

Example
-------
python analysis/regenerate_variant_tables.py \\
    --output-dir /data/ross/assay_calibration/explorer_jobs_pp_revisions_calib/ \\
    --dataset-configs src/igvf_configs/dataset_configs_jul_2026.json \\
    --dataset /data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_20260620_combined79datasets.tsv.gz \\
    --dataset /data/ross/assay_calibration/dataframe/last_batch.expanded.tsv.gz \\
    --datasets BAP1_Waters_2024 CHEK2_Gebbia_2024
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.pipeline.config import PipelineConfig
from src.assay_calibration.pipeline.utils import load_dataset_from_df
from src.assay_calibration.pipeline.variant_evidence import compute_variant_table


def _comp_key(entry) -> str:
    n_c = str(entry["n_c"])
    benign_method = entry.get("benign_method")
    return f"{n_c}_{benign_method}" if benign_method else n_c


def _find_calibration(output_dir: Path, dataset: str, comp_key: str) -> Path:
    p = output_dir / dataset / f"{dataset}_{comp_key}_calibration.json"
    if p.exists():
        return p
    # fall back to bare n_c (older naming)
    n_c = comp_key.split("_", 1)[0]
    p_bare = output_dir / dataset / f"{dataset}_{n_c}_calibration.json"
    if p_bare.exists():
        return p_bare
    raise FileNotFoundError(f"No calibration.json found for {dataset} / {comp_key}")


def regenerate_one(dataset: str, entry: dict, output_dir: Path, source_dfs: list) -> Path:
    comp_key = _comp_key(entry)
    cal_path = _find_calibration(output_dir, dataset, comp_key)
    with open(cal_path) as f:
        calibration = json.load(f)

    n_c_int = int(calibration["n_c"].split("_", 1)[0].replace("c", ""))
    clinvar_release = "2018" if calibration.get("clinvar_2018") else "2026"

    csv_name = dataset.replace("_clinvar_2018", "")
    dataset_df = None
    for df in source_dfs:
        sub = df[df["Dataset"] == csv_name]
        if len(sub) > 0:
            dataset_df = sub.copy()
            break
    if dataset_df is None:
        raise ValueError(f"{dataset} ('{csv_name}') not found in any --dataset source file")
    dataset_df["Dataset"] = dataset

    ds_out_dir = output_dir / dataset
    config = PipelineConfig(
        dataset_csv="",  # unused; we pass dataset_df directly
        dataset_name=dataset,
        output_dir=str(ds_out_dir),
        components=[n_c_int],
        benign_method=calibration.get("benign_method", entry.get("benign_method", "avg")),
        clinvar_release=clinvar_release,
        liberal_monotonicity=bool(calibration.get("liberal_monotonicity", True)),
        compute_oob=False,
    )

    scoreset = load_dataset_from_df(dataset_df, config)
    variant_df = compute_variant_table(scoreset, calibration, config, dataset_splits=None)

    out_path = ds_out_dir / f"{dataset}_{comp_key}_variants.csv"
    variant_df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-configs", required=True)
    parser.add_argument("--dataset", action="append", required=True,
                         help="Integrated dataset TSV/CSV; repeat to search multiple source files")
    parser.add_argument("--datasets", nargs="+", required=True,
                         help="Dataset names to regenerate variants.csv for")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    with open(args.dataset_configs) as f:
        dataset_configs = json.load(f)

    print(f"Loading {len(args.dataset)} source dataset file(s)...")
    source_dfs = []
    for path in args.dataset:
        sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
        print(f"  {path}: {len(df):,} rows, {df['Dataset'].nunique()} datasets")
        source_dfs.append(df)

    for dataset in args.datasets:
        entry = dataset_configs.get(dataset)
        if entry is None:
            print(f"SKIP {dataset}: not in --dataset-configs")
            continue
        try:
            out_path = regenerate_one(dataset, entry, output_dir, source_dfs)
            print(f"OK {dataset}: wrote {out_path}")
        except Exception as e:
            import traceback
            print(f"FAILED {dataset}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
