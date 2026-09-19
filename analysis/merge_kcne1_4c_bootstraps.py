#!/usr/bin/env python3
"""
One-off: merge KCNE1_Muhammad_2024_trafficking's newly-fit "4c" (4-component
GMM) bootstrap results into a main bootstrap-fits file that currently only
has "3c" for that dataset.

Unlike a flat dataset-level dict.update() (see
analysis/merge_splice_ablation_brca2_rerun.py), the merge here has to happen
one level deeper: each dataset's value is {bootstrap_index: {model_key: fit}},
and the source file only has a "4c" model_key at every bootstrap index while
the target only has "3c". A dataset-level update() would replace the whole
per-index dict and silently delete every "3c" fit. Instead, this merges each
bootstrap index's dict in place, so the result has both "3c" and "4c".

Usage:
    python analysis/merge_kcne1_4c_bootstraps.py \\
        --source /data/ross/assay_calibration/TODO/KCNE1_4c_bootstrap_fits.json.gz \\
        --target /data/ross/assay_calibration/pp_final_clinvar2025_bootstraps/final_clinvar2025_bootstrap_results.json.gz \\
        --dataset-key KCNE1_Muhammad_2024_trafficking
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path


def load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save(path: Path, data: dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def merge_nested(source_path: Path, target_path: Path, dataset_key: str) -> None:
    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
    shutil.copy(target_path, backup_path)

    source_data = load(source_path)
    target_data = load(target_path)

    src_entry = source_data[dataset_key]
    tgt_entry = target_data[dataset_key]

    src_idx = set(src_entry.keys())
    tgt_idx = set(tgt_entry.keys())
    if src_idx != tgt_idx:
        raise SystemExit(
            f"Bootstrap-index key mismatch for {dataset_key}: "
            f"source has {len(src_idx)}, target has {len(tgt_idx)}, "
            f"symmetric difference {sorted(src_idx ^ tgt_idx)[:10]}..."
        )

    added, already_present = 0, 0
    for idx, model_dict in src_entry.items():
        before_keys = set(tgt_entry[idx].keys())
        tgt_entry[idx].update(model_dict)
        newly_added = set(model_dict.keys()) - before_keys
        added += len(newly_added)
        already_present += len(set(model_dict.keys()) & before_keys)

    print(f"{dataset_key}: {len(src_idx)} bootstrap indices merged")
    print(f"  model keys newly added: {added}")
    print(f"  model keys already present (overwritten): {already_present}")
    print(f"  post-merge model keys at index '0': {sorted(target_data[dataset_key]['0'].keys())}")

    save(target_path, target_data)
    print(f"  saved {target_path} (backup: {backup_path})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Single-dataset rerun bootstrap_fits.json.gz")
    parser.add_argument("--target", required=True, help="Main bootstrap_results.json.gz to merge into")
    parser.add_argument("--dataset-key", required=True, help="Dataset name key shared by both files")
    args = parser.parse_args()

    merge_nested(Path(args.source), Path(args.target), args.dataset_key)


if __name__ == "__main__":
    main()
