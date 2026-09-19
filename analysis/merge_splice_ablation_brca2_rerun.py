#!/usr/bin/env python3
"""
One-off: merge a per-condition single-dataset rerun's bootstrap fits into the
corresponding condition of a main splice-ablation sweep, looped across every
condition subdirectory.

Each dataset's value is {bootstrap_index: {model_key: fit}} -- a flat
dataset-level dict.update() (the original BRCA2 behavior, where the rerun's
model_key set matched the main file's) would be wrong whenever the rerun
introduces a *new* model_key (e.g. a rerun that adds a "4c" GMM fit for a
dataset that previously only had "3c"): it would replace the whole per-index
dict and silently delete the existing model_key's fits. So the merge here
happens one level deeper -- for each dataset key present in the rerun file,
each bootstrap index's dict is updated in place (adding/overwriting only the
rerun's model_key(s), leaving any other model_key at that index untouched).
Every other dataset key in the main file is untouched either way.

Usage:
    python analysis/merge_splice_ablation_brca2_rerun.py \\
        --main-root /data/ross/assay_calibration/explorer_jobs_pp_spliceAIthresh \\
        --brca2-root /data/ross/assay_calibration/explorer_jobs_pp_spliceAIthresh_BRCA2
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


def merge_condition(main_path: Path, brca2_path: Path) -> None:
    backup_path = main_path.with_suffix(main_path.suffix + ".bak")
    shutil.copy(main_path, backup_path)

    main_data = load(main_path)
    brca2_data = load(brca2_path)
    before_keys = set(main_data.keys())

    print(f"  {main_path.parent.name}:")
    print(f"    source keys: {sorted(brca2_data.keys())}")

    new_dataset_keys = set()
    added, overwritten = 0, 0
    for dataset_key, src_entry in brca2_data.items():
        if dataset_key not in main_data:
            main_data[dataset_key] = src_entry
            new_dataset_keys.add(dataset_key)
            continue
        tgt_entry = main_data[dataset_key]
        src_idx = set(src_entry.keys())
        tgt_idx = set(tgt_entry.keys())
        if src_idx != tgt_idx:
            raise SystemExit(
                f"Bootstrap-index key mismatch for {dataset_key} in {main_path.parent.name}: "
                f"source has {len(src_idx)}, target has {len(tgt_idx)}"
            )
        for idx, model_dict in src_entry.items():
            before_model_keys = set(tgt_entry[idx].keys())
            tgt_entry[idx].update(model_dict)
            added += len(set(model_dict.keys()) - before_model_keys)
            overwritten += len(set(model_dict.keys()) & before_model_keys)

    print(f"    new dataset keys added: {sorted(new_dataset_keys)}")
    print(f"    existing dataset keys nested-merged: {sorted(set(brca2_data.keys()) - new_dataset_keys)}")
    print(f"    model keys newly added: {added}, model keys overwritten: {overwritten}")
    print(f"    total dataset keys before -> after: {len(before_keys)} -> {len(main_data)}")

    save(main_path, main_data)
    print(f"    saved (backup: {backup_path})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--main-root", required=True,
                        help="Main sweep root, e.g. .../explorer_jobs_pp_spliceAIthresh "
                             "(expects <main-root>/<condition>/bootstrap_results.json.gz)")
    parser.add_argument("--brca2-root", required=True,
                        help="BRCA2-rerun sweep root, same per-condition layout as --main-root")
    args = parser.parse_args()

    main_root = Path(args.main_root)
    brca2_root = Path(args.brca2_root)

    condition_dirs = sorted(
        d for d in main_root.iterdir()
        if d.is_dir() and (d / "bootstrap_results.json.gz").exists()
    )
    if not condition_dirs:
        raise SystemExit(f"No <condition>/bootstrap_results.json.gz found under {main_root}")

    print(f"Merging {brca2_root} into {main_root} across {len(condition_dirs)} condition(s)")
    for condition_dir in condition_dirs:
        condition = condition_dir.name
        main_path = condition_dir / "bootstrap_results.json.gz"
        brca2_path = brca2_root / condition / "bootstrap_results.json.gz"
        if not brca2_path.exists():
            print(f"  SKIP {condition}: no matching BRCA2-rerun file at {brca2_path}")
            continue
        merge_condition(main_path, brca2_path)


if __name__ == "__main__":
    main()
