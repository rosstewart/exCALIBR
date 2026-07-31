#!/usr/bin/env python3
"""
Collect selected-model calibration outputs (JSON + PNG) from a batch output
directory into a flat destination with json/ and png/ subfolders.

For datasets with only one component (2c or 3c), that component is used.
For datasets with both components, the one with model_selected=1 is used.
"""

import argparse
import json
import shutil
import socket
import subprocess
from pathlib import Path


def pick_component(dataset_dir: Path) -> Path | None:
    """Return the calibration JSON for the selected (or only) component."""
    jsons = sorted(dataset_dir.glob("*_calibration.json"))
    if not jsons:
        return None
    if len(jsons) == 1:
        return jsons[0]

    # Multiple components — pick the one with model_selected=1
    for j in jsons:
        try:
            data = json.loads(j.read_text())
        except Exception:
            continue
        if data.get("model_selected") == 1:
            return j

    # Fallback: none flagged as selected (shouldn't happen, but warn and skip)
    print(f"  WARNING: {dataset_dir.name} has multiple components but none marked model_selected=1 — skipping")
    return None


def pick_component_from_config(dataset_dir: Path, entry: dict) -> Path | None:
    """Return the calibration JSON matching a dataset_configs.json entry's
    (n_c, benign_method), falling back to the bare-n_c filename (older runs
    that predate the benign_method-suffixed naming).

    Unlike pick_component's model_selected flag — which is only ever set on
    the bare-{n_c} files from the old avg-only final run and knows nothing
    about benign_method — this always reflects the currently configured combo.
    """
    n_c = str(entry.get("n_c", "3c"))
    benign_method = entry.get("benign_method")
    comp = f"{n_c}_{benign_method}" if benign_method else n_c

    p = dataset_dir / f"{dataset_dir.name}_{comp}_calibration.json"
    if p.exists():
        return p
    p_bare = dataset_dir / f"{dataset_dir.name}_{n_c}_calibration.json"
    if p_bare.exists():
        return p_bare

    # Filename can lag behind content -- e.g. the pipeline silently overrides
    # benign_method (avg -> benign) for datasets with no synonymous sample,
    # but the *filename* still reflects whatever benign_method was originally
    # requested when the file was saved. Fall back to checking every
    # calibration JSON's own content against the requested (n_c, benign_method).
    for candidate in sorted(dataset_dir.glob(f"{dataset_dir.name}_*_calibration.json")):
        try:
            data = json.loads(candidate.read_text())
        except Exception:
            continue
        content_n_c = str(data.get("n_c", ""))
        for suf in ("_avg", "_benign"):
            if content_n_c.endswith(suf):
                content_n_c = content_n_c[: -len(suf)]
                break
        if content_n_c == n_c and (benign_method is None or data.get("benign_method") == benign_method):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Batch calibration output directory")
    parser.add_argument("output_dir", type=Path, help="Destination directory")
    parser.add_argument(
        "--dataset-configs", type=Path, default=None,
        help="If given, select each dataset's (n_c, benign_method) calibration from "
             "this dataset_configs.json instead of the model_selected=1 flag (which "
             "only exists on old bare-{n_c} files and ignores benign_method). Datasets "
             "not present in the config file are skipped.",
    )
    args = parser.parse_args()

    dataset_configs = None
    if args.dataset_configs is not None:
        dataset_configs = json.loads(args.dataset_configs.read_text())

    json_dir = args.output_dir / "json"
    png_dir = args.output_dir / "png"
    json_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(p for p in args.input_dir.iterdir() if p.is_dir() and p.name != "logs")

    copied = 0
    skipped = 0
    for dataset_dir in dataset_dirs:
        if dataset_configs is not None:
            entry = dataset_configs.get(dataset_dir.name)
            if entry is None:
                print(f"  SKIP {dataset_dir.name}: not present in --dataset-configs")
                skipped += 1
                continue
            calib_json = pick_component_from_config(dataset_dir, entry)
        else:
            calib_json = pick_component(dataset_dir)
        if calib_json is None:
            print(f"  SKIP {dataset_dir.name}: no calibration JSON found")
            skipped += 1
            continue

        # Derive the PNG path from the JSON path (same stem prefix, different suffix)
        stem = calib_json.stem.replace("_calibration", "_visualization")
        png = dataset_dir / f"{stem}.png"

        # Build clean destination names: strip _2c/_3c tag and _calibration/_visualization suffix
        def clean_name(name: str, ext: str) -> str:
            for tag in ("_2c", "_3c"):
                name = name.replace(tag + "_", "_")
            for tag in ("_avg", "_benign", "_synonymous"):
                name = name.replace(tag + "_", "_")
            for suffix in ("_calibration", "_visualization"):
                if name.endswith(suffix + ext):
                    name = name[: -len(suffix + ext)] + ext
            return name

        dest_json = json_dir / clean_name(calib_json.name, ".json")
        shutil.copy2(calib_json, dest_json)

        if png.exists():
            dest_png = png_dir / clean_name(png.name, ".png")
            shutil.copy2(png, dest_png)
            print(f"  {dataset_dir.name}: {dest_json.name} + {dest_png.name}")
        else:
            print(f"  {dataset_dir.name}: {dest_json.name} (no PNG found)")

        copied += 1

    print(f"\nDone: {copied} datasets copied, {skipped} skipped.")

    # Tar the output directory, then remove it
    tar_path = args.output_dir.with_suffix(".tar.gz")
    subprocess.run(
        ["tar", "-czf", str(tar_path), "-C", str(args.output_dir.parent), args.output_dir.name],
        check=True,
    )
    shutil.rmtree(args.output_dir)
    print(f"  Archive → {tar_path}")

    hostname = socket.getfqdn()
    print(f"\nscp rcstewart@{hostname}:{tar_path} ~/Downloads/")


if __name__ == "__main__":
    main()
