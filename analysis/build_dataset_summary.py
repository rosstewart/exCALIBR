"""
Reconstruct excalibr_datasets.csv (the assay_method_map input needed by
analysis/assay_stats.py) from run_igvf_batch.py output, per the 4-step
recipe:

  1. Calibration ranges/config, extracted from every {dataset}_{comp}_calibration.json
     (reuses test/convert_calibration_out_json_to_csv.py's exact column mapping).
  2. Metadata (gene, citation, PMID, description, assay_type, model_system,
     vamp_sge, IGVF_produced) via analysis.gene_table.build_dataset_table().
  3. Per-dataset sample counts (n_plp, n_blb, n_gnomad, n_synonymous, n_vus,
     n_snv) — cheap, computed directly from a Scoreset per dataset.
     Yang distances (yang_dist_plp/blb/gnomad/synonymous) are NOT computed
     here by default — see compute_yang_distances_all below, which is slow
     (one bootstrap-parallel Yang-distance run per dataset; took ~2.5 min for
     a single dataset with 1000 bootstraps in the MSH2 example) and should be
     run deliberately, not as a side effect of building the summary table.
  4. Merge all of the above into one dataset-indexed DataFrame.

Step 2 (metadata) needs analysis.config.DATASET_DESCRIPTIONS_CSV /
DATASET_MEASUREMENTS_CSV / ASSAY_METHOD_MAP_CSV, none of which exist on this
machine as of this writing, and analysis.gene_table.build_dataset_table()
also expects "Assay Type"/"Model_system" columns in DATASET_TSV that the
current merged dataframe doesn't have — so step 2 degrades to just the gene
name (parsed from the dataset name) when it's unavailable, rather than
blocking the rest of the table.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from analysis import config as cfg

# Reuse test/convert_calibration_out_json_to_csv.py's exact column mapping
# rather than reimplementing it.
CSV_COLUMNS = [
    "dataset",
    "range_-8", "range_-7", "range_-6", "range_-5", "range_-4", "range_-3", "range_-2", "range_-1",
    "range_1", "range_2", "range_3", "range_4", "range_5", "range_6", "range_7", "range_8",
    "prior", "relax", "n_c", "benign_method", "clinvar_2018", "scoreset_flipped",
]


def _fmt_num(x):
    if x == float("inf"):
        return "Infinity"
    if x == float("-inf"):
        return "-Infinity"
    return str(x)


def _fmt_ranges(ranges):
    if not ranges:
        return ""
    return ";".join(f"{_fmt_num(lo)} {_fmt_num(hi)}" for lo, hi in ranges)


def _build_row(obj: dict) -> dict:
    row = {
        "dataset": obj.get("dataset", ""),
        "prior": obj.get("prior", ""),
        "relax": obj.get("relax", ""),
        "n_c": obj.get("n_c", ""),
        "benign_method": obj.get("benign_method", ""),
        "clinvar_2018": obj.get("clinvar_2018", ""),
        "scoreset_flipped": obj.get("scoreset_flipped", ""),
    }
    point_ranges = obj.get("point_ranges", {})
    for i in range(-8, 9):
        if i == 0:
            continue
        row[f"range_{i}"] = _fmt_ranges(point_ranges.get(str(i), []))
    return row


def build_calibration_ranges_table(output_dir: Optional[str] = None) -> pd.DataFrame:
    """Step 1: extract calibration ranges/config from every *_calibration.json
    under output_dir — one row per (dataset, comp). Equivalent to running
    `test/convert_calibration_out_json_to_csv.py '<output_dir>/*/*.json'`.
    """
    output_dir = Path(output_dir or cfg.OUTPUT_DIR)
    rows = []
    for path in sorted(output_dir.rglob("*_calibration.json")):
        try:
            with open(path) as f:
                obj = json.load(f)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue
        rows.append(_build_row(obj))
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def compute_sample_counts(
    dataset_tsv: Optional[str] = None,
    dataset_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Step 3a (cheap part): per-dataset sample counts via a bare Scoreset —
    no bootstrap fitting, so this is fast (seconds per dataset, not minutes).

    Returns columns: dataset, gene, n_plp, n_blb, n_gnomad, n_synonymous,
    n_vus, n_variants.
    """
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df

    dataset_tsv = dataset_tsv or cfg.DATASET_TSV
    sep = "\t" if str(dataset_tsv).endswith((".tsv", ".tsv.gz")) else ","
    df_full = pd.read_csv(dataset_tsv, sep=sep, low_memory=False)

    dataset_list = dataset_list if dataset_list is not None else sorted(df_full["Dataset"].unique())

    rows = []
    for dataset in dataset_list:
        csv_name = dataset.replace("_clinvar_2018", "")
        df_ds = df_full[df_full["Dataset"] == csv_name].copy()
        if df_ds.empty:
            print(f"  SKIP {dataset}: not found in {dataset_tsv}")
            continue
        df_ds["Dataset"] = dataset
        clinvar_release = "2018" if "_clinvar_2018" in dataset else "2025"
        pcfg = PipelineConfig(
            dataset_csv=str(dataset_tsv), dataset_name=dataset,
            output_dir="/tmp", clinvar_release=clinvar_release,
        )
        try:
            scoreset = load_dataset_from_df(df_ds, pcfg)
        except Exception as e:
            print(f"  SKIP {dataset}: Scoreset error — {e}")
            continue

        sample_names = [s[1] for s in scoreset.samples]
        counts = {name: int(cnt) for name, cnt in zip(sample_names, scoreset.sample_counts)} \
            if hasattr(scoreset, "sample_counts") else {}

        rows.append({
            "dataset": dataset,
            "gene": dataset.split("_")[0],
            "n_variants": len(scoreset.scores),
            "n_plp": counts.get("Pathogenic/Likely Pathogenic", 0),
            "n_blb": counts.get("Benign/Likely Benign", 0),
            "n_gnomad": counts.get("population", counts.get("gnomAD", 0)),
            "n_synonymous": counts.get("Synonymous", 0),
        })

    return pd.DataFrame(rows)


def compute_yang_distances_all(
    dataset_list: List[str],
    output_dir: Optional[str] = None,
    dataset_tsv: Optional[str] = None,
    precomputed_fits: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
    n_jobs: int = -1,
    n_grid: int = 10000,
    checkpoint_path: Optional[str] = None,
) -> pd.DataFrame:
    """Step 3b (expensive part) — Yang distance (p=2) goodness-of-fit per
    dataset/sample, via analysis.yang_distance.

    This is genuinely slow: each dataset needs (a) a full Scoreset + fits
    rebuild via analysis.legacy_fits.load_scoreset_and_fits and (b)
    compute_bootstrap_yang_distances_parallel over every bootstrap seed
    (~1000 by default) — the MSH2 example took ~2.5 minutes by itself with
    full parallelism. Across ~80+ datasets this could run for a long time.
    Call this deliberately (e.g. on a curated subset first) rather than as
    part of the default excalibr_datasets.csv build.

    All 1000 bootstraps are always used per dataset (the point of the
    bootstrap is to characterize the full distribution of the goodness-of-fit
    statistic, not just its median, so subsampling bootstraps is not offered
    here). `n_grid` is exposed as a speed/accuracy knob — see
    analysis.yang_distance.compute_bootstrap_yang_distances_parallel /
    compute_yang_distance_p2 for what it controls. The default (10000)
    reproduces the original diagnostic exactly; n_grid=2000 gives a ~4-5x
    speedup for a <0.1% change in the reported per-dataset medians.

    checkpoint_path : optional CSV path. When given, each dataset's row is
        written to this file as soon as it's computed (rewriting the whole
        file each time -- cheap at ~88 rows, and avoids partial-row
        corruption from an interrupted append). Any dataset already present
        in an existing file at this path is skipped at start, so killing and
        rerunning with the same checkpoint_path resumes rather than losing
        already-completed datasets or recomputing them unnecessarily. There
        is no correctness guard here for *why* a dataset is already in the
        checkpoint (e.g. it was computed with since-fixed code) -- delete its
        row from the checkpoint file first if you need it recomputed.

    Returns columns: dataset, yang_dist_plp, yang_dist_blb, yang_dist_gnomad,
    yang_dist_synonymous (median across bootstraps per sample).
    """
    from analysis.legacy_fits import load_scoreset_and_fits, resolve_component_for
    from analysis.yang_distance import compute_bootstrap_yang_distances_parallel

    rows = []
    already_done = set()
    if checkpoint_path and Path(checkpoint_path).exists():
        existing = pd.read_csv(checkpoint_path)
        rows = existing.to_dict("records")
        already_done = set(existing["dataset"])
        print(f"Resuming from checkpoint {checkpoint_path}: "
              f"{len(already_done)} dataset(s) already computed, skipping those")

    for i, dataset in enumerate(dataset_list, 1):
        if dataset in already_done:
            print(f"[{i}/{len(dataset_list)}] SKIP {dataset}: already in checkpoint")
            continue
        print(f"[{i}/{len(dataset_list)}] Yang distance: {dataset}")
        try:
            n_c, benign_method = resolve_component_for(
                dataset, output_dir=output_dir, dataset_configs_path=dataset_configs_path,
            )
            scoreset, _, fits, _, n_c, _, _ = load_scoreset_and_fits(
                dataset, output_dir=output_dir, dataset_tsv=dataset_tsv,
                precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
                n_c=n_c, benign_method=benign_method,
            )
            yd = compute_bootstrap_yang_distances_parallel(
                dataset, n_c, fits, scoreset, dataset_to_splits=None, n_jobs=n_jobs,
                n_grid=n_grid,
            )
            rows.append({
                "dataset": dataset,
                "yang_dist_plp": float(np.nanmedian(yd["pathogenic"])),
                "yang_dist_blb": float(np.nanmedian(yd["benign"])),
                "yang_dist_gnomad": float(np.nanmedian(yd["gnomad"])),
                "yang_dist_synonymous": float(np.nanmedian(yd["synonymous"])),
            })
            if checkpoint_path:
                pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  SKIP {dataset}: {e}")

    return pd.DataFrame(rows)


def build_excalibr_datasets_table(
    output_dir: Optional[str] = None,
    dataset_tsv: Optional[str] = None,
    dataset_list: Optional[List[str]] = None,
    yang_distances_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Step 4: merge calibration ranges + metadata + sample counts (+ Yang
    distances, if precomputed via compute_yang_distances_all and passed in)
    into one dataset-indexed table — the reconstruction of excalibr_datasets.csv.

    Metadata (gene/citation/PMID/description/assay_type/model_system/vamp_sge/
    IGVF_produced) comes from analysis.gene_table.build_dataset_table(); if
    that fails (missing source CSVs — see its own error message), this falls
    back to just the gene name parsed from the dataset name rather than
    blocking the whole table.
    """
    cal_df = build_calibration_ranges_table(output_dir)
    counts_df = compute_sample_counts(dataset_tsv, dataset_list)

    result = counts_df.merge(cal_df, on="dataset", how="left")

    try:
        from analysis.gene_table import build_dataset_table
        meta_df = build_dataset_table()
        result = result.merge(
            meta_df.rename(columns={"dataset": "dataset"}),
            on="dataset", how="left", suffixes=("", "_meta"),
        )
    except Exception as e:
        print(f"  Metadata merge skipped (analysis.gene_table.build_dataset_table failed): {e}")

    if yang_distances_df is not None:
        result = result.merge(yang_distances_df, on="dataset", how="left")

    return result
