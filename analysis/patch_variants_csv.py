#!/usr/bin/env python
"""
Patch the `sample`, `auth_label`, and `is_vus` columns of existing
*_variants.csv files in place, without recomputing any evidence points.

Why this is needed: every *_variants.csv already on disk (written by either
run_igvf_batch.py or test/regenerate_variant_tables.py before the fixes to
src/assay_calibration/pipeline/variant_evidence.py) can be stale in three ways:

  1. `sample` was collapsed to the first matching category instead of the
     full pipe-separated multi-label list (e.g. "Synonymous|population"),
     for any variant genuinely belonging to more than one sample category.
  2. `auth_label` didn't exist at all in files written before auth_label
     export was added to _build_standard_table/_build_continuous_table --
     these are exactly the datasets that make analysis/author_labels.py's
     attach_author_labels() fall back to its slow per-dataset Scoreset
     rebuild in the notebook.
  3. `is_vus` didn't exist at all in files written before its export was
     added -- analysis/evidence.py::build_clinvar_multilabel falls back to a
     negation-derived approximation (matches none of B/LB, P/LP, gnomAD,
     Synonymous) that silently undercounts VUS for any dataset where every
     variant also happens to carry a Synonymous/gnomAD tag (verified to
     report 0/1376 VUS for MSH2_Jia_2021_clinvar_2018 instead of the true
     421) -- and is always wrong for clinvar_2018-mode datasets specifically,
     since Variant.is_vus is hardcoded to ClinVar 2026 regardless of what
     release that dataset's own P/LP/B/LB classification uses.

All three fixes only change what NEW variants.csv writes contain going
forward -- they don't retroactively repair files already written. Since
`score`/`standard_points`/`oob_points` are unaffected by any of them, there's
no need to rerun calibration or bootstrap fitting -- just reload each
dataset's Scoreset (cheap: no fitting, just the same preprocessing
run_igvf_batch.py already does before fitting) and rewrite `sample`,
`auth_label`, and `is_vus`, keyed by variant_id, leaving every other column
untouched.

Usage
-----
python analysis/patch_variants_csv.py \\
    --output-dir /data/ross/assay_calibration/explorer_jobs_pp_revisions_calib/ \\
    --dataset /data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_merged_89datasets.tsv.gz
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.pipeline.config import PipelineConfig
from src.assay_calibration.pipeline.utils import load_dataset_from_df
from src.assay_calibration.pipeline.variant_evidence import _get_variant_ids, _get_variant_is_vus


def build_patch_maps(dataset: str, source_dfs, clinvar_release: str):
    """Return (sample_map, auth_label_map, is_vus_map): variant_id -> new
    value, for `dataset`.

    Only touches Scoreset construction (ClinVar/population/consequence
    preprocessing) -- no fitting, no calibration.json needed -- so this is
    cheap relative to a full pipeline rerun. auth_label_map/is_vus_map are
    empty if the Scoreset has no auth_labels/is_vus attribute (e.g.
    BasicScoreset). is_vus is always ClinVar-2026-based regardless of
    `clinvar_release` -- see _get_variant_is_vus.
    """
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

    config = PipelineConfig(
        dataset_csv="",  # unused; we pass dataset_df directly
        dataset_name=dataset,
        output_dir="",
        components=[2],       # unused for sample/auth_label/is_vus; placeholder
        benign_method="avg",  # unused for sample/auth_label/is_vus; placeholder
        clinvar_release=clinvar_release,
        compute_oob=False,
    )
    scoreset = load_dataset_from_df(dataset_df, config)

    ids = _get_variant_ids(scoreset)
    sample_map = {}
    for idx, vid in enumerate(ids):
        matched = [
            scoreset.sample_names[s_idx]
            for s_idx in range(len(scoreset.sample_names))
            if scoreset._sample_assignments[idx, s_idx]
        ]
        sample_map[vid] = "|".join(matched) if matched else "Unknown"

    auth_labels = getattr(scoreset, "auth_labels", None)
    auth_label_map = {}
    if auth_labels is not None:
        auth_label_map = {vid: auth_labels[idx] for idx, vid in enumerate(ids)}

    is_vus = _get_variant_is_vus(scoreset)
    is_vus_map = {}
    if is_vus is not None:
        is_vus_map = {vid: is_vus[idx] for idx, vid in enumerate(ids)}

    return sample_map, auth_label_map, is_vus_map


def patch_csv(csv_path: Path, sample_map: dict, auth_label_map: dict, is_vus_map: dict) -> dict:
    df = pd.read_csv(csv_path)
    changed = {"sample": 0, "auth_label": 0, "is_vus": 0}
    if "variant_id" not in df.columns:
        print(f"  SKIP {csv_path.name}: missing variant_id column")
        return changed

    if "sample" in df.columns:
        new_sample = df["variant_id"].map(sample_map)
        missing = int(new_sample.isna().sum())
        if missing:
            print(f"  WARN {csv_path.name}: {missing} variant_id(s) not found in reloaded scoreset, keeping old sample value for those rows")
            new_sample = new_sample.fillna(df["sample"])
        changed["sample"] = int((new_sample != df["sample"]).sum())
        df["sample"] = new_sample

    if auth_label_map:
        new_auth = df["variant_id"].map(auth_label_map)
        # Treat empty string (Scoreset's default "no label" sentinel) as missing.
        new_auth = new_auth.replace("", pd.NA)
        if "auth_label" not in df.columns:
            old_auth = pd.Series(pd.NA, index=df.index)
        else:
            old_auth = df["auth_label"]
        changed["auth_label"] = int(
            (new_auth.notna() & (new_auth.astype(str) != old_auth.astype(str))).sum()
        )
        df["auth_label"] = new_auth.where(new_auth.notna(), old_auth)

    if is_vus_map:
        new_vus = df["variant_id"].map(is_vus_map)
        old_vus = df["is_vus"] if "is_vus" in df.columns else pd.Series(pd.NA, index=df.index)
        changed["is_vus"] = int(
            (new_vus.notna() & (new_vus.astype(str) != old_vus.astype(str))).sum()
        )
        df["is_vus"] = new_vus.where(new_vus.notna(), old_vus)

    df.to_csv(csv_path, index=False)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", action="append", required=True,
                         help="Integrated dataset TSV/CSV; repeat to search multiple source files")
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="Restrict to these dataset names (default: every dataset subdir with a *_variants.csv)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Loading {len(args.dataset)} source dataset file(s)...")
    source_dfs = []
    for path in args.dataset:
        sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
        print(f"  {path}: {len(df):,} rows, {df['Dataset'].nunique()} datasets")
        source_dfs.append(df)

    dataset_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
    if args.datasets:
        wanted = set(args.datasets)
        dataset_dirs = [d for d in dataset_dirs if d.name in wanted]

    total_changed = {"sample": 0, "auth_label": 0, "is_vus": 0}
    for ds_dir in dataset_dirs:
        dataset = ds_dir.name
        variant_csvs = sorted(ds_dir.glob(f"{dataset}_*_variants.csv"))
        if not variant_csvs:
            continue
        clinvar_release = "2018" if dataset.endswith("_clinvar_2018") else "2025"
        try:
            sample_map, auth_label_map, is_vus_map = build_patch_maps(dataset, source_dfs, clinvar_release)
        except Exception as e:
            print(f"FAILED {dataset}: could not build patch maps: {e}")
            continue
        for csv_path in variant_csvs:
            try:
                changed = patch_csv(csv_path, sample_map, auth_label_map, is_vus_map)
                total_changed["sample"] += changed["sample"]
                total_changed["auth_label"] += changed["auth_label"]
                total_changed["is_vus"] += changed["is_vus"]
                print(f"OK {csv_path.name}: {changed['sample']} sample value(s), "
                      f"{changed['auth_label']} auth_label value(s), "
                      f"{changed['is_vus']} is_vus value(s) changed")
            except Exception as e:
                import traceback
                print(f"FAILED {csv_path.name}: {e}")
                traceback.print_exc()

    print(f"\nDone. {total_changed['sample']} sample value(s), "
          f"{total_changed['is_vus']} is_vus value(s), "
          f"{total_changed['auth_label']} auth_label value(s) changed in total.")


if __name__ == "__main__":
    main()
