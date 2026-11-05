#!/usr/bin/env python3
"""Supplementary Data 5-6: the Human Domainome 1 calibrations (Beltran et al.
2025; Reviewer 3), in the same formats as Supplementary Data 1-2.

- SD5 `SD5_domainome_dataset_summaries.csv`: supplementary_data.build_sd1
  (SD1 columns) on the Domainome calibrations, with
  analysis/domainome/dataset_metadata_domainome.csv as curated metadata and
  training / held-out Yang distances computed here exactly as for the main
  run (cached_bootstrap_yang_distances on the regular fits, same n_grid).
- SD6 `SD6_domainome_model_fit_visualizations.pdf`:
  extended_data_appendix.build_appendix_pdf (SD2 layout), every scored variant
  ("All variants"; protein-level library) on the "Abundance score" axis.

Both list datasets alphabetically. Inputs are the standard (non
--all-configs) run of the selected configurations
(dataset_configs_domainome.json) with --oob, ClinVar 2025:
  run_igvf_batch.py --dataset-configs dataset_configs_domainome.json \\
      --clinvar-release 2025 --oob --splits-file domainome_clinvar2025_dataset_splits.pkl ...
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis import config as cfg  # noqa: E402

BASE = Path("/data/ross/assay_calibration/domainome")
METADATA_CSV = Path(__file__).resolve().parent / "dataset_metadata_domainome.csv"
SD5_NAME = "SD5_domainome_dataset_summaries.csv"
SD6_NAME = "SD6_domainome_model_fit_visualizations.pdf"
_YANG_KEYS = (("plp", "pathogenic"), ("blb", "benign"), ("gnomad", "gnomad"), ("synonymous", "synonymous"))


def _pct_row(vals, prefix, quantiles):
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(vals).any()
    row = {prefix: float(np.nanmedian(vals)) if ok else np.nan}
    for q in quantiles:
        row[f"{prefix}_p{q:g}"] = float(np.nanpercentile(vals, q)) if ok else np.nan
    return row


def yang_tables(datasets, output_dir, dataset_tsv, precomputed_fits, dataset_configs_path, splits_file,
                cache_dir, n_jobs=-1, n_grid=None):
    """(training, held-out) Yang tables in the layouts build_sd1 reads: the
    run_build_excalibr_datasets_table --with-yang columns and
    yang_distance.heldout_yang_table's columns."""
    from analysis.legacy_fits import load_scoreset_and_fits, resolve_component_for
    from analysis.yang_distance import _resolve_sample_indices, cached_bootstrap_yang_distances

    n_grid = n_grid or cfg.YANG_N_GRID
    with open(splits_file, "rb") as f:
        splits = pickle.load(f)
    train_rows, val_rows = [], []
    for ds in datasets:
        n_c, benign_method = resolve_component_for(ds, output_dir=output_dir,
                                                   dataset_configs_path=dataset_configs_path)
        scoreset, _, fits, _, n_c, _, _ = load_scoreset_and_fits(
            ds, output_dir=output_dir, dataset_tsv=dataset_tsv, precomputed_fits=precomputed_fits,
            dataset_configs_path=dataset_configs_path, n_c=n_c, benign_method=benign_method,
            clinvar_release="2025")
        kw = dict(cache_dir=cache_dir, output_dir=output_dir, dataset_to_splits={ds: splits[ds]},
                  n_jobs=n_jobs, n_grid=n_grid)
        print(f"  Yang distances: {ds} ({n_c})")
        tr = cached_bootstrap_yang_distances(ds, n_c, fits, scoreset, split="train", **kw)
        va = cached_bootstrap_yang_distances(ds, n_c, fits, scoreset, split="val", **kw)
        trow, vrow = {"dataset": ds}, {"dataset": ds, "n_c": n_c}
        idx = _resolve_sample_indices(scoreset)
        for short, key in _YANG_KEYS:
            trow.update(_pct_row(tr.get(key, []), f"yang_dist_{short}", (2.5, 5, 25, 75, 95, 97.5)))
            vrow.update(_pct_row(va.get(key, []), f"yang_val_{short}", (25, 75)))
            col = idx.get(key)
            vrow[f"n_val_{short}"] = (np.nan if col is None else float(np.median(
                [np.asarray(v["val_sample_assignments"])[:, col].sum() for v in splits[ds].values()])))
        train_rows.append(trow)
        val_rows.append(vrow)
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows)


def build_sd5(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path, splits_file, cache_dir,
              n_jobs=-1):
    """SD5 table (SD1 columns) and the problems build_sd1 reported."""
    from analysis.all_variant_evidence import build_all_variant_groups_table
    from analysis.author_labels import attach_author_labels
    from analysis.discovery import discover_outputs, load_all_variants
    from analysis.supplementary_data import build_sd1

    tree, model_selections, calibrations = discover_outputs(Path(output_dir))
    with open(dataset_configs_path) as f:
        dataset_configs = json.load(f)
    datasets = sorted(set(tree) & set(dataset_configs))
    df = load_all_variants(tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
                           methods_filter=None, datasets_filter=datasets, calibrations=calibrations,
                           min_controls=0, dataset_tsv=str(dataset_tsv))
    df = attach_author_labels(df, str(dataset_tsv))
    primary_method = sorted(df["method"].unique())[0]
    df_primary = df[df["method"] == primary_method]
    df_all_groups = build_all_variant_groups_table(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        calibrations=calibrations, dataset_tsv=str(dataset_tsv), datasets=datasets,
        primary_method=primary_method)
    yang_df, yang_heldout_df = yang_tables(datasets, output_dir, dataset_tsv, precomputed_fits,
                                           dataset_configs_path, splits_file, cache_dir, n_jobs=n_jobs)
    sd5, problems = build_sd1(datasets, tree, model_selections, dataset_configs, df_primary,
                              df_all_groups=df_all_groups, yang_df=yang_df, yang_heldout_df=yang_heldout_df,
                              curated_csv=str(METADATA_CSV), output_dir=output_dir, dataset_tsv=dataset_tsv)
    return sd5.sort_values("dataset").reset_index(drop=True), problems


def build_sd6(out_path, output_dir, dataset_tsv, precomputed_fits, dataset_configs_path, n_jobs=7):
    from analysis.extended_data_appendix import build_appendix_pdf

    with open(dataset_configs_path) as f:
        datasets = sorted(json.load(f))
    # Protein-level library (all 19 substitutions per position): plot every
    # scored variant, not only SNV-accessible ones; scores are aPCA abundance.
    build_appendix_pdf(dataset_list=datasets, output_path=out_path, plot_thresholds=True, n_jobs=n_jobs,
                       all_variants=True, score_label="Abundance score", output_dir=output_dir,
                       dataset_tsv=dataset_tsv, precomputed_fits=precomputed_fits,
                       dataset_configs_path=dataset_configs_path)
    return datasets


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(BASE / "exc_domainome_clinvar2025_selected"))
    p.add_argument("--dataset-tsv", default=str(BASE / "excalibr_domainome.integrated.tsv.gz"))
    p.add_argument("--precomputed-fits",
                   default=str(BASE / "jobs_domainome_clinvar2025_2c3c_1000b" / "bootstrap_results.json.gz"))
    p.add_argument("--dataset-configs", default=str(BASE / "dataset_configs_domainome.json"))
    p.add_argument("--splits-file", default=str(BASE / "domainome_clinvar2025_dataset_splits.pkl"))
    p.add_argument("--yang-cache-dir", default=str(BASE / "yang_distance_cache"))
    p.add_argument("--supp-dir", default=str(cfg.SUPP_DATA_DIR))
    p.add_argument("--n-jobs", type=int, default=16)
    p.add_argument("--skip-sd5", action="store_true")
    p.add_argument("--skip-sd6", action="store_true")
    args = p.parse_args()

    supp = Path(args.supp_dir)
    supp.mkdir(parents=True, exist_ok=True)
    if not args.skip_sd5:
        sd5, problems = build_sd5(args.output_dir, args.dataset_tsv, args.precomputed_fits, args.dataset_configs,
                                  args.splits_file, args.yang_cache_dir, n_jobs=args.n_jobs)
        sd5.to_csv(supp / SD5_NAME, index=False)
        print(f"SD5: {len(sd5)} rows -> {supp / SD5_NAME}")
        for prob in problems:
            print(f"  PROBLEM: {prob}")
    if not args.skip_sd6:
        build_sd6(supp / SD6_NAME, args.output_dir, args.dataset_tsv, args.precomputed_fits,
                  args.dataset_configs, n_jobs=min(args.n_jobs, 7))
        print(f"SD6 -> {supp / SD6_NAME}")


if __name__ == "__main__":
    main()
