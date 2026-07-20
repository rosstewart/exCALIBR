#!/usr/bin/env python
"""
Run acmgscaler against every dataset, loading directly from the master
integrated dataframe and building a fresh Scoreset per dataset -- same
pattern as run_igvf_batch.py / slurm/simple_gmm_baseline.py. No dependency
on ExCALIBR pipeline output existing at all: this never reads
*_variants.csv, and --output-dir is purely a fresh destination for
acmgscaler's own figures/CSV, not shared with the pipeline's own output.

Parallelized across datasets (joblib, like run_igvf_batch.py's outer
--n-jobs) -- each dataset's work is an independent Rscript subprocess call,
so there's no shared state to worry about. Defaults to -1 (all cores). The
full master dataframe is pre-partitioned by dataset in the parent process
before dispatch, so each worker only gets pickled its own (small)
per-dataset slice rather than the whole multi-GB dataframe repeatedly (same
trick simple_gmm_baseline.py uses).

Usage
-----
python analysis/run_acmgscaler_all.py \\
    --dataset /data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_merged_89datasets.tsv.gz \\
    --output-dir /data/ross/assay_calibration/acmgscaler_out \\
    --prior 0.1

# Restrict to a subset of datasets:
python analysis/run_acmgscaler_all.py --datasets BAP1_Waters_2024 MSH2_Jia_2021 --prior 0.1
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from joblib import Parallel, delayed

from analysis import config as cfg

# Same set run_igvf_batch.py / slurm/simple_gmm_baseline.py use to decide
# which datasets need clinvar_release="2018" (those genes were only ever
# curated with 2018-era ClinVar significance calls).
GENES_2018 = {"BRCA1", "MSH2", "PTEN", "TP53"}


def _gene_of(dataset_name: str) -> str:
    return dataset_name.split("_")[0]


def _process_dataset(
    dataset: str, df_ds: pd.DataFrame, dataset_tsv: str, output_dir: str,
    prior: float, acmgscaler_dir, no_clinvar_2018: bool, want_combined: bool,
):
    """One dataset's worth of work -- top-level (not nested) so joblib/loky
    can pickle it for cross-process dispatch.
    """
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    from analysis.comparison_methods import (
        run_acmgscaler, make_acmgscaler_figure, build_variants_df_from_scoreset,
        acmgscaler_variants_path,
    )

    clinvar_release = "2018" if not no_clinvar_2018 and _gene_of(dataset) in GENES_2018 else "2026"
    effective_name = dataset if clinvar_release != "2018" else f"{dataset}_clinvar_2018"
    df_ds = df_ds.copy()
    df_ds["Dataset"] = effective_name

    pcfg = PipelineConfig(
        dataset_csv=str(dataset_tsv), dataset_name=effective_name,
        output_dir="/tmp", clinvar_release=clinvar_release,
    )
    try:
        scoreset = load_dataset_from_df(df_ds, pcfg)
    except Exception as e:
        print(f"SKIP {effective_name}: Scoreset error -- {e}")
        return None

    print(f"{effective_name}: running acmgscaler...")
    df_variants = build_variants_df_from_scoreset(scoreset)
    try:
        df_acmg, thresholds = run_acmgscaler(
            df_variants, prior=prior, acmgscaler_dir=acmgscaler_dir, return_thresholds=True,
        )
    except Exception as e:
        print(f"FAILED {effective_name}: {e}")
        return None

    # Always save the per-dataset CSV -- this is what makes acmgscaler results
    # loadable by analysis.confusion/analysis.comparison_methods without ever
    # calling Rscript again (see comparison_methods.load_acmgscaler_variants),
    # the same way ExCALIBR's own *_variants.csv is loaded directly rather
    # than recomputing calibration on every notebook run.
    csv_path = acmgscaler_variants_path(effective_name, output_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_acmg.to_csv(csv_path, index=False)

    try:
        fig_path = make_acmgscaler_figure(
            effective_name, df_variants, output_dir,
            prior=prior, acmgscaler_dir=acmgscaler_dir,
            precomputed=(df_acmg, thresholds),
        )
    except Exception as e:
        print(f"FAILED {effective_name} (figure): {e}")
        fig_path = None
    if fig_path is None and not want_combined:
        return None

    if want_combined:
        df_acmg = df_acmg.copy()
        df_acmg["dataset"] = effective_name
        return df_acmg[[
            "variant_id", "dataset", "score", "sample",
            "acmgscaler_lr", "acmgscaler_lr_lower", "acmgscaler_lr_upper",
            "acmgscaler_evidence", "acmgscaler_points",
        ]]
    return effective_name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help=f"Master integrated dataframe; default: {cfg.DATASET_TSV}")
    parser.add_argument("--output-dir", required=True,
                         help="Destination for acmgscaler figures (fresh directory, not the pipeline's own output_dir)")
    parser.add_argument("--acmgscaler-dir", default=None, help=f"default: {cfg.ACMGSCALER_DIR}")
    parser.add_argument("--prior", type=float, default=0.1)
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="Restrict to these dataset names (default: every dataset in --dataset)")
    parser.add_argument("--no-clinvar-2018", action="store_true",
                         help="Disable automatic ClinVar 2018 mode for BRCA1/MSH2/PTEN/TP53 (matches "
                              "run_igvf_batch.py's flag of the same name)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                         help="Datasets to process in parallel (default: -1 = all cores). Each dataset "
                              "spawns its own Rscript subprocess, so this parallelizes cleanly.")
    parser.add_argument("--combined-output", default=None,
                         help="Optional CSV path to write per-variant acmgscaler calls for all "
                              "datasets combined (variant_id, dataset, acmgscaler_evidence, ...)")
    args = parser.parse_args()

    dataset_tsv = args.dataset or cfg.DATASET_TSV
    sep = "\t" if str(dataset_tsv).endswith((".tsv", ".tsv.gz")) else ","
    print(f"Loading {dataset_tsv} ...")
    df_full = pd.read_csv(dataset_tsv, sep=sep, low_memory=False)

    dataset_list = args.datasets or sorted(df_full["Dataset"].unique())
    print(f"{len(dataset_list)} dataset(s), n_jobs={args.n_jobs}")

    # Pre-partition in the parent so each worker is only pickled its own
    # small slice, not the whole multi-GB dataframe.
    partitions = {ds: df_full[df_full["Dataset"] == ds] for ds in dataset_list}
    missing = [ds for ds in dataset_list if partitions[ds].empty]
    for ds in missing:
        print(f"SKIP {ds}: not found in {dataset_tsv}")
    dataset_list = [ds for ds in dataset_list if ds not in missing]

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_process_dataset)(
            ds, partitions[ds], dataset_tsv, args.output_dir, args.prior,
            args.acmgscaler_dir, args.no_clinvar_2018, bool(args.combined_output),
        )
        for ds in dataset_list
    )

    if args.combined_output:
        combined_rows = [r for r in results if isinstance(r, pd.DataFrame)]
        if combined_rows:
            combined = pd.concat(combined_rows, ignore_index=True)
            combined.to_csv(args.combined_output, index=False)
            print(f"\nWrote {len(combined)} row(s) across {len(combined_rows)} dataset(s) to {args.combined_output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
