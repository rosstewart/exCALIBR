#!/usr/bin/env python3
"""
Stage 1 of the ExCALIBR robustness pipeline: build the perturbed-input CSV
that `hpc/prepare.py basicscoreset` consumes to generate the bootstrap-fit
job manifest analyzed by `analysis/robustness.py`.

This is a reproducible, tracked replacement for the ad hoc cells at the end
of `test/downsample_discordance_test.ipynb` (an untracked/local notebook)
that previously did this by hand: load the integrated dataframe, build a
base `Scoreset` per requested dataset, perturb each with
`Scoreset.from_scoreset(..., downsample_n_variants=N)` (control-count
downsampling) and `Scoreset.from_scoreset(..., discordance_pct=pct)` (label
discordance/noise) across several seeds, flatten every perturbed `Scoreset`
to `(Dataset, score, sample_assignments)` rows, and write the concatenated
long-format CSV.

Also emits one unperturbed "_control" condition per base dataset -- the SAME
base Scoreset, no downsampling/discordance applied, fit through the identical
xl (1000 bootstraps x 8 fits) job manifest as every perturbed condition. This
exists specifically so `analysis/robustness.py` can compare each perturbed
condition against a like-for-like, equal-fit-budget baseline instead of the
main pipeline's own (typically higher-fit-count, e.g. "finest"/100 fits)
reference output -- a stronger model baseline would conflate "perturbation
effect" with "fit-budget effect" in the comparison.

The `{base}_ds{N}_s{seed}` / `{base}_disc{pct:.2f}_s{seed}` / `{base}_control`
condition-name convention here MUST stay consistent with
`analysis.robustness`'s `_COND_RE`/`_CONTROL_RE`/`parse_condition_dirname`
(which parses it back out of the downstream pipeline's output directory
names) and with `hpc/prepare.py`'s `--name-strip` regex (which strips it back
off before per-dataset config lookup) -- this module reuses
`analysis.robustness.GENES_2018` as the single source of truth for which
genes need `clinvar_release="2018"`, rather than redefining it.

Usage
-----
  python analysis/build_robustness_dataset.py \\
      --output-csv /data/ross/assay_calibration/robustness/robustness_all.csv

Next step (printed at the end unless --no-print-prepare-cmd):
  python hpc/prepare.py basicscoreset --dataframe <output-csv> \\
      --output-dir <robustness output dir> --config-file <dataset config> \\
      --name-strip '_ds[0-9]+_s[0-9]+$|_disc[0-9.]+_s[0-9]+$|_control$' \\
      --n-bootstraps 1000 --fits-3c 8 --fits-2c 8
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.data_utils.dataset import Scoreset
from analysis.robustness import GENES_2018

_DEFAULT_DATAFRAME = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
_DEFAULT_DATASETS = [
    "BRCA1_Findlay_2018",
    "BRCA2_Sahu_2025_SGE",
    "TP53_Giacomelli_2018_combined_score",
    "MSH2_Jia_2021",
]
_DEFAULT_DOWNSAMPLE_NS = [1, 2, 4, 8, 16, 32, 64]
_DEFAULT_DISCORDANCE_PCTS = [0.01, 0.10]
_DEFAULT_SEEDS = list(range(10))
_DEFAULT_OUTPUT_CSV = "/data/ross/assay_calibration/robustness/robustness_all.csv"
_DEFAULT_CONFIG_FILE = str(_REPO_ROOT / "src" / "igvf_configs" / "dataset_configs_jul_2026.json")


def build_base_scoresets(df: pd.DataFrame, datasets: List[str]) -> Dict[str, Scoreset]:
    """One Scoreset per requested dataset, clinvar_release="2018" for
    GENES_2018 genes (matching the main pipeline's own convention) else the
    Scoreset default."""
    base_dfs = {dataset: df[df["Dataset"] == dataset] for dataset in datasets}
    base_scoresets = {}
    for dataset, base_df in base_dfs.items():
        if len(base_df) == 0:
            print(f"  {dataset}: SKIP -- no rows in --dataframe")
            continue
        gene = dataset.split("_")[0]
        kw = {"clinvar_release": "2018"} if gene in GENES_2018 else {}
        base_scoresets[dataset] = Scoreset(base_df, **kw)
    return base_scoresets


def build_perturbed_scoresets(
    base_scoresets: Dict[str, Scoreset],
    downsample_ns: List[int],
    discordance_pcts: List[float],
    seeds: List[int],
):
    """Reproduces the notebook's downsample/discordance perturbation loop:
    for each base dataset, build `downsample_n_variants=N` and
    `discordance_pct=pct` perturbed Scoresets across every seed. Skips a
    downsample level entirely if N exceeds both the pathogenic and benign
    control counts (nothing to downsample), matching the notebook exactly."""
    downsamples: Dict[str, Dict[int, List[Scoreset]]] = defaultdict(dict)
    discordance: Dict[str, Dict[float, List[Scoreset]]] = defaultdict(dict)

    for dataset, base_scoreset in base_scoresets.items():
        n_pathogenic = base_scoreset.sample_assignments[:, 0].sum()
        n_benign = base_scoreset.sample_assignments[:, 1].sum()

        for downsample_n in downsample_ns:
            if downsample_n > n_pathogenic and downsample_n > n_benign:
                continue
            downsamples[dataset][downsample_n] = [
                Scoreset.from_scoreset(base_scoreset, downsample_n_variants=downsample_n, perturbation_seed=seed)
                for seed in seeds
            ]

        for discordance_pct in discordance_pcts:
            discordance[dataset][discordance_pct] = [
                Scoreset.from_scoreset(base_scoreset, discordance_pct=discordance_pct, perturbation_seed=seed)
                for seed in seeds
            ]

    return downsamples, discordance


def scoreset_to_rows(scoreset: Scoreset, dataset_name: str) -> pd.DataFrame:
    """Convert a Scoreset to a list of (Dataset, score, sample_assignments) rows."""
    sa = scoreset._sample_assignments   # (N, NSamples) full one-hot
    rows_idx, cols_idx = np.where(sa)
    row_cols = defaultdict(list)
    for r, c in zip(rows_idx, cols_idx):
        row_cols[r].append(str(c))
    N = sa.shape[0]
    return pd.DataFrame({
        "Dataset":            [dataset_name] * N,
        "score":              scoreset.scores,
        "sample_assignments": [",".join(row_cols[i]) if i in row_cols else "" for i in range(N)],
    })


def build_robustness_csv(
    dataframe: str,
    datasets: List[str],
    downsample_ns: List[int],
    discordance_pcts: List[float],
    seeds: List[int],
    output_csv: str,
) -> pd.DataFrame:
    sep = "\t" if dataframe.endswith((".tsv.gz", ".tsv")) else ","
    print(f"Loading {dataframe}...")
    df = pd.read_csv(dataframe, sep=sep, low_memory=False)

    print(f"Building {len(datasets)} base Scoreset(s)...")
    base_scoresets = build_base_scoresets(df, datasets)

    print("Generating perturbed Scoresets "
          f"(downsample N={downsample_ns}, discordance={discordance_pcts}, seeds={seeds})...")
    downsamples, discordance = build_perturbed_scoresets(base_scoresets, downsample_ns, discordance_pcts, seeds)

    parts = []
    for dataset, base_scoreset in base_scoresets.items():
        parts.append(scoreset_to_rows(base_scoreset, f"{dataset}_control"))
    for dataset, ds_by_n in downsamples.items():
        for n, scoresets_list in ds_by_n.items():
            for seed, ss in zip(seeds, scoresets_list):
                name = f"{dataset}_ds{n}_s{seed}"
                parts.append(scoreset_to_rows(ss, name))
    for dataset, ds_by_pct in discordance.items():
        for pct, scoresets_list in ds_by_pct.items():
            for seed, ss in zip(seeds, scoresets_list):
                name = f"{dataset}_disc{pct:.2f}_s{seed}"
                parts.append(scoreset_to_rows(ss, name))

    df_all = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_all.to_csv(output_csv, index=False)
    print(f"Wrote {len(df_all):,} rows x {df_all['Dataset'].nunique()} datasets -> {output_csv}")
    return df_all


def _print_prepare_cmd(output_csv: str, output_dir: str, config_file: str) -> None:
    cmd = f"""\
source activate excalibr && \\
python {_REPO_ROOT}/hpc/prepare.py basicscoreset \\
--dataframe {output_csv} \\
--output-dir {output_dir} \\
--config-file {config_file} \\
--name-strip '_ds[0-9]+_s[0-9]+$|_disc[0-9.]+_s[0-9]+$|_control$' \\
--n-bootstraps 1000 \\
--fits-3c 8 \\
--fits-2c 8 \\
--target-array-size 1000"""
    print("\nNext step -- generate the bootstrap-fit job manifest:\n")
    print(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataframe", default=_DEFAULT_DATAFRAME)
    parser.add_argument("--datasets", nargs="+", default=_DEFAULT_DATASETS)
    parser.add_argument("--downsample-ns", nargs="+", type=int, default=_DEFAULT_DOWNSAMPLE_NS)
    parser.add_argument("--discordance-pcts", nargs="+", type=float, default=_DEFAULT_DISCORDANCE_PCTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=_DEFAULT_SEEDS)
    parser.add_argument("--output-csv", default=_DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-dir", default=None,
                        help="hpc/prepare.py --output-dir to print in the next-step command "
                             "(default: sibling 'explorer_jobs' dir next to --output-csv)")
    parser.add_argument("--config-file", default=_DEFAULT_CONFIG_FILE,
                        help="hpc/prepare.py --config-file to print in the next-step command")
    parser.add_argument("--no-print-prepare-cmd", action="store_true",
                        help="Skip printing the follow-up hpc/prepare.py command")
    args = parser.parse_args()

    build_robustness_csv(
        args.dataframe, args.datasets, args.downsample_ns, args.discordance_pcts,
        args.seeds, args.output_csv,
    )

    if not args.no_print_prepare_cmd:
        output_dir = args.output_dir or str(Path(args.output_csv).parent / "explorer_jobs_pp_robustness")
        _print_prepare_cmd(args.output_csv, output_dir, args.config_file)


if __name__ == "__main__":
    main()
