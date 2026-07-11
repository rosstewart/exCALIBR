#!/usr/bin/env python
"""
Run acmgscaler against every dataset's *selected* config (same comp
resolution run_igvf_batch.py uses -- see analysis.discovery.resolve_component),
saving one visualization per dataset into the same output_dir/{dataset}/
layout ExCALIBR's own `{dataset}_{comp}_visualization.png` uses, plus one
combined CSV of per-variant acmgscaler evidence calls across all datasets.

Usage
-----
python analysis/run_acmgscaler_all.py \\
    --output-dir /data/ross/assay_calibration/explorer_jobs_pp_revisions_calib/ \\
    --dataset-configs src/igvf_configs/dataset_configs_jul_2026.json \\
    --prior 0.1

# Restrict to a subset of datasets:
python analysis/run_acmgscaler_all.py --datasets BAP1_Waters_2024 MSH2_Jia_2021_clinvar_2018 --prior 0.1
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from analysis import config as cfg
from analysis.discovery import discover_outputs, resolve_component, _parse_dataset_config_entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=None, help=f"default: {cfg.OUTPUT_DIR}")
    parser.add_argument("--dataset-configs", default=None, help=f"default: {cfg.DATASET_CONFIGS}")
    parser.add_argument("--acmgscaler-dir", default=None, help=f"default: {cfg.ACMGSCALER_DIR}")
    parser.add_argument("--prior", type=float, default=0.1)
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="Restrict to these dataset names (default: every dataset discovered)")
    parser.add_argument("--combined-output", default=None,
                         help="Optional CSV path to write per-variant acmgscaler calls for all "
                              "datasets combined (variant_id, dataset, acmgscaler_evidence, ...)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or cfg.OUTPUT_DIR)
    dataset_configs_path = args.dataset_configs or cfg.DATASET_CONFIGS
    import json
    with open(dataset_configs_path) as f:
        dataset_configs = json.load(f)

    from analysis.comparison_methods import run_acmgscaler, make_acmgscaler_figure

    tree, model_selections, calibrations = discover_outputs(output_dir)
    dataset_list = args.datasets or sorted(tree.keys())
    print(f"{len(dataset_list)} dataset(s)")

    combined_rows = []
    for i, dataset in enumerate(dataset_list, 1):
        entry = dataset_configs.get(dataset)
        if entry is None:
            print(f"[{i}/{len(dataset_list)}] SKIP {dataset}: not in --dataset-configs")
            continue
        wanted_comp = _parse_dataset_config_entry(entry)
        comp_dict = tree.get(dataset, {})
        available_comps = list(comp_dict.keys())
        comp = resolve_component(dataset, available_comps, model_selections, dataset_configs)
        csv_path = comp_dict.get(comp, {}).get("default")
        if csv_path is None:
            print(f"[{i}/{len(dataset_list)}] SKIP {dataset}: no variants.csv for selected comp {comp}")
            continue

        print(f"[{i}/{len(dataset_list)}] {dataset} ({comp})")
        df_variants = pd.read_csv(csv_path)
        try:
            fig_path = make_acmgscaler_figure(
                dataset, df_variants, output_dir, prior=args.prior,
            )
        except Exception as e:
            print(f"  FAILED {dataset}: {e}")
            continue
        if fig_path is None:
            continue

        if args.combined_output:
            df_acmg = run_acmgscaler(df_variants, prior=args.prior)
            df_acmg["dataset"] = dataset
            combined_rows.append(df_acmg[[
                "variant_id", "dataset", "score", "sample",
                "acmgscaler_lr", "acmgscaler_lr_lower", "acmgscaler_lr_upper",
                "acmgscaler_evidence", "acmgscaler_points",
            ]])

    if args.combined_output and combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(args.combined_output, index=False)
        print(f"\nWrote {len(combined)} row(s) across {len(combined_rows)} dataset(s) to {args.combined_output}")

    print("\nDone.")


if __name__ == "__main__":
    main()
