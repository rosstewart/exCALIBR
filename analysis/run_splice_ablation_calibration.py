#!/usr/bin/env python3
"""
Splice-filter ablation sweep, stage 2: turn each condition's already-
aggregated bootstrap fits (`analysis/build_splice_ablation_jobs.py` + the
`hpc/` SLURM workflow's `aggregate_results.py` step) into that condition's
full calibration output tree, by invoking `run_igvf_batch.py` once per
condition, sequentially.

This has to be a real per-condition `run_igvf_batch.py` invocation (not a
single call scoring every condition at once) because `run_igvf_batch.py`
rebuilds its own Scoreset from `--dataset` -- and that Scoreset's variant
population depends on `--spliceai-threshold`/`--disable-vep-splice-filter`
(`Scoreset.splicing_filter`, `src/assay_calibration/data_utils/dataset.py`),
the exact same two knobs each condition's bootstrap-fitting job used. Reusing
one `run_igvf_batch.py` call across every condition would silently score all
of them against the SAME (default) variant population, regardless of which
condition's point_ranges were being applied -- inconsistent with whatever
population those point_ranges were actually calibrated against.

Each condition's own `bootstrap_results.json.gz` (per `hpc/aggregate_results.py`)
and calibration output both live under `<output-root>/<condition_label>/` --
the same directory `analysis.discovery.discover_outputs` expects for a normal
pipeline run, matching analyze_pipeline_output.py section 3a8's assumption
that each condition subdirectory is "a full, independently-fit ExCALIBR
output tree".

Usage
-----
  python analysis/run_splice_ablation_calibration.py \\
      --dataframe /data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz \\
      --output-root /data/ross/assay_calibration/explorer_jobs_pp_spliceAIthresh \\
      --config-file src/igvf_configs/dataset_configs_aug_2026.json \\
      --clinvar-release 2025

  # Rerun just one condition, restricted to specific datasets:
  python analysis/run_splice_ablation_calibration.py \\
      --output-root /data/ross/assay_calibration/explorer_jobs_pp_spliceAIthresh \\
      --conditions thresh_0.2 \\
      -- --datasets BRCA2_Huang_2025_SGE BRCA2_IGVF CTCF_IGVF PALB2_IGVF RAD51D_IGVF SFPQ_IGVF XRCC2_IGVF
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis import config
from analysis.build_splice_ablation_jobs import (
    _conditions,
    _DEFAULT_CONFIG_FILE,
    _DEFAULT_DATAFRAME,
    _DEFAULT_OUTPUT_ROOT,
    _DEFAULT_THRESHOLDS,
)

_RUN_IGVF_BATCH = _REPO_ROOT / "run_igvf_batch.py"


def build_condition_command(
    condition_label: str,
    spliceai_threshold,
    vep_splice_filter: bool,
    dataframe: str,
    output_root: str,
    config_file: str,
    clinvar_release: str,
    extra_args: List[str],
) -> List[str]:
    """The `run_igvf_batch.py` argv for one condition -- precomputed-fits and
    output-dir both point at this condition's own `<output_root>/<condition_label>/`
    (bootstrap_results.json.gz was already aggregated there, calibration
    output is written alongside it), matching build_condition_command in
    analysis/build_splice_ablation_jobs.py for the bootstrap-fitting side.
    """
    condition_dir = Path(output_root) / condition_label
    cmd = [
        sys.executable, str(_RUN_IGVF_BATCH),
        "--dataset", dataframe,
        "--precomputed-fits", str(condition_dir / "bootstrap_results.json.gz"),
        "--output-dir", str(condition_dir),
        "--dataset-configs", config_file,
        "--clinvar-release", clinvar_release,
        "--spliceai-threshold", "none" if spliceai_threshold is None else str(spliceai_threshold),
    ]
    if not vep_splice_filter:
        cmd.append("--disable-vep-splice-filter")
    cmd.extend(extra_args)
    return cmd


def run_splice_ablation_calibration(
    dataframe: str,
    output_root: str,
    config_file: str,
    clinvar_release: str,
    thresholds: List[float],
    condition_labels: List[str],
    extra_args: List[str],
    dry_run: bool = False,
) -> None:
    conditions = _conditions(thresholds)
    if condition_labels:
        conditions = [c for c in conditions if c[0] in set(condition_labels)]
    print(f"Splice ablation calibration: {len(conditions)} condition(s) -> {output_root}")

    for condition_label, spliceai_threshold, vep_splice_filter in conditions:
        condition_dir = Path(output_root) / condition_label
        bootstrap_fits_path = condition_dir / "bootstrap_results.json.gz"
        if config.warn_if_missing(str(bootstrap_fits_path), f"{condition_label} bootstrap fits"):
            continue

        cmd = build_condition_command(
            condition_label, spliceai_threshold, vep_splice_filter,
            dataframe, output_root, config_file, clinvar_release, extra_args,
        )
        print(f"\n{'='*80}\n[{condition_label}] spliceai_threshold={spliceai_threshold} "
              f"vep_splice_filter={vep_splice_filter}\n{'='*80}")
        print("  " + " ".join(cmd))
        if dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataframe", default=_DEFAULT_DATAFRAME)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT,
                        help="Same parent dir passed to build_splice_ablation_jobs.py -- "
                             "each condition's bootstrap_results.json.gz is read from, and "
                             "calibration output written to, <output-root>/<condition_label>/.")
    parser.add_argument("--config-file", default=_DEFAULT_CONFIG_FILE)
    parser.add_argument("--clinvar-release", default="2025", choices=["2026", "2025", "2018"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=_DEFAULT_THRESHOLDS,
                        help="SpliceAI threshold values to process (default: 0.1..0.9). "
                             "A 'keep_all' condition is always included unless narrowed "
                             "away by --conditions.")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Only run these condition labels (e.g. thresh_0.2 keep_all) "
                             "instead of every threshold in --thresholds + keep_all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print each condition's run_igvf_batch.py command without running it")
    parser.add_argument("run_igvf_batch_args", nargs=argparse.REMAINDER,
                        help="Extra args forwarded verbatim to every run_igvf_batch.py call "
                             "(e.g. -- --datasets BRCA2_IGVF CTCF_IGVF --oob)")
    args = parser.parse_args()

    extra_args = args.run_igvf_batch_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    run_splice_ablation_calibration(
        args.dataframe, args.output_root, args.config_file, args.clinvar_release,
        args.thresholds, args.conditions, extra_args, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
