#!/usr/bin/env python3
"""
Splice-filter ablation sweep: run `hpc/prepare.py pillar_project` once per
condition -- 9 SpliceAI threshold values (VEP splice-consequence filtering
left ON, matching today's default otherwise) plus one "keep_all" condition
(VEP filtering AND SpliceAI thresholding both disabled) -- into its own
subdirectory under a shared root, so each condition is a normal, independent,
full ExCALIBR-shaped output tree once its job manifest is submitted/
aggregated (see hpc/prepare.py's own printed next-step instructions).

Unlike analysis/build_robustness_dataset.py's downsample/discordance sweep,
no perturbed-CSV intermediate is needed here -- the SpliceAI/VEP filters
(Scoreset.splicing_filter, in src/assay_calibration/data_utils/dataset.py)
act directly on the raw per-variant --dataframe rows inside Scoreset
construction, so each condition is just `hpc/prepare.py pillar_project`
invoked with different --spliceai-threshold/--disable-vep-splice-filter
flags -- this script is a thin loop over that, not a data-generation step of
its own.

Usage
-----
  python analysis/build_splice_ablation_jobs.py \\
      --dataframe /data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz \\
      --output-root /data/ross/assay_calibration/explorer_jobs_pp_spliceAIthresh \\
      --config-file src/igvf_configs/dataset_configs_jul_2026.json \\
      --preset medium

Each condition then still needs its own SLURM submission + aggregation
(hpc/submit_array.sh / hpc/aggregate_results.py) -- printed per-condition by
hpc/prepare.py itself, not repeated here.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PREPARE_SCRIPT = _REPO_ROOT / "hpc" / "prepare.py"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_DATAFRAME = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
_DEFAULT_CONFIG_FILE = str(_REPO_ROOT / "src" / "igvf_configs" / "dataset_configs_jul_2026.json")
_DEFAULT_OUTPUT_ROOT = "/data/ross/assay_calibration/explorer_jobs_pp_spliceAIthresh"
_DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# (condition_label, spliceai_threshold_or_None, vep_splice_filter) -- built
# fresh per invocation from --thresholds so a caller can narrow the sweep
# (e.g. rerun just one threshold) without editing this module.
def _conditions(thresholds: List[float]) -> List[tuple]:
    conds = [(f"thresh_{t:.1f}", t, True) for t in thresholds]
    conds.append(("keep_all", None, False))
    return conds


def build_condition_command(
    condition_label: str,
    spliceai_threshold: Optional[float],
    vep_splice_filter: bool,
    dataframe: str,
    output_root: str,
    config_file: str,
    preset: str,
    extra_args: List[str],
) -> List[str]:
    """The `hpc/prepare.py pillar_project` argv for one condition."""
    cmd = [
        sys.executable, str(_PREPARE_SCRIPT), "pillar_project",
        "--dataframe", dataframe,
        "--config-file", config_file,
        "--output-dir", str(Path(output_root) / condition_label),
        "--preset", preset,
        "--spliceai-threshold", "none" if spliceai_threshold is None else str(spliceai_threshold),
    ]
    if not vep_splice_filter:
        cmd.append("--disable-vep-splice-filter")
    cmd.extend(extra_args)
    return cmd


def merge_conditions_into_shared_jobs(
    output_root: str, conditions: List[tuple], target_array_size: int = 1000,
) -> None:
    """Repack every condition's per-bootstrap jobs into one shared jobs/ folder
    under output_root, so the whole sweep submits as a single SLURM array
    instead of one array per condition. Each job is tagged with its condition
    label so hpc/run_array_task.py routes results to
    <output_root>/<condition_label>/<dataset_name>/ -- the same tree
    hpc/aggregate_results.py already expects when a condition is run on its
    own, so per-condition aggregation is unaffected.
    """
    from hpc.prepare import _save_manifest

    all_jobs: list = []
    n_bootstraps = None
    for condition_label, _, _ in conditions:
        jobs_dir = Path(output_root) / condition_label / "jobs"
        with open(jobs_dir / "job_index.json") as f:
            n_bootstraps = json.load(f)["n_bootstraps"]
        for array_file in sorted(jobs_dir.glob("array_*.pkl")):
            with open(array_file, "rb") as f:
                cjobs = pickle.load(f)
            for cjob in cjobs:
                cjob["condition"] = condition_label
            all_jobs.extend(cjobs)
            array_file.unlink()  # superseded by the merged copy below

    print(f"\nMerging {len(all_jobs)} bootstrap jobs from {len(conditions)} "
          f"condition(s) into a single job array under {output_root}/jobs/")
    _save_manifest(output_root, all_jobs, target_array_size, n_bootstraps)

    print(f"\nSubmit the whole sweep in one array job:")
    print(f"  bash hpc/submit_array.sh {output_root}")
    print(f"\nThen aggregate each condition separately once its jobs finish:")
    for condition_label, _, _ in conditions:
        print(f"  python hpc/aggregate_results.py {Path(output_root) / condition_label}")


def run_splice_ablation_sweep(
    dataframe: str,
    output_root: str,
    config_file: str,
    preset: str,
    thresholds: List[float],
    extra_args: List[str],
    dry_run: bool = False,
    merge_jobs: bool = False,
    target_array_size: int = 1000,
) -> None:
    conditions = _conditions(thresholds)
    print(f"Splice ablation sweep: {len(conditions)} condition(s) -> {output_root}")

    array_size_args = ["--target-array-size", str(target_array_size)]
    for condition_label, spliceai_threshold, vep_splice_filter in conditions:
        cmd = build_condition_command(
            condition_label, spliceai_threshold, vep_splice_filter,
            dataframe, output_root, config_file, preset,
            array_size_args + extra_args,
        )
        print(f"\n{'='*80}\n[{condition_label}] spliceai_threshold={spliceai_threshold} "
              f"vep_splice_filter={vep_splice_filter}\n{'='*80}")
        print("  " + " ".join(cmd))
        if dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))

    if merge_jobs and not dry_run:
        merge_conditions_into_shared_jobs(output_root, conditions, target_array_size=target_array_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataframe", default=_DEFAULT_DATAFRAME)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT,
                        help="Parent dir; each condition gets its own subdir under this "
                             "(default: analysis.config.SPLICE_ABLATION_ROOT's default)")
    parser.add_argument("--config-file", default=_DEFAULT_CONFIG_FILE)
    parser.add_argument("--preset", default="medium",
                        choices=["light", "medium", "large", "xl", "finest"],
                        help="Bootstrap/fit-count shorthand forwarded to hpc/prepare.py "
                             "(default: medium == 100 bootstraps x 8 fits)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=_DEFAULT_THRESHOLDS,
                        help="SpliceAI threshold values to sweep, VEP filtering left ON "
                             "for each (default: 0.1..0.9). A 'keep_all' condition "
                             "(VEP off, SpliceAI thresholding disabled) is always appended.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print each condition's hpc/prepare.py command without running it")
    parser.add_argument("--merge-jobs", action="store_true",
                        help="After building every condition, repack all conditions' "
                             "bootstrap jobs into one shared <output-root>/jobs/ array "
                             "so the whole sweep submits (and queues) as a single SLURM "
                             "array job instead of one per condition. Per-condition "
                             "results/aggregation are unaffected.")
    parser.add_argument("--target-array-size", type=int, default=1000,
                        help="Forwarded to each `hpc/prepare.py pillar_project` call, and "
                             "to the --merge-jobs repack step if given. Keep at/below "
                             "SLURM's MaxArraySize (1000-1001 by default) -- larger values "
                             "make sbatch reject the array with 'Invalid job array "
                             "specification' (default: 1000)")
    parser.add_argument("prepare_args", nargs=argparse.REMAINDER,
                        help="Extra args forwarded verbatim to every `hpc/prepare.py "
                             "pillar_project` call (e.g. -- --datasets BRCA1_Findlay_2018 "
                             "--n-jobs 4)")
    args = parser.parse_args()

    extra_args = args.prepare_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    run_splice_ablation_sweep(
        args.dataframe, args.output_root, args.config_file, args.preset,
        args.thresholds, extra_args, dry_run=args.dry_run, merge_jobs=args.merge_jobs,
        target_array_size=args.target_array_size,
    )


if __name__ == "__main__":
    main()
