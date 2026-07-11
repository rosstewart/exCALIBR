#!/usr/bin/env python
"""
Delete pipeline output files for (dataset, comp) combos that are not the
selected config in --dataset-configs (e.g. src/igvf_configs/dataset_configs_jul_2026.json).

run_igvf_batch.py's sweep mode writes one *_calibration.json/*_lr_values.json.gz/
*_visualization.png (and sometimes *_variants.csv) per (n_c, benign_method) combo
it tried, e.g. BAP1_Waters_2024_2c_avg_*, BAP1_Waters_2024_2c_benign_*,
BAP1_Waters_2024_3c_avg_*, BAP1_Waters_2024_3c_benign_* -- but only one of those
combos (the one in --dataset-configs) is actually used downstream. This script
finds and (optionally) removes the rest.

Also catches "old naming" orphans -- bare-comp files like
XRCC2_IGVF_2c_visualization.png (n_c only, no benign_method suffix) --
leftovers from before benign_method was part of the comp token. These are
deleted the same way as any other unselected-comp file IF their bare comp
isn't the dataset's selected comp. A bare-comp file is NEVER auto-deleted
when its comp *is* the selected comp, regardless of file type (calibration.json
included, not just visualization.png) -- that's flagged for manual review
instead. This distinction matters: an earlier version of this script treated
a bare "_3c_" comp as automatically distinct from "_3c_avg_" and deleted it
as an "unselected" sweep leftover even when "_3c_avg_" was the selected comp
and the bare-named file was actually its only calibration.json (just under
old naming, because variants.csv/visualization.png had already been renamed
to "_3c_avg_" by a later run but calibration.json hadn't) -- silently
destroying the only calibration data for that (dataset, comp). Flagging
instead of deleting in this case is the fix.

Uses analysis.discovery.discover_outputs / resolve_component -- the same
functions the rest of analysis/ uses to pick "the" comp for a dataset -- so a
file is only ever flagged for deletion if it belongs to a comp that
resolve_component would NOT have picked, guaranteeing consistency with what
the notebook/CLI actually loads. Dataset-level files with no comp in the name
(*_comparison.png, *_config_metrics.json, model_selection.json, logs/) are
never touched.

Defaults to a dry run -- prints what WOULD be deleted and the total bytes
reclaimed, but deletes nothing. Pass --execute to actually delete.

Usage
-----
python analysis/cleanup_unused_variants.py \\
    --output-dir /data/ross/assay_calibration/explorer_jobs_pp_revisions_calib/ \\
    --dataset-configs src/igvf_configs/dataset_configs_jul_2026.json
# review the printed list, then:
python analysis/cleanup_unused_variants.py \\
    --output-dir /data/ross/assay_calibration/explorer_jobs_pp_revisions_calib/ \\
    --dataset-configs src/igvf_configs/dataset_configs_jul_2026.json \\
    --execute
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.discovery import discover_outputs, resolve_component, _parse_stem

# Comp-tagged file suffixes: (real file extension, tag passed to _parse_stem
# after stripping that extension). _parse_stem already handles the
# dataset/[method]/comp token-splitting logic uniformly -- reused here rather
# than re-implemented so orphan detection stays consistent with how
# discover_outputs parses *_variants.csv / *_calibration.json.
COMP_TAGGED_SUFFIXES = [
    (".csv", "_variants"),
    (".json", "_calibration"),
    (".json.gz", "_lr_values"),
    (".png", "_visualization"),
]


def _parse_any_comp_file(filename: str):
    """Return (dataset, method, comp) for any comp-tagged output filename, or
    None for dataset-level files (*_comparison.png, *_config_metrics.json,
    model_selection.json) that have no comp in the name.
    """
    for ext, tag in COMP_TAGGED_SUFFIXES:
        if filename.endswith(ext):
            stem = filename[: -len(ext)]
            parsed = _parse_stem(stem, tag)
            if parsed:
                return parsed
    return None


def find_files_to_delete(output_dir: Path, dataset_configs: dict):
    """Return (keep_summary, delete_paths, flagged) where:
      keep_summary : [(dataset, selected_comp, available_comps)]
      delete_paths : sorted list of Paths safe to delete
      flagged      : [(path, reason)] -- comp-tagged files matching the
                      selected comp but with no *_calibration.json backing
                      it; not deleted even with --execute.
    """
    tree, model_selections, calibrations = discover_outputs(output_dir)

    delete_paths = set()
    flagged = []
    keep_summary = []

    all_datasets = sorted(set(tree.keys()) | set(calibrations.keys()))
    for dataset in all_datasets:
        comp_dict = tree.get(dataset, {})
        cal_method_dict = calibrations.get(dataset, {})
        available_comps = sorted(
            set(comp_dict.keys())
            | {comp for method_comps in cal_method_dict.values() for comp in method_comps.keys()}
        )
        if not available_comps:
            continue

        selected = resolve_component(dataset, available_comps, model_selections, dataset_configs)
        keep_summary.append((dataset, selected, available_comps))

        comps_with_calibration = {
            comp for method_comps in cal_method_dict.values() for comp in method_comps.keys()
        }

        # Full scan of every comp-tagged file actually on disk for this
        # dataset (not just the ones discover_outputs tracks) -- this is
        # what catches old-naming orphans like a bare "_2c_visualization.png"
        # that discover_outputs never sees (it only walks *_variants.csv /
        # *_calibration.json).
        ds_dir = output_dir / dataset
        if not ds_dir.is_dir():
            continue
        for path in sorted(ds_dir.iterdir()):
            if not path.is_file():
                continue
            parsed = _parse_any_comp_file(path.name)
            if parsed is None:
                continue  # dataset-level file, e.g. *_comparison.png -- never touched
            file_dataset, _method, comp = parsed
            if file_dataset != dataset:
                continue

            if comp == selected:
                if comp not in comps_with_calibration:
                    flagged.append((path, f"{dataset}: matches selected comp '{selected}' but no "
                                            f"*_calibration.json backs it -- possibly the only "
                                            f"remaining artifact for the selected config"))
                continue  # never delete anything matching the selected comp

            delete_paths.add(path)

    return keep_summary, sorted(delete_paths), flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-configs", required=True)
    parser.add_argument("--execute", action="store_true",
                         help="Actually delete files (default: dry run, print only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    with open(args.dataset_configs) as f:
        dataset_configs = json.load(f)

    keep_summary, delete_paths, flagged = find_files_to_delete(output_dir, dataset_configs)

    print(f"Selected comp per dataset ({len(keep_summary)} dataset(s) with output):")
    for dataset, selected, available in keep_summary:
        extra = f"  [also present: {[c for c in available if c != selected]}]" if len(available) > 1 else ""
        print(f"  {dataset}: keeping {selected}{extra}")

    if flagged:
        print(f"\n{len(flagged)} file(s) FLAGGED for manual review (never auto-deleted):")
        for path, reason in flagged:
            print(f"  {path}\n    -> {reason}")

    if not delete_paths:
        print("\nNothing to delete -- every dataset only has its selected comp's files.")
        return

    total_bytes = sum(p.stat().st_size for p in delete_paths if p.exists())
    print(f"\n{'Would delete' if not args.execute else 'Deleting'} {len(delete_paths)} file(s), "
          f"{total_bytes / 1e6:.1f} MB:")
    for p in delete_paths:
        print(f"  {p}")

    if not args.execute:
        print("\nDry run -- nothing deleted. Re-run with --execute to actually delete these files.")
        return

    deleted = 0
    for p in delete_paths:
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  FAILED to delete {p}: {e}")
    print(f"\nDeleted {deleted}/{len(delete_paths)} file(s).")


if __name__ == "__main__":
    main()
