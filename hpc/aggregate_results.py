#!/usr/bin/env python3
"""
Aggregate ExCALIBR bootstrap fit results into a single json.gz file.

Usage:
    python hpc/aggregate_results.py <output_dir> [output_file]

Scans all <output_dir>/<dataset>/bootstrap_*_best_fits.pkl files,
merges them, and writes:
    <output_dir>/bootstrap_results.json.gz   (default)
  or the path given as the second argument.

Output structure (mirrors existing pipeline expectations):
    {
        "<dataset_name>": {
            "<bootstrap_seed>": {
                "2c": { ... fit dict ... },
                "3c": { ... fit dict ... }
            }
        }
    }

Multi-condition mode (--per-condition)
---------------------------------------
For a directory holding several independently-fit conditions side by side --
e.g. analysis/build_splice_ablation_jobs.py's splice-ablation sweep, whose
output_root looks like:

    <output_root>/
        thresh_0.1/<dataset_name>/bootstrap_N_best_fits.pkl
        thresh_0.2/<dataset_name>/bootstrap_N_best_fits.pkl
        ...
        keep_all/<dataset_name>/bootstrap_N_best_fits.pkl
        jobs/, logs/, datasets.txt   (SLURM bookkeeping, not a condition)

running the default (single-directory) mode directly on <output_root> is
WRONG: the scan below is `output_dir.rglob(...)`, which recurses into every
condition subdirectory and keys purely on `pkl.parent.name` (just the
dataset name) -- it has no idea a variant's bootstrap fit came from
thresh_0.1 vs thresh_0.9 vs keep_all, so two conditions' fits for the same
(dataset, seed) silently collide and the later one (by sort order) wins,
producing a single bootstrap_results.json.gz that's a corrupted mix of
several different splice-filter populations.

    python hpc/aggregate_results.py <output_root> --per-condition

instead aggregates each condition subdirectory SEPARATELY -- auto-detected
as any immediate subdirectory of <output_root> containing at least one
bootstrap_*_best_fits.pkl anywhere beneath it (this naturally skips jobs/,
logs/, and any other non-condition bookkeeping directories) -- writing
<output_root>/<condition_label>/bootstrap_results.json.gz per condition,
which is exactly the per-condition layout
analysis/run_splice_ablation_calibration.py expects.
"""
import sys
import json
import gzip
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if _HAS_SKLEARN and isinstance(obj, KMeans):
            return {
                "type": "KMeans",
                "n_clusters": int(obj.n_clusters),
                "cluster_centers": obj.cluster_centers_.tolist() if hasattr(obj, "cluster_centers_") else None,
                "labels": obj.labels_.tolist() if hasattr(obj, "labels_") else None,
                "inertia": float(obj.inertia_) if hasattr(obj, "inertia_") else None,
            }
        return super().default(obj)


def discover_condition_dirs(output_dir: Path) -> list:
    """Immediate subdirectories of `output_dir` that contain at least one
    bootstrap_*_best_fits.pkl anywhere beneath them -- i.e. an actual fitted
    condition (thresh_0.1, keep_all, ...), not SLURM bookkeeping (jobs/,
    logs/) or plain files (datasets.txt, an already-aggregated
    bootstrap_results.json.gz) sitting alongside them at the same level."""
    condition_dirs = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        if next(child.rglob("bootstrap_*_best_fits.pkl"), None) is not None:
            condition_dirs.append(child)
    return condition_dirs


def aggregate_results(output_dir: Path, output_file: Path) -> None:
    all_results: dict = defaultdict(dict)

    pkl_files = sorted(output_dir.rglob("bootstrap_*_best_fits.pkl"))
    if not pkl_files:
        print(f"No bootstrap_*_best_fits.pkl files found under {output_dir}")
        sys.exit(1)

    n_loaded = 0
    n_failed = 0
    for pkl in pkl_files:
        dataset_name = pkl.parent.name
        parts = pkl.stem.split("_")
        try:
            seed = int(parts[1])
        except (IndexError, ValueError):
            print(f"  Warning: could not parse seed from {pkl.name}, skipping")
            n_failed += 1
            continue
        try:
            with open(pkl, "rb") as f:
                fits = pickle.load(f)
            all_results[dataset_name][seed] = fits
            n_loaded += 1
        except Exception as e:
            print(f"  Warning: failed to load {pkl}: {e}")
            n_failed += 1

    print(f"Loaded {n_loaded} files ({n_failed} failed)")
    print(f"Datasets: {len(all_results)}")
    total_bootstraps = sum(len(v) for v in all_results.values())
    print(f"Total bootstrap entries: {total_bootstraps:,}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        json.dump(all_results, f, cls=_NumpyEncoder, indent=2)

    size_mb = output_file.stat().st_size / 1e6
    print(f"Saved → {output_file}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate ExCALIBR bootstrap results")
    parser.add_argument("output_dir", help="ExCALIBR output directory")
    parser.add_argument("output_file", nargs="?",
                        help="Destination json.gz path (default: <output_dir>/bootstrap_results.json.gz). "
                             "Ignored with --per-condition, which always writes "
                             "<output_dir>/<condition>/bootstrap_results.json.gz per condition.")
    parser.add_argument("--per-condition", action="store_true",
                        help="Treat <output_dir> as a multi-condition sweep root (e.g. "
                             "analysis/build_splice_ablation_jobs.py's thresh_0.1../keep_all "
                             "layout) and aggregate each condition subdirectory separately, "
                             "instead of recursively merging everything under <output_dir> "
                             "into one (dataset, seed)-keyed file, which would silently "
                             "collide different conditions' fits for the same dataset/seed.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    if args.per_condition:
        condition_dirs = discover_condition_dirs(output_dir)
        if not condition_dirs:
            print(f"No condition subdirectories with bootstrap_*_best_fits.pkl found under {output_dir}")
            sys.exit(1)
        print(f"Discovered {len(condition_dirs)} condition(s): "
              f"{', '.join(d.name for d in condition_dirs)}")
        for condition_dir in condition_dirs:
            print(f"\n{'='*80}\n[{condition_dir.name}]\n{'='*80}")
            aggregate_results(condition_dir, condition_dir / "bootstrap_results.json.gz")
    else:
        output_file = (
            Path(args.output_file).resolve() if args.output_file else (output_dir / "bootstrap_results.json.gz")
        )
        aggregate_results(output_dir, output_file)
