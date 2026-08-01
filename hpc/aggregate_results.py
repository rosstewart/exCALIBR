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
                        help="Destination json.gz path (default: <output_dir>/bootstrap_results.json.gz)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_file = Path(args.output_file).resolve() if args.output_file else (output_dir / "bootstrap_results.json.gz")

    aggregate_results(output_dir, output_file)
