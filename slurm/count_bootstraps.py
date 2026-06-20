#!/usr/bin/env python3
"""
Report bootstrap fit completion progress for an ExCALIBR output directory.

Usage:
    python slurm/count_bootstraps.py <output_dir> [--verbose]

Reads <output_dir>/jobs/job_index.json to know which datasets and how many
bootstraps are expected, then scans for
    <output_dir>/<dataset>/bootstrap_<N>_best_fits.pkl
and reports per-dataset, per-component counts.
"""
import sys
import json
import pickle  # used only to read the first bootstrap file for component discovery
import argparse
from pathlib import Path
from collections import defaultdict


def count_bootstraps(output_dir: Path, verbose: bool = False) -> None:
    jobs_dir = output_dir / "jobs"
    index_file = jobs_dir / "job_index.json"

    if not index_file.exists():
        print(f"Error: {index_file} not found.  Run prepare_batch_jobs.py first.")
        sys.exit(1)

    with open(index_file) as f:
        index = json.load(f)

    # Collect expected (dataset, bootstrap_seed) pairs and which components each has
    expected: dict = defaultdict(lambda: {"bootstraps": set(), "components": set()})
    for entry in index["job_index"]:
        ds = entry["dataset_name"]
        expected[ds]["bootstraps"].add(entry["bootstrap_seed"])

    n_bootstraps = index.get("n_bootstraps", max(
        len(v["bootstraps"]) for v in expected.values()
    ) if expected else 0)

    # Scan saved results by file existence; also load the lowest-seed file per
    # dataset to discover which components are present for that dataset.
    done: dict = defaultdict(set)          # dataset → {seeds present}
    ds_components: dict = defaultdict(set) # dataset → components found in lowest-seed file

    all_known_components: set = set()

    for ds in expected:
        ds_dir = output_dir / ds
        if not ds_dir.is_dir():
            continue

        seed_to_pkl: dict = {}
        for pkl in ds_dir.glob("bootstrap_*_best_fits.pkl"):
            parts = pkl.stem.split("_")
            try:
                seed = int(parts[1])
            except (IndexError, ValueError):
                continue
            seed_to_pkl[seed] = pkl
            done[ds].add(seed)

        if seed_to_pkl:
            lowest_pkl = seed_to_pkl[min(seed_to_pkl)]
            try:
                with open(lowest_pkl, "rb") as f:
                    data = pickle.load(f)
                ds_components[ds] = {k for k, v in data.items() if v is not None}
                all_known_components |= ds_components[ds]
            except Exception:
                pass

    # Summarise
    all_datasets = sorted(expected.keys())
    n_datasets = len(all_datasets)
    total_expected_bs = sum(len(v["bootstraps"]) for v in expected.values())
    total_done_bs = sum(
        sum(1 for seed in expected[ds]["bootstraps"] if seed in done[ds])
        for ds in all_datasets
    )

    print(f"Output dir : {output_dir}")
    print(f"Datasets   : {n_datasets}")
    print(f"Bootstraps : {total_done_bs:,} / {total_expected_bs:,} "
          f"({100 * total_done_bs / max(total_expected_bs, 1):.1f}%)")
    print()

    sorted_components = sorted(all_known_components)
    col_w = max(len(ds) for ds in all_datasets) + 2 if all_datasets else 20
    comp_cols = "  ".join(f"{c:>8}" for c in sorted_components)
    header = f"{'Dataset':<{col_w}}  {'Done':>6}  {'Exp':>6}  {'Pct':>6}  {comp_cols}"
    print(header)
    print("-" * len(header))

    total_comp_done = defaultdict(int)
    for ds in all_datasets:
        bs_expected = sorted(expected[ds]["bootstraps"])
        n_exp = len(bs_expected)
        n_done = sum(1 for seed in bs_expected if seed in done[ds])
        pct = 100 * n_done / max(n_exp, 1)

        present = ds_components.get(ds, set())
        comp_str = "  ".join(f"{n_done if c in present else 0:>8}" for c in sorted_components)
        for c in sorted_components:
            if c in present:
                total_comp_done[c] += n_done

        print(f"{ds:<{col_w}}  {n_done:>6}  {n_exp:>6}  {pct:>5.1f}%  {comp_str}")

        if verbose:
            missing = [s for s in bs_expected if s not in done[ds]]
            if missing:
                chunks = [missing[i:i+20] for i in range(0, len(missing), 20)]
                for chunk in chunks:
                    print(f"  {'missing seeds:':20s} {chunk}")

    print()
    totals = "  ".join(f"{c}={total_comp_done[c]:,}" for c in sorted_components)
    print(f"Component totals: {totals}  (of {total_expected_bs:,} each)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count ExCALIBR bootstrap fit completions")
    parser.add_argument("output_dir", help="ExCALIBR output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print missing bootstrap seeds per dataset")
    args = parser.parse_args()
    count_bootstraps(Path(args.output_dir).resolve(), verbose=args.verbose)
