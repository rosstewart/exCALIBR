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
import pickle
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

    # Scan saved results
    done: dict = defaultdict(lambda: defaultdict(set))  # dataset → seed → {components}

    for ds, info in expected.items():
        ds_dir = output_dir / ds
        if not ds_dir.is_dir():
            continue
        for pkl in sorted(ds_dir.glob("bootstrap_*_best_fits.pkl")):
            parts = pkl.stem.split("_")
            try:
                seed = int(parts[1])
            except (IndexError, ValueError):
                continue
            try:
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
                components = {k for k, v in data.items() if v is not None}
                done[ds][seed] = components
            except Exception:
                done[ds][seed] = set()

    # Summarise
    all_datasets = sorted(expected.keys())
    n_datasets = len(all_datasets)
    total_expected_bs = sum(len(v["bootstraps"]) for v in expected.values())
    total_done_bs = sum(
        sum(1 for seed in expected[ds]["bootstraps"] if done[ds][seed])
        for ds in all_datasets
    )

    print(f"Output dir : {output_dir}")
    print(f"Datasets   : {n_datasets}")
    print(f"Bootstraps : {total_done_bs:,} / {total_expected_bs:,} "
          f"({100 * total_done_bs / max(total_expected_bs, 1):.1f}%)")
    print()

    col_w = max(len(ds) for ds in all_datasets) + 2 if all_datasets else 20
    header = f"{'Dataset':<{col_w}}  {'Done':>6}  {'Exp':>6}  {'Pct':>6}  Components"
    print(header)
    print("-" * len(header))

    for ds in all_datasets:
        bs_expected = sorted(expected[ds]["bootstraps"])
        n_exp = len(bs_expected)
        n_done = sum(1 for seed in bs_expected if done[ds][seed])
        pct = 100 * n_done / max(n_exp, 1)

        # Collect all component sets that appear
        comp_counts: dict = defaultdict(int)
        for seed in bs_expected:
            for c in done[ds].get(seed, set()):
                comp_counts[c] += 1
        comp_str = "  ".join(f"{c}:{v}" for c, v in sorted(comp_counts.items()))

        print(f"{ds:<{col_w}}  {n_done:>6}  {n_exp:>6}  {pct:>5.1f}%  {comp_str}")

        if verbose:
            missing = [s for s in bs_expected if not done[ds][s]]
            if missing:
                chunks = [missing[i:i+20] for i in range(0, len(missing), 20)]
                for chunk in chunks:
                    print(f"  {'missing seeds:':20s} {chunk}")

    print()
    total_2c = sum(
        sum(1 for seed in expected[ds]["bootstraps"] if "2c" in done[ds].get(seed, set()))
        for ds in all_datasets
    )
    total_3c = sum(
        sum(1 for seed in expected[ds]["bootstraps"] if "3c" in done[ds].get(seed, set()))
        for ds in all_datasets
    )
    print(f"Component totals:  2c={total_2c:,}  3c={total_3c:,}  "
          f"(of {total_expected_bs:,} each, where applicable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count ExCALIBR bootstrap fit completions")
    parser.add_argument("output_dir", help="ExCALIBR output directory")
    parser.add_argument("--verbose", action="store_true",
                        help="Print missing bootstrap seeds per dataset")
    args = parser.parse_args()
    count_bootstraps(Path(args.output_dir).resolve(), verbose=args.verbose)
