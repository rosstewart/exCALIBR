#!/usr/bin/env python3
"""
ExCALIBR bootstrap fit worker — runs one SLURM array task.

Usage:
    python slurm/run_array_task.py <output_dir> <array_idx>

Loads <output_dir>/jobs/array_<NNNN>.pkl, runs the fits it describes,
and writes per-bootstrap results to <output_dir>/<dataset_name>/.

Resume safety
-------------
Before running fits for a bootstrap, the worker checks whether a result
file already exists and which component keys (e.g. '2c', '3c') are already
present.  Only missing components are fitted; completed ones are left intact
and merged back into the saved file on write.

Skip list
---------
If <output_dir>/skip_datasets.txt exists, datasets listed there (one per
line, # comments allowed) are silently skipped.
"""
import sys
import os
import pickle
import concurrent.futures
from collections import defaultdict
from pathlib import Path

import numpy as np

# Allow running from any directory as long as the repo root is on the path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import Fit


# ---------------------------------------------------------------------------
# Module-level worker — must be importable by ProcessPoolExecutor
# ---------------------------------------------------------------------------

def _run_one_fit(fit_spec):
    """Execute a single fit job and return (bootstrap_seed, label, save_dir, fit_idx, result)."""
    full_job, bootstrap_seed, label, save_dir = fit_spec
    fit_idx = full_job.get("fit_idx")
    try:
        result = Fit.execute_fit_job(full_job)
    except Exception as e:
        print(f"  ✗ fit failed (bootstrap={bootstrap_seed}, label={label}): {e}")
        result = None
    return bootstrap_seed, label, save_dir, fit_idx, result


# ---------------------------------------------------------------------------
# Main task function
# ---------------------------------------------------------------------------

def _load_skip_datasets(output_dir: Path) -> set:
    skip_file = output_dir / "skip_datasets.txt"
    if not skip_file.exists():
        return set()
    skip = set()
    for line in skip_file.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            skip.add(line)
    if skip:
        print(f"Skip list loaded ({len(skip)} datasets): {', '.join(sorted(skip))}")
    return skip


def run_array_task(output_dir: str, array_idx: int) -> None:
    output_dir = Path(output_dir).resolve()
    jobs_dir = output_dir / "jobs"
    array_file = jobs_dir / f"array_{array_idx:04d}.pkl"

    if not array_file.exists():
        print(f"Error: {array_file} not found")
        sys.exit(1)

    with open(array_file, "rb") as f:
        jobs = pickle.load(f)

    skip_datasets = _load_skip_datasets(output_dir)
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    total_fits = sum(j["num_fits_total"] for j in jobs)
    print(f"Array task {array_idx}: {len(jobs)} bootstraps, {total_fits} fits, {n_cpus} CPUs")

    # ── Build flat list of fit specs, skipping already-completed components ──
    fit_specs = []
    fits_per_bootstrap: dict = defaultdict(int)   # (bs_seed, save_dir) → pending count

    for cjob in jobs:
        dataset_name = cjob["dataset_name"]
        if dataset_name in skip_datasets:
            continue

        # Derive save_dir from output_dir + dataset_name — ignores the baked-in
        # path so that jobs are portable across filesystem mounts.
        save_dir = output_dir / dataset_name
        bs_seed = cjob["bootstrap_seed"]
        shared = cjob["shared_data"]
        is_mv = cjob.get("multivariate", False)

        # Load any existing results for this bootstrap
        save_file = save_dir / f"bootstrap_{bs_seed}_best_fits.pkl"
        existing: dict = {}
        if save_file.exists():
            try:
                with open(save_file, "rb") as f:
                    existing = pickle.load(f)
            except Exception:
                existing = {}

        nc_keys = sorted(k for k in cjob if k.startswith("jobs_") and isinstance(cjob[k], list))
        for nc_key in nc_keys:
            label = nc_key[len("jobs_"):]               # e.g. "2c", "3c"
            if label in existing and existing[label] is not None:
                continue                                  # already complete

            for minimal_job in cjob[nc_key]:
                full_job = {**minimal_job, **shared,
                            "dataset_name": dataset_name,
                            "save_dir": str(save_dir)}
                if is_mv:
                    full_job["multivariate"] = True
                # Remove stale multivariate kwarg if present
                if "kwargs" in full_job:
                    full_job["kwargs"].pop("multivariate", None)

                fit_specs.append((full_job, bs_seed, label, str(save_dir)))
                fits_per_bootstrap[(bs_seed, str(save_dir))] += 1

    if not fit_specs:
        print("All fits already completed — nothing to do.")
        return

    print(f"Submitting {len(fit_specs)} fits across {len(fits_per_bootstrap)} bootstrap(s)")

    # ── Run fits in parallel, track best per (bootstrap, label, save_dir) ──
    # ProcessPoolExecutor + as_completed() yields results in OS-scheduling
    # order, not submission order, so the tie-break key below includes
    # fit_idx (smaller wins on an exact val_ll tie) — this keeps the best-fit
    # selection deterministic and independent of which worker finishes first,
    # matching the joblib.Parallel-based pipeline path (which preserves
    # submission order).
    best: dict = {}                              # key → (val_ll, -fit_idx, result)
    remaining = dict(fits_per_bootstrap)

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as ex:
        futures = [ex.submit(_run_one_fit, spec) for spec in fit_specs]

        for fut in concurrent.futures.as_completed(futures):
            try:
                bs_seed, label, save_dir, fit_idx, result = fut.result()
            except Exception as e:
                print(f"  Future raised: {e}")
                continue

            val_ll = (result.get("val_ll", -np.inf) if result else -np.inf)
            sort_key = (val_ll, -(fit_idx if fit_idx is not None else 0))
            key = (bs_seed, label, save_dir)
            if sort_key > best.get(key, (-np.inf, None, None))[:2]:
                best[key] = (val_ll, -(fit_idx if fit_idx is not None else 0), result)

            bs_key = (bs_seed, save_dir)
            remaining[bs_key] -= 1

            # ── Save as soon as all fits for this bootstrap are done ──
            if remaining[bs_key] == 0:
                save_path = Path(save_dir) / f"bootstrap_{bs_seed}_best_fits.pkl"
                new_results = {
                    lbl: res
                    for (b, lbl, sd), (_, _, res) in best.items()
                    if b == bs_seed and sd == save_dir
                }
                # Merge with pre-existing (preserves already-complete components)
                existing = {}
                if save_path.exists():
                    try:
                        with open(save_path, "rb") as f:
                            existing = pickle.load(f)
                    except Exception:
                        existing = {}
                existing.update(new_results)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    pickle.dump(existing, f)
                components = sorted(existing.keys())
                print(f"  ✓ bootstrap {bs_seed} → {save_path.parent.name}  [{', '.join(components)}]")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_array_task.py <output_dir> <array_idx>")
        sys.exit(1)
    run_array_task(sys.argv[1], int(sys.argv[2]))
