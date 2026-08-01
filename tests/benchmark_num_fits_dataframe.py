#!/usr/bin/env python3
"""
No-bootstrap, all-dataset restart-count sweep, driven directly from the
IGVF dataframe + dataset-config JSON (same inputs as
`slurm/prepare.py pillar_project`), but on the CPU fitting path
(Fit.generate_fit_jobs + tryToFit), not the GPU/jax_batch path.

`slurm/prepare.py`'s pillar_project/default modes hardcode 1000 bootstraps x
100 fits with no CLI override, so this script builds an unbootstrapped
(bootstrap=False, whole dataset) Fit job list directly instead, runs
`--num-fits` restarts per dataset, and reports (a) how best-of-N degrades as
N shrinks from 100 down to 1, mirroring tests/benchmark_num_fits.py's
Monte-Carlo subsampling, and (b) how often the best restart's skew sign
changed from its own initialization to the converged fit -- the real-data
counterpart of tests/verify_skew_sign_change_univariate.py.

Usage:
    python tests/benchmark_num_fits_dataframe.py /tmp/no_bootstrap_run
    python tests/benchmark_num_fits_dataframe.py /tmp/no_bootstrap_run \\
        --datasets BRCA1_Findlay_2018 PTEN_Matreyek_2018 --num-fits 20 --plot
"""
import argparse
import csv
import json
import multiprocessing as mp
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR_DEFAULT = "/data/ross/assay_calibration/hyperparam_sweep_oldclinvar/benchmark_num_fits_dataframe"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from joblib import Parallel, delayed

from slurm.prepare import _components_from_config, _requires_2018
from src.assay_calibration.data_utils.dataset import Scoreset
from src.assay_calibration.fit_utils.fit import Fit, _weighted_val_ll

RESTART_GRID = sorted({1, 5, 8, 10, 16, 20, 30, 32, 40, 50, 100})
N_SUBSAMPLES = 2000
_DEFAULT_DATAFRAME = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
_DEFAULT_CONFIG = str(_REPO_ROOT / "src" / "igvf_configs" / "dataset_configs_jul_2026.json")
_NAME_MAP_CSV = "/data/ross/assay_calibration/dataframe/new_dataset_names.csv"


class Tee:
    """Write to both a file and stdout simultaneously.

    Duplicated (not imported) from tests/benchmark_num_fits.py, which
    unconditionally imports jax at module scope -- this script and
    tests/compare_lambda_vs_random_init.py are CPU-only and must not require
    a jax install just to get this tiny logging helper.
    """
    def __init__(self, path):
        self._file = open(path, "w")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def best_of_n_stats(val_lls, n_fits_grid, rng):
    """Monte Carlo estimate of E[best of N] for each N in n_fits_grid.

    Duplicated (not imported) from tests/benchmark_num_fits.py for the same
    jax-import-avoidance reason as Tee above; logic is identical.
    """
    val_lls = np.asarray(val_lls)
    valid = val_lls[np.isfinite(val_lls)]
    n_valid = len(valid)
    baseline = float(valid.max()) if n_valid > 0 else -np.inf

    stats = {}
    for N in n_fits_grid:
        if N >= n_valid:
            stats[N] = (baseline, 0.0)
            continue
        bests = np.array([
            valid[rng.choice(n_valid, size=N, replace=False)].max()
            for _ in range(N_SUBSAMPLES)
        ])
        stats[N] = (float(bests.mean()), float(bests.std()))
    return baseline, n_valid, stats


# ---------------------------------------------------------------------------

def build_unbootstrapped_jobs(scoreset, dataset_name, n_c, num_fits, master_seed=None,
                              bootstrap_seed=0):
    """Whole-dataset (bootstrap=False) restart job list for one dataset/n_c.

    bootstrap_seed is fixed (not None) purely so the init_method draw
    (fit.py:647-648) and one-hot RNG are reproducible across runs -- with
    bootstrap=False, train/val split ignores it entirely (fit.py:625-636),
    so no actual resampling happens.

    Shared by this script and tests/compare_lambda_vs_random_init.py so the
    exact same Scoreset/Fit/job-construction path is used by both --
    generate_fit_jobs already assigns a per-restart lambdaIndex + fit_seed
    (fit.py:663-664); compare_lambda_vs_random_init.py's "random" arm simply
    strips lambdaIndex back out of a copy of these jobs' kwargs.
    """
    fitter = Fit(scoreset)
    jobs = fitter.generate_fit_jobs(
        component_range=[n_c], bootstrap=False, bootstrap_seed=bootstrap_seed,
        check_monotonic=True, num_fits=num_fits, master_seed=master_seed,
    )
    for job in jobs:
        job["dataset_name"] = dataset_name
    return fitter, jobs


def _train_ll(job, result):
    """Same guard + call as Fit.run's internal _safe_ll (fit.py:317-330),
    but against a single already-computed tryToFit/execute_fit_job result
    instead of Fit.run's own model list -- lets us keep every restart's LL,
    not just the winner Fit.run would have picked."""
    fit = result.get("fit", result)
    params = fit.get("component_params")
    weights = fit.get("weights")
    if not params or weights is None or any(
        isinstance(p, (list, tuple)) and len(p) == 0 for p in params
    ):
        return -np.inf
    try:
        return _weighted_val_ll(
            job["train_observations"], job["train_sample_assignments"],
            params, weights, job.get("multivariate", False), job.get("kwargs", {}),
        )
    except Exception:
        return -np.inf


def run_restarts(jobs, n_jobs=-1):
    """Run every job through Fit.execute_fit_job (same static dispatcher the
    real pipeline uses, fit.py:685-747) and attach an in-sample train_ll
    (execute_fit_job leaves val_ll=None when there's no val split).

    backend="loky" is required (not just the default), not optional: this is
    called from inside process_dataset_nc/process_dataset_arms, which are
    themselves already dispatched via an outer joblib.Parallel -- joblib
    silently downgrades a *nested* Parallel call to ThreadingBackend by
    default (to avoid runaway subprocess explosion), which for CPU-bound,
    largely-Python EM code (not GIL-releasing the whole time) barely
    parallelizes at all. Forcing backend="loky" here overrides that default
    and gets genuine separate worker processes for the inner level too --
    confirmed empirically (a nested Parallel without this only used the
    single parent PID for every "worker"; with it, distinct child PIDs).
    """
    results = Parallel(n_jobs=n_jobs, batch_size=1, backend="loky")(
        delayed(Fit.execute_fit_job)(job) for job in jobs
    )
    for job, res in zip(jobs, results):
        res["train_ll"] = _train_ll(job, res)
    return results


def sign_change(initial_params, final_params):
    init_signs = tuple(int(np.sign(p[0])) for p in initial_params)
    final_signs = tuple(int(np.sign(p[0])) for p in final_params)
    return init_signs, final_signs, init_signs != final_signs


def resolve_inner_jobs(n_jobs_outer, n_work_items, n_jobs_inner):
    """Auto-scale inner (within-unit) parallelism to actually use all cores.

    Outer parallelism (joblib.Parallel over work_items) can never exceed
    n_work_items regardless of n_jobs_outer -- e.g. `--top-k 20` caps outer
    parallelism at 20 workers even with 128 cores available, silently
    leaving ~108 cores idle if inner parallelism is left at 1. When the
    caller hasn't explicitly set --n-jobs-inner (n_jobs_inner is None), split
    total CPUs between the two levels the same way run_igvf_batch.py does
    (n_jobs_inner = max(1, n_cpus // outer_workers_actually_used)) so the
    product of (actual outer workers) x (inner workers) saturates the
    machine instead of just the outer count.
    """
    if n_jobs_inner is not None:
        return n_jobs_inner
    total_cpus = mp.cpu_count()
    outer_workers = total_cpus if n_jobs_outer == -1 else max(1, n_jobs_outer)
    outer_workers = max(1, min(outer_workers, max(1, n_work_items)))
    return max(1, total_cpus // outer_workers)


# ---------------------------------------------------------------------------

def checkpoint_path_for(checkpoint_dir, dataset, n_c):
    """Stable, filesystem-safe checkpoint filename for one (dataset, n_c)
    unit -- sanitizes anything that isn't alnum/-/_ (dataset names carry
    unicode letters, spaces, parens; a hash suffix disambiguates any
    collisions the sanitization might introduce)."""
    import hashlib
    import re
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", f"{dataset}__{n_c}c")
    digest = hashlib.md5(f"{dataset}|{n_c}".encode()).hexdigest()[:8]
    return Path(checkpoint_dir) / f"{safe}_{digest}.pkl"


def process_dataset_nc(dataset, n_c, scoreset, num_fits, seed, n_jobs_inner, want_plot,
                       checkpoint_dir=None):
    """One (dataset, n_c) unit of work -- everything from job construction
    through best-of-N stats and the sign-change record. Top-level (not a
    closure) so joblib/loky can pickle+dispatch it to a worker process; each
    call is independent (its own Scoreset/jobs/results), so this is the unit
    parallelized *across* datasets (outer), while n_jobs_inner controls the
    joblib pool used for that one dataset's restarts (inner) -- same
    outer/inner split run_igvf_batch.py uses (--n-jobs/--n-jobs-inner) to
    avoid oversubscribing when both levels are parallel at once.

    checkpoint_dir, when given, makes this unit resumable: the result is
    pickled to disk the moment this unit finishes (inside the worker, so a
    later crash of the main process or a sibling unit can't lose it), and if
    that checkpoint file already exists on entry, the unit is skipped
    entirely and the cached result is returned instead. This is the fix for
    a real gap found the hard way: with no per-unit persistence, an OOM kill
    (e.g. from other, unrelated jobs on a shared machine) partway through a
    ~90-unit run previously lost 100% of already-completed work, not just
    the in-flight unit.
    """
    ckpt = checkpoint_path_for(checkpoint_dir, dataset, n_c) if checkpoint_dir else None
    if ckpt is not None and ckpt.exists():
        import pickle
        with open(ckpt, "rb") as f:
            out = pickle.load(f)
        out["log"] = [out["log"][0] + " [resumed from checkpoint]"] + out["log"][1:]
        return out

    def _finish(result):
        if ckpt is not None:
            import pickle
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt, "wb") as f:
                pickle.dump(result, f)
        return result

    label = f"{dataset} ({n_c}c)"
    log = [label]

    _, jobs = build_unbootstrapped_jobs(scoreset, dataset, n_c, num_fits, master_seed=seed)
    if not jobs:
        log.append("  SKIP: no jobs generated")
        return _finish({"dataset": dataset, "n_c": n_c, "log": log})

    results = run_restarts(jobs, n_jobs=n_jobs_inner)
    train_lls = np.array([r["train_ll"] for r in results])

    rng = np.random.default_rng(seed)
    baseline, n_valid, stats = best_of_n_stats(train_lls, RESTART_GRID, rng)
    n_failed = len(train_lls) - n_valid
    log.append(f"  baseline (best/{num_fits}) = {baseline:.4f}"
               f"{f'  ({n_failed} failed)' if n_failed else ''}")

    summary_rows = []
    for N in RESTART_GRID:
        mean, std = stats[N]
        delta = mean - baseline
        log.append(f"    fits={N:3d}:  {mean:.4f} ± {std:.4f}  Δ={delta:+.4f}")
        summary_rows.append({
            "dataset": dataset, "n_c": n_c, "num_fits": N,
            "baseline": round(baseline, 6), "mean_best": round(mean, 6),
            "std_best": round(std, 6), "delta": round(delta, 6),
            "n_valid": n_valid, "n_failed": n_failed,
        })

    out = {
        "dataset": dataset, "n_c": n_c, "log": log,
        "train_lls": train_lls.tolist(), "summary_rows": summary_rows,
        "sign_row": None, "plot_record": None,
    }
    if n_valid == 0:
        log.append("  no valid fits; skipping sign-change/plot record")
        return _finish(out)

    best_idx = int(np.nanargmax(train_lls))
    best = results[best_idx]["fit"]
    init_signs, final_signs, changed = sign_change(best["initial_params"], best["component_params"])
    log.append(f"  best fit sign change: init={init_signs} final={final_signs} "
               f"{'CHANGED' if changed else 'same'}")
    out["sign_row"] = {
        "dataset": dataset, "n_c": n_c, "best_train_ll": round(float(train_lls[best_idx]), 6),
        "init_signs": str(init_signs), "final_signs": str(final_signs), "changed": changed,
    }

    if want_plot:
        out["plot_record"] = {
            "dataset_name": label,
            "observations": jobs[best_idx]["train_observations"],
            "sample_assignments": jobs[best_idx]["train_sample_assignments"],
            "initial_params": best["initial_params"],
            "component_params": best["component_params"],
            "final_weights": np.mean(best["weights"], axis=0) if best.get("weights") is not None else None,
        }
    return _finish(out)


def _load_new_to_old():
    try:
        name_map_df = pd.read_csv(_NAME_MAP_CSV)
        return dict(zip(name_map_df["New_names"], name_map_df["Old_names"]))
    except FileNotFoundError:
        return {}


def build_scoreset_for_dataset(dataset, df_ds, clinvar_release, population_type):
    """One dataset's Scoreset construction -- top-level (not a closure) so
    joblib/loky can dispatch it to a worker process.

    Scoreset.__init__ does real per-dataset work (ClinVar annotation
    lookup, splicing filters -- visible in its own log lines, e.g. "dropped
    N splice-consequence rows"), not the "cheap" preprocessing this script
    originally assumed when it built all 89 datasets' Scoresets in a plain
    sequential for-loop before any parallel fitting started. That loop was
    the actual bottleneck in practice (tens of minutes before the first fit
    even ran) -- parallelizing it here, independently per dataset, fixes
    that. backend="loky" (real processes) rather than threads: it's not
    known how much of Scoreset.__init__ releases the GIL, and processes
    guarantee real parallelism regardless, consistent with how the rest of
    this script already parallelizes.

    Returns (dataset, scoreset_or_None, error_or_None).
    """
    try:
        kw = dict(clinvar_release=clinvar_release, min_clinvar_star=1)
        if population_type:
            kw["population_type"] = population_type
        scoreset = Scoreset(df_ds, **kw)
    except (ValueError, KeyError) as e:
        return dataset, None, str(e)
    if sum(1 for _ in scoreset.samples) < 2:
        return dataset, None, "insufficient samples"
    return dataset, scoreset, None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Unused placeholder for parity with benchmark_num_fits.py's "
                             "positional arg; results always go to --results-dir.")
    parser.add_argument("--dataframe", default=_DEFAULT_DATAFRAME)
    parser.add_argument("--config-file", default=_DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Only process these dataset names (default: all)")
    parser.add_argument("--num-fits", type=int, default=100,
                        help="Max restarts to run per dataset (default: 100)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Outer parallelism: (dataset, n_c) units run concurrently "
                             "(default: -1 = all CPUs)")
    parser.add_argument("--n-jobs-inner", type=int, default=None,
                        help="Inner parallelism: restarts within one (dataset, n_c) unit. "
                             "Default: auto -- split total CPUs between outer and inner so "
                             "the product saturates the machine (matters when the number of "
                             "(dataset, n_c) units is smaller than the core count, e.g. "
                             "--datasets/--top-k restricted to a handful of datasets on a "
                             "128-core box would otherwise leave most cores idle). Set "
                             "explicitly to override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clinvar-release", default="2025",
                        help="Default ClinVar release (pillar_project uses 2025)")
    parser.add_argument("--population-type", default=None)
    parser.add_argument("--results-dir", default=str(_RESULTS_DIR_DEFAULT))
    parser.add_argument("--plot", action="store_true",
                        help="Save a plot_initial_vs_final_skew_grid PNG of each "
                             "dataset's best restart")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tee = Tee(results_dir / "run.log")
    sys.stdout = tee

    print(f"Restart grid  : {RESTART_GRID}")
    print(f"num_fits      : {args.num_fits}")
    print(f"Dataframe     : {args.dataframe}")
    print(f"Config file   : {args.config_file}")
    print(f"Results dir   : {results_dir}")
    print()

    sep = "\t" if args.dataframe.endswith((".tsv", ".tsv.gz")) else ","
    df = pd.read_csv(args.dataframe, sep=sep)

    new_to_old = _load_new_to_old()
    comp_for = _components_from_config(args.config_file, new_to_old)

    datasets = sorted(df["Dataset"].unique())
    if args.datasets:
        requested = set(args.datasets)
        datasets = [d for d in datasets if d in requested]
    print(f"Datasets      : {len(datasets)}")
    print()

    # Build Scoreset objects up front (cheap, sequential) so failures/skips
    # are logged before any parallel dispatch, then fan the actual fitting
    # work out across (dataset, n_c) units -- this is the outer parallelism
    # level; see process_dataset_nc's docstring for the inner/outer split.
    #
    # Scoreset construction itself is also parallelized across datasets
    # (build_scoreset_for_dataset) -- it does real per-dataset work
    # (ClinVar/splicing-filter lookups), not "cheap" preprocessing, and was
    # previously a sequential for-loop that dominated wall time before any
    # fitting even started.
    # Pre-filter each dataset's (small) slice in the main process first --
    # cheap, vectorized, and avoids pickling the full ~48MB dataframe to
    # every worker just to re-filter it there.
    per_dataset_df = {d: df[df["Dataset"] == d].copy() for d in datasets}
    clinvar_releases = {d: ("2018" if _requires_2018(df, d) else args.clinvar_release) for d in datasets}
    print(f"Building {len(datasets)} Scoresets in parallel...", flush=True)
    scoreset_results = Parallel(n_jobs=-1, batch_size=1, backend="loky", verbose=10)(
        delayed(build_scoreset_for_dataset)(
            dataset, per_dataset_df[dataset], clinvar_releases[dataset], args.population_type
        )
        for dataset in datasets
    )

    work_items = []
    for dataset, scoreset, error in scoreset_results:
        if scoreset is None:
            print(f"{dataset}: SKIP ({error})")
            continue
        for n_c in comp_for(dataset):
            work_items.append((dataset, n_c, scoreset))

    n_jobs_inner = resolve_inner_jobs(args.n_jobs, len(work_items), args.n_jobs_inner)
    checkpoint_dir = results_dir / "checkpoints"
    n_cached = sum(
        1 for dataset, n_c, _ in work_items
        if checkpoint_path_for(checkpoint_dir, dataset, n_c).exists()
    )
    print(f"Dispatching {len(work_items)} (dataset, n_c) units: "
          f"n_jobs(outer)={args.n_jobs}  n_jobs_inner={n_jobs_inner}  "
          f"(total CPUs: {mp.cpu_count()})")
    print(f"Checkpoint dir: {checkpoint_dir}  ({n_cached} already cached from a prior run)\n",
          flush=True)

    unit_results = Parallel(n_jobs=args.n_jobs, batch_size=1, verbose=10)(
        delayed(process_dataset_nc)(
            dataset, n_c, scoreset, args.num_fits, args.seed, n_jobs_inner, args.plot,
            checkpoint_dir=checkpoint_dir,
        )
        for dataset, n_c, scoreset in work_items
    )

    train_lls_all = {}
    summary_rows = []
    sign_rows = []
    plot_records = []
    for res in unit_results:
        print("\n".join(res["log"]))
        print()
        if "train_lls" in res:
            train_lls_all[f"{res['dataset']}|{res['n_c']}c"] = res["train_lls"]
        summary_rows.extend(res.get("summary_rows", []))
        if res.get("sign_row"):
            sign_rows.append(res["sign_row"])
        if res.get("plot_record"):
            plot_records.append(res["plot_record"])

    with open(results_dir / "train_lls.json", "w") as f:
        json.dump(train_lls_all, f, indent=2)

    if summary_rows:
        with open(results_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if sign_rows:
        with open(results_dir / "sign_change_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sign_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sign_rows)

        n_changed = sum(1 for r in sign_rows if r["changed"])
        print("=" * 60)
        print(f"SIGN CHANGE SUMMARY: {n_changed}/{len(sign_rows)} best fits "
              f"({100 * n_changed / len(sign_rows):.0f}%) changed skew sign "
              f"from init to converged solution")
        print("=" * 60)

    if plot_records:
        from src.assay_calibration.plot_utils.utils import plot_initial_vs_final_skew_grid
        plot_path = results_dir / "initial_vs_final_skew.png"
        plot_initial_vs_final_skew_grid(plot_records, output_path=plot_path)
        print(f"Saved: {plot_path}")

    print(f"\nSaved: {results_dir / 'train_lls.json'}")
    print(f"Saved: {results_dir / 'summary.csv'}")
    print(f"Saved: {results_dir / 'sign_change_summary.csv'}")
    print(f"Saved: {results_dir / 'run.log'}")

    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
