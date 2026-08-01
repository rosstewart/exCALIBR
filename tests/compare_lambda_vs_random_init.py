#!/usr/bin/env python3
"""
Does the current lambdaIndex enumerated-sign init (fit.py:663,
cfusn/initializations.py:27-38) actually beat random-sign init, especially
at low restart counts, on real IGVF pillar-project data?

Both arms run through the identical CPU code path
(Fit.generate_fit_jobs -> tryToFit / Fit.execute_fit_job) on the same
whole-dataset (bootstrap=False) job list -- the only difference is whether
each restart's kwargs still carries the "lambdaIndex" key
(_lambda_signs falls back to a random +/-1 draw per component when it's
absent, cfusn/initializations.py:38).

Datasets are picked by ranking the 89-dataset precomputed bootstrap fits by
the MEDIAN (across bootstrap seeds) of each seed's max |skew a-param| --
these are the datasets where skew actually matters, so init method should
matter most there.

Usage:
    python tests/compare_lambda_vs_random_init.py /tmp/lambda_vs_random
    python tests/compare_lambda_vs_random_init.py /tmp/lambda_vs_random --top-k 5
"""
import argparse
import csv
import gzip
import json
import multiprocessing as mp
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR_DEFAULT = Path(__file__).resolve().parent / "compare_lambda_vs_random_init"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from joblib import Parallel, delayed

from tests.benchmark_num_fits_dataframe import (
    build_unbootstrapped_jobs, run_restarts, sign_change, Tee, resolve_inner_jobs,
    RESTART_GRID, _load_new_to_old, _DEFAULT_DATAFRAME, _DEFAULT_CONFIG,
)
from slurm.prepare import _components_from_config, _requires_2018
from src.assay_calibration.data_utils.dataset import Scoreset

_DEFAULT_BOOTSTRAP_RESULTS = "/data/ross/assay_calibration/explorer_jobs_pp_merged_89datasets_bootstrap_results.json.gz"


# ---------------------------------------------------------------------------

def rank_datasets_by_median_skew(bootstrap_results_path, comp_for, datasets=None):
    """Median (across bootstrap seeds) of each seed's max|a| skew param, per
    dataset, using that dataset's configured n_c. Descending order."""
    with gzip.open(bootstrap_results_path, "rt", encoding="utf-8") as f:
        all_results = json.load(f)

    scores = {}
    for dataset, seeds in all_results.items():
        if datasets is not None and dataset not in datasets:
            continue
        try:
            n_cs = comp_for(dataset)
        except Exception:
            n_cs = [2, 3]

        n_c_key = None
        for nc in n_cs:
            key = f"{nc}c"
            if any(isinstance(sr, dict) and sr.get(key) for sr in seeds.values()):
                n_c_key = key
                break
        if n_c_key is None:
            continue

        per_seed_max_skew = []
        for seed_results in seeds.values():
            if not isinstance(seed_results, dict):
                continue
            fit_entry = seed_results.get(n_c_key)
            if not fit_entry:
                continue
            params = fit_entry.get("fit", {}).get("component_params")
            if not params:
                continue
            per_seed_max_skew.append(max(abs(a) for a, _, _ in params))

        if per_seed_max_skew:
            scores[dataset] = (float(np.median(per_seed_max_skew)), n_c_key)

    return sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)


def _strip_lambda(jobs):
    """Copy jobs with "lambdaIndex" removed from kwargs (random-sign fallback)."""
    stripped = []
    for job in jobs:
        j = dict(job)
        kwargs = dict(job["kwargs"])
        kwargs.pop("lambdaIndex", None)
        j["kwargs"] = kwargs
        stripped.append(j)
    return stripped


def process_dataset_arms(dataset, n_c, skew_score, scoreset, num_fits, seed, n_jobs_inner, want_plot):
    """Both arms (lambda, random) for one dataset -- top-level (not a
    closure) so joblib/loky can dispatch it to a worker process. This is the
    unit parallelized *across* the top-K skewed datasets (outer); n_jobs_inner
    is the joblib pool used for run_restarts within one arm/dataset (inner) --
    same outer/inner split as tests/benchmark_num_fits_dataframe.py's
    process_dataset_nc.
    """
    log = [f"{dataset} ({n_c}c, median skew={skew_score:.4f})"]
    out = {"dataset": dataset, "n_c": n_c, "log": log, "csv_rows": [], "plot_records": {}}

    _, lambda_jobs = build_unbootstrapped_jobs(scoreset, dataset, n_c, num_fits, master_seed=seed)
    if not lambda_jobs:
        log.append("  SKIP: no jobs generated")
        return out
    random_jobs = _strip_lambda(lambda_jobs)

    arms = {"lambda": (lambda_jobs, run_restarts(lambda_jobs, n_jobs=n_jobs_inner)),
            "random": (random_jobs, run_restarts(random_jobs, n_jobs=n_jobs_inner))}

    for arm_name, (jobs, results) in arms.items():
        train_lls = np.array([r["train_ll"] for r in results])
        for N in RESTART_GRID:
            if N > len(train_lls):
                continue
            # First-N-restarts prefix, not a random subsample: for the
            # lambda arm, restart order *is* the enumerated-sign coverage
            # order, so "first N restarts" is what a real N-fit budget would
            # actually run; using the same prefix for the random arm keeps
            # the comparison apples-to-apples.
            sub = train_lls[:N]
            valid = sub[np.isfinite(sub)]
            best = float(valid.max()) if len(valid) else -np.inf
            n_changed = 0
            for r in results[:N]:
                fit = r.get("fit", {})
                if not fit.get("component_params") or not fit.get("initial_params"):
                    continue
                _, _, changed = sign_change(fit["initial_params"], fit["component_params"])
                n_changed += int(changed)
            log.append(f"    [{arm_name}] N={N:3d}  best_train_ll={best:.4f}  "
                       f"sign_changed={n_changed}/{min(N, len(results))}")
            out["csv_rows"].append({
                "dataset": dataset, "n_c": n_c, "arm": arm_name, "num_fits": N,
                "best_train_ll": round(best, 6),
                "n_valid": int(len(valid)), "n_failed": int(N - len(valid)),
                "n_sign_changed": n_changed,
            })

        if want_plot and len(train_lls) and np.isfinite(train_lls).any():
            best_idx = int(np.nanargmax(train_lls))
            best_fit = results[best_idx].get("fit", {})
            if best_fit.get("component_params") and best_fit.get("initial_params"):
                out["plot_records"][arm_name] = {
                    "dataset_name": f"{dataset} ({arm_name})",
                    "observations": jobs[best_idx]["train_observations"],
                    "sample_assignments": jobs[best_idx]["train_sample_assignments"],
                    "initial_params": best_fit["initial_params"],
                    "component_params": best_fit["component_params"],
                    "final_weights": (np.mean(best_fit["weights"], axis=0)
                                     if best_fit.get("weights") is not None else None),
                }
    return out


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Unused placeholder; results always go to --results-dir.")
    parser.add_argument("--dataframe", default=_DEFAULT_DATAFRAME)
    parser.add_argument("--config-file", default=_DEFAULT_CONFIG)
    parser.add_argument("--bootstrap-results", default=_DEFAULT_BOOTSTRAP_RESULTS)
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of most-skewed datasets to compare (default: 20)")
    parser.add_argument("--num-fits", type=int, default=max(RESTART_GRID),
                        help="Max restarts to run per arm per dataset (default: 100)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Outer parallelism: datasets run concurrently (default: -1 = all CPUs)")
    parser.add_argument("--n-jobs-inner", type=int, default=None,
                        help="Inner parallelism: restarts within one dataset/arm. Default: "
                             "auto -- split total CPUs between outer and inner so the "
                             "product saturates the machine (matters when --top-k is "
                             "smaller than the core count). Set explicitly to override.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clinvar-release", default="2025")
    parser.add_argument("--population-type", default=None)
    parser.add_argument("--results-dir", default=str(_RESULTS_DIR_DEFAULT))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tee = Tee(results_dir / "run.log")
    sys.stdout = tee

    sep = "\t" if args.dataframe.endswith((".tsv", ".tsv.gz")) else ","
    df = pd.read_csv(args.dataframe, sep=sep)
    new_to_old = _load_new_to_old()
    comp_for = _components_from_config(args.config_file, new_to_old)

    print(f"Ranking datasets by median cross-bootstrap skew from {args.bootstrap_results} ...")
    ranked = rank_datasets_by_median_skew(args.bootstrap_results, comp_for,
                                          datasets=set(df["Dataset"].unique()))
    top = ranked[:args.top_k]
    print(f"Top {len(top)} most-skewed datasets:")
    for dataset, (score, n_c_key) in top:
        print(f"  {score:.4f}  {n_c_key}  {dataset}")
    print()

    # Build Scoresets up front (cheap, sequential) so skips are logged
    # before parallel dispatch, then fan out across datasets (outer
    # parallelism) -- see process_dataset_arms' docstring for the
    # outer/inner split.
    work_items = []
    for dataset, (skew_score, n_c_key) in top:
        n_c = int(n_c_key[0])
        df_ds = df[df["Dataset"] == dataset].copy()
        clinvar_release = "2018" if _requires_2018(df, dataset) else args.clinvar_release
        try:
            kw = dict(clinvar_release=clinvar_release, min_clinvar_star=1)
            if args.population_type:
                kw["population_type"] = args.population_type
            scoreset = Scoreset(df_ds, **kw)
        except (ValueError, KeyError) as e:
            print(f"{dataset}: SKIP ({e})")
            continue
        work_items.append((dataset, n_c, skew_score, scoreset))

    n_jobs_inner = resolve_inner_jobs(args.n_jobs, len(work_items), args.n_jobs_inner)
    print(f"Dispatching {len(work_items)} datasets (2 arms each, run sequentially per "
          f"dataset): n_jobs(outer)={args.n_jobs}  n_jobs_inner={n_jobs_inner}  "
          f"(total CPUs: {mp.cpu_count()})\n", flush=True)

    unit_results = Parallel(n_jobs=args.n_jobs, batch_size=1, verbose=10)(
        delayed(process_dataset_arms)(
            dataset, n_c, skew_score, scoreset, args.num_fits, args.seed,
            n_jobs_inner, args.plot,
        )
        for dataset, n_c, skew_score, scoreset in work_items
    )

    csv_rows = []
    plot_records = defaultdict(list)  # arm -> records
    for res in unit_results:
        print("\n".join(res["log"]))
        print()
        csv_rows.extend(res["csv_rows"])
        for arm_name, rec in res["plot_records"].items():
            plot_records[arm_name].append(rec)

    if csv_rows:
        with open(results_dir / "lambda_vs_random.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        print("=" * 70)
        print("SUMMARY — mean Δ(lambda − random) best_train_ll, by N (across all datasets)")
        print("=" * 70)
        by_n = defaultdict(lambda: {"lambda": [], "random": []})
        for row in csv_rows:
            by_n[row["num_fits"]][row["arm"]].append(row["best_train_ll"])
        for N in RESTART_GRID:
            if N not in by_n or not by_n[N]["lambda"] or not by_n[N]["random"]:
                continue
            lam = np.array(by_n[N]["lambda"])
            rnd = np.array(by_n[N]["random"])
            delta = lam - rnd
            print(f"  N={N:3d}:  Δ={np.mean(delta):+.4f}  "
                  f"(lambda mean={lam.mean():.4f}, random mean={rnd.mean():.4f})")

    if args.plot:
        from src.assay_calibration.plot_utils.utils import plot_initial_vs_final_skew_grid
        for arm_name, records in plot_records.items():
            if not records:
                continue
            plot_path = results_dir / f"initial_vs_final_skew_{arm_name}.png"
            plot_initial_vs_final_skew_grid(records, output_path=plot_path)
            print(f"Saved: {plot_path}")

    print(f"\nSaved: {results_dir / 'lambda_vs_random.csv'}")
    print(f"Saved: {results_dir / 'run.log'}")

    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
