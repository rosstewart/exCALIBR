#!/usr/bin/env python
"""Grow an existing converged fit by one new component targeted at a
specific sample class (default: P/LP), at real scale, across bootstraps and
restarts -- see src/assay_calibration/fit_utils/cfusn/fit.py's
`add_component_to_fit` for the underlying mechanism (freezes the source
fit's K components, adds one new component seeded from the target class's
worst-explained residual, re-estimates mixing weights for all classes via
standard EM).

Does NOT re-fit the source/base config from scratch -- reuses an existing
results directory's per-bootstrap `bootstrap_{i}_best_fits.pkl` files
(produced by the normal hpc/prepare.py + run_array_task.py path) as the
frozen base, for bootstraps `0..n_bootstraps-1`. This is deliberately NOT
wired through hpc/prepare.py's `_generate_bootstrap_fit_jobs`/
`run_array_task.py` job-array machinery (shared_data/minimal-job-stripping/
GreedyUnit) -- that machinery exists to search multiple from-scratch
restarts across arbitrary component counts, which isn't what growing one
component onto an already-fixed base needs. This script is a smaller,
self-contained equivalent: build ms once per gene, reconstruct each
bootstrap's exact resampled training data (replicating Fit.__call__'s own
onehot-resolution + pattern_stratified_bootstrap sequence bit-for-bit), grow
the component `--num-fits` times per bootstrap (parallelized via
ProcessPoolExecutor, matching run_local_array.sh's OMP_NUM_THREADS=1
single-thread-per-worker guard), pick the best restart by held-out LL
(`_weighted_val_ll` over pattern_stratified_bootstrap's own out-of-bag
indices -- the same quantity Fit's own best-of-N-restarts selection uses),
and write results into `bootstrap_{i}_best_fits.pkl` under a NEW config key
(default "3c_unc_plus1_plp", chosen to never collide with a real free
"4c_unc" fit) -- so `hpc/aggregate_results.py` works completely unmodified.

Usage
-----
    python hpc/run_grow_component_batch.py \\
        --gene-set predictor --genes BRCA1 BRCA2 F9 JAG1 MSH2 SCN5A TP53 TSC2 \\
        --source-dir /data/ross/assay_calibration/multivariate/jobs_all_100b_3f_092026 \\
        --source-config 3c_unc --target-sample-idx 0 \\
        --n-bootstraps 20 --num-fits 3 \\
        --output-dir /data/ross/assay_calibration/multivariate/jobs_predictor_plp_component_20b_3f_100126
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import pickle
import argparse
import concurrent.futures
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import (
    makeOneHot, pattern_stratified_bootstrap, derive_bootstrap_seed, derive_fit_seed,
    DEFAULT_MASTER_SEED, _weighted_val_ll,
)
from src.assay_calibration.fit_utils.cfusn.fit import add_component_to_fit
from src.assay_calibration.multivariate_data.predictors import (
    load_predictor_ms, predictor_dataset_label, DEFAULT_GENES as PREDICTOR_DEFAULT_GENES,
)

NEW_CONFIG_KEY = "3c_unc_plus1_plp"
PREDICTOR_DATA_DIR = "/data/ross/assay_calibration/predictor_calibrations/single_gene_calibration_data"

_AGGREGATED_JSON_CACHE = {}


def _load_source_entry(source_dir, dataset_name, bootstrap_idx, source_config):
    """Load bootstrap `i`'s `source_config` entry for `dataset_name` from
    `source_dir`, trying the per-bootstrap pkl first (the normal
    hpc/prepare.py output layout) and falling back to the aggregated
    `bootstrap_results.json.gz` (some source dirs -- e.g.
    jobs_combined_intersection_100b_3f_092026 -- have had their per-bootstrap
    pkls archived/removed, keeping only the aggregate). JSON-sourced
    component_params/weights are plain nested lists, not arrays; callers
    (add_component_to_fit's _ensure_cfusn_params/np.asarray calls) already
    coerce these, so no conversion needed here beyond what's done below for
    convenience."""
    pkl_path = Path(source_dir) / dataset_name / f"bootstrap_{bootstrap_idx}_best_fits.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)[source_config]

    cache_key = str(source_dir)
    if cache_key not in _AGGREGATED_JSON_CACHE:
        agg_path = Path(source_dir) / "bootstrap_results.json.gz"
        if not agg_path.exists():
            return None
        import gzip, json
        with gzip.open(agg_path, "rt", encoding="utf-8") as f:
            _AGGREGATED_JSON_CACHE[cache_key] = json.load(f)
    raw = _AGGREGATED_JSON_CACHE[cache_key]
    gene_entry = raw.get(dataset_name)
    if gene_entry is None:
        return None
    boot_entry = gene_entry.get(str(bootstrap_idx))
    if boot_entry is None:
        return None
    entry = boot_entry.get(source_config)
    if entry is None:
        return None
    # Convert component_params'/weights' nested lists to arrays so
    # add_component_to_fit's internal shape/dtype assumptions match the
    # pkl-sourced path exactly (np.asarray is idempotent on real arrays).
    fit = entry["fit"]
    fit["component_params"] = [
        (np.asarray(mu), np.asarray(Delta, dtype=float), np.asarray(Gamma, dtype=float))
        for mu, Delta, Gamma in fit["component_params"]
    ]
    fit["weights"] = np.asarray(fit["weights"], dtype=float)
    return entry


def _build_ms(gene, gene_set, require_both_modalities=False):
    if gene_set == "predictor":
        return load_predictor_ms(gene, PREDICTOR_DATA_DIR, standardize=False)
    if gene_set == "combined":
        from src.assay_calibration.multivariate_data.combined import (
            build_functional_scoresets, build_combined_multiscoreset,
            get_functionally_assayed_protein_variants, DEFAULT_INTEGRATED_DATAFRAME,
        )
        from src.assay_calibration.multivariate_data.common import resolve_clinvar_release
        import pandas as pd
        df = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
        datasets = sorted(df[df["Gene"] == gene]["Dataset"].unique())
        if not datasets:
            raise ValueError(f"no functional datasets for {gene}")
        functional_scoresets = build_functional_scoresets(
            df, gene, datasets, clinvar_release=resolve_clinvar_release(gene))
        functionally_assayed = get_functionally_assayed_protein_variants(df, gene, datasets)
        ms = build_combined_multiscoreset(
            gene, functional_scoresets, datasets, PREDICTOR_DATA_DIR,
            functionally_assayed_variants=functionally_assayed,
            # Must match whichever source fit's own ms was built with --
            # jobs_all_100b_3f_092026 (union, require_both_modalities=False)
            # vs jobs_combined_intersection_100b_3f_092026 (intersection,
            # require_both_modalities=True) -- growing a component onto a
            # source fit with a DIFFERENT variant population than it was
            # trained on would evaluate the frozen components against data
            # they never saw.
            require_both_modalities=require_both_modalities,
        )
        if ms is None:
            raise ValueError(f"no predictor CSVs for {gene} under {PREDICTOR_DATA_DIR}")
        return ms
    raise ValueError(f"unknown gene_set={gene_set!r}")


def _dataset_name(gene, gene_set):
    if gene_set == "predictor":
        return predictor_dataset_label(gene)
    if gene_set == "combined":
        return f"{gene}_combined_mv"
    raise ValueError(gene_set)


def _reconstruct_bootstrap_split(ms, bootstrap_idx, master_seed=DEFAULT_MASTER_SEED):
    """Bit-for-bit replica of Fit.__call__'s own sequence (fit.py:~872-964):
    onehot-resolve ambiguous multi-labeled rows, THEN pattern-stratified
    bootstrap -- both keyed off the same composition_seed, in this order,
    as two independent RandomState draws."""
    composition_seed = derive_bootstrap_seed(master_seed, bootstrap_idx)
    onehot_rng = np.random.RandomState(composition_seed)
    sample_assignments = makeOneHot(ms.sample_assignments, rng=onehot_rng)
    observations = ms.scores
    include = sample_assignments.any(axis=1) & ~np.all(np.isnan(observations), axis=1)
    observations = observations[include]
    sample_assignments = sample_assignments[include]
    train_idx, val_idx = pattern_stratified_bootstrap(observations, sample_assignments, composition_seed)
    return observations, sample_assignments, train_idx, val_idx


def _grow_one_restart(args):
    (train_obs, train_sa, val_obs, val_sa, source_fit, target_sample_idx,
     latent_q, fit_seed) = args
    try:
        grown = add_component_to_fit(
            train_obs, train_sa, source_fit, target_sample_idx=target_sample_idx,
            constrained=False, multivariate=True, latent_q=latent_q,
            fit_seed=fit_seed, verbose=False, max_em_iters=2000,
        )
    except Exception as e:
        return None, None, str(e)
    if len(val_obs) == 0:
        val_ll = float("nan")
    else:
        val_ll = _weighted_val_ll(val_obs, val_sa, grown["component_params"], grown["weights"],
                                   mv=True, fit_kwargs={})
    return grown, val_ll, None


def run_gene(gene, gene_set, source_dir, source_config, target_sample_idx,
             n_bootstraps, num_fits, latent_q, output_dir, max_workers,
             require_both_modalities=False):
    dataset_name = _dataset_name(gene, gene_set)
    print(f"\n=== {gene_set}/{gene} ({dataset_name}) ===", flush=True)
    ms = _build_ms(gene, gene_set, require_both_modalities=require_both_modalities)

    out_gene_dir = Path(output_dir) / dataset_name
    out_gene_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_bootstraps):
        source_entry = _load_source_entry(source_dir, dataset_name, i, source_config)
        if source_entry is None:
            print(f"  bootstrap {i}: SKIP, no source fit found under {source_dir}")
            continue
        source_fit = source_entry["fit"]

        observations, sample_assignments, train_idx, val_idx = _reconstruct_bootstrap_split(ms, i)
        train_obs, train_sa = observations[train_idx], sample_assignments[train_idx]
        val_obs, val_sa = observations[val_idx], sample_assignments[val_idx]

        grown_num_components = len(source_fit["component_params"]) + 1
        restart_args = []
        for r in range(num_fits):
            fit_seed = derive_fit_seed(DEFAULT_MASTER_SEED, i, grown_num_components, r)
            restart_args.append((train_obs, train_sa, val_obs, val_sa, source_fit,
                                  target_sample_idx, latent_q, fit_seed))

        results = []
        if max_workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_grow_one_restart, restart_args))
        else:
            results = [_grow_one_restart(a) for a in restart_args]

        best_grown, best_val_ll, best_r = None, -np.inf, None
        for r, (grown, val_ll, err) in enumerate(results):
            if err is not None:
                print(f"  bootstrap {i} restart {r}: FAILED ({err})")
                continue
            if val_ll is not None and np.isfinite(val_ll) and val_ll > best_val_ll:
                best_val_ll, best_grown, best_r = val_ll, grown, r
        if best_grown is None:
            print(f"  bootstrap {i}: SKIP, no valid restart")
            continue

        entry = {
            "dataset_name": dataset_name,
            "bootstrap_seed": i,
            "num_components": len(best_grown["component_params"]),
            "fit_idx": best_r,
            "fit": best_grown,
            "val_ll": float(best_val_ll),
            "calibrated_dims": source_entry.get("calibrated_dims"),
            "n_cap_hits": 0,
        }
        out_pkl = out_gene_dir / f"bootstrap_{i}_best_fits.pkl"
        existing = {}
        if out_pkl.exists():
            with open(out_pkl, "rb") as f:
                existing = pickle.load(f)
        existing[NEW_CONFIG_KEY] = entry
        with open(out_pkl, "wb") as f:
            pickle.dump(existing, f)
        print(f"  bootstrap {i}: best restart={best_r} val_ll={best_val_ll:.3f} -> {out_pkl}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gene-set", required=True, choices=["predictor", "combined"])
    ap.add_argument("--genes", nargs="+", default=list(PREDICTOR_DEFAULT_GENES))
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--source-config", default="3c_unc")
    ap.add_argument("--target-sample-idx", type=int, default=0)
    ap.add_argument("--n-bootstraps", type=int, default=20)
    ap.add_argument("--num-fits", type=int, default=3)
    ap.add_argument("--latent-q", type=int, default=2)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--require-both-modalities", action="store_true",
                     help="--gene-set combined only: must match whether --source-dir's "
                          "own fit was trained with the intersection-filtered ms "
                          "(jobs_combined_intersection_*) or the union ms "
                          "(jobs_all_100b_3f_092026, default behavior, flag unset).")
    args = ap.parse_args()

    for gene in args.genes:
        try:
            run_gene(gene, args.gene_set, args.source_dir, args.source_config,
                      args.target_sample_idx, args.n_bootstraps, args.num_fits,
                      args.latent_q, args.output_dir, args.max_workers,
                      require_both_modalities=args.require_both_modalities)
        except Exception as e:
            import traceback
            print(f"SKIP {gene}: {e}\n{traceback.format_exc()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
