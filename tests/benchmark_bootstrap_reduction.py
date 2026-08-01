#!/usr/bin/env python3
"""
How much does reducing the number of bootstrap fits degrade the final
calibration -- LR+ at the 5th/95th percentile, and downstream ACMG point
assignments?

Reuses the real pipeline's own calibration logic (run_igvf_batch.py's
_run_one_combo -> process_component_fits) per bootstrap-count level, rather
than reimplementing calibration math -- this script is just an orchestrator
+ comparator around that existing machinery, following the same
reference-vs-condition comparison shape as analysis/robustness.py (which
compares already-run downsample/discordance pipeline outputs; here the
"condition" is a bootstrap-count subsample of one precomputed-fits file
instead of a separately re-run pipeline job).

Usage:
    python tests/benchmark_bootstrap_reduction.py /tmp/bootstrap_reduction
    python tests/benchmark_bootstrap_reduction.py /tmp/bootstrap_reduction \\
        --datasets BRCA1_Findlay_2018 --bootstrap-counts 1000,500,100,20
"""
import argparse
import csv
import json
import multiprocessing as mp
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import Namespace
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR_DEFAULT = "/data/ross/assay_calibration/hyperparam_sweep_oldclinvar/bootstrap_reduction"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from joblib import Parallel, delayed

from run_igvf_batch import _run_one_combo, parse_dataset_config, _compute_all_configs_metrics
from analysis.legacy_fits import _load_calibration_and_lr
from analysis.discovery import recompute_points_from_calibration
from src.assay_calibration.pipeline.config import PipelineConfig
from src.assay_calibration.pipeline.utils import load_dataset_from_df

_DEFAULT_PRECOMPUTED_FITS = "/data/ross/assay_calibration/explorer_jobs_pp_merged_89datasets_bootstrap_results.json.gz"
_DEFAULT_DATAFRAME = "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz"
_DEFAULT_CONFIG = str(_REPO_ROOT / "src" / "igvf_configs" / "dataset_configs_jul_2026.json")
_DEFAULT_LEVELS = [1000, 500, 250, 100, 50, 20]


def _batch_args(dataframe_path):
    """Minimal Namespace with the fields _run_one_combo reads off `args`
    (run_igvf_batch.py:441-464), matching that CLI's own defaults."""
    return Namespace(
        dataset=dataframe_path,
        manual_prior=None,
        min_clinvar_star=1,
        population_type="gnomAD",
        synonymous_exclusive=False,
        acmg_mapping_method="tavtigian",
        acmg_bayes_targets=None,
        acmg_bayes_floor_at_neutral=False,
        seed=None,
    )


# Precomputed-fits keys that don't match any row in the current --dataframe
# at all, even after stripping "_clinvar_2018" -- confirmed NOT covered by
# /data/ross/assay_calibration/dataframe/new_dataset_names.csv (that file is
# stale; it predates these particular renames). Found by inspecting the
# dataframe's actual unique "Dataset" values for the closest match. Add more
# entries here if new mismatches turn up -- Pass 1 will report a plain SKIP
# for anything not covered by this table, so new ones are easy to spot.
_KNOWN_NAME_ALIASES = {
    "CHEK2_McCarthy_Leo_2024": "CHEK2_McCarthy-Leo_2024",
    "LDLR_Tabet_2025_LDL_uptake": "LDLR_Tabet_2025_uptake",
    "LDLR_Tabet_2025_LDLR_cell_surface_abundance": "LDLR_Tabet_2025_abundance",
}


def build_dataset_df(dataset_name, df):
    """Filter the raw dataframe for one dataset and relabel its "Dataset"
    column to the effective (possibly "_clinvar_2018"-suffixed) name.

    The precomputed-fits keys (and this script's dataset_name) carry the
    "_clinvar_2018" suffix for BRCA1/MSH2/PTEN/TP53 (matching
    run_igvf_batch.py's effective_dataset_name/bootstrap_key), but the raw
    CSV's own "Dataset" column never does -- filtering by dataset_name
    directly (without stripping the suffix first) silently returns an empty
    frame for those 4 genes, which Scoreset then rejects with "dataframe
    must contain only one dataset" (0 unique values != 1). Strip the suffix
    for the *filter*, but still relabel the result to the effective
    (suffixed) name, matching run_igvf_batch.py:117-126's csv_name/
    effective_name split. A handful of keys need an explicit alias on top of
    that (_KNOWN_NAME_ALIASES) -- their precomputed-fits name doesn't match
    the current dataframe at all, not just by the clinvar_2018 suffix.
    """
    csv_name = dataset_name.replace("_clinvar_2018", "")
    csv_name = _KNOWN_NAME_ALIASES.get(csv_name, csv_name)
    dataset_df = df[df["Dataset"] == csv_name].copy()
    dataset_df["Dataset"] = dataset_name
    return dataset_df


def run_one_level(dataset_name, N, dataset_df, fits, seeds, n_c_str, benign_method,
                  clinvar_release, output_dir, args):
    """One (dataset, bootstrap-count level) unit of work -- top-level (not a
    closure) so joblib/loky can pickle+dispatch it to a worker process.

    n_bootstrap_jobs=1 (forced below, at the call site) is deliberate: this
    function itself is the thing parallelized across *all* (dataset, level)
    pairs at once (89 datasets x ~6 levels = ~500+ units, far more than any
    core count, so outer parallelism alone saturates the machine) --
    letting process_component_fits ALSO spawn its own internal joblib pool
    per unit would nest Parallel calls, which joblib silently downgrades to
    ThreadingBackend (GIL-limited) unless every nested call site explicitly
    forces backend="loky" (see tests/benchmark_num_fits_dataframe.py's
    run_restarts docstring for the empirical confirmation of this). Rather
    than patch process_component_fits's several internal Parallel() call
    sites (src/assay_calibration/pipeline/visualize.py) to force loky, it's
    simpler and equally fast in aggregate to keep each unit single-threaded
    and rely entirely on outer parallelism.

    Resumable by construction: _run_one_combo already writes this unit's
    calibration.json/lr_values.json.gz to disk as its real output (not just
    an in-memory return value), so if those files already exist on entry
    (from a prior run that got OOM-killed partway through -- found the hard
    way running this at 89-dataset scale on a shared machine), skip
    recomputing this unit entirely and just point at the existing files.

    Any exception from _run_one_combo is caught and turned into a (None)
    skip rather than propagating: joblib's default Parallel() is fail-fast
    -- one dataset/level hitting an unrelated pipeline edge case (seen in
    practice: an IndexError deep in point_ranges.get_fit_prior for a
    dataset missing an expected gnomAD sample) previously aborted the
    entire ~500-unit batch, cancelling every other still-pending unit, not
    just the one that failed.
    """
    level_dir = Path(output_dir) / dataset_name / f"level_{N}"
    comp_key = f"{n_c_str}_{benign_method}"
    calib_path = level_dir / f"{dataset_name}_{comp_key}_calibration.json"
    lr_path = level_dir / f"{dataset_name}_{comp_key}_lr_values.json.gz"
    if calib_path.exists() and lr_path.exists():
        return dataset_name, N, calib_path

    level_dir.mkdir(parents=True, exist_ok=True)
    try:
        comp_key, _ = _run_one_combo(
            n_c_str, benign_method, fits[:N], seeds[:N], dataset_df, dataset_name,
            clinvar_release, str(level_dir), args, n_bootstrap_jobs=1,
        )
    except IndexError as e:
        # Root-caused (PTEN_Mighell_2018_clinvar_2018): the precomputed fit's
        # "weights" array is shaped for however many samples the dataset had
        # when explorer_jobs_pp_merged_89datasets_bootstrap_results.json.gz
        # was generated, but this script builds a FRESH Scoreset from the
        # *current* --dataframe and derives pathogenic/benign/gnomad/
        # synonymous_idx from that current sample list (visualize.py:229-296)
        # -- if the dataset's sample composition changed since the fits were
        # precomputed (e.g. gained/lost a sample class), an index that's
        # valid for the current Scoreset can be out of range for the old
        # fit's weights, surfacing deep inside point_ranges.get_fit_prior as
        # a bare IndexError. Not fixable here without either regenerating
        # this dataset's precomputed fits against the current dataframe or
        # reconstructing its old sample composition -- report the likely
        # cause plainly and skip, rather than a generic exception message.
        n_weight_samples = None
        try:
            n_weight_samples = len(fits[0]["fit"]["weights"])
        except Exception:
            pass
        print(f"{dataset_name} level N={N}: FAILED (IndexError: {e}) -- likely a sample-composition "
              f"mismatch between the precomputed fit (weights for {n_weight_samples} samples) and "
              f"the current --dataframe's sample list for this dataset")
        return dataset_name, N, None
    except Exception as e:
        print(f"{dataset_name} level N={N}: FAILED ({type(e).__name__}: {e})")
        return dataset_name, N, None
    if comp_key is None:
        return dataset_name, N, None
    return dataset_name, N, level_dir / f"{dataset_name}_{comp_key}_calibration.json"


def compare_one_dataset(dataset_name, meta, calib_paths, args, requested_baseline_n):
    """One dataset's Pass-3 comparison -- top-level (not a closure) so
    joblib/loky can dispatch it to a worker process. Builds this dataset's
    Scoreset (real per-dataset cost: ClinVar/splicing-filter lookups) and
    calls compare_levels; this is the unit parallelized across datasets in
    Pass 3, so ~90 Scoresets get built concurrently instead of one at a time
    in the main process (the exact bottleneck already found and fixed for
    tests/benchmark_num_fits_dataframe.py's Scoreset-building pass).
    """
    print(f"{dataset_name}")
    cfg = PipelineConfig(
        dataset_csv=args.dataframe, dataset_name=dataset_name,
        output_dir=str(Path(args.output_dir) / dataset_name),
        components=[int(meta["n_c_str"][0])],
        benign_method=meta["benign_method"], clinvar_release=meta["clinvar_release"],
        min_clinvar_star=1, population_type="gnomAD",
    )
    scoreset = load_dataset_from_df(meta["dataset_df"], cfg)
    rows = compare_levels(dataset_name, scoreset, calib_paths, meta["baseline_n"], requested_baseline_n)
    return dataset_name, rows


def compare_levels(dataset_name, scoreset, calib_paths, baseline_n, requested_baseline_n):
    """Diff every level's LR+ p5/p95 curve and point assignments against
    baseline_n (this dataset's own baseline -- normally requested_baseline_n,
    the largest --bootstrap-counts value, but may be a smaller proxy for
    datasets with fewer than requested_baseline_n valid bootstraps)."""
    if baseline_n not in calib_paths:
        print(f"    SKIP compare: baseline N={baseline_n} missing")
        return []

    lr_dir_for = lambda p: p.parent
    def _lr_path(calib_path):
        # calibration.json and lr_values.json.gz are siblings with the same stem
        return calib_path.with_name(calib_path.name.replace("_calibration.json", "_lr_values.json.gz"))

    baseline_cal = _load_calibration_and_lr(calib_paths[baseline_n], _lr_path(calib_paths[baseline_n]))
    baseline_score_range = np.asarray(baseline_cal["score_range"])
    baseline_pct = baseline_cal["log_lr_pct"]  # (3, N): p5, p50, p95

    variant_df = pd.DataFrame({"score": np.asarray(scoreset.scores, dtype=float)})
    baseline_points = recompute_points_from_calibration(variant_df, calib_paths[baseline_n])["standard_points"]

    baseline_metrics = _compute_all_configs_metrics(scoreset, {"baseline": calib_paths[baseline_n]}).get("baseline", {})

    rows = []
    for N, calib_path in sorted(calib_paths.items(), reverse=True):
        cal = _load_calibration_and_lr(calib_path, _lr_path(calib_path))
        score_range = np.asarray(cal["score_range"])
        if len(score_range) == len(baseline_score_range) and np.allclose(score_range, baseline_score_range):
            pct = cal["log_lr_pct"]
            p5_delta = float(np.nanmax(np.abs(pct[0] - baseline_pct[0])))
            p95_delta = float(np.nanmax(np.abs(pct[2] - baseline_pct[2])))
        else:
            p5_delta = p95_delta = float("nan")

        points = recompute_points_from_calibration(variant_df, calib_path)["standard_points"]
        pct_changed = float(np.mean(points.values != baseline_points.values)) * 100

        metrics = _compute_all_configs_metrics(scoreset, {"level": calib_path}).get("level", {})
        mcc_delta = metrics.get("MCC", np.nan) - baseline_metrics.get("MCC", np.nan)
        acc_delta = metrics.get("accuracy", np.nan) - baseline_metrics.get("accuracy", np.nan)
        cov_delta = metrics.get("coverage", np.nan) - baseline_metrics.get("coverage", np.nan)

        rows.append({
            "dataset": dataset_name, "bootstrap_count": N,
            "baseline_n": baseline_n, "baseline_is_proxy": baseline_n != requested_baseline_n,
            "lr_plus_p5_max_delta": round(p5_delta, 6) if np.isfinite(p5_delta) else None,
            "lr_plus_p95_max_delta": round(p95_delta, 6) if np.isfinite(p95_delta) else None,
            "pct_points_changed": round(pct_changed, 3),
            "mcc_delta": round(mcc_delta, 6) if pd.notna(mcc_delta) else None,
            "accuracy_delta": round(acc_delta, 6) if pd.notna(acc_delta) else None,
            "coverage_delta": round(cov_delta, 6) if pd.notna(cov_delta) else None,
        })
        print(f"    N={N:5d}  Δp5={p5_delta:+.4f}  Δp95={p95_delta:+.4f}  "
              f"pts_changed={pct_changed:.2f}%  ΔMCC={mcc_delta:+.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precomputed-fits", default=_DEFAULT_PRECOMPUTED_FITS)
    parser.add_argument("--dataframe", default=_DEFAULT_DATAFRAME)
    parser.add_argument("--config-file", default=_DEFAULT_CONFIG)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--bootstrap-counts", default=",".join(map(str, _DEFAULT_LEVELS)))
    parser.add_argument("--output-dir", default=str(_RESULTS_DIR_DEFAULT))
    args = parser.parse_args()

    levels = sorted({int(x) for x in args.bootstrap_counts.split(",")}, reverse=True)
    baseline_n = levels[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Bootstrap-count levels: {levels} (baseline: {baseline_n})")

    import gzip
    with gzip.open(args.precomputed_fits, "rt", encoding="utf-8") as f:
        all_bootstrap_results = json.load(f)

    with open(args.config_file) as f:
        dataset_configs = json.load(f)

    sep = "\t" if args.dataframe.endswith((".tsv", ".tsv.gz")) else ","
    df = pd.read_csv(args.dataframe, sep=sep)

    batch_args = _batch_args(args.dataframe)

    datasets = sorted(all_bootstrap_results.keys())
    if args.datasets:
        requested = set(args.datasets)
        datasets = [d for d in datasets if d in requested]

    # Pass 1 (sequential, cheap): resolve each dataset's n_c/benign_method/
    # fits/seeds/clinvar_release and pre-filter its (small) dataframe slice
    # once, skipping datasets with insufficient bootstraps up front.
    dataset_meta = {}
    for dataset_name in datasets:
        config_entry = dataset_configs.get(dataset_name, ["3c", "avg"])
        n_c, benign_method, overrides = parse_dataset_config(config_entry)
        if n_c in ("", "all", None):
            n_c = "3c"
        n_c_str = n_c if n_c.endswith("c") else f"{n_c}c"

        boot = all_bootstrap_results[dataset_name]
        seeds_sorted = sorted(boot, key=lambda k: int(k))
        fits, seeds = [], []
        for seed in seeds_sorted:
            entry = boot[seed]
            if isinstance(entry, dict) and entry.get(n_c_str) is not None:
                fits.append(entry[n_c_str])
                seeds.append(int(seed))
        if len(fits) == 0:
            print(f"{dataset_name}: SKIP (0 bootstraps for {n_c_str} -- unfittable regardless of level)")
            continue
        # Datasets with fewer than `baseline_n` valid bootstraps still get
        # compared: use however many they actually have as that dataset's
        # own "baseline" stand-in (rather than discarding an otherwise-fine
        # dataset just for being a handful short of the nominal max), and
        # cap every other requested level at that same count. Reported
        # explicitly below and in dataset_levels/effective_baseline so the
        # substitution is visible in the summary, not silent.
        dataset_levels = sorted({min(N, len(fits)) for N in levels}, reverse=True)
        effective_baseline = dataset_levels[0]
        if effective_baseline < baseline_n:
            print(f"{dataset_name}: only {len(fits)} bootstraps available for {n_c_str} "
                  f"(need {baseline_n}) -- using {effective_baseline} as this dataset's "
                  f"baseline proxy instead of skipping")

        clinvar_release = "2018" if dataset_name.endswith("_clinvar_2018") else "2025"
        dataset_df = build_dataset_df(dataset_name, df)
        if len(dataset_df) == 0:
            # The precomputed-fits key doesn't match any row in the current
            # raw dataframe even after stripping "_clinvar_2018" -- not
            # covered by new_dataset_names.csv's old/new rename map either
            # (confirmed for e.g. "CHEK2_McCarthy_Leo_2024" vs. the actual
            # "CHEK2_McCarthy-Leo_2024", "LDLR_Tabet_2025_LDL_uptake" vs.
            # "LDLR_Tabet_2025_uptake"). Skip gracefully here instead of
            # letting Scoreset's "dataframe must contain only one dataset"
            # crash deep inside a worker -- with joblib's default
            # fail-fast Parallel(), one such crash previously aborted the
            # entire ~500-unit batch, cancelling everything still pending,
            # not just this one dataset.
            print(f"{dataset_name}: SKIP (no matching rows in --dataframe; "
                  f"stale/renamed key not covered by new_dataset_names.csv)")
            continue
        dataset_meta[dataset_name] = dict(
            n_c_str=n_c_str, benign_method=benign_method,
            fits=fits, seeds=np.array(seeds), clinvar_release=clinvar_release,
            dataset_df=dataset_df, levels=dataset_levels, baseline_n=effective_baseline,
        )
        print(f"{dataset_name}: {n_c_str}, {benign_method}, {len(fits)} bootstraps available")

    # Pass 2: fan every (dataset, level) pair out across all cores at once --
    # see run_one_level's docstring for why this is single-level (not
    # nested) parallelism.
    work_items = [
        (dataset_name, N, meta)
        for dataset_name, meta in dataset_meta.items()
        for N in meta["levels"]
    ]
    print(f"\nDispatching {len(work_items)} (dataset, level) units across "
          f"{mp.cpu_count()} CPUs...\n", flush=True)

    unit_results = Parallel(n_jobs=-1, batch_size=1, backend="loky", verbose=10)(
        delayed(run_one_level)(
            dataset_name, N, meta["dataset_df"], meta["fits"], meta["seeds"],
            meta["n_c_str"], meta["benign_method"], meta["clinvar_release"],
            output_dir, batch_args,
        )
        for dataset_name, N, meta in work_items
    )

    calib_paths_by_dataset = defaultdict(dict)
    for dataset_name, N, calib_path in unit_results:
        if calib_path is None:
            print(f"{dataset_name} level N={N}: SKIP (calibration failed)")
            continue
        calib_paths_by_dataset[dataset_name][N] = calib_path

    # Pass 3: compare every dataset's levels against its own baseline.
    # Parallelized across datasets -- load_dataset_from_df (Scoreset
    # construction: ClinVar/splicing-filter lookups) is real per-dataset
    # work, not "cheap" preprocessing (the same mistake already found and
    # fixed for tests/benchmark_num_fits_dataframe.py's Pass 1 applies here
    # too, just for the comparison step instead of the fitting step).
    print(f"\nComparing {len(dataset_meta)} datasets across {mp.cpu_count()} CPUs...\n", flush=True)
    pass3_results = Parallel(n_jobs=-1, batch_size=1, backend="loky", verbose=10)(
        delayed(compare_one_dataset)(
            dataset_name, meta, calib_paths_by_dataset[dataset_name], args, baseline_n,
        )
        for dataset_name, meta in dataset_meta.items()
    )
    all_rows = []
    for dataset_name, rows in pass3_results:
        all_rows.extend(rows)

    if all_rows:
        summary_path = output_dir / "bootstrap_reduction_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
