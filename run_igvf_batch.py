#!/usr/bin/env python
"""
IGVF Batch Processing - Run calibration pipeline across multiple datasets
with per-dataset configurations loaded from a JSON config file.

Input:
  - A CSV/TSV containing all datasets, distinguished by a "Dataset" column
  - A JSON config file mapping dataset names to [n_c, benign_method, {overrides}]
  - Precomputed bootstrap fits (gzipped JSON) keyed by dataset name

Example:
  python run_igvf_batch.py \\
      --dataset data/integrated_variant_effect_dataset.tsv.gz \\
      --dataset-configs src/igvf_configs/dataset_configs_jan_2026.json \\
      --precomputed-fits /data/ross/assay_calibration/results.json.gz \\
      --output-dir ./igvf_output \\
      --oob --splits-file /data/ross/assay_calibration/splits.pkl
"""
import os
import sys
import json
import argparse
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from src.assay_calibration.pipeline.config import PipelineConfig

GENES_2018 = {"BRCA1", "MSH2", "PTEN", "TP53"}
ALL_CONFIGS = [("2c", "avg"), ("2c", "benign"), ("3c", "avg"), ("3c", "benign")]
from src.assay_calibration.pipeline.visualize import (
    generate_visualizations,
    load_precomputed_fits,
    process_component_fits,
    plot_all_configs_comparison,
)
from src.assay_calibration.pipeline.model_selection import bootstrap_paired_test
from src.assay_calibration.pipeline.variant_evidence import compute_variant_table
from src.assay_calibration.pipeline.utils import (
    setup_logging,
    save_results,
    load_dataset_from_df,
)


def parse_dataset_config(
    config_entry,
) -> Tuple[str, str, Dict]:
    """
    Parse a dataset config entry.

    Supports two formats:
      - ["3c", "avg"]                          -> n_c="3c", benign="avg", overrides={}
      - ["3c", "avg", {"liberal_monotonicity": false}] -> with overrides
    """
    if isinstance(config_entry, dict):
        n_c = config_entry["n_c"]
        benign = config_entry.get("benign_method", "avg")
        overrides = {k: v for k, v in config_entry.items()
                     if k not in ("n_c", "benign_method")}
        return n_c, benign, overrides

    # List/tuple format
    n_c = config_entry[0]
    benign = config_entry[1] if len(config_entry) > 1 else "avg"
    overrides = config_entry[2] if len(config_entry) > 2 and isinstance(config_entry[2], dict) else {}
    return n_c, benign, overrides


def run_single_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    bootstrap_results: Dict,
    n_c: str,
    benign_method: str,
    overrides: Dict,
    args,
    dataset_splits: Optional[Dict] = None,
    acmg_mapping_method: str = "tavtigian",
    output_name_suffix: str = "",
    clinvar_mode: str = "default",
) -> Optional[Dict]:
    """Run calibration for a single dataset within the batch.

    Parameters
    ----------
    n_c : str
        Component key, e.g. ``"2c"`` or ``"3c"``.  The special value
        ``"all"`` processes both 2c and 3c (with model selection saved
        but all components output).
    clinvar_mode : str
        ``"2018"``     – forced ClinVar 2018 (genes_2018 auto-detected or explicit)
        ``"default"``  – use dataset name / ``--clinvar-release`` arg as before
    """

    csv_name = dataset_name.replace("_clinvar_2018", "")
    dataset_df = df[df["Dataset"] == csv_name].copy()
    if len(dataset_df) == 0:
        print(f"  SKIP {dataset_name}: no rows in dataset CSV")
        return None

    # Effective dataset name carries the method suffix; must be set on the
    # DataFrame before load_dataset_from_df, which filters by config.dataset_name.
    effective_name = dataset_name + output_name_suffix
    dataset_df["Dataset"] = effective_name

    # Determine ClinVar release
    if clinvar_mode == "2018":
        clinvar_release = "2018"
    elif "clinvar_2018" in dataset_name and "not_clinvar_2018" not in dataset_name:
        clinvar_release = "2018"
    else:
        clinvar_release = getattr(args, "clinvar_release", "2026")

    # Build component list
    if n_c == "all":
        component_list = [2, 3]
    else:
        component_list = [int(n_c.replace("c", ""))]

    # Build per-dataset config
    config = PipelineConfig(
        dataset_csv=args.dataset,
        dataset_name=effective_name,
        output_dir=os.path.join(args.output_dir, dataset_name),
        components=component_list,
        use_median_prior=True,
        use_2c_equation=False,
        liberal_monotonicity=overrides.get("liberal_monotonicity", True),
        benign_method=benign_method,
        scoreset_flipped_override=overrides.get("scoreset_flipped_override", None),
        compute_oob=args.oob,
        oob_min_samples=args.oob_min_samples,
        n_jobs=args.n_jobs_inner,
        auto_select_model=False,
        clinvar_release=clinvar_release,
        min_clinvar_star=args.min_clinvar_star,
        population_type=args.population_type,
        point_values=[1, 2, 3, 4, 5, 6, 7, 8],
        sample_names=args.sample_names if hasattr(args, "sample_names") else None,
        debug=args.debug if hasattr(args, "debug") else False,
        acmg_mapping_method=acmg_mapping_method,
        synonymous_exclusive=getattr(args, "synonymous_exclusive", False),
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(output_dir), dataset_name)

    print(f"  clinvar={clinvar_release}  mode={clinvar_mode}  acmg={acmg_mapping_method}")

    try:
        # Load scoreset
        scoreset = load_dataset_from_df(dataset_df, config)

        n_samples = len([s for s in scoreset.samples])
        logger.info(f"Dataset {dataset_name}: {len(scoreset.scores)} variants, "
                     f"{n_samples} samples, components={component_list} {benign_method}")

        # Build selected_components dict
        selected_components = {f"{c}c": c for c in component_list}

        # Model selection (when processing multiple components)
        selected_k = None
        if len(component_list) > 1:
            try:
                test_result = bootstrap_paired_test(
                    bootstrap_results, verbose=False
                )
                selected_k = test_result["conservative_k"]
                logger.info(f"  Model selection: {selected_k}c")

                # Save model selection result
                test_file = output_dir / f"{dataset_name}_model_selection.json"
                with open(test_file, "w") as f:
                    json.dump(test_result, f, indent=2)
            except Exception as e:
                logger.warning(f"  Model selection failed: {e}")

        # Generate calibration for ALL components
        results = generate_visualizations(
            bootstrap_results=bootstrap_results,
            config=config,
            selected_components=selected_components,
            logger=logger,
            scoreset=scoreset,
        )

        if not results:
            logger.warning(f"  No results for {dataset_name}")
            return None

        if getattr(args, "viz_only", False):
            logger.info("  --viz-only: skipping variant table and calibration save")
            return results

        # Per-variant evidence table
        for comp_key, calibration in results.items():
            variant_df = compute_variant_table(
                scoreset=scoreset,
                calibration=calibration,
                config=config,
                dataset_splits=dataset_splits,
                logger=logger,
            )
            table_path = output_dir / f"{config.dataset_name}_{comp_key}_variants.csv"
            variant_df.to_csv(table_path, index=False)
            logger.info(f"  Saved: {table_path} ({len(variant_df)} variants)")

        # Save calibration
        save_results(
            results=results,
            bootstrap_results=None,
            config=config,
            logger=logger,
            selected_k=selected_k,
        )

        return results

    except Exception as e:
        logger.error(f"  FAILED {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _wait_child(child_pid: int, timeout: int, label: str, logger) -> bool:
    """Wait for a forked child with timeout. Returns True if it exited cleanly."""
    import time
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        pid, st = os.waitpid(child_pid, os.WNOHANG)
        if pid != 0:
            if os.WIFEXITED(st) and os.WEXITSTATUS(st) == 0:
                return True
            logger.error(f"  Process failed: {label}")
            return False
        time.sleep(1)
    os.kill(child_pid, signal.SIGKILL)
    os.waitpid(child_pid, 0)
    logger.warning(f"  Timed out ({timeout}s): {label}")
    return False


def _load_calibration_from_disk(calib_path: Path, lr_path: Path):
    """Load a calibration dict suitable for comparison viz from saved files.

    Supports both the compact (percentile-only) lr_values format and the
    legacy format that stored all 1000 bootstrap rows.
    """
    import gzip as _gzip
    with open(calib_path) as f:
        c = json.load(f)
    with _gzip.open(lr_path, "rt") as f:
        lr = json.load(f)

    # New compact format: precomputed percentile arrays
    if "log_lr_plus_p5" in lr:
        log_lr_pct = np.array([lr["log_lr_plus_p5"], lr["log_lr_plus_p50"], lr["log_lr_plus_p95"]])
        priors_pct = lr.get("priors_pct", [lr["prior"]] * 3)
    else:
        # Legacy format: full 1000-row matrix — compute percentiles on load
        llr = np.asarray(lr["log_lr_plus"])
        log_lr_pct = np.nanpercentile(llr, [5, 50, 95], axis=0)
        priors_pct = [lr.get("prior")] * 3

    return {
        "prior": c["prior"],
        "priors_pct": priors_pct,
        # Aliases expected by plot_scoreset_best_config: priors as an array so
        # np.percentile([p5,p50,p95],[5,50,95]) returns approximately the same
        # values; log_lr_plus as the 3-row percentile matrix so
        # np.nanpercentile(llr,[5,50,95],axis=0) is a no-op-equivalent.
        "priors": priors_pct,
        "log_lr_plus": log_lr_pct,
        "point_ranges": {int(k): v for k, v in c["point_ranges"].items()},
        "score_range": lr["score_range"],
        "log_lr_pct": log_lr_pct,   # shape (3, n_score_points): p5, p50, p95
        "C": None,
        "scoreset_flipped": c.get("scoreset_flipped", False),
    }


def _run_one_combo(
    n_c_str: str,
    benign_method: str,
    fits: list,
    seeds: np.ndarray,
    dataset_df: pd.DataFrame,
    dataset_name: str,
    clinvar_release: str,
    output_dir_str: str,
    args,
    n_bootstrap_jobs: int,
    precomputed_log_fp_all=None,
    return_log_fp_all: bool = False,
    liberal_monotonicity: bool = True,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Run one (n_c, benign_method) calibration combo and save to disk.

    Top-level function so joblib/loky can pickle it.
    Returns (comp_key, log_fp_all_fits) where log_fp_all_fits is the pre-filter
    log_fp array (shape: n_fits × score_range_points) when return_log_fp_all=True,
    otherwise None.  comp_key is None on skip/failure.
    """
    from argparse import Namespace as _NS
    comp_key = f"{n_c_str}_{benign_method}"
    output_dir = Path(output_dir_str)
    logger = setup_logging(output_dir_str, f"{dataset_name}_{comp_key}")

    n_c_int = int(n_c_str[0])
    config = PipelineConfig(
        dataset_csv=args.dataset,
        dataset_name=dataset_name,
        output_dir=output_dir_str,
        components=[n_c_int],
        use_median_prior=True,
        use_2c_equation=False,
        liberal_monotonicity=liberal_monotonicity,
        benign_method=benign_method,
        clinvar_release=clinvar_release,
        min_clinvar_star=args.min_clinvar_star,
        population_type=args.population_type,
        point_values=[1, 2, 3, 4, 5, 6, 7, 8],
        n_jobs=n_bootstrap_jobs,
        synonymous_exclusive=getattr(args, "synonymous_exclusive", False),
        acmg_mapping_method=getattr(args, "acmg_mapping_method", "tavtigian"),
    )

    # Reload scoreset inside the worker (avoids large object serialization)
    scoreset = load_dataset_from_df(dataset_df, config)

    print(f"  [{dataset_name}] Running {comp_key} (bootstrap_jobs={n_bootstrap_jobs})...")
    calibration = process_component_fits(
        fits=fits,
        scoreset=scoreset,
        config=config,
        n_c=n_c_int,
        logger=logger,
        bootstrap_seeds=seeds,
        precomputed_log_fp_all=precomputed_log_fp_all,
        return_log_fp_all=return_log_fp_all,
    )
    if calibration is None:
        print(f"  [{dataset_name}] SKIP {comp_key}: calibration returned None")
        return None, None

    save_results({comp_key: calibration}, None, config, logger)
    log_fp_all = calibration.get('log_fp_all_fits') if return_log_fp_all else None
    return comp_key, log_fp_all


def run_all_configs_for_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    bootstrap_results: Dict,
    args,
    selected_config: Optional[Tuple[str, str]],
    clinvar_mode: str = "default",
    overrides: Optional[Dict] = None,
) -> None:
    """
    Run calibrations for all 4 (n_c, benign_method) combinations in parallel
    and save JSON + lr_values to disk.  No visualization is generated here.

    Parallelism budget: n_jobs_inner (per-dataset CPU budget) is divided
    evenly across however many combos run concurrently.
    """
    import multiprocessing as _mp

    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = dataset_name.replace("_clinvar_2018", "")
    dataset_df = df[df["Dataset"] == csv_name].copy()
    dataset_df["Dataset"] = dataset_name
    if len(dataset_df) == 0:
        print(f"  SKIP {dataset_name}: no rows in CSV")
        return

    clinvar_release = "2018" if clinvar_mode == "2018" else getattr(args, "clinvar_release", "2026")

    # Extract fits by n_c once in the parent process
    fits_by_nc: Dict[str, list] = {}
    seeds_by_nc: Dict[str, np.ndarray] = {}
    for n_c_str in ("2c", "3c"):
        fits, seeds = [], []
        for seed in sorted(bootstrap_results, key=lambda k: int(k)):
            if n_c_str in bootstrap_results[seed] and bootstrap_results[seed][n_c_str] is not None:
                fits.append(bootstrap_results[seed][n_c_str])
                seeds.append(int(seed))
        if fits:
            fits_by_nc[n_c_str] = fits
            seeds_by_nc[n_c_str] = np.array(seeds)

    # Determine which combos need to run
    combos_to_run = []
    for n_c_str, benign_method in ALL_CONFIGS:
        if n_c_str not in fits_by_nc:
            continue
        comp_key = f"{n_c_str}_{benign_method}"
        calib_path = output_dir / f"{dataset_name}_{comp_key}_calibration.json"
        old_calib_path = output_dir / f"{dataset_name}_{n_c_str}_calibration.json"
        is_selected = (n_c_str, benign_method) == selected_config
        use_old = (not calib_path.exists()) and is_selected and old_calib_path.exists()
        if args.skip_existing and (calib_path.exists() or use_old):
            label = f"{comp_key} (from {n_c_str})" if use_old else comp_key
            print(f"  SKIP {label}: already exists")
        else:
            combos_to_run.append((n_c_str, benign_method))

    if not combos_to_run:
        return

    # Group combos by n_c: avg runs before benign within each group so log_fp
    # can be shared.  Groups run sequentially so the inner loky bootstrap pool
    # gets all available CPUs without competing with a sibling thread's pool.
    nc_groups: Dict[str, list] = {}
    for n_c_str, benign_method in combos_to_run:
        nc_groups.setdefault(n_c_str, []).append(benign_method)

    if args.n_jobs_inner == -1:
        total_inner = _mp.cpu_count()
    else:
        total_inner = max(1, args.n_jobs_inner)
    # Give all inner CPUs to bootstrap workers — no division needed since n_c
    # groups run one at a time.
    n_bootstrap_jobs = total_inner

    liberal_mono = (overrides or {}).get("liberal_monotonicity", True)
    for n_c_str, benign_methods in nc_groups.items():
        ordered = sorted(benign_methods, key=lambda m: 0 if m == "avg" else 1)
        both_needed = set(ordered) == {"avg", "benign"}
        log_fp_all: Optional[np.ndarray] = None
        for benign_method in ordered:
            is_avg = benign_method == "avg"
            comp_key, returned_log_fp = _run_one_combo(
                n_c_str, benign_method,
                fits_by_nc[n_c_str], seeds_by_nc[n_c_str],
                dataset_df, dataset_name, clinvar_release,
                str(output_dir), args, n_bootstrap_jobs,
                precomputed_log_fp_all=log_fp_all if not is_avg else None,
                return_log_fp_all=is_avg and both_needed,
                liberal_monotonicity=liberal_mono,
            )
            if is_avg and returned_log_fp is not None:
                log_fp_all = returned_log_fp


def _compute_all_configs_metrics(
    scoreset,
    cal_paths: Dict,  # comp_key → Path to raw calibration JSON
) -> Dict:
    """Compute confusion matrix, MCC, and coverage for each (n_c, benign) config.

    Loads point_ranges from the raw calibration JSON (same as analyze_pipeline_output.py
    / _recompute_points_from_calibration) to avoid any format mismatch introduced by
    _load_calibration_from_disk.

    Follows the same sign convention as build_confusion_matrix:
      pts < 0  → Normal  (benign evidence)
      pts == 0 → IR
      pts > 0  → Abnormal (pathogenic evidence)
    """
    from src.assay_calibration.plot_utils.utils import flatten_point_ranges, assign_points

    sample_names = [s[1] for s in scoreset.samples]
    if ("Pathogenic/Likely Pathogenic" not in sample_names
            or "Benign/Likely Benign" not in sample_names):
        return {}

    plp_col = sample_names.index("Pathogenic/Likely Pathogenic")
    blb_col = sample_names.index("Benign/Likely Benign")
    plp_mask = scoreset.sample_assignments[:, plp_col].astype(bool)
    blb_mask = scoreset.sample_assignments[:, blb_col].astype(bool)
    raw_scores = np.array(scoreset.scores, dtype=float)

    metrics = {}
    for comp_key, cal_path in cal_paths.items():
        try:
            with open(cal_path) as f:
                cal = json.load(f)
            if not cal.get("point_ranges"):
                continue

            point_ranges_flat = flatten_point_ranges(cal["point_ranges"])
            points = np.array([assign_points(s, point_ranges_flat) for s in raw_scores])

            plp_pts = points[plp_mask]
            blb_pts = points[blb_mask]

            tp = int((plp_pts > 0).sum())
            fn_v = int((plp_pts < 0).sum())
            ir_plp = int((plp_pts == 0).sum())
            fp = int((blb_pts > 0).sum())
            tn = int((blb_pts < 0).sum())
            ir_blb = int((blb_pts == 0).sum())

            denom = float(((tp + fp) * (tp + fn_v) * (tn + fp) * (tn + fn_v)) ** 0.5)
            mcc = float((tp * tn - fp * fn_v) / denom) if denom > 0 else 0.0
            n_total = len(plp_pts) + len(blb_pts)
            n_classified = tp + tn + fp + fn_v
            coverage = float(n_classified / n_total) if n_total > 0 else 0.0
            accuracy = float((tp + tn) / n_classified) if n_classified > 0 else 0.0

            metrics[comp_key] = {
                "TP": tp, "TN": tn, "FP": fp, "FN": fn_v,
                "IR_PLP": ir_plp, "IR_BLB": ir_blb,
                "MCC": round(mcc, 4),
                "accuracy": round(accuracy, 4),
                "coverage": round(coverage, 4),
                "n_PLP": int(len(plp_pts)),
                "n_BLB": int(len(blb_pts)),
            }
        except Exception as e:
            print(f"  Metrics: failed for {comp_key}: {e}")

    return metrics


def generate_all_configs_viz(
    dataset_name: str,
    df: pd.DataFrame,
    bootstrap_results: Dict,
    args,
    selected_config: Optional[Tuple[str, str]],
    clinvar_mode: str = "default",
    overrides: Optional[Dict] = None,
) -> None:
    """
    Load saved calibrations from disk and generate:
      - per-config confusion metrics JSON (MCC, coverage, TP/TN/FP/FN)
      - individual best-config viz per (n_c, benign) combo
      - 4-way comparison PNG
    Skips any PNG that already exists when --skip-existing is set.
    """
    output_dir = Path(args.output_dir) / dataset_name
    logger = setup_logging(str(output_dir), dataset_name)

    clinvar_release = "2018" if clinvar_mode == "2018" else getattr(args, "clinvar_release", "2026")

    csv_name = dataset_name.replace("_clinvar_2018", "")
    dataset_df = df[df["Dataset"] == csv_name].copy()
    dataset_df["Dataset"] = dataset_name

    config_load = PipelineConfig(
        dataset_csv=args.dataset,
        dataset_name=dataset_name,
        output_dir=str(output_dir),
        components=[2],
        use_median_prior=True,
        use_2c_equation=False,
        liberal_monotonicity=(overrides or {}).get("liberal_monotonicity", True),
        benign_method="avg",
        clinvar_release=clinvar_release,
        min_clinvar_star=args.min_clinvar_star,
        population_type=args.population_type,
        point_values=[1, 2, 3, 4, 5, 6, 7, 8],
        n_jobs=1,
        synonymous_exclusive=getattr(args, "synonymous_exclusive", False),
    )
    scoreset = load_dataset_from_df(dataset_df, config_load)

    fits_by_nc: Dict[str, list] = {}
    for n_c_str in ("2c", "3c"):
        fits = [bootstrap_results[seed][n_c_str]
                for seed in sorted(bootstrap_results, key=lambda k: int(k))
                if n_c_str in bootstrap_results[seed] and bootstrap_results[seed][n_c_str] is not None]
        if fits:
            fits_by_nc[n_c_str] = fits

    # Load calibrations from disk (new or old format)
    cal_paths: Dict[str, Path] = {}   # raw JSON paths, for metrics computation
    calibrations: Dict[str, Dict] = {}
    for n_c_str, benign_method in ALL_CONFIGS:
        comp_key = f"{n_c_str}_{benign_method}"
        calib_path = output_dir / f"{dataset_name}_{comp_key}_calibration.json"
        lr_path = output_dir / f"{dataset_name}_{comp_key}_lr_values.json.gz"
        old_calib = output_dir / f"{dataset_name}_{n_c_str}_calibration.json"
        old_lr = output_dir / f"{dataset_name}_{n_c_str}_lr_values.json.gz"
        is_selected = (n_c_str, benign_method) == selected_config

        if calib_path.exists() and lr_path.exists():
            src_c, src_lr = calib_path, lr_path
        elif is_selected and old_calib.exists() and old_lr.exists():
            src_c, src_lr = old_calib, old_lr
        else:
            continue

        cal_paths[comp_key] = src_c
        try:
            calibrations[comp_key] = _load_calibration_from_disk(src_c, src_lr)
        except Exception as e:
            logger.warning(f"  Could not load {comp_key}: {e}")

    # Per-config metrics (confusion matrix, MCC, coverage)
    metrics: Dict = {}
    try:
        metrics = _compute_all_configs_metrics(scoreset, cal_paths)
        if metrics:
            metrics_path = output_dir / f"{dataset_name}_config_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"  Saved: {metrics_path}")
            print(f"\n  Config metrics for {dataset_name}:")
            hdr = f"  {'Config':<22} {'MCC':>8} {'Acc':>7} {'Cov':>7} {'TP':>5} {'TN':>5} {'FP':>5} {'FN':>5} {'IR_P':>6} {'IR_B':>6}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for ck in [f"{n}_{b}" for n, b in ALL_CONFIGS]:
                if ck not in metrics:
                    continue
                m = metrics[ck]
                sel = " *" if (selected_config and ck == f"{selected_config[0]}_{selected_config[1]}") else ""
                print(f"  {ck:<22} {m['MCC']:>8.3f} {m.get('accuracy', 0.0):>7.3f} {m['coverage']:>7.3f} "
                      f"{m['TP']:>5} {m['TN']:>5} {m['FP']:>5} {m['FN']:>5} "
                      f"{m['IR_PLP']:>6} {m['IR_BLB']:>6}{sel}")
    except Exception as e:
        logger.warning(f"  Metrics computation failed: {e}")

    # Individual viz per combo
    for n_c_str, benign_method in ALL_CONFIGS:
        comp_key = f"{n_c_str}_{benign_method}"
        if comp_key not in calibrations or n_c_str not in fits_by_nc:
            continue
        fig_path = output_dir / f"{dataset_name}_{comp_key}_visualization.png"
        if args.skip_existing and fig_path.exists():
            print(f"  SKIP viz {comp_key}: already exists")
            continue

        calib = calibrations[comp_key]
        child_pid = os.fork()
        if child_pid == 0:
            try:
                from src.assay_calibration.plot_utils.utils import plot_scoreset_best_config
                fig = plot_scoreset_best_config(
                    dataset=dataset_name,
                    scoreset=scoreset,
                    indv_summary=calib,
                    fits=fits_by_nc[n_c_str],
                    score_range=np.asarray(calib["score_range"]),
                    config=f"({benign_method})",
                    n_c=comp_key,
                    n_samples=scoreset.n_samples,
                    relax=False,
                    flipped=calib.get("scoreset_flipped", False),
                )
                fig.savefig(fig_path, dpi=150)
                os._exit(0)
            except Exception:
                os._exit(1)
        else:
            ok = _wait_child(child_pid, 600, str(fig_path), logger)
            if ok:
                logger.info(f"  Saved: {fig_path}")

    # Comparison viz
    comp_path = output_dir / f"{dataset_name}_comparison.png"
    if args.skip_existing and comp_path.exists():
        print(f"  SKIP comparison viz: already exists")
        return

    child_pid = os.fork()
    if child_pid == 0:
        try:
            fig = plot_all_configs_comparison(
                dataset_name=dataset_name,
                scoreset=scoreset,
                fits_by_nc=fits_by_nc,
                calibrations=calibrations,
                selected_config=selected_config,
                metrics=metrics or None,
            )
            if fig is not None:
                fig.savefig(comp_path, dpi=120, bbox_inches="tight")
            os._exit(0)
        except Exception:
            import traceback; traceback.print_exc()
            os._exit(1)
    else:
        ok = _wait_child(child_pid, 600, str(comp_path), logger)
        if ok:
            print(f"  Saved comparison: {comp_path}")


def main():
    parser = argparse.ArgumentParser(
        description="IGVF Batch Calibration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset", required=True,
                       help="Path to input CSV/TSV with all datasets (must have 'Dataset' column)")
    parser.add_argument("--dataset-configs", default=None,
                       help="Path to JSON config mapping dataset names to [n_c, benign_method, {overrides}]. "
                            "If omitted, all datasets in --precomputed-fits are processed with "
                            "default settings (2c+3c, avg, model selection enabled).")
    parser.add_argument("--precomputed-fits", required=True,
                       help="Path to precomputed bootstrap fits (gzipped JSON)")
    parser.add_argument("--output-dir", default="./igvf_output",
                       help="Output directory (default: ./igvf_output)")

    # OOB
    parser.add_argument("--oob", action="store_true",
                       help="Compute OOB per-variant evidence")
    parser.add_argument("--splits-file", default=None,
                       help="Path to precomputed splits pickle for OOB")
    parser.add_argument("--oob-min-samples", type=int, default=10,
                       help="Min OOB samples per variant (default: 10)")

    # Filtering
    parser.add_argument("--datasets", nargs="*", default=None,
                       help="Only process these dataset names (default: all in config)")

    # Parallelization
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of datasets to process in parallel (default: 1)")
    parser.add_argument("--n-jobs-inner", type=int, default=-1,
                       help="Number of parallel jobs within each dataset (default: -1 = all CPUs)")

    # ClinVar
    parser.add_argument("--clinvar-release", default="2026", choices=["2026", "2025", "2018"])
    parser.add_argument("--min-clinvar-star", type=int, default=1)
    parser.add_argument("--population-type", default="gnomAD",
                       choices=["all_variants", "all_nsSNV", "all_missense_nsSNV",
                                "gnomAD", "gnomAD_nsSNV", "gnomAD_missense_nsSNV"])

    # Skip already-completed datasets
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip datasets whose calibration JSON output already exists "
                            "(useful for resuming a failed run)")

    parser.add_argument("--synonymous-exclusive", action="store_true",
                       help="Load Scoreset with synonymous_exclusive=True: synonymous variants "
                            "are assigned only to the synonymous sample, not also to P/LP or B/LB. "
                            "Matches the behaviour used in legacy author-label analysis.")

    # ClinVar 2018 gene override
    parser.add_argument("--no-clinvar-2018", action="store_true",
                       help="Disable automatic ClinVar 2018 mode for BRCA1/MSH2/PTEN/TP53 datasets; "
                            "use default dataset names and ClinVar release instead")

    # Model selection override
    parser.add_argument("--auto-select-model", action="store_true",
                       help="Use bootstrap paired test to auto-select n_c instead of config")
    parser.add_argument("--sample-names", type=str, nargs="+", default=None,
                       help="Explicit sample names matching column order in data "
                            "(applied to all datasets; e.g. 'Pathogenic/Likely Pathogenic' "
                            "'Benign/Likely Benign' gnomAD Synonymous)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging (component params, flip detection, point ranges)")
    parser.add_argument("--viz-only", action="store_true",
                       help="Regenerate visualizations only — skip variant tables and calibration JSON save "
                            "(useful for rerunning after fixing a plot bug)")
    parser.add_argument("--all-configs", action="store_true",
                       help="Run all 4 (2c/3c)×(avg/benign) configurations per dataset, "
                            "generate individual and comparison visualizations. "
                            "The config from --dataset-configs is used only to mark the 'selected' "
                            "config in the comparison plot.")
    parser.add_argument("--generate-config-template", metavar="OUTPUT_JSON",
                       help="Write a config template JSON (all datasets with n_c/benign_method=null) "
                            "and exit. Preserves override fields from --dataset-configs if provided.")
    parser.add_argument("--acmg-mapping-method",
                       choices=["tavtigian", "piecewise", "continuous",
                                "strict_additive", "all"],
                       default="tavtigian",
                       help="ACMG-mapping method (default: tavtigian). "
                            "'all' runs tavtigian and piecewise and writes per-method "
                            "output files for comparison.")

    args = parser.parse_args()

    print("=" * 80)
    print("BATCH CALIBRATION PIPELINE")
    print("=" * 80)

    # Load precomputed fits
    print(f"\nLoading precomputed fits from {args.precomputed_fits}...")
    with gzip.open(args.precomputed_fits, "rt", encoding="utf-8") as f:
        all_bootstrap_results = json.load(f)
    print(f"Loaded fits for {len(all_bootstrap_results)} datasets")

    # Load dataset configs (or auto-discover from precomputed fits)
    if args.dataset_configs is not None:
        with open(args.dataset_configs, "r") as f:
            dataset_configs = json.load(f)
        print(f"\nLoaded {len(dataset_configs)} dataset configurations")
    else:
        # No config provided: auto-discover all datasets from precomputed fits
        # Default: process both 2c and 3c with model selection, avg benign method
        print(f"\nNo --dataset-configs provided; using defaults for all "
              f"{len(all_bootstrap_results)} datasets (both 2c+3c, model selection)")
        dataset_configs = {
            name: ["all", "avg"]  # "all" = process 2c+3c with model selection
            for name in all_bootstrap_results
        }

    # Load input data
    sep = "\t" if args.dataset.endswith((".tsv", ".tsv.gz")) else ","
    print(f"\nLoading dataset CSV from {args.dataset}...")
    df = pd.read_csv(args.dataset, sep=sep)
    print(f"Loaded {len(df)} rows")

    # Load splits if provided
    all_splits = None
    if args.splits_file and args.oob:
        print(f"\nLoading splits from {args.splits_file}...")
        with open(args.splits_file, "rb") as f:
            all_splits = pickle.load(f)
        print(f"Loaded splits for {len(all_splits)} datasets")

    # Process datasets
    os.makedirs(args.output_dir, exist_ok=True)

    def _gene_of(dataset_name: str) -> str:
        """Extract gene name (first underscore-delimited token, strip _clinvar_2018)."""
        base = dataset_name.replace("_clinvar_2018", "")
        return base.split("_")[0]

    # Drive iteration from CSV datasets; configs are optional per-dataset overrides
    csv_datasets = sorted(df["Dataset"].unique())
    if args.datasets:
        csv_datasets = [d for d in csv_datasets if d in set(args.datasets)]

    # --generate-config-template: emit a blank config and exit
    if getattr(args, "generate_config_template", None):
        template = {}
        for dataset_name in csv_datasets:
            if not getattr(args, "no_clinvar_2018", False) and _gene_of(dataset_name) in GENES_2018:
                eff = f"{dataset_name}_clinvar_2018"
            else:
                eff = dataset_name
            if eff not in all_bootstrap_results and dataset_name not in all_bootstrap_results:
                continue
            existing = (dataset_configs.get(eff) or dataset_configs.get(dataset_name) or [])
            if isinstance(existing, list) and len(existing) > 2 and isinstance(existing[2], dict):
                overrides = existing[2]
            elif isinstance(existing, dict):
                overrides = {k: v for k, v in existing.items() if k not in ("n_c", "benign_method")}
            else:
                overrides = {}
            entry = {"n_c": None, "benign_method": None}
            if overrides:
                entry.update(overrides)
            template[eff] = entry
        out_path = args.generate_config_template
        with open(out_path, "w") as f:
            json.dump(template, f, indent=2)
        print(f"Config template written to {out_path} ({len(template)} datasets)")
        return
        print(f"Filtered to {len(csv_datasets)} requested datasets")

    datasets_to_process = []
    for dataset_name in csv_datasets:
        # Determine ClinVar mode
        if not getattr(args, "no_clinvar_2018", False) and _gene_of(dataset_name) in GENES_2018:
            clinvar_mode = "2018"
            effective_dataset_name = f"{dataset_name}_clinvar_2018"
            bootstrap_key = effective_dataset_name
        else:
            clinvar_mode = "default"
            effective_dataset_name = dataset_name
            bootstrap_key = dataset_name

        if bootstrap_key not in all_bootstrap_results:
            print(f"  SKIP {dataset_name}: not in precomputed fits "
                  f"(tried '{bootstrap_key}')")
            continue

        # Look up config; fall back to running all components with model selection
        config_entry = (
            dataset_configs.get(effective_dataset_name)
            or dataset_configs.get(dataset_name)
            or ["all", "avg"]
        )

        n_c, benign_method, overrides = parse_dataset_config(config_entry)

        # In single-dataset mode, skip entries with no config set yet (pending inspection)
        if n_c == "" and benign_method == "" and not getattr(args, "all_configs", False):
            print(f"  SKIP {effective_dataset_name}: n_c/benign_method not configured (use --all-configs to generate vizs for inspection)")
            continue

        # Auto-select model if requested (only when config specifies a single n_c;
        # "all" already handles model selection inside run_single_dataset)
        if args.auto_select_model and n_c != "all":
            try:
                test_result = bootstrap_paired_test(
                    all_bootstrap_results[bootstrap_key], verbose=False
                )
                n_c = f"{test_result['conservative_k']}c"
            except Exception:
                pass  # fallback to config

        # Skip if outputs already exist (not applicable when --all-configs handles its own skips)
        if args.skip_existing and not getattr(args, "all_configs", False):
            if n_c == "all":
                # Only expect components that actually have fits in the bootstrap results
                boot = all_bootstrap_results[bootstrap_key]
                available = {"2c", "3c"} & {k for seed in boot.values()
                                              for k in (seed if isinstance(seed, dict) else {})}
                components_to_check = [c for c in ["2c", "3c"] if c in available] or ["2c", "3c"]
            else:
                components_to_check = [n_c]
            existing = [
                c for c in components_to_check
                if os.path.exists(os.path.join(
                    args.output_dir, effective_dataset_name,
                    f"{effective_dataset_name}_{c}_calibration.json"
                ))
            ]
            if set(existing) == set(components_to_check):
                print(f"  SKIP {effective_dataset_name}: output already exists ({', '.join(existing)})")
                continue

        # Get splits for this dataset
        dataset_splits = None
        if all_splits and effective_dataset_name in all_splits:
            dataset_splits = all_splits[effective_dataset_name]

        datasets_to_process.append((
            effective_dataset_name,
            all_bootstrap_results[bootstrap_key],
            n_c, benign_method, overrides, dataset_splits,
            clinvar_mode,
        ))

    # --all-configs: calibration + viz per dataset, parallelised across datasets
    if getattr(args, "all_configs", False):
        import multiprocessing as _mp
        from argparse import Namespace as _NS

        ac_args = _NS(**vars(args))
        n_cpus = _mp.cpu_count()
        if getattr(ac_args, "viz_only", False):
            # Viz only: no bootstrap workers, so use all CPUs across datasets.
            # Respect an explicit --n-jobs; default of 1 means "user didn't set it",
            # so override to all CPUs.
            ac_args.n_jobs_inner = 1
            if args.n_jobs == 1:
                ac_args.n_jobs = -1
        else:
            # Scale inner job count so n_jobs outer × n_jobs_inner ≤ n_cpus
            if args.n_jobs != 1 and args.n_jobs_inner == -1:
                ac_args.n_jobs_inner = max(1, n_cpus // abs(args.n_jobs))

        def _all_configs_job(name, boot_results, sel_cfg, cv_mode, ovr):
            if not getattr(ac_args, "viz_only", False):
                run_all_configs_for_dataset(name, df, boot_results, ac_args, sel_cfg, cv_mode, ovr)
            generate_all_configs_viz(name, df, boot_results, ac_args, sel_cfg, cv_mode, ovr)

        jobs = []
        for name, boot_results, n_c, benign, ovr, _splits, cv_mode in datasets_to_process:
            if n_c == "" and benign == "":
                sel_cfg = None  # no config selected yet; comparison plot shows all without highlight
            else:
                sel_nc = n_c if n_c in ("2c", "3c") else "3c"
                sel_benign = benign if benign in ("avg", "benign") else "avg"
                sel_cfg = (sel_nc, sel_benign)
            jobs.append((name, boot_results, sel_cfg, cv_mode, ovr))

        print(f"\n[all-configs] {len(jobs)} datasets  (n_jobs={ac_args.n_jobs}, "
              f"n_jobs_inner={ac_args.n_jobs_inner})...")
        if ac_args.n_jobs == 1:
            for idx, (name, boot_results, sel_cfg, cv_mode, ovr) in enumerate(jobs, 1):
                print(f"\n{'='*80}\n[{idx}/{len(jobs)}] {name}  (selected: {sel_cfg})\n{'='*80}")
                _all_configs_job(name, boot_results, sel_cfg, cv_mode, ovr)
        else:
            Parallel(n_jobs=ac_args.n_jobs, verbose=5)(
                delayed(_all_configs_job)(name, boot_results, sel_cfg, cv_mode, ovr)
                for name, boot_results, sel_cfg, cv_mode, ovr in jobs
            )
        print(f"\n{'='*80}\nALL-CONFIGS COMPLETE\n{'='*80}")
        return

    acmg_mapping_methods = (["tavtigian", "piecewise"]
                             if args.acmg_mapping_method == "all"
                             else [args.acmg_mapping_method])
    suffix = (lambda m: "" if (len(acmg_mapping_methods) == 1 and m == "tavtigian")
              else f"_{m}")

    print(f"\nProcessing {len(datasets_to_process)} datasets × "
          f"{len(acmg_mapping_methods)} ACMG-mapping method(s)...")

    if args.n_jobs == 1:
        # Sequential processing
        idx = 0
        for name, boot_results, n_c, benign, ovr, splits, cv_mode in datasets_to_process:
            idx += 1
            for acmg_mapping_method in acmg_mapping_methods:
                print(f"\n{'='*80}")
                print(f"[{idx}/{len(datasets_to_process)}] {name} "
                      f"({n_c}, {benign}, acmg_mapping_method={acmg_mapping_method})")
                print(f"{'='*80}")
                run_single_dataset(name, df, boot_results, n_c, benign, ovr, args, splits,
                                   acmg_mapping_method=acmg_mapping_method,
                                   output_name_suffix=suffix(acmg_mapping_method),
                                   clinvar_mode=cv_mode)
    else:
        # Parallel dataset processing — scale inner jobs to avoid CPU oversubscription
        import multiprocessing
        from argparse import Namespace
        parallel_args = Namespace(**vars(args))
        if args.n_jobs_inner == -1:
            n_cpus = multiprocessing.cpu_count()
            parallel_args.n_jobs_inner = max(1, n_cpus // abs(args.n_jobs))
        Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_single_dataset)(
                name, df, boot_results, n_c, benign, ovr, parallel_args, splits,
                acmg_mapping_method=acmg_mapping_method,
                output_name_suffix=suffix(acmg_mapping_method),
                clinvar_mode=cv_mode,
            )
            for name, boot_results, n_c, benign, ovr, splits, cv_mode in datasets_to_process
            for acmg_mapping_method in acmg_mapping_methods
        )

    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
