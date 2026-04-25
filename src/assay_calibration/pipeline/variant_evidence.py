"""
Per-variant evidence assignment (standard and out-of-bag).

Standard: assigns points using the global calibration (point_ranges) for every variant.
OOB: for each variant, rebuilds calibration using only bootstrap iterations where
     that variant was held out (validation set), then assigns points.
     This matches the full processing logic from assign_points.py.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
import logging

from ..fit_utils.fit import calculate_score_ranges, thresholds_from_prior
from ..fit_utils.point_ranges import (
    enforce_monotonicity_point_ranges,
    extend_points_to_xlims,
)
from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flatten_point_ranges(point_ranges: Dict) -> Dict:
    """Flatten {k: [[lo, hi]]} or {k: []} to {k: [lo, hi]} or {k: []}."""
    flat = {}
    for key, ranges in point_ranges.items():
        if isinstance(ranges, np.ndarray):
            ranges = ranges.tolist()
        if isinstance(ranges, list):
            if len(ranges) == 0:
                flat[key] = []
            elif len(ranges) == 1 and isinstance(ranges[0], (list, np.ndarray)):
                flat[key] = list(ranges[0])
            elif len(ranges) == 2 and not isinstance(ranges[0], (list, np.ndarray)):
                flat[key] = list(ranges)  # already flat
            else:
                # multiple sub-ranges -- take bounding interval
                all_lo = min(r[0] for r in ranges)
                all_hi = max(r[1] for r in ranges)
                flat[key] = [all_lo, all_hi]
        else:
            flat[key] = ranges
    return flat


def _assign_points(score: float, flat_ranges: Dict) -> int:
    """Return the evidence point value whose range contains *score*, else 0."""
    if score is None or np.isnan(score):
        return 0
    for key, bounds in flat_ranges.items():
        if len(bounds) == 2:
            lo, hi = bounds
            if lo <= score <= hi:
                return int(key)
    return 0


def _get_variant_ids(scoreset) -> List[str]:
    """Best-effort variant ID extraction; falls back to integer index."""
    ids = []
    if hasattr(scoreset, "get_variants_by_id") and hasattr(scoreset, "_keep_mask"):
        variants_by_id = scoreset.get_variants_by_id()
        kept = 0
        for all_idx, (_, variants) in enumerate(variants_by_id.items()):
            if scoreset._keep_mask[all_idx]:
                v = variants[0]
                if hasattr(v, "ID"):
                    ids.append(f"{v.ID}_{v.Gene}_{v.Chrom}_{v.hgvs_c}")
                else:
                    ids.append(f"variant_{kept}")
                kept += 1
    else:
        ids = [f"variant_{i}" for i in range(len(scoreset.scores))]
    return ids


# ---------------------------------------------------------------------------
# Standard (in-bag) per-variant assignment
# ---------------------------------------------------------------------------

def _build_standard_table(scoreset, calibration: Dict) -> pd.DataFrame:
    """Assign every variant its evidence points from the global calibration."""
    flat = _flatten_point_ranges(calibration["point_ranges"])
    ids = _get_variant_ids(scoreset)

    rows = []
    for idx in range(len(scoreset.scores)):
        score = float(scoreset.scores[idx])
        pts = _assign_points(score, flat)

        sample = "Unknown"
        for s_idx in range(len(scoreset.sample_names)):
            if scoreset._sample_assignments[idx, s_idx]:
                sample = scoreset.sample_names[s_idx]
                break

        rows.append({
            "variant_id": ids[idx] if idx < len(ids) else f"variant_{idx}",
            "score": score,
            "sample": sample,
            "standard_points": pts,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OOB helpers
# ---------------------------------------------------------------------------

def _build_oob_mapping(
    scoreset,
    dataset_splits: Dict[int, Dict],
    valid_bootstrap_seeds: np.ndarray,
) -> Dict[int, List[int]]:
    """
    Map each variant index -> list of *filtered* bootstrap indices where it
    appeared in the validation (OOB) set.

    Parameters
    ----------
    valid_bootstrap_seeds : 1-D int array
        ``valid_bootstrap_seeds[filtered_idx]`` is the original bootstrap seed.
    """
    seed_to_filtered = {int(s): i for i, s in enumerate(valid_bootstrap_seeds)}

    n_seeds_in_splits = len(dataset_splits)
    n_seeds_valid = len(seed_to_filtered)
    print(f"  OOB debug: {n_seeds_in_splits} seeds in splits, "
          f"{n_seeds_valid} valid after prior filtering")

    # Fast lookup: (score, class_idx) -> list of variant indices
    score_cls_map = defaultdict(list)
    for idx in range(len(scoreset.scores)):
        sc = scoreset.scores[idx]
        for cls in np.where(scoreset._sample_assignments[idx])[0]:
            score_cls_map[(sc, int(cls))].append(idx)

    print(f"  OOB debug: {len(score_cls_map)} unique (score, class) keys in scoreset, "
          f"{len(scoreset.scores)} total variants")

    variant_oob: Dict[int, List[int]] = defaultdict(list)
    n_val_obs_total = 0
    n_val_obs_matched = 0

    for seed, split in dataset_splits.items():
        if seed not in seed_to_filtered:
            continue
        fidx = seed_to_filtered[seed]
        for obs, assign in zip(split["val_observations"],
                               split["val_sample_assignments"]):
            n_val_obs_total += 1
            cls = int(np.where(assign)[0][0])
            matches = score_cls_map.get((obs, cls), [])
            if matches:
                n_val_obs_matched += 1
            for vidx in matches:
                variant_oob[vidx].append(fidx)

    print(f"  OOB debug: {n_val_obs_matched}/{n_val_obs_total} val observations "
          f"matched a scoreset variant")

    if n_val_obs_total > 0 and n_val_obs_matched == 0:
        # Diagnose key mismatch
        sample_split_key = None
        for seed in list(dataset_splits.keys())[:1]:
            if seed in seed_to_filtered:
                obs = dataset_splits[seed]["val_observations"]
                assign = dataset_splits[seed]["val_sample_assignments"]
                if len(obs) > 0:
                    sample_split_key = (obs[0], int(np.where(assign[0])[0][0]))
                    break
        sample_scoreset_key = next(iter(score_cls_map.keys()), None)
        print(f"  OOB debug: KEY MISMATCH DETECTED")
        print(f"    Sample split key:    {sample_split_key} "
              f"(types: {type(sample_split_key[0]).__name__ if sample_split_key else '?'}, "
              f"{type(sample_split_key[1]).__name__ if sample_split_key else '?'})")
        print(f"    Sample scoreset key: {sample_scoreset_key} "
              f"(types: {type(sample_scoreset_key[0]).__name__ if sample_scoreset_key else '?'}, "
              f"{type(sample_scoreset_key[1]).__name__ if sample_scoreset_key else '?'})")

    # Report OOB coverage stats
    if variant_oob:
        oob_counts = [len(v) for v in variant_oob.values()]
        print(f"  OOB debug: {len(variant_oob)} variants with OOB data, "
              f"median {np.median(oob_counts):.0f} boots/variant "
              f"(range {min(oob_counts)}-{max(oob_counts)})")

    return dict(variant_oob)


def _process_variant_oob(
    variant_idx: int,
    oob_indices: List[int],
    score: float,
    priors: np.ndarray,
    log_fp: np.ndarray,
    log_fb: np.ndarray,
    score_range: np.ndarray,
    point_values: List[int],
    flipped: bool,
    liberal: bool,
    min_samples: int = 1,
) -> Tuple[int, Optional[Dict]]:
    """
    Compute OOB evidence for one variant using FULL in-bag processing logic.

    This matches _process_single_variant_oob_full in point_ranges.py:
    1. Subset to OOB bootstraps
    2. Filter invalid priors
    3. Compute OOB median prior
    4. Compute OOB LR+ and filter NaN columns
    5. calculate_score_ranges (median prior, 5th/95th percentile LR+)
    6. enforce_monotonicity_point_ranges (first pass)
    7. extend_points_to_xlims
    8. enforce_monotonicity_point_ranges (second pass)
    9. Flatten and assign points

    Returns (variant_idx, result_dict) or (variant_idx, {"_fail": reason}).
    """

    if len(oob_indices) < min_samples:
        return variant_idx, {"_fail": f"too_few_oob ({len(oob_indices)}<{min_samples})"}

    oob_priors = priors[oob_indices]
    oob_fp = log_fp[oob_indices]
    oob_fb = log_fb[oob_indices]

    # Filter invalid priors
    valid = ~np.isnan(oob_priors) & (oob_priors > 0) & (oob_priors < 1)
    n_valid = int(valid.sum())
    oob_priors, oob_fp, oob_fb = oob_priors[valid], oob_fp[valid], oob_fb[valid]

    if len(oob_priors) < min_samples:
        return variant_idx, {"_fail": f"too_few_valid_priors ({n_valid}<{min_samples})"}

    prior = float(np.nanmedian(oob_priors))
    if prior <= 0 or prior >= 1:
        return variant_idx, {"_fail": f"invalid_median_prior ({prior})"}

    # Compute OOB LR+
    lr_plus = oob_fp - oob_fb
    nan_counts = np.isnan(lr_plus).sum(0)
    subset = nan_counts < lr_plus.shape[0]
    if not np.any(subset):
        return variant_idx, {"_fail": f"all_lr_nan (shape={lr_plus.shape})"}

    vsr = score_range[subset]
    vlr = lr_plus[:, subset]

    try:
        # Step 5: calculate_score_ranges (same as in-bag median_prior branch)
        pr_p, pr_b, C = calculate_score_ranges(
            np.nanpercentile(vlr, 5, axis=0),
            np.nanpercentile(vlr, 95, axis=0),
            prior, vsr, point_values,
        )
        pr = {**pr_p, **pr_b}

        # Check if prior is valid (matches in-bag check)
        if prior <= 0 or prior >= 1:
            for point in pr:
                pr[point] = []

        # Step 6: enforce monotonicity (first pass)
        enforce_monotonicity_point_ranges(pr, point_values, vsr, flipped, liberal)

        # Step 7: extend to xlims
        extend_points_to_xlims(pr, point_values, vsr, flipped, inf=True)

        # Step 8: enforce monotonicity (second pass)
        enforce_monotonicity_point_ranges(pr, point_values, vsr, flipped, liberal)

        # Step 9: flatten and assign
        pts = _assign_points(score, _flatten_point_ranges(pr))
    except Exception as e:
        return variant_idx, {"_fail": f"exception: {type(e).__name__}: {str(e)[:200]}"}

    return variant_idx, {
        "points": pts,
        "n_oob": len(oob_indices),
        "n_oob_valid": n_valid,
        "oob_prior": prior,
        "score": float(score),
    }


def _compute_oob_evidence(
    scoreset,
    calibration: Dict,
    dataset_splits: Dict[int, Dict],
    config: PipelineConfig,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict]:
    """Run OOB evidence computation for all variants that have enough OOB samples."""

    def log(msg):
        (logger.info if logger else print)(msg)

    priors = np.asarray(calibration["priors"])
    log_fp = np.asarray(calibration["log_fp"])
    log_fb = np.asarray(calibration["log_fb"])
    score_range = np.asarray(calibration["score_range"])
    flipped = calibration.get("scoreset_flipped", False)
    valid_seeds = np.asarray(calibration["valid_bootstrap_seeds"])

    oob_map = _build_oob_mapping(scoreset, dataset_splits, valid_seeds)
    log(f"  OOB mapping built: {len(oob_map)} variants have OOB data")

    liberal = config.liberal_monotonicity
    n_cores = config.n_jobs if config.n_jobs > 0 else (os.cpu_count() or 1)

    raw = Parallel(n_jobs=min(len(oob_map), n_cores), verbose=5)(
        delayed(_process_variant_oob)(
            vidx, oob_idx, scoreset.scores[vidx],
            priors, log_fp, log_fb, score_range,
            config.point_values, flipped, liberal,
            config.oob_min_samples
        )
        for vidx, oob_idx in oob_map.items()
    )

    # Map filtered index -> variant ID, separating successes from failures
    ids = _get_variant_ids(scoreset)
    out = {}
    fail_reasons = defaultdict(int)

    for vidx, result in raw:
        if result is None:
            fail_reasons["returned_None"] += 1
        elif "_fail" in result:
            reason = result["_fail"].split(" (")[0].split(":")[0]
            fail_reasons[reason] += 1
        else:
            vid = ids[vidx] if vidx < len(ids) else f"variant_{vidx}"
            out[vid] = result

    log(f"  OOB evidence computed for {len(out)}/{len(oob_map)} variants")
    if fail_reasons:
        log(f"  OOB failure breakdown:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            log(f"    {reason}: {count}")

        examples = [(vidx, r) for vidx, r in raw if r is not None and "_fail" in r][:5]
        if examples:
            log(f"  Example failures:")
            for vidx, r in examples:
                n_oob = len(oob_map.get(vidx, []))
                log(f"    variant {vidx} (score={scoreset.scores[vidx]:.4f}, "
                    f"n_oob_boots={n_oob}): {r['_fail']}")

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_variant_table(
    scoreset,
    calibration: Dict,
    config: PipelineConfig,
    dataset_splits: Optional[Dict[int, Dict]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Build a per-variant evidence table.

    Always includes ``standard_points`` (from global calibration).
    When ``config.compute_oob`` is True and *dataset_splits* is provided,
    also includes ``oob_points``, ``oob_n_boots``, and ``oob_prior``.

    Returns
    -------
    pd.DataFrame
    """

    def log(msg):
        (logger.info if logger else print)(msg)

    # --- standard assignment (always) ---
    df = _build_standard_table(scoreset, calibration)
    log(f"  Standard evidence assigned to {len(df)} variants")

    # --- OOB assignment (optional) ---
    if config.compute_oob:
        if dataset_splits is None:
            log("  WARNING: OOB requested but no dataset_splits provided; skipping")
        elif "valid_bootstrap_seeds" not in calibration:
            log("  WARNING: OOB requested but calibration missing valid_bootstrap_seeds; skipping")
        else:
            log("  Computing OOB evidence...")
            oob = _compute_oob_evidence(
                scoreset, calibration, dataset_splits, config, logger,
            )
            if oob:
                oob_df = pd.DataFrame([
                    {"variant_id": vid, "oob_points": r["points"],
                     "oob_n_boots": r["n_oob_valid"], "oob_prior": r["oob_prior"]}
                    for vid, r in oob.items()
                ])
                df = df.merge(oob_df, on="variant_id", how="left")
            else:
                df["oob_points"] = np.nan
                df["oob_n_boots"] = 0
                df["oob_prior"] = np.nan

            n_oob = df["oob_points"].notna().sum()
            log(f"  OOB evidence assigned to {n_oob}/{len(df)} variants")

            # Quick concordance summary
            both = df.dropna(subset=["oob_points"])
            if len(both) > 0:
                agree = (both["standard_points"] == both["oob_points"]).sum()
                log(f"  Standard/OOB agreement: {agree}/{len(both)} "
                    f"({100*agree/len(both):.1f}%)")

    return df
