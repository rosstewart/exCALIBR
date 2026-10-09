#!/usr/bin/env python3
"""
Fit quality of skew-locked vs. regular ExCALIBR, by normalized Yang distance
(p=2; the manuscript's fit-quality metric, Methods "Fit quality").

For each calibrated dataset and its selected component count, the Yang
distance between each sample's empirical training-score CDF and the fitted
mixture CDF is computed for every bootstrap iteration of

  - the regular fit (config.PRECOMPUTED_FITS, via legacy_fits; shares the
    per-dataset cache with Extended Data Fig. 1 / Supplementary Data 1), and
  - the skew-locked refit (config.SKEW_LOCKED_BOOTSTRAP_RESULTS: same
    bootstrap seeds, every component's skewness fixed at 0),

on the same bootstrap splits. One row per dataset: the median across
bootstraps for each sample (pathogenic, benign, gnomad, synonymous) for both
fits, plus the per-dataset maximum over samples (the quantity behind the
"poor fit" threshold, config.YANG_GOOD_FIT_THRESHOLD).

The skew-locked side is slow the first time (~minutes per dataset at full
parallelism); results are cached per dataset under
config.YANG_CACHE_DIR/skew_locked, so reruns are fast.

    python -m analysis.skew_locked_yang [--datasets A B ...] [--n-jobs -1]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from analysis import config as cfg

SAMPLES = ("pathogenic", "benign", "gnomad", "synonymous")
SKEW_LOCKED_CALIB_DIR = Path(cfg.SKEW_LOCKED_BOOTSTRAP_RESULTS).parent / "calib"


def _median(vals) -> float:
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan


def compute(datasets: Iterable[str], n_jobs: int = -1, n_grid: int = cfg.YANG_N_GRID,
            skew_boot: Optional[dict] = None, split: str = "train") -> pd.DataFrame:
    """`split`: "train" (each fit scored on its own training variants) or
    "val" (on that bootstrap's held-out variants)."""
    from analysis.legacy_fits import load_scoreset_and_fits, resolve_component_for
    from analysis.yang_distance import cached_bootstrap_yang_distances

    if skew_boot is None:
        with gzip.open(cfg.SKEW_LOCKED_BOOTSTRAP_RESULTS, "rt") as f:
            skew_boot = json.load(f)
    rows: List[dict] = []
    datasets = list(datasets)
    for i, dataset in enumerate(datasets, 1):
        if dataset not in skew_boot:
            print(f"[{i}/{len(datasets)}] SKIP {dataset}: not in skew-locked results")
            continue
        try:
            n_c, benign_method = resolve_component_for(dataset)
            scoreset, _, fits, _, n_c, _, _ = load_scoreset_and_fits(
                dataset, n_c=n_c, benign_method=benign_method)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[{i}/{len(datasets)}] SKIP {dataset}: {e}")
            continue
        # Same bootstrap seeds, same component count; drop seeds whose fit failed.
        # Keep only what the CDF needs: each Yang task receives the whole fits
        # dict, and the full records (latent_q, initial_params, ...) make that
        # transfer dominate the run time.
        skew_fits = {int(seed): {n_c: {"fit": {k: rec[n_c]["fit"][k] for k in ("component_params", "weights")}}}
                     for seed, rec in skew_boot[dataset].items() if rec.get(n_c) is not None}
        if not skew_fits:
            print(f"[{i}/{len(datasets)}] SKIP {dataset}: no skew-locked {n_c} fits")
            continue
        print(f"[{i}/{len(datasets)}] {dataset} ({n_c}): regular + skew-locked Yang distances")
        reg = cached_bootstrap_yang_distances(
            dataset, n_c, fits, scoreset, cache_dir=cfg.YANG_CACHE_DIR, output_dir=cfg.OUTPUT_DIR,
            dataset_to_splits=None, n_jobs=n_jobs, n_grid=n_grid, split=split)
        skew = cached_bootstrap_yang_distances(
            dataset, n_c, skew_fits, scoreset, cache_dir=Path(cfg.YANG_CACHE_DIR) / "skew_locked",
            output_dir=SKEW_LOCKED_CALIB_DIR, dataset_to_splits=None, n_jobs=n_jobs, n_grid=n_grid, split=split)
        row = {"dataset": dataset, "n_c": n_c, "n_bootstraps_skew_locked": len(skew_fits)}
        # Both distance arrays are in bootstrap-seed order; pair by seed so the
        # comparison is on the same training split (bootstrap resampling
        # noise cancels).
        reg_seeds = getattr(fits, "seeds", list(range(len(fits))))
        skew_seeds = sorted(skew_fits)
        common = sorted(set(reg_seeds) & set(skew_seeds))
        ri = {sd: k for k, sd in enumerate(reg_seeds)}
        si = {sd: k for k, sd in enumerate(skew_seeds)}
        row["n_bootstraps_paired"] = len(common)
        from analysis.yang_distance import production_splits, _resolve_sample_indices
        _sp = (production_splits() or {}).get(dataset, {})
        _idx = _resolve_sample_indices(scoreset)
        for s, col in _idx.items():
            if col is None or not _sp:
                row[f"n_{split}_{s}"] = np.nan
                continue
            row[f"n_{split}_{s}"] = float(np.median([np.asarray(v[f"{split}_sample_assignments"])[:, col].sum()
                                                    for v in _sp.values()]))
        for s in SAMPLES:
            r = np.asarray(reg.get(s, []), dtype=float)
            k = np.asarray(skew.get(s, []), dtype=float)
            row[f"regular_{s}"] = _median(r)
            row[f"skew_locked_{s}"] = _median(k)
            if len(r) and len(k):
                d = np.array([k[si[sd]] - r[ri[sd]] for sd in common])
                row[f"paired_diff_{s}"] = _median(d)
            else:
                row[f"paired_diff_{s}"] = np.nan
        row["regular_max"] = np.nanmax([row[f"regular_{s}"] for s in SAMPLES])
        row["skew_locked_max"] = np.nanmax([row[f"skew_locked_{s}"] for s in SAMPLES])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Per sample: datasets with that sample, median over datasets of the
    per-dataset median paired (same-bootstrap) difference skew-locked minus
    regular, datasets where skew-locked is worse, and a two-sided Wilcoxon
    signed-rank test of the per-dataset paired differences against 0."""
    import scipy.stats as sps

    rows = []
    for s in SAMPLES:
        d = df[f"paired_diff_{s}"].dropna() if f"paired_diff_{s}" in df else pd.Series(dtype=float)
        rows.append({"sample": s, "n": len(d), "median_diff": float(d.median()) if len(d) else np.nan,
                     "n_worse": int((d > 0).sum()),
                     "p_wilcoxon": float(sps.wilcoxon(d).pvalue) if len(d) > 1 else np.nan})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, threshold: float = None) -> dict:
    """Numbers for the supplement text: datasets compared, median of the
    per-dataset worst-sample distance for each fit, datasets above the
    poor-fit threshold, and a paired two-sided Wilcoxon signed-rank test."""
    import scipy.stats as sps

    threshold = cfg.YANG_GOOD_FIT_THRESHOLD if threshold is None else threshold
    d = df.dropna(subset=["regular_max", "skew_locked_max"])
    diff = d["skew_locked_max"] - d["regular_max"]
    return {
        "n": len(d),
        "regular_med": float(d["regular_max"].median()),
        "skew_locked_med": float(d["skew_locked_max"].median()),
        "regular_poor": int((d["regular_max"] >= threshold).sum()),
        "skew_locked_poor": int((d["skew_locked_max"] >= threshold).sum()),
        "n_worse": int((diff > 0).sum()),
        "p_wilcoxon": float(sps.wilcoxon(d["skew_locked_max"], d["regular_max"]).pvalue) if len(d) else np.nan,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None, help="default: every calibrated dataset")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--split", choices=("train", "val"), default="train",
                    help="score fits on their training variants or on each bootstrap's held-out variants")
    ap.add_argument("--out", default=None, help="default: yang_skew_locked_vs_regular[_val].csv next to the fits")
    args = ap.parse_args()
    if args.datasets is None:
        from analysis.discovery import discover_outputs, DEFAULT_EXCLUDED_DATASETS
        tree, _, _ = discover_outputs(Path(cfg.OUTPUT_DIR))
        args.datasets = sorted(d for d in tree if d not in DEFAULT_EXCLUDED_DATASETS)
    if args.out is None:
        args.out = str(Path(cfg.SKEW_LOCKED_BOOTSTRAP_RESULTS).parent
                       / f"yang_skew_locked_vs_regular{'' if args.split == 'train' else '_' + args.split}.csv")
    out = compute(args.datasets, n_jobs=args.n_jobs, split=args.split)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows -> {args.out}")
    print(summarize(out))
    print(summarize_by_sample(out).to_string(index=False))
