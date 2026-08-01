"""
Discovery and loading of run_pipeline.py / run_igvf_batch.py output.

Walks a pipeline output directory for ``*_variants.csv`` / ``*_calibration.json``
files, resolves which (n_c, benign_method) component is "the" calibration for
each dataset, and loads everything into one long-format variants DataFrame.

Extracted from the original monolithic analysis/analyze_pipeline_output.py so
it can be reused by confusion.py / evidence.py / legacy_fits.py without
importing the CLI/notebook module itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# "acmg_bayes" is the current method (supersedes the former separate
# "piecewise"/"continuous"/"strict_additive" tags); the legacy names are kept
# here so filenames from historical runs still parse correctly.
KNOWN_METHODS = {"tavtigian", "acmg_bayes", "piecewise", "continuous", "strict_additive"}

# Excluded by default from analysis unless include_all=True
DEFAULT_EXCLUDED_DATASETS = {
    "F9_Popp_2025_model",
    "TP53_Fayer_2021_meta_clinvar_2018",
    "SFPQ_IGVF",
}

# benign_method suffix attached directly after n_c in output filenames, e.g.
# "..._3c_avg_calibration.json" — part of the component token ("3c_avg"), not the method.
BENIGN_METHODS = {"avg", "benign", "synonymous"}

# Dataset-name spellings that differ between the pipeline/dataset_configs
# naming (used throughout output directory trees) and the master TSV's own
# "Dataset" column, not covered by new_dataset_names.csv. Scoped to this
# analysis module only -- deliberately not added to new_dataset_names.csv,
# which is shared with the rest of the pipeline. pipeline name -> master TSV name.
_DATASET_TSV_NAME_ALIASES = {
    "CHEK2_McCarthy_Leo_2024": "CHEK2_McCarthy-Leo_2024",
}


# ---------------------------------------------------------------------------
# Filename parsing / discovery
# ---------------------------------------------------------------------------

def _parse_stem(stem: str, suffix: str) -> Optional[Tuple[str, str, str]]:
    """Return (dataset, method, comp) from a stem ending in suffix, or None."""
    if not stem.endswith(suffix):
        return None
    core = stem[: -len(suffix)]
    parts = core.split("_")
    if len(parts) < 2:
        return None

    def _is_nc_token(tok: str) -> bool:
        return tok.endswith("c") and tok[:-1].isdigit()

    # Last token is either the component ("3c") or a benign_method suffix
    # ("avg"/"benign"/"synonymous") tacked onto it, e.g. "..._3c_avg_calibration"
    # where comp_key = f"{n_c}_{benign_method}" (see visualize.py). Combine the
    # two into a single comp token: "3c_avg".
    if _is_nc_token(parts[-1]):
        comp = parts.pop()
    elif parts[-1] in BENIGN_METHODS and len(parts) >= 2 and _is_nc_token(parts[-2]):
        benign_method = parts.pop()
        comp = f"{parts.pop()}_{benign_method}"
    else:
        return None

    # Token before component may be a known method
    if parts and parts[-1] in KNOWN_METHODS:
        method = parts.pop()
    else:
        method = "default"

    dataset = "_".join(parts)
    return dataset, method, comp


def _parse_variants_stem(stem: str) -> Optional[Tuple[str, str, str]]:
    return _parse_stem(stem, "_variants")


def _parse_calibration_stem(stem: str) -> Optional[Tuple[str, str, str]]:
    return _parse_stem(stem, "_calibration")


def recompute_points_from_calibration(
    df: pd.DataFrame,
    calibration_path: Path,
) -> pd.DataFrame:
    """Recompute standard_points from score + calibration JSON point_ranges.

    Used when no method-specific variants CSV exists but the calibration JSON
    and a shared (default) variants CSV do. Also reused by
    analysis.robustness to apply a robustness condition's point_ranges to the
    reference (unperturbed) dataset's own variant scores.
    """
    from src.assay_calibration.plot_utils.utils import flatten_point_ranges, assign_points

    with open(calibration_path) as f:
        cal = json.load(f)

    if cal.get("point_ranges") is None:
        raise ValueError(f"No point_ranges in {calibration_path}")

    point_ranges = flatten_point_ranges(cal["point_ranges"])

    df = df.copy()
    scores = df["score"].values
    df["standard_points"] = [assign_points(s, point_ranges) for s in scores]
    if "oob_points" in df.columns:
        df = df.drop(columns=["oob_points", "oob_n_boots", "oob_prior"],
                     errors="ignore")
    return df


# Old private name, kept as an alias so existing call sites in this module
# (and any external code importing it) don't need touching.
_recompute_points_from_calibration = recompute_points_from_calibration


def discover_outputs(
    output_dir: Path,
) -> Tuple[Dict[str, Dict[str, Dict[str, Path]]], Dict[str, Optional[int]], Dict]:
    """Walk output_dir for *_variants.csv and *_calibration.json files.

    Returns
    -------
    tree : {dataset → {comp → {method → csv_path or None}}}
    model_selections : {dataset → conservative_k or None}
    calibrations : {dataset → {method → {comp → cal_path}}}
    """
    tree: Dict[str, Dict[str, Dict[str, Path]]] = {}
    model_selections: Dict[str, Optional[int]] = {}
    calibrations: Dict = {}

    for csv_path in sorted(output_dir.rglob("*_variants.csv")):
        parsed = _parse_variants_stem(csv_path.stem)
        if parsed is None:
            continue
        dataset, method, comp = parsed
        tree.setdefault(dataset, {}).setdefault(comp, {})[method] = csv_path

        ms_path = csv_path.parent / f"{dataset}_model_selection.json"
        if ms_path.exists() and dataset not in model_selections:
            try:
                with open(ms_path) as f:
                    ms = json.load(f)
                model_selections[dataset] = ms.get("conservative_k")
            except Exception:
                pass

    for cal_path in sorted(output_dir.rglob("*_calibration.json")):
        parsed = _parse_calibration_stem(cal_path.stem)
        if parsed is None:
            continue
        dataset, method, comp = parsed
        calibrations.setdefault(dataset, {}).setdefault(method, {})[comp] = cal_path

        # Register method in tree if not already present from a variants CSV
        tree.setdefault(dataset, {}).setdefault(comp, {}).setdefault(method, None)

    return tree, model_selections, calibrations


# ---------------------------------------------------------------------------
# Component resolution
# ---------------------------------------------------------------------------

def _parse_dataset_config_entry(entry) -> str:
    """Extract the component token (n_c[_benign_method]) from a dataset config
    entry, matching the "{n_c}_{benign_method}" naming used in output filenames
    (see visualize.py's comp_key construction)."""
    if isinstance(entry, dict):
        n_c = str(entry.get("n_c", "3c"))
        benign_method = entry.get("benign_method")
        return f"{n_c}_{benign_method}" if benign_method else n_c
    if isinstance(entry, (list, tuple)) and len(entry) > 0:
        return str(entry[0])
    return "3c"


def resolve_component(
    dataset: str,
    available_comps: List[str],
    model_selections: Dict,
    dataset_configs: Optional[Dict],
) -> str:
    """Pick the best component for a dataset."""
    # 1. Dataset configs (same logic as run_igvf_batch.py)
    if dataset_configs and dataset in dataset_configs:
        comp = _parse_dataset_config_entry(dataset_configs[dataset])
        if comp != "all":
            if comp in available_comps:
                return comp
            # benign_method may have been overridden at run time (e.g. no
            # synonymous/benign sample available) — fall back to any available
            # comp sharing the same n_c, e.g. "3c_avg" -> "3c" or "3c_benign"
            n_c = comp.split("_", 1)[0]
            same_nc = sorted(c for c in available_comps if c == n_c or c.startswith(n_c + "_"))
            if same_nc:
                return same_nc[0]

    # 2. model_selection.json
    if model_selections.get(dataset) is not None:
        comp = f"{model_selections[dataset]}c"
        if comp in available_comps:
            return comp
        same_nc = sorted(c for c in available_comps if c.startswith(comp + "_"))
        if same_nc:
            return same_nc[0]

    # 3. Prefer 3c, fall back to 2c, then first available
    for pref in ("3c", "2c"):
        matches = sorted(c for c in available_comps if c == pref or c.startswith(pref + "_"))
        if matches:
            return matches[0]
    return sorted(available_comps)[0]


_master_df_cache: Dict[str, pd.DataFrame] = {}


def load_master_df(dataset_tsv: str) -> pd.DataFrame:
    """Read the master integrated-variant-effect TSV/CSV once per process and
    cache it, keyed by path -- several independent call sites (legacy_fits'
    load_scoreset_and_fits, clingen's build_clingen_confusion, this module's
    own scoreset-rebuild fallbacks) each used to `pd.read_csv(dataset_tsv,
    ...)` fresh on every call, so a single notebook session calling more than
    one of them (e.g. the MSH2 example figure calling load_scoreset_and_fits
    twice, then the Yang-distance diagnostic a third time, all for the same
    dataset) re-read and re-parsed the same multi-MB gzipped file over and
    over. Callers that slice this down to one dataset's rows must still
    `.copy()` before mutating (see _filter_dataset_df) -- the cached frame
    itself is shared and must not be mutated in place."""
    dataset_tsv = str(dataset_tsv)
    if dataset_tsv not in _master_df_cache:
        sep = "\t" if dataset_tsv.endswith((".tsv", ".tsv.gz")) else ","
        _master_df_cache[dataset_tsv] = pd.read_csv(dataset_tsv, sep=sep, low_memory=False)
    return _master_df_cache[dataset_tsv]


def _filter_dataset_df(df_full: pd.DataFrame, dataset: str, dataset_tsv: str) -> pd.DataFrame:
    """Slice the (large, all-datasets) master dataframe down to one dataset's
    rows. Kept separate from scoreset construction so callers that need to
    build several datasets' scoresets (e.g. the parallel fallback in
    load_all_variants) can do this filtering once in the parent process and
    hand each worker only its own small slice -- never the full master frame."""
    csv_name = dataset.replace("_clinvar_2018", "")
    df_ds = df_full[df_full["Dataset"] == csv_name].copy()
    if df_ds.empty:
        from analysis.author_labels import load_name_mapping
        old_to_new, _ = load_name_mapping(str(dataset_tsv))
        alt = old_to_new.get(csv_name)
        if alt:
            df_ds = df_full[df_full["Dataset"] == alt].copy()
    if df_ds.empty:
        alt = _DATASET_TSV_NAME_ALIASES.get(csv_name)
        if alt:
            df_ds = df_full[df_full["Dataset"] == alt].copy()
    if df_ds.empty:
        raise ValueError(f"Dataset '{csv_name}' not found in {dataset_tsv}")
    df_ds["Dataset"] = dataset
    return df_ds


def _resolve_n_jobs(n_requested: int) -> int:
    """CPU-affinity-aware worker count -- same idea as hpc/prepare.py's
    _resolve_n_jobs (cgroup-visible core count, not raw os.cpu_count())."""
    import os as _os
    if n_requested != -1:
        return n_requested
    try:
        return max(1, len(_os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, _os.cpu_count() or 1)


def _merge_oob_columns(df: pd.DataFrame, oob_csv_path: Optional[Path]) -> pd.DataFrame:
    """Left-merge oob_points/oob_n_boots/oob_prior from a saved *_variants.csv
    onto a freshly-built (scoreset + calibration) variant table, by
    variant_id.

    This is the *only* thing the saved CSV is still trusted for -- OOB
    evidence is expensive to recompute (needs dataset_splits that only exist
    mid-pipeline-run) and isn't derivable from calibration.json alone. A
    variant present in the fresh table but absent from the saved CSV (e.g.
    matching failure, or the dataset's saved CSV predates a variant_id fix)
    simply gets NaN oob_* columns here -- callers using effective_points'
    OOB-with-in-bag-fallback convention then correctly fall back to the
    freshly-computed in-bag standard_points/classification for it, rather
    than to a value read from the CSV.
    """
    if oob_csv_path is None:
        return df
    try:
        saved = pd.read_csv(oob_csv_path)
    except Exception:
        return df
    oob_cols = [c for c in ("oob_points", "oob_n_boots", "oob_prior") if c in saved.columns]
    if not oob_cols or "variant_id" not in saved.columns:
        return df
    return df.merge(saved[["variant_id"] + oob_cols], on="variant_id", how="left")


def _run_scoreset_job(dataset: str, method: str, comp: str, cal_path: Path, df_ds: pd.DataFrame,
                       oob_csv_path: Optional[Path] = None):
    """Module-level (not a closure) so joblib/cloudpickle only ever pickles
    the *one* dataset slice passed in as `df_ds` for this call, matching
    hpc/prepare.py's `partitions[ds]` pattern -- a nested/closure-based
    worker here would instead capture and re-serialize every dataset's slice
    on every single task, which is what made an earlier version of this
    silently thrash (huge duplicated payloads) instead of running in
    parallel."""
    try:
        df = _build_variant_table_for_scoreset(dataset, df_ds, cal_path, oob_csv_path)
        return dataset, method, comp, df, None
    except Exception as e:
        return dataset, method, comp, None, e


def _build_variant_table_for_scoreset(dataset: str, df_ds: pd.DataFrame, cal_path: Path,
                                       oob_csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Worker-safe: builds one dataset's scoreset + variant table from an
    already-filtered, single-dataset slice of the master dataframe (small --
    safe to pickle to a joblib worker process) plus one calibration JSON.
    Never touches the full master dataframe itself.

    Reuses `compute_variant_table` (the same function run_igvf_batch.py's
    normal single-config path calls) with `compute_oob=False`, since OOB
    needs `dataset_splits` that only exist mid-pipeline-run -- this builds
    the in-bag standard_points/classification, auth_label, and is_vus
    columns fresh from the scoreset, then optionally merges in oob_* columns
    from *oob_csv_path* via _merge_oob_columns.
    """
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    from src.assay_calibration.pipeline.variant_evidence import compute_variant_table

    with open(cal_path) as f:
        calibration = json.load(f)

    clinvar_release = "2018" if calibration.get("clinvar_2018") else "2026"
    pcfg = PipelineConfig(
        dataset_csv="", dataset_name=dataset, output_dir="/tmp",
        clinvar_release=clinvar_release,
    )
    scoreset = load_dataset_from_df(df_ds, pcfg)
    df = compute_variant_table(scoreset=scoreset, calibration=calibration, config=pcfg)
    return _merge_oob_columns(df, oob_csv_path)


def build_variants_from_scoreset(
    dataset: str,
    cal_path: Path,
    dataset_tsv: str,
    oob_csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Single-dataset convenience wrapper around _build_variant_table_for_scoreset
    for datasets that have no *_variants.csv on disk at all (e.g. runs made
    with `run_igvf_batch.py --all-configs`, which only ever writes
    *_calibration.json / *_lr_values.json.gz -- see
    run_all_configs_for_dataset's docstring: "No visualization is generated
    here" -- and, less obviously, no variant table either).

    Reads the full master TSV itself, so for loading many datasets prefer
    load_all_variants (which reads it once and parallelizes across datasets)
    over calling this in a loop.
    """
    df_full = load_master_df(dataset_tsv)
    df_ds = _filter_dataset_df(df_full, dataset, dataset_tsv)
    return _build_variant_table_for_scoreset(dataset, df_ds, cal_path, oob_csv_path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_variants(
    tree: Dict,
    model_selections: Dict,
    dataset_configs: Optional[Dict],
    methods_filter: Optional[List[str]],
    datasets_filter: Optional[List[str]],
    min_controls: int = 0,
    calibrations: Optional[Dict] = None,
    include_all: bool = False,
    recompute_points: bool = False,
    dataset_tsv: Optional[str] = None,
) -> pd.DataFrame:
    """Load all variants into a single long-format DataFrame.

    Columns: variant_id, score, sample, standard_points, auth_label, is_vus,
             dataset, method, component
    (plus oob_* columns when the saved *_variants.csv has them)

    The variant table (score, sample, standard_points/classification,
    auth_label, is_vus) is always built fresh from a Scoreset + the resolved
    calibration JSON's point_ranges (via compute_variant_table -- the same
    function run_igvf_batch.py itself calls at write time), never trusted
    from an already-saved *_variants.csv: variant_id/auth_label baked into a
    CSV written by older pipeline code can be stale or lossy (see
    _get_variant_ids's mavedb_variant_urn history) relative to rebuilding
    from calibration.json + a freshly-constructed Scoreset, which reproduces
    write-time behavior exactly. The saved *_variants.csv, when present, is
    consulted for exactly the one thing that can't be cheaply recomputed
    here -- oob_points/oob_n_boots/oob_prior (needs dataset_splits that only
    exist mid-pipeline-run) -- merged in by variant_id via
    _merge_oob_columns. A variant with no matching OOB row there (e.g. the
    saved CSV predates a variant_id fix, or simply has no OOB computed)
    keeps its freshly-computed in-bag standard_points/classification rather
    than falling back to a possibly-stale value read from the CSV.

    Falls back to trusting the saved CSV outright only when no calibration
    JSON exists at all for a (dataset, method, comp) -- there is nothing to
    rebuild from in that case.

    recompute_points : retained for backward-compatible call signatures; no
        longer changes behavior, since the base table is now always
        recomputed from calibration.json + Scoreset regardless.

    Unless include_all=True, datasets in DEFAULT_EXCLUDED_DATASETS are skipped, and
    (when dataset_configs is provided) any dataset absent from dataset_configs is
    skipped too.
    """
    frames = []
    pending = []  # [(dataset, method, comp, df)]
    scoreset_jobs = []  # [(dataset, method, comp, cal_path, oob_csv_path)] resolved in parallel below
    for dataset, comp_dict in sorted(tree.items()):
        if datasets_filter and dataset not in datasets_filter:
            continue
        if not include_all:
            if dataset in DEFAULT_EXCLUDED_DATASETS:
                print(f"  SKIP {dataset}: in DEFAULT_EXCLUDED_DATASETS (use include_all=True to override)")
                continue
            if dataset_configs is not None and dataset not in dataset_configs:
                print(f"  SKIP {dataset}: not present in dataset_configs (use include_all=True to override)")
                continue

        comp = resolve_component(dataset, list(comp_dict.keys()), model_selections, dataset_configs)
        method_dict = comp_dict[comp]

        for method, path in method_dict.items():
            if methods_filter and method not in methods_filter:
                continue

            cal_path = (calibrations or {}).get(dataset, {}).get(method, {}).get(comp)
            if cal_path is None:
                # No calibration to rebuild from -- best-effort fall back to
                # trusting the saved CSV outright. Rare: only when a
                # calibration JSON is genuinely missing alongside a
                # variants CSV.
                if path is None:
                    print(f"  SKIP {dataset} / {method}: no variants CSV and no calibration")
                    continue
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    print(f"  WARNING: could not read {path}: {e}")
                    continue
                pending.append((dataset, method, comp, df))
                continue

            # Build fresh from calibration.json + Scoreset. Deferred to the
            # parallel batch below instead of built here, so the master TSV
            # is read/filtered once per dataset rather than once per
            # (dataset, method).
            scoreset_jobs.append((dataset, method, comp, cal_path, path))

    if scoreset_jobs:
        from analysis import config as _cfg
        dataset_tsv = dataset_tsv or _cfg.DATASET_TSV
        df_full = load_master_df(dataset_tsv)

        # Filter to each dataset's own (small) slice once, in this process,
        # *before* dispatching to workers -- so joblib only ever pickles a
        # per-dataset slice to each worker, never the full ~89-dataset master
        # frame (which would otherwise get serialized once per job).
        df_ds_by_dataset = {}
        for dataset, _method, _comp, _cal_path, _oob_path in scoreset_jobs:
            if dataset not in df_ds_by_dataset:
                try:
                    df_ds_by_dataset[dataset] = _filter_dataset_df(df_full, dataset, dataset_tsv)
                except Exception as e:
                    df_ds_by_dataset[dataset] = e
        del df_full

        # Split out slices that failed to filter -- report immediately and
        # don't hand them to workers at all.
        good_jobs = []
        for dataset, method, comp, cal_path, oob_path in scoreset_jobs:
            df_ds = df_ds_by_dataset[dataset]
            if isinstance(df_ds, Exception):
                print(f"  WARNING: scoreset build failed for {dataset} / {method}: {df_ds}")
                continue
            good_jobs.append((dataset, method, comp, cal_path, oob_path, df_ds))

        if good_jobs:
            from joblib import Parallel, delayed
            n_jobs = _resolve_n_jobs(-1)
            print(f"  Building {len(good_jobs)} variant table(s) from scoreset "
                  f"across {n_jobs} worker(s)...")
            # Each `delayed(...)` call below is passed *only* that job's own
            # (small) df_ds as a positional argument -- joblib pickles each
            # call's arguments independently, so every worker receives just
            # its one dataset's slice, never the other datasets' data.
            results = Parallel(n_jobs=n_jobs, verbose=5)(
                delayed(_run_scoreset_job)(dataset, method, comp, cal_path, df_ds, oob_path)
                for dataset, method, comp, cal_path, oob_path, df_ds in good_jobs
            )
            for dataset, method, comp, df, err in results:
                if err is not None:
                    print(f"  WARNING: scoreset build failed for {dataset} / {method}: {err}")
                    continue
                print(f"  Built variant table for {dataset} / {method} from scoreset")
                pending.append((dataset, method, comp, df))

    from analysis.plot_common import sample_matches
    for dataset, method, comp, df in pending:
        n_controls = (
            sample_matches(df, "Pathogenic/Likely Pathogenic").sum()
            + sample_matches(df, "Benign/Likely Benign").sum()
        )
        if n_controls < min_controls:
            print(f"  SKIP {dataset} / {method}: only {n_controls} controls "
                  f"(< min_controls {min_controls})")
            continue

        df["dataset"] = dataset
        df["method"] = method
        df["component"] = comp
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Prior override helpers
# ---------------------------------------------------------------------------

# Point-based (discretizing) methods only -- "acmg_bayes" (like the legacy
# "continuous" it superseded) uses calculate_classification_ranges, a
# different mechanism, and is not supported here.
_PRIOR_OVERRIDE_METHODS = {"tavtigian", "piecewise", "strict_additive"}


def parse_prior_overrides(raw: List[str]) -> Dict[str, float]:
    """Parse ['tavtigian=0.1', 'piecewise=0.5'] → {'tavtigian': 0.1, 'piecewise': 0.5}."""
    result = {}
    for item in raw:
        try:
            method, val = item.split("=", 1)
            result[method.strip()] = float(val.strip())
        except ValueError:
            raise ValueError(
                f"Invalid prior-override entry {item!r}; expected 'method=value' (e.g. tavtigian=0.1)"
            )
    return result


def recompute_points_with_prior_overrides(
    df: pd.DataFrame,
    output_dir: Path,
    prior_overrides: Dict[str, float],
) -> pd.DataFrame:
    """Recompute standard_points using hardcoded priors loaded from lr_values.json.gz.

    For each (method, dataset) where method has an entry in prior_overrides:
      1. Load the lr_values JSON to get the bootstrap LR+ curves and score range.
      2. Call calculate_score_ranges with the new prior to get updated point_ranges.
      3. Re-assign standard_points for every variant in that group.
      4. Clear oob_points (set to NaN) so downstream analysis uses the recomputed values.

    Only methods in _PRIOR_OVERRIDE_METHODS (tavtigian, piecewise, strict_additive) are
    supported; acmg_bayes (and the legacy "continuous" it superseded) uses a
    different mechanism.
    """
    import numpy as np
    from src.assay_calibration.fit_utils.fit import calculate_score_ranges
    from src.assay_calibration.plot_utils.utils import flatten_point_ranges, assign_points
    from analysis.calibration_plots import load_lr_values

    POINT_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]

    df = df.copy()

    for method, new_prior in prior_overrides.items():
        if method not in _PRIOR_OVERRIDE_METHODS:
            print(f"  Prior override: method '{method}' not supported (supported: "
                  f"{sorted(_PRIOR_OVERRIDE_METHODS)}) — skipping")
            continue
        if not (0 < new_prior < 1):
            print(f"  Prior override: prior {new_prior} for '{method}' must be in (0, 1) — skipping")
            continue

        mask = df["method"] == method
        if not mask.any():
            print(f"  Prior override: no variants for method '{method}' — skipping")
            continue

        n_updated = 0
        for dataset in sorted(df.loc[mask, "dataset"].unique()):
            ds_mask = mask & (df["dataset"] == dataset)
            comp = df.loc[ds_mask, "component"].iloc[0]

            lr_data = load_lr_values(output_dir, dataset, method, comp)
            if lr_data is None:
                print(f"  Prior override: lr_values not found for {dataset}/{method}/{comp} — skipping")
                continue

            score_range = lr_data["score_range"]
            orig_prior = float(lr_data["prior"])

            lr_5_log = lr_data["log_lr_pct"][0]
            lr_95_log = lr_data["log_lr_pct"][2]

            try:
                pr_p, pr_b, _ = calculate_score_ranges(
                    lr_5_log, lr_95_log, new_prior, score_range, POINT_VALUES,
                    acmg_mapping_method=method,
                )
            except Exception as e:
                print(f"  Prior override: calculate_score_ranges failed for {dataset}/{method}: {e}")
                continue

            flat = flatten_point_ranges({**pr_p, **pr_b})

            scores = df.loc[ds_mask, "score"].values
            df.loc[ds_mask, "standard_points"] = [assign_points(s, flat) for s in scores]
            if "oob_points" in df.columns:
                df.loc[ds_mask, "oob_points"] = np.nan

            n_updated += int(ds_mask.sum())
            print(f"  Prior override: {dataset}/{method} prior {orig_prior:.4f} → {new_prior:.4f} "
                  f"({int(ds_mask.sum())} variants)")

        print(f"  Prior override summary: method='{method}' prior={new_prior} "
              f"— {n_updated} variants updated")

    return df
