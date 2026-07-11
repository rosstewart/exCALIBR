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

KNOWN_METHODS = {"tavtigian", "piecewise", "continuous", "strict_additive"}

# Excluded by default from analysis unless include_all=True
DEFAULT_EXCLUDED_DATASETS = {
    "F9_Popp_2025_model",
    "TP53_Fayer_2021_meta_clinvar_2018",
    "SFPQ_IGVF",
}

# benign_method suffix attached directly after n_c in output filenames, e.g.
# "..._3c_avg_calibration.json" — part of the component token ("3c_avg"), not the method.
BENIGN_METHODS = {"avg", "benign", "synonymous"}


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


def _recompute_points_from_calibration(
    df: pd.DataFrame,
    calibration_path: Path,
) -> pd.DataFrame:
    """Recompute standard_points from score + calibration JSON point_ranges.

    Used when no method-specific variants CSV exists but the calibration JSON
    and a shared (default) variants CSV do.
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
) -> pd.DataFrame:
    """Load all variants CSVs into a single long-format DataFrame.

    Columns: variant_id, score, sample, standard_points,
             dataset, method, component
    (plus oob_* columns when present)

    When a method-specific variants CSV is absent but a calibration JSON exists
    alongside a shared (default/no-method) variants CSV, standard_points are
    recomputed from scores + calibration point_ranges so no regeneration is needed.

    recompute_points : if True, standard_points is always recomputed from
        score + the resolved (dataset, method, comp) calibration JSON, even when
        a method-specific variants CSV exists on disk.

    Unless include_all=True, datasets in DEFAULT_EXCLUDED_DATASETS are skipped, and
    (when dataset_configs is provided) any dataset absent from dataset_configs is
    skipped too.
    """
    frames = []
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

        # Find the shared (no-method) variants CSV for fallback recomputation.
        default_csv = method_dict.get("default")
        if default_csv is None:
            base_nc = comp.split("_", 1)[0]
            other_csvs = [
                (c, md["default"]) for c, md in comp_dict.items()
                if c != comp and md.get("default") is not None
            ]
            other_csvs.sort(key=lambda cp: cp[0].split("_", 1)[0] != base_nc)
            if other_csvs:
                default_csv = other_csvs[0][1]

        for method, path in method_dict.items():
            if methods_filter and method not in methods_filter:
                continue

            if path is not None and not recompute_points:
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    print(f"  WARNING: could not read {path}: {e}")
                    continue
            elif path is not None and recompute_points:
                cal_path = (calibrations or {}).get(dataset, {}).get(method, {}).get(comp)
                if cal_path is None:
                    try:
                        df = pd.read_csv(path)
                    except Exception as e:
                        print(f"  WARNING: could not read {path}: {e}")
                        continue
                else:
                    try:
                        base_df = pd.read_csv(path)
                        df = _recompute_points_from_calibration(base_df, cal_path)
                        print(f"  Recomputed standard_points for {dataset} / {method} from {cal_path.name}")
                    except Exception as e:
                        print(f"  WARNING: recomputation failed for {dataset} / {method}: {e}")
                        continue
            else:
                cal_path = (calibrations or {}).get(dataset, {}).get(method, {}).get(comp)
                if cal_path is None or default_csv is None:
                    print(f"  SKIP {dataset} / {method}: no variants CSV and no calibration fallback")
                    continue
                try:
                    base_df = pd.read_csv(default_csv)
                    df = _recompute_points_from_calibration(base_df, cal_path)
                    print(f"  Recomputed standard_points for {dataset} / {method} from {cal_path.name}")
                except Exception as e:
                    print(f"  WARNING: recomputation failed for {dataset} / {method}: {e}")
                    continue

            from analysis.plot_common import sample_matches
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
    supported; continuous scoring uses a different mechanism.
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
