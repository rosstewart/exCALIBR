#!/usr/bin/env python
"""
Pipeline-native analysis for run_igvf_batch.py / run_pipeline.py outputs.

Discovers *_variants.csv files written by the pipeline, builds confusion
matrices and evidence-distribution arrays, then generates three figures:

  confusion_heatmap_<m1>_vs_<m2>.png  – aggregate confusion matrices
  evidence_distribution_<method>.png  – combined author + sample panels
  per_gene_scatter_<m1>_vs_<m2>.png  – per-gene accuracy scatter

All figures match the style of test/plot_author_calibration_confusion.py.
No pre-existing files are required beyond the pipeline output directory.

Example
-------
# Compare tavtigian vs piecewise from a batch run
python analysis/analyze_pipeline_output.py \\
    --output-dir ./igvf_output \\
    --dataset data/integrated_variant_effect_dataset.tsv.gz \\
    --dataset-configs src/igvf_configs/dataset_configs_jan_2026.json \\
    --figure-dir ./igvf_output/figures

# Single-method run (no comparison figures)
python analysis/analyze_pipeline_output.py \\
    --output-dir ./calibration_output \\
    --dataset data/integrated_variant_effect_dataset.tsv.gz
"""
import os
import sys
import re
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_METHODS = {"tavtigian", "piecewise", "continuous", "strict_additive"}

# Excluded by default from analysis unless --include-all is passed
DEFAULT_EXCLUDED_DATASETS = {
    "F9_Popp_2025_model",
    "TP53_Fayer_2021_meta_clinvar_2018",
    "SFPQ_IGVF",
}

# benign_method suffix attached directly after n_c in output filenames, e.g.
# "..._3c_avg_calibration.json" — part of the component token ("3c_avg"), not the method.
BENIGN_METHODS = {"avg", "benign", "synonymous"}

# Map sample column values → ClinVar class integers
# 0=B/LB, 1=VUS, 2=P/LP, 3=gnomAD, 4=Synonymous
SAMPLE_TO_CLINVAR = {
    "Benign/Likely Benign":          0,
    "Pathogenic/Likely Pathogenic":  2,
    "gnomAD":                        3,
    "Synonymous":                    4,
}

# Map author label strings → integers used by plot_combined_evidence_distributions
# 0=Normal, 1=Indeterminate, 2=Abnormal
AUTH_LABEL_MAP = {
    "NORMAL":        0,
    "ABNORMAL":      2,
    "INDETERMINATE": 1,
    "NOT SPECIFIED": 1,
}


# ---------------------------------------------------------------------------
# Discovery
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
    # point_ranges are stored in raw score space — scoreset_flipped only
    # affected which side of the range monotonicity constraints were built
    # from (see visualize.py / variant_evidence.py's _build_standard_table,
    # which assigns points from raw scoreset.scores with no sign flip).
    df["standard_points"] = [assign_points(s, point_ranges) for s in scores]
    # oob_points was computed with the original method's calibration — drop it
    # so the caller doesn't silently use stale OOB values from another method
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
        Populated from *_calibration.json files.  Used to recompute
        standard_points when no method-specific variants CSV exists.
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
        a method-specific variants CSV exists on disk. Use this when configs
        (n_c/benign_method/liberal_monotonicity/etc.) have changed since the CSVs
        were last generated, so the CSVs' baked-in standard_points may be stale —
        recomputing is cheap and doesn't require re-running the bootstrap pipeline.
        Any oob_points column is dropped when recomputing, since it was computed
        under the old calibration and recomputing it needs the original bootstrap
        fits (not available here) — see --in-bag, which doesn't use oob_points.

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
                print(f"  SKIP {dataset}: in DEFAULT_EXCLUDED_DATASETS (use --include-all to override)")
                continue
            if dataset_configs is not None and dataset not in dataset_configs:
                print(f"  SKIP {dataset}: not present in --dataset-configs (use --include-all to override)")
                continue

        comp = resolve_component(dataset, list(comp_dict.keys()), model_selections, dataset_configs)
        method_dict = comp_dict[comp]

        # Find the shared (no-method) variants CSV for fallback recomputation.
        # Variants CSVs (scores/samples) don't depend on n_c or benign_method — only
        # the calibration point_ranges do — but the pipeline only writes one copy,
        # under whichever (n_c, benign_method) combo happened to run first. So if
        # the resolved comp has no "default" CSV of its own, search other comps for
        # this dataset, preferring ones that share the resolved comp's n_c.
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
                # Method-specific CSV exists — load directly
                try:
                    df = pd.read_csv(path)
                except Exception as e:
                    print(f"  WARNING: could not read {path}: {e}")
                    continue
            elif path is not None and recompute_points:
                # CSV exists but its baked-in standard_points may be stale
                # relative to the current config — recompute from score + calibration.
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
                # No method-specific CSV — try recomputing from calibration + default CSV
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

            n_controls = (
                (df["sample"] == "Pathogenic/Likely Pathogenic").sum()
                + (df["sample"] == "Benign/Likely Benign").sum()
            )
            if n_controls < min_controls:
                print(f"  SKIP {dataset} / {method}: only {n_controls} controls "
                      f"(< --min-controls {min_controls})")
                continue

            df["dataset"] = dataset
            df["method"] = method
            df["component"] = comp
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Author label loading (optional, requires --dataset)
# ---------------------------------------------------------------------------

def _clinvar_release_for(dataset: str, calibration_dir: Optional[Path] = None) -> str:
    """Infer ClinVar release from dataset name, or read from calibration JSON."""
    if "_clinvar_2018" in dataset and "not_clinvar_2018" not in dataset:
        return "2018"
    genes_2018 = {"BRCA1", "MSH2", "PTEN", "TP53"}
    gene = dataset.split("_")[0]
    if gene in genes_2018:
        return "2018"
    return "2026"


def _load_name_mapping(dataset_tsv: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load new_dataset_names.csv near the dataset file (checks same dir then parent).

    Returns (old_to_new, new_to_old) where:
      old_to_new: reported name  → CSV name   (e.g. BRCA2_unpublished → BRCA2_IGVF)
      new_to_old: CSV name       → reported name
    """
    p = Path(dataset_tsv).parent
    candidates = [p / "new_dataset_names.csv", p.parent / "new_dataset_names.csv"]
    names_csv = next((c for c in candidates if c.exists()), None)
    if names_csv is None:
        return {}, {}
    df_n = pd.read_csv(names_csv)
    old_to_new = dict(zip(df_n["Old_names"], df_n["New_names"]))
    new_to_old = dict(zip(df_n["New_names"], df_n["Old_names"]))
    return old_to_new, new_to_old


def load_author_labels_for_dataset(
    df_full: pd.DataFrame,
    dataset: str,
    clinvar_release: str,
    old_to_new: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Load Scoreset for one dataset and return {variant_id → auth_label}.

    Looks up dataset in df_full by its name first; if not found, tries the
    new CSV name via old_to_new mapping (e.g. BRCA2_unpublished → BRCA2_IGVF).

    _LEGACY_INTERNAL_ALIASES with None value → skip entirely.
    Non-None aliases apply only to fit/calibration lookup, not here.
    """
    base = dataset.replace("_clinvar_2018", "")

    # None-aliased datasets are excluded (no valid data / composite models)
    if base in _LEGACY_INTERNAL_ALIASES and _LEGACY_INTERNAL_ALIASES[base] is None:
        print(f"  SKIP author labels for {dataset}: excluded dataset")
        return {}

    # Try the name as-is first, then fall back via old→new mapping
    csv_name = base
    df_ds = df_full[df_full["Dataset"] == csv_name].copy()
    if df_ds.empty and old_to_new:
        alt_name = old_to_new.get(csv_name)
        if alt_name:
            df_ds = df_full[df_full["Dataset"] == alt_name].copy()
            if not df_ds.empty:
                csv_name = alt_name
    if df_ds.empty:
        print(f"  SKIP author labels for {dataset}: '{base}' not found in dataset CSV")
        return {}
    df_ds["Dataset"] = dataset

    try:
        from src.assay_calibration.pipeline.config import PipelineConfig
        from src.assay_calibration.pipeline.utils import load_dataset_from_df
        from src.assay_calibration.pipeline.variant_evidence import _get_variant_ids

        config = PipelineConfig(
            dataset_csv="",
            dataset_name=dataset,
            output_dir="/tmp",
            clinvar_release=clinvar_release,
        )
        scoreset = load_dataset_from_df(df_ds, config)
        ids = _get_variant_ids(scoreset)
        return {vid: lbl for vid, lbl in zip(ids, scoreset.auth_labels)}
    except Exception as e:
        print(f"  WARNING: author labels failed for {dataset}: {e}")
        return {}


def attach_author_labels(
    df: pd.DataFrame,
    dataset_tsv: str,
) -> pd.DataFrame:
    """Add 'auth_label' column to df by loading Scoreset for each dataset."""
    print("\nLoading author labels from Scoreset...")
    sep = "\t" if dataset_tsv.endswith((".tsv", ".tsv.gz")) else ","
    df_full = pd.read_csv(dataset_tsv, sep=sep, low_memory=False)

    old_to_new, _ = _load_name_mapping(dataset_tsv)
    if old_to_new:
        print(f"  Loaded name mapping ({len(old_to_new)} entries) for CSV fallback lookup")

    label_maps: Dict[str, Dict[str, str]] = {}
    for dataset in df["dataset"].unique():
        cr = _clinvar_release_for(dataset)
        label_maps[dataset] = load_author_labels_for_dataset(
            df_full, dataset, cr, old_to_new=old_to_new
        )

    def _lookup(row):
        return label_maps.get(row["dataset"], {}).get(row["variant_id"])

    df = df.copy()
    df["auth_label"] = df.apply(_lookup, axis=1)
    n_labeled = df["auth_label"].notna().sum()
    print(f"  Author labels attached: {n_labeled}/{len(df)} variants")
    return df


# ---------------------------------------------------------------------------
# Confusion matrix construction
# ---------------------------------------------------------------------------

def _effective_points(df_sub: pd.DataFrame, use_oob: bool) -> pd.Series:
    """Return the best available points per variant.

    When use_oob=True: use oob_points when not NaN, fall back to standard_points.
    When use_oob=False: always use standard_points.
    """
    if not use_oob or "oob_points" not in df_sub.columns:
        return df_sub["standard_points"]
    has_oob = df_sub["oob_points"].notna()
    result = df_sub["standard_points"].copy()
    result[has_oob] = df_sub.loc[has_oob, "oob_points"]
    return result


def build_confusion_matrix(df_sub: pd.DataFrame, use_oob: bool = True) -> Optional[pd.DataFrame]:
    """Build a 2×3 DataFrame from P/LP and B/LB variants.

    Rows: [BLB, PLP]   Cols: [Normal, IR, Abnormal]
    (matches the format expected by plot_aggregate_confusion_matrices)

    When use_oob=True (default), uses oob_points per variant where available,
    falling back to standard_points.  Pass use_oob=False (--in-bag) to always
    use standard_points.
    """
    df_plp = df_sub[df_sub["sample"] == "Pathogenic/Likely Pathogenic"]
    df_blb = df_sub[df_sub["sample"] == "Benign/Likely Benign"]

    plp = _effective_points(df_plp, use_oob)
    blb = _effective_points(df_blb, use_oob)

    if len(plp) == 0 and len(blb) == 0:
        return None

    def _counts(pts):
        if len(pts) == 0:
            return [0, 0, 0]
        return [int((pts < 0).sum()), int((pts == 0).sum()), int((pts > 0).sum())]

    mat = pd.DataFrame(
        [_counts(blb), _counts(plp)],
        index=["BLB", "PLP"],
        columns=["Normal", "IR", "Abnormal"],
    )
    return mat


def build_author_confusion_matrix(df_sub: pd.DataFrame, use_oob: bool = True) -> Optional[pd.DataFrame]:
    """Build a 2×3 DataFrame from variants with Normal / Abnormal author labels.

    Rows: [Normal, Abnormal]   Cols: [Normal, IR, Abnormal]
    """
    if "auth_label" not in df_sub.columns:
        return None

    upper = df_sub["auth_label"].str.upper().fillna("INDETERMINATE")
    df_normal = df_sub[upper == "NORMAL"]
    df_abnormal = df_sub[upper == "ABNORMAL"]

    normal = _effective_points(df_normal, use_oob)
    abnormal = _effective_points(df_abnormal, use_oob)

    if len(normal) == 0 and len(abnormal) == 0:
        return None

    def _counts(pts):
        if len(pts) == 0:
            return [0, 0, 0]
        return [int((pts < 0).sum()), int((pts == 0).sum()), int((pts > 0).sum())]

    mat = pd.DataFrame(
        [_counts(normal), _counts(abnormal)],
        index=["Normal", "Abnormal"],
        columns=["Normal", "IR", "Abnormal"],
    )
    return mat


# ---------------------------------------------------------------------------
# Evidence array construction
# ---------------------------------------------------------------------------

def build_evidence_arrays(
    df_method: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (all_danz, all_clinvar) for one method's variants.

    all_danz   : int array of standard_points
    all_clinvar: int array of ClinVar class (0=B/LB,1=VUS,2=P/LP,3=gnomAD,4=Syn)
    """
    all_danz = df_method["standard_points"].values.astype(int)
    all_clinvar = df_method["sample"].map(SAMPLE_TO_CLINVAR).fillna(1).values.astype(int)
    return all_danz, all_clinvar


def build_author_array(df_method: pd.DataFrame) -> Optional[np.ndarray]:
    """Return int array of author labels (0/1/2) or None if unavailable."""
    if "auth_label" not in df_method.columns:
        return None
    upper = df_method["auth_label"].str.upper().fillna("INDETERMINATE")
    arr = upper.map(AUTH_LABEL_MAP).fillna(1).values.astype(int)
    if np.all(arr == 1):
        return None
    return arr


# ---------------------------------------------------------------------------
# Legacy mode (--legacy flag)
# Mirrors the loading infrastructure from test/plot_author_calibration_confusion.py
# ---------------------------------------------------------------------------

_LEGACY_REPORTED_LIST = [
    "BRCA1_Findlay_2018", "BRCA2_Hu_2024", "VHL_Buckley_2024",
    "JAG1_Gilbert_2024", "BARD1_unpublished", "PALB2_unpublished",
    "RAD51D_unpublished", "CTCF_unpublished",
    "BAP1_Waters_2024", "DDX3X_Radford_2023_cLFC_day15",
    "RHO_Wan_2019", "RAD51C_Olvera-León_2024_z_score_D4_D14",
    "FKRP_Ma_2024", "LARGE1_Ma_2024",
    "CARD11_Meitlis_2020_Ibrutinib_no_introns",
    "CARD11_Meitlis_2020_DMSO_no_introns",
    "ASPA_Grønbæk-Thygesen_2024_abundance",
    "ASPA_Grønbæk-Thygesen_2024_toxicity",
    "BRCA1_Adamovich_2022_Cisplatin", "BRCA1_Adamovich_2022_HDR",
    "CHEK2_Gebbia_2024", "CRX_Shepherdson_2024",
    "G6PD_unpublished", "GCK_Gersing_2023_complementation",
    "GCK_Gersing_2024_abundance", "KCNE1_Muhammad_2024_absence_of_WT",
    "KCNE1_Muhammad_2024_potassium_flux",
    "KCNE1_Muhammad_2024_presence_of_WT",
    "KCNH2_Jiang_2022",
    "KCNH2_Kozek_Glazer_2020", "KCNH2_O_Neill_2024_surface_expression",
    "MSH2_Scott_2022", "NDUFAF6_Sung_2024", "OTC_Lo_2023", "PTEN_Matreyek_2018",
    "PTEN_Mighell_2018", "SCN5A_Glazer_2020",
    "SCN5A_Ma_2024_current_density", "SGCB_Li_2023",
    "TSC2_combined_unpublished",
    "KCNQ4_Zheng_2022_current_homozygous",
    "KCNQ4_Zheng_2022_v12_homozygous",
]

_LEGACY_INTERNAL_ALIASES: Dict[str, Optional[str]] = {
    "F9_Popp_2025_model": None,
    "MSH2_Scott_2022": "MSH2_Jia_2021",
}

_LEGACY_GENES_2018 = {"BRCA1", "MSH2", "PTEN", "TP53"}


def _remove_prefix_with_var_id(s: str) -> str:
    return re.sub(r"^.*?_var\d+_", "", s)


def _legacy_make_variant_id(v) -> str:
    return f"{v.Gene}_{v.Chrom}_{v.hgvs_c}"


def _build_oob_from_variants_csvs(
    output_dir: Path,
) -> Dict[Tuple[str, Optional[str]], Dict[str, Dict]]:
    """Scan output_dir for *_variants.csv and build {(dataset, method): {variant_id: {"points": val}}}.

    Keys are (dataset_name, method) tuples so tavtigian and piecewise OOB data are kept
    separately.  method is None for CSVs with no method token in the filename.
    Only rows where oob_points is not NaN are included.
    """
    result: Dict[Tuple[str, Optional[str]], Dict[str, Dict]] = {}
    for csv_path in sorted(output_dir.rglob("*_variants.csv")):
        parsed = _parse_variants_stem(csv_path.stem)
        if parsed is None:
            continue
        dataset_name, method, _ = parsed
        key = (dataset_name, method)
        if key in result:
            continue
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in ("variant_id", "oob_points"), low_memory=False)
        except Exception:
            continue
        if "oob_points" not in df.columns or "variant_id" not in df.columns:
            continue
        oob_rows = df[df["oob_points"].notna()]
        if oob_rows.empty:
            continue
        result[key] = {
            _remove_prefix_with_var_id(str(row["variant_id"])): {"points": float(row["oob_points"])}
            for _, row in oob_rows.iterrows()
        }
    return result


def _find_calibration_in_output(
    output_dir: Path, dataset_name: str, method: Optional[str] = None
) -> Optional[Path]:
    """Find a calibration JSON for dataset_name in output_dir.

    If method is given, require that token in the filename.
    Otherwise prefer tavtigian, then any match.
    """
    candidates = sorted(output_dir.rglob(f"{dataset_name}*_calibration.json"))
    if not candidates:
        return None
    if method:
        for p in candidates:
            if method in p.name:
                return p
        return None
    for p in candidates:
        if "tavtigian" in p.name:
            return p
    return candidates[0]


def load_legacy_context(args) -> Dict:
    """Load legacy data: integrated dataset from --legacy-dataset, OOB and calibrations from --output-dir."""
    output_dir = Path(args.output_dir)

    print("\nLoading legacy configurations...")
    try:
        from src.assay_calibration.plot_utils.utils import import_dataset_configurations
        _, _, new_dataset_configs, keep_old_list = import_dataset_configurations()
    except Exception as e:
        print(f"  WARNING: import_dataset_configurations failed: {e}")
        new_dataset_configs, keep_old_list = {}, set()

    dataset_path = Path(args.dataset) if getattr(args, "dataset", None) else None
    if dataset_path is None:
        raise ValueError(
            "--dataset is required in legacy mode. "
            "Provide the path to integrated_variant_effect_dataset_analysis.csv.gz"
        )
    print(f"\nLoading legacy dataset from {dataset_path}...")
    df_final = pd.read_csv(dataset_path, low_memory=False)
    print(f"  Loaded {len(df_final):,} rows, {df_final['Dataset'].nunique()} datasets")

    # Load old→new name mapping (reported name → CSV name, e.g. BRCA2_unpublished → BRCA2_IGVF).
    # df_final keeps its native CSV names (new names); we translate at lookup time.
    old_to_new, _ = _load_name_mapping(str(dataset_path))
    if old_to_new:
        print(f"  Loaded name mapping ({len(old_to_new)} entries)")

    # OOB points: prefer pickle if provided, else derive from variants CSVs.
    # Always stored as {(dataset, method_or_None): {variant_id: {"points": val}}}.
    # Pickle is method-agnostic so stored under method=None; CSV data uses the actual method.
    oob_pkl = getattr(args, "legacy_oob_pkl", None)
    if oob_pkl:
        import pickle
        print(f"\nLoading OOB points from pickle {oob_pkl}...")
        with open(oob_pkl, "rb") as f:
            raw_pkl = pickle.load(f)
        variant_to_oob_points = {}
        for ds, vmap in raw_pkl.items():
            if vmap is not None:
                variant_to_oob_points[(ds, None)] = {
                    _remove_prefix_with_var_id(k): v for k, v in vmap.items()
                }
        print(f"  Loaded OOB data for {len(variant_to_oob_points)} datasets")
    else:
        print(f"\nBuilding OOB points from variants CSVs in {output_dir}...")
        variant_to_oob_points = _build_oob_from_variants_csvs(output_dir)
        print(f"  Found OOB data for {len(variant_to_oob_points)} (dataset, method) pairs")

    legacy_calibrations_dir = getattr(args, "legacy_calibrations_dir", None)
    if legacy_calibrations_dir:
        legacy_calibrations_dir = Path(legacy_calibrations_dir)
        print(f"  Using reference calibrations from {legacy_calibrations_dir}")

    return {
        "new_dataset_configs": new_dataset_configs,
        "keep_old_list": keep_old_list,
        "df_final": df_final,
        "variant_to_oob_points": variant_to_oob_points,
        "output_dir": output_dir,
        "old_to_new": old_to_new,
        "legacy_calibrations_dir": legacy_calibrations_dir,
    }


def _build_legacy_oob_confusion_matrix(
    dataset_std_name: str,
    legacy_ctx: Dict,
    method: Optional[str] = None,
    use_oob: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build confusion matrices for one dataset (OOB or in-bag).

    When use_oob=True (default): uses OOB points per variant where available,
    falling back to calibration thresholds.
    When use_oob=False: always uses calibration thresholds (in-bag scoring).

    Mirrors calculate_confusion_mat_oob from test/plot_author_calibration_confusion.py.
    Uses OOB points where available, falls back to in-bag calibration thresholds.
    If method is given, only the calibration for that method is used.
    """
    from src.assay_calibration.data_utils.dataset import Scoreset
    from src.assay_calibration.plot_utils.utils import flatten_point_ranges, assign_points

    df_final = legacy_ctx["df_final"]
    variant_to_oob_points_all = legacy_ctx["variant_to_oob_points"]
    output_dir = Path(legacy_ctx["output_dir"])
    old_to_new = legacy_ctx.get("old_to_new", {})

    # Handle aliases
    dataset = dataset_std_name
    if dataset in _LEGACY_INTERNAL_ALIASES:
        alias = _LEGACY_INTERNAL_ALIASES[dataset]
        if alias is None:
            return None, None
        dataset = alias

    # 2018 gene handling
    gene = dataset.split("_")[0]
    use_2018 = gene in _LEGACY_GENES_2018

    # File/OOB lookups use the pipeline output name (old name, as files were generated).
    # Scott_2022 → Jia_2021 alias applies at the file level only.
    pipeline_base = dataset.replace("_clinvar_2018", "").replace("Scott_2022", "Jia_2021")
    cal_key = pipeline_base + "_clinvar_2018" if use_2018 else pipeline_base

    legacy_calibrations_dir = legacy_ctx.get("legacy_calibrations_dir")
    if legacy_calibrations_dir:
        ref_cal = legacy_calibrations_dir / f"{cal_key}.json"
        calibration_f = ref_cal if ref_cal.exists() else None
    else:
        calibration_f = None
    if calibration_f is None:
        calibration_f = _find_calibration_in_output(output_dir, cal_key, method=method)
    if calibration_f is None:
        print(f"  SKIP {dataset_std_name}: calibration not found for key '{cal_key}'"
              + (f" method='{method}'" if method else ""))
        return None, None

    with open(calibration_f) as f:
        calibration_data = json.load(f)
    if calibration_data.get("point_ranges") is None:
        return None, None
    point_ranges = flatten_point_ranges(calibration_data["point_ranges"])
    scoreset_flipped = calibration_data.get("scoreset_flipped", False)

    # OOB dict key: same old pipeline name (CSVs were generated with old names).
    # variant_to_oob_points_all is keyed by (dataset, method) tuples (or (dataset, None)
    # for pickle-sourced data). Try method-specific first, then method-agnostic.
    oob_base = dataset_std_name.replace("Scott_2022", "Jia_2021")
    oob_key = oob_base + "_clinvar_2018" if (use_2018 and not oob_base.endswith("_clinvar_2018")) else oob_base
    variant_to_oob = (
        variant_to_oob_points_all.get((oob_key, method))
        or variant_to_oob_points_all.get((oob_key, None))
        or {}
    )

    # df_final lookup: translate old name → new CSV name (df_final uses new names)
    csv_name = old_to_new.get(pipeline_base, pipeline_base)
    clinvar_rel = "2018" if use_2018 else "2025"
    df_ds = df_final[df_final["Dataset"] == csv_name]
    try:
        scoreset = Scoreset(df_ds, clinvar_release=clinvar_rel, synonymous_exclusive=False)
    except Exception as e:
        print(f"  SKIP {dataset_std_name}: Scoreset error — {e}")
        return None, None

    sample_names = [s[1] for s in scoreset.samples]
    if "Pathogenic/Likely Pathogenic" not in sample_names or "Benign/Likely Benign" not in sample_names:
        return None, None

    # Determine whether auth matrix is usable (all indeterminate/NaN → None auth)
    _ind_codes = {"NOT SPECIFIED", "INDETERMINATE", "IGNORE", "NAN"}
    _auth_upper = [str(x).upper() if not pd.isna(x) else "NAN" for x in scoreset.auth_labels]
    _auth_valid = not all(u in _ind_codes for u in _auth_upper)

    # Build kept variant id list aligned to scoreset.scores indices
    variants_by_id = scoreset.get_variants_by_id()
    kept_variant_ids = []
    for idx, (_, variants) in enumerate(variants_by_id.items()):
        if scoreset._keep_mask[idx]:
            kept_variant_ids.append(_legacy_make_variant_id(variants[0]))

    plp_mask = scoreset.sample_assignments[:, 0]
    blb_mask = scoreset.sample_assignments[:, 1]

    blb_in_benign = blb_in_ir = blb_in_path = 0
    plp_in_benign = plp_in_ir = plp_in_path = 0

    for idx in range(len(scoreset.scores)):
        vid = kept_variant_ids[idx]
        if use_oob and vid in variant_to_oob:
            points = variant_to_oob[vid]["points"]
        else:
            points = assign_points(scoreset.scores[idx], point_ranges)

        if blb_mask[idx]:
            if points <= -1:
                blb_in_benign += 1
            elif points >= 1:
                blb_in_path += 1
            else:
                blb_in_ir += 1
        if plp_mask[idx]:
            if points <= -1:
                plp_in_benign += 1
            elif points >= 1:
                plp_in_path += 1
            else:
                plp_in_ir += 1

    danz = pd.DataFrame(
        {"BLB": [blb_in_benign, blb_in_ir, blb_in_path],
         "PLP": [plp_in_benign, plp_in_ir, plp_in_path]},
    ).T
    danz.columns = ["Normal", "IR", "Abnormal"]

    # Author confusion matrix
    plp_auth = scoreset.auth_labels[plp_mask]
    blb_auth = scoreset.auth_labels[blb_mask]
    ind_codes = ["NOT SPECIFIED", "INDETERMINATE", "IGNORE"]

    def _safe_upper(arr):
        return np.array([str(x).upper() if not pd.isna(x) else "NAN" for x in arr])

    blb_u = _safe_upper(blb_auth)
    plp_u = _safe_upper(plp_auth)

    auth = pd.DataFrame(
        {
            "BLB": [
                (blb_u == "NORMAL").sum(),
                (np.isin(blb_u, ind_codes) | pd.isna(blb_auth)).sum(),
                (blb_u == "ABNORMAL").sum(),
            ],
            "PLP": [
                (plp_u == "NORMAL").sum(),
                (np.isin(plp_u, ind_codes) | pd.isna(plp_auth)).sum(),
                (plp_u == "ABNORMAL").sum(),
            ],
        }
    ).T
    auth.columns = ["Normal", "IR", "Abnormal"]

    return danz, (auth if _auth_valid else None)


def run_legacy_analysis(args, figure_dir: Path):
    """Run full legacy analysis following test/plot_author_calibration_confusion.py.

    Generates the same three figures as the pipeline mode but reads from
    /data/ross/assay_calibration/ (pre-existing calibrations, OOB pickle, etc.)
    instead of a fresh pipeline output directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 80)
    print("LEGACY MODE")
    print("=" * 80)

    legacy_ctx = load_legacy_context(args)
    new_dataset_configs = legacy_ctx["new_dataset_configs"]
    keep_old_list = legacy_ctx["keep_old_list"]
    df_final = legacy_ctx["df_final"]
    variant_to_oob_points = legacy_ctx["variant_to_oob_points"]

    # Determine which datasets to process
    reported_list = list(args.datasets) if args.datasets else _LEGACY_REPORTED_LIST
    print(f"\nProcessing {len(reported_list)} datasets...")

    old_to_new = legacy_ctx.get("old_to_new", {})

    # Determine which methods to run; fall back to a single no-method-filter pass
    methods_to_run: List[Optional[str]] = getattr(args, "methods", None) or [None]

    # conf_by_method[method]  = (oob_danzs,   ds_names)  — ClinVar OOB,   ALL datasets
    # inbag_by_method[method] = (inbag_danzs,  ds_names)  — ClinVar in-bag, ALL datasets
    # auth_by_method[method]  = (oob_danzs, auths, auth_names) — author-paired subset only
    conf_by_method: Dict[Optional[str], Tuple[List, List[str]]] = {}
    inbag_by_method: Dict[Optional[str], Tuple[List, List[str]]] = {}
    auth_by_method: Dict[Optional[str], Tuple[List, List, List[str]]] = {}
    for method in methods_to_run:
        label = method or "default"
        print(f"\n  Method: {label}")
        oob_danzs: List[pd.DataFrame] = []
        inbag_danzs: List[pd.DataFrame] = []
        auth_oob_danzs: List[pd.DataFrame] = []
        auths: List[pd.DataFrame] = []
        ds_names: List[str] = []
        auth_names: List[str] = []
        total_fn_oob = 0
        total_fn_ib = 0
        total_fp_oob = 0
        total_fp_ib = 0
        fn_records: List[Tuple] = []
        fp_records: List[Tuple] = []
        for ds in reported_list:
            danz_oob, auth = _build_legacy_oob_confusion_matrix(ds, legacy_ctx, method=method, use_oob=True)
            danz_inbag, _ = _build_legacy_oob_confusion_matrix(ds, legacy_ctx, method=method, use_oob=False)
            if danz_oob is None:
                print(f"    Skipped {ds}")
                continue
            name = old_to_new.get(ds, ds)
            ds_names.append(name)
            oob_danzs.append(danz_oob)
            danz_ib = danz_inbag if danz_inbag is not None else danz_oob
            inbag_danzs.append(danz_ib)
            if auth is not None:
                auth_names.append(name)
                auth_oob_danzs.append(danz_oob)
                auths.append(auth)
            # FN = PLP assigned Normal (benign) evidence
            fn_oob = int(danz_oob.loc["PLP", "Normal"]) if "PLP" in danz_oob.index else 0
            fn_ib = int(danz_ib.loc["PLP", "Normal"]) if "PLP" in danz_ib.index else 0
            total_fn_oob += fn_oob
            total_fn_ib += fn_ib
            if fn_oob > 0 or fn_ib > 0:
                plp_total = int(danz_oob.loc["PLP"].sum()) if "PLP" in danz_oob.index else 0
                fn_records.append((name, fn_oob, fn_ib, plp_total))
            # FP = BLB assigned Abnormal (pathogenic) evidence
            fp_oob = int(danz_oob.loc["BLB", "Abnormal"]) if "BLB" in danz_oob.index else 0
            fp_ib = int(danz_ib.loc["BLB", "Abnormal"]) if "BLB" in danz_ib.index else 0
            total_fp_oob += fp_oob
            total_fp_ib += fp_ib
            if fp_oob > 0 or fp_ib > 0:
                blb_total = int(danz_oob.loc["BLB"].sum()) if "BLB" in danz_oob.index else 0
                fp_records.append((name, fp_oob, fp_ib, blb_total))
        print(f"    Built {len(ds_names)}/{len(reported_list)} ClinVar matrices "
              f"({len(auth_names)} with author labels) | "
              f"Total FN: OOB={total_fn_oob} InBag={total_fn_ib} | "
              f"Total FP: OOB={total_fp_oob} InBag={total_fp_ib}")
        fn_records.sort(key=lambda x: x[1], reverse=True)
        fp_records.sort(key=lambda x: x[1], reverse=True)
        if fn_records:
            print(f"    --- False Negatives (PLP → Normal) [{len(fn_records)} datasets] ---")
            for fn_name, fn_oob_v, fn_ib_v, plp_tot in fn_records:
                print(f"      FN {fn_name}: OOB={fn_oob_v} InBag={fn_ib_v}  (PLP total={plp_tot})")
        if fp_records:
            print(f"    --- False Positives (BLB → Abnormal) [{len(fp_records)} datasets] ---")
            for fp_name, fp_oob_v, fp_ib_v, blb_tot in fp_records:
                print(f"      FP {fp_name}: OOB={fp_oob_v} InBag={fp_ib_v}  (BLB total={blb_tot})")
        conf_by_method[method] = (oob_danzs, ds_names)
        inbag_by_method[method] = (inbag_danzs, ds_names)
        auth_by_method[method] = (auth_oob_danzs, auths, auth_names)

    # Use the first method's dataset list as the reference for naming
    first_method = methods_to_run[0]
    _, datasets_out = conf_by_method[first_method]
    if not datasets_out:
        print("ERROR: no datasets produced valid confusion matrices")
        return

    from analysis.plot_utils import make_confusion_figure, make_scatter_figure

    # Helper: align two ClinVar or in-bag lists to the shared dataset set
    def _align2(src1, src2):
        d1, n1 = src1;  d2, n2 = src2
        set2 = set(n2);  n2_idx = {n: k for k, n in enumerate(n2)}
        idx1 = [k for k, n in enumerate(n1) if n in set2]
        shared = [n1[k] for k in idx1]
        return shared, [d1[k] for k in idx1], [d2[n2_idx[n]] for n in shared]

    # --- Fig 1: ExCALIBR vs Author (OOB and in-bag, per method) ---
    print("\nGenerating ExCALIBR vs Author confusion heatmaps (OOB and in-bag)...")
    for method in methods_to_run:
        label = method or "default"
        auth_danzs, auths, auth_names = auth_by_method[method]
        inbag_danzs_all, _ = inbag_by_method[method]
        # align in-bag to the author-paired subset
        _, ib_sub, _ = _align2(inbag_by_method[method], (auth_danzs, auth_names))
        if auth_names:
            make_confusion_figure(
                danzs_m1=auth_danzs, danzs_m2=auths,
                dataset_names=auth_names,
                label1=f"{label}_OOB", label2="Author",
                figure_dir=figure_dir,
            )
            make_confusion_figure(
                danzs_m1=ib_sub, danzs_m2=auths,
                dataset_names=auth_names,
                label1=f"{label}_InBag", label2="Author",
                figure_dir=figure_dir,
            )

    # --- Fig 2: Method comparison (tavtigian vs piecewise) ---
    if len(methods_to_run) >= 2:
        print("\nGenerating method comparison figures...")
        for i in range(len(methods_to_run)):
            for j in range(i + 1, len(methods_to_run)):
                m1, m2 = methods_to_run[i], methods_to_run[j]

                # ClinVar OOB: ALL datasets where both methods have valid matrices.
                # No author-label filtering — the full reported list is compared.
                shared_oob, d1_oob, d2_oob = _align2(conf_by_method[m1], conf_by_method[m2])
                print(f"    ClinVar OOB comparison {m1} vs {m2}: {len(shared_oob)} datasets")
                make_confusion_figure(
                    danzs_m1=d1_oob, danzs_m2=d2_oob,
                    dataset_names=shared_oob, label1=f"{m1}_OOB", label2=f"{m2}_OOB",
                    figure_dir=figure_dir,
                )

                # ClinVar in-bag: ALL datasets, no author dependency
                shared_ib, d1_ib, d2_ib = _align2(inbag_by_method[m1], inbag_by_method[m2])
                print(f"    ClinVar InBag comparison {m1} vs {m2}: {len(shared_ib)} datasets")
                make_confusion_figure(
                    danzs_m1=d1_ib, danzs_m2=d2_ib,
                    dataset_names=shared_ib, label1=f"{m1}_InBag", label2=f"{m2}_InBag",
                    figure_dir=figure_dir,
                )

                # Scatter: OOB and InBag method comparisons (all datasets)
                if shared_oob:
                    make_scatter_figure(
                        conf_by_method={m1: d1_oob, m2: d2_oob}, dataset_names=shared_oob,
                        method1=m1, method2=m2, figure_dir=figure_dir, tag="OOB",
                    )
                if shared_ib:
                    make_scatter_figure(
                        conf_by_method={m1: d1_ib, m2: d2_ib}, dataset_names=shared_ib,
                        method1=m1, method2=m2, figure_dir=figure_dir, tag="InBag",
                    )

                # Author-labeled subset: same ClinVar comparison restricted to
                # datasets that have author confusion matrices in either method.
                auth_names_any = (
                    set(auth_by_method.get(m1, ([], [], []))[2])
                    | set(auth_by_method.get(m2, ([], [], []))[2])
                )
                if auth_names_any:
                    if shared_oob:
                        oob_auth = [
                            (d1, d2, n) for d1, d2, n in zip(d1_oob, d2_oob, shared_oob)
                            if n in auth_names_any
                        ]
                        if oob_auth:
                            ao1, ao2, an = zip(*oob_auth)
                            print(f"    ClinVar OOB (author subset) {m1} vs {m2}: {len(an)} datasets")
                            make_confusion_figure(
                                danzs_m1=list(ao1), danzs_m2=list(ao2),
                                dataset_names=list(an),
                                label1=f"{m1}_OOB", label2=f"{m2}_OOB",
                                figure_dir=figure_dir, tag="AuthSubset",
                            )
                    if shared_ib:
                        ib_auth = [
                            (d1, d2, n) for d1, d2, n in zip(d1_ib, d2_ib, shared_ib)
                            if n in auth_names_any
                        ]
                        if ib_auth:
                            ai1, ai2, an = zip(*ib_auth)
                            print(f"    ClinVar InBag (author subset) {m1} vs {m2}: {len(an)} datasets")
                            make_confusion_figure(
                                danzs_m1=list(ai1), danzs_m2=list(ai2),
                                dataset_names=list(an),
                                label1=f"{m1}_InBag", label2=f"{m2}_InBag",
                                figure_dir=figure_dir, tag="AuthSubset",
                            )

    # --- Fig 2: Evidence distributions ---
    # Load variants directly from CSVs in output_dir; attach author labels from df_final.
    print("\nBuilding evidence arrays from pipeline CSVs...")
    try:
        from analysis.plot_utils import make_evidence_figure

        output_dir = Path(legacy_ctx["output_dir"])
        use_oob = not getattr(args, "in_bag", False)

        ev_tree, ev_model_sel, ev_cals = discover_outputs(output_dir)
        datasets_out_set = set(datasets_out)
        df_ev = load_all_variants(
            tree=ev_tree,
            model_selections=ev_model_sel,
            dataset_configs=None,
            methods_filter=getattr(args, "methods", None),
            datasets_filter=list(datasets_out_set),
            calibrations=ev_cals,
            min_controls=0,
            include_all=True,  # legacy mode already curates its own dataset list
        )

        if not df_ev.empty:
            # Attach author labels via Scoreset (same path as non-legacy mode),
            # passing the integrated dataset so deduplication is handled correctly.
            if getattr(args, "dataset", None):
                df_ev = attach_author_labels(df_ev, args.dataset)

            for method in sorted(df_ev["method"].unique()):
                df_m = df_ev[df_ev["method"] == method].copy()
                if use_oob and "oob_points" in df_m.columns:
                    has_oob = df_m["oob_points"].notna()
                    df_m.loc[has_oob, "standard_points"] = df_m.loc[has_oob, "oob_points"]
                all_danz, all_clinvar = build_evidence_arrays(df_m)
                all_author = build_author_array(df_m)
                label = f"legacy_{method}"
                make_evidence_figure(
                    all_danz=all_danz,
                    all_author=all_author,
                    all_clinvar=all_clinvar,
                    label=label,
                    figure_dir=figure_dir,
                )
        else:
            print("  WARNING: no variants loaded from output_dir for evidence distribution")

    except Exception as e:
        import traceback
        print(f"  WARNING: evidence distribution failed: {e}")
        traceback.print_exc()

    # --- Fig 3: Per-dataset calibration figures ---
    print("\nGenerating per-dataset calibration figures...")
    try:
        from analysis.plot_utils import make_calibration_figure, load_lr_values

        output_dir_path = Path(legacy_ctx["output_dir"])

        for ds in datasets_out:
            # Resolve pipeline key (same logic as confusion matrix)
            pipeline_base = ds.replace("_clinvar_2018", "").replace("Scott_2022", "Jia_2021")
            gene = ds.split("_")[0]
            use_2018 = gene in _LEGACY_GENES_2018
            cal_key = pipeline_base + "_clinvar_2018" if use_2018 else pipeline_base

            cals_by_method: Dict = {}
            lrs_by_method: Dict = {}
            comp_by_method: Dict = {}
            for method in methods_to_run:
                cal_f = _find_calibration_in_output(output_dir_path, cal_key, method=method)
                if cal_f is None:
                    cals_by_method[method or "default"] = None
                    lrs_by_method[method or "default"] = None
                    continue
                comp_match = re.search(r"_(\d+c)_calibration\.json$", cal_f.name)
                comp = comp_match.group(1) if comp_match else "2c"
                comp_by_method[method or "default"] = comp
                try:
                    with open(cal_f) as f:
                        cals_by_method[method or "default"] = json.load(f)
                except Exception:
                    cals_by_method[method or "default"] = None
                lrs_by_method[method or "default"] = load_lr_values(output_dir_path, cal_key, method, comp)

            # Load variants from pipeline CSV for histogram (use first method)
            df_hist = pd.DataFrame()
            for label, comp in comp_by_method.items():
                method_key = None if label == "default" else label
                cal_key_m = cal_key
                found = list(output_dir_path.rglob(
                    f"{cal_key_m}_{label}_{comp}_variants.csv" if label != "default"
                    else f"{cal_key_m}_{comp}_variants.csv"
                ))
                if not found:
                    # try without method token
                    found = list(output_dir_path.rglob(f"{cal_key_m}_{comp}_variants.csv"))
                if found:
                    try:
                        df_hist = pd.read_csv(found[0])
                    except Exception:
                        pass
                    break

            if any(v is not None for v in cals_by_method.values()):
                make_calibration_figure(
                    df_variants=df_hist,
                    calibrations_by_method=cals_by_method,
                    lr_by_method=lrs_by_method,
                    dataset=ds,
                    figure_dir=figure_dir,
                    tag="legacy",
                )
    except Exception as e:
        import traceback
        print(f"  WARNING: calibration figures failed: {e}")
        traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("LEGACY ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Figures saved to: {figure_dir}")


# ---------------------------------------------------------------------------
# Prior override helpers
# ---------------------------------------------------------------------------

_PRIOR_OVERRIDE_METHODS = {"tavtigian", "piecewise", "strict_additive"}


def _parse_prior_overrides(raw: List[str]) -> Dict[str, float]:
    """Parse ['tavtigian=0.1', 'piecewise=0.5'] → {'tavtigian': 0.1, 'piecewise': 0.5}."""
    result = {}
    for item in raw:
        try:
            method, val = item.split("=", 1)
            result[method.strip()] = float(val.strip())
        except ValueError:
            raise ValueError(
                f"Invalid --prior-overrides entry {item!r}; expected 'method=value' (e.g. tavtigian=0.1)"
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
    from src.assay_calibration.fit_utils.fit import calculate_score_ranges
    from src.assay_calibration.plot_utils.utils import flatten_point_ranges, assign_points
    from analysis.plot_utils import load_lr_values

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

            log_lr = lr_data["log_lr_plus"]  # shape [n_boots, n_scores]
            score_range = lr_data["score_range"]
            orig_prior = float(lr_data["prior"])

            lr_5_log = np.nanpercentile(log_lr, 5, axis=0)
            lr_95_log = np.nanpercentile(log_lr, 95, axis=0)

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


# ---------------------------------------------------------------------------
# Reported-list subset helpers
# ---------------------------------------------------------------------------

def _reported_list_new_names(dataset_names_csv: Optional[str]) -> Optional[List[str]]:
    """Translate _LEGACY_REPORTED_LIST old names → new CSV names.

    Returns a list of new names present in the mapping, or None if no CSV provided.
    Names with no mapping entry are passed through unchanged.
    """
    if dataset_names_csv is None:
        return None
    try:
        names_df = pd.read_csv(dataset_names_csv)
        old_to_new = dict(zip(names_df["Old_names"], names_df["New_names"]))
    except Exception as e:
        print(f"  WARNING: could not load {dataset_names_csv}: {e}")
        return None
    return [old_to_new.get(name, name) for name in _LEGACY_REPORTED_LIST]


def _subset_conf_for_datasets(
    conf_by_method: Dict[str, List],
    all_datasets: List[str],
    keep_datasets: List[str],
) -> Tuple[Dict[str, List], List[str]]:
    """Return (subset_conf_by_method, subset_dataset_names) filtered to keep_datasets."""
    keep_set = set(keep_datasets)
    indices = [i for i, d in enumerate(all_datasets) if d in keep_set]
    sub_datasets = [all_datasets[i] for i in indices]
    sub_conf = {m: [mats[i] for i in indices] for m, mats in conf_by_method.items()}
    return sub_conf, sub_datasets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-pipeline analysis: figures from run_igvf_batch.py / run_pipeline.py outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Pipeline output directory (contains *_variants.csv files)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Original integrated dataset TSV/CSV (enables author-label panel)",
    )
    parser.add_argument(
        "--dataset-configs", default=None,
        help="JSON config mapping dataset → [n_c, benign_method, ...] "
             "(same file as --dataset-configs in run_igvf_batch.py; drives component selection)",
    )
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help="Methods to include (default: auto-discover).  "
             "Example: --methods tavtigian piecewise",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Only analyse these dataset names (default: all discovered)",
    )
    parser.add_argument(
        "--include-all", action="store_true",
        help=f"Include datasets normally excluded by default ({sorted(DEFAULT_EXCLUDED_DATASETS)}) "
             "and, when --dataset-configs is given, datasets absent from that config file.",
    )
    parser.add_argument(
        "--figure-dir", default=None,
        help="Output directory for figures (default: <output-dir>/figures)",
    )
    parser.add_argument(
        "--min-controls", type=int, default=5,
        help="Skip datasets with fewer than this many P/LP+B/LB variants (default: 5)",
    )
    parser.add_argument(
        "--in-bag", action="store_true",
        help="Use standard_points (in-bag calibration) for confusion matrices instead "
             "of oob_points (default: use oob_points with per-variant fallback to standard_points)",
    )
    parser.add_argument(
        "--dataset-names-csv", default=None,
        help="Path to new_dataset_names.csv (columns: Old_names, New_names). "
             "When provided, method-comparison figures are generated separately for "
             "the reported-list subset and for all datasets.",
    )
    parser.add_argument(
        "--prior-overrides", nargs="*", default=None, metavar="METHOD=VALUE",
        help="Override the prior used to assign points for one or more methods. "
             "Format: 'method=value', e.g. --prior-overrides tavtigian=0.1 piecewise=0.5. "
             "Recomputes standard_points from the saved LR+ curves (lr_values.json.gz) "
             "using the specified prior instead of the fitted one. "
             "oob_points are cleared so all analysis uses the recomputed values. "
             "Supported methods: tavtigian, piecewise, strict_additive.",
    )

    # Legacy mode arguments
    parser.add_argument(
        "--legacy", action="store_true",
        help="Run legacy author-label analysis. Requires --output-dir (pipeline outputs: "
             "*_variants.csv and *_calibration.json) and --dataset.",
    )
    parser.add_argument(
        "--legacy-calibrations-dir", default=None,
        dest="legacy_calibrations_dir",
        help="Directory of reference calibration JSONs (e.g. calibrations_12_25_25/). "
             "When provided, used in preference to calibrations found in --output-dir.",
    )
    parser.add_argument(
        "--legacy-oob-pkl", default=None,
        dest="legacy_oob_pkl",
        help="Path to reference OOB pickle (variant_to_oob_points_full.pkl). "
             "When provided, overrides OOB points derived from variants CSVs.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir) if args.figure_dir else output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PIPELINE OUTPUT ANALYSIS")
    print("=" * 80)
    print(f"\nOutput dir : {output_dir}")
    print(f"Figure dir : {figure_dir}")

    # Legacy mode: bypass pipeline discovery, use /data/ross/assay_calibration/ directly
    if args.legacy:
        run_legacy_analysis(args, figure_dir)
        return

    # ------------------------------------------------------------------
    # 1. Discover outputs
    # ------------------------------------------------------------------
    print("\nDiscovering pipeline outputs...")
    tree, model_selections, calibrations = discover_outputs(output_dir)
    if not tree:
        print("ERROR: no *_variants.csv or *_calibration.json files found in output directory")
        sys.exit(1)

    all_datasets = sorted(tree.keys())
    print(f"  Found {len(all_datasets)} dataset(s): {', '.join(all_datasets[:5])}"
          f"{'...' if len(all_datasets) > 5 else ''}")

    discovered_methods: set = set()
    for comp_dict in tree.values():
        for method_dict in comp_dict.values():
            discovered_methods.update(method_dict.keys())
    print(f"  Methods: {', '.join(sorted(discovered_methods))}")

    # ------------------------------------------------------------------
    # 2. Load dataset configs if provided
    # ------------------------------------------------------------------
    dataset_configs = None
    if args.dataset_configs:
        with open(args.dataset_configs) as f:
            dataset_configs = json.load(f)
        print(f"\nLoaded {len(dataset_configs)} dataset configs from {args.dataset_configs}")

    # ------------------------------------------------------------------
    # 3. Load all variants
    # ------------------------------------------------------------------
    print("\nLoading variants CSVs...")
    df = load_all_variants(
        tree=tree,
        model_selections=model_selections,
        dataset_configs=dataset_configs,
        methods_filter=args.methods,
        datasets_filter=args.datasets,
        calibrations=calibrations,
        min_controls=args.min_controls,
        include_all=args.include_all,
        recompute_points=args.in_bag,
    )
    if df.empty:
        print("ERROR: no variants loaded — check --methods / --datasets filters")
        sys.exit(1)

    methods = sorted(df["method"].unique())
    datasets = sorted(df["dataset"].unique())
    print(f"  Loaded {len(df):,} variant rows")
    print(f"  Methods: {methods}")
    print(f"  Datasets: {len(datasets)}")

    # ------------------------------------------------------------------
    # 3.5. Apply prior overrides (recompute standard_points from LR+ curves)
    # ------------------------------------------------------------------
    if args.prior_overrides:
        print("\nApplying prior overrides...")
        prior_overrides = _parse_prior_overrides(args.prior_overrides)
        df = recompute_points_with_prior_overrides(df, output_dir, prior_overrides)

    # ------------------------------------------------------------------
    # 4. Optionally attach author labels
    # ------------------------------------------------------------------
    if args.dataset:
        df = attach_author_labels(df, args.dataset)

    # ------------------------------------------------------------------
    # 5. Build per-dataset confusion matrices and evidence arrays
    # ------------------------------------------------------------------
    use_oob = not args.in_bag
    print(f"\nBuilding confusion matrices (use_oob={use_oob})...")

    # conf_by_method[method] = list of matrices, one per dataset (None if missing)
    conf_by_method: Dict[str, List] = {m: [] for m in methods}
    auth_by_method: Dict[str, List] = {m: [] for m in methods}

    for dataset in datasets:
        df_ds = df[df["dataset"] == dataset]
        for method in methods:
            df_m = df_ds[df_ds["method"] == method]
            mat = build_confusion_matrix(df_m, use_oob=use_oob) if not df_m.empty else None
            conf_by_method[method].append(mat)
            auth_mat = build_author_confusion_matrix(df_m, use_oob=use_oob) if not df_m.empty else None
            auth_by_method[method].append(auth_mat)

    # Print FNs and FPs per method, sorted descending by count
    for method in methods:
        fn_records: List[Tuple] = []
        fp_records: List[Tuple] = []
        for mat, dataset in zip(conf_by_method[method], datasets):
            if mat is None:
                continue
            fn = int(mat.loc["PLP", "Normal"]) if "PLP" in mat.index else 0
            fp = int(mat.loc["BLB", "Abnormal"]) if "BLB" in mat.index else 0
            plp_total = int(mat.loc["PLP"].sum()) if "PLP" in mat.index else 0
            blb_total = int(mat.loc["BLB"].sum()) if "BLB" in mat.index else 0
            if fn > 0:
                fn_records.append((dataset, fn, plp_total))
            if fp > 0:
                fp_records.append((dataset, fp, blb_total))
        fn_records.sort(key=lambda x: x[1], reverse=True)
        fp_records.sort(key=lambda x: x[1], reverse=True)
        method_label = f" [{method}]" if len(methods) > 1 else ""
        if fn_records:
            print(f"\n  False Negatives (PLP → Normal){method_label} [{len(fn_records)} datasets]:")
            for ds_name, fn, plp_total in fn_records:
                print(f"    FN {ds_name}: {fn}  (PLP total={plp_total})")
        if fp_records:
            print(f"\n  False Positives (BLB → Abnormal){method_label} [{len(fp_records)} datasets]:")
            for ds_name, fp, blb_total in fp_records:
                print(f"    FP {ds_name}: {fp}  (BLB total={blb_total})")

    # ------------------------------------------------------------------
    # 6. Generate figures
    # ------------------------------------------------------------------
    from analysis.plot_utils import (
        make_confusion_figure,
        make_single_confusion_figure,
        make_evidence_figure,
        make_scatter_figure,
    )

    # Build reported-list subset (when --dataset-names-csv provided and multi-method)
    reported_new_names = _reported_list_new_names(getattr(args, "dataset_names_csv", None))
    if reported_new_names is not None and len(methods) >= 2:
        rep_conf, rep_datasets = _subset_conf_for_datasets(conf_by_method, datasets, reported_new_names)
        print(f"  Reported-list subset: {len(rep_datasets)}/{len(_LEGACY_REPORTED_LIST)} datasets found")
    else:
        rep_conf, rep_datasets = None, None

    def _drop_none_pairs(mats1, mats2, names):
        """Keep only index positions where BOTH matrices are non-None."""
        rows = [(a, b, n) for a, b, n in zip(mats1, mats2, names)
                if a is not None and b is not None]
        if not rows:
            return [], [], []
        a, b, n = zip(*rows)
        return list(a), list(b), list(n)

    def _drop_none_single(mats1, mats2, names):
        """Keep only positions where the second (author) matrix is non-None."""
        rows = [(a, b, n) for a, b, n in zip(mats1, mats2, names) if b is not None]
        if not rows:
            return [], [], []
        a, b, n = zip(*rows)
        return list(a), list(b), list(n)

    def _make_method_comparison_figures(cbm, abm, ds_names, tag=""):
        """Emit all confusion figures for a (method1 vs method2) pair.

        cbm : {method: [ClinVar confusion matrices]}  — no auth dependency
        abm : {method: [author confusion matrices]}   — may be all-None

        ClinVar method comparison always uses the full dataset set (all datasets
        where both methods have valid ClinVar matrices, regardless of author labels).
        Author comparisons are limited to datasets with valid author matrices.
        """
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]

                # ClinVar m1 vs ClinVar m2 — ALL datasets, no author requirement.
                # Explicitly filter to pairs where both ClinVar matrices are non-None.
                cv_m1, cv_m2, cv_names = _drop_none_pairs(cbm[m1], cbm[m2], ds_names)
                if cv_names:
                    print(f"    ClinVar comparison {m1} vs {m2}: {len(cv_names)} datasets (all)")
                    make_confusion_figure(
                        danzs_m1=cv_m1, danzs_m2=cv_m2,
                        dataset_names=cv_names,
                        label1=m1, label2=m2,
                        figure_dir=figure_dir, tag=tag,
                    )
                    make_scatter_figure(
                        conf_by_method={m1: cv_m1, m2: cv_m2},
                        dataset_names=cv_names,
                        method1=m1, method2=m2,
                        figure_dir=figure_dir, metric="accuracy", tag=tag,
                    )

                # ClinVar m1 vs m2 — AUTHOR-LABELED datasets only.
                if abm and cv_names:
                    auth_set = {
                        n for n, a in zip(ds_names, abm.get(m1, [None] * len(ds_names)))
                        if a is not None
                    }
                    auth_set |= {
                        n for n, a in zip(ds_names, abm.get(m2, [None] * len(ds_names)))
                        if a is not None
                    }
                    auth_filt = [(a, b, n) for a, b, n in zip(cv_m1, cv_m2, cv_names) if n in auth_set]
                    if auth_filt:
                        a1, a2, an = zip(*auth_filt)
                        auth_tag = f"{tag}_authsubset" if tag else "authsubset"
                        print(f"    ClinVar comparison {m1} vs {m2}: {len(an)} datasets (author-labeled subset)")
                        make_confusion_figure(
                            danzs_m1=list(a1), danzs_m2=list(a2),
                            dataset_names=list(an),
                            label1=m1, label2=m2,
                            figure_dir=figure_dir, tag=auth_tag,
                        )

                # Author m1 vs Author m2 — only datasets with both author matrices.
                if abm:
                    a_m1, a_m2, a_names = _drop_none_pairs(
                        abm.get(m1, []), abm.get(m2, []), ds_names
                    )
                    if a_names:
                        make_confusion_figure(
                            danzs_m1=a_m1, danzs_m2=a_m2,
                            dataset_names=a_names,
                            label1=f"{m1}_Author", label2=f"{m2}_Author",
                            figure_dir=figure_dir, tag=tag,
                        )

                # Per-method: ClinVar vs Author — only datasets with author matrices.
                if abm:
                    for m in (m1, m2):
                        cv_m, auth_m, auth_names_m = _drop_none_single(
                            cbm[m], abm.get(m, [None] * len(ds_names)), ds_names
                        )
                        if auth_names_m:
                            make_confusion_figure(
                                danzs_m1=cv_m, danzs_m2=auth_m,
                                dataset_names=auth_names_m,
                                label1=m, label2=f"{m}_Author",
                                figure_dir=figure_dir, tag=tag,
                            )

    # --- Fig 1: Confusion heatmap ---
    print("\nGenerating confusion heatmaps...")

    # Calibration confusion matrix per method — always plotted, independent of
    # author labels or other methods being available.
    for m in methods:
        make_single_confusion_figure(
            matrices=conf_by_method[m],
            dataset_names=datasets,
            label=m,
            figure_dir=figure_dir,
        )

    if len(methods) >= 2:
        _make_method_comparison_figures(conf_by_method, auth_by_method, datasets, tag="all")
        if rep_conf is not None and rep_datasets:
            rep_auth, _ = _subset_conf_for_datasets(auth_by_method, datasets, reported_new_names)
            _make_method_comparison_figures(rep_conf, rep_auth, rep_datasets, tag="reported")
    else:
        # Single method: additionally plot ClinVar vs author when author labels exist
        m = methods[0]
        auths = auth_by_method[m]
        has_auth = any(a is not None for a in auths)
        if has_auth:
            make_confusion_figure(
                danzs_m1=conf_by_method[m], danzs_m2=auths,
                dataset_names=datasets,
                label1=m, label2="author",
                figure_dir=figure_dir,
            )

    # --- Fig 2: Evidence distributions ---
    print("\nGenerating evidence distribution figures...")
    for method in methods:
        df_m = df[df["method"] == method]
        all_danz, all_clinvar = build_evidence_arrays(df_m)
        all_author = build_author_array(df_m)
        make_evidence_figure(
            all_danz=all_danz,
            all_author=all_author,
            all_clinvar=all_clinvar,
            label=method,
            figure_dir=figure_dir,
        )

    # Per-dataset calibration figures are not generated here — they're already
    # produced by the main calibration pipeline (run_igvf_batch.py / run_pipeline.py).

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"Figures saved to: {figure_dir}")


if __name__ == "__main__":
    main()
