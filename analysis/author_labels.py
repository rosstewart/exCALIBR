"""
Author functional-classification label loading.

Attaches the original author-reported functional label (Normal / Abnormal /
Indeterminate) to variants loaded from pipeline output, by re-loading the
Scoreset for each dataset from the master integrated dataframe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _clinvar_release_for(dataset: str) -> str:
    """Infer ClinVar release from dataset name."""
    if "_clinvar_2018" in dataset and "not_clinvar_2018" not in dataset:
        return "2018"
    genes_2018 = {"BRCA1", "MSH2", "PTEN", "TP53"}
    gene = dataset.split("_")[0]
    if gene in genes_2018:
        return "2018"
    return "2025"


def load_name_mapping(dataset_tsv: str) -> Tuple[Dict[str, str], Dict[str, str]]:
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


# Legacy dataset aliases: reported name → pipeline/CSV name used for lookup.
# None value → excluded entirely (no valid data / composite models).
LEGACY_INTERNAL_ALIASES: Dict[str, Optional[str]] = {
    "F9_Popp_2025_model": None,
    "MSH2_Scott_2022": "MSH2_Jia_2021",
}


def load_author_labels_for_dataset(
    df_full: pd.DataFrame,
    dataset: str,
    clinvar_release: str,
    old_to_new: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Load Scoreset for one dataset and return {variant_id → auth_label}.

    Looks up dataset in df_full by its name first; if not found, tries the
    new CSV name via old_to_new mapping (e.g. BRCA2_unpublished → BRCA2_IGVF).
    """
    base = dataset.replace("_clinvar_2018", "")

    if base in LEGACY_INTERNAL_ALIASES and LEGACY_INTERNAL_ALIASES[base] is None:
        print(f"  SKIP author labels for {dataset}: excluded dataset")
        return {}

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
    """Ensure df has an 'auth_label' column, rebuilding via Scoreset only
    where needed.

    analysis.discovery.load_all_variants now always builds its variant table
    fresh from a Scoreset + calibration.json (never trusting an already-saved
    *_variants.csv for anything but oob_points -- see its docstring), and
    that same Scoreset construction populates auth_label directly. So every
    dataset loaded via load_all_variants already carries real auth_label
    values by the time this function runs -- this is now effectively a
    validating no-op for that path, not a second, independent Scoreset
    reload. The slow per-dataset Scoreset-rebuild branch below only exists
    as a safety net for callers that hand this function a `df` built some
    other way (e.g. a raw pd.read_csv of an old *_variants.csv, which can
    predate auth_label export entirely).
    """
    df = df.copy()
    if "auth_label" not in df.columns:
        df["auth_label"] = np.nan

    has_label = df.groupby("dataset")["auth_label"].transform(lambda s: s.notna().any())
    datasets_needing_rebuild = sorted(df.loc[~has_label, "dataset"].unique())

    n_already_labeled = int((has_label & df["auth_label"].notna()).sum())
    print(f"\nAuthor labels: {n_already_labeled}/{len(df)} variants already carried "
          f"auth_label (normally from load_all_variants' scoreset+calibration rebuild; "
          f"from the saved *_variants.csv only for a df built some other way).")

    if not datasets_needing_rebuild:
        print("  No datasets need Scoreset rebuild — done.")
        return df

    print(f"  Rebuilding Scoresets for {len(datasets_needing_rebuild)} dataset(s) "
          f"without auth_label: {datasets_needing_rebuild}")
    sep = "\t" if dataset_tsv.endswith((".tsv", ".tsv.gz")) else ","
    df_full = pd.read_csv(dataset_tsv, sep=sep, low_memory=False)

    old_to_new, _ = load_name_mapping(dataset_tsv)
    if old_to_new:
        print(f"  Loaded name mapping ({len(old_to_new)} entries) for CSV fallback lookup")

    label_maps: Dict[str, Dict[str, str]] = {}
    for dataset in datasets_needing_rebuild:
        cr = _clinvar_release_for(dataset)
        label_maps[dataset] = load_author_labels_for_dataset(
            df_full, dataset, cr, old_to_new=old_to_new
        )

    def _lookup(row):
        return label_maps.get(row["dataset"], {}).get(row["variant_id"])

    rebuild_mask = df["dataset"].isin(datasets_needing_rebuild)
    df.loc[rebuild_mask, "auth_label"] = df.loc[rebuild_mask].apply(_lookup, axis=1)
    n_labeled = df["auth_label"].notna().sum()
    print(f"  Author labels attached: {n_labeled}/{len(df)} variants")
    return df


def reported_list_new_names(reported_list, dataset_names_csv: Optional[str]) -> Optional[list]:
    """Translate an old-name reported list → new CSV names via dataset_names_csv.

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
    return [old_to_new.get(name, name) for name in reported_list]
