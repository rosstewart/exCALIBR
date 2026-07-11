"""
Evidence-distribution array construction and plotting for pipeline-native variants.

Plotting is a thin pass-through to src.assay_calibration.plot_utils.utils'
plot_combined_evidence_distributions / plot_evidence_by_clinvar_class_with_stats —
unchanged visuals, new calling convention only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.assay_calibration.plot_utils.utils import (
    plot_combined_evidence_distributions,
    plot_evidence_by_clinvar_class_with_stats,
)
from analysis.plot_common import effective_points, sample_matches

# Sample-column category names actually written by the pipeline's own
# *_variants.csv (see variant_evidence.py::_build_standard_table). "sample" is
# pipe-separated multi-label — a variant can genuinely match more than one of
# these at once (e.g. "Synonymous|population"), so category membership must
# be checked via sample_matches(), never via `==`.
CLINVAR_CATEGORY_NAMES = [
    "Benign/Likely Benign",          # index 0 = B/LB
    None,                             # index 1 = VUS (see build_clinvar_multilabel)
    "Pathogenic/Likely Pathogenic",  # index 2 = P/LP
    "population",                     # index 3 = gnomAD ("population" is the literal value written to disk)
    "Synonymous",                     # index 4 = Synonymous
]

# Map author label strings → integers used by plot_combined_evidence_distributions
# 0=Normal, 1=Indeterminate, 2=Abnormal
AUTH_LABEL_MAP = {
    "NORMAL":        0,
    "ABNORMAL":      2,
    "INDETERMINATE": 1,
    "NOT SPECIFIED": 1,
}


def build_clinvar_multilabel(df_method: pd.DataFrame) -> np.ndarray:
    """Return the (N, 5) multi-label ClinVar-class array
    src.assay_calibration.plot_utils.utils' plotting/table functions expect
    when passed `clinvar_classes` (each row a bool list [B/LB, VUS, P/LP,
    gnomAD, Synonymous]) — built directly from the real pipe-separated
    `sample` column rather than collapsing to one label per variant, so a
    variant genuinely in two categories (e.g. Synonymous *and* gnomAD) is
    counted in both, not just whichever came first.

    VUS (index 1) comes from the `is_vus` column when present — exported
    directly from Variant.is_vus (src/assay_calibration/data_utils/dataset.py),
    which is hardcoded to ClinVar *2026* regardless of what clinvar_release a
    given dataset otherwise uses for its own P/LP/B/LB classification (2018
    for BRCA1/MSH2/PTEN/TP53). VUS is a genuinely independent axis from
    B/LB/P/LP/gnomAD/Synonymous -- consequence type and population frequency
    don't determine clinical significance, so a variant can be a true VUS
    *and* Synonymous/gnomAD at the same time; it is NOT "matches none of the
    other four", which silently zeroes out VUS entirely for any dataset where
    every row happens to also carry a Synonymous/gnomAD/population tag (this
    was verified to undercount MSH2_Jia_2021_clinvar_2018 as 0 VUS variants
    when the real figure is 421).

    Falls back to the old negation-derived approximation (matches none of
    B/LB, P/LP, gnomAD, Synonymous) only when `is_vus` isn't in df_method at
    all -- i.e. variants.csv predates the is_vus export and hasn't been
    patched yet (see analysis/patch_variants_csv.py) -- logging that this
    happened since it's a real accuracy gap, not an equivalent substitute.
    """
    masks = []
    for name in CLINVAR_CATEGORY_NAMES:
        if name is None:
            masks.append(None)  # filled in below
        else:
            masks.append(sample_matches(df_method, name).values)

    if "is_vus" in df_method.columns:
        vus_mask = df_method["is_vus"].fillna(False).astype(bool).values
    else:
        print("  WARNING: 'is_vus' column missing -- falling back to negation-derived VUS "
              "(matches none of B/LB, P/LP, gnomAD, Synonymous), which undercounts any variant "
              "that is a true VUS but also Synonymous/gnomAD. Run analysis/patch_variants_csv.py "
              "to backfill is_vus and remove this warning.")
        explicit = np.stack([m for m in masks if m is not None], axis=1)
        vus_mask = ~explicit.any(axis=1)
    masks[1] = vus_mask
    return np.stack(masks, axis=1)


def build_evidence_arrays(
    df_method: pd.DataFrame, use_oob: bool = True, label: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (all_danz, all_clinvar) for one method's variants.

    all_danz   : int array of evidence points — OOB with in-bag fallback when
                 use_oob=True (matching the legacy script's all_danz_oob),
                 always standard_points when use_oob=False. Logs any in-bag
                 fallback (see analysis.plot_common.effective_points).
    all_clinvar: (N, 5) multi-label bool array — see build_clinvar_multilabel.
    """
    all_danz = effective_points(df_method, use_oob, label=label, context="evidence").values.astype(int)
    all_clinvar = build_clinvar_multilabel(df_method)
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


def build_dataset_info_and_arrays(
    df_method: pd.DataFrame,
    dataset_list,
    use_oob: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Pipeline-native equivalent of the legacy
    load_all_variant_assignments_with_oob: build one concatenated
    (all_danz, all_author, all_clinvar) triple, laid out in the same
    dataset order as `dataset_info_df`, for functions that slice per-dataset
    by `n_variants` (compute_genewise_evidence_table, plot_dataset_point_heatmap).

    `dataset_list` is explicit and adjustable — pass whichever datasets should
    be included (e.g. `datasets` for everything discovered, or a narrower
    curated list) rather than this function silently picking a scope for you.

    Returns
    -------
    dataset_info_df : columns [dataset, gene, n_variants]
    all_danz, all_author, all_clinvar : concatenated arrays, dataset-contiguous
        in the same row order as dataset_info_df. all_author entries are 1
        (Indeterminate) for variants with no auth_label, matching build_author_array.
    """
    info_rows = []
    danz_chunks, author_chunks, clinvar_chunks = [], [], []

    for dataset in dataset_list:
        df_ds = df_method[df_method["dataset"] == dataset]
        if df_ds.empty:
            print(f"  SKIP {dataset}: no variants loaded for this dataset")
            continue
        danz, clinvar = build_evidence_arrays(df_ds, use_oob=use_oob, label=dataset)
        author = build_author_array(df_ds)
        if author is None:
            author = np.ones(len(df_ds), dtype=int)  # all-Indeterminate placeholder

        info_rows.append({"dataset": dataset, "gene": dataset.split("_")[0], "n_variants": len(df_ds)})
        danz_chunks.append(danz)
        author_chunks.append(author)
        clinvar_chunks.append(clinvar)

    dataset_info_df = pd.DataFrame(info_rows)
    all_danz = np.concatenate(danz_chunks) if danz_chunks else np.array([], dtype=int)
    all_author = np.concatenate(author_chunks) if author_chunks else np.array([], dtype=int)
    all_clinvar = np.concatenate(clinvar_chunks) if clinvar_chunks else np.array([], dtype=int)
    return dataset_info_df, all_danz, all_author, all_clinvar


from analysis.plot_common import save_and_show, pretty_method as _pretty
_save = save_and_show


def make_evidence_figure(
    all_danz: np.ndarray,
    all_author: Optional[np.ndarray],
    all_clinvar: np.ndarray,
    label: str,
    figure_dir: Path,
):
    """Combined two-panel evidence distribution figure for one method.

    Top panel: author annotations (Normal / Indeterminate / Abnormal).
    Bottom panel: sample categories (B/LB, VUS, P/LP, gnomAD, Synonymous).

    When author labels are unavailable (all_author is None), falls back to a
    single-panel ClinVar/sample figure.
    """
    has_author = (
        all_author is not None
        and len(all_author) > 0
        and not np.all(all_author == 1)   # not entirely Indeterminate
    )

    if has_author:
        fig = plot_combined_evidence_distributions(
            author_assignments=all_danz,
            author_annotations=all_author,
            clinvar_assignments=all_danz,
            clinvar_classes=all_clinvar,
        )
    else:
        fig = plot_evidence_by_clinvar_class_with_stats(all_danz, all_clinvar)

    fig.suptitle(
        _pretty(label), fontsize=14, fontweight="bold",
        y=1.01 if has_author else 1.02,
    )
    _save(fig, figure_dir / f"evidence_distribution_{label}.png")


def make_combined_evidence_figure(
    author_danz: np.ndarray,
    author_annotations: np.ndarray,
    clinvar_danz: np.ndarray,
    clinvar_classes: np.ndarray,
    label: str,
    figure_dir: Path,
):
    """Combined author + ClinVar evidence-distribution figure, matching
    test/plot_author_calibration_confusion.py's true call:

        fig = plot_combined_evidence_distributions(
            all_danz_oob, all_author,           # author panel scope
            all_danz_oob_full, all_clinvar_full, # ClinVar panel scope
        )

    Unlike make_evidence_figure (which reuses one method's array for both
    panels), the author and ClinVar panels here can come from *different*
    dataset scopes — the legacy script restricted the author panel to only
    the reported/curated dataset list while the ClinVar panel spanned every
    dataset — so this takes all four arrays explicitly.
    """
    fig = plot_combined_evidence_distributions(
        author_assignments=author_danz,
        author_annotations=author_annotations,
        clinvar_assignments=clinvar_danz,
        clinvar_classes=clinvar_classes,
    )
    fig.suptitle(_pretty(label), fontsize=14, fontweight="bold", y=1.01)
    _save(fig, figure_dir / f"combined_evidence_distribution_{label}.png")
