"""
ClinGen expert-panel ground-truth confusion pipeline — ported from
test/plot_author_calibration_confusion.py's ClinGen analysis section.

Unlike the ClinVar-based confusion matrices (analysis/confusion.py), ground
truth here is the ClinGen Variant Curation Expert Panel's own applied ACMG
evidence codes (`Applied Evidence Codes (Met)_ClinGen_repo` — already merged
into the master dataframe, no external file needed), with PS3/BS3 (functional
assay evidence) codes stripped before reclassifying, to avoid circularity
against ExCALIBR's own functional-assay-derived evidence.

This does NOT use the legacy insert_evidence_into_dataframe (which required
the variant_to_oob_points_full.pkl + a directory of raw calibration JSONs) —
instead it reuses the already-loaded pipeline-native variants (variant_id,
effective evidence points, auth_label) and rebuilds one Scoreset per dataset
purely to align each kept variant's ClinGen evidence-code string to its
variant_id (same alignment scheme as analysis/author_labels.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from analysis.acmg_evidence_codes import classify_acmg
from analysis.confusion import _dor_coverage_text
from analysis.discovery import load_master_df
from analysis.multi_scoreset import genomic_variant_key, _merge_points, _merge_author_labels
from analysis.plot_common import effective_points, save_and_show
from src.assay_calibration.plot_utils.utils import compute_classification_metrics

# Functional-assay ACMG codes stripped before reclassifying — keeping these
# would make the "ground truth" partly derived from the same kind of
# functional-assay evidence ExCALIBR itself is being evaluated against.
REMOVE_CODES = {
    "BS3", "BS3_Supporting", "BS3_Moderate", "BS3_Strong", "BS3_Very",
    "PS3", "PS3_Supporting", "PS3_Moderate", "PS3_Strong", "PS3_Very",
}


def filter_and_recalculate(
    evidence_string: Optional[str], strip_functional_evidence: bool = True,
) -> Tuple[str, str]:
    """Reclassify a comma-separated ACMG evidence-code string, optionally
    stripping PS3/BS3 codes first. Returns (updated_classification,
    filtered_evidence_string).

    `strip_functional_evidence=False` skips the PS3/BS3 removal (keeping
    ClinGen's classification exactly as applied, functional evidence and
    all) -- useful as a circularity-check comparison against the default
    stripped behavior, since PS3/BS3 are themselves the kind of
    functional-assay evidence ExCALIBR is being evaluated against."""
    if pd.isna(evidence_string):
        return "VUS", ""

    evidence_list = [code.strip() for code in evidence_string.split(",")]
    filtered_evidence = (
        [code for code in evidence_list if code not in REMOVE_CODES]
        if strip_functional_evidence else evidence_list
    )
    filtered_evidence_string = ",".join(filtered_evidence)
    updated_classification = classify_acmg(filtered_evidence) if filtered_evidence else "VUS"
    return updated_classification, filtered_evidence_string


def _to_row(label: int) -> int:
    # +1, 0, -1 -> row index (0=Pathogenic, 1=Indeterminate, 2=Benign)
    return {1: 0, 0: 1, -1: 2}[label]


def _to_col(label: int) -> int:
    # PLP=+1, BLB=-1 -> column index
    return 0 if label == 1 else 1


def build_clingen_confusion(
    df_variants: pd.DataFrame,
    dataset_tsv: str,
    dataset_list: List[str],
    use_oob: bool = True,
    verbose_recode: bool = False,
    tree: Optional[Dict] = None,
    model_selections: Optional[Dict] = None,
    calibrations: Optional[Dict] = None,
    strip_functional_evidence: bool = True,
) -> Tuple[Dict[str, np.ndarray], Set[str], pd.DataFrame]:
    """Build ExCALIBR-vs-ClinGen and Author-vs-ClinGen 3x2 confusion matrices.

    Rows: 0=Pathogenic, 1=Indeterminate, 2=Benign (ExCALIBR sign / author call)
    Cols: 0=PLP, 1=BLB (ClinGen expert-panel classification)

    df_variants must already have variant_id/dataset/standard_points[/oob_points]
    [/auth_label] columns (i.e. analysis.discovery.load_all_variants +
    analysis.author_labels.attach_author_labels output) -- reused directly
    for every dataset except the `_clinvar_2018`-suffixed ones (see below).

    `_clinvar_2018` datasets (BRCA1/MSH2/PTEN/TP53) get their variant/points
    table rebuilt fresh via analysis.discovery.build_variants_from_scoreset
    instead of being looked up in df_variants: their on-disk *_variants.csv
    was, for some runs, written before fixes that corrected `sample`/`is_vus`
    export for exactly these clinvar_2018-mode datasets (see
    analysis/patch_variants_csv.py's module docstring -- e.g. it verified
    MSH2_Jia_2021_clinvar_2018 undercounted VUS as 0/1376 instead of the true
    421), and ClinGen's own labels are not restricted to whatever P/LP/B/LB/
    VUS scope that stale file happened to carry. Rebuilding recovers every
    kept variant (VUS included) straight from the Scoreset + this dataset's
    resolved calibration.json, so ClinGen-labeled variants outside
    df_variants's narrower scope still get matched. Needs `tree`/
    `model_selections`/`calibrations` (pass in section 1's own globals) to
    resolve the calibration.json path; clinvar_2018 datasets missing from
    `tree` fall back to the plain df_variants lookup like any other dataset.

    `verbose_recode`, if True, prints every evidence-code recode
    (old_classification --> filtered/updated_classification) as the legacy
    script did — off by default since it's extremely verbose across all
    datasets; turn on to audit a specific dataset.

    `strip_functional_evidence` (default True) controls whether PS3/BS3
    codes are removed before reclassifying ClinGen's ground-truth call --
    see `filter_and_recalculate`. Set False to see how the confusion
    matrices look with ClinGen's classification taken as-applied (functional
    evidence and all), as a circularity-check comparison.

    Returns (confusion, seen_genes, records) where `records` is a DataFrame
    with one row per kept, ClinGen-labeled variant (columns: dataset, gene,
    variant_key [genomic_variant_key(vid, gene), assay-independent], clingen_label,
    points [raw effective points, presign], auth_label [raw author string or
    None]) -- lets a caller merge duplicate genomic variants across a gene's
    assays (see `analysis.multi_scoreset.build_gene_deduped_variants` for the
    analogous ClinVar-ground-truth version) before re-tallying the confusion
    matrices, e.g. via `build_gene_deduped_clingen_confusion`.
    """
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    from src.assay_calibration.pipeline.variant_evidence import _get_variant_ids
    from analysis.author_labels import load_name_mapping
    from analysis.discovery import build_variants_from_scoreset, resolve_component, resolve_dataset_tsv_name

    confusion = {
        "auth": np.zeros((3, 2), dtype=int),
        "excalibr": np.zeros((3, 2), dtype=int),
    }
    seen_ids: Set[str] = set()
    seen_genes: Set[str] = set()
    records: List[dict] = []

    sep = "\t" if str(dataset_tsv).endswith((".tsv", ".tsv.gz")) else ","
    evidence_col = "Applied Evidence Codes (Met)_ClinGen_repo"

    # Cheap header-only read first -- avoids paying for a full (multi-MB,
    # gzipped, ~89-dataset) master TSV parse just to discover this dataset_tsv
    # doesn't even have the ClinGen evidence column.
    header_cols = pd.read_csv(dataset_tsv, sep=sep, nrows=0).columns
    if evidence_col not in header_cols:
        print(f"  SKIP ClinGen confusion: '{evidence_col}' not found in {dataset_tsv}")
        return confusion, seen_genes, pd.DataFrame(records)

    df_full = load_master_df(dataset_tsv)
    old_to_new, _ = load_name_mapping(str(dataset_tsv))

    for dataset in dataset_list:
        df_m = df_variants[df_variants["dataset"] == dataset]

        if "_clinvar_2018" in dataset and tree is not None and dataset in tree:
            comp = resolve_component(dataset, list(tree[dataset].keys()), model_selections or {}, None)
            cal_path = (calibrations or {}).get(dataset, {}).get("default", {}).get(comp)
            if cal_path is not None:
                try:
                    df_m = build_variants_from_scoreset(dataset, cal_path, dataset_tsv)
                    df_m["dataset"] = dataset
                    if use_oob and "oob_points" not in df_m.columns:
                        use_oob_this = False
                    else:
                        use_oob_this = use_oob
                except Exception as e:
                    print(f"  WARNING: full reload failed for {dataset} ({e}); "
                          f"falling back to df_variants's own scope")
                    use_oob_this = use_oob
            else:
                use_oob_this = use_oob
        else:
            use_oob_this = use_oob

        if df_m.empty:
            continue

        pts = effective_points(df_m, use_oob_this, label=dataset, context="clingen").values
        danz_by_vid = dict(zip(df_m["variant_id"], pts))
        auth_by_vid = (
            dict(zip(df_m["variant_id"], df_m["auth_label"]))
            if "auth_label" in df_m.columns else {}
        )

        csv_name = dataset.replace("_clinvar_2018", "")
        resolved_name = resolve_dataset_tsv_name(csv_name, df_full, dataset_tsv, old_to_new=old_to_new)
        if resolved_name is None:
            print(f"  SKIP {dataset}: not found in {dataset_tsv}")
            continue
        df_ds = df_full[df_full["Dataset"] == resolved_name].copy()
        df_ds["Dataset"] = dataset

        clinvar_release = "2018" if "_clinvar_2018" in dataset else "2025"
        pcfg = PipelineConfig(
            dataset_csv=str(dataset_tsv), dataset_name=dataset,
            output_dir="/tmp", clinvar_release=clinvar_release,
        )
        try:
            scoreset = load_dataset_from_df(df_ds, pcfg)
        except Exception as e:
            print(f"  SKIP {dataset}: Scoreset error — {e}")
            continue

        ids = _get_variant_ids(scoreset)
        variants_by_id = scoreset.get_variants_by_id()

        kept = 0
        gene = dataset.split("_")[0]
        for all_idx, (_, variants) in enumerate(variants_by_id.items()):
            if not scoreset._keep_mask[all_idx]:
                continue
            v0 = variants[0]
            vid = ids[kept]
            kept += 1

            evidence_str = v0.row.get(evidence_col)
            clingen_class, filtered_str = filter_and_recalculate(evidence_str, strip_functional_evidence)
            if verbose_recode and not pd.isna(evidence_str):
                print(f"  [{dataset}] {evidence_str} --> {filtered_str} ({clingen_class})")

            if clingen_class == "VUS" or pd.isna(clingen_class):
                continue
            if clingen_class not in ("Likely Pathogenic", "Pathogenic", "Likely Benign", "Benign"):
                continue
            clingen_label = 1 if clingen_class in ("Likely Pathogenic", "Pathogenic") else -1

            points = danz_by_vid.get(vid)
            if points is None:
                continue  # variant not present in the loaded pipeline output for this dataset
            excalibr_label = int(np.sign(points))

            auth_class = auth_by_vid.get(vid)
            auth_label = 0
            if isinstance(auth_class, str):
                upper = auth_class.upper()
                if upper == "ABNORMAL":
                    auth_label = 1
                elif upper == "NORMAL":
                    auth_label = -1

            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            seen_genes.add(gene)

            confusion["auth"][_to_row(auth_label), _to_col(clingen_label)] += 1
            confusion["excalibr"][_to_row(excalibr_label), _to_col(clingen_label)] += 1
            records.append({
                "dataset": dataset,
                "gene": gene,
                "variant_key": genomic_variant_key(
                    vid, gene,
                    getattr(v0, "nucleotide_or_aa", None), getattr(v0, "aa_ref", None),
                    getattr(v0, "aa_pos", None), getattr(v0, "aa_alt", None),
                ),
                "clingen_label": clingen_label,
                "points": points,
                "auth_label": auth_class if isinstance(auth_class, str) else None,
            })

    print(f"{len(seen_genes)} gene(s) retained ClinGen evidence: {sorted(seen_genes)}")
    return confusion, seen_genes, pd.DataFrame(records)


def build_gene_deduped_clingen_confusion(records: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Set[str]]:
    """Re-tally the ExCALIBR-vs-ClinGen and Author-vs-ClinGen 3x2 confusion
    matrices (same shape/convention as `build_clingen_confusion`'s) from
    `records` (its third return value), merging duplicate genomic variants
    across a gene's assays first -- the ClinGen analogue of
    `analysis.multi_scoreset.build_gene_deduped_variants`/
    `build_deduped_confusion_matrix` for the ClinVar-ground-truth panels.

    Merge rule per (gene, variant_key) group: `points` via `_merge_points`
    (abs-max across assays if they agree in sign, else 0), `auth_label` via
    `_merge_author_labels` (conflicting Normal/Abnormal across assays ->
    indeterminate). `clingen_label` (the ground truth for this physical
    variant) should already agree across every assay that scored it --
    groups where it doesn't are dropped (a data inconsistency, not a real
    dedup case) rather than silently picking one side.
    """
    confusion = {
        "auth": np.zeros((3, 2), dtype=int),
        "excalibr": np.zeros((3, 2), dtype=int),
    }
    seen_genes: Set[str] = set()
    if records.empty:
        return confusion, seen_genes

    n_dropped_conflicts = 0
    for (gene, _vkey), grp in records.groupby(["gene", "variant_key"], sort=False):
        if grp["clingen_label"].nunique() != 1:
            n_dropped_conflicts += 1
            continue
        clingen_label = int(grp["clingen_label"].iloc[0])

        points = _merge_points(grp["points"].to_numpy())
        excalibr_label = int(np.sign(points))

        merged_auth = _merge_author_labels(grp["auth_label"]) if "auth_label" in grp.columns else None
        auth_label = 0
        if isinstance(merged_auth, str):
            upper = merged_auth.upper()
            if upper == "ABNORMAL":
                auth_label = 1
            elif upper == "NORMAL":
                auth_label = -1

        seen_genes.add(gene)
        confusion["auth"][_to_row(auth_label), _to_col(clingen_label)] += 1
        confusion["excalibr"][_to_row(excalibr_label), _to_col(clingen_label)] += 1

    if n_dropped_conflicts:
        print(f"  gene-deduped ClinGen confusion: dropped {n_dropped_conflicts} variant-key group(s) "
              f"with conflicting ClinGen ground-truth calls across assays")
    print(f"{len(seen_genes)} gene(s) retained ClinGen evidence (gene-deduped): {sorted(seen_genes)}")
    return confusion, seen_genes


def convert_3x2_to_2x3(mat_3x2: np.ndarray) -> np.ndarray:
    """rows: P, I, B  cols: PLP, BLB  ->  rows: BLB, PLP  cols: Benign, Indet, Path."""
    return np.array([
        [mat_3x2[2, 1], mat_3x2[1, 1], mat_3x2[0, 1]],  # BLB row
        [mat_3x2[2, 0], mat_3x2[1, 0], mat_3x2[0, 0]],  # PLP row
    ])


_BLUE_COLORS = ['#F0F8FC', '#99C8DC', '#7AB5D1', '#4B91A6', '#2E6B7E']
_RED_COLORS = ['#FCF0F2', '#E6B1B8', '#D68F99', '#B85C6B', '#943744']
_GRAY_COLORS = ['#F5F5F5', '#CCCCCC', '#999999', '#666666']


def _plot_clingen_confusion_panel(ax, mat: np.ndarray, title: str, letter: str,
                                   xlabel: str = "", ylabel: str = "", show_yticklabels: bool = True):
    """One ClinGen confusion heatmap panel (ClinGen classification x
    evidence-direction/functional-annotation), shared by the 2-panel
    (`plot_2x3_confusions_nature`) and 4-panel (`plot_clingen_confusion_stacked`)
    figures -- identical per-panel visuals (heatmap colors, DOR/determinate-%
    caption, ticks) either way."""
    blue_cmap = LinearSegmentedColormap.from_list("blue_gradient", _BLUE_COLORS)
    red_cmap = LinearSegmentedColormap.from_list("red_gradient", _RED_COLORS)
    gray_cmap = LinearSegmentedColormap.from_list("gray_gradient", _GRAY_COLORS)

    rows, cols = mat.shape
    for i in range(rows):
        row_max = mat[i].max()
        for j in range(cols):
            value = mat[i, j]
            cmap = blue_cmap if j == 0 else (gray_cmap if j == 1 else red_cmap)
            norm = value / row_max if row_max > 0 else 0
            color = cmap(norm)

            ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1, facecolor=color, edgecolor='white', linewidth=2.5,
            ))

            text_color = 'white' if norm > 0.45 else 'black'
            if j == 1:
                text_color = 'white' if norm > 0.7 else 'black'

            ax.text(
                j + 0.5, i + 0.5, f"{value:,}",
                ha='center', va='center', fontsize=16, color=text_color,
            )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    ax.set_facecolor('#F9F9F9')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=10)

    # DOR + determinate-% caption, matching make_confusion_figure's own
    # panels (analysis/confusion.py) -- this heatmap previously showed
    # raw counts only, with no DOR or determinate-% for controls.
    metrics = compute_classification_metrics(pd.DataFrame(mat))
    ax.text(
        0.5, -0.20, _dor_coverage_text(metrics["dor_standard"], 100 * metrics["coverage"], None),
        transform=ax.transAxes, fontsize=11, ha="center", va="top", color="#555555",
    )

    ax.set_xticks([0.5, 1.5, 2.5])
    if "ExCALIBR" in title:
        ax.set_xticklabels(['Benign', 'Indeterminate', 'Pathogenic'], fontsize=12)
    else:
        ax.set_xticklabels(['Normal', 'Indeterminate', 'Abnormal'], fontsize=12)

    ax.set_yticks([0.5, 1.5])
    if show_yticklabels:
        ax.set_yticklabels(['P/LP', 'B/LB'][::-1], fontsize=13)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(-0.10, 1.11, f"({letter})", transform=ax.transAxes, fontsize=18, fontweight='bold', va='top')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)


def plot_2x3_confusions_nature(conf_dict: Dict[str, np.ndarray], figsize=(13, 4)):
    """Moved verbatim from test/plot_author_calibration_confusion.py — no
    visual changes."""
    fig = plt.figure(figsize=figsize)

    left_margin = 0.08
    bottom_margin = 0.15
    top_margin = 0.12
    plot_width = 0.35
    space_left = 0.05

    ax_ex = fig.add_axes([left_margin, bottom_margin, plot_width, 1 - bottom_margin - top_margin])
    ax_auth = fig.add_axes([left_margin + plot_width + space_left, bottom_margin,
                            plot_width, 1 - bottom_margin - top_margin])

    _plot_clingen_confusion_panel(ax_ex, conf_dict['excalibr'], "ExCALIBR Evidence", "A",
                                   xlabel='Evidence Direction', ylabel='ClinGen Classification')
    _plot_clingen_confusion_panel(ax_auth, conf_dict['auth'], "Author Annotations", "B",
                                   xlabel='Functional Annotation', ylabel='')

    return fig


def plot_clingen_confusion_stacked(
    conf_dict_top: Dict[str, np.ndarray], conf_dict_bottom: Dict[str, np.ndarray],
    bottom_title_suffix: str = "\n(with PS3/BS3)",
    figsize=(13, 10),
):
    """4-panel version of `plot_2x3_confusions_nature`: (A) ExCALIBR / (B)
    Author on top (`conf_dict_top`, typically PS3/BS3-stripped -- the
    circularity-avoiding default), (C) ExCALIBR / (D) Author below
    (`conf_dict_bottom`, typically PS3/BS3-kept, for the circularity check
    of how much of ClinGen's own "ground truth" already derives from
    functional-assay evidence) -- same two confusion dicts previously shown
    as two separate 2-panel figures (`clingen_confusion.png` and
    `clingen_confusion_with_ps3bs3.png`), stacked into one figure so both
    scopes are visible together. The bottom row's panel titles get
    `bottom_title_suffix` appended (e.g. "ExCALIBR Evidence (with PS3/BS3)")
    instead of a separate floating row label -- `_plot_clingen_confusion_panel`
    keys its ExCALIBR-vs-author x-tick-label choice off the literal
    substring "ExCALIBR" in the title, so the suffix must come after it.
    """
    fig = plt.figure(figsize=figsize)

    left_margin = 0.08
    top_margin = 0.08
    bottom_margin = 0.09
    # Each panel's DOR/determinate-% caption sits ~20% of the panel's own
    # height below its axes (see _plot_clingen_confusion_panel), and the
    # next row's title + panel-letter sit ~11% above its axes -- on top of
    # each other's text without a wide enough gap between rows here (unlike
    # the single-row 2-panel figure, where there's no second row below to
    # collide with). row_gap has to clear both.
    row_gap = 0.24
    plot_width = 0.35
    space_left = 0.05
    row_height = (1 - top_margin - bottom_margin - row_gap) / 2
    bottom_row_y = bottom_margin
    top_row_y = bottom_margin + row_height + row_gap

    ax_ex_top = fig.add_axes([left_margin, top_row_y, plot_width, row_height])
    ax_auth_top = fig.add_axes([left_margin + plot_width + space_left, top_row_y, plot_width, row_height])
    ax_ex_bottom = fig.add_axes([left_margin, bottom_row_y, plot_width, row_height])
    ax_auth_bottom = fig.add_axes([left_margin + plot_width + space_left, bottom_row_y, plot_width, row_height])

    _plot_clingen_confusion_panel(ax_ex_top, conf_dict_top['excalibr'], "ExCALIBR Evidence", "A",
                                   ylabel='ClinGen Classification')
    _plot_clingen_confusion_panel(ax_auth_top, conf_dict_top['auth'], "Author Annotations", "B", ylabel='')
    _plot_clingen_confusion_panel(ax_ex_bottom, conf_dict_bottom['excalibr'], f"ExCALIBR Evidence{bottom_title_suffix}",
                                   "C", xlabel='Evidence Direction', ylabel='ClinGen Classification')
    _plot_clingen_confusion_panel(ax_auth_bottom, conf_dict_bottom['auth'], f"Author Annotations{bottom_title_suffix}",
                                   "D", xlabel='Functional Annotation', ylabel='')

    return fig
