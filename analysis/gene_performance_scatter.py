"""
Gene-level ExCALIBR-vs-author performance scatter + evidence-strength /
odds-ratio panels ("gene_brnich_OR" figure) + supplementary all-gene OR
figure + OR display table.

Ported from test/auxiliary_fig_creation/plot_gene_performance_scatter.py.
Plotting logic (colors, figsize, layout, titles) is kept byte-for-byte
identical to the source notebook — only where the input data comes from has
changed:

- ``datasets_confusion_mat.pkl`` / ``excalibr_confusion_mat.pkl`` /
  ``auth_confusion_mat.pkl`` (per-dataset ExCALIBR-vs-ClinVar and
  ExCALIBR-vs-author 2x3 confusion matrices, plus the parallel list of
  dataset names) are rebuilt fresh from pipeline output via
  ``analysis.discovery`` + ``analysis.confusion`` + ``analysis.author_labels``,
  rather than loaded from pickles. This matches the shape/semantics of what
  those pickles held exactly: ``build_confusion_matrix`` produces the same
  2x3 [BLB, PLP] x [Normal, IR, Abnormal] matrix that
  ``compute_classification_metrics`` (unchanged, from
  src.assay_calibration.plot_utils.utils) expects, and
  ``build_author_confusion_matrix`` produces the [Normal, Abnormal] x
  [Normal, IR, Abnormal] analog used for "auth accuracy".

- ``datasets_reached_evidence/*.pkl`` (evidence-level dataset counts for
  panels B/C of the combined figure) are NOT reconstructable from
  analysis.discovery/confusion/evidence alone (evidence.py builds per-variant
  distribution arrays, not per-dataset "reached evidence level N" counts), so
  these are still loaded from disk, guarded by
  ``analysis.config.warn_if_missing`` against
  ``analysis.config.EVIDENCE_COUNTS_DIR`` — if missing, panels B/C are simply
  skipped rather than crashing.

- The OR-estimate data -> ``analysis.config.AOU_OR_ESTIMATES_TSV``, a single
  long-format file (``warn_if_missing``-guarded; skip the OR-dependent
  panels/table if unavailable) filtered to ``Grouping/Calibration ==
  "ExCALIBR"`` and a chosen ``Consequence`` (default ``"All"``). Per-gene
  dataset selection is a manual ``GENE_TO_DATASET`` mapping maintained in
  this module (the old ``YURIY_OR_CSV``-derived lookup no longer applies,
  since the new file has multiple datasets per gene rather than one).

- The master integrated variant dataframe -> ``analysis.config.DATASET_TSV``.

- All PNG saves -> under ``analysis.config.FIGURE_DIR``.
"""
from __future__ import annotations

import json
from itertools import groupby
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, FuncFormatter, FixedLocator

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional dependency
    adjust_text = None

from src.assay_calibration.plot_utils.utils import compute_classification_metrics, _bold_italic_gene_title

from analysis import config as cfg
from analysis.discovery import discover_outputs, load_all_variants
from analysis.confusion import build_confusion_matrix, build_author_confusion_matrix
from analysis.author_labels import attach_author_labels


# ---------------------------------------------------------------------------
# Constants (verbatim from the source notebook)
# ---------------------------------------------------------------------------

PANEL_LETTER_Y = 1.08

# Color scheme for evidence strengths
STRENGTH_COLOR = {
    -8: '#1a5c6e', -7: '#2a6c7e', -6: '#3a7c8e', -5: '#4a8c9e',
    -4: '#5a9cae', -3: '#6aacbe', -2: '#7abcce', -1: '#8accde',
    0: '#e0e0e0',
    1: '#d89aa2', 2: '#c8838c', 3: '#b86d76', 4: '#b85c6b',
    5: '#B1535F', 6: '#AA4E58', 7: '#A2484F', 8: '#943744'
}

# Gene to disease mapping
GENE_DISEASE_MAP = {
    "BAP1": "Cancer", "BARD1": "Cancer", "BRCA1": "Cancer", "BRCA2": "Cancer",
    "CALM3": "Cardiovascular", "CHEK2": "Cancer", "G6PD": "Metabolic", "GCK": "Metabolic",
    "KCNE1": "Cardiovascular", "KCNH2": "Cardiovascular", "KCNQ4": "Hearing loss", "LDLR": "Metabolic",
    "MSH2": "Cancer", "OTC": "Metabolic", "PALB2": "Cancer", "PTEN": "Cancer",
    "RAD51C": "Cancer", "RAD51D": "Cancer", "SCN5A": "Cardiovascular",
    "TARDBP": "Rare disease", "TP53": "Cancer", "TSC2": "Cancer"
}

# Manual gene -> dataset pick for genes with more than one Dataset/Predictor
# row under Grouping/Calibration == "ExCALIBR" in AOU_OR_ESTIMATES_TSV (the
# old Yuriy-file-derived build_gene_to_dataset no longer applies, since that
# file had ~one row per gene and the new one doesn't). Picks with a prior
# Yuriy-file precedent keep that choice; the rest were confirmed directly.
# TP53 rows come from the separate AOU_TP53_OR_ESTIMATES_TSV (all 15 TP53
# assays); TP53_Fortuno_2021 is the gene-level pick.
GENE_TO_DATASET = {
    "BAP1": "BAP1_Waters_2024",
    "BARD1": "BARD1_IGVF",
    "BRCA1": "BRCA1_Findlay_2018",
    "BRCA2": "BRCA2_Huang_2025_SGE",
    "CHEK2": "CHEK2_Gebbia_2024",
    "GCK": "GCK_Gersing_2023_complementation",
    "KCNE1": "KCNE1_Muhammad_2024_potassium_flux",
    "KCNH2": "KCNH2_Jiang_2022",
    "KCNQ4": "KCNQ4_Zheng_2022_current_homozygous",
    "LDLR": "LDLR_Tabet_2025_abundance",
    "MSH2": "MSH2_Jia_2021",
    "PALB2": "PALB2_IGVF",
    "PTEN": "PTEN_Mighell_2018",
    "RAD51C": "RAD51C_Olvera-León_2024",
    "RAD51D": "RAD51D_IGVF",
    "SCN5A": "SCN5A_Ma_2024",
    "TP53": "TP53_Fortuno_2021",
    "TSC2": "TSC2_IGVF",
}


# ---------------------------------------------------------------------------
# VUS counting (unchanged)
# ---------------------------------------------------------------------------

def compute_vus_counts(
    df,
    datasets_new_configs,
    group_cols=['Gene', 'Chrom', 'hg38_start', 'ref_allele', 'alt_allele'],
    filter_nonsense=False
):

    # Get genes from dataset names
    genes_new_configs = [d.split('_')[0] for d in datasets_new_configs]

    # ClinVar star filter
    star_set = {
        np.nan,
        'no assertion criteria provided',
        'no assertion provided',
        'no interpretation for the single variant',
        'no classification provided',
        '-'
    }

    # Initial filter
    df_filtered = df[
        df['Gene'].isin(genes_new_configs) &
        (df['clinvar_sig_2025'] == 'Uncertain significance') &
        (~df['clinvar_star_2025'].isin(star_set))
    ].copy()

    # Remove flagged variants
    df_filtered = df_filtered[df_filtered['Flag'] != "*"]

    # Splicing filter
    if 'splice_measure' in df_filtered.columns and df_filtered['splice_measure'].nunique() == 1:
        detects_splice = df_filtered['splice_measure'].iloc[0] == "Yes"
        if not detects_splice:
            df_filtered = df_filtered[
                ~df_filtered['simplified_consequence'].str.lower().isin(
                    ['splice region', 'splice_site_variant']
                )
            ]
            # SpliceAI threshold filter
            for col in ['spliceAI_DS_AG', 'spliceAI_DS_AL', 'spliceAI_DS_DG', 'spliceAI_DS_DL']:
                if col in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered[col] < 0.2]

    # Count unique variants per gene (drop duplicates)
    unique_variants = df_filtered.drop_duplicates(subset=group_cols)
    vus_counts_series = unique_variants.groupby('Gene').size()
    gene_to_vus_count = vus_counts_series.to_dict()

    # Count NaN auth_reported_score
    num_nan_auth = df_filtered['auth_reported_score'].isna().sum() if 'auth_reported_score' in df_filtered.columns else 0

    # Count variants per dataset
    dataset_counts = df_filtered['Dataset'].value_counts() if 'Dataset' in df_filtered.columns else pd.Series()

    return gene_to_vus_count, num_nan_auth, dataset_counts


# ---------------------------------------------------------------------------
# Route (a): rebuild datasets_new_configs / danzs_new_configs_oob /
# auths_new_configs_oob fresh from pipeline output, replacing
# datasets_confusion_mat.pkl / excalibr_confusion_mat.pkl / auth_confusion_mat.pkl
# ---------------------------------------------------------------------------

def build_confusion_data(
    output_dir: Optional[str] = None,
    dataset_tsv: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
) -> Tuple[List[str], List[Optional[pd.DataFrame]], List[Optional[pd.DataFrame]]]:
    """Rebuild (dataset_names, danzs_oob, auths_oob) from pipeline output.

    dataset_names : list of dataset names (replaces datasets_confusion_mat.pkl)
    danzs_oob     : per-dataset ExCALIBR-vs-ClinVar 2x3 confusion matrix
                    (replaces excalibr_confusion_mat.pkl), via
                    analysis.confusion.build_confusion_matrix
    auths_oob     : per-dataset ExCALIBR-vs-author 2x3 confusion matrix
                    (replaces auth_confusion_mat.pkl), via
                    analysis.confusion.build_author_confusion_matrix
    """
    output_dir = output_dir or cfg.OUTPUT_DIR
    dataset_tsv = dataset_tsv or cfg.DATASET_TSV
    dataset_configs_path = dataset_configs_path or cfg.DATASET_CONFIGS

    dataset_configs = None
    if dataset_configs_path and Path(dataset_configs_path).exists():
        with open(dataset_configs_path) as f:
            dataset_configs = json.load(f)

    tree, model_selections, calibrations = discover_outputs(Path(output_dir))
    df_variants = load_all_variants(
        tree, model_selections, dataset_configs,
        methods_filter=None, datasets_filter=None,
        calibrations=calibrations,
    )
    if df_variants.empty:
        print("  WARNING: no pipeline output discovered — confusion data will be empty")
        return [], [], []

    df_variants = attach_author_labels(df_variants, dataset_tsv)

    dataset_names = sorted(df_variants["dataset"].unique())
    danzs_oob: List[Optional[pd.DataFrame]] = []
    auths_oob: List[Optional[pd.DataFrame]] = []
    for dataset in dataset_names:
        sub = df_variants[df_variants["dataset"] == dataset]
        danzs_oob.append(build_confusion_matrix(sub))
        auths_oob.append(build_author_confusion_matrix(sub))

    return dataset_names, danzs_oob, auths_oob


# ---------------------------------------------------------------------------
# Evidence counts (datasets_reached_evidence/*.pkl) — not reconstructable
# from analysis.discovery/confusion/evidence, so still a guarded pickle load.
# ---------------------------------------------------------------------------

def load_evidence_counts(evidence_counts_dir: Optional[str] = None) -> Optional[Dict[str, Dict]]:
    """Load the four datasets_reached_evidence/*.pkl files.

    Returns None (with a printed warning) if the directory or any of the four
    files is missing, rather than raising.
    """
    import pickle

    evidence_counts_dir = evidence_counts_dir or cfg.EVIDENCE_COUNTS_DIR
    if cfg.warn_if_missing(evidence_counts_dir, "evidence counts dir (datasets_reached_evidence)"):
        return None

    dre_dir = Path(evidence_counts_dir)
    names = ["excalibr_path_counts", "excalibr_ben_counts", "auth_path_counts", "auth_ben_counts"]
    result = {}
    for name in names:
        p = dre_dir / f"{name}.pkl"
        if cfg.warn_if_missing(str(p), f"evidence counts file {name}.pkl"):
            return None
        with open(p, "rb") as f:
            result[name] = pickle.load(f)
    return result


# ---------------------------------------------------------------------------
# OR-estimate CSVs (guarded)
# ---------------------------------------------------------------------------

def load_aou_or_estimates(
    aou_or_estimates_tsv: Optional[str] = None,
    consequence: str = "All",
    tp53_or_estimates_tsv: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Load AOU_OR_ESTIMATES_TSV, or None if it's missing.

    Rows from ``AOU_TP53_OR_ESTIMATES_TSV`` (TP53 assays, delivered
    separately) are appended, replacing any main-file rows with the same
    ``Dataset/Predictor``; skipped with a warning if that file is missing.

    Filters to ``Grouping/Calibration == "ExCALIBR"`` and the requested
    ``Consequence``, derives linear-scale ``Odds Ratio``/``OR_LI``/``OR_UI``
    from the file's ``LogOR``/``LogOR_LI``/``LogOR_UI`` columns, and renames
    columns to match what the (unchanged) plotting/table code below expects:
    ``Dataset/Predictor`` -> ``Dataset``, ``Group/Range`` -> ``Classification``,
    ``Wald p-value`` -> ``p-value``, ``Carrier cases``/``Carrier controls`` ->
    ``Cases with variants``/``Controls with variants``.
    """
    aou_or_estimates_tsv = aou_or_estimates_tsv or cfg.AOU_OR_ESTIMATES_TSV

    if cfg.warn_if_missing(aou_or_estimates_tsv, "AoU OR estimates TSV"):
        return None

    df = pd.read_csv(aou_or_estimates_tsv, sep="\t")
    tp53_tsv = tp53_or_estimates_tsv or cfg.AOU_TP53_OR_ESTIMATES_TSV
    if tp53_tsv and not cfg.warn_if_missing(tp53_tsv, "AoU TP53 OR estimates TSV"):
        df_tp53 = pd.read_csv(tp53_tsv, sep="\t")
        df = pd.concat([df[~df["Dataset/Predictor"].isin(set(df_tp53["Dataset/Predictor"]))], df_tp53],
                       ignore_index=True)
    df = df[
        (df["Grouping/Calibration"] == "ExCALIBR") & (df["Consequence"] == consequence)
    ].copy()

    # A few rows have near-infinite CIs from degenerate (zero-cell) logistic
    # fits -- exp() correctly sends those to 0/inf rather than erroring, so
    # just silence the expected overflow warning.
    with np.errstate(over="ignore"):
        df["Odds Ratio"] = np.exp(df["LogOR"])
        df["OR_LI"] = np.exp(df["LogOR_LI"])
        df["OR_UI"] = np.exp(df["LogOR_UI"])

    df = df.rename(columns={
        "Dataset/Predictor": "Dataset",
        "Group/Range": "Classification",
        "Wald p-value": "p-value",
        "Carrier cases": "Cases with variants",
        "Carrier controls": "Controls with variants",
    })

    return df


# ---------------------------------------------------------------------------
# Shared helpers (unchanged)
# ---------------------------------------------------------------------------

def dataset_short_label(dataset_name, gene=None) -> str:
    """Dataset name for figure labels/captions: internal suffix and leading
    gene token dropped, underscores as spaces ("BRCA2_Huang_2025_SGE" ->
    "Huang 2025 SGE")."""
    from analysis.plot_common import display_name
    name = display_name(dataset_name)
    prefix = f"{gene}_" if gene else name.split("_", 1)[0] + "_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name.replace("_", " ")


def order_genes_by_disease(genes):
    """Genes grouped by GENE_DISEASE_MAP category: largest group first (ties
    alphabetical by group name), genes alphabetical within each group."""
    genes = list(dict.fromkeys(genes))
    groups = {}
    for g in genes:
        groups.setdefault(GENE_DISEASE_MAP.get(g, "Unknown"), []).append(g)
    order = sorted(groups, key=lambda d: (-len(groups[d]), d))
    return [g for d in order for g in sorted(groups[d])]


CARRIER_TEXT_COLOR = "#9a9a9a"


def _annotate_carriers(ax, odds_ratio, y, cases):
    """Carrier-case count in small grey text (thin white outline) just above an OR point."""
    from matplotlib import patheffects as pe
    t = ax.annotate(f"{cases}", (odds_ratio, y), xytext=(0, 4.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color=CARRIER_TEXT_COLOR, zorder=3)
    t.set_path_effects([pe.Stroke(linewidth=1.5, foreground="white"), pe.Normal()])


# Groups with few carriers on either side have imprecise ORs. All of Us
# reports such counts only as "<= 20"; the collaborators' filtered OR file
# (AoU-OR-estimates_2026-10-07_filter_lowcarriercontrolcase) blanks exactly
# the groups with carrier cases <= 20 or carrier controls <= 20 (verified on
# all 640 shared ExCALIBR rows), so the same rule flags them here instead of
# dropping them. Strength stays in the color and significance in the fill;
# flagged groups get a thinner dashed CI and a smaller marker.
LOW_CARRIER_MAX = 20
LOW_CARRIER_STYLE = "dashed"  # "dashed" | "alpha" | "grey" | "off"; supplementary OR figures only (Fig. 5C: "off")


def _carrier_count_low(val) -> bool:
    """True for a carrier count <= LOW_CARRIER_MAX, or AoU's suppressed "<= 20" label."""
    if pd.isna(val):
        return False
    try:
        return float(val) <= LOW_CARRIER_MAX
    except (TypeError, ValueError):
        return True


def is_low_carrier(row) -> bool:
    return _carrier_count_low(row.get("Cases with variants")) or _carrier_count_low(row.get("Controls with variants"))


def _draw_or_point(ax, row, y, bar_color, style=None):
    """One OR estimate with its 95% CI: filled marker if significant, open if not;
    low-carrier groups (is_low_carrier) drawn in the LOW_CARRIER_STYLE."""
    style = style or LOW_CARRIER_STYLE
    low = style != "off" and is_low_carrier(row)
    color, alpha, lw, ls, size, cap = bar_color, 1.0, 2, "-", 50, 3
    if low:
        if style == "dashed":
            lw, ls, size, cap = 1.2, (0, (2.5, 1.5)), 26, 0
        elif style == "alpha":
            alpha = 0.35
        elif style == "grey":
            color = "#9e9e9e"
    eb = ax.errorbar(row['Odds Ratio'], y,
                     xerr=[[row['Odds Ratio'] - row['OR_LI']], [row['OR_UI'] - row['Odds Ratio']]],
                     fmt='none', ecolor=color, elinewidth=lw, capsize=cap, capthick=lw, alpha=alpha, zorder=1)
    for lc in eb[2]:
        lc.set_linestyle(ls)
    facecolor = color if row['significant'] else 'none'
    ax.scatter(row['Odds Ratio'], y, marker='o', s=size, facecolors=facecolor, edgecolors=color,
               linewidth=1.2 if low and style == "dashed" else 1.5, alpha=alpha, zorder=2)


def compute_log_xmin(xmax, x_ref=1.0, fraction=0.05):
    """
    Compute xmin for a log-scale axis so that there is a
    fixed proportional buffer to the left of x_ref.
    """
    log_xmax = np.log10(xmax)
    log_xref = np.log10(x_ref)

    log_xmin = log_xref - fraction * (log_xmax - log_xref)
    xmin = 10 ** log_xmin

    return xmin


def shrink_data_min_log(data_min, xmax, fraction=0.05):
    """
    Shrink a data_min value on a log axis by the same fraction used for axis buffer.
    """
    log_data_min = np.log10(data_min)
    log_xmax = np.log10(xmax)

    log_data_min_shrink = log_data_min - fraction * (log_xmax - log_data_min)
    data_min_shrink = 10 ** log_data_min_shrink

    return data_min_shrink


def _has_carriers(val):
    """True if a 'Cases with variants' value indicates >=1 carrier.

    AoU privacy rules replace small counts with a label like "<= 20" rather
    than the real number, so the whole column loads as strings -- including
    the plain-numeric ones (e.g. "0", "1398"). Try to parse it as a number
    first; only a non-numeric label (the suppressed case) is treated as
    "nonzero but hidden".
    """
    if pd.isna(val):
        return False
    try:
        return float(val) > 0
    except (TypeError, ValueError):
        return True


def _get_strength(label):
    if '≤' in label:
        return -int(label.split('-')[1].strip())
    elif '≥' in label:
        return int(label.split('+')[1].strip())
    return 0


FULL_CLASSIFICATION_ORDER = [
    "≤ -8", "≤ -7", "≤ -6", "≤ -5", "≤ -4", "≤ -3", "≤ -2", "≤ -1",
    "≥ +1", "≥ +2", "≥ +3", "≥ +4", "≥ +5", "≥ +6", "≥ +7", "≥ +8"
]


# ---------------------------------------------------------------------------
# Panel A + combined figure (unchanged body, taken from the live cell)
# ---------------------------------------------------------------------------

def plot_combined_analysis(excalibr_path_counts, excalibr_ben_counts,
                          auth_path_counts, auth_ben_counts,
                          danzs, auths, dataset_names, gene_to_vus_count,
                          compute_classification_metrics,
                          df_ors=None,
                          gene_to_dataset=None,
                          bottom_genes=['BRCA1', 'BRCA2', 'PALB2', 'RAD51C', 'RAD51D'],
                          size_by='vus',
                          annotate_gene_scatter=True, seed=42,
                          save_path=None, figsize=(18, 10), show_carrier_counts=False):

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Two separate grids
    gs_top = gridspec.GridSpec(1, 4, figure=fig,
                              left=0.05, right=0.95, top=0.95, bottom=0.55,
                              wspace=0.2, width_ratios=[2, 0.0, 1, 1])
    gs_bottom = gridspec.GridSpec(1, len(bottom_genes), figure=fig,
                                 left=0.05, right=0.95, top=0.40, bottom=0.05,
                                 wspace=0.1)

    ax_gene = fig.add_subplot(gs_top[0, 0])
    ax_path = fig.add_subplot(gs_top[0, 2])
    ax_ben = fig.add_subplot(gs_top[0, 3])
    ax_genes = [fig.add_subplot(gs_bottom[0, i]) for i in range(len(bottom_genes))]

    FONTSIZE_SMALL = 11
    FONTSIZE_AXIS_LABEL = 14
    FONTSIZE_AXIS_TICK = 12
    FONTSIZE_LEGEND = 10
    FONTSIZE_PANEL_LETTER = 18

    # ========== PANEL A: GENE-LEVEL COMPARISON ==========

    POINT_ALPHA = 0.85
    base_blue = "#4C72B0"
    cmap = LinearSegmentedColormap.from_list("excalibr_extreme", ["#9FBCE6", "#4C72B0", "#0B1F4B"])

    gene_danz, gene_auth = {}, {}
    for danz_df, auth_df, dataset_name in zip(danzs, auths, dataset_names):
        if danz_df is None or auth_df is None:
            continue
        gene = dataset_name.split('_')[0]
        if gene not in gene_danz:
            gene_danz[gene] = danz_df.copy()
            gene_auth[gene] = auth_df.copy()
        else:
            gene_danz[gene] += danz_df
            gene_auth[gene] += auth_df

    print(len(gene_danz), 'genes')

    improve_count = 0
    gene_results = []
    for gene in gene_danz:
        danz_metrics = compute_classification_metrics(gene_danz[gene])
        auth_metrics = compute_classification_metrics(gene_auth[gene])
        gene_results.append({
            'gene': gene,
            'danz_accuracy': danz_metrics['accuracy'],
            'auth_accuracy': auth_metrics['accuracy'],
            'vus_count': gene_to_vus_count.get(gene, 100),
            'control_count': int(gene_danz[gene].sum().sum())
        })
        if danz_metrics['accuracy'] != 0 and danz_metrics['accuracy'] > auth_metrics['accuracy']:
            improve_count += 1
        if danz_metrics['accuracy'] == 0:
            print(f"{gene}: {int(gene_danz[gene].sum().sum())} controls, {danz_metrics['accuracy']:.3f} excalibr, {auth_metrics['accuracy']:.3f} auth")

    print(improve_count, 'genes improved')
    finite = [r for r in gene_results if r['danz_accuracy'] not in [0, float('inf')] and r['auth_accuracy'] not in [0, float('inf')]]

    min_val = min(min(r['danz_accuracy'] for r in finite), min(r['auth_accuracy'] for r in finite)) if finite else 0.5
    plot_min = max(0.5, min_val - 0.05)
    plot_max = 1.015

    ax_gene.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.35, linewidth=1.5, zorder=1)

    vus_all = [r['vus_count'] for r in gene_results]
    control_all = [r['control_count'] for r in gene_results]

    if size_by == 'controls':
        size_vals, color_vals = control_all, vus_all
        size_label, color_label = 'Control variants', 'VUS count'
        get_size_metric = lambda r: r['control_count']
        get_color_metric = lambda r: r['vus_count']
    else:
        size_vals, color_vals = vus_all, control_all
        size_label, color_label = 'VUS count', 'Control variants'
        get_size_metric = lambda r: r['vus_count']
        get_color_metric = lambda r: r['control_count']

    def get_size(val):
        if max(size_vals) == min(size_vals):
            return 300
        s = (np.sqrt(val) - np.sqrt(min(size_vals))) / (np.sqrt(max(size_vals)) - np.sqrt(min(size_vals)))
        return 100 + s * 700

    norm = plt.Normalize(0.9*min(color_vals), 1.1*max(color_vals))

    texts = []
    for r in finite:
        color = cmap(norm(get_color_metric(r)))
        ax_gene.scatter(r['auth_accuracy'], r['danz_accuracy'], s=get_size(get_size_metric(r)),
                       c=[color], edgecolors='white', alpha=POINT_ALPHA, linewidth=1.5, zorder=3)

        from matplotlib import patheffects as pe

        if annotate_gene_scatter and r['gene'] not in ['RAD51C', 'OTC', 'SCN5A', 'RAD51D', 'FKRP', 'PALB2', 'KCNQ4', 'BARD1']:
            text = ax_gene.text(r['auth_accuracy'], r['danz_accuracy'],
                               r['gene'], fontsize=FONTSIZE_LEGEND,
                               ha='center', va='center', fontweight='bold', fontstyle='italic', zorder=10)

            # Add white border/stroke
            text.set_path_effects([
                pe.Stroke(linewidth=2.25, foreground='white'),  # White border
                pe.Normal()  # The text itself
            ])

            texts.append(text)

    # Final limits first: adjust_text must place labels within the axes as
    # drawn, otherwise a label (and its leader line) can land outside panel A
    # once the limits are narrowed (e.g. into panel C).
    ax_gene.set_xlim(plot_min * 0.95, plot_max+0.02)
    # Lower y-limit follows the data so no gene is cut off (JAG1 was below the
    # previous fixed 0.8075 floor, leaving only its leader line visible).
    y_floor = min(0.85 * 0.95, min(r['danz_accuracy'] for r in finite) - 0.02) if finite else 0.85 * 0.95
    ax_gene.set_ylim(y_floor, plot_max)
    if annotate_gene_scatter:
        np.random.seed(seed)
        try:
            adjust_text(texts, ax=ax_gene, arrowprops=dict(arrowstyle='-', color='gray', lw=1, alpha=0.5),
                       expand_points=(0.7, 0.7), force_text=0.3, ensure_inside_axes=True)
        except Exception:
            pass
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.text import Annotation
        for _artist in ax_gene.get_children():  # leader lines never leave panel A
            _arrow = _artist.arrow_patch if isinstance(_artist, Annotation) else _artist
            if isinstance(_arrow, FancyArrowPatch):
                _arrow.set_clip_on(True)
                _arrow.set_clip_path(ax_gene.patch)

    ax_gene.set_xlabel('Author Accuracy', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax_gene.set_ylabel('ExCALIBR Accuracy', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax_gene.grid(True, alpha=0.2)
    ax_gene.spines['top'].set_visible(False)
    ax_gene.spines['right'].set_visible(False)
    ax_gene.tick_params(axis='both', labelsize=FONTSIZE_AXIS_TICK)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_gene, fraction=0.035, pad=0.04, shrink=0.75, alpha=POINT_ALPHA)
    cbar.set_label(color_label, fontsize=FONTSIZE_AXIS_TICK, fontweight='bold')

    size_min, size_max = min(size_vals), max(size_vals)
    size_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=np.sqrt(get_size(size_min))/2,
               markerfacecolor=base_blue, markeredgecolor='white', label=f'{int(size_min):,}'),
        Line2D([0], [0], marker='o', linestyle='', markersize=np.sqrt(get_size(size_max))/2,
               markerfacecolor=base_blue, markeredgecolor='white', label=f'{int(size_max):,}'),
    ]
    ax_gene.legend(handles=size_handles, title=size_label, loc='lower right', frameon=True,
                  edgecolor='#999', framealpha=0.95, fontsize=10, title_fontsize=FONTSIZE_LEGEND)
    ax_gene.text(-0.1, PANEL_LETTER_Y, '(A)', transform=ax_gene.transAxes,
                fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold', va='top', ha='left')

    # ========== PANEL B & C: EVIDENCE BARS ==========

    levels = [1, 2, 4, 8]
    excalibr_colors = {8: '#943744', 4: '#B85C6B', 2: '#D68F99', 1: '#E6B1B8'}
    oddspath_colors = {8: '#6B2C91', 4: '#8E4BB8', 2: '#A96BC9', 1: '#C49EDB'}
    x = np.arange(len(levels))
    width = 0.35

    excalibr_path_vals = [excalibr_path_counts[lvl] for lvl in levels]
    auth_path_vals = [auth_path_counts[lvl] for lvl in levels]

    brnich_method = 'OddsPath' # r'Brnich $\it{et\ al.}$'

    for i, lvl in enumerate(levels):
        ax_path.bar(x[i] - width/2, excalibr_path_vals[i], width, color=excalibr_colors[lvl], alpha=0.9,
                   edgecolor='black', linewidth=1.2, label='ExCALIBR' if i == 0 else '')
        ax_path.bar(x[i] + width/2, auth_path_vals[i], width, color=oddspath_colors[lvl], alpha=0.9,
                   edgecolor='black', linewidth=1.2, label=brnich_method if i == 0 else '')

    for i in range(len(levels)):
        ax_path.text(x[i] - width/2, excalibr_path_vals[i], f'{int(excalibr_path_vals[i])}',
                    ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')
        ax_path.text(x[i] + width/2, auth_path_vals[i], f'{int(auth_path_vals[i])}',
                    ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')

    ax_path.set_xlabel('Evidence Level (Pathogenic)', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax_path.set_ylabel('Number of Datasets', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax_path.set_xticks(x)
    ax_path.set_xticklabels([f'+{lvl}' for lvl in levels], fontsize=FONTSIZE_AXIS_TICK)
    ax_path.tick_params(axis='both', labelsize=FONTSIZE_AXIS_TICK)
    ax_path.legend(fontsize=FONTSIZE_LEGEND, frameon=True)
    ax_path.grid(axis='y', alpha=0.3, linestyle='--')
    ax_path.spines['top'].set_visible(False)
    ax_path.spines['right'].set_visible(False)
    ax_path.text(-0.15, PANEL_LETTER_Y, '(B)', transform=ax_path.transAxes,
                fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold', va='top', ha='left')

    excalibr_ben_vals = [excalibr_ben_counts[lvl] for lvl in levels]
    auth_ben_vals = [auth_ben_counts[lvl] for lvl in levels]
    excalibr_colors_ben = {1: '#99C8DC', 2: '#7AB5D1', 4: '#4B91A6', 8: '#2E6B7E'}
    oddspath_colors_ben = {1: '#66C2A5', 2: '#3D9970', 4: '#2A7A5C', 8: '#1A5C42'}

    for i, lvl in enumerate(levels):
        ax_ben.bar(x[i] - width/2, excalibr_ben_vals[i], width, color=excalibr_colors_ben[lvl], alpha=0.9,
                  edgecolor='black', linewidth=1.2, label='ExCALIBR' if i == 0 else '')
        ax_ben.bar(x[i] + width/2, auth_ben_vals[i], width, color=oddspath_colors_ben[lvl], alpha=0.9,
                  edgecolor='black', linewidth=1.2, label=brnich_method if i == 0 else '')

    for i in range(len(levels)):
        ax_ben.text(x[i] - width/2, excalibr_ben_vals[i], f'{int(excalibr_ben_vals[i])}',
                   ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')
        ax_ben.text(x[i] + width/2, auth_ben_vals[i], f'{int(auth_ben_vals[i])}',
                   ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')

    ax_ben.set_xlabel('Evidence Level (Benign)', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax_ben.set_ylabel('Number of Datasets', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax_ben.set_xticks(x)
    ax_ben.set_xticklabels([f'-{lvl}' for lvl in levels], fontsize=FONTSIZE_AXIS_TICK)
    ax_ben.tick_params(axis='both', labelsize=FONTSIZE_AXIS_TICK)
    ax_ben.legend(fontsize=FONTSIZE_LEGEND, frameon=True)
    ax_ben.grid(axis='y', alpha=0.3, linestyle='--')
    ax_ben.spines['top'].set_visible(False)
    ax_ben.spines['right'].set_visible(False)

    # ========== PANELS C-G: OR PLOTS ==========

    panel_letters = ['(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)']
    full_classification_order = FULL_CLASSIFICATION_ORDER

    get_strength = _get_strength

    if df_ors is not None:
        df_filtered = df_ors[df_ors['Classification'] != "0"].copy()
        df_filtered['significant'] = ((df_filtered['LogOR_LI'] > 0) | (df_filtered['LogOR_UI'] < 0))

        # Calculate common limits (like R code)
        excluded_genes = []#["BRCA1", "MSH2", "KCNH2", "TSC2", "GCK"]
        common_limit_data = df_filtered[~df_filtered['Gene'].isin(excluded_genes)]

        if len(common_limit_data) > 0:
            common_xlim_min = min(common_limit_data['OR_LI'].min() * 0.85, 0.8)
            common_xlim_max = common_limit_data['OR_UI'].max() * 1.15
        else:
            common_xlim_min = 0.8
            common_xlim_max = 10

        for i, (ax, gene_name, letter) in enumerate(zip(ax_genes, bottom_genes, panel_letters)):

            # select gene and one dataset
            gene_data = df_filtered[(df_filtered['Gene'] == gene_name) & (df_filtered['Dataset'] == gene_to_dataset[gene_name])].copy()
            # Only rows that actually get drawn (see _has_carriers) should set
            # the axis range, and among those, only ones with a sane
            # (finite, positive-lower-bound) CI -- a quasi-separated fit from
            # very few carriers on one/both sides can have a carrier count
            # that passes _has_carriers yet a near-infinite CI that would
            # otherwise blow up xlim for the whole panel.
            gene_data_plotted = gene_data[gene_data['Cases with variants'].apply(_has_carriers)]
            gene_data_for_axis = gene_data_plotted[
                (gene_data_plotted['OR_LI'] > 0) & np.isfinite(gene_data_plotted['OR_UI'])
            ]

            all_classes_df = pd.DataFrame({'Classification': full_classification_order})

            if len(gene_data) > 0:
                all_classes_df = all_classes_df.merge(
                    gene_data[['Classification', 'Odds Ratio', 'OR_LI', 'OR_UI', 'significant', 'Cases with variants',
                           'Controls with variants']],
                    on='Classification', how='left')
            else:
                all_classes_df['Odds Ratio'] = np.nan
                all_classes_df['OR_LI'] = np.nan
                all_classes_df['OR_UI'] = np.nan
                all_classes_df['significant'] = False

            y_pos = np.arange(len(all_classes_df))

            for y in y_pos:
                ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.2, zorder=0)

            for idx, row in all_classes_df.iterrows():
                if pd.notna(row['Odds Ratio']) and _has_carriers(row['Cases with variants']):
                    y = idx
                    strength = get_strength(row['Classification'])
                    bar_color = STRENGTH_COLOR.get(strength, '#999999')

                    _draw_or_point(ax, row, y, bar_color, style="off")  # Fig. 5C: low-carrier groups drawn normally
                    if show_carrier_counts:
                        _annotate_carriers(ax, row['Odds Ratio'], y, row['Cases with variants'])

            ax.axvline(x=1, linestyle='--', color='black', linewidth=1, alpha=0.5, zorder=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(all_classes_df['Classification'], fontsize=FONTSIZE_AXIS_TICK)
            ax.set_xlabel('Odds Ratio', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')

            show_y_ticks = i == 0
            if show_y_ticks:
                ax.set_ylabel('Evidence Strength', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
            ax.tick_params(axis='both', labelsize=FONTSIZE_AXIS_TICK, left=show_y_ticks, labelleft=show_y_ticks)

            # For each gene, set limits individually (from the plotted rows only)
            ax.set_xscale('log')
            if len(gene_data_for_axis) > 0:
                data_max = gene_data_for_axis['OR_UI'].max()
                if pd.notna(data_max):
                    # Round up to nearest clean limit
                    clean_limits = [3, 5, 10, 30, 50, 100, 1000, 10000]
                    xlim_max = next((lim for lim in clean_limits if lim >= data_max), 10000)
                else:
                    xlim_max = 10
            else:
                xlim_max = 10

            xlim_min = compute_log_xmin(xlim_max, x_ref=1.0, fraction=0.05) # based on plot proportion
            data_min = shrink_data_min_log(gene_data_for_axis['OR_LI'].min(), xlim_max, fraction=0.05)
            if data_min < xlim_min:
                xlim_min = data_min # based on data

            ax.set_xlim(xlim_min, xlim_max)

            # Tick locator/formatter -- same scheme as plot_all_genes_or's
            # per-gene panels, so both figures render OR axes identically.
            def simple_formatter(val, pos):
                if val <= 0:
                    return ''
                if 1 <= val < 10:
                    return str(int(round(val)))
                if val >= 1000:
                    return f'{int(val):,}'
                elif val >= 10:
                    return str(int(round(val)))
                else:
                    return f'{val:.1g}'

            def null_formatter(val, pos):
                return ''

            if xlim_max < 10:
                ticks = [1]
                if xlim_max >= 2:
                    ticks.append(2)
                if xlim_max >= 3:
                    ticks.append(3)
                if xlim_max >= 5:
                    ticks.append(5)
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(simple_formatter))
            elif xlim_max in [10, 30, 50]:
                ticks = [1, 3]
                if xlim_max == 10:
                    ticks.extend([5, 10])
                elif xlim_max == 30:
                    ticks.extend([10, 30])
                elif xlim_max == 50:
                    ticks.extend([10, 50])
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(simple_formatter))
            else:
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
                ax.xaxis.set_major_formatter(FuncFormatter(simple_formatter))

            ax.xaxis.set_minor_formatter(FuncFormatter(null_formatter))

            ax.grid(True, alpha=0.2, axis='x', which='major')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            _bold_italic_gene_title(ax, gene_name, "", fontsize=FONTSIZE_AXIS_LABEL,
                                     x=0.5, y=1.0, ha='center', va='bottom')

            if letter == "(C)":
                ax.text(-0.29, PANEL_LETTER_Y + 0.12, letter, transform=ax.transAxes,
                       fontsize=FONTSIZE_PANEL_LETTER, fontweight='bold', va='top', ha='left')

        # ========== DISEASE BRACKETS ==========

        gene_diseases = [(gene, GENE_DISEASE_MAP.get(gene, "Unknown")) for gene in bottom_genes]
        disease_groups = []
        for disease, genes in groupby(gene_diseases, key=lambda x: x[1]):
            disease_groups.append((disease, [g[0] for g in genes]))

        for disease, genes in disease_groups:
            start_idx = bottom_genes.index(genes[0])
            end_idx = bottom_genes.index(genes[-1])
            bbox_start = ax_genes[start_idx].get_position()
            bbox_end = ax_genes[end_idx].get_position()

            x_left = bbox_start.x0
            x_right = bbox_end.x1
            y_bracket = bbox_start.y1 + 0.035
            bracket_height = 0.008

            fig.add_artist(plt.Line2D([x_left, x_left], [y_bracket, y_bracket + bracket_height],
                                     transform=fig.transFigure, color='black', linewidth=1.5))
            fig.add_artist(plt.Line2D([x_left, x_right], [y_bracket + bracket_height, y_bracket + bracket_height],
                                     transform=fig.transFigure, color='black', linewidth=1.5))
            fig.add_artist(plt.Line2D([x_right, x_right], [y_bracket, y_bracket + bracket_height],
                                     transform=fig.transFigure, color='black', linewidth=1.5))

            fig.text((x_left + x_right) / 2, y_bracket + bracket_height + 0.01, disease,
                    ha='center', va='bottom', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold',
                    transform=fig.transFigure)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")

    return fig, gene_results


# ---------------------------------------------------------------------------
# Supplementary all-genes OR figure (unchanged body)
# ---------------------------------------------------------------------------

def resolve_gene_dataset_entries(df_ors, gene_to_dataset, gene_order=None, dataset_mode="primary"):
    """Return [(gene, dataset), ...] to plot, given `dataset_mode`.

    "primary": one (gene, gene_to_dataset[gene]) pair per gene (today's
    behavior). "all": one pair per Dataset/Predictor row present for that
    gene in df_ors, since the new AOU_OR_ESTIMATES_TSV can have several
    datasets per gene even within Grouping/Calibration == "ExCALIBR".
    """
    genes = gene_order if gene_order is not None else sorted(gene_to_dataset.keys())
    entries = []
    for gene in genes:
        if dataset_mode == "all":
            datasets = sorted(df_ors.loc[df_ors['Gene'] == gene, 'Dataset'].unique())
            entries.extend((gene, dataset) for dataset in datasets)
        else:
            dataset = gene_to_dataset.get(gene)
            if dataset is not None:
                entries.append((gene, dataset))
    return entries


def plot_all_genes_or(df_ors, gene_to_dataset, gene_order=None, dataset_mode="primary",
                       save_path=None, figsize=(18, 12), show_carrier_counts=False,
                       ncols=None, row_height=None):
    """
    Create supplementary figure with OR plots for all genes.

    Parameters
    ----------
    df_ors : pd.DataFrame
        DataFrame with columns ['Gene', 'Dataset', 'Classification', 'Odds Ratio',
                                'OR_LI', 'OR_UI', 'LogOR', 'LogOR_LI', 'LogOR_UI',
                                'Wald p-value']
    gene_to_dataset : dict
        Maps gene name -> dataset name (selects one dataset per gene), used
        when dataset_mode == "primary".
    gene_order : list, optional
        Order of genes to display. If None, sorts alphabetically.
    dataset_mode : {"primary", "all"}
        "primary" shows one dataset per gene (via gene_to_dataset); "all"
        shows a separate panel for every Dataset/Predictor available for
        that gene.
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height); the height is per 3 rows.
    ncols, row_height : optional
        Grid columns and inches per row. Default: 5 columns at figsize[1] / 3
        per row; dataset_mode == "all" defaults to a condensed 8-column grid
        (about 3.1 in per row) so the figure fits one portrait page.
    """

    entries = resolve_gene_dataset_entries(df_ors, gene_to_dataset, gene_order, dataset_mode)
    all_genes = [gene for gene, _dataset in entries]

    n_genes = len(entries)
    condensed = dataset_mode == "all"
    ncols = ncols or (8 if condensed else 5)
    nrows = -(-n_genes // ncols)  # ceil div
    row_height = row_height or (3.1 if condensed else figsize[1] / 3)

    # Fixed space (inches) above each row for the disease bracket, gene and
    # dataset titles, so the layout no longer depends on the number of rows.
    fig_h = row_height * nrows + 0.9
    fig = plt.figure(figsize=(figsize[0], fig_h))
    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                          left=0.05, right=0.98, top=1 - 0.9 / fig_h, bottom=0.45 / fig_h,
                          hspace=0.75 if condensed else 0.6, wspace=0.12 if condensed else 0.15)

    # Create axes, one per entry, filled row-major.
    ax_genes = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(n_genes)]

    FONTSIZE_AXIS_LABEL = 14
    FONTSIZE_AXIS_TICK = 11 if condensed else 12
    FONTSIZE_Y_TICK = 8 if condensed else FONTSIZE_AXIS_TICK
    TITLE_Y = 1.20 if condensed else 1.12

    # Classification order
    full_classification_order = FULL_CLASSIFICATION_ORDER

    get_strength = _get_strength

    # Filter and process data
    df_filtered = df_ors[df_ors['Classification'] != "0"].copy()
    df_filtered['significant'] = ((df_filtered['LogOR_LI'] > 0) | (df_filtered['LogOR_UI'] < 0))

    for i, (ax, (gene_name, dataset_name)) in enumerate(zip(ax_genes, entries)):

        # select gene and dataset
        gene_data = df_filtered[(df_filtered['Gene'] == gene_name) & (df_filtered['Dataset'] == dataset_name)].copy()
        # Only rows that actually get drawn (see _has_carriers) should set
        # the axis range, and among those, only ones with a sane
        # (finite, positive-lower-bound) CI -- a quasi-separated fit from
        # very few carriers on one/both sides can have a carrier count
        # that passes _has_carriers yet a near-infinite CI that would
        # otherwise blow up xlim for the whole panel.
        gene_data_plotted = gene_data[gene_data['Cases with variants'].apply(_has_carriers)]
        gene_data_for_axis = gene_data_plotted[
            (gene_data_plotted['OR_LI'] > 0) & np.isfinite(gene_data_plotted['OR_UI'])
        ]

        # Create complete dataframe with all classification levels
        all_classes_df = pd.DataFrame({'Classification': full_classification_order})

        if len(gene_data) > 0:
            all_classes_df = all_classes_df.merge(
                gene_data[['Classification', 'Odds Ratio', 'OR_LI', 'OR_UI', 'significant', 'Cases with variants',
                           'Controls with variants']],
                on='Classification', how='left')
        else:
            all_classes_df['Odds Ratio'] = np.nan
            all_classes_df['OR_LI'] = np.nan
            all_classes_df['OR_UI'] = np.nan
            all_classes_df['significant'] = False

        y_pos = np.arange(len(all_classes_df))

        # Add horizontal grid lines
        for y in y_pos:
            ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.2, zorder=0)

        # Plot data points
        for idx, row in all_classes_df.iterrows():
            if pd.notna(row['Odds Ratio']) and _has_carriers(row['Cases with variants']):
                y = idx
                strength = get_strength(row['Classification'])
                bar_color = STRENGTH_COLOR.get(strength, '#999999')

                _draw_or_point(ax, row, y, bar_color)
                if show_carrier_counts:
                    _annotate_carriers(ax, row['Odds Ratio'], y, row['Cases with variants'])

        # Vertical line at OR = 1
        ax.axvline(x=1, linestyle='--', color='black', linewidth=1, alpha=0.5, zorder=1)

        # Styling
        ax.set_xscale('log')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_classes_df['Classification'], fontsize=FONTSIZE_Y_TICK)

        # Only show x-label where there's no panel below in the same column
        if i + ncols >= n_genes:
            ax.set_xlabel('Odds Ratio', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')

        # Only show y-label on leftmost column
        show_y_ticks = (i % ncols == 0)
        if show_y_ticks:
            ax.set_ylabel('Evidence Strength', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
        ax.tick_params(axis='x', labelsize=FONTSIZE_AXIS_TICK)
        ax.tick_params(axis='y', labelsize=FONTSIZE_Y_TICK, left=show_y_ticks, labelleft=show_y_ticks)

        # Set x-axis limits (from the plotted rows only)
        if len(gene_data_for_axis) > 0:
            data_max = gene_data_for_axis['OR_UI'].max()
            if pd.notna(data_max):
                clean_limits = [3, 5, 10, 30, 50, 100, 1000, 10000]
                xlim_max = next((lim for lim in clean_limits if lim >= data_max), 10000)
            else:
                xlim_max = 10
        else:
            xlim_max = 10

        xlim_min = compute_log_xmin(xlim_max, x_ref=1.0, fraction=0.05) # based on plot proportion
        data_min = shrink_data_min_log(gene_data_for_axis['OR_LI'].min(), xlim_max, fraction=0.05)
        if data_min < xlim_min:
            xlim_min = data_min # based on data

        ax.set_xlim(xlim_min, xlim_max)

        # Formatter
        def simple_formatter(val, pos):
            if val <= 0:
                return ''
            if 1 <= val < 10:
                return str(int(round(val)))
            if val >= 1000:
                return f'{int(val):,}'
            elif val >= 10:
                return str(int(round(val)))
            else:
                return f'{val:.1g}'

        def null_formatter(val, pos):
            return ''

        # Tick handling
        if xlim_max < 10:
            ticks = [1]
            if xlim_max >= 2:
                ticks.append(2)
            if xlim_max >= 3:
                ticks.append(3)
            if xlim_max >= 5:
                ticks.append(5)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(simple_formatter))
        elif xlim_max in [10, 30, 50]:
            ticks = [1, 3]
            if xlim_max == 10:
                ticks.extend([5, 10])
            elif xlim_max == 30:
                ticks.extend([10, 30])
            elif xlim_max == 50:
                ticks.extend([10, 50])
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(simple_formatter))
        else:
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
            ax.xaxis.set_major_formatter(FuncFormatter(simple_formatter))

        ax.xaxis.set_minor_formatter(FuncFormatter(null_formatter))

        ax.grid(True, alpha=0.2, axis='x', which='major')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Title: gene name (disease will be shown in brackets above
        # regardless). In dataset_mode == "all", the dataset name goes on
        # its own smaller, word-wrapped line below -- these can be long
        # (e.g. "BRCA2_Sahu_2023_exon13_Cisplatin_Resistance") and would
        # otherwise overrun neighboring panels in the narrow "all" grid.
        # Gene title, with the dataset (without its gene prefix) on a smaller
        # line below, in both modes.
        _bold_italic_gene_title(ax, gene_name, "", fontsize=FONTSIZE_AXIS_LABEL,
                                 x=0.5, y=TITLE_Y, ha='center', va='bottom')
        t = ax.text(0.5, 1.0, dataset_short_label(dataset_name, gene_name), transform=ax.transAxes,
                    fontsize=7 if condensed else 8, ha='center', va='bottom', wrap=True)
        t._get_wrap_line_width = lambda: ax.get_window_extent().width * 0.95

    # ========== ADD DISEASE GROUP BRACKETS ==========

    # Group panels by disease for bracket annotations, working off entry
    # *indices* (not gene-name lookups) so repeated gene names under
    # dataset_mode == "all" still group/position correctly.
    entry_diseases = [GENE_DISEASE_MAP.get(gene, "Unknown") for gene in all_genes]

    disease_groups = []  # (disease, start_idx, end_idx, row)
    current_disease = None
    current_row = -1
    group_start = None
    for idx, disease in enumerate(entry_diseases):
        row = idx // ncols
        if row != current_row or disease != current_disease:
            if group_start is not None:
                disease_groups.append((current_disease, group_start, idx - 1, current_row))
            group_start = idx
            current_disease = disease
            current_row = row
    if group_start is not None:
        disease_groups.append((current_disease, group_start, len(entry_diseases) - 1, current_row))

    # Draw brackets for each disease group
    for disease, start_idx, end_idx, row in disease_groups:
        # Get the axis positions in figure coordinates
        bbox_start = ax_genes[start_idx].get_position()
        bbox_end = ax_genes[end_idx].get_position()

        # Calculate bracket positions
        x_left = bbox_start.x0
        x_right = bbox_end.x1
        # Above the gene title (at 1.12 axes heights) plus its text, in
        # inches, so the bracket stays clear of the row above at any size.
        y_bracket = bbox_start.y1 + (TITLE_Y - 1) * bbox_start.height + 0.30 / fig_h
        bracket_height = 0.06 / fig_h

        # Draw the bracket (horizontal line with vertical ends)
        fig.add_artist(plt.Line2D([x_left, x_left], [y_bracket, y_bracket + bracket_height],
                                 transform=fig.transFigure, color='black', linewidth=1.5))
        fig.add_artist(plt.Line2D([x_left, x_right], [y_bracket + bracket_height, y_bracket + bracket_height],
                                 transform=fig.transFigure, color='black', linewidth=1.5))
        fig.add_artist(plt.Line2D([x_right, x_right], [y_bracket, y_bracket + bracket_height],
                                 transform=fig.transFigure, color='black', linewidth=1.5))

        # Add disease label centered above bracket
        x_center = (x_left + x_right) / 2
        y_text = y_bracket + bracket_height + 0.05 / fig_h

        fig.text(x_center, y_text, disease,
                ha='center', va='bottom',
                fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold',
                transform=fig.transFigure)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Supplementary OR plot saved to: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# OR display table (unchanged)
# ---------------------------------------------------------------------------

def prepare_or_display_table(df_ors, gene_to_dataset, gene_order=None, dataset_mode="primary"):
    """
    Filter df_ors to only the columns and rows needed for plot_all_genes_or.

    Parameters
    ----------
    df_ors : pd.DataFrame
        Full OR results table.
    gene_to_dataset : dict
        Maps gene name -> dataset name (selects one dataset per gene), used
        when dataset_mode == "primary".
    gene_order : list, optional
        Genes to include. If None, uses all keys in gene_to_dataset.
    dataset_mode : {"primary", "all"}
        See resolve_gene_dataset_entries.

    Returns
    -------
    pd.DataFrame
        Minimal table with only the rows/columns used for plotting.
    """
    entries = resolve_gene_dataset_entries(df_ors, gene_to_dataset, gene_order, dataset_mode)

    rows = [
        df_ors.loc[(df_ors['Gene'] == gene) & (df_ors['Dataset'] == dataset)]
        for gene, dataset in entries
    ]

    df = pd.concat(rows, ignore_index=True)

    # Columns actually read by plot_all_genes_or
    keep_cols = [
        'Gene',
        'Dataset',
        'Classification',
        'Odds Ratio',
        'OR_LI',
        'OR_UI',
        'p-value',
        'Cases with variants',
    ]

    # Include controls column if available
    if 'Controls with variants' in df_ors.columns:
        keep_cols.append('Controls with variants')

    # Compute significance flag inline (avoids needing LogOR columns at plot time)
    df['significant'] = (df['OR_LI'] > 1) | (df['OR_UI'] < 1)

    keep_cols.append('significant')

    df = df[keep_cols].copy()

    # Drop the "0" (indeterminate) classification — plot already filters it
    df = df[df['Classification'] != "0"]

    return df


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def build_gene_performance_scatter(
    output_dir: Optional[str] = None,
    figure_dir: Optional[str] = None,
    dataset_tsv: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
    evidence_counts_dir: Optional[str] = None,
    evidence_counts: Optional[Dict[str, Dict]] = None,
    aou_or_estimates_tsv: Optional[str] = None,
    consequence: str = "All",
    dataset_mode: str = "primary",
    annotate_gene_scatter: bool = True,
    output_or_display_csv: Optional[str] = None,
    confusion_data: Optional[Tuple[List[str], List, List]] = None,
    show_carrier_counts: bool = False,
    also_all_datasets_figure: bool = False,
) -> Dict:
    """Rebuild the gene-performance-scatter / evidence-strength / OR figures.

    `show_carrier_counts`: print each OR point's carrier-case count in small
    grey text in the supplementary OR figures (never in Fig. 5C).
    `also_all_datasets_figure`: with dataset_mode="primary", additionally save
    supp_OR_all_datasets.png (one OR panel per dataset, every ExCALIBR dataset
    of each gene) without changing the primary-mode figures or display table.

    `confusion_data`: optional precomputed (dataset_names, danzs_oob,
    auths_oob), e.g. the notebook's own datasets / conf_by_method[primary] /
    auths, to skip `build_confusion_data`'s second discovery + load pass.

    Replaces the notebook's top-level script logic. Any input that can't be
    found on disk is skipped (with a warning) rather than crashing:
      - evidence counts (datasets_reached_evidence/*.pkl) missing -> panels
        B/C of the combined figure, and the combined figure itself, are
        skipped. Pass `evidence_counts` (the same
        {"excalibr_path_counts", "excalibr_ben_counts", "auth_path_counts",
        "auth_ben_counts"} dict shape `load_evidence_counts` returns) to
        supply these directly -- e.g. from analyze_pipeline_output.py's own
        section 4d, which computes the ExCALIBR side fresh from pipeline
        output and the author/OddsPath side from OP_EVIDENCE_CODES_CSV --
        instead of requiring the legacy datasets_reached_evidence/*.pkl
        files on disk. Takes precedence over `evidence_counts_dir` when given.
      - AOU_OR_ESTIMATES_TSV missing -> OR panels/supplementary figure/display
        table are skipped.

    `consequence` selects the Consequence facet of AOU_OR_ESTIMATES_TSV to
    use (default "All"); pass e.g. "Missense" or "Stop Gained" and a
    different `output_or_display_csv`/`figure_dir` to generate per-
    consequence figures alongside the default ones. `dataset_mode` controls
    per-gene dataset selection for the supplementary figure/table -- see
    resolve_gene_dataset_entries.

    Returns a dict of whatever was successfully produced, e.g.
    {'gene_results': [...], 'fig_combined': <Figure>, 'fig_supp': <Figure>,
     'df_display': <DataFrame>}.
    """
    output_dir = output_dir or cfg.OUTPUT_DIR
    figure_dir = Path(figure_dir or cfg.FIGURE_DIR)
    figure_dir.mkdir(parents=True, exist_ok=True)
    dataset_tsv = dataset_tsv or cfg.DATASET_TSV

    result: Dict = {}

    print("Loading main dataframe...")
    sep = "\t" if str(dataset_tsv).endswith((".tsv", ".tsv.gz")) else ","
    df_std = pd.read_csv(dataset_tsv, sep=sep, low_memory=False)
    print(f"Loaded dataframe with {len(df_std)} variants")

    print("Building confusion matrices from pipeline output...")
    if confusion_data is not None:
        dataset_names, danzs_oob, auths_oob = (list(x) for x in confusion_data)
    else:
        dataset_names, danzs_oob, auths_oob = build_confusion_data(
            output_dir=output_dir, dataset_tsv=dataset_tsv,
            dataset_configs_path=dataset_configs_path,
        )
    result["dataset_names"] = dataset_names

    if not dataset_names:
        print("  SKIP gene performance scatter: no pipeline output discovered")
        return result

    gene_to_vus_count, num_nan_auth, dataset_vus_counts = compute_vus_counts(
        df_std, dataset_names,
    )
    print(f"Computed VUS counts for {len(gene_to_vus_count)} genes")
    result["gene_to_vus_count"] = gene_to_vus_count

    evidence_counts = evidence_counts if evidence_counts is not None else load_evidence_counts(evidence_counts_dir)
    df_ors = load_aou_or_estimates(aou_or_estimates_tsv, consequence=consequence)
    gene_to_dataset = GENE_TO_DATASET
    result["gene_to_dataset"] = gene_to_dataset

    if evidence_counts is not None:
        print("Generating combined plot...")
        fig, gene_results = plot_combined_analysis(
            evidence_counts["excalibr_path_counts"], evidence_counts["excalibr_ben_counts"],
            evidence_counts["auth_path_counts"], evidence_counts["auth_ben_counts"],
            danzs_oob, auths_oob,
            dataset_names, gene_to_vus_count,
            compute_classification_metrics=compute_classification_metrics,
            annotate_gene_scatter=annotate_gene_scatter,
            df_ors=df_ors,
            gene_to_dataset=gene_to_dataset,
            bottom_genes=['BRCA1', 'MSH2', 'RAD51D', 'TP53', 'TSC2', 'KCNQ4'],
            show_carrier_counts=False,  # Fig. 5C: no carrier counts (supplementary OR figures only)
            save_path=str(figure_dir / f'gene_brnich_OR{"_noannot" if not annotate_gene_scatter else ""}.png'),
            seed=42,
        )
        result["fig_combined"] = fig
        result["gene_results"] = gene_results
    else:
        print("  SKIP combined plot: evidence counts unavailable")

    if df_ors is not None:
        # Disease groups largest first (ties alphabetical), genes alphabetical within.
        gene_order = order_genes_by_disease(
            [g for g in df_ors["Gene"].unique() if g in gene_to_dataset or dataset_mode == "all"])
        fig_supp = plot_all_genes_or(
            df_ors, gene_to_dataset, gene_order=gene_order, dataset_mode=dataset_mode,
            save_path=str(figure_dir / f'supp_OR{"_all_datasets" if dataset_mode == "all" else ""}.png'),
            show_carrier_counts=show_carrier_counts,
        )
        result["fig_supp"] = fig_supp
        if also_all_datasets_figure and dataset_mode != "all":
            result["fig_supp_all_datasets"] = plot_all_genes_or(
                df_ors, gene_to_dataset, gene_order=order_genes_by_disease(df_ors["Gene"].unique()),
                dataset_mode="all", save_path=str(figure_dir / "supp_OR_all_datasets.png"),
                show_carrier_counts=show_carrier_counts,
            )

        df_display = prepare_or_display_table(df_ors, gene_to_dataset, gene_order=gene_order, dataset_mode=dataset_mode)
        print(f"{len(df_display)} rows, {df_display['Gene'].nunique()} genes")
        result["df_display"] = df_display

        if output_or_display_csv:
            df_display.to_csv(output_or_display_csv, index=False)
            print(f"  Saved OR display table to {output_or_display_csv}")
    else:
        print("  SKIP OR-based panels/table: AoU OR estimates TSV unavailable")

    return result


if __name__ == "__main__":
    build_gene_performance_scatter()
