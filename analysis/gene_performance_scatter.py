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

- The two hardcoded OR-estimate CSVs -> ``analysis.config.OR_ESTIMATES_CSV``
  / ``analysis.config.YURIY_OR_CSV``, both ``warn_if_missing``-guarded (skip
  the OR-dependent panels/table if unavailable).

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

from src.assay_calibration.plot_utils.utils import compute_classification_metrics

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
    "KCNE1": "Cardiovascular", "KCNH2": "Cardiovascular", "KCNQ4": "Hearing loss",
    "MSH2": "Cancer", "OTC": "Metabolic", "PALB2": "Cancer", "PTEN": "Cancer",
    "RAD51C": "Cancer", "RAD51D": "Cancer", "SCN5A": "Cardiovascular",
    "TARDBP": "Rare disease", "TP53": "Cancer", "TSC2": "Cancer"
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

def load_or_estimates(
    or_estimates_csv: Optional[str] = None,
    yuriy_or_csv: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load (df_ors, df_ors_old), or (None, None) for whichever is missing."""
    or_estimates_csv = or_estimates_csv or cfg.OR_ESTIMATES_CSV
    yuriy_or_csv = yuriy_or_csv or cfg.YURIY_OR_CSV

    df_ors = None
    if not cfg.warn_if_missing(or_estimates_csv, "OR estimates CSV"):
        df_ors = pd.read_csv(or_estimates_csv)

    df_ors_old = None
    if not cfg.warn_if_missing(yuriy_or_csv, "Yuriy OR CSV"):
        df_ors_old = pd.read_csv(yuriy_or_csv)

    return df_ors, df_ors_old


def build_gene_to_dataset(df_ors_old: pd.DataFrame) -> Dict[str, str]:
    """Map gene -> single chosen dataset name (unchanged logic)."""
    gene_to_dataset = {}

    for gene in df_ors_old.Gene.unique():
        if gene == "TP53":
            dataset = "TP53_Fortuno_2021"
        else:
            dataset = df_ors_old[df_ors_old.Gene == gene].Dataset.unique()[0].replace(
                "rapgap_unpublished", "IGVF"
            ).replace("unpublished", "IGVF").replace(
                "RAD51C_Olvera-León_2024_z_score_D4_D14", "RAD51C_Olvera-León_2024"
            )

        gene_to_dataset[gene] = dataset

    gene_to_dataset["SCN5A"] = "SCN5A_Ma_2024"

    return gene_to_dataset


# ---------------------------------------------------------------------------
# Shared helpers (unchanged)
# ---------------------------------------------------------------------------

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
                          bottom_genes=['BRCA1', 'BRCA2', 'PALB2', 'RAD51C', 'RAD51D'],
                          size_by='vus',
                          annotate_gene_scatter=True, seed=42,
                          save_path=None, figsize=(18, 10)):

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
        if danz_df is None and auth_df is None:
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
                               rf"$\mathit{{{r['gene']}}}$", fontsize=FONTSIZE_LEGEND,
                               ha='center', va='center', fontweight='bold', zorder=10)

            # Add white border/stroke
            text.set_path_effects([
                pe.Stroke(linewidth=2.25, foreground='white'),  # White border
                pe.Normal()  # The text itself
            ])

            texts.append(text)

    if annotate_gene_scatter:
        np.random.seed(seed)
        try:
            adjust_text(texts, ax=ax_gene, arrowprops=dict(arrowstyle='-', color='gray', lw=1, alpha=0.5),
                       expand_points=(0.7, 0.7), force_text=0.3)
        except Exception:
            pass

    ax_gene.set_xlabel('Author Accuracy', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax_gene.set_ylabel('ExCALIBR Accuracy', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax_gene.set_xlim(plot_min * 0.95, plot_max+0.02)
    ax_gene.set_ylim(0.85 * 0.95, plot_max)
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

            all_classes_df = pd.DataFrame({'Classification': full_classification_order})

            if len(gene_data) > 0:
                all_classes_df = all_classes_df.merge(
                    gene_data[['Classification', 'Odds Ratio', 'OR_LI', 'OR_UI', 'significant', 'Cases with variants']],
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
                if pd.notna(row['Odds Ratio']) and row['Cases with variants'] > 0:
                    y = idx
                    strength = get_strength(row['Classification'])
                    bar_color = STRENGTH_COLOR.get(strength, '#999999')

                    ax.errorbar(row['Odds Ratio'], y,
                        xerr=[[row['Odds Ratio'] - row['OR_LI']], [row['OR_UI'] - row['Odds Ratio']]],
                        fmt='none', ecolor=bar_color, elinewidth=2, capsize=3, capthick=2, alpha=1, zorder=1)

                    facecolor = bar_color if row['significant'] else 'none'
                    ax.scatter(row['Odds Ratio'], y, marker='o', s=50,
                              facecolors=facecolor, edgecolors=bar_color, linewidth=1.5, zorder=2)

            ax.axvline(x=1, linestyle='--', color='black', linewidth=1, alpha=0.5, zorder=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(all_classes_df['Classification'], fontsize=FONTSIZE_AXIS_TICK)
            ax.set_xlabel('Odds Ratio', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')

            show_y_ticks = i == 0
            if show_y_ticks:
                ax.set_ylabel('Evidence Strength', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
            ax.tick_params(axis='both', labelsize=FONTSIZE_AXIS_TICK, left=show_y_ticks, labelleft=show_y_ticks)

            # For each gene, set limits individually
            if len(gene_data) > 0:
                data_max = gene_data['OR_UI'].max()
                if pd.notna(data_max):
                    # Round up to nearest clean limit
                    clean_limits = [3, 5, 10, 30, 50, 100, 1000, 10000]
                    xlim_max = next((lim for lim in clean_limits if lim >= data_max), 10000)
                else:
                    xlim_max = 10
            else:
                xlim_max = 10

            xlim_min = compute_log_xmin(xlim_max, x_ref=1.0, fraction=0.05) # based on plot proportion
            data_min = shrink_data_min_log(gene_data['OR_LI'].min(), xlim_max, fraction=0.05)
            if data_min < xlim_min:
                xlim_min = data_min # based on data

            ax.grid(True, alpha=0.2, axis='x', which='major')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(rf"$\mathit{{{gene_name}}}$", fontsize=FONTSIZE_AXIS_LABEL)

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

def plot_all_genes_or(df_ors, gene_to_dataset, gene_order=None, save_path=None, figsize=(18, 12)):
    """
    Create supplementary figure with OR plots for all genes.

    Parameters
    ----------
    df_ors : pd.DataFrame
        DataFrame with columns ['Gene', 'Dataset', 'Classification', 'Odds Ratio',
                                'OR_LI', 'OR_UI', 'LogOR', 'LogOR_LI', 'LogOR_UI',
                                'Wald p-value']
    gene_to_dataset : dict
        Maps gene name -> dataset name (selects one dataset per gene).
    gene_order : list, optional
        Order of genes to display. If None, sorts alphabetically.
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size (width, height)
    """

    # Get all unique genes from data
    if gene_order is None:
        all_genes = sorted(df_ors['Gene'].unique())
    else:
        all_genes = gene_order

    # Arrange in 3 rows: 5, 5, 3
    n_genes = len(all_genes)

    # Create figure with 3 rows
    fig = plt.figure(figsize=figsize)

    # Create grid: 3 rows with different column counts
    gs = gridspec.GridSpec(3, 5, figure=fig,
                          left=0.05, right=0.95, top=0.95, bottom=0.05,
                          hspace=0.4, wspace=0.15)

    # Create axes for each gene
    ax_genes = []
    for row in range(3):
        if row < 2:  # First two rows: 5 columns each
            for col in range(5):
                idx = row * 5 + col
                if idx < n_genes:
                    ax_genes.append(fig.add_subplot(gs[row, col]))
        else:  # Third row: 3 columns, centered
            for col in range(0, 4):  # Columns 1, 2, 3 (centered)
                idx = row * 5 + col - 1
                if idx < n_genes:
                    ax_genes.append(fig.add_subplot(gs[row, col]))

    FONTSIZE_AXIS_LABEL = 14
    FONTSIZE_AXIS_TICK = 12

    # Classification order
    full_classification_order = FULL_CLASSIFICATION_ORDER

    get_strength = _get_strength

    # Filter and process data
    df_filtered = df_ors[df_ors['Classification'] != "0"].copy()
    df_filtered['significant'] = ((df_filtered['LogOR_LI'] > 0) | (df_filtered['LogOR_UI'] < 0))

    for i, (ax, gene_name) in enumerate(zip(ax_genes, all_genes)):

        # select gene and one dataset
        gene_data = df_filtered[(df_filtered['Gene'] == gene_name) & (df_filtered['Dataset'] == gene_to_dataset[gene_name])].copy()

        # Create complete dataframe with all classification levels
        all_classes_df = pd.DataFrame({'Classification': full_classification_order})

        if len(gene_data) > 0:
            all_classes_df = all_classes_df.merge(
                gene_data[['Classification', 'Odds Ratio', 'OR_LI', 'OR_UI', 'significant', 'Cases with variants']],
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
            if pd.notna(row['Odds Ratio']) and row['Cases with variants'] > 0:
                y = idx
                strength = get_strength(row['Classification'])
                bar_color = STRENGTH_COLOR.get(strength, '#999999')

                ax.errorbar(row['Odds Ratio'], y,
                    xerr=[[row['Odds Ratio'] - row['OR_LI']], [row['OR_UI'] - row['Odds Ratio']]],
                    fmt='none', ecolor=bar_color, elinewidth=2, capsize=3, capthick=2, alpha=1, zorder=1)

                facecolor = bar_color if row['significant'] else 'none'
                ax.scatter(row['Odds Ratio'], y, marker='o', s=50,
                          facecolors=facecolor, edgecolors=bar_color, linewidth=1.5, zorder=2)

        # Vertical line at OR = 1
        ax.axvline(x=1, linestyle='--', color='black', linewidth=1, alpha=0.5, zorder=1)

        # Styling
        ax.set_xscale('log')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_classes_df['Classification'], fontsize=FONTSIZE_AXIS_TICK)

        # Only show x-label on bottom row
        if i >= len(all_genes) - 4 or i == 9:  # Bottom row plus the overhangs
            ax.set_xlabel('Odds Ratio', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')

        # Only show y-label on leftmost column
        show_y_ticks = (i % 5 == 0)
        if show_y_ticks:
            ax.set_ylabel('Evidence Strength', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
        ax.tick_params(axis='both', labelsize=FONTSIZE_AXIS_TICK, left=show_y_ticks, labelleft=show_y_ticks)

        # Set x-axis limits
        if len(gene_data) > 0:
            data_max = gene_data['OR_UI'].max()
            if pd.notna(data_max):
                clean_limits = [3, 5, 10, 30, 50, 100, 1000, 10000]
                xlim_max = next((lim for lim in clean_limits if lim >= data_max), 10000)
            else:
                xlim_max = 10
        else:
            xlim_max = 10

        xlim_min = compute_log_xmin(xlim_max, x_ref=1.0, fraction=0.05) # based on plot proportion
        data_min = shrink_data_min_log(gene_data['OR_LI'].min(), xlim_max, fraction=0.05)
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

        # Title with gene name only (disease will be shown in brackets above)
        ax.set_title(rf"$\mathit{{{gene_name}}}$",
                    fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')

    # ========== ADD DISEASE GROUP BRACKETS ==========

    # Group genes by disease for bracket annotations
    gene_diseases = [(gene, GENE_DISEASE_MAP.get(gene, "Unknown")) for gene in all_genes]

    # We need to track which row each gene is in
    gene_to_position = {}
    for idx, gene in enumerate(all_genes):
        row = idx // 5
        col_in_row = idx % 5
        if row == 2:  # Bottom row is offset
            col_in_row += 1
        gene_to_position[gene] = (row, idx)

    # Group consecutive genes with same disease within each row
    current_row = -1
    disease_groups = []
    current_disease = None
    current_genes = []

    for gene, disease in gene_diseases:
        row = gene_to_position[gene][0]

        # New row or different disease
        if row != current_row or disease != current_disease:
            if current_genes:
                disease_groups.append((current_disease, current_genes, current_row))
            current_disease = disease
            current_genes = [gene]
            current_row = row
        else:
            current_genes.append(gene)

    # Don't forget the last group
    if current_genes:
        disease_groups.append((current_disease, current_genes, current_row))

    # Draw brackets for each disease group
    for disease, genes, row in disease_groups:
        # Find the axes indices for first and last gene in this group
        start_gene_idx = all_genes.index(genes[0])
        end_gene_idx = all_genes.index(genes[-1])

        # Get the axis positions in figure coordinates
        bbox_start = ax_genes[start_gene_idx].get_position()
        bbox_end = ax_genes[end_gene_idx].get_position()

        # Calculate bracket positions
        x_left = bbox_start.x0
        x_right = bbox_end.x1
        y_bracket = bbox_start.y1 + 0.02  # Slightly above the axes
        bracket_height = 0.006

        # Draw the bracket (horizontal line with vertical ends)
        fig.add_artist(plt.Line2D([x_left, x_left], [y_bracket, y_bracket + bracket_height],
                                 transform=fig.transFigure, color='black', linewidth=1.5))
        fig.add_artist(plt.Line2D([x_left, x_right], [y_bracket + bracket_height, y_bracket + bracket_height],
                                 transform=fig.transFigure, color='black', linewidth=1.5))
        fig.add_artist(plt.Line2D([x_right, x_right], [y_bracket, y_bracket + bracket_height],
                                 transform=fig.transFigure, color='black', linewidth=1.5))

        # Add disease label centered above bracket
        x_center = (x_left + x_right) / 2
        y_text = y_bracket + bracket_height + 0.007

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

def prepare_or_display_table(df_ors, gene_to_dataset, gene_order=None):
    """
    Filter df_ors to only the columns and rows needed for plot_all_genes_or.

    Parameters
    ----------
    df_ors : pd.DataFrame
        Full OR results table.
    gene_to_dataset : dict
        Maps gene name -> dataset name (selects one dataset per gene).
    gene_order : list, optional
        Genes to include. If None, uses all keys in gene_to_dataset.

    Returns
    -------
    pd.DataFrame
        Minimal table with only the rows/columns used for plotting.
    """
    if gene_order is None:
        gene_order = sorted(gene_to_dataset.keys())

    # Keep only the selected dataset per gene
    rows = []
    for gene in gene_order:
        ds = gene_to_dataset.get(gene)
        if ds is None:
            continue
        mask = (df_ors['Gene'] == gene) & (df_ors['Dataset'] == ds)
        rows.append(df_ors.loc[mask])

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
    or_estimates_csv: Optional[str] = None,
    yuriy_or_csv: Optional[str] = None,
    annotate_gene_scatter: bool = True,
    output_or_display_csv: Optional[str] = None,
) -> Dict:
    """Rebuild the gene-performance-scatter / evidence-strength / OR figures.

    Replaces the notebook's top-level script logic. Any input that can't be
    found on disk is skipped (with a warning) rather than crashing:
      - evidence counts (datasets_reached_evidence/*.pkl) missing -> panels
        B/C of the combined figure, and the combined figure itself, are
        skipped.
      - OR estimate CSVs missing -> OR panels/supplementary figure/display
        table are skipped.

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

    evidence_counts = load_evidence_counts(evidence_counts_dir)
    df_ors, df_ors_old = load_or_estimates(or_estimates_csv, yuriy_or_csv)

    gene_to_dataset = None
    if df_ors_old is not None:
        gene_to_dataset = build_gene_to_dataset(df_ors_old)
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
            df_ors=df_ors if gene_to_dataset is not None else None,
            bottom_genes=sorted(['BAP1', 'BRCA1', 'MSH2', 'TSC2', 'RAD51D']) + sorted(['KCNQ4']),
            save_path=str(figure_dir / f'gene_brnich_OR{"_noannot" if not annotate_gene_scatter else ""}.png'),
            seed=42,
        )
        result["fig_combined"] = fig
        result["gene_results"] = gene_results
    else:
        print("  SKIP combined plot: evidence counts unavailable")

    if df_ors is not None and gene_to_dataset is not None:
        gene_order = ['BAP1', 'BARD1', 'BRCA1', 'BRCA2', 'MSH2', 'PALB2', 'RAD51C', 'RAD51D',
                      'TP53', 'TSC2', 'KCNH2', 'SCN5A', 'KCNQ4', 'GCK']
        fig_supp = plot_all_genes_or(
            df_ors, gene_to_dataset, gene_order=gene_order,
            save_path=str(figure_dir / 'supp_OR.png'),
        )
        result["fig_supp"] = fig_supp

        df_display = prepare_or_display_table(df_ors, gene_to_dataset, gene_order=gene_order)
        print(f"{len(df_display)} rows, {df_display['Gene'].nunique()} genes")
        result["df_display"] = df_display

        if output_or_display_csv:
            df_display.to_csv(output_or_display_csv, index=False)
            print(f"  Saved OR display table to {output_or_display_csv}")
    else:
        print("  SKIP OR-based panels/table: OR estimate CSV(s) unavailable")

    return result


if __name__ == "__main__":
    build_gene_performance_scatter()
