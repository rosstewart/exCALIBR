"""
Assay-level / dataset-level evidence statistics and the point-distribution
heatmap — ported verbatim (plotting/statistics logic unchanged) from
test/plot_author_calibration_confusion.py.

All three functions here need `assay_method_map` (a DataFrame with columns
dataset/vamp_sge/model_system/disease/IGVF_produced/gene) — see
analysis/config.py's ASSAY_METHOD_MAP_CSV and the reconstruction pipeline in
analysis/build_dataset_summary.py. Call sites should guard with
analysis.config.warn_if_missing and skip if that file isn't available yet,
same as other external-data-dependent figures.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_dataset_point_heatmap(dataset_info_df, all_danz_assignments,
                               assay_method_map=None,
                               igvf_image_path=None,
                               new_names_dict=None,
                               figsize=(16, 16),
                               mark_zeros='white',
                               log_scale=False,
                               sort_by='assay_type'):
    """
    Create heatmap of evidence point distributions with letter-coded categorical markers.

    Parameters
    ----------
    dataset_info_df : pd.DataFrame
        DataFrame with columns: dataset, n_variants, gene
    all_danz_assignments : np.ndarray
        Array of DanZ point assignments
    assay_method_map : pd.DataFrame, optional
        DataFrame with columns: dataset, vamp_sge, model_system, disease, IGVF_produced
    igvf_image_path : str, optional
        Path to image file for IGVF marker (if None, uses star symbol)
    new_names_dict : dict, optional
        {dataset -> display name}. Falls back to the raw dataset name for any
        dataset missing from the mapping (legacy script assumed every dataset
        had an entry; this is the only change from the original function).
    figsize : tuple
    mark_zeros : str
    log_scale : bool
    sort_by : str or list of str
        Column(s) to sort rows by. Options:
            'assay_type'    - VAMP-seq/SGE/Meta-analysis/Other  (default)
            'model_system'  - immortalized human cells / murine / yeast / etc.
            'disease'       - cancer / cardiovascular / rare disease / metabolic
            'gene'          - gene name alphabetical
            'dataset'       - dataset name alphabetical
            'igvf'          - IGVF-produced first
        Pass a list for multi-level sorting, e.g. ['disease', 'gene', 'dataset'].
        'dataset' is always appended as a tiebreaker if not already present.
    """
    new_names_dict = new_names_dict or {}

    all_points = list(range(-8, 9))
    negative_points = [p for p in all_points if p < 0]
    positive_points = [p for p in all_points if p > 0]

    # Letter code mappings
    VAMP_SGE_CODES = {
        'VAMP-seq': 'V', 'SGE': 'S', 'Meta-analysis': 'M', 'other': 'O', 'Other': 'O',
        None: 'O', np.nan: 'O',
    }
    MODEL_SYSTEM_CODES = {
        'immortalized human cells': 'H', 'murine primary cells': 'M', 'yeast': 'Y',
        'other': 'O', 'not applicable': 'N', None: 'N', np.nan: 'N',
    }
    DISEASE_CODES = {
        'cancer': 'C', 'cardiovascular': 'V', 'cardio': 'V', 'rare disease': 'R',
        'metabolic': 'M', None: 'O', np.nan: 'O',
    }

    # Color mappings
    VAMP_SGE_COLORS = {'V': '#2EB5AC', 'S': '#E77F3A', 'M': '#6C9EBF', 'O': '#9E9E9E'}
    MODEL_SYSTEM_COLORS = {'H': '#E8927A', 'M': '#D4A574', 'Y': '#C9B87C', 'N': '#BDBDBD', 'O': '#9E9E9E'}
    DISEASE_COLORS = {'C': '#D45F5F', 'V': '#5B8AC4', 'R': '#C5A87A', 'M': '#60B89F', 'O': '#9E9E9E'}

    vamp_sge_map, model_system_map, disease_map, igvf_map, gene_map = {}, {}, {}, {}, {}

    if assay_method_map is not None:
        for _, row in assay_method_map.iterrows():
            dataset = row['dataset']
            vamp_val = row.get('vamp_sge', None)
            model_val = row.get('model_system', None)

            if (pd.isna(vamp_val) or vamp_val == '' or vamp_val == 'not applicable') and \
               (pd.isna(model_val) or model_val == 'not applicable'):
                vamp_val = 'Meta-analysis'
            elif pd.isna(vamp_val) or vamp_val == '' or vamp_val == 'not applicable':
                vamp_val = 'Other'

            vamp_sge_map[dataset] = vamp_val
            model_system_map[dataset] = model_val if not pd.isna(model_val) else 'not applicable'
            disease_map[dataset] = row.get('disease', None)
            igvf_map[dataset] = row.get('IGVF_produced', False)
            gene_map[dataset] = row.get('gene', 'Other')

    dataset_proportions, dataset_names = [], []
    vamp_sge_vals, model_system_vals, disease_vals, is_igvf, genes = [], [], [], [], []

    vt_idx = 0
    for row_idx, row in dataset_info_df.iterrows():
        dataset = row['dataset']
        n_variants = row['n_variants']
        new_vt_idx = vt_idx + n_variants

        points, counts = np.unique(all_danz_assignments[vt_idx:new_vt_idx], return_counts=True)
        point_dict = dict(zip(points, counts))
        proportions = np.array([point_dict.get(p, 0) / n_variants for p in all_points])

        dataset_proportions.append(proportions)
        dataset_names.append(dataset)

        vamp_sge_vals.append(vamp_sge_map.get(dataset, 'Other'))
        model_system_vals.append(model_system_map.get(dataset, 'not applicable'))
        disease_vals.append(disease_map.get(dataset, None))
        is_igvf.append(igvf_map.get(dataset, False))
        genes.append(gene_map.get(dataset, row.get('gene', 'Other')))

        vt_idx = new_vt_idx

    proportion_df = pd.DataFrame(dataset_proportions, index=dataset_names, columns=all_points)

    # ── Sorting ──────────────────────────────────────────────────────
    vamp_sge_order = {'VAMP-seq': 0, 'SGE': 1, 'Meta-analysis': 2, 'other': 3, 'Other': 3}
    model_system_order = {
        'immortalized human cells': 0, 'murine primary cells': 1,
        'yeast': 2, 'other': 3, 'not applicable': 4,
    }
    disease_order = {'cancer': 0, 'cardiovascular': 1, 'cardio': 1, 'rare disease': 2, 'metabolic': 3}

    sort_df = pd.DataFrame({
        '_vamp_sge': vamp_sge_vals,
        '_vamp_sge_order': [vamp_sge_order.get(v, 3) for v in vamp_sge_vals],
        '_model_system': model_system_vals,
        '_model_system_order': [model_system_order.get(v, 4) for v in model_system_vals],
        '_disease': disease_vals,
        '_disease_order': [disease_order.get(v, 99) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 99 for v in disease_vals],
        '_gene': genes,
        '_igvf': is_igvf,
        '_igvf_order': [0 if ig else 1 for ig in is_igvf],
        '_dataset': dataset_names,
    })

    SORT_KEY_MAP = {
        'assay_type': '_vamp_sge_order', 'vamp_sge': '_vamp_sge_order',
        'model_system': '_model_system_order', 'disease': '_disease_order',
        'gene': '_gene', 'dataset': '_dataset', 'igvf': '_igvf_order',
    }

    if isinstance(sort_by, str):
        sort_by = [sort_by]

    sort_cols = []
    for key in sort_by:
        col = SORT_KEY_MAP.get(key)
        if col is None:
            raise ValueError(f"Unknown sort_by key '{key}'. Choose from: {list(SORT_KEY_MAP.keys())}")
        if col not in sort_cols:
            sort_cols.append(col)
    if '_dataset' not in sort_cols:
        sort_cols.append('_dataset')

    sort_df = sort_df.sort_values(sort_cols, ascending=True)
    proportion_df = proportion_df.reindex(sort_df['_dataset'].values)

    # ── Create figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.05, 0.12, 0.50, 0.83])

    blue_cmap = LinearSegmentedColormap.from_list("blue_gradient", ['#F0F8FC', '#99C8DC', '#7AB5D1', '#4B91A6', '#2E6B7E'])
    red_cmap = LinearSegmentedColormap.from_list("red_gradient", ['#FCF0F2', '#E6B1B8', '#D68F99', '#B85C6B', '#943744'])
    gray_cmap = LinearSegmentedColormap.from_list("gray_gradient", ['#F5F5F5', '#CCCCCC', '#999999', '#666666'])

    neg_idx = [all_points.index(p) for p in negative_points]
    zero_idx = [all_points.index(0)]
    pos_idx = [all_points.index(p) for p in positive_points]

    plot_data = proportion_df.values.copy()

    neg_data = np.full_like(plot_data, np.nan)
    neg_data[:, neg_idx] = plot_data[:, neg_idx]
    if mark_zeros == 'white':
        neg_data = np.ma.masked_where(neg_data == 0, neg_data)

    zero_data = np.full_like(plot_data, np.nan)
    zero_data[:, zero_idx] = plot_data[:, zero_idx]
    if mark_zeros == 'white':
        zero_data = np.ma.masked_where(zero_data == 0, zero_data)

    pos_data = np.full_like(plot_data, np.nan)
    pos_data[:, pos_idx] = plot_data[:, pos_idx]
    if mark_zeros == 'white':
        pos_data = np.ma.masked_where(pos_data == 0, pos_data)

    if log_scale:
        non_zero_vals = plot_data[plot_data > 0]
        if len(non_zero_vals) > 0:
            vmin = max(non_zero_vals.min(), 1e-4)
            vmax = non_zero_vals.max()
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
    else:
        norm = None
        vmin, vmax = 0, 1

    ax.imshow(neg_data, aspect='auto', cmap=blue_cmap, norm=norm,
              vmin=vmin if not log_scale else None, vmax=vmax if not log_scale else None)
    im_zero = ax.imshow(zero_data, aspect='auto', cmap=gray_cmap, norm=norm,
                        vmin=vmin if not log_scale else None, vmax=vmax if not log_scale else None)
    ax.imshow(pos_data, aspect='auto', cmap=red_cmap, norm=norm,
              vmin=vmin if not log_scale else None, vmax=vmax if not log_scale else None)

    cbar = plt.colorbar(im_zero, ax=ax, shrink=0.4, pad=0.16, aspect=15)
    cbar_label = 'Proportion of variants' + (' (log scale)' if log_scale else '')
    cbar.set_label(cbar_label, fontsize=11, fontweight='bold', labelpad=8)
    cbar.ax.tick_params(labelsize=9)

    for i in range(proportion_df.shape[0] + 1):
        ax.axhline(y=i - 0.5, color='white', linewidth=1, alpha=0.5, zorder=10)

    y_offset = 0.55
    for i in range(proportion_df.shape[0]):
        for j in range(proportion_df.shape[1]):
            if proportion_df.iloc[i, j] != 0:
                ax.plot([j-0.5, j-0.5], [i-y_offset, i+0.5], color="#E0E0E0", lw=1)
                ax.plot([j-0.5, j+0.5], [i-y_offset, i-y_offset], color="#E0E0E0", lw=1)
                ax.plot([j+0.5, j+0.5], [i-y_offset, i+0.5], color="#E0E0E0", lw=1)
                ax.plot([j-0.5, j+0.5], [i+0.5, i+0.5], color="#E0E0E0", lw=1)

    ax.set_xlabel('Evidence points', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Dataset', fontsize=14, fontweight='bold', labelpad=10)

    xlabels = [f"{p:+d}" if p != 0 else "0" for p in all_points]
    ax.set_xticks(np.arange(len(all_points)))
    ax.set_xticklabels(xlabels, rotation=0, fontsize=11, fontweight='500')

    igvf_image = None
    if igvf_image_path is not None:
        try:
            igvf_image = plt.imread(igvf_image_path)
        except Exception as e:
            print(f"Could not load IGVF image: {e}")
            igvf_image = None

    clean_names = []
    for i, dataset in enumerate(proportion_df.index):
        if igvf_map.get(dataset, False) and not dataset.startswith("HMBS") and not dataset.startswith("TP53"):
            if igvf_image is not None:
                imagebox = OffsetImage(igvf_image, zoom=0.04)
                ab = AnnotationBbox(imagebox, (-0.02, i+0.06),
                                   xycoords=('axes fraction', 'data'),
                                   frameon=False, box_alignment=(1, 0.5))
                ax.add_artist(ab)
            clean_names.append(new_names_dict.get(dataset, dataset) + '      ')
        else:
            clean_names.append(new_names_dict.get(dataset, dataset))

    ax.set_yticks(np.arange(len(proportion_df)))
    ax.set_yticklabels(clean_names, rotation=0, fontsize=8)

    left, right = ax.get_xlim()
    ax.set_xlim(left - 0.3, right)

    zero_idx_pos = all_points.index(0)
    benign_center = zero_idx_pos / 2
    pathogenic_center = (zero_idx_pos + len(all_points)) / 2

    ax.text(benign_center, -0.8, '← Benign Evidence', ha='center', va='bottom',
           fontsize=12, style='italic', fontweight='bold', color='#2166AC')
    ax.text(pathogenic_center, -0.8, 'Pathogenic Evidence →', ha='center', va='bottom',
           fontsize=12, style='italic', fontweight='bold', color='#B2182B')

    marker_width = 0.05
    marker_spacing = 0.01
    marker_col1_x = 1.05
    marker_col2_x = marker_col1_x + marker_width + marker_spacing
    marker_col3_x = marker_col2_x + marker_width + marker_spacing

    ax_pos = ax.get_position()
    header_y_fig = ax_pos.y0 - 0.004

    def axis_to_fig_x(x_axis):
        return ax_pos.x0 + (x_axis - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]) * ax_pos.width - 0.022

    fig.text(axis_to_fig_x(marker_col1_x * (ax.get_xlim()[1] - ax.get_xlim()[0])),
             header_y_fig, 'Assay Type', ha='left', va='top', fontsize=10,
             rotation=-45, transform=fig.transFigure)
    fig.text(axis_to_fig_x(marker_col2_x * (ax.get_xlim()[1] - ax.get_xlim()[0])),
             header_y_fig, 'Model System', ha='left', va='top', fontsize=10,
             rotation=-45, transform=fig.transFigure)
    fig.text(axis_to_fig_x(marker_col3_x * (ax.get_xlim()[1] - ax.get_xlim()[0])),
             header_y_fig, 'Disease', ha='left', va='top', fontsize=10,
             rotation=-45, transform=fig.transFigure)

    for idx, dataset in enumerate(proportion_df.index):
        y_pos = idx
        vamp_code = VAMP_SGE_CODES.get(sort_df.iloc[idx]['_vamp_sge'], 'O')
        model_code = MODEL_SYSTEM_CODES.get(sort_df.iloc[idx]['_model_system'], 'N')
        disease_code = DISEASE_CODES.get(sort_df.iloc[idx]['_disease'], 'O')

        vamp_color = VAMP_SGE_COLORS[vamp_code]
        rect1 = Rectangle((marker_col1_x - marker_width/2, y_pos - 0.4), marker_width, 0.8,
                         transform=ax.get_yaxis_transform(), facecolor=vamp_color,
                         edgecolor='black', linewidth=0.5, clip_on=False, zorder=19)
        ax.add_patch(rect1)
        ax.text(marker_col1_x, y_pos, vamp_code, transform=ax.get_yaxis_transform(),
               ha='center', va='center', fontsize=10, fontweight='bold', color='white', clip_on=False, zorder=20)

        model_color = MODEL_SYSTEM_COLORS[model_code]
        rect2 = Rectangle((marker_col2_x - marker_width/2, y_pos - 0.4), marker_width, 0.8,
                         transform=ax.get_yaxis_transform(), facecolor=model_color,
                         edgecolor='black', linewidth=0.5, clip_on=False, zorder=19)
        ax.add_patch(rect2)
        ax.text(marker_col2_x, y_pos, model_code, transform=ax.get_yaxis_transform(),
               ha='center', va='center', fontsize=10, fontweight='bold', color='white', clip_on=False, zorder=20)

        disease_color = DISEASE_COLORS[disease_code]
        rect3 = Rectangle((marker_col3_x - marker_width/2, y_pos - 0.4), marker_width, 0.8,
                         transform=ax.get_yaxis_transform(), facecolor=disease_color,
                         edgecolor='black', linewidth=0.5, clip_on=False, zorder=19)
        ax.add_patch(rect3)
        ax.text(marker_col3_x, y_pos, disease_code, transform=ax.get_yaxis_transform(),
               ha='center', va='center', fontsize=10, fontweight='bold', color='white', clip_on=False, zorder=20)

    if igvf_image is not None and igvf_image_path is not None:
        imagebox_legend = OffsetImage(igvf_image, zoom=0.07)
        ab_legend = AnnotationBbox(imagebox_legend, (1.005, 1.0007), xycoords='axes fraction',
                                   frameon=False, box_alignment=(1, 1))
        ax.add_artist(ab_legend)
        ax.text(0.94, 0.995, 'IGVF dataset', transform=ax.transAxes, ha='right', va='top', fontsize=10)

        from matplotlib.patches import FancyBboxPatch
        legend_box = FancyBboxPatch(
            (0.757, 0.9805), 0.265, 0.019, transform=ax.transAxes,
            boxstyle="round,pad=0.005", edgecolor='#999999', facecolor='white',
            linewidth=1, alpha=0.9, zorder=1, clip_on=False,
        )
        ax.add_patch(legend_box)
    else:
        igvf_legend = [Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
                             markersize=10, label='IGVF dataset', linestyle='None')]
        ax.legend(handles=igvf_legend, loc='upper right', fontsize=9, framealpha=0.9)

    vamp_legend_text = [
        ('V', 'VAMP-seq', VAMP_SGE_COLORS['V']), ('S', 'SGE', VAMP_SGE_COLORS['S']),
        ('M', 'Meta-analysis', VAMP_SGE_COLORS['M']), ('O', 'Other', VAMP_SGE_COLORS['O']),
    ]
    LEGEND_X = 0.55
    fig.text(LEGEND_X, 0.80, 'Assay Type:', fontsize=11, fontweight='bold')
    for i, (code, label, color) in enumerate(vamp_legend_text):
        y_pos = 0.78 - i*0.025
        rect = Rectangle((LEGEND_X, y_pos - 0.007), 0.015, 0.015, transform=fig.transFigure,
                        facecolor=color, edgecolor='black', linewidth=0.5)
        fig.add_artist(rect)
        fig.text(LEGEND_X+0.02, y_pos, f'{code} = {label}', fontsize=9, va='center')

    model_legend_text = [
        ('H', 'Immortalized\nhuman cells', MODEL_SYSTEM_COLORS['H']),
        ('M', 'Murine\nprimary cells', MODEL_SYSTEM_COLORS['M']),
        ('Y', 'Yeast', MODEL_SYSTEM_COLORS['Y']),
        ('N', 'Not applicable', MODEL_SYSTEM_COLORS['N']),
        ('O', 'Other', MODEL_SYSTEM_COLORS['O']),
    ]
    fig.text(LEGEND_X, 0.60, 'Model System:', fontsize=11, fontweight='bold')
    for i, (code, label, color) in enumerate(model_legend_text):
        y_pos = 0.58 - i*0.025
        rect = Rectangle((LEGEND_X, y_pos - 0.007), 0.015, 0.015, transform=fig.transFigure,
                        facecolor=color, edgecolor='black', linewidth=0.5)
        fig.add_artist(rect)
        fig.text(LEGEND_X+0.02, y_pos, f'{code} = {label}', fontsize=9, va='center')

    disease_legend_text = [
        ('C', 'Cancer', DISEASE_COLORS['C']), ('V', 'Cardiovascular', DISEASE_COLORS['V']),
        ('R', 'Rare disease', DISEASE_COLORS['R']), ('M', 'Metabolic', DISEASE_COLORS['M']),
    ]
    fig.text(LEGEND_X, 0.40, 'Disease:', fontsize=11, fontweight='bold')
    for i, (code, label, color) in enumerate(disease_legend_text):
        y_pos = 0.38 - i*0.025
        rect = Rectangle((LEGEND_X, y_pos - 0.007), 0.015, 0.015, transform=fig.transFigure,
                        facecolor=color, edgecolor='black', linewidth=0.5)
        fig.add_artist(rect)
        fig.text(LEGEND_X+0.02, y_pos, f'{code} = {label}', fontsize=9, va='center')

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax, proportion_df, sort_df


def compute_assay_evidence_stats(dataset_info_df, all_danz_assignments, assay_method_map):
    """
    For each dataset, compute the proportion of variants reaching various
    evidence thresholds. Then summarize by assay type, model system, and disease.

    Parameters
    ----------
    dataset_info_df : pd.DataFrame
        DataFrame with columns: dataset, n_variants, n_plp, n_blb, n_gnomad,
        n_synonymous, n_vus, gene
    all_danz_assignments : np.ndarray
        Array of DanZ point assignments
    assay_method_map : pd.DataFrame
        DataFrame with columns: dataset, vamp_sge, model_system, disease, IGVF_produced

    Returns
    -------
    dataset_stats : pd.DataFrame
        Per-dataset stats with metadata columns.
    summary_dict : dict
        Keys are ('assay_type', 'model_system', 'disease'), values are
        DataFrames summarizing counts/percentages per group.
    """
    meta = {}
    for _, row in assay_method_map.iterrows():
        ds = row['dataset']
        vamp_val = row.get('vamp_sge', None)
        model_val = row.get('model_system', None)

        if (pd.isna(vamp_val) or vamp_val in ('', 'not applicable')) and \
           (pd.isna(model_val) or model_val == 'not applicable'):
            vamp_val = 'Meta-analysis'
        elif pd.isna(vamp_val) or vamp_val in ('', 'not applicable'):
            vamp_val = 'Other'

        meta[ds] = {
            'assay_type': vamp_val,
            'model_system': model_val if not pd.isna(model_val) else 'not applicable',
            'disease': row.get('disease', 'Other'),
            'gene': row.get('gene', ds.split('_')[0]),
        }

    control_cols = ['n_plp', 'n_blb', 'n_gnomad', 'n_synonymous']
    has_controls = all(c in dataset_info_df.columns for c in control_cols)
    optional_cols = ['n_vus']

    records = []
    vt_idx = 0
    for _, row in dataset_info_df.iterrows():
        ds = row['dataset']
        n = row['n_variants']
        pts = all_danz_assignments[vt_idx:vt_idx + n]
        vt_idx += n

        m = meta.get(ds, {})
        rec = {
            'dataset': ds,
            'gene': m.get('gene', ds.split('_')[0]),
            'assay_type': m.get('assay_type', 'Other'),
            'model_system': m.get('model_system', 'Other'),
            'disease': m.get('disease', 'Other'),
            'n_variants': n,
            'pct_pathogenic': (pts > 0).sum() / n * 100,
            'pct_benign': (pts < 0).sum() / n * 100,
            'pct_indeterminate': (pts == 0).sum() / n * 100,
            'pct_ge_plus1': (pts >= 1).sum() / n * 100,
            'pct_ge_plus2': (pts >= 2).sum() / n * 100,
            'pct_ge_plus4': (pts >= 4).sum() / n * 100,
            'pct_ge_plus8': (pts >= 8).sum() / n * 100,
            'pct_le_minus1': (pts <= -1).sum() / n * 100,
            'pct_le_minus2': (pts <= -2).sum() / n * 100,
            'pct_le_minus4': (pts <= -4).sum() / n * 100,
            'pct_le_minus8': (pts <= -8).sum() / n * 100,
            'reaches_ge_plus1': (pts >= 1).any(),
            'reaches_ge_plus2': (pts >= 2).any(),
            'reaches_ge_plus4': (pts >= 4).any(),
            'reaches_ge_plus8': (pts >= 8).any(),
            'reaches_le_minus1': (pts <= -1).any(),
            'reaches_le_minus2': (pts <= -2).any(),
            'reaches_le_minus4': (pts <= -4).any(),
            'reaches_le_minus8': (pts <= -8).any(),
            'pct_any_evidence': (pts != 0).sum() / n * 100,
            'reaches_any_evidence': (pts != 0).any(),
        }

        if has_controls:
            for cc in control_cols:
                rec[cc] = int(row.get(cc, 0))
            rec['n_controls'] = sum(rec[cc] for cc in control_cols)
        for oc in optional_cols:
            if oc in dataset_info_df.columns:
                rec[oc] = int(row.get(oc, 0))

        records.append(rec)

    dataset_stats = pd.DataFrame(records)

    strat_cols = ['assay_type', 'model_system', 'disease']
    pct_cols = [c for c in dataset_stats.columns if c.startswith('pct_')]
    reaches_cols = [c for c in dataset_stats.columns if c.startswith('reaches_')]
    count_cols = [c for c in control_cols + ['n_controls', 'n_vus'] if c in dataset_stats.columns]

    summary_dict = {}
    for strat in strat_cols:
        rows = []
        for group, gdf in dataset_stats.groupby(strat):
            n_datasets = len(gdf)
            n_variants = gdf['n_variants'].sum()
            rec = {strat: group, 'n_datasets': n_datasets, 'n_variants': n_variants}
            for cc in count_cols:
                rec[f'{cc}_total'] = int(gdf[cc].sum())
                rec[f'{cc}_median'] = gdf[cc].median()
                rec[f'{cc}_mean'] = gdf[cc].mean()
            for sc in pct_cols:
                rec[f'{sc}_mean'] = gdf[sc].mean()
                rec[f'{sc}_median'] = gdf[sc].median()
            total_variants = gdf['n_variants'].sum()
            for sc in pct_cols:
                rec[f'{sc}_wmean'] = (gdf[sc] * gdf['n_variants']).sum() / total_variants if total_variants > 0 else 0.0
            for sc in pct_cols:
                rec[f'{sc}_n_above50'] = (gdf[sc] > 50).sum()
            for rc in reaches_cols:
                rec[f'{rc}_count'] = gdf[rc].sum()
            rows.append(rec)
        summary_dict[strat] = pd.DataFrame(rows).sort_values('n_datasets', ascending=False)

    return dataset_stats, summary_dict


def print_assay_evidence_report(dataset_stats, summary_dict):
    """Pretty-print the key findings."""
    has_controls = 'n_controls' in dataset_stats.columns

    print("=" * 80)
    print("ASSAY-LEVEL EVIDENCE STATISTICS")
    print("=" * 80)

    for strat, sdf in summary_dict.items():
        print(f"\n{'─' * 80}")
        print(f"Stratified by: {strat.upper()}")
        print(f"{'─' * 80}")

        for _, row in sdf.iterrows():
            group = row[strat]
            n_ds = int(row['n_datasets'])
            n_var = int(row['n_variants'])
            print(f"\n  {group} ({n_ds} datasets, {n_var:,} variants)")

            if has_controls:
                print(f"    Control variants (total across datasets):")
                print(f"      P/LP: {int(row['n_plp_total']):,}"
                      f"  |  B/LB: {int(row['n_blb_total']):,}"
                      f"  |  gnomAD: {int(row['n_gnomad_total']):,}"
                      f"  |  Syn: {int(row['n_synonymous_total']):,}")
                print(f"    Control variants (median per dataset):")
                print(f"      P/LP: {row['n_plp_median']:.0f}"
                      f"  |  B/LB: {row['n_blb_median']:.0f}"
                      f"  |  gnomAD: {row['n_gnomad_median']:.0f}"
                      f"  |  Syn: {row['n_synonymous_median']:.0f}")
                if 'n_vus_total' in row:
                    print(f"      VUS: {int(row['n_vus_total']):,} total"
                          f"  (median {row['n_vus_median']:.0f}/dataset)")

            print(f"    Direction (variant-weighted across datasets):")
            print(f"      Pathogenic (>0): {row['pct_pathogenic_wmean']:.1f}%")
            print(f"      Benign    (<0):  {row['pct_benign_wmean']:.1f}%")
            print(f"      Indet.    (=0):  {row['pct_indeterminate_wmean']:.1f}%")

            print(f"    Datasets REACHING threshold (≥1 variant):")
            print(f"      ≥+1: {int(row['reaches_ge_plus1_count'])}/{n_ds}"
                  f"  |  ≤-1: {int(row['reaches_le_minus1_count'])}/{n_ds}")
            print(f"      ≥+2: {int(row['reaches_ge_plus2_count'])}/{n_ds}"
                  f"  |  ≤-2: {int(row['reaches_le_minus2_count'])}/{n_ds}")
            print(f"      ≥+4: {int(row['reaches_ge_plus4_count'])}/{n_ds}"
                  f"  |  ≤-4: {int(row['reaches_le_minus4_count'])}/{n_ds}")
            print(f"      ≥+8: {int(row['reaches_ge_plus8_count'])}/{n_ds}"
                  f"  |  ≤-8: {int(row['reaches_le_minus8_count'])}/{n_ds}")

            print(f"    Variant-weighted % reaching threshold:")
            print(f"      ≥+2: {row['pct_ge_plus2_wmean']:.1f}%  |  ≤-2: {row['pct_le_minus2_wmean']:.1f}%")
            print(f"      ≥+4: {row['pct_ge_plus4_wmean']:.1f}%  |  ≤-4: {row['pct_le_minus4_wmean']:.1f}%")
            print(f"      ≥+8: {row['pct_ge_plus8_wmean']:.1f}%  |  ≤-8: {row['pct_le_minus8_wmean']:.1f}%")

            print(f"    Datasets with >50% variants reaching:")
            print(f"      ≥+2: {int(row['pct_ge_plus2_n_above50'])}/{n_ds}"
                  f"  |  ≤-2: {int(row['pct_le_minus2_n_above50'])}/{n_ds}")
            print(f"      Any evidence: {int(row['pct_any_evidence_n_above50'])}/{n_ds}")

    print(f"\n{'=' * 80}")
    print("VAMP-seq vs SGE COMPARISON")
    print("=" * 80)

    for assay in ['VAMP-seq', 'SGE']:
        sub = dataset_stats[dataset_stats['assay_type'] == assay]
        n_ds = len(sub)
        if n_ds == 0:
            continue
        print(f"\n  {assay} ({n_ds} datasets):")
        if has_controls:
            print(f"    Control variants (median per dataset):")
            print(f"      P/LP: {sub['n_plp'].median():.0f}"
                  f"  |  B/LB: {sub['n_blb'].median():.0f}"
                  f"  |  gnomAD: {sub['n_gnomad'].median():.0f}"
                  f"  |  Syn: {sub['n_synonymous'].median():.0f}")
        print(f"    Datasets reaching (≥1 variant):")
        print(f"      ≥+2: {sub['reaches_ge_plus2'].sum()}/{n_ds}"
              f"  |  ≤-2: {sub['reaches_le_minus2'].sum()}/{n_ds}")
        print(f"      ≥+4: {sub['reaches_ge_plus4'].sum()}/{n_ds}"
              f"  |  ≤-4: {sub['reaches_le_minus4'].sum()}/{n_ds}")
        print(f"      ≥+8: {sub['reaches_ge_plus8'].sum()}/{n_ds}"
              f"  |  ≤-8: {sub['reaches_le_minus8'].sum()}/{n_ds}")
        print(f"    Datasets where ≥50% variants reach:")
        print(f"      ≥+2: {(sub['pct_ge_plus2'] > 50).sum()}/{n_ds}"
              f"  |  ≤-2: {(sub['pct_le_minus2'] > 50).sum()}/{n_ds}")
        print(f"      Any evidence: {(sub['pct_any_evidence'] > 50).sum()}/{n_ds}")
        total_v = sub['n_variants'].sum()
        wpath = (sub['pct_pathogenic'] * sub['n_variants']).sum() / total_v
        wben = (sub['pct_benign'] * sub['n_variants']).sum() / total_v
        wind = (sub['pct_indeterminate'] * sub['n_variants']).sum() / total_v
        print(f"    Variant-weighted %: path={wpath:.1f}%"
              f"  ben={wben:.1f}%"
              f"  ind={wind:.1f}%"
              f"  ({total_v:,} variants)")


def plot_evidence_comparison(excalibr_path_counts, excalibr_ben_counts,
                             auth_path_counts, auth_ben_counts,
                             save_path=None, figsize=(14, 6)):
    """
    Create a grouped bar plot comparing ExCALIBR vs OddsPath/Author evidence counts.

    Parameters
    ----------
    excalibr_path_counts, excalibr_ben_counts : dict
        {evidence level (1/2/4/8): count of datasets reaching it} — ExCALIBR side.
        Fully reproducible from pipeline output — see
        analysis.gene_performance_scatter.compute_excalibr_evidence_counts.
    auth_path_counts, auth_ben_counts : dict
        Same, but for the author/OddsPath (Brnich et al.) side — requires the
        external OddsPath evidence-code CSV (not produced by this pipeline);
        pass all-zero dicts (or skip the call) if that file isn't available.
    save_path : str or Path, optional
    figsize : tuple

    Returns
    -------
    fig, (ax1, ax2)
    """
    levels = [1, 2, 4, 8]

    excalibr_colors = {
        8: '#943744', 4: '#B85C6B', 2: '#D68F99', 1: '#E6B1B8', 0: '#E0E0E0',
        -1: '#99C8DC', -2: '#7AB5D1', -4: '#4B91A6', -8: '#2E6B7E',
    }
    oddspath_colors = {
        8: '#6B2C91', 4: '#8E4BB8', 2: '#A96BC9', 1: '#C49EDB', 0: '#999999',
        -1: '#66C2A5', -2: '#3D9970', -4: '#2A7A5C', -8: '#1A5C42',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(levels))
    width = 0.35

    FONTSIZE_SMALL = 11
    FONTSIZE_AXIS_LABEL = 14
    FONTSIZE_AXIS_TICK = 12
    FONTSIZE_LEGEND = 12
    FONTSIZE_PANEL_LETTER = 18

    excalibr_path_vals = [excalibr_path_counts[lvl] for lvl in levels]
    auth_path_vals = [auth_path_counts[lvl] for lvl in levels]

    for i, lvl in enumerate(levels):
        ax1.bar(x[i] - width/2, excalibr_path_vals[i], width, color=excalibr_colors[lvl],
               alpha=0.9, edgecolor='black', linewidth=1.2, label='ExCALIBR' if i == 0 else '')
    for i, lvl in enumerate(levels):
        ax1.bar(x[i] + width/2, auth_path_vals[i], width, color=oddspath_colors[lvl],
               alpha=0.9, edgecolor='black', linewidth=1.2,
               label=r'Brnich $\it{et\ al.}$' if i == 0 else '')

    for i in range(len(levels)):
        ax1.text(x[i] - width/2, excalibr_path_vals[i], f'{int(excalibr_path_vals[i])}',
                ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')
        ax1.text(x[i] + width/2, auth_path_vals[i], f'{int(auth_path_vals[i])}',
                ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')

    ax1.set_xlabel('Evidence Level (Pathogenic)', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax1.set_ylabel('Number of Datasets', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'+{lvl}' for lvl in levels], fontsize=FONTSIZE_AXIS_TICK)
    ax1.legend(fontsize=FONTSIZE_LEGEND, frameon=True, shadow=True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(-0.05, 1.08, '(A)', transform=ax1.transAxes, fontsize=FONTSIZE_PANEL_LETTER,
            fontweight='bold', va='top', ha='left')

    excalibr_ben_vals = [excalibr_ben_counts[lvl] for lvl in levels]
    auth_ben_vals = [auth_ben_counts[lvl] for lvl in levels]

    for i, lvl in enumerate(levels):
        ax2.bar(x[i] - width/2, excalibr_ben_vals[i], width, color=excalibr_colors[-lvl],
               alpha=0.9, edgecolor='black', linewidth=1.2, label='ExCALIBR' if i == 0 else '')
    for i, lvl in enumerate(levels):
        ax2.bar(x[i] + width/2, auth_ben_vals[i], width, color=oddspath_colors[-lvl],
               alpha=0.9, edgecolor='black', linewidth=1.2,
               label=r'Brnich $\it{et\ al.}$' if i == 0 else '')

    for i in range(len(levels)):
        ax2.text(x[i] - width/2, excalibr_ben_vals[i], f'{int(excalibr_ben_vals[i])}',
                ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')
        ax2.text(x[i] + width/2, auth_ben_vals[i], f'{int(auth_ben_vals[i])}',
                ha='center', va='bottom', fontsize=FONTSIZE_SMALL, fontweight='bold')

    ax2.set_xlabel('Evidence Level (Benign)', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax2.set_ylabel('Number of Datasets', fontsize=FONTSIZE_AXIS_LABEL, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'-{lvl}' for lvl in levels], fontsize=FONTSIZE_AXIS_TICK)
    ax2.legend(fontsize=FONTSIZE_LEGEND, frameon=True, shadow=True)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(-0.05, 1.08, '(B)', transform=ax2.transAxes, fontsize=FONTSIZE_PANEL_LETTER,
            fontweight='bold', va='top', ha='left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig, (ax1, ax2)


def compute_excalibr_evidence_counts(dataset_info_df, all_danz_assignments) -> Tuple[Dict[int, int], Dict[int, int]]:
    """The ExCALIBR side of plot_evidence_comparison's inputs — fully
    reproducible from pipeline output (unlike the author/OddsPath side).

    Ported from the legacy script's df_point_distr loop: counts how many
    datasets have at least one variant reaching each pathogenic/benign
    evidence level.
    """
    levels = [1, 2, 4, 8]
    excalibr_path_counts = {lvl: 0 for lvl in levels}
    excalibr_ben_counts = {lvl: 0 for lvl in levels}

    vt_idx = 0
    for _, row in dataset_info_df.iterrows():
        n = row['n_variants']
        pts = all_danz_assignments[vt_idx:vt_idx + n]
        vt_idx += n
        for lvl in levels:
            if (pts >= lvl).any():
                excalibr_path_counts[lvl] += 1
            if (pts <= -lvl).any():
                excalibr_ben_counts[lvl] += 1

    return excalibr_path_counts, excalibr_ben_counts
