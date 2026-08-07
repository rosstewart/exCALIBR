"""
Utilities for evaluating and plotting ExCALIBR-MV vs. assay (z-score) performance.

Main entry points
-----------------
build_agg(results_df, gene_ms, percentile_analyses, compute_assay_labels_zscore,
          gene_to_analysis_cons_b0, non_nu_genes=None)
    -> agg dict (all genes or non-NU subset)

print_latex_table(agg, keep=None)
    -> prints LaTeX tabular with methods as columns, metrics as rows

plot_confusion_matrices(agg, title, save_path=None)
    -> calls plot_aggregate_confusion_matrices on pre-aggregated CMs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from ..plot_utils.utils import (
    compute_classification_metrics,
    plot_aggregate_confusion_matrices,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def points_to_confusion(labels, points):
    """Build 2×3 confusion matrix (B/LB, P/LP) × (Benign, Indeterminate, Pathogenic)."""
    cats = np.where(points > 0, 2, np.where(points < 0, 0, 1))
    cm   = np.zeros((2, 3), dtype=int)
    for row, col in zip(labels.astype(int), cats):
        cm[row, col] += 1
    return pd.DataFrame(cm,
                        index=['B/LB', 'P/LP'],
                        columns=['Benign', 'Indeterminate', 'Pathogenic'])


def _accumulate_into(target, key, config, labels, pts, snv_pts, n_snv, n_ctrl):
    cm = points_to_confusion(labels, pts)
    target[key]['TP']               += cm.iloc[1, 2]
    target[key]['TN']               += cm.iloc[0, 0]
    target[key]['FP']               += cm.iloc[0, 2]
    target[key]['FN']               += cm.iloc[1, 0]
    target[key]['uncertain_benign'] += cm.iloc[0, 1]
    target[key]['uncertain_path']   += cm.iloc[1, 1]
    target[key]['N_ctrl']           += n_ctrl
    if snv_pts is not None:
        target[key]['N_snv']        += n_snv
        target[key]['cov_snv_num']  += int((snv_pts != 0).sum())


def _agg_to_cm(agg, key):
    c = agg[key]
    return pd.DataFrame(
        [[c['TN'], c['uncertain_benign'], c['FP']],
         [c['FN'], c['uncertain_path'],   c['TP']]],
        index=['B/LB', 'P/LP'],
        columns=['Benign', 'Indeterminate', 'Pathogenic'])


def _safe(n, d):
    return n / d if d else float('nan')


# ── Build aggregation ─────────────────────────────────────────────────────────

def build_agg(results_df, gene_ms, percentile_analyses,
              compute_assay_labels_zscore, gene_to_analysis_cons_b0, config,
              non_nu_genes=None, assay_key="Assay"):
    """
    Aggregate confusion matrix counts across genes.

    Parameters
    ----------
    non_nu_genes : set or None
        If provided, restrict to these genes only (non-NU subset).
    assay_key : str
        Key under which assay-based labels are stored in the returned agg dict.
        Override when running multiple labelers so their results don't collide.

    Returns
    -------
    agg : defaultdict  keyed by method string
    """
    agg = defaultdict(lambda: defaultdict(int))

    for gene, gdf in results_df.groupby("gene"):
        if non_nu_genes is not None and gene not in non_nu_genes:
            continue

        ms        = gene_ms[gene]
        sa        = ms._sample_assignments
        is_plp    = sa[:, 0].astype(bool)
        is_blb    = sa[:, 1].astype(bool)
        is_snv    = sa[:, 2].astype(bool)
        eval_mask = is_plp | is_blb
        if eval_mask.sum() == 0:
            continue

        labels = is_plp[eval_mask]
        n_snv  = int(is_snv.sum())
        n_ctrl = int(eval_mask.sum())

        if compute_assay_labels_zscore is not None:
            assay_pts = compute_assay_labels_zscore(ms)
            _accumulate_into(agg, assay_key, config,
                             labels, assay_pts[eval_mask],
                             assay_pts[is_snv], n_snv, n_ctrl)

        for pctile, pdf in gdf.groupby("path_percentile"):
            r = gene_to_analysis_cons_b0 if pctile == 5 else \
                percentile_analyses.get(pctile, {})
            analysis = r.get(gene)
            if analysis is None:
                continue
            result = analysis.results.get(config)
            if result is None:
                continue
            pts = result["points"]
            key = f"MV p{pctile} ({config})"
            _accumulate_into(agg, key, config,
                             labels, pts[eval_mask],
                             pts[is_snv], n_snv, n_ctrl)
            # print(agg)

    return agg


# ── Category-based labeling ───────────────────────────────────────────────────

# Ordinal severity shared across all category strings.
# Neutral = 0; Potentially* = 1; Weakly* = 2; [base] = 3; Strongly* = 4.
_SEVERITY = {
    'Neutral':                0,
    'Potentially activating': 1, 'Weakly activating':   2,
    'Activating':             3, 'Strongly activating': 4,
    'Weakly inactivating':    2, 'Inactivating':        3,
    'Strongly inactivating':  4,
    'Potentially resistant':  1, 'Weakly resistant':    2,
    'Resistant':              3, 'Strongly resistant':  4,
}

_THRESHOLD_LEVEL = {'potentially': 1, 'weakly': 2, 'regular': 3, 'strongly': 4}

THRESHOLDS = ['potentially', 'weakly', 'regular', 'strongly']


def make_category_labeler(threshold):
    """
    Return a ``fn(ms) → np.ndarray{-1, 0, 1}`` that reads per-dimension raw
    func-class strings from ``ms._raw_fc_per_dim`` (populated by MultiScoreset)
    and applies threshold-based combining logic:

      - any dim severity ≥ threshold  →  1  (Abnormal)
      - all dims are Neutral (sev=0)  → -1  (Normal)
      - otherwise                     →  0  (Indeterminate)

    Parameters
    ----------
    threshold : str
        One of 'potentially', 'weakly', 'regular', 'strongly'.
    """
    min_level = _THRESHOLD_LEVEL[threshold]

    def _labeler(ms):
        if not hasattr(ms, '_raw_fc_per_dim'):
            raise AttributeError(
                "ms missing _raw_fc_per_dim — rebuild MultiScoreset with the "
                "updated dataset.py that stores per-dim raw func-class strings."
            )
        mat = ms._raw_fc_per_dim          # (N, D) object array of category strings
        N = mat.shape[0]
        pts = np.zeros(N, dtype=int)
        for i in range(N):
            sevs = np.array([_SEVERITY.get(str(v), 0) for v in mat[i]])
            if np.any(sevs >= min_level):
                pts[i] = 1
            elif np.all(sevs == 0):
                pts[i] = -1
            # else: 0 (indeterminate — above neutral but below threshold)
        return pts

    return _labeler


def build_agg_category_thresholds(results_df, gene_ms, percentile_analyses,
                                  gene_to_analysis_cons_b0,
                                  non_nu_genes=None,
                                  thresholds=THRESHOLDS):
    """
    Run ``build_agg`` once per threshold using the same ``gene_ms`` and merge
    all results into one agg dict.

    Category keys are named ``'Category (≥potentially)'`` etc.
    ExCALIBR-MV keys are accumulated from the first threshold run and are
    identical across runs, so they are not double-counted.

    Parameters
    ----------
    thresholds : list of str
        Subset of THRESHOLDS to evaluate.  Defaults to all four.

    Returns
    -------
    agg : defaultdict  keyed by method string
    """
    merged = defaultdict(lambda: defaultdict(int))
    mv_keys_seen = set()

    for thr in thresholds:
        labeler = make_category_labeler(thr)
        key     = f"Category (≥{thr})"
        sub     = build_agg(results_df, gene_ms, percentile_analyses,
                            labeler, gene_to_analysis_cons_b0,
                            non_nu_genes=non_nu_genes,
                            assay_key=key)
        for method, counts in sub.items():
            if method == key:
                # Category method — accumulate fresh each time
                for stat, val in counts.items():
                    merged[method][stat] += val
            elif method not in mv_keys_seen:
                # MV / percentile methods — copy once to avoid double-counting
                for stat, val in counts.items():
                    merged[method][stat] += val
        mv_keys_seen.update(k for k in sub if k != key)

    return merged


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _latex_val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    if isinstance(v, float) and np.isinf(v):
        return r'$\infty$'
    return f"{v:.3f}"


def _latex_int(v):
    return f"{v:,}"


def _pct(n, d):
    return f"{n:,} ({100 * n / d:.1f}\\%)" if d else '--'


_ROWS = [
    ('Total variants (controls)',
     lambda m: f"{m['N_ctrl']:,}", False),
    ('Determinate assignments',
     lambda m: _pct(m['determinate'], m['N_ctrl']), False),
    ('Indeterminate assignments',
     lambda m: _pct(m['uncertain'],   m['N_ctrl']), False),
    ('Coverage (missense SNV)',
     lambda m: _latex_val(m['cov_snv']), True),
    ('Accuracy',
     lambda m: _latex_val(m['accuracy']), False),
    ('Sensitivity',
     lambda m: _latex_val(m['sensitivity']), False),
    ('Specificity',
     lambda m: _latex_val(m['specificity']), False),
    ('MCC',
     lambda m: _latex_val(m['mcc']), True),
    (r'LR$^+$ (pathogenic vs.\ benign)',
     lambda m: _latex_val(m['lr_plus_standard']), False),
    (r'LR$^+$ (pathogenic vs.\ rest)',
     lambda m: _latex_val(m['lr_plus_pathogenic']), False),
    (r'LR$^+$ (benign vs.\ rest)',
     lambda m: _latex_val(m['lr_plus_benign']), False),
    (r'DOR (pathogenic vs.\ benign)',
     lambda m: _latex_val(m['dor_standard']), False),
    (r'DOR (pathogenic vs.\ rest)',
     lambda m: _latex_val(m['dor_pathogenic']), False),
    (r'DOR (benign vs.\ rest)',
     lambda m: _latex_val(m['dor_benign']), True),
]

DEFAULT_KEEP = {
    'MV p25 (3c_unc)': 'ExCALIBR-MV',
    'Assay':           'Synonymous z-score',
}


def print_latex_table(agg, keep=None):
    """
    Print a LaTeX tabular with methods as columns and metrics as rows.

    Parameters
    ----------
    keep : dict  {agg_key: display_name} — defaults to MV p25 vs Assay
    """
    if keep is None:
        keep = DEFAULT_KEEP

    method_vals = {}
    for key, display in keep.items():
        c  = agg[key]
        cm = _agg_to_cm(agg, key)
        m  = compute_classification_metrics(cm)
        m['N_ctrl']  = c['N_ctrl']
        m['N_snv']   = c['N_snv']
        m['cov_snv'] = _safe(c['cov_snv_num'], c['N_snv'])
        method_vals[display] = m

    dn = list(keep.values())
    print(r'\begin{tabular}{l' + 'c' * len(dn) + '}')
    print(r'\toprule')
    print('Metric & ' + ' & '.join(dn) + r' \\')
    print(r'\midrule')
    for label, fn, midrule in _ROWS:
        vals = [fn(method_vals[d]) for d in dn]
        print(label + ' & ' + ' & '.join(vals) + r' \\')
        if midrule:
            print(r'\midrule')
    print(r'\bottomrule')
    print(r'\end{tabular}')


# ── Build agg from pre-computed results_df ───────────────────────────────────

def build_agg_from_results_df(results_df, gene_ms, config='3c_unc', percentile=25):
    """Build an agg dict from a results_df produced by the per-gene metrics loop.

    results_df must have columns mv_TP, mv_TN, mv_FP, mv_FN (and assay_ equivalents)
    as written by full_metrics().  gene_ms is used only to derive the uncertain split
    (uncertain_path = n_plp - TP - FN, uncertain_benign = n_blb - TN - FP).

    Returns
    -------
    agg      : defaultdict compatible with plot_confusion_matrices
    mv_key   : the key used for the MV method
    assay_key: the key used for the assay method
    """
    mv_key    = f"MV p{percentile} ({config})"
    assay_key = 'Assay'
    agg = defaultdict(lambda: defaultdict(int))

    sub = results_df[
        (results_df['config'] == config) &
        (results_df['path_percentile'] == percentile)
    ]

    for _, row in sub.iterrows():
        gene = row['gene']
        ms   = gene_ms[gene]
        n_plp = int(ms._sample_assignments[:, 0].sum())
        n_blb = int(ms._sample_assignments[:, 1].sum())

        for prefix, key in [('mv', mv_key), ('assay', assay_key)]:
            tp = int(row[f'{prefix}_TP'])
            tn = int(row[f'{prefix}_TN'])
            fp = int(row[f'{prefix}_FP'])
            fn = int(row[f'{prefix}_FN'])

            agg[key]['TP']               += tp
            agg[key]['TN']               += tn
            agg[key]['FP']               += fp
            agg[key]['FN']               += fn
            agg[key]['uncertain_path']   += n_plp - tp - fn
            agg[key]['uncertain_benign'] += n_blb - tn - fp
            agg[key]['N_ctrl']           += n_plp + n_blb

    return agg, mv_key, assay_key


# ── Shared heatmap renderer (blue=Benign, gray=Indeterminate, red=Pathogenic) ──

def _draw_cm_heatmap(df, ax):
    """Render a confusion-matrix DataFrame as a styled heatmap on ax.

    Column color assignment:
      - contains 'Benign' / 'Normal'      → blue
      - contains 'Indeterminate' / 'IR'   → gray
      - contains 'Pathogenic' / 'Abnormal'→ red
      - anything else                     → gray
    Intensity is row-normalized; text flips white when cell > 45% of row max.
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as _mp

    blue_cmap = LinearSegmentedColormap.from_list('_b', ['#F0F8FC', '#99C8DC', '#4B91A6', '#2E6B7E'])
    red_cmap  = LinearSegmentedColormap.from_list('_r', ['#FCF0F2', '#E6B1B8', '#B85C6B', '#943744'])
    gray_cmap = LinearSegmentedColormap.from_list('_g', ['#F5F5F5', '#CCCCCC', '#999999', '#666666'])

    def _cmap_for(col_name):
        s = str(col_name)
        if any(k in s for k in ('Benign', 'Normal', 'BLB', 'B/LB')):
            return blue_cmap
        if any(k in s for k in ('Pathogenic', 'Abnormal', 'PLP', 'P/LP')):
            return red_cmap
        return gray_cmap

    nrows, ncols = df.shape
    for i in range(nrows):
        row_max = df.iloc[i].max()
        for j in range(ncols):
            val  = df.iloc[i, j]
            norm = val / row_max if row_max > 0 else 0
            color = _cmap_for(df.columns[j])(norm)
            ax.add_patch(_mp.Rectangle((j, i), 1, 1,
                                       facecolor=color, edgecolor='white', linewidth=2.5))
            text_color = 'white' if (row_max > 0 and norm > 0.45
                                     and 'Indeterminate' not in str(df.columns[j])
                                     and 'IR' not in str(df.columns[j])) else 'black'
            ax.text(j + 0.5, i + 0.5, f'{val:,}',
                    ha='center', va='center', fontsize=14, color=text_color)

    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(list(df.columns), rotation=0, fontsize=12)
    ax.set_yticklabels(list(df.index),   rotation=0, fontsize=12)
    ax.tick_params(length=0, labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('#F9F9F9')


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrices(agg,
                            mv_key='MV p25 (3c_unc)', assay_key='Assay',
                            mv_title='ExCALIBR-MV',
                            assay_title='Synonymous z-score',
                            figsize=(10, 4), save_path=None,
                            extra_row_mv=None, extra_row_assay=None):
    """
    Plot aggregate confusion matrices for ExCALIBR-MV vs. assay z-score.

    extra_row_mv / extra_row_assay : pd.DataFrame
        Optional extra rows (e.g. Sample 4 / RPV) appended to each panel.
        Must have the same columns as the base CM: ['Benign', 'Indeterminate', 'Pathogenic'].
        Build with e.g.:
            rpv_pts = analysis.results[config]['points'][rpv_mask]
            extra_row_mv = pd.DataFrame({
                'Benign':        [(rpv_pts < 0).sum()],
                'Indeterminate': [(rpv_pts == 0).sum()],
                'Pathogenic':    [(rpv_pts > 0).sum()],
            }, index=['RPV'])
    """
    mv_cm    = _agg_to_cm(agg, mv_key)
    assay_cm = _agg_to_cm(agg, assay_key)
    if extra_row_mv is not None:
        mv_cm = pd.concat([mv_cm, extra_row_mv])
    if extra_row_assay is not None:
        assay_cm = pd.concat([assay_cm, extra_row_assay])

    has_extra = (extra_row_mv is not None) or (extra_row_assay is not None)

    if not has_extra:
        # Original path — delegates to upstream function which computes full metrics
        fig, _, _ = plot_aggregate_confusion_matrices(
            danzs=[mv_cm],
            auths=[assay_cm],
            dataset_names=['aggregate'],
            figsize=figsize,
            letters=True,
        )
        fig.suptitle('')
        for ax in fig.axes:
            title = ax.get_title()
            if 'ExCALIBR Evidence' in title:
                ax.set_title(mv_title, fontsize=18, fontweight='bold', pad=10)
            elif 'Author Annotations' in title:
                ax.set_title(assay_title, fontsize=18, fontweight='bold', pad=10)
    else:
        # Self-contained path — avoids the 2×3 restriction in compute_classification_metrics
        fig = plt.figure(figsize=figsize)
        left = 0.08; gap = 0.05; pw = 0.35; bot = 0.15; top = 0.12
        ax_mv    = fig.add_axes([left,          bot, pw, 1 - bot - top])
        ax_assay = fig.add_axes([left + pw + gap, bot, pw, 1 - bot - top])

        _draw_cm_heatmap(mv_cm,    ax_mv)
        _draw_cm_heatmap(assay_cm, ax_assay)

        ax_mv.text(-0.10, 1.11, '(A)', transform=ax_mv.transAxes,
                   fontsize=18, fontweight='bold', va='top', ha='left')
        ax_assay.text(-0.10, 1.11, '(B)', transform=ax_assay.transAxes,
                      fontsize=18, fontweight='bold', va='top', ha='left')

        ax_mv.set_title(mv_title,    fontsize=18, fontweight='bold', pad=10)
        ax_assay.set_title(assay_title, fontsize=18, fontweight='bold', pad=10)
        ax_mv.set_xlabel('Evidence Direction',    fontsize=14)
        ax_mv.set_ylabel('ClinVar Classification', fontsize=14)
        ax_assay.set_xlabel('Functional Annotation', fontsize=14)
        ax_assay.set_ylabel('')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ── ExCALIBR-MV confusion matrices, one panel per gene ─────────────────────────

def plot_mv_confusion_by_gene(gene_ms, mv_points_by_gene,
                               include_assay=False,
                               compute_assay_labels_zscore=None,
                               non_nu_genes=None, ncols=3,
                               panel_size=(5.5, 5.5), save_path=None):
    """
    Plot a grid of ExCALIBR-MV-only confusion matrices, one per gene.

    Parameters
    ----------
    gene_ms : dict {gene: MultiScoreset}
    mv_points_by_gene : dict {gene: points_array}
        E.g. {g: a.results['3c_unc']['points']
              for g, a in gene_to_analysis_cons_b0_5.items()}
    include_assay : bool
        If True, also plot the synonymous z-score confusion matrix
        side-by-side with ExCALIBR-MV for each gene. Requires
        compute_assay_labels_zscore.
    compute_assay_labels_zscore : callable(ms) -> points_array or None
        Same function passed to build_agg; only needed if include_assay.
    non_nu_genes : set or None
        If provided, restrict to these genes only.
    ncols : int
        Number of gene columns in the subplot grid (each gene occupies
        one or two axes, depending on include_assay).
    panel_size : (w, h)
        Size of each individual panel.
    save_path : str or None

    Returns
    -------
    fig
    """
    if include_assay and compute_assay_labels_zscore is None:
        raise ValueError("include_assay=True requires compute_assay_labels_zscore")

    genes = []
    mv_cms = []
    assay_cms = []
    for gene, ms in gene_ms.items():
        if non_nu_genes is not None and gene not in non_nu_genes:
            continue
        mv_pts = mv_points_by_gene.get(gene)
        if mv_pts is None:
            continue

        sa = ms._sample_assignments
        is_plp = sa[:, 0].astype(bool)
        is_blb = sa[:, 1].astype(bool)
        eval_mask = is_plp | is_blb
        if eval_mask.sum() == 0:
            continue

        labels = is_plp[eval_mask]
        genes.append(gene)
        mv_cms.append(points_to_confusion(labels, mv_pts[eval_mask]))
        if include_assay:
            assay_pts = compute_assay_labels_zscore(ms)
            assay_cms.append(points_to_confusion(labels, assay_pts[eval_mask]))

    n = len(genes)
    nrows = -(-n // ncols)  # ceil

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    if include_assay:
        fig_w = panel_size[0] * 2.15 * ncols
        fig_h = panel_size[1] * 1.15 * nrows
    else:
        fig_w = panel_size[0] * ncols
        fig_h = panel_size[1] * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = GridSpec(nrows, ncols, figure=fig,
                     wspace=0.25, hspace=0.35,
                     left=0.04, right=0.98, top=0.9, bottom=0.05)

    for i, gene in enumerate(genes):
        row, col = divmod(i, ncols)

        if include_assay:
            inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row, col],
                                            wspace=0.12)
            ax_mv = fig.add_subplot(inner[0, 0])
            ax_assay = fig.add_subplot(inner[0, 1])

            _draw_cm_heatmap(mv_cms[i], ax_mv)
            _draw_cm_heatmap(assay_cms[i], ax_assay)
            for ax in (ax_mv, ax_assay):
                ax.tick_params(axis='x', labelsize=11)
                ax.tick_params(axis='y', labelsize=11)
            ax_mv.set_title('ExCALIBR-MV', fontsize=12, pad=4)
            ax_assay.set_title('Synonymous z-score', fontsize=12, pad=4)
            ax_assay.set_yticklabels([])
            ax_assay.set_ylabel('')

            pos_mv, pos_assay = ax_mv.get_position(), ax_assay.get_position()
            fig.text((pos_mv.x0 + pos_assay.x1) / 2, pos_mv.y1 + 0.025, gene,
                     ha='center', va='bottom', fontsize=15, fontweight='bold')
        else:
            ax_mv = fig.add_subplot(outer[row, col])
            _draw_cm_heatmap(mv_cms[i], ax_mv)
            ax_mv.tick_params(axis='x', labelsize=12)
            ax_mv.tick_params(axis='y', labelsize=12)
            ax_mv.set_title(gene, fontsize=15, fontweight='bold', pad=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


# ── RPV classification matrix ─────────────────────────────────────────────────

def plot_rpv_classification_matrix(scores, ms,
                                   sample_idx_map=None,
                                   title=None,
                                   class_label='RPV',
                                   extra_plp_mask=None,
                                   extra_plp_label='P/LP (indet.)',
                                   figsize=None, ax=None,
                                   save_path=None):
    """Heatmap of RPV-style classification breakdown per sample.

    Rows: samples (B/LB, P/LP, and the auxiliary "RPV-like" sample by default)
    Cols: rpv_class categories from score_rpv_penetrance

    Parameters
    ----------
    scores          : pd.DataFrame from analysis.score_rpv_penetrance(...)
    ms              : MultiScoreset (provides _sample_assignments, same N as scores)
    sample_idx_map  : dict {display_name: sa_column_index}, ordered
                      Default: {'B/LB': 1, 'P/LP': 0, 'RPV': 4}
    class_label     : name of the auxiliary sample's class, used in the "low-pen"
                      column label and axis label (e.g. 'BENTA', 'CADINS') — 'RPV'
                      is only the historical TP53-specific name, not universal.
    extra_plp_mask  : bool array (N,) — subset of P/LP variants to show as an
                      additional row, e.g. those not called pathogenic by the main model:
                          plp_mask = ms._sample_assignments[:, 0].astype(bool)
                          pts      = analysis.results[config]['points']
                          extra_plp_mask = plp_mask & (pts <= 0)
    extra_plp_label : row label for the extra P/LP row (default 'P/LP (indet.)')

    Returns
    -------
    fig, ax
    """
    from collections import OrderedDict
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    if sample_idx_map is None:
        sample_idx_map = OrderedDict([('B/LB', 1), ('P/LP', 0), ('RPV', 4)])
    if title is None:
        title = f'{class_label} Classification'

    ordered_classes = ['benign_like', 'intermediate', 'plp_like', 'low_pen_rpv']
    col_labels      = ['Benign-like', 'Intermediate', 'PLP-like', f'Low-pen {class_label}']

    sa = ms._sample_assignments
    rows = {}
    for name, si in sample_idx_map.items():
        if si >= sa.shape[1]:
            continue
        mask   = sa[:, si].astype(bool)
        subset = scores.iloc[np.where(mask)[0]]['rpv_class']
        rows[name] = [int((subset == cls).sum()) for cls in ordered_classes]

    if extra_plp_mask is not None:
        subset = scores.iloc[np.where(extra_plp_mask)[0]]['rpv_class']
        rows[extra_plp_label] = [int((subset == cls).sum()) for cls in ordered_classes]

    df = pd.DataFrame(rows, index=col_labels).T  # (n_samples, n_classes)

    blue_cmap  = LinearSegmentedColormap.from_list('blue',  ['#F0F8FC', '#2E6B7E'])
    gray_cmap  = LinearSegmentedColormap.from_list('gray',  ['#F5F5F5', '#555555'])
    red_cmap   = LinearSegmentedColormap.from_list('red',   ['#FCF0F2', '#943744'])
    green_cmap = LinearSegmentedColormap.from_list('green', ['#F0FBF4', '#2E7D4F'])
    col_cmaps  = [blue_cmap, gray_cmap, red_cmap, green_cmap]

    nrows, ncols = df.shape
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (max(5, ncols * 1.8), max(2, nrows * 1.3)))
    else:
        fig = ax.get_figure()

    colors = np.zeros((nrows, ncols, 4))
    for i in range(nrows):
        row_max = df.iloc[i].max()
        for j in range(ncols):
            val = df.iloc[i, j]
            normalized = val / row_max if row_max > 0 else 0
            colors[i, j] = col_cmaps[j](normalized)

    for i in range(nrows):
        row_max = df.iloc[i].max()
        for j in range(ncols):
            ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1,
                facecolor=colors[i, j], edgecolor='white', linewidth=2.5))
            val = df.iloc[i, j]
            text_color = 'white' if (row_max > 0 and val / row_max > 0.45) else 'black'
            ax.text(j + 0.5, i + 0.5, str(val),
                    ha='center', va='center', fontsize=13, color=text_color)

    ax.set_xlim(0, ncols); ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_yticks(np.arange(nrows) + 0.5)
    ax.set_xticklabels(df.columns, rotation=0, fontsize=11)
    ax.set_yticklabels(df.index, rotation=0, fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(f'{class_label} Classification', fontsize=12)
    ax.set_ylabel('Sample', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_facecolor('#F9F9F9')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


# ── Penetrance score histogram ────────────────────────────────────────────────

def plot_penetrance_score_hist(scores, ms,
                               sample_idx_map=None,
                               class_label='RPV',
                               extra_plp_mask=None,
                               extra_plp_label='P/LP (indet.)',
                               bins=20,
                               title=None,
                               figsize=None,
                               save_path=None):
    """Stacked per-sample histograms of penetrance_score.

    One panel per sample (shared x-axis). Only variants with rpv_points >= 1
    have a penetrance_score; the label shows n_eligible / n_total for each sample.

    Parameters
    ----------
    scores          : pd.DataFrame from analysis.score_rpv_penetrance(...)
    ms              : MultiScoreset
    sample_idx_map  : dict {display_name: sa_column_index}, ordered
                      Default: {'B/LB': 1, 'P/LP': 0, 'RPV': 4}
    class_label     : name of the auxiliary sample's class (e.g. 'BENTA', 'CADINS')
                      used in the x-axis description — 'RPV' is only the historical
                      TP53-specific name, not universal.
    extra_plp_mask  : bool array (N,) for an additional group (e.g. P/LP non-called)
    extra_plp_label : label for that group
    bins            : number of histogram bins (default 20)
    """
    from collections import OrderedDict

    if sample_idx_map is None:
        sample_idx_map = OrderedDict([('B/LB', 1), ('P/LP', 0), ('RPV', 4)])
    if title is None:
        title = f'{class_label} Penetrance Score Distribution'

    _palette = {
        'B/LB':          '#2E6B7E',
        'P/LP':          '#943744',
        'RPV':           '#2E7D4F',
        class_label:     '#2E7D4F',
        extra_plp_label: '#e05c00',
    }
    _default_colors = ['#555555', '#7B2D8B', '#B8860B', '#1A6B3C']

    sa = ms._sample_assignments

    # Build groups: (vals_with_score, n_total)
    groups = {}
    for name, si in sample_idx_map.items():
        if si >= sa.shape[1]:
            continue
        mask  = sa[:, si].astype(bool)
        rows  = scores.iloc[np.where(mask)[0]]
        vals  = rows['penetrance_score'].dropna().values
        groups[name] = (vals, int(mask.sum()))
    if extra_plp_mask is not None:
        rows = scores.iloc[np.where(extra_plp_mask)[0]]
        vals = rows['penetrance_score'].dropna().values
        groups[extra_plp_label] = (vals, int(extra_plp_mask.sum()))

    n_panels = len(groups)
    if n_panels == 0:
        return None, None

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=figsize or (6, 2.2 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    fallback = iter(_default_colors)
    for panel_i, (ax, (name, (vals, n_total))) in enumerate(zip(axes, groups.items())):
        color = _palette.get(name) or next(fallback, '#888888')
        if len(vals):
            ax.hist(vals, bins=bins, range=(0, 1),
                    color=color, alpha=0.75, edgecolor='white', linewidth=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_facecolor('#F9F9F9')
        ax.tick_params(labelsize=8)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        label_x = 0.02 if panel_i == 1 else 0.98
        label_ha = 'left' if panel_i == 1 else 'right'
        ax.text(label_x, 0.95, f'{name}\n{len(vals)}/{n_total}',
                transform=ax.transAxes, ha=label_ha, va='top',
                fontsize=9, color=color, fontweight='bold')

    axes[-1].set_xlabel(
        f'Penetrance score  (0 = low pen. / {class_label}-like,  1 = high pen. / PLP-like)',
        fontsize=10)
    axes[0].set_title(title, fontsize=12, fontweight='bold', pad=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, axes
