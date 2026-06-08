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

from src.assay_calibration.plot_utils.utils import (
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


def _accumulate_into(target, key, labels, pts, snv_pts, n_snv, n_ctrl):
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
              compute_assay_labels_zscore, gene_to_analysis_cons_b0,
              non_nu_genes=None):
    """
    Aggregate confusion matrix counts across genes.

    Parameters
    ----------
    non_nu_genes : set or None
        If provided, restrict to these genes only (non-NU subset).

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

        labels    = is_plp[eval_mask]
        assay_pts = compute_assay_labels_zscore(ms)
        n_snv     = int(is_snv.sum())
        n_ctrl    = int(eval_mask.sum())

        _accumulate_into(agg, "Assay",
                         labels, assay_pts[eval_mask],
                         assay_pts[is_snv], n_snv, n_ctrl)

        for pctile, pdf in gdf.groupby("path_percentile"):
            r = gene_to_analysis_cons_b0 if pctile == 25 else \
                percentile_analyses.get(pctile, {})
            analysis = r.get(gene)
            if analysis is None:
                continue
            result = analysis.results.get("3c_unc")
            if result is None:
                continue
            pts = result["points"]
            key = f"MV p{pctile} (3c_unc)"
            _accumulate_into(agg, key,
                             labels, pts[eval_mask],
                             pts[is_snv], n_snv, n_ctrl)

    return agg


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


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrices(agg,
                            mv_key='MV p25 (3c_unc)', assay_key='Assay',
                            mv_title='ExCALIBR-MV',
                            assay_title='Synonymous z-score',
                            figsize=(10, 4), save_path=None):
    """
    Plot aggregate confusion matrices for ExCALIBR-MV vs. assay z-score.
    Uses pre-aggregated counts so per-gene zero-row-sum genes don't cause skips.
    Overrides the hardcoded titles and removes suptitle post-hoc.
    """
    fig, _, _ = plot_aggregate_confusion_matrices(
        danzs=[_agg_to_cm(agg, mv_key)],
        auths=[_agg_to_cm(agg, assay_key)],
        dataset_names=['aggregate'],
        figsize=figsize,
        letters=True,
    )
    # Override hardcoded titles and strip suptitle
    fig.suptitle('')
    for ax in fig.axes:
        title = ax.get_title()
        if 'ExCALIBR Evidence' in title:
            ax.set_title(mv_title, fontsize=18, fontweight='bold', pad=10)
        elif 'Author Annotations' in title:
            ax.set_title(assay_title, fontsize=18, fontweight='bold', pad=10)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig
