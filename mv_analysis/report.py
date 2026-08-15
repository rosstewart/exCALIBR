"""
Combine MV calibration metrics with UV (univariate) baseline metrics into
one comparison table, using the SAME metric definitions
(points_to_confusion + compute_classification_metrics) the saved
"<gene>_<config>_confusion.txt" reports already use (see
src/assay_calibration/multivariate_analysis/report_gene.py), so MV and UV
rows are directly comparable.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.assay_calibration.multivariate_analysis.gene_set_analysis import (
    build_gene_set_analysis, MODE_DISPLAY_NAMES,
)
from src.assay_calibration.multivariate_analysis.mv_calibration import _PARTIAL_PATTERN_MODES
from src.assay_calibration.multivariate_analysis import eval_plot_utils as epu
from src.assay_calibration.plot_utils.utils import compute_classification_metrics

from mv_analysis import uv_sources, uv_agg

# Two ways to turn continuous evidence points into a 3-way call. Both matter
# when combining multiple evidence sources: "evidence_direction" asks only
# which way the combined evidence leans (any nonzero point counts), while
# "clinical" approximates an actual ACMG-style P/LP vs VUS vs B/LB call --
# pathogenic evidence has to clear a much higher bar (+6) than benign (-1),
# matching real classification practice where a single strong benign result
# is usually enough but pathogenicity requires accumulated evidence.
CLASSIFICATION_THRESHOLDS = {
    "evidence_direction (>=1 / <=-1)": (1, -1),
    "clinical (P/LP>=6 / B/LB<=-1)": (6, -1),
}


def points_to_confusion_thresholded(labels, points, path_threshold, ben_threshold):
    """Like eval_plot_utils.points_to_confusion but with configurable
    pathogenic/benign point thresholds instead of the hardcoded >0/<0
    'which direction does the evidence point' split."""
    cats = np.where(points >= path_threshold, 2, np.where(points <= ben_threshold, 0, 1))
    cm = np.zeros((2, 3), dtype=int)
    for row, col in zip(labels.astype(int), cats):
        cm[row, col] += 1
    return pd.DataFrame(cm, index=['B/LB', 'P/LP'], columns=['Benign', 'Indeterminate', 'Pathogenic'])


def _eval_labels(ms, p_idx, b_idx):
    sa = ms.sample_assignments
    n = sa.shape[0]
    plp_mask = sa[:, p_idx].astype(bool) if p_idx is not None else np.zeros(n, dtype=bool)
    blb_mask = sa[:, b_idx].astype(bool) if b_idx is not None else np.zeros(n, dtype=bool)
    eval_mask = plp_mask | blb_mask
    return eval_mask, plp_mask[eval_mask].astype(int)


def _rows_for_points(method, pts_eval, labels, extra=None):
    """One row per entry in CLASSIFICATION_THRESHOLDS for one point array."""
    rows = []
    for threshold_name, (path_t, ben_t) in CLASSIFICATION_THRESHOLDS.items():
        cm = points_to_confusion_thresholded(labels, pts_eval, path_t, ben_t)
        metrics = compute_classification_metrics(cm)
        rows.append({"method": method, "threshold": threshold_name, **(extra or {}), **metrics})
    return rows


def _mv_metric_rows(analysis, modes, mode_labels, **run_kwargs):
    """Two rows (one per CLASSIFICATION_THRESHOLDS entry) per (config, mode)."""
    all_results = analysis.compare_partial_pattern_modes(modes=modes, **run_kwargs)
    eval_mask, labels = _eval_labels(analysis.ms, analysis.p_idx, analysis.b_idx)

    rows = []
    for mode, results in all_results.items():
        label = mode_labels.get(mode, mode)
        for cfg, r in results.items():
            if r is None:
                rows.append({"method": f"MV {cfg}", "mode": label, "config": cfg, "status": "failed"})
                continue
            rows.extend(_rows_for_points(
                f"MV {cfg}", r["points"][eval_mask], labels, extra={"mode": label, "config": cfg}))
    return pd.DataFrame(rows)


def _uv_metric_rows(gene, gene_set, ms, p_idx, b_idx):
    """('UV non-conflicting'/'UV max' rows, uv_dataset_names) or (empty df, None)
    if no UV comparison is available for this gene/gene-set (see
    uv_sources.py's per-gene-set caveats -- FGFR pending; combined only
    verified for TP53-shaped data so far).

    ``p_idx``/``b_idx`` must be the *effective* indices into
    ``ms.sample_assignments`` (i.e. already remapped past any empty/dropped
    fixed-role columns, as ``MVCalibrationAnalysis._eff_idx`` does) -- NOT
    the raw fixed-role indices 0/1. Passing raw 0/1 silently mis-scores
    genes missing an earlier role class (e.g. no P/LP observations shifts
    B/LB into column 0), comparing the wrong sample pair without erroring."""
    uv = uv_sources.load_uv_points(gene, ms, gene_set)
    if uv is None:
        return pd.DataFrame(), None
    names, mat = uv
    eval_mask, labels = _eval_labels(ms, p_idx, b_idx)

    rows = []
    for key, pts in [("UV non-conflicting", uv_agg.aggregate_nonconflicting(mat)),
                      ("UV max", uv_agg.aggregate_max(mat))]:
        pts_eval = np.nan_to_num(pts[eval_mask], nan=0.0)
        rows.extend(_rows_for_points(key, pts_eval, labels, extra={"mode": "", "config": ""}))
    return pd.DataFrame(rows), names


def build_comparison_table(
    gene, gene_set, ms, results_json,
    dataset_name=None, auxiliary_pathogenic_indices=None,
    modes=_PARTIAL_PATTERN_MODES, mode_labels=None,
    compare_uv=True, **run_kwargs,
):
    """(table, uv_dataset_names). ``table`` has one row per MV (config, mode)
    plus, when a UV source exists for this gene/gene-set, 'UV
    non-conflicting' and 'UV max' rows -- all using identical metric
    definitions. ``uv_dataset_names`` is None when no UV comparison was
    possible (printed reason goes to stdout, matching the rest of this
    pipeline's reporting style).
    """
    mode_labels = mode_labels if mode_labels is not None else MODE_DISPLAY_NAMES
    analysis = build_gene_set_analysis(
        ms, gene, results_json, dataset_name=dataset_name,
        auxiliary_pathogenic_indices=auxiliary_pathogenic_indices,
    )
    mv_table = _mv_metric_rows(analysis, modes, mode_labels, **run_kwargs)

    if not compare_uv:
        return mv_table, None

    uv_table, uv_dataset_names = _uv_metric_rows(gene, gene_set, ms, analysis.p_idx, analysis.b_idx)
    if uv_table.empty:
        reason = "pending data" if gene_set == "fgfr" else "no bridge/UV source for this gene-set"
        print(f"  [{gene}/{gene_set}] UV comparison unavailable ({reason})")
        return mv_table, None

    return pd.concat([mv_table, uv_table], ignore_index=True), uv_dataset_names
