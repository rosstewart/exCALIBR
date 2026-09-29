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
    """One row per entry in CLASSIFICATION_THRESHOLDS for one point array.

    Also attaches n_pathogenic/n_benign (raw label counts in the evaluated
    set, threshold-independent -- unlike TP/FN/TN/FP, which split by
    CLASSIFICATION_THRESHOLDS's cutoffs) and auc (ROC AUC of the continuous
    points against the true labels, also threshold-independent, hence the
    same value repeated across both threshold rows). Both let callers
    detect the single-class-evaluated case (e.g. a gene with zero B/LB
    variants in this comparison) where MCC/AUC are mathematically undefined
    -- compute_classification_metrics silently returns 0.0 for MCC in that
    case rather than NaN, which looks like a real (bad) result unless a
    caller checks these counts directly."""
    from sklearn.metrics import roc_auc_score

    n_pathogenic = int(np.sum(labels == 1))
    n_benign = int(np.sum(labels == 0))
    if n_pathogenic > 0 and n_benign > 0:
        auc = float(roc_auc_score(labels, pts_eval))
    else:
        auc = np.nan

    rows = []
    for threshold_name, (path_t, ben_t) in CLASSIFICATION_THRESHOLDS.items():
        cm = points_to_confusion_thresholded(labels, pts_eval, path_t, ben_t)
        metrics = compute_classification_metrics(cm)
        rows.append({"method": method, "threshold": threshold_name, **(extra or {}), **metrics,
                      "n_pathogenic": n_pathogenic, "n_benign": n_benign, "auc": auc})
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
    # Translate the mv_analysis-registry gene_set name to the UV dispatcher's
    # own key namespace -- they predate each other and don't match exactly
    # ("predictor" vs "predictor-mv"). Confirmed via a real run: "predictor"
    # gene-set's report-table UV rows were unconditionally empty for every
    # gene despite the same UV data bridging fine through
    # individual_predictor_comparison, which already used this mapping.
    #
    # "combined" is handled separately (NOT via _UV_DISPATCH_GENE_SET,
    # unlike "predictor"): that shared dict is also used by
    # individual_predictor_comparison, where "combined" deliberately means
    # functional-only BY DEFAULT (only its own opt-in all_evidence=True
    # switches to "combined-all-evidence") -- folding "combined" into the
    # shared mapping would silently break that default for every OTHER
    # caller of this module too. Here (the results-table's aggregate UV
    # baseline), the merged functional+predictor "combined-all-evidence"
    # baseline is always what's wanted -- confirmed via a real run: bare
    # "combined" dispatches successfully but to the FUNCTIONAL-ONLY loader,
    # which returned None for BRCA1/BRCA2 (undocumented as unverified
    # outside TP53-shaped data) instead of the working merged baseline.
    gene_set = _UV_DISPATCH_GENE_SET.get(gene_set, gene_set)
    if gene_set == "combined":
        gene_set = "combined-all-evidence"

    if gene_set == "combined-all-evidence":
        # Functional and predictor sources aggregated SEPARATELY (each via
        # their own non-conflicting/max rule) and then ADDED -- not one
        # non-conflicting/max rule spanning every individual source as a
        # single flat pool (what load_uv_points+aggregate_*(mat) would do
        # here). See uv_sources.aggregate_functional_plus_predictor.
        eval_mask, labels = _eval_labels(ms, p_idx, b_idx)
        rows = []
        for key, aggregator in [("UV non-conflicting", uv_agg.aggregate_nonconflicting),
                                 ("UV max", uv_agg.aggregate_max)]:
            pts = uv_sources.aggregate_functional_plus_predictor(gene, ms, aggregator)
            if pts is None:
                continue
            pts_eval = np.nan_to_num(pts[eval_mask], nan=0.0)
            rows.extend(_rows_for_points(key, pts_eval, labels, extra={"mode": "", "config": ""}))
        if not rows:
            return pd.DataFrame(), None
        names, _ = uv_sources.load_combined_all_evidence_uv_points(gene, ms) or (None, None)
        return pd.DataFrame(rows), names

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


# mv_analysis registry gene_set names (config.build_multiscoresets_for_gene_set)
# -> uv_sources.load_uv_points's dispatcher keys, which predate and don't
# exactly match the registry's naming ("predictor" vs "predictor-mv") --
# translated here rather than renaming either side, since both names are
# already used elsewhere (cockpit CLI --gene-set choices vs. this
# dispatcher's existing call sites).
_UV_DISPATCH_GENE_SET = {"predictor": "predictor-mv"}


def individual_predictor_comparison(gene, ms, gene_set, p_idx, b_idx, all_evidence=False):
    """One row per INDIVIDUAL evidence source (not aggregated) -- e.g.
    REVEL_TP53/AlphaMissense_TP53/MutPred2_TP53 for gene_set="predictor",
    or those plus each functional dataset for gene_set="combined" with
    all_evidence=True -- alongside the same MCC/accuracy/etc. metrics
    every other row in this module uses. `uv_sources.load_uv_points`
    already builds this per-source matrix internally before
    `uv_agg.aggregate_*` collapses it to one row; this just stops before
    that collapse and scores each source separately instead.

    `all_evidence`: for gene_set="combined", use the dispatcher's
    "combined-all-evidence" key (functional + predictor sources together,
    matching Panel C's baseline) instead of "combined" (functional only).

    Returns an empty DataFrame if no per-source UV data exists for this
    gene/gene_set (same convention as the rest of this module)."""
    dispatch_gene_set = _UV_DISPATCH_GENE_SET.get(gene_set, gene_set)
    if all_evidence and gene_set == "combined":
        dispatch_gene_set = "combined-all-evidence"
    uv = uv_sources.load_uv_points(gene, ms, dispatch_gene_set)
    if uv is None:
        return pd.DataFrame()
    names, mat = uv
    eval_mask, labels = _eval_labels(ms, p_idx, b_idx)

    rows = []
    for i, name in enumerate(names):
        pts_eval = np.nan_to_num(mat[i][eval_mask], nan=0.0)
        rows.extend(_rows_for_points(name, pts_eval, labels, extra={"mode": "", "config": ""}))
    return pd.DataFrame(rows)


def pool_gene_set_metrics(gene_set, per_gene_labels_points, config_name=None, method="MV"):
    """One pooled row per CLASSIFICATION_THRESHOLDS entry, computed by
    CONCATENATING every gene's (labels, points) arrays before building the
    confusion matrix -- equivalent to summing each gene's confusion matrix
    first, matching the manuscript's stated convention ("metrics are pooled
    across genes by summing counts rather than by averaging") rather than
    averaging each gene's own MCC/accuracy/etc.

    ``per_gene_labels_points``: dict {gene: (labels, points)} -- labels/points
    already restricted to that gene's evaluable (P/LP or B/LB) rows, exactly
    what `_eval_labels` + a results dict's `points[eval_mask]` produce (see
    `_mv_metric_rows`/`individual_predictor_comparison` for the per-gene
    shape this expects).
    """
    if not per_gene_labels_points:
        return pd.DataFrame()
    all_labels = np.concatenate([lp[0] for lp in per_gene_labels_points.values()])
    all_points = np.concatenate([lp[1] for lp in per_gene_labels_points.values()])
    extra = {"mode": "", "config": config_name or "", "gene_set": gene_set,
             "n_genes": len(per_gene_labels_points)}
    return pd.DataFrame(_rows_for_points(method, all_points, all_labels, extra=extra))


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
        gene_set=gene_set,
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
