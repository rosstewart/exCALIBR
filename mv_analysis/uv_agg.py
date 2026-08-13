"""
Combine multiple per-assay/per-predictor UV point calls for the same variant
into one per-variant call. Pure matrix functions, no path/gene-set logic --
moved here (unchanged) from
src/assay_calibration/multivariate_analysis/compare_uv_mv_agg.py, which
originally hardcoded this to LABEL-seq. Used by every uv_sources.py adapter
that has to aggregate across multiple UV scoresets for one gene (LABEL-seq's
per-assay scoresets, predictor-mv's per-predictor scoresets, TP53/combined's
per-dataset scoresets).

Two ways of combining evidence across datasets (rows) for the same variant
(column):
  - "non-conflicting": max-magnitude evidence when all non-zero values agree
    in sign; 0 if both a positive and a negative value are present.
  - "max": literal elementwise max across datasets, e.g. (-1, 2) -> 2.
"""
import numpy as np


def aggregate_nonconflicting(mat):
    """Combine evidence across datasets (rows) per variant (columns).

    Same-sign (or zero) evidence -> value with largest magnitude.
    Discordant (both a positive and a negative value present) -> 0.
    All-missing column -> NaN.
    """
    D, N = mat.shape
    out = np.full(N, np.nan)
    for j in range(N):
        vals = mat[:, j]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        pos = vals[vals > 0]
        neg = vals[vals < 0]
        if len(pos) and len(neg):
            out[j] = 0.0
        elif len(pos):
            out[j] = pos.max()
        elif len(neg):
            out[j] = neg.min()
        else:
            out[j] = 0.0
    return out


def aggregate_max(mat):
    """Elementwise max across datasets per variant, e.g. (-1, 2) -> 2.

    If all values for a variant are <= 0, take the one with the largest
    absolute value instead, e.g. (-1, 0) -> -1, (-3, -1) -> -3.

    All-missing column -> NaN.
    """
    D, N = mat.shape
    out = np.full(N, np.nan)
    for j in range(N):
        vals = mat[:, j]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            if np.all(vals <= 0):
                out[j] = vals[np.argmax(np.abs(vals))]
            else:
                out[j] = vals.max()
    return out
