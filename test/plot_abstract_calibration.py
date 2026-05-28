"""
Per-dataset and multi-page calibration plots for *predictor* score calibrations
(AlphaMissense, REVEL, MutPred2, …).

Layout per dataset
------------------
    Row 1 : per-sample ExCALIBR fits (one ax per non-empty sample, with overlay)
    Row 2 : combined experimental score histogram
            ── P/LP, B/LB, and population variants (sample idx 2)
    Row 3 : "Gene-specific calibration"
            ── duplicated bar:
                3a  ExCALIBR point assignments
                3b  Yile point assignments
            counts on each bar reflect *population variants* only
    Row 4 : color legend (evidence strengths)

Both `plot_calibration_predictor` (single full-size figure) and
`plot_multi_calibration_predictor` (multi-page appendix) delegate to the
same `_draw_dataset_panel` so the two are guaranteed to stay in sync.
"""

import os
import math
import logging
import pickle
import gzip
import json

# Silence matplotlib & joblib chatter
os.environ["MPLLOGLEVEL"] = "WARNING"
for name in ("", "matplotlib", "matplotlib.font_manager",
             "matplotlib.pyplot", "joblib", "loky"):
    logging.getLogger(name).setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Nimbus Sans', 'DejaVu Sans']

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from joblib import Parallel, delayed

import sys
sys.path.append('..')
from src.assay_calibration.plot_utils.utils import sample_density


# ─────────────────────────────────────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_COLORS = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']  # P/LP, B/LB, pop, syn
SAMPLE_ALPHAS = [0.5, 0.5, 0.30, 0.4]

STRENGTH_COLOR = {
    -8: '#4b91a6', -7: '#5DA3BD', -6: '#6FAACE', -5: '#74ABCE',
    -4: '#7ab5d1', -3: '#99c8dc', -2: '#d0e8f0', -1: '#e4f1f6',
     0: '#e0e0e0',
     1: '#e6b1b8',  2: '#d68f99',  3: '#ca7682',  4: '#b85c6b',
     5: '#B1535F',  6: '#AA4E58',  7: '#A2484F',  8: '#943744',
}

POINT_LABEL = {i: f"{i:+d}" if i != 0 else "0" for i in range(-8, 9)}

POINT_VALUES_TO_PLOT = [1, 2, 3, 4, 5, 6, 7, 8]
LINESTYLES = [
    'dotted', 'dashed', 'dashdot', (5, (10, 3)),
    (0, (3, 5, 1, 5)), (0, (5, 5)), (0, (3, 1, 1, 1)), 'solid',
]

# Categorical (continuous-method) evidence labels and their display properties.
# Ordered strong-to-weak so the legend reads P → LP | LB → B.
CATEGORICAL_KEYS = frozenset({'P', 'LP', 'LB', 'B'})
CATEGORICAL_ORDER = {'P': 8, 'LP': 4, 'LB': -4, 'B': -8}   # numeric proxy for sorting
CATEGORICAL_COLOR = {
    'P':  STRENGTH_COLOR[8],
    'LP': STRENGTH_COLOR[4],
    'LB': STRENGTH_COLOR[-4],
    'B':  STRENGTH_COLOR[-8],
    0:    STRENGTH_COLOR[0],
}
CATEGORICAL_LABEL = {'P': 'P', 'LP': 'LP', 'LB': 'LB', 'B': 'B', 0: 'VUS'}

SAMPLE_NAME_SHORTENER_FULL = {
    "Pathogenic/Likely Pathogenic": "ClinVar P/LP",
    "Benign/Likely Benign":         "ClinVar B/LB",
    "population":                   "gnomAD",
    "gnomAD":                       "gnomAD",
    "Synonymous":                   "Synonymous",
}
SAMPLE_NAME_SHORTENER_COMPACT = {
    "Pathogenic/Likely Pathogenic": "P/LP",
    "Benign/Likely Benign":         "B/LB",
    "population":                   "gnomAD",
    "gnomAD":                       "gnomAD",
    "Synonymous":                   "Syn",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — calibration intervals & threshold lines
# ─────────────────────────────────────────────────────────────────────────────

def _clip_finite(value, lo, hi):
    """Clip ±inf and out-of-range values to [lo, hi]."""
    if value == -math.inf or value < lo:
        return lo
    if value ==  math.inf or value > hi:
        return hi
    return value


def _is_categorical(point_ranges) -> bool:
    """Return True if point_ranges uses categorical keys (P, LP, LB, B)."""
    return bool(point_ranges and any(k in CATEGORICAL_KEYS for k in point_ranges))


def _normalize_point_ranges(point_ranges):
    """
    Accept either flat ({pv: [start, end]}) or nested ({pv: [[start, end]]})
    representations and return a clean {pv: [start, end] | []} dict.

    Keys may be:
      - integers / digit-strings  → converted to int  (numeric mode)
      - 'P', 'LP', 'LB', 'B'     → kept as str       (categorical mode)
    """
    if point_ranges is None:
        return None
    out = {}
    for k, v in point_ranges.items():
        # Determine key type
        if str(k) in CATEGORICAL_KEYS:
            pv = str(k)
        else:
            try:
                pv = int(k)
            except (TypeError, ValueError):
                continue
        if v is None or len(v) == 0:
            out[pv] = []
            continue
        # Nested form: [[start, end], ...]
        if isinstance(v[0], (list, tuple)):
            out[pv] = list(v[0]) if v else []
        else:
            out[pv] = list(v)
    return out


def build_calibration_intervals(point_ranges, x_min, x_max, flipped):
    """
    Build a list of (point_value, start, end) intervals for a calibration bar.

    Any region of [x_min, x_max] not covered by an explicit (non-zero) point
    range is filled with indeterminate (point=0) intervals.

    `point_ranges` is the FLAT format {pv: [start, end]} or {pv: []}.
    Keys may be int (numeric mode) or str (categorical mode: P/LP/LB/B).
    Infinities and out-of-range bounds are clipped to [x_min, x_max].
    """
    is_cat = _is_categorical(point_ranges)

    def _sort_key(pv):
        return CATEGORICAL_ORDER.get(pv, 0) if is_cat else pv

    nonzero = []
    if point_ranges:
        for pv in sorted((p for p in point_ranges if p != 0), key=_sort_key):
            rng = point_ranges[pv]
            if not rng or len(rng) != 2:
                continue
            start = _clip_finite(rng[0], x_min, x_max)
            end   = _clip_finite(rng[1], x_min, x_max)
            if end <= start:
                continue
            nonzero.append((pv, start, end))

    # Snap tiny gaps between adjacent non-zero intervals (e.g. Yile thresholds
    # quoted to 3 decimals leave ~0.001 slivers between consecutive ranges).
    nonzero.sort(key=lambda x: x[1])
    snap_tol = (x_max - x_min) * 0.01
    snapped = []
    for entry in nonzero:
        if snapped:
            prev_pv, prev_s, prev_e = snapped[-1]
            gap = entry[1] - prev_e
            if 0 < gap < snap_tol:
                mid = (prev_e + entry[1]) / 2
                snapped[-1] = (prev_pv, prev_s, mid)
                entry = (entry[0], mid, entry[2])
        snapped.append(entry)

    # Fill every remaining uncovered slice of [x_min, x_max] with indeterminate.
    intervals = list(snapped)
    cursor = x_min
    for _pv, s, e in snapped:
        if s > cursor:
            intervals.append((0, cursor, s))
        cursor = max(cursor, e)
    if cursor < x_max:
        intervals.append((0, cursor, x_max))

    return sorted(intervals, key=lambda x: x[1])


def _build_thresholds(point_ranges, flipped,
                      point_values=POINT_VALUES_TO_PLOT, linestyles=LINESTYLES):
    """Return [(pv, score, color, ls, lw), …] for axvline overlays on fits panels."""
    if not point_ranges:
        return []
    if _is_categorical(point_ranges):
        return _build_categorical_thresholds(point_ranges, flipped)
    out = []
    for idx, pv in enumerate(point_values):
        ls = linestyles[idx]
        # Benign side: upper bound of -pv when not flipped (where LB/B evidence starts).
        rng = point_ranges.get(-pv)
        if rng and len(rng) == 2:
            tscore = rng[1] if not flipped else rng[0]
            if math.isfinite(tscore):
                out.append((-pv, tscore, '#2166AC', ls, 1.0))
        # Pathogenic side: lower bound of +pv when not flipped (where LP/P evidence starts).
        rng = point_ranges.get(pv)
        if rng and len(rng) == 2:
            tscore = rng[0] if not flipped else rng[1]
            if math.isfinite(tscore):
                out.append((pv, tscore, '#B2182B', ls, 1.0))
    return out


def _build_categorical_thresholds(point_ranges, flipped):
    """Threshold lines for categorical (continuous-method) evidence."""
    out = []
    # Pathogenic side: inner boundary = upper bound of LP / lower bound of P
    for pv, ls in [('LP', 'dashed'), ('P', 'dotted')]:
        rng = point_ranges.get(pv)
        if rng and len(rng) == 2:
            tscore = rng[1] if not flipped else rng[0]
            if math.isfinite(tscore):
                out.append((pv, tscore, '#B2182B', ls, 1.0))
    # Benign side: inner boundary = lower bound of LB / lower bound of B
    for pv, ls in [('LB', 'dashed'), ('B', 'dotted')]:
        rng = point_ranges.get(pv)
        if rng and len(rng) == 2:
            tscore = rng[0] if not flipped else rng[1]
            if math.isfinite(tscore):
                out.append((pv, tscore, '#2166AC', ls, 1.0))
    return out


def _plot_calibration_bar(ax, intervals, scores_for_count, x_min, x_max,
                          *, fontsize_count, min_label_frac=0.06, color_map=None):
    """Render one evidence-strength colored bar with per-bin counts.

    When no intervals are provided, the entire bar is treated as indeterminate.
    `color_map` defaults to STRENGTH_COLOR; pass CATEGORICAL_COLOR for
    continuous-method evidence.
    """
    if color_map is None:
        color_map = STRENGTH_COLOR
    if not intervals:
        intervals = [(0, x_min, x_max)]
    for pv, start, end in intervals:
        ax.axvspan(start, end, color=color_map.get(pv, STRENGTH_COLOR[0]), alpha=1.0)
        count = int(((scores_for_count >= start) & (scores_for_count < end)).sum())
        if (end - start) > (x_max - x_min) * min_label_frac:
            if pv in CATEGORICAL_ORDER:
                pv = CATEGORICAL_ORDER[pv]
            text_color = 'white' if abs(pv) >= 7 else 'black'
            ax.text((start + end) / 2, 0.5, f'{count:,}',
                    ha='center', va='center',
                    fontsize=fontsize_count, color=text_color)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])


def _get_sample_scores(scoreset, sample_num):
    """
    Return scores belonging to sample `sample_num` (e.g. 2 = population),
    accounting for empty samples that drop columns from sample_assignments.
    """
    num_skipped = 0
    for sn in range(len(scoreset.sample_counts)):
        if scoreset.sample_counts[sn] == 0:
            num_skipped += 1
            continue
        if sn == sample_num:
            col = sn - num_skipped
            mask = scoreset.sample_assignments[:, col]
            return scoreset.scores[mask]
    return np.array([])


# ─────────────────────────────────────────────────────────────────────────────
# Core modular panel — drawn into pre-allocated axes
# ─────────────────────────────────────────────────────────────────────────────

def _draw_dataset_panel(
    *,
    axes_fits,
    ax_combined,
    ax_excalibr,
    ax_yile,
    ax_legend,
    scoreset,                 # BasicScoreset for combined hist + counts
    scoreset_for_fit,         # BasicScoreset used during calibration fitting
    fits,                     # bootstrap fits list
    score_range,              # 1D numpy array
    point_ranges_excalibr,    # flat {int_pv: [start, end]}
    point_ranges_yile,        # flat dict, or None if unavailable
    flipped=False,
    prior=None,
    title=None,
    style="full",             # "full" or "compact"
    show_xlabel=True,
):
    """
    Draw the full per-dataset panel into the supplied axes.

    `axes_fits` must have length == number of non-empty samples in
    scoreset_for_fit.  `ax_combined`, `ax_excalibr`, `ax_yile`, `ax_legend`
    are single Axes objects.
    """
    is_full = (style == "full")

    if is_full:
        fontsize_subtitle = 18
        fontsize_legend   = 13
        fontsize_count    = 14
        fontsize_tick     = 10
        bin_div           = 50
        sample_short      = SAMPLE_NAME_SHORTENER_FULL
        fits_legend_fs    = 13
        combined_legend_fs = 13
        side_label_fs     = 12
        side_label_pad    = 10
        threshold_lw      = 1.0
        fit_lw            = 2.0
        title_fs          = fontsize_subtitle
    else:
        fontsize_subtitle = 9
        fontsize_legend   = 6
        fontsize_count    = 7
        fontsize_tick     = 6
        bin_div           = 30
        sample_short      = SAMPLE_NAME_SHORTENER_COMPACT
        fits_legend_fs    = 6
        combined_legend_fs = 6
        side_label_fs     = 7
        side_label_pad    = 4
        threshold_lw      = 0.8
        fit_lw            = 1.5
        title_fs          = 9

    x_min = score_range[0]
    x_max = score_range[-1]
    bin_width = (x_max - x_min) / bin_div

    population_scores = _get_sample_scores(scoreset, sample_num=2)
    is_cat = _is_categorical(point_ranges_excalibr)
    excalibr_color_map = CATEGORICAL_COLOR if is_cat else STRENGTH_COLOR
    excalibr_label_map = CATEGORICAL_LABEL if is_cat else POINT_LABEL
    excalibr_thresholds = _build_thresholds(point_ranges_excalibr, flipped)

    # ── Row 1 : per-sample ExCALIBR fits ────────────────────────────────────
    fit_axes_iter = iter(axes_fits)
    num_skipped = 0
    first_fit_ax = True
    for sample_num in range(len(scoreset_for_fit.sample_counts)):
        if scoreset_for_fit.sample_counts[sample_num] == 0:
            num_skipped += 1
            continue
        sample_idx = sample_num - num_skipped
        try:
            ax = next(fit_axes_iter)
        except StopIteration:
            break

        sample_mask = scoreset_for_fit.sample_assignments[:, sample_idx]
        sample_name = sample_short.get(
            scoreset_for_fit.sample_names[sample_num],
            scoreset_for_fit.sample_names[sample_num],
        )
        color = SAMPLE_COLORS[sample_num]
        hist_data = scoreset_for_fit.scores[sample_mask]
        n_count = int(sample_mask.sum())

        try:
            sns.histplot(hist_data, binwidth=bin_width, stat='density',
                         ax=ax, alpha=0.5, color=color)
        except ValueError:
            sns.histplot(hist_data, stat='density',
                         ax=ax, alpha=0.5, color=color)

        max_h = max((p.get_height() for p in ax.patches), default=1.0)

        if sample_idx < len(fits[0]['fit']['weights']):
            # Fitted total density (bootstrap median + 5/95 band)
            density_sample = sample_density(score_range, fits, sample_idx)
            d_total = np.nansum(density_sample, axis=1)
            d_perc  = np.percentile(d_total, [5, 50, 95], axis=0)
            ax.fill_between(score_range, d_perc[0], d_perc[2], color='gray', alpha=0.3)
            ax.plot(score_range, d_perc[1], color='black', alpha=0.65, linewidth=fit_lw)

        ax.set_xlim(x_min, x_max)
        if not is_full:
            ax.set_ylim([0, max_h * 1.1])

        ax.set_xlabel('')
        if is_full:
            ax.set_ylabel('Density' if first_fit_ax else '', fontsize=fontsize_subtitle)
            ax.tick_params(axis='both', labelsize=9)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='both', bottom=False, left=False,
                           labelbottom=False, labelleft=False)

        # Threshold overlays (ExCALIBR only)
        for pv, tscore, tcol, tls, _ in excalibr_thresholds:
            show = True if is_cat else abs(pv) in (1, 2, 4, 8)
            if show:
                ax.axvline(tscore, color=tcol, linestyle=tls,
                           linewidth=threshold_lw, alpha=0.7)

        # Per-sample mini-legend
        face_rgba = to_rgba(color, 0.5)
        hist_patch = Patch(facecolor=face_rgba, edgecolor='black')
        if sample_num == 2 and prior is not None:
            label = f'{sample_name}\n(n={n_count:,d})\nprior: {prior:.3f}'
        else:
            label = f'{sample_name}\n(n={n_count:,d})'
        if flipped:
            loc = 'upper left'  if sample_num == 0 else 'upper right'
        else:
            loc = 'upper right' if sample_num == 0 else 'upper left'
        ax.legend([hist_patch], [label], loc=loc,
                  fontsize=fits_legend_fs, framealpha=0.9)

        # Title goes on the first fits ax
        if first_fit_ax:
            if is_full:
                ax.set_title('ExCALIBR sample fits', loc='left',
                             pad=3, fontsize=title_fs)
            elif title is not None:
                ax.set_title(title, fontsize=title_fs, fontweight='bold',
                             pad=3, loc='left')
        first_fit_ax = False

    # ── Row 2 : combined experimental histogram ────────────────────────────
    sample_handles = []
    num_skipped = 0
    for sample_num in range(len(scoreset.sample_counts)):
        if scoreset.sample_counts[sample_num] == 0:
            num_skipped += 1
            continue
        if sample_num == 3:                    # skip Synonymous
            continue
        sample_idx  = sample_num - num_skipped
        sample_mask = scoreset.sample_assignments[:, sample_idx]
        sample_name = sample_short.get(
            scoreset.sample_names[sample_num],
            scoreset.sample_names[sample_num],
        )
        color   = SAMPLE_COLORS[sample_num]
        alpha   = SAMPLE_ALPHAS[sample_num]
        hist_d  = scoreset.scores[sample_mask]
        n_count = int(sample_mask.sum())

        try:
            sns.histplot(hist_d, binwidth=bin_width, stat='density',
                         ax=ax_combined, alpha=alpha, color=color)
        except ValueError:
            sns.histplot(hist_d, stat='density',
                         ax=ax_combined, alpha=alpha, color=color)

        sample_handles.append((
            Patch(facecolor=to_rgba(color, alpha), edgecolor='black'),
            f'{sample_name} (n={n_count:,d})',
        ))

    ax_combined.set_xlim(x_min, x_max)
    ax_combined.set_xlabel('')
    if is_full:
        ax_combined.set_ylabel('Density', fontsize=fontsize_subtitle)
        ax_combined.tick_params(axis='both', labelsize=fontsize_tick)
        ax_combined.set_title(f'{title if title is not None else "Experimental"} score distributions', loc='left',
                              pad=3, fontsize=title_fs)
    else:
        ax_combined.set_ylabel('')
        ax_combined.tick_params(axis='both', bottom=False, left=False,
                                labelsize=fontsize_tick,
                                labelbottom=False, labelleft=False)

    handles = [h for h, _ in sample_handles]
    labels  = [l for _, l in sample_handles]
    ax_combined.legend(handles, labels, loc='upper left',
                       fontsize=combined_legend_fs, framealpha=0.8)

    # ── Row 3a : ExCALIBR calibration bar ──────────────────────────────────
    intervals_exc  = build_calibration_intervals(
        point_ranges_excalibr, x_min, x_max, flipped)
    _plot_calibration_bar(
        ax_excalibr, intervals_exc, population_scores, x_min, x_max,
        fontsize_count=fontsize_count, color_map=excalibr_color_map,
    )
    # if is_full:
    #     ax_excalibr.set_title('Gene-specific calibration', loc='left',
    #                           pad=3, fontsize=title_fs)
    ax_excalibr.set_ylabel('ExCALIBR', fontsize=side_label_fs, rotation=0,
                           ha='right', va='center', labelpad=side_label_pad)
    ax_excalibr.tick_params(axis='x', bottom=False, labelbottom=False)

    # ── Row 3b : Yile calibration bar ──────────────────────────────────────
    intervals_yile = build_calibration_intervals(
        point_ranges_yile, x_min, x_max, flipped) \
        if point_ranges_yile is not None else []

    if ax_yile is not None:
        _plot_calibration_bar(
            ax_yile, intervals_yile, population_scores, x_min, x_max,
            fontsize_count=fontsize_count,
            color_map=CATEGORICAL_COLOR if _is_categorical(point_ranges_yile) else STRENGTH_COLOR,
        )
    
        ax_yile.set_ylabel('Yile', fontsize=side_label_fs, rotation=0,
                           ha='right', va='center', labelpad=side_label_pad)
    
        if show_xlabel:
            ax_yile.set_xlabel('Predictor Score',
                               fontsize=fontsize_subtitle if is_full else 8)
            ax_yile.tick_params(axis='x', labelsize=fontsize_tick)
        else:
            ax_yile.set_xlabel('')
            ax_yile.tick_params(axis='x', bottom=False, labelbottom=False)

    # ── Row 4 : color legend (union of point values used by either bar) ─────
    all_pvs = {pv for pv, _, _ in intervals_exc} | {pv for pv, _, _ in intervals_yile}
    if is_cat:
        # Sort by categorical strength: P > LP > 0 > LB > B (reverse for legend readability)
        legend_order = sorted(
            all_pvs,
            key=lambda p: CATEGORICAL_ORDER.get(p, 0),
            reverse=not flipped,
        )
        color_map_leg = excalibr_color_map
        label_map_leg = excalibr_label_map
    else:
        pvs_present = sorted(all_pvs)
        legend_order = sorted(pvs_present) if flipped else sorted(pvs_present, reverse=True)
        color_map_leg = STRENGTH_COLOR
        label_map_leg = POINT_LABEL

    legend_handles = [
        Patch(facecolor=color_map_leg.get(pv, STRENGTH_COLOR[0]),
              label=label_map_leg.get(pv, str(pv)), edgecolor='none')
        for pv in legend_order
    ]

    ax_legend.axis('off')
    if not legend_handles:
        return

    if is_full:
        ax_legend.legend(
            handles=legend_handles, loc='upper center',
            ncol=len(legend_handles), frameon=False,
            fontsize=fontsize_legend,
            columnspacing=0.8, handletextpad=0.6, handlelength=1.0,
        )
    else:
        ncol = (len(legend_handles)
                if len(legend_handles) <= 12
                else len(legend_handles) // 2 + len(legend_handles) % 2)
        ax_legend.legend(
            handles=legend_handles, loc='center',
            ncol=ncol, frameon=False, fontsize=fontsize_legend,
            columnspacing=0.4, handletextpad=0.3, handlelength=0.7,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point — single full-size figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_predictor(
    *,
    dataset,
    scoreset,
    fits,
    score_range,
    point_ranges_excalibr,
    point_ranges_yile,
    flipped=False,
    prior=None,
    scoreset_for_fit=None,
):
    """
    Single full-size figure for one predictor calibration.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. "AM_BRCA1") used as figure title.
    scoreset : BasicScoreset
        Scoreset used for the combined histogram and population counts.
    fits : list
        Bootstrap fit results, one entry per bootstrap replicate.
    score_range : np.ndarray
        Dense 1-D array of scores at which the fits are evaluated.
    point_ranges_excalibr : dict
        ExCALIBR point-range dict.  Either flat ({pv: [s, e]}) or nested
        ({pv: [[s, e]]}) format is accepted; str or int keys.
    point_ranges_yile : dict or None
        Yile thresholds for the same gene, flat format.  Pass None if
        unavailable for this gene.
    flipped : bool
        True if higher scores are benign rather than pathogenic.
    prior : float, optional
        Pathogenic prior to display in the gnomAD legend.
    scoreset_for_fit : BasicScoreset, optional
        Scoreset used during fitting (defaults to `scoreset`).
    """
    point_ranges_excalibr = _normalize_point_ranges(point_ranges_excalibr)
    point_ranges_yile     = _normalize_point_ranges(point_ranges_yile)
    if scoreset_for_fit is None:
        scoreset_for_fit = scoreset

    n_actual = sum(1 for c in scoreset_for_fit.sample_counts if c > 0)

    fig = plt.figure(figsize=(12, 12))
    if point_ranges_yile is not None:
        gs = gridspec.GridSpec(
            5, n_actual,
            height_ratios=[2, 2, 0.6, 0.6, 0.3],
            hspace=0.35, wspace=0.15,
        )
    else:
        gs = gridspec.GridSpec(
            4, n_actual,
            height_ratios=[2, 2, 0.6, 0.3],
            hspace=0.35, wspace=0.15,
        )

    axes_fits   = [plt.subplot(gs[0, i]) for i in range(n_actual)]
    ax_combined = plt.subplot(gs[1, :])
    ax_excalibr = plt.subplot(gs[2, :])
    if point_ranges_yile is not None:
        ax_yile     = plt.subplot(gs[3, :])
        ax_legend   = plt.subplot(gs[4, :])
    else:
        ax_yile = None
        ax_legend = plt.subplot(gs[3, :])

    _draw_dataset_panel(
        axes_fits=axes_fits,
        ax_combined=ax_combined,
        ax_excalibr=ax_excalibr,
        ax_yile=ax_yile,
        ax_legend=ax_legend,
        scoreset=scoreset,
        scoreset_for_fit=scoreset_for_fit,
        fits=fits,
        score_range=score_range,
        point_ranges_excalibr=point_ranges_excalibr,
        point_ranges_yile=point_ranges_yile,
        flipped=flipped,
        prior=prior,
        title=dataset,
        style="full",
        show_xlabel=True,
    )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-page entry point
# ─────────────────────────────────────────────────────────────────────────────

def _plot_single_page_predictor(page_idx, page_panels, nrows, ncols):
    """Render one page of the multi-dataset appendix."""
    # Re-silence loggers in worker processes
    for name in ("", "matplotlib", "matplotlib.font_manager",
                 "matplotlib.pyplot", "joblib", "loky"):
        logging.getLogger(name).setLevel(logging.ERROR)

    fig = plt.figure(figsize=(8.5, 11))
    outer_gs = gridspec.GridSpec(
        nrows, ncols,
        hspace=0.28, wspace=0.12,
        left=0.06, right=0.97, bottom=0.04, top=0.98,
    )

    for plot_idx, panel in enumerate(page_panels):
        row = plot_idx // ncols
        col = plot_idx % ncols

        scoreset_for_fit = panel.get('scoreset_for_fit') or panel['scoreset']
        n_actual = sum(1 for c in scoreset_for_fit.sample_counts if c > 0)

        # 6 rows: fits, hist, exc, yile, spacer, legend
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            6, n_actual,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1.2, 1.2, 0.32, 0.32, 0.10, 0.18],
            hspace=0.25, wspace=0.1,
        )

        axes_fits   = [fig.add_subplot(inner_gs[0, i]) for i in range(n_actual)]
        ax_combined = fig.add_subplot(inner_gs[1, :])
        ax_excalibr = fig.add_subplot(inner_gs[2, :])
        ax_yile     = fig.add_subplot(inner_gs[3, :])
        ax_legend   = fig.add_subplot(inner_gs[5, :])

        _draw_dataset_panel(
            axes_fits=axes_fits,
            ax_combined=ax_combined,
            ax_excalibr=ax_excalibr,
            ax_yile=ax_yile,
            ax_legend=ax_legend,
            scoreset=panel['scoreset'],
            scoreset_for_fit=scoreset_for_fit,
            fits=panel['fits'],
            score_range=panel['score_range'],
            point_ranges_excalibr=_normalize_point_ranges(panel['point_ranges_excalibr']),
            point_ranges_yile=_normalize_point_ranges(panel.get('point_ranges_yile')),
            flipped=panel.get('flipped', False),
            prior=panel.get('prior'),
            title=panel.get('title', panel['dataset']),
            style="compact",
            show_xlabel=(row == nrows - 1),
        )

    return page_idx, fig


def plot_multi_calibration_predictor(
    panels,
    datasets_per_page=4,
    ncols=2,
    debug_first_page_only=False,
    n_jobs=-1,
):
    """
    Multi-page appendix figure across many predictor calibrations.

    Parameters
    ----------
    panels : list of dict
        Each dict needs at minimum:
            dataset                : str
            scoreset               : BasicScoreset (combined hist + counts)
            fits                   : list of bootstrap fits
            score_range            : np.ndarray
            point_ranges_excalibr  : dict
            point_ranges_yile      : dict or None
        Optional:
            scoreset_for_fit (defaults to scoreset),
            flipped (default False),
            prior, title.
    datasets_per_page : int
        Number of dataset panels per page (default 4 → 2×2 grid).
    ncols : int
        Number of columns in the grid (default 2).
    debug_first_page_only : bool
        Render only the first page sequentially (useful for iterating).
    n_jobs : int
        Joblib worker count (-1 = all cores).

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    nrows = datasets_per_page // ncols
    n_pages = (len(panels) + datasets_per_page - 1) // datasets_per_page

    if debug_first_page_only:
        n_pages = 1
        n_jobs  = 1
        print("DEBUG MODE: only plotting first page")

    print(f"Plotting {len(panels)} datasets across {n_pages} pages")

    page_inputs = []
    for page_idx in range(n_pages):
        start = page_idx * datasets_per_page
        end   = min(start + datasets_per_page, len(panels))
        page_inputs.append((page_idx, panels[start:end]))

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_plot_single_page_predictor)(p_idx, p_data, nrows, ncols)
        for p_idx, p_data in page_inputs
    )

    return [fig for _, fig in sorted(results, key=lambda x: x[0])]


def plot_calibration_abstract(
    *,
    dataset,
    scoreset,
    fits,
    calibration_results,
    point_ranges_yile=None,
    score_range_n=2000,
):
    """
    Convenience wrapper around plot_calibration_predictor that derives
    all required arguments from a calibration_results dict and the
    bootstrap fits store.

    Parameters
    ----------
    dataset : str
        Dataset key (e.g. "gof_benta"); must be present in `fits`.
    scoreset : BasicScoreset
        Scoreset used for histograms and population counts.
    fits : dict
        Full bootstrap fits store: fits[dataset][bootstrap_seed][n_c].
    calibration_results : dict
        Calibration JSON for this dataset, containing at minimum:
          'n_c', 'point_ranges', 'scoreset_flipped', 'prior'.
    point_ranges_yile : dict or None
        Optional Yile thresholds; passed through unchanged.
    score_range_n : int
        Number of points in the dense evaluation grid (default 500).
    """
    # ── Resolve n_c ────────────────────────────────────────────────────────
    n_c = calibration_results["n_c"]

    # ── Collect one fit entry per bootstrap seed ────────────────────────────
    # fits[dataset][seed][n_c] — gather every seed that has this n_c
    dataset_fits = fits.get(dataset, {})
    fit_list = []
    for seed, seed_fits in dataset_fits.items():
        if n_c in seed_fits:
            fit_list.append(seed_fits[n_c])
    if not fit_list:
        raise ValueError(
            f"No fits found for dataset={dataset!r}, n_c={n_c!r}. "
            f"Available n_c keys in first seed: "
            f"{list(next(iter(dataset_fits.values()), {}).keys())}"
        )

    # ── Derive score range from all non-empty samples ──────────────────────
    # Use the full score array (sample_assignments filters handled inside)
    all_scores = scoreset.scores
    s_min = float(np.nanmin(all_scores))
    s_max = float(np.nanmax(all_scores))
    score_range = np.linspace(s_min, s_max, score_range_n)

    # ── Unpack calibration results ──────────────────────────────────────────
    point_ranges_excalibr = calibration_results["point_ranges"]
    flipped               = bool(calibration_results.get("scoreset_flipped", False))
    prior                 = calibration_results.get("prior", None)

    # ── Delegate to the full plotting function ──────────────────────────────
    return plot_calibration_predictor(
        dataset=dataset,
        scoreset=scoreset,
        fits=fit_list,
        score_range=score_range,
        point_ranges_excalibr=point_ranges_excalibr,
        point_ranges_yile=point_ranges_yile,
        flipped=flipped,
        prior=prior,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a panel dict from the user's reference data layout
# ─────────────────────────────────────────────────────────────────────────────

def build_panel_from_calibration_file(
    calibration_path,
    df,
    results_all,
    yile_results,
    *,
    n_c="3c",
    score_range_n=2000,
    BasicScoreset_cls=None,
):
    """
    Build a `panel` dict (suitable for plot_calibration_predictor or
    plot_multi_calibration_predictor) from one calibration JSON file plus the
    shared results dict and Yile-thresholds dict shown in the reference code.

    Parameters
    ----------
    calibration_path : str
        Path to a `*_3c_calibration.json` file.
    df : pandas.DataFrame
        Concatenated predictor-scores dataframe (with `Dataset`, `score`,
        `sample_assignments` columns).
    results_all : dict
        The deserialized `sg_predictor_results_*.json.gz` payload.
    yile_results : dict
        Mapping {dataset: {pv_str: [start, end] or []}} (from
        load_yile_thresholds()).
    n_c : str
        Component string used to index `results_all[dataset][bootstrap_key]`
        (default "3c").
    score_range_n : int
        Number of points in the dense score range.
    BasicScoreset_cls : class, optional
        Pass `BasicScoreset` to avoid an import here.  If None, will be
        imported from src.assay_calibration.data_utils.dataset.

    Returns
    -------
    panel dict, or None if the file is unusable.
    """
    if BasicScoreset_cls is None:
        from src.assay_calibration.data_utils.dataset import BasicScoreset as BasicScoreset_cls

    with open(calibration_path, "r") as f:
        calibration_data = json.load(f)

    if calibration_data.get("point_ranges") is None:
        return None

    dataset = calibration_data["dataset"]
    df_d = df[df.Dataset == dataset]
    if len(df_d) == 0:
        return None

    scoreset = BasicScoreset_cls(df_d["score"], df_d["sample_assignments"])

    if dataset not in results_all:
        return None
    fits = [results_all[dataset][k][n_c] for k in results_all[dataset]]

    score_range = np.linspace(
        *np.percentile(scoreset.scores, [0, 100]), score_range_n
    )

    return {
        "dataset":               dataset,
        "scoreset":              scoreset,
        "scoreset_for_fit":      scoreset,
        "fits":                  fits,
        "score_range":           score_range,
        "point_ranges_excalibr": calibration_data["point_ranges"],
        "point_ranges_yile":     yile_results.get(dataset),
        "flipped":               calibration_data.get("flipped", False),
        "prior":                 calibration_data.get("prior"),
        "title":                 dataset,
    }