"""
Per-dataset diagnostic figures, generalized across every ExCALIBR comparison
method (author, acmgscaler, GMM baselines, skew-locked, pathomechanism).

Section 3b's LaTeX tables (analysis/manuscript_stats.py) show only pooled
metrics -- every dataset summed into one confusion matrix per method. That's
the right format for the manuscript, but it hides (a) whether a method's
accuracy/determinate-%/DOR is consistent across datasets or dominated by a
couple of large ones, and (b) which datasets actually drive the pooled
FP/FN counts. These two figures show that, from the exact same per-dataset
confusion-matrix lists the tables use -- no new data loading.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Ellipse

from src.assay_calibration.plot_utils.utils import compute_classification_metrics
from analysis.manuscript_stats import compute_aggregate_metrics_multi

OTHER_COLOR = "#9E9E9E"
_TAB10 = plt.get_cmap("tab10").colors
_PANEL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Same display-name convention as section 3b's per-table method_labels dicts
# (analyze_pipeline_output.py) -- kept in sync manually since the two live in
# different call sites; falls back to title-cased key for anything not
# listed here.
_METHOD_DISPLAY = {
    "excalibr": "ExCALIBR",
    "author": "Author",
    "acmgscaler": "acmgscaler",
    "excalibr_prior_0.1": "ExCALIBR (prior=0.1)",
    "skew_locked": "Skew-locked ExCALIBR",
    "pathomechanism": "Pathomechanism-aware",
    "gmm_plp_blb": "GMM (P/LP+B/LB)",
    "gmm_plp_blb_synon": "GMM (P/LP+B/LB∪Syn)",
    "gmm_all_plp_blb": "GMM (all)",
    "gmm_all_plp_blb_synon": "GMM (all P/LP+B/LB∪Syn)",
}


def _display_name(method: str) -> str:
    return _METHOD_DISPLAY.get(method, method.replace("_", " ").title())


# Fixed order, not per-panel position -- so a given method (e.g. "author")
# always gets the same color everywhere it appears, instead of whatever
# tab10 slot its position happens to land on within one panel's own method
# list (which previously let two different methods in two different panels
# collide on the same color just because they were both "index 1").
_CANONICAL_METHOD_ORDER = [
    "excalibr", "author", "acmgscaler", "excalibr_prior_0.1",
    "skew_locked", "pathomechanism",
    "gmm_plp_blb", "gmm_plp_blb_synon", "gmm_all_plp_blb", "gmm_all_plp_blb_synon",
]
_GLOBAL_METHOD_COLOR = {m: _TAB10[i % len(_TAB10)] for i, m in enumerate(_CANONICAL_METHOD_ORDER)}


def _method_colors(methods: List[str]) -> Dict[str, tuple]:
    colors = {}
    next_idx = len(_CANONICAL_METHOD_ORDER)
    for m in methods:
        if m in _GLOBAL_METHOD_COLOR:
            colors[m] = _GLOBAL_METHOD_COLOR[m]
        elif m not in colors:
            colors[m] = _TAB10[next_idx % len(_TAB10)]
            next_idx += 1
    return colors


def _draw_axis_break(ax, y_break: float):
    """Dashed line + short diagonal marks at both edges, at data-y
    `y_break` -- the standard "this axis isn't drawn to scale above this
    point" cue, used only on the DOR row when at least one method has an
    infinite value (FP or FN == 0 for some dataset) that's shown as a
    separate "infinity" marker above the break rather than silently
    dropped."""
    ax.axhline(y_break, color="#bbbbbb", linestyle=(0, (4, 2)), linewidth=0.8, zorder=1)
    y_frac = ax.transAxes.inverted().transform(ax.transData.transform((0, y_break)))[1]
    d = 0.012
    kwargs = dict(transform=ax.transAxes, color="#bbbbbb", clip_on=False, linewidth=0.8, zorder=1)
    for x0 in (0.0, 1.0):
        ax.plot([x0 - d, x0 + d], [y_frac - d, y_frac + d], **kwargs)


def _matched_indices(matrices_by_method: Dict[str, list], n: int) -> List[int]:
    methods = list(matrices_by_method.keys())
    return [i for i in range(n) if all(matrices_by_method[m][i] is not None for m in methods)]


def _own_calibrated_counts(matrices_by_method: Dict[str, list]) -> Dict[str, int]:
    return {m: sum(1 for mat in mats if mat is not None) for m, mats in matrices_by_method.items()}


def compute_per_dataset_metrics_multi(matrices_by_method: Dict[str, list], dataset_names: list) -> pd.DataFrame:
    """N-method generalization of src.assay_calibration.plot_utils.utils.compute_aggregate_metrics's
    `individual_df` (hardcoded to exactly "danz"/"auth" and 5 rate metrics).

    Restricted to the same matched/intersection semantics as
    analysis.manuscript_stats.compute_aggregate_metrics_multi -- every method
    is compared on identically the same datasets, which is what makes a
    cross-method box plot or stacked-bar comparison meaningful.

    One row per matched dataset, columns "dataset" + "{method}_{metric}" for
    metric in accuracy, coverage (determinate fraction), dor (dor_standard,
    possibly inf when FP or FN is 0 for that dataset), FP, FN.
    """
    methods = list(matrices_by_method.keys())
    matched_idx = _matched_indices(matrices_by_method, len(dataset_names))

    rows = []
    for i in matched_idx:
        row = {"dataset": dataset_names[i]}
        for m in methods:
            met = compute_classification_metrics(matrices_by_method[m][i])
            row[f"{m}_accuracy"] = met["accuracy"]
            row[f"{m}_coverage"] = met["coverage"]
            # compute_classification_metrics returns dor_standard == 0.0 as a
            # sentinel when a dataset has zero determinate calls at all
            # (every P/LP and B/LB variant landed in the indeterminate
            # zone) -- not a genuine "very low odds ratio" measurement.
            # NaN instead, so it's excluded from the violin the same way
            # `np.isfinite` already excludes genuine +inf (FP==0 or
            # FN==0), rather than getting floored and plotted as an
            # extreme-but-real low point.
            row[f"{m}_dor"] = np.nan if met["determinate"] == 0 else met["dor_standard"]
            row[f"{m}_FP"] = met["FP"]
            row[f"{m}_FN"] = met["FN"]
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt_metric(metric: str, value: float) -> str:
    return f"{value:.2f}" if metric != "dor" else f"{value:.1f}"


def _style_violin(vp, methods: List[str], colors: Dict[str, tuple]):
    for body, m in zip(vp["bodies"], methods):
        body.set_facecolor(colors[m])
        body.set_alpha(0.6)
        body.set_edgecolor("#444444")
        body.set_linewidth(0.8)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in vp:
            vp[key].set_color("#444444")
            vp[key].set_linewidth(1)


def plot_comparison_boxplots(
    panel_groups: Dict[str, Dict[str, list]], dataset_names: list, n_total_datasets: int,
) -> plt.Figure:
    """3 rows (accuracy, DOR (log scale), determinate %) x 1 continuous
    axes each -- every comparison group's methods placed along the same
    x-axis (light vertical divider lines mark group boundaries), rather
    than one separate axes per group. Since accuracy/determinate % always
    share the same fixed range and DOR's range is computed across every
    group's data together, a per-group axes (with its own y-axis, ticks,
    and whitespace margins) added nothing but repetition -- one y-axis per
    row, labeled once on the left, is exactly as informative. One uppercase
    panel letter per (group, row) cell, e.g. (A)-(I) for 3 groups x 3 rows,
    ordered group-major/metric-minor (accuracy/DOR/determinate% for group 0,
    then group 1, ...) to match the manuscript caption's per-group letter
    grouping. Placed above each group's own cluster within its row's shared
    axes (get_xaxis_transform, since groups don't have an axes each).

    Violin rather than box: shows the actual per-dataset distribution
    shape, not just quartiles, which also gives a natural way to show
    DOR's infinite-value pileup as a separate concentrated blob rather than
    an arrow -- see below. Method colors are fixed globally (see
    _method_colors) so e.g. "Author" is always the same color everywhere.

    Each violin gets a "median: X / aggregate: Y" label above it: the
    per-dataset median (same array the violin itself is drawn from) and
    the pooled/aggregate value from
    analysis.manuscript_stats.compute_aggregate_metrics_multi (the same
    number section 3b's LaTeX tables report). Each method's own
    "(own_count/n_total)" calibratable count is a second tick-label line
    (rotation_mode="anchor" -- fixes matplotlib's usual overlapping-line
    bug for multi-line rotated tick labels).

    DOR is frequently infinite for a single dataset (dor_standard is inf
    whenever that dataset's FP or FN count is 0), or a NaN sentinel from
    compute_per_dataset_metrics_multi when a dataset had zero determinate
    calls at all (compute_classification_metrics itself returns a 0.0
    placeholder there, not a genuine low DOR). Both are kept, not dropped:
    the violin itself is still drawn from the finite values only (a KDE has
    no way to represent a non-finite value), but a method with any
    non-finite values gets a small filled blob above an axis break instead,
    sized (relative to the whole row) by how many datasets are non-finite,
    labeled "X inf, Y undefined (of matched)" above it (no arrow/connector).
    """
    groups = []
    for group_title, matrices_by_method in panel_groups.items():
        methods = list(matrices_by_method.keys())
        df = compute_per_dataset_metrics_multi(matrices_by_method, dataset_names)
        agg_metrics, own_counts, n_matched = compute_aggregate_metrics_multi(matrices_by_method, dataset_names)
        groups.append({
            "title": group_title, "methods": methods, "df": df,
            "agg_metrics": agg_metrics, "own_counts": own_counts, "n_matched": n_matched,
        })

    # Global x positions: one slot per method, with an extra gap (not just
    # the usual 1-unit spacing) between groups, marked by a divider line.
    GROUP_GAP = 1.6
    positions_by_group = []
    boundaries = []
    x = 1.0
    for gi, g in enumerate(groups):
        pos = [x + i for i in range(len(g["methods"]))]
        positions_by_group.append(pos)
        x = pos[-1] + 1 + GROUP_GAP
        if gi < len(groups) - 1:
            boundaries.append(pos[-1] + GROUP_GAP / 2 + 0.5)
    total_x_max = x - GROUP_GAP

    all_methods_flat = [(gi, m) for gi, g in enumerate(groups) for m in g["methods"]]
    positions_flat = [p for grp in positions_by_group for p in grp]
    colors = _method_colors([m for _, m in all_methods_flat])

    fig_width = max(9.0, 1.05 * len(all_methods_flat) + 1.3 * max(0, len(groups) - 1))
    fig = plt.figure(figsize=(fig_width, 9.5))
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

    row_specs = [
        ("accuracy", "Accuracy", lambda v: v),
        ("dor", "DOR (pathogenic vs. benign)", lambda v: v),
        ("coverage", "Determinate %", lambda v: 100 * v),
    ]

    for row, (metric, ylabel, transform) in enumerate(row_specs):
        ax = fig.add_subplot(outer[row, 0])

        raw_flat = [
            transform(groups[gi]["df"][f"{m}_{metric}"].to_numpy(dtype=float))
            for gi, m in all_methods_flat
        ]

        if metric == "dor":
            finite_flat = [v[np.isfinite(v)] for v in raw_flat]
            n_true_inf = [int(np.sum(np.isinf(v))) for v in raw_flat]  # FP==0 or FN==0
            n_undetermined = [int(np.sum(np.isnan(v))) for v in raw_flat]  # 0 determinate calls at all
            n_inf = [a + b for a, b in zip(n_true_inf, n_undetermined)]
            # dor_standard can be exactly 0 (TP==0 or TN==0, with FP/FN both
            # > 0 -- distinct from the "undefined" NaN case) -- log10(0) is
            # -inf, which broke set_ylim below. Floor to a small positive
            # value so a zero-DOR dataset renders as an extreme low point
            # instead of crashing the axis.
            log_finite = [np.log10(np.maximum(v, 1e-3)) for v in finite_flat]
            plottable = [(i, v) for i, v in enumerate(log_finite) if len(v) > 1]
        else:
            finite_flat = raw_flat
            n_inf = [0] * len(all_methods_flat)
            plottable = [(i, v) for i, v in enumerate(raw_flat) if len(v) > 1]

        if plottable:
            idxs, data = zip(*plottable)
            vp = ax.violinplot(list(data), positions=[positions_flat[i] for i in idxs],
                                widths=0.72, showmedians=True, showextrema=True)
            _style_violin(vp, [all_methods_flat[i][1] for i in idxs], colors)

        ax.set_xlim(0.3, total_x_max + 0.7)

        if metric != "dor":
            ax.set_ylim(0, 118 if metric == "coverage" else 1.18)
            y_lo, y_hi = ax.get_ylim()
            label_offset = 0.045 * (y_hi - y_lo)
        else:
            # Violin drawn in log10 space above but the axis itself stays
            # linear (in log10 units) with hand-picked tick labels showing
            # the true (10^x) value -- matplotlib's violinplot has no
            # log-scale-aware KDE, so log-transforming the data first and
            # faking the tick labels avoids a KDE computed in linear space
            # and then visually squashed by a log axis transform after.
            all_finite_log = np.concatenate([v for v in log_finite if len(v)]) if any(len(v) for v in log_finite) else np.array([0.0])
            y_lo = all_finite_log.min() - 0.3
            y_hi_data = all_finite_log.max() + 0.3
            if any(n_inf):
                break_y = y_hi_data + 1.8
                inf_y = y_hi_data + 3.3
                ax.set_ylim(y_lo, inf_y + 2.3)
                _draw_axis_break(ax, break_y)
            else:
                ax.set_ylim(y_lo, y_hi_data)
            ticks = [t for t in range(int(np.floor(y_lo)), int(np.ceil(y_hi_data)) + 1)]
            ticklabels = [f"$10^{{{t}}}$" for t in ticks]
            if any(n_inf):
                ticks = ticks + [inf_y]
                ticklabels = ticklabels + [""]
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticklabels)
            label_offset = 0.3

        for b in boundaries:
            ax.axvline(b, color="#dddddd", linewidth=1.2, zorder=0)

        for i, (gi, m) in enumerate(all_methods_flat):
            values = finite_flat[i]
            if len(values) == 0:
                continue
            g = groups[gi]
            med = np.median(values)
            agg_val = g["agg_metrics"][m]["accuracy"] if metric == "accuracy" else (
                g["agg_metrics"][m]["dor_standard"] if metric == "dor" else 100 * g["agg_metrics"][m]["coverage"])
            top_y = np.log10(np.max(values)) if metric == "dor" else np.max(values)
            x_med = positions_flat[i]
            text = f"median: {_fmt_metric(metric, med)}\nagg.: {_fmt_metric(metric, agg_val)}"
            ax.text(x_med, top_y + label_offset, text, fontsize=6, ha="center", va="bottom",
                    color="#222222", linespacing=1.3)

            # DOR-only: a small filled blob at the "inf" level for any
            # method with non-finite values, sized (relative to the whole
            # row, across every group) by how many datasets are non-finite
            # -- the "concentration" cue standing in for the arrow this
            # used to have -- plus a plain count label above it.
            if metric == "dor" and n_inf[i]:
                max_inf = max(n_inf) or 1
                blob_r = 0.10 + 0.16 * (n_inf[i] / max_inf)
                ax.add_patch(Ellipse(
                    (x_med, inf_y), width=blob_r * 2, height=blob_r * 2 * 1.8,
                    facecolor=colors[m], alpha=0.7, edgecolor="#444444", linewidth=0.6, clip_on=False,
                ))
                parts = []
                if n_true_inf[i]:
                    parts.append(f"{n_true_inf[i]} inf")
                if n_undetermined[i]:
                    parts.append(f"{n_undetermined[i]} undef.")
                text = ", ".join(parts)
                ax.text(x_med, inf_y + blob_r * 1.8 + 0.3, text,
                        fontsize=5.5, ha="center", va="bottom", color="#222222")

        ax.set_xticks(positions_flat)
        ax.set_xticklabels(
            [f"{_display_name(m)}\n({groups[gi]['own_counts'][m]}/{n_total_datasets})" for gi, m in all_methods_flat],
            rotation=20, ha="right", rotation_mode="anchor", fontsize=7.5,
        )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

        # One uppercase letter per (group, row) cell -- labeled above that
        # group's own cluster of violins, in data-x/axes-y coordinates
        # (get_xaxis_transform) since all of a row's groups share one
        # continuous axes rather than having an axes each. Ordered
        # group-major, metric-minor (accuracy/DOR/determinate% for group 0,
        # then group 1, ...) to match the manuscript caption's per-group
        # letter grouping, not the row-major order the loop itself runs in.
        for gi in range(len(groups)):
            letter_idx = gi * len(row_specs) + row
            center = float(np.mean(positions_by_group[gi]))
            ax.text(center, 1.05, f"({_PANEL_LETTERS[letter_idx]})",
                    transform=ax.get_xaxis_transform(),
                    fontsize=10, fontweight="bold", va="bottom", ha="center")

    return fig


def _rank_and_bucket(values: Dict[str, float], top_n: int):
    """values: {dataset: count}. Returns (top, other_sum) -- top is up to
    top_n (dataset, count) pairs sorted descending (zero counts excluded),
    other_sum is the sum of everything past top_n."""
    items = sorted(((k, v) for k, v in values.items() if v > 0), key=lambda kv: kv[1], reverse=True)
    top = items[:top_n]
    other_sum = sum(v for _, v in items[top_n:])
    return top, other_sum


def _plot_fp_fn_bars(ax, df: pd.DataFrame, methods: List[str], category: str, top_n: int, show_ylabel: bool):
    """One category's (FP or FN) bars, one per method, side by side.
    Each bar is stacked "Other" first (always at the bottom) then the top
    `top_n` contributing datasets for that method (own ranking, so which
    datasets are "top" can differ method to method) -- matches how a reader
    scans bottom-to-top: the bulk ("Other") first, then the named
    contributors on top of it. Y-axis is left to autoscale to whatever
    methods are actually shown in this subplot (no shared/reference scale).
    Legend lives inside the axes (not a separate row) -- up to
    `top_n * len(methods)` entries, likely fewer once datasets repeat
    across methods.
    """
    col = f"_{category}"
    method_top: Dict[str, tuple] = {}
    legend_datasets: List[str] = []
    for m in methods:
        vals = dict(zip(df["dataset"], df[f"{m}{col}"]))
        top, other = _rank_and_bucket(vals, top_n)
        method_top[m] = (top, other)
        for ds, _ in top:
            if ds not in legend_datasets:
                legend_datasets.append(ds)
    ds_color = {ds: _TAB10[i % len(_TAB10)] for i, ds in enumerate(legend_datasets)}

    x = np.arange(len(methods))
    other_heights = np.array([method_top[m][1] for m in methods], dtype=float)
    ax.bar(x, other_heights, color=OTHER_COLOR, edgecolor="white", linewidth=0.5, width=0.6)
    bottoms = other_heights.copy()
    for rank in range(top_n):
        heights = np.zeros(len(methods))
        colors_this = ["none"] * len(methods)
        for mi, m in enumerate(methods):
            top, _ = method_top[m]
            if rank < len(top):
                ds, val = top[rank]
                heights[mi] = val
                colors_this[mi] = ds_color[ds]
        ax.bar(x, heights, bottom=bottoms, color=colors_this, edgecolor="white", linewidth=0.5, width=0.6)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([_display_name(m) for m in methods], rotation=20, ha="right", fontsize=8)
    ax.set_title(category, fontsize=10, fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Count", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

    handles = [Patch(facecolor=OTHER_COLOR, label="Other")]
    handles += [Patch(facecolor=ds_color[ds], label=ds) for ds in legend_datasets]
    ax.legend(handles=handles, fontsize=6, loc="upper right", framealpha=0.85)


def plot_comparison_stacked_bars(
    panel_groups: Dict[str, Dict[str, list]], dataset_names: list, top_n: int = 3,
) -> plt.Figure:
    """One row per comparison group: a FP subplot and an FN subplot side by
    side (divided, sharing the row), each with one bar per method in that
    group (see _plot_fp_fn_bars), plus a third "aggregate metrics" cell
    (accuracy | DOR | determinate %, pooled over the group's matched
    datasets) so the per-dataset FP/FN breakdown can be read against the
    same pooled numbers section 3b's tables use.
    """
    rows = []
    for group_title, matrices_by_method in panel_groups.items():
        methods = list(matrices_by_method.keys())
        df = compute_per_dataset_metrics_multi(matrices_by_method, dataset_names)
        agg_metrics, _, n_matched = compute_aggregate_metrics_multi(matrices_by_method, dataset_names)
        rows.append((group_title, methods, df, agg_metrics, n_matched))

    # Row height scales with how many methods that row's aggregate-metrics
    # column has to fit rotated x-tick labels for -- a 2-method row (Author)
    # and a 6-method row (merged GMM/skew-locked/pathomechanism) can't share
    # a fixed height without either wasting space or overlapping labels.
    row_heights = [1.3 + 0.22 * max(0, len(methods) - 2) for _, methods, _, _, _ in rows]
    fig = plt.figure(figsize=(13, sum(4.6 * h for h in row_heights)))
    outer = gridspec.GridSpec(
        len(rows), 2, figure=fig, width_ratios=[2.3, 1], height_ratios=row_heights, hspace=1.3, wspace=0.4,
    )

    for row_idx, (group_title, methods, df, agg_metrics, n_matched) in enumerate(rows):
        fp_fn_spec = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row_idx, 0], wspace=0.35)
        ax_fp = fig.add_subplot(fp_fn_spec[0])
        ax_fn = fig.add_subplot(fp_fn_spec[1])
        _plot_fp_fn_bars(ax_fp, df, methods, "FP", top_n, show_ylabel=True)
        _plot_fp_fn_bars(ax_fn, df, methods, "FN", top_n, show_ylabel=False)

        ax_fp.text(
            0.0, 1.32, f"{group_title} (n={n_matched})", transform=ax_fp.transAxes,
            fontsize=12, fontweight="bold", ha="left", va="bottom",
        )

        _plot_aggregate_metrics_panel(outer[row_idx, 1], fig, methods, agg_metrics)

    return fig


def _plot_aggregate_metrics_panel(subplot_spec, fig, methods: List[str], agg_metrics: Dict[str, dict]):
    """3 small bar charts (accuracy | DOR | determinate %), one bar per
    method each -- separate axes per metric since their scales don't share
    an axis (accuracy/determinate% in [0,1]-ish, DOR often in the hundreds).
    Reads analysis.manuscript_stats.compute_aggregate_metrics_multi's own
    agg_metrics_by_method dict directly, so this can never disagree with
    section 3b's LaTeX tables built from the same dict.
    """
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=subplot_spec, hspace=3.4)
    colors = _method_colors(methods)
    specs = [
        ("accuracy", "Accuracy", lambda m: agg_metrics[m]["accuracy"]),
        ("dor", "DOR\n(pathogenic vs. benign)", lambda m: agg_metrics[m]["dor_standard"]),
        ("determinate_pct", "Determinate %", lambda m: 100 * agg_metrics[m]["coverage"]),
    ]
    x = np.arange(len(methods))
    for i, (_, label, getter) in enumerate(specs):
        ax = fig.add_subplot(inner[i])
        values = [getter(m) for m in methods]
        ax.bar(x, values, color=[colors[m] for m in methods], alpha=0.85, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([_display_name(m) for m in methods], rotation=35, ha="right", fontsize=6)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
