"""
Categorical Sankey plotting + the strength-color palette, vendored (copied,
not imported) from `tavtigian_sims/comparisons/plot_evidence_sankeys.py` and
`tavtigian_sims/compare.py`.

`tavtigian_sims/` is untracked/local-only and must not be a prerequisite for
running anything in `mv_analysis/` -- this module exists so the
VUS-reclassification Sankeys and any other categorical-flow figure can be
generated from a clean checkout with no dependency on that directory being
present. Keep this file's two pieces (palette + plotter) in sync with the
originals only if the originals are intentionally improved; do not re-import
from `tavtigian_sims` here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import pandas as pd

# ---------------------------------------------------------------------------
# Palette (vendored from tavtigian_sims/compare.py)
# ---------------------------------------------------------------------------

STRENGTH_COLOR = {
    -12: '#2f5a70', -11: '#3a6b81', -10: '#4b91a6', -9: '#516d8a',
    -8: '#4b91a6', -7: '#5DA3BD', -6: '#6FAACE', -5: '#74ABCE',
    -4: '#7ab5d1', -3: '#99c8dc', -2: '#d0e8f0', -1: '#e4f1f6',
     0: '#e0e0e0',
     1: '#e6b1b8',  2: '#d68f99',  3: '#ca7682',  4: '#b85c6b',
     5: '#B1535F',  6: '#AA4E58',  7: '#A2484F',  8: '#943744',
     9: '#7d2e38', 10: '#6b2830', 11: '#59221f', 12: '#472015',
}

# Boundary colours (P, LP, LB, B) keyed to strength-colour anchors.
BND_COLOR = {
    "P":  STRENGTH_COLOR[8],
    "LP": STRENGTH_COLOR[5],
    "LB": STRENGTH_COLOR[-5],
    "B":  STRENGTH_COLOR[-8],
}


# ---------------------------------------------------------------------------
# Sankey plotting (vendored from tavtigian_sims/comparisons/plot_evidence_sankeys.py)
# ---------------------------------------------------------------------------

def plot_categorical_sankey(
    flow_df: pd.DataFrame, source_order: List[str], target_order: List[str],
    source_colors: Dict[str, str], target_colors: Optional[Dict[str, str]] = None,
    ax=None, node_width: float = 0.06, gap_frac: float = 0.015,
    source_fontsize: float = 9, target_fontsize: float = 9,
    show_target_counts: bool = True, target_label_fmt=str,
):
    """flow_df: columns 'source', 'target', 'count'. Two columns of
    proportional stacked node bars (source at x=0, target at x=1) joined by
    filled cubic-Bezier ribbon polygons, colored by source node -- the same
    visual idiom as Figure5_6.Rmd's geom_sankey.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    mat = flow_df.pivot_table(index="source", columns="target", values="count",
                               aggfunc="sum", fill_value=0)
    mat = mat.reindex(index=source_order, columns=target_order, fill_value=0)

    source_totals = mat.sum(axis=1)
    target_totals = mat.sum(axis=0)
    total = source_totals.sum()
    gap = gap_frac * total if total > 0 else 0.0

    def _node_positions(totals, order):
        positions = {}
        y = 0.0
        for cat in order:
            h = float(totals[cat])
            positions[cat] = (y, y + h)
            y += h + gap
        return positions, max(y - gap, 0.0)

    src_pos, src_height = _node_positions(source_totals, source_order)
    tgt_pos, tgt_height = _node_positions(target_totals, target_order)
    max_height = max(src_height, tgt_height)

    x0, x1 = 0.0, 1.0
    tcolors = target_colors or source_colors

    for cat in source_order:
        y0, y1 = src_pos[cat]
        if y1 <= y0:
            continue
        ax.add_patch(mpatches.Rectangle(
            (x0 - node_width / 2, y0), node_width, y1 - y0,
            facecolor=source_colors.get(cat, "#999999"), edgecolor="black",
            linewidth=0.8, zorder=3))
        ax.text(x0 - node_width / 2 - 0.02, (y0 + y1) / 2,
                 f"{cat} ({int(source_totals[cat]):,})", ha="right", va="center",
                 fontsize=source_fontsize)

    for cat in target_order:
        y0, y1 = tgt_pos[cat]
        if y1 <= y0:
            continue
        ax.add_patch(mpatches.Rectangle(
            (x1 - node_width / 2, y0), node_width, y1 - y0,
            facecolor=tcolors.get(cat, "#999999"), edgecolor="black",
            linewidth=0.8, zorder=3))
        cat_label = target_label_fmt(cat)
        label = f"{cat_label} ({int(target_totals[cat]):,})" if show_target_counts else cat_label
        ax.text(x1 + node_width / 2 + 0.02, (y0 + y1) / 2,
                 label, ha="left", va="center", fontsize=target_fontsize)

    src_cursor = {cat: src_pos[cat][0] for cat in source_order}
    tgt_cursor = {cat: tgt_pos[cat][0] for cat in target_order}
    xs0 = x0 + node_width / 2
    xs1 = x1 - node_width / 2
    xm = (xs0 + xs1) / 2

    for src in source_order:
        for tgt in target_order:
            v = float(mat.loc[src, tgt])
            if v <= 0:
                continue
            sy0 = src_cursor[src]
            sy1 = sy0 + v
            src_cursor[src] = sy1
            ty0 = tgt_cursor[tgt]
            ty1 = ty0 + v
            tgt_cursor[tgt] = ty1

            verts = [
                (xs0, sy1), (xm, sy1), (xm, ty1), (xs1, ty1),
                (xs1, ty0), (xm, ty0), (xm, sy0), (xs0, sy0),
                (xs0, sy1),
            ]
            codes = [
                MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.CLOSEPOLY,
            ]
            patch = mpatches.PathPatch(
                MplPath(verts, codes), facecolor=source_colors.get(src, "#999999"),
                edgecolor="none", alpha=0.55, zorder=1)
            ax.add_patch(patch)

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-gap, max_height + gap)
    ax.invert_yaxis()
    ax.axis("off")
    return fig, ax
