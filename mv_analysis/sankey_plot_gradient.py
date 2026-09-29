"""
Matplotlib-based Sankey diagram matching the reference figure style:
5 ordered categories (Pathogenic / Likely pathogenic / Uncertain significance /
Likely benign / Benign) on both sides, colored nodes, curved flow ribbons
colored by source category.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib as mpl

mpl.rcParams["font.family"] = "Liberation Sans"  # metric-compatible Arial substitute

CATEGORY_ORDER = ["Benign", "Likely benign", "Uncertain significance",
                   "Likely pathogenic", "Pathogenic"]

CATEGORY_COLORS = {
    "Pathogenic": "#CA7682",
    "Likely pathogenic": "#E6B1B8",
    "Uncertain significance": "#A0A0A0",
    "Likely benign": "#63A1C4",
    "Benign": "#1D7AAB",
}

def _smoothstep(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)

def draw_sankey(ax, flow_matrix, left_title, right_title,
                 category_order=CATEGORY_ORDER, colors=CATEGORY_COLORS,
                 left_category_order=None, right_category_order=None,
                 gap_frac=0.018, node_width=0.055, flow_alpha=0.65,
                 title=None, show_counts=False, box_all_labels=False,
                 size_transform="sqrt", label_fontsize=14, header_fontsize=16,
                 title_fontsize=16, right_header_fontsize=None, right_header_bold=False):
    """
    Draws a single Sankey diagram onto the given matplotlib ax. Same logic
    as plot_sankey but decoupled from figure creation/saving, so multiple
    Sankeys can be combined into one figure (e.g. a 2x2 grid).

    left_category_order / right_category_order: override category_order
    independently per side (e.g. left side only has "Uncertain significance"
    when every source variant is a ClinVar VUS - no need to draw empty
    Pathogenic/Benign/etc. nodes on that side).
    """
    left_category_order = left_category_order or category_order
    right_category_order = right_category_order or category_order

    def transform(x):
        if size_transform == "sqrt":
            return np.sqrt(x)
        return x

    flow_matrix = {k: v for k, v in flow_matrix.items() if v > 0}
    flow_matrix_scaled = {k: transform(v) for k, v in flow_matrix.items()}

    left_totals = {c: sum(flow_matrix_scaled.get((c, r), 0) for r in right_category_order) for c in left_category_order}
    right_totals = {c: sum(flow_matrix_scaled.get((l, c), 0) for l in left_category_order) for c in right_category_order}
    grand_total = sum(left_totals.values())

    # Enforce a minimum node height so a zero (or near-zero) category still
    # reserves enough vertical space for its label not to collide with the
    # neighboring label - e.g. an empty "Pathogenic" bin sitting right next
    # to "Likely pathogenic" would otherwise get a ~0-height node and both
    # labels would render on top of each other.
    max_total = max(list(left_totals.values()) + list(right_totals.values()) + [1e-9])
    min_height = 0.09 * max_total
    left_totals = {c: max(v, min_height) for c, v in left_totals.items()}
    right_totals = {c: max(v, min_height) for c, v in right_totals.items()}

    gap = gap_frac * grand_total

    def node_extents(totals, order):
        extents = {}
        y = 0.0
        for c in order:
            h = totals[c]
            extents[c] = (y, y + h)
            y += h + gap
        total_height = y - gap
        return extents, total_height

    left_extents, left_h = node_extents(left_totals, left_category_order)
    right_extents, right_h = node_extents(right_totals, right_category_order)
    total_h = max(left_h, right_h)

    left_cursor = {c: left_extents[c][1] for c in left_category_order}
    right_cursor = {c: right_extents[c][1] for c in right_category_order}

    x_left, x_right = 0.0, 1.0
    n_interp = 80
    xs = np.linspace(x_left, x_right, n_interp)
    t = np.linspace(0, 1, n_interp)
    smooth = _smoothstep(t)

    # Order flows so that a left node's outgoing segments are stacked in the
    # same top-to-bottom order as their target categories, AND a right
    # node's incoming segments are stacked in the same top-to-bottom order
    # as their source categories. This is what minimizes unnecessary visual
    # crossing (flows heading to the top of the right side leave from the
    # top of the left side, consistently on both ends) - a plain
    # largest-flow-first order (sorted by count) does not guarantee this.
    left_top_to_bottom = list(reversed(left_category_order))
    right_top_to_bottom = list(reversed(right_category_order))
    for lcat in left_top_to_bottom:
        for rcat in right_top_to_bottom:
            count = flow_matrix_scaled.get((lcat, rcat), 0)
            if count <= 0:
                continue
            l_top = left_cursor[lcat]
            l_bot = l_top - count
            left_cursor[lcat] = l_bot

            r_top = right_cursor[rcat]
            r_bot = r_top - count
            right_cursor[rcat] = r_bot

            y_top = l_top + (r_top - l_top) * smooth
            y_bot = l_bot + (r_bot - l_bot) * smooth

            c_left = np.array(mpl.colors.to_rgb(colors[lcat]))
            c_right = np.array(mpl.colors.to_rgb(colors[rcat]))
            # Build a horizontal gradient raster, then clip it to the ribbon's
            # exact polygon shape -- avoids any seam lines between segments.
            grad = c_left[None, :] * (1 - t)[:, None] + c_right[None, :] * t[:, None]
            grad_img = np.tile(grad[None, :, :], (2, 1, 1))
            y_min, y_max = float(np.min(y_bot)), float(np.max(y_top))
            im = ax.imshow(grad_img, extent=[x_left, x_right, y_min, y_max],
                            origin="lower", aspect="auto", alpha=flow_alpha, zorder=1)
            verts = list(zip(xs, y_top)) + list(zip(xs[::-1], y_bot[::-1]))
            clip_path = mpath.Path(verts)
            clip_patch = mpatches.PathPatch(clip_path, transform=ax.transData)
            im.set_clip_path(clip_patch)

    for c in left_category_order:
        bot, top = left_extents[c]
        ax.add_patch(mpatches.FancyBboxPatch((x_left - node_width, bot), node_width, max(top - bot, 1e-9),
                                              boxstyle="square,pad=0", facecolor=colors[c],
                                              edgecolor="black", linewidth=1.3, zorder=3))
    for c in right_category_order:
        bot, top = right_extents[c]
        ax.add_patch(mpatches.FancyBboxPatch((x_right, bot), node_width, max(top - bot, 1e-9),
                                              boxstyle="square,pad=0", facecolor=colors[c],
                                              edgecolor="black", linewidth=1.3, zorder=3))

    true_left_totals = {c: sum(flow_matrix.get((c, r), 0) for r in right_category_order) for c in left_category_order}
    true_right_totals = {c: sum(flow_matrix.get((l, c), 0) for l in left_category_order) for c in right_category_order}

    def label_text(cat, total_val):
        return f"{cat}\n({int(round(total_val)):,})" if show_counts else cat

    for c in left_category_order:
        bbox_kwargs = dict(boxstyle="round,pad=0.35", facecolor="white",
                            edgecolor="black", linewidth=1.1) if box_all_labels else None
        bot, top = left_extents[c]
        ymid = (bot + top) / 2
        ax.text(x_left - node_width - 0.03, ymid, label_text(c, true_left_totals[c]),
                ha="right", va="center", fontsize=label_fontsize, zorder=4,
                bbox=bbox_kwargs)

    for c in right_category_order:
        bbox_kwargs = dict(boxstyle="round,pad=0.35", facecolor="white",
                            edgecolor="black", linewidth=1.1) if box_all_labels else None
        bot, top = right_extents[c]
        ymid = (bot + top) / 2
        ax.text(x_right + node_width + 0.03, ymid, label_text(c, true_right_totals[c]),
                ha="left", va="center", fontsize=label_fontsize, zorder=4,
                bbox=bbox_kwargs)

    ax.text(x_left - node_width, total_h * 1.02, left_title, ha="center", va="bottom",
            fontsize=header_fontsize, fontweight="bold")
    ax.text(x_right + node_width, total_h * 1.02, right_title, ha="center", va="bottom",
            fontsize=(right_header_fontsize or header_fontsize),
            fontweight=("bold" if right_header_bold else "normal"))
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=30)

    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(-total_h * 0.02, total_h * 1.14)
    ax.axis("off")


def plot_sankey(flow_matrix, left_title, right_title, outpath,
                 category_order=CATEGORY_ORDER, colors=CATEGORY_COLORS,
                 left_category_order=None, right_category_order=None,
                 gap_frac=0.018, node_width=0.055, flow_alpha=0.65,
                 figsize=(11, 9.5), title=None, show_counts=False,
                 box_all_labels=False,
                 size_transform="sqrt", label_fontsize=14, header_fontsize=16):
    """
    flow_matrix: dict[(left_cat, right_cat)] -> count. Missing pairs treated as 0.
    box_all_labels: if True (default), every category label gets the same
    black-bordered white box (uniform styling). If False, no labels get boxed.
    size_transform: "sqrt" (default) compresses the dominant category so
    that small-count flows don't collapse to invisible hairlines - this
    dataset is extremely skewed (439 VUS vs 12 Pathogenic), so a linear
    scale makes anything but the biggest band unreadable. "linear" uses
    raw counts (true-to-scale, but small flows may be near-invisible).
    """
    fig, ax = plt.subplots(figsize=figsize)
    draw_sankey(ax, flow_matrix, left_title, right_title,
                category_order=category_order, colors=colors,
                left_category_order=left_category_order, right_category_order=right_category_order,
                gap_frac=gap_frac, node_width=node_width, flow_alpha=flow_alpha,
                title=title, show_counts=show_counts, box_all_labels=box_all_labels,
                label_fontsize=label_fontsize, header_fontsize=header_fontsize,
                size_transform=size_transform)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")
