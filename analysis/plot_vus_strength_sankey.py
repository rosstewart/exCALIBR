#!/usr/bin/env python
"""
LABEL-seq-only (with NU gene sos2 included) figure: VUS -> our canonical v3
evidence, consolidated into ACMG strength tiers (Supporting/Moderate/Strong/
Very Strong, mirrored for benign) instead of raw -8..+8 point bins, with
gradient (grey -> target color) flow ribbons instead of flat grey.

Two variants of the source population:
  - "stripped": VUS after removing PS3/BS3 (analysis/run_vus_reclassification.py's
    `residual_class`) -- the original figure.
  - "clinvar": genuine original ClinVar/ClinGen VUS (`original_class`) -- no
    stripping involved (there's nothing to strip from a variant that was
    already VUS), included for direct comparison.

Usage
-----
    python analysis/plot_vus_strength_sankey.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

from mv_analysis.sankey_plot import STRENGTH_COLOR

OUTPUT_DIR = "/data/ross/assay_calibration/multivariate/experimental_staged_fit"
# Zero-P/LP ("negative-unlabeled"/NU-mode) LABEL-seq genes, established
# earlier this session: araf, erbb2, grb2, ksr1, ksr2, sos2.
NU_GENES = ["araf", "erbb2", "grb2", "ksr1", "ksr2", "sos2"]

# Consolidated ACMG-strength tiers, most-pathogenic to most-benign, with the
# representative STRENGTH_COLOR anchor each tier uses (matches the canonical
# Tavtigian Supporting/Moderate/Strong/VeryStrong point anchors 1/2/4/8, same
# ones `analysis/vus_reclassification.py`'s POINTS_TO_EVIDENCE_CODE mapping
# treats as one tier: e.g. points 4-7 all map to "PS3_Strong").
_TIERS = [
    ("Very Strong (+8)",  lambda p: p == 8,          STRENGTH_COLOR[8]),
    ("Strong (+4 to +7)", lambda p: 4 <= p <= 7,      STRENGTH_COLOR[4]),
    ("Moderate (+2/+3)",  lambda p: 2 <= p <= 3,      STRENGTH_COLOR[2]),
    ("Supporting (+1)",   lambda p: p == 1,           STRENGTH_COLOR[1]),
    ("No evidence (0)",   lambda p: p == 0,           "#e0e0e0"),
    ("Supporting (-1)",   lambda p: p == -1,          STRENGTH_COLOR[-1]),
    ("Moderate (-2/-3)",  lambda p: -3 <= p <= -2,    STRENGTH_COLOR[-2]),
    ("Strong (-4 to -7)", lambda p: -7 <= p <= -4,    STRENGTH_COLOR[-4]),
    ("Very Strong (-8)",  lambda p: p == -8,          STRENGTH_COLOR[-8]),
]
_TIER_ORDER = [t[0] for t in _TIERS]
_TIER_COLOR = {t[0]: t[2] for t in _TIERS}
_SOURCE_COLOR = "#b0b0b0"


def _tier_of(points: float) -> str:
    p = int(np.clip(round(points), -8, 8))
    for name, pred, _ in _TIERS:
        if pred(p):
            return name
    raise ValueError(f"bug: point value {p} matched no tier")  # -8..8 fully covered


def _bezier(p0, p1, p2, p3, t):
    mt = 1 - t
    x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
    y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def _gradient_ribbon(ax, xs0, xs1, sy0, sy1, ty0, ty1, target_color, n_strips=40, grey_bias=0.45):
    """One flow ribbon from a source slice [sy0,sy1] to a target slice
    [ty0,ty1], filled with a color gradient from `_SOURCE_COLOR` (grey, at
    the source end) to `target_color` (at the target end), approximated as
    `n_strips` thin quadrilaterals along the cubic-Bezier path (same curve
    shape `plot_categorical_sankey` draws, just re-parametrized here so each
    slice can get its own interpolated facecolor -- flat PathPatch fill can't
    vary color along its own path)."""
    xm = (xs0 + xs1) / 2
    top_p0, top_p1, top_p2, top_p3 = (xs0, sy1), (xm, sy1), (xm, ty1), (xs1, ty1)
    bot_p0, bot_p1, bot_p2, bot_p3 = (xs0, sy0), (xm, sy0), (xm, ty0), (xs1, ty0)

    src_rgb = np.array(to_rgb(_SOURCE_COLOR))
    tgt_rgb = np.array(to_rgb(target_color))

    ts = np.linspace(0, 1, n_strips + 1)
    top_xy = [_bezier(top_p0, top_p1, top_p2, top_p3, t) for t in ts]
    bot_xy = [_bezier(bot_p0, bot_p1, bot_p2, bot_p3, t) for t in ts]

    # Build every strip as one PolyCollection instead of individual Polygon
    # patches: adjacent separately-drawn patches leave visible white seams at
    # their shared edge (each gets anti-aliased against the axes background
    # independently); a single collection anti-aliases the whole ribbon as
    # one object, so abutting strips blend into each other with no seam.
    verts, colors = [], []
    for i in range(n_strips):
        t_mid = (ts[i] + ts[i + 1]) / 2
        # grey_bias<1 biases the color ramp toward the target color sooner,
        # so grey occupies less of the ribbon's length (still grey at the
        # very source end, but reaches the target color well before x1).
        t_color = min(1.0, t_mid / grey_bias)
        color = tuple(src_rgb + (tgt_rgb - src_rgb) * t_color)
        (xt0, yt0), (xt1, yt1) = top_xy[i], top_xy[i + 1]
        (xb0, yb0), (xb1, yb1) = bot_xy[i], bot_xy[i + 1]
        verts.append([(xt0, yt0), (xt1, yt1), (xb1, yb1), (xb0, yb0)])
        colors.append(color)
    coll = PolyCollection(verts, facecolors=colors, edgecolors=colors,
                           linewidths=0.4, alpha=0.75, zorder=1, antialiaseds=True)
    ax.add_collection(coll)


def plot_vus_to_strength(df_pivotal, source_label, ax, node_width=0.05):
    counts = df_pivotal["tier"].value_counts()
    total = len(df_pivotal)
    gap = 0.02 * total

    tgt_totals = {tier: int(counts.get(tier, 0)) for tier in _TIER_ORDER}
    tgt_pos = {}
    y = 0.0
    for tier in _TIER_ORDER:
        h = tgt_totals[tier]
        if h == 0:
            continue
        tgt_pos[tier] = (y, y + h)
        y += h + gap
    tgt_height = max(y - gap, 0.0)

    src_height = total
    max_height = max(src_height, tgt_height)

    x0, x1 = 0.0, 1.0
    xs0, xs1 = x0 + node_width / 2, x1 - node_width / 2

    # source node (full VUS column)
    ax.add_patch(mpatches.Rectangle((x0 - node_width / 2, 0), node_width, src_height,
                                     facecolor=_SOURCE_COLOR, edgecolor="black", linewidth=0.8, zorder=3))
    ax.text(x0 - node_width / 2 - 0.02, src_height / 2, f"{source_label} ({total:,})",
            ha="right", va="center", fontsize=10, fontweight="bold")

    # target nodes
    for tier in _TIER_ORDER:
        if tier not in tgt_pos:
            continue
        y0, y1 = tgt_pos[tier]
        ax.add_patch(mpatches.Rectangle((x1 - node_width / 2, y0), node_width, y1 - y0,
                                         facecolor=_TIER_COLOR[tier], edgecolor="black",
                                         linewidth=0.8, zorder=3))
        ax.text(x1 + node_width / 2 + 0.02, (y0 + y1) / 2, f"{tier} ({tgt_totals[tier]:,})",
                ha="left", va="center", fontsize=9)

    # ribbons -- one contiguous source slice per target, stacked in _TIER_ORDER
    src_cursor = 0.0
    for tier in _TIER_ORDER:
        if tier not in tgt_pos:
            continue
        n = tgt_totals[tier]
        sy0, sy1 = src_cursor, src_cursor + n
        src_cursor = sy1
        ty0, ty1 = tgt_pos[tier]
        _gradient_ribbon(ax, xs0, xs1, sy0, sy1, ty0, ty1, _TIER_COLOR[tier])

    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(-gap, max_height + gap)
    ax.invert_yaxis()
    ax.axis("off")


def make_figure(stripped, clinvar, suptitle, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    plot_vus_to_strength(stripped, "ClinGen VUS\n(PS3/BS3 removed)", axes[0])
    plot_vus_to_strength(clinvar, "ClinVar VUS", axes[1])
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        out_path = f"{OUTPUT_DIR}/{out_stem}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def main():
    df_all = pd.read_csv(f"{OUTPUT_DIR}/vus_reclassification_variants.csv")
    df_labelseq = df_all[df_all["gene_set"] == "labelseq"].copy()
    df_labelseq["tier"] = df_labelseq["our_points"].map(_tier_of)
    stripped = df_labelseq[df_labelseq["residual_class"] == "VUS"]

    # Real ClinVar VUS (analysis/compute_true_clinvar_vus_points.py): every
    # variant genuinely labeled "Uncertain significance" by ClinVar itself
    # (same criterion as `Variant.is_vus` in dataset.py -- clinvar_sig_2026 +
    # a review-quality gate), independent of whether ClinGen recorded any
    # evidence codes for it. NOT `original_class=='VUS'` from the
    # reclassification CSV -- that column is a re-derivation from ClinGen's
    # applied evidence codes via `classify_acmg`, restricted to the tiny
    # subset of variants a VCEP actually curated; conflating the two badly
    # undercounted true ClinVar VUS. Also built from an `all_assayed` ms
    # (not the default build), since the default build's keep_mask silently
    # drops "pure" VUS variants with no P/LP/B/LB/gnomAD/Synonymous role.
    clinvar = pd.read_csv(f"{OUTPUT_DIR}/true_clinvar_vus_points.csv")
    clinvar["tier"] = clinvar["our_points"].map(_tier_of)

    make_figure(stripped, clinvar,
                "LABEL-seq genes, all 17 (incl. 6 NU genes): VUS -> canonical v3 ACMG-strength tier",
                "vus_strength_sankey_labelseq_with_nu")

    stripped_no_nu = stripped[~stripped["gene"].isin(NU_GENES)]
    clinvar_no_nu = clinvar[~clinvar["gene"].isin(NU_GENES)]
    make_figure(stripped_no_nu, clinvar_no_nu,
                "LABEL-seq genes, 11 non-NU only: VUS -> canonical v3 ACMG-strength tier",
                "vus_strength_sankey_labelseq_no_nu")


if __name__ == "__main__":
    main()
