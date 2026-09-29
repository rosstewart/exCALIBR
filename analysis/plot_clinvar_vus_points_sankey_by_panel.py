#!/usr/bin/env python
"""
For real ClinVar VUS variants (see analysis/compute_clinvar_vus_points_by_panel.py --
NOT PS3/BS3-stripped ACMG-code VUS, see that script's docstring for the
distinction), show how they distribute across our canonical points scale
(-8..+8), once per panel (functional, predictor, combined) -- VUS (single
source node) -> our_points bin (17 target nodes), reusing plot_categorical_sankey/
STRENGTH_COLOR exactly as analysis/plot_acmg_stripped_vus_points_sankey.py
does for the (unrelated) ACMG-stripping use case, with one addition:
gradient=True, so each ribbon fades from the source node's color to its own
target point-bin's color instead of a flat grey (there's only one source
node here, "VUS", so a flat-ribbon rendering would otherwise carry no color
signal along the flow at all, only at the target node itself).

Usage
-----
    python analysis/plot_clinvar_vus_points_sankey_by_panel.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mv_analysis import config
from mv_analysis.sankey_plot_flat import STRENGTH_COLOR, plot_categorical_sankey

OUTPUT_DIR = Path(config.PAPER_VUS_RECLASSIFICATION_DIR)

_POINT_ORDER = list(range(8, -9, -1))  # +8 ... 0 ... -8, descending top-to-bottom
_POINT_COLOR = {b: STRENGTH_COLOR[b] for b in _POINT_ORDER}
_SOURCE_ORDER = ["VUS"]
_SOURCE_COLOR = {"VUS": "#e0e0e0"}

_PANELS = [
    ("functional", "functional_true_clinvar_vus_points.csv", "Functional"),
    ("predictor", "predictor_true_clinvar_vus_points.csv", "Computational predictors"),
    ("combined", "combined_true_clinvar_vus_points.csv", "Combined (functional+predictor)"),
]


def _signed_label(v: int) -> str:
    return f"+{v}" if v > 0 else str(v)


def make_figure(df, title, out_stem):
    pivotal = df.copy()
    pivotal["point_bin"] = pivotal["our_points"].round().clip(-8, 8).astype(int)
    pivotal["source"] = "VUS"

    flow = (pivotal.groupby(["source", "point_bin"]).size()
            .reset_index(name="count")
            .rename(columns={"point_bin": "target"}))

    fig, ax = plt.subplots(figsize=(7, 9))
    plot_categorical_sankey(
        flow, _SOURCE_ORDER, _POINT_ORDER, _SOURCE_COLOR, _POINT_COLOR, ax=ax,
        target_fontsize=9, show_target_counts=True, gap_frac=0.006,
        target_label_fmt=_signed_label, gradient=True,
    )
    n_genes = pivotal["gene"].nunique()
    ax.set_title(f"{title}\nTrue ClinVar VUS -> our canonical points "
                 f"(n={len(pivotal):,} variants, {n_genes} genes)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = OUTPUT_DIR / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def main():
    for panel, csv_name, title in _PANELS:
        csv_path = OUTPUT_DIR / csv_name
        if not csv_path.exists():
            print(f"[{panel}] {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[{panel}] no rows, skipping")
            continue
        make_figure(df, title, f"{panel}_vus_points_sankey")


if __name__ == "__main__":
    main()
