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


def make_figure(df, title, out_stem, slides=False):
    """No title in either mode (removed -- was redundant with the panel's
    own filename/caption context). ``slides=True``: a simplified, all-text-
    larger rendering for a slideshow (not the paper) -- same data, same
    layout, just bigger source/target labels and a bigger figure so it reads
    from the back of a room."""
    pivotal = df.copy()
    pivotal["point_bin"] = pivotal["our_points"].round().clip(-8, 8).astype(int)
    pivotal["source"] = "VUS"

    flow = (pivotal.groupby(["source", "point_bin"]).size()
            .reset_index(name="count")
            .rename(columns={"point_bin": "target"}))

    figsize = (9, 11) if slides else (7, 9)
    fig, ax = plt.subplots(figsize=figsize)
    plot_categorical_sankey(
        flow, _SOURCE_ORDER, _POINT_ORDER, _SOURCE_COLOR, _POINT_COLOR, ax=ax,
        source_fontsize=20 if slides else 9, target_fontsize=18 if slides else 9,
        show_target_counts=True, gap_frac=0.02 if slides else 0.006,
        # Bigger min_frac in slides mode: small bins (e.g. "+7 (10)") still
        # need enough layout height that their much-larger label doesn't
        # vertically collide with its neighbors' labels.
        min_frac=0.035 if slides else 0.012,
        target_label_fmt=_signed_label, gradient=True,
    )
    fig.tight_layout()
    stem = f"{out_stem}_slides" if slides else out_stem
    for ext in ("pdf", "png"):
        out_path = OUTPUT_DIR / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def main(slides=False):
    for panel, csv_name, title in _PANELS:
        csv_path = OUTPUT_DIR / csv_name
        if not csv_path.exists():
            print(f"[{panel}] {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"[{panel}] no rows, skipping")
            continue
        make_figure(df, title, f"{panel}_vus_points_sankey", slides=slides)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slides", action="store_true",
                     help="Also render the simplified, large-text slideshow version "
                          "(saved alongside the paper version with a _slides suffix).")
    args = ap.parse_args()
    main()
    if args.slides:
        main(slides=True)
