#!/usr/bin/env python
"""
For the variants that become VUS after stripping PS3/BS3 (functional-assay
evidence), show how they distribute across our canonical v3 model's discrete
-8..+8 point scale -- i.e. VUS (single source node) -> our_points bin (17
target nodes), reusing the same `plot_categorical_sankey` primitive and the
`STRENGTH_COLOR` point-bin palette already defined for exactly this kind of
figure in `mv_analysis/sankey_plot.py::
plot_evidence_sankey_pair_points` (there: ClinVar class -> point bin; here:
residual-VUS -> point bin, single source).

Usage
-----
    python analysis/plot_vus_points_sankey.py
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

from mv_analysis.sankey_plot import STRENGTH_COLOR, plot_categorical_sankey

OUTPUT_DIR = "/data/ross/assay_calibration/multivariate/experimental_staged_fit"

_POINT_ORDER = list(range(8, -9, -1))  # +8 ... 0 ... -8, descending top-to-bottom
_POINT_COLOR = {b: STRENGTH_COLOR[b] for b in _POINT_ORDER}
_SOURCE_ORDER = ["VUS"]
_SOURCE_COLOR = {"VUS": "#e0e0e0"}


def _signed_label(v: int) -> str:
    return f"+{v}" if v > 0 else str(v)


def make_figure(df, title, out_stem):
    pivotal = df[df["residual_class"] == "VUS"].copy()
    pivotal["point_bin"] = pivotal["our_points"].round().clip(-8, 8).astype(int)
    pivotal["source"] = "VUS"

    flow = (pivotal.groupby(["source", "point_bin"]).size()
            .reset_index(name="count")
            .rename(columns={"point_bin": "target"}))

    fig, ax = plt.subplots(figsize=(7, 9))
    plot_categorical_sankey(
        flow, _SOURCE_ORDER, _POINT_ORDER, _SOURCE_COLOR, _POINT_COLOR, ax=ax,
        target_fontsize=9, show_target_counts=True, gap_frac=0.006,
        target_label_fmt=_signed_label,
    )
    n_genes = pivotal["gene"].nunique()
    ax.set_title(f"{title}\nPS3/BS3-stripped VUS -> our canonical v3 points "
                 f"(n={len(pivotal):,} variants, {n_genes} genes)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = f"{OUTPUT_DIR}/{out_stem}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def main():
    df_all = pd.read_csv(f"{OUTPUT_DIR}/vus_reclassification_variants.csv")

    make_figure(df_all, "All 16 genes (LABEL-seq + integrated)", "vus_points_sankey_all")

    df_labelseq = df_all[df_all["gene_set"] == "labelseq"]
    make_figure(df_labelseq, "LABEL-seq genes (incl. NU gene sos2)",
                "vus_points_sankey_labelseq_with_nu")
    make_figure(df_labelseq[df_labelseq["gene"] != "sos2"],
                "LABEL-seq genes (excl. NU gene sos2)", "vus_points_sankey_labelseq_no_nu")


if __name__ == "__main__":
    main()
