#!/usr/bin/env python
"""
LABEL-seq-only Sankey variants of the VCEP VUS-reclassification figure
(analysis/plot_vus_reclassification_sankey.py's sibling): restricts to the
9 RASopathy LABEL-seq genes with ClinGen VCEP evidence codes (braf, craf,
kras, mek1, mek2, shp2, sos1, sos2, mras), producing two figures --
with and without sos2 (the one "NU" -- negative-unlabeled, zero-P/LP --
gene among them; NU mode derives fp from population unmixing rather than
real labeled negatives, so it's worth seeing whether its inclusion changes
the picture).

Usage
-----
    python analysis/plot_vus_reclassification_sankey_labelseq.py
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

from mv_analysis.sankey_plot import STRENGTH_COLOR, BND_COLOR as _BND_COLOR, plot_categorical_sankey

OUTPUT_DIR = "/data/ross/assay_calibration/multivariate/experimental_staged_fit"

_CLASS_TO_SHORT = {
    "Pathogenic": "P", "Likely Pathogenic": "LP", "VUS": "VUS",
    "Likely Benign": "LB", "Benign": "B",
}
_ORDER = ["P", "LP", "VUS", "LB", "B"]
_COLOR = {**_BND_COLOR, "VUS": STRENGTH_COLOR[0]}

NU_GENE = "sos2"  # the only NU (zero-P/LP) gene among the 9 LABEL-seq genes with evidence codes


def make_figure(df, title_suffix, out_stem):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    flow_a = (df.groupby(["original_class", "residual_class"]).size()
              .reset_index(name="count")
              .rename(columns={"original_class": "source", "residual_class": "target"}))
    plot_categorical_sankey(flow_a, _ORDER, _ORDER, _COLOR, _COLOR, ax=axes[0])
    axes[0].set_title("Original -> residual\n(PS3/BS3 stripped)", fontsize=13, fontweight="bold")

    flow_b = (df.groupby(["residual_class", "reconstructed_class"]).size()
              .reset_index(name="count")
              .rename(columns={"residual_class": "source", "reconstructed_class": "target"}))
    plot_categorical_sankey(flow_b, _ORDER, _ORDER, _COLOR, _COLOR, ax=axes[1])
    axes[1].set_title("Residual -> reconstructed\n(our canonical v3 evidence substituted)",
                       fontsize=13, fontweight="bold")

    n_genes = df["gene"].nunique()
    fig.suptitle(f"VCEP VUS reclassification, LABEL-seq genes {title_suffix}\n"
                 f"(n={len(df):,} variants, {n_genes} genes)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for ext in ("pdf", "png"):
        out_path = f"{OUTPUT_DIR}/{out_stem}.{ext}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


def main():
    df_all = pd.read_csv(f"{OUTPUT_DIR}/vus_reclassification_variants.csv")
    df_labelseq = df_all[df_all["gene_set"] == "labelseq"].copy()
    for col in ["original_class", "residual_class", "reconstructed_class"]:
        df_labelseq[col] = df_labelseq[col].map(_CLASS_TO_SHORT)

    genes_with_nu = sorted(df_labelseq["gene"].unique())
    genes_without_nu = [g for g in genes_with_nu if g != NU_GENE]
    print(f"LABEL-seq genes with NU ({len(genes_with_nu)}): {genes_with_nu}")
    print(f"LABEL-seq genes without NU ({len(genes_without_nu)}): {genes_without_nu}")

    make_figure(df_labelseq, "(incl. NU gene sos2)", "vus_reclassification_sankey_labelseq_with_nu")
    make_figure(df_labelseq[df_labelseq["gene"] != NU_GENE],
                "(excl. NU gene sos2)", "vus_reclassification_sankey_labelseq_no_nu")


if __name__ == "__main__":
    main()
