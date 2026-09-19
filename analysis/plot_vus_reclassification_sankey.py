#!/usr/bin/env python
"""
Sankey figure for the VCEP VUS-reclassification analysis (analysis/
run_vus_reclassification.py's output). Two panels, reusing the project's own
vendored Sankey plotter (`mv_analysis/sankey_plot.py::
plot_categorical_sankey`), copied there from tavtigian_sims (untracked):

  (A) original ClinVar/ClinGen classification -> residual classification
      after stripping PS3/BS3 (functional-assay evidence).
  (B) residual classification -> reconstructed classification after
      substituting our own canonical-v3 MV evidence in place of PS3/BS3.

Usage
-----
    python analysis/plot_vus_reclassification_sankey.py
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


def main():
    df = pd.read_csv(f"{OUTPUT_DIR}/vus_reclassification_variants.csv")
    for col in ["original_class", "residual_class", "reconstructed_class"]:
        df[col] = df[col].map(_CLASS_TO_SHORT)

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

    fig.suptitle(f"VCEP VUS reclassification, canonical v3 evidence (n={len(df):,} variants, 16 genes)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = f"{OUTPUT_DIR}/vus_reclassification_sankey.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    out_png = f"{OUTPUT_DIR}/vus_reclassification_sankey.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
