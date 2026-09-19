#!/usr/bin/env python
"""
Same 3-panel (A) functional / (B) computational predictors / (C) combined
evidence figure as `mv_analysis/gene_performance_scatter.py`, comparing
ExCALIBR-MV against the UV non-conflicting aggregation baseline -- except
Panel A is restricted to the 17 LABEL-seq genes only (drops TP53, CARD11,
and the 15 plain-integrated genes that the original Panel A also includes).
Panels B and C are unchanged (they're not LABEL-seq-restricted to begin
with -- predictor/combined gene sets are disjoint from LABEL-seq).

Reuses `_gene_row`/`_all_ok_cached`/`_cached_rows`/`plot_mcc_scatter_panel`/
`build_panel_b`/`build_panel_c` from gene_performance_scatter.py unmodified,
including its per-gene `.gene_cache` (genes already cached there from the
full-panel-A run are loaded instantly, no MultiScoreset rebuild needed).

Usage
-----
    python mv_analysis/gene_performance_scatter_labelseq.py \\
        --results-json /data/ross/assay_calibration/multivariate/jobs_all_1000b_8f_v2/bootstrap_results_v3.json.gz \\
        --save-path mv_analysis/figures/gene_performance_labelseq_only.png \\
        --cache-dir mv_analysis/figures/.gene_cache
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from mv_analysis.gene_performance_scatter import (
    _all_ok_cached, _cached_rows, _gene_row, build_panel_b, build_panel_c,
    plot_mcc_scatter_panel,
)
from mv_analysis import config
from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
from src.assay_calibration.multivariate_data.common import gene_set_dataset_label


def build_panel_a_labelseq_only(results_json, cache_dir=None):
    rows = []
    if _all_ok_cached(cache_dir, "A", config.LABELSEQ_GENES):
        print("All LABEL-seq genes cached -- skipping the ms rebuild.")
        rows.extend(_cached_rows(cache_dir, "A", config.LABELSEQ_GENES))
    else:
        print("Building all LABEL-seq scoresets (one-time cost)...")
        labelseq_ms_map = build_labelseq_multiscoresets()
        for gene, ms in labelseq_ms_map.items():
            r = _gene_row(gene, "labelseq", ms, results_json,
                          dataset_name=gene_set_dataset_label(gene, "labelseq"),
                          cache_dir=cache_dir, panel="A")
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def build_gene_performance_figure_labelseq(results_json, save_path=None, cache_dir=None,
                                            panel_a_only=False, ymin=0.6):
    print("=== Panel A: functional (LABEL-seq genes only) ===")
    df_a = build_panel_a_labelseq_only(results_json, cache_dir=cache_dir)

    if panel_a_only:
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_mcc_scatter_panel(ax, df_a, "", "Functional (LABEL-seq genes only)", ymin=ymin)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        print(f"Functional (LABEL-seq only) (n={len(df_a)}): mean UV MCC {df_a['uv_mcc'].mean():.3f} -> "
              f"mean MV MCC {df_a['mv_mcc'].mean():.3f}")
        print(df_a.sort_values("gene").to_string(index=False))
        return fig, {"A": df_a}

    print("=== Panel B: computational predictors ===")
    df_b = build_panel_b(results_json, cache_dir=cache_dir)
    print("=== Panel C: combined functional+predictor ===")
    df_c = build_panel_c(results_json, cache_dir=cache_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_mcc_scatter_panel(axes[0], df_a, "(A)", "Functional (LABEL-seq genes only)", ymin=ymin)
    plot_mcc_scatter_panel(axes[1], df_b, "(B)", "Computational predictors")
    plot_mcc_scatter_panel(axes[2], df_c, "(C)", "Combined evidence")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    for name, df in [("Functional (LABEL-seq only)", df_a), ("Predictors", df_b), ("Combined", df_c)]:
        if df.empty:
            print(f"{name}: no data")
            continue
        print(f"{name} (n={len(df)}): mean UV MCC {df['uv_mcc'].mean():.3f} -> "
              f"mean MV MCC {df['mv_mcc'].mean():.3f}")
        print(df.sort_values("gene").to_string(index=False))

    return fig, {"A": df_a, "B": df_b, "C": df_c}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--save-path", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--panel-a-only", action="store_true")
    ap.add_argument("--ymin", type=float, default=0.6)
    args = ap.parse_args()

    build_gene_performance_figure_labelseq(args.results_json, save_path=args.save_path,
                                            cache_dir=args.cache_dir,
                                            panel_a_only=args.panel_a_only, ymin=args.ymin)
