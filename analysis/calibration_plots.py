"""
Per-dataset calibration detail figures, working directly from pipeline output
(*_variants.csv + *_calibration.json + *_lr_values.json.gz) — no pickled
Scoreset/fits objects required.

Moved verbatim from analysis/plot_utils.py — no visual changes. (For plots
that need the actual fitted mixture-density curves rather than just LR+
percentile bands, see analysis/legacy_fits.py + the original
plot_scoreset_best_config / plot_scoreset_final_pillar_project_v2 functions
in src/assay_calibration/plot_utils/utils.py.)
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

from src.assay_calibration.plot_utils.utils import log_thresholds_with_ylim_pad
from analysis.plot_common import sample_matches


from analysis.plot_common import save_and_show, pretty_method as _pretty
_save = save_and_show


# ---------------------------------------------------------------------------
# LR values loading
# ---------------------------------------------------------------------------

def load_lr_values(output_dir: Path, dataset_name: str, method: Optional[str], comp: str) -> Optional[Dict]:
    """Load *_lr_values.json.gz for a given dataset/method/component.

    Two on-disk formats exist (see run_igvf_batch.py::_load_calibration_from_disk):
      - compact: precomputed log_lr_plus_p5/p50/p95 percentile arrays only
      - legacy: full log_lr_plus matrix (one row per bootstrap)

    Returns dict with keys: score_range, log_lr_pct (shape (3, n_score_points):
    p5/p50/p95 — already-percentile, do NOT re-percentile this), prior,
    scoreset_flipped. Returns None if the file doesn't exist or can't be loaded.
    """
    if method:
        fname = f"{dataset_name}_{method}_{comp}_lr_values.json.gz"
    else:
        fname = f"{dataset_name}_{comp}_lr_values.json.gz"
    candidates = sorted(Path(output_dir).rglob(fname))
    if not candidates:
        return None
    try:
        with gzip.open(candidates[0], "rt", encoding="utf-8") as f:
            data = json.load(f)
        if "log_lr_plus_p5" in data:
            log_lr_pct = np.array([data["log_lr_plus_p5"], data["log_lr_plus_p50"], data["log_lr_plus_p95"]])
        else:
            llr = np.asarray(data["log_lr_plus"])
            log_lr_pct = np.nanpercentile(llr, [5, 50, 95], axis=0)
        return {
            "score_range": np.array(data["score_range"]),
            "log_lr_pct": log_lr_pct,
            "prior": float(data["prior"]),
            "scoreset_flipped": bool(data.get("scoreset_flipped", False)),
        }
    except Exception as e:
        print(f"  WARNING: could not load LR values from {candidates[0]}: {e}")
        return None


# ---------------------------------------------------------------------------
# Log LR+ curves with thresholds, one subplot per dataset
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "tavtigian": "#1f77b4",
    "acmg_bayes": "#2ca02c",
    # Legacy tags from runs predating the ACMG-Bayes consolidation.
    "piecewise": "#ff7f0e",
    "continuous": "#2ca02c",
    "strict_additive": "#9467bd",
    "default": "#1f77b4",
}
_FALLBACK_COLORS = ["#e377c2", "#8c564b", "#bcbd22", "#17becf"]


def make_log_lr_figure(
    lr_by_method: Dict[str, Dict[str, Dict]],
    dataset_names: List[str],
    point_ranges_by_method: Dict[str, Dict[str, Dict]],
    figure_dir: Path,
    tag: str = "",
    ncols: int = 5,
):
    """Grid of log LR+ plots, one subplot per dataset, all methods overlaid."""
    methods = [m for m in lr_by_method if any(d in lr_by_method[m] for d in dataset_names)]
    if not methods:
        print("  SKIP log_lr figure: no LR data loaded")
        return

    fallback_iter = iter(_FALLBACK_COLORS)
    method_color = {m: _METHOD_COLORS.get(m) or next(fallback_iter, "#333333") for m in methods}

    nrows = (len(dataset_names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3.2),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})

    legend_handles: Dict[str, object] = {}

    for idx, ds in enumerate(dataset_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        plotted = False
        for m in methods:
            lr = lr_by_method[m].get(ds)
            if lr is None:
                continue

            score_range = lr["score_range"]
            prior = lr["prior"]
            color = method_color[m]

            llr_curves = lr["log_lr_pct"]

            line, = ax.plot(score_range, llr_curves[1], color=color,
                            linewidth=1.6, label=_pretty(m))
            ax.fill_between(score_range, llr_curves[0], llr_curves[2],
                            color=color, alpha=0.15)
            legend_handles[m] = line
            plotted = True

            pr_dict = (point_ranges_by_method.get(m) or {}).get(ds)
            if pr_dict and m == methods[0]:
                point_values = sorted({abs(int(k)) for k in pr_dict})
                if point_values and prior > 0:
                    try:
                        tauP, tauB, _ylim_top, _ylim_bottom = log_thresholds_with_ylim_pad(
                            prior, point_values,
                        )
                        for tp in tauP:
                            ax.axhline(tp, color="red", linestyle="--",
                                       linewidth=0.7, alpha=0.55)
                        for tb in tauB:
                            ax.axhline(tb, color="steelblue", linestyle="--",
                                       linewidth=0.7, alpha=0.55)
                    except Exception:
                        pass

        if not plotted:
            ax.set_visible(False)
            continue

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(ds.replace("_", " "), fontsize=7, pad=3)
        ax.set_xlabel("Score", fontsize=7)
        ax.set_ylabel("log LR+", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    for idx in range(len(dataset_names), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            [_pretty(m) for m in legend_handles],
            loc="upper center",
            ncol=len(legend_handles),
            fontsize=9,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 1.01),
        )

    tag_suffix = f"_{tag}" if tag else ""
    method_str = "_".join(methods)
    _save(fig, figure_dir / f"log_lr_plus_{method_str}{tag_suffix}.png")


# ---------------------------------------------------------------------------
# Per-dataset calibration details
# Mirrors the layout of plot_scoreset_best_config: histograms / point ranges /
# LR+ curves in three rows, one column per sample present in the variants CSV.
# Works from saved pipeline outputs only — no GMM fits required.
# ---------------------------------------------------------------------------

# "population" is the literal sample string the pipeline's own *_variants.csv
# uses for the gnomAD population sample (compute_variant_table writes
# scoreset.sample_names as-is; only some legacy scripts renamed it to
# "gnomAD" ad hoc) — kept both here so either literal string is recognized.
_SAMPLE_ORDER = [
    "Pathogenic/Likely Pathogenic",
    "Benign/Likely Benign",
    "population",
    "gnomAD",
    "Synonymous",
]
_SAMPLE_COLORS = {
    "Pathogenic/Likely Pathogenic": "#CA7682",
    "Benign/Likely Benign":         "#1D7AAB",
    "population":                   "#A0A0A0",
    "gnomAD":                       "#A0A0A0",
    "Synonymous":                   "#6BAA75",
}
_SAMPLE_ALPHAS = {
    "Pathogenic/Likely Pathogenic": 0.6,
    "Benign/Likely Benign":         0.6,
    "population":                   0.3,
    "gnomAD":                       0.3,
    "Synonymous":                   0.5,
}
_SAMPLE_LABELS = {
    "Pathogenic/Likely Pathogenic": "P/LP",
    "Benign/Likely Benign":         "B/LB",
    "population":                   "gnomAD",
    "gnomAD":                       "gnomAD",
    "Synonymous":                   "Synonymous",
}
_PV_LINESTYLES = {1: "dotted", 2: "dashed", 4: "dashdot", 8: (5, (10, 3))}


def make_calibration_figure(
    df_variants: pd.DataFrame,
    calibrations_by_method: Dict[str, Optional[Dict]],
    lr_by_method: Dict[str, Optional[Dict]],
    dataset: str,
    figure_dir: Path,
    tag: str = "",
):
    """Per-dataset calibration details figure: histograms / point ranges / LR+ curves.

    Saves one PNG per dataset: ``calibration_{dataset}[_{tag}].png``.
    """
    samples_present = [
        s for s in _SAMPLE_ORDER if sample_matches(df_variants, s).any()
    ]
    n_samples = len(samples_present)
    if n_samples == 0:
        print(f"  SKIP calibration figure for {dataset}: no sample data")
        return

    methods = [m for m in calibrations_by_method if calibrations_by_method.get(m) is not None]
    if not methods:
        return

    first_cal = calibrations_by_method[methods[0]]
    flipped = bool(first_cal.get("scoreset_flipped", False))
    pr_first = {int(k): v for k, v in first_cal.get("point_ranges", {}).items()}

    fig, ax = plt.subplots(
        3, n_samples,
        figsize=(5 * n_samples, 14),
        squeeze=False,
        gridspec_kw={"hspace": 0.38, "wspace": 0.30},
    )
    fig.suptitle(dataset.replace("_", " "), fontsize=13, fontweight="bold", y=1.01)

    for col, sname in enumerate(samples_present):
        a = ax[0, col]
        df_s = df_variants[sample_matches(df_variants, sname)]
        color = _SAMPLE_COLORS.get(sname, "#888888")
        alpha = _SAMPLE_ALPHAS.get(sname, 0.5)
        if not df_s.empty:
            if _HAS_SNS:
                sns.histplot(df_s["score"], stat="density", ax=a, alpha=alpha, color=color)
            else:
                a.hist(df_s["score"], density=True, alpha=alpha, color=color, bins=30)
            for pv, ranges in pr_first.items():
                ls = _PV_LINESTYLES.get(abs(pv), "solid")
                col_th = "red" if pv > 0 else "steelblue"
                for lo, hi in ranges:
                    thresh = lo if (pv > 0) != flipped else hi
                    a.axvline(thresh, color=col_th, linestyle=ls, linewidth=0.9, alpha=0.6)
        label = _SAMPLE_LABELS.get(sname, sname)
        a.set_title(f"{label}\n(n={len(df_s):,})", fontsize=9)
        a.set_xlabel("Score", fontsize=8)
        if col == 0:
            a.set_ylabel("Density", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(True, alpha=0.2, linewidth=0.5)

    all_pv: set = set()
    for m in methods:
        cal = calibrations_by_method.get(m) or {}
        all_pv.update(int(k) for k in cal.get("point_ranges", {}))
    sorted_pv = sorted(all_pv)
    pv_to_row = {pv: i for i, pv in enumerate(sorted_pv)}

    for col in range(n_samples):
        a = ax[1, col]
        for midx, m in enumerate(methods):
            cal = calibrations_by_method.get(m) or {}
            pr = {int(k): v for k, v in cal.get("point_ranges", {}).items()}
            mc = _METHOD_COLORS.get(m, "#333333")
            labelled = False
            for pv in sorted(pr):
                row_y = pv_to_row.get(pv, 0) + midx * 0.12
                for lo, hi in pr[pv]:
                    a.plot(
                        [lo, hi], [row_y, row_y],
                        color=mc, linewidth=2.5, alpha=0.8,
                        label=_pretty(m) if not labelled else "",
                    )
                    labelled = True
        a.set_yticks(range(len(sorted_pv)))
        a.set_yticklabels(
            [f"{pv:+d}" for pv in sorted_pv] if col == 0 else [""] * len(sorted_pv),
            fontsize=7,
        )
        if col == 0:
            a.set_ylabel("Points", fontsize=8)
            a.set_title("Point Assignment Ranges", fontsize=9)
            if len(methods) > 1:
                a.legend(fontsize=7, loc="best")
        a.set_xlabel("Score", fontsize=8)
        a.grid(True, alpha=0.2, linewidth=0.5)

    for col in range(n_samples):
        a = ax[2, col]
        for midx, m in enumerate(methods):
            lr = lr_by_method.get(m)
            if lr is None:
                continue
            sr = np.asarray(lr["score_range"])
            prior = float(lr["prior"])
            mc = _METHOD_COLORS.get(m, "#333333")
            pct = np.asarray(lr["log_lr_pct"])
            a.plot(sr, pct[1], color=mc, linewidth=1.5, label=_pretty(m))
            a.fill_between(sr, pct[0], pct[2], color=mc, alpha=0.15)
            if midx == 0:
                cal = calibrations_by_method.get(m) or {}
                pvs = sorted({abs(int(k)) for k in cal.get("point_ranges", {})})
                if pvs and prior > 0:
                    try:
                        tauP, tauB, _ylim_top, _ylim_bottom = log_thresholds_with_ylim_pad(prior, pvs)
                        for tp in tauP:
                            a.axhline(tp, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
                        for tb in tauB:
                            a.axhline(tb, color="steelblue", linestyle="--", linewidth=0.7, alpha=0.5)
                    except Exception:
                        pass
        a.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        a.set_xlabel("Score", fontsize=8)
        if col == 0:
            a.set_ylabel("log LR+", fontsize=8)
            a.set_title("Log LR+  (5th / median / 95th pct)", fontsize=9)
            if len(methods) > 1:
                a.legend(fontsize=7)
        a.tick_params(labelsize=7)
        a.grid(True, alpha=0.2, linewidth=0.5)

    tag_suffix = f"_{tag}" if tag else ""
    _save(fig, figure_dir / f"calibration_{dataset}{tag_suffix}.png")
