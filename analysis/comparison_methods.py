"""
Comparison methods: calibration approaches other than ExCALIBR's own, run
against the same pipeline-native variants (variant_id/score/sample columns
from *_variants.csv) so they can be plotted with the exact same
confusion-matrix/evidence-distribution/scatter functions already used for
ExCALIBR-vs-author (analysis.confusion / analysis.evidence / analysis.scatter)
— just pass this module's per-method points array in wherever those
functions expect ExCALIBR's own.

Methods
-------
acmgscaler   : github.com/badonyi/acmgscaler (Badonyi & Marsh 2025) -- run
               live via Rscript against each dataset's P/LP + B/LB scores.
               Implemented below.
gmm_baseline : slurm/simple_gmm_baseline.py's 2-component Gaussian mixture
               baseline -- run separately (not part of this pipeline).
               load_gmm_baseline_points() reads its output JSON once you've
               run it; see analysis.config.GMM_BASELINE_JSON.
force_gaussian : the canonical ExCALIBR pipeline rerun with
               force_gaussian=True. Not implemented yet -- calibrations for
               this don't exist yet. Once they do, this should be a normal
               *_calibration.json/*_variants.csv output tree (just under a
               different output_dir / dataset-name suffix), loadable via
               analysis.discovery like any other pipeline run rather than
               needing its own loader here.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import config as cfg
from analysis.plot_common import sample_matches, save_and_show

_R_SCRIPT = Path(__file__).parent / "r" / "run_acmgscaler.R"

# acmgscaler's evidence-strength labels -> integer points on the same sign
# convention as ExCALIBR's own standard_points (positive = pathogenic-
# direction, negative = benign-direction, 0 = indeterminate). Magnitudes
# match Tavtigian et al. 2018's canonical ACMG point values
# (Supporting/Moderate/Strong/VeryStrong = 1/2/4/8), which is coarser than
# ExCALIBR's own continuous 1-8 integer scale -- acmgscaler only reports
# these four tiers per direction, not every integer in between. That's fine
# for confusion-matrix comparison (only sign matters: Normal/IR/Abnormal),
# and for scatter/evidence-distribution plots that just need a points value.
ACMGSCALER_EVIDENCE_TO_POINTS = {
    "Pathogenic-VeryStrong": 8,
    "Pathogenic-Strong": 4,
    "Pathogenic-Moderate": 2,
    "Pathogenic-Supporting": 1,
    "indeterminate": 0,
    "Benign-Supporting": -1,
    "Benign-Moderate": -2,
    "Benign-Strong": -4,
    "Benign-VeryStrong": -8,
}


def run_acmgscaler(
    df_variants: pd.DataFrame,
    prior: float = 0.1,
    acmgscaler_dir: Optional[str] = None,
    score_col: str = "score",
    return_thresholds: bool = False,
):
    """Run acmgscaler::calibrate() against one dataset's variants.

    `df_variants` should be one dataset's rows from a pipeline-native
    variants table (has `sample` -- pipe-separated multi-label, see
    analysis.plot_common.sample_matches -- and `score_col`). Class labels
    are derived the same way ExCALIBR's own confusion matrices select
    controls: P/LP -> 'P', B/LB -> 'B', everything else -> '' (acmgscaler
    itself relabels anything not exactly 'P'/'B' to 'U' -- density
    estimation only uses P/B rows, but every row still gets an evidence
    call, same as ExCALIBR scoring VUS variants against its own fit).

    Returns a copy of `df_variants` with acmgscaler columns appended:
    `acmgscaler_lr`, `acmgscaler_lr_lower`, `acmgscaler_lr_upper`,
    `acmgscaler_evidence`, `acmgscaler_points` (see
    ACMGSCALER_EVIDENCE_TO_POINTS). Row order is preserved.

    If return_thresholds, also returns acmgscaler's own $score_thresholds
    table (evidence, value_lower, value, value_upper -- the score at each of
    the 8 tier-boundary LRs) as a second return value, for
    make_acmgscaler_figure.

    Requires Rscript on PATH; raises RuntimeError with the R stderr if the
    call fails.
    """
    acmgscaler_dir = acmgscaler_dir or cfg.ACMGSCALER_DIR
    if not Path(acmgscaler_dir).is_dir():
        raise FileNotFoundError(f"acmgscaler_dir not found: {acmgscaler_dir}")

    is_plp = sample_matches(df_variants, "Pathogenic/Likely Pathogenic")
    is_blb = sample_matches(df_variants, "Benign/Likely Benign")
    class_col = np.where(is_plp, "P", np.where(is_blb, "B", ""))

    r_input = pd.DataFrame({
        "class": class_col,
        "value": df_variants[score_col].values,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = Path(tmpdir) / "input.csv"
        output_csv = Path(tmpdir) / "output.csv"
        thresholds_csv = Path(tmpdir) / "thresholds.csv"
        r_input.to_csv(input_csv, index=False)

        cmd = ["Rscript", str(_R_SCRIPT), str(acmgscaler_dir), str(input_csv), str(output_csv), str(prior)]
        if return_thresholds:
            cmd.append(str(thresholds_csv))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"acmgscaler R call failed:\n{proc.stderr}")

        r_output = pd.read_csv(output_csv)
        thresholds_df = pd.read_csv(thresholds_csv) if return_thresholds else None

    out = df_variants.copy()
    out["acmgscaler_lr"] = r_output["value_lr"].values
    out["acmgscaler_lr_lower"] = r_output["value_lr_lower"].values
    out["acmgscaler_lr_upper"] = r_output["value_lr_upper"].values
    out["acmgscaler_evidence"] = r_output["value_evidence"].values
    out["acmgscaler_points"] = out["acmgscaler_evidence"].map(ACMGSCALER_EVIDENCE_TO_POINTS)

    if return_thresholds:
        return out, thresholds_df
    return out


def build_acmgscaler_confusion_matrix(
    df_variants_with_acmgscaler: pd.DataFrame, label: str = "",
) -> Optional[pd.DataFrame]:
    """Same shape/semantics as analysis.confusion.build_confusion_matrix, but
    using acmgscaler_points (from run_acmgscaler) instead of ExCALIBR's own
    standard_points/oob_points -- so it plugs into the same
    make_confusion_figure/make_single_confusion_figure/make_scatter_figure
    call sites by just adding "acmgscaler" as another key in a
    {method: [matrices...]} dict alongside ExCALIBR's own.

    Rows: [BLB, PLP]   Cols: [Normal, IR, Abnormal]
    """
    df = df_variants_with_acmgscaler
    if "acmgscaler_points" not in df.columns:
        raise KeyError("df_variants_with_acmgscaler must come from run_acmgscaler() first")

    df_plp = df[sample_matches(df, "Pathogenic/Likely Pathogenic")]
    df_blb = df[sample_matches(df, "Benign/Likely Benign")]

    if len(df_plp) == 0 and len(df_blb) == 0:
        return None

    def _counts(sub):
        pts = sub["acmgscaler_points"]
        return [int((pts < 0).sum()), int((pts == 0).sum()), int((pts > 0).sum())]

    return pd.DataFrame(
        [_counts(df_blb), _counts(df_plp)],
        index=["BLB", "PLP"], columns=["Normal", "IR", "Abnormal"],
    )


def load_gmm_baseline_points(json_path: Optional[str] = None) -> dict:
    """Load slurm/simple_gmm_baseline.py's output JSON (once you've run it --
    see analysis.config.GMM_BASELINE_JSON) into a lookup:

        {(dataset_name, variant): {"pathogenic_mean", "pathogenic_std",
                                    "benign_mean", "benign_std",
                                    "per_sample_weights", ...}}

    for `status == "fit"` entries only (skipped datasets are dropped).

    This gives the fitted 2-component Gaussian pair per dataset -- computing
    per-variant evidence points from it (LR = pathogenic_pdf(score) /
    benign_pdf(score) at each variant's score, then the same Tavtigian
    point-value mapping used elsewhere in this pipeline) is deliberately
    left to the caller once real output exists, since the exact scoring
    convention (which of plp_blb / plp_blb_synon, whether to normalize
    against the calibration's own prior) should match how that comparison
    is actually meant to be used, not be guessed at here.
    """
    json_path = json_path or cfg.GMM_BASELINE_JSON
    if not json_path:
        raise FileNotFoundError(
            "No GMM baseline JSON configured -- run slurm/simple_gmm_baseline.py "
            "and set analysis.config.GMM_BASELINE_JSON (or pass json_path explicitly)."
        )
    with open(json_path) as f:
        records = json.load(f)
    return {
        (r["dataset_name"], r["variant"]): r
        for r in records if r.get("status") == "fit"
    }


# Sample colors matching the rest of this codebase's convention (see e.g.
# plot_scoreset_best_config's sample_colors / _SAMPLE_COLORS in
# analysis/calibration_plots.py) -- P/LP red, B/LB blue.
_SAMPLE_HIST_COLORS = {"Pathogenic/Likely Pathogenic": "#CA7682", "Benign/Likely Benign": "#1D7AAB"}


def make_acmgscaler_figure(dataset: str, df_variants: pd.DataFrame, figure_dir: Path,
                            prior: float = 0.1, score_col: str = "score"):
    """Per-dataset acmgscaler figure, analogous to run_igvf_batch.py's
    per-selected-config `{dataset}_{comp}_visualization.png` -- one PNG per
    dataset, saved as `{dataset}_acmgscaler_visualization.png` in the same
    output_dir/{dataset}/ layout ExCALIBR's own visualizations use.

    Two panels:
      left  : P/LP (red) / B/LB (blue) score histograms, with acmgscaler's
              own score_thresholds marked as vertical dashed lines (one per
              evidence-tier boundary).
      right : per-variant acmgscaler LR (log scale) vs score, sorted by
              score, with horizontal dashed lines at each evidence
              tier's LR threshold -- the acmgscaler analogue of ExCALIBR's
              own "Log LR+" panel in plot_scoreset_best_config.

    Returns the Path the figure was saved to, or None if acmgscaler couldn't
    be run for this dataset (e.g. <20 non-NA scores -- see build_grid).
    """
    Path(figure_dir).mkdir(parents=True, exist_ok=True)

    df_acmg, thresholds = run_acmgscaler(
        df_variants, prior=prior, score_col=score_col, return_thresholds=True,
    )
    if df_acmg["acmgscaler_evidence"].isna().all():
        print(f"  SKIP acmgscaler figure for {dataset}: acmgscaler returned no evidence calls "
              f"(likely <20 P/B-labeled variants)")
        return None

    fig, (ax_hist, ax_lr) = plt.subplots(1, 2, figsize=(14, 6))

    for sample_name, color in _SAMPLE_HIST_COLORS.items():
        mask = sample_matches(df_variants, sample_name)
        if mask.any():
            ax_hist.hist(df_variants.loc[mask, score_col], bins=40, density=True,
                         alpha=0.5, color=color, label=sample_name)
    for _, row in thresholds.iterrows():
        if pd.notna(row["value"]):
            ax_hist.axvline(row["value"], color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax_hist.set_xlabel(score_col)
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"{dataset}: score distributions + acmgscaler thresholds")
    ax_hist.legend(fontsize=9)

    plotted = df_acmg.dropna(subset=["acmgscaler_lr"]).sort_values(score_col)
    ax_lr.fill_between(plotted[score_col], plotted["acmgscaler_lr_lower"], plotted["acmgscaler_lr_upper"],
                        color="gray", alpha=0.25, label="95% CI")
    ax_lr.plot(plotted[score_col], plotted["acmgscaler_lr"], color="black", linewidth=1, alpha=0.6, zorder=2)
    # Color points by their assigned evidence tier -- more robust than trying
    # to reconstruct exact horizontal threshold lines from score_thresholds
    # (that requires matching a score-space boundary back to an LR-space
    # value via nearest-score lookup, which is only approximate).
    n_tiers = len(ACMGSCALER_EVIDENCE_TO_POINTS)
    cmap = plt.cm.RdBu_r
    for label, points in ACMGSCALER_EVIDENCE_TO_POINTS.items():
        sub = plotted[plotted["acmgscaler_evidence"] == label]
        if sub.empty:
            continue
        color = cmap(0.5 + points / 16) if points != 0 else "gray"
        ax_lr.scatter(sub[score_col], sub["acmgscaler_lr"], color=color, s=10, zorder=3, label=label)
    ax_lr.legend(fontsize=7, loc="best", ncol=2)
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel(score_col)
    ax_lr.set_ylabel("acmgscaler LR (log scale)")
    ax_lr.set_title(f"{dataset}: acmgscaler LR curve (prior={prior})")

    fig.tight_layout()
    out_dir = Path(figure_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_acmgscaler_visualization.png"
    save_and_show(fig, out_path)
    return out_path
