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
gmm_baseline : simple 2-component GMM baseline, two pooling variants
               ("plp_blb", "plp_blb_synon"), prior=0.1. Its output tree is
               shaped exactly like ExCALIBR's own pipeline output (same
               {dataset}/{dataset}_{comp}_variants.csv layout, just
               comp="plp_blb"/"plp_blb_synon" instead of an (n_c,
               benign_method) token) -- load_comparison_variants() reads it
               generically, and the resulting variants.csv has the same
               sample/standard_points columns ExCALIBR's own does, so
               analysis.confusion.build_confusion_matrix works on it
               completely unchanged. See analysis.config.GMM_BASELINE_OUTPUT_DIR.
               load_gmm_baseline_points() is a fallback for the older,
               leaner bare-JSON-only output format (no variants.csv) --
               prefer load_comparison_variants when GMM_BASELINE_OUTPUT_DIR
               is populated.
force_gaussian : the canonical ExCALIBR pipeline rerun with
               force_gaussian=True (results to come). Once populated, this
               is a normal ExCALIBR output_dir with standard (n_c,
               benign_method) comp naming -- load it via
               analysis.discovery/analysis.confusion like any other pipeline
               run, not via anything in this module.
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


def _find_pvst(prior: float) -> int:
    """Python port of acmgscaler's find_pvst() (R/add_evidence_levels.R) --
    the "very strong" pathogenicity LR threshold for a given prior. Only used
    to derive threshold-line positions for make_acmgscaler_figure; the actual
    per-variant classification always comes from the live R call (run_acmgscaler),
    never from this port -- so a divergence here would only mis-draw a
    reference line, not mis-classify a variant. Verified directly against a
    live `Rscript -e 'source(...); find_pvst(0.1)'` call: both give 350, and
    thresholds_from_pvst(350) matches the R output to displayed precision on
    all 8 tiers (do not confuse this with ExCALIBR's own unrelated Tavtigian
    constant from get_tavtigian_constant, which is a different formula that
    happens to also be an integer -- they are not the same number).

    Ported line-for-line from R, including its `- 1` at the end -- not
    obviously an off-by-one in the R source (find_pvst's own comment: "13 out
    of 14 criteria met"), so reproduced exactly as written rather than
    "corrected".
    """
    if prior <= 0.01:
        return 8573
    if prior > 0.97:
        return 1

    for pvst in range(1, 8574):
        su = pvst ** 0.125
        mo = pvst ** 0.25
        st = pvst ** 0.5
        class_lr = np.array([
            mo * pvst, mo * st, su**2 * st, mo**3, su**2 * mo**2, su**4 * mo,
            st * pvst, mo**2 * pvst, su * mo * pvst, su**2 * pvst, st**2,
            mo**3 * st, su**2 * mo**2 * st, su**4 * mo * st,
        ])
        post_path = (class_lr * prior) / ((class_lr - 1) * prior + 1)
        n_met = int((post_path[:6] >= 0.9).sum()) + int((post_path[6:] >= 0.99).sum())
        if n_met == 13:
            return pvst - 1
    raise ValueError(f"find_pvst: no pvst in [1, 8573] satisfies 13/14 criteria for prior={prior}")


def _thresholds_from_pvst(pvst: float) -> dict:
    """Python port of acmgscaler's thresholds_from_pvst() -- LR value at each
    of the 8 evidence-tier boundaries, for the given Pvst (see _find_pvst)."""
    return {
        "Benign-VeryStrong": 1 / pvst,
        "Benign-Strong": 1 / pvst**0.5,
        "Benign-Moderate": 1 / pvst**0.25,
        "Benign-Supporting": 1 / pvst**0.125,
        "Pathogenic-Supporting": pvst**0.125,
        "Pathogenic-Moderate": pvst**0.25,
        "Pathogenic-Strong": pvst**0.5,
        "Pathogenic-VeryStrong": pvst,
    }


def build_variants_df_from_scoreset(scoreset) -> pd.DataFrame:
    """variant_id/score/sample DataFrame straight from a freshly-built
    Scoreset -- no calibration/pipeline output needed. `sample` is the same
    pipe-separated multi-label string _build_standard_table writes to
    *_variants.csv (see src/assay_calibration/pipeline/variant_evidence.py);
    reused here so run_acmgscaler/make_acmgscaler_figure work identically
    whether df_variants came from a saved *_variants.csv or was built fresh
    from the master dataframe like run_igvf_batch.py/slurm/simple_gmm_baseline.py do.
    """
    from src.assay_calibration.pipeline.variant_evidence import _get_variant_ids

    ids = _get_variant_ids(scoreset)
    rows = []
    for idx in range(len(scoreset.scores)):
        matched = [
            scoreset.sample_names[s_idx]
            for s_idx in range(len(scoreset.sample_names))
            if scoreset._sample_assignments[idx, s_idx]
        ]
        rows.append({
            "variant_id": ids[idx] if idx < len(ids) else f"variant_{idx}",
            "score": float(scoreset.scores[idx]),
            "sample": "|".join(matched) if matched else "Unknown",
        })
    return pd.DataFrame(rows)


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

    Requires Rscript on PATH. Raises ValueError up front (no R subprocess
    call at all) if there are fewer than 10 P/LP or fewer than 10 B/LB
    variants -- acmgscaler's own check_input() would reject this dataset
    with the same 10-per-class minimum anyway, but only after a full R
    startup + the noisy multi-line "Calls: calibrate -> check_input /
    Execution halted" stderr dump; checking here first means a caller doing
    `except Exception: print(f"SKIP: {e}")` gets one clean line instead.
    Any other R failure still raises RuntimeError with the R stderr, as before.
    """
    acmgscaler_dir = acmgscaler_dir or cfg.ACMGSCALER_DIR
    if not Path(acmgscaler_dir).is_dir():
        raise FileNotFoundError(f"acmgscaler_dir not found: {acmgscaler_dir}")

    is_plp = sample_matches(df_variants, "Pathogenic/Likely Pathogenic")
    is_blb = sample_matches(df_variants, "Benign/Likely Benign")
    n_plp, n_blb = int(is_plp.sum()), int(is_blb.sum())
    if n_plp < 10 or n_blb < 10:
        raise ValueError(
            f"insufficient P/B controls for acmgscaler (needs >=10 each): "
            f"n_P={n_plp}, n_B={n_blb}"
        )
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


def acmgscaler_variants_path(dataset: str, output_dir) -> Path:
    """Path convention for the per-dataset acmgscaler CSV run_acmgscaler_all.py
    writes -- {output_dir}/{dataset}/{dataset}_acmgscaler_variants.csv,
    mirroring ExCALIBR's own {dataset}/{dataset}_{comp}_variants.csv layout.
    """
    return Path(output_dir) / dataset / f"{dataset}_acmgscaler_variants.csv"


def load_acmgscaler_variants(dataset: str, output_dir) -> Optional[pd.DataFrame]:
    """Load a previously-saved {dataset}_acmgscaler_variants.csv (from
    run_acmgscaler_all.py) if it exists -- has the same acmgscaler_lr/
    acmgscaler_evidence/acmgscaler_points columns run_acmgscaler() itself
    returns, so build_acmgscaler_confusion_matrix works on it directly with
    no R subprocess call needed. Returns None if no such file exists (caller
    should fall back to run_acmgscaler() for a live computation).
    """
    path = acmgscaler_variants_path(dataset, output_dir)
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_comparison_variants(dataset: str, comp: str, output_dir) -> Optional[pd.DataFrame]:
    """Generic loader for any comparison method whose output is shaped like
    ExCALIBR's own pipeline output -- {output_dir}/{dataset}/{dataset}_{comp}_variants.csv
    -- with the same sample/standard_points columns, so
    analysis.confusion.build_confusion_matrix works on the result completely
    unchanged. Used for the GMM baseline (comp="plp_blb"/"plp_blb_synon",
    see analysis.config.GMM_BASELINE_VARIANTS) and would work identically for
    any future comparison method saved in this same shape.

    Returns None if no such file exists (e.g. that dataset/comp combo was
    skipped -- too few controls, same as ExCALIBR's own calibration would be).
    """
    path = Path(output_dir) / dataset / f"{dataset}_{comp}_variants.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


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
# analysis/calibration_plots.py) -- P/LP red, B/LB blue, gnomAD gray,
# Synonymous green.
_SAMPLE_HIST_COLORS = {
    "Pathogenic/Likely Pathogenic": "#CA7682",
    "Benign/Likely Benign": "#1D7AAB",
    "population": "#A0A0A0",
    "gnomAD": "#A0A0A0",
    "Synonymous": "#6BAA75",
}

# Fixed row order score_thresholds is always written in (see
# acmgscaler's build_threshold_df) -- boundary i is between tier_order[i]
# and tier_order[i+1] with "indeterminate" spliced in at index 4.
_ACMGSCALER_TIER_ORDER = [
    "Benign-VeryStrong", "Benign-Strong", "Benign-Moderate", "Benign-Supporting",
    "indeterminate",
    "Pathogenic-Supporting", "Pathogenic-Moderate", "Pathogenic-Strong", "Pathogenic-VeryStrong",
]


def _point_ranges_from_acmgscaler_thresholds(thresholds: pd.DataFrame) -> dict:
    """Convert acmgscaler's 8-row score_thresholds table (boundary score
    between each pair of adjacent evidence tiers, in the fixed semantic order
    _ACMGSCALER_TIER_ORDER) into the same {point_value: [[lo, hi]]} shape
    ExCALIBR's own point_ranges use -- so the exact "Point Assignments" band
    renderer from plot_scoreset_best_config (src/assay_calibration/plot_utils/
    utils.py) can be reused verbatim.

    Direction-agnostic: the 8 boundary values may increase or decrease along
    _ACMGSCALER_TIER_ORDER depending on whether higher scores are more or
    less pathogenic for this dataset -- inferred by comparing the first and
    last boundary rather than assumed.
    """
    boundary_vals = thresholds["value"].tolist()
    increasing = boundary_vals[0] < boundary_vals[-1]
    edges = ([-np.inf] if increasing else [np.inf]) + boundary_vals + ([np.inf] if increasing else [-np.inf])

    point_ranges = {}
    for i, tier in enumerate(_ACMGSCALER_TIER_ORDER):
        points = ACMGSCALER_EVIDENCE_TO_POINTS[tier]
        if points == 0:
            continue
        lo, hi = edges[i], edges[i + 1]
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        point_ranges[points] = [[lo, hi]]
    return point_ranges


def make_acmgscaler_figure(dataset: str, df_variants: pd.DataFrame, figure_dir: Path,
                            prior: float = 0.1, score_col: str = "score",
                            acmgscaler_dir: Optional[str] = None,
                            precomputed: Optional[tuple] = None):
    """Per-dataset acmgscaler figure, laid out like
    src.assay_calibration.plot_utils.utils.plot_scoreset_best_config (the
    same figure run_igvf_batch.py/run_pipeline.py save as
    `{dataset}_{comp}_visualization.png`) rather than an ad hoc format --
    saved as `{dataset}_acmgscaler_visualization.png` under a `{dataset}/`
    subdirectory of `figure_dir` (figure_dir itself can be any destination,
    not necessarily the pipeline's own output_dir).

    Three rows, reusing that figure's exact styling where the underlying
    data lets it:
      Row 0 : per-sample score histograms (same P/LP-red, B/LB-blue,
              gnomAD-gray, Synonymous-green convention). No fitted mixture
              component overlay curves -- acmgscaler is a density-ratio/KDE
              method, not a parametric mixture fit, so there's no per-
              component curve to draw here the way ExCALIBR's own figure does.
      Row 1 : "Point Assignments" -- the exact horizontal-band renderer from
              plot_scoreset_best_config, fed acmgscaler's own evidence-tier
              score boundaries (converted to ExCALIBR's point_ranges shape by
              _point_ranges_from_acmgscaler_thresholds).
      Row 2 : "Log LR+" -- acmgscaler's own per-variant LR estimate (black
              line) with its 95% CI (gray band), plus the same dashed
              red/blue reference lines plot_scoreset_best_config draws
              (add_thresholds), computed from acmgscaler's own Pvst
              (_find_pvst/_thresholds_from_pvst -- verified against a live R
              call, see their docstrings) rather than ExCALIBR's Tavtigian
              constant, since these are two close-but-not-identical numbers
              (348 vs 350 at prior=0.1) and this figure should show what
              acmgscaler itself actually used.

    `precomputed`, if given, is the (df_acmg, thresholds) tuple already
    returned by a prior run_acmgscaler(..., return_thresholds=True) call --
    pass this when the caller already ran acmgscaler itself (e.g. to also
    save a *_acmgscaler_variants.csv), so this function doesn't invoke the R
    subprocess a second time for the same dataset.

    Returns the Path the figure was saved to, or None if acmgscaler couldn't
    be run for this dataset (e.g. <20 non-NA scores -- see build_grid).
    """
    from src.assay_calibration.plot_utils.utils import add_thresholds

    Path(figure_dir).mkdir(parents=True, exist_ok=True)

    if precomputed is not None:
        df_acmg, thresholds = precomputed
    else:
        df_acmg, thresholds = run_acmgscaler(
            df_variants, prior=prior, score_col=score_col, return_thresholds=True,
            acmgscaler_dir=acmgscaler_dir,
        )
    if df_acmg["acmgscaler_evidence"].isna().all():
        print(f"  SKIP acmgscaler figure for {dataset}: acmgscaler returned no evidence calls "
              f"(likely <20 P/B-labeled variants)")
        return None

    present_samples = [
        name for name in ["Pathogenic/Likely Pathogenic", "Benign/Likely Benign", "population", "Synonymous"]
        if sample_matches(df_variants, name).any()
    ]
    # "population" and "gnomAD" share one color/column -- don't double-count.
    n_cols = max(len(present_samples), 1)

    fig, ax = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14), squeeze=False,
                            gridspec_kw={"hspace": 0.35, "wspace": 0.3})

    xlim = (df_variants[score_col].min(), df_variants[score_col].max())
    point_ranges_dict = _point_ranges_from_acmgscaler_thresholds(thresholds)
    point_ranges = sorted(point_ranges_dict.items())
    point_values = [pr[0] for pr in point_ranges]

    plotted = df_acmg.dropna(subset=["acmgscaler_lr"]).sort_values(score_col)

    for col, sample_name in enumerate(present_samples):
        # Row 0: score histogram for this sample.
        ax_hist = ax[0, col]
        mask = sample_matches(df_variants, sample_name)
        ax_hist.hist(df_variants.loc[mask, score_col], bins=40, density=True,
                     alpha=0.6, color=_SAMPLE_HIST_COLORS[sample_name])
        ax_hist.set_title(f"{sample_name}\n(n={int(mask.sum()):,d})", fontsize=11)
        ax_hist.set_xlabel(score_col)
        ax_hist.set_ylabel("Density")
        ax_hist.set_xlim(xlim)

        # Row 1: Point Assignments -- verbatim rendering logic from
        # plot_scoreset_best_config's Row 1 (see that function for the
        # original), just fed acmgscaler-derived point_ranges.
        ax_points = ax[1, col]
        for pointIdx, (pointVal, scoreRanges) in enumerate(point_ranges):
            for sr in scoreRanges:
                x0 = xlim[0] if np.isneginf(sr[0]) else max(sr[0], xlim[0])
                x1 = xlim[1] if np.isposinf(sr[1]) else min(sr[1], xlim[1])
                ax_points.plot([x0, x1], [pointIdx, pointIdx],
                              color="red" if pointVal > 0 else "blue",
                              linestyle="-", alpha=0.7, linewidth=2)
        ax_points.set_ylim(-1, len(point_values))
        ax_points.set_yticks(range(len(point_values)),
                             labels=[f"{v:+d}" if v != 0 else "0" for v in point_values])
        ax_points.set_xlabel(score_col)
        ax_points.set_ylabel("Points")
        ax_points.set_title("Point Assignments", fontsize=11)
        ax_points.set_xlim(xlim)
        ax_points.grid(linewidth=0.5, alpha=0.3)

        # Row 2: Log LR+ -- acmgscaler's own per-variant LR + CI, with
        # add_thresholds' dashed reference lines at acmgscaler's own tiers.
        ax_lr = ax[2, col]
        ax_lr.fill_between(plotted[score_col], np.log(plotted["acmgscaler_lr_lower"]),
                           np.log(plotted["acmgscaler_lr_upper"]), color="gray", alpha=0.3)
        ax_lr.plot(plotted[score_col], np.log(plotted["acmgscaler_lr"]), color="black", alpha=0.7)

        pvst = _find_pvst(prior)
        tau = _thresholds_from_pvst(pvst)
        tauP = np.log([tau["Pathogenic-Supporting"], tau["Pathogenic-Moderate"],
                       tau["Pathogenic-Strong"], tau["Pathogenic-VeryStrong"]])
        tauB = np.log([tau["Benign-Supporting"], tau["Benign-Moderate"],
                       tau["Benign-Strong"], tau["Benign-VeryStrong"]])
        add_thresholds(tauP, tauB, ax_lr)
        ax_lr.set_xlabel(score_col)
        ax_lr.set_ylabel("Log LR (acmgscaler)")
        ax_lr.set_xlim(xlim)
        ax_lr.set_title(f"prior={prior}, Pvst={pvst}", fontsize=11)

    fig.suptitle(f"{dataset}: acmgscaler", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_dir = Path(figure_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_acmgscaler_visualization.png"
    save_and_show(fig, out_path)
    return out_path
