"""
Extended-data appendix PDF builder.

Unifies two near-duplicate legacy notebook-style scripts:

- ``test/auxiliary_fig_creation/extended_data_fig2_pp.py`` /
  ``extended_data_fig2_pp_utils.py`` — the "full" panel style (per-sample
  mixture-fit row + combined-histogram row + ExCALIBR calibration-strength
  bar + legend row), always drawing threshold lines. Default dataset list was
  "every dataset minus a hand-curated exclude-list minus TP53/F9-prefixed
  datasets".
- ``test/auxiliary_fig_creation/supp_pdf_creation_calibration_paper.py`` /
  ``supp_pdf_creation_calibration_paper_utils.py`` — the "stacked" panel
  style (one histogram+fit-density panel per sample, stacked vertically),
  with thresholds optionally suppressed (``PLOT_THRESHOLDS=False`` in the
  original driver). Dataset list was explicit: every key of the
  ``--dataset-configs`` JSON.

Both drivers are folded into one entry point, :func:`build_appendix_pdf`,
selected by ``plot_thresholds`` (True -> "full" style matching
``extended_data_fig2_pp.py``; False -> "stacked" style matching
``supp_pdf_creation_calibration_paper.py``) plus two convenience wrappers,
:func:`build_extended_data_fig2` and :func:`build_supp_pdf`, that reproduce
each script's exact call.

All plotting code (colors, figsizes, GridSpec ratios, linestyles, legends,
titles) is preserved verbatim from the sources above. What changed:

- Data loading: ``src.assay_calibration.plot_utils.utils.load_dataset_for_plot``
  (which read hand-curated ``point_assignment_*/{dataset}/*.pkl`` files) is
  replaced by ``analysis.legacy_fits.load_scoreset_and_fits``, which assembles
  the same information from ``run_pipeline.py``/``run_igvf_batch.py`` output
  directories. See ``_load_full_style``/``_load_stacked_style`` below for how
  the tuple-unpacking was adapted.
- The hardcoded ``import_dataset_configurations()`` dataset table is no
  longer used for dataset-list discovery: the "full" style's default dataset
  list is built from ``analysis.discovery.discover_outputs``'s tree, and the
  "stacked" style's explicit dataset list is read from the
  ``analysis.config.DATASET_CONFIGS`` JSON.
- Reads of ``new_names_dict.pkl`` / ``datasets_to_exclude.pkl`` are guarded
  with ``analysis.config.warn_if_missing`` — if either file is absent on this
  machine, that specific filtering/renaming step is skipped (with a printed
  warning) rather than crashing the whole builder.

Importing this module has no side effects: nothing is plotted or written to
disk until one of the ``build_*`` functions is called.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Silence matplotlib completely (matches the two source _utils.py files).
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.pyplot").setLevel(logging.ERROR)
logging.getLogger("joblib").setLevel(logging.ERROR)
logging.getLogger("loky").setLevel(logging.ERROR)

import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = [
    'Arial',            # preferred
    'Helvetica',
    'Nimbus Sans',
    'DejaVu Sans'       # guaranteed fallback
]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from joblib import Parallel, delayed

from src.assay_calibration.plot_utils.utils import sample_density

from analysis import config as cfg
from analysis.discovery import discover_outputs
from analysis.legacy_fits import load_scoreset_and_fits
from analysis.plot_common import is_notebook

GENES_2018 = ["BRCA1", "MSH2", "PTEN", "TP53"]


# ---------------------------------------------------------------------------
# Guarded external-file loads (new_names_dict.pkl, datasets_to_exclude.pkl)
# ---------------------------------------------------------------------------

_NEW_NAMES_DICT_PKL = "/data/ross/assay_calibration/dataframe/new_names_dict.pkl"
_DATASETS_TO_EXCLUDE_PKL = "/data/ross/assay_calibration/datasets_to_exclude.pkl"


def _load_new_names_dict() -> Dict[str, str]:
    """Optional legacy-name -> display-name mapping. Skips (returns {}) if missing."""
    if cfg.warn_if_missing(_NEW_NAMES_DICT_PKL, "new_names_dict.pkl (dataset display-name mapping)"):
        return {}
    with open(_NEW_NAMES_DICT_PKL, "rb") as f:
        return pickle.load(f)


def _load_datasets_to_exclude() -> set:
    """Optional hand-curated dataset exclusion list. Skips (returns set()) if missing."""
    if cfg.warn_if_missing(_DATASETS_TO_EXCLUDE_PKL, "datasets_to_exclude.pkl (dataset exclusion list)"):
        return set()
    with open(_DATASETS_TO_EXCLUDE_PKL, "rb") as f:
        return set(pickle.load(f))


# ---------------------------------------------------------------------------
# Default dataset-list construction (one per style)
# ---------------------------------------------------------------------------

def _default_dataset_list_full(output_dir: Optional[str] = None) -> List[str]:
    """Default dataset list for the "full" (plot_thresholds=True) style.

    Replaces extended_data_fig2_pp.py's
    ``[d for d in df.Dataset.unique() if not d.endswith("_clinvar_2018")
    and d not in datasets_to_exclude and not d.startswith("TP53")
    and not d.startswith("F9")]`` with the same exclusion logic applied to
    the dataset names discovered on disk under ``analysis.config.OUTPUT_DIR``.
    """
    output_dir = Path(output_dir or cfg.OUTPUT_DIR)
    tree, _model_selections, _calibrations = discover_outputs(output_dir)
    candidates = list(tree.keys())

    datasets_to_exclude = _load_datasets_to_exclude()

    dataset_list = sorted([
        d for d in candidates
        if not d.endswith("_clinvar_2018")
        and d not in datasets_to_exclude
        and not d.startswith("TP53")
        and not d.startswith("F9")
    ])
    return dataset_list


def _default_dataset_list_configs(dataset_configs_path: Optional[str] = None) -> List[str]:
    """Default (explicit) dataset list for the "stacked" (plot_thresholds=False)
    style: matches supp_pdf_creation_calibration_paper.py's
    ``list(new_dataset_configs.keys())``.
    """
    dataset_configs_path = dataset_configs_path or cfg.DATASET_CONFIGS
    with open(dataset_configs_path) as f:
        new_dataset_configs = json.load(f)
    return list(new_dataset_configs.keys())


# ---------------------------------------------------------------------------
# Per-dataset loading (replaces load_dataset_for_plot / load_scoreset calls)
# ---------------------------------------------------------------------------

def _load_full_style(
    dataset: str,
    output_dir: Optional[str],
    dataset_tsv: Optional[str],
    precomputed_fits: Optional[str],
    dataset_configs_path: Optional[str],
):
    """Load everything needed for one "full"-style panel.

    Mirrors extended_data_fig2_pp_utils.py's ``_plot_single_page`` load block:

        scoreset = load_scoreset(df, dataset, clinvar_release="2025")
        if dataset.split("_")[0] in genes_2018:
            _, indv_summary, fits, score_range_arr, config, n_c, scoreset_flipped, n_samples = \\
                load_dataset_for_plot(dataset + "_clinvar_2018", ...)
            scoreset_2018 = load_scoreset(df, dataset, clinvar_release="2018", for_fit=True)
        else:
            _, indv_summary, fits, score_range_arr, config, n_c, scoreset_flipped, n_samples = \\
                load_dataset_for_plot(dataset, ...)
            scoreset_2018 = load_scoreset(df, dataset, clinvar_release="2025", for_fit=True)

    ``load_scoreset_and_fits`` returns the scoreset it built *together with*
    the fit results derived from it, so the "fit dataset" call below supplies
    both ``scoreset_2018`` (the fit-companion scoreset used for the top row of
    individual-sample fits) and (indv_summary, fits, score_range, ...). The
    plain "display" scoreset used for the combined-histogram/calibration rows
    is loaded from ``dataset`` directly (only needed as a separate call when
    the fit dataset differs, i.e. for 2018-ClinVar genes).

    Returns
    -------
    (scoreset, scoreset_2018, indv_summary, fits, score_range, n_c, scoreset_flipped, n_samples)
    """
    is_2018_gene = dataset.split("_")[0] in GENES_2018
    # `dataset` is normally the base name (no "_clinvar_2018" suffix) --
    # `_default_dataset_list_full` filters those out -- but callers can also
    # pass in a `datasets` list built from `df["dataset"].unique()`
    # (analyze_pipeline_output.py section 1), where BRCA1/MSH2/PTEN/TP53
    # currently only exist on disk *already* carrying the literal
    # "_clinvar_2018" suffix (no separate non-2018 sibling -- same fact noted
    # elsewhere in analyze_pipeline_output.py). Appending the suffix again in
    # that case doubles it (e.g. "BRCA1_Findlay_2018_clinvar_2018" ->
    # "..._clinvar_2018_clinvar_2018"), which load_scoreset_and_fits can't
    # find on disk -- silently dropping these four genes from the appendix
    # (the FileNotFoundError is caught and skipped per-dataset by the caller).
    already_suffixed = dataset.endswith("_clinvar_2018")
    base_dataset = dataset[: -len("_clinvar_2018")] if already_suffixed else dataset
    fit_dataset = dataset if already_suffixed else (
        base_dataset + "_clinvar_2018" if is_2018_gene else base_dataset
    )

    scoreset_2018, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped = (
        load_scoreset_and_fits(
            fit_dataset, output_dir=output_dir, dataset_tsv=dataset_tsv,
            precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
        )
    )

    if is_2018_gene:
        # No separate non-2018 dataset exists on disk for these four genes
        # (only the "_clinvar_2018"-suffixed one was ever run through the
        # pipeline) -- load_scoreset_and_fits's own `pipeline_dataset` param
        # exists exactly for this: reuse fit_dataset's on-disk
        # calibration/fits, but build the Scoreset from `base_dataset` with
        # clinvar_release="2025" for a "current ClinVar" display panel (see
        # that function's docstring, which names BRCA1/MSH2/PTEN/TP53
        # explicitly for this case).
        scoreset, _, _, _, _, _, _ = load_scoreset_and_fits(
            base_dataset, pipeline_dataset=fit_dataset, clinvar_release="2025",
            output_dir=output_dir, dataset_tsv=dataset_tsv,
            precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
        )
    else:
        scoreset = scoreset_2018

    return scoreset, scoreset_2018, indv_summary, fits, score_range, n_c, scoreset_flipped, n_samples


def _load_stacked_style(
    dataset: str,
    output_dir: Optional[str],
    dataset_tsv: Optional[str],
    precomputed_fits: Optional[str],
    dataset_configs_path: Optional[str],
):
    """Load everything needed for one "stacked"-style panel.

    Mirrors supp_pdf_creation_calibration_paper_utils.py's
    ``_plot_single_page_stacked`` load block. Note the original's two
    branches (``if dataset.split("_")[0] in genes_2018 ... else ...``) called
    ``load_dataset_for_plot(dataset, ...)`` identically in both cases — the
    branching was a no-op there, since (unlike the "full" style) this
    dataset_list already carries an explicit "_clinvar_2018" suffix on its
    own entries where relevant. So there is just one load here.

    Returns
    -------
    (scoreset, indv_summary, fits, score_range, n_c, scoreset_flipped, n_samples)
    """
    scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped = (
        load_scoreset_and_fits(
            dataset, output_dir=output_dir, dataset_tsv=dataset_tsv,
            precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
        )
    )
    return scoreset, indv_summary, fits, score_range, n_c, scoreset_flipped, n_samples


def _sample_scores(scoreset, sample_num):
    """Score array for raw sample slot `sample_num` (0=P/LP, 1=B/LB,
    2=gnomAD/population, 3=Synonymous), or None if that sample is empty.
    `sample_assignments`'s columns are compacted (only non-empty samples get
    a column), so the raw slot has to be mapped to its compacted column
    index -- same skip-counting every per-sample plotting loop in this file
    already does inline."""
    if scoreset.sample_counts[sample_num] == 0:
        return None
    sample_idx = sum(1 for i in range(sample_num) if scoreset.sample_counts[i] > 0)
    mask = scoreset.sample_assignments[:, sample_idx]
    return scoreset.scores[mask]


def _compute_robust_bin_width(scoreset, all_scores, x_min, x_max, n_target_bins=30, max_bin_multiplier=4):
    """Shared histogram bin width for a dataset's P/LP, B/LB, and SNV/gnomAD
    panels, robust to a control sample having very few (e.g. 1) variants.

    A single-variant sample's own score range is 0, so it can't sensibly
    define a bin width itself; letting seaborn auto-choose one for just that
    sample (the old `except ValueError:` fallback, which drops `binwidth`
    entirely) picks a width set by seaborn's own default rule, completely
    disconnected from the assay's actual score scale -- the "bin width can
    become out of control if there is only 1 sample" bug. Instead, compute a
    candidate width from each of the P/LP, B/LB, and SNV (`all_scores`)
    samples' own (finite, non-degenerate) ranges, and use the smallest
    candidate for every sample. The well-populated SNV sample almost always
    yields a sensible, fine-grained width, and a degenerate (range-0 or
    single-point) sample simply contributes no candidate at all rather than
    forcing an oversized or auto-chosen one.

    A raw per-sample range can just as easily be pathologically *small*
    instead (e.g. two P/LP variants that happen to score very close
    together) -- taking that as the shared width for the whole visible axis
    then means an enormous bin count, rendering every sample (not just the
    narrow one) as a wall of overplotted, indistinguishably thin bars (the
    "all samples are just black spikes" bug, e.g.
    ASPA_Grønbæk-Thygesen_2024_toxicity). Floor the final width at
    axis_based_width / `max_bin_multiplier` so no single sample's local
    range can push the shared resolution more than that many times finer
    than the plain axis-based default.
    """
    axis_based = (x_max - x_min) / n_target_bins
    candidates = []
    for sample_num in (0, 1):  # P/LP, B/LB
        data = _sample_scores(scoreset, sample_num)
        if data is None or len(data) < 2:
            continue
        data_range = data.max() - data.min()
        if data_range > 0:
            candidates.append(data_range / n_target_bins)
    if all_scores is not None and len(all_scores) >= 2:
        snv_range = all_scores.max() - all_scores.min()
        if snv_range > 0:
            candidates.append(snv_range / n_target_bins)
    if not candidates:
        return axis_based
    return max(min(candidates), axis_based / max_bin_multiplier)


# ---------------------------------------------------------------------------
# Single-dataset demo plot (extended_data_fig2_pp_utils.py::plot_calibration) —
# verbatim except for the sample_density import above.
# ---------------------------------------------------------------------------

def plot_calibration(dataset, scoreset_2018, scoreset, indv_summary, fits, score_range,
                     n_c, n_samples, flipped=False):
    """
    Combined calibration plot with:
    - Top row: Individual sample fits (one per column) showing mixture components
    - Second row: All samples overlayed in one histogram
    - Third row: ExCALIBR calibration
    - Bottom: Legend

    Parameters
    ----------
    scoreset_2018 : Scoreset
        Scoreset with 2018 ClinVar for fitting
    scoreset : Scoreset
        Scoreset with current ClinVar for display
    indv_summary : dict
        Summary containing point_ranges and prior
    fits : array
        Bootstrap fit results
    score_range : array
        Score range for plotting
    n_c : int
        Number of mixture components
    n_samples : int
        Number of samples
    flipped : bool, optional
        Whether scoreset is flipped (default: False)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """

    # Sample colors
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']  # P/LP, B/LB, gnomAD, Synonymous
    sample_alphas = [0.5, 0.5, 0.15, 0.4]

    # Threshold configuration - all evidence codes
    point_values_to_plot = [1, 2, 3, 4, 5, 6, 7, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5)),
                  (0, (5, 5)), (0, (3, 1, 1, 1)), 'solid']
    linewidths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # Strength colors for all evidence codes
    strength_color = {
        # Benign (blue gradient) - anchors at -1, -2, -3, -4, -8
        -8: '#4b91a6',  # Very Strong
        -7: '#5DA3BD',
        -6: '#6FAACE',
        -5: '#74ABCE',
        -4: '#7ab5d1',  # Strong
        -3: '#99c8dc',  # Moderate+
        -2: '#d0e8f0',  # Moderate
        -1: '#e4f1f6',  # Supporting
        # Neutral
        0: '#e0e0e0',   # Indeterminate
        # Pathogenic (red gradient) - anchors at 1, 2, 3, 4, 8
        1: '#e6b1b8',   # Supporting
        2: '#d68f99',   # Moderate
        3: '#ca7682',   # Moderate+
        4: '#b85c6b',   # Strong
        5: '#B1535F',
        6: '#AA4E58',
        7: '#A2484F',
        8: '#943744'    # Very Strong
    }

    # Evidence point labels
    map_point_to_text = {i: f"{i:+d}" if i != 0 else "0" for i in range(-8, 9)}

    # Count actual non-empty samples for layout
    n_actual_samples = sum(1 for count in scoreset_2018.sample_counts if count > 0)

    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 11))

    gs = gridspec.GridSpec(
        4, n_actual_samples,
        height_ratios=[2, 2, 1, 0.3],
        hspace=0.35,
        wspace=0.15
    )

    # Create axes
    ax_fits = []
    ax_hist = plt.subplot(gs[1, :])
    ax_excalibr = plt.subplot(gs[2, :])
    leg_ax = plt.subplot(gs[3, :])
    leg_ax.axis('off')

    # Get score ranges for x-axis limits. `score_range` is the fit's own LR
    # grid, which can be narrower than the actual displayed data (e.g.
    # CARD11_Meitlis_2020_SGE_Ibrutinib_GoF's highest-scoring P/LP variant
    # sits just outside it) -- widen to also cover every real score so
    # set_xlim below can't silently clip a real variant's bar off-screen.
    x_min = min(score_range[0], float(np.nanmin(scoreset.scores)), float(np.nanmin(scoreset_2018.scores)))
    x_max = max(score_range[-1], float(np.nanmax(scoreset.scores)), float(np.nanmax(scoreset_2018.scores)))

    # Get point ranges for threshold plotting
    point_ranges = indv_summary['point_ranges']

    all_scores = scoreset.snv_scores
    bin_width = _compute_robust_bin_width(scoreset, all_scores, x_min, x_max, n_target_bins=50)

    # Pre-compute thresholds for fit panels
    threshold_info = []

    for idx, point_val in enumerate(point_values_to_plot):
        # Benign thresholds
        for pv, score_ranges_pr in point_ranges.items():
            if pv == -point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[0] if not flipped else sr[1]
                    threshold_info.append((pv, threshold_score, '#2166AC', linestyles[idx], linewidths[idx]))
                    break
                break

        # Pathogenic thresholds
        for pv, score_ranges_pr in point_ranges.items():
            if pv == point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[1] if not flipped else sr[0]
                    threshold_info.append((pv, threshold_score, '#B2182B', linestyles[idx], linewidths[idx]))
                    break
                break

    sample_name_shortener = {
        "Pathogenic/Likely Pathogenic": "ClinVar P/LP",
        "Benign/Likely Benign": "ClinVar B/LB",
        "population": "gnomAD",
        "gnomAD": "gnomAD",
        "Synonymous": "Synonymous"
    }

    fontsize_subtitle = 18
    fontsize_legend = 13
    fontsize_count = 14

    # ===== TOP ROW: Individual fits with components =====
    num_skipped = 0
    for sample_num in range(len(scoreset_2018.sample_counts)):
        if scoreset_2018.sample_counts[sample_num] == 0:
            num_skipped += 1
            continue

        sample_idx = sample_num - num_skipped
        ax = plt.subplot(gs[0, sample_idx])
        ax_fits.append(ax)

        sample_mask = scoreset_2018.sample_assignments[:, sample_idx]
        sample_name = sample_name_shortener.get(scoreset_2018.sample_names[sample_num],
                                                scoreset_2018.sample_names[sample_num])
        color = sample_colors[sample_num]

        hist_data = scoreset_2018.scores[sample_mask]
        n_count = sample_mask.sum()

        try:
            sns.histplot(hist_data, binwidth=bin_width, binrange=(x_min, x_max), stat='density', ax=ax,
                   alpha=0.5, color=color)
        except ValueError as e:
            # binwidth+binrange should always succeed since bin count is
            # then derived from the full (non-degenerate) axis span rather
            # than this sample's own (possibly single-point) data range --
            # see _compute_robust_bin_width's docstring on why a dropped
            # binrange previously let seaborn fall back to an
            # axis-scale-independent default width of 1.0. Keep binrange
            # here too so this rarer fallback (e.g. all-NaN data) still
            # can't produce an oversized bar.
            sns.histplot(hist_data, binrange=(x_min, x_max), stat='density', ax=ax,
                   alpha=0.5, color=color)

        density_sample = sample_density(score_range, fits, sample_idx)

        # Plot total fit
        d_total = np.nansum(density_sample, axis=1)
        d_total_perc = np.percentile(d_total, [5, 50, 95], axis=0)

        ax.fill_between(score_range, d_total_perc[0], d_total_perc[2],
                       color='gray', alpha=0.3)
        ax.plot(score_range, d_total_perc[1],
               color='black', alpha=0.65, linewidth=2)

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('')
        ax.set_ylabel('Density' if sample_idx == 0 else '', fontsize=fontsize_subtitle)
        ax.tick_params(axis='both', labelsize=9)

        # Add threshold lines
        for pv, thresh_score, thresh_color, thresh_ls, thresh_lw in threshold_info:
            if abs(pv) in [1,2,4,8]:
                ax.axvline(thresh_score, color=thresh_color, linestyle=thresh_ls,
                          linewidth=thresh_lw, alpha=0.8)

        # Legend
        face_rgba = to_rgba(color, 0.5)
        hist_patch = Patch(facecolor=face_rgba, edgecolor='black')

        if sample_num == 2:  # gnomAD - add prior
            legend_label = f'{sample_name}\nprior: {indv_summary["prior"]:.3f}\n(n={n_count:,d})'
        else:
            legend_label = f'{sample_name}\n(n={n_count:,d})'

        if flipped:
            loc = 'upper left' if sample_num == 0 else 'upper right'
        else:
            loc = 'upper right' if sample_num == 0 else 'upper left'
        ax.legend([hist_patch], [legend_label],
                 loc=loc, fontsize=fontsize_legend, framealpha=0.9)

    ax_fits[0].set_title('ExCALIBR sample fits', loc='left', pad=3, fontsize=fontsize_subtitle)#, style='italic')

    # ===== SECOND ROW: Combined histogram =====
    # Twin y-axes (control variant count on the left, SNV count on the
    # right) instead of a single shared density axis -- same fix as
    # analysis/figure4/panels.py::plot_panel_b / fit_hist_snv_plot.py's
    # use_twin_axes path: 'density' stat on one shared axis lets the
    # much-more-numerous "All SNVs" sample rescale P/LP and B/LB down to
    # near-invisible bars (or vice versa), since density normalizes each
    # sample's bars to its own total count. Counts on two independently-
    # scaled axes keep every sample's bars visually legible. No axis labels
    # are added -- purely an internal left/right scaling split, not meant to
    # be called out on the plot itself.
    sample_handles = []
    ax_twin = ax_hist.twinx()

    num_skipped = 0
    for sample_num in range(len(scoreset.sample_counts)):
        if scoreset.sample_counts[sample_num] == 0:
            num_skipped += 1
            continue
        if sample_num == 3:  # Skip synonymous
            continue

        sample_idx = sample_num - num_skipped
        sample_mask = scoreset.sample_assignments[:, sample_idx]
        sample_name = sample_name_shortener.get(scoreset.sample_names[sample_num],
                                               scoreset.sample_names[sample_num])
        color = sample_colors[sample_num]
        alpha = sample_alphas[sample_num]

        # For gnomAD (sample_num 2), use all SNV scores and rename
        if sample_num == 2:
            hist_data = all_scores
            display_name = 'All SNVs'
            n_count = len(all_scores)
            target_ax = ax_twin
        else:
            hist_data = scoreset.scores[sample_mask]
            display_name = sample_name
            n_count = sample_mask.sum()
            target_ax = ax_hist

        try:
            sns.histplot(hist_data, binwidth=bin_width, binrange=(x_min, x_max), stat='count', ax=target_ax,
                   alpha=alpha, color=color)
        except ValueError as e:
            sns.histplot(hist_data, binrange=(x_min, x_max), stat='count', ax=target_ax,
                   alpha=alpha, color=color)

        face_rgba = to_rgba(color, alpha)
        hist_patch = Patch(facecolor=face_rgba, edgecolor='black')
        sample_handles.append((hist_patch, f'{display_name} (n={n_count:,d})'))

    ax_hist.set_xlim(x_min, x_max)
    ax_twin.set_xlim(x_min, x_max)
    ax_hist.set_ylim(0, 1.18 * max([p.get_height() for p in ax_hist.patches]) if ax_hist.patches else 1.0)
    ax_twin.set_ylim(0, 1.18 * max([p.get_height() for p in ax_twin.patches]) if ax_twin.patches else 1.0)
    ax_hist.set_xlabel('')
    ax_hist.set_ylabel('')
    ax_twin.set_ylabel('')
    ax_hist.tick_params(axis='both', labelsize=10)
    ax_twin.tick_params(axis='y', labelsize=10)
    ax_hist.set_title('Experimental score distributions', loc='left', pad=3, fontsize=fontsize_subtitle)#, style='italic')

    # Sample legend
    sample_legend_handles = [h[0] for h in sample_handles]
    sample_legend_labels = [h[1] for h in sample_handles]

    ax_hist.legend(
        sample_legend_handles,
        sample_legend_labels,
        loc='upper left',
        fontsize=fontsize_legend,
    )

    # ===== THIRD ROW: ExCALIBR calibration =====
    # Build intervals directly from point_ranges without flipped logic
    intervals = []

    # Iterate through all possible point values in order
    all_point_values = sorted([pv for pv in point_ranges.keys() if pv != 0])

    # Add each point value's range(s) -- a point value can have more than one
    # disjoint score interval (e.g. DDX3X_Radford_2023's +1), so every entry
    # in score_ranges_list must become its own interval, not just the first.
    for point_val in all_point_values:
        score_ranges_list = point_ranges[point_val]
        if not score_ranges_list:
            continue

        for range_idx, score_range_tuple in enumerate(score_ranges_list):
            if range_idx == 0 and dataset == "PAX6_McDonnell_2024_LE9_geneticin" and point_val == 8:
                start = x_min
            else:
                start = score_range_tuple[0]
            end = score_range_tuple[1]

            intervals.append((point_val, start, end))

    # Fill every real gap between evidence intervals -- including at the
    # outer axis edges -- with a "0" (indeterminate) interval, not just the
    # single central gap the old sign/flipped-based heuristic assumed.
    # DDX3X_Radford_2023's +1 has a second range entirely on the opposite
    # side of zero from its first (its point_ranges are genuinely
    # non-monotonic across the score axis), leaving its own separate
    # indeterminate gap between that and the nearest other evidence interval
    # that the old single-gap logic never filled. Walking the real intervals
    # in score order and filling whatever gaps actually exist reproduces the
    # old single-gap result exactly for ordinary (monotonic) datasets, and
    # additionally catches any further gaps like this one.
    intervals_by_start = sorted(intervals, key=lambda x: x[1])
    filled = []
    prev_end = x_min
    for pv, s, e in intervals_by_start:
        if s > prev_end:
            filled.append((0, prev_end, s))
        filled.append((pv, s, e))
        prev_end = max(prev_end, e)
    if prev_end < x_max:
        filled.append((0, prev_end, x_max))
    if not intervals_by_start:
        filled = [(0, x_min, x_max)]
    intervals = filled

    # Sort intervals by start position for plotting
    intervals_sorted = sorted(intervals, key=lambda x: x[1])

    # Plot intervals
    for point_val, start, end in intervals_sorted:
        # The outermost point value's interval is unbounded (start=-inf or
        # end=inf, from point_ranges' calibration.json) -- axvspan builds a
        # Rectangle from (start, end) directly, and start + width where
        # width = end - start works out to -inf + inf = NaN whenever start
        # itself is -inf, which silently renders nothing (the "left half of
        # the ExCALIBR bar is blank white" bug -- see
        # analysis/figure4/panels.py plot_panel_b for the same fix). Clip to
        # the axis's own finite data range for drawing only; `count` below
        # still uses the true (possibly infinite) bounds so it keeps
        # including every variant in that tier, not just the ones inside
        # x_min/x_max.
        draw_start, draw_end = max(start, x_min), min(end, x_max)
        ax_excalibr.axvspan(draw_start, draw_end, color=strength_color[point_val], alpha=1.0)
        count = ((all_scores >= start) & (all_scores < end)).sum()
        if (draw_end - draw_start) > 0.2:
            text_color = 'white' if abs(point_val) >= 7 else 'black'
            ax_excalibr.text(
                (draw_start + draw_end) / 2, 0.5, f'{count:,}',
                ha='center', va='center',
                fontsize=fontsize_count, color=text_color
            )

    ax_excalibr.set_xlim(x_min, x_max)
    ax_excalibr.set_ylim(0, 1)
    ax_excalibr.set_yticks([])
    ax_excalibr.set_xlabel('Assay Score', fontsize=fontsize_subtitle)
    ax_excalibr.tick_params(axis='x', labelsize=10)
    ax_excalibr.set_title('SNV evidence strengths', loc='left', pad=3, fontsize=fontsize_subtitle)#, style='italic')

    # Build legend - order depends on flipped status (ONLY LEGEND ORDER)
    if flipped:
        legend_order = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        legend_order = [8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]

    legend_handles = []
    for point_val in legend_order:
        if any(pv == point_val for pv, _, _ in intervals_sorted):
            legend_handles.append(
                Patch(facecolor=strength_color[point_val],
                     label=map_point_to_text[point_val],
                     edgecolor='none')
            )

    leg_ax.legend(
        handles=legend_handles,
        loc='upper center',
        ncol=len(legend_handles),
        frameon=False,
        fontsize=fontsize_legend,
        columnspacing=0.8,
        handletextpad=0.6,
        handlelength=1.0,
    )

    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# "Full" style page rendering (extended_data_fig2_pp_utils.py::_plot_single_page)
# ---------------------------------------------------------------------------

def _plot_single_page_full(page_idx, page_datasets, nrows, ncols,
                            point_values_to_plot, linestyles, sample_colors,
                            sample_alphas, sample_name_shortener, strength_color, map_point_to_text,
                            output_dir=None, dataset_tsv=None, precomputed_fits=None,
                            dataset_configs_path=None):
    """
    Plot a single page of datasets (full style).

    Returns
    -------
    tuple: (page_idx, fig)
    """
    import os, logging
    os.environ["MPLLOGLEVEL"] = "WARNING"

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.ERROR)
    logging.getLogger("joblib").setLevel(logging.ERROR)
    logging.getLogger("loky").setLevel(logging.ERROR)

    # Create figure (8.5" x 11" portrait)
    fig = plt.figure(figsize=(8.5, 11))

    # Main grid for datasets
    outer_gs = gridspec.GridSpec(
        nrows, ncols,
        hspace=0.08,
        wspace=0.12,
        left=0.06, right=0.97,
        bottom=0.04, top=0.98
    )

    new_names_dict = _load_new_names_dict()

    for plot_idx, dataset in enumerate(page_datasets):
        row = plot_idx // ncols
        col = plot_idx % ncols

        # Load dataset with updated loading code
        try:
            scoreset, scoreset_2018, indv_summary, fits, score_range_arr, n_c, scoreset_flipped, n_samples = (
                _load_full_style(dataset, output_dir, dataset_tsv, precomputed_fits, dataset_configs_path)
            )
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            continue

        if dataset in new_names_dict:
            dataset = new_names_dict[dataset]
        else:
            print(dataset,"not in new_names_dict")

        # Count non-empty samples
        n_actual_samples = sum(1 for count in scoreset_2018.sample_counts if count > 0)

        # Create sub-grid: fits row, histogram, calibration, spacer, mini-legend
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            5, n_actual_samples,
            subplot_spec=outer_gs[row, col],
            height_ratios=[1.2, 1.2, 0.5, 0.15, 0.15],
            hspace=0.25,
            wspace=0.1
        )

        # Setup. `score_range_arr` is the fit's own LR grid, which can be
        # narrower than the actual displayed data (e.g.
        # CARD11_Meitlis_2020_SGE_Ibrutinib_GoF's highest-scoring P/LP
        # variant sits just outside it) -- widen to also cover every real
        # score so set_xlim below can't silently clip a real variant's bar
        # off-screen.
        x_min = min(score_range_arr[0], float(np.nanmin(scoreset.scores)), float(np.nanmin(scoreset_2018.scores)))
        x_max = max(score_range_arr[-1], float(np.nanmax(scoreset.scores)), float(np.nanmax(scoreset_2018.scores)))
        all_scores = scoreset.snv_scores
        bin_width = _compute_robust_bin_width(scoreset, all_scores, x_min, x_max, n_target_bins=30)
        point_ranges = indv_summary['point_ranges']

        # Pre-compute thresholds
        threshold_info = []
        for idx, point_val in enumerate(point_values_to_plot):
            for pv, score_ranges_pr in point_ranges.items():
                if pv == -point_val:
                    for sr in score_ranges_pr:
                        threshold_score = sr[0] if not scoreset_flipped else sr[1]
                        threshold_info.append((pv, threshold_score, '#2166AC', linestyles[idx], 1.0))
                        break
                    break

            for pv, score_ranges_pr in point_ranges.items():
                if pv == point_val:
                    for sr in score_ranges_pr:
                        threshold_score = sr[1] if not scoreset_flipped else sr[0]
                        threshold_info.append((pv, threshold_score, '#B2182B', linestyles[idx], 1.0))
                        break
                    break

        # === ROW 1: INDIVIDUAL FITS ===
        num_skipped = 0
        for sample_num in range(len(scoreset_2018.sample_counts)):
            if scoreset_2018.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue

            sample_idx = sample_num - num_skipped
            ax = fig.add_subplot(inner_gs[0, sample_idx])

            sample_mask = scoreset_2018.sample_assignments[:, sample_idx]
            sample_name = sample_name_shortener.get(scoreset_2018.sample_names[sample_num],
                                                    scoreset_2018.sample_names[sample_num])
            color = sample_colors[sample_num]

            hist_data = scoreset_2018.scores[sample_mask]
            n_count = sample_mask.sum()

            try:
                sns.histplot(hist_data, binwidth=bin_width, binrange=(x_min, x_max), stat='density', ax=ax,
                       alpha=0.5, color=color)
            except ValueError as e:
                sns.histplot(hist_data, binrange=(x_min, x_max), stat='density', ax=ax,
                       alpha=0.5, color=color)

            max_hist_density = max([patch.get_height() for patch in ax.patches]) if ax.patches else 1.0

            density_sample = sample_density(score_range_arr, fits, sample_idx)
            d_total = np.nansum(density_sample, axis=1)
            d_total_perc = np.percentile(d_total, [5, 50, 95], axis=0)

            ax.fill_between(score_range_arr, d_total_perc[0], d_total_perc[2],
                           color='gray', alpha=0.3)
            ax.plot(score_range_arr, d_total_perc[1], color='black', alpha=0.65, linewidth=1.5)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim([0, max_hist_density * 1.1])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(
                axis='both',
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False
            )

            # Thresholds (only ±1,2,4,8)
            for pv, thresh_score, thresh_color, thresh_ls, thresh_lw in threshold_info:
                if abs(pv) in [1, 2, 4, 8]:
                    ax.axvline(thresh_score, color=thresh_color, linestyle=thresh_ls,
                              linewidth=0.8, alpha=0.7)

            # Compact legend
            face_rgba = to_rgba(color, 0.5)
            hist_patch = Patch(facecolor=face_rgba, edgecolor='black')

            if sample_num == 2:
                legend_label = f'{sample_name}\n({n_count:,})\nprior: {indv_summary["prior"]:.3f}'
            else:
                legend_label = f'{sample_name}\n({n_count:,})'

            if scoreset_flipped:
                loc = 'upper left' if sample_num == 0 else 'upper right'
            else:
                loc = 'upper right' if sample_num == 0 else 'upper left'

            ax.legend([hist_patch], [legend_label], loc=loc, fontsize=6, framealpha=0.9)

            # Title on first panel only
            if sample_idx == 0:
                gene_name = dataset.split('_')[0]
                author_name = dataset.split('_')[1] if len(dataset.split('_')) > 1 else ''
                ax.set_title(dataset.replace("_clinvar_2018", ""),
                           fontsize=9, fontweight='bold', pad=3, loc='left')

        # === ROW 2: COMBINED HISTOGRAM ===
        # Twin y-axes (control variant count / SNV count), same fix and
        # rationale as plot_calibration's "SECOND ROW" above -- a shared
        # 'density' axis lets the much-more-numerous SNV sample rescale the
        # P/LP and B/LB bars down to near-invisible. No labels/ticks on
        # either y-axis (matches the rest of this compact per-dataset page).
        ax_hist = fig.add_subplot(inner_gs[1, :])
        ax_twin = ax_hist.twinx()

        num_skipped = 0
        sample_handles = []

        for sample_num in range(len(scoreset.sample_counts)):
            if scoreset.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue
            if sample_num == 3:
                continue

            sample_idx = sample_num - num_skipped
            sample_mask = scoreset.sample_assignments[:, sample_idx]
            sample_name = sample_name_shortener.get(scoreset.sample_names[sample_num],
                                                   scoreset.sample_names[sample_num])
            color = sample_colors[sample_num]
            alpha = sample_alphas[sample_num]

            if sample_num == 2:
                hist_data = all_scores
                display_name = 'SNVs'
                n_count = len(all_scores)
                target_ax = ax_twin
            else:
                hist_data = scoreset.scores[sample_mask]
                display_name = sample_name
                n_count = sample_mask.sum()
                target_ax = ax_hist

            try:
                sns.histplot(hist_data, binwidth=bin_width, binrange=(x_min, x_max), stat='count', ax=target_ax,
                       alpha=alpha, color=color)
            except ValueError as e:
                sns.histplot(hist_data, binrange=(x_min, x_max), stat='count', ax=target_ax,
                       alpha=alpha, color=color)

            face_rgba = to_rgba(color, alpha)
            hist_patch = Patch(facecolor=face_rgba, edgecolor='black')
            sample_handles.append((hist_patch, f'{display_name} ({n_count:,})'))

        ax_hist.set_xlim(x_min, x_max)
        ax_twin.set_xlim(x_min, x_max)
        ax_hist.set_ylim(0, 1.18 * max([p.get_height() for p in ax_hist.patches]) if ax_hist.patches else 1.0)
        ax_twin.set_ylim(0, 1.18 * max([p.get_height() for p in ax_twin.patches]) if ax_twin.patches else 1.0)
        ax_hist.set_xlabel('')
        ax_hist.set_ylabel('')
        ax_hist.tick_params(
            axis='both',
            bottom=False,
            left=False,
            labelsize=6,
            labelbottom=False,
            labelleft=False
        )
        ax_twin.set_ylabel('')
        ax_twin.tick_params(axis='y', right=False, labelright=False)

        sample_legend_handles = [h[0] for h in sample_handles]
        sample_legend_labels = [h[1] for h in sample_handles]
        ax_hist.legend(sample_legend_handles, sample_legend_labels,
                      loc='upper left', fontsize=6, framealpha=0.8)

        # === ROW 3: CALIBRATION BAR ===
        ax_calib = fig.add_subplot(inner_gs[2, :])

        # Build intervals
        intervals = []
        all_point_values = sorted([pv for pv in point_ranges.keys() if pv != 0])

        # A point value can have more than one disjoint score interval (e.g.
        # DDX3X_Radford_2023's +1) -- every entry in score_ranges_list needs
        # its own interval, not just the first.
        for point_val in all_point_values:
            score_ranges_list = point_ranges[point_val]
            if not score_ranges_list:
                continue

            for range_idx, score_range_tuple in enumerate(score_ranges_list):
                # visualization issue
                if range_idx == 0 and dataset == "PAX6_McDonnell_2024_LE9_geneticin" and point_val == 8:
                    start = x_min
                else:
                    start = score_range_tuple[0]
                end = score_range_tuple[1]
                intervals.append((point_val, start, end))

        # Fill every real gap between evidence intervals -- including at the
        # outer axis edges -- with a "0" (indeterminate) interval. See the
        # identical rationale in plot_calibration above: DDX3X_Radford_2023's
        # +1 has a second range entirely on the opposite side of zero from
        # its first, leaving its own separate indeterminate gap that the old
        # single-gap sign/flipped-based heuristic never filled. Walking the
        # real intervals in score order reproduces that old single-gap
        # result exactly for ordinary (monotonic) datasets.
        intervals_by_start = sorted(intervals, key=lambda x: x[1])
        filled = []
        prev_end = x_min
        for pv, s, e in intervals_by_start:
            if s > prev_end:
                filled.append((0, prev_end, s))
            filled.append((pv, s, e))
            prev_end = max(prev_end, e)
        if prev_end < x_max:
            filled.append((0, prev_end, x_max))
        if not intervals_by_start:
            filled = [(0, x_min, x_max)]
        intervals = filled
        intervals_sorted = sorted(intervals, key=lambda x: x[1])

        # Plot intervals
        for point_val, start, end in intervals_sorted:
            # See the identical fix + rationale in plot_calibration above /
            # analysis/figure4/panels.py plot_panel_b: unbounded outermost
            # intervals (start=-inf or end=inf) make axvspan's width
            # (end - start) evaluate to NaN and render nothing. Clip to the
            # axis's finite range for drawing only; `count` keeps using the
            # true (possibly infinite) bounds.
            draw_start, draw_end = max(start, x_min), min(end, x_max)
            ax_calib.axvspan(draw_start, draw_end, color=strength_color[point_val], alpha=1.0)
            count = ((all_scores >= start) & (all_scores < end)).sum()
            if (draw_end - draw_start) > (x_max - x_min) * 0.06:
                text_color = 'white' if abs(point_val) >= 7 else 'black'
                ax_calib.text((draw_start + draw_end) / 2, 0.5, f'{count:,}',
                            ha='center', va='center', fontsize=7, color=text_color)

        ax_calib.set_xlim(x_min, x_max)
        ax_calib.set_ylim(0, 1)
        ax_calib.set_yticks([])
        ax_calib.set_xlabel('Assay Score' if row == nrows - 1 else '', fontsize=8)
        ax_calib.tick_params(axis='x', labelsize=6)

        # === ROW 4: MINI LEGEND (skip row 3 spacer) ===
        ax_leg = fig.add_subplot(inner_gs[4, :])
        ax_leg.axis('off')

        if scoreset_flipped:
            legend_order = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            legend_order = [8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8]

        legend_handles = []
        for point_val in legend_order:
            if any(pv == point_val for pv, _, _ in intervals_sorted):
                legend_handles.append(
                    Patch(facecolor=strength_color[point_val],
                         label=map_point_to_text[point_val],
                         edgecolor='none')
                )

        ax_leg.legend(
            handles=legend_handles,
            loc='center',
            ncol=len(legend_handles) if len(legend_handles) <= 12 else (len(legend_handles) // 2 + len(legend_handles) % 2),
            frameon=False,
            fontsize=6,
            columnspacing=0.4,
            handletextpad=0.3,
            handlelength=0.7,
        )

    return page_idx, fig


def _plot_multi_calibration_appendix(dataset_list, datasets_per_page=4, ncols=2,
                                     debug_first_page_only=False, n_jobs=-1,
                                     output_dir=None, dataset_tsv=None,
                                     precomputed_fits=None, dataset_configs_path=None):
    """
    Create multi-panel calibration plots for appendix publication (parallelized),
    full style. Each dataset retains full format: fits, histogram, calibration, legend.

    Returns
    -------
    list of figures
        List of matplotlib figures (one per page)
    """

    # Sample colors
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
    sample_alphas = [0.5, 0.5, 0.15, 0.4]

    # Strength colors
    strength_color = {
        -8: '#4b91a6', -7: '#5DA3BD', -6: '#6FAACE', -5: '#74ABCE',
        -4: '#7ab5d1', -3: '#99c8dc', -2: '#d0e8f0', -1: '#e4f1f6',
        0: '#e0e0e0',
        1: '#e6b1b8', 2: '#d68f99', 3: '#ca7682', 4: '#b85c6b',
        5: '#B1535F', 6: '#AA4E58', 7: '#A2484F', 8: '#943744'
    }

    map_point_to_text = {i: f"{i:+d}" if i != 0 else "0" for i in range(-8, 9)}

    sample_name_shortener = {
        "Pathogenic/Likely Pathogenic": "P/LP",
        "Benign/Likely Benign": "B/LB",
        "population": "gnomAD",
        "gnomAD": "gnomAD",
        "Synonymous": "Syn"
    }

    point_values_to_plot = [1, 2, 3, 4, 5, 6, 7, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5)),
                  (0, (5, 5)), (0, (3, 1, 1, 1)), 'solid']

    # Split datasets into pages
    nrows = datasets_per_page // ncols
    n_pages = (len(dataset_list) + datasets_per_page - 1) // datasets_per_page

    # Limit to first page if debugging
    if debug_first_page_only:
        n_pages = 1
        n_jobs = 1
        print("DEBUG MODE: Only plotting first page")

    print(f"Plotting {len(dataset_list)} datasets across {n_pages} pages")

    # Prepare page data
    page_data = []
    for page_idx in range(n_pages):
        start_idx = page_idx * datasets_per_page
        end_idx = min(start_idx + datasets_per_page, len(dataset_list))
        page_datasets = dataset_list[start_idx:end_idx]
        page_data.append((page_idx, page_datasets))

    # Process pages in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_plot_single_page_full)(
            page_idx, page_datasets, nrows, ncols,
            point_values_to_plot, linestyles, sample_colors, sample_alphas,
            sample_name_shortener, strength_color, map_point_to_text,
            output_dir, dataset_tsv, precomputed_fits, dataset_configs_path,
        )
        for page_idx, page_datasets in page_data
    )

    # Sort by page index and extract figures
    results_sorted = sorted(results, key=lambda x: x[0])
    figures = [fig for _, fig in results_sorted]

    return figures


# ---------------------------------------------------------------------------
# "Stacked" style page rendering
# (supp_pdf_creation_calibration_paper_utils.py::_plot_single_page_stacked)
# ---------------------------------------------------------------------------

def _plot_single_page_stacked(page_idx, page_datasets, nrows, ncols,
                               point_values_to_plot, linestyles, linewidths,
                               sample_colors, sample_alphas, short_labels, PLOT_THRESHOLDS,
                               output_dir=None, dataset_tsv=None, precomputed_fits=None,
                               dataset_configs_path=None):
    """
    Plot a single page with stacked samples.

    Returns
    -------
    tuple: (page_idx, fig)
    """

    import os, logging
    os.environ["MPLLOGLEVEL"] = "WARNING"

    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.ERROR)
    logging.getLogger("joblib").setLevel(logging.ERROR)
    logging.getLogger("loky").setLevel(logging.ERROR)

    # Create figure (8.5" x 11" portrait)
    fig = plt.figure(figsize=(8.5, 11))

    # Create 2x2 grid for main panels
    outer_grid = GridSpec(nrows, ncols, figure=fig, hspace=0.18, wspace=0.1,
                         left=0.06, right=0.98, top=0.97, bottom=0.04)

    new_names_dict = _load_new_names_dict()

    for plot_idx, dataset in enumerate(page_datasets):
        panel_row = plot_idx // ncols
        panel_col = plot_idx % ncols

        # Load dataset
        try:
            scoreset, indv_summary, fits, score_range_arr, n_c, scoreset_flipped, n_samples = (
                _load_stacked_style(dataset, output_dir, dataset_tsv, precomputed_fits, dataset_configs_path)
            )
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
            continue

        if dataset in new_names_dict:
            dataset = new_names_dict[dataset]
        else:
            print(dataset,"not in new_names_dict")

        # Count non-empty samples
        n_actual_samples = sum(1 for count in scoreset.sample_counts if count > 0)

        # Create nested grid for stacked samples
        inner_grid = GridSpecFromSubplotSpec(
            n_actual_samples, 1,
            subplot_spec=outer_grid[panel_row, panel_col],
            hspace=0.06
        )

        point_ranges = indv_summary['point_ranges']

        # Plot each sample
        num_skipped = 0
        for sample_num in range(len(scoreset.sample_counts)):
            if scoreset.sample_counts[sample_num] == 0:
                num_skipped += 1
                continue

            sample_idx = sample_num - num_skipped
            ax = fig.add_subplot(inner_grid[sample_idx, 0])

            sample_mask = scoreset.sample_assignments[:, sample_idx]
            sample_name = scoreset.sample_names[sample_num]

            if sample_name == "population":
                sample_name = "gnomAD"

            # Plot histogram
            sns.histplot(
                scoreset.scores[sample_mask],
                stat='density', ax=ax,
                alpha=sample_alphas[sample_num],
                color=sample_colors[sample_num]
            )

            max_hist_density = max([patch.get_height() for patch in ax.patches]) if ax.patches else 1.0

            # Plot fitted density
            density_sample = sample_density(score_range_arr, fits, sample_idx)
            d = np.nansum(density_sample, axis=1)
            d_perc = np.percentile(d, [5, 50, 95], axis=0)

            ax.plot(score_range_arr, d_perc[1], color='black', alpha=0.5, linewidth=1.5)
            ax.fill_between(score_range_arr, d_perc[0], d_perc[2], color='gray', alpha=0.3)

            handles = []
            if PLOT_THRESHOLDS:
                # Add threshold lines
                for idx, point_val in enumerate(point_values_to_plot):
                    # Benign threshold
                    for pv, score_ranges in point_ranges.items():
                        if pv == -point_val and score_ranges:
                            for sr in score_ranges:
                                threshold_score = sr[0] if not scoreset_flipped else sr[1]
                                ax.axvline(
                                    threshold_score, color='b',
                                    linestyle=linestyles[idx],
                                    linewidth=linewidths[idx], alpha=0.7
                                )
                                break
                            break

                    # Pathogenic threshold
                    for pv, score_ranges in point_ranges.items():
                        if pv == point_val and score_ranges:
                            for sr in score_ranges:
                                threshold_score = sr[1] if not scoreset_flipped else sr[0]
                                ax.axvline(
                                    threshold_score, color='r',
                                    linestyle=linestyles[idx],
                                    linewidth=linewidths[idx], alpha=0.7
                                )
                                break
                            break

                    # Create handle for legend
                    if len(point_ranges.get(point_val, [])) != 0 or len(point_ranges.get(-point_val, [])) != 0:
                        h = mlines.Line2D(
                            [], [], color='gray',
                            linestyle=linestyles[idx],
                            linewidth=linewidths[idx],
                            label=f"±{point_val}"
                        )
                        handles.append(h)

            # Title on first sample only
            if sample_idx == 0:
                ax.set_title(dataset.replace("_clinvar_2018", ""),
                           fontsize=8, fontweight='bold', pad=4, loc='left')# if panel_col == 0 else 'right')

            ax.set_ylim([0, max_hist_density * 1.1])

            # X-axis only on last sample
            is_last_sample = (sample_num == len(scoreset.sample_counts) - 1 or
                            (sample_num == len(scoreset.sample_counts) - 2 and
                             scoreset.sample_counts[-1] == 0))

            if is_last_sample and panel_row == nrows - 1:
                ax.set_xlabel("Assay score", fontsize=8)
            elif is_last_sample:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("")
                ax.set_xticks([])

            if panel_col == 0:
                ax.set_ylabel("Density", fontsize=8)
            else:
                ax.set_ylabel("")

            n_count = sample_mask.sum()

            # Create histogram legend handle
            hist_patch = Patch(
                facecolor=sample_colors[sample_num],
                alpha=0.7, edgecolor='none'
            )

            short_name = short_labels.get(sample_name, sample_name)

            if sample_name == "gnomAD":
                hist_label = f'{short_name}\n(n={n_count:,}, prior={indv_summary["prior"]:.3f})'
            else:
                hist_label = f'{short_name}\n(n={n_count:,})'

            # Histogram legend on the left
            hist_legend = ax.legend(
                [hist_patch],
                [hist_label],
                loc='upper left',
                fontsize=7,
                framealpha=0.8
            )

            ax.add_artist(hist_legend)

            # Threshold legend on the right
            if handles:
                ax.legend(
                    handles,
                    [h.get_label() for h in handles],
                    loc='upper right',
                    ncol=2 if len(handles) > 3 else 1,
                    fontsize=6,
                    framealpha=0.5,
                    handlelength=1.5,
                    columnspacing=0.8
                )
            else:
                ax.text(0.98, 0.95, 'No evidence',
                       transform=ax.transAxes,
                       fontsize=7,
                       ha='right', va='top',
                       bbox=dict(boxstyle='square,pad=0.3',
                                facecolor='white',
                                edgecolor='lightgray',
                                alpha=0.5))

            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6)

    return page_idx, fig


def _plot_multi_datasets_stacked_appendix(dataset_list, datasets_per_page=4, ncols=2,
                                          debug_first_page_only=False,
                                          PLOT_THRESHOLDS=True, n_jobs=-1,
                                          output_dir=None, dataset_tsv=None,
                                          precomputed_fits=None, dataset_configs_path=None):
    """
    Create multi-panel plots with stacked samples for appendix publication (parallelized).
    Each dataset shows samples stacked vertically.

    Returns
    -------
    list of figures
        List of matplotlib figures (one per page)
    """

    # Sample colors
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']
    sample_alphas = [0.7, 0.7, 0.7, 0.7]

    # Threshold configuration
    point_values_to_plot = [1, 2, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3))]
    linewidths = [1.0, 1.0, 1.0, 1.0]

    short_labels = {
        "Pathogenic/Likely Pathogenic": "P/LP",
        "Benign/Likely Benign": "B/LB",
        "population": "gnomAD",
        "gnomAD": "gnomAD",
        "Synonymous": "Synonymous"
    }

    # Split datasets into pages
    nrows = datasets_per_page // ncols
    n_pages = (len(dataset_list) + datasets_per_page - 1) // datasets_per_page

    # Limit to first page if debugging
    if debug_first_page_only:
        n_pages = 1
        n_jobs = 1
        print("DEBUG MODE: Only plotting first page")

    print(f"Plotting {len(dataset_list)} datasets across {n_pages} pages")

    # Prepare page data
    page_data = []
    for page_idx in range(n_pages):
        start_idx = page_idx * datasets_per_page
        end_idx = min(start_idx + datasets_per_page, len(dataset_list))
        page_datasets = dataset_list[start_idx:end_idx]
        page_data.append((page_idx, page_datasets))

    # Process pages in parallel
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_plot_single_page_stacked)(
            page_idx, page_datasets, nrows, ncols,
            point_values_to_plot, linestyles, linewidths,
            sample_colors, sample_alphas, short_labels, PLOT_THRESHOLDS,
            output_dir, dataset_tsv, precomputed_fits, dataset_configs_path,
        )
        for page_idx, page_datasets in page_data
    )

    # Sort by page index and extract figures
    results_sorted = sorted(results, key=lambda x: x[0])
    figures = [fig for _, fig in results_sorted]

    return figures


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def build_appendix_pdf(
    dataset_list: Optional[List[str]] = None,
    output_path=None,
    plot_thresholds: bool = True,
    datasets_per_page: int = 6,
    ncols: int = 2,
    debug_first_page_only: bool = False,
    n_jobs: int = -1,
    output_dir: Optional[str] = None,
    dataset_tsv: Optional[str] = None,
    precomputed_fits: Optional[str] = None,
    dataset_configs_path: Optional[str] = None,
):
    """Build the extended-data appendix PDF, one page per ``datasets_per_page``
    datasets, saved as a single multi-page PDF at ``output_path``.

    Parameters
    ----------
    dataset_list : list of str, optional
        Datasets to include. If None, a default list is built depending on
        ``plot_thresholds``:
          - True  -> :func:`_default_dataset_list_full` (all datasets
            discovered under ``output_dir``/``analysis.config.OUTPUT_DIR``,
            minus "_clinvar_2018"-suffixed entries, minus the optional
            ``datasets_to_exclude.pkl`` list, minus TP53/F9-prefixed
            datasets) — matches extended_data_fig2_pp.py's default.
          - False -> :func:`_default_dataset_list_configs` (every dataset in
            the ``--dataset-configs`` JSON) — matches
            supp_pdf_creation_calibration_paper.py's default.
    output_path : str or Path, optional
        Where to write the combined PDF. Defaults to
        ``Path(analysis.config.FIGURE_DIR) / "extended_data_appendix.pdf"``.
    plot_thresholds : bool
        Selects which of the two original scripts' rendering style to use:
        True -> extended_data_fig2_pp.py's "full" style (fits row + combined
        histogram + calibration-strength bar + legend; thresholds always
        drawn). False -> supp_pdf_creation_calibration_paper.py's "stacked"
        style (one histogram+fit panel per sample; ``plot_thresholds``
        additionally toggles whether threshold lines are drawn within that
        style, matching the original ``PLOT_THRESHOLDS`` flag).
    datasets_per_page, ncols, debug_first_page_only, n_jobs :
        Passed straight through to the page-rendering / parallelization
        helpers (same names/defaults as the original scripts).
    output_dir, dataset_tsv, precomputed_fits, dataset_configs_path :
        Forwarded to ``analysis.legacy_fits.load_scoreset_and_fits`` (and
        dataset-list discovery); default to the matching ``analysis.config``
        values when None.

    Returns
    -------
    Path
        The path the PDF was written to.
    """
    output_path = Path(output_path) if output_path is not None else Path(cfg.FIGURE_DIR) / "extended_data_appendix.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_list is None:
        if plot_thresholds:
            dataset_list = _default_dataset_list_full(output_dir)
        else:
            dataset_list = _default_dataset_list_configs(dataset_configs_path)

    if plot_thresholds:
        figures = _plot_multi_calibration_appendix(
            dataset_list,
            datasets_per_page=datasets_per_page,
            ncols=ncols,
            debug_first_page_only=debug_first_page_only,
            n_jobs=n_jobs,
            output_dir=output_dir,
            dataset_tsv=dataset_tsv,
            precomputed_fits=precomputed_fits,
            dataset_configs_path=dataset_configs_path,
        )
    else:
        figures = _plot_multi_datasets_stacked_appendix(
            dataset_list,
            datasets_per_page=datasets_per_page,
            ncols=ncols,
            debug_first_page_only=debug_first_page_only,
            PLOT_THRESHOLDS=plot_thresholds,
            n_jobs=n_jobs,
            output_dir=output_dir,
            dataset_tsv=dataset_tsv,
            precomputed_fits=precomputed_fits,
            dataset_configs_path=dataset_configs_path,
        )

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, dpi=300)
            if is_notebook():
                plt.show()
            else:
                plt.close(fig)

    print(f"Wrote {len(figures)} page(s) to {output_path}")
    return output_path


def build_extended_data_fig2(output_path=None, n_jobs: int = 14, debug_first_page_only: bool = False):
    """Reproduce extended_data_fig2_pp.py's driver: full style
    (plot_thresholds=True), default (auto-discovered) dataset list,
    datasets_per_page=6.
    """
    return build_appendix_pdf(
        dataset_list=None,
        output_path=output_path or (Path(cfg.FIGURE_DIR) / "extended_data_fig2.pdf"),
        plot_thresholds=True,
        datasets_per_page=6,
        debug_first_page_only=debug_first_page_only,
        n_jobs=n_jobs,
    )


def build_supp_pdf(output_path=None, n_jobs: int = 10, debug_first_page_only: bool = False):
    """Reproduce supp_pdf_creation_calibration_paper.py's driver: stacked
    style (plot_thresholds=False), explicit dataset list = every key in the
    ``--dataset-configs`` JSON, datasets_per_page=6.
    """
    dataset_list = _default_dataset_list_configs()
    return build_appendix_pdf(
        dataset_list=dataset_list,
        output_path=output_path or (Path(cfg.FIGURE_DIR) / "supp_pdf_calibration_paper.pdf"),
        plot_thresholds=False,
        datasets_per_page=6,
        debug_first_page_only=debug_first_page_only,
        n_jobs=n_jobs,
    )
