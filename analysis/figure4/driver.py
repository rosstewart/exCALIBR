"""
Figure 4 driver — adapted from
`test/auxiliary_fig_creation/pillar_project_figure4.py`.

Assembles the data for each of Figure 4's six panels (a-f) and calls
`analysis.figure4.panels.plot_figure4_unified` (moved verbatim from
`test/auxiliary_fig_creation/fig4_pp.py`) to render the unified figure.

Differences from the legacy notebook script:
  - No `sys.path.append("..")` / no top-level script execution — everything
    is wrapped in functions, so `import analysis.figure4.driver` alone has no
    side effects (no file I/O, no plotting).
  - `load_dataset_for_plot(...)` (legacy pickle-backed loader) is replaced by
    `analysis.legacy_fits.load_scoreset_and_fits(...)`, matching the pattern
    already used for the MSH2 example in `analysis/analyze_pipeline_output.py`.
  - The legacy `excalibr_confusion_mat.pkl` / `auth_confusion_mat.pkl` /
    `datasets_confusion_mat.pkl` caches (panel 4c) are replaced by building
    confusion matrices fresh from `analysis.discovery` / `analysis.confusion`
    / `analysis.author_labels`, run against `analysis.config.OUTPUT_DIR`.
  - Panels 4e/4f depend on externally-precomputed REVEL gene-specific
    calibration files (`analysis.config.YILE_DIR`, not produced by this
    pipeline) — guarded with `analysis.config.warn_if_missing` so a missing
    directory skips Figure 4 with a warning instead of crashing.
  - The RAD51D/XRCC2/BARD1 "extra plots" section imports a `fit_hist_snv_plot`
    module that only exists as an import statement in the legacy script — no
    defining file was found anywhere in this repo (see TODO below). That
    section is left as a documented no-op that prints a warning rather than
    a fabricated reimplementation.

No plotting code (colors, figsize, GridSpec ratios, titles, linestyles) has
been altered anywhere below relative to the legacy script — only data
sourcing and control flow (functions instead of top-level `# In[N]:` cells).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import config as cfg
from analysis.author_labels import attach_author_labels
from analysis.confusion import build_author_confusion_matrix, build_confusion_matrix
from analysis.discovery import discover_outputs, load_all_variants
from analysis.legacy_fits import load_scoreset_and_fits
from analysis.figure4.panels import plot_figure4_unified
from analysis.plot_common import is_notebook

MSH2_DATASET = "MSH2_Jia_2021"


# ---------------------------------------------------------------------------
# 4a/4b — MSH2 experimental-score calibration (2018-fit mixture + full scoreset)
# ---------------------------------------------------------------------------

def _load_msh2_calibration(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path):
    """Load the MSH2 mixture fit (2018 ClinVar) + full scoreset for panels 4a/4b.

    Replaces the legacy pair of
    ``load_dataset_for_plot(dataset+"_clinvar_2018", ...)`` +
    hand-rolled ``load_scoreset(df, dataset, clinvar_release=..., for_fit=...)``
    calls with `analysis.legacy_fits.load_scoreset_and_fits`, matching the
    pattern already used for this exact dataset in
    `analysis/analyze_pipeline_output.py`.
    """
    msh2_pipeline_key = f"{MSH2_DATASET}_clinvar_2018"
    scoreset_2018, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped = load_scoreset_and_fits(
        MSH2_DATASET,
        output_dir=output_dir,
        dataset_tsv=dataset_tsv,
        precomputed_fits=precomputed_fits,
        dataset_configs_path=dataset_configs_path,
        pipeline_dataset=msh2_pipeline_key,
        clinvar_release="2018",
    )
    scoreset, _, _, _, _, _, _ = load_scoreset_and_fits(
        MSH2_DATASET,
        output_dir=output_dir,
        dataset_tsv=dataset_tsv,
        precomputed_fits=precomputed_fits,
        dataset_configs_path=dataset_configs_path,
        pipeline_dataset=msh2_pipeline_key,
        clinvar_release="2026",
        n_c=n_c,
    )
    return scoreset_2018, scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped


# ---------------------------------------------------------------------------
# 4c — confusion matrices, built fresh from pipeline output
# ---------------------------------------------------------------------------

def _build_confusion_matrices(output_dir, dataset_tsv, dataset_configs_path):
    """Build (danzs_oob, auths_oob, dataset_names) fresh from pipeline output.

    Replaces the legacy
    ``/data/ross/assay_calibration/flagship/{excalibr,auth,datasets}_confusion_mat.pkl``
    caches. Variable names (danzs_oob / auths_oob / dataset_names) match what
    the legacy script unpickled these pkl files into (danzs_oob, auths_oob,
    datasets) so downstream plotting needs no further edits.
    """
    tree, model_selections, calibrations = discover_outputs(Path(output_dir))
    if not tree:
        return [], [], []

    dataset_configs = None
    if dataset_configs_path and Path(dataset_configs_path).exists():
        with open(dataset_configs_path) as f:
            dataset_configs = json.load(f)

    df = load_all_variants(
        tree=tree,
        model_selections=model_selections,
        dataset_configs=dataset_configs,
        methods_filter=None,
        datasets_filter=None,
        calibrations=calibrations,
        min_controls=0,
    )
    if df.empty:
        return [], [], []

    df = attach_author_labels(df, dataset_tsv)

    methods = sorted(df["method"].unique())
    dataset_names = sorted(df["dataset"].unique())
    primary_method = methods[0]
    df_m = df[df["method"] == primary_method]

    danzs_oob = []
    auths_oob = []
    for dataset in dataset_names:
        df_ds = df_m[df_m["dataset"] == dataset]
        danzs_oob.append(build_confusion_matrix(df_ds, use_oob=True, label=f"{dataset}/{primary_method}") if not df_ds.empty else None)
        auths_oob.append(build_author_confusion_matrix(df_ds, use_oob=True) if not df_ds.empty else None)

    return danzs_oob, auths_oob, dataset_names


# ---------------------------------------------------------------------------
# 4d — simulation-based single-gene predictor calibration cartoon
#
# Pure simulation, no external data dependency — ported verbatim.
# ---------------------------------------------------------------------------

def _posterior_from_lr(prior, LR_pos):
    pre_odds = prior / (1 - prior)
    post_odds = pre_odds * LR_pos
    return post_odds / (1 + post_odds)


def _build_panel_d_data():
    from src.assay_calibration.fit_utils.fit import thresholds_from_prior

    np.random.seed(42)
    p_data_sim = np.random.beta(8, 2, size=1000)   # Pathogenic (skewed high)
    b_data_sim = np.random.beta(2, 6, size=1000)   # Benign (skewed low)

    prior = 0.265
    Post_p, Post_b, c = thresholds_from_prior(prior, [1, 2, 3, 4, 5, 6, 7, 8])

    Post_p = Post_p[::-1]
    Post_b = Post_b[::-1]

    Post_p = np.concatenate([Post_p[:1], Post_p[-4:]])
    Post_b = np.concatenate([Post_b[:1], Post_b[-4:]])

    Post_p = _posterior_from_lr(prior, Post_p)
    Post_b = 1 - _posterior_from_lr(prior, Post_b)

    return prior, Post_p, Post_b, p_data_sim, b_data_sim


# ---------------------------------------------------------------------------
# 4e/4f — REVEL gene-specific vs. gene-aggregate calibration comparison
#
# Depends on externally-precomputed files under analysis.config.YILE_DIR
# (Yile's REVEL threshold/score files) — not produced by this pipeline.
# ---------------------------------------------------------------------------

def _get_stack_bar_plot_data(gene: str, dist: str, yile_dir: str):
    """Load the data `get_StackBarPlot(gene, dist)` needed for panel 4e.

    Ported from the legacy driver's ``get_StackBarPlot`` — only the data
    loading is kept (the legacy function also built + immediately closed an
    internal `fig`/`ax` for its own diagnostic purposes; that stray plot was
    never part of any saved figure, so it is dropped here as dead code, not
    as a restyle of anything rendered).
    """
    categories = [
        "BP4_Very Strong", "BP4_Strong", "BP4_Moderate+", "BP4_Moderate", "BP4_Supporting",
        "IR", "PP3_Supporting", "PP3_Moderate", "PP3_Moderate+", "PP3_Strong", "PP3_Very Strong"
    ]
    path = yile_dir
    threshdf = pd.read_csv(f"{path}/{dist}_gene_specific_calibration_thresholds_20260118.csv", index_col=0).drop("calibration_model", axis=1)
    sorted_thresholds = threshdf.loc[gene]

    labfn = f"{path}/{gene}_{dist}_labeled.txt"
    labdat = pd.read_table(labfn, header=None)

    if dist == 'AM':
        thresh_old = [np.nan, np.nan, 0.07, 0.099, 0.169, 0.792, 0.906, 0.972, 0.99, np.nan]
        scrcol = 8
    if dist == 'MP2':
        thresh_old = [np.nan, 0.01, 0.031, 0.197, 0.391, 0.737, 0.829, 0.895, 0.932, np.nan]
        scrcol = 'MP2'
    if dist == 'REVEL':
        thresh_old = [np.nan, 0.016, 0.052, 0.183, 0.29, 0.644, 0.773, 0.879, 0.932, np.nan]
        scrcol = 7

    thresh_old = pd.Series(thresh_old, index=[x for x in categories if x != 'IR'])

    required_columns = [dist, 'merg_clinvar_sig', 'GeneSymbol']
    try:
        if dist == 'MP2':
            snvdf = pd.read_table(f"{path}/{gene}_{dist}_scores.tsv", index_col=0)
        else:
            snvdf = pd.read_table(f"{path}/{gene}_{dist}_scores.tsv", header=None)
        if not snvdf.empty:
            if dist == 'MP2':
                snvdf = snvdf.drop_duplicates()[[scrcol]].copy()
            else:
                snvdf = snvdf.drop_duplicates().iloc[:, [scrcol]].copy()
            snvdf.columns = [dist]
            snvdf['merg_clinvar_sig'] = 'allSNVs'
            snvdf['GeneSymbol'] = gene
        else:
            snvdf = pd.DataFrame(columns=required_columns)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        snvdf = pd.DataFrame(columns=required_columns)

    oldsorted_thresholds = pd.Series(thresh_old, index=[x for x in categories if x != 'IR'])

    return gene, dist, labdat, snvdf, sorted_thresholds, oldsorted_thresholds


def _build_panel_ef_data(yile_dir: str):
    gene_4e, dist_4e, labdat_4e, snvdf_4e, sorted_thresholds_4e, oldsorted_thresholds_4e = (
        _get_stack_bar_plot_data('MSH2', 'REVEL', yile_dir)
    )

    dist_4f = 'REVEL'
    heatmap_fn = f"{yile_dir}/{dist_4f}_heatmap_data_pillar.csv"
    finalout_4f = pd.read_csv(heatmap_fn)

    return (
        gene_4e, dist_4e, labdat_4e, snvdf_4e, sorted_thresholds_4e, oldsorted_thresholds_4e,
        dist_4f, finalout_4f,
    )


# ---------------------------------------------------------------------------
# RAD51D / XRCC2 / BARD1 extra fit plots
#
# TODO: fit_hist_snv_plot module not found in repo, skipping. Only the import
# statement `from fit_hist_snv_plot import plot_figure_panel_a, plot_figure_panel_b`
# exists in the legacy script (test/auxiliary_fig_creation/pillar_project_figure4.py);
# no file defining `fit_hist_snv_plot` was found anywhere in this repo. Rather
# than fabricate an implementation, this section prints a warning and returns.
# ---------------------------------------------------------------------------

def _build_extra_gene_fits(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path, figure_dir):
    try:
        import fit_hist_snv_plot  # noqa: F401
    except ImportError:
        print(
            "  SKIP RAD51D/XRCC2/BARD1 extra fit plots: 'fit_hist_snv_plot' module "
            "not found in repo (only its import statement exists in the legacy "
            "script test/auxiliary_fig_creation/pillar_project_figure4.py) — "
            "TODO: port/locate this module if these plots are needed."
        )
        return

    # If fit_hist_snv_plot is ever added to the repo, wire it up here following
    # the same load_scoreset_and_fits(dataset) pattern as _load_msh2_calibration,
    # for datasets RAD51D_unpublished / XRCC2_unpublished / BARD1_unpublished,
    # then call fit_hist_snv_plot.plot_figure_panel_a / plot_figure_panel_b and
    # save to figure_dir / {"xrcc2","rad51d","bard1"}_{fits,snv}.png.
    from fit_hist_snv_plot import plot_figure_panel_a, plot_figure_panel_b  # noqa: F401

    for dataset, tag in [
        ("RAD51D_unpublished", "rad51d"),
        ("XRCC2_unpublished", "xrcc2"),
        ("BARD1_unpublished", "bard1"),
    ]:
        try:
            scoreset, indv_summary, fits, score_range, n_c, n_samples, flipped = load_scoreset_and_fits(
                dataset, output_dir=output_dir, dataset_tsv=dataset_tsv,
                precomputed_fits=precomputed_fits, dataset_configs_path=dataset_configs_path,
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"  SKIP extra fit plot for {dataset}: {e}")
            continue

        minimal = dataset == "BARD1_unpublished"
        fig_a = plot_figure_panel_a(
            scoreset, indv_summary, fits, score_range, flipped, n_samples,
            layout='vertical', minimal=minimal, figsize=(6.6, 7.3),
        )
        fig_b = plot_figure_panel_b(
            scoreset, indv_summary, score_range, flipped,
            use_twin_axes=True, minimal=minimal, figsize=(6.6, 7.3),
        )
        fig_a.savefig(figure_dir / f"{tag}_fits.png", dpi=300, bbox_inches='tight')
        fig_b.savefig(figure_dir / f"{tag}_snv.png", dpi=300, bbox_inches='tight')
        plt.close(fig_a)
        plt.close(fig_b)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def build_figure4(
    output_dir=None,
    figure_dir=None,
    danzs_oob=None,
    auths_oob=None,
    dataset_names=None,
):
    """Build Figure 4 (unified a-f panel figure) and save to figure_dir.

    Pass `danzs_oob`/`auths_oob`/`dataset_names` (e.g. straight from the
    notebook's own section-3 `conf_by_method[method]` / `auth_by_method[method]`
    / `datasets`) to reuse confusion matrices already built earlier in the
    same run, instead of re-discovering pipeline output, re-loading every
    variants CSV, and rebuilding author labels from scratch a second time.
    If any of the three is omitted, they're all rebuilt fresh from
    `output_dir` (same as before).

    Also attempts the RAD51D/XRCC2/BARD1 extra fit plots (skipped with a
    warning — see `_build_extra_gene_fits` — since `fit_hist_snv_plot` isn't
    present in this repo).

    No-op safe to call repeatedly; guards missing external data (YILE_DIR)
    and missing/incomplete pipeline output by printing a warning and skipping
    rather than raising.
    """
    output_dir = output_dir or cfg.OUTPUT_DIR
    figure_dir = Path(figure_dir or cfg.FIGURE_DIR)
    figure_dir.mkdir(parents=True, exist_ok=True)

    dataset_tsv = cfg.DATASET_TSV
    precomputed_fits = cfg.PRECOMPUTED_FITS
    dataset_configs_path = cfg.DATASET_CONFIGS

    print("Building Figure 4...")

    try:
        (
            scoreset_2018, scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped,
        ) = _load_msh2_calibration(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  SKIP Figure 4: could not load MSH2 calibration data ({e})")
        return

    if danzs_oob is None or auths_oob is None or dataset_names is None:
        danzs_oob, auths_oob, dataset_names = _build_confusion_matrices(
            output_dir, dataset_tsv, dataset_configs_path
        )
    if not danzs_oob:
        print("  SKIP Figure 4: no confusion matrices could be built from pipeline output")
        return

    prior, Post_p, Post_b, p_data_sim, b_data_sim = _build_panel_d_data()

    if cfg.warn_if_missing(cfg.YILE_DIR, "Figure 4 panels e/f (REVEL gene-specific calibration files)"):
        return

    try:
        (
            gene_4e, dist_4e, labdat_4e, snvdf_4e, sorted_thresholds_4e, oldsorted_thresholds_4e,
            dist_4f, finalout_4f,
        ) = _build_panel_ef_data(cfg.YILE_DIR)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"  SKIP Figure 4: could not load panel e/f data from YILE_DIR ({e})")
        return

    fig = plot_figure4_unified(
        scoreset_2018, scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped,
        danzs_oob, auths_oob, dataset_names,
        prior, Post_p, Post_b, p_data_sim, b_data_sim,
        gene_4e, dist_4e, labdat_4e, snvdf_4e, sorted_thresholds_4e, oldsorted_thresholds_4e,
        dist_4f, finalout_4f,
    )
    fig.savefig(figure_dir / "fig4_PP.png", dpi=300, bbox_inches='tight')
    fig.savefig(figure_dir / "Figure4.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved: {figure_dir / 'Figure4.pdf'}")
    if is_notebook():
        plt.show()
    else:
        plt.close(fig)

    _build_extra_gene_fits(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path, figure_dir)
