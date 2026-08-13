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
  - The legacy script's RAD51D/XRCC2/BARD1 "extra plots" (produced alongside
    Figure 4 but not part of it) live in `analysis.extra_gene_fits` instead,
    called from `analyze_pipeline_output.py` -- kept out of this module so a
    reproduction of just Figure 4 doesn't also need them (they in turn depend
    on a `fit_hist_snv_plot` module that only exists as an import statement in
    the legacy script; see that module's own TODO).

No plotting code (colors, figsize, GridSpec ratios, titles, linestyles) has
been altered anywhere below relative to the legacy script — only data
sourcing and control flow (functions instead of top-level `# In[N]:` cells).

Running standalone
------------------
Figure 4 does *not* require running `analyze_pipeline_output.py` or any other
part of `analysis/` first -- `build_figure4()` rebuilds everything it needs
(confusion matrices, the MSH2 mixture fit, panels e/f's REVEL data) from disk
on its own the moment it's called; the only reason `analyze_pipeline_output.py`
passes it `danzs_oob`/`auths_oob`/`dataset_names` is to avoid redoing that
work a second time in the same notebook run, not because it's required.

    cd exCALIBR && python -m analysis.figure4.driver --help

Every one of Figure 4's five on-disk inputs has its own CLI flag (see --help),
so a caller passing all five needs nothing from `analysis/config.py` at all --
`analysis.config`'s `EXCALIBR_*` environment variables (each documented there
with what it is and its on-disk default) are consulted only as the fallback
for whichever flags are left unset:

    --output-dir        EXCALIBR_OUTPUT_DIR       run_igvf_batch.py output tree (danz/auth confusion, MSH2 scoreset)
    --dataset-tsv       EXCALIBR_DATASET_TSV      master integrated-variant-effect TSV
    --dataset-configs   EXCALIBR_DATASET_CONFIGS  dataset -> (n_c, benign_method) JSON
    --precomputed-fits  EXCALIBR_PRECOMPUTED_FITS gzipped bootstrap fits (MSH2 mixture overlay, panel a)
    --revel-dir         EXCALIBR_YILE_DIR         external REVEL gene-specific calibration files (panels e/f only)
    --figure-dir        EXCALIBR_FIGURE_DIR       where fig4_PP.png / Figure4.pdf get written (default: OUTPUT_DIR/figures)

Missing `--revel-dir`/`EXCALIBR_YILE_DIR` or MSH2 calibration data prints a
warning and skips (see build_figure4's docstring) rather than raising, so a
partial pipeline output directory won't crash this.
"""
from __future__ import annotations

import argparse
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
MSH2_PIPELINE_KEY = f"{MSH2_DATASET}_clinvar_2018"


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
    msh2_pipeline_key = MSH2_PIPELINE_KEY
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
        clinvar_release="2025",
        n_c=n_c,
    )
    return scoreset_2018, scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped


def _load_msh2_calibration_from_scoresets(
    output_dir, dataset_configs_path, precomputed_fits, scoreset_2018_path, scoreset_path,
):
    """Same return shape as `_load_msh2_calibration`, but builds the two
    Scoreset objects from `analysis.figure4.scoreset_io`'s small prebuilt
    arrays instead of the master TSV + pipeline dataframe-to-Scoreset
    construction -- so `dataset_tsv` isn't needed at all in this path (a
    minimal reproduction bundle can then skip shipping the ~54K-row MSH2
    slice of the master TSV entirely). Only preserves what
    `analysis.figure4.panels` actually reads off a Scoreset -- see
    `analysis.figure4.scoreset_io`'s docstring.

    Duplicates `analysis.legacy_fits.load_scoreset_and_fits`'s calibration/LR
    path-resolution and precomputed-fits lookup (rather than reusing that
    function directly) since that function's Scoreset-building half is
    unconditional -- there's no way to call it and skip needing
    `dataset_tsv`.
    """
    from analysis.legacy_fits import resolve_component_for, _load_calibration_and_lr
    from src.assay_calibration.pipeline.visualize import load_precomputed_fits
    from analysis.figure4.scoreset_io import load_scoreset_bundle

    n_c, benign_method = resolve_component_for(MSH2_PIPELINE_KEY, str(output_dir), dataset_configs_path)
    comp = f"{n_c}_{benign_method}"
    ds_output_dir = Path(output_dir) / MSH2_PIPELINE_KEY
    if not ds_output_dir.exists():
        ds_output_dir = Path(output_dir)  # flat layout fallback

    calib_path = ds_output_dir / f"{MSH2_PIPELINE_KEY}_{comp}_calibration.json"
    lr_path = ds_output_dir / f"{MSH2_PIPELINE_KEY}_{comp}_lr_values.json.gz"
    if not calib_path.exists():
        # older naming convention -- bare n_c, no benign_method suffix
        calib_path = ds_output_dir / f"{MSH2_PIPELINE_KEY}_{n_c}_calibration.json"
        lr_path = ds_output_dir / f"{MSH2_PIPELINE_KEY}_{n_c}_lr_values.json.gz"
    if not calib_path.exists() or not lr_path.exists():
        raise FileNotFoundError(
            f"Calibration/LR files not found for {MSH2_PIPELINE_KEY} ({comp}) under {ds_output_dir}"
        )

    indv_summary = _load_calibration_and_lr(calib_path, lr_path)
    score_range = np.asarray(indv_summary["score_range"])
    scoreset_flipped = bool(indv_summary.get("scoreset_flipped", False))

    bootstrap_results = load_precomputed_fits(precomputed_fits, MSH2_PIPELINE_KEY)
    fits = [
        seed_results[n_c] for seed_results in bootstrap_results.values()
        if isinstance(seed_results, dict) and seed_results.get(n_c) is not None
    ]
    if not fits:
        raise KeyError(f"No '{n_c}' fits found for '{MSH2_PIPELINE_KEY}' in {precomputed_fits}")

    scoreset_2018 = load_scoreset_bundle(scoreset_2018_path)
    scoreset = load_scoreset_bundle(scoreset_path)
    n_samples = int((scoreset_2018.sample_counts > 0).sum())

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

def _revel_path(revel_dir: str, filename: str) -> Path:
    """Resolve `revel_dir/filename`, preferring a `.gz`-compressed copy if
    that's what's on disk (e.g. a space-trimmed bundle from
    `export_msh2_bundle.py`) over the plain file -- pandas' read_csv/
    read_table infer gzip compression from the `.gz` extension automatically,
    so callers just get back whichever path actually exists."""
    plain = Path(revel_dir) / filename
    gzipped = Path(revel_dir) / f"{filename}.gz"
    if not plain.exists() and gzipped.exists():
        return gzipped
    return plain


def _get_stack_bar_plot_data(gene: str, dist: str, revel_dir: str):
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
    threshdf = pd.read_csv(
        _revel_path(revel_dir, f"{dist}_gene_specific_calibration_thresholds_20260118.csv"),
        index_col=0,
    ).drop("calibration_model", axis=1)
    sorted_thresholds = threshdf.loc[gene]

    labfn = _revel_path(revel_dir, f"{gene}_{dist}_labeled.txt")
    labdat = pd.read_table(labfn, header=None)
    # Files may carry leading descriptive columns (variant name, short mutation
    # name, ...) before the score/label pair that plot_panel_e actually needs;
    # panel_e indexes those as columns 0/1, so always take the last two columns.
    if labdat.shape[1] > 2:
        labdat = labdat.iloc[:, [-2, -1]]
        labdat.columns = [0, 1]

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
    scores_path = _revel_path(revel_dir, f"{gene}_{dist}_scores.tsv")
    if scores_path.exists():
        if dist == 'MP2':
            snvdf = pd.read_table(scores_path, index_col=0)
        else:
            snvdf = pd.read_table(scores_path, header=None)
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
    else:
        snvdf = pd.DataFrame(columns=required_columns)

    oldsorted_thresholds = pd.Series(thresh_old, index=[x for x in categories if x != 'IR'])

    return gene, dist, labdat, snvdf, sorted_thresholds, oldsorted_thresholds


def _build_panel_ef_data(revel_dir: str):
    gene_4e, dist_4e, labdat_4e, snvdf_4e, sorted_thresholds_4e, oldsorted_thresholds_4e = (
        _get_stack_bar_plot_data('MSH2', 'REVEL', revel_dir)
    )

    dist_4f = 'REVEL'
    heatmap_fn = _revel_path(revel_dir, f"{dist_4f}_heatmap_data_pillar.csv")
    finalout_4f = pd.read_csv(heatmap_fn)

    return (
        gene_4e, dist_4e, labdat_4e, snvdf_4e, sorted_thresholds_4e, oldsorted_thresholds_4e,
        dist_4f, finalout_4f,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def build_figure4(
    output_dir=None,
    figure_dir=None,
    dataset_tsv=None,
    precomputed_fits=None,
    dataset_configs_path=None,
    revel_dir=None,
    panel_c_data=None,
    scoreset_2018_data=None,
    scoreset_data=None,
    danzs_oob=None,
    auths_oob=None,
    dataset_names=None,
    vus_pct_danz=None,
    vus_pct_auth=None,
):
    """Build Figure 4 (unified a-f panel figure) and save to figure_dir.

    `output_dir`/`figure_dir`/`dataset_tsv`/`precomputed_fits`/
    `dataset_configs_path`/`revel_dir` are each independently optional --
    any left as None falls back to the matching `analysis.config` constant
    (itself overridable via the matching `EXCALIBR_*` env var, see this
    module's "Running standalone" docstring section above). Passing them
    explicitly here take priority over both.

    Pass `danzs_oob`/`auths_oob`/`dataset_names` (e.g. straight from the
    notebook's own section-3 `conf_by_method[method]` / `auth_by_method[method]`
    / `datasets`) to reuse confusion matrices already built earlier in the
    same run, instead of re-discovering pipeline output, re-loading every
    variants CSV, and rebuilding author labels from scratch a second time.
    If any of the three is omitted, they're all rebuilt fresh from
    `output_dir` (same as before) -- which, for panel c, means walking and
    rebuilding *every* dataset under `output_dir`, not just MSH2, since panel
    c's matrices are an aggregate across the whole pipeline run, not an
    MSH2-specific quantity.

    `scoreset_2018_data`/`scoreset_data`, if BOTH given, are paths written by
    `analysis.figure4.export_msh2_bundle` / `analysis.figure4.scoreset_io.
    save_scoreset_bundle`: the two small (.scores/.sample_assignments/
    .snv_scores) arrays panels a/b/e actually read off a Scoreset, cached
    once instead of re-deriving them from the full master TSV + pipeline
    dataframe-to-Scoreset construction (splice filtering, ClinVar-release
    parsing, ...) -- meaning `dataset_tsv` isn't needed at all when both are
    given, on top of `panel_c_data` already making `output_dir` not need
    every other dataset. If only one is given, or either path doesn't exist,
    falls back to the normal `dataset_tsv`-based reconstruction for both
    (silently, since a bundle without these is just an older/plainer one,
    not an error).

    `panel_c_data`, if given, is a path written by
    `analysis.figure4.export_msh2_bundle` / `analysis.figure4.panel_c_io.
    save_panel_c_bundle`: the panel c aggregate matrices computed *once* from
    the full pipeline output and cached to a small JSON file, so a minimal
    MSH2-only reproduction bundle (a trimmed dataset_tsv/precomputed_fits/
    dataset_configs plus just MSH2's calibration/LR files and REVEL files)
    doesn't also need to ship (or re-walk) the entire multi-dataset pipeline
    output tree just for this one panel. Takes priority over
    `danzs_oob`/`auths_oob`/`dataset_names`/`vus_pct_danz`/`vus_pct_auth` if
    both are given.

    `vus_pct_danz`/`vus_pct_auth`, if given, are pooled VUS-determinate
    percentages (analysis.confusion.build_vus_coverage /
    build_author_vus_coverage + _aggregate_coverage_pct, over the same
    dataset/method scope as danzs_oob/auths_oob) shown in panel c's
    "Determinate: Controls X%, VUS Y%" caption. Omitted (not faked) when not
    supplied -- panel c previously hardcoded these to 79.7%/93.2% regardless
    of what was actually loaded.

    Renders only Figure 4 itself -- the RAD51D/XRCC2/BARD1 extra fit plots
    that the legacy script also produced alongside Figure 4 are unrelated
    supplementary output, not part of Figure 4, and live in
    `analysis.extra_gene_fits.build_extra_gene_fits` (called from
    `analyze_pipeline_output.py`) instead, so this module can be handed to
    someone reproducing just Figure 4 without also needing that.

    No-op safe to call repeatedly; guards missing external data (YILE_DIR)
    and missing/incomplete pipeline output by printing a warning and skipping
    rather than raising.
    """
    output_dir = output_dir or cfg.OUTPUT_DIR
    figure_dir = Path(figure_dir or cfg.FIGURE_DIR)
    figure_dir.mkdir(parents=True, exist_ok=True)

    dataset_tsv = dataset_tsv or cfg.DATASET_TSV
    precomputed_fits = precomputed_fits or cfg.PRECOMPUTED_FITS
    dataset_configs_path = dataset_configs_path or cfg.DATASET_CONFIGS
    revel_dir = revel_dir or cfg.YILE_DIR

    print("Building Figure 4...")

    use_scoreset_bundle = (
        scoreset_2018_data is not None and scoreset_data is not None
        and Path(scoreset_2018_data).exists() and Path(scoreset_data).exists()
    )
    try:
        if use_scoreset_bundle:
            (
                scoreset_2018, scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped,
            ) = _load_msh2_calibration_from_scoresets(
                output_dir, dataset_configs_path, precomputed_fits, scoreset_2018_data, scoreset_data,
            )
        else:
            (
                scoreset_2018, scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped,
            ) = _load_msh2_calibration(output_dir, dataset_tsv, precomputed_fits, dataset_configs_path)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"  SKIP Figure 4: could not load MSH2 calibration data ({e})")
        return

    if panel_c_data is not None:
        from analysis.figure4.panel_c_io import load_panel_c_bundle
        danzs_oob, auths_oob, dataset_names, vus_pct_danz, vus_pct_auth = load_panel_c_bundle(panel_c_data)
    elif danzs_oob is None or auths_oob is None or dataset_names is None:
        danzs_oob, auths_oob, dataset_names = _build_confusion_matrices(
            output_dir, dataset_tsv, dataset_configs_path
        )
    if not danzs_oob:
        print("  SKIP Figure 4: no confusion matrices could be built from pipeline output")
        return

    prior, Post_p, Post_b, p_data_sim, b_data_sim = _build_panel_d_data()

    if cfg.warn_if_missing(revel_dir, "Figure 4 panels e/f (REVEL gene-specific calibration files)"):
        return

    try:
        (
            gene_4e, dist_4e, labdat_4e, snvdf_4e, sorted_thresholds_4e, oldsorted_thresholds_4e,
            dist_4f, finalout_4f,
        ) = _build_panel_ef_data(revel_dir)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"  SKIP Figure 4: could not load panel e/f data from YILE_DIR ({e})")
        return

    fig = plot_figure4_unified(
        scoreset_2018, scoreset, indv_summary, fits, score_range, n_c, n_samples, scoreset_flipped,
        danzs_oob, auths_oob, dataset_names,
        prior, Post_p, Post_b, p_data_sim, b_data_sim,
        gene_4e, dist_4e, labdat_4e, snvdf_4e, sorted_thresholds_4e, oldsorted_thresholds_4e,
        dist_4f, finalout_4f,
        vus_pct_danz=vus_pct_danz, vus_pct_auth=vus_pct_auth,
    )
    fig.savefig(figure_dir / "fig4_PP.png", dpi=300, bbox_inches='tight')
    fig.savefig(figure_dir / "Figure4.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved: {figure_dir / 'Figure4.pdf'}")
    if is_notebook():
        plt.show()
    else:
        plt.close(fig)


def _main():
    parser = argparse.ArgumentParser(
        description="Build Figure 4 standalone -- every input below is optional; "
                    "anything not passed here falls back to analysis.config (itself "
                    "overridable via the matching EXCALIBR_* env var). Nothing is "
                    "required from analysis.config unless you omit the flag for it.",
    )
    parser.add_argument("--bundle", default=None,
                         help="Convenience shortcut for a bundle directory written by "
                              "analysis.figure4.export_msh2_bundle: fills in --output-dir, "
                              "--dataset-tsv, --dataset-configs, --precomputed-fits, --revel-dir, "
                              "--panel-c-data, --scoreset-2018-data, and --scoreset-data from "
                              "BUNDLE/{calibration,dataset.tsv.gz,dataset_configs.json,"
                              "bootstrap_fits.json.gz,revel,panel_c.json,scoreset_2018.csv.gz,"
                              "scoreset_2025.csv.gz} -- any of those flags passed explicitly still "
                              "takes priority over the bundle's own path for that one input, and "
                              "--dataset-tsv is simply unused if the scoreset files are both present.")
    parser.add_argument("--output-dir", default=None,
                         help="run_igvf_batch.py output tree. "
                              "Falls back to EXCALIBR_OUTPUT_DIR / analysis.config.OUTPUT_DIR.")
    parser.add_argument("--dataset-tsv", default=None,
                         help="Master integrated-variant-effect TSV (the --dataset input to "
                              "run_igvf_batch.py). Falls back to EXCALIBR_DATASET_TSV / "
                              "analysis.config.DATASET_TSV.")
    parser.add_argument("--dataset-configs", default=None,
                         help="Dataset -> (n_c, benign_method) JSON (the --dataset-configs input "
                              "to run_igvf_batch.py). Falls back to EXCALIBR_DATASET_CONFIGS / "
                              "analysis.config.DATASET_CONFIGS.")
    parser.add_argument("--precomputed-fits", default=None,
                         help="Gzipped bootstrap fits JSON (the --precomputed-fits input to "
                              "run_igvf_batch.py; needed for the MSH2 mixture overlay in panel a). "
                              "Falls back to EXCALIBR_PRECOMPUTED_FITS / analysis.config.PRECOMPUTED_FITS.")
    parser.add_argument("--revel-dir", default=None,
                         help="External REVEL gene-specific calibration files, not produced by this "
                              "pipeline (panels e/f only; skipped with a warning if missing). "
                              "Falls back to EXCALIBR_YILE_DIR / analysis.config.YILE_DIR.")
    parser.add_argument("--figure-dir", default=None,
                         help="Where fig4_PP.png / Figure4.pdf get written. "
                              "Falls back to EXCALIBR_FIGURE_DIR / analysis.config.FIGURE_DIR.")
    parser.add_argument("--panel-c-data", default=None,
                         help="Precomputed panel c (ExCALIBR vs. author confusion) aggregate JSON, "
                              "written by analysis.figure4.export_msh2_bundle / "
                              "analysis.figure4.panel_c_io.save_panel_c_bundle. If given, skips "
                              "walking --output-dir for every dataset just to rebuild panel c -- "
                              "the one piece of Figure 4 that isn't MSH2-specific, needed for a "
                              "minimal MSH2-only reproduction bundle.")
    parser.add_argument("--scoreset-2018-data", default=None,
                         help="Precomputed 2018-ClinVar-release MSH2 scoreset, written by "
                              "analysis.figure4.export_msh2_bundle / "
                              "analysis.figure4.scoreset_io.save_scoreset_bundle. If given together "
                              "with --scoreset-data, skips needing --dataset-tsv at all.")
    parser.add_argument("--scoreset-data", default=None,
                         help="Precomputed current-ClinVar-release MSH2 scoreset -- see "
                              "--scoreset-2018-data.")
    args = parser.parse_args()

    bundle = Path(args.bundle) if args.bundle else None
    build_figure4(
        output_dir=args.output_dir or (str(bundle / "calibration") if bundle else None),
        figure_dir=args.figure_dir,
        dataset_tsv=args.dataset_tsv or (str(bundle / "dataset.tsv.gz") if bundle else None),
        precomputed_fits=args.precomputed_fits or (str(bundle / "bootstrap_fits.json.gz") if bundle else None),
        dataset_configs_path=args.dataset_configs or (str(bundle / "dataset_configs.json") if bundle else None),
        revel_dir=args.revel_dir or (str(bundle / "revel") if bundle else None),
        panel_c_data=args.panel_c_data or (str(bundle / "panel_c.json") if bundle else None),
        scoreset_2018_data=args.scoreset_2018_data or (str(bundle / "scoreset_2018.csv.gz") if bundle else None),
        scoreset_data=args.scoreset_data or (str(bundle / "scoreset_2025.csv.gz") if bundle else None),
    )


if __name__ == "__main__":
    _main()
