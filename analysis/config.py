"""
Single place to point the analysis package at data on disk.

Every path here can be overridden by editing this file directly, or by
setting the matching environment variable before import (e.g.
``EXCALIBR_OUTPUT_DIR=/some/other/dir``). This is the only file that should
need editing to point the whole `analysis/` package (and the
`analyze_pipeline_output.py` notebook) at a different run.
"""
import os
from pathlib import Path

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Core pipeline output (run_pipeline.py / run_igvf_batch.py)
# ---------------------------------------------------------------------------

# Master output directory: per-dataset subdirs containing *_variants.csv,
# *_calibration.json, *_lr_values.json.gz written by run_igvf_batch.py.
OUTPUT_DIR = _env(
    "EXCALIBR_OUTPUT_DIR",
    "/data/ross/assay_calibration/exc_pp_clinvar2025_calib",
)

# Master integrated variant-effect dataframe (all datasets, one row per
# variant measurement) — the --dataset input to run_igvf_batch.py.
DATASET_TSV = _env(
    "EXCALIBR_DATASET_TSV",
    "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz",
)

# Dataset -> (n_c, benign_method[, overrides]) config JSON, same file passed
# to run_igvf_batch.py --dataset-configs. Used to resolve which component/
# benign-method combo is "the" calibration for a dataset (see
# analysis.discovery.resolve_component / legacy_fits.resolve_component_for).
DATASET_CONFIGS = _env(
    "EXCALIBR_DATASET_CONFIGS",
    "/home/rcstewart/exCALIBR/src/igvf_configs/dataset_configs_aug_2026.json",
)

# Train/val splits for --oob evidence (run_igvf_batch.py --splits-file),
# matching the current master TSV. Stale versions are archived under
# /data/ross/assay_calibration/archive/stale_splits_20261006/.
SPLITS_FILE = _env(
    "EXCALIBR_SPLITS_FILE", "/data/ross/assay_calibration/dataset_splits_recovered_20260917.pkl"
)

# Precomputed bootstrap fits (gzipped JSON of full per-bootstrap component
# params/weights) — the --precomputed-fits input to run_igvf_batch.py.
# Needed only for plots that overlay the fitted mixture density (MSH2 example,
# cartoon/schematic figures, Figure 4 panels) rather than just the LR+ curve.
PRECOMPUTED_FITS = _env(
    "EXCALIBR_PRECOMPUTED_FITS",
    "/data/ross/assay_calibration/pp_final_clinvar2025_bootstraps/final_clinvar2025_bootstrap_results.json.gz",
)

# Output layout (all real files under OUTPUT_DIR; ~/exCALIBR/manuscript/ holds
# gitignored symlinks to these, and ExCALIBR_latex_src/Figures -> figures/):
#   exploratory_figures/     every figure the notebook draws (working tree)
#   figures/                 Overleaf `Figures/`: only what main.tex/supplement.tex
#                            include (copied from exploratory_figures/ at the end
#                            of each run), plus generated/ (macros.tex, tables)
#   tables/                  LaTeX table files and IGVF data-release CSVs
#   supplementary_data/      Supplementary Data 1-3
#   submission/              flattened .tex (no personal macros) for Nature Genetics
#   pillar_project_figures/  IGVF Pillar Project outputs (other paper)
FIGURE_DIR = _env("EXCALIBR_FIGURE_DIR", str(Path(OUTPUT_DIR) / "exploratory_figures"))
MANUSCRIPT_FIGURE_DIR = _env("EXCALIBR_MANUSCRIPT_FIGURE_DIR", str(Path(OUTPUT_DIR) / "figures"))
TABLES_DIR = _env("EXCALIBR_TABLES_DIR", str(Path(OUTPUT_DIR) / "tables"))
SUPP_DATA_DIR = _env("EXCALIBR_SUPP_DATA_DIR", str(Path(OUTPUT_DIR) / "supplementary_data"))
SUBMISSION_DIR = _env("EXCALIBR_SUBMISSION_DIR", str(Path(OUTPUT_DIR) / "submission"))
ZENODO_DATA_DIR = _env("EXCALIBR_ZENODO_DATA_DIR", str(Path(OUTPUT_DIR) / "zenodo_data"))  # bootstrap fits + splits (analysis/data_deposit.py)
PILLAR_FIGURE_DIR = _env("EXCALIBR_PILLAR_FIGURE_DIR", str(Path(OUTPUT_DIR) / "pillar_project_figures"))

# Dataset name mapping CSV (Old_names, New_names columns) — optional, used to
# translate legacy reported names to current CSV dataset names.
DATASET_NAMES_CSV = _env(
    "EXCALIBR_DATASET_NAMES_CSV",
    str(Path(DATASET_TSV).parent / "new_dataset_names.csv"),
)


# ---------------------------------------------------------------------------
# External, non-pipeline comparison data.
#
# These feed test/auxiliary_fig_creation-derived figures (Figure 4's
# REVEL/AM/MutPred2 panels, the gene-performance odds-ratio scatter) that
# compare ExCALIBR against other predictors/estimates computed by a *separate*
# analysis, not by run_pipeline.py. They are NOT re-derived here — just
# pointed at. Any notebook cell/function that needs one of these prints a
# warning and skips (rather than raising) if the path doesn't exist, since
# they're expected to be filled in/regenerated independently of the pipeline
# refactor.
# ---------------------------------------------------------------------------

YILE_DIR = _env("EXCALIBR_YILE_DIR", "/data/ross/assay_calibration/flagship/yile")

# All of Us biobank odds-ratio-by-evidence-strength estimates (long format:
# Grouping/Calibration x Consequence x Group/Range per Dataset/Predictor).
# Replaces the old, separately-maintained OR_ESTIMATES_CSV + YURIY_OR_CSV
# pair -- see analysis.gene_performance_scatter.load_aou_or_estimates.
AOU_OR_ESTIMATES_TSV = _env(
    "EXCALIBR_AOU_OR_ESTIMATES_TSV",
    "/data/ross/assay_calibration/paper/AoU-OR-estimates_2026-10-02.tsv.gz",
)
# TP53 All of Us estimates, delivered separately (same columns; one
# Dataset/Predictor per TP53 assay under Grouping/Calibration == "ExCALIBR",
# which the main file lacks). Its rows replace any rows for the same
# Dataset/Predictor in AOU_OR_ESTIMATES_TSV.
AOU_TP53_OR_ESTIMATES_TSV = _env(
    "EXCALIBR_AOU_TP53_OR_ESTIMATES_TSV",
    "/data/ross/assay_calibration/paper/excalibr-TP53-OR-estimates_2026-10-05.tsv.gz",
)

# Brnich et al. OddsPath evidence-code CSV (columns: Dataset, "Evidence Code
# Abnormal", "Evidence Code Normal") — a separate statistical analysis, not
# produced by run_pipeline.py. Used only for the "author" side of
# analysis.assay_stats.plot_evidence_comparison; the ExCALIBR side is always
# computed fresh from pipeline output regardless of whether this file exists.
OP_EVIDENCE_CODES_CSV = _env(
    "EXCALIBR_OP_EVIDENCE_CODES_CSV",
    "/data/ross/assay_calibration/dataframe/OddsPath_calibrations.csv",
)
# OddsPath CSV rows that supersede another row for the same assay (updated
# OddsPath calibration): {superseded row: superseding row}. The superseding
# row's values are used under the superseded (calibrated-dataset) name.
OP_SUPERSEDED_BY = {"MSH2_Jia_2021": "MSH2_Scott_2022"}

EVIDENCE_COUNTS_DIR = _env(
    "EXCALIBR_EVIDENCE_COUNTS_DIR",
    "/data/ross/assay_calibration/paper/datasets_reached_evidence",
)

# Dataset description/measurement metadata (used by analysis.gene_table).
DATASET_DESCRIPTIONS_CSV = _env(
    "EXCALIBR_DATASET_DESCRIPTIONS_CSV",
    "/data/ross/assay_calibration/dataset_descriptions.csv",
)
DATASET_MEASUREMENTS_CSV = _env(
    "EXCALIBR_DATASET_MEASUREMENTS_CSV",
    "/data/ross/assay_calibration/dataset_measurements.csv",
)
# Same file as CURATED_DATASET_METADATA_CSV below (current dataset names,
# built from ASSAY_CURATION_CSV by analysis/build_curated_metadata.py), so
# Fig. 6 / the VAMP-seq & SGE numbers and Supplementary Data 1 share one
# metadata source.
ASSAY_METHOD_MAP_CSV = _env(
    "EXCALIBR_ASSAY_METHOD_MAP_CSV",
    str(Path(__file__).resolve().parent / "manuscript_curated" / "dataset_metadata.csv"),
)

# ClinGen Gene-Disease Validity curations (mode-of-inheritance per gene) --
# used by the AD-vs-AR performance-comparison subsection in
# analyze_pipeline_output.py. Raw export has 3 metadata rows + a "+++...+++"
# separator row before the real header (skiprows=4), plus another
# "+++...+++" row right after the header that must be filtered out of the
# data itself.
CLINGEN_MOI_CSV = _env(
    "EXCALIBR_CLINGEN_MOI_CSV",
    "/data/ross/assay_calibration/dataframe/Clingen-Gene-Disease-Summary-2026-04-16.csv",
)

# ---------------------------------------------------------------------------
# Comparison methods (analysis.comparison_methods) -- other calibration
# approaches ExCALIBR gets compared against, beyond the author's own calls.
# ---------------------------------------------------------------------------

# acmgscaler R package source checkout (github.com/badonyi/acmgscaler) --
# base-R only, no install needed; analysis.comparison_methods sources the
# R/*.R files directly from here rather than requiring `install.packages`.
ACMGSCALER_DIR = _env("EXCALIBR_ACMGSCALER_DIR", "/data/tools/acmgscaler")

# Where analysis/run_acmgscaler_all.py writes its per-dataset
# {dataset}_acmgscaler_variants.csv + visualization.png (see
# analysis.comparison_methods.load_acmgscaler_variants) -- run that script
# once, point this at its --output-dir, and the notebook loads results
# instead of calling Rscript again on every run. None until you've run it.
ACMGSCALER_OUTPUT_DIR = _env("EXCALIBR_ACMGSCALER_OUTPUT_DIR", "/data/ross/assay_calibration/acmgscaler_out")

# Simple 2-component GMM baseline -- a full ExCALIBR-pipeline-shaped output
# tree (same {dataset}/{dataset}_{comp}_variants.csv/calibration.json/
# lr_values.json.gz/visualization.png layout as run_igvf_batch.py's own
# output, just with "comp" = "plp_blb" (P/LP + B/LB pool), "plp_blb_synon"
# (P/LP + [B/LB union Synonymous] pool), or the "all_"-prefixed counterpart of
# either (same pooling for the mixing weights, but the GMM's component
# parameters are fit on every variant's score in the dataset, not just the
# labeled pool) instead of an (n_c, benign_method) token -- see
# analysis.comparison_methods.load_comparison_variants and
# GMM_BASELINE_VARIANTS below. Assumes prior=0.1 for all variants.
GMM_BASELINE_OUTPUT_DIR = _env(
    "EXCALIBR_GMM_BASELINE_OUTPUT_DIR", "/data/ross/assay_calibration/simple_gmm_baseline"
)
GMM_BASELINE_VARIANTS = ("plp_blb", "plp_blb_synon", "all_plp_blb", "all_plp_blb_synon")

# Older bare-JSON output (hpc/simple_gmm_baseline.py run standalone,
# without the wrapper that produces the full GMM_BASELINE_OUTPUT_DIR tree
# above) -- per_sample_weights only, no point_ranges/variants.csv. Kept as a
# fallback loader (analysis.comparison_methods.load_gmm_baseline_points) for
# that leaner format; prefer GMM_BASELINE_OUTPUT_DIR when it's populated.
GMM_BASELINE_JSON = _env("EXCALIBR_GMM_BASELINE_JSON", None)

# "Skew-zeroed" ExCALIBR -- the canonical pipeline rerun with
# force_gaussian=True (results to come). Once populated, this is a normal
# ExCALIBR output_dir (same (n_c, benign_method) comp naming as
# EXCALIBR_OUTPUT_DIR) -- load it with analysis.discovery like any other
# pipeline run, not with load_comparison_variants.
FORCE_GAUSSIAN_OUTPUT_DIR = _env("EXCALIBR_FORCE_GAUSSIAN_OUTPUT_DIR", None)

# Robustness analysis (analysis.robustness) -- ExCALIBR calibration
# sensitivity to control-count downsampling / label discordance, compared
# against a fixed reference (unperturbed) dataset. See analysis/robustness.py
# module docstring for the on-disk condition-directory naming convention
# ({base_dataset}_ds{N}_s{seed} / {base_dataset}_disc{pct:.2f}_s{seed}).
ROBUSTNESS_OUTPUT_DIR = _env(
    "EXCALIBR_ROBUSTNESS_OUTPUT_DIR", "/data/ross/assay_calibration/robustness/calib"
)

# Full per-bootstrap component fits (component_params/weights, ~1000
# bootstraps x every condition, same {condition_dirname: {seed: {"2c"/"3c":
# fit}}} format as PRECOMPUTED_FITS above, keyed by the condition directory
# name itself e.g. "BRCA2_Sahu_2025_SGE_ds16_s0") -- produced by the same
# hpc/aggregate_results.py run that made the ROBUSTNESS_OUTPUT_DIR tree.
# Needed for the density-overlay row of plot_robustness_config_summary.
ROBUSTNESS_BOOTSTRAP_RESULTS = _env(
    "EXCALIBR_ROBUSTNESS_BOOTSTRAP_RESULTS",
    "/data/ross/assay_calibration/robustness/bootstrap_results.json.gz",
)

# "Skew-locked" ExCALIBR -- canonical pipeline rerun with each component's
# skew parameter fixed (not freely fit). Same ExCALIBR-shaped output tree as
# OUTPUT_DIR above (per-dataset {dataset}_{comp}_variants.csv/
# calibration.json/lr_values.json.gz), so it's discovered/loaded exactly like
# a normal pipeline run via analysis.discovery -- not analysis.comparison_methods.
SKEW_LOCKED_OUTPUT_DIR = _env(
    "EXCALIBR_SKEW_LOCKED_OUTPUT_DIR", "/data/ross/assay_calibration/skew_locked_gmm/calib"
)

# SpliceAI-threshold / VEP-splice-filter ablation sweep (analysis.
# splice_ablation) -- one subdirectory per condition under this root, each a
# full ExCALIBR-shaped output tree (same layout as OUTPUT_DIR/
# SKEW_LOCKED_OUTPUT_DIR, discovered via analysis.discovery). Condition
# subdirectory names: "thresh_{0.1..0.9}" (VEP splice-consequence filter ON,
# SpliceAI threshold = that value) and "keep_all" (VEP filter OFF, SpliceAI
# threshold disabled) -- see analysis/build_splice_ablation_jobs.py, which
# produces this tree via `hpc/prepare.py pillar_project
# --spliceai-threshold/--disable-vep-splice-filter`.
SPLICE_ABLATION_ROOT = _env(
    "EXCALIBR_SPLICE_ABLATION_ROOT", "/data/ross/assay_calibration/spliceAIthresh"
)

# "Manual-prior" ExCALIBR -- canonical pipeline rerun with the per-dataset
# prior manually fixed at 0.1 instead of ExCALIBR's normal auto-fit prior.
# Same ExCALIBR-shaped output tree as OUTPUT_DIR/SKEW_LOCKED_OUTPUT_DIR
# above, so it's discovered/loaded exactly like a normal pipeline run via
# analysis.discovery, not analysis.comparison_methods. Used for the 3-way
# ExCALIBR / acmgscaler / ExCALIBR(prior=0.1) confusion grid in section 3a2
# of analyze_pipeline_output.py. TEMP PLACEHOLDER PATH -- this run doesn't
# exist on disk yet; update once it's been produced (or set
# EXCALIBR_MANUAL_PRIOR_OUTPUT_DIR).
MANUAL_PRIOR_OUTPUT_DIR = _env(
    "EXCALIBR_MANUAL_PRIOR_OUTPUT_DIR", "/data/ross/assay_calibration/exc_pp_clinvar2025_calib_prior10p"
)

# Full per-bootstrap component fits for the skew-locked run (same
# {dataset: {seed: {"2c"/"3c": {"fit": ..., "val_ll": float}}}} format as
# PRECOMPUTED_FITS above) -- used only to compare each dataset's selected
# component's per-bootstrap validation log-likelihood (val_ll) against the
# regular (unlocked) run's own val_ll for the same bootstrap seed.
SKEW_LOCKED_BOOTSTRAP_RESULTS = _env(
    "EXCALIBR_SKEW_LOCKED_BOOTSTRAP_RESULTS",
    "/data/ross/assay_calibration/skew_locked_gmm/bootstrap_results.json.gz",
)

# Pathomechanism-aware ExCALIBR rerun ("canonical" vs "pathomech_boundary"
# comparison -- see Supplementary Section sec:pathomechanism_prior). Same
# ExCALIBR-shaped output tree as OUTPUT_DIR/SKEW_LOCKED_OUTPUT_DIR, so it's
# discovered/loaded exactly like a normal pipeline run via analysis.discovery.
# Used by analyze_pipeline_output.py's pathomechanism comparison section,
# which reuses the loading logic from
# test/plot_canonical_vs_pathomech_boundary_confusion.py. TEMP PLACEHOLDER
# PATH -- this run is still in progress as of 2026-08-12 (only one dataset
# present on disk so far); update once it's complete, or set
# EXCALIBR_PATHOMECHANISM_OUTPUT_DIR. The previously-generated pathomech
# figures (now under FIGURE_DIR/clinvar_comparisons/pathomechanism/) came
# from a different, since-renamed output tree
# (exc_pp_clinvar2025_calib_pathomech_boundary_oldKCNQ4_DDX3X) and will look
# stale/inconsistent with a fresh run against this directory.
PATHOMECHANISM_OUTPUT_DIR = _env(
    "EXCALIBR_PATHOMECHANISM_OUTPUT_DIR",
    "/data/ross/assay_calibration/exc_pp_clinvar2025_calib_pathomech",
)

# Reconstructed excalibr_datasets.csv -- calibration ranges + sample counts +
# metadata + per-dataset Yang-distance goodness-of-fit (yang_dist_plp/blb/
# gnomad/synonymous columns), built by
# analysis/run_build_excalibr_datasets_table.py --with-yang. Yang distance
# itself is slow (~1-3 min/dataset at full bootstrap resolution -- see
# analysis/yang_distance.py), so the notebook reads this precomputed table
# rather than recomputing it live; rerun that script (with --with-yang) to
# refresh it if pipeline output changes.
EXCALIBR_DATASETS_TABLE_CSV = _env(
    "EXCALIBR_DATASETS_TABLE_CSV", "/data/ross/assay_calibration/dataframe/excalibr_datasets_2026-10-05.csv"
)

# Bootstrap-count-reduction experiment (tests/benchmark_bootstrap_reduction.py)
# -- how much does reducing the number of bootstrap fits (e.g. 1000 -> 20)
# degrade the final calibration? Output tree is
# {dir}/{dataset}/level_{N}/{dataset}_{n_c}_{benign}_calibration.json (+
# _lr_values.json.gz), one calibration per (dataset, N) -- no repeated seeds
# per level, unlike ROBUSTNESS_OUTPUT_DIR above (see
# analysis.robustness.plot_bootstrap_reduction_config_summary's docstring
# for why that's fine: each level already carries its own free bootstrap-
# resampling [p5,p50,p95] uncertainty).
BOOTSTRAP_REDUCTION_OUTPUT_DIR = _env(
    "EXCALIBR_BOOTSTRAP_REDUCTION_OUTPUT_DIR",
    "/data/ross/assay_calibration/hyperparam_sweep_oldclinvar/bootstrap_reduction",
)

# Fit-number (restart-count) comparison experiment
# (tests/benchmark_num_fits_dataframe.py) -- how much does reducing the
# number of EM restarts per fit degrade the best-of-N training
# log-likelihood? Path to that script's summary.csv (columns: dataset, n_c,
# num_fits, baseline, mean_best, std_best, delta, n_valid, n_failed).
FIT_NUMBER_COMPARISON_SUMMARY_CSV = _env(
    "EXCALIBR_FIT_NUMBER_COMPARISON_SUMMARY_CSV",
    "/data/ross/assay_calibration/hyperparam_sweep_oldclinvar/benchmark_num_fits_dataframe/summary.csv",
)
# Sibling file (same directory) keyed "{dataset}|{n_c}c" -> list of every
# valid restart's own train_ll -- used by analysis.robustness.compute_delta_std_column
# to turn summary.csv's raw "delta" (train_ll units, not comparable across
# datasets) into a dimensionless number of restart-to-restart SDs.
FIT_NUMBER_COMPARISON_TRAIN_LLS_JSON = _env(
    "EXCALIBR_FIT_NUMBER_COMPARISON_TRAIN_LLS_JSON",
    "/data/ross/assay_calibration/hyperparam_sweep_oldclinvar/benchmark_num_fits_dataframe/train_lls.json",
)


# Pre-merged predictor+functional evidence CSV (AlphaMissense/MutPred2/REVEL
# calibration scores/points, both domain_aggregate and single_gene scope,
# already joined against an OLDER variant-aggregated functional evidence
# table) -- analysis.all_variant_evidence.build_predictor_only_table strips
# this down to a predictor-only table (drops the stale functional-evidence
# columns and any row with no predictor score at all), meant to be re-joined
# against analysis.all_variant_evidence.build_variant_aggregate_table's own
# (freshly computed) output later.
PREDICTOR_EVIDENCE_SOURCE_CSV = _env(
    "EXCALIBR_PREDICTOR_EVIDENCE_SOURCE_CSV",
    "/data/ross/assay_calibration/IGVFFI1443TQDN.csv.gz",
)


# ---------------------------------------------------------------------------
# Manuscript build (analyze_pipeline_output.py -> Figures/generated/)
# ---------------------------------------------------------------------------

# Nature Genetics manuscript sources; the registry scans these for macro uses.
MANUSCRIPT_TEX_DIR = _env(
    "EXCALIBR_MANUSCRIPT_TEX_DIR", str(Path(__file__).resolve().parent.parent / "ExCALIBR_latex_src")
)

# Fit-quality cutoff for the normalized Yang distance (main text "below 0.2").
YANG_GOOD_FIT_THRESHOLD = 0.2

# Per-dataset cache of every bootstrap's Yang distance (shared by the batch
# table, the fit-quality numbers and Extended Data Fig. 1), and the number of
# parallel workers used when a dataset has to be (re)computed.
YANG_CACHE_DIR = _env("EXCALIBR_YANG_CACHE_DIR", str(Path(OUTPUT_DIR).parent / "yang_distance_cache"))
YANG_N_JOBS = int(_env("EXCALIBR_YANG_N_JOBS", "32"))
# CDF-grid resolution for every Yang distance in the manuscript (table, fit
# numbers, ED Fig. 1); 2000 is ~4-5x faster than 10000 for <0.1% change in
# medians (see analysis/run_build_excalibr_datasets_table.py). Reported in Methods.
YANG_N_GRID = int(_env("EXCALIBR_YANG_N_GRID", "2000"))
# Held-out (out-of-bag) Yang distances: computed on each bootstrap's held-out
# variants (python -m analysis.skew_locked_yang --split val, which also writes
# the per-dataset held-out sample sizes) and reported only for samples with
# at least YANG_HELDOUT_MIN_N held-out variants.
YANG_HELDOUT_CSV = _env("EXCALIBR_YANG_HELDOUT_CSV", str(Path(SKEW_LOCKED_BOOTSTRAP_RESULTS).parent
                                                          / "yang_skew_locked_vs_regular_val.csv"))
YANG_HELDOUT_MIN_N = int(_env("EXCALIBR_YANG_HELDOUT_MIN_N", "5"))

# Hand-curated per-dataset metadata for Supplementary Data 1 (citation, PMID,
# description, assay type, model system, disease, VAMP-seq/SGE, IGVF-produced,
# Woo-style post-processed intervals), keyed by current pipeline dataset name.
# Cache for the slow Part A computation steps (analysis/step_cache.py).
STEP_CACHE_DIR = _env("EXCALIBR_STEP_CACHE_DIR", "/data/ross/assay_calibration/manuscript_step_cache")

# IGVF logo marking IGVF-produced datasets in Fig. 6 (point heatmap).
IGVF_LOGO_PNG = _env("EXCALIBR_IGVF_LOGO_PNG", "/data/ross/assay_calibration/paper/igvf_logo.png")

# Assay curation spreadsheet (one row per assay; column meanings in the
# definitions file). analysis/build_curated_metadata.py turns it into
# CURATED_DATASET_METADATA_CSV below. Older metadata files (dataset
# descriptions, excalibr_datasets_oldnames.csv, var_effect_measurements_
# dataset.csv) are archived under
# /data/ross/assay_calibration/archive/assay_metadata_20261006/.
ASSAY_CURATION_CSV = _env(
    "EXCALIBR_ASSAY_CURATION_CSV",
    "/data/ross/assay_calibration/dataframe/pillar_project_assay_curation.csv",
)
ASSAY_CURATION_COLUMN_DEFINITIONS_CSV = _env(
    "EXCALIBR_ASSAY_CURATION_COLUMN_DEFINITIONS_CSV",
    "/data/ross/assay_calibration/dataframe/pillar_project_assay_curation_column_definitions.csv",
)
CURATED_DATASET_METADATA_CSV = _env(
    "EXCALIBR_CURATED_DATASET_METADATA_CSV",
    str(Path(__file__).resolve().parent / "manuscript_curated" / "dataset_metadata.csv"),
)

# Review aid: render every generated number that differs from the
# pre-automation manuscript (or is new) in red. Turn off once reviewed.
HIGHLIGHT_CHANGED_NUMBERS = _env("EXCALIBR_HIGHLIGHT_CHANGED_NUMBERS", "1") == "1"

# Explicit, reasoned exemptions from the manuscript build's staleness checks:
# {input path: reason}. Every waiver is copied into build_report.md.
STALENESS_WAIVERS: dict = {}

# Run every notebook part (A manuscript, B Pillar Project, C IGVF release,
# D diagnostics) by default; MANUSCRIPT_ONLY=1 skips B-D for a faster build.
MANUSCRIPT_ONLY = os.environ.get("MANUSCRIPT_ONLY", "0") == "1"
STRICT = os.environ.get("EXCALIBR_STRICT", "0") == "1"


def require(path: str, what: str) -> bool:
    """Like `warn_if_missing`, but inverted (True = present) and raising in
    strict mode (EXCALIBR_STRICT=1), so a headless manuscript build fails
    loudly instead of silently skipping a figure/table."""
    if not path or not Path(path).exists():
        msg = f"{what}: path not found ({path})"
        if STRICT and str(path) not in STALENESS_WAIVERS:
            raise FileNotFoundError(msg)
        print(f"  SKIP {msg}")
        return False
    return True


def warn_if_missing(path: str, what: str) -> bool:
    """Print a warning and return True if `path` does not exist on disk.

    Used to guard cells/functions that depend on external, non-pipeline data
    (see module docstring) so a missing file skips that figure instead of
    crashing the whole notebook run.
    """
    if not path or not Path(path).exists():
        print(f"  SKIP {what}: path not found ({path})")
        return True
    return False
