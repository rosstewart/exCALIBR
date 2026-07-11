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
    "/data/ross/assay_calibration/explorer_jobs_pp_revisions_calib",
)

# Master integrated variant-effect dataframe (all datasets, one row per
# variant measurement) — the --dataset input to run_igvf_batch.py.
DATASET_TSV = _env(
    "EXCALIBR_DATASET_TSV",
    "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_merged_89datasets.tsv.gz",
)

# Dataset -> (n_c, benign_method[, overrides]) config JSON, same file passed
# to run_igvf_batch.py --dataset-configs. Used to resolve which component/
# benign-method combo is "the" calibration for a dataset (see
# analysis.discovery.resolve_component / legacy_fits.resolve_component_for).
DATASET_CONFIGS = _env(
    "EXCALIBR_DATASET_CONFIGS",
    "/home/rcstewart/exCALIBR/src/igvf_configs/dataset_configs_jul_2026.json",
)

# Precomputed bootstrap fits (gzipped JSON of full per-bootstrap component
# params/weights) — the --precomputed-fits input to run_igvf_batch.py.
# Needed only for plots that overlay the fitted mixture density (MSH2 example,
# cartoon/schematic figures, Figure 4 panels) rather than just the LR+ curve.
PRECOMPUTED_FITS = _env(
    "EXCALIBR_PRECOMPUTED_FITS",
    "/data/ross/assay_calibration/explorer_jobs_pp_merged_89datasets_bootstrap_results.json.gz",
)

# Where generated figures are written by default.
FIGURE_DIR = _env("EXCALIBR_FIGURE_DIR", str(Path(OUTPUT_DIR) / "figures"))

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

OR_ESTIMATES_CSV = _env(
    "EXCALIBR_OR_ESTIMATES_CSV",
    "/data/ross/assay_calibration/paper/ExCALIBR-or-estimates-2026-02-09.csv.gz",
)

YURIY_OR_CSV = _env(
    "EXCALIBR_YURIY_OR_CSV",
    "/data/ross/assay_calibration/paper/yuriy_odds_ratios.csv",
)

# Brnich et al. OddsPath evidence-code CSV (columns: Dataset, "Evidence Code
# Abnormal", "Evidence Code Normal") — a separate statistical analysis, not
# produced by run_pipeline.py. Used only for the "author" side of
# analysis.assay_stats.plot_evidence_comparison; the ExCALIBR side is always
# computed fresh from pipeline output regardless of whether this file exists.
OP_EVIDENCE_CODES_CSV = _env(
    "EXCALIBR_OP_EVIDENCE_CODES_CSV",
    "/data/ross/assay_calibration/dataframe/OP_clinvar18_25_122325.csv",
)

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
ASSAY_METHOD_MAP_CSV = _env(
    "EXCALIBR_ASSAY_METHOD_MAP_CSV",
    "/data/ross/assay_calibration/dataframe/var_effect_measurements_dataset.csv",
)

# ---------------------------------------------------------------------------
# Comparison methods (analysis.comparison_methods) -- other calibration
# approaches ExCALIBR gets compared against, beyond the author's own calls.
# ---------------------------------------------------------------------------

# acmgscaler R package source checkout (github.com/badonyi/acmgscaler) --
# base-R only, no install needed; analysis.comparison_methods sources the
# R/*.R files directly from here rather than requiring `install.packages`.
ACMGSCALER_DIR = _env("EXCALIBR_ACMGSCALER_DIR", "/data/tools/acmgscaler")

# Simple 2-component GMM baseline (slurm/simple_gmm_baseline.py output) --
# run separately/manually; None until you've run it.
GMM_BASELINE_JSON = _env("EXCALIBR_GMM_BASELINE_JSON", None)

# "Canonical" ExCALIBR fits rerun with force_gaussian=True -- a future
# comparison method; output_dir not yet known, filled in once available.
FORCE_GAUSSIAN_OUTPUT_DIR = _env("EXCALIBR_FORCE_GAUSSIAN_OUTPUT_DIR", None)


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
