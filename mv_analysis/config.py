"""
Path constants for mv_analysis. All *_UV_CALIB_DIR / *_UV_CALIB constants
below point to existing UNIVARIATE (UV) ExCALIBR calibration outputs on
disk -- this is the ground-truth UV data mv_analysis compares its
multivariate (MV) results against. None of these are MV data; MV fits/
results live under jobs_all_1000b_8f/ and its aggregated
bootstrap_results.json.gz (see hpc/aggregate_results.py), passed in
separately via --results-json.

Every constant can be overridden with a matching MV_* environment variable,
same convention as analysis/config.py, but this module intentionally does
not import from analysis/ -- the UV analysis package was only a structural
reference, not a shared dependency.
"""
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
    return os.environ.get(name, default)


# TP53 + the 19 "integrated" genes (BRCA1, BRCA2, MSH2, PTEN, ASPA, etc.)
# ingested from the shared multi-assay dataframe -- one dir per dataset,
# files named "<name>_<n_c>_<benign_method>_variants.csv".
EXC_PP_CLINVAR2025_UV_CALIB = _env(
    "MV_EXC_PP_UV_CALIB", "/data/ross/assay_calibration/exc_pp_clinvar2025_calib")

# The 17 LABEL-seq pathway genes -- one dir per (gene, assay, treatment),
# files named "<name>_3c_variants.csv".
LABELSEQ_UV_CALIB_DIR = _env(
    "MV_LABELSEQ_UV_CALIB", "/data/ross/assay_calibration/labelseq_uv_calib")

# Computational-predictor-only UV calibrations -- one dir per
# (predictor, gene), files named "<PREDICTOR>_<GENE>_3c_variants.csv".
PREDICTOR_UV_CALIB_DIR = _env(
    "MV_PREDICTOR_UV_CALIB",
    "/data/ross/assay_calibration/predictor_calibrations/predictor_calibration_output")

# CARD11 lof/gof UV calibrations, range-based CSV format (different from
# the point-per-row format used by the three sources above).
CARD11_UV_CALIB_DIR = _env(
    "MV_CARD11_UV_CALIB", "/data/ross/assay_calibration/CARD11/calibration_results")

# Raw per-predictor input CSVs (protein_variant, score, sample_assignments)
# used both to build the predictor-mv MultiScoreset (multivariate_data/
# predictors.py) and, here, to bridge predictor UV calibration outputs'
# positional "variant_N" ids back to protein_variant strings -- verified
# empirically (see uv_sources.py) that the UV pipeline consumed the exact
# same filtered per-predictor CSVs in the same row order, so this positional
# bridge is safe.
PREDICTOR_RAW_DATA_DIR = _env(
    "MV_PREDICTOR_RAW_DATA", "/data/ross/assay_calibration/predictor_calibrations/single_gene_calibration_data")

# FGFR UV calibrations are pending -- no comparison source exists yet.
# Every call site must treat None as "skip UV comparison for this gene-set".
FGFR_UV_CALIB_DIR = None

# Canonical n_c/benign_method per dataset for the exc_pp_clinvar2025_calib
# source (same file the UV/analysis/ package uses to pick "the" calibration
# for a dataset -- see analysis/config.py's DATASET_CONFIGS).
DATASET_CONFIGS_PATH = _env(
    "MV_DATASET_CONFIGS", os.path.join(_ROOT, "src/igvf_configs/dataset_configs_aug_2026.json"))

# Raw TP53 source table used to bridge exc_pp UV variant_ids (nucleotide-level
# hgvs_c) onto TP53's own BasicMultiScoreset ids (protein short-form, e.g.
# "R175H", stored in the "Variant" column) -- see uv_sources.py's
# tp53_bridge_source().
TP53_ANNOTATED_VARIANTS_PATH = _env(
    "MV_TP53_ANNOTATED_VARIANTS", "/data/ross/assay_calibration/TP53/tp53_annotated_variants.csv")

# Shared multi-assay dataframe used both to build the "combined"
# (predictor+functional) gene-set's functional dimensions (see
# multivariate_data/combined.py) and, here, to bridge exc_pp UV variant_ids
# onto whichever variant-identity strategy those genes' Scoresets use.
INTEGRATED_VARIANT_EFFECT_DATASET_PATH = _env(
    "MV_INTEGRATED_DATAFRAME",
    "/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz")

# Genes ingested via the LABEL-seq pathway loader.
LABELSEQ_GENES = (
    "araf", "braf", "craf", "egfr", "erbb2", "grb2", "kras", "ksr1", "ksr2",
    "mek1", "mek2", "met", "mras", "ret", "shp2", "sos1", "sos2",
)

# FGFR paralogs.
FGFR_GENES = ("FGFR1", "FGFR2", "FGFR3", "FGFR4")
