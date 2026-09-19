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


# ---------------------------------------------------------------------------
# MV results/provenance registry
# ---------------------------------------------------------------------------
# Every ad hoc script this session hardcoded its own copy of: which results
# JSON, which dataset-key naming convention ({gene}_labelseq_mv vs {GENE}_mv
# vs {GENE}_mv_clinvar_2018 for the GENES_2018 special case vs TP53_tp53_mv
# for the dedicated TP53 gene-set), which MultiScoreset builder. This is the
# single place that knowledge should live instead.

# Canonical (production) results -- v3 confirmed this session as the
# current/correct one (v2 exists on disk too but is superseded/unused).
CANONICAL_RESULTS_JSON = _env(
    "MV_CANONICAL_RESULTS",
    "/data/ross/assay_calibration/multivariate/jobs_all_1000b_8f_v2/bootstrap_results_v3.json.gz")

# Experimental staged-init all_assayed results (mv_analysis/
# experimental_staged_fit.py) -- NOT wired into hpc/prepare.py's production
# path yet (see the cockpit plan's item 2). Only labelseq/integrated support
# all_assayed at all: predictor/combined/card11/tp53 retain no unlabeled bulk
# at ingestion, confirmed this session.
EXPERIMENTAL_STAGED_FIT_DIR = _env(
    "MV_EXPERIMENTAL_STAGED_FIT_DIR",
    "/data/ross/assay_calibration/multivariate/experimental_staged_fit")

GENE_SETS_SUPPORTING_ALL_ASSAYED = ("labelseq", "integrated")


def canonical_dataset_name(gene: str, gene_set: str) -> str:
    """Resolve the results-JSON top-level key for `gene`'s canonical fit
    under `gene_set`. Reuses the two existing, already-correct resolvers
    instead of re-deriving the naming rules here: `gene_set_dataset_label`
    (regular "{gene}_{gene_set}_mv" formula -- labelseq/tp53/card11/
    combined/predictor) and `_find_dataset_key` (irregular per-gene lookup,
    needed for the plain-"integrated" gene-set's GENES_2018/`_mv_clinvar_2018`
    special-casing and other exceptions like "TP53_tp53_mv" existing
    alongside "TP53_mv_clinvar_2018"). Deferred imports to avoid a circular
    import with mv_analysis.gene_performance_scatter (which imports this
    module).
    """
    if gene_set == "integrated":
        from mv_analysis.gene_performance_scatter import _find_dataset_key
        return _find_dataset_key(CANONICAL_RESULTS_JSON, gene.upper())
    from src.assay_calibration.multivariate_data.common import gene_set_dataset_label
    gene_key = gene.lower() if gene_set == "labelseq" else gene.upper()
    return gene_set_dataset_label(gene_key, gene_set)


def staged_init_results_path(gene: str, gene_set: str, n_components: int = 6) -> str:
    if gene_set == "labelseq":
        return (f"{EXPERIMENTAL_STAGED_FIT_DIR}/{gene.lower()}_all_assayed_"
                f"bootstrap_results_stagedinit_big.json.gz")
    if gene_set == "integrated":
        return (f"{EXPERIMENTAL_STAGED_FIT_DIR}/{gene.upper()}_integrated_all_assayed_"
                f"bootstrap_results_stagedinit_big.json.gz")
    raise ValueError(
        f"gene_set={gene_set!r} does not support staged_init_all_assayed -- no "
        f"unlabeled bulk is retained at ingestion for predictor/combined/card11/tp53 "
        f"(confirmed this session). Supported: {GENE_SETS_SUPPORTING_ALL_ASSAYED}."
    )


def staged_init_dataset_name(gene: str, gene_set: str) -> str:
    if gene_set == "labelseq":
        return f"{gene.upper()}_labelseq_mv"
    if gene_set == "integrated":
        return f"{gene.upper()}_mv"
    raise ValueError(
        f"gene_set={gene_set!r} does not support staged_init_all_assayed. "
        f"Supported: {GENE_SETS_SUPPORTING_ALL_ASSAYED}."
    )


def staged_init_config_label(n_components: int = 6) -> str:
    return f"all_assayed_{n_components}c_stagedinit_big"


def build_multiscoresets_for_gene_set(gene_set: str, genes=None, regularization_type=None,
                                       redundancy_collapse_preset=None):
    """MultiScoreset builder dispatch by gene_set. Covers labelseq/integrated
    only -- predictor/combined/card11/tp53's builders currently live inline
    inside mv_analysis/gene_performance_scatter.py's panel-building functions
    (build_panel_b/build_panel_c/build_tp53_multiscoreset/
    build_card11_multiscoreset), not as a standalone (genes, regularization_type)
    -> {gene: ms} callable -- not wired into this registry yet. There is no
    dedicated 'tp53' gene_set entry here either -- the cockpit only reaches
    TP53 via 'integrated' (one of its 19 genes), so `redundancy_collapse_preset`
    (e.g. "tp53_kato_pca2") is the only way to get TP53's Kato_2003 8-assay
    panel collapsed to 2 PCs from the cockpit today; see hpc/prepare.py's
    --redundancy-collapse-preset for the equivalent on the fit-launching side
    (shared definition: redundancy_collapse.PRESETS/apply_preset).
    """
    if gene_set == "labelseq":
        from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
        ms_map = build_labelseq_multiscoresets(genes=genes, regularization_type=regularization_type)
    elif gene_set == "integrated":
        from mv_analysis.integrated_gene_data import build_integrated_multiscoresets
        ms_map = build_integrated_multiscoresets(genes=genes, regularization_type=regularization_type)
    else:
        raise ValueError(
            f"No MultiScoreset builder registered for gene_set={gene_set!r} yet. "
            f"Currently registered: 'labelseq', 'integrated'."
        )
    if redundancy_collapse_preset is not None:
        from src.assay_calibration.multivariate_data import redundancy_collapse as rc
        rc.apply_preset(ms_map, redundancy_collapse_preset)
    return ms_map
