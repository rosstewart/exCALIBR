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

# Real UV calibration data (activation/pemr/futr assays, fit on log-transformed
# scores -- hence "logscores"), confirmed present on disk and wired up via
# uv_sources.load_fgfr_uv_points (verified: reproduces the exact same
# standard_points distribution as the calibration's own exported variants.csv
# for all 3 assays, all 1462 FGFR ms variants bridged).
FGFR_UV_CALIB_DIR = "/data/ross/assay_calibration/FGFR/uv_calib_logscores"

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

# Canonical (production) results -- v3 confirmed as the current/correct one
# (v2 exists on disk too but is superseded/unused).
#
# NOTE: these predate the Sept 2026 EM fixes (completed-data M-step, Gamma
# ridge, corrected Delta cap, q=1/q=2 alignment) and were moved under
# multivariate/pre_092026_em_fixes/ when the refit landed. They are kept
# wired up so existing registry-driven analyses still resolve, but any
# comparison against post-fix fits must account for the fitter differences.
CANONICAL_RESULTS_JSON = _env(
    "MV_CANONICAL_RESULTS",
    "/data/ross/assay_calibration/multivariate/pre_092026_em_fixes/"
    "jobs_all_1000b_8f_v2/bootstrap_results_v3.json.gz")

# Experimental staged-init all_assayed results (mv_analysis/
# experimental_staged_fit.py) -- NOT wired into hpc/prepare.py's production
# path yet (see the cockpit plan's item 2). Only labelseq/integrated support
# all_assayed at all: predictor/combined/card11/tp53 retain no unlabeled bulk
# at ingestion, confirmed this session.
EXPERIMENTAL_STAGED_FIT_DIR = _env(
    "MV_EXPERIMENTAL_STAGED_FIT_DIR",
    "/data/ross/assay_calibration/multivariate/pre_092026_em_fixes/experimental_staged_fit")

GENE_SETS_SUPPORTING_ALL_ASSAYED = ("labelseq", "integrated")

# ---------------------------------------------------------------------------
# Paper-analysis notebook (mv_analysis/analyze_mv_pipeline_output.py) config
# ---------------------------------------------------------------------------
# Named by ROLE ("the fits backing the current paper draft"), not by run
# name -- a future refit is a one-line path update here, not a rename of
# every downstream reference. This is the post-Sept-2026-EM-fixes run
# (backtracking removed from cfusn/fit.py, raise_on_error default flipped,
# staged-init restart diversity fixed, Gamma ridge/Delta cap corrections --
# see git history), superseding CANONICAL_RESULTS_JSON above for anything
# actually going into the manuscript.
PAPER_FITS_DIR = _env(
    "MV_PAPER_FITS_DIR", "/data/ross/assay_calibration/multivariate/jobs_all_100b_3f_092026")

# Aggregation output (hpc/aggregate_results.py {PAPER_FITS_DIR}) -- the
# paper notebook's stage 0 generates this if it doesn't exist yet.
PAPER_RESULTS_JSON = _env("MV_PAPER_RESULTS_JSON", f"{PAPER_FITS_DIR}/bootstrap_results.json.gz")

# Run-specific figure output, NOT the paper repo's Figures/mv/ -- matches
# the tp53_pca10_run/figures precedent; copying into the paper repo is a
# separate, deliberate step the user does themselves.
PAPER_FIGURES_DIR = _env("MV_PAPER_FIGURES_DIR", f"{PAPER_FITS_DIR}/figures")

# Categorized subdirectories under PAPER_FIGURES_DIR -- every save-path call
# site should build off these rather than hardcoding "{PAPER_FIGURES_DIR}/foo"
# directly, so the directory stays organized by figure/data TYPE (not a flat
# dump) as new outputs get added. See the figures-directory restructure plan
# for the full rationale/mapping of pre-existing files into this scheme.
PAPER_FUNCTIONAL_DIR = f"{PAPER_FIGURES_DIR}/functional"
PAPER_PREDICTOR_DIR = f"{PAPER_FIGURES_DIR}/predictor"
PAPER_COMBINED_DIR = f"{PAPER_FIGURES_DIR}/combined"
PAPER_POOLED_DIR = f"{PAPER_FIGURES_DIR}/pooled"
PAPER_ACCURACY_DIR = f"{PAPER_FIGURES_DIR}/accuracy"
PAPER_CONFUSIONS_DIR = f"{PAPER_FIGURES_DIR}/confusions"
PAPER_VUS_RECLASSIFICATION_DIR = f"{PAPER_FIGURES_DIR}/vus_reclassification"
PAPER_BRNICH_COMPARISON_DIR = f"{PAPER_FIGURES_DIR}/brnich_comparison"
PAPER_PER_GENE_DIR = f"{PAPER_FIGURES_DIR}/per_gene"
PAPER_TP53_RPV_DIR = f"{PAPER_FIGURES_DIR}/tp53_rpv"
PAPER_RET_EVIDENCE_DIR = f"{PAPER_FIGURES_DIR}/ret_evidence"
PAPER_CARD11_PHENOTYPE_DIR = f"{PAPER_FIGURES_DIR}/card11_phenotype"
PAPER_CARTOONS_DIR = f"{PAPER_FIGURES_DIR}/cartoons"
PAPER_CACHE_DIR = f"{PAPER_FIGURES_DIR}/.cache"

# The three computational predictors calibrated together for the "predictor"
# and "combined" gene-sets (see src/assay_calibration/multivariate_data/
# predictors.py) -- named here once for the per-individual-predictor
# comparison (mv_analysis/report.py's individual_predictor_comparison).
PREDICTOR_NAMES = ("REVEL", "AlphaMissense", "MutPred2")


def canonical_dataset_name(gene: str, gene_set: str, results_json: str = None) -> str:
    """Resolve the results-JSON top-level key for `gene`'s canonical fit
    under `gene_set`. Reuses the existing, already-correct resolvers instead
    of re-deriving the naming rules here: `gene_set_dataset_label` (regular
    "{gene}_{gene_set}_mv" formula -- labelseq/tp53/card11/combined/fgfr),
    `predictor_dataset_label` ("{gene}_predictors_mv" -- NOTE plural
    "predictors", not the generic formula's "{gene}_predictor_mv", confirmed
    against real results-JSON keys), and `_find_dataset_key` (irregular
    per-gene lookup, needed for the plain-"integrated" gene-set's
    GENES_2018/`_mv_clinvar_2018` special-casing and other exceptions like
    "TP53_tp53_mv" existing alongside "TP53_mv_clinvar_2018"). Deferred
    imports to avoid a circular import with mv_analysis.gene_performance_scatter
    (which imports this module).

    `results_json`, if given, overrides `CANONICAL_RESULTS_JSON` for the
    "integrated" branch's `_find_dataset_key` scan -- e.g. pass
    `PAPER_RESULTS_JSON` to resolve dataset keys against the post-Sept-2026
    fits instead of the pre-fix ones this constant now points to.
    """
    results_json = results_json or CANONICAL_RESULTS_JSON
    if gene_set == "integrated":
        from mv_analysis.gene_performance_scatter import _find_dataset_key
        return _find_dataset_key(results_json, gene.upper())
    if gene_set == "predictor":
        from src.assay_calibration.multivariate_data.predictors import predictor_dataset_label
        return predictor_dataset_label(gene.upper())
    from src.assay_calibration.multivariate_data.common import gene_set_dataset_label
    if gene_set == "fgfr":
        # "FGFR_combined" is already the exact combined-output name (mixed
        # case) -- .upper() would give "FGFR_COMBINED", which doesn't match
        # the real key "FGFR_combined_fgfr_mv" (confirmed via a real
        # KeyError). Use the gene name as given, no case transform.
        gene_key = gene
    else:
        gene_key = gene.lower() if gene_set == "labelseq" else gene.upper()
    return gene_set_dataset_label(gene_key, gene_set)


def _cluster_suffixed_name(dataset_name: str, cluster_idx: int) -> str:
    """Mirrors hpc/prepare.py's `_cluster_suffixed_label` exactly (same
    insertion point -- right before the first "_mv" marker) so a dataset
    name built here always matches what job generation actually wrote.
    cluster_idx==0 (the most-informative cluster) is unsuffixed."""
    if cluster_idx == 0:
        return dataset_name
    marker = "_mv"
    pos = dataset_name.find(marker)
    if pos == -1:
        return f"{dataset_name}_cluster{cluster_idx + 1}"
    return f"{dataset_name[:pos]}_cluster{cluster_idx + 1}{dataset_name[pos:]}"


def list_gene_cluster_variants(gene: str, gene_set: str, results_json: str = None,
                                max_clusters: int = 10) -> "list[str]":
    """[primary_dataset_name, cluster2_name, cluster3_name, ...] -- every
    dataset-name variant actually present in `results_json` for this
    (gene, gene_set), in cluster-informativeness order (see
    Fit._select_all_calibration_clusters / hpc/prepare.py's per-disjoint-
    cluster job generation). For the common single-cluster case this is
    just `[canonical_dataset_name(...)]`; for a gene like BRCA2 under
    --gene-set integrated (8 dims split into a 3-dim and a 4-dim disjoint
    cluster) it also finds "BRCA2_cluster2_mv" if that job has been run and
    aggregated. Stops at the first missing cluster index (no gaps expected
    -- job generation always numbers clusters contiguously from 1).
    """
    results_json = results_json or CANONICAL_RESULTS_JSON
    from mv_analysis.gene_performance_scatter import _load_raw
    raw = _load_raw(results_json)
    primary = canonical_dataset_name(gene, gene_set, results_json=results_json)
    variants = [primary]
    for cluster_idx in range(1, max_clusters):
        candidate = _cluster_suffixed_name(primary, cluster_idx)
        if candidate not in raw:
            break
        variants.append(candidate)
    return variants


def build_ms_for_gene_cluster(gene_set: str, gene: str, cluster_idx: int = 0,
                               min_overlap_rows: int = 30):
    """Build `gene`'s MultiScoreset restricted to disjoint assay-dimension
    cluster `cluster_idx` (0 = most informative, matching
    `Fit._select_all_calibration_clusters`'s ranking -- same ranking
    hpc/prepare.py's job generation uses, so cluster_idx N here always
    corresponds to whatever job generation labeled "_clusterN+1_mv").

    No-op (returns the ms unrestricted) when the gene has only one
    qualifying cluster covering all its dims -- the common case, and
    exactly matches what `build_multiscoresets_for_gene_set` already
    returned before per-cluster fitting existed, so single-cluster genes
    are unaffected by this function existing.

    Needed because, once a gene has multiple clusters, EVERY cluster's
    fit (including cluster 0, the "primary"/unsuffixed dataset name) only
    used a SUBSET of the gene's full dimension set -- rebuilding the full,
    unrestricted ms and scoring against a cluster's fit would trip the
    fit-vs-dataset shape guard (`MVCalibrationAnalysis._validate_fit_shape`),
    exactly as it did for BRCA2 before this was understood.
    """
    ms_map = build_multiscoresets_for_gene_set(gene_set, genes=[gene])
    gene_key = gene.lower() if gene_set == "labelseq" else gene.upper()
    ms = ms_map.get(gene_key) or ms_map.get(gene) or next(iter(ms_map.values()))

    import numpy as np
    from src.assay_calibration.fit_utils.fit import Fit
    obs = np.asarray(ms.scores, dtype=float)
    clusters = []
    if obs.ndim == 2 and obs.shape[1] > 1:
        clusters = Fit._select_all_calibration_clusters(obs, min_overlap_rows=min_overlap_rows)
    if len(clusters) <= 1:
        return ms
    if cluster_idx >= len(clusters):
        raise ValueError(
            f"{gene}/{gene_set} has only {len(clusters)} qualifying cluster(s), "
            f"requested cluster_idx={cluster_idx}."
        )
    from src.assay_calibration.multivariate_data.redundancy_collapse import select_dims
    select_dims(ms, clusters[cluster_idx])
    return ms


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
    """MultiScoreset builder dispatch by gene_set -- one entry point for
    every gene_set the paper analysis notebook needs (labelseq, integrated,
    predictor, card11, tp53, fgfr, combined), instead of each caller
    reimplementing the per-gene-set ingestion logic that used to live only
    inline inside mv_analysis/gene_performance_scatter.py's panel builders.

    `regularization_type` (all_assayed staged-init support) only applies to
    labelseq/integrated (GENE_SETS_SUPPORTING_ALL_ASSAYED) -- predictor/
    combined/card11/tp53/fgfr retain no unlabeled bulk at ingestion
    (confirmed this session), so a non-None value there raises rather than
    being silently ignored.

    `redundancy_collapse_preset`: for gene_set="tp53", defaults (when left
    as `None`, i.e. no explicit preference) to `TP53_DEFAULT_COLLAPSE_PRESET`
    -- matching hpc/prepare.py's own default for the dedicated TP53
    ingestion path, and matching what TP53's actual production fit
    (TP53_tp53_mv, e.g. under jobs_all_100b_3f_092026) was trained on (10
    dims, not the raw 16 -- confirmed via the fit-vs-dataset shape guard
    rejecting the un-collapsed ms). Pass `redundancy_collapse_preset=False`
    (not None) to explicitly opt OUT and get the raw 16-dim ms, mirroring
    hpc/prepare.py's --no-default-redundancy-collapse.
    """
    if regularization_type is not None and gene_set not in GENE_SETS_SUPPORTING_ALL_ASSAYED:
        raise ValueError(
            f"gene_set={gene_set!r} does not support regularization_type={regularization_type!r} -- "
            f"no unlabeled bulk is retained at ingestion. Supported: {GENE_SETS_SUPPORTING_ALL_ASSAYED}."
        )
    if gene_set == "tp53" and redundancy_collapse_preset is None:
        # Deferred import from hpc/prepare.py -- the single source of truth
        # for this default (mv_analysis/build.py already imports prepare.py
        # directly the same way, confirming it's import-safe: no top-level
        # side effects beyond function/argparse definitions).
        import sys as _sys
        from pathlib import Path as _Path
        _hpc_dir = str(_Path(_ROOT) / "hpc")
        if _hpc_dir not in _sys.path:
            _sys.path.insert(0, _hpc_dir)
        import prepare as _hpc_prepare
        redundancy_collapse_preset = _hpc_prepare.TP53_DEFAULT_COLLAPSE_PRESET
    if redundancy_collapse_preset is False:
        redundancy_collapse_preset = None

    if gene_set == "labelseq":
        from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
        ms_map = build_labelseq_multiscoresets(genes=genes, regularization_type=regularization_type)
    elif gene_set == "integrated":
        from mv_analysis.integrated_gene_data import build_integrated_multiscoresets
        ms_map = build_integrated_multiscoresets(genes=genes, regularization_type=regularization_type)
    elif gene_set == "tp53":
        from src.assay_calibration.multivariate_data.tp53 import build_tp53_multiscoreset
        ms_map = {"TP53": build_tp53_multiscoreset()}
    elif gene_set == "card11":
        from src.assay_calibration.multivariate_data.card11 import build_card11_multiscoreset
        ms_map = {"CARD11": build_card11_multiscoreset()}
    elif gene_set == "fgfr":
        from src.assay_calibration.multivariate_data.fgfr import build_fgfr_multiscoresets
        # `genes` here filters the RAW paralog names (FGFR1-4) before
        # combining, not the combined OUTPUT key "FGFR_combined" -- a caller
        # asking for the combined gene by its own output name (e.g. genes=
        # ["FGFR_combined"], matching every other gene-set's "genes filters
        # what you get back" convention) would otherwise filter df_fgfr down
        # to zero rows (no raw gene is literally named "FGFR_combined"),
        # confirmed via a real ValueError ("assayed_variant_level must be
        # constant across rows; got []"). Treat that specific case as "no
        # filter" (every paralog, default combine_genes=True behavior).
        fgfr_genes = None if genes in (None, ["FGFR_combined"]) else genes
        ms_map = build_fgfr_multiscoresets(genes=fgfr_genes)
    elif gene_set == "predictor":
        from src.assay_calibration.multivariate_data.predictors import load_predictor_ms, DEFAULT_GENES
        target_genes = list(genes) if genes else list(DEFAULT_GENES)
        ms_map = {}
        for gene in target_genes:
            try:
                ms_map[gene] = load_predictor_ms(gene, PREDICTOR_RAW_DATA_DIR)
            except ValueError as e:
                print(f"  [{gene}] could not build predictor ms: {e}")
    elif gene_set == "combined":
        # Mirrors gene_performance_scatter.build_panel_c's per-gene loop
        # exactly (functional dims via build_functional_scoresets, merged
        # with predictor dims via build_combined_multiscoreset) -- not
        # duplicated logic, just lifted out of that panel-specific function
        # so any caller can build a "combined" gene's ms directly.
        import pandas as pd
        from src.assay_calibration.multivariate_data.combined import (
            build_functional_scoresets, build_combined_multiscoreset,
            get_functionally_assayed_protein_variants, DEFAULT_INTEGRATED_DATAFRAME,
        )
        from src.assay_calibration.multivariate_data.predictors import DEFAULT_GENES
        from src.assay_calibration.multivariate_data.common import resolve_clinvar_release
        target_genes = list(genes) if genes else list(DEFAULT_GENES)
        df_integrated = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
        ms_map = {}
        for gene in target_genes:
            datasets = sorted(df_integrated[df_integrated["Gene"] == gene]["Dataset"].unique())
            if not datasets:
                print(f"  [{gene}] no functional datasets, skipping")
                continue
            functional_scoresets = build_functional_scoresets(
                df_integrated, gene, datasets, clinvar_release=resolve_clinvar_release(gene))
            functionally_assayed = get_functionally_assayed_protein_variants(df_integrated, gene, datasets)
            ms = build_combined_multiscoreset(
                gene, functional_scoresets, datasets, PREDICTOR_RAW_DATA_DIR,
                functionally_assayed_variants=functionally_assayed)
            if ms is None:
                print(f"  [{gene}] could not build combined ms, skipping")
                continue
            ms_map[gene] = ms
    else:
        raise ValueError(
            f"No MultiScoreset builder registered for gene_set={gene_set!r}. Currently registered: "
            f"'labelseq', 'integrated', 'tp53', 'card11', 'fgfr', 'predictor', 'combined'."
        )
    if redundancy_collapse_preset is not None:
        from src.assay_calibration.multivariate_data import redundancy_collapse as rc
        rc.apply_preset(ms_map, redundancy_collapse_preset)
    return ms_map
