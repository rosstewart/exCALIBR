"""
Per-gene-set adapters that load existing UV (univariate ExCALIBR)
calibration outputs and align them to a gene's MultiScoreset/
BasicMultiScoreset (``ms``), so mv_analysis.report can compare MV results
against real UV baselines instead of leaving that comparison unquantified.

Common interface
-----------------
    load_uv_points(gene, ms, gene_set) -> (dataset_names, points_matrix) | None

``points_matrix`` has shape (n_datasets, ms.n_variants), aligned to
``ms``'s variant order, NaN where a variant/dataset pair has no calibrated
evidence. Returns None when no UV comparison is possible (FGFR: pending
data; predictor-mv: bridging not yet verified -- see below).

Per-gene-set UV data sources
-----------------------------
- card11: range-based ``card11_calibrations.csv`` (CARD11_UV_CALIB_DIR),
  parsed via the existing compare_uv_mv.py CSV machinery.
- labelseq: one dir per (gene, assay, treatment) under LABELSEQ_UV_CALIB_DIR,
  bridged onto ``ms`` via hgvs_c (compare_uv_mv_agg.py, unchanged).
- tp53: one dir per dataset under EXC_PP_CLINVAR2025_UV_CALIB, bridged onto
  TP53's own "Variant" (protein short-form) ids via
  tp53_annotated_variants.csv's clinvar_HGVS_name column. Verified against
  real fits (see mv_analysis/README.md).
- combined (predictor+functional -- BRCA1, BRCA2, F9, JAG1, MSH2, SCN5A,
  TP53, TSC2): same EXC_PP_CLINVAR2025_UV_CALIB source, bridged via the
  integrated dataframe's aa_ref/aa_pos/aa_alt -> short protein_variant
  string (e.g. "A119T"), matching the format these MultiScoresets actually
  key on (confirmed via ms._variants_kept). An earlier version bridged via
  hgvs_p/genomic instead, which never matches this short format -- produced
  an all-NaN points matrix (zero bridged variants) for every one of the 8
  genes tested, not a per-gene issue. Verified against real fits for all 8
  genes after the fix (see mv_analysis/README.md).
- predictor-mv: PREDICTOR_UV_CALIB_DIR/<PREDICTOR>_<GENE>/, one
  ``standard_points`` CSV per predictor (REVEL/MP2/AM), bridged via
  positional "variant_N" ids -> the same row of the raw per-predictor CSV
  (PREDICTOR_RAW_DATA_DIR) -> its protein_variant string -> ms. Verified
  for TP53's three predictors: the raw CSV and its UV output have identical
  row counts with matching 'score' arrays by position (zero rows dropped
  by the MV-side ingestion filter), so the positional bridge is safe --
  spot-check match rate for a new gene before trusting it blindly.
- fgfr: UV calibrations are pending; always returns None.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mv_analysis import config

from src.assay_calibration.multivariate_analysis.compare_uv_mv_agg import (
    discover_gene_uv_datasets, load_uv_scoreset_points, build_hgvs_bridge,
    _extract_hgvs_c, build_uv_points_matrix, classify_score_from_point_ranges,
)
from src.assay_calibration.multivariate_analysis.compare_uv_mv import (
    load_and_classify_from_csv, combine_uv_calibrations,
)
from src.assay_calibration.multivariate_data.labelseq import build_labelseq_dataframe
from src.assay_calibration.multivariate_data.predictors import PREDICTORS


def _dataset_configs():
    with open(config.DATASET_CONFIGS_PATH) as f:
        return json.load(f)


def _calibration_json_path(csv_path):
    stem = csv_path.name[: -len("_variants.csv")]
    return csv_path.with_name(f"{stem}_calibration.json")


def _load_variants_with_canonical_points(csv_path):
    """Read a variants.csv but recompute 'standard_points' from the sibling
    calibration.json's point_ranges instead of trusting the CSV's own
    column, which can be stale relative to the canonical calibration.json
    (see compare_uv_mv_agg.classify_score_from_point_ranges)."""
    df = pd.read_csv(csv_path)
    with open(_calibration_json_path(csv_path)) as f:
        point_ranges = json.load(f)["point_ranges"]
    df = df.copy()
    df["standard_points"] = df["score"].map(lambda s: classify_score_from_point_ranges(s, point_ranges))
    return df


def _uv_variants_csv_path(dataset_dir_name, uv_calib_dir, dataset_configs):
    """<uv_calib_dir>/<dataset>/<dataset>_<n_c>_<benign_method>_variants.csv,
    using the canonical n_c/benign_method combo for this dataset (same file
    analysis/config.py's DATASET_CONFIGS resolves for the UV/analysis/
    package). Returns None if the dataset has no entry."""
    cfg = dataset_configs.get(dataset_dir_name)
    if cfg is None:
        return None
    n_c = cfg.get("n_c", "3c")
    benign_method = cfg.get("benign_method", "avg")
    return Path(uv_calib_dir) / dataset_dir_name / f"{dataset_dir_name}_{n_c}_{benign_method}_variants.csv"


def _discover_exc_pp_datasets(gene, uv_calib_dir=None):
    uv_calib_dir = uv_calib_dir or config.EXC_PP_CLINVAR2025_UV_CALIB
    root = Path(uv_calib_dir)
    prefix = f"{gene.upper()}_"
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.upper().startswith(prefix))


# ── card11 ───────────────────────────────────────────────────────────────

_CARD11_DIM_MAPPING = {
    "lof_clinvar": 0, "lof_cadins": 0,
    "gof_clinvar": 1, "gof_benta": 1,
}


def _load_card11_uv_matrix(ms):
    csv_path = Path(config.CARD11_UV_CALIB_DIR) / "card11_calibrations.csv"
    if not csv_path.exists():
        return None
    pts = load_and_classify_from_csv(ms, str(csv_path), _CARD11_DIM_MAPPING)
    names = list(_CARD11_DIM_MAPPING)
    mat = np.vstack([pts[n] for n in names])
    return names, mat


# ── labelseq ─────────────────────────────────────────────────────────────

def _load_labelseq_uv_matrix(gene, ms):
    df_labelseq = build_labelseq_dataframe()
    datasets, mat = build_uv_points_matrix(ms, gene, df_labelseq, config.LABELSEQ_UV_CALIB_DIR)
    if not datasets:
        return None
    return datasets, mat


# ── tp53 ─────────────────────────────────────────────────────────────────

# clinvar_HGVS_name looks like "NM_000546.6(TP53):c.5A>G (p.Glu2Gly)" --
# extract "NM_000546.6:c.5A>G" to match the hgvs_c string _extract_hgvs_c
# pulls out of an exc_pp UV variant_id (e.g.
# "...NM_000546.6:c.355G>A"), which has no parenthesized gene symbol.
_TP53_HGVS_C_FROM_CLINVAR_NAME_RE = re.compile(r'(N[MR]_\d+\.\d+)\([^)]*\):(c\.[^\s]+)')


def _tp53_hgvs_c_bridge():
    """{hgvs_c: Variant} built from tp53_annotated_variants.csv, to bridge
    exc_pp UV variant_ids (nucleotide-level hgvs_c) onto TP53's own
    BasicMultiScoreset ids (protein short-form "Variant" strings, e.g.
    "R175H"). clinvar_HGVS_name is only populated for ~30% of all TP53 rows
    (those ClinVar has an entry for), which caps the achievable bridge --
    empirically ~71% of exc_pp UV rows with a parseable hgvs_c resolve
    through this bridge, and ~200 of TP53's 405 P/LP+B/LB eval variants end
    up with at least one non-NaN UV point call across the 16 datasets
    (verified against real fits; see mv_analysis/README.md). The remainder
    aren't a bridging bug so much as a real coverage gap: not every clinical
    P/LP/B/LB variant was measured by these particular deep-mutational-scan
    assays, and/or lacks a ClinVar entry to source clinvar_HGVS_name from.
    """
    df = pd.read_csv(config.TP53_ANNOTATED_VARIANTS_PATH)
    bridge = {}
    for variant, name in zip(df["Variant"], df["clinvar_HGVS_name"]):
        if pd.isna(name):
            continue
        m = _TP53_HGVS_C_FROM_CLINVAR_NAME_RE.search(name)
        if not m:
            continue
        bridge[f"{m.group(1)}:{m.group(2)}"] = variant
    return bridge


def load_tp53_uv_points(ms):
    datasets = _discover_exc_pp_datasets("TP53")
    if not datasets:
        return None
    dataset_configs = _dataset_configs()
    bridge = _tp53_hgvs_c_bridge()
    key_index = {vk: i for i, vk in enumerate(ms._variants_kept)}  # flat "Variant" strings, not tuples
    mat = np.full((len(datasets), ms.n_variants), np.nan)
    used = []
    for d_i, dataset_name in enumerate(datasets):
        csv_path = _uv_variants_csv_path(dataset_name, config.EXC_PP_CLINVAR2025_UV_CALIB, dataset_configs)
        if csv_path is None or not csv_path.exists():
            continue
        df = _load_variants_with_canonical_points(csv_path)
        hgvs_c = df["variant_id"].map(_extract_hgvs_c)
        for hc, pts in zip(hgvs_c, df["standard_points"]):
            if hc is None:
                continue
            variant = bridge.get(hc)
            if variant is None:
                continue
            idx = key_index.get(variant)
            if idx is not None:
                mat[d_i, idx] = pts
        used.append(dataset_name)
    if not used:
        return None
    return datasets, mat


# ── combined / "integrated" genes ───────────────────────────────────────

def _integrated_df():
    path = config.INTEGRATED_VARIANT_EFFECT_DATASET_PATH
    sep = "\t" if path.endswith((".tsv.gz", ".tsv")) else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _build_integrated_hgvs_bridge(df_integrated, gene):
    """{hgvs_c: short_protein_variant} for *gene*, where short_protein_variant
    is "<aa_ref><aa_pos><aa_alt>" (e.g. "A119T") -- the SAME short format
    combined.py's own get_functionally_assayed_protein_variants builds and
    that 'combined' gene-set MultiScoresets actually key on (confirmed via
    ms._variants_kept, e.g. ['A102G', 'A102V', ...]). An earlier version of
    this bridge matched against hgvs_p/genomic columns instead, which never
    matches these short-format keys -- confirmed empirically: it produced
    an all-NaN points matrix (zero bridged variants) for every one of the 8
    'combined' genes tested, not just some."""
    sub = df_integrated[df_integrated["Gene"] == gene]
    sub = sub.dropna(subset=["hgvs_c"]).drop_duplicates(subset="hgvs_c")
    bridge = {}
    for row in sub.itertuples(index=False):
        aa_ref, aa_pos, aa_alt = getattr(row, "aa_ref", None), getattr(row, "aa_pos", None), getattr(row, "aa_alt", None)
        if pd.isna(aa_ref) or pd.isna(aa_pos) or pd.isna(aa_alt):
            continue
        bridge[row.hgvs_c] = f"{aa_ref}{int(aa_pos)}{aa_alt}"
    return bridge


def load_combined_uv_points(gene, ms, df_integrated=None):
    """UV bridging for 'combined'/'integrated' genes, via aa_ref/aa_pos/
    aa_alt -> short protein_variant string (matching ms._variants_kept's own
    format) rather than hgvs_p/genomic (see _build_integrated_hgvs_bridge)."""
    datasets = _discover_exc_pp_datasets(gene)
    if not datasets:
        return None
    if df_integrated is None:
        df_integrated = _integrated_df()
    dataset_configs = _dataset_configs()
    bridge = _build_integrated_hgvs_bridge(df_integrated, gene)
    key_index = {vk: i for i, vk in enumerate(ms._variants_kept)}
    mat = np.full((len(datasets), ms.n_variants), np.nan)
    used = []
    for d_i, dataset_name in enumerate(datasets):
        csv_path = _uv_variants_csv_path(dataset_name, config.EXC_PP_CLINVAR2025_UV_CALIB, dataset_configs)
        if csv_path is None or not csv_path.exists():
            continue
        df = _load_variants_with_canonical_points(csv_path)
        hgvs_c = df["variant_id"].map(_extract_hgvs_c)
        for hc, pts in zip(hgvs_c, df["standard_points"]):
            if hc is None:
                continue
            protein_variant = bridge.get(hc)
            if protein_variant is None:
                continue
            idx = key_index.get(protein_variant)
            if idx is not None:
                mat[d_i, idx] = pts
        used.append(dataset_name)
    if not used:
        return None
    return datasets, mat


# ── predictor-mv ─────────────────────────────────────────────────────────

def _predictor_raw_variant_order(gene, predictor, predictor_data_dir):
    """Row-position -> protein_variant for the raw per-predictor CSV, in the
    same filtered row order/count the UV calibration's positional
    'variant_N' ids refer to (empirically verified for TP53's 3 predictors:
    zero rows dropped, UV 'score' column matches this raw CSV's 'score'
    column exactly by position)."""
    path = Path(predictor_data_dir) / gene / f"{gene}_{predictor}.csv.gz"
    df = pd.read_csv(path, compression="gzip")
    df = df.dropna(subset=["score", "protein_variant", "sample_assignments"])
    return df["protein_variant"].reset_index(drop=True)


def load_predictor_mv_uv_points(gene, ms, predictor_data_dir=None, uv_calib_dir=None):
    predictor_data_dir = predictor_data_dir or config.PREDICTOR_RAW_DATA_DIR
    uv_calib_dir = Path(uv_calib_dir or config.PREDICTOR_UV_CALIB_DIR)
    key_index = {vk: i for i, vk in enumerate(ms._variants_kept)}

    names, rows = [], []
    for predictor in PREDICTORS:
        csv_path = uv_calib_dir / f"{predictor}_{gene}" / f"{predictor}_{gene}_3c_variants.csv"
        if not csv_path.exists():
            continue
        try:
            variant_order = _predictor_raw_variant_order(gene, predictor, predictor_data_dir)
        except FileNotFoundError:
            continue
        df = _load_variants_with_canonical_points(csv_path)
        pos = df["variant_id"].str.replace("variant_", "", regex=False).astype(int)
        if pos.max() >= len(variant_order):
            continue  # positional bridge doesn't hold for this gene -- skip rather than misalign
        row = np.full(ms.n_variants, np.nan)
        for p, pts in zip(pos, df["standard_points"]):
            idx = key_index.get(variant_order.iloc[p])
            if idx is not None:
                row[idx] = pts
        rows.append(row)
        names.append(f"{predictor}_{gene}")
    if not rows:
        return None
    return names, np.vstack(rows)


def load_fgfr_uv_points(gene, ms):
    return None


# ── dispatcher ───────────────────────────────────────────────────────────

def load_uv_points(gene, ms, gene_set):
    """Returns (dataset_names, points_matrix) or None -- see module
    docstring for the common interface and per-gene-set caveats."""
    if gene_set == "card11":
        return _load_card11_uv_matrix(ms)
    if gene_set == "labelseq":
        return _load_labelseq_uv_matrix(gene, ms)
    if gene_set == "tp53":
        return load_tp53_uv_points(ms)
    if gene_set == "combined":
        return load_combined_uv_points(gene, ms)
    if gene_set == "predictor-mv":
        return load_predictor_mv_uv_points(gene, ms)
    if gene_set == "fgfr":
        return load_fgfr_uv_points(gene, ms)
    return None
