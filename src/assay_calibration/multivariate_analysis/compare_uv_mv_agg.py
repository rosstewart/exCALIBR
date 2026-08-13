"""
Compare ExCALIBR-MV against univariate (per-scoreset) calibrations, aggregated
across all scoresets belonging to the same gene.

Univariate calibrations live under labelseq_uv_calib/<dataset_name>/, one
directory per scoreset (e.g. braf_abundance_HSP90i, braf_abundance_No_treatment,
braf_activity_No_treatment all belong to gene "braf"). Each directory has a
"<dataset_name>_3c_avg_variants.csv" with a `standard_points` column per variant.

Since these labelseq scoresets have no genomic coordinates (Chrom/hg38_start
are entirely null) and `hgvs_c` is not 100% populated for any gene checked,
MultiScoreset falls back further to keying variants by `hgvs_p` (protein-level
HGVS; see Scoreset._init_matrices `_code_strategy`), so `ms._variants_kept[i]`
is a singleton tuple `(hgvs_p,)`. The univariate `variant_id` column is built
as `f"{ID}_{Gene}_{Chrom}_{hgvs_c}"` (nucleotide-level); since ID and Chrom
are always None/nan here, the trailing "<transcript>:c.<...>" substring is the
reliable hgvs_c. To join against `ms._variants_kept` we bridge hgvs_c -> hgvs_p
using the source `dataframe_processed.csv.gz` (empirically, this is always a
clean many-to-one map with no within-dataset point conflicts across genes
checked). Genomic and raw hgvs_c keys are also tried, in case a future gene
has cleaner data and MultiScoreset picks a different strategy.

Two ways of combining per-scoreset evidence for the same variant:
  - "non-conflicting": max-magnitude evidence when all non-zero values agree
    in sign; 0 if both a positive and a negative value are present.
  - "max": literal elementwise max across scoresets, e.g. (-1, 2) -> 2.

Main entry points
------------------
compute_gene_uv_agg(ms, gene, uv_calib_dir=DEFAULT_UV_CALIB_DIR)
    -> {'datasets', 'per_dataset', 'nonconflicting', 'max'} for one gene

build_agg_uv_comparison(gene_ms, mv_points_by_gene, uv_calib_dir=DEFAULT_UV_CALIB_DIR)
    -> agg dict keyed by method ('MV', 'UV non-conflicting', 'UV max'),
       usable directly with eval_plot_utils.plot_confusion_matrices /
       print_latex_table.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .eval_plot_utils import _accumulate_into
# aggregate_nonconflicting/aggregate_max now live in mv_analysis.uv_agg
# (generalized for LABEL-seq/predictor-mv/TP53-integrated alike); re-exported
# here unchanged for backward compatibility with existing callers. mv_analysis
# lives at the repo root, three levels up from this file, which isn't always
# on sys.path (e.g. from a notebook) -- add it if the plain import fails.
try:
    from mv_analysis.uv_agg import aggregate_nonconflicting, aggregate_max
except ImportError:
    _repo_root = str(Path(__file__).resolve().parents[3])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from mv_analysis.uv_agg import aggregate_nonconflicting, aggregate_max

DEFAULT_UV_CALIB_DIR = "/data/ross/assay_calibration/labelseq_uv_calib"

_HGVS_C_RE = re.compile(r'(N[MR]_\d+\.\d+:c\..*)$')


# ── Variant key matching ────────────────────────────────────────────────────

def _variant_key_index(ms):
    """Map each ms._variants_kept entry to its row index.

    Handles both the genomic 5-tuple key strategy and the hgvs_c
    singleton-tuple fallback (used when Chrom/hg38_start are unavailable,
    as for the labelseq scoresets) by indexing on both the full tuple and,
    for singletons, the bare string.
    """
    lookup = {}
    for i, vk in enumerate(ms._variants_kept):
        lookup[vk] = i
        if len(vk) == 1:
            lookup[vk[0]] = i
    return lookup


def _extract_hgvs_c(variant_id):
    m = _HGVS_C_RE.search(variant_id)
    return m.group(1) if m else None


def classify_score_from_point_ranges(score, point_ranges):
    """Evidence points for *score* using a calibration.json's own
    ``point_ranges`` (``{"<pts>": [[lower, upper], ...]}``, possibly several
    disjoint intervals per point value). This is the CANONICAL classification
    -- do not trust a variants.csv's own precomputed ``standard_points``
    column, which can be stale relative to point_ranges (observed directly:
    braf_activity_No_treatment's CSV assigns +1 to scores its own
    calibration.json's point_ranges place in the -1 bucket; other datasets in
    the same tree are fine, so this isn't a systemic sign-flip bug, just an
    out-of-date CSV -- recomputing from point_ranges sidesteps it either way).
    Boundaries are lower-inclusive, upper-exclusive: [lower, upper). Returns
    0 (uncertain) when no range matches; NaN when the score is NaN.
    """
    if pd.isna(score):
        return np.nan
    for pts_str, ranges in point_ranges.items():
        pts = int(pts_str)
        for lower, upper in ranges:
            if lower <= score < upper:
                return pts
    return 0


# ── Loading univariate calibrations ─────────────────────────────────────────

def discover_gene_uv_datasets(gene, uv_calib_dir=DEFAULT_UV_CALIB_DIR):
    """List univariate calibration dataset dirs belonging to *gene*."""
    root = Path(uv_calib_dir)
    prefix = f"{gene.lower()}_"
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and p.name.lower().startswith(prefix)
    )


def load_uv_scoreset_points(dataset_name, uv_calib_dir=DEFAULT_UV_CALIB_DIR, component='3c_avg'):
    """Load per-variant evidence points for one univariate scoreset, computed
    directly from the canonical calibration.json's point_ranges applied to
    the CSV's 'score' column (see classify_score_from_point_ranges for why).

    Returns a DataFrame with columns ['hgvs_c', 'standard_points']; rows whose
    variant_id doesn't contain a recognizable hgvs_c suffix are dropped.
    """
    base_dir = Path(uv_calib_dir) / dataset_name
    stem = f"{dataset_name}_{component}"
    df = pd.read_csv(base_dir / f"{stem}_variants.csv")
    with open(base_dir / f"{stem}_calibration.json") as f:
        point_ranges = json.load(f)["point_ranges"]
    points = df['score'].map(lambda s: classify_score_from_point_ranges(s, point_ranges))
    hgvs_c = df['variant_id'].map(_extract_hgvs_c)
    out = pd.DataFrame({'hgvs_c': hgvs_c, 'standard_points': points})
    return out.dropna(subset=['hgvs_c'])


def build_hgvs_bridge(df_labelseq, gene):
    """Build a hgvs_c -> {hgvs_p, genomic} lookup for *gene* from the source
    label-seq dataframe (dataframe_processed.csv.gz), used to join univariate
    variant_ids (nucleotide-level hgvs_c) onto whatever variant-identity
    strategy MultiScoreset actually used (genomic / hgvs_c / hgvs_p).
    """
    sub = df_labelseq[df_labelseq['Gene'].str.lower() == gene.lower()]
    sub = sub.dropna(subset=['hgvs_c']).drop_duplicates(subset='hgvs_c')

    bridge = {}
    genomic_cols = ['mapped_hgvs_g_chromosome', 'mapped_hgvs_g_start',
                     'mapped_hgvs_g_ref', 'mapped_hgvs_g_alt']
    has_genomic_cols = all(c in sub.columns for c in genomic_cols)
    for row in sub.itertuples(index=False):
        hgvs_c = row.hgvs_c
        hgvs_p = getattr(row, 'hgvs_p', None)
        genomic = None
        if has_genomic_cols:
            chrom = getattr(row, 'mapped_hgvs_g_chromosome')
            start = getattr(row, 'mapped_hgvs_g_start')
            ref = getattr(row, 'mapped_hgvs_g_ref')
            alt = getattr(row, 'mapped_hgvs_g_alt')
            if pd.notna(chrom) and pd.notna(start) and pd.notna(ref) and pd.notna(alt):
                genomic = (row.Gene, str(chrom), float(start), str(ref), str(alt))
        bridge[hgvs_c] = {'hgvs_p': hgvs_p if pd.notna(hgvs_p) else None, 'genomic': genomic}
    return bridge


def _resolve_index(hgvs_c, bridge, key_index):
    """Try genomic, hgvs_c, then hgvs_p keys (in that priority order, matching
    Scoreset's own strategy-selection order) against *key_index*."""
    info = bridge.get(hgvs_c)
    if info is not None and info['genomic'] is not None:
        idx = key_index.get(info['genomic'])
        if idx is not None:
            return idx
    idx = key_index.get(hgvs_c)
    if idx is not None:
        return idx
    if info is not None and info['hgvs_p'] is not None:
        return key_index.get(info['hgvs_p'])
    return None


def build_uv_points_matrix(ms, gene, df_labelseq, uv_calib_dir=DEFAULT_UV_CALIB_DIR, component='3c_avg'):
    """Return (dataset_names, points_matrix).

    points_matrix has shape (n_datasets, ms.n_variants); NaN where a
    variant/dataset pair has no calibrated evidence (missing score or
    unmatched variant_id). Requires df_labelseq (dataframe_processed.csv.gz)
    to bridge nucleotide-level UV variant_ids onto whatever variant-identity
    strategy MultiScoreset used for this gene (see build_hgvs_bridge).
    """
    datasets = discover_gene_uv_datasets(gene, uv_calib_dir)
    N = ms.n_variants
    if not datasets:
        return [], np.empty((0, N))

    key_index = _variant_key_index(ms)
    bridge = build_hgvs_bridge(df_labelseq, gene)
    mat = np.full((len(datasets), N), np.nan)
    for d_i, dataset_name in enumerate(datasets):
        try:
            pts_df = load_uv_scoreset_points(dataset_name, uv_calib_dir, component)
        except FileNotFoundError:
            continue
        for hgvs_c, pts in zip(pts_df['hgvs_c'], pts_df['standard_points']):
            idx = _resolve_index(hgvs_c, bridge, key_index)
            if idx is not None:
                mat[d_i, idx] = pts
    return datasets, mat


def compute_gene_uv_agg(ms, gene, df_labelseq, uv_calib_dir=DEFAULT_UV_CALIB_DIR, component='3c_avg'):
    """Load + aggregate all univariate scoresets for *gene* against *ms*.

    Returns None if no univariate calibration directories are found for gene.
    """
    datasets, mat = build_uv_points_matrix(ms, gene, df_labelseq, uv_calib_dir, component)
    if not datasets:
        return None
    return {
        'datasets': datasets,
        'per_dataset': mat,
        'nonconflicting': aggregate_nonconflicting(mat),
        'max': aggregate_max(mat),
    }


# ── Aggregate confusion-matrix comparison across genes ──────────────────────

def build_agg_uv_comparison(gene_ms, mv_points_by_gene, df_labelseq,
                             uv_calib_dir=DEFAULT_UV_CALIB_DIR,
                             component='3c_avg', non_nu_genes=None,
                             mv_key='MV',
                             nonconflicting_key='UV non-conflicting',
                             max_key='UV max'):
    """
    Aggregate confusion-matrix counts across genes for three methods:
    ExCALIBR-MV, UV non-conflicting aggregation, and UV max aggregation.

    Parameters
    ----------
    gene_ms : dict {gene: MultiScoreset}
    mv_points_by_gene : dict {gene: points_array}
        E.g. {g: analysis.results['3c_unc']['points']
              for g, analysis in gene_to_analysis_cons_b0_5.items()}
    df_labelseq : pd.DataFrame
        The source dataframe_processed.csv.gz, used to bridge nucleotide-level
        UV variant_ids onto the variant-identity strategy MultiScoreset used.
    non_nu_genes : set or None
        If provided, restrict to these genes only.

    Returns
    -------
    agg : defaultdict keyed by method string, compatible with
        eval_plot_utils.plot_confusion_matrices / print_latex_table.
    """
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(int))

    for gene, ms in gene_ms.items():
        if non_nu_genes is not None and gene not in non_nu_genes:
            continue

        sa = ms._sample_assignments
        is_plp = sa[:, 0].astype(bool)
        is_blb = sa[:, 1].astype(bool)
        is_snv = sa[:, 2].astype(bool)
        eval_mask = is_plp | is_blb
        if eval_mask.sum() == 0:
            continue

        uv = compute_gene_uv_agg(ms, gene, df_labelseq, uv_calib_dir, component)
        if uv is None:
            continue

        labels = is_plp[eval_mask]
        n_snv = int(is_snv.sum())
        n_ctrl = int(eval_mask.sum())

        for key, pts in [(nonconflicting_key, uv['nonconflicting']),
                          (max_key, uv['max'])]:
            pts = np.nan_to_num(pts, nan=0.0)
            _accumulate_into(agg, key, labels, pts[eval_mask], pts[is_snv], n_snv, n_ctrl)

        mv_pts = mv_points_by_gene.get(gene)
        if mv_pts is not None:
            _accumulate_into(agg, mv_key, labels, mv_pts[eval_mask], mv_pts[is_snv], n_snv, n_ctrl)

    return agg
