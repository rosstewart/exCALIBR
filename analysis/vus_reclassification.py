"""
VCEP VUS-reclassification analysis: for genes with real ClinGen Variant Curation
Expert Panel (VCEP) per-variant applied ACMG evidence codes, strip the PS3/BS3
(functional-assay) codes that originally contributed to each variant's
classification, recompute the "residual" classification without them, then
substitute our own canonical v3 MV evidence (mapped to an equivalent PS3/BS3
code) back in and recompute a "reconstructed" classification -- showing how many
VUS (after stripping) get resolved into P/LP or B/LB using our evidence in place
of the original wet-lab functional assay, and whether the resolved direction
agrees with ClinGen's original call.

Reuses, unmodified:
  - `analysis.clingen.filter_and_recalculate`/`REMOVE_CODES` (the strip-PS3/BS3-
    and-reclassify half).
  - `analysis.acmg_evidence_codes.classify_acmg` (Richards 2015 categorical ACMG
    combiner -- kept consistent with clingen.py's own precedent rather than
    switching to the different point-total-based `classify_from_points` in
    tavtigian_sims/comparisons/tavtigian_acmg_compare_lib.py).
  - `analysis.all_variant_evidence.POINTS_TO_EVIDENCE_CODE` (our points -> a
    PS3/BS3-strength code string), after `ACMG_CODE_FIXUP` normalizes its
    suffixes to `classify_acmg`'s exact key format.

Canonical v3 only -- this analysis does NOT use the all_assayed staged-init
experiment's results.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.clingen import REMOVE_CODES, filter_and_recalculate
from analysis.acmg_evidence_codes import classify_acmg
from analysis.all_variant_evidence import POINTS_TO_EVIDENCE_CODE

# `POINTS_TO_EVIDENCE_CODE`'s suffixes don't exactly match
# `get_acmg_evidence_codes()`'s key format (analysis/acmg_evidence_codes.py) --
# "_Very_Strong" isn't a key there (it uses "_Very"), and "_Moderate+" (point
# value 3) isn't a key at all. Confirmed with user: point 3 rounds DOWN to
# Moderate (conservative -- avoids overclaiming reclassification strength for a
# genuinely intermediate point value).
ACMG_CODE_FIXUP = {
    "PS3_Very_Strong": "PS3_Very",
    "BS3_Very_Strong": "BS3_Very",
    "PS3_Moderate+": "PS3_Moderate",
    "BS3_Moderate+": "BS3_Moderate",
}

_PLP = {"Pathogenic", "Likely Pathogenic"}
_BLB = {"Benign", "Likely Benign"}


def _fixup_code(code: str) -> str:
    return ACMG_CODE_FIXUP.get(code, code)


def points_to_injected_code(points: float) -> Optional[str]:
    """Our -8..+8 points -> a PS3/BS3-strength code `classify_acmg` understands,
    or None if there's no evidence to inject (points == 0, or NaN)."""
    if points is None or (isinstance(points, float) and np.isnan(points)):
        return None
    pv = int(round(points))
    pv = max(-8, min(8, pv))
    code = POINTS_TO_EVIDENCE_CODE.get(pv, "NO_EVIDENCE")
    if code == "NO_EVIDENCE":
        return None
    return _fixup_code(code)


def _direction(classification: str) -> int:
    if classification in _PLP:
        return 1
    if classification in _BLB:
        return -1
    return 0


def substitute_and_reclassify(evidence_string: Optional[str], our_points: float) -> dict:
    """Strip PS3/BS3 from `evidence_string`, reclassify (residual), then inject
    our own canonical-v3-points-derived code and reclassify again
    (reconstructed). Returns a dict with all three classifications plus
    `resolved` (residual was VUS, reconstructed isn't) and `concordant`
    (resolved direction matches the ORIGINAL [unstripped] ClinGen classification's
    P/LP-vs-B/LB direction -- undefined/None when the original itself was VUS,
    since there's nothing to be concordant WITH).
    """
    if pd.isna(evidence_string):
        evidence_string = None

    full_list = [c.strip() for c in evidence_string.split(",")] if evidence_string else []
    original_class = classify_acmg(full_list) if full_list else "VUS"

    residual_class, filtered_str = filter_and_recalculate(evidence_string, strip_functional_evidence=True)
    filtered_list = [c for c in full_list if c not in REMOVE_CODES]

    injected_code = points_to_injected_code(our_points)
    if injected_code is None:
        reconstructed_class = residual_class
    else:
        reconstructed_class = classify_acmg(filtered_list + [injected_code])

    resolved = residual_class == "VUS" and reconstructed_class != "VUS"
    concordant = None
    if resolved and original_class != "VUS":
        concordant = _direction(reconstructed_class) == _direction(original_class)

    return {
        "original_class": original_class,
        "residual_class": residual_class,
        "reconstructed_class": reconstructed_class,
        "injected_code": injected_code,
        "resolved": resolved,
        "concordant": concordant,
    }


def build_labelseq_evidence_lookup(gene: str, df_labelseq) -> Dict[str, str]:
    """{protein-HGVS variant string -> evidence-code string} for one gene,
    from the LABEL-seq source dataframe. Extracted out of
    `analysis/run_vus_reclassification.py`'s `main()` (previously inline,
    duplicated per gene-set loop) so the confusion-matrix builders below can
    reuse the exact same alignment instead of re-deriving it.

    The dataframe is long-format (one row per variant x assay x treatment)
    -- a variant can appear many times with a null evidence-code column for
    most of those rows; filter to non-null FIRST, then dedupe by variant, so
    a later null row never overwrites an earlier real one via naive
    dict(zip(...)) (a real bug found and fixed this session)."""
    evidence_col = "clingen_evidence_repository.Applied Evidence Codes (Met)"
    df_gene = df_labelseq[df_labelseq["Gene"].str.lower() == gene.lower()]
    df_evidence = df_gene[df_gene[evidence_col].notna()].drop_duplicates(subset="variant")
    return dict(zip(df_evidence["variant"], df_evidence[evidence_col]))


def build_integrated_evidence_lookup(gene: str, df_integrated) -> Dict[tuple, str]:
    """{genomic 5-tuple key -> evidence-code string} for one gene, from the
    integrated/pillar-project source dataframe. Same extraction rationale as
    `build_labelseq_evidence_lookup`."""
    evidence_col = "Applied Evidence Codes (Met)_ClinGen_repo"
    df_gene = df_integrated[df_integrated["Gene"] == gene.upper()]
    df_gene = df_gene[df_gene[evidence_col].notna()]
    lookup = {}
    for _, row in df_gene.iterrows():
        if pd.isna(row["hg38_start"]):
            continue
        key = (str(row["Gene"]), str(row["Chrom"]), str(int(float(row["hg38_start"]))),
               str(row["ref_allele"]), str(row["alt_allele"]))
        lookup[key] = row[evidence_col]
    return lookup


def _to_row(label: int) -> int:
    """+1/0/-1 -> row index (0=Pathogenic, 1=Indeterminate, 2=Benign) --
    same convention as `analysis.clingen._to_row`, duplicated here (not
    imported) since it's a 1-line pure function and importing it would pull
    in `analysis.clingen`'s heavier module-level imports (matplotlib
    confusion-panel plotting) for no benefit."""
    return {1: 0, 0: 1, -1: 2}[label]


def build_mv_clinvar_confusion(points, sa) -> np.ndarray:
    """3x2 confusion matrix (rows: MV-predicted Pathogenic/Indeterminate/
    Benign; cols: ClinVar-truth PLP/BLB) for one gene's variants, using the
    SAME ground truth already baked into `ms`'s role membership (role 0 =
    P/LP, role 1 = B/LB) -- no evidence-code lookup needed, unlike the
    ClinGen version below. `points`/`sa` must be row-aligned (e.g. `points`
    from `analysis.results[config]['points']`, `sa` from
    `ms._sample_assignments`)."""
    mat = np.zeros((3, 2), dtype=int)
    plp_mask = sa[:, 0].astype(bool)
    blb_mask = sa[:, 1].astype(bool)
    for mask, col in [(plp_mask, 0), (blb_mask, 1)]:
        idx = np.where(mask)[0]
        for i in idx:
            row = _to_row(int(np.sign(points[i])))
            mat[row, col] += 1
    return mat


def build_mv_clingen_confusion(points, hgvs3_index: Dict, evidence_lookup: Dict,
                                strip_functional_evidence: bool = True) -> np.ndarray:
    """3x2 confusion matrix (rows: MV-predicted P/I/B; cols: ClinGen-truth
    PLP/BLB), ground truth from the SAME `classify_acmg`/PS3-BS3-stripping
    logic `substitute_and_reclassify` uses (deduplicated, not
    `analysis.clingen.build_clingen_confusion`'s heavier variant, which
    needs the production `tree`/`model_selections` harness from
    `analysis.discovery` -- not reused here since this module already has
    everything needed to derive ClinGen's classification from the same
    per-gene evidence-code lookup the VUS-reclassification analysis uses).
    Variants where the ClinGen-derived classification is VUS are excluded
    (no ground-truth column to tally into), matching
    `analysis.clingen.build_clingen_confusion`'s own convention."""
    mat = np.zeros((3, 2), dtype=int)
    for key, i in hgvs3_index.items():
        evidence_str = evidence_lookup.get(key)
        if evidence_str is None or pd.isna(evidence_str):
            continue
        full_list = [c.strip() for c in evidence_str.split(",")]
        if strip_functional_evidence:
            classification, _ = filter_and_recalculate(evidence_str, strip_functional_evidence=True)
        else:
            classification = classify_acmg(full_list)
        if classification in _PLP:
            col = 0
        elif classification in _BLB:
            col = 1
        else:
            continue  # VUS ground truth -- no column to tally into
        row = _to_row(int(np.sign(points[i])))
        mat[row, col] += 1
    return mat


def build_hgvs3_index(ms) -> Dict:
    """Map each kept variant's lookup key to its row index in `ms.scores`.
    `ms.kept_variants` entries are either a genomic 5-tuple (Gene, Chrom,
    hg38_start, ref_allele, alt_allele) -- kept as the raw tuple, matching
    `build_integrated_evidence_lookup`'s keys exactly -- or, when a gene's
    hgvs_c couldn't be parsed to genomic coordinates (confirmed for RET/all
    LABEL-seq genes), a 1-tuple protein-HGVS string like
    ('NP_...:p.Met918Thr',), reduced to just the 'p.Met918Thr' suffix to
    match `build_labelseq_evidence_lookup`'s keys (already in that exact
    format in the LABEL-seq source dataframe's `variant` column)."""
    idx: Dict = {}
    for i, kv in enumerate(ms.kept_variants):
        if isinstance(kv, tuple) and len(kv) == 1 and isinstance(kv[0], str):
            s = kv[0]
            key = s.split(":", 1)[1] if ":" in s else s
        else:
            key = kv
        idx[key] = i
    return idx
