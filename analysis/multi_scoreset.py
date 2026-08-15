"""
Cross-assay variant deduplication for genes with more than one MAVE dataset.

Two independently-run assays for the same gene (e.g. two different MAVE
studies of BRCA1) can each score the same underlying genomic variant.
Section 3's per-dataset confusion-matrix sum in analyze_pipeline_output.py
counts that variant once per assay it happens to appear in -- this module
merges duplicate genomic variants across a gene's assays into one row before
building an aggregate confusion matrix, so that per-assay sum and this
deduplicated view can be compared directly.

Matching key: a pipeline `variant_id` is built as
f"{key}_{Gene}_{Chrom}_{hgvs_c}" (see
src.assay_calibration.pipeline.variant_evidence._get_variant_ids), where
*key* (the MaveDB variant-grouping key, itself possibly containing
underscores) is assay-specific but Gene/Chrom/hgvs_c describe the physical
variant and are shared across any assay using the same transcript/HGVS
convention.

Note hgvs_c routinely contains an underscore itself -- RefSeq transcript
accessions are formatted like "NM_007294.4:c.5565A>T" -- so a blind
`variant_id.rsplit("_", 3)` does NOT correctly isolate Gene/Chrom/hgvs_c in
general (an earlier version of this function did exactly that; it happened
to still produce a correct, collision-free key for every dataset checked
here, because reassembling the mis-split pieces with the same "_" separator
losslessly reconstructs the "{Chrom}_{hgvs_c}" tail whenever hgvs_c has
exactly one internal "_" -- but that's an accident of the specific RefSeq
format, not something to rely on, and it silently breaks for any hgvs_c with
two or more internal underscores). `genomic_variant_key` instead takes the
already-known, reliable `gene` value (derived from the dataset name, not
re-parsed from the id string -- see build_gene_deduped_variants) and splits
on the literal f"_{gene}_" marker, so the recovered "{Chrom}_{hgvs_c}" tail
is correct regardless of how many underscores hgvs_c itself contains. No
separate Chrom column is needed -- the pipeline output doesn't carry one --
so Chrom stays folded into the returned tail rather than parsed out
separately; that's fine here since the key only needs to be a consistent,
collision-free identifier of the physical variant, not a column-by-column
decomposition.

Amino-acid-level (protein-level) assay rows need different treatment. A
single aa-level measurement is routinely stored as multiple master-TSV rows
-- one per degenerate codon that can produce the same amino-acid change
(e.g. one p.Ala7* measurement as 3 rows for stop codons TAA/TAG/TGA) -- and
the pipeline's variant_id (src.assay_calibration.pipeline.variant_evidence.
_get_variant_ids) picks whichever codon happens to be `variants[0]`, an
arbitrary choice among them. Keying an aa-level row's variant_key off that
picked Chrom/hgvs_c (as if it were a real nt-level identity) means
cross-assay matching for aa-level variants rests on that pick being
consistent across every assay/dataset scoring the same amino-acid change --
true in practice today (verified empirically) but not a guaranteed
invariant of the underlying data. `genomic_variant_key` therefore branches
on `nucleotide_or_aa`: aa-level rows are keyed on the real amino-acid
identity (Gene + aa_ref + aa_pos + aa_alt, namespaced with an `_aa_` marker)
instead of the picked codon's Chrom/hgvs_c, while nt-level rows keep the
original Chrom/hgvs_c-tail behavior. This also means an aa-level
measurement and an nt-level measurement of a gene/position are never
merged into the same variant_key -- by design: an aa-level score reflects
whichever underlying nt change(s) produced it, which may not be the same
nt change an nt-level assay separately measured, so aa-level and nt-level
pools stay in disjoint keyspaces rather than being coerced together.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from analysis.plot_common import effective_points as _effective_points, sample_matches


def genomic_variant_key(
    variant_id: str,
    gene: str,
    nucleotide_or_aa: Optional[str] = None,
    aa_ref: Optional[str] = None,
    aa_pos=None,
    aa_alt: Optional[str] = None,
) -> str:
    """Assay-independent identity of the physical variant.

    For nt-level rows (nucleotide_or_aa in {"nt", "dna"}, or unknown --
    the safe fallback when the caller doesn't have these fields): recovers
    f'{Gene}_{Chrom}_{hgvs_c}' (Chrom+hgvs_c kept as one unparsed tail --
    see module docstring) from `variant_id`, exactly as before. Uses the
    LAST occurrence of the literal f"_{gene}_" marker, since the
    assay-specific MaveDB key prefix could in principle (rarely) also
    contain the gene name as a substring. Falls back to the raw variant_id
    (never matches across assays) if the marker isn't found at all -- e.g.
    the "variant_N" fallback used when a Scoreset lacks
    get_variants_by_id/_keep_mask.

    For aa-level rows (nucleotide_or_aa == "aa"): keyed on the real
    amino-acid identity (Gene + aa_ref + aa_pos + aa_alt), NOT on
    variant_id's Chrom/hgvs_c tail -- see module docstring for why that tail
    is an arbitrary, unreliable pick for aa-level measurements. Namespaced
    with an "_aa_" marker so an aa-level key can never collide with an
    nt-level key for the same gene/position -- aa-level and nt-level
    measurements are always kept as separate variants, never merged.
    """
    if nucleotide_or_aa == "aa" and pd.notna(aa_ref) and pd.notna(aa_pos) and pd.notna(aa_alt):
        return f"{gene}_aa_{aa_ref}{int(aa_pos)}{aa_alt}"

    marker = f"_{gene}_"
    idx = str(variant_id).rfind(marker)
    if idx == -1:
        return str(variant_id)
    tail = str(variant_id)[idx + len(marker):]
    return f"{gene}_{tail}"


def _merge_points(points: np.ndarray) -> int:
    """abs-max across assays if every nonzero value shares a sign, else 0 --
    (0, 5) -> 5; (-1, -3) -> -3; (2, -1) -> 0."""
    nonzero = points[points != 0]
    if len(nonzero) == 0:
        return 0
    if (nonzero > 0).all():
        return int(nonzero.max())
    if (nonzero < 0).all():
        return int(nonzero.min())
    return 0


def _merge_author_labels(labels: pd.Series) -> Optional[str]:
    """Same conflict rule as _merge_points, applied to the author's own
    call: every non-indeterminate label across assays must agree, else the
    merged call is indeterminate (None, matching auth_label's own
    "no/ambiguous call" convention)."""
    indeterminate_codes = {"NOT SPECIFIED", "INDETERMINATE", "IGNORE"}
    upper = labels.dropna().astype(str).str.upper()
    determinate = upper[~upper.isin(indeterminate_codes)]
    if determinate.empty:
        return None
    if determinate.nunique() == 1:
        return determinate.iloc[0]
    return None


def build_gene_deduped_variants(df_primary: pd.DataFrame, use_oob: bool = True) -> pd.DataFrame:
    """One row per (gene, genomic_variant_key), merging every assay/dataset
    that scored the same physical variant for the same gene.

    `df_primary` should already be filtered to one method (the "primary"
    ExCALIBR method, same convention as the rest of
    analyze_pipeline_output.py) and carry `sample`/`variant_id`/
    `standard_points` (+ `oob_points` if `use_oob`) /`auth_label`/`is_vus`.

    Returns one row per unique (gene, variant_key) with columns: gene,
    variant_key, sample (ClinVar label -- first non-null copy across
    assays; these should already agree since ClinVar truth doesn't depend
    on the assay), points (merged per _merge_points), auth_label (merged
    per _merge_author_labels), is_vus (True if any assay's copy was
    is_vus -- again ClinVar-based so should already agree; `any()` is just
    a defensive OR), n_assays (how many distinct datasets contributed --
    1 means this variant wasn't actually duplicated across assays).
    """
    df = df_primary.copy()
    df["gene"] = df["dataset"].str.split("_").str[0]
    has_aa_fields = all(c in df.columns for c in ("nucleotide_or_aa", "aa_ref", "aa_pos", "aa_alt"))
    if has_aa_fields:
        df["variant_key"] = [
            genomic_variant_key(vid, gene, noa, ar, ap, aa)
            for vid, gene, noa, ar, ap, aa in zip(
                df["variant_id"], df["gene"], df["nucleotide_or_aa"],
                df["aa_ref"], df["aa_pos"], df["aa_alt"],
            )
        ]
    else:
        df["variant_key"] = [
            genomic_variant_key(vid, gene) for vid, gene in zip(df["variant_id"], df["gene"])
        ]
    df["points"] = _effective_points(df, use_oob, label="gene_dedup", context="all")

    rows = []
    for (gene, vkey), grp in df.groupby(["gene", "variant_key"], sort=False):
        sample_vals = grp["sample"].dropna()
        rows.append({
            "gene": gene,
            "variant_key": vkey,
            "sample": sample_vals.iloc[0] if len(sample_vals) else None,
            "points": _merge_points(grp["points"].to_numpy()),
            "auth_label": _merge_author_labels(grp["auth_label"]) if "auth_label" in grp.columns else None,
            "is_vus": bool(grp["is_vus"].fillna(False).astype(bool).any()) if "is_vus" in grp.columns else False,
            "n_assays": grp["dataset"].nunique(),
        })
    return pd.DataFrame(rows)


def build_deduped_confusion_matrix(deduped: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Same [BLB,PLP] x [Normal,IR,Abnormal] shape as
    analysis.confusion.build_confusion_matrix, built from
    build_gene_deduped_variants' merged `points` column instead of a live
    effective_points call (dedup/merge already happened)."""
    df_plp = deduped[sample_matches(deduped, "Pathogenic/Likely Pathogenic")]
    df_blb = deduped[sample_matches(deduped, "Benign/Likely Benign")]
    if df_plp.empty and df_blb.empty:
        return None

    def _counts(sub):
        pts = sub["points"].to_numpy()
        return [int((pts < 0).sum()), int((pts == 0).sum()), int((pts > 0).sum())]

    return pd.DataFrame(
        [_counts(df_blb), _counts(df_plp)], index=["BLB", "PLP"], columns=["Normal", "IR", "Abnormal"],
    )


def build_deduped_author_confusion_matrix(deduped: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Author-side counterpart of build_deduped_confusion_matrix, from the
    merged `auth_label` column -- same all-indeterminate guard as
    analysis.confusion.build_author_confusion_matrix (returns None rather
    than an all-IR matrix when no variant in scope has a determinate merged
    author call)."""
    df_plp = deduped[sample_matches(deduped, "Pathogenic/Likely Pathogenic")]
    df_blb = deduped[sample_matches(deduped, "Benign/Likely Benign")]
    if df_plp.empty and df_blb.empty:
        return None

    indeterminate_codes = {"NOT SPECIFIED", "INDETERMINATE", "IGNORE"}

    def _counts(sub):
        upper = sub["auth_label"].astype(str).str.upper()
        norm = int((upper == "NORMAL").sum())
        abnorm = int((upper == "ABNORMAL").sum())
        ir = int((upper.isin(indeterminate_codes) | sub["auth_label"].isna()).sum())
        return [norm, ir, abnorm]

    mat = pd.DataFrame(
        [_counts(df_blb), _counts(df_plp)], index=["BLB", "PLP"], columns=["Normal", "IR", "Abnormal"],
    )
    if (mat["Normal"].sum() + mat["Abnormal"].sum()) == 0:
        return None
    return mat


def restrict_to_genes_with_author_data(deduped: pd.DataFrame) -> pd.DataFrame:
    """Drop every gene whose merged author calls (across ALL its deduped
    variants) are entirely indeterminate/missing -- same rationale as
    analysis.confusion.build_author_confusion_matrix returning None for a
    dataset with zero determinate author calls: a gene where the author
    functional classification was simply never recorded shouldn't count as
    "author called everything indeterminate" in the gene-deduplicated
    ExCALIBR-vs-author comparison. Genes with at least one determinate
    merged author call are kept in full (including their own indeterminate
    variants)."""
    indeterminate_codes = {"NOT SPECIFIED", "INDETERMINATE", "IGNORE"}

    def _has_determinate_author(labels: pd.Series) -> bool:
        upper = labels.dropna().astype(str).str.upper()
        return bool((~upper.isin(indeterminate_codes)).any())

    genes_with_author_data = deduped.groupby("gene")["auth_label"].apply(_has_determinate_author)
    keep_genes = set(genes_with_author_data[genes_with_author_data].index)
    return deduped[deduped["gene"].isin(keep_genes)]
