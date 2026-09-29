#!/usr/bin/env python
"""
True ClinVar-VUS points, split by panel (functional / combined) -- for every
variant genuinely labeled VUS by ClinVar (clinvar_sig_2026 == "Uncertain
significance"; deliberately NO star/quality filter -- see note below), pull
OUR calibrated MV points for that variant, one CSV per panel.

Extends analysis/compute_true_clinvar_vus_points.py's LABEL-seq-only scope to
also cover the plain-"integrated" functional genes and the "combined"
(functional+predictor) gene-set, resolves against the current paper fits
(config.PAPER_RESULTS_JSON) instead of the old pre-EM-fix canonical results,
and adds a `gene_set` column so downstream Sankey plotting can facet by panel.

This is NOT analysis/run_vus_reclassification.py's ACMG evidence-code
stripping/substitution exercise -- there is no PS3/BS3 stripping or
ClinGen-VCEP-evidence-code involvement here at all. This is simply: real
ClinVar VUS -> our points -> where does it land.

No star/quality filter on VUS (deliberate divergence from
compute_true_clinvar_vus_points.py, whose docstring claims this matches
`Variant.is_vus` but doesn't quite -- see below): the zero-star filter exists
to ensure P/LP/B/LB used as calibration/control anchors are reliably
reviewed. It doesn't apply here -- VUS isn't a calibration anchor, it's the
population whose reclassification this script measures, and filtering to
reviewed-only would systematically exclude the least-characterized variants,
exactly the ones where added evidence is most likely to matter.

Also NOT `Variant.is_vus` (dataset.py:1538, hardcoded to clinvar_sig_2025/
sufficient_quality_2025) -- that hardcoding's own comment ("VUS ALWAYS 2025
(pillar project) SINCE NOT CONTROL") is guarding against an older pillar-
project path using clinvar_release=2018 for controls, not asserting VUS
should differ from whatever release the current pipeline uses. This script
hardcodes clinvar_sig_2026/clinvar_star_2026 directly, matching the release
config.PAPER_RESULTS_JSON's fits actually use throughout.

Panel scope (see plan discussion -- these are real, confirmed limitations,
not oversights):
  - "functional": LABEL-seq genes + plain-"integrated" genes only. TP53/
    CARD11/FGFR are excluded -- their MultiScoreset builders
    (build_tp53_multiscoreset/build_card11_multiscoreset/
    build_fgfr_multiscoresets) don't support regularization_type="all_assayed"
    (confirmed via mv_analysis.config.GENE_SETS_SUPPORTING_ALL_ASSAYED, which
    explicitly excludes them: "predictor/combined/card11/tp53/fgfr retain no
    unlabeled bulk at ingestion"), so VUS rows are unconditionally dropped
    before a Scoreset is even built for those three. Extending each of those
    three separate gene-set modules is real additional scope, deliberately
    deferred.
  - "combined": functional+predictor merge for the 8 PREDICTOR_COMBINED_GENES,
    built the same way gene_performance_scatter.py's build_panel_c does
    (calling build_functional_scoresets/build_combined_multiscoreset
    directly, NOT through config.build_multiscoresets_for_gene_set("combined",
    ...), which explicitly rejects regularization_type at the dispatch level).
    Predictor dimensions are NaN for VUS rows the functional side retains but
    the predictor side has no data for (BasicScoreset/df_to_basic_scoreset
    never carries a VUS bulk at all -- confirmed no ClinVar-labeled VUS
    predictor-score population exists today, see "predictor"-panel note
    below).
  - "predictor" (predictor-only): built by retaining true-VUS rows at the
    data layer rather than modifying predictors.py. The per-predictor CSVs
    (BasicScoreset's source) are a curated labeled+gnomAD subset -- most true
    VUS aren't in them at all (confirmed: only 85/442 BRCA1 VUS are present,
    all already carrying sample_assignments="2"/gnomAD from ordinary
    population overlap, not VUS-specific inclusion). But the SAME integrated
    dataframe used for functional/combined already carries the raw REVEL/
    AM_score/MutPred2 columns directly, for every variant regardless of
    label -- verified bit-identical (max abs diff 0.0 over 522 overlapping
    BRCA1/AlphaMissense variants) to the curated CSV's own score column, so
    no rescaling is needed. For each true-VUS variant with a value in a
    given predictor's column (and not already present in that predictor's
    curated CSV), an extra row is appended with sample_assignments="2"
    (gnomAD) before building the BasicMultiScoreset -- scoring is a pure
    function of the row's score vector via the already-fitted model,
    independent of which role a row carries (same reasoning already
    validated for functional/combined's all_assayed retention), so marking
    these as gnomAD rather than inventing a new role is safe and requires no
    predictors.py changes at all.

Usage
-----
    python analysis/compute_clinvar_vus_points_by_panel.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis.mv_cockpit import pick_best_canonical_config, DEFAULT_RUN_KWARGS
from mv_analysis.gene_performance_scatter import fast_results_json, PLAIN_INTEGRATED_GENES
from mv_analysis.analyze_mv_pipeline_output import PREDICTOR_COMBINED_GENES
from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
from src.assay_calibration.multivariate_data.common import resolve_clinvar_release
from src.assay_calibration.multivariate_data.combined import (
    build_functional_scoresets, build_combined_multiscoreset,
    get_functionally_assayed_protein_variants, DEFAULT_INTEGRATED_DATAFRAME,
)
from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
from src.assay_calibration.multivariate_data.predictors import (
    PREDICTORS, load_predictor_data, build_basic_multi_scoreset,
)

OUTPUT_DIR = Path(config.PAPER_VUS_RECLASSIFICATION_DIR)
RESULTS_JSON = config.PAPER_RESULTS_JSON
RUN_KWARGS = dict(DEFAULT_RUN_KWARGS)

# LABEL-seq's own gene set (lowercase, matching config.LABELSEQ_GENES) +
# plain-integrated genes (uppercase, matching PLAIN_INTEGRATED_GENES) --
# see module docstring for why TP53/CARD11/FGFR aren't included.
_LABELSEQ_GENES = list(config.LABELSEQ_GENES)
_INTEGRATED_GENES = list(PLAIN_INTEGRATED_GENES)
_COMBINED_GENES = list(PREDICTOR_COMBINED_GENES)
_PREDICTOR_GENES = list(PREDICTOR_COMBINED_GENES)

# predictor short-name (predictors.PREDICTORS) -> integrated dataframe's own
# raw-score column name (confirmed present: REVEL, AM_score, MutPred2 --
# NOT PREDICTOR_DATASET_NAMES' display names, which don't match these).
_PREDICTOR_TO_INTEGRATED_COLUMN = {"REVEL": "REVEL", "MP2": "MutPred2", "AM": "AM_score"}


def true_vus_mask(df_gene):
    return df_gene["clinvar_sig_2026"] == "Uncertain significance"


def build_hgvs3_index(ms):
    """LABEL-seq's key convention: ms.kept_variants are "{refseq}:p.Xxx###Yyy"
    strings, OR occasionally 1-tuples wrapping that same string (confirmed
    for MRAS/RET: ('NP_001078518.1:p.Ala125Ter',) -- inconsistent across
    genes, so both forms must be handled) -- strip the "{refseq}:" prefix so
    it matches the raw dataframe's own "variant" column format directly."""
    idx = {}
    for i, kv in enumerate(ms.kept_variants):
        s = kv[0] if isinstance(kv, tuple) else kv
        idx[s.split(":", 1)[1] if ":" in s else s] = i
    return idx


def labelseq_vus_keys(df_gene):
    """Query keys already in LABEL-seq's index format -- just the raw
    "variant" column, no transform needed."""
    return df_gene.loc[true_vus_mask(df_gene), "variant"].drop_duplicates().tolist()


def integrated_vus_keys(gene, df_gene):
    """Plain-"integrated" genes' key convention: ms.kept_variants are
    (gene, chrom_str, start_str, ref, alt) tuples (confirmed via a real
    BRCA1 ms) -- build matching query tuples directly from the raw
    dataframe's Chrom/hg38_start/ref_allele/alt_allele columns (same columns
    uv_sources.load_combined_uv_points already bridges on)."""
    sub = df_gene[true_vus_mask(df_gene)]
    keys = []
    for row in sub.itertuples(index=False):
        chrom, start = getattr(row, "Chrom", None), getattr(row, "hg38_start", None)
        ref, alt = getattr(row, "ref_allele", None), getattr(row, "alt_allele", None)
        if pd.notna(chrom) and pd.notna(start) and pd.notna(ref) and pd.notna(alt):
            keys.append((gene, str(chrom), str(int(start)), ref, alt))
    return sorted(set(keys))


def combined_vus_keys(df_gene):
    """"combined" gene-set's key convention: ms.kept_variants are plain
    protein-variant strings like "A102G" (no gene prefix, no "p." prefix --
    confirmed via a real BRCA1 combined ms) -- build matching query strings
    directly from aa_ref/aa_pos/aa_alt, same convention
    uv_sources.load_combined_uv_points already uses for its own bridging."""
    sub = df_gene[true_vus_mask(df_gene)]
    keys = []
    for row in sub.itertuples(index=False):
        aa_ref, aa_pos, aa_alt = getattr(row, "aa_ref", None), getattr(row, "aa_pos", None), getattr(row, "aa_alt", None)
        if pd.notna(aa_ref) and pd.notna(aa_pos) and pd.notna(aa_alt):
            keys.append(f"{aa_ref}{int(aa_pos)}{aa_alt}")
    return sorted(set(keys))


def score_gene(gene, gene_set, ms, dataset_name, vus_keys, variant_index=None):
    """Shared scoring tail: pick best-MCC config, run the MV analysis, pull
    points for the VUS rows. `variant_index`, if given, must already be keyed
    in ms.kept_variants' own native format (see labelseq/integrated/combined_
    vus_keys' docstrings for what that format is per gene-set) -- defaults to
    a plain 1:1 index over ms.kept_variants (correct for integrated/combined,
    which don't need build_hgvs3_index's gene-prefix stripping).
    Returns a list of row-dicts (possibly empty)."""
    best_config = pick_best_canonical_config(
        gene, gene_set, ms, RESULTS_JSON, dataset_name, RUN_KWARGS)
    if best_config is None:
        print(f"  SKIP {gene}: no canonical config resolved", flush=True)
        return []

    analysis = build_gene_set_analysis(ms, gene.lower(), RESULTS_JSON, dataset_name=dataset_name, gene_set=gene_set)
    with fast_results_json(RESULTS_JSON):
        analysis.run(partial_pattern_mode="trust_global", **RUN_KWARGS)
    if analysis.results.get(best_config) is None:
        print(f"  SKIP {gene}: no valid bootstraps for config {best_config}", flush=True)
        return []
    points = np.asarray(analysis.results[best_config]["points"], dtype=float)

    if variant_index is None:
        variant_index = {kv: i for i, kv in enumerate(ms.kept_variants)}

    rows, n_matched = [], 0
    for v in vus_keys:
        idx = variant_index.get(v)
        if idx is None:
            continue
        n_matched += 1
        rows.append({"gene": gene, "gene_set": gene_set, "variant_key": str(v),
                      "our_points": points[idx], "canonical_config": best_config})
    print(f"  {gene}: {len(vus_keys)} true ClinVar VUS, {n_matched} matched "
          f"(config={best_config})", flush=True)
    return rows


def run_functional_panel(out_path):
    """LABEL-seq + plain-integrated genes only -- see module docstring.
    Writes out_path incrementally (after each gene) so a later crash doesn't
    lose already-scored genes."""
    from src.assay_calibration.multivariate_data.labelseq import build_labelseq_dataframe
    from mv_analysis.integrated_gene_data import build_integrated_multiscoresets

    rows = []

    def _flush():
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print("=== functional panel: LABEL-seq genes ===", flush=True)
    df_labelseq = build_labelseq_dataframe()
    for gene in _LABELSEQ_GENES:
        df_gene = df_labelseq[df_labelseq["Gene"].str.lower() == gene]
        vus_keys = labelseq_vus_keys(df_gene)
        if not vus_keys:
            print(f"  {gene}: 0 true ClinVar VUS variants", flush=True)
            continue
        try:
            ms = build_labelseq_multiscoresets(genes=[gene], regularization_type="all_assayed")[gene.lower()]
            dataset_name = config.canonical_dataset_name(gene, "labelseq", results_json=RESULTS_JSON)
            rows.extend(score_gene(gene, "labelseq", ms, dataset_name, vus_keys,
                                    variant_index=build_hgvs3_index(ms)))
        except Exception as e:
            print(f"  SKIP {gene}: {e}", flush=True)
        _flush()

    print("\n=== functional panel: plain-integrated genes ===", flush=True)
    df_integrated = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
    for gene in _INTEGRATED_GENES:
        df_gene = df_integrated[df_integrated["Gene"] == gene]
        vus_keys = integrated_vus_keys(gene, df_gene)
        if not vus_keys:
            print(f"  {gene}: 0 true ClinVar VUS variants", flush=True)
            continue
        try:
            ms_map = build_integrated_multiscoresets(genes=[gene], regularization_type="all_assayed")
            ms = ms_map.get(gene)
            if ms is None:
                print(f"  SKIP {gene}: no integrated ms built", flush=True)
                continue
            dataset_name = config.canonical_dataset_name(gene, "integrated", results_json=RESULTS_JSON)
            rows.extend(score_gene(gene, "integrated", ms, dataset_name, vus_keys))
        except Exception as e:
            print(f"  SKIP {gene}: {e}", flush=True)
        _flush()

    return pd.DataFrame(rows)


def predictor_vus_rows(df_gene, predictor):
    """{"protein_variant": ..., "score": ...} rows for true-VUS variants that
    have a value in `predictor`'s integrated-dataframe score column --
    caller filters out any already present in that predictor's curated CSV
    before concatenating."""
    col = _PREDICTOR_TO_INTEGRATED_COLUMN[predictor]
    sub = df_gene[true_vus_mask(df_gene) & df_gene[col].notna()]
    sub = sub.dropna(subset=["aa_ref", "aa_pos", "aa_alt"])
    if sub.empty:
        return pd.DataFrame(columns=["protein_variant", "score", "sample_assignments"])
    variant = sub["aa_ref"] + sub["aa_pos"].astype(int).astype(str) + sub["aa_alt"]
    out = pd.DataFrame({"protein_variant": variant, "score": sub[col].astype(float)})
    out["sample_assignments"] = "2"  # gnomAD role -- see module docstring's "predictor" note
    return out.drop_duplicates("protein_variant")


def run_predictor_panel(out_path):
    """8 PREDICTOR_COMBINED_GENES -- see module docstring's "predictor" note
    for how true-VUS rows are retained without touching predictors.py."""
    rows = []

    def _flush():
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print("=== predictor panel ===", flush=True)
    df_integrated = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
    for gene in _PREDICTOR_GENES:
        df_gene = df_integrated[df_integrated["Gene"] == gene]
        vus_keys = combined_vus_keys(df_gene)  # same "A102G" key convention
        if not vus_keys:
            print(f"  {gene}: 0 true ClinVar VUS variants", flush=True)
            continue

        try:
            by_gene = load_predictor_data(config.PREDICTOR_RAW_DATA_DIR, genes=[gene])
            predictor_dfs = dict(by_gene.get(gene, {}))
            if len(predictor_dfs) < len(PREDICTORS):
                print(f"  SKIP {gene}: missing predictor(s) "
                      f"{set(PREDICTORS) - set(predictor_dfs)}", flush=True)
                continue

            extended = {}
            for predictor, df_pred in predictor_dfs.items():
                extra = predictor_vus_rows(df_gene, predictor)
                extra = extra[~extra["protein_variant"].isin(set(df_pred["protein_variant"]))]
                extended[predictor] = pd.concat([df_pred, extra], ignore_index=True)

            ms, info = build_basic_multi_scoreset(gene, extended)
            if ms is None:
                print(f"  SKIP {gene}: could not build predictor ms ({info})", flush=True)
                continue
            dataset_name = config.canonical_dataset_name(gene, "predictor", results_json=RESULTS_JSON)
            rows.extend(score_gene(gene, "predictor-mv", ms, dataset_name, vus_keys))
        except Exception as e:
            print(f"  SKIP {gene}: {e}", flush=True)
        _flush()

    return pd.DataFrame(rows)


def run_combined_panel(out_path):
    """8 PREDICTOR_COMBINED_GENES, functional+predictor merge -- see module
    docstring for why this bypasses config.build_multiscoresets_for_gene_set
    and calls the underlying builders directly instead. Writes out_path
    incrementally (after each gene).

    require_both_modalities is left at its default (True, see combined.py)
    deliberately: regardless of what the stored "combined" fit was itself
    trained on, EVERY evaluation of it -- VUS reclassification here, the
    gene-performance scatter, results tables, confusion matrices -- must be
    restricted to variants observed in at least one predictor dim AND at
    least one functional dim. Scoring a "combined" model on a single-
    modality variant (which is really just a predictor-only or
    functional-only observation wearing a "combined" label) isn't a
    meaningful evaluation of cross-modality calibration; VUS not surviving
    this restriction are naturally dropped by score_gene's variant_index
    lookup (absent key -> None), no extra masking needed here."""
    rows = []

    def _flush():
        pd.DataFrame(rows).to_csv(out_path, index=False)

    print("=== combined panel (both-modalities-only, see docstring) ===", flush=True)
    df_integrated = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
    for gene in _COMBINED_GENES:
        df_gene = df_integrated[df_integrated["Gene"] == gene]
        vus_keys = combined_vus_keys(df_gene)
        if not vus_keys:
            print(f"  {gene}: 0 true ClinVar VUS variants", flush=True)
            continue

        try:
            datasets = sorted(df_gene["Dataset"].unique())
            if not datasets:
                print(f"  {gene}: no functional datasets, skipping", flush=True)
                continue
            clinvar_release = resolve_clinvar_release(gene)
            functional_scoresets = build_functional_scoresets(
                df_integrated, gene, datasets, clinvar_release=clinvar_release,
                regularization_type="all_assayed")
            functionally_assayed = get_functionally_assayed_protein_variants(df_integrated, gene, datasets)
            ms = build_combined_multiscoreset(
                gene, functional_scoresets, datasets, config.PREDICTOR_RAW_DATA_DIR,
                functionally_assayed_variants=functionally_assayed)
            if ms is None:
                print(f"  SKIP {gene}: could not build combined ms", flush=True)
                continue
            dataset_name = config.canonical_dataset_name(gene, "combined", results_json=RESULTS_JSON)
            rows.extend(score_gene(gene, "combined", ms, dataset_name, vus_keys))
        except Exception as e:
            print(f"  SKIP {gene}: {e}", flush=True)
        _flush()

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_functional = OUTPUT_DIR / "functional_true_clinvar_vus_points.csv"
    df_functional = run_functional_panel(out_functional)
    print(f"\nSaved {len(df_functional)} rows to {out_functional}", flush=True)

    out_predictor = OUTPUT_DIR / "predictor_true_clinvar_vus_points.csv"
    df_predictor = run_predictor_panel(out_predictor)
    print(f"Saved {len(df_predictor)} rows to {out_predictor}", flush=True)

    out_combined = OUTPUT_DIR / "combined_true_clinvar_vus_points.csv"
    df_combined = run_combined_panel(out_combined)
    print(f"Saved {len(df_combined)} rows to {out_combined}", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
