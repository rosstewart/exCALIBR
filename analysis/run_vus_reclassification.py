#!/usr/bin/env python
"""
Runner for the VCEP VUS-reclassification analysis (see analysis/
vus_reclassification.py's module docstring and the approved plan at
~/.claude/plans/can-you-give-a-synchronous-nova.md for full context).

Canonical v3 only -- no all_assayed staged-init involvement.

Usage
-----
    python analysis/run_vus_reclassification.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gzip
import json
import numpy as np
import pandas as pd

from src.assay_calibration.multivariate_data.labelseq import build_labelseq_dataframe, build_labelseq_multiscoresets
from mv_analysis.integrated_gene_data import build_integrated_multiscoresets, DEFAULT_DATA_PATH as INTEGRATED_DATA_PATH
from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
from mv_analysis.report import build_comparison_table
from mv_analysis.gene_performance_scatter import fast_results_json, _find_dataset_key
from analysis.vus_reclassification import (
    build_hgvs3_index, substitute_and_reclassify,
    build_labelseq_evidence_lookup, build_integrated_evidence_lookup,
)

OUTPUT_DIR = "/data/ross/assay_calibration/multivariate/experimental_staged_fit"
CANONICAL_RESULTS = "/data/ross/assay_calibration/multivariate/jobs_all_1000b_8f_v2/bootstrap_results_v3.json.gz"

RUN_KWARGS = dict(path_percentile=5, min_valid_boots=1, reestimate_marginal_weights=False,
                  enforce_marginal_monotonicity=False, liberal_marginal_monotonicity=False)

RASOPATHY_GENES = ["braf", "craf", "kras", "mek1", "mek2", "shp2", "sos1", "sos2", "mras"]
INTEGRATED_GENES = ["TP53", "LDLR", "GCK", "F9", "PTEN", "BRCA2", "BRCA1", "PALB2", "KCNQ4"]


def _pick_best_canonical_config(gene, dataset_name, ms):
    with fast_results_json(CANONICAL_RESULTS):
        with gzip.open(CANONICAL_RESULTS, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        if dataset_name not in raw:
            print(f"  SKIP {gene}: '{dataset_name}' not in canonical results", flush=True)
            return None, None
        configs = sorted(raw[dataset_name]["0"].keys())
        table, _ = build_comparison_table(
            gene.lower(), "labelseq", ms, CANONICAL_RESULTS, dataset_name=dataset_name,
            modes=["trust_global"], compare_uv=False, **RUN_KWARGS,
        )
        clinical = table[table["threshold"].str.startswith("clinical")]
        best_config, best_mcc = None, -1
        for cfg in configs:
            row = clinical[clinical["config"] == cfg]
            if row.empty:
                continue
            mcc = row["mcc"].max()
            if mcc is not None and mcc > best_mcc:
                best_mcc, best_config = mcc, cfg
        return best_config or configs[0], best_mcc


def process_gene(gene, gene_set, ms, dataset_name, evidence_lookup):
    """`evidence_lookup`: dict mapping the SAME key shape ms.kept_variants uses
    (protein-HGVS tail string for LABEL-seq genes, genomic 5-tuple for
    integrated genes) -> evidence-code string."""
    best_config, best_mcc = _pick_best_canonical_config(gene, dataset_name, ms)
    if best_config is None:
        return pd.DataFrame()

    analysis = build_gene_set_analysis(ms, gene.lower(), CANONICAL_RESULTS, dataset_name=dataset_name)
    analysis.run(partial_pattern_mode="trust_global", **RUN_KWARGS)
    if analysis.results.get(best_config) is None:
        print(f"  SKIP {gene}: no valid bootstraps for config {best_config}", flush=True)
        return pd.DataFrame()
    points = np.asarray(analysis.results[best_config]["points"], dtype=float)

    rows = []
    hgvs3_index = build_hgvs3_index(ms)
    for key, i in hgvs3_index.items():
        evidence_str = evidence_lookup.get(key)
        if evidence_str is None or pd.isna(evidence_str):
            continue
        result = substitute_and_reclassify(evidence_str, points[i])
        rows.append({"gene": gene, "gene_set": gene_set, "variant_key": key,
                     "evidence_string": evidence_str, "our_points": points[i],
                     "canonical_config": best_config, **result})
    print(f"  {gene}: {len(rows)} variants aligned to evidence codes "
          f"(config={best_config}, mcc={best_mcc})", flush=True)
    return pd.DataFrame(rows)


def main():
    all_rows = []

    print("=== RASopathy LABEL-seq genes ===", flush=True)
    df_labelseq = build_labelseq_dataframe()
    for gene in RASOPATHY_GENES:
        evidence_lookup = build_labelseq_evidence_lookup(gene, df_labelseq)
        if not evidence_lookup:
            print(f"  SKIP {gene}: no evidence-code rows", flush=True)
            continue
        ms = build_labelseq_multiscoresets(genes=[gene])[gene.lower()]
        dataset_name = f"{gene.lower()}_labelseq_mv"
        try:
            df_result = process_gene(gene, "labelseq", ms, dataset_name, evidence_lookup)
        except Exception as e:
            print(f"  SKIP {gene}: process_gene failed ({e})", flush=True)
            continue
        all_rows.append(df_result)

    print("\n=== Integrated-set genes ===", flush=True)
    df_integrated = pd.read_csv(INTEGRATED_DATA_PATH, sep="\t", low_memory=False,
                                 usecols=["Gene", "Chrom", "hg38_start", "ref_allele", "alt_allele",
                                          "Applied Evidence Codes (Met)_ClinGen_repo"])
    for gene in INTEGRATED_GENES:
        evidence_lookup = build_integrated_evidence_lookup(gene, df_integrated)
        if not evidence_lookup:
            print(f"  SKIP {gene}: no evidence-code rows", flush=True)
            continue
        ms = build_integrated_multiscoresets(genes=[gene])[gene]
        try:
            dataset_name = _find_dataset_key(CANONICAL_RESULTS, gene)
        except KeyError as e:
            print(f"  SKIP {gene}: {e}", flush=True)
            continue
        try:
            df_result = process_gene(gene, "integrated", ms, dataset_name, evidence_lookup)
        except Exception as e:
            print(f"  SKIP {gene}: process_gene failed ({e})", flush=True)
            continue
        all_rows.append(df_result)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(f"{OUTPUT_DIR}/vus_reclassification_variants.csv", index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_DIR}/vus_reclassification_variants.csv", flush=True)

    if combined.empty:
        print("No data -- nothing to summarize.", flush=True)
        return

    print("\n=== Per-gene summary ===", flush=True)
    summary_rows = []
    for gene, grp in combined.groupby("gene"):
        n_total = len(grp)
        n_orig_determinate = (grp["original_class"] != "VUS").sum()
        n_pivotal = (grp["residual_class"] == "VUS").sum()
        n_resolved = grp["resolved"].sum()
        concordant_known = grp["concordant"].dropna()
        concordance_rate = concordant_known.mean() if len(concordant_known) else float("nan")
        summary_rows.append({
            "gene": gene, "n_variants": n_total, "n_originally_determinate": n_orig_determinate,
            "n_ps3bs3_pivotal": n_pivotal, "n_resolved_by_our_evidence": n_resolved,
            "concordance_rate": concordance_rate, "n_concordance_known": len(concordant_known),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False), flush=True)
    summary_df.to_csv(f"{OUTPUT_DIR}/vus_reclassification_summary.csv", index=False)

    n_pivotal_total = (combined["residual_class"] == "VUS").sum()
    n_resolved_total = combined["resolved"].sum()
    concordant_all = combined["concordant"].dropna()
    print(f"\n=== Aggregate ===", flush=True)
    print(f"Total variants w/ evidence codes: {len(combined)}", flush=True)
    print(f"PS3/BS3-pivotal (VUS after stripping): {n_pivotal_total}", flush=True)
    print(f"Resolved by our canonical v3 evidence: {n_resolved_total} "
          f"({100*n_resolved_total/n_pivotal_total:.1f}% of pivotal)" if n_pivotal_total else "", flush=True)
    print(f"Concordance rate (resolved direction matches original ClinGen call): "
          f"{concordant_all.mean():.3f} (n={len(concordant_all)})" if len(concordant_all) else "N/A", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
