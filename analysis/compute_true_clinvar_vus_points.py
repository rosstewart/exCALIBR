#!/usr/bin/env python
"""
True ClinVar-VUS points: for each of the 9 RASopathy LABEL-seq genes, find
every variant genuinely labeled VUS by ClinVar (via the SAME criterion
`Variant.is_vus` uses in `src/assay_calibration/data_utils/dataset.py:1538` --
`clinvar_sig_2026 == "Uncertain significance"` AND sufficient review quality,
i.e. `clinvar_star_2026` not in the zero-star statuses -- independent of
whether that variant has any ClinGen VCEP "Applied Evidence Codes" at all),
then pull OUR canonical v3 points for those variants.

This is DIFFERENT from (and a superset scope of) analysis/
run_vus_reclassification.py's `original_class` column, which only covers the
small subset of variants that happen to have ClinGen evidence codes recorded,
and derives its class by re-running `classify_acmg` on those codes rather
than reading ClinVar's own asserted significance -- conflating the two was
a mistake in the earlier version of this analysis (flagged by the user).

Usage
-----
    python analysis/compute_true_clinvar_vus_points.py
"""
import sys
import gzip
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.assay_calibration.multivariate_data.labelseq import build_labelseq_dataframe, build_labelseq_multiscoresets
from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
from mv_analysis.report import build_comparison_table
from mv_analysis.gene_performance_scatter import fast_results_json

OUTPUT_DIR = "/data/ross/assay_calibration/multivariate/experimental_staged_fit"
CANONICAL_RESULTS = "/data/ross/assay_calibration/multivariate/jobs_all_1000b_8f_v2/bootstrap_results_v3.json.gz"

RUN_KWARGS = dict(path_percentile=5, min_valid_boots=1, reestimate_marginal_weights=False,
                  enforce_marginal_monotonicity=False, liberal_marginal_monotonicity=False)

# All 17 LABEL-seq genes -- unlike analysis/run_vus_reclassification.py (which
# needs ClinGen "Applied Evidence Codes" and so is restricted to the 9 genes
# that have any), this script only needs clinvar_sig_2026/clinvar_star_2026 +
# a canonical v3 fit, which every LABEL-seq gene has (confirmed: all 17
# "{gene}_labelseq_mv" keys are present in bootstrap_results_v3.json.gz).
LABELSEQ_GENES = ["araf", "braf", "craf", "egfr", "erbb2", "grb2", "kras", "ksr1", "ksr2",
                  "mek1", "mek2", "met", "mras", "ret", "shp2", "sos1", "sos2"]

_ZERO_STAR_STATUSES = {
    np.nan, "no assertion criteria provided", "no assertion provided",
    "no interpretation for the single variant", "no classification provided", "-",
}


def build_hgvs3_index(ms):
    idx = {}
    for i, kv in enumerate(ms.kept_variants):
        s = kv[0] if isinstance(kv, tuple) else kv
        idx[s.split(":", 1)[1] if ":" in s else s] = i
    return idx


def _pick_best_canonical_config(gene, dataset_name, ms):
    with fast_results_json(CANONICAL_RESULTS):
        with gzip.open(CANONICAL_RESULTS, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        if dataset_name not in raw:
            print(f"  SKIP {gene}: '{dataset_name}' not in canonical results", flush=True)
            return None
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
        return best_config or configs[0]


def main():
    print("Loading LABEL-seq source dataframe...", flush=True)
    df_labelseq = build_labelseq_dataframe()

    rows = []
    for gene in LABELSEQ_GENES:
        df_gene = df_labelseq[df_labelseq["Gene"].str.lower() == gene]
        is_vus = (
            (df_gene["clinvar_sig_2026"] == "Uncertain significance")
            & (~df_gene["clinvar_star_2026"].isin(_ZERO_STAR_STATUSES))
        )
        vus_variants = df_gene.loc[is_vus, "variant"].drop_duplicates()
        if vus_variants.empty:
            print(f"  {gene}: 0 true ClinVar VUS variants", flush=True)
            continue

        # regularization_type="all_assayed" is required here, not the default
        # build: `Scoreset`'s keep_mask is `sample_assignments.any(axis=1)`
        # (src/assay_calibration/data_utils/dataset.py:1156) -- a variant
        # survives only if it's P/LP, B/LB, gnomAD-population, or Synonymous.
        # A "pure" VUS with none of those roles is PERMANENTLY DROPPED from
        # the default build's kept_variants (see the comment right above that
        # line: "drops unlabeled/VUS-only rows"). all_assayed adds a 5th
        # always-true role, bypassing that filter so every assayed variant
        # (VUS included) survives -- component_params from the canonical fit
        # can still score these rows regardless of which ms-build container
        # they came from (scoring is per-row, independent of fit-time
        # row-inclusion), same reasoning already validated this session for
        # the all_assayed staged-init experiments.
        ms = build_labelseq_multiscoresets(genes=[gene], regularization_type="all_assayed")[gene.lower()]
        hgvs3_index = build_hgvs3_index(ms)
        dataset_name = f"{gene.lower()}_labelseq_mv"
        best_config = _pick_best_canonical_config(gene, dataset_name, ms)
        if best_config is None:
            continue

        analysis = build_gene_set_analysis(ms, gene.lower(), CANONICAL_RESULTS, dataset_name=dataset_name)
        analysis.run(partial_pattern_mode="trust_global", **RUN_KWARGS)
        if analysis.results.get(best_config) is None:
            print(f"  SKIP {gene}: no valid bootstraps for config {best_config}", flush=True)
            continue
        points = np.asarray(analysis.results[best_config]["points"], dtype=float)

        n_matched = 0
        for v in vus_variants:
            idx = hgvs3_index.get(v)
            if idx is None:
                continue
            n_matched += 1
            rows.append({"gene": gene, "variant_key": v, "our_points": points[idx],
                        "canonical_config": best_config})
        print(f"  {gene}: {len(vus_variants)} true ClinVar VUS, {n_matched} matched to ms.kept_variants "
              f"(config={best_config})", flush=True)

    df = pd.DataFrame(rows)
    out_path = f"{OUTPUT_DIR}/true_clinvar_vus_points.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
