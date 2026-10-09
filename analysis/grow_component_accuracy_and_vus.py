#!/usr/bin/env python
"""Accuracy table (MCC/accuracy/AUC) and true-ClinVar-VUS reclassification
points for a hpc/run_grow_component_batch.py output directory (config key
"3c_unc_plus1_plp"). Generalizes this session's established
fix_intersection_vus.py pattern (pull points directly from the analysis
object for a specific config, bypassing compute_clinvar_vus_points_by_panel
.score_gene's hardcoded module-level RESULTS_JSON) to work for predictor,
combined (union), and combined (intersection) via CLI args.

Usage
-----
    python analysis/grow_component_accuracy_and_vus.py \\
        --gene-set predictor \\
        --results-dir /data/ross/assay_calibration/multivariate/jobs_predictor_plp_component_20b_3f_100126
"""
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from mv_analysis import mv_cockpit as cockpit
from mv_analysis.gene_performance_scatter import fast_results_json, EVIDENCE_DIRECTION
from mv_analysis.report import _rows_for_points, _eval_labels, _both_modality_mask
from mv_analysis.analyze_mv_pipeline_output import RUN_KWARGS, PREDICTOR_COMBINED_GENES
from src.assay_calibration.multivariate_data.predictors import load_predictor_ms, predictor_dataset_label
from analysis.compute_clinvar_vus_points_by_panel import (
    combined_vus_keys, DEFAULT_INTEGRATED_DATAFRAME,
)

CONFIG_KEY = "3c_unc_plus1_plp"
PREDICTOR_DATA_DIR = "/data/ross/assay_calibration/predictor_calibrations/single_gene_calibration_data"


def _build_ms(gene, gene_set, require_both_modalities):
    if gene_set == "predictor":
        return load_predictor_ms(gene, PREDICTOR_DATA_DIR, standardize=False)
    if gene_set == "combined":
        from src.assay_calibration.multivariate_data.combined import (
            build_functional_scoresets, build_combined_multiscoreset,
            get_functionally_assayed_protein_variants,
        )
        from src.assay_calibration.multivariate_data.common import resolve_clinvar_release, functional_datasets_for_gene
        df = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
        datasets = functional_datasets_for_gene(df, gene)
        functional_scoresets = build_functional_scoresets(
            df, gene, datasets, clinvar_release=resolve_clinvar_release(gene))
        functionally_assayed = get_functionally_assayed_protein_variants(df, gene, datasets)
        return build_combined_multiscoreset(
            gene, functional_scoresets, datasets, PREDICTOR_DATA_DIR,
            functionally_assayed_variants=functionally_assayed,
            require_both_modalities=require_both_modalities,
        )
    raise ValueError(gene_set)


def _dataset_name(gene, gene_set):
    return predictor_dataset_label(gene) if gene_set == "predictor" else f"{gene}_combined_mv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-set", required=True, choices=["predictor", "combined"])
    ap.add_argument("--genes", nargs="+", default=list(PREDICTOR_COMBINED_GENES))
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--require-both-modalities", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_json = str(results_dir / "bootstrap_results.json.gz")

    import gzip, json
    with gzip.open(results_json, "rt") as f:
        available = set(json.load(f).keys())

    df_integrated = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)

    accuracy_rows = []
    vus_rows = []

    for gene in args.genes:
        dataset_name = _dataset_name(gene, args.gene_set)
        if dataset_name not in available:
            print(f"\n=== {gene} === SKIP: not in {results_json}")
            continue
        print(f"\n=== {gene} ===", flush=True)
        try:
            ms = _build_ms(gene, args.gene_set, args.require_both_modalities)

            analysis = cockpit.build_gene_set_analysis(
                ms, gene.lower(), results_json, dataset_name=dataset_name, gene_set=args.gene_set)
            with fast_results_json(results_json):
                analysis.run(partial_pattern_mode="trust_global", **RUN_KWARGS)
            if analysis.results.get(CONFIG_KEY) is None:
                print(f"  {gene}: no valid bootstraps for config {CONFIG_KEY}")
                continue
            points = np.asarray(analysis.results[CONFIG_KEY]["points"], dtype=float)

            # _eval_labels applies the both-modality-observed restriction
            # internally whenever gene_set == "combined" (see
            # mv_analysis/report.py) -- union vs. intersection training
            # populations otherwise aren't comparable on the same metric.
            eval_mask, labels = _eval_labels(ms, analysis.p_idx, analysis.b_idx, gene_set=args.gene_set)
            pts_eval = points[eval_mask]

            rows = _rows_for_points(f"MV {CONFIG_KEY}", pts_eval, labels)
            row = next((r for r in rows if r["threshold"] == EVIDENCE_DIRECTION), None)
            if row is not None:
                accuracy_rows.append({
                    "gene": gene, "mcc": float(row["mcc"]), "accuracy": float(row["accuracy"]),
                    "auc": float(row["auc"]), "n_pathogenic": int(row["n_pathogenic"]),
                    "n_benign": int(row["n_benign"]),
                })
            else:
                print(f"  {gene}: no accuracy row for config {CONFIG_KEY}")

            df_gene = df_integrated[df_integrated["Gene"] == gene]
            vus_keys = combined_vus_keys(df_gene)
            variant_index = {kv: i for i, kv in enumerate(ms.kept_variants)}
            both_mod_full = _both_modality_mask(ms) if args.gene_set == "combined" else None
            n_matched = 0
            for v in vus_keys:
                idx = variant_index.get(v)
                if idx is None:
                    continue
                if both_mod_full is not None and not both_mod_full[idx]:
                    continue
                n_matched += 1
                vus_rows.append({"gene": gene, "variant_key": str(v), "our_points": points[idx]})
            print(f"  {gene}: {len(vus_keys)} true ClinVar VUS, {n_matched} matched"
                  f"{' (both-modality only)' if both_mod_full is not None else ''}")
        except Exception as e:
            import traceback
            print(f"  SKIP {gene}: {e}\n{traceback.format_exc()}")

    acc_df = pd.DataFrame(accuracy_rows)
    vus_df = pd.DataFrame(vus_rows)
    acc_out = results_dir / f"{args.gene_set}_grow_component_accuracy_table.csv"
    vus_out = results_dir / f"{args.gene_set}_grow_component_true_clinvar_vus_points.csv"
    acc_df.to_csv(acc_out, index=False)
    vus_df.to_csv(vus_out, index=False)
    print(f"\nSaved {acc_out}")
    print(f"Saved {vus_out}")

    if not acc_df.empty:
        print("\n=== accuracy table ===")
        print(acc_df.to_string(index=False))
    if not vus_df.empty:
        frac_zero = float((vus_df["our_points"] == 0).mean())
        mean_abs = float(vus_df["our_points"].abs().mean())
        print(f"\nVUS aggregate: n={len(vus_df)} frac_zero={frac_zero:.3f} mean_abs_points={mean_abs:.3f}")


if __name__ == "__main__":
    main()
