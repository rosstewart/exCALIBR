#!/usr/bin/env python
"""Per-gene calibration figures for a hpc/run_grow_component_batch.py output
directory (config key "3c_unc_plus1_plp" by default). Mirrors this
session's established render_plp_component_figs.py scratch-script pattern,
generalized to take --gene-set/--results-json/--require-both-modalities so
it works for predictor, combined (union), and combined (intersection)
without copy-pasting three near-identical scripts.

Usage
-----
    python analysis/render_grow_component_figs.py \\
        --gene-set predictor \\
        --results-dir /data/ross/assay_calibration/multivariate/jobs_predictor_plp_component_20b_3f_100126
"""
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")

from mv_analysis import mv_cockpit as cockpit
from mv_analysis.gene_performance_scatter import fast_results_json
from mv_analysis.analyze_mv_pipeline_output import RUN_KWARGS, PREDICTOR_COMBINED_GENES
from src.assay_calibration.multivariate_analysis.report_gene import generate_gene_report
from src.assay_calibration.multivariate_data.predictors import load_predictor_ms, predictor_dataset_label

CONFIG_KEY = "3c_unc_plus1_plp"
PREDICTOR_DATA_DIR = "/data/ross/assay_calibration/predictor_calibrations/single_gene_calibration_data"


def _build_ms(gene, gene_set, require_both_modalities):
    if gene_set == "predictor":
        return load_predictor_ms(gene, PREDICTOR_DATA_DIR, standardize=False)
    if gene_set == "combined":
        from src.assay_calibration.multivariate_data.combined import (
            build_functional_scoresets, build_combined_multiscoreset,
            get_functionally_assayed_protein_variants, DEFAULT_INTEGRATED_DATAFRAME,
        )
        from src.assay_calibration.multivariate_data.common import resolve_clinvar_release
        import pandas as pd
        df = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
        datasets = sorted(df[df["Gene"] == gene]["Dataset"].unique())
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
    ap.add_argument("--results-dir", required=True,
                     help="A hpc/run_grow_component_batch.py --output-dir (will be aggregated "
                          "via hpc/aggregate_results.py first if bootstrap_results.json.gz is missing).")
    ap.add_argument("--require-both-modalities", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_json = results_dir / "bootstrap_results.json.gz"
    if not results_json.exists():
        import subprocess
        subprocess.run([sys.executable, str(_ROOT / "hpc" / "aggregate_results.py"),
                         str(results_dir)], check=True)

    out_dir = results_dir / "figures_TEMP"
    out_dir.mkdir(parents=True, exist_ok=True)

    import gzip, json
    with gzip.open(results_json, "rt") as f:
        available = set(json.load(f).keys())

    for gene in args.genes:
        dataset_name = _dataset_name(gene, args.gene_set)
        if dataset_name not in available:
            print(f"\n=== {gene} === SKIP: not in {results_json}")
            continue
        print(f"\n=== {gene} ===", flush=True)
        try:
            ms = _build_ms(gene, args.gene_set, args.require_both_modalities)
            analysis = cockpit.build_gene_set_analysis(
                ms, gene.lower(), str(results_json), dataset_name=dataset_name, gene_set=args.gene_set)
            with fast_results_json(str(results_json)):
                analysis.run(partial_pattern_mode="trust_global", **RUN_KWARGS)
            generate_gene_report(
                analysis, gene, str(out_dir / gene), configs=[CONFIG_KEY],
                # cache_dir=False: the "combined" gene-set's cache filename
                # concatenates every dataset name (ENAMETOOLONG bug, already
                # known from this session's BRCA2/TP53 work) -- disabling
                # caching avoids it entirely; this render only runs once per
                # gene anyway, so the cache has no real benefit here.
                cache_dir=False,
            )
        except Exception as e:
            import traceback
            print(f"  SKIP {gene}: {e}\n{traceback.format_exc()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
