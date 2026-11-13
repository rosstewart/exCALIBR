"""Build a small cluster test of predictor models fit on 2 principal
components of (REVEL, MutPred2, AlphaMissense) instead of the 3 raw scores.

Why: with three nearly redundant predictors the extra dimension adds little
information but bootstrap variance, which the evidence percentile pays for
(review_assay_selection --score percentile). PC1 is the predictors' consensus
and PC3 the near signal-free MutPred2-vs-others contrast, so PC1 + PC2 keeps
the consensus of all three and drops the noisiest direction (label-free).

Per gene (predictor-mv data, same as the paper fits): partially observed rows
are first filled with the conditional mean of the missing predictor(s) given
the observed ones (redundancy_collapse.impute_block_conditional_mean; the PCA
is defined only for complete rows), then the block is rotated and the first
--k components kept (redundancy_collapse.collapse_block). Fit settings follow
the paper's predictor fits: --num-fits 3 (Gaussian, MSN, CFUSN), pathogenic
share 0.1, unconstrained, kmeans init, master seed 0, so bootstrap seed s is
the same resample as the paper's (up to strata changed by imputation) and
seeds 0..n-1 of the paper fits are the like-for-like 3-D control. The
component count per gene is the held-out choice on the paper fits (4, F9 3).

Layout as the production runs: <out>/tau_0.1/<gene>_predictors_mv/ results,
<out>/jobs/ one shared array, <out>/tau_0.1/predictor_pca.json describes the
transform (the analysis applies the same deterministic preprocessing).
Built locally (the predictor data are here); copy <out> to the cluster and
submit with `bash hpc/submit_array.sh <out>`.

Usage: python hpc/prepare_predictor_pca_test.py --output-dir DIR
       [--n-bootstraps 20] [--num-fits 3] [--k 2] [--target-array-size 50]
"""
import argparse
import json
import os
import sys
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpc import prepare as P  # noqa: E402

GENES = ["BRCA1", "BRCA2", "F9", "JAG1", "MSH2", "SCN5A", "TP53", "TSC2"]
HELDOUT_K = {g: 4 for g in GENES} | {"F9": 3}
BLOCK = ["REVEL", "MutPred2", "AlphaMissense"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-bootstraps", type=int, default=20)
    ap.add_argument("--num-fits", type=int, default=3)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--target-array-size", type=int, default=50)
    ap.add_argument("--pathogenic-share", type=float, default=0.1)
    ap.add_argument("--predictor-data-dir",
                    default="/data/ross/assay_calibration/predictor_calibrations/single_gene_calibration_data")
    ap.add_argument("--genes", nargs="+", default=GENES)
    a = ap.parse_args()

    from src.assay_calibration.multivariate_data import redundancy_collapse as rc
    from src.assay_calibration.multivariate_data.predictors import (build_basic_multi_scoreset,
                                                                    load_predictor_data, predictor_dataset_label)
    tau = a.pathogenic_share
    label_tau = f"tau_{tau:g}"
    root = a.output_dir
    cond = os.path.join(root, label_tau)
    os.makedirs(os.path.join(cond, "jobs"), exist_ok=True)
    by_gene = load_predictor_data(a.predictor_data_dir, genes=a.genes)
    jobs, info = [], {}
    for gene, dfs in by_gene.items():
        label = predictor_dataset_label(gene)
        ms, why = build_basic_multi_scoreset(gene, dfs)
        if ms is None:
            print(f"  {gene}: skipped ({why})")
            continue
        n_imp = rc.impute_block_conditional_mean(ms, BLOCK)
        tr = rc.collapse_block(ms, BLOCK, a.k, new_names=[f"predPC{i + 1}" for i in range(a.k)])
        K = HELDOUT_K[gene.upper()]
        info[label] = dict(gene=gene, k_pcs=a.k, components=K, n_variants=int(ms.n_variants),
                           n_imputed_rows=n_imp, loadings=tr.evecs.tolist(), mean=tr.mean_.tolist(),
                           dims=list(ms.dataset_names))
        print(f"  {label}: {ms.n_variants} variants, {n_imp} rows imputed, {BLOCK} -> {ms.dataset_names}, K={K}")
        jobs += P._generate_bootstrap_fit_jobs(
            ms, label, gene, os.path.join(cond, label), a.n_bootstraps, a.num_fits,
            [K], ["unc"], 2, "kmeans", min_overlap_rows=1, sample_balance_beta=0,
            sample_weight_transform=None, master_seed=P.DEFAULT_MASTER_SEED, greedy_restarts=False,
            pathogenic_share=tau if tau > 0 else None, fit_options=P._fit_options(Namespace()))
    with open(os.path.join(cond, "predictor_pca.json"), "w") as f:
        json.dump(dict(block=BLOCK, k=a.k, missing="conditional-mean imputation", models=info), f, indent=1)
    P._save_manifest(cond, jobs, a.target_array_size, a.n_bootstraps,
                     {"component_range": sorted(set(HELDOUT_K.values())), "constraint_modes": ["unc"]})
    P._merge_condition_manifests(root, [label_tau], a.target_array_size)
    print(f"\n{len(jobs)} jobs -> {root}/jobs; submit on the cluster with: bash hpc/submit_array.sh {root}")


if __name__ == "__main__":
    main()
