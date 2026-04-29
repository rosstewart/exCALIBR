"""
Local univariate β-sensitivity sweep on a predictor-MV gene.

Mirrors test/prepare_batch_jobs_single_predictor.py settings, but as a
local parallel sweep over β values rather than a SLURM array:

  - Univariate skew-normal mixture, fit per predictor independently
  - K=3 only (no 2c)
  - constrained (check_monotonic=True)            ← production setting
  - 100 fits per (predictor, β); whole dataset (bootstrap=False)
  - default init mix (kmeans / method-of-moments) — no anchored init
    (univariate constrained relies on monotonicity, not anchoring)

Outer parallelism: joblib over (predictor, β) configs, one worker each.
Inner Fit.run is sequential (core_limit=1) so workers don't contend.

Usage
-----
    python test/sweep_beta_univariate.py \\
        --data-dir /path/to/predictor_scores/single_gene_calibration_data \\
        --save-pickle /tmp/sweep_brca1_uv.pkl

    python test/sweep_beta_univariate.py --gene MSH2 \\
        --betas 0 0.5 1.0 --predictors REVEL MP2
"""

import os
import sys
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[1]))
sys.path.insert(0, str(_THIS.parent))

from src.assay_calibration.fit_utils.fit import Fit
from predictor_mv_utils import (
    PREDICTORS, PREDICTOR_DATASET_NAMES,
    load_predictor_data, df_to_basic_scoreset,
)


def fit_for_config(beta, predictor, ds, init_strategy="default", K=3, num_fits=100,
                   anchor_centroid_mode="contrastive", anchor_centroid_verbose=False):
    """Run one univariate (predictor, β, init) config and return the best fit summary.

    Matches prepare_batch_jobs_single_predictor.py: K-component constrained
    skew-normal mixture, whole dataset, num_fits inits.

    init_strategy:
      "default"            — production default (random kmeans / MoM mix)
      "anchored"           — UV anchored init (kmeans_init_anchored)
      "kmeans"             — forced kmeans
      "method_of_moments"  — forced MoM
    """
    fitter = Fit(ds)
    extra = {}
    # Production default uses init_strategy="random"; passing it explicitly is
    # a no-op vs not passing, but keeps the call self-documenting.
    if init_strategy != "default":
        extra["init_strategy"] = init_strategy
    if init_strategy == "anchored":
        extra["anchor_centroid_mode"] = anchor_centroid_mode
        extra["anchor_centroid_verbose"] = anchor_centroid_verbose
    t0 = time.perf_counter()
    models, best_idx, best_ll = fitter.run(
        component_range=[K],
        bootstrap=False,
        check_monotonic=True,           # constrained (production setting)
        sample_balance_beta=beta,
        num_fits=num_fits,
        core_limit=1,                   # avoid nested joblib parallelism
        verbose=False,
        **extra,
    )
    elapsed = time.perf_counter() - t0

    best = models[best_idx] if models else {}
    cps = best.get("component_params", [])
    # Detect the all-fits-failed case: tryToFit returns dict with
    # component_params=[[], [], ...] when the EM/init raises, and
    # _safe_ll gives every such fit -inf so best_idx points at one of them.
    all_failed = (
        not cps
        or any(
            (isinstance(p, (list, tuple)) and len(p) == 0)
            for p in cps
        )
        or best_ll is None
        or not np.isfinite(best_ll)
    )
    if all_failed:
        print(f"  [FAIL] {predictor} init={init_strategy} β={beta:g}: "
              f"all {len(models)} fits failed", flush=True)
        return {
            "predictor": predictor,
            "beta": float(beta),
            "init_strategy": init_strategy,
            "anchor_centroid_mode": anchor_centroid_mode if init_strategy == "anchored" else None,
            "best_ll": None,
            "best_idx": int(best_idx) if best_idx is not None else -1,
            "n_iters": 0,
            "n_fits": int(len(models)),
            "component_params_sorted": [],
            "best_weights": None,
            "elapsed_s": float(elapsed),
            "failed": True,
        }

    # Univariate component_params: list of (a, loc, scale) tuples
    params_sorted = sorted(cps, key=lambda p: float(p[1]))   # by loc
    weights = best["weights"]
    return {
        "predictor": predictor,
        "beta": float(beta),
        "init_strategy": init_strategy,
        "anchor_centroid_mode": anchor_centroid_mode if init_strategy == "anchored" else None,
        "best_ll": float(best_ll),
        "best_idx": int(best_idx),
        "n_iters": int(len(best.get("likelihoods", []))),
        "n_fits": int(len(models)),
        "component_params_sorted": [list(p) for p in params_sorted],
        "best_weights": (weights.tolist() if hasattr(weights, "tolist") else weights),
        "elapsed_s": float(elapsed),
        "failed": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Local univariate β-sensitivity sweep on a predictor-MV gene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gene", default="BRCA1")
    parser.add_argument("--data-dir", type=str,
                        default="/data/ross/assay_calibration/predictor_scores/"
                                "single_gene_calibration_data",
                        help="Directory containing {gene}/{gene}_{predictor}.csv.gz.")
    parser.add_argument("--betas", nargs="+", type=float,
                        default=[0.0, 0.5, 1.0],
                        help="β values to sweep (default: 0 0.5 1).")
    parser.add_argument("--init-strategies", nargs="+",
                        default=["default", "anchored"],
                        choices=["default", "anchored", "kmeans", "method_of_moments"],
                        help="Init strategies to compare. 'default' = production "
                             "(random kmeans/MoM mix). Default sweep: default + anchored.")
    parser.add_argument("--anchor-mode",
                        default="contrastive",
                        choices=["contrastive", "dominant", "mean"],
                        help="How anchored init derives each component μ when bimodality "
                             "is detected by BIC. 'mean' is the override that disables BIC.")
    parser.add_argument("--anchor-debug", action="store_true",
                        help="Print which branch (mean/dominant/contrastive) chose μ.")
    parser.add_argument("--predictors", nargs="+", default=list(PREDICTORS),
                        choices=list(PREDICTORS),
                        help="Predictors to fit (default: all of REVEL/MP2/AM).")
    parser.add_argument("-K", "--components", type=int, default=3)
    parser.add_argument("--num-fits", type=int, default=100,
                        help="Fits per config (default: 100, matches production).")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Outer parallel workers (-1 = one per config, capped by cores).")
    parser.add_argument("--save-pickle", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading {args.gene} predictor data from {args.data_dir} ...")
    by_gene = load_predictor_data(args.data_dir, genes=[args.gene])
    if args.gene not in by_gene:
        raise SystemExit(f"No predictor CSVs found for {args.gene} under {args.data_dir}")
    predictor_dfs = by_gene[args.gene]

    scoresets = {}
    sample_means = {}    # {predictor: (S,) array}
    sample_counts = {}   # {predictor: (S,) list}
    for predictor in args.predictors:
        if predictor not in predictor_dfs:
            print(f"  {predictor}: missing CSV, skipping")
            continue
        ds = df_to_basic_scoreset(predictor_dfs[predictor], predictor)
        scoresets[predictor] = ds

        sa = ds.sample_assignments
        scores = np.asarray(ds.scores, dtype=float)
        S = sa.shape[1]
        means = np.full(S, np.nan)
        counts = np.zeros(S, dtype=int)
        for s in range(S):
            mask = sa[:, s].astype(bool)
            counts[s] = int(mask.sum())
            if mask.any():
                means[s] = float(np.nanmean(scores[mask]))
        sample_means[predictor] = means
        sample_counts[predictor] = counts.tolist()
        means_str = ", ".join(f"{m:+.3f}" for m in means)
        print(f"  {predictor}: {len(scores)} variants, "
              f"counts={counts.tolist()}, means=[{means_str}]")

    if not scoresets:
        raise SystemExit("No predictor scoresets loaded.")

    # Build configs: full grid of (predictor × β × init)
    configs = [
        (b, p, init)
        for p in scoresets.keys()
        for init in args.init_strategies
        for b in args.betas
    ]
    n_jobs = args.n_jobs if args.n_jobs > 0 else min(len(configs), os.cpu_count() or 4)
    print(f"\nSweeping {len(configs)} configs "
          f"({len(scoresets)} predictors × {len(args.betas)} βs × "
          f"{len(args.init_strategies)} inits) with {n_jobs} parallel workers")
    print(f"  K={args.components}, constrained, no bootstrap, num_fits={args.num_fits}")
    print(f"  init strategies: {args.init_strategies}")
    if "anchored" in args.init_strategies:
        print(f"  anchor centroid mode: {args.anchor_mode}"
              + (" (debug logging on)" if args.anchor_debug else ""))

    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_for_config)(
            beta, predictor, scoresets[predictor],
            init_strategy=init,
            K=args.components, num_fits=args.num_fits,
            anchor_centroid_mode=args.anchor_mode,
            anchor_centroid_verbose=args.anchor_debug,
        )
        for (beta, predictor, init) in configs
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTotal sweep wall time: {elapsed:.1f}s")

    # Tabulate
    print()
    header = (f"{'predictor':<10}{'init':<20}{'β':<6}{'best LL':<14}{'iters':<8}"
              f"components (a, loc, scale) sorted by loc")
    print("=" * max(len(header), 110))
    print(header)
    print("=" * max(len(header), 110))
    n_failed = 0
    for r in results:
        if r.get("failed"):
            n_failed += 1
            print(f"{r['predictor']:<10}{r['init_strategy']:<20}{r['beta']:<6}"
                  f"{'FAILED':<14}{'-':<8}all fits failed")
            continue
        params_str = "   ".join(
            f"({p[0]:+.2f}, {p[1]:+.3f}, {p[2]:.3f})"
            for p in r["component_params_sorted"]
        )
        ll_str = f"{r['best_ll']:.4f}" if r["best_ll"] is not None else "NA"
        print(f"{r['predictor']:<10}{r['init_strategy']:<20}{r['beta']:<6}{ll_str:<14}"
              f"{r['n_iters']:<8}{params_str}")
    if n_failed:
        print(f"\n  [WARN] {n_failed}/{len(results)} configs failed entirely")

    if args.save_pickle:
        save_path = Path(args.save_pickle)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        any_ds = next(iter(scoresets.values()))
        sample_names = (
            list(any_ds.sample_names) if hasattr(any_ds, "sample_names")
            else ["P/LP", "B/LB", "gnomAD"][:any_ds.sample_assignments.shape[1]]
        )
        with open(save_path, "wb") as f:
            pickle.dump({
                "gene": args.gene,
                "components": args.components,
                "predictors": list(scoresets.keys()),
                "predictor_dataset_names": [
                    PREDICTOR_DATASET_NAMES[p] for p in scoresets.keys()
                ],
                "sample_names": sample_names,
                "sample_counts": sample_counts,
                "sample_means": {p: m.tolist() for p, m in sample_means.items()},
                "configs": configs,
                "results": results,
                "elapsed_s": elapsed,
                "constrained": True,
                "univariate": True,
            }, f)
        print(f"\nSaved full results to {save_path}")


if __name__ == "__main__":
    main()
