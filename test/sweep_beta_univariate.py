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


def fit_for_config(beta, predictor, ds, K=3, num_fits=100):
    """Run one univariate (predictor, β) config and return the best fit summary.

    Matches prepare_batch_jobs_single_predictor.py: K-component constrained
    skew-normal mixture, whole dataset, num_fits inits, default init mix.
    """
    fitter = Fit(ds)
    t0 = time.perf_counter()
    models, best_idx, best_ll = fitter.run(
        component_range=[K],
        bootstrap=False,
        check_monotonic=True,           # constrained (production setting)
        sample_balance_beta=beta,
        num_fits=num_fits,
        core_limit=1,                   # avoid nested joblib parallelism
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    best = models[best_idx]
    # Univariate component_params: list of (a, loc, scale) tuples
    params_sorted = sorted(
        best["component_params"],
        key=lambda p: float(p[1]),       # by loc (mean parameter)
    )
    weights = best["weights"]
    return {
        "predictor": predictor,
        "beta": float(beta),
        "best_ll": float(best_ll) if best_ll is not None else None,
        "best_idx": int(best_idx),
        "n_iters": int(len(best.get("likelihoods", []))),
        "n_fits": int(len(models)),
        "component_params_sorted": [list(p) for p in params_sorted],
        "best_weights": (weights.tolist() if hasattr(weights, "tolist") else weights),
        "elapsed_s": float(elapsed),
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
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
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

    # Build configs: one per (β, predictor) — joblib parallelises across these
    configs = [(b, p) for p in scoresets.keys() for b in args.betas]
    n_jobs = args.n_jobs if args.n_jobs > 0 else min(len(configs), os.cpu_count() or 4)
    print(f"\nSweeping {len(configs)} configs "
          f"({len(scoresets)} predictors × {len(args.betas)} βs) "
          f"with {n_jobs} parallel workers")
    print(f"  K={args.components}, constrained, no bootstrap, num_fits={args.num_fits}")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_for_config)(
            beta, predictor, scoresets[predictor],
            K=args.components, num_fits=args.num_fits,
        )
        for (beta, predictor) in configs
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTotal sweep wall time: {elapsed:.1f}s")

    # Tabulate
    print()
    header = (f"{'predictor':<10}{'β':<8}{'best LL':<14}{'iters':<8}"
              f"components (a, loc, scale) sorted by loc")
    print("=" * max(len(header), 100))
    print(header)
    print("=" * max(len(header), 100))
    for r in results:
        params_str = "   ".join(
            f"({p[0]:+.2f}, {p[1]:+.3f}, {p[2]:.3f})"
            for p in r["component_params_sorted"]
        )
        ll_str = f"{r['best_ll']:.4f}" if r["best_ll"] is not None else "NA"
        print(f"{r['predictor']:<10}{r['beta']:<8}{ll_str:<14}"
              f"{r['n_iters']:<8}{params_str}")

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
