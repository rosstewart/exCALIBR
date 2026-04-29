"""
Local β-sensitivity sweep on BRCA1 (or any predictor-MV gene).

Runs a CFUSN fit (K=3, q=2, unconstrained, anchored init) on the gene's
BasicMultiScoreset for each β ∈ {0, 0.25, 0.5, 0.75, 1.0}. Single
"bootstrap": we use the whole dataset (no resampling) and pick the best
fit per β by training LL across the 64 lambdaIndex sign patterns.

Outer parallelism: joblib over β values, one worker per (β, init)
config. Inner Fit.run is sequential (`core_limit=1`) so workers don't
contend for cores.

Usage
-----
    # local data dir (override the cluster default):
    python test/sweep_beta_brca1.py \\
        --data-dir /path/to/predictor_scores/single_gene_calibration_data

    # different gene, custom β set, save pickle for plotting:
    python test/sweep_beta_brca1.py --gene MSH2 \\
        --betas 0 0.5 1.0 --save-pickle /tmp/sweep_msh2.pkl

    # include the (kmeans, β=0) baseline alongside the anchored sweep:
    python test/sweep_beta_brca1.py --include-kmeans-baseline
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
from predictor_mv_utils import load_predictor_ms


def fit_for_config(beta, init_strategy, ms, K=3, latent_q=2, num_fits=None):
    """Run the standard fit set for one (β, init) config and return the best.

    With ``bootstrap=False``, Fit.run uses the whole dataset and picks
    the best of NUM_FITS fits by training LL.
    """
    fitter = Fit(ms)
    extra = {"num_fits": num_fits} if num_fits is not None else {}
    t0 = time.perf_counter()
    models, best_idx, best_ll = fitter.run(
        component_range=[K],
        bootstrap=False,
        check_monotonic=False,         # unconstrained (production setting)
        init_strategy=init_strategy,
        sample_balance_beta=beta,
        latent_q=latent_q,
        core_limit=1,                  # avoid nested joblib parallelism
        verbose=False,
        **extra,
    )
    elapsed = time.perf_counter() - t0
    best = models[best_idx]
    means_sorted = sorted(
        [np.asarray(p[0]) for p in best["component_params"]],
        key=lambda mu: float(mu[0]),
    )
    weights = best["weights"]
    return {
        "beta": float(beta),
        "init_strategy": init_strategy,
        "best_ll": float(best_ll) if best_ll is not None else None,
        "best_idx": int(best_idx),
        "n_iters": int(len(best.get("likelihoods", []))),
        "n_fits": int(len(models)),
        "component_means_sorted": [m.tolist() for m in means_sorted],
        "best_params": best["component_params"],
        "best_weights": (weights.tolist() if hasattr(weights, "tolist") else weights),
        "elapsed_s": float(elapsed),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Local β-sensitivity sweep on a predictor-MV gene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gene", default="BRCA1",
                        help="Gene to fit (default: BRCA1).")
    parser.add_argument("--data-dir", type=str,
                        default="/data/ross/assay_calibration/predictor_scores/"
                                "single_gene_calibration_data",
                        help="Directory containing {gene}/{gene}_{predictor}.csv.gz.")
    parser.add_argument("--betas", nargs="+", type=float,
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="β values to sweep (default: 0 0.25 0.5 0.75 1.0).")
    parser.add_argument("-K", "--components", type=int, default=3,
                        help="Number of mixture components (default: 3).")
    parser.add_argument("--latent-q", type=int, default=2,
                        help="CFUSN latent dim q (default: 2).")
    parser.add_argument("--num-fits", type=int, default=None,
                        help="Override fits-per-config (default: dynamic min(4^K, 100)).")
    parser.add_argument("--include-kmeans-baseline", action="store_true",
                        help="Also fit one (kmeans, β=0) baseline for reference.")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Outer parallel workers (-1 = one per config, capped by cores).")
    parser.add_argument("--save-pickle", type=str, default=None,
                        help="Path to save the full results dict (pickle).")
    args = parser.parse_args()

    print(f"Loading {args.gene} predictor data from {args.data_dir} ...")
    ms = load_predictor_ms(args.gene, args.data_dir)
    sample_counts = (
        ms.sample_counts.tolist() if hasattr(ms.sample_counts, "tolist")
        else list(ms.sample_counts)
    )
    print(f"  {ms.n_variants} variants, {ms.n_assays} dims, "
          f"{ms.missing.mean() * 100:.1f}% missing")
    print(f"  Sample counts: {dict(zip(ms.sample_names, sample_counts))}")
    print(f"  Predictors: {ms.dataset_names}")

    # Empirical per-sample mean per dim — used as reference lines on plots
    # (each anchored component's μ should land near its anchor sample's mean).
    scores = ms.scores
    sa = ms.sample_assignments
    sample_means = np.full((sa.shape[1], scores.shape[1]), np.nan)
    for s in range(sa.shape[1]):
        rows = sa[:, s].astype(bool)
        if rows.any():
            sample_means[s] = np.nanmean(scores[rows], axis=0)

    # Build (β, init) configs
    configs = [(b, "anchored") for b in args.betas]
    if args.include_kmeans_baseline:
        configs.append((0.0, "kmeans"))

    n_jobs = args.n_jobs if args.n_jobs > 0 else min(len(configs), os.cpu_count() or 4)
    print(f"\nSweeping {len(configs)} configs with {n_jobs} parallel workers")
    print(f"  K={args.components}, latent_q={args.latent_q}, "
          f"unconstrained, no bootstrap (whole-dataset fit)")
    if args.num_fits is not None:
        print(f"  num_fits override: {args.num_fits}")
    else:
        nf = min((2 ** args.latent_q) ** args.components, 100)
        print(f"  num_fits per config: {nf} (dynamic min(4^K, 100))")

    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_for_config)(
            beta, init_strategy, ms,
            K=args.components, latent_q=args.latent_q, num_fits=args.num_fits,
        )
        for (beta, init_strategy) in configs
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTotal sweep wall time: {elapsed:.1f}s")

    # Tabulate
    print()
    header = f"{'init':<10}{'β':<8}{'best LL':<14}{'iters':<8}component means (sorted by μ[0])"
    print("=" * max(len(header), 95))
    print(header)
    print("=" * max(len(header), 95))
    for r in results:
        means_str = "   ".join(
            "[" + ", ".join(f"{x:+.3f}" for x in m) + "]"
            for m in r["component_means_sorted"]
        )
        ll_str = f"{r['best_ll']:.4f}" if r["best_ll"] is not None else "NA"
        print(f"{r['init_strategy']:<10}{r['beta']:<8}{ll_str:<14}{r['n_iters']:<8}{means_str}")

    if args.save_pickle:
        save_path = Path(args.save_pickle)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "gene": args.gene,
                "components": args.components,
                "latent_q": args.latent_q,
                "sample_names": list(ms.sample_names),
                "sample_counts": sample_counts,
                "sample_means": sample_means,         # (S, p), empirical per-sample mean
                "predictors": list(ms.dataset_names),
                "configs": configs,
                "results": results,
                "elapsed_s": elapsed,
            }, f)
        print(f"\nSaved full results to {save_path}")


if __name__ == "__main__":
    main()
