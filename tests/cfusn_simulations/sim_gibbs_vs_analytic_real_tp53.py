#!/usr/bin/env python3
"""Production analytic q=2 E-step vs. Gibbs-substituted, on REAL TP53 data
(not synthetic) -- speed and practical-output comparison.

Full TP53 (9911 variants x 16 dims) with genuine Gibbs sampling in every
E-step of every EM iteration is not tractable here (extrapolating from the
synthetic timings in sim_gibbs_vs_analytic_bias.py, a single fit would
plausibly take many hours to days). This script instead uses a small,
stratified real subsample -- real distributional shape, real missingness
patterns (not synthetic), just fewer variants/dimensions -- to keep genuine
Gibbs sampling runnable while still answering the practical question: does
substituting real MCMC for the analytic approximation change the fitted
model or the evidence it produces, and at what speed cost?

Since there's no ground truth on real data, "accuracy" here means: do the
two fits' Delta/Gamma structure (especially KawOligo's row, the motivating
sparse dimension) and evidence output meaningfully differ -- not a
recovery-error number like the synthetic experiments.

Usage:
    python tests/cfusn_simulations/sim_gibbs_vs_analytic_real_tp53.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_HPC_DIR = _REPO_ROOT / "hpc"
for p in (_REPO_ROOT, _HPC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import src.assay_calibration.fit_utils.cfusn.update_steps as US
from src.assay_calibration.fit_utils.fit import tryToFit
from tests.cfusn_simulations.sim_utils import gibbs_truncated_mvn_moments

# WAF1nWT, MDM2nWT, BAXnWT (well-observed core assays) + DN_score,
# log_TempSens, KawOligo (the genuinely sparse dimensions, KawOligo being
# the original motivating case: ~1-12% real coverage).
DIM_NAMES = ["WAF1nWT", "MDM2nWT", "BAXnWT", "DN_score", "log_TempSens", "KawOligo"]
DIM_INDICES = [0, 1, 2, 13, 14, 15]


def load_subsampled_tp53(n_per_sample, seed):
    import prepare
    from argparse import Namespace
    from src.assay_calibration.fit_utils.fit import makeOneHot

    args = Namespace(genes=None, exclude_genes=None, rpvs_all=False, kawoligo_seed=0,
                     kawoligo_jitter_sigma=0.1, fgfr_separate=False, dataframe=None,
                     predictor_data_dir=None)
    gene_ms_map = prepare._load_and_filter_gene_ms_map("tp53", args)
    ms = gene_ms_map["TP53"]

    scores = ms.scores[:, DIM_INDICES]
    sa = ms.sample_assignments

    # ms.scores/sample_assignments include every variant, most with an
    # all-zero sample_assignments row (no calibration-sample label, e.g.
    # unclassified VUS) -- drop those first, matching Fit.generate_fit_jobs'
    # own `include = sample_assignments.any(axis=1) & ...` filter, then
    # resolve any multi-label overlap into single-label (one-hot) the same
    # way production does before fitting.
    labeled = sa.any(axis=1)
    scores, sa = scores[labeled], sa[labeled]
    sa = makeOneHot(sa, rng=np.random.RandomState(seed))

    rng = np.random.RandomState(seed)
    keep_rows = []
    for s in range(sa.shape[1]):
        idx = np.where(sa[:, s])[0]
        if len(idx) == 0:
            continue
        n = min(n_per_sample, len(idx))
        keep_rows.append(rng.choice(idx, size=n, replace=False))
    keep_rows = np.unique(np.concatenate(keep_rows))

    return scores[keep_rows], sa[keep_rows]


def _run(X, sa, use_gibbs, max_em_iters, n_gibbs_samples, n_gibbs_burnin, seed):
    if use_gibbs:
        orig = US._mc_truncated_mvn_moments

        def _patched(means, cov, n_mc=500, rng=None):
            return gibbs_truncated_mvn_moments(
                means, cov, n_mc=n_mc, rng=rng,
                n_gibbs_samples=n_gibbs_samples, n_burnin=n_gibbs_burnin,
            )
        US._mc_truncated_mvn_moments = _patched
    t0 = time.perf_counter()
    try:
        result = tryToFit(
            X, sa, num_components=3, constrained=False,
            init_method="kmeans", init_constraint_adjustment="scale",
            multivariate=True, latent_q=2, check_monotonic=False,
            num_fits=1, fit_seed=int(seed), max_em_iters=max_em_iters,
            verbose=False, verbose_init=False,
        )
    finally:
        if use_gibbs:
            US._mc_truncated_mvn_moments = orig
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _report(label, result, elapsed):
    print(f"\n{'─' * 78}")
    print(f"  {label}   (wall time: {elapsed:.1f}s)")
    print(f"{'─' * 78}")
    params = result.get("component_params", [])
    if not params or any(len(p) == 0 for p in params):
        print("  FIT FAILED")
        return
    n_iters = len(result.get("likelihoods", []))
    final_ll = result["likelihoods"][-1] if len(result.get("likelihoods", [])) else float("nan")
    print(f"  iterations used: {n_iters}   final (per-obs) log-likelihood: {final_ll:.5f}")
    kawoligo_idx = DIM_NAMES.index("KawOligo")
    for c, (mu, Delta, Gamma) in enumerate(params):
        mu = np.asarray(mu)
        Delta = np.asarray(Delta)
        Gamma = np.asarray(Gamma)
        print(f"  component {c}: KawOligo mu={mu[kawoligo_idx]:.3f}  "
              f"KawOligo Delta row={np.round(Delta[kawoligo_idx], 3)}  "
              f"KawOligo Gamma_diag={Gamma[kawoligo_idx, kawoligo_idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-sample", type=int, default=40,
                        help="Max real variants sampled per sample class (small by "
                             "necessity for the Gibbs-substituted condition)")
    parser.add_argument("--max-em-iters-analytic", type=int, default=200)
    parser.add_argument("--max-em-iters-gibbs", type=int, default=40)
    parser.add_argument("--n-gibbs-samples", type=int, default=80)
    parser.add_argument("--n-gibbs-burnin", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, sa = load_subsampled_tp53(args.n_per_sample, args.seed)
    print(f"Real TP53 subsample: {X.shape[0]} variants x {X.shape[1]} dims {DIM_NAMES}")
    print(f"Per-dimension observed fraction: "
          f"{dict(zip(DIM_NAMES, np.round((~np.isnan(X)).mean(axis=0), 3)))}")

    result_a, t_a = _run(X, sa, False, args.max_em_iters_analytic, None, None, args.seed)
    _report("PRODUCTION (analytic q=2 fast path)", result_a, t_a)

    result_g, t_g = _run(X, sa, True, args.max_em_iters_gibbs,
                         args.n_gibbs_samples, args.n_gibbs_burnin, args.seed)
    _report("GIBBS-SUBSTITUTED (real MCMC every E-step)", result_g, t_g)

    print(f"\n{'═' * 78}")
    print(f"  SPEED: analytic={t_a:.1f}s   gibbs={t_g:.1f}s   "
          f"ratio={t_g / max(t_a, 1e-9):.1f}x slower")
    print(f"{'═' * 78}\n")


if __name__ == "__main__":
    main()
