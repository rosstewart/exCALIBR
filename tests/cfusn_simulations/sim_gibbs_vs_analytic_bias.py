#!/usr/bin/env python3
"""Does production's q=2 analytic E-step approximation (independent
marginals + a linear correlation correction, in
update_steps._mc_truncated_mvn_moments) meaningfully bias parameter
recovery, specifically in how well EM exploits the correlation between the
two latent skew directions T1, T2?

The approximation's marginal moments (E[T1], E[T2]) are exact closed-form
truncated-normal formulas; only the CROSS term E[T1*T2|x] is approximate.
That cross term is driven by S = I_q - Delta' Omega^-1 Delta (a fixed,
data-independent matrix -- its off-diagonal S[0,1] sets the "induced
posterior correlation" between T1 and T2 for ANY observation, before even
looking at x). So the approximation should matter little when S[0,1]~=0
(marginals really are ~independent) and most when |S[0,1]| is large.

Two experiments:
  (1) Direct E-step bias: for LOW vs. HIGH |S[0,1]| (achieved by choosing
      Delta's two columns to be near-orthogonal vs. more aligned across the
      observed dimensions, Gamma held fixed), compute eta/Psi for a spread
      of posterior means m via BOTH the production analytic path
      (update_steps._mc_truncated_mvn_moments) and a Gibbs-sampled reference
      (sim_utils.gibbs_truncated_mvn_moments, using update_steps.
      _gibbs_sample_tn_q -- genuine MCMC draws from the exact posterior).
      Reports mean absolute bias in the cross-term Psi[0,1] and in eta.
  (2) End-to-end recovery: generate fully-observed CFUSN(q=2) mixture data
      (no missingness -- isolates the E-step approximation question from
      the separate missingness-recovery question already covered by
      sim_missingness_recovery.py / sim_kawoligo_like_recovery.py) for the
      LOW and HIGH correlation ground truths, fit under (a) production's
      analytic path and (b) the Gibbs-substituted path (monkeypatching
      update_steps._mc_truncated_mvn_moments in-process), and compare
      sim_utils.score_recovery's mean_omega_error. Both conditions use a
      capped max_em_iters and a cheap (not high-precision) Gibbs sampler
      config, since real MCMC inside every E-step of every EM iteration is
      slow -- this experiment trades precision for being runnable at all,
      by design (see --n-gibbs-samples/--n-gibbs-burnin/--max-em-iters).

Usage:
    python tests/cfusn_simulations/sim_gibbs_vs_analytic_bias.py
    python tests/cfusn_simulations/sim_gibbs_vs_analytic_bias.py --n-seeds 5
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.assay_calibration.fit_utils.cfusn.update_steps as US
from src.assay_calibration.fit_utils.fit import tryToFit
from tests.cfusn_simulations.sim_utils import (
    sample_cfusn_mixture, score_recovery, gibbs_truncated_mvn_moments,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _induced_S(Delta, Gamma):
    """S = I_q - Delta' Omega^-1 Delta, the (data-independent) covariance of
    the posterior T|x for CFUSN -- its off-diagonal is the induced
    correlation the analytic approximation has to get right."""
    Delta = np.asarray(Delta, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    Omega = Gamma + Delta @ Delta.T
    Omega = 0.5 * (Omega + Omega.T)
    Omega_inv_Delta = np.linalg.solve(Omega, Delta)
    S = np.eye(Delta.shape[1]) - Delta.T @ Omega_inv_Delta
    return 0.5 * (S + S.T)


# LOW: Delta's two columns act on almost disjoint dimensions (near-orthogonal
# influence) -> small induced |S[0,1]|.
# HIGH: Delta's two columns act on overlapping dimensions with the same
# sign pattern (aligned influence) -> larger induced |S[0,1]|.
DELTA_LOW = np.array([[0.7, 0.0], [0.7, 0.0], [0.0, 0.7], [0.0, 0.7]])
DELTA_HIGH = np.array([[0.7, 0.5], [0.6, 0.5], [0.5, 0.6], [0.5, 0.6]])
GAMMA_4D = 0.5 * np.eye(4)


def report_induced_correlations():
    S_low = _induced_S(DELTA_LOW, GAMMA_4D)
    S_high = _induced_S(DELTA_HIGH, GAMMA_4D)
    print(f"Induced posterior correlation S[0,1]: LOW={S_low[0,1]:.4f}, HIGH={S_high[0,1]:.4f}")
    return S_low, S_high


# ── Experiment 1: direct E-step bias ────────────────────────────────────────

def run_direct_bias(n_m_vectors=20, seed=0):
    rng = np.random.RandomState(seed)
    S_low, S_high = report_induced_correlations()
    rows = []
    for label, S in [("LOW", S_low), ("HIGH", S_high)]:
        means = rng.uniform(-1.5, 1.5, size=(n_m_vectors, 2))
        eta_analytic, Psi_analytic = US._mc_truncated_mvn_moments(means, S, rng=rng)
        eta_gibbs, Psi_gibbs = gibbs_truncated_mvn_moments(
            means, S, rng=rng, n_gibbs_samples=3000, n_burnin=300,
        )
        for i in range(n_m_vectors):
            rows.append(dict(
                regime=label,
                s01=float(S[0, 1]),
                eta_bias=float(np.linalg.norm(eta_analytic[i] - eta_gibbs[i])),
                cross_bias=float(abs(Psi_analytic[i, 0, 1] - Psi_gibbs[i, 0, 1])),
            ))
    return rows


def _report_direct_bias(rows):
    print(f"\n{'═' * 78}")
    print("  EXPERIMENT 1: direct E-step bias (analytic vs. Gibbs reference)")
    print(f"{'═' * 78}")
    print(f"  {'regime':>8}  {'S[0,1]':>8}  {'mean |eta bias|':>16}  {'mean |cross-term bias|':>22}")
    by_regime = {}
    for r in rows:
        by_regime.setdefault(r["regime"], []).append(r)
    for regime, group in by_regime.items():
        s01 = group[0]["s01"]
        eta_bias = np.mean([g["eta_bias"] for g in group])
        cross_bias = np.mean([g["cross_bias"] for g in group])
        print(f"  {regime:>8}  {s01:>8.4f}  {eta_bias:>16.4f}  {cross_bias:>22.4f}")


# ── Experiment 2: end-to-end recovery, production vs. Gibbs-substituted ────

N_SAMPLES = 3
COMP_PROBS = np.array([[0.85, 0.15], [0.15, 0.85], [0.5, 0.5]])


def _true_params_for(Delta):
    return [
        (np.array([-1.5, -1.2, -1.0, -0.8]), Delta, GAMMA_4D),
        (np.array([1.5, 1.2, 1.0, 0.8]), -Delta, GAMMA_4D),
    ]


def _run_one_e2e(true_params, seed, use_gibbs, max_em_iters, n_gibbs_samples, n_gibbs_burnin,
                 n_per_sample):
    rng = np.random.RandomState(seed)
    X, sa, _ = sample_cfusn_mixture(true_params, COMP_PROBS, np.full(N_SAMPLES, n_per_sample), rng)

    if use_gibbs:
        orig = US._mc_truncated_mvn_moments

        def _patched(means, cov, n_mc=500, rng=None):
            return gibbs_truncated_mvn_moments(
                means, cov, n_mc=n_mc, rng=rng,
                n_gibbs_samples=n_gibbs_samples, n_burnin=n_gibbs_burnin,
            )
        US._mc_truncated_mvn_moments = _patched
    try:
        result = tryToFit(
            X, sa, num_components=2, constrained=False,
            init_method="kmeans", init_constraint_adjustment="scale",
            multivariate=True, latent_q=2, check_monotonic=False,
            num_fits=1, fit_seed=int(seed), max_em_iters=max_em_iters,
            verbose=False, verbose_init=False,
        )
    finally:
        if use_gibbs:
            US._mc_truncated_mvn_moments = orig

    params = result.get("component_params", [])
    if not params or any(len(p) == 0 for p in params):
        return None
    n_iters_used = len(result.get("likelihoods", []))
    converged = n_iters_used < max_em_iters  # early_stopping fired before hitting the cap
    err = score_recovery(true_params, params)["mean_omega_error"]
    return err, n_iters_used, converged


def run_e2e(n_seeds, max_em_iters, n_gibbs_samples, n_gibbs_burnin, n_per_sample):
    rows = []
    for label, Delta in [("LOW", DELTA_LOW), ("HIGH", DELTA_HIGH)]:
        true_params = _true_params_for(Delta)
        for seed in range(n_seeds):
            for use_gibbs in (False, True):
                out = _run_one_e2e(true_params, seed, use_gibbs, max_em_iters,
                                   n_gibbs_samples, n_gibbs_burnin, n_per_sample)
                if out is None:
                    continue
                err, n_iters_used, converged = out
                rows.append(dict(
                    n_iters_used=n_iters_used, converged=converged,
                    regime=label, seed=seed,
                    variant="gibbs" if use_gibbs else "analytic",
                    mean_omega_error=err,
                ))
    return rows


def _report_e2e(rows):
    print(f"\n{'═' * 90}")
    print("  EXPERIMENT 2: end-to-end recovery (production analytic vs. Gibbs-substituted)")
    print(f"{'═' * 90}")
    print(f"  {'regime':>8}  {'variant':>9}  {'err':>8}  {'mean iters used':>16}  "
          f"{'n converged':>12}  {'n':>4}")
    by_key = {}
    for r in rows:
        by_key.setdefault((r["regime"], r["variant"]), []).append(r)
    for (regime, variant), group in by_key.items():
        errs = np.array([g["mean_omega_error"] for g in group])
        iters = np.array([g["n_iters_used"] for g in group])
        n_converged = sum(1 for g in group if g["converged"])
        print(f"  {regime:>8}  {variant:>9}  {errs.mean():>8.4f}  {iters.mean():>16.1f}  "
              f"{n_converged:>12}  {len(group):>4}")


def _write_csv(rows, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{ts}.csv"
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Seeds for experiment 2 (slow: real Gibbs sampling per E-step)")
    parser.add_argument("--n-m-vectors", type=int, default=20,
                        help="Random posterior-mean vectors for experiment 1")
    parser.add_argument("--max-em-iters", type=int, default=12,
                        help="EM iteration cap for experiment 2 (both conditions, for "
                             "fairness) -- kept low because the Gibbs variant is genuinely "
                             "slow (real MCMC every E-step); this trades full convergence "
                             "for being runnable at all")
    parser.add_argument("--n-gibbs-samples", type=int, default=60,
                        help="Gibbs samples per E-step call in experiment 2 (cheap by "
                             "necessity -- this runs every EM iteration, unlike experiment "
                             "1's one-off high-precision reference)")
    parser.add_argument("--n-gibbs-burnin", type=int, default=15)
    parser.add_argument("--n-per-sample", type=int, default=40,
                        help="Observations per sample class in experiment 2 (small by "
                             "necessity -- see --n-gibbs-samples)")
    args = parser.parse_args()

    direct_rows = run_direct_bias(n_m_vectors=args.n_m_vectors)
    _report_direct_bias(direct_rows)
    _write_csv(direct_rows, "sim_gibbs_direct_bias")

    e2e_rows = run_e2e(args.n_seeds, args.max_em_iters, args.n_gibbs_samples,
                       args.n_gibbs_burnin, args.n_per_sample)
    _report_e2e(e2e_rows)
    _write_csv(e2e_rows, "sim_gibbs_e2e_recovery")
    print()


if __name__ == "__main__":
    main()
