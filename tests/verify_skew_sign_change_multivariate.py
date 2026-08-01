#!/usr/bin/env python3
"""Empirical verification that Delta (skew) signs change during multivariate EM.

Runs end-to-end simulations for the unconstrained multivariate (CFUSN q=1)
fitting path, generating data from a known bivariate skew-normal mixture and
checking whether the fitted Delta direction matches the initialization.

Design:
  - Generate synthetic 2D data from K=2 components with known Delta signs
  - Run all 4 lambdaIndex patterns (covering all ±1 init sign combinations)
  - Record initial Delta sign from lambdaIndex, final Delta sign from result
  - Report how often the direction flips relative to initialization

Usage:
    python tests/verify_skew_sign_change_multivariate.py
    python tests/verify_skew_sign_change_multivariate.py --n-obs 500 --n-reps 20
"""
import argparse
import itertools
import sys
import numpy as np
import pandas as pd
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import tryToFit
from src.assay_calibration.data_utils.dataset import BasicMultiScoreset


# ── synthetic data ────────────────────────────────────────────────────────────
# True bivariate skew-normal mixture: X = mu + Delta*|U| + chol(Gamma)*W
# where U ~ N(0,1) and W ~ N2(0,I) are independent.
#
# Component 0 (benign):    mu=(-1.5,-1.2), Delta=(+0.6,+0.4), Gamma=0.4*I
# Component 1 (pathogenic): mu=(+1.5,+1.2), Delta=(-0.6,-0.4), Gamma=0.4*I
# Component 2 (VUS):        mu=(0,0),        Delta=0 (symmetric), Gamma=0.8*I

TRUE_PARAMS = [
    dict(mu=np.array([-1.5, -1.2]), Delta=np.array([+0.6, +0.4]), Gamma=0.4 * np.eye(2)),
    dict(mu=np.array([+1.5, +1.2]), Delta=np.array([-0.6, -0.4]), Gamma=0.4 * np.eye(2)),
    dict(mu=np.array([0.0,  0.0 ]), Delta=np.array([0.0,  0.0 ]), Gamma=0.8 * np.eye(2)),
]
N_SAMPLES = 3


def _sample_msn(mu, Delta, Gamma, n, rng):
    """Sample from restricted MSN: X = mu + Delta*|U| + chol(Gamma)*W."""
    U = np.abs(rng.randn(n))                         # (n,) half-normal
    W = rng.multivariate_normal(np.zeros(len(mu)), Gamma, size=n)  # (n, p)
    return mu[None, :] + np.outer(U, Delta) + W


def _generate_data(n_per_sample, seed):
    """3 sample classes, each a mixture-weighted draw from the components."""
    rng = np.random.RandomState(seed)

    # Per-sample component probabilities
    comp_probs = np.array([
        [0.80, 0.05, 0.15],  # benign
        [0.05, 0.80, 0.15],  # pathogenic
        [0.35, 0.35, 0.30],  # VUS
    ])

    all_obs, all_sa = [], []
    for s in range(N_SAMPLES):
        n = n_per_sample
        probs = comp_probs[s]
        comp_idx = rng.choice(len(TRUE_PARAMS), size=n, p=probs / probs.sum())
        obs = np.vstack([
            _sample_msn(TRUE_PARAMS[c]["mu"], TRUE_PARAMS[c]["Delta"],
                        TRUE_PARAMS[c]["Gamma"], 1, rng)[0]
            for c in comp_idx
        ])
        all_obs.append(obs)
        sa = np.zeros(n, dtype=int)
        sa[:] = s
        all_sa.append(sa)

    observations = np.vstack(all_obs)
    sample_assignments = np.concatenate(all_sa)
    return observations, sample_assignments


def _make_scoreset(observations, sample_assignments):
    df = pd.DataFrame({
        "assay0": observations[:, 0],
        "assay1": observations[:, 1],
        "sample_assignments": sample_assignments,
    })
    return BasicMultiScoreset.from_dataframe(
        df, score_cols=["assay0", "assay1"],
        sample_assignments_col="sample_assignments",
    )


def _delta_sign(Delta):
    """Dominant sign of a Delta vector: sign of its projection onto [1,1,...]."""
    Delta = np.asarray(Delta).ravel()
    return int(np.sign(Delta.mean())) or 1


def _run_fits(observations, sample_assignments, n_reps, seed):
    """Run n_reps MV fits cycling through all lambdaIndex patterns.

    Returns list of (lambdaIndex, init_signs, final_signs, converged).
    """
    K = 2
    lambda_table = list(itertools.product([-1, 1], repeat=K))
    n_patterns = len(lambda_table)
    results = []

    sa_matrix = np.zeros((len(observations), N_SAMPLES), dtype=bool)
    for s in range(N_SAMPLES):
        sa_matrix[:, s] = sample_assignments == s

    for i in range(n_reps):
        lam_idx = i % n_patterns

        rng = np.random.RandomState(seed + i)
        n = len(observations)
        idx = rng.permutation(n)
        train_idx = idx[:int(0.8 * n)]

        result = tryToFit(
            observations[train_idx],
            sa_matrix[train_idx],
            num_components=K,
            constrained=False,        # unconstrained MV
            init_method="kmeans",
            init_constraint_adjustment="scale",
            multivariate=True,
            lambdaIndex=lam_idx,
            score_min=float(observations.min()),
            score_max=float(observations.max()),
            check_monotonic=False,
            num_fits=1,
            fit_seed=int(seed + i),
        )

        params = result.get("component_params", [])
        init_params = result.get("initial_params", [])
        if not params or any(len(p) == 0 for p in params) \
                or not init_params or any(len(p) == 0 for p in init_params):
            continue

        # Sort components by mu[0] for consistent ordering. Read the init
        # sign from the fit's own returned initial_params (sorted the same
        # way below) instead of lambda_table[lam_idx] -- that raw table
        # entry is in kmeans's unsorted, per-restart-random cluster-label
        # order and does not reliably correspond to the mu[0]-sorted order
        # used for final_signs (see the identical fix + measured 8/40 false
        # positive rate in tests/verify_skew_sign_change_univariate.py).
        params_sorted = sorted(params, key=lambda p: np.asarray(p[0])[0])
        final_signs = tuple(_delta_sign(np.asarray(p[1])) for p in params_sorted)
        init_sorted = sorted(init_params, key=lambda p: np.asarray(p[0])[0])
        init_signs = tuple(_delta_sign(np.asarray(p[1])) for p in init_sorted)

        results.append((lam_idx, init_signs, final_signs))

    return results


def _report(results, lambda_table):
    print(f"\n{'═' * 60}")
    print("  MULTIVARIATE UNCONSTRAINED (q=1 restricted MSN)")
    print(f"{'═' * 60}")

    total = len(results)
    if total == 0:
        print("  No successful fits.")
        return

    sign_changed = sum(1 for _, init, final in results if init != final)
    print(f"  Fits: {total}   Delta sign changes: {sign_changed} ({100*sign_changed/total:.0f}%)")

    print(f"\n  {'lambdaIndex':>12}  {'init_signs':>12}  {'modal_final':>12}  {'changed':>10}")
    print(f"  {'─'*52}")

    by_pattern = {}
    for lam_idx, init, final in results:
        by_pattern.setdefault(lam_idx, []).append((init, final))

    for lam_idx in sorted(by_pattern.keys()):
        rows = by_pattern[lam_idx]
        # init_signs is read per-restart from the fit's own initial_params
        # (see _run_fits), so it isn't guaranteed identical across every rep
        # sharing this lam_idx -- compare each row against its own init.
        init = rows[0][0]
        n_changed = sum(1 for i, f in rows if f != i)
        final_counts = {}
        for _, f in rows:
            final_counts[f] = final_counts.get(f, 0) + 1
        modal_final = max(final_counts, key=final_counts.get)
        mark = "✓ changed" if n_changed > 0 else "  same   "
        print(f"  {lam_idx:>12}  {str(init):>12}  {str(modal_final):>12}  {mark:>10}")

    print(f"\n  True Delta signs: {tuple(_delta_sign(p['Delta']) for p in TRUE_PARAMS[:2])}")
    print(f"  (sorted by mu[0]: benign then pathogenic)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-obs", type=int, default=300)
    parser.add_argument("--n-reps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lambda_table = list(itertools.product([-1, 1], repeat=2))

    print(f"True Delta signs: {[_delta_sign(p['Delta']) for p in TRUE_PARAMS[:2]]}")
    print(f"True params (mu, Delta):")
    for i, p in enumerate(TRUE_PARAMS):
        print(f"  component {i}: mu={p['mu']}, Delta={p['Delta']}")
    print(f"n_obs/sample={args.n_obs}, n_reps={args.n_reps}, seed={args.seed}")

    observations, sample_assignments = _generate_data(args.n_obs, args.seed)
    print(f"Generated {len(observations)} 2D observations from {N_SAMPLES} sample classes")

    results = _run_fits(observations, sample_assignments, args.n_reps, args.seed)
    _report(results, lambda_table)
    print()


if __name__ == "__main__":
    main()
