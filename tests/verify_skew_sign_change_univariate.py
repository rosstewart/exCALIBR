#!/usr/bin/env python3
"""Empirical verification that skew parameter signs change during EM fitting.

Runs end-to-end simulations — data generation through EM convergence — for
both constrained and unconstrained univariate skew-normal mixture fitting.

Design:
  - Generate synthetic data from K=2 components with known skew signs (+, -)
  - Run all 4 lambdaIndex patterns (covering all ±1 init sign combinations)
  - Record initial sign from lambdaIndex, final sign from fitted component_params
  - Report how often the sign differs between init and converged solution

Usage:
    python tests/verify_skew_sign_change_univariate.py
    python tests/verify_skew_sign_change_univariate.py --n-obs 500 --n-reps 20
"""
import argparse
import itertools
import sys
import numpy as np
import scipy.stats as sps
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import tryToFit


# ── synthetic data ────────────────────────────────────────────────────────────

TRUE_PARAMS_K2 = [
    # (a, loc, scale)  — component 0: benign, component 1: pathogenic
    (+3.0, -1.5, 0.6),
    (-3.0,  1.5, 0.6),
]
TRUE_WEIGHTS = [0.5, 0.3, 0.2]   # [benign, pathogenic, VUS]
N_SAMPLES = 3  # benign / pathogenic / VUS


def _generate_data(n_per_sample, seed):
    """Generate K=2 skew-normal mixture with 3 sample classes."""
    rng = np.random.RandomState(seed)
    n_total = n_per_sample * N_SAMPLES

    # Component assignments for each observation
    comp_probs_by_sample = np.array([
        [0.90, 0.10],   # benign sample: mostly component 0
        [0.10, 0.90],   # pathogenic sample: mostly component 1
        [0.50, 0.50],   # VUS sample: mixed
    ])

    scores = []
    sample_assignments = np.zeros((n_total, N_SAMPLES), dtype=bool)

    for s in range(N_SAMPLES):
        n = n_per_sample
        probs = comp_probs_by_sample[s]
        comp_idx = rng.choice(len(TRUE_PARAMS_K2), size=n, p=probs / probs.sum())
        obs = np.array([
            sps.skewnorm.rvs(TRUE_PARAMS_K2[c][0], TRUE_PARAMS_K2[c][1], TRUE_PARAMS_K2[c][2],
                             random_state=rng)
            for c in comp_idx
        ])
        scores.append(obs)
        sample_assignments[s * n:(s + 1) * n, s] = True

    return np.concatenate(scores), sample_assignments


def _run_fits(observations, sample_indicators, constrained, n_reps, seed):
    """Run n_reps fits cycling through all lambdaIndex patterns.

    Returns list of (lambdaIndex, init_signs, final_signs, val_ll).
    """
    K = 2
    lambda_table = list(itertools.product([-1, 1], repeat=K))
    n_patterns = len(lambda_table)
    results = []

    for i in range(n_reps):
        lam_idx = i % n_patterns
        init_signs = lambda_table[lam_idx]

        # train/val split (simple 80/20)
        rng = np.random.RandomState(seed + i)
        n = len(observations)
        idx = rng.permutation(n)
        train_idx = idx[:int(0.8 * n)]
        val_idx = idx[int(0.8 * n):]

        result = tryToFit(
            observations[train_idx],
            sample_indicators[train_idx],
            num_components=K,
            constrained=constrained,
            init_method="kmeans",
            init_constraint_adjustment="scale",
            lambdaIndex=lam_idx,
            score_min=float(observations.min()),
            score_max=float(observations.max()),
            check_monotonic=constrained,
            num_fits=1,
            fit_seed=int(seed + i),
        )

        params = result.get("component_params", [])
        if not params or any(len(p) == 0 for p in params):
            continue

        # Sort by location to get consistent component ordering
        params_sorted = sorted(params, key=lambda p: p[1])
        final_signs = tuple(int(np.sign(p[0])) for p in params_sorted)

        # val_ll: use all held-out observations
        from scipy.stats import skewnorm
        weights = result.get("weights")
        if weights is not None and len(val_idx) > 0:
            w = weights.mean(axis=0)   # average sample weights
            ll = 0.0
            for obs_i in observations[val_idx]:
                comp_ll = np.array([
                    w[c] * skewnorm.pdf(obs_i, params[c][0], params[c][1], params[c][2])
                    for c in range(K)
                ])
                ll += np.log(max(comp_ll.sum(), 1e-300))
            val_ll = ll / len(val_idx)
        else:
            val_ll = float("nan")

        results.append((lam_idx, init_signs, final_signs, val_ll))

    return results


def _report(label, results, lambda_table):
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"{'═' * 60}")

    total = len(results)
    if total == 0:
        print("  No successful fits.")
        return

    sign_changed = sum(1 for _, init, final, _ in results if init != final)
    print(f"  Fits: {total}   Sign changes: {sign_changed} ({100*sign_changed/total:.0f}%)")

    print(f"\n  {'lambdaIndex':>12}  {'init_signs':>12}  {'final_signs':>12}  {'changed':>8}  {'val_ll':>10}")
    print(f"  {'─'*60}")

    by_pattern = {}
    for lam_idx, init, final, vll in results:
        by_pattern.setdefault(lam_idx, []).append((init, final, vll))

    for lam_idx in sorted(by_pattern.keys()):
        rows = by_pattern[lam_idx]
        init = rows[0][0]
        n_changed = sum(1 for _, f, _ in rows if f != init)
        # most common final sign
        final_counts = {}
        for _, f, _ in rows:
            final_counts[f] = final_counts.get(f, 0) + 1
        modal_final = max(final_counts, key=final_counts.get)
        mean_vll = np.nanmean([v for _, _, v in rows])
        mark = "✓ changed" if n_changed > 0 else "  same   "
        print(f"  {lam_idx:>12}  {str(init):>12}  {str(modal_final):>12}  {mark:>8}  {mean_vll:>10.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-obs", type=int, default=300, help="Observations per sample class")
    parser.add_argument("--n-reps", type=int, default=32, help="Fits per mode (covers all lambdaIndex patterns multiple times)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    lambda_table = list(itertools.product([-1, 1], repeat=2))

    print(f"True component skew signs: {[np.sign(a) for a, _, _ in TRUE_PARAMS_K2]}")
    print(f"True params: {TRUE_PARAMS_K2}")
    print(f"n_obs/sample={args.n_obs}, n_reps={args.n_reps}, seed={args.seed}")

    observations, sample_indicators = _generate_data(args.n_obs, args.seed)
    print(f"Generated {len(observations)} observations from 3 sample classes")

    for constrained, label in [(True, "CONSTRAINED"), (False, "UNCONSTRAINED")]:
        results = _run_fits(observations, sample_indicators, constrained,
                            args.n_reps, args.seed)
        _report(label, results, lambda_table)

    print()


if __name__ == "__main__":
    main()
