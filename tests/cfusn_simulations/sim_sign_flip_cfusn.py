#!/usr/bin/env python3
"""Empirical verification that Delta (skew) signs change during REAL CFUSN
(q>=2) EM -- extends tests/verify_skew_sign_change_multivariate.py (which
only exercises q=1 restricted MSN, via _em_update_multivariate) to the full
CFUSN path (_em_update_cfusn), by passing latent_q=2 into tryToFit, matching
what every production gene-set fit actually uses.

Design (mirrors the existing verify_skew_sign_change_*.py convention):
  - Generate synthetic p-D data from K components with known Delta (q=2)
  - Run many fits cycling through all lambdaIndex init-sign patterns
    (4**K patterns at q=2 -- kmeans_init_mv's n_sign_per_cluster = 2**latent_q)
  - For each fit, match init components to final components (Hungarian on
    mu/Omega, via sim_utils.match_components), then for each matched pair
    resolve the best column permutation/sign of final Delta against init
    Delta (sim_utils.resolve_delta_ambiguity) and report whether that
    resolution required an actual sign flip and/or column permutation --
    i.e. did EM's own M-step change the skew direction relative to where it
    started, not relative to ground truth.

Usage:
    python tests/cfusn_simulations/sim_sign_flip_cfusn.py
    python tests/cfusn_simulations/sim_sign_flip_cfusn.py --n-obs 500 --n-reps 200
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assay_calibration.fit_utils.fit import tryToFit
from tests.cfusn_simulations.sim_utils import sample_cfusn_mixture, match_components, resolve_delta_ambiguity


# ── synthetic data (q=2 CFUSN, mirrors verify_skew_sign_change_multivariate.py) ──

TRUE_PARAMS = [
    dict(mu=np.array([-1.5, -1.2]), Delta=np.array([[+0.6, +0.2], [+0.4, -0.15]]), Gamma=0.3 * np.eye(2)),
    dict(mu=np.array([+1.5, +1.2]), Delta=np.array([[-0.6, -0.2], [-0.4, +0.15]]), Gamma=0.3 * np.eye(2)),
]
K = len(TRUE_PARAMS)
N_SAMPLES = 3  # benign / pathogenic / VUS
COMP_PROBS = np.array([
    [0.85, 0.15],   # benign
    [0.15, 0.85],   # pathogenic
    [0.50, 0.50],   # VUS
])


def _generate_data(n_per_sample, seed):
    rng = np.random.RandomState(seed)
    component_params = [(p["mu"], p["Delta"], p["Gamma"]) for p in TRUE_PARAMS]
    weights_per_sample = COMP_PROBS
    sample_sizes = np.full(N_SAMPLES, n_per_sample)
    X, sa, _ = sample_cfusn_mixture(component_params, weights_per_sample, sample_sizes, rng)
    return X, sa


def _run_fits(observations, sample_indicators, n_reps, seed):
    """Returns list of dicts: {lam_idx, n_matched, n_changed, changed_details}."""
    n_patterns = 4 ** K   # kmeans_init_mv: n_sign_per_cluster = 2**latent_q = 4 for q=2
    results = []

    for i in range(n_reps):
        lam_idx = i % n_patterns
        rng = np.random.RandomState(seed + i)
        n = len(observations)
        idx = rng.permutation(n)
        train_idx = idx[: int(0.8 * n)]

        result = tryToFit(
            observations[train_idx],
            sample_indicators[train_idx],
            num_components=K,
            constrained=False,
            init_method="kmeans",
            init_constraint_adjustment="scale",
            multivariate=True,
            latent_q=2,
            lambdaIndex=lam_idx,
            check_monotonic=False,
            num_fits=1,
            fit_seed=int(seed + i),
            verbose=False,
            verbose_init=False,
        )

        params = result.get("component_params", [])
        init_params = result.get("initial_params", [])
        if not params or any(len(p) == 0 for p in params) \
                or not init_params or any(len(p) == 0 for p in init_params):
            continue

        row_ind, col_ind, _ = match_components(init_params, params)
        n_matched = len(row_ind)
        n_changed = 0
        col_flip_counts = np.zeros(2, dtype=int)  # q=2
        for ti, fi in zip(row_ind, col_ind):
            Delta_init = np.asarray(init_params[ti][1])
            Delta_final = np.asarray(params[fi][1])
            _, perm, signs, _ = resolve_delta_ambiguity(Delta_init, Delta_final)
            changed = (tuple(perm) != (0, 1)) or any(s == -1 for s in signs)
            if changed:
                n_changed += 1
            for col, s in enumerate(signs):
                if s == -1:
                    col_flip_counts[col] += 1

        results.append(dict(
            lam_idx=lam_idx, n_matched=n_matched, n_changed=n_changed,
            col_flip_counts=col_flip_counts,
        ))

    return results


def _report(results):
    print(f"\n{'═' * 60}")
    print("  CFUSN (q=2) SIGN-FLIP CHECK -- real _em_update_cfusn path")
    print(f"{'═' * 60}")

    total_fits = len(results)
    if total_fits == 0:
        print("  No successful fits.")
        return

    total_components = sum(r["n_matched"] for r in results)
    total_changed = sum(r["n_changed"] for r in results)
    total_col_flips = np.sum([r["col_flip_counts"] for r in results], axis=0)

    print(f"  Fits: {total_fits}   Matched components: {total_components}")
    print(f"  Components with a changed (perm/sign) Delta vs. init: "
          f"{total_changed} ({100 * total_changed / max(total_components, 1):.1f}%)")
    for col in range(2):
        print(f"  Column {col} sign-flip rate: "
              f"{total_col_flips[col]} / {total_components} "
              f"({100 * total_col_flips[col] / max(total_components, 1):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-obs", type=int, default=300, help="Observations per sample class")
    parser.add_argument("--n-reps", type=int, default=128,
                        help="Fits (covers all 4**K=16 lambdaIndex patterns multiple times)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"True params (mu, Delta):")
    for i, p in enumerate(TRUE_PARAMS):
        print(f"  component {i}: mu={p['mu']}, Delta=\n{p['Delta']}")
    print(f"n_obs/sample={args.n_obs}, n_reps={args.n_reps}, seed={args.seed}")

    observations, sample_indicators = _generate_data(args.n_obs, args.seed)
    print(f"Generated {len(observations)} 2D observations from {N_SAMPLES} sample classes")

    results = _run_fits(observations, sample_indicators, args.n_reps, args.seed)
    _report(results)
    print()


if __name__ == "__main__":
    main()
