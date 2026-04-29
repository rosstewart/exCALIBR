"""
Benchmark + math-invariance harness for CFUSN fit speedups.

Usage
-----
    # Capture a baseline fingerprint + timing
    python test/bench_cfusn_speedup.py baseline

    # After making code changes, verify the iterate sequence is unchanged
    # and report the new median per-fit time
    python test/bench_cfusn_speedup.py compare

The harness fits a small synthetic 3D, q=2, K=2 mixture with a fixed RNG seed.
It records (component_params, weights, likelihoods) for the first fit and
median wall-clock over a small sample of fits. Compare mode asserts the
fingerprint matches the baseline within tight tolerances and prints the
speedup factor.
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.assay_calibration.fit_utils.cfusn.fit import single_fit


N_SYN = 800   # observations
P_DIM = 3     # data dim
Q_LATENT = 2  # CFUSN latent dim
K = 2         # mixture components
N_SAMPLES = 4 # P/LP, B/LB, gnomAD, syn
N_FITS_TIMING = 5
SEED = 12345
MISSING_FRAC = 0.15  # fraction of (variant, dim) cells that are NaN

BASELINE_PATH = Path(__file__).parent / "_bench_cfusn_baseline.pkl"


def make_synthetic_data(seed: int = SEED):
    """Build a reproducible 3D mixture with missingness + 4 sample classes."""
    rng = np.random.default_rng(seed)

    # True components
    means = np.array([[-1.5, -1.0, -0.8], [1.2, 0.9, 1.1]])
    cov_a = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])
    cov_b = np.array([[0.8, -0.1, 0.0], [-0.1, 0.9, 0.2], [0.0, 0.2, 0.7]])
    covs = [cov_a, cov_b]
    mixture_weights = np.array([0.55, 0.45])

    # Generate observations
    comp_choice = rng.choice(K, size=N_SYN, p=mixture_weights)
    obs = np.zeros((N_SYN, P_DIM))
    for k in range(K):
        mask = comp_choice == k
        obs[mask] = rng.multivariate_normal(means[k], covs[k], size=mask.sum())

    # Inject missingness
    nan_mask = rng.random(obs.shape) < MISSING_FRAC
    obs[nan_mask] = np.nan
    # Ensure no all-NaN rows
    all_nan = np.isnan(obs).all(axis=1)
    if all_nan.any():
        obs[all_nan, 0] = rng.normal(0, 1, all_nan.sum())

    # Sample assignments — 4 classes, roughly balanced, one-hot
    sample_assign = np.zeros((N_SYN, N_SAMPLES), dtype=bool)
    sa_choice = rng.integers(0, N_SAMPLES, size=N_SYN)
    sample_assign[np.arange(N_SYN), sa_choice] = True

    return obs, sample_assign


def serialize_params(params):
    """Convert params list of (mu, Delta, Gamma) to plain numpy."""
    out = []
    for mu, Delta, Gamma in params:
        out.append((np.asarray(mu, float), np.asarray(Delta, float),
                    np.asarray(Gamma, float)))
    return out


def fit_once(obs, sample_assign, max_em_iters=200, seed=SEED):
    """Run one CFUSN fit with a fixed init seed for reproducibility.

    latent_q is left at the cfusn/fit.py default (2) — passing it
    explicitly hits a pre-existing duplicate-kwarg edge case in
    kmeans_init_mv that production also avoids by relying on the default.
    """
    np.random.seed(seed)
    return single_fit(
        observations=obs,
        sample_indicators=sample_assign,
        N_components=K,
        constrained=False,
        init_method="kmeans",
        init_constraint_adjustment="scale",
        multivariate=True,
        max_em_iters=max_em_iters,
        early_stopping=True,
        verbose=False,
        bootstrap_seed=seed,
    )


def fingerprint(result):
    """Extract a comparable fingerprint from a fit result."""
    return {
        "component_params": serialize_params(result["component_params"]),
        "weights": np.asarray(result["weights"], float),
        "likelihoods": np.asarray(result["likelihoods"], float),
        "n_iters": len(result["likelihoods"]),
    }


def time_fits(obs, sample_assign, n_fits=N_FITS_TIMING):
    """Return median wall-clock per fit across n_fits runs (different seeds)."""
    times = []
    for i in range(n_fits):
        np.random.seed(SEED + i)
        t0 = time.perf_counter()
        single_fit(
            observations=obs,
            sample_indicators=sample_assign,
            N_components=K,
            constrained=False,
            init_method="kmeans",
            init_constraint_adjustment="scale",
            multivariate=True,
            max_em_iters=200,
            early_stopping=True,
            verbose=False,
            bootstrap_seed=SEED + i,
        )
        times.append(time.perf_counter() - t0)
    return np.median(times), times


def cmd_baseline():
    obs, sample_assign = make_synthetic_data()
    print(f"[baseline] data: N={N_SYN}, p={P_DIM}, q={Q_LATENT}, K={K}, "
          f"missing={MISSING_FRAC:.0%}")

    print(f"[baseline] running 1 fit for fingerprint...")
    result = fit_once(obs, sample_assign)
    fp = fingerprint(result)
    print(f"[baseline]   converged in {fp['n_iters']} iterations, "
          f"final LL={fp['likelihoods'][-1]:.6f}")

    print(f"[baseline] timing {N_FITS_TIMING} fits...")
    median_t, times = time_fits(obs, sample_assign)
    print(f"[baseline]   per-fit times: "
          f"{[f'{t:.3f}s' for t in times]}")
    print(f"[baseline]   median: {median_t:.3f}s")

    with open(BASELINE_PATH, "wb") as f:
        pickle.dump({
            "fingerprint": fp,
            "median_time": median_t,
            "times": times,
        }, f)
    print(f"[baseline] saved to {BASELINE_PATH}")


def assert_fingerprint_match(new_fp, base_fp,
                              params_atol=1e-9, weights_atol=1e-9,
                              ll_atol=1e-7):
    """Assert that fingerprints match within tolerances."""
    if new_fp["n_iters"] != base_fp["n_iters"]:
        raise AssertionError(
            f"iteration count changed: baseline={base_fp['n_iters']} "
            f"vs new={new_fp['n_iters']}"
        )
    np.testing.assert_allclose(
        new_fp["weights"], base_fp["weights"], atol=weights_atol,
        err_msg="weights drifted",
    )
    np.testing.assert_allclose(
        new_fp["likelihoods"], base_fp["likelihoods"], atol=ll_atol,
        err_msg="LL trajectory drifted",
    )
    for k, ((mu_n, D_n, G_n), (mu_b, D_b, G_b)) in enumerate(zip(
        new_fp["component_params"], base_fp["component_params"]
    )):
        np.testing.assert_allclose(mu_n, mu_b, atol=params_atol,
                                   err_msg=f"mu[{k}] drifted")
        np.testing.assert_allclose(D_n, D_b, atol=params_atol,
                                   err_msg=f"Delta[{k}] drifted")
        np.testing.assert_allclose(G_n, G_b, atol=params_atol,
                                   err_msg=f"Gamma[{k}] drifted")


def cmd_compare():
    if not BASELINE_PATH.exists():
        sys.exit(f"baseline missing — run `python {__file__} baseline` first")

    with open(BASELINE_PATH, "rb") as f:
        base = pickle.load(f)
    base_fp = base["fingerprint"]
    base_median = base["median_time"]

    obs, sample_assign = make_synthetic_data()
    print(f"[compare] data: N={N_SYN}, p={P_DIM}, q={Q_LATENT}, K={K}, "
          f"missing={MISSING_FRAC:.0%}")
    print(f"[compare] checking math invariance...")
    new_fp = fingerprint(fit_once(obs, sample_assign))
    assert_fingerprint_match(new_fp, base_fp)
    print(f"[compare]   PASSED (n_iters={new_fp['n_iters']}, "
          f"final LL={new_fp['likelihoods'][-1]:.6f})")

    print(f"[compare] timing {N_FITS_TIMING} fits...")
    new_median, times = time_fits(obs, sample_assign)
    print(f"[compare]   per-fit times: "
          f"{[f'{t:.3f}s' for t in times]}")
    print(f"[compare]   median: {new_median:.3f}s "
          f"(baseline {base_median:.3f}s)")
    print(f"[compare]   speedup: {base_median / new_median:.2f}x")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"baseline", "compare"}:
        print(f"usage: python {sys.argv[0]} {{baseline|compare}}")
        sys.exit(1)
    {"baseline": cmd_baseline, "compare": cmd_compare}[sys.argv[1]]()
