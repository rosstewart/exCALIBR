"""Regression cases for the categorical-dimension feature.

The categorical location-model support must leave every fit that passes no
categorical spec byte-identical to the code before it existed. This module
defines a fixed set of small, fast fits; `make_golden()` stores their outputs
once (from the pre-change code), and tests/test_categorical_cfusn.py re-runs
`run_regression_cases()` and compares bitwise.

Regenerate (only from a known-good baseline):
    source activate excalibr
    OMP_NUM_THREADS=8 taskset -c 0-7 python tests/golden/categorical_regression_cases.py
"""
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "cfusn_simulations"))

from src.assay_calibration.fit_utils.cfusn.fit import single_fit, add_component_to_fit
from src.assay_calibration.fit_utils.fit import Fit
from sim_utils import sample_cfusn_mixture, inject_missingness

GOLDEN_PATH = Path(__file__).with_name("categorical_regression_golden.pkl")

_TRUE_PARAMS = [
    (np.array([0.0, 0.0, 0.0]),
     np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 0.3]]),
     np.eye(3) * 0.6),
    (np.array([2.5, 2.0, -1.5]),
     np.array([[-0.8, 0.0], [0.0, 0.6], [0.2, 0.0]]),
     np.eye(3) * 0.5),
]
_WEIGHTS = np.array([[0.7, 0.3], [0.25, 0.75]])
_SIZES = [220, 180]


def make_data(seed=0):
    rng = np.random.RandomState(seed)
    X, S, _ = sample_cfusn_mixture(_TRUE_PARAMS, _WEIGHTS, _SIZES, rng)
    X = inject_missingness(X, [0.0, 0.15, 0.6], rng)
    return X, S


def _clean(result):
    """Keep only deterministic, comparable fields of a single_fit result."""
    keep = {}
    for key in ("component_params", "weights", "likelihoods", "overshoot_stopped", "latent_q"):
        if key in result:
            keep[key] = result[key]
    return keep


def run_regression_cases():
    X, S = make_data()
    common = dict(constrained=False, init_method="kmeans", init_constraint_adjustment=None,
                  multivariate=True, max_em_iters=25, n_mc_truncated=50, verbose=False,
                  raise_on_error=True)
    out = {}
    out["q2"] = _clean(single_fit(X, S, 2, latent_q=2, fit_seed=0, **common))
    out["q1"] = _clean(single_fit(X, S, 2, latent_q=1, fit_seed=1, **common))
    out["gaussian"] = _clean(single_fit(X, S, 2, latent_q=2, fit_seed=2,
                                        force_gaussian=True, **common))
    grown = add_component_to_fit(
        X, S, single_fit(X, S, 2, latent_q=2, fit_seed=3, **common),
        target_sample_idx=1, latent_q=2, max_em_iters=15, n_mc_truncated=50,
        verbose=False, fit_seed=4,
    )
    out["grown"] = _clean(grown)

    scoreset = SimpleNamespace(scores=X, sample_assignments=S)
    jobs = Fit(scoreset).generate_fit_jobs(
        [2], num_fits=3, bootstrap=True, check_monotonic=False, master_seed=0,
        bootstrap_seed=0, max_em_iters=20, n_mc_truncated=50,
    )
    out["jobs"] = []
    for job in jobs:
        res = Fit.execute_fit_job(job)
        out["jobs"].append({
            "val_ll": res["val_ll"],
            "component_params": res["fit"]["component_params"],
            "weights": res["fit"]["weights"],
            "train_observations": job["train_observations"],
        })
    return out


def make_golden():
    out = run_regression_cases()
    with open(GOLDEN_PATH, "wb") as f:
        pickle.dump(out, f)
    print("wrote", GOLDEN_PATH)


if __name__ == "__main__":
    make_golden()
