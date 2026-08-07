"""Tests for master-seed-driven bootstrap-composition reproducibility.

See fit_utils.fit.derive_bootstrap_seed / derive_fit_seed and
PipelineConfig.seed's docstring (pipeline/config.py) for the three modes
this exercises:
  - master_seed == 0 (the default): bootstrap composition reproduces the
    historical bootstrap_idx-keyed values exactly, and EM fit_seed becomes
    reproducible (not None) for the first time.
  - master_seed == some other int: composition itself also genuinely
    changes per seed value, deterministically.
  - master_seed is None (explicit opt-out, e.g. CLI --seed none): true
    entropy, non-reproducible by design -- for users who want independent
    random draws across repeated invocations.

Run with:
    source activate excalibr
    pytest tests/test_seed_reproducibility.py -v -s
"""
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.assay_calibration.data_utils.dataset import BasicScoreset
from src.assay_calibration.fit_utils.fit import (
    Fit, DEFAULT_MASTER_SEED, sample_specific_bootstrap,
)

EXAMPLE_DIR = REPO_ROOT / "example"


def _make_univariate_scoreset(seed=0, n_per_sample=150):
    rng = np.random.RandomState(seed)
    scores = np.concatenate([
        rng.normal(-1.5, 0.5, n_per_sample),
        rng.normal(1.5, 0.7, n_per_sample),
        rng.normal(0.0, 1.0, n_per_sample),
    ])
    sa = np.zeros((len(scores), 3), dtype=bool)
    sa[:n_per_sample, 0] = True
    sa[n_per_sample:2 * n_per_sample, 1] = True
    sa[2 * n_per_sample:, 2] = True
    return BasicScoreset(scores=scores, sample_assignments=sa)


# ── in-process: bootstrap-composition semantics (fit_utils.fit) ────────────

def test_default_seed_matches_historical_composition():
    ss = _make_univariate_scoreset(seed=1)
    fitter = Fit(ss)
    for bootstrap_idx in (0, 1, 5):
        jobs = fitter.generate_fit_jobs(
            component_range=[2], bootstrap_seed=bootstrap_idx,
            master_seed=DEFAULT_MASTER_SEED, num_fits=1, check_monotonic=False,
        )
        expected_train, expected_val = sample_specific_bootstrap(
            ss.sample_assignments, bootstrap_idx
        )
        np.testing.assert_array_equal(
            jobs[0]["train_observations"], ss.scores[expected_train]
        )
        if len(expected_val):
            np.testing.assert_array_equal(
                jobs[0]["val_observations"], ss.scores[expected_val]
            )
        assert jobs[0]["kwargs"]["fit_seed"] is not None, (
            "default seed should give a reproducible (non-None) fit_seed"
        )


def test_nonzero_seed_perturbs_composition_deterministically():
    ss = _make_univariate_scoreset(seed=2)
    fitter = Fit(ss)
    bootstrap_idx = 3

    def _run(master_seed):
        jobs = fitter.generate_fit_jobs(
            component_range=[2], bootstrap_seed=bootstrap_idx,
            master_seed=master_seed, num_fits=1, check_monotonic=False,
        )
        return jobs[0]["train_observations"], jobs[0]["kwargs"]["fit_seed"]

    train_1a, fit_seed_1a = _run(1)
    train_1b, fit_seed_1b = _run(1)
    train_2, _ = _run(2)
    train_0, _ = _run(DEFAULT_MASTER_SEED)

    np.testing.assert_array_equal(train_1a, train_1b)  # same seed -> reproducible
    assert fit_seed_1a == fit_seed_1b
    assert not np.array_equal(train_1a, train_2), (
        "different explicit seeds should give different bootstrap composition"
    )
    assert not np.array_equal(train_1a, train_0), (
        "a non-default seed should differ from the default's composition"
    )


def test_none_seed_is_truly_random():
    ss = _make_univariate_scoreset(seed=3)
    fitter = Fit(ss)
    bootstrap_idx = 0

    def _run(master_seed):
        jobs = fitter.generate_fit_jobs(
            component_range=[2], bootstrap_seed=bootstrap_idx,
            master_seed=master_seed, num_fits=1, check_monotonic=False,
        )
        return jobs[0]["train_observations"], jobs[0]["kwargs"]["fit_seed"]

    train_a, fit_seed_a = _run(None)
    train_b, fit_seed_b = _run(None)

    assert fit_seed_a is None and fit_seed_b is None
    assert not np.array_equal(train_a, train_b), (
        "explicit None opt-out should give true randomness, not reproducibility"
    )


# ── end-to-end CLI, small-scale (10 bootstraps x 1 fit/bootstrap) ──────────

def _run_pipeline(name, output_dir, extra_args):
    cmd = [
        sys.executable, str(REPO_ROOT / "run_pipeline.py"),
        "--dataset", str(EXAMPLE_DIR / "brca_findlay_example.csv"),
        "--name", name,
        "--sample-names", "Pathogenic/Likely Pathogenic", "Benign/Likely Benign",
        "gnomAD", "Synonymous",
        "--n-bootstraps", "10", "--fits-per-bootstrap", "1",
        "--output-dir", str(output_dir),
        *extra_args,
    ]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (
        f"run_pipeline.py failed for {name}:\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )


def _load_component_params(output_dir, name):
    path = Path(output_dir) / f"{name}_bootstrap_fits.json.gz"
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    out = []
    for seed_key in sorted(data.keys(), key=int):
        fit3c = data[seed_key].get("3c")
        out.append(fit3c["fit"]["component_params"] if fit3c else None)
    return out


def test_cli_default_seed_is_reproducible(tmp_path):
    _run_pipeline("seed_default_a", tmp_path / "a", [])
    _run_pipeline("seed_default_b", tmp_path / "b", [])
    _run_pipeline("seed_explicit_0", tmp_path / "c", ["--seed", "0"])

    params_a = _load_component_params(tmp_path / "a", "seed_default_a")
    params_b = _load_component_params(tmp_path / "b", "seed_default_b")
    params_c = _load_component_params(tmp_path / "c", "seed_explicit_0")

    assert params_a == params_b, "omitting --seed should be reproducible"
    assert params_a == params_c, "default should match explicit --seed 0"


def test_cli_nonzero_seed_changes_output(tmp_path):
    _run_pipeline("seed_default2", tmp_path / "a", [])
    _run_pipeline("seed_seven", tmp_path / "b", ["--seed", "7"])

    params_default = _load_component_params(tmp_path / "a", "seed_default2")
    params_seven = _load_component_params(tmp_path / "b", "seed_seven")

    assert params_default != params_seven, "--seed 7 should differ from default"


def test_cli_none_seed_is_non_reproducible(tmp_path):
    _run_pipeline("seed_none_a", tmp_path / "a", ["--seed", "none"])
    _run_pipeline("seed_none_b", tmp_path / "b", ["--seed", "none"])

    params_a = _load_component_params(tmp_path / "a", "seed_none_a")
    params_b = _load_component_params(tmp_path / "b", "seed_none_b")

    assert params_a != params_b, "--seed none should give non-reproducible output"


# ── hpc/prepare.py threading, lightweight (direct function calls) ─────────

def _load_hpc_prepare():
    spec = importlib.util.spec_from_file_location(
        "hpc_prepare_under_test", REPO_ROOT / "hpc" / "prepare.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_basicscoreset_df(seed=0, n_per_sample=20):
    rng = np.random.RandomState(seed)
    scores = np.concatenate([
        rng.normal(-1.5, 0.5, n_per_sample),
        rng.normal(0.0, 1.0, n_per_sample),
    ])
    labels = ["0"] * n_per_sample + ["2"] * n_per_sample
    return pd.DataFrame({"score": scores, "sample_assignments": labels})


def test_hpc_prepare_threads_master_seed():
    hpc_prepare = _load_hpc_prepare()
    df = _make_basicscoreset_df()

    def _fit_seed(master_seed):
        jobs = hpc_prepare._process_basicscoreset_dataset(
            dataset_name="test_ds", dataset_df=df, output_dir="/tmp/unused",
            N_BOOTSTRAPS=1, num_fits_by_nc={"default": 1},
            selected_components=[2], master_seed=master_seed,
        )
        return jobs[0]["jobs_2c"][0]["kwargs"]["fit_seed"]

    fit_seed_default_a = _fit_seed(hpc_prepare.DEFAULT_MASTER_SEED)
    fit_seed_default_b = _fit_seed(hpc_prepare.DEFAULT_MASTER_SEED)
    fit_seed_5 = _fit_seed(5)
    fit_seed_none_a = _fit_seed(None)
    fit_seed_none_b = _fit_seed(None)

    assert fit_seed_default_a == fit_seed_default_b
    assert fit_seed_default_a is not None
    assert fit_seed_5 != fit_seed_default_a
    assert fit_seed_none_a is None and fit_seed_none_b is None
