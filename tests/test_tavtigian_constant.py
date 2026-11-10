"""Tavtigian c: exact default vs. the manuscript's 3-decimal rounding.

Run with:
    source activate excalibr
    pytest tests/test_tavtigian_constant.py -v
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.assay_calibration.fit_utils.evidence_thresholds as et  # noqa: E402

GRID = [float(q) for q in np.round(np.arange(0.01, 0.991, 0.01), 2)]
POINTS = np.array([1, 2, 4, 8])
P_RULES = et._P_RULES + [[0, 1, 0, 1]]   # modified rules: 1 VS + 1 M is pathogenic
LP_RULES = et._LP_RULES + [[0, 0, 2, 0]]  # ... and 2 S is likely pathogenic


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(et.TAVTIGIAN_POSTERIOR_DECIMALS_ENV, raising=False)


def _posterior(c, prior, points):
    lr = c ** (points / 8)
    return lr * prior / ((lr - 1) * prior + 1)


def _meets_all_rules(c, prior, tol=1e-12):
    return (all(_posterior(c, prior, np.dot(r, POINTS)) >= 0.99 - tol for r in P_RULES)
            and all(_posterior(c, prior, np.dot(r, POINTS)) >= 0.90 - tol for r in LP_RULES)
            and all(_posterior(c, prior, -np.dot(r, POINTS)) <= 0.01 + tol for r in et._B_RULES)
            and all(_posterior(c, prior, -np.dot(r, POINTS)) <= 0.10 + tol for r in et._LB_RULES))


def test_closed_form_at_prior_01():
    # 6 points must reach 0.90 at prior 0.1: c^(3/4) = 81 -> c = 81^(4/3)
    assert et.tavtigian_constant_closed_form(0.1) == pytest.approx(81 ** (4 / 3))
    assert et.get_tavtigian_constant(0.1) == 351


@pytest.mark.parametrize("prior", GRID)
def test_exact_default_is_ceil_of_closed_form(prior):
    c = et.get_tavtigian_constant(prior)
    assert isinstance(c, int)
    assert c == math.ceil(et.tavtigian_constant_closed_form(prior) * (1 - 1e-12))
    assert _meets_all_rules(c, prior)


@pytest.mark.parametrize("prior", [0.02, 0.1, 0.25, 0.4, 0.6])
def test_exact_c_is_smallest(prior):
    c = et.get_tavtigian_constant(prior)
    assert _meets_all_rules(c, prior) and not _meets_all_rules(c - 1, prior, tol=0.0)


def test_rounded_reproduces_manuscript():
    assert et.get_tavtigian_constant(0.1, posterior_decimals=3) == 348
    assert et.get_tavtigian_constant(0.044, posterior_decimals=3) == 1127
    assert et.get_tavtigian_constant(0.25, posterior_decimals=3) == 92
    assert et.get_tavtigian_constant(0.4, posterior_decimals=3) == 1268


@pytest.mark.parametrize("prior", GRID[::7])
def test_no_rule_violations_in_rounded_mode(prior):
    c = et.get_tavtigian_constant(prior, posterior_decimals=3)
    assert int(et._tavtigian_fails(np.array([c]), prior, False, False, decimals=3)[0]) == 0


def test_env_var_and_explicit_override(monkeypatch):
    monkeypatch.setenv(et.TAVTIGIAN_POSTERIOR_DECIMALS_ENV, "3")
    assert et.get_tavtigian_constant(0.1) == 348
    assert et.get_tavtigian_constant(0.1, posterior_decimals=None) == 351
    monkeypatch.setenv(et.TAVTIGIAN_POSTERIOR_DECIMALS_ENV, "none")
    assert et.get_tavtigian_constant(0.1) == 351


def test_analysis_package_pins_manuscript_rounding(monkeypatch):
    import importlib
    import analysis
    importlib.reload(analysis)
    assert et.get_tavtigian_constant(0.1) == 348
