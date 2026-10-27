"""Predictor score transforms (multivariate_data/predictors.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.assay_calibration.multivariate_data import predictors as pr  # noqa: E402


@pytest.mark.parametrize("transform", ["logit", "rank_int"])
def test_monotone_and_finite(transform):
    x = np.array([0.0, 0.05, 0.2, 0.2, 0.5, 0.9999, 1.0, np.nan])
    y = pr.transform_predictor_scores(x, x, transform)
    ok = ~np.isnan(x)
    assert np.isnan(y[~ok]).all() and np.isfinite(y[ok]).all()
    assert np.all(np.diff(y[ok]) >= 0) and y[2] == y[3]           # order kept, ties stay tied
    assert np.all(np.diff(y[ok][[0, 1, 2, 4, 5, 6]]) > 0)          # distinct scores stay distinct


def test_rank_int_is_fixed_map():
    rng = np.random.default_rng(0)
    ref = rng.uniform(size=200)
    full = pr.transform_predictor_scores(ref, ref, "rank_int")
    sub = pr.transform_predictor_scores(ref[:50], ref, "rank_int")  # same reference -> same values
    assert np.allclose(sub, full[:50])
    assert abs(np.mean(full)) < 1e-9 and abs(np.std(full) - 1) < 0.05


def test_logit_rejects_out_of_range():
    with pytest.raises(ValueError):
        pr.transform_predictor_scores(np.array([0.5, 1.2]), None, "logit")


def test_scoreset_transform_keeps_rows(monkeypatch):
    df = pd.DataFrame({"protein_variant": ["A1T", "B2C", "C3D", "D4E"],
                       "score": [0.1, 0.99, 1.0, 0.4],
                       "sample_assignments": ["1", "0", "0", "2"]})
    raw = pr.df_to_basic_scoreset(df, "AM")
    lg = pr.df_to_basic_scoreset(df, "AM", transform="logit")
    assert list(raw._ids) == list(lg._ids)
    assert np.allclose(lg.scores, np.log(np.clip(raw.scores, pr.LOGIT_EPS, 1 - pr.LOGIT_EPS))
                       - np.log1p(-np.clip(raw.scores, pr.LOGIT_EPS, 1 - pr.LOGIT_EPS)))
    monkeypatch.setenv(pr.PREDICTOR_TRANSFORM_ENV, "logit")
    assert np.allclose(pr.df_to_basic_scoreset(df, "AM").scores, lg.scores)  # env default
    monkeypatch.setenv(pr.PREDICTOR_TRANSFORM_ENV, "bogus")
    with pytest.raises(ValueError):
        pr.df_to_basic_scoreset(df, "AM")
