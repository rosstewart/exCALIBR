"""combined_calibration_clusters (multivariate_data/combined.py)."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.assay_calibration.multivariate_data.combined import combined_calibration_clusters  # noqa: E402

NAMES = ["REVEL", "MutPred2", "AlphaMissense", "A1", "A2", "B1", "B2", "C1"]


def _obs():
    # predictors everywhere; functional blocks A (rows 0-49), B (50-99), C (100-119)
    # never share a variant
    X = np.full((150, 8), np.nan)
    X[:, :3] = np.random.default_rng(0).normal(size=(150, 3))
    X[:50, 3:5] = 1.0
    X[50:100, 5:7] = 1.0
    X[100:120, 7] = 1.0
    return X


def test_functional_clusters_each_get_the_predictors():
    cl = combined_calibration_clusters(_obs(), NAMES, min_overlap_rows=1)
    assert sorted(map(tuple, cl)) == [(0, 1, 2, 3, 4), (0, 1, 2, 5, 6)]   # isolated C1 dropped


def test_single_functional_readout_keeps_all_columns():
    X = _obs()[:, [0, 1, 2, 3]]
    assert combined_calibration_clusters(X, NAMES[:4], min_overlap_rows=1) == [[0, 1, 2, 3]]
