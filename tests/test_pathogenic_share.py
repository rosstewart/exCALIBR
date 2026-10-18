"""Pathogenic-share weighting: compute_sample_weights' opt-in scale anchoring
and hpc/prepare.py's factor rule and condition merge."""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.assay_calibration.fit_utils.cfusn.fit import compute_sample_weights  # noqa: E402
from hpc import prepare  # noqa: E402


def _sa(counts):
    rows = []
    for s, n in enumerate(counts):
        for _ in range(n):
            r = np.zeros(len(counts), bool)
            r[s] = True
            rows.append(r)
    return np.array(rows)


def _ms(counts, n_dims=2):
    sa = _sa(counts)
    return SimpleNamespace(sample_assignments=sa, scores=np.zeros((len(sa), n_dims)))


def test_default_weights_unchanged():
    sa = _sa([5, 10, 40, 20])
    w = compute_sample_weights(sa, sample_proportions=[10, 10, 40, 20])
    assert np.isclose(w.sum(), 1.0)  # historical normalisation kept by default
    assert compute_sample_weights(sa) is None


def test_anchored_scale_min_weight_one():
    sa = _sa([5, 10, 40, 20])
    w = compute_sample_weights(sa, sample_proportions=[15, 10, 40, 20], anchor_weight_scale=True)
    assert np.isclose(w.min(), 1.0)
    assert np.allclose(w[:5], 3.0) and np.allclose(w[5:], 1.0)


def test_factor_rule():
    # CARD11-like: 15 P/LP, 143 B/LB, 670 gnomAD, 418 syn, 6 BENTA, 23 CADINS
    p, f = prepare.pathogenic_share_proportions(_ms([15, 143, 670, 418, 6, 23]), 0.1)
    assert np.isclose(f, 0.1 * (143 + 670 + 418) / (15 + 6 + 23))
    assert np.allclose(np.array(p)[[0, 4, 5]], np.array([15, 6, 23]) * f)
    assert np.allclose(np.array(p)[[1, 2, 3]], [143, 670, 418])


def test_factor_rule_edge_cases():
    assert prepare.pathogenic_share_proportions(_ms([10, 50, 100, 40]), 0) == (None, 1.0)
    assert prepare.pathogenic_share_proportions(_ms([0, 50, 100, 40]), 0.1) == (None, 1.0)  # nothing to weight
    assert prepare.pathogenic_share_proportions(_ms([100, 50, 100, 40]), 0.1) == (None, 1.0)  # share already met
    p, f = prepare.pathogenic_share_proportions(_ms([2, 0, 300, 0]), 0.1)  # no B/LB or syn: weight vs gnomAD
    assert np.isclose(f, 15.0) and np.isclose(p[0], 30.0)


def test_merge_condition_order(tmp_path):
    for label, n in (("tau_0.1", 3), ("tau_0", 2), ("tau_0.2", 4)):
        jobs = [{"dataset_name": f"d{i}", "bootstrap_seed": i, "num_fits_total": 6} for i in range(n)]
        prepare._save_manifest(str(tmp_path / label), jobs, 1000, 10, {"component_range": [3, 4]})
    prepare._merge_condition_manifests(str(tmp_path), ["tau_0.1", "tau_0", "tau_0.2"], 4)
    merged = []
    for f in sorted((tmp_path / "jobs").glob("array_*.pkl")):
        merged += pickle.load(open(f, "rb"))
    assert [j["condition"] for j in merged] == ["tau_0.1"] * 3 + ["tau_0"] * 2 + ["tau_0.2"] * 4
    index = json.load(open(tmp_path / "jobs" / "job_index.json"))
    assert index["condition_order"] == ["tau_0.1", "tau_0", "tau_0.2"]
    assert index["conditions"]["tau_0"]["job_index_range"] == [3, 5]
