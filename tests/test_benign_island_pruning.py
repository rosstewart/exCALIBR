"""Tests for prune_small_benign_blocks (auto-bidirectional benign-side pruning).

On the auto-bidirectional path, clean_benign_fragments_no_extend keeps benign
evidence wherever the aggregated LR+ curve crosses a benign threshold, without
extending it to the score-range edges, so a curve that barely dips below a
threshold leaves an isolated sliver of benign evidence (e.g. CARD11
gof_clinvar_v2: -1 over score 0.063-0.070, 0.45% of the gnomAD sample).
prune_small_benign_blocks removes any contiguous benign block holding less than
min_frac of the population sample, judging whole blocks so that a genuine
benign core's thin graded shoulders survive.

Run with:
    source activate excalibr
    pytest tests/test_benign_island_pruning.py -v
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.assay_calibration.fit_utils.point_ranges import prune_small_benign_blocks
from src.assay_calibration.pipeline.variant_evidence import _process_variant_oob

POINT_VALUES = [1, 2, 3, 4, 5]


def _empty_ranges():
    return {pv: [] for pv in POINT_VALUES + [-pv for pv in POINT_VALUES]}


def test_tiny_interior_block_is_pruned():
    pr = _empty_ranges()
    pr[2] = [[-np.inf, -1.0]]
    pr[-1] = [[0.0625, 0.0700]]
    reference = np.linspace(-3.0, 3.0, 1001)  # ~0.13% of scores fall in the sliver

    pruned = prune_small_benign_blocks(pr, POINT_VALUES, reference, min_frac=0.01)

    assert pr[-1] == []
    assert pr[2] == [[-np.inf, -1.0]], "pathogenic levels must be untouched"
    assert len(pruned) == 1
    lo, hi, share = pruned[0]
    assert (lo, hi) == (0.0625, 0.0700)
    assert share < 0.01


def test_large_multilevel_block_kept_whole_including_thin_shoulders():
    pr = _empty_ranges()
    # -1 / -2 / -1 contiguous core plus a near-empty -1 shoulder segment that
    # would fall below 1% on its own.
    pr[-1] = [[-1.0, -0.5], [0.5, 1.0], [1.0, 1.001]]
    pr[-2] = [[-0.5, 0.5]]
    reference = np.linspace(-3.0, 3.0, 1001)
    before = {k: [list(r) for r in v] for k, v in pr.items()}

    pruned = prune_small_benign_blocks(pr, POINT_VALUES, reference, min_frac=0.01)

    assert pruned == []
    assert pr == before


def test_indeterminate_gap_splits_blocks():
    pr = _empty_ranges()
    pr[-2] = [[-1.0, 1.0]]                 # large block, kept
    pr[-1] = [[2.0000, 2.0050]]            # separated by a gap -> its own tiny block
    reference = np.linspace(-3.0, 3.0, 1001)

    pruned = prune_small_benign_blocks(pr, POINT_VALUES, reference, min_frac=0.01)

    assert pr[-2] == [[-1.0, 1.0]]
    assert pr[-1] == []
    assert [p[:2] for p in pruned] == [[2.0, 2.005]]


def test_min_frac_zero_is_noop():
    """min_frac=0 is what the hidden --no-benign-island-prune flag sets."""
    pr = _empty_ranges()
    pr[-1] = [[0.0625, 0.0700]]
    reference = np.linspace(-3.0, 3.0, 1001)

    assert prune_small_benign_blocks(pr, POINT_VALUES, reference, min_frac=0.0) == []
    assert pr[-1] == [[0.0625, 0.0700]]


def _u_shaped_ensemble(n_boot=40, n_grid=121):
    """Same synthetic LoF/GoF U-shape as tests/test_oob_bidirectional_postprocess.py:
    pathogenic at both extremes, benign in the middle (|x| < 1)."""
    score_range = np.linspace(-3.0, 3.0, n_grid)
    log_lr = 3.0 * (score_range ** 2 - 1.0)
    log_lr_boot = np.tile(log_lr, (n_boot, 1))
    return (list(range(n_boot)), np.full(n_boot, 0.1),
            log_lr_boot / 2.0, -log_lr_boot / 2.0, score_range)


def test_oob_applies_same_pruning():
    oob_indices, priors, log_fp, log_fb, score_range = _u_shaped_ensemble()
    common = dict(variant_idx=0, oob_indices=oob_indices, score=0.0, priors=priors,
                  log_fp=log_fp, log_fb=log_fb, score_range=score_range,
                  point_values=POINT_VALUES, flipped=False, liberal=True,
                  postprocess=True, force_no_postprocess=True, benign_center=0.0)

    # Population spread over the whole axis: the benign core holds ~1/3 of it.
    _, kept = _process_variant_oob(**common,
                                   benign_island_reference_scores=np.linspace(-3.0, 3.0, 1001),
                                   benign_island_min_frac=0.01)
    # Population entirely outside the benign core: share 0 -> pruned.
    _, pruned = _process_variant_oob(**common,
                                     benign_island_reference_scores=np.full(500, 2.5),
                                     benign_island_min_frac=0.01)
    # No reference passed (calibration reconstructed from disk): no pruning.
    _, default = _process_variant_oob(**common)

    assert kept["points"] < 0
    assert pruned["points"] == 0
    assert default["points"] == kept["points"]
