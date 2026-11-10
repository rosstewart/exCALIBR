"""drop_wrong_side_pathogenic_fragments: plain-path rule for raw pathogenic fragments on the far side of the
benign reference (see its docstring)."""
import copy

import numpy as np

from src.assay_calibration.fit_utils.point_ranges import (
    drop_wrong_side_pathogenic_fragments,
    enforce_monotonicity_point_ranges,
    extend_points_to_xlims,
)

PV = [1, 2, 3, 4, 5, 6, 7, 8]


def _empty():
    return {k: [] for k in PV + [-k for k in PV]}


def _post(pr, sr, flipped):
    enforce_monotonicity_point_ranges(pr, PV, sr, scoreset_flipped=flipped, liberal=True)
    extend_points_to_xlims(pr, PV, sr, flipped, inf=True)
    enforce_monotonicity_point_ranges(pr, PV, sr, scoreset_flipped=flipped, liberal=True)
    return pr


def test_expected_side_only_is_noop():
    pr = _empty()
    pr[1] = [[-2.0, -1.0]]; pr[-1] = [[0.5, 2.0]]
    before = copy.deepcopy(pr)
    info = drop_wrong_side_pathogenic_fragments(pr, PV, 0.0, False, support_scores=[-1.5, -1.2])
    assert pr == before and info["action"] == "none" and info["dropped"] == []


def test_both_sides_drops_far_tail_and_fixes_whole_range():
    # BARD1-like: genuine +1 on the low side, spurious +1 island past the benign peak.
    sr = np.linspace(-3, 3, 601)
    pr = _empty()
    pr[1] = [[-3.0, -1.0], [2.5, 3.0]]; pr[-1] = [[-0.5, 2.0]]
    raw = copy.deepcopy(pr)
    current = _post(copy.deepcopy(raw), sr, False)
    # current code keeps the island nearest the benign side and extends it to -inf: +1 over [-inf, 3]
    assert current[1] == [[-np.inf, 3.0]]
    info = drop_wrong_side_pathogenic_fragments(pr, PV, 0.0, False, support_scores=np.r_[np.full(50, -2.0), 2.8])
    assert info["action"] == "dropped_high_side" and info["dropped"] == [[1, 2.5, 3.0]]
    assert info["support_low"] == 50 and info["support_high"] == 1 and info["flag"] is None
    fixed = _post(pr, sr, False)
    assert fixed[1] == [[-np.inf, -1.0]]


def test_flipped_mirror():
    pr = _empty()
    pr[2] = [[1.0, 3.0], [-3.0, -2.5]]
    info = drop_wrong_side_pathogenic_fragments(pr, PV, 0.0, True, support_scores=[2.0, 2.0, -2.7])
    assert info["action"] == "dropped_low_side" and pr[2] == [[1.0, 3.0]]


def test_tie_goes_to_expected_side():
    pr = _empty()
    pr[1] = [[-3.0, -1.0], [2.5, 3.0]]
    info = drop_wrong_side_pathogenic_fragments(pr, PV, 0.0, False, support_scores=[])
    assert info["action"] == "dropped_high_side" and pr[1] == [[-3.0, -1.0]]


def test_opposite_side_only_is_flagged_not_changed():
    pr = _empty()
    pr[3] = [[1.0, 3.0]]
    before = copy.deepcopy(pr)
    info = drop_wrong_side_pathogenic_fragments(pr, PV, 0.0, False, support_scores=[2.0])
    assert pr == before and info["action"] == "flag_opposite_side_only" and info["flag"]


def test_opposite_side_supported_is_flagged_not_changed():
    pr = _empty()
    pr[1] = [[-3.0, -2.8], [1.0, 3.0]]
    before = copy.deepcopy(pr)
    info = drop_wrong_side_pathogenic_fragments(pr, PV, 0.0, False, support_scores=[-2.9, 2.0, 2.0, 2.5])
    assert pr == before and info["action"] == "flag_opposite_side_supported" and info["flag"]


def test_no_benign_center_skips():
    pr = _empty(); pr[1] = [[-3.0, -1.0], [2.5, 3.0]]
    before = copy.deepcopy(pr)
    info = drop_wrong_side_pathogenic_fragments(pr, PV, None, False)
    assert pr == before and info["action"] == "skipped_no_benign_center"


def _asymmetric_ensemble(n_boot=40):
    """Genuine pathogenic rise at low scores plus a spurious far-tail rise at high scores (BARD1-like)."""
    x = np.linspace(-3.0, 3.0, 121)
    log_lr = -3.0 + 9.0 / (1 + np.exp((x + 1.3) / 0.15)) + 7.0 / (1 + np.exp(-(x - 2.6) / 0.15))
    boot = np.tile(log_lr, (n_boot, 1))
    return list(range(n_boot)), np.full(n_boot, 0.1), boot / 2.0, -boot / 2.0, x


def test_oob_plain_path_applies_rule_only_when_enabled():
    from src.assay_calibration.pipeline.variant_evidence import _process_variant_oob
    oob, priors, log_fp, log_fb, x = _asymmetric_ensemble()
    kw = dict(variant_idx=0, oob_indices=oob, score=2.9, priors=priors, log_fp=log_fp, log_fb=log_fb,
              score_range=x, point_values=PV, flipped=False, liberal=True, postprocess=True,
              force_no_postprocess=False, benign_center=0.0)
    _, off = _process_variant_oob(**kw)
    _, on = _process_variant_oob(**kw, drop_wrong_side_pathogenic=True,
                                 pathogenic_side_support_scores=np.full(30, -2.0))
    # a variant in the spurious far tail: pathogenic without the rule, benign with it
    assert off["points"] > 0
    assert on["points"] < 0


def test_bidirectional_oob_branch_ignores_rule():
    from src.assay_calibration.pipeline.variant_evidence import _process_variant_oob
    oob, priors, log_fp, log_fb, x = _asymmetric_ensemble()
    kw = dict(variant_idx=0, oob_indices=oob, score=2.9, priors=priors, log_fp=log_fp, log_fb=log_fb,
              score_range=x, point_values=PV, flipped=False, liberal=True, postprocess=True,
              force_no_postprocess=True, benign_center=0.0)
    _, a = _process_variant_oob(**kw)
    _, b = _process_variant_oob(**kw, drop_wrong_side_pathogenic=True,
                                pathogenic_side_support_scores=np.full(30, -2.0))
    assert a["points"] == b["points"]
