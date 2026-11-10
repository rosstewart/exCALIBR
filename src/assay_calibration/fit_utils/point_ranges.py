import sys
sys.path.append('..')
import os
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging
from .fit import (calculate_score_ranges,thresholds_from_prior,calculate_score_ranges_dual)  # noqa: E402
from .cfusn import density_utils  # noqa: E402
from ..data_utils.dataset import Scoreset  # noqa: E402
from .utils import serialize_dict  # noqa: E402
from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# logging.getLogger('matplotlib').setLevel(logging.ERROR)

def enforce_monotonicity_point_ranges(point_ranges, point_values, score_range, scoreset_flipped=False, liberal=False, log_f=None):

    
    if liberal:
        # print('enforcing monotonicity in points...',file=log_f)
        for i in point_values:
            point = i # pathogenic
            if len(point_ranges[point]) != 0:
                point_ranges[point] = [point_ranges[point][-1]] if not scoreset_flipped else [point_ranges[point][0]]
                    
            point = -i # benign
            if len(point_ranges[point]) != 0:
                point_ranges[point] = [point_ranges[point][0]] if not scoreset_flipped else [point_ranges[point][-1]]

        # make sure none overlap
        for i in point_values:
            for j in point_values:
                if j <= i:
                    continue

                
                point_i = i # pathogenic
                point_j = j # pathogenic

                if len(point_ranges[point_i]) != 0 and len(point_ranges[point_j]) != 0:
                    # 2d array, should be flattened but idk if will break things
                    # if point_j encompasses point_i (in case of sudden spike), remove point_i
                    if point_ranges[point_i][0][0] >= point_ranges[point_j][0][0] and point_ranges[point_i][0][1] <= point_ranges[point_j][0][1]:
                        point_ranges[point_i] = []

                
                point_i = -i # benign
                point_j = -j # benign
                
                if len(point_ranges[point_i]) != 0 and len(point_ranges[point_j]) != 0:
                    # print(f"Checking if {point_j} encompasses {point_i}: "
                          # f"{point_ranges[point_i][0]} vs {point_ranges[point_j][0]}", file=log_f)
                    
                    if point_ranges[point_i][0][0] >= point_ranges[point_j][0][0] and \
                       point_ranges[point_i][0][1] <= point_ranges[point_j][0][1]:
                        print(f"  -> Removing {point_i}", file=log_f)
                        point_ranges[point_i] = []
                
        return
    
    max_path_points = None
    max_ben_points = None
    
    # print('enforcing monotonicity in points (std)...',file=log_f)
    for i in point_values:
        point = i # pathogenic

        if abs(point) == 1 and len(point_ranges[point]) != 0 and len(point_ranges[point+1]) == 0: # highest evidence at 1
            l,h = point_ranges[point][0][0], point_ranges[point][-1][-1] # could be more than one range

            # A prior extend_points_to_xlims(inf=True) pass may have already
            # pushed l/h out to -inf/+inf rather than the finite score_range
            # edge -- that still counts as "touching the edge", not as
            # evidence that dipped back to nothing in the interior.
            if (l != score_range[0] and not np.isneginf(l)) and \
               (h != score_range[-1] and not np.isposinf(h)): # evidence not at min or max, evidence goes back to 0. remove
                print(f'supporting evidence ({point}) goes back to no evidence. removing...', file=log_f)
                max_path_points = point

        if max_path_points is not None:
            point_ranges[point] = []
        elif len(point_ranges[point]) > 1: # e.g. --_-

            point_h = point + 1
            if point_h in point_ranges and len(point_ranges[point_h]) != 0 and point_ranges[point_h][0][0] != point_ranges[point][0][-1] and point_ranges[point_h][-1][-1] != point_ranges[point][-1][0]:
                # if dips into no evidence/switches sides and not up into higher point ranges
                idx_to_keep = []
                for range_idx, range_ in enumerate(point_ranges[point]):
                    if range_[0] == point_ranges[point_h][-1][-1] or range_[-1] == point_ranges[point_h][0][0]:
                        # valid range. keep
                        idx_to_keep.append(range_idx)
                print(f'point ranges {point}: before removing dipping {point_ranges[point]}', file=log_f)
                if len(idx_to_keep) == 0:
                    point_ranges[point] = []
                    max_path_points = point
                else:
                    point_ranges[point] = list(np.array(point_ranges[point])[np.array(idx_to_keep)])
                print(f'point ranges {point}: after removing dipping {point_ranges[point]}', file=log_f)

            if len(point_ranges[point]) > 1: # if didn't dip or still needs flattening
                print(f'flattening ({point}): {point_ranges[point]}', file=log_f)
                
                # flatten
                point_ranges[point] = [[point_ranges[point][0][0], point_ranges[point][-1][-1]]]
                if max_path_points is None:
                    max_path_points = point
                
        point = -i # benign

        if abs(point) == 1 and len(point_ranges[point]) != 0 and len(point_ranges[point-1]) == 0: # highest evidence at -1
            l,h = point_ranges[point][0][0], point_ranges[point][-1][-1] # could be more than one range

            if (l != score_range[0] and not np.isneginf(l)) and \
               (h != score_range[-1] and not np.isposinf(h)): # evidence not at min or max, evidence goes back to 0. remove
                print(f'supporting evidence ({point}) goes back to no evidence. removing...', file=log_f)
                max_ben_points = point

        if max_ben_points is not None:
            point_ranges[point] = []
        elif len(point_ranges[point]) > 1: # e.g. --_-

            point_h = point - 1
            if point_h in point_ranges and len(point_ranges[point_h]) != 0 and point_ranges[point_h][0][0] != point_ranges[point][0][-1] and point_ranges[point_h][-1][-1] != point_ranges[point][-1][0]:
                # if dips into no evidence/switches sides and not up into higher point ranges
                idx_to_keep = []
                for range_idx, range_ in enumerate(point_ranges[point]):
                    if range_[0] == point_ranges[point_h][-1][-1] or range_[-1] == point_ranges[point_h][0][0]:
                        # valid range. keep
                        idx_to_keep.append(range_idx)
                print(f'point ranges {point}: before removing dipping {point_ranges[point]}', file=log_f)
                if len(idx_to_keep) == 0:
                    point_ranges[point] = []
                    max_ben_points = point
                else:
                    point_ranges[point] = list(np.array(point_ranges[point])[np.array(idx_to_keep)])
                print(f'point ranges {point}: after removing dipping {point_ranges[point]}', file=log_f)

            if len(point_ranges[point]) > 1: # if didn't dip or still needs flattening
                print(f'flattening ({point}): {point_ranges[point]}', file=log_f)
                
                # flatten
                point_ranges[point] = [[point_ranges[point][0][0], point_ranges[point][-1][-1]]]
                if max_ben_points is None:
                    max_ben_points = point



def extend_points_to_xlims(point_ranges, point_values, score_range, scoreset_flipped, log_f=None, inf=False):
    # print('extending points to xlims...',file=log_f)
    left = -np.inf if inf else score_range[0]
    right = np.inf if inf else score_range[-1]
    for i in point_values:
        point = i # pathogenic
        if len(point_ranges[point]) != 0:
            j = 1
            all_no_evidence = True
            while point+j in point_ranges:
                if len(point_ranges[point+j]) != 0:
                    all_no_evidence = False
                j += 1
            
            if all_no_evidence:
                # extend to xlims
                point_ranges[point] = [[left, point_ranges[point][-1][-1]]] if not scoreset_flipped else [[point_ranges[point][0][0], right]]
                
        point = -i # benign
        if len(point_ranges[point]) != 0:
            j = 1
            all_no_evidence = True
            while point-j in point_ranges:
                if len(point_ranges[point-j]) != 0:
                    all_no_evidence = False
                j += 1
            
            if all_no_evidence:
                # extend to xlims
                point_ranges[point] = [[left, point_ranges[point][-1][-1]]] if scoreset_flipped else [[point_ranges[point][0][0], right]]

    # check if one evidence extends entire range, in case of incorrectly determined flipped scoreset (BAD!)
    for i in point_values:
        pos, neg = i,-i
        if len(point_ranges[pos]) != 0 and point_ranges[pos][0][0] == left and point_ranges[pos][-1][-1] == right:
            point_ranges[pos] = []
            print(pos,'extends the whole score range, removing...')
        if len(point_ranges[neg]) != 0 and point_ranges[neg][0][0] == left and point_ranges[neg][-1][-1] == right:
            point_ranges[neg] = []
            print(neg,'extends the whole score range, removing...')


def drop_wrong_side_pathogenic_fragments(point_ranges, point_values, benign_center, scoreset_flipped,
                                         support_scores=None):
    """Single-direction postprocessing: drop raw pathogenic fragments on the wrong side of the benign reference.

    Called on the RAW point ranges, before the first enforce_monotonicity_point_ranges pass of the plain
    (non-bidirectional) postprocessing. On that path, liberal monotonicity keeps, per tier, the fragment
    nearest the benign side, and extend_points_to_xlims then stretches it to the pathogenic-side limit. A
    spurious far-tail rise of LR+ on the benign side (e.g. a handful of variants past the benign peak)
    therefore either becomes evidence across the whole score range (BARD1_IGVF at assay scale; SFPQ_IGVF
    with prior 0.1) or cuts the genuine pathogenic range back to an island (PALB2_IGVF, CTCF_IGVF at
    assay scale). The whole-range guard in extend_points_to_xlims misses these when the range ends at
    the score-range maximum rather than at +/-inf.

    Rule: each raw pathogenic fragment is assigned to a side of ``benign_center`` by its midpoint. The
    expected side is low scores (high if ``scoreset_flipped``).
      - fragments on the expected side only (or none): no change;
      - fragments on both sides: keep the side whose fragments hold more ``support_scores`` (the
        P/LP sample, or the population sample in NU mode); ties go to the expected side. If the
        expected side is kept, the other side's fragments are dropped. If the opposite side has more
        support, nothing is changed and the dataset is flagged (``flag``) for review;
      - fragments on the opposite side only: nothing is changed; flagged for review.
    In the flagged cases the mean-score direction (scoreset_flipped) disagrees with where the
    pathogenic LR+ is, and the right response (review the direction override or flip) is not automated.

    Modifies ``point_ranges`` in place. Returns a JSON-serialisable record:
    {action, expected_side, sides, support_low, support_high, dropped: [[points, lo, hi], ...], flag}.
    """
    expected = "high" if scoreset_flipped else "low"
    info = dict(action="none", expected_side=expected, sides=[], support_low=0, support_high=0,
                dropped=[], flag=None)
    if benign_center is None or not np.isfinite(benign_center):
        info["action"] = "skipped_no_benign_center"
        return info
    frags = []
    for pv in point_values:
        for lo, hi in point_ranges.get(pv, []):
            mid = (lo + hi) / 2
            frags.append((pv, lo, hi, "low" if mid < benign_center else "high"))
    sides = sorted({f[3] for f in frags})
    info["sides"] = sides
    s = np.asarray(support_scores if support_scores is not None else [], dtype=float)
    s = s[np.isfinite(s)]
    for side in ("low", "high"):
        info[f"support_{side}"] = int(sum(((s >= lo) & (s <= hi)).sum() for _, lo, hi, sd in frags if sd == side))
    if not sides or sides == [expected]:
        return info
    other = "low" if expected == "high" else "high"
    if len(sides) == 1:
        info["action"] = "flag_opposite_side_only"
        info["flag"] = (f"pathogenic evidence only on the {other} side of the benign reference "
                        f"({benign_center:.4g}); scoreset_flipped={scoreset_flipped} expects the {expected} side")
        return info
    if info[f"support_{other}"] > info[f"support_{expected}"]:
        info["action"] = "flag_opposite_side_supported"
        info["flag"] = (f"pathogenic fragments on both sides of the benign reference ({benign_center:.4g}) and the "
                        f"{other} side holds more support ({info[f'support_{other}']} vs "
                        f"{info[f'support_{expected}']}); scoreset_flipped={scoreset_flipped} expects the {expected} side")
        return info
    for pv in point_values:
        if pv in point_ranges:
            point_ranges[pv] = [[lo, hi] for p, lo, hi, sd in frags if p == pv and sd == expected]
    info["dropped"] = [[int(p), float(lo), float(hi)] for p, lo, hi, sd in frags if sd == other]
    info["action"] = f"dropped_{other}_side"
    return info


# Fraction of bootstrap fits that must be flagged bidirectional (by either
# is_bidirectional_by_weights or is_bidirectional_by_raw_points) before a
# dataset's postprocess_point_ranges is auto-forced to False. Prototyping
# constant, not currently exposed as a CLI flag.
BIDIRECTIONAL_VOTE_THRESHOLD = 0.5


def _skewnorm_mean(component_params):
    a, loc, scale = component_params
    delta = a / np.sqrt(1.0 + a * a)
    return loc + scale * delta * np.sqrt(2.0 / np.pi)


def get_effective_benign_weights(weights, benign_method, benign_idx, synonymous_idx):
    """Resolve the per-component "effective benign" mixture-weight vector for
    a fit, honoring benign_method the same way the rest of the pipeline does
    (see e.g. the w_bs blend in the 2c-equation branch of process_component_fits,
    and the ben_mean_score blend used for scoreset_flipped detection): average
    of benign+synonymous when benign_method="avg" and both are present, else
    whichever of the two is configured/available. Returns None if neither
    benign nor synonymous is present (nothing to compare against).
    """
    if benign_method == "avg" and benign_idx is not None and synonymous_idx is not None:
        return (np.asarray(weights[benign_idx]) + np.asarray(weights[synonymous_idx])) / 2
    if benign_method == "synonymous" and synonymous_idx is not None:
        return np.asarray(weights[synonymous_idx])
    if benign_idx is not None:
        return np.asarray(weights[benign_idx])
    if synonymous_idx is not None:
        return np.asarray(weights[synonymous_idx])
    return None


def resolve_bidirectional_weight_vectors(weights, benign_method, pathogenic_idx, benign_idx,
                                          synonymous_idx, gnomad_idx):
    """Resolve the (source, reference) weight vectors for is_bidirectional_by_weights,
    handling the same PN/PU/NU sample-availability cases as get_fit_prior /
    resolve_pathomechanism_anchor elsewhere in this module:

      - PN (pathogenic + benign/synonymous both present): source=pathogenic,
        reference=effective benign (get_effective_benign_weights).
      - PU (pathogenic present, benign/synonymous absent): no benign-like
        anchor exists, so reference falls back to gnomAD/population.
      - NU (pathogenic absent, benign/synonymous present): no pathogenic-like
        sample exists, so source falls back to gnomAD/population instead
        (a component gnomAD over-weights relative to effective benign is the
        NU analogue of "pathogenic-like").

    Returns (source_weights, reference_weights), either of which may be None
    if there isn't enough information to form a comparison (e.g. only gnomAD
    is present, with no pathogenic AND no benign/synonymous).
    """
    reference_weights = get_effective_benign_weights(weights, benign_method, benign_idx, synonymous_idx)

    if pathogenic_idx is not None:
        source_weights = np.asarray(weights[pathogenic_idx])
        if reference_weights is None and gnomad_idx is not None:
            reference_weights = np.asarray(weights[gnomad_idx])  # PU mode
    elif reference_weights is not None and gnomad_idx is not None:
        source_weights = np.asarray(weights[gnomad_idx])  # NU mode
    else:
        source_weights = None

    return source_weights, reference_weights


def is_bidirectional_by_weights(component_params, source_weights, reference_weights):
    """Method B auto-detection: does this fit look bidirectional (e.g. LoF/GoF)?

    Sorts components by skew-normal mean along the score axis, labels each
    "pathogenic-like" if the source sample's mixture weight in that component
    exceeds the reference sample's weight (else "benign-like"), and flags the
    fit if any benign-like component has at least one pathogenic-like
    component before it and at least one after it in the sorted order.
    Generalizes to any K: e.g. P,B,B,P or P,B,P,B both flag; only a fully
    monotonic label sequence (e.g. B,B,P,P) does not.

    ``source_weights``/``reference_weights`` are per-component weight vectors
    -- normally (pathogenic, effective-benign), but see
    ``resolve_bidirectional_weight_vectors`` for the PU/NU fallbacks (gnomAD
    substituted for whichever of pathogenic/benign is unavailable).
    """
    order = sorted(range(len(component_params)), key=lambda k: _skewnorm_mean(component_params[k]))
    path_like = [source_weights[k] > reference_weights[k] for k in order]

    any_p_before = False
    for i, is_p in enumerate(path_like):
        if is_p:
            any_p_before = True
            continue
        if any_p_before and any(path_like[i + 1:]):
            return True
    return False


def is_bidirectional_by_raw_points(ranges_p, ranges_b):
    """UNUSED since 2026-11-09: two-component bidirectional detection is
    disabled in ``visualize.py``. A two-component LR+ is monotonic wherever
    the density constraint is enforced (Supplementary Theorem S1), so every
    pattern this flagged in a 2c fit was a tail density violation: the wider
    abnormal component's density outlasting the normal one past e^-7 on the
    far side, covering ~0.1% of scores (checked on SFPQ_IGVF, LDLR_Tabet_2025_uptake,
    CTCF_IGVF, TSC2_IGVF), not bidirectional signal. Kept for reference only.

    Method A ("raw points") bidirectional auto-detection. Was used in
    ``visualize.py``'s bidirectional-detect block for n_c < 3 fits, where the
    component-weights method (is_bidirectional_by_weights) can't apply (it
    needs a pathogenic-like component on each side of the reference, i.e.
    >=3 components total). Does this fit's RAW (pre-postprocess) set of
    pathogenic/benign point ranges show a pathogenic -> benign -> pathogenic
    pattern along the score axis (pathogenic evidence on both sides of a
    benign region)?

    ``ranges_p``/``ranges_b`` are the per-fit dicts of point_value -> flat
    [lo, hi, lo, hi, ...] arrays produced by ``get_point_ranges`` (positive
    point values in ``ranges_p``, negative in ``ranges_b``), as already
    computed per-bootstrap-fit in ``process_component_fits`` before
    aggregation/postprocessing.
    """
    segments = []  # (lo, sign)
    for flat in ranges_p.values():
        arr = np.asarray(flat).reshape(-1, 2)
        for lo, _hi in arr:
            segments.append((lo, 1))
    for flat in ranges_b.values():
        arr = np.asarray(flat).reshape(-1, 2)
        for lo, _hi in arr:
            segments.append((lo, -1))
    if not segments:
        return False

    segments.sort(key=lambda s: s[0])
    signs = [s[1] for s in segments]
    merged = [signs[0]]
    for s in signs[1:]:
        if s != merged[-1]:
            merged.append(s)

    any_p_before = False
    for i, s in enumerate(merged):
        if s == 1:
            any_p_before = True
            continue
        if any_p_before and any(x == 1 for x in merged[i + 1:]):
            return True
    return False


def benign_reference_center(component_params, reference_weights):
    """ARCHIVED, unused: superseded by simply reusing the actual mean score of
    the benign/synonymous-labeled sample (ben_mean_score, already computed in
    process_component_fits for scoreset_flipped detection) as the reference
    position for extend_bidirectional_pathogenic_islands -- simpler and more
    robust than this per-fit EM-component approach, since components have no
    stable identity across bootstrap fits (only their weights/positions are
    comparable fit-to-fit, requiring an extra median-across-fits aggregation
    this function's callers had to do). Kept here for reference only.

    For a single bootstrap fit, returns the score-axis position (skew-normal
    mean) of whichever component the effective-benign/reference sample
    weights most heavily -- the fit's own local "benign center of mass".
    """
    k = int(np.argmax(reference_weights))
    return _skewnorm_mean(component_params[k])


def extend_bidirectional_pathogenic_islands(point_ranges, point_values, score_range, benign_center, inf=True):
    """ARCHIVED, unused: superseded by clean_bidirectional_pathogenic_evidence,
    which reuses the exact canonical enforce_monotonicity_point_ranges/
    extend_points_to_xlims machinery per-side instead of this bespoke
    edge-extension algorithm -- that correctly reduces a sandwiched-peak
    pattern (e.g. raw +1,+2,+3,+2,+1) to a properly nested +3,+2,+1 chain
    with the strongest tier extending outward, which this function's
    "extend whatever fragment sits at the literal edge" approach did not do
    (it would extend the weakest edge fragment instead). Kept for reference
    only.

    Custom postprocessing for auto-detected bidirectional assays (see
    PipelineConfig.auto_bidirectional), used INSTEAD of
    enforce_monotonicity_point_ranges + extend_points_to_xlims -- those
    assume a single monotonic direction (scoreset_flipped), which does not
    apply here since evidence can be pathogenic-leaning on BOTH extremes
    with benign evidence in the middle.

    Rationale: bidirectional evidence is assumed pathogenic-leaning at the
    extremes with benign in the middle (never at an axis edge), so only
    pathogenic evidence ever needs extending -- benign ranges are left
    untouched. Interior structure (whatever tiers/ranges exist between the
    two extremes) is also left untouched: no monotonicity-nesting is
    enforced, since it may be genuinely non-monotonic.

    Algorithm: merge all positive (pathogenic) point-tier sub-ranges into
    contiguous score-axis "islands" (adjacent/overlapping sub-ranges from
    any tier become one island). For the leftmost island, extend its left
    edge to the axis limit only if ``benign_center`` lies entirely to its
    right (i.e. this island is unambiguously the far-left lobe). Mirror for
    the rightmost island (extend right edge only if benign_center lies
    entirely to its left). If there is only a single island overall, only
    the side away from benign_center is extended (and not at all if
    benign_center falls inside the island itself -- ambiguous). Any islands
    strictly between the two extremes are left untouched, as are all
    benign (negative) ranges.

    Mutates point_ranges in place; returns nothing (matches
    enforce_monotonicity_point_ranges/extend_points_to_xlims convention).
    """
    left = -np.inf if inf else score_range[0]
    right = np.inf if inf else score_range[-1]

    all_ranges = []  # (lo, hi, point_value, index_in_point_ranges[point_value])
    for pv in point_values:
        for i, (lo, hi) in enumerate(point_ranges.get(pv, [])):
            all_ranges.append((lo, hi, pv, i))
    if not all_ranges or benign_center is None:
        return

    all_ranges.sort(key=lambda r: r[0])

    islands = []  # [{"lo", "hi", "members": [(pv, i, lo, hi), ...]}]
    for lo, hi, pv, i in all_ranges:
        if islands and lo <= islands[-1]["hi"]:
            islands[-1]["hi"] = max(islands[-1]["hi"], hi)
            islands[-1]["members"].append((pv, i, lo, hi))
        else:
            islands.append({"lo": lo, "hi": hi, "members": [(pv, i, lo, hi)]})

    def _extend_left_edge(island):
        for pv, i, lo, hi in island["members"]:
            if lo == island["lo"]:
                point_ranges[pv][i][0] = left
                return

    def _extend_right_edge(island):
        for pv, i, lo, hi in island["members"]:
            if hi == island["hi"]:
                point_ranges[pv][i][1] = right
                return

    left_island, right_island = islands[0], islands[-1]

    if left_island is right_island:
        # Only one pathogenic island total -- extend only the side away
        # from the benign center; if benign_center falls inside the
        # island itself, that's ambiguous, so leave it untouched.
        if benign_center > left_island["hi"]:
            _extend_left_edge(left_island)
        elif benign_center < left_island["lo"]:
            _extend_right_edge(left_island)
        return

    if benign_center > left_island["hi"]:
        _extend_left_edge(left_island)
    if benign_center < right_island["lo"]:
        _extend_right_edge(right_island)


def clean_benign_fragments_no_extend(point_ranges, point_values, score_range=None, liberal=True):
    """Auto-bidirectional benign-side cleanup (see PipelineConfig.auto_bidirectional).

    Benign evidence is assumed to NEVER be bidirectional (a single, unimodal
    region). Never calls extend_points_to_xlims regardless of ``liberal``
    (no reason to assume unobserved territory beyond the benign region
    continues indefinitely).

    ``liberal`` mirrors PipelineConfig.liberal_monotonicity -- respect the
    dataset's configured strictness instead of always using one behavior:

    liberal=True (default): the raw per-tier fragments can still show noisy
    alternation (e.g. -1,-2,-1,-2,-1) when the percentile-aggregated LR+
    curve wobbles back and forth across adjacent thresholds instead of
    crossing each one exactly once. Process benign tiers from STRONGEST
    (most negative) to WEAKEST (closest to zero); each tier's final range is
    the envelope (min-to-max span) of its own raw fragments, MINUS whatever
    territory a stronger tier has already claimed. Example: raw
    -1: [[A,B],[C,D],[E,F]], -2: [[B,C],[D,E]] (i.e. -1,-2,-1,-2,-1 reading
    left to right) becomes -2: [[B,E]] (claimed first, its own envelope)
    then -1: [[A,B],[E,F]] (envelope [A,F] minus the already-claimed [B,E])
    -- i.e. a clean -1,-2,-1, with the noisy middle -1 fragment correctly
    absorbed into -2's territory. (Plain liberal-mode
    enforce_monotonicity_point_ranges is NOT used here even though
    liberal=True mirrors it conceptually -- its "keep only one fragment
    total" rule would discard one of the two genuine flanking regions
    rather than merely absorbing the noise between them.)

    liberal=False: reuses the EXACT canonical strict
    enforce_monotonicity_point_ranges algorithm, called TWICE with no
    extend_points_to_xlims in between (matching the non-bidirectional
    path's "remove evidence that goes back to indeterminate" intent, minus
    the extend-to-axis-limit step, which is never appropriate for benign in
    bidirectional mode) -- restricted to the benign/negative tiers only (the
    positive/pathogenic tiers are snapshotted and restored unchanged around
    both calls). Each pass treats a highest-evidence tier that doesn't touch
    either score_range edge as "evidence goes back to neutral" and wipes
    that tier and everything weaker/beyond it; running it twice catches
    tiers that only became "the sole remaining evidence" as a result of the
    first pass's own removals (e.g. a fragmented weak tier that gets
    flattened to a single span in pass 1, with nothing stronger left beside
    it, gets wiped in pass 2 too). For a genuinely bidirectional dataset,
    benign sits in the MIDDLE by construction (the pathogenic islands
    occupy the true edges), so this ends up discarding ALL benign evidence
    here in practice -- matching how a liberal_monotonicity=False dataset
    was treated before auto-bidirectional detection existed. Requires
    ``score_range``.

    Mutates point_ranges in place for negative point_values only.
    """
    if not liberal:
        raw_pathogenic = {pv: list(point_ranges[pv]) for pv in point_values}
        enforce_monotonicity_point_ranges(
            point_ranges, point_values, score_range, scoreset_flipped=False, liberal=False,
        )
        enforce_monotonicity_point_ranges(
            point_ranges, point_values, score_range, scoreset_flipped=False, liberal=False,
        )
        for pv in point_values:
            point_ranges[pv] = raw_pathogenic[pv]
        return

    def _envelope(ranges):
        if not ranges:
            return None
        return [min(r[0] for r in ranges), max(r[1] for r in ranges)]

    def _subtract(interval, claimed_list):
        pieces = [list(interval)]
        for c_lo, c_hi in claimed_list:
            next_pieces = []
            for lo, hi in pieces:
                if c_hi <= lo or c_lo >= hi:
                    next_pieces.append([lo, hi])
                    continue
                if c_lo > lo:
                    next_pieces.append([lo, c_lo])
                if c_hi < hi:
                    next_pieces.append([c_hi, hi])
            pieces = next_pieces
        return pieces

    claimed = []
    for pv in sorted(point_values, reverse=True):  # strongest (most negative) first
        point = -pv
        env = _envelope(point_ranges.get(point, []))
        if env is None:
            point_ranges[point] = []
            continue
        point_ranges[point] = _subtract(env, claimed)
        claimed.append(env)


BENIGN_ISLAND_MIN_FRAC = 0.01


def prune_small_benign_blocks(point_ranges, point_values, reference_scores,
                              min_frac=BENIGN_ISLAND_MIN_FRAC):
    """Auto-bidirectional benign-side pruning of tiny benign blocks (see
    PipelineConfig.benign_island_min_frac).

    Runs after clean_benign_fragments_no_extend, which keeps benign evidence
    wherever the percentile-aggregated LR+ curve crosses a benign threshold
    and never extends it to the score-range edges. On the standard
    (non-bidirectional) path, extend_points_to_xlims carries benign evidence
    to an edge, so a stray crossing can't survive on its own; here it can. A
    curve that barely dips below a threshold over a few grid points (e.g.
    CARD11 gof_clinvar_v2: 95th-percentile log LR+ 0.0017 below the -1
    threshold over score 0.063-0.070) then leaves an isolated benign sliver.

    A block is a maximal run of contiguous benign sub-ranges, possibly
    spanning several benign levels (e.g. -1/-2/-1); an indeterminate gap
    splits blocks. Each block is judged WHOLE by its share of
    ``reference_scores`` (the population sample's scores) within [lo, hi]:
    a block below ``min_frac`` is removed from every benign level (left
    indeterminate, never reassigned). Judging whole blocks rather than single
    levels keeps the thin graded shoulders of a genuine benign core (each of
    which can hold well under 1% of the population on its own) intact.

    ``min_frac`` <= 0 disables pruning. Returns the pruned blocks as a list of
    [lo, hi, share]. Mutates point_ranges in place for negative point_values
    only.
    """
    if min_frac is None or min_frac <= 0:
        return []
    reference_scores = np.asarray(reference_scores, dtype=float)
    reference_scores = reference_scores[~np.isnan(reference_scores)]
    if reference_scores.size == 0:
        return []

    segments = sorted(
        (lo, hi, -pv) for pv in point_values for lo, hi in point_ranges.get(-pv, [])
    )
    if not segments:
        return []

    blocks = []  # [lo, hi]
    for lo, hi, _ in segments:
        if blocks and lo <= blocks[-1][1]:
            blocks[-1][1] = max(blocks[-1][1], hi)
        else:
            blocks.append([lo, hi])

    pruned = []
    for lo, hi in blocks:
        share = float(np.mean((reference_scores >= lo) & (reference_scores <= hi)))
        if share >= min_frac:
            continue
        pruned.append([lo, hi, share])
        for pv in point_values:
            point_ranges[-pv] = [
                [a, b] for a, b in point_ranges.get(-pv, []) if not (a >= lo and b <= hi)
            ]
    return pruned


def clean_bidirectional_pathogenic_evidence(point_ranges, point_values, score_range, benign_center,
                                             liberal=True, inf=True):
    """Auto-bidirectional pathogenic-side cleanup (see PipelineConfig.auto_bidirectional).

    Splits each pathogenic tier's raw fragments by position relative to
    ``benign_center`` into a "left" and a "right" subset, then runs the
    EXACT canonical enforce_monotonicity_point_ranges + extend_points_to_xlims
    machinery independently on each subset -- as if it were an ordinary
    single-direction (non-bidirectional) pathogenic problem. Left fragments
    are processed as scoreset_flipped=False (pathogenic extends toward
    -inf); right fragments as scoreset_flipped=True (pathogenic extends
    toward +inf).

    This correctly handles both:
      - a genuine two-sided (pathogenic-benign-pathogenic) pattern: each
        side gets its own canonical nested/extended structure independently
        (no cross-contamination between the two lobes), and
      - a single-sided "peak" pattern where a stronger tier is sandwiched
        inside weaker flanking fragments (e.g. raw +1,+2,+3,+2,+1 all left
        of benign_center): canonical liberal monotonicity naturally
        collapses this into a properly nested +3,+2,+1 chain (discarding
        the redundant weaker-tier fragment on the "wrong" side of the peak,
        since it's already covered once the strongest tier's edge is
        extended outward) -- matching exactly what the ordinary
        non-bidirectional path does for a normal monotonic assay.

    Mutates point_ranges in place for positive point_values only.
    """
    if benign_center is None:
        return

    left = {pv: [] for pv in point_values}
    right = {pv: [] for pv in point_values}
    for pv in point_values:
        left[-pv] = []
        right[-pv] = []
        for lo, hi in point_ranges.get(pv, []):
            mid = (lo + hi) / 2.0
            (left if mid < benign_center else right)[pv].append([lo, hi])

    enforce_monotonicity_point_ranges(left, point_values, score_range, scoreset_flipped=False, liberal=liberal)
    extend_points_to_xlims(left, point_values, score_range, False, inf=inf)
    enforce_monotonicity_point_ranges(left, point_values, score_range, scoreset_flipped=False, liberal=liberal)

    enforce_monotonicity_point_ranges(right, point_values, score_range, scoreset_flipped=True, liberal=liberal)
    extend_points_to_xlims(right, point_values, score_range, True, inf=inf)
    enforce_monotonicity_point_ranges(right, point_values, score_range, scoreset_flipped=True, liberal=liberal)

    for pv in point_values:
        point_ranges[pv] = left[pv] + right[pv]


def prior_equation_2c(w_p, w_b, w_g):
    return (w_g[1] - w_b[1]) / (w_p[1] - w_b[1])

def prior_invalid(prior):
    return prior <= 0 or prior >= 1


def _get_benign_reference_weights(weights, benign_method, benign_idx, synonymous_idx):
    """Effective benign/synonymous mixture weight vector per `benign_method`.

    Shared by compute_lr_filtered_pathogenic_mask and get_fit_prior so the
    benign_method blend logic (benign-only / synonymous-only / avg-of-both)
    lives in exactly one place.
    """
    if benign_method == 'synonymous' and synonymous_idx is not None:
        return weights[synonymous_idx]
    if benign_method == 'avg' and benign_idx is not None and synonymous_idx is not None:
        return (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
    return weights[synonymous_idx if benign_idx is None else benign_idx]


def estimate_prior_from_class_densities(
    f1, f2, mode,
    max_em_steps=10000, tolerance=1e-6,
    no_signal_prior=0.1, floor_low=0.01, discard_high=0.99, initial_prior=0.5,
):
    """Numeric core of prior estimation, shared by get_fit_prior (below) and
    the multivariate analysis harness (src/assay_calibration/multivariate_analysis/mv_calibration.py), so both
    pipelines run the exact same math instead of maintaining independent
    copies.

    `f1`, `f2` are 1-D linear-space density arrays evaluated at the same
    population/gnomAD points (already filtered to whatever entries the
    caller considers valid). Their meaning depends on `mode`:

        mode == 'standard' (both a pathogenic- and a benign/synonymous-
        labeled sample exist): f1 = pathogenic density, f2 = benign density.
        Runs the iterative joint-EM loop and returns the RAW prior estimate,
        UNBOUNDED -- the caller is responsible for its own discard/fallback
        policy on the returned value (this intentionally differs across
        existing call sites -- see each call site for why).

        mode in ('positive_unlabeled', 'negative_unlabeled'): f1 = the
        labeled class's density (pathogenic for PU, benign for NU), f2 =
        the population's own raw mixture density. Runs the boundary/
        unmixing estimator (Blanchard, Lee & Scott 2010; Scott 2015) and
        DOES apply floor/discard here, since that policy is already
        identical across every current call site.

    Returns a float, or np.nan for the PU/NU branch when the boundary
    estimate is too close to 1 to trust (see `discard_high`), OR when
    `boundary > 1` -- i.e. the labeled-vs-mixture ratio never dips below 1
    anywhere in the reference sample. For mode='positive_unlabeled',
    prior_raw = boundary directly, so boundary > 1 already triggers the
    `prior_raw >= discard_high` check above (discard_high < 1). For
    mode='negative_unlabeled', prior_raw = 1 - boundary, so the SAME
    boundary > 1 condition instead produces a NEGATIVE prior_raw -- without
    an explicit check, `max(prior_raw, floor_low)` would silently floor this
    up to floor_low, which is backwards: boundary > 1 means no point in the
    reference sample ever looked "purely" like the other class, i.e. there
    is no evidence constraining the estimate to be small at all (the
    opposite of what flooring to a low value implies). Discarding here
    mirrors how the positive_unlabeled branch already handles this same
    condition, rather than introducing a second, differently-signed
    treatment for the negative_unlabeled branch.
    """
    if mode == 'standard':
        # EM starting value; the estimate stays here when the likelihood is flat
        # (pathogenic and benign densities equal at every population point).
        # Only the prior simulation (analysis/prior_simulation) changes it.
        prior_estimate = initial_prior
        converged = False
        em_steps = 0
        while not converged and em_steps < max_em_steps:
            em_steps += 1
            with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                posteriors = 1 / (
                    1 + (1 - prior_estimate) / prior_estimate * f2 / f1
                )
            new_prior = np.nanmean(posteriors)
            if abs(new_prior - prior_estimate) < tolerance:
                converged = True
            prior_estimate = new_prior
            if prior_estimate < 0 or prior_estimate > 1:
                break
        return prior_estimate

    # PU/NU boundary/unmixing estimator.
    valid = f1 > 0
    if not np.any(valid):
        return no_signal_prior
    boundary = float(np.min(f2[valid] / f1[valid]))
    prior_raw = boundary if mode == 'positive_unlabeled' else (1.0 - boundary)
    if prior_raw >= discard_high or prior_raw < 0:
        return np.nan
    return float(max(prior_raw, floor_low))


def compute_lr_filtered_pathogenic_mask(
    fits, scores, sample_assignments, benign_method,
    pathogenic_idx, benign_idx, synonymous_idx, gnomad_idx=None,
    percentile=5.0, n_jobs=-1,
):
    """
    For every pathogenic-labeled row, evaluate each bootstrap fit's log-likelihood-
    ratio-like statistic directly at that row's own score (using the frozen fitted
    mixture curves, same primitive as _compute_log_fp_only). Take the
    `percentile`-th percentile across bootstraps per row, and keep rows where
    that conservative statistic is > 0.

    When a benign or synonymous sample exists, this is the real log_lr_plus =
    log_fp - log_fb (LR+ > 1 at the conservative bound).

    When neither exists (PU-only datasets), there is no fb curve to compute a
    true LR+ against. Using the prior to unmix an fb curve here would be
    circular (the prior is exactly what filtering is meant to improve) and
    compounds whatever separation problem caused the unfiltered PU prior
    estimate to be unreliable in the first place. Instead this falls back to
    log_fp - log_f_population (requires gnomad_idx): a single-pass, prior-free
    statistic that keeps only pathogenic-labeled rows whose fitted pathogenic
    density exceeds the raw (mixed) population density at their own score --
    directly excluding the poorly-separated rows that dilute the PU mean-
    matching denominator (E_labeled[fp]) and inflate the estimated prior. This
    is NOT literal LR+ (no fb involved) -- it is a PU-specific separation
    criterion with the same >0 threshold semantics.

    Returns (mask, n_labeled, n_kept):
      mask       : bool array, shape (len(scores),), True for kept pathogenic rows
                   (global indexing, aligned to `scores`/`sample_assignments`)
      n_labeled  : total pathogenic-labeled rows before filtering
      n_kept     : rows kept after filtering
    """
    lab_mask = sample_assignments[:, pathogenic_idx].astype(bool)
    labeled_scores = scores[lab_mask]
    n_labeled = len(labeled_scores)
    if n_labeled == 0:
        return np.zeros(len(scores), dtype=bool), 0, 0

    is_pu = benign_idx is None and synonymous_idx is None

    def _per_fit(fit):
        params = fit['fit']['component_params']
        weights = fit['fit']['weights']
        log_fp = density_utils.mixture_pdf(labeled_scores, params, weights[pathogenic_idx])
        if is_pu:
            log_ref = density_utils.mixture_pdf(labeled_scores, params, weights[gnomad_idx])
        else:
            w_b = _get_benign_reference_weights(weights, benign_method, benign_idx, synonymous_idx)
            log_ref = density_utils.mixture_pdf(labeled_scores, params, w_b)
        stat = log_fp - log_ref
        # Bootstrap fits don't all span the same score range; evaluating a fit's
        # mixture curves at a score outside its fitted xlims is extrapolation, not
        # interpolation, and can swing wildly (see the same guard in
        # get_bootstrap_score_ranges). Mask those out so nanpercentile below isn't
        # contaminated by a handful of unstable extrapolated fits.
        xmin, xmax = fit['fit']['xlims']
        in_range = (labeled_scores >= xmin) & (labeled_scores <= xmax)
        stat = np.where(in_range, stat, np.nan)
        return stat

    log_lr = np.array(Parallel(n_jobs=min(len(fits), max(n_jobs, 1)), verbose=0)(
        delayed(_per_fit)(fit) for fit in fits
    ))  # shape (n_fits, n_labeled)

    lr_percentile = np.nanpercentile(log_lr, percentile, axis=0)  # conservative per-row
    keep = lr_percentile > 0

    mask = np.zeros(len(scores), dtype=bool)
    mask[np.where(lab_mask)[0][keep]] = True
    return mask, n_labeled, int(keep.sum())


def _w_D_subtraction(w_P_raw, w_N):
    """Excess-weight w_D construction: max(0, w_P_raw - w_N), renormalized.

    Returns None (degenerate) when w_P_raw <= w_N on every component. See
    compute_pathomechanism_pathogenic_density's docstring for the full
    rationale -- N gets priority on ambiguous/overlapping mass, D only gets
    credit for clearly exceeding it.
    """
    excess = np.maximum(np.asarray(w_P_raw, dtype=float) - np.asarray(w_N, dtype=float), 0.0)
    total = float(excess.sum())
    if total <= 0.0:
        return None
    return excess / total


def _w_D_masking(w_P_raw, w_N):
    """Hard-mask w_D construction: keep w_P_raw[k] wherever w_P_raw[k] >
    w_N[k], zero elsewhere, renormalized. Simpler alternative to
    _w_D_subtraction -- a component just barely exceeding w_N keeps its FULL
    raw weight (not just the excess), which is more generous toward
    borderline-overlapping components than subtraction, at the cost of a
    harder/less smooth transition across the w_P_raw==w_N boundary (subtraction
    degrades gracefully toward zero near a tie; masking does not).

    Returns None (degenerate) when no component has w_P_raw > w_N.
    """
    w_P_raw = np.asarray(w_P_raw, dtype=float)
    w_N = np.asarray(w_N, dtype=float)
    mask = w_P_raw > w_N
    if not np.any(mask):
        return None
    w_D = np.where(mask, w_P_raw, 0.0)
    return w_D / w_D.sum()


def _mechanism_match_em(labeled_scores, population, params, w_D, w_N,
                         mechanism_match_frac_init, max_em_steps, tolerance, multivariate):
    """Shared EM core for compute_pathomechanism_pathogenic_density and its
    masked sibling: given FIXED f_D (via w_D) and f_N (via w_N) densities,
    estimate P(M=1|Y=1) (the pathogenic-labeled sample's mechanistic-coverage
    fraction -- the fraction of it whose disease mechanism this assay
    measures) via a two-known-densities mixture-proportion EM against
    labeled_scores, then evaluate the resulting f_D at `population`.

    Returns (pathogenic_density_at_population_points, mechanism_match_frac_hat).
    """
    log_fD = density_utils.mixture_pdf(labeled_scores, params, w_D, multivariate=multivariate)
    log_fN = density_utils.mixture_pdf(labeled_scores, params, w_N, multivariate=multivariate)
    match_frac = float(mechanism_match_frac_init)
    for _ in range(max_em_steps):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            posteriors = 1.0 / (1.0 + (1.0 - match_frac) / match_frac * np.exp(log_fN - log_fD))
        new_match_frac = float(np.nanmean(posteriors))
        if abs(new_match_frac - match_frac) < tolerance:
            match_frac = new_match_frac
            break
        match_frac = new_match_frac
        if not np.isfinite(match_frac) or match_frac <= 0.0 or match_frac >= 1.0:
            break

    pathogenic_density = density_utils.joint_densities(
        population, params, w_D, multivariate=multivariate
    ).sum(axis=0)
    return pathogenic_density, match_frac


def compute_pathomechanism_pathogenic_density(
    labeled_scores, population, params, w_P_raw, w_N,
    mechanism_match_frac_init=0.5, max_em_steps=10000, tolerance=1e-8,
    multivariate=False,
):
    """
    EXPERIMENTAL. Estimates the pathogenic-labeled sample's "assay
    mechanistic coverage" fraction P(M=1|Y=1) in

        f_pathogenic_labeled(x) = P(M=1|Y=1) * f_D(x) + (1 - P(M=1|Y=1)) * f_N(x)

    and returns the resulting "assay-relevant" pathogenic density f_D
    evaluated at `population` (the gnomAD/population points used elsewhere
    in get_fit_prior), for use in place of the raw pathogenic density.

    `w_N` is the fixed anchor density's weight vector -- the caller decides
    what it represents. For PN/standard-mode datasets it's the benign/
    synonymous reference (get_fit_prior's `w_b`). For PU-only datasets
    (no benign/synonymous sample) it's the raw, UNPURIFIED gnomAD/
    population weights (`weights[gnomad_idx]`) -- the same anchor
    compute_lr_filtered_pathogenic_mask's PU fallback already uses
    (log_fp - log_f_population), just formalized as a proper generative
    mixture instead of a hard per-row threshold. No attempt is made to
    unmix gnomAD's own (typically small) real-disease-variant content
    first: writing f_gnomAD = P(Y=1)*f_D + (1-P(Y=1))*f_N, the excess
    construction below is provably invariant to P(Y=1) -- (w_P_raw -
    w_gnomAD) = (P(M=1|Y=1)-P(Y=1)) * (w_D-w_N), and P(M=1|Y=1) > P(Y=1)
    always holds (a curated pathogenic-labeled sample is by construction
    more disease-enriched than the general population), so renormalizing
    after max(0, ...) cancels the (P(M=1|Y=1)-P(Y=1)) scale factor
    regardless of its value. The one place P(Y=1)'s absence from a pure
    reference does matter is the mechanism_match_frac_hat estimate itself,
    which becomes a mildly conservative (downward-biased) estimate of true
    mechanistic coverage: mechanism_match_frac_hat = 1 -
    (1-mechanism_match_frac_true)/(1-P(Y=1)) <= mechanism_match_frac_true,
    with zero bias when P(Y=1) ~ 0 (empirically confirmed on
    TARDBP_Bolognesi_Faure_2019, where gnomAD carries ~no disease-relevant
    mass: the estimate's median matched the LR-filter's independently-
    computed kept-row fraction almost exactly).

    f_D is built directly from the per-component EXCESS of the raw
    pathogenic-labeled weights over the anchor weights:

        w_D[k] = max(0, w_P_raw[k] - w_N[k]),  renormalized to sum to 1

    This is a fixed, non-tautological density (it does not depend on
    P(M=1|Y=1)), constructed independently of any such value -- unlike an
    earlier version of this estimator that solved for it via
    w_D(m) = (w_P_raw - (1-m)*w_N)/m, which reconstructs
    f_pathogenic_labeled EXACTLY for every m and therefore carries zero
    likelihood information about m (a proven tautology -- only the
    non-negativity boundary constraint identified anything in that
    framework). Building w_D directly from the excess breaks that
    tautology, at the cost of a known, deliberate asymmetry: on any shared
    component where both D and N have real weight (i.e. the two densities
    genuinely overlap), this construction credits ALL of that component's
    weight to N first and only assigns D whatever excess remains -- N gets
    priority on ambiguous/overlapping mass, D only gets credit for
    clearly-exceeding it. This is an appropriately conservative choice for
    an evidence-assignment pipeline (a variant should only look
    "assay-relevant damaging" once it clearly exceeds what the benign
    reference alone explains) and is treated as fixed/accepted, not a bug
    to fix. It also naturally handles more than one "benign-like" component
    without any hardcoded component classification.

    Because f_D (via w_D) and f_N (via w_N) are now both FIXED densities,
    estimating P(M=1|Y=1) from the pathogenic-labeled sample's own scores is
    a classic two-known-densities mixture-proportion EM -- the same update
    rule as the standard-mode joint prior EM below, just applied here
    against the pathogenic-labeled sample instead of the population sample.
    This is provably well-behaved (a concave problem in the single scalar
    P(M=1|Y=1)) and was confirmed empirically init-robust to ~1e-10 across
    mechanism_match_frac_init in [0.01, 0.99] on real data -- unlike the
    free-weight-vector EM this replaced, which was strongly initialization-
    dependent (median estimate shifted from ~0.91 to ~0.81 across init=0.9
    vs 0.1).

    Returns (pathogenic_density_at_population_points, mechanism_match_frac_hat).
    Returns (None, np.nan) when w_P_raw <= w_N on every component (no
    assay-relevant excess signal at all -- degenerate). get_fit_prior
    discards this individual bootstrap fit from prior estimation rather
    than falling back to the raw pathogenic density for it: that raw
    density is itself the degenerate case this correction exists to move
    away from (e.g. a fit that collapses w_P_raw onto the same component as
    the anchor, so the two densities are identical and any downstream prior
    EM is a trivial fixed point). process_component_fits falls back to
    raw-density prior estimation only if every bootstrap fit for the combo
    is degenerate this way.
    """
    w_D = _w_D_subtraction(w_P_raw, w_N)
    if w_D is None:
        return None, np.nan
    return _mechanism_match_em(labeled_scores, population, params, w_D, w_N,
                                mechanism_match_frac_init, max_em_steps, tolerance, multivariate)


def compute_pathomechanism_pathogenic_density_masked(
    labeled_scores, population, params, w_P_raw, w_N,
    mechanism_match_frac_init=0.5, max_em_steps=10000, tolerance=1e-8,
    multivariate=False,
):
    """
    EXPERIMENTAL. Alternative to compute_pathomechanism_pathogenic_density
    using a hard per-component mask instead of excess-subtraction to build
    f_D: w_D[k] = w_P_raw[k] if w_P_raw[k] > w_N[k] else 0, renormalized.

    Same contract as compute_pathomechanism_pathogenic_density in every other
    respect (same EM core via _mechanism_match_em, same (None, np.nan)
    degenerate return when no component has w_P_raw > w_N). The two differ
    only in how a component that just barely exceeds w_N is treated:
    subtraction keeps only the thin excess (degrading gracefully toward
    zero near a tie), masking keeps the component's FULL raw weight the
    instant it exceeds w_N (a harder, less smooth transition, more generous
    toward borderline components, but a bigger jump in response to
    bootstrap noise right at the w_P_raw==w_N boundary). Kept here purely as
    an experimental point of comparison against subtraction, not as a
    replacement for it.

    Returns (pathogenic_density_at_population_points, mechanism_match_frac_hat).
    """
    w_D = _w_D_masking(w_P_raw, w_N)
    if w_D is None:
        return None, np.nan
    return _mechanism_match_em(labeled_scores, population, params, w_D, w_N,
                                mechanism_match_frac_init, max_em_steps, tolerance, multivariate)


# Below this, match_frac is "effectively zero": the boundary estimate found
# a labeled point whose f_N/f_Praw ratio sits right at (or just under) 1, so
# P(M=1|Y=1) is a genuine but vanishingly small estimate -- not the
# no-candidate-found default of 1.0, and not NaN/invalid either. The
# unmixing formula f_D = (f_Praw - (1-match_frac)*f_N) / match_frac divides
# by match_frac, so at this scale it amplifies any numerator noise (fit
# imprecision, floating-point error) by a factor of >=100, producing a
# wildly non-monotonic, meaningless f_D/log-LR curve rather than a merely
# noisy one. Below this floor, f_D is treated as not existing: both
# functions fall back to reporting "no mechanism-specific evidence"
# (pathogenic-direction LR == 1 everywhere) instead of constructing it.
_PATHOMECHANISM_MATCH_FRAC_FLOOR = 0.01


def _pathomechanism_boundary_match_frac(f_N_at_labeled, f_Praw_at_labeled):
    """Estimate P(M=1|Y=1) via the Blanchard-Lee-Scott boundary identity,
    restricted to labeled points where f_N(x) >= f_Praw(x) (ratio <= 1) --
    shared by compute_pathomechanism_pathogenic_density_boundary and
    compute_pathomechanism_lr_curves so the two never drift apart on this
    logic, the same way _w_D_subtraction/_w_D_masking are shared for the
    other two methods.

    Points where f_Praw(x) > f_N(x) provide no evidence that ANY of the
    labeled sample needs explaining by the anchor at that location -- unlike
    an unfiltered min (which such points could still corrupt if the true
    minimum-ratio point happens to sit among them, producing a ratio > 1 and
    silently inverting 1-boundary into a negative number), they're excluded
    from the candidate set entirely. If NO labeled point has f_N(x) >=
    f_Praw(x) anywhere, there is no evidence constraining P(M=1|Y=1) below 1
    at all. Unlike genuine PU/NU population-prior estimation (where a LOW
    default is the sensible prior for an untested gene), the sensible
    default here is HIGH: assays usually do measure the dominant disease
    mechanism, so absent a labeled point demonstrating otherwise, assume the
    whole PLP-labeled sample is on-mechanism.

    Returns match_frac in (0, 1] -- exactly 1 is a legitimate value here
    (unlike P(Y=1)/P(Y=1,M=1) elsewhere, which treat exactly 1 as a
    degenerate/untrustworthy saturation artifact): it means "no evidence
    constrains this below full mechanism coverage", and the downstream
    unmixing formula f_D = (f_Praw - (1-match_frac)*f_N) / match_frac is
    perfectly well-defined at match_frac=1 (reduces to f_D = f_Praw exactly,
    no division-by-zero -- the divisor is match_frac itself, not 1-match_frac).
    Callers must use `<= 1` (not the pipeline's usual strict `< 1`) when
    validating this specific return value. Returns np.nan if f_N_at_labeled
    is zero everywhere (no informative points at all)."""
    valid = f_N_at_labeled > 0
    if not np.any(valid):
        return np.nan
    ratio = f_Praw_at_labeled[valid] / f_N_at_labeled[valid]
    candidates = ratio <= 1.0
    if not np.any(candidates):
        return 1.0
    boundary = float(np.min(ratio[candidates]))
    return 1.0 - boundary


def compute_pathomechanism_pathogenic_density_boundary(
    labeled_scores, population, params, w_P_raw, w_N, multivariate=False,
):
    """
    Density-level alternative to compute_pathomechanism_pathogenic_density /
    _masked: instead of building f_D from the fitted mixture model's
    per-component WEIGHTS (which requires the shared components to
    individually align with f_D's true support -- not guaranteed with only
    2-3 components), this estimates P(M=1|Y=1) and f_D directly from
    DENSITIES, via the same Blanchard-Lee-Scott boundary/min-ratio identity
    underlying this codebase's PU/NU prior estimator -- but with its own
    dedicated candidate-filtering and default-direction logic (see
    _pathomechanism_boundary_match_frac), since this application's sensible
    default (high, absent counter-evidence) is the opposite of PU/NU
    population-prior estimation's (low, absent evidence) -- just pointed at
    the pathogenic-labeled sample in place of the population sample:

        P(M=1|Y=1) = 1 - min_{s in labeled_scores} [ f_P_raw(s) / f_N(s) ]

    evaluated ONLY at the pathogenic-labeled sample's own observed score
    points (never at a synthetic grid or in unlabeled tails), then f_D is
    recovered by the same _unmix_density rearrangement already used for
    PU/NU density unmixing:

        f_D(x) = ( f_P_raw(x) - (1-P(M=1|Y=1)) * f_N(x) ) / P(M=1|Y=1)

    This needs no per-fit EM (unlike _mechanism_match_em's iterative core)
    since the boundary estimate is closed-form. It also requires no
    assumption that f_D lies in the span of the shared mixture components --
    only the density-level mixture identity f_P_raw = P(M=1|Y=1)*f_D +
    (1-P(M=1|Y=1))*f_N, the same assumption already underlying every PU/NU
    call site in this module.

    Validated empirically: on 4 PN datasets (ASPA toxicity/abundance, both
    GCK assays), the resulting pathogenic-direction prior matched the
    subtraction method's within ~0.001-0.0015; on ASPA toxicity, the
    P(M=1|Y=1) estimate (~0.75) independently matched DistCurve
    (Zeiberg/Jain/Radivojac 2020) run on the same raw score arrays.

    Returns (pathogenic_density_at_population_points, mechanism_match_frac_hat).
    Returns (None, np.nan) when the boundary estimate is invalid (mirrors
    estimate_prior_from_class_densities' own floor/discard contract).
    """
    f_N_at_labeled = density_utils.joint_densities(
        labeled_scores, params, w_N, multivariate=multivariate
    ).sum(axis=0)
    f_Praw_at_labeled = density_utils.joint_densities(
        labeled_scores, params, w_P_raw, multivariate=multivariate
    ).sum(axis=0)
    match_frac = _pathomechanism_boundary_match_frac(f_N_at_labeled, f_Praw_at_labeled)
    # match_frac == 1 is a legitimate value here (see
    # _pathomechanism_boundary_match_frac's docstring) -- unlike every other
    # prior-type quantity in this module, so this uses <= 1, not < 1.
    if not (np.isfinite(match_frac) and 0 < match_frac <= 1):
        return None, np.nan

    f_N_at_pop = density_utils.joint_densities(
        population, params, w_N, multivariate=multivariate
    ).sum(axis=0)
    if match_frac < _PATHOMECHANISM_MATCH_FRAC_FLOOR:
        # f_D is not estimable at this scale (see
        # _PATHOMECHANISM_MATCH_FRAC_FLOOR) -- fall back to the anchor
        # density itself, so downstream pathogenic-direction evidence
        # against this same anchor comes out as LR=1 (no evidence) rather
        # than corrupted.
        return f_N_at_pop, match_frac
    f_Praw_at_pop = density_utils.joint_densities(
        population, params, w_P_raw, multivariate=multivariate
    ).sum(axis=0)
    pathogenic_density = _unmix_density(f_Praw_at_pop, 1 - match_frac, f_N_at_pop)
    return pathogenic_density, match_frac


def resolve_pathomechanism_anchor(weights, benign_method, benign_idx, synonymous_idx, gnomad_idx):
    """Resolve the fixed "N" anchor weight vector used by both the
    pathomechanism prior-estimation methods and compute_pathomechanism_lr_curves,
    so the two never drift out of sync. PN/standard mode anchors against the
    benign/synonymous blend (_get_benign_reference_weights); PU-only mode (no
    benign/synonymous sample) anchors against the raw gnomAD/population
    weights instead (see compute_pathomechanism_pathogenic_density's docstring
    for why this is valid: the excess/mask construction is invariant to
    gnomAD's own small disease-relevant content).

    Returns (w_N_anchor, is_pu).
    """
    is_pu = benign_idx is None and synonymous_idx is None
    if is_pu:
        return weights[gnomad_idx], True
    return _get_benign_reference_weights(weights, benign_method, benign_idx, synonymous_idx), False


def _unmix_density(pop_linear, prior, positive_linear):
    """f_negative = (f_pop - prior*f_positive) / (1-prior), floor-clipped at
    pop_linear*1e-10 -- same formula/floor compute_single_fit_log_densities
    has always used for PU/NU unmixing, generalized to take an arbitrary
    (prior, positive-class density) pair rather than being hardcoded to
    (P(Y=1), f_P_raw), so it can serve either the benign-direction
    (P(Y=1), f_P_raw) or the pathogenic-direction (P(Y=1,M=1), f_D) unmixing
    below."""
    with np.errstate(invalid='ignore'):
        f_negative_linear = (pop_linear - prior * positive_linear) / (1 - prior)
    return np.maximum(f_negative_linear, pop_linear * 1e-10)


def compute_pathomechanism_lr_curves(fit, score_range, w_P_raw, w_N_anchor,
                                      pathomechanism_method, is_pu=False,
                                      prior_pathogenic=None, prior_benign=None,
                                      multivariate=False, labeled_scores=None):
    """Log-density curves for the two-LR/two-prior PS3/BS3 scheme, evaluated
    at score_range.

    PN mode (is_pu=False): w_N_anchor is a genuine, already-known labeled
    density (the benign/synonymous blend) -- population plays no role, so
    it's used directly, unmodified, as BOTH curves' denominator. No unmixing.

    PU mode (is_pu=True): w_N_anchor IS the raw gnomAD/population weights
    (see resolve_pathomechanism_anchor) -- population is itself a mixture
    containing an unknown pathogenic-like fraction, so using it directly as a
    shared denominator understates BOTH LRs (confirmed on
    PTEN_Mighell_2018_clinvar_2018: log LR+ inflated to +24 in the raw/legacy
    single-prior path's own unmixed curve vs. ~2 here without unmixing).
    Each direction must instead be purified against population using ITS OWN
    native (prior, density) pair -- the two are NOT interchangeable:
      - benign-direction:     f_B  = unmix(pop, prior_benign, f_P_raw)   [P(Y=1)]
        (unchanged from the historical PU/NU unmixing formula -- P(Y=1) and
        f_P_raw are estimated identically regardless of pathomechanism_method)
      - pathogenic-direction: f_B' = unmix(pop, prior_pathogenic, f_D)   [P(Y=1,M=1)]
        (new: derived from f_pop = P(Y=1,M=1)*f_D + (1-P(Y=1,M=1))*f_B',
        where f_B' is provably a valid convex combination of the raw anchor
        and f_B, representing "off-mechanism-pathogenic-looking OR truly
        benign" -- the correct negative reference class for PS3 evidence)
    prior_pathogenic/prior_benign are required (and assumed already validated
    -- non-NaN, in (0,1) -- by the caller's upstream valid_mask filtering)
    when is_pu=True.

    labeled_scores (the pathogenic-labeled sample's own real score points) is
    required only when pathomechanism_method == 'boundary' -- unlike
    subtraction/masking's w_D (a fixed weight-vector construction independent
    of P(M=1|Y=1)), the boundary method's f_D construction needs P(M=1|Y=1)
    itself, which can only be estimated from the labeled sample's own scores
    (see compute_pathomechanism_pathogenic_density_boundary).

    Returns (log_fD_or_None, log_fN_benign, log_fP_raw, log_fN_pathogenic_or_None):
      log_fN_benign, log_fP_raw are always arrays.
      log_fD is None when f_D is degenerate on this fit (mirrors
      compute_pathomechanism_pathogenic_density's (None, nan) contract) --
      this fit contributes no pathogenic-direction curve; its P(Y=1,M=1) was
      already nan from get_fit_prior and is already excluded from the
      P(Y=1,M=1) aggregation/valid_mask upstream. log_fN_pathogenic is None
      exactly when log_fD is None.
      In PN mode, log_fN_benign and log_fN_pathogenic are identical (both the
      direct w_N_anchor mixture) -- unchanged from prior behavior.

    Caller forms lrD = log_fD - log_fN_pathogenic (pathogenic-direction,
    paired with P(Y=1,M=1)) and lrPB = log_fP_raw - log_fN_benign
    (benign-direction, paired with P(Y=1)).
    """
    params = fit['fit']['component_params']
    log_fP_raw = density_utils.mixture_pdf(score_range, params, w_P_raw, multivariate=multivariate)

    if pathomechanism_method == 'boundary':
        if labeled_scores is None:
            raise ValueError("labeled_scores is required when pathomechanism_method='boundary'")
        f_N_at_labeled = density_utils.joint_densities(
            labeled_scores, params, w_N_anchor, multivariate=multivariate).sum(axis=0)
        f_Praw_at_labeled = density_utils.joint_densities(
            labeled_scores, params, w_P_raw, multivariate=multivariate).sum(axis=0)
        match_frac = _pathomechanism_boundary_match_frac(f_N_at_labeled, f_Praw_at_labeled)
        log_fD = None
        # match_frac == 1 is legitimate here (see
        # _pathomechanism_boundary_match_frac's docstring) -- <= 1, not < 1.
        if np.isfinite(match_frac) and 0 < match_frac <= 1:
            f_N_at_range = np.exp(density_utils.mixture_pdf(
                score_range, params, w_N_anchor, multivariate=multivariate))
            if match_frac < _PATHOMECHANISM_MATCH_FRAC_FLOOR:
                # f_D not estimable at this scale (see
                # _PATHOMECHANISM_MATCH_FRAC_FLOOR) -- rather than divide by
                # match_frac and amplify numerator noise into a corrupted,
                # non-monotonic curve, fall back to log_fD == the anchor
                # density itself. This makes lrD = log_fD - log_fN_pathogenic
                # exactly 0 everywhere below (log_fN_pathogenic reduces to
                # the same anchor density here too: unmixing pop_linear
                # against a "positive" density equal to pop_linear itself is
                # a no-op, f_negative = pop_linear regardless of prior) --
                # i.e. no pathogenic-direction evidence, in both PN and PU
                # mode, instead of a meaningless one.
                log_fD = np.log(f_N_at_range)
            else:
                fd_linear = _unmix_density(np.exp(log_fP_raw), 1 - match_frac, f_N_at_range)
                log_fD = np.log(fd_linear)
    else:
        build_w_D = {
            'subtraction': _w_D_subtraction,
            'masking': _w_D_masking,
        }[pathomechanism_method]
        w_D = build_w_D(w_P_raw, w_N_anchor)
        log_fD = (density_utils.mixture_pdf(score_range, params, w_D, multivariate=multivariate)
                  if w_D is not None else None)

    if not is_pu:
        log_fN = density_utils.mixture_pdf(score_range, params, w_N_anchor, multivariate=multivariate)
        return log_fD, log_fN, log_fP_raw, (log_fN if log_fD is not None else None)

    if prior_pathogenic is None or prior_benign is None:
        raise ValueError("prior_pathogenic and prior_benign are required when is_pu=True")

    pop_linear = np.exp(density_utils.mixture_pdf(score_range, params, w_N_anchor, multivariate=multivariate))
    fp_raw_linear = np.exp(log_fP_raw)
    log_fN_benign = np.log(_unmix_density(pop_linear, prior_benign, fp_raw_linear))

    log_fN_pathogenic = None
    if log_fD is not None:
        fd_linear = np.exp(log_fD)
        log_fN_pathogenic = np.log(_unmix_density(pop_linear, prior_pathogenic, fd_linear))

    return log_fD, log_fN_benign, log_fP_raw, log_fN_pathogenic


def apply_pathomechanism_correction(pathomechanism_method, labeled_scores, population,
                                     params, w_P_raw, w_N, multivariate=False, **kwargs):
    """Shared method-dispatch entry point for the pathomechanism prior
    correction, used by both the univariate (get_fit_prior, above) and
    multivariate (src/assay_calibration/multivariate_analysis/mv_calibration.py) prior-EM paths so the two never
    drift out of sync on which f_D construction "subtraction"/"masking" maps
    to, or on the off/None contract.

    pathomechanism_method is None -> disabled, returns (None, np.nan)
    (mirrors compute_pathomechanism_pathogenic_density's own degenerate-case
    contract, so callers can treat "off" and "degenerate excess/mask" the
    same way downstream).

    'boundary' takes no mechanism_match_frac_init/max_em_steps/tolerance
    kwargs (its estimate is closed-form, not iterative) -- callers must omit
    these in **kwargs when pathomechanism_method='boundary' (passing them
    raises a TypeError, since compute_pathomechanism_pathogenic_density_boundary
    has no **kwargs catch-all); see get_fit_prior's conditional kwarg
    construction for the pattern.

    Returns (pathogenic_density_at_population_points_or_None, mechanism_match_frac_hat).
    """
    if pathomechanism_method is None:
        return None, np.nan
    if pathomechanism_method == 'boundary':
        return compute_pathomechanism_pathogenic_density_boundary(
            labeled_scores, population, params, w_P_raw, w_N, multivariate=multivariate)
    density_fn = {
        'subtraction': compute_pathomechanism_pathogenic_density,
        'masking': compute_pathomechanism_pathogenic_density_masked,
    }[pathomechanism_method]
    return density_fn(labeled_scores, population, params, w_P_raw, w_N,
                       multivariate=multivariate, **kwargs)


def get_fit_prior(fit, scoreset_or_scores, benign_method, pathogenic_idx=0, benign_idx=1, gnomad_idx=2, synonymous_idx=3,
                  sample_assignments=None, pathogenic_row_mask=None, **kwargs):
    if benign_idx is None:
        benign_idx = synonymous_idx
    if synonymous_idx is None:
        synonymous_idx = benign_idx

    if benign_method == 'synonymous':
        benign_idx = synonymous_idx

    params = fit['fit']['component_params']
    weights = fit['fit']['weights']

    # Accept either a Scoreset object or pre-extracted (scores, sample_assignments) arrays.
    # Passing raw numpy arrays avoids pickling the full Scoreset for every parallel task.
    if sample_assignments is not None:
        scores = scoreset_or_scores
        sa = sample_assignments
    else:
        scores = scoreset_or_scores.scores
        sa = scoreset_or_scores.sample_assignments

    # LR-filtered "effective pathogenic sample": recompute this fit's pathogenic
    # mixture weights restricted to just the rows in pathogenic_row_mask (a fixed
    # subset of pathogenic-labeled rows selected by compute_lr_filtered_pathogenic_mask
    # as reliably separated from the benign/synonymous/population reference). This is
    # the same single M-step (frozen component_params, current weights as the E-step
    # prior) that originally produced weights[pathogenic_idx] from the full labeled
    # sample -- see get_sample_weights/get_sample_weights_and_ll in
    # fit_utils/cfusn/update_steps.py -- just re-run on fewer rows. Components mostly
    # explained by excluded ("looks normal") rows naturally end up with near-zero
    # weight; everywhere below that reads weights[pathogenic_idx] then sees this
    # cleaned vector instead of the raw one, so the ordinary standard-EM/PU/NU
    # dispatch below runs unmodified on an "effective pathogenic" reference density
    # that excludes mass overlapping the negative class.
    if pathogenic_row_mask is not None and pathogenic_idx is not None:
        filtered_scores = scores[pathogenic_row_mask]
        if len(filtered_scores) > 0:
            posts = density_utils.component_posteriors(
                filtered_scores, params, weights[pathogenic_idx]
            )
            weights = list(weights)
            weights[pathogenic_idx] = posts.mean(1)

    # EXPERIMENTAL, opt-in alternative to the LR-filtered pathogenic_row_mask
    # above: a weight-vector-based estimate of the pathogenic-labeled
    # sample's mechanistic-coverage fraction (see
    # compute_pathomechanism_pathogenic_density / _masked above), in place of
    # an ad-hoc row filter. pathomechanism_method selects the f_D construction
    # ("subtraction" or "masking"); None disables the mechanism entirely.
    # Needs a pathogenic-labeled sample; anchors against the benign/synonymous
    # density when present (PN/standard mode) or, for PU-only datasets,
    # against the raw gnomAD/population density directly (mirrors
    # compute_lr_filtered_pathogenic_mask's own PU fallback -- see
    # compute_pathomechanism_pathogenic_density's docstring). Mutually
    # exclusive with pathogenic_row_mask.
    pathomechanism_method = kwargs.get("pathomechanism_method", None)
    mechanism_match_frac_hat = np.nan
    pathomechanism_no_pu_support = False
    if pathomechanism_method is not None and pathogenic_idx is not None and pathogenic_row_mask is not None:
        raise ValueError(
            "pathomechanism_method and filter_pathogenic_sample_by_lr are "
            "mutually exclusive pathogenic-sample-cleaning strategies -- "
            "enable only one (pass --no-filter-pathogenic-sample-by-lr "
            "alongside --pathomechanism-method)."
        )

    population = scores[sa[:, gnomad_idx]]
    # print(f"population: {len(population)} samples")
    
    pop_density = density_utils.joint_densities(
        population, params, weights[gnomad_idx]
    ).sum(axis=0)
    
    # Compute pathogenic density if available
    pathogenic_density = []
    if pathogenic_idx is not None:
        pathogenic_density = density_utils.joint_densities(
            population, params, weights[pathogenic_idx]
        ).sum(axis=0)
        assert len(pathogenic_density) == len(population)
    
    # Compute benign density if available
    benign_density = []
    if benign_idx is not None and synonymous_idx is not None:
        w_b = _get_benign_reference_weights(weights, benign_method, benign_idx, synonymous_idx)
        benign_density = density_utils.joint_densities(
            population, params, w_b
        ).sum(axis=0)
        assert len(benign_density) == len(population)
    # print(f"benign_density: {benign_density}")

    if pathomechanism_method is not None and pathogenic_idx is not None and len(pathogenic_density) != 0:
        is_pu = len(benign_density) == 0
        w_N_anchor = weights[gnomad_idx] if is_pu else w_b
        labeled_scores = scores[sa[:, pathogenic_idx].astype(bool)]
        pathomechanism_kwargs = {}
        if pathomechanism_method != "boundary":
            # 'boundary' is closed-form -- no EM init/steps/tolerance to pass.
            pathomechanism_kwargs = dict(
                mechanism_match_frac_init=kwargs.get("pathomechanism_match_frac_init", 0.5),
                max_em_steps=kwargs.get("pathomechanism_max_em_steps", 10000),
                tolerance=kwargs.get("pathomechanism_tolerance", 1e-8),
            )
        corrected, mechanism_match_frac_hat = apply_pathomechanism_correction(
            pathomechanism_method, labeled_scores, population, params,
            weights[pathogenic_idx], w_N_anchor, **pathomechanism_kwargs,
        )
        if corrected is None:
            # No assay-relevant excess anywhere (w_P_raw <= w_N on every
            # component) -- this bootstrap fit's own raw pathogenic density
            # IS the degenerate reference this correction exists to move
            # away from (e.g. a fit that collapses the pathogenic-labeled
            # sample onto the same component as the benign/gnomAD anchor,
            # making the two densities identical and any prior EM built on
            # them a trivial fixed point at its own initialization --
            # observed on SCN5A_Glazer_2020's 2c fits). Silently falling
            # back to that raw density here would just reproduce the
            # degeneracy under a different name, so discard this fit from
            # prior estimation instead of using it. If every fit for this
            # combo is degenerate this way, process_component_fits falls
            # back to raw-density prior estimation for the whole combo.
            return np.nan, np.nan
        pathogenic_density = corrected

        # For 'boundary' in the default "product" joint-prior mode, the
        # caller (visualize.py) computes P(Y=1,M=1) as P(Y=1)*P(M=1|Y=1)
        # and never reads this function's own prior_estimate return value at
        # all -- so skip the standard/PU/NU EM below entirely rather than
        # run a full iterative re-derivation against f_D on every bootstrap
        # fit just to discard it. Only "boundary" has a meaningfully
        # skippable direct re-derivation here; subtraction/masking's EM
        # (_mechanism_match_em, already run inside apply_pathomechanism_
        # correction above) is the SAME computation that produces
        # mechanism_match_frac_hat, not a separate one, so there's nothing
        # extra to skip for them.
        if (pathomechanism_method == "boundary"
                and kwargs.get("pathomechanism_boundary_joint_prior", "product") == "product"):
            return np.nan, mechanism_match_frac_hat

        if is_pu:
            # With a typically small gnomAD sample, "zero density
            # everywhere the population lands" means f_D's isolated
            # disease-relevant component(s) have no detectable trace in
            # gnomAD -- itself the boundary estimator's floor signal
            # (P(Y=1) -> 0), not an absence of information. Confirmed on
            # TARDBP_Bolognesi_Faure_2019 (68 gnomAD points): treating
            # this as "no info" and discarding/defaulting to 0.1 (the
            # generic PU fallback below) reproduced a selection-biased
            # median over only the fits with fortuitous population
            # overlap; flooring instead used ~2x as many bootstrap fits
            # and gave a prior consistent with the assay's actual
            # (near-zero) detectable disease signal in this population.
            pathomechanism_no_pu_support = not np.any(pathogenic_density > 1e-300)

    if len(pathogenic_density) != 0 and len(benign_density) != 0:
        mode = 'standard'  # Both labeled classes available
        prior_estimate = 0.5
        # print("standard prior estimation")
    elif len(pathogenic_density) != 0 and len(benign_density) == 0:
        mode = 'positive_unlabeled'  # Only pathogenic available
        prior_estimate = 0.1
        # print("PU prior estimation")
    elif len(pathogenic_density) == 0 and len(benign_density) != 0:
        mode = 'negative_unlabeled'  # Only benign available
        prior_estimate = 0.9
    else:
        raise ValueError("Must have at least one of pathogenic or benign density")

    # default_prior = 0.1
    # if mode == 'negative_unlabeled' or mode == 'positive_unlabeled':
    #     kl_divergence = np.mean(np.abs((benign_density if mode == 'negative_unlabeled' else pathogenic_density) - pop_density) / (pop_density + 1e-10))
    #     if kl_divergence < 0.1:
    #         return default_prior
    
    if mode == 'standard':
        # Standard EM -- see estimate_prior_from_class_densities for the shared
        # implementation (also used by src/assay_calibration/multivariate_analysis/mv_calibration.py).
        prior_estimate = estimate_prior_from_class_densities(
            pathogenic_density, benign_density, mode='standard',
            max_em_steps=kwargs.get("max_em_steps", 10000),
            tolerance=kwargs.get("tolerance", 1e-6),
        )
        # Joint EM is the well-behaved estimator (uses both fp and fb) -- it
        # doesn't suffer the saturation pathology PU/NU unmixing does, and
        # floor-instead-of-discard here only ends up changing which bootstrap
        # fits' curves feed the final LR+ plot for datasets that never had a
        # saturation problem to begin with. Keep the original behavior.
        # keep_extreme_prior (PipelineConfig.keep_extreme_prior_fits): keep the fit and its
        # raw estimate (possibly 0 or 1); the caller clips the final median prior instead.
        if (not kwargs.get("keep_extreme_prior", False)
                and (prior_estimate <= 0.001 or prior_estimate >= 0.999)):
            return np.nan, mechanism_match_frac_hat
        return prior_estimate, mechanism_match_frac_hat

    # Boundary/"unmixing" estimator (Blanchard, Lee & Scott 2010; Scott 2015),
    # evaluated at the gnomAD/population sample points (pop_density,
    # pathogenic_density, benign_density are already computed there above):
    #   f_pop(x) = P(Y=1)*f_labeled(x) + (1-P(Y=1))*f_other(x), f_other(x) >= 0
    #   everywhere requires P(Y=1) <= f_pop(x)/f_labeled(x) pointwise, so the
    #   tightest identifiable P(Y=1) is inf_x [f_pop(x) / f_labeled(x)].
    # More intuitive than mean-matching for PU/NU (formerly used here, and
    # confirmed empirically non-functional: its iterative update is a
    # tautology -- its fixed point doesn't depend on the current estimate --
    # so in finite samples it diverges geometrically to 0 or 1 depending only
    # on per-fit noise, not signal). The boundary/unmixing estimator instead asks
    # "what's the largest P(Y=1) consistent with a non-negative residual
    # population density everywhere", the direct generalization of the
    # standard PN joint-EM logic (above) to the one-labeled-class case.
    # Floor/discard (0.01 / >=0.99) is applied inside
    # estimate_prior_from_class_densities -- identical policy shared with MV.
    if mode == 'negative_unlabeled':
        return estimate_prior_from_class_densities(
            benign_density, pop_density, mode='negative_unlabeled'), mechanism_match_frac_hat

    # positive_unlabeled. See pathomechanism_no_pu_support comment above:
    # distinguish "pathomechanism correction found no population support at
    # all" (a real floor signal, no_signal_prior=0.01) from the generic
    # no-signal fallback used everywhere else (no_signal_prior=0.1).
    no_signal_prior = 0.01 if pathomechanism_no_pu_support else 0.1
    return estimate_prior_from_class_densities(
        pathogenic_density, pop_density, mode='positive_unlabeled',
        no_signal_prior=no_signal_prior), mechanism_match_frac_hat

def get_bootstrap_score_ranges(fitIdx, fit, fp, fb, score_range, fit_priors, point_values,
                                acmg_mapping_method="tavtigian", acmg_bayes_targets=None,
                                acmg_bayes_floor_at_neutral=False):
    fit_xmin, fit_xmax = fit['fit']['xlims']
    mask = (score_range >= fit_xmin) & (score_range <= fit_xmax)# & ((fp > -7.0) | (fb > -7.0)) # add min density check

    # log_fp_local = np.zeros_like(fp)
    # log_fb_local = np.zeros_like(fb)

    # CRITICAL: IGNORE BOOTSTRAPS THAT DON'T SPAN DATA POINT. MARKING 0 WILL CAUSE STRANGE LR+ CURVES AT EXTREMES
    log_fp_local = np.full_like(fp, np.nan, dtype=float)
    log_fb_local = np.full_like(fb, np.nan, dtype=float)

    log_fp_local[mask] = fp[mask]
    log_fb_local[mask] = fb[mask]

    lrP = log_fp_local[mask] - log_fb_local[mask]
    s = score_range[mask]

    if acmg_mapping_method == "acmg_bayes":
        from .fit import calculate_classification_ranges
        ranges_p, ranges_b, thresholds = calculate_classification_ranges(
            lrP, lrP, fit_priors[fitIdx], s, targets=acmg_bayes_targets,
            floor_at_neutral=acmg_bayes_floor_at_neutral,
        )
        # For acmg_bayes, "C" slot carries the LR+ threshold dict (not an int).
        C = thresholds
    else:
        ranges_p, ranges_b, C = calculate_score_ranges(
            lrP, lrP, fit_priors[fitIdx], s, point_values,
            acmg_mapping_method=acmg_mapping_method,
        )
        if C is not None:
            C = int(C)  # tavtigian path returns int

    if prior_invalid(fit_priors[fitIdx]):
        log_fp_local = np.full_like(fp, np.nan, dtype=float)
        log_fb_local = np.full_like(fb, np.nan, dtype=float)
        for key in ranges_p:
            ranges_p[key] = []
        for key in ranges_b:
            ranges_b[key] = []
        C = np.nan

    return fitIdx, log_fp_local, log_fb_local, ranges_p, ranges_b, C


def get_bootstrap_score_ranges_dual(fitIdx, fit, log_fD, log_fN_benign, log_fP_raw,
                                     score_range, fit_priors_pathogenic, fit_priors_benign,
                                     point_values, acmg_mapping_method="tavtigian",
                                     log_fN_pathogenic=None):
    """Dual-(prior, LR)-pair sibling of get_bootstrap_score_ranges for the
    pathomechanism two-LR/two-prior PS3/BS3 scheme: pathogenic-direction
    thresholds are derived from (P(Y=1,M=1), log_fD - log_fN_pathogenic),
    benign-direction thresholds from (P(Y=1), log_fP_raw - log_fN_benign) -- see
    compute_pathomechanism_lr_curves and calculate_score_ranges_dual.

    log_fN_benign and log_fN_pathogenic are the SAME array in PN mode (no
    unmixing -- compute_pathomechanism_lr_curves returns one shared curve
    there) but genuinely DIFFERENT purified densities in PU mode (each
    direction unmixed against population using its own native prior).
    log_fN_pathogenic defaults to log_fN_benign for callers that only have
    one curve (PN mode, or pre-unmixing-fix callers).

    log_fD may be None for this fit (degenerate excess/mask -- see
    compute_pathomechanism_lr_curves); in that case the pathogenic-direction
    ranges/C are returned empty/nan for this fit, exactly as
    get_bootstrap_score_ranges already does when fit_priors[fitIdx] is
    invalid, while the benign-direction ranges (which don't depend on log_fD)
    are still computed normally.

    acmg_mapping_method == "acmg_bayes" is not supported here (no dual-prior
    sibling for calculate_classification_ranges yet) -- raises
    NotImplementedError rather than silently using the wrong thresholds.
    """
    if acmg_mapping_method == "acmg_bayes":
        raise NotImplementedError(
            "acmg_mapping_method='acmg_bayes' is not yet supported together "
            "with pathomechanism_method (no dual-prior sibling for "
            "calculate_classification_ranges); use 'tavtigian', "
            "or set pathomechanism_method=None."
        )
    if log_fN_pathogenic is None:
        log_fN_pathogenic = log_fN_benign

    fit_xmin, fit_xmax = fit['fit']['xlims']
    mask = (score_range >= fit_xmin) & (score_range <= fit_xmax)

    def _masked(arr):
        out = np.full_like(arr, np.nan, dtype=float)
        out[mask] = arr[mask]
        return out

    log_fN_benign_local = _masked(log_fN_benign)
    log_fN_pathogenic_local = _masked(log_fN_pathogenic)
    log_fP_raw_local = _masked(log_fP_raw)
    lrPB = log_fP_raw_local[mask] - log_fN_benign_local[mask]
    s = score_range[mask]

    prior_pathomechanism = fit_priors_pathogenic[fitIdx]  # P(Y=1, M=1)
    prior_raw = fit_priors_benign[fitIdx]  # P(Y=1)

    if log_fD is not None:
        log_fD_local = _masked(log_fD)
        lrD = log_fD_local[mask] - log_fN_pathogenic_local[mask]
    else:
        log_fD_local = np.full_like(log_fN_benign, np.nan, dtype=float)
        lrD = np.full_like(s, np.nan, dtype=float)

    ranges_p, ranges_b, C_p, C_b = calculate_score_ranges_dual(
        lrD, lrPB, prior_pathomechanism, prior_raw, s, point_values,
        acmg_mapping_method=acmg_mapping_method,
    )

    if log_fD is None or prior_invalid(prior_pathomechanism):
        log_fD_local = np.full_like(log_fN_benign, np.nan, dtype=float)
        log_fN_pathogenic_local = np.full_like(log_fN_benign, np.nan, dtype=float)
        for key in ranges_p:
            ranges_p[key] = []
        C_p = np.nan
    elif prior_pathomechanism < _PATHOMECHANISM_MATCH_FRAC_FLOOR:
        # log_fD already exists as a flat, no-evidence curve here (this
        # fit's own match_frac was below the same floor, so
        # compute_pathomechanism_lr_curves' own gate already built log_fD
        # as the anchor density rather than unmixing -- see
        # _PATHOMECHANISM_MATCH_FRAC_FLOOR) -- keep that curve for display
        # (log_fD_local/log_fN_pathogenic_local untouched; lrD is already
        # ~0 everywhere, so this changes no evidence assignment on its
        # own), but discard only the point-range/C computation above: an
        # unfloored prior this tiny degenerates get_tavtigian_constant's
        # C-search into C=1 (no C in its search range can satisfy any ACMG
        # posterior target against such a small prior), making every
        # threshold trivially satisfied by any LR -- including this fit's
        # own flat LR=1 curve, which would otherwise spuriously fill every
        # tier up to the maximum rather than contributing no evidence.
        for key in ranges_p:
            ranges_p[key] = []
        C_p = np.nan

    if prior_invalid(prior_raw):
        log_fP_raw_local = np.full_like(log_fN_benign, np.nan, dtype=float)
        log_fN_benign_local = np.full_like(log_fN_benign, np.nan, dtype=float)
        for key in ranges_b:
            ranges_b[key] = []
        C_b = np.nan

    return (fitIdx, log_fD_local, log_fN_benign_local, log_fP_raw_local,
            ranges_p, ranges_b, C_p, C_b, log_fN_pathogenic_local)


def check_thresholds_reached(lrPlus, tau, point_values, pathogenicOrBenign):
    
    if pathogenicOrBenign == "benign":
        point_values = -1 * np.array(point_values)
    
    reached = {}
    
    for p in point_values:
        if pathogenicOrBenign == "pathogenic":
            # Check if LR+ ever exceeds threshold
            reached[p] = np.any(lrPlus >= tau[abs(p)-1]) # list idx not dict
        else:
            # Check if LR+ ever goes below threshold
            reached[p] = np.any(lrPlus <= tau[abs(p)-1]) # list idx not dict
    
    return reached



def _compute_log_fp_only(fit, score_range, pathogenic_idx):
    """Compute only log pathogenic density for one bootstrap fit (top-level for pickling)."""
    params = fit['fit']['component_params']
    weights = fit['fit']['weights']
    return density_utils.mixture_pdf(score_range, params, weights[pathogenic_idx])


def compute_single_fit_log_densities(fit, prior, score_range, benign_method,
                                     pathogenic_idx=0, benign_idx=1,
                                     gnomad_idx=2, synonymous_idx=3,
                                     log_density_threshold=-7.0,
                                     precomputed_log_fp=None):
    """
    Compute log pathogenic and benign densities for a single fit.
    
    When both densities fall below log_density_threshold, sets both to the 
    threshold value so that log_fp - log_fb ≈ 0 (LR+ ≈ 1, neutral evidence).
    
    [Keep all existing parameters]
    
    log_density_threshold : float, optional
        Threshold below which both densities are set equal to avoid 
        spurious evidence from dividing tiny numbers. Default: -7.0
    
    Returns
    -------
    log_fp : np.ndarray or None
        Log pathogenic density (None if prior invalid)
    log_fb : np.ndarray or None
        Log benign density (None if prior invalid)
    """
    # Skip if prior estimation failed
    if np.isnan(prior) or prior <= 0 or prior >= 1:
        return None, None
    
    params = fit['fit']['component_params']
    weights = fit['fit']['weights']
    
    # Get population density (always available)
    log_pop = density_utils.mixture_pdf(score_range, params, weights[gnomad_idx])
    pop_linear = np.exp(log_pop)
    
    have_pathogenic = pathogenic_idx is not None
    have_benign = (benign_idx is not None) or (synonymous_idx is not None)
    
    if not have_pathogenic and not have_benign:
        raise ValueError("Must have at least one of pathogenic or benign sample")
    
    if have_pathogenic:
        if precomputed_log_fp is not None:
            log_fp = precomputed_log_fp
        else:
            log_fp = density_utils.mixture_pdf(score_range, params, weights[pathogenic_idx])
    else:
        # Get effective benign weights
        if benign_method == 'synonymous' and synonymous_idx is not None:
            w_benign_eff = weights[synonymous_idx]
        elif benign_method == 'avg' and benign_idx is not None and synonymous_idx is not None:
            w_benign_eff = (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
        else:
            w_benign_eff = weights[synonymous_idx if benign_idx is None else benign_idx]
        
        log_fb_temp = density_utils.mixture_pdf(score_range, params, w_benign_eff)
        fb_linear = np.exp(log_fb_temp)
        
        # Unmix: f_p = [f_pop - (1-alpha)*f_b] / alpha
        fp_linear = (pop_linear - (1 - prior) * fb_linear) / prior
        
        # Clip negative values (numerical issues)
        fp_linear = np.maximum(fp_linear, pop_linear * 1e-10)  # At least 1e-10 of population
        log_fp = np.log(fp_linear)
    
    if have_benign:
        # Get effective benign weights
        if benign_method == "synonymous" and synonymous_idx is not None:
            w_benign_eff = weights[synonymous_idx]
        elif benign_method == 'avg' and benign_idx is not None and synonymous_idx is not None:
            w_benign_eff = (np.array(weights[benign_idx]) + np.array(weights[synonymous_idx])) / 2
        else:
            w_benign_eff = weights[synonymous_idx if benign_idx is None else benign_idx]
        
        log_fb = density_utils.mixture_pdf(score_range, params, w_benign_eff)
    else:
        fp_linear = np.exp(log_fp)
        
        # Unmix: f_b = [f_pop - alpha*f_p] / (1-alpha)
        fb_linear = (pop_linear - prior * fp_linear) / (1 - prior)
        
        # Clip negative values
        fb_linear = np.maximum(fb_linear, pop_linear * 1e-10)  # At least 1e-10 of population
        log_fb = np.log(fb_linear)
    
    # # WHERE BOTH DENSITIES ARE VERY LOW, SET THEM EQUAL
    # # This ensures log_fp - log_fb ≈ 0
    # low_density_mask = np.logical_and(
    #     log_fp < log_density_threshold,
    #     log_fb < log_density_threshold
    # )
    
    # # Set both to the threshold value where both are low
    # log_fp = np.where(low_density_mask, log_density_threshold, log_fp)
    # log_fb = np.where(low_density_mask, log_density_threshold, log_fb)
    
    return log_fp, log_fb


def get_variant_oob_bootstrap_indices(scoreset, dataset_splits, valid_mask):
    """
    For each variant in scoreset, find which bootstrap iterations (filtered indices) 
    it appears in the validation set.
    
    Parameters:
    -----------
    valid_mask : np.ndarray
        Boolean mask of valid bootstrap iterations (from fit_priors filtering)
    
    Returns:
    --------
    variant_to_oob_boots : dict
        Maps scoreset index -> list of FILTERED bootstrap indices
    """
    
    # Create mapping: original boot_idx -> filtered boot_idx
    original_to_filtered = {}
    filtered_idx = 0
    for original_idx in range(len(valid_mask)):
        if valid_mask[original_idx]:
            original_to_filtered[original_idx] = filtered_idx
            filtered_idx += 1
    
    # Build index mapping for fast lookup
    score_class_to_indices = defaultdict(list)
    
    for idx in range(len(scoreset.scores)):
        score = scoreset.scores[idx]
        class_indices = np.where(scoreset.sample_assignments[idx])[0]
        
        for class_idx in class_indices:
            key = (score, class_idx)
            score_class_to_indices[key].append(idx)
    
    # Map each scoreset index to its OOB bootstrap iterations
    variant_to_oob_boots = defaultdict(list)
    
    print(f"Building OOB mapping for {len(scoreset.scores)} variants...")
    print(f"  Valid bootstraps: {sum(valid_mask)}/{len(valid_mask)}")
    
    for boot_idx in sorted(dataset_splits.keys()):
        # Skip if this bootstrap was filtered out
        if not valid_mask[boot_idx]:
            continue
        
        # Get filtered index
        filtered_boot_idx = original_to_filtered[boot_idx]
        
        val_obs = dataset_splits[boot_idx]["val_observations"]
        val_assign = dataset_splits[boot_idx]["val_sample_assignments"]
        
        for obs, assign in zip(val_obs, val_assign):
            class_idx = np.where(assign)[0][0]
            key = (obs, class_idx)
            matching_indices = score_class_to_indices.get(key, [])
            
            # Add FILTERED bootstrap index to all matching variants
            for variant_idx in matching_indices:
                variant_to_oob_boots[variant_idx].append(filtered_boot_idx)
    
    print(f"Found OOB samples for {len(variant_to_oob_boots)}/{len(scoreset.scores)} variants")
    
    return variant_to_oob_boots

import pandas as pd

def make_variant_id(v):
    return f"{v.ID}_{v.Gene}_{v.Chrom}_{v.hgvs_c}"

def assign_points(assay_score, point_ranges):
    """Assign points based on which sub-range the assay_score falls into.

    `point_ranges` is the raw {key: [[lo, hi], ...]} structure -- a point
    value may have zero, one, or multiple disjoint sub-ranges (e.g. from
    bidirectional postprocessing), so every sub-range is checked directly
    rather than pre-flattening to a single bounding interval.
    """
    if assay_score is None or pd.isna(assay_score):
        return None

    for point_str, ranges in point_ranges.items():
        if not ranges:
            continue
        subranges = ranges if isinstance(ranges[0], (list, tuple)) else [ranges]
        for low, high in subranges:
            if low <= assay_score <= high:
                return int(point_str)
    return 0
