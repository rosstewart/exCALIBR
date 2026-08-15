"""
Per-variant evidence assignment (standard and out-of-bag).

Standard: assigns points using the global calibration (point_ranges) for every variant.
OOB: for each variant, rebuilds calibration using only bootstrap iterations where
     that variant was held out (validation set), then assigns points.
     This matches the full processing logic from assign_points.py.
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from joblib import Parallel, delayed
import logging

from ..fit_utils.fit import calculate_score_ranges, thresholds_from_prior
from ..fit_utils.point_ranges import (
    enforce_monotonicity_point_ranges,
    extend_points_to_xlims,
    clean_benign_fragments_no_extend,
    clean_bidirectional_pathogenic_evidence,
)
from .config import PipelineConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _assign_points(score: float, point_ranges: Dict) -> int:
    """Return the evidence point value whose sub-range contains *score*, else 0.

    `point_ranges` is the raw {key: [[lo, hi], ...]} structure (a point value
    may have zero, one, or multiple disjoint sub-ranges -- e.g. from
    bidirectional postprocessing, see `clean_bidirectional_pathogenic_evidence`)
    -- every sub-range is checked directly, with no prior flattening step.
    Flattening multiple disjoint sub-ranges into their bounding interval
    (the old behavior) silently merged the neutral score band between two
    real islands into one of them.
    """
    if score is None or np.isnan(score):
        return 0
    for key, ranges in point_ranges.items():
        if isinstance(ranges, np.ndarray):
            ranges = ranges.tolist()
        if not ranges:
            continue
        subranges = ranges if isinstance(ranges[0], (list, tuple, np.ndarray)) else [ranges]
        for lo, hi in subranges:
            if lo <= score <= hi:
                return int(key)
    return 0


def _get_variant_ids(scoreset) -> List[str]:
    """Best-effort variant ID extraction; falls back to integer index.

    *key* here is exactly get_variants_by_id's own grouping key
    (mavedb_variant_urn when the variant has one, else variant.ID) -- use it
    directly rather than re-deriving an id from v.ID, which is a different,
    typically-unset attribute (renders as the literal string "None" in every
    exported id otherwise). Two variants that are genomically identical
    (same Gene/Chrom/hgvs_c) but are distinct MaveDB library entries with
    independently measured scores -- e.g. different barcoded/codon-level
    designs collapsing to the same net single-nucleotide cDNA change -- are
    correctly kept as separate groups by get_variants_by_id (grouped by
    mavedb_variant_urn), but previously collided into one exported
    variant_id string once v.ID (None for both) made the distinguishing
    field disappear. Using *key* preserves that distinction end to end.
    """
    ids = []
    if hasattr(scoreset, "get_variants_by_id") and hasattr(scoreset, "_keep_mask"):
        variants_by_id = scoreset.get_variants_by_id()
        kept = 0
        for all_idx, (key, variants) in enumerate(variants_by_id.items()):
            if scoreset._keep_mask[all_idx]:
                v = variants[0]
                if hasattr(v, "ID"):
                    ids.append(f"{key}_{v.Gene}_{v.Chrom}_{v.hgvs_c}")
                else:
                    ids.append(f"variant_{kept}")
                kept += 1
    else:
        ids = [f"variant_{i}" for i in range(len(scoreset.scores))]
    return ids


def _get_variant_aa_fields(scoreset) -> Optional[List[Dict]]:
    """Per-kept-variant {nucleotide_or_aa, aa_ref, aa_pos, aa_alt}, aligned
    1:1 with _get_variant_ids's output -- exact structural mirror of
    _get_variant_is_vus/_get_variant_is_snv (same _keep_mask gate, same
    lockstep with the kept-population loop).

    These fields distinguish an amino-acid-level (protein-level) assay
    measurement from a nucleotide-level one -- needed because a single
    aa-level measurement is routinely stored as multiple master-TSV rows
    (one per degenerate codon that produces the same amino-acid change), so
    the Chrom/hgvs_c that _get_variant_ids picks (`variants[0]`) is an
    arbitrary choice among those codons for aa-level rows, not a stable
    identity -- see analysis.multi_scoreset.genomic_variant_key, which uses
    these fields instead of Chrom/hgvs_c for aa-level cross-assay matching.
    Read off variants[0], same object _get_variant_ids itself uses for
    Gene/Chrom/hgvs_c -- these four fields are constant within a
    mavedb_variant_urn group (verified against the master TSV), so this is
    exact, not an approximation.
    """
    if not (hasattr(scoreset, "get_variants_by_id") and hasattr(scoreset, "_keep_mask")):
        return None
    fields = []
    variants_by_id = scoreset.get_variants_by_id()
    for all_idx, (_, variants) in enumerate(variants_by_id.items()):
        if scoreset._keep_mask[all_idx]:
            v0 = variants[0]
            fields.append({
                "nucleotide_or_aa": getattr(v0, "nucleotide_or_aa", None),
                "aa_ref": getattr(v0, "aa_ref", None),
                "aa_pos": getattr(v0, "aa_pos", None),
                "aa_alt": getattr(v0, "aa_alt", None),
            })
    return fields


def _get_variant_is_vus(scoreset) -> Optional[List[bool]]:
    """Per-kept-variant VUS flag, aligned 1:1 with _get_variant_ids's output.

    Always ClinVar-2026-based, regardless of the Scoreset's own
    clinvar_release (used only for that variant's own P/LP/B/LB
    classification) -- see Variant.is_vus in data_utils/dataset.py, which is
    hardcoded to clinvar_sig_2026 ("VUS ALWAYS 2026 SINCE NOT CONTROL") even
    for clinvar_2018-mode datasets (BRCA1/MSH2/PTEN/TP53). Deriving VUS by
    negation from the `sample` column instead (not P/LP, not B/LB, not
    gnomAD, not Synonymous) would silently use whichever release the dataset
    itself loaded controls from, which is wrong for clinvar_2018 datasets.

    Returns None (not a per-row list) when the Scoreset has no per-variant
    ClinVar annotation at all (e.g. BasicScoreset) -- same "only export if
    the scoreset actually has this information" contract as auth_labels.
    A group with multiple raw rows (deduplicated by ID) counts as VUS if ANY
    row in the group does, matching how Scoreset itself builds _vus_scores
    at construction time.
    """
    if not (hasattr(scoreset, "get_variants_by_id") and hasattr(scoreset, "_keep_mask")):
        return None
    is_vus = []
    variants_by_id = scoreset.get_variants_by_id()
    for all_idx, (_, variants) in enumerate(variants_by_id.items()):
        if scoreset._keep_mask[all_idx]:
            is_vus.append(any(getattr(v, "is_vus", False) for v in variants))
    return is_vus


def _get_variant_is_snv(scoreset) -> Optional[List[bool]]:
    """Per-kept-variant SNV flag, aligned 1:1 with _get_variant_ids's output --
    exact structural mirror of _get_variant_is_vus (same _keep_mask gate, same
    lockstep with the kept-population loop). A group counts as SNV if ANY raw
    row in it does (see Variant.is_snv), matching is_vus's any-of-group rule.
    """
    if not (hasattr(scoreset, "get_variants_by_id") and hasattr(scoreset, "_keep_mask")):
        return None
    is_snv = []
    variants_by_id = scoreset.get_variants_by_id()
    for all_idx, (_, variants) in enumerate(variants_by_id.items()):
        if scoreset._keep_mask[all_idx]:
            is_snv.append(any(getattr(v, "is_snv", False) for v in variants))
    return is_snv


def _get_vus_only_extra_rows(scoreset) -> List[Dict]:
    """Extra per-variant rows for ClinVar-VUS variants that `_get_variant_ids`/
    `_get_variant_is_vus` drop -- variants outside every fitting-relevant
    sample group (Pathogenic/Benign/population/Synonymous), so
    `scoreset._keep_mask` is False for them even though they're perfectly
    real, analyzable variants. `keep_mask`/`sample_assignments`/
    `scoreset.scores` (the fitting-relevant Scoreset internals) are NEVER
    touched here or anywhere in this module -- this is a purely additive,
    independent side-channel, read directly off the raw Variant objects via
    the same re-callable `get_variants_by_id()` grouping `_get_variant_ids`
    itself uses (so a group's inclusion/exclusion can never disagree between
    the two: `_keep_mask` is computed once per group, never partially).

    Returns one dict per excluded VUS group: {variant_id, score, auth_label,
    is_vus (always True), is_snv, nucleotide_or_aa, aa_ref, aa_pos, aa_alt}.
    `variant_id` reuses _get_variant_ids's own id-construction convention so
    ids stay unique and consistent whether a variant ends up in the main
    (kept) population or here.
    """
    if not (hasattr(scoreset, "get_variants_by_id") and hasattr(scoreset, "_keep_mask")):
        return []
    extra = []
    variants_by_id = scoreset.get_variants_by_id()
    excluded = 0
    for all_idx, (key, variants) in enumerate(variants_by_id.items()):
        if scoreset._keep_mask[all_idx]:
            continue
        if not any(getattr(v, "is_vus", False) for v in variants):
            continue
        v0 = variants[0]
        if hasattr(v0, "ID"):
            variant_id = f"{key}_{v0.Gene}_{v0.Chrom}_{v0.hgvs_c}"
        else:
            variant_id = f"vus_only_{excluded}"
        extra.append({
            "variant_id": variant_id,
            "score": float(v0.assay_score),
            "auth_label": v0.auth_label,
            "is_vus": True,
            "is_snv": any(getattr(v, "is_snv", False) for v in variants),
            "nucleotide_or_aa": getattr(v0, "nucleotide_or_aa", None),
            "aa_ref": getattr(v0, "aa_ref", None),
            "aa_pos": getattr(v0, "aa_pos", None),
            "aa_alt": getattr(v0, "aa_alt", None),
        })
        excluded += 1
    return extra


# ---------------------------------------------------------------------------
# Standard (in-bag) per-variant assignment
# ---------------------------------------------------------------------------

def _compute_bootstrap_lr_percentiles(
    scores: np.ndarray,
    calibration: Dict,
    percentile: float = 5.0,
    benign_percentile: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Interpolate per-bootstrap LR+ curves at each variant score and return
    (lr_plus_5th, lr_plus_95th, posterior_5th, posterior_95th) arrays of shape
    (n_variants,), or (None, None, None, None) when bootstrap curves unavailable.

    ``benign_percentile`` defaults to ``100 - percentile`` (historical
    symmetric pairing) when not given explicitly.
    """
    if benign_percentile is None:
        benign_percentile = 100 - percentile
    score_range = np.asarray(calibration.get("score_range", []))
    if len(score_range) == 0:
        return None, None, None, None

    # Resolve per-bootstrap log-LR+ matrix: prefer log_lr_plus (direct), else
    # compute from log_fp - log_fb
    if "log_lr_plus" in calibration:
        log_lr = np.asarray(calibration["log_lr_plus"])
    elif "log_fp" in calibration and "log_fb" in calibration:
        log_lr = np.asarray(calibration["log_fp"]) - np.asarray(calibration["log_fb"])
    else:
        return None, None, None, None

    if log_lr.ndim != 2 or log_lr.shape[1] != len(score_range):
        return None, None, None, None

    prior = float(calibration["prior"])

    # Interpolate each bootstrap curve at all variant scores: (n_boots, n_variants)
    log_lr_at_scores = np.array([
        np.interp(scores, score_range, log_lr[i],
                  left=np.nan, right=np.nan)
        for i in range(log_lr.shape[0])
    ])

    lr_5 = np.exp(np.nanpercentile(log_lr_at_scores, percentile, axis=0))
    lr_95 = np.exp(np.nanpercentile(log_lr_at_scores, benign_percentile, axis=0))

    def _posterior(lr_arr: np.ndarray) -> np.ndarray:
        out = np.full_like(lr_arr, np.nan)
        valid = (lr_arr > 0) & ~np.isnan(lr_arr)
        out[valid] = (prior * lr_arr[valid]) / (prior * lr_arr[valid] + (1.0 - prior))
        return out

    return lr_5, lr_95, _posterior(lr_5), _posterior(lr_95)


def _build_standard_table(scoreset, calibration: Dict, percentile: float = 5.0,
                           benign_percentile: Optional[float] = None) -> pd.DataFrame:
    """Assign every variant its evidence points from the global calibration.

    Always includes standard_points.  Also includes lr_plus_5th, lr_plus_95th,
    posterior_5th, posterior_95th when bootstrap LR+ curves are present in the
    calibration dict. Column names stay fixed regardless of `percentile`/
    `benign_percentile` (schema stability) -- only the values used to
    populate them change. ``benign_percentile`` defaults to
    ``100 - percentile`` when not given explicitly.
    """
    point_ranges = calibration["point_ranges"]
    ids = _get_variant_ids(scoreset)

    scores = np.array([float(scoreset.scores[i]) for i in range(len(scoreset.scores))])
    lr_5, lr_95, post_5, post_95 = _compute_bootstrap_lr_percentiles(
        scores, calibration, percentile, benign_percentile)
    has_percentiles = lr_5 is not None
    auth_labels = getattr(scoreset, "auth_labels", None)
    is_vus = _get_variant_is_vus(scoreset)
    is_snv = _get_variant_is_snv(scoreset)
    aa_fields = _get_variant_aa_fields(scoreset)

    rows = []
    for idx in range(len(scoreset.scores)):
        score = scores[idx]
        pts = _assign_points(score, point_ranges)

        # scoreset._sample_assignments is a multi-label one-hot row — a
        # variant can genuinely belong to more than one sample category at
        # once (e.g. both Synonymous and gnomAD/population, since consequence
        # type and population frequency are independent axes; see
        # Scoreset._init_matrices's independent `if` checks, not `elif`).
        # Keep every matching category, pipe-separated, rather than
        # collapsing to whichever comes first — that collapse silently
        # dropped real multi-label membership for anything downstream that
        # filtered on exact-equality "sample" strings.
        matched = [
            scoreset.sample_names[s_idx]
            for s_idx in range(len(scoreset.sample_names))
            if scoreset._sample_assignments[idx, s_idx]
        ]
        sample = "|".join(matched) if matched else "Unknown"

        row = {
            "variant_id": ids[idx] if idx < len(ids) else f"variant_{idx}",
            "score": score,
            "sample": sample,
            "standard_points": pts,
        }
        if has_percentiles:
            row["lr_plus_5th"] = float(lr_5[idx])
            row["lr_plus_95th"] = float(lr_95[idx])
            row["posterior_5th"] = float(post_5[idx])
            row["posterior_95th"] = float(post_95[idx])
        if auth_labels is not None:
            row["auth_label"] = auth_labels[idx]
        if is_vus is not None:
            row["is_vus"] = is_vus[idx]
        if is_snv is not None:
            row["is_snv"] = is_snv[idx]
        if aa_fields is not None:
            row.update(aa_fields[idx])

        rows.append(row)

    # Pure ClinVar-VUS variants outside every fitting-relevant sample group
    # (keep_mask=False) never appear in the loop above -- append their rows
    # here, scored the same way (standard_points from the same point_ranges;
    # bootstrap percentiles re-interpolated on just these extra scores), but
    # with no sample-group membership (sample="Unknown") and correctly no
    # oob_points downstream (they were never part of fitting/held out of a
    # bootstrap -- see compute_variant_table's OOB merge, a plain left-merge
    # that leaves these NaN, and analysis/plot_common.py::effective_points,
    # which already falls back to standard_points whenever oob is NaN).
    extra = _get_vus_only_extra_rows(scoreset)
    if extra:
        extra_scores = np.array([e["score"] for e in extra])
        ex_lr_5, ex_lr_95, ex_post_5, ex_post_95 = _compute_bootstrap_lr_percentiles(
            extra_scores, calibration, percentile, benign_percentile)
        for i, e in enumerate(extra):
            row = {
                "variant_id": e["variant_id"],
                "score": e["score"],
                "sample": "Unknown",
                "standard_points": _assign_points(e["score"], point_ranges),
            }
            if has_percentiles:
                row["lr_plus_5th"] = float(ex_lr_5[i]) if ex_lr_5 is not None else np.nan
                row["lr_plus_95th"] = float(ex_lr_95[i]) if ex_lr_95 is not None else np.nan
                row["posterior_5th"] = float(ex_post_5[i]) if ex_post_5 is not None else np.nan
                row["posterior_95th"] = float(ex_post_95[i]) if ex_post_95 is not None else np.nan
            if auth_labels is not None:
                row["auth_label"] = e["auth_label"]
            if is_vus is not None:
                row["is_vus"] = e["is_vus"]
            if is_snv is not None:
                row["is_snv"] = e["is_snv"]
            if aa_fields is not None:
                row["nucleotide_or_aa"] = e["nucleotide_or_aa"]
                row["aa_ref"] = e["aa_ref"]
                row["aa_pos"] = e["aa_pos"]
                row["aa_alt"] = e["aa_alt"]
            rows.append(row)

    return pd.DataFrame(rows)


def _classify_continuous_one(l5, l95, prior, targets, floor_at_neutral):
    """One variant's (lr_plus, posterior, classification, display_points) from
    its 5th/95th-percentile log(LR+) bounds -- factored out of
    _build_continuous_table's per-idx loop so the extra (VUS-only,
    keep_mask=False) rows it appends can reuse the identical decision logic
    rather than duplicating it. See _build_continuous_table's own docstring
    for the pathogenic/benign-candidate selection rule this implements.
    """
    from assay_calibration.fit_utils.bayesian_thresholds import (
        bayes_posterior_from_lr, continuous_classify, piecewise_display_points,
    )

    if np.isnan(l5) and np.isnan(l95):
        return np.nan, np.nan, "Unknown", np.nan

    cat_p5 = (str(continuous_classify(np.exp(l5), prior, targets, floor_at_neutral))
              if not np.isnan(l5) else "Unknown")
    cat_p95 = (str(continuous_classify(np.exp(l95), prior, targets, floor_at_neutral))
               if not np.isnan(l95) else "Unknown")
    path_ok = cat_p5 in ("LP", "P")
    ben_ok = cat_p95 in ("LB", "B")

    if path_ok and ben_ok:
        use_path = abs(l5) >= abs(l95)
    elif path_ok:
        use_path = True
    elif ben_ok:
        use_path = False
    else:
        use_path = None

    if use_path is None:
        l_rep = l5 if (not np.isnan(l5) and (np.isnan(l95) or abs(l5) <= abs(l95))) else l95
        lr_i = float(np.exp(l_rep))
        post_i = float(bayes_posterior_from_lr(lr_i, prior))
        return lr_i, post_i, "VUS", 0.0

    l_use = l5 if use_path else l95
    lr_i = float(np.exp(l_use))
    post_i = float(bayes_posterior_from_lr(lr_i, prior))
    label = cat_p5 if use_path else cat_p95
    display_pts = float(piecewise_display_points(l_use, prior, targets, floor_at_neutral))
    return lr_i, post_i, label, display_pts


def _build_continuous_table(scoreset, calibration: Dict, config) -> pd.DataFrame:
    """ACMG-Bayes per-variant table.

    Never classifies from a single median/point-estimate log(LR+) curve.
    Instead, mirrors point_ranges.py's discrete OOB evidence-assignment rule
    (5th percentile of the bootstrap log(LR+) distribution for
    pathogenic-direction evidence, 95th percentile for benign-direction
    evidence -- the same conservative bounds _build_standard_table already
    uses for Tavtigian's standard_points via
    _compute_bootstrap_lr_percentiles), generalized to the continuous
    posterior/classification/display-point representation:

      - Evaluate the pathogenic candidate from the 5th-percentile bound and
        the benign candidate from the 95th-percentile bound independently.
      - A candidate is only used if it actually leaves VUS under
        continuous_classify at the gene's prior/targets -- NOT if its
        log(LR+) is merely positive/negative. A hardcoded log(LR+)=0 cutoff
        is only correct when the real LP/LB decision boundary happens to
        sit at neutral evidence; it doesn't in general (e.g. prior below the
        LB target with floor_at_neutral off, where the real LB boundary
        sits at a *positive* log(LR+) -- exactly the miscalibration this
        method exists to fix).
      - If both candidates leave VUS (shouldn't normally happen), keep
        whichever is farther from neutral. If neither does, the variant is
        VUS by construction, and the milder (closer-to-neutral) bound is
        reported as lr_plus/posterior for display only.
    """
    import sys, os
    _SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    from assay_calibration.fit_utils.bayesian_thresholds import (
        bayes_posterior_from_lr, continuous_classify, piecewise_display_points,
    )

    prior = float(calibration["prior"])
    targets = getattr(config, "acmg_bayes_targets", None)
    floor_at_neutral = getattr(config, "acmg_bayes_floor_at_neutral", False)
    percentile = getattr(config, "pathogenic_percentile", 5.0)
    benign_percentile = getattr(config, "benign_percentile", None)
    ids = _get_variant_ids(scoreset)
    auth_labels = getattr(scoreset, "auth_labels", None)
    is_vus = _get_variant_is_vus(scoreset)
    is_snv = _get_variant_is_snv(scoreset)
    aa_fields = _get_variant_aa_fields(scoreset)

    scores = np.array([float(scoreset.scores[i]) for i in range(len(scoreset.scores))])
    lr_5, lr_95, _, _ = _compute_bootstrap_lr_percentiles(scores, calibration, percentile, benign_percentile)
    if lr_5 is None:
        raise ValueError(
            "_build_continuous_table (ACMG-Bayes) requires bootstrap log(LR+) curves "
            "(calibration['log_lr_plus'] as a per-bootstrap matrix, or log_fp/log_fb) "
            "to compute the 5th/95th percentile bounds -- ACMG-Bayes never classifies "
            "from a single median/point-estimate curve."
        )
    log_lr_5 = np.log(lr_5)
    log_lr_95 = np.log(lr_95)

    rows = []
    for idx in range(len(scoreset.scores)):
        score = scores[idx]
        lr_i, post_i, label, display_pts = _classify_continuous_one(
            log_lr_5[idx], log_lr_95[idx], prior, targets, floor_at_neutral)

        # See _build_standard_table's identical logic — sample_assignments is
        # multi-label; keep every matching category, pipe-separated.
        matched = [
            scoreset.sample_names[s_idx]
            for s_idx in range(len(scoreset.sample_names))
            if scoreset._sample_assignments[idx, s_idx]
        ]
        sample = "|".join(matched) if matched else "Unknown"

        row = {
            "variant_id": ids[idx] if idx < len(ids) else f"variant_{idx}",
            "score": score,
            "sample": sample,
            "lr_plus": lr_i,
            "posterior": post_i,
            "classification": label,
            "display_points": display_pts,
        }
        if auth_labels is not None:
            row["auth_label"] = auth_labels[idx]
        if is_vus is not None:
            row["is_vus"] = is_vus[idx]
        if is_snv is not None:
            row["is_snv"] = is_snv[idx]
        if aa_fields is not None:
            row.update(aa_fields[idx])
        rows.append(row)

    # Pure ClinVar-VUS variants outside every fitting-relevant sample group
    # (keep_mask=False) -- see _build_standard_table's identical block for
    # the full rationale. Classified via the same _classify_continuous_one
    # helper on bootstrap percentiles re-interpolated for just these scores;
    # sample="Unknown" (no group membership).
    extra = _get_vus_only_extra_rows(scoreset)
    if extra:
        extra_scores = np.array([e["score"] for e in extra])
        ex_lr_5, ex_lr_95, _, _ = _compute_bootstrap_lr_percentiles(
            extra_scores, calibration, percentile, benign_percentile)
        ex_log_lr_5 = np.log(ex_lr_5) if ex_lr_5 is not None else np.full(len(extra), np.nan)
        ex_log_lr_95 = np.log(ex_lr_95) if ex_lr_95 is not None else np.full(len(extra), np.nan)
        for i, e in enumerate(extra):
            lr_i, post_i, label, display_pts = _classify_continuous_one(
                ex_log_lr_5[i], ex_log_lr_95[i], prior, targets, floor_at_neutral)
            row = {
                "variant_id": e["variant_id"],
                "score": e["score"],
                "sample": "Unknown",
                "lr_plus": lr_i,
                "posterior": post_i,
                "classification": label,
                "display_points": display_pts,
            }
            if auth_labels is not None:
                row["auth_label"] = e["auth_label"]
            if is_vus is not None:
                row["is_vus"] = e["is_vus"]
            if is_snv is not None:
                row["is_snv"] = e["is_snv"]
            if aa_fields is not None:
                row["nucleotide_or_aa"] = e["nucleotide_or_aa"]
                row["aa_ref"] = e["aa_ref"]
                row["aa_pos"] = e["aa_pos"]
                row["aa_alt"] = e["aa_alt"]
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# OOB helpers
# ---------------------------------------------------------------------------

def _build_oob_mapping(
    scoreset,
    dataset_splits: Dict[int, Dict],
    valid_bootstrap_seeds: np.ndarray,
) -> Dict[int, List[int]]:
    """
    Map each variant index -> list of *filtered* bootstrap indices where it
    appeared in the validation (OOB) set.

    Parameters
    ----------
    valid_bootstrap_seeds : 1-D int array
        ``valid_bootstrap_seeds[filtered_idx]`` is the original bootstrap seed.
    """
    seed_to_filtered = {int(s): i for i, s in enumerate(valid_bootstrap_seeds)}

    n_seeds_in_splits = len(dataset_splits)
    n_seeds_valid = len(seed_to_filtered)
    print(f"  OOB debug: {n_seeds_in_splits} seeds in splits, "
          f"{n_seeds_valid} valid after prior filtering")

    # Check whether any split has val_variant_indices (new index-based format)
    has_indices = any(
        split.get("val_variant_indices") is not None
        for split in dataset_splits.values()
    )

    variant_oob: Dict[int, List[int]] = defaultdict(list)

    if has_indices:
        n_index_total = 0
        n_index_hits = 0
        for seed, split in dataset_splits.items():
            if seed not in seed_to_filtered:
                continue
            fidx = seed_to_filtered[seed]
            indices = split.get("val_variant_indices")
            if indices is None:
                continue
            for vidx in indices:
                variant_oob[int(vidx)].append(fidx)
                n_index_hits += 1
            n_index_total += len(indices)
        print(f"  OOB debug: index-based matching — {n_index_hits}/{n_index_total} "
              f"observations mapped ({len(scoreset.scores)} total variants)")
    else:
        # Legacy fallback: match by (score, class_idx) float equality
        score_cls_map = defaultdict(list)
        for idx in range(len(scoreset.scores)):
            sc = scoreset.scores[idx]
            for cls in np.where(scoreset._sample_assignments[idx])[0]:
                score_cls_map[(sc, int(cls))].append(idx)

        print(f"  OOB debug: score-based matching (legacy) — "
              f"{len(score_cls_map)} unique (score, class) keys, "
              f"{len(scoreset.scores)} total variants")

        n_val_obs_total = 0
        n_val_obs_matched = 0

        for seed, split in dataset_splits.items():
            if seed not in seed_to_filtered:
                continue
            fidx = seed_to_filtered[seed]
            for obs, assign in zip(split["val_observations"],
                                   split["val_sample_assignments"]):
                n_val_obs_total += 1
                cls = int(np.where(assign)[0][0])
                matches = score_cls_map.get((obs, cls), [])
                if matches:
                    n_val_obs_matched += 1
                for vidx in matches:
                    variant_oob[vidx].append(fidx)

        print(f"  OOB debug: {n_val_obs_matched}/{n_val_obs_total} val observations "
              f"matched a scoreset variant")

        if n_val_obs_total > 0 and n_val_obs_matched == 0:
            sample_split_key = None
            for seed in list(dataset_splits.keys())[:1]:
                if seed in seed_to_filtered:
                    obs = dataset_splits[seed]["val_observations"]
                    assign = dataset_splits[seed]["val_sample_assignments"]
                    if len(obs) > 0:
                        sample_split_key = (obs[0], int(np.where(assign[0])[0][0]))
                        break
            sample_scoreset_key = next(iter(score_cls_map.keys()), None)
            print(f"  OOB debug: KEY MISMATCH DETECTED")
            print(f"    Sample split key:    {sample_split_key} "
                  f"(types: {type(sample_split_key[0]).__name__ if sample_split_key else '?'}, "
                  f"{type(sample_split_key[1]).__name__ if sample_split_key else '?'})")
            print(f"    Sample scoreset key: {sample_scoreset_key} "
                  f"(types: {type(sample_scoreset_key[0]).__name__ if sample_scoreset_key else '?'}, "
                  f"{type(sample_scoreset_key[1]).__name__ if sample_scoreset_key else '?'})")

    # Report OOB coverage stats
    if variant_oob:
        oob_counts = [len(v) for v in variant_oob.values()]
        print(f"  OOB debug: {len(variant_oob)} variants with OOB data, "
              f"median {np.median(oob_counts):.0f} boots/variant "
              f"(range {min(oob_counts)}-{max(oob_counts)})")

    return dict(variant_oob)


def _process_variant_oob(
    variant_idx: int,
    oob_indices: List[int],
    score: float,
    priors: np.ndarray,
    log_fp: np.ndarray,
    log_fb: np.ndarray,
    score_range: np.ndarray,
    point_values: List[int],
    flipped: bool,
    liberal: bool,
    min_samples: int = 1,
    acmg_mapping_method: str = "tavtigian",
    percentile: float = 5.0,
    benign_percentile: Optional[float] = None,
    postprocess: bool = True,
    force_no_postprocess: bool = False,
    benign_center: Optional[float] = None,
) -> Tuple[int, Optional[Dict]]:
    """
    Compute OOB evidence for one variant using FULL in-bag processing logic.

    This matches _process_single_variant_oob_full in point_ranges.py:
    1. Subset to OOB bootstraps
    2. Filter invalid priors
    3. Compute OOB median prior
    4. Compute OOB LR+ and filter NaN columns
    5. calculate_score_ranges (median prior, 5th/95th percentile LR+)
    6. Postprocess -- same branch the global in-bag fit took for this dataset
       (see visualize.py's process_component_fits): bidirectional cleanup
       (clean_benign_fragments_no_extend + clean_bidirectional_pathogenic_evidence)
       when force_no_postprocess is True, else the plain
       enforce_monotonicity_point_ranges -> extend_points_to_xlims ->
       enforce_monotonicity_point_ranges sequence, else no postprocessing at all.
       force_no_postprocess/benign_center are dataset-level values computed once
       from the FULL bootstrap ensemble in visualize.py -- reused verbatim here,
       never recomputed from this variant's (much smaller, noisier) OOB subset.
    9. Flatten and assign points

    Returns (variant_idx, result_dict) or (variant_idx, {"_fail": reason}).
    """

    if len(oob_indices) < min_samples:
        return variant_idx, {"_fail": f"too_few_oob ({len(oob_indices)}<{min_samples})"}

    oob_priors = priors[oob_indices]
    oob_fp = log_fp[oob_indices]
    oob_fb = log_fb[oob_indices]

    # Filter invalid priors
    valid = ~np.isnan(oob_priors) & (oob_priors > 0) & (oob_priors < 1)
    n_valid = int(valid.sum())
    oob_priors, oob_fp, oob_fb = oob_priors[valid], oob_fp[valid], oob_fb[valid]

    if len(oob_priors) < min_samples:
        return variant_idx, {"_fail": f"too_few_valid_priors ({n_valid}<{min_samples})"}

    prior = float(np.nanmedian(oob_priors))
    if prior <= 0 or prior >= 1:
        return variant_idx, {"_fail": f"invalid_median_prior ({prior})"}

    # Compute OOB LR+
    lr_plus = oob_fp - oob_fb
    nan_counts = np.isnan(lr_plus).sum(0)
    subset = nan_counts < lr_plus.shape[0]
    if not np.any(subset):
        return variant_idx, {"_fail": f"all_lr_nan (shape={lr_plus.shape})"}

    vsr = score_range[subset]
    vlr = lr_plus[:, subset]
    _benign_percentile = benign_percentile if benign_percentile is not None else 100 - percentile

    try:
        # Step 5: calculate_score_ranges (same as in-bag median_prior branch)
        pr_p, pr_b, C = calculate_score_ranges(
            np.nanpercentile(vlr, percentile, axis=0),
            np.nanpercentile(vlr, _benign_percentile, axis=0),
            prior, vsr, point_values, acmg_mapping_method=acmg_mapping_method,
        )
        pr = {**pr_p, **pr_b}

        # Check if prior is valid (matches in-bag check)
        if prior <= 0 or prior >= 1:
            for point in pr:
                pr[point] = []

        if force_no_postprocess:
            # Same branch as the in-bag fit for this dataset: auto-detected
            # bidirectional assay, postprocess_point_ranges is force-overridden
            # regardless of its raw value (matches visualize.py's
            # `elif force_no_postprocess:` branch).
            clean_benign_fragments_no_extend(pr, point_values, vsr, liberal=liberal)
            clean_bidirectional_pathogenic_evidence(
                pr, point_values, vsr, benign_center, liberal=liberal, inf=True,
            )
        elif postprocess:
            # Step 6: enforce monotonicity (first pass)
            enforce_monotonicity_point_ranges(pr, point_values, vsr, flipped, liberal)

            # Step 7: extend to xlims
            extend_points_to_xlims(pr, point_values, vsr, flipped, inf=True)

            # Step 8: enforce monotonicity (second pass)
            enforce_monotonicity_point_ranges(pr, point_values, vsr, flipped, liberal)
        # else: no postprocessing at all (matches visualize.py's --no-postprocess branch).

        # Step 9: flatten and assign
        pts = _assign_points(score, pr)
    except Exception as e:
        return variant_idx, {"_fail": f"exception: {type(e).__name__}: {str(e)[:200]}"}

    return variant_idx, {
        "points": pts,
        "n_oob": len(oob_indices),
        "n_oob_valid": n_valid,
        "oob_prior": prior,
        "score": float(score),
    }


def _compute_oob_evidence(
    scoreset,
    calibration: Dict,
    dataset_splits: Dict[int, Dict],
    config: PipelineConfig,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Dict]:
    """Run OOB evidence computation for all variants that have enough OOB samples."""

    def log(msg):
        (logger.info if logger else print)(msg)

    priors = np.asarray(calibration["priors"])
    log_fp = np.asarray(calibration["log_fp"])
    log_fb = np.asarray(calibration["log_fb"])
    score_range = np.asarray(calibration["score_range"])
    flipped = calibration.get("scoreset_flipped", False)
    # Dataset-level bidirectional-postprocessing decision from the global
    # in-bag fit (visualize.py's process_component_fits). Deliberately absent
    # from on-disk *_calibration.json (not in calibration_compact's whitelist
    # in pipeline/utils.py:save_results), so any calibration dict reconstructed
    # purely from disk JSON falls back to "no override" here.
    force_no_postprocess = calibration.get("force_no_postprocess", False)
    benign_center = calibration.get("benign_center", None)
    valid_seeds = np.asarray(calibration["valid_bootstrap_seeds"])

    oob_map = _build_oob_mapping(scoreset, dataset_splits, valid_seeds)
    log(f"  OOB mapping built: {len(oob_map)} variants have OOB data")

    liberal = config.liberal_monotonicity
    postprocess = getattr(config, "postprocess_point_ranges", True)
    n_cores = config.n_jobs if config.n_jobs > 0 else (os.cpu_count() or 1)

    oob_acmg_mapping_method = getattr(config, "acmg_mapping_method", "tavtigian")
    raw = Parallel(n_jobs=min(len(oob_map), n_cores), verbose=5)(
        delayed(_process_variant_oob)(
            vidx, oob_idx, scoreset.scores[vidx],
            priors, log_fp, log_fb, score_range,
            config.point_values, flipped, liberal,
            min_samples=config.oob_min_samples,
            acmg_mapping_method=oob_acmg_mapping_method,
            percentile=getattr(config, "pathogenic_percentile", 5.0),
            benign_percentile=getattr(config, "benign_percentile", None),
            postprocess=postprocess,
            force_no_postprocess=force_no_postprocess,
            benign_center=benign_center,
        )
        for vidx, oob_idx in oob_map.items()
    )

    # Map filtered index -> variant ID, separating successes from failures
    ids = _get_variant_ids(scoreset)
    out = {}
    fail_reasons = defaultdict(int)

    for vidx, result in raw:
        if result is None:
            fail_reasons["returned_None"] += 1
        elif "_fail" in result:
            reason = result["_fail"].split(" (")[0].split(":")[0]
            fail_reasons[reason] += 1
        else:
            vid = ids[vidx] if vidx < len(ids) else f"variant_{vidx}"
            out[vid] = result

    log(f"  OOB evidence computed for {len(out)}/{len(oob_map)} variants")
    if fail_reasons:
        log(f"  OOB failure breakdown:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            log(f"    {reason}: {count}")

        examples = [(vidx, r) for vidx, r in raw if r is not None and "_fail" in r][:5]
        if examples:
            log(f"  Example failures:")
            for vidx, r in examples:
                n_oob = len(oob_map.get(vidx, []))
                log(f"    variant {vidx} (score={scoreset.scores[vidx]:.4f}, "
                    f"n_oob_boots={n_oob}): {r['_fail']}")

    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_variant_table(
    scoreset,
    calibration: Dict,
    config: PipelineConfig,
    dataset_splits: Optional[Dict[int, Dict]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Build a per-variant evidence table.

    Always includes ``standard_points`` (from global calibration).
    When ``config.compute_oob`` is True and *dataset_splits* is provided,
    also includes ``oob_points``, ``oob_n_boots``, and ``oob_prior``.

    Returns
    -------
    pd.DataFrame
    """

    def log(msg):
        (logger.info if logger else print)(msg)

    # --- standard assignment (always) ---
    acmg_mapping_method = calibration.get(
        "acmg_mapping_method",
        getattr(config, "acmg_mapping_method", "tavtigian"),
    )
    if acmg_mapping_method == "acmg_bayes":
        df = _build_continuous_table(scoreset, calibration, config)
    else:
        df = _build_standard_table(
            scoreset, calibration,
            getattr(config, "pathogenic_percentile", 5.0),
            getattr(config, "benign_percentile", None),
        )
    log(f"  Standard evidence assigned to {len(df)} variants "
        f"(acmg_mapping_method={acmg_mapping_method})")

    # --- OOB assignment (optional) ---
    if config.compute_oob and acmg_mapping_method == "acmg_bayes":
        log("  Note: OOB evidence not supported for acmg_bayes acmg_mapping_method; skipping")
    elif config.compute_oob:
        if dataset_splits is None:
            log("  WARNING: OOB requested but no dataset_splits provided; skipping")
        elif "valid_bootstrap_seeds" not in calibration:
            log("  WARNING: OOB requested but calibration missing valid_bootstrap_seeds; skipping")
        else:
            log("  Computing OOB evidence...")
            oob = _compute_oob_evidence(
                scoreset, calibration, dataset_splits, config, logger,
            )
            if oob:
                oob_df = pd.DataFrame([
                    {"variant_id": vid, "oob_points": r["points"],
                     "oob_n_boots": r["n_oob_valid"], "oob_prior": r["oob_prior"]}
                    for vid, r in oob.items()
                ])
                df = df.merge(oob_df, on="variant_id", how="left")
            else:
                df["oob_points"] = np.nan
                df["oob_n_boots"] = 0
                df["oob_prior"] = np.nan

            n_oob = df["oob_points"].notna().sum()
            log(f"  OOB evidence assigned to {n_oob}/{len(df)} variants")

            # Quick concordance summary
            both = df.dropna(subset=["oob_points"])
            if len(both) > 0:
                agree = (both["standard_points"] == both["oob_points"]).sum()
                log(f"  Standard/OOB agreement: {agree}/{len(both)} "
                    f"({100*agree/len(both):.1f}%)")

    return df
