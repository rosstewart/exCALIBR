"""
Variant-assay / variant-aggregate / predictor-only evidence tables.

Builds the CSVs that `test/create_all_variant_evidence_csv.ipynb`,
`test/make_variant_aggregate_evidence_table.ipynb`, and
`test/merge_functional_predictor_evidence.ipynb` used to hand-roll from ad
hoc pickle/glob-loaded external files, reusing the same pipeline machinery
`analyze_pipeline_output.py`'s own manuscript-summary section already uses
(`src.assay_calibration.pipeline.variant_evidence.get_all_variant_groups`)
instead of reimplementing filtering/scoring from scratch.

Every table here is in-bag only (`standard_points`) -- there is deliberately
no OOB, and no OOB-with-in-bag-fallback, anywhere in this module. Most of
the variants these tables cover were never in any bootstrap's train/
validation split (they're outside every calibration-relevant sample group /
`keep_mask`), so no OOB value exists for them at all; using OOB when
available for the small subset that *does* have one would make otherwise-
identical rows silently switch scoring conventions depending on keep_mask
membership. See `get_all_variant_groups`'s own docstring for the same
reasoning.

Alongside the integer `points`/`evidence` columns, the variant-assay and
variant-aggregate tables carry a continuous `lr_plus`/`posterior` pair plus
the two raw directional bounds (`lr_plus_p5`/`lr_plus_p95`) they were combined
from. `lr_plus` is always reconciled against the row's own postprocessed
points, so the two scales can never disagree -- see
`_reconcile_log_lr_with_points`.

Grain, and two schema changes worth knowing about:

- The variant-assay table is **one row per input dataframe row**, not per
  measured variant, so every nucleotide route to an aa-level measurement gets
  its own coordinate-joinable row (`build_variant_assay_table`). The
  variant-aggregate table is correspondingly **one row per nucleotide
  coordinate** for aa- and nt-level measurements alike, with
  `nucleotide_or_aa` kept as evidence provenance.
- `is_benign`/`is_pathogenic`/`is_vus`/`is_gnomad` are now each row's **own**
  ClinVar/gnomAD status. The OR-across-the-group values -- which is what
  these columns used to hold, and what the calibration actually saw -- are
  preserved as `is_*_group`.
- There is a single `points`/`evidence` pair; the old
  `points_clinvar_2018`/`evidence_clinvar_2018` twin columns are gone, with
  `clinvar_release` (`"clinvar_2018"`/`"clinvar_2025"`) recording which
  ClinVar release the row's calibration was fitted against.

`build_dataframe_with_points` additionally returns the input dataframe itself
with a narrow set of evidence columns appended, NaN wherever a row never
reached a Scoreset, annotated with `excalibr_filter_reason`. Every added
column is `excalibr_`-prefixed there so it cannot be mistaken for one of the
input's own: `excalibr_clinvar_release` (bare `"2018"`/`"2025"` -- the column
name already says which release it is), `excalibr_points`,
`excalibr_acmg_evidence_code`, then `excalibr_lr_plus` ->  `excalibr_prior` ->
`excalibr_posterior`. It carries no `dataset` column (that would duplicate the
input's own `Dataset`) and none of the per-assay detail -- raw p5/p95 bounds,
group score/size, `is_*_group` promotion flags -- which stays in the
variant-assay table, joinable on `_input_row_id`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.discovery import _filter_dataset_df, _resolve_n_jobs, load_master_df, resolve_component
from analysis.multi_scoreset import _merge_points

# int evidence points -> ACMG PS3/BS3 evidence-strength code, hardcoded
# identically in make_variant_aggregate_evidence_table.ipynb /
# create_all_variant_evidence_csv.ipynb -- doesn't exist anywhere else in
# analysis/ or src/ (analysis/acmg_evidence_codes.py is a different mapping,
# ACMG code -> strength label, not points -> code).
POINTS_TO_EVIDENCE_CODE = {
    1: "PS3_Supporting", 2: "PS3_Moderate", 3: "PS3_Moderate+", 4: "PS3_Strong",
    5: "PS3_Strong", 6: "PS3_Strong", 7: "PS3_Strong", 8: "PS3_Very_Strong",
    0: "NO_EVIDENCE",
    -1: "BS3_Supporting", -2: "BS3_Moderate", -3: "BS3_Moderate+", -4: "BS3_Strong",
    -5: "BS3_Strong", -6: "BS3_Strong", -7: "BS3_Strong", -8: "BS3_Very_Strong",
}

# (The `GENES_2018` set that used to live here is gone with the
# points_clinvar_2018 column pair: there is one points column now, and
# `clinvar_release` records which release each row's calibration used. The
# same constant is independently defined by every module that still needs it
# -- analysis/robustness.py, run_igvf_batch.py, etc.)

# Integer point tiers point_ranges is built over, in ascending magnitude --
# same list every calculate_score_ranges/thresholds_from_prior caller uses
# (see analysis/discovery.py::recompute_points_with_prior_overrides's
# POINT_VALUES and fit_utils/point_ranges.py's own default).
POINT_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]

# Nudge (in log(LR+) space) keeping a clamped value strictly inside the band
# its points imply, rather than parked exactly on a tier threshold -- see
# _reconcile_log_lr_with_points. Six-plus orders of magnitude below the
# narrowest real tier width (~0.4 in log space at any prior/mapping method),
# so it never perturbs the reported value meaningfully.
_LOG_LR_BAND_EPS = 1e-6

# acmg_mapping_method values thresholds_from_prior actually understands. The
# `method` token discovered from output filenames is a pipeline method slot
# ("default" when the filename carries no method token, see
# analysis/discovery.py::_parse_output_stem), not necessarily a mapping
# method, so anything unrecognized falls back to PipelineConfig's own default
# ("tavtigian", src/assay_calibration/pipeline/config.py).
_MAPPING_METHODS = {"tavtigian", "piecewise", "strict_additive"}

# Synthetic column carrying each input row's master-dataframe index through
# Scoreset construction, so a variant GROUP can name its member INPUT ROWS.
# Safe to inject: Variant._init_variant_info copies every input column onto
# the Variant with no whitelist, no master-TSV column starts with "_", and
# this name matches neither the `_fcc_<int>_` special shape nor any attribute
# the class assigns after that copy.
INPUT_ROW_ID = "_input_row_id"


def _resolve_mapping_method(method: Optional[str]) -> str:
    """`method` slot -> the acmg_mapping_method thresholds_from_prior wants."""
    return method if method in _MAPPING_METHODS else "tavtigian"


def _combine_directional_log_lr(l_p5: np.ndarray, l_p95: np.ndarray) -> np.ndarray:
    """Collapse the two directional log(LR+) candidates into one signed value.

    Python mirror of the front end's own `combineDirectionalLogLr`
    (chartUtils.ts) / `_classify_continuous_one`'s `abs(l5) >= abs(l95)`
    selection: floor the pathogenic candidate (5th-percentile bound) at
    neutral, cap the benign candidate (95th-percentile bound) at neutral, then
    keep whichever is farther from neutral. NaN wherever both candidates are
    NaN.

    Deliberately done in LOG space even though the tables report LR+: after
    the floor/cap, `LR_ben <= 1 <= LR_path` always holds, so comparing raw
    LR-space magnitudes could never select the benign candidate. "Farther
    from neutral" is |log(LR+)| -- multiplicative distance from 1.

    Note the magnitude comparison never actually decides anything while both
    bounds come from the same bootstrap distribution: `p5 <= p95` by
    definition of a percentile, so `max(0, p5)` and `min(0, p95)` can never
    both be non-zero (p5 > 0 forces p95 > 0, hence a neutral benign
    candidate; p95 < 0 forces p5 < 0, hence a neutral pathogenic one; and
    otherwise p5 <= 0 <= p95 collapses both to 0, i.e. LR+ = 1, "the interval
    spans neutral"). Verified on every curve file in the current output tree:
    0 of 198,386 comparable grid points have p5 > p95. The general form is
    kept anyway because it is the contract the front end implements, and
    because the ONE configuration that would break the invariant is reachable
    here: under `pathomechanism_method`, `_lr_posterior_columns` takes the
    pathogenic candidate from `log_lr_pathogenic_p5`, a percentile of a
    *different* (rho, LR_D) distribution than `log_lr_plus_p95`, so the two
    bounds are no longer ordered and the tie-break becomes live.

    Deliberate deviation from the TS mirror: "missing" is tested with
    `isnan`, not `isfinite`, so an INFINITE bound counts as maximal evidence
    rather than as no evidence. A saved percentile curve can genuinely hold
    +-inf where the opposing density estimate went to zero, and discarding
    that would silently turn the strongest available evidence into the
    weakest. It only changes the outcome when the infinity points in its own
    evidential direction (`p5=+inf`, `p95=-inf`), since the floor/cap flattens
    the other two cases to neutral anyway.
    """
    l_p5 = np.asarray(l_p5, dtype=float)
    l_p95 = np.asarray(l_p95, dtype=float)

    floored_path = np.where(~np.isnan(l_p5), np.maximum(0.0, l_p5), np.nan)
    floored_ben = np.where(~np.isnan(l_p95), np.minimum(0.0, l_p95), np.nan)

    # -inf magnitude for a missing candidate so the other one always wins;
    # both missing -> the np.where below restores NaN.
    mag_path = np.where(~np.isnan(floored_path), np.abs(floored_path), -np.inf)
    mag_ben = np.where(~np.isnan(floored_ben), np.abs(floored_ben), -np.inf)

    combined = np.where(mag_path >= mag_ben, floored_path, floored_ben)
    return np.where(np.isneginf(mag_path) & np.isneginf(mag_ben), np.nan, combined)


def _log_lr_band_for_points(points: float, log_lr_p: np.ndarray,
                            log_lr_b: np.ndarray) -> tuple:
    """The `(lo, hi)` log(LR+) interval a variant assigned `points` falls in.

    `log_lr_p[i]` / `log_lr_b[i]` are the log(LR+) thresholds for
    +POINT_VALUES[i] / -POINT_VALUES[i] from `thresholds_from_prior`, and the
    tiers themselves are `fit_utils/fit.py::assign_p`/`assign_b`'s:

      - `assign_p`: `lr >= tau[i] and lr < tau[i+1]`
      - `assign_b`: `lr <= tau[i] and lr > tau[i+1]`
      - neither fires -> 0, so the neutral tier spans (log_lr_b[0], log_lr_p[0])

    Which end each direction treats as inclusive is left out deliberately:
    `_reconcile_log_lr_with_points` keeps its result strictly inside (lo, hi),
    which is inside the tier either way.

    Returns (-inf, +inf) -- no clamping -- for points that aren't one of the
    +-POINT_VALUES tiers, or are NaN.
    """
    if points is None or (isinstance(points, float) and np.isnan(points)):
        return -np.inf, np.inf
    pts = int(points)
    if pts == 0:
        return log_lr_b[0], log_lr_p[0]
    mag = abs(pts)
    if mag not in POINT_VALUES:
        return -np.inf, np.inf
    i = POINT_VALUES.index(mag)
    last = i + 1 >= len(POINT_VALUES)
    if pts > 0:
        return log_lr_p[i], (np.inf if last else log_lr_p[i + 1])
    return (-np.inf if last else log_lr_b[i + 1]), log_lr_b[i]


def _reconcile_log_lr_with_points(log_lr: np.ndarray, points: np.ndarray,
                                  log_lr_p: np.ndarray,
                                  log_lr_b: np.ndarray) -> np.ndarray:
    """Clamp each combined log(LR+) into the band its assigned points imply.

    `points` come from `point_ranges`, which the pipeline builds with
    monotonicity enforcement/relaxation (see fit_utils/point_ranges.py) -- so
    a variant's raw interpolated log(LR+) can genuinely disagree with its
    assigned tier (points +1 while the raw value sits below the +1
    threshold, or vice versa). The tables report the POSTPROCESSED evidence,
    so the points are the authority and the reported LR+ is moved to the
    nearest value consistent with them:

      - outside the band -> that side's threshold, moved `_LOG_LR_BAND_EPS`
        strictly inside the band
      - NaN (the saved percentile curve has no value at this score) with
        nonzero points -> likewise, at the tier's weakest-evidence threshold
      - NaN with 0 points -> log(LR+) = 0, i.e. LR+ = 1: the points say
        "no evidence", and LR+ = 1 says exactly that

    The result is always kept strictly inside (lo, hi) rather than parked on
    a threshold, in both directions. One side of each tier is inclusive
    (`assign_p` takes `lr >= tau`, `assign_b` takes `lr <= tau`), so landing
    exactly on it would be legal in exact arithmetic -- but the value is
    stored as `exp(log_lr)`, and that round trip can move it a ulp onto the
    neighbouring tier's side of the threshold. `_LOG_LR_BAND_EPS` is ~6
    orders of magnitude below the narrowest real tier width, so nudging
    inward is free.
    """
    log_lr = np.asarray(log_lr, dtype=float)
    out = log_lr.copy()

    for idx in range(len(out)):
        pts = points[idx] if idx < len(points) else np.nan
        lo, hi = _log_lr_band_for_points(pts, log_lr_p, log_lr_b)
        lo_target = lo + _LOG_LR_BAND_EPS if np.isfinite(lo) else lo
        hi_target = hi - _LOG_LR_BAND_EPS if np.isfinite(hi) else hi

        val = out[idx]
        if np.isnan(val):
            # Points are the authority: anchor just inside the tier's
            # weakest-evidence threshold, which is `lo` for a pathogenic tier
            # and `hi` for a benign one -- keyed off the sign of the POINTS,
            # not of the bound, since a tier's band can straddle log(LR+)=0
            # under prior-adaptive thresholds (piecewise at a prior below the
            # LB target puts both boundaries on the same side of neutral).
            if not np.isnan(pts) and pts > 0 and np.isfinite(lo):
                out[idx] = lo_target
                continue
            if not np.isnan(pts) and pts < 0 and np.isfinite(hi):
                out[idx] = hi_target
                continue
            # Neutral tier (or no points at all): "no evidence" is LR+ = 1,
            # so report that rather than leaving the row empty. Falls through
            # to the clamp below, which normally leaves 0.0 untouched but
            # keeps the tier invariant intact for a prior-adaptive neutral
            # band that doesn't contain log(LR+)=0.
            val = 0.0
            out[idx] = val
        if val <= lo:
            out[idx] = lo_target
        elif val >= hi:
            out[idx] = hi_target

    return out


def _posterior_from_log_lr(log_lr: np.ndarray, prior: float) -> np.ndarray:
    """Posterior from log(LR+), overflow-safe.

    Algebraically identical to
    `fit_utils/bayesian_thresholds.py::bayes_posterior_from_lr(exp(log_lr), prior)`
    -- divide that function's `lr*p / ((lr-1)*p + 1)` through by `lr` -- but
    written so a log(LR+) outside the double range saturates to 1.0/0.0
    instead of evaluating inf/inf = NaN.
    """
    log_lr = np.asarray(log_lr, dtype=float)
    p = float(prior)
    with np.errstate(over="ignore"):
        odds_ratio = np.exp(-log_lr)          # +inf as log_lr -> -inf
        return p / (p + (1.0 - p) * odds_ratio)


def _lr_posterior_columns(scores: np.ndarray, points: np.ndarray, lr_path: Path,
                          prior: float, mapping_method: str) -> Optional[Dict[str, np.ndarray]]:
    """(lr_plus_p5, lr_plus_p95, lr_plus, posterior) arrays for one dataset.

    The 5th/95th-percentile log(LR+) curves are read from the sibling
    `*_lr_values.json.gz` (written by
    src/assay_calibration/pipeline/utils.py::save_results) and interpolated at
    each variant's score, the same way
    analysis/discovery.py::recompute_points_with_prior_overrides already
    consumes them. They CANNOT come from the calibration JSON: its compact
    on-disk form carries neither `score_range` nor `log_lr_plus`, so
    variant_evidence.py::_compute_bootstrap_lr_percentiles returns None for
    any disk-loaded calibration.

    Note this interpolates the already-reduced percentile curves, whereas the
    pipeline's in-memory path percentiles the per-bootstrap curves after
    interpolating each one. Only the reduced form is persisted, so this is the
    only option available offline; np.interp is linear in score, so the two
    agree except where the bootstrap ordering itself changes between adjacent
    score-grid points.

    Returns None (caller fills NaN) when the file is missing or unusable.
    """
    if not lr_path.exists():
        print(f"  WARNING: no lr_values file at {lr_path} -- LR+/posterior left NaN")
        return None
    try:
        import gzip
        with gzip.open(lr_path, "rt", encoding="utf-8") as f:
            lr = json.load(f)
        score_range = np.asarray(lr["score_range"], dtype=float)
        # Pathogenic-direction (rho, LR_D) curve when pathomechanism mode
        # produced one, else the mechanism-agnostic curve -- same preference
        # the front end's dual-panel branch uses.
        p5_key = ("log_lr_pathogenic_p5" if "log_lr_pathogenic_p5" in lr
                  else "log_lr_plus_p5")
        curve_p5 = np.asarray(lr[p5_key], dtype=float)
        curve_p95 = np.asarray(lr["log_lr_plus_p95"], dtype=float)
    except Exception as e:
        print(f"  WARNING: could not read {lr_path}: {e} -- LR+/posterior left NaN")
        return None

    if len(score_range) == 0 or len(curve_p5) != len(score_range) or len(curve_p95) != len(score_range):
        print(f"  WARNING: malformed curves in {lr_path} -- LR+/posterior left NaN")
        return None

    scores = np.asarray(scores, dtype=float)
    # np.interp's default edge-clamping (curve[0] / curve[-1]) for scores past
    # either end of the labeled score_range, NOT the left/right=NaN convention
    # variant_evidence.py::_compute_bootstrap_lr_percentiles uses: the
    # calibration's own `point_ranges` already extend the outermost tiers to
    # +-inf, so a score beyond the labeled range still carries that tier's
    # evidence, and the LR+ reported alongside it should say the same thing
    # rather than go missing.
    log_p5 = np.interp(scores, score_range, curve_p5)
    log_p95 = np.interp(scores, score_range, curve_p95)

    combined = _combine_directional_log_lr(log_p5, log_p95)

    from src.assay_calibration.fit_utils.fit import thresholds_from_prior

    lr_p, lr_b, _ = thresholds_from_prior(
        prior, POINT_VALUES, acmg_mapping_method=mapping_method)
    reconciled = _reconcile_log_lr_with_points(
        combined, np.asarray(points, dtype=float),
        np.log(np.asarray(lr_p, dtype=float)), np.log(np.asarray(lr_b, dtype=float)))

    # A raw bound can exceed the double range (log(LR+) > 709), and so can a
    # reconciled +8/-8 value, whose band is open on the outer side -- exp then
    # gives +inf, which is honest for the raw diagnostic columns but makes
    # bayes_posterior_from_lr's lr*p/((lr-1)*p+1) evaluate inf/inf = NaN. The
    # posterior is therefore taken from the log value directly (logistic form,
    # algebraically identical), which saturates to 1.0/0.0 instead.
    with np.errstate(over="ignore"):
        lr_plus = np.exp(reconciled)
        return {
            "lr_plus_p5": np.exp(log_p5),
            "lr_plus_p95": np.exp(log_p95),
            "lr_plus": lr_plus,
            "posterior": _posterior_from_log_lr(reconciled, prior),
        }


# ---------------------------------------------------------------------------
# Part A: variant-assay / variant-aggregate tables (pipeline output)
# ---------------------------------------------------------------------------

def _build_variant_groups_for_dataset(
    dataset: str, df_ds: pd.DataFrame, cal_path: Path, dataset_tsv: str,
    method: Optional[str] = None,
) -> List[Dict]:
    """One dataset's full (keep_mask-independent) variant-group rows -- see
    `get_all_variant_groups`. Module-level (not a notebook-local closure) so
    joblib only ever pickles this function + its own small `df_ds` argument,
    same rationale as `analysis.discovery._run_scoreset_job`.

    Each row also gets `lr_plus_p5`/`lr_plus_p95` (the raw directional
    bounds) and `lr_plus`/`posterior` (combined, then reconciled against the
    row's own postprocessed `standard_points`) -- see
    `_lr_posterior_columns`. `method` only selects the acmg_mapping_method
    whose tier thresholds the reconciliation clamps to.

    `_input_row_id` (the master-dataframe index of each input row) is injected
    into `df_ds` before the Scoreset is built so that
    `get_all_variant_groups` can report each group's member rows -- see
    `build_variant_assay_table` (which explodes on them, to keep every
    nucleotide route to an aa-level measurement) and
    `build_dataframe_with_points`. `Variant._init_variant_info` copies every
    input column onto the Variant verbatim, so no plumbing is needed beyond
    adding the column.
    """
    from src.assay_calibration.pipeline.config import PipelineConfig
    from src.assay_calibration.pipeline.utils import load_dataset_from_df
    from src.assay_calibration.pipeline.variant_evidence import get_all_variant_groups

    with open(cal_path) as f:
        calibration = json.load(f)
    if calibration.get("point_ranges") is None:
        return []

    clinvar_release = "2018" if dataset.endswith("_clinvar_2018") else "2025"
    # Reported value is deliberately NOT the bare year: "2018"/"2025" survive a
    # CSV round-trip as int64, so any consumer comparing == "2018" would
    # silently never match. The prefixed token stays a string on read.
    clinvar_release_label = f"clinvar_{clinvar_release}"
    pcfg = PipelineConfig(
        dataset_csv=str(dataset_tsv), dataset_name=dataset, output_dir="/tmp",
        clinvar_release=clinvar_release,
    )
    if INPUT_ROW_ID not in df_ds.columns:
        df_ds = df_ds.copy()
        df_ds[INPUT_ROW_ID] = df_ds.index
    scoreset = load_dataset_from_df(df_ds, pcfg)
    rows = get_all_variant_groups(scoreset, calibration["point_ranges"])
    for r in rows:
        r["dataset"] = dataset
        r["gene"] = dataset.split("_")[0]
        r["sample"] = None
        # Which ClinVar release the calibration that produced this row's
        # points was fitted against ("clinvar_2018"/"clinvar_2025") -- the only
        # thing the old points_clinvar_2018/evidence_clinvar_2018 column pair
        # actually conveyed, now that it's a single points/evidence pair.
        r["clinvar_release"] = clinvar_release_label
        # The calibration prior the posterior was computed at -- carried so
        # `posterior` is reproducible from `lr_plus` alone
        # (_posterior_from_log_lr(log(lr_plus), prior)) without having to go
        # back to the dataset's calibration JSON.
        r["prior"] = (float(calibration["prior"])
                      if calibration.get("prior") is not None else np.nan)

    lr_cols = None
    if rows and calibration.get("prior") is not None:
        # Exact sibling of the calibration JSON (both written by
        # pipeline/utils.py::save_results into the same dataset dir) -- no
        # need for load_lr_values' rglob, and method/comp aren't parsed here.
        lr_path = Path(str(cal_path).replace("_calibration.json", "_lr_values.json.gz"))
        lr_cols = _lr_posterior_columns(
            scores=np.array([r["score"] for r in rows], dtype=float),
            points=np.array([r["standard_points"] for r in rows], dtype=float),
            lr_path=lr_path,
            prior=float(calibration["prior"]),
            mapping_method=_resolve_mapping_method(method),
        )
    for key in ("lr_plus_p5", "lr_plus_p95", "lr_plus", "posterior"):
        for i, r in enumerate(rows):
            r[key] = float(lr_cols[key][i]) if lr_cols is not None else np.nan
    return rows


def build_all_variant_groups_table(
    tree: Dict,
    model_selections: Dict,
    dataset_configs: Optional[Dict],
    calibrations: Dict,
    dataset_tsv: str,
    datasets: List[str],
    primary_method: str,
) -> pd.DataFrame:
    """Every variant every one of `datasets`' assay measured, scored in-bag
    against `primary_method`'s calibration, regardless of keep_mask/sample
    membership.

    Reusable promotion of `analyze_pipeline_output.py` section 10's own
    `_build_all_variant_groups_for_dataset`/parallel-loop block -- unchanged
    behavior (including treating any `_clinvar_2018`-suffixed entry in
    `datasets` as its own independent dataset, exactly like every other
    caller of `load_all_variants`/`discover_outputs`). `standard_points` is
    always in-bag -- see module docstring.

    `primary_method` doubles as the acmg_mapping_method whose tier thresholds
    each row's `lr_plus` is reconciled against (see
    `_build_variant_groups_for_dataset` / `_reconcile_log_lr_with_points`).
    """
    from joblib import Parallel, delayed

    df_full = load_master_df(dataset_tsv)
    df_ds_by_dataset = {}
    for dataset in datasets:
        if dataset not in tree:
            continue
        try:
            df_ds_by_dataset[dataset] = _filter_dataset_df(df_full, dataset, dataset_tsv)
        except Exception as e:
            print(f"  WARNING: could not slice {dataset} for all-variant-groups: {e}")
    del df_full

    jobs = []
    for dataset, df_ds in df_ds_by_dataset.items():
        comp = resolve_component(dataset, list(tree[dataset].keys()), model_selections, dataset_configs)
        cal_path = (calibrations or {}).get(dataset, {}).get(primary_method, {}).get(comp)
        if cal_path is None:
            continue
        jobs.append((dataset, df_ds, cal_path))

    if not jobs:
        return pd.DataFrame()

    n_jobs = _resolve_n_jobs(-1)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_build_variant_groups_for_dataset)(
            dataset, df_ds, cal_path, dataset_tsv, primary_method)
        for dataset, df_ds, cal_path in jobs
    )
    return pd.DataFrame([row for rows in results for row in rows])


def _protein_variant(aa_ref, aa_pos, aa_alt) -> Optional[str]:
    """`p.{aa_ref}{aa_pos}{aa_alt}`-style short form (e.g. "T2P"), or None if
    any of the three fields is missing -- same construction
    make_variant_aggregate_evidence_table.ipynb's record-building loop uses."""
    if pd.isna(aa_ref) or pd.isna(aa_pos) or pd.isna(aa_alt) or aa_ref == "" or aa_alt == "":
        return None
    try:
        return f"{aa_ref}{int(float(aa_pos))}{aa_alt}"
    except (ValueError, TypeError):
        return None


# Group-level (OR-promoted) copies of the ClinVar/gnomAD membership flags --
# see build_variant_assay_table for why both meanings are reported.
_GROUP_FLAG_COLS = {
    "is_benign": "is_benign_group",
    "is_pathogenic": "is_pathogenic_group",
    "is_vus": "is_vus_group",
    "is_gnomad": "is_gnomad_group",
}

# Per-input-row identity taken from the master dataframe rather than from the
# group's first member. Key = master-TSV column, value = output name.
#
# `transcript_id` is sourced here from "Ensembl Transcript ID" (with spaces,
# as the TSV spells it). get_all_variant_groups reads it as
# `getattr(v0, "Ensembl_transcript_ID")`, which never matches -- that column
# was 100% null in every previously shipped table.
_ROW_IDENTITY_COLS = {
    "Chrom": "chrom", "hg38_start": "POS", "ref_allele": "REF", "alt_allele": "ALT",
    "hgvs_c": "hgvs_c", "hgvs_p": "hgvs_p",
    "aa_ref": "aa_ref", "aa_pos": "aa_pos", "aa_alt": "aa_alt",
    "nucleotide_or_aa": "nucleotide_or_aa",
    "Ensembl Transcript ID": "transcript_id",
    "mavedb_variant_urn": "mavedb_variant_urn",
    "simplified_consequence": "simplified_consequence",
    "Gene": "gene_symbol",
}

# Per-member flag lists get_all_variant_groups emits alongside
# `input_row_ids`, exploded in parallel with it -> the per-row `is_*` columns.
_ROW_FLAG_COLS = {
    "row_is_benign": "is_benign",
    "row_is_pathogenic": "is_pathogenic",
    "row_is_vus": "is_vus",
    "row_is_gnomad": "is_gnomad",
}


def build_variant_assay_table(df_all_groups: pd.DataFrame,
                              df_input: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per (INPUT DATAFRAME ROW, dataset), carrying that row's own
    variant identity plus its measurement's evidence.

    Previously this emitted one row per *measured* variant, taking identity
    from the group's first member -- which silently dropped every other
    nucleotide route to an aa-level measurement (e.g. MSH2_Jia_2021 has 53,800
    distinct (POS,REF,ALT) across 17,746 amino-acid changes, so ~2/3 of its
    nucleotide variants had no row at all, and coordinate-based joins missed
    them). `get_all_variant_groups` now reports each group's member
    `input_row_ids`, so the group rows are exploded and re-joined against
    `df_input` to give every nucleotide route its own row.

    What is per-MEASUREMENT (identical across a group's rows, by definition):
    `group_auth_score` -- the mean score the Scoreset actually fitted, see
    `get_all_variant_groups` -- plus `points`, `evidence` and the
    LR+/posterior columns derived from it. The row's own raw score stays in
    the input dataframe's `auth_reported_score`.

    What is per-ROW: the identity columns (`_ROW_IDENTITY_COLS`), and the
    `is_benign`/`is_pathogenic`/`is_vus`/`is_gnomad` flags. Those four used to
    be the group-OR value; they are now the row's own status, with the OR
    preserved alongside as `is_*_group`. Both are needed and neither
    substitutes for the other: the OR is what made the *measurement* a P/LP
    or gnomAD control during calibration, while the per-row value is the only
    honest answer at nucleotide level (~46k rows are pathogenic only by
    promotion, ~97k only-gnomAD by promotion).

    There is a single `points`/`evidence` pair, with `clinvar_release`
    recording which ClinVar release the calibration was fitted against. The
    old `points_clinvar_2018`/`evidence_clinvar_2018` twin pair is gone: the
    22 `*_clinvar_2018` datasets are twin-ONLY on disk, so the two pairs were
    disjoint in every row and the second conveyed nothing but the release.
    A twin dataset still reports under its assay name (the `_clinvar_2018`
    suffix is stripped from `dataset`).
    """
    if df_all_groups.empty:
        return df_all_groups

    df = df_all_groups.copy()
    # Report under the assay's name; `clinvar_release` carries what the
    # suffix used to encode.
    df["dataset"] = df["dataset"].str.replace("_clinvar_2018$", "", regex=True)

    # Guard: the single points/evidence pair assumes a variant is scored
    # under at most one ClinVar release per assay. Twin-only is true for
    # every dataset on disk today; if a tree ever carries both a base and a
    # twin calibration they now surface as two rows rather than being
    # silently coalesced, which changes the aggregate's n_assays.
    if "clinvar_release" in df.columns:
        dup = df.groupby(["dataset", "variant_id"])["clinvar_release"].nunique()
        n_dup = int((dup > 1).sum())
        if n_dup:
            print(f"  WARNING: {n_dup:,} (dataset, variant) pairs are scored under BOTH "
                  f"ClinVar releases -- they will appear as separate rows, and the "
                  f"variant-aggregate table's n_assays will count each release separately")

    df = df.rename(columns={"score": "group_auth_score"})
    for src, dst in _GROUP_FLAG_COLS.items():
        if src in df.columns:
            df = df.rename(columns={src: dst})

    # --- expand groups to their member input rows --------------------------
    if df_input is None or "input_row_ids" not in df.columns:
        raise ValueError(
            "build_variant_assay_table needs `df_input` and an `input_row_ids` "
            "column to emit per-input-row rows; pass the master dataframe "
            "(see build_all_variant_groups_table)."
        )

    # Everything the group supplies that is NOT per-row: v0-derived identity
    # columns are dropped, since each member row's own identity is joined in
    # from df_input below.
    v0_identity = set(_ROW_IDENTITY_COLS.values()) | {
        "chrom", "pos", "ref", "alt", "mavedb_urn", "variant_id",
        "simplified_consequence", "nucleotide_or_aa", "gene", "sample",
    }
    explode_cols = ["input_row_ids"] + [c for c in _ROW_FLAG_COLS if c in df.columns]
    measurement_cols = [
        c for c in df.columns if c not in v0_identity and c not in explode_cols
    ]
    # Multi-column explode keeps input_row_ids aligned with its parallel
    # per-member flag lists (pandas >= 1.3).
    exploded = df[measurement_cols + explode_cols].explode(explode_cols, ignore_index=True)
    exploded = exploded.rename(
        columns={"input_row_ids": INPUT_ROW_ID, **_ROW_FLAG_COLS})
    exploded = exploded[exploded[INPUT_ROW_ID].notna()]

    row_cols = [c for c in _ROW_IDENTITY_COLS if c in df_input.columns]
    rows_df = df_input[row_cols].rename(
        columns={k: v for k, v in _ROW_IDENTITY_COLS.items() if k in row_cols})
    merged = exploded.merge(rows_df, left_on=INPUT_ROW_ID, right_index=True, how="left")

    # --- derived columns ---------------------------------------------------
    merged["points"] = merged["standard_points"]
    merged["evidence"] = merged["points"].map(POINTS_TO_EVIDENCE_CODE)
    merged["protein_variant"] = [
        _protein_variant(ar, ap, aa)
        for ar, ap, aa in zip(merged["aa_ref"], merged["aa_pos"], merged["aa_alt"])
    ]
    merged["#CHROM"] = np.where(
        merged["chrom"].isna(), None, "chr" + merged["chrom"].astype(str))

    cols = [
        "gene_symbol", "protein_variant", "#CHROM", "POS", "REF", "ALT",
        "transcript_id", "hgvs_c", "hgvs_p", "nucleotide_or_aa", "aa_ref", "aa_pos", "aa_alt",
        "is_benign", "is_pathogenic", "is_vus", "is_gnomad",
        "is_benign_group", "is_pathogenic_group", "is_vus_group", "is_gnomad_group",
        "dataset", "clinvar_release", "mavedb_variant_urn", "group_n_rows",
        "group_auth_score", "points", "evidence",
        "lr_plus", "prior", "posterior", "lr_plus_p5", "lr_plus_p95", INPUT_ROW_ID,
    ]
    return merged[[c for c in cols if c in merged.columns]]


def build_variant_aggregate_table(variant_assay_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse `build_variant_assay_table`'s (variant, dataset) rows to one
    row per physical variant -- the "merge every assay's copy of a variant
    into a canonical evidence call" step
    `make_variant_aggregate_evidence_table.ipynb` hand-rolls (splice-variant
    exclusion aside -- Scoreset's own splicing_filter already dropped those
    upstream, see `data_utils/dataset.py::splicing_filter`), driven by real
    identity columns (#CHROM/POS/REF/ALT or aa_ref/aa_pos/aa_alt) instead of
    the notebook's own string/id matching, and reusing
    `analysis.multi_scoreset._merge_points`'s abs-max-if-same-sign rule for
    the final merged points instead of the notebook's separate conflict-mask
    + idxmax logic.

    A variant whose assays disagree in sign (`_merge_points` would return 0)
    is dropped entirely rather than reported as 0/NO_EVIDENCE, matching the
    notebook's own `conflicting_fxn_data` exclusion -- 0 there means "assays
    agree there's no evidence", not "assays disagree".

    Every row is aggregated on its NUCLEOTIDE coordinate
    (`gene_symbol`/`#CHROM`/`POS`/`REF`/`ALT`), aa-level measurements
    included, so the table is one row per physical nucleotide variant and
    joins cleanly on coordinates. This is only correct because
    `build_variant_assay_table` now emits one row per input dataframe row: an
    aa-level measurement contributes a row at each of its real nucleotide
    routes, rather than only at the first one. `nucleotide_or_aa` survives as
    PROVENANCE -- whether the reported evidence came from an aa-level or
    nt-level assay -- which matters because thousands of coordinates are
    measured by both. It needs no special rule: the `best` row supplies it
    like every other non-points column, so it names the winning assay's level.

    Rows with no usable nucleotide coordinate keep the old aa-level key
    instead of being dropped.

    `lr_plus`/`posterior` (and the raw `lr_plus_p5`/`lr_plus_p95` bounds) are
    inherited from the same `best` row every other non-points column comes
    from, with no separate selection rule. That stays consistent with the
    merged `points`: `_merge_points` is abs-max-if-same-sign and sign-
    conflicting groups are dropped above, so the merged value always equals
    the `best` row's own points -- the tier `best`'s `lr_plus` was already
    reconciled into.
    """
    if variant_assay_df.empty:
        return variant_assay_df

    df = variant_assay_df.copy()

    # One points column now (see build_variant_assay_table): the ClinVar-2018
    # twin pair and the GENES_2018 switch it required are both gone -- the
    # 2018 calibration's points live in `points` like any other, with
    # `clinvar_release` recording the provenance.
    has_coord = df[["#CHROM", "POS", "REF", "ALT"]].notna().all(axis=1)
    df_nt, df_aa = df[has_coord].copy(), df[~has_coord].copy()

    def _aggregate(sub: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        rows = []
        for _, grp in sub.groupby(group_cols, sort=False, dropna=False):
            pts = grp["points"].fillna(0).to_numpy()
            nonzero = pts[pts != 0]
            if len(nonzero) and (nonzero > 0).any() and (nonzero < 0).any():
                continue  # conflicting_fxn_data -- excluded, not zeroed
            best = grp.loc[grp["points"].fillna(0).abs().idxmax()]
            row = best.to_dict()
            row["points"] = _merge_points(pts)
            row["evidence"] = POINTS_TO_EVIDENCE_CODE.get(row["points"])
            row["n_assays"] = grp["dataset"].nunique()
            rows.append(row)
        return pd.DataFrame(rows)

    agg_nt = _aggregate(df_nt, ["gene_symbol", "#CHROM", "POS", "REF", "ALT"])
    agg_aa = _aggregate(df_aa, ["gene_symbol", "aa_ref", "aa_pos", "aa_alt", "transcript_id"])

    out = pd.concat([agg_nt, agg_aa], ignore_index=True)
    # `_input_row_id` is meaningless after merging assays -- the merged row
    # descends from `best`'s input row, not from any single one.
    return out.drop(columns=[c for c in (INPUT_ROW_ID,) if c in out.columns])


# ---------------------------------------------------------------------------
# Part A2: the input dataframe, annotated with the evidence it produced
# ---------------------------------------------------------------------------

# Evidence carried back onto each input row by build_dataframe_with_points,
# in output order, deliberately a SUBSET of what the variant-assay table
# carries: just the headline call (points/evidence code) and the Bayesian
# triple it rests on (lr_plus -> prior -> posterior, in that reading order).
#
# Deliberately NOT carried here, to keep the annotated input dataframe narrow:
# the raw directional bounds (`lr_plus_p5`/`lr_plus_p95`), `group_auth_score`,
# `group_n_rows` and the four `is_*_group` promotion flags. All of them remain
# available per (variant, assay) in variant_assay_specific_evidence.csv.gz,
# joinable on `_input_row_id`.
#
# `dataset` is also absent -- it duplicates the input dataframe's own
# `Dataset` column exactly (verified: equal in every scored row).
_DF_EVIDENCE_COLS = [
    "clinvar_release", "points", "evidence",
    "lr_plus", "prior", "posterior",
]

# Renames applied only to `dataframe_with_points`: every added column is
# `excalibr_`-prefixed there, since they are additions to somebody else's
# dataframe and must not be mistaken for its own 95 columns. The other tables
# keep the bare names, where they are the table's own subject.
_DF_EVIDENCE_RENAMES = {
    "clinvar_release": "excalibr_clinvar_release",
    "points": "excalibr_points",
    "evidence": "excalibr_acmg_evidence_code",
    "lr_plus": "excalibr_lr_plus",
    "prior": "excalibr_prior",
    "posterior": "excalibr_posterior",
}

# VEP consequences Scoreset.splicing_filter drops when the assay does not
# itself measure splicing -- kept in sync with data_utils/dataset.py's own
# list (see _derive_filter_reason for why this is duplicated rather than
# imported).
_VEP_SPLICE_CONSEQUENCES = {
    "splice region", "splice_site_variant",
    "splice_acceptor_variant", "splice_donor_variant",
}
_SPLICEAI_COLS = ("spliceAI_DS_AG", "spliceAI_DS_AL", "spliceAI_DS_DG", "spliceAI_DS_DL")


def _derive_filter_reason(df_input: pd.DataFrame, scored_ids: set,
                          scored_datasets: set,
                          score_col: str = "auth_reported_score",
                          spliceai_threshold: float = 0.2,
                          vep_splice_filter: bool = True) -> pd.Series:
    """Why each input row produced no evidence ("" when it did).

    `Scoreset` drops rows without recording a reason, so rather than change
    its behaviour these predicates are re-applied here, mirroring
    `data_utils/dataset.py::_init_dataframe`'s drop chain in the same order:
    NaN score (`:813`), `Flag == "*"` (`:861-870`), then -- only for assays
    that do NOT measure splicing themselves (`splice_measure != "Yes"`) --
    VEP splice consequences (`:923-933`) and the SpliceAI threshold
    (`:936-942`).

    Any row with no evidence that none of these explain is labelled
    `filtered_other`, which is asserted to be empty in verification: a
    nonzero count means this duplicated logic has drifted from the real
    filter and needs re-syncing (the alternative, importing the predicates,
    would mean restructuring Scoreset purely for reporting).
    """
    reason = pd.Series("", index=df_input.index, dtype=object)
    unscored = ~df_input.index.isin(scored_ids)

    excluded = unscored & ~df_input["Dataset"].isin(scored_datasets)
    reason[excluded] = "excluded_dataset"

    todo = unscored & (reason == "")
    if score_col in df_input.columns:
        bad = todo & pd.to_numeric(df_input[score_col], errors="coerce").isna()
        reason[bad] = "nan_score"

    todo = unscored & (reason == "")
    if "Flag" in df_input.columns:
        reason[todo & df_input["Flag"].eq("*")] = "flag"

    # splice filters only run for assays that don't measure splicing
    detects_splice = df_input["splice_measure"].eq("Yes") if "splice_measure" in df_input.columns \
        else pd.Series(False, index=df_input.index)

    todo = unscored & (reason == "") & ~detects_splice
    if vep_splice_filter and "simplified_consequence" in df_input.columns:
        cons = df_input["simplified_consequence"].astype(str).str.lower()
        reason[todo & cons.isin(_VEP_SPLICE_CONSEQUENCES)] = "vep_splice"

    todo = unscored & (reason == "") & ~detects_splice
    if spliceai_threshold is not None and all(c in df_input.columns for c in _SPLICEAI_COLS):
        ds = df_input[list(_SPLICEAI_COLS)].apply(pd.to_numeric, errors="coerce")
        reason[todo & (ds >= spliceai_threshold).any(axis=1)] = "spliceai"

    reason[unscored & (reason == "")] = "filtered_other"
    return reason


def build_dataframe_with_points(df_input: pd.DataFrame,
                                variant_assay_df: pd.DataFrame) -> pd.DataFrame:
    """The input dataframe, unchanged, plus the evidence each row produced.

    Every input row is kept and the index is preserved, so this is the input
    file with columns appended -- including rows that never reached a
    Scoreset. Those get NaN evidence and a non-empty `excalibr_filter_reason`
    saying why (see `_derive_filter_reason`).

    The join is exactly 1:1 on `_input_row_id`: `_filter_dataset_df`'s
    per-dataset slices partition the master dataframe (their row counts sum to
    its length), so no input row is scored by two datasets.
    """
    if variant_assay_df.empty or INPUT_ROW_ID not in variant_assay_df.columns:
        raise ValueError(
            "build_dataframe_with_points needs a variant-assay table carrying "
            f"{INPUT_ROW_ID!r} (see build_variant_assay_table)")

    ev_cols = [c for c in _DF_EVIDENCE_COLS if c in variant_assay_df.columns]
    ev = variant_assay_df[[INPUT_ROW_ID] + ev_cols].copy()
    # Bare release year here (the prefixed COLUMN name carries the provenance),
    # so "clinvar_2018" -> "2018". Note this reads back from CSV as int64.
    if "clinvar_release" in ev.columns:
        ev["clinvar_release"] = (
            ev["clinvar_release"].astype(str).str.replace("^clinvar_", "", regex=True))
    ev = ev.rename(columns=_DF_EVIDENCE_RENAMES)

    dup = int(ev[INPUT_ROW_ID].duplicated().sum())
    if dup:
        print(f"  WARNING: {dup:,} input rows are scored more than once -- the "
              f"per-dataset slices are expected to partition the input, so this "
              f"will fan out rows; keeping the first occurrence of each")
        ev = ev.drop_duplicates(subset=[INPUT_ROW_ID], keep="first")

    ev = ev.set_index(INPUT_ROW_ID)
    ev.index = ev.index.astype(df_input.index.dtype)
    out = df_input.join(ev, how="left")

    out["excalibr_filter_reason"] = _derive_filter_reason(
        df_input,
        scored_ids=set(ev.index),
        scored_datasets=set(variant_assay_df["dataset"].unique()),
    )
    return out


# ---------------------------------------------------------------------------
# Part B: predictor-only table (from a separately-produced, already-merged
# predictor/functional CSV -- see analysis.config.PREDICTOR_EVIDENCE_SOURCE_CSV)
# ---------------------------------------------------------------------------

# Superseded by build_variant_aggregate_table's own output once this table
# is joined against it -- dropped here rather than carried forward stale.
PREDICTOR_STALE_FUNCTIONAL_COLS = [
    "experimental_dataset_highest_evidence", "assay_score",
    "points_experimental", "evidence_experimental",
    "points_experimental_clinvar_2018", "evidence_experimental_clinvar_2018",
]

# Every "does this row have a predictor calibration at all" column, across
# all three predictors and both calibration scopes (domain_aggregate =
# per-protein-domain calibration, single_gene = per-gene calibration).
PREDICTOR_POINTS_COLS = [
    "points_AlphaMissense_domain_aggregate", "points_AlphaMissense_single_gene",
    "points_MutPred2_domain_aggregate", "points_MutPred2_single_gene",
    "points_REVEL_domain_aggregate", "points_REVEL_single_gene",
]


def build_predictor_only_table(path: Optional[str] = None) -> pd.DataFrame:
    """Strip the pre-merged predictor+functional CSV
    (`analysis.config.PREDICTOR_EVIDENCE_SOURCE_CSV`, already shaped like
    `variant_aggregated_experimental_predictive_evidence.csv.gz`) down to a
    predictor-only table: drop the stale functional-evidence-derived columns
    (`PREDICTOR_STALE_FUNCTIONAL_COLS` -- superseded by
    `build_variant_aggregate_table`'s own output) and any row with no
    predictor calibration score at all, from any of AlphaMissense/MutPred2/
    REVEL, in either the domain_aggregate or single_gene calibration scope
    (`PREDICTOR_POINTS_COLS`). Meant to be joined against
    `build_variant_aggregate_table`'s output on (gene_symbol, #CHROM, POS,
    REF, ALT) later -- not done here.
    """
    from analysis import config as _cfg

    path = path or _cfg.PREDICTOR_EVIDENCE_SOURCE_CSV
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in PREDICTOR_STALE_FUNCTIONAL_COLS if c in df.columns])
    points_cols = [c for c in PREDICTOR_POINTS_COLS if c in df.columns]
    has_predictor = df[points_cols].notna().any(axis=1)
    n_dropped = int((~has_predictor).sum())
    print(f"  Predictor-only table: dropping {n_dropped:,}/{len(df):,} rows with no "
          f"predictor calibration score from any source (AM/MP2/REVEL, domain or gene)")
    return df[has_predictor].reset_index(drop=True)


# Group columns identifying a physical variant, shared by
# build_variant_aggregate_table's output and build_predictor_only_table's
# output -- the join key for merge_variant_aggregate_with_predictors.
_VARIANT_GROUP_COLS = ["gene_symbol", "#CHROM", "POS", "REF", "ALT"]

# Columns that can legitimately appear on BOTH sides of the merge (the
# variant-aggregate table carries its own hgvs_c/hgvs_p/transcript_id/
# protein_variant from the functional assay; the predictor table carries its
# own copies from whatever VEP annotation produced it) -- coalesced (prefer
# the variant-aggregate/functional side, fall back to the predictor side)
# rather than kept as two separate `_pred`-suffixed columns.
_SHARED_IDENTITY_COLS = ["protein_variant", "transcript_id", "hgvs_c", "hgvs_p"]


def merge_variant_aggregate_with_predictors(
    variant_aggregate_df: pd.DataFrame, predictor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Outer-join `build_variant_aggregate_table`'s functional-evidence rows
    with `build_predictor_only_table`'s predictor-score rows on
    `_VARIANT_GROUP_COLS` -- the join `test/merge_functional_predictor_evidence.ipynb`
    hand-rolled per-predictor-CSV, now done once against the already-merged
    AM/MP2/REVEL predictor-only table instead.

    Outer (not inner/left) so a variant with predictor scores but no
    functional assay coverage, or vice versa, is kept rather than dropped --
    matching the source notebook's own outer merge + `_merge` indicator
    reporting. `_SHARED_IDENTITY_COLS` are coalesced (functional side wins,
    predictor side fills gaps) rather than duplicated.
    """
    agg = variant_aggregate_df.copy()
    pred = predictor_df.copy()
    for df_ in (agg, pred):
        df_["POS"] = pd.to_numeric(df_["POS"], errors="coerce")

    merged = agg.merge(
        pred, on=_VARIANT_GROUP_COLS, how="outer", suffixes=("", "_pred"), indicator=True,
    )
    for col in _SHARED_IDENTITY_COLS:
        pred_col = f"{col}_pred"
        if pred_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[pred_col])
            else:
                merged[col] = merged[pred_col]
            merged = merged.drop(columns=[pred_col])

    n_matched = int((merged["_merge"] == "both").sum())
    n_predictor_only = int((merged["_merge"] == "right_only").sum())
    n_functional_only = int((merged["_merge"] == "left_only").sum())
    print(f"  Merged variant-aggregate x predictor-only: {n_matched:,} matched, "
          f"{n_predictor_only:,} predictor-only (no functional assay coverage), "
          f"{n_functional_only:,} functional-only (no predictor score)")
    return merged.drop(columns=["_merge"])
