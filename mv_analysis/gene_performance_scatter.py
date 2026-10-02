#!/usr/bin/env python3
"""
Gene-level ExCALIBR-MV-vs-ExCALIBR(UV) MCC scatter, 3 panels (A: functional, B:
computational predictors, C: combined functional+predictor evidence), plus an
optional 4th panel showing TP53's RPV (reduced-penetrance-variant) penetrance-score
distribution.

Mirrors analysis/gene_performance_scatter.py's Panel-A visual style (diagonal
reference line, point size by N, gene-name labels via adjustText) but compares
MV vs UV MCC per gene rather than ExCALIBR-vs-author accuracy.

Per-gene MV/UV values are read straight off mv_analysis.report.build_comparison_table
(same machinery used throughout this session's ad hoc analyses), always at the
evidence-direction threshold (points >=1 / <=-1) and the best-MCC MV config, with the
UV baseline standardized on "non-conflicting" aggregation everywhere.
"""
import contextlib
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Slideshow-mode per-panel colors (A/B/C) -- the dataviz skill's first three
# categorical slots (references/palette.md), the only subset of its 8-hue
# theme validated all-pairs-safe for a 3-series scatter (worst-pair CVD
# Delta-E 9.2, normal-vision Delta-E 24.0): aqua/teal reads as "experimental/
# assay data", orange as "computational predictor/ML", blue as "combined,
# unified evidence".
_SLIDES_PANEL_COLOR = {"A": "#1baf7a", "B": "#eb6834", "C": "#2a78d6"}

# Slideshow-mode gene-label display-name overrides -- "FGFR_COMBINED" (the
# pooled FGFR1/2/3/4 dataset_name) reads as a single unfamiliar gene on a
# slide; the paper version leaves this alone since its audience already
# knows the FGFR panel's construction.
_SLIDES_GENE_LABEL = {"FGFR_COMBINED": "FGFR1/2/3/4"}

from src.assay_calibration.data_utils.dataset import MultiScoreset
from src.assay_calibration.multivariate_analysis import mv_calibration as _mv_calibration_mod
from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
from src.assay_calibration.multivariate_analysis import eval_plot_utils as epu
from src.assay_calibration.plot_utils.utils import compute_classification_metrics
from src.assay_calibration.multivariate_data.card11 import build_card11_multiscoreset
from src.assay_calibration.multivariate_data.tp53 import build_tp53_multiscoreset
from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
from src.assay_calibration.multivariate_data.combined import (
    build_functional_scoresets, build_combined_multiscoreset,
    get_functionally_assayed_protein_variants, DEFAULT_INTEGRATED_DATAFRAME,
)
from src.assay_calibration.multivariate_data.predictors import (
    load_predictor_ms, predictor_dataset_label, DEFAULT_GENES,
)
from src.assay_calibration.multivariate_data.common import (
    resolve_clinvar_release, gene_set_dataset_label, build_multiscoreset_from_long_dataframe,
)
from src.assay_calibration.fit_utils.fit import Fit

from mv_analysis.report import build_comparison_table
from mv_analysis import config

EVIDENCE_DIRECTION = "evidence_direction (>=1 / <=-1)"

PLAIN_INTEGRATED_GENES = [
    "ASPA", "BRCA1", "BRCA2", "CBS", "CHEK2", "F9", "GCK", "HMBS",
    "KCNE1", "KCNH2", "KCNQ4", "LDLR", "PALB2", "PAX6", "PTEN",
]
COMBINED_GENES = list(DEFAULT_GENES)  # BRCA1, BRCA2, F9, JAG1, MSH2, SCN5A, TP53, TSC2

# Meta-analysis "datasets" that aggregate other rows in the same gene's group
# rather than representing an independent assay dimension (excluded by
# hpc/prepare.py::_discover_gene_groups for the plain-"integrated" gene-set;
# combined.py's own functional-scoreset builder doesn't exclude these, which
# is fine for the "combined" gene-set (different pipeline) but was wrong for
# Panel A's plain-integrated genes -- caused a dimension mismatch for F9).
META_ANALYSIS_DATASETS = {"F9_Popp_2025_model", "TP53_Fayer_2021_meta"}

RUN_KWARGS = dict(
    path_percentile=5, min_valid_boots=1,
    reestimate_marginal_weights=False,
    enforce_marginal_monotonicity=False,
    liberal_marginal_monotonicity=False,
)
_AUX_INDICES = {"tp53": [4], "card11": [4, 5]}


# ── fast repeated-analysis loading: parse the (large) results json once ──────

_RAW_CACHE = {}
_REAL_GZIP_OPEN = _mv_calibration_mod.gzip.open
_REAL_JSON_LOAD = _mv_calibration_mod.json.load


def _load_raw(results_json):
    if results_json not in _RAW_CACHE:
        print(f"Loading and caching {results_json} (one-time cost)...")
        with _REAL_GZIP_OPEN(results_json, "rt", encoding="utf-8") as f:
            _RAW_CACHE[results_json] = _REAL_JSON_LOAD(f)
    return _RAW_CACHE[results_json]


class _CachedFileSentinel:
    """Stands in for the gzip file handle MVCalibrationAnalysis opens, so the
    matching json.load(f) call below can recognize it (by identity) and
    short-circuit to the cached dict -- WITHOUT touching json.load/gzip.open
    for any other path/file. A naive `mock.patch(json, "load", ...)` patches
    the process-wide `json` module singleton (every `import json` anywhere
    shares it), which previously broke uv_sources.py's unrelated json.load
    calls (dataset_configs_aug_2026.json, per-dataset calibration.json) during
    the same context -- confirmed this turn: it silently returned the cached
    MV results dict in place of those files' real content, making
    load_tp53_uv_points appear broken when called from within this context
    even though it works perfectly standalone."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextlib.contextmanager
def fast_results_json(results_json):
    """Pre-parse results_json once; for the duration of this context, opening
    THIS SPECIFIC path via mv_calibration's gzip.open returns a sentinel, and
    json.load(sentinel) returns the cached dict -- every other gzip.open/
    json.load call (different path, or a different file object entirely)
    passes through to the real functions unchanged."""
    raw = _load_raw(results_json)
    sentinel = _CachedFileSentinel()

    def _patched_gzip_open(path, *a, **kw):
        if str(path) == str(results_json):
            return sentinel
        return _REAL_GZIP_OPEN(path, *a, **kw)

    def _patched_json_load(f, *a, **kw):
        if f is sentinel:
            return raw
        return _REAL_JSON_LOAD(f, *a, **kw)

    with mock.patch.object(_mv_calibration_mod.gzip, "open", side_effect=_patched_gzip_open), \
         mock.patch.object(_mv_calibration_mod.json, "load", side_effect=_patched_json_load):
        yield


def _find_dataset_key(results_json, gene):
    """For the 15 plain-'integrated' genes, the results-json key isn't a fixed
    formula (e.g. 'ASPA_mv', 'BRCA1_mv_clinvar_2018', 'PTEN_mv_clinvar_2018') --
    search the loaded keys for gene_mv[_*]."""
    raw = _load_raw(results_json)
    prefix = f"{gene}_mv"
    matches = [k for k in raw if k == prefix or k.startswith(prefix + "_")]
    if not matches:
        raise KeyError(f"No '{prefix}[_*]' key found for {gene} in {results_json}")
    return matches[0]


def _cache_path(cache_dir, panel, gene):
    return Path(cache_dir) / f"{panel}_{gene}.json"


def _load_cached(cache_dir, panel, gene):
    if cache_dir is None:
        return None
    p = _cache_path(cache_dir, panel, gene)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_cached(cache_dir, panel, gene, record):
    if cache_dir is None:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(_cache_path(cache_dir, panel, gene), "w") as f:
        json.dump(record, f)


_METRIC_FIELDS = ("mv_mcc", "uv_mcc", "mv_accuracy", "uv_accuracy", "mv_auc", "uv_auc",
                  "n_pathogenic", "n_benign", "n_eval")


def _row_from_cache_record(c):
    return {"gene": c["gene"], **{f: c[f] for f in _METRIC_FIELDS}}


def _valid_cache_entry(c):
    """A cache entry is only trustworthy as-is if it's either a settled
    non-"ok" status (failed/skipped -- schema-independent) or an "ok" record
    that actually has every field the CURRENT schema expects. Old "ok"
    records written before mv_accuracy/uv_accuracy/mv_auc/uv_auc/
    n_pathogenic/n_benign were added (e.g. from an earlier run this session)
    fail this check and are treated as a cache miss -- self-healing, no
    manual cache-clearing needed after a schema change."""
    if c is None:
        return False
    if c.get("status") != "ok":
        return True
    return all(f in c for f in _METRIC_FIELDS)


def _all_ok_cached(cache_dir, panel, genes):
    """True if every gene in `genes` already has ANY VALID cache entry (ok,
    failed, or skipped) -- lets callers skip an expensive bulk ms-build (e.g.
    LABEL-seq's 17-gene, ~10min build_labelseq_multiscoresets()) entirely
    when nothing in that block actually needs (re)computing. A "failed"
    status is treated as settled, not auto-retried, on the assumption that a
    structural cause (e.g. a gene missing P/LP entirely) won't change on its
    own -- delete that gene's specific cache file to force a retry after a
    code fix that might actually resolve it."""
    if cache_dir is None:
        return False
    return all(_valid_cache_entry(_load_cached(cache_dir, panel, g)) for g in genes)


def _cached_rows(cache_dir, panel, genes):
    rows = []
    for g in genes:
        c = _load_cached(cache_dir, panel, g)
        if _valid_cache_entry(c) and c.get("status") == "ok":
            rows.append(_row_from_cache_record(c))
    return rows


def _extract_mv_uv(table):
    """Metrics dict from a build_comparison_table() result, at the
    evidence-direction threshold, MV = best-MCC config across configs (its
    accuracy/auc are taken from that SAME best-by-MCC row, not independently
    re-maximized per metric), UV = non-conflicting only (the
    session-standardized rule).

    n_pathogenic/n_benign are threshold-independent raw label counts in the
    evaluated set (see report.py's _rows_for_points) -- 0 in either means
    MCC/AUC are mathematically undefined for that gene (compute_
    classification_metrics silently returns 0.0 rather than NaN in that
    case), which callers should treat as "exclude from the MCC/AUC plots"
    rather than "genuinely zero performance". Accuracy has no such
    degeneracy (well-defined even with one class present), so it does NOT
    get filtered the same way."""
    sub = table[table["threshold"] == EVIDENCE_DIRECTION]
    mv_rows = sub[sub["method"].str.startswith("MV ")]
    uv_rows = sub[sub["method"] == "UV non-conflicting"]

    out = {"mv_mcc": np.nan, "mv_accuracy": np.nan, "mv_auc": np.nan,
           "uv_mcc": np.nan, "uv_accuracy": np.nan, "uv_auc": np.nan,
           "n_pathogenic": 0, "n_benign": 0, "n_eval": 0}

    if not mv_rows.empty:
        best = mv_rows.loc[mv_rows["mcc"].idxmax()]
        out["mv_mcc"] = float(best["mcc"])
        out["mv_accuracy"] = float(best["accuracy"])
        out["mv_auc"] = float(best["auc"])
        out["n_pathogenic"] = int(best["n_pathogenic"])
        out["n_benign"] = int(best["n_benign"])
    if not uv_rows.empty:
        u = uv_rows.iloc[0]
        out["uv_mcc"] = float(u["mcc"])
        out["uv_accuracy"] = float(u["accuracy"])
        out["uv_auc"] = float(u["auc"])
    out["n_eval"] = int(sub["total"].iloc[0]) if not sub.empty else 0
    return out


def _gene_row(gene, gene_set, ms, results_json, dataset_name, aux_idx=None,
              cache_dir=None, panel=None):
    """(mv_mcc, uv_mcc, n_eval) for one gene, cached to disk so a crash or a
    bug fix only requires rerunning genes that never got a successful ("ok")
    result -- not the whole panel. `ms` may be None here ONLY when the
    caller already confirmed a cache hit upstream (see build_panel_a's
    LABEL-seq/plain-integrated blocks) to avoid needing to build it at all."""
    cached = _load_cached(cache_dir, panel, gene)
    if cached is not None and cached.get("status") == "ok" and _valid_cache_entry(cached):
        print(f"  [{gene}] cache hit (ok), skipping recompute")
        return _row_from_cache_record(cached)

    try:
        with fast_results_json(results_json):
            table, uv_names = build_comparison_table(
                gene, gene_set, ms, results_json,
                dataset_name=dataset_name, auxiliary_pathogenic_indices=aux_idx,
                modes=["trust_global"], compare_uv=True, **RUN_KWARGS,
            )
        metrics = _extract_mv_uv(table)
        if np.isnan(metrics["mv_mcc"]) or np.isnan(metrics["uv_mcc"]):
            reason = f"mv_mcc={metrics['mv_mcc']}, uv_mcc={metrics['uv_mcc']} (missing MV or UV data)"
            print(f"  [{gene}] SKIPPED from figure: {reason}")
            _save_cached(cache_dir, panel, gene, {"gene": gene, "status": "skipped", "reason": reason})
            return None
        record = {"gene": gene, "status": "ok", **metrics}
        _save_cached(cache_dir, panel, gene, record)
        return _row_from_cache_record(record)
    except Exception as e:
        print(f"  [{gene}] FAILED: {e}")
        _save_cached(cache_dir, panel, gene, {"gene": gene, "status": "failed", "reason": str(e)})
        return None


def _select_connected_datasets(df_integrated, gene, datasets, scoreset_kwargs, min_overlap_rows=30):
    """Reduce `datasets` down to the subset the real bootstrap fit was
    actually trained on, matching hpc/prepare.py's job-generation logic
    (Fit._select_calibration_dims: pairwise-overlap->=min_overlap_rows graph,
    keep the largest multi-dim connected component). Building the plain-
    integrated ms from ALL currently-available datasets for a gene (rather
    than the historically-selected subset) produces more dimensions than the
    stored fit's parameters have rows for -- confirmed via IndexError
    ("index 7 is out of bounds for axis 0 with size 4") for BRCA2, and via
    a direct pairwise-overlap check for KCNH2 (Jiang_2022<->
    O_Neill_2024_surface_expression overlap=43 clears 30; Kozek_Glazer_2020's
    overlap with either is only 1/18, so it's correctly excluded from the
    real 2D historical fit but was previously included here unconditionally).
    Returns the filtered (possibly empty) dataset list."""
    if len(datasets) <= 1:
        return datasets
    ms_full = build_multiscoreset_from_long_dataframe(
        df_integrated, gene, datasets, scoreset_kwargs=scoreset_kwargs,
    )
    if ms_full is None:
        return datasets
    kept_idx = Fit._select_calibration_dims(ms_full.scores, min_overlap_rows)
    if not kept_idx:
        return []
    return [ms_full.dataset_names[i] for i in kept_idx]


# ── Panel A: functional-only, 34 genes ───────────────────────────────────────

def build_panel_a(results_json, cache_dir=None):
    rows = []

    if _all_ok_cached(cache_dir, "A", ["TP53"]):
        rows.extend(_cached_rows(cache_dir, "A", ["TP53"]))
    else:
        # Default-collapse TP53's Kato_2003 panel (16 -> 10 dims), matching
        # hpc/prepare.py's TP53_DEFAULT_COLLAPSE_PRESET and
        # config.build_multiscoresets_for_gene_set("tp53")'s default --
        # TP53_tp53_mv's actual stored fit (e.g. under jobs_all_100b_3f_092026)
        # was trained on the collapsed space, so scoring the raw 16-dim ms
        # trips the fit-vs-dataset shape guard (confirmed: silently dropped
        # this row via _gene_row's own try/except before this fix).
        ms = config.build_multiscoresets_for_gene_set("tp53")["TP53"]
        r = _gene_row("TP53", "tp53", ms, results_json,
                      dataset_name="TP53_tp53_mv", aux_idx=_AUX_INDICES["tp53"],
                      cache_dir=cache_dir, panel="A")
        if r:
            rows.append(r)

    if _all_ok_cached(cache_dir, "A", ["CARD11"]):
        rows.extend(_cached_rows(cache_dir, "A", ["CARD11"]))
    else:
        ms = build_card11_multiscoreset()
        r = _gene_row("CARD11", "card11", ms, results_json,
                      dataset_name="CARD11_card11_mv", aux_idx=_AUX_INDICES["card11"],
                      cache_dir=cache_dir, panel="A")
        if r:
            rows.append(r)

    if _all_ok_cached(cache_dir, "A", ["FGFR_combined"]):
        rows.extend(_cached_rows(cache_dir, "A", ["FGFR_combined"]))
    else:
        # A real gap this was missing entirely (confirmed: zero "fgfr"/"FGFR"
        # mentions anywhere in this file before this fix) -- now that
        # uv_sources.load_fgfr_uv_points is wired up (config.FGFR_UV_CALIB_DIR),
        # FGFR can actually be plotted here, not just scored-and-dropped for
        # lack of a UV comparison value.
        ms = config.build_multiscoresets_for_gene_set("fgfr")["FGFR_combined"]
        r = _gene_row("FGFR_combined", "fgfr", ms, results_json,
                      dataset_name="FGFR_combined_fgfr_mv",
                      cache_dir=cache_dir, panel="A")
        if r:
            rows.append(r)

    if _all_ok_cached(cache_dir, "A", config.LABELSEQ_GENES):
        print("All LABEL-seq genes cached -- skipping the ~10min ms rebuild.")
        rows.extend(_cached_rows(cache_dir, "A", config.LABELSEQ_GENES))
    else:
        print("Building all LABEL-seq scoresets (one-time cost)...")
        labelseq_ms_map = build_labelseq_multiscoresets()
        for gene, ms in labelseq_ms_map.items():
            r = _gene_row(gene, "labelseq", ms, results_json,
                          dataset_name=gene_set_dataset_label(gene, "labelseq"),
                          cache_dir=cache_dir, panel="A")
            if r:
                rows.append(r)

    if _all_ok_cached(cache_dir, "A", PLAIN_INTEGRATED_GENES):
        print("All plain-integrated genes cached -- skipping dataframe reload.")
        rows.extend(_cached_rows(cache_dir, "A", PLAIN_INTEGRATED_GENES))
    else:
        print("Loading integrated dataframe for plain-integrated genes...")
        df_integrated = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
        for gene in PLAIN_INTEGRATED_GENES:
            cached = _load_cached(cache_dir, "A", gene)
            if cached is not None and cached.get("status") == "ok" and _valid_cache_entry(cached):
                rows.append(_row_from_cache_record(cached))
                continue
            datasets = sorted(
                d for d in df_integrated[df_integrated["Gene"] == gene]["Dataset"].unique()
                if d not in META_ANALYSIS_DATASETS
            )
            if not datasets:
                print(f"  [{gene}] no functional datasets in integrated dataframe, skipping")
                continue
            scoreset_kwargs = dict(
                clinvar_release=resolve_clinvar_release(gene),
                min_clinvar_star=1, population_type="gnomAD",
            )
            # Reduce to the historically-fitted dimension subset (see
            # _select_connected_datasets) -- the raw `datasets` list above is
            # every dataset currently in the dataframe for this gene, which
            # can include more dimensions than the stored bootstrap fit was
            # actually trained on (confirmed for BRCA2: 8 datasets now vs. a
            # 4-dim stored fit -> IndexError; KCNH2: Kozek_Glazer_2020 has
            # negligible overlap with the other 2 datasets and was correctly
            # excluded from the real 2D fit).
            selected = _select_connected_datasets(
                df_integrated, gene, datasets, scoreset_kwargs, min_overlap_rows=30)
            if not selected:
                print(f"  [{gene}] no dataset pair clears the overlap threshold, skipping")
                continue
            if set(selected) != set(datasets):
                print(f"  [{gene}] dimension-selection reduced {len(datasets)} -> "
                      f"{len(selected)} datasets: {selected}")
            datasets = selected
            # Canonical builder for the plain-"integrated" gene-set (matches
            # hpc/prepare.py::_process_multivariate_gene exactly, which is
            # what these genes' MV fits were actually trained on) -- NOT
            # combined.py::build_functional_scoresets (that one lacks the
            # min_samples=2 per-dataset filter and doesn't exclude the two
            # known meta-analysis datasets, which produced a dimension
            # mismatch against the real fit for at least F9 -- confirmed
            # "0/1000 valid bootstraps" for BRCA2/F9/KCNH2 with the old
            # builder).
            ms = build_multiscoreset_from_long_dataframe(
                df_integrated, gene, datasets, scoreset_kwargs=scoreset_kwargs,
            )
            if ms is None:
                print(f"  [{gene}] fewer than 2 usable dimensions, skipping")
                continue
            try:
                dataset_name = _find_dataset_key(results_json, gene)
            except KeyError as e:
                print(f"  [{gene}] {e}")
                continue
            r = _gene_row(gene, "integrated-functional", ms, results_json,
                          dataset_name=dataset_name, cache_dir=cache_dir, panel="A")
            if r:
                rows.append(r)

    return pd.DataFrame(rows)


# ── Panel B: computational predictors only, 8 genes ──────────────────────────

def build_panel_b(results_json, cache_dir=None):
    rows = []
    for gene in COMBINED_GENES:
        cached = _load_cached(cache_dir, "B", gene)
        if cached is not None and cached.get("status") == "ok" and _valid_cache_entry(cached):
            rows.append(_row_from_cache_record(cached))
            continue
        try:
            ms = load_predictor_ms(gene, config.PREDICTOR_RAW_DATA_DIR)
        except ValueError as e:
            print(f"  [{gene}] could not build predictor ms: {e}")
            continue
        r = _gene_row(gene, "predictor-mv", ms, results_json,
                      dataset_name=predictor_dataset_label(gene),
                      cache_dir=cache_dir, panel="B")
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


# ── Panel C: combined functional+predictor evidence, 8 genes ────────────────

def build_panel_c(results_json, cache_dir=None):
    rows = []
    if _all_ok_cached(cache_dir, "C", COMBINED_GENES):
        print("All combined-evidence genes cached -- skipping dataframe reload.")
        return pd.DataFrame(_cached_rows(cache_dir, "C", COMBINED_GENES))

    print("Loading integrated dataframe for combined genes...")
    df_integrated = pd.read_csv(DEFAULT_INTEGRATED_DATAFRAME, sep="\t", low_memory=False)
    for gene in COMBINED_GENES:
        cached = _load_cached(cache_dir, "C", gene)
        if cached is not None and cached.get("status") == "ok" and _valid_cache_entry(cached):
            rows.append(_row_from_cache_record(cached))
            continue
        datasets = sorted(df_integrated[df_integrated["Gene"] == gene]["Dataset"].unique())
        if not datasets:
            print(f"  [{gene}] no functional datasets, skipping")
            continue
        functional_scoresets = build_functional_scoresets(
            df_integrated, gene, datasets, clinvar_release=resolve_clinvar_release(gene))
        functionally_assayed = get_functionally_assayed_protein_variants(df_integrated, gene, datasets)
        ms = build_combined_multiscoreset(
            gene, functional_scoresets, datasets, config.PREDICTOR_RAW_DATA_DIR,
            functionally_assayed_variants=functionally_assayed)
        if ms is None:
            print(f"  [{gene}] could not build combined ms, skipping")
            continue
        # Panel C's UV baseline: functional + predictor UV merged into ONE
        # non-conflicting aggregate (not two separate ones) -- see
        # uv_sources.load_combined_all_evidence_uv_points.
        r = _gene_row(gene, "combined-all-evidence", ms, results_json,
                      dataset_name=gene_set_dataset_label(gene, "combined"),
                      cache_dir=cache_dir, panel="C")
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

_PALETTE_CMAP = LinearSegmentedColormap.from_list(
    "excalibr_mv", ["#9FBCE6", "#4C72B0", "#0B1F4B"])


_METRIC_AXIS_LABEL = {"mcc": "MCC", "accuracy": "Accuracy", "auc": "AUC (ROC)"}


def plot_metric_scatter_panel(ax, df, letter, title, metric="mcc", ymin=None,
                               slides=False, color=None):
    """Generic version of plot_mcc_scatter_panel supporting mcc/accuracy/auc.

    ``slides=True``: a simplified, large-font rendering for a slideshow
    (not the paper) -- uniform dot size (no N-control-variants size
    encoding, no legend at all), a single flat panel color (``color``,
    defaulting to _SLIDES_PANEL_COLOR[letter.strip("()")] when not given)
    instead of the paper version's size-only color, larger gene-name labels,
    and axis labels renamed to "Independent calibration" (x, i.e. UV/
    univariate) vs. "Multidimensional calibration" (y, i.e. MV) instead of
    the paper version's method-name/metric-name labels.

    mcc/auc are mathematically undefined when the evaluated set has zero
    P/LP or zero B/LB variants (compute_classification_metrics/roc_auc_score
    can't compute sensitivity/specificity/AUC against an absent class) --
    genes in that state get silently assigned a fallback value (0.0 for MCC,
    NaN for AUC already handled by report.py) rather than erroring, so they
    must be filtered out here by checking the actual class counts
    (n_pathogenic/n_benign), not by pattern-matching the fallback value
    itself (the old "both MCCs are exactly 0" heuristic this replaced missed
    asymmetric cases, e.g. a gene with mv_mcc=1.0 but uv_mcc=0.0 where the UV
    side's denominator -- not the MV side's -- was the one undefined).
    Accuracy has no such degeneracy (well-defined with only one class
    present, e.g. "100% accuracy" for an all-pathogenic gene the classifier
    calls entirely correctly) and is NOT filtered this way -- this is
    exactly why a separate accuracy panel can include genes the mcc/auc
    panels must exclude."""
    mv_col, uv_col = f"mv_{metric}", f"uv_{metric}"
    if not df.empty:
        df = df.dropna(subset=[mv_col, uv_col])
        if metric in ("mcc", "auc"):
            df = df[(df["n_pathogenic"] > 0) & (df["n_benign"] > 0)]

    if df.empty:
        ax.set_title(f"{title}\n(no data)", fontsize=11)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    lo = max(0.0, min(df[mv_col].min(), df[uv_col].min()) - 0.05)
    hi = 1.02
    diag_line = ax.plot([lo, hi], [lo, hi], "k--", alpha=0.35,
                         linewidth=2.0 if slides else 1.5,
                         zorder=1, label="Equal performance")[0]

    panel_color = color or _SLIDES_PANEL_COLOR.get(letter.strip("()"), "#4C72B0")

    if slides:
        # No size encoding at all -- every dot the same size, one flat color,
        # no legend (see docstring).
        ax.scatter(df[uv_col], df[mv_col], s=480, c=panel_color,
                   edgecolors="white", alpha=0.9, linewidth=2.0, zorder=3)
    else:
        # Size is the only encoding (was redundantly duplicated by color
        # before); a single solid color avoids implying a second variable is
        # shown.
        n_vals = df["n_eval"].values
        size_min, size_max = n_vals.min(), n_vals.max()

        def _size_for(v):
            return 60 + 500 * (np.sqrt(v) - np.sqrt(size_min)) / max(1e-9, (np.sqrt(size_max) - np.sqrt(size_min)))

        size = _size_for(n_vals)
        ax.scatter(df[uv_col], df[mv_col], s=size, c=panel_color,
                   edgecolors="white", alpha=0.85, linewidth=1.5, zorder=3)

    # Dense panels (e.g. functional, ~34 genes) need a smaller label size than
    # sparse ones (predictor/combined, ~8 genes) to avoid the labels
    # overlapping each other or getting shoved far outside the axes by
    # adjustText's repulsion -- confirmed this happens at a flat fontsize=15
    # for panel A specifically.
    if slides:
        label_fontsize = 19 if len(df) <= 15 else 12
    else:
        label_fontsize = 8
    label_stroke = 3.5 if slides else 2.25
    texts = []
    for _, row in df.iterrows():
        # clip_on=True: Text, unlike the scatter dots above, does NOT clip to
        # the axes by default -- a gene whose point falls below panel A's
        # explicit ymin=0.6 floor (e.g. mv_mcc=0.0) had its dot correctly
        # hidden but its label still rendered, floating with no visible
        # anchor below the plot (confirmed for ASPA/CBS/PAX6).
        gene_label = str(row["gene"]).upper()
        if slides:
            gene_label = _SLIDES_GENE_LABEL.get(gene_label, gene_label)
        t = ax.text(row[uv_col], row[mv_col], gene_label, fontsize=label_fontsize,
                    ha="center", va="center", fontweight="bold", zorder=10, clip_on=True)
        t.set_path_effects([pe.Stroke(linewidth=label_stroke, foreground="white"), pe.Normal()])
        texts.append(t)
    if adjust_text is not None and texts:
        try:
            # only_move + a expand/force cap keeps adjustText from shoving
            # labels arbitrarily far from their point when a panel is dense
            # (confirmed this happened at the default settings for the
            # ~34-gene functional panel in slides mode -- several labels
            # landed entirely outside the axes).
            adjust_text(texts, ax=ax, expand=(1.15, 1.3), force_text=(0.3, 0.5),
                        force_static=(0.2, 0.3),
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.5))
        except Exception:
            pass

    if slides:
        ax.set_xlabel("Independent calibration", fontsize=26, fontweight="bold", labelpad=12)
        ax.set_ylabel("Multidimensional calibration", fontsize=26, fontweight="bold", labelpad=12)
    else:
        axis_label = _METRIC_AXIS_LABEL[metric]
        ax.set_xlabel(f"ExCALIBR (UV, non-conflicting) {axis_label}", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"ExCALIBR-MV {axis_label}", fontsize=11, fontweight="bold")
    # Slides mode pads the visible range past `lo`/`hi` (beyond the diagonal
    # line's own extent) so a big-font label on a gene at/near the 0.0 or 1.0
    # boundary has room and doesn't get mid-word clipped by clip_on=True
    # above (confirmed both edges needed it: "MRAS"/"EGFR" at x=0 clipped on
    # the left, "SCN5A" near mv=1.0 clipped on the right/top).
    lo_axis = lo - 0.05 if slides else lo
    hi_axis = hi + 0.045 if slides else hi
    ax.set_xlim(lo_axis, hi_axis)
    ax.set_ylim(lo_axis if ymin is None else ymin, hi_axis)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if slides:
        # Thin dotted reference lines at 0 and 1 -- the theoretical
        # floor/ceiling of MCC/accuracy/AUC -- distinct from the dashed
        # diagonal "equal performance" line above.
        for v in (0.0, 1.0):
            if lo_axis < v < hi_axis:
                ax.axvline(v, color="#999999", linestyle=":", linewidth=1.3, alpha=0.6, zorder=0)
                ax.axhline(v, color="#999999", linestyle=":", linewidth=1.3, alpha=0.6, zorder=0)
    if slides:
        ax.tick_params(axis="both", labelsize=20)
        ax.set_title(title, fontsize=28, fontweight="bold", pad=18)
    else:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="top", ha="left")

    if not slides:
        # One legend: the diagonal reference line plus size-encoding min/max markers.
        size_handles = [
            diag_line,
            Line2D([0], [0], marker="o", linestyle="", markersize=np.sqrt(_size_for(size_min)) / 2,
                   markerfacecolor=panel_color, markeredgecolor="white", label=f"N={int(size_min):,}"),
            Line2D([0], [0], marker="o", linestyle="", markersize=np.sqrt(_size_for(size_max)) / 2,
                   markerfacecolor=panel_color, markeredgecolor="white", label=f"N={int(size_max):,}"),
        ]
        ax.legend(handles=size_handles, title="N control variants (size)", loc="lower right",
                  frameon=True, edgecolor="#999", framealpha=0.95, fontsize=7, title_fontsize=7)


def plot_mcc_scatter_panel(ax, df, letter, title, ymin=None):
    """Back-compat wrapper for callers whose df only ever has mv_mcc/uv_mcc/
    n_eval (e.g. mv_cockpit.run_gene_performance_scatter, built via
    run_results_table rather than _gene_row) -- no n_pathogenic/n_benign
    columns are available there, so this keeps the older (imperfect, see
    plot_metric_scatter_panel's docstring) both-exactly-zero heuristic
    rather than erroring on missing columns."""
    if not df.empty:
        df = df.dropna(subset=["mv_mcc", "uv_mcc"])
        df = df[~((df["mv_mcc"] == 0) & (df["uv_mcc"] == 0))]
        df = df.assign(n_pathogenic=1, n_benign=1)  # bypass plot_metric_scatter_panel's own filter
    plot_metric_scatter_panel(ax, df, letter, title, metric="mcc", ymin=ymin)


def build_gene_performance_figure(results_json, save_path=None, cache_dir=None, slides=False):
    """cache_dir, if given, persists each gene's full metrics record (mcc/
    accuracy/auc for MV+UV, n_pathogenic/n_benign/n_eval) to disk as it's
    computed -- a crash, a bug fix, or an interrupted run only requires
    rerunning genes that don't already have a successful ("ok") cache entry,
    not the whole panel/script. See _gene_row/_cache_path.

    Builds THREE figures, not one: MCC (save_path, the original/default),
    plus accuracy and AUC siblings saved alongside it (same directory,
    "_accuracy"/"_auc" inserted before the extension). MCC and AUC panels
    exclude genes with zero P/LP or zero B/LB in the evaluated set (both are
    mathematically undefined there); the accuracy panel does not, since
    accuracy has no such degeneracy -- see plot_metric_scatter_panel's
    docstring for why these need different filtering, not just different
    y-axes on the same data.

    ``slides=True``: simplified, large-font slideshow rendering (see
    plot_metric_scatter_panel's docstring) -- one flat color per panel
    (_SLIDES_PANEL_COLOR), uniform dot size, no legend, renamed axis labels.
    Not for the paper; a separate rendering pass, same underlying data."""
    print("=== Panel A: functional ===")
    df_a = build_panel_a(results_json, cache_dir=cache_dir)
    print("=== Panel B: computational predictors ===")
    df_b = build_panel_b(results_json, cache_dir=cache_dir)
    print("=== Panel C: combined functional+predictor ===")
    df_c = build_panel_c(results_json, cache_dir=cache_dir)

    panels = [("(A)", "Functional", df_a, 0.6), ("(B)", "Computational predictors", df_b, None),
              ("(C)", "Combined evidence", df_c, None)]

    figures = {}
    for metric in ("mcc", "accuracy", "auc"):
        fig, axes = plt.subplots(1, 3, figsize=(28, 8.5) if slides else (18, 6))
        for ax, (letter, title, df, ymin) in zip(axes, panels):
            plot_metric_scatter_panel(ax, df, letter, title, metric=metric, ymin=ymin, slides=slides)
        if slides:
            # Explicit spacing instead of tight_layout -- at the larger
            # slides fontsize, tight_layout let an axis label from one panel
            # bleed into its neighbor's plot area.
            fig.subplots_adjust(wspace=0.18, left=0.05, right=0.98, bottom=0.14, top=0.94)
        else:
            plt.tight_layout()
        figures[metric] = fig

        if save_path:
            p = Path(save_path)
            stem = f"{p.stem}_slides" if slides else p.stem
            metric_path = p.with_name(f"{stem}{p.suffix}") if metric == "mcc" else p.with_name(f"{stem}_{metric}{p.suffix}")
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(metric_path, dpi=300, bbox_inches="tight")
            print(f"Saved {metric} figure to {metric_path}")

    for name, df in [("Functional", df_a), ("Predictors", df_b), ("Combined", df_c)]:
        if df.empty:
            print(f"{name}: no data")
            continue
        print(f"{name} (n={len(df)}): mean UV MCC {df['uv_mcc'].mean():.3f} -> "
              f"mean MV MCC {df['mv_mcc'].mean():.3f}")
        print(df.sort_values("gene").to_string(index=False))

    return figures["mcc"], {"A": df_a, "B": df_b, "C": df_c}


# ── Panel D: TP53 RPV penetrance-score distribution ─────────────────────────

_RPV_COLORS = {"P/LP": "#943744", "P/LP indeterminate": "#e05c00", "RPV": "#2E7D4F"}


def plot_rpv_penetrance_panel(results_json, config_name="6c_unc", axes=None, save_path=None):
    """P/LP (with its indeterminate subset overlaid on the SAME axes, not a
    separate row) and RPV penetrance-score distributions -- no B/LB, per the
    user's request. Reuses MVCalibrationAnalysis.score_rpv_penetrance (already
    used in real TP53 reports via report_gene.py)."""
    ms = build_tp53_multiscoreset()
    with fast_results_json(results_json):
        analysis = build_gene_set_analysis(
            ms, "TP53", results_json, dataset_name="TP53_tp53_mv",
            auxiliary_pathogenic_indices=[4],
        )
        analysis.run(partial_pattern_mode="trust_global", aux_path_percentile=50,
                     aux_ben_percentile=50, **RUN_KWARGS)
        rpv_scores = analysis.score_rpv_penetrance(config_name, fixed_idx=4)

    # Raw, unfiltered sample_assignments (fixed role indices: P/LP=0, B/LB=1,
    # gnomAD=2, Synonymous=3, RPV=4) -- NOT the .sample_assignments property,
    # which drops empty-count columns (here, Synonymous has 0 observations for
    # this build) and silently shifts RPV from raw index 4 down to effective
    # index 3, as MVCalibrationAnalysis's own "effective indices: [3]" log
    # line for this run confirms.
    sa = ms._sample_assignments
    plp_mask = sa[:, 0].astype(bool)
    points = analysis.results[config_name]["points"]
    indet_mask = plp_mask & (points <= 0)
    rpv_mask = sa[:, 4].astype(bool)

    own_fig = axes is None
    if own_fig:
        fig, axes = plt.subplots(2, 1, figsize=(6, 4.4), sharex=True)
    else:
        fig = axes[0].figure

    plp_scores = rpv_scores.iloc[np.where(plp_mask)[0]]["penetrance_score"].dropna().values
    indet_scores = rpv_scores.iloc[np.where(indet_mask)[0]]["penetrance_score"].dropna().values
    rpv_scores_vals = rpv_scores.iloc[np.where(rpv_mask)[0]]["penetrance_score"].dropna().values

    ax = axes[0]
    ax.hist(plp_scores, bins=20, range=(0, 1), color=_RPV_COLORS["P/LP"],
            alpha=0.75, edgecolor="white", linewidth=0.5, label="P/LP (all)")
    ax.hist(indet_scores, bins=20, range=(0, 1), color=_RPV_COLORS["P/LP indeterminate"],
            alpha=0.75, edgecolor="white", linewidth=0.5, label="P/LP (indeterminate)")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_facecolor("#F9F9F9")
    ax.legend(fontsize=8, frameon=True, loc="upper right")
    ax.text(0.02, 0.95, f"P/LP\n{len(plp_scores)} total, {len(indet_scores)} indet.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color=_RPV_COLORS["P/LP"], fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.hist(rpv_scores_vals, bins=20, range=(0, 1), color=_RPV_COLORS["RPV"],
            alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_xlabel("Penetrance score (0 = low-penetrance/RPV-like, 1 = high-penetrance/P/LP-like)", fontsize=9)
    ax.set_facecolor("#F9F9F9")
    ax.text(0.02, 0.95, f"RPV\n{len(rpv_scores_vals)}/{int(rpv_mask.sum())}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color=_RPV_COLORS["RPV"], fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if own_fig:
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved RPV panel to {save_path}")

    return fig, axes


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--save-path", default=None)
    ap.add_argument("--rpv-save-path", default=None)
    ap.add_argument("--cache-dir", default=None,
                     help="Per-gene result cache dir. Genes already cached with "
                          "status='ok' are loaded instantly instead of recomputed; "
                          "'failed'/'skipped' genes (or a bug fix that should change "
                          "their outcome) are retried automatically. Rerunning this "
                          "script with the same --cache-dir after an interruption or "
                          "a code fix only redoes the genes that need it.")
    args = ap.parse_args()

    build_gene_performance_figure(args.results_json, save_path=args.save_path, cache_dir=args.cache_dir)
    plot_rpv_penetrance_panel(args.results_json, save_path=args.rpv_save_path)
