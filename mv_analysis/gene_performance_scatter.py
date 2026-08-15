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


def _row_from_cache_record(c):
    return {"gene": c["gene"], "mv_mcc": c["mv_mcc"], "uv_mcc": c["uv_mcc"], "n_eval": c["n_eval"]}


def _all_ok_cached(cache_dir, panel, genes):
    """True if every gene in `genes` already has ANY cache entry (ok, failed,
    or skipped) -- lets callers skip an expensive bulk ms-build (e.g.
    LABEL-seq's 17-gene, ~10min build_labelseq_multiscoresets()) entirely
    when nothing in that block actually needs (re)computing. A "failed"
    status is treated as settled, not auto-retried, on the assumption that a
    structural cause (e.g. a gene missing P/LP entirely) won't change on its
    own -- delete that gene's specific cache file to force a retry after a
    code fix that might actually resolve it."""
    if cache_dir is None:
        return False
    return all(_load_cached(cache_dir, panel, g) is not None for g in genes)


def _cached_rows(cache_dir, panel, genes):
    rows = []
    for g in genes:
        c = _load_cached(cache_dir, panel, g)
        if c and c.get("status") == "ok":
            rows.append(_row_from_cache_record(c))
    return rows


def _extract_mv_uv(table):
    """(mv_mcc, uv_mcc, n_eval) from a build_comparison_table() result, at the
    evidence-direction threshold, MV = best MCC across configs, UV = non-conflicting
    only (the session-standardized rule)."""
    sub = table[table["threshold"] == EVIDENCE_DIRECTION]
    mv_rows = sub[sub["method"].str.startswith("MV ")]
    uv_rows = sub[sub["method"] == "UV non-conflicting"]
    mv_mcc = float(mv_rows["mcc"].max()) if not mv_rows.empty else np.nan
    uv_mcc = float(uv_rows["mcc"].iloc[0]) if not uv_rows.empty else np.nan
    n_eval = int(sub["total"].iloc[0]) if not sub.empty else 0
    return mv_mcc, uv_mcc, n_eval


def _gene_row(gene, gene_set, ms, results_json, dataset_name, aux_idx=None,
              cache_dir=None, panel=None):
    """(mv_mcc, uv_mcc, n_eval) for one gene, cached to disk so a crash or a
    bug fix only requires rerunning genes that never got a successful ("ok")
    result -- not the whole panel. `ms` may be None here ONLY when the
    caller already confirmed a cache hit upstream (see build_panel_a's
    LABEL-seq/plain-integrated blocks) to avoid needing to build it at all."""
    cached = _load_cached(cache_dir, panel, gene)
    if cached is not None and cached.get("status") == "ok":
        print(f"  [{gene}] cache hit (ok), skipping recompute")
        return _row_from_cache_record(cached)

    try:
        with fast_results_json(results_json):
            table, uv_names = build_comparison_table(
                gene, gene_set, ms, results_json,
                dataset_name=dataset_name, auxiliary_pathogenic_indices=aux_idx,
                modes=["trust_global"], compare_uv=True, **RUN_KWARGS,
            )
        mv_mcc, uv_mcc, n_eval = _extract_mv_uv(table)
        if np.isnan(mv_mcc) or np.isnan(uv_mcc):
            reason = f"mv_mcc={mv_mcc}, uv_mcc={uv_mcc} (missing MV or UV data)"
            print(f"  [{gene}] SKIPPED from figure: {reason}")
            _save_cached(cache_dir, panel, gene, {"gene": gene, "status": "skipped", "reason": reason})
            return None
        record = {"gene": gene, "status": "ok", "mv_mcc": mv_mcc, "uv_mcc": uv_mcc, "n_eval": n_eval}
        _save_cached(cache_dir, panel, gene, record)
        return _row_from_cache_record(record)
    except Exception as e:
        print(f"  [{gene}] FAILED: {e}")
        _save_cached(cache_dir, panel, gene, {"gene": gene, "status": "failed", "reason": str(e)})
        return None


# ── Panel A: functional-only, 34 genes ───────────────────────────────────────

def build_panel_a(results_json, cache_dir=None):
    rows = []

    if _all_ok_cached(cache_dir, "A", ["TP53"]):
        rows.extend(_cached_rows(cache_dir, "A", ["TP53"]))
    else:
        ms = build_tp53_multiscoreset()
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
            if cached is not None and cached.get("status") == "ok":
                rows.append(_row_from_cache_record(cached))
                continue
            datasets = sorted(
                d for d in df_integrated[df_integrated["Gene"] == gene]["Dataset"].unique()
                if d not in META_ANALYSIS_DATASETS
            )
            if not datasets:
                print(f"  [{gene}] no functional datasets in integrated dataframe, skipping")
                continue
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
                df_integrated, gene, datasets,
                scoreset_kwargs=dict(
                    clinvar_release=resolve_clinvar_release(gene),
                    min_clinvar_star=1, population_type="gnomAD",
                ),
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
        if cached is not None and cached.get("status") == "ok":
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
        if cached is not None and cached.get("status") == "ok":
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


def plot_mcc_scatter_panel(ax, df, letter, title, ymin=None):
    # Never plot a gene missing either MCC (defensive -- _gene_row already
    # excludes these before caching, but don't trust that silently). Also
    # drop genes where BOTH MCCs are exactly 0 -- this isn't a real "zero
    # concordance" result, it's compute_classification_metrics's fallback
    # value when the denominator is undefined (e.g. PAX6: only 2 B/LB
    # variants total, so there's ~no negative-class coverage to compute
    # specificity/MCC against on either the MV or UV side -- confirmed via
    # its cached record and the run log's "Benign+Syn: 0.0% correct" line).
    if not df.empty:
        df = df.dropna(subset=["mv_mcc", "uv_mcc"])
        df = df[~((df["mv_mcc"] == 0) & (df["uv_mcc"] == 0))]

    if df.empty:
        ax.set_title(f"{title}\n(no data)", fontsize=11)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    lo = max(0.0, min(df["mv_mcc"].min(), df["uv_mcc"].min()) - 0.05)
    hi = 1.02
    diag_line = ax.plot([lo, hi], [lo, hi], "k--", alpha=0.35, linewidth=1.5,
                         zorder=1, label="Equal performance")[0]

    # Size is the only encoding (was redundantly duplicated by color before);
    # a single solid color avoids implying a second variable is shown.
    n_vals = df["n_eval"].values
    size_min, size_max = n_vals.min(), n_vals.max()

    def _size_for(v):
        return 60 + 500 * (np.sqrt(v) - np.sqrt(size_min)) / max(1e-9, (np.sqrt(size_max) - np.sqrt(size_min)))

    size = _size_for(n_vals)

    ax.scatter(df["uv_mcc"], df["mv_mcc"], s=size, c="#4C72B0",
               edgecolors="white", alpha=0.85, linewidth=1.5, zorder=3)

    texts = []
    for _, row in df.iterrows():
        t = ax.text(row["uv_mcc"], row["mv_mcc"], str(row["gene"]).upper(), fontsize=8,
                    ha="center", va="center", fontweight="bold", zorder=10)
        t.set_path_effects([pe.Stroke(linewidth=2.25, foreground="white"), pe.Normal()])
        texts.append(t)
    if adjust_text is not None and texts:
        try:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.5))
        except Exception:
            pass

    ax.set_xlabel("ExCALIBR (UV, non-conflicting) MCC", fontsize=11, fontweight="bold")
    ax.set_ylabel("ExCALIBR-MV MCC", fontsize=11, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(ymin if ymin is not None else lo, hi)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="top", ha="left")

    # One legend: the diagonal reference line plus size-encoding min/max markers.
    size_handles = [
        diag_line,
        Line2D([0], [0], marker="o", linestyle="", markersize=np.sqrt(_size_for(size_min)) / 2,
               markerfacecolor="#4C72B0", markeredgecolor="white", label=f"N={int(size_min):,}"),
        Line2D([0], [0], marker="o", linestyle="", markersize=np.sqrt(_size_for(size_max)) / 2,
               markerfacecolor="#4C72B0", markeredgecolor="white", label=f"N={int(size_max):,}"),
    ]
    ax.legend(handles=size_handles, title="N control variants (size)", loc="lower right",
              frameon=True, edgecolor="#999", framealpha=0.95, fontsize=7, title_fontsize=7)


def build_gene_performance_figure(results_json, save_path=None, cache_dir=None):
    """cache_dir, if given, persists each gene's (mv_mcc, uv_mcc, n_eval) to
    disk as it's computed -- a crash, a bug fix, or an interrupted run only
    requires rerunning genes that don't already have a successful ("ok")
    cache entry, not the whole panel/script. See _gene_row/_cache_path."""
    print("=== Panel A: functional ===")
    df_a = build_panel_a(results_json, cache_dir=cache_dir)
    print("=== Panel B: computational predictors ===")
    df_b = build_panel_b(results_json, cache_dir=cache_dir)
    print("=== Panel C: combined functional+predictor ===")
    df_c = build_panel_c(results_json, cache_dir=cache_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_mcc_scatter_panel(axes[0], df_a, "(A)", "Functional", ymin=0.6)
    plot_mcc_scatter_panel(axes[1], df_b, "(B)", "Computational predictors")
    plot_mcc_scatter_panel(axes[2], df_c, "(C)", "Combined evidence")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    for name, df in [("Functional", df_a), ("Predictors", df_b), ("Combined", df_c)]:
        if df.empty:
            print(f"{name}: no data")
            continue
        print(f"{name} (n={len(df)}): mean UV MCC {df['uv_mcc'].mean():.3f} -> "
              f"mean MV MCC {df['mv_mcc'].mean():.3f}")
        print(df.sort_values("gene").to_string(index=False))

    return fig, {"A": df_a, "B": df_b, "C": df_c}


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
