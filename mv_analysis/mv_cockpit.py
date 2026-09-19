# ---
# jupytext:
#   text_representation:
#     extension: .py
#     format_name: percent
#     format_version: '1.3'
#     jupytext_version: 1.19.4
#   kernelspec:
#     display_name: excalibr
#     language: python
#     name: excalibr
# ---

# %% [markdown]
# # MV cockpit
#
# One entry point for the multivariate (MV) calibration pipeline: load an
# existing fit (canonical or the experimental staged-init all_assayed
# variant) OR launch a new one, tune hyperparameters, and run any
# combination of analyses (gene-performance scatter, 3D interactive evidence
# viz, UV comparison, results table, VUS-reclassification Sankeys) for a
# chosen gene-set/gene.
#
# All results-provenance (which JSON, which dataset key, which MultiScoreset
# builder) is resolved through `mv_analysis/config.py`'s registry -- nothing
# here hardcodes a path. `tavtigian_sims/` is NOT a dependency anywhere in
# this file (see `mv_analysis/sankey_plot.py`, vendored out of it).
#
# Run as a plain script (`python mv_analysis/mv_cockpit.py --gene-set
# labelseq --fit-type canonical --analyses results-table`) for the CLI, or
# open in Jupyter (`jupytext --sync mv_cockpit.py` first) for the
# cells-mode workflow below -- same dual-mode pattern as
# `analysis/analyze_pipeline_output.py`.
#
# ## Status (be honest about what's wired up vs. scaffolded)
# - **Working**: load-existing-fit path (canonical + staged_init_all_assayed),
#   registry-driven; `results-table`; `uv-comparison`; `gene-performance-scatter`
#   (labelseq/integrated); `evidence-3d` (generic, no gene-specific aux
#   disease groups); `vus-sankey` (shells out to the already-built
#   `analysis/run_vus_reclassification.py` + `analysis/
#   plot_vus_reclassification_sankey.py`, labelseq RASopathy genes only,
#   matching that module's current scope).
# - **Launch-new-fit path**: wired to `hpc/prepare.py`'s `p_multi` subcommand
#   + `hpc/run_local_array.sh`, but `p_multi` does not yet expose
#   `--regularization-type`/staged-init -- that flag needs adding to
#   `hpc/prepare.py` before this path can launch all_assayed fits (canonical
#   fits can already be launched today).
# - **Not yet wired**: `confusion-matrices` (analysis/clingen.py's real
#   builder needs the production `tree`/`model_selections` harness from
#   analysis/discovery.py, deliberately NOT reused this session for the VUS
#   work due to its complexity -- still open); `brnich-comparison` (needs
#   the LABELseq_brnich.csv join, ported from the user-supplied cell but not
#   yet added here).

# %%
import sys
from pathlib import Path


def _running_as_notebook() -> bool:
    """True when executed by Jupyter/IPython rather than `python file.py`."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
if _running_as_notebook():
    try:
        matplotlib.use("module://matplotlib_inline.backend_inline")
    except ImportError:
        pass
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mv_analysis import config
from mv_analysis.report import build_comparison_table
from mv_analysis.gene_performance_scatter import plot_mcc_scatter_panel, fast_results_json
from src.assay_calibration.multivariate_analysis.gene_set_analysis import build_gene_set_analysis
from src.assay_calibration.fit_utils.fit import Fit

DEFAULT_RUN_KWARGS = dict(
    path_percentile=5, min_valid_boots=1, reestimate_marginal_weights=False,
    enforce_marginal_monotonicity=False, liberal_marginal_monotonicity=False,
)


# %% [markdown]
# ## Registry-driven fit resolution

# %%
def resolve_fit(gene_set: str, fit_type: str, gene: str, n_components: int = 6):
    """Returns (results_json_path, dataset_name, config_hint) for one gene.
    `config_hint`, if not None, is the exact config string to use (staged-init
    fits only ever have one config per file); canonical fits have several
    ("Nc_unc") and the caller picks the best-MCC one itself (see
    `pick_best_canonical_config`)."""
    if fit_type == "canonical":
        return config.CANONICAL_RESULTS_JSON, config.canonical_dataset_name(gene, gene_set), None
    if fit_type == "staged_init_all_assayed":
        return (config.staged_init_results_path(gene, gene_set, n_components),
                config.staged_init_dataset_name(gene, gene_set),
                config.staged_init_config_label(n_components))
    raise ValueError(f"Unknown fit_type={fit_type!r}. Expected 'canonical' or 'staged_init_all_assayed'.")


def pick_best_canonical_config(gene: str, gene_set: str, ms, results_json: str, dataset_name: str,
                                run_kwargs: dict) -> str:
    import gzip, json
    with fast_results_json(results_json):
        with gzip.open(results_json, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        configs = sorted(raw[dataset_name]["0"].keys())
        table, _ = build_comparison_table(
            gene.lower(), gene_set, ms, results_json, dataset_name=dataset_name,
            modes=["trust_global"], compare_uv=False, **run_kwargs,
        )
        clinical = table[table["threshold"].str.startswith("clinical")]
        best_config, best_mcc = None, -1
        for cfg in configs:
            row = clinical[clinical["config"] == cfg]
            if row.empty:
                continue
            mcc = row["mcc"].max()
            if mcc is not None and mcc > best_mcc:
                best_mcc, best_config = mcc, cfg
        return best_config or configs[0]


def load_gene(gene_set: str, fit_type: str, gene: str, n_components: int = 6,
              run_kwargs: dict = None, regularization_type: str = None,
              redundancy_collapse_preset: str = None):
    """Build the MultiScoreset, resolve the fit config, and run
    `MVCalibrationAnalysis` for one gene. Returns (ms, analysis, config_name).
    The fit-vs-dataset shape guard in `MVCalibrationAnalysis.__init__` will
    raise/warn immediately if the resolved fit no longer matches the
    freshly-built ms (see mv_calibration.py's `_validate_fit_shape`).

    `redundancy_collapse_preset` (e.g. "tp53_kato_pca2") CHANGES the ms's
    dimensionality -- only meaningful when scoring a fit that was actually
    trained on the collapsed space (via hpc/prepare.py's
    --redundancy-collapse-preset launching that fit). Applying it while
    loading an EXISTING fit trained on the raw/uncollapsed dims will
    correctly trip the shape guard above (dimension count mismatch) rather
    than silently scoring against the wrong space."""
    run_kwargs = run_kwargs or DEFAULT_RUN_KWARGS
    if regularization_type is None and fit_type == "staged_init_all_assayed":
        regularization_type = "all_assayed"
    ms_map = config.build_multiscoresets_for_gene_set(
        gene_set, genes=[gene], regularization_type=regularization_type,
        redundancy_collapse_preset=redundancy_collapse_preset)
    gene_key = gene.lower() if gene_set == "labelseq" else gene.upper()
    ms = ms_map.get(gene_key) or ms_map.get(gene) or next(iter(ms_map.values()))

    results_json, dataset_name, config_name = resolve_fit(gene_set, fit_type, gene, n_components)
    if config_name is None:
        config_name = pick_best_canonical_config(gene, gene_set, ms, results_json, dataset_name, run_kwargs)

    analysis = build_gene_set_analysis(ms, gene.lower(), results_json, dataset_name=dataset_name)
    analysis.run(partial_pattern_mode="trust_global", **run_kwargs)
    return ms, analysis, config_name


# %% [markdown]
# ## Analysis: results table (MCC/coverage/accuracy/DOR/sensitivity/specificity)

# %%
def run_results_table(gene_set: str, fit_type: str, genes, n_components=6, run_kwargs=None,
                       compare_uv=True, redundancy_collapse_preset=None):
    run_kwargs = run_kwargs or DEFAULT_RUN_KWARGS
    rows = []
    for gene in genes:
        try:
            regularization_type = "all_assayed" if fit_type == "staged_init_all_assayed" else None
            ms_map = config.build_multiscoresets_for_gene_set(
                gene_set, genes=[gene], regularization_type=regularization_type,
                redundancy_collapse_preset=redundancy_collapse_preset)
            gene_key = gene.lower() if gene_set == "labelseq" else gene.upper()
            ms = ms_map.get(gene_key) or ms_map.get(gene) or next(iter(ms_map.values()))
            results_json, dataset_name, config_name = resolve_fit(gene_set, fit_type, gene, n_components)
            table, _ = build_comparison_table(
                gene.lower(), gene_set, ms, results_json, dataset_name=dataset_name,
                modes=["trust_global"], compare_uv=compare_uv, **run_kwargs,
            )
            table.insert(0, "gene", gene)
            if config_name is not None:
                table = table[table["config"] == config_name]
            rows.append(table)
        except Exception as e:
            print(f"  [{gene}] results-table failed: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# %% [markdown]
# ## Analysis: gene-performance scatter (MV vs UV MCC)

# %%
def run_gene_performance_scatter(gene_set: str, fit_type: str, genes, n_components=6, run_kwargs=None,
                                  ax=None, title=None, ymin=None):
    run_kwargs = run_kwargs or DEFAULT_RUN_KWARGS
    table = run_results_table(gene_set, fit_type, genes, n_components, run_kwargs, compare_uv=True)
    if table.empty:
        print("No data for gene-performance-scatter")
        return None
    clinical = table[table["threshold"].str.startswith("clinical")]
    mv_rows = clinical[clinical["method"] == "MV"] if "method" in clinical.columns else clinical
    rows = []
    for gene, grp in mv_rows.groupby("gene"):
        mv_mcc = grp["mcc"].max()
        uv_row = clinical[(clinical["gene"] == gene) & (clinical.get("method", "UV") != "MV")]
        uv_mcc = uv_row["mcc"].max() if not uv_row.empty else np.nan
        n_eval = grp["total"].max() if "total" in grp.columns else np.nan
        rows.append({"gene": gene, "mv_mcc": mv_mcc, "uv_mcc": uv_mcc, "n_eval": n_eval})
    df = pd.DataFrame(rows)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    plot_mcc_scatter_panel(ax, df, "", title or f"{gene_set} ({fit_type})", ymin=ymin)
    return df


# %% [markdown]
# ## Analysis: 3D interactive evidence scatter (generic, no gene-specific aux groups)

# %%
def run_evidence_3d(gene_set: str, fit_type: str, gene: str, n_components=6, run_kwargs=None,
                     save_path=None):
    import plotly.graph_objects as go

    ms, analysis, config_name = load_gene(gene_set, fit_type, gene, n_components, run_kwargs)
    points = np.asarray(analysis.results[config_name]["points"], dtype=float)

    ndim = ms.scores.shape[1]
    if ndim <= 3:
        dims = list(range(ndim))
    else:
        calibrated = Fit._select_calibration_dims(np.asarray(ms.scores, dtype=float), min_overlap_rows=30)
        dims = (calibrated if len(calibrated) >= 2 else list(range(ndim)))[:3]
    is_3d = len(dims) >= 3

    sa = ms._sample_assignments
    n_roles = min(sa.shape[1], 4)
    ROLE_NAMES = {0: "P/LP", 1: "B/LB", 2: "gnomAD", 3: "Synonymous"}
    max_pt = float(np.nanmax(np.abs(points))) or 1.0

    if is_3d:
        x_i, y_i, z_i = dims[:3]
        coords = [ms.scores[:, i] for i in (x_i, y_i, z_i)]
        complete = ~np.isnan(np.stack(coords, axis=1)).any(axis=1)
    else:
        x_i, y_i = dims[:2]
        coords = [ms.scores[:, i] for i in (x_i, y_i)]
        complete = ~np.isnan(np.stack(coords, axis=1)).any(axis=1)

    traces = []
    for role in range(n_roles):
        mask = sa[:, role].astype(bool) & complete
        if not mask.any():
            continue
        marker = dict(size=5, color=points[mask], colorscale="RdBu_r", cmin=-max_pt, cmax=max_pt,
                      showscale=(role == 0), colorbar=dict(title="Evidence points") if role == 0 else None,
                      opacity=0.8)
        name = f"{ROLE_NAMES.get(role, role)} (n={int(mask.sum())})"
        if is_3d:
            traces.append(go.Scatter3d(x=coords[0][mask], y=coords[1][mask], z=coords[2][mask],
                                        mode="markers", name=name, marker=marker))
        else:
            traces.append(go.Scatter(x=coords[0][mask], y=coords[1][mask],
                                      mode="markers", name=name, marker=marker))

    title = f"{gene.upper()} {fit_type} ({config_name}): evidence points"
    if is_3d:
        layout = go.Layout(title=title, scene=dict(
            xaxis_title=ms.dataset_names[dims[0]], yaxis_title=ms.dataset_names[dims[1]],
            zaxis_title=ms.dataset_names[dims[2]]))
    else:
        layout = go.Layout(title=title, xaxis_title=ms.dataset_names[dims[0]],
                            yaxis_title=ms.dataset_names[dims[1]])
    fig = go.Figure(data=traces, layout=layout)
    if save_path:
        fig.write_html(save_path)
        print(f"Saved {save_path}")
    return fig


# %% [markdown]
# ## Analysis: VUS-reclassification Sankeys (shells out to analysis/run_vus_reclassification.py)

# %%
def run_vus_sankey(output_dir: str = None):
    import subprocess
    print("Running analysis/run_vus_reclassification.py (writes into its own hardcoded "
          "OUTPUT_DIR for now -- not yet parameterized by this cockpit's --output-dir)...")
    subprocess.run([sys.executable, str(_ROOT / "analysis" / "run_vus_reclassification.py")], check=True)
    subprocess.run([sys.executable, str(_ROOT / "analysis" / "plot_vus_reclassification_sankey.py")], check=True)


# %% [markdown]
# ## Analysis: confusion matrices (MV-vs-ClinVar, MV-vs-ClinGen with/without PS3/BS3)
#
# Deliberately NOT `analysis.clingen.build_clingen_confusion` (needs the
# production `tree`/`model_selections` harness from `analysis.discovery`,
# out of scope for this cockpit's registry-driven loading) -- instead reuses
# `analysis.vus_reclassification`'s already-built evidence-code lookup +
# `classify_acmg` logic (same one the VUS-reclassification analysis uses) to
# derive ClinGen ground truth, and `ms`'s own role membership for the
# ClinVar version. Only the RENDERING (`_plot_clingen_confusion_panel`,
# `convert_3x2_to_2x3`) is reused from `analysis/clingen.py` -- those are
# pure/lightweight, no harness dependency. NOT `plot_2x3_confusions_nature`
# itself -- its titles ("ExCALIBR Evidence"/"Author Annotations") are
# hardcoded for a different comparison (MV vs. author functional calls);
# both of THIS analysis's panels are MV-predicted (only the ground truth
# differs), so both need "ExCALIBR" in the title for
# `_plot_clingen_confusion_panel`'s x-tick-label logic to pick the right
# labels.

# %%
def run_confusion_matrices(gene_set: str, fit_type: str, genes, n_components=6, run_kwargs=None,
                            strip_functional_evidence=True, save_path=None):
    from analysis.clingen import convert_3x2_to_2x3, _plot_clingen_confusion_panel
    from analysis.vus_reclassification import (
        build_mv_clinvar_confusion, build_mv_clingen_confusion,
        build_hgvs3_index, build_labelseq_evidence_lookup, build_integrated_evidence_lookup,
    )

    run_kwargs = run_kwargs or DEFAULT_RUN_KWARGS
    clinvar_mat = np.zeros((3, 2), dtype=int)
    clingen_mat = np.zeros((3, 2), dtype=int)

    df_source = None
    if gene_set == "labelseq":
        from src.assay_calibration.multivariate_data.labelseq import build_labelseq_dataframe
        df_source = build_labelseq_dataframe()
    elif gene_set == "integrated":
        df_source = pd.read_csv(config.INTEGRATED_VARIANT_EFFECT_DATASET_PATH, sep="\t", low_memory=False)

    for gene in genes:
        try:
            ms, analysis, config_name = load_gene(gene_set, fit_type, gene, n_components, run_kwargs)
        except Exception as e:
            print(f"  [{gene}] confusion-matrices: load failed ({e})")
            continue
        points = np.asarray(analysis.results[config_name]["points"], dtype=float)
        clinvar_mat += build_mv_clinvar_confusion(points, ms._sample_assignments)

        if df_source is None:
            continue
        evidence_lookup = (build_labelseq_evidence_lookup(gene, df_source) if gene_set == "labelseq"
                            else build_integrated_evidence_lookup(gene, df_source))
        if not evidence_lookup:
            continue
        hgvs3_index = build_hgvs3_index(ms)
        clingen_mat += build_mv_clingen_confusion(points, hgvs3_index, evidence_lookup,
                                                   strip_functional_evidence=strip_functional_evidence)

    # Both panels are ExCALIBR-MV-predicted (only the ground truth differs:
    # ClinVar vs. ClinGen) -- use titles containing "ExCALIBR" for BOTH, since
    # `_plot_clingen_confusion_panel` keys its x-tick label choice
    # (Benign/Indeterminate/Pathogenic vs. Normal/Indeterminate/Abnormal)
    # off that literal substring, and "Normal/Abnormal" (its default for a
    # non-ExCALIBR title) would be wrong here -- neither panel is author
    # functional annotations.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    strip_note = " (PS3/BS3 stripped)" if strip_functional_evidence else " (PS3/BS3 kept)"
    _plot_clingen_confusion_panel(axes[0], convert_3x2_to_2x3(clinvar_mat), "ExCALIBR vs ClinVar", "A",
                                   xlabel="Evidence Direction", ylabel="ClinVar Classification")
    _plot_clingen_confusion_panel(axes[1], convert_3x2_to_2x3(clingen_mat), f"ExCALIBR vs ClinGen{strip_note}",
                                   "B", xlabel="Evidence Direction", ylabel="")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved {save_path}")
    return clinvar_mat, clingen_mat, fig


# %% [markdown]
# ## Analysis: Brnich external-tool comparison ("genes reaching each evidence level")
#
# Ported from the original LABEL-seq analysis notebook's cell (user-supplied
# this session) -- generalized off the old `precomputed_grids` dict to use
# this cockpit's own `load_gene`, and extended with a 4th bar: ExCALIBR's UV
# ("combined, independently calibrated evidence") baseline, via
# `mv_analysis.uv_sources.load_uv_points` + `uv_agg.aggregate_nonconflicting`
# -- same points scale as MV, so per-gene max/min is directly comparable.

# %%
BRNICH_TO_EXCALIBR = {"RAF1": "CRAF", "MAP2K1": "MEK1", "MAP2K2": "MEK2", "PTPN11": "SHP2"}
BRNICH_CSV_PATH = "/data/ross/assay_calibration/labelseq/LABELseq_brnich.csv"

_EVIDENCE_POINTS = {
    "PS3_supporting": 1, "BS3_supporting": -1, "PS3_moderate": 2, "BS3_moderate": -2,
    "PS3": 4, "BS3": -4, "PS3_strong": 4, "BS3_strong": -4,
    "PS3_very_strong": 8, "BS3_very_strong": -8, "Indeterminate": 0,
}


def _brnich_ev(s):
    return _EVIDENCE_POINTS.get(str(s).strip(), 0) if pd.notna(s) else 0


def run_brnich_comparison(gene_set: str, fit_type: str, genes, n_components=6, run_kwargs=None,
                           save_path=None):
    from mv_analysis.uv_sources import load_uv_points
    from mv_analysis.uv_agg import aggregate_nonconflicting

    run_kwargs = run_kwargs or DEFAULT_RUN_KWARGS
    df_brnich = pd.read_csv(BRNICH_CSV_PATH)
    df_brnich["brnich_all_path"] = df_brnich["evidence_outside_all"].apply(_brnich_ev)
    df_brnich["brnich_all_ben"] = df_brnich["evidence_inside_all"].apply(_brnich_ev)
    df_brnich["brnich_mis_path"] = df_brnich["evidence_outside_missense"].apply(_brnich_ev)
    df_brnich["brnich_mis_ben"] = df_brnich["evidence_inside_missense"].apply(_brnich_ev)
    df_brnich["gene"] = df_brnich["gene"].replace(BRNICH_TO_EXCALIBR)
    df_brnich = df_brnich.set_index("gene")

    excalibr_rows = []
    for gene in genes:
        try:
            ms, analysis, config_name = load_gene(gene_set, fit_type, gene, n_components, run_kwargs)
        except Exception as e:
            print(f"  [{gene}] brnich-comparison: load failed ({e})")
            continue
        pts = np.asarray(analysis.results[config_name]["points"], dtype=float)
        is_nu = analysis.p_idx is None

        uv_path = uv_ben = 0
        try:
            uv_result = load_uv_points(gene, ms, gene_set)
            if uv_result is not None:
                _, uv_mat = uv_result
                uv_points = aggregate_nonconflicting(uv_mat)
                uv_path, uv_ben = int(np.nanmax(uv_points)), int(np.nanmin(uv_points))
        except Exception as e:
            print(f"  [{gene}] UV points unavailable for brnich-comparison: {e}")

        excalibr_rows.append({
            "gene": gene.upper(), "excalibr_path": int(np.nanmax(pts)), "excalibr_ben": int(np.nanmin(pts)),
            "excalibr_uv_path": uv_path, "excalibr_uv_ben": uv_ben, "is_nu": is_nu,
        })
    df_ex = pd.DataFrame(excalibr_rows).set_index("gene")

    brnich_cols = ["brnich_all_path", "brnich_all_ben", "brnich_mis_path", "brnich_mis_ben"]
    df = df_ex.join(df_brnich[brnich_cols], how="left")
    df[brnich_cols] = df[brnich_cols].fillna(0).astype(int)

    methods = {
        "ExCALIBR-MV": ("excalibr_path", "excalibr_ben"),
        "ExCALIBR-UV": ("excalibr_uv_path", "excalibr_uv_ben"),
        "Brnich (all variants)": ("brnich_all_path", "brnich_all_ben"),
        "Brnich (missense)": ("brnich_mis_path", "brnich_mis_ben"),
    }
    method_colors = ["#c0392b", "#8e44ad", "#2980b9", "#27ae60"]
    mv_color, mv_nu_color = "#c0392b", "#e8a090"

    ev_levels = [1, 2, 4, 8, -1, -2, -4, -8]
    counts = {}
    for method, (pc, bc) in methods.items():
        counts[method] = {}
        for lv in [1, 2, 4, 8]:
            counts[method][lv] = int((df[pc] >= lv).sum())
        for lv in [-1, -2, -4, -8]:
            counts[method][lv] = int((df[bc] <= lv).sum())
    counts_nu, counts_non_nu = {}, {}
    for lv in [1, 2, 4, 8]:
        counts_nu[lv] = int(((df["excalibr_path"] >= lv) & df["is_nu"]).sum())
        counts_non_nu[lv] = int(((df["excalibr_path"] >= lv) & ~df["is_nu"]).sum())
    for lv in [-1, -2, -4, -8]:
        counts_nu[lv] = int(((df["excalibr_ben"] <= lv) & df["is_nu"]).sum())
        counts_non_nu[lv] = int(((df["excalibr_ben"] <= lv) & ~df["is_nu"]).sum())

    n_methods = len(methods)
    width = 0.9 / n_methods
    x = np.arange(len(ev_levels))
    fig, ax = plt.subplots(figsize=(12, 4.5))

    xpos = x + (0 - (n_methods - 1) / 2) * width
    bot = [counts_non_nu[lv] for lv in ev_levels]
    top = [counts_nu[lv] for lv in ev_levels]
    ax.bar(xpos, bot, width, label="ExCALIBR-MV", color=mv_color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar(xpos, top, width, bottom=bot, label="ExCALIBR-MV (NU)", color=mv_nu_color, alpha=0.85,
           edgecolor="white", linewidth=0.5)

    for i, (method, color) in enumerate(list(zip(methods, method_colors))[1:], start=1):
        vals = [counts[method][lv] for lv in ev_levels]
        bars = ax.bar(x + (i - (n_methods - 1) / 2) * width, vals, width, label=method, color=color,
                       alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(v),
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{"+" + str(lv) if lv > 0 else str(lv)}' for lv in ev_levels], fontsize=9)
    ax.set_xlabel("Evidence level", fontsize=10)
    ax.set_ylabel("Number of genes", fontsize=10)
    ax.set_title(f"Genes reaching each evidence level ({gene_set}, {fit_type})", fontsize=11, fontweight="bold")
    ax.axvline(3.5, color="#aaaaaa", lw=0.8, ls="--")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved {save_path}")
    return df, fig


# %% [markdown]
# ## CLI entry point

# %%
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene-set", required=True,
                     choices=["labelseq", "integrated", "predictor", "combined", "card11", "tp53"])
    ap.add_argument("--fit-type", default="canonical",
                     choices=["canonical", "staged_init_all_assayed"])
    ap.add_argument("--genes", nargs="+", default=None,
                     help="Subset override; default is every gene in --gene-set's registry.")
    ap.add_argument("--n-components", type=int, default=6)
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--analyses", nargs="+", default=["results-table"],
                     choices=["results-table", "uv-comparison", "gene-performance-scatter",
                              "evidence-3d", "vus-sankey", "confusion-matrices", "brnich-comparison"])
    ap.add_argument("--no-strip-functional-evidence", action="store_true",
                     help="For --analyses confusion-matrices: keep PS3/BS3 in the ClinGen "
                          "ground truth instead of stripping (circularity-check mode).")
    ap.add_argument("--path-percentile", type=float, default=DEFAULT_RUN_KWARGS["path_percentile"])
    from src.assay_calibration.multivariate_data.redundancy_collapse import PRESETS as _RC_PRESETS
    ap.add_argument("--redundancy-collapse-preset", choices=sorted(_RC_PRESETS.keys()), default=None,
                     help="e.g. 'tp53_kato_pca2' -- collapses TP53's 8-dim Kato_2003 panel to 2 "
                          "PCs before scoring. Only meaningful if the fit being loaded was itself "
                          "trained on that collapsed space (via hpc/prepare.py's equivalent flag); "
                          "otherwise the fit-vs-dataset shape guard in MVCalibrationAnalysis will "
                          "correctly raise on the resulting dimension mismatch. Currently only "
                          "threaded through --analyses results-table; other analyses would need "
                          "the same redundancy_collapse_preset kwarg added to their run_* function.")
    args = ap.parse_args()

    run_kwargs = dict(DEFAULT_RUN_KWARGS)
    run_kwargs["path_percentile"] = args.path_percentile

    genes = args.genes
    if genes is None:
        if args.gene_set == "labelseq":
            genes = list(config.LABELSEQ_GENES)
        else:
            raise ValueError(f"--genes is required for --gene-set {args.gene_set!r} "
                              f"(no default gene list registered yet)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if "results-table" in args.analyses or "uv-comparison" in args.analyses:
        table = run_results_table(args.gene_set, args.fit_type, genes, args.n_components, run_kwargs,
                                   redundancy_collapse_preset=args.redundancy_collapse_preset)
        out = Path(args.output_dir) / f"{args.gene_set}_{args.fit_type}_results_table.csv"
        table.to_csv(out, index=False)
        print(f"Saved {out}")

    if "gene-performance-scatter" in args.analyses:
        fig, ax = plt.subplots(figsize=(6, 6))
        run_gene_performance_scatter(args.gene_set, args.fit_type, genes, args.n_components, run_kwargs, ax=ax)
        out = Path(args.output_dir) / f"{args.gene_set}_{args.fit_type}_gene_performance.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved {out}")

    if "evidence-3d" in args.analyses:
        for gene in genes:
            out = Path(args.output_dir) / f"{gene}_{args.fit_type}_evidence_3d.html"
            run_evidence_3d(args.gene_set, args.fit_type, gene, args.n_components, run_kwargs, save_path=str(out))

    if "vus-sankey" in args.analyses:
        run_vus_sankey(args.output_dir)

    if "confusion-matrices" in args.analyses:
        out = Path(args.output_dir) / f"{args.gene_set}_{args.fit_type}_confusion_matrices.png"
        run_confusion_matrices(args.gene_set, args.fit_type, genes, args.n_components, run_kwargs,
                                strip_functional_evidence=not args.no_strip_functional_evidence,
                                save_path=str(out))

    if "brnich-comparison" in args.analyses:
        out = Path(args.output_dir) / f"{args.gene_set}_{args.fit_type}_brnich_comparison.png"
        run_brnich_comparison(args.gene_set, args.fit_type, genes, args.n_components, run_kwargs,
                               save_path=str(out))


# %%
if __name__ == "__main__" and not _running_as_notebook():
    main()

# %% [markdown]
# ## Jupyter cells-mode playground
#
# Everything below only runs interactively (guarded by `_running_as_notebook()`
# implicitly via not being under the CLI `main()` call above) -- pick a
# gene-set/fit-type/gene here and inspect intermediate objects (`ms`,
# `analysis`, `analysis.results[config_name]`) directly.

# %%
if _running_as_notebook():
    GENE_SET = "labelseq"
    FIT_TYPE = "canonical"
    GENE = "ret"

    ms, analysis, config_name = load_gene(GENE_SET, FIT_TYPE, GENE)
    print(f"config={config_name}, n_variants={ms.scores.shape[0]}, n_dims={ms.scores.shape[1]}")
