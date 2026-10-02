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
# # ExCALIBR-MV paper analysis: every manuscript figure/table, end to end
#
# The paper-specific counterpart to `mv_analysis/mv_cockpit.py` (which stays
# the flexible "load and analyze one dataset of choosing" tool) -- this
# notebook runs *every* analysis `ExCALIBR_latex_092026/ExCALIBR_MV.tex`
# needs, from one fits directory (`mv_analysis.config.PAPER_FITS_DIR`), the
# same role `analysis/analyze_pipeline_output.py` plays for the univariate
# paper. Lives in `mv_analysis/`, not `analysis/`, so MV output never mixes
# with the univariate pipeline's -- it only *imports* reusable code from
# `analysis/` (e.g. `analysis.clingen`, `analysis.vus_reclassification`),
# exactly as `mv_cockpit.py` already does.
#
# All paths/dependencies are resolved through `mv_analysis/config.py` --
# nothing here hardcodes a path. Figures land in `config.PAPER_FIGURES_DIR`
# (a run-specific directory, e.g. `{PAPER_FITS_DIR}/figures/`), NOT directly
# into the paper repo's `Figures/mv/` -- copying those over is a separate,
# deliberate step.
#
# Run as a plain script (`python mv_analysis/analyze_mv_pipeline_output.py
# --output-dir ...`) for the CLI, or open in Jupyter (`jupytext --sync`
# first) for the cells-mode workflow -- same dual-mode pattern as
# `mv_analysis/mv_cockpit.py` / `analysis/analyze_pipeline_output.py`.
#
# ## Status (be honest about what's verified vs. designed-but-unrun)
# - **Verified against the real, still-in-progress fits at
#   `config.PAPER_FITS_DIR`**: aggregation, the tp53/card11/predictor/fgfr/
#   combined registry entries, pooled gene-set metrics, individual-predictor
#   comparison, the gene-set-agnostic ClinVar confusion matrix (pooled +
#   per-gene), RET and CARD11 phenotype-evidence plots, and TP53's RPV
#   report (all 4 configs, full figure set).
# - **Some datasets are still missing** because the SLURM array that
#   produced `PAPER_FITS_DIR` was still running as of this notebook's last
#   edit -- most `_combined_mv` genes besides BRCA1/BRCA2, and TP53's own
#   `_combined_mv`. Sections below skip a missing dataset with a printed
#   message (matching `run_results_table`'s existing per-gene try/except)
#   rather than failing the whole run; re-run `hpc/aggregate_results.py`
#   once the array finishes for the full picture.
# - **Not run end-to-end in one sitting here** (each section was verified
#   individually/on a small gene subset, not the full genes lists below, to
#   keep iteration fast) -- run the CLI in full before trusting the complete
#   output set.

# %%
import sys
from pathlib import Path


import json


def _running_as_notebook() -> bool:
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
from mv_analysis import mv_cockpit as cockpit
from mv_analysis import report
from mv_analysis import phenotype_evidence as pheno
from mv_analysis import tp53_rpv_report
from mv_analysis.gene_performance_scatter import build_gene_performance_figure, PLAIN_INTEGRATED_GENES, EVIDENCE_DIRECTION

# ---------------------------------------------------------------------------
# Gene lists per gene-set (single source of truth for THIS notebook's runs;
# the per-gene-set MultiScoreset/analysis machinery itself lives in
# config.py/mv_cockpit.py, not duplicated here).
#
# "Functional" (Figure fig:mv_scatter Panel A / Table mv_gene_performance's
# functional half) spans FIVE gene-sets, not just labelseq -- labelseq (17),
# integrated (15 plain genes), card11 (1), tp53 (1), fgfr (1 combined
# "gene"), 35 total. Each gene-set is still run/reported SEPARATELY (its own
# per-gene table + its own pooled row) as well as combined into one overall
# "functional (all)" pooled row spanning every functional gene together --
# see run_functional_results.
# ---------------------------------------------------------------------------
FUNCTIONAL_GENE_SETS = (
    ("labelseq", list(config.LABELSEQ_GENES)),
    ("integrated", list(PLAIN_INTEGRATED_GENES)),
    ("card11", ["CARD11"]),
    ("tp53", ["TP53"]),
    ("fgfr", ["FGFR_combined"]),
)
PREDICTOR_COMBINED_GENES = ("BRCA1", "BRCA2", "F9", "JAG1", "MSH2", "SCN5A", "TP53", "TSC2")
RUN_KWARGS = dict(cockpit.DEFAULT_RUN_KWARGS)

FIT_TYPE = "paper"  # resolves against config.PAPER_RESULTS_JSON (see mv_cockpit.resolve_fit)


# %% [markdown]
# ## Stage 0: config + aggregation
#
# Prints the resolved paths (matching `analysis/config.py`'s convention of
# always surfacing what it resolved to) and generates `PAPER_RESULTS_JSON`
# from the raw per-bootstrap pickles if it doesn't exist yet -- safely
# re-runnable at any point while the fits array is still in progress.

# %%
def ensure_aggregated():
    print(f"PAPER_FITS_DIR    = {config.PAPER_FITS_DIR}")
    print(f"PAPER_RESULTS_JSON = {config.PAPER_RESULTS_JSON}")
    print(f"PAPER_FIGURES_DIR  = {config.PAPER_FIGURES_DIR}")
    # Ensure every categorized subdirectory exists upfront -- see the
    # figures-directory restructure: every save-path call site below writes
    # into one of these rather than dumping flat into PAPER_FIGURES_DIR.
    for d in (config.PAPER_FUNCTIONAL_DIR, config.PAPER_PREDICTOR_DIR, config.PAPER_COMBINED_DIR,
              config.PAPER_POOLED_DIR, config.PAPER_ACCURACY_DIR, config.PAPER_CONFUSIONS_DIR,
              config.PAPER_VUS_RECLASSIFICATION_DIR, config.PAPER_BRNICH_COMPARISON_DIR,
              config.PAPER_PER_GENE_DIR, config.PAPER_TP53_RPV_DIR, config.PAPER_RET_EVIDENCE_DIR,
              config.PAPER_CARD11_PHENOTYPE_DIR, config.PAPER_CARTOONS_DIR, config.PAPER_CACHE_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    if not Path(config.PAPER_RESULTS_JSON).exists():
        import subprocess
        print("Aggregating raw per-bootstrap pickles...")
        subprocess.run(
            [sys.executable, str(_ROOT / "hpc" / "aggregate_results.py"), config.PAPER_FITS_DIR],
            check=True,
        )
    else:
        print("PAPER_RESULTS_JSON already exists -- re-run hpc/aggregate_results.py "
              "manually once the fits array finishes for a fresh aggregate.")


# %% [markdown]
# ## Section 2: whole-aggregate gene-performance scatter (Figure fig:mv_scatter)

# %%
def run_gene_performance_scatter_all(save_path=None, cache_dir=None, slides=False):
    """cache_dir defaults to {PAPER_CACHE_DIR}/scatter_cache -- build_gene_performance_figure's
    own per-(panel, gene) disk cache (_load_cached/_save_cached in gene_performance_scatter.py)
    was previously never reached from here since this wrapper never forwarded a cache_dir,
    silently forcing a full ~34+8+8-gene, zero-cache recompute on every call regardless of
    whether a prior run's cache existed. Pass cache_dir=False to force a full recompute.

    ``slides=True``: simplified, large-font slideshow rendering (see
    build_gene_performance_figure/plot_metric_scatter_panel) saved alongside
    the paper version with a "_slides" suffix -- reuses the SAME cache (the
    underlying per-gene MV/UV numbers don't change, only the rendering)."""
    save_path = save_path or f"{config.PAPER_ACCURACY_DIR}/gene_performance_scatter.png"
    if cache_dir is None:
        cache_dir = f"{config.PAPER_CACHE_DIR}/scatter_cache"
    elif cache_dir is False:
        cache_dir = None
    return build_gene_performance_figure(config.PAPER_RESULTS_JSON, save_path=save_path,
                                          cache_dir=cache_dir, slides=slides)


# %% [markdown]
# ## Section 3: per-gene-set results tables (functional / predictor / combined)
#
# Each returns (per_gene_table, pooled_row, per_gene_lp) -- `pooled_row` sums
# counts across genes before computing MCC/accuracy/etc. (the manuscript's
# stated convention), not an average of per-gene metrics. `per_gene_lp`
# ({gene: (labels, points)}) is exposed so callers spanning MULTIPLE
# gene-sets (see `run_functional_results`) can merge before pooling once,
# rather than re-running `load_gene` a second time.

# %%
def _lp_cache_path(cache_dir, gene_set, gene, cluster_idx):
    return Path(cache_dir) / f"lp_{gene_set}_{FIT_TYPE}_{gene}_cluster{cluster_idx}.json"


def run_gene_set_results(gene_set, genes, run_kwargs=None, redundancy_collapse_preset=None, cache_dir=None):
    """`cache_dir`, if given, persists both the per-gene results table (via
    cockpit.run_results_table's own cache) and this function's separate
    pooling pass ((labels, points) per gene/cluster) to disk -- neither was
    cached before, so re-running just a LATER section (e.g. after fixing an
    unrelated bug in a downstream section) had to redo every gene-set's full
    100-bootstrap scoring from scratch every time. Delete the relevant cache
    file(s) to force a recompute after a code change that could change the
    actual numbers (this mirrors gene_performance_scatter.py's existing
    cache_dir convention, not a new caching philosophy)."""
    run_kwargs = run_kwargs or RUN_KWARGS
    per_gene_table = cockpit.run_results_table(
        gene_set, FIT_TYPE, genes, run_kwargs=run_kwargs,
        redundancy_collapse_preset=redundancy_collapse_preset, cache_dir=cache_dir,
    )
    if per_gene_table.empty:
        return per_gene_table, pd.DataFrame(), {}

    # Mirrors cockpit.run_results_table's own cluster-variant loop (see its
    # docstring) so pooled metrics include EVERY disjoint cluster's data for
    # a multi-cluster gene (e.g. BRCA2 under --gene-set integrated), not
    # just the primary one.
    per_gene_lp = {}
    for gene in genes:
        try:
            results_json, primary_dataset_name, _ = cockpit.resolve_fit(gene_set, FIT_TYPE, gene)
            if FIT_TYPE == "staged_init_all_assayed":
                variants = [primary_dataset_name]
            else:
                variants = config.list_gene_cluster_variants(gene, gene_set, results_json=results_json)
        except Exception as e:
            print(f"  [{gene}] pooling skipped (variant lookup): {e}")
            continue
        for cluster_idx, dataset_name in enumerate(variants):
            label = gene if cluster_idx == 0 else f"{gene}:cluster{cluster_idx + 1}"

            cache_path = _lp_cache_path(cache_dir, gene_set, gene, cluster_idx) if cache_dir else None
            if cache_path is not None and cache_path.exists():
                with open(cache_path) as f:
                    cached = json.load(f)
                if cached.get("status") == "ok":
                    per_gene_lp[label] = (np.array(cached["labels"]), np.array(cached["points"]))
                continue

            try:
                ms = config.build_ms_for_gene_cluster(gene_set, gene, cluster_idx=cluster_idx)
                config_name = cockpit.pick_best_canonical_config(
                    gene, gene_set, ms, results_json, dataset_name, run_kwargs)
                analysis = cockpit.build_gene_set_analysis(ms, gene.lower(), results_json, dataset_name=dataset_name, gene_set=gene_set)
                analysis.run(partial_pattern_mode="trust_global", **run_kwargs)
                points = np.asarray(analysis.results[config_name]["points"], dtype=float)
                eval_mask, labels = report._eval_labels(ms, analysis.p_idx, analysis.b_idx)
                lp = (labels, points[eval_mask])
                per_gene_lp[label] = lp
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump({"status": "ok", "labels": lp[0].tolist(), "points": lp[1].tolist()}, f)
            except Exception as e:
                print(f"  [{label}] pooling skipped: {e}")
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump({"status": "failed", "reason": str(e)}, f)
    pooled_row = report.pool_gene_set_metrics(gene_set, per_gene_lp, method="MV")
    return per_gene_table, pooled_row, per_gene_lp


def run_functional_results(run_kwargs=None, cache_dir=None):
    """Runs all FIVE functional gene-sets (labelseq/integrated/card11/tp53/
    fgfr) separately -- each gets its own per_gene table + its own pooled
    row, matching this notebook's existing per-gene-set convention -- AND
    combines every functional gene together into one overall "functional
    (all)" pooled row, matching Figure fig:mv_scatter Panel A / Table
    mv_gene_performance's actual scope (35 genes total, not just the 17
    LABEL-seq ones).

    Returns (combined_table, combined_pooled, per_gene_set_results) where
    per_gene_set_results = {gene_set: (table, pooled_row)}.
    """
    run_kwargs = run_kwargs or RUN_KWARGS
    tables, all_lp, per_gene_set_results = [], {}, {}
    for gene_set, genes in FUNCTIONAL_GENE_SETS:
        table, pooled, lp = run_gene_set_results(gene_set, genes, run_kwargs=run_kwargs, cache_dir=cache_dir)
        per_gene_set_results[gene_set] = (table, pooled)
        if not table.empty:
            table = table.copy()
            table.insert(1, "gene_set", gene_set)
            tables.append(table)
        all_lp.update({f"{gene_set}:{g}": lp_val for g, lp_val in lp.items()})
    combined_table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    combined_pooled = report.pool_gene_set_metrics("functional (all)", all_lp, method="MV")
    return combined_table, combined_pooled, per_gene_set_results


# %% [markdown]
# ## Section 4: predictor individual-vs-combined comparison
#
# REVEL/AlphaMissense/MutPred2 scored individually (via
# `report.individual_predictor_comparison`), alongside the existing
# "UV non-conflicting" aggregate and MV rows already in the results table.

# %%
def run_predictor_individual_comparison(genes=PREDICTOR_COMBINED_GENES, run_kwargs=None):
    run_kwargs = run_kwargs or RUN_KWARGS
    rows = []
    for gene in genes:
        try:
            ms, analysis, config_name = cockpit.load_gene("predictor", FIT_TYPE, gene, run_kwargs=run_kwargs)
        except Exception as e:
            print(f"  [{gene}] predictor-individual: load failed ({e})")
            continue
        df = report.individual_predictor_comparison(gene, ms, "predictor", analysis.p_idx, analysis.b_idx)
        if df.empty:
            continue
        df.insert(0, "gene", gene)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# %% [markdown]
# ## Section 5: confusion matrices (pooled + per-gene) per gene-set

# %%
def run_confusion_matrices_for_gene_set(gene_set, genes, run_kwargs=None, save_path=None):
    run_kwargs = run_kwargs or RUN_KWARGS
    save_path = save_path or f"{config.PAPER_CONFUSIONS_DIR}/{gene_set}_clinvar_confusion.png"
    return cockpit.run_clinvar_confusion_matrix(
        gene_set, FIT_TYPE, genes, run_kwargs=run_kwargs, per_gene=True, save_path=save_path,
    )


# %% [markdown]
# ## Section 6/7: disease-phenotype evidence -- RET (MEN2/Hirschsprung), CARD11 (BENTA/CADINS)
#
# See `mv_analysis/phenotype_evidence.py`'s module docstring for why these
# two (and only these two) share one mechanism, distinct from TP53's RPV.

# %%
def run_ret_phenotype_evidence(x_dim="abundance_No_treatment", y_dim="abundance_HSP90i",
                                z_dim="activity_No_treatment", config_name="4c_unc", save_path=None,
                                run_kwargs=None):
    from src.assay_calibration.multivariate_data.labelseq import build_labelseq_multiscoresets
    save_path = save_path or f"{config.PAPER_RET_EVIDENCE_DIR}/ret_3d_evidence.png"
    ms_map = build_labelseq_multiscoresets(genes=["RET"])
    ms = ms_map["ret"]
    masks = pheno.ret_phenotype_masks(ms)
    return pheno.plot_phenotype_evidence(
        "ret", x_dim, y_dim, z_dim, config.PAPER_RESULTS_JSON, masks,
        replace_role="P/LP", config=config_name, z_log10=False, ms_map=ms_map, save_path=save_path,
        run_kwargs=run_kwargs,
    )


def run_card11_phenotype_evidence(config_name="4c_unc", save_path=None, run_kwargs=None):
    save_path = save_path or f"{config.PAPER_CARD11_PHENOTYPE_DIR}/card11_phenotype_evidence.png"
    ms = config.build_multiscoresets_for_gene_set("card11")["CARD11"]
    return pheno.plot_card11_phenotype_evidence(
        ms, config.PAPER_RESULTS_JSON, config=config_name, save_path=save_path, run_kwargs=run_kwargs)


# %% [markdown]
# ## Section 8: TP53 special figures (penetrance histogram, RPV classification/quadrant, confusion, marginals)

# %%
def run_tp53_special_figures(run_kwargs=None):
    return tp53_rpv_report.run_tp53_rpv_report(
        results_json=config.PAPER_RESULTS_JSON,
        output_dir=config.PAPER_TP53_RPV_DIR,
        run_kwargs=run_kwargs,
    )


# %% [markdown]
# ## Section 9: VUS-reclassification Sankey / ClinVar+ClinGen confusion / Brnich comparison
#
# Already-working `mv_cockpit` functions -- re-pointed at the paper fits via
# FIT_TYPE="paper" (vus-sankey shells out to `analysis/run_vus_reclassification.py`,
# which is canonical-v3-only per its own module docstring -- NOT re-pointed
# at PAPER_RESULTS_JSON; that script's own results-JSON constant would need
# updating separately if the paper wants VUS-reclassification numbers from
# the post-2026-fix fits specifically).

# %%
def run_vus_and_brnich(labelseq_genes=None, run_kwargs=None):
    run_kwargs = run_kwargs or RUN_KWARGS
    labelseq_genes = labelseq_genes or list(config.LABELSEQ_GENES)
    cockpit.run_vus_sankey(config.PAPER_FIGURES_DIR)
    cockpit.run_confusion_matrices(
        "labelseq", FIT_TYPE, labelseq_genes, run_kwargs=run_kwargs,
        save_path=f"{config.PAPER_CONFUSIONS_DIR}/mv_confusion_matrices.png",
    )
    cockpit.run_brnich_comparison(
        "labelseq", FIT_TYPE, labelseq_genes, run_kwargs=run_kwargs,
        save_path=f"{config.PAPER_BRNICH_COMPARISON_DIR}/mv_brnich_comparison.png",
    )


def run_per_gene_diagnostics(cache_dir=None):
    """Universal per-gene diagnostic report (confusion matrix + calibration/
    marginal-density/LR+ plots, dimension-adaptive: native 2D pairwise at
    D=2, pairwise-LR+-grid-plus-marginals at D>2) for EVERY gene across
    EVERY gene-set -- functional (all 5 FUNCTIONAL_GENE_SETS sub-sets),
    predictor, and combined. Previously this only ever ran for TP53 (via
    tp53_rpv_report.py, itself calling the same generate_gene_report this
    reuses) -- every other gene got none of this.

    Renders EVERY config with valid bootstrap results per gene (`configs=
    None`, generate_gene_report's own default) -- in this pipeline's
    component_range that's always exactly {3c_unc, 4c_unc}, so both get a
    `{gene}_{config}_mv_calibration.png` (previously only the best-MCC
    config rendered, silently missing whichever of 3c/4c wasn't picked as
    best -- there is no cross-config cache reuse in precompute_mv_plot_data,
    so this genuinely ~doubles this rollout's cost; worth it for complete
    coverage). Does NOT pass include_dim_densities (stays at
    generate_gene_report's new default, False) -- the separate per-dimension
    density PNGs are redundant with the main calibration plot's own
    marginal rows for D>2 and are dropped everywhere, including TP53's own
    tp53_rpv_report.py call (explicitly confirmed not wanted anywhere, not
    just outside the TP53-specific report).

    Does NOT replace or touch run_tp53_special_figures/
    run_ret_phenotype_evidence/run_card11_phenotype_evidence -- those add
    their own extra domain-specific figures (RPV penetrance, phenotype-
    subgroup 3D evidence) on top of, not instead of, this universal report.
    """
    from src.assay_calibration.multivariate_analysis.report_gene import generate_gene_report

    if cache_dir is None:
        cache_dir = f"{config.PAPER_CACHE_DIR}/per_gene_diagnostics_cache"

    gene_sets = list(FUNCTIONAL_GENE_SETS) + [
        ("predictor", list(PREDICTOR_COMBINED_GENES)),
        ("combined", list(PREDICTOR_COMBINED_GENES)),
    ]

    for gene_set, genes in gene_sets:
        for gene in genes:
            print(f"\n=== per-gene diagnostics: {gene_set}/{gene} ===", flush=True)
            try:
                ms_map = config.build_multiscoresets_for_gene_set(gene_set, genes=[gene])
                gene_key = gene.lower() if gene_set == "labelseq" else gene.upper()
                ms = ms_map.get(gene_key) or ms_map.get(gene) or next(iter(ms_map.values()), None)
                if ms is None:
                    print(f"  SKIP {gene}: no ms built for gene_set={gene_set}")
                    continue
                dataset_name = config.canonical_dataset_name(gene, gene_set, results_json=config.PAPER_RESULTS_JSON)
                best_config = cockpit.pick_best_canonical_config(
                    gene, gene_set, ms, config.PAPER_RESULTS_JSON, dataset_name, RUN_KWARGS)
                if best_config is None:
                    print(f"  SKIP {gene}: no canonical config resolved")
                    continue
                analysis = cockpit.build_gene_set_analysis(
                    ms, gene.lower(), config.PAPER_RESULTS_JSON, dataset_name=dataset_name, gene_set=gene_set)
                analysis.run(partial_pattern_mode="trust_global", **RUN_KWARGS)
                if analysis.results.get(best_config) is None:
                    print(f"  SKIP {gene}: no valid bootstraps for config {best_config}")
                    continue
                output_dir = f"{config.PAPER_PER_GENE_DIR}/{gene_set}/{gene}"
                generate_gene_report(
                    analysis, gene, output_dir, configs=None,
                    cache_dir=f"{cache_dir}/{gene_set}/{gene}",
                )
            except Exception as e:
                print(f"  SKIP {gene}: {e}")


def run_clinvar_vus_points_by_panel():
    """True-ClinVar-VUS -> our-points Sankeys, by panel (functional/combined --
    NOT the ACMG-evidence-code-stripping VUS sankey `run_vus_and_brnich`
    builds via cockpit.run_vus_sankey, a different analysis entirely). Shells
    out to the two standalone scripts (matching cockpit.run_vus_sankey's own
    subprocess pattern) since both are also independently runnable/debuggable
    from the command line."""
    import subprocess
    subprocess.run(
        [sys.executable, str(_ROOT / "analysis" / "compute_clinvar_vus_points_by_panel.py")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(_ROOT / "analysis" / "plot_clinvar_vus_points_sankey_by_panel.py")],
        check=True,
    )


# %% [markdown]
# ## Section 10: manuscript-number summary
#
# Prints the values `ExCALIBR_MV.tex`'s `\Num*`/`\MCC*` macros currently
# hold as placeholders -- mirrors `analyze_pipeline_output.py`'s final
# section's role (a single place to read off the numbers going into the
# manuscript, not a LaTeX-writing step).

# %%
def print_manuscript_summary(functional_table, predictor_table, combined_table,
                              functional_pooled, predictor_pooled, combined_pooled):
    """`*_table`: per-gene tables (for NumXGenes -- distinct gene counts).
    `*_pooled`: the ACTUAL pooled rows from report.pool_gene_set_metrics
    (sum-counts-across-genes, matching the manuscript's stated convention) --
    NOT re-derived from the per-gene table here, since a naive .max()/mean()
    across per-gene MV rows is a different (and mislabeled) statistic, not
    "pooled" in that sense. A previous version of this function did exactly
    that AND filtered method=="MV" (which never matches -- _rows_for_points
    builds method as "MV {config}", e.g. "MV 3c_unc"), so it silently printed
    NaN for every MCC; both bugs are fixed by using the pooled tables the
    caller already computed."""
    def _mcc(pooled_table):
        if pooled_table is None or pooled_table.empty:
            return float("nan")
        # evidence_direction, not "clinical" -- matches the config-selection
        # criterion (cockpit.pick_best_canonical_config's new default) and
        # the actual manuscript accuracy-scatter figure's own selection
        # (gene_performance_scatter.py's _extract_mv_uv); previously this
        # reported a clinical-threshold MCC for a config chosen under a
        # DIFFERENT (clinical) criterion too, which was at least internally
        # consistent then -- now that selection is evidence_direction-based,
        # reporting stays matched to it rather than mixing two thresholds.
        row = pooled_table[pooled_table["threshold"] == EVIDENCE_DIRECTION]
        return row["mcc"].max() if not row.empty else float("nan")

    print(f"NumFunctionalGenes = {functional_table['gene'].nunique() if not functional_table.empty else 0}")
    print(f"NumPredictorGenes  = {predictor_table['gene'].nunique() if not predictor_table.empty else 0}")
    print(f"NumCombinedGenes   = {combined_table['gene'].nunique() if not combined_table.empty else 0}")
    print(f"MCCFunctionalMV (pooled) = {_mcc(functional_pooled):.3f}")
    print(f"MCCPredictorMV  (pooled) = {_mcc(predictor_pooled):.3f}")
    print(f"MCCCombinedMV   (pooled) = {_mcc(combined_pooled):.3f}")


# %% [markdown]
# ## CLI entry point

# %%
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", nargs="+", default=["all"],
                     choices=["all", "aggregate", "scatter", "results-tables", "predictor-individual",
                              "confusion", "ret", "card11", "tp53", "vus-brnich",
                              "vus-points-by-panel", "per-gene-diagnostics", "summary"])
    ap.add_argument("--path-percentile", type=float, default=RUN_KWARGS["path_percentile"],
                     help="Percentile of the bootstrap LR+ distribution a pathogenic call must clear "
                          "(ben_percentile derives as 100 - this, matching analysis.run()'s own "
                          "convention). Default matches mv_cockpit.py's CLI default (5).")
    ap.add_argument("--aux-path-percentile", type=float, default=None,
                     help="Separate percentile for auxiliary-pathogenic-sample scoring (RET/CARD11 "
                          "phenotype groups, TP53's RPV) -- defaults to --path-percentile if unset.")
    ap.add_argument("--aux-ben-percentile", type=float, default=None,
                     help="Separate benign-side percentile for auxiliary samples -- defaults to "
                          "100 - --aux-path-percentile if unset.")
    ap.add_argument("--results-cache-dir", default=None,
                     help="Disk cache for per-gene results-table/pooling computations (each is a "
                          "100-bootstrap scoring pass, otherwise redone from scratch on every run "
                          "-- see run_gene_set_results's docstring). Defaults to "
                          "{PAPER_CACHE_DIR}/results_cache. Delete stale entries after a code/"
                          "hyperparameter change that could change the actual numbers -- this cache "
                          "does not know what changed, only whether a cached record exists.")
    ap.add_argument("--no-results-cache", action="store_true",
                     help="Disable results-table/pooling caching entirely (always recompute).")
    args = ap.parse_args()
    cache_dir = None if args.no_results_cache else (
        args.results_cache_dir or f"{config.PAPER_CACHE_DIR}/results_cache")
    run_all = "all" in args.sections
    sections = set(args.sections)

    run_kwargs = dict(RUN_KWARGS)
    run_kwargs["path_percentile"] = args.path_percentile
    if args.aux_path_percentile is not None:
        run_kwargs["aux_path_percentile"] = args.aux_path_percentile
    if args.aux_ben_percentile is not None:
        run_kwargs["aux_ben_percentile"] = args.aux_ben_percentile

    ensure_aggregated()

    if run_all or "scatter" in sections:
        run_gene_performance_scatter_all()

    functional_table = predictor_table = combined_table = pd.DataFrame()
    functional_pooled = predictor_pooled = combined_pooled = pd.DataFrame()
    if run_all or "results-tables" in sections:
        # "Functional" spans all 5 gene-sets (labelseq/integrated/card11/
        # tp53/fgfr) -- each also gets its own separate table/pooled-row
        # output file, not just the combined one, per explicit request.
        functional_table, functional_pooled, functional_by_gene_set = run_functional_results(
            run_kwargs=run_kwargs, cache_dir=cache_dir)
        for gene_set, (table, pooled) in functional_by_gene_set.items():
            out = Path(config.PAPER_FUNCTIONAL_DIR) / f"functional_{gene_set}_results_table.csv"
            table.to_csv(out, index=False)
            pooled.to_csv(Path(config.PAPER_POOLED_DIR) / f"functional_{gene_set}_pooled_metrics.csv", index=False)
            print(f"Saved {out}")

        predictor_table, predictor_pooled, _ = run_gene_set_results(
            "predictor", list(PREDICTOR_COMBINED_GENES), run_kwargs=run_kwargs, cache_dir=cache_dir)
        combined_table, combined_pooled, _ = run_gene_set_results(
            "combined", list(PREDICTOR_COMBINED_GENES), run_kwargs=run_kwargs, cache_dir=cache_dir)
        for name, results_dir, (table, pooled) in [
            ("functional_all", config.PAPER_FUNCTIONAL_DIR, (functional_table, functional_pooled)),
            ("predictor", config.PAPER_PREDICTOR_DIR, (predictor_table, predictor_pooled)),
            ("combined", config.PAPER_COMBINED_DIR, (combined_table, combined_pooled)),
        ]:
            out = Path(results_dir) / f"{name}_results_table.csv"
            table.to_csv(out, index=False)
            pooled.to_csv(Path(config.PAPER_POOLED_DIR) / f"{name}_pooled_metrics.csv", index=False)
            print(f"Saved {out}")

    if run_all or "predictor-individual" in sections:
        df = run_predictor_individual_comparison(run_kwargs=run_kwargs)
        out = Path(config.PAPER_PREDICTOR_DIR) / "predictor_individual_comparison.csv"
        df.to_csv(out, index=False)
        print(f"Saved {out}")

    if run_all or "confusion" in sections:
        # Each functional gene-set separately (matching results-tables), plus
        # predictor/combined -- NOT a combined "all functional" confusion
        # matrix (unlike the results-table's pooled row), since confusion
        # matrices are already naturally poolable by eye across separate
        # panels and mixing 5 very different gene-sets into one 3x2 would
        # obscure more than it shows.
        for gene_set, genes in list(FUNCTIONAL_GENE_SETS) + [
            ("predictor", list(PREDICTOR_COMBINED_GENES)),
            ("combined", list(PREDICTOR_COMBINED_GENES)),
        ]:
            run_confusion_matrices_for_gene_set(gene_set, genes, run_kwargs=run_kwargs)

    if run_all or "ret" in sections:
        run_ret_phenotype_evidence(run_kwargs=run_kwargs)

    if run_all or "card11" in sections:
        run_card11_phenotype_evidence(run_kwargs=run_kwargs)

    if run_all or "tp53" in sections:
        run_tp53_special_figures(run_kwargs=run_kwargs)

    if run_all or "vus-brnich" in sections:
        run_vus_and_brnich(run_kwargs=run_kwargs)

    if run_all or "vus-points-by-panel" in sections:
        run_clinvar_vus_points_by_panel()

    # Deliberately NOT included in "all" -- see run_per_gene_diagnostics's
    # docstring for the cost rationale; must be requested explicitly.
    if "per-gene-diagnostics" in sections:
        run_per_gene_diagnostics()

    if run_all or "summary" in sections:
        print_manuscript_summary(functional_table, predictor_table, combined_table,
                                  functional_pooled, predictor_pooled, combined_pooled)


# %%
if __name__ == "__main__" and not _running_as_notebook():
    main()

# %% [markdown]
# ## Jupyter cells-mode playground

# %%
if _running_as_notebook():
    ensure_aggregated()
