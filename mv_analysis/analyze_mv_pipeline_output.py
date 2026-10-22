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
# notebook runs *every* analysis `ExCALIBR-MV_latex_092026/ExCALIBR_MV.tex`
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
from mv_analysis import uv_sources, uv_agg
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
                eval_mask, labels = report._eval_labels(ms, analysis.p_idx, analysis.b_idx, gene_set=gene_set)
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


def run_ret_phenotype_specific_evidence(run_kwargs=None):
    """Manuscript Figure fig:mv_ret: MEN2A/B-specific vs Hirschsprung-specific
    vs benign evidence for RET (pheno.ret_phenotype_evidence -- post-hoc
    phenotype weights on the canonical fit, see its TODO(next rerun)), the
    same variants with no calibration at all (make_ret_class_3d.py), and the
    per-dimension phenotype histograms (Extended Data). Returns the evidence
    dict plus the per-class variant counts the manuscript reports."""
    import subprocess
    ev = pheno.ret_phenotype_evidence(config.PAPER_RESULTS_JSON, run_kwargs=run_kwargs or RUN_KWARGS)
    _, counts = pheno.plot_ret_phenotype_evidence_3d(
        ev, html_path=f"{config.PAPER_RET_EVIDENCE_DIR}/ret_phenotype_evidence_3d.html",
        png_path=f"{config.PAPER_RET_EVIDENCE_DIR}/ret_phenotype_evidence_3d.png")
    scripts = _ROOT / "mv_analysis" / "figures" / "scripts"
    for script in ("make_ret_class_3d.py", "make_ret_dimension_plots_with_phenotypes.py"):
        subprocess.run([sys.executable, str(scripts / script)], check=True)
    return ev, counts


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


def run_cartoons():
    """Conceptual figures (Figure 1, the Gaussian/MSN/CFUSN illustration, the
    1D-vs-2D separability illustration) and the GCK worked example (real
    canonical fit), written under PAPER_CARTOONS_DIR."""
    import subprocess
    scripts = _ROOT / "mv_analysis" / "figures" / "scripts"
    out = Path(config.PAPER_CARTOONS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    for script, target in (("make_fig1_motivation.py", "fig1_cartoon_mv.png"),
                           ("make_gaussian_msn_cfusn_cartoon.py", "gaussian_msn_cfusn_cartoon.png"),
                           ("make_slide_separability_cartoon.py", "slide_separability_cartoon.png")):
        subprocess.run([sys.executable, str(scripts / script), "--out", str(out / target)], check=True)
    subprocess.run([sys.executable, str(scripts / "make_gck_slideshow_calibration.py")], check=True)


def run_correlation_and_conflict():
    """Within-class correlation between evidence sources (correlation_analysis.py)
    and per-gene whole-panel correlation / independently-calibrated evidence
    conflict rates (panel_wide_analysis.py), under PAPER_CORRELATION_DIR."""
    import subprocess
    scripts = _ROOT / "mv_analysis" / "figures" / "scripts"
    for script in ("correlation_analysis.py", "panel_wide_analysis.py"):
        subprocess.run([sys.executable, str(scripts / script), "--out-dir", config.PAPER_CORRELATION_DIR],
                       check=True)


def run_manuscript_figures():
    """The composed manuscript figures (Figures 1, 2, 3-5, 7 and the P-KNN
    Extended Data figure), drawn from outputs the other sections write:
    conflict tables (correlation), the GCK precompute, the per-gene results
    tables, the TP53 RPV scores and the P-KNN / out-of-bag comparison."""
    import subprocess
    scripts = _ROOT / "mv_analysis" / "figures" / "scripts"
    for script in ("make_fig1_conflict.py", "make_fig2_method.py",
                   "make_panel_performance.py", "make_fig7_tp53_rpv.py", "make_pknn_comparison.py"):
        subprocess.run([sys.executable, str(scripts / script)], check=True)


# %% [markdown]
# ## Section 11: manuscript macros + figure export
#
# Every number the manuscript reports is a `\Name{}` macro written here to
# `{MANUSCRIPT_DIR}/Figures/generated/macros.tex` (analysis/manuscript_numbers.py's
# registry, the same one the univariate paper uses). Values render red for
# review; any macro the .tex uses that this run cannot compute is written as
# a red XX. Figures the .tex includes are copied into Figures/generated/.

# %%
# {file name in Figures/generated/: source under PAPER_FIGURES_DIR}
MANUSCRIPT_FIGURES = {
    "fig1_conflict.pdf": "cartoons/fig1_conflict.pdf",
    "fig2_method.pdf": "cartoons/fig2_method.pdf",
    "fig_performance_functional.pdf": "accuracy/fig_performance_functional.pdf",
    "fig_performance_predictor.pdf": "accuracy/fig_performance_predictor.pdf",
    "fig_performance_combined.pdf": "accuracy/fig_performance_combined.pdf",
    "functional_vus_points_sankey.pdf": "vus_reclassification/functional_vus_points_sankey.pdf",
    "predictor_vus_points_sankey.pdf": "vus_reclassification/predictor_vus_points_sankey.pdf",
    "combined_vus_points_sankey.pdf": "vus_reclassification/combined_vus_points_sankey.pdf",
    "ret_phenotype_evidence_3d.png": "ret_evidence/ret_phenotype_evidence_3d.png",
    "fig7_tp53_rpv_classification.pdf": "tp53_rpv/fig7_tp53_rpv_classification.pdf",
    "mv_confusion_matrices.png": "confusions/mv_confusion_matrices.png",
    "mv_brnich_comparison.png": "brnich_comparison/mv_brnich_comparison.png",
    "fig_pknn_comparison.pdf": "pknn/fig_pknn_comparison.pdf",
}


def _mcc_from_counts(tp, tn, fp, fn):
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def _pooled_from_table(table, method_prefix, keys=None):
    """Pool one method's per-gene rows (evidence-direction threshold) by
    summing confusion counts across genes -- the manuscript's convention. For
    MV, each gene contributes its canonical config (the highest-MCC config,
    as cockpit.pick_best_canonical_config selects); UV rows are unique."""
    t = table[(table["threshold"] == EVIDENCE_DIRECTION) & table["method"].str.startswith(method_prefix)]
    if t.empty:
        return None
    if method_prefix == "MV":
        t = config.select_canonical_mv_rows(t)
    if keys is not None:
        t = t[t.set_index(["gene", "cluster"]).index.isin(keys)]
    s = t[["TP", "TN", "FP", "FN", "total", "determinate"]].sum()
    return dict(mcc=_mcc_from_counts(s.TP, s.TN, s.FP, s.FN), coverage=s.determinate / s.total,
                n_genes=t["gene"].nunique(), mv_better=None, rows=t)


def _gene_label(g):
    g = str(g)
    return ["FGFR1", "FGFR2", "FGFR3", "FGFR4"] if g.upper().startswith("FGFR") else [g.upper()]


def run_predictor_baseline_comparison(genes=PREDICTOR_COMBINED_GENES):
    """Independent predictor calibration: our own per-predictor ExCALIBR fits
    vs Chen et al.'s gene-specific calibrations (the paper's baseline), on the
    same ClinVar evaluation set as the predictor results tables. One row per
    gene, source and predictor (plus the nonconflicting aggregate), at the
    evidence-direction threshold. Writes predictor_baseline_comparison.csv."""
    rows = []
    for gene in genes:
        try:
            ms = config.build_multiscoresets_for_gene_set("predictor", genes=[gene])[gene]
            dataset_name = config.canonical_dataset_name(gene, "predictor", results_json=config.PAPER_RESULTS_JSON)
            analysis = cockpit.build_gene_set_analysis(ms, gene.lower(), config.PAPER_RESULTS_JSON,
                                                       dataset_name=dataset_name, gene_set="predictor")
            eval_mask, labels = report._eval_labels(ms, analysis.p_idx, analysis.b_idx, gene_set="predictor")
        except Exception as e:  # noqa: BLE001
            print(f"  [{gene}] predictor baseline comparison skipped: {e}")
            continue
        for source in ("excalibr", "chen"):
            uv = uv_sources.load_predictor_mv_uv_points(gene, ms, source=source)
            if uv is None:
                continue
            names, mat = uv
            series = [(n.split("_")[0], mat[i]) for i, n in enumerate(names)]
            series.append(("aggregate", uv_agg.aggregate_nonconflicting(mat)))
            for predictor, pts in series:
                pts_eval = np.nan_to_num(np.asarray(pts, float)[eval_mask], nan=0.0)
                r = [x for x in report._rows_for_points(source, pts_eval, labels)
                     if x["threshold"] == EVIDENCE_DIRECTION][0]
                rows.append({"gene": gene, "source": source, "predictor": predictor, **r})
    out = pd.DataFrame(rows)
    path = Path(config.PAPER_PREDICTOR_DIR) / "predictor_baseline_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if not out.empty:
        agg = out[out["predictor"] == "aggregate"]
        print(agg.pivot_table(index="gene", columns="source", values=["mcc", "determinate"]).round(3).to_string())
    print(f"wrote {path}")
    return out


def run_combined_single_predictor_comparison(genes=PREDICTOR_COMBINED_GENES):
    """Combined-evidence analogue of run_predictor_baseline_comparison: the
    independent baseline that adds functional evidence (nonconflicting rule
    across assays) to ONE predictor's points (Chen et al.), for each predictor,
    alongside the standard baseline adding the predictors' nonconflicting
    aggregate -- on the combined evaluation set (variants with at least one
    functional and one predictor score). Writes
    combined_single_predictor_comparison.csv."""
    rows = []
    for gene in genes:
        try:
            ms = config.build_ms_for_gene_cluster("combined", gene, results_json=config.PAPER_RESULTS_JSON)
            dataset_name = config.canonical_dataset_name(gene, "combined", results_json=config.PAPER_RESULTS_JSON)
            analysis = cockpit.build_gene_set_analysis(ms, gene.lower(), config.PAPER_RESULTS_JSON,
                                                       dataset_name=dataset_name, gene_set="combined")
            eval_mask, labels = report._eval_labels(ms, analysis.p_idx, analysis.b_idx, gene_set="combined")
            functional = uv_sources.load_combined_uv_points(gene, ms)
            predictor = uv_sources.load_predictor_mv_uv_points(gene, ms)
        except Exception as e:  # noqa: BLE001
            print(f"  [{gene}] combined single-predictor comparison skipped: {e}")
            continue
        if functional is None or predictor is None:
            continue
        func = np.nan_to_num(uv_agg.aggregate_nonconflicting(functional[1]), nan=0.0)
        names, mat = predictor
        series = [(n.split("_")[0], mat[i]) for i, n in enumerate(names)]
        series.append(("aggregate", uv_agg.aggregate_nonconflicting(mat)))
        for pred, pts in series:
            total = func + np.nan_to_num(np.asarray(pts, float), nan=0.0)
            r = [x for x in report._rows_for_points("chen", total[eval_mask], labels)
                 if x["threshold"] == EVIDENCE_DIRECTION][0]
            rows.append({"gene": gene, "source": "chen", "predictor": pred, **r})
    out = pd.DataFrame(rows)
    path = Path(config.PAPER_COMBINED_DIR) / "combined_single_predictor_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if not out.empty:
        print(out.pivot_table(index="gene", columns="predictor", values="mcc").round(3).to_string())
    print(f"wrote {path}")
    return out


PKNN_PANEL_DIRS = {"functional": config.PAPER_FUNCTIONAL_DIR, "predictor": config.PAPER_PREDICTOR_DIR,
                   "combined": config.PAPER_COMBINED_DIR}
PKNN_METHODS = {  # setting file -> method label in the comparison tables
    "main": "P-KNN LOO",
    "gnomadreg": "P-KNN LOO (gnomAD regularization)",
    "N5": "P-KNN LOO (N=5)",
    "N25": "P-KNN LOO (N=25)",
    "genomewide": "P-KNN genome-wide",
}
def run_pknn_analyses(workers=4, inner=8):
    """Compute every P-KNN result the comparison uses (prep, leave-one-out
    settings, genome-wide predictors, score concordance, VUS); finished
    outputs are skipped. CPU-heavy on first run (hours)."""
    from mv_analysis.comparison_methods import pknn_runs
    pknn_runs.run_all(workers=workers, inner=inner)


def run_production_oob_evaluation(workers=12, inner=2):
    """Out-of-bag evaluation under production scoring on the paper's own
    bootstrap splits, for MV, P-KNN and supervised classifiers
    (mv_analysis/production_oob.py). Writes production_oob_comparison.csv:
    one row per dataset and method, evidence-direction metrics on the
    paper's evaluation rows; variants never out of bag are excluded for all
    methods."""
    from mv_analysis import production_oob
    from mv_analysis.comparison_methods import supervised
    import json
    root = production_oob.run_all(workers=workers, inner=inner)
    rows = []
    for f in sorted(root.glob("*.csv")):
        o = pd.read_csv(f)
        meta = json.load(open(Path(config.PKNN_DIR) / "prep" / f"{f.stem}.json"))
        never = int((o["n_oob"] == 0).sum())
        o = o[o["n_oob"] > 0]
        for method in ["mv_insample", "MV", "P-KNN", *supervised.MODELS]:
            if method not in o.columns:
                continue
            pts = o[method].to_numpy(float)
            r = [x for x in report._rows_for_points(method, np.nan_to_num(pts, nan=0.0), o["label"].to_numpy(int))
                 if x["threshold"] == EVIDENCE_DIRECTION][0]
            rows.append({"key": f.stem, "panel": meta["panel"], "gene_set": meta["gene_set"], "gene": meta["gene"],
                         "n_never_oob": never, "n_unscored": int(np.isnan(pts).sum()),
                         "min_oob_splits": int(o["n_oob"].min()), "det_rate": r["determinate"] / r["total"], **r})
    out = pd.DataFrame(rows)
    out_dir = Path(config.PAPER_PRODUCTION_OOB_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "production_oob_comparison.csv", index=False)
    if not out.empty:
        both = out[(out["n_pathogenic"] > 0) & (out["n_benign"] > 0)]
        print(both.pivot_table(index="panel", columns="method", values="mcc", aggfunc="median").round(3).to_string())
    print(f"wrote {out_dir / 'production_oob_comparison.csv'}")
    return out


def run_pknn_comparison():
    """P-KNN (Lin & Brandes 2026) beside MV and the independent baseline on
    every per-gene evaluation (comparison_methods/pknn_runs.py writes the
    out-of-sample points to config.PKNN_DIR; run_pknn_analyses runs it). Writes, without touching the existing tables:
      {panel}/{panel}_pknn_comparison.csv  one row per gene and P-KNN setting
          (results-table schema, evidence-direction threshold)
      pknn/pknn_per_gene.csv  MV (in-sample), UV, every P-KNN setting, and MV
          vs P-KNN out of bag on the paper's bootstrap splits, per gene
      pknn/pknn_vus.csv  VUS evidence split, P-KNN vs MV
      pknn/pknn_genomewide_controls.csv  genome-wide P-KNN vs MV out of bag:
          pathogenic evidence on B/LB, P/LP and VUS, per predictor gene
    P-KNN points use ExCALIBR's LR base at the gene prior (points column);
    points_pknn_logbase (P-KNN's own base) is summarized as a sensitivity."""
    import json
    root = Path(config.PKNN_DIR)
    out_dir = Path(config.PAPER_PKNN_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = {p.stem: json.load(open(p)) for p in sorted((root / "prep").glob("*.json"))}
    rows = []
    for key, m in metas.items():
        for setting, method in PKNN_METHODS.items():
            f = root / key / f"{setting}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[d["query"]]
            for col, tag in (("points", ""), ("points_pknn_logbase", " [P-KNN LR base]")):
                if col not in d.columns:
                    continue
                # P-KNN returns no score (NaN) when too many of its bootstraps
                # resample no P/LP variant (mutual-information weights 0/0):
                # scored as no evidence, and counted
                pts = d[col].to_numpy(float)
                r = [x for x in report._rows_for_points(method + tag, np.nan_to_num(pts, nan=0.0),
                                                        d["label"].to_numpy(int))
                     if x["threshold"] == EVIDENCE_DIRECTION][0]
                rows.append({"panel": m["panel"], "gene_set": m["gene_set"], "gene": m["gene"], "key": key,
                             "setting": setting, "prior": m["prior"], "fold_scheme": d["fold_scheme"].iloc[0],
                             "n_unscored": int(np.isnan(pts).sum()), **r})
    pk = pd.DataFrame(rows)
    if pk.empty:
        print("no P-KNN outputs under", root)
        return pk
    for panel, pdir in PKNN_PANEL_DIRS.items():
        sub = pk[pk["panel"] == panel]
        if not sub.empty:
            sub.to_csv(Path(pdir) / f"{panel}_pknn_comparison.csv", index=False)

    # one row per gene: MV / UV from the paper tables, P-KNN settings, MV OOB
    sys.path.insert(0, str(_ROOT / "mv_analysis" / "figures" / "scripts"))
    import make_panel_performance as mpp
    per = []
    for panel in PKNN_PANEL_DIRS:
        d = mpp.per_gene(Path(config.PAPER_FIGURES_DIR) / mpp.PANELS[panel][0])
        d = d.assign(panel=panel)
        per.append(d)
    per = pd.concat(per, ignore_index=True)
    gs_of = {(m["panel"], m["gene"]): k for k, m in metas.items()}
    per["key"] = [gs_of.get((p, g)) for p, g in zip(per["panel"], per["gene"])]
    main_pk = pk[~pk["method"].str.endswith("]")]
    wide = main_pk.pivot_table(index="key", columns="setting", values=["mcc", "determinate", "total"], aggfunc="first")
    wide.columns = [f"pknn_{s}_{v}" for v, s in wide.columns]
    for s in PKNN_METHODS:
        if f"pknn_{s}_determinate" in wide.columns:
            wide[f"pknn_{s}_det"] = wide[f"pknn_{s}_determinate"] / wide[f"pknn_{s}_total"]
    per = per.merge(wide, left_on="key", right_index=True, how="left")
    # matched out-of-bag comparison under production scoring (section
    # "production-oob", run_production_oob_evaluation)
    oob_f = Path(config.PAPER_PRODUCTION_OOB_DIR) / "production_oob_comparison.csv"
    if oob_f.exists():
        o = pd.read_csv(oob_f)
        o = o[o["method"].isin(["MV", "P-KNN"])]
        w = o.pivot_table(index="key", columns="method", values=["mcc", "det_rate"], aggfunc="first")
        w.columns = [f"{'mv' if m == 'MV' else 'pknn'}_oob_{'det' if v == 'det_rate' else 'mcc'}" for v, m in w.columns]
        per = per.merge(w, left_on="key", right_index=True, how="left")
    per.to_csv(out_dir / "pknn_per_gene.csv", index=False)

    vus_f = root / "vus" / "pknn_vus_points.csv"
    if vus_f.exists():
        v = pd.read_csv(vus_f)
        rows = []
        for panel, setting in (("functional", "main"), ("predictor", "genomewide"), ("predictor", "main"),
                               ("combined", "main")):
            mv_f = Path(config.PAPER_VUS_RECLASSIFICATION_DIR) / f"{panel}_true_clinvar_vus_points.csv"
            gs = {"functional": ("labelseq", "integrated"), "predictor": ("predictor-mv",),
                  "combined": ("combined",)}[panel]
            sub = v[v["gene_set"].isin(gs) & (v["setting"] == setting)]
            if sub.empty or not mv_f.exists():
                continue
            mv = pd.read_csv(mv_f)
            mv["variant_key"] = mv["variant_key"].astype(str)
            both = sub.merge(mv[["gene", "variant_key", "our_points"]], on=["gene", "variant_key"], how="inner")
            pk_name = {"main": "P-KNN per gene", "genomewide": "P-KNN genome-wide"}[setting]
            for method, col in (("MV", "our_points"), (pk_name, "pknn_points")):
                if method == "MV" and any(r["panel"] == panel and r["method"] == "MV" for r in rows):
                    continue
                x = both[col]
                rows.append(dict(panel=panel, method=method, n_vus=len(x), n_genes=both["gene"].nunique(),
                                 pathogenic=int((x >= 1).sum()), benign=int((x <= -1).sum()),
                                 indeterminate=int(((x > -1) & (x < 1)).sum())))
        pd.DataFrame(rows).to_csv(out_dir / "pknn_vus.csv", index=False)

    # genome-wide P-KNN per predictor gene: pathogenic evidence on B/LB and P/LP
    # controls (gene held out) beside joint calibration out of bag, and on VUS
    vus_gw = pd.DataFrame()
    mv_vus_f = Path(config.PAPER_VUS_RECLASSIFICATION_DIR) / "predictor_true_clinvar_vus_points.csv"
    if vus_f.exists() and mv_vus_f.exists():
        v = pd.read_csv(vus_f)
        v = v[(v["gene_set"] == "predictor-mv") & (v["setting"] == "genomewide")]
        mv = pd.read_csv(mv_vus_f)
        mv["variant_key"] = mv["variant_key"].astype(str)
        vus_gw = v.merge(mv[["gene", "variant_key", "our_points"]], on=["gene", "variant_key"], how="inner")
        vus_gw["gene"] = vus_gw["gene"].astype(str).str.upper()
    gw_rows = []
    for key, m in metas.items():
        gf, of = root / key / "genomewide.csv", root / "production_oob" / f"{key}.csv"
        if not (gf.exists() and of.exists()):
            continue
        c = pd.read_csv(gf).merge(pd.read_csv(of)[["variant", "MV"]], on="variant")
        r = {"gene": m["gene"], "key": key}
        for lab, tag in ((0, "blb"), (1, "plp")):
            x = c[c["label"] == lab]
            r[f"n_{tag}"] = len(x)
            r[f"{tag}_path_pknn_gw"] = int((np.nan_to_num(x["points"].to_numpy(float)) >= 1).sum())
            r[f"{tag}_path_mv_oob"] = int((x["MV"] >= 1).sum())
        x = vus_gw[vus_gw["gene"] == str(m["gene"]).upper()] if len(vus_gw) else vus_gw
        r["n_vus"] = len(x)
        r["vus_path_pknn_gw"] = int((x["pknn_points"] >= 1).sum()) if len(x) else 0
        r["vus_path_mv"] = int((x["our_points"] >= 1).sum()) if len(x) else 0
        gw_rows.append(r)
    if gw_rows:
        pd.DataFrame(gw_rows).to_csv(out_dir / "pknn_genomewide_controls.csv", index=False)
    show = per[["panel", "gene", "mv_mcc", "uv_mcc"] + [c for c in ("mv_oob_mcc", "pknn_oob_mcc", "pknn_main_mcc",
                                                                     "pknn_genomewide_mcc")
                                                       if c in per.columns]]
    print(show.round(3).to_string(index=False))
    print(f"wrote {out_dir}")
    return per


def run_manuscript(ret_counts=None, ret_ev=None, strict=False):
    """Register every manuscript number from the canonical outputs on disk,
    write Figures/generated/macros.tex, and export the included figures."""
    import shutil
    import re as _re
    from analysis.manuscript_numbers import ManuscriptRegistry

    REG = ManuscriptRegistry(
        tex_paths=[config.MANUSCRIPT_MAIN_TEX, config.MANUSCRIPT_SUPP_TEX],
        out_dir=config.MANUSCRIPT_GENERATED_DIR, repo_root=_ROOT, strict=strict)
    figs = Path(config.PAPER_FIGURES_DIR)

    # -- run configuration ------------------------------------------------
    import gzip
    with gzip.open(config.PAPER_RESULTS_JSON, "rt") as f:
        raw = json.load(f)
    n_boot = {k: len(v) for k, v in raw.items()}
    REG.num("MethNumBootstraps", int(np.median(list(n_boot.values()))), kind="method",
            source="median bootstrap count per dataset in PAPER_RESULTS_JSON", inputs=[config.PAPER_RESULTS_JSON])
    ks = set()
    for k, boots in raw.items():
        if k.startswith("TP53_tp53"):
            continue
        b0 = next(iter(boots.values()))
        ks |= {int(c.split("c_")[0]) for c in b0 if b0[c] is not None and _re.match(r"\d+c_", c)}
    REG.num("MethComponentsLow", min(ks), kind="method", source="configs in PAPER_RESULTS_JSON")
    REG.num("MethComponentsHigh", max(ks), kind="method", source="configs in PAPER_RESULTS_JSON")
    tp53_ks = set()
    for k, boots in raw.items():
        if k.startswith("TP53_tp53"):
            b0 = next(iter(boots.values()))
            tp53_ks |= {int(c.split("c_")[0]) for c in b0 if b0[c] is not None}
    if tp53_ks:
        REG.num("MethTPFiftyThreeComponentsLow", min(tp53_ks), kind="method", source="TP53_tp53_mv configs")
        REG.num("MethTPFiftyThreeComponentsHigh", max(tp53_ks), kind="method", source="TP53_tp53_mv configs")
    # restarts per (bootstrap, K): from the array logs' "N bootstraps, M fits" header
    log_dir = Path(config.PAPER_FITS_DIR) / "logs"
    per = []
    for log in sorted(log_dir.glob("array_*.log"))[:200]:
        m = _re.search(r"(\d+) bootstraps, (\d+) fits", log.read_text(errors="ignore")[:500])
        if m and int(m.group(1)):
            per.append(int(m.group(2)) / int(m.group(1)))
    if per:
        REG.num("MethNumRestarts", int(round(np.median(per) / len(ks))), kind="method",
                source=f"array logs: median fits per bootstrap / {len(ks)} component counts")
    REG.num("MethPathPercentile", RUN_KWARGS["path_percentile"], kind="method", source="RUN_KWARGS")
    REG.num("MethBenPercentile", 100 - RUN_KWARGS["path_percentile"], kind="method", source="RUN_KWARGS")

    # -- dimensions ----------------------------------------------------------
    def _ndim(key):
        boots = raw.get(key)
        if not boots:
            return None
        for b in boots.values():
            for c in b.values():
                if c and c.get("fit") and c["fit"].get("component_params"):
                    n_cat = len((c["fit"].get("cat_spec") or {}).get("positions", []))
                    return len(c["fit"]["component_params"][0][0]) + n_cat
        return None
    REG.num("MethTPFiftyThreeDims", _ndim("TP53_tp53_mv"), kind="method", source="TP53_tp53_mv fit dimension")
    tp53c = _ndim("TP53_combined_mv")
    if tp53c is not None:
        REG.num("MethTPFiftyThreeCombinedFunctionalDims", tp53c - 3, kind="method",
                source="TP53_combined_mv fit dimension minus 3 predictors")
    try:
        from src.assay_calibration.multivariate_data import tp53 as _tp53
        REG.num("MethTPFiftyThreeDimsRaw", len(_tp53.SCORE_COLS), kind="method", source="tp53.SCORE_COLS")
    except Exception as e:
        print(f"  TP53 raw dimension count unavailable: {e}")

    # -- gene counts + pooled performance ----------------------------------
    tables = {
        "Functional": Path(config.PAPER_FUNCTIONAL_DIR) / "functional_all_results_table.csv",
        "Predictor": Path(config.PAPER_PREDICTOR_DIR) / "predictor_results_table.csv",
        "Combined": Path(config.PAPER_COMBINED_DIR) / "combined_results_table.csv",
    }
    all_genes = set()
    for panel, path in tables.items():
        if not REG.require(path, f"{panel} per-gene results table"):
            continue
        t = pd.read_csv(path)
        genes = set()
        for g in t["gene"].unique():
            genes.update(_gene_label(g))
        all_genes |= genes
        REG.num(f"Num{panel}Genes", len(genes), source=str(path), inputs=[path], words=True)
        # Pool MV and UV over the SAME genes (those with both), so the two
        # pooled numbers compare like with like.
        mv_all, uv_all = _pooled_from_table(t, "MV"), _pooled_from_table(t, "UV non-conflicting")
        mv = uv = None
        if mv_all and uv_all:
            keys = (set(map(tuple, mv_all["rows"][["gene", "cluster"]].values))
                    & set(map(tuple, uv_all["rows"][["gene", "cluster"]].values)))
            mv, uv = _pooled_from_table(t, "MV", keys), _pooled_from_table(t, "UV non-conflicting", keys)
            missing_uv = sorted({g for g, _ in set(map(tuple, mv_all["rows"][["gene", "cluster"]].values)) - keys})
            if missing_uv:
                REG._fail_soft("no-uv-baseline", f"{panel}: no UV rows for {missing_uv} (excluded from pooled MCC)")
        if mv and uv:
            REG.num(f"ResMCC{panel}MV", mv["mcc"], fmt=".2f", source=f"pooled MV, {path.name}", inputs=[path])
            REG.num(f"ResMCC{panel}UV", uv["mcc"], fmt=".2f", source=f"pooled UV, {path.name}", inputs=[path])
            REG.pct(f"ResDetRate{panel}MV", mv["coverage"], 1, digits=1, source=f"pooled MV determinate rate, {path.name}")
            REG.pct(f"ResDetRate{panel}UV", uv["coverage"], 1, digits=1, source=f"pooled UV determinate rate, {path.name}")
            # Per-gene comparison (abstract + scatter): paired over genes with
            # both MV and UV rows, excluding genes whose evaluated set lacks a
            # class (MCC undefined -- the scatter's own filter). A method making
            # no determinate calls in a gene scores MCC 0 there.
            m = mv["rows"].set_index(["gene", "cluster"])
            u = uv["rows"].set_index(["gene", "cluster"])
            both = m.index.intersection(u.index)
            m, u = m.loc[both], u.loc[both]
            ok = (m["n_pathogenic"] > 0) & (m["n_benign"] > 0)
            m, u = m[ok], u[ok]
            diff = m["mcc"] - u["mcc"]
            REG.num(f"ResMedMCC{panel}MV", float(m["mcc"].median()), fmt=".2f",
                    source=f"median per-gene MV MCC over {len(m)} genes with both classes, {path.name}")
            REG.num(f"ResMedMCC{panel}UV", float(u["mcc"].median()), fmt=".2f",
                    source=f"median per-gene UV MCC over {len(u)} genes with both classes, {path.name}")
            REG.num(f"Res{panel}GenesImproved", int((diff > 1e-9).sum()), source="per-gene MV MCC > UV MCC",
                    words=True)
            REG.num(f"Res{panel}GenesWorse", int((diff < -1e-9).sum()), source="per-gene MV MCC < UV MCC",
                    words=True)
            REG.num(f"Res{panel}GenesCompared", len(diff),
                    source="genes with both MV and UV and both classes", words=True)
    REG.num("NumGenesTotal", len(all_genes), source="union of functional/predictor/combined genes, "
            "case-normalized, FGFR1-4 separate", words=True)
    REG.num("NumLabelseqGenes", len(config.LABELSEQ_GENES), source="config.LABELSEQ_GENES", words=True)

    # -- VUS -----------------------------------------------------------------
    for panel in ("Functional", "Predictor", "Combined"):
        p = Path(config.PAPER_VUS_RECLASSIFICATION_DIR) / f"{panel.lower()}_true_clinvar_vus_points.csv"
        if not REG.require(p, f"{panel} ClinVar VUS points"):
            continue
        d = pd.read_csv(p)
        pts = d["our_points"].fillna(0)
        REG.num(f"ResVUS{panel}N", len(d), source=p.name, inputs=[p])
        REG.num(f"ResVUS{panel}Genes", d["gene"].nunique(), source=p.name, words=True)
        REG.pct(f"ResVUS{panel}DetPct", int((pts != 0).sum()), len(d), source=p.name)
        REG.pct(f"ResVUS{panel}PathPct", int((pts > 0).sum()), len(d), source=p.name)
        REG.pct(f"ResVUS{panel}BenPct", int((pts < 0).sum()), len(d), source=p.name)

    # -- conflict + correlation (panel_wide_analysis / correlation_analysis) --
    cdir = Path(config.PAPER_CORRELATION_DIR)
    for panel in ("functional", "predictor", "combined"):
        p = cdir / f"panel_{panel}_by_gene.csv"
        if not REG.require(p, f"{panel} panel-wide conflict table"):
            continue
        d = pd.read_csv(p).dropna(subset=["pct_conflict"])
        cap = panel.capitalize()
        REG.pct(f"ResConflict{cap}Pct", int(d["n_gnomad_conflicting"].sum()),
                int(d["n_gnomad_for_conflict"].sum()),
                source=f"{p.name}: conflicting / all population variants (same denominator as pct_conflict)")
        top = d.sort_values("pct_conflict", ascending=False).iloc[0]
        REG.num(f"ResConflict{cap}MaxPct", float(top["pct_conflict"]), fmt=".1f", source=p.name)
        REG.num(f"ResConflict{cap}Genes", len(d), source=f"{p.name}: genes in this conflict-rate row")
        REG.text(f"ResConflict{cap}MaxGene", f"\\textit{{{top['gene']}}}", source=p.name)
    for label, fname in (("Functional", "functional_correlation_table.csv"),
                         ("Predictor", "predictor_correlation_table.csv"),
                         ("Cross", "cross_correlation_table.csv")):
        p = cdir / fname
        if not REG.require(p, f"{label} correlation table"):
            continue
        d = pd.read_csv(p)
        d["abs_rho"] = d["spearman_rho"].abs()
        for role_value, short in (("P/LP", "PLP"), ("B/LB", "BLB"), ("gnomAD", "GnomAD")):
            sub = d[d["class"] == role_value].dropna(subset=["abs_rho"])
            if sub.empty:
                continue
            REG.num(f"ResMedianRho{label}{short}", float(sub["abs_rho"].median()), fmt=".2f", source=fname)
            REG.num(f"ResCorrPairs{label}{short}", len(sub), source=fname)
            if "gene" in sub.columns:
                REG.num(f"ResCorrGenes{label}{short}", sub["gene"].nunique(), source=fname)

    ev = cdir / "cross_evidence_correlation_table.csv"
    if REG.require(ev, "functional-vs-predictor evidence correlation table"):
        d = pd.read_csv(ev)
        for role_value, short in (("P/LP", "PLP"), ("B/LB", "BLB"), ("gnomAD", "GnomAD")):
            sub = d[d["class"] == role_value].dropna(subset=["rho_strength"])
            if sub.empty:
                continue
            REG.num(f"ResMedianRhoCrossEv{short}", float(sub["rho_strength"].abs().median()), fmt=".2f",
                    source=f"{ev.name}: median |Spearman| of aggregated functional vs predictor points")
            REG.num(f"ResCorrGenesCrossEv{short}", sub["gene"].nunique(), source=ev.name)

    # Dependence of independently calibrated evidence POINTS given the true
    # class (correlation_analysis.evidence_point_dependence and its summaries).
    cats = {"functional-functional": "Func", "predictor-predictor": "Pred",
            "functional-predictor": "Cross", "functional-predictor (aggregated)": "CrossAgg"}
    classes = {"P/LP": "PLP", "B/LB": "BLB", "gnomAD": "GnomAD"}
    pts = cdir / "evidence_point_dependence_table.csv"
    if REG.require(pts, "evidence-point dependence table"):
        d = pd.read_csv(pts)
        for (cat, cls), sub in d.dropna(subset=["rho_points"]).groupby(["category", "class"]):
            if cat not in cats or cls not in classes:
                continue
            tag = f"{cats[cat]}{classes[cls]}"
            REG.num(f"ResPtsRho{tag}", float(sub["rho_points"].median()), fmt=".2f",
                    source=f"{pts.name}: median Spearman of evidence points, {cat}, {cls}")
            REG.num(f"ResPtsPairs{tag}", len(sub), source=pts.name)
            REG.num(f"ResPtsGenes{tag}", sub["gene"].nunique(), source=pts.name)
    err = cdir / "evidence_error_dependence_summary.csv"
    if REG.require(err, "evidence error co-occurrence summary"):
        d = pd.read_csv(err)
        for r in d.to_dict("records"):
            if r["category"] not in cats or r["class"] not in classes:
                continue
            tag = f"{cats[r['category']]}{classes[r['class']]}"
            REG.num(f"ResErrGenes{tag}", int(r["genes"]), source=f"{err.name}: genes in the error analysis")
            REG.num(f"ResErrGivenErr{tag}", 100 * r["p_err2_given_err1"], fmt=".1f",
                    source=f"{err.name}: % wrong-direction evidence from one source when the other is wrong")
            REG.num(f"ResErrGivenOk{tag}", 100 * r["p_err2_given_not_err1"], fmt=".1f",
                    source=f"{err.name}: % wrong-direction evidence from one source when the other is not")
            if pd.notna(r.get("mh_or", np.nan)):
                REG.num(f"ResErrMHOR{tag}", float(r["mh_or"]), fmt=".1f", source=f"{err.name}: Mantel-Haenszel OR")
                REG.text(f"ResErrMHP{tag}", f"{r['mh_p']:.1g}" if r["mh_p"] >= 1e-3 else "$<$0.001",
                         source=f"{err.name}: Mantel-Haenszel test of OR = 1")
    chk = cdir / "evidence_summing_check.csv"
    if REG.require(chk, "evidence summing check"):
        d = pd.read_csv(chk)
        for r in d.itertuples():
            if r.category not in cats:
                continue
            tag = f"{cats[r.category]}{r.direction.capitalize()}"
            for col, name in (("implied_lr_both", "Implied"), ("observed_lr_both", "Observed"),
                              ("observed_over_implied", "Ratio")):
                v = getattr(r, col)
                if np.isfinite(v):
                    fmt = ".2f" if name == "Ratio" else (".0f" if v >= 10 else ".2g")
                    REG.num(f"ResSumLR{name}{tag}", float(v), fmt=fmt,
                            source=f"{chk.name}: {col}, pooled over genes and source pairs")

    # Independent predictor calibration: ExCALIBR vs Chen et al. (baseline).
    pb = Path(config.PAPER_PREDICTOR_DIR) / "predictor_baseline_comparison.csv"
    if REG.require(pb, "predictor baseline comparison"):
        d = pd.read_csv(pb)
        both = d[(d["n_pathogenic"] > 0) & (d["n_benign"] > 0)]
        for src, short in (("excalibr", "Excalibr"), ("chen", "Chen")):
            for pred, pshort in (("aggregate", ""), ("REVEL", "REVEL"), ("AM", "AM"), ("MP2", "MP")):
                sub = both[(both["source"] == src) & (both["predictor"] == pred)]
                if sub.empty:
                    continue
                REG.num(f"ResPredBase{pshort}MedMCC{short}", float(sub["mcc"].median()), fmt=".2f",
                        source=f"{pb.name}: median per-gene MCC, {src}, {pred}")
                s = d[(d["source"] == src) & (d["predictor"] == pred)][["determinate", "total"]].sum()
                REG.pct(f"ResPredBase{pshort}DetRate{short}", int(s["determinate"]), int(s["total"]),
                        source=f"{pb.name}: pooled determinate rate, {src}, {pred}")
        agg = both[both["predictor"] == "aggregate"].pivot_table(index="gene", columns="source", values="mcc")
        if {"excalibr", "chen"} <= set(agg.columns):
            agg = agg.dropna()
            REG.num("ResPredBaseGenes", len(agg), source=f"{pb.name}: genes with both classes")
            REG.num("ResPredBaseChenBetter", int((agg["chen"] > agg["excalibr"] + 1e-9).sum()), source=pb.name)
            REG.num("ResPredBaseChenWorse", int((agg["chen"] < agg["excalibr"] - 1e-9).sum()), source=pb.name)

    # Joint calibration against baselines built from each predictor alone
    # (Figures 4C and 5C), counted on the same genes the figures plot.
    sys.path.insert(0, str(_ROOT / "mv_analysis" / "figures" / "scripts"))
    import make_panel_performance as mpp_single
    for panel, ptag in (("predictor", "Pred"), ("combined", "Comb")):
        src = Path(config.PAPER_FIGURES_DIR) / mpp_single.SINGLE_PREDICTOR_PANELS[panel][0]
        if not REG.require(src, f"{panel} single-predictor comparison"):
            continue
        d = mpp_single.per_gene(Path(config.PAPER_FIGURES_DIR) / mpp_single.PANELS[panel][0])
        s = mpp_single.single_predictor_mcc(d, panel)
        for key, short in (("REVEL", "REVEL"), ("AM", "AM"), ("MP2", "MP")):
            diff = (s["mv_mcc"] - s[f"{key}_mcc"]).dropna()
            for name, val in (("Better", (diff > 1e-9).sum()), ("Tied", (diff.abs() <= 1e-9).sum()),
                              ("Worse", (diff < -1e-9).sum()), ("Genes", len(diff))):
                REG.num(f"ResSingle{ptag}{short}{name}", int(val),
                        source=f"{src.name}: joint vs {key}-only baseline, per-gene MCC")

    # P-KNN (Lin & Brandes 2026) against joint calibration, both out of sample:
    # genome-wide P-KNN (gene held out) for predictors (Figure 4D), and P-KNN
    # calibrated per gene on the joint model's bootstrap splits (Extended
    # Data); joint calibration out of bag on its own splits, production rule
    pk_dir = Path(config.PAPER_PKNN_DIR)
    pg = pk_dir / "pknn_per_gene.csv"
    if REG.require(pg, "P-KNN per-gene comparison"):
        d = pd.read_csv(pg)
        d = d[d["both_classes"]]

        def _pair(tag, x, a, b, det_a, det_b, what):
            diff = x[a] - x[b]
            REG.num(f"{tag}Genes", len(x), source=f"{pg.name}: {what}, genes with both classes")
            REG.num(f"{tag}MedMCCMV", float(x[a].median()), fmt=".2f", source=f"{pg.name}: {a} median")
            REG.num(f"{tag}MedMCC", float(x[b].median()), fmt=".2f", source=f"{pg.name}: {b} median")
            REG.num(f"{tag}MedDetMV", 100 * float(x[det_a].median()), fmt=".1f", source=f"{pg.name}: {det_a} median (%)")
            REG.num(f"{tag}MedDet", 100 * float(x[det_b].median()), fmt=".1f", source=f"{pg.name}: {det_b} median (%)")
            REG.num(f"{tag}MVBetter", int((diff > 1e-9).sum()), source=f"{pg.name}: {a} > {b}")
            REG.num(f"{tag}MVWorse", int((diff < -1e-9).sum()), source=f"{pg.name}: {a} < {b}")

        # sensitivity of the per-gene comparison to P-KNN's settings: median
        # per-gene MCC difference (joint out of bag minus P-KNN leave-one-out);
        # N_cal 50 is not run, as it reaches the size of the smaller genes
        meds = [float((x["mv_oob_mcc"] - x[f"pknn_{st}_mcc"]).dropna().median())
                for _, x in d.groupby("panel") for st in ("main", "N5", "N25", "gnomadreg")
                if f"pknn_{st}_mcc" in x.columns and x[f"pknn_{st}_mcc"].notna().any()]
        if meds:
            REG.num("ResPKNNSensMin", min(meds), fmt=".2f", source=f"{pg.name}: smallest median MCC gap over settings")
            REG.num("ResPKNNSensMax", max(meds), fmt=".2f", source=f"{pg.name}: largest median MCC gap over settings")
        pred = d[d["panel"] == "predictor"].dropna(subset=["mv_oob_mcc", "pknn_genomewide_mcc"])
        if not pred.empty:
            _pair("ResPKNNPredGW", pred, "mv_oob_mcc", "pknn_genomewide_mcc", "mv_oob_det", "pknn_genomewide_det",
                  "joint out of bag vs genome-wide P-KNN")
        for panel, ptag in (("functional", "Func"), ("predictor", "Pred"), ("combined", "Comb")):
            x = d[d["panel"] == panel].dropna(subset=["mv_oob_mcc", "pknn_oob_mcc"])
            if not x.empty:
                _pair(f"ResPKNN{ptag}OOB", x, "mv_oob_mcc", "pknn_oob_mcc", "mv_oob_det", "pknn_oob_det",
                      f"{panel}: joint vs per-gene P-KNN, both out of bag")
    oob_f = Path(config.PAPER_PRODUCTION_OOB_DIR) / "production_oob_comparison.csv"
    if REG.require(oob_f, "production-scored out-of-bag evaluation"):
        o = pd.read_csv(oob_f)
        o = o[o["method"] == "MV"]
        REG.num("ResOOBMinSplits", int(o["min_oob_splits"].min()), source=f"{oob_f.name}: fewest out-of-bag fits")
        REG.num("ResOOBNeverMax", int(o["n_never_oob"].max()), source=f"{oob_f.name}: most never-out-of-bag variants")
        n_oob = np.concatenate([pd.read_csv(f)["n_oob"].to_numpy() for f in
                                sorted((Path(config.PKNN_DIR) / "production_oob").glob("*.csv"))])
        n_oob = n_oob[n_oob > 0]
        for name, q in (("ResOOBMedSplits", 50), ("ResOOBSplitsLo", 25), ("ResOOBSplitsHi", 75)):
            REG.num(name, int(round(np.percentile(n_oob, q))), source="production_oob/*.csv: out-of-bag fits per variant")
    pv = pk_dir / "pknn_vus.csv"
    if REG.require(pv, "P-KNN VUS evidence"):
        v = pd.read_csv(pv)
        for panel, ptag, method in (("functional", "Func", "P-KNN per gene"), ("predictor", "Pred", "P-KNN genome-wide"),
                                    ("combined", "Comb", "P-KNN per gene")):
            r = v[(v["panel"] == panel) & (v["method"] == method)]
            if r.empty:
                continue
            r = r.iloc[0]
            REG.pct(f"ResPKNNVUS{ptag}PathPct", int(r["pathogenic"]), int(r["n_vus"]), source=f"{pv.name}: {method}")
            REG.pct(f"ResPKNNVUS{ptag}BenPct", int(r["benign"]), int(r["n_vus"]), source=f"{pv.name}: {method}")
    gc = pk_dir / "pknn_genomewide_controls.csv"
    if REG.require(gc, "genome-wide P-KNN evidence on controls and VUS"):
        g = pd.read_csv(gc)
        g["blb_rate"] = g["blb_path_pknn_gw"] / g["n_blb"]
        g = g.sort_values("blb_rate", ascending=False)
        for (_, r), tag in zip(g.head(2).iterrows(), ("One", "Two")):
            REG.text(f"ResPKNNGWBLBGene{tag}", f"\\textit{{{r['gene']}}}", source=f"{gc.name}: B/LB given pathogenic evidence, rank {tag}")
            REG.pct(f"ResPKNNGWBLBPct{tag}", int(r["blb_path_pknn_gw"]), int(r["n_blb"]), source=f"{gc.name}: {r['gene']}")
        REG.num("ResPKNNGWBLBMVMaxPct", 100 * float((g["blb_path_mv_oob"] / g["n_blb"]).max()), fmt=".1f",
                source=f"{gc.name}: joint out of bag, largest per-gene B/LB pathogenic rate (%)")
        REG.pct("ResPKNNGWVUSTopTwoShare", int(g.head(2)["vus_path_pknn_gw"].sum()), int(g["vus_path_pknn_gw"].sum()),
                source=f"{gc.name}: share of genome-wide P-KNN pathogenic VUS in the two genes above")

    # -- RET -------------------------------------------------------------------
    if ret_ev is not None:
        REG.num("ResRETMenTwoN", int(ret_ev["masks"]["MEN2"].sum()), source="phenotype_evidence lists in ms")
        REG.num("ResRETHirschN", int(ret_ev["masks"]["Hirschsprung"].sum()), source="phenotype_evidence lists in ms")
        for g, short in (("MEN2", "MenTwo"), ("Hirschsprung", "Hirsch")):
            m = ret_ev["masks"][g]
            REG.num(f"ResRET{short}Correct", int((ret_ev["aux_points"][g][m] > 0).sum()),
                    source=f"{g} variants receiving {g}-specific pathogenic evidence")
            REG.num(f"ResRET{short}Prior", ret_ev["aux_prior"][g], fmt=".3f", source=f"{g} median prior")
            REG.num(f"ResRET{short}PLPPoints", int((ret_ev["points"][m] > 0).sum()),
                    source=f"{g} variants receiving pooled P/LP-vs-B/LB pathogenic evidence")
    if ret_counts:
        REG.num("ResRETBenignEvidenceN", ret_counts["benign"], source="ret 3D evidence classes")
        REG.num("ResRETMenTwoEvidenceN", ret_counts["men2"], source="ret 3D evidence classes")
        REG.num("ResRETHirschEvidenceN", ret_counts["hirschsprung"], source="ret 3D evidence classes")

    # -- Figure 1 example variant ----------------------------------------------
    f1 = Path(config.PAPER_CARTOONS_DIR) / "fig1_conflict_facts.json"
    if REG.require(f1, "Figure 1 example-variant facts"):
        f = json.loads(f1.read_text())
        REG.text("ResFigOneGene", f"\\textit{{{f['gene']}}}", source=f1.name)
        REG.text("ResFigOneVariant", f["variant_p"], source=f1.name)
        REG.text("ResFigOneAssayA", f["assay_a"].replace(" et al.", " \\textit{et al.}"), source=f1.name)
        REG.text("ResFigOneAssayB", f["assay_b"].replace(" et al.", " \\textit{et al.}"), source=f1.name)
        REG.text("ResFigOnePointsA", f"{f['points_a']:+d}", source=f1.name)
        REG.text("ResFigOnePointsB", f"{f['points_b']:+d}".replace("-", "$-$"), source=f1.name)
        REG.num("ResFigOnePriorA", f["prior_a"], fmt=".2f", source=f1.name)
        REG.num("ResFigOnePriorB", f["prior_b"], fmt=".2f", source=f1.name)

    # -- TP53 functional panel + reduced penetrance -----------------------------
    tp = Path(config.PAPER_FUNCTIONAL_DIR) / "functional_tp53_results_table.csv"
    if REG.require(tp, "TP53 functional results table"):
        t = pd.read_csv(tp)
        t = t[t["threshold"] == EVIDENCE_DIRECTION]
        mv_t = t[t["method"] == f"MV {TP53_CONFIG}"].head(1)
        uv_t = t[t["method"] == "UV non-conflicting"].head(1)
        if not mv_t.empty and not uv_t.empty:
            REG.num("ResTPFiftyThreeMCCMV", float(mv_t["mcc"].iloc[0]), fmt=".2f", source=tp.name)
            REG.num("ResTPFiftyThreeMCCUV", float(uv_t["mcc"].iloc[0]), fmt=".2f", source=tp.name)
            REG.pct("ResTPFiftyThreeDetRateMV", int(mv_t["determinate"].iloc[0]), int(mv_t["total"].iloc[0]), source=tp.name)
            REG.pct("ResTPFiftyThreeDetRateUV", int(uv_t["determinate"].iloc[0]), int(uv_t["total"].iloc[0]), source=tp.name)
            REG.num("ResTPFiftyThreeNControls", int(mv_t["total"].iloc[0]), source=tp.name)
    f7 = Path(config.PAPER_TP53_RPV_DIR) / "fig7_tp53_rpv_facts.json"
    if REG.require(f7, "TP53 reduced-penetrance classification counts"):
        f = json.loads(f7.read_text())
        short = {"B/LB": "BLB", "P/LP": "PLP", "Reduced penetrance": "RPV"}
        cls = {"benign_like": "Benign", "intermediate": "Intermediate", "plp_like": "Path", "low_pen_rpv": "LowPen"}
        for row, counts in f["counts"].items():
            REG.num(f"ResTPFiftyThree{short[row]}N", sum(counts), source=f7.name)
            for c, n in zip(f["classes"], counts):
                REG.num(f"ResTPFiftyThree{short[row]}{cls[c]}", n, source=f7.name)

    # -- Extended Data: per-gene performance table -------------------------------
    sys.path.insert(0, str(_ROOT / "mv_analysis" / "figures" / "scripts"))
    import make_panel_performance as mpp
    rows = []
    for panel, title in (("functional", "Functional"), ("predictor", "Computational"),
                         ("combined", "Combined")):
        d = mpp.per_gene(Path(config.PAPER_FIGURES_DIR) / mpp.PANELS[panel][0])
        d = d.sort_values("label")
        for _, r in d.iterrows():
            mcc = (f"{r['uv_mcc']:.2f} & {r['mv_mcc']:.2f}" if r["both_classes"] else "-- & --")
            rows.append(f"{title} & {r['label']} & {int(r['n']):,} & {mcc} & "
                        f"{100 * r['uv_det']:.0f} & {100 * r['mv_det']:.0f} \\\\")
    body = ("\\begin{tabular}{llrrrrr}\n\\toprule\n"
            "Evidence & Gene & Controls & \\multicolumn{2}{c}{MCC} & \\multicolumn{2}{c}{Determinate rate (\\%)} \\\\\n"
            " & & & Indep. & Joint & Indep. & Joint \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    tdir = Path(config.MANUSCRIPT_GENERATED_DIR) / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "mv_gene_performance.tex").write_text(body)
    REG.output(tdir / "mv_gene_performance.tex", "A", "Extended Data per-gene performance table")

    # -- Extended Data: out-of-bag comparison with supervised classifiers --------
    # on the genes every method can be evaluated on (classifiers need at least
    # five training controls of each class)
    if oob_f.exists():
        o = pd.read_csv(oob_f)
        names = {"MV": "Joint calibration", "P-KNN": "P-KNN (per gene)",
                 "B-LR": "Logistic regression", "B-LR-nomiss": "Logistic regression, no missingness indicators",
                 "B-spline": "Spline logistic regression", "B-GB": "Gradient-boosted trees",
                 "B-GB-cal": "Gradient-boosted trees, recalibrated"}
        classifiers = ["B-LR", "B-LR-nomiss", "B-spline", "B-GB", "B-GB-cal"]
        rows = []
        for panel, ptag, title in (("functional", "Func", "Functional"), ("predictor", "Pred", "Computational"),
                                   ("combined", "Comb", "Combined")):
            x = o[(o["panel"] == panel) & o["method"].isin(names)]
            mcc = x.pivot_table(index="key", columns="method", values="mcc", aggfunc="first").dropna()
            det = x.pivot_table(index="key", columns="method", values="det_rate", aggfunc="first").loc[mcc.index]
            first = True
            for mth in names:
                diff = mcc["MV"] - mcc[mth]
                bw = "-- & --" if mth == "MV" else f"{int((diff > 1e-9).sum())} & {int((diff < -1e-9).sum())}"
                lead = f"{title} ({len(mcc)})" if first else ""
                rows.append(f"{lead} & {names[mth]} & {mcc[mth].median():.2f} & {100 * det[mth].median():.0f} & {bw} \\\\")
                first = False
            rows.append("\\midrule")
            worse = {c: int((mcc["MV"] < mcc[c] - 1e-9).sum()) for c in classifiers}
            top = max(worse, key=worse.get)
            REG.num(f"ResSup{ptag}Genes", len(mcc), source=f"{oob_f.name}: {panel} genes evaluable by every method")
            REG.num(f"ResSup{ptag}MaxWorse", worse[top], source=f"{oob_f.name}: {panel}, most genes a classifier beats joint")
            REG.text(f"ResSup{ptag}MaxWorseMethod", names[top].lower(), source=f"{oob_f.name}: {panel}, that classifier")
            REG.num(f"ResSup{ptag}MaxWorseMVBetter", int((mcc["MV"] > mcc[top] + 1e-9).sum()),
                    source=f"{oob_f.name}: {panel}, genes joint beats that classifier")
            assert all(mcc[c].median() <= mcc["MV"].median() for c in classifiers), \
                f"{panel}: a classifier's median MCC exceeds joint calibration's; revise the text"
        body = ("\\begin{tabular}{llrrrr}\n\\toprule\n"
                "Evidence (genes) & Method & Median MCC & Median determinate rate (\\%) & Joint higher & Joint lower \\\\\n"
                "\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
        (tdir / "oob_supervised_baselines.tex").write_text(body)
        REG.output(tdir / "oob_supervised_baselines.tex", "A", "Extended Data out-of-bag comparison with classifiers")

    # -- figures -------------------------------------------------------------
    dest = Path(config.MANUSCRIPT_GENERATED_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    for name, rel in MANUSCRIPT_FIGURES.items():
        src = figs / rel
        if not src.exists():
            REG._fail_soft("missing-figure", f"{name}: {src} does not exist")
            continue
        _link_figure(src, dest / name)
        _record_linked_figure(REG, name, src)
    tp53 = sorted(Path(config.PAPER_TP53_RPV_DIR).glob("TP53_*_RPV_penetrance_hist.png"))
    best = _tp53_best_config()
    pick = next((p for p in tp53 if p.name.startswith(f"TP53_{best}_")), tp53[0] if tp53 else None)
    if pick is not None:
        _link_figure(pick, dest / "tp53_rpv_penetrance.png")
        _record_linked_figure(REG, "tp53_rpv_penetrance.png", pick)
    _prune_unreferenced_figures(REG, dest)

    REG.write(config.MANUSCRIPT_FIGURE_DIR, old_values={}, highlight_changed=True,
              placeholder_unregistered=True, generator="mv_analysis/analyze_mv_pipeline_output.py")
    return REG


def _link_figure(src, link):
    """Point the manuscript at the pipeline's own output file -- one copy of
    every figure, so the manuscript can never hold a stale duplicate."""
    link = Path(link)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path(src).resolve())


def _prune_unreferenced_figures(REG, dest):
    """Remove figure symlinks in the manuscript's generated folder that no .tex
    file includes, so copying the manuscript folder (e.g. to Overleaf) carries
    only the paper's own figures. Only symlinks are removed -- the pipeline's
    outputs stay where they are -- and macros/tables are never touched."""
    _, includes = REG._scan_tex()
    used = {Path(rel).name for _, _, rel in includes}
    used |= {n + ext for n in used for ext in (".pdf", ".png")}  # extensionless \includegraphics
    removed = []
    for f in Path(dest).iterdir():
        if f.is_symlink() and f.name not in used:
            f.unlink()
            removed.append(f.name)
    REG.figures = {k: v for k, v in REG.figures.items() if k not in removed}
    if removed:
        print(f"  Removed {len(removed)} figure link(s) the manuscript does not include: {', '.join(sorted(removed))}")


def _record_linked_figure(REG, name, src):
    """Record a symlinked figure in the build report and flag it if its
    source predates the fits it is drawn from (PAPER_RESULTS_JSON) -- the
    only way a linked figure can be stale, since there is no copy to drift."""
    import datetime as _dt
    src = Path(src)
    fits_mtime = Path(config.PAPER_RESULTS_JSON).stat().st_mtime
    stale = src.stat().st_mtime < fits_mtime
    REG.figures[name] = {"path": str(src), "exists": True, "mtime": src.stat().st_mtime,
                         "mtime_iso": _dt.datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
                         "fresh_this_run": not stale, "part": "A", "inputs": []}
    if stale:
        REG._fail_soft("stale-figure", f"{name}: {src} is older than {config.PAPER_RESULTS_JSON}")


TP53_CONFIG = config.FIXED_MV_CONFIGS[("TP53", "tp53")]  # 4 components throughout (author decision, 2026-10-07)


def _tp53_best_config():
    """TP53's configuration for every manuscript number and figure: fixed at
    TP53_CONFIG rather than chosen by MCC."""
    return TP53_CONFIG


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
                              "vus-points-by-panel", "per-gene-diagnostics", "summary",
                              "cartoons", "correlation", "ret-phenotype", "figures", "manuscript",
                              "predictor-baselines", "pknn", "production-oob"])
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

    ret_ev = ret_counts = None
    if run_all or "ret-phenotype" in sections or "manuscript" in sections:
        ret_ev, ret_counts = run_ret_phenotype_specific_evidence(run_kwargs=run_kwargs)

    if run_all or "predictor-baselines" in sections:
        run_predictor_baseline_comparison()
        run_combined_single_predictor_comparison()

    if run_all or "production-oob" in sections:
        run_production_oob_evaluation()

    if run_all or "pknn" in sections:
        run_pknn_analyses()
        run_pknn_comparison()

    if run_all or "cartoons" in sections:
        run_cartoons()

    if run_all or "correlation" in sections:
        run_correlation_and_conflict()

    if run_all or "figures" in sections:
        run_manuscript_figures()

    if run_all or "manuscript" in sections:
        run_manuscript(ret_counts=ret_counts, ret_ev=ret_ev)

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
