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
    """The composed manuscript figures (Figures 1, 2, 3-5 A-B, 7), drawn from
    outputs the other sections write: conflict tables (correlation), the GCK
    precompute, the per-gene results tables, and the TP53 RPV scores."""
    import subprocess
    scripts = _ROOT / "mv_analysis" / "figures" / "scripts"
    for script in ("make_fig1_conflict.py", "make_fig2_method.py",
                   "make_panel_performance.py", "make_fig7_tp53_rpv.py"):
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
        t = t.sort_values(["mcc", "determinate"], ascending=False).groupby(["gene", "cluster"], as_index=False).head(1)
    if keys is not None:
        t = t[t.set_index(["gene", "cluster"]).index.isin(keys)]
    s = t[["TP", "TN", "FP", "FN", "total", "determinate"]].sum()
    return dict(mcc=_mcc_from_counts(s.TP, s.TN, s.FP, s.FN), coverage=s.determinate / s.total,
                n_genes=t["gene"].nunique(), mv_better=None, rows=t)


def _gene_label(g):
    g = str(g)
    return ["FGFR1", "FGFR2", "FGFR3", "FGFR4"] if g.upper().startswith("FGFR") else [g.upper()]


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


TP53_CONFIG = "4c_unc"  # TP53 uses 4 components throughout (author decision, 2026-10-07)


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
                              "cartoons", "correlation", "ret-phenotype", "figures", "manuscript"])
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
