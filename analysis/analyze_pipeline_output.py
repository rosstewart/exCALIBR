# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: excalibr
#     language: python
#     name: excalibr
# ---

# %% [markdown]
# # Pipeline output analysis
#
# Reproduces, from `run_pipeline.py` / `run_igvf_batch.py` output only:
#
# - confusion matrices (ExCALIBR vs author, method vs method) — was
#   `test/plot_author_calibration_confusion.py`
# - evidence distributions and per-gene accuracy scatter
# - per-dataset calibration detail plots and the MSH2 example / "final pillar
#   project" style figures — was `test/plot_MSH2_ex.py`
# - the Yang-distance bootstrap diagnostic — was `test/yang_dist.py`
# - Figure 4, the extended-data appendix, and the gene-performance/OR scatter
#   — was `test/auxiliary_fig_creation/*.py`
#
# This file is a jupytext-paired notebook (percent format). Open it directly
# in Jupyter/VSCode, or run `jupytext --sync analyze_pipeline_output.py` to
# generate/refresh the paired `.ipynb`.
#
# Running it as a plain script (`python analyze_pipeline_output.py
# --output-dir ... --dataset ...`) instead runs the CLI `main()` below and
# exits — none of the notebook cells below execute in that mode, so CLI args
# are actually honored. Opening it in Jupyter (an `ipykernel` is detected)
# runs the notebook cells in order instead.
#
# **All paths are set in `analysis/config.py`** — edit that file (or set the
# matching `EXCALIBR_*` environment variable) to point this notebook at a
# different run. The cell below just displays the resolved values.

# %%
# Auto-reload analysis/*.py on every cell run, so edits to those files take
# effect immediately without restarting the kernel (plain `import` only reads
# a module's source once). `get_ipython()` is None outside Jupyter/IPython, so
# this is a no-op (and stays valid Python, unlike bare `%magic` syntax) when
# this file is run as a plain script instead.
_ip = get_ipython() if "get_ipython" in dir() else None
if _ip is not None:
    _ip.run_line_magic("load_ext", "autoreload")
    _ip.run_line_magic("autoreload", "2")

# %%
import warnings
warnings.filterwarnings("ignore")

import sys
import json
import argparse
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    """Walk upward from `start` looking for the repo root (has run_pipeline.py
    and an analysis/ package). Falls back to `start` if not found — e.g. when
    the notebook's cwd is somewhere unexpected."""
    for candidate in [start] + list(start.parents):
        if (candidate / "run_pipeline.py").exists() and (candidate / "analysis").is_dir():
            return candidate
    return start


# `__file__` is set when run as a script (`python analyze_pipeline_output.py`)
# but is NOT defined inside a Jupyter/IPython kernel — fall back to cwd there
# (Jupyter's cwd is normally the directory the notebook file lives in).
try:
    _start_dir = Path(__file__).resolve().parent
except NameError:
    _start_dir = Path.cwd()

_ROOT = _find_repo_root(_start_dir)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _running_as_notebook() -> bool:
    """True when executed by Jupyter/IPython rather than `python file.py`."""
    return "ipykernel" in sys.modules or "IPython" in sys.modules


import matplotlib
if _running_as_notebook():
    # Interactive/inline backend so figures actually render in cell output —
    # Agg (used for headless CLI runs below) can only ever write to disk.
    try:
        matplotlib.use("module://matplotlib_inline.backend_inline")
    except ImportError:
        pass  # fall back to whatever backend the kernel already configured
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import config
from analysis.discovery import discover_outputs, load_all_variants, resolve_component
from analysis.author_labels import attach_author_labels
from analysis.confusion import (
    build_confusion_matrix,
    build_author_confusion_matrix,
    build_vus_coverage,
    build_author_vus_coverage,
    build_both_determinate_confusion_matrices,
    make_confusion_figure,
    make_single_confusion_figure,
    make_confusion_grid_figure,
    _aggregate_coverage_pct,
)
from analysis.evidence import build_evidence_arrays, build_author_array, make_evidence_figure
from analysis.calibration_plots import load_lr_values, make_calibration_figure
from analysis.plot_common import save_and_show, save_latex_table

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATASET_TSV = config.DATASET_TSV
DATASET_CONFIGS_PATH = config.DATASET_CONFIGS
PRECOMPUTED_FITS = config.PRECOMPUTED_FITS
FIGURE_DIR = Path(config.FIGURE_DIR)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def figure_subdirs(figure_dir: Path) -> dict:
    """Named subfolders every figure-producing section below writes into,
    grouped by comparison/ablation type rather than left flat in
    `figure_dir` -- see each section's own call sites for which subfolder it
    uses. Created up front so no individual plotting call needs its own
    mkdir. `robustness_downsampling`/`robustness_label_noise` and
    `bootstrap_reduction`/`fit_number_comparison` additionally get their own
    dataset-level subfolders, created lazily by analysis/robustness.py
    itself (see plot_robustness_config_summary /
    plot_bootstrap_reduction_config_summary)."""
    subdirs = {
        "author": figure_dir / "clinvar_comparisons" / "author",
        "acmgscaler": figure_dir / "clinvar_comparisons" / "acmgscaler",
        "gmm_baseline": figure_dir / "clinvar_comparisons" / "gmm_baseline",
        "skew_locked": figure_dir / "clinvar_comparisons" / "skew_locked",
        "pathomechanism": figure_dir / "clinvar_comparisons" / "pathomechanism",
        "clingen": figure_dir / "clingen_comparisons",
        "path_percentile": figure_dir / "path_percentile_ablation",
        "manuscript": figure_dir / "manuscript_figures",
        "tables": figure_dir / "manuscript_figures" / "tables",
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return subdirs


FIGURE_SUBDIRS = figure_subdirs(FIGURE_DIR)

print(f"OUTPUT_DIR       = {OUTPUT_DIR}")
print(f"DATASET_TSV      = {DATASET_TSV}")
print(f"DATASET_CONFIGS  = {DATASET_CONFIGS_PATH}")
print(f"PRECOMPUTED_FITS = {PRECOMPUTED_FITS}")
print(f"FIGURE_DIR       = {FIGURE_DIR}")


# %% [markdown]
# ## CLI entry point
#
# Defined here (rather than at the bottom) so it can run *before* any of the
# exploratory cells below when this file is executed as a plain script.

# %%
def main():
    parser = argparse.ArgumentParser(
        description="Post-pipeline analysis: figures from run_igvf_batch.py / run_pipeline.py outputs",
    )
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--dataset", default=config.DATASET_TSV)
    parser.add_argument("--dataset-configs", default=config.DATASET_CONFIGS)
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--min-controls", type=int, default=0)
    parser.add_argument("--in-bag", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir) if args.figure_dir else output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_subdirs_ = figure_subdirs(figure_dir)

    print("=" * 80)
    print("PIPELINE OUTPUT ANALYSIS")
    print("=" * 80)

    tree, model_selections, calibrations = discover_outputs(output_dir)
    if not tree:
        print("ERROR: no *_variants.csv or *_calibration.json files found in output directory")
        sys.exit(1)

    with open(args.dataset_configs) as f:
        dataset_configs = json.load(f)

    df = load_all_variants(
        tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
        methods_filter=args.methods, datasets_filter=args.datasets,
        calibrations=calibrations, min_controls=args.min_controls,
        recompute_points=args.in_bag,
    )
    if df.empty:
        print("ERROR: no variants loaded — check --methods / --datasets filters")
        sys.exit(1)

    df = attach_author_labels(df, args.dataset)

    methods_ = sorted(df["method"].unique())
    datasets_ = sorted(df["dataset"].unique())
    use_oob_ = not args.in_bag

    conf_by_method_ = {m: [] for m in methods_}
    auth_by_method_ = {m: [] for m in methods_}
    vus_by_method_ = {m: [] for m in methods_}
    auth_vus_by_method_ = {m: [] for m in methods_}
    for dataset_ in datasets_:
        df_ds = df[df["dataset"] == dataset_]
        for method_ in methods_:
            df_m = df_ds[df_ds["method"] == method_]
            conf_by_method_[method_].append(
                build_confusion_matrix(df_m, use_oob=use_oob_, label=f"{dataset_}/{method_}") if not df_m.empty else None
            )
            auth_by_method_[method_].append(
                build_author_confusion_matrix(df_m, use_oob=use_oob_) if not df_m.empty else None
            )
            vus_by_method_[method_].append(
                build_vus_coverage(df_m, use_oob=use_oob_, label=f"{dataset_}/{method_}") if not df_m.empty else None
            )
            auth_vus_by_method_[method_].append(
                build_author_vus_coverage(df_m) if not df_m.empty else None
            )

    for m in methods_:
        make_single_confusion_figure(
            conf_by_method_[m], datasets_, label=m, figure_dir=figure_subdirs_["author"],
            vus_coverages=vus_by_method_[m], filename=f"excalibr_vs_clinvar_{m}.png",
        )

    primary_method_ = methods_[0]
    auths_ = auth_by_method_[primary_method_]
    if any(a is not None for a in auths_):
        make_confusion_figure(
            danzs_m1=conf_by_method_[primary_method_], danzs_m2=auths_,
            dataset_names=datasets_, label1=primary_method_, label2="author",
            figure_dir=figure_subdirs_["author"], filename="excalibr_vs_author.png",
            vus_coverages_m1=vus_by_method_[primary_method_],
            vus_coverages_m2=auth_vus_by_method_[primary_method_],
        )

    for method_ in methods_:
        df_m = df[df["method"] == method_]
        all_danz, all_clinvar = build_evidence_arrays(df_m)
        all_author = build_author_array(df_m)
        make_evidence_figure(
            all_danz, all_author, all_clinvar, label=method_, figure_dir=figure_subdirs_["manuscript"],
        )

    print(f"\n{'=' * 80}\nANALYSIS COMPLETE\n{'=' * 80}")
    print(f"Figures saved to: {figure_dir}")


# %%
if __name__ == "__main__" and not _running_as_notebook():
    main()
    sys.exit(0)

# %% [markdown]
# Everything below only runs interactively (Jupyter/VSCode notebook cells) —
# a plain `python analyze_pipeline_output.py ...` invocation exits above.

# %% [markdown]
# ## 1. Discover pipeline outputs and load variants

# %%
tree, model_selections, calibrations = discover_outputs(OUTPUT_DIR)
print(f"Discovered {len(tree)} datasets")

with open(DATASET_CONFIGS_PATH) as f:
    dataset_configs = json.load(f)
print(f"Loaded {len(dataset_configs)} dataset configs")

df = load_all_variants(
    tree=tree,
    model_selections=model_selections,
    dataset_configs=dataset_configs,
    methods_filter=None,
    datasets_filter=None,
    calibrations=calibrations,
    min_controls=0,
)
print(f"Loaded {len(df):,} variant rows across {df['dataset'].nunique()} datasets, "
      f"methods={sorted(df['method'].unique())}")

# %% [markdown]
# ## 2. Attach author labels (enables confusion vs. author + evidence-by-author panels)

# %%
df = attach_author_labels(df, DATASET_TSV)

# %% [markdown]
# ## 3. Confusion matrices — matches `test/plot_author_calibration_confusion.py`

# %%
methods = sorted(df["method"].unique())
datasets = sorted(df["dataset"].unique())
use_oob = True

conf_by_method = {m: [] for m in methods}
auth_by_method = {m: [] for m in methods}
vus_by_method = {m: [] for m in methods}
auth_vus_by_method = {m: [] for m in methods}
both_det_excalibr_by_method = {m: [] for m in methods}
both_det_author_by_method = {m: [] for m in methods}

for dataset in datasets:
    df_ds = df[df["dataset"] == dataset]
    for method in methods:
        df_m = df_ds[df_ds["method"] == method]
        conf_by_method[method].append(
            build_confusion_matrix(df_m, use_oob=use_oob, label=f"{dataset}/{method}") if not df_m.empty else None
        )
        auth_by_method[method].append(
            build_author_confusion_matrix(df_m, use_oob=use_oob) if not df_m.empty else None
        )
        vus_by_method[method].append(
            build_vus_coverage(df_m, use_oob=use_oob, label=f"{dataset}/{method}") if not df_m.empty else None
        )
        auth_vus_by_method[method].append(
            build_author_vus_coverage(df_m) if not df_m.empty else None
        )
        both_det_excalibr, both_det_author = (
            build_both_determinate_confusion_matrices(df_m, use_oob=use_oob, label=f"{dataset}/{method}")
            if not df_m.empty else (None, None)
        )
        both_det_excalibr_by_method[method].append(both_det_excalibr)
        both_det_author_by_method[method].append(both_det_author)

# Calibration-vs-ClinVar confusion matrix, per method — always plotted.
for m in methods:
    make_single_confusion_figure(
        conf_by_method[m], datasets, label=m, figure_dir=FIGURE_SUBDIRS["author"],
        vus_coverages=vus_by_method[m], filename=f"excalibr_vs_clinvar_{m}.png",
    )

# ExCALIBR vs. author, for whichever method is primary (first discovered).
# auth_by_method entries are already None for any dataset with zero
# determinate author calls (see build_author_confusion_matrix) -- such a
# dataset means the author functional classification was never recorded,
# not that the author genuinely called every control indeterminate, so it's
# excluded from this aggregate rather than padding it with synthetic IR.
primary_method = methods[0]
auths = auth_by_method[primary_method]
if any(a is not None for a in auths):
    make_confusion_figure(
        danzs_m1=conf_by_method[primary_method], danzs_m2=auths,
        dataset_names=datasets, label1=primary_method, label2="author",
        figure_dir=FIGURE_SUBDIRS["author"], filename="excalibr_vs_author.png",
        vus_coverages_m1=vus_by_method[primary_method],
        vus_coverages_m2=auth_vus_by_method[primary_method],
    )

# %% [markdown]
# ### 3a1b. ExCALIBR vs. author, determinate-determinate calls only
#
# Same two [BLB,PLP] x [Normal,IR,Abnormal] confusion matrices as the panel
# above (ExCALIBR vs ClinVar, author vs ClinVar), just restricted to the
# subset of P/LP+B/LB variants where BOTH ExCALIBR (points != 0) and the
# author (Normal/Abnormal, not an indeterminate code or missing) made a
# determinate call -- see `build_both_determinate_confusion_matrices`. The
# IR column is necessarily all-zero on both sides by construction; what
# this isolates is whether ExCALIBR's and the author's *accuracy* against
# ClinVar (not their coverage) differ once both have actually committed to
# a call.

# %%
both_det_excalibr = both_det_excalibr_by_method[primary_method]
both_det_author = both_det_author_by_method[primary_method]
if any(m is not None for m in both_det_excalibr) and any(m is not None for m in both_det_author):
    make_confusion_figure(
        danzs_m1=both_det_excalibr, danzs_m2=both_det_author,
        dataset_names=datasets, label1=primary_method, label2="author",
        figure_dir=FIGURE_SUBDIRS["author"], filename="excalibr_vs_author_both_determinate.png",
    )
else:
    print("  SKIP ExCALIBR-vs-author both-determinate figure: no dataset had any determinate-determinate call")

# %% [markdown]
# ### 3a1c. Gene-deduplicated confusion matrices
#
# Section 3's per-dataset sum above counts a variant once per assay it
# appears in -- if two different MAVE studies of the same gene both scored
# the same genomic variant, it contributes twice to that aggregate. This
# panel instead merges every assay's copy of the same physical variant
# (matched by Gene/Chrom/hgvs_c, not the assay-specific MaveDB key -- see
# `analysis.multi_scoreset.genomic_variant_key`) into one row per gene
# before building the confusion matrix: the merged evidence points are the
# abs-max across assays if they agree in sign, else 0 (0,5 -> 5; -1,-3 ->
# -3; 2,-1 -> 0); merged author calls follow the same rule (conflicting
# Normal/Abnormal calls across assays -> indeterminate).
#
# The ExCALIBR-vs-author panel additionally drops any gene whose author
# calls are entirely indeterminate/missing across every one of its
# deduped variants (`restrict_to_genes_with_author_data`) -- same
# rationale as build_author_confusion_matrix's own all-indeterminate
# guard in section 3: a gene where the author functional classification
# was simply never recorded shouldn't count as "author called everything
# indeterminate". The standalone ExCALIBR-vs-ClinVar panel above it keeps
# every gene, since that panel doesn't involve author calls at all.

# %%
from analysis.multi_scoreset import (
    build_gene_deduped_variants, build_deduped_confusion_matrix, build_deduped_author_confusion_matrix,
    restrict_to_genes_with_author_data,
)

df_primary_dedup = df[df["method"] == primary_method]
deduped_variants = build_gene_deduped_variants(df_primary_dedup, use_oob=use_oob)
n_genes_deduped = deduped_variants["gene"].nunique()
n_multi_assay = int((deduped_variants["n_assays"] > 1).sum())
print(f"Gene-deduplicated: {len(deduped_variants):,} unique variants across {n_genes_deduped} genes "
      f"({n_multi_assay:,} scored by more than one assay)")

deduped_matrix = build_deduped_confusion_matrix(deduped_variants)

deduped_variants_with_author = restrict_to_genes_with_author_data(deduped_variants)
n_genes_with_author = deduped_variants_with_author["gene"].nunique()
print(f"Gene-deduplicated, restricted to genes with real author data: {n_genes_with_author}/{n_genes_deduped} genes "
      f"({len(deduped_variants_with_author):,} variants)")
deduped_matrix_for_author = build_deduped_confusion_matrix(deduped_variants_with_author)
deduped_author_matrix = build_deduped_author_confusion_matrix(deduped_variants_with_author)

if deduped_matrix is not None:
    make_single_confusion_figure(
        [deduped_matrix], ["gene_deduped"], label=primary_method, figure_dir=FIGURE_SUBDIRS["author"],
        filename="excalibr_vs_clinvar_gene_deduped.png",
        title_suffix=f"({n_genes_deduped} genes, {len(deduped_variants):,} unique variants, gene-deduplicated)",
    )
else:
    print("  SKIP gene-deduplicated confusion figure: no P/LP or B/LB variants in deduped scope")

if deduped_matrix_for_author is not None and deduped_author_matrix is not None:
    make_confusion_figure(
        danzs_m1=[deduped_matrix_for_author], danzs_m2=[deduped_author_matrix], dataset_names=["gene_deduped"],
        label1=primary_method, label2="author", figure_dir=FIGURE_SUBDIRS["author"],
        filename="excalibr_vs_author_gene_deduped.png",
    )
else:
    print("  SKIP gene-deduplicated ExCALIBR-vs-author figure: no matrix on one or both sides")

# %% [markdown]
# ### 3a2. ExCALIBR vs. other comparison methods
#
# Each method below produces a per-dataset confusion matrix the same shape
# as ExCALIBR's own (`build_confusion_matrix`'s [BLB,PLP] x [Normal,IR,
# Abnormal]), then `_compare_vs_excalibr` (defined once, reused for every
# method) produces two confusion figures for it (no per-gene scatter here --
# that lives only in the combined scatter/OR/Brnich figure, section 7):
#   all_datasets     : every dataset ExCALIBR has a matrix for is kept in the
#                       aggregate -- any dataset the other method couldn't be
#                       calibrated for contributes its full BLB/PLP row
#                       totals to the IR (indeterminate) column instead of
#                       being dropped, so the two methods' aggregate
#                       denominators match exactly.
#   matched_datasets  : restricted to only the datasets where the other
#                       method actually produced a result -- a stricter
#                       apples-to-apples comparison, but over fewer datasets.
#
# Methods (see analysis/comparison_methods.py):
#   acmgscaler        : github.com/badonyi/acmgscaler (Badonyi & Marsh 2025),
#                       prior=0.1, loaded from precomputed CSVs under
#                       analysis.config.ACMGSCALER_OUTPUT_DIR -- run
#                       run_acmgscaler_all.py beforehand to generate those
#                       (it runs Rscript per dataset in parallel via joblib).
#                       This notebook never invokes Rscript itself; a dataset
#                       missing its CSV is treated as "could not be
#                       calibrated" (e.g. <10 P or <10 B labeled variants).
#   gmm_plp_blb / gmm_plp_blb_synon :
#                       simple 2-component GMM baseline, prior=0.1, two
#                       control-pooling variants (P/LP+B/LB only, vs.
#                       P/LP+[B/LB union Synonymous]) -- a naive baseline
#                       with no monotonicity constraint on its point ranges
#                       (unlike ExCALIBR's own), loaded from
#                       analysis.config.GMM_BASELINE_OUTPUT_DIR. Its
#                       variants.csv already has ExCALIBR's own sample/
#                       standard_points columns, so build_confusion_matrix
#                       works on it completely unchanged -- no live
#                       computation, no separate builder function needed.

# %%
from analysis.comparison_methods import (
    build_acmgscaler_confusion_matrix, load_acmgscaler_variants,
    load_comparison_variants,
)

def _all_indeterminate_matrix(excalibr_matrix):
    """Same BLB/PLP row totals as excalibr_matrix, but every count placed in
    the IR column -- represents "this method could not be calibrated for
    this dataset" without dropping it from the aggregate denominator. None
    if excalibr_matrix itself is None (nothing to match totals against)."""
    if excalibr_matrix is None:
        return None
    totals = excalibr_matrix.sum(axis=1)
    return pd.DataFrame(
        {"Normal": 0, "IR": totals, "Abnormal": 0}, index=excalibr_matrix.index,
    )[["Normal", "IR", "Abnormal"]]

comparison_matches = {}  # method_label -> (matched_excalibr, matched_other, matched_datasets), filled below

# Which clinvar_comparisons/ subfolder each comparison method's figures land
# in -- keyed by the exact method_label string each call site below passes.
_COMPARISON_SUBDIR_KEY = {
    "acmgscaler": "acmgscaler",
    "skew_locked": "skew_locked",
    "gmm_plp_blb": "gmm_baseline",
    "gmm_plp_blb_synon": "gmm_baseline",
    "gmm_all_plp_blb": "gmm_baseline",
    "gmm_all_plp_blb_synon": "gmm_baseline",
}

def _compare_vs_excalibr(conf_raw, method_label):
    """conf_raw : per-dataset confusion matrices (or None) for `method_label`,
    same order as `datasets`. Produces both the all_datasets and
    matched_datasets confusion figures against conf_by_method[primary_method],
    routed into the clinvar_comparisons/{acmgscaler,gmm_baseline,skew_locked}/
    subfolder matching `method_label` (see _COMPARISON_SUBDIR_KEY), and
    records the matched-datasets triple in `comparison_matches` for the
    aggregate performance report in 3b."""
    figure_dir = FIGURE_SUBDIRS[_COMPARISON_SUBDIR_KEY[method_label]]
    n_missing = sum(
        1 for m, d in zip(conf_raw, conf_by_method[primary_method])
        if m is None and d is not None
    )
    print(f"  {method_label}: {len(datasets) - n_missing}/{len(datasets)} datasets calibrated "
          f"({n_missing} treated as all-indeterminate in the all_datasets version)")

    conf_all = [
        m if m is not None else _all_indeterminate_matrix(d)
        for m, d in zip(conf_raw, conf_by_method[primary_method])
    ]
    if any(m is not None for m in conf_all):
        make_confusion_figure(
            danzs_m1=conf_by_method[primary_method], danzs_m2=conf_all,
            dataset_names=datasets, label1=primary_method, label2=method_label,
            figure_dir=figure_dir, filename=f"excalibr_vs_{method_label}_all_datasets.png",
        )
    else:
        print(f"  SKIP {method_label} comparison (all_datasets): no ExCALIBR matrices to pair with")

    matched_idx = [i for i, m in enumerate(conf_raw) if m is not None]
    if matched_idx:
        matched_datasets = [datasets[i] for i in matched_idx]
        matched_excalibr = [conf_by_method[primary_method][i] for i in matched_idx]
        matched_other = [conf_raw[i] for i in matched_idx]
        make_confusion_figure(
            danzs_m1=matched_excalibr, danzs_m2=matched_other,
            dataset_names=matched_datasets, label1=primary_method, label2=method_label,
            figure_dir=figure_dir, filename=f"excalibr_vs_{method_label}_matched_datasets.png",
        )
        comparison_matches[method_label] = (matched_excalibr, matched_other, matched_datasets)
    else:
        print(f"  SKIP {method_label} comparison (matched_datasets): no dataset produced a matrix")

# --- manual-prior ExCALIBR rerun (prior=0.1 fixed, not auto-fit) ---
# Loaded here (rather than down in 3a3 alongside skew-locked) so it's
# available for the 3-way ExCALIBR/acmgscaler/manual-prior grid right below.
# analysis.config.MANUAL_PRIOR_OUTPUT_DIR is currently a TEMP PLACEHOLDER
# PATH -- this rerun doesn't exist on disk yet, so this cell just prints a
# warning and leaves manual_prior_conf_raw as None until it's populated.
# Same full ExCALIBR-shaped output tree as OUTPUT_DIR/SKEW_LOCKED_OUTPUT_DIR,
# so discovered/loaded via analysis.discovery exactly like a normal
# pipeline run, not analysis.comparison_methods. use_oob=False to match the
# skew-locked/GMM-baseline convention below (flip to True once/if this rerun
# carries oob_* columns).
manual_prior_conf_raw = None
if not config.warn_if_missing(config.MANUAL_PRIOR_OUTPUT_DIR, "ExCALIBR manual-prior (0.1) comparison"):
    mp_tree, mp_model_selections, mp_calibrations = discover_outputs(Path(config.MANUAL_PRIOR_OUTPUT_DIR))
    mp_df = load_all_variants(
        tree=mp_tree, model_selections=mp_model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=datasets, calibrations=mp_calibrations, min_controls=0,
    )
    manual_prior_conf_raw = []
    for dataset in datasets:
        df_mp = mp_df[mp_df["dataset"] == dataset] if not mp_df.empty else mp_df
        manual_prior_conf_raw.append(
            build_confusion_matrix(df_mp, use_oob=False, label=f"{dataset}/manual_prior_0.1")
            if not df_mp.empty else None
        )

# --- acmgscaler ---
# Precomputed by run_acmgscaler_all.py (analysis.config.ACMGSCALER_OUTPUT_DIR)
# for every dataset it could calibrate -- this is a disk read only, no live
# Rscript call. A missing CSV means acmgscaler genuinely couldn't calibrate
# that dataset (e.g. <10 P or <10 B controls), not "not computed yet"; run
# run_acmgscaler_all.py separately (it parallelizes across datasets via
# joblib) if ACMGSCALER_OUTPUT_DIR is stale or unset.
if not config.warn_if_missing(config.ACMGSCALER_OUTPUT_DIR, "acmgscaler comparison"):
    acmgscaler_conf_raw = [
        build_acmgscaler_confusion_matrix(df_acmg, label=dataset)
        if (df_acmg := load_acmgscaler_variants(dataset, config.ACMGSCALER_OUTPUT_DIR)) is not None
        else None
        for dataset in datasets
    ]
    _compare_vs_excalibr(acmgscaler_conf_raw, "acmgscaler")

    # 3-way grid: ExCALIBR (normal auto-fit prior) vs acmgscaler vs ExCALIBR
    # rerun with prior manually fixed at 0.1 -- lets you see at a glance
    # whether acmgscaler's disagreement with ExCALIBR tracks the prior
    # choice itself rather than the calibration method.
    #
    # Restricted to the intersection of datasets where all three actually
    # produced a matrix -- acmgscaler skips datasets it can't calibrate
    # (<10 P or <10 B controls) and the manual-prior rerun is its own,
    # separately-run pipeline output tree that may simply not cover every
    # dataset the main run does (e.g. one added after the manual-prior rerun
    # was last generated) -- pooling each panel over its own, independently
    # -sized set of non-None datasets would silently compare unequal
    # populations panel to panel.
    if manual_prior_conf_raw is not None and any(m is not None for m in manual_prior_conf_raw):
        _grid_excalibr = conf_by_method[primary_method]
        _common_idx = [
            i for i in range(len(datasets))
            if _grid_excalibr[i] is not None and acmgscaler_conf_raw[i] is not None
            and manual_prior_conf_raw[i] is not None
        ]
        _missing_datasets = {
            "excalibr": [datasets[i] for i in range(len(datasets)) if _grid_excalibr[i] is None],
            "acmgscaler": [datasets[i] for i in range(len(datasets)) if acmgscaler_conf_raw[i] is None],
            "manual_prior_0.1": [datasets[i] for i in range(len(datasets)) if manual_prior_conf_raw[i] is None],
        }
        for _panel, _miss in _missing_datasets.items():
            if _miss:
                print(f"  3-way grid: {_panel} missing {len(_miss)} dataset(s) not in the common scope: {_miss}")
        if _common_idx:
            make_confusion_grid_figure(
                panels=[
                    (primary_method, [_grid_excalibr[i] for i in _common_idx]),
                    ("acmgscaler", [acmgscaler_conf_raw[i] for i in _common_idx]),
                    ("manual prior=0.1", [manual_prior_conf_raw[i] for i in _common_idx]),
                ],
                figure_dir=FIGURE_SUBDIRS["acmgscaler"],
                filename="excalibr_vs_acmgscaler_vs_manual_prior_grid.png",
                suptitle=f"ExCALIBR vs. acmgscaler vs. ExCALIBR (manual prior=0.1) "
                         f"({len(_common_idx)} common datasets)",
            )
        else:
            print("  SKIP 3-way ExCALIBR/acmgscaler/manual-prior grid: no dataset had a matrix in all three")
    else:
        print("  SKIP 3-way ExCALIBR/acmgscaler/manual-prior grid: manual-prior output not available")

# --- simple GMM baseline, both pooling variants ---
if config.GMM_BASELINE_OUTPUT_DIR:
    for variant in config.GMM_BASELINE_VARIANTS:
        gmm_conf_raw = []
        for dataset in datasets:
            df_gmm = load_comparison_variants(dataset, variant, config.GMM_BASELINE_OUTPUT_DIR)
            gmm_conf_raw.append(
                build_confusion_matrix(df_gmm, use_oob=False, label=f"{dataset}/gmm_{variant}")
                if df_gmm is not None else None
            )
        _compare_vs_excalibr(gmm_conf_raw, f"gmm_{variant}")
else:
    print("  SKIP GMM baseline comparison: analysis.config.GMM_BASELINE_OUTPUT_DIR not set")

# %% [markdown]
# ### 3a3. Skew-locked ExCALIBR comparison
#
# "Skew-locked" = the canonical pipeline rerun with each component's skew
# parameter fixed (not freely fit), instead of ExCALIBR's normal freely-fit
# skew-normal components. Unlike acmgscaler/gmm above, this is a full
# ExCALIBR-shaped output tree (`analysis.config.SKEW_LOCKED_OUTPUT_DIR`) --
# discovered/loaded with `analysis.discovery` exactly like `OUTPUT_DIR`
# itself, not `analysis.comparison_methods`. Its `*_variants.csv` files
# already carry `auth_label` (no `attach_author_labels` needed) but no
# `oob_*` columns, so matrices are built with `use_oob=False`, same as the
# GMM baseline. Compared against the primary method with the same
# `_compare_vs_excalibr` helper used above, so its matched-datasets
# aggregate performance is picked up automatically by 3b, and its evidence
# distribution is added alongside section 4's.

# %%
if not config.warn_if_missing(config.SKEW_LOCKED_OUTPUT_DIR, "skew-locked ExCALIBR comparison"):
    skew_tree, skew_model_selections, skew_calibrations = discover_outputs(Path(config.SKEW_LOCKED_OUTPUT_DIR))
    skew_df = load_all_variants(
        tree=skew_tree, model_selections=skew_model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=datasets, calibrations=skew_calibrations, min_controls=0,
    )
    skew_locked_conf_raw = []
    for dataset in datasets:
        df_sk = skew_df[skew_df["dataset"] == dataset] if not skew_df.empty else skew_df
        skew_locked_conf_raw.append(
            build_confusion_matrix(df_sk, use_oob=False, label=f"{dataset}/skew_locked")
            if not df_sk.empty else None
        )
    _compare_vs_excalibr(skew_locked_conf_raw, "skew_locked")
else:
    skew_tree, skew_model_selections, skew_calibrations = {}, {}, {}
    skew_df = pd.DataFrame()

# %% [markdown]
# ### 3a4. Skew-locked vs. regular ExCALIBR: per-bootstrap validation likelihood
#
# For each dataset's selected component (same `(n_c, benign_method)` chosen
# by `resolve_component` for the regular run -- `tree`/`model_selections`
# from section 1), pull every bootstrap's validation log-likelihood
# (`val_ll`) for that component count from the full per-bootstrap fit files
# (`analysis.config.PRECOMPUTED_FITS` / `SKEW_LOCKED_BOOTSTRAP_RESULTS`, both
# `{dataset: {bootstrap_seed: {n_c: {"val_ll": float, ...}}}}`), pair up by
# bootstrap seed (both runs share the same bootstrap resampling), and report
# the median + IQR of the per-seed difference `skew_locked_val_ll -
# regular_val_ll` for that dataset. A negative median means the skew-locked
# fit's held-out likelihood is typically worse than the freely-fit skew
# model's, i.e. locking skew costs fit quality on that dataset.

# %%
import gzip as _gzip

def _load_bootstrap_json(path):
    with _gzip.open(path, "rt") as f:
        return json.load(f)

if not skew_df.empty and not config.warn_if_missing(config.SKEW_LOCKED_BOOTSTRAP_RESULTS, "skew-locked bootstrap val_ll") \
        and not config.warn_if_missing(config.PRECOMPUTED_FITS, "regular bootstrap val_ll"):
    _regular_boot = _load_bootstrap_json(config.PRECOMPUTED_FITS)
    _skew_boot = _load_bootstrap_json(config.SKEW_LOCKED_BOOTSTRAP_RESULTS)

    _ll_rows = []
    for dataset in datasets:
        if dataset not in _regular_boot or dataset not in _skew_boot:
            continue
        available_comps = list(tree.get(dataset, {}).keys())
        if not available_comps:
            continue
        comp = resolve_component(dataset, available_comps, model_selections, dataset_configs)
        n_c = comp.split("_", 1)[0]

        reg_by_seed = _regular_boot[dataset]
        skew_by_seed = _skew_boot[dataset]
        common_seeds = sorted(
            set(reg_by_seed) & set(skew_by_seed),
            key=lambda s: int(s),
        )
        diffs = [
            skew_by_seed[s][n_c]["val_ll"] - reg_by_seed[s][n_c]["val_ll"]
            for s in common_seeds
            # .get(n_c) rather than "in" -- a seed can have an n_c key present
            # but mapped to None (that bootstrap's fit for this component
            # count failed to converge), which "in" alone doesn't catch.
            if reg_by_seed[s].get(n_c) is not None and skew_by_seed[s].get(n_c) is not None
        ]
        if not diffs:
            continue
        diffs = np.array(diffs)
        q25, q50, q75 = np.percentile(diffs, [25, 50, 75])
        _ll_rows.append({
            "dataset": dataset, "n_c": n_c, "n_bootstraps": len(diffs),
            "median_diff": q50, "iqr_lo": q25, "iqr_hi": q75, "iqr_width": q75 - q25,
        })

    del _regular_boot, _skew_boot  # both are full 89-dataset JSONs, free memory once done

    ll_diff_df = pd.DataFrame(_ll_rows)
    if not ll_diff_df.empty:
        ll_diff_df = ll_diff_df.sort_values("median_diff").reset_index(drop=True)
        print(f"\n{'=' * 80}\nSKEW-LOCKED vs REGULAR: per-bootstrap val_ll difference "
              f"(skew_locked - regular), by selected component\n{'=' * 80}")
        print(ll_diff_df.to_string(index=False))
    else:
        print("  SKIP skew-locked vs regular val_ll comparison: no dataset had matching bootstrap data")
else:
    ll_diff_df = pd.DataFrame()

ll_diff_df

# %% [markdown]
# ### 3a4b. Pathomechanism-aware prior/LR+ comparison
#
# "canonical" = the standard ExCALIBR fit (OUTPUT_DIR). "pathomech_boundary"
# = the same pipeline rerun with the pathomechanism-aware prior/likelihood
# ratio (Supplementary Section sec:pathomechanism_prior) enabled via
# --pathomechanism-prior --pathomechanism-method boundary, read from
# config.PATHOMECHANISM_OUTPUT_DIR. Logic ported from (not duplicated by
# reference to) test/plot_canonical_vs_pathomech_boundary_confusion.py, which
# remains runnable standalone. config.PATHOMECHANISM_OUTPUT_DIR is currently
# a TEMP PLACEHOLDER PATH (run still in progress as of 2026-08-12) -- this
# cell prints a warning and skips until it's populated.

# %%
if not config.warn_if_missing(config.PATHOMECHANISM_OUTPUT_DIR, "pathomechanism comparison"):
    pm_tree, pm_model_selections, pm_calibrations = discover_outputs(Path(config.PATHOMECHANISM_OUTPUT_DIR))
    pm_df_all = load_all_variants(
        tree=pm_tree, model_selections=pm_model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=None, calibrations=pm_calibrations, min_controls=0,
    )
    pm_method = sorted(pm_df_all["method"].unique())[0] if not pm_df_all.empty else None
    pm_df = pm_df_all[pm_df_all["method"] == pm_method] if pm_method else pm_df_all

    pathomech_datasets = sorted(set(datasets) | set(pm_df["dataset"].unique()))
    canonical_conf, pathomech_conf = [], []
    for ds in pathomech_datasets:
        df_c = df[(df["dataset"] == ds) & (df["method"] == primary_method)]
        df_p = pm_df[pm_df["dataset"] == ds]
        canonical_conf.append(build_confusion_matrix(df_c, use_oob=True, label=f"{ds}/canonical") if not df_c.empty else None)
        pathomech_conf.append(build_confusion_matrix(df_p, use_oob=True, label=f"{ds}/pathomech_boundary") if not df_p.empty else None)

    make_single_confusion_figure(
        canonical_conf, pathomech_datasets, label="canonical",
        figure_dir=FIGURE_SUBDIRS["pathomechanism"], filename="excalibr_canonical.png",
    )
    make_single_confusion_figure(
        pathomech_conf, pathomech_datasets, label="pathomech_boundary",
        figure_dir=FIGURE_SUBDIRS["pathomechanism"], filename="excalibr_pathomech_boundary.png",
    )
    make_confusion_figure(
        danzs_m1=canonical_conf, danzs_m2=pathomech_conf, dataset_names=pathomech_datasets,
        label1="canonical", label2="pathomech_boundary", figure_dir=FIGURE_SUBDIRS["pathomechanism"],
        filename="canonical_vs_pathomech_boundary_all_datasets.png",
    )

    pathomech_diff_rows = []
    for ds, c, p in zip(pathomech_datasets, canonical_conf, pathomech_conf):
        if c is None or p is None or c.equals(p):
            continue
        pathomech_diff_rows.append({
            "dataset": ds,
            "canonical_PLP_Normal": int(c.loc["PLP", "Normal"]), "pathomech_PLP_Normal": int(p.loc["PLP", "Normal"]),
            "canonical_PLP_Abnormal": int(c.loc["PLP", "Abnormal"]), "pathomech_PLP_Abnormal": int(p.loc["PLP", "Abnormal"]),
            "canonical_BLB_Abnormal": int(c.loc["BLB", "Abnormal"]), "pathomech_BLB_Abnormal": int(p.loc["BLB", "Abnormal"]),
        })
    pathomech_diff_df = pd.DataFrame(pathomech_diff_rows)
    print(f"Datasets with a CHANGED confusion matrix under the pathomechanism prior: "
          f"{len(pathomech_diff_df)}/{len(pathomech_datasets)}")
    if not pathomech_diff_df.empty:
        pathomech_diff_df.to_csv(FIGURE_SUBDIRS["pathomechanism"] / "canonical_vs_pathomech_diff.csv", index=False)

# %% [markdown]
# ### 3a5. Robustness analysis (downsampling / label discordance)
#
# How sensitive is ExCALIBR's calibration to shrinking (downsampling) or
# discordant (mislabeled) P/LP and B/LB control counts? Perturbed conditions
# (see `test/downsample_discordance_test.ipynb`, not reproduced here) were
# generated at 7 downsample levels (control count N in [1,2,4,8,16,32,64])
# and 2 discordance levels (fraction relabeled in [0.01, 0.10]), each with
# 10 random seeds, then run through the normal pipeline.
#
# Each perturbed condition's own confusion matrix / variant scores are NOT
# used directly (too few variants, and downsampled `variant_id`s don't
# correspond by position to the reference dataset — see
# `analysis/robustness.py` module docstring). Instead, for every
# (perturbation_type, level), `plot_robustness_config_summary` aggregates
# across that level's 10 seeds and draws the same 3-row layout
# `plot_scoreset_best_config` uses:
#  - **fits row**: the reference (fixed, unperturbed) population's score
#    histograms, overlaid with mixture-density curves computed from every
#    seed's per-bootstrap fit at this level flattened into one bootstrap x
#    seed pool — shaded by that pool's 5th/50th/95th percentile density.
#  - **point-assignment row**: all 10 seeds' point-range bars overlaid at low
#    alpha, so more-opaque regions are where more seeds agree on a call.
#  - **Log LR+ row**: each seed already has its own correct [5th,50th,95th]
#    percentile curve (across that seed's own bootstraps); the bands shown
#    are the seed-to-seed IQR spread of the 5th-percentile curve and,
#    separately, of the 95th-percentile curve, with the reference curve
#    overlaid in bold black for comparison.
#
# Base datasets are discovered dynamically from whatever's on disk under
# `analysis.config.ROBUSTNESS_OUTPUT_DIR` — not a hardcoded list.
#
# Confusion-matrix-based summary metrics (accuracy/coverage/DOR vs. level)
# are still computed below and kept available for later use, but not plotted
# here — right now, understanding *how* the calibration shifts under
# perturbation (the plots above) matters more than a scalar accuracy trend.

# %%
from analysis.robustness import (
    discover_robustness_base_datasets, load_reference_variants,
    compute_robustness_confusion_matrices, compute_robustness_max_strengths,
    robustness_confusion_matrices_to_metrics, run_config_summary_plots_batch,
    plot_robustness_confusion_matrix_grid,
)

if not config.warn_if_missing(config.ROBUSTNESS_OUTPUT_DIR, "robustness analysis"):
    robustness_bases = discover_robustness_base_datasets(config.ROBUSTNESS_OUTPUT_DIR)
    print(f"Discovered {len(robustness_bases)} base dataset(s) with robustness conditions: {robustness_bases}")

    # Reference (unperturbed) population for every base dataset comes from
    # each base dataset's own "_control" condition under ROBUSTNESS_OUTPUT_DIR
    # -- fit at the SAME xl (1000 bootstraps x 8 fits) budget as every
    # perturbed condition, not the main pipeline's own (typically higher,
    # e.g. "finest"/100 fits) budget -- see analysis.robustness's module
    # docstring (REFERENCE SOURCE) for why a higher-budget baseline would
    # bias every comparison. tree/model_selections/calibrations (section 1's
    # own main-pipeline discovery) are no longer needed here as a result.
    robustness_summaries = {}
    robustness_matrices = {}
    robustness_max_strengths = {}
    config_summary_jobs = []
    for base_ds in robustness_bases:
        try:
            reference_df, ref_cal_path = load_reference_variants(
                base_ds, reference_source="robustness_control",
                robustness_output_dir=config.ROBUSTNESS_OUTPUT_DIR,
            )
        except FileNotFoundError as e:
            print(f"  SKIP {base_ds}: {e}")
            continue

        matrices = compute_robustness_confusion_matrices(
            base_ds, reference_df, ref_cal_path, config.ROBUSTNESS_OUTPUT_DIR,
        )
        if matrices:
            robustness_summaries[base_ds] = robustness_confusion_matrices_to_metrics(matrices, base_ds)
            robustness_matrices[base_ds] = matrices
            robustness_max_strengths[base_ds] = compute_robustness_max_strengths(
                base_ds, reference_df, ref_cal_path, config.ROBUSTNESS_OUTPUT_DIR,
            )

        config_summary_jobs.append((base_ds, reference_df, ref_cal_path))

    # All base datasets' (perturbation_type, level) config-summary figures
    # rendered in one parallel batch -- each is an independent, self-
    # contained unit of work (its own pooled_fits_for_level + sample_density
    # call), the actual cost driver for this section.
    run_config_summary_plots_batch(
        config_summary_jobs, figure_dir=FIGURE_DIR, robustness_output_dir=config.ROBUSTNESS_OUTPUT_DIR,
    )

    # Confusion-matrix grids: 4 columns (one per robustness base dataset) x
    # one row per condition level (control first, then descending downsample
    # N / ascending discordance fraction) -- the same diverging Blue/Gray/Red
    # style used everywhere else, plus each cell's strongest pathogenic/
    # benign evidence point ("Max strengths: +X, -Y").
    if robustness_matrices:
        plot_robustness_confusion_matrix_grid(
            robustness_matrices, robustness_max_strengths, "downsample",
            base_datasets=list(robustness_matrices.keys()), figure_dir=FIGURE_DIR,
        )
        plot_robustness_confusion_matrix_grid(
            robustness_matrices, robustness_max_strengths, "discordance",
            base_datasets=list(robustness_matrices.keys()), figure_dir=FIGURE_DIR,
        )

# %% [markdown]
# ### 3a6. Bootstrap-count-reduction analysis
#
# How much does reducing the number of bootstrap fits (e.g. 1000 -> 20) used
# to build a calibration degrade it? Unlike the downsample/discordance
# robustness conditions above, `tests/benchmark_bootstrap_reduction.py`
# computes exactly ONE calibration per (dataset, bootstrap-count level) --
# no repeated seeds per level. Deliberately not adding those (rerunning each
# level several times with independent bootstrap subsamples just to measure
# seed-to-seed spread) was a cost call, not an oversight: each level already
# carries its own free bootstrap-resampling uncertainty --
# `process_component_fits` computes `[p5,p50,p95]` across whatever bootstrap
# fits that level actually got, so comparing every level's own band directly
# already shows (a) how the median LR+ curve drifts as N shrinks and (b) how
# each level's own reported uncertainty widens as N shrinks, with zero extra
# fitting. See `analysis.robustness.plot_bootstrap_reduction_config_summary`'s
# docstring for the full reasoning.
#
# Data assembly (matching `tests/benchmark_bootstrap_reduction.py`'s on-disk
# `{dataset}/level_{N}/..._calibration.json` layout, and the fixed reference
# population every level's histogram/LR+ curve is compared against) is
# reused directly from `tests/plot_bootstrap_reduction_config.py` — the
# standalone script and this notebook cell call the exact same
# `_process_one_dataset`/`build_reference_df` helpers, so there is only one
# place that logic lives.

# %%
if not config.warn_if_missing(config.BOOTSTRAP_REDUCTION_OUTPUT_DIR, "bootstrap-count-reduction analysis"):
    import gzip as _gzip
    import multiprocessing as _mp
    from joblib import Parallel as _Parallel, delayed as _delayed
    from run_igvf_batch import parse_dataset_config as _parse_dataset_config
    from tests.benchmark_bootstrap_reduction import build_dataset_df as _build_bsr_dataset_df
    from tests.plot_bootstrap_reduction_config import _process_one_dataset as _process_bsr_dataset

    br_dir = Path(config.BOOTSTRAP_REDUCTION_OUTPUT_DIR)
    bsr_datasets = sorted(
        d.name for d in br_dir.iterdir() if d.is_dir() and any(d.glob("level_*"))
    )
    print(f"Discovered {len(bsr_datasets)} dataset(s) with bootstrap-reduction output")

    with open(config.DATASET_CONFIGS) as f:
        bsr_dataset_configs = json.load(f)
    _sep = "\t" if config.DATASET_TSV.endswith((".tsv", ".tsv.gz")) else ","
    bsr_df = pd.read_csv(config.DATASET_TSV, sep=_sep)

    bsr_bootstrap_results = None
    if config.PRECOMPUTED_FITS and Path(config.PRECOMPUTED_FITS).exists():
        print(f"  Loading {config.PRECOMPUTED_FITS} for Row 0 density overlay...")
        with _gzip.open(config.PRECOMPUTED_FITS, "rt", encoding="utf-8") as f:
            bsr_bootstrap_results = json.load(f)

    bsr_per_dataset_df = {d: _build_bsr_dataset_df(d, bsr_df) for d in bsr_datasets}
    print(f"  Rendering {len(bsr_datasets)} bootstrap-reduction config-summary figures "
          f"across {_mp.cpu_count()} CPUs...")
    bsr_results = _Parallel(n_jobs=-1, batch_size=1, backend="loky", verbose=5)(
        _delayed(_process_bsr_dataset)(
            dataset_name, br_dir / dataset_name, bsr_dataset_configs, config.DATASET_TSV,
            bsr_per_dataset_df[dataset_name], bsr_bootstrap_results, FIGURE_DIR,
        )
        for dataset_name in bsr_datasets
    )
    for dataset_name, err in bsr_results:
        if err is not None:
            print(f"  SKIP {dataset_name}: {err}")

# %% [markdown]
# ### 3a7. Fit-number (restart-count) comparison
#
# How much does reducing the number of EM restarts per fit degrade the
# best-of-N training log-likelihood? Median + IQR ribbon (across every
# dataset x n_c row) vs. restart count, from
# `tests/benchmark_num_fits_dataframe.py`'s `summary.csv`.

# %%
if not config.warn_if_missing(config.FIT_NUMBER_COMPARISON_SUMMARY_CSV, "fit-number comparison"):
    from analysis.robustness import plot_fit_number_comparison_curve

    fit_number_summary = pd.read_csv(config.FIT_NUMBER_COMPARISON_SUMMARY_CSV)
    print(f"Loaded {len(fit_number_summary)} rows, "
          f"{fit_number_summary['dataset'].nunique()} datasets, "
          f"num_fits levels: {sorted(fit_number_summary['num_fits'].unique())}")
    plot_fit_number_comparison_curve(fit_number_summary, figure_dir=FIGURE_DIR, label="all_datasets")

# %% [markdown]
# ### 3a8. SpliceAI threshold / VEP splice-consequence filter ablation
#
# `Scoreset.splicing_filter` (`src/assay_calibration/data_utils/dataset.py`)
# drops rows flagged as likely splicing aberrations -- by VEP consequence
# and/or a SpliceAI DS_{AG,AL,DG,DL} >= 0.2 threshold -- for any assay that
# doesn't itself detect splice effects. How sensitive is calibration
# performance to that threshold, and to disabling either filter entirely?
#
# Unlike the downsample/discordance robustness conditions in 3a5, each
# condition here (`analysis/build_splice_ablation_jobs.py`'s
# `thresh_0.1`..`thresh_0.9` + `keep_all` subdirectories under
# `analysis.config.SPLICE_ABLATION_ROOT`) is a full, independently-fit
# ExCALIBR output tree -- discovered/loaded exactly like a normal pipeline
# run (same pattern as 3a3's skew-locked comparison), no reference-
# population indirection needed.

# %%
from analysis.splice_ablation import run_splice_ablation_analysis, plot_splice_ablation_curve

if not config.warn_if_missing(config.SPLICE_ABLATION_ROOT, "splice ablation analysis"):
    splice_ablation_summary = run_splice_ablation_analysis(
        config.SPLICE_ABLATION_ROOT, dataset_configs=dataset_configs,
    )
    if not splice_ablation_summary.empty:
        print(f"Splice ablation: {splice_ablation_summary['condition_label'].nunique()} condition(s), "
              f"{splice_ablation_summary['dataset'].nunique()} dataset(s)")
        plot_splice_ablation_curve(splice_ablation_summary, figure_dir=FIGURE_DIR, label="all_datasets")

# %% [markdown]
# ### 3b. Aggregate performance report + manuscript LaTeX table
#
# `print_aggregate_performance` (src/assay_calibration/plot_utils/utils.py)
# sums the confusion matrices above and prints the full text report (per
# dataset + aggregate accuracy/sensitivity/specificity/MCC/LR+/DOR for both
# ExCALIBR and author annotations). `latex_performance_table_clinvar` turns
# those same computed metrics into the manuscript's LaTeX table — every
# number in it comes from the confusion matrices built above, nothing
# hardcoded.

# %%
from src.assay_calibration.plot_utils.utils import print_aggregate_performance
from analysis.manuscript_stats import latex_performance_table_clinvar

_auth_pairs = [
    (d, a, n) for d, a, n in zip(conf_by_method[primary_method], auths, datasets)
    if d is not None and a is not None
]
if _auth_pairs:
    _danzs_auth, _auths_auth, _names_auth = zip(*_auth_pairs)
    danz_agg_metrics, auth_agg_metrics, individual_metrics_df = print_aggregate_performance(
        list(_danzs_auth), list(_auths_auth), list(_names_auth),
    )
    _clinvar_perf_latex = latex_performance_table_clinvar(danz_agg_metrics, auth_agg_metrics)
    save_latex_table(_clinvar_perf_latex, FIGURE_SUBDIRS["tables"] / "author_clinvar_performance.tex")
else:
    print("  SKIP aggregate performance report: no datasets with both ExCALIBR and author matrices")

# Same aggregate report, restricted to the subset of datasets each comparison
# method (acmgscaler, the gmm baselines) could actually be calibrated on --
# `comparison_matches` was populated by `_compare_vs_excalibr` in 3a2, keyed
# by method_label -> (matched_excalibr, matched_other, matched_datasets).
for _method_label, (_matched_excalibr, _matched_other, _matched_names) in comparison_matches.items():
    print(f"\n{'-' * 80}\nExCALIBR vs {_method_label} (matched datasets, n={len(_matched_names)})\n{'-' * 80}")
    print_aggregate_performance(_matched_excalibr, _matched_other, _matched_names)

# %% [markdown]
# ### 3c. Per-dataset FP / FN breakdown (ranked)
#
# Raw false-positive (B/LB called Abnormal/Pathogenic) and false-negative
# (P/LP called Normal/Benign) counts per dataset, read directly off the same
# `conf_by_method[primary_method]` matrices built in section 3 (rows
# [BLB,PLP] x cols [Normal,IR,Abnormal] -- see `build_confusion_matrix`) --
# no recomputation. Ranked descending so the worst-offending datasets surface
# first; ties broken by the other column.

# %%
_fp_fn_rows = [
    {
        "dataset": _ds,
        "FP": int(_mat.loc["BLB", "Abnormal"]),
        "FN": int(_mat.loc["PLP", "Normal"]),
        "n_BLB": int(_mat.loc["BLB"].sum()),
        "n_PLP": int(_mat.loc["PLP"].sum()),
    }
    for _ds, _mat in zip(datasets, conf_by_method[primary_method])
    if _mat is not None
]
fp_fn_df = pd.DataFrame(_fp_fn_rows)

if not fp_fn_df.empty:
    fp_ranked = fp_fn_df.sort_values(["FP", "FN"], ascending=False).reset_index(drop=True)
    fn_ranked = fp_fn_df.sort_values(["FN", "FP"], ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 80}\nPER-DATASET FP/FN ({primary_method} vs ClinVar), ranked descending\n{'=' * 80}")
    print(f"\n-- Ranked by FP (B/LB called Pathogenic), n={len(fp_ranked)} datasets --")
    print(fp_ranked[["dataset", "FP", "FN", "n_BLB", "n_PLP"]].to_string(index=False))
    print(f"\n-- Ranked by FN (P/LP called Benign), n={len(fn_ranked)} datasets --")
    print(fn_ranked[["dataset", "FN", "FP", "n_BLB", "n_PLP"]].to_string(index=False))
else:
    fp_ranked = fn_ranked = fp_fn_df
    print("  SKIP per-dataset FP/FN breakdown: no ExCALIBR confusion matrices")

fp_ranked

# %% [markdown]
# ## 4. Evidence distributions and per-gene accuracy scatter

# %%
for method in methods:
    df_m = df[df["method"] == method]
    all_danz, all_clinvar = build_evidence_arrays(df_m)
    all_author = build_author_array(df_m)
    make_evidence_figure(all_danz, all_author, all_clinvar, label=method, figure_dir=FIGURE_SUBDIRS["manuscript"])

# Skew-locked evidence distribution, same shape as the loop above -- skew_df
# is built in 3a3 (empty DataFrame if that section skipped, e.g.
# SKEW_LOCKED_OUTPUT_DIR missing). Routed alongside the rest of the
# skew-locked ablation's figures rather than into manuscript_figures/.
if not skew_df.empty:
    skew_all_danz, skew_all_clinvar = build_evidence_arrays(skew_df)
    skew_all_author = build_author_array(skew_df)
    make_evidence_figure(
        skew_all_danz, skew_all_author, skew_all_clinvar, label="skew_locked",
        figure_dir=FIGURE_SUBDIRS["skew_locked"],
    )

# %% [markdown]
# ### 4b. Combined author + ClinVar evidence distribution, and gene-wise evidence table
#
# Matches `plot_combined_evidence_distributions(all_danz_oob, all_author,
# all_danz_oob_full, all_clinvar_full)` in the legacy script — the author
# panel and the ClinVar panel can legitimately come from *different* dataset
# scopes (the legacy script restricted the author panel to its curated
# reported-list while the ClinVar panel spanned every dataset). Both are set
# explicitly below — adjust either list if you want a narrower scope; there's
# no `_clinvar_2018`-duplicate dedup needed since the current data has one
# entry per dataset (no separate non-2018 sibling for BRCA1/MSH2/PTEN/TP53).

# %%
from analysis.evidence import build_dataset_info_and_arrays, make_combined_evidence_figure
from src.assay_calibration.plot_utils.utils import compute_genewise_evidence_table

# Author-annotated subset -- reused below (and again for CLINGEN_DATASETS)
# rather than recomputed. Restricting the author panel to this subset (while
# the ClinVar panel below spans every dataset) matches the legacy script's
# original two-scope design (see make_combined_evidence_figure's docstring);
# leaving both panels at the full `datasets` list here was an unfilled
# placeholder that made the two panels coincidentally identical.
datasets_with_author = [d for d, a in zip(datasets, auth_by_method[primary_method]) if a is not None]

AUTHOR_PANEL_DATASETS = datasets_with_author  # restrict to a curated subset here if desired
CLINVAR_PANEL_DATASETS = datasets             # restrict to a curated subset here if desired

df_primary = df[df["method"] == primary_method]

dataset_info_df, all_danz_oob, all_author, all_clinvar = build_dataset_info_and_arrays(
    df_primary, AUTHOR_PANEL_DATASETS, use_oob=True,
)
dataset_info_df_full, all_danz_oob_full, _, all_clinvar_full = build_dataset_info_and_arrays(
    df_primary, CLINVAR_PANEL_DATASETS, use_oob=True,
)

if len(all_danz_oob) and len(all_danz_oob_full):
    make_combined_evidence_figure(
        all_danz_oob, all_author, all_danz_oob_full, all_clinvar_full,
        label=primary_method, figure_dir=FIGURE_SUBDIRS["manuscript"],
    )

    gwe_author_table, gwe_clinvar_table, gwe_latex_str = compute_genewise_evidence_table(
        all_danz_oob, all_author, dataset_info_df,
        all_danz_oob_full, all_clinvar_full[:, :4], dataset_info_df_full,
    )
    print(gwe_latex_str)
    save_latex_table(gwe_latex_str, FIGURE_SUBDIRS["tables"] / "gene_wise_evidence_distribution.tex")
else:
    print("  SKIP combined evidence distribution / gene-wise table: no variants in scope")

# %% [markdown]
# ### 4c. ClinGen expert-panel ground-truth confusion
#
# Ground truth here is ClinGen's own applied ACMG evidence codes (already
# merged into the master dataframe as `*_ClinGen_repo` columns — no external
# file needed), with PS3/BS3 functional-evidence codes stripped before
# reclassifying to avoid circularity against ExCALIBR's own functional-assay
# evidence. Set `verbose_recode=True` in the call below to print every
# evidence-code recode for auditing a specific run.
#
# Reuses section 1's `df_primary`/`tree`/`model_selections`/`calibrations`
# directly (no separate discovery/loading pass) for every dataset except the
# `_clinvar_2018`-suffixed ones (BRCA1/MSH2/PTEN/TP53), which get their
# variant/points table reloaded fresh from the Scoreset -- see
# `build_clingen_confusion`'s docstring for why (their on-disk
# *_variants.csv can undercount VUS for exactly these datasets).
#
# Scope is restricted to `datasets_with_author` -- the same subset of
# datasets used for the ExCALIBR-vs-author ClinVar panels in section 3,
# where `build_author_confusion_matrix` found at least one determinate
# (Normal/Abnormal) author call. A dataset outside that subset never had its
# author functional classification recorded at all, so including it here
# would (same rationale as section 3) make it look like the author called
# every one of its ClinGen-labeled variants indeterminate, when really the
# author column was simply never populated for that dataset.
#
# Four variants of the panel are built below:
# - per-assay sum, PS3/BS3 stripped (the original/default behavior)
# - gene-deduplicated, PS3/BS3 stripped (merges each gene's assays' copies
#   of the same physical variant first, same idea as section 3a1c)
# - per-assay sum, PS3/BS3 KEPT (circularity check: how much of ClinGen's
#   "ground truth" is itself derived from functional-assay evidence)
# - gene-deduplicated, PS3/BS3 KEPT

# %%
from analysis.clingen import (
    build_clingen_confusion, build_gene_deduped_clingen_confusion,
    convert_3x2_to_2x3, plot_2x3_confusions_nature,
)
from analysis.manuscript_stats import latex_performance_table_clingen

# datasets_with_author computed once, above in section 4b.
print(f"ClinGen scope: {len(datasets_with_author)}/{len(datasets)} datasets have real author data")

CLINGEN_DATASETS = datasets_with_author  # explicit, adjustable — narrow if desired

CLINGEN_VARIANTS = [
    ("clingen_confusion", "gene-deduped=False, PS3/BS3 stripped=True", False, True),
    ("clingen_confusion_gene_deduped", "gene-deduped=True, PS3/BS3 stripped=True", True, True),
    ("clingen_confusion_with_ps3bs3", "gene-deduped=False, PS3/BS3 stripped=False", False, False),
    ("clingen_confusion_gene_deduped_with_ps3bs3", "gene-deduped=True, PS3/BS3 stripped=False", True, False),
]

for tag, desc, gene_dedup, strip_ps3bs3 in CLINGEN_VARIANTS:
    print(f"\n--- ClinGen confusion ({desc}) ---")
    clingen_confusion, clingen_genes, clingen_records = build_clingen_confusion(
        df_primary, DATASET_TSV, CLINGEN_DATASETS, use_oob=True,
        tree=tree, model_selections=model_selections, calibrations=calibrations,
        strip_functional_evidence=strip_ps3bs3,
    )
    if gene_dedup:
        clingen_confusion, clingen_genes = build_gene_deduped_clingen_confusion(clingen_records)

    if clingen_genes:
        fig = plot_2x3_confusions_nature({
            'auth': convert_3x2_to_2x3(clingen_confusion['auth']),
            'excalibr': convert_3x2_to_2x3(clingen_confusion['excalibr']),
        })
        save_and_show(fig, FIGURE_SUBDIRS["clingen"] / f"{tag}.png")
        _clingen_perf_latex = latex_performance_table_clingen({
            'excalibr': convert_3x2_to_2x3(clingen_confusion['excalibr']),
            'auth': convert_3x2_to_2x3(clingen_confusion['auth']),
        })
        save_latex_table(_clingen_perf_latex, FIGURE_SUBDIRS["tables"] / f"{tag}_performance.tex")
    else:
        print(f"  SKIP ClinGen confusion ({desc}): no genes with usable ClinGen evidence codes in scope")

# %% [markdown]
# ### 4d. Evidence-level comparison + assay-level statistics
#
# The ExCALIBR side (how many datasets reach each evidence level) is always
# computed fresh from pipeline output. The author/OddsPath (Brnich et al.)
# side and the assay-type/model-system/disease breakdowns need external data
# (`analysis.config.OP_EVIDENCE_CODES_CSV`, `analysis.config.ASSAY_METHOD_MAP_CSV`)
# not produced by this pipeline — those cells skip gracefully if missing.

# %%
from analysis.assay_stats import (
    compute_excalibr_evidence_counts, plot_evidence_comparison,
    compute_assay_evidence_stats, print_assay_evidence_report,
    plot_dataset_point_heatmap,
)

excalibr_path_counts, excalibr_ben_counts = compute_excalibr_evidence_counts(
    dataset_info_df_full, all_danz_oob_full,
)
print(f"ExCALIBR datasets reaching pathogenic evidence ±X: {excalibr_path_counts}")
print(f"ExCALIBR datasets reaching benign evidence ±X: {excalibr_ben_counts}")

if not config.warn_if_missing(config.OP_EVIDENCE_CODES_CSV, "OddsPath evidence-code CSV (author side)"):
    from src.assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant

    df_op = pd.read_csv(config.OP_EVIDENCE_CODES_CSV)
    levels = [1, 2, 4, 8]

    # OP_EVIDENCE_CODES_CSV now carries raw OddsAbnormal_clinvar_18_25 /
    # OddsNormal_clinvar_18_25 odds-of-pathogenicity values (plus a
    # per-dataset prior) instead of precomputed "Evidence Code
    # Abnormal"/"Evidence Code Normal" PS3/BS3 tier strings -- classify tiers
    # here using the same Tavtigian-constant thresholds
    # (C**(level/8), fixed prior=0.1, matching ExCALIBR's own
    # thresholds_from_prior elsewhere in this codebase) rather than the
    # row's own prior column. OddsAbnormal is a direct odds-of-pathogenicity
    # (higher = stronger PS3 evidence); OddsNormal is computed within the
    # benign-labeled subset and observed in [0, 1] (lower = stronger BS3
    # evidence), so its thresholds are the reciprocal.
    _C = get_tavtigian_constant(0.1)
    _path_thresholds = {lvl: _C ** (lvl / 8) for lvl in levels}
    _ben_thresholds = {lvl: 1.0 / (_C ** (lvl / 8)) for lvl in levels}

    auth_path_counts = {lvl: set() for lvl in levels}
    auth_ben_counts = {lvl: set() for lvl in levels}
    for ds in dataset_info_df_full["dataset"]:
        op_rows = df_op[df_op.Dataset == ds]
        if op_rows.empty:
            continue
        odds_abnormal = pd.to_numeric(op_rows["OddsAbnormal_clinvar_18_25"].iloc[0], errors="coerce")
        odds_normal = pd.to_numeric(op_rows["OddsNormal_clinvar_18_25"].iloc[0], errors="coerce")

        if pd.notna(odds_abnormal):
            for lvl in sorted(levels, reverse=True):
                if odds_abnormal >= _path_thresholds[lvl]:
                    for lvl2 in levels:
                        if lvl2 <= lvl:
                            auth_path_counts[lvl2].add(ds)
                    break
        if pd.notna(odds_normal):
            for lvl in sorted(levels, reverse=True):
                if odds_normal <= _ben_thresholds[lvl]:
                    for lvl2 in levels:
                        if lvl2 <= lvl:
                            auth_ben_counts[lvl2].add(ds)
                    break
    auth_path_counts = {k: len(v) for k, v in auth_path_counts.items()}
    auth_ben_counts = {k: len(v) for k, v in auth_ben_counts.items()}
    fig, _ = plot_evidence_comparison(
        excalibr_path_counts, excalibr_ben_counts, auth_path_counts, auth_ben_counts,
    )
    save_and_show(fig, FIGURE_SUBDIRS["manuscript"] / "num_datasets_reach_evidence.png")

if not config.warn_if_missing(config.ASSAY_METHOD_MAP_CSV, "assay method map (dataset_stats/point heatmap)"):
    assay_method_map = pd.read_csv(config.ASSAY_METHOD_MAP_CSV)
    dataset_stats, summary_dict = compute_assay_evidence_stats(
        dataset_info_df_full, all_danz_oob_full, assay_method_map,
    )
    print_assay_evidence_report(dataset_stats, summary_dict)

    fig, ax, _, _ = plot_dataset_point_heatmap(
        dataset_info_df_full, all_danz_oob_full,
        assay_method_map=assay_method_map, sort_by='model_system',
    )
    save_and_show(fig, FIGURE_SUBDIRS["manuscript"] / "points_heatmap_sort_model_system.png")

# %% [markdown]
# ## 5. Per-dataset calibration figures
#
# These already exist — `run_igvf_batch.py`/`run_pipeline.py` write
# `{dataset}_{comp}_visualization.png` for every dataset/component during the
# original run (via `generate_visualizations`/`plot_scoreset_best_config`).
# No regeneration needed; just display one as a sanity check that the run's
# own output is where you expect it.

# %%
from IPython.display import Image, display

_example_dataset = datasets[0]
_vis_candidates = sorted((OUTPUT_DIR / _example_dataset).glob(f"{_example_dataset}_*_visualization.png"))
if _vis_candidates:
    print(f"Existing visualizations for {_example_dataset}: {[p.name for p in _vis_candidates]}")
    display(Image(filename=str(_vis_candidates[0])))
else:
    print(f"  No existing visualization.png found for {_example_dataset} under {OUTPUT_DIR / _example_dataset}")

# %% [markdown]
# ## 5b. Four-dataset model-fit comparison figure (main text Figure "fig:fits")
#
# Reproduces the BRCA1/GCK/PTEN/CRX 2x2 model-fit comparison figure
# (`model_fits_comparison.png` in the paper). This was previously only
# produced by the standalone `test/plot_MSH2_ex.py`, via
# `plot_four_datasets_publication` (`src/assay_calibration/plot_utils/utils.py`)
# reading from a hand-curated `point_assignment_*/{dataset}/*.pkl` directory
# tree that no longer exists on disk for any dataset as of the pipeline
# refactor (that script is left fixed but unrunnable as-is; see its own
# `datasets_to_plot` comment). Here we instead pass `plot_four_datasets_publication`
# a `loader_fn` backed by `analysis.legacy_fits.load_scoreset_and_fits`, the
# current bridge from saved pipeline output (calibration/LR-values/
# PRECOMPUTED_FITS) to the same `(scoreset, indv_summary, fits, score_range,
# config, n_c, flipped, n_samples)` shape the plotting function expects --
# same visual output, current data source.

# %%
from analysis import legacy_fits
from src.assay_calibration.plot_utils.utils import plot_four_datasets_publication

FOUR_PANEL_DATASETS = [
    "BRCA1_Findlay_2018_clinvar_2018",
    "GCK_Gersing_2024_abundance",
    "PTEN_Mighell_2018_clinvar_2018",
    "CRX_Shepherdson_2024",
]

def _four_panel_loader(dataset):
    n_c, benign_method = legacy_fits.resolve_component_for(
        dataset, output_dir=str(OUTPUT_DIR), dataset_configs_path=DATASET_CONFIGS_PATH,
    )
    scoreset, indv_summary, fits, score_range, n_c2, n_samples, flipped = legacy_fits.load_scoreset_and_fits(
        dataset, output_dir=str(OUTPUT_DIR), dataset_tsv=DATASET_TSV, precomputed_fits=PRECOMPUTED_FITS,
        dataset_configs_path=DATASET_CONFIGS_PATH, n_c=n_c, benign_method=benign_method,
    )
    return scoreset, indv_summary, fits, score_range, (n_c, benign_method), n_c2, flipped, n_samples

_missing_four_panel = [d for d in FOUR_PANEL_DATASETS if d not in datasets]
if _missing_four_panel:
    print(f"  SKIP four-dataset model-fit figure: {_missing_four_panel} not in section 1's discovered datasets")
else:
    fig = plot_four_datasets_publication(
        FOUR_PANEL_DATASETS, dataset_configs, {}, set(), loader_fn=_four_panel_loader,
    )
    save_and_show(fig, FIGURE_SUBDIRS["manuscript"] / "model_fits_comparison.png")

# %% [markdown]
# ## 6. Yang distance bootstrap diagnostic
#
# Yang-distance goodness-of-fit is slow to compute live (~1-3 min/dataset at
# full bootstrap resolution — see `analysis/yang_distance.py`), so rather
# than recomputing it here, this reads `analysis.config.EXCALIBR_DATASETS_TABLE_CSV`
# — built once (offline) via `analysis/run_build_excalibr_datasets_table.py
# --with-yang`, which computes it for every dataset in one batch and also
# reconstructs the rest of `excalibr_datasets.csv` (calibration ranges,
# sample counts, metadata) alongside it. Rerun that script if pipeline
# output has changed since the table was last built.
#
# (The MSH2 example / "final pillar project" figure previously shown here
# was removed as redundant with section 7's Figure 4 / extended-data
# appendix panels.)

# %%
if not config.warn_if_missing(config.EXCALIBR_DATASETS_TABLE_CSV, "Yang distance table"):
    excalibr_datasets_table = pd.read_csv(config.EXCALIBR_DATASETS_TABLE_CSV)
    yang_cols = [c for c in excalibr_datasets_table.columns if c.startswith("yang_dist_")]
    in_scope = excalibr_datasets_table[excalibr_datasets_table["dataset"].isin(datasets)]
    if yang_cols and not in_scope.empty:
        print(f"\n{'=' * 80}\nYANG DISTANCE (goodness-of-fit), from {config.EXCALIBR_DATASETS_TABLE_CSV}\n{'=' * 80}")
        print(in_scope[["dataset"] + yang_cols].to_string(index=False))
    else:
        print("  SKIP Yang distance table: no yang_dist_* columns, or none of section 1's datasets present "
              "(rerun run_build_excalibr_datasets_table.py --with-yang)")
    in_scope[["dataset"] + yang_cols] if yang_cols else None

# %% [markdown]
# ### 6b. Yang distance 4-panel diagnostic figure (Extended Data Figure "fig:dists")
#
# Unlike the table above (pre-reduced medians from a batch-computed CSV),
# this recomputes the full per-bootstrap Yang-distance distribution live for
# just the same 4 datasets used in section 5b's Figure "fig:fits" (BRCA1/GCK/
# PTEN/CRX) -- the manuscript's own `fig:dists` shows exactly this scope, not
# all datasets, so recomputing the other ~76 datasets' full distributions
# here would be wasted cost. Reuses `_four_panel_loader` from section 5b so
# both figures are guaranteed to be built from the identical underlying fit.

# %%
if _missing_four_panel:
    print(f"  SKIP Yang distance diagnostic figure: {_missing_four_panel} not in section 1's discovered datasets")
else:
    from analysis.calibration_plots import plot_yang_distance_diagnostic

    plot_yang_distance_diagnostic(
        FOUR_PANEL_DATASETS, _four_panel_loader, FIGURE_SUBDIRS["manuscript"],
    )

# %% [markdown]
# ## 7. Figure 4, extended-data appendix, gene-performance/OR scatter
#
# Ported from `test/auxiliary_fig_creation/`. Some panels need external,
# non-pipeline comparison data (REVEL/AM/MutPred2 thresholds, OR estimates) —
# see `analysis/config.py`; those cells print a warning and skip if the
# corresponding path doesn't exist, rather than failing the whole run.

# %%
from analysis.figure4 import driver as figure4_driver

# Reuse the confusion matrices already built in section 3 (primary_method's
# danz/auth matrices) instead of having build_figure4 re-discover pipeline
# output, re-load every variants CSV, and rebuild author labels from scratch.
figure4_driver.build_figure4(
    output_dir=OUTPUT_DIR, figure_dir=FIGURE_SUBDIRS["manuscript"],
    danzs_oob=conf_by_method[primary_method],
    auths_oob=auth_by_method[primary_method],
    dataset_names=datasets,
    vus_pct_danz=_aggregate_coverage_pct(vus_by_method[primary_method]),
    vus_pct_auth=_aggregate_coverage_pct(auth_vus_by_method[primary_method]),
)

# %% [markdown]
# ### 7a. RAD51D/XRCC2/BARD1 extra fit plots
#
# Supplementary to Figure 4 in the legacy script, but not one of its panels --
# kept separate (`analysis.extra_gene_fits`) so `analysis/figure4/driver.py`
# can be handed to someone reproducing just Figure 4 without also needing
# these. Currently a no-op (prints a warning and skips): depends on a
# `fit_hist_snv_plot` module that only exists as an import statement in the
# legacy script, see that module's own TODO.

# %%
from analysis.extra_gene_fits import build_extra_gene_fits

build_extra_gene_fits(
    OUTPUT_DIR, DATASET_TSV, PRECOMPUTED_FITS, DATASET_CONFIGS_PATH,
    FIGURE_SUBDIRS["manuscript"],
)

# %%
from analysis import extended_data_appendix

# dataset_list=datasets (section 1's own list) instead of the module default
# (None -> _default_dataset_list_full, which re-discovers datasets from
# OUTPUT_DIR itself and applies its own datasets_to_exclude.pkl + hardcoded
# TP53/F9 exclusions) -- keeps the appendix in sync with whatever's actually
# in scope for the rest of this notebook, with no separate exclusion list.
extended_data_appendix.build_appendix_pdf(
    dataset_list=datasets,
    output_path=FIGURE_SUBDIRS["manuscript"] / "extended_data_appendix.pdf",
    plot_thresholds=True,
)

# %%
from analysis import gene_performance_scatter

gene_performance_scatter.build_gene_performance_scatter(output_dir=OUTPUT_DIR, figure_dir=FIGURE_SUBDIRS["manuscript"])

# %% [markdown]
# ## 8. Dataset description table

# %%
from analysis.gene_table import build_dataset_table

dataset_table = build_dataset_table()
dataset_table.head()

# %% [markdown]
# ## 9. Per-scoreset VUS evidence breakdown, ranked by indeterminate count
#
# For each dataset (scoreset), among only the `is_vus` variants (ClinVar
# VUS -- always ClinVar-2026-based per-variant, see
# `variant_evidence.py::_get_variant_is_vus`, regardless of whatever
# `clinvar_release` that dataset's own P/LP/B/LB controls use), classify each
# VUS by the sign of its effective evidence points (`analysis.plot_common.
# effective_points`, OOB with in-bag fallback -- same points
# `build_confusion_matrix` uses for BLB/PLP rows): negative -> benign-leaning
# evidence, zero -> indeterminate (no evidence, inside the IR range),
# positive -> pathogenic-leaning evidence. Ranked descending by the raw
# indeterminate VUS count -- the scoresets contributing the most "VUS that
# stay VUS" (no evidence pull either direction) surface first.

# %%
from analysis.plot_common import effective_points as _effective_points_vus

_vus_rows = []
for _ds in datasets:
    _df_ds = df_primary[df_primary["dataset"] == _ds]
    if "is_vus" not in _df_ds.columns or _df_ds.empty:
        continue
    _df_vus = _df_ds[_df_ds["is_vus"].fillna(False).astype(bool)]
    _n_vus = len(_df_vus)
    if _n_vus == 0:
        continue
    _pts = _effective_points_vus(_df_vus, use_oob=True, label=_ds, context="VUS")
    _n_neg = int((_pts < 0).sum())
    _n_ind = int((_pts == 0).sum())
    _n_pos = int((_pts > 0).sum())
    _vus_rows.append({
        "dataset": _ds,
        "n_vus": _n_vus,
        "n_indeterminate": _n_ind,
        "pct_indeterminate": 100 * _n_ind / _n_vus,
        "n_negative": _n_neg,
        "pct_negative": 100 * _n_neg / _n_vus,
        "n_positive": _n_pos,
        "pct_positive": 100 * _n_pos / _n_vus,
    })

vus_evidence_df = pd.DataFrame(_vus_rows)
if not vus_evidence_df.empty:
    vus_evidence_df = vus_evidence_df.sort_values(
        "n_indeterminate", ascending=False,
    ).reset_index(drop=True)
    print(f"\n{'=' * 80}\nPER-SCORESET VUS EVIDENCE ({primary_method}), "
          f"ranked by # indeterminate VUS\n{'=' * 80}")
    print(vus_evidence_df.round(1).to_string(index=False))
else:
    print("  SKIP per-scoreset VUS evidence breakdown: no is_vus variants in scope")

vus_evidence_df
