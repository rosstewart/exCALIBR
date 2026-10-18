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
# # Pipeline output analysis — canonical pipeline for the ExCALIBR manuscript
#
# One notebook regenerates every number, figure, table and Supplementary Data
# file of the Nature Genetics manuscript (`ExCALIBR_latex_src/main.tex`,
# `supplement.tex`) from `run_pipeline.py` / `run_igvf_batch.py` output, plus
# the outputs other projects use. **Run All is the button.** For a headless
# run of this same notebook: `bash scripts/build_manuscript.sh`.
#
# Layout (real files under `OUTPUT_DIR`; `~/exCALIBR/manuscript/` symlinks to
# each, and `ExCALIBR_latex_src/Figures` -> `figures/`; see analysis/config.py):
#
# - `exploratory_figures/` (`FIGURE_DIR`): every figure drawn below.
# - `figures/` (`MANUSCRIPT_FIGURE_DIR`, uploaded to Overleaf as `Figures/`):
#   only the files the .tex includes, copied from `exploratory_figures/` by the
#   Write step, plus `generated/`.
# - `tables/`, `supplementary_data/`, `submission/`, `pillar_project_figures/`.
#
# - **Part A — this manuscript, in manuscript order.** A1 shared state, A2
#   Methods/Data numbers, A3 Fig. 2 + ED Fig. 1, A4 Fig. 3 + ED Tables 1-2, A5
#   Fig. 4 + ED Tables 3-4, A6 Fig. 5 + ED Fig. 2, A7 ED Fig. 3 + ED Table 5,
#   A8 Fig. 6, A9 ED Fig. 4, A10 ED Table 6, A11 Supplementary Tables, A12
#   Supplementary Data 1-3. Every number quoted in the .tex is registered with
#   `REG` (`analysis/manuscript_numbers.py`) where it is computed.
# - **Part B — IGVF Pillar Project** (Figure 4, extra gene fits, all
#   F9/TP53-excluded outputs) -> `pillar_project_figures/`.
# - **Part C — IGVF data release tables** (`tables/`).
# - **Part D — diagnostics / exploratory outputs** (unchanged paths).
# - **Write** — `Figures/generated/macros.tex` (one `\newcommand` per number,
#   in manuscript order), `tables/*.tex` (tabular bodies), `manifest.json`,
#   `build_report.md`, and flattened submission copies (`submission/`).
#
# `MANUSCRIPT_ONLY=1` skips Parts B-D. `EXCALIBR_STRICT=1` turns stale or
# missing inputs into errors (the headless wrapper sets it); exemptions go in
# `config.STALENESS_WAIVERS`. The manuscript is the source of truth for all
# wording: the pipeline only supplies numbers, and generated table bodies are
# checked against the manuscript's own tables with numbers masked.
#
# **All paths are set in `analysis/config.py`** (or the matching `EXCALIBR_*`
# environment variables).
#
# Running this file as a plain script with CLI arguments still runs the small
# CLI `main()` below (confusion + evidence figures only) and exits.

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


import os
import re
import matplotlib
if os.environ.get("MPLBACKEND"):
    # Headless notebook execution (scripts/build_manuscript.sh sets
    # MPLBACKEND=Agg): write figures to disk without embedding them.
    matplotlib.use(os.environ["MPLBACKEND"])
elif _running_as_notebook():
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
from analysis.discovery import discover_outputs, load_all_variants, resolve_component, resolve_dataset_tsv_name
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
FIGURE_DIR = Path(config.FIGURE_DIR)  # exploratory_figures/: working tree for every figure
MANUSCRIPT_FIGURE_DIR = Path(config.MANUSCRIPT_FIGURE_DIR)  # Overleaf Figures/
SUPP_DATA_DIR = Path(config.SUPP_DATA_DIR)
SUBMISSION_DIR = Path(config.SUBMISSION_DIR)
for _d in (FIGURE_DIR, MANUSCRIPT_FIGURE_DIR, SUPP_DATA_DIR, SUBMISSION_DIR):
    _d.mkdir(parents=True, exist_ok=True)


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
        "manual_prior": figure_dir / "clinvar_comparisons" / "manual_prior",
        "pathomechanism": figure_dir / "clinvar_comparisons" / "pathomechanism",
        "clingen": figure_dir / "clingen_comparisons",
        "path_percentile": figure_dir / "path_percentile_ablation",
        "manuscript": figure_dir / "manuscript_figures",
        "tables": Path(config.TABLES_DIR),
        # Part B: IGVF Pillar Project outputs (Figure 4, extra gene fits,
        # F9/TP53-excluded variants), kept apart from this manuscript's.
        "pillar_project": Path(config.PILLAR_FIGURE_DIR),
        "pillar_project_tables": Path(config.PILLAR_FIGURE_DIR) / "tables",
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return subdirs


FIGURE_SUBDIRS = figure_subdirs(FIGURE_DIR)

# Manuscript registry: every number/table/figure the .tex uses is registered
# where it is computed; the final "Write" section emits macros.tex etc.
from analysis.manuscript_numbers import ManuscriptRegistry

MAIN_TEX = Path(config.MANUSCRIPT_TEX_DIR) / "main.tex"
SUPP_TEX = Path(config.MANUSCRIPT_TEX_DIR) / "supplement.tex"
REG = ManuscriptRegistry(
    tex_paths=[MAIN_TEX, SUPP_TEX], out_dir=MANUSCRIPT_FIGURE_DIR / "generated", repo_root=_ROOT,
    waivers=config.STALENESS_WAIVERS,
)

print(f"OUTPUT_DIR       = {OUTPUT_DIR}")
print(f"DATASET_TSV      = {DATASET_TSV}")
print(f"DATASET_CONFIGS  = {DATASET_CONFIGS_PATH}")
print(f"PRECOMPUTED_FITS = {PRECOMPUTED_FITS}")
print(f"FIGURE_DIR       = {FIGURE_DIR}  (manuscript copies -> {MANUSCRIPT_FIGURE_DIR})")
print(f"MANUSCRIPT .tex  = {MAIN_TEX}, {SUPP_TEX}")
print(f"strict={REG.strict}  manuscript_only={config.MANUSCRIPT_ONLY}")


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
# Everything below runs as notebook cells (Jupyter / VSCode / `jupyter
# nbconvert --execute`); a plain `python analyze_pipeline_output.py ...`
# invocation exits above.

# %% [markdown]
# # Part A — Nature Genetics manuscript (manuscript order)

# %% [markdown]
# ## A1. Shared state: pipeline outputs, variants, author labels, confusion matrices

# %% [markdown]
# ### A1a. Discover pipeline outputs and load variants

# %%
# Slow Part A steps are cached on disk (analysis/step_cache.py); these are the
# canonical-output fingerprints most of them depend on.
from analysis.step_cache import cached
CANONICAL_INPUTS = [DATASET_TSV, (OUTPUT_DIR, "*/*_variants.csv"), (OUTPUT_DIR, "*/*_calibration.json")]

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
# ### A1a (cont.). Attach author labels (enables confusion vs. author + evidence-by-author panels)

# %%
df = attach_author_labels(df, DATASET_TSV)


# %% [markdown]
# ### A1b. Per-dataset confusion matrices (every comparison below reuses these)

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

primary_method = methods[0]
auths = auth_by_method[primary_method]
both_det_excalibr = both_det_excalibr_by_method[primary_method]
both_det_author = both_det_author_by_method[primary_method]

# %% [markdown]
# ### A1c. Gene-deduplicated confusion matrices (ED Table 4, ED Table 6)
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
# points_col="points": build_gene_deduped_variants' own merged evidence-points
# column (abs-max across assays), not ExCALIBR's raw standard_points/
# oob_points -- see build_vus_coverage. deduped_variants already carries
# is_vus (any() across assays, see build_gene_deduped_variants), so this
# needs no further plumbing.
deduped_vus_coverage = build_vus_coverage(deduped_variants, points_col="points")

deduped_variants_with_author = restrict_to_genes_with_author_data(deduped_variants)
n_genes_with_author = deduped_variants_with_author["gene"].nunique()
print(f"Gene-deduplicated, restricted to genes with real author data: {n_genes_with_author}/{n_genes_deduped} genes "
      f"({len(deduped_variants_with_author):,} variants)")
deduped_matrix_for_author = build_deduped_confusion_matrix(deduped_variants_with_author)
deduped_author_matrix = build_deduped_author_confusion_matrix(deduped_variants_with_author)
# Same restricted (genes-with-real-author-data) scope on both sides, so
# ExCALIBR's own VUS-determinate% and the author VUS-determinate% shown in
# this panel's caption describe the identical gene subset as the matrices
# above them -- same principle as build_author_vus_coverage's own guard.
deduped_vus_for_author = build_vus_coverage(deduped_variants_with_author, points_col="points")
deduped_author_vus = build_author_vus_coverage(deduped_variants_with_author)

if deduped_matrix is not None:
    make_single_confusion_figure(
        [deduped_matrix], ["gene_deduped"], label=primary_method, figure_dir=FIGURE_SUBDIRS["author"],
        filename="excalibr_vs_clinvar_gene_deduped.png",
        title_suffix=f"({n_genes_deduped} genes, {len(deduped_variants):,} unique variants, gene-deduplicated)",
        vus_coverages=[deduped_vus_coverage],
    )
else:
    print("  SKIP gene-deduplicated confusion figure: no P/LP or B/LB variants in deduped scope")

if deduped_matrix_for_author is not None and deduped_author_matrix is not None:
    make_confusion_figure(
        danzs_m1=[deduped_matrix_for_author], danzs_m2=[deduped_author_matrix], dataset_names=["gene_deduped"],
        label1=primary_method, label2="author", figure_dir=FIGURE_SUBDIRS["author"],
        filename="excalibr_vs_author_gene_deduped.png",
        vus_coverages_m1=[deduped_vus_for_author],
        vus_coverages_m2=[deduped_author_vus],
    )
else:
    print("  SKIP gene-deduplicated ExCALIBR-vs-author figure: no matrix on one or both sides")

# %% [markdown]
# ### A1d. Evidence arrays shared by Fig. 3, Fig. 5, Fig. 6, ED Tables 1-2
#
# Author panel restricted to datasets with real author annotations; the
# ClinVar panel spans every dataset (see `make_combined_evidence_figure`).

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

# %% [markdown]
# ## A2. Data and Methods numbers (main text Methods; Supplementary Section 1)
#
# Counts of the harmonized resource vs. the calibrated set, assay types,
# exclusions, circularity handling, and method constants. Method constants are
# read from the code that implements them (bootstrap preset, percentiles,
# SpliceAI threshold, model-selection alpha), so the text cannot drift from the
# code.

# %%
from analysis.discovery import DEFAULT_EXCLUDED_DATASETS, load_master_df
from analysis.plot_common import sample_matches as _sample_matches
from analysis.supplementary_data import _resolved_calibration
import ast as _ast
from src.assay_calibration.pipeline.config import PipelineConfig
from src.assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant

# Dataset config edits alone (e.g. adding a side-analysis dataset) do not
# make the canonical calibrations stale, so only the TSV is compared.
REG.require(OUTPUT_DIR, "canonical pipeline output (calibration JSONs)", newer_than=[DATASET_TSV],
            own_pattern="*_calibration.json")
REG.require(DATASET_TSV, "master integrated variant-effect TSV")

_gene_of = lambda d: d.split("_")[0]
REG.num("NumDatasetsMalvika", len(tree), source="len(tree): datasets in OUTPUT_DIR (harmonized resource)",
        inputs=[OUTPUT_DIR])
REG.num("NumGenes", len({_gene_of(d) for d in tree}), source="genes of tree datasets")
REG.num("NumDatasets", len(datasets), source="len(datasets): calibrated (DEFAULT_EXCLUDED_DATASETS dropped)")
REG.num("NumGenesCalibrated", len({_gene_of(d) for d in datasets}), source="genes of calibrated datasets")
REG.num("NumSynonymousDatasets", df.loc[_sample_matches(df, "Synonymous"), "dataset"].nunique(),
        source="datasets with >=1 Synonymous-sample variant")

_excluded = sorted(set(tree) & set(DEFAULT_EXCLUDED_DATASETS))
REG.check("excluded-datasets", len(_excluded) == 3 and len(tree) - len(datasets) == 3,
          f"text names two meta-analyses + SFPQ; excluded now = {_excluded}", "main.tex Methods/Data")

# Harmonized-resource variant counts straight from the master TSV, over every
# tree dataset. Unique variant = the pipeline's own identity rule
# (analysis.multi_scoreset.genomic_variant_key): nt-level rows by
# Gene/Chrom/hgvs_c, aa-level rows by Gene + amino-acid change.
_master = load_master_df(str(DATASET_TSV))
_tree_tsv_names = {d.replace("_clinvar_2018", "") for d in tree}
_m = _master[_master["Dataset"].isin(_tree_tsv_names)]
_is_nt = _m["nucleotide_or_aa"].eq("nt")
_vkey = (_m["Gene"].astype(str) + "|" + _m["Chrom"].astype(str) + "|" + _m["hgvs_c"].astype(str)).where(
    _is_nt, _m["Gene"].astype(str) + "|aa|" + _m["aa_ref"].astype(str) + _m["aa_pos"].astype(str)
    + _m["aa_alt"].astype(str))
# Fixed by the authors: distinct (hgvs_c, hgvs_g) pairs in the IGVF harmonized
# resource. hgvs_g is not in DATASET_TSV, so it can't be recomputed here; the
# pipeline's own identity rule (genomic_variant_key) is printed for comparison.
REG.num("DataNumUniqueVariantsHarmonized", 325808,
        source="distinct (hgvs_c, hgvs_g) pairs in the IGVF harmonized resource (author-provided; hgvs_g not in DATASET_TSV)")
print(f"  unique variants by genomic_variant_key (not used in text): {_vkey.nunique():,}")
_splice = _m.groupby("Dataset")["splice_measure"].agg(lambda s: s.dropna().astype(str).str.strip().eq("Yes").any())
REG.num("DataNumSpliceDetectingAssays", int(_splice.sum()), source="tree datasets with splice_measure == Yes")
# gnomAD allele frequencies (no AF filter is applied): common variants among
# the unique gnomAD variants of the calibrated datasets.
_calib_tsv_names = {d.replace("_clinvar_2018", "") for d in datasets}
_g = _master[_master["Dataset"].isin(_calib_tsv_names) & _master["gnomad_MAF"].notna()]
_g = _g.drop_duplicates(["Gene", "Chrom", "hg38_start", "ref_allele", "alt_allele"])
REG.num("DataNumGnomadUnique", len(_g), source="unique gnomAD variants (Gene/Chrom/hg38_start/ref_allele/alt_allele), calibrated datasets",
        inputs=[DATASET_TSV])
REG.num("DataNumGnomadCommon", int((_g["gnomad_MAF"] >= 0.01).sum()), source="of those, gnomad_MAF >= 1%")
REG.pct("DataPctGnomadCommon", int((_g["gnomad_MAF"] >= 0.01).sum()), len(_g), digits=2,
        source="% of unique gnomAD variants with gnomad_MAF >= 1%")
REG.num("DataNumCdnaAssays", int((~_splice).sum()), source="tree datasets with splice_measure == No")
REG.check("assay-split-sums", int(_splice.sum()) + int((~_splice).sum()) == len(tree),
          f"{int(_splice.sum())} + {int((~_splice).sum())} vs {len(tree)} harmonized datasets", "main.tex Methods/Data")

# Circularity: genes whose selected calibration uses pre-2018 ClinVar controls.
_resolved_cals = {d: _resolved_calibration(d, tree, model_selections, dataset_configs) for d in datasets}
_circ_genes = sorted({_gene_of(d) for d, c in _resolved_cals.items() if c.get("clinvar_2018")})
_it = [f"\\textit{{{g}}}" for g in _circ_genes]
REG.text("DataCircularityGenes", ", ".join(_it[:-1]) + ", and " + _it[-1] if len(_it) > 2 else " and ".join(_it),
         source="genes of datasets calibrated with clinvar_2018 controls")

_curated_meta = pd.read_csv(config.CURATED_DATASET_METADATA_CSV, dtype=str)
REG.num("DataNumWooIntervalDatasets",
        int(_curated_meta[_curated_meta["dataset"].isin(datasets)]["new_functional_classifications_created"]
            .eq("Yes").sum()), words=True,
        source="curated new_functional_classifications_created == Yes", inputs=[config.CURATED_DATASET_METADATA_CSV])

# Method constants, read from the implementing code.
_PIPELINE_PRESET = "finest"
# Read run_pipeline.BOOTSTRAP_PRESETS from source: importing run_pipeline
# reconfigures sys.stdout, which a Jupyter kernel's stream doesn't support.
_presets_node = next(
    n for n in _ast.parse((_ROOT / "run_pipeline.py").read_text()).body
    if isinstance(n, _ast.Assign) and any(getattr(t, "id", None) == "BOOTSTRAP_PRESETS" for t in n.targets)
)
BOOTSTRAP_PRESETS = _ast.literal_eval(_presets_node.value)
_n_boot, _n_restarts = BOOTSTRAP_PRESETS[_PIPELINE_PRESET]
_pcfg = PipelineConfig(dataset_csv=str(DATASET_TSV), dataset_name="defaults", output_dir="/tmp")  # defaults only
FIXED_PRIOR = 0.1
REG.num("NumBootstraps", _n_boot, kind="method", source=f"run_pipeline.BOOTSTRAP_PRESETS[{_PIPELINE_PRESET!r}]")
REG.num("NumRestarts", _n_restarts, kind="method", source=f"run_pipeline.BOOTSTRAP_PRESETS[{_PIPELINE_PRESET!r}]")
REG.num("MethBootstrapAgreementPct", 100 - _pcfg.pathogenic_percentile, fmt=".0f", kind="method",
        source="100 - PipelineConfig.pathogenic_percentile")
REG.num("MethPathPercentile", _pcfg.pathogenic_percentile, fmt=".0f", kind="method",
        source="PipelineConfig.pathogenic_percentile")
REG.num("MethBenPercentile", _pcfg.benign_percentile if _pcfg.benign_percentile is not None
        else 100 - _pcfg.pathogenic_percentile, fmt=".0f", kind="method", source="PipelineConfig.benign_percentile")
REG.num("MethSpliceAIThreshold", _pcfg.spliceai_threshold, fmt=".1f", kind="method",
        source="PipelineConfig.spliceai_threshold")
REG.num("SuppModelSelectionPct", 100 * (1 - _pcfg.model_selection_alpha), fmt=".0f", kind="method",
        source="100 * (1 - PipelineConfig.model_selection_alpha)")
REG.num("MethFixedPrior", FIXED_PRIOR, fmt=".1f", kind="method",
        source="fixed prior: manual-prior rerun, acmgscaler, OddsPath tiering (A6)")
REG.num("MethFixedPriorPct", 100 * FIXED_PRIOR, fmt=".0f", kind="method", source="MethFixedPrior as a percentage")
REG.num("ResFitYangThreshold", config.YANG_GOOD_FIT_THRESHOLD, fmt=".1f", kind="method",
        source="config.YANG_GOOD_FIT_THRESHOLD")
# The text quotes the cited papers' rounded constants (350, 1124); the code's
# exact values are within 1% of them.
# Tavtigian c vs. prior (Discussion limitation): c is smallest near 0.25.
_c_grid = np.round(np.arange(0.01, 0.991, 0.01), 2)
_c_vals = np.array([get_tavtigian_constant(float(q)) for q in _c_grid])
REG.num("MethTavtigianCMinPrior", float(_c_grid[np.argmin(_c_vals)]), fmt=".2f", kind="method",
        source="prior minimizing get_tavtigian_constant (0.01 grid)")
REG.num("MethTavtigianCMin", float(_c_vals.min()), kind="method", source="min c over prior grid")
REG.num("MethTavtigianCPriorFour", get_tavtigian_constant(0.4), kind="method", source="c at prior 0.4")
# Methods/Discussion (Tejura et al. Supplementary Note): -1 point (LR = c^(-1/8))
# leaves a posterior > 10% once the prior exceeds this value.
_g_m1 = np.round(np.arange(0.01, 0.6, 0.001), 3)
_post_m1 = np.array([(q / (1 - q)) * get_tavtigian_constant(float(q)) ** (-1 / 8) for q in _g_m1])
_post_m1 = _post_m1 / (1 + _post_m1)
_m1_prior = float(_g_m1[np.argmax(_post_m1 > 0.10)])
REG.num("MethMinusOneTenPrior", 100 * _m1_prior, fmt=".1f", kind="method",
        source="prior (%) above which -1 point gives posterior > 10%")
REG.check("minus-one-ten-prior", 0.10 < _m1_prior < 0.25, f"-1 point posterior exceeds 10% above prior {_m1_prior:.3f}",
          "main.tex Methods (calibration framework)")
REG.check("tavtigian-c-0.1", abs(get_tavtigian_constant(0.1) / 350 - 1) < 0.01,
          f"c(0.1) = {get_tavtigian_constant(0.1):.1f} (text: 350)", "main.tex Methods/Calibration framework")
REG.check("tavtigian-c-0.044", abs(get_tavtigian_constant(0.044) / 1124 - 1) < 0.01,
          f"c(0.044) = {get_tavtigian_constant(0.044):.1f} (text: 1124)", "main.tex Methods/Calibration framework")

# Supplement: datasets whose evidence thresholds were removed rather than
# extended (liberal_monotonicity == 0 in the selected calibration).
_thr_removed = sorted(d for d, c in _resolved_cals.items() if c.get("liberal_monotonicity") == 0)
REG.num("SuppNumThresholdRemoved", len(_thr_removed), words=True,
        source="calibrations with liberal_monotonicity == 0")
REG.text("SuppThresholdRemovedDatasets",
         ", ".join("\\texttt{" + d.replace("_clinvar_2018", "").replace("_", "\\_") + "}" for d in _thr_removed),
         source="calibrations with liberal_monotonicity == 0")

# %% [markdown]
# ExCALIBR scores every variant the underlying assay measured, not just the
# subset that ends up in a calibration-relevant sample group (P/LP, B/LB,
# gnomAD, Synonymous) or is a ClinVar VUS -- but `load_all_variants` (and
# therefore `df`/`deduped_variants` used everywhere else in this notebook)
# only ever exports that kept-plus-VUS subset, since that's all
# ExCALIBR's own evidence/confusion-matrix pipeline needs. "Variant effect
# measurements" for the manuscript, though, means literally everything the
# assay measured -- so this rebuilds each dataset's Scoreset (same pattern
# `analysis.clingen`/`analysis.author_labels` already use for their own
# side-channel needs) and pulls every `get_variants_by_id()` group via the
# new `get_all_variant_groups` (src/assay_calibration/pipeline/
# variant_evidence.py), not just the kept/VUS-only ones. This never touches
# `df`/`load_all_variants`'s own output -- purely additive, computed once
# here and reused by both `_compute_manuscript_summary` calls below.


# %%
# `_build_all_variant_groups_for_dataset` used to be defined inline here --
# it's now `analysis.all_variant_evidence.build_all_variant_groups_table`
# (reused, unchanged behavior, by analysis/run_build_all_variant_evidence_tables.py
# and the variant-assay/variant-aggregate table section right below), so this
# cell is now just the call site.
from analysis.all_variant_evidence import build_all_variant_groups_table

df_all_groups = build_all_variant_groups_table(
    tree=tree, model_selections=model_selections, dataset_configs=dataset_configs,
    calibrations=calibrations, dataset_tsv=str(DATASET_TSV),
    datasets=sorted(df_primary_dedup["dataset"].unique()), primary_method=primary_method,
)
print(f"All-variant-groups population: {len(df_all_groups):,} rows across "
      f"{df_all_groups['dataset'].nunique()} datasets (vs {len(df_primary_dedup):,} rows in the "
      f"kept+VUS-only population load_all_variants exports)")

# %% [markdown]
# ## A3. Figure 2 (model fits) and Extended Data Figure 1 (Yang distances); fit-quality numbers

# %% [markdown]
# ### A3a. Four-dataset model-fit comparison figure (main text Figure "fig:fits")
#
# Reproduces the BRCA1/LDLR/PTEN/CRX 2x2 model-fit comparison figure
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
    "G6PD_IGVF",  # abundance assay assigning no benign evidence (was LDLR_Tabet_2025_abundance, which assigns weak benign)
    "PTEN_Mighell_2018_clinvar_2018",
    "DDX3X_Radford_2023",  # pathogenic evidence at both score extremes (was CRX_Shepherdson_2024, no evidence)
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
    # Fig. 2: count histograms with fitted densities scaled to counts, and an
    # evidence strip (score range and number of SNVs per evidence strength)
    # under each dataset instead of threshold lines.
    fig = plot_four_datasets_publication(
        FOUR_PANEL_DATASETS, dataset_configs, {}, set(), loader_fn=_four_panel_loader,
        EVIDENCE_STRIP=True, COUNTS=True,
    )
    save_and_show(fig, FIGURE_SUBDIRS["manuscript"] / "model_fits_comparison.png")
REG.figure("fig:fits", FIGURE_SUBDIRS["manuscript"] / "model_fits_comparison.png",
           inputs=[PRECOMPUTED_FITS, DATASET_TSV])

# %% [markdown]
# ### A3b. Yang distance table (fit quality numbers; Supplementary Data 1)
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
    from analysis.supplementary_data import canonical_yang_rows
    excalibr_datasets_table = pd.read_csv(config.EXCALIBR_DATASETS_TABLE_CSV)
    # One row per dataset: the table has a row per fitted component.
    excalibr_datasets_table = canonical_yang_rows(
        excalibr_datasets_table, [d for d in datasets if d in set(excalibr_datasets_table["dataset"])],
        tree, model_selections, dataset_configs)
    yang_cols = [c for c in excalibr_datasets_table.columns if c.startswith("yang_dist_")]
    in_scope = excalibr_datasets_table[excalibr_datasets_table["dataset"].isin(datasets)]
    if yang_cols and not in_scope.empty:
        print(f"\n{'=' * 80}\nYANG DISTANCE (goodness-of-fit), from {config.EXCALIBR_DATASETS_TABLE_CSV}\n{'=' * 80}")
        print(in_scope[["dataset"] + yang_cols].to_string(index=False))
    else:
        print("  SKIP Yang distance table: no yang_dist_* columns, or none of section 1's datasets present "
              "(rerun run_build_excalibr_datasets_table.py --with-yang)")
    in_scope[["dataset"] + yang_cols] if yang_cols else None

# %%
# Fit-quality numbers (Results "ExCALIBR fits diverse score distributions").
if REG.require(config.EXCALIBR_DATASETS_TABLE_CSV, "Yang-distance table (Supp Data 1, fit-quality numbers)",
               newer_than=[(OUTPUT_DIR, "*_calibration.json")],
               expect_datasets=set(datasets),
               # The table can also hold excluded datasets; only coverage of `datasets` matters.
               found_datasets=set(pd.read_csv(config.EXCALIBR_DATASETS_TABLE_CSV)["dataset"]) & set(datasets)
               if Path(config.EXCALIBR_DATASETS_TABLE_CSV).exists() else set(),
               regen="python -m analysis.run_build_excalibr_datasets_table --with-yang "
                     f"--output {config.EXCALIBR_DATASETS_TABLE_CSV}"):
    _thr = config.YANG_GOOD_FIT_THRESHOLD
    from analysis.supplementary_data import canonical_yang_rows
    _y = pd.read_csv(config.EXCALIBR_DATASETS_TABLE_CSV)
    _y = canonical_yang_rows(_y[_y["dataset"].isin(datasets)], datasets, tree, model_selections,
                             dataset_configs).set_index("dataset")
    _ycols = ["yang_dist_plp", "yang_dist_blb", "yang_dist_gnomad", "yang_dist_synonymous"]
    _yv = _y[_ycols].apply(pd.to_numeric, errors="coerce")
    _pop_syn_ok = (_yv["yang_dist_gnomad"] < _thr) & (_yv["yang_dist_synonymous"].isna() | (_yv["yang_dist_synonymous"] < _thr))
    REG.check("fit-pop-syn-all", bool(_pop_syn_ok.all()) and len(_yv) == len(datasets),
              f"{int(_pop_syn_ok.sum())}/{len(datasets)} datasets have gnomAD and synonymous distances < {_thr}",
              "main.tex Results/fit quality")
    _all_ok = (_yv.isna() | (_yv < _thr)).all(axis=1)
    REG.num("ResFitGoodFitAllDatasets", int(_all_ok.sum()), source="datasets with every available Yang distance < threshold",
            inputs=[config.EXCALIBR_DATASETS_TABLE_CSV])
    REG.pct("ResFitGoodFitAllPct", int(_all_ok.sum()), len(datasets), digits=0, source="ResFitGoodFitAllDatasets / NumDatasets")
    _poor = sorted(_all_ok.index[~_all_ok])
    # Unique genes in dataset order (the text reads "from CARD11 and SCN5A").
    _poor_genes = [f"\\textit{{{g}}}" for g in dict.fromkeys(_gene_of(d) for d in _poor)]
    REG.num("ResFitPoorFitNum", len(_poor), words=True, source="datasets with any Yang distance >= threshold")
    REG.text("ResFitPoorFitGenes", " and ".join(_poor_genes) if len(_poor_genes) <= 2
             else ", ".join(_poor_genes[:-1]) + ", and " + _poor_genes[-1], source="genes of poor-fit datasets")
    REG.check("fit-all-datasets", not _poor, f"datasets with any Yang distance >= {_thr}: {_poor}",
              "main.tex Results/fit quality (all datasets met the threshold)")
    # Held-out (out-of-bag) distances, per sample with >= YANG_HELDOUT_MIN_N held-out variants.
    if not config.warn_if_missing(config.YANG_HELDOUT_CSV, "held-out Yang distances"):
        from analysis.yang_distance import heldout_yang_table
        _yh = heldout_yang_table().set_index("dataset").reindex(datasets)
        REG.require(config.YANG_HELDOUT_CSV, "held-out Yang distances (Results fit quality, Supp Data 1)",
                    expect_datasets=set(datasets), found_datasets=set(_yh.dropna(how="all").index),
                    regen="python -m analysis.skew_locked_yang --split val")
        REG.num("ResFitHeldOutMinN", config.YANG_HELDOUT_MIN_N, kind="method",
                source="config.YANG_HELDOUT_MIN_N (held-out variants needed to report a distance)")
        _bad = {}
        for _s, _tag in (("plp", "PLP"), ("blb", "BLB"), ("gnomad", "Gnomad"), ("synonymous", "Syn")):
            _ok = _yh[f"n_val_{_s}"] >= config.YANG_HELDOUT_MIN_N
            REG.num(f"ResFitHeldOutN{_tag}", int(_ok.sum()),
                    source=f"datasets with >= {config.YANG_HELDOUT_MIN_N} held-out {_s} variants", inputs=[config.YANG_HELDOUT_CSV])
            _bad[_s] = sorted(_yh.index[_ok & (_yh[f"yang_val_{_s}"] >= _thr)])
        REG.check("fit-heldout-all-below", not any(_bad.values()),
                  f"held-out distances >= {_thr} (samples with enough held-out variants): {_bad}",
                  "main.tex Results/fit quality (held-out sentence)")
    _nplp_col = "n_plp_fit" if "n_plp_fit" in _y.columns else "n_plp"
    if _poor and _nplp_col in _y.columns:
        REG.num("ResFitPoorFitPLPBound", int(pd.to_numeric(_y.loc[_poor, _nplp_col]).max()) + 1,
                source=f"max {_nplp_col} among poor-fit datasets + 1 (text: n < bound)")

# %% [markdown]
# ### A3c. Yang distance 4-panel diagnostic figure (Extended Data Figure "fig:dists")
#
# Unlike the table above (pre-reduced medians from a batch-computed CSV),
# this recomputes the full per-bootstrap Yang-distance distribution live for
# just the same 4 datasets used in A3a's Figure "fig:fits" (BRCA1/LDLR/
# PTEN/CRX) -- the manuscript's own `fig:dists` shows exactly this scope, not
# all datasets, so recomputing the other datasets' full distributions
# here would be wasted cost. Reuses `_four_panel_loader` from A3a so
# both figures are guaranteed to be built from the identical underlying fit.


# %%
if _missing_four_panel:
    print(f"  SKIP Yang distance diagnostic figure: {_missing_four_panel} not in section 1's discovered datasets")
else:
    from analysis.calibration_plots import plot_yang_distance_diagnostic

    # Cached per dataset (config.YANG_CACHE_DIR): only datasets whose
    # calibration changed since their last computation are recomputed.
    plot_yang_distance_diagnostic(
        FOUR_PANEL_DATASETS, _four_panel_loader, FIGURE_SUBDIRS["manuscript"],
        good_fit_threshold=config.YANG_GOOD_FIT_THRESHOLD, n_jobs=config.YANG_N_JOBS,
        cache_dir=config.YANG_CACHE_DIR, output_dir=OUTPUT_DIR, n_grid=config.YANG_N_GRID,
    )
REG.num("MethYangNGrid", config.YANG_N_GRID, kind="method", source="config.YANG_N_GRID (CDF grid points)")
REG.figure("fig:dists", FIGURE_SUBDIRS["manuscript"] / "yang_distance_diagnostic.png", inputs=[PRECOMPUTED_FITS])

# %% [markdown]
# ## A4. Figure 3 (evidence distributions) and Extended Data Tables 1-2 (gene-wise evidence)

# %% [markdown]
# ### A4a. Combined author + ClinVar evidence distribution (Fig. 3), and gene-wise evidence tables (ED Tables 1-2)
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

# %%
REG.figure("fig:author_clinvar_compare",
           FIGURE_SUBDIRS["manuscript"] / f"combined_evidence_distribution_{primary_method}.png")
_gwe_tabs = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", gwe_latex_str, flags=re.S)
REG.table("tab:gene_evidence_distr_auth", _gwe_tabs[0], data_row_label_re=r"\\textit\{")
REG.table("tab:gene_evidence_distr_clinvar", _gwe_tabs[1], data_row_label_re=r"\\textit\{")

# Results "Evidence distributions are concordant ..." percentages: same
# tables as ED Tables 1-2, so prose and tables agree by construction.
_ga, _gc = gwe_author_table, gwe_clinvar_table
REG.num("ResAuthAbnormalPathPct", _ga.loc["All", ("Abnormal", "Pathogenic (>0)", "%")], fmt=".0f",
        source="ED Table 1 All row: Abnormal -> pathogenic %")
REG.num("ResAuthNormalBenPct", _ga.loc["All", ("Normal", "Benign (<0)", "%")], fmt=".0f",
        source="ED Table 1 All row: Normal -> benign %")
_abn = np.asarray(all_author) == 2
_pts_auth = np.asarray(all_danz_oob, dtype=float)
REG.pct("ResAuthAbnormalStrongPct", int(((_pts_auth >= 4) & _abn).sum()), int(_abn.sum()), digits=0, words=True,
        source="author Abnormal variants with OOB points >= +4")
REG.pct("ResAuthAbnormalVeryStrongPct", int(((_pts_auth >= 8) & _abn).sum()), int(_abn.sum()), digits=0,
        source="author Abnormal variants with OOB points >= +8")
for _grp, _tag in (("P/LP", "PLP"), ("B/LB", "BLB"), ("VUS", "VUS")):
    REG.num(f"ResClinvarNum{_tag}", _gc.loc["All", (_grp, "Total", "n")], source=f"ED Table 2 All row: {_grp} n")
    REG.num(f"ResClinvar{_tag}PathPct", _gc.loc["All", (_grp, "Pathogenic (>0)", "%")], fmt=".0f",
            source=f"ED Table 2 All row: {_grp} pathogenic %")
    REG.num(f"ResClinvar{_tag}IndPct", _gc.loc["All", (_grp, "Indeterminate (0)", "%")], fmt=".0f",
            source=f"ED Table 2 All row: {_grp} indeterminate %")
    REG.num(f"ResClinvar{_tag}BenPct", _gc.loc["All", (_grp, "Benign (<0)", "%")], fmt=".0f",
            source=f"ED Table 2 All row: {_grp} benign %")
REG.num("ResClinvarVUSDetPct", 100 - _gc.loc["All", ("VUS", "Indeterminate (0)", "%")], fmt=".0f", words=True,
        source="ED Table 2 All row: VUS determinate %")
# Results: asymmetry of misassigned controls (same OOB arrays as ED Table 2;
# label columns 0 = B/LB, 2 = P/LP), and how many sit at the weakest level.
_pts_cv = np.asarray(all_danz_oob_full, dtype=float)
_lab_cv = np.asarray(all_clinvar_full)[:, :4].astype(bool)
_fn = _lab_cv[:, 2] & (_pts_cv < 0)
_fp = _lab_cv[:, 0] & (_pts_cv > 0)
REG.num("ResClinvarNumPLPBen", int(_fn.sum()), source="P/LP measurements with benign OOB evidence")
REG.num("ResClinvarNumPLPBenMinusOne", int((_fn & (_pts_cv == -1)).sum()), source="... of which at exactly -1")
REG.pct("ResClinvarPLPBenMinusOnePct", int((_fn & (_pts_cv == -1)).sum()), int(_fn.sum()), digits=0,
        source="% of P/LP benign assignments at -1")
REG.num("ResClinvarNumBLBPath", int(_fp.sum()), source="B/LB measurements with pathogenic OOB evidence")
REG.num("ResClinvarNumBLBPathPlusOne", int((_fp & (_pts_cv == 1)).sum()), source="... of which at exactly +1")
REG.pct("ResClinvarBLBPathPlusOnePct", int((_fp & (_pts_cv == 1)).sum()), int(_fp.sum()), digits=0,
        source="% of B/LB pathogenic assignments at +1")
REG.check("clinvar-fn-exceeds-fp", int(_fn.sum()) > int(_fp.sum()) and int((_fn & (_pts_cv == -1)).sum()) > int((_fp & (_pts_cv == 1)).sum()),
          f"P/LP->benign {int(_fn.sum())} (at -1: {int((_fn & (_pts_cv == -1)).sum())}) vs "
          f"B/LB->pathogenic {int(_fp.sum())} (at +1: {int((_fp & (_pts_cv == 1)).sum())})",
          "main.tex Results (asymmetric misassignments)")

# %% [markdown]
# ## A5. Figure 4 (ExCALIBR vs. author) and Extended Data Tables 3-4 (performance)
#
# Figure 4 is the ExCALIBR-vs-author confusion figure; ED Table 3 adds the
# both-determinate comparison requested in review (A5b); ED Table 4 is the
# gene-deduplicated version (A1c's matrices).

# %%
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
if any(a is not None for a in auths):
    make_confusion_figure(
        danzs_m1=conf_by_method[primary_method], danzs_m2=auths,
        dataset_names=datasets, label1=primary_method, label2="author",
        figure_dir=FIGURE_SUBDIRS["author"], filename="excalibr_vs_author.png",
        vus_coverages_m1=vus_by_method[primary_method],
        vus_coverages_m2=auth_vus_by_method[primary_method],
    )
REG.figure("fig:confusion", FIGURE_SUBDIRS["author"] / "excalibr_vs_author.png")

# %% [markdown]
# ### A5b. ExCALIBR vs. author, both-determinate calls only (ED Table 3 right columns)
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
if any(m is not None for m in both_det_excalibr) and any(m is not None for m in both_det_author):
    make_confusion_figure(
        danzs_m1=both_det_excalibr, danzs_m2=both_det_author,
        dataset_names=datasets, label1=primary_method, label2="author",
        figure_dir=FIGURE_SUBDIRS["author"], filename="excalibr_vs_author_both_determinate.png",
    )
else:
    print("  SKIP ExCALIBR-vs-author both-determinate figure: no dataset had any determinate-determinate call")

# %% [markdown]
# ### A5c. Aggregate performance report + ED Tables 3-4
#
# **ExCALIBR vs. author** (unchanged): `print_aggregate_performance`
# (src/assay_calibration/plot_utils/utils.py) sums the confusion matrices
# above and prints the full text report (per dataset + aggregate accuracy/
# sensitivity/specificity/MCC/LR+/DOR for both ExCALIBR and author
# annotations). `latex_performance_table_clinvar` turns those same computed
# metrics into the manuscript's LaTeX table — every number in it comes from
# the confusion matrices built above, nothing hardcoded. Also now includes
# the both-determinate variant (section 3a1b's `both_det_excalibr`/
# `both_det_author`, previously only a confusion figure) as its own table.
#
# **ExCALIBR vs. every other comparison method**: `latex_performance_table_multi`
# (analysis/manuscript_stats.py) generalizes that same dynamic-table approach
# to an arbitrary number of methods (not just one "Author Annotations"
# column), via `compute_aggregate_metrics_multi` (pools each method's
# confusion matrices over the intersection of datasets where every method in
# the table produced one). Three grouped tables, matching how these methods
# are actually meant to be compared:
#   - ExCALIBR vs. acmgscaler vs. ExCALIBR with prior manually fixed at 0.1
#     (`manual_prior_conf_raw`, section 3a2) -- isolates whether acmgscaler's
#     disagreement with ExCALIBR tracks the prior choice itself.
#   - ExCALIBR vs. skew-locked vs. the pathomechanism-aware prior mode
#     (section 3a3 / 3a4b).
#   - ExCALIBR vs. all four GMM baseline pooling variants together
#     (`config.GMM_BASELINE_VARIANTS`, section 3a2).
# Every table's "Datasets calibratable" row shows each method's own
# non-None-matrix count out of the full dataset count -- not just the
# intersection size the metrics below it are pooled over (stated in the
# caption) -- so it's visible per-method whether/how much a method's own
# calibratable subset differs from the others'.


# %%
from src.assay_calibration.plot_utils.utils import print_aggregate_performance
from analysis.manuscript_stats import latex_performance_table_clinvar

_auth_pairs = [
    (d, a, n) for d, a, n in zip(conf_by_method[primary_method], auths, datasets)
    if d is not None and a is not None
]
danz_agg_metrics = auth_agg_metrics = None
if _auth_pairs:
    _danzs_auth, _auths_auth, _names_auth = zip(*_auth_pairs)
    danz_agg_metrics, auth_agg_metrics, individual_metrics_df = print_aggregate_performance(
        list(_danzs_auth), list(_auths_auth), list(_names_auth),
    )
else:
    print("  SKIP aggregate performance report: no datasets with both ExCALIBR and author matrices")

# Same aggregate report, restricted to the both-determinate subset (section
# 3a1b's both_det_excalibr/both_det_author -- variants where BOTH ExCALIBR
# and the author actually made a determinate call; the IR column is
# all-zero on both sides by construction).
_auth_pairs_both_det = [
    (d, a, n) for d, a, n in zip(both_det_excalibr, both_det_author, datasets)
    if d is not None and a is not None
]
danz_agg_both_det = auth_agg_both_det = None
if _auth_pairs_both_det:
    _danzs_bd, _auths_bd, _names_bd = zip(*_auth_pairs_both_det)
    danz_agg_both_det, auth_agg_both_det, _ = print_aggregate_performance(
        list(_danzs_bd), list(_auths_bd), list(_names_bd),
    )
else:
    print("  SKIP aggregate performance report (both-determinate): no dataset had any determinate-determinate call")

# Merged table: "All variants" + "Both determinate" side by side (one
# ExCALIBR/Author column pair each) instead of two separate 2-column
# tables/files.
_clinvar_perf_groups = []
if danz_agg_metrics is not None:
    _clinvar_perf_groups.append(("All variants", danz_agg_metrics, auth_agg_metrics))
if danz_agg_both_det is not None:
    _clinvar_perf_groups.append(("Both determinate", danz_agg_both_det, auth_agg_both_det))
if _clinvar_perf_groups:
    _clinvar_perf_latex = latex_performance_table_clinvar(_clinvar_perf_groups)
    save_latex_table(_clinvar_perf_latex, FIGURE_SUBDIRS["tables"] / "author_clinvar_performance.tex")

# Same aggregate report + LaTeX table, gene-deduplicated -- reuses the
# deduped_matrix_for_author/deduped_author_matrix matrices built in 3a1c.
if deduped_matrix_for_author is not None and deduped_author_matrix is not None:
    danz_agg_deduped, auth_agg_deduped, _ = print_aggregate_performance(
        [deduped_matrix_for_author], [deduped_author_matrix], ["gene_deduped"],
    )
    _latex_deduped = latex_performance_table_clinvar([("All variants", danz_agg_deduped, auth_agg_deduped)])
    save_latex_table(_latex_deduped, FIGURE_SUBDIRS["tables"] / "author_clinvar_performance_gene_deduped.tex")
else:
    print("  SKIP aggregate performance report (gene-deduped): no matrix on one or both sides")

# %%
REG.table("tab:author_performance", _clinvar_perf_latex)
if deduped_matrix_for_author is not None and deduped_author_matrix is not None:
    REG.table("tab:author_performance_variantdedup", _latex_deduped)
    REG.num("ResDedupDORExcalibr", danz_agg_deduped["dor_standard"], fmt=",.1f",
            source="ED Table 4: ExCALIBR DOR (P vs. B), each variant counted once")
    REG.num("ResDedupDORAuthor", auth_agg_deduped["dor_standard"], fmt=",.1f",
            source="ED Table 4: author DOR (P vs. B), each variant counted once")
    REG.num("ResDedupDetExcalibrPct", 100 * danz_agg_deduped["coverage"], fmt=".1f",
            source="ED Table 4: ExCALIBR determinate %, each variant counted once")
    REG.num("ResDedupDetAuthorPct", 100 * auth_agg_deduped["coverage"], fmt=".1f",
            source="ED Table 4: author determinate %, each variant counted once")
    # Cross-assay conflict rates (ED Table 4 scope: ClinVar P/LP and B/LB in
    # genes with author annotations, variants measured by >= 2 assays): share
    # whose per-assay determinate calls disagree in direction, over all such
    # variants and over those with >= 2 determinate calls from that method.
    from analysis.plot_common import sample_matches as _sm
    _dv = deduped_variants_with_author
    _dv = _dv[(_sm(_dv, "Pathogenic/Likely Pathogenic") | _sm(_dv, "Benign/Likely Benign")) & (_dv["n_assays"] >= 2)]
    REG.num("ResDedupMultiAssayN", len(_dv), source="ED Table 4 scope: P/LP+B/LB variants measured by >= 2 assays")
    for _m, _tag in (("points", "Excalibr"), ("auth", "Author")):
        _conf = _dv[f"{_m}_conflict"].astype(bool)
        _two = _dv[f"{_m}_n_det"] >= 2
        REG.pct(f"ResDedupConflict{_tag}Pct", int(_conf.sum()), len(_dv), digits=1,
                source=f"{_tag}: conflicting direction across assays / multi-assay variants")
        REG.pct(f"ResDedupConflict{_tag}PctOfTwoDet", int(_conf.sum()), int(_two.sum()), digits=1,
                source=f"{_tag}: conflicting / multi-assay variants with >= 2 determinate calls")
        print(f"  {_tag}: {int(_conf.sum())} conflicting of {len(_dv)} multi-assay variants "
              f"({int(_two.sum())} with >= 2 determinate calls)")
    REG.check("dedup-det-excalibr-higher", danz_agg_deduped["coverage"] > auth_agg_deduped["coverage"],
              f"dedup determinate {danz_agg_deduped['coverage']:.3f} vs author {auth_agg_deduped['coverage']:.3f}",
              "main.tex Results (counted-once sentence)")
# Methods: on both-determinate variants ExCALIBR is better on every metric.
if danz_agg_both_det is not None:
    _hi = ["accuracy", "sensitivity", "specificity", "mcc", "ppv", "lr_plus_standard", "lr_plus_pathogenic",
           "lr_plus_benign", "dor_standard", "dor_pathogenic", "dor_benign"]
    _worse = [k for k in _hi if not danz_agg_both_det[k] >= auth_agg_both_det[k]]
    if not danz_agg_both_det["fdr"] <= auth_agg_both_det["fdr"]:
        _worse.append("fdr")
    REG.check("both-det-every-metric", not _worse,
              f"ExCALIBR not better on both-determinate: {_worse}" if _worse else "ExCALIBR >= author on every metric",
              "main.tex Results (both-determinate sentence)")
# Results numbers: tied to ED Table 3's "All variants" columns.
_dm, _am = danz_agg_metrics, auth_agg_metrics
REG.num("NumAuthorAnnotatedDatasets", len(datasets_with_author), source="datasets with determinate author calls")
REG.num("ResPerfAccExcalibrPct", 100 * _dm["accuracy"], fmt=".1f", source="ED Table 3 All: ExCALIBR accuracy")
REG.num("ResPerfAccAuthorPct", 100 * _am["accuracy"], fmt=".1f", source="ED Table 3 All: author accuracy")
REG.num("ResPerfDORExcalibr", _dm["dor_standard"], fmt=".1f", source="ED Table 3 All: ExCALIBR DOR (P vs. B)")
REG.num("ResPerfDORAuthor", _am["dor_standard"], fmt=".1f", source="ED Table 3 All: author DOR (P vs. B)")
REG.num("ResPerfDetExcalibrPct", 100 * _dm["coverage"], fmt=".1f", source="ED Table 3 All: ExCALIBR determinate %")
REG.num("ResPerfDetAuthorPct", 100 * _am["coverage"], fmt=".1f", source="ED Table 3 All: author determinate %")
for _k, _tag in (("lr_plus_pathogenic", "LRPath"), ("dor_pathogenic", "DORPath"),
                 ("lr_plus_benign", "LRBen"), ("dor_benign", "DORBen")):
    REG.num(f"ResPerf{_tag}Excalibr", _dm[_k], fmt=".1f", source=f"ED Table 3 All: ExCALIBR {_k}")
    REG.num(f"ResPerf{_tag}Author", _am[_k], fmt=".1f", source=f"ED Table 3 All: author {_k}")
_better = {k: _dm[k] > _am[k] for k in ("accuracy", "sensitivity", "specificity", "mcc", "lr_plus_standard",
                                         "lr_plus_pathogenic", "lr_plus_benign", "dor_standard", "dor_pathogenic",
                                         "dor_benign")}
REG.check("clinvar-outperforms-all-metrics", all(_better.values()),
          f"ExCALIBR > author on: {_better}", "main.tex Results/accuracy")
REG.check("clinvar-lower-determinate", _dm["coverage"] < _am["coverage"],
          f"determinate {_dm['coverage']:.3f} vs {_am['coverage']:.3f}", "main.tex Results/accuracy")

# %% [markdown]
# ## A6. Figure 5 (gene accuracy, datasets reaching evidence, All of Us ORs) and Extended Data Figure 2
#
# Panel B needs the OddsPath (Brnich et al.) side; panels C and ED Fig. 2 need
# the All of Us odds ratios (`config.AOU_OR_ESTIMATES_TSV`, collaborator file).

# %%
from analysis.assay_stats import (
    compute_excalibr_evidence_counts, plot_evidence_comparison,
    compute_assay_evidence_stats, print_assay_evidence_report,
    plot_dataset_point_heatmap,
)

# %%
excalibr_path_counts, excalibr_ben_counts = compute_excalibr_evidence_counts(
    dataset_info_df_full, all_danz_oob_full,
)
print(f"ExCALIBR datasets reaching pathogenic evidence ±X: {excalibr_path_counts}")
print(f"ExCALIBR datasets reaching benign evidence ±X: {excalibr_ben_counts}")

# Cache for section 7's build_gene_performance_scatter (its combined
# gene_brnich_OR figure needs this same {excalibr,auth}_{path,ben}_counts
# shape) -- avoids relying on the legacy datasets_reached_evidence/*.pkl
# files, which no longer exist on disk. Only populated below if
# OP_EVIDENCE_CODES_CSV is present (auth_path_counts/auth_ben_counts need it).
evidence_counts_cache = None

if not config.warn_if_missing(config.OP_EVIDENCE_CODES_CSV, "OddsPath evidence-code CSV (author side)"):
    from src.assay_calibration.fit_utils.evidence_thresholds import get_tavtigian_constant

    from analysis.supplementary_data import oddspath_resolution, load_oddspath_csv

    df_op = load_oddspath_csv()  # config.OP_SUPERSEDED_BY applied (MSH2_Scott_2022 -> MSH2_Jia_2021)
    levels = [1, 2, 4, 8]

    # Tiers are re-derived here from the raw OddsAbnormal/OddsNormal values
    # with the Tavtigian-constant thresholds (C**(level/8), fixed prior
    # FIXED_PRIOR, matching ExCALIBR's own thresholds_from_prior) rather than
    # read from the CSV's own "Evidence Code" columns, so both sides of
    # Fig. 5B use identical cutoffs. OddsAbnormal is a direct odds of
    # pathogenicity (higher = stronger PS3); OddsNormal lies in [0, 1] (lower
    # = stronger BS3), so its thresholds are the reciprocal.
    _C = get_tavtigian_constant(FIXED_PRIOR)
    _path_thresholds = {lvl: _C ** (lvl / 8) for lvl in levels}
    _ben_thresholds = {lvl: 1.0 / (_C ** (lvl / 8)) for lvl in levels}

    # The OddsPath CSV names datasets without the "_clinvar_2018" suffix (and
    # by some older names); resolve exactly as \NumDatasetsBrnichNA and
    # Supplementary Data 3 do, so all three agree.
    _op_names = oddspath_resolution(list(dataset_info_df_full["dataset"]))
    auth_path_counts = {lvl: set() for lvl in levels}
    auth_ben_counts = {lvl: set() for lvl in levels}
    for ds in dataset_info_df_full["dataset"]:
        op_rows = df_op[df_op.Dataset == _op_names.get(ds)]
        if op_rows.empty:
            continue
        odds_abnormal = pd.to_numeric(op_rows["OddsAbnormal"].iloc[0], errors="coerce")
        odds_normal = pd.to_numeric(op_rows["OddsNormal"].iloc[0], errors="coerce")

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
    evidence_counts_cache = {
        "excalibr_path_counts": excalibr_path_counts,
        "excalibr_ben_counts": excalibr_ben_counts,
        "auth_path_counts": auth_path_counts,
        "auth_ben_counts": auth_ben_counts,
    }
    fig, _ = plot_evidence_comparison(
        excalibr_path_counts, excalibr_ben_counts, auth_path_counts, auth_ben_counts,
    )
    save_and_show(fig, FIGURE_SUBDIRS["manuscript"] / "num_datasets_reach_evidence.png")

# %%
from analysis import gene_performance_scatter

# Flip these and rerun to generate per-consequence and/or all-datasets-per-
# gene variants of the OR figures/table (see
# gene_performance_scatter.build_gene_performance_scatter's docstring).
OR_CONSEQUENCE = "All"
OR_DATASET_MODE = "primary"
OR_SHOW_CARRIER_COUNTS = False  # carrier-case counts in grey above each point; off: low-carrier groups are drawn dashed instead
OR_ALSO_ALL_DATASETS = True     # also save supp_OR_all_datasets.png (every dataset per gene)

gene_perf_result = gene_performance_scatter.build_gene_performance_scatter(
    output_dir=OUTPUT_DIR, figure_dir=FIGURE_SUBDIRS["manuscript"],
    evidence_counts=evidence_counts_cache,
    consequence=OR_CONSEQUENCE, dataset_mode=OR_DATASET_MODE,
    show_carrier_counts=OR_SHOW_CARRIER_COUNTS, also_all_datasets_figure=OR_ALSO_ALL_DATASETS,
    # A1b's own matrices, instead of a second discover/load pass.
    confusion_data=(datasets, conf_by_method[primary_method], auths),
)

# %%
REG.require(config.OP_EVIDENCE_CODES_CSV, "OddsPath evidence CSV (Fig. 5B, Supp Data 3)",
            newer_than=[config.DATASET_TSV], regen="python -m analysis.oddspath_computation.compute_oddspath")
REG.require(config.AOU_OR_ESTIMATES_TSV, "All of Us odds-ratio estimates (Fig. 5C, ED Fig. 2; collaborator)")
REG.require(config.AOU_TP53_OR_ESTIMATES_TSV, "All of Us TP53 odds-ratio estimates (Fig. 5C, ED Fig. 2; collaborator)")
_AOU_INPUTS = [config.AOU_OR_ESTIMATES_TSV, config.AOU_TP53_OR_ESTIMATES_TSV]
REG.figure("fig:gene_brnich_OR", FIGURE_SUBDIRS["manuscript"] / "gene_brnich_OR.png",
           inputs=[config.OP_EVIDENCE_CODES_CSV, *_AOU_INPUTS])
REG.figure("fig:all_OR", FIGURE_SUBDIRS["manuscript"] / "supp_OR.png", inputs=_AOU_INPUTS)

from analysis.supplementary_data import oddspath_resolution, load_oddspath_csv
_op_resolved = oddspath_resolution(datasets)
oddspath_gap_datasets = sorted(d for d, r in _op_resolved.items() if r is None)
REG.num("NumDatasetsBrnichNA", len(oddspath_gap_datasets), source="calibrated datasets absent from the OddsPath CSV",
        inputs=[config.OP_EVIDENCE_CODES_CSV])
_op_msh2 = load_oddspath_csv()
_op_msh2 = _op_msh2[_op_msh2["Dataset"].str.startswith("MSH2")]
# One MSH2 row (Scott et al. 2022 values under the calibrated MSH2_Jia_2021
# name), and the calibrated MSH2 dataset resolves to it.
REG.check("oddspath-msh2-once", len(_op_msh2) == 1
          and "MSH2_Scott_2022" in str(_op_msh2["Pseudocount Details"].iloc[0])
          and [r for d, r in _op_resolved.items() if d.startswith("MSH2")] == ["MSH2_Jia_2021"],
          f"MSH2 OddsPath rows after OP_SUPERSEDED_BY: {len(_op_msh2)} (one, with the Scott et al. 2022 values)",
          "Fig. 5B / Supp Data 3")

_gr = gene_perf_result.get("gene_results", [])
_gr_det = [r for r in _gr if r["danz_accuracy"] not in (0, float("inf")) and r["auth_accuracy"] not in (0, float("inf"))]
REG.num("NumAuthorAnnotatedGenes", len(_gr), source="Fig. 5A genes (ExCALIBR and author matrices)")
REG.num("NumAuthorAnnotatedGenesDeterminate", len(_gr_det), source="Fig. 5A genes with non-zero accuracy")
_n_improved = sum(r["danz_accuracy"] > r["auth_accuracy"] for r in _gr_det)
REG.num("ResGeneAccNumImproved", _n_improved, source="determinate genes with ExCALIBR accuracy > author")
REG.pct("ResGeneAccImprovedPct", _n_improved, len(_gr_det), digits=1, source="ResGeneAccNumImproved / determinate genes")
_indet = sorted([r for r in _gr if r["danz_accuracy"] == 0], key=lambda r: -r["control_count"])
REG.num("ResGeneAccNumAllIndet", len(_indet), words=True, source="Fig. 5A genes with all-indeterminate ExCALIBR calls")
_join = lambda xs: " and ".join(xs) if len(xs) <= 2 else ", ".join(xs[:-1]) + ", and " + xs[-1]
REG.text("ResGeneAccAllIndetGenes", _join([f"\\textit{{{r['gene']}}}" for r in _indet]), source="genes, by control count")
REG.text("ResGeneAccAllIndetControls", _join([f"{int(r['control_count'])}" for r in _indet]), source="control variants")
REG.text("ResGeneAccAllIndetAuthAcc", _join([f"{r['auth_accuracy']:.3f}" for r in _indet]), source="author accuracies")

from analysis.gene_performance_scatter import _has_carriers
_or = gene_perf_result.get("df_display", pd.DataFrame())
if not _or.empty:
    _path_rows = _or[_or["Classification"].astype(str).str.startswith("≥")
                     & _or["Cases with variants"].apply(_has_carriers) & (_or["OR_LI"] > 1)]
    REG.num("ResAoUNumSigGenes", _path_rows["Gene"].nunique(), words=True,
            source="genes with a significant pathogenic bin (OR 95% CI > 1)", inputs=_AOU_INPUTS)
    # Results: the gene-disease pairs without a significant association (same
    # primary-dataset scope) and whether every pathogenic group of theirs has
    # <= LOW_CARRIER_MAX carrier cases or controls.
    _gps6 = gene_performance_scatter
    _ns_genes = [g for g in _gps6.order_genes_by_disease(_or["Gene"].unique())
                 if g not in set(_path_rows["Gene"])]
    REG.num("ResAoUNumNonsigGenes", len(_ns_genes), words=True, source="AoU gene-disease pairs without a significant pathogenic bin")
    REG.text("ResAoUNonsigGenes", _join([f"\\textit{{{g}}}" for g in _ns_genes]) if _ns_genes else "none",
             source="genes without a significant pathogenic bin")
    _ns_p = _or[_or["Gene"].isin(_ns_genes) & _or["Classification"].astype(str).str.startswith("≥")
                & _or["Cases with variants"].apply(_has_carriers)]
    REG.check("aou-nonsig-genes-all-low-carrier", bool(len(_ns_p)) and bool(_ns_p.apply(_gps6.is_low_carrier, axis=1).all()),
              f"non-significant genes {_ns_genes}: {int(_ns_p.apply(_gps6.is_low_carrier, axis=1).sum())}/{len(_ns_p)} pathogenic groups low-carrier",
              "main.tex Results (All of Us: non-significant pairs all low-carrier)")
    # ED Fig. 2 caption: the dataset shown for each gene, in the figure's order.
    _or_genes = gene_performance_scatter.order_genes_by_disease(_or["Gene"].unique())
    REG.text("ResAoUGeneDatasets", "; ".join(
        f"\\textit{{{g}}}, {gene_performance_scatter.dataset_short_label(gene_performance_scatter.GENE_TO_DATASET[g], g)}"
        for g in _or_genes if g in gene_performance_scatter.GENE_TO_DATASET),
        source="GENE_TO_DATASET, ED Fig. 2 order")
    REG.num("ResAoUNumGenes", _or["Gene"].nunique(), source="genes in the All of Us OR table (ExCALIBR grouping)",
            inputs=_AOU_INPUTS)
    # ED Fig. 2 is the all-datasets figure (supp_OR_all_datasets.png): one panel per dataset.
    _or_all = gene_performance_scatter.load_aou_or_estimates(consequence=OR_CONSEQUENCE)
    _or_entries = gene_performance_scatter.resolve_gene_dataset_entries(
        _or_all, gene_performance_scatter.GENE_TO_DATASET,
        gene_performance_scatter.order_genes_by_disease(_or_all["Gene"].unique()), "all")
    REG.num("ResAoUNumDatasetsOR", len(_or_entries), source="panels (datasets) in ED Fig. 2 (all-datasets OR figure)",
            inputs=_AOU_INPUTS)
    # Results: non-significant pathogenic groups in ED Fig. 2 (95% CI includes 1,
    # the figure's open circles), split by the low-carrier flag (dashed groups:
    # <= LOW_CARRIER_MAX carrier cases or controls).
    _gps = gene_performance_scatter
    _pg = _or_all[_or_all["Odds Ratio"].notna() & _or_all["Cases with variants"].apply(_gps._has_carriers)
                  & (_or_all["Classification"].map(_gps._get_strength) > 0)].copy()
    _pg["nonsig"] = ~((_pg["LogOR_LI"] > 0) | (_pg["LogOR_UI"] < 0))
    _pg["low"] = _pg.apply(_gps.is_low_carrier, axis=1)
    for _tag, _sel in (("Low", _pg["low"]), ("High", ~_pg["low"])):
        REG.num(f"ResAoU{_tag}N", int(_sel.sum()),
                source=f"ED Fig. 2 pathogenic groups, {'<=' if _tag == 'Low' else '>'} {_gps.LOW_CARRIER_MAX} carrier cases or controls")
        REG.num(f"ResAoUNonsig{_tag}N", int((_sel & _pg["nonsig"]).sum()), source="... of which non-significant")
    REG.check("aou-nonsig-mostly-low-carrier",
              (_pg["low"] & _pg["nonsig"]).sum() > (~_pg["low"] & _pg["nonsig"]).sum(),
              f"non-significant pathogenic groups: {int((_pg.low & _pg.nonsig).sum())}/{int(_pg.low.sum())} low-carrier vs "
              f"{int((~_pg.low & _pg.nonsig).sum())}/{int((~_pg.low).sum())} others; well-powered non-significant: "
              f"{sorted(set(_pg.loc[~_pg.low & _pg.nonsig, 'Dataset']))}",
              "main.tex Results (All of Us: non-significant groups mostly low-carrier)")

# %% [markdown]
# ## A7. Extended Data Figure 3 and Extended Data Table 5 (ClinGen Evidence Repository)

# %% [markdown]
# ### A7a. ClinGen expert-panel ground-truth confusion
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
    convert_3x2_to_2x3, plot_2x3_confusions_nature, plot_clingen_confusion_stacked,
)
from analysis.manuscript_stats import latex_performance_table_clingen

# datasets_with_author computed once, above in section 4b.
print(f"ClinGen scope: {len(datasets_with_author)}/{len(datasets)} datasets have real author data")

CLINGEN_DATASETS = datasets_with_author  # explicit, adjustable — narrow if desired

# One call to build_clingen_confusion per strip_functional_evidence value,
# not one per (gene-dedup x strip) combination -- gene-dedup is a pure
# post-processing step over already-extracted per-variant ClinGen evidence
# (build_gene_deduped_clingen_confusion works directly off `clingen_records`),
# so a second build_clingen_confusion call with the same strip value would
# just redo the exact same, expensive per-dataset work (this function
# reloads a fresh Scoreset from `df_ds`/PipelineConfig for *every* dataset in
# CLINGEN_DATASETS, not only the four _clinvar_2018-suffixed genes -- that's
# needed to align each variant's raw ClinGen evidence-code string, which
# isn't present in the already-loaded df_primary/df_variants at all) for a
# result it then throws away.
CLINGEN_STRIP_VARIANTS = [
    (True, "clingen_confusion", "clingen_confusion_gene_deduped", "PS3/BS3 stripped"),
    (False, "clingen_confusion_with_ps3bs3", "clingen_confusion_gene_deduped_with_ps3bs3", "PS3/BS3 kept"),
]

clingen_genes_default = None  # captured from the "clingen_confusion" (per-assay, PS3/BS3-stripped) variant below
clingen_records_by_strip = {}  # strip_functional_evidence -> per-variant records (Results numbers in A7)
# Non-gene-deduped PS3/BS3-stripped/kept conf_dicts, collected here and fed
# into the combined 4-panel figure + merged table below instead of each
# getting its own separate figure/table.
_clingen_nondedup_conf_dicts = {}

for strip_ps3bs3, tag, dedup_tag, strip_desc in CLINGEN_STRIP_VARIANTS:
    print(f"\n--- ClinGen confusion ({strip_desc}) ---")
    import analysis.clingen as _clingen_mod
    (clingen_confusion, clingen_genes, clingen_records), _ = cached(
        f"clingen_confusion_strip{int(strip_ps3bs3)}",
        lambda: build_clingen_confusion(
            df_primary, DATASET_TSV, CLINGEN_DATASETS, use_oob=True,
            tree=tree, model_selections=model_selections, calibrations=calibrations,
            strip_functional_evidence=strip_ps3bs3,
        ),
        inputs=CANONICAL_INPUTS, code=[_clingen_mod], key=sorted(CLINGEN_DATASETS),
    )
    clingen_records_by_strip[strip_ps3bs3] = clingen_records
    if strip_ps3bs3:
        clingen_genes_default = clingen_genes

    if not clingen_genes:
        print(f"  SKIP ClinGen confusion ({strip_desc}): no genes with usable ClinGen evidence codes in scope")
    else:
        conf_dict = {
            'excalibr': convert_3x2_to_2x3(clingen_confusion['excalibr']),
            'auth': convert_3x2_to_2x3(clingen_confusion['auth']),
        }
        _clingen_nondedup_conf_dicts[tag] = conf_dict

    # Gene-deduplicated variant, reusing the same `clingen_records` instead
    # of a second build_clingen_confusion call (see comment above).
    dedup_confusion, dedup_genes = build_gene_deduped_clingen_confusion(clingen_records)
    if not dedup_genes:
        print(f"  SKIP ClinGen confusion (gene-deduped, {strip_desc}): no genes with usable ClinGen evidence codes in scope")
        continue
    dedup_conf_dict = {
        'excalibr': convert_3x2_to_2x3(dedup_confusion['excalibr']),
        'auth': convert_3x2_to_2x3(dedup_confusion['auth']),
    }
    fig = plot_2x3_confusions_nature(dedup_conf_dict)
    save_and_show(fig, FIGURE_SUBDIRS["clingen"] / f"{dedup_tag}.png")
    _clingen_perf_latex = latex_performance_table_clingen([("ClinGen", dedup_conf_dict)])
    save_latex_table(_clingen_perf_latex, FIGURE_SUBDIRS["tables"] / f"{dedup_tag}_performance.tex")

# Combined (non-gene-deduped) PS3/BS3-excluded + PS3/BS3-included figure and
# table -- replaces the two separate 2-panel figures/2-column tables these
# used to get with one 4-panel figure (A/B = PS3/BS3 excluded, C/D = PS3/BS3
# included) and one merged 4-column table.
if "clingen_confusion" in _clingen_nondedup_conf_dicts and "clingen_confusion_with_ps3bs3" in _clingen_nondedup_conf_dicts:
    _conf_stripped = _clingen_nondedup_conf_dicts["clingen_confusion"]
    _conf_kept = _clingen_nondedup_conf_dicts["clingen_confusion_with_ps3bs3"]
    fig = plot_clingen_confusion_stacked(_conf_stripped, _conf_kept)
    save_and_show(fig, FIGURE_SUBDIRS["clingen"] / "clingen_confusion_combined.png")
    _clingen_perf_latex = latex_performance_table_clingen([
        ("PS3/BS3 excluded", _conf_stripped),
        ("PS3/BS3 included", _conf_kept),
    ])
    save_latex_table(_clingen_perf_latex, FIGURE_SUBDIRS["tables"] / "clingen_confusion_performance.tex")
else:
    print("  SKIP combined ClinGen confusion figure/table: one or both non-gene-deduped variants unavailable")

# %%
REG.figure("fig:confusion_clingen", FIGURE_SUBDIRS["clingen"] / "clingen_confusion_combined.png")
if "clingen_confusion" in _clingen_nondedup_conf_dicts and "clingen_confusion_with_ps3bs3" in _clingen_nondedup_conf_dicts:
    # _clingen_perf_latex is the combined (excluded + included) table here.
    REG.table("tab:clingen_performance", _clingen_perf_latex)
REG.num("NumAuthorAnnotatedGenesClinGen", len(clingen_genes_default) if clingen_genes_default else None,
        source="genes in the PS3/BS3-stripped ClinGen comparison")
_cg = np.asarray(_clingen_nondedup_conf_dicts["clingen_confusion"]["excalibr"])  # rows BLB, PLP; cols B, I, P
REG.num("ResClinGenNumPLP", _cg[1].sum(), source="ED Table 5 (PS3/BS3 excluded): P/LP measurements")
REG.num("ResClinGenNumBLB", _cg[0].sum(), source="ED Table 5 (PS3/BS3 excluded): B/LB measurements")
REG.num("ResClinGenNumBLBPath", int(_cg[0, 2]), source="ED Table 5 (PS3/BS3 excluded): B/LB measurements with pathogenic evidence")
_rec = clingen_records_by_strip[True]
_plp_ben = _rec[(_rec["clingen_label"] == 1) & (_rec["points"] < 0)]
REG.num("ResClinGenNumPLPBen", len(_plp_ben), source="ClinGen P/LP with benign ExCALIBR evidence (PS3/BS3 stripped)")
_join = lambda xs: " and ".join(xs) if len(xs) <= 2 else ", ".join(xs[:-1]) + ", and " + xs[-1]  # also in A6; repeated so A7a runs alone
REG.text("ResClinGenPLPBenGenes", _join([f"\\textit{{{g}}}" for g in sorted(_plp_ben["gene"].unique())]),
         source="genes of ClinGen P/LP with benign evidence")
REG.num("ResClinGenPLPBenSupporting", int((_plp_ben["points"] == -1).sum()), source="... at -1")
REG.num("ResClinGenPLPBenModerate", int(_plp_ben["points"].between(-3, -2).sum()), source="... at -2/-3")
REG.check("clingen-plp-ben-weak", bool((_plp_ben["points"] > -4).all()),
          f"ClinGen P/LP benign points: {sorted(_plp_ben['points'].tolist())}", "main.tex Results (ClinGen)")

# %% [markdown]
# ## A8. Figure 6 (per-assay evidence heatmap) and VAMP-seq / SGE numbers

# %%
from analysis.assay_stats import (  # also imported in A6; repeated so A8 runs on its own
    compute_assay_evidence_stats, print_assay_evidence_report, plot_dataset_point_heatmap,
)

if not config.warn_if_missing(config.ASSAY_METHOD_MAP_CSV, "assay method map (dataset_stats/point heatmap)"):
    assay_method_map = pd.read_csv(config.ASSAY_METHOD_MAP_CSV)
    dataset_stats, summary_dict = compute_assay_evidence_stats(
        dataset_info_df_full, all_danz_oob_full, assay_method_map,
    )
    print_assay_evidence_report(dataset_stats, summary_dict)

    fig, ax, _, _ = plot_dataset_point_heatmap(
        dataset_info_df_full, all_danz_oob_full,
        assay_method_map=assay_method_map, sort_by='assay_type',
        igvf_image_path=config.IGVF_LOGO_PNG if Path(config.IGVF_LOGO_PNG).exists() else None,
    )
    save_and_show(fig, FIGURE_SUBDIRS["manuscript"] / "points_heatmap_sort_assay_type.png")

# %%
REG.require(config.ASSAY_METHOD_MAP_CSV, "curated dataset metadata (assay types)")
REG.require(config.IGVF_LOGO_PNG, "IGVF logo (Fig. 6 IGVF-dataset marker)")
REG.figure("fig:dataset_heatmap", FIGURE_SUBDIRS["manuscript"] / "points_heatmap_sort_assay_type.png",
           inputs=[config.ASSAY_METHOD_MAP_CSV])
_missing_type = sorted(set(datasets) - set(dataset_stats["dataset"]))
REG.check("heatmap-covers-datasets", not _missing_type, f"datasets without stats: {_missing_type}", "Fig. 6")
_at = summary_dict["assay_type"].set_index("assay_type")
for _type, _tag in (("VAMP-seq", "Vamp"), ("SGE", "Sge")):
    _sub = dataset_stats[dataset_stats["assay_type"] == _type]
    REG.num(f"Res{_tag}NumDatasets", len(_sub), words=True, source=f"{_type} datasets")
    for _col, _name in (("reaches_ge_plus2", "ReachPathMod"), ("reaches_le_minus2", "ReachBenMod"),
                        ("reaches_ge_plus4", "ReachPathStrong"), ("reaches_le_minus4", "ReachBenStrong")):
        REG.num(f"Res{_tag}Num{_name}", int(_sub[_col].sum()), words=True, source=f"{_type} datasets with {_col}")
    if _type in _at.index:
        REG.num(f"Res{_tag}PctPathMod", _at.loc[_type, "pct_ge_plus2_wmean"], fmt=".1f",
                source=f"{_type} variant-weighted % >= +2")
        REG.num(f"Res{_tag}PctBenMod", _at.loc[_type, "pct_le_minus2_wmean"], fmt=".1f",
                source=f"{_type} variant-weighted % <= -2")
# "all ten" when every VAMP-seq dataset reaches +2, else "nine of ten".
if "ResVampNumReachPathModWord" in REG.macros and "ResVampNumDatasetsWord" in REG.macros:
    _n_pm, _n_v = REG.macros["ResVampNumReachPathMod"].value, REG.macros["ResVampNumDatasets"].value
    REG.text("ResVampNumReachPathModOfAll",
             f"all {REG.macros['ResVampNumDatasetsWord'].rendered}" if _n_pm == _n_v
             else f"{REG.macros['ResVampNumReachPathModWord'].rendered} of {REG.macros['ResVampNumDatasetsWord'].rendered}",
             source="VAMP-seq datasets reaching +2, as 'all N' or 'X of N'")
if "ResVampNumReachBenStrongWord" in REG.macros and REG.macros["ResVampNumReachBenStrongWord"].value == 0:
    REG.text("ResVampNumReachBenStrongWord", "none", source="0 rendered as 'none'")
_unlabeled = sorted(set(datasets) - set(pd.read_csv(config.ASSAY_METHOD_MAP_CSV)["dataset"].dropna()))
REG.check("assay-types-curated", not _unlabeled, f"datasets without curated assay type: {_unlabeled}", "Fig. 6")

# %% [markdown]
# ## A9. Extended Data Figure 4: baselines and ablations (acmgscaler, fixed prior, GMM, skew-locked, pathomechanism)

# %% [markdown]
# ### A9a. ExCALIBR vs. other comparison methods
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
#   excalibr_prior_0.1 : the canonical pipeline rerun with the prior fixed
#                       at 0.1 instead of auto-fit
#                       (analysis.config.MANUAL_PRIOR_OUTPUT_DIR) -- a full
#                       ExCALIBR-shaped output tree, so discovered/loaded
#                       via analysis.discovery, and compared over every
#                       dataset (not just the acmgscaler-matched subset the
#                       3-way grid further down uses).
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
    "excalibr_prior_0.1": "manual_prior",
    "gmm_plp_blb": "gmm_baseline",
    "gmm_plp_blb_synon": "gmm_baseline",
    "gmm_all_plp_blb": "gmm_baseline",
    "gmm_all_plp_blb_synon": "gmm_baseline",
}

def _compare_vs_excalibr(conf_raw, method_label, vus_raw=None):
    """conf_raw : per-dataset confusion matrices (or None) for `method_label`,
    same order as `datasets`. Produces both the all_datasets and
    matched_datasets confusion figures against conf_by_method[primary_method],
    routed into the clinvar_comparisons/{acmgscaler,gmm_baseline,skew_locked}/
    subfolder matching `method_label` (see _COMPARISON_SUBDIR_KEY), and
    records the matched-datasets triple in `comparison_matches` for the
    aggregate performance report in 3b.

    `vus_raw`, if given: per-dataset (n_determinate, n_vus)-or-None VUS
    coverage for `method_label` itself (same order as `conf_raw`/`datasets`,
    built via analysis.confusion.build_vus_coverage -- e.g. with
    points_col="acmgscaler_points" for acmgscaler, or the default
    standard_points/oob_points path for gmm/skew_locked, which are already
    ExCALIBR-shaped). Paired with ExCALIBR's own vus_by_method[primary_method]
    (already computed once in section 3, reused here unchanged) so every
    comparison method's confusion figure shows a "Determinate: ExCALIBR X%,
    {method_label} Y%" caption the same way the ExCALIBR-vs-author panel
    does, via the same vus_coverages_m1/m2 mechanism -- no separate caption
    logic needed per method.
    """
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
            vus_coverages_m1=vus_by_method[primary_method] if vus_raw is not None else None,
            vus_coverages_m2=vus_raw,
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
            vus_coverages_m1=(
                [vus_by_method[primary_method][i] for i in matched_idx] if vus_raw is not None else None
            ),
            vus_coverages_m2=([vus_raw[i] for i in matched_idx] if vus_raw is not None else None),
        )
        comparison_matches[method_label] = (matched_excalibr, matched_other, matched_datasets)
    else:
        print(f"  SKIP {method_label} comparison (matched_datasets): no dataset produced a matrix")

# --- manual-prior ExCALIBR rerun (prior=0.1 fixed, not auto-fit) ---
# Loaded here (rather than down in 3a3 alongside skew-locked) so it's
# available for the 3-way ExCALIBR/acmgscaler/manual-prior grid right below.
# If analysis.config.MANUAL_PRIOR_OUTPUT_DIR doesn't exist, this cell just
# prints a warning and leaves manual_prior_conf_raw as None.
# Same full ExCALIBR-shaped output tree as OUTPUT_DIR/SKEW_LOCKED_OUTPUT_DIR,
# so discovered/loaded via analysis.discovery exactly like a normal
# pipeline run, not analysis.comparison_methods. use_oob=True: this rerun's
# *_variants.csv carries oob_points/oob_n_boots/oob_prior columns, same as
# the canonical pipeline's own output.
manual_prior_conf_raw = None
if not config.warn_if_missing(config.MANUAL_PRIOR_OUTPUT_DIR, "ExCALIBR manual-prior (0.1) comparison"):
    mp_tree, mp_model_selections, mp_calibrations = discover_outputs(Path(config.MANUAL_PRIOR_OUTPUT_DIR))
    mp_df = load_all_variants(
        tree=mp_tree, model_selections=mp_model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=datasets, calibrations=mp_calibrations, min_controls=0,
    )
    manual_prior_conf_raw = []
    manual_prior_vus_raw = []
    for dataset in datasets:
        df_mp = mp_df[mp_df["dataset"] == dataset] if not mp_df.empty else mp_df
        manual_prior_conf_raw.append(
            build_confusion_matrix(df_mp, use_oob=True, label=f"{dataset}/manual_prior_0.1")
            if not df_mp.empty else None
        )
        # No points_col override: this rerun is a full ExCALIBR output tree,
        # so the default standard_points/oob_points path applies unchanged --
        # same reason build_confusion_matrix above needs no special handling.
        manual_prior_vus_raw.append(
            build_vus_coverage(df_mp, use_oob=True, label=f"{dataset}/manual_prior_0.1")
            if not df_mp.empty else None
        )

    # Head-to-head ExCALIBR vs. ExCALIBR-with-prior-0.1 over EVERY dataset,
    # independent of the 3-way grid below (which is restricted to the
    # datasets acmgscaler could also calibrate). Same two-figure treatment
    # every other comparison method gets: all_datasets pads any dataset the
    # manual-prior rerun doesn't cover into the IR column so both panels
    # share a denominator, matched_datasets drops it. Unlike acmgscaler, a
    # missing dataset here means the rerun simply doesn't include it (not
    # that a prior of 0.1 was uncalibratable), so matched_datasets is the
    # more honest of the two whenever the counts below differ.
    _compare_vs_excalibr(manual_prior_conf_raw, "excalibr_prior_0.1", vus_raw=manual_prior_vus_raw)

# --- acmgscaler ---
# Precomputed by run_acmgscaler_all.py (analysis.config.ACMGSCALER_OUTPUT_DIR)
# for every dataset it could calibrate -- this is a disk read only, no live
# Rscript call. A missing CSV means acmgscaler genuinely couldn't calibrate
# that dataset (e.g. <10 P or <10 B controls), not "not computed yet"; run
# run_acmgscaler_all.py separately (it parallelizes across datasets via
# joblib) if ACMGSCALER_OUTPUT_DIR is stale or unset.
acmgscaler_conf_raw = None  # populated below; stays None if ACMGSCALER_OUTPUT_DIR is missing (see section 3b)
if not config.warn_if_missing(config.ACMGSCALER_OUTPUT_DIR, "acmgscaler comparison"):
    acmgscaler_conf_raw = []
    acmgscaler_vus_raw = []
    for dataset in datasets:
        df_acmg = load_acmgscaler_variants(dataset, config.ACMGSCALER_OUTPUT_DIR)
        acmgscaler_conf_raw.append(
            build_acmgscaler_confusion_matrix(df_acmg, label=dataset) if df_acmg is not None else None
        )
        # points_col="acmgscaler_points": acmgscaler's own evidence direction,
        # not ExCALIBR's standard_points/oob_points -- see build_vus_coverage.
        # Requires is_vus/auth_label on df_acmg, which only exist once
        # run_acmgscaler_all.py has been rerun with the comparison_methods.py
        # fix that added those columns to build_variants_df_from_scoreset.
        acmgscaler_vus_raw.append(
            build_vus_coverage(df_acmg, points_col="acmgscaler_points") if df_acmg is not None else None
        )
    _compare_vs_excalibr(acmgscaler_conf_raw, "acmgscaler", vus_raw=acmgscaler_vus_raw)

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
gmm_conf_raw_by_variant = {}  # variant -> per-dataset conf_raw list, for section 3b's combined table
if config.GMM_BASELINE_OUTPUT_DIR:
    for variant in config.GMM_BASELINE_VARIANTS:
        gmm_conf_raw = []
        gmm_vus_raw = []
        for dataset in datasets:
            df_gmm = load_comparison_variants(dataset, variant, config.GMM_BASELINE_OUTPUT_DIR)
            gmm_conf_raw.append(
                build_confusion_matrix(df_gmm, use_oob=False, label=f"{dataset}/gmm_{variant}")
                if df_gmm is not None else None
            )
            # No points_col override -- the GMM baseline's variants.csv is
            # already ExCALIBR-shaped (standard_points column), same reason
            # build_confusion_matrix above needs no method-specific handling.
            gmm_vus_raw.append(
                build_vus_coverage(df_gmm, use_oob=False, label=f"{dataset}/gmm_{variant}")
                if df_gmm is not None else None
            )
        _compare_vs_excalibr(gmm_conf_raw, f"gmm_{variant}", vus_raw=gmm_vus_raw)
        gmm_conf_raw_by_variant[variant] = gmm_conf_raw
else:
    print("  SKIP GMM baseline comparison: analysis.config.GMM_BASELINE_OUTPUT_DIR not set")

# %% [markdown]
# ### A9b. Skew-locked ExCALIBR comparison
#
# "Skew-locked" = the canonical pipeline rerun with each component's skew
# parameter fixed (not freely fit), instead of ExCALIBR's normal freely-fit
# skew-normal components. Unlike acmgscaler/gmm above, this is a full
# ExCALIBR-shaped output tree (`analysis.config.SKEW_LOCKED_OUTPUT_DIR`) --
# discovered/loaded with `analysis.discovery` exactly like `OUTPUT_DIR`
# itself, not `analysis.comparison_methods`. Its `*_variants.csv` files
# already carry `auth_label` (no `attach_author_labels` needed) and
# `oob_points`/`oob_n_boots`/`oob_prior` columns, so matrices are built with
# `use_oob=True`, same as the canonical pipeline. Compared against the
# primary method with the same `_compare_vs_excalibr` helper used above, so
# its matched-datasets
# aggregate performance is picked up automatically by 3b, and its evidence
# distribution is added alongside section 4's.


# %%
skew_locked_conf_raw = None  # populated below; stays None if SKEW_LOCKED_OUTPUT_DIR is missing (see section 3b)
if not config.warn_if_missing(config.SKEW_LOCKED_OUTPUT_DIR, "skew-locked ExCALIBR comparison"):
    skew_tree, skew_model_selections, skew_calibrations = discover_outputs(Path(config.SKEW_LOCKED_OUTPUT_DIR))
    skew_df = load_all_variants(
        tree=skew_tree, model_selections=skew_model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=datasets, calibrations=skew_calibrations, min_controls=0,
    )
    skew_locked_conf_raw = []
    skew_locked_vus_raw = []
    for dataset in datasets:
        df_sk = skew_df[skew_df["dataset"] == dataset] if not skew_df.empty else skew_df
        skew_locked_conf_raw.append(
            build_confusion_matrix(df_sk, use_oob=True, label=f"{dataset}/skew_locked")
            if not df_sk.empty else None
        )
        skew_locked_vus_raw.append(
            build_vus_coverage(df_sk, use_oob=True, label=f"{dataset}/skew_locked")
            if not df_sk.empty else None
        )
    _compare_vs_excalibr(skew_locked_conf_raw, "skew_locked", vus_raw=skew_locked_vus_raw)
else:
    skew_tree, skew_model_selections, skew_calibrations = {}, {}, {}
    skew_df = pd.DataFrame()

# %% [markdown]
# ### A9c. Pathomechanism-aware prior/LR+ comparison
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
pathomech_conf_raw_aligned = None  # populated below; stays None if PATHOMECHANISM_OUTPUT_DIR is missing (see section 3b)
if not config.warn_if_missing(config.PATHOMECHANISM_OUTPUT_DIR, "pathomechanism comparison"):
    pm_tree, pm_model_selections, pm_calibrations = discover_outputs(Path(config.PATHOMECHANISM_OUTPUT_DIR))
    pm_df_all = load_all_variants(
        tree=pm_tree, model_selections=pm_model_selections, dataset_configs=dataset_configs,
        methods_filter=None, datasets_filter=None, calibrations=pm_calibrations, min_controls=0,
    )
    pm_method = sorted(pm_df_all["method"].unique())[0] if not pm_df_all.empty else None
    pm_df = pm_df_all[pm_df_all["method"] == pm_method] if pm_method else pm_df_all

    # `datasets`-aligned (not pathomech_datasets' union below) so this lines
    # up index-for-index with conf_by_method[primary_method]/skew_locked_conf_raw
    # for section 3b's ExCALIBR/skew-locked/pathomechanism table.
    pathomech_conf_raw_aligned = [
        build_confusion_matrix(
            pm_df[pm_df["dataset"] == ds], use_oob=True, label=f"{ds}/pathomech_boundary",
        ) if not pm_df[pm_df["dataset"] == ds].empty else None
        for ds in datasets
    ]

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
# ### A9d. Ablation / baseline performance tables

# %%
from analysis.manuscript_stats import compute_aggregate_metrics_multi, latex_performance_table_multi

# --- ExCALIBR vs acmgscaler vs ExCALIBR (prior=0.1), on datasets acmgscaler could calibrate ---
if acmgscaler_conf_raw is not None and manual_prior_conf_raw is not None:
    _agg, _own, _n_matched = compute_aggregate_metrics_multi(
        {
            "excalibr": conf_by_method[primary_method],
            "acmgscaler": acmgscaler_conf_raw,
            "excalibr_prior_0.1": manual_prior_conf_raw,
        },
        datasets,
    )
    _latex = latex_performance_table_multi(
        _agg,
        {"excalibr": "ExCALIBR", "acmgscaler": "acmgscaler", "excalibr_prior_0.1": "ExCALIBR (prior=0.1)"},
        _own, len(datasets), _n_matched,
        caption=r"Performance comparison of out-of-bag \excalibr-calibrated evidence vs. "
                r"ACMGScaler (Badonyi \& Marsh 2025) and \excalibr\ re-calibrated with a fixed "
                r"prior of 0.1.",
        label="tab:acmgscaler_prior01_performance",
    )
    save_latex_table(_latex, FIGURE_SUBDIRS["tables"] / "excalibr_vs_acmgscaler_vs_prior01_performance.tex")
    REG.table("tab:pooled_acmgscaler_prior01", _latex)  # pooled metrics (Supplementary Table)
else:
    print("  SKIP acmgscaler/prior=0.1 performance table: acmgscaler or manual-prior output unavailable")

# --- ExCALIBR vs skew-locked vs pathomechanism-aware prior vs GMM baselines,
# all in ONE table -- matches section 3b2's comparison_boxplots grouping
# below (same rationale: their calibratable datasets are nearly identical,
# so one comparison group/table with everything side by side is more
# informative than three near-identical dataset scopes split across
# panels). `_ablation_baseline_matrices` is built once here and reused by
# 3b2's panel_groups instead of being reconstructed there, so the table and
# figure can't disagree about which methods/variants are included. Only
# "plp_blb" and "all_plp_blb" GMM variants shown (not their "_synon"
# siblings, which just add Synonymous to the benign pool and are strictly
# no better) -- still computed/available in gmm_conf_raw_by_variant for
# anyone who wants their own pairwise comparison.
_ablation_baseline_matrices = {"excalibr": conf_by_method[primary_method]}
_ablation_baseline_labels = {"excalibr": "ExCALIBR"}
if skew_locked_conf_raw is not None:
    _ablation_baseline_matrices["skew_locked"] = skew_locked_conf_raw
    _ablation_baseline_labels["skew_locked"] = "Skew-locked ExCALIBR"
if pathomech_conf_raw_aligned is not None:
    _ablation_baseline_matrices["pathomechanism"] = pathomech_conf_raw_aligned
    _ablation_baseline_labels["pathomechanism"] = "Pathomechanism-aware prior"
_gmm_display = {"plp_blb": "GMM (P/LP+B/LB)", "all_plp_blb": "GMM (all P/LP+B/LB)"}
for _variant, _conf in gmm_conf_raw_by_variant.items():
    if _variant in ("plp_blb", "all_plp_blb"):
        _ablation_baseline_matrices[f"gmm_{_variant}"] = _conf
        _ablation_baseline_labels[f"gmm_{_variant}"] = _gmm_display[_variant]

if len(_ablation_baseline_matrices) > 1:
    _agg, _own, _n_matched = compute_aggregate_metrics_multi(_ablation_baseline_matrices, datasets)
    _latex = latex_performance_table_multi(
        _agg, _ablation_baseline_labels, _own, len(datasets), _n_matched,
        caption=r"Performance comparison of out-of-bag \excalibr-calibrated evidence vs. the "
                r"skew-locked calibration, the pathomechanism-aware prior/likelihood-ratio variant, "
                r"and two simple two-component GMM calibration baselines (prior=0.1), differing only "
                r"in which variants are pooled as benign controls.",
        label="tab:ablation_baseline_performance",
    )
    save_latex_table(_latex, FIGURE_SUBDIRS["tables"] / "excalibr_vs_ablations_baselines_performance.tex")
    REG.table("tab:pooled_ablations_baselines", _latex)  # pooled metrics (Supplementary Table)
else:
    print("  SKIP skew-locked/pathomechanism/GMM baseline performance table: no comparison output available")

# %% [markdown]
# ### A9e. ED Fig. 4 box plots + stacked FP/FN/indeterminate breakdown
#
# Section 3b's tables pool every dataset into one summed confusion matrix per
# method -- the right format for the manuscript, but it hides (a) whether a
# method's accuracy/coverage is consistent across datasets or dominated by a
# couple of large ones, and (b) which datasets actually drive the pooled
# FP/FN/indeterminate counts. Both figures below (`analysis.comparison_diagnostics`)
# reuse the exact same per-dataset confusion-matrix lists as section 3b --
# no new data loading, only new aggregation/plotting -- and read section 3b's
# own `compute_aggregate_metrics_multi` dict directly for their aggregate
# reference numbers, so neither figure can disagree with the tables.
#
# GMM baseline(s), skew-locked, and pathomechanism are merged into one group
# here -- reusing 3b's `_ablation_baseline_matrices` directly (same methods/
# GMM variants) so this figure can't disagree with that table about which
# comparisons are included.


# %%
from analysis.comparison_diagnostics import plot_comparison_boxplots, plot_comparison_stacked_bars

_panel_groups = {}
if any(a is not None for a in auths):
    _panel_groups["ExCALIBR vs. Author"] = {"excalibr": conf_by_method[primary_method], "author": auths}
if acmgscaler_conf_raw is not None and manual_prior_conf_raw is not None:
    _panel_groups["ExCALIBR vs. ACMGScaler vs. ExCALIBR (prior=0.1)"] = {
        "excalibr": conf_by_method[primary_method], "acmgscaler": acmgscaler_conf_raw,
        "excalibr_prior_0.1": manual_prior_conf_raw,
    }
if len(_ablation_baseline_matrices) > 1:
    _panel_groups["ExCALIBR vs. GMM / Skew-locked / Pathomechanism"] = _ablation_baseline_matrices

if _panel_groups:
    fig_box = plot_comparison_boxplots(_panel_groups, datasets, len(datasets))
    save_and_show(fig_box, FIGURE_SUBDIRS["manuscript"] / "comparison_boxplots.png")

    fig_stack = plot_comparison_stacked_bars(_panel_groups, datasets)
    save_and_show(fig_stack, FIGURE_SUBDIRS["manuscript"] / "comparison_stacked_bars.png")
else:
    print("  SKIP per-dataset diagnostic figures: no comparison group had usable matrices")

# %%
REG.figure("fig:comparison_boxplots", FIGURE_SUBDIRS["manuscript"] / "comparison_boxplots.png")
REG.num("NumACMGScalerDatasets",
        sum(m is not None for m in acmgscaler_conf_raw) if acmgscaler_conf_raw is not None else None,
        source="datasets for which acmgscaler estimated likelihood ratios", inputs=[config.ACMGSCALER_OUTPUT_DIR])

# Per-dataset medians quoted in the Results (ED Fig. 4): the same per-dataset
# arrays the violins are drawn from (comparison_diagnostics.
# compute_per_dataset_metrics_multi, matched datasets per group), DOR over
# finite values only -- exactly the figure's "median:" labels.
from analysis.comparison_diagnostics import compute_per_dataset_metrics_multi


def _register_medians(group, tag_by_method, n_tag):
    _df = compute_per_dataset_metrics_multi(group, datasets)
    REG.num(n_tag, len(_df), source="matched datasets in this ED Fig. 4 comparison group")
    for _m, _tag in tag_by_method.items():
        _acc = _df[f"{_m}_accuracy"].to_numpy(dtype=float)
        _dor = _df[f"{_m}_dor"].to_numpy(dtype=float)
        REG.num(f"ResCmpAccMed{_tag}", float(np.median(_acc[np.isfinite(_acc)])), fmt=".3f",
                source=f"ED Fig. 4 median accuracy: {_m}")
        REG.num(f"ResCmpDORMed{_tag}", float(np.median(_dor[np.isfinite(_dor)])), fmt=",.1f",
                source=f"ED Fig. 4 median DOR (finite values): {_m}")


if "ExCALIBR vs. ACMGScaler vs. ExCALIBR (prior=0.1)" in _panel_groups:
    _register_medians(_panel_groups["ExCALIBR vs. ACMGScaler vs. ExCALIBR (prior=0.1)"],
                      {"excalibr": "ExcalibrAcmgSet", "acmgscaler": "Acmg", "excalibr_prior_0.1": "PriorFixed"},
                      "ResCmpAcmgNumMatched")
# Results (acmgscaler comparison): ExCALIBR on the datasets acmgscaler could
# not calibrate, and the pooled accuracies of Supplementary Table "pooled
# acmgscaler" with and without the two LDLR Tabet datasets (several hundred
# controls each), which the text says drive acmgscaler's small pooled edge.
if acmgscaler_conf_raw is not None and manual_prior_conf_raw is not None:
    _no_acmg = [i for i, m in enumerate(acmgscaler_conf_raw) if m is None]
    REG.num("ResCmpNumNoAcmg", len(_no_acmg), source="calibrated datasets acmgscaler could not calibrate")
    _df_na = compute_per_dataset_metrics_multi(
        {"excalibr": [conf_by_method[primary_method][i] for i in _no_acmg]}, [datasets[i] for i in _no_acmg])
    _acc_na = _df_na["excalibr_accuracy"].to_numpy(dtype=float)
    REG.num("ResCmpAccMedExcalibrNoAcmg", float(np.median(_acc_na[np.isfinite(_acc_na)])), fmt=".3f",
            source=f"median per-dataset ExCALIBR accuracy, {len(_df_na)} datasets without acmgscaler calibration")

    def _pooled_acc(keep):
        _agg_k, _, _ = compute_aggregate_metrics_multi(
            {m: [v[i] for i in keep] for m, v in (("excalibr", conf_by_method[primary_method]),
                                                  ("acmgscaler", acmgscaler_conf_raw),
                                                  ("excalibr_prior_0.1", manual_prior_conf_raw))},
            [datasets[i] for i in keep])
        return _agg_k["excalibr"]["accuracy"], _agg_k["acmgscaler"]["accuracy"]

    _all_i = list(range(len(datasets)))
    _no_ldlr = [i for i in _all_i if not datasets[i].startswith("LDLR_Tabet_2025")]
    _pe, _pa = _pooled_acc(_all_i)
    _pe2, _pa2 = _pooled_acc(_no_ldlr)
    REG.num("ResCmpPooledAccExcalibrAcmgSet", _pe, fmt=".3f", source="pooled accuracy, ExCALIBR (acmgscaler-matched datasets)")
    REG.num("ResCmpPooledAccAcmg", _pa, fmt=".3f", source="pooled accuracy, acmgscaler")
    REG.num("ResCmpPooledAccExcalibrNoLDLR", _pe2, fmt=".3f", source="same, without LDLR_Tabet_2025_* datasets")
    REG.num("ResCmpPooledAccAcmgNoLDLR", _pa2, fmt=".3f", source="same, without LDLR_Tabet_2025_* datasets")
    REG.check("pooled-acmg-gap-ldlr", (_pa - _pe) > 0 and (_pa2 - _pe2) < 0.5 * (_pa - _pe),
              f"pooled accuracy gap acmgscaler-ExCALIBR: {_pa - _pe:+.4f} with LDLR, {_pa2 - _pe2:+.4f} without",
              "main.tex Results (acmgscaler: 'mainly due to two LDLR calibrations')")

if "ExCALIBR vs. GMM / Skew-locked / Pathomechanism" in _panel_groups:
    _abl = _panel_groups["ExCALIBR vs. GMM / Skew-locked / Pathomechanism"]
    _register_medians(_abl, {k: t for k, t in (("excalibr", "Excalibr"), ("skew_locked", "SkewLocked"),
                                                ("gmm_plp_blb", "Gmm"), ("gmm_all_plp_blb", "GmmAll"))
                             if k in _abl}, "ResCmpAblationNumMatched")
for _p, _what in ((config.ACMGSCALER_OUTPUT_DIR, "acmgscaler outputs (run_acmgscaler_all.py)"),
                  (config.MANUAL_PRIOR_OUTPUT_DIR, "fixed-prior (0.1) ExCALIBR rerun"),
                  (config.SKEW_LOCKED_OUTPUT_DIR, "skew-locked ExCALIBR rerun")):
    REG.require(_p, _what, newer_than=[DATASET_TSV])
# GMM baseline and pathomechanism runs: content-based staleness (only the
# datasets whose inputs changed since those runs need rerunning; see
# analysis/sweep_staleness.py SIDE_RUN_CHANGED_DATASETS).
from analysis.sweep_staleness import side_run_stale, canonical_input_changes
_gmm_stale = side_run_stale(config.GMM_BASELINE_OUTPUT_DIR, datasets)
REG.require(config.GMM_BASELINE_OUTPUT_DIR, "GMM baseline outputs",
            expect_datasets=set(datasets), found_datasets=set(datasets) - set(_gmm_stale),
            regen="python hpc/simple_gmm_baseline.py --output-dir "
                  f"{config.GMM_BASELINE_OUTPUT_DIR} --dataset <name>  # for each of {sorted(_gmm_stale)}")
_pathomech_stale = {**canonical_input_changes(config.PATHOMECHANISM_OUTPUT_DIR, datasets, tree, model_selections,
                                              dataset_configs),
                    **side_run_stale(config.PATHOMECHANISM_OUTPUT_DIR, datasets)}
REG.require(config.PATHOMECHANISM_OUTPUT_DIR, "pathomechanism-aware ExCALIBR rerun",
            expect_datasets=set(datasets), found_datasets=set(datasets) - set(_pathomech_stale),
            regen="python run_igvf_batch.py ... --pathomechanism-prior --oob --splits-file "
                  f"{config.SPLITS_FILE} --datasets {' '.join(sorted(_pathomech_stale))}")
if manual_prior_conf_raw is not None:
    REG.require(config.MANUAL_PRIOR_OUTPUT_DIR, "fixed-prior rerun dataset coverage", expect_datasets=set(datasets),
                found_datasets={d for d, m in zip(datasets, manual_prior_conf_raw) if m is not None})
if skew_locked_conf_raw is not None:
    REG.require(config.SKEW_LOCKED_OUTPUT_DIR, "skew-locked rerun dataset coverage", expect_datasets=set(datasets),
                found_datasets={d for d, m in zip(datasets, skew_locked_conf_raw) if m is not None})

# %% [markdown]
# ## A10. Extended Data Table 6: mode of inheritance (AD vs. AR)

#
# Supplementary analysis for a reviewer rebuttal: does ExCALIBR's
# mode-of-inheritance (MOI)-agnostic calibration systematically bias
# performance between Autosomal Dominant (AD) and Autosomal Recessive (AR)
# genes? Mirrors section 3's main ExCALIBR-vs-ClinVar confusion matrix, but
# split into two pooled panels (AD genes vs AR genes), plus a per-gene
# statistical comparison (mean/median/n/p-value per metric).
#
# Gene-level MOI comes from ClinGen Gene-Disease Validity curations
# (`config.CLINGEN_MOI_CSV`), filtered to Moderate+ confidence
# (Definitive/Strong/Moderate -- drops Limited/Disputed/Refuted/No Known
# Disease Relationship) and MOI in {AD, AR} (drops X-linked/Semidominant/
# Mitochondrial/Undetermined-only genes entirely). Some genes have a
# moderate+ entry for BOTH AD and AR (real biology -- e.g. BRCA1 is AD for
# cancer predisposition, AR for Fanconi anemia); this is run twice, once
# counting those dual-MOI genes in both groups and once dropping them from
# both, so the AD/AR comparison doesn't depend on how that edge case is
# handled. Gene scope is the full, unfiltered `deduped_variants` (matching
# section 3's main confusion figure) -- not `EXCLUDED_GENES`-filtered.


# %%
from scipy import stats as sps
from src.assay_calibration.plot_utils.utils import compute_classification_metrics
from analysis.manuscript_stats import latex_moi_comparison_table

_MOI_METRICS = ["dor_standard", "accuracy", "coverage", "mcc", "sensitivity", "specificity"]
_MOI_METRIC_LABELS = {
    "dor_standard": "DOR", "accuracy": "Accuracy", "coverage": "Determinate %",
    "mcc": "MCC", "sensitivity": "Sensitivity", "specificity": "Specificity",
}

if config.warn_if_missing(config.CLINGEN_MOI_CSV, "ClinGen gene-disease MOI CSV (AD vs AR comparison)"):
    gene_to_moi_groups = {}
else:
    clingen_moi = pd.read_csv(config.CLINGEN_MOI_CSV, skiprows=4)
    clingen_moi = clingen_moi[clingen_moi["GENE SYMBOL"] != "+++++++++++"]
    clingen_moi = clingen_moi[clingen_moi["CLASSIFICATION"].isin(["Definitive", "Strong", "Moderate"])]
    clingen_moi = clingen_moi[clingen_moi["MOI"].isin(["AD", "AR"])]
    gene_to_moi_groups = clingen_moi.groupby("GENE SYMBOL")["MOI"].apply(set).to_dict()


def _compute_moi_comparison(drop_dual_genes: bool, label: str):
    all_genes_in_scope = sorted(deduped_variants["gene"].unique())
    ad_genes = [g for g in all_genes_in_scope if "AD" in gene_to_moi_groups.get(g, set())]
    ar_genes = [g for g in all_genes_in_scope if "AR" in gene_to_moi_groups.get(g, set())]
    if drop_dual_genes:
        dual_genes = sorted(set(ad_genes) & set(ar_genes))
        ad_genes = [g for g in ad_genes if g not in dual_genes]
        ar_genes = [g for g in ar_genes if g not in dual_genes]
    else:
        dual_genes = sorted(set(ad_genes) & set(ar_genes))

    print(f"\n{'=' * 80}\nMODE OF INHERITANCE COMPARISON ({label})\n{'=' * 80}")
    print(f"AD genes (n={len(ad_genes)}): {ad_genes}")
    print(f"AR genes (n={len(ar_genes)}): {ar_genes}")
    print(f"Dual-MOI genes (AD and AR, {'dropped' if drop_dual_genes else 'counted in both'}, "
          f"n={len(dual_genes)}): {dual_genes}")

    per_gene_rows = []
    for gene in sorted(set(ad_genes) | set(ar_genes)):
        matrix = build_deduped_confusion_matrix(deduped_variants[deduped_variants["gene"] == gene])
        if matrix is None:
            print(f"  SKIP {gene}: no P/LP or B/LB variants in scope")
            continue
        metrics = compute_classification_metrics(matrix)
        per_gene_rows.append({"gene": gene, **{m: metrics[m] for m in _MOI_METRICS}})
    per_gene_metrics = pd.DataFrame(per_gene_rows)

    long_rows = []
    if not per_gene_metrics.empty:
        for group_label, group_genes in (("AD", ad_genes), ("AR", ar_genes)):
            sub = per_gene_metrics[per_gene_metrics["gene"].isin(group_genes)].copy()
            sub["moi_group"] = group_label
            long_rows.append(sub)
    moi_long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

    summary_rows = []
    for metric in _MOI_METRICS:
        ad_vals = moi_long_df.loc[moi_long_df["moi_group"] == "AD", metric].to_numpy()
        ar_vals = moi_long_df.loc[moi_long_df["moi_group"] == "AR", metric].to_numpy()
        # dor_standard is deliberately float('inf') from compute_classification_metrics
        # whenever a gene's pooled controls have zero FP or zero FN -- expected and
        # common for our small-per-gene-control-count genes. Averaging inf into a
        # mean/median silently breaks the whole comparison, so exclude non-finite
        # values from this metric's summary stats and report the exclusion count
        # instead of hiding it.
        ad_finite = ad_vals[np.isfinite(ad_vals)]
        ar_finite = ar_vals[np.isfinite(ar_vals)]
        n_inf_ad = len(ad_vals) - len(ad_finite)
        n_inf_ar = len(ar_vals) - len(ar_finite)
        if n_inf_ad or n_inf_ar:
            print(f"  {_MOI_METRIC_LABELS[metric]}: excluded {n_inf_ad} non-finite AD value(s), "
                  f"{n_inf_ar} non-finite AR value(s) from mean/median/p-value")
        if len(ad_finite) and len(ar_finite):
            p_value = sps.mannwhitneyu(ad_finite, ar_finite, alternative="two-sided").pvalue
        else:
            p_value = None
        summary_rows.append({
            "metric": _MOI_METRIC_LABELS[metric],
            "mean_AD": np.mean(ad_finite) if len(ad_finite) else None,
            "mean_AR": np.mean(ar_finite) if len(ar_finite) else None,
            "median_AD": np.median(ad_finite) if len(ad_finite) else None,
            "median_AR": np.median(ar_finite) if len(ar_finite) else None,
            "n_AD": len(ad_finite),
            "n_AR": len(ar_finite),
            "p_value": p_value,
        })
    moi_comparison_df = pd.DataFrame(summary_rows)
    # DOR is on a much larger, non-percentage scale than the other metrics
    # (accuracy/coverage/mcc/sensitivity/specificity, all in [0, 1]) -- round
    # it to whole numbers instead of 4 decimal places, which is far more
    # digits than DOR's scale ever needs.
    _moi_display_df = moi_comparison_df.copy()
    _dor_row = _moi_display_df["metric"] == "DOR"
    _dor_cols = ["mean_AD", "mean_AR", "median_AD", "median_AR"]
    _moi_display_df.loc[_dor_row, _dor_cols] = _moi_display_df.loc[_dor_row, _dor_cols].round(0)
    _moi_display_df.loc[~_dor_row, _dor_cols] = _moi_display_df.loc[~_dor_row, _dor_cols].round(4)
    _moi_display_df["p_value"] = _moi_display_df["p_value"].round(4)
    print(f"\n{_moi_display_df.to_string(index=False)}")

    _moi_latex = latex_moi_comparison_table(moi_comparison_df, label=label)
    save_latex_table(
        _moi_latex,
        FIGURE_SUBDIRS["tables"] / f"moi_comparison{'_dual_dropped' if drop_dual_genes else '_dual_counted'}.tex",
    )
    if drop_dual_genes:  # ED Table 6 is the dual-MOI-dropped version
        REG.table("tab:moi_comparison_dual_dropped", _moi_latex, inputs=[config.CLINGEN_MOI_CSV])
        _moi_p = moi_comparison_df["p_value"].dropna()
        REG.num("ResMoiMinP", float(_moi_p.min()), fmt=".2f", source="ED Table 6: smallest Mann-Whitney U p-value")
        REG.check("moi-no-significant-difference", bool((_moi_p >= 0.05).all()),
                  f"ED Table 6 p-values: {sorted(_moi_p.round(3).tolist())}", "main.tex Results (MOI)")

    ad_pooled_matrix = build_deduped_confusion_matrix(deduped_variants[deduped_variants["gene"].isin(ad_genes)])
    ar_pooled_matrix = build_deduped_confusion_matrix(deduped_variants[deduped_variants["gene"].isin(ar_genes)])
    fig_filename = f"supp_fig_AD_vs_AR_confusion{'_dual_dropped' if drop_dual_genes else ''}.png"
    if ad_pooled_matrix is not None and ar_pooled_matrix is not None:
        make_confusion_figure(
            danzs_m1=[ad_pooled_matrix], danzs_m2=[ar_pooled_matrix], dataset_names=["AD_vs_AR"],
            label1="AD", label2="AR", figure_dir=FIGURE_SUBDIRS["manuscript"], filename=fig_filename,
        )
    else:
        print(f"  SKIP {fig_filename}: no P/LP or B/LB variants in one or both groups")

    return moi_comparison_df


moi_comparison_dual_counted_df = _compute_moi_comparison(drop_dual_genes=False, label="dual-MOI genes counted in both groups")
moi_comparison_dual_dropped_df = _compute_moi_comparison(drop_dual_genes=True, label="dual-MOI genes dropped")

# %% [markdown]
# ## A11. Supplementary Tables (robustness, bootstrap count, restarts, splice ablation, skew-locked likelihood)
#
# These read experiment outputs produced by separate runs (see each
# subsection); the build checks each one covers the current datasets.

# %% [markdown]
# ### A11a. Robustness analysis (downsampling / label discordance) -> Supplementary Tables S3-S4
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
    def _compute_robustness():
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
        return robustness_summaries, robustness_matrices, robustness_max_strengths, config_summary_jobs

    import analysis.robustness as _robustness_mod
    (robustness_summaries, robustness_matrices, robustness_max_strengths, config_summary_jobs), _rob_hit = cached(
        "robustness_matrices", _compute_robustness,
        inputs=[(config.ROBUSTNESS_OUTPUT_DIR, "**/*_calibration.json"),
                (config.ROBUSTNESS_OUTPUT_DIR, "**/*_variants.csv")],
        code=[_robustness_mod, _compute_robustness], key=sorted(robustness_bases),
    )
    # All base datasets' (perturbation_type, level) config-summary figures
    # rendered in one parallel batch -- each is an independent, self-
    # contained unit of work (its own pooled_fits_for_level + sample_density
    # call), the actual cost driver for this section. display_images=False:
    # still rendered and saved to disk, just not dumped inline here too --
    # this is easily 30+ figures per run, too many to scroll through.
    # Exploratory figures only; unchanged inputs (cache hit) -> already on disk.
    if not _rob_hit:
        run_config_summary_plots_batch(
            config_summary_jobs, figure_dir=FIGURE_DIR, robustness_output_dir=config.ROBUSTNESS_OUTPUT_DIR,
            display_images=False,
        )

    # Confusion-matrix grids: 4 columns (one per robustness base dataset) x
    # one row per condition level (control first, then descending downsample
    # N / ascending discordance fraction) -- the same diverging Blue/Gray/Red
    # style used everywhere else, plus each cell's strongest pathogenic/
    # benign evidence point ("Max strengths: +X, -Y"). show=False: still
    # saved to disk, just not displayed inline (same reasoning as
    # display_images=False above).
    if robustness_matrices:
        plot_robustness_confusion_matrix_grid(
            robustness_matrices, robustness_max_strengths, "downsample",
            base_datasets=list(robustness_matrices.keys()), figure_dir=FIGURE_DIR, show=False,
        )
        plot_robustness_confusion_matrix_grid(
            robustness_matrices, robustness_max_strengths, "discordance",
            base_datasets=list(robustness_matrices.keys()), figure_dir=FIGURE_DIR, show=False,
        )

        # Same per-cell seed-to-seed data as the grids above, as a LaTeX
        # table (accuracy + determinate % + max pathogenic/benign evidence
        # strength, median [IQR] across the 10 seeds) instead of a figure
        # caption.
        from analysis.robustness import robustness_seed_metrics_table
        from analysis.manuscript_stats import latex_robustness_metrics_table

        # Brnich OddsPath evidence under the same label perturbations (S4),
        # for base datasets whose controls have author functional classes.
        # Uses the robustness fits' own control sets (ClinVar 2018 for
        # BRCA1/MSH2), so values can differ from Supplementary Data 3.
        from analysis.robustness import compute_robustness_oddspath
        import analysis.robustness as _robustness_mod
        robustness_oddspath, _ = cached(
            "robustness_oddspath", lambda: compute_robustness_oddspath(list(robustness_matrices.keys())),
            inputs=[DATASET_TSV], code=[_robustness_mod.compute_robustness_oddspath,
                                        _robustness_mod.oddspath_from_controls,
                                        _robustness_mod.oddspath_from_scoreset],
            key=sorted(robustness_matrices.keys()),
        )

        _robustness_tables = {}  # perturbation type -> LaTeX (Supplementary Tables S3/S4)
        for _ptype in ("downsample", "discordance"):
            _robustness_table_df = robustness_seed_metrics_table(
                robustness_summaries, robustness_max_strengths, _ptype, list(robustness_matrices.keys()),
                oddspath=robustness_oddspath if _ptype == "discordance" else None,
            )
            if not _robustness_table_df.empty:
                _robustness_latex = latex_robustness_metrics_table(_robustness_table_df, _ptype)
                _robustness_tables[_ptype] = _robustness_latex
                _robustness_n_seeds = int(_robustness_table_df["n_seeds"].max())
                save_latex_table(
                    _robustness_latex, FIGURE_SUBDIRS["tables"] / f"robustness_{_ptype}_confusion_grid_metrics.tex",
                )

# %% [markdown]
# ### A11b. Bootstrap-count-reduction analysis -> Supplementary Table S5
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
    # Restrict to the manuscript's calibrated datasets (the sweep folder also
    # holds excluded ones, e.g. SFPQ_IGVF and the TP53 meta-analysis), under
    # their current names; the coverage check is in A11.
    from tests.benchmark_bootstrap_reduction import _KNOWN_NAME_ALIASES as _bsr_name_aliases
    _bsr_excluded = [d for d in bsr_datasets if _bsr_name_aliases.get(d, d) not in set(datasets)]
    bsr_datasets = [d for d in bsr_datasets if _bsr_name_aliases.get(d, d) in set(datasets)]
    _bsr_found_datasets = {_bsr_name_aliases.get(d, d) for d in bsr_datasets}
    print(f"  using {len(bsr_datasets)} calibrated dataset(s); not in the manuscript set: {_bsr_excluded}")

    with open(config.DATASET_CONFIGS) as f:
        bsr_dataset_configs = json.load(f)
    # Figures + per-(dataset, level) metrics; cached (analysis/step_cache.py),
    # so an unchanged sweep skips loading the precomputed fits and re-rendering.
    def _compute_bsr():
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

        # Standard classification metrics (accuracy/coverage/sensitivity/
        # specificity/DOR/MCC) per (dataset, bootstrap-count level), pooled
        # (median [IQR]) across datasets at each level -- the tabular
        # complement to the [p5,p50,p95] LR+-curve figures just rendered above.
        from analysis.robustness import (
            compute_bootstrap_reduction_confusion_metrics, bootstrap_reduction_metrics_table,
        )
        from analysis.manuscript_stats import latex_bootstrap_reduction_metrics_table

        print(f"  Computing bootstrap-reduction standard metrics across {_mp.cpu_count()} CPUs...")
        _bsr_metric_results = _Parallel(n_jobs=-1, batch_size=1, backend="loky")(
            _delayed(compute_bootstrap_reduction_confusion_metrics)(
                dataset_name, br_dir / dataset_name, bsr_dataset_configs, config.DATASET_TSV,
                bsr_per_dataset_df[dataset_name],
            )
            for dataset_name in bsr_datasets
        )
        return _bsr_metric_results

    import analysis.robustness as _robustness_mod
    import tests.plot_bootstrap_reduction_config as _bsr_plot_mod
    _bsr_metric_results, _ = cached(
        "bootstrap_reduction_metrics", _compute_bsr,
        inputs=[(br_dir, "*/level_*/*_calibration.json"), (br_dir, "*/level_*/*_lr_values.json.gz"),
                config.DATASET_TSV, config.DATASET_CONFIGS],
        code=[_robustness_mod, _bsr_plot_mod, _compute_bsr], key=sorted(bsr_datasets),
    )
    from analysis.robustness import bootstrap_reduction_metrics_table
    from analysis.manuscript_stats import latex_bootstrap_reduction_metrics_table
    _bsr_metric_rows = [row for rows in _bsr_metric_results for row in rows]
    if _bsr_metric_rows:
        _bsr_metrics_df = pd.DataFrame(_bsr_metric_rows)
        _bsr_table_df = bootstrap_reduction_metrics_table(_bsr_metrics_df)
        _bsr_latex = latex_bootstrap_reduction_metrics_table(_bsr_table_df)
        save_latex_table(_bsr_latex, FIGURE_SUBDIRS["tables"] / "bootstrap_reduction_metrics.tex")
    else:
        print("  SKIP bootstrap-reduction metrics table: no dataset produced a usable confusion matrix")

# %% [markdown]
# ### A11c. Fit-number (restart-count) comparison -> Supplementary Table S6
#
# How much does reducing the number of EM restarts per fit degrade the
# best-of-N training log-likelihood? Median + IQR ribbon (across every
# dataset x n_c row) vs. restart count, from
# `tests/benchmark_num_fits_dataframe.py`'s `summary.csv`.


# %%
if not config.warn_if_missing(config.FIT_NUMBER_COMPARISON_SUMMARY_CSV, "fit-number comparison"):
    from analysis.robustness import (
        plot_fit_number_comparison_curve, compute_delta_std_column, summarize_delta_std_table,
    )

    fit_number_summary = pd.read_csv(config.FIT_NUMBER_COMPARISON_SUMMARY_CSV)
    # Sweep names lack the _clinvar_2018 suffix; keep only the manuscript's
    # calibrated datasets ("dataset" stays as-is: train_lls.json is keyed on it).
    from analysis.sweep_staleness import restart_sweep_name_map
    _rs_map = restart_sweep_name_map(fit_number_summary["dataset"].unique())
    fit_number_summary["pipeline_dataset"] = fit_number_summary["dataset"].map(_rs_map)
    _rs_excluded = sorted(set(fit_number_summary["pipeline_dataset"]) - set(datasets))
    fit_number_summary = fit_number_summary[fit_number_summary["pipeline_dataset"].isin(datasets)].reset_index(drop=True)
    print(f"  restart sweep restricted to calibrated datasets; not in the manuscript set: {_rs_excluded}")
    # Only each dataset's selected component count (Supplementary Data 1), not
    # every 2c/3c fit the sweep ran.
    _sel_nc = {}
    for _d in fit_number_summary["pipeline_dataset"].unique():
        _comps = list(tree.get(_d, {}).keys())
        if _comps:
            _sel_nc[_d] = int(resolve_component(_d, _comps, model_selections, dataset_configs).split("_", 1)[0].rstrip("c"))
    _n_before = len(fit_number_summary)
    fit_number_summary = fit_number_summary[
        fit_number_summary["n_c"].astype(int) == fit_number_summary["pipeline_dataset"].map(_sel_nc)].reset_index(drop=True)
    print(f"  restart sweep restricted to selected components: {len(fit_number_summary)} of {_n_before} rows")
    print(f"Loaded {len(fit_number_summary)} rows, "
          f"{fit_number_summary['dataset'].nunique()} datasets, "
          f"num_fits levels: {sorted(fit_number_summary['num_fits'].unique())}")
    plot_fit_number_comparison_curve(fit_number_summary, figure_dir=FIGURE_DIR, label="all_datasets")

    # LL-metrics table: delta_std (dimensionless, in units of each
    # (dataset, n_c)'s own restart-to-restart SD -- see
    # compute_delta_std_column's docstring for why raw "delta" alone isn't
    # comparable across datasets), median [P25, P75] across every
    # (dataset, n_c) row at each restart-count level -- the tabular twin of
    # the curve just plotted above.
    if not config.warn_if_missing(
        config.FIT_NUMBER_COMPARISON_TRAIN_LLS_JSON, "fit-number comparison train_lls.json (for delta_std)",
    ):
        from analysis.manuscript_stats import latex_fit_number_comparison_table

        import analysis.robustness as _robustness_mod
        fit_number_summary, _ = cached(
            "restart_delta_std",
            lambda: compute_delta_std_column(fit_number_summary, config.FIT_NUMBER_COMPARISON_TRAIN_LLS_JSON),
            inputs=[config.FIT_NUMBER_COMPARISON_SUMMARY_CSV, config.FIT_NUMBER_COMPARISON_TRAIN_LLS_JSON],
            code=[_robustness_mod], key=sorted(fit_number_summary["dataset"].unique()),
        )
        _fit_number_levels = [
            l for l in (1, 5, 10, 20, 30, 40, 50, 100) if l in set(fit_number_summary["num_fits"].unique())
        ]
        _fit_number_table_df = summarize_delta_std_table(
            fit_number_summary, levels=_fit_number_levels, metric="delta_std",
        )
        _fit_number_latex = latex_fit_number_comparison_table(_fit_number_table_df, metric="delta_std")
        save_latex_table(_fit_number_latex, FIGURE_SUBDIRS["tables"] / "fit_number_comparison_metrics.tex")

# %% [markdown]
# ### A11d. SpliceAI threshold / VEP splice-consequence filter ablation
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
    import analysis.splice_ablation as _splice_mod
    splice_ablation_summary, _ = cached(
        "splice_ablation_summary",
        lambda: run_splice_ablation_analysis(config.SPLICE_ABLATION_ROOT, dataset_configs=dataset_configs),
        inputs=[(config.SPLICE_ABLATION_ROOT, "*/*/*_calibration.json"), (config.SPLICE_ABLATION_ROOT, "*/*/*_variants.csv"),
                config.DATASET_CONFIGS],
        code=[_splice_mod],
    )
    if not splice_ablation_summary.empty:
        print(f"Splice ablation: {splice_ablation_summary['condition_label'].nunique()} condition(s), "
              f"{splice_ablation_summary['dataset'].nunique()} dataset(s)")
        plot_splice_ablation_curve(splice_ablation_summary, figure_dir=FIGURE_DIR, label="all_datasets")

# %% [markdown]
# ### A11e. Splice ablation: pooled aggregate performance + population drift -> Supplementary Table S7
#
# Two condition-level summary tables, both pooling every dataset into a
# single combined confusion matrix per condition (like section 3b's
# aggregate performance report, but with "condition" as the row axis
# instead of one aggregate row):
#  - **Table A** (`compute_splice_ablation_aggregate_summary`): each
#    condition's own population + own calibration -- the full end-to-end
#    comparison, including how much the P/LP/B/LB/gnomAD/Synonymous
#    population itself drifts per condition (`pct_change_n_*` columns).
#  - **Table B** (`compute_splice_ablation_fixed_population_summary`):
#    `thresh_0.2`'s population held fixed, every other condition's own
#    calibration re-applied to it -- isolates the calibration-threshold
#    effect from population composition drift (Table A's percent changes
#    mix both effects together; Table B's don't).
# Both compare against `thresh_0.2` by default, matching
# `Scoreset.splicing_filter`'s own hardcoded defaults (spliceai_threshold=0.2,
# vep_splice_filter=True).


# %%
from analysis.splice_ablation import (
    compute_splice_ablation_aggregate_summary, compute_splice_ablation_fixed_population_summary,
)
from analysis.manuscript_stats import (
    latex_splice_ablation_summary_table, latex_splice_ablation_fixed_population_table,
)

if not config.warn_if_missing(config.SPLICE_ABLATION_ROOT, "splice ablation aggregate summary"):
    import analysis.splice_ablation as _splice_mod
    splice_ablation_agg_summary, _ = cached(
        "splice_ablation_aggregate",
        lambda: compute_splice_ablation_aggregate_summary(config.SPLICE_ABLATION_ROOT, dataset_configs=dataset_configs),
        inputs=[(config.SPLICE_ABLATION_ROOT, "*/*/*_calibration.json"), (config.SPLICE_ABLATION_ROOT, "*/*/*_variants.csv"),
                config.DATASET_CONFIGS],
        code=[_splice_mod],
    )
    if not splice_ablation_agg_summary.empty:
        print(splice_ablation_agg_summary.to_string(index=False))
        _splice_ablation_latex = latex_splice_ablation_summary_table(splice_ablation_agg_summary)
        save_latex_table(_splice_ablation_latex, FIGURE_SUBDIRS["tables"] / "splice_ablation_summary.tex")

    splice_ablation_fixed_pop_summary, _ = cached(
        "splice_ablation_fixed_population",
        lambda: compute_splice_ablation_fixed_population_summary(
            config.SPLICE_ABLATION_ROOT, dataset_configs=dataset_configs),
        inputs=[(config.SPLICE_ABLATION_ROOT, "*/*/*_calibration.json"), (config.SPLICE_ABLATION_ROOT, "*/*/*_variants.csv"),
                config.DATASET_CONFIGS],
        code=[_splice_mod],
    )
    if not splice_ablation_fixed_pop_summary.empty:
        print(splice_ablation_fixed_pop_summary.to_string(index=False))
        _splice_ablation_fixed_pop_latex = latex_splice_ablation_fixed_population_table(
            splice_ablation_fixed_pop_summary,
        )
        save_latex_table(
            _splice_ablation_fixed_pop_latex,
            FIGURE_SUBDIRS["tables"] / "splice_ablation_fixed_population_summary.tex",
        )

# %% [markdown]
# ### A11f. Skew-locked vs. regular ExCALIBR: fit quality by Yang distance (supplement text)
#
# For each calibrated dataset and its selected component count, the
# normalized Yang distance (the manuscript's fit-quality metric) between each
# sample's empirical CDF and the fitted mixture CDF, for every bootstrap of
# the regular fit and of the skew-locked refit (same seeds and splits; every
# component's skewness fixed at 0). Per dataset: the median across
# bootstraps per sample, and the largest of those over samples (the quantity
# compared with the poor-fit threshold in Results). Computed by
# `analysis/skew_locked_yang.py` (per-dataset caches under
# config.YANG_CACHE_DIR; `python -m analysis.skew_locked_yang` warms them).

# %%
import analysis.skew_locked_yang as _sly

# Gated on its own inputs so this section also runs alone.
if not config.warn_if_missing(config.SKEW_LOCKED_BOOTSTRAP_RESULTS, "skew-locked bootstrap fits") \
        and not config.warn_if_missing(config.PRECOMPUTED_FITS, "regular bootstrap fits"):
    skew_yang_df, _ = cached(
        "skew_locked_yang_val", lambda: _sly.compute(datasets, n_jobs=config.YANG_N_JOBS, split="val"),
        inputs=[config.PRECOMPUTED_FITS, config.SKEW_LOCKED_BOOTSTRAP_RESULTS, config.DATASET_CONFIGS],
        code=[_sly.compute, Path(_sly.__file__).parent / "yang_distance.py"],
        key=(sorted(datasets), config.YANG_N_GRID),
    )
    if not skew_yang_df.empty:
        print(skew_yang_df.sort_values("skew_locked_max").to_string(index=False))
        # Exploratory: per-dataset worst-sample distance, regular vs. skew-locked.
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(skew_yang_df["regular_max"], skew_yang_df["skew_locked_max"], s=14, color="k")
        _lim = [0, max(skew_yang_df[["regular_max", "skew_locked_max"]].max().max() * 1.05, 0.25)]
        ax.plot(_lim, _lim, color="grey", lw=0.8)
        for _t in (config.YANG_GOOD_FIT_THRESHOLD,):
            ax.axvline(_t, color="r", lw=0.6, ls="--"); ax.axhline(_t, color="r", lw=0.6, ls="--")
        ax.set_xlim(_lim); ax.set_ylim(_lim)
        ax.set_xlabel("Regular fit: largest per-sample Yang distance")
        ax.set_ylabel("Skew-locked fit: largest per-sample Yang distance")
        fig.tight_layout()
        _p = Path(FIGURE_SUBDIRS["skew_locked"]) / "skew_locked_vs_regular_yang.png"
        _p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_p, dpi=200); plt.close(fig)
        print(f"  Saved: {_p}")
else:
    skew_yang_df = pd.DataFrame()

# %%
# Skew-locked vs. regular fit quality on held-out variants: reported in the
# supplement text (Sensitivity and robustness analyses), not as a table.
if not skew_yang_df.empty:
    REG.num("SuppSkewLockedNumDatasets", len(skew_yang_df), source="datasets in the skew-locked held-out Yang comparison",
            inputs=[config.SKEW_LOCKED_BOOTSTRAP_RESULTS, PRECOMPUTED_FITS])
    # Paired (same bootstrap's held-out variants) skew-locked minus regular, per sample.
    def _p_tex(pv):
        if not np.isfinite(pv):
            return "--"
        if pv >= 0.01:
            return f"{pv:.2f}"
        m, e = f"{pv:.1e}".split("e")
        return f"{m}\\times 10^{{{int(e)}}}"
    _bys = _sly.summarize_by_sample(skew_yang_df).set_index("sample")
    for _s, _tag in (("pathogenic", "PLP"), ("benign", "BLB"), ("gnomad", "Gnomad"), ("synonymous", "Syn")):
        _r = _bys.loc[_s]
        REG.num(f"SuppSkewLockedPaired{_tag}N", _r["n"], source=f"datasets with a {_s} sample (paired Yang)")
        REG.num(f"SuppSkewLockedPaired{_tag}Worse", _r["n_worse"],
                source=f"datasets where skew-locked {_s} distance is larger (median paired diff > 0)")
        REG.num(f"SuppSkewLockedPaired{_tag}Diff", _r["median_diff"], fmt=".4f",
                source=f"median over datasets of median paired diff, {_s}")
        REG.text(f"SuppSkewLockedPaired{_tag}P", _p_tex(_r["p_wilcoxon"]),
                 source=f"two-sided Wilcoxon signed-rank p, {_s} = {_r['p_wilcoxon']:.3g}; use in math mode")
    REG.check("skew-locked-worse-every-sample",
              bool(((_bys["median_diff"] > 0) & (_bys["p_wilcoxon"] < 0.05)).all()),
              "held-out paired diff > 0 and p < 0.05 per sample: "
              + ", ".join(f"{k}: {r.median_diff:+.5f}, p={r.p_wilcoxon:.2g}" for k, r in _bys.iterrows()),
              "supplement.tex Skew-locked fit quality (worsened the fit for every sample)")
    REG.require(config.SKEW_LOCKED_BOOTSTRAP_RESULTS, "skew-locked bootstrap fits (supplement text)",
                expect_datasets=set(datasets), found_datasets=set(skew_yang_df["dataset"]),
                regen="python -m analysis.skew_locked_yang  # then rerun A11f")
# Sweep coverage: datasets missing from a sweep or whose fit inputs changed
# since it ran (analysis/sweep_staleness.py) count as not covered; the regen
# command reruns only those.
from analysis.sweep_staleness import bootstrap_reduction_stale, restart_sweep_stale, rerun_commands
_bsr_stale = bootstrap_reduction_stale(datasets)
_rs_stale = restart_sweep_stale(datasets, selected_nc={
    _d: int(resolve_component(_d, list(tree[_d].keys()), model_selections, dataset_configs).split("_", 1)[0].rstrip("c"))
    for _d in datasets if tree.get(_d)})
_sweep_cmds = rerun_commands(_bsr_stale, _rs_stale)
for _name, _st in (("S5", _bsr_stale), ("S6", _rs_stale)):
    for _d, _why in _st.items():
        print(f"  Supp Table {_name} stale: {_d}: {_why}")
# Supplementary Tables S3-S7: register the generated bodies and check inputs.
if "_robustness_tables" in globals():
    for _ptype, _latex in _robustness_tables.items():
        REG.table(f"tab:robustness_{_ptype}_metrics", _latex, data_row_label_re=r"\\textit\{",
                  inputs=[config.ROBUSTNESS_OUTPUT_DIR])
    REG.num("SuppRobustnessNumSeeds", _robustness_n_seeds, source="seeds per robustness level")
if "_bsr_latex" in globals():
    REG.table("tab:bootstrap_reduction_metrics", _bsr_latex, data_row_label_re=r"[\d,]+ &",
              inputs=[config.BOOTSTRAP_REDUCTION_OUTPUT_DIR])
    REG.require(config.BOOTSTRAP_REDUCTION_OUTPUT_DIR, "bootstrap-count-reduction sweep (Supp Table S5)",
                expect_datasets=set(datasets), found_datasets=set(datasets) - set(_bsr_stale),
                regen=_sweep_cmds["bootstrap_reduction"])
if "_fit_number_latex" in globals():
    REG.table("tab:fit_number_comparison", _fit_number_latex, data_row_label_re=r"\d+ &",
              inputs=[config.FIT_NUMBER_COMPARISON_SUMMARY_CSV])
    REG.require(config.FIT_NUMBER_COMPARISON_SUMMARY_CSV, "restart-count sweep (Supp Table S6)",
                expect_datasets=set(datasets), found_datasets=set(datasets) - set(_rs_stale),
                regen=_sweep_cmds["restart_sweep"])
if "_splice_ablation_fixed_pop_latex" in globals():
    REG.table("tab:splice_ablation_fixed_population", _splice_ablation_fixed_pop_latex,
              inputs=[config.SPLICE_ABLATION_ROOT])

# %% [markdown]
# ### A11g. SpliceAI-ablation input staleness (Supplementary Table S7)
#
# Separate from A11d/e so a partial rebuild (scripts/rebuild_sections.py)
# can recheck it without reloading every condition's outputs. Coverage =
# calibrated datasets with a calibration in every threshold condition;
# staleness = datasets whose inputs changed since the ablation ran
# (analysis/sweep_staleness.py SIDE_RUN_CHANGED_DATASETS) and that have not
# been refit since.

# %%
from analysis.sweep_staleness import side_run_stale
_splice_root = Path(config.SPLICE_ABLATION_ROOT)
if _splice_root.exists():
    _splice_conditions = sorted(p.name for p in _splice_root.iterdir()
                                if p.is_dir() and (p.name == "keep_all" or p.name.startswith("thresh_")))
    _splice_found = {d for d in datasets
                     if all(any((_splice_root / c / d).glob("*_calibration.json")) for c in _splice_conditions)}
    _splice_stale = side_run_stale(_splice_root, datasets, conditions=_splice_conditions)
    REG.require(_splice_root, "SpliceAI-threshold ablation (Supp Table S7)",
                expect_datasets=set(datasets), found_datasets=_splice_found - set(_splice_stale),
                regen="python analysis/build_splice_ablation_jobs.py (HPC refits per threshold) for "
                      f"{sorted(_splice_stale)}, then run the generated jobs")
else:
    REG.require(_splice_root, "SpliceAI-threshold ablation (Supp Table S7)")

# %% [markdown]
# ## A12. Supplementary Data 1-3
#
# SD1 dataset summaries (calibration ranges, priors, sample sizes, Yang
# distances, curated metadata), SD2 model-fit visualizations for every dataset,
# SD3 OddsPath evidence (with "Not available" rows for datasets the OddsPath
# approach could not evaluate). SD4 (All of Us) is a collaborator file.

# %%
from analysis import extended_data_appendix

# dataset_list=datasets (section 1's own list) instead of the module default
# (None -> _default_dataset_list_full, which re-discovers datasets from
# OUTPUT_DIR itself and applies its own datasets_to_exclude.pkl + hardcoded
# TP53/F9 exclusions) -- keeps the appendix in sync with whatever's actually
# in scope for the rest of this notebook, with no separate exclusion list.
extended_data_appendix.build_appendix_pdf(
    dataset_list=datasets,
    output_path=SUPP_DATA_DIR / "SD2_model_fit_visualizations.pdf",
    plot_thresholds=True,
)

# %%
from analysis.supplementary_data import build_sd1, build_sd3

_yang_df = pd.read_csv(config.EXCALIBR_DATASETS_TABLE_CSV) if Path(config.EXCALIBR_DATASETS_TABLE_CSV).exists() else None
from analysis.yang_distance import heldout_yang_table
_yang_heldout_df = heldout_yang_table() if Path(config.YANG_HELDOUT_CSV).exists() else None
sd1_df, _sd1_problems = build_sd1(datasets, tree, model_selections, dataset_configs, df_primary,
                                  df_all_groups=df_all_groups, yang_df=_yang_df, yang_heldout_df=_yang_heldout_df)
sd1_df.to_csv(SUPP_DATA_DIR / "SD1_dataset_summaries.csv", index=False)
print(f"  SD1: {len(sd1_df)} rows -> {SUPP_DATA_DIR / 'SD1_dataset_summaries.csv'}")
for _p in _sd1_problems:
    REG.check("sd1", False, _p, "Supplementary Data 1")
if not _sd1_problems:  # register a pass too, so a partial rebuild replaces earlier failures
    REG.check("sd1", True, "no problems", "Supplementary Data 1")

REG.figure("supp_data_2", SUPP_DATA_DIR / "SD2_model_fit_visualizations.pdf", inputs=[PRECOMPUTED_FITS])

sd3_df, _sd3_problems = build_sd3(datasets)
sd3_df.to_csv(SUPP_DATA_DIR / "SD3_oddspath_evidence.csv", index=False)
print(f"  SD3: {len(sd3_df)} rows ({int((sd3_df['Evidence Code Normal'] == 'Not available').sum())} 'Not available' normal) "
      f"-> {SUPP_DATA_DIR / 'SD3_oddspath_evidence.csv'}")
for _p in _sd3_problems:
    REG.check("sd3", False, _p, "Supplementary Data 3")
if not _sd3_problems:  # register a pass too, so a partial rebuild replaces earlier failures
    REG.check("sd3", True, "no problems", "Supplementary Data 3")
# Gap rows (dataset absent from the OddsPath CSV: both odds "Not available")
# must number \NumDatasetsBrnichNA (registered in A6, or carried over).
_sd3_gap_rows = int(((sd3_df["OddsNormal"] == "Not available") & (sd3_df["OddsAbnormal"] == "Not available")).sum())
REG.check("sd3-gap-matches-macro", _sd3_gap_rows == REG.macros["NumDatasetsBrnichNA"].value,
          f"SD3 has {_sd3_gap_rows} gap rows; \\NumDatasetsBrnichNA = {REG.macros['NumDatasetsBrnichNA'].value}",
          "Supplementary Data 3")
# Supplementary Data 4: the collaborators' All of Us workbook (cohort and
# phenotype definitions, exclusions) plus the OR results the supplementary OR
# figure plots, with column definitions (supplementary_data.build_sd4).
from analysis.supplementary_data import build_sd4
REG.require(config.AOU_SD4_COLLABORATOR_XLSX, "All of Us workbook (Supp Data 4; collaborator)")
_sd4_problems = build_sd4(SUPP_DATA_DIR / "SD4_AoU_biobank_validation.xlsx")
print(f"  SD4 -> {SUPP_DATA_DIR / 'SD4_AoU_biobank_validation.xlsx'}")
for _p in _sd4_problems:
    REG.check("sd4", False, _p, "Supplementary Data 4")
if not _sd4_problems:
    REG.check("sd4", True, "collaborator sheets + OR results", "Supplementary Data 4")
REG.static("Figures_preautomation/ross_figs/fig1_cartoon.png", "Fig. 1 conceptual illustration; hand-made, maintained on Overleaf")

# %% [markdown]
# ## A13. Zenodo data record: bootstrap fits and splits (Data availability)
#
# Writes config.ZENODO_DATA_DIR with the regular and skew-locked bootstrap
# fits (restricted to the calibrated datasets), each dataset's sample order,
# and a README with structure and checksums (analysis/data_deposit.py). The
# bootstrap splits are not deposited: the released code regenerates them
# exactly (master seed 0), which is checked here against config.SPLITS_FILE
# for every dataset. Uploading to Zenodo is manual.

# %%
import analysis.data_deposit as _deposit

_zen_inputs = [config.SPLITS_FILE, config.PRECOMPUTED_FITS, config.SKEW_LOCKED_BOOTSTRAP_RESULTS, config.DATASET_TSV]
for _f, _what in zip(_zen_inputs, ["bootstrap splits", "regular bootstrap fits", "skew-locked bootstrap fits",
                                   "master integrated variant-effect TSV"]):
    REG.require(_f, f"{_what} (Zenodo data record)")
_zen_files = _deposit.FILES + ["README.md"]
_zen_res, _zen_hit = cached("zenodo_data", lambda: _deposit.build_deposit(datasets, config.ZENODO_DATA_DIR),
                            inputs=_zen_inputs, code=[_deposit], key=sorted(datasets))
if _zen_hit and not all((Path(config.ZENODO_DATA_DIR) / f).exists() for f in _zen_files):
    _zen_res = _deposit.build_deposit(datasets, config.ZENODO_DATA_DIR)  # files removed since the cached run
for _p in _zen_res["problems"]:
    REG.check("zenodo-data", False, _p, "main.tex Data availability (Zenodo data record)")
if not _zen_res["problems"]:
    REG.check("zenodo-data", True, "regenerated splits match production for every dataset; fits for every dataset",
              "main.tex Data availability (Zenodo data record)")
for _f, _size in _zen_res["files"].items():
    print(f"  Zenodo data: {Path(config.ZENODO_DATA_DIR) / _f} ({_size / 1e6:,.1f} MB)")

# %% [markdown]
# # Part B — IGVF Pillar Project (other paper)
#
# Figure 4 driver, RAD51D/XRCC2/BARD1 extra fits, and every F9/TP53-excluded
# output (F9 and TP53 use separate classifier models to integrate multiple
# datasets in that paper). Outputs: `pillar_project_figures/`.

# %%
if not config.MANUSCRIPT_ONLY:
    # ExCALIBR vs. author, F9/TP53 excluded -- same F9/TP53 special-casing
    # section 10 applies to the manuscript summary numbers (those two genes use
    # separate classifier models to integrate multiple datasets), applied here
    # to the primary confusion panel itself rather than only the headline
    # numbers. Gene is derived from the dataset name the same way section 10
    # and _build_all_variant_groups_for_dataset already do.
    F9_TP53_GENES = {"F9", "TP53"}
    _dataset_genes = [d.split("_")[0] for d in datasets]
    _keep_idx = [i for i, g in enumerate(_dataset_genes) if g not in F9_TP53_GENES]
    datasets_no_f9_tp53 = [datasets[i] for i in _keep_idx]
    conf_no_f9_tp53 = [conf_by_method[primary_method][i] for i in _keep_idx]
    auth_no_f9_tp53 = [auth_by_method[primary_method][i] for i in _keep_idx]
    vus_no_f9_tp53 = [vus_by_method[primary_method][i] for i in _keep_idx]
    auth_vus_no_f9_tp53 = [auth_vus_by_method[primary_method][i] for i in _keep_idx]

    if any(a is not None for a in auth_no_f9_tp53):
        make_confusion_figure(
            danzs_m1=conf_no_f9_tp53, danzs_m2=auth_no_f9_tp53,
            dataset_names=datasets_no_f9_tp53, label1=primary_method, label2="author",
            figure_dir=FIGURE_SUBDIRS["pillar_project"], filename="excalibr_vs_author_no_f9_tp53.png",
            vus_coverages_m1=vus_no_f9_tp53,
            vus_coverages_m2=auth_vus_no_f9_tp53,
        )

# %%
if not config.MANUSCRIPT_ONLY:
    # Gene-deduplicated ExCALIBR-vs-author, F9/TP53 excluded -- same rationale
    # as the regular F9/TP53-excluded panel above, applied to the gene-deduped
    # scope instead.
    deduped_variants_no_f9_tp53 = deduped_variants[~deduped_variants["gene"].isin(F9_TP53_GENES)]
    deduped_variants_with_author_no_f9_tp53 = restrict_to_genes_with_author_data(deduped_variants_no_f9_tp53)
    deduped_matrix_for_author_no_f9_tp53 = build_deduped_confusion_matrix(deduped_variants_with_author_no_f9_tp53)
    deduped_author_matrix_no_f9_tp53 = build_deduped_author_confusion_matrix(deduped_variants_with_author_no_f9_tp53)
    deduped_vus_for_author_no_f9_tp53 = build_vus_coverage(deduped_variants_with_author_no_f9_tp53, points_col="points")
    deduped_author_vus_no_f9_tp53 = build_author_vus_coverage(deduped_variants_with_author_no_f9_tp53)

    if deduped_matrix_for_author_no_f9_tp53 is not None and deduped_author_matrix_no_f9_tp53 is not None:
        make_confusion_figure(
            danzs_m1=[deduped_matrix_for_author_no_f9_tp53], danzs_m2=[deduped_author_matrix_no_f9_tp53],
            dataset_names=["gene_deduped"], label1=primary_method, label2="author",
            figure_dir=FIGURE_SUBDIRS["pillar_project"], filename="excalibr_vs_author_gene_deduped_no_f9_tp53.png",
            vus_coverages_m1=[deduped_vus_for_author_no_f9_tp53],
            vus_coverages_m2=[deduped_author_vus_no_f9_tp53],
        )
    else:
        print("  SKIP gene-deduplicated (F9/TP53 excluded) ExCALIBR-vs-author figure: no matrix on one or both sides")

# %%
if not config.MANUSCRIPT_ONLY:
    # Same aggregate report + LaTeX table, F9/TP53 excluded -- reuses the
    # conf_no_f9_tp53/auth_no_f9_tp53/datasets_no_f9_tp53 lists built in 3a1
    # above, no new matrix construction.
    _auth_pairs_no_f9_tp53 = [
        (d, a, n) for d, a, n in zip(conf_no_f9_tp53, auth_no_f9_tp53, datasets_no_f9_tp53)
        if d is not None and a is not None
    ]
    if _auth_pairs_no_f9_tp53:
        _danzs_a, _auths_a, _names_a = zip(*_auth_pairs_no_f9_tp53)
        danz_agg_no_f9_tp53, auth_agg_no_f9_tp53, _ = print_aggregate_performance(
            list(_danzs_a), list(_auths_a), list(_names_a),
        )
        _latex_no_f9_tp53 = latex_performance_table_clinvar([("All variants", danz_agg_no_f9_tp53, auth_agg_no_f9_tp53)])
        save_latex_table(_latex_no_f9_tp53, FIGURE_SUBDIRS["pillar_project_tables"] / "author_clinvar_performance_no_f9_tp53.tex")
    else:
        print("  SKIP aggregate performance report (F9/TP53 excluded): no datasets with both matrices")
    # Same aggregate report + LaTeX table, gene-deduplicated with F9/TP53
    # excluded -- reuses the matrices built in 3a1c's F9/TP53-excluded panel.
    if deduped_matrix_for_author_no_f9_tp53 is not None and deduped_author_matrix_no_f9_tp53 is not None:
        danz_agg_deduped_no_f9_tp53, auth_agg_deduped_no_f9_tp53, _ = print_aggregate_performance(
            [deduped_matrix_for_author_no_f9_tp53], [deduped_author_matrix_no_f9_tp53], ["gene_deduped"],
        )
        _latex_deduped_no_f9_tp53 = latex_performance_table_clinvar(
            [("All variants", danz_agg_deduped_no_f9_tp53, auth_agg_deduped_no_f9_tp53)],
        )
        save_latex_table(
            _latex_deduped_no_f9_tp53, FIGURE_SUBDIRS["pillar_project_tables"] / "author_clinvar_performance_gene_deduped_no_f9_tp53.tex",
        )
    else:
        print("  SKIP aggregate performance report (gene-deduped, F9/TP53 excluded): no matrix on one or both sides")

# %% [markdown]
# ## B1. Pillar Project Figure 4
#
# Ported from `test/auxiliary_fig_creation/`. Some panels need external,
# non-pipeline comparison data (REVEL/AM/MutPred2 thresholds, OR estimates) —
# see `analysis/config.py`; those cells print a warning and skip if the
# corresponding path doesn't exist, rather than failing the whole run.


# %%
if not config.MANUSCRIPT_ONLY:
    from analysis.figure4 import driver as figure4_driver

    # Reuse the confusion matrices already built in section 3a1 (F9/TP53
    # excluded, non-deduplicated -- variant-effect-measurement level, not
    # 3a1c's gene-deduplicated F9/TP53-excluded variant) instead of having
    # build_figure4 re-discover pipeline output, re-load every variants CSV,
    # and rebuild author labels from scratch.
    figure4_driver.build_figure4(
        output_dir=OUTPUT_DIR, figure_dir=FIGURE_SUBDIRS["pillar_project"],
        danzs_oob=conf_no_f9_tp53,
        auths_oob=auth_no_f9_tp53,
        dataset_names=datasets_no_f9_tp53,
        # `partner=`: the matrices passed alongside these percentages are pooled
        # by plot_aggregate_confusion_matrices, which drops any dataset where
        # either side is None -- keep the VUS percentages on that same subset.
        vus_pct_danz=_aggregate_coverage_pct(vus_no_f9_tp53, partner=auth_vus_no_f9_tp53),
        vus_pct_auth=_aggregate_coverage_pct(auth_vus_no_f9_tp53, partner=vus_no_f9_tp53),
    )

# %% [markdown]
# ## B2. RAD51D/XRCC2/BARD1 extra fit plots
#
# Supplementary to Figure 4 in the legacy script, but not one of its panels --
# kept separate (`analysis.extra_gene_fits`) so `analysis/figure4/driver.py`
# can be handed to someone reproducing just Figure 4 without also needing
# these. `build_extra_gene_fits` is dataset-agnostic; the dataset list is
# specified here. BARD1 uses `minimal=True` (matching the legacy script),
# which hides some panel elements shown for the other two.


# %%
if not config.MANUSCRIPT_ONLY:
    from analysis.extra_gene_fits import build_extra_gene_fits

    build_extra_gene_fits(
        OUTPUT_DIR, DATASET_TSV, PRECOMPUTED_FITS, DATASET_CONFIGS_PATH,
        FIGURE_SUBDIRS["pillar_project"],
        dataset_specs=[
            ("RAD51D_IGVF", "rad51d"),
            ("XRCC2_IGVF", "xrcc2"),
            ("BARD1_IGVF", "bard1", True),
        ],
    )

# %%
if not config.MANUSCRIPT_ONLY:
    # Second copy with F9/TP53 datasets dropped -- same `datasets` list filtered
    # by gene prefix, for whichever downstream use (e.g. sharing without the
    # separate-classifier-model F9/TP53 pages) wants them excluded.
    _datasets_no_f9_tp53_appendix = [d for d in datasets if d.split("_")[0] not in ("F9", "TP53")]
    extended_data_appendix.build_appendix_pdf(
        dataset_list=_datasets_no_f9_tp53_appendix,
        output_path=FIGURE_SUBDIRS["pillar_project"] / "extended_data_appendix_no_f9_tp53.pdf",
        plot_thresholds=True,
    )

# %% [markdown]
# # Part C — IGVF data release tables

# %% [markdown]
# ## C1. Variant-assay / variant-aggregate / predictor-only evidence tables
#
# Reshapes `df_all_groups` (section 10's own in-bag, keep_mask-independent
# population) into the two CSVs `test/create_all_variant_evidence_csv.ipynb`
# / `test/make_variant_aggregate_evidence_table.ipynb` used to hand-roll from
# ad hoc pickle/glob-loaded external files:
#
# - **variant-assay**: one row per (variant, dataset) -- every variant every
#   assay measured, in-bag only (no OOB anywhere in these tables -- see
#   `analysis/all_variant_evidence.py`'s module docstring for why).
# - **variant-aggregate**: one row per physical variant, collapsing the
#   variant-assay rows across assays (abs-max-if-same-sign merge; assays that
#   disagree in sign are dropped as conflicting rather than reported as
#   zero/no-evidence).
#
# Also builds a **predictor-only** table (AlphaMissense/MutPred2/REVEL
# calibration scores, stripped of the stale functional-evidence columns baked
# into `config.PREDICTOR_EVIDENCE_SOURCE_CSV`) meant to be joined against the
# variant-aggregate table above later (not done here) instead of the older
# functional evidence that source CSV already carries.


# %%
if not config.MANUSCRIPT_ONLY:
    from analysis.all_variant_evidence import (
        build_dataframe_with_points, build_predictor_only_table,
        build_variant_aggregate_table, build_variant_assay_table,
        merge_variant_aggregate_with_predictors,
    )
    from analysis.discovery import load_master_df

    # One row per INPUT dataframe row, not per measured variant -- an aa-level
    # measurement contributes a row at each of its nucleotide routes, so the
    # table can be joined on coordinates without silently missing ~2/3 of them.
    # `df_all_groups` above keeps its own measurement grain (the "variant effect
    # measurements" count) untouched.
    df_input_master = load_master_df(str(DATASET_TSV))
    variant_assay_df = build_variant_assay_table(df_all_groups, df_input=df_input_master)
    variant_assay_path = FIGURE_SUBDIRS["tables"] / "variant_assay_specific_evidence.csv.gz"
    variant_assay_df.to_csv(variant_assay_path, index=False, compression="gzip")
    print(f"Variant-assay table: {len(variant_assay_df):,} entries, "
          f"{variant_assay_df['dataset'].nunique()} datasets, "
          f"{variant_assay_df['gene_symbol'].nunique()} genes -> {variant_assay_path}")

    variant_aggregate_df = build_variant_aggregate_table(variant_assay_df)
    variant_aggregate_path = FIGURE_SUBDIRS["tables"] / "variant_aggregated_functional_evidence.csv.gz"
    variant_aggregate_df.to_csv(variant_aggregate_path, index=False, compression="gzip")
    print(f"Variant-aggregate table: {len(variant_aggregate_df):,} unique variants, "
          f"{variant_aggregate_df['gene_symbol'].nunique()} genes -> {variant_aggregate_path}")

    df_with_points = build_dataframe_with_points(df_input_master, variant_assay_df)
    df_with_points_path = FIGURE_SUBDIRS["tables"] / "dataframe_with_points.csv.gz"
    df_with_points.to_csv(df_with_points_path, index=False, compression="gzip")
    print(f"Dataframe-with-points: {len(df_with_points):,} rows "
          f"({int(df_with_points['excalibr_points'].notna().sum()):,} with evidence) -> {df_with_points_path}")

    if config.warn_if_missing(config.PREDICTOR_EVIDENCE_SOURCE_CSV, "predictor-only evidence table"):
        predictor_only_df = pd.DataFrame()
    else:
        predictor_only_df = build_predictor_only_table()
        predictor_only_path = FIGURE_SUBDIRS["tables"] / "predictor_only_evidence.csv.gz"
        predictor_only_df.to_csv(predictor_only_path, index=False, compression="gzip")
        print(f"Predictor-only table: {len(predictor_only_df):,} rows -> {predictor_only_path}")

        variant_aggregate_with_predictors_df = merge_variant_aggregate_with_predictors(
            variant_aggregate_df, predictor_only_df,
        )
        variant_aggregate_with_predictors_path = (
            FIGURE_SUBDIRS["tables"] / "variant_aggregated_experimental_predictive_evidence.csv.gz"
        )
        variant_aggregate_with_predictors_df.to_csv(
            variant_aggregate_with_predictors_path, index=False, compression="gzip",
        )
        print(f"Variant-aggregate x predictor table: {len(variant_aggregate_with_predictors_df):,} rows "
              f"-> {variant_aggregate_with_predictors_path}")

# %% [markdown]
# # Part D — diagnostics and exploratory outputs
#
# Not quoted in the manuscript; kept so nothing the notebook used to produce
# is lost. Skipped when `MANUSCRIPT_ONLY=1`.

# %% [markdown]
# ## D1. Per-dataset FP / FN breakdown (ranked)
#
# Raw false-positive (B/LB called Abnormal/Pathogenic) and false-negative
# (P/LP called Normal/Benign) counts per dataset, read directly off the same
# `conf_by_method[primary_method]` matrices built in section 3 (rows
# [BLB,PLP] x cols [Normal,IR,Abnormal] -- see `build_confusion_matrix`) --
# no recomputation. Ranked descending so the worst-offending datasets surface
# first; ties broken by the other column.


# %%
if not config.MANUSCRIPT_ONLY:
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
# ## D2. Per-method evidence distributions


# %%
if not config.MANUSCRIPT_ONLY:
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
# ## D3. Per-dataset calibration figures
#
# These already exist — `run_igvf_batch.py`/`run_pipeline.py` write
# `{dataset}_{comp}_visualization.png` for every dataset/component during the
# original run (via `generate_visualizations`/`plot_scoreset_best_config`).
# No regeneration needed; just display one as a sanity check that the run's
# own output is where you expect it.


# %%
if not config.MANUSCRIPT_ONLY:
    from IPython.display import Image, display

    _example_dataset = datasets[0]
    _vis_candidates = sorted((OUTPUT_DIR / _example_dataset).glob(f"{_example_dataset}_*_visualization.png"))
    if _vis_candidates:
        print(f"Existing visualizations for {_example_dataset}: {[p.name for p in _vis_candidates]}")
        display(Image(filename=str(_vis_candidates[0])))
    else:
        print(f"  No existing visualization.png found for {_example_dataset} under {OUTPUT_DIR / _example_dataset}")

# %% [markdown]
# ## D4. Legacy dataset description table


# %%
if not config.MANUSCRIPT_ONLY:
    # analysis.gene_table.build_dataset_table expects metadata columns the
    # current master TSV no longer has, and has raised on every input since the
    # pipeline refactor; Supplementary Data 1 (A12) supersedes it. Kept, guarded,
    # so Run All continues past it.
    from analysis.gene_table import build_dataset_table

    try:
        dataset_table = build_dataset_table()
        dataset_table.head()
    except Exception as _e:  # noqa: BLE001
        print(f"  SKIP legacy dataset table (superseded by Supplementary Data 1): {type(_e).__name__}: {_e}")

# %% [markdown]
# ## D5. Per-scoreset VUS evidence breakdown, ranked by indeterminate count
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
if not config.MANUSCRIPT_ONLY:
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

# %% [markdown]
# ## D6. Headline summary printouts (all genes / excluding F9, TP53, SFPQ)
#
# Computes the headline numbers behind the ExCALIBR Results paragraph:
# genes reached, overall benign/pathogenic evidence totals, ExCALIBR-vs-
# functional-annotation agreement with ClinVar controls, VUS coverage of the
# indeterminate range, and the OddsPath-approach coverage gap. Printed twice:
# once for every gene in this pipeline run ("all genes" -- still excludes
# `DEFAULT_EXCLUDED_DATASETS`' F9_Popp_2025_model/TP53_Fayer_2021_meta_
# clinvar_2018/SFPQ_IGVF, same as every other section in this notebook, since
# those never make it into `df` at all -- see analysis/discovery.py), and
# once with `EXCLUDED_GENES` (F9/TP53 use separate classifier models to
# integrate multiple datasets; SFPQ remains a candidate disease gene) fully
# dropped -- biobank/All-of-Us association numbers are out of scope entirely
# (nothing in this pipeline supports them).
#
# **In-bag vs OOB.** Every number in this section uses in-bag evidence
# (`standard_points`) ONLY -- gene counts, variant-effect-measurement and
# unique-variant totals, and the OddsPath-gap datasets. The two exceptions,
# tagged `[OOB]` in the printed output, are the ClinVar-agreement accuracy
# and the VUS indeterminate-range coverage, which are deliberately OOB
# (that's the whole point of holding variants out). Letting OOB points leak
# into the rest inflates the headline gene count (36 rather than 34: two
# genes clear zero evidence only via OOB) and can add OOB-only datasets to
# the OddsPath gap.
#
# Both OOB statistics are pooled over the datasets/genes where ExCALIBR and
# the author functional annotation BOTH have data, never over each side's
# own subset -- `compute_aggregate_metrics` and
# `plot_aggregate_confusion_matrices` already drop a dataset when either
# matrix is None, `restrict_to_genes_with_author_data` does the same at the
# gene level for the deduplicated branch, and `_aggregate_coverage_pct`'s
# `partner=` argument does it for the VUS coverage percentages.
#
# `_compute_manuscript_summary`'s `excluded_genes` argument only ever derives
# new, function-local variables -- it never filters or mutates
# `deduped_variants`, `df_primary_dedup`, or any object shared with the
# figures/matrices built earlier in this notebook, which keep including
# every gene exactly as before.


# %%
if not config.MANUSCRIPT_ONLY:
    from src.assay_calibration.plot_utils.utils import compute_aggregate_metrics

    EXCLUDED_GENES = {"F9", "TP53", "SFPQ"}


    def _compute_manuscript_summary(excluded_genes: set, label: str, use_gene_dedup: bool = True) -> dict:
        # -- 1. Gene scope (OOB-free population; see step 2) ----------------------
        # `deduped_scope` is the OOB-scored kept+VUS population -- steps 3/4 below
        # need it (ClinVar agreement / VUS coverage are deliberately OOB), but the
        # gene counts must not come from it: every headline number in this section
        # other than those two statistics is in-bag only, and two genes clear zero
        # evidence solely via OOB points. The counts themselves are therefore
        # computed in step 2 from `df_all_scope`/`deduped_all_scope`.
        deduped_scope = deduped_variants[~deduped_variants["gene"].isin(excluded_genes)]

        # -- 2. Overall variant effect measurement / unique variant totals --------
        # Uses df_all_groups (every variant the assay measured), not df_primary_dedup
        # (which only carries the kept sample-group + VUS-only subset) -- see the
        # markdown cell above. standard_points only (no OOB available for most of
        # this population).
        df_all_scope = df_all_groups[~df_all_groups["gene"].isin(excluded_genes)]
        n_vem_total = len(df_all_scope)
        n_vem_benign = int((df_all_scope["standard_points"] < 0).sum())
        n_vem_pathogenic = int((df_all_scope["standard_points"] > 0).sum())
        deduped_all_scope = build_gene_deduped_variants(df_all_scope, use_oob=False)
        n_unique_variants = len(deduped_all_scope)
        n_unique_variants_with_evidence = int((deduped_all_scope["points"] != 0).sum())
        # Gene counts from this same in-bag, all-variants-the-assay-measured frame
        # (not just VUS or the four fitting sample groups), so "N unique variants
        # across M genes" quotes one population rather than two.
        n_genes_summary = df_all_scope["gene"].nunique()
        n_genes_with_evidence = int(deduped_all_scope.loc[deduped_all_scope["points"] != 0, "gene"].nunique())

        # Kept sample-group + VUS-only population (df_primary_dedup, with real
        # OOB points) -- used below (step 5) to decide whether ExCALIBR's actual
        # evidence-assignment pipeline reached a dataset, and (step 3/4) via
        # deduped_variants for ClinVar-agreement/VUS-coverage, which only make
        # sense for ClinVar-labeled variants.
        df_scope = df_primary_dedup.copy()
        df_scope["gene"] = df_scope["dataset"].str.split("_").str[0]
        df_scope = df_scope[~df_scope["gene"].isin(excluded_genes)]

        # -- 3/4. ExCALIBR vs functional-annotation agreement with ClinVar
        # controls, and VUS indeterminate-range coverage (ExCALIBR vs author) ------
        if use_gene_dedup:
            # Variant-level, gene-deduplicated metrics (one row per unique
            # physical variant, section 3a1c's population).
            deduped_scope_with_author = restrict_to_genes_with_author_data(deduped_scope)
            excalibr_matrix_summary = build_deduped_confusion_matrix(deduped_scope_with_author)
            author_matrix_summary = build_deduped_author_confusion_matrix(deduped_scope_with_author)
            if excalibr_matrix_summary is not None and author_matrix_summary is not None:
                danz_agg_summary, auth_agg_summary, _ = compute_aggregate_metrics(
                    danzs=[excalibr_matrix_summary], auths=[author_matrix_summary],
                    dataset_names=["gene_deduped_scope"],
                )
                excalibr_agreement_pct = danz_agg_summary["accuracy"] * 100
                author_agreement_pct = auth_agg_summary["accuracy"] * 100
            else:
                excalibr_agreement_pct = None
                author_agreement_pct = None

            excalibr_vus_summary = build_vus_coverage(deduped_scope_with_author, points_col="points")
            author_vus_summary = build_author_vus_coverage(deduped_scope_with_author)

            def _pct_indeterminate(coverage):
                if coverage is None:
                    return None
                n_determinate, n_vus = coverage
                return 100 * (n_vus - n_determinate) / n_vus if n_vus else None

            pct_indeterminate_excalibr = _pct_indeterminate(excalibr_vus_summary)
            pct_indeterminate_author = _pct_indeterminate(author_vus_summary)
        else:
            # Variant-effect-measurement statistics: reuses section 3's own
            # per-dataset conf_by_method/auth_by_method/vus_by_method/
            # auth_vus_by_method lists (one entry per assay's copy of a variant,
            # not deduplicated to one row per physical variant), restricted to
            # this call's gene scope by dataset name -- same
            # dataset.split("_")[0] convention used above and in
            # _build_all_variant_groups_for_dataset.
            _dataset_genes_summary = [d.split("_")[0] for d in datasets]
            _keep_idx_summary = [i for i, g in enumerate(_dataset_genes_summary) if g not in excluded_genes]
            _conf_scope = [conf_by_method[primary_method][i] for i in _keep_idx_summary]
            _auth_scope = [auth_by_method[primary_method][i] for i in _keep_idx_summary]
            _datasets_scope_names = [datasets[i] for i in _keep_idx_summary]
            _vus_scope = [vus_by_method[primary_method][i] for i in _keep_idx_summary]
            _auth_vus_scope = [auth_vus_by_method[primary_method][i] for i in _keep_idx_summary]

            if any(a is not None for a in _auth_scope):
                danz_agg_summary, auth_agg_summary, _ = compute_aggregate_metrics(
                    danzs=_conf_scope, auths=_auth_scope, dataset_names=_datasets_scope_names,
                )
                excalibr_agreement_pct = danz_agg_summary["accuracy"] * 100
                author_agreement_pct = auth_agg_summary["accuracy"] * 100
            else:
                excalibr_agreement_pct = None
                author_agreement_pct = None

            # `partner=`: build_author_vus_coverage returns None strictly more
            # often than build_vus_coverage (missing auth_label, no ClinVar
            # controls, all-indeterminate author calls), so aggregating each list
            # independently would give ExCALIBR a denominator containing datasets
            # the author side never saw. Pair them, matching what
            # compute_aggregate_metrics above already does to the matrices.
            _excalibr_det_pct = _aggregate_coverage_pct(_vus_scope, partner=_auth_vus_scope)
            _author_det_pct = _aggregate_coverage_pct(_auth_vus_scope, partner=_vus_scope)
            pct_indeterminate_excalibr = 100 - _excalibr_det_pct if _excalibr_det_pct is not None else None
            pct_indeterminate_author = 100 - _author_det_pct if _author_det_pct is not None else None

        # -- 5. OddsPath-gap datasets ----------------------------------------------
        # Every dataset in scope IS calibrated: df_primary_dedup only carries
        # datasets ExCALIBR produced a calibration for (discover_outputs ->
        # load_all_variants, which skips a dataset with no calibration JSON).
        # This deliberately does NOT test whether any variant reached a nonzero
        # evidence point -- a calibration that assigns only indeterminate
        # (0-point) evidence is still a calibration, and the OddsPath comparison
        # asks which datasets the established approach could calibrate AT ALL,
        # not which ones cleared an evidence threshold. Because no points are
        # consulted, the in-bag/OOB distinction doesn't arise here at all.
        calibrated_datasets = sorted(df_scope["dataset"].unique())

        n_additional = 0
        n_vem_additional = 0
        n_variants_additional = 0
        n_variants_additional_with_evidence = 0
        if not config.warn_if_missing(config.OP_EVIDENCE_CODES_CSV, "OddsPath evidence-code CSV (manuscript summary)"):
            from analysis.supplementary_data import load_oddspath_csv
            df_op_summary = load_oddspath_csv()
            # OP_EVIDENCE_CODES_CSV spells datasets without the pipeline's
            # "_clinvar_2018" suffix (e.g. "BRCA1_Findlay_2018", "MSH2_Jia_2021",
            # "TP53_Fortuno_2021") and can differ by the same old/frozen-name
            # renames the master TSV does -- reuse resolve_dataset_tsv_name's
            # canonical resolution chain (as-is -> new_dataset_names.csv ->
            # DATASET_RENAMES) against this CSV instead of a bare `in` check,
            # which would otherwise mistake 8 already-covered _clinvar_2018
            # datasets (BRCA1_Adamovich x2, BRCA1_Findlay_2018, MSH2_Jia_2021,
            # PTEN_Matreyek_2018, PTEN_Mighell_2018, TP53_Fortuno_2021,
            # TP53_Funk_2025) for OddsPath coverage gaps.
            additional_datasets = [
                d for d in calibrated_datasets
                if resolve_dataset_tsv_name(d.replace("_clinvar_2018", ""), df_op_summary, config.DATASET_TSV) is None
            ]
            n_additional = len(additional_datasets)
            df_additional = df_all_scope[df_all_scope["dataset"].isin(additional_datasets)]
            n_vem_additional = len(df_additional)
            deduped_additional = build_gene_deduped_variants(df_additional, use_oob=False)
            n_variants_additional = len(deduped_additional)
            n_variants_additional_with_evidence = int((deduped_additional["points"] != 0).sum())

        # -- Summary ----------------------------------------------------------------
        print(f"\n{'=' * 80}\nMANUSCRIPT SUMMARY NUMBERS ({label})\n{'=' * 80}")
        print("(in-bag evidence throughout, except the two [OOB]-tagged statistics)")
        print(f"Genes in scope: {n_genes_summary}")
        print(f"Genes with >=1 point of pathogenic or benign evidence: {n_genes_with_evidence}/{n_genes_summary}")
        print(f"Variant effect measurements: {n_vem_total:,} total "
              f"({n_vem_benign:,} benign evidence, {n_vem_pathogenic:,} pathogenic evidence)")
        print(f"Unique variants (gene-deduplicated, aa/nt-separated): {n_unique_variants:,} "
              f"across {n_genes_summary} genes")
        print(f"  ...of which {n_unique_variants_with_evidence:,} have pathogenic or benign evidence, "
              f"across {n_genes_with_evidence} genes")
        if excalibr_agreement_pct is not None:
            print(f"Agreement with ClinVar PLP/BLB controls [OOB]: "
                  f"ExCALIBR {excalibr_agreement_pct:.1f}% vs functional annotation {author_agreement_pct:.1f}%")
        else:
            print("  SKIP agreement comparison: no author/ExCALIBR confusion matrix available in scope")
        if pct_indeterminate_excalibr is not None:
            print(f"VUS assigned to indeterminate range [OOB]: "
                  f"functional annotation {pct_indeterminate_author:.1f}% vs ExCALIBR {pct_indeterminate_excalibr:.1f}%"
                  if pct_indeterminate_author is not None else
                  f"VUS assigned to indeterminate range (ExCALIBR) [OOB]: {pct_indeterminate_excalibr:.1f}%")
        else:
            print("  SKIP VUS coverage comparison: no VUS in scope")
        print(f"Datasets calibrated by ExCALIBR but not covered by the OddsPath approach: {n_additional} "
              f"({n_vem_additional:,} variant effect measurements, {n_variants_additional:,} unique variants, "
              f"{n_variants_additional_with_evidence:,} with pathogenic or benign evidence)")

        return {
            "scope": label,
            "n_genes": n_genes_summary,
            "n_genes_with_evidence": n_genes_with_evidence,
            "n_vem_total": n_vem_total,
            "n_vem_benign": n_vem_benign,
            "n_vem_pathogenic": n_vem_pathogenic,
            "n_unique_variants": n_unique_variants,
            "n_unique_variants_with_evidence": n_unique_variants_with_evidence,
            "excalibr_agreement_pct": excalibr_agreement_pct,
            "author_agreement_pct": author_agreement_pct,
            "pct_indeterminate_excalibr": pct_indeterminate_excalibr,
            "pct_indeterminate_author": pct_indeterminate_author,
            "n_additional_oddspath_gap_datasets": n_additional,
            "n_vem_additional_oddspath_gap": n_vem_additional,
            "n_unique_variants_additional_oddspath_gap": n_variants_additional,
            "n_unique_variants_additional_oddspath_gap_with_evidence": n_variants_additional_with_evidence,
        }


    manuscript_summary_all_genes = _compute_manuscript_summary(set(), label="all genes")
    manuscript_summary_df = pd.DataFrame([
        manuscript_summary_all_genes,
        _compute_manuscript_summary(EXCLUDED_GENES, label=f"excluding {sorted(EXCLUDED_GENES)}"),
        _compute_manuscript_summary(set(), label="all genes (VEM stats)", use_gene_dedup=False),
        _compute_manuscript_summary(
            EXCLUDED_GENES, label=f"excluding {sorted(EXCLUDED_GENES)} (VEM stats)", use_gene_dedup=False,
        ),
    ])
    manuscript_summary_df

# %% [markdown]
# # Write: macros, tables, manifest, build report, flattened submission copies
#
# Replaces the old section 12 printout. `macros.tex` is sorted by each macro's
# first use in `main.tex`/`supplement.tex`; `build_report.md` lists old -> new
# values, stale inputs, table-template differences, prose claims that no
# longer hold, and Nature Genetics checks. Failures raise here in strict mode.

# %%
from analysis.manuscript_numbers import OLD_VALUES_JSON, nature_genetics_checks
from analysis.flatten_manuscript import flatten_manuscript
import json as _json

# Pre-automation literal values (snapshot taken when the .tex numbers were
# replaced by macros), for the report's old -> new column.
_old_values = _json.loads(OLD_VALUES_JSON.read_text()) if OLD_VALUES_JSON.exists() else {}
# Copy the figures the .tex includes into the Overleaf folder (figures/),
# so it holds exactly the manuscript's files.
REG.export_figures(FIGURE_DIR, MANUSCRIPT_FIGURE_DIR)
REG.write(MANUSCRIPT_FIGURE_DIR, nature_checks=nature_genetics_checks(MAIN_TEX, SUPP_TEX), old_values=_old_values,
          highlight_changed=config.HIGHLIGHT_CHANGED_NUMBERS)
_flat_problems = flatten_manuscript([MAIN_TEX, SUPP_TEX], REG.out_dir / "macros.tex", MANUSCRIPT_FIGURE_DIR,
                                    SUBMISSION_DIR)
if _flat_problems and REG.strict:
    raise RuntimeError(f"flattening problems: {_flat_problems}")
print((REG.out_dir / "build_report.md").read_text()[:4000])
