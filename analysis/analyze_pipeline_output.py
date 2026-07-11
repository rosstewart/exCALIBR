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

from analysis import config
from analysis.discovery import discover_outputs, load_all_variants, resolve_component
from analysis.author_labels import attach_author_labels
from analysis.confusion import (
    build_confusion_matrix,
    build_author_confusion_matrix,
    make_confusion_figure,
    make_single_confusion_figure,
)
from analysis.evidence import build_evidence_arrays, build_author_array, make_evidence_figure
from analysis.scatter import make_scatter_figure
from analysis.calibration_plots import load_lr_values, make_calibration_figure
from analysis.legacy_fits import load_scoreset_and_fits
from analysis.yang_distance import compute_bootstrap_yang_distances_parallel
from src.assay_calibration.plot_utils.utils import plot_scoreset_final_pillar_project_v2
from analysis.plot_common import save_and_show

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATASET_TSV = config.DATASET_TSV
DATASET_CONFIGS_PATH = config.DATASET_CONFIGS
PRECOMPUTED_FITS = config.PRECOMPUTED_FITS
FIGURE_DIR = Path(config.FIGURE_DIR)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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

    for m in methods_:
        make_single_confusion_figure(conf_by_method_[m], datasets_, label=m, figure_dir=figure_dir)

    primary_method_ = methods_[0]
    auths_ = auth_by_method_[primary_method_]
    if any(a is not None for a in auths_):
        make_confusion_figure(
            danzs_m1=conf_by_method_[primary_method_], danzs_m2=auths_,
            dataset_names=datasets_, label1=primary_method_, label2="author",
            figure_dir=figure_dir,
        )

    for method_ in methods_:
        df_m = df[df["method"] == method_]
        all_danz, all_clinvar = build_evidence_arrays(df_m)
        all_author = build_author_array(df_m)
        make_evidence_figure(all_danz, all_author, all_clinvar, label=method_, figure_dir=figure_dir)

    # Per-gene accuracy scatter: ExCALIBR vs author (not two ACMG-mapping
    # methods against each other) -- same primary_method/auths pairing as the
    # confusion figure above, just wrapped in the {label: matrices} shape
    # make_scatter_figure expects.
    if any(a is not None for a in auths_):
        make_scatter_figure(
            {primary_method_: conf_by_method_[primary_method_], "author": auths_},
            datasets_, primary_method_, "author", figure_dir=figure_dir,
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

# Calibration-vs-ClinVar confusion matrix, per method — always plotted.
for m in methods:
    make_single_confusion_figure(conf_by_method[m], datasets, label=m, figure_dir=FIGURE_DIR)

# ExCALIBR vs. author, for whichever method is primary (first discovered).
primary_method = methods[0]
auths = auth_by_method[primary_method]
if any(a is not None for a in auths):
    make_confusion_figure(
        danzs_m1=conf_by_method[primary_method], danzs_m2=auths,
        dataset_names=datasets, label1=primary_method, label2="author",
        figure_dir=FIGURE_DIR,
    )

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
    latex_performance_table_clinvar(danz_agg_metrics, auth_agg_metrics)
else:
    print("  SKIP aggregate performance report: no datasets with both ExCALIBR and author matrices")

# %% [markdown]
# ## 4. Evidence distributions and per-gene accuracy scatter

# %%
for method in methods:
    df_m = df[df["method"] == method]
    all_danz, all_clinvar = build_evidence_arrays(df_m)
    all_author = build_author_array(df_m)
    make_evidence_figure(all_danz, all_author, all_clinvar, label=method, figure_dir=FIGURE_DIR)

# Per-gene accuracy scatter: ExCALIBR vs author (not two ACMG-mapping methods
# against each other) -- same primary_method/auths pairing as the confusion
# figure above (section 3), just wrapped in the {label: matrices} shape
# make_scatter_figure expects.
if any(a is not None for a in auths):
    make_scatter_figure(
        {primary_method: conf_by_method[primary_method], "author": auths},
        datasets, primary_method, "author", figure_dir=FIGURE_DIR,
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

AUTHOR_PANEL_DATASETS = datasets    # restrict to a curated subset here if desired
CLINVAR_PANEL_DATASETS = datasets   # restrict to a curated subset here if desired

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
        label=primary_method, figure_dir=FIGURE_DIR,
    )

    gwe_author_table, gwe_clinvar_table, gwe_latex_str = compute_genewise_evidence_table(
        all_danz_oob, all_author, dataset_info_df,
        all_danz_oob_full, all_clinvar_full[:, :4], dataset_info_df_full,
    )
    print(gwe_latex_str)
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

# %%
from analysis.clingen import build_clingen_confusion, convert_3x2_to_2x3, plot_2x3_confusions_nature
from analysis.manuscript_stats import latex_performance_table_clingen

CLINGEN_DATASETS = datasets  # explicit, adjustable — narrow if desired

clingen_confusion, clingen_genes = build_clingen_confusion(
    df_primary, DATASET_TSV, CLINGEN_DATASETS, use_oob=True,
)
if clingen_genes:
    fig = plot_2x3_confusions_nature({
        'auth': convert_3x2_to_2x3(clingen_confusion['auth']),
        'excalibr': convert_3x2_to_2x3(clingen_confusion['excalibr']),
    })
    save_and_show(fig, FIGURE_DIR / "clingen_confusion.png")
    latex_performance_table_clingen({
        'excalibr': convert_3x2_to_2x3(clingen_confusion['excalibr']),
        'auth': convert_3x2_to_2x3(clingen_confusion['auth']),
    })
else:
    print("  SKIP ClinGen confusion: no genes with usable ClinGen evidence codes in scope")

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
    df_op = pd.read_csv(config.OP_EVIDENCE_CODES_CSV)
    levels = [1, 2, 4, 8]
    path_codes = ['PS3_supporting', 'PS3_moderate', 'PS3_strong', 'PS3_very_strong']
    ben_codes = ['BS3_supporting', 'BS3_moderate', 'BS3_strong', 'BS3_very_strong']
    auth_path_counts = {lvl: set() for lvl in levels}
    auth_ben_counts = {lvl: set() for lvl in levels}
    for ds in dataset_info_df_full["dataset"]:
        op_rows = df_op[df_op.Dataset == ds]
        if op_rows.empty:
            continue
        path_evidence = op_rows["Evidence Code Abnormal"].iloc[0]
        ben_evidence = op_rows["Evidence Code Normal"].iloc[0]
        for i, code in enumerate(path_codes):
            if path_evidence == code:
                for j in range(i, -1, -1):
                    auth_path_counts[2 ** j].add(ds)
                break
        for i, code in enumerate(ben_codes):
            if ben_evidence == code:
                for j in range(i, -1, -1):
                    auth_ben_counts[2 ** j].add(ds)
                break
    auth_path_counts = {k: len(v) for k, v in auth_path_counts.items()}
    auth_ben_counts = {k: len(v) for k, v in auth_ben_counts.items()}
    fig, _ = plot_evidence_comparison(
        excalibr_path_counts, excalibr_ben_counts, auth_path_counts, auth_ben_counts,
    )
    save_and_show(fig, FIGURE_DIR / "num_datasets_reach_evidence.png")

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
    save_and_show(fig, FIGURE_DIR / "points_heatmap_sort_model_system.png")

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
# ## 6. MSH2 example / "final pillar project" style figure — matches `test/plot_MSH2_ex.py`
#
# Needs the full per-bootstrap mixture fits (`analysis.config.PRECOMPUTED_FITS`),
# not just the LR+ percentile curves used above — see `analysis/legacy_fits.py`.
#
# BRCA1/MSH2/PTEN/TP53 were only ever run through the pipeline under their
# `_clinvar_2018` name (see `run_igvf_batch.py`'s `GENES_2018` auto-detection),
# so `pipeline_dataset` points there for both scoresets; `dataset` +
# `clinvar_release` control which ClinVar release labels each Scoreset itself
# is built with.

# %%
from analysis.legacy_fits import resolve_component_for

MSH2_DATASET = "MSH2_Jia_2021"
MSH2_PIPELINE_KEY = f"{MSH2_DATASET}_clinvar_2018"

try:
    n_c_msh2, benign_method_msh2 = resolve_component_for(
        MSH2_PIPELINE_KEY, output_dir=OUTPUT_DIR, dataset_configs_path=DATASET_CONFIGS_PATH,
    )
    scoreset_2018, indv_summary, fits, score_range, n_c_msh2, n_samples, flipped = load_scoreset_and_fits(
        MSH2_DATASET, output_dir=OUTPUT_DIR, dataset_tsv=DATASET_TSV,
        precomputed_fits=PRECOMPUTED_FITS, dataset_configs_path=DATASET_CONFIGS_PATH,
        pipeline_dataset=MSH2_PIPELINE_KEY, clinvar_release="2018",
        n_c=n_c_msh2, benign_method=benign_method_msh2,
    )
    scoreset_2025, _, _, _, _, _, _ = load_scoreset_and_fits(
        MSH2_DATASET, output_dir=OUTPUT_DIR, dataset_tsv=DATASET_TSV,
        precomputed_fits=PRECOMPUTED_FITS, dataset_configs_path=DATASET_CONFIGS_PATH,
        pipeline_dataset=MSH2_PIPELINE_KEY, clinvar_release="2026",
        n_c=n_c_msh2, benign_method=benign_method_msh2,
    )
    # Exact call from test/plot_MSH2_ex.py (config/n_c passed as None — see
    # the fix to plot_scoreset_final_pillar_project_v2 in src/ that derives
    # the component count from `fits` directly when n_c is None).
    fig = plot_scoreset_final_pillar_project_v2(
        MSH2_DATASET, scoreset_2018, scoreset_2025, indv_summary, fits, score_range,
        None, None, n_samples, relax=None, flipped=flipped,
    )
    save_and_show(fig, FIGURE_DIR / "MSH2_final_pillar_project_v2.png")
except (FileNotFoundError, KeyError, ValueError) as e:
    print(f"  SKIP MSH2 example figure: {e}")

# %% [markdown]
# ## 7. Yang distance bootstrap diagnostic — matches `test/yang_dist.py`

# %%
try:
    scoreset, _, fits, _, n_c, _, _ = load_scoreset_and_fits(
        MSH2_DATASET, output_dir=OUTPUT_DIR, dataset_tsv=DATASET_TSV,
        precomputed_fits=PRECOMPUTED_FITS, dataset_configs_path=DATASET_CONFIGS_PATH,
        pipeline_dataset=MSH2_PIPELINE_KEY, clinvar_release="2026",
    )
    yang_distances = compute_bootstrap_yang_distances_parallel(
        MSH2_PIPELINE_KEY, n_c, fits, scoreset, dataset_to_splits=None, n_jobs=-1,
    )
    for sample_key, dists in yang_distances.items():
        print(f"  {sample_key}: median={np.nanmedian(dists):.4f}  n={np.sum(~np.isnan(dists))}")
except (FileNotFoundError, KeyError, ValueError) as e:
    print(f"  SKIP Yang distance diagnostic: {e}")

# %% [markdown]
# ## 8. Figure 4, extended-data appendix, gene-performance/OR scatter
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
    output_dir=OUTPUT_DIR, figure_dir=FIGURE_DIR,
    danzs_oob=conf_by_method[primary_method],
    auths_oob=auth_by_method[primary_method],
    dataset_names=datasets,
)

# %%
from analysis import extended_data_appendix

extended_data_appendix.build_appendix_pdf(
    dataset_list=None,  # None -> auto-discover from OUTPUT_DIR, per module default
    output_path=FIGURE_DIR / "extended_data_appendix.pdf",
    plot_thresholds=True,
)

# %%
from analysis import gene_performance_scatter

gene_performance_scatter.build_gene_performance_scatter(output_dir=OUTPUT_DIR, figure_dir=FIGURE_DIR)

# %% [markdown]
# ## 9. Dataset description table

# %%
from analysis.gene_table import build_dataset_table

dataset_table = build_dataset_table()
dataset_table.head()
