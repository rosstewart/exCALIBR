# `analysis/` — pipeline-native figure generation

All figures are generated from `run_pipeline.py` / `run_igvf_batch.py` output
plus the master integrated dataframe — no hand-curated, run-specific pickles.

**Start here:** `analysis/analyze_pipeline_output.py` is a jupytext-paired
notebook (percent format) that walks through every figure group. Open it
directly in Jupyter/VSCode, or run `jupytext --sync analyze_pipeline_output.py`
to generate/refresh the paired `.ipynb`. It also runs headless as a CLI:
`python analysis/analyze_pipeline_output.py --output-dir ... --dataset ...`.

**All paths are set in `analysis/config.py`** (or via matching `EXCALIBR_*`
env vars) — that's the only file you should need to edit to point everything
at a different run.

## Module map

| Module | What it does |
|---|---|
| `config.py` | All settable paths (output dir, dataframe, dataset configs, precomputed fits, external comparison data). |
| `discovery.py` | Discovers `*_variants.csv` / `*_calibration.json` under a pipeline output dir, resolves which (n_c, benign_method) component to use per dataset, loads everything into one long-format DataFrame. |
| `author_labels.py` | Attaches original author functional-classification labels to loaded variants. |
| `confusion.py` | Builds 2×3 confusion matrices (ClinVar / author vs. evidence direction) and plots them — thin wrapper around `src/assay_calibration/plot_utils/utils.py::plot_aggregate_confusion_matrices`. |
| `evidence.py` | Builds evidence-distribution arrays and plots them — thin wrapper around `plot_combined_evidence_distributions` / `plot_evidence_by_clinvar_class_with_stats`. |
| `scatter.py` | Per-gene accuracy scatter, method vs. method. |
| `calibration_plots.py` | Per-dataset calibration detail figures (histograms / point ranges / LR+ curves) built purely from pipeline CSV/JSON output — no fitted mixture object needed. |
| `legacy_fits.py` | Bridge: assembles `(scoreset, indv_summary, fits, score_range, ...)` from pipeline output + the master dataframe + a precomputed bootstrap-fits file, for plots that need the actual fitted mixture density (not just LR+ percentiles) — replaces `load_dataset_for_plot`'s dependency on `point_assignment_*/{dataset}/*.pkl`. |
| `yang_distance.py` | Yang-distance (p=2) bootstrap goodness-of-fit diagnostic; regenerates train/val splits on demand (deterministic given `(dataset, bootstrap_idx)`) instead of reading `dataset_splits_recovered.pkl`. |
| `figure4/` | Figure 4 (panels.py = pure plotting, driver.py = data assembly). |
| `extended_data_appendix.py` | Multi-page appendix PDF of per-dataset calibration plots. |
| `gene_performance_scatter.py` | Gene-level odds-ratio / Brnich evidence-strength comparison figures (some panels need external, non-pipeline data — see below). |
| `gene_table.py` | Dataset description/metadata summary table. |

## External, non-pipeline data

A few figures (Figure 4's REVEL/AM/MutPred2 panels, the gene-performance
odds-ratio scatter) compare ExCALIBR against other predictors or pre-computed
odds-ratio estimates from a *separate* analysis — this data is not produced
by `run_pipeline.py` and is not re-derived here. Their paths live in
`config.py` (`YILE_DIR`, `OR_ESTIMATES_CSV`, `YURIY_OR_CSV`,
`EVIDENCE_COUNTS_DIR`); any cell/function using one of these prints a warning
and skips if the path doesn't resolve, rather than failing the whole run.
