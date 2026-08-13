# mv_analysis

Post-fit analysis for ExCALIBR-MV (multidimensional skew-normal mixture
calibration): builds each gene-set's MultiScoreset, runs
`MVCalibrationAnalysis` across every fitted mixture config (3c-6c) and
partial-pattern mode, and -- where an existing UV (univariate ExCALIBR)
baseline exists -- reports it alongside the MV numbers using identical
metric definitions, so the two are directly comparable.

Structurally this mirrors `analysis/` (the UV-only analysis package: one
`config.py` + many single-purpose modules) in spirit, but is its own
top-level package, not nested inside `analysis/` -- the two packages don't
depend on each other.

Entry point: `python run_mv_analysis.py --results-json <bootstrap_results.json.gz> --gene-set <set> [--gene <gene>]`
(a thin shim over `mv_analysis/cli.py`, kept at the repo root for backward
compatibility with existing invocations).

## Module map

| File | Purpose |
|---|---|
| `config.py` | Env-overridable path constants: the four UV calibration directories, `dataset_configs_aug_2026.json`, the TP53/integrated bridging source dataframes, and the LABEL-seq/FGFR gene lists. |
| `build.py` | Thin wrapper around `hpc/prepare.py`'s gene-set ingestion dispatch (reused, not duplicated) -- `build_ms(args) -> (gene, ms, dataset_name)`. |
| `uv_agg.py` | The two ways of combining multiple per-assay/per-predictor UV point calls into one per-variant call: `aggregate_nonconflicting` and `aggregate_max`. Moved here (generalized) from `compare_uv_mv_agg.py`, which re-exports them for backward compatibility. |
| `uv_sources.py` | Per-gene-set adapters (`load_uv_points(gene, ms, gene_set)`) that load UV calibration outputs from disk and align them to `ms`'s variant order. See the data-source table and caveats below. |
| `report.py` | `build_comparison_table(...)` -- runs the MV analysis, builds UV rows via `uv_sources` + `uv_agg`, and returns one table using `points_to_confusion` + `compute_classification_metrics` (the same metric pipeline that produces the saved `*_confusion.txt` reports) for every row. |
| `cli.py` | Argparse entry point; `--compare-uv`/`--no-compare-uv` toggles the UV comparison (on by default except for `fgfr`/`predictor-mv`, where no UV source is available/bridgeable yet). |

## UV data sources by gene-set

| Gene-set | UV source | Format | Status |
|---|---|---|---|
| `card11` | `CARD11_UV_CALIB_DIR/card11_calibrations.csv` | range-based CSV (lof/gof x clinvar/cadins/benta) | Verified: 158/158 P/LP+B/LB eval variants bridged. |
| `labelseq` | `LABELSEQ_UV_CALIB_DIR/<gene>_<assay>_<treatment>/` | one `standard_points` CSV per assay, bridged via hgvs_c | Pre-existing, unchanged logic (just relocated/parameterized). |
| `tp53` | `EXC_PP_CLINVAR2025_UV_CALIB/<Dataset>/` | one `standard_points` CSV per dataset, bridged via hgvs_c (from `TP53_ANNOTATED_VARIANTS_PATH`'s `clinvar_HGVS_name` column) onto TP53's own protein-short-form "Variant" ids | Verified: ~200/405 P/LP+B/LB eval variants bridged (real coverage gap -- not every clinical variant was measured by these assays / has a ClinVar entry to source `clinvar_HGVS_name` from, not a bridging bug). |
| `combined` (BRCA1, BRCA2, MSH2, PTEN, ASPA, and the other "integrated" genes) | same `EXC_PP_CLINVAR2025_UV_CALIB` source, bridged via `INTEGRATED_VARIANT_EFFECT_DATASET_PATH`'s own hgvs_c/genomic columns | same shape as `tp53`'s adapter | **Not yet verified per-gene** -- only exercised against TP53-shaped data during development. Sanity-check match rate before trusting numbers for a gene not already checked. |
| `predictor-mv` | none wired up | -- | **Not implemented.** Sampled `predictor_calibration_output` `variants.csv` files use positional `variant_N` ids, not `protein_variant`/hgvs -- bridging would require confirming the UV calibration script processed the exact same filtered per-predictor dataframe/row order as `multivariate_data/predictors.py::load_predictor_data`, which hasn't been confirmed. Returns `None` (comparison unavailable) rather than risk a silently-misaligned join. |
| `fgfr` | none | -- | UV calibrations are pending; always returns `None`. |

`flatten_point_ranges` (mentioned during planning) has zero live call sites
anywhere in the repo -- only dead code in stale `.ipynb_checkpoints` files --
so nothing needed removing from the active pipeline.
