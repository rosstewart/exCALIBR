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

## `gene_performance_scatter.py`: the 3-panel MCC comparison figure

This is the module that produces the actual gene-level MV-vs-UV MCC scatter
figure and the TP53 RPV penetrance-score figure (the two artifacts backing
the preliminary-results abstract), as opposed to `report.py`/`cli.py`'s
single-gene ad hoc table output. It builds one `MultiScoreset` per gene
across four gene-sets, runs `mv_analysis.report.build_comparison_table` for
each, and plots MV MCC vs. UV MCC per gene in three panels:

- **Panel A** (functional-only): TP53 + CARD11 + 17 LABEL-seq genes + 15
  plain-"integrated" genes (`PLAIN_INTEGRATED_GENES`), 34 total.
- **Panel B** (computational predictors only): the 8 `DEFAULT_GENES` from
  `multivariate_data/predictors.py`.
- **Panel C** (combined functional+predictor evidence): same 8 genes as
  Panel B.
- **RPV panel** (`plot_rpv_penetrance_panel`, standalone): TP53's
  reduced-penetrance-variant penetrance-score distributions, built fresh
  (not cached) each run.

### Reproducing the figures

```bash
python3 mv_analysis/gene_performance_scatter.py \
  --results-json /data/ross/assay_calibration/multivariate/jobs_all_1000b_8f_v2/bootstrap_results.json.gz \
  --save-path mv_analysis/figures/gene_performance_scatter.png \
  --rpv-save-path mv_analysis/figures/tp53_rpv_penetrance.png \
  --cache-dir mv_analysis/figures/.gene_cache
```

A from-empty-cache run takes on the order of an hour+ (34 genes x 4 configs
x 1000 bootstraps for Panel A alone, plus a ~10-minute one-time LABEL-seq
`MultiScoreset` rebuild). Re-running the same command after a code fix or an
interruption only recomputes genes whose `.gene_cache/{panel}_{gene}.json`
entry isn't `status: "ok"` -- delete a gene's cache file (or the whole
`.gene_cache/` dir) to force it to recompute.

### Paths relied on

Beyond `--results-json` (the MV bootstrap fits, produced upstream by
`hpc/aggregate_results.py`), everything else comes from `config.py`'s
env-overridable `MV_*` constants (see the table above) plus:

| Path | Role |
|---|---|
| `/data/ross/assay_calibration/dataframe/integrated_variant_effect_dataset_pp_final.tsv.gz` (`combined.DEFAULT_INTEGRATED_DATAFRAME`) | Source `ms` + UV-bridging dataframe for the 15 plain-integrated genes |
| `/data/ross/assay_calibration/labelseq/labelseq-annotated-20260529.flat.tsv.gz` (`labelseq.DEFAULT_DATA_PATH`) | Raw LABEL-seq dataframe |
| `mv_analysis/figures/.gene_cache/{A,B,C}_{gene}.json` | Per-gene result cache (written) |
| `mv_analysis/figures/gene_performance_scatter.png`, `mv_analysis/figures/tp53_rpv_penetrance.png` | Output figures (written) |

FGFR is **not in any of the three panels yet**. Its UV calibration is done
(populated at `/data/ross/assay_calibration/FGFR/uv_calib/{pemr,futr,
activation}/`, using the updated log-scores), but the MV side still needs to
be re-fit against those same updated log-scores -- the only FGFR MV fit
currently in `jobs_all_1000b_8f_v2` (`FGFR_combined_fgfr_mv`) predates that
change and is stale. Once the MV refit lands, wiring FGFR in requires:
`uv_sources.load_fgfr_uv_points` (currently unimplemented -- `load_uv_points`
returns `None` for `gene_set == "fgfr"`), plus adding FGFR to Panel A's
gene-set loop (it isn't in `PLAIN_INTEGRATED_GENES` since it's pooled across
FGFR1-4 rather than a single-gene dataframe row, so it needs its own loader
call similar to TP53/CARD11's dedicated builders, not the plain-integrated
path).

### Dimension selection for plain-integrated genes

Some genes' current dataframe rows include more assay datasets than their
stored bootstrap fit was actually trained on (fits can predate a later data
addition). Building `ms` from every currently-available dataset then
produces more dimensions than the fit's parameter arrays have rows for,
crashing with e.g. `IndexError: index 7 is out of bounds for axis 0 with
size 4` (confirmed for BRCA2: 8 current datasets vs. a 4-dim stored fit).
`_select_connected_datasets` reproduces `hpc/prepare.py`'s job-generation
logic (`Fit._select_calibration_dims`: pairwise-overlap>=30 graph, keep the
largest multi-dim connected component) to reduce to the historically-fitted
subset before building `ms`. Confirmed correct for:

- **BRCA2**: 8 datasets -> 4. The 4 `Sahu_2023_exon13_*` variants (same
  underlying assay, 4 readouts) all pairwise-overlap at 110 shared variants
  and form the largest component; `Hu_2024`/`Huang_2025_SGE`/`Sahu_2025_SGE`
  (a separate 3-node cluster, overlaps 252-2543) and `IGVF` (0 overlap with
  everything) are dropped.
- **KCNH2**: 3 datasets -> 2. `Jiang_2022`<->`O_Neill_2024_surface_expression`
  overlap=43 (clears 30); `Kozek_Glazer_2020`'s overlap with either is only
  1 and 18 (both below 30), so it's dropped as an isolated node.

### Known data-quality caveats (not bugs)

- **ASPA, CBS**: both MCC=0.0 for MV *and* UV, on every config/threshold
  tested. Root cause verified two ways: (1) their exc_pp calibration.json
  `point_ranges` have every negative-direction range (`-1`..`-8`) empty for
  all 4 underlying datasets, so no UV benign call is structurally possible;
  (2) the MV bootstrap fit's per-config point distributions are likewise
  never negative. The *median* per-variant bootstrap LR+ for B/LB does lean
  benign (ASPA 4c_unc: median=-1.65; CBS: median=-0.44), but the pipeline's
  conservative percentile convention (`ben_percentile=95`, i.e. the
  worst-case bootstrap must also call benign) never lets that signal clear a
  point, given how widely these two assays' per-bootstrap LR+ swings (ASPA:
  -8.5 to +500). This is a genuine benign-separation/statistical-power
  limitation of these specific assays (ASPA has only 6 B/LB variants; CBS's
  benign scores sit within noise of its Synonymous/gnomAD controls), not a
  bridging or threshold bug -- excluded from the figure by the
  both-MCC-zero filter.
- **PAX6**: UV MCC=1.0 (56/56 bridged) but MV MCC=0.0 -- the MV fit gives 0%
  correct on Benign+Syn across all 4 configs despite UV bridging now being
  fixed. Not yet root-caused; likely a similar conservative-percentile/
  separation issue to ASPA/CBS above, unconfirmed.
- **6 LABEL-seq genes** (araf, erbb2, grb2, ksr1, ksr2, sos2): MCC=0.0 for
  MV by construction -- these genes have zero P/LP observations at all (see
  `report._eval_labels`'s `p_idx is None` handling), so no pathogenic class
  exists to score against. Genuinely excluded, not a bridging failure.
