← [Back to README](../README.md)

# Input Data Formats

Three formats are supported:

### 1. IGVF / PillarProject format (standard)

A rich per-variant metadata table (one row per variant, or per variant per
transcript effect), matching the schema produced by the IGVF Coding
Variants Focus Group pipeline (Tejura et al. 2026, bioRxiv
2026.02.14.705848). `example/MSH2_Jia_2021.csv` is a complete real
example — see it (or `Scoreset`/`Variant` in
`src/assay_calibration/data_utils/dataset.py`) for the exact column list.

```bash
python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021
```

Unlike the BasicScoreset format below, you don't manually assign each row
to a sample group. Instead, group membership (Pathogenic/Likely
Pathogenic, Benign/Likely Benign, gnomAD, Synonymous) is derived
automatically from ClinVar classification, gnomAD allele frequency, and
variant consequence. The columns that drive this:

- `Dataset` — assay/dataset name. A file can contain many datasets;
  `--name` selects which one to run.
- `auth_reported_score` — the functional assay score for each variant.
- `clinvar_sig_2026` / `clinvar_star_2026` (or the `_2025`/`_2018` suffixed
  equivalents, selected via `--clinvar-release`) — ClinVar clinical
  significance and review-status star rating. Determines which variants
  count as Pathogenic/Likely Pathogenic vs. Benign/Likely Benign.
  `--min-clinvar-star` sets how many stars are required to trust a
  classification (default: 1).
- `gnomad_MAF` — gnomAD population allele frequency. Any variant with a
  non-missing value here is treated as a gnomAD/population sample member.
- `simplified_consequence` — the variant's molecular consequence (e.g.
  `missense_variant`, `synonymous_variant`). Identifies synonymous
  variants and is also used to filter out splice-affecting consequences.
- `splice_measure` — `"Yes"` or `"No"`: whether this particular assay
  itself measures splicing effects. When `"No"`, splice-consequence rows
  and SpliceAI-flagged rows (`spliceAI_DS_*` columns) are automatically
  dropped, since a non-splicing assay's scores for those variants aren't
  meaningful.
- `Flag` — rows with `Flag == "*"` (author-flagged as unreliable) are
  dropped automatically.

At least one of Benign or Synonymous samples must be present after
filtering. Everything else in the schema (gene/transcript/position
metadata, MaveDB interval-classification columns, REVEL/AlphaMissense/
SpliceAI annotations, etc.) is optional context used by specific
downstream features — not required to get a basic calibration running.

### 2. BasicScoreset (bare-bones CSV)

Minimal format: `score` column + `sample_assignments` column. No ClinVar
metadata required — you just tell it which rows belong to which sample
group yourself. `sample_assignments` is an integer per row (`0`, `1`,
`2`, `3`, ...), or a comma-separated string for variants that belong to
more than one group (e.g. `"1,2"`). By convention, column `0` = Pathogenic,
`1` = Benign, `2` = gnomAD/population, `3` = Synonymous — use
`--sample-names` to relabel them if your groups don't match that
convention.

`example/brca_findlay_example.csv` is a ready-to-run BasicScoreset example
(BRCA1 SGE functional scores from Findlay et al. 2018, with rows labeled
by ClinVar/population group membership):

```bash
python run_pipeline.py \
    --dataset example/brca_findlay_example.csv --name brca_findlay_example \
    --sample-names "Pathogenic/Likely Pathogenic" "Benign/Likely Benign" "gnomAD" "Synonymous"
```

### 3. MaveDB format

MaveDB-style CSV with functional classification columns. Used for batch runs via `slurm/prepare.py mavedb`.

Implementation: `src/assay_calibration/data_utils/dataset.py` — `BasicScoreset`, `Scoreset`, `MultiScoreset`, `BasicMultiScoreset`.

> **Multivariate calibration is also supported but not yet documented
> here.** `MultiScoreset`/`BasicMultiScoreset` let you calibrate several
> assays for the same gene jointly (multi-dimensional fitting, rather than
> one score axis at a time) — see the `multivariate` subcommand of
> `slurm/prepare.py`.
>
> **Calibrating computational variant-effect predictor scores (e.g.
> REVEL, AlphaMissense) is also supported but not yet documented here.**
> Per-gene predictor-score CSVs are loaded and calibrated the same way as
> functional assay data (as a `BasicMultiScoreset`, one dimension per
> predictor) — see the `predictor-mv` subcommand of `slurm/prepare.py` and
> `predictor_mv_utils.py`.
>
> Ask in the repo/issues if you want to use either of these before they're
> written up properly.
