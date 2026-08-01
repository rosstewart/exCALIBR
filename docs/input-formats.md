← [Back to README](../README.md)

# Input Data Formats

Three formats are supported. **Start with BasicScoreset** if you're
bringing your own assay data — it's the fastest way to get from "I have a
CSV of scores" to a running calibration.

### 1. BasicScoreset (start here)

Minimal format: `score` column + `sample_assignments` column. No ClinVar
metadata required — you just tell it which rows belong to which sample
group yourself.

```
score,sample_assignments
-0.37,1
-0.05,"1,2"
1.30,2
0.66,"1,2,3"
```

- `score` — your assay's per-variant numeric score. Any real number; higher/lower doesn't need to mean pathogenic/benign in any particular direction (the pipeline auto-detects which direction is which).
- `sample_assignments` — an integer per row (`0`, `1`, `2`, `3`, ...) marking which control group(s) that variant belongs to, or a comma-separated string for a variant that belongs to more than one group (e.g. `"1,2"`). By convention:
  - `0` = Pathogenic/Likely Pathogenic (known disease-causing variants)
  - `1` = Benign/Likely Benign (known benign variants)
  - `2` = gnomAD/population (variants observed in the general population, presumed mostly benign)
  - `3` = Synonymous (silent/synonymous-coding variants, a common benign proxy)
  - Rows with no value (or a value outside 0-3, if you're not using extra `--sample-names`) are treated as unlabeled variants-of-uncertain-significance — they still get scored/classified in the output, they just don't inform the fit itself.
- At least one of Pathogenic + (Benign or Synonymous) must be present for the calibration to run at all.
- Use `--sample-names` to relabel the convention above if your columns are in a different order, or to add more than 4 groups.

`example/brca_findlay_example.csv` is a ready-to-run BasicScoreset example
(BRCA1 SGE functional scores from Findlay et al. 2018, with rows labeled
by ClinVar/population group membership):

```bash
python run_pipeline.py \
    --dataset example/brca_findlay_example.csv --name brca_findlay_example \
    --sample-names "Pathogenic/Likely Pathogenic" "Benign/Likely Benign" "gnomAD" "Synonymous"
```

### 2. IGVF / PillarProject format

A rich per-variant metadata table (one row per variant, or per variant per
transcript effect), matching the schema produced by the IGVF Coding
Variants Focus Group pipeline (Tejura et al. 2026, bioRxiv
2026.02.14.705848). `example/MSH2_Jia_2021.csv` is a complete real
example — see it (or `Scoreset`/`Variant` in
`src/assay_calibration/data_utils/dataset.py`) for the exact column list.

```bash
python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021
```

Unlike BasicScoreset above, you don't manually assign each row
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
