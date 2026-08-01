← [Back to README](../README.md)

# Input Data Formats

### 1. BasicScoreset (start here)

`score` column + `sample_assignments` column. No ClinVar metadata required.

```
score,sample_assignments
-0.37,1
-0.05,"1,2"
1.30,2
0.66,"1,2,3"
```

- `score` — per-variant numeric score (direction auto-detected).
- `sample_assignments` — integer (`0`,`1`,`2`,`3`,...) or comma-separated for multi-group rows (e.g. `"1,2"`). Convention: `0`=Pathogenic/LP, `1`=Benign/LB, `2`=gnomAD/population, `3`=Synonymous. Unlabeled/out-of-range rows are treated as VUS (still scored, don't inform the fit). Override with `--sample-names`.
- Requires at least Pathogenic + (Benign or Synonymous).

```bash
python run_pipeline.py \
    --dataset example/brca_findlay_example.csv --name brca_findlay_example \
    --sample-names "Pathogenic/Likely Pathogenic" "Benign/Likely Benign" "gnomAD" "Synonymous"
```

### 2. IGVF / PillarProject format

Per-variant metadata table matching the IGVF Coding Variants Focus Group schema (Tejura et al. 2026, bioRxiv 2026.02.14.705848). Example: `example/MSH2_Jia_2021.csv`; exact columns in `Scoreset`/`Variant` (`src/assay_calibration/data_utils/dataset.py`).

```bash
python run_pipeline.py --dataset example/MSH2_Jia_2021.csv --name MSH2_Jia_2021
```

Sample groups are derived automatically (not assigned by hand) from:

- `Dataset` — dataset name; `--name` selects which one in a multi-dataset file.
- `auth_reported_score` — the functional score.
- `clinvar_sig_2026`/`clinvar_star_2026` (or `_2025`/`_2018`, via `--clinvar-release`) — determines Pathogenic/LP vs. Benign/LB; `--min-clinvar-star` sets the trust threshold (default 1).
- `gnomad_MAF` — non-missing ⇒ gnomAD/population member.
- `simplified_consequence` — identifies synonymous variants; also drives splice filtering.
- `splice_measure` (`Yes`/`No`) — if `No`, splice-consequence and SpliceAI-flagged rows are dropped.
- `Flag` — rows with `Flag == "*"` are dropped.

Requires Benign or Synonymous after filtering. All other columns are optional context for specific downstream features.

### 3. MaveDB format

MaveDB-style CSV with functional classification columns; used via `hpc/prepare.py mavedb`.

Implementation: `src/assay_calibration/data_utils/dataset.py` — `BasicScoreset`, `Scoreset`, `MultiScoreset`, `BasicMultiScoreset`.

> Multivariate calibration (`MultiScoreset`/`BasicMultiScoreset`, jointly fitting several assays per gene — `hpc/prepare.py multivariate`) and calibration of computational predictor scores like REVEL/AlphaMissense (`hpc/prepare.py predictor-mv`, `predictor_mv_utils.py`) are also supported but not yet documented here. Ask in the repo/issues if you want to use either.
