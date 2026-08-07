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
- `gnomAD`/population is **always required**, in every mode. Beyond that, requires at least Pathogenic OR (Benign or Synonymous) — see [PN/PU/NU modes](#pnpunu-modes-missing-class-inference) below for what changes when only one of Pathogenic/Benign is available.

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

### PN/PU/NU modes (missing-class inference)

Examples: `example/brca_findlay_PU_example.csv`, `example/brca_findlay_NU_example.csv`.

Which mode a dataset runs in is determined automatically by which control samples survive filtering — gnomAD/population is required in every mode; only Pathogenic/LP and Benign/LB (or Synonymous) are optional (at least one of the two is required):

- **PN** (standard): both Pathogenic/LP and Benign/LB (or Synonymous) present. Prior and evidence come from a joint EM fit between the two labeled classes directly.
- **PU** (positive-unlabeled): only Pathogenic/LP present, no Benign/LB or Synonymous. The real pathogenic density is used as-is; the missing benign-direction density is instead *reconstructed* by unmixing gnomAD/population against it (see below) — gnomAD isn't compared against directly, it's the raw material used to recover an artificial benign density.
- **NU** (negative-unlabeled): only Benign/LB (or Synonymous) present, no Pathogenic/LP. Mirrors PU: the real benign/synonymous density is used as-is; the missing pathogenic-direction density is reconstructed by unmixing gnomAD against it.

**Unmixing.** Both directions rely on the same idea (`compute_single_fit_log_densities`, `fit_utils/point_ranges.py`): gnomAD/population is modeled as a `prior`-weighted mixture of the pathogenic and benign densities, `f_pop = prior * f_pathogenic + (1 - prior) * f_benign`. Whichever side is actually labeled is plugged in directly; the missing side is solved for algebraically from that equation (e.g. in NU mode, `f_pathogenic = (f_pop - (1 - prior) * f_benign) / prior`) — an artificial density standing in for the missing class, built from gnomAD plus the real labeled density, not from gnomAD alone.

**NU evidence direction.** Evidence (LR+) is still computed as usual — real benign/synonymous density vs. the reconstructed artificial pathogenic density — so functional-abnormality evidence in NU mode is measured *relative to the Benign/LB or Synonymous sample*, not relative to gnomAD itself; gnomAD only supplies the population mixture the artificial pathogenic density is unmixed out of. Because that artificial density was never directly observed in a labeled pathogenic sample, pathogenic-direction evidence in NU mode should be read as *functional abnormality*, not as evidence of clinical pathogenicity — nothing in NU mode establishes that assay abnormality in that direction is disease-relevant; that link has to come from outside knowledge of the assay/gene, not from the calibration itself.

**Non-overlap assumption.** The scalar prior that both the Bayes-factor thresholds and the unmixing formula above depend on is itself estimated with a mixture-proportion ("unmixing") boundary estimator (Blanchard, Lee & Scott 2010; Scott 2015 — `estimate_prior_from_class_densities` in `fit_utils/point_ranges.py`): `f_pop(x) = alpha * f_labeled(x) + (1 - alpha) * f_missing(x)`, and the tightest recoverable `alpha` is `inf_x [f_pop(x) / f_labeled(x)]`. This only recovers the *true* prior — rather than a biased underestimate — if some region of score space has (near-)zero density under the missing class, i.e. the missing class's distribution doesn't fully overlap with the labeled class's. A biased prior here doesn't just skew the Bayes-factor thresholds — it's plugged directly into the unmixing formula above, so it also distorts the reconstructed artificial density itself.

### 3. MaveDB format

MaveDB-style CSV with functional classification columns; used via `hpc/prepare.py mavedb`.

Implementation: `src/assay_calibration/data_utils/dataset.py` — `BasicScoreset`, `Scoreset`, `MultiScoreset`, `BasicMultiScoreset`.

> Multivariate calibration (`MultiScoreset`/`BasicMultiScoreset`, jointly fitting several assays per gene — `hpc/prepare.py multivariate`) and calibration of computational predictor scores like REVEL/AlphaMissense (`hpc/prepare.py predictor-mv`, `predictor_mv_utils.py`) are also supported but not yet documented here. Ask in the repo/issues if you want to use either.
