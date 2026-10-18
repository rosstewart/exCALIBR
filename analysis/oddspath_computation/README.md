# OddsPath recomputation

`compute_oddspath.py` computes OddsPath (Brnich et al. 2019) for every dataset
in our integrated variant-effect table (`config.DATASET_TSV`). It uses the
collaborators' method, porting
`igvf-scalable-vus-resolution/notebooks/analysis/OddsPath_calculations.ipynb`
(see `README_OddsPath_calculations.md` next to it). Its output is
`config.OP_EVIDENCE_CODES_CSV`, which feeds Fig. 5B, `\NumDatasetsBrnichNA`,
Supplementary Data 3 and the D6 summary.

    python -m analysis.oddspath_computation.compute_oddspath [--tsv PATH] [--out-dir DIR]

Run it again whenever `DATASET_TSV` changes. The build's `REG.require` flags
the CSV as stale if it is older than the TSV.

## Outputs (`output/`)

| File | Contents |
|---|---|
| `OddsPath_calibrations_recomputed.csv` | One row per dataset, with the collaborators' CSV columns |
| `excluded.csv` | Rows computed but dropped by `analysis.discovery.DEFAULT_EXCLUDED_DATASETS` (F9_Popp_2025_model, SFPQ_IGVF, TP53_Fayer_2021_meta) |
| `not_computable.csv` | Datasets with no variant classed functionally normal or abnormal |
| `comparison_to_collaborator.csv` | Every value that differs from the collaborators' CSV (`config.OP_EVIDENCE_CODES_CSV_COLLABORATOR`) |

## Same as the notebook

- CHEK2_Gebbia_2024 `Filter_CI` flags (`*`). Our TSV already carries them.
- Removes `Flag == '*'` variants. Removes splice variants (SpliceAI DS ≥ 0.2 or
  `splice_site_variant`, broadcast across an amino-acid change in amino-acid
  assays), except in datasets curated `splice_measure == 'Yes'`.
- Functional class comes from MaveDB's `auth_reported_func_class_category`.
  Where the curation sheet reports score intervals but no classification, it
  comes from the score intervals instead.
- MSH2_Scott_2022: MSH2_Jia_2021 scores with the Scott et al. 2022
  (PMID 36550560) intervals, Normal (-inf, 0) and Abnormal [0.4, inf).
- Controls are ClinVar 2018 P/LP and B/LB for BRCA1, PTEN, MSH2 and TP53, and
  ClinVar 2025 otherwise. One call per MaveDB variant; conflicts are not
  controls.
- `Odds_LR` (+1 pseudocounts), the evidence-code thresholds, and the
  KCNQ4_Zheng_2022_v12_homozygous Abnormal-code mask.

## Changes from the notebook

- **Which datasets qualify.** A dataset qualifies when at least one of its
  variants is classed functionally normal or abnormal. The notebook used the
  curation sheet's "Reported"/"Yes" flags instead, which leave out
  CHEK2_Gebbia_2024 and F9_Popp_2025_light_chain even though both carry MaveDB
  classes. The 51 qualifying calibrated datasets are exactly the
  author-annotated datasets.
- **CHEK2_Gebbia_2024.** The authors' labels are LLR-based evidence strengths,
  which MaveDB maps to normal (benign supporting or stronger, score ≥ 0.88)
  and abnormal (pathogenic supporting or stronger, score ≤ 0.43).
  Indeterminate and unlabelled variants are not classed.
- **No hard-coded exclusions.** SFPQ_IGVF and the four F9_Popp_2025 datasets
  are computed. Only `DEFAULT_EXCLUDED_DATASETS` is applied, at the end.
- **MSH2.** Only the MSH2_Scott_2022 row is written.
  `supplementary_data.load_oddspath_csv` (`config.OP_SUPERSEDED_BY`) puts it
  under the calibrated dataset's name, MSH2_Jia_2021. The build check
  `oddspath-msh2-once` asserts that this gives one MSH2 row.
- `Odds_LR` tests for zero controls before computing the prior. The notebook
  divided first and returned an error row.

## Differences from the collaborators' CSV

All differences on shared datasets trace to the input, not the method:

- **Nucleotide-level assays** with `splice_measure == 'Yes'`: BAP1_Waters_2024,
  BARD1_IGVF, BRCA1_Findlay_2018, both CARD11_Meitlis_2020_SGE datasets,
  CTCF_IGVF, DDX3X_Radford_2023, PALB2_IGVF, RAD51C_Olvera-León_2024,
  RAD51D_IGVF, TP53_Funk_2025, VHL_Buckley_2024 and XRCC2_IGVF. The
  collaborators' values are reproduced exactly when splice variants are
  dropped from these assays. Their run did not keep splice variants in
  splicing-capable assays; the notebook's filter, followed here, does.
- **TSC2_IGVF**: their values are reproduced exactly from the 2026-08-13 input
  table. TSC2 was updated in `DATASET_TSV` on 2026-09-16.
