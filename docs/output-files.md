← [Back to README](../README.md)

# Output Files

For each run, `run_pipeline.py` and `run_igvf_batch.py` produce:

| File | Description |
|------|-------------|
| `<name>_<Kc>_calibration.json` | Calibration thresholds, prior, point ranges, fit metadata |
| `<name>_<Kc>_visualization.png` | Score distribution plot with calibrated thresholds |
| `<name>_<Kc>_variants.csv` | Per-variant point assignment table |
| `<name>_<Kc>_lr_values.json.gz` | Full LR+ curves over score range |
| `<name>_model_selection.json` | 2c vs. 3c bootstrap test results |
| `<name>_bootstrap_fits.json.gz` | Saved bootstrap fit results (when fitting fresh) |

`calibration.json` (abbreviated; full field list in `src/assay_calibration/pipeline/utils.py:save_results`):
```json
{
  "dataset": "MSH2_Jia_2021",
  "n_c": "3c",
  "benign_method": "benign",
  "prior": 0.0223,
  "prior_unstable": 0,
  "pct_bootstraps_kept": 1.0,
  "point_ranges": {
    "1": [[0.509, 0.690]],
    "2": [[0.690, 0.865]],
    "-1": [[-2.381, -1.079]],
    "-2": [[-3.516, -2.381]]
  },
  "scoreset_flipped": 1,
  "liberal_monotonicity": 1,
  "pathomechanism_prior": null,
  "pathomechanism_method": null,
  "uncalibratable_reason": null,
  "model_selected": null
}
```
- `prior` — the estimated prior probability of pathogenicity (pi), used to derive the ACMG evidence-tier thresholds.
- `point_ranges` — score intervals mapped to signed ACMG point values (positive = pathogenic-direction, negative = benign-direction; `point_ranges["0"]` covers indeterminate/no-evidence).
- `pathomechanism_prior` / `pathomechanism_method` — only populated when `--pathomechanism-prior` is used (see [Pathomechanism prior](configuration.md#pathomechanism-prior-advanced)); `null` otherwise.
- `model_selected` — which component count (`2c`/`3c`) was preferred by model selection, only set when both were fit and compared.
