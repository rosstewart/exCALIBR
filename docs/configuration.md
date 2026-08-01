← [Back to README](../README.md)

# Configuration Options

### Bootstrap parameters

`--preset` is the main quality/speed control:

```bash
python run_pipeline.py --dataset my.csv --name MyGene --preset large
```

| Preset | `--n-bootstraps` | `--fits-per-bootstrap` |
|---|---|---|
| light (default) | 20   | 8   |
| medium           | 100  | 8   |
| large            | 500  | 8   |
| xl               | 1000 | 8   |
| finest           | 1000 | 100 |

For manual control, `--n-bootstraps`/`--fits-per-bootstrap` override the preset's value for that one setting:

```bash
python run_pipeline.py --dataset my.csv --name MyGene --n-bootstraps 1000 --fits-per-bootstrap 100
```

Large/xl/finest on many datasets: use the [batch HPC workflow](batch-hpc-workflow.md) instead of running one at a time.

#### Quality vs. speed presets

Tested on 87 real datasets (~85,000 variants), comparing each `--n-bootstraps` level against a ~1000-bootstrap reference run of the same dataset (`--fits-per-bootstrap` fixed at 100 for this test). Per-variant evidence-point-value changes, plus worst-case (90th-percentile) drift in aggregate classification accuracy, MCC, and coverage:

| `--n-bootstraps` | Unchanged | Shifted, same direction | Gained/lost evidence (↔ 0) | Sign flip | Accuracy Δ | MCC Δ | Coverage Δ |
|---|---|---|---|---|---|---|---|
| 20  | 79.1% | 18.0% | 3.0% | 0.0% | 3.3pp | 5.0pp | 9.3pp |
| 50  | 83.4% | 13.3% | 3.4% | 0.0% | 2.2pp | 4.1pp | 4.9pp |
| 100 | 90.0% | 7.7%  | 2.3% | 0.0% | 2.7pp | 4.0pp | 4.2pp |
| 250 | 92.9% | 6.0%  | 1.1% | 0.0% | 0.9pp | 0.4pp | 1.8pp |
| 500 | 95.7% | 3.2%  | 1.0% | 0.0% | 0.7pp | 0.6pp | 2.0pp |

#### Speed estimates

Timed on `example/MSH2_Jia_2021.csv` (1579 variants) at Light, scaled linearly for other presets/core counts:

| Preset | 4 cores | 16 cores | 64 cores |
|---|---|---|---|
| Light  | ~2.5 hours | ~35 min | ~10 min |
| Medium | ~12 hours  | ~3 hours | ~45 min |
| Large  | ~2.5 days  | ~16 hours | ~4 hours |
| XL     | ~5 days    | ~1.3 days | ~8 hours |
| Finest | ~2 months  | ~16 days | ~4 days |

`--n-jobs` sets core count (`-1` = all available). 3c fitting is ~15-20% slower than 2c. Reproduce with `tests/benchmark_run_pipeline_speed.py`.

### Component selection

```bash
python run_pipeline.py --dataset my.csv --name MyGene                  # default: 3c only
python run_pipeline.py --dataset my.csv --name MyGene --components 2 3 # fit both, auto-select
python run_pipeline.py --dataset my.csv --name MyGene --components 5   # 5-component
```

### Prior estimation

The prior is the estimated probability that a variant in this gene is pathogenic; it sets the ACMG evidence-strength thresholds. Estimated from your data by default (EM fit to pathogenic/benign score distributions).

```bash
python run_pipeline.py --dataset my.csv --name MyGene                    # default: EM estimation
python run_pipeline.py --dataset my.csv --name MyGene --no-median-prior  # 5th/95th percentile thresholds instead
python run_pipeline.py --dataset my.csv --name MyGene --manual-prior 0.001  # supply your own, skip estimation
```

`--manual-prior` takes a probability in (0, 1) and is used for every bootstrap fit's thresholds instead of a per-fit estimate. Use it if you have an external prevalence estimate or too few controls for a reliable data-driven estimate.

#### Pathomechanism prior (advanced)

For assays that only detect one of several disease mechanisms (e.g. an assay measuring loss-of-function won't flag a dominant-negative pathogenic variant as abnormal). This dilutes the pathogenic sample with variants the assay can't detect, understating pathogenic-direction evidence strength.

```bash
python run_pipeline.py --dataset my.csv --name MyGene --pathomechanism-prior
```

Estimates the fraction of ClinVar pathogenic variants the assay actually detects, and computes a separate pathogenic-direction prior/evidence pair from just that subset (benign-direction evidence is untouched). PN/PU mode only; mutually exclusive with `--filter-pathogenic-sample-by-lr`. Reported as `PLP_frac_pathomechanism_measured` and `pathomechanism_prior` in the output JSON.

### Benign sample method

```bash
--benign-method avg         # average benign + synonymous (default when both exist)
--benign-method benign      # benign (ClinVar B/LB) only
--benign-method synonymous  # synonymous only
```

### Point-range postprocessing

```bash
python run_pipeline.py --dataset my.csv --name MyGene                          # default: enforce monotonicity + extend to score-axis limits
python run_pipeline.py --dataset my.csv --name MyGene --no-postprocess         # raw, unprocessed LR-threshold intervals (debugging only)
python run_pipeline.py --dataset my.csv --name MyGene --conservative-monotonicity  # stricter monotonicity enforcement
```

### Bidirectional assay auto-detection

Some assays (e.g. LoF/GoF) show pathogenic-leaning evidence on both sides of a benign region, breaking the standard single-direction postprocessing. Auto-detected by default (`n_c >= 3` only) — `--no-postprocess` above is not needed for this case.

- Per bootstrap fit: mixture components are sorted by score and labeled pathogenic-like/benign-like by comparing pathogenic vs. benign sample weights (PU/NU modes fall back to gnomAD). Flagged if a benign-like component has a pathogenic-like component on each side.
- If a majority of fits are flagged: pathogenic tiers are re-nested and extended independently on each side of the benign region. Benign tiers are cleaned up (fragmentation merged) but never extended to the axis limit.
- Otherwise: standard postprocessing runs unchanged.

```bash
python run_pipeline.py --dataset my.csv --name MyGene --components 3                                        # default: on
python run_pipeline.py --dataset my.csv --name MyGene --components 3 --no-auto-bidirectional                # off
python run_pipeline.py --dataset my.csv --name MyGene --components 3 --no-auto-bidirectional --no-postprocess  # off + raw output
```
