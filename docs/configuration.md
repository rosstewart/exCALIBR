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

Tested on 87 real datasets (~85,000 variants pooled) against each dataset's own ~1000-bootstrap reference run (`--fits-per-bootstrap` fixed at 100). Evidence-strength bands: Supporting = 1 pt, Moderate = 2–3 pt, Strong = 4–7 pt, Very Strong = 8 pt.

Per-variant point-value changes vs. reference (% of all variants):

| `--n-bootstraps` | Unchanged | Same band | Diff. band, same direction | 0 → pathogenic | 0 → benign | Pathogenic → 0 | Benign → 0 | Sign flip |
|---|---|---|---|---|---|---|---|---|
| 20  | 79.1% | 10.3% | 7.7% | 0.6% | 1.9% | 0.1% | 0.3% | 0.0% |
| 50  | 83.4% | 7.9%  | 5.4% | 0.2% | 1.6% | 0.1% | 1.5% | 0.0% |
| 100 | 90.0% | 4.2%  | 3.5% | 0.1% | 1.2% | 0.1% | 0.9% | 0.0% |
| 250 | 92.9% | 2.8%  | 3.2% | 0.1% | 0.2% | 0.0% | 0.8% | 0.0% |
| 500 | 95.7% | 1.6%  | 1.6% | 0.0% | 0.1% | 0.0% | 0.9% | 0.0% |

Sign flip (pathogenic ↔ benign) is 0.0% at every level tested.

Aggregate-metric drift across datasets, median (IQR: 25th–75th percentile), absolute Δ:

| `--n-bootstraps` | Accuracy Δ | MCC Δ | Coverage Δ |
|---|---|---|---|
| 20  | 0.06pp (0–0.73) | 0.04pp (0–0.91) | 0.84pp (0–2.52) |
| 50  | 0.01pp (0–0.23) | 0.01pp (0–0.39) | 0.28pp (0–1.68) |
| 100 | 0.00pp (0–0.23) | 0.00pp (0–0.28) | 0.17pp (0–1.68) |
| 250 | 0.00pp (0–0.04) | 0.00pp (0–0.02) | 0.00pp (0–0.35) |
| 500 | 0.00pp (0–0.00) | 0.00pp (0–0.00) | 0.00pp (0–0.19) |

#### Speed estimates

Timed on `example/MSH2_Jia_2021.csv` (1,579 variants) at `--preset light`, scaled linearly for other presets/core counts:

| Preset | 4 cores | 16 cores | 64 cores | V100S GPU |
|---|---|---|---|---|
| Light  | ~2.5 hours | ~35 min | ~10 min | ~5.6 min |
| Medium | ~12 hours  | ~3 hours | ~45 min | ~28 min |
| Large  | ~2.5 days  | ~16 hours | ~4 hours | ~2.3 hours |
| XL     | ~5 days    | ~1.3 days | ~8 hours | ~4.6 hours |
| Finest | ~2 months  | ~16 days  | ~4 days  | ~2.4 days |

`--n-jobs` sets core count (`-1` = all available). 3c fitting is ~15-20% slower than 2c. GPU column is for `--device cuda:N` with a Tesla V100S-PCIE-32GB and includes JAX JIT compilation (the first dataset in a process always pays this cost; ~33s overhead). See [GPU Acceleration](gpu-acceleration.md) for details and hardware context. Reproduce timing with `tests/benchmark_run_pipeline_speed.py`.

#### Fits per bootstrap

`--fits-per-bootstrap` controls how many random EM restarts each fit tries. Every preset except `finest` defaults to 8: a 3-component mixture has `2³ = 8` skew-sign combinations, enumerated (not randomly searched) across restarts — 8 is the minimum that covers all of them.

Median (IQR) degradation vs. `--fits-per-bootstrap`, tested on 91 real datasets:

| `--fits-per-bootstrap` | Δ log-likelihood (SDs from best) | % of best likelihood achieved* |
|---|---|---|
| 1   | -0.59 (-1.29 to -0.39) | 92.7% (83.4–96.3%) |
| 8   | -0.05 (-0.17 to -0.02) | 99.4% (98.4–99.8%) |
| 20  | -0.01 (-0.05 to -0.00) | 99.8% (99.2–99.9%) |
| 50  | -0.00 (-0.02 to -0.00) | 100.0% (99.7–100.0%) |
| 100 | 0.00 (0.00 to 0.00) | 100.0% (100.0–100.0%) |

Skew-sign changes between init and converged fit: 0/117 (0%).

*Geometric-mean likelihood ratio vs. best-of-100, not an accuracy score. Reproduce with `tests/benchmark_num_fits_dataframe.py` + `tests/plot_fit_number_comparison.py`.

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

If your dataset is missing Pathogenic or Benign/Synonymous controls entirely, prior estimation switches to a different mode (PU/NU) with its own assumptions — see [PN/PU/NU modes](input-formats.md#pnpunu-modes-missing-class-inference).

#### Pathomechanism prior (advanced)

For assays that only detect one of several disease mechanisms (e.g. an assay measuring loss-of-function won't flag a dominant-negative pathogenic variant as abnormal). This dilutes the pathogenic sample with variants the assay can't detect, understating pathogenic-direction evidence strength.

```bash
python run_pipeline.py --dataset my.csv --name MyGene --pathomechanism-prior
```

Estimates the fraction of ClinVar pathogenic variants the assay actually detects (`P(M=1|Y=1)`), by default via the same closed-form boundary/mixture-proportion estimator (Blanchard, Lee & Scott 2010; Scott 2015) used for the [PN/PU/NU prior](input-formats.md#pnpunu-modes-missing-class-inference), and computes a separate pathogenic-direction prior (`P(Y=1,M=1)`) and evidence pair from just that subset (benign-direction evidence, `P(Y=1)`, is untouched). [PN/PU](input-formats.md#pnpunu-modes-missing-class-inference) mode only; mutually exclusive with `--filter-pathogenic-sample-by-lr`. Reported as `PLP_frac_pathomechanism_measured` and `pathomechanism_prior` in the output JSON.

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
